"""
GraphManager -- central orchestrator for the MADDENING simulation graph.

Owns all node state, builds the execution schedule, and JIT-compiles the
full graph step into a single XLA computation via ``jax.jit``.

Supports multi-rate timesteps: each node declares its own ``delta_t``,
and the graph manager derives a *base timestep* (GCD of all node
timesteps).  The compiled step advances at the base rate; each node
updates only when its own sub-step counter fires.  For JAX traceability
the update is always computed but conditionally applied via
``jnp.where``.
"""

from __future__ import annotations

import functools
import logging
import math
import warnings
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import jax
# `jax.core` is not re-exported by `jax/__init__.py`, so the attribute
# only resolves for a type checker when the submodule is imported by name.
import jax.core
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # `save_state` / `load_state` annotate with `Path`; the runtime
    # import stays inside the method so importing this module does
    # not pay for it.
    from pathlib import Path
    from typing import TextIO

    # The inspection methods return it; the module is imported lazily
    # inside them (see "Read-only inspection" at the end of the class).
    from maddening.core.inspection import InspectionTable

from maddening.core._quiet_warnings import quiet_warnings
from maddening.core.coupling import CouplingGroup, coupling_group_kwargs
from maddening.core.coupling.acceleration import (
    convergence_criterion,
    float_fields_of,
    reported_converged,
    reported_error_estimate,
    residual_precision_floor,
    spectral_error_bound,
    spectral_rate_settled,
)
from maddening.core.edge import EdgeSpec
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import (
    SimulationNode, _detached_config, _method_accepts_params, _mutable_snapshot,
    _mutated_keys, _refuse_params_no_keyword_reaches, _replaced_keys,
)
from maddening.core.params import (
    ParamSpec,
    check_bounds as _check_bounds,
    constrain as _constrain,
    trainable_mask as _trainable_mask,
    unconstrain as _unconstrain,
)
from maddening.core.compliance.stability import stability
from maddening.core.schedule import (
    detect_cycles,
    find_strongly_connected_components,
    identify_back_edges,
    topological_sort,
)

# The module-level machinery this class runs lives in private modules and
# is read through them (``_graph_specs._META_KEY``), never imported by name:
# a private helper is not an attribute of this module, so a stale
# ``graph_manager._helper`` reference fails instead of naming a copy that
# nothing reads.
from maddening.core import _adaptive_scan, _graph_specs, _param_probes
from maddening.core.coupling import (
    _bounds, _coupled_block, _group_layout, _interface_plan, _reports,
)
# Public names defined with the machinery that uses them stay importable from here.
from maddening.core.coupling._bounds import GRADIENT_PROBE_ENTRY_LIMIT as GRADIENT_PROBE_ENTRY_LIMIT


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class ShardingIssue:
    """A single issue found by :meth:`GraphManager.validate_sharding`.

    ``severity`` is ``"error"`` (raise-worthy) or ``"warning"``
    (advisory).  ``code`` is a short slug callers can switch on.
    """
    severity: str   # "error" | "warning"
    code: str       # short stable slug, e.g. "sharded_node_mesh_axes_mismatch"
    message: str    # human-readable explanation
    affected_nodes: list[str]


@dataclass(frozen=True)
class ExternalInputSpec:
    """Declares an external input that flows into a node's boundary_inputs.

    External inputs come from outside the graph (controllers, sensors,
    user commands) rather than from other nodes via edges.
    """
    target_node: str
    target_field: str
    shape: tuple
    dtype: Any = jnp.float32

    def to_dict(self) -> dict:
        """Serialise for :meth:`GraphManager.to_dict`.

        ``dtype`` is written by name (``"int32"``, ``"float32"``).  It
        used to be left out, so an ``int32`` input reloaded as
        ``float32`` — a node using it as an index then failed on the
        reloaded graph, and one doing arithmetic with it got a different
        trace.  Every field of this dataclass has a slot here, and
        ``tests/core/test_external_input_serialisation.py`` asserts that from
        ``dataclasses.fields`` so the next field added cannot be dropped
        silently.
        """
        return {
            "target_node": self.target_node,
            "target_field": self.target_field,
            "shape": list(self.shape),
            "dtype": jnp.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "ExternalInputSpec":
        """Rebuild from :meth:`to_dict`.

        ``dtype`` is optional: a config written before it was recorded
        reloads at ``float32``, which is what such a graph got then.
        """
        return cls(
            target_node=config["target_node"],
            target_field=config["target_field"],
            shape=tuple(config.get("shape", ())),
            dtype=(
                jnp.dtype(config["dtype"]) if config.get("dtype") is not None
                else jnp.float32
            ),
        )


# ------------------------------------------------------------------
# Observer event names
# ------------------------------------------------------------------
EVENT_NODE_ADDED = "node_added"
EVENT_NODE_REMOVED = "node_removed"
EVENT_EDGE_ADDED = "edge_added"
EVENT_EDGE_REMOVED = "edge_removed"
EVENT_COMPILED = "compiled"
EVENT_STEP = "step"
# Emitted by maddening.sysid.fit / fit_lm / fit_multiple_shooting.
EVENT_FIT_PROGRESS = "fit_progress"


class _BakedParamWrite(ValueError):
    """The refusal of :meth:`GraphManager._refuse_baked_param_writes`, with
    what it found as attributes -- the node, the leaf, the value the tree
    holds, the node's own (``None`` when it has none) and why the node
    cannot read the leaf -- for a caller that words the refusal for its own
    interface (``POST /checkpoint/load`` names routes, not methods)."""

    def __init__(self, message: str, *, owner: str, key: str, value: Any,
                 own: Any, reason: str) -> None:
        super().__init__(message)
        self.owner = owner
        self.key = key
        self.value = value
        self.own = own
        self.reason = reason


@stability(StabilityLevel.STABLE)
class GraphManager:
    """Build, validate, compile and run a simulation graph.

    Supports multi-rate scheduling: nodes may have different timesteps.
    The graph steps at the *base timestep* (GCD of all node timesteps).
    Each node updates only on the sub-steps that are multiples of its
    own rate divider.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _graph_specs._NodeSpec] = {}
        self._edges: list[EdgeSpec] = []
        self._state: dict[str, dict] = {}
        self._schedule: list[str] = []
        self._compiled_step: Optional[Callable] = None
        self._dirty: bool = True
        # Bumped by every ``compile()``.  The scan cache keys on it, so a
        # cached scan is exactly as fresh as ``_compiled_step``: every
        # graph mutation marks the graph dirty, every public entry point
        # recompiles a dirty graph, and the recompile invalidates the
        # cache.
        self._compile_generation: int = 0
        # ``jax.lax.scan`` programs built by ``run_scan`` and its
        # siblings, keyed by ``_cached_scan``.
        self._scan_cache: dict[tuple, Callable] = {}
        self._n_scan_traces: int = 0
        self._observers: list[Callable] = []
        self._back_edges: list[EdgeSpec] = []
        self._external_inputs: list[ExternalInputSpec] = []
        self._is_multirate: bool = False
        # v0.2 #3: snapshot of per-node static_data hashes captured at
        # ``compile()`` time.  Used by ``_check_static_data_dirty`` so a
        # node whose ``static_data`` changes after compile (typical
        # case: ``replace_node`` brings a different mesh) forces a
        # recompile on the next ``step()``.
        self._static_data_hashes: dict[str, int] = {}
        self._rate_dividers: dict[str, int] = {}
        self._coupling_groups: list[CouplingGroup] = []
        # Multi-GPU state (set by enable_multigpu)
        self._multigpu_mesh = None
        self._multigpu_device_map: Optional[dict[str, int]] = None
        # Differentiable graph parameters — the third pytree of the
        # compiled step, next to state and external inputs.  The compiled
        # step reads it on every call: edit it in place (or pass
        # ``params=`` to step/run_scan) to change constants without a
        # recompile, and differentiate with respect to it for
        # calibration / system identification.  Read through the
        # ``params`` property, which first takes in any ``node.params``
        # write (``_sync_node_param_writes``).
        self._params: dict = {"nodes": {}, "mappings": {}}
        # What the last sync saw of each node: its params mapping, its write
        # counts and a copy of its in-place-mutable values.  A ``node.params``
        # write is what moved them since.
        self._node_writes_seen: dict[str, tuple[Any, dict, dict]] = {}
        # Graph-level ParamSpec overrides: {node: {key: ParamSpec}}.
        self._param_spec_overrides: dict[str, dict[str, ParamSpec]] = {}
        # The raw (uncounted, unjitted) step of the last compile, for the
        # one trace ``_params_read_by_step`` takes; leaves of ``params``
        # already checked by ``_refuse_baked_param_writes``, by identity;
        # and that trace's answer, keyed by compile generation.
        self._raw_step_fn: Optional[Callable] = None
        self._params_verified: dict[str, dict[str, Any]] = {}
        self._step_reads: Optional[tuple[int, Optional[set]]] = None
        # Per node: the keys its own hooks read with every declared
        # boundary input supplied, keyed by compile generation and by the
        # node object (a node replaced under the same name is asked again).
        self._node_reads: dict[str, tuple[int, Any, set]] = {}
        # The underflow-range check (``_warn_underflow_range``): pending
        # until the first untraced step after each compile, and the groups
        # already warned about, which are never warned about again.
        self._underflow_check_pending = False
        self._underflow_warned: set[str] = set()
        # The state-layout check of a stepped state
        # (``_store_stepped_state``): which trace of which compile the
        # last compared step was (``None`` until one has been compared),
        # so a new compile and a retrace each bring the next comparison.
        self._layout_checked_trace: Optional[tuple[int, int]] = None
        # Escaped-tracer bookkeeping; see ``_recover_from_escaped_tracers``.
        self._state_traced = False
        self._state_before_trace: Optional[dict] = None
        # What the last step left, kept from the first write made to the
        # state after it; see ``_keep_state_for_reports``.
        self._state_as_reported: Optional[tuple[dict, dict]] = None
        # Groups whose report was loaded from a checkpoint saved after
        # their state was written; see ``_note_loaded_after_a_write``.
        self._reports_loaded_after_a_write: dict[str, tuple] = {}
        # The rate dividers of the step that is actually compiled, as
        # opposed to ``_rate_dividers``, which ``compile`` overwrites on
        # its way through and leaves behind if it raises.  This is what
        # the sub-step phase in ``_meta`` is indexed by.
        self._committed_rate_dividers: dict[str, int] = {}
        # The coupling groups the compiled step was built from, by group
        # key.  ``coupling_diagnostics()`` judges the report slots under
        # the group that wrote them, which until the next compile is this
        # one and not a replacement registered since.
        self._committed_coupling_groups: dict[str, CouplingGroup] = {}
        self._committed_floor_inputs: dict[str, tuple] = {}
        # Per group key, the keys of the geometry-dependent mapped edges
        # the compiled step's pass resolves (experimental; empty for a
        # group without one).  Such a group reports no bound.
        self._committed_geometry_edges: dict[str, tuple] = {}
        self._committed_geometry_refusals: dict[str, Optional[str]] = {}
        # Per group key, the floating constants the gradient bound probed
        # as a whole rather than entry by entry (``_probe_plan``), as
        # ``(name, entries)``; written when the step is traced.
        self._gradient_whole_probes: dict[str, tuple] = {}
        # MADD-ANO-068, per compile generation: ``[generation, candidates,
        # hazards]``; see ``_refuse_xla_loop_hazards``.
        self._xla_loop_hazards: Optional[list] = None

    def _snapshot_params(self) -> dict:
        return {
            "nodes": {
                name: spec.node.params_pytree()
                for name, spec in self._nodes.items()
                if spec.accepts_params
            },
            # A copy of each table: ``gm.params`` is edited in place, and a
            # mapping may hand out the dict it keeps.  Stored as it came,
            # an edit of the live weights rewrote the mapping's own --
            # ``reset_params()`` then restored the edit, and ``to_dict``
            # compared the live weights with themselves and never warned.
            "mappings": {
                edge.key: dict(edge.mapping.params_pytree())
                for edge in self._edges
                if edge.mapping is not None
            },
        }

    @property
    def params(self) -> dict:
        """The graph parameter pytree, ``{"nodes": {name: {key: leaf}},
        "mappings": {edge: {key: leaf}}}``, which the compiled step reads on
        every call.

        Reading it first takes in every ``node.params`` write the graph has
        not seen (:meth:`_sync_node_param_writes`), so whatever reads the
        parameters -- your code, ``save_state``, ``to_dict``, the FMU export,
        ``GET /graph/params``, a ``jax.jit`` of a sysid loss handed
        ``gm.params`` -- reads the values the next step will run.  Assigning
        it replaces the tree; editing a leaf in place is a write the next
        step uses.
        """
        self._sync_node_param_writes()
        return self._params

    @params.setter
    def params(self, value: dict) -> None:
        # A node write made before this assignment is older than it: taken
        # in first, then replaced.
        self._sync_node_param_writes()
        self._params = value

    def _sync_node_param_writes(self) -> bool:
        """Take every ``node.params`` write since the last sync into the
        graph: the one place a node write reaches ``gm.params``.  Return
        whether it marked the graph dirty.

        Called on every read (and assignment) of :attr:`params`, by every
        entry point (through :meth:`_check_static_data_dirty`) and at the
        start of :meth:`compile`, so every reader and every runner sees one
        state.

        * A write to a constant ``gm.params`` carries is copied into that
          leaf, at the leaf's dtype: the compiled step reads ``gm.params``
          on every call, so it needs no recompile.
        * A write to anything else -- a structural value, any constant of a
          node on the three-argument contract -- is read when the step is
          traced, so the graph is marked dirty and the next run of any entry
          point recompiles.

        Order: a pending node write is newer than every read of
        ``gm.params`` before it, because each read syncs, so it wins over a
        ``gm.params`` value written through such a read; and a
        ``gm.params`` write made after it reads ``gm.params`` first, which
        takes the node write in, so that write wins in turn.  (Only a leaf
        written through a reference held across the node write, with no
        read in between, loses to it.)  A write is a counted write to the
        node's ``_ParamsDict``; a key the mapping's replacement changed
        (``node.params = {**node.params, "k": v}`` is stored as a new
        ``_ParamsDict`` that takes over the old one's counts and counts
        ``k``, not the keys it hands back with the value they had:
        ``_ParamsDict._succeed``, MADD-ANO-203); or a value changed in place
        -- an element of a list or a NumPy array, which no method of the
        mapping sees -- found by comparing each such value with its copy
        from the last sync (:func:`~maddening.core.node._mutated_keys`).  So
        a keyed write of the value the node already held counts, and one
        into a list is not lost (MADD-ANO-175).  Never compiles, so it is safe from a getter and
        from :meth:`compile`; node values are taken under
        ``jax.ensure_compile_time_eval``, so a sync inside a trace stores
        concrete arrays.
        """
        marked = False
        nodes_live = self._params.get("nodes") or {}
        for name, spec in self._nodes.items():
            params = spec.node.params
            seen = self._node_writes_seen.get(name)
            if seen is None:
                continue                    # added since the last compile
            was, counts_seen, snapshot = seen
            counts = _param_probes._write_counts(params)
            # The mapping the last sync saw, or one that has taken over from
            # it since (``node.params = {...}``: ``_ParamsDict._succeed``),
            # whose counts continue its counts.
            continued = params is was or (
                counts is not None
                and getattr(params, "_lineage", None) is getattr(was, "_lineage", object()))
            if counts is None and params is not was:
                # A mapping that does not count (``node.__dict__["params"]``
                # assigned directly) put in the old one's place: nothing says
                # which keys were written, so all were.
                written = set(params) | set(was if isinstance(was, dict) else ())
            elif counts is None:
                written = _mutated_keys(params, snapshot)
            elif continued:
                # Counted writes -- a replacement counts the keys it changed,
                # not the ones it handed back (MADD-ANO-203) -- and values
                # changed in place.
                written = {k for k in set(counts) | set(counts_seen)
                           if counts.get(k, 0) != counts_seen.get(k, 0)}
                written |= _mutated_keys(params, snapshot)
            else:
                # A counting mapping from elsewhere stored as it is (a
                # sharded wrapper shares its node's): its counts say nothing
                # about this graph's last sync, so its values do, beside
                # whatever the old mapping had pending when it went.
                before = _param_probes._write_counts(was) or {}
                written = {k for k in set(before) | set(counts_seen)
                           if before.get(k, 0) != counts_seen.get(k, 0)}
                written |= _mutated_keys(was, snapshot)
                if isinstance(was, dict):
                    written |= _replaced_keys(was, params)
                else:
                    written |= set(params)
            if not written and params is was:
                continue
            live = nodes_live.get(name)
            values: dict = {}
            if written and live is not None:
                with jax.ensure_compile_time_eval():
                    values = spec.node.params_pytree()
            for key in written:
                if live is None or key not in live or key not in values \
                        or jnp.shape(values[key]) != jnp.shape(live[key]):
                    marked = True
                    continue
                with jax.ensure_compile_time_eval():
                    live[key] = jnp.asarray(values[key], dtype=jnp.asarray(live[key]).dtype)
            self._node_writes_seen[name] = (params, counts or {}, _mutable_snapshot(params))
        if marked:
            self._dirty = True
        return marked

    def _record_param_sync(self) -> None:
        """After a compile commits: every node's mapping, write counts and
        in-place-mutable values (a copy of each list or array,
        :func:`~maddening.core.node._mutable_snapshot`), as of now, are what
        the next sync compares against."""
        self._node_writes_seen = {
            name: (spec.node.params, _param_probes._write_counts(spec.node.params) or {},
                   _mutable_snapshot(spec.node.params))
            for name, spec in self._nodes.items()
        }

    def _merge_live_params(self, fresh: dict, live: Optional[dict]) -> dict:
        """``fresh`` (the nodes' own values) with every leaf of ``live``
        (``gm.params``) that still fits written over it; leaves that no
        longer fit (owner or key gone, shape changed) are dropped with a
        ``RuntimeWarning``.

        ``live`` wins because :meth:`compile` syncs first: every
        ``node.params`` write is already in it by the time this runs
        (:meth:`_sync_node_param_writes`), so what it carries is the newest
        value of every leaf -- a calibration survives a recompile, and a
        node write reaches it.
        """
        if not live:
            return fresh
        dropped = []
        for section in ("nodes", "mappings"):
            fresh_sec = fresh.setdefault(section, {})
            for owner, leaves in (live.get(section) or {}).items():
                if owner not in fresh_sec:
                    if leaves:
                        dropped.append(f"{section}[{owner!r}]")
                    continue
                for key, value in leaves.items():
                    base = fresh_sec[owner].get(key)
                    if base is None:
                        dropped.append(f"{section}[{owner!r}][{key!r}]")
                        continue
                    base_dtype = jnp.asarray(base).dtype
                    if jnp.shape(value) != jnp.shape(base):
                        dropped.append(f"{section}[{owner!r}][{key!r}] (shape changed)")
                        continue
                    # A Python float assigned into gm.params is float64
                    # under x64 and weak-typed otherwise: coerce to the
                    # leaf's own dtype (strongly typed) so it is kept and
                    # the jitted step is not retraced.
                    fresh_sec[owner][key] = jnp.asarray(value, dtype=base_dtype)
        if dropped:
            warnings.warn(
                "compile() dropped live gm.params leaves that no longer fit "
                f"the graph: {dropped}", RuntimeWarning, stacklevel=3,
            )
        return fresh

    def _check_param_shapes(self, section: str, owner: str, leaves: dict) -> None:
        """A leaf of the wrong shape would broadcast the node's *state* to
        that shape for good; shapes are static, so this is free."""
        expected = (getattr(self, "_params_shapes", None) or {}).get(section, {}).get(owner)
        if not expected:
            return
        for key, v in leaves.items():
            want = expected.get(key)
            if want is not None and tuple(jnp.shape(v)) != want:
                raise ValueError(
                    f"params[{section!r}][{owner!r}][{key!r}] has shape "
                    f"{tuple(jnp.shape(v))}, expected {want}"
                )

    def _coerce_params_leaves(self, tree: dict) -> None:
        """In place: non-array / weak-typed leaves -> strongly typed arrays
        of the dtype the leaf had at compile time (float32 fallback)."""
        dtypes = getattr(self, "_params_dtypes", {}) or {}
        for section in ("nodes", "mappings"):
            for owner, leaves in (tree.get(section) or {}).items():
                if not isinstance(leaves, dict):
                    continue
                for key, v in leaves.items():
                    dt = dtypes.get(section, {}).get(owner, {}).get(key)
                    if not hasattr(v, "dtype"):
                        leaves[key] = jnp.asarray(v, dtype=dt or jnp.float32)
                    elif getattr(v, "weak_type", False):
                        leaves[key] = v.astype(v.dtype)
                self._check_param_shapes(section, owner, leaves)

    @property
    def trace_count(self) -> int:
        """How many times the compiled step has been traced since the
        last ``compile()`` (0 before the first step; more than 1 after
        steady state means the step is being retraced).

        This counts ``step`` / ``run`` only.  ``run_scan`` and its
        siblings build their own program around the step and are counted
        by :attr:`scan_trace_count`; a loop that kept recompiling a scan
        used to leave ``trace_count`` at 1 and so look healthy.
        """
        return int(getattr(self, "_n_traces", 0))

    @property
    def scan_trace_count(self) -> int:
        """How many scan programs have been traced since the last
        ``compile()``.

        ``run_scan``, ``run_scan_with_history``, ``run_sweep`` and
        ``run_adaptive_scan`` each build a ``jax.lax.scan`` around the
        step function and cache it (see :meth:`_cached_scan`).  This
        counts the Python traces of those programs -- one per XLA
        compile.  Calling the same entry point again with the same step
        count and the same argument shapes must not increase it.
        """
        return int(getattr(self, "_n_scan_traces", 0))

    def _count_scan_trace(self) -> None:
        """Record one Python trace of a cached scan program."""
        self._n_scan_traces += 1

    def reset_params(self) -> None:
        """Discard live/calibrated values: ``gm.params`` becomes the
        constructor snapshot again (no recompile needed)."""
        self.params = self._snapshot_params()
        self._params_verified = {
            owner: dict(leaves) for owner, leaves in self.params.get("nodes", {}).items()
        }

    def _params_or_default(self, params) -> dict:
        """``gm.params`` when ``params`` is None; otherwise ``params``
        completed from ``gm.params``: a node or key the caller left out
        keeps its *live* value (not the constructor constant), so a
        partial pytree means "override these" and nothing else."""
        if params is None:
            # A Python scalar assigned into gm.params (``gm.params[...] =
            # 32.0``) would reach the jitted step weak-typed and retrace
            # it; coerce such leaves in place to the leaf dtype recorded
            # at compile time.
            self._coerce_params_leaves(self.params)
            self._refuse_baked_param_writes(self.params, live=True)
            return self.params
        if not isinstance(params, dict):
            return params                  # let _validate_params complain
        out = {}
        for section in ("nodes", "mappings"):
            live_sec = self.params.get(section, {})
            given = params.get(section, {}) or {}
            merged = {owner: dict(leaves) for owner, leaves in live_sec.items()}
            for owner, leaves in given.items():
                if owner in merged and isinstance(leaves, dict):
                    # keep the live leaf's dtype for a Python scalar the
                    # caller hands in (weak types retrace the step)
                    fixed = {
                        k: (jnp.asarray(v, dtype=jnp.asarray(merged[owner][k]).dtype)
                            if k in merged[owner] and not hasattr(v, "dtype") else v)
                        for k, v in leaves.items()
                    }
                    merged[owner] = {**merged[owner], **fixed}
                else:
                    merged[owner] = leaves      # unknown owner: validation reports it
            out[section] = merged
        for k, v in params.items():
            if k not in ("nodes", "mappings"):
                out[k] = v
        # The live leaves this completion carries over are checked as live
        # ones; the caller's own leaves are not refused (see
        # _refuse_baked_param_writes for why).
        self._refuse_baked_param_writes(
            {"nodes": {o: {k: v for k, v in leaves.items()
                           if self.params.get("nodes", {}).get(o, {}).get(k) is v}
                       for o, leaves in out.get("nodes", {}).items()
                       if isinstance(leaves, dict)}},
            live=True,
        )
        return _graph_specs._strong_typed(out)

    def _validate_params(self, params: dict) -> None:
        """Reject a ``params`` pytree that names something the step cannot
        use.  Runs Python-side at trace time (dict keys are static), so
        it costs nothing per step.

        A node that does not take ``params`` is *absent* from
        ``gm.params`` — passing an entry for it would be silently
        ignored, and a gradient with respect to it silently zero — so an
        explicit entry is an error, as is an unknown parameter name.
        """
        nodes = params.get("nodes", {}) if isinstance(params, dict) else None
        if nodes is None:
            raise TypeError(
                "params must be a dict with a 'nodes' entry (see GraphManager.params)"
            )
        for node_name, node_params in nodes.items():
            spec = self._nodes.get(node_name)
            if spec is None:
                raise ValueError(
                    f"params['nodes'] names unknown node {node_name!r}; "
                    f"graph nodes: {sorted(self._nodes)}"
                )
            if not spec.accepts_params:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] given, but "
                    f"{type(spec.node).__name__}.update() takes no 'params' "
                    "keyword: its constants are baked into the trace, so this "
                    "entry would be ignored and any gradient with respect to it "
                    "would be zero.  Migrate the node (declare "
                    "update(self, state, boundary_inputs, dt, *, params=None) "
                    "and read constants from params) or drop the entry."
                )
            known = set(spec.node.params_pytree())
            unknown = set(node_params) - known
            if unknown:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] has unknown key(s) "
                    f"{sorted(unknown)}; {type(spec.node).__name__}.params_pytree() "
                    f"exposes {sorted(known)}"
                )
            missing = known - set(node_params)
            if missing:
                raise ValueError(
                    f"params['nodes'][{node_name!r}] is missing key(s) "
                    f"{sorted(missing)}.  The compiled step needs a complete "
                    "pytree (a missing leaf would silently fall back to the "
                    "constructor constant); pass a partial tree through "
                    "gm.step / gm.run_scan(params=...), which completes it "
                    "from the live gm.params."
                )
            self._check_param_shapes("nodes", node_name, node_params)
        absent = [
            n for n, sp in self._nodes.items() if sp.accepts_params and n not in nodes
        ]
        if absent:
            raise ValueError(
                f"params['nodes'] is missing node(s) {sorted(absent)}.  The "
                "compiled step needs a complete pytree (a missing node would "
                "silently use its constructor constants); pass a partial tree "
                "through gm.step / gm.run_scan(params=...), which completes it "
                "from the live gm.params."
            )
        mapped = {e.key: e for e in self._edges if e.mapping is not None}
        given_maps = params.get("mappings", {}) or {}
        absent_maps = sorted(set(mapped) - set(given_maps))
        if absent_maps:
            raise ValueError(
                f"params['mappings'] is missing edge(s) {absent_maps}; pass a "
                "partial tree through gm.step / gm.run_scan(params=...)."
            )
        for key, weights in given_maps.items():
            edge = mapped.get(key)
            if edge is None:
                raise ValueError(
                    f"params['mappings'] names unknown edge {key!r}; mapped "
                    f"edges: {sorted(mapped)}"
                )
            mapping = edge.mapping
            assert mapping is not None  # `mapped` is filtered on it above
            known = set(mapping.params_pytree())
            unknown = set(weights) - known
            if unknown:
                raise ValueError(
                    f"params['mappings'][{key!r}] has unknown key(s) "
                    f"{sorted(unknown)}; the mapping exposes {sorted(known)}"
                )
            self._check_param_shapes("mappings", key, weights)

    # ------------------------------------------------------------------
    # A write to a leaf the compiled step cannot read is refused
    # ------------------------------------------------------------------

    def _params_read_by_step(self) -> Optional[set]:
        """``{(node, key)}`` of ``params["nodes"]`` the compiled step reads,
        or ``None`` when that cannot be told (no compiled step, a dirty
        graph, a step that does not trace).

        One trace of the step per compile, taken lazily: only a leaf whose
        value differs from its node's asks.  See :func:`_param_leaves_read`.
        """
        gen = self._compile_generation
        cached = self._step_reads
        if cached is not None and cached[0] == gen:
            return cached[1]
        reads: Optional[set] = None
        step_fn = self._raw_step_fn
        if step_fn is not None and not self._dirty:
            try:
                with quiet_warnings():
                    # A node that warns at trace time warned on the real
                    # trace already; this one is bookkeeping.
                    reads = _param_probes._param_leaves_read(
                        step_fn, self._state, self._default_external_inputs(),
                        self.params,
                    )
            except Exception:  # noqa: BLE001 - the real step reports it
                # Not cached: a pytree the step refuses (a stray key) is
                # the step's to report, and the next check asks again.
                return None
        self._step_reads = (gen, reads)
        return reads

    def _baked_leaf_reason(self, owner: str, key: str) -> Optional[str]:
        """Why the compiled step cannot read ``params["nodes"][owner][key]``,
        or ``None`` when it can (or that cannot be told).

        Two sources, declared first: a parameter a
        :meth:`~maddening.core.node.SimulationNode.static_data_deps` entry
        names is baked into that static when the node is constructed -- the
        step reads the static -- and a parameter that neither the traced
        step nor the node's own hooks (with every declared boundary input
        supplied) have a path from: an ``initial_*`` entry, which only
        ``initial_state()`` reads, or geometry a node consumed in
        ``__init__``.  A parameter only *this* graph does not exercise (a
        ball's ``elasticity`` without a table edge) is not refused.
        """
        spec = self._nodes.get(owner)
        if spec is None:
            return None
        declared_reason = _param_probes._static_deps_reason(spec.node, key)
        if declared_reason is not None:
            return declared_reason
        reads = self._params_read_by_step()
        if reads is None or (owner, key) in reads:
            return None
        # Dead in *this* step is not enough: a ball's ``elasticity`` is read
        # only when a ``table_position`` edge exists, and a value carried in
        # gm.params for it is latent, not ignored -- add the edge and the
        # carried value is the one used, which is what to_dict() records.
        # Refused only when the node's own hooks cannot read it either, with
        # every boundary input it declares supplied.
        node_reads = self._node_param_reads(owner)
        if node_reads is None or key in node_reads:
            return None
        return (
            "the node cannot read it: no operation of the compiled step takes "
            f"it as an input, and {type(spec.node).__name__}'s own update / "
            "flux / interface-correction hooks do not either with every "
            "boundary input they declare supplied (an initial condition, "
            "which only initial_state() reads, from the node; or a value the "
            "node consumed when it was constructed)"
        )

    def _node_param_reads(self, owner: str) -> Optional[set]:
        """Keys of ``params["nodes"][owner]`` the node's own hooks read.

        One trace of ``update`` (and of ``compute_boundary_fluxes`` /
        ``compute_interface_correction`` where the graph passes them
        ``params``) with a zero value for every input
        ``boundary_input_spec()`` declares, walked like the step (see
        :func:`_live_jaxpr_inputs`).  ``None`` when the hooks do not trace
        that way (an input the spec does not declare, say): then nothing
        is refused on this ground.  Before the node's first compile its own
        :meth:`~maddening.core.node.SimulationNode.params_pytree` stands in
        for its ``gm.params`` entry.  Cached per compile and per node
        object, so a node replaced under the same name between two
        compiles is asked again.
        """
        gen = self._compile_generation
        spec = self._nodes[owner]
        node = spec.node
        cached = self._node_reads.get(owner)
        if cached is not None and cached[0] == gen and cached[1] is node:
            return cached[2]
        try:
            bi = _param_probes._declared_boundary_zeros(node)
            leaves = self.params.get("nodes", {}).get(owner)
            if leaves is None:
                leaves = node.params_pytree()
            with quiet_warnings():
                args = (self._state[owner], bi, leaves)
                closed = jax.make_jaxpr(
                    lambda st, b, p: _param_probes._hook_outputs(spec, st, b, p))(*args)
            paths = [pth for pth, _ in jax.tree_util.tree_flatten_with_path(args)[0]]
            if len(paths) != len(closed.jaxpr.invars):
                return None
            live = _param_probes._live_jaxpr_inputs(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
        except Exception:  # noqa: BLE001 - not traceable this way: refuse nothing
            return None
        reads = {
            pth[1].key for pth, keep in zip(paths, live)
            if keep and len(pth) >= 2 and getattr(pth[0], "idx", None) == 2
        }
        self._node_reads[owner] = (gen, node, reads)
        return reads

    def _param_leaves_the_step_cannot_read(self) -> dict[tuple[str, str], str]:
        """``{(node, key): reason}`` for every leaf of ``params["nodes"]``
        that a write to the pytree alone could not change the step for.

        A leaf a ``static_data_deps`` entry names (the step reads the static
        built from it), and -- when the compiled step can be traced -- a
        leaf no operation of the compiled step takes as an input.  Unlike
        :meth:`_baked_leaf_reason` this does *not* spare a latent leaf the
        node's own hooks would read with an input this graph does not
        connect (a ball's ``elasticity`` without a table edge): it answers
        for a frozen graph, which is what an exported FMU is -- no edge can
        be added to one, so such a leaf is a knob that does nothing.  Used
        by :func:`maddening.fmi.model_description.build_model_description`
        and the FMI sidecar, which write the pytree and nothing else.
        """
        reads = self._params_read_by_step()
        out: dict[tuple[str, str], str] = {}
        for owner, leaves in (self.params.get("nodes") or {}).items():
            spec = self._nodes.get(owner)
            if spec is None or not spec.accepts_params or not isinstance(leaves, dict):
                continue
            for key in leaves:
                reason = _param_probes._static_deps_reason(spec.node, key)
                if reason is None and reads is not None and (owner, key) not in reads:
                    reason = (
                        "no operation of the compiled step takes it as an "
                        "input (an initial condition, which only "
                        "initial_state() reads; a value the node consumed "
                        "when it was constructed; or one only an input this "
                        "graph does not connect would read)"
                    )
                if reason is None:
                    reason = self._mapping_point_leaf_reason(owner, key)
                if reason is not None:
                    out[(owner, key)] = reason
        return out

    # ------------------------------------------------------------------
    # A write under an interface mapping's point reference
    # ------------------------------------------------------------------

    def _mapping_node_references(self, owner: str) -> list[tuple[str, str, str]]:
        """``(edge key, point-set argument, field)`` for every
        ``{"node": owner, "field": ...}`` point reference a mapped edge of
        this graph was built from."""
        out: list[tuple[str, str, str]] = []
        for edge in self._edges:
            spec = getattr(edge.mapping, "spec", None) if edge.mapping is not None else None
            points = getattr(spec, "points", None)
            if not isinstance(points, dict):
                continue
            for arg, ref in points.items():
                if (isinstance(ref, dict) and ref.get("node") == owner
                        and isinstance(ref.get("field"), str)):
                    out.append((edge.key, arg, ref["field"]))
        return out

    def _saved_params_bases(self, owner: str, live: Optional[dict]) -> list[dict]:
        """The params a guard rebuilds node ``owner`` from when it asks what
        a save of the graph would reload, most faithful first: the node's
        own ``params`` with the live leaves ``live`` written over them
        (what :meth:`to_dict` writes; ``live=None`` reads the graph's own
        leaves, through no property), then the node's own ``params`` alone.

        The second is there for a graph whose live leaves cannot be read
        or that its constructor refuses.  It used to be the only one: a
        guard then judged a write beside the values the node was *built*
        with, which a fit or a checkpoint load leaves behind, and its
        answer changed with them (:meth:`_mapped_points_moved_by`).
        """
        shared = getattr(self._nodes[owner].node, "params", None)
        shared = dict(shared) if isinstance(shared, dict) else {}
        if live is None:
            live = (self._params.get("nodes") or {}).get(owner)
        bases = []
        try:
            saved = self._node_params_with_leaves(owner, live)
        except Exception:  # noqa: BLE001 - the graph cannot say what it would save
            saved = None
        if saved is not None:
            bases.append({**shared, **saved})
        bases.append(shared)
        return bases

    def _mapped_points_moved_by(self, owner: str, changes: dict[str, Any],
                                live: Optional[dict] = None, *,
                                _probing: bool = False,
                                ) -> Optional[list[tuple[str, str, str]]]:
        """The point references of :meth:`_mapping_node_references` whose
        points a node rebuilt with ``changes`` written into its params reads
        differently from the node rebuilt from its own; ``[]`` when none
        moves, ``None`` when that cannot be told (no reference to ``owner``,
        no key of ``changes`` is a constructor parameter, or the node is not
        rebuilt from its params -- the constructor refusing the new values
        is the callers' other checks' to report, not this).

        Rebuilt the way :meth:`from_dict` rebuilds a saved node, from its
        class and params -- the node, or the node a wrapper wraps, whichever
        is rebuilt that way (:func:`_params_holders`) -- and the field read
        by the rule a point reference resolves by
        (``mapping_spec._node_point_field``).  Both sides are rebuilt, so
        nothing about the live node object (a sharded slice) enters the
        comparison.

        The node with ``changes`` is the one a save would reload: built from
        the node's params with the live leaves ``live`` (the node's leaves
        of :attr:`params` before the write; ``None`` reads the graph's)
        written over them, then ``changes`` (:meth:`_saved_params_bases`).
        It used to be built from the node's own params alone, and the
        constructor refusing *those* with ``changes`` counted as "cannot be
        told": after a fit or a checkpoint load had moved another leaf (a
        rod's diffusivity, down), a ``length`` that is unstable at the
        diffusivity the rod was built with and stable at the one it runs
        with was refused by no check -- this one could not build the node,
        and the check that asks the constructor asked it with the live
        value, which takes it.  ``PUT /graph/params`` answered 200, a
        ``gm.params`` write ran, the mapping kept the old grid's operator
        and the save did not load.  The same in one request that writes the
        length with a diffusivity it is stable at, so the callers ask this
        of a whole write as well as of each key.

        Where the constructor refuses ``changes`` whatever they are written
        beside, the node cannot be built to read its points, and each
        changed key is asked instead whether *another* value of it moves
        them (:meth:`_points_moved_by_some_value`): a key the points are
        derived from is reported as moving them.  So the answer no longer
        waits on a different check to refuse the value: ``gm.params`` has no
        check that asks the constructor, and took a rod's unstable
        ``length`` under a mapping.
        """
        refs = self._mapping_node_references(owner)
        spec = self._nodes.get(owner)
        if not refs or spec is None:
            return None
        node = spec.node
        shared = getattr(node, "params", None)
        if not isinstance(shared, dict):
            return None
        changes = {k: v for k, v in changes.items() if k in shared}
        if not changes:
            return None
        from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
            PointReferenceError,
            _node_point_field,
        )
        for candidate in _param_probes._params_holders(node):
            cls = type(candidate)

            def build(params, cls=cls, candidate=candidate):
                with quiet_warnings():
                    return cls(name=candidate.name, timestep=candidate.delta_t, **params)

            try:
                before = build(dict(shared))
            except Exception:  # noqa: BLE001 - not rebuilt from its params
                continue
            after = None
            for base in self._saved_params_bases(owner, live):
                try:
                    after = build({**base, **changes})
                except Exception:  # noqa: BLE001 - the constructor's refusal, reported elsewhere
                    continue
                break
            if after is None:
                if _probing:
                    return None
                moved = []
                for key, value in changes.items():
                    if _param_probes._leaf_values_equal(value, shared[key]):
                        continue
                    for found in self._points_moved_by_some_value(owner, key) or ():
                        if found not in moved:
                            moved.append(found)
                return moved or None
            moved = []
            for edge_key, arg, field_name in refs:
                try:
                    was = _node_point_field(before, owner, field_name)
                except PointReferenceError:
                    continue        # not this node's field: to_dict names a stale reference
                try:
                    now = _node_point_field(after, owner, field_name)
                except PointReferenceError:
                    moved.append((edge_key, arg, field_name))
                    continue
                if not _param_probes._leaf_values_equal(was, now) or was.dtype != now.dtype:
                    moved.append((edge_key, arg, field_name))
            return moved
        return None

    def _mapping_point_write_reason(self, owner: str, changes: dict[str, Any],
                                    live: Optional[dict] = None) -> Optional[str]:
        """Why writing ``changes`` into node ``owner``'s params would leave
        an interface mapping on the points of the old values, or ``None``.

        A mapped edge built from a ``{"node", "field"}`` point reference
        reads the node's field **once**, when the mapping is built; the
        weights are fixed from then on (``mapping_spec``'s "A node
        reference does not follow a calibrated parameter").  A field the
        node *derives* from a parameter when it is constructed -- a uniform
        ``HeatNode``'s ``grid_x``, built from ``length`` -- is therefore not
        moved by a write of that parameter, even where the node's own step
        reads the parameter itself (the rod's ``dx = length / n_cells``).
        Such a write used to be taken: ``PUT /graph/params`` answered 200
        and a ``gm.params`` write ran, the rod stepping with the new length
        and the mapping interpolating from the old grid, and the config
        :meth:`to_dict` then wrote did not load (:meth:`from_dict` rebuilds
        the mapping from the new grid, and the recorded ``sha256`` refuses
        it).  ``changes`` maps keys to the values ``node.params`` would hold;
        ``live`` is the node's live leaves beside which they are written
        (:meth:`_mapped_points_moved_by`).
        """
        return self._moved_points_reason(
            owner, self._mapped_points_moved_by(owner, changes, live))

    def _moved_points_reason(self, owner: str,
                             moved: Optional[list[tuple[str, str, str]]]) -> Optional[str]:
        """The refusal for the point references ``moved`` names, or ``None``."""
        if not moved:
            return None
        cls = type(self._nodes[owner].node).__name__
        edges = sorted({edge_key for edge_key, _, _ in moved})
        fields = sorted({f"{owner}.{field_name} ({arg})" for _, arg, field_name in moved})
        return (
            f"the interface mapping on edge {', '.join(repr(e) for e in edges)} was "
            f"built from {', '.join(fields)}, which {cls} derives from it when it "
            "is constructed, and a mapping reads its points once: the node would "
            "run with the new value while the mapping kept the weights of the old "
            "points, and a graph saved with it would not reload this one "
            "(from_dict() rebuilds the mapping from the new points, which a "
            "recorded sha256 refuses)"
        )

    def _mapping_point_leaf_reason(self, owner: str, key: str) -> Optional[str]:
        """:meth:`_mapping_point_write_reason` for *any* new value of the
        leaf: asked with the node's own value moved by half again (or by
        half, if the constructor refuses that), for the callers that decide
        once whether a leaf is a parameter at all -- an exported FMU's
        tunable set (:meth:`_param_leaves_the_step_cannot_read`), where a
        write reaches the compiled step directly and no later check sees
        it.  ``None`` when the leaf moves no referenced point set, or that
        cannot be told."""
        return self._moved_points_reason(owner, self._points_moved_by_some_value(owner, key))

    def _points_moved_by_some_value(self, owner: str,
                                    key: str) -> Optional[list[tuple[str, str, str]]]:
        """:meth:`_mapped_points_moved_by` for the node's own value of
        ``key`` moved by half again, or by half if the constructor refuses
        that; ``None`` when neither can be built or the value is not a
        number."""
        spec = self._nodes.get(owner)
        if spec is None or not self._mapping_node_references(owner):
            return None
        current = (getattr(spec.node, "params", None) or {}).get(key)
        if current is None or isinstance(current, bool):
            return None
        try:
            arr = np.asarray(current, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        for factor in (1.5, 0.5):
            probe = np.where(arr == 0.0, 1.0, arr * factor)
            moved = self._mapped_points_moved_by(owner, {key: probe.tolist()}, _probing=True)
            if moved is not None:
                return moved
        return None

    def _unused_node_write_reason(self, owner: str, key: str, value: Any) -> Optional[str]:
        """Why a write of ``params[key] = value`` to node ``owner`` that
        reaches the node itself -- ``node.params`` and, for a leaf of the
        params pytree, :attr:`params` as well, which is what ``PUT
        /graph/params`` writes -- would be used by nothing the running
        graph computes; ``None`` when it would be used, or when that cannot
        be told.  ``value`` is what would be stored in ``node.params``; the
        caller has already established that it differs from the node's
        current value.

        :meth:`_refuse_baked_param_writes` cannot see such a write: it
        compares ``gm.params`` against the node's own value, and this write
        changes both.  But writing ``node.params`` rebuilds nothing the
        node derived from the value when it was constructed, so the value
        would be reported by ``GET``, written out by :meth:`to_dict` and
        :meth:`save_state`, and ignored by every step -- a reloaded graph
        runs a different model.  The decision, in order:

        1. a :meth:`~maddening.core.node.SimulationNode.static_data_deps`
           entry names ``key``: refused (the step reads the static);
        2. a leaf of the params pytree that the compiled step, or the
           node's own hooks with every declared boundary input supplied,
           read from the injected params: used, from the next step (the
           liveness walk of :meth:`_baked_leaf_reason`);
        3. a structural key (not a pytree leaf): used when the node's hooks
           trace differently with the new value in ``node.params`` -- the
           write marks the graph dirty, and the recompile traces them
           again;
        4. otherwise: used when ``initial_state()`` returns something else
           with the new value -- it takes effect at the next reset, the way
           an ``initial_*`` condition does;
        5. otherwise: refused -- the node consumed the value when it was
           constructed, or nothing reads it at all (a value nothing reads
           cannot make a reloaded graph differ, but it is not a parameter
           of the running graph either, and the two cannot be told apart
           without constructing the node again).

        Steps 3 and 4 run the node's code on a shallow copy that reads the
        new value (:func:`_param_probe_pair`: the node, or the node a
        wrapper that cannot be copied wraps), never on the node itself.
        When no faithful copy can be made at all, the write is refused.
        An earlier revision accepted it, on the grounds that nothing could
        be told, so a node holding a method bound to itself -- and every
        sharded wrapper, which closes over the node it wraps -- had its
        structural writes answered 200 whether or not they were used.
        When the copied code raises, nothing is refused.  A value the node
        consumed at construction *and* reads again later (a geometry
        parameter that also shapes the initial fill) passes step 4: like
        the graph-level walk, this detects "no path at all", so a node
        that bakes a parameter must declare it in ``static_data_deps``,
        which refuses it on both surfaces (``LBMPipeNode`` declares its
        geometry since 0.4.0 for exactly that reason).
        """
        spec = self._nodes.get(owner)
        if spec is None:
            return None
        node = spec.node
        declared_reason = _param_probes._static_deps_reason(node, key)
        if declared_reason is not None:
            return declared_reason
        cls = type(node).__name__
        live = self.params.get("nodes", {}).get(owner) or {}
        pytree_leaf = spec.accepts_params and (key in live or key in node.params_pytree())
        if pytree_leaf:
            reads = self._params_read_by_step()
            if reads is not None and (owner, key) in reads:
                return None
            node_reads = self._node_param_reads(owner)
            if node_reads is None or key in node_reads:
                return None
            where = (
                "no operation of the compiled step takes it as an input, "
                f"{cls}'s own update / flux / interface-correction hooks do "
                "not read it with every boundary input they declare supplied"
            )
        else:
            if _param_probes._param_probe_pair(node, key, value) is None:
                return _param_probes._unprobeable_write_reason(node)
            if self._hooks_trace_depends(owner, key, value) is not False:
                return None
            where = (
                f"{cls}'s update / flux / interface-correction hooks trace "
                "identically with the new value in node.params, so the "
                "recompile the write asks for would not use it"
            )
        if key in node.params:
            if _param_probes._param_probe_pair(node, key, value) is None:
                return _param_probes._unprobeable_write_reason(node)
            if self._initial_state_depends(owner, key, value) is not False:
                return None
        return (
            f"{where}, and initial_state() returns the same state with it: "
            f"nothing {cls} computes while it runs reads the new value (it "
            "consumed the value when it was constructed, if it uses it at all)"
        )

    def _hooks_trace_depends(self, owner: str, key: str, value: Any) -> Optional[bool]:
        """Do the node's hooks trace differently with ``node.params[key] =
        value``?  ``None`` when that cannot be told (no faithful copy, a
        trace that raises).  Compared on shallow copies holding the current
        and the new value, as jaxpr text plus constant values.

        A wrapper that cannot be copied is answered for by the node it
        wraps (:func:`_param_probe_pair`): that node's own hooks, traced on
        the state it builds, which is the physics the wrapper distributes.
        """
        spec = self._nodes[owner]
        node = spec.node
        pair = _param_probes._param_probe_pair(node, key, value)
        if pair is None:
            return None
        traces = []
        try:
            bi = _param_probes._declared_boundary_zeros(node)
            leaves = None
            if spec.accepts_params:
                leaves = self.params.get("nodes", {}).get(owner)
                if leaves is None:
                    leaves = node.params_pytree()
            descended = pair[2]
            state = pair[0].initial_state() if descended else self._state[owner]
            for probe in pair[:2]:
                probe_spec = _graph_specs._NodeSpec(
                    node=probe, update_fn=probe.update, timestep=spec.timestep,
                    accepts_params=(
                        _method_accepts_params(probe, "update") and leaves is not None
                        if descended else spec.accepts_params),
                    flux_accepts_params=(
                        _method_accepts_params(probe, "compute_boundary_fluxes")
                        and leaves is not None
                        if descended else spec.flux_accepts_params),
                )
                with quiet_warnings():
                    closed = jax.make_jaxpr(
                        lambda st, b, p, _s=probe_spec: _param_probes._hook_outputs(_s, st, b, p),
                    )(state, bi, leaves)
                traces.append((str(closed.jaxpr), list(closed.consts)))
        except Exception:  # noqa: BLE001 - cannot tell: refuse nothing
            return None
        (text_a, consts_a), (text_b, consts_b) = traces
        if text_a != text_b or len(consts_a) != len(consts_b):
            return True
        return not all(_param_probes._leaf_values_equal(a, b) for a, b in zip(consts_a, consts_b))

    def _constructor_write_reason(self, owner: str, key: str, value: Any,
                                  others: Optional[dict[str, Any]] = None,
                                  live: Optional[dict] = None) -> Optional[str]:
        """Why the node's own constructor refuses its params with
        ``params[key] = value`` -- and the other changes of the same write,
        *others*, applied with it -- or ``None`` when it takes them or that
        cannot be told.

        *others* matters for a value that is valid only together with
        another: ``HeatNode`` ``stencil_order: 4`` at a Fourier number of
        0.4 is unstable, and with ``thermal_diffusivity: 0.2`` in the same
        request it is not; asked alone, the request was refused.

        A graph is saved (:meth:`to_dict`) as each node's class and params,
        and loaded by calling the class with them (:meth:`from_dict`), so a
        value the constructor refuses is one a saved graph cannot load.
        Written straight into ``node.params`` it bypassed that validation:
        ``PUT /graph/params`` took ``HeatNode``'s ``stencil_order=3``, the
        running rod kept its 2nd-order stencil (there is no 3rd), and the
        saved graph raised ``stencil_order must be 2 or 4`` on load.

        Asked of the node, or of the node a wrapper wraps (it shares the
        params dict, and is the one built from them), whichever is rebuilt
        from its current params as :meth:`from_dict` would rebuild it; a
        node that is not rebuilt that way is not asked.

        Asked with the params a save would carry -- the node's own with its
        live leaves ``live`` (``None``: the graph's) over them
        (:meth:`_saved_params_bases`) -- since those are what
        :meth:`from_dict` calls the constructor with.  With the node's own
        values alone, a ``stencil_order: 4`` that is stable at the
        diffusivity a checkpoint load had installed was refused for the
        diffusivity the rod was built with, and taken on a graph with the
        same live values that had reached them by a ``PUT``.
        """
        node = self._nodes[owner].node
        shared = getattr(node, "params", None)
        for candidate in _param_probes._params_holders(node):
            cls = type(candidate)

            def build(params, cls=cls, candidate=candidate):
                with quiet_warnings():
                    return cls(name=candidate.name, timestep=candidate.delta_t, **params)

            try:
                build(dict(shared or {}))
            except Exception:  # noqa: BLE001 - not rebuilt from its params
                continue
            try:
                build({**self._saved_params_bases(owner, live)[0], **(others or {}),
                       key: value})
            except Exception as exc:  # noqa: BLE001 - the constructor refuses it
                return (
                    f"{cls.__name__}'s constructor refuses it ({type(exc).__name__}: "
                    f"{exc}), so a graph saved with it would not load"
                )
            return None
        return None

    def _state_shape_write_reason(self, owner: str, key: str, value: Any) -> Optional[str]:
        """Why ``node.params[key] = value`` cannot be taken by the running
        node because it changes the layout of the state the node builds --
        the fields of ``initial_state()``, their shapes or dtypes (a cell
        count, a grid shape) -- or makes ``initial_state()`` raise; ``None``
        when the layout is unchanged or that cannot be told.

        Such a value is not an initial condition that "takes effect at the
        next reset": the running state keeps its old shape until then, and
        the step recompiled for the new value is traced against it.
        Unsharded, a ``HeatNode`` given a new ``n_cells`` failed its next
        step (a 500 through ``PUT /graph/params``, after a 200); sharded,
        it stepped the old 16 cells with ``dx = L/17`` and never closed the
        right rod end.  Refused on both, before anything is written.

        Evaluated on shallow copies holding the current and the new value
        (:func:`_param_probe_pair`), never on the node itself; a wrapper
        that cannot be copied is answered for by the node it wraps, which
        builds the state the wrapper places.  When no copy can be made this
        says nothing: :meth:`_unused_node_write_reason`, asked next, refuses
        such a write.
        """
        node = self._nodes[owner].node
        probes = _param_probes._param_probe_pair(node, key, value)
        if probes is None:
            return None
        with quiet_warnings():
            try:
                before = probes[0].initial_state()
            except Exception:  # noqa: BLE001 - cannot tell: refuse nothing
                return None
            try:
                after = probes[1].initial_state()
            except Exception as exc:  # noqa: BLE001 - the new value breaks it
                return (
                    f"{type(node).__name__}.initial_state() raises with it "
                    f"({type(exc).__name__}: {exc})"
                )

        def layout(state) -> dict:
            flat = jax.tree_util.tree_flatten_with_path(state)[0]
            return {
                jax.tree_util.keystr(path): (tuple(np.shape(leaf)),
                                             str(np.asarray(leaf).dtype))
                for path, leaf in flat
            }

        was, now = layout(before), layout(after)
        if was == now:
            return None
        changed = [
            f"{field} {was.get(field, 'absent')} -> {now.get(field, 'absent')}"
            for field in sorted(set(was) | set(now))
            if was.get(field) != now.get(field)
        ]
        return (
            "it changes the layout of the state the node builds ((shape, "
            f"dtype) of {'; '.join(changed)}), and the running state keeps "
            "its layout until it is reset: the step recompiled for the new "
            "value would be traced against a state it was not written for"
        )

    def _initial_state_depends(self, owner: str, key: str, value: Any) -> Optional[bool]:
        """Does ``initial_state()`` return something else with
        ``node.params[key] = value``?  ``None`` when that cannot be told.
        Evaluated on shallow copies holding the current and the new
        value, never on the node itself -- of the node a wrapper that
        cannot be copied wraps, when it is that one
        (:func:`_param_probe_pair`)."""
        node = self._nodes[owner].node
        pair = _param_probes._param_probe_pair(node, key, value)
        if pair is None:
            return None
        probes = pair[:2]
        try:
            with quiet_warnings():
                before = probes[0].initial_state()
                after = probes[1].initial_state()
        except Exception:  # noqa: BLE001 - cannot tell: refuse nothing
            return None
        if jax.tree.structure(before) != jax.tree.structure(after):
            return True
        return not _param_probes._leaf_values_equal(before, after)

    def _refuse_baked_param_writes(self, tree: Any, *, live: bool) -> None:
        """Refuse a leaf that differs from its node's value but that the
        compiled step cannot read.

        Such a write used to be accepted by ``gm.params``, ``check_params``
        and every run method, ignored by the step, and then serialised by
        :meth:`to_dict` -- so a reloaded graph ran a different model from
        the one that produced the numbers (``HeatNode.grid_points``, 0.093 K
        after one step; ``WaveletAdaptiveNode.mass``; the ``initial_*``
        leaves).  ``docs/user_guide/parameters.md`` promises the opposite:
        "not a silently ignored leaf".

        Only leaves of :attr:`params` itself are refused (``live=True``): a
        leaf that passes is remembered by identity, so a steady run pays one
        ``is`` per leaf per call and a value is compared only when a new
        object was written.  A caller's explicit ``params=`` pytree is not
        refused -- it is never serialised, and a leaf the step ignores may
        be one the caller's own code consumes (a residual that seeds the
        initial state from ``initial_velocity``, say); the live leaves a
        partial pytree is completed from are checked as live ones.  A
        traced leaf (a fit, an FIM) cannot be compared and is left alone.
        ``live=False`` checks without remembering.  The reference is the
        node's own
        :meth:`~maddening.core.node.SimulationNode.params_pytree`, so a write
        that also reaches the node is not refused here: ``PUT
        /graph/params`` writes both, and asks
        :meth:`_unused_node_write_reason` first, before writing anything.
        """
        nodes = tree.get("nodes") if isinstance(tree, dict) else None
        if not isinstance(nodes, dict):
            return
        verified = self._params_verified
        ctor_cache: dict[str, dict] = {}
        for owner, leaves in nodes.items():
            spec = self._nodes.get(owner)
            if spec is None or not spec.accepts_params or not isinstance(leaves, dict):
                continue            # _validate_params names these
            seen = verified.get(owner, {})
            for key, value in leaves.items():
                if seen.get(key) is value:
                    continue
                if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(value)):
                    continue
                if owner not in ctor_cache:
                    ctor_cache[owner] = spec.node.params_pytree()
                ctor = ctor_cache[owner].get(key)
                if ctor is None or not _param_probes._leaf_values_equal(value, ctor):
                    reason = self._baked_leaf_reason(owner, key)
                    consequence = (
                        "The value would be ignored by every run and then "
                        "written out by to_dict() / save_state(), so a reloaded "
                        "graph would run a different model from the one that "
                        "produced these results.  To change it, rebuild the "
                        "node with the new value (remove_node, then add_node)"
                    )
                    if reason is None and key in (getattr(spec.node, "params", None) or {}):
                        # Read by the node's own step, and still not a value
                        # the running graph can take: a mapped edge was
                        # built from points the node derives from it.
                        # Beside the node's other leaves in this tree: the
                        # node a save would reload is built from them all.
                        reason = self._mapping_point_write_reason(
                            owner, {key: np.asarray(value).tolist()},
                            live={k: v for k, v in leaves.items() if not any(
                                isinstance(x, jax.core.Tracer)
                                for x in jax.tree.leaves(v))})
                        consequence = (
                            "To change it, rebuild the node "
                            "with the new value and the mapped edge from its new "
                            "points (remove_node, add_node, then add_edge with a "
                            "mapping built from them)"
                        )
                    if reason is not None:
                        where = "gm.params" if live else "params"
                        shown = ""
                        if ctor is not None and np.size(ctor) <= 8:
                            shown = " " + np.array2string(
                                np.asarray(ctor), precision=7, separator=", ")
                        raise _BakedParamWrite(
                            f"{where}['nodes'][{owner!r}][{key!r}] differs from "
                            f"the node's own value{shown}, but {reason}.  "
                            f"{consequence}; to drop the edit, restore the leaf or "
                            "call gm.reset_params().",
                            owner=owner, key=key, value=value, own=ctor, reason=reason,
                        )
                if live:
                    verified.setdefault(owner, {})[key] = value

    # ------------------------------------------------------------------
    # ParamSpec: trainable mask, bounds, reparametrisation
    # ------------------------------------------------------------------

    @stability(StabilityLevel.EVOLVING)
    def param_specs(self) -> dict:
        """``{"nodes": {name: {key: ParamSpec}}, "mappings": {}}`` mirroring
        :attr:`params`: each node's :meth:`SimulationNode.param_specs`
        with the graph's :meth:`set_param_spec` overrides applied.
        Leaves without an entry use the default (trainable, unbounded)."""
        out: dict = {"nodes": {}, "mappings": {}}
        for name, spec in self._nodes.items():
            if not spec.accepts_params:
                continue
            merged = dict(spec.node.param_specs())
            merged.update(self._param_spec_overrides.get(name, {}))
            out["nodes"][name] = merged
        # Interface-mapping weights are geometry-derived operators, not
        # physical constants: frozen unless a learned edge opts in with
        # ``set_param_spec(edge.key, "H", ParamSpec())``.
        for edge in self._edges:
            if edge.mapping is None:
                continue
            merged = {
                k: ParamSpec(trainable=False, description="interface mapping weights")
                for k in edge.mapping.params_pytree()
            }
            merged.update(self._param_spec_overrides.get(edge.key, {}))
            out["mappings"][edge.key] = merged
        return out

    @stability(StabilityLevel.EVOLVING)
    def set_param_spec(self, node: str, key: str, spec: ParamSpec) -> None:
        """Override one parameter's :class:`ParamSpec` for this graph
        (e.g. freeze a node's ``mass`` when the data cannot identify it,
        or make a mapped edge's weights trainable by passing the edge
        key — ``"<src>.<field>-><tgt>.<field>"`` — as ``node``).

        Specs are optimiser-side metadata, so this does not dirty the
        graph — *unless* ``key`` is one a
        :meth:`~maddening.core.node.SimulationNode.static_data_deps`
        entry names.  Then it decides whether ``compile()`` refuses the
        graph (a static baked from a trainable parameter loses the
        gradient through it), so the verdict has to be taken again."""
        if not isinstance(spec, ParamSpec):
            raise TypeError(f"spec must be a ParamSpec, got {type(spec).__name__}")
        mapped = {e.key: e for e in self._edges if e.mapping is not None}
        if node in mapped:
            node_mapping = mapped[node].mapping
            assert node_mapping is not None  # `mapped` is filtered on it above
            known = node_mapping.params_pytree()
            if key not in known:
                raise KeyError(
                    f"mapping on edge {node!r} has no weight {key!r}; it exposes "
                    f"{sorted(known)}"
                )
            self._param_spec_overrides.setdefault(node, {})[key] = spec
            return
        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}")
        if not self._nodes[node].accepts_params:
            raise ValueError(
                f"node {node!r} takes no params; nothing to specify"
            )
        if key not in self._nodes[node].node.params_pytree():
            raise KeyError(
                f"node {node!r} has no parameter {key!r}; "
                f"params_pytree() exposes "
                f"{sorted(self._nodes[node].node.params_pytree())}"
            )
        # Asked *before* the override is committed: ``static_data_deps``
        # is a node-supplied method and may raise, and an override stored
        # without the dirty flag that goes with it is the half-applied
        # mutation the atomicity work is about.  It forwards from wrapped
        # nodes, so the outer declaration is enough to know whether this
        # key is load-bearing for the compile-time refusal.
        declared = self._nodes[node].node.static_data_deps() or {}
        dirties = any(key in names for names in declared.values())
        self._param_spec_overrides.setdefault(node, {})[key] = spec
        if dirties:
            self._dirty = True

    @stability(StabilityLevel.EVOLVING)
    def trainable_mask(self, params: Optional[dict] = None) -> dict:
        """``params``-shaped pytree of Python bools (``True`` = an
        optimiser may move the leaf)."""
        return _trainable_mask(self._params_or_default(params), self.param_specs())

    def unconstrain(self, params: Optional[dict] = None) -> dict:
        """Map trainable leaves to unconstrained optimiser coordinates
        (``log`` for positive constants, ``logit`` for intervals); other
        leaves pass through.  Inverse of :meth:`constrain`."""
        return _unconstrain(self._params_or_default(params), self.param_specs())

    def constrain(self, u: dict) -> dict:
        """Map optimiser coordinates back to a physical ``params`` pytree
        (also clips bounded identity leaves)."""
        return _constrain(u, self.param_specs())

    def check_params(self, params: Optional[dict] = None) -> None:
        """Raise ``ValueError`` if any leaf is outside its declared bounds
        or the pytree names a node/key the step cannot use."""
        params = self._params_or_default(params)
        self._validate_params(params)
        _check_bounds(params, self.param_specs())

    def nodes_without_params(self) -> list[str]:
        """Names of nodes whose ``update`` takes no ``params`` keyword —
        their constants are not differentiable through the graph."""
        return [n for n, s in self._nodes.items() if not s.accepts_params]

    def effective_node_params(self, name: str, params: Optional[dict] = None) -> dict:
        """The node's constructor ``params`` with the live values of
        :attr:`params` (or ``params``) written over them, as plain Python
        scalars/lists.  This is what serialisation stores, so a calibrated
        graph reloads with the calibrated constants."""
        live = self._params_or_default(params).get("nodes", {}).get(name, {})
        return self._node_params_with_leaves(name, live)

    def _node_params_with_leaves(self, name: str, live: Optional[dict]) -> dict:
        """:meth:`effective_node_params` for the leaves ``live`` (``{key:
        value}`` of one node), read through no property: the params a save
        of the graph would carry for the node if those were its live
        leaves.  The guards on a params write ask it of the leaves a write
        would leave, from inside the check :attr:`params` itself runs."""
        spec = self._nodes[name]
        # A copy all the way down: a list this handed out used to be the
        # node's own (MADD-ANO-205).
        out = _detached_config(spec.node.params)
        live = live if isinstance(live, dict) else {}
        snapshot = spec.node.params_pytree()
        for key, value in live.items():
            # Only constructor params can be written back; a derived leaf
            # (a surrogate's ``weights['scale']``) is not a constructor
            # argument and would break reconstruction.  Checkpoints carry
            # those.
            if key not in spec.node.params:
                continue
            # Only overlay a leaf that actually changed: the pytree holds
            # float32 promotions of the constructor floats (0.05 ->
            # 0.05000000074505806), and an uncalibrated constant should
            # serialise exactly as it was given.
            base = snapshot.get(key)
            if base is not None and np.array_equal(np.asarray(base), np.asarray(value)):
                continue
            out[key] = np.asarray(value).tolist()
        return out

    @stability(StabilityLevel.EVOLVING)
    def param_spec_overrides(self) -> dict[str, dict[str, ParamSpec]]:
        """Graph-level overrides set with :meth:`set_param_spec`."""
        return {n: dict(o) for n, o in self._param_spec_overrides.items() if o}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> SimulationNode:
        """The :class:`~maddening.core.node.SimulationNode` registered as ``name``.

        The read-only counterpart of :meth:`add_node`, for callers that
        need the node object itself — its ``static_data``, ``params`` or
        ``boundary_input_spec`` — rather than the graph's view of it; a
        point reference ``{"node": ..., "field": ...}`` of an interface
        mapping resolves through it.  An unknown name is a ``KeyError``.
        """
        if name not in self._nodes:
            raise KeyError(f"unknown node {name!r}; the graph has {sorted(self._nodes)}")
        return self._nodes[name].node

    def add_node(self, node: SimulationNode) -> None:
        """Register a node and initialise its state."""
        # Into the state that is kept: added to a traced one (right after
        # ``jax.grad`` of a loss that stepped the graph), the node's state
        # was lost when the next entry point put the graph back.
        self._recover_from_escaped_tracers()
        if node.name in self._nodes:
            raise ValueError(f"Node '{node.name}' already exists in the graph.")
        refusal = _graph_specs._node_name_refusal(node.name)
        if refusal is not None:
            raise ValueError(refusal)
        try:
            timestep = float(node.delta_t)
        except (TypeError, ValueError):
            timestep = math.nan
        if not (math.isfinite(timestep) and timestep > 0.0):
            # NaN or an infinity made every step a 400 until the node was
            # deleted (the multirate schedule takes an integer ratio of the
            # timesteps), and 0 or a negative value stepped the node not at
            # all, or backwards, with every reply a 200.  No node in the
            # library runs with a timestep that is not a positive number.
            raise ValueError(
                f"Node {node.name!r} has timestep {node.delta_t!r}: a node's "
                "timestep must be a finite number > 0."
            )
        # A hook that names ``params`` where no keyword reaches it
        # (``params=None, /`` or ``*params``) is not a node without params:
        # it would step on its constructor's constants with every write to
        # ``gm.params`` ignored.  It raised ``TypeError`` at its first
        # trace in every release; it is refused here, by name.
        _refuse_params_no_keyword_reaches(node)

        spec = _graph_specs._NodeSpec(
            node=node,
            update_fn=node.update,
            timestep=node.delta_t,
            accepts_params=_graph_specs._update_accepts_params(node),
            flux_accepts_params=_graph_specs._flux_accepts_params(node),
        )
        # Atomic on purpose: build the state *before* committing to either
        # dict.  ``initial_state()`` is a documented, recoverable failure
        # point -- an ``AdaptiveNode`` raises ``AdaptiveNodeBlindnessError``
        # at a Palais trap and the developer guide's recovery is to perturb
        # the parameters and re-add under the same name.  Registering the
        # spec first left ``_nodes[name]`` populated and ``_state[name]``
        # missing: the name was taken for good (``add_node`` raised
        # "already exists", ``remove_node`` raised ``KeyError``),
        # ``compile()`` accepted the graph, ``params["nodes"]`` carried a
        # node that can never run, and ``step()`` died much later with a
        # bare ``KeyError`` inside the compiled step.  A failed ``add_node``
        # must leave the graph exactly as it was.
        state = node.initial_state()
        self._nodes[node.name] = spec
        self._keep_state_for_reports()
        self._state[node.name] = state
        self._dirty = True
        self._notify(EVENT_NODE_ADDED, node.name)

    def add_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
        transform: Optional[Callable] = None,
        additive: bool = False,
        source_units: Optional[str] = None,
        target_units: Optional[str] = None,
        mapping: Optional[Any] = None,
        *,
        geometry: Optional[Any] = None,
    ) -> None:
        """Add a data-dependency edge between two nodes.

        ``mapping`` (a :class:`maddening.core.coupling.mapping.Mapping`)
        transfers the source field onto the target interface before
        ``transform`` is applied; its weights are snapshotted into
        ``params["mappings"][edge.key]`` at compile time and passed as a
        traced input on every step.  Its ``n_source`` must equal the
        source field's size; ``n_target`` must match the target's
        declared ``boundary_input_spec`` shape when that is an array.
        A mapping of a class other than ``StaticLinearMapping`` is a
        ``ValueError`` unless its ``params_pytree()`` is a plain dict
        from identifiers to concrete, finite, floating-point JAX arrays,
        the same on every call (the contract is spelled out in
        :func:`~maddening.core.coupling.mapping_registry.register_mapping`).
        A :class:`~maddening.core.coupling.mapping_spec.MappingSpec` (or
        its dict form) is rebuilt first with :meth:`point_resolver`.

        ``geometry`` (experimental) names the moving geometry a
        geometry-dependent mapping reads: ``("source", field)`` or
        ``("target", field)``, a state field of this edge's own source or
        target node (the dict ``{"anchor": ..., "field": ...}`` is taken
        too).  There is no default anchor.  A source-anchored geometry is
        read at the time level the edge's value is read at; a
        target-anchored one is the field of the state the target's hook
        is called with.  It is required by a mapping whose
        ``needs_geometry`` is true and refused with any other; the
        field's presence, dtype (float32 or float64) and shape (the
        mapping's ``geometry_shape``) are checked by :meth:`validate`.
        An edge without one behaves exactly as it did.

        The *transform* parameter accepts either a callable or a
        string name registered via ``@register_transform``.  String
        names are resolved immediately; a ``KeyError`` is raised if
        the name is not in the registry.

        Parameters
        ----------
        source_units : str or None
            Physical units of the source field (e.g. ``"lattice"``).
            Informational -- used for documentation and validation.
        target_units : str or None
            Physical units after transform (e.g. ``"N"``).
            Checked against the target node's ``expected_units``.

        Raises
        ------
        ValueError
            If a field's name could not be written wherever an edge is: it
            spells ``NaN``, ``Infinity`` or ``-Infinity``, or contains
            ``#``, one of U+0000 to U+001F (but tab, line feed and
            carriage return), a surrogate or U+FFFE / U+FFFF.  A
            target field the target node does not declare is taken (a node
            may read an input it does not declare).
            Also if ``geometry`` is malformed, is given without a
            geometry-dependent mapping, or is missing for one.
        """
        for what, name in (("source", source_field), ("target", target_field)):
            refusal = _graph_specs._field_name_refusal(name)
            if refusal is not None:
                raise ValueError(
                    f"An edge from {source!r} to {target!r}: its {what} field "
                    f"{name!r} is invalid: {refusal}."
                )
        if isinstance(transform, str):
            from maddening.core.transforms import resolve_transform
            transform = resolve_transform(transform)
        if mapping is not None:
            from maddening.core.coupling.mapping_spec import MappingSpec  # noqa: PLC0415
            if isinstance(mapping, (MappingSpec, dict)):
                # A spec (or its dict form, as written by to_dict): rebuild
                # the mapping from this graph's node fields; asset paths
                # are relative to the working directory.
                if isinstance(mapping, dict):
                    mapping = MappingSpec.from_dict(mapping)
                mapping = mapping.build(self.point_resolver())
            self._check_mapping_shapes(source, source_field, target, target_field, mapping)
        geometry = _graph_specs._checked_geometry(
            f"{source}.{source_field}->{target}.{target_field}", geometry, mapping)
        ordinal = 0
        if mapping is not None:
            # Mapping weights live in params["mappings"][edge.key]; a
            # second mapped edge on the same field pair (two additive
            # contributions, say) gets its own slot via the ordinal
            # instead of silently sharing -- and using -- the other's
            # weights.
            base = f"{source}.{source_field}->{target}.{target_field}"
            ordinal = sum(
                1 for e in self._edges
                if e.mapping is not None and e.key.split("#")[0] == base
            )
        edge = EdgeSpec(source, target, source_field, target_field,
                        transform, additive, source_units, target_units,
                        mapping=mapping, ordinal=ordinal, geometry=geometry)
        self._edges.append(edge)
        self._dirty = True
        self._notify(EVENT_EDGE_ADDED, edge)

    def point_resolver(self, base_dir=None) -> Callable[[dict], Any]:
        """``resolve_points`` for :meth:`MappingSpec.build`: node-field
        references are looked up in this graph's nodes, asset paths are
        relative to ``base_dir`` (the working directory when ``None``)."""
        from maddening.core.coupling.mapping_spec import make_point_resolver  # noqa: PLC0415
        return make_point_resolver(self, base_dir)

    def _check_mapping_shapes(self, source, source_field, target, target_field, mapping):
        for attr in ("apply", "params_pytree", "n_source", "n_target"):
            if not hasattr(mapping, attr):
                raise TypeError(
                    f"mapping must implement the Mapping protocol (missing {attr!r})"
                )
        # What the mapping puts into params["mappings"][edge.key].  Every
        # reader of that entry -- checkpoints, POST /checkpoint/load, sysid,
        # the FMU archive, to_dict's "live weights differ" warning -- takes
        # it for a flat table of floating-point arrays and none of them
        # checks, so a mapping class of the caller's own is asked here,
        # where the edge is added, before anything can read it.
        from maddening.core.coupling.mapping import (  # noqa: PLC0415
            _params_contract_problem,
        )
        problem = _params_contract_problem(mapping)
        if problem is not None:
            raise ValueError(
                f"mapping {mapping!r} on {source}.{source_field} -> "
                f"{target}.{target_field}: {problem}"
            )
        leads = _graph_specs._mapping_field_leads(mapping)
        src_spec = self._nodes.get(source)
        if src_spec is not None:
            src_state = src_spec.node.initial_state()
            if source_field in src_state and leads is not None:
                have = tuple(np.shape(src_state[source_field])[:len(leads[0])])
                if have != leads[0]:
                    raise ValueError(
                        f"mapping {mapping!r} reads a field whose leading axes are "
                        f"{leads[0]}, which does not match {source}.{source_field} "
                        f"(shape {tuple(np.shape(src_state[source_field]))})"
                    )
            elif source_field in src_state:
                n = int(np.prod(np.shape(src_state[source_field])[:1] or (1,)))
                if mapping.n_source != n:
                    raise ValueError(
                        f"mapping n_source={mapping.n_source} does not match "
                        f"{source}.{source_field} (size {n} along axis 0)"
                    )
        tgt_spec = self._nodes.get(target)
        if tgt_spec is not None:
            bspec = tgt_spec.node.boundary_input_spec().get(target_field)
            shape: tuple[Any, ...] = (
                tuple(getattr(bspec, "shape", ()) or ()) if bspec is not None else ())
            if shape and leads is not None:
                if tuple(int(n) for n in shape[:len(leads[1])]) != leads[1]:
                    raise ValueError(
                        f"mapping {mapping!r} delivers a field whose leading axes are "
                        f"{leads[1]}, which does not match {target}.{target_field} "
                        f"declared shape {shape}"
                    )
            elif shape and mapping.n_target != int(shape[0]):
                raise ValueError(
                    f"mapping n_target={mapping.n_target} does not match "
                    f"{target}.{target_field} declared shape {shape}"
                )

    @property
    def edges(self) -> list[EdgeSpec]:
        return list(self._edges)

    def resolve_boundary_inputs(self, node_name: str, params: Optional[dict] = None) -> dict:
        """Boundary inputs ``node_name`` would receive from the *current*
        state: every incoming edge (mapping, transform, additive) plus the
        zero defaults of its external inputs, which replace an edge into
        the same field as they do in the step.  A debugging / inspection
        helper; the compiled step resolves edges itself."""
        self._recover_from_escaped_tracers()
        return self._boundary_inputs_from(self._state, node_name, params)

    def _boundary_inputs_from(self, state, node_name: str, params: Optional[dict] = None,
                              *, fluxes: Optional[dict] = None) -> dict:
        """:meth:`resolve_boundary_inputs` of ``node_name`` from *state*
        rather than from the graph's own.

        The one place outside the compiled step that turns edges into a
        node's boundary inputs: each incoming edge through
        :func:`_apply_edge` (the mapping, then the transform, with the
        weights in ``params["mappings"]``), additive edges summed, and
        then the zero default of each external input -- which, as in the
        step, *replaces* whatever edges delivered to the same field.
        Every reader of the edges that is not the step --
        :meth:`resolve_boundary_inputs`, the conservation diagnostic, the
        surrogate dataset generator -- goes through it, so none of them
        can apply an edge differently from the others.  It is traceable
        in *state* (``jax.vmap`` over a state history gives the inputs at
        every sample).

        An edge whose source field is not in *state* reads a flux output:
        it is looked up in ``fluxes[source_node]`` when *fluxes* is given,
        and is a ``KeyError`` otherwise.
        """
        if node_name not in self._nodes:
            raise KeyError(f"unknown node {node_name!r}")
        p = self._params_or_default(params)
        resolved = _graph_specs._ResolvedParams(p.get("nodes", {}), p.get("mappings", {}))
        out: dict[str, Any] = {}
        for edge in self._edges:
            if edge.target_node != node_name:
                continue
            src_fields = state[edge.source_node]
            if edge.source_field not in src_fields and fluxes is not None \
                    and edge.source_field in fluxes.get(edge.source_node, {}):
                value = fluxes[edge.source_node][edge.source_field]
            else:
                value = src_fields[edge.source_field]
            value = _graph_specs._apply_edge(
                edge, value, resolved,
                _graph_specs._edge_geom(edge, state, lambda: state[node_name]))
            if edge.additive and edge.target_field in out:
                out[edge.target_field] = out[edge.target_field] + value
            else:
                out[edge.target_field] = value
        for ei in self._external_inputs:
            # No ``not in out`` guard: the step writes an external input
            # over the edges into its field (``_resolve_and_update_node``),
            # and an omitted external input is zeros.
            if ei.target_node == node_name:
                out[ei.target_field] = jnp.zeros(ei.shape, dtype=ei.dtype)
        return out

    def add_external_input(
        self,
        target_node: str,
        target_field: str,
        shape: tuple = (),
        dtype: Any = jnp.float32,
    ) -> None:
        """Declare an external input that will be injected each step.

        External inputs appear in the target node's ``boundary_inputs``
        dict alongside edge-delivered values.  They are supplied via the
        ``external_inputs`` argument to :meth:`step` or :meth:`run`.

        Parameters
        ----------
        target_node : str
            Name of the node that receives this input.
        target_field : str
            Key in the node's ``boundary_inputs`` dict.
        shape : tuple
            Array shape (default ``()`` for scalar).
        dtype
            JAX dtype (default ``jnp.float32``, also under
            ``jax_enable_x64``: pass ``jnp.float64`` for a float64 input).
            It is the dtype the input runs in.  A value handed to
            :meth:`step`, :meth:`run_scan` or any other entry point is cast
            to it at the step boundary, and the FMU export and the REST
            server give the input the same type, so every entry point runs
            the same number.  A value the cast changes (a float64 ``0.1``
            into a float32 input, a fraction into an integer one) is
            reported with a ``UserWarning``, once per input.
        """
        refusal = _graph_specs._field_name_refusal(target_field)
        if refusal is not None:
            # The rule of an edge's fields (see add_edge): an external
            # input's is written to the config beside them.
            raise ValueError(
                f"An external input of {target_node!r}: its field "
                f"{target_field!r} is invalid: {refusal}."
            )
        spec = ExternalInputSpec(target_node, target_field, shape, dtype)
        self._external_inputs.append(spec)
        self._dirty = True

    def remove_node(self, name: str) -> None:
        """Remove a node and everything of the graph that names it.

        * Every edge to or from the node, and every external input into
          it, is removed with it (with a mapped edge's live weights and
          :class:`~maddening.core.params.ParamSpec` overrides), as are the
          node's own live parameters and overrides.
        * A coupling group loses the node as a member and keeps its
          options; an ``accelerated_fields`` entry for the node goes with
          it, and if that leaves the option selecting no field it becomes
          ``None`` (the interface fields, the default).  A group left with
          fewer than two members is removed, options and all: one node is
          not iterated against itself.  Until 0.4.0 the group went on
          naming the removed node -- :meth:`validate`, :meth:`compile` and
          every step then failed ("coupling group references non-existent
          node") and the graph's own :meth:`to_dict` did not load, until a
          node of that name was added (``MADD-ANO-214``).  A
          ``UserWarning`` names each group that changes: to rebuild a
          member under its name and keep the group, add the group again
          (:meth:`remove_coupling_group`, :meth:`add_coupling_group`) after
          the node and its edges -- adding the node and its edges back
          used to be enough, the group having gone on naming it.
        * Refused (``ValueError``, nothing removed) when an interface
          mapping on an edge between two *other* nodes was built from a
          ``{"node": name, "field": ...}`` point reference: that edge
          would stay, with a reference :meth:`to_dict` could no longer
          resolve.  Remove the edge first.

        Raises
        ------
        KeyError
            The graph has no node of that name.
        ValueError
            A mapping on an edge that would remain references the node's
            points.
        """
        # From the state that is kept (see ``add_node``); here, so that its
        # warning names the caller as it did.
        self._recover_from_escaped_tracers()
        for message in self._remove_node(name, replacing=False):
            warnings.warn(message, UserWarning, stacklevel=2)

    def _remove_node(self, name: str, *, replacing: bool) -> list[str]:
        """:meth:`remove_node`, returning what it would warn about (one
        sentence for each coupling group that changed) instead of warning.
        With ``replacing``, for a caller that adds a node of the same name
        back before anything else reads the graph
        (``surrogates.replace.replace_node``, which restores the edges
        itself): the coupling groups and the point references of other
        edges' mappings go on naming it."""
        notes: list[str] = []
        # From the state that is kept (see ``add_node``).
        self._recover_from_escaped_tracers()
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        if not replacing:
            kept = {e.key for e in self._edges
                    if e.source_node != name and e.target_node != name}
            held = sorted({
                f"{edge_key} ({arg}: {name}.{field_name})"
                for edge_key, arg, field_name in self._mapping_node_references(name)
                if edge_key in kept})
            if held:
                raise ValueError(
                    f"Cannot remove node '{name}': the interface mapping on edge "
                    f"{', '.join(held)} was built from its points, and that edge "
                    "would remain with a reference to_dict() could not resolve.  "
                    "Remove the edge first (remove_edge).")
            # Built before anything is removed: CouplingGroup's own
            # validation can refuse, and must leave the graph whole.
            groups = []
            for group in self._coupling_groups:
                smaller = _group_layout._group_without_member(group, name)
                if smaller is not None:
                    groups.append(smaller)
                if smaller is not group:
                    notes.append(
                        f"Removing node '{name}' "
                        + ("removed the coupling group of "
                           f"{sorted(group.nodes)} with it (one member would remain)"
                           if smaller is None else
                           f"took it out of the coupling group of {sorted(group.nodes)}, "
                           f"which keeps its options over {sorted(smaller.nodes)}")
                        + ".  A node added back under the name is not a member: add "
                          "the group again (add_coupling_group) after the node and "
                          "its edges.")
        del self._nodes[name]
        if not replacing:
            self._coupling_groups[:] = groups
        # ``pop`` rather than ``del``: a graph whose state entry is missing
        # must still be removable, so the removal cannot itself fail
        # half-way and leave ``_nodes`` and ``_state`` disagreeing.
        self._keep_state_for_reports()
        self._state.pop(name, None)
        self._edges = [
            e for e in self._edges
            if e.source_node != name and e.target_node != name
        ]
        self._external_inputs = [
            e for e in self._external_inputs if e.target_node != name
        ]
        # ParamSpec overrides for the node and for mapped edges that
        # touched it would otherwise survive and break to_dict/from_dict;
        # its live params entry is discarded on purpose (an intentional
        # removal / replacement must not warn at the next compile).
        self._param_spec_overrides.pop(name, None)
        for key in list(self._param_spec_overrides):
            if key.startswith(f"{name}.") or f"->{name}." in key:
                self._param_spec_overrides.pop(key, None)
        self.params.get("nodes", {}).pop(name, None)
        for key in list(self.params.get("mappings", {})):
            if key.startswith(f"{name}.") or f"->{name}." in key:
                self.params["mappings"].pop(key, None)
        self._dirty = True
        self._notify(EVENT_NODE_REMOVED, name)
        return notes

    def remove_edge(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: str,
    ) -> None:
        """Remove a specific edge."""
        edge = EdgeSpec(source, target, source_field, target_field)
        self._edges = [
            e for e in self._edges
            if not (
                e.source_node == edge.source_node
                and e.target_node == edge.target_node
                and e.source_field == edge.source_field
                and e.target_field == edge.target_field
            )
        ]
        for key in list(self._param_spec_overrides):
            if key.split("#")[0] == edge.key:
                self._param_spec_overrides.pop(key, None)
        for key in list(self.params.get("mappings", {})):
            if key.split("#")[0] == edge.key:
                self.params["mappings"].pop(key, None)
        self._dirty = True
        self._notify(EVENT_EDGE_REMOVED, edge)

    # ------------------------------------------------------------------
    # Coupling groups
    # ------------------------------------------------------------------

    def add_coupling_group(
        self,
        nodes: Sequence[str],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> CouplingGroup:
        """Register an iteratively-coupled group of nodes.

        Within each timestep, the nodes in the group are executed
        repeatedly until convergence or *max_iterations*.
        All edges between nodes in the group use current-iteration
        values rather than staggered (previous-timestep) values.

        Parameters
        ----------
        nodes : sequence of str
            Node names forming the coupling group.  Must all exist in
            the graph and should form (part of) a cycle.
        max_iterations : int
            Maximum iterations per timestep.
        tolerance : float
            Convergence threshold (L2 norm of state change).  Read
            **only** under ``convergence_norm="l2"``; ``"mixed"`` and
            ``"interface"`` carry their tolerances in ``atol`` /
            ``rtol`` and test against a fixed threshold of ``1.0``.
            Setting a knob the chosen norm ignores warns
            (``UserWarning``) rather than turning silently.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`~maddening.core.coupling.CouplingGroup`
            (e.g. ``convergence_norm``, ``acceleration``,
            ``iteration_mode``, ``diagnostics``).

        Returns
        -------
        CouplingGroup
            The created coupling group descriptor.
        """
        new_set = frozenset(nodes)
        # Check for overlap with existing coupling groups
        for existing in self._coupling_groups:
            overlap = new_set & existing.nodes
            if overlap:
                raise ValueError(
                    f"Nodes {overlap} already belong to a coupling group."
                )
        group = self._make_coupling_group(
            nodes, max_iterations, tolerance, **kwargs
        )
        _group_layout._refuse_colliding_group_keys([*self._coupling_groups, group])
        self._coupling_groups.append(group)
        self._dirty = True
        return group

    def _make_coupling_group(
        self,
        nodes: Sequence[str],
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> CouplingGroup:
        """Validate and construct a group without registering it.

        Everything that can refuse a group -- an unknown node name,
        ``CouplingGroup``'s own validation of the knobs -- happens here,
        so a caller replacing several groups at once can build them all
        before it touches the graph.
        """
        for name in nodes:
            if name not in self._nodes:
                raise KeyError(f"No node named '{name}'.")
        return CouplingGroup(
            nodes=frozenset(nodes),
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
        )

    def remove_coupling_group(self, nodes: Sequence[str]) -> None:
        """Remove a coupling group by its node set."""
        target = frozenset(nodes)
        self._coupling_groups = [
            g for g in self._coupling_groups if g.nodes != target
        ]
        self._dirty = True

    def auto_couple(
        self,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        **kwargs,
    ) -> list[CouplingGroup]:
        """Automatically create coupling groups from graph cycles.

        Uses Tarjan's algorithm to find strongly connected components
        and creates a coupling group for each SCC with more than one
        node.  Existing coupling groups are cleared first.

        Parameters
        ----------
        max_iterations : int
            Maximum iterations per timestep.
        tolerance : float
            Convergence threshold (L2 norm of state change).
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`add_coupling_group`.

        Returns
        -------
        list of CouplingGroup
            The created coupling groups.
        """
        # Every group built before any of them is registered.  The old
        # order cleared the groups first and built them one at a time, so
        # a ``kwargs`` the ``CouplingGroup`` constructor refuses -- a
        # misspelled knob -- destroyed the groups the graph already had
        # and put nothing back.  Clearing is itself a change to the
        # compiled step, so the dirty flag is part of the same commit:
        # ``add_coupling_group`` used to be the only thing that set it,
        # and an ``auto_couple`` that found no cycles left the graph
        # describing itself as uncoupled while still running the coupled
        # step it was compiled with.
        sccs = find_strongly_connected_components(
            list(self._nodes.keys()), self._edges
        )
        groups = [
            self._make_coupling_group(scc, max_iterations, tolerance, **kwargs)
            for scc in sccs
        ]
        _group_layout._refuse_colliding_group_keys(groups)
        self._coupling_groups[:] = groups
        self._dirty = True
        return groups

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check graph integrity.  Returns a list of warning/error strings."""
        self._recover_from_escaped_tracers()
        issues: list[str] = []
        node_names = set(self._nodes.keys())

        # Edge endpoint checks
        for e in self._edges:
            if e.source_node not in node_names:
                issues.append(f"ERROR: edge references non-existent source node '{e.source_node}'")
            if e.target_node not in node_names:
                issues.append(f"ERROR: edge references non-existent target node '{e.target_node}'")

            # Field existence (check state fields and flux fields)
            if e.source_node in self._state:
                if e.source_field not in self._state[e.source_node]:
                    # Also check flux fields from compute_boundary_fluxes
                    is_flux_field = False
                    if e.source_node in self._nodes:
                        node_obj = self._nodes[e.source_node].node
                        from maddening.core.node import SimulationNode as _SimBase
                        if type(node_obj).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                            flux_keys = node_obj.compute_boundary_fluxes(
                                self._state[e.source_node], {}, 0.0
                            ).keys()
                            if e.source_field in flux_keys:
                                is_flux_field = True
                    if not is_flux_field:
                        issues.append(
                            f"ERROR: source field '{e.source_field}' not in state of node '{e.source_node}'. "
                            f"Available: {list(self._state[e.source_node].keys())}"
                        )

        # Geometry-dependent mappings (experimental): what each edge's
        # geometry must be.  Nothing for a graph without one.
        issues.extend(_graph_specs._geometry_edge_issues(self._edges, self._nodes, self._state))

        # Edge validation: shape, dtype, units against BoundaryInputSpec
        for e in self._edges:
            if e.target_node not in self._nodes:
                continue
            bi_spec = self._nodes[e.target_node].node.boundary_input_spec()
            if e.target_field not in bi_spec:
                continue
            spec = bi_spec[e.target_field]

            # Shape check: compare when both source and spec shapes are
            # concrete.  ``spec.shape == ()`` means the input is a scalar
            # — non-scalar sources still get flagged.
            source_state = self._state.get(e.source_node, {})
            src_val = source_state.get(e.source_field)
            if src_val is not None:
                src_shape = tuple(int(d) for d in getattr(src_val, "shape", ()))
                spec_shape = tuple(spec.shape)
                leads = (None if e.mapping is None
                         else _graph_specs._mapping_field_leads(e.mapping))
                if leads is not None and src_shape:
                    # A mapping that declares its field shapes replaces
                    # the leading axes it reads by the ones it delivers.
                    src_shape = leads[1] + src_shape[len(leads[0]):]
                elif e.mapping is not None and src_shape:
                    # The mapping changes axis 0 to its n_target; the
                    # rest of the shape (vector components) passes through.
                    src_shape = (int(e.mapping.n_target),) + src_shape[1:]
                # Skip when spec leaves any dimension symbolic (negative
                # convention) or when a transform may reshape on the fly.
                if (e.transform is None
                        and all(d >= 0 for d in spec_shape)
                        and src_shape != spec_shape):
                    issues.append(
                        f"WARNING[shape]: edge "
                        f"{e.source_node}.{e.source_field} -> "
                        f"{e.target_node}.{e.target_field}: "
                        f"source shape {src_shape} disagrees with "
                        f"target BoundaryInputSpec shape {spec_shape} "
                        f"and no transform is set"
                    )

            # Dtype check: only when both source and spec dtypes are set.
            if src_val is not None and spec.dtype is not None:
                src_dtype = getattr(src_val, "dtype", None)
                if src_dtype is not None and e.transform is None and e.mapping is None:
                    if str(src_dtype) != str(jnp.dtype(spec.dtype)):
                        issues.append(
                            f"WARNING[dtype]: edge "
                            f"{e.source_node}.{e.source_field} -> "
                            f"{e.target_node}.{e.target_field}: "
                            f"source dtype {src_dtype} disagrees with "
                            f"target BoundaryInputSpec dtype "
                            f"{jnp.dtype(spec.dtype)} "
                            f"and no transform is set"
                        )

            # Unit checks (existing behaviour, retained).
            if (e.target_units is not None
                    and spec.expected_units is not None
                    and e.target_units != spec.expected_units):
                issues.append(
                    f"WARNING[units]: unit mismatch on edge "
                    f"{e.source_node}.{e.source_field} -> "
                    f"{e.target_node}.{e.target_field}: "
                    f"edge declares target_units='{e.target_units}' "
                    f"but node expects '{spec.expected_units}'"
                )
            if (e.source_units is not None
                    and spec.expected_units is not None
                    and e.source_units != spec.expected_units
                    and e.transform is None):
                issues.append(
                    f"WARNING[units]: edge "
                    f"{e.source_node}.{e.source_field} -> "
                    f"{e.target_node}.{e.target_field} has "
                    f"source_units='{e.source_units}' but target "
                    f"expects '{spec.expected_units}' and no "
                    f"transform is set"
                )

        # External input endpoint checks
        for ei in self._external_inputs:
            if ei.target_node not in node_names:
                issues.append(
                    f"ERROR: external input references non-existent node '{ei.target_node}'"
                )

        # Disconnected-node warning.  Only meaningful when the graph
        # has multiple nodes -- a single-node graph is trivially
        # "disconnected" but the warning is just noise (the quickstart
        # shape).  v0.2.1 gates this behind ``len(node_names) > 1``.
        if len(node_names) > 1:
            connected = set()
            for e in self._edges:
                connected.add(e.source_node)
                connected.add(e.target_node)
            for ei in self._external_inputs:
                connected.add(ei.target_node)
            for n in node_names:
                if n not in connected:
                    issues.append(
                        f"WARNING: node '{n}' is disconnected "
                        "(no edges or external inputs)"
                    )

        # Multi-rate timestep informational message, from the timesteps
        # ``compile()`` schedules: a sub-cycling group whose members differ
        # is one macro rate, not a multi-rate graph.  (The nodes' own
        # timesteps used to be read here, which called such a graph
        # multi-rate and gave the step and dividers no compile uses.)
        scheduled = _graph_specs._scheduled_timesteps(self._nodes, self._coupling_groups)
        if len(set(scheduled.values())) > 1:
            base_dt = _graph_specs._step_duration(scheduled)
            try:
                dividers = _graph_specs._rate_dividers(scheduled)
            except ValueError as exc:
                # What compile() raises: a schedule that would not keep a
                # node's clock used to be reported here as enabled, with a
                # rate divider of 0.
                issues.append(f"ERROR: {exc}")
            else:
                issues.append(
                    f"INFO: multi-rate scheduling enabled. "
                    f"Base timestep: {base_dt}, rate dividers: {dividers}"
                )

        # Coupling group validation
        coupled_nodes: set[str] = set()
        for group in self._coupling_groups:
            for n in group.nodes:
                if n not in node_names:
                    issues.append(
                        f"ERROR: coupling group references non-existent node '{n}'"
                    )
            # Check uniform timestep within coupling group
            # (relaxed when subcycling is enabled)
            group_timesteps = {
                self._nodes[n].timestep
                for n in group.nodes
                if n in self._nodes
            }
            for n in sorted(group.nodes):
                if n in self._nodes:
                    try:
                        _group_layout._declared_evaluations(self._nodes[n].node)
                    except ValueError as exc:
                        issues.append(f"ERROR: {exc}")
            if len(group_timesteps) > 1 and not group.subcycling:
                issues.append(
                    f"ERROR: coupling group {set(group.nodes)} has mixed "
                    f"timesteps {group_timesteps}. All nodes in a coupling "
                    f"group must share the same timestep.  Set "
                    f"subcycling=True to enable mixed-timestep coupling."
                )
            elif len(group_timesteps) > 1:
                issues.extend(_group_layout._subcycling_ratio_errors(group, self._nodes))
            plan = _interface_plan.interface_plan(
                group.nodes, self._edges, sorted(group.nodes), self._state, self._nodes)
            issues.extend(_group_layout._flux_edge_coupling_errors(
                group, self._nodes, plan, self._state,
            ))
            issues.extend(_group_layout._geometry_edge_coupling_errors(group, plan))
            coupled_nodes |= group.nodes
            issues.extend(self._coupling_group_advisories(group))

        # Cycle detection (only on edges with valid endpoints)
        valid_edges = [
            e for e in self._edges
            if e.source_node in node_names and e.target_node in node_names
        ]
        cycles = detect_cycles(list(self._nodes.keys()), valid_edges)
        for cyc in cycles:
            # Check if cycle is covered by a coupling group
            cyc_set = set(cyc)
            covered = any(cyc_set <= g.nodes for g in self._coupling_groups)
            if covered:
                issues.append(
                    f"INFO: cycle {' -> '.join(cyc)} handled by iterative "
                    f"coupling (Gauss-Seidel)."
                )
            else:
                # Uncovered cycles are handled by staggering (back-edges
                # read previous-timestep values) -- not an error and not
                # something the user can usually act on at compile time.
                # v0.2.1 demotes this from a UserWarning to a
                # ``logging.info`` record so it stops bubbling up through
                # downstream ``filterwarnings=["error"]`` configs.  The
                # prefix flip from ``WARNING:`` to ``INFO:`` also takes
                # the message out of compile()'s warning-emission loop.
                msg = (
                    f"cycle detected: {' -> '.join(cyc)}. "
                    "Back-edges will use previous-timestep values "
                    "(staggering)."
                )
                logger.info(msg)
                issues.append(f"INFO: {msg}")

        return issues

    def _coupling_group_advisories(self, group: CouplingGroup) -> list[str]:
        """``WARNING:`` issues a member's class raises about a group as a whole.

        A private convention between the built-in nodes and
        :meth:`validate`, like ``_halo_boundary_hint`` is between them and
        ``ShardedStencilNode``.  It is not yet part of the node contract.
        A node class may define a static ``_coupling_group_advisories``
        taking the keyword context below and returning ``WARNING: ...``
        strings.  :meth:`compile` emits each one as a ``UserWarning``,
        like every other advisory ``validate`` returns.  Each distinct hook
        runs once per group, however many members share it.  Only warnings
        go through here.  A hook never refuses a graph, and it sees plain
        Python data, not the step.

        ``HeatNode`` uses it for two rods coupled end to end past their
        coupled-pair Fourier limit (MADD-ANO-050).
        """
        members = {
            name: self._nodes[name] for name in sorted(group.nodes)
            if name in self._nodes
        }
        hooks: list = []
        for spec in members.values():
            hook = getattr(type(spec.node), "_coupling_group_advisories", None)
            if callable(hook) and not any(hook is h for h in hooks):
                hooks.append(hook)
        if not hooks:
            return []
        # How many values arrive at each (node, input) from anywhere in the
        # graph: a hook reasoning about "this input is that node's value"
        # needs to know nothing else writes to it.
        feeds: dict[tuple[str, str], int] = defaultdict(int)
        for e in self._edges:
            feeds[(e.target_node, e.target_field)] += 1
        for ei in self._external_inputs:
            feeds[(ei.target_node, ei.target_field)] += 1
        context = dict(
            group=group,
            nodes={name: spec.node for name, spec in members.items()},
            timesteps={name: spec.timestep for name, spec in members.items()},
            edges=_interface_plan.internal_edges(self._edges, members),
            feeds=dict(feeds),
            live_params=dict((self.params or {}).get("nodes") or {}),
        )
        out: list[str] = []
        for hook in hooks:
            out.extend(hook(**context))
        return out

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self) -> None:
        """Topologically sort the graph and JIT-compile the step function."""
        # Preserving the state across the rebuild is only safe if the
        # state is usable; a graph still holding a transform's tracers
        # goes back to the state it had before it first.
        self._recover_from_escaped_tracers()
        # A multi-rate schedule that would not keep a node's clock is a
        # ValueError naming the two nodes, before anything else is asked
        # (validate() lists it among its errors too).
        scheduled = _graph_specs._scheduled_timesteps(self._nodes, self._coupling_groups)
        if len(set(scheduled.values())) > 1:
            _graph_specs._rate_dividers(scheduled)
        issues = self.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            raise RuntimeError(
                "Cannot compile graph with errors:\n" + "\n".join(errors)
            )
        # Aggregate every problem so the user sees them all in a single
        # pass.  Since v0.2.1, shape and dtype mismatches are hard
        # errors (pre-announced in v0.2.0 release notes; semver
        # carve-out documented in docs/developer_guide/
        # edge_validation_migration.md).  They are collected here and
        # raised together as an ExceptionGroup.  Unit mismatches stay
        # as warnings; plain advisory "WARNING:" issues stay as
        # UserWarning.
        from maddening.warnings import (
            DtypeMismatchError,
            EdgeValidationError,  # noqa: F401 — exported for callers
            ExceptionGroup,
            ShapeMismatchError,
            UnitMismatchWarning,
        )
        validation_errors: list[EdgeValidationError] = []
        for issue in issues:
            if not issue.startswith("WARNING"):
                continue
            if issue.startswith("WARNING[shape]"):
                validation_errors.append(ShapeMismatchError(issue))
            elif issue.startswith("WARNING[dtype]"):
                validation_errors.append(DtypeMismatchError(issue))
            elif issue.startswith("WARNING[units]"):
                warnings.warn(issue, UnitMismatchWarning, stacklevel=2)
            else:
                warnings.warn(issue, stacklevel=2)
        if validation_errors:
            raise ExceptionGroup(
                "edge validation failed", validation_errors
            )

        # Everything from here to the commit point at the end of the
        # method is computed into locals.  The ``accelerated_fields``
        # validation, the static-data refusal and ``_build_step_fn`` can
        # all still raise, and a compile that fails must leave the graph
        # bit-identical to what it was -- not describing a step that was
        # never built.  See ``_StepPlan``.
        node_names = list(self._nodes.keys())
        schedule = _group_layout._block_schedule(
            topological_sort(node_names, self._edges), self._coupling_groups)
        back_edges = identify_back_edges(schedule, self._edges)
        for warning_text in _group_layout._loop_through_outside_nodes(
                schedule, self._edges, self._coupling_groups, back_edges):
            warnings.warn(warning_text, UserWarning, stacklevel=2)
        for warning_text in _group_layout._staggered_across_components(
                schedule, self._edges, self._coupling_groups, back_edges):
            warnings.warn(warning_text, UserWarning, stacklevel=2)
        # Each group's edges, described once (``_interface_plan``): what the
        # seeding of its slots below and its reports rest on.
        interface_plans = {
            "+".join(sorted(g.nodes)): _interface_plan.interface_plan(
                g.nodes, self._edges, [n for n in schedule if n in g.nodes],
                self._state, self._nodes)
            for g in self._coupling_groups
        }

        # Explicit accelerated_fields must name state fields of the group's
        # nodes (a boundary flux is not a state field; use the default,
        # which maps a flux edge to the producer's state fields).
        #
        # Before *everything* that reads the field, and in particular
        # before the ``iqn-imvj`` ``_meta`` seeding below, which calls
        # ``flatten_coupled_state(..., fields=...)`` with the user's list
        # and dies on an unknown field with a bare ``KeyError: 'typo'``.
        # That shadowed this message under the one acceleration in which
        # ``accelerated_fields`` is most used, while it fired cleanly
        # under ``acceleration="none"``, where ``CouplingGroup`` already
        # warns that the field is ignored altogether.  The block reads
        # only ``self._coupling_groups``, ``self._nodes`` and
        # ``self._state``, all of which are final here.
        for g in self._coupling_groups:
            if g.accelerated_fields is None:
                continue
            for nn, fields in g.accelerated_fields.items():
                if nn not in self._nodes or nn not in g.nodes:
                    raise ValueError(
                        f"accelerated_fields names node {nn!r}, not in coupling "
                        f"group {sorted(g.nodes)}"
                    )
                have = set(self._state.get(nn, {}).keys())
                bad = [f for f in fields if f not in have]
                if bad:
                    raise ValueError(
                        f"accelerated_fields[{nn!r}] names {bad}: not a state field "
                        f"of {nn!r} (state fields: {sorted(have)})"
                    )
            # Only floating fields are accelerated: an integer, boolean or
            # PRNG-key field is computed by every pass and cannot be
            # relaxed, and relaxing it rounded it through float32
            # (``_floating_accel_fields``).  Such a field is dropped from
            # the selection; a selection with nothing else in it would
            # leave the quasi-Newton problem empty, so it is refused here
            # rather than as a shape error inside the traced loop.
            if g.acceleration in ("iqn-ils", "iqn-imvj"):
                floating = float_fields_of(
                    self._state, [nn for nn in g.accelerated_fields if nn in g.nodes],
                )
                if not any(f in floating[nn]
                           for nn, fields in g.accelerated_fields.items()
                           for f in fields):
                    raise ValueError(
                        f"accelerated_fields={dict(g.accelerated_fields)!r} of coupling "
                        f"group {sorted(g.nodes)} names no floating-point field.  Only "
                        "floating fields are accelerated: an integer, boolean or "
                        "PRNG-key field is computed by every pass and cannot be "
                        "relaxed.  Name at least one floating field, or leave "
                        "accelerated_fields=None to use the interface fields."
                    )

        # ``_meta`` is *state*, not derived data: ``step_count`` decides
        # which sub-steps a node with a rate divider > 1 fires on, and the
        # ``coupling_*`` entries are the predictor history and the IQN
        # warm start.  A recompile preserves node state and ``params``; it
        # has to preserve these for the same reason.  A mid-run structural
        # edit (adding an edge or an external input, the profiler or the
        # REST server recompiling behind your back) used to replace this
        # dict, silently re-phasing a multi-rate schedule and restarting
        # every warm start.  Snapshotted here, *before* the rate dividers
        # are recomputed, and re-applied below over the key set the new
        # graph expects.  ``reset_state()`` remains the explicit way to
        # zero the counters.
        previous_meta = dict(self._state.get(_graph_specs._META_KEY, {}))
        # From the last *successful* compile, not from ``_rate_dividers``:
        # a compile that raises after recomputing them (the static-data
        # refusal, a failing ``_build_step_fn``) leaves them describing a
        # step that was never built, and comparing against those would
        # restart the phase on the repair.  (The ``accelerated_fields``
        # typo used to be one of those; it is now refused above, before
        # the dividers are touched at all.)
        previous_dividers = dict(self._committed_rate_dividers)

        # Compute multi-rate info.
        # For nodes in subcycling coupling groups, use the group's
        # macro timestep (max of member timesteps) for rate divider
        # computation, since the coupling block handles sub-stepping.
        # ``self.timestep`` reads the same two helpers, so the step it
        # reports is the one these dividers count in.
        effective_timesteps = _graph_specs._scheduled_timesteps(
            self._nodes, self._coupling_groups,
        )

        timesteps = sorted(set(effective_timesteps.values()))
        if len(timesteps) > 1:
            is_multirate = True
            base_dt = _graph_specs._step_duration(effective_timesteps)
            # Refuses (ValueError) a schedule that would not keep a node's
            # clock; the dividers of one it keeps are what they were.
            rate_dividers = _graph_specs._rate_dividers(effective_timesteps)
        else:
            is_multirate = False
            rate_dividers = {name: 1 for name in self._nodes}

        # Build ``_meta`` fresh over the key set *this* graph needs, so a
        # key whose owning coupling group is gone cannot linger in the
        # scan carry, then carry the previous values back over it below.
        meta: dict = {}
        if is_multirate:
            meta["step_count"] = jnp.array(0, dtype=jnp.int32)

        # Ensure _meta exists with correct structure when coupling
        # diagnostics are enabled.  Pre-populate diagnostic keys so
        # the pytree structure is stable across lax.scan iterations.
        has_diagnostics = any(
            g.diagnostics or g.solver == "ift" for g in self._coupling_groups
        )
        has_imvj = any(
            g.acceleration == "iqn-imvj" for g in self._coupling_groups
        )
        has_predictor = any(
            g.predictor != "none" for g in self._coupling_groups
        )
        if has_diagnostics or has_imvj or has_predictor:
            for g in self._coupling_groups:
                key = "+".join(sorted(g.nodes))
                if g.diagnostics or g.solver == "ift":
                    meta[f"coupling_{key}_iterations"] = jnp.array(
                        0, dtype=jnp.int32
                    )
                    if _group_layout._group_waveform_sweeps(g, self._nodes) > 1:
                        # Passes summed over the step's waveform sweeps;
                        # a group running one sweep has no such slot and
                        # reports ``iterations`` for it.  Same condition
                        # as the write in ``_run_coupled_block_impl``.
                        meta[f"coupling_{key}_total_iterations"] = jnp.array(
                            0, dtype=jnp.int32
                        )
                    # Seed in the dtype the residual is computed in (the
                    # group's floating state), so a float64 graph under
                    # x64 keeps a stable scan carry / trace signature.
                    res_dtype = _group_layout._group_residual_dtype(self._state, g.nodes)
                    meta[f"coupling_{key}_residual"] = jnp.array(0.0, dtype=res_dtype)
                    # The amplification 1/(1-rho) the error bound is
                    # built from; 0.0 reads as "no usable estimate".
                    meta[f"coupling_{key}_amplification"] = jnp.array(
                        0.0, dtype=res_dtype
                    )
                    if _group_layout._reads_mapping_weights(g, interface_plans[key]):
                        # The residual's float floor per evaluation, which
                        # the step measures where the interface norm reads
                        # an edge through its mapping (the delivered value
                        # depends on the weights the step ran with; an edge
                        # read at its source does not); NaN reads as "not
                        # measured".  Same condition as the write in
                        # ``_run_coupled_block_impl``.
                        meta[f"coupling_{key}_reading_floor"] = jnp.array(
                            jnp.nan, dtype=res_dtype
                        )
                    if g.diagnostics and g.solver == "ift":
                        # The Arnoldi triple (Ritz radius, residual,
                        # resolvent norm) behind the spectral bound; NaN
                        # reads as "not computed".  Same
                        # condition as the write in
                        # ``_run_coupled_block_impl``, and the same dtype:
                        # the analysis's, at least float32, so a 16-bit
                        # group's report is not rounded to its fields'
                        # resolution (CPL-087).
                        spec_dtype = _bounds._analysis_dtype(res_dtype)
                        meta[f"coupling_{key}_rho_spectral"] = jnp.array(
                            jnp.nan, dtype=spec_dtype
                        )
                        meta[f"coupling_{key}_spectral_residual"] = jnp.array(
                            jnp.nan, dtype=spec_dtype
                        )
                        meta[f"coupling_{key}_spectral_amplification"] = jnp.array(
                            jnp.nan, dtype=spec_dtype
                        )
                        # The bound on the IFT gradient's relative
                        # error, built on the triple; NaN reads as
                        # "not computed" too.
                        meta[f"coupling_{key}_gradient_relative_error_bound"] = jnp.array(
                            jnp.nan, dtype=spec_dtype
                        )
                        # The evaluations the pass rounds like, measured
                        # at the step's state with each same-pass read
                        # gain-weighted; NaN reads as "not measured" and
                        # the report falls back to the structural count.
                        meta[f"coupling_{key}_pass_evaluations"] = jnp.array(
                            jnp.nan, dtype=spec_dtype
                        )
                        if (interface_plans[key].resolved_geometry_edges()
                                and _group_layout._geometry_diagnostics_refusal(
                                    g, self._nodes, interface_plans[key]) is None):
                            # The self-check of the pass's product along
                            # a geometry (experimental): written by the
                            # step under the same condition.
                            meta[f"coupling_{key}_geometry_gap"] = jnp.array(
                                jnp.nan, dtype=spec_dtype
                            )
                if g.acceleration == "iqn-imvj":
                    # Pre-populate V/W matrices for IQN-IMVJ
                    from maddening.core.coupling.acceleration import (
                        flatten_coupled_state,
                    )
                    # The same field set the step flattens (floating
                    # fields only), or the warm start's length would not
                    # match the vector it seeds.
                    af = _group_layout._group_accel_fields(g, interface_plans[key], self._state)
                    flat0 = flatten_coupled_state(
                        self._state, list(g.nodes), fields=af
                    )
                    n_dof = flat0.shape[0]
                    max_cols = max(g.max_iterations - 1, 1)
                    # In the dtype the step iterates the secant matrices
                    # in (``acc_dtype`` in ``_run_coupled_block_impl``),
                    # not at canonical precision: float64 under x64 beside
                    # a float32 group was a scan-carry dtype mismatch.
                    vw_dtype = _bounds._analysis_dtype(flat0.dtype)
                    meta[f"coupling_{key}_V"] = jnp.zeros(
                        (n_dof, max_cols), vw_dtype
                    )
                    meta[f"coupling_{key}_W"] = jnp.zeros(
                        (n_dof, max_cols), vw_dtype
                    )
                if g.predictor != "none":
                    # Pre-populate predictor history with flattened
                    # node states.  Use flatten_coupled_state with
                    # all fields (no acceleration field filtering).
                    from maddening.core.coupling.acceleration import (
                        flatten_coupled_state as _fcs_pred,
                    )
                    # In the order the step flattens it (the group's
                    # schedule), not ``frozenset`` order, which follows
                    # the per-process string hash: the seed was written
                    # in a different order from one run to the next.
                    group_names_pred = [n for n in schedule if n in g.nodes]
                    flat0 = _fcs_pred(
                        self._state, group_names_pred,
                        fields=float_fields_of(self._state, group_names_pred),
                    )
                    n_pred = 3 if g.predictor == "quadratic" else 2
                    for pi in range(n_pred):
                        meta[f"coupling_{key}_pred_{pi}"] = flat0
                    # Counter for how many converged states have
                    # been stored (0 at start, up to n_pred).
                    meta[f"coupling_{key}_pred_count"] = jnp.array(
                        0, dtype=jnp.int32
                    )

        # Carry the live ``_meta`` back over the seeds, key by key.  Only
        # keys the new graph expects are kept (a group that was removed
        # takes its diagnostics and warm start with it), and only when the
        # live value still fits the seed's shape and dtype -- a group
        # whose interface DOF count changed gets a fresh, correctly shaped
        # warm start rather than a crash inside ``lax.scan``.
        # ``step_count`` restarts only when a divider moved, because the
        # sub-step it indexes is then not the sub-step it indexed before.
        # Judged over the nodes that survived the edit only: adding or
        # removing a node must not re-phase the ones already running,
        # which is the whole point of preserving the counter.
        phase_still_means_the_same = all(
            previous_dividers[name] == divider
            for name, divider in rate_dividers.items()
            if name in previous_dividers
        )
        # A group replaced by a different one over the same nodes (a
        # tightened tolerance, another norm) keeps its slot names, but the
        # report slots describe a step the new group did not judge:
        # carried over, ``coupling_diagnostics()`` re-derived ``converged``
        # from the old residual under the new criterion, and read
        # ``converged=False`` three passes into a fifty-pass budget.  They
        # restart at their seeds instead -- no report until the new group
        # has stepped, as after ``reset_state()`` -- while its warm starts
        # (IQN matrices, predictor history), which are state, carry on.
        stale_report_slots = {
            f"coupling_{gkey}_{suffix}"
            for g in self._coupling_groups
            for gkey in ("+".join(sorted(g.nodes)),)
            if gkey in self._committed_coupling_groups
            and self._committed_coupling_groups[gkey] != g
            for suffix in _reports._REPORT_SLOT_SUFFIXES
        }
        for key_, seed in meta.items():
            if key_ == "step_count" and not phase_still_means_the_same:
                continue
            if key_ in stale_report_slots:
                continue
            live = previous_meta.get(key_)
            if live is None:
                continue
            live = jnp.asarray(live)
            if live.shape == jnp.shape(seed) and live.dtype == jnp.asarray(seed).dtype:
                meta[key_] = live
        # Computed here, committed at the very end of ``compile`` -- the
        # validation below, the static-data refusal and ``_build_step_fn``
        # can all still raise, and a compile that fails must leave the
        # sub-step phase and the warm starts exactly as it found them.

        # ``subcycling=True`` on a group whose nodes all share a
        # timestep is demoted to ``use_subcycling = False`` in
        # ``_run_coupled_block_impl``, which leaves
        # ``waveform_iterations`` and ``boundary_interpolation`` dead
        # while ``CouplingGroup``'s own ``subcycling`` predicate says
        # they are live.  The group cannot see that -- it does not know
        # its members' timesteps -- so the rule is enforced here, where
        # the nodes are known and the first step has not run yet.
        from maddening.core.coupling.group import (
            _FIELD_DEFAULTS,
            inert_uniform_timestep_message,
        )
        _SUBCYCLED_ONLY = ("waveform_iterations", "boundary_interpolation")
        for g in self._coupling_groups:
            if not g.subcycling:
                continue        # the CouplingGroup rule already covers it
            timesteps = {self._nodes[nn].timestep for nn in g.nodes
                         if nn in self._nodes}
            if len(timesteps) > 1:
                continue        # genuinely subcycled: the knobs are live
            named = tuple(
                name for name in _SUBCYCLED_ONLY
                if getattr(g, name) != _FIELD_DEFAULTS[name]
            )
            if named:
                warnings.warn(
                    inert_uniform_timestep_message(g, named),
                    UserWarning,
                    stacklevel=2,
                )

        # Persistent XLA cache, if the user asked for one via the env var
        # (see maddening.core.simulation.compile_cache).
        from maddening.core.simulation.compile_cache import enable_from_env
        enable_from_env()

        # A weak-typed leaf in the seed state would retrace the jitted
        # step once it comes back strongly typed after the first step.
        state = _graph_specs._strong_typed(self._state)
        # Zero external inputs are allocated once per compile, not per
        # step (``jnp.zeros`` per input per call cost ~1.5 ms/step on GPU).
        default_ext_leaves = {
            (ei.target_node, ei.target_field): jnp.zeros(ei.shape, dtype=ei.dtype)
            for ei in self._external_inputs
        }

        # Snapshot the differentiable parameters before building the
        # step so the closure default (``params=None``) is this snapshot.
        # Live values survive a recompile: a calibrated leaf whose
        # node/key/shape/dtype still exist is carried over (adding an
        # edge or an external input must not discard a fit), unless only
        # the node's value was written since the last compile, which then
        # wins (see ``_merge_live_params``); anything that no longer fits
        # is dropped with a warning.  ``reset_params`` restores the
        # constructor values on purpose.
        # Every node.params write first goes into gm.params (or marks the
        # graph dirty, which this compile is about to clear), so the merge
        # below can let the live value win.
        self._sync_node_param_writes()
        fresh = self._snapshot_params()
        # Which leaves are still the constructor snapshot after the merge:
        # those need no baked-write check (see _refuse_baked_param_writes).
        fresh_leaves = {o: dict(v) for o, v in fresh.get("nodes", {}).items()}
        params = self._merge_live_params(fresh, self._params)
        params_dtypes = {
            section: {
                owner: {k: jnp.asarray(v).dtype for k, v in leaves.items()}
                for owner, leaves in params.get(section, {}).items()
            }
            for section in ("nodes", "mappings")
        }
        params_shapes = {
            section: {
                owner: {k: tuple(jnp.shape(v)) for k, v in leaves.items()}
                for owner, leaves in params.get(section, {}).items()
            }
            for section in ("nodes", "mappings")
        }
        baked = self.nodes_without_params()
        if baked:
            logger.info(
                "nodes without a params keyword (constants baked, not "
                "differentiable through the graph): %s", baked,
            )

        # D10 step 3: a static derived from a *trainable* parameter is
        # refused outright.  The static is baked into the HLO as a
        # constant while the parameter is traced, so the gradient would
        # be missing the term through the static -- silently, and in the
        # direction an optimiser is pushing.  No rebuild hook can fix
        # that, so the graph does not compile.
        #
        # Declared, not inferred: ``compile`` cannot see which values a
        # traced closure reads, so ``static_data_deps`` is the node's own
        # statement of provenance.  The walk reaches wrapped nodes, each
        # resolved against its own specs, so a wrapper cannot hide one.
        # Placed in the same region as the invalidation below: before
        # ``_build_step_fn`` and before the static-data hash snapshot.
        # Resolved against the *merged* specs -- the node's own with this
        # graph's ``set_param_spec`` overrides applied -- because that is
        # the view ``trainable_mask``, ``unconstrain``, ``check_params``
        # and ``maddening.sysid`` optimise against.  Reading the node
        # alone made the rule disagree with the optimiser both ways: a
        # graph-level unfreeze walked past the refusal into a silently
        # wrong gradient, and a graph-level freeze -- the first remedy the
        # message below names -- did not clear it.
        from maddening.core.node import static_data_dep_violations
        for name, spec in self._nodes.items():
            for owner, static_key, param_key in static_data_dep_violations(
                spec.node, self._param_spec_overrides.get(name)
            ):
                where = (
                    f"node {name!r}" if owner == name
                    else f"node {name!r} (declared by the wrapped node {owner!r})"
                )
                raise ValueError(
                    f"{where} declares static_data[{static_key!r}] as derived "
                    f"from parameter {param_key!r}, which is trainable.  "
                    f"static_data is baked into the compiled HLO as a "
                    f"constant, and you cannot differentiate through a "
                    f"constant: the gradient with respect to {param_key!r} "
                    f"would silently omit the term through "
                    f"{static_key!r}, so a fit would move {param_key!r} "
                    f"while {static_key!r} stayed at its __init__ value.  "
                    f"Either declare {param_key!r} as "
                    f"ParamSpec(trainable=False), or stop deriving "
                    f"{static_key!r} from it and compute the quantity "
                    f"inside update() from the traced parameter instead."
                )

        # A node may keep its own materialised copy of its static arrays
        # (the sharded wrappers cache the per-device placement, keyed on
        # the arrays' identity).  Such a key cannot see a static whose
        # buffer was rewritten in place, and the cache lives on the node,
        # so without this the step just rebuilt would be traced against
        # the previous buffer.  ``compile()`` is the framework's explicit
        # "rebuild everything", so it has to reach those caches too; it
        # runs rarely, and the cost is one re-materialisation per sharded
        # static per compile, paid lazily on the next trace.
        #
        # ``invalidate_static_cache`` is a ``SimulationNode`` contract
        # method whose default forwards to any node this one wraps, so a
        # cache nested inside a wrapper (a sharded node inside a
        # HybridNode) is reached too.  The getattr probe stays for the
        # duck-typed node objects the graph also accepts.
        #
        # Ordered before the build rather than after it.  Both work today
        # only because ``_build_step_fn`` and ``jax.jit`` are lazy and
        # materialise nothing; clearing first is correct whether or not
        # that stays true, and it still precedes the static-data hash
        # snapshot below, which is the other ordering constraint.
        for spec in self._nodes.values():
            invalidate = getattr(spec.node, "invalidate_static_cache", None)
            if callable(invalidate):
                invalidate()

        # Built against the plan, not against the graph: the build is the
        # last thing that can raise, and it must be able to fail without
        # having moved the graph off the step it is running.
        plan = _graph_specs._StepPlan(
            schedule=schedule,
            back_edges=back_edges,
            is_multirate=is_multirate,
            rate_dividers=rate_dividers,
            params=params,
        )
        step_fn = self._build_step_fn(plan)

        def _counted_step(full_state, external_inputs, params=None):
            self._n_traces += 1
            return step_fn(full_state, external_inputs, params)

        compiled_step = jax.jit(_counted_step)
        # Which graph, and which of its compiles, this step is.  The step
        # outlives the graph's next compile wherever it was handed out
        # (``SidecarConfig(step_fn=gm._compiled_step)``), and it bakes in
        # every structural value it reads when it is traced, so the FMU
        # sidecar and bridge refuse it once the graph it came from has
        # changed or been compiled again
        # (``maddening.fmi.model_description._graph_changed_since``).
        setattr(compiled_step, "_maddening_compile",  # noqa: B010 - not a typed attribute
                (weakref.ref(self), self._compile_generation + 1))

        # Snapshot static_data hashes so we can detect drift.
        # ``static_data_hash`` is a node-supplied method, so this is the
        # last thing in the method that can raise -- it stays above the
        # commit point.
        static_data_hashes = {
            name: spec.node.static_data_hash()
            for name, spec in self._nodes.items()
        }

        # ------------------------------------------------------------------
        # Commit point.  Nothing below raises, so everything computed above
        # is written onto the graph here, together: the plan the step was
        # built from, the state and parameter snapshots it closes over, the
        # ``_meta`` built above (see there) and the dividers the next
        # compile will judge its phase against.  Up to here, a raise leaves
        # the graph running exactly the step it was running before.
        # ------------------------------------------------------------------
        self._schedule = plan.schedule
        self._back_edges = plan.back_edges
        self._is_multirate = plan.is_multirate
        self._rate_dividers = plan.rate_dividers
        self._params = plan.params
        self._record_param_sync()
        self._params_dtypes = params_dtypes
        self._params_shapes = params_shapes
        self._state = state
        self._default_ext_leaves = default_ext_leaves
        if meta:
            self._state[_graph_specs._META_KEY] = meta
        else:
            self._state.pop(_graph_specs._META_KEY, None)
        self._committed_rate_dividers = dict(plan.rate_dividers)
        self._committed_coupling_groups = {
            "+".join(sorted(g.nodes)): g for g in self._coupling_groups
        }
        # What the report's float floor rests on, as the step was built:
        # each group's structural evaluation count, whether every member
        # declared it, and its internal edges in the order its norm sums
        # them (``coupling_diagnostics``; ``InterfacePlan.norm_edges``).
        self._committed_floor_inputs = {
            "+".join(sorted(g.nodes)): (
                *_group_layout._group_evaluations(g, self._nodes, self._schedule, self._edges),
                interface_plans["+".join(sorted(g.nodes))].norm_edges())
            for g in self._coupling_groups
        }
        self._committed_geometry_edges = {
            key: tuple(r.key for r in plan.resolved_geometry_edges())
            for key, plan in interface_plans.items()
        }
        # Why each such group's report withholds its bounds whatever the
        # step measures (``None``: the diagnostics read its geometry).
        self._committed_geometry_refusals = {
            "+".join(sorted(g.nodes)): _group_layout._geometry_diagnostics_refusal(
                g, self._nodes, interface_plans["+".join(sorted(g.nodes))])
            for g in self._coupling_groups
        }
        # Count Python-level traces of the step: a robust, JAX-version-
        # independent retrace probe (the jit object's C++ cache count is
        # not comparable across versions).  ``trace_count`` is 0 right
        # after compile() and 1 after the first step of a well-behaved
        # graph; a growing count means something in the call signature
        # (weak types, dtypes, params structure) keeps changing.
        self._n_traces = 0
        self._compiled_step = compiled_step
        self._raw_step_fn = step_fn
        self._step_reads = None
        self._node_reads = {}
        self._params_verified = {
            owner: {k: v for k, v in leaves.items()
                    if fresh_leaves.get(owner, {}).get(k) is v}
            for owner, leaves in plan.params.get("nodes", {}).items()
        }
        self._static_data_hashes = static_data_hashes

        self._dirty = False
        self._underflow_check_pending = bool(self._coupling_groups)
        # A rebuilt step invalidates every scan built against the old
        # one.  Bumping the generation as well as clearing means a scan
        # a caller still holds can never be re-entered into the cache.
        self._compile_generation += 1
        self._scan_cache.clear()
        self._n_scan_traces = 0
        self._notify(EVENT_COMPILED, self._schedule)

    def _check_static_data_dirty(self) -> bool:
        """Return True (and set ``self._dirty=True``) if any node's
        :attr:`static_data` has changed shape/dtype since the last
        ``compile()``.

        Called from each public entry-point (``step``, ``run``, etc.)
        before the standard dirty-check so a stale JIT cache is caught
        without requiring the caller to mark the graph dirty manually.  It
        also catches a ``node.params`` write since the last compile
        (:meth:`_sync_node_param_writes`), so every entry point runs the
        same model.
        """
        if self._sync_node_param_writes():
            return True
        for name, spec in self._nodes.items():
            if spec.node.static_data_hash() != self._static_data_hashes.get(name, 0):
                self._dirty = True
                return True
        return False

    # ------------------------------------------------------------------
    # Sharding validation (v0.2 #3 follow-up)
    # ------------------------------------------------------------------

    def validate_sharding(self) -> list["ShardingIssue"]:
        """Structural checks for the sharding spec across the graph.

        Today: sharding spec consistency only.

        Returns a list of :class:`ShardingIssue` instances (each typed
        with a ``severity`` and a ``code``); the empty list means
        healthy.  Callers decide severity — turn into a raising call
        by filtering for ``severity == "error"`` and calling ``raise``.

        Scope (deliberately tight to avoid a god-method):

        * A sharded node exists, but the graph has no device mesh
          configured.
        * A sharded node's mesh disagrees with the graph's device mesh.

        NOT in scope here:

        * Edge-validation (compile-time; see
          :meth:`validate`).
        * Multi-rate divisibility.
        * Cycle detection.

        If you find yourself wanting to extend this method past
        sharding-spec consistency, consider a sibling
        ``validate_<topic>()`` instead.
        """
        issues: list[ShardingIssue] = []
        # Identify sharded nodes by either explicit class or by carrying
        # a `_mesh` attribute (the Sharded*Node convention).
        sharded_nodes = [
            (name, spec.node) for name, spec in self._nodes.items()
            if hasattr(spec.node, "_mesh") and getattr(spec.node, "_mesh", None) is not None
        ]

        if not sharded_nodes:
            return issues  # nothing to validate

        if self._multigpu_mesh is None:
            issues.append(ShardingIssue(
                severity="warning",
                code="sharded_node_without_graph_mesh",
                message=(
                    f"{len(sharded_nodes)} sharded node(s) present but the "
                    f"GraphManager has no device mesh configured.  "
                    f"Call gm.enable_multigpu(...) or remove the sharding "
                    f"from the affected nodes."
                ),
                affected_nodes=[name for name, _ in sharded_nodes],
            ))
            return issues

        # All sharded nodes must agree with the graph's mesh.
        graph_axis_names = tuple(self._multigpu_mesh.axis_names)
        for name, node in sharded_nodes:
            node_mesh = getattr(node, "_mesh", None)
            assert node_mesh is not None  # `sharded_nodes` is filtered on it
            node_axes = tuple(node_mesh.axis_names)
            if node_axes != graph_axis_names:
                issues.append(ShardingIssue(
                    severity="error",
                    code="sharded_node_mesh_axes_mismatch",
                    message=(
                        f"Sharded node {name!r} uses mesh axes {node_axes!r} "
                        f"but the graph's mesh is {graph_axis_names!r}.  "
                        f"Re-create the node with the graph's mesh, or "
                        f"call enable_multigpu with matching axes."
                    ),
                    affected_nodes=[name],
                ))
        return issues

    # ------------------------------------------------------------------
    # Multi-GPU
    # ------------------------------------------------------------------

    def enable_multigpu(
        self,
        n_devices: Optional[int] = None,
        partition_strategy: str = "auto",
        *,
        mesh_shape: Optional[tuple[int, ...]] = None,
        mesh_axes: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Enable multi-GPU coupling and (in v0.2) stencil sharding.

        Requires at least one coupling group with ``iteration_mode="jacobi"``.
        Uses ``jax.shard_map`` to distribute node updates
        across a device mesh.

        Parameters
        ----------
        n_devices : int, optional
            Number of devices to use.  Defaults to ``prod(mesh_shape)`` when
            ``mesh_shape`` is provided, otherwise all available devices.
        partition_strategy : str
            ``"auto"`` (default) assigns coupled nodes to the same device.
        mesh_shape : tuple[int, ...], optional
            Mesh shape for N-D (pencil) decomposition, e.g. ``(2, 4)`` for
            an 8-device 2-D pencil mesh.  When omitted the mesh is 1-D
            (slab decomposition, v0.1 behaviour).
        mesh_axes : tuple[str, ...], optional
            Axis names for the mesh.  Defaults: ``("devices",)`` for 1-D
            and ``("spatial_y", "spatial_z")`` for 2-D.  Length must match
            ``len(mesh_shape)``.
        """
        from maddening.cloud.multigpu.device_mesh import create_device_mesh
        from maddening.cloud.multigpu.partition import assign_nodes_to_devices

        jacobi_groups = [
            g for g in self._coupling_groups
            if g.iteration_mode == "jacobi"
        ]
        if not jacobi_groups:
            raise ValueError(
                "enable_multigpu requires at least one coupling group "
                "with iteration_mode='jacobi'"
            )

        # Mesh and device map into locals, committed together below.
        # ``assign_nodes_to_devices`` and ``EdgeSpec.to_dict`` can raise,
        # and a mesh committed on its own leaves ``validate_sharding``
        # judging the nodes against a mesh the graph is not using while
        # the step still runs on the previous device map -- with
        # ``_dirty`` never set, so nothing recompiles to reconcile them.
        mesh = create_device_mesh(
            n_devices, shape=mesh_shape, axis_names=mesh_axes
        )
        n = len(mesh.devices.reshape(-1))

        coupling_sets = [set(g.nodes) for g in self._coupling_groups]
        edges_dicts = [e.to_dict() for e in self._edges]
        device_map = assign_nodes_to_devices(
            node_names=list(self._nodes.keys()),
            edges=edges_dicts,
            coupling_groups=coupling_sets,
            n_devices=n,
        )
        self._multigpu_mesh = mesh
        self._multigpu_device_map = device_map
        self._dirty = True

    def _committed_plan(self) -> "_graph_specs._StepPlan":
        """The plan the graph is currently running, as last committed by
        ``compile()``.  Rebuilding the step from this reproduces the step
        that is live, which is what a caller that asks for a step function
        outside ``compile()`` means."""
        return _graph_specs._StepPlan(
            schedule=list(self._schedule),
            back_edges=list(self._back_edges),
            is_multirate=self._is_multirate,
            rate_dividers=dict(self._rate_dividers),
            params=self.params,
        )

    def _build_step_fn(self, plan: Optional["_graph_specs._StepPlan"] = None) -> Callable:
        """Create a pure function ``(full_state, ext_inputs) -> full_state``.

        When multi-rate is active, the step function increments an
        internal step counter and conditionally applies each node's
        update based on whether ``step_count % rate_divider == 0``.
        The update is always *computed* (to keep the function
        JAX-traceable with static structure), but the result is applied
        only when the node should fire.

        When coupling groups are defined, nodes within each group are
        wrapped in a ``jax.lax.while_loop`` that iterates
        (Gauss-Seidel) until convergence or max_iterations.

        Parameters
        ----------
        plan : _StepPlan, optional
            The schedule, multi-rate info and parameter snapshot to build
            against.  ``compile()`` passes the plan it has computed but
            not yet committed, so the build -- which can raise -- happens
            before the graph is touched.  ``None`` builds against what is
            committed on the graph, which is what a caller rebuilding the
            step of an already-compiled graph wants.
        """
        if plan is None:
            plan = self._committed_plan()
        schedule = list(plan.schedule)
        nodes = dict(self._nodes)
        back_edge_set = set(plan.back_edges)
        is_multirate = plan.is_multirate
        rate_dividers = dict(plan.rate_dividers)
        coupling_groups = list(self._coupling_groups)

        # Map node -> coupling group
        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group
        # The mesh ``strict_convergence`` raises on every device of, on a
        # step spanning several (MADD-ANO-162).
        strict_mesh = self._strict_mesh(coupling_groups)

        # Build block schedule: list of (type, data) where type is
        # "node" (single node) or "coupled" (CouplingGroup, node_list)
        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    # Collect nodes in this group in schedule order
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        # Pre-index edges by target node -- O(E) setup, O(degree) per node
        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        # Pre-index external inputs by target node
        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)
        # The dtype each one is declared with: every step program casts
        # the inputs it is handed to it (``_cast_external_inputs``).
        declared_input_dtypes = _graph_specs._declared_input_dtypes(
            self._external_inputs)

        # Capture which nodes have external inputs (for the fast path)
        has_external = set(ext_by_target.keys())

        has_coupling = bool(coupling_groups)

        # Track flux outputs for flux-based edges in non-coupled path
        flux_state: dict[str, dict] = {}

        # ``params=None`` on the step means "the compile-time snapshot":
        # baked in as constants, exactly the pre-params behaviour.  An
        # explicit ``params`` is a traced input, so ``jax.grad`` reaches
        # it and a new value needs no recompile.
        params_snapshot = plan.params

        def _resolve_params(params):
            if params is None:
                params = params_snapshot
            else:
                self._validate_params(params)
            return _graph_specs._ResolvedParams(params.get("nodes", {}), params.get("mappings", {}))

        # Nodes with an incoming edge whose mapping reads a geometry from
        # the node's own state (experimental; empty for every other graph).
        target_geometry_nodes = frozenset(
            e.target_node for e in self._edges
            if e.geometry is not None and e.geometry[0] == "target")

        def _resolve_and_update_node(
            node_name, new_state, full_state, external_inputs, node_params,
            force_forward_edges=None, hold=None,
        ):
            """Resolve boundary inputs and update a single node.

            Parameters
            ----------
            force_forward_edges : set or None
                If provided, edges in this set are treated as forward
                (use new_state) even if they are in back_edge_set.
            hold : callable or None
                Multi-rate only, for a node whose rate divider is above
                one: maps the result of ``update`` to the state the node
                holds after this base step (that result on a base step
                the node fires on, its unchanged state on any other).
                The flux hook is called on what it returns, so a reader
                of a flux sees the flux of a state the graph holds, as a
                reader of a state field does.
            """
            boundary_inputs: dict[str, Any] = {}
            # Static: whether an incoming edge reads a geometry from this
            # node's own state.  Only then is what each edge read kept,
            # for the flux hook below.
            reads_own_geometry = node_name in target_geometry_nodes
            delivered: list = []

            for edge in edges_by_target[node_name]:
                # Determine source state: back-edges read from full_state
                # (previous timestep), forward edges from new_state.
                if edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                ):
                    src_state = full_state
                else:
                    src_state = new_state
                # Check state first, then flux outputs
                src_nn = edge.source_node
                src_dict = src_state.get(src_nn, {})
                if edge.source_field in src_dict:
                    value = src_dict[edge.source_field]
                elif src_nn in flux_state and edge.source_field in flux_state[src_nn]:
                    value = flux_state[src_nn][edge.source_field]
                else:
                    value = src_state[src_nn][edge.source_field]
                raw = value
                # A geometry (experimental) is read where the value was:
                # from ``src_state`` for a source anchor, and for a target
                # anchor from the state ``update`` is about to receive.
                value = _graph_specs._apply_edge(
                    edge, value, node_params,
                    _graph_specs._edge_geom(edge, src_state, new_state[node_name]))
                if reads_own_geometry:
                    delivered.append((edge, raw, src_state, value))
                if edge.additive and edge.target_field in boundary_inputs:
                    boundary_inputs[edge.target_field] = (
                        boundary_inputs[edge.target_field] + value
                    )
                else:
                    boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = external_inputs.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes[node_name]
            new_node_state = _graph_specs._node_update(
                spec, new_state[node_name], boundary_inputs, spec.timestep,
                node_params.nodes.get(node_name),
            )
            if hold is not None:
                # Before the flux hook, not after it: the hook used to be
                # called on the result of ``update`` on every base step,
                # so between a slow node's firings its readers received
                # the flux of a state the step then threw away (and a
                # geometry the node holds was read from that state).
                new_node_state = hold(new_node_state)

            # Compute fluxes for this node if it produces them
            from maddening.core.node import SimulationNode as _SimBase
            if type(spec.node).compute_boundary_fluxes is not _SimBase.compute_boundary_fluxes:
                if reads_own_geometry:
                    # The flux hook is called with the post-update state,
                    # so a geometry this node holds is read from that
                    # state: the edges are resolved again, in the same
                    # order, each from the value it read above.
                    boundary_inputs = {}
                    for edge, raw, src_state, value in delivered:
                        if edge.geometry is not None and edge.geometry[0] == "target":
                            value = _graph_specs._apply_edge(
                                edge, raw, node_params,
                                _graph_specs._edge_geom(edge, src_state, new_node_state))
                        if edge.additive and edge.target_field in boundary_inputs:
                            boundary_inputs[edge.target_field] = (
                                boundary_inputs[edge.target_field] + value
                            )
                        else:
                            boundary_inputs[edge.target_field] = value
                    if node_name in has_external:
                        node_ext = external_inputs.get(node_name, {})
                        for ei in ext_by_target[node_name]:
                            if ei.target_field in node_ext:
                                boundary_inputs[ei.target_field] = node_ext[ei.target_field]
                fluxes = _graph_specs._node_fluxes(
                    spec, new_node_state, boundary_inputs, spec.timestep,
                    node_params.nodes.get(node_name),
                )
                if fluxes:
                    flux_state[node_name] = fluxes

            return new_node_state

        def _run_coupled_block(group, group_schedule, new_state,
                               full_state, external_inputs, node_params,
                               runtime_dt=None, fires=None):
            """Execute a coupling group with Gauss-Seidel iteration.

            In Gauss-Seidel coupling, each iteration re-solves the SAME
            timestep with updated boundary conditions from the latest
            iteration.  Crucially, each node integrates from the
            *initial* state (beginning of timestep), NOT from the
            previous iteration's output.  Only the boundary conditions
            change between iterations.

            Parameters
            ----------
            group : CouplingGroup
                Configuration for this coupling group.
            group_schedule : list of str
                Node names in execution order within the group.
            new_state : dict
                Current accumulated state for this timestep.
            full_state : dict
                State from the previous timestep (for back-edges).
            external_inputs : dict
                External inputs dict.
            runtime_dt : JAX scalar or None
                If provided, overrides each node's compiled timestep
                (used by adaptive timestepping).
            fires : JAX bool or None
                Multi-rate only: whether this base step keeps the
                group's solve (see ``_run_coupled_block_impl``).
            """
            return _coupled_block._run_coupled_block_impl(
                group, group_schedule, new_state, full_state,
                external_inputs, runtime_dt,
                nodes=nodes, edges_by_target=edges_by_target,
                ext_by_target=ext_by_target, back_edge_set=back_edge_set,
                has_external=has_external, all_edges=self._edges,
                multigpu_device_map=self._multigpu_device_map,
                node_params=node_params, fires=fires,
                probe_sink=self._gradient_whole_probes,
                strict_mesh=strict_mesh,
            )

        if not is_multirate and not has_coupling:
            # ---- Uniform-rate, no coupling: fast path ----
            def graph_step(full_state, external_inputs, params=None):
                external_inputs = _graph_specs._cast_external_inputs(
                    external_inputs, declared_input_dtypes)
                node_params = _resolve_params(params)
                new_state = {k: v for k, v in full_state.items()}

                for node_name in schedule:
                    new_state[node_name] = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs, node_params
                    )
                return new_state

            return graph_step

        if has_coupling and not is_multirate:
            # ---- Coupling groups, uniform rate ----
            def graph_step_coupled(full_state, external_inputs, params=None):
                external_inputs = _graph_specs._cast_external_inputs(
                    external_inputs, declared_input_dtypes)
                node_params = _resolve_params(params)
                new_state = {k: v for k, v in full_state.items()}

                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        new_state[node_name] = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs, node_params
                        )
                    else:
                        _, group, group_schedule = block
                        new_state = _run_coupled_block(
                            group, group_schedule, new_state,
                            full_state, external_inputs, node_params,
                        )

                return new_state

            return graph_step_coupled

        # ---- Multi-rate path (with or without coupling) ----
        def graph_step_multirate(full_state, external_inputs, params=None):
            external_inputs = _graph_specs._cast_external_inputs(
                external_inputs, declared_input_dtypes)
            node_params = _resolve_params(params)
            step_count = full_state[_graph_specs._META_KEY]["step_count"]
            new_state = {k: v for k, v in full_state.items()}

            def _held(node_name, current_state):
                """What ``update``'s result becomes on this base step.

                ``None`` for a node that fires on every base step.  For
                a slower one, a selection between the result and the
                state the node already holds, applied inside
                ``_resolve_and_update_node`` before the flux hook.
                """
                rd = rate_dividers[node_name]
                if rd == 1:
                    return None
                should_run = (step_count % rd) == 0
                held_state = current_state[node_name]
                return lambda updated: jax.tree.map(
                    lambda new_val, old_val: jnp.where(should_run, new_val, old_val),
                    updated,
                    held_state,
                )

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        node_name = block[1]
                        new_state[node_name] = _resolve_and_update_node(
                            node_name, new_state, full_state, external_inputs, node_params,
                            hold=_held(node_name, new_state),
                        )
                    else:
                        _, group, group_schedule = block
                        # Every member shares one divider: a sub-cycling
                        # group is scheduled at its macro timestep and any
                        # other group at the single timestep ``compile``
                        # requires of it.
                        group_rd = rate_dividers[group_schedule[0]]
                        fires = (None if group_rd == 1
                                 else (step_count % group_rd) == 0)

                        def _solve(state_in, group=group,
                                   group_schedule=group_schedule, fires=fires):
                            return _run_coupled_block(
                                group, group_schedule, state_in,
                                full_state, external_inputs, node_params,
                                fires=fires,
                            )

                        if fires is None:
                            coupled_result = _solve(new_state)
                        else:
                            # Solve only on a base step that keeps the
                            # result; the other branch hands the state
                            # through untouched.  The group's node states
                            # *and* its ``_meta`` slots -- diagnostics,
                            # predictor history, IQN-IMVJ warm start -- are
                            # therefore those of the last applied solve.
                            # The solve used to run on every base step with
                            # only the node states selected afterwards, so
                            # between firings the slots described solves
                            # the step threw away (the report, the profiler,
                            # sysid's mask, the predictor and the warm start
                            # all read them), and a group at divider ``d``
                            # paid its fixed-point iteration ``d`` times per
                            # solve it applied.  Under ``vmap`` over states
                            # at different phases the batched ``cond``
                            # selects per element between the two branches'
                            # outputs, which is the same rule.
                            coupled_result = jax.lax.cond(
                                fires, _solve, dict, new_state)
                        for nn in group_schedule:
                            new_state[nn] = coupled_result[nn]
                        if _graph_specs._META_KEY in coupled_result:
                            new_state[_graph_specs._META_KEY] = {
                                **new_state.get(_graph_specs._META_KEY, {}),
                                **coupled_result[_graph_specs._META_KEY],
                            }
            else:
                for node_name in schedule:
                    new_state[node_name] = _resolve_and_update_node(
                        node_name, new_state, full_state, external_inputs, node_params,
                        hold=_held(node_name, new_state),
                    )

            # Increment step counter (preserve diagnostic keys)
            new_state[_graph_specs._META_KEY] = {
                **new_state.get(_graph_specs._META_KEY, {}),
                "step_count": step_count + 1,
            }
            return new_state

        return graph_step_multirate

    def _default_external_inputs(self) -> dict[str, dict]:
        """Build a zero-valued external_inputs dict matching declared specs."""
        if not self._external_inputs:
            return _graph_specs._EMPTY_EXTERNAL_INPUTS
        cache = getattr(self, "_default_ext_leaves", None) or {}
        ext: dict[str, dict] = {}
        for ei in self._external_inputs:
            leaf = cache.get((ei.target_node, ei.target_field))
            if leaf is None or leaf.shape != tuple(ei.shape) or leaf.dtype != jnp.dtype(ei.dtype):
                leaf = jnp.zeros(ei.shape, dtype=ei.dtype)
            # Fresh outer dicts each call (callers may edit them); the
            # zero arrays themselves are immutable and shared.
            ext.setdefault(ei.target_node, {})[ei.target_field] = leaf
        return ext

    def _warn_of_input_casts(self, external_inputs: dict[str, dict]) -> None:
        """Warn, once per input of this graph, of a supplied value that the
        cast to its declared dtype changes.

        Every step program casts an external input to the dtype it was
        declared with (``_graph_specs._cast_external_inputs``).  Until
        0.4.0 the step ran a supplied value in the dtype it arrived in, so
        a graph whose values the cast changes -- a float64 value into the
        default float32 declaration under ``jax_enable_x64``, a fraction
        into an integer input -- now computes other numbers, and says so
        here.  Nothing is said of a value the cast leaves as it was.
        """
        warned = self.__dict__.setdefault("_input_cast_warned", set())
        for ei in self._external_inputs:
            key = (ei.target_node, ei.target_field)
            if key in warned:
                continue
            fields = external_inputs.get(ei.target_node)
            if not fields or ei.target_field not in fields:
                continue
            dtype = jax.dtypes.canonicalize_dtype(ei.dtype)
            kind = _graph_specs._input_cast_changes(fields[ei.target_field], dtype)
            if kind is None:
                continue
            warned.add(key)
            warnings.warn(
                f"external input '{ei.target_node}.{ei.target_field}' is declared "
                f"{dtype.name} and was handed {kind} value that {dtype.name} does "
                f"not hold as given: the step runs it cast to {dtype.name}.  An "
                f"external input is cast to its declared dtype at the step "
                f"boundary, which is what the FMU export, the zeros of an omitted "
                f"input and the saved configuration already took it to be; before "
                f"0.4.0 the step ran a supplied value in the dtype it arrived in, "
                f"so this graph's numbers have changed.  To run the value as "
                f"handed in, declare the input with its dtype: "
                f"add_external_input({ei.target_node!r}, {ei.target_field!r}, "
                f"dtype=...) (float32 is the default, also under "
                f"jax_enable_x64).  This is said once per input of a graph.",
                UserWarning,
                stacklevel=4,
            )

    def _resolve_external_inputs(
        self, external_inputs: Optional[dict[str, dict]],
    ) -> dict[str, dict]:
        """Complete and validate a caller's ``external_inputs``.

        ``None`` means "zeros for every declared input", which was
        already documented.  A *partial* dict now means the same for the
        inputs it omits, rather than leaving them out of
        ``boundary_inputs`` altogether and letting the node fall back to
        its own default -- a 98 N difference in the case that found this,
        with nothing said about it anywhere.

        An unknown ``(node, field)`` pair is an error naming the declared
        ones.  A typo'd node or field name used to be accepted in
        silence, which is exactly the failure mode ``_validate_params``
        spends a paragraph per case avoiding for the ``params`` argument
        of the very same call.
        """
        if external_inputs is None:
            return self._default_external_inputs()
        declared = {
            (ei.target_node, ei.target_field) for ei in self._external_inputs
        }
        unknown = sorted(
            f"{node}.{field}"
            for node, fields in external_inputs.items()
            for field in fields
            if (node, field) not in declared
        )
        if unknown:
            known = sorted(f"{n}.{f}" for n, f in declared)
            raise ValueError(
                f"external_inputs names {unknown}, which this graph does not "
                f"declare; declared external inputs: {known or ['(none)']}.  "
                f"An undeclared name never reaches the node, so accepting it "
                f"would mean the value silently did nothing.  Declare it with "
                f"add_external_input(), or fix the name."
            )
        if not declared:
            return external_inputs
        self._warn_of_input_casts(external_inputs)
        # Complete from the per-compile zero cache.  Fresh outer dicts,
        # like ``_default_external_inputs``: callers may edit them.
        out: dict[str, dict] = {}
        for node, fields in self._default_external_inputs().items():
            out[node] = {**fields, **external_inputs.get(node, {})}
        return out

    # ------------------------------------------------------------------
    # Escaped tracers
    # ------------------------------------------------------------------

    def _store_stepped_state(self, new_state: dict) -> None:
        """:meth:`_store_state` for the result of one compiled step, which
        must have the layout of the state it replaces.

        A node whose ``update`` returns a leaf of another shape -- a
        constant given as a list where the node's state is a scalar, an
        edge or an external input delivering a vector into a scalar field
        -- broadcast the leaf at that step, and :meth:`step` and
        :meth:`run` stored the result without a word: the state no longer
        had the layout of ``initial_state()``, so a checkpoint saved after
        the step did not load into the graph after :meth:`reset_state` or
        into the graph rebuilt from :meth:`to_dict`.  :meth:`run_scan` and
        its siblings have always refused it, loudly, as a scan carry of
        another type.  Refused here by name, and nothing is stored.

        Compared once per trace of the step: a traced step is one
        program, so the layout it returns is fixed by the layout it is
        given, and the next comparison is due when the step is traced
        again (a new compile; an external input of another shape).  The
        comparison is of keys, shapes and kinds of dtype, host-side: no
        trace, no call of a node's ``update``, and nothing is added to
        the step.  Every other step costs one tuple comparison.
        """
        trace = (self._compile_generation, self._n_traces)
        if self._layout_checked_trace != trace:
            self._refuse_layout_drift(new_state)
            self._layout_checked_trace = trace
        self._store_state(new_state)

    def _refuse_layout_drift(self, new_state: dict) -> None:
        """Raise if *new_state* has not the keys, shapes and kinds of
        dtype of the state the graph holds (see
        :meth:`_store_stepped_state`).  A dtype's width is not compared:
        with x64 enabled a stock node's update returns ``float64`` for the
        ``float32`` its ``initial_state()`` builds."""
        drift = _param_probes._state_layout_drift(self._state, new_state)
        if drift:
            raise ValueError(
                "one step changes the layout of the graph's state, so the "
                "stepped state could not be saved and reloaded, reset or "
                "scanned: " + "; ".join(drift) + " (leaves are named "
                "node/field).  A node's update() must return its state "
                "with the shapes and kinds of dtype initial_state() gave "
                "it: check the node's constructor values and the shapes "
                "its edges and external inputs deliver.  Nothing was "
                "stored."
            )

    def _store_state(self, new_state: dict) -> None:
        """Write *new_state* back, remembering whether it is traced.

        ``step``, ``run``, ``run_scan`` and their siblings are stateful:
        they assign their result into ``self._state``.  Under a JAX
        transform that result is a pytree of tracers, so a loss that
        calls one -- the recipe ``docs/user_guide/quickstart.md`` shows
        -- left the graph holding tracers once the transform returned,
        and every later ``step`` / ``run_scan`` / ``save_state`` failed
        with an error pointing at JAX rather than at the framework.

        The write still happens, because a Python loop of ``gm.step()``
        *inside* a trace depends on it.  What is added is the state to
        come back to: see :meth:`_recover_from_escaped_tracers`.
        """
        if _graph_specs._holds_tracer(new_state):
            if not self._state_traced:
                self._state_before_trace = self._state
            self._state_traced = True
        else:
            self._state_traced = False
            self._state_before_trace = None
            # The state being stored is what its own step left.  (A traced
            # one is put back to the state before it, whose reports the
            # kept copy still describes.)
            self._state_as_reported = None
        self._state = new_state
        if self._underflow_check_pending and not self._state_traced:
            self._underflow_check_pending = False
            self._warn_underflow_range(new_state)

    def _warn_underflow_range(self, state: dict) -> None:
        """Warn once per coupled group whose fields are in the subnormal range.

        Runs on the host, once per compile, on the first state a stepper
        stores outside a transform (:meth:`_store_state`): the remedy --
        rescaling the field's units -- is a decision about the model's
        configuration, and the first step is where every caller passes,
        whether or not they ever read :meth:`coupling_diagnostics`.  It reads
        each group field once (one device-to-host copy per compile) and
        nothing inside the compiled step changes, so stepping and its
        results are untouched.  A state that decays into the range after the
        first step is not re-checked.  See
        :class:`~maddening.warnings.UnderflowRangeWarning`.
        """
        from maddening.warnings import UnderflowRangeWarning  # noqa: PLC0415

        for key, hits in _reports._underflow_range_fields(self._coupling_groups, state).items():
            if key in self._underflow_warned or not hits:
                continue
            self._underflow_warned.add(key)
            (nn, fld, mag, dtype), more = hits[0], len(hits) - 1
            info = jnp.finfo(dtype)
            threshold = float(info.tiny) / float(info.eps)
            warnings.warn(
                f"coupling group {key!r}: field {nn}.{fld} has magnitude {mag:.3g}, "
                f"inside the {dtype} subnormal range (below tiny/eps = {threshold:.3g}): "
                f"a change of one ulp of it is smaller than the smallest normal number, "
                f"which XLA's CPU backend flushes to zero, so the nodes' arithmetic on it "
                f"loses resolution, and below tiny = {float(info.tiny):.3g} the group's "
                f"convergence norm reads it as exactly zero.  Rescale the field's units "
                f"so that it is of order one."
                + (f"  {more} more field(s) of this group are in the range too." if more else ""),
                UnderflowRangeWarning, stacklevel=4,
            )

    def _recover_from_escaped_tracers(self, *, warn: bool = True) -> None:
        """Put the graph back to the last untraced state, if it needs it.

        Called from every public entry point that reads or writes the
        state: the steppers, ``compile``, ``coupling_diagnostics``,
        ``save_state`` / ``load_state``, ``get_node_state`` /
        ``set_node_state``, ``reset_state``, ``resolve_boundary_inputs``,
        ``validate`` and ``add_node`` / ``remove_node``.  It costs one
        attribute test when there is nothing to do, which is always
        except right after a transform that stepped the graph.  Inside a
        transform it does nothing, so a traced multi-step loop still
        works.

        ``warn=False`` is for an entry point that replaces the whole
        state anyway (``reset_state``): the warning's advice -- set the
        state you want explicitly -- is what the caller is doing, and
        the state put back is never observed.  An entry point that missed
        this call read the traced state: ``reset_state`` sized a
        predictor group's seed from it and raised
        ``UnexpectedTracerError``, and ``set_node_state`` wrote into it,
        so the next entry point put the graph back over the write.
        """
        if not self._state_traced or not _graph_specs._outside_jax_trace():
            return
        restored = self._state_before_trace
        if restored is None:            # pragma: no cover - defensive
            self._state_traced = False
            return
        # State first, flags after, so a graph that somehow failed to be
        # put back is still marked as holding tracers and tries again.
        self._state = restored
        self._state_traced = False
        self._state_before_trace = None
        if not warn:
            return
        warnings.warn(
            "the graph held JAX tracers left behind by a transform and has "
            "been put back to the state it had before it.  step() / run() / "
            "run_scan() assign their result into the graph, so a loss that "
            "calls one leaves tracers in it when jax.grad returns; the "
            "gradient itself is unaffected.  Set the state you want "
            "explicitly (set_node_state / reset_state / load_state) after "
            "differentiating if the recovered state is not the one you meant.",
            RuntimeWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------
    # A report describes the step that ran
    # ------------------------------------------------------------------

    def _keep_state_for_reports(self) -> None:
        """Keep what the last step left, before a write that is not a step's.

        ``coupling_diagnostics()`` measures the residual's float floor on
        the state the step returned.  The ``_meta`` slots are the step's
        own record, but the node states are live, so the same step's
        ``spectral_error_bound``, ``precision_limited`` and
        ``spectral_usable`` moved when the state was written afterwards:
        a float32 pair stalled at ``residual == 0.0`` read a bound of
        ``0.0``, ``spectral_usable=True``, after ``set_node_state`` to
        zeros, 9e-5 from its fixed point.

        Called by every door that writes node states in place
        (``set_node_state`` -- so ``PUT /graph/state`` and
        ``load_state`` -- ``add_node``, ``remove_node``) *before* its
        write.  The first such write after a step keeps a shallow copy of
        the state (references to the arrays, no data) beside the report
        slots it goes with; later ones find it there.  A step drops it
        (``_store_state``), so it holds the replaced arrays only until
        the next step.  The reader (``_state_a_report_describes``) uses
        it only while the group's slots are still the very objects kept
        here, so a door that brings its own slots -- a step, a loaded
        checkpoint, ``reset_state``, a recompile that restarts a
        replaced group -- needs no bookkeeping: the report then reads
        the live state, which is what came with those slots.

        Nothing is kept while the state holds tracers: the graph is put
        back to the state before the trace, which the copy already kept
        (or the live state) describes.
        """
        if self._state_traced or not self._committed_floor_inputs:
            return
        meta = self._state.get(_graph_specs._META_KEY)
        if not meta:
            return
        kept = self._state_as_reported
        if (kept is not None and len(kept[1]) == len(meta)
                and all(kept[1].get(slot) is value for slot, value in meta.items())):
            return
        self._state_as_reported = (
            {name: fields for name, fields in self._state.items()
             if name != _graph_specs._META_KEY},
            dict(meta),
        )

    def _state_a_report_describes(self, slots: Sequence[str], nodes) -> dict:
        """The node states the step that wrote ``slots`` left.

        The copy kept before the first later write
        (:meth:`_keep_state_for_reports`) while ``slots`` are still the
        objects it was kept with; otherwise the live state.
        """
        kept = self._state_as_reported
        if kept is None:
            return self._state
        state, kept_meta = kept
        meta = self._state.get(_graph_specs._META_KEY, {})
        if (all(slot in meta and kept_meta.get(slot) is meta[slot] for slot in slots)
                and all(name in state for name in nodes)):
            return state
        return self._state

    def _groups_written_after_their_step(self) -> list[str]:
        """The groups a checkpoint marks "state written after the last step".

        A checkpoint is a copy of the state, ``_meta`` included, and holds
        the *written* state: what the step returned, which the report's
        float floor is measured on, is not in it.  So the archive carries
        the fact beside the state (``checkpoint.save_state``), and the
        graph that loads it reports the group's floor-dependent entries
        as not usable (:meth:`_loaded_after_a_write`).

        A group is marked where its report is measured on a kept copy
        because a member has since been written to other values, and
        where this graph itself loaded it marked and has not stepped it
        since.  Not a group that stores its floor in the step (CPL-188).
        """
        meta = self._state.get(_graph_specs._META_KEY, {})
        if not meta:
            return []
        marked = []
        for key in self._committed_floor_inputs:
            nodes = key.split("+")
            slots = (f"coupling_{key}_iterations", f"coupling_{key}_residual")
            if any(slot not in meta for slot in slots) or int(meta[slots[0]]) <= 0:
                continue    # no report to qualify
            stored = meta.get(f"coupling_{key}_reading_floor")
            if stored is not None and np.isfinite(np.asarray(stored)):
                continue    # the step stored this group's floor itself
            if self._loaded_after_a_write(key):
                marked.append(key)
                continue
            state = self._state_a_report_describes(slots, nodes)
            if state is self._state:
                continue
            for name in nodes:
                live = self._state.get(name)
                if live is None or state[name] is live:
                    continue
                if (set(live) != set(state[name]) or any(
                        np.asarray(live[f]).tobytes() != np.asarray(state[name][f]).tobytes()
                        or np.asarray(live[f]).dtype != np.asarray(state[name][f]).dtype
                        for f in live)):
                    marked.append(key)
                    break
        return marked

    def _note_loaded_after_a_write(self, keys) -> None:
        """Record that a checkpoint just loaded marks *keys* as written
        after their last step (``checkpoint.load_state``, on success).

        Kept beside the report slots the load installed: the record holds
        while they are the ones in the graph, so the group's next step, a
        reset or another load ends it with no bookkeeping.
        """
        meta = self._state.get(_graph_specs._META_KEY, {})
        self._reports_loaded_after_a_write = {
            key: (meta.get(f"coupling_{key}_iterations"), meta.get(f"coupling_{key}_residual"))
            for key in keys if key in self._committed_floor_inputs
        }

    def _loaded_after_a_write(self, key: str) -> bool:
        """Whether *key*'s report came from a checkpoint saved after its
        members were written (see :meth:`_note_loaded_after_a_write`)."""
        noted = self._reports_loaded_after_a_write.get(key)
        if noted is None:
            return False
        meta = self._state.get(_graph_specs._META_KEY, {})
        return (noted[0] is not None
                and meta.get(f"coupling_{key}_iterations") is noted[0]
                and meta.get(f"coupling_{key}_residual") is noted[1])

    # ------------------------------------------------------------------
    # Internal helpers for _meta stripping
    # ------------------------------------------------------------------

    def _user_state(self, full_state: dict) -> dict:
        """The caller's view of *full_state*: no internal ``_meta`` key,
        and a fresh dict per node.

        It used to hand back ``self._state`` itself for a graph with no
        ``_meta`` (the common uncoupled, uniform-rate case), and the
        per-node dicts even when it did copy the outer one, so a caller
        clamping a value in the dict it was given rewrote the running
        simulation.  The arrays are immutable and stay shared; only the
        dicts are new, which is one small allocation per node per step.
        """
        return {
            name: dict(fields) if type(fields) is dict else fields
            for name, fields in full_state.items()
            if name != _graph_specs._META_KEY
        }

    def coupling_diagnostics(self) -> dict[str, dict]:
        """Return coupling convergence info from the last step.

        On a multi-rate graph a coupling group solves only on the base
        steps its rate divider fires on, and between those its entry is
        the most recent applied solve's.

        Under ``convergence_norm="interface"`` "the state this step
        returned" is, in every entry below, **the iterate the loop
        accepted**.  The step returns that iterate with each floating
        field the norm does not measure whole -- one no internal edge
        reads, or one read only through a mapping or a transform --
        recomputed by one plain pass at it (not counted in
        ``iterations``), so the state held afterwards is within the
        reported residual of the reported iterate on what the internal
        edges deliver, and is not itself an iterate of the loop.  Of the
        state held, in the norm taken at it, ``spectral_error_bound``
        reads ``(bound + residual) / (1 - rtol * residual * sqrt(N))``
        with ``N`` the entries the norm pools (nothing where ``rtol *
        residual * sqrt(N) >= 1``: a cap reached with a reading still
        moving by its own size); ``gradient_relative_error_bound`` is of
        the state held as it stands, every returned field carrying the
        implicit derivative of the reported iterate.

        Returns
        -------
        dict
            Keyed by coupling group identifier (sorted node names
            joined by ``"+"``), each containing:

            - ``"iterations"`` : int — coupling passes used to produce
              the state this step returned, counting the first
              staggered pass.  Equal to ``max_iterations`` exactly when
              the group exhausted its budget, whichever solver ran, so
              ``iterations >= max_iterations`` is a usable cap check.
              With ``waveform_iterations > 1`` on a group that
              sub-cycles, every sweep is a fixed-point solve with a
              budget of ``max_iterations`` of its own (the sweeps are
              restarts of one solve, MADD-ANO-027), and
              this is the **largest** sweep's count, so the same check
              reads "some sweep exhausted its budget", exactly.  The
              passes the step ran in all are ``"total_iterations"``.
              (Before 0.4.0 it was the last sweep's count, which read
              ``1`` beside an earlier sweep stopped at the cap:
              MADD-ANO-026.)
            - ``"total_iterations"`` : int — coupling passes the step
              ran, summed over its waveform sweeps: the work done, where
              ``"iterations"`` is the cap check.  Equal to
              ``"iterations"`` for a group that runs one sweep -- every
              group that does not sub-cycle, and
              ``waveform_iterations=1`` -- and at most
              ``waveform_iterations * max_iterations``.  Like
              ``"iterations"`` it counts the passes that produced the
              state, not the one extra evaluation a sweep stopped at the
              cap spends measuring its residual.
            - ``"residual"`` : float — ``||F(x) - x||`` in the group's
              convergence norm for the state ``x`` this step returned.
              At ``max_iterations=1`` it is the distance the single
              pass moved, which is the same thing measured one pass
              earlier.  ``inf`` on a non-finite state, whatever the
              norm evaluates to there -- NaN included (see
              ``"converged"``).  **It carries a floating-point noise floor**,
              and near the fixed point that floor is the whole value:
              see the note on ``solver`` below.
            - ``"amplification"`` : float — the estimated
              ``1 / (1 - rho)``, with
              ``rho = max(r_k / r_{k-1}, sqrt(r_k / r_{k-2}))`` taken from
              the last three residuals: the worse of the one-step ratio
              and the two-step rate, so an alternating (non-normal)
              sequence cannot flatter it
              (:func:`~maddening.core.coupling.acceleration.error_amplification`).
              ``rho`` is the rate of the mode that dominates the *step*,
              which need not be the slowest mode (see
              ``"ratio_usable"``).  ``nan`` when the estimate was
              rejected.
            - ``"error_estimate"`` : float — ``residual * omega *
              amplification``, an estimate of ``||x - x*||`` in the
              same norm: how far the returned state is from the fixed
              point, rather than how far the last pass moved.
              ``omega`` is the relaxation factor under
              ``acceleration="fixed"`` and 1 otherwise — the series is
              over the steps the iterate takes, and over-relaxation
              makes those longer than the residual that is measured.
              Falls back to ``residual`` when the estimate was
              rejected.  **It is an estimate, not a bound**: it can
              understate, and by large factors — see ``"ratio_usable"``
              and
              ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
            - ``"ratio_usable"`` : bool — **whether the contraction
              ratio was usable this step, and nothing more.**  The
              estimate above rests on four conditions; this flag checks
              exactly one of them, the fourth.

              *What ``True`` means*: ``rho < 1``, the predecessor
              residual was non-zero, and every residual in the ratio
              was finite — so ``1/(1 - rho)`` is a number worth
              extrapolating from, and the criterion used the estimate.

              *What ``True`` does not mean*: it does **not** certify
              that the estimate bounds the distance to the fixed point.
              Three further conditions are unchecked — the measure
              obeying the triangle inequality, ``rho`` being at least
              the *asymptotic* rate rather than the rate of the mode
              that happens to dominate the step, and the step scale
              being the one actually applied (uncorrected under
              ``"aitken"`` and ``"iqn-*"``).  Each is measured broken
              in 0.4.0; the worst is a **122x** understatement with
              ``ratio_usable=True`` and ``converged=True``.  See
              :func:`_fixed_point_while` for all four.

              ``False`` on a non-monotone (non-normal) sequence, on a
              zero or non-finite predecessor, and at
              ``max_iterations=1``, where there is no pair of residuals
              to take a ratio of.  The criterion then falls back to the
              raw residual test, which is what ``converged`` reports.
              **``False`` does not mean the sequence stopped
              contracting.**  Each residual carries about its float
              floor of rounding, so the ratio carries about
              ``2 floor / residual``, and a group contracting at a rate
              near 1 reads ``rho >= 1`` from noise once its residual is
              within about ``2 floor / (1 - rho)`` of the floor: a
              monotone single-mode group at rate 0.995 stopped this way
              up to 200 thresholds from its fixed point -- wherever its
              raw residual already met the threshold -- with
              ``precision_limited=False`` (which reads the residual
              against its floor, not the ratio).  Read
              ``ratio_usable=False`` beside ``converged=True`` as the
              raw residual test and nothing more (MADD-ANO-005).

              *Renamed during 0.4.0's development* from
              ``"bound_valid"``, which asserted all four conditions while
              checking one; no release carried the old name.  It is
              still readable through 0.4.x, warns, and is removed in
              0.5.0.
            - ``"gradient_error_estimate"`` : float — how far the IFT
              adjoint may be from a finite difference of this group's
              own forward, ``residual * cond(I - dF/dx)`` estimated
              along the observed slowest mode (numerically the same
              number as ``"error_estimate"``: both are
              ``(I - dF/dx)^-1`` applied to a residual).  ``inf`` when
              ``ratio_usable`` is ``False`` — no contraction was
              observed, so nothing constrains the disagreement.  Being
              the same number, it inherits every way
              ``"error_estimate"`` can understate, which is why it is
              an estimate and not a bound.  *Renamed in 0.4.0* from
              ``"gradient_error_bound"``, on the same alias terms as
              ``"ratio_usable"``.
            - ``"converged"`` : bool — the *error estimate* met the
              group's threshold (``tolerance`` for the L2 norm, ``1.0``
              for the mixed / interface norms), compared in the
              residual's dtype with the threshold rounded to it -- the
              comparison the loop, ``strict_convergence`` and the sysid
              mask make in-graph
              (:func:`~maddening.core.coupling.acceleration.reported_converged`),
              so all of them give one verdict on every solve (with
              ``waveform_iterations > 1`` this flag is the last sweep's,
              while ``strict_convergence`` raises about any sweep that
              stopped unconverged at the cap: see below); ``"error_estimate"`` is
              the float32 estimate of a float32 group, not the float64
              product of its factors.  ``False`` means one of two
              things: the group hit ``max_iterations`` *and* the state
              it returned is still outside the threshold, or the state
              it returned is not finite (below) -- which can come
              *before* the cap, so ``iterations < max_iterations`` beside
              ``converged=False`` is not a contradiction: under
              ``convergence_norm="interface"`` a field no edge reads can
              be NaN while the fields the norm reads converge, and the
              loop stops on its criterion.  Under ``solver="ift"`` the
              gradient through that step is unreliable either way.
              **A non-finite state is the one exception to reading the
              slots literally, by design** (MADD-ANO-019): when any
              floating field of the group is NaN or inf, or beyond the
              magnitude its dtype can measure a change at, the step
              records ``residual=inf`` -- not the NaN the norm would
              evaluate to -- so ``"error_estimate"`` is ``inf`` and
              ``converged`` is ``False``, and recomputing the norm from
              the returned state does not reproduce the stored
              residual.
              With ``waveform_iterations > 1`` it is the **last** sweep's
              verdict, which is the verdict on the returned state: every
              sweep iterates the same one-pass map from where the one
              before it stopped, and the ``"ift"`` gradient is the last
              sweep's (the implicit-function derivative does not depend
              on the initial guess).  An earlier sweep stopped at the
              cap shows in ``"iterations"``, not here.
              ``strict_convergence=True`` under ``solver="ift"`` checks
              every sweep, so such a step raises instead.
              **``True`` on a stalled float32 iterate.**  When
              ``(1 - rho) * |x - x*|`` falls below half an ulp a pass
              changes nothing, the residual is exactly ``0.0``, and
              ``converged`` -- and ``strict_convergence`` -- report
              success although the state can be far from its fixed
              point: 38 348 ulps (a relative distance of 4.2e-3,
              against a tolerance of 1e-6) on a relay contracting at
              0.99999, after one pass.  The criterion is deliberately
              not changed for this (a precision floor in it would make a
              tight float32 tolerance unreachable); the bound keys below
              are where it shows -- ``"spectral_error_bound"`` carries
              the residual's float resolution and
              ``"precision_limited"`` is ``True`` -- so read them, with
              ``diagnostics=True``, before trusting ``converged`` on a
              slow group.  (Planned for 0.5.0, not a promise of this
              release: ``strict_convergence`` consulting the spectral
              bound when diagnostics are on.)
            - ``"rho_spectral"`` : float — the spectral radius of
              ``dF/dx`` at the returned state, from eight Arnoldi steps
              on the Jacobian-vector product the IFT adjoint already
              builds (:func:`~maddening.core.coupling.acceleration.arnoldi_spectral_radius`).
              Unlike ``"amplification"``'s ``rho``, which is read off
              the residual sequence and reports the mode dominating the
              *step*, this sees every mode whatever its current
              amplitude: on the two-mode case where the sequence reads
              0.2 it reads 0.999.  **Where** ``"spectral_usable"``
              **is** ``True`` **and no more than eight scalars cross
              the group's edges, it is within 5% of**
              ``1 - rho_spectral`` **of the radius**: that is the
              margin the flag holds the analysis to.  For a group
              whose Jacobian has rank at most eight -- rank is at most
              the number of boundary scalars crossing the group's edges
              -- it is the radius of a matrix within the analysis
              dtype's rounding of the Jacobian (in the norm's weights):
              exact to that rounding for a normal Jacobian (1e-9 and
              better in float64), while a non-normal one can turn a
              rounding into more, which is measured and withdraws the
              flag where it exceeds the margin (a float32 hub with a
              field 1e-4 of what drives it).  An estimate otherwise,
              which ``"spectral_usable"`` reports (there the flag says
              that the Arnoldi residual and a ninth Krylov vector's
              movement of the radius are within the margin, which
              bounds the error for a normal Jacobian only: twelve
              non-normal float32 scalars read 0.273 for 0.219 with the
              flag set): from below for a
              normal ``dF/dx`` (in the norm's weights), from either side
              for a non-normal one, whose Ritz values can lie outside
              the spectrum's convex hull (1.17 on a Jacobi ring of nine
              relays whose every eigenvalue has modulus 0.95).  For a
              bfloat16 or float16 group the analysis and the value
              reported are float32, but the Jacobian-vector products are
              the map's own, rounded to the group's dtype, so the radius
              is exact to about one ``eps`` of that dtype times the
              Jacobian's norm (0.0036 on a bfloat16 pair of radius 0.90,
              under half a bfloat16 ``eps``) rather than to float32.  **Only
              under ``solver="ift"`` with ``diagnostics=True``**; NaN
              for ``"fori"``, for ``diagnostics=False``, at
              ``max_iterations=1`` (no fixed point was solved) and on
              a non-finite state (a Jacobian there describes nothing).
              Costs nine Jacobian-vector
              products per group per step, which is why it is gated.
            - ``"spectral_error_bound"`` : float — ``residual`` plus
              its own float resolution
              (:func:`~maddening.core.coupling.acceleration.residual_precision_floor`:
              four units of ``eps * max|field|`` in every entry the norm
              reads *per evaluation* of the map a coupling pass rounds
              like -- each node's sub-cycling divider times its
              :meth:`~maddening.core.node.SimulationNode.update_evaluations`,
              the worst node's under Jacobi and, under Gauss-Seidel, the
              sum along the longest chain of same-pass reads (a node
              reading a member scheduled before it reads that member's
              already-rounded output; ``_group_evaluations``); with
              ``diagnostics=True`` every read weighted by its relative
              gain measured at the returned state, so a link that
              amplifies a rounding (``u**2``) or a node whose terms cancel
              counts for what it does, never below that structural count
              (see :data:`~maddening.core.coupling.acceleration.PRECISION_FLOOR_ULPS`).
              The count is the one the step was built with and measured,
              not re-derived from the graph at report time: an edit made
              since -- a node rebuilt with another declared count -- does
              not move the report of a step that already ran, and a group
              a member of which was removed since has no entry.  Under
              ``convergence_norm="interface"`` the entries are what that
              norm reads on each internal edge: the value the edge
              delivers, or the source field itself where a static mapping
              delivers more entries than the source holds.  A value
              delivered through an interface mapping
              depends on the mapping weights the step ran with
              (``params["mappings"]``, which a caller may override for one
              step): the step measures that group's floor itself and the
              report reads it, so a ``params`` override, or an edit of the
              weights since, does not move it either -- measured
              in that norm), times the larger of
              ``||(I - H)^{-1}||_2`` (the resolvent norm of the
              Krylov-compressed Jacobian, in the group's own norm) and
              ``1 / (1 - rho_spectral)`` with a margin for an unresolved
              Krylov space
              (:func:`~maddening.core.coupling.acceleration.spectral_error_bound`).
              The resolution is *added*, because a computed residual
              differs from the exact map's by the map's evaluation
              error: without it a stalled float32 iterate (see
              ``"converged"``) read a bound of ``0.0`` thousands of ulps
              from its fixed point, and with it the bound there is what
              float32 can resolve about a group that slow.  The Krylov
              space is continued from the residual itself whenever it
              breaks down, so the residual the resolvent is applied to
              provably lies in the invariant space the resolvent was
              measured on -- a space from one start vector broke down
              early where an eigenvalue was repeated, and the bound read
              0.92x the true distance there -- and a dead-banded field
              keeps a positive weight in the spectrum (its own
              magnitude's), so a small field on the coupling loop no
              longer cuts the
              loop out of it (``rho_spectral`` read 0.0 for a radius of
              0.9, and the bound 0.15-0.29x the true distance of a field
              the norm keeps); the dead-banded fields' share of the
              residual, which ``"residual"`` does not contain, is
              measured and folded into the factor.
              For a linear ``F`` the error of *any* iterate is
              ``(A - I)^{-1}`` of its residual, whatever the step
              sequence, relaxation or accelerator did -- so this is a
              **bound** on ``||x - x*||`` where ``"error_estimate"`` is
              an estimate, and it needs none of the four conditions
              above: measured 8x *over* the true distance on the
              two-mode case (``"error_estimate"``: 122x under), 1.3x
              over under ``aitken`` and ``iqn-ils`` (2x and 1000x
              under), and never below the true distance on random
              normal contractions of dimension 2-6 with negative and
              near-1 eigenvalues.  The resolvent term is what holds on
              a non-normal Jacobian: the Jacobi map of the heterogeneous
              benchmark fixture read 32-42x under with
              ``1/(1 - rho_spectral)`` alone, and 2.9-6.5x over with
              it -- at the price of being loose elsewhere on that map
              (up to 95x over, and 911x on a precision-limited step
              whose floor is the bound).  **What it still is not**: for a
              non-linear ``F`` it is asymptotic (Ostrowski) -- exact to
              float32 on a log map within ``tolerance`` of its fixed
              point, an estimate far from one; it is in the group's
              norm at the returned state -- each field divided by its
              own ``max|field|`` *at the returned state*, which is what
              the resolvent factor is measured in, and the residual it
              multiplies is put in those weights too: ``"residual"``
              divides each field by the larger of its magnitude at the
              returned state and after one more pass, and until 0.4.0's
              round-5 fix the bound used it as it was, which on a group
              still growing toward its fixed point read 0.94x the true
              distance in the returned state's weights, usable
              (MADD-ANO-146) -- so the dead band's excluded
              fields are outside what it bounds and the norm's scale
              drifts with the iterate exactly as it does for
              ``"residual"``.  Under ``convergence_norm="interface"``
              the "fields" are what that norm reads: the value each
              internal edge *delivers* -- its source value through the
              edge's interface mapping, with the weights the step ran
              with, and then its transform -- or, where a static mapping
              delivers more entries than its source field holds, that
              source field itself (the compact side: a group's verdict
              does not depend on the size of a grid a few values are
              scattered onto), each over its own magnitude,
              and the spectrum is taken on that reading (the Jacobian
              of the reading's own iteration, ``Phi' G'`` for ``F = G o
              Phi``, applied through state tangents, so no mapping or
              transform is inverted) -- taken on the raw source fields
              instead, the bound multiplied a residual in one set of
              coordinates by a resolvent in another and read
              0.0014-0.098x the true distance, usable, with an offset
              (a unit conversion) or ``"extract_last"`` on an edge; and
              while the norm and the analysis left an edge's mapping
              out, residual and bound alike described the source field
              and not what the target is handed (0.064-0.318x the true
              distance in the delivered values, usable, with the
              selection written as a mapping).  The norm counts a
              field once for every internal edge that reads it, and
              so does the reading: with the fields counted once each
              instead, the bound read 0.26-0.73x the true distance,
              usable, on a star whose hub's field every leaf reads
              (MADD-ANO-213).  A transform that is
              not affine makes the map non-linear in the reading, with
              the asymptotic reading above.  And its float floor is a model of the
              map's rounding (see
              :data:`~maddening.core.coupling.acceleration.PRECISION_FLOOR_ULPS`),
              which a node that cancels catastrophically inside its own
              update can exceed, and so can one that sub-steps inside
              ``update`` without declaring it -- see
              ``"spectral_usable"`` for where that is caught.  ``inf`` when ``rho_spectral`` (with
              margin) is at or above one; NaN where ``"rho_spectral"``
              is, and on a non-finite state.  It is reported, not
              applied: ``"converged"`` and the iteration counts are
              exactly what they were.
            - ``"spectral_usable"`` : bool — the bound above is finite
              and the Arnoldi space had settled: the Krylov space
              closed within the eight steps, any direction the
              breakdown test discarded as rounding is at most
              5% of ``1 - rho_spectral``, and no rounding of the size
              one more Jacobian-vector product measures can move the
              radius by that much
              (:func:`~maddening.core.coupling.acceleration.spectral_rate_settled`).
              ``False`` where nothing was computed (see
              ``"rho_spectral"``), where the bound is ``inf``, for
              a group whose Krylov space is still growing at the cap
              -- more than seven independent interface scalars, or
              more than eight where they are the whole state; there
              ``"rho_spectral"`` is an estimate (from below only for a
              normal ``dF/dx``) and the bound carries only the margin --
              where the residual never entered the Krylov space
              (its outside fraction is reported as unresolved), and
              where the residual is at its float floor
              (``"precision_limited"``) while a node in the group has
              not declared
              :meth:`~maddening.core.node.SimulationNode.update_evaluations`.
              There the floor *is* the bound, and it rests on how many
              evaluations the pass rounds like, which nothing outside the
              node can see: a relay whose node takes 15-200 explicit
              Euler sub-steps inside ``update`` read 0.07-0.96x its true
              distance with this flag set.  A group whose nodes all
              declare keeps the flag at float32 convergence -- its floor
              counting the declared evaluations along a Gauss-Seidel
              pass's longest chain of same-pass reads, without which a
              stalled 32-relay ring read 0.51x its true distance with the
              flag set (MADD-ANO-094).
              **The flag assumes the counted floor is honest, which
              nothing here can check.**  The count trusts each node to
              declare ``update_evaluations()`` truthfully and not to
              cancel inside itself.  Measured on 154 fixture
              configurations (float32 and float64, CPU, the 0.4.0 floor
              study): a node taking 2000 sub-steps per update but
              declaring one evaluation read 0.26-0.60x the true distance
              with this flag set, and a node forming its output as the
              difference of two terms about 1000x its size read
              0.02-0.65x, flag set.  In the other direction the count
              adds gain magnitudes, so mixed-sign Gauss-Seidel chains read
              very conservatively (up to 1e7x the true distance) and
              sub-cycled groups 1e3-1e5x.  A measured, opt-in
              ``diagnostics="rounding"`` level is planned for 0.5.0.
              Like ``"ratio_usable"``, it reports what the code
              checked and nothing more: a settled space has settled
              *somewhere*, and the linearity condition is not checked
              by anything.
            - ``"gradient_relative_error_bound"`` : float — a bound on
              the **relative** error of the IFT gradient that comes
              from the forward stopping at the returned iterate ``x_k``
              instead of the fixed point ``x*``.  The adjoint is solved
              at ``x_k``, and the tangent it returns differs from the
              fixed point's by exactly ``(I - dF/dx(x*))^{-1}`` applied
              to the change, between the two points, of the one-pass
              map's Jacobian-vector product along that tangent.  The
              bound is ``"spectral_error_bound"`` (the distance) times
              the factor that bound applies to a residual (the
              resolvent) times that change per unit distance -- a
              second difference of the Jacobian-vector product along
              the Newton correction ``(I - dF/dx)^{-1} (F(x_k) - x_k)``
              -- taken for one probe per floating constant the group's
              map reads (every parameter, the pre-step states, the
              states of outside nodes it reads), relative to the norm
              of the tangent the adjoint returns for that probe, and
              reported for the worst probe.  Read it as
              ``|g_k - g*| <= bound * |g_k|`` for the gradient with
              respect to one scalar constant, its norms the group's over
              the state's own fields -- under
              ``convergence_norm="interface"`` with a mapping or a
              transform on an internal edge, the source fields the edges
              read, before the mapping and the transform, not the reading
              ``"spectral_error_bound"`` is taken in.  Measured (jaxlib 0.11.0,
              float32) at every cap of a ``max_iterations`` sweep that
              stops the forward early by construction (caps 3-8): never
              below the true error on a concave and a convex map,
              1.1-3.1x it for the parameter whose error is the larger
              and up to 12.6x for the other, which reads its gap to the
              worst probe (1.1-2.1x and 10.5x before 0.4.0's round-6
              Kantorovich terms); 1.35x for a parameter multiplying the
              state of an affine map; 1.19-1.70x for a spring pair's
              stiffnesses and masses (``k = 6000``, ``c = 60``, ``dt =
              0.01``, caps 2-6); 26x on a hidden slow mode.  The distance carries the
              residual's float resolution, as ``"spectral_error_bound"``
              does, so a stalled iterate no longer reads ``0.0`` (it did,
              against true errors of 1.5-3% at ``F'(x*) = 0.999``); where
              the residual is at that resolution and so carries no
              direction, the curvature is taken along a floor-sized
              vector's resolvent image instead of the Newton correction.
              And it carries a Newton-Kantorovich check on how far the
              linearisation at ``x_k`` can be trusted at ``x*``: ``h`` is
              the larger of ``beta * ||(J(x_k + delta) - J(x_k)) delta||
              / ||delta||`` (``beta`` the full resolvent norm at ``x_k``,
              the Jacobian's change along the Newton correction
              ``delta``) and Deuflhard's affine-covariant ``||(I -
              J(x_k))^{-1} (J(x_k + delta) - J(x_k))||``, an operator
              norm on the directions the Jacobian reads (``3 k`` more
              Jacobian-vector products); the bound is multiplied by
              ``1 / sqrt(1 - 2h)`` (the resolvent at the fixed point),
              its distance is at least Kantorovich's radius ``t*``, each
              probe adds the Newton step's second-order miss (``beta *
              ||secant|| * (t* - ||delta||) / (||delta|| * ||t_k|| *
              sqrt(1 - 2h))``: the change of its linearisation over the
              part of ``x* - x_k`` the Newton step misses, at the rate
              its own secant shows), and it is ``inf`` -- unusable -- at
              ``h >= 1/2``, where nothing measured at ``x_k`` bounds the
              resolvent at ``x*``.  Uncorrected it read 0.20-0.96x the
              true error, flag ``True``, at ``F'(x*) = 0.99`` with the
              forward 0.65-4.5% short; ``h`` there is 0.48-0.58, so two
              of those four now read ``inf`` and two hold at 3.3-4.0x.
              With ``h`` along ``delta`` alone and no second-order term
              it read 0.986x the true error at ``max_iterations=2``
              (``h = 0.19``) and 0.81x on a pair whose ``h`` read 0.37
              along ``delta`` and is 0.65 in the affine-covariant form,
              both on bilinear pairs far from their fixed points, flag
              ``True``.  ``h`` is exactly zero on an affine map, and so
              is every correction here.  **Only
              under ``solver="ift"`` with ``diagnostics=True``**; NaN
              for ``"fori"``, for ``diagnostics=False``, at
              ``max_iterations=1``; ``inf``
              or NaN where ``"spectral_error_bound"`` is; NaN where the
              fixed point responds to no constant, where a constant's
              tangent through the group is not finite (that constant
              has no gradient to bound) and where the returned state is
              not finite.  **Only for a constant the pass resolves.**  A
              relative error is a statement about a gradient that has a
              size: a constant is in the bound where moving it by its
              own magnitude (by its array's largest magnitude where the
              entry is zero, by 1 where the array is) moves one pass
              from ``x_k`` by more than the pass's float resolution, the
              floor ``"spectral_error_bound"`` adds to the residual,
              measured as that bound's distance is; and where its
              tangent through the group is not exactly zero.  A constant
              below that has a gradient that can be what the iterate's
              last rounding left of it, its relative error is then of
              order one or undefined, and the bound says nothing of it
              (it is not the worst probe): the centre or the curve of a
              nonlinearity evaluated on its centre, a term multiplied by
              a field that has converged to zero -- and also a gain or
              a forcing so weak that its whole value moves the pass by
              less than the pass resolves, whose gradient may be exact
              and is not bounded here.  The gradient with respect to such a
              constant is small beside the others in the same units
              (measured: 1.5e-34 for an error of 1.7e-34 where the
              resolved constants' are up to 3.5e3, float64); check a
              constant with one Jacobian-vector product of the pass.
              Costs
              ``11 + 4 k + 5 n_p + 2 k n_p`` Jacobian-vector products per group per
              step beside the spectral bound's eight (plus one
              linearisation and ``k`` reverse-mode products where the
              state has more than ``k`` entries), ``k <= 8`` and ``n_p``
              the number of probes: every entry of a floating constant of
              at most 64 entries, one for a larger constant (see
              ``_gradient_error_bound_at``).

              *Why a bound*: each factor is taken on its conservative
              side -- the distance is the spectral bound, not the
              Newton step's length and never ``"error_estimate"``
              (which reads 100x short on a hidden slow mode, where
              this bound holds); the resolvent factor is the larger of
              the two the spectral bound uses.  *What it rests on*,
              each of which can fail: every condition of
              ``"spectral_error_bound"``; the change in the
              linearisation being linear in the distance and along the
              Newton correction (exact for an affine map, leading-order
              otherwise, which ``h`` above measures along ``delta``
              only); and the probes -- a field-valued constant is
              probed along one random direction (fixed-seed, drawn in
              the order the map reads its constants, which follows the
              build order: the same group built in another order can
              report another bound, 83.5 to 99.5 over the six orders of
              one three-member Jacobi group, each still at least the
              true error), and the bound is
              relative to the tangent's norm in the group's norm, so a
              scalar loss whose gradient nearly cancels across the
              state can carry a larger relative error.  One probe per
              constant rather than one combined probe because a
              combined direction can cancel: on a spring pair it read
              0.0 while the stiffness gradient was 0.8-4.8% off (the
              random signs moved stiffness and mass by the same
              relative amount, and the dynamics see only their ratio).

              *Its own rounding.*  The curvature factor is the
              difference of two Jacobian-vector products of the map,
              and where the curvature along the step is only a few
              float32 ulps of those products the bound carries their
              rounding: two compilations of the same step can disagree
              in its leading digit.  Measured on a pair whose map is
              bilinear in its state and a coupled constant: 2.01e-06
              unbatched and 1.84e-06 under ``jax.vmap`` for the same
              step -- the products one ulp apart between the two
              programs, their difference twelve ulps -- against a true
              relative error of 9.5e-07, so both hold.  That is the
              float32 resolution of the gradient itself (about
              ``amplification * eps``), not a defect of batching.
              Under ``vmap`` the step is XLA's batched program, which
              may associate a reduction or lower a divide differently
              from the unbatched one, so ``residual`` can differ by an
              ulp, ``amplification`` -- a ratio of residuals -- by more
              (470 ulps measured, the residual's ulp magnified by
              ``1 / (1 - rho)**2``), and the spectral keys by a few
              ulps.  The verdict is built from those, so a member whose
              estimate lies within that rounding of the threshold can
              stop on a different pass in a batch than alone (19 against
              20 measured, at a threshold placed an ulp from the
              estimate), and its state then differs by one pass's
              change -- inside the tolerance.  Away from that window the
              forward state and ``iterations`` are bit-identical on
              every configuration measured.

              **This is a statement about the gradient, not about the
              solve.**  On a map that is affine in its state with
              additive parameters the IFT gradient is the fixed
              point's from *any* iterate, so this truthfully reads
              ``0.0`` while the state is far off: 0.0 on a two-mode
              map sitting 1.1e-2 from its fixed point with
              ``converged=True``, and 3.1e-7 -- zero to float32 -- on the
              stiff spring pair under ``iqn-ils`` with the interface
              norm, whose velocities are 1.7% off.  The returned
              ``(value, gradient)`` pair is then mutually inconsistent
              -- the gradient is ``d(fixed point)/dtheta`` and the value
              is not the fixed point -- and nothing in this key says
              so.  For the health of the solve read
              ``"spectral_error_bound"``, within its norm (under the
              interface norm it covers the interface fields only: on
              that spring pair it reads 9.0e-3).  This key answers
              only "how far would tightening the forward move the
              gradient".  It is not ``"gradient_error_estimate"``,
              which is ``"error_estimate"`` under another name.
            - ``"gradient_bound_usable"`` : bool — the bound above is
              finite and ``"spectral_usable"`` is ``True``: its distance
              and its resolvent factor are the spectral bound's, and
              are settled on the same condition.  ``False`` where
              nothing was computed, where the bound is ``inf`` or NaN
              (including a group whose Jacobian range the eight-vector
              basis did not capture, and one that fails the
              Kantorovich check, ``h >= 1/2``).  What it certifies is
              that check and the bound built on it: the Kantorovich
              radius and resolvent with the Jacobian's change across the
              Newton step taken as an operator (affine-covariantly), and
              the Newton step's miss carried at the rate the probes' own
              secants show.  Both rates are measured across the step, so the
              flag rests on the map changing no faster between ``x_k``
              and ``x*`` than across that step -- exact for an affine
              map, and the condition a group far from its fixed point
              (a large ``h``, a large distance) is likeliest to break.
              Like the other flags it reports what the code checked, and
              not the linearity or probe conditions above.
            - ``"precision_limited"`` : bool — the residual is at or
              below its own float resolution (the floor
              ``"spectral_error_bound"`` adds): the last pass moved no
              entry by more than a few ulps of its field's magnitude, so
              ``"residual"`` and ``"error_estimate"`` are rounding
              rather than motion, at least half of each bound key is the
              floor, and neither iterating further nor tightening the
              tolerance can reduce the bounds by more than half -- only
              a wider dtype can.  Reported for every group, whatever the
              solver.  ``False`` for a non-finite residual and for a
              group whose norm reads no field.  It clears
              ``"spectral_usable"`` only in a group with a node that has
              not declared its evaluation count (see there); with every
              count declared the bound with its floor *is* a bound, and
              a group converged to float32 -- the best a float32 group
              can do -- stays usable.
              The floor is measured on the state the step returned,
              and stays so when the state is written afterwards
              (``set_node_state``, ``PUT /graph/state``, a node replaced):
              the entry describes the step that ran until the group steps
              again, the state is reset or a member is removed.  A
              checkpoint is a copy of the state, ``_meta`` included, so
              one saved after such a write to a member holds the written
              state and not the returned one; it carries a marker saying
              so, and the graph that loads it reports the group with
              ``"spectral_error_bound"`` NaN, ``"spectral_usable"``,
              ``"gradient_bound_usable"`` and ``"precision_limited"``
              ``False`` and a ``"not_usable_reason"``, until the group
              steps.  The entry's other numbers are the slots' own.

            ``converged=True`` is a statement about the state this step
            returned: both solvers stop on the iterate whose residual
            met the criterion rather than on the update it went on to
            produce, so recomputing ``||F(x) - x||`` on the state you
            were handed reproduces ``"residual"``.  Since 0.4.0 it is
            also an *estimate of the distance to the fixed point* and
            not only of the last step: the threshold is applied to
            ``omega * residual / (1 - rho)`` with ``rho`` measured from
            the residual sequence, which is what MADD-ANO-005 recorded
            as missing.  Where the ratio is unusable the flag degrades
            to the old residual test and says so through
            ``"ratio_usable"``.  It is strictly stronger than the
            pre-0.4.0 flag in every case and still not a guarantee —
            do not treat ``converged=True`` as certifying a distance,
            and in particular not on a stalled float32 iterate (see
            ``"converged"`` above and ``"precision_limited"``).

            **A group that resolves a geometry-dependent mapping**
            (experimental: an edge into a member, from inside the group
            or outside it, added with ``add_edge(..., geometry=...)``)
            reports everything above as any other group does where
            every such mapping is a ``multilinear_grid``, the group's
            ``convergence_norm`` is ``"l2"`` or ``"mixed"`` and the
            group does not sub-cycle; with ``diagnostics=True`` its
            step then compares its own Jacobian-vector product along
            the positions with a finite difference of the pass, and
            the bounds stand where the two agree to
            ``GEOMETRY_GAP_TOLERANCE``.  Any other such group, and one
            whose step failed that check, reports the solve's own
            ``"iterations"``,
            ``"total_iterations"``, ``"residual"`` and ``"converged"``
            and nothing else of the above: the diagnostics do not read
            its moving geometry in 0.4.0, so ``"amplification"``,
            ``"error_estimate"``, ``"rho_spectral"``,
            ``"spectral_error_bound"`` and
            ``"gradient_relative_error_bound"`` are NaN,
            ``"gradient_error_estimate"`` is ``inf``, and
            ``"ratio_usable"``, ``"spectral_usable"``,
            ``"gradient_bound_usable"`` and ``"precision_limited"`` are
            ``False``.  Such an entry has one more key,
            ``"not_usable_reason"`` : str, which names the edges and
            says which case it is; no other group's entry has it.  The values are
            withheld **here**: the internal ``_meta`` entry of the state
            (which ``GET /graph/state`` of the REST server and an FMU
            state archive carry verbatim) still holds what the step
            itself computed for such a group, and those raw slots are
            not a report and promise nothing.

            **After** :meth:`run_adaptive` **or** :meth:`run_adaptive_scan`
            the report describes the last accepted attempt's two kept
            half steps, the two solves ``strict_convergence`` checks:
            ``"iterations"`` is the larger half's count (so the cap check
            reads "a kept solve exhausted its budget"),
            ``"total_iterations"`` the sum where the group owns that slot
            (``waveform_iterations > 1``; a one-sweep group has none and
            reports ``"iterations"``), and every per-solve key --
            ``"residual"``, ``"amplification"`` and what is derived from
            them, the spectral keys, the gradient bound -- comes from one
            half: the first when it alone did not converge, else the
            second, which produced the returned state.  ``"converged"`` is
            therefore ``False`` exactly when a kept solve did not
            converge, the verdict ``strict_convergence`` acts on.  (It
            used to be the second half's report alone, which said
            ``converged=True`` beside a first half stopped at the cap:
            MADD-ANO-095.)

            ``"ift"`` (the default) and the legacy ``"fori"`` run the
            same passes, stop on the same pass and derive every value
            here by the same rule -- the three spectral keys and the
            two gradient-bound keys excepted,
            which ``"fori"`` has no linearisation to compute and
            reports as NaN / ``False`` -- so migrating a graph between
            them does not move the answer or the verdict.  The returned
            state agrees to float32 round-off -- bit-identical on most
            graphs, and 7.3e-07 relative in the worst of 480
            configurations of a subcycled multi-rate group, none of
            which disagreed on ``converged``.

            **The reported ``"residual"`` is the one number that moves
            further than that, and only by float32 round-off.**  Every norm
            here divides ``F(x) - x`` by a scale, so it is a
            *cancellation*: near the fixed point the numerator is the
            difference of two nearly equal float32 states, and one unit
            in the last place of either is a full-size contribution to
            it.  The two solvers run their passes in different loop
            constructs -- ``"ift"`` in a ``lax.while_loop`` so it can
            exit early, ``"fori"`` in a ``lax.fori_loop`` -- which XLA
            is free to compile to differently rounded arithmetic, and
            it does: one ulp on a couple of components of the map's
            output is enough to move a residual of 1e-05 to ``0.0``.
            So ``"residual"`` agrees between the solvers to about
            ``eps_float32 / rtol`` under the mixed norm and
            ``eps_float32 * sqrt(n)`` under the L2 norm, which is the
            measurement's own resolution and not a bound on anything
            physical.  A residual at that floor means "converged to
            float32", and comparing two of them -- across solvers,
            across JAX versions or across backends -- compares rounding.
            Above that floor, ``converged``, ``iterations`` and the
            returned state agree between the solvers.  Pinned by
            ``tests/core/test_coupling_solver_equivalence.py``.  **At or
            below it they can disagree too.**  With a ``tolerance``
            finer than the norm's float32 resolution, "converged"
            means exact stationarity, and the two loops' rounding
            decides which pass reaches it.  Measured: on the
            mixed-rate spring pair at ``tolerance=1e-8`` (L2 norm,
            O(1) fields), 14 ``"ift"``/``"fori"`` pairs disagreed on
            ``converged`` from the third step on, out of 308
            configurations.  If the two solvers' verdicts must
            agree, choose a tolerance above the floor.

            Reported for every group under ``solver="ift"``; ``"fori"``
            groups only with ``diagnostics=True``.  The three spectral
            keys and the two gradient-bound keys are present for every
            group and carry a value only under ``solver="ift"`` with
            ``diagnostics=True``.  A group that has not taken a step
            yet -- before the first ``step()``, and again after
            ``reset_state()`` -- has no entry, so the dict is empty
            until something has run.
        """
        # A transform that stepped the graph (``jax.grad`` of a loss
        # calling ``run_scan``) leaves tracers in it; reading ``_meta``
        # then raised ``UnexpectedTracerError``.  Put back first, like
        # every other entry point.
        self._recover_from_escaped_tracers()
        meta = self._state.get(_graph_specs._META_KEY, {})
        result: dict[str, dict] = {}
        for current in self._coupling_groups:
            key = "+".join(sorted(current.nodes))
            # Judged under the group the slots were written under: the one
            # the compiled step was built from.  A replacement registered
            # since has not stepped, and re-deriving the last step's
            # verdict under *its* criterion described a step it never
            # judged.  ``compile()`` restarts the slots of a replaced
            # group, so after a recompile the two agree.
            group = self._committed_coupling_groups.get(key, current)
            iter_key = f"coupling_{key}_iterations"
            res_key = f"coupling_{key}_residual"
            amp_key = f"coupling_{key}_amplification"
            # ``compile()`` and ``reset_state()`` seed the counter at 0
            # and every coupled step reports at least one pass, so 0 is
            # "no step taken yet".  The seeds beside it (a residual of
            # 0.0, a rejected amplification) are there to keep the scan
            # carry's structure, not to be read: reported, they said
            # ``converged=True`` about a group that had never run.
            if iter_key in meta and int(meta[iter_key]) > 0:
                iterations = int(meta[iter_key])
                # Only a group running more than one waveform sweep owns
                # the sum's slot; with one sweep the sum *is* the count.
                # It is never below the largest sweep's count, which it
                # contains: a slot still at its seed beside a counter that
                # is not -- seeded after the counter was written, by a
                # recompile that turned on ``waveform_iterations`` or a
                # checkpoint from before the slot existed, and read before
                # the next step -- reads as that count.
                total_iterations = max(
                    int(meta.get(f"coupling_{key}_total_iterations", iterations)),
                    iterations,
                )
                # The slots as stored, in the dtype the solve computed
                # them in: the estimate and the verdict below are taken in
                # that dtype, as the loop took them (see
                # ``reported_converged``).
                residual_slot = np.asarray(meta[res_key])
                amp_slot = np.asarray(meta.get(amp_key, 0.0))
                residual = float(residual_slot)
                amp = float(amp_slot)
                # A valid amplification is ``1/(1 - rho)`` with
                # ``rho`` in ``[0, 1)``, so it is always >= 1; the
                # solvers write 0.0 for "rejected".
                valid = amp >= 1.0
                # The geometric series is over the steps the iterate
                # takes, which are ``relaxation`` times the residual
                # that is measured under ``acceleration="fixed"``.
                # Both solvers apply the same factor to the same
                # criterion, so this reproduces their ``converged``; the
                # profiler and ``sysid`` read the same criterion.
                threshold, scale = convergence_criterion(group)
                error_estimate = reported_error_estimate(residual_slot, amp_slot, scale)
                converged = reported_converged(residual_slot, amp_slot, scale, threshold)
                # The spectral triple (Ritz radius, Arnoldi residual,
                # resolvent norm) is present only under solver="ift"
                # with diagnostics=True,
                # and NaN there until a step has
                # computed it (and at max_iterations=1, which solves no
                # fixed point).  Absent or NaN both read as "not
                # computed": the bound is NaN and the flag False.
                rho_spec = float(meta.get(
                    f"coupling_{key}_rho_spectral", float("nan")))
                spec_resid = float(meta.get(
                    f"coupling_{key}_spectral_residual", float("nan")))
                spec_amp = float(meta.get(
                    f"coupling_{key}_spectral_amplification", float("nan")))
                # The residual's own float resolution at the returned
                # state: a computed residual differs from the exact
                # ``F(x) - x`` by the map's evaluation error, so a
                # distance bound has to add it -- a stalled float32
                # iterate reads ``residual=0.0`` arbitrarily far from
                # its fixed point.  Taken from the state the step left,
                # by the norm's own field rule.
                # A pass that sub-cycles a node, or whose nodes loop inside
                # ``update``, rounds like several single ones: the floor is
                # per evaluation (``_group_evaluations``).
                #
                # Taken from what the compiled step was built from
                # (``_committed_floor_inputs``, snapshot at ``compile()``),
                # not the live graph: an edit since -- the remove-and-re-add
                # recipe with a node declaring another count -- moved the
                # bound of a step that had already run (17x measured), and
                # a removed member raised ``KeyError`` here.  Where the step
                # measured the count with its same-pass reads gain-weighted
                # (``pass_evaluations``, ``diagnostics=True`` under ``"ift"``)
                # that is the count.
                committed = self._committed_floor_inputs.get(key)
                if committed is None:
                    # Slots without a snapshot: not written by a step this
                    # graph's ``compile()`` built.  Nothing to judge them by.
                    continue
                evaluations, declared, internal_edges = committed
                measured = float(meta.get(f"coupling_{key}_pass_evaluations", float("nan")))
                if math.isfinite(measured):
                    evaluations = max(evaluations, measured)
                if any(nn not in self._state for nn in group.nodes):
                    # A member removed since the step: its state, which the
                    # floor is measured on, is gone, and so is this report.
                    continue
                # Where the group's norm reads an edge through its mapping
                # (as delivered; an edge read at its source is the stored
                # field), what that edge delivers depends on the mapping
                # weights the step ran with (``params["mappings"]``, which
                # a caller may override for one step), so the step measured
                # the floor per evaluation
                # itself (``reading_floor``) and the count multiplies it in
                # the slot's own dtype, as the function does.  A slot that
                # is absent or was never written (a state this build's
                # step did not produce) falls back to the graph's own
                # weights, which is what a ``params=None`` step runs with.
                measured_floor = meta.get(f"coupling_{key}_reading_floor")
                if measured_floor is not None and np.isfinite(np.asarray(measured_floor)):
                    unit = np.asarray(measured_floor)
                    floor = float(unit * np.asarray(evaluations, unit.dtype))
                else:
                    # On the state the step left, not on whatever has been
                    # written to it since (``_keep_state_for_reports``).
                    floor = float(residual_precision_floor(
                        self._state_a_report_describes((iter_key, res_key), group.nodes),
                        sorted(group.nodes), group.convergence_norm,
                        group.atol, group.rtol, list(internal_edges),
                        evaluations=evaluations,
                        mappings=(self._params or {}).get("mappings"),
                    ))
                spectral_bound = float(spectral_error_bound(
                    residual, rho_spec, spec_resid, spec_amp, floor=floor,
                ))
                precision_limited = bool(
                    floor > 0.0 and math.isfinite(residual) and residual <= floor
                )
                # Where the residual is at its floor the floor *is* the
                # bound, and the floor rests on each node's evaluation
                # count.  Unless every node declared it, that is not
                # checked, and a node that sub-steps inside ``update``
                # made the bound read 0.07-0.96x the true distance.
                spectral_usable = bool(
                    math.isfinite(spectral_bound)
                    and spectral_rate_settled(rho_spec, spec_resid)
                    and (declared or not precision_limited)
                )
                # The gradient's bound is stored ready-made: its
                # ingredients are vectors the state does not keep.  It
                # is usable on the spectral bound's condition -- its
                # distance and resolvent factor come from there -- and
                # its own finiteness.
                grad_bound = float(meta.get(
                    f"coupling_{key}_gradient_relative_error_bound", float("nan")))
                result[key] = _reports._CouplingDiagnostics({
                    "iterations": iterations,
                    "total_iterations": total_iterations,
                    "residual": residual,
                    "amplification": amp if valid else float("nan"),
                    "error_estimate": error_estimate,
                    "ratio_usable": valid,
                    "gradient_error_estimate": (
                        error_estimate if valid else float("inf")
                    ),
                    "converged": converged,
                    "rho_spectral": rho_spec,
                    "spectral_error_bound": spectral_bound,
                    "spectral_usable": spectral_usable,
                    "gradient_relative_error_bound": grad_bound,
                    "gradient_bound_usable": bool(
                        spectral_usable and math.isfinite(grad_bound)
                    ),
                    "precision_limited": precision_limited,
                })
                if self._loaded_after_a_write(key) and not (
                        measured_floor is not None
                        and np.isfinite(np.asarray(measured_floor))):
                    # The checkpoint this report was loaded from was saved
                    # after the group's state had been written: the state
                    # the step returned, which the floor is measured on,
                    # is not in it.  Everything built on the floor is
                    # withheld; the slots' own numbers stand.
                    result[key].update({
                        "spectral_error_bound": float("nan"),
                        "spectral_usable": False,
                        "gradient_bound_usable": False,
                        "precision_limited": False,
                        "not_usable_reason": _group_layout._WRITTEN_BEFORE_SAVE_REASON,
                    })
                geometry_keys = self._committed_geometry_edges.get(key, ())
                geometry_reason = (self._committed_geometry_refusals.get(key)
                                   if geometry_keys else None)
                gap = meta.get(f"coupling_{key}_geometry_gap") if geometry_keys else None
                if geometry_reason is None and gap is not None and math.isfinite(rho_spec):
                    # The step compared its own Jacobian-vector product
                    # along the geometry with a finite difference of the
                    # pass (``_bounds._geometry_product_gap``).  Read only
                    # where the step computed a spectrum to check.
                    if not float(gap) <= _bounds.GEOMETRY_GAP_TOLERANCE:
                        geometry_reason = _group_layout._geometry_self_check_reason(
                            geometry_keys, float(gap), _bounds.GEOMETRY_GAP_TOLERANCE)
                if geometry_reason is not None:
                    # Experimental: where the diagnostics do not read this
                    # group's moving geometry (another mapping kind, the
                    # interface norm, a sub-cycled group), or where the
                    # step's self-check of the geometry term failed, the
                    # group reports the solve's own outcome and nothing
                    # built on the float floor or on the contraction
                    # estimates: every bound is NaN (the gradient estimate
                    # ``inf``, as where the ratio is rejected), every
                    # ``*_usable`` flag False, and the report says why.
                    result[key].update({
                        "amplification": float("nan"),
                        "error_estimate": float("nan"),
                        "ratio_usable": False,
                        "gradient_error_estimate": float("inf"),
                        "rho_spectral": float("nan"),
                        "spectral_error_bound": float("nan"),
                        "spectral_usable": False,
                        "gradient_relative_error_bound": float("nan"),
                        "gradient_bound_usable": False,
                        "precision_limited": False,
                        "not_usable_reason": geometry_reason,
                    })
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(
        self,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Advance the simulation by one base timestep.

        Parameters
        ----------
        external_inputs : dict, optional
            Values injected from outside the graph, structured as
            ``{node_name: {field_name: value, ...}, ...}``.
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.  Passing a modified pytree changes node
            constants for this step without recompiling.

        Returns the full state dict after the step (excluding internal
        metadata).

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        the state this one step reaches, so the next :meth:`step` or
        ``run_*`` call continues from there rather than from the state
        the graph held before this call.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        self._refuse_xla_loop_hazards("step", scan=False)

        step_fn = self._compiled_step
        assert step_fn is not None  # `compile()` above always sets it

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        self._store_stepped_state(self._call_surfacing_strict(
            step_fn, self._state, external_inputs, params))
        user_state = self._user_state(self._state)
        self._notify(EVENT_STEP, user_state)
        return user_state

    def run(
        self,
        n_steps: int,
        callback: Optional[Callable] = None,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> None:
        """Run *n_steps* simulation steps (at the base timestep rate).

        Parameters
        ----------
        n_steps : int
            Number of base-rate steps to execute.
        callback : callable, optional
            Called after every step with ``(step_index, state_dict)``.
            The state dict excludes internal metadata.
        external_inputs : dict, optional
            Static external inputs applied every step.  For dynamic
            inputs that change each step, use :meth:`step` in a loop
            or use a ``CommandReceiver`` with ``RealtimeRunner``.
            Completed and validated as in :meth:`step`.

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        the state the last of the *n_steps* steps reaches, so the next
        :meth:`step` or ``run_*`` call continues from there rather than
        from the state the graph held before this call.  The state moves
        at every iteration, so an exception part-way through leaves the
        graph at the last completed step, not where it started.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        self._refuse_xla_loop_hazards("run", scan=False)

        step_fn = self._compiled_step
        assert step_fn is not None  # `compile()` above always sets it

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        for i in range(n_steps):
            self._store_stepped_state(self._call_surfacing_strict(
                step_fn, self._state, external_inputs, params))
            user_state = self._user_state(self._state)
            self._notify(EVENT_STEP, user_state)
            if callback is not None:
                callback(i, user_state)

    def _strict_mesh(self, coupling_groups=None):
        """The device mesh ``strict_convergence`` raises on every device of, or ``None``.

        ``None`` unless some group checks ``strict_convergence`` under
        ``solver="ift"`` *and* the step spans more than one device
        (:func:`_multi_device_mesh`); see :func:`_strict_error_if`.
        """
        groups = self._coupling_groups if coupling_groups is None else coupling_groups
        if not any(g.strict_convergence and g.solver == "ift" for g in groups):
            return None
        return _graph_specs._multi_device_mesh(self._nodes, self._state,
                                  getattr(self, "_multigpu_mesh", None))

    def _call_surfacing_strict(self, fn: Callable, *args):
        """``fn(*args)``; on a step spanning several devices with a strict group,
        blocked on, with every device drained before an error propagates.

        ``strict_convergence`` raises on every device of such a step
        (:func:`_strict_error_if`); the first device's error reaches Python
        while the others may still be raising, and a process that exits then
        aborts in teardown.  So the call is waited on here, where the error
        surfaces, and the mesh drained (:func:`_drain_mesh`) before it
        propagates.  Elsewhere -- one device, no strict group, or under a
        transform -- this is ``fn(*args)`` and stays asynchronous.
        """
        mesh = self._strict_mesh()
        if mesh is None:
            return fn(*args)
        try:
            out = fn(*args)
            if not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(out)):
                jax.block_until_ready(out)
            return out
        except Exception:
            _graph_specs._drain_mesh(mesh)
            raise

    def _cached_scan(self, key: tuple, build: Callable[[], Callable]) -> Callable:
        """Return the jitted scan program for *key*, building it once.

        Building a scan means tracing the whole in-XLA loop and
        compiling it: hundreds of milliseconds upward.  ``run_scan`` and
        its siblings used to do that on every call, so a caller driving
        a simulation from a Python loop, a slider or an HTTP handler
        paid a compile per call.

        Parameters
        ----------
        key : tuple
            Everything that determines the built program *beyond* the
            graph structure: which entry point, the scan length, and any
            Python value the body closes over (the adaptive controller's
            constants, for instance).  ``_compile_generation`` is
            prepended, which covers the structure -- every graph
            mutation marks the graph dirty, every entry point recompiles
            a dirty graph before it gets here, and ``compile()`` clears
            this cache.
        build : callable
            Zero-argument builder returning the jitted program.  Called
            only on a miss.

        Returns
        -------
        Callable
            The cached (or freshly built) jitted program.

        Notes
        -----
        Shapes and dtypes of the runtime values are deliberately *not*
        in the key.  State, external inputs and params are arguments of
        the jitted program rather than constants closed over by it, so
        JAX's own cache retraces when an aval changes and reuses the
        compilation when it does not.  Baking them into the key instead
        would mean hashing arrays -- and closing over them, as the old
        code did, is what made the rebuild mandatory in the first place.
        """
        full_key = (self._compile_generation,) + tuple(key)
        fn = self._scan_cache.get(full_key)
        if fn is None:
            fn = build()
            # Bounded, because an HTTP handler taking ``n_steps`` from the
            # request would otherwise grow this without limit.  Insertion
            # order eviction: a workload cycling over more than
            # ``_SCAN_CACHE_MAX`` distinct step counts is pathological,
            # and pays what it used to pay on every call.
            if len(self._scan_cache) >= _graph_specs._SCAN_CACHE_MAX:
                self._scan_cache.pop(next(iter(self._scan_cache)))
            self._scan_cache[full_key] = fn
        return fn

    def _xla_loop_candidates(self) -> list[tuple[str, Any, bool]]:
        """``(graph node, wrapper, in a coupling group)`` for every
        ``ShardedStencilNode`` with a sharded static replicated over a mesh
        axis of two or more devices -- the declaration half of the
        MADD-ANO-068 condition, cheap and taken without tracing.

        The wrapper is the outermost ``ShardedStencilNode`` under the graph
        node (through ``HybridNode`` and nesting): it is the one whose
        ``update`` runs, since an inner wrapper only forwards
        ``update_padded``.
        """
        grouped: set = set()
        for group in self._committed_coupling_groups.values():
            grouped |= set(group.nodes)
        out = []
        for name, spec in self._nodes.items():
            wrapper = _graph_specs._outermost_stencil_wrapper(spec.node)
            if wrapper is not None and wrapper._statics_replicated_over_devices():
                out.append((name, wrapper, name in grouped))
        return out

    def _refuse_xla_loop_hazards(self, entry: str, *, scan: bool) -> None:
        """Refuse a path that would put a miscompiled step inside a loop.

        MADD-ANO-068: XLA (jaxlib 0.10.2, 0.11.0 and 0.11.2; not 0.5.3)
        compiles a ``ShardedStencilNode`` step wrongly when it is traced
        inside ``lax.scan``, ``fori_loop`` or ``while_loop`` and the node
        reads a sharded ``StaticArray`` in its halo, reads another array
        through a window at a ``shard_info`` offset, and the static is
        replicated over a mesh axis of two or more devices.  The result is
        silently wrong (every cell 1e-3 to 1e-2 off on a mesh axis the
        ``axis_map`` leaves unused) or fails to compile (a pencil).  The
        step under ``jit`` alone is right.

        ``scan=True`` is an entry point that builds a loop over the graph
        step (``run_scan`` and its siblings, ``sysid``): any such node
        refuses it.  ``scan=False`` is one that does not (``step``,
        ``run``, ``run_adaptive``): only a node inside a coupling group
        refuses it, because the group's iteration -- or a sub-cycling
        group's sub-steps -- runs its update inside a loop within the step
        itself.

        Which nodes read their statics that way is learnt from one trace
        of the step (``jax.eval_shape``, never compiled) under
        :func:`~maddening.cloud.multigpu._scan_hazards.probe_scan_hazards`,
        taken only when a wrapper declares a static replicated that way,
        and kept for the compile generation.  A wrapper the trace did not
        reach is refused, not passed.
        """
        gen = self._compile_generation
        cache = self._xla_loop_hazards
        if cache is None or cache[0] != gen:
            cache = [gen, self._xla_loop_candidates(), None]
            self._xla_loop_hazards = cache
        candidates = cache[1]
        if not any(scan or grouped for _, _, grouped in candidates):
            return
        if cache[2] is None:
            cache[2] = self._probe_xla_loop_hazards(candidates)
        hazards = [(name, grouped, h) for name, grouped, h in cache[2] if scan or grouped]
        if not hazards:
            return
        name, grouped, hazard = hazards[0]
        others = sorted({n for n, _, _ in hazards[1:] if n != name})
        axes = ", ".join(repr(a) for a, _ in hazard.replicated_over)
        # A member of a coupling group is refused on every path, step() too.
        coupled = grouped
        where = (
            f"graph node {name!r} is a member of a coupling group, whose "
            "iteration runs its update inside a loop on every entry point"
            if coupled else
            f"{entry} traces the step of graph node {name!r} inside a loop"
        )
        fixes = (
            f"use a mesh without mesh axis {axes} (a sharded static is split "
            "along one mesh axis and copied along every other one); or "
            + ("take the node out of the coupling group." if coupled else
               "advance the graph with step() / run(), which are not affected.")
        )
        raise RuntimeError(
            f"{entry}: refused (MADD-ANO-068).  {where}, and XLA (jaxlib 0.10.2 to "
            "0.11.2) miscompiles that step there: "
            f"{hazard.describe()}.  Inside a loop the result is silently wrong "
            "(every cell 1e-3 to 1e-2 off) or fails to compile.  Workarounds: "
            f"{fixes}"
            + (f"  Also affected: {others}." if others else "")
        )

    def _probe_xla_loop_hazards(self, candidates) -> list[tuple[str, bool, Any]]:
        """``(graph node, in a coupling group, ScanHazard)`` for each hazard
        found by one ``eval_shape`` trace of the step; see
        :meth:`_refuse_xla_loop_hazards`."""
        from maddening.cloud.multigpu._scan_hazards import (  # noqa: PLC0415
            ScanHazard,
            probe_scan_hazards,
        )
        with quiet_warnings(), probe_scan_hazards() as found:
            # A node that warns at trace time warns on the real trace too.
            jax.eval_shape(self._build_step_fn(), self._state,
                           self._default_external_inputs(), self.params)
        out = []
        for name, wrapper, grouped in candidates:
            record = found.get(id(wrapper))
            if record is None:
                inner = _graph_specs._innermost_wrapped(wrapper)
                for key, axes in sorted(wrapper._statics_replicated_over_devices().items()):
                    out.append((name, grouped, ScanHazard(
                        node=wrapper.name, node_type=type(inner).__name__, static=key,
                        shard_axis=wrapper._sharded_static[key].shard_axis,
                        replicated_over=axes, window="")))
                continue
            out.extend((name, grouped, hazard) for hazard in record[1])
        return out

    def run_scan(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Run *n_steps* using ``jax.lax.scan`` for maximum performance.

        Unlike :meth:`run`, this method pushes the entire loop into XLA
        via ``jax.lax.scan``, eliminating Python-loop and JAX-dispatch
        overhead.  The full computation is JIT-compiled into a single
        XLA program.

        Trade-offs compared to :meth:`run` / :meth:`step`:

        * No per-step callback or observer notifications.
        * External inputs are **static** -- the same values are applied
          at every timestep.  For dynamic per-step inputs, use
          :meth:`step` in a loop or ``RealtimeRunner``.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.
        params : dict, optional
            Graph parameter pytree; ``None`` uses :attr:`params`.

        Returns
        -------
        dict[str, dict]
            The final state of the graph after *n_steps* (excluding
            internal metadata).

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        the final state of the scan -- the same state this call returns
        -- so the next :meth:`step` or ``run_*`` call continues from
        there rather than from the state the graph held before this
        call.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        self._refuse_xla_loop_hazards("run_scan", scan=True)

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        # External inputs and params are *arguments* of the jitted scan,
        # not constants closed over by it: that is what lets the built
        # program be cached across calls whose values differ.
        def build():
            step_fn = self._build_step_fn()

            def scan(state, ext, params):
                self._count_scan_trace()

                def scan_body(carry, _unused):
                    return step_fn(carry, ext, params), None

                final, _ = jax.lax.scan(
                    scan_body, state, None, length=int(n_steps),
                )
                return final

            return jax.jit(scan)

        fn = self._cached_scan(("run_scan", int(n_steps)), build)
        self._store_state(self._call_surfacing_strict(
            fn, self._state, external_inputs, params))
        return self._user_state(self._state)

    def run_scan_with_history(
        self,
        n_steps: int,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict[str, dict]]:
        """Run *n_steps* via ``jax.lax.scan``, returning all intermediate states.

        Like :meth:`run_scan` but also collects the state at every
        timestep into stacked arrays, which is useful for plotting and
        post-hoc analysis without Python-loop overhead.

        Parameters
        ----------
        n_steps : int
            Number of base-rate simulation steps to execute.
        external_inputs : dict, optional
            Static external inputs applied identically every step.
            Zeros are used for every declared input this does not
            supply, ``None`` included; an undeclared ``node.field``
            is a ``ValueError`` naming the declared ones.

        Returns
        -------
        (final_state, history) : tuple
            *final_state* is the state dict after the last step
            (same as :meth:`run_scan` would return), excluding internal
            metadata.
            *history* has the same nested-dict structure as state (also
            excluding internal metadata), but each leaf is a JAX array
            with an extra leading axis of size *n_steps*.
            ``history["ball"]["position"]`` is a 1-D array of shape
            ``(n_steps,)`` (or ``(n_steps, *field_shape)`` for
            non-scalar fields) holding the value **after** each step.

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        *final_state*, the **last** entry of *history* and not the
        first, so the next :meth:`step` or ``run_*`` call continues from
        there rather than from the state the graph held before this
        call.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        self._refuse_xla_loop_hazards("run_scan_with_history", scan=True)

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        def build():
            step_fn = self._build_step_fn()

            def scan(state, ext, params):
                self._count_scan_trace()

                def scan_body(carry, _unused):
                    new_state = step_fn(carry, ext, params)
                    return new_state, new_state  # carry, stacked output

                return jax.lax.scan(
                    scan_body, state, None, length=int(n_steps),
                )

            return jax.jit(scan)

        fn = self._cached_scan(("run_scan_with_history", int(n_steps)), build)
        final_state, history = self._call_surfacing_strict(
            fn, self._state, external_inputs, params)
        self._store_state(final_state)
        return self._user_state(final_state), self._user_state(history)

    # ------------------------------------------------------------------
    # Parameter sweeps via vmap
    # ------------------------------------------------------------------

    def run_sweep(
        self,
        n_steps: int,
        initial_states: dict[str, dict],
        external_inputs: Optional[dict[str, dict]] = None,
        return_history: bool = False,
        *,
        params: Optional[dict] = None,
    ):
        """Run a batch of simulations over different initial conditions.

        Uses ``jax.vmap`` over ``jax.lax.scan`` to execute all
        variations in parallel (vectorised on GPU/TPU).

        Parameters
        ----------
        n_steps : int
            Number of steps per simulation.
        initial_states : dict[str, dict]
            Batched initial states.  Each leaf array must have a
            leading batch dimension of the same size.  For example::

                {"ball": {"position": jnp.array([1.0, 2.0, 3.0]),
                          "velocity": jnp.zeros(3)}}

            runs 3 simulations with initial positions 1, 2, 3.
        external_inputs : dict, optional
            Static external inputs (not batched — same for all runs).
            Completed and validated as in :meth:`step`.
        return_history : bool
            If True, return ``(final_states, histories)`` where
            histories has shape ``(batch, n_steps, ...)``.
            If False (default), return only ``final_states``.

        Notes
        -----
        Multi-rate and coupled graphs are supported.  Their internal
        ``_meta`` (the sub-step counter, the coupling diagnostics and the
        predictor / IQN warm starts) is not part of ``initial_states``:
        every simulation in the batch starts from the graph's current
        ``_meta`` and evolves its own copy from there, and none of it
        appears in the returned states.  Pass an explicit ``_meta`` entry
        in ``initial_states`` — batched like any other leaf — to start
        each simulation from a different phase.

        **Not stateful, unlike every other ``run_*`` entry point.**  The
        graph's own state is *not* advanced.  This call reads
        ``self._state`` only for the ``_meta`` seed described above and
        writes nothing back, so the graph is left exactly as it was and
        an identical second call returns identical results; the batch
        runs from *initial_states*, which the caller supplies.
        :meth:`step`, :meth:`run`, :meth:`run_scan`,
        :meth:`run_scan_with_history`, :meth:`run_adaptive` and
        :meth:`run_adaptive_scan` all *do* advance it.

        Returns
        -------
        final_states : dict[str, dict]
            Batched final states (leading batch dimension on each leaf).
        histories : dict[str, dict], optional
            Only if ``return_history=True``.  Batched histories with
            shape ``(batch, n_steps, ...)`` on each leaf.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        self._refuse_xla_loop_hazards("run_sweep", scan=True)

        external_inputs = self._resolve_external_inputs(external_inputs)
        params = self._params_or_default(params)

        # Every other entry point carries ``self._state``, which
        # ``compile()`` seeded with ``_meta``; the batched carry is the
        # caller's ``initial_states``, which has none.  Without this a
        # multi-rate graph raised ``KeyError: '_meta'`` and a coupled one
        # with diagnostics a scan carry mismatch, though nothing
        # documented either as unsupported.  Passed as an argument rather
        # than closed over so a cached program cannot serve a stale
        # counter, and unbatched inside ``vmap`` so each simulation forks
        # its own copy of the warm start.
        meta = self._state.get(_graph_specs._META_KEY)

        def build():
            step_fn = self._build_step_fn()

            def sweep(init_states, ext, params, meta0):
                self._count_scan_trace()

                def simulate(init_state):
                    carry = dict(init_state)
                    if meta0 is not None:
                        carry.setdefault(_graph_specs._META_KEY, meta0)

                    def scan_body(state, _unused):
                        new_state = step_fn(state, ext, params)
                        return new_state, (new_state if return_history else None)

                    final, hist = jax.lax.scan(
                        scan_body, carry, None, length=int(n_steps),
                    )
                    if return_history:
                        # `scan_body` yields `None` for the stacked output
                        # only when `return_history` is False.
                        assert hist is not None
                        return self._user_state(final), self._user_state(hist)
                    return self._user_state(final)

                return jax.vmap(simulate)(init_states)

            return jax.jit(sweep)

        fn = self._cached_scan(
            ("run_sweep", int(n_steps), bool(return_history)), build,
        )
        return self._call_surfacing_strict(fn, initial_states, external_inputs, params, meta)

    # ------------------------------------------------------------------
    # Adaptive timestepping
    # ------------------------------------------------------------------

    def _build_dt_step_fn(self, *, collect_strict: bool = False) -> Callable:
        """Build a step function parameterised by ``dt``.

        Returns a function ``(state, external_inputs, dt) -> new_state``
        where *dt* is a JAX scalar that overrides each node's compiled
        timestep.  Used by :meth:`run_adaptive`.

        With ``collect_strict=True`` it returns ``(new_state, verdicts)``
        instead and ``strict_convergence`` raises nothing inside it:
        ``verdicts`` maps the key of every ``solver="ift"`` group with
        ``strict_convergence=True`` to ``(non_finite, unconverged)``, each
        true when some solve of the step failed that way, and the
        function's ``strict_messages`` attribute maps the same keys to the
        two messages.  The adaptive steppers compute solves they discard
        -- the error estimate's full step, every rejected attempt -- and
        check only the ones they keep, as a multi-rate step checks only
        the solves it applies (MADD-ANO-044).
        """
        self._refuse_adaptive_geometry("_build_dt_step_fn")
        schedule = list(self._schedule)
        nodes_dict = dict(self._nodes)
        back_edge_set = set(self._back_edges)
        coupling_groups = list(self._coupling_groups)
        strict_mesh = self._strict_mesh(coupling_groups)

        node_to_group: dict[str, CouplingGroup] = {}
        for group in coupling_groups:
            for name in group.nodes:
                node_to_group[name] = group

        blocks: list[tuple] = []
        handled_groups: set[int] = set()
        for node_name in schedule:
            if node_name in node_to_group:
                group = node_to_group[node_name]
                gid = id(group)
                if gid not in handled_groups:
                    handled_groups.add(gid)
                    group_schedule = [n for n in schedule if n in group.nodes]
                    blocks.append(("coupled", group, group_schedule))
            else:
                blocks.append(("node", node_name))

        edges_by_target: dict[str, list[EdgeSpec]] = defaultdict(list)
        for edge in self._edges:
            edges_by_target[edge.target_node].append(edge)

        ext_by_target: dict[str, list[ExternalInputSpec]] = defaultdict(list)
        for ei in self._external_inputs:
            ext_by_target[ei.target_node].append(ei)
        declared_input_dtypes = _graph_specs._declared_input_dtypes(
            self._external_inputs)

        has_external = set(ext_by_target.keys())
        has_coupling = bool(coupling_groups)

        params_snapshot = self.params

        from maddening.core.node import SimulationNode as _SimBase
        flux_producers = {
            nn for nn, sp in nodes_dict.items()
            if type(sp.node).compute_boundary_fluxes
            is not _SimBase.compute_boundary_fluxes
        }

        def _resolve_and_update(node_name, new_state, full_state, ext, dt,
                                node_params, flux_state,
                                force_forward_edges=None):
            boundary_inputs: dict[str, Any] = {}
            for edge in edges_by_target[node_name]:
                back = edge in back_edge_set and (
                    force_forward_edges is None
                    or edge not in force_forward_edges
                )
                src_state = full_state if back else new_state
                src_dict = src_state.get(edge.source_node, {})
                if edge.source_field in src_dict:
                    value = src_dict[edge.source_field]
                elif (not back and edge.source_field
                      in flux_state.get(edge.source_node, {})):
                    # A flux edge, resolved as the fixed-step graph
                    # resolves it: from the fluxes the source node produced
                    # earlier in this step.  This used to be a bare
                    # ``KeyError`` naming the field.
                    value = flux_state[edge.source_node][edge.source_field]
                else:
                    raise ValueError(
                        f"run_adaptive / run_adaptive_scan cannot resolve edge "
                        f"{edge.key!r}: {edge.source_field!r} is not a state "
                        f"field of {edge.source_node!r}"
                        + (" and it is a back edge, whose flux would have to "
                           "come from the previous step's boundary inputs, "
                           "which the adaptive step does not keep"
                           if back else
                           ", nor a flux it produced earlier in the step")
                        + ".  Use step / run_scan, or feed the node from a "
                        "state field."
                    )
                value = _graph_specs._apply_edge(edge, value, node_params)
                if edge.additive and edge.target_field in boundary_inputs:
                    boundary_inputs[edge.target_field] = (
                        boundary_inputs[edge.target_field] + value
                    )
                else:
                    boundary_inputs[edge.target_field] = value

            if node_name in has_external:
                node_ext = ext.get(node_name, {})
                for ei in ext_by_target[node_name]:
                    if ei.target_field in node_ext:
                        boundary_inputs[ei.target_field] = node_ext[ei.target_field]

            spec = nodes_dict[node_name]
            new_node_state = _graph_specs._node_update(
                spec, new_state[node_name], boundary_inputs, dt,
                node_params.nodes.get(node_name),
            )
            if node_name in flux_producers:
                fluxes = _graph_specs._node_fluxes(
                    spec, new_node_state, boundary_inputs, dt,
                    node_params.nodes.get(node_name),
                )
                if fluxes:
                    flux_state[node_name] = fluxes
            return new_node_state

        def dt_step_fn(state, external_inputs, dt, params=None):
            external_inputs = _graph_specs._cast_external_inputs(
                external_inputs, declared_input_dtypes)
            if params is None:
                params = params_snapshot
            else:
                self._validate_params(params)
            node_params = _graph_specs._ResolvedParams(
                params.get("nodes", {}), params.get("mappings", {}),
            )
            new_state = {k: v for k, v in state.items()}
            # Per call, not per build: a flux is a value of *this* step.
            flux_state: dict[str, dict] = {}
            verdicts: dict[str, tuple] = {}

            if has_coupling:
                for block in blocks:
                    if block[0] == "node":
                        nn = block[1]
                        new_state[nn] = _resolve_and_update(
                            nn, new_state, state, external_inputs, dt,
                            node_params, flux_state,
                        )
                    else:
                        _, group, group_schedule = block
                        sink: Optional[list] = [] if collect_strict else None
                        new_state = _coupled_block._run_coupled_block_impl(
                            group, group_schedule, new_state, state,
                            external_inputs, runtime_dt=dt,
                            nodes=nodes_dict,
                            edges_by_target=edges_by_target,
                            ext_by_target=ext_by_target,
                            back_edge_set=back_edge_set,
                            has_external=has_external,
                            all_edges=self._edges,
                            multigpu_device_map=self._multigpu_device_map,
                            node_params=node_params,
                            strict_sink=sink,
                            probe_sink=self._gradient_whole_probes,
                            strict_mesh=strict_mesh,
                        )
                        if sink:
                            # One entry per checked solve (one per waveform
                            # sweep), reduced to "some solve failed so".
                            verdicts["+".join(sorted(group.nodes))] = (
                                functools.reduce(jnp.logical_or, [nf for nf, _ in sink]),
                                functools.reduce(jnp.logical_or, [uc for _, uc in sink]),
                            )
            else:
                for nn in schedule:
                    new_state[nn] = _resolve_and_update(
                        nn, new_state, state, external_inputs, dt,
                        node_params, flux_state,
                    )

            if collect_strict:
                return new_state, verdicts
            return new_state

        cast(Any, dt_step_fn).strict_messages = {
            "+".join(sorted(g.nodes)): _reports._strict_convergence_messages(g)
            for g in coupling_groups
            if g.strict_convergence and g.solver == "ift"
        }
        # ``run_adaptive_scan`` raises the kept solves' verdicts in-graph,
        # on every device of a step spanning several.
        cast(Any, dt_step_fn).strict_mesh = strict_mesh
        # The report of an accepted attempt covers both kept half steps
        # (``_fold_kept_half_step_reports``); the steppers apply it to the
        # second half step's state.
        cast(Any, dt_step_fn).fold_kept_halves = functools.partial(
            _reports._fold_kept_half_step_reports, tuple(coupling_groups))
        return dt_step_fn

    def _refuse_adaptive_geometry(self, entry: str) -> None:
        """Raise for a graph with a geometry-dependent mapping: the
        adaptive steppers take half steps and discard attempts, and the
        time level a geometry is read at across those is not defined."""
        keys = _graph_specs._geometry_edge_keys(self._edges)
        if keys:
            raise RuntimeError(
                f"{entry}: this graph has geometry-dependent mapping(s) on edge(s) "
                f"{keys}, which the adaptive steppers do not support in 0.4.0: the time "
                f"level a geometry is read at across half steps and rejected attempts "
                f"is not defined yet. Use step / run_scan."
            )

    def _adaptive_multirate_message(self, entry: str) -> str:
        """Why ``run_adaptive*`` refuses this (multi-rate) graph.

        What it enforces is that the graph is not multi-rate, which is not
        "every node shares one timestep": nodes of different timesteps
        are accepted inside a coupling group with ``subcycling=True``,
        which ``compile()`` schedules at the group's macro timestep.
        """
        return (
            f"{entry}: adaptive timestepping is incompatible with multi-rate "
            "graphs, and this graph is one (rate dividers "
            f"{dict(sorted(self._rate_dividers.items()))}).  The adaptive "
            "step advances every node by the same dt; nodes of different "
            "timesteps can share it only inside one coupling group with "
            "subcycling=True, whose faster members are sub-stepped at "
            "dt * node_timestep / group_macro_timestep.  Give the other "
            "nodes the same timestep, put them in such a group, or use "
            "step / run_scan."
        )

    def run_adaptive(
        self,
        t_end: float,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
        callback: Optional[Callable] = None,
        *,
        params: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict]:
        """Run with adaptive timestepping until *t_end*.

        Uses Richardson extrapolation (step-doubling) for error
        estimation and a PI controller for step-size adjustment.
        Incompatible with multi-rate graphs: every node advances by the
        same ``dt``.  Nodes of different timesteps are accepted inside a
        coupling group with ``subcycling=True``, whose members keep their
        ratio to the group's macro timestep: each of a fast node's
        ``round(macro / node_timestep)`` sub-steps is
        ``dt * node_timestep / macro``, so at an integer ratio every
        member covers ``dt``.  (Before 0.4.0 each sub-step was handed the
        whole ``dt``, advancing the fast node by ``divider * dt``.)

        Parameters
        ----------
        t_end : float
            Target simulation end time.
        dt_initial : float
            Initial timestep guess.
        atol, rtol : float
            Absolute and relative error tolerances.  The step-doubling
            error is the RMS of ``|fine - coarse| / (atol + rtol
            max(|fine|, |coarse|))`` over the *floating* state fields
            only: an integer, boolean or PRNG-key leaf (a counter, a tag,
            a flag) carries no truncation error and adds neither a term
            nor an element.  Before 0.4.0 it was read too, so a ``bool``
            raised ``TypeError``, a ``uint32`` wrapped and rejected almost
            every step, and an ``int32`` counter moved the step sequence.
        dt_min, dt_max : float
            Timestep bounds.
        external_inputs : dict, optional
            Static external inputs applied every step.  Completed and
            validated as in :meth:`step`.
        callback : callable, optional
            Called after every *accepted* step with
            ``(sim_time, dt_used, state_dict)``.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.

        Returns
        -------
        (final_state, info) : tuple
            *final_state* is the state dict (excluding metadata).
            *info* is a dict with ``n_steps``, ``n_rejected``,
            ``dt_history`` (list of used timesteps), and
            ``t_history`` (list of simulation times).

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        the state reached at ``t_end``, so the next :meth:`step` or
        ``run_*`` call continues from there rather than from the state
        the graph held before this call.  The simulation clock reported
        in *info* restarts from zero on the next call; the state does
        not.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        # Judged after the recompile, not before it.  ``_is_multirate`` is
        # derived by ``compile()``, so on a graph that has been edited --
        # or never compiled -- it describes the step last built rather
        # than the one about to run.  Asked first, the refusal let a graph
        # that had just become multi-rate through, and refused one that
        # had just stopped being multi-rate.
        if self._is_multirate:
            raise RuntimeError(self._adaptive_multirate_message("run_adaptive"))
        self._refuse_adaptive_geometry("run_adaptive")
        self._refuse_xla_loop_hazards("run_adaptive", scan=False)

        external_inputs = self._resolve_external_inputs(external_inputs)

        from maddening.core.simulation.adaptive import (
            AdaptiveConfig,
            _tree_error_norm,
            step_decision,
        )

        config = AdaptiveConfig(
            dt_initial=dt_initial,
            atol=atol,
            rtol=rtol,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        # ``strict_convergence`` is checked here, on the host, and only
        # for the solves the stepper keeps: the error estimate's full step
        # and a rejected attempt are discarded, and raising about them
        # stopped the controller from recovering by rejecting the very
        # step whose coupling had not converged.
        dt_step_fn = self._build_dt_step_fn(collect_strict=True)
        strict_messages = cast(Any, dt_step_fn).strict_messages
        # JIT-compile the dt-parameterised step
        dt_step_jit = jax.jit(dt_step_fn)
        fold_kept_halves = jax.jit(cast(Any, dt_step_fn).fold_kept_halves)
        params = self._params_or_default(params)

        t = 0.0
        dt = dt_initial
        n_steps = 0
        n_rejected = 0
        dt_history = []
        t_history = []
        state = self._state

        while t < t_end:
            # Clamp dt so we don't overshoot t_end
            dt = min(dt, t_end - t)
            dt = max(dt, dt_min)

            dt_jax = jnp.array(dt)

            # Full step
            state_full, _discarded = dt_step_jit(state, external_inputs, dt_jax, params)
            # Two half-steps
            half_dt = dt_jax / 2.0
            state_half_1, verdicts_1 = dt_step_jit(state, external_inputs, half_dt, params)
            state_half, verdicts_2 = dt_step_jit(state_half_1, external_inputs, half_dt, params)
            # The report covers both kept half steps, as the strict check does.
            state_half = fold_kept_halves(state_half_1, state_half)

            # Error estimate
            user_full = self._user_state(state_full)
            user_half = self._user_state(state_half)
            error_norm = float(_tree_error_norm(
                user_half, user_full, config.atol, config.rtol
            ))

            # The acceptance rule and the next timestep are
            # ``run_adaptive_scan``'s, from one function
            # (``adaptive.step_decision``): accepted within tolerance, or
            # when the attempt was already made at ``dt_min``.  This loop
            # used to accept a *rejected* attempt larger than ``dt_min``
            # whenever shrinking it would reach ``dt_min``, and so took a
            # different, larger step than the scan -- overshooting
            # ``t_end`` on a decay held at ``dt_min``.
            accepted, forced, dt_next, _factor = step_decision(
                error_norm, dt, dt_min, dt_max, safety=config.safety,
                order=config.order, min_factor=config.min_factor,
                max_factor=config.max_factor, xp=np)
            if accepted:
                # Use the more accurate (half-step) result -- unless a
                # solve the step keeps did not converge.
                _reports._raise_if_a_kept_solve_failed(strict_messages, (verdicts_1, verdicts_2))
                # The first step the run keeps is compared with the state
                # the graph holds (``_refuse_layout_drift``), before any
                # warning about it, and before a callback or an observer is
                # handed it: every later one comes from the same program,
                # given the layout this one has.
                if n_steps == 0:
                    self._refuse_layout_drift(state_half)
                if forced:
                    warnings.warn(
                        f"Adaptive stepper hit dt_min={dt_min} at t={t:.6g} "
                        f"(error={error_norm:.3e}). Accepting step.",
                        stacklevel=2,
                    )
                # The clock advances by the step the two half steps covered
                # (MADD-ANO-061).
                state = state_half
                t += dt
                n_steps += 1
                dt_history.append(dt)
                t_history.append(t)

                if callback is not None:
                    callback(t, dt, self._user_state(state))
                # With the step's dt: a relay adds it to its clock, not the
                # graph's fixed step (MADD-ANO-096's adaptive residual).
                self._notify(EVENT_STEP, _graph_specs._StepState(self._user_state(state), float(dt)))
            else:
                n_rejected += 1
            dt = float(dt_next)

        self._store_state(state)
        info = {
            "n_steps": n_steps,
            "n_rejected": n_rejected,
            "dt_history": dt_history,
            "t_history": t_history,
        }
        return self._user_state(self._state), info

    def run_adaptive_scan(
        self,
        t_end: float,
        max_steps: int = 10000,
        dt_initial: float = 0.01,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        dt_min: float = 1e-8,
        dt_max: float = 0.1,
        external_inputs: Optional[dict[str, dict]] = None,
        *,
        params: Optional[dict] = None,
    ) -> tuple[dict[str, dict], dict[str, dict], dict]:
        """Adaptive timestepping via ``jax.lax.scan`` (differentiable).

        Like :meth:`run_adaptive` but fully JIT-compiled and
        differentiable.  Uses a fixed *max_steps* allocation; steps
        past ``t_end`` are no-ops.

        Parameters
        ----------
        t_end : float
            Target end time.
        max_steps : int
            Maximum number of steps (scan length).  Steps after reaching
            ``t_end`` produce no-op outputs.
        dt_initial, atol, rtol, dt_min, dt_max : float
            Same as :meth:`run_adaptive`.
        external_inputs : dict, optional
            Static external inputs.  Completed and validated as in
            :meth:`step`.
        params : dict, optional
            Graph parameter pytree (see :attr:`params`).  ``None`` uses
            :attr:`params`.  It is a traced argument of the scan, so
            ``jax.grad`` reaches it without writing a tracer into
            :attr:`params`.

        Returns
        -------
        (final_state, history, info) : tuple
            *final_state*: state after last accepted step.
            *history*: stacked state at each step (shape ``(max_steps, ...)``).
            *info*: dict with ``n_steps`` (actual steps taken, as JAX array).

        Notes
        -----
        **Stateful: the graph's own state is advanced.**  ``self`` is left at
        *final_state*, the state after the last accepted step and not
        the first entry of *history*, so the next :meth:`step` or
        ``run_*`` call continues from there rather than from the state
        the graph held before this call.  The simulation clock reported
        in *info* restarts from zero on the next call; the state does
        not.

        Building observations more than once from a single
        :class:`GraphManager` therefore measures a *different* initial
        condition each time.  For repeated measurements from one initial
        condition, construct a fresh :class:`GraphManager` inside the
        measurement loop, or snapshot and restore explicitly with
        :meth:`save_state` / :meth:`load_state`; :meth:`reset_state`
        returns the graph to its nodes' ``initial_state()``.
        :meth:`run_sweep` is the one batch entry point that leaves the
        graph's state untouched.
        """
        self._recover_from_escaped_tracers()
        self._check_static_data_dirty()
        if self._dirty or self._compiled_step is None:
            self.compile()
        # After the recompile, for the reason given in ``run_adaptive``.
        if self._is_multirate:
            raise RuntimeError(self._adaptive_multirate_message("run_adaptive_scan"))
        self._refuse_adaptive_geometry("run_adaptive_scan")
        self._refuse_xla_loop_hazards("run_adaptive_scan", scan=True)

        external_inputs = self._resolve_external_inputs(external_inputs)

        from maddening.core.simulation.adaptive import AdaptiveConfig

        config = AdaptiveConfig(
            dt_initial=dt_initial, atol=atol, rtol=rtol,
            dt_min=dt_min, dt_max=dt_max,
        )

        # The tolerances and the time bounds ride in as arguments
        # (``knobs``) so a caller sweeping them reuses the compilation;
        # only the controller constants, which this signature does not
        # expose, are baked in and therefore part of the cache key.
        fn = self._cached_scan(
            ("run_adaptive_scan", int(max_steps), config.safety,
             config.order, config.min_factor, config.max_factor),
            lambda: _adaptive_scan._build_adaptive_scan(
                self._build_dt_step_fn(collect_strict=True), self._user_state,
                int(max_steps),
                config.safety, config.order, config.min_factor,
                config.max_factor, self._count_scan_trace,
            ),
        )
        knobs = (
            jnp.array(t_end), jnp.array(dt_initial), jnp.array(atol),
            jnp.array(rtol), jnp.array(dt_min), jnp.array(dt_max),
        )
        (final_state, final_t, final_dt, n_accepted), history = self._call_surfacing_strict(
            fn, self._state, external_inputs, self._params_or_default(params), knobs,
        )

        self._store_state(final_state)
        info = {"n_steps": n_accepted, "final_t": final_t, "final_dt": final_dt}
        return self._user_state(final_state), history, info

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_node_state(self, name: str) -> dict:
        """The node's current state fields.

        A fresh dict, not the internal one: the arrays inside are
        immutable and shared, but writing a new value into the dict you
        are handed does not reach the simulation.  Use
        :meth:`set_node_state` for that.
        """
        # After ``jax.grad`` of a loss that stepped the graph, the state
        # holds tracers; hand back the state the graph is put back to,
        # not a dict of escaped tracers.
        self._recover_from_escaped_tracers()
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        if name not in self._state:
            raise KeyError(f"No node named '{name}'.")
        return dict(self._state[name])

    def set_node_state(self, name: str, state: dict) -> None:
        """Overwrite one node's state fields.

        A traced value is accepted -- writing the argument of a loss in
        is how a differentiable initial condition is expressed -- and
        noted, so the graph can be put back afterwards rather than
        keeping the tracer (see ``_recover_from_escaped_tracers``).

        Outside a transform the graph is put back first, so a write made
        right after ``jax.grad`` of a loss that stepped the graph -- the
        remedy the recovery warning names -- lands in the state that is
        kept.  It used to land in the traced state, and the next entry
        point put the graph back over it.
        """
        self._recover_from_escaped_tracers()
        if name not in self._nodes:
            raise KeyError(f"No node named '{name}'.")
        state = _graph_specs._strong_typed(state)
        self._keep_state_for_reports()
        if _graph_specs._holds_tracer({name: state}) and not self._state_traced:
            # Snapshot before the write, per node, so the recovery has
            # something untraced to go back to.  Only on the tracer path,
            # so an ordinary call allocates nothing extra.
            self._state_before_trace = {
                k: (dict(v) if type(v) is dict else v)
                for k, v in self._state.items()
            }
            self._state_traced = True
        self._state[name] = state

    def _meta_reset_seeds(self, fresh: dict) -> dict:
        """``{slot: value -> seed}`` for every ``_meta`` slot ``compile()`` seeds.

        Keyed by the exact slot name each group owns (see
        ``_GROUP_META_SUFFIXES``; ``_refuse_colliding_group_keys`` makes
        the names unambiguous), so :meth:`reset_state` reproduces the
        seeds value for value: NaN for the spectral triple and the
        gradient bound ("not computed" -- zero would read as a spectral
        radius of 0 and a gradient exact to float32), zero for the
        counters, the residual, the amplification and the IQN-IMVJ warm
        start, and the flattened *fresh* state for the predictor history,
        which is what ``compile()`` seeds it with.
        """
        from maddening.core.coupling.acceleration import (  # noqa: PLC0415
            flatten_coupled_state,
        )

        def zeros(value):
            return jnp.zeros_like(value)

        def nan(value):
            return jnp.full_like(value, jnp.nan)

        seeds: dict = {"step_count": zeros, "sub_step": zeros}
        for group in self._coupling_groups:
            key = "+".join(sorted(group.nodes))
            for suffix in ("rho_spectral", "spectral_residual",
                           "spectral_amplification", "gradient_relative_error_bound",
                           "pass_evaluations", "reading_floor", "geometry_gap"):
                seeds[f"coupling_{key}_{suffix}"] = nan
            for suffix in ("iterations", "total_iterations", "residual",
                           "amplification", "pred_count", "V", "W"):
                seeds[f"coupling_{key}_{suffix}"] = zeros
            if group.predictor != "none":
                # The order the step (and ``compile()``) flattens in.
                names = ([n for n in self._schedule if n in group.nodes]
                         or sorted(group.nodes))
                flat0 = flatten_coupled_state(
                    fresh, names, fields=float_fields_of(fresh, names),
                )

                def history(value, flat0=flat0):
                    if flat0.shape != jnp.shape(value):
                        return jnp.zeros_like(value)
                    return jnp.asarray(flat0, jnp.asarray(value).dtype)

                for pi in range(3):
                    seeds[f"coupling_{key}_pred_{pi}"] = history
        return seeds

    def reset_state(self) -> None:
        """Reset every node to its ``initial_state()`` and the internal
        counters in ``_meta`` to zero, keeping the compiled step valid.

        Prefer this to assigning ``initial_state()`` into ``_state``
        directly: the seed values are normalised the way ``compile``
        normalises them (weak types stripped), so the jitted step does not
        retrace after a reset, and ``_meta``'s structure is preserved.
        """
        # A transform that stepped the graph (``jax.grad`` of a loss
        # calling ``run_scan``) leaves tracers in it, and the predictor
        # history's seed below is sized from the live slot: reading the
        # traced one raised ``UnexpectedTracerError``, so the remedy the
        # recovery warning names failed on any group with a predictor.
        # Put back first like every other entry point -- quietly, since
        # everything put back is about to be replaced.
        self._recover_from_escaped_tracers(warn=False)
        # Every ``initial_state()`` first, then one commit: an
        # ``initial_state`` that raises (an ``AdaptiveNode`` at a Palais
        # trap) must not leave half the graph reset and half of it carrying
        # the state from before the call.  ``_meta`` is computed in the
        # same pass and committed with them: it is state too, so a reset
        # that leaves the node states fresh and the sub-step phase stale
        # would re-phase a multi-rate graph exactly the way a recompile
        # used to.
        fresh = {
            name: _graph_specs._strong_typed(spec.node.initial_state())
            for name, spec in self._nodes.items()
        }
        live_meta = self._state.get(_graph_specs._META_KEY)
        fresh_meta = None
        if live_meta is not None:
            fresh_meta = dict(live_meta)
            seeds = self._meta_reset_seeds(fresh)
            for key, value in live_meta.items():
                # By exact slot name, never by suffix: a group whose key
                # itself ends in ``_spectral`` (a node named
                # ``probe_spectral``) owns ``coupling_<key>_residual``,
                # which *ends* in ``_spectral_residual``, and matched by
                # suffix it was put back to NaN where ``compile()`` seeds
                # 0.0.  The target is the ``_meta`` ``compile()`` seeds,
                # value for value; a key no group owns is left alone.
                seed = seeds.get(key)
                if seed is not None:
                    fresh_meta[key] = seed(value)

        self._state.update(fresh)
        if live_meta is not None and fresh_meta is not None:
            # Refilled, not replaced: the compiled step and any caller
            # holding the dict keep the object they were given.
            live_meta.clear()
            live_meta.update(fresh_meta)
        # Cleared last, with the commit: a reset that raised would
        # otherwise have taken the graph's way back from an escaped
        # tracer with it while leaving the tracer in place.
        self._state_traced = False
        self._state_before_trace = None

    # ------------------------------------------------------------------
    # Observer pattern
    # ------------------------------------------------------------------

    def add_observer(self, callback: Callable) -> None:
        """Register a callback.  Called as ``callback(event, data)``."""
        self._observers.append(callback)

    def _notify(self, event: str, data: Any = None) -> None:
        for cb in self._observers:
            cb(event, data)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _warn_about_unsaved_mapping_weights(self) -> None:
        """``UserWarning`` for every mapped edge whose live weights are no
        longer the ones its recipe rebuilds.

        A config carries the ``MappingSpec``, not the weights, so weights
        moved by ``sysid`` or edited in ``params["mappings"]`` are dropped
        by a config-only round trip (checkpoints carry them, and win).
        The graph is the only place that knows both, so it says so here.
        """
        live = (self.params or {}).get("mappings") or {}
        for e in self._edges:
            if e.mapping is None:
                continue
            weights = live.get(e.key)
            if not weights:
                continue
            recipe = e.mapping.params_pytree()
            drifted = sorted(
                k for k, v in weights.items()
                if k in recipe and not (
                    np.shape(v) == np.shape(recipe[k])
                    and np.array_equal(np.asarray(v), np.asarray(recipe[k]))
                )
            )
            if drifted:
                warnings.warn(
                    f"edge {e.key}: live mapping weights {drifted} differ from what the "
                    f"MappingSpec rebuilds; a config carries the recipe only, so these "
                    f"values are not in it — save a checkpoint (gm.save_state(...)) and "
                    f"load it after the config to keep them",
                    UserWarning, stacklevel=3,
                )

    def to_dict(self, *, strict_mappings: bool = True) -> dict:
        """Serialise the graph structure (not runtime state).

        Node ``params`` are the *effective* values — the constructor
        arguments with the live :attr:`params` written over them (see
        :meth:`effective_node_params`) — and ``param_specs`` carries the
        graph's :meth:`set_param_spec` overrides.  An edge's interface
        mapping is written as its
        :class:`~maddening.core.coupling.mapping_spec.MappingSpec` (kind,
        hyper-parameters, point references — never the weights, which
        checkpoints carry).  With ``strict_mappings`` (the default) a
        mapping that cannot be rebuilt from a config — no spec, or a
        point set neither referenced nor small enough to inline — is a
        ``ValueError`` naming the ``source_ref=`` / ``target_ref=`` /
        ``asset=`` argument to pass — as is a node reference that no
        longer resolves to the points it was built from (a removed node,
        a renamed field).  ``strict_mappings=False`` writes whatever the
        mapping describes, for display, and checks nothing.

        Only the *recipe* is written, so live weights that were trained
        or hand-edited away from it would be lost: a ``UserWarning``
        naming the edge says so, pointing at :meth:`save_state`.

        ``coupling_groups`` carries *every* field of every group (see
        :meth:`~maddening.core.coupling.group.CouplingGroup.to_dict`),
        and is absent when the graph has none.  Partial would be worse
        than nothing: a group that came back missing its acceleration or
        its iteration cap would still be a group, and would quietly
        solve the same graph a different way.

        The result is JSON-*valid*, not merely JSON-shaped: a non-finite
        float anywhere in it -- a diverged param, an infinite ParamSpec
        bound -- is the quoted token ``"NaN"`` / ``"Infinity"`` /
        ``"-Infinity"``, not the bare token ``json.dumps`` writes by
        default and no conforming reader accepts (``MADD-ANO-006``).
        :meth:`from_dict` turns those back into floats, and reads the
        bare tokens an older config carries as well.  The encoding is
        applied once, to the assembled tree, so each part's own
        ``to_dict`` still returns plain floats for the callers that want
        numbers (the USD writer sets typed stage attributes from
        :meth:`CouplingGroup.to_dict`).  See
        :mod:`maddening.serialization.json_codec`.

        Because the encoding is applied here rather than at the write
        boundary -- so that plain ``json.dumps`` of this result is valid,
        which is what every caller in the tree does -- the result is
        *already encoded*.  Write it with ``json.dumps`` or
        :func:`~maddening.serialization.json_codec.dumps_encoded`, and
        **not** with :func:`~maddening.serialization.json_codec.dumps`:
        that one encodes what it is given, the encoding is not
        idempotent, and the second walk refuses the tokens the first one
        wrote (``$.param_specs.<node>.<key>.bounds[0]: the string
        '-Infinity' cannot be written to JSON``, from any graph with an
        unbounded :class:`~maddening.core.params.ParamSpec`).  Read it
        back with :func:`~maddening.serialization.json_codec.loads`,
        which *is* composable.
        """
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            encode_non_finite,
        )
        if strict_mappings:
            from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
                check_mapping_serialisable,
            )
            resolve = self.point_resolver()
            for e in self._edges:
                if e.mapping is not None:
                    check_mapping_serialisable(e.mapping, edge_key=e.key,
                                               resolve_points=resolve)
            self._warn_about_unsaved_mapping_weights()
        nodes = []
        for name, spec in self._nodes.items():
            d = spec.node.to_dict()
            if spec.accepts_params:
                d["params"] = self.effective_node_params(name)
            nodes.append(d)
        overrides = {
            n: {k: s.to_dict() for k, s in o.items()}
            for n, o in self.param_spec_overrides().items()
        }
        # Detached last, over the whole tree: whatever a part's own
        # ``to_dict`` handed out -- a node's params, a mapped edge's point
        # sets, a sharded wrapper's axis map -- the config shares no
        # container with the graph, so editing it edits nothing else
        # (MADD-ANO-205).
        return encode_non_finite(_detached_config({
            "nodes": nodes,
            **({"param_specs": overrides} if overrides else {}),
            "edges": [e.to_dict() for e in self._edges],
            "external_inputs": [ei.to_dict() for ei in self._external_inputs],
            # Every field of every group, or the key is absent: a config
            # that carried only some of a group's solver settings would
            # reload as a graph that *runs* differently -- a fixed point
            # iterated to convergence becoming a single staggered pass --
            # without anything saying so.  Absent, like ``param_specs``,
            # when there is nothing to say, so an uncoupled graph writes
            # exactly the config it wrote before this key existed.
            **({"coupling_groups": [g.to_dict() for g in self._coupling_groups]}
               if self._coupling_groups else {}),
        }))

    @classmethod
    def from_dict(
        cls,
        config: dict,
        node_registry: dict[str, type],
        *,
        base_dir=None,
    ) -> "GraphManager":
        """Reconstruct a GraphManager from a serialised config.

        *node_registry* maps node type names (e.g. ``"BallNode"``) to
        the corresponding class.  Edge mappings are rebuilt from their
        ``MappingSpec`` (node-field references resolve against the
        nodes just created; ``{"asset": ...}`` paths are relative to
        ``base_dir``, the directory the config was read from — the
        working directory when ``None``) and registered in
        ``params["mappings"]`` exactly as ``add_edge(mapping=)`` does.
        A checkpoint loaded afterwards overwrites the rebuilt weights.

        ``param_specs`` overrides are applied *after* the edges, because
        an override may name a mapped edge's key (``set_param_spec(
        edge.key, "H", ParamSpec())`` — trainable mapping weights) and
        those slots only exist once the edge does; node overrides do not
        depend on the edges, so the order is safe for them too.

        ``coupling_groups`` are rebuilt with :meth:`add_coupling_group`,
        so a stored group is checked exactly like a hand-written one; a
        group that cannot be rebuilt — an unknown node, a node already
        in another group, a misspelled enum — raises ``ValueError``
        naming the group and what is wrong with it.  A config without
        the key (one written before it existed) loads unchanged.

        Non-finite numbers are decoded first, so both spellings load:
        the quoted ``"NaN"`` / ``"Infinity"`` / ``"-Infinity"`` that
        :meth:`to_dict` writes since 0.4.0, and the bare tokens an older
        config carries, which ``json.loads`` has already turned into
        floats by the time the dict arrives here (``MADD-ANO-006``).

        A node saved inside a sharded wrapper (``ShardedPointwiseNode``,
        ``ShardedStencilNode``, ``ShardedUnstructuredNode``) is rebuilt
        *unsharded*, as the node it wraps, with a ``UserWarning`` naming
        the wrapper, the settings the config recorded and the call that
        wraps it again: a config does not carry the device mesh (nor an
        unstructured partition layout), which need not exist on the
        machine loading it.  Until 0.4.0 the sharding was dropped with no
        word (MADD-ANO-036).

        A node whose class can estimate what it would allocate (the
        built-in grid and basis nodes) is checked before its constructor
        runs: one that would take more memory than this machine has raises
        ``ValueError`` naming the node, where it used to run the process
        into the OOM killer.
        """
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            decode_non_finite,
        )

        def warn_unsharded(nd: dict) -> None:
            # Rebuilding the wrapper needs a mesh (and, unstructured, a
            # partition layout), which a config does not carry and the
            # loading machine may not have, so the node comes back unsharded
            # -- said out loud, with the settings and the call that restores
            # them, rather than refused: the wrapped node's model is what a
            # wrapper computes, so the reloaded graph is a correct one.
            name, inner = nd["name"], nd["type"]
            if nd.get("sharding") == "unstructured":
                wrapper, lost = "ShardedUnstructuredNode", "device mesh and partition layout"
                settings = {k: nd[k] for k in ("n_devices", "exchange") if k in nd}
                call = (f"ShardedUnstructuredNode(gm.get_node({name!r}), mesh, layout, "
                        f"exchange={nd.get('exchange', 'all_to_all')!r})")
            elif nd.get("sharded_stencil"):
                wrapper, lost = "ShardedStencilNode", "device mesh"
                settings = {k: nd[k] for k in ("axis_map", "boundary") if k in nd}
                call = (f"ShardedStencilNode(gm.get_node({name!r}), mesh, "
                        f"axis_map={nd.get('axis_map')!r}, boundary={nd.get('boundary')!r})")
            else:
                wrapper, lost = "ShardedPointwiseNode", "device mesh"
                settings = {k: nd[k] for k in ("shard_axes",) if k in nd}
                axes = nd.get("shard_axes", [0])
                axes = tuple(axes) if isinstance(axes, list) else axes
                call = f"ShardedPointwiseNode(gm.get_node({name!r}), mesh, shard_axes={axes!r})"
            warnings.warn(
                f"node {name!r} was saved as a {wrapper} {settings} and is rebuilt "
                f"unsharded, as a plain {inner}: a config does not carry the {lost}, "
                f"which need not exist on the machine loading it.  It now runs on one "
                f"device.  To shard it again, swap the wrapper in before compile(): "
                f"maddening.surrogates.replace.replace_node(gm, {name!r}, {call}).",
                UserWarning, stacklevel=3,
            )

        from maddening.core._size_estimate import (  # noqa: PLC0415
            refuse_beyond_memory,
        )

        config = decode_non_finite(config)
        gm = cls()
        for nd in config["nodes"]:
            node_cls = node_registry[nd["type"]]
            # A config is untrusted input, and its params can name a node no
            # machine holds (a wavelet basis of n_levels=10_000_000): told
            # from the class's own size estimate, before its constructor
            # runs, as a ValueError naming the node.  Only what cannot fit
            # in this machine's memory is refused.
            refuse_beyond_memory(node_cls, nd["name"], nd.get("params", {}))
            node = node_cls(name=nd["name"], timestep=nd["timestep"], **nd.get("params", {}))
            gm.add_node(node)
            if nd.get("sharded"):
                warn_unsharded(nd)
        resolve = gm.point_resolver(base_dir)
        for ed in config["edges"]:
            mapping = None
            if ed.get("mapping") is not None:
                mapping = gm._rebuild_mapping(ed, resolve)
            gm.add_edge(
                source=ed["source_node"],
                target=ed["target_node"],
                source_field=ed["source_field"],
                target_field=ed["target_field"],
                transform=ed.get("transform"),          # registered name
                additive=bool(ed.get("additive", False)),
                source_units=ed.get("source_units"),
                target_units=ed.get("target_units"),
                mapping=mapping,
                geometry=ed.get("geometry"),
            )
        for owner, overrides in config.get("param_specs", {}).items():
            for key, spec_dict in overrides.items():
                try:
                    gm.set_param_spec(owner, key, ParamSpec.from_dict(spec_dict))
                except KeyError as exc:
                    # set_param_spec reports an unknown owner / key as a
                    # KeyError; the loader speaks ValueError like the rest
                    # of from_dict, and names what it was applying.
                    raise ValueError(
                        f"param_specs[{owner!r}][{key!r}] cannot be applied to this "
                        f"config ({owner!r} is neither one of its nodes nor one of its "
                        f"mapped edge keys): {exc}"
                    ) from exc
        for ei in config.get("external_inputs", []):
            spec = ExternalInputSpec.from_dict(ei)
            gm.add_external_input(
                target_node=spec.target_node,
                target_field=spec.target_field,
                shape=spec.shape,
                dtype=spec.dtype,
            )
        for i, cg in enumerate(config.get("coupling_groups", [])):
            # Straight back through ``add_coupling_group``, so a loaded
            # group is checked by the same code as a hand-written one:
            # the node names against this graph, the node set against the
            # groups already registered, and every enum by
            # ``CouplingGroup.__post_init__``.  What those checks do not
            # know is *which* group of a multi-group config they are
            # talking about, which is the only thing that makes a
            # hand-edited file actionable -- so name it here.
            try:
                nodes, kwargs = coupling_group_kwargs(cg)
                gm.add_coupling_group(nodes, **kwargs)
            except (KeyError, TypeError, ValueError) as exc:
                named = ""
                if isinstance(cg, dict) and isinstance(cg.get("nodes"), (list, tuple)):
                    named = f" (nodes {sorted(cg['nodes'])})"
                # ``str(KeyError)`` is the *repr* of its message; unwrap it,
                # and say what a bare missing key means.
                detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
                if isinstance(exc, KeyError) and detail == "nodes":
                    detail = "it has no 'nodes' key"
                raise ValueError(
                    f"coupling_groups[{i}]{named} cannot be rebuilt: {detail}"
                ) from exc
        return gm

    @staticmethod
    def _rebuild_mapping(edge_dict: dict, resolve_points) -> Any:
        """The mapping of a serialised edge, rebuilt from its spec dict and
        checked against the ``shape`` recorded with it.

        *Every* way a spec can fail — a malformed dict, a reference that
        does not resolve or no longer matches, an unreadable / oversized
        asset, a hyper-parameter of the wrong type, a singular solve —
        comes back as a
        :class:`~maddening.core.coupling.mapping_spec.MappingRebuildError`
        (a ``ValueError``) naming this edge, with the original exception
        chained: a config is untrusted input and the edge it broke on is
        the only thing that makes the failure actionable.  So does a
        factory's ``ImportError`` (an optional package it needs is not
        installed).

        A kind added with
        :func:`~maddening.core.coupling.mapping_registry.register_mapping`
        runs a factory this library did not write, which may raise
        anything; for such a kind *every* ``Exception`` is wrapped, not
        only the types the built-in factories and the resolver raise.
        """
        import zipfile  # noqa: PLC0415
        from maddening.core.coupling.mapping_registry import (  # noqa: PLC0415
            _lookup,
        )
        from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
            MappingRebuildError,
            MappingSpec,
        )
        # MappingRebuildError prefixes "edge "; this is just the key.
        where = (f"{edge_dict['source_node']}.{edge_dict['source_field']} -> "
                 f"{edge_dict['target_node']}.{edge_dict['target_field']}")
        d = edge_dict["mapping"]
        kind = d.get("kind") if isinstance(d, dict) else None
        try:
            mapping = MappingSpec.from_dict(d).build(resolve_points)
            shape = d.get("shape")
            if shape is not None:
                if (isinstance(shape, (str, bytes)) or not isinstance(shape, (list, tuple))
                        or len(shape) != 2
                        or any(isinstance(s, bool) or not isinstance(s, int) for s in shape)):
                    raise ValueError(
                        f"'shape' must be a two-element list of ints "
                        f"[n_target, n_source], got {shape!r}"
                    )
                if [mapping.n_target, mapping.n_source] != list(shape):
                    raise ValueError(
                        f"rebuilt mapping has shape "
                        f"{[mapping.n_target, mapping.n_source]} but the config recorded "
                        f"{list(shape)}; the referenced point sets changed"
                    )
        except (ValueError, TypeError, KeyError, OSError, MemoryError,
                zipfile.BadZipFile, ImportError) as exc:
            # json.JSONDecodeError is a ValueError and numpy's
            # UFuncTypeError a TypeError, so both land here too.  An
            # ImportError is a factory that needs a package this
            # environment lacks: the edge that needs it is what the user
            # has to be told, whichever kind's factory it is.
            raise MappingRebuildError(where, kind, exc) from exc
        except Exception as exc:
            # Any other type is the edge's to report only when the kind's
            # factory is not one of ours: what the built-in kinds can raise
            # is listed above, and anything else from them is a defect that
            # should surface as itself.
            entry = _lookup(kind)
            if entry is None or entry.builtin:
                raise
            raise MappingRebuildError(where, kind, exc) from exc
        return mapping

    # ------------------------------------------------------------------
    # Checkpoint / restore
    # ------------------------------------------------------------------

    def save_state(self, path) -> "Path":
        """Save all node states to an ``.npz`` file.

        See :func:`maddening.core.simulation.checkpoint.save_state` for details.
        """
        from maddening.core.simulation.checkpoint import save_state
        self._recover_from_escaped_tracers()
        # A checkpoint stores gm.params: a leaf the step cannot read would
        # be restored as if it had produced the saved state.
        self._refuse_baked_param_writes(self.params, live=True)
        return save_state(self, path)

    def load_state(self, path) -> None:
        """Load node states from an ``.npz`` file.

        See :func:`maddening.core.simulation.checkpoint.load_state` for details.
        The parameter leaves are restored as a ``gm.params`` write is taken:
        not checked against a ``ParamSpec``'s bounds or a reload, which
        ``POST /checkpoint/load`` asks as ``PUT /graph/params`` does.  This
        is by decision: bounds are metadata to a graph, Python code may hold
        a value outside them on purpose (a ``gm.params`` write is not
        checked either), and a graph so written must resume its own
        checkpoint into a freshly built graph.  Text and booleans for a
        numeric leaf are refused, as values no save writes.
        """
        from maddening.core.simulation.checkpoint import load_state
        self._recover_from_escaped_tracers()
        load_state(self, path)

    def _transaction_snapshot(self, also=(), *, leaves=None):
        """Everything this graph holds, as it is now, for
        :meth:`_transaction_restore` to put back: the state, the params
        tree, the nodes and their own ``params``, the edges, the coupling
        groups, the external inputs and every piece of compile bookkeeping
        (``maddening.core._graph_transaction`` says how, and what it shares
        instead of copying -- every array, so the cost does not grow with
        the size of a field).  *also*: further objects to record with it (a
        server's own containers that describe the graph).  Reads nothing
        through a property, so it takes in no pending ``node.params`` write
        and compiles nothing: the graph it records is the one it found."""
        from maddening.core._graph_transaction import _Snapshot
        return _Snapshot([self, *also], leaves=leaves)

    def _transaction_restore(self, snapshot) -> None:
        """Put back what :meth:`_transaction_snapshot` recorded: every
        attribute of this graph, and every container reachable from one, is
        as it was, in place.  Notifies no observer."""
        snapshot.restore()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> float:
        """The simulated time one :meth:`step` advances the graph by.

        The GCD of the node timesteps *as* :meth:`compile` *schedules
        them*: every member of a ``subcycling=True`` coupling group is
        scheduled at the group's largest member timestep, because the group
        solves once per macro step and sub-steps its faster members inside
        that solve.  For a uniform-rate graph this is the common timestep.
        For a multi-rate graph it is the smallest step at which the
        compiled function advances, and node ``n`` fires every
        ``rate_dividers[n]`` steps of it.  ``n_steps * timestep`` is the
        simulated time ``n_steps`` calls of :meth:`step` cover, which is
        what :class:`~maddening.viz.runner.RealtimeRunner`, the state
        relays, the FMU's default step and the USD ``baseDt`` report.

        Computed from the graph as it stands, so on a graph modified since
        its last compile it is the step the next compile will take.

        Until 0.4.0 this was the GCD of the nodes' own timesteps, which on
        a graph with a sub-cycling group is shorter than a step: two nodes
        at 0.01 and 0.02 in one such group read 0.01 while every step
        advanced 0.02 (MADD-ANO-096).

        Raises
        ------
        RuntimeError
            If the graph has no nodes.
        """
        return _graph_specs._step_duration(
            _graph_specs._scheduled_timesteps(self._nodes, self._coupling_groups)
        )

    @property
    def is_multirate(self) -> bool:
        """Whether the graph has nodes with different timesteps."""
        return self._is_multirate

    @property
    def rate_dividers(self) -> dict[str, int]:
        """Per-node rate divider (node_dt / base_dt, rounded).

        ``base_dt`` is :attr:`timestep`, and ``node_dt`` the node's
        scheduled timestep: a member of a sub-cycling coupling group is
        scheduled at the group's largest member timestep.  Only meaningful
        after :meth:`compile`.
        """
        return dict(self._rate_dividers)

    @property
    def base_timestep(self) -> float:
        """Alias for :attr:`timestep`: the simulated time one step advances."""
        return self.timestep

    @property
    def node_names(self) -> list[str]:
        return list(self._nodes.keys())

    @property
    def schedule(self) -> list[str]:
        return list(self._schedule)

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = len(self._edges)
        ei = len(self._external_inputs)
        compiled = "compiled" if not self._dirty else "dirty"
        parts = [f"{n} nodes", f"{e} edges"]
        if ei:
            parts.append(f"{ei} external inputs")
        if self._is_multirate:
            parts.append("multi-rate")
        return f"GraphManager({', '.join(parts)}, {compiled})"

    # ------------------------------------------------------------------
    # Read-only inspection (logic in ``maddening.core.inspection``)
    # ------------------------------------------------------------------
    #
    # Every method below reads the graph and changes nothing: not the
    # state, ``_meta``, ``params``, node parameters, the dirty / compiled
    # flags, the schedule or any cache.  None compiles or traces the step,
    # and none puts a graph holding escaped tracers back (that is a
    # write); each reports such a graph as it stands instead.
    # ``tests/core/test_inspection_read_only.py`` pins this for every
    # method on every kind of graph.

    @stability(StabilityLevel.EXPERIMENTAL)
    def format_graph(self, *, width: int = 100) -> str:
        """The graph's structure as plain text: what :meth:`print_graph` prints.

        Sections, in order:

        * a header: node, edge, coupling-group and external-input counts,
          the compile status, the base timestep and whether the graph is
          multi-rate, and the multi-GPU mesh if one is enabled;
        * **Nodes** -- each node's type (a wrapper shows what it wraps:
          ``ShardedPointwiseNode(MyNode)``), timestep and rate divider,
          its coupling group and sub-cycling factor, its state fields as
          ``dtype[shape]`` (a sharded axis reads ``16@devices``), and for
          a sharded node the wrapper, mesh axes and what is split;
        * **Edges** -- ``source.field -> target.input`` with whether the
          source is a state field or a boundary flux, the transform's
          name, the mapping, ``additive``, units, and whether the edge is
          iterated inside a coupling group or is a back edge;
        * **Coupling groups** -- members, solver, acceleration, iteration
          mode, norm with the tolerance it reads, ``max_iterations``,
          diagnostics, strict convergence, any other non-default setting
          and the sub-cycling factors;
        * **External inputs** -- ``node.field`` with dtype and shape;
        * **Execution order** -- the compiled schedule, a coupling group
          as one block, and how often each block fires.

        Long names wrap onto their own line at ``width`` characters and
        are never split.  Deterministic: no terminal size, colour or
        object address enters it.

        Read-only.  On a graph that has never been compiled the rate
        dividers and the execution order read "not compiled" (they are
        not computed here); on one modified since its last compile they
        are the last compile's, marked stale.  On a graph holding JAX
        tracers (after ``jax.grad`` of a loss that calls ``run_scan``)
        the state fields' shapes and dtypes are read from the tracers.

        Parameters
        ----------
        width : int
            Wrap width of the plain-text layout.

        Returns
        -------
        str
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.format_graph(self, width=width)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_graph(self, *, file: Optional[TextIO] = None, width: int = 100,
                    rich: bool = False) -> None:
        """Print :meth:`format_graph` to ``file`` (default ``sys.stdout``).

        ``rich=True`` renders the same sections as trees with the
        optional ``rich`` package (``pip install maddening[terminal]``)
        and raises ``ImportError`` when it is missing.  Plain text is the
        default and is never replaced by ``rich`` on its own.  Read-only,
        as :meth:`format_graph`.
        """
        from maddening.core import inspection  # noqa: PLC0415
        inspection.print_graph(self, file=file, width=width, rich=rich)

    @stability(StabilityLevel.EXPERIMENTAL)
    def to_mermaid(self, *, direction: str = "LR") -> str:
        """The graph as a Mermaid flowchart (plain text).

        Nodes are labelled with their name and type, coupling groups are
        subgraphs, edges are labelled ``field→input`` (plus the transform,
        mapping and ``additive`` where set; a flux edge is dotted), and
        each external input is a parallelogram feeding its node.  Every
        name is escaped, so quotes, ``#``, angle brackets and newlines in
        a node name cannot break the chart or inject HTML.  Node ids are
        ``n0, n1, ...`` in insertion order.

        Paste the text into anything that renders Mermaid (GitHub
        Markdown, the Mermaid live editor, MyST with sphinxcontrib-mermaid).
        No dependency is needed.  Read-only; on an uncompiled graph the
        structure is shown as registered.

        Parameters
        ----------
        direction : {"LR", "RL", "TB", "BT"}
            Flowchart direction.
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.to_mermaid(self, direction=direction)

    @stability(StabilityLevel.EXPERIMENTAL)
    def to_dot(self, *, rankdir: str = "LR") -> str:
        """The graph in Graphviz DOT (plain text).

        The same content as :meth:`to_mermaid`: coupling groups as
        ``cluster_`` subgraphs, edges labelled ``field→input``, external
        inputs as parallelograms, every label escaped.  Render it with
        ``dot -Tsvg`` or any DOT reader; nothing here needs Graphviz.
        Read-only.

        Parameters
        ----------
        rankdir : {"LR", "RL", "TB", "BT"}
            Graphviz ``rankdir``.
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.to_dot(self, rankdir=rankdir)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_graph_diagram(self, *, theme: Optional[str] = None, direction: str = "LR",
                            use_ascii: bool = False, file: Optional[TextIO] = None) -> None:
        """Draw the graph in the terminal as boxes and arrows.

        The content of :meth:`to_mermaid`, drawn by the optional
        ``termaid`` package (``pip install "maddening[terminal]"``): each
        node a box reading ``name :: Type``, each coupling group a frame
        titled ``coupling group a+b``, each edge an arrow labelled
        ``field→input`` (dotted for a flux edge), each external input a
        parallelogram feeding its node.  Raises ``ImportError`` naming
        the extra when ``termaid`` is missing.

        With ``rich`` installed (the same extra) the drawing is coloured
        by ``theme`` when ``file`` is a terminal; any other file gets the
        same drawing as plain text.  Without ``rich``, ``theme=None``
        draws it uncoloured and a named theme raises ``ImportError``.

        For reading, not parsing: the layout is termaid's, and the few
        characters termaid's parser cannot carry in a label (``"``,
        a backtick, ``%%``, ``:::``, a literal ``\\n``, a line break) are
        shown as look-alikes.  :meth:`to_mermaid` and :meth:`format_graph`
        carry every name exactly.  Read-only, as :meth:`format_graph`:
        nothing is compiled, and the state, ``params`` and flags are
        untouched.

        Parameters
        ----------
        theme : str, optional
            One of termaid's themes: ``"default"``, ``"terra"``,
            ``"neon"``, ``"mono"``, ``"amber"``, ``"phosphor"``,
            ``"gruvbox"``, ``"monokai"``, ``"dracula"``, ``"nord"``,
            ``"solarized"``; any other name raises ``ValueError``.
            ``None`` (the default) colours with ``"default"`` when
            ``rich`` is installed.
        direction : {"LR", "RL", "TB", "BT"}
            Flowchart direction; ``"TB"`` stacks a wide graph vertically.
        use_ascii : bool
            Draw the boxes and arrows in ASCII instead of Unicode box
            drawing, and write the edge labels' ``→`` as ``->``.  Names
            are drawn as they are.
        file : text stream, optional
            Where to write (default ``sys.stdout``).

        Notes
        -----
        Usage (the user guide's inspection page shows a drawing)::

            gm.print_graph_diagram()                       # coloured on a terminal
            gm.print_graph_diagram(theme="amber", direction="TB")
        """
        from maddening.core import inspection  # noqa: PLC0415
        inspection.print_graph_diagram(self, theme=theme, direction=direction,
                                       use_ascii=use_ascii, file=file)

    @stability(StabilityLevel.EXPERIMENTAL)
    def state_summary(self, *, include_meta: bool = False) -> InspectionTable:
        """Per-field statistics of the state the graph holds now.

        One row per state field, sorted by node then field, with keys
        ``node``, ``field``, ``shape``, ``dtype``, ``min``, ``max``,
        ``mean`` (over the finite entries), ``nan``, ``inf`` (counts),
        ``bytes`` and ``flags`` (a field holding NaN or inf is flagged).
        ``print(gm.state_summary())`` prints it as a table;
        :meth:`print_state_summary` does the same.

        Computed on the host from a copy of each array (``np.asarray``):
        nothing on the graph is written, and no JAX computation runs.

        * Never compiled: the state ``add_node`` initialised, noted.
        * Holding JAX tracers (after ``jax.grad`` of a loss that calls
          ``run_scan``): shapes, dtypes and bytes are the tracers';
          the value columns are ``None`` and a note says so -- call a
          stepping method, or ``get_node_state``, which puts the graph
          back to its pre-transform state, then inspect again.  This
          method does not put it back.

        Parameters
        ----------
        include_meta : bool
            Also summarise the internal ``_meta`` entries (the multi-rate
            step counter, coupling diagnostics and warm starts).

        Returns
        -------
        InspectionTable
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.state_summary(self, include_meta=include_meta)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_state_summary(self, *, file: Optional[TextIO] = None,
                            include_meta: bool = False, width: int = 100,
                            rich: bool = False) -> None:
        """Print :meth:`state_summary` (see :meth:`InspectionTable.print`)."""
        self.state_summary(include_meta=include_meta).print(file, width=width, rich=rich)

    @stability(StabilityLevel.EXPERIMENTAL)
    def params_table(self) -> InspectionTable:
        """One row per leaf of the graph's parameters, with its ParamSpec.

        Keys: ``section`` (``"nodes"`` or ``"mappings"``), ``owner`` (the
        node, or the mapped edge's key), ``param``, ``value`` (the value
        of a one-element leaf; ``None`` for an array, whose ``shape``
        says what it is), ``shape``, ``dtype``, ``trainable``,
        ``bounds`` (``(lo, hi)``, ``None`` for unbounded), ``transform``,
        ``units``, ``out_of_bounds`` (:meth:`ParamSpec.check`'s rule,
        applied on the host) and ``flags``.  Rows are sorted by section,
        owner and parameter.

        The values are the graph's effective parameters: :attr:`params`,
        with the specs of :meth:`param_specs` (each node's own with the
        :meth:`set_param_spec` overrides applied).  Read-only --
        :attr:`params` is read as it stands; unlike :meth:`check_params`
        this does not coerce a Python-scalar leaf in place.  On a graph
        never compiled, :attr:`params` is still empty, so the table shows
        the values ``compile()`` would take (each node's
        ``params_pytree()``, built fresh and not stored) and says so.
        A traced leaf is flagged and has no value.

        Returns
        -------
        InspectionTable
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.params_table(self)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_params_table(self, *, file: Optional[TextIO] = None, width: int = 100,
                           rich: bool = False) -> None:
        """Print :meth:`params_table` (see :meth:`InspectionTable.print`)."""
        self.params_table().print(file, width=width, rich=rich)

    @stability(StabilityLevel.EXPERIMENTAL)
    def coupling_report(self) -> InspectionTable:
        """:meth:`coupling_diagnostics` as one row per coupling group, with its caveats flagged.

        Keys: ``group``, ``solver``, ``max_iterations`` and the report's
        ``iterations``, ``total_iterations``, ``converged``,
        ``residual``, ``error_estimate``, ``amplification``,
        ``ratio_usable``, ``precision_limited``, ``rho_spectral``,
        ``spectral_error_bound`` and ``spectral_usable`` (``None`` where
        there is no report), and ``flags``, which names the documented
        caveats wherever they apply:

        * the group hit ``max_iterations``;
        * ``converged=False``;
        * ``ratio_usable=False`` -- the criterion fell back to the raw
          residual test, and ``converged`` reports that test;
        * ``precision_limited=True`` -- the residual is rounding, and
          ``converged`` can be ``True`` on a stalled iterate;
        * ``spectral_usable=False`` where a spectral bound was computed;
        * in place of the three above, ``not_usable_reason`` for a group
          that resolves a geometry-dependent mapping the diagnostics do
          not read (experimental: any but a single-rate
          ``multilinear_grid`` group under ``"l2"`` or ``"mixed"``
          whose step passed its self-check): its bounds, estimates and
          ``*_usable`` flags are withheld;
        * ``not_usable_reason`` for a group loaded from a checkpoint saved
          after its state was written: the bound and the flags that rest
          on the float floor are withheld;
        * why a group has no report (``solver="fori"`` without
          ``diagnostics``, no step since ``compile()`` /
          ``reset_state()``, added since the last compile).

        A graph with no coupling groups, or not compiled, gives a table
        that says so.  Read-only: :meth:`coupling_diagnostics` is read
        only on a graph holding no tracers, where it writes nothing (on
        one that does, it would put the graph back first, so this
        reports "state holds tracers" instead).  It evaluates the
        residual's float floor with eager ``jax.numpy`` operations,
        which JAX compiles once per shape on first use; the step is not
        traced.

        Returns
        -------
        InspectionTable
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.coupling_report(self)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_coupling_report(self, *, file: Optional[TextIO] = None, width: int = 100,
                              rich: bool = False) -> None:
        """Print :meth:`coupling_report` (see :meth:`InspectionTable.print`)."""
        self.coupling_report().print(file, width=width, rich=rich)

    @stability(StabilityLevel.EXPERIMENTAL)
    def memory_estimate(self) -> InspectionTable:
        """State memory per node, and in total, from shapes and dtypes.

        One row per node and one for ``_meta`` when the graph has it,
        with keys ``node``, ``fields``, ``bytes`` (global, logical size),
        ``per_device_bytes`` (what one device holds: a sharded field's
        shard, an unsharded or replicated field in full), ``devices``
        and ``sharding`` (e.g. ``"devices@0"``, ``"replicated x4"``).
        :attr:`InspectionTable.summary` carries ``state_bytes``,
        ``meta_bytes``, ``total_bytes`` and ``total_per_device_bytes``.

        **State memory only**: XLA workspace, compiled programs, the
        copies a step makes, scan histories, ``params`` and external
        inputs are not counted, so this is a floor on what a run needs.
        Read-only; it reads shapes, dtypes and shardings and never the
        values, so it works on an uncompiled graph (no ``_meta`` yet) and
        on one holding JAX tracers alike.

        Returns
        -------
        InspectionTable
        """
        from maddening.core import inspection  # noqa: PLC0415
        return inspection.memory_estimate(self)

    @stability(StabilityLevel.EXPERIMENTAL)
    def print_memory_estimate(self, *, file: Optional[TextIO] = None, width: int = 100,
                              rich: bool = False) -> None:
        """Print :meth:`memory_estimate` (see :meth:`InspectionTable.print`)."""
        self.memory_estimate().print(file, width=width, rich=rich)
