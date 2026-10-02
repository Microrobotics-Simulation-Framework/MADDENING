"""Sidecar for the MADDENING FMU shim, and its (pickle) in-process protocol.

Architecture
~~~~~~~~~~~~

The FMU itself is a compiled DLL/dylib/so loaded into a host
co-simulation tool (FMPy / Simulink / OpenModelica).  The host calls
FMI 3.0 C functions on it; the DLL marshals each call into a
length-prefixed frame and forwards it over TCP to
:class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, which drives a
long-running Python :class:`FmuSidecar` holding the JAX-JITted graph.
This is the only way to avoid paying XLA's startup cost on every FMU
instantiation.  No ZeroMQ is involved.

Wire format
~~~~~~~~~~~

.. warning::

   :meth:`FmuSidecar.handle` speaks a **pickled** request/response
   protocol, and unpickling a request executes whatever the sender put
   in it.  It is therefore **off by default** since 0.4.0: a sidecar has
   to be built with ``SidecarConfig(allow_pickle_rpc=True)`` before
   ``handle`` will answer at all, and that opt-in is only ever
   appropriate for an in-process caller whose bytes you already trust as
   much as your own code.  Nothing in MADDENING calls it: the shipped
   transport is :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, which
   speaks length-prefixed JSON and raw arrays and never unpickles
   anything.

Every message is a length-prefixed bytes blob; the payload is a
Python pickle.  Two message kinds:

* Request  (host → sidecar): ``("step",      external_inputs)``
                              ``("get_dd",   kind, x, v)``
                              ``("get_state",)``
                              ``("set_state", payload, token)``
* Response (sidecar → host): ``("ok",        result)``
                              ``("err",       traceback)``

Stability
~~~~~~~~~

The protocol is tagged ``@stability(EVOLVING)`` — settled enough that
the FMU's C wrapper can be written against it, but additions
(per-clock event signalling, FMU-state caching) may grow before M4.

Reference implementation
~~~~~~~~~~~~~~~~~~~~~~~~

The Python sidecar lives in this module as :class:`FmuSidecar`.  The
C wrapper that ships in the FMU (``maddening/fmi/c/maddening_fmu.c``)
talks to it through :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, a
TCP transport carrying length-prefixed JSON (no libzmq needed on the
importer's side); the pickle protocol below stays for in-process use
and Python clients.  Tests can call the sidecar directly without a
socket — see ``tests/fmi/test_sidecar.py``.
"""

from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.params import check_bounds
from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)
from maddening.fmi.fmu_state import (
    FMUState,
    deserialize_fmu_state,
    serialize_fmu_state,
)


def _copy_tree(tree: Any) -> Any:
    """Shallow-copy the dict spine so ``set_params`` never mutates the
    caller's pytree (leaves are immutable arrays)."""
    if isinstance(tree, dict):
        return {k: _copy_tree(v) for k, v in tree.items()}
    return tree


def _not_tunable_error(name: str, reason: str, current: Any) -> ValueError:
    """The refusal for a new value of a parameter the step cannot read.

    Shared by :meth:`FmuSidecar.set_params`, :meth:`FmuSidecar.set_fmu_state`
    and the bridge's ``set_state``, so every door into the parameter tree
    says the same thing.
    """
    return ValueError(
        f"parameter {name!r} is not tunable: {reason}.  The FMU would report "
        f"the new value while every step kept computing with "
        f"{np.array2string(np.asarray(current), threshold=8)}; nothing was "
        "written"
    )


def _same_leaf(a: Any, b: Any) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return x.shape == y.shape and bool(np.array_equal(x, y))


def _number_kind_error(a: np.ndarray, target: np.dtype, what: str) -> Optional[ValueError]:
    """The refusal for an incoming array whose *kind* is not a number the
    target can take, or ``None``.

    Numbers only: a string (``"45"``, which ``astype`` would parse), an
    object array, or anything else that is not an integer or a float is
    refused rather than coerced -- the rule :func:`maddening.fmi.tcp_bridge._value_reference`
    already applies to a value reference.  A boolean is a number only to
    a boolean leaf (``True`` used to be stored as ``1.0`` in a float
    parameter), and a complex value only to a complex one (``astype``
    would drop the imaginary part).
    """
    kind, to = a.dtype.kind, target.kind
    if kind in "iuf" or (kind == "b" and to == "b") or (kind == "c" and to == "c"):
        return None
    what_it_is = {"b": "a boolean", "c": "a complex number", "U": "a string",
                  "S": "a byte string", "O": "an object"}.get(kind, f"dtype {a.dtype}")
    return ValueError(f"{what}: value must be a number, got {what_it_is} "
                      f"({np.array2string(a, threshold=8)}); nothing was written")


def _checked_value(arr: Any, dtype: Any, *, what: str) -> np.ndarray:
    """``arr`` cast to ``dtype``, refused unless the model can hold it.

    The one value check on every write path into a sidecar's parameters
    and state: :meth:`FmuSidecar.set_params` here, and the TCP bridge's
    ``set`` and ``set_state`` through
    :func:`maddening.fmi.tcp_bridge.checked_value`, which delegates to
    this function.  It lives in this module because the bridge imports
    the sidecar and not the other way round.

    Raises
    ------
    ValueError
        If the incoming value is not a number (a string, an object, a
        boolean for a numeric leaf: :func:`_number_kind_error`), is not
        finite, or if ``dtype`` cannot hold it: a float32 leaf set to
        ``1e39`` would be stored (and read back) as ``inf``, an integer
        would wrap or truncate silently, and a boolean leaf takes a
        boolean or exactly 0 or 1 -- ``0.5``, ``2.0`` and ``-3.0`` used to
        be stored as ``True`` by their truthiness, from a ``set`` and from
        an FMU-state archive alike.
    """
    a = np.asarray(arr)
    target = np.dtype(dtype)
    refusal = _number_kind_error(a, target, what)
    if refusal is not None:
        raise refusal
    if np.issubdtype(a.dtype, np.inexact) and not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{what}: value must be finite")
    if target.kind == "b" and a.dtype.kind in "iuf" and not bool(np.all((a == 0) | (a == 1))):
        raise ValueError(
            f"{what}: a Boolean takes true / false or exactly 1 / 0, got "
            f"{np.array2string(a, threshold=8)}; nothing was written")
    with np.errstate(over="ignore", invalid="ignore"):
        cast = a.astype(dtype)
    if np.issubdtype(cast.dtype, np.floating):
        fits = bool(np.all(np.isfinite(cast)))
    elif np.issubdtype(cast.dtype, np.integer):
        fits = bool(np.array_equal(cast.astype(np.float64), a.astype(np.float64)))
    else:
        fits = True                     # bool: only 0 / 1 reach here (above)
    if not fits:
        raise ValueError(f"{what}: value does not fit its type {dtype}")
    return cast


def _restored_leaf(value: Any, live: Any, *, what: str,
                   initial: Any = None) -> np.ndarray:
    """A snapshot leaf, checked against the live leaf it would replace.

    The shape must match and the value must pass :func:`_checked_value` in
    the live leaf's dtype.  A leaf with no live counterpart is checked in
    its own dtype (finite only).

    ``initial`` is the leaf as the FMU was instantiated, when the caller
    has it: a value equal to it (``NaN`` equal to ``NaN``) is one the FMU
    held itself, so it restores even where it is not finite.  A coupling
    group with ``diagnostics=True`` seeds its spectral ``_meta`` slots with
    ``NaN`` until a solve fills them, so a snapshot taken at instantiation
    or after a reset holds them, and both restore paths refused it while
    ``GraphManager.load_state`` restored the graph's own checkpoint.  A
    field that *became* non-finite -- a diverged model -- still does not
    restore.
    """
    arr = np.asarray(value)
    if live is None:
        return _checked_value(arr, arr.dtype, what=what)
    live_arr = np.asarray(live)
    if arr.shape != live_arr.shape:
        raise ValueError(f"{what}: shape {arr.shape} != {live_arr.shape}")
    if initial is not None and np.issubdtype(arr.dtype, np.inexact) \
            and not bool(np.all(np.isfinite(arr))):
        start = np.asarray(initial)
        with np.errstate(over="ignore", invalid="ignore"):
            cast = arr.astype(live_arr.dtype)
        if start.shape == cast.shape and bool(np.array_equal(cast, start, equal_nan=True)):
            return cast
    return _checked_value(arr, live_arr.dtype, what=what)


def _state_key(node: str, field: str) -> str:
    """How a state field is named in a key-set refusal (the bridge's member name)."""
    return f"s/{node}/{field}"


def _param_key(section: str, owner: str, key: str) -> str:
    """How a parameter leaf is named in a key-set refusal (the bridge's member name)."""
    return f"p/{section}/{owner}/{key}"


def _key_set_error(what: str, expected: set, got: set) -> Optional[ValueError]:
    """The refusal for a snapshot whose ``what`` (``"fields"``,
    ``"parameters"`` or, for the bridge's archive, ``"inputs"``) are not
    exactly the live model's, or ``None``.

    Shared by :meth:`FmuSidecar.set_fmu_state` and the TCP bridge's
    ``set_state``, which name members the same way (``s/<node>/<field>``,
    ``p/<section>/<owner>/<key>``), so the two restore paths refuse a
    partial or padded snapshot with the same words.
    """
    if got == expected:
        return None
    return ValueError(f"FMU state {what} differ from the model: "
                      f"missing {sorted(expected - got)}, extra {sorted(got - expected)}")


def _state_leaves(state: Any, *, what: str) -> dict[tuple[str, str], Any]:
    """``{(node, field): leaf}`` of a graph state (``node -> {field: array}``)."""
    if not isinstance(state, dict):
        raise ValueError(f"{what}: expected a mapping of nodes, got {type(state).__name__}")
    out: dict[tuple[str, str], Any] = {}
    for node, fields in state.items():
        if not isinstance(fields, dict):
            # A graph state is node -> {field: array}; anything else is
            # not a snapshot of this sidecar's model.
            raise ValueError(f"{what} {node}: expected a mapping of fields, "
                             f"got {type(fields).__name__}")
        for field, value in fields.items():
            out[(node, field)] = value
    return out


def _param_leaves(params: Any, *, what: str) -> dict[tuple[str, str, str], Any]:
    """``{(section, owner, key): leaf}`` of a parameter tree
    (``section -> {owner: {key: array}}``, the ``GraphManager.params`` layout)."""
    if not isinstance(params, dict):
        raise ValueError(f"{what}: expected a mapping of sections, got {type(params).__name__}")
    out: dict[tuple[str, str, str], Any] = {}
    for section, owners in params.items():
        if not isinstance(owners, dict):
            raise ValueError(f"{what} {section}: expected a mapping of owners, "
                             f"got {type(owners).__name__}")
        for owner, leaves in owners.items():
            if not isinstance(leaves, dict):
                raise ValueError(f"{what} {section}/{owner}: expected a mapping of "
                                 f"parameters, got {type(leaves).__name__}")
            for key, value in leaves.items():
                out[(section, owner, key)] = value
    return out


def _copy_specs(specs: Optional[dict]) -> Optional[dict]:
    """The ``section -> owner -> {key: ParamSpec}`` spine copied, so a
    spec the bridge adds is never written into the caller's tree."""
    if specs is None:
        return None
    return {section: ({owner: dict(leaves) for owner, leaves in owners.items()}
                      if isinstance(owners, dict) else owners)
            for section, owners in specs.items()}


def _declared_inputs_resolver(
    declared: Mapping[tuple[str, str], Any], *, whose: str = "this graph",
) -> Callable[[Optional[dict]], dict]:
    """A resolver with :meth:`GraphManager._resolve_external_inputs
    <maddening.core.graph_manager.GraphManager._resolve_external_inputs>`'s
    rule, over a fixed set of declared inputs.

    ``declared`` maps every declared ``(node, field)`` to the zero leaf it
    takes when a caller omits it.  The returned function completes a
    caller's ``external_inputs`` with those zeros and refuses a name it
    does not declare -- what ``GraphManager.step`` does, for a sidecar
    that has no graph to ask (:class:`maddening.fmi.tcp_bridge.FmuTcpBridge`
    builds one from its model description: the inputs it exports and the
    ones it holds at zero).
    """
    zeros = {pair: leaf for pair, leaf in declared.items()}

    def resolve(external_inputs: Optional[dict]) -> dict:
        given = external_inputs or {}
        unknown = sorted(f"{node}.{field}" for node, fields in given.items()
                         for field in fields if (node, field) not in zeros)
        if unknown:
            known = sorted(f"{n}.{f}" for n, f in zeros)
            raise ValueError(
                f"external_inputs names {unknown}, which {whose} does not "
                f"declare; declared external inputs: {known or ['(none)']}.  "
                f"An undeclared name never reaches the node, so accepting it "
                f"would mean the value silently did nothing; fix the name."
            )
        out: dict[str, dict] = {}
        for (node, field), leaf in zeros.items():
            out.setdefault(node, {})[field] = leaf
        for node, fields in given.items():
            out[node] = {**out.get(node, {}), **fields}
        return out

    return resolve


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class SidecarConfig:
    """Static configuration handed to :class:`FmuSidecar` at startup.

    Attributes
    ----------
    schema_token : str
        The instantiation token of the
        :class:`maddening.fmi.model_description.ModelDescription` this
        FMU was built from.  Used to validate snapshots on SetFMUState.
    step_fn : callable
        ``step_fn(state, external_inputs) -> new_state``.  Typically
        :meth:`GraphManager.step` bound to a particular graph.
    initial_state : dict
        The seed state at FMU instantiation time.
    unknown_fn : callable, optional
        ``unknown_fn(x) -> y`` for directional derivative requests.
        If ``None``, ``get_dd`` requests will error.
    params : dict, optional
        The graph parameter pytree (``GraphManager.params`` layout).
        When given, ``step_fn`` is called as
        ``step_fn(state, external_inputs, params)`` — the compiled
        step's contract — and the FMI ``parameter`` variables
        (``<node>.params.<key>``) are served from / written into it by
        :meth:`FmuSidecar.get_params` / :meth:`FmuSidecar.set_params`
        and carried in the FMU state snapshot.
    param_specs : dict, optional
        ``GraphManager.param_specs()`` for the same graph.  When given,
        :meth:`FmuSidecar.set_params` rejects a value outside a leaf's
        declared ``ParamSpec.bounds`` (the ``min`` / ``max`` the model
        description advertises), so an importer cannot drive the step with
        a constant the graph declares invalid, and
        :meth:`FmuSidecar.set_fmu_state` rejects a snapshot that would
        install one -- a value the parameter neither holds now nor held
        when the FMU was instantiated (a graph runs a value outside its
        bounds, which are metadata to it, so an FMU can start outside them
        and must restore its own snapshots).  The FMU-state archive path in
        :class:`maddening.fmi.tcp_bridge.FmuTcpBridge` checks through the
        same method (:meth:`FmuSidecar._check_restored_params`), so no door
        into the parameter tree is wider than another.  A bridge holds
        its sidecar to the ``min`` / ``max`` its model description
        advertises whether or not this was given: for an exported
        parameter with no spec here it adds one carrying those bounds.
    fixed_params : mapping of str to str, optional
        ``{"<node>.params.<key>": reason}`` for parameters the compiled step
        cannot read -- pass :attr:`ModelDescription.fixed_parameters
        <maddening.fmi.model_description.ModelDescription.fixed_parameters>`.
        :meth:`FmuSidecar.set_params` and :meth:`FmuSidecar.set_fmu_state`
        refuse a *new* value for any of them (re-setting the current value
        is accepted): it would be reported by :meth:`FmuSidecar.get_params`
        and ignored by every step.  A sidecar holds only the compiled step,
        so it cannot work this out alone; :class:`FmuTcpBridge
        <maddening.fmi.tcp_bridge.FmuTcpBridge>` fills it from the model
        description it serves, whatever was passed here.
    allow_pickle_rpc : bool, default False
        Let :meth:`FmuSidecar.handle` serve the pickled RPC protocol.
        Unpickling a request runs arbitrary code from whoever supplied
        the bytes, so ``handle`` refuses unless this is set; the TCP
        bridge does not use it and does not need it.
    input_resolver : callable, optional
        ``input_resolver(external_inputs) -> external_inputs``, applied
        before every step: pass the graph's own
        ``gm._resolve_external_inputs``.  It completes the inputs a caller
        omits with zeros and refuses a ``(node, field)`` the graph does not
        declare -- what ``GraphManager.step`` does -- so
        :meth:`FmuSidecar.step` computes what the graph computes.  Without
        it, ``step_fn`` (``gm._compiled_step``, which validates nothing)
        receives the inputs as given: an omitted input never reaches its
        node, which then takes its own "input missing" branch (a ball with
        no table falls through the floor), and a misspelt node name does
        nothing at all.  :class:`FmuTcpBridge
        <maddening.fmi.tcp_bridge.FmuTcpBridge>` installs a resolver built
        from its model description when none was given here, covering the
        inputs it exports and those the description holds at zero.
    """
    schema_token: str
    step_fn: Callable[..., dict]
    initial_state: dict[str, dict[str, Any]]
    unknown_fn: Optional[Callable[[Any], Any]] = None
    params: Optional[dict] = None
    param_specs: Optional[dict] = None
    fixed_params: Optional[Mapping[str, str]] = None
    allow_pickle_rpc: bool = False
    input_resolver: Optional[Callable[[Optional[dict]], dict]] = None


@stability(StabilityLevel.EVOLVING)
class FmuSidecar:
    """In-process FMU sidecar — handles FMI 3.0 RPC messages.

    Holds the JAX-JITted graph in long-running memory.  A real
    deployment runs it in a Python process of its own, apart from the
    importer's, behind :class:`~maddening.fmi.tcp_bridge.FmuTcpBridge`;
    tests instantiate it directly and call :meth:`handle` to exercise the
    protocol without involving a real socket.
    """

    def __init__(self, config: SidecarConfig) -> None:
        self._config = config
        self._state = dict(config.initial_state)
        self._params = (
            None if config.params is None else _copy_tree(config.params)
        )
        #: The state and the parameter tree as instantiated, for the restore
        #: checks (:meth:`_initial_state_leaf`, :meth:`_check_restored_params`);
        #: never written.
        self._initial_state = {n: dict(f) for n, f in config.initial_state.items()}
        self._initial_params = (
            None if config.params is None else _copy_tree(config.params)
        )
        self._fixed: dict[str, str] = dict(config.fixed_params or {})
        # Copies, because a bridge adds to both from its model description
        # (``_adopt_advertised_bounds``, ``_adopt_input_resolver``) and must
        # never write into the caller's config.
        self._param_specs: Optional[dict] = _copy_specs(config.param_specs)
        self._input_resolver = config.input_resolver

    def _refuse_new_values_for(self, fixed: Mapping[str, str]) -> None:
        """Add ``{"<node>.params.<key>": reason}`` to the parameters whose
        new values are refused (the bridge's model-description contract)."""
        for name, reason in fixed.items():
            self._fixed.setdefault(name, reason)

    def _adopt_advertised_bounds(self, specs: Mapping[tuple[str, str], Any]) -> None:
        """Add a ``ParamSpec`` for every ``(node, key)`` in ``specs`` that
        has none yet (the bridge's model-description contract: the ``min``
        / ``max`` it advertises hold however this sidecar was configured).
        A spec the sidecar already has is kept: it is the graph's own
        declaration, which the description was built from."""
        tree = self._param_specs if self._param_specs is not None else {}
        nodes = tree.setdefault("nodes", {})
        for (node, key), spec in specs.items():
            nodes.setdefault(node, {}).setdefault(key, spec)
        self._param_specs = tree

    def _adopt_input_resolver(self, resolver: Callable[[Optional[dict]], dict]) -> None:
        """Use ``resolver`` before every step unless the sidecar was
        configured with one (the graph's own is the better authority)."""
        if self._input_resolver is None:
            self._input_resolver = resolver

    def _adopt_instantiation_params(self, params: dict) -> None:
        """Make ``params`` both the current parameters and the ones this
        FMU was instantiated with (the restore check's exemption)."""
        self._params = _copy_tree(params)
        self._initial_params = _copy_tree(params)

    @property
    def input_resolver(self) -> Optional[Callable[[Optional[dict]], dict]]:
        """What completes and validates a step's external inputs, if anything.

        ``SidecarConfig.input_resolver``, or the resolver a bridge built
        from its model description when none was configured; ``None``
        means :meth:`step` hands its inputs to ``step_fn`` as given.
        """
        return self._input_resolver

    @property
    def fixed_params(self) -> dict[str, str]:
        """The parameters this sidecar refuses a new value for, with why."""
        return dict(self._fixed)

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return self._state

    @property
    def params(self) -> Optional[dict]:
        return self._params

    @property
    def param_specs(self) -> Optional[dict]:
        """The ``ParamSpec`` tree this sidecar validates against, if any.

        ``{"nodes": {name: {key: ParamSpec}}, "mappings": {...}}``, the
        layout :meth:`GraphManager.param_specs` returns.  Exposed so that
        every writer into the parameter tree -- ``set_params`` here and
        the FMU-state archive in
        :meth:`maddening.fmi.tcp_bridge.FmuTcpBridge._decode_state` --
        checks against the same declarations.  A bridge adds a spec for
        every exported parameter that had none, carrying the ``min`` /
        ``max`` its model description advertises.
        """
        return self._param_specs

    # -- High-level handlers -------------------------------------------------

    def step(self, external_inputs: Optional[dict[str, dict[str, Any]]]) -> dict:
        """Advance one step and commit it; returns the new state.

        With an :attr:`input_resolver` (``SidecarConfig.input_resolver``,
        or the one a bridge installs), ``external_inputs`` is completed and
        checked as ``GraphManager.step`` does: an input it omits is zero,
        and a ``(node, field)`` the graph does not declare is a
        ``ValueError`` with nothing advanced.

        Raises
        ------
        ValueError
            If the resolver refuses ``external_inputs``.
        """
        self._state = self._advanced(self._state, external_inputs)
        return self._state

    def _advanced(self, state: dict[str, dict[str, Any]],
                  external_inputs: Optional[dict[str, dict[str, Any]]]) -> dict:
        """``state`` one step on, under the current parameters, without
        committing it.

        :meth:`step` is this plus the commit.  The TCP bridge runs a
        communication step's sub-steps through here on a local state and
        commits once at the end, so a failed sub-step leaves nothing
        behind and a step still running when the bridge is stopped writes
        nothing into a stopped bridge's sidecar.  The inputs go through
        :attr:`input_resolver` first, on both paths.
        """
        if self._input_resolver is not None:
            external_inputs = self._input_resolver(external_inputs)
        if self._params is None:
            return self._config.step_fn(state, external_inputs)
        return self._config.step_fn(state, external_inputs, self._params)

    def get_params(self) -> dict[str, Any]:
        """``{"<node>.params.<key>": value}`` for every FMI ``parameter``
        variable (the names ``build_model_description`` emits)."""
        if self._params is None:
            return {}
        return {
            f"{node}.params.{key}": value
            for node, leaves in self._params.get("nodes", {}).items()
            for key, value in leaves.items()
        }

    def set_params(self, updates: dict[str, Any]) -> None:
        """Write FMI ``parameter`` variables (``fmi3SetFloat*`` on a
        parameter value reference) into the pytree the next step uses.

        Keys are ``"<node>.params.<key>"``; an unknown name, a shape
        that differs from the current leaf, a value that is not a number
        (``"45"`` or ``True`` for a float leaf, which used to be stored as
        45.0 and 1.0), not finite, or that the leaf's dtype cannot hold (a
        float32 leaf set to ``1e39`` would read back as ``inf``), a new
        value for a parameter the step cannot read
        (``SidecarConfig.fixed_params``), or a value outside the leaf's
        ``ParamSpec.bounds`` (when the sidecar has a spec for it: from
        ``param_specs``, or added by a bridge from the ``min`` / ``max``
        its model description advertises) is an error, so an importer
        cannot silently tune a constant the step never reads or declares
        invalid.  The
        value check is the TCP bridge's own, so ``set_params`` refuses
        what the bridge's ``set`` refuses, with or without
        ``param_specs``.  The call is atomic: nothing is written unless
        every update is valid.
        """
        if self._params is None:
            raise RuntimeError(
                "Sidecar wasn't configured with a params pytree; the FMU "
                "exposes no parameter variables.",
            )
        nodes = self._params.get("nodes", {})
        spec_nodes = (self._param_specs or {}).get("nodes", {})
        staged: list[tuple[str, str, Any]] = []
        for name, value in updates.items():
            node, sep, key = name.partition(".params.")
            if not sep or node not in nodes or key not in nodes[node]:
                raise KeyError(
                    f"unknown parameter {name!r}; known: {sorted(self.get_params())}",
                )
            current = nodes[node][key]
            # Checked before the cast, not after: ``jnp.asarray(1e39,
            # dtype=float32)`` is ``inf``, NaN passes straight through, and
            # without ``param_specs`` nothing below looks at the value.
            new = jnp.asarray(_checked_value(
                value, current.dtype, what=f"parameter {name!r}"))
            if new.shape != current.shape:
                raise ValueError(
                    f"parameter {name!r} has shape {current.shape}, got {new.shape}",
                )
            if name in self._fixed and not _same_leaf(new, current):
                raise _not_tunable_error(name, self._fixed[name], current)
            spec = spec_nodes.get(node, {}).get(key)
            if spec is not None:
                spec.check(new, name=name)
            staged.append((node, key, new))
        for node, key, new in staged:
            nodes[node][key] = new

    def get_directional_derivative(
        self, kind: DirectionalDerivativeKind, x: Any, v: Any,
    ) -> Any:
        if self._config.unknown_fn is None:
            raise RuntimeError(
                "Sidecar wasn't configured with an unknown_fn; cannot "
                "answer get_directional_derivative requests.",
            )
        return get_directional_derivative(
            self._config.unknown_fn, kind=kind, x=x, v=v,
        )

    def get_fmu_state(self) -> FMUState:
        return serialize_fmu_state(
            state=self._state,
            schema_token=self._config.schema_token,
            params=self._params,
        )

    def set_fmu_state(self, fmu_state: FMUState) -> None:
        """Restore a snapshot taken by :meth:`get_fmu_state` (or built with
        :func:`~maddening.fmi.fmu_state.serialize_fmu_state`).

        A snapshot is a door into the state and parameter tree, so it is
        held to the checks the TCP bridge's ``set_state`` applies, with the
        same messages, all before anything is committed: it must carry
        exactly the live model's state fields and, when this sidecar has a
        parameter tree, exactly its parameters -- no node, field or
        parameter missing and none extra; every state leaf and every
        parameter must be finite and representable in the dtype (and have
        the shape) of the live leaf it replaces; a parameter the step
        cannot read (``SidecarConfig.fixed_params``) may not change; and a
        restored parameter must lie inside its declared ``ParamSpec``
        bounds when the config carries ``param_specs``, unless it is the
        value the parameter holds now or held when the FMU was
        instantiated (:meth:`_check_restored_params`).  A
        snapshot of a *diverged* model -- one holding ``inf`` or ``NaN`` the
        FMU did not start with -- therefore does not restore; the error
        names the field.  A non-finite value the FMU was instantiated with
        restores (a ``diagnostics=True`` coupling group's spectral ``_meta``
        slots are seeded ``NaN``).  A sidecar without a parameter tree
        refuses a snapshot that carries parameters, as the bridge's
        ``set_state`` does (it used to ignore them).

        Raises
        ------
        ValueError
            On a schema-token mismatch or an unreadable payload (see
            :func:`~maddening.fmi.fmu_state.deserialize_fmu_state`), or if
            any of the checks above fails.  Nothing is written.
        """
        state, params = deserialize_fmu_state(
            fmu_state, expected_schema_token=self._config.schema_token,
            return_params=True,
        )
        # The key sets first, as the bridge compares its archive's members:
        # a snapshot missing a node used to restore and leave the next step
        # to fail with a KeyError (the sidecar stayed broken), one with an
        # extra field restored it, and one missing a parameter leaf failed
        # the next step's completeness check.
        got_state = _state_leaves(state, what="FMU state")
        live_state = _state_leaves(self._state, what="live state")
        refusal = _key_set_error("fields", {_state_key(*k) for k in live_state},
                                 {_state_key(*k) for k in got_state})
        if refusal is not None:
            raise refusal
        # Rebuilt on the live skeleton, so a node with no fields -- which no
        # key names -- is kept as it is.
        new_state: dict[str, dict[str, Any]] = {node: {} for node in self._state}
        for (node, field), live in live_state.items():
            new_state[node][field] = _restored_leaf(
                got_state[(node, field)], live, what=f"FMU state {node}.{field}",
                initial=self._initial_state_leaf(node, field))
        new_params = None
        got_params = ({} if params is None
                      else _param_leaves(params, what="FMU state params"))
        if self._params is None and got_params:
            # A snapshot carrying parameters into a model with none: the
            # bridge's set_state refuses it (its archive directory names
            # members the model cannot hold), and this door used to drop
            # them in silence.  The same refusal, from the same helper.
            refusal = _key_set_error("parameters", set(),
                                     {_param_key(*k) for k in got_params})
            if refusal is not None:
                raise refusal
        if self._params is not None:
            live_params = _param_leaves(self._params, what="live params")
            refusal = _key_set_error("parameters",
                                     {_param_key(*k) for k in live_params},
                                     {_param_key(*k) for k in got_params})
            if refusal is not None:
                raise refusal
            new_params = {section: {owner: {} for owner in owners}
                          for section, owners in self._params.items()}
            for (section, owner, key), live in live_params.items():
                new_params[section][owner][key] = jnp.asarray(_restored_leaf(
                    got_params[(section, owner, key)], live,
                    what=f"FMU state param {owner}.params.{key}"))
            # Tunability and the declared bounds, through the one method the
            # bridge's set_state calls too, before anything is committed.
            self._check_restored_params(new_params)
        self._state = new_state
        if new_params is not None:
            self._params = new_params

    def _initial_state_leaf(self, node: str, field: str) -> Any:
        """The state field as this FMU was instantiated, or ``None``.

        A restore accepts a non-finite value equal to it
        (:func:`_restored_leaf`): the FMU held it itself.  Read by
        :meth:`set_fmu_state` and by the TCP bridge's ``set_state``, so the
        two restore paths judge a snapshot alike.
        """
        return self._initial_state.get(node, {}).get(field)

    def _check_restored_params(self, new_params: dict) -> None:
        """Refuse a restored parameter tree that installs what
        :meth:`set_params` could not: a new value for a parameter the step
        cannot read (``fixed_params``), or a value outside a leaf's declared
        ``ParamSpec.bounds`` that the parameter neither holds now nor held
        when this FMU was instantiated.

        The one check of :meth:`set_fmu_state` and of the TCP bridge's
        ``set_state`` (:meth:`maddening.fmi.tcp_bridge.FmuTcpBridge._decode_state`),
        so the two restore paths refuse the same snapshots with the same
        words.  Bounds are checked on the values a restore would *install*,
        not on every leaf, for the reason ``fixed_params`` already compares
        with the current value: a graph runs whatever its constructor was
        given, bounds being metadata to it, so an FMU can be instantiated
        with a parameter outside them -- and it then refused to restore the
        snapshot it had handed out itself, while ``GraphManager.load_state``
        restored the same graph's checkpoint.  Every value a snapshot of this
        FMU can hold is its instantiation value or one :meth:`set_params`
        accepted (inside the bounds), so its own snapshots restore and a
        forged out-of-bounds value is still refused.

        Raises
        ------
        ValueError
            Naming the first parameter refused; nothing is written.
        """
        live = self._params or {}
        live_nodes = live.get("nodes", {})
        for name, reason in self._fixed.items():
            node, _, key = name.partition(".params.")
            if key not in live_nodes.get(node, {}):
                continue
            current = live_nodes[node][key]
            restored = new_params.get("nodes", {}).get(node, {}).get(key, current)
            if not _same_leaf(restored, current):
                raise _not_tunable_error(name, reason, current)
        held = (live, self._initial_params or {})
        installed: dict = {}
        for section, owners in new_params.items():
            for owner, leaves in (owners or {}).items():
                for key, value in (leaves or {}).items():
                    if any(_same_leaf(value, tree[section][owner][key]) for tree in held
                           if key in tree.get(section, {}).get(owner, {})):
                        continue
                    installed.setdefault(section, {}).setdefault(owner, {})[key] = value
        check_bounds(installed, self._param_specs or {})

    # -- Wire-level RPC -----------------------------------------------------

    def handle(self, request: bytes) -> bytes:
        """Parse one wire request and produce one wire response.

        Both directions are pickled tuples.  Errors are caught and
        returned as ``("err", traceback_string)`` so the C wrapper
        can surface the failure to the FMI runtime via
        ``fmi3Status`` without losing the Python traceback.

        .. warning::

           ``pickle.loads`` on the request executes whatever produced the
           bytes.  This method therefore refuses to run unless the
           sidecar was built with
           ``SidecarConfig(allow_pickle_rpc=True)``.  The shipped
           transport, :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`,
           does not go through here; prefer it.

        Raises
        ------
        RuntimeError
            If the sidecar was not built with ``allow_pickle_rpc=True``.
        """
        if not self._config.allow_pickle_rpc:
            raise RuntimeError(
                "FmuSidecar.handle speaks a pickled protocol, and unpickling "
                "a request executes whatever produced it.  It is disabled "
                "unless the sidecar is built with "
                "SidecarConfig(..., allow_pickle_rpc=True), which is only "
                "appropriate for an in-process caller you trust as much as "
                "your own code.  The shipped FMU transport is "
                "maddening.fmi.tcp_bridge.FmuTcpBridge, which never "
                "unpickles anything.",
            )
        try:
            payload = pickle.loads(request)
        except Exception as exc:  # pragma: no cover — pickle errors
            return pickle.dumps(("err", f"unpickle failed: {exc!r}"))

        try:
            kind = payload[0]
            if kind == "step":
                result = self.step(payload[1])
                return pickle.dumps(("ok", result))
            if kind == "get_dd":
                _, dd_kind, x, v = payload
                result = self.get_directional_derivative(dd_kind, x, v)
                return pickle.dumps(("ok", result))
            if kind == "get_state":
                fmu_state = self.get_fmu_state()
                return pickle.dumps(("ok", fmu_state))
            if kind == "get_params":
                return pickle.dumps(("ok", {
                    k: np.asarray(v) for k, v in self.get_params().items()
                }))
            if kind == "set_params":
                _, updates = payload
                self.set_params(updates)
                return pickle.dumps(("ok", None))
            if kind == "set_state":
                _, fmu_state = payload
                self.set_fmu_state(fmu_state)
                return pickle.dumps(("ok", None))
            return pickle.dumps((
                "err", f"unknown request kind {kind!r}",
            ))
        except Exception:  # noqa: BLE001 — broad catch by design
            return pickle.dumps(("err", traceback.format_exc()))


__all__ = [
    "FmuSidecar",
    "SidecarConfig",
]
