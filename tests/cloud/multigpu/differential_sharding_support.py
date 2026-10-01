"""Generated configurations and two paths for the sharded-wrapper differential tests.

The oracle (``docs/developer_guide/testing_standards.md``, "Differential
tests", "Sharding wrappers"): for a node and a composition of sharded
wrappers around it, the graph built with the sharded composition and the
same graph built with the node unwrapped give the same answer --

* the trajectory after N steps through ``step``, ``run`` and ``run_scan``;
* ``run_scan(params=)``, and a ``gm.params`` write (a ``node.params``
  write for a node on the three-argument contract) followed by
  ``compile()``;
* a ``set_node_state`` write of a global state followed by ``run_scan``;
* the gradient of a loss of ``run_scan``'s final state;

-- or the sharded composition refuses, loudly, before it produces a state.

This module holds no tests.  It generates the configurations (the
Hypothesis strategies below), builds both paths for one configuration
(:func:`build_case`), runs one surface on each (:func:`run_surface`) and
compares them with the tolerances stated in :func:`assert_paths_agree`.
The tests are in ``test_differential_sharding*.py`` beside it.

What a configuration can draw
-----------------------------
Three synthetic node families, one per wrapper, each written so that its
unsharded ``update`` and its per-shard ``update_padded`` call one shared
kernel.  The per-cell arithmetic of the two paths is then the same
sequence of IEEE operations, and the only thing that can make them differ
is what the wrapper delivers to the kernel: the halo cells, the
``shard_info`` offsets and extents, the sharded and replicated statics,
the boundary inputs, the injected params, and the reduction of a domain
integral.  That is what lets the grid fields be compared bit for bit.

* **Cartesian stencil** (:class:`ShardedStencilNode`): 1-D or 2-D grid;
  halo width 1 or 2 per axis; global fill ``edge``/``zero``/``periodic``,
  declared through ``halo_boundary()`` or not; the legacy three-argument
  or the params contract; a domain integral in the state (none, scalar,
  2-vector, per-shard), named to sort before or after the grid field and
  listed in ``state_fields()`` or not; ``shard_info`` read or not; a
  sharded ``StaticArray`` read in the interior only or in the halo too; a
  replicated ``StaticArray`` read through the offsets, in the interior or
  the halo; a per-cell, scalar or mis-shaped source, a scalar gain and
  face inputs applied on the global edges; float32 or float64.
* **Pointwise** (:class:`ShardedPointwiseNode`): 1-D or 2-D state sharded
  along axis 0 or 1; contract; per-cell, scalar or mis-shaped source;
  gain; a 0-d whole-domain total; dtype.
* **Unstructured** (:class:`ShardedUnstructuredNode`): a ring plus random
  chords, partitioned unevenly (an empty shard possible), in balanced
  global order or balanced out of global order; contract; integral kinds
  as above, masked with ``shard_info["n_local"]``; a partitioned per-cell
  weight read on owned cells only or on ghosts too; per-cell, scalar or
  mis-shaped source; gain; dtype.

Meshes: 1, 2 or 4 devices on a 1-D mesh, and the 2-D pencils (2, 2),
(1, 4) and (4, 1), whose size-1 axes are sharded "over one device".
Wrapping: the wrapper alone; nested one level (the stencil and pointwise
wrappers); ``HybridNode`` around the wrapper; and, for the pointwise
wrapper, the wrapper around a ``HybridNode``.

What the oracle cannot see
--------------------------
Anything both paths share: ``GraphManager`` itself, the graph-level
params validation, and the node's own kernel (a kernel that computes the
wrong physics computes it on both paths).  A specification error in the
wrapper's documented contract is invisible too when the harness node is
written to the same contract -- the static-halo fill rule, for example,
is taken from the documented one, so a harness that agreed with a wrong
document would agree with a wrong wrapper.  See the testing-standards
section for the full list.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from jax import lax

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
    partition_value,
)
from maddening.cloud.multigpu.sharded_node import (
    ShardedPointwiseNode,
    ShardedStencilNode,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.core.static_data import StaticArray

#: Devices this process can see (the directory conftest forces 16 virtual
#: CPU devices before JAX is imported).
N_AVAILABLE = len(jax.devices())

NODE_NAME = "gen"
FILLS = ("edge", "zero", "periodic")
_PAD_MODE = {"edge": "edge", "zero": "constant", "periodic": "wrap"}
#: How a sharded ``StaticArray``'s halo is filled at the global edges: the
#: documented rule (``SimulationNode.update_padded``), periodic under a
#: periodic wrapper and the edge cell repeated under ``edge`` and ``zero``.
_STATIC_PAD_MODE = {"edge": "edge", "zero": "edge", "periodic": "wrap"}

#: Graph surfaces the oracle compares.
SURFACES = ("write_compile", "scan_params", "set_state", "run_scan", "step",
            "run", "gradient")
#: The surfaces cheap enough for the per-push tests (the gradient compiles
#: a reverse-mode scan per path and has per-push tests of its own).
FORWARD_SURFACES = tuple(s for s in SURFACES if s != "gradient")


@contextlib.contextmanager
def precision(dtype: str):
    """``jax_enable_x64`` on for ``"float64"``, off for ``"float32"``, restored after."""
    want = dtype == "float64"
    prior = bool(jax.config.read("jax_enable_x64"))
    if want != prior:
        jax.config.update("jax_enable_x64", want)
    try:
        yield
    finally:
        if want != prior:
            jax.config.update("jax_enable_x64", prior)


def _np_dtype(dtype: str):
    return np.float64 if dtype == "float64" else np.float32


# ---------------------------------------------------------------------------
# Kernels shared by the two paths
# ---------------------------------------------------------------------------


def _shift(arr, halo: tuple, axis: Optional[int], k: int):
    """The interior of a padded ``arr``, displaced by ``k`` cells along ``axis``."""
    idx = []
    for b, hb in enumerate(halo):
        n_b = arr.shape[b] - 2 * hb
        start = hb + (k if b == axis else 0)
        idx.append(slice(start, start + n_b))
    return arr[tuple(idx)]


def _second_difference(f_pad, halo: tuple, axis: int):
    """Central second difference along ``axis``: 3-point at halo 1, 5-point at 2."""
    s = functools.partial(_shift, f_pad, halo, axis)
    if halo[axis] == 1:
        return s(-1) - 2.0 * s(0) + s(1)
    return (-s(-2) + 16.0 * s(-1) - 30.0 * s(0) + 16.0 * s(1) - s(2)) / 12.0


def _smooth_along(arr_pad, h: int, axis: int, n: int):
    """``(a[i-1] + 2 a[i] + a[i+1]) / 4`` along ``axis`` of a padded profile."""
    def at(k):
        return lax.slice_in_dim(arr_pad, h + k, h + k + n, axis=axis)
    return (at(-1) + 2.0 * at(0) + at(1)) / 4.0


# ---------------------------------------------------------------------------
# The Cartesian stencil family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StencilConfig:
    """One generated configuration of the stencil family."""

    mesh_shape: tuple
    axis_names: tuple
    axis_map: tuple            # ((mesh axis, spatial axis), ...)
    shape: tuple               # global grid shape
    halo: tuple                # per spatial axis
    fill: str
    declares: bool
    contract: str              # "legacy" | "params"
    integral: Optional[str]    # None | "scalar" | "vector" | "per_shard"
    integral_name: str
    integral_listed: bool
    reads_shard_info: bool
    kappa: Optional[str]       # sharded StaticArray: None | "interior" | "halo"
    kappa_axis: int
    table: Optional[str]       # replicated StaticArray via offsets
    source: str                # "none" | "per_cell" | "scalar" | "misshapen"
    misshapen_shape: tuple
    gain: bool
    faces: bool
    dtype: str
    wrapping: str              # "single" | "nested" | "hybrid"
    steps: int
    seed: int
    surface: str
    family: str = "stencil"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def sharded_axes(self) -> dict:
        """``{spatial axis: devices on its mesh axis}``."""
        sizes = dict(zip(self.axis_names, self.mesh_shape))
        return {sa: int(sizes[ma]) for ma, sa in self.axis_map}

    @property
    def integral_shape(self) -> tuple:
        return (2,) if self.integral == "vector" else ()


class _StencilBase(SimulationNode):
    """The generated stencil node; signatures are added by :func:`stencil_node_class`."""

    def __init__(self, cfg: StencilConfig, *, rate: float = 0.5,
                 decay: float = 0.25, timestep: float = 0.02) -> None:
        super().__init__(name=NODE_NAME, timestep=timestep, rate=rate, decay=decay)
        self.cfg = cfg
        dt = _np_dtype(cfg.dtype)
        rng = np.random.default_rng(cfg.seed)
        # Zero-mean data: a large common offset would make the gradient
        # of the sum of squares cancel and the comparison ill-conditioned.
        self._f0 = rng.standard_normal(cfg.shape).astype(dt)
        statics: dict = {}
        self._kappa = None
        if cfg.kappa:
            prof = [1] * cfg.ndim
            prof[cfg.kappa_axis] = cfg.shape[cfg.kappa_axis]
            self._kappa = jnp.asarray((0.5 + rng.random(prof)).astype(dt))
            statics["kappa"] = StaticArray(value=self._kappa, replication="shard",
                                           shard_axis=cfg.kappa_axis)
        self._table = None
        if cfg.table:
            self._table = jnp.asarray((0.5 + rng.random(cfg.shape[0])).astype(dt))
            statics["table"] = StaticArray(value=self._table, replication="replicate")
        self._static = statics

    @property
    def static_data(self) -> dict:
        return self._static

    def halo_width(self) -> dict[int, int]:
        return {a: int(h) for a, h in enumerate(self.cfg.halo)}

    def state_fields(self) -> list[str]:
        cfg = self.cfg
        if cfg.integral and cfg.integral_listed:
            return ["f", cfg.integral_name]
        return ["f"]

    def domain_integral_fields(self) -> set[str]:
        return {self.cfg.integral_name} if self.cfg.integral else set()

    def domain_integral_axes(self) -> dict:
        if self.cfg.integral == "per_shard":
            return {self.cfg.integral_name: ()}
        return {}

    def initial_state(self) -> dict:
        cfg = self.cfg
        state = {"f": jnp.asarray(self._f0)}
        if cfg.integral:
            state[cfg.integral_name] = jnp.zeros(cfg.integral_shape,
                                                 _np_dtype(cfg.dtype))
        return state

    def boundary_input_spec(self) -> dict:
        cfg = self.cfg
        face = tuple(cfg.shape[1:])
        return {
            "source": BoundaryInputSpec(shape=tuple(cfg.shape),
                                        description="per-cell forcing"),
            "gain": BoundaryInputSpec(shape=(), description="scalar gain"),
            "lo": BoundaryInputSpec(shape=face, description="forcing on the low face of axis 0"),
            "hi": BoundaryInputSpec(shape=face, description="forcing on the high face of axis 0"),
        }

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, None), transform="log", units="1/s"),
                "decay": ParamSpec(bounds=(0.0, None), units="1/s")}

    # -- the shared kernel -------------------------------------------------

    def _p(self, params):
        return self.params if params is None else {**self.params, **params}

    def _kernel(self, f_pad, bi, dt, p, *, kappa_pad, offsets, extents,
                source_interior):
        cfg = self.cfg
        h = cfg.halo
        dtype = f_pad.dtype
        f = _shift(f_pad, h, None, 0)
        lap = _second_difference(f_pad, h, 0)
        for a in range(1, cfg.ndim):
            lap = lap + _second_difference(f_pad, h, a)
        coef = jnp.ones_like(f)
        if kappa_pad is not None:
            ka = cfg.kappa_axis
            n_ka = f.shape[ka]
            if cfg.kappa == "interior":
                kc = lax.slice_in_dim(kappa_pad, h[ka], h[ka] + n_ka, axis=ka)
            else:
                kc = _smooth_along(kappa_pad, h[ka], ka, n_ka)
            coef = coef * kc
        if cfg.table:
            n0 = extents[0]
            if cfg.table == "interior":
                tc = lax.dynamic_slice_in_dim(self._table, offsets[0], n0)
            else:
                tpad = jnp.pad(self._table, (h[0], h[0]),
                               mode=_STATIC_PAD_MODE[cfg.fill])
                window = lax.dynamic_slice_in_dim(tpad, offsets[0], n0 + 2 * h[0])
                tc = _smooth_along(window, h[0], 0, n0)
            coef = coef * tc.reshape((n0,) + (1,) * (cfg.ndim - 1))
        f_new = f + dt * p["rate"] * coef * lap - dt * p["decay"] * f
        if source_interior is not None:
            gain = jnp.asarray(bi.get("gain", 1.0), dtype)
            f_new = f_new + dt * gain * source_interior
        if cfg.faces:
            rows = offsets[0] + jnp.arange(extents[0])
            lo_mask = (rows == 0).astype(dtype)
            hi_mask = (rows == cfg.shape[0] - 1).astype(dtype)
            lo = jnp.asarray(bi.get("lo", jnp.zeros(cfg.shape[1:], dtype)), dtype)
            hi = jnp.asarray(bi.get("hi", jnp.zeros(cfg.shape[1:], dtype)), dtype)
            if cfg.ndim == 2:
                lo = lax.dynamic_slice_in_dim(lo, offsets[1], extents[1])
                hi = lax.dynamic_slice_in_dim(hi, offsets[1], extents[1])
                lo_mask, hi_mask = lo_mask[:, None], hi_mask[:, None]
            f_new = f_new + dt * (lo_mask * lo + hi_mask * hi)
        return f_new

    def _integral(self, f_new):
        cfg = self.cfg
        total = jnp.sum(f_new + 1.0)
        if cfg.integral == "vector":
            return total * jnp.asarray([1.0, -2.0], f_new.dtype)
        return total

    # -- the two paths -----------------------------------------------------

    def _update(self, state, bi, dt, params):
        cfg = self.cfg
        p = self._p(params)
        f = state["f"]
        f_pad = jnp.pad(f, [(h, h) for h in cfg.halo], mode=_PAD_MODE[cfg.fill])
        kappa_pad = None
        if self._kappa is not None:
            pads = [(0, 0)] * cfg.ndim
            ka = cfg.kappa_axis
            pads[ka] = (cfg.halo[ka], cfg.halo[ka])
            kappa_pad = jnp.pad(self._kappa, pads, mode=_STATIC_PAD_MODE[cfg.fill])
        source = bi.get("source")
        src = (None if source is None
               else jnp.broadcast_to(jnp.asarray(source, f.dtype), f.shape))
        f_new = self._kernel(f_pad, bi, dt, p, kappa_pad=kappa_pad,
                             offsets=(0,) * cfg.ndim, extents=tuple(f.shape),
                             source_interior=src)
        out = {"f": f_new}
        if cfg.integral:
            out[cfg.integral_name] = self._integral(f_new)
        return out

    def _update_padded(self, state_padded, bi, dt, static_padded, shard_info,
                       params):
        cfg = self.cfg
        p = self._p(params)
        f_pad = state_padded["f"]
        h = cfg.halo
        local = tuple(int(f_pad.shape[a]) - 2 * h[a] for a in range(cfg.ndim))
        info = shard_info or {}
        offsets = tuple(info[a][0] if a in info else 0 for a in range(cfg.ndim))
        extents = tuple(int(info[a][1]) if a in info else local[a]
                        for a in range(cfg.ndim))
        kappa_pad = (static_padded or {}).get("kappa") if cfg.kappa else None
        source = bi.get("source")
        src = None
        if source is not None:
            source = jnp.asarray(source, f_pad.dtype)
            if tuple(source.shape) == tuple(f_pad.shape):
                src = _shift(source, h, None, 0)     # delivered halo-padded
            else:
                src = jnp.broadcast_to(source, local)
        f_new = self._kernel(f_pad, bi, dt, p, kappa_pad=kappa_pad,
                             offsets=offsets, extents=extents,
                             source_interior=src)
        out = {"f": jnp.pad(f_new, [(hh, hh) for hh in h])}
        if cfg.integral:
            out[cfg.integral_name] = self._integral(f_new)
        return out


@functools.lru_cache(maxsize=None)
def stencil_node_class(contract: str, reads_shard_info: bool, declares: bool):
    """The stencil node with the signatures a configuration asks for.

    The wrappers probe signatures (``params``, ``shard_info``), so the
    variants have to be distinct methods, not flags read inside one.
    """
    ns: dict = {}
    if contract == "params":
        def update(self, state, boundary_inputs, dt, *, params=None):
            return self._update(state, boundary_inputs, dt, params)
        if reads_shard_info:
            def update_padded(self, state_padded, boundary_inputs, dt, *,
                              static_padded=None, shard_info=None, params=None):
                return self._update_padded(state_padded, boundary_inputs, dt,
                                           static_padded, shard_info, params)
        else:
            def update_padded(self, state_padded, boundary_inputs, dt, *,
                              static_padded=None, params=None):
                return self._update_padded(state_padded, boundary_inputs, dt,
                                           static_padded, None, params)
    else:
        def update(self, state, boundary_inputs, dt):
            return self._update(state, boundary_inputs, dt, None)
        if reads_shard_info:
            def update_padded(self, state_padded, boundary_inputs, dt, *,
                              static_padded=None, shard_info=None):
                return self._update_padded(state_padded, boundary_inputs, dt,
                                           static_padded, shard_info, None)
        else:
            def update_padded(self, state_padded, boundary_inputs, dt, *,
                              static_padded=None):
                return self._update_padded(state_padded, boundary_inputs, dt,
                                           static_padded, None, None)
    ns["update"] = update
    ns["update_padded"] = update_padded
    if declares:
        def halo_boundary(self):
            return self.cfg.fill
        ns["halo_boundary"] = halo_boundary
    name = f"GenStencil_{contract}_{'info' if reads_shard_info else 'noinfo'}" \
           f"_{'declared' if declares else 'undeclared'}"
    return type(name, (_StencilBase,), ns)


def make_stencil_node(cfg: StencilConfig, **kw) -> SimulationNode:
    return stencil_node_class(cfg.contract, cfg.reads_shard_info, cfg.declares)(cfg, **kw)


# ---------------------------------------------------------------------------
# The pointwise family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointwiseConfig:
    n_devices: int
    shape: tuple
    shard_axis: int
    contract: str
    total: bool                # a 0-d whole-domain total in the state
    source: str
    misshapen_shape: tuple
    gain: bool
    dtype: str
    wrapping: str              # "single" | "nested" | "hybrid" | "hybrid_inside"
    steps: int
    seed: int
    surface: str
    family: str = "pointwise"

    @property
    def integral(self) -> Optional[str]:
        return "scalar" if self.total else None

    integral_name = "total"
    integral_shape = ()


class _PointwiseBase(SimulationNode):
    def __init__(self, cfg: PointwiseConfig, *, rate: float = 0.5,
                 decay: float = 0.25, timestep: float = 0.05) -> None:
        super().__init__(name=NODE_NAME, timestep=timestep, rate=rate, decay=decay)
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._x0 = rng.standard_normal(cfg.shape).astype(_np_dtype(cfg.dtype))

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        state = {"x": jnp.asarray(self._x0)}
        if self.cfg.total:
            state["total"] = jnp.zeros((), _np_dtype(self.cfg.dtype))
        return state

    def boundary_input_spec(self) -> dict:
        return {"source": BoundaryInputSpec(shape=tuple(self.cfg.shape),
                                            description="per-cell target"),
                "gain": BoundaryInputSpec(shape=(), description="scalar gain")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, None), transform="log", units="1/s"),
                "decay": ParamSpec(bounds=(0.0, None), units="1/s")}

    def _update(self, state, bi, dt, params):
        p = self.params if params is None else {**self.params, **params}
        x = state["x"]
        source = bi.get("source")
        target = (jnp.zeros_like(x) if source is None
                  else jnp.broadcast_to(jnp.asarray(source, x.dtype), x.shape))
        gain = jnp.asarray(bi.get("gain", 1.0), x.dtype)
        x_new = x + dt * p["rate"] * (gain * target - x) - dt * p["decay"] * x * x * x
        out = {"x": x_new}
        if self.cfg.total:
            out["total"] = jnp.sum(x_new + 1.0)
        return out


@functools.lru_cache(maxsize=None)
def pointwise_node_class(contract: str):
    if contract == "params":
        def update(self, state, boundary_inputs, dt, *, params=None):
            return self._update(state, boundary_inputs, dt, params)
    else:
        def update(self, state, boundary_inputs, dt):
            return self._update(state, boundary_inputs, dt, None)
    return type(f"GenPointwise_{contract}", (_PointwiseBase,), {"update": update})


# ---------------------------------------------------------------------------
# The unstructured family
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnstructuredConfig:
    n_devices: int
    n_cells: int
    assignment: tuple          # partition_assignment
    partition: str             # "uneven" | "balanced_global" | "balanced_nonglobal"
    chords: tuple              # extra edges beyond the ring
    contract: str
    integral: Optional[str]
    integral_name: str
    integral_listed: bool
    weight: Optional[str]      # partitioned StaticArray: None | "interior" | "halo"
    source: str
    misshapen_len: int
    gain: bool
    dtype: str
    wrapping: str              # "single" | "hybrid"
    steps: int
    seed: int
    surface: str
    exchange: str = "all_to_all"   # or "ppermute"
    family: str = "unstructured"

    @property
    def edges(self) -> np.ndarray:
        ring = [(i, (i + 1) % self.n_cells) for i in range(self.n_cells)]
        return np.asarray(ring + list(self.chords), dtype=np.int32).reshape(-1, 2)

    @property
    def integral_shape(self) -> tuple:
        return (2,) if self.integral == "vector" else ()


def unstructured_layout(cfg: UnstructuredConfig):
    return build_unstructured_partition(
        partition_assignment=np.asarray(cfg.assignment, dtype=np.int32),
        edges=cfg.edges, n_devices=cfg.n_devices)


class _UnstructuredBase(SimulationNode):
    """Relaxation towards the (weighted) mean of a cell's graph neighbours.

    One object plays both roles: ``update`` runs over the global
    connectivity, ``update_padded`` over a shard's slab through a
    partitioned neighbour table in slab indices (as
    ``property_support.UnstructuredRelaxNode`` does).
    """

    def __init__(self, cfg: UnstructuredConfig, layout, *, rate: float = 0.5,
                 decay: float = 0.25, timestep: float = 0.1) -> None:
        super().__init__(name=NODE_NAME, timestep=timestep, rate=rate, decay=decay)
        self.cfg = cfg
        dt = _np_dtype(cfg.dtype)
        rng = np.random.default_rng(cfg.seed)
        self._x0 = rng.standard_normal(cfg.n_cells).astype(dt)
        self._w = (0.5 + rng.random(cfg.n_cells)).astype(dt)
        self._table = self._neighbour_table(cfg)
        statics = {"slab_table": StaticArray(
            value=self._slab_table(layout), replication="partition",
            partition_assignment=layout.partition_assignment)}
        if cfg.weight:
            statics["w"] = StaticArray(value=jnp.asarray(self._w),
                                       replication="partition",
                                       partition_assignment=layout.partition_assignment)
        self._static = statics

    @staticmethod
    def _neighbour_table(cfg) -> np.ndarray:
        adjacency: list[list[int]] = [[] for _ in range(cfg.n_cells)]
        for u, v in cfg.edges:
            u, v = int(u), int(v)
            if u != v:
                if v not in adjacency[u]:
                    adjacency[u].append(v)
                if u not in adjacency[v]:
                    adjacency[v].append(u)
        width = max((len(a) for a in adjacency), default=1) or 1
        out = np.full((cfg.n_cells, width), -1, dtype=np.int32)
        for i, neigh in enumerate(adjacency):
            out[i, : len(neigh)] = sorted(neigh)
        return out

    def _slab_table(self, layout) -> np.ndarray:
        table = np.full_like(self._table, -1)
        ghost_slot = [{int(g): j for j, g in enumerate(layout.ghost_global_ids[d])}
                      for d in range(layout.n_devices)]
        for g in range(self.cfg.n_cells):
            owner = int(layout.partition_assignment[g])
            for j, nb in enumerate(self._table[g]):
                nb = int(nb)
                if nb < 0:
                    continue
                if int(layout.partition_assignment[nb]) == owner:
                    table[g, j] = layout.local_index_of(owner, nb)
                else:
                    table[g, j] = layout.n_local_max + ghost_slot[owner][nb]
        return table

    @property
    def static_data(self) -> dict:
        return self._static

    def halo_width(self) -> dict[int, int]:
        return {}

    def state_fields(self) -> list[str]:
        cfg = self.cfg
        if cfg.integral and cfg.integral_listed:
            return ["x", cfg.integral_name]
        return ["x"]

    def domain_integral_fields(self) -> set[str]:
        return {self.cfg.integral_name} if self.cfg.integral else set()

    def domain_integral_axes(self) -> dict:
        if self.cfg.integral == "per_shard":
            return {self.cfg.integral_name: ()}
        return {}

    def initial_state(self) -> dict:
        cfg = self.cfg
        state = {"x": jnp.asarray(self._x0)}
        if cfg.integral:
            state[cfg.integral_name] = jnp.zeros(cfg.integral_shape, _np_dtype(cfg.dtype))
        return state

    def boundary_input_spec(self) -> dict:
        return {"source": BoundaryInputSpec(shape=(self.cfg.n_cells,),
                                            description="per-cell forcing"),
                "gain": BoundaryInputSpec(shape=(), description="scalar gain")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, None), transform="log", units="1"),
                "decay": ParamSpec(bounds=(0.0, None), units="1/s")}

    def _kernel(self, x_slab, n_owned_rows, table, w_slab, src, bi, dt, p):
        """New values of the first ``n_owned_rows`` slab rows."""
        cfg = self.cfg
        valid = table >= 0
        idx = jnp.where(valid, table, 0)
        gathered = jnp.take(x_slab, idx, axis=0)
        if cfg.weight == "halo":
            wn = jnp.where(valid, jnp.take(w_slab, idx, axis=0), 0.0)
            total = jnp.where(valid, wn * gathered, 0.0).sum(axis=1)
            denom = jnp.maximum(wn.sum(axis=1), 1e-3)
        else:
            total = jnp.where(valid, gathered, 0.0).sum(axis=1)
            denom = jnp.maximum(valid.sum(axis=1), 1).astype(x_slab.dtype)
        owned = x_slab[:n_owned_rows]
        rate = p["rate"]
        if cfg.weight == "interior":
            rate = rate * w_slab[:n_owned_rows]
        x_new = owned + rate * (total / denom - owned) - dt * p["decay"] * owned
        if src is not None:
            gain = jnp.asarray(bi.get("gain", 1.0), x_slab.dtype)
            x_new = x_new + dt * gain * src
        return x_new

    def _integral(self, x_new, mask=None):
        terms = x_new + 1.0
        if mask is not None:
            terms = jnp.where(mask, terms, 0.0)
        total = jnp.sum(terms)
        if self.cfg.integral == "vector":
            return total * jnp.asarray([1.0, -2.0], x_new.dtype)
        return total

    def _update(self, state, bi, dt, params):
        cfg = self.cfg
        p = self.params if params is None else {**self.params, **params}
        x = state["x"]
        source = bi.get("source")
        src = (None if source is None
               else jnp.broadcast_to(jnp.asarray(source, x.dtype), x.shape))
        x_new = self._kernel(x, cfg.n_cells, jnp.asarray(self._table),
                             jnp.asarray(self._w, x.dtype), src, bi, dt, p)
        out = {"x": x_new}
        if cfg.integral:
            out[cfg.integral_name] = self._integral(x_new)
        return out

    def _update_padded(self, state_padded, bi, dt, static_padded, shard_info, params):
        cfg = self.cfg
        p = self.params if params is None else {**self.params, **params}
        x = state_padded["x"]
        n_local_max = int(shard_info[0][1])
        table = static_padded["slab_table"][:n_local_max]
        w_slab = static_padded["w"] if cfg.weight else jnp.ones_like(x)
        source = bi.get("source")
        src = None
        if source is not None:
            # A per-cell input arrives as this shard's slab (owned rows,
            # then ghosts); a scalar arrives whole.  The node cannot tell a
            # replicated array of the slab's length from a delivered slab.
            src = jnp.broadcast_to(jnp.asarray(source, x.dtype), x.shape)[:n_local_max]
        x_new = self._kernel(x, n_local_max, table, w_slab, src, bi, dt, p)
        out = {"x": jnp.concatenate([x_new, x[n_local_max:]])}
        if cfg.integral:
            mask = jnp.arange(n_local_max) < shard_info["n_local"]
            out[cfg.integral_name] = self._integral(x_new, mask)
        return out


@functools.lru_cache(maxsize=None)
def unstructured_node_class(contract: str):
    if contract == "params":
        def update(self, state, boundary_inputs, dt, *, params=None):
            return self._update(state, boundary_inputs, dt, params)

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None, params=None):
            return self._update_padded(state_padded, boundary_inputs, dt,
                                       static_padded, shard_info, params)
    else:
        def update(self, state, boundary_inputs, dt):
            return self._update(state, boundary_inputs, dt, None)

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None):
            return self._update_padded(state_padded, boundary_inputs, dt,
                                       static_padded, shard_info, None)
    return type(f"GenUnstructured_{contract}", (_UnstructuredBase,),
                {"update": update, "update_padded": update_padded})


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Every ``sampled_from`` below lists its richest value first, on purpose.
# Hypothesis biases generation towards the first element of a
# ``sampled_from`` and the low end of an ``integers``, and a per-push test
# sees only ``EXAMPLES_COSTLY`` (20) examples.  Ordered the other way,
# half of a per-push run was single-device on a one-cell grid and a
# quarter mis-shaped inputs that end at a refusal, and the per-push tests
# missed three of the harness mutants (a padded integral, and two dropped
# ``params``) that the same oracle catches on a richer draw.  So: four
# devices, the params contract, an integral, statics read in the halo, a
# composition of wrappers and the write surfaces come first; the trivial
# values stay reachable.


def _counts(*wanted: int) -> list[int]:
    return [n for n in wanted if n <= N_AVAILABLE] or [1]


def _misshapen_candidates(shape: tuple, block: dict, halo: tuple) -> list[tuple]:
    """Shapes the unsharded node refuses for a per-cell ``(shape)`` input.

    A shard's block (what MADD-ANO-057 was), a halo-padded grid, and the
    grid with one cell too many on the last axis.  Anything that happens
    to broadcast to the grid is dropped: that is an accepted value.
    """
    out = []
    n0 = shape[0]
    d0 = block.get(0, 1)
    out.append((n0 // d0,) + tuple(shape[1:]))
    out.append(tuple(n + 2 * h for n, h in zip(shape, halo)))
    out.append(tuple(shape[:-1]) + (shape[-1] + 1,))
    keep = []
    for cand in out:
        try:
            ok = np.broadcast_shapes(cand, shape) == tuple(shape)
        except ValueError:
            ok = False
        if not ok and cand not in keep:
            keep.append(cand)
    return keep


#: A mis-shaped per-cell input whose length is a shard's slab (owned plus
#: ghost rows) is taken by ``ShardedUnstructuredNode`` and read by every
#: shard as its own slab, where the unsharded node refuses it.  Excluded
#: from :func:`unstructured_configs` while that is pending a fix, so the
#: generated property stays green; the exact case is a strict xfail
#: (``test_a_slab_length_input_the_unsharded_node_refuses_is_refused_sharded``),
#: and the fix flips both.
PENDING_UNSTRUCTURED_SLAB_INPUT = True

#: A per-shard (unreduced) domain integral under a ``ShardedStencilNode``
#: nested in another has its initial value stacked by both wrappers'
#: ``initial_state``, so ``run_scan`` refuses the carry and ``step``
#: changes the state's shape.  Drawn as a vector integral instead while
#: that is pending a fix; the exact case is a strict xfail
#: (``test_a_per_shard_integral_under_a_nested_wrapper_runs_as_unsharded``).
PENDING_NESTED_PER_SHARD_INTEGRAL = True

#: Reverse mode through ``run_scan`` of a ``ShardedUnstructuredNode`` whose
#: partition has no ghost cell at all (no edge crosses a shard boundary --
#: every cell on one device, for instance) segfaults XLA's compiler on
#: jaxlib 0.11.2: ``exchange_unstructured`` returns the local block joined
#: to a zero-size ghost tail, and the scan's transpose of it crashes the
#: process.  A segfault cannot be an xfail in-process, so the gradient
#: surface is skipped on such partitions while that is pending a fix; the
#: exact case runs in a subprocess, a strict xfail on jaxlib 0.11.2
#: (``test_a_gradient_through_a_partition_without_ghosts_does_not_crash``).
PENDING_ZERO_GHOST_REVERSE_SCAN = True


@st.composite
def stencil_configs(draw, *, ndim: Optional[int] = None, pencils: Optional[bool] = None,
                    surfaces=FORWARD_SURFACES, dtypes=("float32", "float64"),
                    max_steps: int = 3) -> StencilConfig:
    """A configuration of the stencil family (see the module docstring)."""
    ndim = draw(st.sampled_from([2, 1])) if ndim is None else ndim
    if ndim == 1:
        n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
        mesh_shape, axis_names, axis_map = (n_dev,), ("devices",), (("devices", 0),)
    else:
        use_pencil = (draw(st.sampled_from([True, False])) if pencils is None
                      else pencils) and N_AVAILABLE >= 4
        if use_pencil:
            mesh_shape = draw(st.sampled_from([(2, 2), (1, 4), (4, 1)]))
            axis_names, axis_map = ("px", "py"), (("px", 0), ("py", 1))
        else:
            n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
            mesh_shape, axis_names = (n_dev,), ("devices",)
            axis_map = (("devices", draw(st.sampled_from([0, 1]))),)
    sizes = dict(zip(axis_names, mesh_shape))
    block = {sa: int(sizes[ma]) for ma, sa in axis_map}
    halo = tuple(draw(st.sampled_from([2, 1])) for _ in range(ndim))
    # At least ``halo`` cells per shard: a halo wider than a shard is a
    # known late refusal with a test of its own.
    shape = tuple(block.get(a, 1) * draw(st.integers(halo[a], halo[a] + 2))
                  for a in range(ndim))
    reads = draw(st.sampled_from([True, False]))
    source = draw(st.sampled_from(["per_cell", "scalar", "none", "misshapen"]))
    misshapen = ()
    if source == "misshapen":
        cands = _misshapen_candidates(shape, block, halo)
        misshapen = draw(st.sampled_from(cands))
    integral = draw(st.sampled_from(["vector", "per_shard", "scalar", None]))
    wrapping = draw(st.sampled_from(["nested", "hybrid", "single"]))
    if PENDING_NESTED_PER_SHARD_INTEGRAL and wrapping == "nested" and integral == "per_shard":
        integral = "vector"
    return StencilConfig(
        mesh_shape=tuple(mesh_shape), axis_names=tuple(axis_names),
        axis_map=tuple(axis_map), shape=shape, halo=halo,
        fill=draw(st.sampled_from(["periodic", "edge", "zero"])),
        declares=draw(st.sampled_from([True, False])),
        contract=draw(st.sampled_from(["params", "legacy"])),
        integral=integral,
        integral_name=draw(st.sampled_from(["a_total", "z_total"])),
        integral_listed=draw(st.sampled_from([True, False])),
        reads_shard_info=reads,
        kappa=draw(st.sampled_from(["halo", "interior", None])),
        kappa_axis=draw(st.sampled_from(list(range(ndim)))),
        table=draw(st.sampled_from(["halo", "interior", None])) if reads else None,
        source=source, misshapen_shape=misshapen,
        gain=draw(st.sampled_from([True, False])),
        faces=draw(st.sampled_from([True, False])) if reads else False,
        dtype=draw(st.sampled_from(list(dtypes))),
        wrapping=wrapping,
        steps=draw(st.integers(1, max_steps)),
        seed=draw(st.integers(0, 2**16)),
        surface=draw(st.sampled_from(list(surfaces))),
    )


@st.composite
def pointwise_configs(draw, *, surfaces=FORWARD_SURFACES,
                      dtypes=("float32", "float64"), max_steps: int = 3) -> PointwiseConfig:
    n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
    ndim = draw(st.sampled_from([2, 1]))
    shard_axis = draw(st.sampled_from(list(range(ndim))[::-1]))
    shape = tuple((n_dev if a == shard_axis else 1) * draw(st.integers(2, 3))
                  for a in range(ndim))
    source = draw(st.sampled_from(["per_cell", "scalar", "none", "misshapen"]))
    misshapen = ()
    if source == "misshapen":
        cands = _misshapen_candidates(shape, {shard_axis: n_dev} if shard_axis == 0 else {},
                                      (1,) * ndim)
        misshapen = draw(st.sampled_from(cands))
    return PointwiseConfig(
        n_devices=n_dev, shape=shape, shard_axis=shard_axis,
        contract=draw(st.sampled_from(["params", "legacy"])),
        total=draw(st.sampled_from([True, False])), source=source,
        misshapen_shape=misshapen, gain=draw(st.sampled_from([True, False])),
        dtype=draw(st.sampled_from(list(dtypes))),
        wrapping=draw(st.sampled_from(["nested", "hybrid_inside", "hybrid", "single"])),
        steps=draw(st.integers(1, max_steps)), seed=draw(st.integers(0, 2**16)),
        surface=draw(st.sampled_from(list(surfaces))),
    )


@st.composite
def unstructured_configs(draw, *, surfaces=FORWARD_SURFACES,
                         dtypes=("float32", "float64"),
                         max_steps: int = 3) -> UnstructuredConfig:
    n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
    kinds = ["uneven"] + (["balanced_nonglobal"] if n_dev > 1 else []) + ["balanced_global"]
    partition = draw(st.sampled_from(kinds))
    if partition == "uneven":
        n = draw(st.integers(max(3, n_dev), 3 * n_dev + 2))
        assignment = tuple(draw(st.lists(st.integers(0, n_dev - 1), min_size=n, max_size=n)))
    else:
        # Every shard full: n_devices * per cells, at least 3 for a ring.
        per = max(draw(st.integers(1, 3)), -(-3 // n_dev))
        n = n_dev * per
        if partition == "balanced_global":
            assignment = tuple(int(i // per) for i in range(n))
        else:
            assignment = tuple(int(i % n_dev) for i in range(n))
    chords = tuple(draw(st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)),
                                 max_size=3)))
    source = draw(st.sampled_from(["per_cell", "scalar", "none", "misshapen"]))
    integral = draw(st.sampled_from(["vector", "per_shard", "scalar", None]))
    cfg = UnstructuredConfig(
        n_devices=n_dev, n_cells=n, assignment=assignment, partition=partition,
        chords=chords, contract=draw(st.sampled_from(["params", "legacy"])),
        integral=integral,
        integral_name=draw(st.sampled_from(["a_total", "z_total"])),
        integral_listed=draw(st.sampled_from([True, False])),
        weight=draw(st.sampled_from(["halo", "interior", None])),
        source=source, misshapen_len=0, gain=draw(st.sampled_from([True, False])),
        dtype=draw(st.sampled_from(list(dtypes))),
        wrapping=draw(st.sampled_from(["hybrid", "single"])),
        steps=draw(st.integers(1, max_steps)), seed=draw(st.integers(0, 2**16)),
        surface=draw(st.sampled_from(list(surfaces))),
        exchange=draw(st.sampled_from(["all_to_all", "ppermute"])),
    )
    if source == "misshapen":
        layout = unstructured_layout(cfg)
        n_layout = layout.n_devices * layout.n_local_max
        slab = layout.n_local_max + layout.n_ghost_max
        pending = {slab} if PENDING_UNSTRUCTURED_SLAB_INPUT else set()
        cands = sorted({n + 1, slab, layout.n_local_max} - {1, n, n_layout} - pending)
        if not cands:
            cands = [n + 1] if n + 1 != n_layout else [n + 2]
        cfg = replace(cfg, misshapen_len=draw(st.sampled_from(cands)))
    return cfg


# ---------------------------------------------------------------------------
# Building the two paths
# ---------------------------------------------------------------------------


def _correction(state, boundary_inputs, dt):
    """The ``HybridNode`` correction: elementwise, on the first grid field only."""
    key = "f" if "f" in state else "x"
    v = state[key]
    return {key: -0.05 * dt * v * v}


@dataclass
class Case:
    """Both paths of one configuration, built fresh on every call."""

    cfg: Any
    make: Callable[[bool], SimulationNode]       # sharded? -> outermost node
    externals: Callable[[bool], dict]           # sharded? -> {field: value}
    gather: Callable[[bool, dict], dict]         # sharded?, node state -> global numpy
    gather_traced: Callable[[bool, dict], dict]  # same, for a traced state
    inner_of: Callable[[SimulationNode], SimulationNode]  # node -> the innermost node
    #: ``None`` when the sharded composition must run; otherwise
    #: ``(stage, exception type, regex)`` it must be refused with, where
    #: stage is ``"construct"`` or ``"run"`` (the first step or scan,
    #: before any state is produced).
    refusal: Optional[tuple] = None
    known: Optional[str] = None
    layout: Any = None
    #: The unsharded node refuses the same configuration (a mis-shaped
    #: input); otherwise the refusal is the sharded path's own and the
    #: unsharded node runs.
    unsharded_refuses: bool = False
    #: The node's name in the graph.
    name: str = NODE_NAME


def _innermost(node):
    while True:
        nxt = getattr(node, "_inner", None) or getattr(node, "physics_node", None)
        if nxt is None:
            return node
        node = nxt


def _stencil_case(cfg: StencilConfig) -> Case:
    n_dev_total = int(np.prod(cfg.mesh_shape))
    mesh_args = dict(shape=tuple(cfg.mesh_shape),
                     axis_names=tuple(cfg.axis_names))
    axis_map = dict(cfg.axis_map)

    def make(sharded: bool) -> SimulationNode:
        node = make_stencil_node(cfg)
        if not sharded:
            return HybridNode(node, _correction) if cfg.wrapping == "hybrid" else node
        mesh = create_device_mesh(**mesh_args)
        wrapped = ShardedStencilNode(node, mesh, axis_map, boundary=cfg.fill)
        if cfg.wrapping == "nested":
            wrapped = ShardedStencilNode(wrapped, mesh, axis_map, boundary=cfg.fill)
        elif cfg.wrapping == "hybrid":
            wrapped = HybridNode(wrapped, _correction)
        return wrapped

    dt = _np_dtype(cfg.dtype)
    rng = np.random.default_rng(cfg.seed + 1)

    def externals(sharded: bool) -> dict:
        ext = {}
        r = np.random.default_rng(cfg.seed + 1)
        if cfg.source == "per_cell":
            ext["source"] = r.standard_normal(cfg.shape).astype(dt)
        elif cfg.source == "scalar":
            ext["source"] = np.asarray(r.standard_normal(), dt)
        elif cfg.source == "misshapen":
            ext["source"] = r.standard_normal(cfg.misshapen_shape).astype(dt)
        if cfg.gain:
            ext["gain"] = np.asarray(0.75, dt)
        if cfg.faces:
            ext["lo"] = (1.0 + r.random(cfg.shape[1:])).astype(dt)
            ext["hi"] = (-1.0 - r.random(cfg.shape[1:])).astype(dt)
        return ext

    del rng, n_dev_total

    refusal = None
    if cfg.kappa and cfg.kappa_axis not in cfg.sharded_axes:
        refusal = ("construct", ValueError, r"declares shard_axis")
    elif cfg.source == "misshapen":
        refusal = ("run", ValueError, r"boundary input 'source' has shape")

    def gather(sharded: bool, state: dict) -> dict:
        return {k: np.asarray(jax.device_get(v)) for k, v in state.items()}

    def gather_traced(sharded: bool, state: dict) -> dict:
        return dict(state)

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=gather_traced, inner_of=_innermost, refusal=refusal,
                unsharded_refuses=cfg.source == "misshapen")


def _pointwise_case(cfg: PointwiseConfig) -> Case:
    cls = pointwise_node_class(cfg.contract)

    def make(sharded: bool) -> SimulationNode:
        node = cls(cfg)
        if not sharded:
            return HybridNode(node, _correction) \
                if cfg.wrapping in ("hybrid", "hybrid_inside") else node
        mesh = create_device_mesh(shape=(cfg.n_devices,))
        if cfg.wrapping == "hybrid_inside":
            return ShardedPointwiseNode(HybridNode(node, _correction), mesh,
                                        shard_axes=(cfg.shard_axis,))
        wrapped = ShardedPointwiseNode(node, mesh, shard_axes=(cfg.shard_axis,))
        if cfg.wrapping == "nested":
            wrapped = ShardedPointwiseNode(wrapped, mesh, shard_axes=(cfg.shard_axis,))
        elif cfg.wrapping == "hybrid":
            wrapped = HybridNode(wrapped, _correction)
        return wrapped

    dt = _np_dtype(cfg.dtype)

    def externals(sharded: bool) -> dict:
        ext = {}
        r = np.random.default_rng(cfg.seed + 1)
        if cfg.source == "per_cell":
            ext["source"] = r.standard_normal(cfg.shape).astype(dt)
        elif cfg.source == "scalar":
            ext["source"] = np.asarray(r.standard_normal(), dt)
        elif cfg.source == "misshapen":
            ext["source"] = r.standard_normal(cfg.misshapen_shape).astype(dt)
        if cfg.gain:
            ext["gain"] = np.asarray(0.75, dt)
        return ext

    # The pointwise wrapper validates no inputs: the inner update's own
    # broadcast refuses a mis-shaped one, on both paths alike.
    refusal = (("run", (ValueError, TypeError), r"broadcast|[Ii]ncompatible shapes")
               if cfg.source == "misshapen" else None)

    def gather(sharded, state):
        return {k: np.asarray(jax.device_get(v)) for k, v in state.items()}

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=lambda sharded, s: dict(s), inner_of=_innermost,
                refusal=refusal, unsharded_refuses=cfg.source == "misshapen")


def layout_rows_of_global(layout) -> np.ndarray:
    """``rows[g]`` = the partition-layout row holding global cell ``g``."""
    n = int(np.asarray(layout.partition_assignment).size)
    rows = np.zeros(n, dtype=np.int32)
    for d, ids in enumerate(layout.local_global_ids):
        for j, g in enumerate(np.asarray(ids)):
            rows[int(g)] = d * layout.n_local_max + j
    return rows


def _unstructured_case(cfg: UnstructuredConfig) -> Case:
    layout = unstructured_layout(cfg)
    cls = unstructured_node_class(cfg.contract)
    rows = layout_rows_of_global(layout)
    n_layout = layout.n_devices * layout.n_local_max

    def make(sharded: bool) -> SimulationNode:
        node = cls(cfg, layout)
        if sharded:
            node = ShardedUnstructuredNode(node, create_device_mesh(shape=(cfg.n_devices,)),
                                           layout, exchange=cfg.exchange)
        return HybridNode(node, _correction) if cfg.wrapping == "hybrid" else node

    dt = _np_dtype(cfg.dtype)

    def to_layout(value: np.ndarray) -> np.ndarray:
        per_shard = partition_value(value=value, layout=layout)
        return per_shard.reshape((n_layout,) + per_shard.shape[2:])

    def externals(sharded: bool) -> dict:
        ext = {}
        r = np.random.default_rng(cfg.seed + 1)
        if cfg.source == "per_cell":
            src = r.standard_normal(cfg.n_cells).astype(dt)
            ext["source"] = to_layout(src) if sharded else src
        elif cfg.source == "scalar":
            ext["source"] = np.asarray(r.standard_normal(), dt)
        elif cfg.source == "misshapen":
            ext["source"] = r.standard_normal(cfg.misshapen_len).astype(dt)
        if cfg.gain:
            ext["gain"] = np.asarray(0.75, dt)
        return ext

    refusal = None
    known = None
    if cfg.source == "per_cell" and cfg.partition == "balanced_nonglobal":
        # Known and decided: on a balanced partition out of global order a
        # global-order array and a partition-layout one have the same shape.
        refusal = ("run", ValueError, r"cannot tell a global-order array")
        known = "balanced non-global partition refuses per-cell inputs"
    elif cfg.source == "misshapen":
        refusal = ("run", (ValueError, TypeError),
                   r"boundary input 'source'|broadcast|[Ii]ncompatible shapes")

    state_set = {"x"}

    def gather(sharded: bool, state: dict) -> dict:
        out = {}
        for k, v in state.items():
            host = np.asarray(jax.device_get(v))
            out[k] = host[rows] if (sharded and k in state_set) else host
        return out

    def gather_traced(sharded: bool, state: dict) -> dict:
        return {k: (jnp.take(v, jnp.asarray(rows), axis=0)
                    if (sharded and k in state_set) else v)
                for k, v in state.items()}

    def inner_of(node):
        return _innermost(node)

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=gather_traced, inner_of=inner_of, refusal=refusal,
                known=known, layout=layout,
                unsharded_refuses=cfg.source == "misshapen")


# ---------------------------------------------------------------------------
# Built-in nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatConfig:
    """A ``HeatNode`` rod, sharded along its one axis."""

    n_devices: int
    order: int                 # stencil_order, 2 or 4
    n_cells: int
    nonuniform: bool
    ends: bool                 # left/right temperature inputs
    source: str                # "none" | "per_cell" | "scalar" | "misshapen"
    wrapping: str              # "single" | "nested" | "hybrid"
    steps: int
    seed: int
    surface: str
    dtype: str = "float32"
    contract: str = "params"
    integral: Optional[str] = None
    family: str = "heat"


@dataclass(frozen=True)
class LBMConfig:
    """An ``LBMNode`` channel with walls and an obstacle."""

    lattice: str               # "D2Q9" | "D3Q19"
    mesh_shape: tuple
    axis_names: tuple
    axis_map: tuple
    shape: tuple
    force: str                 # "none" | "uniform" | "per_cell" | "misshapen"
    pressure: bool             # inlet/outlet pressure on the x faces
    wrapping: str
    steps: int
    seed: int
    surface: str
    dtype: str = "float32"
    contract: str = "params"
    integral: Optional[str] = None
    family: str = "lbm"


#: The built-in node's differentiable constant each case writes and differentiates.
BUILTIN_PARAM = {"heat": "thermal_diffusivity", "lbm": "viscosity"}


def _heat_node(cfg: HeatConfig):
    from maddening.nodes.heat import HeatNode  # noqa: PLC0415
    n = cfg.n_cells
    rng = np.random.default_rng(cfg.seed)
    t0 = (300.0 + 40.0 * np.sin(np.linspace(0.0, 3.0, n))
          + rng.standard_normal(n)).tolist()
    alpha = 0.01
    if cfg.nonuniform:
        # Strictly increasing, inside the rod; Fourier number 0.2 on the
        # node's own spacing (#167 refuses an unstable non-uniform rod).
        from maddening.nodes.heat import _nonuniform_fourier_spacing  # noqa: PLC0415
        gaps = 0.5 + rng.random(n)
        x = np.cumsum(gaps)
        x = (x - x[0]) / (x[-1] - x[0]) * 0.9 + 0.05
        dt = 0.2 * _nonuniform_fourier_spacing(x.tolist()) / alpha
        return HeatNode(NODE_NAME, dt, n_cells=n, thermal_diffusivity=alpha,
                        initial_temperature=t0, stencil_order=2,
                        grid_points=x.tolist())
    dx = 1.0 / n
    limit = 0.2 if cfg.order == 2 else 0.15
    return HeatNode(NODE_NAME, limit * dx * dx / alpha, n_cells=n,
                    thermal_diffusivity=alpha, initial_temperature=t0,
                    stencil_order=cfg.order)


def _heat_case(cfg: HeatConfig) -> Case:
    def make(sharded: bool) -> SimulationNode:
        node = _heat_node(cfg)
        if not sharded:
            return HybridNode(node, _heat_correction) if cfg.wrapping == "hybrid" else node
        mesh = create_device_mesh(shape=(cfg.n_devices,))
        wrapped = ShardedStencilNode(node, mesh, {"devices": 0}, boundary="edge")
        if cfg.wrapping == "nested":
            wrapped = ShardedStencilNode(wrapped, mesh, {"devices": 0}, boundary="edge")
        elif cfg.wrapping == "hybrid":
            wrapped = HybridNode(wrapped, _heat_correction)
        return wrapped

    def externals(sharded: bool) -> dict:
        r = np.random.default_rng(cfg.seed + 1)
        ext = {}
        if cfg.ends:
            ext["left_temperature"] = np.asarray(250.0, np.float32)
            ext["right_temperature"] = np.asarray(380.0, np.float32)
        n = cfg.n_cells
        if cfg.source == "per_cell":
            ext["heat_source"] = (5.0 * r.standard_normal(n)).astype(np.float32)
        elif cfg.source == "scalar":
            ext["heat_source"] = np.asarray(3.0, np.float32)
        elif cfg.source == "misshapen":
            bad = n // cfg.n_devices if cfg.n_devices > 1 else n + 1
            if bad in (1, n):
                bad = n + 1
            ext["heat_source"] = r.standard_normal(bad).astype(np.float32)
        return ext

    refusal = known = None
    misshapen = (ValueError, r"boundary input 'heat_source' has shape")
    if cfg.nonuniform:
        # Loud, at the first step; a mis-shaped source is refused first.
        refusal = ("run", (NotImplementedError, ValueError),
                   r"non-uniform grids under sharding"
                   + (f"|{misshapen[1]}" if cfg.source == "misshapen" else ""))
        known = "HeatNode.update_padded does not support a non-uniform grid"
    elif cfg.source == "misshapen":
        refusal = ("run",) + misshapen

    def gather(sharded, state):
        return {k: np.asarray(jax.device_get(v)) for k, v in state.items()}

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=lambda sharded, st_: dict(st_), inner_of=_innermost,
                refusal=refusal, known=known,
                unsharded_refuses=cfg.source == "misshapen")


def _heat_correction(state, boundary_inputs, dt):
    t = state["temperature"]
    return {"temperature": -1e-4 * dt * (t - 300.0)}


def _lbm_node(cfg: LBMConfig):
    from maddening.nodes.lbm import LBMNode  # noqa: PLC0415
    shape = tuple(cfg.shape)
    mask = np.zeros(shape, dtype=bool)
    # Channel walls on the last axis, and a one-cell obstacle off-centre.
    mask[..., 0] = True
    mask[..., -1] = True
    rng = np.random.default_rng(cfg.seed)
    obstacle = tuple(int(rng.integers(0, n)) for n in shape[:-1]) + \
        (int(rng.integers(1, shape[-1] - 1)),)
    mask[obstacle] = True
    return LBMNode(NODE_NAME, 1.0, grid_shape=shape, viscosity=0.1,
                   lattice=cfg.lattice, wall_mask=mask)


def _lbm_case(cfg: LBMConfig) -> Case:
    axis_map = dict(cfg.axis_map)
    D = len(cfg.shape)

    def make(sharded: bool) -> SimulationNode:
        node = _lbm_node(cfg)
        if not sharded:
            return HybridNode(node, _correction) if cfg.wrapping == "hybrid" else node
        mesh = create_device_mesh(shape=tuple(cfg.mesh_shape), axis_names=tuple(cfg.axis_names))
        wrapped = ShardedStencilNode(node, mesh, axis_map, boundary="periodic")
        if cfg.wrapping == "nested":
            wrapped = ShardedStencilNode(wrapped, mesh, axis_map, boundary="periodic")
        elif cfg.wrapping == "hybrid":
            wrapped = HybridNode(wrapped, _correction)
        return wrapped

    def externals(sharded: bool) -> dict:
        r = np.random.default_rng(cfg.seed + 1)
        ext = {}
        if cfg.force == "uniform":
            ext["body_force"] = (1e-4 * (1.0 + r.random(D))).astype(np.float32)
        elif cfg.force == "per_cell":
            ext["body_force"] = (1e-4 * r.standard_normal(tuple(cfg.shape) + (D,))
                                 ).astype(np.float32)
        elif cfg.force == "misshapen":
            bad = (cfg.shape[0] + 1,) + tuple(cfg.shape[1:]) + (D,)
            ext["body_force"] = (1e-4 * r.standard_normal(bad)).astype(np.float32)
        if cfg.pressure:
            ext["inlet_pressure"] = np.asarray(1.002 / 3.0, np.float32)
            ext["outlet_pressure"] = np.asarray(0.998 / 3.0, np.float32)
        return ext

    sizes = dict(zip(cfg.axis_names, cfg.mesh_shape))
    split_x = any(sa == 0 and sizes[ma] > 1 for ma, sa in cfg.axis_map)
    refusal = None
    if cfg.force == "misshapen":
        refusal = ("run", ValueError, r"boundary input 'body_force' has shape")
    elif cfg.pressure and split_x:
        refusal = ("run", ValueError, r"imposes inlet_pressure")

    def gather(sharded, state):
        return {k: np.asarray(jax.device_get(v)) for k, v in state.items()}

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=lambda sharded, st_: dict(st_), inner_of=_innermost,
                refusal=refusal, unsharded_refuses=cfg.force == "misshapen")


@st.composite
def heat_configs(draw, *, surfaces=FORWARD_SURFACES, max_steps: int = 4) -> HeatConfig:
    n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
    order = draw(st.sampled_from([4, 2]))
    halo = 1 if order == 2 else 2
    per = draw(st.integers(halo, halo + 3))
    n = n_dev * per
    while order == 4 and n < 5:
        n += n_dev
    nonuniform = draw(st.booleans()) if order == 2 else False
    while nonuniform and n < 3:
        n += n_dev
    return HeatConfig(
        n_devices=n_dev, order=order, n_cells=n, nonuniform=nonuniform,
        ends=draw(st.sampled_from([True, False])),
        source=draw(st.sampled_from(["per_cell", "scalar", "none", "misshapen"])),
        wrapping=draw(st.sampled_from(["nested", "hybrid", "single"])),
        steps=draw(st.integers(1, max_steps)), seed=draw(st.integers(0, 2**16)),
        surface=draw(st.sampled_from([s for s in surfaces if s != "set_state"]
                                     or list(surfaces))))


@st.composite
def lbm_configs(draw, *, lattices=("D2Q9", "D3Q19"), surfaces=FORWARD_SURFACES,
                max_steps: int = 3, vary_cells: bool = True) -> LBMConfig:
    """An LBM channel configuration.

    ``vary_cells=False`` fixes the cells per shard, so the grid shape
    depends on the mesh alone: ``LBMNode``'s constructor and
    ``initial_state`` run eagerly, and compile each operation afresh for
    every new grid shape (about 0.4 s an example), which a per-push test
    cannot afford on every draw.
    """
    lattice = draw(st.sampled_from(list(lattices)))
    D = 2 if lattice == "D2Q9" else 3
    if draw(st.sampled_from([True, False])) and N_AVAILABLE >= 4:
        mesh_shape = draw(st.sampled_from([(2, 2), (1, 4), (4, 1)]))
        axis_names, axis_map = ("px", "py"), (("px", 0), ("py", 1))
    else:
        n_dev = draw(st.sampled_from(_counts(4, 2, 1)))
        mesh_shape, axis_names = (n_dev,), ("devices",)
        axis_map = (("devices", draw(st.sampled_from(list(range(D))))),)
    sizes = dict(zip(axis_names, mesh_shape))
    block = {sa: int(sizes[ma]) for ma, sa in axis_map}
    shape = []
    for a in range(D):
        lo = 3 if a == D - 1 else 1     # room for two walls and a fluid row
        c = draw(st.integers(lo, lo + 1)) if (D == 2 and vary_cells) else lo
        shape.append(block.get(a, 1) * c)
    return LBMConfig(
        lattice=lattice, mesh_shape=tuple(mesh_shape), axis_names=tuple(axis_names),
        axis_map=tuple(axis_map), shape=tuple(shape),
        force=draw(st.sampled_from(["per_cell", "uniform", "none", "misshapen"])),
        pressure=draw(st.booleans()),
        wrapping=draw(st.sampled_from(["nested", "hybrid", "single"])),
        steps=draw(st.integers(1, max_steps)), seed=draw(st.integers(0, 2**16)),
        surface=draw(st.sampled_from([s for s in surfaces if s != "set_state"]
                                     or list(surfaces))))


@dataclass(frozen=True)
class ExampleConfig:
    """One of the wrapper-family examples in ``property_support``, in a graph."""

    label: str                 # "pointwise" | "stencil" | "unstructured"
    n_devices: int
    steps: int
    surface: str
    seed: int = 0
    dtype: str = "float32"
    contract: str = "params"
    integral: Optional[str] = None
    family: str = "example"


def _example_case(cfg: ExampleConfig) -> Case:
    from tests.cloud.multigpu.property_support import WRAPPER_FAMILY  # noqa: PLC0415

    def build():
        return WRAPPER_FAMILY[cfg.label](n_devices=cfg.n_devices, rate=0.5)

    probe = build()
    name = probe.inner.name
    layout = getattr(probe.wrapped, "layout", None)
    rows = layout_rows_of_global(layout) if layout is not None else None

    def make(sharded: bool) -> SimulationNode:
        c = build()
        return c.wrapped if sharded else c.inner

    def externals(sharded: bool) -> dict:
        return {k: np.asarray(v) for k, v in probe.boundary_inputs.items()}

    def gather(sharded, state):
        out = {}
        for k, v in state.items():
            host = np.asarray(jax.device_get(v))
            out[k] = host[rows] if (sharded and rows is not None) else host
        return out

    def gather_traced(sharded, state):
        if not (sharded and rows is not None):
            return dict(state)
        return {k: jnp.take(v, jnp.asarray(rows), axis=0) for k, v in state.items()}

    return Case(cfg=cfg, make=make, externals=externals, gather=gather,
                gather_traced=gather_traced, inner_of=_innermost, name=name,
                layout=layout)


def build_case(cfg) -> Case:
    return {"stencil": _stencil_case, "pointwise": _pointwise_case,
            "unstructured": _unstructured_case, "heat": _heat_case,
            "lbm": _lbm_case, "example": _example_case}[cfg.family](cfg)


# ---------------------------------------------------------------------------
# Running one surface on one path
# ---------------------------------------------------------------------------


def build_graph(case: Case, sharded: bool, node: Optional[SimulationNode] = None):
    node = case.make(sharded) if node is None else node
    gm = GraphManager()
    gm.add_node(node)
    for fld, value in case.externals(sharded).items():
        value = np.asarray(value)
        gm.add_external_input(target_node=node.name, target_field=fld,
                              shape=tuple(value.shape), dtype=value.dtype)
    gm.compile()
    return gm, node


def _ext(case: Case, sharded: bool) -> Optional[dict]:
    ext = case.externals(sharded)
    return {case.name: {k: jnp.asarray(v) for k, v in ext.items()}} if ext else None


#: ``family -> (parameter a write surface moves, its constructed value)``.
_PARAM = {"stencil": ("rate", 0.5), "pointwise": ("rate", 0.5),
          "unstructured": ("rate", 0.5), "heat": ("thermal_diffusivity", 0.01),
          "lbm": ("viscosity", 0.1), "example": ("rate", 0.5)}


def param_name(cfg) -> str:
    return _PARAM[cfg.family][0]


def grid_key(case: Case) -> str:
    """The node's (first) grid field."""
    fam = case.cfg.family
    if fam == "stencil":
        return "f"
    if fam in ("pointwise", "unstructured"):
        return "x"
    return next(iter(case.make(False).initial_state()))


def written_rate(cfg) -> float:
    """The value a write surface puts into :func:`param_name`.

    Between 1.07 and 1.5 times the constructed value: far enough from it
    that the trajectory moves by orders of magnitude more than any
    tolerance, and inside every node's stability limit (a ``HeatNode``
    built at Fourier number 0.2 or 0.15 stays below 1/2 and 5/16).
    """
    return _PARAM[cfg.family][1] * (1.0 + 0.5 * ((cfg.seed % 7) + 1) / 7.0)


def _final(gm, case: Case, sharded: bool) -> dict:
    return case.gather(sharded, gm.get_node_state(case.name))


def run_surface(case: Case, surface: str, sharded: bool):
    """The global final state (or, for ``gradient``, ``(loss, grad)``) on one path."""
    cfg = case.cfg
    n = cfg.steps
    if surface == "gradient":
        return _gradient(case, sharded)
    gm, node = build_graph(case, sharded)
    ext = _ext(case, sharded)
    if surface == "step":
        for _ in range(n):
            gm.step(ext)
    elif surface == "run":
        gm.run(n, external_inputs=ext)
    elif surface == "run_scan":
        gm.run_scan(n, ext)
    elif surface == "scan_params":
        if cfg.contract == "params":
            p = jax.tree.map(lambda x: x, gm.params)
            leaf = p["nodes"][case.name][param_name(cfg)]
            p["nodes"][case.name][param_name(cfg)] = jnp.asarray(written_rate(cfg),
                                                                 leaf.dtype)
            gm.run_scan(n, ext, params=p)
        else:
            # A node on the three-argument contract is absent from
            # gm.params: an entry for it is refused, on both paths.
            p = {"nodes": {case.name: {param_name(cfg): jnp.asarray(written_rate(cfg))}}}
            try:
                gm.run_scan(n, ext, params=p)
            except ValueError as e:
                # The message names the outermost node's class, which is
                # the wrapper on one path and the node on the other.
                return {"__refused__": "takes no 'params' keyword" in str(e)}
            return {"__refused__": False}
    elif surface == "write_compile":
        # Run before the write, so a trace the write must invalidate
        # exists: a recompile that kept the wrapper's compiled step would
        # step the old value from here on (MADD-ANO-032).
        gm.run_scan(n, ext)
        if cfg.contract == "params":
            leaf = gm.params["nodes"][case.name][param_name(cfg)]
            gm.params["nodes"][case.name][param_name(cfg)] = jnp.asarray(
                written_rate(cfg), leaf.dtype)
        else:
            case.inner_of(node).params[param_name(cfg)] = written_rate(cfg)
        gm.compile()
        gm.run_scan(n, ext)
    elif surface == "set_state":
        state = new_global_state(case, gm, sharded)
        gm.set_node_state(case.name, state)
        gm.run_scan(n, ext)
    else:  # pragma: no cover - a typo in a strategy
        raise ValueError(f"unknown surface {surface!r}")
    return _final(gm, case, sharded)


def new_global_state(case: Case, gm, sharded: bool) -> dict:
    """A state to write with ``set_node_state``: new grid values, integrals kept.

    Cartesian wrappers take the global array; the unstructured one takes
    it through ``partition_value`` (the documented conversion).
    """
    cfg = case.cfg
    current = gm.get_node_state(case.name)
    key = grid_key(case)
    ref = np.asarray(jax.device_get(case.make(False).initial_state()[key]))
    new = (np.random.default_rng(cfg.seed + 7).standard_normal(ref.shape)
           .astype(ref.dtype))
    if sharded and case.layout is not None:
        per = partition_value(value=new, layout=case.layout)
        new = per.reshape((-1,) + per.shape[2:])
    out = dict(current)
    out[key] = jnp.asarray(new)
    return out


def loss_of(case: Case, state: dict, sharded: bool, initial: dict):
    """Sum of squared departures from the initial state, over every field.

    The departure rather than the state itself: the built-in nodes start
    far from zero (a rod near 300 K, populations near their weights), and
    the sum of squares of such a state has a gradient that is a small
    difference of large terms -- a comparison of rounding, not of the two
    paths.  A per-shard integral is summed over its shards first.
    """
    cfg = case.cfg
    g = case.gather_traced(sharded, state)
    g0 = case.gather_traced(sharded, initial)
    total = 0.0
    for k, v in g.items():
        d = v - g0[k]
        if k == getattr(cfg, "integral_name", None) and cfg.integral == "per_shard" and sharded:
            lead = jnp.ndim(d) - len(cfg.integral_shape)
            d = jnp.sum(d, axis=tuple(range(lead)))
        if not jnp.issubdtype(jnp.asarray(d).dtype, jnp.floating):
            continue
        total = total + jnp.sum(d * d)
    return total


def _theta_runner(case: Case, sharded: bool):
    """``(initial state, theta0, theta -> final node state)`` on a fresh graph.

    ``theta`` is the node's differentiable constant through ``params=``
    for a params-contract node, and the external ``gain`` for a node on
    the three-argument contract (which ``gm.params`` does not carry);
    ``None`` when there is neither.
    """
    cfg = case.cfg
    gm, _ = build_graph(case, sharded)
    ext = _ext(case, sharded) or {case.name: {}}
    dtype = jnp.zeros((), _np_dtype(cfg.dtype)).dtype
    initial = gm.get_node_state(case.name)
    if cfg.contract == "params":
        theta0 = jnp.asarray(written_rate(cfg), dtype)

        def run(theta):
            return gm.run_scan(cfg.steps, ext,
                               params={"nodes": {case.name: {param_name(cfg): theta}}}
                               )[case.name]
    else:
        if "gain" not in ext[case.name]:
            return None
        theta0 = jnp.asarray(0.75, dtype)

        def run(theta):
            return gm.run_scan(cfg.steps, {case.name: {**ext[case.name], "gain": theta}}
                               )[case.name]
    return initial, theta0, run


def _gradient(case: Case, sharded: bool):
    """``(loss, d loss / d theta)`` through ``run_scan`` (see :func:`_theta_runner`)."""
    if (PENDING_ZERO_GHOST_REVERSE_SCAN and case.layout is not None
            and int(case.layout.n_ghost_max) == 0):
        return None
    got = _theta_runner(case, sharded)
    if got is None:
        return None
    initial, theta0, run = got
    value, grad = jax.value_and_grad(
        lambda t: loss_of(case, run(t), sharded, initial))(theta0)
    return float(value), float(grad)


def gradient_scales(case: Case) -> dict:
    """What :func:`gradient_bound` is built from, on the unsharded path.

    One forward-mode pass gives the final state ``s``, its tangent
    ``ds = d s / d theta``, the departure ``dev = s - s0`` over every
    floating field, and the loss's own forward-mode derivative ``g_fwd``
    in the run's precision.  ``A = sum|2 dev ds|`` (the gradient's terms
    before they cancel), ``B = sum|ds|``, ``S = sum|dev|``, ``Q = sum
    dev**2`` (the loss), ``T = max|ds|`` and ``M = max|s|``.
    """
    initial, theta0, run = _theta_runner(case, False)

    def flat(state):
        g = case.gather_traced(False, state)
        parts = [jnp.ravel(v) for v in g.values()
                 if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]
        return jnp.concatenate(parts)

    def both(t):
        final = run(t)
        return flat(final), loss_of(case, final, False, initial)

    s0 = np.asarray(flat(initial), np.float64)
    (s, _), (ds, g_fwd) = jax.jvp(both, (theta0,), (jnp.ones_like(theta0),))
    s, ds = np.asarray(s, np.float64), np.asarray(ds, np.float64)
    dev = s - s0
    return {"A": float(np.sum(np.abs(2.0 * dev * ds))), "B": float(np.sum(np.abs(ds))),
            "S": float(np.sum(np.abs(dev))), "Q": float(np.sum(dev * dev)),
            "T": float(np.max(np.abs(ds))), "M": float(np.max(np.abs(s))),
            "n": int(s.size), "g_fwd": float(g_fwd)}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def eps_of(dtype: str) -> float:
    return float(np.finfo(_np_dtype(dtype)).eps)


#: ``(kind, dtype, observed difference / bound)`` for every comparison made
#: in this process: what the measured margins in the testing standards
#: were read from.  Nothing asserts on it.
OBSERVED: list = []

#: Fused multiply-add contraction sites per cell per step that may round
#: differently on the two paths (see :func:`grid_atol`).
FMA_SITES = 16
#: Bound on how much one step of a harness node can amplify a difference
#: already in the state (its explicit update is a contraction plus at most
#: ``dt * rate * coef * sum|stencil weights|`` < 0.5).
STEP_GROWTH = 1.5


def grid_atol(cfg, reference) -> float:
    """Per-cell bound on a grid field computed on the two paths.

    The two paths run one kernel on the same data, so they would agree
    bit for bit but for one thing: XLA:CPU contracts ``a * b + c`` into a
    fused multiply-add inside a fusion (measured on jaxlib 0.11.0: every
    one of 1e5 float32 ``a*b + c`` matched the correctly rounded FMA, and
    23% differed from the product rounded first), and the sharded path's
    ``shard_map`` is a fusion boundary the unsharded program does not
    have.  An FMA skips one rounding, which moves a result by at most one
    ulp of the larger of the product and the result; every product and
    sum in the harness kernels is bounded by the field's magnitude ``M``
    (the stencil weights are divided down and multiplied by
    ``dt * rate * coef`` < 0.05 before they meet the field).  So a step
    moves a cell by at most ``FMA_SITES * eps * max(1, M)``, and a
    difference already present is amplified by at most ``STEP_GROWTH``
    per later step.  A wrapper defect -- a wrong halo cell, offset,
    static, input or parameter -- moves a cell by a fraction of ``M``,
    five or more orders of magnitude above this.
    """
    m = max(1.0, float(np.max(np.abs(np.asarray(reference, np.float64))))) \
        if np.size(reference) else 1.0
    growth = sum(STEP_GROWTH ** k for k in range(int(cfg.steps)))
    return FMA_SITES * eps_of(cfg.dtype) * m * growth


def integral_bound(cfg, unsharded_state: dict, grid_key: str) -> float:
    """Bound for a domain integral computed on the two paths.

    Two parts.  Reordering: any order of summing ``n`` terms is within
    ``(n - 1) u sum|t|`` of the exact sum, to first order (``u = eps/2``),
    so two orders are within ``(n - 1) eps sum|t|``; the terms are
    ``(f + 1)`` times the integral's weight (largest component 2).  And
    the grid field it sums may already differ by :func:`grid_atol` per
    cell, ``n`` cells of it.
    """
    f = np.asarray(unsharded_state[grid_key], dtype=np.float64)
    n = max(int(f.size), 2)
    weight = 2.0 if cfg.integral == "vector" else 1.0
    reorder = (n - 1) * eps_of(cfg.dtype) * weight * float(np.sum(np.abs(f + 1.0)))
    return reorder + n * weight * grid_atol(cfg, f)


def assert_paths_agree(case: Case, sharded: dict, unsharded: dict, context: str) -> None:
    """Grid fields within :func:`grid_atol`; a domain integral within :func:`integral_bound`.

    Shapes and dtypes must match exactly.  The sharded path sums a
    domain integral per shard and then over the shards (``psum``, or
    the stacked per-shard values, summed here); the unsharded one sums
    once.
    """
    cfg = case.cfg
    gkey = grid_key(case)
    integral = getattr(cfg, "integral_name", None) if cfg.integral else None
    assert set(sharded) == set(unsharded), f"{context}: fields {set(sharded)} vs {set(unsharded)}"
    for k, u in unsharded.items():
        s = np.asarray(sharded[k])
        u = np.asarray(u)
        if k == integral:
            if cfg.integral == "per_shard":
                lead = s.ndim - u.ndim
                assert lead >= 0, f"{context}: per-shard {k} shape {s.shape} vs {u.shape}"
                s = s.astype(np.float64).sum(axis=tuple(range(lead)))
            assert s.shape == u.shape, f"{context}: {k} shape {s.shape} vs {u.shape}"
            bound = integral_bound(cfg, unsharded, gkey)
            diff = float(np.max(np.abs(s.astype(np.float64) - u.astype(np.float64)))) \
                if u.size else 0.0
            OBSERVED.append(("integral", cfg.dtype, diff / bound))
            assert diff <= bound, (
                f"{context}: integral {k} differs by {diff:.3e} > bound "
                f"{bound:.3e} (sharded {s}, unsharded {u})")
        else:
            assert s.shape == u.shape, f"{context}: {k} shape {s.shape} vs {u.shape}"
            assert s.dtype == u.dtype, f"{context}: {k} dtype {s.dtype} vs {u.dtype}"
            atol = grid_atol(cfg, u)
            diff = np.abs(s.astype(np.float64) - u.astype(np.float64))
            OBSERVED.append(("grid", cfg.dtype, float(diff.max()) / atol if diff.size else 0.0))
            if not np.all(diff <= atol):
                bad = np.argwhere(diff > atol)
                raise AssertionError(
                    f"{context}: grid field {k} differs at {len(bad)} of {u.size} cells "
                    f"(first {bad[:4].tolist()}; max |diff| {float(diff.max()):.3e} > "
                    f"bound {atol:.3e})")


def _reverse_terms(cfg) -> int:
    """Terms a cell's cotangent sums per step (stencil, statics, inputs, correction)."""
    if cfg.family == "stencil":
        return 2 * sum(cfg.halo) + 9
    if cfg.family in ("pointwise", "example"):
        return 9
    if cfg.family == "heat":
        return 2 * (1 if cfg.order == 2 else 2) + 9
    if cfg.family == "lbm":
        q = 9 if cfg.lattice == "D2Q9" else 19
        return 2 * q + 9      # the Q streamed-in neighbours and the Q moment terms
    width = int(_UnstructuredBase._neighbour_table(cfg).shape[1])
    return 2 * width + 9


def gradient_bound(cfg, scales: dict, g_rev: float) -> tuple[float, float]:
    """``(loss bound, gradient bound)``, absolute, for the two paths.

    The loss is ``L = sum dev**2`` and its gradient ``g = sum 2 dev ds``
    (:func:`gradient_scales` names the sums).  The gradient bound is the
    larger of two things.

    **A derived base**, from the rule that any two orders of a sum of
    ``m`` terms differ by at most ``(m - 1) eps`` times its absolute sum:
    the reverse pass reorders sums (the transpose of a halo exchange adds
    a neighbour's cotangent into a cell after the cell's own stencil
    terms; the loss is reduced per shard first), ``(n + steps * terms) *
    eps * A``; the forward state may differ by :func:`grid_atol` per
    cell, ``2 * grid_atol * B``; and the tangent by ``FMA_SITES * eps *
    growth * T`` per cell, times ``2 S``.

    **The gradient's own sensitivity to reordering, measured**: four times
    the distance between the unsharded path's reverse-mode gradient
    ``g_rev`` and its forward-mode one ``g_fwd``, in the same precision.
    The two compute the same derivative with every operation in another
    order; sharded and unsharded reverse modes reorder only the terms at
    the shard boundaries.  The base above assumes the cancellation is in
    the final reduction; it is not always.  On an unstructured ring
    relaxed to nearly equal values, ``d x / d rate`` is built from
    ``mean - owned``, a small difference of nearly equal numbers, and the
    float32 gradient is 1e-4 off the float64 one on *either* path, with
    the sharded path the closer of the two; sharded and unsharded then
    differed by 0.5x the forward/reverse distance in float32 and 0.8x in
    float64 (both paths agree to 5e-13 in float64).  The factor 4 allows
    for comparing one sample of each.
    """
    eps = eps_of(cfg.dtype)
    growth = sum(STEP_GROWTH ** k for k in range(int(cfg.steps)))
    atol = FMA_SITES * eps * max(1.0, scales["M"]) * growth
    t_atol = FMA_SITES * eps * scales["T"] * growth
    n = scales["n"]
    base = ((n + cfg.steps * _reverse_terms(cfg)) * eps * scales["A"]
            + 2.0 * atol * scales["B"] + 2.0 * t_atol * scales["S"])
    measured = 4.0 * abs(g_rev - scales["g_fwd"])
    l_bound = n * eps * scales["Q"] + 2.0 * atol * scales["S"]
    return l_bound, max(base, measured)


def assert_gradients_agree(case: Case, sharded, unsharded, context: str,
                           scales: Optional[dict] = None) -> None:
    if sharded is None and unsharded is None:
        return
    (ls, gs), (lu, gu) = sharded, unsharded
    scales = gradient_scales(case) if scales is None else scales
    l_bound, g_bound = gradient_bound(case.cfg, scales, gu)
    assert np.isfinite(gs) and np.isfinite(gu), f"{context}: {gs} {gu}"
    OBSERVED.append(("gradient", case.cfg.dtype, abs(gs - gu) / max(g_bound, 1e-300)))
    assert abs(ls - lu) <= l_bound, (
        f"{context}: loss {ls!r} vs {lu!r}, |diff| {abs(ls - lu):.3e} > bound {l_bound:.3e}")
    assert abs(gs - gu) <= g_bound, (
        f"{context}: gradient {gs!r} vs {gu!r}, |diff| {abs(gs - gu):.3e} > "
        f"bound {g_bound:.3e} (scales {scales})")


# ---------------------------------------------------------------------------
# The oracle, end to end
# ---------------------------------------------------------------------------


def _expect_refusal(case: Case, run: Callable[[], Any], stage_now: str, context: str):
    """Run ``run``; if a refusal is predicted at ``stage_now``, require it."""
    want = case.refusal
    if want is None or want[0] != stage_now:
        return run(), False
    _, exc, pattern = want
    import re
    try:
        run()
    except exc as e:  # noqa: PERF203
        assert re.search(pattern, str(e)), f"{context}: refused with {e!r}, not /{pattern}/"
        return None, True
    raise AssertionError(f"{context}: expected a refusal matching /{pattern}/ "
                         f"at {stage_now}, but the sharded path ran")


def check_construction_refusals(case: Case) -> None:
    """The fills a stencil wrapper must refuse at construction, every time.

    A node that declares ``halo_boundary()`` refuses any other fill; an
    outer wrapper refuses a fill different from the inner wrapper's.
    """
    cfg = case.cfg
    if cfg.family != "stencil":
        return
    import pytest
    mesh = create_device_mesh(shape=tuple(cfg.mesh_shape), axis_names=tuple(cfg.axis_names))
    other = next(f for f in FILLS if f != cfg.fill)
    if cfg.declares:
        with pytest.raises(ValueError, match="declares halo_boundary"):
            ShardedStencilNode(make_stencil_node(cfg), mesh, dict(cfg.axis_map),
                               boundary=other)
    if cfg.wrapping == "nested" and case.refusal is None:
        inner = ShardedStencilNode(make_stencil_node(cfg), mesh, dict(cfg.axis_map),
                                   boundary=cfg.fill)
        with pytest.raises(ValueError, match="is itself a ShardedStencilNode"):
            ShardedStencilNode(inner, mesh, dict(cfg.axis_map), boundary=other)


def check_config(cfg, *, surfaces: Optional[tuple] = None) -> dict:
    """The differential oracle for one configuration; returns what ran."""
    with precision(cfg.dtype):
        case = build_case(cfg)
        context = f"{cfg}"
        # Construction (and its predicted refusals).  Without a predicted
        # construction refusal the first surface constructs it anyway.
        if case.refusal is not None and case.refusal[0] == "construct":
            _expect_refusal(case, lambda: case.make(True), "construct", context)
            return {"refused": "construct"}
        check_construction_refusals(case)
        surfaces = (cfg.surface,) if surfaces is None else surfaces
        if case.refusal is not None and case.refusal[0] == "run":
            _, refused = _expect_refusal(
                case, lambda: run_surface(case, "run_scan", True), "run", context)
            # ...and a mis-shaped input is refused by the unsharded node too;
            # any other refusal is the sharded path's own, and the
            # unsharded node runs.
            try:
                run_surface(case, "run_scan", False)
            except Exception:  # noqa: BLE001
                assert case.unsharded_refuses, f"{context}: the unsharded node refused too"
            else:
                assert not case.unsharded_refuses, (
                    f"{context}: the unsharded node took the mis-shaped input")
            return {"refused": "run", "known": case.known}
        done = {}
        for surface in surfaces:
            sharded = run_surface(case, surface, True)
            unsharded = run_surface(case, surface, False)
            if surface == "gradient":
                assert_gradients_agree(case, sharded, unsharded, f"{surface}: {context}")
            elif "__refused__" in unsharded:
                assert sharded == unsharded, f"{surface}: {context}: {sharded} vs {unsharded}"
            else:
                assert_paths_agree(case, sharded, unsharded, f"{surface}: {context}")
            done[surface] = (sharded, unsharded)
        return done
