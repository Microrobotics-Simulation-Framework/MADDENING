"""Shared toys, builders and strategies for the sharded-surface property tests.

The properties themselves live in the ``test_property_*.py`` modules
beside this one.  They are in ``tests/cloud/multigpu/`` rather than in
``tests/property/`` because the device count is part of every one of
them: only this directory's ``conftest.py`` sets
``XLA_FLAGS=--xla_force_host_platform_device_count``, and it is the only
file under it allowed to (``test_conftest_device_policy.py`` enforces
that).  A second copy of that policy under ``tests/property/`` would be
a second thing to keep right.

Three things live here.

**A wrapper family.**  :data:`WRAPPER_FAMILY` maps a label to a builder
returning a :class:`WrapperCase`: an inner node, the sharded wrapper
around it, the name of a differentiable constant, and a way to run both
and compare.  Every property that says "the wrapper behaves like the
node it wraps" is parametrised over this mapping, so a fourth wrapper is
covered by adding one entry.

**Toy nodes.**  Deliberately tiny: one example of a property compiles a
fresh ``shard_map``, so the physics is the cheapest thing that still
exercises a halo exchange, a sharded ``StaticArray``, a grid-shaped
boundary input and an injected parameter.

**Partition strategies.**  :func:`partition_layouts` generates
``UnstructuredPartitionLayout`` objects, including the awkward shapes a
hand-written test tends to skip: an empty shard, a single device, one
cell per device, self edges and duplicate edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
)
from maddening.cloud.multigpu.sharded_node import (
    ShardedPointwiseNode,
    ShardedStencilNode,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray

# ---------------------------------------------------------------------------
# Device counts
# ---------------------------------------------------------------------------
# Read once, at import: the multigpu conftest has already set XLA_FLAGS by
# the time any test module is imported, and ``jax.devices()`` cannot change
# afterwards.  1 is always in the list so the strategies below are never
# empty -- the directory's own policy test imports every module here with
# ``MADDENING_VIRTUAL_DEVICES=0``, i.e. on a single device.
_AVAILABLE_DEVICES = len(jax.devices())

#: Device counts a property may ask for.  Capped at 4 on purpose: an
#: example compiles a fresh ``shard_map`` per device count and the shared
#: CPU box runs several agents at once, so a wider mesh buys coverage of
#: the same code path at several times the wall clock.
DEVICE_COUNTS = tuple(n for n in (1, 2, 4) if n <= _AVAILABLE_DEVICES)

def device_counts() -> st.SearchStrategy[int]:
    """A device count the local mesh can actually supply."""
    return st.sampled_from(DEVICE_COUNTS)


def param_values() -> st.SearchStrategy[float]:
    """A positive, finite constant to write into a node's parameter.

    Inside every family member's declared bounds and well away from the
    stability limit of the toy integrators: the properties here are about
    a written value reaching the physics, not about what the physics does
    with an absurd one.  Out-of-bounds writes have a property of their
    own.
    """
    # Both bounds are exactly representable in float32, which
    # ``width=32`` requires.
    return st.floats(min_value=0.0625, max_value=1.0, allow_nan=False,
                     allow_infinity=False, width=32)


# ---------------------------------------------------------------------------
# Toy nodes
# ---------------------------------------------------------------------------


class PointwiseRelaxNode(SimulationNode):
    """Pointwise relaxation ``x <- x + rate * dt * (source - x)``.

    Pointwise (empty ``halo_width``), so it is the
    :class:`ShardedPointwiseNode` member of the family.  ``rate`` is a
    differentiable constant with a spec of its own, which is what makes
    it useful for the parameter-contract properties: a wrapper that
    falls back to the base class's default spec loses the bounds.
    """

    def __init__(self, name: str = "relax", n_cells: int = 8,
                 rate: float = 0.5, timestep: float = 0.1) -> None:
        super().__init__(name=name, timestep=timestep, rate=rate,
                         n_cells=int(n_cells))

    def halo_width(self) -> dict[int, int]:
        return {}

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        n = int(self.params["n_cells"])
        return {"x": jnp.arange(n, dtype=jnp.float32) / n}

    def boundary_input_spec(self) -> dict:
        return {"source": BoundaryInputSpec(
            shape=(int(self.params["n_cells"]),),
            description="per-cell target value", expected_units="1")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, None), transform="log",
                                  units="1/s")}

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        source = boundary_inputs.get("source", jnp.zeros_like(state["x"]))
        return {"x": state["x"] + p["rate"] * dt * (source - state["x"])}


class StencilDiffusion1D(SimulationNode):
    """1-D diffusion with a sharded mask and a per-cell boundary input.

    ``f <- f + rate * mask * laplacian(f) * dt + dt * source``, periodic.
    Everything the stencil wrapper has to get right is in one node: a
    halo-1 stencil, a ``StaticArray`` declared ``replication="shard"``, a
    grid-shaped boundary input, a replicated scalar boundary input and an
    injected ``params``.
    """

    def __init__(self, name: str = "diff", n_cells: int = 16,
                 rate: float = 0.5, mask: Optional[np.ndarray] = None,
                 timestep: float = 0.05) -> None:
        super().__init__(name=name, timestep=timestep, rate=rate,
                         n_cells=int(n_cells))
        n = int(n_cells)
        if mask is None:
            mask = 0.5 + 0.5 * (np.arange(n, dtype=np.float32) % 3) / 3.0
        self._mask = jnp.asarray(np.asarray(mask, dtype=np.float32))
        if self._mask.shape != (n,):
            raise ValueError(f"mask shape {self._mask.shape} != ({n},)")
        # Built once, as SimulationNode.static_data requires ("stable
        # across calls for a given node instance"): the sharded wrapper
        # snapshots it at construction, and a node that rebuilt it per
        # call would have the two paths reading different arrays.
        self._static = {"mask": StaticArray(value=self._mask,
                                            replication="shard", shard_axis=0)}

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def state_fields(self) -> list[str]:
        return ["f"]

    @property
    def static_data(self) -> dict:
        return self._static

    def initial_state(self) -> dict:
        n = int(self.params["n_cells"])
        x = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        return {"f": jnp.asarray(np.sin(2 * np.pi * x, dtype=np.float32))}

    def boundary_input_spec(self) -> dict:
        return {
            "source": BoundaryInputSpec(
                shape=(int(self.params["n_cells"]),),
                description="per-cell forcing", expected_units="1/s"),
            "gain": BoundaryInputSpec(
                shape=(), description="scalar multiplier on the forcing",
                expected_units="1"),
        }

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, None), transform="log",
                                  units="m^2/s")}

    @staticmethod
    def _gain(boundary_inputs) -> jnp.ndarray:
        return jnp.asarray(boundary_inputs.get("gain", 1.0), jnp.float32)

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        """Unsharded reference: periodic Laplacian on the whole field."""
        p = self.params if params is None else {**self.params, **params}
        f = state["f"]
        f_pad = jnp.concatenate([f[-1:], f, f[:1]])
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        source = boundary_inputs.get("source", jnp.zeros_like(f))
        f_new = (f + p["rate"] * self._mask * lap * dt
                 + dt * self._gain(boundary_inputs) * source)
        return {"f": f_new}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        f_pad = state_padded["f"]
        mask = static_padded["mask"][1:-1]
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        source = boundary_inputs.get("source")
        source = (jnp.zeros_like(f_pad[1:-1]) if source is None
                  else source[1:-1])
        f_new = (f_pad[1:-1] + p["rate"] * mask * lap * dt
                 + dt * self._gain(boundary_inputs) * source)
        return {"f": jnp.concatenate([f_pad[:1], f_new, f_pad[-1:]])}


class UnstructuredRelaxNode(SimulationNode):
    """Relaxation towards the mean of a cell's graph neighbours.

    The :class:`ShardedUnstructuredNode` member of the family.  The
    padded path reads the ghost cells the exchange delivered, so the
    unsharded reference has to gather over the same global connectivity.
    """

    def __init__(self, *, name: str = "cells", n_global_cells: int = 16,
                 edges: np.ndarray, rate: float = 0.5,
                 layout=None, timestep: float = 1.0) -> None:
        super().__init__(name=name, timestep=timestep, rate=rate)
        self._n_global = int(n_global_cells)
        self._edges = np.asarray(edges, dtype=np.int32)
        self._table = self._neighbour_table()
        self._layout = layout
        # Built once: see the note in StencilDiffusion1D.
        self._static = ({} if layout is None
                        else {"local_neighbours": StaticArray(
                            value=self._slab_index_table(layout),
                            replication="partition",
                            partition_assignment=layout.partition_assignment)})

    @property
    def static_data(self) -> dict:
        return self._static

    def _slab_index_table(self, layout) -> np.ndarray:
        """The neighbour table translated into per-shard slab indices.

        Row ``g`` is delivered to the device that owns cell ``g``, so the
        translation is well defined globally: the slab index of a
        neighbour ``h`` of ``g`` is its local index on ``pa[g]`` when
        ``pa[h] == pa[g]``, and ``n_local_max`` plus its position in that
        device's ghost list otherwise.
        """
        table = np.full_like(self._table, -1)
        ghost_slot = [
            {int(g): j for j, g in enumerate(layout.ghost_global_ids[d])}
            for d in range(layout.n_devices)
        ]
        for g in range(self._n_global):
            owner = int(layout.partition_assignment[g])
            for j, h in enumerate(self._table[g]):
                h = int(h)
                if h < 0:
                    continue
                if int(layout.partition_assignment[h]) == owner:
                    table[g, j] = layout.local_index_of(owner, h)
                else:
                    table[g, j] = layout.n_local_max + ghost_slot[owner][h]
        return table

    def _neighbour_table(self) -> np.ndarray:
        adjacency: list[list[int]] = [[] for _ in range(self._n_global)]
        for u, v in self._edges:
            if int(u) != int(v):
                adjacency[int(u)].append(int(v))
                adjacency[int(v)].append(int(u))
        width = max((len(a) for a in adjacency), default=0)
        out = np.full((self._n_global, max(width, 1)), -1, dtype=np.int32)
        for i, neigh in enumerate(adjacency):
            out[i, : len(neigh)] = neigh
        return out

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        return {"x": jnp.asarray(
            np.arange(self._n_global, dtype=np.float32) + 1.0)}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "rate": ParamSpec(bounds=(0.0, 1.0), units="1")}

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        x = state["x"]
        table = jnp.asarray(self._table)
        valid = table >= 0
        gathered = jnp.take(x, jnp.where(valid, table, 0), axis=0)
        total = jnp.where(valid, gathered, 0.0).sum(axis=1)
        denom = jnp.maximum(valid.sum(axis=1), 1).astype(x.dtype)
        return {"x": x + p["rate"] * (total / denom - x)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None, params=None) -> dict:
        """Per-shard: relax towards the mean over the padded slab.

        The slab is ``local + ghost``; ``local_neighbours`` (a sharded
        static) holds, per owned cell, the slab indices of its
        neighbours, so the result is identical to the unsharded gather.
        """
        p = self.params if params is None else {**self.params, **params}
        x = state_padded["x"]
        # The wrapper ghost-exchanges partitioned statics too, so the
        # table arrives with the same ghost tail as the state slab; only
        # the owned rows have meaning.
        n_local = shard_info[0][1] if shard_info else x.shape[0]
        table = static_padded["local_neighbours"][:n_local]
        valid = table >= 0
        gathered = jnp.take(x, jnp.where(valid, table, 0), axis=0)
        total = jnp.where(valid, gathered, 0.0).sum(axis=1)
        denom = jnp.maximum(valid.sum(axis=1), 1).astype(x.dtype)
        owned = x[:n_local]
        return {"x": owned + p["rate"] * (total / denom - owned)}


class LegacyNoParamsNode(SimulationNode):
    """A node on the 3-argument contract: no ``params`` keyword anywhere."""

    def __init__(self, name: str = "legacy", n_cells: int = 8,
                 rate: float = 0.5, timestep: float = 0.1) -> None:
        super().__init__(name=name, timestep=timestep, rate=rate,
                         n_cells=int(n_cells))

    def halo_width(self) -> dict[int, int]:
        return {}

    def state_fields(self) -> list[str]:
        return ["x"]

    def initial_state(self) -> dict:
        return {"x": jnp.zeros(int(self.params["n_cells"]), jnp.float32)}

    def update(self, state, boundary_inputs, dt) -> dict:
        return {"x": state["x"] + self.params["rate"] * dt}


# ---------------------------------------------------------------------------
# The wrapper family
# ---------------------------------------------------------------------------


@dataclass
class WrapperCase:
    """One member of the sharded-wrapper family, ready to run.

    ``inner`` is the node as it would run on a single device; ``wrapped``
    is the same node behind its sharded wrapper on an ``n_devices`` mesh.
    Both share one ``params`` dict, as the wrappers promise.
    """

    label: str
    inner: SimulationNode
    wrapped: SimulationNode
    param_name: str
    boundary_inputs: dict
    n_devices: int
    #: For a wrapper whose sharded state is a padded slab rather than the
    #: global field (the unstructured one), a 1/0 mask over the leading
    #: axis marking the slots that hold owned cells.  ``None`` when the
    #: sharded state has the same layout as the unsharded state.
    slab_mask: Optional[jnp.ndarray] = None

    def objective(self, state: dict, *, sharded: bool) -> jnp.ndarray:
        """Sum of squares over the cells that exist, for gradient tests.

        A padded slab carries slots that no global cell maps to; they
        pick up values from the exchange and would make the sharded
        objective a different function from the unsharded one.
        """
        total = jnp.float32(0.0)
        for value in state.values():
            if sharded and self.slab_mask is not None:
                shape = (-1,) + (1,) * (jnp.ndim(value) - 1)
                value = value * self.slab_mask.reshape(shape)
            total = total + jnp.sum(value ** 2)
        return total

    def gather(self, state: dict, *, sharded: bool) -> dict:
        """The state as global, host-side arrays, whichever path ran it."""
        if sharded and hasattr(self.wrapped, "gather_global"):
            state = self.wrapped.gather_global(state)
        return {k: np.asarray(jax.device_get(v)) for k, v in state.items()}

    def run(self, *, steps: int, sharded: bool, params=None) -> dict:
        """``steps`` eager updates on one path, gathered to global arrays."""
        node = self.wrapped if sharded else self.inner
        state = node.initial_state()
        for _ in range(steps):
            state = node.update(state, self.boundary_inputs, node.delta_t,
                                params=params)
        return self.gather(state, sharded=sharded)


def _mask_for(n_cells: int) -> np.ndarray:
    return (0.5 + 0.5 * (np.arange(n_cells, dtype=np.float32) % 3) / 3.0)


def _source_for(n_cells: int) -> jnp.ndarray:
    x = np.linspace(0.0, 1.0, n_cells, endpoint=False, dtype=np.float32)
    return jnp.asarray(np.cos(4 * np.pi * x, dtype=np.float32))


def build_pointwise(*, n_devices: int, rate: float = 0.5,
                    n_cells: int = 8) -> WrapperCase:
    inner = PointwiseRelaxNode(name="relax", n_cells=n_cells, rate=rate)
    mesh = create_device_mesh(shape=(n_devices,))
    return WrapperCase(
        label="pointwise", inner=inner, param_name="rate", n_devices=n_devices,
        wrapped=ShardedPointwiseNode(inner, mesh, shard_axes=(0,)),
        boundary_inputs={"source": _source_for(n_cells)},
    )


def build_stencil(*, n_devices: int, rate: float = 0.5, n_cells: int = 16,
                  mask: Optional[np.ndarray] = None) -> WrapperCase:
    inner = StencilDiffusion1D(
        name="diff", n_cells=n_cells, rate=rate,
        mask=_mask_for(n_cells) if mask is None else mask)
    mesh = create_device_mesh(shape=(n_devices,))
    return WrapperCase(
        label="stencil", inner=inner, param_name="rate", n_devices=n_devices,
        wrapped=ShardedStencilNode(inner, mesh, axis_map={"devices": 0},
                                   boundary="periodic"),
        boundary_inputs={"source": _source_for(n_cells),
                         "gain": jnp.float32(0.75)},
    )


def ring_edges(n_cells: int) -> np.ndarray:
    """A connected ring, the cheapest connectivity worth partitioning."""
    return np.array([[i, (i + 1) % n_cells] for i in range(n_cells)],
                    dtype=np.int32)


def build_unstructured(*, n_devices: int, rate: float = 0.5,
                       n_cells: int = 16) -> WrapperCase:
    edges = ring_edges(n_cells)
    pa = (np.arange(n_cells) % n_devices).astype(np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    # One node object plays both roles: its ``update`` is the
    # single-device reference over the global connectivity, its
    # ``update_padded`` is what each shard runs.  Nothing here fixes up
    # the params dict -- whether the wrapper shares it is the wrapper's
    # business, and a property below checks it.
    inner = UnstructuredRelaxNode(name="cells", n_global_cells=n_cells,
                                  edges=edges, rate=rate, layout=layout)
    mesh = create_device_mesh(shape=(n_devices,))
    wrapped = ShardedUnstructuredNode(inner, mesh, layout)
    owned = np.zeros((layout.n_devices, layout.n_local_max), dtype=np.float32)
    for device, ids in enumerate(layout.local_global_ids):
        owned[device, : len(ids)] = 1.0
    return WrapperCase(
        label="unstructured", inner=inner, param_name="rate",
        n_devices=n_devices, wrapped=wrapped, boundary_inputs={},
        slab_mask=jnp.asarray(owned.reshape(-1)),
    )


#: Label -> builder for every sharded wrapper.  A new wrapper is covered
#: by every property in this package the moment it is added here.
WRAPPER_FAMILY: dict[str, Callable[..., WrapperCase]] = {
    "pointwise": build_pointwise,
    "stencil": build_stencil,
    "unstructured": build_unstructured,
}


def wrapper_labels() -> st.SearchStrategy[str]:
    return st.sampled_from(sorted(WRAPPER_FAMILY))


# ---------------------------------------------------------------------------
# Partition strategies for the unstructured exchange
# ---------------------------------------------------------------------------


@st.composite
def partition_layouts(draw, *, max_devices: Optional[int] = None):
    """A valid :class:`UnstructuredPartitionLayout` of an awkward shape.

    Deliberately generates the partitions a hand-written test skips: a
    single device, one cell per device, a shard that owns nothing, self
    edges, duplicate edges and an edge-disjoint partition (no ghosts at
    all).  Only the constraints ``build_unstructured_partition`` really
    imposes are respected -- at least one cell, and every partition value
    in ``[0, n_devices)``.
    """
    n_devices = draw(st.sampled_from(
        [n for n in DEVICE_COUNTS if max_devices is None or n <= max_devices]))
    n_cells = draw(st.integers(min_value=1, max_value=4 * n_devices))
    assignment = draw(st.lists(
        st.integers(min_value=0, max_value=n_devices - 1),
        min_size=n_cells, max_size=n_cells))
    pa = np.asarray(assignment, dtype=np.int32)

    pairs = draw(st.lists(
        st.tuples(st.integers(0, n_cells - 1), st.integers(0, n_cells - 1)),
        min_size=0, max_size=2 * n_cells))
    edges = (np.asarray(pairs, dtype=np.int32) if pairs
             else np.zeros((0, 2), dtype=np.int32))
    return build_unstructured_partition(
        partition_assignment=pa, edges=edges, n_devices=n_devices)
