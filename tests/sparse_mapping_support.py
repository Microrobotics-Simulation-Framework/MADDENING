"""Shared support for the sparse-mapping tests.  Nothing here is a test.

* :func:`densify` scatters a sparse mapping's rows into a dense zero
  matrix -- the operator it is, as a ``StaticLinearMapping`` would hold it.
* :func:`assert_within_rows` and :func:`assert_within_dense` are the
  tolerance a float evaluation of a row sum is held to: per output entry
  within ``T * eps * sum |w| |f|`` of the exact sum, with ``T`` the terms
  of the row plus two and ``eps`` the result dtype's.  The sparse result
  and the dense one are two such evaluations of one real sum.
* :data:`CASES` builds each of the three sparse kinds -- the nearest
  neighbour in its two modes and both conservative forms -- between two
  small interfaces, from references a config can carry, together with the
  dense kind that holds the same matrix.  The parametrised harnesses
  (config and USD round trips, restart, the write doors, the FMU archive,
  system identification, every numeric domain) run over it.
* :data:`SPARSE_POINT_KINDS` describes ``sparse_nearest_neighbor`` the way
  ``tests/registered_mapping_kinds.py`` describes a registered kind, so
  the harnesses written for a kind that maps one point set onto another
  run it unchanged.
* :class:`Interface` and :func:`mapped_pair` are the graph those cases are
  put on: two 1-D interfaces that publish their cell centres and cell
  boundaries, joined by a mapped edge each way, every array referenced the
  way a config can carry it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
    sparse_projection_1d_mapping,
)
from tests.registered_mapping_kinds import RegisteredKind

SPARSE_KIND_NAMES = ("sparse_matrix", "sparse_nearest_neighbor", "sparse_projection_1d")


@contextlib.contextmanager
def x64(on: bool):
    """``jax_enable_x64`` set to *on* for the block, restored after it."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def valid_slots(mapping: StaticSparseMapping) -> np.ndarray:
    """The boolean ``(n_rows, k)`` mask of the slots that hold an entry."""
    rows, k = mapping.indices.shape
    if mapping.counts is None:
        return np.ones((rows, k), dtype=bool)
    return np.arange(k)[None, :] < np.asarray(mapping.counts)[:, None]


def slot_entries(mapping: StaticSparseMapping) -> tuple[np.ndarray, np.ndarray]:
    """``(target, source)`` of every slot, each ``(n_rows, k)`` (meaningless
    where :func:`valid_slots` is false)."""
    rows = np.broadcast_to(np.arange(mapping.indices.shape[0])[:, None],
                           mapping.indices.shape)
    index = np.asarray(mapping.indices)
    return (rows, index) if mapping.layout == "gather" else (index, rows)


def densify(mapping: StaticSparseMapping, weights: Any = None) -> np.ndarray:
    """The ``(n_target, n_source)`` matrix the rows of *mapping* are, at the
    weights' own dtype; repeated entries add (in slot order)."""
    w = np.asarray(mapping.weights if weights is None else weights)
    valid = valid_slots(mapping)
    target, source = slot_entries(mapping)
    dense = np.zeros((mapping.n_target, mapping.n_source), dtype=w.dtype)
    np.add.at(dense, (target[valid], source[valid]), w[valid])
    return dense


def entries_per_target(mapping: StaticSparseMapping) -> np.ndarray:
    """How many slots add into each target (``k_i`` of the tolerance)."""
    valid = valid_slots(mapping)
    target, _source = slot_entries(mapping)
    return np.bincount(target[valid], minlength=mapping.n_target)


#: The reference's precision: x86-64's 80-bit extended where there is one,
#: so that a float64 row sum is judged against something finer than itself.
LD = np.longdouble
_LD_IS_FINER = float(np.finfo(LD).eps) < 1e-18


def _host64(array: Any) -> np.ndarray:
    """*array* as float64 on the host (a 16-bit or 32-bit float widens
    exactly, bfloat16 included)."""
    host = np.asarray(array)
    return host.astype(np.float64) if host.dtype != np.float64 else host


def row_reference(mapping: StaticSparseMapping, field: Any, *, weights: Any = None,
                  transpose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(exact, scale, entries)`` of ``apply`` (``apply_T`` with
    *transpose*): per output entry the sum over the slots that add into it
    of ``w * f`` and of ``|w| |f|``, in the reference precision, and how
    many slots those are.  Taken slot by slot, never through a matrix: two
    slots of one row that name the same index are two terms."""
    w = _host64(mapping.weights if weights is None else weights).astype(LD)
    f = _host64(field).astype(LD)
    valid = valid_slots(mapping)
    target, source = slot_entries(mapping)
    out, read = (source, target) if transpose else (target, source)
    size = mapping.n_source if transpose else mapping.n_target
    terms = w[valid].reshape((-1,) + (1,) * (f.ndim - 1)) * f[read[valid]]
    exact = np.zeros((size,) + f.shape[1:], dtype=LD)
    scale = np.zeros_like(exact)
    np.add.at(exact, out[valid], terms)
    np.add.at(scale, out[valid], np.abs(terms))
    return exact, scale, np.bincount(out[valid], minlength=size)


def _assert_within(got: Any, exact: np.ndarray, scale: np.ndarray, entries: np.ndarray,
                   what: str, extra: float) -> None:
    got = np.asarray(got)
    eps = float(jax.numpy.finfo(got.dtype).eps)
    if not _LD_IS_FINER:
        extra = extra + float(np.max(entries, initial=0)) + 2.0    # the reference rounds too
    terms = np.asarray(entries, dtype=np.float64).reshape((-1,) + (1,) * (exact.ndim - 1))
    bound = (terms + extra) * eps * np.asarray(scale, dtype=np.float64)
    error = np.asarray(np.abs(_host64(got).astype(LD) - exact), dtype=np.float64)
    assert got.shape == exact.shape, f"{what}: shape {got.shape}, expected {exact.shape}"
    worst = np.unravel_index(int(np.argmax(error - bound)), error.shape)
    assert np.all(error <= bound), (
        f"{what}: entry {tuple(int(i) for i in worst)} is {float(_host64(got)[worst])!r}, "
        f"the exact sum is {float(exact[worst])!r}; off by {error[worst]:.3e}, allowed "
        f"{bound[worst]:.3e}")


def assert_within_rows(got: Any, mapping: StaticSparseMapping, field: Any, *,
                       weights: Any = None, transpose: bool = False, what: str = "",
                       extra: float = 2.0) -> None:
    """*got* is ``mapping.apply(field)`` (``apply_T`` with *transpose*) to
    the rounding of one row sum: per entry within ``(k_i + extra) * eps *
    sum |w| |f|`` of the exact sum over the row's ``k_i`` slots, ``eps``
    the result dtype's."""
    exact, scale, entries = row_reference(mapping, field, weights=weights,
                                          transpose=transpose)
    _assert_within(got, exact, scale, entries, what, extra)


def assert_within_dense(got: Any, dense_matrix: Any, field: Any, entries: Any, *,
                        what: str = "", extra: float = 2.0) -> None:
    """*got* is within the rounding of one row sum of ``dense_matrix @
    field``: per entry within ``(entries_i + extra) * eps * sum_j |H_ij|
    |f_j|`` of the exact product, ``eps`` the result dtype's.

    For a result computed from the dense matrix itself *entries* is its
    column count; for a sparse mapping holding the same matrix (no index
    repeated in a row, so every slot is one matrix entry) it is the row's
    slots.  Both are float evaluations of one real sum.
    """
    H = _host64(dense_matrix).astype(LD)
    f = _host64(field).astype(LD)
    entries = np.broadcast_to(np.asarray(entries), (H.shape[0],))
    _assert_within(got, H @ f, np.abs(H) @ np.abs(f), entries, what, extra)


# ---------------------------------------------------------------------------
# The kind that maps one point set onto another, as a registered kind
# ---------------------------------------------------------------------------

SPARSE_NEAREST_NEIGHBOR = "sparse_nearest_neighbor"

#: ``sparse_nearest_neighbor`` for the harnesses parametrised over
#: ``tests.registered_mapping_kinds.KINDS``: same shape (two point sets,
#: ``source_ref=`` / ``target_ref=``), one weight ``W``.  ``transpose`` is
#: left out of the drawn hyper-parameters: ``"scatter"`` is refused with
#: ``mode="consistent"``, and it has its own cases in :data:`CASES`.
SPARSE_POINT_KINDS: dict[str, RegisteredKind] = {
    SPARSE_NEAREST_NEIGHBOR: RegisteredKind(
        SPARSE_NEAREST_NEIGHBOR, sparse_nearest_neighbor_mapping, ("W",),
        "source_ref", "target_ref", {"mode": ("consistent", "conservative")}),
}


# ---------------------------------------------------------------------------
# Every sparse kind between two small interfaces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SparseCase:
    """One sparse mapping between an interface of ``n_source`` values and
    one of ``n_target``, and the dense mapping holding the same matrix.

    ``build(points, references)`` and ``dense(points)`` take
    ``{"source_points", "target_points", "source_boundaries",
    "target_boundaries"}`` -- the cell centres and cell boundaries of the
    two interfaces -- and, for ``build``, a reference for each (``None`` to
    leave one out).  ``assets`` are the arrays a ``sparse_matrix`` case
    needs saved next to a config, as ``{file name: array}``.
    """

    name: str
    kind: str
    mode: str
    layout: str
    arrays: tuple[str, ...]
    make: Callable[..., StaticSparseMapping]
    make_dense: Callable[..., Any]
    assets: Callable[[dict], dict] = lambda points: {}

    def build(self, points: dict, references: Optional[dict] = None) -> StaticSparseMapping:
        return self.make(points, references or {})

    def dense(self, points: dict):
        return self.make_dense(points)

    def __repr__(self) -> str:      # the pytest id
        return self.name


def _nearest(mode: str, transpose: str) -> SparseCase:
    def make(points, references):
        return sparse_nearest_neighbor_mapping(
            points["source_points"], points["target_points"], mode=mode,
            transpose=transpose, source_ref=references.get("source_points"),
            target_ref=references.get("target_points"))

    def dense(points):
        return nearest_neighbor_mapping(points["source_points"], points["target_points"],
                                        mode=mode)

    suffix = "" if mode == "consistent" else f"-{transpose}"
    return SparseCase(f"nearest-{mode}{suffix}", "sparse_nearest_neighbor", mode,
                      "scatter" if transpose == "scatter" else "gather",
                      ("source_points", "target_points"), make, dense)


def _projection() -> SparseCase:
    def make(points, references):
        return sparse_projection_1d_mapping(
            points["source_boundaries"], points["target_boundaries"],
            source_ref=references.get("source_boundaries"),
            target_ref=references.get("target_boundaries"))

    def dense(points):
        return projection_1d_mapping(points["source_boundaries"], points["target_boundaries"])

    return SparseCase("projection", "sparse_projection_1d", "conservative", "gather",
                      ("source_boundaries", "target_boundaries"), make, dense)


def stencil_rows(points: dict) -> tuple[np.ndarray, np.ndarray]:
    """A linear-interpolation stencil from the source cell centres to the
    target ones, as ``sparse_matrix`` rows: two entries per target, one
    where a target lies outside the source centres (so a slot is unused and
    marked ``-1``), written unused slot *first* in every other such row.
    The values are at the default float dtype (float64 under
    ``jax_enable_x64``), which the weights keep."""
    xs = np.asarray(points["source_points"], dtype=np.float64).ravel()
    xt = np.asarray(points["target_points"], dtype=np.float64).ravel()
    indices = np.full((xt.size, 2), -1, dtype=np.int64)
    values = np.zeros((xt.size, 2), dtype=np.dtype(jnp.result_type(float)))
    outside = 0
    for i, x in enumerate(xt):
        j = int(np.searchsorted(xs, x))
        if j == 0 or j == xs.size:
            slot = outside % 2          # the used slot is not always the first
            outside += 1
            indices[i, slot] = min(j, xs.size - 1)
            values[i, slot] = 1.0
            continue
        t = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
        indices[i] = (j, j - 1)         # descending: the order given is kept
        values[i] = (t, 1.0 - t)
    return indices, values


def _matrix() -> SparseCase:
    def make(points, references):
        indices, values = stencil_rows(points)
        return sparse_matrix_mapping(
            indices, values, n_source=int(np.size(points["source_points"])),
            name="stencil", indices_asset=references.get("indices"),
            values_asset=references.get("values"))

    def dense(points):
        indices, values = stencil_rows(points)
        H = np.zeros((indices.shape[0], int(np.size(points["source_points"]))), values.dtype)
        used = indices >= 0
        rows = np.broadcast_to(np.arange(indices.shape[0])[:, None], indices.shape)
        np.add.at(H, (rows[used], indices[used]), values[used])
        return matrix_mapping(H)

    def assets(points):
        indices, values = stencil_rows(points)
        return {"indices": indices, "values": values}

    return SparseCase("matrix", "sparse_matrix", "consistent", "gather",
                      ("indices", "values"), make, dense, assets)


CASES: dict[str, SparseCase] = {case.name: case for case in (
    _nearest("consistent", "gather"),
    _nearest("conservative", "gather"),
    _nearest("conservative", "scatter"),
    _projection(),
    _matrix(),
)}


def interface_points(n_source: int, n_target: int, dtype="float64") -> dict:
    """Two 1-D interfaces on ``[0, 1]``: uniform cells, the target's centres
    reaching past the source's at both ends when it is the finer one."""
    def cells(n):
        boundaries = np.linspace(0.0, 1.0, n + 1)
        return 0.5 * (boundaries[:-1] + boundaries[1:]), boundaries
    xs, bs = cells(n_source)
    xt, bt = cells(n_target)
    return {"source_points": xs.astype(dtype), "target_points": xt.astype(dtype),
            "source_boundaries": bs.astype(dtype), "target_boundaries": bt.astype(dtype)}


# ---------------------------------------------------------------------------
# A graph to put them on
# ---------------------------------------------------------------------------

class Interface(SimulationNode):
    """A 1-D interface of ``n`` cells on ``[0, 1]``: a vector that relaxes
    towards its mapped input, in a chosen dtype, publishing its cell
    centres (``points``) and cell boundaries (``boundaries``) as static
    data for a mapping to reference."""

    def __init__(self, name, timestep, n=4, dtype="float32", rate=0.5):
        super().__init__(name, timestep, n=n, dtype=dtype, rate=rate)
        boundaries = np.linspace(0.0, 1.0, n + 1)
        self._boundaries = boundaries
        self._points = 0.5 * (boundaries[:-1] + boundaries[1:])

    @property
    def static_data(self):
        return {"points": StaticArray(self._points),
                "boundaries": StaticArray(self._boundaries)}

    def initial_state(self):
        n, dtype = self.params["n"], self.params["dtype"]
        return {"x": jnp.linspace(1.0, 2.0, n).astype(dtype)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],),
                                         dtype=jnp.dtype(self.params["dtype"]),
                                         description="the other interface, mapped")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        rate = (self.params if params is None else params)["rate"]
        x = state["x"]
        # A mapped input arrives at the weights' dtype; the interface keeps its own.
        target = boundary_inputs.get("inp", jnp.zeros_like(x)).astype(x.dtype)
        return {"x": (x + dt * rate * (target - x)).astype(x.dtype)}


REGISTRY = {"Interface": Interface}

A2B = "a.x->b.inp"
B2A = "b.x->a.inp"


def node_points(gm: GraphManager, source: str, target: str) -> dict:
    """The four arrays a case is built from, read from two interfaces."""
    def static(name, field):
        return np.asarray(gm.get_node(name).static_data[field].value)
    return {"source_points": static(source, "points"),
            "target_points": static(target, "points"),
            "source_boundaries": static(source, "boundaries"),
            "target_boundaries": static(target, "boundaries")}


def edge_mapping(case: SparseCase, gm: GraphManager, source: str, target: str, *,
                 base_dir: Optional[Path] = None) -> StaticSparseMapping:
    """*case* from interface *source* onto *target*, every array referenced
    the way a config can carry it: a point set or a boundary array by node
    field, and the two arrays of a ``sparse_matrix`` as members of one
    ``.npz`` saved in *base_dir* (left unreferenced, so unserialisable,
    without one)."""
    points = node_points(gm, source, target)
    if case.kind == "sparse_matrix":
        if base_dir is None:
            return case.build(points)
        name = f"{source}_to_{target}.npz"
        np.savez(Path(base_dir) / name, **case.assets(points))
        return case.build(points, {array: {"asset": name, "key": array}
                                   for array in case.arrays})
    side = {"source": source, "target": target}
    return case.build(points, {
        array: {"node": side[array.split("_")[0]], "field": array.split("_")[1]}
        for array in case.arrays})


def mapped_pair(case_ab: SparseCase, case_ba: Optional[SparseCase] = None, *,
                n_a: int = 4, n_b: int = 6, dtype_a: str = "float32",
                dtype_b: str = "float32", dt_a: float = 0.125, dt_b: float = 0.125,
                group: Optional[dict] = None, base_dir: Optional[Path] = None,
                compile: bool = True) -> GraphManager:
    """Interfaces ``a`` and ``b``, ``a`` mapped onto ``b`` by *case_ab* and
    ``b`` back onto ``a`` by *case_ba* (*case_ab* again when omitted),
    optionally in a coupling group.  Timesteps of 0.125 are exactly
    representable in every float dtype."""
    gm = GraphManager()
    gm.add_node(Interface("a", dt_a, n=n_a, dtype=dtype_a))
    gm.add_node(Interface("b", dt_b, n=n_b, dtype=dtype_b, rate=0.25))
    gm.add_edge("a", "b", "x", "inp", mapping=edge_mapping(case_ab, gm, "a", "b",
                                                           base_dir=base_dir))
    gm.add_edge("b", "a", "x", "inp", mapping=edge_mapping(case_ba or case_ab, gm, "b", "a",
                                                           base_dir=base_dir))
    if group is not None:
        gm.add_coupling_group(["a", "b"], **group)
    if compile:
        gm.compile()
    return gm

