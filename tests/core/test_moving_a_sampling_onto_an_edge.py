"""A node's own sampling of a cell-centred grid, and the same sampling on an edge.

The worked example of ``docs/user_guide/moving_a_sampling_onto_an_edge.md``,
in two and three dimensions and in both precisions.  A "fluid" node holds
a velocity on a Cartesian, CELL-CENTRED mesh; a "body" node owns a few
points, is carried by the velocity it is given at them, and leaves the
mesh.

**Before.**  The fluid takes the points as a boundary input and returns the
velocity at them, interpolated multilinearly between cell centres and
clamped outside them, inside its own ``update`` (:class:`SamplingFluid`,
:func:`sample_at_cell_centres`).  Two plain edges, and a cycle.

**After.**  The fluid offers its field and nothing else (:class:`Fluid`);
the sampling is the edge's: ``multilinear_grid_mapping(origin=<the first
cell CENTRE>, spacing=<the cell size>, shape=<the number of cells>)`` with
``geometry=("target", "points")``.

What is held:

* the two graphs carry the body alike, to rounding, over a run in which
  it crosses several cells and leaves the mesh;
* points inside, on a cell centre, on an outermost cell centre, between
  an outermost cell centre and the mesh's edge, and outside the mesh are
  each sampled as an interpolator that knows nothing of either
  implementation samples them (scipy's, on the cell-centre coordinates,
  at the point's projection onto their box);
* the lattice whose origin is the mesh's corner is half a cell off: on a
  field of known gradient, by that gradient times half a cell;
* ``jax.grad`` of the body's final position with respect to the initial
  field and to the initial points is the same in the two graphs, and is
  the central difference in float64;
* the value the body receives at a step is the field after the fluid's
  update of that step at the points the body held before it; in the old
  graph that depends on the order the nodes were added in;
* points owned by a third node reach the edge through a copy, one step
  old;
* a held body's transfer as a static sparse mapping, built from the same
  weights, delivers what the geometry edge delivers, and reads a field
  stored flat;
* on a kink of the interpolant the two samplers take the same slope with
  respect to a position, except exactly on an outermost cell centre;
* (slow) under ``convergence_norm="interface"`` a closed loop is as close
  to its fixed point at 256 x 256 cells as at 16 x 16 when point-sized
  fields cross its internal edges, whether the node or the edge does the
  sampling, and drifts away as the grid grows when the grid itself
  crosses a plain edge;
* (slow) inside a group the two point-sized wirings have different fixed
  points, by one step of the body's motion.

Seeded faults in ``src/maddening/core/coupling/grid_mapping.py`` that
this module catches (nine, each run against the tests named for it; the
list is in the pull request that added the file): the lattice's origin
moved by half a spacing, either way; the clamp removed, below, above and
on both sides, and moved from the last lattice point to the mesh's edge;
each corner's weight taken from the other side.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.sparse_mapping import sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

# ---------------------------------------------------------------------------
# The mesh, the fields and the points
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Mesh:
    """A Cartesian mesh of cells; a field's values sit at the cell centres."""

    shape: tuple[int, ...]          # cells per axis
    corner: tuple[float, ...]       # the mesh's low corner: a cell FACE
    cell: tuple[float, ...]         # the size of a cell

    @property
    def d(self) -> int:
        return len(self.shape)

    def axes(self) -> list[np.ndarray]:
        """The cell-centre coordinates along each axis."""
        return [c + h * (np.arange(n) + 0.5)
                for n, c, h in zip(self.shape, self.corner, self.cell)]

    def centres(self) -> np.ndarray:
        """``(*shape, d)``: the position of every cell centre."""
        return np.stack(np.meshgrid(*self.axes(), indexing="ij"), axis=-1)

    def at(self, s) -> np.ndarray:
        """The position ``s`` cells from the first cell centre, axis by axis."""
        s = np.asarray(s, np.float64)[..., :self.d]
        return np.asarray(self.corner) + (s + 0.5) * np.asarray(self.cell)

    def in_cells(self, points) -> np.ndarray:
        """The inverse of :meth:`at`, in float64."""
        points = np.asarray(points, np.float64)
        return (points - np.asarray(self.corner)) / np.asarray(self.cell) - 0.5


MESHES = {2: Mesh((8, 6), (-1.0, 0.0), (0.25, 0.5)),
          3: Mesh((8, 5, 4), (-1.0, 0.0, 0.5), (0.25, 0.5, 0.4))}

#: The velocity is ``OFFSET + GRADIENT @ x`` at the cell centres: its first
#: component is positive everywhere on both meshes, so the flow carries
#: the body towards +x and out.
GRADIENT = np.array([[0.8, 0.2, 0.1], [-0.4, 0.1, 0.05], [0.1, -0.2, 0.3]])
OFFSET = np.array([2.0, 0.3, -0.2])

#: The moving body's points, in cells from the first cell centre: inside;
#: on a cell centre; between the first row of centres and the mesh's edge
#: (already clamped across that row); about to leave on the right.  The
#: first two columns are the page's points.
MOVING = np.array([[1.42, 2.1, 0.6], [2.0, 1.0, 2.0], [4.7, -0.3, 1.3], [6.3, 3.9, 2.45]])
#: The same with no point on a cell-centre plane, where a central
#: difference is not a derivative.
GENERIC = MOVING + np.array([[0.0, 0.0, 0.0], [0.23, 0.31, -0.27], [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])


def linear_field(mesh: Mesh) -> np.ndarray:
    d = mesh.d
    return OFFSET[:d] + mesh.centres() @ GRADIENT[:d, :d].T


def rough_field(mesh: Mesh) -> np.ndarray:
    """The linear field plus a seeded perturbation of every value, so a
    weight on the wrong corner changes what is delivered."""
    noise = np.random.default_rng(74).uniform(-1.0, 1.0, mesh.shape + (mesh.d,))
    return linear_field(mesh) + 0.25 * noise


def classes(mesh: Mesh) -> dict[str, np.ndarray]:
    """One point of every class the page names, in cells from the first centre."""
    top = np.array(mesh.shape, np.float64) - 1.0
    points = {
        "inside": [1.42, 2.1, 0.6],
        "on a cell centre": [2.0, 1.0, 2.0],
        "on an outermost cell centre": [0.0, 1.3, 0.7],
        "between the first centre and the mesh's edge": [-0.3, 2.4, 1.2],
        "between the last centre and the mesh's edge": [top[0] + 0.3, 0.6, 2.2],
        "outside, below": [-1.7, 1.3, 0.4],
        "outside, above": [top[0] + 2.2, 3.1, 1.1],
        "outside on every axis": [-0.9, top[1] + 1.5, -2.0],
    }
    return {name: np.asarray(s)[:mesh.d] for name, s in points.items()}


@contextlib.contextmanager
def precision(dtype: str):
    """``jax_enable_x64`` as *dtype* needs it, restored afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@dataclasses.dataclass(frozen=True, eq=False)
class Case:
    mesh: Mesh
    dtype: str
    field0: np.ndarray              # (*mesh.shape, d)
    points0: np.ndarray             # (n_points, d), positions
    dt: float = 0.05

    @property
    def n_points(self) -> int:
        return int(self.points0.shape[0])

    def array(self, value):
        return jnp.asarray(value, self.dtype)

    def zeros(self, *shape):
        return jnp.zeros(shape, self.dtype)

    def lattice(self) -> dict:
        """The grid ``multilinear_grid`` is told of: its origin is the first
        cell CENTRE, its shape the number of cells."""
        return dict(origin=self.mesh.at(np.zeros(3)), spacing=self.mesh.cell,
                    shape=self.mesh.shape, n_points=self.n_points)


def case_of(d: int, dtype: str, points=MOVING, field=rough_field) -> Case:
    mesh = MESHES[d]
    return Case(mesh, dtype, field(mesh), mesh.at(points))


# ---------------------------------------------------------------------------
# The nodes: stand-ins, as on the page
# ---------------------------------------------------------------------------


def advance(velocity, dt):
    """The fluid's own step: each value relaxes towards its upstream neighbour."""
    return velocity + dt * 4.0 * (jnp.roll(velocity, 1, axis=0) - velocity)


def cell_centre_stencil(points, mesh: Mesh):
    """The corners and weights of multilinear interpolation between cell
    centres, clamped outside them: ``(index, weight)`` per corner, the
    index a tuple of one integer array per axis."""
    dtype = points.dtype
    top = jnp.asarray(mesh.shape, dtype) - 1.0
    s = (points - jnp.asarray(mesh.corner, dtype)) / jnp.asarray(mesh.cell, dtype) - 0.5
    s = jnp.clip(s, 0.0, top)
    low = jnp.clip(jnp.floor(s), 0.0, top - 1.0)
    t = s - low
    low = low.astype(jnp.int32)
    for corner in itertools.product((0, 1), repeat=mesh.d):
        c = np.array(corner)
        yield tuple((low + c).T), jnp.prod(jnp.where(c == 1, t, 1.0 - t), axis=1)


def sample_at_cell_centres(field, points, mesh: Mesh):
    """The node's own sampling: what the port moves onto the edge."""
    return sum(w.astype(field.dtype)[:, None] * field[index]
               for index, w in cell_centre_stencil(points, mesh))


def spread_from_points(amount, points, mesh: Mesh):
    """The transpose: each point's amount shared among its corners."""
    out = jnp.zeros(mesh.shape + amount.shape[1:], amount.dtype)
    for index, w in cell_centre_stencil(points, mesh):
        out = out.at[index].add(w.astype(amount.dtype)[:, None] * amount)
    return out


class _Node(SimulationNode):
    def __init__(self, name: str, case: Case):
        super().__init__(name, case.dt)
        self.case = case


class SamplingFluid(_Node):
    """Before: the fluid samples its own field for the body's sake."""

    def initial_state(self):
        c = self.case
        return {"velocity": c.array(c.field0),
                "velocity_at_points": c.zeros(c.n_points, c.mesh.d)}

    def boundary_input_spec(self):
        c = self.case
        return {"sample_points": BoundaryInputSpec(shape=(c.n_points, c.mesh.d), dtype=c.dtype)}

    def update(self, state, boundary_inputs, dt):
        c = self.case
        velocity = advance(state["velocity"], dt)
        points = boundary_inputs.get("sample_points", c.zeros(c.n_points, c.mesh.d))
        return {"velocity": velocity,
                "velocity_at_points": sample_at_cell_centres(velocity, points, c.mesh)}


class Fluid(_Node):
    """After: the fluid offers its field and nothing else."""

    flat = False

    def initial_state(self):
        c = self.case
        field = c.array(c.field0)
        return {"velocity": field.reshape(-1, c.mesh.d) if self.flat else field}

    def update(self, state, boundary_inputs, dt):
        c = self.case
        shaped = state["velocity"].reshape(c.mesh.shape + (c.mesh.d,))
        return {"velocity": advance(shaped, dt).reshape(state["velocity"].shape)}


class FlatFluid(Fluid):
    """The same fluid with its field stored raveled, ``(cells, d)``."""

    flat = True


class Body(_Node):
    """Owns the points, is carried by the velocity it is given at them."""

    moves = True

    def initial_state(self):
        c = self.case
        return {"points": c.array(c.points0), "seen": c.zeros(c.n_points, c.mesh.d)}

    def boundary_input_spec(self):
        c = self.case
        return {"fluid_velocity": BoundaryInputSpec(shape=(c.n_points, c.mesh.d), dtype=c.dtype)}

    def update(self, state, boundary_inputs, dt):
        seen = boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["seen"]))
        points = state["points"] + dt * seen if self.moves else state["points"]
        return {"points": points, "seen": seen}


class HeldBody(Body):
    """Points that never move."""

    moves = False


def before(case: Case, *, body_first: bool = False) -> GraphManager:
    gm = GraphManager()
    nodes = [SamplingFluid("fluid", case), Body("body", case)]
    for node in (reversed(nodes) if body_first else nodes):
        gm.add_node(node)
    gm.add_edge("body", "fluid", "points", "sample_points")
    gm.add_edge("fluid", "body", "velocity_at_points", "fluid_velocity")
    gm.compile()
    return gm


def after(case: Case, *, origin=None, body=Body, fluid=Fluid, body_first: bool = False,
          mapping=None) -> GraphManager:
    gm = GraphManager()
    nodes = [fluid("fluid", case), body("body", case)]
    for node in (reversed(nodes) if body_first else nodes):
        gm.add_node(node)
    geometry = ("target", "points")
    if mapping is None:
        lattice = case.lattice()
        if origin is not None:
            lattice["origin"] = origin
        mapping = multilinear_grid_mapping(
            mode="consistent", layout="flat" if fluid.flat else "shaped", **lattice)
    elif not getattr(mapping, "needs_geometry", False):
        geometry = None
    gm.add_edge("fluid", "body", "velocity", "fluid_velocity", mapping=mapping,
                geometry=geometry)
    gm.compile()
    return gm


def snapshot(gm: GraphManager, names=("fluid", "body")) -> dict:
    return {name: {key: np.asarray(value) for key, value in gm.get_node_state(name).items()}
            for name in names}


def trajectory(gm: GraphManager, steps: int, names=("fluid", "body")) -> list[dict]:
    """The state before the first step, then after each of *steps* steps."""
    out = [snapshot(gm, names)]
    for _ in range(steps):
        gm.step()
        out.append(snapshot(gm, names))
    return out


def in_units_of_rounding(got, expected, dtype: str, scale=None) -> float:
    """``max|got - expected|`` over ``eps * scale``; the scale is the
    largest magnitude of *expected* unless one is given (the field's,
    where a single value is compared)."""
    got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
    scale = float(np.max(np.abs(expected))) if scale is None else float(scale)
    return float(np.max(np.abs(got - expected))) / (float(np.finfo(dtype).eps) * scale)


def reference(mesh: Mesh, field, points) -> np.ndarray:
    """Multilinear interpolation between cell centres at each point's
    projection onto the box of centres, by an interpolator that shares no
    code with the node or the kernel.  Float64."""
    axes = mesh.axes()
    low = np.array([a[0] for a in axes])
    high = np.array([a[-1] for a in axes])
    interpolate = RegularGridInterpolator(axes, np.asarray(field, np.float64), method="linear")
    return interpolate(np.clip(np.asarray(points, np.float64), low, high))


#: Each dimension and each precision once per push; the other two
#: combinations in the slow lane.
CELLS = [(2, "float32"), (3, "float64"),
         pytest.param(2, "float64", marks=pytest.mark.slow),
         pytest.param(3, "float32", marks=pytest.mark.slow)]
RUN_STEPS = 10
#: The widest disagreement allowed between the two graphs' body states,
#: in ``eps * max|field|``.  Measured over the four cells on jax 0.10.2,
#: 0.11.0 and 0.11.2: at most 0.9 after ten steps (the node divides by the
#: cell size and then subtracts a half; the kernel subtracts the first
#: centre and then divides, so a weight can differ in its last bit, and
#: each step's sample moves the point the next step samples at).  Sixteen
#: leaves a factor of ten; half a cell is some 750,000 roundings in
#: float32.
ROUNDING = 16.0


# ---------------------------------------------------------------------------
# 1. The two graphs are the same graph
# ---------------------------------------------------------------------------


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_the_two_graphs_carry_the_body_alike_across_cells_and_out_of_the_mesh[2-float32] and tests/core/test_moving_a_sampling_onto_an_edge.py::test_the_two_graphs_carry_the_body_alike_across_cells_and_out_of_the_mesh[3-float64] (the run, in each dimension and each precision once)
@pytest.mark.parametrize("d, dtype", CELLS)
def test_the_two_graphs_carry_the_body_alike_across_cells_and_out_of_the_mesh(d, dtype):
    with precision(dtype):
        case = case_of(d, dtype)
        old = trajectory(before(case), RUN_STEPS)
        new = trajectory(after(case), RUN_STEPS)
    mesh = case.mesh
    # The premises: the body crossed several cells, one point ended
    # outside the mesh on one side, and one is still inside it.
    travelled = mesh.in_cells(new[-1]["body"]["points"]) - mesh.in_cells(new[0]["body"]["points"])
    assert np.max(travelled[:, 0]) > 3.0, travelled
    right_edge = mesh.corner[0] + mesh.shape[0] * mesh.cell[0]
    final_x = new[-1]["body"]["points"][:, 0]
    assert np.any(final_x > right_edge) and np.any(final_x < right_edge - mesh.cell[0]), final_x
    worst = 0.0
    for n in range(1, RUN_STEPS + 1):
        assert np.array_equal(old[n]["fluid"]["velocity"], new[n]["fluid"]["velocity"])
        for field in ("seen", "points"):
            worst = max(worst, in_units_of_rounding(old[n]["body"][field], new[n]["body"][field],
                                                    dtype))
    assert worst <= ROUNDING, f"{d}-D {dtype}: the body's state differs by {worst:.3g} roundings"


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_every_class_of_point_is_sampled_as_an_independent_interpolator_samples_it[2-float32] and tests/core/test_moving_a_sampling_onto_an_edge.py::test_every_class_of_point_is_sampled_as_an_independent_interpolator_samples_it[3-float64] (the classes of point, in each dimension and each precision once)
@pytest.mark.parametrize("d, dtype", CELLS)
def test_every_class_of_point_is_sampled_as_an_independent_interpolator_samples_it(d, dtype):
    mesh = MESHES[d]
    named = classes(mesh)
    points = mesh.at(np.stack(list(named.values())))
    # The clamp, said without an interpolator: a point outside the box of
    # cell centres receives what its projection onto the box receives.
    # The projections ride in the same graphs, as eight more points.
    axes = mesh.axes()
    projected = np.clip(points, [a[0] for a in axes], [a[-1] for a in axes])
    moved = np.any(projected != points, axis=1)
    assert int(moved.sum()) == 5, "premise: five of the eight points are outside the box"
    with precision(dtype):
        case = Case(mesh, dtype, rough_field(mesh), np.concatenate([points, projected]))
        stepped = {"before": trajectory(before(case), 1), "after": trajectory(after(case), 1)}
    k = len(named)
    for wiring, (start, end) in stepped.items():
        # As stored: a float32 point is where float32 put it.
        expected = reference(mesh, end["fluid"]["velocity"], start["body"]["points"])
        largest = np.max(np.abs(end["fluid"]["velocity"]))
        seen = end["body"]["seen"]
        for i, name in enumerate(named):
            off = in_units_of_rounding(seen[i], expected[i], dtype, scale=largest)
            assert off <= ROUNDING, f"{wiring}, {d}-D {dtype}, {name}: {off:.3g} roundings off"
        off = in_units_of_rounding(seen[:k][moved], seen[k:][moved], dtype)
        assert off <= ROUNDING, f"{wiring}, {d}-D {dtype}: {off:.3g} roundings from the projection"
    off = in_units_of_rounding(stepped["before"][1]["body"]["seen"],
                               stepped["after"][1]["body"]["seen"], dtype)
    assert off <= ROUNDING, f"{d}-D {dtype}: the two graphs are {off:.3g} roundings apart"


# ---------------------------------------------------------------------------
# 2. The half-cell origin
# ---------------------------------------------------------------------------


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_a_lattice_that_starts_at_the_meshs_corner_is_half_a_cell_off[2-float32] and tests/core/test_moving_a_sampling_onto_an_edge.py::test_a_lattice_that_starts_at_the_meshs_corner_is_half_a_cell_off[3-float64] (the half cell, in each dimension and each precision once)
@pytest.mark.parametrize("d, dtype", CELLS)
def test_a_lattice_that_starts_at_the_meshs_corner_is_half_a_cell_off(d, dtype):
    """On a field of known gradient, at points the half-cell shift does not
    carry past the last centre: off by the gradient times half a cell."""
    mesh = MESHES[d]
    interior = np.array([[1.42, 2.1, 0.6], [2.0, 1.0, 2.0], [3.3, 0.4, 1.7], [0.2, 2.9, 0.1]])
    with precision(dtype):
        case = Case(mesh, dtype, linear_field(mesh), mesh.at(interior))
        # The mapping alone, on the field as it starts.
        field = case.array(case.field0)
        points = case.array(case.points0)
        right = multilinear_grid_mapping(mode="consistent", layout="shaped", **case.lattice())
        wrong = multilinear_grid_mapping(mode="consistent", layout="shaped",
                                         **{**case.lattice(), "origin": mesh.corner})
        exact = OFFSET[:d] + np.asarray(points, np.float64) @ GRADIENT[:d, :d].T
        shift = GRADIENT[:d, :d] @ (0.5 * np.asarray(mesh.cell))
        direct = jax.jit(lambda f, p: (right.apply(f, None, p), wrong.apply(f, None, p),
                                       sample_at_cell_centres(f, p, mesh)))
        got_right, got_wrong, in_node = (np.asarray(v, np.float64) for v in direct(field, points))
        # In the graph, over a run.
        old = trajectory(before(case), 3)
        new = trajectory(after(case, origin=mesh.corner), 3)
    eps = float(np.finfo(dtype).eps)
    scale = float(np.max(np.abs(case.field0)))
    assert np.max(np.abs(got_right - exact)) <= ROUNDING * eps * scale
    assert np.max(np.abs(in_node - exact)) <= ROUNDING * eps * scale
    assert np.max(np.abs(got_wrong - (exact + shift))) <= ROUNDING * eps * scale
    assert np.min(np.abs(shift)) > 0.01                     # every component is off
    for n in (1, 2, 3):
        off = in_units_of_rounding(new[n]["body"]["seen"], old[n]["body"]["seen"], dtype)
        assert off > 1000 * ROUNDING, f"step {n}: the wrong origin is only {off:.3g} roundings off"


# ---------------------------------------------------------------------------
# 3. Gradients
# ---------------------------------------------------------------------------


def _loss_of(gm: GraphManager, case: Case, steps: int):
    """The body's final position, weighted so that every coordinate counts,
    as a function of the initial field and the initial points."""
    start = {name: dict(gm.get_node_state(name)) for name in ("fluid", "body")}
    weight = case.array(1.0 + 0.1 * np.arange(case.n_points * case.mesh.d).reshape(
        case.n_points, case.mesh.d))

    def loss(field0, points0):
        gm.set_node_state("fluid", {**start["fluid"], "velocity": field0})
        gm.set_node_state("body", {**start["body"], "points": points0})
        final = gm.run_scan(n_steps=steps)
        return jnp.sum(weight * final["body"]["points"])

    return loss


def _gradients(gm: GraphManager, case: Case, steps: int):
    loss = _loss_of(gm, case, steps)
    grads = jax.grad(loss, argnums=(0, 1))(case.array(case.field0), case.array(case.points0))
    gm.reset_state()            # the transform left tracers in the graph's state
    return loss, [np.asarray(g, np.float64) for g in grads]


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_the_gradients_of_the_two_graphs_agree_and_are_the_central_difference[2] (the same comparison in two dimensions)
@pytest.mark.parametrize("d", (2, pytest.param(3, marks=pytest.mark.slow)))
def test_the_gradients_of_the_two_graphs_agree_and_are_the_central_difference(d):
    steps = 6
    with precision("float64"):
        case = case_of(d, "float64", points=GENERIC)
        mesh = case.mesh
        # The premise of a central difference: no point comes within a
        # thousandth of a cell of a cell-centre plane (a kink) in the run.
        run = trajectory(after(case), steps)
        s = np.concatenate([mesh.in_cells(state["body"]["points"]) for state in run[:-1]])
        assert np.min(np.abs(s - np.round(s))) > 1e-3
        assert np.any(s > np.array(mesh.shape) - 1.0), "premise: a point is clamped in the run"

        old, new = before(case), after(case)
        loss_old, g_old = _gradients(old, case, steps)
        loss_new, g_new = _gradients(new, case, steps)
        rng = np.random.default_rng(d)
        for which, (a, b) in enumerate(zip(g_old, g_new)):
            size = float(np.max(np.abs(b)))
            assert size > 1e-3, "premise: the loss depends on this input"
            assert np.max(np.abs(a - b)) <= 1e-12 * size, (which, np.max(np.abs(a - b)), size)
        base = [np.asarray(case.field0, np.float64), np.asarray(case.points0, np.float64)]
        h = 1e-6
        for which in (0, 1):
            for _ in range(3):
                v = rng.choice([-1.0, 1.0], size=base[which].shape)
                up = [x + h * v if k == which else x for k, x in enumerate(base)]
                down = [x - h * v if k == which else x for k, x in enumerate(base)]
                slope = float(np.vdot(g_new[which], v))
                for loss in (loss_old, loss_new):
                    fd = (float(loss(*map(case.array, up))) - float(loss(*map(case.array, down)))) / (
                        2 * h)
                    assert abs(fd - slope) <= 1e-7 * max(abs(slope), 1.0), (which, fd, slope)


def test_the_gradients_agree_in_float32_with_a_point_on_a_cell_centre():
    """The page's own run: float32, a point that starts on a cell centre
    (both samplers take the slope of the cell above it there)."""
    with precision("float32"):
        case = case_of(2, "float32")
        _, g_old = _gradients(before(case), case, 6)
        _, g_new = _gradients(after(case), case, 6)
    for a, b in zip(g_old, g_new):
        assert np.max(np.abs(b)) > 1e-3
        assert np.max(np.abs(a - b)) <= 1e-5 * np.max(np.abs(b))


def test_on_a_kink_the_two_samplers_take_the_same_slope_except_on_the_hulls_faces():
    """With respect to a position: on an interior cell-centre plane both
    take the slope of the cell above; exactly on an outermost cell centre
    the kernel takes the whole interior slope and a node that clamps with
    ``jnp.clip`` half of it; strictly outside, both are zero."""
    with precision("float64"):
        mesh = MESHES[2]
        top = mesh.shape[0] - 1.0
        s = np.array([[2.0, 1.0], [0.0, 1.3], [top, 2.2], [-0.7, 1.3], [1.42, 2.1]])
        field = jnp.asarray(rough_field(mesh))
        mapping = multilinear_grid_mapping(
            origin=mesh.at(np.zeros(3)), spacing=mesh.cell, shape=mesh.shape, n_points=len(s),
            mode="consistent", layout="shaped")

        nudge = np.array([1e-6, 0.0])
        places = jnp.asarray(np.stack([mesh.at(s), mesh.at(s + nudge), mesh.at(s - nudge)]))

        def slopes(sample):
            """d(first component at each point) / d(that point's x), on the
            kink, just above it and just below it."""
            one = jax.grad(lambda points: jnp.sum(sample(points)[:, 0]))
            return np.asarray(jax.jit(jax.vmap(one))(places))[:, :, 0]

        on_node, _, _ = slopes(lambda points: sample_at_cell_centres(field, points, mesh))
        on_kernel, above, below = slopes(lambda points: mapping.apply(field, None, points))
    assert abs(above[0] - below[0]) > 0.1, "premise: the slope jumps at the interior plane"
    for got in (on_node, on_kernel):
        assert got[0] == pytest.approx(above[0], rel=1e-9)      # an interior centre: the cell above
        assert got[3] == 0.0                                    # strictly outside
        assert got[4] == pytest.approx(above[4], rel=1e-9)      # inside a cell: no kink
    # The hull's two faces: the interior one-sided slope, or half of it.
    assert on_kernel[1] == pytest.approx(above[1], rel=1e-9)
    assert on_kernel[2] == pytest.approx(below[2], rel=1e-9)
    assert on_node[1] == pytest.approx(0.5 * above[1], rel=1e-9)
    assert on_node[2] == pytest.approx(0.5 * below[2], rel=1e-9)
    assert abs(above[1]) > 0.1 and abs(below[2]) > 0.1


# ---------------------------------------------------------------------------
# 4. The time level, and points held by a third node
# ---------------------------------------------------------------------------


def test_the_body_receives_the_field_after_the_step_at_the_points_before_it():
    with precision("float32"):
        case = case_of(2, "float32")
        mesh = case.mesh
        new = trajectory(after(case), 4)
        added_last = trajectory(after(case, body_first=True), 4)
        old = trajectory(before(case), 4)
        old_body_first = trajectory(before(case, body_first=True), 4)

        def sampled(field, points):
            return np.asarray(sample_at_cell_centres(jnp.asarray(field), jnp.asarray(points), mesh))

        for n in (1, 2, 3, 4):
            field_now, field_then = new[n]["fluid"]["velocity"], new[n - 1]["fluid"]["velocity"]
            held_before, held_after = new[n - 1]["body"]["points"], new[n]["body"]["points"]
            received = new[n]["body"]["seen"]
            assert in_units_of_rounding(received, sampled(field_now, held_before),
                                        "float32") <= ROUNDING
            # Neither the points after the step nor the field before it.
            assert in_units_of_rounding(received, sampled(field_now, held_after), "float32") > 1e4
            assert in_units_of_rounding(received, sampled(field_then, held_before), "float32") > 1e4
            # One edge, no cycle: the order the nodes are added in changes nothing.
            assert np.array_equal(added_last[n]["body"]["seen"], received)
            # The old graph, fluid added first, is at the same time level ...
            assert in_units_of_rounding(old[n]["body"]["seen"], received, "float32") <= ROUNDING
            # ... and with the body added first it reads the field of the
            # step before (nothing at all at the first step).
            late = old_body_first[n]["body"]["seen"]
            if n == 1:
                assert not np.any(late)
            else:
                then = old_body_first[n - 1]
                assert in_units_of_rounding(late, sampled(then["fluid"]["velocity"],
                                                          then["body"]["points"]),
                                            "float32") <= ROUNDING
                assert in_units_of_rounding(late, received, "float32") > 1e4


class Carrier(_Node):
    """A third node that owns the points and moves them."""

    def initial_state(self):
        return {"points": self.case.array(self.case.points0)}

    def update(self, state, boundary_inputs, dt):
        drift = self.case.array([0.9, -0.3, 0.2][:self.case.mesh.d])
        return {"points": state["points"] + dt * drift}


class Probe(_Node):
    """Keeps a copy of the carrier's points: the edge reads its own target."""

    def initial_state(self):
        c = self.case
        return {"points": c.array(c.points0), "seen": c.zeros(c.n_points, c.mesh.d)}

    def boundary_input_spec(self):
        c = self.case
        shape = (c.n_points, c.mesh.d)
        return {"points_in": BoundaryInputSpec(shape=shape, dtype=c.dtype),
                "fluid_velocity": BoundaryInputSpec(shape=shape, dtype=c.dtype)}

    def update(self, state, boundary_inputs, dt):
        return {"points": boundary_inputs.get("points_in", state["points"]),
                "seen": boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["seen"]))}


def test_points_held_by_a_third_node_are_copied_and_read_one_step_old():
    names = ("fluid", "carrier", "probe")
    with precision("float32"):
        case = case_of(2, "float32")
        mapping = multilinear_grid_mapping(mode="consistent", layout="shaped", **case.lattice())
        gm = GraphManager()
        gm.add_node(Fluid("fluid", case))
        gm.add_node(Carrier("carrier", case))
        gm.add_node(Probe("probe", case))
        with pytest.raises(ValueError, match="held by any other node is not supported"):
            gm.add_edge("fluid", "probe", "velocity", "fluid_velocity", mapping=mapping,
                        geometry=("carrier", "points"))
        gm.add_edge("carrier", "probe", "points", "points_in")
        gm.add_edge("fluid", "probe", "velocity", "fluid_velocity", mapping=mapping,
                    geometry=("target", "points"))
        gm.compile()
        run = trajectory(gm, 4, names)
        for n in (1, 2, 3, 4):
            field = jnp.asarray(run[n]["fluid"]["velocity"])
            received = run[n]["probe"]["seen"]

            def at(points):
                return np.asarray(sample_at_cell_centres(field, jnp.asarray(points), case.mesh))

            # The copy is current at the end of every step ...
            assert np.array_equal(run[n]["probe"]["points"], run[n]["carrier"]["points"])
            # ... and the edge read the one made a step earlier.
            assert in_units_of_rounding(received, at(run[n - 1]["carrier"]["points"]),
                                        "float32") <= ROUNDING
            assert in_units_of_rounding(received, at(run[n]["carrier"]["points"]), "float32") > 1e4


# ---------------------------------------------------------------------------
# 5. A held body: the same transfer as a static sparse mapping
# ---------------------------------------------------------------------------


def static_rows(case: Case):
    """``(indices, values)``, ``2**d`` entries a row, from the node's own
    stencil at the points the body is held at."""
    with precision("float64"):
        rows = list(cell_centre_stencil(jnp.asarray(case.points0, jnp.float64), case.mesh))
        indices = np.stack([np.ravel_multi_index([np.asarray(i) for i in index], case.mesh.shape)
                            for index, _ in rows], axis=1)
        values = np.stack([np.asarray(w) for _, w in rows], axis=1)
    return indices, values.astype(case.dtype)


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_a_held_bodys_static_sparse_mapping_delivers_what_the_geometry_edge_delivers[2-float32] and tests/core/test_moving_a_sampling_onto_an_edge.py::test_a_held_bodys_static_sparse_mapping_delivers_what_the_geometry_edge_delivers[3-float64] (the held body, in each dimension and each precision once)
@pytest.mark.parametrize("d, dtype", CELLS)
def test_a_held_bodys_static_sparse_mapping_delivers_what_the_geometry_edge_delivers(d, dtype):
    mesh = MESHES[d]
    with precision(dtype):
        case = Case(mesh, dtype, rough_field(mesh), mesh.at(np.stack(list(classes(mesh).values()))))
        indices, values = static_rows(case)
        assert indices.shape == values.shape == (case.n_points, 2 ** d)
        assert np.allclose(values.sum(axis=1), 1.0)
        static = sparse_matrix_mapping(indices, values, n_source=int(np.prod(mesh.shape)))
        on_the_edge = trajectory(after(case, body=HeldBody, fluid=FlatFluid), 5)
        fixed = after(case, body=HeldBody, fluid=FlatFluid, mapping=static)
        key = "fluid.velocity->body.fluid_velocity"
        assert fixed.params["mappings"][key]["W"].shape == (case.n_points, 2 ** d)
        frozen = trajectory(fixed, 5)
        # A static sparse mapping reads its source along the first axis:
        # the field stored with the grid's shape is refused.
        with pytest.raises(ValueError, match="does not match fluid.velocity"):
            after(case, body=HeldBody, fluid=Fluid, mapping=static)
    for n in range(1, 6):
        assert np.array_equal(frozen[n]["body"]["points"], frozen[0]["body"]["points"])
        off = in_units_of_rounding(frozen[n]["body"]["seen"], on_the_edge[n]["body"]["seen"], dtype)
        assert off <= ROUNDING, f"{d}-D {dtype}, step {n}: {off:.3g} roundings apart"


# ---------------------------------------------------------------------------
# 6. A closed loop under the interface norm, at three grid sizes
# ---------------------------------------------------------------------------

#: The loop: the body drags on the fluid where it is.  One pass multiplies
#: an error by ``RESPONSE * DRAG`` times the sum of a point's squared
#: weights; the points sit a sixth of a cell from a cell centre at every
#: size (see :func:`loop_points`), which makes that 0.63.
RESPONSE, DRAG, MASS = 1.0, 1.2, 5.0
#: Small enough that the body moves under two hundredths of the finest
#: cell in the step: the three sizes are then one problem.
LOOP_DT = 2e-4
LOOP_SIZES = (16, 64, 256)
RTOL, TIGHT_RTOL = 1e-4, 1e-11
#: The wirings: the node's own sampling and spreading, plain edges that
#: carry point-sized fields; the grid handed whole to the body, which
#: samples it; the sampling and the spreading on mapped edges.
WIRINGS = ("in the fluid node", "grid handed whole", "on the edges")


def loop_points() -> np.ndarray:
    """Four points of the unit square, a sixth of a cell from a cell centre
    on a mesh of 16, 64 or 256 cells a side (multiples of 1/48)."""
    return np.array([[10, 13], [22, 20], [31, 35], [40, 8]]) / 48.0


def loop_case(n: int, dtype: str = "float64") -> Case:
    mesh = Mesh((n, n), (0.0, 0.0), (1.0 / n, 1.0 / n))
    x, y = np.moveaxis(mesh.centres(), -1, 0)
    field = np.stack([1.0 + np.sin(3 * x) * np.cos(2 * y), 0.5 * np.cos(4 * x + y)], axis=-1)
    return Case(mesh, dtype, field, loop_points(), dt=LOOP_DT)


class LoopBody(_Node):
    """Dragged by the fluid; ``samples`` says whether it is handed the
    velocity at its points or the whole grid."""

    samples = False

    def initial_state(self):
        c = self.case
        own = np.array([[0.3, 0.1], [-0.2, 0.25], [0.1, -0.3], [0.2, 0.2]])
        return {"points": c.array(c.points0), "velocity": c.array(own),
                "force": c.zeros(c.n_points, 2), "seen": c.zeros(c.n_points, 2)}

    def boundary_input_spec(self):
        c = self.case
        if self.samples:
            return {"fluid_field": BoundaryInputSpec(shape=c.mesh.shape + (2,), dtype=c.dtype)}
        return {"fluid_velocity": BoundaryInputSpec(shape=(c.n_points, 2), dtype=c.dtype)}

    def update(self, state, boundary_inputs, dt):
        c = self.case
        if self.samples:
            field = boundary_inputs.get("fluid_field", c.zeros(*c.mesh.shape, 2))
            seen = sample_at_cell_centres(field, state["points"], c.mesh)
        else:
            seen = boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["seen"]))
        force = DRAG * (state["velocity"] - seen)           # on the fluid
        velocity = state["velocity"] - dt * force / MASS
        return {"points": state["points"] + dt * velocity, "velocity": velocity,
                "force": force, "seen": seen}


class SamplingLoopBody(LoopBody):
    samples = True


class LoopFluid(_Node):
    """Takes the body's force on its cells."""

    def initial_state(self):
        return {"velocity": self.case.array(self.case.field0)}

    def boundary_input_spec(self):
        c = self.case
        return {"body_force": BoundaryInputSpec(shape=c.mesh.shape + (2,), dtype=c.dtype)}

    def update(self, state, boundary_inputs, dt):
        force = boundary_inputs.get("body_force", jnp.zeros_like(state["velocity"]))
        return {"velocity": advance(state["velocity"], dt) + RESPONSE * force}


class SamplingLoopFluid(_Node):
    """Takes the body's points and its force, spreads and samples itself."""

    def initial_state(self):
        c = self.case
        return {"velocity": c.array(c.field0), "velocity_at_points": c.zeros(c.n_points, 2)}

    def boundary_input_spec(self):
        c = self.case
        spec = BoundaryInputSpec(shape=(c.n_points, 2), dtype=c.dtype)
        return {"sample_points": spec, "point_force": spec}

    def update(self, state, boundary_inputs, dt):
        c = self.case
        points = boundary_inputs.get("sample_points", c.zeros(c.n_points, 2))
        force = boundary_inputs.get("point_force", c.zeros(c.n_points, 2))
        velocity = advance(state["velocity"], dt) + RESPONSE * spread_from_points(
            force, points, c.mesh)
        return {"velocity": velocity,
                "velocity_at_points": sample_at_cell_centres(velocity, points, c.mesh)}


def loop(case: Case, wiring: str, rtol: float) -> GraphManager:
    gm = GraphManager()
    if wiring == "on the edges":
        gm.add_node(LoopFluid("fluid", case))
        gm.add_node(LoopBody("body", case))
        lattice = dict(layout="shaped", **case.lattice())
        gm.add_edge("fluid", "body", "velocity", "fluid_velocity",
                    mapping=multilinear_grid_mapping(mode="consistent", **lattice),
                    geometry=("target", "points"))
        gm.add_edge("body", "fluid", "force", "body_force",
                    mapping=multilinear_grid_mapping(mode="conservative", **lattice),
                    geometry=("source", "points"))
    else:
        gm.add_node(SamplingLoopFluid("fluid", case))
        whole = wiring == "grid handed whole"
        gm.add_node((SamplingLoopBody if whole else LoopBody)("body", case))
        gm.add_edge("body", "fluid", "points", "sample_points")
        gm.add_edge("body", "fluid", "force", "point_force")
        if whole:
            gm.add_edge("fluid", "body", "velocity", "fluid_field")
        else:
            gm.add_edge("fluid", "body", "velocity_at_points", "fluid_velocity")
    gm.add_coupling_group(["fluid", "body"], max_iterations=200, convergence_norm="interface",
                          rtol=rtol)
    gm.compile()
    return gm


def solve_once(case: Case, wiring: str, rtol: float) -> dict:
    """One step of a fresh graph: the report and the value the body received."""
    gm = loop(case, wiring, rtol)
    gm.step()
    report = gm.coupling_diagnostics()["body+fluid"]
    return {"converged": bool(report["converged"]), "iterations": int(report["iterations"]),
            "seen": np.asarray(gm.get_node_state("body")["seen"], np.float64)}


def tolerances_from_the_fixed_point(n: int, wiring: str) -> dict:
    """How far the value the body received is from a solve seven decades
    tighter, as the norm counts it: RMS over ``rtol * max|value|``."""
    with precision("float64"):
        case = loop_case(n)
        loose = solve_once(case, wiring, RTOL)
        tight = solve_once(case, wiring, TIGHT_RTOL)
    assert loose["converged"] and tight["converged"], (n, wiring, loose, tight)
    distance = np.sqrt(np.mean((loose["seen"] - tight["seen"]) ** 2)) / (
        RTOL * np.max(np.abs(tight["seen"])))
    return {"distance": float(distance), "iterations": loose["iterations"]}


def test_the_loop_on_mapped_edges_converges_within_a_tolerance_of_its_fixed_point():
    """The per-push end of the measurement below: the smallest grid and
    the mapped pair."""
    seen = tolerances_from_the_fixed_point(LOOP_SIZES[0], "on the edges")
    assert seen["distance"] <= 1.0, seen
    assert seen["iterations"] > 5, "premise: the loop needs passes"


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_the_loop_on_mapped_edges_converges_within_a_tolerance_of_its_fixed_point (the smallest grid, the mapped pair)
@pytest.mark.slow
def test_point_sized_readings_keep_a_converged_loop_near_its_fixed_point_at_every_grid_size():
    """The page's table, with loose bounds.

    Measured (jax 0.10.2, 0.11.0 and 0.11.2, float64, Gauss-Seidel,
    ``rtol=1e-4``), tolerances from the fixed point at 16, 64 and 256
    cells a side: 0.13, 0.13, 0.18 with the sampling in the fluid node;
    0.13, 0.13, 0.14 on the edges; 0.83, 3.4, 13.8 with the grid handed
    whole to the body.

    The bounds.  One tolerance for the two point-sized wirings: the loop
    contracts by 0.63 a pass, so an iterate whose residual met the
    criterion is within ``0.63 / 0.37 = 1.7`` residuals of the fixed
    point and the state returned (one more pass) within 1.1, of a
    residual that is at most one tolerance; measured, a seventh of it.
    For the grid handed whole: the criterion pools every cell with the
    few values that move, so its distance grows as the square root of
    the entry count, sixteenfold from 16 to 256 cells a side; a third of
    that growth is asked for, and three tolerances at the largest size.
    """
    seen = {wiring: [tolerances_from_the_fixed_point(n, wiring) for n in LOOP_SIZES]
            for wiring in WIRINGS}
    table = {wiring: [round(row["distance"], 3) for row in rows] for wiring, rows in seen.items()}
    for wiring in ("in the fluid node", "on the edges"):
        distances = [row["distance"] for row in seen[wiring]]
        assert max(distances) <= 1.0, table
        assert len({row["iterations"] for row in seen[wiring]}) == 1, seen[wiring]
    whole = [row["distance"] for row in seen["grid handed whole"]]
    assert whole[2] >= 3.0, table
    assert whole[2] / whole[0] >= (LOOP_SIZES[2] / LOOP_SIZES[0]) / 3.0, table
    passes = [row["iterations"] for row in seen["grid handed whole"]]
    assert passes[0] > passes[1] > passes[2], f"it stops earlier as the grid grows: {passes}"


# Per push: tests/core/test_moving_a_sampling_onto_an_edge.py::test_the_loop_on_mapped_edges_converges_within_a_tolerance_of_its_fixed_point (the mapped loop against its own tight solve)
@pytest.mark.slow
def test_inside_a_group_the_port_moves_the_sampling_by_one_step_of_the_bodys_motion():
    """A plain edge that carries the points reads the iterate, so the old
    fluid sampled at the body's end-of-step positions; the target-anchored
    edge reads the positions from before the step.  The two loops have
    different fixed points, first order in the step: measured 2.2e-4 of
    the value at ``LOOP_DT`` and 2.2e-3 at ten times it (16 cells a side,
    jax 0.10.2, 0.11.0 and 0.11.2)."""
    gaps = []
    with precision("float64"):
        for factor in (1, 10):
            case = dataclasses.replace(loop_case(LOOP_SIZES[0]), dt=factor * LOOP_DT)
            node = solve_once(case, "in the fluid node", TIGHT_RTOL)
            edge = solve_once(case, "on the edges", TIGHT_RTOL)
            assert node["converged"] and edge["converged"]
            gaps.append(float(np.sqrt(np.mean((node["seen"] - edge["seen"]) ** 2))
                              / np.max(np.abs(edge["seen"]))))
    assert 1e-4 < gaps[0] < 5e-4, gaps                 # far above the tight tolerance, 1e-11
    assert 8.0 < gaps[1] / gaps[0] < 12.0, gaps        # first order in the step
