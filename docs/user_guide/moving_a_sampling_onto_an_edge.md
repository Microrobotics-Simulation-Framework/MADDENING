# Moving a node's own sampling onto an edge

*A worked example.  It uses `add_edge(..., geometry=...)` and the
`multilinear_grid` kind, which are experimental in 0.4.0: see
[Geometry-dependent mappings](geometry_dependent_mappings.md).*

A node that holds a field on a grid is often asked for the field at
points another node owns: a flow solver for the velocity at the markers
of a body.  One way to answer is inside the grid node: it takes the
points as a boundary input and interpolates in its own `update`.  On
0.4.0 the interpolation belongs on the edge.  This page makes that port
once, on a small example, and checks each step by running it: the two
graphs deliver the same values, and the three places where the port goes
wrong are shown going wrong.

The mesh is **cell-centred**: a field's values sit at the centres of the
cells.  The example has two dimensions so that its numbers fit on a
screen; [In three dimensions](#in-three-dimensions) says what changes.
Both nodes are stand-ins of a few lines: a "fluid" whose velocity relaxes
towards its upstream neighbour, and a "body" whose points are carried by
the velocity they are given.

Every block on this page runs, one after the other, as one program
(`tests/compliance/test_docs_snippets.py` runs it on every push), and
`tests/core/test_moving_a_sampling_onto_an_edge.py` holds the same two
graphs to each other in two and three dimensions, in float32 and
float64.

## Before: the node samples

The fluid takes `sample_points` as an input and returns
`velocity_at_points`: bilinear interpolation between cell centres,
clamped outside them.  That input exists for the body's sake.

```python
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

NX, NY = 8, 6                        # cells along x and along y
CORNER = np.array([-1.0, 0.0])       # the mesh's low corner: a cell FACE
CELL = np.array([0.25, 0.5])         # the size of a cell
N_POINTS = 4
DT = 0.05

# A velocity with a known gradient, stored at the cell centres: (NX, NY, 2).
GRADIENT = np.array([[0.8, 0.2], [-0.4, 0.1]])
OFFSET = np.array([2.0, 0.3])
CENTRES = np.stack(np.meshgrid(CORNER[0] + CELL[0] * (np.arange(NX) + 0.5),
                               CORNER[1] + CELL[1] * (np.arange(NY) + 0.5),
                               indexing="ij"), axis=-1)
VELOCITY0 = jnp.asarray(OFFSET + CENTRES @ GRADIENT.T, jnp.float32)

POINTS0 = jnp.asarray([[-0.52, 1.30],     # inside the mesh
                       [-0.375, 0.75],    # on a cell centre
                       [0.30, 0.10],      # between the first row of centres and the mesh's edge
                       [0.70, 2.20]],     # about to leave the mesh on the right
                      jnp.float32)


def advance(velocity, dt):
    """The fluid's own step: each value relaxes towards its upstream neighbour."""
    return velocity + dt * 4.0 * (jnp.roll(velocity, 1, axis=0) - velocity)


def cell_centre_stencil(points):
    """The four corners and weights of bilinear interpolation between cell
    centres; a point outside the box of centres is clamped onto it."""
    top = jnp.asarray([NX - 1.0, NY - 1.0])
    s = jnp.clip((points - CORNER) / CELL - 0.5, 0.0, top)    # in cells, from the first centre
    low = jnp.clip(jnp.floor(s), 0.0, top - 1.0)
    t = s - low
    low = low.astype(jnp.int32)
    for corner in itertools.product((0, 1), repeat=2):
        c = np.array(corner)
        yield tuple((low + c).T), jnp.prod(jnp.where(c == 1, t, 1.0 - t), axis=1)


def sample_at_cell_centres(field, points):
    return sum(w[:, None] * field[index] for index, w in cell_centre_stencil(points))


class SamplingFluid(SimulationNode):
    """The fluid, sampling its own field for another node."""

    def initial_state(self):
        return {"velocity": VELOCITY0,
                "velocity_at_points": jnp.zeros((N_POINTS, 2), jnp.float32)}

    def boundary_input_spec(self):
        return {"sample_points": BoundaryInputSpec(shape=(N_POINTS, 2), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        velocity = advance(state["velocity"], dt)
        points = boundary_inputs.get("sample_points", jnp.zeros((N_POINTS, 2), jnp.float32))
        return {"velocity": velocity,
                "velocity_at_points": sample_at_cell_centres(velocity, points)}


class Body(SimulationNode):
    """Owns the points; they are carried by the velocity they are given."""

    def initial_state(self):
        return {"points": POINTS0, "seen": jnp.zeros((N_POINTS, 2), jnp.float32)}

    def boundary_input_spec(self):
        return {"fluid_velocity": BoundaryInputSpec(shape=(N_POINTS, 2), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        seen = boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["seen"]))
        return {"points": state["points"] + dt * seen, "seen": seen}


before = GraphManager()
before.add_node(SamplingFluid("fluid", DT))
before.add_node(Body("body", DT))
before.add_edge("body", "fluid", "points", "sample_points")
before.add_edge("fluid", "body", "velocity_at_points", "fluid_velocity")
before.compile()

text = before.format_graph().splitlines()
print("\n".join(text[text.index("Edges (2)"):text.index("Coupling groups (0)")]))
assert "back edge" in text[text.index("  body.points -> fluid.sample_points") + 1]
```

The two edges close a cycle, and outside a coupling group a cycle is cut
at a *back edge*, which reads the previous step's value.  `format_graph()`
says which one: `body.points -> fluid.sample_points`, because the fluid
was added first.  So in this graph the fluid runs first and samples its
new field at the points the body held when the step began.  Keep that in
mind: it is the time level the edge will have to reproduce.

## After: the edge samples

The fluid offers its field and nothing else.  The edge maps it:

<!-- snippet: continues -->
```python
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping


class Fluid(SimulationNode):
    """The fluid, with no input that exists for another node."""

    def initial_state(self):
        return {"velocity": VELOCITY0}

    def update(self, state, boundary_inputs, dt):
        return {"velocity": advance(state["velocity"], dt)}


lattice = dict(origin=CORNER + 0.5 * CELL,      # the first cell CENTRE
               spacing=CELL,
               shape=(NX, NY),                  # the number of cells
               n_points=N_POINTS)
gather = multilinear_grid_mapping(mode="consistent", layout="shaped", **lattice)

after = GraphManager()
after.add_node(Fluid("fluid", DT))
after.add_node(Body("body", DT))
after.add_edge("fluid", "body", "velocity", "fluid_velocity",
               mapping=gather, geometry=("target", "points"))
after.compile()
```

One edge instead of two, and no cycle.  The body is unchanged.  The next
three sections are the three arguments of that call that the port gets
wrong.

### The lattice starts at the first cell centre

`multilinear_grid` puts its values on lattice *points*:
value `(i, j)` is at `origin + (i, j) * spacing`.  A cell-centred field
has value `(i, j)` at the centre of cell `(i, j)`, so **the lattice's
origin is the first cell centre**, the mesh's corner plus half a cell
along every axis, and **its shape is the number of cells**.  The mesh's
corner as the origin is half a cell off, silently.  The field here is
linear, so both results can be compared with the exact value:

<!-- snippet: continues -->
```python
inside = jnp.asarray([[-0.52, 1.30], [-0.375, 0.75], [0.10, 2.00], [0.55, 1.10]],
                     jnp.float32)
exact = OFFSET + np.asarray(inside) @ GRADIENT.T

wrong = multilinear_grid_mapping(mode="consistent", layout="shaped",
                                 **{**lattice, "origin": CORNER})
right_error = np.asarray(gather.apply(VELOCITY0, None, inside)) - exact
wrong_error = np.asarray(wrong.apply(VELOCITY0, None, inside)) - exact
print(np.abs(right_error).max())        # 1.7e-07: float32 rounding
print(wrong_error[0])                   # [ 0.15  -0.025]: the gradient times half a cell
assert np.abs(right_error).max() < 1e-6
assert np.allclose(wrong_error, GRADIENT @ (0.5 * CELL), atol=1e-6)

# The node's own sampling and the edge's agree on the field as it starts.
assert np.allclose(sample_at_cell_centres(VELOCITY0, inside),
                   gather.apply(VELOCITY0, None, inside), atol=1e-6)

# The other half of the mistake, the number of cell FACES as the shape, is refused.
faces = GraphManager()
faces.add_node(Fluid("fluid", DT))
faces.add_node(Body("body", DT))
try:
    faces.add_edge("fluid", "body", "velocity", "fluid_velocity",
                   mapping=multilinear_grid_mapping(mode="consistent", layout="shaped",
                                                    **{**lattice, "shape": (NX + 1, NY + 1)}),
                   geometry=("target", "points"))
except ValueError as refusal:
    print(refusal)
else:
    raise AssertionError("a lattice of 9 x 7 points was accepted for 8 x 6 cells")
```

Outside the lattice's hull the kind clamps (`outside="clamp"`, the only
choice): a point is read at its projection onto the box of cell centres.
That box ends half a cell *inside* the mesh, so a point between the last
cell centre and the mesh's edge is already clamped, as it was in the
node.  The third of `POINTS0` starts in that strip.

### `mode` and `layout`

`mode="consistent"` interpolates a *value* at the points (a velocity, a
temperature): what the fluid's `velocity_at_points` was.
`mode="conservative"` is its transpose and spreads an *amount* held at
the points onto the grid (a force): the total arrives, and the work done
on the two sides is the same number.  Use it for the edge that returns
the body's force to the fluid, as in the closed loop below.

`layout` says how the node stores the field: `"shaped"` for
`(nx, ny)`, `"flat"` for the same values raveled in C order, `(nx * ny,)`.
Either may have one trailing axis of components, as the velocity here.

<!-- snippet: continues -->
```python
flat = multilinear_grid_mapping(mode="consistent", layout="flat", **lattice)
assert np.array_equal(flat.apply(VELOCITY0.reshape(NX * NY, 2), None, inside),
                      gather.apply(VELOCITY0, None, inside))

scatter = multilinear_grid_mapping(mode="conservative", layout="shaped", **lattice)
force = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5], [0.25, 0.25]], jnp.float32)
on_the_grid = scatter.apply(force, None, inside)
assert on_the_grid.shape == (NX, NY, 2)
assert np.allclose(on_the_grid.sum(axis=(0, 1)), force.sum(axis=0))       # the total
assert np.isclose(jnp.vdot(on_the_grid, VELOCITY0),                       # the work
                  jnp.vdot(force, gather.apply(VELOCITY0, None, inside)))
```

### Who holds the points, and when they are read

The geometry must be a state field of the edge's own source or target.
Here the body holds its points in its state, and the body is the target:
`geometry=("target", "points")`.

A target-anchored geometry is read from the state the target's `update`
receives: the points **before** the body moves them in this step.  The
value is the fluid's from this step, because the fluid runs first.  With
a moving body the three candidates are different numbers:

<!-- snippet: continues -->
```python
def history_of(gm, steps, names=("fluid", "body")):
    """The state before the first step and after each one; the graph is put back."""
    def snapshot():
        return {n: {k: np.asarray(v) for k, v in gm.get_node_state(n).items()} for n in names}
    out = [snapshot()]
    for _ in range(steps):
        gm.step()
        out.append(snapshot())
    gm.reset_state()
    return out


run = history_of(after, 10)
for n in (1, 2, 3):
    received = run[n]["body"]["seen"]
    field_after, field_before = run[n]["fluid"]["velocity"], run[n - 1]["fluid"]["velocity"]
    points_before, points_after = run[n - 1]["body"]["points"], run[n]["body"]["points"]
    # The field after the fluid's update of step n, at the points the body held before it.
    assert np.allclose(received, sample_at_cell_centres(field_after, points_before), atol=1e-6)
    # Not at the points the body holds after the step, and not the field before it.
    assert not np.allclose(received, sample_at_cell_centres(field_after, points_after), atol=1e-3)
    assert not np.allclose(received, sample_at_cell_centres(field_before, points_before), atol=1e-3)
```

That is what the old graph did *as it was built above*.  The old graph's
time level depended on which node was added first, because that decided
which of its two edges was the back edge; with the body added first, the
body received the field of the step before (and nothing at its first
step).  The new graph has one edge and no cycle, and the order the nodes
are added in changes nothing.  If your old graph was built the other way
round, the port changes its numbers by one step of the field, and that is
the port being right.  The other call sites (a flux hook, a coupling
group, a multi-rate graph) are in
[the time level a geometry is read at](geometry_dependent_mappings.md#the-time-level-a-geometry-is-read-at-in-short).

**Points held by a third node.**  If the points belong to neither end of
the edge (a rigid body that feeds a sampler), `add_edge` refuses to
anchor the geometry there.  In 0.4.0 the target keeps a copy in its own
state, written by its `update` from a plain edge, and the edge reads the
copy the next step: the positions are one step old.

<!-- snippet: continues -->
```python
class Carrier(SimulationNode):
    """A third node that owns the points and moves them."""

    def initial_state(self):
        return {"points": POINTS0}

    def update(self, state, boundary_inputs, dt):
        return {"points": state["points"] + dt * jnp.asarray([0.9, -0.3], jnp.float32)}


class Probe(SimulationNode):
    """Keeps a copy of the carrier's points, for the edge to read."""

    def initial_state(self):
        return {"points": POINTS0,            # start the copy where the carrier starts
                "seen": jnp.zeros((N_POINTS, 2), jnp.float32)}

    def boundary_input_spec(self):
        spec = BoundaryInputSpec(shape=(N_POINTS, 2), dtype=jnp.float32)
        return {"points_in": spec, "fluid_velocity": spec}

    def update(self, state, boundary_inputs, dt):
        return {"points": boundary_inputs.get("points_in", state["points"]),      # the copy
                "seen": boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["seen"]))}


third = GraphManager()
third.add_node(Fluid("fluid", DT))
third.add_node(Carrier("carrier", DT))
third.add_node(Probe("probe", DT))
try:
    third.add_edge("fluid", "probe", "velocity", "fluid_velocity",
                   mapping=gather, geometry=("carrier", "points"))
except ValueError as refusal:
    print(refusal)
else:
    raise AssertionError("a geometry held by a third node was accepted")
third.add_edge("carrier", "probe", "points", "points_in")
third.add_edge("fluid", "probe", "velocity", "fluid_velocity",
               mapping=gather, geometry=("target", "points"))
third.compile()

run3 = history_of(third, 4, names=("fluid", "carrier", "probe"))
for n in (1, 2, 3, 4):
    field, received = run3[n]["fluid"]["velocity"], run3[n]["probe"]["seen"]
    # At the end of a step the copy is the carrier's points ...
    assert np.array_equal(run3[n]["probe"]["points"], run3[n]["carrier"]["points"])
    # ... and the edge read the copy made one step earlier.
    assert np.allclose(received, sample_at_cell_centres(field, run3[n - 1]["carrier"]["points"]),
                       atol=1e-6)
    assert not np.allclose(received, sample_at_cell_centres(field, run3[n]["carrier"]["points"]),
                           atol=1e-3)
```

## They are the same graph

Ten steps carry the points across four to six cells along `x`.  The last
one is past the last cell centre after two steps and outside the mesh
after three, and is clamped from then on.

<!-- snippet: continues -->
```python
old, new = history_of(before, 10), run
last_centre, right_edge = CENTRES[-1, 0, 0], CORNER[0] + NX * CELL[0]
assert new[1]["body"]["points"][3, 0] < last_centre < new[2]["body"]["points"][3, 0]
assert new[3]["body"]["points"][3, 0] > right_edge

eps = np.finfo(np.float32).eps
worst = 0.0
for n in range(1, 11):
    assert np.array_equal(old[n]["fluid"]["velocity"], new[n]["fluid"]["velocity"])
    for field in ("seen", "points"):
        difference = np.abs(old[n]["body"][field] - new[n]["body"][field]).max()
        worst = max(worst, difference / (eps * np.abs(new[n]["body"][field]).max()))
print(f"the body's state differs by at most {worst:.2f} roundings")
assert worst <= 16
```

"A rounding" is `eps` times the largest magnitude of the field compared.
On this run the two graphs print `0.00`: they are identical to the bit.
That is not a rule.  The node divides by the cell size and then subtracts
a half, the kernel subtracts the first cell centre and then divides, so
a weight can differ in its last bit.  The test file measures the same
ten steps on a field that is not linear, in two and in three dimensions,
in float32 and in float64: at most 0.7 of a rounding, and it holds the
two graphs to 16.  On those runs a lattice with the mesh's corner as its
origin is 750,000 float32 roundings from the old graph.

Gradients agree as well.  `jax.grad` of the points' final `x` with
respect to the initial field and to the initial points:

<!-- snippet: continues -->
```python
def final_x(gm, velocity0, points0):
    # set_node_state replaces a node's whole state: hand back its other fields.
    gm.set_node_state("fluid", {**gm.get_node_state("fluid"), "velocity": velocity0})
    gm.set_node_state("body", {**gm.get_node_state("body"), "points": points0})
    return jnp.sum(gm.run_scan(n_steps=6)["body"]["points"][:, 0])


gradients = {}
for name, gm in (("before", before), ("after", after)):
    gradients[name] = jax.grad(final_x, argnums=(1, 2))(gm, VELOCITY0, POINTS0)
    gm.reset_state()            # the transform left tracers in the graph's state

for old_g, new_g in zip(gradients["before"], gradients["after"]):
    assert float(jnp.abs(new_g).max()) > 0.1
    assert np.allclose(old_g, new_g, rtol=1e-5, atol=1e-6)
print(np.round(np.asarray(gradients["after"][1]), 3))     # d final x / d initial points
```

In float64 the two gradients agree to `3e-16` of their size and with a
central difference to `2e-9` (measured in two and three dimensions; the
test above holds them to `1e-12` and `1e-7`).  Two things to know when
you compare gradients of your own port:

* **On a cell-centre plane both samplers take the slope of the cell
  above**, so the second point, which starts on a cell centre, has the
  same gradient in both graphs.  A central difference across such a
  plane is not a derivative.
* **Exactly on an outermost cell centre they differ.**  A node that
  clamps with `jnp.clip`, as the one above, gets half the interior slope
  there (`jnp.clip` splits the derivative at a tie); the kernel takes the
  whole interior slope.  Strictly outside, both are zero.

## What the edge gives that the node did not

### The transfer is verified on its own

`verify_mapping` holds an edge to what its kind claims
([Level 2: the edge](../developer_guide/verification.md#level-2-the-edge)).
Tell it where the two sides are: the cell centres, in the field's flat
order, and the points.

<!-- snippet: continues -->
```python
from maddening.testing.mapping import verify_mapping

centres = CENTRES.reshape(-1, 2)
claims = dict(geometry=np.asarray(inside), polynomial_order=1,
              source_coordinates=centres, target_coordinates=lambda points: points,
              hull=(centres[0], centres[-1]), outside="clamp",
              max_examples=20, derandomize=True)       # 20 draws keep this page quick
results = verify_mapping(after.edges[0], **claims)
print({name: result.status for name, result in results.items()})
assert all(result.passed for result in results.values())
for name in ("consistent", "adjoint", "geometry_derivative", "outside_hull", "round_trip"):
    assert results[name].status == "PASS"

# Told where the cell centres are, it finds the half-cell mistake.
assert verify_mapping(wrong, checks=["consistent"], **claims)["consistent"].failed
```

On this edge: `structure`, `linearity`, `consistent` (linear fields are
reproduced at the points), `adjoint`, `geometry_derivative` (the
derivative with respect to a position against a central difference),
`outside_hull` (the clamp), `dtype_float32`, `jit_consistent` and
`round_trip` pass.  `conservative` is a skip, because a gather does not
claim it, and `dtype_float64` is a skip without `jax_enable_x64`.

### The fluid's own battery needs no sample points

`verify_node` draws a node's state and its boundary inputs.  For the old
fluid it has to invent the body's points, and left to its default
envelope it draws them where there is no mesh, so the interpolation it
exercises is almost only the clamp.  The author of the fluid has to say
where another node's points make sense.  The new fluid has no such
input.

<!-- snippet: continues -->
```python
from maddening.testing.verification import verify_node

drawn = []


def record(boundary_inputs):
    drawn.append(np.asarray(boundary_inputs["sample_points"]))
    return boundary_inputs


kw = dict(bounds={"velocity": (-5.0, 5.0)}, max_examples=20, derandomize=True)
verify_node(SamplingFluid("fluid", DT), constrain_boundary=record, **kw)
points = np.concatenate(drawn)
between_centres = np.all((points > centres[0]) & (points < centres[-1]), axis=1)
print(f"{between_centres.sum()} of {len(points)} drawn points lie between the outer cell "
      f"centres; the largest coordinate drawn is {np.abs(points).max():g}")
assert between_centres.mean() < 0.1

old_battery = verify_node(SamplingFluid("fluid", DT),
                          boundary_bounds={"sample_points": (-2.0, 4.0)}, **kw)
new_battery = verify_node(Fluid("fluid", DT), **kw)
assert all(r.passed for r in old_battery.values()) and all(r.passed for r in new_battery.values())
```

The sampling itself is now checked where it lives, by the section above.

### In a coupling group, the convergence test still reads a few values

Close the loop: the body drags on the fluid where it is.  The force
returns through a second edge, `mode="conservative"`, anchored at its
source (the body holds the points it pushes at).

<!-- snippet: continues -->
```python
RESPONSE, DRAG, MASS = 0.5, 1.5, 4.0


class PushedFluid(SimulationNode):
    """The fluid of the closed loop: it takes a force on its cells."""

    def initial_state(self):
        return {"velocity": VELOCITY0}

    def boundary_input_spec(self):
        return {"body_force": BoundaryInputSpec(shape=(NX, NY, 2), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        force = boundary_inputs.get("body_force", jnp.zeros_like(state["velocity"]))
        return {"velocity": advance(state["velocity"], dt) + RESPONSE * force}


class DraggedBody(SimulationNode):
    """A body with a velocity of its own, dragged by the fluid at its points."""

    def initial_state(self):
        zero = jnp.zeros((N_POINTS, 2), jnp.float32)
        return {"points": POINTS0, "velocity": zero, "force": zero}

    def boundary_input_spec(self):
        return {"fluid_velocity": BoundaryInputSpec(shape=(N_POINTS, 2), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        seen = boundary_inputs.get("fluid_velocity", jnp.zeros_like(state["velocity"]))
        force = DRAG * (state["velocity"] - seen)          # on the fluid
        velocity = state["velocity"] - dt * force / MASS
        return {"points": state["points"] + dt * velocity, "velocity": velocity, "force": force}


loop = GraphManager()
loop.add_node(PushedFluid("fluid", DT))
loop.add_node(DraggedBody("body", DT))
loop.add_edge("fluid", "body", "velocity", "fluid_velocity",
              mapping=gather, geometry=("target", "points"))
loop.add_edge("body", "fluid", "force", "body_force",
              mapping=scatter, geometry=("source", "points"))
loop.add_coupling_group(["fluid", "body"], max_iterations=100,
                        convergence_norm="interface", rtol=1e-4)
loop.compile()
loop.step()
report = loop.coupling_diagnostics()["body+fluid"]
print(int(report["iterations"]), bool(report["converged"]))
assert report["converged"]
```

`convergence_norm="interface"` judges the group on what crosses its
internal edges, and reads a mapped edge on its compact side: the few
velocities the body receives, the few forces it returns and the points
they are spread at, whatever the size of the grid.  What it reads on each
kind of edge, and what `coupling_diagnostics()` reports for such a group,
is in [Geometry-dependent mappings](geometry_dependent_mappings.md)
("What `convergence_norm="interface"` reads on a geometry edge" and
"Limits in 0.4.0").

**Measured**, on a loop of this kind at three grid sizes (a unit square
of 16, 64 and 256 cells a side, four points, a loop that contracts by
0.63 a pass, float64, `rtol=1e-4`, the default Gauss-Seidel sweep).  The
number is how far the velocity the body received at a `converged=True`
step is from a solve seven decades tighter, in tolerances (RMS over
`rtol` times the largest value); the passes taken are in brackets.

| What crosses the group's internal edges | 16 x 16 | 64 x 64 | 256 x 256 |
|---|---|---|---|
| the sampling and the spreading in the fluid node; plain edges carry the points, the force and the velocity at the points | 0.13 (22) | 0.13 (22) | 0.18 (22) |
| the sampling and the spreading on the edges, as above | 0.13 (22) | 0.13 (22) | 0.14 (22) |
| the grid handed whole to the body, which samples it itself, on a plain edge | 0.83 (18) | 3.4 (15) | 13.8 (12) |

Read it this way:

* **The port keeps what the old wiring had.**  A fluid that samples for
  the body already sent point-sized fields across its plain edges, and
  the criterion read those.  Moving the sampling onto the edge does not
  change how close a converged step is: a seventh of a tolerance at
  every size, in the same number of passes.
* **The wiring to avoid is the third.**  A plain edge that carries the
  whole grid puts every cell into the criterion, of which the few around
  the points move.  The criterion loosens as the grid grows, the loop
  stops earlier, and the body's value is 14 tolerances off at 256 x 256
  cells beside `converged=True`.  If a node of yours receives a grid and
  samples it, this port is how to stop: the mapped edge is read at the
  points.

**Inside a group the port moves the sampling by one step of the body's
motion.**  A plain edge that carries the points reads the *iterate*, so
the old fluid sampled at the positions of the pass in hand: the body's
end-of-step positions, once converged.  The target-anchored edge reads
the positions the body held before the step, at every pass.  The two
loops therefore converge to different fixed points, first order in the
step.  On the 16 x 16 loop of the table the velocities the body receives
at the two fixed points differ by `2.2e-4` of their size (the body moves
a thousandth of a cell in the step), and by ten times that at ten times
the step.  The table compares each wiring with its own tight solve.

## A held body: the same transfer as a static mapping

Points that never move need no geometry.  The same weights, four entries
a row, are a static sparse mapping
([Sparse mappings](../algorithm_guide/coupling/interface_mapping.md#sparse-mappings)).
It reads its source along the field's *first* axis, so the field must be
stored flat, `(nx * ny, 2)`; `layout="flat"` is the geometry edge on the
same field.

<!-- snippet: continues -->
```python
from maddening.core.coupling.sparse_mapping import sparse_matrix_mapping


class FlatFluid(SimulationNode):
    """The same fluid with its field stored raveled, (NX * NY, 2)."""

    def initial_state(self):
        return {"velocity": VELOCITY0.reshape(NX * NY, 2)}

    def update(self, state, boundary_inputs, dt):
        shaped = state["velocity"].reshape(NX, NY, 2)
        return {"velocity": advance(shaped, dt).reshape(NX * NY, 2)}


class HeldBody(Body):
    """Points that never move."""

    def update(self, state, boundary_inputs, dt):
        return {**super().update(state, boundary_inputs, dt), "points": state["points"]}


# The node's own stencil at the held points, as rows: flat index = i * NY + j.
rows = list(cell_centre_stencil(POINTS0))
indices = np.stack([np.asarray(i) * NY + np.asarray(j) for (i, j), _ in rows], axis=1)
values = np.stack([np.asarray(weight) for _, weight in rows], axis=1)
static = sparse_matrix_mapping(indices, values, n_source=NX * NY)
assert indices.shape == values.shape == (N_POINTS, 4)


def held(fluid, **edge):
    gm = GraphManager()
    gm.add_node(fluid("fluid", DT))
    gm.add_node(HeldBody("body", DT))
    gm.add_edge("fluid", "body", "velocity", "fluid_velocity", **edge)
    gm.compile()
    return gm


with_geometry = held(FlatFluid, mapping=flat, geometry=("target", "points"))
with_static = held(FlatFluid, mapping=static)
a, b = history_of(with_geometry, 5), history_of(with_static, 5)
for n in range(1, 6):
    assert np.allclose(a[n]["body"]["seen"], b[n]["body"]["seen"], rtol=0, atol=2e-6)

# The field stored with the grid's shape is refused.
try:
    held(Fluid, mapping=static)
except ValueError as refusal:
    print(refusal)
else:
    raise AssertionError("a static sparse mapping accepted a field of shape (NX, NY, 2)")

# The static mapping's weights are parameters of the graph; a geometry edge has none.
key = "fluid.velocity->body.fluid_velocity"
assert with_static.params["mappings"][key]["W"].shape == (N_POINTS, 4)
assert with_geometry.params["mappings"][key] == {}

# An adaptive run takes the static mapping and refuses the geometry edge.
with_static.run_adaptive(t_end=0.2)
try:
    with_geometry.run_adaptive(t_end=0.2)
except RuntimeError as refusal:
    print(refusal)
else:
    raise AssertionError("run_adaptive accepted a graph with a geometry edge")
```

A static mapping is an ordinary mapped edge.  Its weights are an entry
of `gm.params["mappings"]`, which the sparse-mappings section linked
above describes, and the adaptive steppers take it.  What else a geometry
edge cannot do in this release is listed in
[Limits in 0.4.0](geometry_dependent_mappings.md#limits-in-040): read
that list before choosing a geometry edge for points that will never
move.

## In three dimensions

Nothing changes in kind.  `origin`, `spacing` and `shape` have three
entries, the origin is the mesh's corner plus half a cell along each of
the three axes, the geometry is `(n_points, 3)`, each point reads eight
corners, and a field stored `(nx, ny, nz)` or `(nx, ny, nz, C)` is
`layout="shaped"`.  A held body's static mapping has eight entries a
row, and its flat index is `(i * ny + j) * nz + k`.  The test file runs
this page's comparisons on a mesh of 8 x 5 x 4 cells with a
three-component velocity.

## What stays in the node

The port is for a transfer that is *linear in the field and local to a
cell*: an interpolation at points, and its transpose.  Two things that
often sit beside it are not this port:

* **A boundary treatment that is not a linear transfer** (bounce-back on
  a lattice, a penalty immersed boundary, a contact law) stays in the
  node.  An edge carries a field; it does not decide what the solver does
  with it.
* **A kernel that 0.4.0 does not ship** (a Gaussian or a Peskin spread
  over several cells) is a mapping kind of your own: register it with
  `register_mapping(..., needs_geometry=True)`
  ([Registering your own mapping kind](../algorithm_guide/coupling/interface_mapping.md#registering-your-own-mapping-kind)).
  Such a kind is stepped, and iterated in a coupling group, like the
  shipped one.  In 0.4.0 it gets less than `multilinear_grid` does there:
  `convergence_norm="interface"` on a group that resolves it is refused
  at `compile()`, and `coupling_diagnostics()` reports its solve and no
  bound (a Gaussian gather in the loop above: converged under `"l2"`,
  refused under `"interface"`).  See
  [What is refused](geometry_dependent_mappings.md#what-is-refused) and
  the [limits](geometry_dependent_mappings.md#limits-in-040).
