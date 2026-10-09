# Geometry-dependent mappings

*Experimental in 0.4.0.  The keyword `geometry=` of `add_edge`, the
`multilinear_grid` kind and the registry flag `needs_geometry` may change
in a minor release.*

An interface mapping transfers a field from the points of one node to the
points of another.  The mappings of
[Interface mapping](../algorithm_guide/coupling/interface_mapping.md) are
built once, from positions that never change.  A **geometry-dependent**
mapping reads the positions from the graph's state at every step, so the
points may move: markers carried by a flow, a body that crosses a grid.

You say where the positions are with `geometry=(anchor, field)` on the
edge: `anchor` is `"source"` or `"target"`, the edge's own source or
target node, and `field` is a state field of that node.

## An example

A row of markers drifts across a one-dimensional grid.  One edge samples
the grid's field at the markers (a *gather*, `mode="consistent"`); the
other deposits the markers' values back on the grid (a *scatter*,
`mode="conservative"`).  Both read the markers' `pos`: it is the target of
the first edge and the source of the second.

```python
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

N_GRID, N_MARKERS = 6, 4


class Grid(SimulationNode):
    """A field on the lattice 0, 0.5, .., 2.5 that decays and takes deposits."""

    def initial_state(self):
        return {"x": jnp.linspace(1.0, 2.0, N_GRID, dtype=jnp.float32)}

    def boundary_input_spec(self):
        return {"deposit": BoundaryInputSpec(shape=(N_GRID,), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        deposit = boundary_inputs.get("deposit", jnp.zeros_like(state["x"]))
        return {"x": 0.5 * state["x"] + 0.1 * deposit}


class Markers(SimulationNode):
    """Points that drift to the right and remember what they sampled."""

    def initial_state(self):
        pos = 0.3 + 0.4 * np.arange(N_MARKERS)
        return {"x": jnp.zeros(N_MARKERS, jnp.float32),
                "pos": jnp.asarray(pos.reshape(N_MARKERS, 1), jnp.float32)}

    def boundary_input_spec(self):
        return {"sampled": BoundaryInputSpec(shape=(N_MARKERS,), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        sampled = boundary_inputs.get("sampled", jnp.zeros_like(state["x"]))
        return {"x": 0.5 * state["x"] + sampled, "pos": state["pos"] + dt * 0.5}


grid = dict(origin=[0.0], spacing=[0.5], shape=[N_GRID], n_points=N_MARKERS)
gm = GraphManager()
gm.add_node(Grid("grid", 0.01))
gm.add_node(Markers("markers", 0.01))
gm.add_edge("grid", "markers", "x", "sampled",
            mapping=multilinear_grid_mapping(mode="consistent", **grid),
            geometry=("target", "pos"))
gm.add_edge("markers", "grid", "x", "deposit",
            mapping=multilinear_grid_mapping(mode="conservative", **grid),
            geometry=("source", "pos"))
gm.compile()

gm.step()
# The grid runs first and halves its field: 0.5 at the lattice point 0.0 and
# 0.6 at 0.5.  The first marker is at 0.3 when it samples, 3/5 of the way
# between them: 0.4 * 0.5 + 0.6 * 0.6.
sampled = np.asarray(gm.get_node_state("markers")["x"])
assert np.isclose(sampled[0], 0.56)
assert np.isclose(float(gm.get_node_state("markers")["pos"][0, 0]), 0.305)
print(gm.format_graph())    # each edge's text names its geometry
```

The geometry is ordinary node state.  It is integrated by its node, saved
by `save_state` and restored by `load_state`, batched by `run_sweep`, and
`jax.grad` differentiates through it.  The mapping itself has no weights:
its entry of `gm.params["mappings"]` is `{}`.  `to_dict` / `from_dict`
and a USD stage carry the edge's `geometry`, and an FMU of such a graph
steps as the graph does.

## What is refused

`add_edge` raises `ValueError`; `validate()` returns the others as
`ERROR:` issues, which `compile()` raises as `RuntimeError`.  Each message
names the edge and says what to do.

| Where | Refused |
|---|---|
| `add_edge` | `geometry=` without `mapping=`, or with a static mapping (it would be ignored) |
| `add_edge` | a geometry-dependent mapping without `geometry=` |
| `add_edge` | anything but `("source", <field>)` or `("target", <field>)`: a geometry held by a third node is not supported |
| `compile` | a field the anchor node's state does not hold, or one of its boundary fluxes |
| `compile` | a field that is not float32 or float64, or whose shape is not the mapping's `geometry_shape` |
| `compile` | a `multilinear_grid` whose grid the geometry's dtype cannot resolve to 1/16 of a cell (a warning from `validate()` at 1/1024) |
| `compile` | an edge with a sharded node at either end |
| `compile` | `convergence_norm="interface"` on a coupling group with a geometry-dependent mapping on an internal edge |
| `run_adaptive`, `run_adaptive_scan` | any graph with a geometry edge |
| `replace_node`, `POST /surrogate/activate`, `POST /surrogate/deactivate` | a replacement that does not hold the geometry field with the same shape and a float32 or float64 dtype; nothing is changed (the REST routes answer 409) |
| `DatasetGenerator` | a target node fed through a geometry edge |
| the `multilinear_grid` kind | a non-floating field, when the step is traced |
| any entry point that traces a step (`step`, `run`, `run_scan`, `resolve_boundary_inputs`, ...) | a geometry whose dtype a state write made after `compile()` (`set_node_state`), or a node's own `update`, changed to one `compile()` refuses: not float32 or float64, or too coarse for the grid.  A program is traced again when a dtype changes, and the rule is asked then |

## Limits in 0.4.0

* **Coupling diagnostics read a geometry in one case.**  The step, the
  coupling passes and their gradients read the geometry everywhere.
  `coupling_diagnostics()` reports the bounds of a group that resolves a
  geometry-dependent mapping (`rho_spectral`, `spectral_error_bound`,
  `gradient_relative_error_bound`, the estimates and the `*_usable`
  flags, as for any other group) when all of these hold:
  every such mapping is a `multilinear_grid`; the group's
  `convergence_norm` is `"l2"` or `"mixed"`; the group does not
  sub-cycle.  With `diagnostics=True` the step then also checks itself:
  it compares its Jacobian-vector product along the positions with a
  finite difference of the pass along the same direction, and where the
  two differ by more than a quarter (`GEOMETRY_GAP_TOLERANCE`) the
  report withholds the bounds.  An honest pass reads the finite
  difference's own error: about 6e-4 in float32 (at most 6e-3 on 235
  drawn examples of six cells, and at most 1e-2 on three thousand more
  away from a lattice plane) and under 5e-6 in float64.  The check fails
  for a node or a mapping whose derivative is not the derivative of its
  value (a `stop_gradient` on an input, a rounding, a branch on a
  position): a pass whose product saw no geometry at all read 0.31 to
  0.98 in float32 on those draws and 1.0 in float64.  It is a coarse
  check, and it has three measured limits:

  - **A wrong derivative of a weak term is not seen.**  A term the
    product misses reads `G / (G + 32 res)`, where the term moves a field
    by `G` and `res` is the field's float resolution: 0.22, under the
    tolerance, on a deposit so weak that it moved the grid's field by
    nine resolutions.  The spectral radius reported there was 0.003
    against 0.025.
  - **An honest report can be withheld beside a lattice plane.**
    Where a member reads, in the same sweep, positions that another
    member has just built, and one of them is within the check's own
    step of a lattice plane -- `sqrt(eps)` of a spacing: 3.45e-4 in
    float32, 1.5e-8 in float64 -- the check's step carries it across
    the plane and the difference is not a derivative.  Of 2829 drawn
    float32 examples of two such Gauss-Seidel groups (1489 within 2e-5
    of a spacing of a plane), 47 read over 0.05, two over 0.2 and one
    0.59; of a thousand further than 2e-5 from a plane none read over
    5e-3.  An independent audit that placed fixed points at every
    distance from a plane had 180 honest reports withheld in 65,714,
    each with a position the pass reads within that step of a plane (up
    to 2.9e-4 of a spacing in float32, 6.8e-9 in float64): 45 in
    float32 and 135 in float64, two of them weakly coupled Jacobi groups
    (MADD-ANO-246).
  - **An honest float32 report is withheld behind a strongly cancelling
    gather.**  A Gauss-Seidel group whose gather samples a field that
    changes sign across a cell reads 0.25 to 0.75 once the lattice
    values are about a thousand times the sample (one band at 0.14
    passes).  The bound is too low there anyway (MADD-ANO-212).

  `not_usable_reason` therefore names both readings of a gap over the
  tolerance, a derivative that is not the value's and a finite
  difference that could not be formed: one number does not tell them
  apart.  A step where the two could not be compared at all (a state
  that is not finite, or a geometry the pass reads from a constant the
  step could not move) has a reason of its own.  The tolerance is not
  lower because of the second limit: at 0.05 the check would catch the
  weak deposit and would withhold 7 to 9 of the 4,450 honest examples of
  the test suite's searches, where it now withholds none or one.
* **Beside a lattice plane the flags are withdrawn and the numbers
  kept.**  What it costs in ordinary use first, measured on the example
  above with `add_coupling_group(["grid", "markers"], diagnostics=True)`
  over 300 steps (honest reports, their numbers right against a float64
  reference, that lose `spectral_usable` and `gradient_bound_usable`;
  the two go together):

  | the group | honest reports that lose both flags |
  |---|---|
  | the example as written (markers advected across the lattice) | about 4%: 8 of 204 float32 reports at tolerances of 1e-4 and 1e-3 and 15 (7%) at 1e-2; 11 of 300 float64 reports at 1e-9, 1e-6 and 1e-3.  The same with the markers swept first and under the mixed norm |
  | markers at rest inside cells, or outside the hull | none |
  | a marker at rest exactly on a lattice point or on a face of the hull, its position read from the iterate | **every step** |
  | every position a constant of the pass (both edges target-anchored), or fed back from the sampled value | none |
  | both edges source-anchored | 3 of 68 (float32), 1 of 51 (float64) |
  | the lattice moved to an origin of 20 | 13 of 83 (16%, float32), 11 of 300 (float64) |

  So a body resting exactly on lattice planes, whose positions the group
  solves, **has neither flag**, on any step and at any tolerance: the
  rule cannot know that it is at rest.  Two ways out were measured, in
  the table: hold the positions constant during the pass (anchor the
  geometry at a target that `update` reads, or keep the body outside
  the group), or place it a fraction of a cell off the planes.  For a
  marker that is merely *near* a plane, a tighter tolerance clears it.
  A search aimed at planes shows the worst of it: on an audit's scan
  that places every fixed point beside a plane at tolerances of 1e-5 to
  0.1, 75% of the honest reports that had the gradient's flag lost it,
  45% on pairs whose positions the pass reads from the iterate and that
  do not move, and none where every position is a constant of the pass.
  And none of this matters to a group with more than eight independent
  interface scalars (five markers in the example, whose positions and
  values both move with the iterate): such a group has no flag to lose,
  with or without a geometry, because the spectral estimate takes eight
  Krylov steps and does not settle.

  *Why.*  A multilinear stencil is one polynomial of the positions
  inside a lattice cell and another in the next, so the pass's Jacobian
  jumps where a position crosses a lattice plane, or a face of the
  grid's hull (outside it the kernel clamps).  `rho_spectral`,
  `spectral_error_bound` and `gradient_relative_error_bound` are the
  linearisation at the returned iterate: they describe the polynomial
  the pass is in the lattice cells its positions are in *there*, and
  `spectral_error_bound` is the distance to *that polynomial's* fixed
  point.  If it lies past a lattice plane, the pass has no fixed point
  there.  Measured: the returned iterate and the Newton point both a
  little short of a plane, the polynomial's fixed point two millionths
  of a spacing past it, the next cell expanding, and the pass's fixed
  point two cells on, 17 to 1,294 bounds away, with `spectral_usable`
  set; and a gradient bound 15 to 70,000 times under the true error with
  its flag set (MADD-ANO-248).  The Newton-Kantorovich check behind the
  gradient bound cannot see this: it takes the Jacobian at the iterate
  and at the Newton point, and both were one cell's.
* **Both flags need no lattice plane inside the Newton-Kantorovich ball
  around the returned iterate.**  They stand only where every position
  the pass reads from the iterate is further from its nearest lattice
  plane than the fixed point of its cell's polynomial can be from it:
  the Newton step's own move of that position, plus one more Newton step
  and the float floor, in the group's norm.  The iterate, the Newton
  point and that fixed point are then in one lattice cell, where the
  pass is one polynomial, so the fixed point is the pass's and the
  bounds are the smooth ones.  The step stores the margin
  (`geometry_plane_margin`: how many radii of that ball the nearest
  plane is away) and the flags need it over one; a margin that is
  absent or not a number counts as none.  The rule does not look past
  the plane, so it also withdraws the flags where the next cell would
  have kept the bound.  **This withdraws honest reports**, and is meant
  to: a wrong number with its flag set is the worse outcome.  The
  positions concerned are a member's source-anchored geometry and the
  target-anchored geometry of a member that computes fluxes; a position
  that is a constant of the pass (a target-anchored geometry read by
  `update`, a node outside the group) does not move between the iterate
  and the fixed point and withdraws nothing.
* **And, with a plane within twice `spectral_error_bound`, a
  Newton-Kantorovich check that passed** (`gradient_relative_error_bound`
  finite).  The ball's argument assumes that check's condition for the
  cell's polynomial; this is where the step measures it.  Measured on a
  marker whose fixed point was 2e-4 of a spacing past a plane with the
  Newton point across it: a radius of 0.28 before the plane and 0.97
  after it, and a bound 0.13 times the true distance on a converged
  solve (MADD-ANO-242).
* **`not_usable_reason` gives every cause of a `False` flag** of such a
  group whose step computed the estimate: a plane in the ball (with the
  margin), a position on a plane, a plane in reach without a passed
  check (a gradient bound that was not computed, NaN, told from one that
  did not certify, `inf`), the float floor (the example in float32 at
  the default tolerance of 1e-6 is precision-limited on every step, and
  has no flag for that reason), an estimate that did not settle.  It
  names a plane only where a plane rule is a cause, and suggests a
  tighter tolerance only where one can help.
* **A position within eight float resolutions of a lattice plane is on
  it**, for both rules: its distance is zero.  The resolution is `eps`
  of the position's dtype times the largest coordinate of the lattice's
  axis, or the position's own magnitude where that is larger (in
  float32 on a lattice out to 20 with a spacing of 0.25: 9.5e-6 of a
  spacing, so the window is 7.6e-5 of one).  The kernel decides the
  cell from a rounded quotient and a member builds a position with a
  rounding of its own, so nearer than that the cell the floats
  evaluated is not the value's to say: an iterate 4.7 resolutions before
  a plane with its fixed point 0.7 past it kept both flags on a gradient
  bound 25 times under the error.  Both flags are withdrawn there, so
  a `rho_spectral` that is the radius of the cell the float pass
  evaluated rather than exact arithmetic's (13 to 22% of `1 - rho` apart
  in seven audited float32 Gauss-Seidel examples; MADD-ANO-239) is no
  longer flagged through a geometry.  On a plane the Newton-Kantorovich
  check's own outcome is rounding's: for a marker at rest at coordinate
  0.0 on the lower face of a lattice whose origin is 0.0,
  `gradient_relative_error_bound` reads `inf` on some steps and a number
  on others of the same resting state (the Newton point lands 1e-20
  inside or outside the hull).  The flags and the reason do not follow
  it.
* **Everywhere else the report says so.**  For any other group that
  resolves a geometry-dependent mapping (another mapping kind, a
  sub-cycled group, the interface norm with a geometry edge entering
  from outside), `coupling_diagnostics()` reports `iterations`,
  `total_iterations`, `residual` and `converged`, and no bound or
  estimate.  Every bound is NaN, every `*_usable` flag is `False`, and
  the entry's `not_usable_reason` says which case it is.
  `coupling_report()` and `print_coupling_report()` print that reason.
  The internal `_meta` entry of the state (which `GET /graph/state`
  returns) still holds what the step computed for such a group; it is
  not a report and promises nothing.
* **No interface norm** for a group with a geometry-dependent mapping on
  an internal edge: use `convergence_norm="l2"` or `"mixed"`.
* **No adaptive stepping** and **no sharded nodes** on a geometry edge.
* **The geometry is a state field of the edge's own source or target.**
  To use positions another node holds, carry them in the source's or the
  target's state.
* **A geometry is interpolated componentwise** for a sub-cycled member of
  a coupling group.  If its components are not affine coordinates (a
  quaternion, a wrapped angle), set `boundary_interpolation="constant"`.
* `POST /graph/edges` cannot create a mapped edge, so it cannot create one
  with a geometry: a body with a `mapping` or a `geometry` key is refused
  (422, the key named) and no edge is added; `GET /graph` shows the key.
* One kind ships, `multilinear_grid`.  Your own kind registers with
  `register_mapping(..., needs_geometry=True)` and declares
  `needs_geometry = True` and `geometry_shape`; see the `Mapping`
  protocol's docstring for the optional attributes.

## The time level a geometry is read at, in short

| Call site | A target-anchored geometry is |
|---|---|
| The target's `update` | the `g` of the state `update` receives |
| The target's flux hook (`compute_boundary_fluxes`) | the post-update `g` |
| Multi-rate, the flux hook of a node slower than the base step | the `g` the node holds after the base step: post-update when it fires, the held `g` when it does not |
| A slow source's value and source-anchored geometry | what the source holds between its steps |

The full table (coupling passes, sub-cycled members, back edges), and the
kernel's mathematics, are in the algorithm guide:
[Geometry-dependent mappings](../algorithm_guide/coupling/interface_mapping.md#geometry-dependent-mappings).
