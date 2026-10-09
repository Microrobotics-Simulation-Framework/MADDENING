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
| `compile` | `convergence_norm="interface"` on a coupling group with a geometry-dependent mapping on an internal edge, where the mapping is not a `multilinear_grid` or the group sub-cycles |
| `run_adaptive`, `run_adaptive_scan` | any graph with a geometry edge |
| `replace_node`, `POST /surrogate/activate`, `POST /surrogate/deactivate` | a replacement that does not hold the geometry field with the same shape and a float32 or float64 dtype; nothing is changed (the REST routes answer 409) |
| `DatasetGenerator` | a target node fed through a geometry edge |
| the `multilinear_grid` kind | a non-floating field, when the step is traced |
| any entry point that traces a step (`step`, `run`, `run_scan`, `resolve_boundary_inputs`, ...) | a geometry whose dtype or shape a state write made after `compile()` (`set_node_state`), or a node's own `update`, changed to one `compile()` refuses: not float32 or float64, too coarse for the grid, or not of the shape the mapping reads.  A program is traced again when a dtype or a shape changes, and the rules are asked then.  The warning at 1/1024 of a cell is not: `validate()` and `compile()` give it, of the state they are called with |

## Limits in 0.4.0

* **A group that solves the positions of a geometry-dependent mapping
  has no usable flag.**  Where a coupling group's pass reads a position
  from the iterate, or builds one and reads it in the same pass,
  `spectral_usable` and `gradient_bound_usable` are `False` on every
  step, whatever the numbers read.  The numbers (`rho_spectral`,
  `spectral_error_bound`, `gradient_relative_error_bound`) are reported
  as computed, uncertified, and `not_usable_reason` names the positions
  and says this.  **Two ways to keep the flags**, both of which fix every
  position during the pass:

  - anchor the geometry at a target that `update` reads
    (`geometry=("target", field)`: the member's pre-step positions), on
    every geometry edge of the group;
  - keep the node that holds the positions outside the group (it then
    moves between the group's solves, not inside them).

  The positions a pass solves are a member's source-anchored geometry
  (`geometry=("source", field)` on an edge inside the group: in the
  example above, the scatter's) and the target-anchored geometry of a
  member that computes fluxes.  No tolerance and no `convergence_norm`
  restores the flags of such a group in 0.4.0, and a group with more
  than eight independent interface scalars has no flag with or without
  a geometry (the spectral estimate takes eight Krylov steps).

  *Why.*  A multilinear stencil is one polynomial of the positions
  inside a lattice cell and another in the next, so the pass's Jacobian
  jumps where a position crosses a lattice plane, or a face of the
  grid's hull (outside it the kernel clamps).  The three numbers are the
  linearisation at the returned iterate: they describe the polynomial
  the pass is in the lattice cells its positions are in *there*, and
  `spectral_error_bound` is the distance to *that polynomial's* fixed
  point, which is the pass's only if it lies in those cells.  Three
  rules in turn tried to certify that it does, and an independent audit
  of each found `spectral_usable` set beside a wrong number near a
  lattice plane: a bound 0.13 times the true distance (MADD-ANO-242);
  17 to 1,294 times under it, and a gradient bound 15 to 70,000 times
  under the true error; and last, 34 to 1,874 times under it on a
  `converged` solve in a cell whose polynomial has no fixed point at
  all (MADD-ANO-252).  A sharper rule is left to a later release; until
  then the numbers of such a group are an estimate and not a
  certificate.  Away from lattice planes they are usually right: an
  audit of the example above over 300 steps found no wrong number among
  the reports the last rule had flagged, against a float64 reference.
* **For a group that solves positions, prefer
  `convergence_norm="interface"`**, which measures a position in lattice
  spacings.  Under `"l2"` and `"mixed"` a position field is scaled by
  its own largest magnitude, so a marker near the coordinate origin is
  weighted enormously and what the solve calls converged depends on
  where the origin is: with one marker 5e-4 from coordinate zero (its
  position weighted by 2,008) `rho_spectral` read 0.245 for 0.153 in
  float32.
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
* **How far the lattice planes are is still stored, as two numbers.**
  For a group that solves positions the step writes, in the state's
  internal `_meta` entry, `geometry_plane_limit` (the largest
  `spectral_error_bound` at which no position the pass reads from the
  iterate is within twice the bound of a lattice plane or of a face of
  the hull) and `geometry_plane_margin` (how many radii of the
  Newton-Kantorovich ball around the returned iterate the nearest plane
  is away: the Newton step's own move of the position, plus one more
  Newton step and the float floor, in the group's norm).  A position
  within eight float resolutions of a plane counts as on it (the kernel
  decides the cell from a rounded quotient), and a marker at rest on a
  lattice point or a hull face has a margin of zero on every step.  Both
  read `inf` for a group whose positions are all fixed during the pass.
  **No flag stands on either in 0.4.0**: that a margin over one puts the
  fixed point in the iterate's cell is proved only where the cell's
  polynomial satisfies the Newton-Kantorovich condition, which the step
  does not prove (the algorithm guide has the argument and the case
  that defeats it).
* **`not_usable_reason` of a group whose positions are fixed during the
  pass gives every cause of a `False` flag** of a step that computed the
  estimate: the float floor, an estimate that did not settle, a gradient
  bound that was not computed (NaN) told from one that did not certify
  (`inf`).  It never names a lattice plane: no plane can come between
  the iterate and the fixed point of such a pass.
* **Everywhere else the report says so.**  For any other group that
  resolves a geometry-dependent mapping (another mapping kind, a
  sub-cycled group, the interface norm), `coupling_diagnostics()` reports `iterations`,
  `total_iterations`, `residual` and `converged`, and no bound or
  estimate.  Every bound is NaN, every `*_usable` flag is `False`, and
  the entry's `not_usable_reason` says which case it is.
  `coupling_report()` and `print_coupling_report()` print that reason.
  The internal `_meta` entry of the state (which `GET /graph/state`
  returns) still holds what the step computed for such a group; it is
  not a report and promises nothing.
* **The interface norm solves, and reports no bound yet.**  See the next
  section for what it reads.  A group under `convergence_norm="interface"`
  with a geometry-dependent mapping reports its solve and withholds the
  bounds, as above; with another mapping kind on an internal edge, or in a
  sub-cycled group, it is refused at `compile()`: use `"l2"` or `"mixed"`.
* **Under the interface norm, positions need a dtype that resolves the
  tolerance where they are.**  A position `u` spacings from zero is stored
  to `eps * u` spacings (`eps` is 1.2e-7 in float32 and 2.2e-16 in
  float64).  That rounding enters what the criterion reads in two ways:
  - **a scatter anchored at its source** reads those positions themselves
    and asks them to change by less than `rtol` of one spacing;
  - **a gather** (and any mapping read as delivered, at either anchor)
    reads a value interpolated at those positions.  A position moved by
    `eps * u` spacings moves an interpolation weight by as much, and the
    delivered value by up to that fraction of its own size (the worst
    case: a field that varies by its own size across one cell), of which
    the criterion asks a change below `rtol`.

  The float floor of the residual counts four such roundings per
  evaluation of a pass for every entry of either reading
  (`PRECISION_FLOOR_ULPS`; `E` evaluations a pass: one where every node
  reads the previous iterate and evaluates once, two for a Gauss-Seidel
  pair), and `compile()` warns (`UserWarning`) of each edge whose
  positions put that count at the tolerance or above:

  ```text
  4 * E * eps * max|u| >= rtol
  ```

  with `u` taken axis by axis in that axis's spacing and `eps` that of the
  dtype the positions are stored in, on the state `compile()` is called
  with.  In float32 at `rtol=1e-4` that is from 210 spacings from zero
  (105 for a Gauss-Seidel pair); at the default `rtol=1e-6`, from two.
  From there on the rounding counted for the positions is the tolerance
  asked of the reading, or more: rounding alone can keep the group from
  converging (it then runs to `max_iterations`), and where it does
  converge the criterion says little of those positions, or of the last
  digits of that value.  It is a warning, and the step is built as it
  would be without it: where the positions settle to the last bit the
  group converges as before, and a gathered field that varies little
  across a cell is moved by less than the count allows for.  The message
  names the edge, the node and field that store the positions, which of
  the two readings it is, the distance and the dtype, and three remedies:
  - hold the positions in float64.  This needs `jax_enable_x64`; the
    mapping computes its weights in the geometry's dtype and casts them to
    the field's, so the other fields can stay float32;
  - use coordinates local to the grid, so that the positions are small
    numbers (the origin of the coordinates near the markers);
  - loosen `rtol`.

  Positions written after `compile()` (`set_node_state`) are not asked
  again until the next `compile()`.  A gather anchored at its target is
  asked at the target's positions in that state (what a step started from
  it reads).  A scatter anchored at its target is not asked: its reading
  is its source value alone, and the float floor counts no position for
  it.  Neither are the `"l2"` and `"mixed"` norms, which measure positions
  against their own size and read no delivered value.
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

## What `convergence_norm="interface"` reads on a geometry edge

The interface norm judges a group on what crosses its internal edges, and
reads a mapped edge on its compact side (see
[Interface mapping](../algorithm_guide/coupling/interface_mapping.md)).
For a `multilinear_grid` edge inside a group that does not sub-cycle:

| The edge | The norm reads |
|---|---|
| a gather (grid to points), and any mapping that does not deliver more entries than it reads | the delivered value, at the positions the step uses |
| a scatter of a few points onto a larger grid, `geometry=("source", ...)` | the source value **and the positions**, each a reading of its own |
| the same scatter with `geometry=("target", ...)` | the source value alone |

* **Positions are measured in grid spacings**, axis by axis: a point's
  change over `rtol` spacings, pooled with the other entries.  Not
  relative to the positions' own size, which depends on where the origin
  of your coordinates is: the same problem a thousand spacings away takes
  the same passes.  `atol`, a dead band on a quantity's magnitude, does
  not apply to a position.
* **A target-anchored geometry is the target's state before the step**,
  the same at every pass.  It has no residual, so it is not a reading of
  a scatter anchored there; a gather anchored there is read at it.
* **A solve returns what the norm measures whole as it accepted it**: the
  source value of such a scatter and, anchored at its source, its
  positions.  Every other field is recomputed by one pass, as for any
  group under this norm.
* **A float32 position far from the origin limits the tolerance.**  A
  position `u` spacings from zero is stored to about `1e-7 u` spacings;
  at `rtol=1e-4` that is the whole tolerance by a thousand spacings, and
  `compile()` warns from a quarter of that distance (an eighth for a
  Gauss-Seidel pair; the limits above have the arithmetic), of a scatter
  that reads the positions and of a gather whose value is interpolated at
  them.  Hold the positions in float64, keep the origin near the grid, or
  loosen `rtol`.
* **No bound is reported under this norm yet**: `rho_spectral`,
  `spectral_error_bound`, the gradient bound and `precision_limited` are
  withheld with a `not_usable_reason`, as in the limits above.

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
