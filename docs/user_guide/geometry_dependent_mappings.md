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

*Porting a node that samples a cell-centred grid itself?  See the worked example [Moving a node's own sampling onto an edge](moving_a_sampling_onto_an_edge.md).*

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
* **The interface norm solves, and its bounds are not reported in
  0.4.0.**  See the next section for what it reads.  A group under
  `convergence_norm="interface"` with a geometry-dependent mapping
  reports its solve and the float floor of its residual
  (`precision_limited` and `residual_precision_floor`, below), and
  withholds the bounds and the estimates, as above.  For a diagnostic
  run that reports them, run the same group under `"mixed"` or `"l2"`
  with `diagnostics=True`.  With another mapping kind on an internal
  edge, or in a sub-cycled group, the interface norm is refused at
  `compile()`: use `"l2"` or `"mixed"`.
* **Under the interface norm, positions need a dtype that resolves the
  tolerance where they are, on the grid as well as in your
  coordinates.**  A position is rounded at the larger of two distances,
  in spacings (`eps` is 1.2e-7 in float32 and 2.2e-16 in float64):
  - its distance `u` from the **zero of your coordinates**: it is stored
    to `eps * u` spacings;
  - its **lattice coordinate**, its distance from the **grid's first
    point** on that axis: the mapping forms `(x - origin) / spacing` in
    the positions' dtype, so its weights are resolved to `eps` times
    that, however small the position itself is.

  Call the larger `r`.  **No choice of the coordinates' origin changes a
  lattice coordinate**: markers in the middle of a grid of 16001 points
  are 8000 spacings from its first point wherever you put the zero, and
  float32 weights there are resolved to 9.5e-4 of a cell.

  That rounding enters what the criterion reads in two ways.
  Which one is decided by the entry counts of the mapping (the table in
  the next section), not by its direction:
  - **a mapping that delivers more entries than it reads, anchored at
    its source** (a scatter of fewer points than the grid has entries)
    reads those positions themselves and asks them to change by less
    than `rtol` of one spacing;
  - **a mapping that does not deliver more entries than it reads** (a
    gather onto no more points than the grid has entries; either anchor)
    reads a value computed at those positions.  A position moved by
    `eps * r` spacings moves an interpolation weight by as much, and the
    delivered value by that times the field's variation across the cell
    over the value's own size.  **The count below takes that ratio to be
    one**: a field that varies across a cell by about the size of the
    value delivered.

  The float floor of the residual counts four such roundings per
  evaluation of a pass for every entry of either reading
  (`PRECISION_FLOOR_ULPS`; `E` evaluations a pass: one where every node
  reads the previous iterate and evaluates once, two for a Gauss-Seidel
  pair), and `compile()` warns (`UserWarning`) of each edge whose
  positions put that count at the tolerance or above:

  ```text
  4 * E * eps * max r >= rtol
  ```

  with `r` taken axis by axis in that axis's spacing and `eps` that of the
  dtype the positions are stored in, on the state `compile()` is called
  with.  In float32 at `rtol=1e-4` that is from 210 spacings (105 for a
  Gauss-Seidel pair) from zero **or from the grid's first point**; at
  `rtol=1e-5`, from 21 (10.5); at the default `rtol=1e-6`, from two.  So
  on a grid longer than that, float32 positions are warned of over most
  of the grid, and float64 positions are the remedy.  (Counted from the
  coordinates' zero alone, as it was before, four float32 markers within
  4 spacings of zero on grids of 16001 and 100001 points centred there
  compiled silently and read a floor of 0.26 to 0.50 tolerances and
  `converged=True` with the state 2 to 20 and 34 tolerances from the
  float64 fixed point; the floor there is now 623 and 3890 under
  Gauss-Seidel at `rtol=1e-5`, both grids are warned of, and the same
  problem reads the same floor wherever the zero is put.)
  For positions that are read themselves, the rounding counted is from
  there on the tolerance asked of them, or more: rounding alone can keep
  the group from converging (it then runs to `max_iterations`), and
  where it does converge the criterion says little of those positions.
  It is a warning, and the step is built as it would be without it:
  where the positions settle to the last bit the group converges as
  before.

  **For a delivered value the count is an assumption, not a bound, and
  it is off in both directions** (measured in float32 at `rtol=1e-4`,
  jaxlib 0.11.0, CPU; MAP-050 has the tables):
  - *a field that varies less across a cell is moved by less, and the
    warning is early.*  On two pairs of gathers the count reached the
    tolerance at 132 and 222 spacings; the float32 pairs took float64's
    passes at every distance to 5032 spacings, their readings within
    0.03 and 0.05 of a tolerance of float64's at the threshold and
    within 1.4 at 48 and 24 times it;
  - *a value delivered far smaller than the field's variation across a
    cell is moved by more, and is not warned of.*  A gather that samples
    a field near its zero (markers on the zero contour of a level set, a
    velocity at a stagnation point) 100 spacings from zero, where the
    count is 0.96 and `compile()` is silent, reported `converged=True`
    7.3 tolerances from its fixed point where the field varies across a
    cell by 250 times the value delivered, and 23 tolerances at 2500
    times; at 10 spacings (count 0.097) and 2500 times, 2.8.  With the
    same positions held in float64 every one of those is within 0.5.
    Hold such positions in float64 (keeping the coordinates' zero at
    the markers makes `u` small and leaves their lattice coordinate
    where it was).  This is MADD-ANO-247 (open) entered through the
    positions.

  **The advisory is asked once; the report reads the floor at every
  step.**  `compile()` asks of the state it is called with.  Markers
  that move afterwards, by their own update, by `set_node_state` or
  from a loaded checkpoint, are not asked again by it.  The run-time
  reading is in the report: for such a group
  `coupling_diagnostics()[key]` carries `residual_precision_floor`, the
  float floor of the residual at the state that step returned (in
  tolerances; the positions enter it at their rounding there), and
  `precision_limited`, `True` where the residual is at or below it, by
  the rule of every group's report.  **The report's floor is pooled;
  the advisory's number is one part's.**  The floor is one root mean
  square over every entry the norm reads in the group, as the residual
  is; the number in a `compile()` warning is what the positions put
  into the floor of the one part named, by itself.  With finer entries
  beside that part the pooled floor is the smaller (four markers a
  thousand spacings out beside two plain edges of 1e5 entries: 95.7 in
  the warning, 0.61 in the report).  Compare the warning's number with
  one to know whether those positions are resolved to the tolerance
  asked of them, and the report's floor with its `residual` and with
  one to know whether the group's criterion is.  **The floor is the
  same with `diagnostics=True` and without**: it takes the pass's
  structural evaluation count either way (the count a step measures
  with diagnostics grows with the positions' distance from zero, which
  this floor already counts).  **Read the floor's size against
  the tolerance.**  A floor of one or more says that rounding alone is
  the tolerance asked, which is what `compile()` warns of for the state
  it sees; `precision_limited=True` beside a floor far under one (a
  float64 group whose pass settled exactly, with a residual of 0.0) is
  harmless.  A float32 Gauss-Seidel pair at
  `rtol=1e-5` compiled 6 spacings from zero (the advisory starts at
  10.5) whose markers then drift 950 spacings a step reports
  `converged=True` and `residual=0.0` on every step with its positions
  3 to 20 tolerances from the fixed point (one float32 rounding of a
  position there is 6 to 46 tolerances): its report reads
  `precision_limited=True` and a floor of 117 to 557 tolerances, where
  the same pair in float64 reads a floor of a millionth of one.  A
  pair like it without the drift, 7000 spacings out (written there, or
  loaded from a checkpoint: the two behave alike to the bit), can run to
  its cap with `converged=False` and a residual of 14.8: one marker's
  position alternates between two neighbouring float32 numbers from pass
  to pass, and one rounding there is 49 tolerances.  The report says
  `precision_limited=True` beside a floor of 546.
  `precision_limited` is more sensitive than the
  advisory: it is `True` wherever the residual is at or below the
  pooled floor, which a converged residual often is well under the
  advisory's distance.  The floor counts a position at the rounding of
  the dtype it is stored in, as the advisory does, and a value at the
  coarsest floating dtype among the group's fields (a field computed
  from a coarser member's output may carry that member's rounding).  So
  float64 positions beside float32 fields put only float64's rounding
  of their distance into it (1e-12 of a spacing 5000 spacings out):
  the float32 pair above without its drift and with
  its positions held in float64 reads a floor of 0.078 of a tolerance
  (its values' share) placed 100 or 5000 spacings out or written 7000
  out, where float32 positions read 8.1, 390 and 546, and
  `precision_limited=False` wherever its residual is above that.

  The message names the edge, the node and field that store the
  positions, which of the two readings it is, the distance it counts
  (from zero or from the grid's first point, whichever is the larger,
  with the other beside it), the dtype, and the remedies that apply:
  - hold the positions in float64.  This needs `jax_enable_x64`; the
    mapping computes its weights in the geometry's dtype and casts them to
    the field's, so the other fields can stay float32: the advisory,
    the solve and the report's floor all take the positions at
    float64's rounding (the floor is then the float32 values' own, as
    above; hold the group's fields in float64 too for a floor at
    float64's).  Compute the positions' update in float64 as well: a
    move computed in float32 and added to float64 positions carries
    float32's rounding of the move, which the floor does not count (a
    950-spacing move left them 1.8 to 2.9 tolerances off at
    `rtol=1e-5`, with `converged=True` and a floor of 0.078);
  - use coordinates local to the grid, so that the positions are small
    numbers.  This is offered only where the positions are near enough
    to the **grid's first point** for it to help (within `rtol / (4 E
    eps)` spacings of it): further into a grid the lattice coordinate
    is what rounds, the message says that no choice of origin brings
    the count under the tolerance, and float64 positions or a looser
    tolerance are what is left;
  - loosen `rtol`.

  **What is not asked**, because the floor counts nothing for it: a
  delivered value the dead band drops (`atol` above its magnitude); a
  delivered value for a coordinate its mapping does not read, namely one
  on an axis of one lattice point and one of a point clamped to the
  hull (outside it by more than its rounding can cross: the weights are
  exactly 0 and 1); a mapping read at its source and anchored at its
  target, whose reading is its source value alone; and the `"l2"` and
  `"mixed"` norms, which measure positions against their own size and
  read no delivered value.  **The target-anchored scatter is a gap, not
  a guarantee** (MADD-ANO-258, open): the mapping still forms its
  weights from those positions in their dtype, and nothing counts that.
  Beside a gather at the same positions (the usual pair) the gather's
  floor flags the group.  With no such edge in the group, float32
  positions 8000 and 50000 spacings from the grid's first point read a
  floor of 0.095, no warning and `converged=True` 2.6 and 16 tolerances
  from the float64 fixed point (constructed); hold them in float64, or
  anchor the scatter at its source.  Positions that are read themselves are
  asked of every coordinate, because the criterion reads every
  coordinate of them, one on an axis of one lattice point included (in
  the spacing declared for that axis).  A delivered value anchored at
  its target is asked at the target's positions in the state `compile()`
  sees (what a step started from it reads).
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

A node that still samples a neighbour's grid inside its own `update` (handed the whole field
through a plain edge) gets none of what follows: the norm reads that edge at the grid's entries,
and a converged step left the sampled values up to 179 tolerances off at 3e5 cells (see "A mapped
edge is read on its compact side" in [the interface mapping guide](../algorithm_guide/coupling/interface_mapping.md)).
Put the sampling on the edge, as below.

The interface norm judges a group on what crosses its internal edges, and
reads a mapped edge on its compact side (see
[Interface mapping](../algorithm_guide/coupling/interface_mapping.md)).
The side is decided by the entry counts the mapping declares, never by
its direction.  For a `multilinear_grid` edge between `M` points and a
grid of `N` entries, inside a group that does not sub-cycle:

| The edge | The norm reads |
|---|---|
| does not deliver more entries than it reads: a gather (grid to points) onto `M <= N` points; a scatter (points to grid) of `M >= N` points | the delivered value, at the positions the step uses |
| delivers more entries than it reads, `geometry=("source", ...)`: a scatter of `M < N` points, whose positions the markers hold; a gather onto `M > N` points, whose positions the grid node holds | the source value **and the positions**, each a reading of its own |
| delivers more entries than it reads, `geometry=("target", ...)` | the source value alone |

With a few markers on a large grid, the usual case, the gather is the
first row and the scatter the second or third.  With more markers than
grid entries (several particles per cell) it is the other way round: the
gather is read at the grid field and the scatter as delivered.

* **Positions are measured in grid spacings**, axis by axis: a point's
  change over `rtol` spacings, pooled with the other entries.  Not
  relative to the positions' own size, which depends on where the origin
  of your coordinates is: the same problem a thousand spacings away takes
  the same passes.  `atol`, a dead band on a quantity's magnitude, does
  not apply to a position.
* **A target-anchored geometry is the target's state before the step**,
  the same at every pass.  It has no residual, so it is not a reading of
  an edge anchored there that is read at its source; an edge read as
  delivered and anchored there is read at it.
* **A solve returns what the norm measures whole as it accepted it**: the
  source value of an edge that delivers more entries than it reads and,
  anchored at its source, its positions.  Every other field is
  recomputed by one pass, as for any group under this norm.  With a few
  markers on a large grid that keeps the markers' value and positions;
  with more markers than grid entries it keeps the grid field the gather
  reads, and the markers' value and positions are recomputed.
* **A float32 position far from the origin, or far into a grid, limits
  the tolerance.**  A position `r` spacings from the coordinates' zero
  or from the grid's first point, whichever is further, is resolved to
  about `1e-7 r` spacings; at `rtol=1e-4` that is the whole tolerance by
  a thousand spacings, and `compile()` warns from a quarter of that
  distance (an eighth for a Gauss-Seidel pair; the limits above have
  the arithmetic, and what the count assumes), of an edge that reads
  the positions themselves and of one whose delivered value is computed
  at them.  Hold the positions in float64, or loosen `rtol`; moving the
  coordinates' origin helps only near the grid's first point.  At run
  time the report's `precision_limited` and `residual_precision_floor`
  say where the positions are then.
* **`converged=True` is a local statement.**  It reads the contraction
  the last passes showed, and it places the group within a few
  tolerances of its fixed point only where the pass keeps that rate all
  the way there.  A pass through this mapping can lose it in two ways:
  a marker thrown across a lattice plane into a cell where the pass
  contracts slowly (250 tolerances off, constructed), and, on a grid of
  two or three axes, a pass that slows down **inside one cell**, where
  the interpolation is not linear in the position (14,000 tolerances
  off, constructed, with no plane crossed).  So "no marker crosses a
  lattice plane" does not secure it.  Where it matters, tighten `rtol`
  and compare, or run the group under `"mixed"` or `"l2"` with
  `diagnostics=True` for a bound.
* **No bound is reported under this norm in 0.4.0**: `rho_spectral`,
  `spectral_error_bound`, the gradient bound and the estimates are
  withheld with a `not_usable_reason`, as in the limits above;
  `precision_limited` and `residual_precision_floor` are reported.  A
  diagnostic run of the same group under `"mixed"` or `"l2"` reports the
  bounds.

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
