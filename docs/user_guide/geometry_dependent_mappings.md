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
  difference's own error: about 6e-4 in float32 (at most 6e-3 on 236
  drawn examples; 0.11 on a field that cancels three thousandfold inside
  a gather) and under 3e-6 in float64.  The check fails for a node or
  a mapping whose derivative is not the derivative of its value (a
  `stop_gradient` on an input, a rounding, a branch on a position), and
  on a step where the two could not be compared (a state that is not
  finite, or a geometry the pass reads from a constant the step could not
  move); `not_usable_reason` says which of the two it was.  It compares
  nothing in a field the positions move by less than 32 float resolutions
  of the residual, so a wrong derivative of a term that weak is not seen:
  a pass whose product saw no geometry at all read 0.31 to 0.98 in
  float32 on those draws and 1.0 in float64, and 0.22, under the
  tolerance, on a deposit so weak that the term moved the spectral radius
  by 0.02.
* **Across a lattice plane the flags are withdrawn and the numbers
  kept.**  A multilinear stencil is one polynomial of the positions
  inside a lattice cell and another in the next, so the pass's Jacobian
  jumps where a position crosses a lattice plane, or a face of the
  grid's hull (outside it the kernel clamps).  `rho_spectral`,
  `spectral_error_bound` and `gradient_relative_error_bound` are the
  linearisation at the returned iterate: they describe the pass in the
  lattice cells its positions are in *there*.  Where a position the pass
  reads from the iterate is within twice `spectral_error_bound` of a
  lattice plane, the bound stands only if the step certified its
  linearisation across the Newton step to the fixed point (the
  Newton-Kantorovich check behind `gradient_relative_error_bound`, which
  is then finite).  Otherwise `spectral_usable` and
  `gradient_bound_usable` are `False`, the numbers are reported as
  computed, and `not_usable_reason` names the case: the fixed point may
  be in the next cell, where the pass contracts at another rate.
  Measured on a marker whose fixed point was 2e-4 of a spacing past a
  plane: a radius of 0.28 before the plane and 0.97 after it, and a
  bound 0.13 times the true distance on a converged solve.  The
  positions concerned are a member's source-anchored geometry and the
  target-anchored geometry of a member that computes fluxes; a position
  that is a constant of the pass (a target-anchored geometry read by
  `update`, a node outside the group) does not move between the iterate
  and the fixed point and withdraws nothing.  A tighter tolerance
  usually brings the returned iterate into the fixed point's cell.
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
  with a geometry; `GET /graph` shows the key.
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
