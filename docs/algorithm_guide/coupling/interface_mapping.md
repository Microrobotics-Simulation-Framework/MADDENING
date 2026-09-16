# Interface mapping

When two coupled nodes discretise their shared interface differently
(a coarse and a fine rod, a surface mesh and a point cloud), an edge
needs a **mapping** that turns a field on the source interface into a
field on the target interface before the node consumes it.

```python
from maddening.core.coupling.mapping import rbf_mapping

gm.add_edge("fluid", "solid", "traction", "force",
            mapping=rbf_mapping(fluid_points, solid_points,
                                kernel="thin_plate_spline", mode="conservative"))
```

The edge applies `mapping` first, then the scalar `transform` (unit
conversion, sign), then `additive` accumulation, exactly as before.

## The `Mapping` protocol

```python
class Mapping(Protocol):
    kind: str; mode: str
    n_source: int; n_target: int
    def params_pytree(self) -> dict: ...
    def apply(self, field, weights=None, geom=None): ...
    def apply_T(self, field, weights=None, geom=None): ...
```

`apply` maps a source field (`(n_source,)` or `(n_source, C)`) to the
target; `apply_T` is the transpose.  `weights` is the mapping's entry of
the graph parameter pytree and `geom` is reserved for mappings that
depend on a moving interface (planned for 0.5.0; static mappings ignore
it).

## Weights live in `params["mappings"]`

A mapping's weights are **not** baked into a closure.  At compile time
the graph snapshots `mapping.params_pytree()` into
`gm.params["mappings"]["<src>.<field>-><tgt>.<field>"]`, and every step
receives them as a traced input, exactly like node constants (see
[graph parameters](../../user_guide/parameters.md)).  Consequences:

* `jax.grad` of a trajectory loss reaches the mapping weights (learned
  edges, sensitivity of a coupled result to the interface operator);
* a replaced matrix takes effect on the next step without recompiling;
* the IFT rule of a coupling group carries the dependence, so a mapped
  edge inside a coupling group is differentiable end-to-end.

`gm.params["mappings"]` is validated like the node entries: an unknown
edge key is a `ValueError` at trace time.

## `StaticLinearMapping` and its factories

The first implementation is a dense matrix `H` (`n_target × n_source`),
`target = H @ source`, with `params_pytree() == {"H": H}`.

| factory | what it builds |
|---------|----------------|
| `rbf_mapping(src_pts, tgt_pts, kernel=, epsilon=, polynomial=True, ridge=1e-8, mode=)` | radial-basis interpolation; kernels `gaussian`, `multiquadric`, `inverse_multiquadric`, `thin_plate_spline` |
| `nearest_neighbor_mapping(src_pts, tgt_pts, mode=)` | 0/1 selection matrix |
| `projection_1d_mapping(src_boundaries, tgt_boundaries)` | cell-average projection between 1D grids (conservative) |
| `matrix_mapping(H, mode=)` | bring your own weights (e.g. supermesh weights precomputed offline) |

### Polynomial augmentation and the patch test

With `polynomial=True` (the default) the RBF interpolant is augmented
with the linear polynomial `1, x_1, …, x_D`:

```
[Φ_ss + λI   P_s] [c]   [v]
[P_sᵀ         0 ] [d] = [0]        H = [Φ_ts  P_t] · A⁻¹[:, :n_source]
```

This is the well-posed form for thin-plate splines and it reproduces
constant and linear fields exactly, which is the **patch test** the
implementation is gated on (`tests/core/test_mapping.py`: constants and
linear fields across random non-conforming point sets to float32
round-off, and to 1e-8 in float64).  A plain Gaussian interpolant does
not pass it.

The system is solved, not inverted, and the ridge `λ = ridge · max|Φ_ss|`
is relative to the kernel matrix, so `ridge=1e-8` means the same thing
for a Gaussian (entries ≤ 1) and a thin-plate spline on a metre-sized
interface.

### Consistent vs conservative

Following preCICE:

* `mode="consistent"` transfers a *value* (temperature, displacement,
  velocity): `H @ v` interpolates, so a constant field stays constant.
* `mode="conservative"` transfers an *integral quantity* (force, heat
  flow, mass flux): the total over the target equals the total over the
  source.  It is the transpose of the consistent mapping in the
  opposite direction, `H = H_{t→s}ᵀ`, and it preserves sums exactly
  when `H_{t→s}` reproduces constants (`1ᵀ H_{t→s}ᵀ v = (H_{t→s} 1)ᵀ v =
  1ᵀ v`) — which is why polynomial augmentation is on by default.  The
  **conservation test** in `tests/core/test_mapping.py` checks this for
  every kernel and for nearest-neighbour.

Choose the mode by what the field *is*, not by which node sends it: a
fluid sending tractions is conservative; a solid sending displacements
is consistent.

## Shapes and validation

`add_edge` checks `mapping.n_source` against the source field's size
and, when the target declares a `boundary_input_spec` shape,
`mapping.n_target` against it — a mismatch is a `ValueError` at graph
construction, not a broadcast surprise inside the jit.

## Serialisation

`gm.to_dict()` records `mapping.describe()` (kind, mode, shape,
hyper-parameters — never the weights); `from_dict` and `save_graph_to_usd`
refuse a mapped edge for now.  Registry serialisation of a `MappingSpec`
(kind + hyper-parameters + point-source references, rebuilt on load) is
planned with the USD read path in 0.5.0, together with matrix-free
mappings for moving interfaces (`geom`), Wendland / partition-of-unity
sparsity, and scaled-consistent / nearest-projection variants.

## Legacy closures

`maddening.core.coupling.interface_mapping.rbf_interpolation` and friends
still return closures usable as `transform=`; they now share the
polynomial-augmented, solve-based matrix construction (`rbf_matrix`).
Prefer `mapping=`: a closure's matrix is a baked constant the graph can
neither differentiate nor replace.
