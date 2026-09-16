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

A mapping is written to a config (`gm.to_dict()`, JSON / YAML) or a USD
stage (`save_graph_to_usd`, attribute `maddening:mappingSpecJson`) as its
**`MappingSpec`** — the recipe, never the weights:

```json
"mapping": {"kind": "rbf", "mode": "conservative", "shape": [12, 6],
            "kernel": "thin_plate_spline", "epsilon": 2.0,
            "polynomial": true, "ridge": 1e-8,
            "points": {"source_points": {"node": "fluid", "field": "grid_x"},
                       "target_points": {"asset": "solid_points.npy"}}}
```

`kind` names the factory, the flat keys are its hyper-parameters
(`shape` is informational and checked on reload), and `points` maps each
array argument of the factory (`source_points` / `target_points`,
`source_boundaries` / `target_boundaries`, `H`) to a **point reference**.
Every factory attaches the spec to the mapping it returns
(`mapping.spec`, `mapping.describe()`); `GraphManager.from_dict` and
`load_graph_from_usd` rebuild the mapping by calling the same factory on
the resolved points (`MappingSpec.build(resolve_points)`), so the rebuilt
`H` is bitwise equal to the original, and register it in
`params["mappings"]` exactly as `add_edge(mapping=)` does.  `add_edge`
also accepts a `MappingSpec` (or its dict) directly.

### Point references

| reference | resolves to |
|-----------|-------------|
| `{"node": "<name>", "field": "<key>"}` | the node's `static_data[key]` (a `StaticArray` is unwrapped) or, failing that, an array-valued constructor parameter `node.params[key]` — e.g. `{"node": "rod", "field": "grid_x"}` for a `HeatNode` |
| `{"asset": "<path>.npy"}`, `{"asset": "<path>.npz", "key": "<member>"}` | a NumPy file, **relative to the directory the config / stage lives in** (`from_dict(..., base_dir=)`; `load_graph_from_usd` defaults to the stage file's directory).  Absolute paths and `..` are refused. |
| `{"inline": [...], "dtype": "float64"}` (or a plain list) | the points themselves — at most `INLINE_POINT_LIMIT` (64) of them |

Tell the factory where its points came from with `source_ref=` /
`target_ref=` (and `matrix_mapping(H, asset="H.npy")` for an explicit
matrix, which is never inlined — save it yourself with `numpy.save` next
to the config):

```python
gm.add_edge("fluid", "solid", "traction", "force",
            mapping=rbf_mapping(fluid_pts, solid_pts, mode="conservative",
                                source_ref={"node": "fluid", "field": "grid_x"},
                                target_ref={"asset": "solid_points.npy"}))
```

Without a reference, a set of at most 64 points is inlined automatically;
a larger one leaves the mapping usable but **not serialisable**:
`gm.to_dict()` and `save_graph_to_usd` refuse with a message naming the
argument to pass (`gm.to_dict(strict_mappings=False)` still describes it,
for display — the REST `GET /graph` uses that).  A hand-built
`StaticLinearMapping` or a custom `Mapping` object has no spec and is
refused the same way.

### Weights: config vs checkpoint

The config carries the recipe; a checkpoint (`save_state`) carries the
actual `params["mappings"]` weights, which may have been trained by
`sysid` or edited.  Loading a config rebuilds the geometric weights; a
checkpoint loaded afterwards overwrites them — **the checkpoint wins**
(`tests/core/test_mapping_spec_serialisation.py`).  The FMI exporter
never exposes mapping weights, so the exported `modelDescription.xml`
is identical with and without a mapping on an edge.

Still planned for 0.5.0: matrix-free mappings for moving interfaces
(`geom`), Wendland / partition-of-unity sparsity, and
scaled-consistent / nearest-projection variants.

## Legacy closures

`maddening.core.coupling.interface_mapping.rbf_interpolation` and friends
still return closures usable as `transform=`; they now share the
polynomial-augmented, solve-based matrix construction (`rbf_matrix`).
Prefer `mapping=`: a closure's matrix is a baked constant the graph can
neither differentiate nor replace.
