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
            "points": {"source_points": {"node": "fluid", "field": "grid_x",
                                         "sha256": "8b6cb7ec8637..."},
                       "target_points": {"asset": "solid_points.npy",
                                         "sha256": "ba24aeb27cac..."}}}
```

`kind` names the factory, the flat keys are its hyper-parameters
(`shape` is informational and checked on reload), and `points` maps each
array argument of the factory (`source_points` / `target_points`,
`source_boundaries` / `target_boundaries`, `H`) to a **point reference**.
Hyper-parameters are type-checked on the way in and out: `epsilon` and
`ridge` must be finite reals (a config must stay valid JSON — no
`Infinity`), `polynomial` a bool, `kernel` / `mode` / `label` strings.
Every factory attaches the spec to the mapping it returns
(`mapping.spec`, `mapping.describe()`); `GraphManager.from_dict` and
`load_graph_from_usd` rebuild the mapping by calling the same factory on
the resolved points (`MappingSpec.build(resolve_points)`), so the rebuilt
`H` is bitwise equal to the original, and register it in
`params["mappings"]` exactly as `add_edge(mapping=)` does.  `add_edge`
also accepts a `MappingSpec` (or its dict) directly.  Anything that goes
wrong while rebuilding one edge — a malformed spec, a reference that does
not resolve, an unreadable asset, a hyper-parameter of the wrong type, a
singular solve — is a `MappingRebuildError` (a `ValueError`) naming that
edge, with the original exception chained as `__cause__`.

`describe()["kind"]` is the mapping's **user-facing** kind: for
`matrix_mapping(H, kind="supermesh")` it stays `"supermesh"` (that is
what `GET /graph` and dashboards key on), while the spec underneath
keeps `kind: "matrix"` — the factory to call — and carries the label as
the `label` hyper-parameter.  Reading `describe()` back with
`MappingSpec.from_dict` restores both.

### Point references

| reference | resolves to |
|-----------|-------------|
| `{"node": "<name>", "field": "<key>"}` | the node's `static_data[key]` (a `StaticArray` is unwrapped) or, failing that, an array-valued constructor parameter `node.params[key]` — e.g. `{"node": "rod", "field": "grid_x"}` for a `HeatNode` |
| `{"asset": "<path>.npy"}`, `{"asset": "<path>.npz", "key": "<member>"}` | a NumPy file, **relative to the directory the config / stage lives in** (`from_dict(..., base_dir=)`; `load_graph_from_usd` defaults to the stage file's directory).  Absolute paths and `..` are refused; the path is then `resolve()`d and the *resolved* file must still lie under the resolved `base_dir`, so a symlink (to a file or to a directory) that leaves it is refused too.  `key` selects an `.npz` member and is an error on a `.npy`. |
| `{"inline": [...], "dtype": "float64"}` (or a plain list) | the points themselves — at most `INLINE_POINT_LIMIT` (64) points and `INLINE_ELEMENT_LIMIT` (1024) numbers in total, finite, of a bool / integer / float dtype |

Asset files are read defensively, because a config is untrusted input:
the `.npy` header (or the `.npz` directory entry) is read first and the
array is refused before anything is allocated when it would exceed
`MAX_ASSET_BYTES` (256 MiB — a module constant you can raise for a
genuinely large interface), when the header claims more data than the
file holds, or when its dtype is not bool / integer / float.

A `{"node": ...}` reference resolves through `gm.get_node(name)`: the
node's `static_data[key]` first (a `StaticArray` is unwrapped), then an
array-valued constructor parameter `node.params[key]`; a scalar (0-d) or
non-numeric field is a `PointReferenceError`.  Wrapper nodes classify
their inner node's static data at build time and expose none of their
own — `ShardedStencilNode` and `ShardedUnstructuredNode` included — so a
node reference cannot reach through them; save those points as an asset
instead.

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

### References are checked against the points they describe

Each reference records `sha256`, the content hash
(`point_array_digest`: dtype, shape and C-order bytes) of the array the
factory was actually given.  A reference that resolves to something else
— a typo naming the wrong node field, a node field or asset file that
moved since the config was written — is refused instead of silently
rebuilding a different operator:

* `gm.to_dict()` and `save_graph_to_usd` resolve every **node**
  reference as they write and refuse one that no longer resolves (a
  removed node) or no longer matches the hash, naming the edge;
* `MappingSpec.build` / `from_dict` / `load_graph_from_usd` check every
  reference again — node, asset and inline — as they rebuild.

A hand-written reference may omit `sha256`; then nothing is checked and
the hash is recorded on the first rebuild.

### Weights: config vs checkpoint

The config carries the recipe; a checkpoint (`save_state`) carries the
actual `params["mappings"]` weights, which may have been trained by
`sysid` or edited.  Loading a config rebuilds the geometric weights; a
checkpoint loaded afterwards overwrites them — **the checkpoint wins**
(`tests/core/test_mapping_spec_serialisation.py`).  Because the config
cannot carry them, `gm.to_dict()` emits a `UserWarning` naming the edge
when the live `params["mappings"]` weights are no longer the ones the
recipe rebuilds, so trained weights are not lost silently; save a
checkpoint and load it after the config.  Which weights an optimiser may
move is metadata, not data, so it *is* in the config:
`gm.set_param_spec(edge.key, "H", ParamSpec())` round-trips through both
JSON and USD.  The FMI exporter never exposes mapping weights, so the
exported `modelDescription.xml` is identical with and without a mapping
on an edge.

Still planned for 0.5.0: matrix-free mappings for moving interfaces
(`geom`), Wendland / partition-of-unity sparsity, and
scaled-consistent / nearest-projection variants.

## Legacy closures

`maddening.core.coupling.interface_mapping.rbf_interpolation` and friends
still return closures usable as `transform=`; they now share the
polynomial-augmented, solve-based matrix construction (`rbf_matrix`).
Prefer `mapping=`: a closure's matrix is a baked constant the graph can
neither differentiate nor replace.
