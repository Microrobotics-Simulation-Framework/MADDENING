# Interface mapping

When two coupled nodes discretise their shared interface differently
(a coarse and a fine rod, a surface mesh and a point cloud), an edge
needs a **mapping** that turns a field on the source interface into a
field on the target interface before the node consumes it.

<!-- snippet: no-run, reason: fragment: an edge on a fluid-solid graph; gm and the point sets are the reader's -->
```python
from maddening.core.coupling.mapping import rbf_mapping

gm.add_edge("fluid", "solid", "traction", "force",
            mapping=rbf_mapping(fluid_points, solid_points,
                                kernel="thin_plate_spline", mode="conservative"))
```

The edge applies `mapping` first, then the scalar `transform` (unit
conversion, sign), then `additive` accumulation, exactly as before.

## The `Mapping` protocol

<!-- snippet: no-run, reason: fragment: a summary of the Mapping protocol in maddening.core.coupling.mapping, not its definition -->
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
the graph parameter pytree and `geom` is the geometry of a mapping that
depends on a moving interface (experimental, see *Geometry-dependent
mappings* below; static mappings ignore it).  A mapping whose weights are applied through a structure the
parameter tree does not hold may also define `structure_digest()`; see
[Sparse mappings](#sparse-mappings).

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

The factories check the coordinates they are given, and refuse what their
formula does not cover with a `ValueError` naming the argument and the first
offending index (a `MappingRebuildError` when the mapping is rebuilt from a
config or a USD stage):

* cell boundaries (`projection_1d_mapping`) must be one-dimensional, hold at
  least two values, be finite and be **strictly increasing**.  They are never
  sorted or reversed for you: the field keeps its cell order.  The two grids
  need not cover the same interval — a cell outside the other grid is
  treated as empty, so the integral is preserved when the target grid covers
  the source grid, a constant is reproduced on target cells the source grid
  covers, and two grids that share no interval give a matrix of zeros;
* point sets (`rbf_mapping`, `nearest_neighbor_mapping`) must be finite
  `(n,)` or `(n, d)` arrays of one `d`, and the set the operator
  interpolates from (the source in consistent mode, the target in
  conservative mode) must hold a point.  Coincident points are accepted:
  nearest neighbour takes the lowest index, and the RBF ridge shares the
  weight between them;
* a matrix (`matrix_mapping`) must be finite.

An accepted input gives the same operator as before the checks, bit for bit
(MADD-ANO-192 has the inputs that used to give a wrong one).

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

`kind` names the factory -- one of the four above, one of the three
[sparse kinds](#sparse-mappings), or a kind you
[registered yourself](#registering-your-own-mapping-kind) -- the flat keys
are its hyper-parameters (`shape` is informational and checked on reload),
and `points` maps each array argument of the factory (`source_points` /
`target_points`, `source_boundaries` / `target_boundaries`, `H`) to a
**point reference**.
Hyper-parameters are type-checked on the way in and out: `epsilon` and
`ridge` must be finite reals (a config must stay valid JSON — no
`Infinity` — and an integer too large to be a float64 is refused the same
way), `polynomial` a bool, `kernel` / `mode` / `label` strings.
Every factory attaches the spec to the mapping it returns
(`mapping.spec`, `mapping.describe()`); `GraphManager.from_dict` and
`load_graph_from_usd` rebuild the mapping by calling the same factory on
the resolved points (`MappingSpec.build(resolve_points)`), so the rebuilt
`H` is bitwise equal to the original, and register it in
`params["mappings"]` exactly as `add_edge(mapping=)` does.  `add_edge`
also accepts a `MappingSpec` (or its dict) directly.  Anything that goes
wrong while rebuilding one edge — a malformed spec, a reference that does
not resolve, an unreadable asset, a hyper-parameter of the wrong type, a
singular solve, a package a factory cannot import — is a
`MappingRebuildError` (a `ValueError`) naming that edge, with the original
exception chained as `__cause__`.

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
| `{"inline": [...], "dtype": "float64"}` (or a plain list) | the points themselves — at most `INLINE_POINT_LIMIT` (64) points and `INLINE_ELEMENT_LIMIT` (1024) numbers in total, finite, of an accepted dtype (below) |

A node reference is resolved **once**, when the mapping is built. If the
static it names is derived from a trainable parameter, such as a uniform
`HeatNode`'s `grid_x`, which is built from `length`, then calibrating
that parameter -- a fit, or any `params=` pytree -- leaves the weights at
the constructor's geometry. Nothing refuses it or warns. This is
**MADD-ANO-022**. A value *written* into the running graph is refused
(**MADD-ANO-063**): `gm.params` at the next run, `to_dict()` and
`save_state()`, `PUT /graph/params` with a 400, a checkpoint at
`POST /checkpoint/load`, and an exported FMU leaves the parameter fixed,
because the node would use it while the mapping kept the old points and a
saved config would not load. See
[Calibrating a parameter that a mapped edge's grid derives from](../../user_guide/parameters.md#calibrating-a-parameter-that-a-mapped-edges-grid-derives-from)
for what it does to a fit and for the workarounds.

#### Accepted dtypes

Whatever the reference form, a point set must be a bool, integer or float
array of **at most 8 bytes per element**: `bool`, `int8`…`int64`,
`uint8`…`uint64`, `float16`, `float32`, `float64`.  Complex, string,
object and datetime arrays are refused, and so is **extended precision** —
`numpy.longdouble`, spelled `float96` on 32-bit x86 and `float128` on
x86-64 and aarch64.

An extended-precision point set is **refused with an error, not narrowed
to `float64`**.  It fails both halves of what a reference has to do:
`arr.tolist()` yields `numpy.longdouble` objects that `json.dumps` cannot
write, so the config could not be saved; and `point_array_digest` is not
stable for it, because the padding bytes of an 80-bit value in its
16-byte slot are not zeroed, so two arrays that compare equal can hash
differently and a reference to them is rejected at random.  Downcasting
quietly would hide a precision loss you did not ask for — a well-known
source of numerical bugs that are very hard to trace back to their cause
— so the error names the dtype and the fix:

<!-- snippet: no-run, reason: fragment: fluid_pts and solid_pts are the reader's point sets -->
```python
mapping = rbf_mapping(np.asarray(fluid_pts, dtype=np.float64),
                      np.asarray(solid_pts, dtype=np.float64))
```

The same applies to the *values* of an inline reference that names no
`"dtype"`. Such a payload is read as `float64`. `numpy.longdouble` arrays
and the `numpy.longdouble` scalars their `tolist()` returns used to be
rounded to `float64` without a word. They are now refused, and the error
says what you can do instead. No reference form keeps the extra
precision, so the points can be at most `float64` whichever route you take:

- convert them first: `np.asarray(points, dtype=np.float64).tolist()`
  gives plain Python floats;
- or add `"dtype": "float64"` to the reference to accept the rounding
  explicitly;
- or, for a set too large to inline, save the `float64` array with
  `numpy.save` and pass `{"asset": "<file>.npy"}`.

Three more kinds of value were changed by the same coercion and are
refused the same way, each with a message saying what to do instead:

- **integers `float64` cannot represent** (magnitude above `2**53` and
  not a float64 value, as a Python `int` or an 8-byte NumPy integer):
  `2**53 + 1` used to become `2**53`.  Add `"dtype": "int64"` (or
  `"uint64"`) to keep integer points exact, or `"dtype": "float64"` to
  accept the rounding;
- **`decimal.Decimal` values `float64` cannot represent**: they went
  through `float()`.  No reference keeps decimal digits; convert first or
  add `"dtype": "float64"`;
- **complex values, at any width** (`numpy.clongdouble` included) —
  refused **with or without** a `"dtype"`, because the coercion to a real
  dtype drops the imaginary part (with only a NumPy `ComplexWarning`)
  rather than rounding it.  Pass `np.real(points)` if the imaginary parts
  are zero, or the two parts as two real columns.

Values `float64` holds exactly (`2**53`, powers of two beyond it,
`Decimal("0.5")`) and every other input are coerced as before.

This is a limit of the *serialised* form, not a judgement about extended
precision, and it is not a closed door.  If a real interface ever needs
it, extended precision can be supported later behind the same API — a
composite inline representation (the value bytes, or a mantissa /
exponent pair, written as JSON-safe integers) together with a canonical
digest, or an FFI path that formats and hashes the value itself.  Nothing
in the current format assumes 8 bytes is the last word; the constraint is
explicit today so that you can trust what a saved config holds.

Asset files are read defensively, because a config is untrusted input:
the `.npy` header (or the `.npz` directory entry) is read first and the
array is refused before anything is allocated when it would exceed
`MAX_ASSET_BYTES` (256 MiB — a module constant you can raise for a
genuinely large interface), when the header claims more data than the
file holds, or when its dtype is not an accepted one — including
extended precision, which is caught from the header before any read.

The resolved path is opened exactly once, with `O_NOFOLLOW`, and the
size check (`fstat` on that descriptor), the header and the data all
come from it — re-opening the name would let a writer in the config
directory swap the checked file for a symlink in between.  A reference
that *names* a symlink is unaffected: the link is followed by the
resolution, and it is the target that is opened.

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

<!-- snippet: no-run, reason: fragment: an edge on a fluid-solid graph the guide does not build -->
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
`StaticLinearMapping` or a custom `Mapping` object that carries no spec
is refused the same way; to make a mapping of your own serialisable,
[register its kind](#registering-your-own-mapping-kind).

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

`python -m maddening.examples.coupling.interface_mapping_demo` builds a
mapped edge between an 8-cell and a 24-cell rod from node references,
saves it with `to_dict()`, reloads it with `from_dict()` (the rebuilt
`H` is bitwise equal and the two graphs step identically), and shows the
MADD-ANO-063 refusal of a `length` write that would move the mapped
points.

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

Still planned for 0.5.0: further kinds for moving interfaces (0.4.0 has
one, experimental: *Geometry-dependent mappings* below), Wendland /
partition-of-unity sparsity, and
scaled-consistent / nearest-projection variants.

## Registering your own mapping kind

*Experimental in 0.4.0: `register_mapping` may change in a minor release.*

The four kinds above are entries of one registry
(`maddening.core.coupling.mapping_registry`), and `register_mapping` adds
yours to it.  A mapping of a registered kind is written by `to_dict()`
and `save_graph_to_usd` and rebuilt by `from_dict()` and
`load_graph_from_usd` exactly as a built-in one is: the recipe travels,
the weights are recomputed bit for bit by your factory.

```python
import json

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import StaticLinearMapping, register_mapping
from maddening.core.coupling.mapping_spec import MappingSpec, reference_for_array
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode


@register_mapping(
    "inverse_distance",
    arrays=("source_points", "target_points"),
    hyperparameters={"power": float, "normalise": bool},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
def inverse_distance_mapping(source_points, target_points, *, power=2.0,
                             normalise=True, source_ref=None, target_ref=None):
    """Each target value is a distance-weighted mean of the source values."""
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 1)
    tgt = np.asarray(target_points, dtype=np.float64).reshape(-1, 1)
    weights = 1.0 / (1.0 + np.abs(tgt - src.T) ** power)
    if normalise:
        weights = weights / weights.sum(axis=1, keepdims=True)
    spec = MappingSpec(
        "inverse_distance",
        {"power": float(power), "normalise": bool(normalise)},
        {"source_points": reference_for_array(source_points, source_ref,
                                              name="source_points"),
         "target_points": reference_for_array(target_points, target_ref,
                                              name="target_points")},
    )
    return StaticLinearMapping(jnp.asarray(weights, dtype=jnp.float32),
                               kind="inverse_distance", spec=spec)


gm = GraphManager()
gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
x_coarse = gm.get_node("coarse").static_data["grid_x"].value
x_fine = gm.get_node("fine").static_data["grid_x"].value
gm.add_edge("coarse", "fine", "temperature", "heat_source",
            mapping=inverse_distance_mapping(
                x_coarse, x_fine, power=3.0,
                source_ref={"node": "coarse", "field": "grid_x"},
                target_ref={"node": "fine", "field": "grid_x"}))
gm.compile()

config = json.loads(json.dumps(gm.to_dict()))
stored = config["edges"][0]["mapping"]
assert stored["kind"] == "inverse_distance" and stored["power"] == 3.0
assert "H" not in stored                      # the recipe, never the weights

reloaded = GraphManager.from_dict(config, {"HeatNode": HeatNode})
reloaded.compile()
key = "coarse.temperature->fine.heat_source"
assert np.array_equal(np.asarray(reloaded.params["mappings"][key]["H"]),
                      np.asarray(gm.params["mappings"][key]["H"]))
```

What the declaration says:

* **`arrays`** -- the factory's array arguments, the point sets or
  matrices a spec refers to by [reference](#point-references).
* **`hyperparameters`** -- every other argument a spec may carry, with
  its type: `str`, `bool`, `int` or `float`.  `float` is a real number: a
  finite `int` or `float` that is not a `bool`, handed to the factory as a
  `float`.  `int` is an integer (a count, a size): an `int` that is not a
  `bool`, of a magnitude a float64 can hold, handed to the factory as an
  `int`; a float is refused for it, whatever its value.  A spec's
  hyper-parameters are checked against these declarations **before** your
  factory is called, as the built-in ones are, and an unknown one is
  refused.  The names `kind`, `points`, `shape` and `label` are reserved.
* **`references`** -- for each array, the factory keyword that carries
  its reference (`"<array>_ref"` when omitted).

The rebuild calls `factory(**arrays, **hyperparameters, **references)`.
The declaration is checked against the factory's signature when it is
registered, so a misspelt name fails at the decorator rather than while
someone loads a config.  A required argument must be one a keyword can
supply: a factory that requires a positional-only argument is refused.
A kind name is any non-empty string but the three
a config reserves for non-finite numbers (`NaN`, `Infinity`,
`-Infinity`); registering a name that is taken -- a built-in kind, or
another factory's -- is a `ValueError`, registering the same factory
again with the same declaration is a no-op, and nothing removes a kind.

**What the factory returns** is checked when a spec is rebuilt:

* a `Mapping` whose `kind` is the registered kind;
* carrying, as `.spec`, a `MappingSpec` of that kind which records the
  references and the hyper-parameters it was given.  Build each reference
  with `reference_for_array(array, reference, name=...)`, passing the
  array **as you received it** (its content hash covers the dtype, so
  hash it before any conversion) and the reference keyword through;
* whose `params_pytree()` meets the contract below.

The short way is to return a `StaticLinearMapping`, as above: a dense
matrix whose one weight is `H`.  Pass `kind=` (left at its default, the
mapping would be written as a `matrix`; `to_dict()` refuses that).  A
class of your own needs the members of the `Mapping` protocol and a
`spec` attribute, and no `describe()` method: the edge writes the spec
for you.

### What `params_pytree()` may contain

The graph snapshots `mapping.params_pytree()` into
`gm.params["mappings"][edge.key]`, and checkpoints, `POST
/checkpoint/load`, system identification, the FMU's state archive and
`to_dict()`'s "live weights differ" warning all read that entry as a flat
table of arrays.  For a mapping of any class other than
`StaticLinearMapping`, `add_edge` therefore refuses, with a `ValueError`
naming the edge, a `params_pytree()` that is not

* a plain `dict` -- empty for a mapping without weights;
* keyed by Python identifiers (a key is a member name in a checkpoint
  archive and a key of a config's `param_specs`);
* holding under each key one concrete JAX array of a floating-point
  dtype, of any shape, all finite -- not a nested container, not a NumPy
  array or a Python number (`jnp.asarray(value)` gives it a dtype that no
  longer depends on `jax_enable_x64`), not an integer or complex array.
  Indices and other structure stay attributes of the mapping, outside the
  parameter tree;
* the same on every call: it is what `reset_params()` restores and what
  `to_dict()` compares the live weights with.

Every weight is frozen by default and opts in to a fit by name
(`gm.set_param_spec(edge.key, "<weight>", ParamSpec())`), is carried by a
checkpoint under its own name, and is fixed in an exported FMU, like `H`.

### A file can only name a kind

A config, a USD stage and a checkpoint are untrusted input.  All one of
them can say about a mapping is the name of its kind, which is looked up
in the registry and nowhere else: nothing imports a module, follows a
dotted path or discovers an entry point because of what a file contains.
A kind exists only because the running program imported the code that
registered it.  A file naming a kind this process has not registered is
a `MappingRebuildError` listing the registered kinds; import the module
that registers it, then load.

The limits on [point references](#point-references) -- the asset size
cap, the inline limits, the accepted dtypes, the containment of asset
paths in the config directory, the content hash -- are enforced by the
reference resolver before a factory runs, so they hold for a registered
kind exactly as for a built-in one.  The coordinate checks of
[Shapes and validation](#shapes-and-validation) do not: each built-in
factory makes them itself, so your factory is handed whatever its
references resolve to, and geometry its formula does not cover -- an
unsorted grid, a NaN coordinate -- is its own to refuse, with a
`ValueError`.  Whatever a registered factory
raises, `from_dict` and `load_graph_from_usd` report as a
`MappingRebuildError` naming the edge and the kind, with the factory's
exception chained as `__cause__` -- an `ImportError` included, for a
factory that needs a package the loading environment lacks.

## Sparse mappings

*Experimental in 0.4.0: everything in
`maddening.core.coupling.sparse_mapping` may change in a minor release.*

A `StaticLinearMapping` stores the whole `n_target × n_source` matrix.
Between two interfaces of 100 000 points each that is 40 GB of float32,
of which a nearest-neighbour selection uses one entry per row.  A
`StaticSparseMapping` stores only the entries a row uses:

* an integer **index** of shape `(n_rows, k)`, kept on the mapping as a
  read-only host array -- structure, not a parameter;
* one floating-point weight array **`W`** of the same shape, which is the
  mapping's whole entry of the parameter tree:
  `params_pytree() == {"W": W}`.

Use a sparse mapping when the dense matrix does not fit in memory or is
mostly zeros: a nearest-neighbour transfer, a cell-average projection, a
stencil or supermesh operator computed offline.  Keep the dense kinds for
small interfaces and for RBF interpolation, whose matrix is full.

### The three kinds

| factory | what it builds |
|---------|----------------|
| `sparse_nearest_neighbor_mapping(src_pts, tgt_pts, mode=, transpose=)` | the matrix of `nearest_neighbor_mapping`, found with a k-d tree instead of an `n_target × n_source` distance table |
| `sparse_projection_1d_mapping(src_boundaries, tgt_boundaries)` | the matrix of `projection_1d_mapping`, found with a sorted sweep instead of a double loop |
| `sparse_matrix_mapping(indices, values, n_source=, mode=, name=)` | bring your own rows: `target[i] = sum_j values[i, j] * source[indices[i, j]]` |

A sparse mapping goes on an edge like any other, and its kind is
registered, so `to_dict()` / `from_dict()` and the USD writer and reader
carry it:

```python
import json

import numpy as np

from maddening.core.coupling.sparse_mapping import sparse_nearest_neighbor_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

gm = GraphManager()
gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
x_coarse = gm.get_node("coarse").static_data["grid_x"].value
x_fine = gm.get_node("fine").static_data["grid_x"].value
mapping = sparse_nearest_neighbor_mapping(
    x_coarse, x_fine,
    source_ref={"node": "coarse", "field": "grid_x"},
    target_ref={"node": "fine", "field": "grid_x"})
gm.add_edge("coarse", "fine", "temperature", "heat_source", mapping=mapping)
gm.compile()

key = "coarse.temperature->fine.heat_source"
assert mapping.indices.shape == (12, 1)                    # one source per target
assert gm.params["mappings"][key]["W"].shape == (12, 1)    # one weight per slot

config = json.loads(json.dumps(gm.to_dict()))
stored = config["edges"][0]["mapping"]
assert stored["kind"] == "sparse_nearest_neighbor"
assert "W" not in stored and "indices" not in stored       # the recipe, nothing else

reloaded = GraphManager.from_dict(config, {"HeatNode": HeatNode})
rebuilt = reloaded.edges[0].mapping
assert np.array_equal(rebuilt.indices, mapping.indices)
assert rebuilt.structure_digest() == mapping.structure_digest()
```

**The same matrix as the dense kind.**  Scattering the rows of
`sparse_nearest_neighbor_mapping` or `sparse_projection_1d_mapping` into a
zero matrix gives the dense factory's `H` bit for bit, in both modes.  For
the nearest neighbour that includes ties: they go where the dense kind
sends them, to the **lowest index** among the points at the minimal
float64 squared distance.  The k-d tree only proposes candidates; wherever
two candidates are closer to one another than rounding can tell apart, the
dense expression decides, so the result does not depend on the tree or on
the scipy version.  The equality holds for coordinates up to `1e150` in
magnitude (a larger one is refused: its squared distance overflows).

`sparse_projection_1d_mapping` requires **strictly increasing** boundaries
on both sides and refuses anything else.  It does not sort or reverse them
for you, because the field keeps its cell order.  The sparse builders ask
of their coordinates what the dense factories ask (*Shapes and validation*
above), through the same checks and in the same words, and three things
more: both point sets must hold a point (a sparse mapping has at least one
row and one column), a complex, text or object array is refused by name,
and so is a coordinate past `1e150`.

`sparse_matrix_mapping` takes an integer `indices` array and a
floating-point `values` array, both `(n_target, k)`, and `n_source`, which
cannot be inferred.  `-1` in `indices` marks an unused slot (a row with
fewer than `k` entries), and the value there must be exactly 0: a weight
that would be dropped is refused rather than dropped.  The same index may
appear twice in a row; its values add.  Like an explicit dense matrix, the
two arrays are never inlined into a config: save them with `numpy.save`
(or as two members of one `.npz`) next to the config and name the files
with `indices_asset=` and `values_asset=`.  `name=` is a free label for
display; the mapping's `kind` stays `"sparse_matrix"`.

### The index is structure, the weights are parameters

Only `W` is in `gm.params["mappings"]`.  It is a traced input like any
other weight: a gradient reaches it, a replaced `W` takes effect on the
next step without recompiling, it is frozen by default and opts in to a
fit by name, a checkpoint carries it, and an exported FMU fixes it.

The index is **baked into the compiled step as a constant**.  A weight
write can therefore never change the sparsity pattern: a new pattern is a
new mapping, a new edge and a recompile.  The mapping object is immutable
(the index array is read-only, and a copy of the object is the object).

A config carries a sparse mapping's recipe -- kind, hyper-parameters and
references -- and never its index or its weights; `from_dict` calls the
same builder on the same arrays, so the reloaded index is the one that was
saved, bit for bit.

A checkpoint carries `W` and nothing of the index, and `W` means
something only against the index it was built for.  So `save_state` writes
the SHA-256 of the pattern (`mapping.structure_digest()`: layout, sizes,
index and row counts) beside the weights, and `load_state` -- and `POST
/checkpoint/load` -- refuses weights whose digest is not the live
mapping's, before anything is loaded: weights saved for other points or
another index of the same shape, for the same index in the other layout,
or for a dense mapping (and a dense mapping refuses a sparse one's).  A
mapping class of your own that applies its weights through a structure
gets the same check by defining `structure_digest()`.

### The conservative nearest neighbour: two forms

The conservative nearest neighbour is the transpose of the reverse
selection: each source adds its value to the target nearest to it.  A
target's row is as long as the number of sources that share it, and
`transpose=` chooses how those rows are held:

* `transpose="gather"` (the default): for each target, the sources that
  add to it, **padded to the longest row**, summed along the row.  Each
  output is reduced on its own, so one compiled program gives one result
  -- measured on the CPU and on a GPU.  The cost is the padding: a
  pattern in which many sources share one target has one long row and
  pads every other row to it.  Past `MAX_SPARSE_STRUCTURE_BYTES` the
  builder refuses, before allocating, with an error that names
  `transpose="scatter"`.
* `transpose="scatter"`: one entry **per source**, applied as a
  scatter-add.  Compact whatever the pattern.  Measured on the CPU it is
  the in-order sum, one result on every call.  Measured on a GPU the same
  program gave **a different result on every run** (twenty results in
  twenty runs, hundreds to tens of thousands of ulp apart).  Choose it
  when the gather form is refused and run-to-run reproducibility on a GPU
  is not needed.

That is what was measured (jax / jaxlib 0.11.0, 2026-10-05) and nothing
more is claimed: not bit-equality between the CPU and a GPU for either
form, and not GPU behaviour on another driver or jaxlib.  The GPU
statements are one attended measurement that no test runs (GPU is not a
verified configuration, MADD-ANO-001); the CPU statements are tests, and
held on the three jaxlib versions they were run on (0.10.2, 0.11.0,
0.11.2).  The choice is
recorded in the spec, so a reloaded mapping is applied the way it was
built.  The two forms hold the same matrix, and agree with one another
and with the dense conservative kind to rounding.

The reverse-mode derivative of a gather with respect to the *field* is
itself a scatter-add.  On a GPU a gradient with respect to the source
field through either form may therefore differ in its last bits between
runs even where the forward result does not.

### What "equal to the dense mapping" means

The sparse and the dense mapping are two float evaluations of the same
sums.  Each output entry is within `(k + 2) · eps · Σ|w|·|f|` of the exact
sum over its row's `k` entries (`eps` of the result dtype), so the two
agree to rounding, entry by entry -- not bit for bit.  The row sum is
deterministic for one compiled program; like `H @ field`, it is not
bit-stable between a `jax.vmap` of a step and the step alone.

With one entry per row the two are the same number when the mapping is
applied on its own.  Inside a compiled step not even that carries over,
whatever the weights: a graph with sparse edges and the same graph with
dense ones are two programs, and the compiler evaluates the arithmetic
around a mapping differently in each (it may fuse a product with an
addition next to it -- an additive edge, a transform -- in one and not in
the other).  A graph with sparse edges and its dense twin therefore step
to within rounding of one another, not to the same bits.  Under a
constant iterator the two were at most 7.5 `eps` of the largest state
entry apart, over every structure, layout and domain the coupling
topology harness compares them in; the harness allows 256.  The harness
runs over sparse edges as it does over dense ones: a graph with sparse
edges satisfies the same monolithic reference the dense graph is held to,
and its gradients are the dense graph's.  In bfloat16 and float16, where
a rounding is a percent of the state, the twin is not compared; a renamed
or reordered sparse graph returns the original's states bit for bit.

One difference is deliberate.  A row reads only its own entries, so an
infinity or a NaN in the source field reaches only the targets that list
it; `H @ field` multiplies it by the zeros of every other row and returns
NaN everywhere.

### Limits

| what | bound |
|------|-------|
| each referenced array | `MAX_ASSET_BYTES` (256 MiB), read from the file's header before anything is allocated -- the reference resolver's, as for every kind |
| the row structure (index plus weights) | `sparse_mapping.MAX_SPARSE_STRUCTURE_BYTES`, by default the same 256 MiB: 33.5 million float32 slots, say 4.2 million targets with eight entries each.  Checked from the row counts **before** the padded arrays are allocated; the refusal names the row count, the longest and the median row and the number of entries |
| a nearest-neighbour search over a degenerate point set | `TIE_CANDIDATES_PER_POINT` (8) candidates per searched point plus `TIE_CANDIDATES_FLOOR` (a million) in total.  Very many points at one distance from very many others (points on a sphere around the ones they are searched from) are refused, not resolved.  The search runs 4096 points at a time and counts a chunk's candidates before it collects any, so the refusal comes after the chunk that passes the bound, not after every point has been measured against every other |
| index values | `0 <= index < n_source <= 2**31 - 1` (the index is `int32`) |

Each refusal is a `SparseMappingLimitError`, a `ValueError`; from
`from_dict` or `load_graph_from_usd` it is a `MappingRebuildError` naming
the edge.  `n_source` sizes nothing, so a config cannot allocate through
it.  In a process capped at 6 GiB of address space, a pattern whose padded
gather would need 160 GB, a tie set of 9e8 candidates and a projection row
of five million slots among fifty thousand rows are each a
`SparseMappingLimitError`, not a `MemoryError`.  scipy, which the nearest-neighbour builder imports for its k-d tree,
is a declared dependency; it is imported by that builder, not by
`import maddening`.

### The scale it has been measured at

One process, capped at 8 GiB of address space, jax 0.11.0 on three CPU
cores of a shared machine (2026-10-05); the times are indicative.

| a million points (or cells, or rows) a side | build | first call (compile and apply) | a later call |
|---|---|---|---|
| nearest neighbour in 3-D, consistent (`k = 1`) | 5.6 s | 0.06 s | 2 ms |
| nearest neighbour, conservative, gather (`k = 11`) | 5.4 s | 1.0 s | 44 ms |
| nearest neighbour, conservative, scatter | 5.0 s | 0.06 s | 4 ms |
| projection onto 700 000 cells (`k = 23`) | 0.34 s | | |
| your own rows, `k = 8` | 0.10 s | 0.6 s | 25 ms |

The whole run peaked at 1.4 GiB of resident memory with all five mappings
alive.  The index is embedded in every compiled program that applies the
mapping (the step, each cached scan length, a sweep): four bytes per slot
per program, 32 MB at a million rows of eight.  The same script runs at a
hundred thousand points on every push
(`tests/core/test_sparse_mapping_builders.py`).

Not in 0.4.0: a sparse mapping on a moving interface (`geom` is ignored),
sharded graphs, and sparse RBF kinds.

## Geometry-dependent mappings

*Experimental in 0.4.0: `add_edge(..., geometry=...)`, the registry flag
`needs_geometry` and everything in `maddening.core.coupling.grid_mapping`
may change in a minor release.  The user guide page
[Geometry-dependent mappings](../../user_guide/geometry_dependent_mappings.md)
has a runnable example, the refusals and the limits of this release.*

Every mapping above is **static**: its operator is fixed when the graph is
built.  A **geometry-dependent** mapping also reads a *geometry*, a state
field of the edge's own source or target node that moves during the run
(marker positions, a deforming interface), and builds its operator from it
at every call:

```
value_at_target = mapping.apply(value_at_source, weights, geom)
```

The edge names the field, `add_edge(..., mapping=m, geometry=(anchor,
field))` with `anchor` `"source"` or `"target"`, and the graph resolves it
as it resolves the value.  The geometry is ordinary node state: it is
integrated by its node, batched by `run_sweep`, differentiated by
`jax.grad`, and saved by a checkpoint.  The mapping declares
`needs_geometry = True` and a static `geometry_shape` (see the `Mapping`
protocol's optional attributes), and must be a pure function of its three
arguments.

### The time level a geometry is read at

Two rules cover every call site.

* **Source-anchored** (`geometry=("source", g)`): the geometry is read
  exactly as an ordinary edge `source.g -> target` would be read there,
  from the same state the edge's value is read from.  Where the value is
  interpolated between two snapshots (a sub-cycled member of a coupling
  group), the geometry is interpolated between the same two snapshots
  with the same weight, componentwise, and the mapping is applied to the
  interpolated value at the interpolated geometry.
* **Target-anchored** (`geometry=("target", g)`): the geometry is field
  `g` of the `state` argument of the hook the boundary inputs are being
  resolved for, `update` or `compute_boundary_fluxes`.

| Call site | Value | Source-anchored geometry | Target-anchored geometry |
|---|---|---|---|
| Plain step, forward edge | the source's value from this step | the source's `g` from this step | the target's pre-step `g` (the state `update` receives) |
| Plain step, back edge | the previous step | the source's `g` of the previous step | the target's pre-step `g` |
| Plain step, the target's flux hook | as for its `update` | as for its `update` | the target's **post-update** `g`: the edges are resolved again for the hook |
| Multi-rate, slow source | the state the source holds between its steps | the same held state | the target's current state |
| Group member's `update`, Gauss-Seidel | the in-pass state | the in-pass `g` | the member's pre-step `g`, at every pass |
| Group member's `update`, Jacobi | the incoming iterate | the incoming iterate's `g` | the member's pre-step `g` |
| Group member's flux, seeded before a pass | the incoming iterate | the incoming iterate's `g` | the incoming iterate's `g` of the member |
| Group member's flux after its update | the in-pass state | the in-pass `g` | the member's in-pass, post-update `g` |
| Sub-cycled member, `boundary_interpolation="linear"`, sub-step `k` of `d` | `v_prev + a (v_cur - v_prev)`, `a = (k + 1) / d` | `g_prev + a (g_cur - g_prev)`, then the mapping | the member's state before this sub-step |
| Sub-cycled member, `"constant"` | the end-of-step value | the end-of-step `g` | the member's state before this sub-step |
| An edge from outside the group | not interpolated | read with the value | by the hook, as above |
| `resolve_boundary_inputs` (inspection) | the current state | the current state's `g` | the node's current `g` |

A target-anchored geometry is therefore the same at every pass of a
coupling solve, because members integrate from the pre-step state; a
source-anchored one is part of the iterate.

The interpolation of a geometry is **componentwise**.  A geometry whose
components are not affine coordinates (a unit quaternion, an angle across
its wrap) should use `boundary_interpolation="constant"`.

### `multilinear_grid`: gather and scatter on a uniform grid

The reference kind, `multilinear_grid_mapping(origin, spacing, shape, *,
n_points, mode, layout="flat")`, transfers between a uniform grid of one
to three axes and `n_points` moving points whose positions, shape
`(n_points, d)`, are the geometry.

**Grid.**  The grid is the lattice of sample points: value
$(i_0, \dots, i_{d-1})$ sits at $x_a = o_a + i_a h_a$,
$0 \le i_a \le n_a - 1$.  The flat index is C order,
$I = \sum_a s_a i_a$ with strides $s_a = \prod_{b > a} n_b$.

**Index arithmetic.**  For point $p$ and axis $a$, in the geometry's
dtype:

$$
u = \frac{x_{p,a} - o_a}{h_a}, \qquad
\bar u = \min(\max(u, 0),\, n_a - 1), \qquad
i^0 = \min(\lfloor \bar u \rfloor,\, n_a - 2), \qquad
t = \bar u - i^0, \qquad
i^1 = \min(i^0 + 1,\, n_a - 1).
$$

$u$ is computed in a static power-of-two frame of $h_a$ (the position, the
origin and the spacing are multiplied by the same power of two, which
changes no bit of the quotient and keeps every intermediate a normal
number), so a spacing far below one is resolved.  On an axis with a single
point $i^0 = i^1 = 0$ and $t = 0$.

**Stencil and weights.**  Each point reads the $2^d$ corners
$c \in \{0, 1\}^d$ of its cell:

$$
I_{p,c} = \sum_a s_a\, i^{c_a}_{p,a}, \qquad
W_{p,c} = \prod_a \bigl(c_a\, t_{p,a} + (1 - c_a)(1 - t_{p,a})\bigr),
\qquad \sum_c W_{p,c} = 1 .
$$

**Gather and scatter are one operator and its transpose.**

$$
\text{consistent (gather):}\quad y_p = \sum_c W_{p,c}\, f_{I_{p,c}},
\qquad
\text{conservative (scatter):}\quad g_j = \sum_{p,c\,:\,I_{p,c} = j} W_{p,c}\, y_p .
$$

Gather reproduces any field that is multilinear in the coordinates, at
every point inside the hull of the lattice.  Scatter preserves the plain
sum, $\sum_j g_j = \sum_p y_p$: it deposits *amounts*.  It divides by no
cell volume and applies no quadrature weight; turning the result into a
density belongs to a node or to the edge's `transform`.  `apply_T` of one
mode is `apply` of the other at the same positions.

**Outside the grid.**  A point outside the hull is clamped to it,
coordinate by coordinate ($\bar u$ above): gather extrapolates constantly,
scatter deposits on the boundary, and the weights still sum to one.  A
non-finite coordinate is not clamped: every weight of that point is NaN at
flat index 0, so gather returns NaN for that point only and scatter puts
NaN in the cells of index 0.

**Derivative with respect to a position.**  Inside a cell,
$\partial y_p / \partial x_{p,a}$ is the slope of the multilinear
interpolant along axis $a$, $\tfrac{1}{h_a}\sum_c \pm W^{(a)}_{p,c} f_{I_{p,c}}$
with $W^{(a)}$ the product over the other axes.  The interpolant has a
kink at every lattice plane; there the derivative is that of the cell
above the plane (of the last cell at the top of the hull), so on the two
faces of the hull it is the interior one-sided derivative, and strictly
outside the hull it is zero.

**Precision.**  Indices and weights are computed in the geometry's dtype
and the weights are cast to the field's dtype, so the result has the
field's dtype: a float32 field with a float64 geometry gives a float32
input computed from float64 positions.  A position $x$ resolves
$\varepsilon |x|$, so on axis $a$ a point is located to

$$
\varepsilon\, \frac{\max(|o_a|,\, |o_a + (n_a - 1) h_a|)}{h_a}
$$

cells.  When that is 1/16 of a cell or more for the dtype the geometry
field has, `compile()` refuses the graph; at 1/1024 or more, `validate()`
reports a warning.  Hold the geometry in float64 or move the origin of the
coordinates closer to the grid.

**Determinism.**  Gather is a fixed left fold over the $2^d$ corners.
Scatter is a scatter-add with repeated indices: on CPU it accumulates in
update order (points, then corners), so the result is a deterministic
function of the inputs including the order of the points, and a
permutation of the points changes it by rounding only.  On an accelerator
a cell that receives several contributions may differ by rounding from
run to run.

### What reads a geometry in 0.4.0, and what does not

The step reads it: the plain step, a coupling group's passes under every
solver and acceleration, sub-cycled members, multi-rate graphs, and the
gradients through all of them.  IQN's automatic interface set includes a
source-anchored geometry that is internal to the group, since it is part
of the iterate.

The **diagnostics do not**.  A coupling group whose pass resolves a
geometry-dependent mapping (on an edge into a member, from inside the
group or from outside) reports its solve, `iterations`,
`total_iterations`, `residual` and `converged`, and withholds everything
built on the float floor or on the contraction estimates: the bounds and
estimates are NaN (`gradient_error_estimate` is `inf`), `ratio_usable`,
`spectral_usable`, `gradient_bound_usable` and `precision_limited` are
`False`, and the entry has a `not_usable_reason` string that says so.
`convergence_norm="interface"` is refused at `compile()` for a group with
a geometry-dependent mapping on an internal edge (the norm would leave
the geometry out, and declare a group converged while its geometry still
moves); use `"l2"` or `"mixed"`.  The adaptive steppers and edges with a
sharded end are refused too.

A graph without a geometry edge is not affected by any of this: it
compiles to the same programs as before the feature existed
(`tests/core/test_step_program_digests.py` compares the lowered program
text of 24 graphs with a capture taken before it).

## Legacy closures

`maddening.core.coupling.interface_mapping.rbf_interpolation` and friends
still return closures usable as `transform=`; they now share the
polynomial-augmented, solve-based matrix construction (`rbf_matrix`).
Prefer `mapping=`: a closure's matrix is a baked constant the graph can
neither differentiate nor replace.
