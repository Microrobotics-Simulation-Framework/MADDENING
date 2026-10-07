# Inspecting a graph

A `GraphManager` can describe itself: its structure as text or as a
diagram, its state, parameters, coupling convergence and memory as
tables, and the software it is running on as a version report.  All of
it is new in 0.4.0 and marked **experimental**: the methods may still
change in a minor release.

Every method on this page is **read-only**.  It does not write the
state, `_meta`, `gm.params`, a node's parameters, the dirty or compiled
flags, the schedule or a cache, and it does not compile or trace the
step.  On a graph that is not ready to be read as it stands -- never
compiled, modified since its last compile, or left holding JAX tracers
by `jax.grad` -- it says so instead of fixing it (see "Graphs that are
not ready" below).
`tests/core/test_inspection_read_only.py` checks this for every method
on every kind of graph by fingerprinting the whole graph around each
call.

Output is **plain text by default, and deterministic**: the same graph
prints the same text on every machine, wrapped at `width` characters
(default 100) rather than at the terminal's width.  Pass `rich=True` to
any `print_*` method to render it with the optional
[rich](https://github.com/Textualize/rich) package
(`pip install maddening[terminal]`); that is the only way those methods
use `rich`, and without the package it raises an `ImportError` naming
the extra.  The one exception to plain text is `print_graph_diagram()`
(at the end of this page), a drawing that needs the same extra.

## The graph at a glance

```python
from maddening import GraphManager
from maddening.nodes import SpringDamperNode

gm = GraphManager()
gm.add_node(SpringDamperNode("left", 0.01, stiffness=30.0, damping=2.0,
                             initial_position=1.5))
gm.add_node(SpringDamperNode("right", 0.01, stiffness=30.0, damping=2.0))
gm.add_node(SpringDamperNode("probe", 0.02, stiffness=5.0, damping=1.0))
gm.add_edge("left", "right", "position", "anchor_position")
gm.add_edge("right", "left", "position", "anchor_position")
gm.add_edge("right", "probe", "position", "anchor_position")
gm.add_coupling_group(["left", "right"], max_iterations=8, tolerance=1e-6)
gm.compile()

assert "[left+right] coupled block" in gm.format_graph()
gm.print_graph()
```

prints

<!-- output -->
```text
GraphManager: 3 nodes, 3 edges, 1 coupling group, 0 external inputs
status: compiled
base timestep: 0.01 (multi-rate)

Nodes (3)
  left  SpringDamperNode
      timestep 0.01, rate divider 1
      coupling group left+right
      state: position float32[], velocity float32[]
  right  SpringDamperNode
      timestep 0.01, rate divider 1
      coupling group left+right
      state: position float32[], velocity float32[]
  probe  SpringDamperNode
      timestep 0.02, rate divider 2
      state: position float32[], velocity float32[]

Edges (3)
  left.position -> right.anchor_position
      state; iterated inside its coupling group
  right.position -> left.anchor_position
      state; iterated inside its coupling group
  right.position -> probe.anchor_position
      state

Coupling groups (1)
  left+right
      members: left, right
      solver ift, acceleration none, mode gauss-seidel
      norm l2 (tolerance 1e-06), max_iterations 8
      diagnostics off, strict off

External inputs (0)
  (none)

Execution order
  1. [left+right] coupled block: left, right
      every base step
  2. probe
      every 2 base steps
```

`gm.format_graph()` returns the same text as a string.  The sections:

- **Nodes**: the type (a wrapper shows what it wraps, as
  `ShardedStencilNode(HeatNode)`), timestep and rate divider, the
  coupling group and how many times a sub-cycled node runs per coupling
  pass, each state field as `dtype[shape]`, and for a sharded node the
  wrapper, the mesh axes and which state axis is split.  A sharded axis
  reads `32@devices`.
- **Edges**: `source.field -> target.input`, whether the source is a state
  field or a boundary flux, the transform's registered name, the mapping
  and its `params["mappings"]` key, `additive`, units, and whether the edge
  is iterated inside a coupling group or is a back edge that reads the
  previous step's value.
- **Coupling groups**: members, solver, acceleration, iteration mode, the
  norm with the tolerance it actually reads (`tolerance` for `"l2"`,
  `atol`/`rtol` for `"mixed"` and `"interface"`), `max_iterations`,
  diagnostics, strict convergence, and any other setting that is not its
  default.
- **External inputs** and the **execution order**: the compiled schedule,
  a coupling group as one block, and how often each block fires.

Long node names wrap onto their own line and are never split.

## Diagrams: Mermaid and Graphviz

`to_mermaid()` and `to_dot()` return the same structure as plain text.
Coupling groups are subgraphs, edges are labelled `field→input`, a flux
edge or external input is dashed, and every name is escaped, so quotes,
angle brackets or newlines in a node name cannot break the diagram.
Nothing here needs Graphviz or Mermaid installed.

<!-- snippet: continues -->
```python
print(gm.to_mermaid())
```

<!-- output -->
```text
flowchart LR
    subgraph g0["coupling group left+right"]
        n0["left<br/>SpringDamperNode"]
        n1["right<br/>SpringDamperNode"]
    end
    n2["probe<br/>SpringDamperNode"]
    n0 -->|"position→anchor_position"| n1
    n1 -->|"position→anchor_position"| n0
    n1 -->|"position→anchor_position"| n2
```

Paste it into GitHub Markdown or the Mermaid live editor.  For Graphviz,
write the DOT text to a file:

<!-- snippet: continues -->
```python
from pathlib import Path

dot = gm.to_dot()
assert dot.startswith("digraph maddening {") and "subgraph cluster_g0" in dot
Path("graph.dot").write_text(dot)
```

and render it with `dot -Tsvg graph.dot -o graph.svg`.  To draw the
structure in the terminal instead, see "Drawing the graph in the
terminal" at the end of this page.

## Tables: state, parameters, memory and coupling

Four methods return an `InspectionTable`, and each has a `print_*`
counterpart that prints it:

| returns the table | prints it |
|---|---|
| `gm.state_summary()` | `gm.print_state_summary()` |
| `gm.params_table()` | `gm.print_params_table()` |
| `gm.memory_estimate()` | `gm.print_memory_estimate()` |
| `gm.coupling_report()` | `gm.print_coupling_report()` |

An `InspectionTable` is a sequence of `dict` rows -- `table[0]["max"]`,
`list(table)`, `pandas.DataFrame(list(table))` -- and `print(table)` shows
it as text.  `table.notes` holds the caveats printed under it,
`table.summary` any table-wide values, and a row's `"flags"` entry the
warnings about that row, which the text prints beneath the table with a
`!`.  The table is a snapshot: editing a row never touches the graph.

### State

<!-- snippet: continues -->
```python
table = gm.state_summary()
print(table)
```

<!-- output -->
```text
State summary: 6 fields in 3 nodes
  node   field     shape  dtype    min  max  mean  nan  inf  bytes
  -----  --------  -----  -------  ---  ---  ----  ---  ---  -----
  left   position  ()     float32  1.5  1.5   1.5    0    0      4
  left   velocity  ()     float32    0    0     0    0    0      4
  probe  position  ()     float32    0    0     0    0    0      4
  probe  velocity  ()     float32    0    0     0    0    0      4
  right  position  ()     float32    0    0     0    0    0      4
  right  velocity  ()     float32    0    0     0    0    0      4

notes:
  - min, max and mean are taken over the finite entries; nan and inf count the rest
```

<!-- snippet: continues -->
```python
row = table[0]
assert (row["node"], row["field"], row["max"]) == ("left", "position", 1.5)
assert all(r["nan"] == 0 and r["inf"] == 0 for r in table)
```

The statistics are computed on the host from a copy of each array.  A
field holding NaN or inf is flagged.  `include_meta=True` adds the
internal `_meta` entries (the multi-rate step counter, the coupling
diagnostics and warm starts).

### Parameters

`params_table()` lists every leaf of `gm.params` with its `ParamSpec`
(the node's own, with `set_param_spec` overrides applied) and checks it
against its bounds by `ParamSpec.check()`'s rule:

<!-- snippet: continues -->
```python
from maddening.core.params import ParamSpec

gm.set_param_spec("left", "damping", ParamSpec(bounds=(0.0, 1.0)))
print(gm.params_table())
```

<!-- output -->
```text
Parameters: 18 leaves
  owner  param             value  shape  dtype    trainable  bounds  transform  out_of_bounds
  -----  ----------------  -----  -----  -------  ---------  ------  ---------  -------------
  left   damping               2  ()     float32  yes        (0, 1)  -          yes
  left   initial_position    1.5  ()     float32  no         (-, -)  -          no
  left   initial_velocity      0  ()     float32  no         (-, -)  -          no
  left   mass                  1  ()     float32  yes        (0, -)  log        no
  left   rest_length           1  ()     float32  yes        (-, -)  -          no
  left   stiffness            30  ()     float32  yes        (0, -)  log        no
  probe  damping               1  ()     float32  yes        (0, -)  -          no
  probe  initial_position      0  ()     float32  no         (-, -)  -          no
  probe  initial_velocity      0  ()     float32  no         (-, -)  -          no
  probe  mass                  1  ()     float32  yes        (0, -)  log        no
  probe  rest_length           1  ()     float32  yes        (-, -)  -          no
  probe  stiffness             5  ()     float32  yes        (0, -)  log        no
  right  damping               2  ()     float32  yes        (0, -)  -          no
  right  initial_position      0  ()     float32  no         (-, -)  -          no
  right  initial_velocity      0  ()     float32  no         (-, -)  -          no
  right  mass                  1  ()     float32  yes        (0, -)  log        no
  right  rest_length           1  ()     float32  yes        (-, -)  -          no
  right  stiffness            30  ()     float32  yes        (0, -)  log        no

flags:
  ! left.damping: out of bounds: above the upper bound 1

notes:
  - value is the scalar for a one-element leaf; an array leaf shows its shape. out_of_bounds applies
    ParamSpec.check()'s rule (non-finite values count)
```

`gm.params` is read as it stands.  Unlike `check_params()`, which
converts a Python float written into `gm.params` to an array in place,
this changes nothing.  On a graph that has never been compiled
`gm.params` is still empty; the table then shows the values `compile()`
would take, and says so.

### Memory

<!-- snippet: continues -->
```python
print(gm.memory_estimate())
```

<!-- output -->
```text
State memory estimate: 40 B in 4 entries
  node   fields  bytes  per_device_bytes  devices  sharding
  -----  ------  -----  ----------------  -------  --------
  left        2      8                 8        1  -
  probe       2      8                 8        1  -
  right       2      8                 8        1  -
  _meta       4     16                16        1  -

state_bytes: 24 (24 B)
meta_bytes: 16 (16 B)
total_bytes: 40 (40 B)
total_per_device_bytes: 40 (40 B)

notes:
  - state memory only, computed from shapes and dtypes: XLA workspace, compiled programs, the copies
    a step makes, scan histories, params and external inputs are not included
  - bytes is a node's global (logical) size; per_device_bytes is what one device holds of it (a
    sharded field's shard, a replicated or unsharded field in full); total_per_device_bytes adds
    those, the most one device holds if every node's share lands on it
```

This counts **state memory only**: it is a floor on what a run needs, not
an estimate of it.  For a sharded node, `per_device_bytes` is one shard
and `devices` the number of devices it is spread over;
`python -m maddening.examples.advanced.sharding_demo` shows it, and the
wrapper in `print_graph()`, on four emulated CPU devices.

### Coupling convergence

`coupling_report()` is `coupling_diagnostics()` as one row per coupling
group, with the documented caveats flagged where they apply: a group
that hit `max_iterations`, `converged=False`, `ratio_usable=False` (the
criterion fell back to the raw residual test, which is then all
`converged` reports), `precision_limited=True` (the residual is rounding,
and `converged` can be true on a stalled iterate), and an unsettled
spectral bound.  A group with no report says why.

<!-- snippet: continues -->
```python
gm.step()
report = gm.coupling_report()
(row,) = report
assert row["group"] == "left+right" and 1 <= row["iterations"] <= 8
gm.print_coupling_report()
```

prints, for this graph,

```text
Coupling report: 1 group
  left+right
      iterations 2, total_iterations 2, max_iterations 8, converged yes, residual 0,
        error_estimate 0, amplification 1, ratio_usable yes, precision_limited yes,
        rho_spectral nan, spectral_error_bound nan, spectral_usable no
      ! precision_limited=True: the residual is at its float floor, so residual and error_estimate
          are rounding and converged can be True on a stalled iterate; read spectral_error_bound
          (solver='ift', diagnostics=True)

notes:
  - error_estimate is an estimate, not a bound: it can understate the distance to the fixed point by
    large factors even with ratio_usable=True; spectral_error_bound (solver='ift', diagnostics=True)
    is the bound where spectral_usable is True, for a linear map (asymptotic for a non-linear one),
    in the group's own norm at the returned state (under 'interface', what each edge delivers: its
    mapping, then its transform). See coupling_diagnostics()
  - multi-rate graph: a group's entry is its most recent applied solve
```

(the counts and residuals are the solver's, so they can differ in the last
digits between JAX versions).  Part 5 of
`python -m maddening.examples.coupling.convergence_diagnostics_demo`
prints the report for a converged group and a capped one, and shows
`strict_convergence=True` raising instead of reporting.  `diagnostics=True` on a `solver="ift"`
group fills `rho_spectral` and `spectral_error_bound`.

**What a node declares for `spectral_usable` at the float floor.**  A group whose residual is at
its float floor (`precision_limited=True`: any converged float32 group at the default tolerance)
reports `spectral_usable=False`, and so `gradient_bound_usable=False`, unless every node in it
declares `update_evaluations()`: there the floor is the whole bound, and the floor counts each
node's evaluations.  No node shipped in `maddening.nodes` declares it in 0.4.0, so such a group of
stock nodes never reads usable at the floor; the numbers are still reported.  On your own node,
return how many sequential sub-steps one `update` takes (`1` for a single explicit step or a
relay, `N` for a loop of `N` sub-steps), and read
[the algorithm guide](../developer_guide/coupling_algorithm_guide.md) for what the count assumes;
a float32 group with a field under a hundredth of what drives it should be run in float64 before
its flag is relied on (MADD-ANO-230).

**What `diagnostics=True` costs.**  It is opt-in per group, and the work is done in every step of
that group.  Beside the solve, the step runs 9 Jacobian-vector products for the spectrum (18 under
`convergence_norm="interface"` with a mapping or a transform on an internal edge, or a field more
than one internal edge reads) and `11 + 4 k + 5 n_p + 2 k n_p` more for the gradient bound
(`k <= 8`, `n_p` the probed constants), then small dense factorisations (QR, linear solves, an
SVD) and one non-symmetric eigenvalue solve of a matrix of at most 9 x 9.  The eigenvalue solve
runs in LAPACK on the host; on a GPU backend it is a device round trip in every step.  Measured
once on a two-spring pair in float32 (an RTX A2000 laptop GPU, jax 0.11.0,
`benchmarks/results/gpu_eigvals_probe/RESULT.md`): a step takes 0.3 ms with diagnostics off and
6 to 8 ms with them on, about 4 ms of it the eigenvalue solve; on CPU the same diagnostics-on step
takes 0.25 to 0.5 ms.

**A report describes the step that ran.**  The bound's float floor is measured on the state the
step returned.  Writing the state afterwards (`set_node_state`, `PUT /graph/state/{node}`) does not
change the entry: it describes that step until the group steps again.  A checkpoint is a copy of the
state, so one saved after a member's state was written holds the written state and not the returned
one.  It says so, and the graph that loads it reports that group's `spectral_error_bound` as NaN and
its `spectral_usable`, `gradient_bound_usable` and `precision_limited` as `False`, with a
`not_usable_reason`, until the group steps.

**A number whose flag is `False` is not a number to compare.**  Where `gradient_bound_usable` is
`False` the value beside it can be finite, `inf` or NaN, and at the float floor it can differ in
kind between backends: on the pair above after 40 float32 steps (residual exactly 0),
`gradient_relative_error_bound` read 4.3e-6 on CPU and NaN on the GPU, with the flag `False` on
both (MADD-ANO-232).

## Graphs that are not ready

None of these methods compiles a graph or puts it back after a
transform.  What each reports instead:

| | never compiled | modified since the last compile | holding JAX tracers |
|---|---|---|---|
| `format_graph`, `to_mermaid`, `to_dot` | structure as registered; rate dividers and execution order "not compiled" | the last compile's schedule, marked stale | shapes and dtypes read from the tracers |
| `print_graph_diagram` | structure as registered | structure as it stands now | structure (it reads no state) |
| `state_summary` | the state `add_node` initialised | the state it holds now | shapes only; no values, with a note |
| `params_table` | the values `compile()` would take, built and not stored | the live `gm.params` | `gm.params` (it is not traced) |
| `coupling_report` | "not compiled" | the last step's report | "state holds tracers" (it does not call `coupling_diagnostics()`, which would put the graph back) |
| `memory_estimate` | node state, no `_meta` yet | the state it holds now | from the tracers' shapes and dtypes |

A graph holds tracers after `jax.grad` of a loss that calls `run_scan`
(see the quickstart).  The next stepping method, or `get_node_state`,
puts it back to the state it had before the transform; inspect it after
that to see values.

## Version report

`maddening.show_versions()` prints what a bug report needs: the
MADDENING version and where it is installed; the Python, platform, JAX,
jaxlib, NumPy, SciPy and lineax versions; the JAX backend, its devices and
whether 64-bit floats are on; `XLA_FLAGS` and the other allowlisted
JAX/XLA/MADDENING environment variables that are set; and which optional
extras are installed.

<!-- snippet: continues -->
```python
import maddening

report = maddening.show_versions(as_dict=True)
assert report["dependencies"]["jax"] and report["jax"]["backend"]
maddening.show_versions()
```

It begins, for example,

```text
MADDENING version information
=============================
maddening       0.4.0  (/path/to/site-packages/maddening)
python          3.12.3 (CPython)
platform        Linux-6.8.0-x86_64-with-glibc2.39
machine         x86_64

dependencies
  jax             0.11.0
  jaxlib          0.11.0
  numpy           2.4.6
  scipy           1.17.1
  lineax          0.0.7

JAX runtime
  backend         cpu
  devices         1 x cpu (cpu)
  x64 enabled     False
```

and goes on to the environment variables and the extras.  The same report
from the command line, as text or JSON:

```bash
python -m maddening info
python -m maddening info --json
```

It prints no secrets: environment variables are read from a fixed
allowlist (`maddening.info.ENV_ALLOWLIST`), so an API token or a cloud
key in the environment never reaches the report.  Extras are located
without being imported, and listing the devices compiles nothing.

## Readable fit and FIM results

`print()` of a `FitResult` (from `fit`, `fit_lm` or
`fit_multiple_shooting`) or a `FIMReport` (from `fim`) gives a short
summary -- whether the fit converged, its losses and the iterate it
returned, the identifiability guard's verdict and the fitted values; the
FIM's rank, conditioning, weakest direction and each parameter's
Cramér–Rao bound -- where `repr()` lists every field, as before.

<!-- snippet: continues -->
```python
import jax.numpy as jnp
from maddening.sysid import fim

info = fim(lambda p: jnp.stack([2.0 * p["k"], p["k"] + p["c"]]),
           {"k": jnp.float32(3.0), "c": jnp.float32(1.0)}, scale=None)
assert str(info).startswith("FIMReport: rank 2 of 2 parameters")
print(info)
```

prints

```text
FIMReport: rank 2 of 2 parameters (all determined); cond 6.854
  least identifiable: ['c'] (weight 0.973 in the weakest direction)
  Cramér–Rao bound on each variance (inf: not identifiable):
    ['c']  1.25
    ['k']  0.25
```

## Drawing the graph in the terminal

`print_graph_diagram()` draws a graph's structure -- what `to_mermaid()`
exports -- as boxes and arrows in the terminal, with the optional
[termaid](https://pypi.org/project/termaid/) package
(`pip install "maddening[terminal]"`).  Each node is a box reading
`name :: Type`, a coupling group is a titled frame, edges carry their
`field→input` labels, and a flux edge or external input is drawn dotted.
With `rich` installed (the same extra) a terminal gets colour:
`theme=` picks one of termaid's themes (`"default"`, `"terra"`, `"neon"`,
`"mono"`, `"amber"`, `"phosphor"`, `"gruvbox"`, `"monokai"`, `"dracula"`,
`"nord"`, `"solarized"`), and a file or pipe gets the same drawing as
plain text.  `direction="TB"` stacks a wide graph vertically, and
`use_ascii=True` draws without Unicode box characters.  The diagram is
for reading: the few characters termaid cannot carry in a label (a
double quote, a backtick, `%%`, `:::`) are drawn as look-alikes, while
`to_mermaid()` and `format_graph()` keep every name exact.

<!-- snippet: requires: termaid -->
```python
from maddening import GraphManager
from maddening.nodes import HeatNode

rods = GraphManager()
for name in ("rod_a", "rod_b"):
    rods.add_node(HeatNode(name, 1e-4, n_cells=8, thermal_diffusivity=0.1,
                           initial_temperature=300.0))
rods.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
              transform="extract_last")
rods.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
              transform="extract_first")
rods.add_coupling_group(["rod_a", "rod_b"])
rods.add_external_input("rod_a", "left_temperature")
rods.print_graph_diagram()
```

prints

<!-- output -->
```text

                         ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
                         │ coupling group rod_a+rod_b                                                                   │
                         │                                                                                              │
                         │                                                                                              │
/────────────────────/   │ ┌─────────────────────┐                                              ┌─────────────────────┐ │
│                    │   │ │                     │                                              │                     │ │
│     external:      │   │ │  rod_a :: HeatNode  │ temperature→left_temperature (extract_last)  │  rod_b :: HeatNode  │ │
│  left_temperature  ├┄┄┄┼►│                     ├─────────────────────────────────────────────►│                     │ │
│                    │   │ │                     │                                              │                     │ │
/────────────────────/   │ └─────────────────────┘                                              └──────────┬──────────┘ │
                         │            ▲temperature→right_temperature (extract_first)                       │            │
                         └────────────┴────────────────────────────────────────────────────────────────────┴────────────┘
```

Like the rest of this page it is read-only and compiles nothing; without
`termaid` it raises an `ImportError` naming the extra.
