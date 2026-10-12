# Upgrading from 0.3.1

This page lists what a graph written for MADDENING 0.3.1 can run into on
0.4.0, in the order you meet it: installing, building the graph,
compiling, stepping, saving and loading, the HTTP API, FMU export,
sharding, and the names that are deprecated.  For each change it says
when it bites, what you see, and what to do.  Where 0.4.0 refuses
something 0.3.1 accepted, the refusal's text is quoted as 0.4.0 raises it,
and a short block shows the refusal and then the fix.  Every Python block
on this page runs as written on 0.4.0.

Most graphs need only the first section and the {term}`coupling <Coupling>` notes under
[Compiling](#compiling).  The [release notes](../release_notes/v0.4.0.md)
list what is new.

## Installing

| Requirement | 0.3.1 | 0.4.0 |
|---|---|---|
| Python | 3.10 or later | 3.12 or later |
| `jax` and `jaxlib` | 0.4 or 0.5 | 0.10 or 0.11 (`>=0.10,<0.13`; there is no 0.12) |
| `lineax` | the `[ift]` extra | a base dependency |
| `scipy` | not declared | a base dependency ({term}`JAX` already installed it) |

The two JAX ranges do not overlap, so move your project to Python 3.12
and JAX 0.10 or 0.11 first, then install 0.4.0:

```bash
python -m pip install "maddening>=0.4,<0.5"
python -m maddening info        # the versions and devices JAX sees
```

- `lineax` is in the base install because the default coupling solver's
  gradient needs it.  `pip install "maddening[ift]"` still works and
  installs nothing more.
- The `[verify]` extra installs only `hypothesis`.
- The `[terminal]` extra adds `termaid`, for `gm.print_graph_diagram()`.
- On an NVIDIA GPU, JAX computes float32 matrix products at reduced
  precision unless you set `JAX_DEFAULT_MATMUL_PRECISION=highest`; see
  [Installation](installation.md).

## Building a graph

The blocks in this section and the next three share one small {term}`node <Node>`, a
relaxation `x <- gain * x_in + 1`, written as a 0.3.1 node would be:

```python
import copy
import warnings

import jax.numpy as jnp

from maddening import GraphManager, SimulationNode
from maddening.core.node import BoundaryInputSpec


class Relax(SimulationNode):
    """x <- gain * x_in + 1."""

    def __init__(self, name, timestep, gain=0.5, **kwargs):
        super().__init__(name, timestep, gain=gain, **kwargs)

    def initial_state(self):
        return {"x": jnp.array(0.0)}

    def boundary_input_spec(self):
        return {"x_in": BoundaryInputSpec(shape=())}

    def update(self, state, boundary_inputs, dt):
        return {"x": self.params["gain"] * boundary_inputs.get("x_in", 0.0) + 1.0}


def pair(dt_a=0.01, dt_b=0.01):
    """Two Relax nodes, each feeding the other."""
    gm = GraphManager()
    gm.add_node(Relax("a", dt_a))
    gm.add_node(Relax("b", dt_b))
    gm.add_edge("a", "b", source_field="x", target_field="x_in")
    gm.add_edge("b", "a", source_field="x", target_field="x_in")
    return gm


class refused:
    """`with refused(ValueError) as r:` -- the block must raise; r.text is why."""

    def __init__(self, error):
        self.error, self.text = error, ""

    def __enter__(self):
        return self

    def __exit__(self, kind, err, traceback):
        if kind is None:
            raise AssertionError(f"no {self.error.__name__} was raised")
        if not issubclass(kind, self.error):
            return False                     # another error: let it through
        self.text = str(err)
        return True
```

Your own nodes keep working.  Their constants are not parameters a fit or
a REST write can reach until the node reads them from `params`:
`gm.nodes_without_params()` lists the nodes that still bake them in, and
[where a constant belongs](../developer_guide/node_authoring.md#where-a-constant-belongs-and-what-a-saved-graph-carries)
shows the change.

### `add_node`: timesteps and names

A node's timestep must be a finite number above zero, and some names are
refused: `_meta`, `_params`, `_params_mappings` and `_reports` (keys the
graph keeps for itself), an empty name, one containing `/`, `#` or `->`,
the spellings `NaN`, `Infinity` and `-Infinity`, and one holding a control
character other than tab, line feed and carriage return.  0.3.1 refused
only a name already in the graph.

```text
ValueError: Node 'b1' has timestep 0.0: a node's timestep must be a finite number > 0.
ValueError: Node name '_meta' is invalid: it is a key the graph reserves for its own state and checkpoints (_meta, _params, _params_mappings, _reports).  A different spelling ('meta', say) is fine.
```

<!-- snippet: continues -->
```python
gm = GraphManager()
with refused(ValueError) as r:
    gm.add_node(Relax("b1", 0.0))
assert "must be a finite number > 0" in r.text
with refused(ValueError) as r:
    gm.add_node(Relax("_meta", 0.01))
assert "reserves for its own state" in r.text

gm.add_node(Relax("b1", 0.01))           # a positive timestep
gm.add_node(Relax("meta", 0.01))         # another spelling
assert gm.node_names == ["b1", "meta"]
```

### A second declaration of an external input

`add_external_input` for a field that is already declared with another
shape or dtype raises; 0.3.1 kept the second declaration.  Declaring it
again with the same shape and dtype changes nothing.  `GraphManager.from_dict`
and `load_graph_from_usd` refuse a file that holds two different
declarations (see [Saving and loading](#saving-and-loading)).

```text
ValueError: external input 'a.x_in' is already declared, with shape () and dtype float32 (an earlier add_external_input call, or the file this graph was loaded from), and add_external_input('a', 'x_in', ...) declares it again with shape (3,) and dtype float32. [...] Declare the input once, with the shape and dtype it is to run in: there is no remove_external_input, so leave the other declaration out where the graph is built (or out of the file it is loaded from), or take the node out with remove_node, which drops its external inputs, and add it again. Until 0.4.0 the second declaration was taken and the graph ran the last one, so keeping the last keeps what it did. Declaring it again with the same shape and dtype is accepted and changes nothing.
```

<!-- snippet: continues -->
```python
gm = GraphManager()
gm.add_node(Relax("a", 0.01))
gm.add_external_input("a", "x_in", shape=())
gm.add_external_input("a", "x_in", shape=())          # the same: accepted
with refused(ValueError) as r:
    gm.add_external_input("a", "x_in", shape=(3,))
assert "is already declared, with shape ()" in r.text
```

### Coupling-group options are checked

`add_coupling_group` refuses an option it cannot run with: a boolean
option given something that is not `True` or `False`, `rtol=0` under
`convergence_norm="mixed"` or `"interface"` (which divide by it), and
`waveform_iterations` below 1.

```text
TypeError: CouplingGroup.diagnostics=1: it must be True or False, not int.
ValueError: CouplingGroup.rtol=0.0 under convergence_norm='mixed': it must be > 0.  This norm divides each field's change by rtol * max|field| (atol is a dead band below which a field is not read, not an absolute tolerance), so at 0 the residual is 0/0 and nothing converges.
ValueError: CouplingGroup.waveform_iterations=0: it must be an integer >= 1. A sub-cycling group runs this many waveform sweeps per step, and at 0 it ran none: its members never stepped.
```

<!-- snippet: continues -->
```python
for error, options in ((TypeError, {"diagnostics": 1}),
                       (ValueError, {"convergence_norm": "mixed", "rtol": 0.0}),
                       (ValueError, {"waveform_iterations": 0})):
    with refused(error) as r:
        pair().add_coupling_group(["a", "b"], **options)
    print(error.__name__, r.text.split(":")[0])

pair().add_coupling_group(["a", "b"], diagnostics=True,
                          convergence_norm="mixed", rtol=1e-6,
                          waveform_iterations=1)
```

### `copy.copy(gm)` raises

A shallow copy shared the graph's nodes and parameters but not its
compiled step.  Use `copy.deepcopy`.

```text
TypeError: copy.copy() of a GraphManager is refused: a shallow copy would share the graph's nodes, parameter tree and record of node.params writes but not its compiled step, so after a write one of the two graphs would run a value neither reports.  Use copy.deepcopy(gm) for a graph of its own.
```

<!-- snippet: continues -->
```python
gm = pair()
with refused(TypeError) as r:
    copy.copy(gm)
assert "copy.deepcopy(gm)" in r.text
twin = copy.deepcopy(gm)
assert twin.node_names == gm.node_names
```

### `remove_node` takes the node out of its coupling group

In 0.3.1 a group went on naming a removed node, and every compile and step
failed until a node of that name came back.  Now the node leaves the
group, and a group left with one member is removed, with a `UserWarning`.
To rebuild a member and keep its group, add the group again after the
node and its edges.

```text
UserWarning: Removing node 'b' removed the coupling group of ['a', 'b'] with it (one member would remain).  A node added back under the name is not a member: add the group again (add_coupling_group) after the node and its edges.
```

<!-- snippet: continues -->
```python
gm = pair()
gm.add_coupling_group(["a", "b"])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gm.remove_node("b")
assert "removed the coupling group" in str(caught[0].message)

gm.add_node(Relax("b", 0.01))                          # rebuild the member,
gm.add_edge("a", "b", source_field="x", target_field="x_in")
gm.add_edge("b", "a", source_field="x", target_field="x_in")
gm.add_coupling_group(["a", "b"])                      # then the group
gm.compile()
```

### Built-in nodes

- **`HeatNode`** refuses an unstable timestep at construction.  Keep
  `timestep * thermal_diffusivity / dx**2` at or below 1/2 (the default
  `stencil_order=2`) or 5/16 (`stencil_order=4`).  It also applies its
  boundary temperatures at the rod ends now (so `T[0]` is no longer equal
  to `left_temperature`; read the value you supplied), evolves an end with
  no boundary data under a zero-flux closure, and reports the flux at the
  end, so a coupled interface fitted to the old flux needs fitting again.
  `boundary_flux_spec()` declares `K*m/s`: the flux is `-alpha dT/dx`, so
  do not multiply it by `rho * c_p` to convert units.
- **`LBMPipeNode`** refuses a propeller outside its grid.  The default
  `propeller_x=10` needs `nx > 10`; pass an in-grid `propeller_x` for a
  shorter pipe.  The lattice Boltzmann constructors refuse a non-finite
  viscosity or relaxation time.
- **`LBMNode.meta.algorithm_id`** is `MADD-NODE-011`.  `RigidBodyNode`
  keeps `MADD-NODE-007`, which the two used to share.

```text
ValueError: timestep 1.0 is unstable for this rod: the Fourier number dt*alpha/dx^2 is 100, above the 0.5 limit of the order-2 stencil (dx = length/n_cells = 0.1, alpha = 1.0).  The explicit update diverges to NaN there rather than losing accuracy gracefully.  Use timestep <= 0.005, or more cells, or a smaller thermal_diffusivity.
ValueError: propeller_x=10 is outside the grid: the pipe has nx=8 planes along x, indexed 0 to 7, so the propeller disc would cover no cell and exert no force.  Pass a propeller_x in [0, 8) (the default, 10, needs nx > 10).
```

<!-- snippet: continues -->
```python
from maddening.nodes import HeatNode, LBMPipeNode

with refused(ValueError) as r:
    HeatNode("rod", 1.0, n_cells=10, length=1.0, thermal_diffusivity=1.0)
assert "Use timestep <= 0.005" in r.text
HeatNode("rod", 0.005, n_cells=10, length=1.0, thermal_diffusivity=1.0)

with refused(ValueError) as r:
    LBMPipeNode("pipe", 1.0, nx=8, ny=8, nz=8)
assert "propeller_x=10 is outside the grid" in r.text
LBMPipeNode("pipe", 1.0, nx=8, ny=8, nz=8, propeller_x=4)
```

## Compiling

### An edge and an external input on one field

A declared external input replaces whatever an edge delivers to its field
on every step, with zeros when it is not fed, so the edge never reached
the node.  `compile()` now raises, and `gm.validate()` lists the same text
as an error.  Keep one of the two: leave out the `add_external_input`
call where the graph is built (there is no `remove_external_input`), or
remove the edge.

```text
ValueError: an edge and a declared external input target the same field: edge a.x -> b.x_in and external input b.x_in. A graph with both on one field is not compiled: an external input replaces whatever an edge delivers to its field on every step, with zeros when it is not fed, so the edge would never reach the node. [...] Keep one of the two. To use the edge, do not declare the external input (there is no remove_external_input: leave add_external_input('b', 'x_in', ...) out where the graph is built). To use the external input, remove the edge: remove_edge('a', 'b', 'x', 'x_in'). [...]
```

<!-- snippet: continues -->
```python
def chain(declare_input):
    gm = GraphManager()
    gm.add_node(Relax("a", 0.01))
    gm.add_node(Relax("b", 0.01))
    gm.add_edge("a", "b", source_field="x", target_field="x_in")
    if declare_input:
        gm.add_external_input("b", "x_in")
    return gm


with refused(ValueError) as r:
    chain(declare_input=True).compile()
assert "target the same field" in r.text
chain(declare_input=False).compile()       # keep the edge: no declaration
```

### The default coupling solver, and what `converged` means

- Coupling groups default to `solver="ift"`: a step's coupling loop stops
  when the group has converged instead of after a fixed number of passes,
  and its gradient is taken by the implicit function theorem.
  `solver="fori"` still works, warns, and goes in the next minor release.
- `converged=True` means "within tolerance of the fixed point": the test
  applies to an error estimate, `residual / (1 - rho)`, not to the last
  change alone.  Expect a different `residual` and sometimes more passes.
  A group that now stops at `max_iterations` reports `converged=False`:
  raise the cap or loosen the tolerance on purpose.
- Under `convergence_norm="mixed"` and `"interface"`, `tolerance` and
  `rtol` are relative to each field's magnitude, and `atol` is a dead
  band, not an absolute tolerance.  Leave `atol` at its default `0.0`:
  `compile()` warns (`UserWarning`) for every group that sets it above
  zero, and under `-W error` that warning raises.
- Recorded trajectories of groups with `acceleration="aitken"` or an IQN
  acceleration move.

Here the pair needs eleven passes: at `max_iterations=10` it stops short
and says so.

<!-- snippet: continues -->
```python
for cap in (10, 20):
    gm = pair()
    gm.add_coupling_group(["a", "b"], max_iterations=cap)
    gm.compile()
    gm.step()
    report = gm.coupling_diagnostics()["a+b"]
    print(f"max_iterations={cap}: {report['iterations']} passes, "
          f"converged={report['converged']}")
assert report["converged"] and report["iterations"] == 11

gm = pair()
gm.add_coupling_group(["a", "b"], convergence_norm="mixed", atol=1e-6)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gm.compile()
assert any("Set atol=0.0" in str(w.message) for w in caught)
```

```text
max_iterations=10: 10 passes, converged=False
max_iterations=20: 11 passes, converged=True
```

### Loops a coupling group cannot iterate

These configurations failed inside the step in 0.3.1, or iterated
something the convergence test could not see.  `compile()` refuses them:

- **A flux edge inside a group that the loop reads from the state.**  A
  flux is what `compute_boundary_fluxes` returns; it is not a field of the
  producer's state.  `convergence_norm="interface"` measures the values
  internal edges carry, and a sub-cycled member's `"linear"` or
  `"quadratic"` boundary interpolation interpolates them, so neither can
  read a flux.  Use `"mixed"` or `"l2"` for the norm, `"constant"` for the
  interpolation, or carry the quantity in a state field.
- **Integer, Boolean or PRNG-key fields computed from one another around
  a closed loop inside a group.**  The iteration reads its convergence
  from floating-point fields, so such fields could still be changing when
  the group reported `converged=True`.  Store one field of the loop in
  floating point, compute it in a node outside the group, or set
  `max_iterations=1`.

```text
RuntimeError: Cannot compile graph with errors:
ERROR: coupling group ['k', 's'] uses convergence_norm='interface', which measures the values the group's internal edges carry, but edge 's.spring_force->k.x_in' carries the boundary flux 'spring_force', which 's' computes in compute_boundary_fluxes and does not hold in its state, so the norm cannot read it.  Use convergence_norm='mixed' or 'l2', which measure the state the flux is computed from.
ValueError: Coupling group ['c1', 'c2']: non-floating fields are computed from one another in a closed loop: c1.n (int32) -> c2.n (int32) -> c1.n (int32). [...] Store one field of the loop in floating point (the residual then measures it), or compute it in a node outside the group, or set max_iterations=1 (one staggered pass, which iterates nothing).
```

<!-- snippet: continues -->
```python
from maddening.nodes import SpringDamperNode


def spring_loop(**options):
    """A spring whose force (a flux) drives a Relax node that moves its anchor."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.add_node(Relax("k", 0.01))
    gm.add_edge("s", "k", source_field="spring_force", target_field="x_in")
    gm.add_edge("k", "s", source_field="x", target_field="anchor_position")
    gm.add_coupling_group(["s", "k"], **options)
    return gm


with refused(RuntimeError) as r:
    spring_loop(convergence_norm="interface").compile()
assert "carries the boundary flux 'spring_force'" in r.text
spring_loop(convergence_norm="mixed").compile()


class Counter(SimulationNode):
    """Counts the steps of its neighbour: n <- n_in + 1, an integer."""

    def initial_state(self):
        return {"n": jnp.array(0, dtype=jnp.int32), "t": jnp.array(0.0)}

    def boundary_input_spec(self):
        return {"n_in": BoundaryInputSpec(shape=(), dtype=jnp.int32)}

    def update(self, state, boundary_inputs, dt):
        n_in = boundary_inputs.get("n_in", jnp.array(0, dtype=jnp.int32))
        return {"n": n_in + 1, "t": state["t"] + dt}


def counters(**options):
    gm = GraphManager()
    gm.add_node(Counter("c1", 0.01))
    gm.add_node(Counter("c2", 0.01))
    gm.add_edge("c1", "c2", source_field="n", target_field="n_in")
    gm.add_edge("c2", "c1", source_field="n", target_field="n_in")
    gm.add_coupling_group(["c1", "c2"], **options)
    return gm


with refused(ValueError) as r:
    counters().compile()
assert "computed from one another in a closed loop" in r.text
counters(max_iterations=1).compile()        # one staggered pass
```

### Multi-rate graphs keep every clock

- **A sub-cycled member's timestep must divide the group's largest.**  A
  member of a group with `subcycling=True` takes a whole number of
  sub-steps per pass; in 0.3.1 one whose timestep did not divide the
  group's largest covered less time than the rest of the graph, silently.
- **Timesteps must have a usable common step.**  The graph steps every
  node on a common base step.  Timesteps more than about 1e9 apart, or
  with no common step the computation finds, no longer compile.
- **`gm.timestep` is the time one `step()` advances.**  On a graph with a
  sub-cycling group it is the group's largest member timestep.  Compute a
  step count as `duration / gm.timestep`; the live runner's clock, a USD
  stage's `baseDt` and an FMU's default step follow it.

```text
RuntimeError: Cannot compile graph with errors:
ERROR: coupling group ['a', 'b']: node 'b' has timestep 0.003, which does not divide the group's macro timestep 0.01 (ratio 3.33333).  A sub-cycled node takes a whole number of sub-steps per coupling pass, so it would take 3 of 0.003 and cover 0.009 per macro step instead of 0.01.  Give it a timestep that divides 0.01 -- the nearest are 0.00333333 (0.01/3) and 0.0025 (0.01/4) -- or change the macro timestep.
ValueError: nodes 'fast' (timestep 1e-10) and 'slow' (timestep 1.0) cannot be scheduled together: it is not above 1e-09 of the largest timestep, which the common step (1.0) is computed to, so it would be stepped once per step of 1.0 and advance its own 1e-10 each time.  Give the nodes timesteps that are whole multiples of one common step, no more than about 1e+09 times apart.
```

<!-- snippet: continues -->
```python
gm = pair(0.01, 0.003)
gm.add_coupling_group(["a", "b"], subcycling=True)
with refused(RuntimeError) as r:
    gm.compile()
assert "does not divide the group's macro timestep" in r.text

gm = pair(0.01, 0.0025)                     # 0.0025 divides 0.01
gm.add_coupling_group(["a", "b"], subcycling=True)
gm.compile()
assert gm.timestep == 0.01                  # one step() advances 0.01

gm = GraphManager()
gm.add_node(Relax("slow", 1.0))
gm.add_node(Relax("fast", 1e-10))
gm.add_edge("slow", "fast", source_field="x", target_field="x_in")
with refused(ValueError) as r:
    gm.compile()
assert "cannot be scheduled together" in r.text
```

## Stepping

### A step that changes the shape of the state

In 0.3.1 a step that returned a state leaf of another shape or dtype kind
was stored, and the graph could then not be saved and reloaded, reset or
scanned.  The step now raises `ValueError` and stores nothing.  A node's
`update()` must return each field with the shape and kind of dtype
`initial_state()` gave it.

```text
ValueError: one step changes the layout of the graph's state, so the stepped state could not be saved and reloaded, reset or scanned: 'g/x' has shape (2,) before the update and (3,) after it (leaves are named node/field).  A node's update() must return its state with the shapes and kinds of dtype initial_state() gave it: check the node's constructor values and the shapes its edges and external inputs deliver.  Nothing was stored.
```

<!-- snippet: continues -->
```python
class Grow(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros(2)}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.zeros(3)}       # not the shape initial_state() gave


gm = GraphManager()
gm.add_node(Grow("g", 0.01))
gm.compile()
with refused(ValueError) as r:
    gm.step()
assert "'g/x' has shape (2,) before the update and (3,) after it" in r.text
assert gm.get_node_state("g")["x"].shape == (2,)      # nothing was stored
```

### Results that move

These change numbers without raising anything:

- **`run_sweep` and `run_scan` may differ by about one ulp.**  Constants
  are no longer compiled into the step, so {term}`XLA` may fuse a product
  differently in the two compiled shapes.  Loosen a test that asserts the
  two are bitwise equal to a few ulps.
- **Coupling groups** iterate to a different stopping point (see
  [above](#the-default-coupling-solver-and-what-converged-means)), and
  under `convergence_norm="interface"` every member is computed from
  readings that met the criterion.
- **On a {term}`multi-rate <Multi-rate>` graph a slow node's flux** is the flux of the state it
  holds.
- **`HeatNode`** boundary values and fluxes (see
  [Built-in nodes](#built-in-nodes)).

## Saving and loading

### `to_dict()` returns a copy

Editing the dictionary `to_dict()` returns no longer changes the graph it
came from.  Build a variant from the copy with `GraphManager.from_dict`.

<!-- snippet: continues -->
```python
gm = GraphManager()
gm.add_node(Relax("a", 0.01, gain=0.25))
config = gm.to_dict()
config["nodes"][0]["params"]["gain"] = 0.9
assert gm.get_node("a").params["gain"] == 0.25      # the graph is unchanged
variant = GraphManager.from_dict(config, {"Relax": Relax})
assert variant.get_node("a").params["gain"] == 0.9
```

### `to_dict()` warns of a constructor argument it does not save

A saved node is rebuilt as `cls(name=..., timestep=..., **params)`, so a
constructor argument kept as an attribute instead of passed to
`super().__init__` is lost on reload.  `to_dict()` (and
`save_graph_to_usd`) now warn.  Pass every constructor argument to
`super().__init__` and read it from `self.params`.

```text
UserWarning: node 'k' (Kept) does not save the constructor argument gain.  Kept's constructor (or one it inherits) does not hand it to SimulationNode.__init__ under that name, so it is not in node.params, and a saved node is rebuilt as Kept(name=..., timestep=..., **params): a graph loaded from this save runs with the default gain=0.5.  Hand it on -- super().__init__(name, timestep, gain=gain) -- and read it from self.params; or, if the node saves it another way, override to_dict() to say so.
```

<!-- snippet: continues -->
```python
class Kept(SimulationNode):
    def __init__(self, name, timestep, gain=0.5):
        super().__init__(name, timestep)
        self.gain = gain                 # kept outside params: not saved

    def initial_state(self):
        return {"x": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + self.gain * dt}


gm = GraphManager()
gm.add_node(Kept("k", 0.01, gain=2.0))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gm.to_dict()
assert "does not save the constructor argument gain" in str(caught[0].message)

gm = GraphManager()
gm.add_node(Relax("k", 0.01, gain=2.0))  # Relax hands gain to super().__init__
with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert gm.to_dict()["nodes"][0]["params"] == {"gain": 2.0}
```

### Loading a file

- **Two different declarations of one external input** in a config's
  `external_inputs` are refused; delete the entry the graph is not to run
  with.  A graph saved by 0.3.1 after a second `add_external_input` holds
  both, and ran the later one.
- **A node saved inside a sharded wrapper** is rebuilt unsharded, as the
  node it wraps, with a `UserWarning` naming the call that wraps it again:
  a config does not carry the device mesh.  0.3.1 dropped the wrapper
  without a word.

```text
ValueError: external input 'a.x_in' is already declared, with shape () and dtype float32 (external_inputs[0] of this config), and external_inputs[1] declares it again with shape (3,) and dtype float32. [...] Delete from the config's "external_inputs" the entry the graph is not to run with, and load it again. A graph saved before the second declaration was refused (0.4.0) could hold both, and ran the later one. Declaring it again with the same shape and dtype is accepted and changes nothing.
```

<!-- snippet: continues -->
```python
config = {
    "nodes": [{"type": "Relax", "name": "a", "timestep": 0.01,
               "params": {"gain": 0.5}}],
    "edges": [],
    "external_inputs": [
        {"target_node": "a", "target_field": "x_in", "shape": [], "dtype": "float32"},
        {"target_node": "a", "target_field": "x_in", "shape": [3], "dtype": "float32"},
    ],
}
with refused(ValueError) as r:
    GraphManager.from_dict(config, {"Relax": Relax})
assert "is already declared" in r.text
del config["external_inputs"][1]          # the shape the graph is to run with
gm = GraphManager.from_dict(config, {"Relax": Relax})
gm.compile()
```

### Loading a USD stage

`load_graph_from_usd` builds only the built-in node classes and the ones
you allow: a stage names the Python class of every node, so importing
whatever it names would run code from the stage.  Pass the classes as
`node_registry=`, call `register_node_class` once, or pass
`allow_import=True` for a stage you trust as much as a script.

```text
KeyError: "Node class '__main__.Relax' is not allowed by this load. A USD stage is untrusted input and names its own Python classes, so only classes the caller allows are instantiated.  Pass it in: load_graph_from_usd(stage, node_registry={'__main__.Relax': Relax}), or call register_node_class(Relax) first.  To go back to importing whatever the stage names -- which runs '__main__' at import time and is safe only for a stage you trust as much as a script -- pass allow_import=True."
```

<!-- snippet: requires: pxr -->
```python
import jax.numpy as jnp
from pxr import Usd

from maddening import GraphManager, SimulationNode
from maddening.usd import load_graph_from_usd, save_graph_to_usd


class Relax(SimulationNode):
    def __init__(self, name, timestep, gain=0.5, **kwargs):
        super().__init__(name, timestep, gain=gain, **kwargs)

    def initial_state(self):
        return {"x": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + self.params["gain"] * dt}


gm = GraphManager()
gm.add_node(Relax("a", 0.01, gain=0.25))
stage = Usd.Stage.CreateInMemory()
save_graph_to_usd(gm, stage)
try:
    load_graph_from_usd(stage)
except KeyError as err:
    assert "is not allowed by this load" in str(err)
else:
    raise AssertionError("the stage's own class was built")
loaded = load_graph_from_usd(stage, node_registry={"Relax": Relax})
assert loaded.get_node("a").params["gain"] == 0.25
```

## The HTTP API

The changes below close the network exposure 0.3.1 had.  [Serving the
REST API on a network](serving_the_rest_api.md) walks through a
deployment with the token.

- **A server not bound to loopback asks every request for
  `Authorization: Bearer <token>`**, except `/healthz` and the bundled
  `/viz/*` pages.  Set `MADDENING_API_TOKEN`, or read the token the server
  logs once at start-up, and pass `SimulationServer(bind_host=...)` (or
  set `MADDENING_HOST`) so the server knows which rule applies.
- **A loopback-bound server asks the token of any peer that is not a
  direct loopback connection**, and of a request carrying
  `X-Forwarded-For` or `Forwarded`.  FastAPI's in-process `TestClient`
  reports its peer as `"testclient"`, so a test suite that used
  `TestClient(app)` gets 401: give it the token, or build it as a loopback
  client.  Serve with uvicorn's `--no-proxy-headers`.
- **Checkpoint paths are relative to `checkpoint_root`**
  (`SimulationServer(checkpoint_root=...)`, by default `./checkpoints`),
  and a path that leaves it is refused.
- **Request bodies are strict.**  A number field given `true`, `"0.5"` or
  `" 0.25 "` is a 422 (a parameter value in `PUT /graph/params/{node}`, a
  400), and `POST /graph/edges` and `POST /graph/nodes` refuse a key they
  do not apply with 422.  An edge with `transform`, `additive`, `mapping`
  or `geometry` is added in Python with `gm.add_edge`.
- **`POST /graph/nodes` answers 400 where 0.3.1 answered 201** for a name
  containing `/`, `#` or `->`, an empty name, the names `NaN`, `Infinity`
  and `-Infinity`, and a `HeatNode` timestep above its stencil's stability
  limit; a repeated request for one of them is a 400, not a 409.  An array dimension over the cap is a
  422.
- **A cloud launch opens no API port by default** (`JobConfig.ports` is
  `[]`).  Reach the API over `ssh -L 8000:127.0.0.1:8000`, or set
  `ports=[8000]` and `MADDENING_API_TOKEN`.
- **The {term}`WebRTC` signaling server checks the client's token**; in 0.3.1 it
  admitted every client.

```text
{"detail": "This server was configured for a loopback bind ('127.0.0.1'), and a token is needed without a direct connection from a loopback address: the request arrived from 'testclient', which is not a loopback address (an in-process client such as Starlette's TestClient reports a name: present server.auth.token, or construct it with base_url='http://127.0.0.1' and client=('127.0.0.1', <port>)). Present 'Authorization: Bearer <token>'. If the bind really is public, pass bind_host to SimulationServer (or set MADDENING_HOST) so the token is logged at start-up."}
```

<!-- snippet: requires: fastapi httpx -->
```python
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.nodes import BallNode, TableNode

server = SimulationServer(node_registry={"BallNode": BallNode, "TableNode": TableNode},
                          checkpoint_root="checkpoints")
app = server.create_app()

# 0.3.1 test code: TestClient(app) -- now refused.
assert TestClient(app).get("/graph").status_code == 401

# Either present the server's token...
client = TestClient(app, headers={"Authorization": f"Bearer {server.auth.token}"})
assert client.get("/graph").status_code == 200
# ...or build the client as a direct loopback connection.
client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
assert client.get("/graph").status_code == 200
```

<!-- snippet: continues -->
```python
node = {"type": "BallNode", "name": "ball", "timestep": 0.01}
assert client.post("/graph/nodes", json=node).status_code == 201
client.post("/graph/nodes", json={"type": "TableNode", "name": "table", "timestep": 0.01})
edge = {"source_node": "table", "target_node": "ball",
        "source_field": "position", "target_field": "table_position"}

reply = client.post("/graph/edges", json={**edge, "additive": True})
assert reply.status_code == 422                       # a key the route does not apply
print(reply.json()["detail"][0]["loc"], reply.json()["detail"][0]["msg"])
assert client.post("/graph/edges", json=edge).status_code == 201

reply = client.post("/graph/nodes", json={**node, "name": "b2", "timestep": "0.5"})
assert reply.status_code == 422                       # a number sent as a string
reply = client.post("/graph/nodes", json={**node, "name": "a/b"})
assert reply.status_code == 400                       # 201 in 0.3.1
print(reply.json()["detail"])

assert client.post("/checkpoint/save?path=run1.npz").status_code == 200
reply = client.post("/checkpoint/save?path=../outside.npz")
assert reply.status_code == 400
print(reply.json()["detail"])
```

```text
['body', 'additive'] Extra inputs are not permitted
Node name 'a/b' is invalid: must be non-empty and must not contain ['/']
checkpoint path must stay under the checkpoint root, naming a file inside it (got '../outside.npz')
```

### ZeroMQ transports

`NetworkRelay`, `CommandPublisher` and the multi-job `Coordinator` bind to
`127.0.0.1` by default; 0.3.1 bound every interface.  A remote subscriber
stops receiving until you either tunnel to the loopback port over SSH or
bind a non-loopback address.  Such a bind is encrypted with {term}`ZMQ` CURVE and
refuses to start without a shared secret: set `MADDENING_TRANSPORT_TOKEN`
on both ends, to a value different from `MADDENING_API_TOKEN`.
[ZeroMQ transports](zeromq_transports.md) covers both routes.

## FMU export

- **`FMIVariable` is keyword-only.**  0.4.0 added `node` and `field`
  before `shape`, so a positional call written for 0.3.1 would have bound
  the wrong fields; it now raises `TypeError`.  Build it with keywords.
- **`build_model_description` refuses what it used to round or skip**: a
  `selected_outputs` pair that names no state field, and a
  `default_step_size` that is not a whole number of graph steps.
- **FMU-state blobs are `npz`, not pickle.**  A snapshot written before
  0.4.0 is refused rather than unpickled; regenerate any blob you stored.

```text
TypeError: FMIVariable.__init__() takes 1 positional argument but 9 were given
ValueError: selected_outputs names output(s) this FMU cannot export, which would be left out of it without a word: ('ball', 'height'): node 'ball' has no state field 'height' (its state fields: ['position', 'velocity']).  Fix the name, or leave the pair out.
ValueError: default_step_size=0.015 is not a whole number of steps of graph_timestep=0.01: it is 1.5 of them, and the nearest whole number, 2, is 0.02 [...]
```

```python
import warnings

import jax
import jax.numpy as jnp

from maddening import GraphManager
from maddening.fmi.model_description import FMIVariable, build_model_description
from maddening.nodes import BallNode, TableNode


class refused:
    """`with refused(ValueError) as r:` -- the block must raise; r.text is why."""

    def __init__(self, error):
        self.error, self.text = error, ""

    def __enter__(self):
        return self

    def __exit__(self, kind, err, traceback):
        if kind is None:
            raise AssertionError(f"no {self.error.__name__} was raised")
        if not issubclass(kind, self.error):
            return False                     # another error: let it through
        self.text = str(err)
        return True


with refused(TypeError) as r:
    FMIVariable("plant.x", 1, "float32", "output", "continuous", "", "m", (3,))
print(r.text)
variable = FMIVariable(name="plant.x", value_reference=1, dtype="float32",
                       causality="output", variability="continuous",
                       unit="m", shape=(3,))
assert variable.shape == (3,)

gm = GraphManager()
gm.add_node(BallNode("ball", 0.01))
gm.add_node(TableNode("table", 0.01))
gm.add_edge("table", "ball", source_field="position", target_field="table_position")
gm.compile()
for kwargs in ({"selected_outputs": [("ball", "height")]},
               {"default_step_size": 0.015}):
    with refused(ValueError) as r:
        build_model_description(gm, model_name="Ball", **kwargs)
    print(r.text.split(":")[0])
build_model_description(gm, model_name="Ball", selected_outputs=[("ball", "position")],
                        default_step_size=0.02)
```

## Sharding

### The sharded wrappers refuse settings that gave wrong answers

- `ShardedStencilNode` takes one boundary mode for every axis, as a
  string; a per-axis dictionary and an empty `axis_map` are refused.
- Around a `HeatNode` it takes only `boundary="edge"` (the default):
  `"zero"` and `"periodic"` are refused, since the sharded node would have
  computed another model than the unsharded one.  A cold end is a
  boundary input, as unsharded: the `left_temperature` and
  `right_temperature` inputs (in a graph, `gm.add_external_input(<node>,
  "left_temperature")`, which defaults to `0.0`).
- Around an `LBMNode` it takes only `boundary="periodic"`, which you pass
  (the default `"edge"` is refused).  A `heat_source`
  or `body_force` of a shape the unsharded node refuses is refused too.
- `ShardedUnstructuredNode` refuses a Cartesian stencil node (one with a
  halo), a per-cell state that is not one row per layout cell, and a
  per-cell input the length of one shard's slab.

```text
ValueError: ShardedStencilNode: HeatNode 'rod' declares halo_boundary() == 'edge', the one fill of the halos at the edges of the global grid it takes, but was given boundary='zero'.  With another fill the sharded node would silently compute a different model from the unsharded one, or silently ignore the fill.  Pass boundary='edge'.  A HeatNode end held at a temperature is a boundary input, exactly as unsharded: pass left_temperature=0.0 / right_temperature=0.0 for a Dirichlet T=0 end (in a graph, gm.add_external_input(<node>, 'left_temperature') and 'right_temperature', which default to 0.0).  There is no periodic HeatNode.
ValueError: ShardedStencilNode: boundary must be one mode for every axis, one of ('periodic', 'edge', 'zero'); got dict {'devices': 'edge'}.  Per-mesh-axis modes are a halo_exchange feature: the wrapper also fills halo axes axis_map leaves unsharded, which have no mesh axis to key a mode by.
```

<!-- snippet: continues -->
```python
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes import HeatNode

mesh = create_device_mesh(shape=(1,))       # one device is enough to see the checks


def rod():
    return HeatNode("rod", 0.001, n_cells=16, length=1.0, thermal_diffusivity=0.01)


for boundary in ("zero", {"devices": "edge"}):
    with refused(ValueError) as r:
        ShardedStencilNode(rod(), mesh, axis_map={"devices": 0}, boundary=boundary)
    print(r.text[:90], "...")

gm = GraphManager()
gm.add_node(ShardedStencilNode(rod(), mesh, axis_map={"devices": 0}))   # "edge"
gm.add_external_input("rod", "left_temperature")      # cold ends: 0.0 unless fed
gm.add_external_input("rod", "right_temperature")
gm.compile()
gm.step()
```

### `sharded_cg` and `sharded_gmres`

- **Differentiating a solve needs `differentiable=True`.**  `jax.grad`,
  `jax.jvp`, `jax.jacfwd` and `jax.jacrev` through a call without it raise
  `ValueError`; 0.3.1 differentiated the iteration, which was wrong when
  the tangent was not of the right-hand side's size.  If the solve is a
  constant of the differentiation, put `jax.lax.stop_gradient` on what
  goes into the call.
- **`converged` and `residual_norm` describe the returned value**:
  `converged` is `||b - A x|| <= max(atol, rtol * ||b||)` on the `x`
  returned.  A float32 solve that asks for a tolerance it cannot reach
  answers `converged=False` with the true residual.  If your code branches
  on `converged`, ask for a reachable `rtol`, solve in float64, or read
  `residual_norm`.
- **The default `atol` is relative to `b`**, no longer an absolute `1e-8`,
  and a negative or NaN `rtol` or `atol` raises `ValueError`.
- **The default `backend="auto"`** solves again with the loop backend when
  lineax's CG returns a non-finite value; `backend="lineax"`, asked for by
  name, still returns it.
- The result's solution is `.value`.

```text
ValueError: sharded_cg was differentiated without differentiable=True: jax.jvp, jax.grad, jax.jacfwd, jax.jacrev or a transformation built on them reached the result of this call. [...] Pass differentiable=True: the derivative is then that of the linear solve, taken with the same backend and preconditioner, and .iters reads -1. [...]
```

<!-- snippet: continues -->
```python
from maddening.cloud.multigpu.iterative_solver import sharded_cg

A = jnp.diag(jnp.arange(1.0, 9.0))


def total(b, **options):
    return sharded_cg(lambda x: A @ x, b, **options).value.sum()


with refused(ValueError) as r:
    jax.grad(total)(jnp.ones(8))
assert "without differentiable=True" in r.text
grad = jax.grad(lambda b: total(b, differentiable=True))(jnp.ones(8))
assert jnp.allclose(grad, 1.0 / jnp.arange(1.0, 9.0), rtol=1e-5)
```

## Deprecated names

| 0.3.1 | 0.4.0 | Removed in |
|---|---|---|
| `calibrate`, `tune_coupling_params` (`maddening.core.simulation.calibration`) | `maddening.sysid.fit` | 0.5.0 |
| `solver="fori"` on a coupling group | the default `solver="ift"` | the next minor release |
| `maddening.core.simulation.checkpoint.download_and_load_state` | `maddening.cloud.download_and_load_state`; pass `manifest_url=` (or set `RESUME_MANIFEST_URL`) with a presigned URL | 1.0 |

Each old name still works and warns with a `DeprecationWarning`:

```text
DeprecationWarning: maddening.core.simulation.calibration.calibrate is deprecated; use maddening.sysid.fit, which optimises the graph params pytree under ParamSpec bounds and a trainable mask.  Removed in 0.5.0.
DeprecationWarning: CouplingGroup solver='fori' is deprecated and will be removed in the next minor release; the default solver='ift' exits early on convergence and supports forward- and reverse-mode AD.
```

`calibrate` fitted a function of a parameter dictionary.  `sysid.fit`
fits the graph's own parameters (`gm.params`), within each parameter's
`ParamSpec` bounds; here it recovers gravity from a recorded fall:

<!-- snippet: continues -->
```python
from maddening import sysid
from maddening.core.simulation.calibration import calibrate

with warnings.catch_warnings(record=True) as caught:      # 0.3.1
    warnings.simplefilter("always")
    calibrate(lambda p: 0.5 * p["gravity"], {"gravity": jnp.array(-5.0)},
              jnp.array(-4.905), n_iters=400, learning_rate=0.1)
assert issubclass(caught[0].category, DeprecationWarning)

gm = GraphManager()                                        # 0.4.0
gm.add_node(BallNode("ball", 0.01, initial_position=10.0))
gm.compile()
start_state = {n: gm.get_node_state(n) for n in gm.node_names}
_, history = gm.run_scan_with_history(50)
observations = sysid.observations_from_history(start_state, history)

guess = jax.tree.map(lambda x: x, gm.params)
guess["nodes"]["ball"]["gravity"] = jnp.asarray(-5.0)
mask = gm.trainable_mask()
mask["nodes"]["ball"]["elasticity"] = False                # the fall never bounces
loss = lambda p: sysid.windowed_loss(
    gm, p, observations, obs_fn=lambda h: h["ball"]["position"], window=10)
result = sysid.fit(gm, loss, params=guess, mask=mask, n_iter=300, lr=0.1)
print(f"gravity: {float(result.params['nodes']['ball']['gravity']):.3f}")
assert abs(float(result.params["nodes"]["ball"]["gravity"]) + 9.81) < 0.01
```

The [parameters guide](parameters.md#system-identification-maddeningsysid)
covers `sysid` in full.

**A registry of your own.**  `validate_anomaly_registry()` requires
`resolution_status` on every entry and checks its value, rejects unknown
keys and resolves `affected_components`.  Add the field to every entry,
or pass `resolve_references=False` for a check of the schema alone.
