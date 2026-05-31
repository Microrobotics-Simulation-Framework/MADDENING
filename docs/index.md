# MADDENING

**Modular Automatic Differentiation and Data Enhanced Neural-network INteracting
Graph** — a pure-{term}`JAX`, autodifferentiable framework for composing physical
models from typed {term}`nodes <Node>` connected by unit-aware {term}`edges <Edge>`.
MADDENING is the base framework on which
[MIME](https://microrobotica.org/mime/) and the
[MICROROBOTICA](https://microrobotica.org/) IDE are built.

::::{grid} 1 2 2 2
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} Differentiable end-to-end
:class-card: msf-card

Every {term}`Graph step` compiles into a single {term}`XLA` op. Take
`jax.grad` through 1000-step rollouts; vmap across initial conditions for free.
:::

:::{grid-item-card} Node-graph composition
:class-card: msf-card

Author a {term}`Node` as a {term}`pure function <Pure function>` of
`(state, dt, inputs)`. Wire it into the graph with unit-aware
{term}`Edge`s and let the scheduler {term}`topologically sort
<Topological sort>` and parallelise it.
:::

:::{grid-item-card} Coupling & multi-rate
:class-card: msf-card

{term}`Gauss-Seidel` and {term}`Jacobi` {term}`Coupling` for strongly
coupled subsystems, {term}`Aitken <Aitken acceleration>` / {term}`IQN-ILS`
acceleration, GCD-based {term}`Multi-rate` scheduling, and a
{term}`Richardson <Richardson extrapolation>` + {term}`PI <PI controller>`
adaptive timestepper.
:::

:::{grid-item-card} Neural surrogates
:class-card: msf-card

Train an {term}`MLP`, {term}`DeepONet`, or {term}`FNO` {term}`Surrogate`
against a slow physics node, then hot-swap it into the graph at runtime —
the rest of the simulation never notices.
:::

::::

## How MADDENING is wired

MADDENING is split into a small **core** ({term}`node <SimulationNode>` ABC,
typed edges, {term}`graph manager <GraphManager>`), a **scheduler** that
resolves coupling and multi-rate timing, an **{term}`XLA`-compilation** path,
a **surrogate** training subsystem, and a **cloud** layer for distributing
all of it. MIME plugs in at the node level — same runtime,
microrobotics-specific physics + metadata.

```{mermaid-tips}
nodes:
  Node:     SimulationNode ABC — a pure function of (state, dt, boundary_inputs). Every physics block ultimately subclasses this.
  Edge:     EdgeSpec carries the field name, dtype, units, and the source/sink node refs. Units are checked at graph compile time.
  Graph:    GraphManager — topologically sorts nodes per step and breaks cycles by staggering them into iterative Gauss-Seidel sweeps.
  Coup:     Coupling solvers — Gauss-Seidel and Jacobi sweeps for strongly coupled subsystems.
  Accel:    Convergence acceleration — Aitken (scalar) and IQN-ILS (vector quasi-Newton) re-weight the Gauss-Seidel iterates.
  Multi:    Multi-rate scheduler — nodes can advance at different dt's; the scheduler aligns them on the GCD step.
  Adapt:    Adaptive timestepper — Richardson extrapolation gives a local error estimate, a PI controller grows/shrinks dt.
  XLA:      The whole step compiles into a single XLA op; jax.jit caches the binary, jax.vmap parallelises across batches, jax.grad differentiates end-to-end.
  Train:    Surrogate trainers — MLP / DeepONet / FNO architectures, with paired data loaders and metric callbacks.
  Swap:     Hot-swap registry — at runtime a physics node can be replaced by a trained surrogate without restarting the graph.
  Sky:      SkyPilot orchestration provisions GPU VMs (RunPod, Lambda, AWS, GCP) and tears them down when the job ends.
  ZMQ:      ZeroMQ rendezvous wires distributed graph fragments and streams state to clients.
  WRTC:     WebRTC streams rendered viewport frames from the remote VM to the local browser or IDE.
  API:      FastAPI + WebSocket server — remote control of the graph and live state subscriptions.
  USD:      Codeless USD schemas + a serialiser turn the graph and its trajectory into a USD stage MICROROBOTICA can render.
  MIM:      MIME extends SimulationNode with biocompatibility, anatomical regime, and SOUP classification — same coupling, same autodiff.
edges:
  Node->Graph:      Every node registers itself with the manager and declares its input / output edges.
  Edge->Graph:      Edge specs are how the manager builds the dependency DAG and validates units at compile time.
  Graph->Coup:      For any strongly connected component the manager hands off to a coupling solver per step.
  Coup->Accel:      Each iterate goes through Aitken or IQN-ILS to shrink the residual faster.
  Graph->Multi:     The scheduler asks the manager which nodes are due on this step.
  Graph->Adapt:     The adaptive controller re-estimates dt from the latest Richardson residual.
  Graph->XLA:       Once the order and dt's are fixed, jax.jit compiles the whole walk into one XLA op.
  Train->Swap:      Trained weights are pushed into the swap registry alongside the original physics node.
  Swap->Graph:      On the next compile the manager picks the surrogate over the physics — no graph edits needed.
  Graph->USD:       Each tick's outputs are streamed into a USD stage layer the IDE later renders.
  Graph->ZMQ:       For distributed graphs each fragment publishes its outputs as named ZMQ messages.
  Sky->ZMQ:         SkyPilot brings up the worker VMs and seeds them with the rendezvous address.
  Graph->API:       FastAPI exposes graph state and step controls over WebSocket for live UIs.
  Node->MIM:        MIME's domain nodes inherit from SimulationNode and add the per-asset regulatory metadata MADDENING itself does not carry.
```

```{mermaid}
flowchart LR
    subgraph CORE["core"]
      direction TB
      Node["SimulationNode<br/><i>pure (state, dt, in) → out</i>"]
      Edge["EdgeSpec<br/><i>typed, unit-aware</i>"]
      Graph["GraphManager<br/><i>topological sort + cycle staggering</i>"]
    end

    subgraph SCHED["scheduling"]
      direction TB
      Coup["coupling<br/><i>Gauss-Seidel &middot; Jacobi</i>"]
      Accel["acceleration<br/><i>Aitken &middot; IQN-ILS</i>"]
      Multi["multi-rate<br/><i>GCD scheduler</i>"]
      Adapt["adaptive dt<br/><i>Richardson + PI</i>"]
    end

    XLA["XLA compilation<br/><i>jit &middot; vmap &middot; grad</i>"]

    subgraph SURR["surrogates"]
      direction TB
      Train["MLP / DeepONet / FNO<br/><i>training + data loaders</i>"]
      Swap["hot-swap registry"]
    end

    subgraph CLOUD["cloud / API"]
      direction TB
      Sky["SkyPilot"]
      ZMQ["ZMQ rendezvous"]
      WRTC["WebRTC streaming"]
      API["FastAPI + WebSocket"]
    end

    USD["USD codeless schemas<br/><i>graph + trajectory serializer</i>"]

    Node --> Graph
    Edge --> Graph
    Graph --> Coup
    Coup --> Accel
    Graph --> Multi
    Graph --> Adapt
    Graph --> XLA
    Train --> Swap
    Swap --> Graph
    Graph --> USD
    Graph --> ZMQ
    Sky --> ZMQ
    Graph --> API
    ZMQ -.-> WRTC

    MIM["MIME<br/><i>SimulationNode subclasses<br/>+ biocompat / SOUP / regime</i>"]:::ext
    Node --> MIM

    classDef ext stroke-dasharray:5 3
```

```{toctree}
:caption: User Guide
:maxdepth: 2

user_guide/installation
user_guide/quickstart
user_guide/cloud_resume
glossary
```

```{toctree}
:caption: Algorithm Guide
:maxdepth: 2
:glob:

algorithm_guide/nodes/*
algorithm_guide/coupling/*
algorithm_guide/uq/*
```

```{toctree}
:caption: Developer Guide
:maxdepth: 2

developer_guide/node_authoring
developer_guide/testing_standards
developer_guide/documentation_standards
developer_guide/edge_validation_migration
developer_guide/stability_report
developer_guide/versioned_docs
```

```{toctree}
:caption: Validation
:maxdepth: 2

validation/cou_template
validation/framework_verification
validation/soup_package
```

```{toctree}
:caption: Regulatory
:maxdepth: 2

regulatory/intended_use
regulatory/iec62304_mapping
regulatory/eu_mdr_guidelines
regulatory/mdcg_2019_11
regulatory/downstream_integration
```

```{toctree}
:caption: Release notes
:maxdepth: 1

release_notes/index
```
