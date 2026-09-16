# Exporting a graph as an FMU

MADDENING exports a compiled graph as an FMI 3.0 co-simulation FMU.  The
FMU binary holds no simulation state: it is a thin C wrapper that forwards
every FMI call to a Python *sidecar* holding the JAX-JITted graph, so XLA's
compile cost is paid once, not on every instantiation.

```
importer (FMPy, OpenModelica, ...)      Python process
 ┌──────────────────────────┐            ┌──────────────────────────┐
 │ plant.fmu                │  TCP/JSON  │ FmuTcpBridge             │
 │  binaries/.../           │ ─────────▶ │   └─ FmuSidecar          │
 │    maddening_fmu.so      │ ◀───────── │        └─ compiled step  │
 │  resources/endpoint.txt  │            └──────────────────────────┘
 └──────────────────────────┘
```

## Building and packaging

```python
from maddening.fmi import (
    FmuTcpBridge, MODEL_IDENTIFIER, build_fmu_binary, build_model_description, write_fmu,
)
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig

gm.compile()
md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)
binary = build_fmu_binary("build/")                  # needs a C compiler; libc only

sidecar = FmuSidecar(SidecarConfig(
    schema_token=md.instantiation_token, step_fn=gm._compiled_step,
    initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
))
bridge = FmuTcpBridge(sidecar, md, master_dt=gm_base_dt, port=5555).start()
write_fmu(md, "plant.fmu", binary=binary, endpoint=bridge.endpoint)
```

The importer then loads `plant.fmu` as usual; the wrapper reads
`resources/endpoint.txt` (or `MADDENING_FMU_ENDPOINT`) and connects.  A
communication step of `h` runs `round(h / master_dt)` graph steps.
`fmi3GetFMUState` / `fmi3SetFMUState` / serialization round-trip the
sidecar's state, and `fmi3Reset` returns to the initial state and
parameters.  Model exchange and scheduled execution are refused at
instantiation.

**Variables.**  Outputs are `<node>.<field>`; external inputs are
`<node>.<field>` of the target boundary field (start value 0, description
and unit from the target node's `boundary_input_spec`); parameters are
`<node>.params.<key>` with `ParamSpec` bounds as `min` / `max`.  Setting a
parameter goes through the sidecar's bounds check, so an importer cannot
drive the graph with a constant it declares invalid.

**Transport.**  Each message is a 4-byte big-endian length followed by one
JSON object (`{"op": "set"|"get"|"step"|"get_state"|"set_state"|"reset"|
"terminate"|"hello", ...}`); see `maddening.fmi.tcp_bridge` for the exact
schema.  Nothing but libc is required on the importer's side, which is why
ZMQ is not used for the FMU path.

## Multi-rate graphs: clocks

`build_model_description(gm, ..., multi_clock=True)` emits one FMI 3.0
`<Clock>` per distinct node timestep (`clock_0`, `clock_1`, ... by
increasing interval, `intervalVariability="constant"`), and tags every
output and external input of a node with its clock (`clocks=` attribute,
`variability="discrete"`).  An importer then knows that a node on a five
times coarser rate only changes on every fifth master step.  The fastest
clock equals the default experiment step size.  Clocks are off by default,
so a single-clock FMU is byte-for-byte what v0.3.0 produced.

## Verification

`tests/fmi/test_c_wrapper.py` builds the binary, packages an FMU, and has
FMPy's `simulate_fmu` drive it against a bridge with a set parameter and a
driven input; the outputs reproduce `gm.run_scan` to float32 round-off.
`tests/fmi/test_multi_clock.py` validates the multi-clock description with
FMPy's schema and model-structure validation.
