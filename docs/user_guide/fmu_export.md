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

**Transport.**  Each message is a 4-byte big-endian length prefix followed
by one frame: a JSON object (`{"op": "set"|"get"|"step"|"get_state"|
"set_state"|"reset"|"terminate"|"hello", ...}`) or, once negotiated, a
*binary* frame for bulk payloads; see "Wire protocol" below and
`maddening.fmi.tcp_bridge` for the exact schema.  Nothing but libc is
required on the importer's side, which is why ZMQ is not used for the FMU
path.

## Wire protocol

```
 length prefix (4 bytes, big-endian)          payload
 ┌─┬───────────────────────────────┐
 │0│      payload length (31 bit)  │  one UTF-8 JSON object        (JSON frame)
 └─┴───────────────────────────────┘
 ┌─┬───────────────────────────────┐  ┌────────────┬─────────────┬───────────┐
 │1│      payload length (31 bit)  │  │ header_len │ header JSON │ raw bytes │  (binary frame)
 └─┴───────────────────────────────┘  │  u32 BE    │             │           │
                                      └────────────┴─────────────┴───────────┘
```

Bit 31 of the prefix marks a binary frame; the low 31 bits are the payload
length (64 MiB limit either way).  A binary payload is a short JSON
*header* carrying `op` and metadata, then the *raw* data, so bulk values
and FMU-state blobs never pass through JSON text (no `%.17g` / `strtod`,
no base64):

| message                | header                                          | raw part                    |
|------------------------|-------------------------------------------------|-----------------------------|
| `set` request          | `{"op":"set","vr":[..],"n":N,"dtype":"f64"}`    | N little-endian float64     |
| `get` reply            | `{"ok":true,"n":N,"dtype":"f64"}`               | N little-endian float64     |
| `set_state` request    | `{"op":"set_state","n":L}`                      | L bytes of `npz`            |
| `get_state` reply      | `{"ok":true,"n":L}`                             | L bytes of `npz`            |

Everything else (`hello`, `get` *requests*, `step`, `reset`, `terminate`,
every error reply) is a JSON frame.

**Negotiation.**  The C wrapper opens with
`{"op":"hello","protocol":2,"binary":true}`.  A bridge of this version
answers `{"ok":true, ..., "protocol":2, "binary":true}` and, for that
connection only, sends `get` / `get_state` replies as binary frames and
accepts binary `set` / `set_state` requests.  A client whose hello lacks
`protocol` (or says `protocol: 1`, or omits `binary: true`) gets the
protocol-1 behaviour: JSON everywhere, byte for byte what v0.3.0 spoke.
A client announcing a protocol the bridge does not know is refused at
hello with a clear error.  Conversely, the new wrapper against an older
bridge (a hello reply without `protocol`) falls back to JSON for every
message.  The bridge counts what it did in `binary_frames_served` (binary
replies) and `binary_frames_received` (well-formed binary requests).

**Float64 only.**  The FMU's variable surface is Float64 (the wrapper
widens every FMI width, `fmi3GetFloat32` included, to `double`), so the
wire carries float64 only: a float32 output is widened on the way out and
a float32 input narrowed by the bridge on the way in.  Wire order is
little-endian; the C side converts only on a big-endian host (a `memcpy`
everywhere else).

**Robustness.**  Nothing in a binary frame is trusted: the bridge checks
`header_len` against the payload, `n` against the raw length, the dtype,
the value references, bounds, read-only-ness and the state archive exactly
as for JSON, and answers a malformed frame with an error reply rather
than dropping the instance; the C side checks the header count against
the raw length and against the caller's array before any `memcpy`, and
refuses a flagged length over the limit before allocating for it.  The
serialized FMU state is opaque to the importer; its encoding (raw `npz`
on protocol 2, base64 text on protocol 1) is that of the connection it
came from, so a state serialized under one protocol must be restored
under the same one.

**Why.**  A `get` of a 10^6-element field costs ~2.3 s as JSON text
(~18 bytes per value plus `json.loads`) and ~27 ms as a binary frame
(8 bytes per value) over loopback, measured by
`tests/fmi/test_binary_frames.py::test_million_element_get_binary_is_faster_than_json`.

## Why TCP + JSON, and when ZMQ would be worth it

The v0.3.0 plan called the sidecar protocol "ZMQ".  The shipped wrapper
uses a raw TCP socket with 4-byte-length-prefixed JSON frames instead.
The trade-off, recorded so the choice can be revisited deliberately:

**TCP + JSON (current)**

* Nothing to link on the importer's side: the FMU binary depends on
  libc only, so it loads in any FMI 3 importer on any machine without
  a libzmq of a matching ABI.  This is the reason it was chosen.
* One request in flight per instance, strictly request/reply, which is
  exactly the FMI call pattern; framing is 20 lines of C and was easy to
  unit-test and fuzz.
* Costs: no built-in reconnect or heartbeat (a dead sidecar surfaces as
  `fmi3Error` from the next call), one connection per instance, and no
  transport-level multiplexing.

**ZMQ (`REQ`/`REP` or `DEALER`/`ROUTER`)**

* Pros: automatic reconnect and queueing, `inproc://`/`ipc://`/`tcp://`
  behind one API, message framing for free, easier fan-out to several
  sidecars behind a `ROUTER`, and the Python side already has pyzmq for
  the multi-VM coordinator.
* Cons: the FMU binary would link libzmq (a C++ library) and the
  importer's process must be able to load it; version/ABI pinning
  becomes part of the FMU's packaging story; `REQ`/`REP` state machines
  make error recovery *harder* for a strict request/reply client, and
  the C API needs the same length-prefixed JSON payload anyway.

**Recommendation.**  Stay on TCP + JSON until a concrete need appears:
many FMU instances against one sidecar process (then `ROUTER` on the
Python side, still TCP framing in C), or a deployment where the sidecar
restarts and instances must survive it (then reconnect logic, which
ZMQ gives for free).  The frame payloads (JSON objects and the binary
header + raw layout above) would not change either way, so the C
wrapper's request builder and reply parser carry over; only the 4-byte
length prefix would be replaced by ZMQ's own message framing (the binary
flag would move into the header).

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
