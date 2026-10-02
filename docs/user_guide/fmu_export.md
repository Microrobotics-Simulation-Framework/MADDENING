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

<!-- snippet: no-run, reason: fragment: gm is the reader's graph; it also needs a C compiler and opens a TCP port -->
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
    fixed_params=md.fixed_parameters,   # the bridge applies it anyway
    input_resolver=gm._resolve_external_inputs,   # inputs as GraphManager.step takes them
))
bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep, port=5555).start()
write_fmu(md, "plant.fmu", binary=binary, endpoint=bridge.endpoint)
```

`python -m maddening.examples.advanced.fmu_export_demo` runs the
build-and-check half of this without a TCP port: the description (with
the parameters it leaves fixed, and why), the binary and the package in
a temporary directory, FMPy's `validate_fmu`, and the sidecar driven
in-process against `run_scan(params=...)`.  It skips the stages whose C
compiler or FMPy is missing, saying so.

The importer then loads `plant.fmu` as usual; the wrapper reads
`resources/endpoint.txt` (or `MADDENING_FMU_ENDPOINT`) and connects.  A
communication step of `h` runs `h / master_dt` graph steps.  Pass
`master_dt=gm.timestep`, the simulated time one graph step advances; it is
also the description's default `stepSize`.  On a graph with a sub-cycling
coupling group it is the group's largest member timestep, not the smallest
node timestep.  The description records it as `md.graph_timestep`, and the
bridge refuses any other `master_dt`, and a description whose
`default_step_size` is not a whole number of graph steps.  Until 0.4.0
shipped it took any `master_dt`, so a 0.01 s graph served with
`master_dt=0.005` ran two graph steps per 0.01 s `doStep` and reported half
the time its state was at (MADD-ANO-100).  `default_step_size=` may
advertise a coarser communication step, such as five graph steps; the
bridge then runs five per `doStep`, and `master_dt` is still the graph's
step.  One `doStep` runs at most `max_steps_per_request` graph steps
(100000 by default, `FmuTcpBridge(..., max_steps_per_request=)`), and a
long one stops at the first graph step after `stop()`, committing nothing.
`fmi3GetFMUState` / `fmi3SetFMUState` / serialization round-trip the
sidecar's state.  Model exchange and scheduled execution are refused at
instantiation.

**Instances and time.**  One bridge serves one FMU instance at a time, and
every instance starts where FMI 3.0 starts one.  When a new connection
claims the bridge (`fmi3InstantiateCoSimulation`), the bridge resets to
the state it was built over.  Every parameter goes back to the `start` value
the model description advertises, every input to zero, and the time to
zero.  `fmi3Reset` goes to the same point.  So a master that frees an
instance and instantiates another between runs, as FMPy's `simulate_fmu`
does on every call, gets the same answer for the same arguments.  Until
0.4.0 shipped, the new instance inherited the previous one's state, tuned
parameters, pending inputs and time.  A sidecar built with other parameter
values than the description advertises is brought into line, with a
`UserWarning` naming them.

`fmi3EnterInitializationMode(startTime)` sets the FMU's time, which is
what the `time` variable reads from then on.  It is accepted until the
instance's first step.  Each `fmi3DoStep` must start at the FMU's current
time: the previous communication point plus the previous step size, or
first the start time.  The tolerance is a millionth of a master step, which
absorbs any importer's floating-point accumulation.  A step size is held to
the same absolute tolerance off a whole number of master steps, so a step
the FMU accepts never makes the next legal `doStep` fail; until 0.4.0
shipped, the step size was allowed a millionth per master step it covered.
A point inside it is
adopted, so the importer's clock and the FMU's never drift apart.  A point
outside it is `fmi3Error` with nothing advanced: an importer that jumped
from 0.01 to 100 used to get one master step of physics labelled 100.01.
To go back in time, restore an FMU state; the clock moves with it.

`fmi3Terminate` puts the instance in FMI 3.0's Terminated state.  There,
`fmi3Get*`, the FMU-state functions and `fmi3Reset` are allowed, and
`fmi3DoStep`, `fmi3Set*` and `fmi3EnterInitializationMode` are `fmi3Error`
with nothing written, until `fmi3Reset` (or a new instance) starts it again.
A step after terminate used to advance the model.  Restoring an FMU state
does not leave Terminated.

**Variables.**  Outputs are `<node>.<field>`.  External inputs are
`<node>.<field>` of the target boundary field (start value 0, description
and unit from the target node's `boundary_input_spec`).  Parameters are
`<node>.params.<key>`, with the `ParamSpec` bounds as `min` / `max`.

Every `start`, `min` and `max` is written in its type's lexical form, which
FMI 3.0's schema requires: `true` / `false` for a Boolean, an integer
literal for an integer type, an `xs:double` for a float.  An integer's
bounds are the integers inside the declared interval, and a Boolean carries
no `min` / `max`.  A Boolean or integer input or output is
`variability="discrete"`, since FMI 3.0 allows `continuous` only for floats.
Until 0.4.0 shipped, a Boolean or Int32 input was written `start="0.0"` and
`continuous`.  FMPy's `simulate_fmu`, which validates the description by
default, refused such an FMU.

The `instantiationToken` covers every variable's start, bounds, variability,
value reference and clocks, besides its name, type, causality and shape.  An
FMU packaged from a description with other start values or bounds than the
one a bridge serves fails `fmi3InstantiateCoSimulation` against it with a
token mismatch.  It used to instantiate, advertising starts the bridge did
not use.  Rebuild and re-package an FMU whenever its graph's parameters
change.

A node's name may hold `.params.` (`rig.params.v2`).  Its parameter
variables (`rig.params.v2.params.elasticity`) carry their `(node, key)`, and
the sidecar matches a parameter name exactly rather than splitting it.  Such
a parameter used to be refused as unknown on every set.

Each variable is read and written only through the `fmi3Get` / `fmi3Set`
function of its declared type, as FMI 3.0 requires: a Float32 output with
`fmi3GetFloat32`, the `time` variable with `fmi3GetFloat64`.  Any other is
`fmi3Error` with nothing read or written (MADD-ANO-102): the wrapper names
the type of every call, and the bridge checks it.  Until 0.4.0 shipped,
`fmi3SetBoolean` on a Float32 parameter stored 1.0, and `fmi3GetInt32` on a
Float32 output returned 0 for 0.5, both `fmi3OK`.  FMPy's `simulate_fmu`
already uses the right functions; code that called `getFloat64` on a
Float32 variable must call `getFloat32`.  A Boolean variable takes true or
false, which travel as 1 and 0; any other number is refused rather than
read by its truthiness (MADD-ANO-101).  A `set` that names one value
reference twice is refused, and so is a numeric set of a `<Clock>`, whose
ticks are implied by time; both used to be answered `ok` and dropped.

A parameter constructed outside its `ParamSpec` bounds is advertised with a
`start` outside its `min` / `max`, and the FMU starts there, as the graph
does.  FMI 3.0 does not require `start` to lie in `[min, max]`: `min` and
`max` "define the region in which the FMU is designed to operate", and the
standard's section on range violations says the FMU should not rely on
them being observed.  FMPy's `validate_fmu` accepts such a description.

`build_model_description(..., selected_inputs=[...])` exports a subset of
the inputs.  A declared input left out is not an FMU variable, but the
graph still reads it, so the bridge holds it at zero on every step, as
`GraphManager.step` does for an input its caller omits.  It is listed in
`md.held_inputs`, and the builder warns.  Until 0.4.0 shipped, the bridge
passed only the exported inputs, and a node whose input was left out took
its own "input missing" branch instead: a ball with no table fell through
the floor.  A name the graph does not declare is a `ValueError`.

The in-process sidecar applies the same rule when it is given the graph's
resolver (`input_resolver=gm._resolve_external_inputs`).
`FmuSidecar.step` then completes a partial `external_inputs` with zeros and
refuses a misspelt name, as `GraphManager.step` does.  A sidecar behind a
bridge gets a resolver built from the model description if it was given
none.

Setting a parameter is held to the `min` / `max` the description
advertises, whether or not the sidecar was built with `param_specs`.  The
bridge adds a bounds-only spec for every exported parameter its sidecar has
no spec for.  So an importer cannot drive the graph with a constant it
declares invalid.  Only parameters the
compiled step reads are exported, all `variability="tunable"`: an
`initial_*` condition (the initial state is already built), a value a node
consumed when it was constructed (`LBMPipeNode.pipe_radius`) or declares in
`static_data_deps` (`WaveletAdaptiveNode.mass`), or one only an unconnected
input would read (a ball's `elasticity` with no table) would be a knob that
does nothing, so it is left out and listed, with the reason, in
`md.fixed_parameters`.  So is one from which a node derives the points an
interface mapping was built from (a uniform `HeatNode`'s `length` under a
mapping on its `grid_x`): the step would use a new value and the mapping
would keep the old points' weights.  The bridge applies that list to its sidecar: no
door into the parameter tree -- `fmi3Set*`, `fmi3SetFMUState`, the
sidecar's own `set_params` / `set_fmu_state` -- installs a new value for
one of them; the request is `fmi3Error` naming the parameter and why, and
nothing is written.  Until 0.4.0 shipped they were exported as tunable,
and the sidecar accepted and reported a value its step never used.  A node name may
itself contain a `.` (`tank.1`, `hx.hot`): the variable carries its
`(node, field)` pair, so the bridge never has to guess where the name
splits, and an export in which two nodes would spell the same variable
name is refused naming both.  A `model_name` of exactly `NaN`,
`Infinity` or `-Infinity` is refused by `build_model_description`: the
bridge's `hello` reply carries the model name, and those three strings
are how the wire spells a non-finite number (`MADD-ANO-010`), so such a
reply could not be written as unambiguous JSON.  Any other spelling
(`nan`, `Infinity_2`) is fine, as is any node name — `add_node` applies
the same three-token rule.

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
length.  Frames are limited to 64 MiB in both directions and for both
kinds: the bridge drops a connection that announces a longer *request*
(the length is not to be trusted), answers a `get` / `get_state` whose
*reply* would be longer with a JSON error instead (the connection stays in
sync; a `get` of more than about 8 M values needs to be split), and the C
wrapper refuses to read a longer frame of either kind and closes the
connection, so the instance fails every later call instead of parsing
stale bytes.  A binary payload is a short JSON
*header* carrying `op` and metadata, then the *raw* data, so bulk values
and FMU-state blobs never pass through JSON text (no `%.17g` / `strtod`,
no base64):

| message                | header                                          | raw part                    |
|------------------------|-------------------------------------------------|-----------------------------|
| `set` request          | `{"op":"set","type":T,"vr":[..],"n":N,"dtype":"f64"}` | N little-endian float64 |
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
protocol-1 behaviour: JSON everywhere, as v0.3.0 spoke it, except that the
hello reply now also carries `"protocol": 2, "binary": false` (a v0.3.0
client ignores the extra keys).
A client announcing a protocol the bridge does not know is refused at
hello with a clear error.  Conversely, the new wrapper against an older
bridge (a hello reply without `protocol`) falls back to JSON for every
message.  The bridge counts what it did in `binary_frames_served` (binary
replies) and `binary_frames_received` (well-formed binary requests).

**Float64 on the wire, typed calls.**  The wire carries every value as a
float64: a float32 output is widened on the way out and a float32 input
narrowed by the bridge on the way in, after the check that the type can
hold it.  Every `get` and `set` request names the FMI type of the function
that made it (`"type": "Float32"`, also in a binary `set` header), and the
bridge refuses a variable of another type.  A JSON client that names no
type is not checked.  The C wrapper also refuses a reply value its getter's
C type cannot hold (a fraction, `NaN` or an out-of-range number for an
integer type, anything but 0 or 1 for a Boolean, a finite number beyond
`FLT_MAX` for a Float32), so no conversion is undefined behaviour, and an
`fmi3SetInt64` / `fmi3SetUInt64` value a double cannot carry exactly (above
2^53 in magnitude) is refused rather than rounded.  Wire order is
little-endian; the C side converts only on a big-endian host (a `memcpy`
everywhere else).

**Robustness.**  Nothing in a binary frame is trusted: the bridge checks
`header_len` against the payload, `n` against the raw length, the dtype,
the value references, bounds, read-only-ness and the state archive exactly
as for JSON (a header nested too deeply for the JSON parser is a malformed
request like any other), and answers a malformed frame with an error reply
rather than dropping the instance.  Values are checked in the variable's
own type: a float32 input set to `1e308` is refused, not stored as `inf`.
A `set_state` archive is refused from its directory alone, before any
member is decompressed, unless every member is one the model expects
(`_token`, `_time`, its state fields, parameters and inputs, all `.npy`)
and declares no more than the live array it replaces, with a cap on the
total as well.  The C side checks the header count against the raw length
and against the caller's array before any `memcpy`, refuses a reply
length over the limit (binary or JSON) before allocating for it, drops
the connection on such a reply or on one cut short by the peer, and never
sends a `set` or a `set_state` whose frame, header included, would exceed
the limit.  (`fmi3SetFMUState` used to check the blob alone, so a blob of
exactly 64 MiB was sent, and the bridge dropped the connection.)  A
`get` reply must carry exactly the `nValues` the importer asked for:
a longer one is `fmi3Error`, as a shorter one always was, rather than
`fmi3OK` with the rest dropped.  A send interrupted by a signal is
retried, and any other failed send closes the connection, because part of
the frame may already be on the wire.  An empty instantiation token is a
mismatch.  Numbers on the JSON path are written and read in the C
locale, whatever `LC_NUMERIC` the importer runs under: an importer under a
locale with a `','` decimal point (a GUI tool on a Dutch or German desktop)
used to send `"dt":0,01`, and every `fmi3DoStep` failed.  The wrapper's own
clock, which `fmi3DoStep` reports as `lastSuccessfulTime` when it fails,
follows `fmi3SetFMUState` (the bridge's reply carries the restored time)
and `fmi3Reset` (once the reset has succeeded).  The wrapper's socket has
`TCP_NODELAY` set: a frame goes out as
two writes, and with Nagle's algorithm the second waited for the bridge's
delayed ACK, so until 0.4.0 shipped every FMI call took at least 40 ms on
Linux.  The
serialized FMU state is opaque to the importer; its encoding (raw `npz`
on protocol 2, base64 text on protocol 1) is that of the connection it
came from, so a state serialized under one protocol must be restored
under the same one.

**Why.**  A `get` of a 10^6-element field costs ~2.3 s as JSON text
(~18 bytes per value plus `json.loads`) and ~27 ms as a binary frame
(8 bytes per value) over loopback, measured by
`tests/fmi/test_binary_frames.py::test_million_element_get_binary_is_faster_than_json`.

## The trust boundary

The importer is **untrusted**.  Bind the bridge to `127.0.0.1` unless the
network is trusted, and know what the bridge does and does not promise.

**Nothing on the socket is unpickled or evaluated.**  The FMU-state blob
is an `npz` of plain arrays read with `allow_pickle=False`, and its
archive directory is checked before anything is decompressed.

**No wait on a connection is unbounded.**  A connection holds the
bridge's single FMU instance for as long as it lives, so each wait has a
finite budget: ten seconds to begin the first frame, five minutes of
silence between frames once the peer has spoken, and two minutes to
finish a frame whose length it has announced -- two minutes in total,
whether the rest of the frame dribbles in or stops arriving.  Overrunning
any of them ends the connection exactly as EOF does.  The five minutes are
`FmuTcpBridge(idle_timeout=...)`: a master that pauses longer between calls
(a debugging session, a slow partner model) passes more, or `None` for no
limit.  It also bounds sending a reply.  One `step` request
may ask for at most `max_steps_per_request` graph steps, and a step stops at
the next graph step once `stop()` is called; until 0.4.0 shipped,
`dt = 1e9 * master_dt` held the worker for as long as a billion steps take.
The instance slot is claimed
when a peer sends its first complete frame, not when it connects, so a
peer that connects and says nothing — a crashed importer, a dropped
link, a port scan — claims nothing.  At most sixteen connection threads
are live at once; further connections are closed on accept.  `stop()`
shuts every live connection down and joins its worker.  A bridge serves
once: calling `start()` again, or after `stop()`, raises `RuntimeError`;
build a new `FmuTcpBridge` to serve again.

**A value that `set` refuses, `set_state` refuses too.**  Both doors
into the state and parameter tree apply the same checks: every value
must be a number (a string such as `"45"` or a boolean is refused, not
parsed), finite, and representable in the dtype of the array it
replaces, and every parameter must lie inside its declared `ParamSpec`
bounds — the `min` / `max` the model description advertises, which the
bridge enforces however its sidecar was built.  An importer therefore
cannot use an FMU-state archive to install a constant the graph declares
invalid.  An archive cannot change interface-mapping weights either
(`params["mappings"]`): they are not FMI variables, so no `set` reaches
them, and an archive that replaced them made the FMU compute a coupling its
description does not describe (MADD-ANO-119).  A Boolean takes true / false
or exactly 0 / 1 on both doors.  An
archive must carry its time, and exactly the model's state fields,
parameters and pending inputs: an archive missing an input used to restore
with that input at zero (MADD-ANO-103).  The sidecar's own in-process
doors apply the same checks with the same messages: `FmuSidecar.set_params`
refuses what `set` refuses, and `FmuSidecar.set_fmu_state` refuses what
`set_state` refuses, including parameters in a snapshot for a model with
no parameter tree.  A sidecar behind a bridge enforces the advertised
bounds; a stand-alone one enforces the bounds of the `param_specs` it was
given.  A `step`'s `dt` and `t` must be numbers too.

A restore holds to the bounds the values it would *install*: a parameter
the snapshot carries at the value it holds now, or held when the FMU was
instantiated, is not a new value.  A graph runs whatever its constructor
was given, bounds being metadata to it, so an FMU can be exported with a
parameter outside its bounds; until 0.4.0's fix both restore paths then
refused the snapshot the FMU had handed out itself, while
`GraphManager.load_state` restored the graph's checkpoint.  A `set` is
still held to the bounds whatever it sets, the starting value included.

```{warning}
The consequence is that a snapshot of a *diverged* model — one whose
state holds `inf` or `NaN` — does not restore.  The error names the
field.  A non-finite value the FMU was *instantiated* with is not a
divergence and restores: a coupling group with `diagnostics=True` seeds
its spectral `_meta` slots with `NaN` until a solve fills them, and until
0.4.0's fix a snapshot taken before that was refused by both doors.
```

**The C wrapper waits on the bridge for a bounded time.**  Every receive
and send on its connection times out after `MADDENING_FMU_TIMEOUT` seconds
of silence (600 by default; `0` waits for ever, as before 0.4.0), and a
timeout closes the connection, so that call and every later one return
`fmi3Error`.  Ten minutes covers a first `doStep` that compiles a large
graph and a step of many graph steps; raise it for a longer one.
Connecting (on Linux) and the hello get the smaller of that and 30 s: a
live bridge answers a hello at once, so an endpoint that accepts and never
answers fails `fmi3InstantiateCoSimulation` instead of blocking it.  A value
that is not a number of seconds from 0 to 1e6 fails instantiation.  The
deadline is on silence, not on a whole reply: a bridge that keeps sending
is not cut off.

**`FmuSidecar.handle` is not part of this.**  It speaks a pickled
request/response protocol, so unpickling a request runs whatever
produced the bytes.  It refuses to run unless the sidecar was built with
`SidecarConfig(..., allow_pickle_rpc=True)`, which is only appropriate
for an in-process caller you trust as much as your own code.  The FMU
does not use it.

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
clock equals the default experiment step size.  Clocks are off by default:
a single-clock FMU has no `<Clock>` variable and no `clocks=` attribute,
and every output is continuous.  The clocks
are constant-interval and tick with time.  `fmi3GetClock` and
`fmi3SetClock` are `fmi3Error`, because FMI 3.0 allows them only in Event
Mode, which this FMU does not have (`hasEventMode="false"`).  They used to
answer `fmi3OK` for any value reference.

## Verification

`tests/fmi/test_c_wrapper.py` builds the binary, packages an FMU, and has
FMPy's `simulate_fmu` drive it against a bridge with a set parameter and a
driven input; the outputs reproduce `gm.run_scan` to float32 round-off.
`tests/fmi/test_multi_clock.py` validates the multi-clock description with
FMPy's schema and model-structure validation.
