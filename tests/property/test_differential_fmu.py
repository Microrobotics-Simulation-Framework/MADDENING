"""Differential oracle 5: three FMU transports are one model.

An FMU's state can be reached three ways: the TCP bridge an importer's C
wrapper talks to (:class:`~maddening.fmi.tcp_bridge.FmuTcpBridge`, over a real
socket), the in-process :class:`~maddening.fmi.sidecar.FmuSidecar` it wraps,
driven through its Python API, and the :class:`GraphManager` the FMU was
exported from, driven directly.  ``tests/property/test_stateful_bridge.py``
compares the bridge over TCP with a second bridge in process; both run the
bridge's own dispatch, so a defect in that dispatch is on both sides.  Here
the three paths share nothing but the compiled step.

For generated sequences of ``set`` / ``get`` / ``step`` / ``get_state`` /
``set_state`` / ``reset`` / ``initialize`` / free-and-re-instantiate (a new
connection claiming the bridge's instance slot), refused steps and
misspelt inputs:

* every value -- time, inputs, parameters, outputs -- and the full state
  (``_meta`` included) agree **bit for bit** across the three after every
  operation;
* a parameter value is refused by the bridge's ``set`` exactly when
  ``FmuSidecar.set_params`` refuses it and ``GraphManager.check_params``
  refuses the same tree, and the bridge's message ends with the sidecar's
  reason (the two name the value differently: ``variable 'x'`` against
  ``parameter 'x'``);
* a snapshot edited to carry a non-finite, out-of-bounds, unrepresentable,
  misshapen, missing or extra member is refused by the bridge's
  ``set_state`` and by ``FmuSidecar.set_fmu_state`` with the **same**
  message, and changes nothing on either;
* a refusal leaves all three paths where they were;
* a re-instantiated FMU is a fresh one: the bridge's new instance agrees
  with a fresh sidecar and a reset graph (it used to carry the previous
  instance's state, parameters, inputs and time over);
* a step whose communication point is not the FMU's time, or is not a
  number, is refused with nothing advanced, and ``initialize`` moves the
  clock until the first step only;
* an external input the model description does not export
  (``selected_inputs``) is held at zero: the bridge (through the resolver it
  builds from its description), the sidecar (through the graph's own
  ``_resolve_external_inputs``) and ``GraphManager.step`` (which zero-fills
  an omitted input) agree, and a misspelt input name is refused by all
  three.

A refused *wire* value is the bridge's alone to check: the sidecar and the
graph take external inputs as arrays, so a refused input is checked
against the bridge's documented contract (finite, and representable in the
input's dtype) and the other two paths simply do not apply it.

Tolerance: none -- the three paths call one compiled computation with the
same arguments.

What it cannot see: a defect in the compiled step itself (all three share
it), and anything the model description leaves out (a parameter it does not
export cannot be addressed by ``set``).
"""

from __future__ import annotations

import base64
import io
import math
import re
import socket
import warnings
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import check_bounds
from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.fmu_state import deserialize_fmu_state, serialize_fmu_state
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import (
    FmuTcpBridge,
    recv_message,
    send_message,
    state_of,
    values_of,
)
from maddening.nodes import BallNode, HeatNode, SpringDamperNode, TableNode

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    note,
    assert_trees_identical,
    no_cloud_launch,
    tmp_dir,
)
from tests.property.node_catalogue import f32, floats32


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def _plant():
    """Uniform rate: a spring anchored to an external input, a ball on a table."""
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.25))
    gm.add_node(BallNode("ball", 0.01, initial_position=1.5, elasticity=0.7))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=30.0, damping=2.0,
                                 rest_length=0.4, initial_position=0.5))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _multirate():
    """Two rates, so the FMU state carries ``_meta``'s step counter and
    sub-step phase; an array input and an array output."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.01, n_cells=5, thermal_diffusivity=0.01,
                         initial_temperature=np.linspace(1.0, 2.0, 5).tolist()))
    gm.add_node(SpringDamperNode("spring", 0.02, stiffness=15.0, rest_length=0.3))
    gm.add_edge("spring", "rod", "position", "left_temperature")
    gm.add_external_input("rod", "heat_source", shape=(5,))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _coupled():
    """A coupling group with a predictor: ``_meta`` carries its history."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5,
                                 initial_position=0.2, damping=0.5))
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.add_coupling_group(["ball", "spring"], predictor="linear", acceleration="aitken",
                          max_iterations=5)
    gm.compile()
    return gm


def _held():
    """An input the FMU does not export, read by a node whose "input
    missing" branch differs from zero: a ball's ``table_position``.  The
    graph zero-fills it (the ball rests on a table at height 0); a path
    that leaves it out lets the ball fall through the floor.  The ball
    starts *on* the table, so the two differ from the first step: a ball
    dropped from any height would not reach the table within a generated
    sequence, and the oracle could not tell the paths apart (a mutant
    that left the held input out survived until it did)."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=0.0, elasticity=0.7))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=30.0, damping=2.0,
                                 rest_length=0.4, initial_position=0.5))
    gm.add_external_input("ball", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


GRAPHS = {"plant": _plant, "multirate": _multirate, "coupled": _coupled, "held": _held}

#: ``build_model_description`` keyword arguments per family: ``held``
#: exports the spring's input and holds the ball's at zero.
DESCRIPTIONS: dict[str, dict] = {"held": {"selected_inputs": ["spring.anchor_position"]}}


# ---------------------------------------------------------------------------
# The three paths
# ---------------------------------------------------------------------------

class Model:
    """One graph family exported once: the reference graph whose compiled
    step the sidecars share, its model description, and a second graph
    built the same way for the direct path.

    A plain class rather than a dataclass: Hypothesis prints a dataclass
    argument of a strategy field by field, and these fields are graphs and a
    model description."""

    def __init__(self, ref: GraphManager, md: Any, direct: GraphManager,
                 initial_state: dict, initial_params: dict, label: str = "model") -> None:
        self.ref, self.md, self.direct = ref, md, direct
        self.initial_state, self.initial_params, self.label = (
            initial_state, initial_params, label)
        #: ``(bridge, connection)`` kept open across sequences, or ``None``.
        self.served: Optional[tuple] = None

    def stop(self) -> None:
        if self.served is not None:
            bridge, conn = self.served
            self.served = None
            try:
                conn.close()
            finally:
                bridge.stop()

    def __repr__(self) -> str:          # printed in every falsifying example
        return f"Model({self.label})"

    @classmethod
    def build(cls, factory, label: str = "model", **md_kw) -> "Model":
        ref = factory()
        with warnings.catch_warnings():
            # selected_inputs that hold an input at zero say so; expected here
            warnings.filterwarnings("ignore", message=".*will be held at zero",
                                    category=UserWarning)
            # and so does an input of a node the FMU does not export (a drawn
            # graph may hold an EVOLVING node), which is held at zero as well
            warnings.filterwarnings("ignore", message=".*feed node.s. this FMU does not export",
                                    category=UserWarning)
            md = build_model_description(ref, model_name="Diff",
                                         model_identifier=MODEL_IDENTIFIER, **md_kw)
        direct = factory()
        return cls(ref, md, direct,
                   {n: dict(f) for n, f in ref._state.items()},  # noqa: SLF001
                   _copy_params(ref.params), label)

    @property
    def dt(self) -> float:
        return float(self.ref.timestep)

    def variables(self, causality: str):
        return [v for v in self.md.variables if v.causality == causality and not v.is_clock]

    def sidecar(self, *, resolver: bool = True) -> FmuSidecar:
        """The in-process path's sidecar: with the graph's own input
        resolver, as a sidecar built for a graph should be.  The bridge's
        is built *without* one (``resolver=False``), so the bridge's own
        description-derived resolver is what holds an unexported input at
        zero on that path -- an independent implementation of the rule."""
        return FmuSidecar(SidecarConfig(
            schema_token=self.md.instantiation_token,
            step_fn=self.ref._compiled_step,                       # noqa: SLF001
            initial_state={n: dict(f) for n, f in self.initial_state.items()},
            params=_copy_params(self.initial_params),
            param_specs=self.ref.param_specs(),
            fixed_params=self.md.fixed_parameters,
            input_resolver=self.ref._resolve_external_inputs if resolver else None,  # noqa: SLF001
        ))

    def declared_inputs(self) -> list[tuple[str, str]]:
        """Every external input the graph reads: exported and held."""
        return sorted((ei.target_node, ei.target_field)
                      for ei in self.ref._external_inputs)          # noqa: SLF001


def _nan_canonical(values: np.ndarray) -> np.ndarray:
    """``values`` with every NaN replaced by the one quiet NaN the JSON
    wire's ``"NaN"`` token decodes to."""
    return np.where(np.isnan(values), np.float64(np.nan), values)


def _copy_params(params: dict) -> dict:
    return {s: {o: dict(v) for o, v in owners.items()} for s, owners in params.items()}


class Paths:
    """The bridge (over TCP), the sidecar (in process) and the graph, each
    with the inputs and time the sidecar and the graph do not hold
    themselves."""

    def __init__(self, model: Model, *, persistent: bool = False) -> None:
        """``persistent``: serve this sequence from the model's long-lived
        bridge and connection, reset first (a bridge's ``reset`` restores
        the state, parameters, inputs and time it started with, which is
        what a new bridge holds -- and the oracle checks it at the start of
        every sequence).  Saves a bridge start and stop per example."""
        self.m = model
        self.persistent = persistent
        if persistent and model.served is not None:
            self.bridge, self.conn = model.served
            assert self._wire({"op": "reset"}) == {"ok": True}
        else:
            self.bridge = FmuTcpBridge(model.sidecar(resolver=False), model.md,
                                       master_dt=model.dt)
            self.bridge.start()
            self.conn = self._connect()
            if persistent:
                model.served = (self.bridge, self.conn)
        self.side = model.sidecar()
        self.side_inputs = self._zero_inputs()
        self.side_time = 0.0
        #: Has the FMU stepped since it was instantiated or reset?  Only
        #: then does ``initialize`` still move its clock.
        self.stepped = False
        gm = model.direct
        gm.reset_state()
        gm.params = _copy_params(model.initial_params)
        self.gm_inputs = self._zero_inputs()
        self.gm_time = 0.0
        self.snapshots: list[tuple] = []
        self._tmp = tmp_dir()
        self.tmp = self._tmp.__enter__()

    def close(self) -> None:
        try:
            if not self.persistent:
                try:
                    self.conn.close()
                finally:
                    self.bridge.stop()
        finally:
            self._tmp.__exit__(None, None, None)

    def _zero_inputs(self) -> dict:
        out: dict = {}
        for var in self.m.variables("input"):
            node, field = var.node_field()
            out.setdefault(node, {})[field] = jnp.zeros(var.shape or (), dtype=var.dtype)
        return out

    def _connect(self) -> socket.socket:
        """A new connection past its hello: a new FMU instance."""
        host, port = self.bridge.endpoint.split(":")
        conn = socket.create_connection((host, int(port)), timeout=30)
        send_message(conn, {"op": "hello", "protocol": 2, "binary": False})
        reply = recv_message(conn)
        assert reply is not None and reply["ok"], reply
        return conn

    def _wire(self, message: dict) -> dict:
        send_message(self.conn, message)
        reply = recv_message(self.conn)
        assert reply is not None, "the bridge closed the connection"
        return reply

    # -- reading ------------------------------------------------------------

    def _read(self, var, state: dict, params: dict, inputs: dict, t: float) -> np.ndarray:
        if var.causality == "independent":
            return np.asarray([t], np.float64)
        if var.causality == "parameter":
            node, _, key = var.name.partition(".params.")
            return np.asarray(params["nodes"][node][key], np.float64).ravel()
        node, field = var.node_field()
        if var.causality == "input":
            return np.asarray(inputs[node][field], np.float64).ravel()
        return np.asarray(state[node][field], np.float64).ravel()

    def values(self) -> dict[str, np.ndarray]:
        """``{path: every variable's value, in model-description order}``."""
        vrs = [v.value_reference for v in self.m.md.variables if not v.is_clock]
        reply = self._wire({"op": "get", "vr": vrs})
        assert reply["ok"], reply
        wire = values_of(reply)
        readable = [v for v in self.m.md.variables if not v.is_clock]
        side = np.concatenate([self._read(v, self.side.state, self.side.params,
                                          self.side_inputs, self.side_time)
                               for v in readable])
        gm = self.m.direct
        gm_state = {n: gm.get_node_state(n) for n in gm.node_names}
        direct = np.concatenate([self._read(v, gm_state, gm.params, self.gm_inputs,
                                            self.gm_time) for v in readable])
        return {"bridge": wire, "sidecar": side, "graph": direct}

    def assert_agree(self, what: str) -> None:
        vals = self.values()
        names = [v.name for v in self.m.md.variables if not v.is_clock]
        for other in ("sidecar", "graph"):
            # The JSON wire writes a NaN as the token "NaN", which has no sign
            # and no payload (``json_codec``): a NaN the sidecar holds with
            # its sign bit set reads back positive over the wire.  So a NaN
            # is compared as a NaN, and every other value bit for bit (the
            # states below, which no wire carries, stay bit for bit too).
            a, b = _nan_canonical(vals["bridge"]), _nan_canonical(vals[other])
            if a.tobytes() != b.tobytes():
                bad = [n for n, x, y in zip(names, a, b)
                       if np.float64(x).tobytes() != np.float64(y).tobytes()]
                raise AssertionError(f"{what}: bridge and {other} disagree on {bad[:6]}: "
                                     f"{a.tolist()} vs {b.tolist()}")
        side_state = {n: dict(f) for n, f in self.side.state.items()}
        gm_state = {n: dict(f) for n, f in self.m.direct._state.items()}  # noqa: SLF001
        bridge_state = {n: dict(f) for n, f in self.bridge._sidecar.state.items()}  # noqa: SLF001
        assert_trees_identical(bridge_state, side_state, what=f"{what}: bridge vs sidecar state")
        assert_trees_identical(bridge_state, gm_state, what=f"{what}: bridge vs graph state")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def _value_problem(var, chunk: np.ndarray) -> Optional[str]:
    """The bridge's value check on one variable's slice (``checked_value``):
    finite, and representable in the variable's dtype."""
    if not np.all(np.isfinite(chunk)):
        return f"variable {var.name!r}: value must be finite"
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.all(np.isfinite(chunk.astype(var.dtype))):
            return f"variable {var.name!r}: value does not fit its type {var.dtype}"
    return None


def op_set(paths: Paths, names: list[str], values: list[float]) -> None:
    """One ``set`` through the bridge, held against the sidecar's and the
    graph's opinion of the same values.

    The request is split the way the bridge splits it -- every value
    reference resolved and sized first, then each variable's slice checked in
    order.  A request naming one variable twice is refused when the second
    name is reached (it used to be judged slice by slice, the last slice
    written and the first dropped in silence).
    """
    md = paths.m.md
    by_name = {v.name: v for v in md.variables}
    vrs = [by_name[n].value_reference if n in by_name else 99_999 for n in names]
    reply = paths._wire({"op": "set", "vr": vrs, "values": values})
    note(f"set {list(zip(names, values))} -> {reply}")

    staged: list[tuple[Any, np.ndarray]] = []
    expected_error: Optional[str] = None
    pos = 0
    named: set[str] = set()
    for name in names:
        var = by_name.get(name)
        if var is None:
            expected_error = "unknown value reference"
            break
        if name in named:
            expected_error = "is named more than once in one set"
            break
        named.add(name)
        n = int(np.prod(var.shape)) if var.shape else 1
        chunk = np.asarray(values[pos:pos + n], np.float64).reshape(-1)
        pos += n
        if chunk.size != n:
            expected_error = "expects"
            break
        staged.append((var, chunk.reshape(var.shape or ())))
    if expected_error is None and pos != len(values):
        expected_error = "trailing values"
    params: dict[str, np.ndarray] = {}
    inputs: list[tuple[str, str, np.ndarray]] = []
    value_error: Optional[tuple[Any, np.ndarray, str]] = None
    if expected_error is None:
        for var, chunk in staged:
            # The bridge checks finiteness for every variable first (an
            # output included), then representability for the writable ones.
            problem = _value_problem(var, chunk)
            if problem is not None and (var.causality in ("parameter", "input")
                                        or "finite" in problem):
                value_error = (var, chunk, problem)
                expected_error = problem
                break
            if var.causality == "parameter":
                params[var.name] = chunk
            elif var.causality == "input":
                node, field = var.node_field()
                inputs.append((node, field, chunk.astype(var.dtype)))
            else:
                expected_error = f"variable {var.name!r} ({var.causality}) is read-only"
                break

    if value_error is not None:
        var, chunk, problem = value_error
        assert reply == {"ok": False, "error": f"ValueError: {problem}"}, (reply, problem)
        if var.causality == "parameter":
            # The sidecar's own check of the same value names it differently
            # (``parameter 'x'``) and gives the same reason.
            probe = paths.m.sidecar()
            with pytest.raises(ValueError) as caught:
                probe.set_params({var.name: chunk})
            assert reply["error"].endswith(str(caught.value).split(": ", 1)[-1]), (
                reply["error"], str(caught.value))
        return
    if expected_error is not None:
        assert reply["ok"] is False and expected_error in reply["error"], (
            reply, expected_error)
        return

    # Every slice passed the bridge's value check: what remains is the
    # sidecar's (bounds, tunability) and the graph's (check_params).
    side_error = None
    if params:
        probe = paths.m.sidecar()
        probe._params = _copy_params(paths.side.params)            # noqa: SLF001
        try:
            probe.set_params(dict(params))
        except (KeyError, ValueError) as exc:
            side_error = exc
    graph_refuses = False
    if params:
        # The graph's opinion of the written leaves alone: a generated graph
        # may hold *other* leaves outside their declared bounds (bounds are
        # metadata to a graph), which no write is to blame for.
        written: dict = {"nodes": {}}
        for name, arr in params.items():
            node, _, key = name.partition(".params.")
            ref = np.asarray(paths.m.direct.params["nodes"][node][key])
            written["nodes"].setdefault(node, {})[key] = jnp.asarray(arr.astype(ref.dtype))
        try:
            check_bounds(written, paths.m.direct.param_specs())
        except ValueError:
            graph_refuses = True
    if side_error is not None:
        assert reply["ok"] is False, (f"the bridge accepted what the sidecar refuses "
                                      f"({side_error}): {reply}")
        assert reply["error"].endswith(str(side_error).split(": ", 1)[-1]), (
            reply["error"], str(side_error))
        assert graph_refuses, (
            f"the sidecar refused {params} ({side_error}) and check_params took it")
        return
    assert reply == {"ok": True}, reply
    assert not graph_refuses, f"check_params refuses {params}, which the FMU took"
    if params:
        paths.side.set_params(dict(params))
        for name, arr in params.items():
            node, _, key = name.partition(".params.")
            ref = np.asarray(paths.m.direct.params["nodes"][node][key])
            paths.m.direct.params["nodes"][node][key] = jnp.asarray(arr.astype(ref.dtype))
    for node, field, arr in inputs:
        paths.side_inputs[node][field] = jnp.asarray(arr)
        paths.gm_inputs[node][field] = jnp.asarray(arr)


def op_step(paths: Paths, n: int) -> None:
    reply = paths._wire({"op": "step", "t": paths.side_time, "dt": n * paths.m.dt})
    assert reply["ok"], reply
    for _ in range(n):
        paths.side.step(paths.side_inputs)
        paths.m.direct.step(external_inputs=paths.gm_inputs)
    paths.side_time = paths.side_time + n * paths.m.dt
    paths.gm_time = paths.side_time
    paths.stepped = True
    assert reply["t"] == paths.side_time, (reply["t"], paths.side_time)


def op_bad_step(paths: Paths, kind: str, k: int) -> None:
    """A step the FMU must refuse, with nothing advanced on any path: a
    communication point ``k`` master steps away from the FMU's time (FMI:
    a step starts where the previous one ended), or a time that is not a
    number.  The other two paths are not stepped at all."""
    request: dict = {"op": "step", "t": paths.side_time, "dt": paths.m.dt}
    if kind == "jump":
        request["t"] = paths.side_time + k * paths.m.dt
    elif kind == "string_t":
        request["t"] = str(paths.side_time)
    elif kind == "bool_dt":
        request["dt"] = True
    else:
        request["dt"] = str(paths.m.dt)
    reply = paths._wire(request)
    note(f"bad step {kind} {request} -> {reply}")
    assert reply["ok"] is False, (kind, reply)


def op_initialize(paths: Paths, start: float) -> None:
    """``fmi3EnterInitializationMode``: the start time is the FMU's time,
    accepted until its first step since instantiation or reset."""
    reply = paths._wire({"op": "initialize", "t": start})
    if paths.stepped:
        assert reply["ok"] is False and "has stepped" in reply["error"], reply
        return
    assert reply == {"ok": True, "t": start}, reply
    paths.side_time = paths.gm_time = start


def op_get_state(paths: Paths) -> None:
    reply = paths._wire({"op": "get_state"})
    assert reply["ok"], reply
    blob = state_of(reply)
    snap = paths.side.get_fmu_state()
    path = paths.m.direct.save_state(f"{paths.tmp}/snap{len(paths.snapshots)}.npz")
    paths.snapshots.append((blob, snap, {n: dict(f) for n, f in paths.side_inputs.items()},
                            paths.side_time, path))


def op_set_state(paths: Paths, index: int) -> None:
    blob, snap, inputs, t, path = paths.snapshots[index % len(paths.snapshots)]
    reply = paths._wire({"op": "set_state",
                         "state": base64.b64encode(blob).decode("ascii")})
    # the reply carries the restored time (the C wrapper's clock)
    assert reply == {"ok": True, "t": t}, reply
    paths.side.set_fmu_state(snap)
    paths.side_inputs = {n: dict(f) for n, f in inputs.items()}
    paths.side_time = t
    paths.m.direct.load_state(path)
    paths.gm_inputs = {n: dict(f) for n, f in inputs.items()}
    paths.gm_time = t


def _fresh_oracles(paths: Paths) -> None:
    """The sidecar and the graph where a freshly instantiated FMU starts."""
    paths.side = paths.m.sidecar()
    paths.side_inputs, paths.side_time = paths._zero_inputs(), 0.0
    paths.m.direct.reset_state()
    paths.m.direct.params = _copy_params(paths.m.initial_params)
    paths.gm_inputs, paths.gm_time = paths._zero_inputs(), 0.0
    paths.stepped = False


def op_reset(paths: Paths) -> None:
    assert paths._wire({"op": "reset"}) == {"ok": True}
    _fresh_oracles(paths)


def op_reinstantiate(paths: Paths) -> None:
    """``fmi3FreeInstance`` then ``fmi3InstantiateCoSimulation``: the C
    wrapper closes its connection and opens a new one, which claims the
    bridge's instance slot.  FMI starts the new instance at the model
    description's start values -- which is where a fresh sidecar and a
    reset graph are -- so the oracle needs nothing but those."""
    paths.conn.close()
    paths.conn = paths._connect()
    if paths.persistent:
        paths.m.served = (paths.bridge, paths.conn)
    _fresh_oracles(paths)


def op_misspelt_input(paths: Paths, pick: int, how: str) -> None:
    """An input under a name the graph does not declare: ``GraphManager.step``
    refuses it, the sidecar refuses it with the graph's own words (it has
    the graph's resolver), and the bridge's sidecar refuses it through the
    resolver the bridge built.  Nothing advances anywhere."""
    declared = paths.m.declared_inputs()
    node, field = declared[pick % len(declared)]
    leaf = paths.m.ref._default_external_inputs()[node][field]      # noqa: SLF001
    if how == "node":
        node = node + "x"
    else:
        field = field + "_"
    bad = {node: {field: leaf}}
    with pytest.raises(ValueError, match="does not declare") as graph_refusal:
        paths.m.direct.step(external_inputs=bad)
    with pytest.raises(ValueError) as side_refusal:
        paths.side.step(bad)
    assert str(side_refusal.value) == str(graph_refusal.value)
    bridge_side = paths.bridge._sidecar                             # noqa: SLF001
    with pytest.raises(ValueError, match=re.escape(f"names ['{node}.{field}']")):
        bridge_side._advanced(bridge_side.state, bad)               # noqa: SLF001


# -- edited snapshots ---------------------------------------------------------

def _edit_archive(blob: bytes, edit) -> bytes:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    edit(members, "archive")
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def _edit_snapshot(paths: Paths, snap, edit):
    state, params = deserialize_fmu_state(
        snap, expected_schema_token=paths.m.md.instantiation_token, return_params=True)
    members = {f"s/{n}/{f}": np.asarray(v) for n, fields in state.items()
               for f, v in fields.items()}
    for section, owners in (params or {}).items():
        for owner, leaves in owners.items():
            for k, v in leaves.items():
                members[f"p/{section}/{owner}/{k}"] = np.asarray(v)
    edit(members, "snapshot")
    new_state: dict = {n: {} for n in state}
    new_params: dict = {"nodes": {}, "mappings": {}}
    for key, value in members.items():
        if key.startswith("s/"):
            _, node, field = key.split("/", 2)
            new_state.setdefault(node, {})[field] = value
        elif key.startswith("p/"):
            _, section, owner, k = key.split("/", 3)
            new_params.setdefault(section, {}).setdefault(owner, {})[k] = value
    return serialize_fmu_state(state=new_state, schema_token=paths.m.md.instantiation_token,
                               params=new_params)


def op_set_bad_state(paths: Paths, kind: str, pick: int) -> None:
    """Refusal parity: one edit, applied to the bridge's archive and to the
    sidecar's snapshot alike."""
    reply = paths._wire({"op": "get_state"})
    blob = state_of(reply)
    snap = paths.side.get_fmu_state()
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        state_keys = sorted(k for k in data.files if k.startswith("s/"))
        param_keys = sorted(k for k in data.files if k.startswith("p/nodes/"))
    specs = paths.m.ref.param_specs()["nodes"]
    bounded = [k for k in param_keys
               if specs.get(k.split("/")[2], {}).get(k.split("/")[3]) is not None
               and any(b is not None for b in specs[k.split("/")[2]][k.split("/")[3]].bounds)]

    def edit(members, _fmt):
        if kind == "nan_state":
            key = state_keys[pick % len(state_keys)]
            members[key] = np.full_like(members[key], np.nan) \
                if np.issubdtype(members[key].dtype, np.floating) else members[key]
        elif kind == "inf_param" and param_keys:
            key = param_keys[pick % len(param_keys)]
            members[key] = np.full_like(members[key], np.inf)
        elif kind == "overflow_param" and param_keys:
            key = param_keys[pick % len(param_keys)]
            members[key] = np.full(members[key].shape, 1e39, np.float64)
        elif kind == "out_of_bounds_param" and bounded:
            key = bounded[pick % len(bounded)]
            node, k = key.split("/")[2], key.split("/")[3]
            lo, hi = specs[node][k].bounds
            members[key] = np.full_like(members[key], (lo - 1.0) if lo is not None else hi + 1.0)
        elif kind == "misshapen_state":
            key = state_keys[pick % len(state_keys)]
            members[key] = np.zeros(np.shape(members[key]) + (2,), members[key].dtype)
        elif kind == "missing_state":
            members.pop(state_keys[pick % len(state_keys)])
        elif kind == "extra_state":
            members["s/ghost/x"] = np.zeros((), np.float32)
        elif kind == "missing_param" and param_keys:
            members.pop(param_keys[pick % len(param_keys)])

    bad_blob = _edit_archive(blob, edit)
    bad_snap = _edit_snapshot(paths, snap, edit)
    before = paths.values()
    reply = paths._wire({"op": "set_state",
                         "state": base64.b64encode(bad_blob).decode("ascii")})
    try:
        paths.side.set_fmu_state(bad_snap)
        side_error = None
    except ValueError as exc:
        side_error = exc
    note(f"bad state {kind}: bridge {reply}, sidecar {side_error!r}")
    if side_error is None:
        assert reply["ok"] is True, (kind, reply)
        # The edit was a no-op (no such member): both restored the snapshot.
        return
    assert reply["ok"] is False, (kind, reply, str(side_error))
    assert reply["error"] == f"ValueError: {side_error}", (reply["error"], str(side_error))
    after = paths.values()
    for path in before:
        assert before[path].tobytes() == after[path].tobytes(), (
            f"refused {kind} snapshot changed the {path}")


# ---------------------------------------------------------------------------
# Generated sequences
# ---------------------------------------------------------------------------

BAD_STATES = ("nan_state", "inf_param", "overflow_param", "out_of_bounds_param",
              "misshapen_state", "missing_state", "extra_state", "missing_param")


@st.composite
def _set_request(draw, model: Model):
    writable = model.variables("input") + model.variables("parameter")
    readonly = model.variables("output")
    if not writable:
        # A graph with nothing to set (two tables): every set is a refusal.
        writable = readonly
    if not writable:
        # Nothing exported at all (a deprecated node exports no STABLE
        # surface): the one set left is of a name the FMU does not have.
        return ("set", ["no.such.variable"], [1.0])
    specs = model.ref.param_specs()["nodes"]
    names, values = [], []
    for _ in range(draw(st.integers(min_value=1, max_value=3))):
        roll = draw(st.integers(0, 19))
        if roll == 0 and readonly:
            var = draw(st.sampled_from(readonly))
        elif roll == 1:
            names.append("no.such.variable")
            values.append(1.0)
            continue
        else:
            var = draw(st.sampled_from(writable))
        n = int(np.prod(var.shape)) if var.shape else 1
        kind = draw(st.sampled_from(["valid"] * 6 + ["boundary", "outside", "nan", "inf",
                                                     "overflow", "short"]))
        lo, hi = -2.0, 2.0
        bounds = (None, None)
        if var.causality == "parameter":
            node, _, key = var.name.partition(".params.")
            spec = specs.get(node, {}).get(key)
            current = float(np.asarray(model.initial_params["nodes"][node][key]).ravel()[0])
            lo, hi = sorted((current * 0.5, current * 1.5)) if current else (0.0, 1.0)
            bounds = spec.bounds if spec is not None else (None, None)
        lo, hi = f32(lo), f32(hi)
        if kind == "valid":
            vals = [draw(floats32(lo, hi)) for _ in range(n)]
        elif kind == "boundary":
            b = [x for x in bounds if x is not None] or [0.0]
            vals = [draw(st.sampled_from(b))] * n
        elif kind == "outside":
            b = ([bounds[0] - 1.0] if bounds[0] is not None else []) + \
                ([bounds[1] + 1.0] if bounds[1] is not None else [])
            vals = [draw(st.sampled_from(b or [-1.0]))] * n
        elif kind == "nan":
            vals = [math.nan] * n
        elif kind == "inf":
            vals = [draw(st.sampled_from([math.inf, -math.inf]))] * n
        elif kind == "overflow":
            vals = [1e39] * n
        else:
            vals = [0.0] * max(n - 1, 0)
        names.append(var.name)
        values.extend(vals)
    return ("set", names, values)


BAD_STEPS = ("jump", "string_t", "bool_dt", "string_dt")


def _ops(model: Model):
    ops = [
        _set_request(model),
        st.tuples(st.just("step"), st.integers(min_value=1, max_value=3)),
        st.just(("get_state",)),
        st.tuples(st.just("set_state"), st.integers(min_value=0, max_value=5)),
        st.just(("reset",)),
        st.tuples(st.just("bad_state"), st.sampled_from(BAD_STATES),
                  st.integers(min_value=0, max_value=7)),
        st.just(("reinstantiate",)),
        st.tuples(st.just("bad_step"), st.sampled_from(BAD_STEPS),
                  st.sampled_from([-2, -1, 1, 2, 1000])),
        st.tuples(st.just("initialize"), st.sampled_from([0.0, 0.5, 2.0])),
    ]
    if model.declared_inputs():
        ops.append(st.tuples(st.just("misspelt_input"), st.integers(min_value=0, max_value=7),
                             st.sampled_from(["node", "field"])))
    return st.lists(st.one_of(*ops), min_size=1, max_size=10)


def run_sequence(model: Model, ops: list, *, persistent: bool = False) -> None:
    paths = Paths(model, persistent=persistent)
    try:
        paths.assert_agree("at start")
        for op in ops:
            note(f"op: {op}")
            event(f"op {op[0]}")
            if op[0] == "set":
                op_set(paths, op[1], op[2])
            elif op[0] == "step":
                op_step(paths, op[1])
            elif op[0] == "get_state":
                op_get_state(paths)
            elif op[0] == "set_state":
                if not paths.snapshots:
                    continue
                op_set_state(paths, op[1])
            elif op[0] == "reset":
                op_reset(paths)
            elif op[0] == "bad_state":
                op_set_bad_state(paths, op[1], op[2])
            elif op[0] == "reinstantiate":
                op_reinstantiate(paths)
            elif op[0] == "bad_step":
                op_bad_step(paths, op[1], op[2])
            elif op[0] == "initialize":
                op_initialize(paths, op[1])
            elif op[0] == "misspelt_input":
                op_misspelt_input(paths, op[1], op[2])
            paths.assert_agree(f"after {op[0]}")
    finally:
        paths.close()


_MODELS: dict[str, Model] = {}


@pytest.fixture(scope="module", autouse=True)
def _stop_served_bridges():
    yield
    for model in _MODELS.values():
        model.stop()


def _model(name: str) -> Model:
    if name not in _MODELS:
        _MODELS[name] = Model.build(GRAPHS[name], name, **DESCRIPTIONS.get(name, {}))
    return _MODELS[name]


@pytest.mark.parametrize("graph", sorted(GRAPHS))
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_the_bridge_the_sidecar_and_the_graph_agree_on_every_sequence(graph, data):
    model = _model(graph)
    run_sequence(model, data.draw(_ops(model), label="ops"), persistent=True)


# Per push: tests/property/test_differential_fmu.py::test_the_bridge_the_sidecar_and_the_graph_agree_on_every_sequence
@pytest.mark.slow  # two graphs and a model description built per example
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_the_three_fmu_paths_agree_on_a_generated_graph(data):
    from tests.property.strategies import ALL_NODE_KINDS, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=3,
                                     allow_mappings=False), label="recipe")
    # ``strategies`` draws ParamSpec bounds as metadata, so a recipe may start
    # outside them, and a calibrated leaf may be pushed past its node's own
    # bound (``param_overrides`` scales by up to 2).  Both are drawn: an FMU
    # restores its own snapshot whatever its parameters
    # (``test_an_fmu_restores_its_own_snapshot_whatever_its_parameters``).
    note(f"recipe: {recipe}")
    event(f"starts inside its declared bounds: {_within_declared_bounds(recipe.build())}")
    model = Model.build(recipe.build)
    run_sequence(model, data.draw(_ops(model), label="ops"))


def _within_declared_bounds(gm: GraphManager) -> bool:
    try:
        check_bounds(gm.params, gm.param_specs())
    except ValueError:
        return False
    return True


def _spring_with_a_bound_it_starts_outside():
    from maddening.core.params import ParamSpec

    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=30.0, rest_length=0.4))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    # A graph takes a ParamSpec as metadata and runs regardless.
    gm.set_param_spec("spring", "stiffness", ParamSpec(bounds=(50.0, None)))
    return gm


def test_an_fmu_restores_its_own_snapshot_whatever_its_parameters():
    """Found by ``test_the_three_fmu_paths_agree_on_a_generated_graph``.  The
    graph runs with ``stiffness = 30`` under a declared lower bound of 50
    (bounds are metadata to a graph); the FMU exported it, stepped it, and
    then could not restore the state it had handed out itself (bridge
    ``set_state`` and ``FmuSidecar.set_fmu_state`` alike), while
    ``GraphManager.load_state`` restored the graph's checkpoint."""
    model = Model.build(_spring_with_a_bound_it_starts_outside, "out-of-bounds")
    paths = Paths(model)
    try:
        op_step(paths, 2)
        op_get_state(paths)
        op_step(paths, 1)
        # The graph's own checkpoint restores; the FMU's own snapshot must too.
        op_set_state(paths, 0)
        paths.assert_agree("after restoring the FMU's own snapshot")
    finally:
        paths.close()
