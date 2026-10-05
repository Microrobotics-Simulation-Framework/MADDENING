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

A second family of sequences (the end of this module) adds the compiled C
wrapper as a fourth path, through ``ctypes`` against a bridge of its own, and
``node.params`` writes -- a constant and a structural value, with and
without ``compile()`` before ``build_model_description`` -- on a multi-rate
graph and on a sub-cycled coupling group, each write followed by an export
wired as the guide wires one.  The graph path takes the same write and runs
what ``gm.step`` runs.  A structural write pending at the export must be
refused, naming ``compile()`` (B1-H3, fixed in 0.4.0: the FMU ran the
model from before the write).

Tolerance: none -- the paths call one compiled computation with the same
arguments.

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
                 initial_state: dict, initial_params: dict, label: str = "model",
                 md_kw: Optional[dict] = None) -> None:
        self.ref, self.md, self.direct = ref, md, direct
        self.initial_state, self.initial_params, self.label = (
            initial_state, initial_params, label)
        #: ``build_model_description`` keywords, for an export after a write.
        self.md_kw = dict(md_kw or {})
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
                   _copy_params(ref.params), label, md_kw)

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
        # one name per value: an array variable reads as several
        names = [v.name for v in self.m.md.variables if not v.is_clock
                 for _ in range(int(np.prod(v.shape)) if v.shape else 1)]
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


# ===========================================================================
# node.params writes before an export, and the C wrapper as a fourth path
# ===========================================================================
#
# FMU-040: an FMU exported after a ``node.params`` write runs what the graph
# runs, "like every other reader of gm.params".  The sequences below write a
# node's constant (a ``gm.params`` leaf) or a structural value (a
# ``HeatNode``'s ``stencil_order``) into the graph the FMU is exported from,
# with or without ``compile()`` before ``build_model_description``, and export
# again, wired the way the guide wires a sidecar (``step_fn=gm._compiled_step``,
# ``params=gm.params``, ``initial_state=gm._state``).  The graph path gets the
# same write and runs whatever ``gm.step`` runs (it recompiles a dirty graph).
# The reference graph's step is traced before the first write -- as it is in
# any session that ran the graph, or stepped an FMU built from it -- since
# a write before the first trace is taken in by the trace and proves nothing.
#
# The fourth path is the compiled C wrapper an importer loads, driven through
# ``ctypes`` against a bridge of its own over the same compiled step: every
# value it reads through ``fmi3Get*`` must equal the TCP bridge's, its own
# bridge's full state must equal the TCP bridge's, and each operation it can
# express (set, step, get / set / deserialize an FMU state, reset, a new
# instance, ``fmi3EnterInitializationMode``, a step off the FMU's time) must
# succeed exactly when the bridge's does.  A JSON-level operation the C API
# cannot express (a string time, a misspelt input) is skipped on that path.
# Without a C compiler the comparison runs with three paths and says so.

import ctypes                                                    # noqa: E402
import os                                                        # noqa: E402

from maddening.fmi.package import build_fmu_binary, find_c_compiler  # noqa: E402

#: numpy dtype name -> (FMI 3.0 type, the C type of its fmi3Get / fmi3Set).
_FMI_C_TYPES = {
    "float32": ("Float32", ctypes.c_float), "float64": ("Float64", ctypes.c_double),
    "int8": ("Int8", ctypes.c_int8), "uint8": ("UInt8", ctypes.c_uint8),
    "int16": ("Int16", ctypes.c_int16), "uint16": ("UInt16", ctypes.c_uint16),
    "int32": ("Int32", ctypes.c_int32), "uint32": ("UInt32", ctypes.c_uint32),
    "int64": ("Int64", ctypes.c_int64), "uint64": ("UInt64", ctypes.c_uint64),
    "bool": ("Boolean", ctypes.c_bool),
}
_FMI_TYPE = {dtype: fmi_type for dtype, (fmi_type, _) in _FMI_C_TYPES.items()}


class WrapperPath:
    """The C wrapper, one instance at a time, against a bridge of its own."""

    OK = 0

    def __init__(self, wrapper, model: Model) -> None:
        self.w = wrapper
        self.m = model
        self.bridge: Optional[FmuTcpBridge] = None
        self.inst = None
        self.states: list = []
        #: The one ``fmi3FMUState`` variable a rollback master keeps
        #: (:meth:`save_for_rollback`).
        self.rollback = ctypes.c_void_p()
        self.time = 0.0
        self.restart()

    def restart(self) -> None:
        """A new bridge over the model's current export and a new instance:
        what an importer has after the FMU is exported again."""
        self.close()
        self.bridge = FmuTcpBridge(self.m.sidecar(resolver=False), self.m.md,
                                   master_dt=self.m.dt)
        self.bridge.start()
        self.inst = self._instantiate()
        self.time = 0.0

    #: FMI 3.0's co-simulation state, as the wrapper enforces it:
    #: ``"instantiated"`` (after instantiation or ``fmi3Reset``: no ``Get``,
    #: no ``DoStep``), ``"initialization"`` and ``"step"``.
    phase = "instantiated"

    def _instantiate(self):
        assert self.bridge is not None
        saved = os.environ.get("MADDENING_FMU_ENDPOINT")
        os.environ["MADDENING_FMU_ENDPOINT"] = self.bridge.endpoint
        try:
            inst = self.w.instantiate(self.m.md.instantiation_token)
        finally:
            if saved is None:
                os.environ.pop("MADDENING_FMU_ENDPOINT", None)
            else:
                os.environ["MADDENING_FMU_ENDPOINT"] = saved
        assert inst, self.w.logs[-3:]
        self.phase = "instantiated"
        return inst

    def _free_states(self) -> None:
        for handle in (*self.states, self.rollback):
            if handle is not None:
                self.w.lib.fmi3FreeFMUState(self.inst, ctypes.byref(handle))
        self.states = []
        self.rollback = ctypes.c_void_p()

    def close(self) -> None:
        if self.inst is not None:
            self._free_states()
            self.w.lib.fmi3FreeInstance(self.inst)
            self.inst = None
        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None

    # -- the operations ---------------------------------------------------

    def _into_step_mode(self) -> None:
        """Where an importer is before its first ``fmi3DoStep``: from
        Instantiated, ``fmi3EnterInitializationMode`` at the FMU's time and
        ``fmi3ExitInitializationMode``; from Initialization Mode, the exit."""
        if self.phase == "instantiated":
            assert self.w.lib.fmi3EnterInitializationMode(
                self.inst, False, 0.0, self.time, False, 0.0) == self.OK, self.w.logs[-2:]
            self.phase = "initialization"
        if self.phase == "initialization":
            assert self.w.lib.fmi3ExitInitializationMode(self.inst) == self.OK, \
                self.w.logs[-2:]
            self.phase = "step"

    def values(self) -> Optional[np.ndarray]:
        """Every value through ``fmi3Get*``; ``None`` in the Instantiated
        state, where FMI 3.0 allows no getter (the bridge's state is still
        compared)."""
        if self.phase == "instantiated":
            return None
        out = []
        for var in self.m.md.variables:
            if var.is_clock:
                continue
            n = int(np.prod(var.shape)) if var.shape else 1
            status, got = self.w.call("Get", _FMI_TYPE[var.dtype], self.inst,
                                      [var.value_reference], [0] * n)
            assert status == self.OK, (var.name, self.w.logs[-2:])
            out.extend(float(x) for x in got)
        return np.asarray(out, np.float64)

    def set(self, names: list[str], values: list[float]) -> Optional[bool]:
        """The same ``set`` through the typed setter; ``None`` when the
        request mixes FMI types, which one C call cannot carry."""
        by_name = {v.name: v for v in self.m.md.variables}
        types = {_FMI_TYPE[by_name[n].dtype] if n in by_name else "Float32" for n in names}
        if len(types) != 1:
            return None
        vrs = [by_name[n].value_reference if n in by_name else 99_999 for n in names]
        status, _ = self.w.call("Set", types.pop(), self.inst, vrs, list(values))
        return status == self.OK

    def step(self, t: float, h: float) -> bool:
        self._into_step_mode()
        status, last = self.w.step(self.inst, t, h)
        if status == self.OK:
            self.time = last
        return status == self.OK

    def get_state(self) -> None:
        handle = ctypes.c_void_p()
        assert self.w.lib.fmi3GetFMUState(self.inst, ctypes.byref(handle)) == self.OK, \
            self.w.logs[-2:]
        self.states.append(handle)

    def set_state(self, index: int) -> bool:
        return self.w.lib.fmi3SetFMUState(self.inst, self.states[index]) == self.OK

    def save_for_rollback(self) -> None:
        """``fmi3GetFMUState`` into the one variable a rollback master keeps,
        before every step.  FMI 3.0 has a non-NULL ``*FMUState`` be a state
        "no longer needed" that the FMU overwrites: the wrapper must hand
        the same object back (it allocated a new one over it on every call,
        which nothing could free)."""
        held = self.rollback.value
        assert self.w.lib.fmi3GetFMUState(self.inst, ctypes.byref(self.rollback)) == self.OK, \
            self.w.logs[-2:]
        assert self.rollback.value and (held is None or self.rollback.value == held), \
            "fmi3GetFMUState returned another object than the one it was handed"

    def roll_back(self) -> bool:
        return self.w.lib.fmi3SetFMUState(self.inst, self.rollback) == self.OK

    def set_blob(self, blob: bytes) -> bool:
        handle = ctypes.c_void_p()
        assert self.w.lib.fmi3DeserializeFMUState(self.inst, blob, len(blob),
                                                  ctypes.byref(handle)) == self.OK
        try:
            return self.w.lib.fmi3SetFMUState(self.inst, handle) == self.OK
        finally:
            self.w.lib.fmi3FreeFMUState(self.inst, ctypes.byref(handle))

    def reset(self) -> bool:
        ok = self.w.lib.fmi3Reset(self.inst) == self.OK
        if ok:
            self.time = 0.0
            self.phase = "instantiated"
        return ok

    def reinstantiate(self) -> None:
        """``fmi3FreeInstance`` and a new instance.  The FMU states saved so
        far go across the way an importer carries them: serialized before
        the free (``fmi3SerializeFMUState``) and deserialized into the new
        instance -- the TCP path's snapshots survive a new instance too."""
        blobs = []
        for handle in self.states:
            size = ctypes.c_size_t()
            assert self.w.lib.fmi3SerializedFMUStateSize(self.inst, handle,
                                                          ctypes.byref(size)) == self.OK
            buf = (ctypes.c_char * size.value)()
            assert self.w.lib.fmi3SerializeFMUState(self.inst, handle, buf,
                                                    size.value) == self.OK
            blobs.append(bytes(buf))
        self._free_states()
        self.w.lib.fmi3FreeInstance(self.inst)
        self.inst = self._instantiate()
        self.time = 0.0
        for blob in blobs:
            handle = ctypes.c_void_p()
            assert self.w.lib.fmi3DeserializeFMUState(self.inst, blob, len(blob),
                                                      ctypes.byref(handle)) == self.OK
            self.states.append(handle)

    def initialize(self, start: float) -> bool:
        """The bridge's ``initialize``.  From Instantiated it is
        ``fmi3EnterInitializationMode``.  A second one before the first
        step, which the bridge protocol allows and FMI 3.0 does not, is
        applied to the wrapper's bridge in process (the C API cannot express
        it); in Step Mode the wrapper refuses it, as the bridge refuses an
        ``initialize`` after a step."""
        if self.phase == "initialization":
            assert self.bridge is not None
            ok = self.bridge.handle({"op": "initialize", "t": start}).get("ok") is True
        else:
            ok = self.w.lib.fmi3EnterInitializationMode(self.inst, False, 0.0, start,
                                                        False, 0.0) == self.OK
            if ok:
                self.phase = "initialization"
        if ok:
            self.time = start
        return ok


def _load_wrapper():
    """The compiled wrapper with every prototype the fourth path calls, or
    ``None`` without a C compiler."""
    if find_c_compiler() is None:
        return None
    import tempfile

    from tests.fmi.test_c_wrapper_refuses_what_it_used_to_coerce import _Wrapper

    directory = tempfile.mkdtemp(prefix="maddening-diff-wrapper-")
    wrapper = _Wrapper(build_fmu_binary(directory))
    lib = wrapper.lib
    # every FMI numeric type's getter and setter, for the typed battery
    for fmi_type, ctype in _FMI_C_TYPES.values():
        wrapper.ctypes_of[fmi_type] = ctype
        for op in ("Get", "Set"):
            f = getattr(lib, f"fmi3{op}{fmi_type}")
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                          ctypes.POINTER(ctype), ctypes.c_size_t]
    for name in ("fmi3Reset", "fmi3Terminate", "fmi3ExitInitializationMode"):
        getattr(lib, name).restype = ctypes.c_int
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.fmi3GetFMUState.restype = ctypes.c_int
    lib.fmi3GetFMUState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.fmi3SerializedFMUStateSize.restype = ctypes.c_int
    lib.fmi3SerializedFMUStateSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_size_t)]
    lib.fmi3SerializeFMUState.restype = ctypes.c_int
    lib.fmi3SerializeFMUState.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                                          ctypes.c_size_t]
    return wrapper


_WRAPPER: list = []


def wrapper_or_none():
    if not _WRAPPER:
        _WRAPPER.append(_load_wrapper())
    return _WRAPPER[0]


class FourPaths(Paths):
    """:class:`Paths` with the C wrapper as a fourth path, and exports
    that follow ``node.params`` writes."""

    def __init__(self, model: Model, wrapper) -> None:
        super().__init__(model)
        self.cw = WrapperPath(wrapper, model) if wrapper is not None else None

    def close(self) -> None:
        try:
            if self.cw is not None:
                self.cw.close()
        finally:
            super().close()

    def restart(self) -> None:
        """Serve the model's current export: a new bridge and connection, a
        new in-process sidecar, a new C-wrapper instance, the graph reset."""
        self.conn.close()
        self.bridge.stop()
        self.bridge = FmuTcpBridge(self.m.sidecar(resolver=False), self.m.md,
                                   master_dt=self.m.dt)
        self.bridge.start()
        self.conn = self._connect()
        self.snapshots = []
        _fresh_oracles(self)
        self.m.direct.reset_state()
        if self.cw is not None:
            self.cw.restart()

    def assert_agree(self, what: str) -> None:
        super().assert_agree(what)
        if self.cw is None:
            return
        read = self.cw.values()
        wire = _nan_canonical(self.values()["bridge"])
        mine = wire if read is None else _nan_canonical(read)
        names = [v.name for v in self.m.md.variables if not v.is_clock
                 for _ in range(int(np.prod(v.shape)) if v.shape else 1)]
        if wire.tobytes() != mine.tobytes():
            raise AssertionError(f"{what}: the C wrapper reads {mine.tolist()}, the bridge "
                                 f"{wire.tolist()} ({names})")
        assert self.cw.bridge is not None
        assert_trees_identical(
            {n: dict(f) for n, f in self.bridge._sidecar.state.items()},    # noqa: SLF001
            {n: dict(f) for n, f in self.cw.bridge._sidecar.state.items()},  # noqa: SLF001
            what=f"{what}: TCP bridge vs C wrapper's bridge state")


def op_write_and_export(paths: FourPaths, node: str, key: str, value: Any,
                        compile_first: bool) -> None:
    """``node.params[key] = value`` on the graph the FMU comes from and on
    the graph path; ``compile()`` the former or not; export again as the
    guide does, and serve the new export on every FMU path.

    A structural write (an ``int``: ``stencil_order``) pending at the export
    is refused, naming ``compile()`` (B1-H3's fix, FMU-040): the FMU would
    otherwise be built over the step from before the write.  The advice is
    then followed and the export made again."""
    m = paths.m
    for gm in (m.ref, m.direct):
        gm.get_node(node).params[key] = value
    if compile_first:
        m.ref.compile()

    def export():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*will be held at zero",
                                    category=UserWarning)
            return build_model_description(m.ref, model_name="Diff",
                                           model_identifier=MODEL_IDENTIFIER, **m.md_kw)
    if isinstance(value, int) and not compile_first:
        with pytest.raises(ValueError, match=r"compile\(\)"):
            export()
        m.ref.compile()
    m.md = export()
    m.initial_state = {n: dict(f) for n, f in m.ref._state.items()}       # noqa: SLF001
    m.initial_params = _copy_params(m.ref.params)
    paths.restart()


def _mirror(paths: FourPaths, op: tuple, bridge_ok: Optional[bool]) -> None:
    """Drive the C wrapper through ``op`` and hold its verdict to the
    bridge's (``bridge_ok``, where the bridge's verdict is not implied)."""
    cw = paths.cw
    if cw is None:
        return
    kind = op[0]
    if kind == "set":
        verdict = cw.set(op[1], op[2])
        if verdict is not None:
            assert verdict is bridge_ok, (op, verdict, cw.w.logs[-2:])
    elif kind == "step":
        assert cw.step(cw.time, op[1] * paths.m.dt), (op, cw.w.logs[-2:])
    elif kind == "get_state":
        cw.get_state()
    elif kind == "set_state":
        assert cw.set_state(op[1] % len(cw.states)), (op, cw.w.logs[-2:])
        cw.time = paths.side_time
    elif kind == "reset":
        assert cw.reset(), cw.w.logs[-2:]
    elif kind == "reinstantiate":
        cw.reinstantiate()
    elif kind == "initialize":
        assert cw.initialize(op[1]) is bridge_ok, (op, cw.w.logs[-2:])
    elif kind == "bad_step" and op[1] == "jump":
        assert not cw.step(cw.time + op[2] * paths.m.dt, paths.m.dt), op


def run_write_sequence(model: Model, ops: list, wrapper) -> None:
    """:func:`run_sequence` over four paths, with ``write`` operations."""
    paths = FourPaths(model, wrapper)
    original_wire = paths._wire

    def recording_wire(message: dict) -> dict:
        reply = original_wire(message)
        paths._last_reply = reply                                   # noqa: SLF001
        return reply

    paths._wire = recording_wire                                    # type: ignore[method-assign]
    try:
        paths.assert_agree("at start")
        for op in ops:
            note(f"op: {op}")
            kind = op[0]
            paths._last_reply = {}                                  # noqa: SLF001
            if kind == "write":
                op_write_and_export(paths, *op[1:])
            elif kind == "set":
                op_set(paths, op[1], op[2])
                _mirror(paths, op, paths._last_reply.get("ok") is True)  # noqa: SLF001
            elif kind == "step":
                op_step(paths, op[1])
                _mirror(paths, op, True)
            elif kind == "get_state":
                op_get_state(paths)
                _mirror(paths, op, True)
            elif kind == "set_state":
                if not paths.snapshots:
                    continue
                op_set_state(paths, op[1])
                _mirror(paths, op, True)
            elif kind == "reset":
                op_reset(paths)
                _mirror(paths, op, True)
            elif kind == "reinstantiate":
                op_reinstantiate(paths)
                _mirror(paths, op, True)
            elif kind == "initialize":
                op_initialize(paths, op[1])
                _mirror(paths, op, paths._last_reply.get("ok") is True)  # noqa: SLF001
            elif kind == "bad_step":
                op_bad_step(paths, op[1], op[2])
                _mirror(paths, op, False)
            elif kind == "bad_state":
                op_set_bad_state(paths, op[1], op[2])
            paths.assert_agree(f"after {kind}")
    finally:
        paths.close()


# -- graphs whose writes matter ------------------------------------------------

def _multirate_rod():
    """``_multirate`` with a rod long enough for a fourth-order stencil."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.01, n_cells=8, thermal_diffusivity=0.01,
                         initial_temperature=np.linspace(1.0, 2.0, 8).tolist()))
    gm.add_node(SpringDamperNode("spring", 0.02, stiffness=15.0, rest_length=0.3))
    gm.add_edge("spring", "rod", "position", "left_temperature")
    gm.add_external_input("rod", "heat_source", shape=(8,))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _subcycled_rods():
    """Two rods end to end in a sub-cycling coupling group, the right one
    at twice the left's timestep, the left one heated from outside."""
    gm = GraphManager()
    gm.add_node(HeatNode("left", 0.01, n_cells=8, thermal_diffusivity=0.01,
                         initial_temperature=np.linspace(1.0, 2.0, 8).tolist()))
    gm.add_node(HeatNode("right", 0.02, n_cells=8, thermal_diffusivity=0.01,
                         initial_temperature=0.5))
    gm.add_edge("left", "right", "temperature", "left_temperature", transform="extract_last")
    gm.add_edge("right", "left", "temperature", "right_temperature",
                transform="extract_first")
    gm.add_external_input("left", "heat_source", shape=(8,))
    gm.add_coupling_group(["left", "right"], subcycling=True, max_iterations=4,
                          predictor="linear")
    gm.compile()
    return gm


#: Per graph: a constant write (a ``gm.params`` leaf) and a structural one.
WRITE_GRAPHS = {
    "multirate": (_multirate_rod, {"constant": ("spring", "stiffness", 40.0),
                                   "structural": ("rod", "stencil_order", 4)}),
    "subcycled": (_subcycled_rods, {"constant": ("right", "thermal_diffusivity", 0.02),
                                    "structural": ("left", "stencil_order", 4)}),
}


def write_model(graph: str) -> Model:
    """A fresh model whose reference graph has traced its step once."""
    factory, _ = WRITE_GRAPHS[graph]
    model = Model.build(factory, graph)
    ref = model.ref
    ref.step(external_inputs=ref._default_external_inputs())        # noqa: SLF001
    ref.reset_state()
    return model


def _ops_around_a_write(write: tuple, compile_first: bool) -> list:
    """A fixed sequence: run, save, write and export, run, restore the new
    export's own snapshot, reset, and write the same key back."""
    node, key, value = write
    return [("step", 2), ("get_state",), ("write", node, key, value, compile_first),
            ("step", 3), ("get_state",), ("step", 1), ("set_state", 0), ("step", 2),
            ("reset",), ("step", 1), ("reinstantiate",), ("step", 2)]


def _write_cases():
    for graph in sorted(WRITE_GRAPHS):
        for kind in ("constant", "structural"):
            for compile_first in (False, True):
                yield pytest.param(graph, kind, compile_first,
                                   id=f"{graph}-{kind}-{'compiled' if compile_first else 'pending'}")


@pytest.mark.parametrize("graph, kind, compile_first", list(_write_cases()))
def test_four_fmu_paths_agree_after_a_node_params_write(graph, kind, compile_first):
    """One write before an export, with and without ``compile()`` first,
    on a multi-rate and a sub-cycled graph: the graph, the sidecar, the
    bridge and the C wrapper agree on every value and the full state after
    every operation."""
    model = write_model(graph)
    try:
        write = WRITE_GRAPHS[graph][1][kind]
        run_write_sequence(model, _ops_around_a_write(write, compile_first),
                           wrapper_or_none())
    finally:
        model.stop()


def test_the_c_wrapper_is_a_fourth_path_here():
    """The four-way comparison runs four paths wherever a C compiler is
    available (CI has one): a harness that silently fell back to three
    would compare less than it claims."""
    if find_c_compiler() is None:
        pytest.skip("no C compiler: the FMU write sequences compare three paths here")
    assert wrapper_or_none() is not None


@st.composite
def _write_ops(draw, model: Model, graph: str):
    """Generated operations with writes among them: a constant or a
    structural write, with or without ``compile()``, anywhere in the
    sequence, each followed by an export."""
    writes = WRITE_GRAPHS[graph][1]
    base = _ops(model)
    ops = draw(base)
    n_writes = draw(st.integers(min_value=1, max_value=2))
    for _ in range(n_writes):
        kind = draw(st.sampled_from(["constant", "structural"]))
        node, key, value = writes[kind]
        compile_first = draw(st.booleans())
        at = draw(st.integers(min_value=0, max_value=len(ops)))
        ops.insert(at, ("write", node, key, value, compile_first))
    return [op for op in ops if op[0] != "misspelt_input"]


# Per push: tests/property/test_differential_fmu.py::test_four_fmu_paths_agree_after_a_node_params_write
@pytest.mark.slow  # two graphs compiled, a C-wrapper instance and an export per write, per example
@pytest.mark.parametrize("graph", sorted(WRITE_GRAPHS))
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_four_fmu_paths_agree_on_sequences_with_node_params_writes(graph, data):
    model = write_model(graph)
    try:
        run_write_sequence(model, data.draw(_write_ops(model, graph), label="ops"),
                           wrapper_or_none())
    finally:
        model.stop()


# ===========================================================================
# Clock patterns: how an importer keeps its time, with restores in between
# ===========================================================================
#
# FMU-023: the time the FMU reports stays within a millionth plus a tenth of
# a master step of the time it has simulated, whatever arithmetic the
# importer keeps its clock in -- a point computed as ``start + k * h``, a
# running sum, step sizes that are all a little long, points that are all a
# little late -- and an importer whose errors add up past that is refused,
# with nothing advanced.  The bound is counted from the start time and the
# master steps taken since, and that count is part of the FMU state: saving
# and restoring a state (``fmi3GetFMUState`` / ``fmi3SetFMUState``) must not
# move it.  It did -- a restore started the count again at the restored time
# -- so a master that restored between steps was never refused (B1 round 9,
# F4): its drift was forgiven at every restore.
#
# Each pattern below runs three ways -- no restore, a save and restore after
# every step, and one rollback part-way (a state saved, two more steps or a
# refusal, back to it, and on) -- over all four paths: the TCP bridge, the
# sidecar and the graph (which have no clock, and must hold the state the
# accepted steps produce, restored as the bridge restores), and the compiled
# wrapper against a bridge of its own, saving into the one ``fmi3FMUState``
# variable a rollback master keeps.  The oracle: a restore changes no reply.
# Every step gets, to the bit, the reply it gets without one, including a
# step taken again after going back; and the paths agree on every value and
# the full state after every operation.

from fractions import Fraction                                  # noqa: E402


def _clock_graph(dt: float):
    def build():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("spring", dt, stiffness=30.0, damping=2.0,
                                     rest_length=0.4, initial_position=0.5))
        gm.add_external_input("spring", "anchor_position")
        gm.compile()
        return gm
    return build


#: ``name: (master step, start time, clock, steps tried)``, the clock as
#: ``clock(k, t_fmu, t_own) -> (t, h)``: the step index, the time the FMU
#: reports and the importer's own running clock in; the communication point
#: and the step size out.
CLOCKS = {
    # FMPy's arithmetic: one rounding per point.  Never refused.
    "start + k * h": (7e-10, 20000.0, lambda k, t_fmu, t_own: (20000.0 + k * 7e-10, 7e-10), 40),
    # An honest running sum, where an ulp of the time is 0.005 of the step
    # and every sum rounds the same way: refused after 47 steps.
    "running sum": (7e-10, 20000.0, lambda k, t_fmu, t_own: (t_own, 7e-10), 120),
    # Every step size 0.45 millionths of a step long, on a running sum.
    "long steps": (1e-2, 0.0, lambda k, t_fmu, t_own: (t_own, 1e-2 * (1 + 0.45e-6)), 12),
    # Every point 0.05 of a step past the time the FMU reports, which the
    # communication-point tolerance alone adopts.
    "late points": (7e-10, 20000.0, lambda k, t_fmu, t_own: (t_fmu + 0.05 * 7e-10, 7e-10), 12),
}

#: The claimed bound on |reported - simulated|, in master steps: a millionth
#: plus a tenth, and the float64 clock's own rounding (:func:`drift_bound`).
DRIFT_BOUND = Fraction(1, 10) + Fraction(1, 10 ** 6)
CLOCK_ROUNDING_ULPS = 3


def drift_bound(dt: float, start: float, steps: int, reported: float) -> Fraction:
    """The bound FMU-023 states on |reported - simulated|, in master steps,
    in exact arithmetic: a millionth plus a tenth of a step, plus three ulps
    of the largest time involved (the reported time, the start time, the
    time elapsed), which is what the clock's own float64 sums can round by
    -- the reported time is one sum, the simulated time it was checked
    against another.  Where the clock resolves a step (16 ulps within a
    tenth of one) that is under 0.02 of a step."""
    simulated = Fraction(start) + steps * Fraction(dt)
    biggest = max(abs(reported), abs(float(simulated)), abs(start), steps * dt)
    return DRIFT_BOUND + CLOCK_ROUNDING_ULPS * Fraction(float(np.spacing(biggest))) / Fraction(dt)

_CLOCK_MODELS: dict[float, Model] = {}


def _clock_model(dt: float) -> Model:
    if dt not in _CLOCK_MODELS:
        _CLOCK_MODELS[dt] = Model.build(_clock_graph(dt), f"clock-{dt:g}")
    return _CLOCK_MODELS[dt]


class ClockRun:
    """One importer over the four paths."""

    def __init__(self, name: str, wrapper) -> None:
        self.dt, self.start, self.clock, self.tried = CLOCKS[name]
        self.paths = FourPaths(_clock_model(self.dt), wrapper)
        self.k, self.done, self.t_own = 0, 0, self.start
        #: Every reply to step ``k``, in the order they came (a step taken
        #: again after a rollback is answered again).
        self.replies: dict[int, list] = {}

    def begin(self) -> None:
        self.paths.assert_agree("at start")
        op_initialize(self.paths, self.start)
        _mirror(self.paths, ("initialize", self.start), True)
        self.paths.assert_agree("after initialize")

    def step(self) -> bool:
        paths = self.paths
        t, h = self.clock(self.k, paths.side_time, self.t_own)
        reply = paths._wire({"op": "step", "t": t, "dt": h})          # noqa: SLF001
        note(f"clocked step {self.k}: t={t!r} h={h!r} -> {reply}")
        self.replies.setdefault(self.k, []).append(reply)
        if paths.cw is not None:
            # the same doubles through fmi3DoStep: the same verdict
            assert paths.cw.step(t, h) is reply["ok"], (self.k, reply, paths.cw.w.logs[-2:])
        if reply["ok"]:
            paths.side.step(paths.side_inputs)
            paths.m.direct.step(external_inputs=paths.gm_inputs)
            paths.side_time = paths.gm_time = reply["t"]
            paths.stepped = True
            self.k, self.done, self.t_own = self.k + 1, self.done + 1, t + h
            simulated = Fraction(self.start) + self.done * Fraction(self.dt)
            drift = (Fraction(reply["t"]) - simulated) / Fraction(self.dt)
            assert abs(drift) <= drift_bound(self.dt, self.start, self.done, reply["t"]), (
                self.k, float(drift))
        paths.assert_agree(f"after clocked step {self.k}")           # a refusal advanced nothing
        return reply["ok"]

    def save(self, *, rollback_variable: bool) -> tuple:
        """Save the state on every path; the importer's own clock goes with
        it.  With ``rollback_variable`` the wrapper saves into the one
        variable it reuses, otherwise into a new one."""
        paths = self.paths
        op_get_state(paths)
        if paths.cw is not None:
            if rollback_variable:
                paths.cw.save_for_rollback()
            else:
                paths.cw.get_state()
        paths.assert_agree("after get_state")
        return (len(paths.snapshots) - 1, self.k, self.done, self.t_own, rollback_variable)

    def restore(self, saved: tuple) -> None:
        paths = self.paths
        index, self.k, self.done, self.t_own, rollback_variable = saved
        op_set_state(paths, index)
        if paths.cw is not None:
            cw = paths.cw
            assert cw.roll_back() if rollback_variable else cw.set_state(len(cw.states) - 1), \
                cw.w.logs[-2:]
            cw.time = paths.side_time
        paths.assert_agree("after set_state")

    def close(self) -> None:
        self.paths.close()


def run_clock(name: str, restore: str, wrapper) -> tuple[dict, int]:
    """``name`` under ``restore`` (``"none"``, ``"every step"`` or
    ``"rollback"``): the replies per step index and the steps accepted."""
    run = ClockRun(name, wrapper)
    saved, rolled_back = None, False
    try:
        run.begin()
        while run.k < run.tried:
            if not run.step():
                if restore == "rollback" and saved is not None and not rolled_back:
                    run.restore(saved)
                    rolled_back = True
                    continue
                break
            if restore == "every step":
                run.restore(run.save(rollback_variable=True))
            elif restore == "rollback":
                if saved is None and run.k == max(1, MIDWAY[name]):
                    saved = run.save(rollback_variable=False)
                elif saved is not None and not rolled_back and run.k == saved[1] + 2:
                    run.restore(saved)
                    rolled_back = True
        if restore == "rollback":
            assert rolled_back, "the run never went back"
        return run.replies, run.done
    finally:
        run.close()


#: Steps each clock is accepted for with no restore (``None``: all tried),
#: pinned so that a pattern that stopped being refused -- or started --
#: is seen, and the step a rollback saves at.
CLOCK_ACCEPTED = {"start + k * h": None, "running sum": 47, "long steps": 2, "late points": 1}
MIDWAY = {"start + k * h": 20, "running sum": 24, "long steps": 1, "late points": 1}


@pytest.fixture(scope="module", autouse=True)
def _stop_clock_models():
    yield
    for model in _CLOCK_MODELS.values():
        model.stop()


@pytest.mark.parametrize("name", sorted(CLOCKS))
def test_four_fmu_paths_hold_each_clock_pattern_to_the_drift_bound(name):
    """With no restore: the pattern is accepted for the steps the bound
    allows and refused at the next, naming the simulated time, on the
    bridge and through the compiled wrapper alike, and the four paths agree
    after every step and after the refusal."""
    replies, done = run_clock(name, "none", wrapper_or_none())
    accepted = CLOCK_ACCEPTED[name]
    assert done == (CLOCKS[name][3] if accepted is None else accepted)
    if accepted is not None:
        refusal = replies[accepted][-1]
        assert refusal["ok"] is False
        assert "from the time the FMU has simulated" in refusal["error"], refusal


@pytest.mark.parametrize("restore", ["every step", "rollback"])
@pytest.mark.parametrize("name", sorted(CLOCKS))
def test_a_restore_between_steps_changes_no_reply_on_any_fmu_path(name, restore):
    """Each clock pattern with a save and restore after every step, and
    with one rollback part-way, over the four paths: every step is answered
    as it is without a restore, to the bit -- the reported time, and the
    refusal at the same step in the same words -- and a step taken again
    after going back is answered as it was the first time."""
    wrapper = wrapper_or_none()
    plain, plain_done = run_clock(name, "none", wrapper)
    replies, done = run_clock(name, restore, wrapper)
    assert done == plain_done, (done, plain_done)
    assert sorted(replies) == sorted(plain)
    for k, seen in replies.items():
        for reply in seen:
            assert reply == plain[k][0], (k, reply, plain[k][0])
    if restore == "rollback":
        assert any(len(seen) > 1 for seen in replies.values())


# ===========================================================================
# Every FMI type, both directions, at its extreme values
# ===========================================================================
#
# FMU-024 / FMU-043 / FMU-045: a value is checked in the variable's own
# type, and the wire carries every value as a float64.  A float64 holds
# every value of every FMI type but two: an Int64 or UInt64 above 2**53 in
# magnitude is a float64 only where it is a multiple of the spacing there.
# The paths used to round such a value in silence (B1 round 9, F5): a get
# widened the model's int64 to float64, so fmi3GetInt64 answered fmi3OK with
# a neighbouring integer, and a JSON set of 2**53 + 1 was stored as 2**53.
#
# The battery: one node with an input and an output of every FMI type (and
# the two float parameters a graph can have), under x64 so the 64-bit types
# exist.  For every type, each of its extreme values goes in through the
# typed setter -- the TCP bridge's JSON set, and fmi3Set<Type> of the
# compiled wrapper against a bridge of its own, over binary frames and over
# JSON -- and comes back through the typed getter, as an input and, a step
# later, as an output; the sidecar and the graph are given the value as an
# array of the type itself, never through a float64.  The oracle, with no
# tolerance:
#
# * a value the wire can carry reads back as itself, to the bit, on every
#   path and through every getter;
# * a value the type holds but a float64 does not is *refused* by the set
#   (bridge and wrapper, nothing written) -- and, once the model holds it,
#   put there through the FMU-state archive, which carries the type's own
#   bytes, by every get of it (bridge and wrapper, nothing read), while the
#   four paths still hold exactly that value;
# * a value the type cannot hold is refused by every door.

import contextlib                                                # noqa: E402

import jax                                                       # noqa: E402

from maddening.core.compliance.metadata import StabilityLevel   # noqa: E402
from maddening.core.compliance.stability import stability       # noqa: E402
from maddening.core.node import BoundaryInputSpec, SimulationNode  # noqa: E402

TYPED = tuple(_FMI_C_TYPES)
FLOAT_PARAMS = ("float32", "float64")


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@stability(StabilityLevel.STABLE)
class _Latch(SimulationNode):
    """``out_<type> <- in_<type>`` for every FMI type, and ``held_<type> <-
    gain_<type>`` for the two float types a graph parameter can have: a
    value set on an input (or a parameter) is an output a step later, in
    its own type all the way."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep,
                         **{f"gain_{t}": jnp.asarray(1.0, t) for t in FLOAT_PARAMS})

    def initial_state(self):
        return {**{f"out_{t}": jnp.zeros((), t) for t in TYPED},
                **{f"held_{t}": jnp.zeros((), t) for t in FLOAT_PARAMS}}

    def boundary_input_spec(self):
        return {f"in_{t}": BoundaryInputSpec(shape=(), dtype=np.dtype(t),
                                             default=jnp.zeros((), t)) for t in TYPED}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        out = {f"out_{t}": jnp.asarray(boundary_inputs.get(f"in_{t}", jnp.zeros((), t)), t)
               for t in TYPED}
        out.update({f"held_{t}": jnp.asarray(p[f"gain_{t}"], t) for t in FLOAT_PARAMS})
        return out


def _latch():
    gm = GraphManager()
    gm.add_node(_Latch("latch", 0.01))
    for t in TYPED:
        gm.add_external_input("latch", f"in_{t}", dtype=np.dtype(t))
    gm.compile()
    return gm


def _f32(x: float) -> float:
    return float(np.float32(x))


_F32, _F64 = np.finfo(np.float32), np.finfo(np.float64)

#: Per type: values it holds (its extremes, and the largest integers it and
#: the float64 wire hold exactly), and values it cannot hold.
TYPE_VALUES: dict[str, tuple[list, list]] = {
    "float32": ([float(_F32.max), -float(_F32.max), float(_F32.tiny), -float(_F32.tiny),
                 _f32(1e-45), -_f32(1e-45), -0.0, 16777216.0, 16777217.0, _f32(1 / 3)],
                [3.5e38, -3.5e38, 1e-50, -1e-50]),
    "float64": ([float(_F64.max), -float(_F64.max), float(_F64.tiny), 5e-324, -5e-324, -0.0,
                 2.0 ** 53, 2.0 ** 53 + 2.0, 1 / 3, 2 ** 53 + 1],
                []),
    "bool": ([True, False], [2, -1, 0.5]),
    "int64": ([0, 1, -1, 2 ** 53, -(2 ** 53), 2 ** 53 + 1, -(2 ** 53) - 1, 2 ** 53 + 2,
               2 ** 62, -(2 ** 63), 2 ** 63 - 1024, 2 ** 63 - 1, -(2 ** 63) + 1],
              [2 ** 63, -(2 ** 63) - 1025, 2 ** 64, 0.5]),
    "uint64": ([0, 1, 2 ** 53, 2 ** 53 + 1, 2 ** 53 + 2, 2 ** 63, 2 ** 63 + 1,
                2 ** 64 - 2048, 2 ** 64 - 1],
               [2 ** 64, -1, 0.5]),
}
for _bits in (8, 16, 32):
    TYPE_VALUES[f"int{_bits}"] = (
        [0, 1, -1, 2 ** (_bits - 1) - 1, -(2 ** (_bits - 1))],
        [2 ** (_bits - 1), -(2 ** (_bits - 1)) - 1, 0.5, 2 ** 53 + 1])
    TYPE_VALUES[f"uint{_bits}"] = (
        [0, 1, 2 ** _bits - 1], [2 ** _bits, -1, 0.5, 2 ** 53 + 1])


def _exact(dtype: str, value) -> np.ndarray:
    """``value`` as a scalar of ``dtype``, built in the type itself."""
    return np.asarray(value, dtype=np.dtype(dtype))


def wire_carries(dtype: str, value) -> bool:
    """Is ``value``, held in ``dtype``, a float64?  Decided here in Python
    integers -- an independent statement of the rule, not the bridge's."""
    if np.dtype(dtype).kind not in "iu":
        return True
    return int(float(int(value))) == int(value)


class TypedPaths(FourPaths):
    """The four paths over the typed latch, compared in each type's own
    bytes (the generic comparison reads every variable as a float64, which
    is what this battery is about)."""

    def var(self, name: str):
        return next(v for v in self.m.md.variables if v.name == name)

    def states(self) -> dict[str, dict]:
        assert self.cw is not None and self.cw.bridge is not None
        return {"bridge": self.bridge._sidecar.state,                   # noqa: SLF001
                "sidecar": self.side.state,
                "graph": self.m.direct._state,                          # noqa: SLF001
                "wrapper": self.cw.bridge._sidecar.state}               # noqa: SLF001

    def assert_states_identical(self, what: str) -> None:
        trees = {k: {n: dict(f) for n, f in v.items()} for k, v in self.states().items()}
        for other in ("sidecar", "graph", "wrapper"):
            assert_trees_identical(trees["bridge"], trees[other],
                                   what=f"{what}: bridge vs {other} state")
        assert self.cw is not None and self.cw.bridge is not None
        for mine in (self.bridge._inputs, self.cw.bridge._inputs):      # noqa: SLF001
            assert_trees_identical({n: dict(f) for n, f in mine.items()},
                                   {n: dict(f) for n, f in self.side_inputs.items()},
                                   what=f"{what}: a bridge's pending inputs vs the oracle's")

    # -- typed access --------------------------------------------------------
    def tcp_set(self, var, value) -> dict:
        wire = (1 if value else 0) if isinstance(value, bool) else value
        return self._wire({"op": "set", "type": _FMI_TYPE[var.dtype],
                           "vr": [var.value_reference], "values": [wire]})

    def tcp_get(self, var) -> dict:
        return self._wire({"op": "get", "type": _FMI_TYPE[var.dtype],
                           "vr": [var.value_reference]})

    def c_set(self, var, value) -> int:
        assert self.cw is not None
        return self.cw.w.call("Set", _FMI_TYPE[var.dtype], self.cw.inst,
                              [var.value_reference], [value])[0]

    def c_get(self, var, sentinel=1):
        """``(status, the C value)``; the output slot starts at a sentinel,
        which a refused get must leave."""
        assert self.cw is not None
        status, got = self.cw.w.call("Get", _FMI_TYPE[var.dtype], self.cw.inst,
                                     [var.value_reference], [sentinel])
        return status, got[0]

    def reads_back(self, var, value, what: str) -> None:
        """Every getter answers ``value``, exactly, in ``var``'s type."""
        want = _exact(var.dtype, value)
        reply = self.tcp_get(var)
        assert reply["ok"], (what, reply)
        got = reply["values"][0]
        status, c_value = self.c_get(var)
        assert status == WrapperPath.OK, (what, self.cw.w.logs[-2:])
        if np.dtype(var.dtype).kind in "iu":
            assert float(got).is_integer() and int(got) == int(want), (what, got, int(want))
            assert int(c_value) == int(want), (what, c_value, int(want))
        elif np.dtype(var.dtype).kind == "b":
            assert got in (0.0, 1.0) and bool(got) is bool(want) and bool(c_value) is bool(want)
        else:
            assert np.float64(got).tobytes() == np.float64(want).tobytes(), (what, got, want)
            assert _exact(var.dtype, c_value).tobytes() == want.tobytes(), (what, c_value, want)

    def get_is_refused(self, var, value, what: str) -> None:
        """Neither getter answers for a value the wire cannot carry: an
        error naming the variable and the integer, and nothing read."""
        reply = self.tcp_get(var)
        assert reply["ok"] is False, (what, reply)
        assert f"variable {var.name!r} holds {int(value)}" in reply["error"], (what, reply)
        assert "nothing was read" in reply["error"]
        status, c_value = self.c_get(var, sentinel=7)
        assert status != WrapperPath.OK and c_value == 7, (what, status, c_value)
        assert f"holds {int(value)}" in self.cw.w.logs[-1], self.cw.w.logs[-1]

    def step_all(self) -> None:
        op_step(self, 1)
        assert self.cw is not None
        assert self.cw.step(self.cw.time, self.m.dt), self.cw.w.logs[-2:]

    def install_input_through_the_archive(self, var, value) -> dict:
        """Put ``value`` into ``var`` (an input) through the FMU-state door,
        whose archive carries the type's own bytes: the TCP bridge's
        set_state, and fmi3DeserializeFMUState / fmi3SetFMUState of the
        wrapper.  Returns the TCP reply; the wrapper's verdict must match."""
        node, field = var.node_field()

        def edited(blob: bytes) -> bytes:
            return _edit_archive(blob, lambda members, _fmt: members.__setitem__(
                f"i/{node}/{field}", np.asarray(value)))

        blob = state_of(self._wire({"op": "get_state"}))
        reply = self._wire({"op": "set_state",
                            "state": base64.b64encode(edited(blob)).decode("ascii")})
        cw = self.cw
        assert cw is not None
        cw.get_state()
        handle = cw.states.pop()
        size = ctypes.c_size_t()
        assert cw.w.lib.fmi3SerializedFMUStateSize(cw.inst, handle, ctypes.byref(size)) == cw.OK
        buf = (ctypes.c_char * size.value)()
        assert cw.w.lib.fmi3SerializeFMUState(cw.inst, handle, buf, size.value) == cw.OK
        cw.w.lib.fmi3FreeFMUState(cw.inst, ctypes.byref(handle))
        raw = bytes(buf)
        if raw.startswith(b"PK"):                  # binary frames: the npz itself
            forged = edited(raw)
        else:                                      # JSON: the npz as base64 text
            forged = base64.b64encode(edited(base64.b64decode(raw)))
        assert cw.set_blob(forged) is reply["ok"], (reply, cw.w.logs[-2:])
        return reply


_TYPED_MODEL: list = []


def _typed_model() -> Model:
    """The latch, built once (under x64, as every use of it is)."""
    if not _TYPED_MODEL:
        _TYPED_MODEL.append(Model.build(_latch, "typed-latch"))
    return _TYPED_MODEL[0]


@pytest.fixture(scope="module", autouse=True)
def _stop_typed_model():
    yield
    for model in _TYPED_MODEL:
        model.stop()


def _json_only(bridge: FmuTcpBridge) -> None:
    """Make ``bridge`` answer hello as a JSON-only bridge does, so the
    wrapper carries every value as ``%.17g`` text and reads replies with
    ``strtod``."""
    dispatch = bridge._dispatch                                         # noqa: SLF001

    def answer(req):
        reply = dispatch(req)
        if req.get("op") == "hello" and reply.get("ok"):
            reply = {**reply, "binary": False}
        return reply

    bridge._dispatch = answer                                           # type: ignore[method-assign]  # noqa: SLF001


@contextlib.contextmanager
def typed_paths(wire: str):
    """The four paths over the latch, in Step Mode, the wrapper's bridge
    speaking ``wire`` (``"binary"`` or ``"json"``)."""
    wrapper = wrapper_or_none()
    if wrapper is None:
        pytest.skip("no C compiler: the typed battery needs the compiled wrapper as its "
                    "fourth path")
    with _x64():
        paths = TypedPaths(_typed_model(), wrapper)
        try:
            cw = paths.cw
            assert cw is not None
            if wire == "json":
                cw.close()
                cw.bridge = FmuTcpBridge(paths.m.sidecar(resolver=False), paths.m.md,
                                         master_dt=paths.m.dt)
                _json_only(cw.bridge)
                cw.bridge.start()
                cw.inst = cw._instantiate()                             # noqa: SLF001
                cw.time = 0.0
            cw._into_step_mode()                                        # noqa: SLF001
            # the wire form asked for is the one in use: one get through the
            # wrapper is answered in a binary frame, or is not
            assert cw.bridge.binary_frames_served == 0
            status, _ = paths.c_get(paths.var("time"), sentinel=0.0)
            assert status == cw.OK, cw.w.logs[-2:]
            assert cw.bridge.binary_frames_served == (1 if wire == "binary" else 0)
            yield paths
        finally:
            paths.close()


@pytest.mark.parametrize("wire", ["binary", "json"])
@pytest.mark.parametrize("dtype", TYPED)
def test_every_fmi_type_carries_its_extreme_values_or_refuses_them_on_all_four_paths(dtype, wire):
    """Set and get, as an input and as an output, each value the type
    holds: exact on every path where a float64 is that value, refused --
    set and get, bridge and wrapper -- where it is not (an Int64 / UInt64
    above 2**53), with the four paths still holding the model's own value."""
    holds, _ = TYPE_VALUES[dtype]
    with typed_paths(wire) as paths:
        inp, out = paths.var(f"latch.in_{dtype}"), paths.var(f"latch.out_{dtype}")
        node, field = inp.node_field()
        refused_some = False
        for value in holds:
            what = f"{dtype} {value!r} over {wire}"
            want = jnp.asarray(_exact(dtype, value))
            if wire_carries(dtype, value):
                assert paths.tcp_set(inp, value) == {"ok": True}, what
                assert paths.c_set(inp, _exact(dtype, value).item()) == WrapperPath.OK, (
                    what, paths.cw.w.logs[-2:])
                paths.side_inputs[node][field] = paths.gm_inputs[node][field] = want
                paths.assert_states_identical(f"{what}: after set")
                paths.reads_back(inp, value, f"{what}: the input")
                paths.step_all()
                paths.assert_states_identical(f"{what}: after the step")
                held = np.asarray(paths.side.state["latch"][f"out_{dtype}"])
                assert held.dtype == np.dtype(dtype) and held.tobytes() == np.asarray(want).tobytes()
                paths.reads_back(out, value, f"{what}: the output")
                continue
            # The type holds it and a float64 does not: no set stores a neighbour ...
            refused_some = True
            before = paths.tcp_get(inp)
            reply = paths.tcp_set(inp, value)
            assert reply["ok"] is False and "cannot be carried exactly" in reply["error"], (what, reply)
            assert f"the integer {int(value)}" in reply["error"], reply
            served = paths.cw.bridge.requests_served
            assert paths.c_set(inp, value) != WrapperPath.OK, what
            assert "cannot be carried exactly" in paths.cw.w.logs[-1]
            assert paths.cw.bridge.requests_served == served          # refused before sending
            assert paths.tcp_get(inp) == before
            paths.assert_states_identical(f"{what}: after the refused set")
            # ... and, once the model holds it, no get answers with one.
            assert paths.install_input_through_the_archive(inp, _exact(dtype, value))["ok"], what
            paths.side_inputs[node][field] = paths.gm_inputs[node][field] = want
            paths.assert_states_identical(f"{what}: installed through the archive")
            paths.get_is_refused(inp, value, f"{what}: the input")
            paths.step_all()
            paths.assert_states_identical(f"{what}: after the step")
            held = np.asarray(paths.bridge._sidecar.state["latch"][f"out_{dtype}"])  # noqa: SLF001
            assert held.dtype == np.dtype(dtype) and int(held) == int(value), (what, held)
            paths.get_is_refused(out, value, f"{what}: the output")
            # the refusals broke nothing: another variable still reads
            paths.reads_back(paths.var("latch.out_float32"), 0.0, f"{what}: a neighbour")
        assert refused_some is (dtype in ("int64", "uint64"))


@pytest.mark.parametrize("dtype", TYPED)
def test_a_value_an_fmi_type_cannot_hold_is_refused_by_every_door(dtype):
    """A value outside the type: the bridge's set refuses it and writes
    nothing, and so does the FMU-state archive, in whichever dtype the
    archive carries it."""
    _, outside = TYPE_VALUES[dtype]
    with typed_paths("binary") as paths:
        inp = paths.var(f"latch.in_{dtype}")
        before = paths.tcp_get(inp)
        for value in outside:
            what = f"{dtype} {value!r}"
            reply = paths.tcp_set(inp, value)
            assert reply["ok"] is False, (what, reply)
            assert paths.tcp_get(inp) == before, what
            for carrier in (np.float64, np.int64, np.uint64):
                try:
                    member = np.asarray(value, dtype=carrier)
                except (OverflowError, ValueError):
                    continue                       # this carrier cannot spell the value
                if member.item() != value:
                    continue
                assert paths.install_input_through_the_archive(inp, member)["ok"] is False, (
                    what, carrier)
                assert paths.tcp_get(inp) == before, what
            paths.assert_states_identical(f"{what}: after the refusals")


@pytest.mark.parametrize("dtype", FLOAT_PARAMS)
def test_a_float_parameter_carries_its_extreme_values_on_all_four_paths(dtype):
    """The parameter direction, for the two types a graph parameter has:
    set through both typed setters, read back through both getters, and
    the output a step later is the parameter on every path."""
    holds, outside = TYPE_VALUES[dtype]
    with typed_paths("binary") as paths:
        par, out = paths.var(f"latch.params.gain_{dtype}"), paths.var(f"latch.held_{dtype}")
        for value in holds:
            what = f"parameter {dtype} {value!r}"
            assert paths.tcp_set(par, value) == {"ok": True}, what
            assert paths.c_set(par, _exact(dtype, value).item()) == WrapperPath.OK, what
            leaf = jnp.asarray(_exact(dtype, value))
            paths.side.set_params({par.name: leaf})
            paths.m.direct.params["nodes"]["latch"][f"gain_{dtype}"] = leaf
            paths.reads_back(par, value, f"{what}: the parameter")
            paths.step_all()
            paths.assert_states_identical(f"{what}: after the step")
            paths.reads_back(out, value, f"{what}: the output")
        for value in outside:
            reply = paths.tcp_set(par, value)
            assert reply["ok"] is False and "does not fit its type" in reply["error"], (value, reply)
