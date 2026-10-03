"""Differential oracle: every surface that writes a parameter accepts the same values.

A parameter value reaches a graph through five doors, and each checks it
with its own code:

* ``GraphManager.check_params`` -- the Python API's check of a tree;
* ``FmuSidecar.set_params`` -- an in-process FMU, built with the graph's
  ``param_specs``;
* the TCP bridge's ``set`` -- once over a sidecar built *with* the graph's
  specs, once over one built *without* them, which then holds only the
  ``min`` / ``max`` its model description advertises;
* the bridge's ``set_state`` -- an FMU-state archive whose parameter
  member was replaced by the value, over both sidecars;
* ``PUT /graph/params/{node}`` -- the REST route, in process, with every
  cloud launcher stubbed to raise.

The oracle: for a drawn :class:`~maddening.core.params.ParamSpec` (bounds
``None``, finite, or infinite the unbounded way, ``0``, a float32 subnormal;
every transform) and a drawn value (inside, on and one float32 step past
each bound, the open edge of a ``log`` / ``logit`` coordinate, subnormals,
values that flush to ``+-0``, that overflow, non-finite numbers, integers,
booleans and values of the wrong type), all of them **accept or refuse
together**, and every door that accepts stores the same value, bit for bit.

Documented differences, which the oracle allows and nothing else:

* **D1 -- the leaf's dtype.**  ``check_params`` is handed a tree, and a
  tree holds each leaf in its dtype (``gm.params`` casts a write to it,
  ``_sync_node_param_writes``).  So it judges ``value`` *after* the cast
  to the leaf's dtype.  A finite non-zero value the dtype overflows to an
  infinity or flushes to zero is refused by every other door as "does not
  fit its type" (``maddening.fmi.sidecar._checked_value``,
  ``maddening.api.server._unrepresentable``), while ``check_params`` sees
  the infinity (refused as non-finite) or the zero (judged against the
  bounds like any zero).  For those values the other doors are compared
  with each other and ``check_params`` with ``ParamSpec.check`` of the cast.
* **D2 -- the type of a value.**  A boolean, a string, ``None``, a list or
  a mapping is a wire value.  The FMU doors take numbers only and the REST
  route refuses each of these; ``check_params`` is not asked (a Python
  caller hands JAX what it likes, and ``gm.params`` makes a number of
  ``True`` -- not documented anywhere; recorded in the PR that added this
  harness).
* **D3 -- the instantiation value.**  A snapshot restores a parameter at
  the value the FMU was instantiated with (or holds now) whatever its
  bounds (``FmuSidecar.set_fmu_state``), so ``set_state`` of exactly that
  value is accepted where the bounds would refuse it.
* **D4 -- REST bounds an integer.**  The route's request model refuses a
  JSON integer above ``MAX_NODE_PARAM_INT`` in magnitude (422) whatever the
  leaf it is for, as ``POST /graph/nodes`` does: a node may turn an integer
  into an array dimension.  The other doors take ``2**24 + 1`` and store
  the float32 it rounds to.

Known disagreements, pinned as strict xfails:

* **B1-M2** -- ``transform="log"`` with no lower bound advertises no ``min``,
  so a bridge whose sidecar has no specs accepts ``<= 0``.
* **N1** -- the REST route stores a numeric string (``"1.5"``) in a float
  leaf as the number, while every FMU door refuses a string and the
  route's own comment says a string is a 400.
* **N2** -- a ``log`` / ``logit`` bound in the band where a float32's
  spacing is subnormal (``TINY <= |b| < 2**-102``) is advertised one float
  inside it, a distance XLA flushes to zero, where ``ParamSpec.check``
  refuses; a bridge whose sidecar has no specs takes it.

The restore and construction doors and the wrapper nodes (the end of the
module) add: **B2-H1** -- a checkpoint load (``load_state``,
``POST /checkpoint/load``) restores what PUT refuses: out of bounds,
non-finite, a boolean, a value the constructor refuses; **B2-L10** --
``POST /graph/nodes`` applies no ``ParamSpec`` bounds; **B2-H2** -- a write
to a ``HybridNode`` is lost; **N3** -- ``POST /graph/nodes`` takes a boolean
for a float constant; **N4** -- the FMU doors and ``check_params`` take a
value the node's constructor refuses (a ``HeatNode`` past its Fourier
limit).  The sharded wrappers agree.

Tolerance: none.  Acceptance is a yes or no, and a stored value is compared
bit for bit.

What it cannot see: a value every door mishandles the same way (they share
``ParamSpec.check`` and, for the FMU and REST, the representability test),
and arrays (one scalar leaf is written).
"""

from __future__ import annotations

import base64
import io
import json
import math
import warnings
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.api.server import MAX_NODE_PARAM_INT, SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes import HeatNode, SpringDamperNode

from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    assert_nothing_written,
    graph_snapshot,
    no_cloud_launch,
    note,
    tmp_dir,
)

LEAF_DTYPE = np.dtype(np.float32)
TINY = float(np.finfo(np.float32).tiny)
SUBNORMAL_MIN = float(np.nextafter(np.float32(0), np.float32(1)))
#: The spring's stiffness when every FMU in this module is instantiated.
INITIAL_VALUE = np.float32(30.0)


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def _spring_graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=float(INITIAL_VALUE), damping=2.0,
                                 rest_length=0.4, initial_position=0.5))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


class Subject:
    """One compiled graph, the leaf the doors write (``node``, ``key``), and
    a REST client over the graph with a checkpoint root of its own."""

    def __init__(self, label: str, factory, node: str, key: str, root: str) -> None:
        self.label, self.node, self.key = label, node, key
        self.name = f"{node}.params.{key}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # a lone node is "disconnected"
            self.gm = factory()
        self.initial = jax.tree.map(lambda x: x, self.gm.params)
        self.initial_value = np.asarray(self.initial["nodes"][node][key])
        self.root = root
        self.client = TestClient(
            SimulationServer(node_registry={"SpringDamperNode": SpringDamperNode},
                             graph_manager=self.gm, checkpoint_root=root).create_app(),
            raise_server_exceptions=False)

    def __repr__(self) -> str:
        return f"Subject({self.label})"

    def reset(self, spec: Optional[ParamSpec]) -> None:
        """Back to the compiled state and parameters; ``spec`` declared on the
        leaf (``None``: the node's own declaration is kept)."""
        self.gm.params = jax.tree.map(lambda x: x, self.initial)
        self.gm.reset_state()
        if spec is not None:
            self.gm.set_param_spec(self.node, self.key, spec)

    def leaf(self) -> np.ndarray:
        return np.asarray(self.gm.params["nodes"][self.node][self.key])


@pytest.fixture(scope="module")
def doors(_no_cloud):
    with tmp_dir() as root:
        yield Subject("spring", _spring_graph, "spring", "stiffness", root)


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------

Outcome = tuple[bool, Optional[np.ndarray], str]      # (accepted, stored, message)


def _accepted(stored) -> Outcome:
    return True, np.asarray(stored), ""


def _refused(message: str) -> Outcome:
    return False, None, message


def door_check_params(s: Subject, value) -> Outcome:
    """``check_params`` of the live tree with the leaf at ``value`` held in
    the leaf's dtype (D1).  A value that is not a number is not asked (D2)."""
    if not is_number(value):
        return _refused("not asked: not a number (D2)")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        held = np.asarray(value, np.float64).astype(LEAF_DTYPE)
    tree = jax.tree.map(lambda x: x, s.gm.params)
    tree["nodes"][s.node][s.key] = jnp.asarray(held)
    try:
        s.gm.check_params(tree)
    except ValueError as exc:
        return _refused(str(exc))
    return _accepted(held)


def _sidecar(gm: GraphManager, md, *, specs: bool) -> FmuSidecar:
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token,
        step_fn=gm._compiled_step,                                   # noqa: SLF001
        initial_state={n: dict(f) for n, f in gm._state.items()},    # noqa: SLF001
        params=jax.tree.map(lambda x: x, gm.params),
        param_specs=gm.param_specs() if specs else None,
        fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs,                  # noqa: SLF001
    ))


def door_sidecar(s: Subject, md, value) -> Outcome:
    side = _sidecar(s.gm, md, specs=True)
    try:
        side.set_params({s.name: value})
    except (KeyError, ValueError) as exc:
        return _refused(str(exc))
    return _accepted(side.params["nodes"][s.node][s.key])


class BridgeRefusedToStart(Exception):
    """The bridge would not serve this description at all."""


def _bridge(gm: GraphManager, md, *, specs: bool) -> FmuTcpBridge:
    """A bridge over a fresh sidecar.  One whose description advertises a
    ``min`` above its ``max`` refuses to start (``ValueError``): a ``logit``
    range whose open ends are both within a flushed distance of zero admits
    no value at all, and every other door refuses every value there too, so
    a bridge that will not start is a door refusing."""
    try:
        return FmuTcpBridge(_sidecar(gm, md, specs=specs), md, master_dt=gm.timestep)
    except ValueError as exc:
        raise BridgeRefusedToStart(str(exc)) from exc


def _stored_by_bridge(bridge: FmuTcpBridge, vr: int) -> np.ndarray:
    reply = bridge.handle({"op": "get", "vr": [vr]})
    assert reply["ok"], reply
    return np.asarray(reply["values"][0], np.float64).astype(LEAF_DTYPE)


def door_bridge_set(s: Subject, md, value, *, specs: bool) -> Outcome:
    try:
        bridge = _bridge(s.gm, md, specs=specs)
    except BridgeRefusedToStart as exc:
        return _refused(f"the bridge refused to start: {exc}")
    vr = next(v.value_reference for v in md.variables if v.name == s.name)
    try:
        reply = bridge.handle({"op": "set", "vr": [vr], "values": [value]})
        if not reply["ok"]:
            return _refused(reply["error"])
        return _accepted(_stored_by_bridge(bridge, vr))
    finally:
        bridge.stop()


def _archive_member(value) -> Any:
    """``value`` as an archive member (an FMU state or a checkpoint): in its
    own dtype (a Python float is float64, as an importer's double), so a
    representability check sees the value and not a float32 already cast;
    ``None`` for what no archive can carry (``None``, a mapping)."""
    if isinstance(value, (bool, int, float, str)):
        return np.asarray(value)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    return None


def door_bridge_set_state(s: Subject, md, value, *, specs: bool) -> Outcome:
    member = _archive_member(value)
    if member is None:
        return _refused("not representable in an FMU-state archive")
    try:
        bridge = _bridge(s.gm, md, specs=specs)
    except BridgeRefusedToStart as exc:
        return _refused(f"the bridge refused to start: {exc}")
    vr = next(v.value_reference for v in md.variables if v.name == s.name)
    try:
        blob = state_of(bridge.handle({"op": "get_state"}))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members[f"p/nodes/{s.node}/{s.key}"] = member
        buf = io.BytesIO()
        np.savez(buf, **members)
        reply = bridge.handle({"op": "set_state",
                               "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        if not reply["ok"]:
            return _refused(reply["error"])
        return _accepted(_stored_by_bridge(bridge, vr))
    finally:
        bridge.stop()


def _refusal_changed_nothing(s: Subject, before: dict, door: str, value) -> None:
    assert_nothing_written(s.gm, before, what=f"{door} refused {value!r}")


def door_rest(s: Subject, value) -> Outcome:
    before = graph_snapshot(s.gm)
    body = json.dumps({"params": {s.key: value}}, allow_nan=True)
    resp = s.client.put(f"/graph/params/{s.node}", content=body,
                        headers={"content-type": "application/json"})
    assert resp.status_code < 500, f"{resp.status_code}: {resp.text}"
    if resp.status_code != 200:
        _refusal_changed_nothing(s, before, "PUT /graph/params", value)
        return _refused(f"{resp.status_code}: {resp.json().get('detail')}")
    return _accepted(s.leaf())


def _checkpoint_with(s: Subject, value) -> Optional[str]:
    """A checkpoint of the graph as it stands with the leaf's member
    replaced by ``value``, as a file under the subject's checkpoint root;
    ``None`` for a value no archive can carry."""
    member = _archive_member(value)
    if member is None:
        return None
    base = s.gm.save_state(Path(s.root) / "base.npz")
    with np.load(base, allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    key = f"_params/{s.node}/{s.key}"
    assert key in members, sorted(members)
    members[key] = member
    np.savez(Path(s.root) / "edited.npz", **members)
    return "edited.npz"


def door_load_state(s: Subject, value) -> Outcome:
    name = _checkpoint_with(s, value)
    if name is None:
        return _refused("not representable in a checkpoint")
    before = graph_snapshot(s.gm)
    try:
        s.gm.load_state(Path(s.root) / name)
    except (ValueError, TypeError, KeyError) as exc:
        _refusal_changed_nothing(s, before, "load_state", value)
        return _refused(f"{type(exc).__name__}: {exc}")
    return _accepted(s.leaf())


def door_rest_load(s: Subject, value) -> Outcome:
    name = _checkpoint_with(s, value)
    if name is None:
        return _refused("not representable in a checkpoint")
    before = graph_snapshot(s.gm)
    resp = s.client.post("/checkpoint/load", params={"path": name})
    assert resp.status_code < 500, f"{resp.status_code}: {resp.text}"
    if resp.status_code != 200:
        _refusal_changed_nothing(s, before, "POST /checkpoint/load", value)
        return _refused(f"{resp.status_code}: {resp.json().get('detail')}")
    return _accepted(s.leaf())


def door_post_node(s: Subject, value) -> Outcome:
    """``POST /graph/nodes`` building a node of the subject's class with
    ``key`` at ``value``, into a graph of its own (the class's own specs)."""
    gm = GraphManager()
    with tmp_dir() as root:
        client = TestClient(SimulationServer(node_registry={"SpringDamperNode": SpringDamperNode},
                                             graph_manager=gm, checkpoint_root=root).create_app(),
                            raise_server_exceptions=False)
        body = json.dumps({"type": "SpringDamperNode", "name": "built", "timestep": 0.01,
                           "params": {s.key: value}}, allow_nan=True)
        resp = client.post("/graph/nodes", content=body,
                           headers={"content-type": "application/json"})
    assert resp.status_code < 500, f"{resp.status_code}: {resp.text}"
    if resp.status_code not in (200, 201):
        assert "built" not in gm._nodes, "a refused POST left its node behind"  # noqa: SLF001
        return _refused(f"{resp.status_code}: {resp.json().get('detail')}")
    built = gm.get_node("built")
    # A value the class does not treat as a float constant (a boolean) is
    # not a leaf of its pytree: what the node holds is then what it stores.
    return _accepted(built.params_pytree().get(s.key, np.asarray(built.params[s.key])))


WRITE_DOORS = ("check_params", "sidecar.set_params", "bridge.set (specs)",
               "bridge.set (no specs)", "bridge.set_state (specs)",
               "bridge.set_state (no specs)", "REST PUT")
RESTORE_DOORS = ("load_state", "POST /checkpoint/load")


def every_door(s: Subject, spec: Optional[ParamSpec], value, *, restore: bool = False,
               construct: bool = False) -> dict[str, Outcome]:
    """Each door's answer to writing ``value`` under ``spec``, on a graph
    put back where it started.  The FMU doors are asked only where the
    description exports the leaf (a wrapper class need not be ``STABLE``)."""
    s.reset(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        md = build_model_description(s.gm, model_name="Acceptance")
    out = {"check_params": door_check_params(s, value)}
    if any(v.name == s.name for v in md.variables):
        out.update({
            "sidecar.set_params": door_sidecar(s, md, value),
            "bridge.set (specs)": door_bridge_set(s, md, value, specs=True),
            "bridge.set (no specs)": door_bridge_set(s, md, value, specs=False),
            "bridge.set_state (specs)": door_bridge_set_state(s, md, value, specs=True),
            "bridge.set_state (no specs)": door_bridge_set_state(s, md, value, specs=False),
        })
    out["REST PUT"] = door_rest(s, value)
    s.reset(spec)
    if restore:
        out["load_state"] = door_load_state(s, value)
        s.reset(spec)
        out["POST /checkpoint/load"] = door_rest_load(s, value)
        s.reset(spec)
    if construct:
        out["POST /graph/nodes"] = door_post_node(s, value)
    return out


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def lost_in_the_dtype(value) -> bool:
    """D1: a finite non-zero number the leaf's dtype holds as an infinity or
    a zero."""
    if not is_number(value) or not math.isfinite(value) or value == 0:
        return False
    with np.errstate(over="ignore", under="ignore"):
        held = np.float64(value).astype(LEAF_DTYPE)
    return bool(not np.isfinite(held) or held == 0)


def check_acceptance(s: Subject, spec: Optional[ParamSpec], value, *, restore: bool = False,
                     construct: bool = False, only: Optional[tuple[str, ...]] = None) -> str:
    """Hold every door (or ``only`` these) to the oracle for one value."""
    outcomes = every_door(s, spec, value, restore=restore, construct=construct)
    if only is not None:
        outcomes = {door: o for door, o in outcomes.items() if door in only}
    note(f"{s!r} spec={spec!r} value={value!r}")
    for door, (ok, stored, message) in outcomes.items():
        note(f"  {door:28} {'accepted ' + repr(stored) if ok else 'refused: ' + message[:140]}")
    compared = dict(outcomes)
    if not is_number(value) or lost_in_the_dtype(value):
        # D2 (not asked) and D1 (asked about the cast; checked on its own).
        check = compared.pop("check_params")
        if lost_in_the_dtype(value):
            with np.errstate(over="ignore", under="ignore"):
                held = np.float64(value).astype(LEAF_DTYPE)
            declared = spec if spec is not None else s.gm.param_specs()["nodes"][s.node][s.key]
            try:
                declared.check(jnp.asarray(held), name=s.key)
                expected = True
            except ValueError:
                expected = False
            assert check[0] is expected, (
                f"check_params of the cast {held!r} answered {check[0]}, "
                f"ParamSpec.check {expected}")
    if isinstance(value, int) and not isinstance(value, bool) \
            and abs(value) > MAX_NODE_PARAM_INT:
        # D4: the REST request model bounds every JSON integer, whatever the
        # leaf it is for (``SetNodeParamsRequest``: a node may turn one into
        # an array dimension), so it is a 422 there.
        for door in ("REST PUT", "POST /graph/nodes"):
            if door in compared:
                ok, _, message = compared.pop(door)
                assert not ok and message.startswith("422"), (door, message)
    if is_number(value) and math.isfinite(value) and not lost_in_the_dtype(value) \
            and np.float32(value) == s.initial_value:
        # D3: a snapshot restores a parameter at the value the FMU was
        # instantiated with whatever its bounds (``set_fmu_state``).
        for door in ("bridge.set_state (specs)", "bridge.set_state (no specs)"):
            if door in compared:
                ok, _, message = compared.pop(door)
                assert ok or "refused to start" in message, (door, message)
    verdicts = {door: ok for door, (ok, _, _) in compared.items()}
    assert verdicts, "no door was compared"
    assert len(set(verdicts.values())) == 1, (
        f"the doors disagree on {value!r} under {spec!r} ({s!r}): "
        + "; ".join(f"{d}: {'accepted' if ok else 'refused (' + compared[d][2][:100] + ')'}"
                    for d, ok in verdicts.items()))
    accepted = next(iter(verdicts.values()))
    if accepted:
        # The same resulting model: every door that took the value holds
        # the same leaf, bit for bit.
        stored = {door: st_ for door, (_, st_, _) in compared.items()}
        reference_door = next(d for d in ("sidecar.set_params", "check_params", "REST PUT")
                              if d in stored) if any(
            d in stored for d in ("sidecar.set_params", "check_params", "REST PUT")) else next(
            iter(stored))
        reference = stored[reference_door]
        for door, st_ in stored.items():
            assert st_.dtype == reference.dtype and st_.tobytes() == reference.tobytes(), (
                f"{door} holds {st_!r} after accepting {value!r}, {reference_door} {reference!r}")
    return "accepted" if accepted else "refused"


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------

#: Special bound values: zero of both signs, the smallest normal, float32
#: subnormals, one and a large finite value.
_BOUND_EDGES = (0.0, -0.0, TINY, SUBNORMAL_MIN, 1e-40, 1.0, -1.0, 1e30)
_BOUND = st.one_of(st.sampled_from(_BOUND_EDGES),
                   st.floats(-100.0, 100.0, allow_nan=False, width=32))
#: Below this magnitude a float32's spacing is a subnormal, which XLA's CPU
#: arithmetic flushes to zero: ``nextafter(b) - b`` is 0 to the step.
FLUSHED_SPACING = 2.0 ** -102


def in_the_flushed_band(b) -> bool:
    """N2's domain: a normal float32 bound whose neighbour is a flushed
    distance away (``TINY <= |b| < 2**-102``)."""
    return b is not None and math.isfinite(b) and TINY <= abs(float(np.float32(b))) < FLUSHED_SPACING


def _out_of_the_band(b):
    """A bound moved out of N2's band, as far as a factor of ``2**80``
    takes it (a power of two: the bound's float32 rounding is unchanged)."""
    return b * 2.0 ** 80 if in_the_flushed_band(b) else b


_STRICT_BOUND = _BOUND.map(_out_of_the_band)
_BAND_BOUND = st.one_of(st.sampled_from([TINY, -TINY, 2 * TINY, 1e-35, -1e-35, 1e-32]),
                        st.floats(TINY, FLUSHED_SPACING / 2, width=32))


@st.composite
def specs(draw, *, domain: str = "main") -> ParamSpec:
    """A valid ``ParamSpec`` of every shape.

    ``domain="main"`` leaves out the two known disagreements' domains: a
    ``log`` spec with no lower bound (B1-M2) and a ``log`` / ``logit`` bound in
    the flushed band (N2).  ``"M2"`` and ``"N2"`` draw only theirs."""
    if domain == "M2":
        return ParamSpec(bounds=(None, None), transform="log")
    transform = draw(st.sampled_from(["log", "logit"] if domain == "N2"
                                     else [None, "log", "logit"]))
    strict = _BAND_BOUND if domain == "N2" else _STRICT_BOUND
    if transform == "log":
        return ParamSpec(bounds=(draw(strict), None), transform="log")
    if transform == "logit":
        a = draw(strict)
        b = draw(_STRICT_BOUND)
        lo, hi = sorted((a, b))
        if not lo < hi:
            hi = lo + max(1.0, abs(lo))
        return ParamSpec(bounds=(lo, hi), transform="logit")
    a, b = draw(_BOUND), draw(_BOUND)
    lo, hi = min(a, b), max(a, b)
    if not lo < hi:                                # two finite bounds need lo < hi
        hi = lo + max(1.0, abs(lo))
    lo = draw(st.sampled_from([lo, lo, None, -math.inf]))
    hi = draw(st.sampled_from([hi, hi, None, math.inf]))
    return ParamSpec(bounds=(lo, hi))


def _f32_neighbours(x: float) -> list[float]:
    """``x`` in float32 and one float32 step either side of it."""
    v = np.float32(x)
    return [float(np.nextafter(v, np.float32(-np.inf))), float(v),
            float(np.nextafter(v, np.float32(np.inf)))]


#: Values every spec is asked about: zeros, subnormals (some flush to
#: +-0 in float32, some round to a subnormal), the smallest normal, an
#: overflow, the largest float32, non-finite numbers and integers (one a
#: float32 cannot hold exactly).
_SPECIAL_VALUES = (0.0, -0.0, 1e-40, -1e-40, SUBNORMAL_MIN, -SUBNORMAL_MIN, 1e-50, -1e-50,
                   TINY, -TINY, 1e39, -1e39, float(np.finfo(np.float32).max), math.nan,
                   math.inf, -math.inf, 0, 1, -1, 2 ** 24 + 1, 30.0, 2.0)
#: Wire values that are not numbers (a numeric string is N1's, drawn there).
_NOT_NUMBERS = (True, False, "abc", "", None, [1.0], [], {"v": 1.0})


@st.composite
def values(draw, spec: ParamSpec,
           kinds: tuple[str, ...] = ("near_bound", "inside", "special", "not_a_number")):
    kind = draw(st.sampled_from(kinds))
    if kind == "not_a_number":
        return draw(st.sampled_from(_NOT_NUMBERS))
    if kind == "special":
        return draw(st.sampled_from(_SPECIAL_VALUES))
    finite = [b for b in spec.bounds if b is not None and math.isfinite(b)]
    if spec.transform == "log" and spec.bounds[0] is None:
        finite.append(0.0)                         # measured from 0 (ParamSpec)
    if kind == "near_bound" and finite:
        return draw(st.sampled_from([n for b in finite for n in _f32_neighbours(b)]))
    lo = max([b for b in finite if b <= 30.0] or [-100.0])
    hi = min([b for b in finite if b > lo] or [lo + 100.0])
    u = draw(st.floats(0.0, 1.0))
    return float(np.float32(lo + u * (hi - lo)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# EXAMPLES_STANDARD: an example builds a description, four bridges and a
# sidecar and makes one REST request over a graph compiled once (about 25 ms).
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_every_door_accepts_or_refuses_a_parameter_value_together(doors, data):
    spec = data.draw(specs(), label="spec")
    value = data.draw(values(spec), label="value")
    event(f"transform={spec.transform}")
    event(check_acceptance(doors, spec, value))


@pytest.mark.parametrize("value", [SUBNORMAL_MIN, 2 * SUBNORMAL_MIN, TINY, -0.0, 1e-50])
def test_every_door_agrees_at_a_subnormal_lower_bound_of_a_log_coordinate(doors, value):
    """A float32 subnormal lower bound under ``log``: ``ParamSpec.check``
    flushes it (and the value) as the step's arithmetic does, while the
    advertised ``min`` is rounded up to the smallest normal.  The two rules
    must draw the line in the same place."""
    check_acceptance(doors, ParamSpec(bounds=(1e-40, None), transform="log"), value)


@pytest.mark.parametrize("value", [-1.0, 0.0, -0.0, -1e-40, 1e-40])
@pytest.mark.xfail(strict=True, reason=(
    "B1-M2: ParamSpec(transform='log') with no lower bound advertises no min, so a bridge "
    "whose sidecar has no specs accepts a value <= 0; pending fix"))
def test_a_log_spec_without_a_lower_bound_is_held_by_every_door(doors, value):
    check_acceptance(doors, ParamSpec(transform="log"), value)


@pytest.mark.xfail(strict=True, reason=(
    "B1-M2: ParamSpec(transform='log') with no lower bound advertises no min, so a bridge "
    "whose sidecar has no specs accepts a value <= 0; pending fix"))
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_every_door_agrees_on_a_log_spec_without_a_lower_bound(doors, data):
    spec = data.draw(specs(domain="M2"), label="spec")
    check_acceptance(doors, spec, data.draw(values(spec), label="value"))


_N2_REASON = (
    "N2: a log/logit bound b with TINY <= |b| < 2**-102 is advertised as nextafter(b), "
    "a flushed distance from b that ParamSpec.check refuses, so a bridge whose sidecar has "
    "no specs accepts it (model_description.py _advertised_bound); pending fix")


@pytest.mark.parametrize("spec, value", [
    (ParamSpec(bounds=(TINY, None), transform="log"), float(np.nextafter(np.float32(TINY), 1))),
    (ParamSpec(bounds=(1e-35, None), transform="log"),
     float(np.nextafter(np.float32(1e-35), np.float32(1)))),
    (ParamSpec(bounds=(1e-35, 1.0), transform="logit"),
     float(np.nextafter(np.float32(1e-35), np.float32(1)))),
    (ParamSpec(bounds=(-1.0, -1e-35), transform="logit"),
     float(np.nextafter(np.float32(-1e-35), np.float32(-1)))),
], ids=["log-at-TINY", "log-at-1e-35", "logit-lower", "logit-upper"])
@pytest.mark.xfail(strict=True, reason=_N2_REASON)
def test_an_open_bound_in_the_flushed_band_is_held_by_every_door(doors, spec, value):
    check_acceptance(doors, spec, value)


@pytest.mark.xfail(strict=True, reason=_N2_REASON)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_every_door_agrees_on_an_open_bound_in_the_flushed_band(doors, data):
    spec = data.draw(specs(domain="N2"), label="spec")
    # The disagreement is at the bounds, so the values are drawn there.
    check_acceptance(doors, spec, data.draw(values(spec, kinds=("near_bound",)), label="value"))


@pytest.mark.parametrize("value", [0.0, TINY, -TINY, 1e-40, 0.5])
def test_a_logit_range_no_value_can_enter_is_refused_by_every_door(doors, value):
    """``logit`` on ``(0, 1e-40)``: both ends flush to zero in the step's
    arithmetic, so no value has a coordinate.  Every door refuses every
    value; the bridge refuses to start at all, since the description it
    would serve advertises ``min = TINY`` above ``max = -TINY``."""
    outcomes = every_door(doors, ParamSpec(bounds=(0.0, 1e-40), transform="logit"), value)
    assert not any(ok for ok, _, _ in outcomes.values()), outcomes
    assert "refused to start" in outcomes["bridge.set (no specs)"][2]


@pytest.mark.parametrize("value", ["1.5", "30", " 2.0 ", "1e3"])
@pytest.mark.xfail(strict=True, reason=(
    "N1: PUT /graph/params stores a numeric string in a float leaf as the number "
    "(server.py set_node_params_locked: jnp.asarray(value, dtype) parses it), while every "
    "FMU door refuses a string; pending fix"))
def test_a_numeric_string_is_refused_by_every_door(doors, value):
    check_acceptance(doors, ParamSpec(bounds=(0.0, None)), value)


def test_a_large_integer_is_refused_by_the_rest_route_alone(doors):
    """D4, per push (the broad draw found it): ``2**24 + 1`` is stored as
    the float32 it rounds to by every door but the REST route, whose
    request model bounds a JSON integer whatever the leaf."""
    assert check_acceptance(doors, ParamSpec(bounds=(0.0, None)), 2 ** 24 + 1) == "accepted"


def test_a_number_written_as_text_is_refused_by_the_fmu_doors(doors):
    """The FMU half of N1, which holds today: a numeric string is a wire
    value of the wrong type to the sidecar and the bridge, over both
    sidecars and both write paths."""
    outcomes = every_door(doors, ParamSpec(bounds=(0.0, None)), "1.5")
    for door, (ok, _, message) in outcomes.items():
        if door.startswith(("sidecar", "bridge")):
            assert not ok and ("number" in message or "string" in message), (door, message)


# ===========================================================================
# Restore doors, construction, and wrapper nodes
# ===========================================================================
#
# Every write *and restore* path either accepts a value with the same
# resulting model or refuses it with nothing changed: ``GraphManager.load_state``
# and ``POST /checkpoint/load`` of a checkpoint carrying the value (the
# checkpoint is the graph's own, with the leaf's member replaced), and
# ``POST /graph/nodes`` building a node of the class with the value (against
# the class's own specs).  A refusal by a door on the graph -- PUT, either
# load -- must leave every node's params, every ``gm.params`` leaf, the whole
# state and the dirty flag where they were.

_B2_H1 = ("B2-H1: GraphManager.load_state and POST /checkpoint/load restore a parameter "
          "value PUT /graph/params refuses -- outside its ParamSpec bounds, non-finite, a "
          "boolean, a numeric string, or one the node's constructor refuses -- and answer "
          "200; pending fix")
_B2_L10 = ("B2-L10: POST /graph/nodes builds a node with a parameter outside its class's "
           "ParamSpec bounds (it applies none); pending fix")
_B2_H2 = ("B2-H2: a write to a HybridNode is lost: PUT /graph/params answers 200, and the "
          "next read of gm.params syncs the wrapped node's unchanged value over it "
          "(HybridNode copies its physics node's params at construction); pending fix")
_N3 = ("N3: POST /graph/nodes builds a node with a boolean for a float constant, which "
       "then drops out of the params pytree (structural), where PUT refuses a boolean "
       "and every FMU door takes numbers only; pending fix")
_N4 = ("N4: check_params, FmuSidecar.set_params and the bridge's set and set_state accept "
       "a value the node's constructor refuses (a HeatNode thermal_diffusivity past its "
       "Fourier limit, so the rod diverges), which PUT /graph/params refuses; pending fix")


@pytest.mark.parametrize("value", [45.0, 0.5, float(INITIAL_VALUE)])
def test_a_restore_door_takes_what_every_write_door_takes(doors, value):
    """A checkpoint carrying a value every write door takes restores it,
    and the restored graph holds the leaf the write doors store."""
    assert check_acceptance(doors, ParamSpec(bounds=(0.0, None)), value,
                            restore=True) == "accepted"


@pytest.mark.parametrize("value", ["abc", [1.0], None, 1e39, 1e-50])
def test_a_restore_door_refuses_what_no_door_takes_with_nothing_changed(doors, value):
    """A string that is no number, a misshapen member, a value no archive
    holds, and values float32 overflows or flushes: every door refuses, and
    the graph a refused load leaves is the graph before it."""
    assert check_acceptance(doors, ParamSpec(bounds=(0.0, None)), value,
                            restore=True) == "refused"


# (A numeric string is left to N1's test: PUT takes one too, so it could not
# show B2-H1's fix.)
@pytest.mark.parametrize("value", [-1.0, -5.0, math.nan, math.inf, True],
                         ids=["below-bound", "damping-like", "nan", "inf", "bool"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_H1)
def test_a_restore_door_refuses_what_the_write_doors_refuse(doors, value):
    check_acceptance(doors, ParamSpec(bounds=(0.0, None)), value, restore=True)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_H1)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_every_restore_door_agrees_with_the_write_doors(doors, data):
    spec = data.draw(specs(), label="spec")
    check_acceptance(doors, spec, data.draw(values(spec), label="value"), restore=True)


def _rod_graph() -> GraphManager:
    """A rod at a Fourier number of 0.25: ``thermal_diffusivity`` 0.4 puts
    it at 2.0, which the constructor refuses (limit 0.5)."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.05, n_cells=10, length=1.0, thermal_diffusivity=0.05))
    gm.add_external_input("rod", "heat_source", shape=(10,))
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def rod(_no_cloud):
    with tmp_dir() as root:
        yield Subject("rod", _rod_graph, "rod", "thermal_diffusivity", root)


def test_a_stable_diffusivity_is_taken_by_every_door(rod):
    """The rod's own neighbourhood: a diffusivity inside the stability limit
    is accepted and held alike by every write and restore door."""
    assert check_acceptance(rod, None, 0.06, restore=True) == "accepted"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_H1)
def test_a_restore_door_refuses_a_diffusivity_the_constructor_refuses(rod):
    """B2-H1's second case: the loads against PUT alone (the FMU doors'
    answer to the same value is N4's)."""
    check_acceptance(rod, None, 0.4, restore=True,
                     only=("REST PUT", "load_state", "POST /checkpoint/load"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_N4)
def test_every_write_door_refuses_a_diffusivity_the_constructor_refuses(rod):
    check_acceptance(rod, None, 0.4)


# -- construction ---------------------------------------------------------------

@pytest.fixture(scope="module")
def own_specs(_no_cloud):
    """The spring with its class's own declaration on ``damping``
    (``bounds=(0, None)``): what ``POST /graph/nodes`` builds against."""
    with tmp_dir() as root:
        yield Subject("spring, own specs", _spring_graph, "spring", "damping", root)


@pytest.mark.parametrize("value", [1.5, 0.0, 2.0])
def test_a_node_built_with_a_value_every_door_takes_holds_it(own_specs, value):
    assert check_acceptance(own_specs, None, value, construct=True) == "accepted"


@pytest.mark.parametrize("value", [math.nan, "abc", [1.0, 2.0]])
def test_a_node_is_not_built_with_a_value_no_door_takes(own_specs, value):
    assert check_acceptance(own_specs, None, value, construct=True) == "refused"


@pytest.mark.parametrize("value", [-5.0, -1e-30])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_L10)
def test_a_node_is_not_built_with_a_value_outside_its_bounds(own_specs, value):
    check_acceptance(own_specs, None, value, construct=True)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_N3)
def test_a_node_is_not_built_with_a_boolean_for_a_float_constant(own_specs):
    check_acceptance(own_specs, None, True, construct=True)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_L10)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_every_construction_door_agrees_with_the_write_doors(own_specs, data):
    spec = own_specs.gm.param_specs()["nodes"]["spring"]["damping"]
    value = data.draw(values(spec, kinds=("near_bound", "inside", "special")), label="value")
    check_acceptance(own_specs, None, value, construct=True)


# -- wrapper and composite nodes ----------------------------------------------------

def _zero_correction(state, boundary_inputs, dt):
    return {}


def _hybrid_graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(HybridNode(SpringDamperNode("spring", 0.01, stiffness=float(INITIAL_VALUE),
                                            initial_position=0.5), _zero_correction))
    gm.compile()
    return gm


def _one_device_mesh():
    from maddening.cloud.multigpu.device_mesh import create_device_mesh

    return create_device_mesh(shape=(1,))


def _sharded_stencil_graph() -> GraphManager:
    """The sharding harness's generated stencil node (``decay`` declared
    ``bounds=(0, None)``) inside ``ShardedStencilNode`` on one device."""
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
    from tests.cloud.multigpu import differential_sharding_support as D

    cfg = D.StencilConfig(
        mesh_shape=(1,), axis_names=("devices",), axis_map=(("devices", 0),), shape=(4,),
        halo=(1,), fill="periodic", declares=True, contract="params", integral=None,
        integral_name="a_total", integral_listed=False, reads_shard_info=False, kappa=None,
        kappa_axis=0, table=None, source="none", misshapen_shape=(), gain=False, faces=False,
        dtype="float32", wrapping="single", steps=1, seed=0, surface="run_scan")
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(D.make_stencil_node(cfg), _one_device_mesh(),
                                   {"devices": 0}, boundary="periodic"))
    gm.compile()
    return gm


def _sharded_unstructured_graph() -> GraphManager:
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from tests.cloud.multigpu import differential_sharding_support as D

    cfg = D.UnstructuredConfig(
        n_devices=1, n_cells=5, assignment=(0,) * 5, partition="uneven", chords=(),
        contract="params", integral=None, integral_name="a_total", integral_listed=False,
        weight=None, source="none", misshapen_len=0, gain=False, dtype="float32",
        wrapping="single", steps=1, seed=2, surface="run_scan")
    layout = D.unstructured_layout(cfg)
    gm = GraphManager()
    gm.add_node(ShardedUnstructuredNode(D.unstructured_node_class("params")(cfg, layout),
                                        _one_device_mesh(), layout))
    gm.compile()
    return gm


def _sharded_rod_graph() -> GraphManager:
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    gm = GraphManager()
    gm.add_node(ShardedStencilNode(HeatNode("rod", 0.01, n_cells=8, thermal_diffusivity=0.01),
                                   _one_device_mesh(), {"devices": 0}))
    gm.compile()
    return gm


WRAPPERS = {
    "hybrid": (_hybrid_graph, "spring", "stiffness"),
    "sharded-stencil": (_sharded_stencil_graph, "gen", "decay"),
    "sharded-unstructured": (_sharded_unstructured_graph, "gen", "decay"),
    "sharded-heat-rod": (_sharded_rod_graph, "rod", "thermal_diffusivity"),
}
_WRAPPED: dict[str, Subject] = {}


@pytest.fixture(scope="module")
def wrapper_root(_no_cloud):
    with tmp_dir() as root:
        yield root
    _WRAPPED.clear()


def wrapped(label: str, root: str) -> Subject:
    if label not in _WRAPPED:
        factory, node, key = WRAPPERS[label]
        directory = Path(root) / label
        directory.mkdir(parents=True, exist_ok=True)
        _WRAPPED[label] = Subject(label, factory, node, key, str(directory))
    return _WRAPPED[label]


@pytest.mark.parametrize("label, value", [
    ("sharded-stencil", 0.5), ("sharded-unstructured", 0.5), ("sharded-heat-rod", 0.02),
])
def test_a_wrapper_node_holds_a_write_and_a_restore_of_a_value_every_door_takes(
        wrapper_root, label, value):
    assert check_acceptance(wrapped(label, wrapper_root), None, value,
                            restore=True) == "accepted"


@pytest.mark.parametrize("label", sorted(WRAPPERS))
def test_a_wrapper_node_refuses_a_value_outside_its_bounds_with_nothing_changed(
        wrapper_root, label):
    assert check_acceptance(wrapped(label, wrapper_root), None, -1.0) == "refused"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_H2)
def test_a_hybrid_node_holds_a_write_every_door_takes(wrapper_root):
    check_acceptance(wrapped("hybrid", wrapper_root), None, 45.0)


@pytest.mark.parametrize("label", ["sharded-stencil", "sharded-unstructured"])
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_sharded_wrapper_takes_a_parameter_write_as_every_door_does(wrapper_root, label,
                                                                      data):
    s = wrapped(label, wrapper_root)
    spec = data.draw(specs(), label="spec")
    check_acceptance(s, spec, data.draw(values(spec), label="value"))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_B2_H2)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_hybrid_node_takes_a_parameter_write_as_every_door_does(wrapper_root, data):
    s = wrapped("hybrid", wrapper_root)
    spec = data.draw(specs(), label="spec")
    check_acceptance(s, spec, data.draw(values(spec, kinds=("near_bound", "inside")),
                                        label="value"))


# Per push: tests/property/test_differential_param_acceptance.py::test_every_door_accepts_or_refuses_a_parameter_value_together
@pytest.mark.slow  # four times the per-push draws: 5-10 s on CI
@settings(max_examples=EXAMPLES_CHEAP, derandomize=True)
@given(data=st.data())
def test_every_door_accepts_or_refuses_a_parameter_value_together_broadly(doors, data):
    spec = data.draw(specs(), label="spec")
    check_acceptance(doors, spec, data.draw(values(spec), label="value"))
