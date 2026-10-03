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

Known disagreements, pinned as strict xfails:

* **M2** -- ``transform="log"`` with no lower bound advertises no ``min``,
  so a bridge whose sidecar has no specs accepts ``<= 0``.
* **N1** -- the REST route stores a numeric string (``"1.5"``) in a float
  leaf as the number, while every FMU door refuses a string and the
  route's own comment says a string is a 400.

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
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of
from maddening.nodes import SpringDamperNode

from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import no_cloud_launch, note, tmp_dir

NODE, KEY = "spring", "stiffness"
NAME = f"{NODE}.params.{KEY}"
LEAF_DTYPE = np.dtype(np.float32)
TINY = float(np.finfo(np.float32).tiny)
SUBNORMAL_MIN = float(np.nextafter(np.float32(0), np.float32(1)))
#: The leaf's value when every FMU in this module is instantiated.
INITIAL_VALUE = np.float32(30.0)


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(NODE, 0.01, stiffness=float(INITIAL_VALUE), damping=2.0,
                                 rest_length=0.4,
                                 initial_position=0.5))
    gm.add_external_input(NODE, "anchor_position")
    gm.compile()
    return gm


class Doors:
    """One compiled graph and one REST client over it, for the module."""

    def __init__(self, root: str) -> None:
        self.gm = _graph()
        self.initial = jax.tree.map(lambda x: x, self.gm.params)
        self.client = TestClient(
            SimulationServer(node_registry={"SpringDamperNode": SpringDamperNode},
                             graph_manager=self.gm, checkpoint_root=root).create_app(),
            raise_server_exceptions=False)

    def reset(self, spec: ParamSpec) -> None:
        self.gm.params = jax.tree.map(lambda x: x, self.initial)
        self.gm.set_param_spec(NODE, KEY, spec)


@pytest.fixture(scope="module")
def doors(_no_cloud):
    with tmp_dir() as root:
        yield Doors(root)


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------

Outcome = tuple[bool, Optional[np.ndarray], str]      # (accepted, stored, message)


def _accepted(stored) -> Outcome:
    return True, np.asarray(stored), ""


def _refused(message: str) -> Outcome:
    return False, None, message


def door_check_params(gm: GraphManager, value) -> Outcome:
    """``check_params`` of the live tree with the leaf at ``value`` held in
    the leaf's dtype (D1).  A value that is not a number is not asked (D2)."""
    if not is_number(value):
        return _refused("not asked: not a number (D2)")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        held = np.asarray(value, np.float64).astype(LEAF_DTYPE)
    tree = jax.tree.map(lambda x: x, gm.params)
    tree["nodes"][NODE][KEY] = jnp.asarray(held)
    try:
        gm.check_params(tree)
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


def door_sidecar(gm: GraphManager, md, value) -> Outcome:
    side = _sidecar(gm, md, specs=True)
    try:
        side.set_params({NAME: value})
    except (KeyError, ValueError) as exc:
        return _refused(str(exc))
    return _accepted(side.params["nodes"][NODE][KEY])


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


def door_bridge_set(gm: GraphManager, md, value, *, specs: bool) -> Outcome:
    try:
        bridge = _bridge(gm, md, specs=specs)
    except BridgeRefusedToStart as exc:
        return _refused(f"the bridge refused to start: {exc}")
    vr = next(v.value_reference for v in md.variables if v.name == NAME)
    try:
        reply = bridge.handle({"op": "set", "vr": [vr], "values": [value]})
        if not reply["ok"]:
            return _refused(reply["error"])
        return _accepted(_stored_by_bridge(bridge, vr))
    finally:
        bridge.stop()


def _archive_member(value) -> Any:
    """``value`` as an FMU-state archive member: in its own dtype (a Python
    float is float64, as an importer's double), so the bridge's
    representability check sees the value and not a float32 already cast."""
    if isinstance(value, (bool, int, float)):
        return np.asarray(value)
    if isinstance(value, str):
        return np.asarray(value)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    return None                                         # not writable into an archive


def door_bridge_set_state(gm: GraphManager, md, value, *, specs: bool) -> Outcome:
    member = _archive_member(value)
    if member is None:
        # ``None`` or a mapping cannot be an npz member at all: an archive
        # cannot carry it, so this door is refused by construction.
        return _refused("not representable in an FMU-state archive")
    try:
        bridge = _bridge(gm, md, specs=specs)
    except BridgeRefusedToStart as exc:
        return _refused(f"the bridge refused to start: {exc}")
    vr = next(v.value_reference for v in md.variables if v.name == NAME)
    try:
        blob = state_of(bridge.handle({"op": "get_state"}))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members[f"p/nodes/{NODE}/{KEY}"] = member
        buf = io.BytesIO()
        np.savez(buf, **members)
        reply = bridge.handle({"op": "set_state",
                               "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        if not reply["ok"]:
            return _refused(reply["error"])
        return _accepted(_stored_by_bridge(bridge, vr))
    finally:
        bridge.stop()


def door_rest(doors: Doors, value) -> Outcome:
    body = json.dumps({"params": {KEY: value}}, allow_nan=True)
    resp = doors.client.put(f"/graph/params/{NODE}", content=body,
                            headers={"content-type": "application/json"})
    assert resp.status_code < 500, f"{resp.status_code}: {resp.text}"
    if resp.status_code != 200:
        return _refused(f"{resp.status_code}: {resp.json().get('detail')}")
    return _accepted(np.asarray(doors.gm.params["nodes"][NODE][KEY]))


def every_door(doors: Doors, spec: ParamSpec, value) -> dict[str, Outcome]:
    """Each door's answer to writing ``value`` under ``spec``, on a graph
    put back where it started."""
    doors.reset(spec)
    gm = doors.gm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        md = build_model_description(gm, model_name="Acceptance")
    out = {
        "check_params": door_check_params(gm, value),
        "sidecar.set_params": door_sidecar(gm, md, value),
        "bridge.set (specs)": door_bridge_set(gm, md, value, specs=True),
        "bridge.set (no specs)": door_bridge_set(gm, md, value, specs=False),
        "bridge.set_state (specs)": door_bridge_set_state(gm, md, value, specs=True),
        "bridge.set_state (no specs)": door_bridge_set_state(gm, md, value, specs=False),
        "REST PUT": door_rest(doors, value),
    }
    doors.reset(spec)
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


def check_acceptance(doors: Doors, spec: ParamSpec, value) -> str:
    outcomes = every_door(doors, spec, value)
    note(f"spec={spec!r} value={value!r}")
    for door, (ok, stored, message) in outcomes.items():
        note(f"  {door:28} {'accepted ' + repr(stored) if ok else 'refused: ' + message[:140]}")
    compared = dict(outcomes)
    if not is_number(value) or lost_in_the_dtype(value):
        # D2 (not asked) and D1 (asked about the cast; checked on its own).
        check = compared.pop("check_params")
        if lost_in_the_dtype(value):
            with np.errstate(over="ignore", under="ignore"):
                held = np.float64(value).astype(LEAF_DTYPE)
            try:
                spec.check(jnp.asarray(held), name=KEY)
                expected = True
            except ValueError:
                expected = False
            assert check[0] is expected, (
                f"check_params of the cast {held!r} answered {check[0]}, "
                f"ParamSpec.check {expected}")
    if is_number(value) and math.isfinite(value) and not lost_in_the_dtype(value) \
            and np.float32(value) == INITIAL_VALUE:
        # D3: a snapshot restores a parameter at the value the FMU was
        # instantiated with whatever its bounds (``set_fmu_state``).
        for door in ("bridge.set_state (specs)", "bridge.set_state (no specs)"):
            ok, _, message = compared.pop(door)
            assert ok or "refused to start" in message, (door, message)
    verdicts = {door: ok for door, (ok, _, _) in compared.items()}
    assert len(set(verdicts.values())) == 1, (
        f"the doors disagree on {value!r} under {spec!r}: "
        + "; ".join(f"{d}: {'accepted' if ok else 'refused (' + compared[d][2][:100] + ')'}"
                    for d, ok in verdicts.items()))
    accepted = next(iter(verdicts.values()))
    if accepted:
        stored = {door: s for door, (_, s, _) in outcomes.items()}
        reference = stored["sidecar.set_params"]
        for door, s in stored.items():
            assert s.dtype == reference.dtype and s.tobytes() == reference.tobytes(), (
                f"{door} stored {s!r}, the sidecar {reference!r}")
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
    ``log`` spec with no lower bound (M2) and a ``log`` / ``logit`` bound in
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
    "M2: ParamSpec(transform='log') with no lower bound advertises no min, so a bridge "
    "whose sidecar has no specs accepts a value <= 0; pending fix"))
def test_a_log_spec_without_a_lower_bound_is_held_by_every_door(doors, value):
    check_acceptance(doors, ParamSpec(transform="log"), value)


@pytest.mark.xfail(strict=True, reason=(
    "M2: ParamSpec(transform='log') with no lower bound advertises no min, so a bridge "
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


def test_a_number_written_as_text_is_refused_by_the_fmu_doors(doors):
    """The FMU half of N1, which holds today: a numeric string is a wire
    value of the wrong type to the sidecar and the bridge, over both
    sidecars and both write paths."""
    outcomes = every_door(doors, ParamSpec(bounds=(0.0, None)), "1.5")
    for door, (ok, _, message) in outcomes.items():
        if door.startswith(("sidecar", "bridge")):
            assert not ok and ("number" in message or "string" in message), (door, message)


# Per push: tests/property/test_differential_param_acceptance.py::test_every_door_accepts_or_refuses_a_parameter_value_together
@pytest.mark.slow  # four times the per-push draws: 5-10 s on CI
@settings(max_examples=EXAMPLES_CHEAP, derandomize=True)
@given(data=st.data())
def test_every_door_accepts_or_refuses_a_parameter_value_together_broadly(doors, data):
    spec = data.draw(specs(), label="spec")
    check_acceptance(doors, spec, data.draw(values(spec), label="value"))
