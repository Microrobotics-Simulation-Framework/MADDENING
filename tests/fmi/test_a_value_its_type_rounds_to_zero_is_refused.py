"""A value the variable's type cannot hold is refused at either end of its range.

The FMU's value check refused a finite value its type overflows -- a
float32 set to ``1e308`` would read back as ``inf`` -- but stored one its
type underflows: ``1e-50`` read back as ``0.0``, the value lost as entirely,
its sign with it.  It is now refused in the same words ("does not fit its
type"), on every write path the check guards: ``set``, ``set_state`` (an
FMU-state archive) and :meth:`FmuSidecar.set_params`; REST's twin of the
check (``_unrepresentable``) refuses it too.  A value that rounds to a
subnormal keeps its sign and magnitude, so it is accepted -- and whether it
is inside a parameter's bounds is then ``ParamSpec.check``'s question, which
compares it exactly (``tests/core/test_bounds_checks_compare_exactly.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import base64
import io

import numpy as np
import pytest

from maddening.api.server import _unrepresentable
from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, checked_value, state_of, values_of
from maddening.nodes.spring import SpringDamperNode


@pytest.mark.parametrize("value", [1e-50, -1e-50, 1e-46])
def test_a_non_zero_value_the_cast_flushes_to_zero_is_refused(value):
    with pytest.raises(ValueError, match="does not fit its type float32"):
        checked_value(np.asarray([1.0, value]), np.dtype(np.float32), what="v")
    assert _unrepresentable([1.0, value], np.float32) == "value does not fit its type float32"


@pytest.mark.parametrize("value", [1e-40, -1e-40, 0.0, -0.0])
def test_a_subnormal_keeps_its_sign_and_magnitude_and_zero_is_zero(value):
    cast = checked_value(np.asarray(value), np.float32, what="v")
    assert cast.tobytes() == np.float32(value).tobytes()
    assert _unrepresentable(value, np.float32) is None


def test_a_wider_type_holds_what_float32_cannot():
    assert float(checked_value(np.asarray(1e-50), np.float64, what="v")) == 1e-50


def _bridge():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.5))
    gm.add_external_input("s", "anchor_position")
    gm.compile()
    md = build_model_description(gm, model_name="m")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        input_resolver=gm._resolve_external_inputs))
    vr = {v.name: v.value_reference for v in md.variables}
    return FmuTcpBridge(sidecar, md, master_dt=gm.timestep), sidecar, vr


def test_every_write_path_refuses_a_value_that_would_be_stored_as_zero():
    """``set`` of an input and of a parameter, an FMU-state archive and the
    sidecar's in-process door: each refuses ``1e-50`` with nothing written,
    and each keeps ``1e-40``, which is a float32 (a subnormal)."""
    bridge, sidecar, vr = _bridge()
    try:
        def put(name, value):
            return bridge.handle({"op": "set", "type": "Float32", "vr": [vr[name]],
                                  "values": [value]})

        def get(name):
            return float(values_of(bridge.handle({"op": "get", "type": "Float32",
                                                  "vr": [vr[name]]}))[0])

        for name in ("s.anchor_position", "s.params.rest_length"):
            before = get(name)
            refused = put(name, 1e-50)
            assert not refused["ok"] and "does not fit its type float32" in refused["error"]
            assert get(name) == before
        assert put("s.anchor_position", 1e-40)["ok"]
        assert get("s.anchor_position") == float(np.float32(1e-40))

        with pytest.raises(ValueError, match="does not fit its type float32"):
            sidecar.set_params({"s.params.rest_length": 1e-50})

        blob = state_of(bridge.handle({"op": "get_state"}))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        key = next(k for k in members if k.endswith("/rest_length"))
        members[key] = np.asarray(1e-50, np.float64)       # an archive in float64
        buf = io.BytesIO()
        np.savez(buf, **members)
        refused = bridge.handle({"op": "set_state",
                                 "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        assert not refused["ok"] and "does not fit its type float32" in refused["error"]
        assert get("s.params.rest_length") == 1.0
    finally:
        bridge.stop()


def test_a_parameter_cannot_be_set_below_its_bound_by_a_subnormal():
    """The value check keeps ``-1e-40`` (a subnormal), and the bounds check
    then refuses it: ``damping``'s bound is 0, on every write path."""
    bridge, sidecar, vr = _bridge()
    try:
        reply = bridge.handle({"op": "set", "type": "Float32", "vr": [vr["s.params.damping"]],
                               "values": [-1e-40]})
        assert not reply["ok"] and "below bound 0.0" in reply["error"]
        with pytest.raises(ValueError, match="below bound 0.0"):
            sidecar.set_params({"s.params.damping": -1e-40})
        read = float(values_of(bridge.handle({"op": "get", "type": "Float32",
                                              "vr": [vr["s.params.damping"]]}))[0])
        assert read == 2.0
    finally:
        bridge.stop()
