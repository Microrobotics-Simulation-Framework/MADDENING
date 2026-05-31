"""Tests for the FMU sidecar protocol + reference implementation."""

import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi.directional_derivatives import DirectionalDerivativeKind
from maddening.fmi.fmu_state import FMUState
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig


def _make_sidecar(unknown_fn=None):
    def step(state, ext):
        new = {}
        for k, fields in state.items():
            new[k] = {f: v + ext.get(k, {}).get(f, 0.0)
                      for f, v in fields.items()}
        return new

    initial = {
        "node": {
            "x": jnp.array(1.0, dtype=jnp.float32),
        },
    }
    return FmuSidecar(SidecarConfig(
        schema_token="test-token",
        step_fn=step,
        initial_state=initial,
        unknown_fn=unknown_fn,
    ))


class TestStep:

    def test_step_updates_state(self):
        sc = _make_sidecar()
        sc.step({"node": {"x": 0.5}})
        np.testing.assert_allclose(float(sc.state["node"]["x"]), 1.5)


class TestGetDirectionalDerivative:

    def test_forward_mode(self):
        def f(x):
            return {"y": x["a"] ** 2.0}
        sc = _make_sidecar(unknown_fn=f)
        out = sc.get_directional_derivative(
            DirectionalDerivativeKind.FORWARD,
            x={"a": jnp.array(3.0)},
            v={"a": jnp.array(1.0)},
        )
        # d/da (a^2) = 2a -> at a=3 the directional derivative is 6.
        np.testing.assert_allclose(float(out["y"]), 6.0, atol=1e-5)

    def test_reverse_mode(self):
        def f(x):
            return {"y": x["a"] ** 2.0}
        sc = _make_sidecar(unknown_fn=f)
        out = sc.get_directional_derivative(
            DirectionalDerivativeKind.REVERSE,
            x={"a": jnp.array(3.0)},
            v={"y": jnp.array(1.0)},
        )
        np.testing.assert_allclose(float(out["a"]), 6.0, atol=1e-5)

    def test_get_dd_without_unknown_fn_raises(self):
        sc = _make_sidecar(unknown_fn=None)
        with pytest.raises(RuntimeError, match="unknown_fn"):
            sc.get_directional_derivative(
                DirectionalDerivativeKind.FORWARD,
                x={"a": jnp.array(1.0)},
                v={"a": jnp.array(1.0)},
            )


class TestStateRoundTrip:

    def test_get_set_round_trip(self):
        sc = _make_sidecar()
        sc.step({"node": {"x": 5.0}})
        snapshot = sc.get_fmu_state()
        assert isinstance(snapshot, FMUState)
        assert snapshot.schema_token == "test-token"
        # Step again to mutate state, then restore.
        sc.step({"node": {"x": 100.0}})
        sc.set_fmu_state(snapshot)
        np.testing.assert_allclose(float(sc.state["node"]["x"]), 6.0)


class TestWireProtocol:

    def test_handle_step(self):
        sc = _make_sidecar()
        request = pickle.dumps(("step", {"node": {"x": 2.5}}))
        response = sc.handle(request)
        ok, result = pickle.loads(response)
        assert ok == "ok"
        np.testing.assert_allclose(float(result["node"]["x"]), 3.5)

    def test_handle_get_state(self):
        sc = _make_sidecar()
        response = sc.handle(pickle.dumps(("get_state",)))
        ok, fmu_state = pickle.loads(response)
        assert ok == "ok"
        assert isinstance(fmu_state, FMUState)

    def test_handle_get_dd_forward(self):
        def f(x):
            return {"y": 2.0 * x["a"]}
        sc = _make_sidecar(unknown_fn=f)
        request = pickle.dumps((
            "get_dd",
            DirectionalDerivativeKind.FORWARD,
            {"a": jnp.array(7.0)},
            {"a": jnp.array(1.0)},
        ))
        response = sc.handle(request)
        ok, result = pickle.loads(response)
        assert ok == "ok"
        np.testing.assert_allclose(float(result["y"]), 2.0, atol=1e-5)

    def test_handle_unknown_request_errors(self):
        sc = _make_sidecar()
        response = sc.handle(pickle.dumps(("bogus_kind",)))
        kind, msg = pickle.loads(response)
        assert kind == "err"
        assert "unknown" in msg.lower()

    def test_handle_set_state_round_trip(self):
        sc = _make_sidecar()
        snapshot_resp = sc.handle(pickle.dumps(("get_state",)))
        _, fmu_state = pickle.loads(snapshot_resp)
        sc.step({"node": {"x": 100.0}})
        # Restore.
        sc.handle(pickle.dumps(("set_state", fmu_state)))
        np.testing.assert_allclose(float(sc.state["node"]["x"]), 1.0)
