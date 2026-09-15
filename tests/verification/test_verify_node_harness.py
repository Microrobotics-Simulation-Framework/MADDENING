"""The ``verify_node`` battery must fire on planted faults, not just pass.

Each fault class isolates one detector: a forward NaN, a gradient-only NaN,
a narrow-window NaN (exercises shrinking), a jit/eager divergence, and an
energy gain.  A harness that has never been seen to fail is decoration.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode
from maddening.testing.verification import (
    assert_node_verified,
    verify_node,
)


class _Scalar(SimulationNode):
    def initial_state(self):
        return {"x": jnp.array(1.0, jnp.float32)}


class ForwardNaN(_Scalar):
    def update(self, s, bi, dt):
        return {"x": s["x"] + dt / s["x"]}


class GradientOnlyNaN(_Scalar):
    """``sqrt(x**2)`` is finite everywhere; d/dx is NaN at x == 0."""
    def update(self, s, bi, dt):
        return {"x": jnp.sqrt(s["x"] ** 2)}


class NarrowWindowNaN(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros(4, jnp.float32)}

    def update(self, s, bi, dt):
        bad = jnp.any(jnp.abs(s["x"] - 37.0) < 5.0)
        return {"x": jnp.where(bad, jnp.nan, s["x"])}


class HostBranch(_Scalar):
    """Python ``if`` on an array value: eager and jit disagree."""
    def update(self, s, bi, dt):
        try:
            flip = bool(s["x"] > 0.5)
        except Exception:  # traced under jit -> concretisation error
            flip = False
        return {"x": -s["x"] if flip else s["x"]}


class EnergyGain(_Scalar):
    def update(self, s, bi, dt):
        return {"x": s["x"] * (1.0 + dt)}


KW = dict(max_examples=200, derandomize=True)


def test_forward_nan_caught_and_shrunk():
    r = verify_node(ForwardNaN(name="n", timestep=0.01),
                    bounds={"x": (0.0, 1.0)}, checks=["finite"], **KW)["finite"]
    assert r.failed
    assert float(r.counterexample["state"]["x"]) == 0.0


def test_gradient_only_nan_caught():
    res = verify_node(GradientOnlyNaN(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)},
                      checks=["finite", "gradient_finite"], **KW)
    assert res["finite"].passed
    assert res["gradient_finite"].failed
    assert float(res["gradient_finite"].counterexample["state"]["x"]) == 0.0


def test_narrow_window_found_and_shrunk():
    # Not derandomized: the derandomized generator favours boundary and
    # "simple" values and can miss an interior window entirely.
    r = verify_node(NarrowWindowNaN(name="n", timestep=0.01),
                    bounds={"x": (-100.0, 100.0)}, checks=["finite"],
                    max_examples=500)["finite"]
    assert r.failed
    x = np.asarray(r.counterexample["state"]["x"])
    assert np.any(np.abs(x - 37.0) < 5.0)


def test_jit_eager_divergence_caught():
    r = verify_node(HostBranch(name="n", timestep=0.01),
                    bounds={"x": (0.0, 1.0)}, checks=["jit_consistent"],
                    **KW)["jit_consistent"]
    assert r.failed


def test_energy_gain_caught():
    r = verify_node(EnergyGain(name="n", timestep=0.01),
                    bounds={"x": (0.1, 1.0)}, checks=[],
                    energy_fn=lambda s: s["x"] ** 2, **KW)["energy_monotone"]
    assert r.failed


def test_custom_invariant_and_output_bounds():
    node = EnergyGain(name="n", timestep=0.01)
    res = verify_node(
        node, bounds={"x": (0.0, 1.0)}, checks=[],
        output_bounds={"x": (0.0, 0.5)},
        invariants={"nonneg": lambda s_in, s_out, bi, dt: bool(s_out["x"] >= 0)},
        **KW,
    )
    assert res["boundedness"].failed
    assert res["nonneg"].passed


def test_unknown_check_name_rejected():
    with pytest.raises(ValueError, match="unknown checks"):
        verify_node(EnergyGain(name="n", timestep=0.01), checks=["nope"])


def test_builtin_node_passes_full_battery():
    assert_node_verified(
        SpringDamperNode(name="s", timestep=0.01, stiffness=10.0,
                         damping=1.0, rest_length=1.0),
        max_examples=100, derandomize=True,
    )


def test_assert_node_verified_reports_all_failures():
    with pytest.raises(AssertionError) as ei:
        assert_node_verified(ForwardNaN(name="n", timestep=0.01),
                             bounds={"x": (0.0, 1.0)}, **KW)
    msg = str(ei.value)
    assert "finite: FAIL" in msg and "gradient_finite: FAIL" in msg
