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


class ParamsPathDivergence(_Scalar):
    """Reads ``rest`` from ``self.params`` on the baked path but forgets
    it on the injected path — the calibration-of-the-wrong-model fault."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=2.0, rest=1.0)

    def update(self, s, bi, dt, *, params=None):
        if params is None:
            return {"x": s["x"] - dt * self.params["k"] * (s["x"] - self.params["rest"])}
        return {"x": s["x"] - dt * params["k"] * s["x"]}


class ParamsGradientNaN(_Scalar):
    """Forward is finite for every ``k``; d/dk of ``sqrt(k**2)`` is NaN at
    the baked value ``k == 0``."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=0.0)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": s["x"] + dt * jnp.sqrt(p["k"] ** 2)}


class ParamsDeadLeaf(_Scalar):
    """``rate`` is in ``params_pytree()`` (a float) but ``update`` reads
    it from ``self.params``: injected and baked paths agree, the gradient
    wrt the injected leaf is identically zero — the heat-node ``length``
    trap this check was written for."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=2.0, rate=0.5)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": s["x"] - dt * p["k"] * s["x"] * self.params["rate"]}


class ParamsSometimesEffective(_Scalar):
    """``bounce`` only matters when ``x < 0`` (half the samples)."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=2.0, bounce=0.5)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        x = s["x"] - dt * p["k"] * s["x"]
        return {"x": jnp.where(x < 0, -x * p["bounce"], x)}


class ParamsClean(_Scalar):
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=2.0, rest=1.0, label="a", n=3)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": s["x"] - dt * p["k"] * (s["x"] - p["rest"])}


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


def test_params_path_divergence_caught():
    res = verify_node(ParamsPathDivergence(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)},
                      checks=["params_consistent", "params_gradient_finite"], **KW)
    assert res["params_consistent"].failed
    assert "baked/injected params mismatch in 'x'" in res["params_consistent"].detail
    assert res["params_gradient_finite"].passed


def test_params_gradient_only_nan_caught():
    res = verify_node(ParamsGradientNaN(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)}, **KW)
    assert res["finite"].passed
    assert res["gradient_finite"].passed          # d/dx is fine
    assert res["params_consistent"].passed
    assert res["params_gradient_finite"].failed   # d/dk is not
    assert "wrt param 'k'" in res["params_gradient_finite"].detail


def test_params_dead_leaf_caught_by_effective_check():
    res = verify_node(ParamsDeadLeaf(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)}, **KW)
    assert res["params_consistent"].passed        # both paths agree ...
    assert res["params_gradient_finite"].passed   # ... and zero is finite
    assert res["params_effective"].failed         # but 'rate' is dead
    assert "['rate']" in res["params_effective"].detail
    assert "'k'" not in res["params_effective"].detail


def test_params_effective_aggregates_over_samples():
    node = ParamsSometimesEffective(name="n", timestep=0.01)
    res = verify_node(node, bounds={"x": (-1.0, 1.0)},
                      checks=["params_effective"], **KW)
    assert res["params_effective"].passed
    # Sampling only x > 0 never exercises the bounce branch: reported,
    # so an envelope that cannot reach a parameter is visible.
    res = verify_node(node, bounds={"x": (0.5, 1.0)},
                      checks=["params_effective"], **KW)
    assert res["params_effective"].failed
    assert "['bounce']" in res["params_effective"].detail


def test_params_effective_respects_trainable_false():
    from maddening.core.params import ParamSpec

    class Declared(ParamsDeadLeaf):
        def param_specs(self):
            return {"rate": ParamSpec(trainable=False, description="fixed")}

    res = verify_node(Declared(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)}, checks=["params_effective"], **KW)
    assert res["params_effective"].passed


def test_params_checks_skip_for_nodes_without_params():
    res = verify_node(EnergyGain(name="n", timestep=0.01),
                      bounds={"x": (0.0, 1.0)},
                      checks=["params_consistent", "params_gradient_finite"], **KW)
    for name in ("params_consistent", "params_gradient_finite"):
        assert res[name].skipped and res[name].passed
        assert "does not take a params keyword" in str(res[name])
    # SKIP never fails the assertion form.
    assert_node_verified(EnergyGain(name="n", timestep=0.01),
                         bounds={"x": (0.0, 1.0)},
                         checks=["params_consistent"], **KW)


def test_params_checks_pass_for_clean_node_and_ignore_structural_entries():
    node = ParamsClean(name="n", timestep=0.01)
    assert set(node.params_pytree()) == {"k", "rest"}   # not label / n
    res = verify_node(node, bounds={"x": (0.0, 1.0)}, **KW)
    assert all(r.passed for r in res.values()), [str(r) for r in res.values()]
    assert res["params_consistent"].n_examples > 0
    assert res["params_gradient_finite"].n_examples > 0


class Monitor(SimulationNode):
    """bool + int32 state: the sampler must keep those dtypes."""
    def initial_state(self):
        return {"ok": jnp.array(True), "count": jnp.array(0, jnp.int32),
                "x": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt):
        return {"ok": s["ok"] & (s["x"] < 1e3), "count": s["count"] + 1,
                "x": s["x"] * 0.5}


class MonitorDtypeDrift(Monitor):
    def update(self, s, bi, dt):
        out = super().update(s, bi, dt)
        out["count"] = out["count"].astype(jnp.float32)   # drift
        return out


def test_non_float_state_sampled_with_its_dtype_and_structure_checked():
    res = verify_node(Monitor(name="m", timestep=0.01),
                      bounds={"x": (0.0, 1.0), "count": (0, 100)}, **KW)
    assert all(r.passed for r in res.values()), [str(r) for r in res.values()]
    assert res["gradient_finite"].passed     # only 'x' differentiated
    bad = verify_node(MonitorDtypeDrift(name="m", timestep=0.01),
                      bounds={"x": (0.0, 1.0)}, checks=["structure"], **KW)
    assert bad["structure"].failed
    assert "dtype mismatch in 'count'" in bad["structure"].detail


def test_assert_node_verified_reports_all_failures():
    with pytest.raises(AssertionError) as ei:
        assert_node_verified(ForwardNaN(name="n", timestep=0.01),
                             bounds={"x": (0.0, 1.0)}, **KW)
    msg = str(ei.value)
    assert "finite: FAIL" in msg and "gradient_finite: FAIL" in msg
