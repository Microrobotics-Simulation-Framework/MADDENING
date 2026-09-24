"""The ``verify_node`` battery must fire on planted faults, not just pass.

Each fault class isolates one detector: a forward NaN, a gradient-only NaN,
a narrow-window NaN (exercises shrinking), a jit/eager divergence, and an
energy gain.  A harness that has never been seen to fail is decoration.
"""

import jax
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


# ---------------------------------------------------------------------------
# params_effective through the solver paths (derivatives / implicit_residual)
# ---------------------------------------------------------------------------
#
# Every fixture below decays ``x`` at rate ``k``; the state envelope
# ``(0.25, 1.0)`` keeps ``x`` away from zero so ``d(-k x)/dk = -x`` is
# non-zero on every sample -- a fixture sitting at ``x == 0`` (the analogue
# of a spring at its rest length) has a zero sensitivity to ``k`` whatever
# the path does with it, and could not express the fault.

class _Decay(_Scalar):
    """``update`` and ``derivatives`` both read ``k`` from the merged dict."""
    def __init__(self, name, timestep, **extra):
        super().__init__(name, timestep, k=2.0, **extra)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": s["x"] - dt * p["k"] * s["x"]}

    def derivatives(self, s, bi, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": -p["k"] * s["x"]}


class DerivativesIgnoreInjectedParams(_Decay):
    """Takes the keyword and reads ``self.params`` anyway: the fault the
    guard cannot see (the signature is right) and this probe exists for."""
    def derivatives(self, s, bi, *, params=None):
        return {"x": -self.params["k"] * s["x"]}


class DerivativesLegacySignature(_Decay):
    """No keyword at all: ``integrate_node(..., params=...)`` refuses it."""
    def derivatives(self, s, bi):
        return {"x": -self.params["k"] * s["x"]}


class DerivativesOmitALeaf(_Decay):
    """``bounce`` matters to ``update`` (a floor) and is legitimately absent
    from the continuous right-hand side -- the ball's ``elasticity``."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, bounce=0.5)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        x = s["x"] - dt * p["k"] * s["x"]
        return {"x": jnp.where(x < 0.5, 0.5 + (0.5 - x) * p["bounce"], x)}


class ResidualIgnoresInjectedParams(_Decay):
    """A standalone ``implicit_residual`` (not built on ``derivatives``)
    that takes the keyword and reads ``self.params``."""
    def implicit_residual(self, s_new, s_old, bi, dt, *, params=None):
        return {"x": s_new["x"] - s_old["x"] + dt * self.params["k"] * s_new["x"]}


class ResidualClean(_Decay):
    def implicit_residual(self, s_new, s_old, bi, dt, *, params=None):
        d = self.derivatives(s_new, bi, params=params)
        return {"x": s_new["x"] - s_old["x"] - dt * d["x"]}


class DerivativesNotApplicable(_Decay):
    """A discrete node: the override exists only to say so."""
    def derivatives(self, s, bi, *, params=None):
        raise NotImplementedError("discrete update, no ODE form")


_DECAY_BOUNDS = {"x": (0.25, 1.0)}


def test_derivatives_that_ignore_the_injected_params_are_caught():
    res = verify_node(DerivativesIgnoreInjectedParams(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, **KW)
    assert res["params_consistent"].passed      # update() is clean
    assert res["params_effective"].failed       # derivatives() is not
    detail = res["params_effective"].detail
    assert "derivatives() reads ['k'] from self.params" in detail
    assert "update()" not in detail.split("derivatives()")[0]  # update is not blamed


def test_a_legacy_derivatives_signature_is_caught():
    res = verify_node(DerivativesLegacySignature(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert res["params_effective"].failed
    assert "update() takes params but derivatives() does not" in res["params_effective"].detail


def test_a_clean_derivatives_override_passes_and_the_detail_names_the_paths():
    res = verify_node(_Decay(name="n", timestep=0.01), bounds=_DECAY_BOUNDS, **KW)
    assert all(r.passed for r in res.values()), [str(r) for r in res.values()]
    assert res["params_effective"].detail == "paths checked: update, derivatives"


def test_a_leaf_the_right_hand_side_does_not_consume_is_reported_not_failed():
    res = verify_node(DerivativesOmitALeaf(name="n", timestep=0.01),
                      bounds={"x": (0.25, 0.75)}, checks=["params_effective"], **KW)
    assert res["params_effective"].passed, res["params_effective"].detail
    assert "not consumed by derivatives(): ['bounce']" in res["params_effective"].detail
    assert "'k'" not in res["params_effective"].detail


def test_an_implicit_residual_that_ignores_the_injected_params_is_caught():
    res = verify_node(ResidualIgnoresInjectedParams(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert res["params_effective"].failed
    assert "implicit_residual() reads ['k'] from self.params" in res["params_effective"].detail
    assert "derivatives() reads" not in res["params_effective"].detail


def test_a_clean_implicit_residual_is_checked_and_named():
    res = verify_node(ResidualClean(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert res["params_effective"].passed, res["params_effective"].detail
    assert res["params_effective"].detail == "paths checked: update, derivatives, implicit_residual"


def test_a_not_applicable_derivatives_override_is_recorded_not_failed():
    res = verify_node(DerivativesNotApplicable(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert res["params_effective"].passed, res["params_effective"].detail
    assert "derivatives(): not applicable" in res["params_effective"].detail
    assert "paths checked: update" in res["params_effective"].detail


class TwoCompartmentExchange(SimulationNode):
    """A correct, conserving node: ``dA/dt = -k (A - B)``, ``dB/dt = +k (A -
    B)``, ``k`` read from the merged dict on every path.  ``A + B`` is
    conserved exactly, so ``d(sum of outputs)/dk == 0`` on every sample
    although the injected ``k`` moves both fields -- the plain-sum probe
    FAILed this node with "update() probably reads them from
    self.params"."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=2.0)

    def initial_state(self):
        return {"a": jnp.asarray(0.8, jnp.float32), "b": jnp.asarray(0.2, jnp.float32)}

    def boundary_input_spec(self):
        return {}

    def halo_width(self):
        return {}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        flux = p["k"] * (s["a"] - s["b"])
        return {"a": s["a"] - dt * flux, "b": s["b"] + dt * flux}

    def derivatives(self, s, bi, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        flux = p["k"] * (s["a"] - s["b"])
        return {"a": -flux, "b": flux}


class DerivativesReadOneLeafFromSelf(_Decay):
    """Two leaves; ``derivatives`` takes the injected ``k`` and reads ``c``
    from ``self.params`` -- half wrong, which the check must still name."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, c=0.5)

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": s["x"] - dt * (p["k"] * s["x"] + p["c"])}

    def derivatives(self, s, bi, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": -(p["k"] * s["x"] + self.params["c"])}


def test_a_conserving_node_passes_the_effective_check():
    """The plain sum of the outputs is blind to a conserved exchange; the
    projected probe is not.  The fixture must be able to express it: the
    plain-sum gradient really is zero here."""
    node = TwoCompartmentExchange(name="n", timestep=0.01)
    s0 = node.initial_state()
    plain = jax.grad(lambda p: sum(jnp.sum(v) for v in node.update(s0, {}, 0.01, params=p).values()))(
        {"k": jnp.asarray(2.0)}
    )
    assert float(plain["k"]) == 0.0
    res = verify_node(node, bounds={"a": (0.1, 0.9), "b": (0.1, 0.9)}, **KW)
    assert all(r.passed for r in res.values()), [str(r) for r in res.values()]
    assert res["params_effective"].detail == "paths checked: update, derivatives"
    assert_node_verified(node, bounds={"a": (0.1, 0.9), "b": (0.1, 0.9)}, **KW)


def test_a_derivatives_that_reads_one_leaf_from_self_is_still_caught_by_the_projected_probe():
    res = verify_node(DerivativesReadOneLeafFromSelf(name="n", timestep=0.01),
                      bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert res["params_effective"].failed
    assert "derivatives() reads ['c'] from self.params" in res["params_effective"].detail
    assert "'k'" not in res["params_effective"].detail


def test_the_projection_is_fixed_by_its_seed():
    """Two independent runs see the same weights, so a verdict is
    reproducible from the documented seed rather than from process state."""
    from maddening.testing.verification import _projected, _projection_weights
    w1 = _projection_weights("temperature", (5,))
    w2 = _projection_weights("temperature", (5,))
    assert np.array_equal(w1, w2)
    assert np.all(np.abs(w1) >= 0.5) and np.all(np.abs(w1) <= 1.5)
    # Different fields of the same shape get different weights, so a
    # law that exchanges a quantity between two same-shaped fields with
    # equal coefficients cannot cancel.
    assert not np.array_equal(w1, _projection_weights("pressure", (5,)))
    out = {"a": jnp.ones(3), "b": jnp.ones(3), "n": jnp.zeros(3, jnp.int32)}
    assert float(_projected(out)) == pytest.approx(
        float(np.sum(_projection_weights("a", (3,))) + np.sum(_projection_weights("b", (3,))))
    )


def test_the_constructor_value_probe_restores_the_node():
    node = DerivativesIgnoreInjectedParams(name="n", timestep=0.01)
    before = dict(node.params)
    verify_node(node, bounds=_DECAY_BOUNDS, checks=["params_effective"], **KW)
    assert node.params == before


# ---------------------------------------------------------------------------
# params_effective: each path on its own, per element, by value
# ---------------------------------------------------------------------------
#
# The release audit planted five nodes on which an injected leaf has no
# effect (or half its effect) somewhere the graph or an integrator reads it,
# and one correct node; the previous check passed all five and failed the
# sixth.  ``_Osc`` is the correct reference every fixture perturbs:
# dx = v, dv = -k (x - x0) - c v + g, flux F = -k (x - x0).

from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec  # noqa: E402
from maddening.core.params import ParamSpec  # noqa: E402

_OSC_BOUNDS = {"x": (-1.0, 1.0), "v": (-1.0, 1.0)}


class _Osc(SimulationNode):
    def __init__(self, name="n", timestep=1e-2, k=4.0, c=0.3, x0=0.5, g=(1.0, 2.0, 3.0)):
        super().__init__(name, timestep, k=k, c=c, x0=x0, g=list(g))
        self._k_cache = k

    def initial_state(self):
        return {"x": jnp.zeros(3, jnp.float32), "v": jnp.zeros(3, jnp.float32)}

    def _p(self, params):
        return self.params if params is None else {**self.params, **params}

    def _rhs(self, s, p, k=None, g=None):
        k = p["k"] if k is None else k
        g = jnp.asarray(p["g"], jnp.float32) if g is None else g
        return {"x": s["v"], "v": -k * (s["x"] - p["x0"]) - p["c"] * s["v"] + g}

    def update(self, s, bi, dt, *, params=None):
        d = self._rhs(s, self._p(params))
        return {f: s[f] + dt * d[f] for f in s}

    def derivatives(self, s, bi, *, params=None):
        return self._rhs(s, self._p(params))

    def boundary_flux_spec(self):
        return {"F": BoundaryFluxSpec(shape=(3,))}

    def compute_boundary_fluxes(self, s, bi, dt, *, params=None):
        p = self._p(params)
        return {"F": -p["k"] * (s["x"] - p["x0"])}


class _UpdateIgnoresKFluxReadsIt(_Osc):
    """The flux reads the injected ``k`` and used to mask the dead update."""
    def update(self, s, bi, dt, *, params=None):
        d = self._rhs(s, self._p(params), k=self.params["k"])
        return {f: s[f] + dt * d[f] for f in s}


class _DerivativesReadCachedK(_Osc):
    """A copy made in ``__init__``: varying ``node.params`` cannot see it."""
    def derivatives(self, s, bi, *, params=None):
        return self._rhs(s, self._p(params), k=self._k_cache)


class _VectorLeafPartlyIgnored(_Osc):
    """``g[0]`` injected, ``g[1:]`` from ``self.params``: some element acts."""
    def update(self, s, bi, dt, *, params=None):
        p = self._p(params)
        g = jnp.concatenate([jnp.asarray(p["g"], jnp.float32)[:1],
                             jnp.asarray(self.params["g"][1:], jnp.float32)])
        d = self._rhs(s, p, g=g)
        return {f: s[f] + dt * d[f] for f in s}


class _HalfAndHalf(_Osc):
    """Non-zero gradient, half the effect."""
    def update(self, s, bi, dt, *, params=None):
        p = self._p(params)
        d = self._rhs(s, p, k=0.5 * p["k"] + 0.5 * self.params["k"])
        return {f: s[f] + dt * d[f] for f in s}


class _SplitWithInitCopy(_Osc):
    """Half the injected ``k``, half a copy made in ``__init__``: the
    injected value moves the output (half as far), and the in-place
    ``node.params`` swap moves nothing, so only a rebuild shows it."""
    def update(self, s, bi, dt, *, params=None):
        p = self._p(params)
        d = self._rhs(s, p, k=0.5 * p["k"] + 0.5 * self._k_cache)
        return {f: s[f] + dt * d[f] for f in s}


class _InterfaceCorrectionIgnoresK(_Osc):
    """The interface correction was never probed."""
    def interface_dof_indices(self):
        return {"x_bc": ("x", 0)}

    def boundary_input_spec(self):
        return {"x_bc": BoundaryInputSpec(shape=())}

    def compute_interface_correction(self, pre, bi, dt, *, params=None):
        k = self.params["k"]
        return {"x": [(0, pre["x"][0] + dt * pre["v"][0] - dt * dt * k * pre["x"][0])]}


class _ThresholdInDerivatives(_Osc):
    """CORRECT: ``x0`` reaches ``derivatives`` only through a comparison
    (a dead zone), so its gradient there is zero -- and the previous check
    called that a ``self.params`` read."""
    def derivatives(self, s, bi, *, params=None):
        p = self._p(params)
        on = (s["x"] > p["x0"]).astype(jnp.float32)
        return {"x": s["v"],
                "v": -p["k"] * s["x"] * on - p["c"] * s["v"] + jnp.asarray(p["g"], jnp.float32)}


def _effective(node, **kw):
    return verify_node(node, _OSC_BOUNDS, checks=["params_consistent", "params_effective"],
                       max_examples=30, derandomize=True, **kw)


def test_the_correct_oscillator_passes_and_every_path_is_named():
    res = _effective(_Osc())
    assert res["params_consistent"].passed and res["params_effective"].passed, res
    assert res["params_effective"].detail.startswith(
        "paths checked: update, compute_boundary_fluxes, derivatives")


@pytest.mark.parametrize("cls, expected", [
    pytest.param(_UpdateIgnoresKFluxReadsIt, "update() reads ['k'] from self.params",
                 id="update-dead-masked-by-flux"),
    pytest.param(_DerivativesReadCachedK, "derivatives() ignores the injected ['k']",
                 id="cached-in-init"),
    pytest.param(_VectorLeafPartlyIgnored, "update() reads ['g[1]', 'g[2]'] from self.params",
                 id="vector-leaf-per-element"),
    pytest.param(_HalfAndHalf, "update() does not apply the injected ['k']",
                 id="half-injected-half-self"),
    pytest.param(_SplitWithInitCopy,
                 "update() does not apply the injected ['k'] the way a node "
                 "constructed with that value does",
                 id="half-injected-half-init-copy"),
    pytest.param(_InterfaceCorrectionIgnoresK,
                 "compute_interface_correction() reads ['k'] from self.params",
                 id="interface-correction"),
])
def test_an_injected_leaf_that_does_not_act_somewhere_is_named(cls, expected):
    """``params_consistent`` passes every one of them -- at the
    constructor value both spellings read the same number."""
    res = _effective(cls())
    assert res["params_consistent"].passed
    assert res["params_effective"].failed, res["params_effective"]
    assert expected in res["params_effective"].detail, res["params_effective"].detail


def test_a_leaf_read_only_through_a_comparison_is_not_a_self_params_read():
    res = _effective(_ThresholdInDerivatives())
    assert res["params_effective"].passed, res["params_effective"].detail


def test_the_half_and_half_verdict_reports_the_fraction_missed():
    detail = _effective(_HalfAndHalf())["params_effective"].detail
    assert "0.50 of the perturbation's own effect" in detail


def test_the_split_with_an_init_copy_verdict_reports_the_fraction_missed():
    detail = _effective(_SplitWithInitCopy())["params_effective"].detail
    assert "0.50 of the perturbation's own effect" in detail


class _SplitCopyAsAudited(SimulationNode):
    """The confirmation audit's node, verbatim in substance: in a graph an
    injected ``k=4`` gave ``x(50)=0.218`` against ``0.130`` for a node
    built with ``k=4``, and ``params_effective`` passed it."""

    def __init__(self, name="n", timestep=0.01, k=2.0, initial_x=1.0):
        super().__init__(name, timestep, k=k, initial_x=initial_x)
        self._k0 = float(k)

    def initial_state(self):
        return {"x": jnp.asarray(self.params["initial_x"], jnp.float32)}

    def update(self, state, bi, dt, *, params=None):
        p = {**self.params, **(params or {})}
        k_eff = 0.5 * p["k"] + 0.5 * self._k0
        return {"x": state["x"] - dt * k_eff * state["x"]}

    def param_specs(self):
        return {**super().param_specs(), "k": ParamSpec(bounds=(0.0, None))}


def test_a_constant_split_with_an_init_copy_fails_params_effective():
    res = verify_node(_SplitCopyAsAudited(), bounds={"x": (-10.0, 10.0)},
                      checks=["params_consistent", "params_effective"],
                      max_examples=40, derandomize=True)
    assert res["params_consistent"].passed
    assert res["params_effective"].failed, res["params_effective"].detail
    assert "a copy made at construction" in res["params_effective"].detail


def test_the_value_probes_restore_the_node():
    """Both the in-place swap and the rebuild leave the node as it was."""
    for cls in (_DerivativesReadCachedK, _VectorLeafPartlyIgnored, _SplitWithInitCopy):
        node = cls()
        before = {k: (list(v) if isinstance(v, list) else v) for k, v in node.params.items()}
        _effective(node)
        assert node.params == before
