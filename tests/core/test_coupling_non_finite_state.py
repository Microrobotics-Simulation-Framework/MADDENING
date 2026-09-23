"""A coupling state the norm cannot evaluate never reads as converged.

MADD-ANO-019.  Every coupling norm divides a field's change by the
field's own magnitude and drops a field whose magnitude sits inside the
dead band.  A NaN magnitude compared False against the dead band, so a
field that had just diverged to NaN was *dropped* -- it contributed
exactly zero, and a relaxed iteration that overflowed float32 was
reported ``residual=0.0, converged=True``.  Under the L2 norm a second
route reached the same verdict one pass earlier, on a *finite* state
of ``-1.2e38``: the scale's float32 reciprocal is subnormal there and
the broadcast divide flushed it to zero.

The fixture is the two-mode relay from ``test_coupling_error_bound``
with eigenvalues ``(-0.95, 0.3)`` under ``acceleration="fixed",
relaxation=1.5``: the relaxed operator has eigenvalue ``-1.925`` on
the first mode, so the iterate doubles in magnitude every pass and
leaves float32 range after about 136 of them.  A fresh graph is built
for every measurement; ``run_scan`` and ``step`` both advance the
graph's own state.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    _scaled_change,
    coupling_residual_interface,
    coupling_residual_l2,
    coupling_residual_mixed,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

SOLVERS = ("ift", "fori")
NORMS = ("l2", "mixed", "interface")

#: Enough passes to leave float32 range (about 136) with margin, and
#: small enough that the ``fori`` path's unrolled cap stays cheap.
_CAP = 200


class _TwoMode(SimulationNode):
    """``x <- rho * u + c`` on two independent modes at once."""

    def __init__(self, name, rho, c):
        super().__init__(name=name, timestep=1.0)
        self._rho = jnp.asarray(rho, jnp.float32)
        self._c = jnp.asarray(c, jnp.float32)

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._rho * boundary_inputs["u"] + self._c}


def _diverging_graph(solver, norm, **group_kw):
    """A fresh relay whose relaxed iteration diverges on its first mode."""
    gm = GraphManager()
    gm.add_node(_TwoMode("a", (-0.95, 0.3), (1.0, 1.0)))
    gm.add_node(_TwoMode("b", (1.0, 1.0), (0.0, 0.0)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, solver=solver, convergence_norm=norm,
              acceleration="fixed", relaxation=1.5, max_iterations=_CAP)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _state_is_finite(gm) -> bool:
    return all(
        math.isfinite(float(v))
        for n in ("a", "b") for v in gm.get_node_state(n)["x"]
    )


def _assert_reported_as_diverged(d, where):
    assert d["converged"] is False, f"{where}: converged on a diverged state: {d}"
    assert not math.isfinite(d["residual"]), (
        f"{where}: residual {d['residual']!r} is finite on a diverged state"
    )
    assert d["residual"] != 0.0, f"{where}: residual reads zero: {d}"
    assert d["ratio_usable"] is False, f"{where}: {d}"
    assert not math.isfinite(d["error_estimate"]), f"{where}: {d}"


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_a_non_finite_state_never_reads_as_converged(solver, norm):
    """Both solvers, every norm: the diverged step is reported diverged.

    Before the fix all six combinations reported ``residual=0.0,
    converged=True``; ``mixed`` and ``interface`` returned a NaN state
    and ``l2`` returned the last finite iterate at ``-1.2e38`` with
    ``ratio_usable=True`` as well.
    """
    gm = _diverging_graph(solver, norm)
    gm.step()
    assert not _state_is_finite(gm), (
        "the fixture no longer diverges; it cannot express the defect"
    )
    d = gm.coupling_diagnostics()["a+b"]
    _assert_reported_as_diverged(d, f"{solver}/{norm}")
    assert d["iterations"] == _CAP, (
        f"{solver}/{norm}: a diverged iteration must run to its cap, "
        f"not exit early on a criterion; got {d['iterations']}"
    )


@pytest.mark.parametrize("norm", NORMS)
def test_strict_convergence_names_the_non_finite_state(norm):
    """``strict_convergence=True`` raises, and says why.

    The pre-existing ``without converging`` message diagnoses a cap
    that was too small; a diverged iteration is a different failure
    and no larger cap would help, so the message has to name it.
    """
    gm = _diverging_graph("ift", norm, strict_convergence=True)
    with pytest.raises(Exception, match="state is non-finite") as info:
        gm.step()
    msg = str(info.value)
    assert "no larger max_iterations would help" in msg
    assert "Raise max_iterations" not in msg, (
        "exactly one message must fire, and it must be the one that "
        "names the cause rather than the one that advises a larger cap"
    )


@pytest.mark.parametrize("solver", SOLVERS)
def test_run_scan_after_divergence_reports_the_same_verdict(solver):
    """The scan entry point carries the non-finite verdict, step after step."""
    gm = _diverging_graph(solver, "l2")
    out = gm.run_scan(3)
    assert not bool(jnp.all(jnp.isfinite(out["a"]["x"])))
    _assert_reported_as_diverged(gm.coupling_diagnostics()["a+b"], f"{solver}/run_scan")


# ----------------------------------------------------------------------
# The helper every norm is built on
# ----------------------------------------------------------------------

def _pair(new, old):
    return jnp.asarray(new, jnp.float32), jnp.asarray(old, jnp.float32)


@pytest.mark.parametrize("rtol", [1.0, 1e-6], ids=["l2-scale", "mixed-scale"])
@pytest.mark.parametrize("new, old", [
    pytest.param([np.nan, 1.4], [np.inf, 1.4], id="inf-to-nan"),
    pytest.param([np.nan, np.nan], [np.nan, np.nan], id="nan-to-nan"),
    pytest.param([np.inf, 1.4], [-1.22e38, 1.4], id="finite-to-inf"),
    pytest.param([-np.inf, 1.4], [np.inf, 1.4], id="inf-to-minus-inf"),
    pytest.param([np.nan, 1.4], [1.0, 1.4], id="finite-to-nan"),
])
def test_a_non_finite_field_is_active_and_contributes_inf(new, old, rtol):
    """The dead band never swallows a non-finite field.

    ``NaN > atol`` is False, which is how the field used to become
    inactive and contribute zero.  A field it cannot evaluate has to
    be *active* (it counts) and *infinite* (it fails).
    """
    scaled, active = _scaled_change(*_pair(new, old), 0.0, rtol)
    assert bool(active)
    assert bool(jnp.all(jnp.isposinf(scaled))), np.asarray(scaled)


def test_a_reference_beyond_the_dtype_range_contributes_inf():
    """The L2 route: finite, but the scale's reciprocal is subnormal.

    ``1.16e38`` against ``-1.22e38`` is a change of 1.95 times the
    reference.  At ``rtol=1.0`` the divide read exactly ``0.0`` on the
    CPU backend, which is how the L2 norm reported convergence on a
    finite state one pass before it overflowed.  At ``rtol=1e-6`` the
    same pair is in range and must still be measured, not refused.
    """
    scaled, active = _scaled_change(*_pair([1.16e38, 1.4], [-1.22e38, 1.4]), 0.0, 1.0)
    assert bool(active)
    assert bool(jnp.all(jnp.isposinf(scaled))), np.asarray(scaled)
    scaled, active = _scaled_change(*_pair([1.16e38, 1.4], [-1.22e38, 1.4]), 0.0, 1e-6)
    assert bool(active)
    assert math.isclose(float(scaled[0]), 1.9508195e6, rel_tol=1e-6), float(scaled[0])
    assert float(scaled[1]) == 0.0


def test_the_dead_band_still_drops_a_field_that_is_exactly_zero():
    """The guard changes nothing about what the dead band is for."""
    scaled, active = _scaled_change(*_pair([0.0, 0.0], [0.0, 0.0]), 0.0, 1e-6)
    assert not bool(active)
    assert bool(jnp.all(scaled == 0.0))
    # Inside a declared dead band: at zero within tolerance, dropped.
    scaled, active = _scaled_change(*_pair([1e-9, 0.0], [2e-9, 0.0]), 1e-8, 1e-6)
    assert not bool(active)
    assert bool(jnp.all(scaled == 0.0))


def test_a_finite_field_is_bit_identical_through_the_guard():
    """On a finite in-range field the helper is the formula it was.

    Written against the closed form rather than a recorded number, so
    it holds on every backend the divide is exact on; the 438 recorded
    coupling verdicts and the ``expensive-pair`` baseline rest on this.
    """
    key = jax.random.PRNGKey(19)
    k1, k2, k3 = jax.random.split(key, 3)
    for scale_exp in (-20.0, -3.0, 0.0, 6.0, 30.0):
        mag = 10.0 ** scale_exp
        old = mag * jax.random.normal(k1, (64,), jnp.float32)
        new = old + 1e-3 * mag * jax.random.normal(k2, (64,), jnp.float32)
        for rtol in (1.0, 1e-6):
            got, active = _scaled_change(new, old, 0.0, rtol)
            ref = jnp.maximum(jnp.max(jnp.abs(new)), jnp.max(jnp.abs(old)))
            want = jnp.abs(new - old) / jnp.where(rtol * ref > 0, rtol * ref, 1.0)
            assert bool(active)
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # And an integer-valued field inside a step is unaffected too.
    old = jax.random.normal(k3, (8,), jnp.float32)
    got, _ = _scaled_change(old, old, 0.0, 1e-6)
    assert bool(jnp.all(got == 0.0))


def _norm_states(value):
    s_old = {"a": {"x": jnp.asarray([1.0, 2.0], jnp.float32)}}
    s_new = {"a": {"x": jnp.asarray([value, 2.0], jnp.float32)}}
    return s_new, s_old


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_every_norm_reads_a_non_finite_field_as_inf(value):
    """All three norms inherit the rule from the one helper; none is special-cased."""
    s_new, s_old = _norm_states(value)
    edge = EdgeSpec(source_node="a", target_node="a", source_field="x", target_field="u")
    assert bool(jnp.isposinf(coupling_residual_l2(s_new, s_old, ["a"])))
    assert bool(jnp.isposinf(coupling_residual_mixed(s_new, s_old, ["a"], 0.0, 1e-6)))
    assert bool(jnp.isposinf(coupling_residual_interface(s_new, s_old, [edge], 0.0, 1e-6)))


# ---------------------------------------------------------------------------
# A non-finite field the norm does not read
#
# The rule above lives in the norm, and a norm only sees the fields it
# reads.  ``convergence_norm="interface"`` reads edge sources, so a NaN
# in an internal field -- one no edge reads -- left the residual finite:
# ``converged=True``, ``strict_convergence`` silent, ``spectral_usable``
# True, on a state that was not finite.  The verdict is now taken over
# every floating field of the returned state.
# ---------------------------------------------------------------------------


class _WithInternal(SimulationNode):
    """``x <- 0.5 u + c`` (read by the relay) beside ``z <- z_pre + k + w`` (read by nothing)."""

    def __init__(self, name, z0=0.0, k=0.1):
        super().__init__(name=name, timestep=1.0, c=jnp.float32(1.0), k=jnp.float32(k))
        self._z0 = z0

    def initial_state(self):
        return {"x": jnp.float32(1.0), "z": jnp.float32(self._z0)}

    def state_fields(self):
        return ["x", "z"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=jnp.float32(0)),
                "w": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=jnp.float32(0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": 0.5 * boundary_inputs["u"] + p["c"],
                "z": state["z"] + p["k"] + boundary_inputs["w"]}


class _ScalarRelay(SimulationNode):
    def initial_state(self):
        return {"x": jnp.float32(1.0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=jnp.float32(0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs["u"]}


_ROUTES = ("initial state", "parameter", "external input", "none")


def _internal_field_graph(route, norm, solver="ift", strict=False):
    gm = GraphManager()
    gm.add_node(_WithInternal("a", z0=np.nan if route == "initial state" else 0.0,
                              k=np.nan if route == "parameter" else 0.1))
    gm.add_node(_ScalarRelay("b", timestep=1.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_external_input("a", "w", shape=(), dtype=jnp.float32)
    kw = dict(tolerance=1e-4) if norm == "l2" else dict(rtol=1e-3)
    if solver == "ift":
        kw["strict_convergence"] = strict
    gm.add_coupling_group(["a", "b"], convergence_norm=norm, diagnostics=True,
                          max_iterations=40, solver=solver, **kw)
    gm.compile()
    ext = ({"a": {"w": jnp.float32(np.inf)}} if route == "external input"
           else {"a": {"w": jnp.float32(0.0)}})
    return gm, ext


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("route", _ROUTES[:3])
def test_a_non_finite_field_no_edge_reads_is_still_a_non_finite_state(route, norm, solver):
    """Initial state, parameter or external input; every norm, both solvers.

    ``interface`` is the norm that used to miss it (``l2`` and
    ``mixed`` read every float field); the other two are here so the
    three agree by construction rather than by coincidence.
    """
    gm, ext = _internal_field_graph(route, norm, solver)
    gm.step(ext)
    z = float(gm.get_node_state("a")["z"])
    assert not math.isfinite(z), "fixture premise: z is non-finite"
    assert math.isfinite(float(gm.get_node_state("a")["x"])), (
        "fixture premise: the interface field is finite, so only the "
        "all-fields rule can see the non-finite state"
    )
    d = gm.coupling_diagnostics()["a+b"]
    _assert_reported_as_diverged(d, f"{route}/{norm}/{solver}")
    assert d["spectral_usable"] is False, d
    assert d["gradient_bound_usable"] is False, d
    if solver == "ift":
        # A Jacobian at a destroyed state describes nothing: NaN, never
        # a radius that reads as "contracts at 0.5".
        assert math.isnan(d["rho_spectral"]), d


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("route", _ROUTES[:3])
def test_strict_convergence_raises_on_a_non_finite_field_no_edge_reads(route, norm):
    """The strict guard reads the same verdict, so it names the non-finite state."""
    gm, ext = _internal_field_graph(route, norm, strict=True)
    with pytest.raises(Exception, match="state is non-finite"):
        gm.step(ext)
        jax.block_until_ready(gm.get_node_state("a")["x"])


@pytest.mark.parametrize("norm", NORMS)
def test_a_finite_internal_field_leaves_the_verdict_alone(norm):
    """The control: the same graph with every field finite converges as before."""
    gm, ext = _internal_field_graph("none", norm)
    gm.step(ext)
    d = gm.coupling_diagnostics()["a+b"]
    assert d["converged"] is True, d
    assert math.isfinite(d["residual"]) and d["ratio_usable"] is True, d
    assert d["spectral_usable"] is True, d


# ---------------------------------------------------------------------------
# The underflow end of the range rule
#
# Under "mixed" and "interface" the scale is ``rtol * max|v|``.  Below
# ``finfo.tiny / rtol`` (~1.2e-32 at the default rtol) that product is
# subnormal, the CPU backend flushes it to zero, and the field used to go
# *inactive* -- a dead band the caller never declared: ``iterations=1,
# residual=0.0, converged=True`` on a field 90% from its fixed point.
# The field itself is a normal float32 with all its bits, so it is
# measured on a rescaled pair instead of being excluded or failed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("magnitude", (1e-31, 1e-33, 1e-36))
def test_a_field_whose_scale_underflows_is_still_measured(magnitude):
    """A 100% change reads ``1 / rtol`` at every normal magnitude.

    Against the float64 closed form, not bit-for-bit: the rescaled
    quotient is exact in real arithmetic but rounds on its own path.
    """
    rtol = 1e-6
    old = jnp.array([magnitude, 0.5 * magnitude], jnp.float32)
    new = 2.0 * old
    if magnitude < 1e-32:
        assert 2.0 * magnitude * rtol < float(np.finfo(np.float32).tiny), (
            "fixture premise: the scale is subnormal"
        )
    got, active = _scaled_change(new, old, 0.0, rtol)
    assert bool(active)
    o64, n64 = np.asarray(old, np.float64), np.asarray(new, np.float64)
    want = np.abs(n64 - o64) / (rtol * np.max(np.abs(n64)))
    np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=1e-6)
    s_old = {"n": {"x": old}}
    s_new = {"n": {"x": new}}
    edge = EdgeSpec(source_node="n", target_node="n", source_field="x", target_field="u")
    assert float(coupling_residual_mixed(s_new, s_old, ["n"], 0.0, rtol)) > 1e5
    assert float(coupling_residual_interface(s_new, s_old, [edge], 0.0, rtol)) > 1e5


def test_the_dead_band_still_wins_at_the_underflow_end():
    """A caller who declared the field noise still gets it excluded."""
    old = jnp.array([1e-33], jnp.float32)
    got, active = _scaled_change(2.0 * old, old, 1e-30, 1e-6)
    assert not bool(active)
    assert float(got[0]) == 0.0


class _TinyLinear(SimulationNode):
    """``x <- g u + c`` on a length-1 field, for fixed points at any magnitude."""

    def __init__(self, name, g, c):
        super().__init__(name=name, timestep=1.0)
        self._g, self._c = g, c

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._g * boundary_inputs["u"] + self._c}


def _tiny_graph(norm, c):
    gm = GraphManager()
    gm.add_node(_TinyLinear("a", 0.9, c))
    gm.add_node(_TinyLinear("b", 1.0, 0.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], convergence_norm=norm, diagnostics=True,
                          max_iterations=20)
    gm.compile()
    return gm


@pytest.mark.parametrize("norm", ("mixed", "interface"))
def test_the_verdict_does_not_change_below_the_scale_underflow(norm):
    """``x* = 10 c``: the same group at ``c = 1e-30`` and ``c = 1e-33``.

    The criterion is a ratio, so moving every quantity by three decades
    must not move the verdict or the pass count.  At 1e-33 the scale
    ``rtol * max|x|`` is subnormal; before the fix that group reported
    one pass and ``converged=True``.
    """
    ref = _tiny_graph(norm, 1e-30)
    ref.step()
    want = ref.coupling_diagnostics()["a+b"]
    gm = _tiny_graph(norm, 1e-33)
    gm.step()
    got = gm.coupling_diagnostics()["a+b"]
    assert got["iterations"] == want["iterations"], (got, want)
    assert got["converged"] is want["converged"] is False, (got, want)
    assert got["residual"] == pytest.approx(want["residual"], rel=1e-4)
