"""What ``converged=True`` is allowed to mean about the fixed point.

Until 0.4.0 it meant *the last pass was small*: the group compared
``||F(x) - x||`` against its threshold and said nothing about the
distance to the fixed point, which is larger by the amplification
``1/(1 - rho)`` of the slowest mode (``MADD-ANO-005``).  Two changes
close that, and they are tested here together because neither is worth
much without the other:

*The criterion is an error estimate.*  ``rho`` costs nothing -- it is
``r_k / r_{k-1}``, two numbers both solvers already carry -- so the
threshold is applied to ``r_k / (1 - rho)``.  The estimate is never
smaller than ``r_k``, so the criterion is never looser than the one it
replaces, and a group that used to stop on a single small step now has
to earn it.  Where the ratio cannot be trusted the raw residual test
stands in and ``ratio_usable`` says so, which is the honest answer and
not a silent one.

*The norm is scale-aware.*  A bound quoted in a field's own units is not
a bound anyone can read.  Every field's change is divided by that
field's own magnitude, and ``atol`` stops being a floor under the scale
-- which made every criterion absolute below ``atol/rtol`` -- and
becomes a dead band: at or below ``atol`` a field counts as zero and
leaves the norm.  So the same physics converges the same way whether a
force is written in newtons or micronewtons, and a field that is
legitimately at zero neither divides by something tiny nor blocks the
group forever.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling.acceleration import (
    error_amplification,
    estimated_error,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

# ---------------------------------------------------------------------------
# An analytic linear fixed point with a known contraction rate
#
# Gauss-Seidel on ``a = 0.5 b + 1``, ``b = 0.5 a`` advances ``b`` by
# ``b <- 0.25 b + 0.5``, so ``rho = 0.25`` exactly and the fixed point is
# ``(4/3, 2/3)``.  Both the rate and the answer are known in closed
# form, which is what lets the bound be checked rather than trusted.
# ---------------------------------------------------------------------------

_RHO = 0.25
_AMPLIFICATION = 1.0 / (1.0 - _RHO)          # 4/3
_FIXED_POINT = (4.0 / 3.0, 2.0 / 3.0)


class _Affine(SimulationNode):
    """``x <- gain * u + bias``, scaled into whatever units are asked for."""

    def __init__(self, name, gain, bias, x0=0.0, scale=1.0):
        super().__init__(name=name, timestep=1.0, gain=gain,
                         bias=bias * scale, x0=x0 * scale)

    def initial_state(self):
        return {"x": jnp.asarray(self.params["x0"], jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": jnp.asarray(p["gain"]) * boundary_inputs["u"]
                + jnp.asarray(p["bias"])}


def _contracting_graph(*, scale=1.0, solver="ift", **group_kw):
    """The ``rho = 0.25`` cycle, optionally rescaled into small units."""
    gm = GraphManager()
    gm.add_node(_Affine("a", gain=0.5, bias=1.0, scale=scale))
    gm.add_node(_Affine("b", gain=0.5, bias=0.0, scale=scale))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, solver=solver, max_iterations=20)
    kw.update(group_kw)
    # ``tolerance`` is read only under the "l2" norm, and setting a knob a
    # norm never reads warns (fatally, under `filterwarnings = ["error"]`).
    # Default it only where it is live, so a caller choosing "mixed" or
    # "interface" does not inherit an inert one from this helper.
    if kw.get("convergence_norm", "l2") == "l2":
        kw.setdefault("tolerance", 1e-4)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _relative_distance(state, reference):
    """The group's L2 norm between two states -- the units the bound is in.

    Each field's difference is divided by that field's own magnitude,
    exactly as ``coupling_residual_l2`` does, so the number is directly
    comparable with ``error_estimate``.
    """
    total = 0.0
    for got, want in zip(state, reference):
        ref = max(abs(got), abs(want))
        if ref == 0.0:
            continue
        total += ((got - want) / ref) ** 2
    return total ** 0.5


def _ab(gm):
    return (float(gm.get_node_state("a")["x"]),
            float(gm.get_node_state("b")["x"]))


# ---------------------------------------------------------------------------
# The bound itself
# ---------------------------------------------------------------------------

def test_the_bound_recovers_the_contraction_rate_of_a_known_fixed_point():
    """``amplification`` is ``1/(1 - rho)`` for a group whose rho is known.

    The whole change rests on ``rho`` being free.  If the number the
    solver extracts from its own residual sequence is not the rate of
    the iteration it is running, nothing downstream of it means
    anything.
    """
    gm = _contracting_graph()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["ratio_usable"] is True
    assert d["amplification"] == pytest.approx(_AMPLIFICATION, rel=0.05), (
        f"measured 1/(1-rho) = {d['amplification']} for an iteration "
        f"whose rate is exactly {_RHO}"
    )


def test_converged_means_the_state_is_within_tolerance_of_the_fixed_point():
    """The invariant the change exists to establish, on a known answer.

    ``MADD-ANO-005`` is exactly this assertion failing: the old
    criterion compared the *step* against the tolerance, so a slowly
    contracting group could report success while sitting
    ``1/(1 - rho)`` steps away from the answer.  Here the answer is
    known in closed form, so the claim can be checked rather than
    estimated.
    """
    tol = 1e-4
    gm = _contracting_graph(tolerance=tol)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["converged"] is True
    distance = _relative_distance(_ab(gm), _FIXED_POINT)
    assert distance <= tol, (
        f"reported converged at tolerance {tol} but the returned state "
        f"is {distance} from the fixed point"
    )
    assert d["error_estimate"] >= distance * (1 - 1e-4), (
        f"the reported estimate {d['error_estimate']} has to bound the "
        f"distance it estimates ({distance}); the 1e-4 is float32 slack, "
        "and on this fixture the two agree to eight figures because the "
        "map is exactly linear"
    )


def test_the_bound_is_what_costs_the_extra_iterations():
    """The price of an honest criterion, made visible.

    The bound is never smaller than the residual, so it can only ever
    make a group iterate longer -- and on this fixture it does, by the
    ``log(1/(1-rho))/log(rho)`` passes the amplification is worth.  A
    change that cost nothing would not have been doing anything.
    """
    tol = 1e-4
    strict = _contracting_graph(tolerance=tol)
    strict.step()
    bounded = strict.coupling_diagnostics()["a+b"]

    # The same group held to the raw residual instead: loosen the
    # threshold by the amplification and it stops where the old
    # criterion would have.
    loose = _contracting_graph(tolerance=tol * _AMPLIFICATION)
    loose.step()
    unbounded = loose.coupling_diagnostics()["a+b"]

    assert bounded["iterations"] >= unbounded["iterations"]
    assert bounded["residual"] <= unbounded["residual"]
    assert _relative_distance(_ab(strict), _FIXED_POINT) <= _relative_distance(
        _ab(loose), _FIXED_POINT
    )


# ---------------------------------------------------------------------------
# When the estimate must be rejected
# ---------------------------------------------------------------------------

def test_a_growing_residual_is_rejected_rather_than_extrapolated():
    """``rho >= 1`` is not a contraction, so there is nothing to sum."""
    assert float(error_amplification(5.0, 0.5, 0.5)) == 0.0
    assert float(error_amplification(1.0, 1.0, 1.0)) == 0.0
    assert float(error_amplification(jnp.nan, 1.0, 1.0)) == 0.0
    assert float(error_amplification(0.5, 0.0, 0.0)) == 0.0
    # A rejected estimate reads as an amplification of one, i.e. the raw
    # residual test that shipped before 0.4.0.
    assert float(estimated_error(0.5, error_amplification(5.0, 0.5, 0.5))) == 0.5


def test_an_alternating_sequence_is_not_flattered_by_its_good_half():
    """The non-monotone case D2 was argued on, ``0.5, 5, 0.25, 2.5``.

    Every second one-step ratio on that sequence reads ``0.05`` and
    would claim an amplification of ``1.05`` for an iteration that is
    actually contracting at ``sqrt(0.5) = 0.71`` per pass -- a bound
    fourteen times too optimistic, on exactly the non-normal group that
    motivated the decision.  Taking the worse of the one-step ratio and
    the two-step ``sqrt(r_k / r_{k-2})`` is what makes the estimate
    survive alternation: the two-step rate cannot be fooled by it.
    """
    naive = 1.0 / (1.0 - 0.25 / 5.0)          # what r_k / r_{k-1} alone says
    assert naive == pytest.approx(1.0526, rel=1e-3), "premise"
    measured = float(error_amplification(0.25, 5.0, 0.5))
    assert measured == pytest.approx(1.0 / (1.0 - 0.5 ** 0.5), rel=1e-3), (
        "the two-step rate sqrt(r_k / r_{k-2}) = sqrt(0.5) is the one that "
        "describes this sequence"
    )
    assert measured > 3 * naive
    # Growth is still rejected outright: the *other* half of the same
    # sequence has no contraction to extrapolate at all.
    assert float(error_amplification(5.0, 0.5, 0.25)) == 0.0


def test_the_first_pass_has_no_ratio_and_reports_that_it_has_none():
    """``max_iterations=1`` gets one residual, so it gets no bound.

    A ratio needs two measurements and a single staggered pass makes
    one.  Inventing a rate there would be the "trusted bad estimate"
    the guard exists to avoid, so the group falls back to the raw
    residual test and says so, rather than reporting a bound it cannot
    have.
    """
    gm = _contracting_graph(max_iterations=1, tolerance=1e3)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["ratio_usable"] is False
    assert jnp.isnan(d["amplification"])
    assert d["error_estimate"] == pytest.approx(d["residual"])
    assert d["gradient_error_estimate"] == float("inf"), (
        "no observed contraction means nothing bounds the adjoint gap"
    )
    assert d["converged"] is True, (
        "the fallback is the criterion that shipped before 0.4.0, not a "
        "refusal to converge"
    )


# ---------------------------------------------------------------------------
# The scale-aware norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
def test_the_verdict_does_not_depend_on_the_units_a_field_is_written_in(norm):
    """The same physics, six decades smaller, converges the same way.

    This is the defect in its general form.  ``"l2"`` compared
    ``||dx||`` against ``tolerance`` unscaled and ``"mixed"`` /
    ``"interface"`` scaled by ``atol + rtol*|v|``, which is ``atol``
    alone once ``|v| << atol/rtol`` -- so a group whose quantities
    happened to be small in SI units met its criterion passes before
    one whose quantities were large, at the same physical accuracy.
    MIME's D2 drag force is 1.7e-05 N against a default ``atol=1e-8``;
    it satisfied its criterion on pass one while still percent-sized
    from its fixed point.
    """
    # Only the knobs this norm reads: "l2" reads ``tolerance`` (supplied
    # by the helper), the other two read ``atol``/``rtol``.
    kw = dict(convergence_norm=norm)
    if norm != "l2":
        kw.update(atol=1e-12, rtol=1e-4)
    big = _contracting_graph(scale=1.0, **kw)
    small = _contracting_graph(scale=1e-6, **kw)
    big.step()
    small.step()
    d_big = big.coupling_diagnostics()["a+b"]
    d_small = small.coupling_diagnostics()["a+b"]
    assert d_big["converged"] == d_small["converged"]
    assert d_big["iterations"] == d_small["iterations"]
    assert d_small["residual"] == pytest.approx(d_big["residual"], rel=1e-3)


def test_a_field_that_is_legitimately_zero_does_not_block_the_group():
    """The case the dead band exists for.

    Dividing by a field's own magnitude is only safe if something says
    what "zero" is, and nothing dimensionless can: ``atol`` is that
    statement, in the field's own units.  A field at or below it leaves
    the norm entirely rather than contributing a ratio of two
    round-offs, so a group carrying one still converges -- and the
    group is still held to its criterion on every field above the band.
    """
    gm = GraphManager()
    # ``b`` is pinned at zero: its bias and its gain are both zero, so
    # its state never leaves 0.0 and its relative change is 0/0.
    gm.add_node(_Affine("a", gain=0.0, bias=1.0))
    gm.add_node(_Affine("b", gain=0.0, bias=0.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=6,
                          tolerance=1e-6, atol=1e-8)
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert float(gm.get_node_state("b")["x"]) == 0.0, "fixture premise"
    assert jnp.isfinite(d["residual"]), (
        "a field at zero must not put a NaN or an inf in the norm"
    )
    assert d["converged"] is True, (
        "a group whose only unconverged field is one that is at zero has "
        "converged"
    )


class _TwoScale(SimulationNode):
    """``big <- gb*ub + bias_big`` and ``small <- gs*us + bias_small``.

    One node, two fields, as many decades apart as the caller asks for
    -- the shape the dead band was measured swallowing.  The two fields
    do not interact, so each one's fixed point is the scalar
    ``bias / (1 - gain**2)`` of its own cycle and the group's answer can
    be checked field by field.
    """

    def __init__(self, name, gain_big, gain_small, bias_big, bias_small):
        super().__init__(name=name, timestep=1.0, gain_big=gain_big,
                         gain_small=gain_small, bias_big=bias_big,
                         bias_small=bias_small)

    def initial_state(self):
        return {"big": jnp.asarray(0.0, jnp.float32),
                "small": jnp.asarray(0.0, jnp.float32)}

    def state_fields(self):
        return ["big", "small"]

    def boundary_input_spec(self):
        z = jnp.float32(0.0)
        return {"ub": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=z),
                "us": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=z)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {
            "big": (jnp.asarray(p["gain_big"]) * boundary_inputs["ub"]
                    + jnp.asarray(p["bias_big"])),
            "small": (jnp.asarray(p["gain_small"]) * boundary_inputs["us"]
                      + jnp.asarray(p["bias_small"])),
        }


def _two_scale_graph(small_bias, *, small_gain=0.5, big_gain=0.0, **group_kw):
    """A two-node cycle carrying one O(1) field and one tiny one.

    ``big_gain=0.0`` is the audit's fixture: the large field reaches its
    fixed point on the first pass and contributes nothing to any later
    residual, so whether the group keeps iterating is decided by the
    small field alone.  That is what made the dead band's exclusion
    visible as a wrong answer rather than as a slightly loose one.
    """
    gm = GraphManager()
    gm.add_node(_TwoScale("a", gain_big=big_gain, gain_small=small_gain,
                          bias_big=1.0, bias_small=small_bias))
    gm.add_node(_TwoScale("b", gain_big=big_gain, gain_small=small_gain,
                          bias_big=0.0, bias_small=0.0))
    for src, dst in (("a", "b"), ("b", "a")):
        gm.add_edge(source=src, target=dst, source_field="big",
                    target_field="ub")
        gm.add_edge(source=src, target=dst, source_field="small",
                    target_field="us")
    kw = dict(diagnostics=True, max_iterations=40)
    kw.update(group_kw)
    if kw.get("convergence_norm", "l2") == "l2":
        kw.setdefault("tolerance", 1e-6)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _fixed_point(bias, gain):
    """``a``'s value at the fixed point of ``a = g*b + bias, b = g*a``."""
    return bias / (1.0 - gain ** 2)


@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
def test_a_small_field_far_from_its_fixed_point_is_not_reported_converged(norm):
    """The defect, pinned at the magnitude it was measured at.

    ``atol`` used to default to ``1e-8``, which is a perfectly ordinary
    magnitude for a physical quantity in SI units -- the group here
    carries an O(1) field and an O(1e-9) one, and a 1.7e-05 N drag
    force is the case ``coupling_residual_mixed``'s own docstring is
    written around.  The dead band dropped the small field out of the
    norm, so the group met its criterion on the large field alone and
    returned after one pass reporting ``residual=0.0,
    error_estimate=0.0, converged=True`` with the small field 50% from
    its fixed point.  No ``tolerance`` could contradict a residual of
    exactly zero, and the one knob that could -- ``atol`` -- was the
    knob ``CouplingGroup`` warned was ignored.

    Both halves are asserted: the small field arrives, and it took more
    than the one pass the defect stopped after.  Under every norm,
    because the dead band lives in the shared ``_scaled_change`` and
    all three residuals call it.
    """
    kw = {"convergence_norm": norm}
    if norm != "l2":
        kw["rtol"] = 1e-6
    gm = _two_scale_graph(1e-9, **kw)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    exact_small = _fixed_point(1e-9, 0.5)
    small = float(gm.get_node_state("a")["small"])
    assert float(gm.get_node_state("a")["big"]) == pytest.approx(
        _fixed_point(1.0, 0.0), rel=1e-4,
    ), "fixture premise: the large field converges on the first pass"
    assert small == pytest.approx(exact_small, rel=1e-3), (
        f"the small field is {abs(small - exact_small) / exact_small:.0%} "
        f"from its fixed point while the group reports {d}"
    )
    assert d["iterations"] > 1, (
        "one pass cannot converge a field with a contraction of 0.25"
    )


def test_the_dead_band_excludes_a_field_only_when_the_caller_declares_it():
    """Raising ``atol`` is the opt-in, and it is honoured.

    The dead band is not being removed -- a field at the noise floor
    should not dominate a relative norm -- it is being made a statement
    the caller makes about their own units.  Here the small field
    contracts at 0.98 per pass, so it is nowhere near its fixed point
    within the cap and, in the norm, it holds the whole group there.
    Declaring it noise with ``atol`` above its magnitude releases the
    group; leaving ``atol`` at its default does not.  That difference
    is the knob doing its job, and under the default norm, where the
    library used to warn that it did nothing.
    """
    declared = _two_scale_graph(1e-9, small_gain=0.98, atol=1e-8)
    default = _two_scale_graph(1e-9, small_gain=0.98)
    declared.step()
    default.step()
    d_declared = declared.coupling_diagnostics()["a+b"]
    d_default = default.coupling_diagnostics()["a+b"]
    assert d_declared["converged"] is True, (
        "a field the caller called zero cannot hold the group"
    )
    assert d_default["converged"] is False, (
        "fixture premise: undeclared, the small field is what holds it"
    )
    assert d_declared["iterations"] < d_default["iterations"]


# ---------------------------------------------------------------------------
# The gradient-trust estimate
# ---------------------------------------------------------------------------

def test_the_gradient_trust_bound_bounds_the_adjoint_finite_difference_gap():
    """"My finite difference does not match" becomes a number to read.

    The IFT adjoint differentiates the *fixed point*; a finite
    difference of the forward differentiates the *truncated iterate*.
    They differ by roughly ``residual * cond(I - dF/dx)``, which is the
    number now reported -- estimated along the slowest mode the
    residual sequence reveals, which is the mode that dominates the
    conditioning.  Before this it was a documented sentence with no
    value attached.

    **The inequality asserted below is a property of this fixture, not
    of the field.**  ``gradient_error_estimate`` is numerically
    ``error_estimate`` and inherits every way that number understates:
    the same assertion fails by 122x on the two-mode contraction, which
    ``benchmarks/results/audit_040_final/coupling/repro_gradient_error_bound.py``
    reproduces.  This fixture is a single-mode contraction with a known
    rate, where the estimate is tight -- so what this pins is that the
    reported number is the right order for a well-conditioned group, not
    that it bounds anything in general.  The function's name predates
    the 0.4.0 rename and is referenced from the recorded audit report;
    read "bound" in it as "estimate".
    """
    def loss(p, gm=None):
        gm = gm if gm is not None else _contracting_graph(tolerance=1e-3)
        return jnp.sum(gm.run_scan(1, params=p)["a"]["x"])

    base = _contracting_graph(tolerance=1e-3).params
    analytic = float(jax.grad(loss)(base)["nodes"]["a"]["bias"])

    def shifted(delta):
        p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p["nodes"]["a"]["bias"] = base["nodes"]["a"]["bias"] + delta
        return float(loss(p))

    h = 1e-2
    fd = (shifted(h) - shifted(-h)) / (2 * h)

    gm = _contracting_graph(tolerance=1e-3)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["ratio_usable"] is True, "fixture premise: a measured contraction"
    assert d["gradient_error_estimate"] == pytest.approx(d["error_estimate"])
    assert abs(analytic - fd) <= max(d["gradient_error_estimate"], 1e-5), (
        f"analytic {analytic} vs finite difference {fd}: the adjoint may "
        f"only be as wrong as the reported estimate "
        f"({d['gradient_error_estimate']})"
    )


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_both_solvers_report_the_same_bound(solver):
    """``solver`` stays invisible in ``coupling_diagnostics()``.

    D2's guarantee is that the two paths return the same state and the
    same verdict; the new fields are derived from the same residual
    sequence on both, so they have to agree too or the guarantee has
    been narrowed to the fields that existed when it was written.
    """
    ift = _contracting_graph(solver="ift")
    fori = _contracting_graph(solver="fori")
    ift.step()
    fori.step()
    a, b = ift.coupling_diagnostics()["a+b"], fori.coupling_diagnostics()["a+b"]
    assert a["converged"] == b["converged"]
    assert a["ratio_usable"] == b["ratio_usable"]
    assert a["iterations"] == b["iterations"]
    assert a["residual"] == pytest.approx(b["residual"], rel=1e-5)
    assert a["error_estimate"] == pytest.approx(b["error_estimate"], rel=1e-5)
    assert _ab(ift) == pytest.approx(_ab(fori), rel=1e-6)


# ---------------------------------------------------------------------------
# What the estimate is short of, from the 2026-09-19 coupling audit
#
# Two mechanisms, both independent of the already-recorded one (the
# measure is not a metric, ``test_the_triangle_inequality_does_not_hold``).
# One is fixed here; one is not, and is pinned as a strict xfail so that
# fixing it is noticed.  See
# ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
# ---------------------------------------------------------------------------


def _slow_cycle(gain, **group_kw):
    """The ``rho = gain**2`` cycle -- slow enough for relaxation to matter.

    ``_contracting_graph``'s ``rho = 0.25`` converges in a handful of
    passes, which leaves no tail for a geometric series to be wrong
    about.  ``gain = 0.95`` gives ``rho = 0.9025`` and a fixed point of
    ``(1, gain) / (1 - rho)``.
    """
    gm = GraphManager()
    gm.add_node(_Affine("a", gain=gain, bias=1.0))
    gm.add_node(_Affine("b", gain=gain, bias=0.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


#: Relaxation factors either side of 1.  0.5 was 2.0x *conservative*
#: before the fix and 1.95 was 1.97x optimistic, so a test that only
#: looked at over-relaxation would have read the safe direction as
#: correct.
_RELAXATIONS = (0.5, 0.8, 1.0, 1.3, 1.6, 1.9)

#: ``tolerance`` for the relaxation sweep.  Not tighter: the norm is
#: relative, the fixed point is ~10, and float32 resolves ~1e-7 of it,
#: so below ~1e-4 the residual *ratio* the estimate rests on is reading
#: round-off and the comparison stops measuring the criterion.
_RELAXATION_TOLERANCE = 1e-3


@pytest.mark.parametrize("relaxation", _RELAXATIONS)
def test_the_estimate_is_invariant_to_the_relaxation_factor(relaxation):
    """Over-relaxation moves the iterate further, not the estimate less.

    ``x_{k+1} = x_k + omega * (F(x_k) - x_k)``, so the step the iterate
    takes is ``omega`` times the residual that is measured.  Summing
    residuals instead of steps made the reported distance short by
    exactly ``omega``: the audit measured ``est/true`` at 0.68 for
    ``omega=1.5`` and 0.51 for ``omega=1.95``, with ``converged=True``
    and ``ratio_usable=True``.  The estimate describes a distance, and a
    distance does not depend on the knob used to travel it.
    """
    gain = 0.95
    fixed_point = (1.0 / (1.0 - gain * gain), gain / (1.0 - gain * gain))
    gm = _slow_cycle(
        gain, max_iterations=400, tolerance=_RELAXATION_TOLERANCE,
        acceleration="fixed", relaxation=relaxation,
    )
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _relative_distance(_ab(gm), fixed_point)

    assert d["ratio_usable"] and d["converged"], d
    ratio = d["error_estimate"] / distance
    assert 0.9 <= ratio <= 1.15, (
        f"relaxation={relaxation}: reported {d['error_estimate']:.4e} for a "
        f"true distance of {distance:.4e} (ratio {ratio:.4f}).  A ratio "
        f"near 1/omega means the geometric series is summing residuals "
        f"rather than the steps the iterate takes."
    )


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


#: The audit's case.  The slow mode carries 1e-5 per pass but is
#: amplified 1000x, so it owns 1.25e-2 of the answer; the fast mode
#: carries 1.0 per pass and is amplified 1.25x.  The *step* is the fast
#: mode's until 0.2**k drops below 1e-5, which is about seven passes
#: after the criterion is met.
_TWO_MODE_RHO = (0.999, 0.2)
_TWO_MODE_C = (1e-5, 1.0)


def _two_mode_group(**group_kw):
    gm = GraphManager()
    gm.add_node(_TwoMode("a", _TWO_MODE_RHO, _TWO_MODE_C))
    gm.add_node(_TwoMode("b", (1.0, 1.0), (0.0, 0.0)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


def _two_mode_distance(gm):
    """Distance to the analytic fixed point, in the group's own norm."""
    exact = [c / (1.0 - r) for r, c in zip(_TWO_MODE_RHO, _TWO_MODE_C)]
    total = 0.0
    for node in ("a", "b"):
        got = [float(v) for v in gm.get_node_state(node)["x"]]
        ref = max(max(abs(g) for g in got), max(abs(e) for e in exact))
        if ref == 0.0:
            continue
        total += sum(((g - e) / ref) ** 2 for g, e in zip(got, exact))
    return total ** 0.5


@pytest.mark.xfail(strict=True, reason=(
    "Recorded, not accepted: `rho` is read from the residual sequence, "
    "which reports the mode dominating the *step*, and the mode "
    "dominating the *remaining error* can be a different and much "
    "slower one.  On (0.999, 0.2) at tolerance=1e-4 the estimate is "
    "9.19e-05 against a true distance of 1.12e-02 -- 122x -- with "
    "ratio_usable=True and converged=True.  No test on the residual "
    "sequence separates this from a genuine single-mode decay at 0.2: "
    "for the first several passes the two sequences are identical, so "
    "the two-step sqrt guard reads the same fast rate.  A real fix "
    "needs the spectrum (e.g. a power iteration on dF/dx, available "
    "only under solver='ift') and is post-0.4.0 work.  Flipping this "
    "to a pass means the estimate became a bound: update "
    "benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md and "
    "the caveat on _fixed_point_while."
))
def test_the_estimate_is_never_smaller_than_the_distance_it_estimates():
    """The property the field's name claims, on a two-mode contraction.

    This is the statement ``error_estimate`` would have to satisfy to
    be a bound.  It does not, and the gap is not small.
    """
    gm = _two_mode_group(max_iterations=60, tolerance=1e-4)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _two_mode_distance(gm)
    assert d["ratio_usable"] and d["converged"], d
    assert d["error_estimate"] >= distance, (
        f"reported {d['error_estimate']:.4e} for a true distance of "
        f"{distance:.4e} ({distance / d['error_estimate']:.0f}x)"
    )


def test_a_hidden_slow_mode_is_the_recorded_size_and_is_not_flagged():
    """The same case as a measurement, so the memo's number is pinned.

    The strict xfail above says the bound fails; this says *by how
    much* and that nothing in the report warns.  Kept separate so a
    partial improvement that shrinks 122x to 3x is visible here rather
    than silently still failing there.
    """
    gm = _two_mode_group(max_iterations=60, tolerance=1e-4)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _two_mode_distance(gm)
    understatement = distance / d["error_estimate"]
    assert d["ratio_usable"] is True
    assert d["converged"] is True
    assert understatement > 50.0, (
        f"the two-mode understatement is now {understatement:.0f}x, not the "
        f"~122x recorded in ERROR_BOUND_DECISION.md -- if the estimate "
        f"improved, update the memo and the xfail above"
    )


# ---------------------------------------------------------------------------
# The 0.4.0 field names
#
# ``bound_valid`` named all four conditions the estimate rests on while
# checking the fourth, and ``gradient_error_bound`` called a number a
# bound that is numerically ``error_estimate``.  Both moved; both are
# readable through 0.4.x and warn.  These tests are the only place in
# the repository that may name the old keys -- ``filterwarnings =
# ["error"]`` in pyproject.toml turns any other read into a failure, so
# the deprecation enforces itself across the suite.
# ---------------------------------------------------------------------------

#: old key -> new key, and what each is expected to be on a contracting
#: group, so that an alias wired to the wrong target is caught by value
#: and not only by "it returned something".
_RENAMES = {
    "bound_valid": "ratio_usable",
    "gradient_error_bound": "gradient_error_estimate",
}


def test_the_diagnostics_report_the_0_4_0_field_names():
    """The report's own keys are the new names, and only those.

    Pinned as an exact set rather than a membership check: the old
    names must be absent from ``keys()`` so that ``dict(diag)``, a JSON
    dump and any recorded artefact carry a name that still exists in
    0.5.0.
    """
    gm = _contracting_graph()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert set(d) == {
        "iterations", "residual", "amplification", "error_estimate",
        "ratio_usable", "gradient_error_estimate", "converged",
    }
    # ``dict()`` copies through the real items, not the aliases.
    assert set(dict(d)) == set(d)
    assert not set(_RENAMES) & set(dict(d).keys())


@pytest.mark.parametrize("old,new", sorted(_RENAMES.items()))
def test_the_pre_0_4_0_field_name_still_reads_and_warns(old, new):
    """Reading the old key warns and returns the new key's value.

    Every read path a caller has: subscript, ``get`` (which on a plain
    ``dict`` subclass would bypass ``__getitem__`` entirely and silently
    miss the key) and ``in``.
    """
    gm = _contracting_graph()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    expected = d[new]                       # the new name does not warn

    with pytest.warns(DeprecationWarning, match=new):
        assert d[old] == expected
    with pytest.warns(DeprecationWarning, match=new):
        assert d.get(old) == expected
    with pytest.warns(DeprecationWarning, match=new):
        assert old in d


def test_the_deprecated_names_are_wired_to_their_own_replacements():
    """Each alias resolves to *its* field, not merely to some field.

    ``bound_valid`` is a bool and ``gradient_error_estimate`` a float,
    so a crossed mapping would pass a loose "returns something" check.
    The group is chosen so the two differ in value as well as in type.
    """
    gm = _contracting_graph()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    with pytest.warns(DeprecationWarning):
        assert d["bound_valid"] is d["ratio_usable"] is True
    with pytest.warns(DeprecationWarning):
        assert d["gradient_error_bound"] == pytest.approx(
            d["gradient_error_estimate"]
        )
    assert d["gradient_error_estimate"] != d["ratio_usable"]


def test_an_unknown_key_is_a_keyerror_and_does_not_warn():
    """The alias layer intercepts exactly two names.

    Without this, a mapping that warned on any miss -- or that swallowed
    one into a ``None`` -- would satisfy every test above.
    """
    gm = _contracting_graph()
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning fails the test
        with pytest.raises(KeyError):
            d["no_such_field"]
        assert d.get("no_such_field") is None
        assert "no_such_field" not in d


def test_the_deprecated_names_do_not_survive_a_group_without_the_new_one():
    """An alias never invents a value the report does not carry.

    ``_resolve`` forwards only when the replacement is present, so a
    diagnostics mapping that is missing the field (a hand-built one, or
    a future report that drops it) raises rather than warning about a
    key it cannot answer.
    """
    from maddening.core.graph_manager import _CouplingDiagnostics

    partial = _CouplingDiagnostics({"iterations": 3, "residual": 1e-6})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError):
            partial["bound_valid"]
        assert partial.get("gradient_error_bound") is None
        assert "bound_valid" not in partial
