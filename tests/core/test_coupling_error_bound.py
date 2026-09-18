"""What ``converged=True`` is allowed to mean about the fixed point.

Until 0.4.0 it meant *the last pass was small*: the group compared
``||F(x) - x||`` against its threshold and said nothing about the
distance to the fixed point, which is larger by the amplification
``1/(1 - rho)`` of the slowest mode (``MADD-ANO-005``).  Two changes
close that, and they are tested here together because neither is worth
much without the other:

*The criterion is an error bound.*  ``rho`` costs nothing -- it is
``r_k / r_{k-1}``, two numbers both solvers already carry -- so the
threshold is applied to ``r_k / (1 - rho)``.  The estimate is never
smaller than ``r_k``, so the criterion is never looser than the one it
replaces, and a group that used to stop on a single small step now has
to earn it.  Where the ratio cannot be trusted the raw residual test
stands in and ``bound_valid`` says so, which is the honest answer and
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
    kw = dict(diagnostics=True, solver=solver, max_iterations=20,
              tolerance=1e-4)
    kw.update(group_kw)
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
    assert d["bound_valid"] is True
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
    assert d["bound_valid"] is False
    assert jnp.isnan(d["amplification"])
    assert d["error_estimate"] == pytest.approx(d["residual"])
    assert d["gradient_error_bound"] == float("inf"), (
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
    kw = dict(convergence_norm=norm, tolerance=1e-4, atol=1e-12, rtol=1e-4)
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


# ---------------------------------------------------------------------------
# The gradient-trust bound
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
    assert d["bound_valid"] is True, "fixture premise: a measured contraction"
    assert d["gradient_error_bound"] == pytest.approx(d["error_estimate"])
    assert abs(analytic - fd) <= max(d["gradient_error_bound"], 1e-5), (
        f"analytic {analytic} vs finite difference {fd}: the adjoint may "
        f"only be as wrong as the reported bound "
        f"({d['gradient_error_bound']})"
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
    assert a["bound_valid"] == b["bound_valid"]
    assert a["iterations"] == b["iterations"]
    assert a["residual"] == pytest.approx(b["residual"], rel=1e-5)
    assert a["error_estimate"] == pytest.approx(b["error_estimate"], rel=1e-5)
    assert _ab(ift) == pytest.approx(_ab(fori), rel=1e-6)
