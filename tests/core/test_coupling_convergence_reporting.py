"""What a coupling group is allowed to claim about its own convergence.

Three invariants live here.

*Aitken needs a second opinion before it stops.*  It re-derives a
scalar relaxation factor from each pair of residuals and clips it to
[0.01, 2.0].  When its single-dominant-mode assumption fails the
factor saturates alternately at both bounds and the residual sequence
goes non-monotone, dipping orders of magnitude below its own trend for
a single pass while the iterate has barely moved.  Stopping there
returns a state far from the fixed point -- and since the dip
undershoots any plausible threshold, tightening ``tolerance`` does not
move the exit, so ``tolerance`` stops controlling accuracy at all.
Aitken must therefore meet the threshold on two *consecutive* passes
before the loop may stop.  Nothing else pays for the guard: ``none``
and ``fixed`` advance by one constant linear operator, and IQN's
least-squares step is not a clipped scalar and has never been measured
dipping.

*The residual is a statement about the state you were handed.*  Every
pass measures the iterate it starts from, so the loop's last
measurement lags the state it returns by one update.  On an exit that
met the criterion the lag is harmless (the measurement is at or below
the threshold and the returned state is one further update along); on
an exit at ``max_iterations`` it is the whole question, because an
Aitken step routinely arrives on the pass that had no successor.  So
that exit -- and only that one -- pays one more evaluation of ``F`` and
reports the returned state's own residual.  Both coupling solvers
follow the same rule, so ``solver`` is invisible in
``coupling_diagnostics()``, and the extra measurement doubles as the
guard's second opinion: a dip springs back exactly there.

*A cap of one is a report, not an exemption.*  ``max_iterations=1``
means "one staggered pass"; it is a legitimate setting, and it has to
say what that pass left behind rather than inheriting the zeros
``compile()`` seeds the diagnostics with.  It is also the one cap that
never spends a second evaluation on measuring: asking for one pass has
to cost one pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager, _fixed_point_while
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.spring import SpringDamperNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scripted_loop(
    acceleration, residuals, *, threshold=1e-2, max_iter=30, first_res=None,
):
    """Run ``_fixed_point_while`` against a *scripted* residual sequence.

    ``step_pure`` returns ``(F(x), residual)`` as two independent
    values, so the residual the loop tests can be dictated outright
    instead of being coaxed out of a physical fixture.  ``x[1]`` is a
    pass counter: ``sub_idx=(0,)`` restricts the acceleration to
    ``x[0]``, and the loop gives every non-accelerated entry the raw
    ``F(x)`` value, so the counter increments exactly once per pass
    whatever the acceleration does to ``x[0]``.

    The counter also makes the *reported* residual scriptable: when
    the loop leaves at the cap it measures the state it is about to
    return, and that measurement reads the next entry of the schedule
    -- which is exactly the "does the dip spring back?" question the
    two-pass guard is asking.

    ``first_res`` is the residual of the pass before the loop, which
    seeds the two-consecutive-passes streak.  The default is above any
    threshold used here: "the run did not start out converged".

    Returns ``(n_iters, final_res)``.
    """
    x0 = jnp.asarray([1.0, 0.0])
    schedule = jnp.asarray(residuals, dtype=x0.dtype)
    last = schedule.shape[0] - 1
    seed = jnp.asarray(jnp.inf if first_res is None else first_res, x0.dtype)

    def step_pure(x):
        k = jnp.clip(x[1].astype(jnp.int32), 0, last)
        return jnp.stack([x[0] * 0.5, x[1] + 1.0]), schedule[k]

    empty = jnp.zeros((1, 4), dtype=x0.dtype)
    accel_init = (
        (empty, empty) if acceleration in ("iqn-ils", "iqn-imvj") else ()
    )
    _x, n_iters, final_res, _vw = _fixed_point_while(
        step_pure, x0, (), accel_init, seed, threshold, max_iter,
        acceleration, 1.0, 0, (0,),
    )
    return int(n_iters), float(final_res)


#: Two plateaus above the threshold, each ending in one pass that dips
#: two decades below it, and only then a genuine sub-threshold pair.
#: A criterion that fires on one value stops on pass 3; one that wants
#: two in a row stops on pass 7.
_DIP_SCHEDULE = [1.0, 1.0, 1e-3, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]


def _coupled_springs(dt=0.01, k=200.0, c=0.5, pos_a=0.0, pos_b=3.0):
    """Two spring-damper nodes, each the other's anchor (a 2-cycle)."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring_a", timestep=dt, stiffness=k,
                                 damping=c, mass=1.0, rest_length=1.0,
                                 initial_position=pos_a))
    gm.add_node(SpringDamperNode(name="spring_b", timestep=dt, stiffness=k,
                                 damping=c, mass=1.0, rest_length=1.0,
                                 initial_position=pos_b))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    return gm


def _group_l2(before, after, names):
    """L2 norm of the float-field change across ``names`` -- the same
    quantity ``coupling_residual_l2`` computes, recomputed here from
    the public state accessor."""
    total = 0.0
    for nn in names:
        for fld, new in after[nn].items():
            if not jnp.issubdtype(jnp.asarray(new).dtype, jnp.floating):
                continue
            diff = jnp.asarray(new) - jnp.asarray(before[nn][fld])
            total += float(jnp.sum(diff ** 2))
    return total ** 0.5


# ---------------------------------------------------------------------------
# An analytic fixed point
#
# The claims in this module are about *which state* a residual
# describes, so the fixture has to be one where every iterate and every
# residual can be written down.  ``_Affine`` is a pure map -- its next
# state depends only on its boundary input, not on its own state or on
# ``dt`` -- so the group's coupling iteration is the plain linear
# fixed point ``a = b/2 + 1``, ``b = a/2`` with the exact solution
# ``a = 4/3``, ``b = 2/3``, and ``_affine_pass`` below reproduces the
# graph's Gauss-Seidel sweep exactly.
# ---------------------------------------------------------------------------

_FIXED_POINT = (4.0 / 3.0, 2.0 / 3.0)


class _Affine(SimulationNode):
    """``x_new = gain * u + bias``: one node of a linear fixed point."""

    def __init__(self, name, gain, bias, x0=0.0):
        super().__init__(name=name, timestep=1.0, gain=gain, bias=bias,
                         x0=x0)

    def initial_state(self):
        return {"x": jnp.asarray(self.params["x0"], jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.asarray(self.params["gain"]) * boundary_inputs["u"]
                + jnp.asarray(self.params["bias"])}


def _affine_graph(x0a=0.0, x0b=0.0, **group_kw):
    gm = GraphManager()
    gm.add_node(_Affine("a", gain=0.5, bias=1.0, x0=x0a))
    gm.add_node(_Affine("b", gain=0.5, bias=0.0, x0=x0b))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _affine_pass(a, b):
    """One coupling pass of :func:`_affine_graph`, in the graph's order."""
    a_new = 0.5 * b + 1.0
    b_new = 0.5 * a_new
    return a_new, b_new


def _affine_residual(a, b):
    """``||F(x) - x||`` for the state ``(a, b)`` -- the group's L2 norm."""
    a_new, b_new = _affine_pass(a, b)
    return ((a_new - a) ** 2 + (b_new - b) ** 2) ** 0.5


def _affine_state(gm):
    return (float(gm.get_node_state("a")["x"]),
            float(gm.get_node_state("b")["x"]))


# ---------------------------------------------------------------------------
# CF-01 -- the stopping criterion under Aitken
# ---------------------------------------------------------------------------

def test_aitken_needs_two_consecutive_passes_to_stop():
    """A lone sub-threshold residual must not end the iteration.

    Aitken re-derives a clipped scalar relaxation factor from each pair
    of residuals.  When its single-dominant-mode assumption fails the
    factor saturates alternately at both clip bounds and the residual
    collapses for one pass and springs back on the next, while the
    iterate has barely moved.  One value at or below the threshold is
    therefore not evidence of arrival; the loop must see two in a row.
    """
    n_iters, final_res = _scripted_loop("aitken", _DIP_SCHEDULE)
    assert n_iters == 7, (
        f"aitken stopped after {n_iters} passes; the schedule's lone "
        "dips are at passes 3 and 6 and the first genuine pair ends at "
        "pass 7"
    )
    assert final_res == pytest.approx(1e-3)


@pytest.mark.parametrize(
    "acceleration", ["none", "fixed", "iqn-ils", "iqn-imvj"]
)
def test_other_accelerations_stop_on_the_first_sub_threshold_pass(
    acceleration,
):
    """The second opinion is scoped to ``_TWO_PASS_EXIT``.

    ``none`` and ``fixed`` advance by one constant linear operator, so
    their residual sequence is asymptotically monotone.  IQN's step is
    a least-squares solve over an accumulating secant basis rather
    than a clipped scalar, it converges superlinearly in 2-4 passes
    across every sweep fixture, and no dip has been measured in one --
    a mandatory extra pass would cost it 30-50% of its budget for no
    measured gain.  Charging any of them would be a silent slowdown.
    """
    n_iters, final_res = _scripted_loop(acceleration, _DIP_SCHEDULE)
    assert n_iters == 3
    assert final_res == pytest.approx(1e-3)


def test_aitken_at_the_cap_does_not_report_a_lone_dip_as_convergence():
    """A cap landing on a dip must not be reported as convergence.

    With a residual that alternates above and below the threshold the
    pair never arrives, the loop exhausts ``max_iterations``, and the
    last measurement happens to be a dip.  Because the criterion was
    not met, the reported residual is a fresh measurement of the state
    being returned -- and a dip is by definition the value that springs
    back on the next pass, so that measurement is above the threshold
    and the ``residual <= threshold`` flag downstream stays False.

    This is the same guarantee as reporting the worse of the last two
    residuals, without the cost of that rule: the number reported
    belongs to the state the caller is handed, not to an iterate two
    updates behind it.
    """
    never_paired = [1.0, 1e-3] * 8
    n_iters, final_res = _scripted_loop(
        "aitken", never_paired, max_iter=9,
    )
    assert n_iters == 8, "the loop runs at most max_iter - 1 body passes"
    assert final_res > 1e-2, (
        f"reported {final_res}: the last pass dipped to 1e-3, and the "
        "measurement of the state that dip produced springs back to 1.0"
    )


# ---------------------------------------------------------------------------
# CF-02 -- what ``max_iterations=1`` reports
# ---------------------------------------------------------------------------

def test_single_pass_group_reports_the_residual_it_measured():
    """``max_iterations=1`` must report its own pass, not seeded zeros.

    The cap-1 branch returns before the fixed-point loop, so it used to
    leave the ``_meta`` entries at the ``iterations=0, residual=0.0``
    that ``compile()`` seeds, and ``coupling_diagnostics()`` read
    ``0.0 <= tolerance`` as ``converged=True`` whatever the state.
    """
    gm = _coupled_springs()
    gm.add_coupling_group(["spring_a", "spring_b"],
                          max_iterations=1, tolerance=1e-6)
    gm.compile()
    names = ["spring_a", "spring_b"]
    before = {n: dict(gm.get_node_state(n)) for n in names}
    gm.step()
    after = {n: dict(gm.get_node_state(n)) for n in names}

    diag = gm.coupling_diagnostics()["spring_a+spring_b"]
    expected = _group_l2(before, after, names)
    assert expected > 1e-6, "the fixture must actually move, or this is vacuous"
    assert diag["residual"] == pytest.approx(expected, rel=1e-4)
    assert diag["iterations"] == 1
    assert diag["converged"] is False


def test_single_pass_group_still_reports_convergence_when_it_converges():
    """``converged`` is measured at a cap of one, not hard-wired False."""
    gm = _coupled_springs()
    gm.add_coupling_group(["spring_a", "spring_b"],
                          max_iterations=1, tolerance=1e3)
    gm.compile()
    gm.step()
    diag = gm.coupling_diagnostics()["spring_a+spring_b"]
    assert diag["converged"] is True


def test_single_pass_group_honours_strict_convergence():
    """``strict_convergence`` must be able to fire at a cap of one.

    The guard lives on the fixed-point path, which a cap of one never
    reaches, so the one setting most likely to leave a group
    unconverged was also the one setting the guard could not see.
    """
    gm = _coupled_springs()
    gm.add_coupling_group(["spring_a", "spring_b"],
                          max_iterations=1, tolerance=1e-6,
                          strict_convergence=True)
    gm.compile()
    with pytest.raises(Exception, match="without converging"):
        jax.block_until_ready(gm.step())


def test_one_iteration_profile_variant_reports_diagnostics():
    """The profiler's coupling-overhead measurement needs the report.

    ``profile_graph(measure_coupling=True)`` recompiles with every
    group capped at one iteration and subtracts that step's time, so
    the difference is "the extra iterations only" *provided* the
    one-iteration step still does everything else the real step does,
    diagnostics included.  While the cap-1 branch skipped the residual
    and the ``_meta`` write, their cost was being billed to coupling
    iteration overhead.
    """
    from maddening.core.simulation.profiler import _one_iteration_variant

    gm = _coupled_springs()
    gm.add_coupling_group(["spring_a", "spring_b"],
                          max_iterations=12, tolerance=1e-8)
    gm.compile()
    gm.step()

    with _one_iteration_variant(gm):
        gm.step()
        diag = gm.coupling_diagnostics()["spring_a+spring_b"]
        assert diag["iterations"] == 1
        assert diag["residual"] > 0.0


# ---------------------------------------------------------------------------
# What the reported residual describes, and who agrees about it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("acceleration", ["none", "aitken", "iqn-ils"])
@pytest.mark.parametrize("cap", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("tolerance", [0.5, 0.05, 1e-3])
def test_both_coupling_solvers_report_the_same_residual_and_converged_flag(
    acceleration, cap, tolerance,
):
    """``solver`` must not be visible in ``coupling_diagnostics()``.

    ``solver="ift"`` and the legacy ``solver="fori"`` run the same
    passes on the same iterates with the same stopping rule; which of
    the two a user picked is an implementation choice, and
    ``converged`` is documented without reference to it.  A graph
    migrated from v0.3's ``fori`` to v0.4's default must not start
    reporting a different verdict on the same physics -- and
    ``strict_convergence`` exists only on the ift path, so a
    disagreement turns a clean run into a runtime error.

    ``iterations`` is deliberately not compared: the two paths count
    passes differently (one counts body iterations, the other every
    pass) and that difference predates this file.
    """
    if acceleration == "iqn-ils" and cap == 1:
        # A cap of one returns before any accelerator is constructed,
        # so there is nothing solver-specific left to compare; the
        # other accelerations cover the same branch.
        pytest.skip("cap 1 returns before the accelerator is built")
    seen = {}
    for solver in ("ift", "fori"):
        gm = _affine_graph(acceleration=acceleration, solver=solver,
                           max_iterations=cap, tolerance=tolerance)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        seen[solver] = (d["residual"], d["converged"])
    assert seen["ift"][1] == seen["fori"][1], (
        f"converged differs by solver: {seen}"
    )
    assert seen["ift"][0] == pytest.approx(seen["fori"][0], abs=1e-7), (
        f"residual differs by solver: {seen}"
    )


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_group_that_reached_its_fixed_point_reports_it_converged(solver):
    """Arriving on the last pass is arrival, not failure.

    Aitken's whole purpose is a large correction on the pass that
    happens to be the last one the cap allows.  The residual measured
    *during* that pass describes the iterate the correction started
    from, which is far from the fixed point; reporting it leaves a
    group sitting on its fixed point to float32 round-off claiming
    ``converged=False`` with a residual four decades too large.
    """
    gm = _affine_graph(acceleration="aitken", solver=solver,
                       max_iterations=3, tolerance=1e-3)
    gm.step()
    a, b = _affine_state(gm)
    assert abs(a - _FIXED_POINT[0]) < 1e-6, (
        "fixture premise: three passes of Aitken land on the fixed point"
    )
    d = gm.coupling_diagnostics()["a+b"]
    assert d["residual"] == pytest.approx(_affine_residual(a, b), abs=1e-7)
    assert d["residual"] < 1e-3
    assert d["converged"] is True


def test_a_group_at_its_fixed_point_does_not_raise_under_strict_convergence():
    """A simulation that has converged must not error.

    ``strict_convergence`` exists to stop a training loop taking an
    IFT gradient at a point that is not a fixed point.  The gradient is
    taken at the state the solver *returns*, so the guard has to be
    driven by that state's residual; driven by an earlier iterate's it
    raises on a graph that is sitting on its fixed point.
    """
    gm = _affine_graph(acceleration="aitken", max_iterations=3,
                       tolerance=1e-3, strict_convergence=True)
    jax.block_until_ready(gm.step())
    a, _b = _affine_state(gm)
    assert a == pytest.approx(_FIXED_POINT[0], abs=1e-6)
    assert gm.coupling_diagnostics()["a+b"]["converged"] is True


@pytest.mark.parametrize("acceleration", ["none", "aitken", "iqn-ils"])
@pytest.mark.parametrize("cap", [1, 2, 3, 4])
@pytest.mark.parametrize("tolerance", [0.5, 0.05])
def test_converged_survives_recomputation_on_a_contractive_group(
    acceleration, cap, tolerance,
):
    """``converged=True`` names a state within tolerance -- here.

    Whatever the stopping rule did, ``converged=True`` ought to survive
    the caller recomputing ``||F(x) - x||`` on the state they got back:
    that is what a step-size controller, a CI assertion or
    ``strict_convergence`` reads it as.

    **This test does not establish that in general, and its name used
    to claim it did.**  ``_affine_graph`` is contractive, so its
    residual sequence is monotone and the one-update lag on a criterion
    exit can only make the reported number conservative.  On a
    non-normal group the sequence is not monotone and the guarantee
    fails -- see
    ``test_converged_is_proof_the_returned_state_is_within_tolerance``,
    the strict xfail that pins it.  Keep this one for the contractive case
    it does cover; do not read it as the general contract.
    """
    gm = _affine_graph(acceleration=acceleration, max_iterations=cap,
                       tolerance=tolerance)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    returned = _affine_residual(*_affine_state(gm))
    if d["converged"]:
        assert returned <= tolerance, (
            f"reported converged at {d['residual']} but the returned "
            f"state's own residual is {returned} > {tolerance}"
        )


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_the_aitken_guard_is_answerable_at_every_cap(cap):
    """The guard must not be switched off where it cannot buy a pass.

    At ``max_iterations=2`` the loop only gets to measure one residual
    inside itself, so "two consecutive passes" used to be waived there
    -- the one cap where a user is most likely to read ``converged``
    and believe it.  It is not waived now: the streak is seeded with
    the residual of the pass before the loop, and when the pair still
    does not arrive the group measures what it returns instead of
    trusting the single value.

    Concretely, with ``tolerance=0.5`` the pass before the loop is at
    1.118 and the first pass inside it at 0.2795.  ``none`` stops on
    that one value and reports it.  Aitken may not, so at caps 2 and 3
    it reports the measured residual of the state it returns, which is
    a smaller, different number -- the guard is doing something.
    """
    tol = 0.5
    verdicts = {}
    for acceleration in ("none", "aitken"):
        gm = _affine_graph(acceleration=acceleration, max_iterations=cap,
                           tolerance=tol)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        verdicts[acceleration] = (d["residual"], d["converged"],
                                  _affine_residual(*_affine_state(gm)))

    for acceleration, (res, conv, returned) in verdicts.items():
        if cap > 1:
            assert res == pytest.approx(returned, abs=1e-7) or res <= tol, (
                f"{acceleration} at cap {cap} reported {res}, which is "
                f"neither the returned state's residual ({returned}) nor "
                "a value its criterion proved"
            )
        if conv:
            assert returned <= tol

    if cap == 1:
        # One pass, no accelerated iterate exists and no accelerator
        # runs, so the two paths must agree exactly -- and neither has
        # met the threshold (see ``test_a_cap_of_one_costs_one_pass``
        # for what that single measurement is).
        assert verdicts["none"] == verdicts["aitken"]
        assert verdicts["aitken"][1] is False
    else:
        assert verdicts["none"][0] != pytest.approx(verdicts["aitken"][0]), (
            "Aitken inherited `none`'s single-measurement verdict: the "
            "guard is inactive at this cap"
        )


def test_a_cap_of_one_costs_one_pass():
    """Asking for one staggered pass has to cost one evaluation.

    Every other cap pays a second evaluation of ``F`` when it is about
    to report a failure, so that the residual describes the state
    returned.  ``max_iterations=1`` is the one setting that is a
    request about *cost*, so it keeps reporting the residual its single
    pass measured -- the distance the pass moved -- and the profiler's
    one-iteration variant stays a one-iteration step.
    """
    gm = _affine_graph(acceleration="none", max_iterations=1, tolerance=1e-9)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert _affine_state(gm) == (pytest.approx(1.0), pytest.approx(0.5))
    assert d["residual"] == pytest.approx(_affine_residual(0.0, 0.0), abs=1e-6)
    assert d["converged"] is False


# ---------------------------------------------------------------------------
# What ``strict_convergence`` and ``converged`` are allowed to miss
#
# Both are read as "is the state this step returned a fixed point?".
# The two fixtures below are the two ways that question can be
# answered wrongly: a residual that is not a number, and an early exit
# whose measurement belongs to an iterate the caller never sees.
# ---------------------------------------------------------------------------


def _blowup_graph(cap, **group_kw):
    """A group that overflows float32 within a couple of passes.

    ``b``'s gain sends the iterate past ``float32`` range on the second
    pass, so the *returned* state has an ``inf`` in it while the pass
    before it was still finite.  Measuring the returned state is then
    ``inf - inf``: a NaN, which is False against every comparison.
    """
    gm = GraphManager()
    gm.add_node(_Affine("a", gain=10.0, bias=1.0))
    gm.add_node(_Affine("b", gain=1e30, bias=0.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, solver="ift", acceleration="none",
              max_iterations=cap, tolerance=1e-6)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("cap", [2, 3, 4])
def test_a_residual_that_is_not_a_number_raises_under_strict_convergence(cap):
    """NaN is the one residual that certainly is not convergence.

    ``strict_convergence`` exists to stop a training loop taking an IFT
    gradient at a state that is not a fixed point.  Written as
    ``residual > threshold`` the guard is silent on NaN, because NaN is
    False against ``>`` just as it is against ``<=`` -- so a group that
    diverged until it overflowed returned an ``inf`` state with no
    error at all, while ``coupling_diagnostics()`` (which asks
    ``residual <= threshold``) called the same run ``converged=False``.
    The guard has to ask the same question the flag asks.

    The cap matters: measuring the *returned* state, which is what this
    module's other invariants require, is exactly what turns the
    reportable ``inf`` of the pass before the cap into a NaN.  At
    ``cap=2`` the in-loop measurement is still ``inf``, so this is also
    the case where the guard used to fire and stopped.
    """
    gm = _blowup_graph(cap, strict_convergence=True)
    with pytest.raises(Exception, match="without converging"):
        jax.block_until_ready(gm.step())


@pytest.mark.parametrize("cap", [2, 3, 4])
def test_a_diverged_group_is_never_reported_as_converged(cap):
    """Without the guard the same run must still report the failure."""
    gm = _blowup_graph(cap, strict_convergence=False)
    jax.block_until_ready(gm.step())
    assert gm.coupling_diagnostics()["a+b"]["converged"] is False


def _amplifying_graph(solver, cap, tolerance):
    """A convergent group whose residual is not monotone.

    Jacobi on ``a = 10 b + 1``, ``b = 0.05 a`` contracts by ``0.5``
    every two passes, but the iteration matrix is strongly non-normal,
    so a single pass multiplies the residual by ten and the next
    divides it by twenty.  From ``(12, 1.1)`` the measured sequence is
    ``0.5, 5, 0.25, 2.5, 0.125, ...``: every second measurement is
    below ``tolerance=1.0`` and the state it produces is well above it.
    Nothing here is scripted -- it is one linear graph, run normally.
    """
    gm = GraphManager()
    gm.add_node(_Affine("a", gain=10.0, bias=1.0, x0=12.0))
    gm.add_node(_Affine("b", gain=0.05, bias=0.0, x0=1.1))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(
        ["a", "b"], diagnostics=True, solver=solver, iteration_mode="jacobi",
        acceleration="none", max_iterations=cap, tolerance=tolerance,
    )
    gm.compile()
    return gm


def _amplifying_residual(a, b):
    a_new, b_new = 10.0 * b + 1.0, 0.05 * a
    return ((a_new - a) ** 2 + (b_new - b) ** 2) ** 0.5


@pytest.mark.parametrize("cap", [3, 4, 5])
def test_the_fori_solver_never_claims_a_state_it_did_not_measure(cap):
    """``fori`` freezes on the iterate whose residual passed.

    This is the companion of the ``xfail`` below: the same graph, the
    same stopping pass, and here ``converged=True`` does survive the
    caller recomputing the residual, because the state handed back is
    the one that was measured.
    """
    gm = _amplifying_graph("fori", cap, 1.0)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    a = float(gm.get_node_state("a")["x"])
    b = float(gm.get_node_state("b")["x"])
    assert d["converged"] is True
    assert _amplifying_residual(a, b) == pytest.approx(d["residual"], abs=1e-5)
    assert _amplifying_residual(a, b) <= 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, pre-existing (identical on release/0.4.0): the ift loop "
        "measures the iterate each pass starts from, and on an exit that "
        "met the criterion it reports that measurement while returning "
        "one further update.  The re-measurement added for the cap exit "
        "is gated on the criterion, so this exit keeps the lag and "
        "converged=True does not imply the returned state is within "
        "tolerance.  Closing it means either always re-measuring (one "
        "extra F per converged group per step) or returning the iterate "
        "that passed, as fori does (which also changes the state ift "
        "returns); both are design calls, not an audit fix."
    ),
)
@pytest.mark.parametrize("cap", [3, 4, 5])
def test_converged_is_proof_the_returned_state_is_within_tolerance(cap):
    """``converged=True`` has to survive the caller remeasuring.

    A step-size controller, a CI assertion and ``strict_convergence``
    all read the flag as a statement about the state they were handed.
    On :func:`_amplifying_graph` the ift solver stops on the pass that
    measured 0.25, hands back the state one update later, and that
    state's own residual is 2.5 -- two and a half times the tolerance
    it just reported meeting.
    """
    gm = _amplifying_graph("ift", cap, 1.0)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    a = float(gm.get_node_state("a")["x"])
    b = float(gm.get_node_state("b")["x"])
    if d["converged"]:
        assert _amplifying_residual(a, b) <= 1.0, (
            f"reported converged at {d['residual']} but the returned "
            f"state's own residual is {_amplifying_residual(a, b)}"
        )
