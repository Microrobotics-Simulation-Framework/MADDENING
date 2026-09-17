"""What a coupling group is allowed to claim about its own convergence.

Two invariants live here.

*Extrapolating accelerations need a second opinion.*  Aitken and IQN
choose a step that annihilates whichever modes dominate the residual.
That is what makes them fast, and it is also what decouples the
residual from the error: the residual sequence goes non-monotone and
can dip orders of magnitude below its own trend for a single pass
while the iterate has barely moved.  Stopping there reports
``converged=True`` far from the fixed point -- and since the dip
undershoots any plausible threshold, tightening ``tolerance`` does not
move the exit, so ``tolerance`` stops controlling accuracy at all.
Those accelerations must therefore meet the threshold on two
*consecutive* passes; ``none`` and ``fixed`` advance by one constant
linear operator, keep the monotone-residual argument, and must not pay
for the guard.

*A cap of one is a report, not an exemption.*  ``max_iterations=1``
means "one staggered pass"; it is a legitimate setting, and it has to
say what that pass left behind rather than inheriting the zeros
``compile()`` seeds the diagnostics with.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager, _fixed_point_while
from maddening.nodes.spring import SpringDamperNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scripted_loop(acceleration, residuals, *, threshold=1e-2, max_iter=30):
    """Run ``_fixed_point_while`` against a *scripted* residual sequence.

    ``step_pure`` returns ``(F(x), residual)`` as two independent
    values, so the residual the loop tests can be dictated outright
    instead of being coaxed out of a physical fixture.  ``x[1]`` is a
    pass counter: ``sub_idx=(0,)`` restricts the acceleration to
    ``x[0]``, and the loop gives every non-accelerated entry the raw
    ``F(x)`` value, so the counter increments exactly once per pass
    whatever the acceleration does to ``x[0]``.

    Returns ``(n_iters, final_res)``.
    """
    x0 = jnp.asarray([1.0, 0.0])
    schedule = jnp.asarray(residuals, dtype=x0.dtype)
    last = schedule.shape[0] - 1

    def step_pure(x):
        k = jnp.clip(x[1].astype(jnp.int32), 0, last)
        return jnp.stack([x[0] * 0.5, x[1] + 1.0]), schedule[k]

    empty = jnp.zeros((1, 4), dtype=x0.dtype)
    accel_init = (
        (empty, empty) if acceleration in ("iqn-ils", "iqn-imvj") else ()
    )
    _x, n_iters, final_res, _vw = _fixed_point_while(
        step_pure, x0, (), accel_init, threshold, max_iter,
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
# CF-01 -- the stopping criterion under an extrapolating acceleration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("acceleration", ["aitken", "iqn-ils", "iqn-imvj"])
def test_extrapolating_acceleration_needs_two_consecutive_passes_to_stop(
    acceleration,
):
    """A lone sub-threshold residual must not end the iteration.

    Under Aitken/IQN a single value at or below the threshold is not
    evidence of arrival: the accelerator annihilates the modes that
    dominate the residual, so the residual can collapse for one pass
    and spring back on the next while the iterate is still far from
    the fixed point.  The loop must keep going until it sees two in a
    row.
    """
    n_iters, final_res = _scripted_loop(acceleration, _DIP_SCHEDULE)
    assert n_iters == 7, (
        f"{acceleration} stopped after {n_iters} passes; the schedule's "
        "lone dips are at passes 3 and 6 and the first genuine pair "
        "ends at pass 7"
    )
    assert final_res == pytest.approx(1e-3)


@pytest.mark.parametrize("acceleration", ["none", "fixed"])
def test_plain_iteration_stops_on_the_first_sub_threshold_pass(acceleration):
    """The guard is scoped to the adaptive accelerations.

    ``none`` and ``fixed`` advance by one constant linear operator, so
    their residual sequence is asymptotically monotone and one value
    at or below the threshold is enough.  Making them pay an extra
    pass would be a silent slowdown of every uncoupled-from-the-bug
    configuration.
    """
    n_iters, final_res = _scripted_loop(acceleration, _DIP_SCHEDULE)
    assert n_iters == 3
    assert final_res == pytest.approx(1e-3)


def test_aitken_at_the_cap_reports_the_pair_it_never_got():
    """A cap landing on a dip must not be reported as convergence.

    With a residual that alternates above and below the threshold the
    pair never arrives, the loop exhausts ``max_iterations``, and the
    last measurement happens to be a dip.  The reported residual is
    what the criterion tested -- the worse of the last two -- so the
    ``residual <= threshold`` flag downstream stays False.
    """
    never_paired = [1.0, 1e-3] * 8
    n_iters, final_res = _scripted_loop(
        "aitken", never_paired, max_iter=9,
    )
    assert n_iters == 8, "the loop runs at most max_iter - 1 body passes"
    assert final_res > 1e-2, (
        f"reported {final_res}: the last pass dipped to 1e-3 but the one "
        "before it was 1.0, so the criterion was never met"
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
