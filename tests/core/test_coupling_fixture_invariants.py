"""Invariants the coupling benchmark fixtures must satisfy.

These matter more than the timings in
``benchmarks/results/coupling_sweep_cpu.json``: a configuration that is
fast because it converged somewhere else, or that truncates a divergent
group at the cap and calls it a day, is a bug rather than a speed-up.

The fixtures themselves live in ``benchmarks/coupling_fixtures.py``
(next to the sweep driver that is their main consumer) and are loaded
here by path, the same way
``tests/cloud/multigpu/test_run_pod_dry_run.py`` reaches the runner
script.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import maddening
from maddening.core.coupling.group import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

_REPO_ROOT = Path(maddening.__file__).resolve().parents[2]
_BENCHMARKS = _REPO_ROOT / "benchmarks"
_FIXTURES_PY = _BENCHMARKS / "coupling_fixtures.py"
_SWEEP_PY = _BENCHMARKS / "bench_coupling_sweep.py"


def _load_by_path(module_name, path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: ``@dataclass`` resolves the defining
    # module out of ``sys.modules`` while the class body is running.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cf = _load_by_path("maddening_coupling_fixtures", _FIXTURES_PY)
#: The sweep driver is tested here too: its summary fields are what a
#: reader of ``benchmarks/results/*.json`` trusts, and two of them
#: (``fixed_point_agreement``, ``best``) are claims about the same
#: invariants this file asserts.
sweep = _load_by_path("maddening_bench_coupling_sweep", _SWEEP_PY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(built, n_steps):
    """Step *built* and return (state by node/field, per-group diagnostics)."""
    gm = built.gm
    for _ in range(n_steps):
        gm.step()
    state = {}
    for name in sorted(gm.node_names):
        for field, value in sorted(gm.get_node_state(name).items()):
            state[(name, field)] = np.asarray(value, dtype=np.float64).ravel()
    return state, gm.coupling_diagnostics()


def _run_history(built, n_steps):
    """Per-group iteration counts for every step, not just the last one."""
    gm = built.gm
    history = {key: [] for key in built.group_keys}
    for _ in range(n_steps):
        gm.step()
        diag = gm.coupling_diagnostics()
        for key in history:
            history[key].append(int(diag[key]["iterations"]))
    return history


def _flat(state, nodes=None):
    """One array, for the bitwise-determinism comparisons."""
    keys = [k for k in sorted(state) if nodes is None or k[0] in nodes]
    return np.concatenate([state[k] for k in keys])


def _group_nodes(built):
    """The nodes inside some coupling group of *built*."""
    out = set()
    for key in built.group_keys:
        out.update(key.split("+"))
    return frozenset(out)


def _field_scales(state, nodes):
    """Largest magnitude of each *field name*, across the nodes in *nodes*."""
    scales = {}
    for (node, field), arr in state.items():
        if node not in nodes:
            continue
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        scales[field] = max(scales.get(field, 0.0), m)
    return scales


def _relative_spread(a, b, nodes):
    """Worst deviation of *b* from *a* over *nodes*, against each field's scale.

    Three scalings have to be got right at once.  A single global scale
    is a velocity scale — these graphs carry velocities of order
    ``position/dt`` — so a disagreement about position hides behind it.
    Normalising each element against itself explodes whenever an
    oscillating trajectory passes through zero, and on the spring
    fixtures every field *is* a single element, so per-array
    normalisation is per-element normalisation and has the same problem.
    Taking the scale of a field name across all the nodes that carry it
    is the one that works: a ring node passing through zero is still
    measured against the ring's amplitude, while the grid's temperature
    and a scalar node's position keep their own very different scales.

    *nodes* is the coupling group, and restricting to it is the fourth
    thing that has to be right.  Every fixture carries a driver node
    outside every group, and on ``heterogeneous`` that driver's
    ``position`` is ~0.88 against the coupled probes' ~0.011 — so
    including it divided every probe disagreement by eighty, and a
    genuine 20.6% error scored 2.6e-3 against a 5e-3 threshold.  The
    driver is not part of the fixed point being compared and has no
    business setting the scale it is compared against.
    """
    scales = _field_scales(a, nodes)
    worst = 0.0
    for key, ref in a.items():
        if key[0] not in nodes:
            continue
        scale = max(scales[key[1]], 1e-12)
        worst = max(worst, float(np.max(np.abs(ref - b[key]))) / scale)
    return worst


# ---------------------------------------------------------------------------
# The registry builds
# ---------------------------------------------------------------------------


def test_every_registered_fixture_builds_and_steps():
    """Each fast fixture compiles and takes a step under both modes."""
    for name in cf.fixture_names(include_slow=False):
        spec = cf.FIXTURES[name]
        for mode in ("gauss-seidel", "jacobi"):
            built = spec.build(cf.CouplingConfig(iteration_mode=mode))
            built.gm.step()
            assert built.group_keys, f"{name} declared no coupling group"
            diag = built.gm.coupling_diagnostics()
            assert set(diag) == set(built.group_keys), (
                f"{name}: diagnostics keys {sorted(diag)} != declared "
                f"{sorted(built.group_keys)}"
            )


# ---------------------------------------------------------------------------
# Same fixed point, whatever the algorithm
# ---------------------------------------------------------------------------


#: Agreement threshold for the rows that stop on the *absolute* L2
#: tolerance.  Every fixture here is built with ``tolerance=1e-4`` on an
#: O(1) state, and 25 steps of compounding keeps the measured worst case
#: an order of magnitude inside this.
_L2_AGREEMENT = 5e-3
#: Agreement threshold for the rows that stop on the interface norm.
#: That criterion is *relative* (``atol``/``rtol``), so it stops earlier
#: than the absolute one and the trajectories drift correspondingly
#: further apart — the algorithm guide says so and the sweep records it
#: per fixture.  A separate, looser number rather than one band for both
#: is what lets the L2 rows keep the tight threshold instead of
#: inheriting the interface rows' slack.
_INTERFACE_AGREEMENT = 2.5e-2

#: Configurations that do *not* reach the common fixed point today,
#: each with the defect that explains it.  Listed, not tolerated: the
#: test below asserts every entry is still disagreeing, so fixing the
#: defect turns this file red with an instruction to delete the entry.
#: Silently widening the threshold instead would have thrown away the
#: one measurement that catches the defect.
_KNOWN_DISAGREEMENTS = {
    ("heterogeneous-2000", "jac/aitken/l2"):
        "aitken's stopping criterion is not a bound on the error when "
        "its omega estimate saturates at a clip bound, which is what "
        "jacobi does on this fixture's degenerate modes; the row "
        "reports converged=True 15-31x above its own tolerance.  Being "
        "fixed separately in the coupling library.",
}


def _fixture_build(name):
    """Builder for *name*, including the test-only reduced grid fixture."""
    if name == "heterogeneous-2000":
        # The registry's `heterogeneous` is 6e4 cells and minutes long.
        # 2 000 cells keeps the property that matters here — one
        # large-state node among small ones, four orders of magnitude of
        # scale in one group — at a couple of seconds.
        return lambda config: cf.build_heterogeneous(config, n_cells=2000)
    return cf.FIXTURES[name].build


def _assert_same_fixed_point(fixture, norms, n_steps=25):
    """Every configuration over *norms* lands on one trajectory."""
    build = _fixture_build(fixture)
    reference = None
    ref_nodes = None
    deviations = {}
    for config in cf.sweep_configs(norms):
        built = build(config)
        state, diag = _run(built, n_steps)
        assert all(d["converged"] for d in diag.values()), (
            f"{fixture} / {config.label} did not converge; the fixture is "
            f"supposed to be inside every configuration's reach"
        )
        assert np.all(np.isfinite(_flat(state))), (
            f"{fixture} / {config.label}: non-finite state")
        if reference is None:
            reference, ref_nodes = state, _group_nodes(built)
            continue
        deviations[config.label] = (
            _relative_spread(reference, state, ref_nodes), config)

    drifted, repaired = [], []
    for label, (dev, config) in sorted(deviations.items()):
        limit = (_L2_AGREEMENT if config.convergence_norm == "l2"
                 else _INTERFACE_AGREEMENT)
        known = _KNOWN_DISAGREEMENTS.get((fixture, label))
        if dev > limit and known is None:
            drifted.append(f"{label} drifted {dev:.2e} (limit {limit:.0e})")
        elif dev <= limit and known is not None:
            repaired.append(f"{label} now agrees to {dev:.2e}")
    assert not drifted, (
        f"{fixture}: these configurations left the gauss-seidel/none "
        f"trajectory after {n_steps} steps: " + "; ".join(drifted)
    )
    assert not repaired, (
        f"{fixture}: " + "; ".join(repaired) + " — the defect recorded in "
        f"_KNOWN_DISAGREEMENTS is fixed; delete the entry so the "
        f"invariant is enforced again"
    )


@pytest.mark.parametrize("fixture", ["stiff-pair-0.5", "heterogeneous-2000"])
def test_every_l2_configuration_reaches_the_same_fixed_point(fixture):
    """The fast lane of the "same fixed point" invariant.

    The full sweep below is slow-marked and the repository default is
    ``-m 'not slow'``, so for as long as it was the only lane the first
    of the five required invariants never ran.  This lane keeps the
    invariant in the default run at a few seconds: the six L2
    configurations, on one spring fixture and one grid fixture.

    The grid fixture is the point of including a second shape.  The
    slow lane's three fixtures are all springs, whose nodes share one
    scale; a fixture with four orders of magnitude between its grid and
    its scalars is where an accelerator can move a small node a long
    way and still look converged, and it is the shape that turned out
    to be hiding one.
    """
    _assert_same_fixed_point(fixture, ("l2",))


@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["stiff-pair-0.5", "chain-5", "ring-8"])
def test_every_configuration_reaches_the_same_fixed_point(fixture):
    """All 24 sweep configurations of a fixture agree on the trajectory.

    Each configuration stops at its own tolerance, so the states are not
    bitwise equal; what must hold is that they stay within a small
    multiple of that tolerance of each other after enough steps for any
    disagreement to compound.  An accelerator that converges to a
    *different* fixed point shows up here as a deviation that grows with
    the step count, not as a few ulps.

    Slow-marked because it is 24 builds and 24 compiles per fixture;
    the fast lane above covers the same invariant over a reduced grid.
    """
    _assert_same_fixed_point(fixture, ("l2", "interface"))


# ---------------------------------------------------------------------------
# The ring: Gauss-Seidel is order-dependent, Jacobi is not
# ---------------------------------------------------------------------------


_RING_N = 8
#: Few enough iterations that the group never converges within a step,
#: so what is compared is the *iterate*, which is where the schedule
#: shows up.  A converged group would agree whatever the order.
_RING_CONFIG = dict(max_iterations=3, tolerance=1e-30)


def _ring_states(mode, *, rotation=0, reverse=False, n_steps=5):
    config = cf.CouplingConfig(iteration_mode=mode, **_RING_CONFIG)
    built = cf.build_ring(_RING_N, config, rotation=rotation, reverse=reverse)
    return _run(built, n_steps)[0], _group_nodes(built)


def test_ring_schedule_follows_node_insertion_order():
    """The premise of the two tests below: rotating the build rotates the sweep."""
    base = cf.build_ring(_RING_N, cf.CouplingConfig()).gm.schedule
    rotated = cf.build_ring(
        _RING_N, cf.CouplingConfig(), rotation=3).gm.schedule
    ring = [n for n in base if n.startswith("seg")]
    ring_rot = [n for n in rotated if n.startswith("seg")]
    assert set(ring) == set(ring_rot)
    assert ring != ring_rot, (
        "rotating the insertion order left the schedule unchanged, so the "
        "order-dependence tests would pass vacuously"
    )


def test_gauss_seidel_on_a_ring_is_order_dependent():
    """A closed cycle has no natural first node, and GS picks one anyway."""
    base, nodes = _ring_states("gauss-seidel")
    rotated, _ = _ring_states("gauss-seidel", rotation=3)
    reversed_, _ = _ring_states("gauss-seidel", reverse=True)
    assert _relative_spread(base, rotated, nodes) > 1e-3, (
        "Gauss-Seidel gave the same under-converged iterate for two "
        "different sweep orders on a ring; either the schedule is not "
        "following insertion order or the ring is symmetric enough that "
        "the order cannot matter"
    )
    assert _relative_spread(base, reversed_, nodes) > 1e-3


def test_jacobi_on_a_ring_is_order_independent():
    """Jacobi reads a frozen state, so the sweep order cannot reach the result."""
    base, nodes = _ring_states("jacobi")
    rotated, _ = _ring_states("jacobi", rotation=3)
    reversed_, _ = _ring_states("jacobi", reverse=True)
    assert _relative_spread(base, rotated, nodes) < 1e-6
    assert _relative_spread(base, reversed_, nodes) < 1e-6


# ---------------------------------------------------------------------------
# Past the convergence limit
# ---------------------------------------------------------------------------


def info_cap(gm, key):
    """``max_iterations`` of the group identified by *key*."""
    for group in gm._coupling_groups:
        if "+".join(sorted(group.nodes)) == key:
            return group.max_iterations
    raise KeyError(key)


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("acceleration,relaxation", [
    ("none", 1.0), ("aitken", 1.0), ("fixed", 0.5), ("fixed", 0.8),
])
def test_divergent_group_reports_unconverged_rather_than_truncating(
        mode, acceleration, relaxation):
    """A contraction factor above one must be visible in the diagnostics.

    ``stiff-pair-1.2`` has a coupling gain of 1.2, so every *fixed-point*
    iteration diverges: the loop exhausts the cap and the residual grows.
    The failure this guards against is the group returning the last
    iterate with ``converged=True`` because the cap, rather than the
    tolerance, ended the loop.

    Under-relaxation is included because it is the textbook cure and
    does not work here: the Gauss-Seidel eigenvalue is ``+1.44``, and
    relaxing by omega maps it to ``1 - omega + 1.44*omega``, which stays
    above 1 for every ``omega > 0``.  Relaxation only rescues a
    *negative* eigenvalue outside the unit circle.
    """
    spec = cf.FIXTURES["stiff-pair-1.2"]
    built = spec.build(cf.CouplingConfig(
        iteration_mode=mode, acceleration=acceleration,
        relaxation=relaxation))
    gm = built.gm
    key = built.group_keys[0]
    cap = info_cap(gm, key)
    for _ in range(3):
        gm.step()
        info = gm.coupling_diagnostics()[key]
        assert not info["converged"], (
            f"a group with contraction factor 1.2 reported converged "
            f"(residual {info['residual']})"
        )
        assert info["iterations"] >= cap - 1 or not math.isfinite(
            info["residual"]
        ), (
            "a divergent group exited early with a finite residual, which "
            "means the loop condition let it out without converging"
        )


@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_quasi_newton_converges_where_the_fixed_point_iteration_diverges(
        acceleration):
    """IQN is a root solver, so a contraction factor above one is not fatal.

    This is the other half of the invariant above, and the reason it is
    worded as "report *itself* unconverged" rather than "fail": on
    ``stiff-pair-1.2`` every fixed-point method walks off to NaN within
    forty steps while IQN solves the same coupled system exactly, in a
    handful of iterations, and both nodes land on the same finite
    answer.  If this ever starts failing, the quasi-Newton step has been
    turned back into a relaxation.
    """
    built = cf.FIXTURES["stiff-pair-1.2"].build(cf.CouplingConfig(
        acceleration=acceleration,
        jacobian_reuse=5 if acceleration == "iqn-imvj" else 0))
    gm = built.gm
    key = built.group_keys[0]
    for _ in range(40):
        gm.step()
    info = gm.coupling_diagnostics()[key]
    assert info["converged"], f"{acceleration} did not converge: {info}"
    positions = [float(gm.get_node_state(n)["position"]) for n in ("a", "b")]
    assert all(math.isfinite(p) for p in positions), positions
    assert abs(positions[0]) < 10.0, (
        f"{acceleration} stayed 'converged' while the state ran away: "
        f"{positions}"
    )
    # Both nodes are driven by the same external oscillator and have the
    # same parameters, so the exact solution of the coupled system puts
    # them in the same place.
    assert abs(positions[0] - positions[1]) < 1e-3 * max(
        abs(positions[0]), 1.0)


def test_convergent_neighbour_of_the_divergent_fixture_still_converges():
    """The divergence at gain 1.2 is the gain, not the fixture's plumbing."""
    built = cf.FIXTURES["stiff-pair-0.5"].build(cf.CouplingConfig())
    gm = built.gm
    for _ in range(5):
        gm.step()
    info = gm.coupling_diagnostics()[built.group_keys[0]]
    assert info["converged"]
    assert info["iterations"] < info_cap(gm, built.group_keys[0]) - 1


# ---------------------------------------------------------------------------
# IQN on a degenerate interface
# ---------------------------------------------------------------------------


def _one_dof_pair(acceleration, accelerated_fields, *, dt=0.05, gain=0.5):
    """Two mutually anchored springs, accelerating one scalar at most.

    Written out here rather than taken from the fixture registry because
    the point is the *hand-written* ``accelerated_fields``: no fixture
    produces a one-dimensional quasi-Newton problem, since a coupling
    group is a cycle and every member feeds the interface.  Naming a
    single node's single field is the only way to get one, and is
    exactly what a user does when overriding the auto-detection.

    The constants mirror ``benchmarks/coupling_fixtures.py``: unit
    ``dt**2 k/m`` so the edge weight *is* the coupling gain, and a
    damping of ``(1 - (1 - gain) * 0.8) / dt`` so the coupled solve's
    ``1/(1 - gain)`` amplification stays time-step stable.  It also
    takes the registry's driver node, and that is not cosmetic: the
    damping the stability constraint forces means an undriven pair
    decays by ~0.52 per step, so by step 20 its state is ~5e-7 and a
    comparison against it measures nothing.  The driver keeps the
    trajectory O(1) for the whole run, which is what makes the
    comparison below able to fail.
    """
    gm = GraphManager()
    stiffness = 1.0 / (dt * dt)
    damping = (1.0 - (1.0 - gain) * 0.8) / dt
    for name, x0 in (("a", -1.0), ("b", 1.0)):
        gm.add_node(SpringDamperNode(
            name, dt, stiffness=stiffness, damping=damping, mass=1.0,
            rest_length=0.0, initial_position=x0))
    from maddening.core.transforms import scale
    w = scale(gain)
    gm.add_edge("a", "b", "position", "anchor_position",
                transform=w, additive=True)
    gm.add_edge("b", "a", "position", "anchor_position",
                transform=w, additive=True)
    cf._add_driver(gm, dt, ["a", "b"])
    gm.add_coupling_group(
        ["a", "b"], max_iterations=40, tolerance=1e-5,
        acceleration=acceleration, accelerated_fields=accelerated_fields)
    gm.compile()
    return gm


#: Amplitude below which the 1-DOF comparison would be measuring the
#: float32 floor rather than an answer.  The driven pair runs at ~0.4.
_ONE_DOF_LIVE = 0.1


@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_iqn_on_a_single_dof_interface_agrees_with_plain_iteration(acceleration):
    """One accelerated scalar makes the least-squares rank-deficient.

    ``V`` is then a ``1 x max_cols`` matrix with many collinear columns,
    so the secant system has no unique solution.  The implementation
    uses ``pinv``, whose minimum-norm solution degenerates to the scalar
    secant (Aitken) step; what must not happen is a silently wrong
    answer — a NaN swallowed by the validity check, or convergence to a
    different fixed point.

    Three things make this able to fail, and all three were missing
    before.  The comparison runs over the whole trajectory rather than
    its last point, so a disagreement that only opens up mid-run
    counts.  It is scaled by the reference trajectory's own amplitude
    rather than by ``max(|ref|, 1.0)``, which on a decaying undriven
    pair was a flat 1e-3 *absolute* — six thousand times the signal, so
    that an accelerator returning zero for every field would have
    passed.  And the amplitude is asserted to be live, so the scaling
    cannot quietly become vacuous again.  The iteration counts are
    asserted too: the undriven pair converged in one iteration at the
    point of comparison, which does not exercise the rank-deficient
    least-squares this test is named after.
    """
    plain = _one_dof_pair("none", None)
    accel = _one_dof_pair(acceleration, {"a": ("position",)})
    keys = [(n, f) for n in ("a", "b") for f in ("position", "velocity")]
    trajectory = {k: [] for k in keys}
    iterations = []
    for _ in range(20):
        plain.step()
        accel.step()
        for name, field in keys:
            trajectory[(name, field)].append((
                float(plain.get_node_state(name)[field]),
                float(accel.get_node_state(name)[field]),
            ))
        iterations.append(int(accel.coupling_diagnostics()["a+b"]["iterations"]))

    for (name, field), pairs in trajectory.items():
        amplitude = max(abs(ref) for ref, _got in pairs)
        assert amplitude > _ONE_DOF_LIVE, (
            f"{name}.{field} only reached {amplitude:.2e} over the run, so "
            f"comparing against it measures decay rather than agreement"
        )
        assert all(math.isfinite(got) for _ref, got in pairs), (
            f"{acceleration} produced a non-finite {name}.{field}")
        worst = max(abs(got - ref) for ref, got in pairs)
        assert worst <= 1e-3 * amplitude, (
            f"{acceleration} on a 1-DOF interface drifted {worst:.2e} from "
            f"plain iteration on {name}.{field}, which runs at {amplitude:.2e}"
        )

    assert max(iterations) > 1, (
        f"{acceleration} never took a second iteration ({iterations}), so "
        f"the rank-deficient least-squares this test is about was never "
        f"entered"
    )
    info = accel.coupling_diagnostics()["a+b"]
    assert info["converged"]


def test_accelerated_fields_must_select_at_least_one_field_in_the_group():
    """An empty or foreign mapping is a configuration error, not a crash.

    Left unvalidated it surfaces as ``Need at least one array to
    concatenate`` from inside the traced coupling loop, with nothing
    naming the setting responsible.
    """
    with pytest.raises(ValueError, match="selects no field"):
        CouplingGroup(nodes=frozenset({"a", "b"}), accelerated_fields={})
    with pytest.raises(ValueError, match="selects no field"):
        CouplingGroup(nodes=frozenset({"a", "b"}),
                      accelerated_fields={"a": ()})
    with pytest.raises(ValueError, match="not in the group"):
        CouplingGroup(nodes=frozenset({"a", "b"}),
                      accelerated_fields={"z": ("position",)})


# ---------------------------------------------------------------------------
# Two groups, two schedules
# ---------------------------------------------------------------------------


#: Few enough iterations that neither group converges within a step, so
#: what is compared is the *iterate* — which is the only place a
#: schedule is visible.  Same device as the ring tests above.
_MIXED_UNDERCONVERGED = dict(max_iterations=3, tolerance=1e-30)


def test_mixed_mode_graph_gives_each_group_its_own_schedule():
    """Each group *honours* its declared mode; reading the field back does not.

    The previous version of this test asserted
    ``group.iteration_mode == "jacobi"``, which checks that the fixture
    set the field.  If ``iteration_mode`` were ignored for the second
    group of a graph — the exact failure "two groups keep their own
    schedules" exists to rule out — that assertion would still pass.

    What distinguishes the two modes is the iterate, so the test builds
    the counterfactual: the same graph with one group's mode flipped.
    The chain's under-converged iterate must move when the chain is
    made Jacobi, and the star's must move when the star is made
    Gauss-Seidel, which can only be true if each group ran the schedule
    it declared.  The third assertion is the independence half: flipping
    the star's mode must leave the chain bit-identical, since nothing
    flows from the star back to the chain.
    """
    config = cf.CouplingConfig(**_MIXED_UNDERCONVERGED)
    base = cf.build_mixed_modes(config)
    chain_as_jacobi = cf.build_mixed_modes(config, chain_mode="jacobi")
    star_as_gs = cf.build_mixed_modes(config, star_mode="gauss-seidel")
    chain_key, star_key = base.group_keys
    chain_nodes, star_nodes = set(chain_key.split("+")), set(star_key.split("+"))

    n_steps = 5
    state, diag = _run(base, n_steps)
    jacobi_chain, _ = _run(chain_as_jacobi, n_steps)
    gs_star, _ = _run(star_as_gs, n_steps)

    assert set(diag) == {chain_key, star_key}

    assert not np.array_equal(_flat(state, chain_nodes),
                              _flat(jacobi_chain, chain_nodes)), (
        "the chain group's iterate did not change when its mode was "
        "switched to jacobi, so it was not running the gauss-seidel "
        "schedule it declares"
    )
    assert not np.array_equal(_flat(state, star_nodes),
                              _flat(gs_star, star_nodes)), (
        "the star group's iterate did not change when its mode was "
        "switched to gauss-seidel, so it was not running the jacobi "
        "schedule it declares"
    )
    np.testing.assert_array_equal(
        _flat(state, chain_nodes), _flat(gs_star, chain_nodes),
        err_msg="changing the star group's iteration mode moved the chain "
                "group's state; the two groups are not independent",
    )


def test_mixed_mode_graph_steps_deterministically():
    """Two runs of the same two-group graph agree bitwise."""
    first = cf.FIXTURES["mixed-modes"].build(cf.CouplingConfig())
    second = cf.FIXTURES["mixed-modes"].build(cf.CouplingConfig())
    a, diag_a = _run(first, 12)
    b, diag_b = _run(second, 12)
    np.testing.assert_array_equal(_flat(a), _flat(b))
    assert {k: v["iterations"] for k, v in diag_a.items()} == \
           {k: v["iterations"] for k, v in diag_b.items()}


def test_mixed_mode_groups_keep_independent_iteration_counts():
    """Changing one group's problem must not move the other's schedule.

    Widening the star leaves the chain group untouched, so the chain's
    iteration count must not move; if the two groups shared a loop or a
    convergence flag it would.

    The whole per-step history is compared rather than the last step's
    count.  The histories are in fact identical step for step, so the
    stronger comparison is free, and a coupling between the groups that
    happened to agree on the final step would no longer slip through.
    """
    narrow = cf.FIXTURES["mixed-modes"].build(cf.CouplingConfig())
    wide = cf.build_mixed_modes(cf.CouplingConfig(), n_leaves=16)
    history_narrow = _run_history(narrow, 10)
    history_wide = _run_history(wide, 10)
    chain_key = narrow.group_keys[0]
    assert history_narrow[chain_key] == history_wide[chain_key], (
        "widening the star changed the chain group's iteration history"
    )
    assert narrow.group_keys[1] != wide.group_keys[1]


# ---------------------------------------------------------------------------
# The sweep driver's summary fields
# ---------------------------------------------------------------------------


def _signature_row(label, entries):
    """A minimal ``_same_fixed_point`` row carrying *entries*."""
    return {"ok": True, "converged_fraction": 1.0, "label": label,
            "state_signature": entries}


def _uniform_entry(value, n):
    """Signature entry for an *n*-element array whose elements are *value*."""
    return [value * n, abs(value) * math.sqrt(n), abs(value), n]


@pytest.mark.parametrize("n", [10, 60_000])
def test_fixed_point_agreement_is_size_invariant(n):
    """The same physical disagreement must score the same on any array size.

    ``sum`` and the L2 norm are extensive and ``max_abs`` is intensive,
    so comparing the raw triple against a scale taken from ``max_abs``
    made the score proportional to the element count: a per-cell
    disagreement of 7e-7 on a 60 000-cell grid was reported as a
    relative deviation of 4.2e-2, and a grid fixture could never score
    as well as a scalar one for the same physics.
    """
    ref = _signature_row("ref", {"grid.temperature": _uniform_entry(1.0, n)})
    off = _signature_row("off", {"grid.temperature": _uniform_entry(1.01, n)})
    got = sweep._same_fixed_point([ref, off])
    assert got["max_relative_deviation"] == pytest.approx(0.01, rel=1e-6)
    assert got["worst_config"] == "off"


def test_agreement_metric_ignores_nodes_outside_every_group():
    """A driver node must not set the scale a coupled node is measured against.

    ``heterogeneous``'s driver is a spring carrying ``position`` ~0.88
    while the coupled probes carry ~0.011.  While the signature covered
    every node, the field scale for ``position`` came from the driver —
    which is in no coupling group and cannot disagree with anything —
    and a 10% error on a probe scored 1.3e-3 against a 5e-3 threshold.
    """
    built = cf.build_heterogeneous(cf.CouplingConfig(), n_cells=2000)
    gm = built.gm
    for _ in range(5):
        gm.step()
    coupled = sweep._coupled_nodes(built)
    assert "driver" in gm.node_names and "driver" not in coupled

    sig = sweep._state_signature(gm, coupled)
    assert not any(key.startswith("driver.") for key in sig), sorted(sig)
    assert sweep._field_scales(sig)["position"] == max(
        sig[f"probe{i}.position"][2] for i in range(4)), (
        "the position scale is not coming from the coupled probes"
    )

    perturbed = {k: list(v) for k, v in sig.items()}
    entry = perturbed["probe0.position"]
    perturbed["probe0.position"] = [1.10 * entry[0], 1.10 * entry[1],
                                    1.10 * entry[2], entry[3]]
    got = sweep._same_fixed_point([_signature_row("ref", sig),
                                   _signature_row("off", perturbed)])
    assert got["max_relative_deviation"] >= 0.05, (
        f"a 10% error on a coupled probe scored "
        f"{got['max_relative_deviation']:.2e}"
    )


def _timing_row(label, acceleration, median, p95, iterations):
    return {"ok": True, "converged_fraction": 1.0, "label": label,
            "acceleration": acceleration, "iteration_mode": "gauss-seidel",
            "convergence_norm": "interface", "mean_step_ms": median,
            "median_step_ms": median, "p95_step_ms": p95,
            "iterations_mean": iterations}


def test_best_picker_prefers_fewer_iterations_when_times_overlap():
    """Two rows whose recorded spreads overlap tie, and iterations decide.

    The numbers are ``star-8``'s.  A fixed 10% band around the fastest
    mean put ``gs/aitken/interface`` outside it and reported the row
    with 2.6x the iterations as the fixture's best configuration, even
    though the fastest row's own median-to-p95 swing was 84% — far wider
    than the 14% gap between them.
    """
    slow_but_scattered = _timing_row(
        "gs/fixed0.8/interface", "fixed", median=0.299, p95=0.549,
        iterations=18.62)
    steady = _timing_row(
        "gs/aitken/interface", "aitken", median=0.368, p95=0.458,
        iterations=7.08)
    best = sweep._best([slow_but_scattered, steady])
    assert best["label"] == "gs/aitken/interface", best

    # ... and a row that really is slower is still not tied.
    genuinely_slower = _timing_row(
        "gs/iqn-ils/interface", "iqn-ils", median=3.0, p95=3.2,
        iterations=2.0)
    best = sweep._best([slow_but_scattered, steady, genuinely_slower])
    assert best["label"] == "gs/aitken/interface", best


def test_heterogeneous_boundary_edges_are_all_additive():
    """Mixing an additive and a non-additive edge on one field is order luck.

    ``_resolve_boundary`` accumulates only when the target field is
    already present, so a non-additive edge overwrites whatever was
    resolved before it.  ``heterogeneous`` wires both the grid and the
    driver into each probe's ``anchor_position``; with one of them
    non-additive the fixture was correct only because the grid edge
    happens to be inserted first, and the other order would have
    silently dropped the driver.
    """
    built = cf.build_heterogeneous(cf.CouplingConfig(), n_cells=64)
    shared = [e for e in built.gm.edges
              if e.target_field == "anchor_position"]
    assert len(shared) > 4, "expected both grid and driver edges per probe"
    assert all(e.additive for e in shared), [
        (e.source, e.target, e.additive) for e in shared if not e.additive
    ]


def test_coupling_statistics_do_not_depend_on_the_timed_step_count():
    """``n_stat_steps`` pins the iteration statistics window.

    The statistics pass used to run ``min(n_steps, 50)`` steps from
    wherever the timed run left the state, so shortening a timing run
    also halved the sample count *and* moved the window earlier in the
    trajectory.  On a fixture driven by a 44-step oscillator that moved
    the mean iteration count by tens of percent, which makes rows
    recorded at different ``--steps`` silently incomparable.
    """
    from maddening.core.simulation.profiler import profile_graph

    means = []
    for n_steps in (4, 11):
        built = cf.FIXTURES["stiff-pair-0.5"].build(cf.CouplingConfig())
        report = profile_graph(built.gm, n_steps=n_steps, n_warmup=2,
                               measure_coupling=False, n_stat_steps=12)
        means.append({k: v["mean"] for k, v in report.coupling_iter_stats.items()})
    assert means[0] and means[0] == means[1], means
