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
_FIXTURES_PY = _REPO_ROOT / "benchmarks" / "coupling_fixtures.py"


_MODULE_NAME = "maddening_coupling_fixtures"


def _load_fixtures():
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(_FIXTURES_PY))
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: ``@dataclass`` resolves the defining
    # module out of ``sys.modules`` while the class body is running.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


cf = _load_fixtures()


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


def _flat(state):
    """One array, for the bitwise-determinism comparisons."""
    return np.concatenate([state[k] for k in sorted(state)])


def _field_scales(state):
    """Largest magnitude of each *field name*, across every node carrying it."""
    scales = {}
    for (_node, field), arr in state.items():
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        scales[field] = max(scales.get(field, 0.0), m)
    return scales


def _relative_spread(a, b):
    """Worst deviation of *b* from *a*, per field name against that field's scale.

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
    """
    scales = _field_scales(a)
    worst = 0.0
    for key, ref in a.items():
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
    """
    spec = cf.FIXTURES[fixture]
    n_steps = 25
    reference = None
    worst = (0.0, "")
    for config in cf.sweep_configs(("l2", "interface")):
        built = spec.build(config)
        state, diag = _run(built, n_steps)
        assert all(d["converged"] for d in diag.values()), (
            f"{fixture} / {config.label} did not converge; the fixture is "
            f"supposed to be inside every configuration's reach"
        )
        assert np.all(np.isfinite(_flat(state))), (
            f"{fixture} / {config.label}: non-finite state")
        if reference is None:
            reference = state
            continue
        dev = _relative_spread(reference, state)
        if dev > worst[0]:
            worst = (dev, config.label)
    assert worst[0] < 5e-3, (
        f"{fixture}: {worst[1]} drifted {worst[0]:.2e} from the "
        f"gauss-seidel/none/l2 trajectory after {n_steps} steps"
    )


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
    return _run(built, n_steps)[0]


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
    base = _ring_states("gauss-seidel")
    rotated = _ring_states("gauss-seidel", rotation=3)
    reversed_ = _ring_states("gauss-seidel", reverse=True)
    assert _relative_spread(base, rotated) > 1e-3, (
        "Gauss-Seidel gave the same under-converged iterate for two "
        "different sweep orders on a ring; either the schedule is not "
        "following insertion order or the ring is symmetric enough that "
        "the order cannot matter"
    )
    assert _relative_spread(base, reversed_) > 1e-3


def test_jacobi_on_a_ring_is_order_independent():
    """Jacobi reads a frozen state, so the sweep order cannot reach the result."""
    base = _ring_states("jacobi")
    rotated = _ring_states("jacobi", rotation=3)
    reversed_ = _ring_states("jacobi", reverse=True)
    assert _relative_spread(base, rotated) < 1e-6
    assert _relative_spread(base, reversed_) < 1e-6


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
    ``1/(1 - gain)`` amplification stays time-step stable.
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
    gm.add_coupling_group(
        ["a", "b"], max_iterations=40, tolerance=1e-5,
        acceleration=acceleration, accelerated_fields=accelerated_fields)
    gm.compile()
    return gm


@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_iqn_on_a_single_dof_interface_agrees_with_plain_iteration(acceleration):
    """One accelerated scalar makes the least-squares rank-deficient.

    ``V`` is then a ``1 x max_cols`` matrix with many collinear columns,
    so the secant system has no unique solution.  The implementation
    uses ``pinv``, whose minimum-norm solution degenerates to the scalar
    secant (Aitken) step; what must not happen is a silently wrong
    answer — a NaN swallowed by the validity check, or convergence to a
    different fixed point.
    """
    plain = _one_dof_pair("none", None)
    accel = _one_dof_pair(acceleration, {"a": ("position",)})
    for _ in range(20):
        plain.step()
        accel.step()
    for name in ("a", "b"):
        for field in ("position", "velocity"):
            ref = float(plain.get_node_state(name)[field])
            got = float(accel.get_node_state(name)[field])
            assert math.isfinite(got), f"{acceleration} produced {got} for {name}.{field}"
            assert abs(got - ref) <= 1e-3 * max(abs(ref), 1.0), (
                f"{acceleration} on a 1-DOF interface converged to "
                f"{name}.{field}={got}, plain iteration gives {ref}"
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


def test_mixed_mode_graph_gives_each_group_its_own_schedule():
    """The chain group runs Gauss-Seidel and the star group runs Jacobi."""
    built = cf.FIXTURES["mixed-modes"].build(cf.CouplingConfig())
    gm = built.gm
    modes = {"+".join(sorted(g.nodes)): g.iteration_mode
             for g in gm._coupling_groups}
    chain_key, star_key = built.group_keys
    assert modes[chain_key] == "gauss-seidel"
    assert modes[star_key] == "jacobi"

    gm.step()
    diag = gm.coupling_diagnostics()
    assert set(diag) == {chain_key, star_key}
    for key in (chain_key, star_key):
        assert diag[key]["iterations"] >= 1


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
    """
    narrow = cf.FIXTURES["mixed-modes"].build(cf.CouplingConfig())
    wide = cf.build_mixed_modes(cf.CouplingConfig(), n_leaves=16)
    _, diag_narrow = _run(narrow, 10)
    _, diag_wide = _run(wide, 10)
    chain_key = narrow.group_keys[0]
    assert diag_narrow[chain_key]["iterations"] == \
        diag_wide[chain_key]["iterations"]
    assert narrow.group_keys[1] != wide.group_keys[1]
