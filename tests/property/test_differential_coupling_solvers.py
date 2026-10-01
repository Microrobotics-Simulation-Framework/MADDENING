"""Differential oracles: the two coupling solvers, and ``diagnostics`` on/off.

Two pairs of paths that must agree, stated over generated coupled graphs
of synthetic nodes (:mod:`tests.property.coupled_graphs`):

**fori == ift.**  ``CouplingGroup.solver`` documents that the legacy
``"fori"`` loop and the default ``"ift"`` while-loop "run the same passes,
stop on the same pass and derive every value here by the same rule", and
that the returned state "agrees to float32 round-off -- bit-identical on
most graphs".  Where that parity is documented (``coupling_diagnostics``):
above the norm's float32 floor ``iterations`` and ``converged`` agree; at or
below it they "can disagree too", and the state then differs by about one
residual.  So:

* ``iterations`` and ``converged`` must be equal whenever the threshold is
  above the floor (:func:`residual_noise_floor`, the derivation
  ``tests/core/test_coupling_solver_equivalence.py`` makes) -- unless one
  solver's error estimate sits within its own rounding of the threshold
  (:func:`~tests.property.coupled_graphs.criterion_is_resolved`: the
  residual's floor, amplified through the rate estimate), where an ulp
  decides the pass;
* where the iteration counts agree, the states agree to float32
  round-off: the documented difference is about one ulp on a couple of
  components per pass (``while_loop`` and ``fori_loop`` bodies compile to
  differently rounded arithmetic), and the bound allowed is the forward
  error of each node's sum, ``2 T eps sum |term|``, per pass, summed
  without any help from the contraction
  (:func:`~tests.property.coupled_graphs.rounding_bound`) -- at least
  eight ulps of ``|x|`` a pass, more where large terms cancel;
* non-float leaves are bit-identical, and equal to their closed form.

**diagnostics on == off.**  ``diagnostics=True`` reports on the forward and
must not change it (``test_coupling_diagnostics_leave_the_state_alone.py``
pins that on ``chain-5``).  Here: states, ``iterations``, ``converged``, the
warm-start slots (predictor history, IMVJ ``V``/``W``) and gradients through
``run_scan``, **bitwise**, on both solvers -- the two settings run the same
loop on the same inputs, so there is no rounding to excuse.

What these oracles cannot see: anything both paths share -- the one-pass
map, the norms, the criterion's arithmetic, the predictor -- and a
specification error in the documented contract itself.  Those are the
business of the fixed-point oracle (``test_differential_fixed_point.py``).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.core.test_coupling_solver_equivalence import residual_noise_floor
from tests.property import coupled_graphs as cg

#: The threshold must clear the norm's float32 floor by this factor for
#: the documented iteration parity to be demanded.  Below it the criterion
#: is made of rounding and the docs say the two loops may stop on
#: different passes.
_FLOOR_MARGIN = 4.0

_STEPS = 3


def _threshold(group: dict) -> float:
    return float(group.get("tolerance", 1e-6)) if group.get(
        "convergence_norm", "l2") == "l2" else 1.0


def _n_float(state: dict, nodes) -> int:
    return sum(int(np.asarray(v).size) for nm in nodes for v in state[nm].values()
               if np.issubdtype(np.asarray(v).dtype, np.floating))


def _parity_demanded(gdef, group, state) -> bool:
    """Is the threshold above the norm's float32 floor (with margin)?"""
    floor = residual_noise_floor(group.get("convergence_norm", "l2"),
                                 group.get("rtol", 1e-6),
                                 _n_float(state, gdef.group_nodes))
    return _threshold(group) >= _FLOOR_MARGIN * floor


def _sweeps(gdef, group) -> int:
    return int(group.get("waveform_iterations", 1)) if group.get("subcycling") else 1


def _check_leaves(gdef, state, step):
    """Oracle 8 riding along: every non-float leaf at its closed form."""
    for nd in gdef.nodes:
        for field, want in cg.expected_leaves(nd, step).items():
            got = np.asarray(state[nd.name][field])
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (
                f"{nd.name}.{field} = {got!r} after {step} steps, want {want!r}")


def assert_solvers_agree(gdef, group, fori, ift, values):
    """The fori == ift oracle on two trajectories of one configuration."""
    for k, ((s_f, m_f, d_f), (s_i, m_i, d_i)) in enumerate(zip(fori, ift), start=1):
        note(f"step {k}: fori {d_f} ift {d_i}")
        _check_leaves(gdef, s_f, k)
        _check_leaves(gdef, s_i, k)
        same_passes = d_f["iterations"] == d_i["iterations"]
        floor = residual_noise_floor(group.get("convergence_norm", "l2"),
                                     group.get("rtol", 1e-6),
                                     _n_float(s_f, gdef.group_nodes))
        resolved = cg.criterion_is_resolved(d_f, floor) and cg.criterion_is_resolved(d_i, floor)
        if _parity_demanded(gdef, group, s_f) and (same_passes or resolved):
            assert same_passes, (
                f"step {k}: fori took {d_f['iterations']} passes, ift "
                f"{d_i['iterations']}, with the threshold above the float floor")
            assert d_f["converged"] == d_i["converged"], f"step {k}: verdicts differ"
        if not same_passes:
            # Below the floor the two loops may stop on adjacent passes;
            # nothing finer than "about one residual" is promised, and the
            # trajectories have parted, so later steps prove nothing.
            return
        passes = d_f["iterations"] * _sweeps(gdef, group)
        bound = cg.rounding_bound(gdef, values, s_f, passes)
        gap = cg.relative_gap(s_f, s_i)
        assert gap <= bound, (
            f"step {k}: states {gap:.3e} apart relatively after {passes} "
            f"identical passes (bound {bound:.3e})")


@functools.lru_cache(maxsize=None)
def _compiled(structure: str, group_items: tuple, solver: str, diagnostics: bool):
    group = dict(group_items, solver=solver, diagnostics=diagnostics)
    return cg.build_graph(cg.STRUCTURES[structure], group)


def _key(group: dict) -> tuple:
    return tuple(sorted(group.items()))


#: Per-push configurations of the solver oracle: every acceleration, every
#: norm, both iteration modes, both predictors with history, a cap of one,
#: IMVJ carrying columns across steps, and the flux and non-linear
#: structures.  Thresholds above the float floor, so parity is demanded.
_SOLVER_CASES = {
    "none-l2-gs-linear": ("triangle", dict(
        acceleration="none", convergence_norm="l2", tolerance=1e-4,
        max_iterations=40, predictor="linear")),
    "aitken-mixed-jacobi": ("triangle", dict(
        acceleration="aitken", convergence_norm="mixed", rtol=1e-3,
        iteration_mode="jacobi", max_iterations=12)),
    "fixed-interface-gs": ("triangle", dict(
        acceleration="fixed", relaxation=1.3, convergence_norm="interface",
        rtol=1e-3, max_iterations=40)),
    "iqn-ils-l2-jacobi": ("triangle", dict(
        acceleration="iqn-ils", convergence_norm="l2", tolerance=1e-4,
        iteration_mode="jacobi", max_iterations=12)),
    "iqn-imvj-l2-quadratic": ("triangle", dict(
        acceleration="iqn-imvj", jacobian_reuse=3, convergence_norm="l2",
        tolerance=1e-4, max_iterations=12, predictor="quadratic")),
    "cap-one": ("triangle", dict(max_iterations=1, tolerance=1e-4)),
    "flux-aitken": ("flux-pair", dict(
        acceleration="aitken", tolerance=1e-4, max_iterations=12)),
    "nonlinear-iqn-imvj-interface": ("nonlinear-ring", dict(
        acceleration="iqn-imvj", jacobian_reuse=2, convergence_norm="interface",
        rtol=1e-3, iteration_mode="jacobi", max_iterations=20)),
}


@pytest.mark.parametrize("case", sorted(_SOLVER_CASES))
# Costly tier: every example is two three-step rollouts of compiled graphs
# plus their host-side comparison, ~0.2 s on six cores.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_two_solvers_return_the_same_steps_on_generated_spectra(case, data):
    """fori == ift over drawn spectra of fixed structures (per push).

    The structure and configuration are fixed per case, so each solver is
    compiled once; Hypothesis draws the gains (rate up to 0.999, normal or
    not), the biases' scale and the initial state, which reach the
    compiled step as arguments.  Slow sibling:
    :func:`test_the_two_solvers_return_the_same_steps_on_generated_graphs`.
    """
    structure, group = _SOLVER_CASES[case]
    gdef = cg.STRUCTURES[structure]
    values = data.draw(cg.drawn_values(gdef))
    # ``"fori"`` reports only with ``diagnostics=True``; ``"ift"`` always
    # reports, and its diagnostics compile the spectral machinery that the
    # on/off oracle below owns -- so it is left off here.
    fori = cg.trajectory(_compiled(structure, _key(group), "fori", True), gdef, values, _STEPS)
    ift = cg.trajectory(_compiled(structure, _key(group), "ift", False), gdef, values, _STEPS)
    assert_solvers_agree(gdef, group, fori, ift, values)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_coupling_solvers.py::test_the_two_solvers_return_the_same_steps_on_generated_spectra
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_two_solvers_return_the_same_steps_on_generated_graphs(data):
    """fori == ift with the structure and the configuration drawn too.

    Every draw compiles two graphs, so this is the broad matrix: group
    sizes 2-4, 1-3 dimensional fields, chords, flux edges, ``tanh``,
    drivers, every acceleration, norm, mode, predictor and cap, and the
    float32-floor threshold.  Per-push sibling:
    :func:`test_the_two_solvers_return_the_same_steps_on_generated_spectra`.
    """
    gdef = data.draw(cg.graph_defs())
    group = data.draw(cg.group_configs(gdef))
    values = data.draw(cg.drawn_values(gdef))
    note(f"{gdef}\n{group}")
    fori = cg.trajectory(cg.build_graph(gdef, dict(group, solver="fori", diagnostics=True)),
                         gdef, values, _STEPS)
    ift = cg.trajectory(cg.build_graph(gdef, dict(group, solver="ift",
                                                  diagnostics=data.draw(st.booleans()))),
                        gdef, values, _STEPS)
    assert_solvers_agree(gdef, group, fori, ift, values)


# ---------------------------------------------------------------------------
# diagnostics on == off
# ---------------------------------------------------------------------------


#: The ``_meta`` slots whose presence does not depend on ``diagnostics``:
#: warm starts always, and under ``"ift"`` the loop's own report.
def _shared_slots(meta_off: dict, solver: str) -> list:
    return sorted(meta_off)


def assert_diagnostics_inert(gdef, solver, off, on):
    for k, ((s0, m0, d0), (s1, m1, d1)) in enumerate(zip(off, on), start=1):
        moved = cg.bitwise_differences(s0, s1)
        assert not moved, f"{solver} step {k}: diagnostics=True moved {moved}"
        slots = _shared_slots(m0, solver)
        changed = [s for s in slots if s not in m1
                   or m0[s].tobytes() != m1[s].tobytes()]
        assert not changed, f"{solver} step {k}: diagnostics=True changed slots {changed}"
        if solver == "ift":
            assert d0["converged"] == d1["converged"]
            assert d0["iterations"] == d1["iterations"]


_DIAGNOSTICS_CASES = {
    "ift-iqn-imvj-predictor": ("ift", "triangle", dict(
        acceleration="iqn-imvj", jacobian_reuse=3, tolerance=1e-6,
        max_iterations=12, predictor="linear")),
    "ift-aitken-interface-jacobi": ("ift", "nonlinear-ring", dict(
        acceleration="aitken", convergence_norm="interface", rtol=1e-4,
        iteration_mode="jacobi", max_iterations=20)),
    "fori-iqn-imvj-quadratic": ("fori", "triangle", dict(
        acceleration="iqn-imvj", jacobian_reuse=2, tolerance=2e-7,
        max_iterations=12, predictor="quadratic")),
    "fori-fixed-mixed-flux": ("fori", "flux-pair", dict(
        acceleration="fixed", relaxation=0.8, convergence_norm="mixed",
        rtol=1e-4, max_iterations=20)),
}


def _diagnostics_cases():
    """``ift-iqn-imvj-predictor`` costs 8 s on CI (the spectral machinery
    compiled for a three-node, two-dimensional group) and runs slow; the
    per-push ift case is ``ift-aitken-interface-jacobi``, beside the chain-5
    cells of ``test_coupling_diagnostics_leave_the_state_alone.py``."""
    for case in sorted(_DIAGNOSTICS_CASES):
        slow = case == "ift-iqn-imvj-predictor"
        yield pytest.param(case, marks=(pytest.mark.slow,) if slow else ())


# Per push: tests/property/test_differential_coupling_solvers.py::test_diagnostics_leave_every_returned_value_bit_identical[ift-aitken-interface-jacobi]
@pytest.mark.parametrize("case", list(_diagnostics_cases()))
# Costly tier, for the same reason as the solver cases above.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_diagnostics_leave_every_returned_value_bit_identical(case, data):
    """States, ``iterations``, ``converged`` and warm starts, bitwise (per push).

    Slow sibling:
    :func:`test_diagnostics_leave_generated_graphs_bit_identical`.
    """
    solver, structure, group = _DIAGNOSTICS_CASES[case]
    gdef = cg.STRUCTURES[structure]
    values = data.draw(cg.drawn_values(gdef))
    off = cg.trajectory(_compiled(structure, _key(group), solver, False), gdef, values, _STEPS)
    on = cg.trajectory(_compiled(structure, _key(group), solver, True), gdef, values, _STEPS)
    assert_diagnostics_inert(gdef, solver, off, on)


def _gradient(gm, gdef, values, steps):
    """``d sum(x of every group node after *steps*) / d params`` via ``run_scan``.

    ``run_scan`` leaves the traced final state in the graph, which
    :func:`~tests.property.coupled_graphs.recover` puts back afterwards.
    """
    names = list(gdef.group_nodes)
    cg.set_initial(gm, values)

    def loss(p):
        out = gm.run_scan(steps, params=p)
        return sum(jnp.sum(out[nm]["x"]) for nm in names)

    grads = jax.grad(loss)(cg.params_for(gm, values))
    cg.recover(gm)
    return {nm: {k: np.asarray(v) for k, v in p.items()}
            for nm, p in grads["nodes"].items()}


def _gradient_cases():
    """``ift``, seed 0, per push (allowlisted: the only per-push check that
    ``diagnostics=True`` leaves the IFT adjoint bit-identical, 11-15 s on
    CI for the backward compile of two graphs); the rest slow."""
    for solver in ("ift", "fori"):
        for seed in (0, 1, 2):
            slow = not (solver == "ift" and seed == 0)
            yield pytest.param(solver, seed, id=f"{solver}-{seed}",
                               marks=(pytest.mark.slow,) if slow else ())


# Per push: tests/property/test_differential_coupling_solvers.py::test_diagnostics_leave_gradients_through_run_scan_bit_identical[ift-0]
@pytest.mark.parametrize("solver,seed", list(_gradient_cases()))
def test_diagnostics_leave_gradients_through_run_scan_bit_identical(solver, seed):
    """``jax.grad`` through ``run_scan`` is the same bits with or without them.

    Under ``"ift"`` the backward pass is the implicit-function adjoint at
    the returned state; ``diagnostics=True`` adds Jacobian-vector products
    after the loop (the spectral and gradient bounds) behind
    ``stop_gradient``, which must change no bit of the adjoint.  Under
    ``"fori"`` it adds carries to the loop, which must change no bit of
    the derivative through the iterates.  Fixed draws: each gradient
    retraces the rollout.  Slow sibling (drawn graphs):
    :func:`test_diagnostics_leave_generated_graphs_bit_identical`.
    """
    group = dict(acceleration="iqn-ils", tolerance=1e-5, max_iterations=12,
                 predictor="linear")
    gdef = cg.TRIANGLE
    values = cg.draw_values(np.random.default_rng(seed), gdef, (0.3, 0.9, 0.99)[seed],
                            nonnormal=bool(seed % 2))
    off = _gradient(_compiled("triangle", _key(group), solver, False), gdef, values, 2)
    on = _gradient(_compiled("triangle", _key(group), solver, True), gdef, values, 2)
    moved = cg.bitwise_differences(off, on)
    assert not moved, f"{solver}: diagnostics=True moved the gradient of {moved}"


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_coupling_solvers.py::test_diagnostics_leave_every_returned_value_bit_identical
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data(), solver=st.sampled_from(["ift", "fori"]))
def test_diagnostics_leave_generated_graphs_bit_identical(data, solver):
    """diagnostics on == off with the structure and configuration drawn,
    gradients through ``run_scan`` included.

    Per-push siblings:
    :func:`test_diagnostics_leave_every_returned_value_bit_identical` and
    :func:`test_diagnostics_leave_gradients_through_run_scan_bit_identical`.
    """
    gdef = data.draw(cg.graph_defs())
    group = data.draw(cg.group_configs(gdef))
    values = data.draw(cg.drawn_values(gdef))
    note(f"{gdef}\n{group}")
    gm_off = cg.build_graph(gdef, dict(group, solver=solver, diagnostics=False))
    gm_on = cg.build_graph(gdef, dict(group, solver=solver, diagnostics=True))
    off = cg.trajectory(gm_off, gdef, values, _STEPS)
    on = cg.trajectory(gm_on, gdef, values, _STEPS)
    assert_diagnostics_inert(gdef, solver, off, on)
    moved = cg.bitwise_differences(_gradient(gm_off, gdef, values, 2),
                                   _gradient(gm_on, gdef, values, 2))
    assert not moved, f"{solver}: diagnostics=True moved the gradient of {moved}"


# ---------------------------------------------------------------------------
# A disagreement: a flag that reads a coupled input
# ---------------------------------------------------------------------------


def _sign_pair(solver, acceleration="none"):
    """``a: x = -0.5 u - 0.1`` and ``b: x = u``, ``a`` holding ``sign = sum(u) > 0``.

    ``b`` starts at ``+1`` and the fixed point is ``x = -1/15`` on both, so
    ``a``'s input is positive on the first pass and negative at the fixed
    point: a flag computed from the input changes sign during the solve.
    """
    gdef = cg.GraphDef(
        n=1,
        nodes=(cg.NodeDef("a", 1, leaves=("sign",)), cg.NodeDef("b", 1)),
        edges=(cg.EdgeDef("b", "a", 0), cg.EdgeDef("a", "b", 0)),
        group_nodes=("a", "b"))
    values = {"a": {"G": [np.array([[-0.5]], np.float32)], "b": np.array([-0.1], np.float32),
                    "x0": np.array([0.0], np.float32)},
              "b": {"G": [np.array([[1.0]], np.float32)], "b": np.array([0.0], np.float32),
                    "x0": np.array([1.0], np.float32)}}
    gm = _compiled_sign(solver, acceleration, gdef)
    s, _m, d = cg.trajectory(gm, gdef, values, 1)[0]
    return s, d


@functools.lru_cache(maxsize=None)
def _compiled_sign(solver, acceleration, gdef):
    return cg.build_graph(gdef, dict(solver=solver, acceleration=acceleration,
                                     max_iterations=30, tolerance=1e-5, diagnostics=True))


def test_a_flag_reading_a_coupled_input_is_computed_from_the_returned_iterate_under_fori():
    """The control: plain ``fori`` recomputes the flag on every pass.

    The returned flag is the one the node computes from the input of the
    pass that produced the returned state, which is negative (``x =
    -1/15``), so ``sign`` is ``False``.
    """
    s, d = _sign_pair("fori")
    assert d["converged"]
    assert float(s["b"]["x"][0]) == pytest.approx(-1.0 / 15.0, rel=1e-4)
    assert bool(s["a"]["sign"]) is False


def test_a_flag_reading_a_coupled_input_agrees_between_the_solvers():
    """fori == ift on a non-float leaf that depends on the iterate.

    ``_run_ift_forward`` keeps only floating fields in the fixed-point
    vector.  It used to restore every other field from
    ``state_after_first``, the output of the *first* pass, on the premise
    -- stated in MADD-ANO-059's resolution and in
    ``_floating_accel_fields`` -- that such a leaf "is recomputed from the
    pre-step state on every pass, so its first-pass value is already the
    converged one".  Nothing in the node contract makes that so: a flag
    may read a boundary input (a contact flag reads a gap), and the ift
    state carried a flag computed from an iterate the solve had long left
    (``sign=True`` beside ``x = -1/15``).  The non-floating fields are now
    recomputed from the returned floating ones by one more evaluation of
    the pass.  Neighbouring cases (every acceleration, a predictor under
    every norm, sub-cycling, the gradient):
    ``tests/core/test_coupling_nonfloat_leaves_track_the_iterate.py``.
    """
    s_fori, _ = _sign_pair("fori")
    s_ift, _ = _sign_pair("ift")
    assert float(s_ift["b"]["x"][0]) == pytest.approx(float(s_fori["b"]["x"][0]), rel=1e-6)
    assert bool(s_ift["a"]["sign"]) == bool(s_fori["a"]["sign"])


def test_reset_state_after_differentiating_through_run_scan_restores_the_graph():
    """The remedy the recovery warning names must itself work.

    ``_recover_from_escaped_tracers`` is "called from every entry point"
    and its warning tells the user to "set the state you want explicitly
    (set_node_state / reset_state / load_state) after differentiating".
    ``reset_state`` did not call it, and ``_meta_reset_seeds`` read the
    live predictor history -- the traced final state -- to size the seed,
    so the documented remedy raised ``UnexpectedTracerError`` on any group
    with ``predictor`` set.  It now puts the graph back first, quietly.
    The other entry points that missed the call:
    ``tests/core/test_escaped_tracer_recovery_at_every_entry_point.py``.
    """
    gdef = cg._cycle(2, 1, outside=False, leaves=())
    gm = cg.build_graph(gdef, dict(predictor="linear", tolerance=1e-5))
    values = cg.draw_values(np.random.default_rng(0), gdef, 0.5)
    params = cg.params_for(gm, values)
    jax.grad(lambda p: jnp.sum(gm.run_scan(2, params=p)["g0"]["x"]))(params)
    gm.reset_state()
    assert np.asarray(gm.get_node_state("g0")["x"]).tobytes() == np.zeros(1, np.float32).tobytes()


# ---------------------------------------------------------------------------
# IQN-IMVJ's W slot: zero exactly when float32 cannot resolve the response
# ---------------------------------------------------------------------------


def _two_springs(stiffness_a, stiffness_b, solver):
    """Two springs anchored to each other, IQN-IMVJ with a reuse window.

    The graph ``tests/property/test_differential_checkpoint.py`` carries,
    where ``coupling_a+b_W`` never moved: the coupling gain one spring sees
    of the other is ``dt**2 k / m``.
    """
    from maddening.core.graph_manager import GraphManager  # noqa: PLC0415
    from maddening.nodes.spring import SpringDamperNode  # noqa: PLC0415

    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=stiffness_a, rest_length=0.5,
                                 initial_position=0.2, damping=0.5))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=stiffness_b, rest_length=0.3,
                                 initial_position=-0.4, damping=0.2))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=6, tolerance=1e-8, solver=solver,
                          predictor="quadratic", acceleration="iqn-imvj", jacobian_reuse=2)
    gm.compile()
    for _ in range(5):
        gm.step()
    meta = gm._state["_meta"]  # noqa: SLF001
    positions = [float(gm.get_node_state(n)["position"]) for n in ("a", "b")]
    return (np.asarray(meta["coupling_a+b_V"]), np.asarray(meta["coupling_a+b_W"]),
            max(abs(p) for p in positions))


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_imvj_records_the_output_response_wherever_float32_resolves_it(solver):
    """``W`` is zero only where the raw output's response rounds to nothing.

    ``iqn_ils_update`` stores ``V`` = differences of residuals and ``W`` =
    differences of the *raw outputs* ``F(x_k) - F(x_{k-1})``.  On the
    state/IO harness's graph (stiffness 40 and 30 at ``dt = 0.01``: a gain
    ``dt**2 k / m`` of 0.004) the iterate moves by ~1.6e-07 between passes
    -- that is ``V`` -- and the output responds by ``0.004`` of that,
    ~7e-10, far below float32's resolution at ``|x| ~ 0.3`` (3e-08): ``W``
    is exactly zero, correctly, and the quasi-Newton step reduces to the
    plain one.  Make the gain resolvable (stiffness x100, gain 0.4) and
    ``W`` fills, under both solvers.  A ``W`` that is never written would
    fail the second half.
    """
    V, W, x = _two_springs(40.0, 30.0, solver)
    assert np.any(V != 0.0) and not np.any(W != 0.0)
    gain = 0.01 ** 2 * 40.0
    assert gain * float(np.max(np.abs(V))) < float(np.spacing(np.float32(x))), (
        "the zero W must be explained by float32 resolution")
    V, W, _x = _two_springs(4000.0, 3000.0, solver)
    assert int(np.sum(np.any(W != 0.0, axis=0))) >= 1, (
        "a resolvable coupling must leave secant columns in W")
