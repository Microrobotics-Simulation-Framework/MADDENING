"""Differential oracle: integer, ``uint32`` and ``bool`` leaves survive every configuration.

A non-float leaf that does not read a coupled input -- a step counter, a
linear-congruential tag, a toggling flag -- is a pure function of the
pre-step state.  Whatever a coupling group does to its floating fields, the
leaf after ``k`` updates of its node is a closed form
(:func:`~tests.property.coupled_graphs.expected_leaves`): ``2**24 + 1 + k``
(a value float32 cannot hold), ``0xDEADBEEF`` stepped ``k`` times through
``t * 1664525 + 1013904223 mod 2**32``, and ``k`` toggles.  The oracle is
that closed form, **bitwise and in dtype**, after every step of every
configuration -- the two solvers, every acceleration (MADD-ANO-059 was the
accelerators rounding these through float32), predictors, a cap of one,
sub-cycling (a fast member updates ``d`` times per step), waveform sweeps,
multi-rate dividers (only firing steps count), adaptive stepping (two half
steps per accepted step) and ``vmap``.

The solver, multi-rate, sub-cycling and adaptive oracles check the same
closed form on their own runs; this module covers the configurations they
do not reach and holds the two crashes the harness found in this corner.

What it cannot see: a leaf that *does* read a coupled input -- its value
depends on which iterate it was computed from, which is the solver
oracle's business (``test_a_flag_reading_a_coupled_input_agrees_between_the_solvers``).
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg

_STEPS = 3


def _all_leaves(gdef: cg.GraphDef) -> cg.GraphDef:
    return gdef.with_leaves(cg.LEAF_POOL, names=[nd.name for nd in gdef.nodes])


def assert_leaves(gdef, state, updates: dict, label: str = ""):
    for nd in gdef.nodes:
        for field, want in cg.expected_leaves(nd, updates[nd.name]).items():
            got = np.asarray(state[nd.name][field])
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (
                f"{label}{nd.name}.{field} = {got!r} ({got.dtype}) after "
                f"{updates[nd.name]} updates, want {want!r} ({want.dtype})")


_VALUES_RNG_SEED = 20261001


@functools.lru_cache(maxsize=None)
def _values(gdef):
    return cg.draw_values(np.random.default_rng(_VALUES_RNG_SEED), gdef, 0.9, nonnormal=True)


#: Configurations of one structure with every leaf on every node, each
#: stepped ``_STEPS`` times.  The other oracles reach the plain solver
#: matrix; these are the corners they do not.
_CASES = {
    "fori-aitken": dict(solver="fori", acceleration="aitken", tolerance=1e-5, max_iterations=12),
    "fori-fixed": dict(solver="fori", acceleration="fixed", relaxation=0.7, tolerance=1e-5,
                       max_iterations=12),
    "ift-iqn-imvj-named-int": dict(acceleration="iqn-imvj", jacobian_reuse=2, tolerance=1e-5,
                                   max_iterations=12,
                                   accelerated_fields={"g0": ("count", "x"), "g1": ("x",)}),
    "fori-iqn-ils-named-int": dict(solver="fori", acceleration="iqn-ils", tolerance=1e-5,
                                   max_iterations=12,
                                   accelerated_fields={"g0": ("tag", "x"), "g2": ("flag", "x")}),
    "ift-quadratic-predictor": dict(predictor="quadratic", acceleration="aitken",
                                    tolerance=1e-5, max_iterations=12),
    "fori-cap-one": dict(solver="fori", diagnostics=True, max_iterations=1, tolerance=1e-5),
    "ift-interface-jacobi": dict(convergence_norm="interface", rtol=1e-4,
                                 iteration_mode="jacobi", acceleration="iqn-ils",
                                 max_iterations=12),
}


@functools.lru_cache(maxsize=None)
def _case_graph(case):
    gdef = _all_leaves(cg.TRIANGLE)
    return gdef, cg.build_graph(gdef, dict(_CASES[case]))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_non_float_leaves_keep_their_closed_form_under_every_accelerator(case):
    """Per push; slow sibling :func:`test_non_float_leaves_survive_generated_configurations`."""
    gdef, gm = _case_graph(case)
    traj = cg.trajectory(gm, gdef, _values(gdef), _STEPS)
    for k, (state, _m, _r) in enumerate(traj, start=1):
        assert_leaves(gdef, state, {nd.name: k for nd in gdef.nodes}, f"step {k}: ")


def test_non_float_leaves_survive_sub_cycling_with_waveform_sweeps():
    """A fast member updates ``d`` times per macro step, whatever the sweeps."""
    d = 4
    gdef = _all_leaves(cg._cycle(3, 1, chords=((0, 2),)))
    gdef = gdef.with_timesteps({nd.name: (0.5 / d if nd.name == "g1" else 0.5)
                                for nd in gdef.nodes})
    gm = cg.build_graph(gdef, dict(subcycling=True, waveform_iterations=3,
                                   boundary_interpolation="constant",
                                   acceleration="aitken", tolerance=1e-5, max_iterations=12))
    traj = cg.trajectory(gm, gdef, _values(gdef), _STEPS)
    for k, (state, _m, _r) in enumerate(traj, start=1):
        assert_leaves(gdef, state, {nd.name: k * (d if nd.name == "g1" else 1)
                                    for nd in gdef.nodes}, f"step {k}: ")


def test_non_float_leaves_count_only_the_steps_a_multirate_block_fires_on():
    gdef = _all_leaves(cg.TRIANGLE).with_timesteps(
        {"drv": 1.0, "g0": 3.0, "g1": 3.0, "g2": 3.0, "sink": 2.0})
    gm = cg.build_graph(gdef, dict(acceleration="iqn-ils", tolerance=1e-5,
                                   max_iterations=12, predictor="linear"))
    traj = cg.trajectory(gm, gdef, _values(gdef), 7)
    for k, (state, _m, _r) in enumerate(traj, start=1):
        fired = {nd.name: -(-k // int(nd.timestep)) for nd in gdef.nodes}
        assert_leaves(gdef, state, fired, f"base step {k}: ")


def test_non_float_leaves_count_two_half_steps_per_accepted_adaptive_step():
    """Explicit-Euler nodes, so the controller settles.  No ``bool`` leaf (the
    adaptive steppers raise on one) and no ``uint32`` tag (it wraps in the
    error norm and drives the step to ``dt_min``): strict xfails in
    ``test_differential_schedules.py``."""
    gdef = (_all_leaves(cg.TRIANGLE).as_ode(-1.0).without_leaves(("flag", "tag"))
            .with_timesteps({nd.name: 0.25 for nd in cg.TRIANGLE.nodes}))
    gm = cg.build_graph(gdef, dict(acceleration="aitken", tolerance=1e-5, max_iterations=12))
    values = _values(gdef)
    for scan in (False, True):
        cg.set_initial(gm, values)
        params = cg.params_for(gm, values)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Adaptive stepper hit dt_min")
            if scan:
                _s, _h, info = gm.run_adaptive_scan(1.0, max_steps=200, dt_initial=0.25,
                                                    atol=1e-3, rtol=1e-3, dt_min=1e-3,
                                                    dt_max=0.25, params=params)
                n = int(info["n_steps"])
            else:
                _s, info = gm.run_adaptive(1.0, dt_initial=0.25, atol=1e-3, rtol=1e-3,
                                           dt_min=1e-3, dt_max=0.25, params=params)
                n = info["n_steps"]
        assert_leaves(gdef, cg.snapshot(gm), {nd.name: 2 * n for nd in gdef.nodes},
                      "run_adaptive_scan: " if scan else "run_adaptive: ")


def test_non_float_leaves_survive_a_batched_sweep():
    """``run_sweep`` (``vmap`` over ``lax.scan``): every batch member's leaves."""
    gdef = _all_leaves(cg.TRIANGLE)
    gm = cg.build_graph(gdef, dict(acceleration="iqn-imvj", jacobian_reuse=2,
                                   tolerance=1e-5, max_iterations=12, predictor="linear"))
    values = _values(gdef)
    cg.set_initial(gm, values)
    base = cg.snapshot(gm)
    batch = 3
    rng = np.random.default_rng(1)
    initial = {nm: {f: jnp.stack([jnp.asarray(v) if f != "x" else
                                  jnp.asarray(rng.normal(size=v.shape), jnp.float32)
                                  for _ in range(batch)])
                    for f, v in s.items()} for nm, s in base.items()}
    out = gm.run_sweep(_STEPS, initial, params=cg.params_for(gm, values))
    for b in range(batch):
        member = {nm: {f: np.asarray(v[b]) for f, v in s.items()} for nm, s in out.items()}
        assert_leaves(gdef, member, {nd.name: _STEPS for nd in gdef.nodes}, f"member {b}: ")


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_leaves.py::test_non_float_leaves_keep_their_closed_form_under_every_accelerator
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_non_float_leaves_survive_generated_configurations(data):
    """Every leaf, every node, drawn structure, configuration and schedule.

    Per-push sibling:
    :func:`test_non_float_leaves_keep_their_closed_form_under_every_accelerator`.
    """
    gdef = _all_leaves(data.draw(cg.graph_defs()))
    group = data.draw(cg.group_configs(gdef))
    group = dict(group, solver=data.draw(st.sampled_from(["ift", "fori"])),
                 diagnostics=data.draw(st.booleans()))
    group = cg.steer_around_known_crashes(gdef, group)
    gdef = cg.steer_leaves_around_known_crashes(gdef, group)
    schedule = data.draw(st.sampled_from(["uniform", "subcycled", "multirate"]))
    fast = None
    d = 1
    if schedule == "subcycled":
        d = data.draw(st.sampled_from([2, 3, 4]))
        fast = data.draw(st.sampled_from(gdef.group_nodes))
        gdef = gdef.with_timesteps({nd.name: (0.5 / d if nd.name == fast else 0.5)
                                    for nd in gdef.nodes})
        group = dict(group, subcycling=True, boundary_interpolation="constant",
                     waveform_iterations=data.draw(st.integers(1, 3)))
    elif schedule == "multirate":
        div = float(data.draw(st.sampled_from([2, 3])))
        gdef = gdef.with_timesteps({nd.name: (div if nd.name in gdef.group_nodes else 1.0)
                                    for nd in gdef.nodes})
    note(f"{schedule} fast={fast} d={d}\n{gdef}\n{group}")
    gm = cg.build_graph(gdef, group)
    values = data.draw(cg.drawn_values(gdef))
    steps = 5 if schedule == "multirate" else _STEPS
    for k, (state, _m, _r) in enumerate(cg.trajectory(gm, gdef, values, steps), start=1):
        if schedule == "multirate":
            updates = {nd.name: -(-k // int(nd.timestep)) for nd in gdef.nodes}
        else:
            updates = {nd.name: k * (d if nd.name == fast else 1) for nd in gdef.nodes}
        assert_leaves(gdef, state, updates, f"step {k}: ")


# ---------------------------------------------------------------------------
# Two crashes the harness found in this corner
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "differential: predictor + convergence_norm='mixed' + an int/bool leaf in "
    "the group raises KeyError at trace (the predictor rebuilds the starting "
    "iterate from float fields only); pending fix"))
@pytest.mark.parametrize("predictor", ["linear", "quadratic"])
def test_a_predictor_under_the_mixed_norm_steps_a_group_with_an_integer_leaf(predictor):
    """The l2 and interface norms step this group; the mixed norm raises.

    ``_run_coupled_block_impl``'s predictor block extrapolates the floating
    fields and writes ``new_state[nn] = predicted[nn]``, where
    ``predicted`` comes from ``unflatten_coupled_state(..., fields=
    pred_fields)`` and so holds the floating fields *only*: the group's
    integer and boolean leaves vanish from the iterate the solve starts
    from.  ``one_pass`` recomputes them from the pre-step state, so the
    l2 and interface norms (which never read them) do not notice;
    ``coupling_residual_mixed`` reads ``s_old[nn][field]`` for every field
    of the new iterate *before* skipping non-floats, and the first
    residual raises ``KeyError`` naming the leaf.  The predictor's seed is
    in ``_meta`` from ``compile()``, so this is the first step.
    """
    gdef = cg._cycle(2, 1, outside=False, leaves=("count",))
    gm = cg.build_graph(gdef, dict(predictor=predictor, convergence_norm="mixed", rtol=1e-4))
    values = cg.draw_values(np.random.default_rng(0), gdef, 0.5)
    state, _m, _r = cg.trajectory(gm, gdef, values, 1)[0]
    assert_leaves(gdef, state, {nd.name: 1 for nd in gdef.nodes})


def _mutual_flux_pair() -> cg.GraphDef:
    """Two flux producers exchanging fluxes: ``g0.q -> g1``, ``g1.q -> g0``."""
    return cg.GraphDef(
        n=1,
        nodes=(cg.NodeDef("g0", 1, flux=True), cg.NodeDef("g1", 1, flux=True)),
        edges=(cg.EdgeDef("g0", "g1", 0, "q"), cg.EdgeDef("g1", "g0", 0, "q")),
        group_nodes=("g0", "g1"))


def test_two_flux_producers_exchanging_fluxes_step_under_gauss_seidel():
    """The control: the Gauss-Seidel pass seeds producers' fluxes in two sweeps."""
    gdef = _mutual_flux_pair()
    gm = cg.build_graph(gdef, dict(tolerance=1e-5, max_iterations=30))
    values = cg.draw_values(np.random.default_rng(0), gdef, 0.5)
    _s, _m, r = cg.trajectory(gm, gdef, values, 1)[0]
    assert r["converged"]


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "differential: iteration_mode='jacobi' with a flux producer that reads "
    "another node's flux raises KeyError at trace (the Jacobi pass lacks the "
    "Gauss-Seidel pass's two-sweep flux seed); pending fix"))
def test_two_flux_producers_exchanging_fluxes_step_under_jacobi():
    """Jacobi == Gauss-Seidel in what they can step.

    ``one_pass_jacobi`` precomputes every producer's flux from the
    previous iterate with ``_resolve_boundary(nn, latest_results)`` --
    strict, and with no flux dict -- so a producer whose own input is a
    flux edge cannot resolve it and the trace raises ``KeyError: 'q'``.
    ``one_pass_gs`` has the fix (two sweeps, the first tolerating a
    missing flux); the Jacobi pass never got it.  Two slabs exchanging
    boundary heat fluxes are this graph.
    """
    gdef = _mutual_flux_pair()
    gm = cg.build_graph(gdef, dict(tolerance=1e-5, max_iterations=30, iteration_mode="jacobi"))
    values = cg.draw_values(np.random.default_rng(0), gdef, 0.5)
    _s, _m, r = cg.trajectory(gm, gdef, values, 1)[0]
    assert r["converged"]
