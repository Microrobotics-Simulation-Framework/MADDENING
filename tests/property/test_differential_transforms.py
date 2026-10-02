"""Differential oracles: ``vmap`` == a per-sample loop, ``jit`` == eager.

A coupled step is a pure function of ``(state, external inputs, params)``,
so batching it and running it sample by sample must agree, and so must the
compiled step and the same Python function run with ``jax.disable_jit()``.
Stated over generated coupled graphs of synthetic nodes
(:mod:`tests.property.coupled_graphs`):

* **``vmap`` == loop**, through the public batch entry point
  ``run_sweep`` against ``run_scan`` from each member's initial state, and
  through ``jax.vmap`` of the compiled step against the step per member --
  states, the loop's own ``_meta`` report and the warm starts, and
  ``jax.grad`` of a rollout batched against per member.  On a multi-rate
  graph ``run_sweep`` documents that a batched ``_meta`` starts each member
  at its own phase; that is checked against members run one at a time
  from those phases.
* **``jit`` == eager**: the step function compiled and run eagerly.

Tolerances.  ``tests/core/test_coupling_gradient_bound_under_vmap.py``
pins the states and the loop's slots bit-identical under ``vmap`` on a
scalar group.  With ``n x n`` gains the batched program reduces and
multiplies in different kernels, so: states to the forward error of each
node's sum per pass (:func:`~tests.property.coupled_graphs.rounding_bound`),
``iterations`` equal, the residual (a cancellation) to the norm's
float32 floor (:func:`residual_noise_floor`), ``rho_spectral`` to its
own Arnoldi residual -- the resolution the key documents for a Ritz
value -- plus ``sqrt(8 eps)`` absolute: a non-normal compressed Jacobian
can be nearly defective, and a Jordan block of size two moves its
eigenvalue by the square root of a few batched ulps (measured: 1e-4
relative on a non-normal three-node group), and
``gradient_relative_error_bound`` is *known* to differ in its leading
digit under ``vmap`` (2.01e-06 against 1.84e-06, ~8%, documented in the
0.4.0 release notes) -- held to the same 25% that module allows, not
re-reported.

What these cannot see: a fault that is the same in the batched and the
unbatched program -- which is every fault in the step itself; these
oracles only catch what batching or compiling changes.
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
from tests.core.test_coupling_solver_equivalence import residual_noise_floor
from tests.property import coupled_graphs as cg

_STEPS = 3
_BATCH = 3


def _key(group):
    return tuple(sorted(group.items()))


@functools.lru_cache(maxsize=None)
def _graph(structure, group_items, timesteps=None):
    gdef = cg.STRUCTURES[structure]
    if timesteps is not None:
        gdef = gdef.with_timesteps(dict(timesteps))
    return gdef, cg.build_graph(gdef, dict(group_items))


def _members(gdef, values, rng):
    """``_BATCH`` initial states: *values*' ``x0`` perturbed per member."""
    out = []
    for _ in range(_BATCH):
        v = {k: dict(x) for k, x in values.items()}
        for nm in v:
            v[nm]["x0"] = np.asarray(v[nm]["x0"] + rng.normal(size=v[nm]["x0"].shape),
                                     np.float32)
        out.append(v)
    return out


def _stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def _member(tree, b):
    return jax.tree.map(lambda x: np.asarray(x[b]), tree)


# ---------------------------------------------------------------------------
# run_sweep == run_scan per member
# ---------------------------------------------------------------------------


def assert_sweep_matches_members(gdef, gm, values, seed, steps=_STEPS):
    rng = np.random.default_rng(seed)
    members = _members(gdef, values, rng)
    params = cg.params_for(gm, values)
    singles = []
    for v in members:
        cg.set_initial(gm, v)
        gm.run_scan(steps, params=params)
        singles.append(cg.snapshot(gm))
    cg.set_initial(gm, members[0])
    initial = _stack([{nm: {f: (jnp.asarray(v[nm]["x0"]) if f == "x" else jnp.asarray(x))
                            for f, x in cg.snapshot(gm)[nm].items()} for nm in gm.node_names}
                      for v in members])
    batched = gm.run_sweep(steps, initial, params=params)
    passes = steps * int(gm._coupling_groups[0].max_iterations)  # noqa: SLF001
    for b, single in enumerate(singles):
        got = _member(batched, b)
        gap = cg.relative_gap(got, single)
        assert gap <= cg.rounding_bound(gdef, values, single, passes), (
            f"member {b}: run_sweep {gap:.3e} from run_scan")
        for nd in gdef.nodes:
            for f, want in cg.expected_leaves(nd, steps).items():
                assert got[nd.name][f].tobytes() == want.tobytes(), (b, nd.name, f)


_SWEEP_CASES = {
    "iqn-imvj-predictor": ("triangle", dict(acceleration="iqn-imvj", jacobian_reuse=2,
                                            tolerance=1e-5, max_iterations=12,
                                            predictor="linear")),
    "fori-aitken-diagnostics": ("triangle", dict(solver="fori", diagnostics=True,
                                                 acceleration="aitken", convergence_norm="mixed",
                                                 rtol=1e-4, max_iterations=12)),
    "nonlinear-jacobi-interface": ("nonlinear-ring", dict(
        iteration_mode="jacobi", convergence_norm="interface", rtol=1e-4, max_iterations=20)),
}


@pytest.mark.parametrize("case", sorted(_SWEEP_CASES))
# Costly tier: per example, three run_scans and one batched sweep.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_batched_sweep_is_each_member_run_on_its_own(case, data):
    """``run_sweep`` == ``run_scan`` per member (per push).

    Slow sibling: :func:`test_transforms_agree_on_generated_graphs`.
    """
    structure, group = _SWEEP_CASES[case]
    gdef, gm = _graph(structure, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    assert_sweep_matches_members(gdef, gm, values, data.draw(st.integers(0, 2**31 - 1)))


def test_a_batched_sweep_starts_each_member_at_its_own_multirate_phase():
    """``run_sweep``'s documented batched ``_meta``: one phase per member.

    "Pass an explicit ``_meta`` entry in ``initial_states`` -- batched like
    any other leaf -- to start each simulation from a different phase."
    Each member is the graph stepped ``p`` times first (phase ``p``), and
    must then sweep exactly as the graph itself continues from there.
    """
    group = dict(acceleration="iqn-ils", tolerance=1e-5, max_iterations=12,
                 predictor="linear")
    ts = (("drv", 1.0), ("g0", 3.0), ("g1", 3.0), ("g2", 3.0), ("sink", 1.0))
    gdef, gm = _graph("triangle", _key(group), ts)
    values = cg.draw_values(np.random.default_rng(3), gdef, 0.9)
    params = cg.params_for(gm, values)
    starts, finals = [], []
    for phase in range(3):
        cg.set_initial(gm, values)
        for _ in range(phase):      # one scan length for every phase: one compile
            gm.run_scan(1, params=params)
        starts.append({**{nm: dict(s) for nm, s in cg.snapshot(gm).items()},
                       "_meta": {k: np.asarray(v) for k, v in gm._state["_meta"].items()}})  # noqa: SLF001
        gm.run_scan(4, params=params)
        finals.append(cg.snapshot(gm))
    cg.set_initial(gm, values)
    batched = gm.run_sweep(4, _stack(starts), params=params)
    for b, want in enumerate(finals):
        got = _member(batched, b)
        moved = cg.bitwise_differences(got, want)
        gap = cg.relative_gap(got, want)
        assert gap <= cg.rounding_bound(gdef, values, want, 4 * 12), (
            f"member at phase {b}: {gap:.3e} from the graph continued from that phase "
            f"({moved})")


# ---------------------------------------------------------------------------
# vmap of the compiled step == the step per member; gradients batched
# ---------------------------------------------------------------------------


def _step_fn(gm):
    return gm._build_step_fn()  # noqa: SLF001


_PROGRAMS: dict = {}


def _programs(gm, gdef, steps):
    """``(run, batched, grad, vgrad)`` for *gm*, jitted once per graph.

    Each is a function of ``(params, state)``, so drawn values and
    initial states reuse one compilation; building them per example spent
    minutes compiling the same programs.
    """
    key = (id(gm), steps)
    if key not in _PROGRAMS:
        step = _step_fn(gm)
        ext = gm._resolve_external_inputs(None)  # noqa: SLF001

        def run(p, s):
            return _rollout(step, s, ext, p, steps)

        def loss(p, s):
            out = run(p, s)
            return sum(jnp.sum(out[nm]["x"]) for nm in gdef.group_nodes)

        _PROGRAMS[key] = (gm, jax.jit(run), jax.jit(jax.vmap(run, in_axes=(None, 0))),
                          jax.jit(jax.grad(loss)),
                          jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0))))
    return _PROGRAMS[key][1:]


def _rollout(step, state, ext, params, steps):
    def body(s, _):
        return step(s, ext, params), None
    return jax.lax.scan(body, state, None, length=steps)[0]


def _member_states(gm, members):
    out = []
    for v in members:
        cg.set_initial(gm, v)
        out.append(jax.tree.map(jnp.asarray, gm._state))  # noqa: SLF001
    return out


#: Known to differ by ~8% under vmap (documented); held to 25%.
_GRADIENT_BOUND_SLOT = "gradient_relative_error_bound"


def assert_vmapped_step_matches(gdef, gm, values, seed, steps=_STEPS):
    rng = np.random.default_rng(seed)
    members = _members(gdef, values, rng)
    params = cg.params_for(gm, values)
    states = _member_states(gm, members)
    run, batched_run, _g, _vg = _programs(gm, gdef, steps)
    singles = [run(params, s) for s in states]
    batched = batched_run(params, _stack(states))
    key = gdef.key
    passes = steps * int(gm._coupling_groups[0].max_iterations)  # noqa: SLF001
    n_float = sum(int(np.size(v)) for nm in gdef.group_nodes
                  for v in jax.tree.leaves(singles[0][nm])
                  if np.issubdtype(np.asarray(v).dtype, np.floating))
    group = gm._coupling_groups[0]  # noqa: SLF001
    floor = residual_noise_floor(group.convergence_norm, group.rtol, n_float)
    for b, single in enumerate(singles):
        got = _member(batched, b)
        want = jax.tree.map(np.asarray, single)
        user_got = {k: v for k, v in got.items() if k != "_meta"}
        user_want = {k: v for k, v in want.items() if k != "_meta"}
        gap = cg.relative_gap(user_got, user_want)
        assert gap <= cg.rounding_bound(gdef, values, user_want, passes), (
            f"member {b}: vmap {gap:.3e} from the loop")
        # A uniform-rate graph whose fori group reports nothing has no _meta.
        meta_g, meta_w = got.get("_meta", {}), want.get("_meta", {})
        for slot in ("iterations", "total_iterations"):
            k = f"coupling_{key}_{slot}"
            if k in meta_w:
                assert int(meta_g[k]) == int(meta_w[k]), (b, slot)
        k = f"coupling_{key}_residual"
        if k in meta_w and np.isfinite(meta_w[k]):
            # A batched reduction rounds differently: the residual is a
            # cancellation, so it agrees to the norm's float32 floor.
            assert abs(float(meta_g[k]) - float(meta_w[k])) <= floor, (
                b, float(meta_g[k]), float(meta_w[k]), floor)
        k = f"coupling_{key}_rho_spectral"
        r = f"coupling_{key}_spectral_residual"
        if k in meta_w and np.isfinite(meta_w[k]):
            # The Ritz radius is resolved to the Arnoldi residual (every
            # Ritz value of a normal map lies within it of an eigenvalue),
            # plus its rounding.  A non-normal compressed Jacobian can be
            # nearly defective, and a Jordan block of size two moves its
            # eigenvalue by the *square root* of a perturbation: a few
            # batched ulps on an order-one operator, sqrt(8 eps) ~ 1e-3.
            tol = max(float(meta_g[r]), float(meta_w[r])) + float(np.sqrt(8 * cg.EPS32))
            assert abs(float(meta_g[k]) - float(meta_w[k])) <= tol, (
                b, float(meta_g[k]), float(meta_w[k]), tol)
        k = f"coupling_{key}_{_GRADIENT_BOUND_SLOT}"
        if k in meta_w and np.isfinite(meta_w[k]) and meta_w[k] > 0:
            a, w = float(meta_g[k]), float(meta_w[k])
            assert abs(a - w) <= 0.25 * max(a, w), (b, a, w)


def assert_vmapped_gradient_matches(gdef, gm, values, seed, steps=2):
    """``vmap(grad)`` over members' initial states == ``grad`` per member."""
    rng = np.random.default_rng(seed)
    members = _members(gdef, values, rng)
    params = cg.params_for(gm, values)
    states = _member_states(gm, members)
    _r, _br, grad, vgrad = _programs(gm, gdef, steps)
    singles = [grad(params, s) for s in states]
    batched = vgrad(params, _stack(states))
    for b, single in enumerate(singles):
        got = _member(batched, b)["nodes"]
        want = jax.tree.map(np.asarray, single)["nodes"]
        # The adjoint is a GMRES solve whose iterate rounds like the
        # forward; the forward agreed to round-off per pass, and the gradient
        # amplifies it by at most the conditioning the forward did.
        gap = cg.relative_gap(got, want)
        assert gap <= 1e-4, f"member {b}: vmap(grad) {gap:.3e} from grad per member"


_VMAP_CASES = {
    "ift-iqn-predictor": ("triangle", dict(acceleration="iqn-ils", tolerance=1e-5,
                                           max_iterations=12, predictor="linear")),
    "ift-iqn-diagnostics": ("triangle", dict(acceleration="iqn-ils", tolerance=1e-5,
                                             max_iterations=12, diagnostics=True,
                                             predictor="linear")),
    "ift-aitken-jacobi": ("flux-pair", dict(acceleration="aitken", tolerance=1e-5,
                                            max_iterations=12)),
    "fori-fixed-mixed": ("triangle", dict(solver="fori", diagnostics=True, acceleration="fixed",
                                          relaxation=0.8, convergence_norm="mixed", rtol=1e-4,
                                          max_iterations=12)),
}


def _vmap_cases(per_push):
    """``ift-iqn-diagnostics`` compiles the spectral and gradient bounds into
    a batched and an unbatched program (14 s on CI), so it runs in the slow
    lane; the bound under ``vmap`` is read per push by
    ``tests/core/test_coupling_gradient_bound_through_a_vmapped_step.py``."""
    for case in sorted(_VMAP_CASES):
        slow = case == "ift-iqn-diagnostics" or case not in per_push
        yield pytest.param(case, marks=(pytest.mark.slow,) if slow else ())


# Per push: tests/property/test_differential_transforms.py::test_a_vmapped_coupled_step_is_the_step_per_member[ift-iqn-predictor]
@pytest.mark.parametrize("case", list(_vmap_cases(_VMAP_CASES)))
# Costly tier: per example, a batched and three unbatched rollouts.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_vmapped_coupled_step_is_the_step_per_member(case, data):
    """``vmap`` of the step == the step per member, report included (per push)."""
    structure, group = _VMAP_CASES[case]
    gdef, gm = _graph(structure, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    assert_vmapped_step_matches(gdef, gm, values, data.draw(st.integers(0, 2**31 - 1)))


# Slow: the batched and unbatched backward programs compile in 11 s on CI.
# Per push: tests/property/test_differential_transforms.py::test_a_vmapped_gradient_through_an_ift_step_is_the_gradient_per_member
# (one fixed draw through the smallest IFT group, one step).
@pytest.mark.slow
@pytest.mark.parametrize("case", ["ift-iqn-predictor", "fori-fixed-mixed"])
# Costly tier: per example, a batched gradient and three unbatched ones.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_vmapped_gradient_is_the_gradient_per_member(case, data):
    """``vmap(grad)`` == ``grad`` per member, over drawn values (slow lane)."""
    structure, group = _VMAP_CASES[case]
    gdef, gm = _graph(structure, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9)))
    assert_vmapped_gradient_matches(gdef, gm, values, data.draw(st.integers(0, 2**31 - 1)))


def test_a_vmapped_gradient_through_an_ift_step_is_the_gradient_per_member():
    """``vmap(grad)`` == ``grad`` per member through an IFT-coupled step,
    at one fixed draw (per push).

    The same oracle as the property above, on the flux pair under Aitken
    (``ift-aitken-jacobi``), the smallest IFT group of the vmap cases,
    over one step: the batched and unbatched backward programs compile
    once each (about 4.5 s on three local cores).  Without it no push
    batches a gradient through the IFT rule's ``custom_linear_solve``."""
    structure, group = _VMAP_CASES["ift-aitken-jacobi"]
    gdef, gm = _graph(structure, _key(group))
    values = cg.draw_values(np.random.default_rng(7), gdef, 0.9, nonnormal=True)
    assert_vmapped_gradient_matches(gdef, gm, values, 3, steps=1)


# ---------------------------------------------------------------------------
# jit == eager
# ---------------------------------------------------------------------------


def assert_eager_matches_jit(gdef, gm, values, steps=2):
    step = _step_fn(gm)
    ext = gm._resolve_external_inputs(None)  # noqa: SLF001
    params = cg.params_for(gm, values)
    cg.set_initial(gm, values)
    state0 = jax.tree.map(jnp.asarray, gm._state)  # noqa: SLF001
    compiled = jax.jit(step)
    s_jit = state0
    for _ in range(steps):
        s_jit = compiled(s_jit, ext, params)
    s_eager = state0
    with jax.disable_jit():
        for _ in range(steps):
            s_eager = step(s_eager, ext, params)
    got = jax.tree.map(np.asarray, s_eager)
    want = jax.tree.map(np.asarray, s_jit)
    passes = steps * int(gm._coupling_groups[0].max_iterations)  # noqa: SLF001
    gap = cg.relative_gap({k: v for k, v in got.items() if k != "_meta"},
                          {k: v for k, v in want.items() if k != "_meta"})
    assert gap <= cg.rounding_bound(gdef, values, want, passes), f"eager {gap:.3e} from jit"
    k = f"coupling_{gdef.key}_iterations"
    if k in want.get("_meta", {}):
        assert int(got["_meta"][k]) == int(want["_meta"][k]), "eager and jit took different passes"


_EAGER_CASES = {
    "ift-iqn-imvj": ("triangle", dict(acceleration="iqn-imvj", jacobian_reuse=2,
                                      tolerance=1e-5, max_iterations=12, predictor="linear")),
    "fori-aitken": ("flux-pair", dict(solver="fori", diagnostics=True, acceleration="aitken",
                                      tolerance=1e-5, max_iterations=12)),
}


@pytest.mark.parametrize("case", sorted(_EAGER_CASES))
def test_the_eager_coupled_step_is_the_compiled_step(case):
    """``jax.disable_jit()`` == ``jax.jit`` on a coupled step (per push).

    One fixed draw per case: the eager step dispatches every operation of
    every pass separately, which is seconds per step, so this is not a
    property.  Slow sibling: :func:`test_transforms_agree_on_generated_graphs`.
    """
    structure, group = _EAGER_CASES[case]
    gdef, gm = _graph(structure, _key(group))
    values = cg.draw_values(np.random.default_rng(11), gdef, 0.9, nonnormal=True)
    assert_eager_matches_jit(gdef, gm, values)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_transforms.py::test_a_vmapped_coupled_step_is_the_step_per_member
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_transforms_agree_on_generated_graphs(data):
    """``vmap`` == loop and ``jit`` == eager with structure and config drawn.

    Per-push siblings: :func:`test_a_batched_sweep_is_each_member_run_on_its_own`,
    :func:`test_a_vmapped_coupled_step_is_the_step_per_member` and
    :func:`test_the_eager_coupled_step_is_the_compiled_step`.
    """
    gdef = data.draw(cg.graph_defs())
    group = data.draw(cg.group_configs(gdef, caps=(1, 2, 5, 12)))
    group = dict(group, solver=data.draw(st.sampled_from(["ift", "fori"])),
                 diagnostics=data.draw(st.booleans()))
    note(f"{gdef}\n{group}")
    gm = cg.build_graph(gdef, group)
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    seed = data.draw(st.integers(0, 2**31 - 1))
    assert_sweep_matches_members(gdef, gm, values, seed)
    assert_vmapped_step_matches(gdef, gm, values, seed)
    if data.draw(st.booleans()):
        assert_eager_matches_jit(gdef, gm, values, steps=1)
