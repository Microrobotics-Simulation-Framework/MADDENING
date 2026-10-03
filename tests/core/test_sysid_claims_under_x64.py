"""Rows of ``docs/validation/sysid_fmu_claims.yaml`` under ``jax_enable_x64``.

``jax_enable_x64`` is a supported mode -- ``fim`` recommends it for an
ill-conditioned problem -- and the sysid claims are stated for any floating
parameters.  Each test here runs one claim twice, in an x64 process:

* ``float64``: every leaf of the problem in float64, so the claim is held
  at float64's own edges (its ``eps``, its range, its subnormals);
* ``mixed``: the trainable leaves in float32 in the same x64 process,
  where ``ravel_pytree`` promotes the fitters' own coordinates to float64
  around them -- the configuration in which a fitter that took its
  precision from the promoted vector rather than from the leaves would
  answer with float64's confidence about float32 arithmetic.

The float32 versions of every claim are the tests each row cites for its
``f32`` domain; these hold the same claims, at the same edges where the
claim has one, in the other two.  Where the tree does not meet a claim the
test is a strict xfail whose reason starts with the row's id.
"""

from __future__ import annotations

import contextlib
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid
from maddening.sysid import (
    PrecisionLimitWarning,
    fim,
    fim_core,
    fit,
    fit_lm,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

#: ``float64``: every leaf float64; ``mixed``: float32 leaves in the x64 process.
LEAVES = ("float64", "mixed")
N, WINDOW = 40, 10


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _dt(leaves):
    """The leaves' dtype: float64, or float32 for the mixed configuration."""
    return {"float64": np.float64, "mixed": np.float32}[leaves]


def _eps(leaves):
    return float(np.finfo(_dt(leaves)).eps)


def _spring(leaves, **kw):
    """The spring with every constant, and its state, in the leaves' dtype.

    A node seeds its state in float32, and the first float64 update would
    promote it, which a graph scan refuses (MADD-ANO-017): the state is
    cast to the leaves' dtype here, as a float64 user's graph must be."""
    p = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0,
             initial_position=0.5, initial_velocity=0.0)
    p.update(kw)
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, **{k: np.asarray(v, _dt(leaves))
                                                 for k, v in p.items()}))
    gm.compile()
    _cast_state(gm, leaves)
    return gm


def _cast_state(gm, leaves):
    for name in gm.node_names:
        state = gm.get_node_state(name)
        gm.set_node_state(name, {
            k: (jnp.asarray(v, _dt(leaves)) if jnp.issubdtype(jnp.asarray(v).dtype,
                                                              jnp.floating) else v)
            for k, v in state.items()})


def _record(gm, leaves, n=N):
    init = {name: gm.get_node_state(name) for name in gm.node_names}
    _, hist = gm.run_scan_with_history(n)
    gm.reset_state()
    _cast_state(gm, leaves)
    return observations_from_history(init, hist)


def _position(h):
    return h["s"]["position"]


def _with(gm, leaves, **values):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, value in values.items():
        p["nodes"]["s"][key] = jnp.asarray(value, _dt(leaves))
    return p


def _rollout_residual(gm, obs, names, length):
    """The spring's position over ``length`` steps from the record's first
    state, minus the record, as a function of the named constants."""
    step_fn = gm._build_step_fn()               # noqa: SLF001
    ext = gm._default_external_inputs()         # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    measured = obs["s"]["position"][1:length + 1]

    @jax.jit
    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for key in names:
            p["nodes"]["s"][key] = sub[key]

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        return jax.lax.scan(body, init, None, length=length)[1] - measured

    return residual


def _central_difference_fim(residual, base, leaves, rel_step):
    """``J^T J`` with ``J`` by central differences, in float64 on the host,
    each column divided by the step the leaves' dtype actually took."""
    names = sorted(base)                         # ``ravel_pytree`` order
    cols = []
    for key in names:
        h = rel_step * (1.0 + abs(float(base[key])))
        plus = {**base, key: jnp.asarray(float(base[key]) + h, _dt(leaves))}
        minus = {**base, key: jnp.asarray(float(base[key]) - h, _dt(leaves))}
        step = 0.5 * (float(plus[key]) - float(minus[key]))
        cols.append((np.asarray(residual(plus), np.float64)
                     - np.asarray(residual(minus), np.float64)) / (2 * step))
    jac = np.stack(cols, axis=1)
    return jac.T @ jac


# ---------------------------------------------------------------------------
# windowed_loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_teacher_forced_loss_is_the_sum_of_independent_windows_under_x64(leaves):
    """SYS-006: every window restarts from the measured state and is its own
    scan: the loss is exactly zero at the generating parameters, and over the
    record it is the sum of the losses over each window's slice, to the
    summation's rounding in the leaves' precision."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves)
        assert obs["s"]["position"].dtype == _dt(leaves)
        assert float(windowed_loss(gm, gm.params, obs, obs_fn=_position, window=WINDOW)) == 0.0
        p = _with(gm, leaves, stiffness=41.0)
        whole = windowed_loss(gm, p, obs, obs_fn=_position, window=WINDOW)
        assert whole.dtype == _dt(leaves)
        parts = sum(float(windowed_loss(gm, p, jax.tree.map(
            lambda x, w=w: x[w * WINDOW:(w + 1) * WINDOW + 1], obs),
            obs_fn=_position, window=WINDOW)) for w in range(N // WINDOW))
        assert float(whole) > 0.0
        assert float(whole) == pytest.approx(parts, rel=16 * _eps(leaves))


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_multiple_shooting_loss_is_zero_at_the_truth_and_penalises_gaps_under_x64(leaves):
    """SYS-008: with the window states seeded from the record, each window's
    end is the next one's start, so at the generating parameters the loss with
    any continuity weight is exactly zero; a window start moved off the record
    costs misfit and continuity both, and the gradient in the states is finite."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves)
        ws = init_window_states(obs, WINDOW)

        def loss(p, w, weight=1.0):
            return windowed_loss(gm, p, obs, obs_fn=_position, window=WINDOW,
                                 window_states=w, continuity_weight=weight)

        for weight in (1.0, 1e3):
            assert float(loss(gm.params, ws, weight)) == 0.0, weight
        bad = jax.tree.map(lambda x: x, ws)
        bad["s"]["position"] = bad["s"]["position"].at[1].add(0.5)
        assert float(loss(gm.params, bad)) > 1e-2
        g = jax.grad(lambda w: loss(gm.params, w))(bad)["s"]["position"]
        assert g.dtype == _dt(leaves)
        assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g[1])) > 0.0


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_loss_gradient_is_finite_and_matches_a_difference_under_x64(leaves):
    """SYS-013: ``windowed_loss`` composes with ``jax.jit`` and ``jax.grad``,
    and its gradient is finite and informative: it has the sign of the error
    and matches a central difference of the same loss to the accuracy the
    leaves' precision allows (float64: a step of 1e-5 leaves a truncation
    error of order 1e-10 relative; float32: a step of 1e-2 and 1e-3 relative)."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves)
        loss = jax.jit(lambda p: windowed_loss(gm, p, obs, obs_fn=_position, window=WINDOW))
        k = 36.0
        g = jax.grad(loss)(_with(gm, leaves, stiffness=k))["nodes"]["s"]["stiffness"]
        assert g.dtype == _dt(leaves)
        assert bool(jnp.isfinite(g)) and float(g) > 0.0       # k too high: the loss rises
        h = (1e-5 if leaves == "float64" else 1e-2) * k
        plus = float(loss(_with(gm, leaves, stiffness=k + h)))
        minus = float(loss(_with(gm, leaves, stiffness=k - h)))
        step = 0.5 * (float(_dt(leaves)(k + h)) - float(_dt(leaves)(k - h)))
        rel = 1e-6 if leaves == "float64" else 1e-3
        assert float(g) == pytest.approx((plus - minus) / (2 * step), rel=rel)


# ---------------------------------------------------------------------------
# fim
# ---------------------------------------------------------------------------


def test_fim_is_jtj_of_a_float32_spring_rollout_in_an_x64_process():
    """SYS-020 with mixed dtypes: ``fim`` of a float32 spring's rollout in an
    x64 process is ``J^T J`` of that rollout -- matching a central difference
    at float32's tolerance (2e-2 of the matrix's scale), in float32."""
    leaves = "mixed"
    with _x64():
        gm = _spring(leaves, stiffness=60.0, damping=1.5, mass=1.2)
        obs = _record(gm, leaves, 20)
        assert float(jnp.var(obs["s"]["position"])) > 1e-2, "the rollout barely moved"
        names = ("damping", "stiffness")
        residual = _rollout_residual(gm, obs, names, 20)
        base = {k: gm.params["nodes"]["s"][k] for k in names}
        report = fim(residual, base, scale=None)
        assert report.fim.dtype == np.float32
        F = np.asarray(report.fim, np.float64)
        F_fd = _central_difference_fim(residual, base, leaves, 3e-3)
        assert np.abs(F - F_fd).max() <= 2e-2 * np.abs(F).max(), (F, F_fd)


def _coupled(leaves):
    gm = GraphManager()
    for name, x0 in (("s", 0.5), ("t", 2.0)):
        gm.add_node(SpringDamperNode(name, 0.01, **{k: np.asarray(v, _dt(leaves)) for k, v in dict(
            stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0, initial_position=x0,
            initial_velocity=0.0).items()}))
    gm.add_edge("s", "t", "position", "anchor_position")
    gm.add_edge("t", "s", "position", "anchor_position")
    tolerance = 1e-12 if leaves == "float64" else 1e-6
    gm.add_coupling_group(["s", "t"], max_iterations=60, tolerance=tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the coupled-pair stability advisory
        gm.compile()
    _cast_state(gm, leaves)
    return gm


@pytest.mark.parametrize("leaves", LEAVES)
def test_fim_reaches_through_a_coupled_step_under_x64(leaves):
    """SYS-021: ``jacfwd`` works through a coupled step "because the IFT rule is
    a custom_jvp": the matrix matches a central difference of the same coupled
    rollout -- to 1e-6 of its scale in float64 (a group converged to 1e-12),
    and to float32's 2e-2 with float32 members in an x64 process."""
    with _x64():
        gm = _coupled(leaves)
        step_fn = gm._build_step_fn()              # noqa: SLF001
        ext = gm._default_external_inputs()        # noqa: SLF001
        init = gm._state                           # noqa: SLF001
        names = ("damping", "stiffness")

        @jax.jit
        def residual(sub):
            p = jax.tree.map(lambda x: x, gm.params)
            for key, value in sub.items():
                p["nodes"]["s"][key] = value

            def body(s, _):
                s = step_fn(s, ext, p)
                return s, s["t"]["position"]

            return jax.lax.scan(body, init, None, length=20)[1]

        base = {k: gm.params["nodes"]["s"][k] for k in names}
        report = fim(residual, base, scale=None)
        assert report.fim.dtype == _dt(leaves)
        F = np.asarray(report.fim, np.float64)
        F_fd = _central_difference_fim(residual, base, leaves,
                                       1e-5 if leaves == "float64" else 1e-2)
        rel = 1e-6 if leaves == "float64" else 2e-2
        assert np.abs(F).max() > 0.0
        assert np.abs(F - F_fd).max() <= rel * np.abs(F).max(), (F, F_fd)


def _linear_fim(eig_ratio, leaves, *, m=3, seed=0, **kw):
    """``fim`` of a linear two-parameter residual over ``m`` rows whose Fisher
    matrix has eigenvalues ``eig_ratio`` and 1, in the leaves' dtype."""
    rng = np.random.default_rng(seed)
    q, rr = np.linalg.qr(rng.standard_normal((2, 2)))
    V = q * np.sign(np.diag(rr))
    U = np.linalg.qr(rng.standard_normal((m, 2)))[0]
    A = jnp.asarray((U * np.sqrt([eig_ratio, 1.0])) @ V.T, _dt(leaves))
    params = {"p0": jnp.asarray(1.0, _dt(leaves)), "p1": jnp.asarray(1.0, _dt(leaves))}
    return fim(lambda p: A @ jnp.stack([p["p0"], p["p1"]]), params, scale=None, **kw)


def _cutoff(leaves, m, n=2):
    return max(n, np.sqrt(m)) * _eps(leaves)


@pytest.mark.parametrize("m", [3, 40], ids=["n-term", "sqrt-m-term"])
@pytest.mark.parametrize("leaves", LEAVES)
def test_the_rank_cutoff_is_max_of_n_and_sqrt_m_times_the_decompositions_eps(leaves, m):
    """SYS-022: rank counts the eigenvalues strictly above ``rank_rtol *
    eigvals[-1]``, ``rank_rtol`` defaulting to ``max(n, sqrt(m)) * eps`` --
    the eps of the precision ``F`` was formed and decomposed in.  A ratio
    three cutoffs up is resolved and one a third of a cutoff down is not,
    in float64 at float64's cutoff; with float32 leaves in an x64 process at
    float32's, so a ratio of 1e-9 that float64 resolves is not resolved from
    float32 arithmetic.  Near the cutoff the warning names it."""
    with _x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionLimitWarning)
        cutoff = _cutoff(leaves, m)
        for ratio, rank in ((3 * cutoff, 2), (cutoff / 3, 1)):
            report = _linear_fim(ratio, leaves, m=m)
            assert report.eigvals.dtype == _dt(leaves)
            assert report.rank == rank, (ratio, cutoff, report.eigvals)
            assert np.isinf(np.asarray(report.crb)).all() == (rank == 1)
        assert _linear_fim(1e-9, leaves, m=m).rank == (2 if leaves == "float64" else 1)
    with _x64(), pytest.warns(PrecisionLimitWarning) as rec:
        assert _linear_fim(1.5 * cutoff, leaves, m=m).rank == 2
    assert f"cutoff of {cutoff:.4g}" in str(rec[0].message), str(rec[0].message)


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_rank_verdict_does_not_move_when_the_residual_is_rescaled_under_x64(leaves):
    """SYS-023: the verdict does not move when the residual is rescaled,
    because the threshold scales with the matrix -- over the range of sigma
    that keeps the cutoff ``rank_rtol * max(eigvals)`` a normal number of the
    leaves' precision (1e-140 to 1e140 in float64, 1e-15 to 1e15 with float32
    leaves; beyond it the verdict is documented as provisional), full rank and
    deficient alike."""
    sigmas = (1e-140, 1e-60, 1.0, 1e60, 1e140) if leaves == "float64" else \
        (1e-15, 1e-6, 1.0, 1e6, 1e15)
    with _x64():
        dt = _dt(leaves)
        full = {"a": jnp.asarray(1.5, dt), "b": jnp.asarray(-0.5, dt)}
        A = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dt)
        B = jnp.asarray([[1.0, 2.0], [2.0, 4.0], [0.5, 1.0]], dt)     # rank 1
        for sigma in sigmas:
            r = fim(lambda p: A @ jnp.stack([p["a"], p["b"]]) - 1.0, full, scale=None,
                    noise_std=sigma)
            assert r.rank == 2 and np.isfinite(np.asarray(r.crb)).all(), sigma
            r = fim(lambda p: B @ jnp.stack([p["a"], p["b"]]), full, scale=None,
                    noise_std=sigma)
            assert r.rank == 1 and np.isinf(np.asarray(r.crb)).all(), sigma


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_rank_below_the_normal_range_warns_under_x64(leaves):
    """SYS-129: the verdict holds under rescaling while ``F`` can be formed;
    once the cutoff ``rank_rtol * max(eigvals)`` falls below ``2 * m * tiny``
    of the precision ``F`` is formed in, the smallest directions' products
    flush to zero, and ``fim`` says so with a PrecisionLimitWarning naming that
    precision, and ``fim_core`` (jitted) with ``precision_limited``.  For this
    ``|J|`` of 1e-3 float64's range ends near sigma 2e143 (1e147 is past it,
    with ``F`` still above float64's tiny); float32 leaves in an x64 process
    end at float32's, near sigma 1e14."""
    rng = np.random.default_rng(0)
    m = 50
    A = np.stack([rng.normal(size=m), 1e-2 * rng.normal(size=m)], axis=1)
    inside, below, name = ((1.0, 1e60, 1e140), 1e147, "float64") if leaves == "float64" else \
        ((1.0, 1e6, 1e12), 1e15, "float32")
    with _x64():
        dt = _dt(leaves)
        J0 = jnp.asarray(1e-3 * A, dt)

        def residual(p):
            return J0 @ jnp.stack([p["x"], p["y"]])

        p0 = {"x": jnp.asarray(1.0, dt), "y": jnp.asarray(1.0, dt)}
        for sigma in inside:
            with warnings.catch_warnings():
                warnings.simplefilter("error", PrecisionLimitWarning)
                assert fim(residual, p0, scale=None, noise_std=sigma).rank == 2, sigma
            assert not bool(fim_core(residual, p0, scale=None, noise_std=sigma).precision_limited)
        with pytest.warns(PrecisionLimitWarning, match=f"below {name}'s normal range") as rec:
            fim(residual, p0, scale=None, noise_std=below)
        msg = str(rec[0].message)
        # x64 is on: the float32 decomposition is the leaves', and re-running
        # under x64 is no remedy (SYS-024's rule, for this warning too).
        assert "re-run under x64" not in msg, msg
        if leaves == "mixed":
            assert "hold the parameters and the state in float64" in msg, msg
        core = jax.jit(lambda q: fim_core(residual, q, scale=None, noise_std=below))(p0)
        assert bool(core.precision_limited)


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_fisher_matrix_flushed_to_zero_warns_under_x64(leaves):
    """SYS-129 at the end of the range: a Jacobian whose every product
    ``J_ij * J_ik`` is below the precision's smallest normal number gives an
    ``F`` of exact zeros, and ``rank`` 0 is flagged rather than reported --
    for float64 leaves at ``|J|`` near ``1e-170``, for float32 leaves in the
    x64 process at float32's ``1e-20``."""
    from tests.core.test_sysid_claims_edges import _rank2_integer_jacobian

    A, _ = _rank2_integer_jacobian()
    scale = 1e-170 if leaves == "float64" else 1e-20
    with _x64():
        dt = _dt(leaves)
        M = jnp.asarray(A, dt)
        p0 = {"p": jnp.asarray([1.0, 2.0, 3.0], dt)}

        def residual(p):
            return M @ p["p"] * jnp.asarray(scale, dt)

        with pytest.warns(PrecisionLimitWarning, match="came out exactly zero although J"):
            assert fim(residual, p0, scale=None).rank == 0
        core = jax.jit(lambda q: fim_core(residual, q, scale=None))(p0)
        assert bool(core.precision_limited)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrecisionLimitWarning)
            assert fim(lambda p: M @ p["p"], p0, scale=None).rank == 2


def test_a_float32_rank_at_the_floor_in_an_x64_process_names_a_remedy_that_settles_it():
    """SYS-024 with mixed dtypes: the warning names the ratio, the cutoff and
    the remedy.  In an x64 process the float32 decomposition comes from float32
    leaves, and re-running under x64 -- what the warning said -- changes
    nothing: the remedy that settles it, and the one named, is float64
    leaves."""
    with _x64(), pytest.warns(PrecisionLimitWarning) as rec:
        report = _linear_fim(2.08e-07, "mixed")
    assert report.rank == 1
    ev = np.asarray(report.eigvals, np.float64)
    msg = str(rec[0].message)
    assert f"{ev[0] / ev[-1]:.4g}" in msg and f"{_cutoff('mixed', 3):.4g}" in msg
    assert "jax_enable_x64', True)" not in msg, msg
    assert "x64 is already on" in msg and "float64" in msg, msg


def _spring_scale_fim(leaves, **kw):
    gm = _spring(leaves)
    obs = _record(gm, leaves, 200)
    names = ("stiffness", "damping", "mass")
    residual = _rollout_residual(gm, obs, names, 200)
    return fim(residual, {k: gm.params["nodes"]["s"][k] for k in names}, **kw)


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_springs_scale_direction_is_unidentifiable_under_x64(leaves):
    """SYS-025 and SYS-036: with position-only data k, c and m enter only as
    k/m and c/m, so the direction (1, 1, 1)/sqrt(3) is null in relative
    coordinates.  In float64, where the null eigenvalue sits near 1e-17 of the
    largest against a cutoff of 3e-15, and from float32 leaves: rank is 2 of
    3, every crb is +inf (each parameter has support in the null space), and
    least_identifiable names the largest component of that direction with its
    weight."""
    with _x64():
        report = _spring_scale_fim(leaves)
        assert report.eigvals.dtype == _dt(leaves)
        assert report.rank == 2, report.eigvals
        assert np.isinf(np.asarray(report.crb)).all(), report.crb
        v = np.asarray(report.eigvecs[:, 0], np.float64)
        assert abs(v @ np.ones(3)) / np.sqrt(3.0) > 1.0 - 64 * _eps(leaves), v
        name, weight = report.least_identifiable()
        assert name in report.param_names
        assert weight == pytest.approx(float(np.max(np.abs(v))), rel=1e-6)
        assert weight == pytest.approx(1 / np.sqrt(3.0), rel=1e-3)


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_bound_is_in_the_parameters_units_under_x64(leaves):
    """SYS-026: each residual row divided by its sigma makes ``crb`` the
    Cramer-Rao bound in the parameters' own units: sigma**2 times the
    unit-noise bound, ``c**2`` times as large for a parameter measured in
    units ``c`` times smaller under ``scale=None``, and the same under
    ``"relative"`` -- to the leaves' rounding (float64: ``c`` = 1e6, a spread
    of 1e12 in ``F`` that float64 resolves and float32 could not)."""
    with _x64():
        dt = _dt(leaves)
        A = jnp.asarray([[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dt)

        def linear(p):
            return A @ jnp.stack([p["a"], p["b"]]) - 1.0

        p = {"a": jnp.asarray(1.5, dt), "b": jnp.asarray(-0.5, dt)}
        rtol = 64 * _eps(leaves)
        unit = np.asarray(fim(linear, p, scale=None).crb, np.float64)
        noisy = np.asarray(fim(linear, p, scale=None, noise_std=0.01).crb, np.float64)
        np.testing.assert_allclose(noisy, unit * 1e-4, rtol=rtol)
        c = 1e6 if leaves == "float64" else 1e3

        def rescaled(q):
            return linear({"a": q["a"] / c, "b": q["b"]})

        q = {"a": jnp.asarray(1.5 * c, dt), "b": jnp.asarray(-0.5, dt)}
        np.testing.assert_allclose(np.asarray(fim(rescaled, q, scale=None).crb, np.float64),
                                   unit * np.array([c ** 2, 1.0]), rtol=16 * rtol)
        np.testing.assert_allclose(np.asarray(fim(rescaled, q).crb, np.float64),
                                   np.asarray(fim(linear, p).crb, np.float64), rtol=16 * rtol)


@pytest.mark.parametrize("leaves", LEAVES)
def test_fim_names_the_parameters_relative_scaling_zeroed_under_x64(leaves):
    """SYS-027: ``zero_scaled`` names the parameters ``scale="relative"`` found
    at 0.0 (either sign of it), whose column the scaling zeroed; a parameter
    at the smallest normal of the leaves' precision keeps its column and is
    not named; empty under ``scale=None``, which resolves every one."""
    with _x64():
        dt = _dt(leaves)
        tiny = float(np.finfo(dt).tiny)
        params = {"a": jnp.asarray(2.0, dt), "neg": jnp.asarray(-0.0, dt),
                  "tiny": jnp.asarray(tiny, dt), "zero": jnp.asarray(0.0, dt)}

        def identity(p):
            return jnp.stack([p["a"], p["neg"], p["tiny"] / tiny, p["zero"]])

        rel = fim(identity, params)
        assert rel.zero_scaled == ("['neg']", "['zero']"), rel.zero_scaled
        crb = dict(zip(rel.param_names, np.asarray(rel.crb, np.float64)))
        assert np.isinf(crb["['neg']"]) and np.isinf(crb["['zero']"])
        assert np.isfinite(crb["['a']"]) and np.isfinite(crb["['tiny']"])
        assert rel.rank == 2
        # (``tiny``'s absolute column, 1 / tiny, would overflow F: left out.)
        three = {k: v for k, v in params.items() if k != "tiny"}
        absolute = fim(lambda p: jnp.stack([p["a"], p["neg"], p["zero"]]), three, scale=None)
        assert absolute.zero_scaled == () and absolute.rank == 3


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_nominal_policy_table_under_x64(leaves):
    """SYS-028: ``scale="nominal"`` multiplies each column by the width of a
    finite ``bounds`` (whatever the transform), by ``p - lo`` for ``log`` with
    a lower bound, and by ``p`` for ``log`` without one or an identity spec
    with no width -- the last three named in ``value_scaled``; nothing falls
    back to one.  Checked column by column against ``scale=None``'s matrix,
    with widths a float32 leaf could not carry squared (1e-30) in float64."""
    with _x64():
        dt = _dt(leaves)
        values = {"w": 0.25, "lg": 3.0, "lg0": 2.0, "idn": -4.0}
        params = {k: jnp.asarray(v, dt) for k, v in values.items()}
        M = jnp.asarray(np.random.default_rng(3).standard_normal((6, 4)), dt)

        def residual(p):
            return M @ jnp.stack([p[k] for k in sorted(p)])

        width = 1e-30 if leaves == "float64" else 1e-3
        specs = {"w": ParamSpec(bounds=(0.0, width), transform="logit"),
                 "lg": ParamSpec(bounds=(1.0, None), transform="log"),
                 "lg0": ParamSpec(transform="log"),
                 "idn": ParamSpec(bounds=(None, 0.0))}
        params["w"] = jnp.asarray(0.25 * width, dt)
        nom = fim(residual, params, scale="nominal", specs=specs)
        raw = np.asarray(fim(residual, params, scale=None).fim, np.float64)
        names = [n.strip("[]'") for n in nom.param_names]
        col = {"w": width, "lg": 3.0 - 1.0, "lg0": 2.0, "idn": -4.0}
        d = np.array([col[n] for n in names])
        np.testing.assert_allclose(np.asarray(nom.fim, np.float64), raw * np.outer(d, d),
                                   rtol=64 * _eps(leaves), atol=0.0)
        assert nom.value_scaled == ("['idn']", "['lg']", "['lg0']")
        assert nom.zero_scaled == ()


@pytest.mark.parametrize("leaves", LEAVES)
def test_fim_and_fim_core_agree_on_rank_and_infinities_under_x64(leaves):
    """SYS-040: ``fim`` and ``fim_core`` (jitted) agree on ``rank`` and on
    which ``crb`` entries are ``+inf`` over a spread of spectra away from the
    cutoff -- full rank, deficient, a zero-valued parameter under
    ``"relative"`` -- and their finite ``crb`` differ only in the last bits
    (the eps of the leaves' precision, times the condition number)."""
    import functools

    with _x64(), warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionLimitWarning)
        cutoff = _cutoff(leaves, 40)
        dt = _dt(leaves)
        cases = []
        for ratio in (1e-2, 30 * cutoff, cutoff / 30, 0.0):
            rng = np.random.default_rng(7)
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            U = np.linalg.qr(rng.standard_normal((40, 3)))[0]
            A = jnp.asarray((U * np.sqrt([ratio, 0.5, 1.0])) @ q.T, dt)
            params = {f"p{i}": jnp.asarray(1.0 + i, dt) for i in range(3)}
            cases.append((lambda p, A=A: A @ jnp.stack([p["p0"], p["p1"], p["p2"]]),
                          params, {"scale": None}))
        zero = {"a": jnp.asarray(2.0, dt), "b": jnp.asarray(0.0, dt)}
        cases.append((lambda p: jnp.stack([p["a"] + p["b"], p["a"] - p["b"]]), zero, {}))
        for fn, params, kw in cases:
            report = fim(fn, params, **kw)
            core = jax.jit(functools.partial(fim_core, fn, **kw))(params)
            assert int(core.rank) == report.rank
            inf = np.isinf(np.asarray(report.crb))
            assert np.array_equal(np.isinf(np.asarray(core.crb)), inf)
            # "Differ in the last bits": of the inverse, whose rounding is the
            # matrix's eps amplified by its condition number.
            rtol = 64 * _eps(leaves) * float(np.max(np.asarray(report.eigvals))
                                             / np.min(np.asarray(report.eigvals)[-report.rank:]))
            np.testing.assert_allclose(np.asarray(core.crb)[~inf],
                                       np.asarray(report.crb)[~inf], rtol=rtol)


# ---------------------------------------------------------------------------
# The fitters
# ---------------------------------------------------------------------------

#: The minimum of :func:`_bowl`: a closed-form loss over the spring's four
#: constants, in the coordinate each spec optimises in (``log`` for stiffness
#: and mass), so a fit compiles in milliseconds and Adam's path is controlled.
TARGET = {"stiffness": 20.0, "damping": 1.5, "mass": 1.3, "rest_length": 1.2}


def _bowl_residual(p):
    s = p["nodes"]["s"]
    return jnp.stack([jnp.log(s["stiffness"] / TARGET["stiffness"]),
                      s["damping"] - TARGET["damping"],
                      jnp.log(s["mass"] / TARGET["mass"]),
                      2.0 * (s["rest_length"] - TARGET["rest_length"])])


def _bowl(p):
    r = _bowl_residual(p)
    return jnp.sum(r * r)


def _bits(tree):
    return [np.asarray(x).tobytes() for x in jax.tree.leaves(tree)]


def _recorder():
    seen = {}

    def callback(i, loss, params):
        seen[i] = (loss, params)
    return seen, callback


def _only(gm, *keys):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in keys:
        mask["nodes"]["s"][key] = True
    return mask


@pytest.mark.parametrize("leaves", LEAVES)
def test_losses_zero_is_the_starting_points_loss_under_x64(leaves):
    """SYS-050: ``losses[i]`` is the loss before update ``i + 1``, so
    ``losses[0]`` is the start's, bit for bit (the run evaluates the start as
    it went in, not its ``log`` round trip: SYS-071); for ``fit_lm`` the half
    sum of squares."""
    with _x64():
        gm = _spring(leaves)
        mask = _only(gm, "stiffness")

        def loss(p):
            return (p["nodes"]["s"]["stiffness"] - 25.0) ** 2

        res = fit(gm, loss, mask=mask, n_iter=4, lr=0.1)
        assert float(res.losses[0]) == float(loss(gm.params))

        def residual(p):
            return jnp.atleast_1d(p["nodes"]["s"]["stiffness"] - 25.0)

        lm = fit_lm(gm, residual, mask=mask, n_iter=4)
        assert float(lm.losses[0]) == float(0.5 * jnp.sum(residual(gm.params) ** 2))


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_lowest_loss_iterate_is_returned_and_indexed_under_x64(leaves):
    """SYS-051 and SYS-052: Adam overshoots the bowl and climbs back out, so its
    lowest loss is mid-run: ``params`` is that iterate bit for bit (as the
    callback was shown it), ``losses[best_iteration] == best_loss``.  A run
    whose loss never rose returns its last iterate, which the loop never
    evaluated: ``best_iteration == len(losses)``, its loss is not in ``losses``
    and the callback never saw it.  ``n_iter=0``: ``best_loss`` is None."""
    with _x64():
        gm = _spring(leaves)
        seen, callback = _recorder()
        res = fit(gm, _bowl, n_iter=12, lr=0.1, callback=callback, hold_undetermined=False)
        k = res.best_iteration
        assert 0 < k < len(res.losses) - 1, (k, res.losses)
        assert res.losses.dtype == _dt(leaves) or res.losses.dtype == np.float64
        assert res.best_loss == res.losses[k] == res.losses.min()
        assert _bits(res.params) == _bits(seen[k + 1][1])
        seen, callback = _recorder()
        down = fit(gm, _bowl, n_iter=5, lr=0.01, callback=callback, hold_undetermined=False)
        assert np.all(np.diff(down.losses) < 0.0), down.losses
        assert down.best_iteration == len(down.losses) == 5 and sorted(seen) == [1, 2, 3, 4, 5]
        assert down.best_loss < down.losses[-1]
        assert down.best_loss == pytest.approx(float(_bowl(down.params)), rel=8 * _eps(leaves))
        none = fit(gm, _bowl, n_iter=0)
        assert none.best_loss is None and none.best_iteration == 0
        assert _bits(none.params) == _bits(gm.params)


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_tie_and_a_tol_stop_return_the_right_iterate_under_x64(leaves):
    """SYS-052: a tie goes to the later iterate -- a loss that is exactly 1.0
    everywhere with a gradient of 1 returns Adam's last iterate -- and a run
    stopped by ``tol`` returns the iterate that met it."""
    with _x64():
        gm = _spring(leaves)

        def flat(p):
            k = p["nodes"]["s"]["stiffness"]
            return 1.0 + (k - jax.lax.stop_gradient(k))

        seen, callback = _recorder()
        tie = fit(gm, flat, mask=_only(gm, "stiffness"), n_iter=4, lr=0.1, callback=callback)
        assert [float(x) for x in tie.losses] == [1.0] * 4
        assert tie.best_iteration == 4
        assert float(tie.params["nodes"]["s"]["stiffness"]) < float(seen[4][1]["nodes"]["s"]["stiffness"])
        seen, callback = _recorder()
        kw = dict(n_iter=40, lr=0.01, hold_undetermined=False)
        probe = fit(gm, _bowl, callback=callback, **kw)
        tol = 0.8 * float(probe.losses[0])
        res = fit(gm, _bowl, tol=tol, **kw)
        assert res.converged and res.losses[-1] <= tol and np.all(res.losses[:-1] > tol)
        assert res.best_iteration == len(res.losses) - 1 and res.best_loss == res.losses[-1]
        assert _bits(res.params) == _bits(seen[len(res.losses)][1])


@pytest.mark.parametrize("leaves", LEAVES)
def test_converged_is_the_tol_test_alone_for_fit_under_x64(leaves):
    """SYS-053: with the default ``tol=0.0`` ``fit`` never reports converged,
    even at a loss of exactly zero; a positive ``tol`` stops early once the
    loss is at or below it."""
    with _x64():
        gm = _spring(leaves)
        mask = _only(gm, "stiffness")
        zero = fit(gm, lambda p: 0.0 * p["nodes"]["s"]["stiffness"], mask=mask, n_iter=3)
        assert [float(x) for x in zero.losses] == [0.0] * 3 and zero.converged is False
        res = fit(gm, lambda p: (p["nodes"]["s"]["stiffness"] - 25.0) ** 2, mask=mask,
                  n_iter=400, lr=0.1, tol=1e-6)
        assert res.converged and res.n_iter < 400 and res.losses[-1] <= 1e-6


def _mixed_spring(leaves):
    """The spring with constants a float32 cast would move (30.1, 0.1): every
    leaf in float64, or the trainable stiffness and damping in float32 beside
    float64 ones -- one tree, two dtypes, which ``ravel_pytree`` promotes."""
    narrow = np.float32 if leaves == "mixed" else np.float64
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=narrow(30.1), damping=narrow(0.1), mass=np.float64(1.0),
        rest_length=np.float64(1.1), initial_position=np.float64(0.5),
        initial_velocity=np.float64(0.0)))
    gm.compile()
    _cast_state(gm, "float64")
    return gm


@pytest.mark.parametrize("leaves", LEAVES)
def test_every_leaf_no_step_moved_is_the_value_that_went_in_under_x64(leaves):
    """SYS-054: every leaf no step moved comes back bit for bit -- the ones
    outside the mask after a real fit, and every one when the run stopped
    before its first update (``n_iter=0``, a ``tol`` met at the start) --
    including ``log`` leaves, which ``exp(log(p))`` would move by an ulp, and
    float64 values a pass through float32 would round."""
    with _x64():
        gm = _mixed_spring(leaves)
        dtypes = {k: v.dtype for k, v in gm.params["nodes"]["s"].items()}
        assert dtypes["stiffness"] == (np.float32 if leaves == "mixed" else np.float64)
        assert dtypes["rest_length"] == np.float64
        before = _bits(gm.params)
        for kw in ({"n_iter": 0}, {"n_iter": 5, "tol": 1e30}):
            assert _bits(fit(gm, _bowl, **kw).params) == before, kw
            assert _bits(fit_lm(gm, _bowl_residual, **kw).params) == before, kw
        res = fit(gm, _bowl, mask=_only(gm, "damping"), n_iter=5, lr=0.05)
        for key, value in res.params["nodes"]["s"].items():
            same = np.asarray(value).tobytes() == np.asarray(gm.params["nodes"]["s"][key]).tobytes()
            assert same == (key != "damping"), key
            assert value.dtype == dtypes[key], key


@pytest.mark.parametrize("leaves", LEAVES)
def test_log_and_logit_coordinates_stay_inside_when_pushed_hard_under_x64(leaves):
    """SYS-057: the optimiser works in the unconstrained coordinates, so a
    positive parameter cannot cross zero and a bounded one cannot leave its
    interval, and every fitter keeps a log or logit coordinate short of where
    ``exp`` floors or overflows in the leaves' precision (float64's are near
    -745 and 709, float32's near -103 and 88): a linear loss pushes each
    coordinate at its edge with Adam steps of 50 for 40 iterations, and the
    parameter that comes back is inside, finite, and re-unconstrains."""
    with _x64():
        gm = _spring(leaves)
        lo, hi = 2.0, 10.0
        gm.set_param_spec("s", "rest_length", ParamSpec(bounds=(lo, hi), transform="logit"))
        start = _with(gm, leaves, rest_length=6.0)
        mask = _only(gm, "stiffness", "rest_length")
        for direction in (-1.0, 1.0):
            def push(p, d=direction):
                q = p["nodes"]["s"]
                return d * (jnp.log(q["stiffness"]) + 50.0 * q["rest_length"])

            res = fit(gm, push, params=start, mask=mask, n_iter=40, lr=50.0)
            gm.check_params(res.params)
            q = res.params["nodes"]["s"]
            k, length = float(q["stiffness"]), float(q["rest_length"])
            assert q["stiffness"].dtype == _dt(leaves)
            assert 0.0 < k < np.inf and lo < length < hi, (direction, k, length)
            assert (k > 30.0 and length > 6.0) if direction < 0 else (k < 30.0 and length < 6.0)
            u = gm.unconstrain(res.params)["nodes"]["s"]
            assert np.isfinite(float(u["stiffness"])) and np.isfinite(float(u["rest_length"]))


def test_a_clipped_float32_coordinate_comes_back_in_an_x64_process():
    """SYS-058 with mixed dtypes: a float32 identity-transform leaf with
    bounds ``(0, None)``, in an x64 process where the optimiser's vector is
    float64: Adam's step carries it past 0, it is projected back onto the
    bound and climbs back to the interior truth; ``fit_lm`` from far above
    reaches it too, converged."""
    leaves = "mixed"
    truth = 0.05
    with _x64():
        source = _spring(leaves, damping=truth, initial_position=0.2)
        obs = source.run_scan_with_history(100)[1]["s"]["position"]
        gm = _spring(leaves, damping=2.0, initial_position=0.2)
        assert gm.params["nodes"]["s"]["damping"].dtype == np.float32

        def residual(p):
            g = _spring(leaves, initial_position=0.2)   # run_scan stores its final state
            return g.run_scan_with_history(100, params=p)[1]["s"]["position"] - obs

        res = fit(gm, jax.jit(lambda p: 0.5 * jnp.sum(residual(p) ** 2)),
                  mask=_only(gm, "damping"), n_iter=150, lr=0.5)
        assert float(res.params["nodes"]["s"]["damping"]) == pytest.approx(truth, rel=2e-2)
        far = _spring(leaves, damping=4.0, initial_position=0.2)
        lm = fit_lm(far, residual, mask=_only(far, "damping"), n_iter=50)
        assert lm.converged
        assert float(lm.params["nodes"]["s"]["damping"]) == pytest.approx(truth, rel=1e-3)


@pytest.mark.parametrize("leaves", LEAVES)
def test_multiple_shooting_that_walks_away_returns_its_start_and_seed_states_under_x64(leaves):
    """SYS-070: started 0.1% off the truth, Adam's step of about ``lr`` walks
    every update away; the parameters and the window states come back from
    the start, together, bit for bit."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves, 200)
        start = _with(gm, leaves, stiffness=1.001 * 30.0)
        res, ws = fit_multiple_shooting(gm, obs, obs_fn=_position, window=20, params=start,
                                        n_iter=60, lr=0.05, hold_undetermined=False)
        assert res.losses[-1] > 10.0 * res.losses[0], res.losses
        assert res.best_iteration == 0 and res.best_loss == res.losses[0]
        assert _bits(res.params) == _bits(start)
        assert _bits(ws) == _bits(init_window_states(obs, 20))


@pytest.mark.parametrize("leaves", LEAVES)
def test_multiple_shooting_best_loss_is_exactly_the_returned_pairs_under_x64(leaves):
    """SYS-071: without the guard the pair is exact -- the parameters and the
    window states are returned as selected and ``best_loss`` is their loss --
    here for a run that returns its start (SYS-070), whose stiffness, 30.03, a
    ``log`` round trip moves by one float64 ulp; float32 leaves round-trip
    through the float64 optimiser's coordinates exactly.  The run evaluated
    the round trip until the fitters' objectives took the leaves no step
    moved as they went in (``_exact_physical_params``)."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves, 200)
        start = _with(gm, leaves, stiffness=1.001 * 30.0)
        res, ws = fit_multiple_shooting(gm, obs, obs_fn=_position, window=20, params=start,
                                        n_iter=60, lr=0.05, hold_undetermined=False)
        assert res.best_iteration == 0 and _bits(res.params) == _bits(start)
        again = float(windowed_loss(gm, res.params, obs, obs_fn=_position, window=20,
                                    window_states=ws, continuity_weight=1.0))
        assert again == float(res.best_loss), (again, res.best_loss)


@pytest.mark.parametrize("leaves", LEAVES)
def test_multiple_shooting_best_loss_is_the_returned_pairs_with_the_guard_under_x64(leaves):
    """SYS-071 with the guard: when it holds the scale direction, ``best_loss``
    is the loss of the returned pair within the guard's tolerance, ``2**10 *
    eps`` relative in the leaves' precision."""
    with _x64():
        gm = _spring(leaves)
        gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
        gm.set_param_spec("s", "mass", ParamSpec(bounds=(0.0, None), transform="log"))
        obs = _record(gm, leaves)
        start = _with(gm, leaves, stiffness=36.0)
        res, ws = fit_multiple_shooting(gm, obs, obs_fn=_position, window=WINDOW, params=start,
                                        mask=_only(gm, "stiffness", "damping", "mass"),
                                        n_iter=60, lr=0.05)
        again = float(windowed_loss(gm, res.params, obs, obs_fn=_position, window=WINDOW,
                                    window_states=ws, continuity_weight=1.0))
        assert res.excited_rank == 2 and res.undetermined_drift > 0.0, res
        assert again == pytest.approx(res.best_loss, rel=2.0 ** 10 * _eps(leaves))


@pytest.mark.parametrize("leaves", LEAVES)
def test_no_converged_test_fires_while_a_bound_coordinate_could_descend_under_x64(
        leaves, monkeypatch):
    """SYS-082: while a coordinate on its bound could lower the loss by moving
    into the range, no proposal converges ``fit_lm``, however small -- pinned
    by making every iterate look that way; the same run converges without it."""
    with _x64():
        source = _spring(leaves, damping=0.05, initial_position=0.2)
        obs = source.run_scan_with_history(60)[1]["s"]["position"]

        def residual(p):
            g = _spring(leaves, initial_position=0.2)
            return g.run_scan_with_history(60, params=p)[1]["s"]["position"] - obs

        gm = _spring(leaves, damping=0.5, initial_position=0.2)
        free = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=12)
        assert free.converged
        monkeypatch.setattr(sysid._CoordinateBounds, "inward_descent",  # noqa: SLF001
                            lambda self, *args, **kwargs: True)
        held = fit_lm(gm, residual, mask=_only(gm, "damping"), n_iter=12)
        assert not held.converged


_A = np.random.default_rng(0).normal(size=(30, 3))
_B = _A @ np.array([40.0, 3.0, 1.2])


@pytest.mark.parametrize("scale", [1e-30, 1e-20, 1e-10, 1e-4, 1.0, 1e4, 1e10, 1e20, 1e30])
def test_fit_lm_answers_the_same_for_every_scaling_with_float32_leaves_under_x64(scale):
    """SYS-083 with mixed dtypes: the Marquardt floor is eps times each column's
    own ``diag(J^T J)``, so ``r = s (A x - b)`` has the optimum of ``A x - b``
    for every ``s`` -- with float32 leaves and a float32 residual in an x64
    process, where the solve's vector is float64."""
    leaves = "mixed"
    with _x64():
        gm = _spring(leaves)
        ft = jnp.float32

        def residual(p):
            q = p["nodes"]["s"]
            x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]])
            return scale * (jnp.asarray(_A, ft) @ x - jnp.asarray(_B, ft))

        res = fit_lm(gm, residual, mask=_only(gm, "stiffness", "damping", "rest_length"),
                     n_iter=50)
        q = res.params["nodes"]["s"]
        assert q["stiffness"].dtype == np.float32
        assert res.converged and res.n_iter <= 6, (res.converged, res.n_iter)
        np.testing.assert_allclose([float(q["stiffness"]), float(q["damping"]),
                                    float(q["rest_length"])], [40.0, 3.0, 1.2], rtol=2e-6)


#: A noisy right-hand side, so the fit's residual at its optimum is the
#: noise (``1e-3``) and not its own rounding.
_B_NOISY = _B + 1e-3 * np.random.default_rng(7).normal(size=_B.shape)


@pytest.mark.parametrize("scale", [1e-157, 1e-150, 1e-100, 1e100, 1e150])
def test_fit_lm_answers_the_same_for_every_scaling_of_the_residual_under_x64(scale):
    """SYS-083 at float64's own range: ``r * r`` flushes below ``1e-154`` and
    overflows above ``1e154``, and ``JᵀJ`` and ``Jᵀr`` flush below about
    ``1e-154`` too -- at ``1e-157`` the bare products gave ``converged=True``
    33% off.  Framed, the fit is the unscaled one (to ``1e-9`` where the loss
    at the optimum is a float64 subnormal, ``1e-319``, and its comparisons are
    that coarse).  The range ends where ``0.5 ||r||²`` itself leaves float64
    at some iterate: above about ``1e153`` it overflows (refused as
    non-finite), and below the smallest subnormal it reads 0.0, which
    ``fit_lm`` does not call converged
    (``test_a_loss_that_underflows_float64_is_never_converged``)."""
    optimum = np.linalg.lstsq(_A, _B_NOISY, rcond=None)[0]
    with _x64():
        gm = _spring("float64")

        def residual(p):
            q = p["nodes"]["s"]
            x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]])
            return scale * (jnp.asarray(_A) @ x - jnp.asarray(_B_NOISY))

        res = fit_lm(gm, residual, mask=_only(gm, "stiffness", "damping", "rest_length"),
                     n_iter=50)
        q = res.params["nodes"]["s"]
    assert res.converged and res.n_iter <= 8, (res.converged, res.n_iter)
    np.testing.assert_allclose([float(q["stiffness"]), float(q["damping"]),
                                float(q["rest_length"])], optimum, rtol=1e-9)
    assert 0.0 < res.best_loss < res.losses[0] < np.inf


@pytest.mark.parametrize("leaves, unit", [("float64", 1e-160), ("float64", 1e160),
                                          ("mixed", 1e-30), ("mixed", 1e30)])
def test_fit_lm_answers_the_same_at_every_parameter_scale_under_x64(leaves, unit):
    """SYS-084/SYS-088 at each precision's range: damping, an identity
    parameter, at a natural scale near either end of it -- the same answer
    as at unit 1, with the defaults."""
    with _x64():
        base, damping, stiffness = _units_fit(leaves, 1.0)
        res, damping_u, stiffness_u = _units_fit(leaves, unit)
    rel = 1e-9 if leaves == "float64" else 1e-4
    assert base.converged and res.converged, (base.n_iter, res.n_iter)
    assert damping_u == pytest.approx(damping, rel=rel), (damping_u, damping)
    assert stiffness_u == pytest.approx(stiffness, rel=rel), (stiffness_u, stiffness)


def _units_fit(leaves, unit, **kw):
    """The linear problem over (stiffness, damping, rest_length) with damping
    (identity, bounds ``(0, None)``) measured in units ``unit``, from the same
    physical start, in the leaves' dtype: ``(result, damping, stiffness)``."""
    gm = _spring(leaves, initial_position=0.2)
    rng = np.random.default_rng(0).normal(size=(12, 3))
    b = rng @ np.array([40.0, 3.0, 1.2])
    dt = _dt(leaves)
    u = jnp.asarray([1.0, unit, 1.0], dt)

    def residual(p):
        q = p["nodes"]["s"]
        x = jnp.stack([q["stiffness"], q["damping"], q["rest_length"]]) * u
        return jnp.asarray(rng, dt) @ x - jnp.asarray(b, dt)

    start = _with(gm, leaves, damping=2.0 / unit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a declined hold says so
        res = fit_lm(gm, residual, params=start,
                     mask=_only(gm, "stiffness", "damping", "rest_length"), n_iter=50, **kw)
    q = res.params["nodes"]["s"]
    return res, float(q["damping"]) * unit, float(q["stiffness"])


@pytest.mark.parametrize("unit", [1e-5, 1e-3, 1e4, 1e6])
def test_the_marquardt_step_does_not_depend_on_the_units_of_a_float32_parameter_under_x64(unit):
    """SYS-084 with mixed dtypes: the same problem with float32 leaves in an x64
    process, damping in other units, reaches the same answer in about as many
    iterations (the guard off -- it is SYS-063's)."""
    with _x64():
        base, damping, stiffness = _units_fit("mixed", 1.0, hold_undetermined=False)
        res, damping_u, stiffness_u = _units_fit("mixed", unit, hold_undetermined=False)
    assert base.params["nodes"]["s"]["damping"].dtype == np.float32
    assert base.converged and res.converged, (base, res)
    assert damping == pytest.approx(3.0, rel=1e-5) and stiffness == pytest.approx(40.0, rel=1e-5)
    assert damping_u == pytest.approx(3.0, rel=1e-4) and stiffness_u == pytest.approx(40.0, rel=1e-4)
    assert abs(res.n_iter - base.n_iter) <= 2, (base.n_iter, res.n_iter)


@pytest.mark.parametrize("unit", [1e-5, 1e-3, 1e6])
@pytest.mark.parametrize("leaves", LEAVES)
def test_fit_lm_answers_the_same_in_any_units_with_its_defaults_under_x64(leaves, unit):
    """SYS-088: the answer depends neither on the residual's units nor on any
    parameter's -- ``fit_lm`` as called by default, guard included, in float64
    (to 1e-9) and with float32 leaves (to 1e-4)."""
    with _x64():
        _, damping, stiffness = _units_fit(leaves, 1.0)
        _, damping_u, stiffness_u = _units_fit(leaves, unit)
    rel = 1e-9 if leaves == "float64" else 1e-4
    assert damping == pytest.approx(3.0, rel=rel)
    assert damping_u == pytest.approx(damping, rel=rel), (damping_u, damping)
    assert stiffness_u == pytest.approx(stiffness, rel=rel), (stiffness_u, stiffness)


@pytest.mark.parametrize("leaves", LEAVES)
def test_fit_lm_reports_the_iterate_each_ending_returns_under_x64(leaves):
    """SYS-085: ``fit_lm`` returns its last iterate: ``best_iteration`` is
    ``len(losses)`` when it ended on an accepted step (budget, or a step within
    ``step_tol``), ``len(losses) - 1`` when it ended on ``tol``; ``best_loss``
    is the loss its acceptance test computed for the returned iterate."""
    with _x64():
        gm = _spring(leaves)
        start = _with(gm, leaves, stiffness=60.0, damping=4.0)
        res = fit_lm(gm, _bowl_residual, params=start, n_iter=2)
        assert len(res.losses) == 2 and not res.converged and res.best_iteration == 2
        assert res.best_loss < res.losses[-1] < res.losses[0]
        r = np.asarray(_bowl_residual(res.params), np.float64)
        assert 0.5 * float(r @ r) == pytest.approx(res.best_loss, rel=1e-4,
                                                    abs=64 * _eps(leaves) * res.losses[0])
        by_tol = fit_lm(gm, _bowl_residual, params=start, n_iter=10, tol=1e6)
        assert by_tol.converged and by_tol.best_iteration == 0 == len(by_tol.losses) - 1
        by_step = fit_lm(gm, _bowl_residual, params=start, n_iter=50, step_tol=1e-3)
        assert by_step.converged and by_step.n_iter < 50
        assert by_step.best_iteration == len(by_step.losses)


# ---------------------------------------------------------------------------
# The identifiability guard
# ---------------------------------------------------------------------------

#: The flat problem: the data reads ``d = log k - 2 log m`` alone, so
#: ``s = 2 log k + log m`` is undetermined and the guard holds it at its
#: start; a bump in ``s`` the gradient cannot see makes that hold raise the
#: loss, and it is declined.
_SHIFT, _WIDTH = 0.5, 1e-3


def _d(p):
    s = p["nodes"]["s"]
    return jnp.log(s["stiffness"]) - 2.0 * jnp.log(s["mass"])


def _s(p):
    s = p["nodes"]["s"]
    return 2.0 * jnp.log(s["stiffness"]) + jnp.log(s["mass"])


def _flat_problem(leaves, fitter, bumped):
    gm = _spring(leaves)
    d0, s0 = float(_d(gm.params)), float(_s(gm.params))

    def residual(p):
        rows = [_d(p) - (d0 - _SHIFT)]
        if bumped:
            rows.append(jax.lax.stop_gradient(jnp.exp(-((_s(p) - s0) / _WIDTH) ** 2)))
        return jnp.stack(rows)

    mask = _only(gm, "stiffness", "mass")
    if fitter == "fit":
        def call(hold):
            return fit(gm, lambda p: jnp.sum(residual(p) ** 2), mask=mask, n_iter=60,
                       lr=0.05, hold_undetermined=hold)
    else:
        def call(hold):
            return fit_lm(gm, residual, mask=mask, n_iter=10, hold_undetermined=hold)
    return s0, call


@pytest.mark.parametrize("fitter", ["fit", "fit_lm"])
@pytest.mark.parametrize("leaves", LEAVES)
def test_a_hold_that_would_raise_the_loss_is_declined_under_x64(leaves, fitter):
    """SYS-063: a hold that would raise the loss by more than ``2**10 * eps``
    relative is declined -- ``hold_declined`` is True, a RuntimeWarning gives
    both losses and the selected iterate comes back bit for bit -- and the
    same hold on the flat problem without the bump is made, putting the
    undetermined combination back at its start."""
    with _x64():
        s0, call = _flat_problem(leaves, fitter, bumped=True)
        with pytest.warns(RuntimeWarning, match="would raise the loss"):
            res = call(True)
        raw = call(False)
        assert res.hold_declined is True and res.excited_rank == 1, res
        assert abs(float(_s(res.params)) - s0) > 10 * _WIDTH
        assert _bits(res.params) == _bits(raw.params)
        assert (res.best_iteration, res.best_loss) == (raw.best_iteration, raw.best_loss)
        s0, call = _flat_problem(leaves, fitter, bumped=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            made = call(True)
        assert made.hold_declined is False and made.excited_rank == 1, made
        assert abs(float(_s(made.params)) - s0) <= 64 * _eps(leaves) * max(1.0, abs(s0))


def _scale_problem(leaves, *, all_log):
    """The spring's ``(k, c, m)`` from noisy position data, ``rest_length``
    frozen; ``all_log`` puts damping on ``log`` too, so the scale direction
    is fixed in the optimiser's coordinates (the shipped identity damping
    makes it rotate as ``c`` moves)."""
    gm = _spring(leaves)
    gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False))
    if all_log:
        gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), transform="log"))
    obs = _record(gm, leaves, 120)
    draw = np.random.default_rng(20260920).normal(0.0, 0.02, size=obs["s"]["position"].shape)
    obs["s"]["position"] = obs["s"]["position"] + jnp.asarray(draw, _dt(leaves))
    loss = jax.jit(lambda p: windowed_loss(gm, p, obs, obs_fn=_position, window=20))
    return gm, loss, _with(gm, leaves, stiffness=45.0, damping=3.0)


@pytest.mark.parametrize("leaves", LEAVES)
def test_excited_rank_none_means_the_question_was_not_answered_under_x64(leaves, monkeypatch):
    """SYS-065: ``excited_rank is None`` means the question was not answered:
    with ``hold_undetermined=False``, above the parameter cap (lowered here),
    with fewer iterations than coordinates, and when every gradient is zero;
    ``hold_undetermined`` must be a bool."""
    with _x64():
        gm, loss, start = _scale_problem(leaves, all_log=True)
        kw = dict(params=start, lr=0.2, notify_every=0)
        for res in (fit(gm, loss, n_iter=20, hold_undetermined=False, **kw),
                    fit(gm, loss, n_iter=2, **kw),
                    fit(gm, lambda p: 0.0 * _bowl(p), n_iter=20, **kw)):
            assert res.excited_rank is None, res
            assert res.undetermined_drift is None and res.hold_declined is None, res
        with monkeypatch.context() as m:
            m.setattr(sysid, "_EXCITATION_MAX_PARAMS", 2)
            capped = fit(gm, loss, n_iter=20, **kw)
        assert capped.excited_rank is None and capped.hold_declined is None
        assert fit(gm, loss, n_iter=20, **kw).excited_rank is not None   # it does answer
        with pytest.raises((TypeError, ValueError), match="hold_undetermined"):
            fit(gm, loss, n_iter=2, hold_undetermined=1, **kw)


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_rotating_null_direction_is_reported_as_full_rank_under_x64(leaves):
    """SYS-066: with the shipped identity-transform damping the scale
    direction rotates in the optimiser's coordinates as ``c`` moves, the
    accumulated matrix has full rank, and the guard reports full
    ``excited_rank`` and does nothing -- ``undetermined_drift`` exactly 0.0,
    the parameters the unguarded run's."""
    with _x64():
        gm, loss, start = _scale_problem(leaves, all_log=False)
        kw = dict(params=start, n_iter=200, lr=0.2, notify_every=0)
        held = fit(gm, loss, **kw)
        raw = fit(gm, loss, hold_undetermined=False, **kw)
        assert held.excited_rank == 3, held.excited_rank
        assert held.undetermined_drift == 0.0 and held.hold_declined is False
        assert _bits(held.params) == _bits(raw.params)


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_weakly_identified_direction_is_kept_not_held_under_x64(leaves):
    """SYS-066: the cutoff is numerical, so a merely weakly identified
    direction is kept: full ``excited_rank``, nothing held, the unguarded
    iterate.  The data reads it at 1e-3 of the other direction, and in
    float64 at 1e-6 as well -- 1e-12 in ``J^T J``, far above float64's
    resolution and below float32's, where holding it is the guard's job."""
    with _x64():
        gm = _spring(leaves)
        mask = _only(gm, "stiffness", "mass")
        for weak in ((1e-3, 1e-6) if leaves == "float64" else (1e-3,)):
            def residual(p, w=weak):
                return jnp.stack([_d(p) - 0.3, w * (_s(p) - 7.0)])

            held = fit_lm(gm, residual, mask=mask, n_iter=8)
            raw = fit_lm(gm, residual, mask=mask, n_iter=8, hold_undetermined=False)
            assert held.excited_rank == 2 and held.undetermined_drift == 0.0, (weak, held)
            assert _bits(held.params) == _bits(raw.params)


@pytest.mark.parametrize("leaves", LEAVES)
def test_the_guard_reads_the_iterate_the_fitter_returns_under_x64(leaves):
    """SYS-069: Adam on the flat problem with a step that overshoots ``d`` and
    climbs back out selects a mid-run iterate; the guard holds ``s`` at its
    start from *that* iterate, so the returned ``d`` is the selected iterate's
    -- the one the callback was shown -- and not the last one's."""
    with _x64():
        gm = _spring(leaves)
        d0, s0 = float(_d(gm.params)), float(_s(gm.params))
        seen, callback = _recorder()
        res = fit(gm, lambda p: (_d(p) - (d0 - _SHIFT)) ** 2, mask=_only(gm, "stiffness", "mass"),
                  n_iter=40, lr=0.15, callback=callback)
        k = res.best_iteration
        assert 0 < k < len(res.losses), (k, res.losses)
        assert res.excited_rank == 1 and res.hold_declined is False, res
        selected, last = seen[k + 1][1], seen[len(res.losses)][1]
        assert abs(float(_d(selected)) - float(_d(last))) > 1e-3
        tol = 64 * _eps(leaves) * max(1.0, abs(d0), abs(s0))
        assert abs(float(_d(res.params)) - float(_d(selected))) <= tol
        assert abs(float(_s(res.params)) - s0) <= tol


# ---------------------------------------------------------------------------
# Truth recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leaves", LEAVES)
def test_fit_recovers_stiffness_and_damping_with_mass_frozen_under_x64(leaves):
    """SYS-087: ``fit`` recovers the spring's stiffness and damping from 45 and
    5 with mass frozen by its ``ParamSpec`` -- physical, inside bounds: the
    frozen leaves come back bit for bit, stiffness never crosses zero, and
    with ``tol=0`` the run uses its budget, not converged."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves, 200)
        gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
        loss = jax.jit(lambda p: windowed_loss(gm, p, obs, obs_fn=_position, window=20))
        seen = []
        start = _with(gm, leaves, stiffness=45.0, damping=5.0)
        res = fit(gm, loss, params=start, n_iter=300, lr=0.1,
                  callback=lambda i, l, p: seen.append(float(p["nodes"]["s"]["stiffness"])))
        s = res.params["nodes"]["s"]
        assert abs(float(s["stiffness"]) - 30.0) / 30.0 < 0.05, float(s["stiffness"])
        assert abs(float(s["damping"]) - 2.0) / 2.0 < 0.10, float(s["damping"])
        for key in ("mass", "initial_position"):              # frozen by their specs
            assert np.asarray(s[key]).tobytes() == np.asarray(start["nodes"]["s"][key]).tobytes()
        assert res.losses[-1] < 1e-3 * res.losses[0]
        assert res.n_iter == 300 and not res.converged and min(seen) > 0.0


@pytest.mark.parametrize("leaves", LEAVES)
def test_multiple_shooting_recovers_from_noisy_window_starts_under_x64(leaves):
    """SYS-072: with noise on the observations teacher forcing seeds every
    window with a noisy state; multiple shooting lets the starts move, and
    recovers the stiffness to 5% from 1.3 times it."""
    with _x64():
        gm = _spring(leaves)
        obs = _record(gm, leaves, 200)
        rng = np.random.default_rng(0)
        noisy = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0, 0.01, x.shape), x.dtype), obs)
        gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
        gm.set_param_spec("s", "rest_length", ParamSpec(trainable=False))
        res, ws = fit_multiple_shooting(
            gm, noisy, obs_fn=_position, window=20,
            params=_with(gm, leaves, stiffness=39.0, damping=3.0), continuity_weight=1.0,
            n_iter=250, lr=0.1, lr_states=0.01)
        k = float(res.params["nodes"]["s"]["stiffness"])
        assert abs(k - 30.0) / 30.0 < 0.05, k
        assert ws["s"]["position"].shape == (10,) and ws["s"]["position"].dtype == _dt(leaves)
        assert res.losses[-1] < res.losses[0]


# ---------------------------------------------------------------------------
# fit_lm with float32 constants in an x64 graph: no creep (MADD-ANO-174)
# ---------------------------------------------------------------------------


class _AB(SimulationNode):
    """Carries the constants ``a`` and ``b``; the residuals read them."""

    def __init__(self, name, specs=None, **params):
        self._specs = specs or {}
        super().__init__(name, 0.01, **params)

    def param_specs(self):
        return {**super().param_specs(), "a": ParamSpec(), "b": ParamSpec(), **self._specs}

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + 0.0 * (p["a"] + p["b"])}


def _ab_fit(a, b, A, y, specs=None, n_iter=50):
    gm = GraphManager()
    gm.add_node(_AB("h", specs, a=a, b=b))
    gm.compile()
    Aj, yj = jnp.asarray(A), jnp.asarray(y)
    return fit_lm(gm, lambda p: Aj @ jnp.stack([p["nodes"]["h"]["a"].astype(jnp.float64),
                                                p["nodes"]["h"]["b"]]) - yj, n_iter=n_iter)


def test_float32_constants_in_an_x64_graph_converge_as_fast_as_float64_ones():
    """SYS-084/SYS-086's "a few iterations" in the mixed domain.  The float32
    leaf's coordinate was float64 (``ravel_pytree`` promotes it): a step of
    less than one float32 ulp left the leaf where it was, the joint step's
    compensation in ``b`` raised the loss, candidates were rejected and the
    coordinate crept by ~1e-11 an iteration -- 25-39 iterations against 5-6,
    and a truth on a bound unconverged after 100
    (audit_040_p4_10/fmu-sysid/repro_fit_lm_mixed_dtype_creep.py).  The
    coordinate now stays on its leaf's grid and a move that rounds away is
    held while the others are solved again."""
    rng = np.random.default_rng(5)
    A = rng.normal(size=(30, 2))
    with _x64():
        for _ in range(6):
            truth = rng.uniform(0.2, 1.8, size=2)
            y = A @ truth
            wide = _ab_fit(np.float64(1.0), np.float64(1.0), A, y)
            narrow = _ab_fit(np.float32(1.0), np.float64(1.0), A, y)
            assert wide.converged and narrow.converged, (wide.n_iter, narrow.n_iter)
            assert narrow.n_iter <= wide.n_iter + 2, (wide.n_iter, narrow.n_iter)
            a = narrow.params["nodes"]["h"]["a"]
            assert a.dtype == jnp.float32
            assert float(a) == pytest.approx(truth[0], rel=1e-6)
        # The truth exactly on a bound of the float32 leaf.
        rng1 = np.random.default_rng(1)
        A1 = rng1.normal(size=(30, 2))
        res = _ab_fit(np.float32(1.0), np.float64(1.0), A1, A1 @ np.array([0.1, 0.7]),
                      specs={"a": ParamSpec(bounds=(0.1, 2.0))}, n_iter=100)
        assert res.converged and res.n_iter <= 10, (res.n_iter, res.best_loss)
        assert float(res.params["nodes"]["h"]["a"]) == float(np.float32(0.1))


def test_a_float32_leafs_coordinate_stays_on_its_grid():
    """The helper: only the narrower leaves' coordinates are rounded, to
    exactly their leaf's dtype, and nothing at all when no leaf is narrower."""
    from maddening.sysid import _leaf_grid

    with _x64():
        tree = {"a": jnp.asarray(1.0, jnp.float32), "b": jnp.asarray(1.0, jnp.float64),
                "c": jnp.asarray([1.0, 2.0], jnp.float32)}
        narrow, to_grid = _leaf_grid(tree, np.arange(4), jnp.float64)
        assert narrow.tolist() == [True, False, True, True]
        theta = jnp.asarray([0.1, 0.1, 0.2, 0.3], jnp.float64)
        got = np.asarray(to_grid(theta))
        assert got[1] == 0.1
        assert got[[0, 2, 3]].tolist() == [float(np.float32(v)) for v in (0.1, 0.2, 0.3)]
        wide, identity = _leaf_grid({"b": tree["b"]}, np.arange(1), jnp.float64)
        assert not wide.any()
        assert np.asarray(identity(theta[:1])).tolist() == [0.1]


@pytest.mark.parametrize("leaves", LEAVES)
def test_fit_recovers_the_truth_when_its_gradient_flushes_under_x64(leaves):
    """SYS-132 at each precision: a loss so small that its gradient's
    products flush -- ``0.5 * ||s * r||²`` with ``s = 1e-170`` for float64
    leaves, ``1e-20`` for float32 ones in the x64 process -- is fitted as at
    scale one, its gradients taken with a power-of-two cotangent."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 2))
    y = A @ np.array([0.3, 0.7])
    scale = 1e-170 if leaves == "float64" else 1e-20
    with _x64():
        dt = _dt(leaves)
        results = []
        for s in (1.0, scale):
            gm = GraphManager()
            gm.add_node(_AB("h", a=dt(1.0), b=dt(1.0)))
            gm.compile()
            Aj, yj = jnp.asarray(A, dt), jnp.asarray(y, dt)
            sj = jnp.asarray(s, dt)

            def loss(p, Aj=Aj, yj=yj, sj=sj):
                q = p["nodes"]["h"]
                return 0.5 * jnp.sum((sj * (Aj @ jnp.stack([q["a"], q["b"]]) - yj)) ** 2)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # a flushed loss says so
                results.append(fit(gm, jax.jit(loss), n_iter=300, lr=0.05))
    base, small = results
    for key, truth in (("a", 0.3), ("b", 0.7)):
        got = float(small.params["nodes"]["h"][key])
        assert got == pytest.approx(float(base.params["nodes"]["h"][key]), rel=1e-4), key
        assert got == pytest.approx(truth, rel=1e-3), key
