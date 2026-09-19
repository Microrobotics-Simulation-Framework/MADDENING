"""Large-coupling-group IFT solver — matrix-free lineax backward.

The IFT backward used to build a dense ``(I - dF/dx)`` Jacobian via
``jax.jacrev`` and solve it with ``jnp.linalg.solve``.  That is
O(N^2) memory + O(N^3) compute and OOMs at compile time for realistic
coupled-fluid groups (thousands of state elements).

The current backward uses ``lineax.linear_solve`` driving a
``FunctionLinearOperator`` whose matvec is the F-vjp at the fixed
point.  No Jacobian is materialised — memory is O(N) and each matvec
is one F-vjp.

These tests exercise that path:

1. **Small chain (N=20, ~40 floats):** forward and backward parity
   against ``solver='fori'`` at tight tolerance.
2. **Large chain (N=120, ~240 floats):** the backward completes
   without OOM, and the gradient agrees with finite differences on
   a small random subset of input entries.
"""

from __future__ import annotations

import os

# Force CPU for these graphs — much faster than warming up CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


# ----------------------------------------------------------------------
# Chain fixture: N spring-damper nodes, head-to-tail coupled, all in a
# single coupling group.  Each node has 2 floats of state (position,
# velocity), so total group state size is 2*N.
# ----------------------------------------------------------------------


def _make_chain_gm(
    n_nodes: int,
    solver: str,
    *,
    dt: float = 0.001,
    max_iters: int = 30,
    tol: float = 1e-8,
    init_positions: list[float] | None = None,
    linear_solver: str = "gmres",
) -> GraphManager:
    """Build a coupling group of ``n_nodes`` chained spring-dampers.

    Each node reads its left neighbour's position as anchor; the
    leftmost node's anchor is the rightmost node's position so the
    cycle closes the coupling group.
    """
    gm = GraphManager()
    names = [f"s{i}" for i in range(n_nodes)]
    if init_positions is None:
        init_positions = [float(i) * 0.5 for i in range(n_nodes)]
    for i, nm in enumerate(names):
        node = SpringDamperNode(
            name=nm,
            timestep=dt,
            stiffness=50.0,
            damping=1.0,
            mass=1.0,
            rest_length=0.5,
            initial_position=init_positions[i],
        )
        gm.add_node(node)
    # Chain: each node reads the previous node's position; close the cycle
    # by feeding the last node's position back to the first.
    for i in range(n_nodes):
        src = names[(i - 1) % n_nodes]
        tgt = names[i]
        gm.add_edge(src, tgt, "position", "anchor_position")
    gm.add_coupling_group(
        names,
        max_iterations=max_iters,
        tolerance=tol,
        acceleration="none",
        solver=solver,
        linear_solver=linear_solver,
    )
    return gm


# ----------------------------------------------------------------------
# 1. Small chain — forward + backward parity vs fori
# ----------------------------------------------------------------------


def test_small_chain_forward_parity():
    """N=20 chain: ``solver='ift'`` matches ``solver='fori'`` forward."""
    n = 20
    gm_fori = _make_chain_gm(n, "fori")
    gm_ift = _make_chain_gm(n, "ift")

    s_fori = gm_fori.step()
    s_ift = gm_ift.step()
    for nm in [f"s{i}" for i in range(n)]:
        for fld, v_fori in s_fori[nm].items():
            v_ift = s_ift[nm][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"forward parity failed on {nm}.{fld}: "
                f"fori={v_fori}, ift={v_ift}"
            )


def _loss_chain(state, names):
    return sum(jnp.sum(state[n]["position"] ** 2) for n in names)


def _grad_through_compiled_step(gm: GraphManager, perturbed_node: str):
    """d(loss)/d(initial_position of ``perturbed_node``)."""
    _ = gm.step()  # warm up to populate _compiled_step
    compiled = gm._compiled_step
    assert compiled is not None
    initial_state = gm._state
    names = [k for k in initial_state.keys() if k != "_meta"]

    def loss_of_pert(pos):
        state = {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        state[perturbed_node] = dict(state[perturbed_node])
        state[perturbed_node]["position"] = pos
        new_state = compiled(state, {})
        return _loss_chain(new_state, names)

    p0 = initial_state[perturbed_node]["position"]
    return jax.grad(loss_of_pert)(p0)


def test_small_chain_backward_parity():
    """N=20 chain: jax.grad through jitted step agrees fori vs ift."""
    n = 20
    gm_fori = _make_chain_gm(n, "fori")
    gm_ift = _make_chain_gm(n, "ift")

    # Perturb a middle node to exercise the full adjoint propagation.
    perturbed = f"s{n // 2}"
    g_fori = _grad_through_compiled_step(gm_fori, perturbed)
    g_ift = _grad_through_compiled_step(gm_ift, perturbed)
    assert jnp.allclose(g_fori, g_ift, atol=1e-3, rtol=1e-3), (
        f"backward parity failed: fori={g_fori}, ift={g_ift}"
    )


# ----------------------------------------------------------------------
# 2. Large chain — backward completes without OOM, FD-sampled gradient
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_large_chain_backward_finite_diff():
    """N=120 chain (~240 floats of group state): IFT backward via
    lineax completes and the gradient agrees with finite differences
    on a few sampled input entries.

    The whole point of the lineax swap is to make this feasible — the
    dense ``jacrev + solve`` path would build a 240x240 Jacobian inside
    the coupling group and OOM under more realistic per-node state
    sizes.  Even at N=120 the matrix-free path stays light.

    N was reduced from 250 to 120 (2026-06-12): at 250 the backward
    compile graph exceeded the ~7 GB ``ubuntu-latest`` runner and the
    scheduled slow lane was OOM-killed.  120 keeps the dense-vs-matrix-
    free contrast meaningful while fitting the hosted runner.

    Initial positions are kept small (0.05 + 0.01*i) so the central
    finite difference ``(L(p+eps) - L(p-eps)) / (2 eps)`` of the
    sum-of-squared-positions loss does not lose float32 precision
    against the perturbation magnitude.
    """
    n = 120
    init_positions = [0.05 + 0.01 * i for i in range(n)]
    gm = _make_chain_gm(
        n, "ift", max_iters=40, tol=1e-7, init_positions=init_positions
    )
    _ = gm.step()
    compiled = gm._compiled_step
    assert compiled is not None
    initial_state = gm._state
    names = [k for k in initial_state.keys() if k != "_meta"]

    # Build a single loss function over a length-N vector of positions
    # so we can use jax.grad once and index into the result.  Building
    # the grad inside a per-node loop re-traces ``compiled`` every
    # iteration and pays the (~minute-scale) trace+compile cost N times.
    base_positions = jnp.array(
        [float(initial_state[f"s{i}"]["position"]) for i in range(n)],
        dtype=jnp.float32,
    )
    base_velocities = jnp.array(
        [float(initial_state[f"s{i}"]["velocity"]) for i in range(n)],
        dtype=jnp.float32,
    )

    def loss_fn(positions):
        state = {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        for i in range(n):
            state[f"s{i}"] = dict(state[f"s{i}"])
            state[f"s{i}"]["position"] = positions[i]
            state[f"s{i}"]["velocity"] = base_velocities[i]
        new_state = compiled(state, {})
        return _loss_chain(new_state, names)

    grad_fn = jax.jit(jax.grad(loss_fn))
    g_full = grad_fn(base_positions)
    assert jnp.all(jnp.isfinite(g_full)), "non-finite entries in gradient"

    # Random sample of 5 nodes (deterministic seed for reproducibility).
    rng = jax.random.PRNGKey(0)
    idxs = jax.random.choice(
        rng, jnp.arange(n), shape=(5,), replace=False
    )
    loss_jit = jax.jit(loss_fn)
    eps = 1e-3
    for idx in idxs:
        i = int(idx)
        g_an = g_full[i]
        p_plus = base_positions.at[i].add(eps)
        p_minus = base_positions.at[i].add(-eps)
        lp = loss_jit(p_plus)
        lm = loss_jit(p_minus)
        g_fd = (lp - lm) / (2.0 * eps)
        # Tolerance: float32 central FD on a sum-of-squares loss with
        # eps=1e-3 against positions ~0.05-1.25 gives ~1e-3 absolute
        # noise; 5% relative + 5e-3 absolute is the right band to
        # catch a structurally wrong gradient without flagging the
        # expected float32 truncation error.
        assert jnp.allclose(g_an, g_fd, atol=5e-3, rtol=5e-2), (
            f"gradient mismatch at s{i}: analytic={g_an}, fd={g_fd}"
        )


# ----------------------------------------------------------------------
# 3. linear_solver dispatch — gmres / bicgstab / dense parity
# ----------------------------------------------------------------------


def test_bicgstab_known_breakdown_with_function_operator():
    """Pins the lineax 0.0.7 BiCGStab limitation that motivated
    keeping ``"bicgstab"`` out of the CouplingGroup ``linear_solver``
    Literal.

    Investigation (2026-05-30): the BiCGStab dispatch arm in
    ``_ift_linear_solve`` is wired correctly, but the underlying
    ``lineax.BiCGStab`` returns NaN whenever it drives a
    ``FunctionLinearOperator`` (the matrix-free shape MADDENING's
    IFT backward uses) — including on a well-conditioned ``0.5*I``
    operator.  This is a lineax 0.0.7 bug, not a property of the
    coupling Jacobian.  ``MatrixLinearOperator`` works.

    This test asserts the failure mode directly: BiCGStab via the
    same FunctionLinearOperator shape used in the backward returns
    a non-finite solution.  When a future lineax version fixes this
    (FunctionLinearOperator-driven BiCGStab returns finite values),
    the assertion will flip, and that is the signal to widen the
    ``linear_solver`` Literal on CouplingGroup to include
    ``"bicgstab"`` as a supported config value.
    """
    import lineax as lx  # noqa: PLC0415

    def _matvec(v):
        return 0.5 * v

    g = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    op = lx.FunctionLinearOperator(_matvec, jax.eval_shape(lambda: g))
    try:
        result = lx.linear_solve(
            op, g,
            solver=lx.BiCGStab(rtol=1e-6, atol=1e-8, max_steps=200),
        )
    except Exception:
        # The current behaviour: lineax raises an EquinoxRuntimeError
        # about non-finite output from the solver, before returning.
        return
    # If lineax 0.0.7 stops raising and returns a value, it should
    # still be NaN (the underlying bug).  The expected value is 2*g.
    expected = 2.0 * g
    if jnp.all(jnp.isfinite(result.value)) and jnp.allclose(
        result.value, expected, atol=1e-4, rtol=1e-4
    ):
        pytest.fail(
            "lineax BiCGStab now solves the FunctionLinearOperator "
            "case correctly — widen CouplingGroup.linear_solver "
            "Literal to include 'bicgstab' and add a parity test "
            "against GMRES."
        )


def test_dense_matches_gmres_gradient_small_chain():
    """``linear_solver='dense'`` agrees with ``'gmres'`` on a small chain.

    The dense path builds the full ``(I - dF/dx)`` Jacobian via jacrev
    and solves with ``jnp.linalg.solve``.  It is O(N^2) memory and is
    promoted from the env-var-gated fallback to a first-class config
    option here; this test keeps both paths in sync.
    """
    n = 20
    perturbed = f"s{n // 2}"
    gm_gmres = _make_chain_gm(n, "ift", linear_solver="gmres")
    gm_dense = _make_chain_gm(n, "ift", linear_solver="dense")
    g_gmres = _grad_through_compiled_step(gm_gmres, perturbed)
    g_dense = _grad_through_compiled_step(gm_dense, perturbed)
    assert jnp.allclose(g_gmres, g_dense, atol=1e-3, rtol=1e-3), (
        f"dense vs gmres gradient mismatch: "
        f"gmres={g_gmres}, dense={g_dense}"
    )


# ----------------------------------------------------------------------
# 4. GMRES-restart-too-small silent-corruption regression guard
# ----------------------------------------------------------------------
#
# Background: lineax.GMRES defaults ``restart`` to 20 (the dim of the
# Krylov subspace it builds).  When the adjoint system has N >> 20,
# the default-20 GMRES can converge to a low-rank approximation that
# satisfies the projected-subspace residual but lives in a 20-D
# subspace of the N-D problem.  ``result.value`` looks correct
# (no NaN, no error, ``result.stats`` reports success) but the
# returned ``u`` is structurally wrong, producing structurally
# wrong gradients.  This silent-corruption mode was hit during the
# initial lineax migration at N=250 and motivated the explicit
# ``restart=min(N, 50)`` in ``_ift_linear_solve``.
#
# Reproducing the silent-corruption gradient empirically is finicky:
# whether the 20-D Krylov subspace happens to contain (a projection
# of) the true adjoint depends on the spectrum of ``(I - dF/dx)^T``
# and on the right-hand side ``g``.  For the spring-chain fixture at
# N<=100, restart=20 happens to be enough; the corruption regime
# kicks in for harder problems (denser coupling Jacobians, smaller
# damping, larger N).
#
# So instead of chasing a fixture that exhibits the corruption (and
# being at the mercy of float32 noise), this test pins the *structural*
# invariant that prevents the regression: the production code must
# call ``lx.GMRES`` with ``restart`` of at least ``min(N, 50)``, not
# the lineax default of 20.  We spy on the GMRES constructor's
# kwargs at trace time and assert the override is in place.
#
# A programmer who "simplifies" the GMRES call by dropping the
# explicit ``restart=`` argument trips this immediately — even on
# small fixtures where the empirical bug wouldn't be detectable.


def test_gmres_call_uses_explicit_restart_at_least_minN50(monkeypatch):
    """Production IFT backward must call ``lx.GMRES`` with
    ``restart >= min(N, 50)``, not the lineax default of 20.

    This is the regression guard against silent gradient corruption
    described in the comment block above.  See also the long-form
    comment in ``_ift_linear_solve`` (search for "GMRES restart
    gotcha").
    """
    import lineax as lx  # noqa: PLC0415

    real_gmres = lx.GMRES
    seen_kwargs: list[dict] = []

    def _spy_gmres(*args, **kwargs):
        seen_kwargs.append(dict(kwargs))
        return real_gmres(*args, **kwargs)

    monkeypatch.setattr(lx, "GMRES", _spy_gmres)

    # N=60 chain ⇒ group state size 120 floats, well above the
    # lineax default restart of 20.  We expect the production code
    # to pass restart=50 (= min(120, 50)).
    n = 60
    gm = _make_chain_gm(n, "ift")
    _ = gm.step()
    compiled = gm._compiled_step
    initial_state = gm._state
    names = [k for k in initial_state.keys() if k != "_meta"]

    # Force the backward to be traced by computing a gradient.
    def loss_fn(pos):
        state = {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        state["s0"] = dict(state["s0"])
        state["s0"]["position"] = pos
        new_state = compiled(state, {})
        return _loss_chain(new_state, names)

    _ = jax.grad(loss_fn)(initial_state["s0"]["position"])

    assert seen_kwargs, (
        "lx.GMRES was never called during the IFT backward — the spy "
        "is not reaching the production solver."
    )
    # All calls (there may be more than one if the bwd is re-traced)
    # must use restart >= min(2*N, 50) = 50.
    expected_min_restart = min(2 * n, 50)
    for kw in seen_kwargs:
        restart = kw.get("restart")
        assert restart is not None, (
            "lx.GMRES called without an explicit restart= kwarg.  This "
            "means the lineax default-20 restart is in effect, which "
            "silently corrupts gradients for N>20 (see comment in "
            "_ift_linear_solve).  Restore the explicit restart=min(N,50)."
        )
        assert restart >= expected_min_restart, (
            f"lx.GMRES called with restart={restart}, but the production "
            f"floor is {expected_min_restart}.  See the GMRES restart "
            f"gotcha comment in _ift_linear_solve."
        )
        # Likewise, ``max_steps`` must be at least 4*restart so the
        # algorithm has headroom for several restart cycles.
        max_steps = kw.get("max_steps")
        assert max_steps is not None and max_steps >= 4 * restart, (
            f"lx.GMRES called with max_steps={max_steps}, but the "
            f"production floor is 4*restart={4*restart}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# ----------------------------------------------------------------------
# 5. Stiff small group — the adjoint solve must not crash on the
#    default path
# ----------------------------------------------------------------------
#
# Coupling audit 2026-09-19 (audit_040_final/coupling): ``jax.grad``
# through ``solver="ift", linear_solver="gmres"`` -- both defaults --
# raised ``_EquinoxRuntimeError: iterative breakdown`` on a *4-DOF*
# two-node cycle whose slow contraction mode is 0.999.  ``jax.jvp`` on
# the same graph was fine and ``linear_solver="dense"`` was fine, so it
# was the transpose (adjoint) solve specifically.
#
# The mechanism is a tolerance-versus-conditioning interaction, not a
# Krylov breakdown: ``cond(I - dF/dx) ~ 1/(1 - rho)`` puts the
# attainable float32 accuracy (``eps * cond``) *below* the tolerance
# lineax is asked for, GMRES exhausts its Krylov space without passing
# the test, and the redundant restart cycle reports breakdown.  The
# remedy lineax prints -- raise ``restart`` -- cannot help, because
# ``restart`` is already ``min(N, 50)``.  See the long-form comment in
# ``graph_manager._ift_linear_solve``.
#
# It is also non-monotone in stiffness (0.998 and 0.999 raised, 0.9995
# did not), because whether the float32 iterate lands inside an
# unreachable tolerance is a round-off lottery in the cotangent.  So
# the scan below covers the whole band rather than the one rate that
# happened to fail.


_STIFF_MODES = (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)


def _make_two_mode_cycle(rho_slow: float, linear_solver: str = "gmres"):
    """A two-node cycle whose flat group state is 4 floats.

    ``a`` applies a diagonal contraction ``diag(rho_slow, 0.2)`` to its
    boundary input and adds ``gain * (1e-5, 1.0)``; ``b`` is the
    identity that closes the cycle.  The fixed point is
    ``gain * c / (1 - rho)`` per mode, so
    ``d(sum x*)/d(gain) = sum(c / (1 - rho))`` in closed form.
    """
    from maddening.core.node import BoundaryInputSpec, SimulationNode

    rho = jnp.asarray([rho_slow, 0.2])
    c = jnp.asarray([1e-5, 1.0])

    class _Contract(SimulationNode):
        def __init__(self, name, dt, gain=1.0):
            super().__init__(name, dt, gain=gain)

        def initial_state(self):
            return {"x": jnp.zeros(2)}

        def boundary_input_spec(self):
            return {"u": BoundaryInputSpec(shape=(2,), description="u")}

        def update(self, state, bi, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            return {"x": rho * bi.get("u", jnp.zeros(2)) + p["gain"] * c}

    class _Relay(SimulationNode):
        def initial_state(self):
            return {"y": jnp.zeros(2)}

        def boundary_input_spec(self):
            return {"v": BoundaryInputSpec(shape=(2,), description="v")}

        def update(self, state, bi, dt, *, params=None):
            return {"y": bi.get("v", jnp.zeros(2))}

    gm = GraphManager()
    gm.add_node(_Contract("a", 0.01))
    gm.add_node(_Relay("b", 0.01))
    gm.add_edge("a", "b", "x", "v")
    gm.add_edge("b", "a", "y", "u")
    gm.add_coupling_group(
        ["a", "b"], max_iterations=60, tolerance=1e-4, diagnostics=True,
        solver="ift", linear_solver=linear_solver,
    )
    gm.compile()
    return gm


def _gain_gradient(rho_slow: float, linear_solver: str = "gmres") -> float:
    def loss(p):
        gm = _make_two_mode_cycle(rho_slow, linear_solver)
        return jnp.sum(gm.run_scan(1, params=p)["a"]["x"])

    base = _make_two_mode_cycle(rho_slow, linear_solver).params
    return float(jax.grad(loss)(base)["nodes"]["a"]["gain"])


@pytest.mark.parametrize("rho_slow", _STIFF_MODES)
def test_default_adjoint_solve_returns_the_analytic_gradient_when_stiff(
    rho_slow,
):
    """``jax.grad`` on the default solver path neither raises nor lies.

    Regression for the audit's 4-DOF breakdown.  The closed-form
    gradient is available here, so this asserts the *answer* and not
    merely the absence of an exception -- a fallback that silently
    returned the unconverged Krylov iterate would pass the weaker test.
    """
    exact = float(1e-5 / (1.0 - rho_slow) + 1.0 / (1.0 - 0.2))
    got = _gain_gradient(rho_slow)
    assert got == pytest.approx(exact, rel=2e-3), (
        f"rho_slow={rho_slow}: adjoint gradient {got} != analytic {exact}"
    )


@pytest.mark.parametrize("rho_slow", (0.998, 0.999))
def test_stiff_adjoint_agrees_across_linear_solvers(rho_slow):
    """The two rates that used to raise agree with the dense backend.

    ``"gmres"`` now re-solves densely when lineax reports failure, so
    the two backends must produce the same number on exactly the
    configurations where they used to produce a number and an
    exception.
    """
    assert _gain_gradient(rho_slow, "gmres") == pytest.approx(
        _gain_gradient(rho_slow, "dense"), rel=1e-5,
    )


def test_unaffordable_dense_fallback_names_the_remedies_that_work(
    monkeypatch,
):
    """Above the fallback size a failed adjoint raises MADDENING's error.

    The matrix-free path exists so that large groups never materialise
    ``N**2``, so the fallback is deliberately capped; above the cap the
    failure has to surface.  What it must *not* do is surface lineax's
    own message, whose only remedy ("increase ``restart``") is already
    at its maximum and does not address the mechanism.

    Driving a genuine >50-DOF breakdown is a round-off lottery, so the
    cap is lowered to 0 instead and the 4-DOF case that reliably fails
    is reused.  That exercises the same branch on the same failure.
    """
    from maddening.core import graph_manager as gm_mod

    monkeypatch.setattr(gm_mod, "_DENSE_ADJOINT_FALLBACK_MAX_DOF", 0)
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 — eqx runtime error
        _gain_gradient(0.999)
    message = str(excinfo.value)
    assert "linear_solver='dense'" in message, message
    assert "MADDENING_IFT_DENSE_SOLVE" in message, message
    assert "restart" in message and "NOT help" in message, message
