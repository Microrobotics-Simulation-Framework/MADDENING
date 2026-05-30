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
2. **Large chain (N=250, ~500 floats):** the backward completes
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
    """N=250 chain (~500 floats of group state): IFT backward via
    lineax completes and the gradient agrees with finite differences
    on a few sampled input entries.

    The whole point of the lineax swap is to make this feasible — the
    dense ``jacrev + solve`` path would build a 500x500 Jacobian inside
    the coupling group and OOM under more realistic per-node state
    sizes.  Even at N=250 the matrix-free path stays light.

    Initial positions are kept small (0.05 + 0.01*i) so the central
    finite difference ``(L(p+eps) - L(p-eps)) / (2 eps)`` of the
    sum-of-squared-positions loss does not lose float32 precision
    against the perturbation magnitude.
    """
    n = 250
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
        # eps=1e-3 against positions ~0.05-2.55 gives ~1e-3 absolute
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
    ``_ift_solve_bwd`` is wired correctly, but the underlying
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
# ``restart=min(N, 50)`` in ``_ift_solve_bwd``.
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
    comment in ``_ift_solve_bwd`` (search for "GMRES restart
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
            "_ift_solve_bwd).  Restore the explicit restart=min(N,50)."
        )
        assert restart >= expected_min_restart, (
            f"lx.GMRES called with restart={restart}, but the production "
            f"floor is {expected_min_restart}.  See the GMRES restart "
            f"gotcha comment in _ift_solve_bwd."
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
