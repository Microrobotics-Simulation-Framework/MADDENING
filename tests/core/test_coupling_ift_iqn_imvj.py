"""IFT coupling solver with **IQN-IMVJ** acceleration.

Extends the IFT path (previously gated on ``acceleration='none'`` /
``'aitken'``) to support ``acceleration='iqn-imvj'``: an interface
quasi-Newton acceleration where each forward iteration uses a
secant-LS update to approximate the residual Jacobian's inverse.

Per-step variant only — V/W matrices are zeroed at the start of each
timestep, then rebuilt within the while_loop and discarded.  No
cross-timestep warm-start (deferred follow-up).

The IFT backward remains acceleration-agnostic: at the fixed point
``x* = F(x*)`` the IFT adjoint differentiates ``F`` (the bare
contraction), not the QN-relaxed iterate.  So gradients computed by
ift+iqn-imvj should match ift+none / ift+aitken / fori+none up to
small numerical noise.

Tests:

1. Forward parity:  ``solver='ift'+acceleration='iqn-imvj'`` matches
   ``solver='fori'+acceleration='iqn-imvj'`` on a spring scene where
   both converge.
2. Backward parity through ``jax.jit``:  ``jax.grad`` through the
   jitted compiled step matches the other accelerations.
3. Convergence-rate evidence — the key result.  Construct a stiff
   non-symmetric synthetic contraction map and confirm IQN-IMVJ
   converges in strictly fewer iterations than Aitken (and ``none``).
   This is the existence-proof that IQN-IMVJ is doing something the
   simpler relaxation cannot.
4. At-scale FD sanity.  N=50-node spring chain (~50 floats of group
   state).  Forward + backward complete; FD on a handful of inputs
   matches autodiff gradient to a loose tolerance.
"""

from __future__ import annotations

import os

# Force CPU for these small graphs — much faster than warming up CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import (
    GraphManager,
    _ift_fixed_point_fwd_impl,
)
from maddening.nodes.spring import SpringDamperNode


# ----------------------------------------------------------------------
# Fixture: same stiff 2-node spring scene as the Aitken IFT tests.
# ----------------------------------------------------------------------


def _make_gm(solver: str, acceleration: str,
             dt: float = 0.001, max_iters: int = 30,
             tol: float = 1e-8) -> GraphManager:
    gm = GraphManager()
    a = SpringDamperNode(
        name="spring_a", timestep=dt, stiffness=500.0, damping=2.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    )
    b = SpringDamperNode(
        name="spring_b", timestep=dt, stiffness=5.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=2.0,
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    gm.add_coupling_group(
        ["spring_a", "spring_b"],
        max_iterations=max_iters,
        tolerance=tol,
        acceleration=acceleration,
        solver=solver,
    )
    return gm


# ----------------------------------------------------------------------
# 1. Forward parity:  IFT+IQN-IMVJ vs fori+IQN-IMVJ
# ----------------------------------------------------------------------


def test_forward_parity_ift_iqn_imvj_vs_fori_iqn_imvj():
    """Converged ``x*`` agrees between ift+iqn-imvj and fori+iqn-imvj."""
    gm_fori = _make_gm("fori", "iqn-imvj")
    gm_ift = _make_gm("ift", "iqn-imvj")

    s_fori = gm_fori.step()
    s_ift = gm_ift.step()

    for nn in ("spring_a", "spring_b"):
        for fld, v_fori in s_fori[nn].items():
            v_ift = s_ift[nn][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"forward parity (ift+iqn-imvj vs fori+iqn-imvj) failed on "
                f"{nn}.{fld}: fori={v_fori}, ift={v_ift}"
            )


def test_forward_parity_ift_iqn_imvj_vs_ift_none():
    """ift+iqn-imvj and ift+none reach the same fixed point.

    Acceleration is a forward-pass technique; ``x*`` is intrinsic to
    the contraction map.  Both should converge to the same answer.
    """
    gm_none = _make_gm("ift", "none")
    gm_iqn = _make_gm("ift", "iqn-imvj")

    s_none = gm_none.step()
    s_iqn = gm_iqn.step()

    for nn in ("spring_a", "spring_b"):
        for fld, v_none in s_none[nn].items():
            v_iqn = s_iqn[nn][fld]
            assert jnp.allclose(v_none, v_iqn, atol=1e-5, rtol=1e-5), (
                f"forward parity (ift+none vs ift+iqn-imvj) failed on "
                f"{nn}.{fld}: none={v_none}, iqn={v_iqn}"
            )


# ----------------------------------------------------------------------
# 2. Backward parity through jax.jit  — IFT backward is acceleration-agnostic
# ----------------------------------------------------------------------


def _loss_from_state(state, nodes=("spring_a", "spring_b")):
    return sum(jnp.sum(state[n]["position"] ** 2) for n in nodes)


def _grad_through_compiled_step(gm: GraphManager, pert_node="spring_a"):
    _ = gm.step()
    compiled = gm._compiled_step
    assert compiled is not None, "gm._compiled_step must be set after step()"

    initial_state = gm._state

    def loss_of_position(pos):
        state = {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        state[pert_node] = dict(state[pert_node])
        state[pert_node]["position"] = pos
        new_state = compiled(state, {})
        return _loss_from_state(new_state)

    return jax.grad(loss_of_position)(jnp.array(0.0))


def test_backward_parity_ift_iqn_imvj_through_jit():
    """jax.grad through the jitted step matches across all four configs.

    Pairwise parity:
        fori+none, ift+none, ift+aitken, ift+iqn-imvj
    should all yield (almost) the same gradient at the fixed point.
    This is the test that proves the IFT backward is truly
    acceleration-agnostic — wrapping ``F`` in IQN-IMVJ inside the
    forward while_loop does not perturb the cotangent that flows out.
    """
    g_fori_none = _grad_through_compiled_step(_make_gm("fori", "none"))
    g_ift_none = _grad_through_compiled_step(_make_gm("ift", "none"))
    g_ift_ait = _grad_through_compiled_step(_make_gm("ift", "aitken"))
    g_ift_iqn = _grad_through_compiled_step(_make_gm("ift", "iqn-imvj"))

    assert jnp.allclose(g_ift_iqn, g_ift_none, atol=1e-3, rtol=1e-3), (
        f"ift backward is not acceleration-agnostic: "
        f"none={g_ift_none}, iqn-imvj={g_ift_iqn}"
    )
    assert jnp.allclose(g_ift_iqn, g_ift_ait, atol=1e-3, rtol=1e-3), (
        f"ift+iqn-imvj backward did not match ift+aitken: "
        f"aitken={g_ift_ait}, iqn-imvj={g_ift_iqn}"
    )
    assert jnp.allclose(g_ift_iqn, g_fori_none, atol=1e-3, rtol=1e-3), (
        f"fori+none vs ift+iqn-imvj backward mismatch: "
        f"fori_none={g_fori_none}, ift_iqn={g_ift_iqn}"
    )


# ----------------------------------------------------------------------
# 3. Convergence-rate evidence on a stiff non-symmetric contraction.
#
# We use ``_ift_fixed_point_fwd_impl`` directly with a hand-built
# F(x) = A x + b where ``A`` is a non-symmetric matrix with
# eigenvalues  (0.98, 0.5)  — a contraction with spectral radius
# 0.98 (so Gauss-Seidel converges, but very slowly) and a strongly
# non-orthogonal eigenbasis (the IMVJ secant approximation should
# capture both modes after a few iterations).
#
# The existence-proof claim: on this contraction, IQN-IMVJ converges
# in strictly fewer iterations than Aitken (and ``none``).
# ----------------------------------------------------------------------


def _stiff_nonsym_F():
    """Return ``(F, x_star_true)`` for the stiff non-symmetric scene.

    A = P @ diag(0.98, 0.5) @ P^{-1} with non-orthogonal P.  This gives
    a contraction map whose iteration matrix has spectral radius 0.98
    and a stiffness ratio (1/(1-0.98)) / (1/(1-0.5)) = 25× between
    the slow and fast modes — fits the "stiff non-symmetric" pattern
    the task asks for.
    """
    # Use float32 to match the surrounding code's default precision.
    P = jnp.array([[1.0, 0.99], [0.5, 1.0]], dtype=jnp.float32)
    D = jnp.array([[0.98, 0.0], [0.0, 0.5]], dtype=jnp.float32)
    A = P @ D @ jnp.linalg.inv(P)
    b = jnp.array([0.1, 0.05], dtype=jnp.float32)
    x_star = jnp.linalg.solve(jnp.eye(2, dtype=jnp.float32) - A, b)

    def F(x, *consts):
        del consts
        return A @ x + b

    return F, x_star


def test_convergence_rate_iqn_imvj_beats_aitken_on_stiff_scene():
    """IQN-IMVJ converges in strictly fewer iters than Aitken on a stiff scene.

    This is the existence-proof that IQN-IMVJ is doing something
    Aitken cannot.  Tolerance and max_iter are matched across the
    three accelerations.
    """
    F, x_star_true = _stiff_nonsym_F()
    x0 = jnp.zeros(2, dtype=jnp.float32)
    consts = ()
    tol = 1e-5  # float32-comfortable
    max_iter = 500

    n_iters = {}
    x_star = {}
    for acc in ("none", "aitken", "iqn-imvj"):
        xs, nit = _ift_fixed_point_fwd_impl(
            F, x0, consts, tol, max_iter, acc,
        )
        n_iters[acc] = int(nit)
        x_star[acc] = xs

    # All three should converge to (approximately) the same fixed point.
    for acc in ("none", "aitken", "iqn-imvj"):
        err = jnp.linalg.norm(x_star[acc] - x_star_true)
        assert err < 1e-2, (
            f"{acc} did not converge: x_star={x_star[acc]}, "
            f"true={x_star_true}, err={err}, iters={n_iters[acc]}"
        )

    # The key convergence-rate claim: IQN-IMVJ wins.
    assert n_iters["iqn-imvj"] < n_iters["aitken"], (
        f"IQN-IMVJ did not beat Aitken on the stiff scene: "
        f"iqn-imvj={n_iters['iqn-imvj']}, aitken={n_iters['aitken']}"
    )
    assert n_iters["iqn-imvj"] < n_iters["none"], (
        f"IQN-IMVJ did not beat bare GS on the stiff scene: "
        f"iqn-imvj={n_iters['iqn-imvj']}, none={n_iters['none']}"
    )
    # Print for visibility under -s.
    print(
        f"\n[stiff convergence] none={n_iters['none']}, "
        f"aitken={n_iters['aitken']}, iqn-imvj={n_iters['iqn-imvj']}"
    )


# ----------------------------------------------------------------------
# 4. At-scale finite-difference sanity check.
#
# N=50 coupled spring nodes in a ring — group state ~ 50 floats of
# position (the interface field auto-detected by add_coupling_group).
# Forward + backward must complete without OOM, and FD on a handful
# of sampled coupling inputs must match the lineax-computed gradient
# to ~5 % relative.
# ----------------------------------------------------------------------


def _make_chain_gm(n: int, solver: str, acceleration: str,
                   dt: float = 0.001, max_iters: int = 30,
                   tol: float = 1e-7) -> GraphManager:
    """Build an N-node spring chain coupled head-to-tail in a ring."""
    gm = GraphManager()
    nodes = []
    for i in range(n):
        # Mild stiffness variation across the ring so the coupling
        # actually does something.
        k = 100.0 + 5.0 * (i % 7)
        nd = SpringDamperNode(
            name=f"s{i}", timestep=dt, stiffness=k, damping=0.5,
            mass=1.0, rest_length=1.0, initial_position=float(i) * 0.1,
        )
        nodes.append(nd)
        gm.add_node(nd)
    # Ring topology — each spring sees its next neighbour.
    for i in range(n):
        gm.add_edge(
            f"s{i}", f"s{(i + 1) % n}", "position", "anchor_position",
        )
    gm.add_coupling_group(
        [f"s{i}" for i in range(n)],
        max_iterations=max_iters,
        tolerance=tol,
        acceleration=acceleration,
        solver=solver,
    )
    return gm


def test_atscale_fd_matches_autodiff_ift_iqn_imvj():
    """N=50 chain: autodiff gradient agrees with FD on sampled inputs."""
    n = 50
    gm = _make_chain_gm(n, "ift", "iqn-imvj")
    _ = gm.step()
    compiled = gm._compiled_step
    assert compiled is not None
    initial_state = gm._state

    def loss_of_positions(positions):
        """Loss as a function of every node's initial position."""
        state = {
            k: ({kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in initial_state.items()
        }
        for i in range(n):
            state[f"s{i}"] = dict(state[f"s{i}"])
            state[f"s{i}"]["position"] = positions[i]
        new_state = compiled(state, {})
        return sum(
            jnp.sum(new_state[f"s{i}"]["position"] ** 2) for i in range(n)
        )

    p0 = jnp.array(
        [initial_state[f"s{i}"]["position"] for i in range(n)],
        dtype=jnp.float32,
    )
    # Autodiff gradient.
    g_ad = jax.grad(loss_of_positions)(p0)
    assert jnp.all(jnp.isfinite(g_ad)), "autodiff gradient has NaN/inf"

    # Finite differences on 5 sampled indices.  ``eps=1e-1`` is sized
    # against the float32 loss precision: ``L`` is ~400 so the loss
    # difference at smaller eps drops below the float32 machine epsilon
    # of ~1.2e-7 * 400 = 5e-5, which is too coarse for a tight FD check.
    # The looser eps gives a usable signal at the cost of O(eps^2)
    # truncation error in the 2nd-order finite difference, which a 5 %
    # tolerance comfortably absorbs.
    eps = 1e-1
    sample_idxs = (0, 7, 17, 33, 49)
    for idx in sample_idxs:
        p_plus = p0.at[idx].add(eps)
        p_minus = p0.at[idx].add(-eps)
        L_plus = loss_of_positions(p_plus)
        L_minus = loss_of_positions(p_minus)
        g_fd = (L_plus - L_minus) / (2 * eps)
        rel = jnp.abs(g_fd - g_ad[idx]) / (jnp.abs(g_fd) + 1e-6)
        assert rel < 5e-2, (
            f"FD vs autodiff mismatch at idx={idx}: "
            f"g_fd={g_fd}, g_ad={g_ad[idx]}, rel={rel}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
