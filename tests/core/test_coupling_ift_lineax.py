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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
