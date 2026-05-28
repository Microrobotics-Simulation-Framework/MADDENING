"""IFT coupling solver with **Aitken** delta-squared acceleration.

The IFT path used to gate strictly on ``acceleration='none'`` and fall
through to the ``fori_loop`` solver for anything else.  This file
exercises ``solver='ift'`` + ``acceleration='aitken'``, which wraps each
``F(x)`` evaluation in Aitken relaxation inside the IFT forward
while_loop.

The IFT backward is **acceleration-agnostic**: at the fixed point
``x* = F(x*)`` the IFT adjoint differentiates ``F`` (the bare
contraction), not the Aitken-relaxed iterate.  Acceleration only
matters for the forward pass — it gets you to ``x*`` faster.  So the
gradient computed by IFT-Aitken should match IFT-none should match
fori-Aitken should match fori-none, modulo iteration-count differences
that show up as small numerical noise.

Tests:

1. Forward parity:  ``solver='ift'+acceleration='aitken'`` matches
   ``solver='fori'+acceleration='aitken'`` on a stiff coupling group
   that needs Aitken to converge in a reasonable iteration budget.
2. Backward parity through ``jax.jit``:  ``jax.grad`` through the
   jitted compiled step matches between solvers.  This is the
   sensitivity-analysis use case that motivated the feature.
"""

from __future__ import annotations

import os

# Force CPU for these small graphs — much faster than warming up CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


# ----------------------------------------------------------------------
# Fixture: stiff 2-node coupling where bare Gauss-Seidel oscillates and
# Aitken's relaxation actually matters.  Stiffness ratio ~100 between
# the two springs gives a contraction whose un-accelerated iterate
# converges slowly enough that the iter-budget difference matters but
# Aitken converges in a handful of iterations.
# ----------------------------------------------------------------------


def _make_gm(solver: str, acceleration: str,
             dt: float = 0.001, max_iters: int = 30,
             tol: float = 1e-8) -> GraphManager:
    gm = GraphManager()
    # Stiff vs soft coupling pair.  k_a=500, k_b=5 gives a contraction
    # that bare GS resolves but slowly; Aitken collapses it.
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
# 1. Forward parity:  IFT+Aitken vs fori+Aitken
# ----------------------------------------------------------------------


def test_forward_parity_ift_aitken_vs_fori_aitken():
    """Converged ``x*`` agrees between ift+aitken and fori+aitken."""
    gm_fori = _make_gm("fori", "aitken")
    gm_ift = _make_gm("ift", "aitken")

    s_fori = gm_fori.step()
    s_ift = gm_ift.step()

    for nn in ("spring_a", "spring_b"):
        for fld, v_fori in s_fori[nn].items():
            v_ift = s_ift[nn][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"forward parity (ift+aitken vs fori+aitken) failed on "
                f"{nn}.{fld}: fori={v_fori}, ift={v_ift}"
            )


def test_forward_parity_ift_aitken_vs_ift_none():
    """ift+aitken and ift+none reach the same fixed point.

    Acceleration is a forward-pass technique; ``x*`` is intrinsic to
    the contraction map.  Both should converge to the same answer.
    """
    gm_none = _make_gm("ift", "none")
    gm_ait = _make_gm("ift", "aitken")

    s_none = gm_none.step()
    s_ait = gm_ait.step()

    for nn in ("spring_a", "spring_b"):
        for fld, v_none in s_none[nn].items():
            v_ait = s_ait[nn][fld]
            assert jnp.allclose(v_none, v_ait, atol=1e-5, rtol=1e-5), (
                f"forward parity (ift+none vs ift+aitken) failed on "
                f"{nn}.{fld}: none={v_none}, aitken={v_ait}"
            )


# ----------------------------------------------------------------------
# 2. Backward parity through jax.jit  — the key new test
# ----------------------------------------------------------------------


def _loss_from_state(state, nodes=("spring_a", "spring_b")):
    """Scalar loss for grad testing: sum of squared positions."""
    return sum(jnp.sum(state[n]["position"] ** 2) for n in nodes)


def _grad_through_compiled_step(gm: GraphManager, pert_node="spring_a"):
    """Return ``d(loss)/d(initial_position_<pert_node>)`` via jitted step."""
    # Initialise compiled step.
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


def test_backward_parity_ift_aitken_through_jit():
    """jax.grad through the *jitted* step matches across all four configs.

    The combinatorial parity:
        fori + none, fori + aitken, ift + none, ift + aitken
    should all yield (almost) the same gradient at the fixed point.
    This is the test that proves the IFT backward is truly
    acceleration-agnostic — wrapping ``F`` in Aitken inside the forward
    while_loop does not perturb the cotangent that flows out.
    """
    g_fori_none = _grad_through_compiled_step(_make_gm("fori", "none"))
    g_fori_ait = _grad_through_compiled_step(_make_gm("fori", "aitken"))
    g_ift_none = _grad_through_compiled_step(_make_gm("ift", "none"))
    g_ift_ait = _grad_through_compiled_step(_make_gm("ift", "aitken"))

    # Pairwise parity, single-precision tolerances.  fori unrolls a
    # fixed iteration count regardless of convergence, so tiny noise
    # between fori vs ift is expected; aitken vs none differ only by
    # whether the convergence flag fired sooner (after which both
    # paths freeze the state via the `_merge` mask).
    assert jnp.allclose(g_fori_ait, g_ift_ait, atol=1e-3, rtol=1e-3), (
        f"ift+aitken backward did not match fori+aitken: "
        f"fori={g_fori_ait}, ift={g_ift_ait}"
    )
    assert jnp.allclose(g_ift_none, g_ift_ait, atol=1e-3, rtol=1e-3), (
        f"ift backward is not acceleration-agnostic: "
        f"none={g_ift_none}, aitken={g_ift_ait}"
    )
    assert jnp.allclose(g_fori_none, g_ift_ait, atol=1e-3, rtol=1e-3), (
        f"fori+none vs ift+aitken backward mismatch: "
        f"fori_none={g_fori_none}, ift_aitken={g_ift_ait}"
    )


# ----------------------------------------------------------------------
# 3. Embedded coupling group (outside-state read by group member) with
#    Aitken — confirms the closure_convert + IFT bwd still works when
#    aitken_relaxation lives inside the while_loop body.
# ----------------------------------------------------------------------


def _make_embedded_gm(solver: str, acceleration: str,
                      c_initial_position: float = 0.5,
                      dt: float = 0.001, max_iters: int = 30,
                      tol: float = 1e-8) -> GraphManager:
    gm = GraphManager()
    a = SpringDamperNode(
        name="A", timestep=dt, stiffness=500.0, damping=2.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    )
    b = SpringDamperNode(
        name="B", timestep=dt, stiffness=5.0, damping=0.5,
        mass=1.0, rest_length=1.0, initial_position=2.0,
    )
    c = SpringDamperNode(
        name="C", timestep=dt, stiffness=20.0, damping=0.5,
        mass=1.0, rest_length=0.0, initial_position=c_initial_position,
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(c)
    gm.add_edge("A", "B", "position", "anchor_position", additive=True)
    gm.add_edge("B", "A", "position", "anchor_position", additive=True)
    gm.add_edge("C", "A", "position", "anchor_position", additive=True)
    gm.add_coupling_group(
        ["A", "B"], max_iterations=max_iters, tolerance=tol,
        acceleration=acceleration, solver=solver,
    )
    return gm


def test_embedded_forward_parity_ift_aitken():
    """ift+aitken handles the embedded (outside-read) case."""
    gm_fori = _make_embedded_gm("fori", "aitken")
    gm_ift = _make_embedded_gm("ift", "aitken")

    s_fori = gm_fori.step()
    s_ift = gm_ift.step()

    for nn in ("A", "B", "C"):
        for fld, v_fori in s_fori[nn].items():
            v_ift = s_ift[nn][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"embedded forward parity (ift+aitken) failed on "
                f"{nn}.{fld}: fori={v_fori}, ift={v_ift}"
            )


def test_embedded_backward_parity_ift_aitken_through_jit():
    """Gradient w.r.t. outside-node initial state, through embedded IFT-Aitken."""
    gm_fori = _make_embedded_gm("fori", "aitken")
    gm_ift = _make_embedded_gm("ift", "aitken")

    # Initialise jitted step.
    _ = gm_fori.step()
    _ = gm_ift.step()

    def grad_of(gm):
        compiled = gm._compiled_step
        initial_state = gm._state

        def loss_of_pos_c(pos_c):
            state = {
                k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in initial_state.items()
            }
            state["C"] = dict(state["C"])
            state["C"]["position"] = pos_c
            new_state = compiled(state, {})
            return (
                jnp.sum(new_state["A"]["position"] ** 2)
                + jnp.sum(new_state["B"]["position"] ** 2)
            )

        return jax.grad(loss_of_pos_c)(jnp.array(0.5))

    g_fori = grad_of(gm_fori)
    g_ift = grad_of(gm_ift)
    assert jnp.allclose(g_fori, g_ift, atol=1e-3, rtol=1e-3), (
        f"embedded ift+aitken backward did not match fori+aitken: "
        f"fori={g_fori}, ift={g_ift}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
