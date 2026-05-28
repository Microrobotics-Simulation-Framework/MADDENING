"""IFT coupling solver with **embedded** coupling group.

The existing IFT prototype handles closed coupling groups (every edge
into a group member originates from another group member).  Real
graphs frequently have group members that also read from non-group
nodes — the "embedded" case.

These tests construct a minimal 3-node graph::

    C (outside)  ──►  A ◄────► B
                          coupling group [A, B]

C is a free-running spring/oscillator.  A is in the coupling group
and reads ``anchor_position`` additively from both B (the in-group
edge that closes the cycle) and from C (the embedded edge).  B reads
from A.

This pattern reproduces the bug where IFT's flatten/unflatten only
carries group-member state through the operator, so ``one_pass`` then
fails to resolve C's state when building A's boundary inputs.

We assert both:

1. Forward parity between ``solver='fori'`` and ``solver='ift'`` over
   the converged group state.
2. Backward parity (``jax.grad`` through the jitted compiled step) of
   a loss on A's position w.r.t. C's initial position.  This proves
   gradients flow back through the non-group "outside" state via the
   IFT adjoint.
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
# Fixture: A ↔ B coupling group, plus an outside C → A embedded edge.
# ----------------------------------------------------------------------


def _make_gm(solver: str, c_initial_position: float = 0.5,
             dt: float = 0.001, max_iters: int = 30,
             tol: float = 1e-8) -> GraphManager:
    gm = GraphManager()

    # Coupling group members.
    a = SpringDamperNode(
        name="A", timestep=dt, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    )
    b = SpringDamperNode(
        name="B", timestep=dt, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=2.0,
    )
    # Outside-the-group driver.  C is uncoupled — it just oscillates
    # from its initial offset around the origin.
    c = SpringDamperNode(
        name="C", timestep=dt, stiffness=20.0, damping=0.5,
        mass=1.0, rest_length=0.0, initial_position=c_initial_position,
    )
    gm.add_node(a)
    gm.add_node(b)
    gm.add_node(c)

    # In-group cycle (closes the coupling): A ↔ B.
    gm.add_edge("A", "B", "position", "anchor_position", additive=True)
    gm.add_edge("B", "A", "position", "anchor_position", additive=True)
    # Embedded edge: C → A.  C is NOT in the coupling group, but its
    # state is read while solving the coupling fixed point.  This is
    # the case the original IFT prototype could not handle.
    gm.add_edge("C", "A", "position", "anchor_position", additive=True)

    gm.add_coupling_group(
        ["A", "B"],
        max_iterations=max_iters,
        tolerance=tol,
        acceleration="none",
        solver=solver,
    )
    return gm


# ----------------------------------------------------------------------
# 1. Forward parity (embedded group)
# ----------------------------------------------------------------------


def test_forward_parity_embedded():
    """``solver='ift'`` matches ``solver='fori'`` when group reads outside."""
    gm_fori = _make_gm("fori")
    gm_ift = _make_gm("ift")

    s_fori = gm_fori.step()
    s_ift = gm_ift.step()

    for nn in ("A", "B", "C"):
        for fld, v_fori in s_fori[nn].items():
            v_ift = s_ift[nn][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"forward parity failed on {nn}.{fld}: "
                f"fori={v_fori}, ift={v_ift}"
            )


# ----------------------------------------------------------------------
# 2. Backward parity through jax.jit  (embedded group)
# ----------------------------------------------------------------------


def _loss_from_state(state):
    """Scalar loss: sum of squared positions of group members A and B."""
    return jnp.sum(state["A"]["position"] ** 2) + jnp.sum(
        state["B"]["position"] ** 2
    )


def _grad_through_compiled_step(gm: GraphManager):
    """Return ``d(loss)/d(initial_position_C)`` via jax.grad of jitted step.

    The gradient must flow through the IFT adjoint of the coupling
    fixed point *and* through C's state being read by A inside that
    fixed point.  This is the embedded-coupling backward path.
    """
    # Initialize the compiled step.
    _ = gm.step()
    compiled = gm._compiled_step
    assert compiled is not None, "gm._compiled_step must be set after step()"

    initial_state = gm._state

    def loss_of_pos_c(pos_c):
        state = {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        state["C"] = dict(state["C"])
        state["C"]["position"] = pos_c
        new_state = compiled(state, {})
        return _loss_from_state(new_state)

    return jax.grad(loss_of_pos_c)(jnp.array(0.5))


def test_backward_parity_embedded_through_jit():
    """jax.grad through the jitted step matches between solvers.

    Reads C's initial position, runs one full step (which includes the
    embedded coupling-group solve), takes a loss over A and B, and
    compares ``d loss / d C.position_0`` between the fori and IFT
    solvers.  Tolerance is single-precision; IFT converges to true
    fixed point while fori unrolls a fixed iteration count, so the
    gradients differ by a small amount.
    """
    gm_fori = _make_gm("fori")
    gm_ift = _make_gm("ift")

    g_fori = _grad_through_compiled_step(gm_fori)
    g_ift = _grad_through_compiled_step(gm_ift)

    assert jnp.allclose(g_fori, g_ift, atol=1e-3, rtol=1e-3), (
        f"embedded backward parity through jit failed: "
        f"fori={g_fori}, ift={g_ift}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
