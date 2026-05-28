"""IFT (implicit-function-theorem) coupling solver — parity and perf.

Three things:

1. Forward parity: ``solver='ift'`` matches ``solver='fori'`` to float32
   tolerance for the converged state.
2. Backward parity through ``jax.jit``: ``jax.grad`` of a loss applied
   to ``gm._compiled_step`` matches between the two solvers.  This is
   the regression test for the JIT-compat problem (DynamicJaxprTracer
   leaking through ``custom_vjp``) that previously blocked the IFT path.
3. Perf measurement: time both solvers on the same step and document
   the speed-up (the IFT path terminates early on convergence).
"""

from __future__ import annotations

import os
import time

# Force CPU for these small graphs — much faster than warming up CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.edge import EdgeSpec  # noqa: F401 (imported for parity with other tests)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


# ----------------------------------------------------------------------
# Fixture: two bidirectionally-coupled spring/dampers — a textbook
# 2-node cyclic coupling group, identical to the case used in the
# existing test_coupling.py.
# ----------------------------------------------------------------------


def _make_gm(solver: str, dt: float = 0.001, max_iters: int = 30,
             tol: float = 1e-8) -> GraphManager:
    gm = GraphManager()
    a = SpringDamperNode(
        name="spring_a", timestep=dt, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    )
    b = SpringDamperNode(
        name="spring_b", timestep=dt, stiffness=50.0, damping=1.0,
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
        acceleration="none",
        solver=solver,
    )
    return gm


# ----------------------------------------------------------------------
# 1. Forward parity
# ----------------------------------------------------------------------


def test_forward_parity_ift_vs_fori():
    """One step under ``solver='ift'`` matches ``solver='fori'``."""
    gm_fori = _make_gm("fori")
    gm_ift = _make_gm("ift")

    # Drive both forward one step from identical initial state.
    s_fori = gm_fori.step()
    s_ift = gm_ift.step()

    for nn in ("spring_a", "spring_b"):
        for fld, v_fori in s_fori[nn].items():
            v_ift = s_ift[nn][fld]
            assert jnp.allclose(v_fori, v_ift, atol=1e-5, rtol=1e-5), (
                f"forward parity failed on {nn}.{fld}: "
                f"fori={v_fori}, ift={v_ift}"
            )


# ----------------------------------------------------------------------
# 2. Backward parity through jax.jit  — the regression test
# ----------------------------------------------------------------------


def _loss_from_state(state, nodes=("spring_a", "spring_b")):
    """Scalar loss for grad testing: sum of squared positions."""
    return sum(jnp.sum(state[n]["position"] ** 2) for n in nodes)


def _grad_through_compiled_step(gm: GraphManager):
    """Return ``d(loss)/d(initial_position_a)`` via jax.grad of jitted step."""
    # Initialize the compiled step.
    _ = gm.step()
    compiled = gm._compiled_step
    assert compiled is not None, "gm._compiled_step must be set after step()"

    initial_state = gm._state

    def loss_of_position_a(pos_a):
        # Perturb initial spring_a position, run one step, then loss.
        state = {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in initial_state.items()
        }
        state["spring_a"] = dict(state["spring_a"])
        state["spring_a"]["position"] = pos_a
        new_state = compiled(state, {})
        return _loss_from_state(new_state)

    return jax.grad(loss_of_position_a)(jnp.array(0.0))


def test_backward_parity_through_jit():
    """jax.grad through the *jitted* step matches between solvers.

    Previously the IFT path could not be differentiated through
    ``gm._compiled_step`` because the custom_vjp rule captured tracers
    via Python closure, which JAX's pjit infrastructure rejects
    (the ``DynamicJaxprTracer`` leak).  The current implementation
    routes through a closure-converted top-level ``F_pure``, so this
    test should pass.
    """
    gm_fori = _make_gm("fori")
    gm_ift = _make_gm("ift")

    g_fori = _grad_through_compiled_step(gm_fori)
    g_ift = _grad_through_compiled_step(gm_ift)

    # Gradients should agree to single-precision tolerance.  IFT and
    # unrolled fori will differ slightly because fori unrolls a fixed
    # number of iterations regardless of convergence.
    assert jnp.allclose(g_fori, g_ift, atol=1e-3, rtol=1e-3), (
        f"backward parity through jit failed: fori={g_fori}, ift={g_ift}"
    )


# ----------------------------------------------------------------------
# 3. Perf measurement (informational — does not fail the suite)
# ----------------------------------------------------------------------


def test_perf_ift_vs_fori_measurement(capsys):
    """Time the jitted step under both solvers.  Reports speed-up.

    Not a hard assertion — JIT cache & noise make a strict speed-up
    requirement flaky.  We just record the numbers for the report.
    """
    n_warmup = 3
    n_meas = 25

    gm_fori = _make_gm("fori", max_iters=30)
    gm_ift = _make_gm("ift", max_iters=30)

    # Warm up the jit cache.
    for _ in range(n_warmup):
        gm_fori.step()
        gm_ift.step()

    def time_steps(gm: GraphManager, n: int) -> float:
        # Block on the result each iteration so timings are honest.
        t0 = time.perf_counter()
        for _ in range(n):
            s = gm.step()
            jax.block_until_ready(s["spring_a"]["position"])
        return (time.perf_counter() - t0) / n

    dt_fori = time_steps(gm_fori, n_meas)
    dt_ift = time_steps(gm_ift, n_meas)
    speedup = dt_fori / dt_ift if dt_ift > 0 else float("inf")

    with capsys.disabled():
        print(
            f"\n[IFT perf] fori = {dt_fori*1e6:.1f} us/step, "
            f"ift = {dt_ift*1e6:.1f} us/step, "
            f"speedup = {speedup:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
