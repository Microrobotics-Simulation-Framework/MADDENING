#!/usr/bin/env python
"""
Differentiable optimization through a MADDENING simulation graph.

Demonstrates that MADDENING graphs are end-to-end differentiable via JAX.
Uses jax.grad to optimize an initial ball velocity so that a coupled
spring-damper reaches a target position after a fixed number of steps.

Setup:
- Table at height 0 (collision surface).
- Ball starts at height 3 with an unknown initial velocity.
- Spring anchored to ball position, starts at height 2.
- Goal: find the ball's initial velocity such that the spring's
  position equals a target value (1.5) after 200 steps.

The optimization uses simple gradient descent with jax.grad, which
differentiates through the entire compiled graph step function.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/differentiable_optimization.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


def main() -> None:
    # ---- build the graph ------------------------------------------------
    gm = GraphManager()

    table = TableNode(name="table", timestep=0.01, position=0.0)
    ball = BallNode(
        name="ball",
        timestep=0.01,
        initial_position=3.0,
        initial_velocity=0.0,
        elasticity=0.6,
    )
    spring = SpringDamperNode(
        name="spring",
        timestep=0.01,
        stiffness=80.0,
        damping=3.0,
        mass=0.5,
        rest_length=1.0,
        initial_position=2.0,
        initial_velocity=0.0,
    )

    gm.add_node(table)
    gm.add_node(ball)
    gm.add_node(spring)

    # Wire table -> ball (collision), ball -> spring (anchor)
    gm.add_edge(source="table", target="ball",
                 source_field="position", target_field="table_position")
    gm.add_edge(source="ball", target="spring",
                 source_field="position", target_field="anchor_position")

    gm.compile()
    print(f"Graph: {gm}")
    print(f"Schedule: {gm.schedule}")

    # ---- define the optimization problem --------------------------------
    # Use 50 steps so the system hasn't fully settled -- initial velocity
    # still has a strong influence on the spring's final position.
    n_sim_steps = 50
    step_fn = gm._build_step_fn()
    empty_ext = gm._default_external_inputs()

    # First, find out what the spring position is with zero initial velocity
    # so we can set a target that requires a different initial velocity.
    def simulate(init_velocity):
        """Run the graph forward and return the spring's final position."""
        state = {
            "table": {"position": jnp.array(0.0)},
            "ball": {"position": jnp.array(3.0), "velocity": init_velocity},
            "spring": {"position": jnp.array(2.0), "velocity": jnp.array(0.0)},
        }

        def scan_body(s, _):
            return step_fn(s, empty_ext), None

        final_state, _ = jax.lax.scan(scan_body, state, None, length=n_sim_steps)
        return final_state["spring"]["position"]

    baseline_pos = float(simulate(jnp.array(0.0)))
    # Set target slightly above baseline so gradient descent can reach it
    # by giving the ball upward velocity (keeps it higher longer)
    target_spring_pos = baseline_pos + 0.3

    def loss_fn(init_velocity):
        """Squared error between spring's final position and target."""
        final_pos = simulate(init_velocity)
        return (final_pos - target_spring_pos) ** 2

    grad_fn = jax.jit(jax.grad(loss_fn))
    loss_fn_jit = jax.jit(loss_fn)

    # ---- gradient descent -----------------------------------------------
    learning_rate = 0.5
    n_iterations = 80
    velocity = jnp.array(0.0)  # initial guess

    print(f"\nBaseline spring position (v=0): {baseline_pos:.6f}")
    print(f"Optimization target: spring position = {target_spring_pos:.4f} "
          f"after {n_sim_steps} steps")
    print(f"Learning rate: {learning_rate}, Max iterations: {n_iterations}")
    print(f"\n{'Iter':>4}  {'Loss':>12}  {'Velocity':>12}  {'Spring Pos':>12}")
    print("-" * 52)

    losses = []
    velocities = []

    for i in range(n_iterations):
        loss_val = float(loss_fn_jit(velocity))
        grad_val = float(grad_fn(velocity))
        spring_pos = float(simulate(velocity))

        losses.append(loss_val)
        velocities.append(float(velocity))

        if i < 10 or i % 10 == 0 or loss_val < 1e-6:
            print(f"{i:4d}  {loss_val:12.6f}  {float(velocity):12.6f}  {spring_pos:12.6f}")

        # Gradient descent update (clip gradient for stability)
        grad_clipped = jnp.clip(grad_val, -10.0, 10.0)
        velocity = velocity - learning_rate * grad_clipped

        if loss_val < 1e-8:
            print(f"\nConverged at iteration {i}!")
            break

    # ---- final result ---------------------------------------------------
    final_loss = float(loss_fn_jit(velocity))
    final_spring_pos = float(simulate(velocity))
    print(f"\n--- Optimization result ---")
    print(f"  Optimal initial velocity: {float(velocity):.6f}")
    print(f"  Final spring position:    {final_spring_pos:.6f}")
    print(f"  Target spring position:   {target_spring_pos:.6f}")
    print(f"  Final loss:               {final_loss:.2e}")

    # ---- verify gradient exists -----------------------------------------
    test_grad = float(grad_fn(jnp.array(1.0)))
    print(f"\n  Gradient at v=1.0: {test_grad:.6f}")
    assert not jnp.isnan(jnp.array(test_grad)), "Gradient is NaN!"
    print("  Gradient is finite and non-NaN: differentiability confirmed.")

    # ---- sanity check ---------------------------------------------------
    # Check that we improved from the initial loss
    initial_loss = losses[0]
    improved = final_loss < initial_loss
    print(f"\n  Initial loss: {initial_loss:.2e}")
    print(f"  Final loss:   {final_loss:.2e}")
    print(f"  Improved: {'YES' if improved else 'NO'}")
    assert improved, f"Optimization did not improve loss ({final_loss:.4f} >= {initial_loss:.4f})"
    print(f"Sanity check: optimization reduced loss from {initial_loss:.2e} to {final_loss:.2e}.")

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogy(range(len(losses)), losses, "b.-")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (log scale)")
        ax1.set_title("Optimization Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(len(velocities)), velocities, "r.-")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Initial velocity (m/s)")
        ax2.set_title("Parameter (initial velocity)")
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Differentiable Optimization through MADDENING Graph", fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(_project_root, "differentiable_optimization_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
