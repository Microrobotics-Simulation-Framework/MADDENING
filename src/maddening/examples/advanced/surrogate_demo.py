#!/usr/bin/env python
"""
Surrogate Demo -- Train an MLP surrogate for a free-falling BallNode.

Demonstrates the full pipeline:
    1. Run physics simulation to generate training data
    2. Train an MLP surrogate
    3. Replace the physics node with the surrogate
    4. Compare trajectories

Usage:
    python maddening/examples/surrogate_demo.py
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# Use CPU by default to avoid GPU OOM on small GPUs
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.trainer import SurrogateTrainer
from maddening.surrogates.replace import replace_node
from maddening.surrogates.validator import SurrogateValidator
from maddening.surrogates.architectures.mlp import MLPDirect


def main():
    print("=== MADDENING Surrogate Demo ===\n")

    # ------------------------------------------------------------------
    # 1. Physics simulation
    # ------------------------------------------------------------------
    print("1. Building physics graph...")
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm.compile()

    # ------------------------------------------------------------------
    # 2. Generate training data
    # ------------------------------------------------------------------
    print("2. Generating training data (500 steps)...")
    ds = DatasetGenerator.from_graph(gm, "ball", n_steps=500)
    print(f"   Dataset: {ds.states['position'].shape[0]} samples")
    print(f"   State spec: {ds.state_spec}")
    print(f"   Boundary spec: {ds.boundary_spec}")

    # ------------------------------------------------------------------
    # 3. Train surrogate
    # ------------------------------------------------------------------
    print("\n3. Training MLP surrogate...")
    arch = MLPDirect(hidden_sizes=(64, 64))
    trainer = SurrogateTrainer(arch, ds)

    def progress(epoch, metrics):
        if epoch % 20 == 0 or epoch == 99:
            print(f"   Epoch {epoch:3d}: train_loss={metrics['train_loss']:.6e}, "
                  f"val_loss={metrics['val_loss']:.6e}")

    result = trainer.train(
        n_epochs=100, batch_size=64,
        rng_key=jax.random.PRNGKey(42),
        callback=progress,
    )
    print(f"   Final train loss: {result.train_losses[-1]:.6e}")
    print(f"   Final val loss:   {result.val_losses[-1]:.6e}")

    # ------------------------------------------------------------------
    # 4. Replace physics node
    # ------------------------------------------------------------------
    print("\n4. Replacing BallNode with surrogate...")
    surrogate = result.to_node(
        name="ball", timestep=0.01,
        initial_values={"position": 10.0, "velocity": 0.0},
    )

    gm_surr = GraphManager()
    gm_surr.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm_surr.compile()
    replace_node(gm_surr, "ball", surrogate)
    gm_surr.compile()

    # ------------------------------------------------------------------
    # 5. Compare trajectories
    # ------------------------------------------------------------------
    print("\n5. Comparing trajectories (100 steps)...")
    gm_phys = GraphManager()
    gm_phys.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm_phys.compile()

    report = SurrogateValidator.compare_graphs(gm_phys, gm_surr, 100, "ball")
    print(report.summary())

    print("\nDone!")


if __name__ == "__main__":
    main()
