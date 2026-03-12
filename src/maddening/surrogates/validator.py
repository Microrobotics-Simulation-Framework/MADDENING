"""
SurrogateValidator -- compare surrogate predictions against physics nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp

from maddening.surrogates.dataset import SurrogateDataset


@dataclass
class ValidationReport:
    """Per-field validation metrics."""
    node_name: str
    per_field_mse: dict[str, float]
    per_field_max_error: dict[str, float]
    per_field_relative_error: dict[str, float]
    per_timestep_errors: Optional[dict[str, Any]] = None

    def summary(self) -> str:
        lines = [f"Validation Report for '{self.node_name}'"]
        lines.append("-" * 50)
        lines.append(f"{'Field':<20} {'MSE':>12} {'Max Err':>12} {'Rel Err':>12}")
        lines.append("-" * 50)
        for field_name in sorted(self.per_field_mse.keys()):
            mse = self.per_field_mse[field_name]
            max_e = self.per_field_max_error[field_name]
            rel_e = self.per_field_relative_error[field_name]
            lines.append(f"{field_name:<20} {mse:>12.6e} {max_e:>12.6e} {rel_e:>12.6e}")
        return "\n".join(lines)


class SurrogateValidator:
    """Compare surrogate vs physics predictions."""

    @staticmethod
    def compare_nodes(
        physics_node,
        surrogate_node,
        test_dataset: SurrogateDataset,
    ) -> ValidationReport:
        """One-step prediction accuracy on test data.

        Parameters
        ----------
        physics_node : SimulationNode
            The original physics node.
        surrogate_node : SurrogateNode
            The surrogate replacement.
        test_dataset : SurrogateDataset
            Test data (states, boundary_inputs, next_states).

        Returns
        -------
        ValidationReport
        """
        ds = test_dataset
        n_samples = next(iter(ds.states.values())).shape[0]

        # Run surrogate on each sample
        def predict_one(idx):
            state = {k: v[idx] for k, v in ds.states.items()}
            boundary = {k: v[idx] for k, v in ds.boundary_inputs.items()}
            return surrogate_node.update(state, boundary, ds.dt)

        surrogate_preds = jax.vmap(
            lambda i: predict_one(i)
        )(jnp.arange(n_samples))

        # Compute per-field metrics
        per_field_mse = {}
        per_field_max_error = {}
        per_field_relative_error = {}

        for field_name in ds.state_spec:
            target = ds.next_states[field_name]
            pred = surrogate_preds[field_name]
            diff = pred - target
            per_field_mse[field_name] = float(jnp.mean(diff ** 2))
            per_field_max_error[field_name] = float(jnp.max(jnp.abs(diff)))
            target_norm = jnp.sqrt(jnp.mean(target ** 2))
            rel = jnp.where(
                target_norm > 1e-10,
                jnp.sqrt(jnp.mean(diff ** 2)) / target_norm,
                jnp.sqrt(jnp.mean(diff ** 2)),
            )
            per_field_relative_error[field_name] = float(rel)

        return ValidationReport(
            node_name=ds.node_name,
            per_field_mse=per_field_mse,
            per_field_max_error=per_field_max_error,
            per_field_relative_error=per_field_relative_error,
        )

    @staticmethod
    def compare_graphs(
        gm_physics,
        gm_surrogate,
        n_steps: int,
        node_name: str,
    ) -> ValidationReport:
        """Multi-step rollout comparison.

        Runs both graphs for ``n_steps`` and compares the target node's
        trajectory, revealing error accumulation over time.

        Parameters
        ----------
        gm_physics : GraphManager
            Graph with the physics node.
        gm_surrogate : GraphManager
            Graph with the surrogate replacement.
        n_steps : int
            Number of steps to simulate.
        node_name : str
            Node to compare.

        Returns
        -------
        ValidationReport
            Includes per-timestep error arrays.
        """
        _, hist_phys = gm_physics.run_scan_with_history(n_steps)
        _, hist_surr = gm_surrogate.run_scan_with_history(n_steps)

        phys_node = hist_phys[node_name]
        surr_node = hist_surr[node_name]

        per_field_mse = {}
        per_field_max_error = {}
        per_field_relative_error = {}
        per_timestep_errors = {}

        for field_name in phys_node:
            target = phys_node[field_name]
            pred = surr_node[field_name]
            diff = pred - target

            per_field_mse[field_name] = float(jnp.mean(diff ** 2))
            per_field_max_error[field_name] = float(jnp.max(jnp.abs(diff)))

            target_norm = jnp.sqrt(jnp.mean(target ** 2))
            rel = jnp.where(
                target_norm > 1e-10,
                jnp.sqrt(jnp.mean(diff ** 2)) / target_norm,
                jnp.sqrt(jnp.mean(diff ** 2)),
            )
            per_field_relative_error[field_name] = float(rel)

            # Per-timestep absolute error (mean over spatial dims if any)
            if diff.ndim > 1:
                per_timestep_errors[field_name] = jnp.mean(jnp.abs(diff), axis=tuple(range(1, diff.ndim)))
            else:
                per_timestep_errors[field_name] = jnp.abs(diff)

        return ValidationReport(
            node_name=node_name,
            per_field_mse=per_field_mse,
            per_field_max_error=per_field_max_error,
            per_field_relative_error=per_field_relative_error,
            per_timestep_errors=per_timestep_errors,
        )
