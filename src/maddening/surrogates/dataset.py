"""
SurrogateDataset and DatasetGenerator -- extract training data from physics
simulations for surrogate model training.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@dataclass
class SurrogateDataset:
    """Training dataset for a surrogate model.

    All arrays have a leading sample dimension of the same size.
    """
    states: dict[str, Any]            # {field: (n_samples, *shape)}
    boundary_inputs: dict[str, Any]   # {field: (n_samples, *shape)}
    next_states: dict[str, Any]       # {field: (n_samples, *shape)}
    dt: float
    node_name: str
    state_spec: dict[str, tuple]      # {field: shape}
    boundary_spec: dict[str, tuple]   # {field: shape}


@stability(StabilityLevel.EXPERIMENTAL)
class DatasetGenerator:
    """Generate training datasets from physics graph simulations."""

    @staticmethod
    def from_graph(gm, target_node: str, n_steps: int) -> SurrogateDataset:
        """Extract a dataset from a single trajectory.

        Runs ``gm.run_scan_with_history(n_steps)`` and extracts
        (state_t, boundary_inputs_t, state_{t+1}) triples for the
        target node.

        Parameters
        ----------
        gm : GraphManager
            Compiled graph (must contain ``target_node``).
        target_node : str
            Name of the node to generate data for.
        n_steps : int
            Number of simulation steps to run.

        Returns
        -------
        SurrogateDataset
            Dataset with ``n_steps - 1`` samples.
        """
        node_obj = gm._nodes[target_node].node
        node_dt = node_obj.delta_t

        # Run simulation and collect history
        _final, history = gm.run_scan_with_history(n_steps)

        # Extract state arrays for target node: shape (n_steps, *field_shape)
        node_history = history[target_node]

        # Build state_spec and boundary_spec from the node
        state_spec = {k: v.shape for k, v in node_obj.initial_state().items()}

        # Reconstruct boundary_inputs from edges and external inputs
        boundary_spec, boundary_arrays = DatasetGenerator._reconstruct_boundary(
            gm, target_node, history, n_steps,
        )

        # States at time t (drop last), next_states at time t+1 (drop first)
        n_samples = n_steps - 1
        states = {k: v[:-1] for k, v in node_history.items()}
        next_states = {k: v[1:] for k, v in node_history.items()}
        boundary = {k: v[:n_samples] for k, v in boundary_arrays.items()}

        return SurrogateDataset(
            states=states,
            boundary_inputs=boundary,
            next_states=next_states,
            dt=node_dt,
            node_name=target_node,
            state_spec=state_spec,
            boundary_spec=boundary_spec,
        )

    @staticmethod
    def from_sweep(
        gm,
        target_node: str,
        n_steps: int,
        initial_states_batch: dict[str, dict],
    ) -> SurrogateDataset:
        """Extract a dataset from a batched parameter sweep.

        Uses ``gm.run_sweep(n_steps, initial_states_batch, return_history=True)``
        and reshapes from ``(batch, n_steps, ...)`` to a flat dataset.

        Parameters
        ----------
        gm : GraphManager
            Compiled graph.
        target_node : str
            Node to generate data for.
        n_steps : int
            Steps per simulation.
        initial_states_batch : dict[str, dict]
            Batched initial conditions with leading batch dimension.

        Returns
        -------
        SurrogateDataset
            Dataset with ``batch * (n_steps - 1)`` samples.
        """
        node_obj = gm._nodes[target_node].node
        node_dt = node_obj.delta_t

        _finals, histories = gm.run_sweep(
            n_steps, initial_states_batch, return_history=True,
        )

        node_history = histories[target_node]
        state_spec = {k: v.shape for k, v in node_obj.initial_state().items()}

        # Reconstruct boundary inputs from edges -- use first batch element
        # for shape inference, then extract from batched history
        boundary_spec, boundary_arrays = DatasetGenerator._reconstruct_boundary_batched(
            gm, target_node, histories, n_steps,
        )

        # node_history fields have shape (batch, n_steps, *field_shape)
        # Pair states at t with next_states at t+1, then flatten batch dim
        states = {}
        next_states = {}
        for k, v in node_history.items():
            # v shape: (batch, n_steps, *field_shape)
            states[k] = v[:, :-1].reshape((-1,) + v.shape[2:])
            next_states[k] = v[:, 1:].reshape((-1,) + v.shape[2:])

        boundary = {}
        for k, v in boundary_arrays.items():
            # v shape: (batch, n_steps, *field_shape)
            n_samples_per_batch = n_steps - 1
            boundary[k] = v[:, :n_samples_per_batch].reshape((-1,) + v.shape[2:])

        return SurrogateDataset(
            states=states,
            boundary_inputs=boundary,
            next_states=next_states,
            dt=node_dt,
            node_name=target_node,
            state_spec=state_spec,
            boundary_spec=boundary_spec,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_boundary(gm, target_node, history, n_steps):
        """Reconstruct boundary_inputs from edge sources and external inputs.

        Returns (boundary_spec, boundary_arrays) where boundary_arrays
        has shape (n_steps, *field_shape) for each field.
        """
        boundary_spec = {}
        boundary_arrays = {}

        # From edges
        for edge in gm._edges:
            if edge.target_node != target_node:
                continue
            source_data = history[edge.source_node][edge.source_field]
            if edge.transform is not None:
                source_data = jnp.vectorize(
                    edge.transform, signature="()->()"
                )(source_data) if source_data.ndim <= 1 else edge.transform(source_data)
            boundary_arrays[edge.target_field] = source_data
            # Infer shape (strip time axis)
            boundary_spec[edge.target_field] = source_data.shape[1:]

        # From external inputs (use zeros)
        for ei in gm._external_inputs:
            if ei.target_node != target_node:
                continue
            shape = (n_steps,) + ei.shape
            boundary_arrays[ei.target_field] = jnp.zeros(shape, dtype=ei.dtype)
            boundary_spec[ei.target_field] = ei.shape

        return boundary_spec, boundary_arrays

    @staticmethod
    def _reconstruct_boundary_batched(gm, target_node, histories, n_steps):
        """Like _reconstruct_boundary but for batched (sweep) histories.

        histories fields have shape (batch, n_steps, *field_shape).
        Returns boundary_arrays with the same leading (batch, n_steps, ...) shape.
        """
        boundary_spec = {}
        boundary_arrays = {}

        for edge in gm._edges:
            if edge.target_node != target_node:
                continue
            source_data = histories[edge.source_node][edge.source_field]
            if edge.transform is not None:
                source_data = edge.transform(source_data)
            boundary_arrays[edge.target_field] = source_data
            # shape: strip batch and time axes
            boundary_spec[edge.target_field] = source_data.shape[2:]

        for ei in gm._external_inputs:
            if ei.target_node != target_node:
                continue
            # Need to figure out batch size from any existing field
            batch_size = None
            for v in histories.values():
                for arr in v.values():
                    batch_size = arr.shape[0]
                    break
                if batch_size is not None:
                    break
            if batch_size is None:
                batch_size = 1
            shape = (batch_size, n_steps) + ei.shape
            boundary_arrays[ei.target_field] = jnp.zeros(shape, dtype=ei.dtype)
            boundary_spec[ei.target_field] = ei.shape

        return boundary_spec, boundary_arrays
