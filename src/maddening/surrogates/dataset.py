"""
SurrogateDataset and DatasetGenerator -- extract training data from physics
simulations for surrogate model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


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
    """Generate training datasets from physics graph simulations.

    A sample pairs the target node's state with the boundary inputs its
    edges deliver and with its next state.  The inputs are rebuilt from
    the run's state history by the graph's own edge rule (interface
    mapping, transform, additive edges, external inputs as zeros).

    **Known limit (MADD-ANO-176).**  The inputs of a sample are rebuilt
    from the *sources' states at the sample's own step*.  That is what
    the step read on a back edge; on a forward edge (the source runs
    earlier in the schedule) the step read the source's state one step
    later, so for an input that changes in time the dataset is one step
    behind what the node was fed.  Inputs that do not change are exact.
    """

    @staticmethod
    def from_graph(
        gm: GraphManager, target_node: str, n_steps: int,
    ) -> SurrogateDataset:
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

        Notes
        -----
        **Stateful: *gm*'s own state is advanced.**  This calls
        :meth:`~maddening.core.graph_manager.GraphManager.run_scan_with_history`,
        which leaves *gm* at the last state of the trajectory, so a
        second ``from_graph`` on the same graph samples a *continued*
        trajectory rather than a second one from the same initial
        condition.  Nothing fails; the dataset is simply drawn from a
        part of state space the caller did not choose.

        To generate several datasets from one initial condition, build a
        fresh :class:`~maddening.core.graph_manager.GraphManager` per
        dataset, call ``gm.reset_state()`` between calls, or snapshot
        with ``gm.save_state()`` / ``gm.load_state()`` around each one.
        :meth:`from_sweep` runs through ``run_sweep`` and leaves *gm*
        untouched.
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
        gm: GraphManager,
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

        Notes
        -----
        **Not stateful, unlike :meth:`from_graph`.**  The batch runs from
        *initial_states_batch*, which the caller supplies, and
        ``run_sweep`` writes nothing back, so *gm* is left exactly as it
        was and an identical second call returns an identical dataset.
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
        return DatasetGenerator._boundary_from_history(gm, target_node, history, lead=1)

    @staticmethod
    def _reconstruct_boundary_batched(gm, target_node, histories, n_steps):
        """Like _reconstruct_boundary but for batched (sweep) histories.

        histories fields have shape (batch, n_steps, *field_shape).
        Returns boundary_arrays with the same leading (batch, n_steps, ...) shape.
        """
        return DatasetGenerator._boundary_from_history(gm, target_node, histories, lead=2)

    @staticmethod
    def _boundary_from_history(gm, target_node, history, *, lead):
        """The target's boundary inputs at every sample of *history*, whose
        fields carry *lead* leading sample axes (time, or batch and time).

        Each sample is resolved by the graph's own edge rule
        (``GraphManager._boundary_inputs_from``, which the step's edge
        application and ``resolve_boundary_inputs`` share): the edge's
        interface mapping, then its transform, additive edges summed,
        external inputs as zeros.  The rule is applied to one state at a
        time and mapped over the sample axes, so a mapping or a transform
        sees the field it sees in the step, never the history's time axis.

        An edge that reads a flux output is a ``KeyError``: fluxes are not
        part of a state history.
        """
        sources = sorted({e.source_node for e in gm._edges if e.target_node == target_node})

        def resolve(state):
            return gm._boundary_inputs_from(state, target_node)

        if sources:
            per_sample = resolve
            for _ in range(lead):
                per_sample = jax.vmap(per_sample)
            boundary_arrays = per_sample({name: history[name] for name in sources})
        else:
            # External inputs only: nothing to map over, so give the zero
            # defaults the history's sample axes.
            sample_shape = next(
                leaf.shape[:lead] for fields in history.values() for leaf in fields.values())
            boundary_arrays = {
                k: jnp.broadcast_to(v, sample_shape + v.shape)
                for k, v in resolve({}).items()
            }
        boundary_spec = {k: v.shape[lead:] for k, v in boundary_arrays.items()}
        return boundary_spec, boundary_arrays
