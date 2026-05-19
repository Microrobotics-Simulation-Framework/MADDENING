"""
HybridNode -- physics node with a pluggable additive correction.

Wraps an existing :class:`SimulationNode` and adds a correction term
to its output.  The correction can be any JAX-traceable function
``(state, boundary_inputs, dt) -> {field: correction_array}``.

This is the building block for learned integration correctors:
train an MLP (or any function) to predict the integration error
at coarse dt, then add it to the coarse physics output to match
fine-dt accuracy.

The HybridNode delegates ``initial_state``, ``boundary_input_spec``,
``compute_boundary_fluxes``, ``interface_dof_indices``, and
``compute_interface_correction`` to the wrapped physics node,
so it is a drop-in replacement in the graph.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from maddening.core.node import SimulationNode


class HybridNode(SimulationNode):
    """A physics node augmented with an additive correction function.

    Parameters
    ----------
    physics_node : SimulationNode
        The underlying physics node.
    correction_fn : callable
        ``(state, boundary_inputs, dt) -> {field: correction_array}``
        A JAX-traceable function returning additive corrections
        for each field.  Fields not in the dict get zero correction.
    name : str or None
        Optional override name.  Defaults to the physics node's name.
    """

    def __init__(
        self,
        physics_node: SimulationNode,
        correction_fn: Callable,
        name: Optional[str] = None,
    ):
        super().__init__(
            name=name or physics_node.name,
            timestep=physics_node.delta_t,
            **physics_node.params,
        )
        self.physics_node = physics_node
        self.correction_fn = correction_fn

    def halo_width(self) -> dict[int, int]:
        """Delegates to the wrapped physics node."""
        return self.physics_node.halo_width()

    def initial_state(self) -> dict:
        return self.physics_node.initial_state()

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Physics update + additive correction."""
        physics_result = self.physics_node.update(state, boundary_inputs, dt)
        correction = self.correction_fn(state, boundary_inputs, dt)
        result = {}
        for k in physics_result:
            if k in correction:
                result[k] = physics_result[k] + correction[k]
            else:
                result[k] = physics_result[k]
        return result

    def state_fields(self) -> list[str]:
        return self.physics_node.state_fields()

    def boundary_input_spec(self):
        return self.physics_node.boundary_input_spec()

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return self.physics_node.compute_boundary_fluxes(
            state, boundary_inputs, dt
        )

    def interface_dof_indices(self):
        return self.physics_node.interface_dof_indices()

    def compute_interface_correction(self, pre_state, boundary_inputs, dt):
        return self.physics_node.compute_interface_correction(
            pre_state, boundary_inputs, dt
        )

    def to_dict(self):
        d = self.physics_node.to_dict()
        d["type"] = "HybridNode"
        d["physics_type"] = type(self.physics_node).__name__
        return d


def generate_correction_data(
    node: SimulationNode,
    dt_coarse: float,
    dt_fine: float,
    n_steps: int,
    initial_state: Optional[dict] = None,
    boundary_inputs: Optional[dict] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate training data for a learned integration corrector.

    Runs the node at both coarse and fine timesteps, computing the
    error between them.

    Parameters
    ----------
    node : SimulationNode
        The physics node to generate data for.
    dt_coarse : float
        Coarse timestep.
    dt_fine : float
        Fine timestep (used as ground truth).
    n_steps : int
        Number of coarse steps to run.
    initial_state : dict or None
        Starting state.  Uses ``node.initial_state()`` if None.
    boundary_inputs : dict or None
        Boundary inputs (static).  Uses empty dict if None.

    Returns
    -------
    (inputs, targets, states) : tuple of lists
        ``inputs[i]`` = (state, boundary_inputs, dt_coarse)
        ``targets[i]`` = {field: correction_array}
        ``states[i]`` = state after i coarse steps (fine trajectory)
    """
    import jax.numpy as jnp

    if initial_state is None:
        initial_state = node.initial_state()
    if boundary_inputs is None:
        boundary_inputs = {}

    # Run fine trajectory
    fine_steps_per_coarse = max(round(dt_coarse / dt_fine), 1)
    actual_dt_fine = dt_coarse / fine_steps_per_coarse

    inputs = []
    targets = []
    states = []

    state_fine = initial_state
    state_coarse = initial_state

    for step in range(n_steps):
        # Record input
        inputs.append({
            "state": state_coarse,
            "boundary_inputs": boundary_inputs,
            "dt": dt_coarse,
        })

        # Fine trajectory: take fine_steps_per_coarse sub-steps
        s = state_coarse  # Start from same state
        for _ in range(fine_steps_per_coarse):
            s = node.update(s, boundary_inputs, actual_dt_fine)
        state_fine_result = s

        # Coarse trajectory: single step
        state_coarse_result = node.update(
            state_coarse, boundary_inputs, dt_coarse
        )

        # Correction = fine - coarse
        correction = {}
        for field in state_coarse_result:
            correction[field] = (
                state_fine_result[field] - state_coarse_result[field]
            )

        targets.append(correction)
        states.append(state_fine_result)

        # Advance using fine trajectory as ground truth
        state_coarse = state_fine_result

    return inputs, targets, states
