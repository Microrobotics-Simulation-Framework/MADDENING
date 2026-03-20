"""
SurrogateNode -- a SimulationNode powered by a neural surrogate.

Drop-in replacement for any physics node: same state dict, same
boundary_inputs contract, works with jit/scan/grad/vmap.
"""

import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import SimulationNode
from maddening.core.compliance.stability import stability
from maddening.surrogates.architecture import SurrogateArchitecture


# ------------------------------------------------------------------
# Built-in integrators for derivative-mode surrogates
# ------------------------------------------------------------------

def euler_integrator(state: dict, deriv_fn, dt: float) -> dict:
    """Forward Euler: state + dt * d(state)/dt."""
    derivs = deriv_fn(state)
    return {k: state[k] + dt * derivs[k] for k in state}


def rk4_integrator(state: dict, deriv_fn, dt: float) -> dict:
    """Classical 4th-order Runge-Kutta over a state dict."""
    k1 = deriv_fn(state)
    s2 = {k: state[k] + 0.5 * dt * k1[k] for k in state}
    k2 = deriv_fn(s2)
    s3 = {k: state[k] + 0.5 * dt * k2[k] for k in state}
    k3 = deriv_fn(s3)
    s4 = {k: state[k] + dt * k3[k] for k in state}
    k4 = deriv_fn(s4)
    return {
        k: state[k] + (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k])
        for k in state
    }


# ------------------------------------------------------------------
# SurrogateNode
# ------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
class SurrogateNode(SimulationNode):
    """A simulation node backed by a trained neural surrogate.

    Parameters
    ----------
    name : str
        Node name (must match the original node being replaced).
    timestep : float
        Simulation timestep.
    architecture : SurrogateArchitecture
        The neural network architecture descriptor.
    weights : PyTree
        Trained network weights (stored in ``self.params["weights"]``).
    state_spec : dict[str, tuple]
        ``{field_name: shape}`` describing the state dict.
    boundary_spec : dict[str, tuple]
        ``{field_name: shape}`` describing boundary inputs.
    initial_values : dict[str, float | array]
        Initial values for each state field.
    integrator : callable, optional
        Integration function for derivative mode.  Signature:
        ``(state, deriv_fn, dt) -> new_state``.
        Defaults to :func:`euler_integrator`.
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        architecture: SurrogateArchitecture,
        weights,
        state_spec: dict[str, tuple],
        boundary_spec: dict[str, tuple],
        initial_values: dict,
        integrator=None,
    ):
        super().__init__(
            name,
            timestep,
            weights=weights,
            state_spec=state_spec,
            boundary_spec=boundary_spec,
            initial_values=initial_values,
        )
        self.architecture = architecture
        self.boundary_spec = boundary_spec
        self.state_spec = state_spec
        self._integrator = integrator or euler_integrator

    @property
    def requires_halo(self) -> bool:
        """Surrogate nodes are pointwise (no spatial neighbor access)."""
        return False

    def initial_state(self) -> dict:
        vals = self.params["initial_values"]
        return {
            field: jnp.asarray(vals[field], dtype=jnp.float32)
            for field in self.state_spec
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        weights = self.params["weights"]
        arch = self.architecture

        if arch.mode == "direct":
            return arch.forward(weights, state, boundary_inputs, dt)
        else:
            # derivative mode: integrate d(state)/dt
            def deriv_fn(s):
                return arch.forward(weights, s, boundary_inputs, dt)
            return self._integrator(state, deriv_fn, dt)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["architecture"] = type(self.architecture).__name__
        d["mode"] = self.architecture.mode
        # Weights are not serialised inline -- they go to .npz
        d.pop("params", None)
        return d
