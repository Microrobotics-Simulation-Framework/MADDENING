"""
SurrogateArchitecture ABC -- the contract for neural network forward passes.

Architectures are pure JAX functions: they take weights, state, boundary
inputs and dt, and return either a new state (direct mode) or d(state)/dt
(derivative mode).  No framework dependency -- Equinox/Flax/etc. are only
used inside concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Any

from maddening.core.metadata import StabilityLevel
from maddening.core.stability import stability

PyTree = Any


@stability(StabilityLevel.EXPERIMENTAL)
class SurrogateArchitecture(ABC):
    """Abstract base for surrogate model architectures.

    Parameters
    ----------
    mode : str
        ``"direct"`` -- forward() returns new_state.
        ``"derivative"`` -- forward() returns d(state)/dt.
    """

    mode: str  # "direct" or "derivative"

    @abstractmethod
    def init_params(
        self,
        rng_key,
        state_spec: dict[str, tuple],
        boundary_spec: dict[str, tuple],
    ) -> PyTree:
        """Initialise network weights.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
        state_spec : dict mapping field name -> shape tuple
        boundary_spec : dict mapping field name -> shape tuple

        Returns
        -------
        PyTree of JAX arrays (the network weights).
        """
        ...

    @abstractmethod
    def forward(
        self,
        params: PyTree,
        state: dict,
        boundary_inputs: dict,
        dt: float,
    ) -> dict:
        """Pure JAX-traceable forward pass.

        Parameters
        ----------
        params : PyTree
            Network weights (from init_params or training).
        state : dict
            Current node state ``{field: array}``.
        boundary_inputs : dict
            Boundary inputs ``{field: array}``.
        dt : float
            Timestep (scalar).

        Returns
        -------
        dict
            If mode == "direct": new state dict (same keys as state).
            If mode == "derivative": d(state)/dt dict (same keys as state).
        """
        ...
