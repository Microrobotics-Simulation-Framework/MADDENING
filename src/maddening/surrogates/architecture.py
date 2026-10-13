"""
SurrogateArchitecture ABC -- the contract for neural network forward passes.

Architectures are pure JAX functions: they take weights, state, boundary
inputs and dt, and return either a new state (direct mode) or d(state)/dt
(derivative mode).  No framework dependency -- Equinox/Flax/etc. are only
used inside concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
# `PyTree` is re-exported here so the historical
# `surrogates.architecture.PyTree` import path keeps working; it and the
# state-dict aliases are defined once, in `surrogates.types`.
from maddening.surrogates.types import MutableStateDict, PyTree, StateDict

if TYPE_CHECKING:
    import jax


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
        rng_key: jax.Array,
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

    # Dynamic keys (the node's own fields): an alias, not a TypedDict.
    @abstractmethod
    def forward(
        self,
        params: PyTree,
        state: StateDict,
        boundary_inputs: StateDict,
        dt: float,
    ) -> MutableStateDict:
        """Pure JAX-traceable forward pass.

        Parameters
        ----------
        params : PyTree
            Network weights (from init_params or training).
        state : StateDict
            Current node state ``{field: array}``.
        boundary_inputs : StateDict
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
