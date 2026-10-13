"""
SurrogateNode -- a SimulationNode powered by a neural surrogate.

Drop-in replacement for any physics node: same state dict, same
boundary_inputs contract, works with jit/scan/grad/vmap.
"""

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import SimulationNode
from maddening.core.compliance.stability import stability
from maddening.surrogates.architecture import PyTree, SurrogateArchitecture
from maddening.surrogates.types import (
    DerivFn,
    FieldValues,
    Integrator,
    MutableStateDict,
    StateDict,
    WeightOverrides,
)


# ------------------------------------------------------------------
# Built-in integrators for derivative-mode surrogates
# ------------------------------------------------------------------

# Dynamic keys (the node's own fields): an alias, not a TypedDict --
# the key set is a runtime property of the caller's node.
def euler_integrator(
    state: StateDict,
    deriv_fn: DerivFn,
    dt: float,
) -> MutableStateDict:
    """Forward Euler: state + dt * d(state)/dt."""
    derivs = deriv_fn(state)
    return {k: state[k] + dt * derivs[k] for k in state}


# Dynamic keys, an alias not a TypedDict: see `euler_integrator`.
def rk4_integrator(
    state: StateDict,
    deriv_fn: DerivFn,
    dt: float,
) -> MutableStateDict:
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

_WEIGHTS_PREFIX = "weights"


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

    Notes
    -----
    The weights are reachable through the graph parameter pytree:
    :meth:`params_pytree` exposes every floating leaf of ``weights`` as
    a flat entry ``"weights<path>"`` (e.g. ``"weights['scale']"``) so
    ``gm.params["nodes"][name]`` stays a flat dict of arrays (what
    checkpoints and the REST endpoint expect), and :meth:`update`
    rebuilds the weights pytree from the injected leaves.  ``jax.grad``
    of a trajectory loss with respect to ``gm.params`` therefore reaches
    the surrogate weights, and fine-tuned weights take effect without
    a recompile.
    """

    def __init__(
        self,
        name: str,
        timestep: float,
        architecture: SurrogateArchitecture,
        weights: PyTree,
        state_spec: dict[str, tuple],
        boundary_spec: dict[str, tuple],
        initial_values: FieldValues,
        integrator: Integrator | None = None,
    ) -> None:
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

    def halo_width(self) -> dict[int, int]:
        """Surrogate nodes are pointwise (no spatial neighbour access)."""
        return {}

    def initial_state(self) -> dict:
        vals = self.params["initial_values"]
        return {
            field: jnp.asarray(vals[field], dtype=jnp.float32)
            for field in self.state_spec
        }

    # -- weights <-> flat params leaves -------------------------------

    def _weight_leaves(self):
        """``(paths, leaves, treedef)`` of the weights pytree."""
        flat, treedef = jax.tree_util.tree_flatten_with_path(self.params["weights"])
        paths = [jax.tree_util.keystr(path) for path, _ in flat]
        leaves = [leaf for _, leaf in flat]
        return paths, leaves, treedef

    @staticmethod
    def _is_float_leaf(leaf) -> bool:
        try:
            return jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
        except (TypeError, ValueError):
            return False

    def params_pytree(self) -> dict:
        """Floating leaves of ``weights`` as flat ``"weights<path>"``
        entries (plus any float-valued scalar in ``self.params``)."""
        out = {
            k: v for k, v in super().params_pytree().items()
            if k != _WEIGHTS_PREFIX
        }
        paths, leaves, _ = self._weight_leaves()
        for path, leaf in zip(paths, leaves):
            if self._is_float_leaf(leaf):
                out[f"{_WEIGHTS_PREFIX}{path}"] = jnp.asarray(leaf)
        return out

    def _resolve_weights(self, params):
        if params is None:
            return self.params["weights"]
        paths, leaves, treedef = self._weight_leaves()
        merged = [
            params.get(f"{_WEIGHTS_PREFIX}{path}", leaf)
            for path, leaf in zip(paths, leaves)
        ]
        return jax.tree_util.tree_unflatten(treedef, merged)

    # Dynamic keys: an alias, not a TypedDict.  `params` is keyed by the
    # weight pytree's own leaf paths, computed at call time (see
    # `_weight_leaves`), and `state` by the surrogated node's fields.
    def update(
        self,
        state: StateDict,
        boundary_inputs: StateDict,
        dt: float,
        *,
        params: WeightOverrides | None = None,
    ) -> MutableStateDict:
        weights = self._resolve_weights(params)
        arch = self.architecture

        if arch.mode == "direct":
            return arch.forward(weights, state, boundary_inputs, dt)
        else:
            # derivative mode: integrate d(state)/dt
            def deriv_fn(s: StateDict) -> MutableStateDict:
                return arch.forward(weights, s, boundary_inputs, dt)
            return self._integrator(state, deriv_fn, dt)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["architecture"] = type(self.architecture).__name__
        d["mode"] = self.architecture.mode
        # Weights are not serialised inline -- they go to .npz
        d.pop("params", None)
        return d
