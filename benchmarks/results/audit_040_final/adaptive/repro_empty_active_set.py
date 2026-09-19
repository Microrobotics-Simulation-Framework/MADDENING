"""Repro: a cold-start selection rule that scores |c| selects nothing, and the
base class diagnoses it as a Palais symmetry trap."""
import sys, warnings
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from maddening.core.solver_utils import ift_linear_solve
from maddening.core.params import ParamSpec
from maddening.nodes.adaptive import AdaptiveNode, AdaptiveNodeBlindnessError


class CoefficientThresholdNode(AdaptiveNode):
    """Wavelet-style rule: keep candidates whose current coefficient is large.

    Textbook adaptive-wavelet selection (Vasilyev & Paolucci: threshold the
    coefficients).  At cold start ``c`` is all zeros, so nothing is selected.
    """
    def __init__(self, n=16, eps=1e-3, **kw):
        super().__init__("wave", 1.0, n_max=n, theta=0.5, **kw)
        self._n = n
        self._eps = eps
        self.d = jnp.arange(1.0, n + 1.0)

    def param_specs(self):
        return {**super().param_specs(), "theta": ParamSpec()}

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        return jnp.abs(state["c"]) > self._eps        # <- empty at cold start

    def solve_frozen(self, state, mask, params):
        diag = jnp.where(mask, self.d * params["theta"], 1.0)
        rhs = jnp.where(mask, jnp.ones(self._n), 0.0)
        return {"c": ift_linear_solve(lambda v: diag * v, rhs, solver="dense")}

    def objective(self, state, params):
        return jnp.sum(state["c"])


node = CoefficientThresholdNode()
try:
    st = node.initial_state()
    print("initial_state returned; n_active =", int(jnp.sum(st["mask"])))
except AdaptiveNodeBlindnessError as e:
    print("AdaptiveNodeBlindnessError raised by initial_state():\n")
    print(str(e))

print("\n--- ground truth about the point ---")
n2 = CoefficientThresholdNode(blindness_gate=False)
st2 = n2.initial_state()
print("n_active           =", int(jnp.sum(st2["mask"])))
print("gradient_capture_ratio =", n2.gradient_capture_ratio(st2))
print("is_trapped_at      =", n2.is_trapped_at(st2))
print("dJ/dtheta (frozen) =", jax.grad(lambda t: n2.objective(n2.update(st2, {}, 1.0, params={'theta': t}), {**n2.params, 'theta': t}))(jnp.asarray(0.5)))
print("the problem has no symmetry at all: the operator is diag(1..16)*theta")
