"""Probe: degenerate active sets (empty, all-true, ties) through AdaptiveNode."""
import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from maddening.core.solver_utils import ift_linear_solve
from maddening.core.params import ParamSpec
from maddening.nodes.adaptive import AdaptiveNode


class Knob(AdaptiveNode):
    """Selection rule is a constructor-chosen constant mask."""
    def __init__(self, mask_kind="empty", n=8, **kw):
        self._kind = mask_kind
        super().__init__("knob", 1.0, n_max=n, theta=0.5, blindness_gate=False, **kw)
        self._n = n
        self.d = jnp.arange(1.0, n + 1.0)

    def param_specs(self):
        return {**super().param_specs(), "theta": ParamSpec()}

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        if self._kind == "empty":
            return jnp.zeros(self._n, dtype=bool)
        if self._kind == "all":
            return jnp.ones(self._n, dtype=bool)
        if self._kind == "half":
            return jnp.arange(self._n) < self._n // 2
        raise AssertionError

    def solve_frozen(self, state, mask, params):
        diag = jnp.where(mask, self.d * params["theta"], 1.0)
        rhs = jnp.where(mask, jnp.ones(self._n), 0.0)
        return {"c": ift_linear_solve(lambda v: diag * v, rhs, solver="dense")}

    def objective(self, state, params):
        return jnp.sum(state["c"])


for kind in ("empty", "half", "all"):
    n = Knob(kind)
    st = n.initial_state()
    print(f"--- mask_kind={kind}")
    print("   c      =", jnp.asarray(st["c"]))
    print("   mask   =", jnp.asarray(st["mask"]).astype(int))
    print("   n_active =", int(jnp.sum(st["mask"])))
    # gradient of the objective wrt theta through update
    def J(theta):
        out = n.update(st, {}, 1.0, params={"theta": theta})
        return n.objective(out, {**n.params, "theta": theta})
    g = jax.grad(J)(jnp.asarray(0.5))
    print("   dJ/dtheta =", g)
    # diagnostics
    print("   ratio  =", n.gradient_capture_ratio(st))
    try:
        print("   trapped=", n.is_trapped_at(st))
    except Exception as e:
        print("   trapped raised:", type(e).__name__, e)
    try:
        sb = n.symmetry_break(st)
        print("   symmetry_break ->", sb)
    except Exception as e:
        print("   symmetry_break raised:", type(e).__name__, e)
