"""Repro: compute_active_set's *shape* is checked, its *dtype* is not.
A subclass that returns scores (or an argsort permutation) instead of a
boolean mask is silently reinterpreted as `!= 0`, and every diagnostic
reports a healthy node."""
import sys, warnings
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING-wt/audit/adaptive/tests")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from nodes.adaptive._toys import PoissonSineTopKNode


class ReturnsScores(PoissonSineTopKNode):
    """The author forgot the comparison and returned the score array."""
    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        return jnp.abs(self.rhs(params))            # float, not bool


class ReturnsPermutation(PoissonSineTopKNode):
    """The author reached for argsort and forgot to slice + scatter."""
    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        return jnp.argsort(jnp.abs(self.rhs(params)))   # int, shape (n,)


for cls in (PoissonSineTopKNode, ReturnsScores, ReturnsPermutation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        node = cls(n=64, k=8, theta=0.42)
        st = node.initial_state()
        ratio = node.gradient_capture_ratio(st)
    print(f"{cls.__name__:22s} n_active={int(st['mask'].sum()):3d}/64  "
          f"ratio={ratio:.4f}  trapped={node.is_trapped_at(st)}  "
          f"J={float(node.objective(st, node.params)): .10f}")
print("\nno error, no warning: the shape check at base.py:794 and base.py:1034 passes,")
print("and jnp.asarray(x, dtype=bool) turns every nonzero entry into an active mode.")
