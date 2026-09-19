"""FINDING: fim()'s documented fail-safe (`crb` is +inf for an unidentifiable
parameter) inverts to crb == 0.0 whenever the Fisher matrix is NaN."""
import numpy as np, jax.numpy as jnp
from maddening import sysid

p = {"a": jnp.float32(2.0), "b": jnp.float32(3.0)}
def residual(q):
    return jnp.stack([q["a"], q["b"]])            # perfectly identifiable, F = I

print("healthy:          ", end="")
r = sysid.fim(residual, p, scale=None)
print(f"rank={r.rank} cond={r.cond} crb={np.asarray(r.crb)}")

print("noise_std = 0.0:  ", end="")
r = sysid.fim(residual, p, scale=None, noise_std=0.0)
print(f"rank={r.rank} cond={r.cond} eigvals={np.asarray(r.eigvals)} crb={np.asarray(r.crb)}")
print("   FIMReport says crb is '+inf ... because it fails safe: crb < threshold is then")
print("   False for an unidentifiable parameter'.  Here every eigenvalue is NaN, rank is 0,")
print("   and crb is 0.0 -- the *most* trustworthy value it can report.")
assert r.rank == 0 and np.all(np.asarray(r.crb) == 0.0)
print("   crb < 1e-6  ->", bool(np.all(np.asarray(r.crb) < 1e-6)), " (a caller's 'is it identified?' test passes)")

print()
print("noise_std = -1.0: ", end="")
r = sysid.fim(residual, p, scale=None, noise_std=-1.0)
print(f"rank={r.rank} crb={np.asarray(r.crb)}   (a negative sigma is accepted silently)")

print()
print("NaN in the residual:", end=" ")
r = sysid.fim(lambda q: jnp.stack([q["a"] * jnp.float32("nan"), q["b"]]), p, scale=None)
print(f"rank={r.rank} cond={r.cond} crb={np.asarray(r.crb)}")
