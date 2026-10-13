"""fim's param_names / mask index alignment with array and zero-size leaves."""
import numpy as np, jax, jax.numpy as jnp
from maddening import sysid

params = {
    "a": jnp.asarray([1.0, 2.0, 3.0], jnp.float32),   # 3 elements
    "b": jnp.zeros((0,), jnp.float32),                 # 0 elements
    "c": jnp.asarray(7.0, jnp.float32),                # 1 element
    "d": jnp.asarray([[1.0, 2.0]], jnp.float32),       # 2 elements
}
W = jnp.asarray(np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[:, :6], jnp.float32)

def residual(p):
    flat = jnp.concatenate([p["a"].ravel(), p["b"].ravel(),
                            jnp.atleast_1d(p["c"]).ravel(), p["d"].ravel()])
    return W @ flat

r = sysid.fim(residual, params, scale=None)
print("names:", r.param_names)
print("eigvals:", np.asarray(r.eigvals))
print("crb:", np.asarray(r.crb), " (expect 1, 1/4, 1/9, 1/16, 1/25, 1/36)")
print("expected:", [1/ (i+1)**2 for i in range(6)])

# mask only 'c' and 'd'
mask = {"a": False, "b": False, "c": True, "d": True}
r2 = sysid.fim(residual, params, scale=None, mask=mask)
print()
print("masked names:", r2.param_names)
print("masked crb  :", np.asarray(r2.crb), " (expect 1/16, 1/25, 1/36 for c,d[0],d[1])")

# mask only 'b' (zero-size)
try:
    r3 = sysid.fim(residual, params, scale=None, mask={"a": False, "b": True, "c": False, "d": False})
    print()
    print("mask={b} only -> names", r3.param_names, "fim", np.asarray(r3.fim).shape,
          "rank", r3.rank, "cond", r3.cond, "crb", np.asarray(r3.crb))
    print("  ('mask selects no parameters' is NOT raised: a zero-size leaf marked True")
    print("   passes the emptiness test but contributes no indices.)")
except Exception as e:
    print("\nmask={b} only ->", type(e).__name__, e)
