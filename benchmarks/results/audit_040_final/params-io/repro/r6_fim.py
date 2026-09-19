import numpy as np, jax, jax.numpy as jnp
from maddening import sysid

# A residual with an exactly known Jacobian so rank/crb can be checked by hand.
def make(J):
    J = jnp.asarray(J, jnp.float32)
    def residual(p):
        x = jnp.stack([p["a"], p["b"], p["c"]])
        return J @ x
    return residual

p = {"a": jnp.float32(2.0), "b": jnp.float32(3.0), "c": jnp.float32(5.0)}

print("=== full-rank, scale=None ===")
J = np.eye(3) * [1.0, 2.0, 3.0]
r = sysid.fim(make(J), p, scale=None)
print("  eigvals", np.asarray(r.eigvals), "rank", r.rank, "cond", r.cond)
print("  crb    ", np.asarray(r.crb), "(expect 1, 1/4, 1/9)")

print("\n=== rank-deficient (c invisible) ===")
J = np.array([[1.0, 0, 0], [0, 1.0, 0]])
r = sysid.fim(make(J), p, scale=None)
print("  eigvals", np.asarray(r.eigvals), "rank", r.rank, "cond", r.cond)
print("  crb    ", np.asarray(r.crb), "names", r.param_names)
print("  least identifiable:", r.least_identifiable())

print("\n=== scale='relative' with a parameter that is exactly 0 ===")
p0 = {"a": jnp.float32(2.0), "b": jnp.float32(0.0), "c": jnp.float32(5.0)}
J = np.eye(3)
r = sysid.fim(make(J), p0, scale="relative")
print("  eigvals", np.asarray(r.eigvals), "rank", r.rank, "cond", r.cond)
print("  crb    ", np.asarray(r.crb))
print("  -> 'b' is perfectly identifiable in absolute terms; relative scaling makes it look dead")

print("\n=== degenerate noise_std ===")
for sd in (0.0, -1.0, float("nan"), float("inf"), 1e-30):
    try:
        r = sysid.fim(make(np.eye(3)), p, scale=None, noise_std=sd)
        print(f"  noise_std={sd!r:8}: rank={r.rank} cond={r.cond} eigvals={np.asarray(r.eigvals)} crb={np.asarray(r.crb)}")
    except Exception as e:
        print(f"  noise_std={sd!r:8}: raised {type(e).__name__}: {e}")

print("\n=== residual with a NaN / a non-finite Jacobian ===")
def bad(p):
    return jnp.stack([p["a"] * jnp.float32("nan"), p["b"], p["c"]])
try:
    r = sysid.fim(bad, p, scale=None)
    print("  rank", r.rank, "cond", r.cond, "eigvals", np.asarray(r.eigvals), "crb", np.asarray(r.crb))
    print("  -> silently returns a 'report' built on NaN")
except Exception as e:
    print("  raised", type(e).__name__, e)

print("\n=== zero residual function (J == 0) ===")
r = sysid.fim(lambda q: jnp.zeros(3, jnp.float32) * q["a"], p, scale=None)
print("  rank", r.rank, "cond", r.cond, "crb", np.asarray(r.crb))

print("\n=== rank_rtol validation ===")
for rt in (None, 0.0, 1e-3, -1.0, float("nan"), 1e30):
    try:
        r = sysid.fim(make(np.diag([1.0, 1e-4, 1e-9])), p, scale=None, rank_rtol=rt)
        print(f"  rank_rtol={rt!r:8}: rank={r.rank}")
    except Exception as e:
        print(f"  rank_rtol={rt!r:8}: {type(e).__name__}: {e}")

print("\n=== mask with zero-size leaf / empty selection ===")
p2 = {"a": jnp.float32(2.0), "e": jnp.zeros((0,), jnp.float32)}
def r2(q):
    return jnp.concatenate([jnp.atleast_1d(q["a"]), q["e"]])
rep = sysid.fim(r2, p2, scale=None)
print("  names", rep.param_names, "fim shape", rep.fim.shape, "rank", rep.rank)
