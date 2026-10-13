"""ParamSpec / params round-trip fidelity: awkward leaves."""
import json, math
import numpy as np
import jax.numpy as jnp
from maddening.core.params import ParamSpec

print("--- ParamSpec.to_dict/from_dict fidelity ---")
cases = [
    ParamSpec(trainable=False, bounds=(0.0, None), transform="log", description="d", units="m"),
    ParamSpec(bounds=(-1.0, 1.0), transform="logit"),
    ParamSpec(bounds=(None, None), transform=None, description="unicode ° µ ✓", units="kg·m/s²"),
]
for s in cases:
    d = s.to_dict()
    back = ParamSpec.from_dict(json.loads(json.dumps(d)))
    print(f"  {s} -> {back}  equal={s == back}")

# awkward bounds
print("--- bounds numeric fidelity ---")
s = ParamSpec(bounds=(1e-320, 1e308))         # subnormal lo
b = ParamSpec.from_dict(json.loads(json.dumps(s.to_dict())))
print("  subnormal/huge:", s.bounds, "->", b.bounds, "equal=", s.bounds == b.bounds)

# transform="log" with an integer-typed lower bound
s = ParamSpec(bounds=(0, None), transform="log")
b = ParamSpec.from_dict(json.loads(json.dumps(s.to_dict())))
print("  int bound 0:", s.bounds, "->", b.bounds, "equal=", s == b, "(int vs float)")

s = ParamSpec(bounds=(1, 2))
b = ParamSpec.from_dict(json.loads(json.dumps(s.to_dict())))
print("  int bounds (1,2):", s.bounds, "->", b.bounds, "equal=", s == b)

print("--- unconstrain/constrain round trip dtype ---")
for dt in (jnp.float32, jnp.float64, jnp.int32, jnp.bfloat16, jnp.float16):
    for spec in (ParamSpec(), ParamSpec(bounds=(0.0, 100.0)),
                 ParamSpec(bounds=(0.0, None), transform="log"),
                 ParamSpec(bounds=(0.0, 100.0), transform="logit")):
        try:
            p = jnp.asarray(3.5, dtype=dt)
        except Exception as e:
            print(f"  {dt.__name__}: cannot create ({e})"); break
        try:
            u = spec.to_unconstrained(p)
            q = spec.to_constrained(u)
            print(f"  dtype={np.dtype(p.dtype).name:9s} transform={spec.transform!s:6s} "
                  f"bounds={spec.bounds} -> u.dtype={np.dtype(u.dtype).name:9s} "
                  f"p'.dtype={np.dtype(q.dtype).name:9s} p'={float(q)!r}")
        except Exception as e:
            print(f"  dtype={dt} spec={spec}: raised {type(e).__name__}: {e}")
