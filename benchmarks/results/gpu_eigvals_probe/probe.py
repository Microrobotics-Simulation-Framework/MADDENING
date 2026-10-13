"""One-off local GPU probe (maintainer-approved, 2026-10-07): does the
non-symmetric eigenvalue solve that coupling diagnostics now use run on a
GPU, inside jit / scan / vmap, and does a coupled group with diagnostics on
give the same report on GPU as on CPU?

Run twice: PROBE_PLATFORM=cpu and PROBE_PLATFORM=gpu (sets JAX_PLATFORMS).
Each run writes probe_<platform>.json; compare.py diffs them.
"""
import json, os, sys, time
plat = os.environ["PROBE_PLATFORM"]
os.environ["JAX_PLATFORMS"] = "cuda,cpu" if plat == "gpu" else "cpu"
x64 = os.environ.get("PROBE_X64") == "1"
import jax
if x64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

out = {"platform": plat, "x64": x64, "jax": jax.__version__,
       "default_backend": jax.default_backend(),
       "devices": [str(d) for d in jax.devices()]}
print(out, flush=True)

def record(name, fn):
    t = time.perf_counter()
    try:
        v = fn()
        out[name] = {"ok": True, "value": v, "seconds": round(time.perf_counter() - t, 3)}
    except Exception as e:  # the probe's point is to see what raises
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "seconds": round(time.perf_counter() - t, 3)}
    print(name, json.dumps(out[name])[:400], flush=True)

rng = np.random.default_rng(0)
dt = jnp.float64 if x64 else jnp.float32
mats = {n: jnp.asarray(rng.standard_normal((n, n)) * 0.4, dt) for n in (2, 5, 9)}
# a non-normal one: graded upper Hessenberg, the shape the estimator compresses to
H = np.triu(rng.standard_normal((9, 9)), -1) * (0.7 ** np.arange(9))[:, None]
mats["hess9"] = jnp.asarray(H, dt)

def radius(M):
    return jnp.max(jnp.abs(jnp.linalg.eigvals(M)))

for k, M in mats.items():
    record(f"eigvals_eager_{k}", lambda M=M: float(radius(M)))
    record(f"eigvals_jit_{k}", lambda M=M: float(jax.jit(radius)(M)))
ref = {k: float(np.max(np.abs(np.linalg.eigvals(np.asarray(M, np.float64))))) for k, M in mats.items()}
out["numpy_float64_reference"] = ref

M9 = mats["hess9"]
record("eigvals_in_scan", lambda: [float(x) for x in jax.jit(
    lambda M: jax.lax.scan(lambda c, s: (c, radius(M * s)), 0.0, jnp.asarray([0.5, 1.0, 1.5], dt))[1])(M9)])
record("eigvals_in_vmap", lambda: [float(x) for x in jax.jit(jax.vmap(radius))(
    jnp.stack([M9 * s for s in (0.5, 1.0, 1.5)]))])
record("eigvals_in_cond", lambda: float(jax.jit(
    lambda M: jax.lax.cond(M[0, 0] > -1e9, radius, lambda m: jnp.zeros((), m.dtype), M))(M9)))

# the estimator's own function
from maddening.core.coupling import acceleration as acc
record("estimator_spectral_radius", lambda: float(jax.jit(acc._spectral_radius)(M9)))
record("estimator_squaring_fallback", lambda: float(jax.jit(acc._spectral_radius_small)(M9)))

# a stock coupled pair with diagnostics on and off
from maddening import GraphManager
from maddening.nodes import SpringDamperNode

def pair(diagnostics):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("left", 0.01, stiffness=30.0, damping=2.0, initial_position=1.5))
    gm.add_node(SpringDamperNode("right", 0.01, stiffness=30.0, damping=2.0))
    gm.add_edge("left", "right", "position", "anchor_position")
    gm.add_edge("right", "left", "position", "anchor_position")
    gm.add_coupling_group(["left", "right"], max_iterations=8, tolerance=1e-6,
                          solver="ift", diagnostics=diagnostics)
    gm.compile()
    return gm

def run_pair(diagnostics, n=40):
    gm = pair(diagnostics)
    t = time.perf_counter(); gm.step(); first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(n - 1):
        gm.step()
    per = (time.perf_counter() - t) / (n - 1)
    st = {k: {f: np.asarray(v).tolist() for f, v in d.items()} for k, d in gm._state.items()
          if not k.startswith("_")}
    rep = {}
    if diagnostics:
        for g, r in gm.coupling_diagnostics().items():
            rep[str(g)] = {k: (np.asarray(v).tolist() if not isinstance(v, (str, bool, type(None))) else v)
                           for k, v in r.items()}
    return {"first_step_s": round(first, 3), "per_step_ms": round(per * 1e3, 3), "state": st, "report": rep}

record("pair_diagnostics_off", lambda: run_pair(False))
record("pair_diagnostics_on", lambda: run_pair(True))
record("pair_diagnostics_on_run_scan", lambda: (lambda gm: (gm.run_scan(20), {str(g): {k: (np.asarray(v).tolist() if not isinstance(v, (str, bool, type(None))) else v) for k, v in r.items()} for g, r in gm.coupling_diagnostics().items()})[1])(pair(True)))

json.dump(out, open(f"probe_{plat}{'_x64' if x64 else ''}.json", "w"), indent=1, default=str)
print("written", flush=True)
