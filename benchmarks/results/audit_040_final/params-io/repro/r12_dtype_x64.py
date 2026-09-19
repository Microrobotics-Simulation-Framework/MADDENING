"""Dtype honesty under JAX_ENABLE_X64=1: does a float64 user parameter survive?"""
import jax
print("x64 enabled:", jax.config.jax_enable_x64)
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening import sysid


class N(SimulationNode):
    def __init__(self, name, timestep=0.01, **kw):
        super().__init__(name, timestep); self.params = dict(kw)
    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float64)}
    def update(self, state, boundary_inputs, dt, params=None):
        p = params or self.params
        return {"x": state["x"] + dt * jnp.asarray(p["gain"]).sum()}

VAL = 0.1234567890123456789          # needs float64 to survive
gm = GraphManager()
gm.add_node(N("n", 0.01,
              gain=np.float64(VAL),                        # numpy float64 scalar
              arr64=np.array([VAL, 2 * VAL], dtype=np.float64),   # float64 array
              pyfloat=VAL,                                 # Python float
              lst=[VAL, 2 * VAL],                          # list of Python floats
              arr32=np.array([VAL], dtype=np.float32)))
gm.compile()
print("\nparams_pytree dtypes after compile:")
for k, v in sorted(gm.params["nodes"]["n"].items()):
    print(f"  {k:9s} dtype={np.dtype(v.dtype).name:8s} value={np.asarray(v).ravel()[0]!r}")
print(f"  (the user's value is {VAL!r})")

print("\nConfig to_dict/from_dict:")
cfg = gm.to_dict()
print("  stored:", {k: (v if not isinstance(v, list) else v[:1])
                    for k, v in cfg["nodes"][0]["params"].items()})
gm2 = GraphManager.from_dict(cfg, {"N": N}); gm2.compile()
for k in sorted(gm.params["nodes"]["n"]):
    a, b = gm.params["nodes"]["n"][k], gm2.params["nodes"]["n"][k]
    print(f"  {k:9s} {np.dtype(a.dtype).name:8s} -> {np.dtype(b.dtype).name:8s} "
          f"bits_equal={np.asarray(a).tobytes()==np.asarray(b).tobytes()}")

print("\nunconstrain/constrain dtype under x64:")
spec = ParamSpec(bounds=(0.0, None), transform="log")
p = jnp.asarray(VAL, jnp.float64)
u = spec.to_unconstrained(p); q = spec.to_constrained(u)
print(f"  float64 log: p.dtype={np.dtype(p.dtype).name} u.dtype={np.dtype(u.dtype).name} "
      f"q.dtype={np.dtype(q.dtype).name}  rel_err={abs(float(q)-VAL)/VAL:.3e}")
spec2 = ParamSpec(bounds=(0.0, 1.0), transform="logit")
q2 = spec2.to_constrained(spec2.to_unconstrained(p))
print(f"  float64 logit: q.dtype={np.dtype(q2.dtype).name} rel_err={abs(float(q2)-VAL)/VAL:.3e}")

print("\nFMI model description start-value precision:")
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER)
for v in md.variables:
    if v.causality == "parameter":
        print(f"  {v.name:16s} dtype={v.dtype:8s} start={v.start!r:26.26}")

print("\nFMI wire (float64 only) round trip of a float64 parameter:")
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                              initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))
br = FmuTcpBridge(sc, md, master_dt=0.01)
vr = {v.name: v.value_reference for v in md.variables}
for nm in ("n.params.gain", "n.params.pyfloat"):
    br.handle({"op": "set", "vr": [vr[nm]], "values": [VAL]})
    got = br.handle({"op": "get", "vr": [vr[nm]]})["values"][0]
    print(f"  {nm:16s} set {VAL!r} -> get {got!r}  exact={got == VAL}")
