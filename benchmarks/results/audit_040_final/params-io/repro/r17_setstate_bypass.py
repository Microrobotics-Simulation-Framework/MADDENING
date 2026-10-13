"""FINDING: `set_state` bypasses every value check `set` applies.

`FmuTcpBridge._set` -> `_in_dtype` refuses a value the variable's dtype cannot
hold ("a float32 input set to 1e308 is refused, not stored as inf") and
`FmuSidecar.set_params` refuses a parameter outside its declared
`ParamSpec.bounds` ("an importer cannot silently tune a constant the graph
declares invalid").  `_decode_state` checks only the token, the key set and
the shapes, then casts with `jnp.asarray(arr, dtype=live.dtype)`.
"""
import base64, io
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes.spring import SpringDamperNode

DT = 1e-2
gm = GraphManager()
gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, damping=2.0, mass=1.5,
                             initial_position=0.5))
gm.compile()
gm.set_param_spec("spring", "mass", ParamSpec(bounds=(0.1, 10.0), transform="logit"))
md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER)
sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                              initial_state=gm._state, params=gm.params,
                              param_specs=gm.param_specs()))
br = FmuTcpBridge(sc, md, master_dt=DT)
vr = {v.name: v.value_reference for v in md.variables}

print("=== what the documented door refuses ===")
print("  set mass = -1.0     ->", br.handle({"op": "set", "vr": [vr["spring.params.mass"]],
                                             "values": [-1.0]}))
print("  set mass = 1e308    ->", br.handle({"op": "set", "vr": [vr["spring.params.mass"]],
                                             "values": [1e308]}))
print("  set position = 1e308->", br.handle({"op": "set", "vr": [vr["spring.anchor_position"]],
                                             "values": [1e308]})
      if "spring.anchor_position" in vr else "(no input)")

print()
print("=== the same values through set_state ===")
blob = base64.b64decode(br.handle({"op": "get_state"})["state"])
z = np.load(io.BytesIO(blob), allow_pickle=False)
members = {k: z[k] for k in z.files}
print("  members:", sorted(members))
members["p/nodes/spring/mass"] = np.array(-1.0, dtype=np.float32)      # outside (0.1, 10.0)
members["s/spring/position"] = np.array(1e300, dtype=np.float64)        # float32 field <- 1e300
members["s/spring/velocity"] = np.array(float("nan"), dtype=np.float32)
buf = io.BytesIO(); np.savez(buf, **members)
print("  set_state ->", br.handle({"op": "set_state",
                                   "state": base64.b64encode(buf.getvalue()).decode()}))
print("  sidecar params['nodes']['spring']['mass'] =",
      float(sc.params["nodes"]["spring"]["mass"]))
print("  sidecar state['spring']['position']       =",
      float(np.asarray(sc.state["spring"]["position"])),
      " dtype", np.asarray(sc.state["spring"]["position"]).dtype)
print("  sidecar state['spring']['velocity']       =",
      float(np.asarray(sc.state["spring"]["velocity"])))
print()
print("  read back through get:",
      br.handle({"op": "get", "vr": [vr["spring.params.mass"], vr["spring.position"]]}))
print("  one step ->", br.handle({"op": "step", "t": 0.0, "dt": DT}))
print("  state after the step:",
      {k: float(np.asarray(v)) for k, v in sc.state["spring"].items()})
print()
print("  the model description advertises mass min/max =",
      next(v.min for v in md.variables if v.name == "spring.params.mass"),
      next(v.max for v in md.variables if v.name == "spring.params.mass"))
