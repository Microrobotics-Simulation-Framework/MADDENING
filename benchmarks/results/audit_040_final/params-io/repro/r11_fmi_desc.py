"""FMI model description round trip for awkward parameter values."""
import numpy as np, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.nodes.spring import SpringDamperNode

gm = GraphManager()
gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
gm.compile()

print("=== 1. a non-finite live parameter ===")
gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(float("inf"), jnp.float32)
gm.params["nodes"]["s"]["damping"] = jnp.asarray(float("nan"), jnp.float32)
md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER)
for v in md.variables:
    if v.causality == "parameter":
        print(f"  {v.name:26s} start={v.start!r:24} min={v.min!r} max={v.max!r} dtype={v.dtype}")
xml = md.to_xml() if hasattr(md, "to_xml") else None
if xml is None:
    from maddening.fmi.model_description import model_description_xml as _x
    xml = _x(md)
import re
for line in xml.splitlines():
    if "stiffness" in line or "damping" in line:
        print("  XML:", line.strip()[:170])
try:
    import xml.etree.ElementTree as ET
    ET.fromstring(xml); print("  XML parses: yes")
except Exception as e:
    print("  XML parses: NO ->", e)

print()
print("=== 2. unicode in ParamSpec description / units ===")
gm2 = GraphManager()
gm2.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
gm2.compile()
gm2.set_param_spec("s", "stiffness", ParamSpec(bounds=(0.0, 1e9), transform="logit",
                                               description="стійкість ✓", units="N·m⁻¹"))
md2 = build_model_description(gm2, model_name="P", model_identifier=MODEL_IDENTIFIER)
v = next(v for v in md2.variables if v.name == "s.params.stiffness")
print(f"  description={v.description!r} unit={v.unit!r} min={v.min} max={v.max}")

print()
print("=== 3. zero-size / array parameter ===")
from maddening.nodes.heat import HeatNode
gm3 = GraphManager(); gm3.add_node(HeatNode("h", 0.01, grid_points=4)); gm3.compile()
md3 = build_model_description(gm3, model_name="P", model_identifier=MODEL_IDENTIFIER)
for v in md3.variables:
    if v.causality == "parameter":
        print(f"  {v.name:26s} shape={v.shape} start={v.start!r:40.40} dtype={v.dtype}")

print()
print("=== 4. instantiation token stability under a param VALUE change ===")
gm4 = GraphManager(); gm4.add_node(SpringDamperNode("s", 0.01, stiffness=30.0)); gm4.compile()
t1 = build_model_description(gm4, model_name="P", model_identifier=MODEL_IDENTIFIER).instantiation_token
gm4.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
t2 = build_model_description(gm4, model_name="P", model_identifier=MODEL_IDENTIFIER).instantiation_token
print("  token unchanged by a calibration:", t1 == t2)
