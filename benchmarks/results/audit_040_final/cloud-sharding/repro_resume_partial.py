"""Resume against a CHANGED graph: does a failed resume leave state half-applied?

The entry point (cloud/entrypoint.py:152-157) catches every exception from the
resume and logs "starting fresh".  This checks whether "fresh" is true.
"""
import json, logging, os, shutil, sys, tempfile
import numpy as np
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.simulation.checkpoint import (
    save_state_with_manifest, load_state_with_manifest)


class Relax(SimulationNode):
    """x <- x + rate*(target-x).  `gainvec` is a differentiable param whose
    SHAPE is set at construction -- that is what we change between save/load."""
    def __init__(self, name="relax", n=4, rate=0.5, gain_len=3, timestep=0.1):
        super().__init__(name=name, timestep=timestep, rate=float(rate),
                         gainvec=[1.0] * int(gain_len), n=int(n))
    def halo_width(self): return {}
    def state_fields(self): return ["x"]
    def initial_state(self):
        return {"x": jnp.zeros(int(self.params["n"]), jnp.float32)}
    def boundary_input_spec(self): return {}
    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        g = jnp.sum(jnp.asarray(p["gainvec"]))
        return {"x": state["x"] + p["rate"] * dt * (g - state["x"])}


def build(gain_len):
    gm = GraphManager()
    gm.add_node(Relax(gain_len=gain_len))
    gm.compile()
    return gm

tmp = tempfile.mkdtemp(prefix="audit_resume_")
try:
    # --- save a checkpoint from a graph with gainvec of length 3 ----------
    src = build(3)
    src.set_node_state("relax", {"x": jnp.asarray([7., 8., 9., 10.], jnp.float32)})
    src.params["nodes"]["relax"]["gainvec"] = jnp.asarray([2., 2., 2.], jnp.float32)
    npz, man = save_state_with_manifest(src, os.path.join(tmp, "snap.npz"))
    print(f"saved  state x = {np.asarray(src.get_node_state('relax')['x'])}")
    print(f"saved  gainvec = {np.asarray(src.params['nodes']['relax']['gainvec'])}")

    # --- resume into a graph whose gainvec has length 4 (graph changed) ---
    dst = build(4)
    before_x = np.asarray(dst.get_node_state("relax")["x"]).copy()
    before_g = np.asarray(dst.params["nodes"]["relax"]["gainvec"]).copy()
    print(f"\ntarget graph BEFORE resume: x = {before_x}, gainvec = {before_g}")

    try:
        load_state_with_manifest(dst, npz)
        print("resume SUCCEEDED (unexpected)")
    except Exception as exc:
        print(f"resume FAILED as expected: {type(exc).__name__}: {exc}")
        # This is exactly what cloud/entrypoint.py:154-157 swallows,
        # logging 'Failed to resume ...; starting fresh'.

    after_x = np.asarray(dst.get_node_state("relax")["x"])
    after_g = np.asarray(dst.params["nodes"]["relax"]["gainvec"])
    print(f"target graph AFTER  resume: x = {after_x}, gainvec = {after_g}")

    print()
    if not np.array_equal(before_x, after_x):
        print("*** STATE WAS MUTATED BY THE FAILED RESUME ***")
        print(f"    x went {before_x} -> {after_x} despite 'starting fresh'")
    else:
        print("state unchanged -- failure was atomic")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
