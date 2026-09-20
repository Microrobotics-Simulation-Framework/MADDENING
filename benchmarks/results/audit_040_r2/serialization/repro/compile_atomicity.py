"""compile() atomicity: a failed compile must leave the graph exactly as it was.

Snapshot is taken AFTER the pre-compile mutation and BEFORE compile(), so only
what compile() itself changes shows up.
"""
import os, sys, copy, dataclasses, traceback, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.core.node import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode

def snap(gm):
    def tree(t):
        return jax.tree_util.tree_map(
            lambda x: (tuple(jnp.shape(x)), str(jnp.asarray(x).dtype),
                       np.asarray(x).ravel()[:6].tolist()), t)
    return {
        "_schedule": list(gm._schedule),
        "_is_multirate": gm._is_multirate,
        "_rate_dividers": dict(gm._rate_dividers),
        "_committed_rate_dividers": dict(gm._committed_rate_dividers),
        "params": tree(gm.params),
        "_params_dtypes": copy.deepcopy(gm._params_dtypes),
        "_params_shapes": copy.deepcopy(gm._params_shapes),
        "state": tree(gm._state),
        "state_keys": sorted(gm._state.keys()),
        "meta_keys": sorted(gm._state.get("_meta", {}).keys()),
        "compiled_step_id": id(gm._compiled_step),
        "_static_data_hashes": dict(gm._static_data_hashes),
        "_dirty": gm._dirty,
        "_compile_generation": gm._compile_generation,
        "_n_traces": gm._n_traces,
        "scan_cache_len": len(gm._scan_cache),
        "default_ext": sorted(map(str, gm._default_ext_leaves.keys())),
    }

def diff(a, b):
    return [(k, a[k], b[k]) for k in a if a[k] != b[k]]

def build(n_extra_steps=4, **grp):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s1", timestep=0.01, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_node(SpringDamperNode(name="s2", timestep=0.01, stiffness=10.0, damping=1.0,
                                 initial_position=0.2))
    gm.add_edge("s1", "s2", "position", "anchor_position")
    gm.add_edge("s2", "s1", "position", "anchor_position")
    kw = dict(max_iterations=5, tolerance=1e-8, diagnostics=True, predictor="linear")
    kw.update(grp)
    gm.add_coupling_group(["s1", "s2"], **kw)
    gm.compile()
    for _ in range(n_extra_steps):
        gm.step()
    return gm

def attempt(label, gm, mutate):
    mutate(gm)
    before = snap(gm)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
    except BaseException as exc:
        after = snap(gm)
        d = diff(before, after)
        print(f"\n### {label}")
        print(f"  raised {type(exc).__name__}: {str(exc)[:160]}")
        if d:
            print(f"  !! GRAPH CHANGED by the failed compile, {len(d)} field(s):")
            for k, x, y in d:
                print(f"     - {k}:")
                print(f"         before = {repr(x)[:400]}")
                print(f"         after  = {repr(y)[:400]}")
        else:
            print("  graph unchanged by the failed compile (atomic)")
        return
    print(f"\n### {label}\n  compile() did NOT raise")

# 1. accelerated_fields typo (raises after _meta is recomputed)
gm = build()
def m1(g):
    g._coupling_groups[0] = dataclasses.replace(
        g._coupling_groups[0], accelerated_fields={"s1": ["no_such_field"]})
attempt("accelerated_fields names a non-state field", gm, m1)

# 2. static_data_hash raises (last statement before the commit point)
gm = build()
def m2(g):
    def boom(): raise RuntimeError("static_data_hash exploded")
    g._nodes["s2"].node.static_data_hash = boom
attempt("static_data_hash() raises just before the commit point", gm, m2)

# 3. update() raises during _build_step_fn
gm = build()
def m3(g):
    def boom(*a, **k): raise RuntimeError("update exploded at trace time")
    g._nodes["s2"] = dataclasses.replace(g._nodes["s2"], update_fn=boom)
attempt("update() raises during _build_step_fn", gm, m3)

# 4. new node (different timestep -> new dividers), then compile fails
gm = build()
def m4(g):
    g.add_node(SpringDamperNode(name="s3", timestep=0.03, stiffness=1.0, damping=1.0))
    g._coupling_groups[0] = dataclasses.replace(
        g._coupling_groups[0], accelerated_fields={"s1": ["nope"]})
attempt("add_node changes rate dividers, then compile fails", gm, m4)

# 5. add a coupling group whose member set changes the meta key set, then fail
gm = build()
def m5(g):
    g.add_node(SpringDamperNode(name="s3", timestep=0.01, stiffness=1.0, damping=1.0))
    g.add_node(SpringDamperNode(name="s4", timestep=0.01, stiffness=1.0, damping=1.0))
    g.add_edge("s3", "s4", "position", "anchor_position")
    g.add_edge("s4", "s3", "position", "anchor_position")
    g.add_coupling_group(["s3", "s4"], max_iterations=4, diagnostics=True,
                         predictor="linear", accelerated_fields={"s3": ["bogus"]})
attempt("second coupling group with a bogus accelerated field", gm, m5)

# 6. failure inside a multi-rate build
gm = build()
def m6(g):
    g.add_node(HeatNode(name="h", timestep=0.04, n_cells=8))
    g.add_edge("h", "s1", "temperature", "anchor_position")   # shape mismatch -> ExceptionGroup
attempt("multi-rate rebuild that fails validation", gm, m6)

# 7. reset_state() then a failing compile
gm = build()
def m7(g):
    g.reset_state()
    g._coupling_groups[0] = dataclasses.replace(
        g._coupling_groups[0], accelerated_fields={"s1": ["nope"]})
attempt("reset_state() then a failing compile", gm, m7)
