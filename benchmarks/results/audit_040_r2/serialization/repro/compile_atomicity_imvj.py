"""Two more compile() failure points: inside the _meta build, and a coupling group."""
import os, copy, dataclasses, warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp, numpy as np
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

def snap(gm):
    t = lambda x: jax.tree_util.tree_map(
        lambda v: (tuple(jnp.shape(v)), str(jnp.asarray(v).dtype),
                   np.asarray(v).ravel()[:4].tolist()), x)
    return dict(sched=list(gm._schedule), mr=gm._is_multirate,
                rd=dict(gm._rate_dividers), crd=dict(gm._committed_rate_dividers),
                params=t(gm.params), state=t(gm._state),
                meta=sorted(gm._state.get("_meta", {})), step=id(gm._compiled_step),
                gen=gm._compile_generation, dirty=gm._dirty,
                hashes=dict(gm._static_data_hashes), traces=gm._n_traces,
                scans=len(gm._scan_cache))

def build(**grp):
    g = GraphManager()
    g.add_node(SpringDamperNode(name="a", timestep=0.01, stiffness=30.0, damping=2.0))
    g.add_node(SpringDamperNode(name="b", timestep=0.01, stiffness=10.0, damping=1.0))
    g.add_edge("a", "b", "position", "anchor_position")
    g.add_edge("b", "a", "position", "anchor_position")
    kw = dict(max_iterations=5, tolerance=1e-8, diagnostics=True,
              acceleration="iqn-imvj", predictor="quadratic")
    kw.update(grp)
    g.add_coupling_group(["a", "b"], **kw)
    g.compile()
    for _ in range(4): g.step()
    return g

def attempt(label, gm, mutate):
    mutate(gm); before = snap(gm)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); gm.compile()
    except BaseException as e:
        d = [(k, before[k], snap(gm)[k]) for k in before if before[k] != snap(gm)[k]]
        print(f"\n### {label}\n  raised {type(e).__name__}: {str(e)[:130]}")
        print("  graph unchanged (atomic)" if not d else f"  !! CHANGED: {[k for k,_,_ in d]}")
        for k, x, y in d:
            print(f"     {k}: {repr(x)[:200]}  ->  {repr(y)[:200]}")
        return
    print(f"\n### {label}\n  compile() did NOT raise")

gm = build()
def m(g):
    g._coupling_groups[0] = dataclasses.replace(
        g._coupling_groups[0], accelerated_fields={"a": ["not_a_field"]})
attempt("iqn-imvj: bogus accelerated field reaches flatten_coupled_state in the _meta build", gm, m)

gm = build()
def m2(g):
    # a node the group names is removed -> the group is stale
    g.remove_node("b")
attempt("remove a node that a coupling group still names", gm, m2)

gm = build()
def m3(g):
    g._nodes["a"].node.static_data_hash = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
attempt("static_data_hash raises (imvj graph)", gm, m3)
