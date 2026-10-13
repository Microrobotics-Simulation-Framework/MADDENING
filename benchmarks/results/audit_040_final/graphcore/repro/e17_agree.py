"""step / run / run_scan / run_scan_with_history agreement on non-trivial graphs."""
import numpy as np, jax, jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode

def multirate():
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.0))
    gm.add_node(BallNode("ball", 0.03, initial_position=1.0))
    gm.add_edge("table","ball","position","table_position")
    gm.compile(); return gm

def coupled():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.001, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.001, initial_position=2.0))
    gm.add_edge("a","b","position","anchor_position")
    gm.add_edge("b","a","position","anchor_position")
    gm.add_coupling_group(["a","b"], tolerance=1e-10, max_iterations=25,
                          solver="ift", diagnostics=True)
    gm.compile(); return gm

def flat(gm):
    return {f"{n}/{f}": np.asarray(v) for n, d in gm._state.items() if n != "_meta"
            for f, v in d.items()}

for tag, mk, N in (("multirate", multirate, 9), ("coupled", coupled, 7)):
    a = mk(); [a.step() for _ in range(N)]
    b = mk(); b.run(N)
    c = mk(); c.run_scan(N)
    d = mk(); fin, hist = d.run_scan_with_history(N)
    fa, fb, fc, fd = flat(a), flat(b), flat(c), flat(d)
    def cmp(x, y): return all(np.array_equal(x[k], y[k]) for k in x)
    print(f"{tag}: step==run {cmp(fa,fb)}  step==run_scan {cmp(fa,fc)}  "
          f"step==scan_hist {cmp(fa,fd)}")
    # last history row must equal the final state
    ok = all(np.array_equal(np.asarray(hist[n][f])[-1], np.asarray(fin[n][f]))
             for n in fin for f in fin[n])
    print(f"   history[-1] == final: {ok}")
    # meta agreement
    for label, g in (("step", a), ("run", b), ("run_scan", c)):
        m = {k: np.asarray(v).tolist() for k, v in g._state.get("_meta", {}).items()}
        print(f"   {label} meta: {m}")
