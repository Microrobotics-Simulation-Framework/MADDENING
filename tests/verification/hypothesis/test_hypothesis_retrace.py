"""The compiled step traces once, for any graph built from built-in nodes.

Guards the weak-type normalisation in ``GraphManager.compile`` /
``set_node_state``: a node whose ``initial_state`` uses ``jnp.array(0.0)``
(no dtype) yields weak-typed leaves that come back strongly typed after
one step and retrace the jitted step.  Over random graphs (node mix,
edges, coupling on/off, multi-rate) the cache must hold exactly one
entry after several steps, and ``set_node_state`` with weak-typed values
must not add one either.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


class WeakScalar(SimulationNode):
    """Deliberately weak-typed seeds (what many downstream nodes do)."""
    def initial_state(self):
        return {"x": jnp.array(float(self.params.get("x0", 1.0))),
                "acc": jnp.array(0.0), "n": jnp.array(0)}

    def update(self, s, bi, dt, *, params=None):
        drive = bi.get("drive", jnp.array(0.0))
        x = s["x"] + dt * (drive - s["x"])
        acc = jnp.where(s["n"] > 1, s["acc"] + x, s["acc"])   # changes from step 3
        return {"x": x, "acc": acc, "n": s["n"] + 1}

    def boundary_input_spec(self):
        from maddening.core.node import BoundaryInputSpec
        return {"drive": BoundaryInputSpec(shape=(), description="drive")}


def _graph(kinds, couple, multirate, seed):
    rng = np.random.default_rng(seed)
    gm = GraphManager()
    names = []
    for i, kind in enumerate(kinds):
        dt = 0.01 * (2 if (multirate and i % 2) else 1)
        name = f"n{i}"
        if kind == "spring":
            gm.add_node(SpringDamperNode(name, dt, stiffness=float(rng.uniform(5, 50)),
                                         damping=1.0, initial_position=float(rng.uniform(-1, 1))))
        elif kind == "ball":
            gm.add_node(BallNode(name, dt, initial_position=float(rng.uniform(1, 5))))
        elif kind == "heat":
            gm.add_node(HeatNode(name, dt * 1e-3, n_cells=6, thermal_diffusivity=0.1))
        else:
            gm.add_node(WeakScalar(name, dt, x0=float(rng.uniform(-2, 2))))
        names.append(name)
    # A chain of scalar edges between consecutive nodes where the types allow.
    scalar_out = {"spring": "position", "ball": "position", "weak": "x", "heat": None}
    scalar_in = {"spring": "anchor_position", "ball": "table_position", "weak": "drive",
                 "heat": None}
    edges = []
    for (a, ka), (b, kb) in zip(zip(names, kinds), zip(names[1:], kinds[1:])):
        if scalar_out[ka] and scalar_in[kb]:
            gm.add_edge(a, b, scalar_out[ka], scalar_in[kb])
            edges.append((a, b))
    if couple and edges and not multirate:
        a, b = edges[0]
        # make it a genuine cycle
        ka, kb = kinds[names.index(a)], kinds[names.index(b)]
        if scalar_out[kb] and scalar_in[ka]:
            gm.add_edge(b, a, scalar_out[kb], scalar_in[ka])
            gm.add_coupling_group([a, b], max_iterations=10, tolerance=1e-6)
    gm.compile()
    return gm


kinds_st = st.lists(st.sampled_from(["spring", "ball", "heat", "weak"]), min_size=1, max_size=4)


@given(kinds=kinds_st, couple=st.booleans(), multirate=st.booleans(),
       seed=st.integers(0, 2**31))
@settings(max_examples=40, deadline=None)
def test_step_traces_once_over_random_graphs(kinds, couple, multirate, seed):
    gm = _graph(kinds, couple, multirate, seed)
    assert all(not getattr(l, "weak_type", False) for l in jax.tree.leaves(gm._state))
    for _ in range(4):
        gm.step()
    assert gm._compiled_step._cache_size() == 1
    # Weak-typed values handed in later are normalised too.
    for name in gm.node_names:
        s = gm.get_node_state(name)
        gm.set_node_state(name, jax.tree.map(
            lambda x: jnp.array(np.asarray(x).tolist()) if x.ndim == 0 and
            jnp.issubdtype(x.dtype, jnp.floating) else x, s))
    for _ in range(3):
        gm.step()
    assert gm._compiled_step._cache_size() == 1
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(gm._state)
               if jnp.issubdtype(x.dtype, jnp.floating))


@given(kinds=kinds_st, seed=st.integers(0, 2**31))
@settings(max_examples=15, deadline=None)
def test_run_scan_and_step_agree_after_normalisation(kinds, seed):
    """Normalising the seed state changes the trace signature only."""
    a = _graph(kinds, False, False, seed)
    b = _graph(kinds, False, False, seed)
    for _ in range(5):
        a.step()
    final = b.run_scan(5)
    for name in a.node_names:
        for f, v in a.get_node_state(name).items():
            np.testing.assert_allclose(np.asarray(v), np.asarray(final[name][f]),
                                       rtol=1e-5, atol=1e-6)
