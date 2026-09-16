import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp, pytest
from maddening.core.graph_manager import GraphManager
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.heat import HeatNode

N = 16
DT = 1e-3

def _heat(**kw):
    return HeatNode(name="rod", timestep=DT, n_cells=N, thermal_diffusivity=0.05,
                    initial_temperature=100.0, length=1.0, **kw)

def _sharded_graph(with_source):
    mesh = create_device_mesh(n_devices=4)
    node = ShardedStencilNode(_heat(), mesh, axis_map={mesh.axis_names[0]: 0}, boundary="edge")
    gm = GraphManager(); gm.add_node(node)
    if with_source:
        gm.add_external_input("rod", "heat_source", shape=(N,))
    gm.compile()
    return gm

def test_sharded_heat_accepts_params_contract():
    gm = _sharded_graph(False)
    assert "rod" in gm.params["nodes"], gm.params
    T0 = jnp.linspace(0.0, 100.0, N, dtype=jnp.float32)
    gm.set_node_state("rod", {"temperature": T0})
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["rod"]["thermal_diffusivity"] = jnp.asarray(0.5, jnp.float32)
    T_inj = np.asarray(gm.run_scan(5, params=p)["rod"]["temperature"])
    g2 = _sharded_graph(False); g2.set_node_state("rod", {"temperature": T0})
    T_def = np.asarray(g2.run_scan(5)["rod"]["temperature"])
    mesh = create_device_mesh(n_devices=4)
    node = ShardedStencilNode(HeatNode(name="rod", timestep=DT, n_cells=N, thermal_diffusivity=0.5,
                    initial_temperature=100.0, length=1.0), mesh, axis_map={mesh.axis_names[0]: 0}, boundary="edge")
    ref = GraphManager(); ref.add_node(node); ref.compile(); ref.set_node_state("rod", {"temperature": T0})
    T_ref = np.asarray(ref.run_scan(5)["rod"]["temperature"])
    assert not np.allclose(T_inj, T_def)
    np.testing.assert_allclose(T_inj, T_ref, rtol=1e-5)
    print("differ?", np.abs(T_inj - T_def).max())
    assert not np.allclose(T_inj, T_def)

def test_sharded_heat_grid_shaped_source_matches_unsharded():
    src = jnp.linspace(0.0, 10.0, N, dtype=jnp.float32)
    gm = _sharded_graph(True)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["rod"]["thermal_diffusivity"] = jnp.asarray(0.2, jnp.float32)
    out = gm.run_scan(3, external_inputs={"rod": {"heat_source": src}}, params=p)["rod"]["temperature"]
    # unsharded reference with the same Neumann semantics: no external T inputs, edge boundary
    ref_node = _heat()
    # emulate: unsharded HeatNode.update uses Dirichlet T_left=T[0] (adiabatic) when no input -> same as "edge"
    ref = GraphManager(); ref.add_node(ref_node); ref.add_external_input("rod", "heat_source", shape=(N,)); ref.compile()
    pr = jax.tree.map(lambda x: x, ref.params); pr["nodes"]["rod"]["thermal_diffusivity"] = jnp.asarray(0.2, jnp.float32)
    T_ref = ref.run_scan(3, external_inputs={"rod": {"heat_source": src}}, params=pr)["rod"]["temperature"]
    np.testing.assert_allclose(np.asarray(out), np.asarray(T_ref), rtol=1e-5)
