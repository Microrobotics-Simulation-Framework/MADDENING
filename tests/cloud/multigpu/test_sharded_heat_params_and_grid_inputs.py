"""``ShardedStencilNode`` around a ``HeatNode``: the grid-shaped
boundary-input heuristic does not misclassify a profile along an
unsharded axis, the wrapper exposes the inner node's params, an injected
diffusivity drives the sharded step (and is differentiable through it),
and a halo-padded per-cell heat source matches the unsharded node.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


# ---------------------------------------------------------------------------
# #8: (n,) profile on an n x n grid is NOT grid-shaped
# ---------------------------------------------------------------------------

class Plate(SimulationNode):
    """2-D field on an ``(n, n)`` grid, stencil along axis 0 only, with a
    per-column profile input of shape ``(n,)`` along the unsharded axis."""

    def __init__(self, name, timestep, n=8):
        super().__init__(name, timestep, n=n)

    def halo_width(self):
        return {0: 1}

    def initial_state(self):
        n = self.params["n"]
        return {"T": jnp.asarray(np.arange(n * n, dtype=np.float32).reshape(n, n))}

    def boundary_input_spec(self):
        n = self.params["n"]
        return {"profile": BoundaryInputSpec(shape=(n,), description="per-column source")}

    def _step(self, T, profile, dt):
        lap = jnp.roll(T, 1, 0) + jnp.roll(T, -1, 0) - 2 * T
        return T + dt * (0.1 * lap + profile[None, :])

    def update(self, state, bi, dt):
        n = self.params["n"]
        return {"T": self._step(state["T"], bi.get("profile", jnp.zeros(n, jnp.float32)), dt)}

    def update_padded(self, state_padded, bi, dt, *, static_padded=None, shard_info=None):
        n = self.params["n"]
        T_pad = state_padded["T"]
        T_new = self._step(T_pad, bi.get("profile", jnp.zeros(n, jnp.float32)), dt)
        return {"T": jnp.concatenate([T_pad[:1], T_new[1:-1], T_pad[-1:]], axis=0)}


class Rect(Plate):
    def initial_state(self):
        return {"T": jnp.zeros((8, 5), jnp.float32)}

    def boundary_input_spec(self):
        return {"profile": BoundaryInputSpec(shape=(5,), description="")}

    def update(self, state, bi, dt):
        return {"T": self._step(state["T"], bi.get("profile", jnp.zeros(5)), dt)}

    def update_padded(self, sp, bi, dt, *, static_padded=None, shard_info=None):
        T_pad = sp["T"]
        T_new = self._step(T_pad, bi.get("profile", jnp.zeros(5)), dt)
        return {"T": jnp.concatenate([T_pad[:1], T_new[1:-1], T_pad[-1:]], axis=0)}


@pytest.mark.parametrize("cls, n_prof", [(Plate, 8), (Rect, 5)])
def test_profile_input_along_unsharded_axis_is_replicated(cls, n_prof):
    mesh = create_device_mesh(shape=(4,))
    un = cls("p", 1.0)
    sh = ShardedStencilNode(cls("p", 1.0), mesh, axis_map={"devices": 0}, boundary="periodic")
    st = un.initial_state()
    bi = {"profile": jnp.asarray(np.linspace(0, 1, n_prof), jnp.float32)}
    assert sh._grid_shaped_boundary_inputs(st, bi) == frozenset()
    # a genuine per-cell field on the same node is still grid-shaped
    full = {"src": jnp.zeros(st["T"].shape, jnp.float32),
            "src3": jnp.zeros(st["T"].shape + (2,), jnp.float32)}
    assert sh._grid_shaped_boundary_inputs(st, full) == {"src", "src3"}
    a = sh.update(st, bi, 1.0)["T"]
    b = un.update(st, bi, 1.0)["T"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)


# ---------------------------------------------------------------------------
# HeatNode on the sharded path accepts (and uses) injected params
# ---------------------------------------------------------------------------

def _heat(alpha=0.1):
    n = 16
    return HeatNode(name="heat", timestep=1e-4, n_cells=n, length=1.0,
                    thermal_diffusivity=alpha,
                    initial_temperature=np.sin(np.pi * np.linspace(0, 1, n)).astype(np.float32))


def _graph(alpha=0.1, sharded=True):
    gm = GraphManager()
    if sharded:
        mesh = create_device_mesh(shape=(4,))
        gm.add_node(ShardedStencilNode(_heat(alpha), mesh, axis_map={"devices": 0},
                                       boundary="edge"))
    else:
        gm.add_node(_heat(alpha))
    gm.compile()
    return gm


def test_sharded_heat_exposes_the_same_params_as_unsharded():
    sh, un = _graph(), _graph(sharded=False)
    assert sh._nodes["heat"].node.accepts_params()
    assert set(sh.params["nodes"]["heat"]) == set(un.params["nodes"]["heat"])
    assert "thermal_diffusivity" in sh.params["nodes"]["heat"]


def test_injected_diffusivity_drives_the_sharded_step():
    sh = _graph(0.1)
    p = jax.tree.map(lambda x: x, sh.params)
    p["nodes"]["heat"]["thermal_diffusivity"] = jnp.float32(0.3)
    injected = sh.run_scan(3, params=p)["heat"]["temperature"]
    # identical numerics to a sharded node *constructed* with alpha=0.3
    ctor = _graph(0.3).run_scan(3)["heat"]["temperature"]
    np.testing.assert_allclose(np.asarray(injected), np.asarray(ctor), rtol=1e-6, atol=1e-7)
    # and different from the constructor value 0.1
    base = _graph(0.1).run_scan(3)["heat"]["temperature"]
    assert not np.allclose(np.asarray(injected), np.asarray(base))


def test_gradient_wrt_diffusivity_through_sharded_heat_step():
    sh = _graph(0.1)
    step, ext = sh._build_step_fn(), sh._default_external_inputs()

    def loss(alpha):
        p = jax.tree.map(lambda x: x, sh.params)
        p["nodes"]["heat"]["thermal_diffusivity"] = alpha
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), sh._state, None, length=3)
        return jnp.sum(final["heat"]["temperature"] ** 2)

    g = jax.grad(loss)(jnp.float32(0.1))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_sharded_heat_takes_a_grid_shaped_source_like_the_unsharded_node():
    """audit round 4: ShardedStencilNode halo-pads a grid-shaped input and
    HeatNode.update_padded expected it at the unpadded local shape."""
    import numpy as np
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
    from maddening.nodes.heat import HeatNode

    def mk():
        return HeatNode("h", 1e-4, n_cells=16, thermal_diffusivity=0.1, initial_temperature=1.0)

    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(mk(), mesh, axis_map={"devices": 0}, boundary="edge")
    un = mk()
    rng = np.random.default_rng(0)
    # only the per-cell source: the global Dirichlet ends are a separate
    # (pre-existing) sharded-heat concern and are not what this checks
    bi = {"heat_source": jnp.asarray(rng.standard_normal(16), jnp.float32)}
    a = b = un.initial_state()
    for _ in range(3):
        a, b = sh.update(a, bi, 1e-4), un.update(b, bi, 1e-4)
    # interior cells: the two global end cells follow the wrapper's halo
    # policy ("edge") rather than the unsharded node's self-Dirichlet
    # fallback, a pre-existing sharded-heat semantic outside this check
    np.testing.assert_allclose(np.asarray(a["temperature"])[1:-1],
                               np.asarray(b["temperature"])[1:-1], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# The cell count is structural: it cannot be changed under a running rod
# ---------------------------------------------------------------------------


def _uniform_rod():
    """``_heat`` at a scalar initial temperature, so another ``n_cells`` can
    build a state at all -- the refusal under test is about its shape."""
    return HeatNode(name="heat", timestep=1e-4, n_cells=16, length=1.0,
                    thermal_diffusivity=0.1, initial_temperature=0.3)


def _rod_graph(sharded):
    node = _uniform_rod()
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(node, create_device_mesh(shape=(4,)),
                                   axis_map={"devices": 0}) if sharded else node)
    gm.add_external_input("heat", "left_temperature")
    gm.add_external_input("heat", "right_temperature")
    gm.compile()
    return gm, node


@pytest.mark.parametrize("n_cells", [17, 24])
def test_a_rest_write_of_the_cell_count_is_refused_on_a_sharded_rod(n_cells):
    """Sharded, ``PUT n_cells=17`` answered 200 and the next step returned
    a plausible field for a rod that does not exist: the 16 cells stepped
    with ``dx = L/17`` and the right end never closed (8.5e-2 from the
    correct step), where the unsharded node's step was a 500.  Refused on
    both before anything is written; the sharded rod still steps as the
    rod it was built as."""
    from fastapi.testclient import TestClient
    from maddening.api.server import SimulationServer

    gm, node = _rod_graph(sharded=True)
    client = TestClient(SimulationServer({"HeatNode": HeatNode}, gm).create_app(),
                        raise_server_exceptions=False)
    resp = client.put("/graph/params/heat", json={"params": {"n_cells": n_cells}})
    assert resp.status_code == 400, resp.text
    assert "changes the layout of the state" in resp.json()["detail"]
    assert node.params["n_cells"] == 16 and not gm._dirty
    assert client.post("/sim/step", json={}).status_code == 200
    ref = _uniform_rod()
    ends = {"left_temperature": jnp.float32(0.0), "right_temperature": jnp.float32(0.0)}
    want = ref.update(ref.initial_state(), ends, 1e-4)["temperature"]
    np.testing.assert_allclose(np.asarray(gm.get_node_state("heat")["temperature"]),
                               np.asarray(want), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("n_cells", [17, 24])
def test_a_cell_count_written_under_a_sharded_rod_is_refused_by_its_step(n_cells):
    """The same write made directly (``node.params`` and ``compile()``),
    which no route guards.  Unsharded, the next step raises: the state has
    16 cells and the node now broadcasts to ``n_cells``.  Sharded it used
    to step: ``update_padded`` never sees the global grid and took the
    extent from the params.  The wrapper now compares the state with what
    ``initial_state()`` builds (24 is a whole number of the 4-cell shard
    blocks, so only that comparison can see it), and refuses by name."""
    for sharded in (False, True):
        gm, node = _rod_graph(sharded)
        gm.step()
        node.params["n_cells"] = n_cells
        gm.compile()
        with pytest.raises(Exception) as excinfo:
            gm.step()
        if sharded:
            assert excinfo.type is ValueError
            assert "state field 'temperature' has shape (16,)" in str(excinfo.value)
            assert f"now builds ({n_cells},)" in str(excinfo.value)
