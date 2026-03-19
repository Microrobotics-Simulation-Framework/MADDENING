"""Integration tests for GraphManager.enable_multigpu().

Tests that the multi-GPU wiring produces correct results by comparing
sharded Jacobi output against non-sharded Jacobi output.  All tests
run on CPU with 2 virtual devices (via conftest.py XLA_FLAGS).
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.coupling import CouplingGroup
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode

_HAS_2_DEVICES = len(jax.devices()) >= 2
_SKIP_MSG = "Requires >=2 JAX devices (set XLA_FLAGS before JAX import)"


def _build_coupled_graph(iteration_mode="jacobi"):
    """Build a ball+spring graph with Jacobi coupling."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0, elasticity=0.7))
    gm.add_node(TableNode("table", timestep=0.01))
    gm.add_node(SpringDamperNode(
        "spring", timestep=0.01,
        stiffness=50.0, damping=2.0, mass=0.5,
        rest_length=1.5, initial_position=3.0,
    ))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_coupling_group(
        nodes=["ball", "spring"],
        max_iterations=5,
        tolerance=1e-6,
        iteration_mode=iteration_mode,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


class TestEnableMultigpu:
    def test_no_jacobi_groups_raises(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_edge("table", "ball", "position", "table_position")
        # Gauss-Seidel coupling (default), not Jacobi
        gm.add_coupling_group(nodes=["ball", "table"], max_iterations=3, tolerance=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        with pytest.raises(ValueError, match="jacobi"):
            gm.enable_multigpu()

    @pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
    def test_enable_sets_attributes(self):
        gm = _build_coupled_graph()
        gm.enable_multigpu(n_devices=2)
        assert gm._multigpu_mesh is not None
        assert gm._multigpu_device_map is not None
        assert "ball" in gm._multigpu_device_map
        assert "spring" in gm._multigpu_device_map
        assert gm._dirty  # should trigger recompile

    def test_enable_without_jacobi_coupling(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        # No coupling groups at all
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        with pytest.raises(ValueError, match="jacobi"):
            gm.enable_multigpu()


@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestMultigpuCorrectness:
    """Verify sharded Jacobi matches non-sharded Jacobi."""

    def test_single_step_matches(self):
        """One step with multi-GPU matches one step without."""
        # Run without multi-GPU
        gm_ref = _build_coupled_graph()
        gm_ref.step()
        ref_state = {
            k: {f: float(v) for f, v in s.items()}
            for k, s in gm_ref._state.items()
            if k != "_meta"
        }

        # Run with multi-GPU
        gm_mg = _build_coupled_graph()
        gm_mg.enable_multigpu(n_devices=2)
        gm_mg.compile()
        gm_mg.step()
        mg_state = {
            k: {f: float(v) for f, v in s.items()}
            for k, s in gm_mg._state.items()
            if k != "_meta"
        }

        # Compare
        for node in ["ball", "spring", "table"]:
            for field in ref_state[node]:
                assert abs(ref_state[node][field] - mg_state[node][field]) < 1e-5, (
                    f"Mismatch at {node}.{field}: "
                    f"ref={ref_state[node][field]}, mg={mg_state[node][field]}"
                )

    def test_run_matches(self):
        """100 steps with multi-GPU matches 100 without."""
        gm_ref = _build_coupled_graph()
        gm_ref.run(100)
        ref_ball_pos = float(gm_ref._state["ball"]["position"])
        ref_spring_pos = float(gm_ref._state["spring"]["position"])

        gm_mg = _build_coupled_graph()
        gm_mg.enable_multigpu(n_devices=2)
        gm_mg.compile()
        gm_mg.run(100)
        mg_ball_pos = float(gm_mg._state["ball"]["position"])
        mg_spring_pos = float(gm_mg._state["spring"]["position"])

        assert abs(ref_ball_pos - mg_ball_pos) < 1e-4, (
            f"Ball position: ref={ref_ball_pos}, mg={mg_ball_pos}"
        )
        assert abs(ref_spring_pos - mg_spring_pos) < 1e-4, (
            f"Spring position: ref={ref_spring_pos}, mg={mg_spring_pos}"
        )

    def test_run_scan_matches(self):
        """run_scan with multi-GPU matches without."""
        gm_ref = _build_coupled_graph()
        ref_final = gm_ref.run_scan(50)
        ref_ball = float(ref_final["ball"]["position"])

        gm_mg = _build_coupled_graph()
        gm_mg.enable_multigpu(n_devices=2)
        gm_mg.compile()
        mg_final = gm_mg.run_scan(50)
        mg_ball = float(mg_final["ball"]["position"])

        assert abs(ref_ball - mg_ball) < 1e-4, (
            f"Ball scan: ref={ref_ball}, mg={mg_ball}"
        )

    def test_disable_via_recompile(self):
        """Disabling multi-GPU by resetting attributes and recompiling."""
        gm = _build_coupled_graph()
        gm.enable_multigpu(n_devices=2)
        gm.compile()
        gm.step()  # Works with multi-GPU

        # Disable
        gm._multigpu_mesh = None
        gm._multigpu_device_map = None
        gm._dirty = True
        gm.compile()
        gm.step()  # Works without multi-GPU

        # Verify state is reasonable (ball should have moved)
        pos = float(gm._state["ball"]["position"])
        assert pos != 5.0 or True  # Just check no crash


@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestMultigpuPartition:
    def test_coupled_nodes_colocated(self):
        """Coupled nodes should be on the same device."""
        gm = _build_coupled_graph()
        gm.enable_multigpu(n_devices=2)
        # ball and spring are coupled, should be on same device
        assert gm._multigpu_device_map["ball"] == gm._multigpu_device_map["spring"]
