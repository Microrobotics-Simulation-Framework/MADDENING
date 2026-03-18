"""Tests for sharded Jacobi coupling step."""

import os

# Try to force 2 CPU devices — only effective if set before JAX import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.partition import assign_nodes_to_devices
from maddening.cloud.multigpu.sharded_step import (
    build_sharded_jacobi_pass,
    shard_state,
)

_HAS_2_DEVICES = len(jax.devices()) >= 2
_SKIP_MSG = "Requires >=2 JAX devices (set XLA_FLAGS before JAX import)"


# -- Minimal node stubs -----------------------------------------------

def _update_a(state, boundary, dt):
    """Node A: position += velocity * dt, influenced by boundary force."""
    force = boundary.get("force", jnp.array(0.0))
    new_vel = state["velocity"] + force * dt
    new_pos = state["position"] + new_vel * dt
    return {"position": new_pos, "velocity": new_vel}


def _update_b(state, boundary, dt):
    """Node B: temperature diffuses, influenced by boundary heat_flux."""
    flux = boundary.get("heat_flux", jnp.array(0.0))
    new_temp = state["temperature"] + flux * dt
    return {"temperature": new_temp}


def _resolve_boundary(node_name, full_state):
    """Simple boundary resolution: A gets force from B, B gets flux from A."""
    if node_name == "a":
        return {"force": full_state["b"]["temperature"] * 0.1}
    elif node_name == "b":
        return {"heat_flux": full_state["a"]["velocity"] * 0.5}
    return {}


# -- Tests -------------------------------------------------------------

@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestBuildShardedJacobiPass:
    def test_basic_pass(self):
        mesh = create_device_mesh(n_devices=2)
        device_map = {"a": 0, "b": 1}

        jacobi_pass = build_sharded_jacobi_pass(
            group_node_names=["a", "b"],
            update_fns={"a": _update_a, "b": _update_b},
            resolve_boundary_fn=_resolve_boundary,
            device_map=device_map,
            mesh=mesh,
            get_dt_fn=lambda nn: jnp.array(0.01),
        )

        initial = {
            "a": {"position": jnp.array(0.0), "velocity": jnp.array(1.0)},
            "b": {"temperature": jnp.array(300.0)},
        }
        latest = {
            "a": {"position": jnp.array(0.0), "velocity": jnp.array(1.0)},
            "b": {"temperature": jnp.array(300.0)},
        }

        result = jacobi_pass(initial, latest)
        assert "a" in result and "b" in result
        assert result["a"]["position"].shape == ()
        assert result["b"]["temperature"].shape == ()

    def test_matches_single_device(self):
        """Sharded output matches single-device Jacobi within tolerance."""
        mesh = create_device_mesh(n_devices=2)
        device_map = {"a": 0, "b": 1}
        dt_fn = lambda nn: jnp.array(0.01)

        # Sharded pass
        sharded_pass = build_sharded_jacobi_pass(
            group_node_names=["a", "b"],
            update_fns={"a": _update_a, "b": _update_b},
            resolve_boundary_fn=_resolve_boundary,
            device_map=device_map,
            mesh=mesh,
            get_dt_fn=dt_fn,
        )

        # Single-device pass (all on device 0)
        single_pass = build_sharded_jacobi_pass(
            group_node_names=["a", "b"],
            update_fns={"a": _update_a, "b": _update_b},
            resolve_boundary_fn=_resolve_boundary,
            device_map={"a": 0, "b": 0},
            mesh=mesh,
            get_dt_fn=dt_fn,
        )

        initial = {
            "a": {"position": jnp.array(1.0), "velocity": jnp.array(2.0)},
            "b": {"temperature": jnp.array(350.0)},
        }
        latest = dict(initial)

        sharded_result = sharded_pass(initial, latest)
        single_result = single_pass(initial, latest)

        # Results should match exactly (Jacobi reads from frozen state)
        for node in ["a", "b"]:
            for field in sharded_result[node]:
                assert jnp.allclose(
                    sharded_result[node][field],
                    single_result[node][field],
                    atol=1e-6,
                ), f"Mismatch at {node}.{field}"


@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestShardState:
    def test_shard_state(self):
        mesh = create_device_mesh(n_devices=2)
        device_map = {"a": 0, "b": 1}

        state = {
            "a": {"position": jnp.array(1.0)},
            "b": {"temperature": jnp.array(300.0)},
        }

        sharded = shard_state(state, device_map, mesh)
        assert "a" in sharded and "b" in sharded
        # Values should be preserved
        assert float(sharded["a"]["position"]) == 1.0
        assert float(sharded["b"]["temperature"]) == 300.0
