"""One ``verify_node`` check on a sharded wrapper, on every push.

``test_sharded_boundary_inputs.py::test_verify_node_battery_passes_on_the_sharded_wrapper``
runs the whole battery -- eight checks, 20 examples each -- on a
:class:`ShardedStencilNode` over four devices, and is slow-marked for it
(8-11 s on CI).  So without this module no push would put a sharded
wrapper through ``verify_node`` at all.

``verify_node`` samples every declared boundary input at its declared
shape.  For the wrapped LBM node two of them are per-cell grids
(``body_force``, ``wall_mask_update``), which the wrapper must shard and
halo-pad exactly like state, while the two pressures stay replicated; a
wrapper that mishandled a sampled input fails whichever check calls
``update`` first.  This runs one check, ``jit_consistent``: every example
goes through the wrapper both eagerly and under ``jax.jit`` and the two
must agree, so it also sees a wrapper whose traced path differs from its
eager one.
"""

from __future__ import annotations

import jax
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.testing.verification import verify_node
from tests.cloud.multigpu.test_sharded_boundary_inputs import _lbm

N_DEV = 4

pytestmark = pytest.mark.skipif(len(jax.devices()) < N_DEV,
                                reason=f"needs {N_DEV} CPU-virtual devices")


def test_a_verify_node_check_passes_on_the_sharded_wrapper():
    sharded = ShardedStencilNode(_lbm(), create_device_mesh(shape=(N_DEV,)),
                                 axis_map={"devices": 1}, boundary="periodic")
    # What makes the check reach the wrapper's input sharding: two of the
    # inputs it samples are per-cell grids, not scalars.
    spec = sharded.boundary_input_spec()
    assert tuple(spec["body_force"].shape) == (8, 8, 2), spec["body_force"]
    assert tuple(spec["wall_mask_update"].shape) == (8, 8), spec["wall_mask_update"]
    pressure = (0.3, 0.4)
    # ``verify_node`` carries its own Hypothesis settings, so the profile in
    # tests/conftest.py does not reach it; 20 is the house floor, and one
    # example runs the wrapper through a four-device ``shard_map`` twice.
    results = verify_node(
        sharded, bounds={"f": (0.02, 0.2), "wall_mask": (0.0, 1.0)},
        checks=["jit_consistent"], max_examples=20, dt_range=(1.0, 1.0), derandomize=True,
        boundary_bounds={"inlet_pressure": pressure, "outlet_pressure": pressure,
                         "body_force": (-1e-3, 1e-3), "wall_mask_update": (0.0, 1.0)},
    )
    assert set(results) == {"jit_consistent"}, sorted(results)
    result = results["jit_consistent"]
    # PASS, not SKIP: the examples ran.
    assert result.status == "PASS", (result.status, result.detail, result.counterexample)
    assert result.n_examples >= 20, result.n_examples
