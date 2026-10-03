"""The coupling inventory's report, bound and gradient claims with a sharded member.

``tests/core/test_coupling_claims_in_every_domain.py`` states each claim
once, as a check keyed by its row id, and runs it in every numeric domain
but this one, which needs the virtual CPU devices this directory's
conftest provides.  Here the same checks run on the same group with one
member partitioned across four devices:

* ``a``: a scalar ``x_a <- g_a * mean(x_b) + c_a``, replicated;
* ``b``: a field of eight entries ``x_b <- g_b * x_a + c_b`` sharded by
  :class:`~maddening.cloud.multigpu.sharded_node.ShardedPointwiseNode`, its
  forcing varying along the field.

The edge ``b -> a`` takes the field's mean, a cross-device reduction, and
the group's norm and floor read the sharded field entry by entry -- the
two places a sharded member can change what the report says.  The fixed
point is the battery's algebra on ``(x_a, mean x_b)``, and ``x_b* = g_b x_a*
+ c_b`` entry by entry, so every check is the battery's, unchanged.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import test_coupling_claims_in_every_domain as battery

N_DEV = 4
N_CELLS = 8

pytestmark = pytest.mark.skipif(len(jax.devices()) < N_DEV,
                                reason=f"needs {N_DEV} CPU-virtual devices")


class _ForcedField(SimulationNode):
    """``x <- g * u + c`` entry by entry, ``c = b0 + b1 * clock``: pointwise in
    the field, so it shards along its one axis."""

    def __init__(self, name, timestep, *, g, b0, b1, n=N_CELLS):
        super().__init__(name, timestep, g=jnp.float32(g),
                         b0=jnp.full((n,), b0, jnp.float32), b1=jnp.full((n,), b1, jnp.float32))
        self._n = n

    def initial_state(self):
        z = jnp.zeros((self._n,), jnp.float32)
        return {"x": z, "c": z, "clock": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        clock = state["clock"] + jnp.asarray(dt, jnp.float32)
        c = (p["b0"] + p["b1"] * clock).astype(jnp.float32)
        u = jnp.asarray(boundary_inputs.get("u", jnp.float32(0.0)), jnp.float32)
        return {"x": (p["g"] * u + c).astype(jnp.float32), "c": c, "clock": clock}

    def update_evaluations(self):
        return 1


CFG = battery.Config("sharded", jnp.float32, jnp.float32)


def build(cfg: battery.Config, kind: str, diagnostics: bool) -> GraphManager:
    s = battery.SCENARIOS["moving"]
    mesh = create_device_mesh(shape=(N_DEV,))
    gm = GraphManager()
    gm.add_node(battery._Forced("a", battery.DT, jnp.float32, g=s["g_a"], b0=s["b0_a"],
                                b1=s["b1_a"]))
    gm.add_node(ShardedPointwiseNode(
        _ForcedField("b", battery.DT, g=s["g_b"], b0=s["b0_b"], b1=s["b1_b"]), mesh,
        shard_axes=0))
    gm.add_edge("b", "a", "x", "u", transform=jnp.mean)
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], diagnostics=diagnostics, **cfg.group(kind))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


_RUN: dict = {}


def _run(tmp_path_factory) -> battery.DomainRun:
    if "run" not in _RUN:
        try:
            _RUN["run"] = battery.domain_run("sharded", tmp_path_factory, cfg=CFG, builder=build)
        except Exception as exc:        # one failed build fails every row, once
            _RUN["run"] = exc
    if isinstance(_RUN["run"], Exception):
        raise _RUN["run"]
    return _RUN["run"]


def test_the_member_is_partitioned():
    """The premise: ``b``'s field lives on four devices, one shard each."""
    gm = build(CFG, "plain", diagnostics=False)
    field = gm._state["b"]["x"]
    assert len(field.sharding.device_set) == N_DEV
    assert len(field.addressable_shards) == N_DEV
    assert field.addressable_shards[0].data.shape == (N_CELLS // N_DEV,)


@pytest.mark.parametrize("row", battery._rows("sharded"))
def test_the_claim_holds_with_a_sharded_member(row, tmp_path_factory):
    """Each row's check, as in every other domain, on the group with ``b`` sharded."""
    battery.CHECKS[row](_run(tmp_path_factory))
