"""``strict_convergence`` on a step that spans several devices raises on every device.

``equinox.error_if`` raises from a host callback, and in a program
partitioned over several devices a callback runs once, on the first device:
the other devices went on to the step's next all-reduce and waited there for
a device that had stopped, until XLA aborted the process (MADD-ANO-162).
The check now runs inside a ``shard_map`` replicated over the step's mesh
(``_strict_error_if``), so every device raises at the same program point,
and its result is a zero token OR-ed into the bits of the checked state, so
nothing is gathered and nothing is rounded.

The raise itself is checked in a subprocess
(``test_coupling_claims_on_a_sharded_graph.py::test_strict_convergence_raises_with_a_sharded_member``):
a regression to the abort would take a test process with it.  Here, per
push, the parts that can be checked without risking that: the token is an
exact identity on every float dtype and special value, the checked state
keeps its partitioning, a strict group that converges steps exactly as the
same group without the check, and the mesh the check raises over is found
wherever the graph has one.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
from maddening.core.graph_manager import (
    GraphManager,
    _multi_device_mesh,
    _strict_error_if,
)
from tests.cloud.multigpu import test_coupling_claims_on_a_sharded_graph as sharded
from tests.core import test_coupling_claims_in_every_domain as battery

N_DEV = 4

pytestmark = pytest.mark.skipif(len(jax.devices()) < N_DEV,
                                reason=f"needs {N_DEV} CPU-virtual devices")


@pytest.fixture(scope="module")
def mesh():
    return create_device_mesh(shape=(N_DEV,))


def _bits(v):
    v = np.asarray(v)
    return v.view(np.dtype(f"uint{v.dtype.itemsize * 8}"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_the_token_is_an_exact_identity_on_every_value(mesh, dtype):
    """Signed zeros, a subnormal, the extremes, infinities and NaN come back
    bit for bit: the token is OR-ed into the bits, not added (``v + 0.0``
    turns ``-0.0`` into ``+0.0``, and a flushing add zeroes a subnormal)."""
    fi = jnp.finfo(dtype)
    vals = jnp.asarray([0.0, -0.0, float(fi.smallest_subnormal), -float(fi.tiny),
                        float(fi.max), -float(fi.max), np.inf, -np.inf, np.nan, 1.0, -3.5,
                        float(fi.eps)], dtype)
    vals = jnp.concatenate([vals, vals[:4]])        # sixteen: four per device
    x = jax.device_put(vals, NamedSharding(mesh, P("devices")))
    # The verdict is an argument, as a step's is: a constant ``False`` lets
    # the compiler fold the check and the token away.
    out = jax.jit(lambda v, p: _strict_error_if(
        {"v": v, "n": jnp.int32(3)}, p, "never", mesh))(x, jnp.asarray(False))
    assert np.array_equal(_bits(out["v"]), _bits(vals))
    assert int(out["n"]) == 3


def test_the_checked_state_keeps_its_partitioning(mesh):
    """The token is replicated and the OR elementwise: a field sharded over
    the mesh comes back sharded the same way, not gathered onto every
    device."""
    x = jax.device_put(jnp.arange(16.0, dtype=jnp.float32), NamedSharding(mesh, P("devices")))
    out = jax.jit(lambda v: _strict_error_if(v * 2.0, jnp.sum(v) < 0, "never", mesh))(x)
    assert len(out.sharding.device_set) == N_DEV
    assert out.addressable_shards[0].data.shape == (16 // N_DEV,)
    assert np.array_equal(np.asarray(out), np.arange(16.0) * 2.0)


def _strict_pair(strict):
    cfg = battery.Config("sharded", jnp.float32, jnp.float32, strict=False)
    gm = sharded.build(cfg, "strict" if strict else "plain", diagnostics=False)
    return gm


def test_a_converged_strict_step_with_a_sharded_member_is_the_unchecked_step():
    """A strict group that converges steps bit for bit as the same group
    without the check, and its sharded member stays partitioned: the per-device
    check and its token change nothing a run can see."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict = _strict_pair(True)
        plain = _strict_pair(False)
        p_strict = battery.params_for(strict, "steady")
        p_plain = battery.params_for(plain, "steady")
        for _ in range(3):
            strict.step(params=p_strict)
            plain.step(params=p_plain)
    assert strict._strict_mesh() is not None
    assert plain._strict_mesh() is None
    field = strict._state["b"]["x"]
    assert len(field.sharding.device_set) == N_DEV
    for n in ("a", "b"):
        for f in strict._state[n]:
            assert np.array_equal(_bits(strict._state[n][f]), _bits(plain._state[n][f])), (n, f)


def test_the_mesh_is_found_from_a_sharded_node_or_a_sharded_field(mesh):
    """A sharded node's mesh; a field placed with a ``NamedSharding`` over
    several devices when no node carries one; ``None`` on one device."""
    gm = GraphManager()
    gm.add_node(battery._Forced("a", battery.DT, jnp.float32, g=0.5, b0=1.0, b1=0.0))
    gm.add_node(ShardedPointwiseNode(
        sharded._ForcedField("b", battery.DT, g=0.5, b0=1.0, b1=0.0), mesh, shard_axes=0))
    assert _multi_device_mesh(gm._nodes, {}) is mesh

    plain = GraphManager()
    plain.add_node(battery._Forced("a", battery.DT, jnp.float32, g=0.5, b0=1.0, b1=0.0))
    assert _multi_device_mesh(plain._nodes, {"a": {"x": jnp.zeros(4)}}) is None
    placed = jax.device_put(jnp.zeros(8), NamedSharding(mesh, P("devices")))
    assert _multi_device_mesh(plain._nodes, {"a": {"x": placed}}) is mesh
    one = create_device_mesh(shape=(1,))
    single = jax.device_put(jnp.zeros(8), NamedSharding(one, P("devices")))
    assert _multi_device_mesh(plain._nodes, {"a": {"x": single}}) is None
