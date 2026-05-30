"""Tests for ShardedStencilNode runtime-slicing of sharded StaticArray.

v0.2.1 closed the gap where ``StaticArray(replication="shard", shard_axis=K)``
was metadata-only: GraphManager replicated the full array to every device
regardless of the declaration.  This file is the acceptance test for the
"3a materialisation" step — ShardedStencilNode now actually slices the
StaticArray per-device, halo-exchanges it with ``boundary="edge"``, and
delivers the sliced+padded slab to the inner node as ``static_padded``.

The test fixture is a synthetic 1-D stencil node with:

* one halo-1 state field ``f``,
* one sharded ``StaticArray`` mask ``mask`` (``replication="shard"``,
  ``shard_axis=0``) whose value at each interior cell scales the diffusion
  rate locally — this is the channel that proves the per-device slice is
  correct, since the global mask cannot all fit on one shard,
* one halo-stripped state output ``f``,
* one declared ``domain_integral_fields()`` output ``total_f`` whose value
  is the sum of f over the whole lattice — this is the channel that proves
  the wrapper applies ``lax.psum`` across the mesh.

Two assertions: (a) the sharded run on a CPU virtual-device mesh is
bit-compatible with a single-device baseline that exposes the same mask
via the unsharded path, and (b) the cross-device sum reduces correctly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray


_HAS_4_DEVICES = len(jax.devices()) >= 4
_SKIP_4 = "Requires >=4 JAX devices"


# ---------------------------------------------------------------------------
# Synthetic stencil node with a sharded mask + a domain-integral output.
# ---------------------------------------------------------------------------


class MaskedDiffusion1D(SimulationNode):
    """1-D diffusion modulated by a per-cell sharded mask.

    Update rule per interior cell ``i``:

        f_new[i] = f[i] + alpha * mask[i] * (f[i-1] - 2*f[i] + f[i+1]) * dt

    Boundary: ``boundary="edge"`` from the wrapper, so the ghost cells
    on the global boundary replicate the local edge value — equivalent
    to a zero-Neumann condition.  The wrapped ``mask`` slab arrives with
    matching ``"edge"`` halos from the static-data path.

    Domain-integral output: ``total_f = jnp.sum(f_new_interior)``.
    Under sharding this requires a cross-device ``lax.psum``.
    """

    def __init__(self, name: str, n: int, alpha: float, mask: jnp.ndarray):
        super().__init__(name=name, timestep=0.01)
        self._n = int(n)
        self._alpha = float(alpha)
        self._mask = jnp.asarray(mask, dtype=jnp.float32)
        if self._mask.shape != (self._n,):
            raise ValueError(
                f"mask shape {self._mask.shape} does not match n={self._n}"
            )

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    @property
    def static_data(self) -> dict:
        return {
            "mask": StaticArray(
                value=self._mask, replication="shard", shard_axis=0,
            ),
        }

    def state_fields(self) -> list[str]:
        return ["f"]

    def domain_integral_fields(self) -> set[str]:
        return {"total_f"}

    def initial_state(self) -> dict:
        rng = np.random.default_rng(42)
        return {"f": jnp.asarray(
            rng.standard_normal(self._n).astype(np.float32)
        )}

    def update(self, state, boundary_inputs, dt):
        # Unsharded fallback: do the masked Laplacian directly with
        # edge-replicating boundary (jnp.pad with mode="edge" mirrors
        # what the wrapper's boundary="edge" delivers).
        f = state["f"]
        f_pad = jnp.pad(f, 1, mode="edge")
        m = self._mask
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        f_new = f + self._alpha * m * lap * dt
        return {"f": f_new, "total_f": jnp.sum(f_new)}

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        f_pad = state_padded["f"]                  # shape (local_n + 2,)
        m_pad = static_padded["mask"]              # shape (local_n + 2,)
        # Interior: stripped of one cell on each side, since halo=1.
        f = f_pad[1:-1]
        m = m_pad[1:-1]
        lap = f_pad[2:] - 2 * f_pad[1:-1] + f_pad[:-2]
        f_new = f + self._alpha * m * lap * dt
        # Re-pad f_new so the wrapper can strip halos to the original shape.
        f_new_padded = jnp.concatenate(
            [f_pad[:1], f_new, f_pad[-1:]], axis=0,
        )
        return {
            "f": f_new_padded,
            "total_f": jnp.sum(f_new),  # local partial; wrapper psums.
        }


# ---------------------------------------------------------------------------
# Bit-compat: sharded vs unsharded one step.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_sharded_static_data_matches_unsharded_single_step():
    """One step of the sharded path matches the unsharded reference."""
    n = 16
    rng = np.random.default_rng(7)
    mask = jnp.asarray(0.5 + rng.random(n).astype(np.float32))
    node = MaskedDiffusion1D(name="diff", n=n, alpha=0.5, mask=mask)

    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="edge",
    )

    state = node.initial_state()
    dt = 0.01

    ref = node.update(state, {}, dt)
    out = sharded.update(state, {}, dt)

    np.testing.assert_allclose(
        np.asarray(out["f"]), np.asarray(ref["f"]), atol=0.0,
    )
    np.testing.assert_allclose(
        float(out["total_f"]), float(ref["total_f"]),
        atol=1e-5,  # psum reorders FP adds across shards — allow 1 ULP-ish.
    )


# ---------------------------------------------------------------------------
# Multi-step: 50 iterations should still match (sliding caches, halo
# exchange between every step, static_data unchanged across iterations).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_sharded_static_data_multi_step():
    n = 32
    rng = np.random.default_rng(11)
    mask = jnp.asarray(0.2 + 0.6 * rng.random(n).astype(np.float32))
    node = MaskedDiffusion1D(name="diff", n=n, alpha=0.4, mask=mask)

    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="edge",
    )

    state_ref = node.initial_state()
    state_shd = {k: v for k, v in state_ref.items()}  # shallow copy
    dt = 0.01
    for _ in range(50):
        new_ref = node.update(state_ref, {}, dt)
        new_shd = sharded.update(state_shd, {}, dt)
        state_ref = {"f": new_ref["f"]}
        state_shd = {"f": new_shd["f"]}

    np.testing.assert_allclose(
        np.asarray(state_shd["f"]), np.asarray(state_ref["f"]),
        atol=1e-5, rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Construction-time validation: shard_axis must line up with a spatial
# axis the wrapper actually shards.
# ---------------------------------------------------------------------------


class _BadShardAxisNode(SimulationNode):
    """Declares a sharded mask on axis 1, but only axis 0 is sharded."""

    def __init__(self):
        super().__init__(name="bad", timestep=0.01)
        self._mask = jnp.zeros((4, 4), dtype=jnp.float32)

    def halo_width(self) -> dict[int, int]:
        return {0: 1, 1: 1}

    @property
    def static_data(self) -> dict:
        return {
            # shard_axis=1 but the wrapper below only shards axis 0.
            "mask": StaticArray(
                value=self._mask, replication="shard", shard_axis=1,
            ),
        }

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        return {"f": jnp.zeros((4, 4), dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        return state_padded


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_sharded_static_shard_axis_must_match_wrapper_axes():
    """A StaticArray sharded on an axis the wrapper does not shard is rejected."""
    mesh = create_device_mesh(shape=(4,))
    node = _BadShardAxisNode()
    with pytest.raises(ValueError, match="shard_axis=1"):
        ShardedStencilNode(
            node, mesh, axis_map={"devices": 0}, boundary="edge",
        )


class _LegacyNoStaticPaddedNode(SimulationNode):
    """Declares a sharded mask but its update_padded omits ``static_padded=``."""

    def __init__(self):
        super().__init__(name="legacy", timestep=0.01)
        self._mask = jnp.zeros(4, dtype=jnp.float32)

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    @property
    def static_data(self) -> dict:
        return {
            "mask": StaticArray(
                value=self._mask, replication="shard", shard_axis=0,
            ),
        }

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        return {"f": jnp.zeros(4, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state

    def update_padded(self, state_padded, boundary_inputs, dt):
        # Legacy signature — no static_padded kwarg.  The wrapper must
        # reject construction with a clear message rather than silently
        # dropping the sliced slab.
        return state_padded


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_sharded_static_requires_static_padded_kwarg():
    """A node with sharded statics but no ``static_padded`` kwarg is rejected."""
    mesh = create_device_mesh(shape=(4,))
    node = _LegacyNoStaticPaddedNode()
    with pytest.raises(ValueError, match="does not accept 'static_padded'"):
        ShardedStencilNode(
            node, mesh, axis_map={"devices": 0}, boundary="edge",
        )


# ---------------------------------------------------------------------------
# shard_info delivery: extent matches local slab, offsets cover global
# domain, types are JAX scalars (not Python ints).
# ---------------------------------------------------------------------------


class _ShardInfoProbe(SimulationNode):
    """Halo-1 stencil that captures the ``shard_info`` it receives."""

    def __init__(self):
        super().__init__(name="probe", timestep=0.01)
        self.captured: dict = {}  # populated after .update()

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        return {"f": jnp.arange(16, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state

    def update_padded(
        self, state_padded, boundary_inputs, dt, *,
        static_padded=None, shard_info=None,
    ):
        # Stash the shard_info into a host-side dict at trace time.
        # ``offset`` is a traced scalar; we record it by emitting it as
        # an output that the test can read after JIT.
        if shard_info is None:
            return state_padded
        # Return one output per axis: offset_<axis> and extent_<axis>.
        out = dict(state_padded)
        for sax, (off, ext) in shard_info.items():
            out[f"_shardinfo_offset_{sax}"] = jnp.asarray(off, dtype=jnp.int32)
            out[f"_shardinfo_extent_{sax}"] = jnp.asarray(ext, dtype=jnp.int32)
        return out


@pytest.mark.skipif(not _HAS_4_DEVICES, reason=_SKIP_4)
def test_shard_info_arrives_at_update_padded():
    """The wrapper computes (global_offset, local_extent) per sharded axis."""
    mesh = create_device_mesh(shape=(4,))
    node = _ShardInfoProbe()
    # Add the shard_info outputs to state_fields-equivalent surface by
    # declaring them as domain integrals (so the wrapper psums them and
    # they appear as scalar replicated outputs we can read on host).
    node.state_fields = lambda: ["f"]  # type: ignore
    node.domain_integral_fields = lambda: {  # type: ignore
        "_shardinfo_offset_0", "_shardinfo_extent_0",
    }
    sharded = ShardedStencilNode(
        node, mesh, axis_map={"devices": 0}, boundary="edge",
    )

    state = node.initial_state()
    out = sharded.update(state, {}, 0.01)

    # Local extent must be global_n / n_devices = 16 / 4 = 4.  After
    # psum over 4 shards each holding extent=4, the value is 4*4 = 16.
    assert int(out["_shardinfo_extent_0"]) == 16
    # global_offset on shard r is r * 4.  psum across r in {0,1,2,3} is
    # 4 * (0 + 1 + 2 + 3) = 24.
    assert int(out["_shardinfo_offset_0"]) == 24
