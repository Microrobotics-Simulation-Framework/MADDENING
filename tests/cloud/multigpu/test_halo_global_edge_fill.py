"""The halos at the edges of the global grid, on every mesh-axis size and halo width.

``halo_exchange`` fills a halo between two shards from the neighbouring
shard, and a halo at an edge of the *global* grid from its ``boundary``
mode.  Two properties of that second fill are pinned here, each against a
reference derived without the implementation:

* **A mesh axis of one device is not special.**  Its one shard owns both
  global edges, so both of its halos are the boundary fill.  Until 0.4.0
  the fill was skipped on such an axis and the exchange's self-send
  handed the shard its own opposite edge: ``"edge"`` and ``"zero"`` both
  came out periodic, and a sharded heat rod on one device conducted heat
  from one end to the other as if it were a ring (MADD-ANO-028).
* **``"edge"`` replicates the edge cell across the whole halo**
  (``numpy.pad`` ``mode="edge"``).  Until 0.4.0 a halo two cells wide got
  ``r0, r1`` in front of ``r0`` -- the shard's first cells, not its edge
  cell repeated (MADD-ANO-029).

The primitive reference is the global array padded by ``numpy.pad`` and
cut into the shards' padded blocks; the adjoint reference scatters a
cotangent back through the same index map.  The node-level tests hold
:class:`ShardedStencilNode` to the unsharded node on 1, 2 and 4 devices.

What "the unsharded node" means for ``HeatNode``.  At ``stencil_order=2``
the node's own ghost for an end with no boundary input is ``T[0]``, the
``"edge"`` fill, so the sharded rod is compared with ``update`` itself.
At ``stencil_order=4`` no halo fill reproduces ``update``: its ghosts are
a cubic Dirichlet closure through the three end cells, which
``update_padded`` does not apply, so the reference there is the same
node's ``update_padded`` on the whole rod padded on one host -- the model
the sharded path computes, on every device count.  ``"zero"`` is compared
the same way: it is a different boundary condition from the node's
default, chosen by the caller.

Each configuration is compiled once: the primitive cases of one mesh run
as a single jitted program, and each node configuration is a single
cached ``shard_map``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo import halo_exchange
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode

_N_AVAILABLE = len(jax.devices())

#: ``numpy.pad`` spelling of each boundary mode.
_NP_MODE = {"periodic": "wrap", "edge": "edge", "zero": "constant"}
_MODES = ("periodic", "edge", "zero")


def _needs(n_devices: int):
    return pytest.mark.skipif(
        _N_AVAILABLE < n_devices,
        reason=f"needs >= {n_devices} JAX devices (the multigpu conftest forces 16 on CPU)",
    )


# ---------------------------------------------------------------------------
# NumPy reference: pad the global array, then cut it into padded blocks
# ---------------------------------------------------------------------------


def _padded_index_map(shape, halos, modes):
    """Global padded array of flat source indices, ``-1`` where a zero goes.

    Axes are padded in order, as ``halo_exchange`` exchanges them, so a
    corner comes from the second axis's fill of the first axis's halo.
    """
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    for axis, (h, mode) in enumerate(zip(halos, modes)):
        if h == 0:
            continue
        width = [(0, 0)] * len(shape)
        width[axis] = (h, h)
        if mode == "zero":
            idx = np.pad(idx, width, mode="constant", constant_values=-1)
        else:
            idx = np.pad(idx, width, mode=_NP_MODE[mode])
    return idx


def _blocks(padded_idx, mesh_shape, halos):
    """The shards' padded blocks, assembled the way shard_map returns them."""
    out = padded_idx
    for axis, (p, h) in enumerate(zip(mesh_shape, halos)):
        n = (out.shape[axis] - 2 * h) // p
        pieces = [
            np.take(out, np.arange(d * n, d * n + n + 2 * h), axis=axis)
            for d in range(p)
        ]
        out = np.concatenate(pieces, axis=axis)
    return out


def _reference(a, mesh_shape, halos, modes, cotangent_fn):
    """``(forward, adjoint)`` of the exchange, from the index map alone."""
    idx = _blocks(_padded_index_map(a.shape, halos, modes), mesh_shape, halos)
    flat = a.reshape(-1)
    forward = np.where(idx >= 0, flat[np.maximum(idx, 0)], 0.0).astype(a.dtype)
    ct = cotangent_fn(idx.shape)
    keep = idx >= 0
    adjoint = np.bincount(idx[keep], weights=ct[keep], minlength=flat.size)
    return forward, adjoint.reshape(a.shape).astype(a.dtype)


def _cotangent(shape):
    """Distinct small integers: every adjoint sum is exact in float32."""
    return (np.arange(int(np.prod(shape))) % 7 + 1).reshape(shape).astype(np.float32)


def test_the_reference_pads_the_way_the_documentation_says():
    """The reference, read off by hand on 8 cells, 2 shards, halo 2."""
    def row(mode, p=2):
        return _blocks(_padded_index_map((8,), (2,), (mode,)), (p,), (2,)).tolist()

    assert row("edge") == [0, 0, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 7, 7]
    assert row("zero") == [-1, -1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, -1, -1]
    assert row("periodic") == [6, 7, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 0, 1]
    # one shard: both global halos on the same block
    assert row("edge", p=1) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
    assert row("zero", p=1) == [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1]


# ---------------------------------------------------------------------------
# The primitive: every mode and width of one mesh in one compiled program
# ---------------------------------------------------------------------------


_1D_WIDTHS = (1, 2, 3)
_2D_WIDTHS = (1, 2)
_2D_MODE_PAIRS = tuple((m, m) for m in _MODES) + (
    ("edge", "periodic"), ("periodic", "zero"), ("zero", "edge"),
)


def _run_cases(mesh, spec, a, cases):
    """``{case: (forward, adjoint)}`` for ``cases = [(axes, boundary), ...]``."""
    fns = [
        shard_map(
            lambda x, _axes=axes, _b=boundary: halo_exchange(
                x, mesh=mesh, axes=_axes, boundary=_b),
            mesh=mesh, in_specs=spec, out_specs=spec,
        )
        for axes, boundary in cases
    ]
    shapes = [jax.eval_shape(fn, a).shape for fn in fns]
    cts = [jnp.asarray(_cotangent(s)) for s in shapes]

    @jax.jit
    def every_case(x, cotangents):
        return [(fn(x), jax.vjp(fn, x)[1](ct)[0]) for fn, ct in zip(fns, cotangents)]

    return [(np.asarray(f), np.asarray(g)) for f, g in every_case(a, cts)]


@pytest.fixture(scope="module", params=[1, 2, 4], ids=lambda n: f"{n}dev")
def slab_results(request):
    n = request.param
    if _N_AVAILABLE < n:
        pytest.skip(f"needs >= {n} JAX devices")
    mesh = create_device_mesh(shape=(n,))
    a = np.arange(24, dtype=np.float32) + 1.0
    cases = [([("devices", 0, h)], mode) for h in _1D_WIDTHS for mode in _MODES]
    got = _run_cases(mesh, P("devices"), jnp.asarray(a), cases)
    return n, a, dict(zip([(h, mode) for h in _1D_WIDTHS for mode in _MODES], got))


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("halo", _1D_WIDTHS)
def test_a_slab_exchange_fills_every_halo_slot_as_numpy_pad_does(slab_results, halo, mode):
    """1-D, on 1, 2 and 4 devices: forward and adjoint against NumPy, exactly."""
    n, a, results = slab_results
    forward, adjoint = results[(halo, mode)]
    want_fwd, want_adj = _reference(a, (n,), (halo,), (mode,), _cotangent)
    np.testing.assert_array_equal(forward, want_fwd, err_msg=f"{n} devices, halo {halo}, {mode}")
    np.testing.assert_array_equal(adjoint, want_adj, err_msg=f"{n} devices, halo {halo}, {mode}")


@pytest.fixture(scope="module", params=[(1, 1), (1, 2), (2, 1), (2, 2)],
                ids=lambda s: f"mesh{s[0]}x{s[1]}")
def pencil_results(request):
    shape = request.param
    if _N_AVAILABLE < shape[0] * shape[1]:
        pytest.skip(f"needs >= {shape[0] * shape[1]} JAX devices")
    mesh = create_device_mesh(shape=shape)
    a = (np.arange(8 * 12) % 97 + 1).reshape(8, 12).astype(np.float32)
    keys, cases = [], []
    for h in _2D_WIDTHS:
        for my, mz in _2D_MODE_PAIRS:
            keys.append((h, my, mz))
            cases.append(([("spatial_y", 0, h), ("spatial_z", 1, h)],
                          {"spatial_y": my, "spatial_z": mz}))
    got = _run_cases(mesh, P("spatial_y", "spatial_z"), jnp.asarray(a), cases)
    return shape, a, dict(zip(keys, got))


@pytest.mark.parametrize("modes", _2D_MODE_PAIRS, ids=lambda m: f"y={m[0]}-z={m[1]}")
@pytest.mark.parametrize("halo", _2D_WIDTHS)
def test_a_pencil_exchange_fills_every_halo_slot_as_numpy_pad_does(pencil_results, halo, modes):
    """2-D meshes, size-1 axes included, per-axis modes included."""
    shape, a, results = pencil_results
    forward, adjoint = results[(halo, *modes)]
    want_fwd, want_adj = _reference(a, shape, (halo, halo), modes, _cotangent)
    ctx = f"mesh {shape}, halo {halo}, modes {modes}"
    np.testing.assert_array_equal(forward, want_fwd, err_msg=ctx)
    np.testing.assert_array_equal(adjoint, want_adj, err_msg=ctx)


def test_one_device_zero_fill_is_zero_not_the_opposite_edge():
    """The primitive-level report of the defect, as written: 8 cells, halo 1.

    On one device this returned ``[8, 1, ..., 8, 1]`` -- the opposite
    edges -- against ``[0, 1, ..., 8, 0]`` on two.
    """
    mesh = create_device_mesh(shape=(1,))
    fn = shard_map(
        lambda x: halo_exchange(x, mesh=mesh, mesh_axis="devices", spatial_axis=0,
                                halo=1, boundary="zero"),
        mesh=mesh, in_specs=P("devices"), out_specs=P("devices"),
    )
    out = np.asarray(jax.jit(fn)(jnp.arange(8, dtype=jnp.float32) + 1))
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]


# ---------------------------------------------------------------------------
# The wrapper: sharded and unsharded halo axes, on a node every slot reaches
# ---------------------------------------------------------------------------


class _WideStencil2D(SimulationNode):
    """``f <- sum_k w_k (f[i+k, j] + 2 f[i, j+k])`` for ``|k| <= h``, asymmetric ``w``.

    Every halo cell of both axes enters some interior cell with its own
    weight, and the weights are not symmetric, so a mirrored or shifted
    fill changes the answer.  Integer data and weights keep every sum
    exact in float32, so sharded and unsharded agree to the bit on any
    backend.  Test-only; the built-in nodes are covered below.
    """

    def __init__(self, name: str, h: int, shape=(8, 8)):
        super().__init__(name=name, timestep=1.0)
        self._h = int(h)
        self._shape = tuple(shape)

    def halo_width(self) -> dict[int, int]:
        return {0: self._h, 1: self._h}

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        n = int(np.prod(self._shape))
        return {"f": jnp.asarray((np.arange(n) % 5 + 1).reshape(self._shape), jnp.float32)}

    def _interior(self, f_pad):
        h = self._h
        ny, nx = f_pad.shape[0] - 2 * h, f_pad.shape[1] - 2 * h
        acc = jnp.zeros((ny, nx), f_pad.dtype)
        for k in range(-h, h + 1):
            w = float(k + h + 1)                         # 1, 2, ..., 2h+1
            acc = acc + w * f_pad[h + k:h + k + ny, h:h + nx]
            acc = acc + 2.0 * w * f_pad[h:h + ny, h + k:h + k + nx]
        return acc

    def update(self, state, boundary_inputs, dt):
        raise NotImplementedError("the reference pads on the host and calls update_padded")

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        f_pad = state_padded["f"]
        h = self._h
        new = f_pad.at[h:-h, h:-h].set(self._interior(f_pad))
        return {"f": new}


_WRAPPER_MESHES = (
    # (mesh shape, axis_map): the axis_map's values are the sharded spatial
    # axes; a halo axis it leaves out is padded locally by the wrapper.
    pytest.param((1,), {"devices": 0}, id="1dev-axis1-local"),
    pytest.param((2,), {"devices": 0}, id="2dev-axis1-local"),
    pytest.param((2,), {"devices": 1}, id="2dev-axis0-local"),
    pytest.param((1, 2), {"spatial_y": 0, "spatial_z": 1}, id="mesh1x2"),
    pytest.param((2, 1), {"spatial_y": 0, "spatial_z": 1}, id="mesh2x1"),
    pytest.param((2, 2), {"spatial_y": 0, "spatial_z": 1}, id="mesh2x2"),
)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("h", (1, 2))
@pytest.mark.parametrize("mesh_shape,axis_map", _WRAPPER_MESHES)
def test_the_wrapper_fills_sharded_and_unsharded_halo_axes_alike(mesh_shape, axis_map, h, mode):
    """``ShardedStencilNode`` equals ``update_padded`` on the host-padded grid.

    Covers both routes a halo takes: exchanged on a sharded axis (sizes 1
    and 2) and padded locally on an axis ``axis_map`` leaves out -- the
    second is where the width-2 ``"edge"`` fill was wrong a second time.
    """
    if _N_AVAILABLE < int(np.prod(mesh_shape)):
        pytest.skip(f"needs >= {int(np.prod(mesh_shape))} JAX devices")
    node = _WideStencil2D("w", h)
    state = node.initial_state()
    f = np.asarray(state["f"])
    pad = f
    for axis in (0, 1):
        width = [(0, 0), (0, 0)]
        width[axis] = (h, h)
        pad = np.pad(pad, width, mode=_NP_MODE[mode])
    want = np.asarray(node._interior(jnp.asarray(pad)))

    sharded = ShardedStencilNode(node, create_device_mesh(shape=mesh_shape),
                                 axis_map=axis_map, boundary=mode)
    got = np.asarray(sharded.update(state, {}, 1.0)["f"])
    np.testing.assert_array_equal(got, want, err_msg=f"mesh {mesh_shape} {axis_map}, halo {h}, {mode}")


# ---------------------------------------------------------------------------
# HeatNode: sharded against unsharded on 1, 2 and 4 devices
# ---------------------------------------------------------------------------

_N_CELLS, _LENGTH, _ALPHA, _STEPS = 64, 1.0, 0.01, 20
_DX = _LENGTH / _N_CELLS
_DT = 0.25 * _DX * _DX / _ALPHA
#: A ramp -- cold left end, hot right end -- plus a wiggle, so that heat
#: flowing round a ring, a mirrored ghost and a shifted ghost all move the
#: end cells within the first step.
_X = np.linspace(_DX / 2, _LENGTH - _DX / 2, _N_CELLS)
_T0 = (_X + 0.3 * np.sin(3 * np.pi * _X)).astype(np.float32)
#: float32 rounding on O(1) temperatures; every defect here is >= 1e-3.
_ATOL = 1e-6


def _heat(order: int) -> HeatNode:
    return HeatNode("heat", timestep=_DT, n_cells=_N_CELLS, length=_LENGTH,
                    thermal_diffusivity=_ALPHA, initial_temperature=_T0,
                    stencil_order=order)


def _run(node, steps=_STEPS):
    state = node.initial_state()
    for _ in range(steps):
        state = node.update(state, {}, _DT)
    return np.asarray(state["temperature"])


def _run_host_padded(order: int, mode: str) -> np.ndarray:
    """The unsharded node's ``update_padded`` on the whole rod, padded here."""
    node = _heat(order)
    h = node.halo_width()[0]
    step = jax.jit(lambda T: node.update_padded(
        {"temperature": jnp.pad(T, h, mode=_NP_MODE[mode])}, {}, _DT)["temperature"][h:-h])
    T = jnp.asarray(_T0)
    for _ in range(_STEPS):
        T = step(T)
    return np.asarray(T)


def _sharded(order: int, mode: str, mesh_shape, axis_map=None) -> np.ndarray:
    axis_map = axis_map or {"devices": 0}
    return _run(ShardedStencilNode(_heat(order), create_device_mesh(shape=mesh_shape),
                                   axis_map=axis_map, boundary=mode))


_HEAT_MESHES = (
    pytest.param((1,), None, id="1dev"),
    pytest.param((2,), None, id="2dev", marks=_needs(2)),
    pytest.param((4,), None, id="4dev", marks=_needs(4)),
    pytest.param((1, 2), {"spatial_y": 0}, id="mesh1x2-cells-on-the-size-1-axis", marks=_needs(2)),
)


@pytest.mark.parametrize("mesh_shape,axis_map", _HEAT_MESHES)
def test_a_sharded_second_order_rod_is_the_unsharded_rod(mesh_shape, axis_map):
    """Default ``"edge"``: the node's own end ghost, so ``update`` is the reference.

    On one device this was 0.40 away after 50 steps -- both ends drifting
    to the mean, heat crossing from the hot end to the cold one.
    """
    np.testing.assert_allclose(_sharded(2, "edge", mesh_shape, axis_map), _run(_heat(2)),
                               rtol=0, atol=_ATOL)


@pytest.mark.parametrize("order", (2, 4))
@pytest.mark.parametrize("mode", ("edge", "zero"))
@pytest.mark.parametrize("mesh_shape,axis_map", _HEAT_MESHES)
def test_a_sharded_rod_computes_the_host_padded_model_on_every_device_count(
        mesh_shape, axis_map, mode, order):
    """Every order and mode: the answer does not depend on the device count.

    The reference is the node's ``update_padded`` on the whole rod padded
    by ``numpy.pad`` -- one host, no exchange.  At ``stencil_order=4`` with
    the default fill this was 7.3e-03 away on two and four devices (the
    ``r0, r1`` halo) and 0.40 on one (the ring).
    """
    np.testing.assert_allclose(_sharded(order, mode, mesh_shape, axis_map),
                               _run_host_padded(order, mode), rtol=0, atol=_ATOL)
