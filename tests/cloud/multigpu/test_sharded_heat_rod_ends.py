"""A sharded ``HeatNode`` is the unsharded node, for the same boundary inputs.

``HeatNode.update`` closes each rod end with a ghost built from the
Dirichlet datum -- ``left_temperature`` / ``right_temperature``, or the
end cell when neither is given -- by ``2*T_b - T[0]`` at
``stencil_order=2`` and a cubic through the rod end at ``stencil_order=4``.
Since 0.4.0 ``update_padded`` builds the same ghosts on the block that
holds a rod end.  Until then it used whatever the halo exchange put
there and never read either temperature input (MADD-ANO-030), so:

* a sharded rod with its ends held at a temperature ran with the halo
  fill instead -- 0.87 from the unsharded rod with both ends at 0 after
  50 steps on the reference ramp below, whatever the device count;
* at ``stencil_order=4`` no fill could reproduce the cubic closure, so
  even a rod with no inputs was 2.5e-3 away (MADD-ANO-029's residual);
* ``boundary="zero"`` was the documented way to cool the ends, imposing
  0 at the ghost centres, half a cell outside the rod.

Now the wrapper's fill at the global ends never reaches the answer, so
``HeatNode.halo_boundary()`` declares ``"edge"`` and the wrapper refuses
the fills it would otherwise ignore, saying to pass the inputs instead.

The reference throughout is ``HeatNode.update`` itself, stepped with the
same boundary inputs.  Each configuration compiles one ``shard_map``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.heat import HeatNode

_N_AVAILABLE = len(jax.devices())


def _needs(n_devices: int):
    return pytest.mark.skipif(
        _N_AVAILABLE < n_devices,
        reason=f"needs >= {n_devices} JAX devices (the multigpu conftest forces 16 on CPU)",
    )


_N_CELLS, _LENGTH, _ALPHA, _STEPS = 64, 1.0, 0.01, 20
_DX = _LENGTH / _N_CELLS
_DT = 0.25 * _DX * _DX / _ALPHA
#: A ramp with a wiggle: every candidate closure moves the end cells in
#: the first step, and an end held at a temperature pulls against it.
_X = np.linspace(_DX / 2, _LENGTH - _DX / 2, _N_CELLS)
_T0 = (_X + 0.3 * np.sin(3 * np.pi * _X)).astype(np.float32)
#: float32 rounding on O(1) temperatures.  Measured: 0 at stencil_order=2,
#: <= 2.4e-7 at 4.  Every defect this file pins is >= 1e-3.
_ATOL = 1e-6

_INPUTS = {
    "no-inputs": {},
    "both-ends-at-0": {"left_temperature": jnp.float32(0.0),
                       "right_temperature": jnp.float32(0.0)},
    "ends-at-0.2-and-1.3": {"left_temperature": jnp.float32(0.2),
                            "right_temperature": jnp.float32(1.3)},
    "left-end-only-with-source": {
        "left_temperature": jnp.float32(0.5),
        "heat_source": jnp.asarray(np.sin(3 * _X).astype(np.float32)),
    },
}

_MESHES = (
    pytest.param((1,), {"devices": 0}, id="1dev"),
    pytest.param((2,), {"devices": 0}, id="2dev", marks=_needs(2)),
    pytest.param((4,), {"devices": 0}, id="4dev", marks=_needs(4)),
    pytest.param((1, 2), {"spatial_y": 0}, id="mesh1x2-cells-on-the-size-1-axis",
                 marks=_needs(2)),
)


def _heat(order: int, n_cells: int = _N_CELLS, dt: float = _DT) -> HeatNode:
    x = np.linspace(0.5 / n_cells, 1 - 0.5 / n_cells, n_cells)
    t0 = _T0 if n_cells == _N_CELLS else (x + 0.3 * np.sin(3 * np.pi * x)).astype(np.float32)
    return HeatNode("heat", timestep=dt, n_cells=n_cells, length=_LENGTH,
                    thermal_diffusivity=_ALPHA, initial_temperature=t0,
                    stencil_order=order)


def _run(node, inputs, steps=_STEPS, dt=_DT):
    state = node.initial_state()
    for _ in range(steps):
        state = node.update(state, inputs, dt)
    return np.asarray(state["temperature"])


@pytest.mark.parametrize("inputs", list(_INPUTS), ids=list(_INPUTS))
@pytest.mark.parametrize("order", (2, 4))
@pytest.mark.parametrize("mesh_shape,axis_map", _MESHES)
def test_a_sharded_rod_is_the_unsharded_rod_for_the_same_boundary_inputs(
        mesh_shape, axis_map, order, inputs):
    """Orders 2 and 4, with and without end temperatures, 1/2/4 devices."""
    bi = _INPUTS[inputs]
    sharded = ShardedStencilNode(_heat(order), create_device_mesh(shape=mesh_shape),
                                 axis_map=axis_map)
    np.testing.assert_allclose(_run(sharded, bi), _run(_heat(order), bi),
                               rtol=0, atol=_ATOL)


@_needs(8)
def test_a_thin_shard_closes_its_rod_end_through_the_halo():
    """Two cells per shard at ``stencil_order=4``: the cubic closure reads
    three end cells, the third of which is on the neighbouring shard."""
    n = 16
    dt = 0.25 * (1.0 / n) ** 2 / _ALPHA
    bi = _INPUTS["ends-at-0.2-and-1.3"]
    sharded = ShardedStencilNode(_heat(4, n, dt), create_device_mesh(shape=(8,)),
                                 axis_map={"devices": 0})
    np.testing.assert_allclose(_run(sharded, bi, dt=dt), _run(_heat(4, n, dt), bi, dt=dt),
                               rtol=0, atol=_ATOL)


@pytest.mark.parametrize("inputs", ["no-inputs", "ends-at-0.2-and-1.3"])
@pytest.mark.parametrize("order", (2, 4))
def test_the_fill_at_the_rod_ends_never_reaches_the_answer(order, inputs):
    """``update_padded`` on the whole rod, called directly (no
    ``shard_info``), is ``update`` whatever sits in the end halos -- with
    no inputs too, where the datum must come from the end cell and not
    from the halo beside it."""
    node = _heat(order)
    h = node.halo_width()[0]
    T = jnp.asarray(_T0)
    bi = _INPUTS[inputs]
    want = np.asarray(node.update({"temperature": T}, bi, _DT)["temperature"])
    for fill in (jnp.pad(T, h, mode="edge"), jnp.pad(T, h),
                 jnp.pad(T, h, constant_values=99.0)):
        got = np.asarray(node.update_padded({"temperature": fill}, bi, _DT)["temperature"])
        np.testing.assert_allclose(got[h:-h], want, rtol=0, atol=_ATOL)


@pytest.mark.parametrize("boundary", ["zero", "periodic"])
def test_a_fill_the_heat_rod_would_ignore_is_refused_naming_the_inputs(boundary):
    """``"zero"`` used to stand in for a cold end.  It is refused now,
    because the rod builds its own end ghosts, and the message says how
    to hold an end at 0 -- the same way as unsharded."""
    with pytest.raises(ValueError) as info:
        ShardedStencilNode(_heat(2), create_device_mesh(shape=(1,)),
                           axis_map={"devices": 0}, boundary=boundary)
    message = str(info.value)
    assert "HeatNode 'heat' declares halo_boundary() == 'edge'" in message
    assert f"was given boundary={boundary!r}" in message
    assert "Pass boundary='edge'." in message
    assert "left_temperature=0.0" in message
    assert "right_temperature=0.0" in message


def test_the_default_fill_is_the_declared_one():
    node = _heat(4)
    assert node.halo_boundary() == "edge"
    wrapped = ShardedStencilNode(node, create_device_mesh(shape=(1,)), axis_map={"devices": 0})
    assert wrapped.to_dict()["boundary"] == "edge"
