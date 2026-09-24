"""A sharded ``LBMNode`` is the unsharded node, or it refuses.

Two ways ``ShardedStencilNode(LBMNode)`` used to compute a different model
from the node it wraps, with no error:

* **A pressure face on a sharded axis.**  ``update_padded`` applies the
  Zou-He closure to the edge plane of its own slab, so sharding a
  pressure-driven channel along its inlet/outlet axis forced every seam
  between shards to the face pressures (on the 16x10 D2Q9 channel below:
  density 0.990 / 1.010 either side of the seam, centreline velocity 2.10x
  the unsharded one).  The first sharded step that imposes a pressure on
  such a face now raises, naming the axis to shard instead.
* **The halo fill at the global edges.**  The wrapper filled those halos
  with ``boundary="edge"`` by default, while the unsharded node streams
  periodically; sharding the same channel across its walls moved the
  centreline velocity by 0.64% after 200 steps.  The node now declares
  ``halo_boundary() == "periodic"`` and the wrapper refuses any other fill
  at construction -- its own default ``"edge"`` included (the STABLE
  signature is unchanged).  Wrap an ``LBMNode`` with
  ``boundary="periodic"``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.lbm import LBMNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")

CS2 = 1.0 / 3.0
NX, NY = 16, 10
PRESSURES = {"inlet_pressure": jnp.float32(CS2 * 1.01),
             "outlet_pressure": jnp.float32(CS2 * 0.99)}


def _mesh(n=2):
    return Mesh(np.array(jax.devices()[:n]), ("a",))


def _channel(**kw):
    """The auditor's walled channel: walls at y = 0 and y = NY - 1,
    inlet on x_min, outlet on x_max."""
    wall = np.zeros((NX, NY), bool)
    wall[:, 0] = wall[:, -1] = True
    return LBMNode("lbm", 1.0, grid_shape=(NX, NY), viscosity=1.0 / 6.0,
                   lattice="D2Q9", wall_mask=wall, **kw)


def _wrap(node, mesh=None, axis=1):
    """The one accepted way to shard an ``LBMNode``: periodic halos."""
    return ShardedStencilNode(node, mesh if mesh is not None else _mesh(),
                              axis_map={"a": axis}, boundary="periodic")


def _run(stepper, node, inputs, n_steps):
    """``n_steps`` of ``stepper`` from the node's initial state, as one
    compiled loop -- an eager Python loop spent seconds per test on
    op-by-op dispatch, and a graph jits its step anyway."""
    run = jax.jit(lambda st: jax.lax.fori_loop(
        0, n_steps, lambda _, s: stepper.update(s, inputs, 1.0), st))
    return {k: np.asarray(v) for k, v in run(node.initial_state()).items()}


# -- the pressure-face axis ---------------------------------------------------

def test_sharding_the_inlet_outlet_axis_of_a_pressure_channel_is_refused():
    node = _channel()
    wrapped = _wrap(node, axis=0)
    with pytest.raises(ValueError) as info:
        wrapped.update(node.initial_state(), PRESSURES, 1.0)
    message = str(info.value)
    assert "inlet_pressure on face 'x_min' (axis 0)" in message
    assert "outlet_pressure on face 'x_max' (axis 0)" in message
    assert "axis 0 into slabs of 8 of its 16 cells" in message
    assert "every seam between two shards" in message
    assert "axis 1 here" in message and "leave axis 0 unsharded" in message
    assert "body_force" in message


def test_one_imposed_pressure_on_a_sharded_face_axis_is_enough_to_refuse():
    node = _channel()
    wrapped = _wrap(node, axis=0)
    for key in PRESSURES:
        with pytest.raises(ValueError, match=f"{key} on face"):
            wrapped.update(node.initial_state(), {key: PRESSURES[key]}, 1.0)


def test_only_the_faces_actually_imposed_count():
    """Inlet on x_min, outlet on y_max, only the outlet imposed: sharding
    axis 0 splits no imposed face, runs, and is the unsharded node."""
    node = _channel(outlet_face="y_max")
    inputs = {"outlet_pressure": PRESSURES["outlet_pressure"]}
    got = _run(_wrap(node, axis=0), node, inputs, 20)
    want = _run(node, node, inputs, 20)
    for key in ("f", "density", "velocity"):
        np.testing.assert_allclose(got[key], want[key], rtol=1e-5, atol=1e-6, err_msg=key)


def test_a_face_axis_held_whole_by_one_device_is_not_a_seam():
    """A one-device mesh axis leaves the face axis in one slab: nothing to
    refuse, and the step is the unsharded one."""
    node = _channel()
    got = _run(_wrap(node, _mesh(1), axis=0), node, PRESSURES, 20)
    want = _run(node, node, PRESSURES, 20)
    np.testing.assert_allclose(got["velocity"], want["velocity"], rtol=1e-5, atol=1e-6)


def test_a_body_force_driven_node_may_shard_any_axis():
    """The faces act only when a pressure arrives, which is why the refusal
    is at the first step and not at construction."""
    node = _channel()
    inputs = {"body_force": jnp.asarray([1e-5, 0.0], jnp.float32)}
    for axis in (0, 1):
        got = _run(_wrap(node, axis=axis), node, inputs, 50)
        want = _run(node, node, inputs, 50)
        np.testing.assert_allclose(got["velocity"], want["velocity"], rtol=1e-5, atol=1e-6,
                                   err_msg=f"axis {axis}")


# -- the halo fill at the global edges ----------------------------------------

def test_with_periodic_halos_a_pressure_channel_sharded_across_its_walls_is_the_unsharded_node():
    """The auditor's layout, 200 steps, ``boundary="periodic"``: float32
    rounding (measured max|du| 2.6e-7).  With the wrapper's default
    ``"edge"``, which used to run: centreline 2.517e-2 against 2.534e-2
    (0.64%), max|du| 4.1e-4 -- now refused (below)."""
    node = _channel()
    wrapped = _wrap(node, axis=1)
    assert wrapped.to_dict()["boundary"] == "periodic"
    got = _run(wrapped, node, PRESSURES, 200)
    want = _run(node, node, PRESSURES, 200)
    centre = want["velocity"][NX // 2, NY // 2, 0]
    assert centre > 2e-2                        # the flow is really driven
    assert np.max(np.abs(got["velocity"] - want["velocity"])) < 2e-6
    np.testing.assert_allclose(got["density"], want["density"], rtol=1e-6)
    assert abs(got["velocity"][NX // 2, NY // 2, 0] / centre - 1.0) < 1e-4


def test_wrapping_an_lbm_node_with_the_default_boundary_is_refused_at_construction():
    """The wrapper's STABLE default stays ``"edge"``; for a node that
    declares ``"periodic"`` it is refused before anything is traced, and
    the message names the node, both fills and what to pass."""
    with pytest.raises(ValueError) as info:
        ShardedStencilNode(_channel(), _mesh(), axis_map={"a": 1})
    message = str(info.value)
    assert message.startswith("ShardedStencilNode: LBMNode 'lbm' declares "
                              "halo_boundary() == 'periodic'")
    assert "was given boundary='edge' (the default)" in message
    assert "would silently compute a different model from the unsharded one" in message
    assert message.endswith("Pass boundary='periodic'.")


@pytest.mark.parametrize("boundary", ["edge", "zero"])
def test_an_explicit_halo_fill_other_than_the_declared_one_is_refused(boundary):
    with pytest.raises(ValueError) as info:
        ShardedStencilNode(_channel(), _mesh(), axis_map={"a": 1}, boundary=boundary)
    message = str(info.value)
    assert "halo_boundary() == 'periodic'" in message
    assert f"was given boundary={boundary!r}" in message
    assert ("(the default)" in message) == (boundary == "edge")
    assert "Pass boundary='periodic'." in message


def test_a_node_that_declares_no_halo_boundary_is_wrapped_exactly_as_before():
    """Every other stencil node is unchanged: the default is ``"edge"``,
    any valid fill is taken as given, and the default step is the
    explicit-``"edge"`` step to the bit."""
    heat = HeatNode("h", 1e-4, n_cells=16, thermal_diffusivity=0.1)
    assert not hasattr(heat, "halo_boundary")
    default = ShardedStencilNode(heat, _mesh(), axis_map={"a": 0})
    assert default.to_dict()["boundary"] == "edge"
    for boundary in ("edge", "periodic", "zero"):
        wrapped = ShardedStencilNode(heat, _mesh(), axis_map={"a": 0}, boundary=boundary)
        assert wrapped.to_dict()["boundary"] == boundary
    explicit = ShardedStencilNode(heat, _mesh(), axis_map={"a": 0}, boundary="edge")
    a = _run(default, heat, {}, 5)
    b = _run(explicit, heat, {}, 5)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_a_declared_halo_boundary_that_is_not_a_mode_is_refused():
    class Declares(LBMNode):
        def halo_boundary(self):
            return "wrap"

    node = Declares("d", 1.0, grid_shape=(8, 4), viscosity=0.1, lattice="D2Q9")
    with pytest.raises(ValueError, match=r"halo_boundary\(\) returned 'wrap'"):
        ShardedStencilNode(node, _mesh(), axis_map={"a": 0}, boundary="periodic")
