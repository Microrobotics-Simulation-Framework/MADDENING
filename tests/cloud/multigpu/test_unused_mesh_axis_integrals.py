"""A mesh axis the ``axis_map`` leaves unused: integrals replicated along it.

Every device along such an axis holds the same block, so a domain
integral is neither summed nor stacked over it.  The default reduction
used to ``psum`` over it, which JAX refused at the first step with an
error about ``psum`` naming neither the node nor the axis; a per-shard
integral was stacked over it, its partials repeated, so the stacked
values summed to twice the domain integral.

The neighbours: ``domain_integral_axes()`` naming the unused axis, or
naming only it; a third mesh axis; nesting; a gradient through
``run_scan``; and ``ShardedUnstructuredNode``, which treats every mesh
axis but its own this way and is the model followed.

The XLA miscompile found while widening the differential harness to
such meshes (MADD-ANO-068) is pinned at the end: ``run_scan`` refuses the
two configurations it was found in, ``step`` still answers as unsharded,
and the pure-JAX reproducer still miscompiles on this jaxlib.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 8,
    reason="needs 8 devices (the directory conftest forces 16 virtual CPU devices)")

#: A 1-D rod of 4 cells on a (2, 2) mesh; the axis_map shards it over one
#: mesh axis, chosen per test.
_ROD = D.StencilConfig(
    mesh_shape=(2, 2), axis_names=("px", "py"), axis_map=(("px", 0),), shape=(4,),
    halo=(1,), fill="edge", declares=False, contract="params", integral="scalar",
    integral_name="a_total", integral_listed=True, reads_shard_info=True, kappa="halo",
    kappa_axis=0, table=None, source="per_cell", misshapen_shape=(), gain=True,
    faces=True, dtype="float32", wrapping="single", steps=2, seed=4, surface="run_scan")


def _with_axes(axes):
    base = D.stencil_node_class(_ROD.contract, _ROD.reads_shard_info, _ROD.declares)
    return type("Axes", (base,), {"domain_integral_axes": lambda self: {"a_total": axes}})


def _step_both(cls, used, integral="scalar"):
    cfg = replace(_ROD, axis_map=((used, 0),), integral=integral)
    mesh = create_device_mesh(shape=cfg.mesh_shape, axis_names=cfg.axis_names)
    sharded = ShardedStencilNode(cls(cfg), mesh, {used: 0}, boundary=cfg.fill)
    plain = cls(cfg)
    got = sharded.update(sharded.initial_state(), {}, plain.delta_t)
    want = plain.update(plain.initial_state(), {}, plain.delta_t)
    return np.asarray(jax.device_get(got["a_total"])), np.asarray(want["a_total"])


@pytest.mark.parametrize("used", ["px", "py"])
@pytest.mark.parametrize("axes, shape", [
    (None, ()),                 # the default: summed over the axes that split the grid
    (("px", "py"), ()),         # naming the unused axis too changes nothing
    (("px",), "per-used"),      # reduced over px: reduced if px is used, stacked if not
    (("py",), "per-used"),
    ((), (2,)),                 # per-shard: stacked over the used axis only
], ids=["default", "both-named", "px-named", "py-named", "per-shard"])
def test_the_unused_axis_is_neither_reduced_nor_stacked_over(used, axes, shape):
    cls = D.stencil_node_class(_ROD.contract, _ROD.reads_shard_info, _ROD.declares) \
        if axes is None else _with_axes(axes)
    got, want = _step_both(cls, used)
    if shape == "per-used":
        shape = () if used in axes else (2,)
    assert got.shape == shape
    total = got.sum() if got.shape else got
    np.testing.assert_allclose(total, want, rtol=1e-5)


def test_a_misspelt_axis_is_still_refused():
    with pytest.raises(ValueError, match=r"names mesh axes \['pz'\]"):
        _step_both(_with_axes(("px", "pz")), "px")


@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
def test_a_third_mesh_axis_left_unused_on_a_pencil(integral):
    """A 2-D grid on a (2, 2, 2) mesh sharded over two of its axes."""
    cfg = D.StencilConfig(
        mesh_shape=(2, 2, 2), axis_names=("px", "py", "pz"), axis_map=(("px", 0), ("pz", 1)),
        shape=(4, 4), halo=(1, 1), fill="periodic", declares=True, contract="params",
        integral=integral, integral_name="a_total", integral_listed=False,
        reads_shard_info=True, kappa="halo", kappa_axis=1, table=None, source="scalar",
        misshapen_shape=(), gain=False, faces=False, dtype="float32", wrapping="single",
        steps=2, seed=6, surface="run_scan")
    out = D.check_config(cfg, surfaces=("run_scan", "step"))
    if integral == "per_shard":
        assert np.shape(out["run_scan"][0]["a_total"]) == (2, 2)


@pytest.mark.parametrize("integral", ["scalar", "per_shard"])
def test_a_nested_wrapper_on_an_unused_axis_answers_as_unsharded(integral):
    D.check_config(replace(_ROD, integral=integral, wrapping="nested"),
                   surfaces=("run_scan", "step", "set_state"))


def test_a_gradient_through_run_scan_with_an_integral_on_an_unused_axis():
    D.check_config(replace(_ROD, integral="vector", surface="gradient"))


@pytest.mark.parametrize("integral", ["scalar", "vector", "per_shard"])
@pytest.mark.parametrize("mesh_axis", ["px", "py"])
def test_the_unstructured_wrapper_on_a_2d_mesh_answers_as_unsharded(integral, mesh_axis):
    """The model followed: every mesh axis but the wrapper's own replicates the node."""
    cfg = D.UnstructuredConfig(
        n_devices=2, n_cells=5, assignment=(0, 0, 1, 1, 1), partition="uneven", chords=(),
        contract="params", integral=integral, integral_name="a_total", integral_listed=True,
        weight="halo", source="none", misshapen_len=0, gain=False, dtype="float32",
        wrapping="single", steps=2, seed=3, surface="run_scan")
    layout = D.unstructured_layout(cfg)
    cls = D.unstructured_node_class("params")
    mesh = create_device_mesh(shape=(2, 2), axis_names=("px", "py"))
    sharded = ShardedUnstructuredNode(cls(cfg, layout), mesh, layout, mesh_axis=mesh_axis)
    plain = cls(cfg, layout)
    s, u = sharded.initial_state(), plain.initial_state()
    for _ in range(cfg.steps):
        s, u = sharded.update(s, {}, 0.1), plain.update(u, {}, 0.1)
    got = sharded.gather_global(s)
    np.testing.assert_allclose(got["x"], np.asarray(u["x"]), rtol=0, atol=1e-6)
    total = got["a_total"].sum(axis=0) if integral == "per_shard" else got["a_total"]
    if integral == "per_shard":
        assert got["a_total"].shape == (2,)
    np.testing.assert_allclose(total, np.asarray(u["a_total"]), rtol=1e-5)


# ---------------------------------------------------------------------------
# MADD-ANO-068: XLA miscompiles a sharded static replicated over a mesh
# axis inside lax.scan.  GraphManager refuses the loop; step() is right.
# These were strict xfails until the refusal; the pure-JAX one is now the
# tripwire that says when an XLA release has fixed the compiler.
# ---------------------------------------------------------------------------

#: ``kappa`` (a sharded static) read in the halo, ``table`` (a replicated
#: array) read through a window at the shard's offset, on a mesh axis
#: the axis_map leaves unused: what the generated harness found.
_XLA_UNUSED_AXIS = replace(_ROD, integral=None, table="halo", source="none", gain=False,
                           faces=False, steps=1, seed=0)
#: The same kernel on a (2, 2) pencil, its static split along axis 0 only.
_XLA_PENCIL = replace(_XLA_UNUSED_AXIS, axis_map=(("px", 0), ("py", 1)), shape=(4, 4),
                      halo=(1, 1))


@pytest.mark.parametrize("cfg", [_XLA_UNUSED_AXIS, _XLA_PENCIL], ids=["unused-axis", "pencil"])
def test_run_scan_refuses_the_miscompiled_configurations_and_step_answers_as_unsharded(cfg):
    """Silent (every cell 1e-3 to 1e-2 off) on the unused axis, a compile failure
    on the pencil; both are now a refusal naming the static and mesh axis ``py``,
    and ``step`` still answers as the unsharded node."""
    assert D.loop_refusal_expected(cfg)
    out = D.check_config(cfg, surfaces=("step", "run_scan"))
    assert out["run_scan"][0] == "refused"


@pytest.mark.parametrize("cfg", [_XLA_UNUSED_AXIS, _XLA_PENCIL], ids=["unused-axis", "pencil"])
def test_the_refusal_leaves_the_graph_where_it_was(cfg):
    """Refused before anything runs: the state is not advanced, and a ``step``
    afterwards continues from the initial state."""
    case = D.build_case(cfg)
    gm, _ = D.build_graph(case, True)
    before = np.asarray(jax.device_get(gm.get_node_state(case.name)["f"]))
    with pytest.raises(RuntimeError, match=r"MADD-ANO-068.*'kappa'.*mesh axis 'py'"):
        gm.run_scan(1)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(gm.get_node_state(case.name)["f"])), before)
    gm.step()
    plain, _ = D.build_graph(case, False)
    plain.step()
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(gm.get_node_state(case.name)["f"])),
        np.asarray(plain.get_node_state(case.name)["f"]))


def _pure_jax_case():
    """The miscompile without MADDENING: a closed-over sharded constant, replicated
    over ``py``, halo-exchanged with an edge fill, times a smoothed window of a
    replicated constant at ``axis_index("px") * 2``; ``jit`` vs ``jit(scan)``."""
    from jax import shard_map
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("px", "py"))
    table = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    spec = NamedSharding(mesh, P("px"))
    k = jax.device_put(jnp.asarray([1.0, 2.0, 3.0, 4.0]), spec)
    f = jax.device_put(jnp.zeros(4), spec)

    def body(f, k):
        lh = lax.ppermute(k[-1:], "px", [(0, 1), (1, 0)])
        lh = jnp.where(lax.axis_index("px") == 0, k[:1], lh)
        w = lax.dynamic_slice_in_dim(table, lax.axis_index("px") * 2, 4)
        return f + jnp.concatenate([lh, k])[:-1] * (w[:-2] + 0.5 * w[1:-1] + 0.25 * w[2:])

    sm = jax.jit(shard_map(body, mesh=mesh, in_specs=(P("px"), P("px")), out_specs=P("px")))
    once = np.asarray(jax.jit(lambda f: sm(f, k))(f))
    scanned = np.asarray(jax.jit(
        lambda f: lax.scan(lambda c, _: (sm(c, k), None), f, None, length=1)[0])(f))
    return once, scanned


def test_the_pure_jax_reproducer_is_right_outside_a_scan():
    once, _ = _pure_jax_case()
    np.testing.assert_array_equal(once, [27.5, 45.0, 125.0, 240.0])


@pytest.mark.skipif(jax.default_backend() != "cpu",
                    reason="MADD-ANO-068 was measured on XLA's CPU backend; the tripwire "
                           "pins it there, and an accelerator backend compiles another program")
def test_the_pure_jax_reproducer_still_miscompiles_inside_a_scan():
    """The upstream defect the refusal exists for, on the jaxlib under test.

    When this fails, the installed XLA compiles the reproducer right: check
    the two configurations above with the refusal lifted, and if they agree
    too, lift ``GraphManager``'s MADD-ANO-068 refusal for jaxlib versions
    from that one and close the anomaly.  (Before the refusal this was a
    strict xfail of the opposite assertion.)
    """
    import jaxlib

    once, scanned = _pure_jax_case()
    assert not np.array_equal(scanned, once), (
        f"jaxlib {jaxlib.__version__}: jit(scan) now agrees with jit "
        f"({scanned} == {once}); see MADD-ANO-068")
