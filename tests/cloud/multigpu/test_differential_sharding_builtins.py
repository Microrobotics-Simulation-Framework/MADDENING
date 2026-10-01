"""Differential tests: built-in nodes, refusals, and the known-and-decided cases.

The same oracle as ``test_differential_sharding.py`` (see that module and
``differential_sharding_support``), applied to the built-in nodes a user
actually shards -- ``HeatNode`` at stencil orders 2 and 4, uniform and
non-uniform; ``LBMNode`` D2Q9 and D3Q19 channels with walls and an
obstacle -- and to the configurations the wrappers must refuse.

"Refused" means loudly, before the sharded graph produces a state: at
construction, at ``compile()``, or at the first step (``compile()`` does
not trace the step, so every check a wrapper makes in ``update`` fires
there).  The known-and-decided cases are asserted in that form, so a fix
that moves a refusal earlier keeps them passing.
"""

from __future__ import annotations

import re
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.hybrid_node import HybridNode
from tests.cloud.multigpu import differential_sharding_support as D
from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(
    not _HAS_4, reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")

#: Per-push: the profile's costly depth, the same examples on every run.
PER_PUSH = settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
#: The slow broad part: the profile's standard depth, every surface per example.
BROAD = settings(max_examples=EXAMPLES_STANDARD, derandomize=True)


def assert_refused_before_a_state(construct, exc=Exception, match=None):
    """The sharded node is refused at construction, ``compile()`` or the first step."""
    try:
        node = construct()
    except exc as e:
        assert match is None or re.search(match, str(e)), str(e)
        return "construct"
    gm = GraphManager()
    gm.add_node(node)
    with pytest.raises(exc) as info:
        gm.compile()
        gm.step()
    assert match is None or re.search(match, str(info.value)), str(info.value)
    return "run"


# ---------------------------------------------------------------------------
# HeatNode
# ---------------------------------------------------------------------------


@given(cfg=D.heat_configs())
@PER_PUSH
def test_a_generated_heat_rod_graph_answers_as_the_unsharded_rod(cfg):
    """Orders 2 and 4, rod-end temperatures, per-cell/scalar/mis-shaped sources.

    A non-uniform rod is refused sharded (``HeatNode.update_padded``
    supports only a uniform grid) and runs unsharded: asserted, as the
    harness predicts, by :func:`differential_sharding_support.check_config`.
    """
    D.check_config(cfg)


# Per push: tests/cloud/multigpu/test_differential_sharding_builtins.py::test_a_generated_heat_rod_graph_answers_as_the_unsharded_rod
@pytest.mark.slow  # every surface on every example at EXAMPLES_STANDARD depth: a few seconds an example, minutes in all, nearly all XLA compile
@given(cfg=D.heat_configs(surfaces=D.SURFACES, max_steps=6))
@BROAD
def test_the_full_heat_matrix_answers_as_unsharded_on_every_surface(cfg):
    """Per-push sibling: ``test_a_generated_heat_rod_graph_answers_as_the_unsharded_rod``."""
    D.check_config(cfg, surfaces=tuple(s for s in D.SURFACES if s != "set_state"))


def test_a_non_uniform_rod_is_refused_sharded_and_runs_unsharded():
    cfg = D.HeatConfig(n_devices=2, order=2, n_cells=6, nonuniform=True, ends=True,
                       source="none", wrapping="single", steps=2, seed=4,
                       surface="run_scan")
    case = D.build_case(cfg)
    assert_refused_before_a_state(lambda: case.make(True), NotImplementedError,
                                  "non-uniform grids under sharding")
    gm, _ = D.build_graph(case, False)
    gm.run_scan(cfg.steps, D._ext(case, False))
    assert np.all(np.isfinite(np.asarray(gm.get_node_state(case.name)["temperature"])))


# ---------------------------------------------------------------------------
# LBMNode
# ---------------------------------------------------------------------------


#: One compile per path per example, and a grid shape fixed by the mesh
#: (``vary_cells=False``), keep an LBM example near a quarter of a second;
#: the write-and-recompile and ``run`` surfaces (a second compile, or the
#: step program ``step`` already covers) and varied grids are in the slow
#: sibling.
_LBM_PER_PUSH_SURFACES = ("run_scan", "step", "scan_params")


@given(cfg=D.lbm_configs(lattices=("D2Q9",), surfaces=_LBM_PER_PUSH_SURFACES, max_steps=2,
                        vary_cells=False))
@PER_PUSH
def test_a_generated_d2q9_channel_graph_answers_as_the_unsharded_channel(cfg):
    """A walled D2Q9 channel with an obstacle, on slabs and pencils.

    Body force uniform, per cell or mis-shaped; pressure faces on the x
    faces, refused when the sharded step would split that axis.
    """
    D.check_config(cfg)


_D3Q19_CASES = {
    "pencil-1x4": D.LBMConfig(
        lattice="D3Q19", mesh_shape=(1, 4), axis_names=("px", "py"),
        axis_map=(("px", 0), ("py", 1)), shape=(2, 4, 3), force="per_cell",
        pressure=True, wrapping="single", steps=2, seed=21, surface="run_scan"),
    "slab-z-hybrid": D.LBMConfig(
        lattice="D3Q19", mesh_shape=(2,), axis_names=("devices",),
        axis_map=(("devices", 2),), shape=(2, 2, 6), force="uniform",
        pressure=False, wrapping="hybrid", steps=2, seed=22, surface="write_compile"),
}


@pytest.mark.parametrize("name", sorted(_D3Q19_CASES))
def test_a_d3q19_channel_graph_answers_as_the_unsharded_channel(name):
    D.check_config(_D3Q19_CASES[name])


# Per push: tests/cloud/multigpu/test_differential_sharding_builtins.py::test_a_generated_d2q9_channel_graph_answers_as_the_unsharded_channel
# Per push: tests/cloud/multigpu/test_differential_sharding_builtins.py::test_a_d3q19_channel_graph_answers_as_the_unsharded_channel
@pytest.mark.slow  # every surface, D2Q9 and D3Q19, varied grids: one LBM compile per path per surface, minutes in all
@given(cfg=D.lbm_configs(surfaces=D.SURFACES))
@BROAD
def test_the_full_lbm_matrix_answers_as_unsharded_on_every_surface(cfg):
    """Per-push siblings: ``test_a_generated_d2q9_channel_graph_answers_as_the_unsharded_channel``
    and ``test_a_d3q19_channel_graph_answers_as_the_unsharded_channel``."""
    D.check_config(cfg, surfaces=tuple(s for s in D.SURFACES if s != "set_state"))


# ---------------------------------------------------------------------------
# Gradients through run_scan, built-in nodes
# ---------------------------------------------------------------------------

_BUILTIN_GRADIENTS = {
    "heat-order4-nested-ends-source": D.HeatConfig(
        n_devices=4, order=4, n_cells=12, nonuniform=False, ends=True,
        source="per_cell", wrapping="nested", steps=4, seed=31, surface="gradient"),
    "heat-order2-hybrid": D.HeatConfig(
        n_devices=2, order=2, n_cells=8, nonuniform=False, ends=False,
        source="scalar", wrapping="hybrid", steps=3, seed=32, surface="gradient"),
    "lbm-d2q9-pencil-2x2": D.LBMConfig(
        lattice="D2Q9", mesh_shape=(2, 2), axis_names=("px", "py"),
        axis_map=(("px", 0), ("py", 1)), shape=(4, 6), force="uniform",
        pressure=False, wrapping="single", steps=2, seed=33, surface="gradient"),
}


@pytest.mark.parametrize("name", sorted(_BUILTIN_GRADIENTS))
def test_a_gradient_through_a_built_in_node_is_the_unsharded_gradient(name):
    """d(loss)/d(thermal_diffusivity or viscosity) through ``run_scan``."""
    D.check_config(_BUILTIN_GRADIENTS[name])


# ---------------------------------------------------------------------------
# A grid the mesh cannot split: the refusal's advice, followed
# ---------------------------------------------------------------------------


def _counts_available():
    return [n for n in (2, 3, 4) if n <= len(jax.devices())]


@st.composite
def _non_dividing(draw):
    n_dev = draw(st.sampled_from(_counts_available()))
    k = draw(st.integers(1, 3))
    r = draw(st.integers(1, n_dev - 1))
    pencil = draw(st.booleans()) and n_dev == 2
    fill = draw(st.sampled_from(D.FILLS))
    if pencil:
        # One axis of a (2, 2) pencil does not divide; the other does.
        bad_axis = draw(st.sampled_from([0, 1]))
        shape = tuple(2 * k + r if a == bad_axis else 4 for a in range(2))
        return D.StencilConfig(
            mesh_shape=(2, 2), axis_names=("px", "py"), axis_map=(("px", 0), ("py", 1)),
            shape=shape, halo=(1, 1), fill=fill, declares=False, contract="params",
            integral=None, integral_name="a_total", integral_listed=False,
            reads_shard_info=True, kappa=None, kappa_axis=0, table=None, source="per_cell",
            misshapen_shape=(), gain=False, faces=True, dtype="float32", wrapping="single",
            steps=2, seed=draw(st.integers(0, 99)), surface="run_scan"), bad_axis
    return D.StencilConfig(
        mesh_shape=(n_dev,), axis_names=("devices",), axis_map=(("devices", 0),),
        shape=(n_dev * k + r,), halo=(1,), fill=fill, declares=False, contract="params",
        integral="scalar", integral_name="a_total", integral_listed=False,
        reads_shard_info=True, kappa="halo", kappa_axis=0, table=None, source="per_cell",
        misshapen_shape=(), gain=False, faces=True, dtype="float32", wrapping="single",
        steps=2, seed=draw(st.integers(0, 99)), surface="run_scan"), 0


def _follow_the_advice(cfg, bad_axis):
    with pytest.raises(ValueError) as info:
        D.build_case(cfg).make(True)
    msg = str(info.value)
    next_up = int(re.search(r"the next one up is (\d+)", msg).group(1))
    divisors = [int(d) for d in
                re.search(r"device counts that do divide \d+ \(\[([\d, ]+)\]\)", msg)
                .group(1).split(",")]
    mesh_axis = dict((sa, ma) for ma, sa in cfg.axis_map)[bad_axis]
    n_dev = dict(zip(cfg.axis_names, cfg.mesh_shape))[mesh_axis]
    assert next_up % n_dev == 0 and next_up > cfg.shape[bad_axis]
    assert divisors and all(cfg.shape[bad_axis] % d == 0 for d in divisors)

    resized = list(cfg.shape)
    resized[bad_axis] = next_up
    D.check_config(replace(cfg, shape=tuple(resized)))
    for d in divisors:
        mesh = tuple(d if name == mesh_axis else size
                     for name, size in zip(cfg.axis_names, cfg.mesh_shape))
        D.check_config(replace(cfg, mesh_shape=mesh))


def _advice_case(n_dev, extent, pencil_axis=None):
    """A non-dividing configuration, as :func:`_non_dividing` draws them."""
    if pencil_axis is not None:
        shape = tuple(extent if a == pencil_axis else 4 for a in range(2))
        return D.StencilConfig(
            mesh_shape=(2, 2), axis_names=("px", "py"), axis_map=(("px", 0), ("py", 1)),
            shape=shape, halo=(1, 1), fill="periodic", declares=False, contract="params",
            integral=None, integral_name="a_total", integral_listed=False,
            reads_shard_info=True, kappa=None, kappa_axis=0, table=None, source="per_cell",
            misshapen_shape=(), gain=False, faces=True, dtype="float32", wrapping="single",
            steps=2, seed=1, surface="run_scan"), pencil_axis
    return D.StencilConfig(
        mesh_shape=(n_dev,), axis_names=("devices",), axis_map=(("devices", 0),),
        shape=(extent,), halo=(1,), fill="edge", declares=False, contract="params",
        integral="scalar", integral_name="a_total", integral_listed=False,
        reads_shard_info=True, kappa="halo", kappa_axis=0, table=None, source="per_cell",
        misshapen_shape=(), gain=False, faces=True, dtype="float32", wrapping="single",
        steps=2, seed=1, surface="run_scan"), 0


@pytest.mark.parametrize("n_dev, extent, pencil_axis", [
    (4, 6, None),      # next multiple 8; device counts dividing 6: [1, 2]
    (3, 7, None),      # next multiple 9; only 1 divides 7
    (2, 5, 1),         # one axis of a (2, 2) pencil
], ids=["6-cells-on-4", "7-cells-on-3", "pencil-axis-1"])
def test_the_non_dividing_grid_refusal_gives_advice_that_works_when_followed(
        n_dev, extent, pencil_axis):
    """Both ways out the message names -- the next multiple, or a device count
    that divides -- construct, and the sharded graph then answers as unsharded."""
    _follow_the_advice(*_advice_case(n_dev, extent, pencil_axis))


# Per push: tests/cloud/multigpu/test_differential_sharding_builtins.py::test_the_non_dividing_grid_refusal_gives_advice_that_works_when_followed
@pytest.mark.slow  # up to five sharded graphs compiled per example (the resized grid and each dividing device count)
@given(drawn=_non_dividing())
@BROAD
def test_the_non_dividing_grid_refusal_advice_works_over_generated_grids(drawn):
    """Per-push sibling: ``test_the_non_dividing_grid_refusal_gives_advice_that_works_when_followed``."""
    _follow_the_advice(*drawn)


# ---------------------------------------------------------------------------
# Known and decided: asserted, not reported again
# ---------------------------------------------------------------------------


def _small_pipe():
    from maddening.nodes.lbm_pipe import LBMPipeNode
    return LBMPipeNode("pipe", 1.0, nx=8, ny=6, nz=6, propeller_x=3)


_BASE = D.StencilConfig(
    mesh_shape=(2,), axis_names=("devices",), axis_map=(("devices", 0),), shape=(4,),
    halo=(1,), fill="periodic", declares=True, contract="params", integral=None,
    integral_name="a_total", integral_listed=False, reads_shard_info=True, kappa=None,
    kappa_axis=0, table=None, source="none", misshapen_shape=(), gain=False, faces=False,
    dtype="float32", wrapping="single", steps=1, seed=0, surface="run_scan")

_KNOWN_LATE = {
    # A halo wider than a shard: 1 cell per shard, halo 2.
    "halo-wider-than-a-shard": lambda: ShardedStencilNode(
        D.make_stencil_node(replace(_BASE, shape=(2,), halo=(2,), fill="edge", declares=False)),
        create_device_mesh(shape=(2,)), {"devices": 0}, boundary="edge"),
    # Two mesh axes sharding one spatial axis.
    "two-mesh-axes-on-one-spatial-axis": lambda: ShardedStencilNode(
        D.make_stencil_node(replace(_BASE, shape=(8, 4), halo=(1, 1), fill="edge",
                                    declares=False)),
        create_device_mesh(shape=(2, 2), axis_names=("px", "py")), {"px": 0, "py": 0},
        boundary="edge"),
    # LBMPipeNode declares a halo and has no update_padded.
    "stencil-wrapper-around-lbm-pipe": lambda: ShardedStencilNode(
        _small_pipe(), create_device_mesh(shape=(2,)), {"devices": 0}, boundary="edge"),
    # HybridNode forwards neither halo_boundary() nor update_padded: a
    # periodic node inside it is accepted under the default "edge" fill...
    "stencil-wrapper-around-a-hybrid-node": lambda: ShardedStencilNode(
        HybridNode(D.make_stencil_node(_BASE), D._correction),
        create_device_mesh(shape=(2,)), {"devices": 0}, boundary="edge"),
}


@pytest.mark.parametrize("name", sorted(_KNOWN_LATE))
def test_a_known_late_refusal_is_still_loud_before_any_state(name):
    """Known and decided: these are refused at the first step, not at construction.

    ...and the hybrid one is then refused there, because the default
    ``update_padded`` raises for a node with a halo.  A fix that refuses
    any of them earlier keeps this passing.
    """
    assert_refused_before_a_state(_KNOWN_LATE[name])


def test_the_lbm_pipe_node_runs_unsharded():
    gm = GraphManager()
    gm.add_node(_small_pipe())
    gm.compile()
    gm.step()
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("pipe")["f"])))


def test_a_per_cell_input_on_a_balanced_out_of_order_partition_is_refused():
    """Known and decided: a global-order and a partition-layout input look alike there."""
    out = D.check_config(D.UnstructuredConfig(
        n_devices=2, n_cells=6, assignment=(0, 1, 0, 1, 0, 1),
        partition="balanced_nonglobal", chords=(), contract="params", integral=None,
        integral_name="a_total", integral_listed=False, weight=None, source="per_cell",
        misshapen_len=0, gain=False, dtype="float32", wrapping="single", steps=1, seed=5,
        surface="run_scan"))
    assert out == {"refused": "run",
                   "known": "balanced non-global partition refuses per-cell inputs"}


def test_madd_ano_035_a_global_order_state_on_a_balanced_out_of_order_partition_is_read_as_layout():
    """Known and decided (MADD-ANO-035): ``set_node_state`` cannot tell the orders apart.

    A global-order state written to a balanced partition out of global
    order is read as partition layout: the sharded graph steps the state
    whose cell ``g`` holds the written array's entry at ``g``'s layout
    row.  Asserted exactly, both ways: it equals the unsharded graph
    stepped from that permuted state, and differs from the unsharded
    graph stepped from the state as written.
    """
    cfg = D.UnstructuredConfig(
        n_devices=2, n_cells=6, assignment=(0, 1, 0, 1, 0, 1),
        partition="balanced_nonglobal", chords=((0, 3),), contract="params",
        integral=None, integral_name="a_total", integral_listed=False, weight=None,
        source="none", misshapen_len=0, gain=False, dtype="float32", wrapping="single",
        steps=2, seed=6, surface="set_state")
    case = D.build_case(cfg)
    written = np.random.default_rng(1).standard_normal(cfg.n_cells).astype(np.float32)
    rows = D.layout_rows_of_global(case.layout)

    def run(sharded, x):
        gm, _ = D.build_graph(case, sharded)
        gm.set_node_state(case.name, {"x": jnp.asarray(x)})
        gm.run_scan(cfg.steps)
        return case.gather(sharded, gm.get_node_state(case.name))["x"]

    sharded = run(True, written)
    as_layout = run(False, written[rows])
    as_written = run(False, written)
    np.testing.assert_allclose(sharded, as_layout, rtol=0,
                               atol=D.grid_atol(cfg, as_layout))
    assert not np.allclose(sharded, as_written, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# A mesh axis the axis_map leaves unused
# ---------------------------------------------------------------------------

_UNUSED_AXIS = replace(_BASE, mesh_shape=(2, 2), axis_names=("px", "py"),
                       axis_map=(("px", 0),), fill="edge", declares=False,
                       kappa="halo", source="per_cell", steps=2)


def test_grid_fields_on_a_mesh_axis_the_axis_map_leaves_unused_answer_as_unsharded():
    """A 1-D grid on a (2, 2) mesh sharded over ``px`` only: replicated over ``py``."""
    D.check_config(_UNUSED_AXIS)


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "differential: a domain integral on a mesh with an axis the axis_map leaves "
    "unused fails at the first step inside lax.psum, naming neither; pending fix"))
def test_a_domain_integral_on_a_mesh_axis_the_axis_map_leaves_unused_is_summed_or_refused_early():
    """Sharded = unsharded, or refused at construction: either fix flips this.

    The default reduction ``psum``-s over every mesh axis, and the state is
    replicated over the unused one; JAX's varying-axes check refuses the
    sum at the first trace with ``jax.lax.psum can only accept
    axis_name ...``.  (A per-shard integral runs there, stacked over both
    axes, its partials repeated along the unused one.)
    """
    cfg = replace(_UNUSED_AXIS, integral="scalar")
    case = D.build_case(cfg)
    try:
        case.make(True)
    except ValueError as e:
        assert "py" in str(e)
        return
    D.check_config(cfg)
