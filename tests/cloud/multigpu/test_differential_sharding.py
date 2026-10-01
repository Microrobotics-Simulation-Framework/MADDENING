"""Differential tests: a sharded graph answers as the same graph unsharded.

The oracle and the feature matrix are described in
``differential_sharding_support`` and in the "Differential tests" section
of ``docs/developer_guide/testing_standards.md``.  In one line: for a
generated node and a composition of sharded wrappers around it, every
graph surface -- ``step``, ``run``, ``run_scan``, ``run_scan(params=)``,
a parameter write followed by ``compile()``, ``set_node_state``, and a
gradient through ``run_scan`` -- gives the unsharded graph's answer, or
the sharded composition refuses loudly before it produces a state.

This module covers the three synthetic node families (one per wrapper)
and the wrapper-family examples of ``property_support``; the built-in
nodes, the refusals and the known-and-decided cases are in
``test_differential_sharding_builtins.py``.

Budget.  The per-push tests draw one surface per example at the
``EXAMPLES_COSTLY`` depth, with ``derandomize=True`` so CI runs the same
examples every time.  Each ``@pytest.mark.slow`` test is the broad
version of a named per-push test: every surface on every example, over
the whole matrix, gradients included.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.cloud.multigpu import differential_sharding_support as D
from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(
    not _HAS_4, reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")

#: Per-push: the profile's costly depth, the same examples on every run.
PER_PUSH = settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
#: The slow broad part searches at the profile's standard depth (50 under
#: ``dev``, 200 under ``ci``): one example there runs every surface, so
#: it is the full matrix the per-push tests sample one surface of.
BROAD = settings(max_examples=EXAMPLES_STANDARD, derandomize=True)


# ---------------------------------------------------------------------------
# Per-push: the generated feature matrix, one surface per example
# ---------------------------------------------------------------------------


@given(cfg=D.stencil_configs(ndim=1))
@PER_PUSH
def test_a_generated_1d_stencil_graph_answers_as_the_unsharded_graph(cfg):
    """``ShardedStencilNode`` on a rod: 1, 2 or 4 devices, every node feature."""
    D.check_config(cfg)


@given(cfg=D.stencil_configs(ndim=2))
@PER_PUSH
def test_a_generated_2d_stencil_graph_on_pencils_and_slabs_answers_as_the_unsharded_graph(cfg):
    """``ShardedStencilNode`` on a 2-D grid: pencils (2, 2), (1, 4), (4, 1) and slabs."""
    D.check_config(cfg)


@given(cfg=D.pointwise_configs())
@PER_PUSH
def test_a_generated_pointwise_graph_answers_as_the_unsharded_graph(cfg):
    """``ShardedPointwiseNode``: either axis, nested, around or inside a ``HybridNode``."""
    D.check_config(cfg)


@given(cfg=D.unstructured_configs())
@PER_PUSH
def test_a_generated_unstructured_graph_answers_as_the_unsharded_graph(cfg):
    """``ShardedUnstructuredNode``: uneven, balanced, global- and non-global-order partitions."""
    D.check_config(cfg)


@pytest.mark.parametrize("label", ["pointwise", "stencil", "unstructured"])
@pytest.mark.parametrize("surface", D.FORWARD_SURFACES)
def test_the_wrapper_family_examples_answer_as_unsharded_on_every_graph_surface(label, surface):
    """The ``property_support`` examples, through every graph surface on 4 devices.

    Their node-level equivalence is ``test_property_sharded_equals_unsharded``;
    this is the graph around them (``run``, ``run_scan``, ``params=``, a
    write and ``compile()``, ``set_node_state``).
    """
    D.check_config(D.ExampleConfig(label=label, n_devices=4, steps=3, surface=surface))


# ---------------------------------------------------------------------------
# Per-push: gradients through run_scan, on a fixed set of configurations
# ---------------------------------------------------------------------------

_GRADIENT_CASES = {
    "stencil-1d-periodic-halo-static-vector-integral": D.StencilConfig(
        mesh_shape=(4,), axis_names=("devices",), axis_map=(("devices", 0),),
        shape=(8,), halo=(1,), fill="periodic", declares=True, contract="params",
        integral="vector", integral_name="a_total", integral_listed=True,
        reads_shard_info=True, kappa="halo", kappa_axis=0, table="halo",
        source="per_cell", misshapen_shape=(), gain=True, faces=True,
        dtype="float32", wrapping="single", steps=3, seed=11, surface="gradient"),
    "stencil-2d-pencil-legacy-gain-nested": D.StencilConfig(
        mesh_shape=(2, 2), axis_names=("px", "py"), axis_map=(("px", 0), ("py", 1)),
        shape=(4, 6), halo=(2, 1), fill="edge", declares=False, contract="legacy",
        integral="vector", integral_name="z_total", integral_listed=False,
        reads_shard_info=True, kappa="interior", kappa_axis=1, table="interior",
        source="per_cell", misshapen_shape=(), gain=True, faces=True,
        dtype="float32", wrapping="nested", steps=2, seed=12, surface="gradient"),
    "stencil-2d-slab-x64-hybrid": D.StencilConfig(
        mesh_shape=(2,), axis_names=("devices",), axis_map=(("devices", 1),),
        shape=(3, 4), halo=(1, 2), fill="zero", declares=True, contract="params",
        integral="scalar", integral_name="a_total", integral_listed=False,
        reads_shard_info=False, kappa="halo", kappa_axis=1, table=None,
        source="scalar", misshapen_shape=(), gain=False, faces=False,
        dtype="float64", wrapping="hybrid", steps=2, seed=13, surface="gradient"),
    "pointwise-axis1-nested": D.PointwiseConfig(
        n_devices=4, shape=(2, 8), shard_axis=1, contract="params", total=True,
        source="per_cell", misshapen_shape=(), gain=True, dtype="float32",
        wrapping="nested", steps=3, seed=14, surface="gradient"),
    # Found by the slow lane (on (0, 0, 0), which has no ghost cells; see
    # the zero-ghost test below): a ring relaxed to nearly equal values,
    # so d x / d rate is a small difference of nearly equal numbers and the
    # float32 gradient is ill-conditioned on either path.
    "unstructured-relaxed-ring-ill-conditioned": D.UnstructuredConfig(
        n_devices=2, n_cells=3, assignment=(0, 0, 1), partition="uneven", chords=(),
        contract="params", integral="vector", integral_name="a_total",
        integral_listed=True, weight="halo", source="scalar", misshapen_len=0,
        gain=True, dtype="float32", wrapping="hybrid", steps=4, seed=605,
        surface="gradient"),
    "unstructured-uneven-per-shard-weights-hybrid": D.UnstructuredConfig(
        n_devices=4, n_cells=9, assignment=(0, 0, 0, 1, 3, 3, 1, 0, 3),
        partition="uneven", chords=((0, 4), (2, 7)), contract="params",
        integral="per_shard", integral_name="a_total", integral_listed=True,
        weight="halo", source="per_cell", misshapen_len=0, gain=True,
        dtype="float32", wrapping="hybrid", steps=3, seed=15, surface="gradient"),
}


@pytest.mark.parametrize("name", sorted(_GRADIENT_CASES))
def test_a_gradient_through_run_scan_is_the_unsharded_gradient(name):
    """d(loss)/d(theta) through ``run_scan``, both paths, within the derived bound.

    ``theta`` is the node's ``rate`` through ``params=`` (or, for the node
    on the three-argument contract, the external ``gain``).  The bound is
    :func:`differential_sharding_support.gradient_bound`, built from the
    absolute sums of a forward-mode pass.
    """
    D.check_config(_GRADIENT_CASES[name])


# ---------------------------------------------------------------------------
# Slow: the full matrix, every surface on every example
# ---------------------------------------------------------------------------


# Per push: tests/cloud/multigpu/test_differential_sharding.py::test_a_generated_1d_stencil_graph_answers_as_the_unsharded_graph
# Per push: tests/cloud/multigpu/test_differential_sharding.py::test_a_generated_2d_stencil_graph_on_pencils_and_slabs_answers_as_the_unsharded_graph
@pytest.mark.slow  # every surface on every example at EXAMPLES_STANDARD depth: a few seconds an example, minutes in all, nearly all XLA compile
@given(cfg=D.stencil_configs(surfaces=D.SURFACES, max_steps=4))
@BROAD
def test_the_full_stencil_matrix_answers_as_unsharded_on_every_surface(cfg):
    """Per-push siblings: ``test_a_generated_1d_stencil_graph_answers_as_the_unsharded_graph``
    and ``test_a_generated_2d_stencil_graph_on_pencils_and_slabs_answers_as_the_unsharded_graph``."""
    D.check_config(cfg, surfaces=D.SURFACES)


# Per push: tests/cloud/multigpu/test_differential_sharding.py::test_a_generated_pointwise_graph_answers_as_the_unsharded_graph
@pytest.mark.slow  # every surface on every example at EXAMPLES_STANDARD depth: a few seconds an example, minutes in all, nearly all XLA compile
@given(cfg=D.pointwise_configs(surfaces=D.SURFACES, max_steps=4))
@BROAD
def test_the_full_pointwise_matrix_answers_as_unsharded_on_every_surface(cfg):
    """Per-push sibling: ``test_a_generated_pointwise_graph_answers_as_the_unsharded_graph``."""
    D.check_config(cfg, surfaces=D.SURFACES)


# Per push: tests/cloud/multigpu/test_differential_sharding.py::test_a_generated_unstructured_graph_answers_as_the_unsharded_graph
@pytest.mark.slow  # every surface on every example at EXAMPLES_STANDARD depth: a few seconds an example, minutes in all, nearly all XLA compile
@given(cfg=D.unstructured_configs(surfaces=D.SURFACES, max_steps=4))
@BROAD
def test_the_full_unstructured_matrix_answers_as_unsharded_on_every_surface(cfg):
    """Per-push sibling: ``test_a_generated_unstructured_graph_answers_as_the_unsharded_graph``."""
    D.check_config(cfg, surfaces=D.SURFACES)


# Per push: tests/cloud/multigpu/test_differential_sharding.py::test_a_gradient_through_run_scan_is_the_unsharded_gradient
@pytest.mark.slow  # a reverse-mode scan and a forward-mode pass compiled per path per example, over the whole matrix
@given(cfg=st.one_of(D.stencil_configs(surfaces=("gradient",)),
                     D.pointwise_configs(surfaces=("gradient",)),
                     D.unstructured_configs(surfaces=("gradient",))))
@BROAD
def test_a_gradient_over_the_generated_matrix_is_the_unsharded_gradient(cfg):
    """Per-push sibling: ``test_a_gradient_through_run_scan_is_the_unsharded_gradient``."""
    D.check_config(cfg)


# ---------------------------------------------------------------------------
# Disagreements found by the harness, fixed: the exact cases it found
# ---------------------------------------------------------------------------

#: 4 cells on 4 devices in a ring: each shard owns 1 cell and has 2 ghosts,
#: so its slab is 3 rows long, and a 3-value ``source`` -- which the
#: unsharded node refuses (it is not one value per cell, and does not
#: broadcast to 4) -- is the length of a delivered slab.
_SLAB_INPUT = D.UnstructuredConfig(
    n_devices=4, n_cells=4, assignment=(0, 1, 2, 3), partition="balanced_global",
    chords=(), contract="params", integral=None, integral_name="a_total",
    integral_listed=False, weight=None, source="misshapen", misshapen_len=3,
    gain=False, dtype="float32", wrapping="single", steps=1, seed=0,
    surface="run_scan")


def test_a_slab_length_input_the_unsharded_node_refuses_is_refused_sharded():
    """A per-cell input the unsharded node refuses must not run sharded.

    ``ShardedStencilNode`` refuses an input the node declares per cell
    when it is neither the grid nor broadcastable to it (MADD-ANO-057).
    ``ShardedUnstructuredNode._cell_boundary_inputs`` had no such check:
    any leading length other than the layout's rows or the global cell
    count was replicated and handed whole to every shard, and here, where
    the source's length is a shard's slab, every shard read it as its own
    slab and the step ran.  The wrapper now refuses it before tracing,
    naming the global shape the node declares (the harness asserts the
    message, and that the unsharded node refuses too).
    """
    D.check_config(_SLAB_INPUT)


@pytest.mark.parametrize("mesh_shape, axis_names, axis_map, shape", [
    ((2,), ("devices",), (("devices", 0),), (4,)),
    ((2, 2), ("px", "py"), (("px", 0), ("py", 1)), (2, 2)),
], ids=["rod-on-2", "pencil-2x2"])
def test_a_per_shard_integral_under_a_nested_wrapper_runs_as_unsharded(
        mesh_shape, axis_names, axis_map, shape):
    """Found by the per-push 2-D test; reduced to the smallest grid that shows it.

    ``ShardedStencilNode.initial_state`` places a per-shard integral by
    broadcasting its value over the unreduced mesh axes
    (``_place_integral``).  When the node it wraps was itself a
    ``ShardedStencilNode``, it took the value from that wrapper's
    ``initial_state``, already stacked, and stacked it again: the state
    started at ``(2, 2)`` on two devices where one step returns ``(2,)``,
    ``step()`` silently changed the shape, and ``run_scan`` raised a
    ``TypeError`` about the scan carry.  The outer wrapper now places the
    value the node itself builds.
    """
    D.check_config(D.StencilConfig(
        mesh_shape=mesh_shape, axis_names=axis_names, axis_map=axis_map,
        shape=shape, halo=(1,) * len(shape), fill="edge", declares=False,
        contract="legacy", integral="per_shard", integral_name="a_total",
        integral_listed=False, reads_shard_info=True, kappa=None, kappa_axis=0,
        table=None, source="none", misshapen_shape=(), gain=False, faces=False,
        dtype="float32", wrapping="nested", steps=1, seed=0, surface="run_scan"))


def _listed_integral_config(listed: bool) -> D.UnstructuredConfig:
    return D.UnstructuredConfig(
        n_devices=2, n_cells=6, assignment=(0, 1, 0, 1, 1, 0),
        partition="balanced_nonglobal", chords=(), contract="params",
        integral="scalar", integral_name="a_total", integral_listed=listed,
        weight=None, source="none", misshapen_len=0, gain=False,
        dtype="float32", wrapping="single", steps=1, seed=3, surface="run_scan")


def _gather_global_against_harness(listed: bool, cfg=None) -> None:
    cfg = _listed_integral_config(listed) if cfg is None else cfg
    case = D.build_case(cfg)
    gm, node = D.build_graph(case, True)
    gm.run_scan(cfg.steps)
    state = gm.get_node_state(case.name)
    mine = case.gather(True, state)
    theirs = node.gather_global(state)
    assert set(theirs) == set(mine)
    for k in mine:
        np.testing.assert_array_equal(np.asarray(theirs[k]), mine[k], err_msg=k)


def test_gather_global_agrees_with_the_harness_gather_for_an_unlisted_integral():
    """``ShardedUnstructuredNode.gather_global`` against an independent gather."""
    _gather_global_against_harness(listed=False)


def test_gather_global_passes_an_integral_listed_in_state_fields_through():
    """``gather_global`` says domain integrals pass through unchanged.

    It decided by ``state_fields()`` membership, and the default
    ``state_fields()`` lists every ``initial_state`` key, an integral
    included: the scalar was reshaped to ``(n_devices, n_local_max)`` and
    numpy refused.  It now recognises an integral first, as the step does.
    """
    _gather_global_against_harness(listed=True)


def test_gather_global_passes_a_listed_per_shard_integral_through_when_a_shard_holds_one_cell():
    """The silent form of the case above.

    With ``n_local_max == 1`` the stacked per-shard values reshaped to
    ``(n_devices, 1)`` without complaint and were gathered as if they were
    cells: on a partition where device ``d`` owns cell ``3 - d`` the four
    shards' totals came back in reverse order, and nothing was raised.
    """
    _gather_global_against_harness(listed=True, cfg=D.UnstructuredConfig(
        n_devices=4, n_cells=4, assignment=(3, 2, 1, 0), partition="balanced_nonglobal",
        chords=(), contract="params", integral="per_shard", integral_name="a_total",
        integral_listed=True, weight=None, source="none", misshapen_len=0, gain=False,
        dtype="float32", wrapping="single", steps=1, seed=8, surface="run_scan"))


#: Two devices, two cells each, and no edge between the shards: the
#: partition has no ghost cell.
#: The subprocess inherits this process's environment, so it sees the
#: devices the directory conftest arranged (the only place allowed to set
#: them; ``test_conftest_device_policy.py``).
_ZERO_GHOST_SCRIPT = textwrap.dedent("""
    import jax, jax.numpy as jnp, numpy as np
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import SimulationNode

    class Decay(SimulationNode):
        def __init__(self):
            super().__init__(name="cells", timestep=1.0, rate=0.5)
        def initial_state(self):
            return {"x": jnp.arange(4, dtype=jnp.float32) + 1.0}
        def update(self, state, boundary_inputs, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            return {"x": state["x"] - p["rate"] * state["x"]}
        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None, params=None):
            p = self.params if params is None else {**self.params, **params}
            return {"x": state_padded["x"] - p["rate"] * state_padded["x"]}

    layout = build_unstructured_partition(
        partition_assignment=np.array([0, 0, 1, 1], np.int32),
        edges=np.array([[0, 1], [2, 3]], np.int32), n_devices=2)
    assert layout.n_ghost_max == 0
    grads = {}
    for exchange in ("all_to_all", "ppermute", None):
        node = Decay()
        if exchange is not None:
            node = ShardedUnstructuredNode(node, create_device_mesh(shape=(2,)), layout,
                                           exchange=exchange)
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()
        loss = lambda r: jnp.sum(
            gm.run_scan(2, params={"nodes": {"cells": {"rate": r}}})["cells"]["x"] ** 2)
        grads[exchange] = float(jax.grad(loss)(jnp.float32(0.5)))
    for exchange in ("all_to_all", "ppermute"):
        assert abs(grads[exchange] - grads[None]) <= 1e-5 * abs(grads[None]), grads
    print("OK", grads)
""")


def test_a_gradient_through_a_partition_without_ghosts_does_not_crash():
    """Found by the harness on CI's jaxlib 0.11.2 lane; jaxlib 0.10.2 and 0.11.0 ran it.

    Run in a subprocess: a segfault would take the test process with it.
    ``exchange_unstructured`` returned ``concatenate([local, zeros((0,
    ...))])`` when the layout had no ghost cell, and the transpose of that
    inside a ``lax.scan`` crashed XLA's compiler on jaxlib 0.11.2.  It now
    returns ``local`` itself, under both transports (the ``ppermute`` one
    built the same zero-size tail).  CI runs this on 0.10.2 and 0.11.2;
    the generated gradient tests draw such partitions too (every cell on
    one device, or shards no edge joins).
    """
    result = subprocess.run([sys.executable, "-c", _ZERO_GHOST_SCRIPT],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"exit {result.returncode} (a negative code is a signal: -11 is SIGSEGV)\n"
        f"{result.stderr[-2000:]}")
    assert result.stdout.startswith("OK"), result.stdout
