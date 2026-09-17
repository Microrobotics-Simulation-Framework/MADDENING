"""Sharding changes where the arithmetic happens, not what it computes.

This is the headline correctness property of the whole multi-GPU
subsystem, and the one a user leans on hardest: a model is developed on
one device, wrapped for many, and the single-device answer is the
reference it is trusted against.  Stating it once, over a generated node
size, device count and step count, is worth more than any number of
fixed-size examples -- the whole-tree audit's W6 is a case where the two
paths ran *different physics* and every example-based test in the
directory still passed, because none of them used the field that got
dropped.

What is generated: which wrapper, how many devices, how many cells per
shard, how many steps, the parameter value, and the data that has to
survive partitioning -- a per-cell boundary input, a replicated scalar
boundary input and a sharded ``StaticArray``.

Tolerances.  The halo exchange is pure communication, so a 1-D stencil
is bit-exact under sharding and the tolerance below is far looser than
it needs to be; it is stated in one place, for all three wrappers, so
that a future wrapper that does reduce across devices has a number to
meet.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.cloud.multigpu.property_support import (
    WRAPPER_FAMILY,
    build_stencil,
    device_counts,
    param_values,
    wrapper_labels,
)
from tests.conftest import EXAMPLES_COSTLY

#: What "agree" means for a sharded run against its unsharded reference.
RTOL, ATOL = 1e-5, 1e-6


def _assert_agrees(sharded: dict, unsharded: dict, context: str) -> None:
    assert set(sharded) == set(unsharded), context
    for field, value in unsharded.items():
        np.testing.assert_allclose(sharded[field], value, rtol=RTOL, atol=ATOL,
                                   err_msg=f"{context}: {field}")


@given(label=wrapper_labels(), n_devices=device_counts(),
       cells_per_shard=st.integers(min_value=1, max_value=4),
       steps=st.integers(min_value=1, max_value=4), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_sharded_run_matches_the_unsharded_run_of_the_same_node(
        label, n_devices, cells_per_shard, steps, rate):
    """For any wrapper, mesh and node size: same node, same trajectory."""
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=rate,
                                 n_cells=n_devices * cells_per_shard)
    _assert_agrees(case.run(steps=steps, sharded=True),
                   case.run(steps=steps, sharded=False),
                   f"{label} on {n_devices} devices")


@given(n_devices=device_counts(),
       cells_per_shard=st.integers(min_value=1, max_value=4),
       gain=param_values(), steps=st.integers(min_value=1, max_value=3),
       seed=st.integers(0, 2**16))
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_per_cell_boundary_input_survives_partitioning(
        n_devices, cells_per_shard, gain, steps, seed):
    """A grid-shaped boundary input reaches the shard that owns the cell.

    Two halves, because either one alone can pass while the feature is
    broken: the sharded answer matches the unsharded one, *and* the
    input actually moved the answer.  W6 is the case where the second
    half fails silently -- a field the sharded path ignores is accepted
    by ``validate()`` and discarded, so the sharded run agrees with
    nothing but itself.
    """
    n_cells = n_devices * cells_per_shard
    rng = np.random.default_rng(seed)
    source = jnp.asarray(rng.standard_normal(n_cells).astype(np.float32))
    # Enough of a forcing that its effect clears RTOL/ATOL by orders of
    # magnitude: the second half of this property asserts that the input
    # moved the answer, and a forcing at the tolerance would make that a
    # coin toss rather than a statement about the wrapper.
    assume(float(jnp.max(jnp.abs(source))) > 0.1)

    case = build_stencil(n_devices=n_devices, n_cells=n_cells)
    case.boundary_inputs = {"source": source, "gain": jnp.float32(gain)}
    with_source = case.run(steps=steps, sharded=True)
    _assert_agrees(with_source, case.run(steps=steps, sharded=False),
                   f"per-cell input on {n_devices} devices")

    case.boundary_inputs = {"source": jnp.zeros_like(source),
                            "gain": jnp.float32(gain)}
    without = case.run(steps=steps, sharded=True)
    assert not np.allclose(with_source["f"], without["f"],
                           rtol=RTOL, atol=ATOL), (
        "the per-cell boundary input changed nothing on the sharded path")


@given(n_devices=device_counts(),
       cells_per_shard=st.integers(min_value=1, max_value=4),
       steps=st.integers(min_value=1, max_value=3),
       seed=st.integers(0, 2**16))
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_sharded_static_array_survives_partitioning(
        n_devices, cells_per_shard, steps, seed):
    """Each shard gets its own slice of a ``StaticArray(replication="shard")``.

    Stated so that it cannot pass vacuously: the same mask reversed is
    run too, and the property is asserted only where the reversal
    changes the *unsharded* answer -- i.e. where the per-cell values
    genuinely matter.  Whenever they do, both masks must give the
    single-device answer on every mesh, which a wrapper that replicated
    one shard's slice to all of them could not do.

    (The first draft asserted instead that the reversal changes the
    sharded answer, and Hypothesis immediately produced a two-cell field
    whose periodic Laplacian is identically zero: there the mask cannot
    matter and the assertion was about the fixture, not the wrapper.
    The second draft patched the mask onto an already-wrapped node,
    which the ``static_data`` contract forbids -- "stable across calls
    for a given node instance" -- and the wrapper duly kept the array it
    snapshotted at construction.  The mask is a constructor argument
    here for that reason.)
    """
    n_cells = n_devices * cells_per_shard
    rng = np.random.default_rng(seed)
    mask = (0.25 + rng.random(n_cells)).astype(np.float32)

    case = build_stencil(n_devices=n_devices, n_cells=n_cells, mask=mask)
    flipped = build_stencil(n_devices=n_devices, n_cells=n_cells,
                            mask=mask[::-1].copy())

    reference = case.run(steps=steps, sharded=False)
    flipped_reference = flipped.run(steps=steps, sharded=False)
    assume(not np.allclose(reference["f"], flipped_reference["f"],
                           rtol=RTOL, atol=ATOL))

    _assert_agrees(case.run(steps=steps, sharded=True), reference,
                   f"sharded mask on {n_devices} devices")
    _assert_agrees(flipped.run(steps=steps, sharded=True), flipped_reference,
                   f"reversed mask on {n_devices} devices")


@given(label=wrapper_labels(), n_devices=device_counts(),
       cells_per_shard=st.integers(min_value=1, max_value=3),
       steps=st.integers(min_value=1, max_value=3), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_gradient_through_the_sharded_path_matches_the_unsharded_one(
        label, n_devices, cells_per_shard, steps, rate):
    """Sharding is transparent to reverse-mode AD as well as to the forward run.

    Calibration is the reason the parameter contract exists at all, and
    a gradient that is merely finite (what a smoke test checks) can
    still be the gradient of the wrong function.
    """
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=rate,
                                 n_cells=n_devices * cells_per_shard)

    def loss(value, *, sharded):
        node = case.wrapped if sharded else case.inner
        state = node.initial_state()
        for _ in range(steps):
            state = node.update(state, case.boundary_inputs, node.delta_t,
                                params={case.param_name: value})
        return case.objective(state, sharded=sharded)

    x = jnp.float32(rate)
    np.testing.assert_allclose(
        float(loss(x, sharded=True)), float(loss(x, sharded=False)),
        rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        float(jax.grad(lambda v: loss(v, sharded=True))(x)),
        float(jax.grad(lambda v: loss(v, sharded=False))(x)),
        rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# The same statement one level up: a graph, not a node.
# ---------------------------------------------------------------------------


@given(n_devices=device_counts(),
       cells_per_shard=st.integers(min_value=1, max_value=3),
       steps=st.integers(min_value=1, max_value=3), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_graph_whose_node_is_sharded_matches_the_same_graph_unsharded(
        n_devices, cells_per_shard, steps, rate):
    """Wrapping one node of a graph for many devices changes no answer.

    A node-level comparison misses everything ``GraphManager`` does
    around the node: ordering the step, delivering an edge into a
    boundary input, and reading state back out.  Here an ordinary edge
    drives the sharded node's replicated scalar input, which is the
    shape of coupling that W6 showed can vanish on the sharded path.
    """
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.spring import SpringDamperNode

    from tests.cloud.multigpu.property_support import (
        StencilDiffusion1D,
        _mask_for,
        _source_for,
    )
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    n_cells = n_devices * cells_per_shard

    def build(sharded: bool) -> GraphManager:
        diffusion = StencilDiffusion1D(name="diff", n_cells=n_cells, rate=rate,
                                       mask=_mask_for(n_cells))
        gm = GraphManager()
        gm.add_node(SpringDamperNode("drive", 0.05, stiffness=8.0,
                                     initial_position=1.0))
        if sharded:
            mesh = create_device_mesh(shape=(n_devices,))
            gm.add_node(ShardedStencilNode(diffusion, mesh,
                                           axis_map={"devices": 0},
                                           boundary="periodic"))
        else:
            gm.add_node(diffusion)
        gm.add_edge(source="drive", target="diff", source_field="position",
                    target_field="gain")
        gm.add_external_input(target_node="diff", target_field="source",
                              shape=(n_cells,))
        gm.compile()
        return gm

    external = {"diff": {"source": _source_for(n_cells)}}
    results = []
    for sharded in (True, False):
        gm = build(sharded)
        for _ in range(steps):
            gm.step(external)
        results.append({
            node: {field: np.asarray(jax.device_get(value))
                   for field, value in gm.get_node_state(node).items()}
            for node in ("drive", "diff")})

    sharded_run, unsharded_run = results
    for node in unsharded_run:
        _assert_agrees(sharded_run[node], unsharded_run[node], node)
