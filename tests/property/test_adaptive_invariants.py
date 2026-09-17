"""Property-based invariants of the ``AdaptiveNode`` contract.

The concrete subclasses are the test-only toys in
``tests/nodes/adaptive/_toys.py``: a 1-D Helmholtz--Poisson problem in
the sine eigenbasis with top-K selection, and a small dense SPD system
with a non-diagonal operator.

Four invariants, over random parameters and step counts:

1. the state buffer is **fixed size** -- ``c`` and ``mask`` keep shape
   ``(n_max,)`` however many steps run and however the active set moves.
   That is the whole point of masking instead of resizing;
2. ``c`` is **exactly zero off the mask** -- the packing step erases the
   solver's off-set output, so no inactive coefficient can leak into a
   downstream field;
3. inside one active-set region the returned gradient is the derivative
   of the branch the forward pass selected, and a central finite
   difference **taken with a step verified to stay inside the region**
   agrees with it;
4. a config and a checkpoint round trip of a graph containing an
   adaptive node preserve its behaviour exactly.

Deliberately absent: anything about the jump *across* a switch.  The
frozen-set objective is discontinuous there, the returned gradient omits
a first-order term, and that is measured, documented and registered as
``MADD-ANO-003``; ``tests/nodes/adaptive/test_active_set_switch.py``
owns it.  Repeating it here would only re-find a known limitation.

Precision
---------
The two gradient properties run under ``jax_enable_x64`` -- a central
finite difference in float32 measures the cancellation, not the adjoint.
The node is constructed *inside* the enabling block, because
``AdaptiveNode`` resolves its coefficient dtype at construction time.
Everything else runs at the suite's default float32, which is also the
only precision at which an adaptive node composes with the float32 state
of the other nodes (see
``test_an_adaptive_node_under_x64_cannot_drive_a_float32_node``).
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode
from tests.property.invariants import (
    assert_leaf_tree_identical,
    assert_params_identical,
    assert_states_identical,
)

TOYS: dict[str, type] = {
    "PoissonSineTopKNode": PoissonSineTopKNode,
    "MaskedDenseNode": MaskedDenseNode,
}
REGISTRY = {**TOYS, "BallNode": BallNode}


@contextlib.contextmanager
def x64_enabled():
    """``jax_enable_x64`` for the duration of the block.

    A context manager rather than a fixture: a *function*-scoped fixture
    around a ``@given`` test is a Hypothesis health-check failure, and a
    *module*-scoped one would force every property in the file -- the
    graph round trips included -- into float64.
    """
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveRecipe:
    """A toy node's class and constructor arguments.

    The node itself is built by the test, not drawn: ``AdaptiveNode``
    resolves its coefficient dtype and precomputes its basis in
    ``__init__``, so a node built during the draw would be pinned to
    whatever precision was active then.
    """

    type_name: str
    kwargs: tuple[tuple[str, Any], ...]

    def build(self, name: str = "adaptive", timestep: float = 1.0):
        # ``blindness_gate=False``: the cold-start diagnostic costs a
        # full-basis solve plus a gradient per construction, and at random
        # parameters a low capture ratio is an expected consequence of a
        # small budget -- not a defect these properties are about.
        return TOYS[self.type_name](
            name=name, timestep=timestep, blindness_gate=False, **dict(self.kwargs)
        )


_THETA_SINE = st.floats(min_value=0.15, max_value=0.85,
                        allow_nan=False, allow_infinity=False)
_THETA_DENSE = st.floats(min_value=0.05, max_value=2.0,
                         allow_nan=False, allow_infinity=False)


@st.composite
def adaptive_recipes(draw) -> AdaptiveRecipe:
    """Both toys, over their parameters, budgets and basis sizes."""
    if draw(st.booleans()):
        return AdaptiveRecipe("PoissonSineTopKNode", (
            ("theta", draw(_THETA_SINE)),
            ("sigma", draw(st.sampled_from([0.03, 0.05, 0.08]))),
            ("n", draw(st.sampled_from([16, 32]))),
            ("k", draw(st.integers(min_value=2, max_value=8))),
            ("selection", draw(st.sampled_from(["b", "c"]))),
            ("sensor_x", draw(st.sampled_from([0.25, 1.0 / 3.0, 0.7]))),
        ))
    return AdaptiveRecipe("MaskedDenseNode", (
        ("theta", draw(_THETA_DENSE)),
        ("n", draw(st.sampled_from([8, 16]))),
        ("k", draw(st.integers(min_value=2, max_value=6))),
        ("seed", draw(st.integers(min_value=0, max_value=4))),
    ))


# ---------------------------------------------------------------------------
# 1 + 2: the fixed-size buffer
# ---------------------------------------------------------------------------

@given(recipe=adaptive_recipes(), n_steps=st.integers(min_value=1, max_value=5))
def test_the_state_buffer_keeps_its_shape_across_steps(recipe, n_steps):
    """``c`` and ``mask`` are ``(n_max,)`` at every step.  A node that
    resized its buffer with the active set could not be scanned, jitted
    or checkpointed against a fixed structure."""
    node = recipe.build()
    state = node.initial_state()
    expected = {"c": (node.n_max,), "mask": (node.n_max,)}
    for step in range(n_steps):
        assert {k: v.shape for k, v in state.items()} == expected, (step, state)
        assert state["mask"].dtype == jnp.bool_
        state = node.update(state, {}, node.delta_t)
    assert {k: v.shape for k, v in state.items()} == expected


@given(recipe=adaptive_recipes(), n_steps=st.integers(min_value=1, max_value=5))
def test_coefficients_are_exactly_zero_off_the_active_set(recipe, n_steps):
    """Off-mask entries are zeroed by the packing step, exactly -- not
    merely small.  Anything the frozen solve left there (an identity row,
    a Krylov residual) must not reach a downstream field."""
    node = recipe.build()
    state = node.initial_state()
    for _ in range(n_steps + 1):
        mask = np.asarray(state["mask"])
        c = np.asarray(state["c"])
        assert np.all(c[~mask] == 0.0), c[~mask]
        # ...and something is always active, so "all zero" cannot pass
        # this test vacuously.
        assert int(mask.sum()) >= 1
        state = node.update(state, {}, node.delta_t)


# ---------------------------------------------------------------------------
# 3: the in-region gradient
# ---------------------------------------------------------------------------

def _objective(node, state, theta):
    """``J(theta)`` with the active set recomputed -- what a user sees."""
    params = {**node.params, "theta": theta}
    return node.objective(node.update(state, {}, 1.0, params={"theta": theta}), params)


def _branch_objective(node, state, mask, theta):
    """``J(theta)`` with the active set held at ``mask`` -- the smooth
    branch the forward pass selected, and the only valid oracle here
    (spike recommendation 3, ``MADD-ANO-003``)."""
    params = {**node.params, "theta": theta}
    return node.objective(node._solve_and_pack(state, mask, params), params)  # noqa: SLF001


def _mask_at(node, state, theta) -> np.ndarray:
    params = {**node.params, "theta": theta}
    return np.asarray(node.compute_active_set(state, params, prev=state["mask"]))


@given(recipe=adaptive_recipes())
def test_the_gradient_inside_an_active_set_region_matches_a_finite_difference(recipe):
    """Strictly inside a region the frozen-set gradient is exact.

    The step is *verified* to stay inside: the active set at
    ``theta - h`` and ``theta + h`` must equal the one at ``theta``.
    Where no such step exists the draw sits on (or within float noise of)
    a switch, and a finite difference is not a valid oracle there -- that
    is ``MADD-ANO-003``, covered by
    ``tests/nodes/adaptive/test_active_set_switch.py``, so it is assumed
    away rather than asserted on.
    """
    with x64_enabled():
        node = recipe.build()
        theta0 = float(node.params["theta"])
        state = node.initial_state()
        mask = _mask_at(node, state, theta0)

        step = None
        for candidate in (1e-5, 1e-6):
            if (np.array_equal(_mask_at(node, state, theta0 - candidate), mask)
                    and np.array_equal(_mask_at(node, state, theta0 + candidate), mask)):
                step = candidate
                break
        assume(step is not None)
        note(f"theta={theta0!r} step={step!r} |active|={int(mask.sum())}")

        gradient = float(
            jax.grad(lambda t: _objective(node, state, t))(jnp.asarray(theta0))
        )
        frozen = jnp.asarray(mask)

        def branch(theta):
            return float(_branch_objective(node, state, frozen, jnp.asarray(theta)))

        # Richardson extrapolation of the central difference.  A plain
        # central difference is only O(h^2), and the sine toy's source is
        # a Gaussian of width ``sigma`` as small as 0.03, so its third
        # derivative in theta runs to ~1/sigma^3 and the truncation error
        # swamps the adjoint's own accuracy (measured 6e-5 relative at
        # h=1e-4).  Cancelling the h^2 term costs two more evaluations
        # and leaves an oracle good to ~1e-9 relative.
        values = {d: branch(theta0 + d) for d in (step, -step, step / 2, -step / 2)}
        coarse = (values[step] - values[-step]) / (2 * step)
        fine = (values[step / 2] - values[-step / 2]) / step
        difference = (4 * fine - coarse) / 3
        magnitude = max(abs(v) for v in values.values())

    # However exact the adjoint is, a difference of nearly equal values
    # cannot resolve better than ~eps * |J| / h -- and where the branch is
    # flat to float64 that floor is the whole budget, since a relative
    # comparison against a finite difference of exactly zero says nothing
    # about the gradient.
    floor = 128 * np.finfo(np.float64).eps * magnitude / step
    tolerance = 1e-6 * max(abs(difference), abs(gradient)) + floor + 1e-15
    assert abs(gradient - difference) <= tolerance, (gradient, difference, step, tolerance)


@given(recipe=adaptive_recipes())
def test_the_returned_gradient_is_the_selected_branchs_gradient(recipe):
    """The stronger, step-free half of the same statement: the returned
    gradient equals ``jax.grad`` of the frozen branch to machine
    precision, because the mask is committed under ``stop_gradient``."""
    with x64_enabled():
        node = recipe.build()
        theta0 = float(node.params["theta"])
        state = node.initial_state()
        mask = jnp.asarray(_mask_at(node, state, theta0))

        returned = float(
            jax.grad(lambda t: _objective(node, state, t))(jnp.asarray(theta0))
        )
        branch = float(
            jax.grad(lambda t: _branch_objective(node, state, mask, t))(jnp.asarray(theta0))
        )
    assert returned == pytest.approx(branch, rel=1e-11, abs=1e-14)


# ---------------------------------------------------------------------------
# 4: round trips of a graph holding an adaptive node
# ---------------------------------------------------------------------------

def _graph_with(recipe: AdaptiveRecipe) -> GraphManager:
    """The adaptive node driving a ball through its first coefficient.

    ``AdaptiveNode.update`` ignores its boundary inputs, so an adaptive
    node can only be an edge *source*; this is the smallest graph that
    makes it one.
    """
    node = recipe.build()
    gm = GraphManager()
    gm.add_node(node)
    gm.add_node(BallNode("ball", node.delta_t, initial_position=2.0, elasticity=0.5))
    gm.add_edge(node.name, "ball", "c", "table_position", transform="extract_first")
    gm.compile()
    return gm


@given(recipe=adaptive_recipes(), n_steps=st.integers(min_value=1, max_value=3))
def test_a_config_round_trip_of_a_graph_with_an_adaptive_node_preserves_behaviour(
    recipe, n_steps,
):
    """The node's structural settings (``n_max``, carried by the
    subclass's own ``n``; the budget; the selection rule; the diagnostic
    policy) all have to survive ``to_dict`` / ``from_dict`` for the
    reloaded graph to run the same problem."""
    gm = _graph_with(recipe)
    config = gm.to_dict()
    note(f"config: {config}")
    reloaded = GraphManager.from_dict(config, REGISTRY)
    reloaded.compile()

    assert reloaded.get_node("adaptive").n_max == gm.get_node("adaptive").n_max
    assert_params_identical(gm.params, reloaded.params)
    assert_states_identical(gm.run_scan(n_steps), reloaded.run_scan(n_steps),
                            what="trajectory")


@given(recipe=adaptive_recipes(), n_steps=st.integers(min_value=1, max_value=3))
def test_a_checkpoint_of_a_graph_with_an_adaptive_node_restores_it_exactly(
    recipe, n_steps,
):
    """The boolean ``mask`` is state like any other: it has to come back
    from the ``.npz`` with its dtype intact, or the resumed run selects a
    different branch from the one that was saved."""
    # ``run`` (a Python loop over the one jitted step), not ``run_scan``:
    # a fused ``lax.scan`` is a different compiled program per trip
    # count, so splitting a rollout would be compared against a
    # different program -- and it costs a trace per length.  See
    # ``tests/property/test_round_trips.py::test_a_split_rollout_is_step_for_step_identical``.
    gm = _graph_with(recipe)
    gm.run(n_steps)
    with tempfile.TemporaryDirectory() as tmp:
        path = gm.save_state(Path(tmp) / "adaptive.npz")
        resumed = _graph_with(recipe)
        resumed.load_state(path)

        assert_leaf_tree_identical(
            {n: gm.get_node_state(n) for n in gm.node_names},
            {n: resumed.get_node_state(n) for n in resumed.node_names},
            what="restored state",
        )
        gm.run(n_steps)
        resumed.run(n_steps)
        assert_states_identical(
            {n: gm.get_node_state(n) for n in gm.node_names},
            {n: resumed.get_node_state(n) for n in resumed.node_names},
            what="continued trajectory",
        )


# ---------------------------------------------------------------------------
# Pinned: the precision clash this suite ran into
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "AdaptiveNode is the only node whose state dtype follows "
    "jax_enable_x64 (it resolves the canonical float in __init__), while "
    "every other node pins float32 in initial_state().  Under x64 an edge "
    "out of the adaptive node's float64 'c' therefore promotes the "
    "downstream node's float32 state, and run_scan's carry types stop "
    "matching: 'scan body function carry input and carry output must have "
    "equal types'.  Fixing it is an API decision -- either the nodes stop "
    "pinning float32, or AdaptiveNode stops following the flag, or the "
    "graph coerces each node's update output back to its carry dtype -- so "
    "it is pinned here rather than patched.  Found by "
    "tests/property/test_adaptive_invariants.py."
))
def test_an_adaptive_node_under_x64_cannot_drive_a_float32_node():
    recipe = AdaptiveRecipe("MaskedDenseNode", (("theta", 1.0), ("n", 8), ("k", 2)))
    with x64_enabled():
        _graph_with(recipe).run_scan(2)
