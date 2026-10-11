"""An edge applies its interface mapping first and its transform second, on every route.

``GraphManager.add_edge`` documents the order and one function holds it
(``maddening.core.edge._delivered``).  A test of the order needs a
transform that does **not** commute with the mapping: a scaling commutes
with every linear mapping, and an offset commutes with every mapping whose
rows sum to one (a selection, an interpolation).  With such a pair the two
orders deliver the same value, and a test passes on either one.

Every pair here is one on which the orders differ.  Each expected value is
worked by hand beside its test, with what the other order would deliver,
and :func:`test_the_other_order_delivers_another_value_on_every_pair`
holds the premise in numpy.

The routes:

* the query (``resolve_boundary_inputs``) and the step;
* the adaptive step, which resolves its edges in a loop of its own;
* an edge from a node to itself;
* two additive edges into one input, each with its own mapping and its
  own transform;
* a coupling group under both schedules: an edge into it, two inside it
  (one of them additive) and one out of it -- and the same group with one
  member sub-cycled, whose edges are resolved at each of its sub-steps by
  a function of their own (the time-interpolated one);
* a legacy closure of ``interface_mapping`` as the transform of a mapped
  edge: two matrices, applied in the order mapping, closure.

Not here: a transform that changes the shape (a selection) after a
mapping.  ``add_edge`` holds the mapping's ``n_target`` to the shape the
target input declares, whatever the transform does next, so such an edge
is refused before either order could be taken.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.interface_mapping import linear_interpolation_1d
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

DT = 0.01


class Holds(SimulationNode):
    """Keeps the value it was built with: a source whose field is known."""

    def __init__(self, name, x0):
        super().__init__(name, DT)
        self._x0 = np.asarray(x0, np.float32)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def update(self, state, boundary_inputs, dt):
        return state


class Takes(SimulationNode):
    """``x <- u``: after a step its state is what its input delivered."""

    def __init__(self, name, n, x0=None, timestep=DT):
        super().__init__(name, timestep)
        self._x0 = np.zeros(n, np.float32) if x0 is None else np.asarray(x0, np.float32)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=self._x0.shape, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": boundary_inputs["u"]}


def _x(gm, name):
    return np.asarray(gm.get_node_state(name)["x"])


def _plus_one(v):
    return v + 1.0


def _at_most_four(v):
    return jnp.clip(v, 0.0, 4.0)


def _at_most_three(v):
    return jnp.clip(v, 0.0, 3.0)


#: The source field of most tests, and a mapping onto two entries whose
#: rows sum to 2: ``H @ S = [1 + 2, 2 + 3] = [3, 5]``.
S = np.array([1.0, 2.0, 3.0], np.float32)
H = np.array([[1, 1, 0], [0, 1, 1]], np.float32)

#: ``{name: (transform, what the edge delivers, what the other order would)}``
#: for ``S`` through ``H``:
#:
#: * offset: ``[3, 5] + 1 = [4, 6]``; the other order is
#:   ``H @ (S + 1) = H @ [2, 3, 4] = [5, 7]`` (each row adds the offset
#:   twice);
#: * clip: ``clip([3, 5], 0, 4) = [3, 4]``; the other order is
#:   ``H @ clip(S, 0, 4) = H @ S = [3, 5]`` (no entry of ``S`` is above 4,
#:   one of their sums is).
ONE_EDGE = {
    "offset": (_plus_one, [4.0, 6.0], [5.0, 7.0]),
    "clip": (_at_most_four, [3.0, 4.0], [3.0, 5.0]),
}

#: A second mapping from ``S``, for the additive pair:
#: ``H2 @ S = [1 + 3, 2 * 1] = [4, 2]``, clipped to at most 3: ``[3, 2]``.
#: The other order is ``H2 @ clip(S, 0, 3) = H2 @ S = [4, 2]``.
H2 = np.array([[1, 0, 1], [2, 0, 0]], np.float32)

#: An edge from a node to itself: ``X0 = [1, 2]`` through
#: ``LOOP @ X0 = [2 * 2, 1 + 2] = [4, 3]``, then the offset: ``[5, 4]``.
#: The other order is ``LOOP @ (X0 + 1) = LOOP @ [2, 3] = [6, 5]``.
X0 = np.array([1.0, 2.0], np.float32)
LOOP = np.array([[0, 2], [1, 1]], np.float32)

#: The coupling group ``{a, b}`` between a source ``s`` and a reader ``r``
#: (every node but ``s`` takes its input as its state):
#:
#: * ``s -> a``: ``H``, then the offset (into the group, additive);
#: * ``b -> a``: ``BACK``, no transform (inside the group, additive);
#: * ``a -> b``: ``FORTH``, then the offset (inside the group);
#: * ``b -> r``: ``OUT``, then the offset (out of the group).
#:
#: The fixed point, by hand.  ``a = (H @ S + 1) + BACK @ b = [4 + b1 / 2, 6]``
#: and ``b = FORTH @ a + 1 = [a0 + a1 + 1, 2 a1 + 1]``.  So ``a1 = 6``,
#: ``b1 = 13``, ``a0 = 4 + 6.5 = 10.5``, ``b0 = 10.5 + 6 + 1 = 17.5``:
#: ``a = [10.5, 6]``, ``b = [17.5, 13]``, and
#: ``r = OUT @ b + 1 = [17.5 + 13 + 1, 17.5 + 1] = [31.5, 18.5]``.
#: ``a1`` depends on nothing in the group, ``b1`` on ``a1``, ``a0`` on
#: ``b1`` and ``b0`` on ``a``: the loop ends, so plain passes reach the
#: fixed point exactly (every number above is a float32).  A member taken
#: in two sub-steps ends on the same values: its last sub-step reads the
#: other member at the end of the step, and it keeps nothing but its input.
#:
#: The other order, by hand.  ``a = H @ (S + 1) + BACK @ b = [5 + b1 / 2, 7]``
#: and ``b = FORTH @ (a + 1) = [a0 + a1 + 2, 2 a1 + 2]``.  So ``a1 = 7``,
#: ``b1 = 16``, ``a0 = 13``, ``b0 = 22``: ``a = [13, 7]``, ``b = [22, 16]``,
#: and ``r = OUT @ (b + 1) = [23 + 17, 23] = [40, 23]``.
BACK = np.array([[0, 0.5], [0, 0]], np.float32)
FORTH = np.array([[1, 1], [0, 2]], np.float32)
OUT = np.array([[1, 1], [1, 0]], np.float32)
GROUP = {"a": [10.5, 6.0], "b": [17.5, 13.0], "r": [31.5, 18.5]}
GROUP_OTHER_ORDER = {"a": [13.0, 7.0], "b": [22.0, 16.0], "r": [40.0, 23.0]}

#: A legacy closure as the transform: linear interpolation from the points
#: ``[0, 1]`` onto ``[0.25, 0.75]``, the matrix ``[[.75, .25], [.25, .75]]``.
#: After the mapping ``FORTH`` of ``V = [4, 8]``: ``FORTH @ V = [12, 16]``,
#: interpolated: ``[.75 * 12 + .25 * 16, .25 * 12 + .75 * 16] = [13, 15]``.
#: The other order interpolates first (``[5, 7]``), then maps:
#: ``FORTH @ [5, 7] = [12, 14]``.
V = np.array([4.0, 8.0], np.float32)
INTERPOLATION = np.array([[0.75, 0.25], [0.25, 0.75]], np.float32)


def _one_edge(transform):
    gm = GraphManager()
    gm.add_node(Holds("s", S))
    gm.add_node(Takes("t", 2))
    gm.add_edge("s", "t", "x", "u", mapping=matrix_mapping(H), transform=transform)
    gm.compile()
    return gm


def test_the_other_order_delivers_another_value_on_every_pair():
    """The premise, in numpy: on each pair used here the transform applied
    before the mapping gives the value written beside it, which is not the
    value the edge must deliver."""
    np_transforms = {"offset": lambda v: v + 1.0, "clip": lambda v: np.clip(v, 0.0, 4.0)}
    for name, (_transform, delivered, other) in ONE_EDGE.items():
        t = np_transforms[name]
        np.testing.assert_array_equal(t(H @ S), delivered)
        np.testing.assert_array_equal(H @ t(S), other)
        assert delivered != other
    np.testing.assert_array_equal(np.clip(H2 @ S, 0.0, 3.0), [3.0, 2.0])
    np.testing.assert_array_equal(H2 @ np.clip(S, 0.0, 3.0), [4.0, 2.0])
    np.testing.assert_array_equal(LOOP @ X0 + 1.0, [5.0, 4.0])
    np.testing.assert_array_equal(LOOP @ (X0 + 1.0), [6.0, 5.0])
    for order, want in ((lambda m, v: m @ v + 1.0, GROUP),
                        (lambda m, v: m @ (v + 1.0), GROUP_OTHER_ORDER)):
        a, b = np.asarray(want["a"]), np.asarray(want["b"])
        np.testing.assert_array_equal(order(H, S) + BACK @ b, a)
        np.testing.assert_array_equal(order(FORTH, a), b)
        np.testing.assert_array_equal(order(OUT, b), want["r"])
    assert all(GROUP[k] != GROUP_OTHER_ORDER[k] for k in GROUP)
    np.testing.assert_array_equal(INTERPOLATION @ (FORTH @ V), [13.0, 15.0])
    np.testing.assert_array_equal(FORTH @ (INTERPOLATION @ V), [12.0, 14.0])


@pytest.mark.parametrize("pair", sorted(ONE_EDGE))
def test_the_query_and_the_step_deliver_the_mapped_value_transformed(pair):
    """What ``resolve_boundary_inputs`` answers and what the step hands the
    node are both the transform of the mapped value."""
    transform, delivered, _other = ONE_EDGE[pair]
    gm = _one_edge(transform)
    np.testing.assert_array_equal(np.asarray(gm.resolve_boundary_inputs("t")["u"]), delivered)
    gm.step()
    np.testing.assert_array_equal(_x(gm, "t"), delivered)


@pytest.mark.parametrize("pair", sorted(ONE_EDGE))
def test_the_adaptive_step_delivers_the_mapped_value_transformed(pair):
    """``run_adaptive`` resolves its edges itself, by the same rule."""
    transform, delivered, _other = ONE_EDGE[pair]
    gm = _one_edge(transform)
    _, info = gm.run_adaptive(2 * DT, dt_initial=DT, dt_max=DT)
    assert info["n_steps"] >= 1
    np.testing.assert_array_equal(_x(gm, "t"), delivered)


def test_an_edge_from_a_node_to_itself_maps_its_own_value_then_transforms_it():
    gm = GraphManager()
    gm.add_node(Takes("a", 2, x0=X0))
    gm.add_edge("a", "a", "x", "u", mapping=matrix_mapping(LOOP), transform=_plus_one)
    gm.compile()
    gm.step()
    np.testing.assert_array_equal(_x(gm, "a"), [5.0, 4.0])


def test_additive_edges_each_apply_their_own_mapping_then_their_own_transform():
    """Two edges into one input: ``[4, 6]`` (``H``, then the offset) plus
    ``[3, 2]`` (``H2``, then the clip) is ``[7, 8]``.  The other order
    would sum ``[5, 7]`` and ``[4, 2]``: ``[9, 9]``."""
    gm = GraphManager()
    gm.add_node(Holds("s", S))
    gm.add_node(Takes("t", 2))
    gm.add_edge("s", "t", "x", "u", mapping=matrix_mapping(H), transform=_plus_one,
                additive=True)
    gm.add_edge("s", "t", "x", "u", mapping=matrix_mapping(H2), transform=_at_most_three,
                additive=True)
    gm.compile()
    np.testing.assert_array_equal(np.asarray(gm.resolve_boundary_inputs("t")["u"]), [7.0, 8.0])
    gm.step()
    np.testing.assert_array_equal(_x(gm, "t"), [7.0, 8.0])


@pytest.mark.parametrize("sub_cycled", [None, "a", "b"])
@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
def test_edges_into_inside_and_out_of_a_coupling_group_map_then_transform(schedule, sub_cycled):
    """The group's fixed point and the reader behind it are the ones worked
    by hand above for the order mapping, transform -- under both schedules,
    and with either member taken in two sub-steps (``subcycling=True``),
    whose inputs the group resolves with its time-interpolating function
    and not with the one a plain pass uses."""
    half = {sub_cycled: DT / 2}
    gm = GraphManager()
    gm.add_node(Holds("s", S))
    gm.add_node(Takes("a", 2, timestep=half.get("a", DT)))
    gm.add_node(Takes("b", 2, timestep=half.get("b", DT)))
    gm.add_node(Takes("r", 2))
    gm.add_edge("s", "a", "x", "u", mapping=matrix_mapping(H), transform=_plus_one,
                additive=True)
    gm.add_edge("b", "a", "x", "u", mapping=matrix_mapping(BACK), additive=True)
    gm.add_edge("a", "b", "x", "u", mapping=matrix_mapping(FORTH), transform=_plus_one)
    gm.add_edge("b", "r", "x", "u", mapping=matrix_mapping(OUT), transform=_plus_one)
    gm.add_coupling_group(["a", "b"], max_iterations=12, tolerance=1e-6,
                          iteration_mode=schedule, acceleration="none",
                          subcycling=sub_cycled is not None)
    gm.compile()
    gm.step()
    for name, want in GROUP.items():
        np.testing.assert_array_equal(_x(gm, name), want, err_msg=name)


def test_a_legacy_closure_as_the_transform_is_applied_after_the_mapping():
    """Two matrices on one edge, the ``mapping=`` one first."""
    closure = linear_interpolation_1d(np.array([0.0, 1.0]), np.array([0.25, 0.75]))
    np.testing.assert_allclose(np.asarray(closure(jnp.eye(2, dtype=jnp.float32))).T,
                               INTERPOLATION.T, err_msg="premise: the closure's matrix")
    gm = GraphManager()
    gm.add_node(Holds("s", V))
    gm.add_node(Takes("t", 2))
    gm.add_edge("s", "t", "x", "u", mapping=matrix_mapping(FORTH), transform=closure)
    gm.compile()
    np.testing.assert_array_equal(np.asarray(gm.resolve_boundary_inputs("t")["u"]), [13.0, 15.0])
    gm.step()
    np.testing.assert_array_equal(_x(gm, "t"), [13.0, 15.0])
