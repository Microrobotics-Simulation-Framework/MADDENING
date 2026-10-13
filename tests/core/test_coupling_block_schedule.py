"""A coupling group runs as one block, and its back edges are decided over the block order.

When a group is only part of a larger feedback loop -- its members and some
outside nodes form one strongly connected component -- ``topological_sort``
keeps that component in ``add_node`` order, so an outside node could be
scheduled *between* two members.  The step runs the group as one block at
its first member's place, so that node in fact ran after the whole group;
but back edges were decided over the node order, and its read of a later
member was staggered to the previous step although the member had already
run.  The same graph built with the outside nodes added in another order
stepped differently, silently (the round-5 audit, CPL-181; every release
since 0.1.0).  The schedule now places each group's members together at its
first member's place, which is the block order the step runs, and back
edges are decided over it.

What remains order-dependent is inherent: a loop through outside nodes has
to be closed by *some* edge read from the previous step, and which one
follows the order the nodes were added.  ``compile()`` now says so with a
``UserWarning`` naming the group, the outside nodes and the staggered edge.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

_LOOP = "part of a larger feedback loop"


class _Relay(SimulationNode):
    """``x <- a x_pre + b + sum_p g_p u_p`` on a scalar field."""

    def __init__(self, name, gains, b, a=0.0):
        super().__init__(name, 0.1)
        self._g, self._b, self._a = gains, b, a

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def boundary_input_spec(self):
        return {p: BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                     default=jnp.zeros(1, jnp.float32)) for p in self._g}

    def update(self, state, boundary_inputs, dt):
        out = jnp.float32(self._a) * state["x"] + jnp.float32(self._b)
        for p, g in self._g.items():
            out = out + jnp.float32(g) * boundary_inputs[p]
        return {"x": out}


#: Group {A, B} (A <-> B) closed into a larger loop through D and E: B -> D -> E -> A.
_SPEC = {"A": ({"ub": 0.5, "ue": 0.3}, 1.0), "B": ({"ua": 0.6}, 0.2),
         "D": ({"ub": 1.0}, 0.0, 0.5), "E": ({"ud": 1.0}, 0.0, 0.5)}
_EDGES = [("B", "A", "ub"), ("E", "A", "ue"), ("A", "B", "ua"), ("B", "D", "ub"), ("D", "E", "ud")]


def _graph(order, edges=_EDGES, group=("A", "B")):
    gm = GraphManager()
    for nm in order:
        gm.add_node(_Relay(nm, *_SPEC[nm]))
    for src, dst, port in edges:
        gm.add_edge(src, dst, "x", port)
    gm.add_coupling_group(list(group), max_iterations=100, tolerance=1e-7)
    return gm


def _trajectory(gm, steps=3):
    with warnings.catch_warnings():
        # The loop through outside nodes is the fixture's point.
        warnings.filterwarnings("ignore", message=f".*{_LOOP}", category=UserWarning)
        gm.compile()
    out = []
    for _ in range(steps):
        state = gm.step()
        out.append({n: float(state[n]["x"][0]) for n in "ABDE"})
    return out


def test_an_outside_node_added_between_the_members_reads_the_group_this_step():
    """``A, D, B, E`` steps as ``A, B, D, E`` does, and its schedule keeps the group together.

    Before: ``D`` was scheduled between ``A`` and ``B`` and read ``B`` one
    step late -- ``D`` 0.0 after the first step where it is 1.14.
    """
    ref = _trajectory(_graph(["A", "B", "D", "E"]))
    gm = _graph(["A", "D", "B", "E"])
    got = _trajectory(gm)
    assert gm.schedule == ["A", "B", "D", "E"]
    assert got == ref
    assert got[0]["D"] == pytest.approx(1.142857, rel=1e-5)


def test_a_group_inside_a_larger_loop_warns_naming_the_edge_read_late():
    gm = _graph(["E", "A", "B", "D"])
    with pytest.warns(UserWarning, match=_LOOP) as record:
        gm.compile()
    text = next(str(w.message) for w in record if _LOOP in str(w.message))
    assert "['A', 'B']" in text and "['D', 'E']" in text, text
    assert "D.x -> E.ud" in text, text           # the back edge this add order chose


def test_a_group_that_is_its_whole_loop_does_not_warn():
    gm = _graph(["A", "B", "D"], edges=[("B", "A", "ub"), ("A", "B", "ua"), ("B", "D", "ub")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()


def test_two_groups_on_one_loop_each_warn():
    """Each group is a strict subset of the component the two make together."""
    gm = GraphManager()
    for nm, gains in (("A1", {"a2": 0.5, "b2": 0.4}), ("A2", {"a1": 0.5}),
                      ("B1", {"b2": 0.5, "a2": 0.4}), ("B2", {"b1": 0.5})):
        gm.add_node(_Relay(nm, gains, 1.0 if nm == "A1" else 0.0))
    for src, dst, port in (("A2", "A1", "a2"), ("A1", "A2", "a1"), ("B2", "B1", "b2"),
                           ("B1", "B2", "b1"), ("A2", "B1", "a2"), ("B2", "A1", "b2")):
        gm.add_edge(src, dst, "x", port)
    gm.add_coupling_group(["A1", "A2"], max_iterations=100, tolerance=1e-7)
    gm.add_coupling_group(["B1", "B2"], max_iterations=100, tolerance=1e-7)
    with pytest.warns(UserWarning, match=_LOOP) as record:
        gm.compile()
    texts = [str(w.message) for w in record if _LOOP in str(w.message)]
    assert any("['A1', 'A2']" in t for t in texts) and any("['B1', 'B2']" in t for t in texts)
