"""The predictor history ``compile()`` and ``reset_state()`` seed is in the step's order.

The step flattens a group's floating fields node by node in the group's
schedule order.  The seeds were flattened in ``frozenset`` order, which
follows the per-process string hash, so the same graph seeded its history
in a different order from one run to the next -- harmless only because a
seed is never extrapolated from (the predictor waits for two stored
states), but ``_meta`` differed between identical runs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from maddening.core.coupling.acceleration import flatten_coupled_state, float_fields_of
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

NAMES = ("n0", "n1", "n2", "n3", "n4")
KEY = "+".join(sorted(NAMES))


def _ring():
    """Five springs in a ring: a ``frozenset`` of five names iterates in the
    schedule's order in one process of 120."""
    gm = GraphManager()
    for i, name in enumerate(NAMES):
        gm.add_node(SpringDamperNode(name, 0.01, stiffness=5.0, damping=1.0,
                                     initial_position=float(i)))
    for i, name in enumerate(NAMES):
        gm.add_edge(name, NAMES[(i + 1) % len(NAMES)], "position", "anchor_position")
    gm.add_coupling_group(list(NAMES), predictor="quadratic", max_iterations=6,
                          tolerance=1e-5)
    gm.compile()
    return gm


def _in_schedule_order(gm):
    names = [n for n in gm._schedule if n in NAMES]
    return np.asarray(flatten_coupled_state(gm._state, names,
                                            fields=float_fields_of(gm._state, names)))


def test_the_compile_time_seed_is_in_the_schedule_order():
    gm = _ring()
    want = _in_schedule_order(gm)
    for i in range(3):
        np.testing.assert_array_equal(np.asarray(gm._state["_meta"][f"coupling_{KEY}_pred_{i}"]),
                                      want)


def test_the_reset_seed_is_in_the_schedule_order():
    gm = _ring()
    gm.step()
    gm.reset_state()
    want = _in_schedule_order(gm)
    for i in range(3):
        np.testing.assert_array_equal(np.asarray(gm._state["_meta"][f"coupling_{KEY}_pred_{i}"]),
                                      want)
