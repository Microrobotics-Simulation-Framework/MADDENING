"""``coupling_diagnostics()`` judges the last step under the group that took it.

The verdict keys (``converged``, ``error_estimate``) are re-derived on the
host from the ``_meta`` residual and amplification slots.  Replacing a
group by another over the same nodes -- ``remove_coupling_group`` then
``add_coupling_group``, to tighten a tolerance -- keeps the slot names, and
the report used to re-derive the old step's verdict under the *new*
group's criterion: ``converged=False`` three passes into a fifty-pass
budget, the combination the docstring says cannot happen.  Now the report
keeps the verdict the step reached until the graph is recompiled, and a
recompile restarts a replaced group's report slots, so it has no entry
until it has stepped -- as after ``reset_state()``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

KEY = "a+b"


def _pair(tolerance=1e-3):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=2000.0, damping=5.0,
                                 initial_position=-1.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=2000.0, damping=5.0,
                                 initial_position=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=50, tolerance=tolerance)
    gm.compile()
    return gm


def _as_text(report):
    """The report with every value as its ``repr`` (NaN compares equal to NaN)."""
    return {k: repr(v) for k, v in dict(report).items()}


def _regroup(gm, tolerance):
    gm.remove_coupling_group(["a", "b"])
    gm.add_coupling_group(["a", "b"], max_iterations=50, tolerance=tolerance)


@pytest.fixture
def stepped():
    gm = _pair()
    gm.step()
    report = gm.coupling_diagnostics()[KEY]
    # The precondition: a converged step whose estimate the tighter
    # tolerance below would reject.
    assert report["converged"] is True and report["iterations"] < 50
    assert report["error_estimate"] > 1e-7, report
    return gm, _as_text(report)


def test_a_replacement_group_does_not_rejudge_the_last_step(stepped):
    gm, report = stepped
    _regroup(gm, 1e-7)
    again = gm.coupling_diagnostics()[KEY]
    assert again["converged"] is True
    assert _as_text(again) == report


def test_recompiling_a_replaced_group_restarts_its_report(stepped):
    gm, _ = stepped
    _regroup(gm, 1e-7)
    gm.compile()
    assert KEY not in gm.coupling_diagnostics()
    gm.step()
    report = dict(gm.coupling_diagnostics()[KEY])
    # Judged under the group that took this step.
    assert report["converged"] is (report["error_estimate"] <= 1e-7), report


def test_recompiling_an_unchanged_group_keeps_its_report(stepped):
    """A recompile for another reason (an edit elsewhere) is not a regroup."""
    gm, report = stepped
    gm.add_node(SpringDamperNode("c", 0.01, initial_position=0.0))
    gm.compile()
    assert _as_text(gm.coupling_diagnostics()[KEY]) == report


def test_an_equal_replacement_keeps_its_report(stepped):
    gm, report = stepped
    _regroup(gm, 1e-3)
    gm.compile()
    assert _as_text(gm.coupling_diagnostics()[KEY]) == report
