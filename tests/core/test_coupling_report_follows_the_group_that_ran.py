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

# ``_pair`` is past MADD-ANO-098's limit (c < k*dt: its centre of mass
# grows by 1.19 a step), so every compile warns.  These tests are about
# which group a coupling report describes, not the springs' stability.
pytestmark = pytest.mark.filterwarnings("ignore:.*MADD-ANO-098:UserWarning")


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


# ---------------------------------------------------------------------------
# The report's float floor is the step's, not the live graph's
# ---------------------------------------------------------------------------

import jax.numpy as jnp  # noqa: E402

from maddening.core.node import BoundaryInputSpec, SimulationNode  # noqa: E402


class _Relay(SimulationNode):
    def __init__(self, name, g, c, x0, evaluations=1):
        super().__init__(name, 1.0, g=jnp.float32(g), c=jnp.float32(c))
        self._x0, self._ev = x0, evaluations

    def initial_state(self):
        return {"x": jnp.asarray([self._x0], jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["c"]}

    def update_evaluations(self):
        return self._ev


_NAMES = ["n0", "n1", "n2"]
_G, _C = 0.99 ** (1 / 3), 1.0 - 0.99 ** (1 / 3)
_KEY = "n0+n1+n2"


def _stalled_ring(diagnostics=True):
    gm = GraphManager()
    for nm in _NAMES:
        gm.add_node(_Relay(nm, _G, _C, 0.999))
    for k in range(3):
        gm.add_edge(_NAMES[k - 1], _NAMES[k], "x", "u")
    gm.add_coupling_group(_NAMES, max_iterations=2000, tolerance=1e-12,
                          diagnostics=diagnostics)
    gm.compile()
    gm.step()
    return gm


@pytest.mark.parametrize("diagnostics", [True, False])
def test_rebuilding_a_member_before_the_next_step_leaves_the_report_alone(diagnostics):
    """The documented remove-and-re-add recipe, no step since: the report is unchanged.

    The floor the report adds (``spectral_error_bound`` at a stalled
    residual, ``precision_limited``) was re-derived from the live graph:
    the rebuilt node declaring 50 evaluations moved the bound of the step
    that had already run 17x.  It is now taken from what ``compile()``
    built the step from, and -- with ``diagnostics=True`` -- from the count
    the step measured.
    """
    gm = _stalled_ring(diagnostics)
    before = dict(gm.coupling_diagnostics()[_KEY])
    assert before["precision_limited"], "fixture premise: the residual is the floor"
    gm.remove_node("n1")
    gm.add_node(_Relay("n1", _G, _C, 0.999, evaluations=50))
    gm.add_edge("n0", "n1", "x", "u")
    gm.add_edge("n1", "n2", "x", "u")
    after = dict(gm.coupling_diagnostics()[_KEY])
    for key in before:
        a, b = before[key], after[key]
        assert a == b or (a != a and b != b), (key, a, b)


def test_a_member_removed_since_the_step_takes_the_report_with_it():
    """No ``KeyError``: the group has no entry, and the coupling report says why."""
    gm = _stalled_ring()
    gm.remove_node("n1")
    assert _KEY not in gm.coupling_diagnostics()
    row = next(r for r in gm.coupling_report() if r["group"] == _KEY)
    assert any("removed" in flag for flag in row["flags"]), row
