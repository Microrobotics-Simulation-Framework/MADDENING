"""Which node entry points a calibrated ``params`` value reaches.

``update`` and ``compute_interface_correction`` take ``params`` and
apply the ``{**self.params, **params}`` rule.  ``derivatives`` and
``implicit_residual`` take no ``params`` at all, and neither does
``integrate_node``, which hands ``node.derivatives`` straight to a
stepper.  A value produced by ``maddening.sysid.fit`` therefore changes
one half of a node's behaviour and not the other, silently.

That divergence is registered as ``MADD-ANO-018`` and is **not** fixed
here: threading ``params`` through is 0.5.0 work.  This module pins the
current behaviour so the anomaly cannot be resolved by accident and
cannot drift -- when the fix lands, these tests fail and say what to do,
which is the point of pinning a defect rather than only describing it.

The divergence dates from 0.4.0, not from 0.1.0: before the graph
parameter pytree there was no way to inject a value, and ``update`` and
``derivatives`` both read ``self.params``, so the two agreed.
"""

import jax.numpy as jnp
import pytest

from maddening.core.node import SimulationNode
from maddening.core.simulation.implicit import implicit_euler_step
from maddening.core.simulation.integrators import (
    euler_step,
    integrate_node,
)
from maddening.nodes.spring import SpringDamperNode


#: Constructor stiffness, and the value a calibration produces.  A
#: factor of four apart so no tolerance can confuse them, and both
#: chosen with the state below so the two paths give round numbers.
K_CONSTRUCTED = 100.0
K_CALIBRATED = 400.0

DT = 0.01

#: Displaced one unit from the rest length with zero velocity.  The
#: displacement is what makes the stiffness observable at all: a spring
#: started *at* its rest length has a zero one-step residual whatever
#: ``k`` is, and every assertion below would hold against a node that
#: ignored both the constructor value and the injected one.
STATE = {"position": jnp.asarray(2.0), "velocity": jnp.asarray(0.0)}
REST_LENGTH = 1.0


def _spring() -> SpringDamperNode:
    return SpringDamperNode(
        "s", timestep=DT, stiffness=K_CONSTRUCTED, damping=1.0, mass=1.0,
        rest_length=REST_LENGTH,
    )


def _velocity(state: dict) -> float:
    return float(state["velocity"])


CALIBRATED = {"stiffness": jnp.asarray(K_CALIBRATED)}


# ------------------------------------------------------------------
# The fixture must be able to express the defect
# ------------------------------------------------------------------

def test_the_spring_fixture_responds_to_stiffness_at_all():
    """A node started at its rest length could not show this.

    Guards every test below.  ``update`` is the path that *does* honour
    ``params``, so if injecting a different stiffness does not move its
    answer, the fixture is inert and the comparisons that follow prove
    nothing about where ``params`` goes.
    """
    node = _spring()
    baseline = node.update(STATE, {}, DT)
    injected = node.update(STATE, {}, DT, params=CALIBRATED)
    assert _velocity(injected) != _velocity(baseline), (
        "injecting a four-times stiffness did not change update()'s "
        "answer, so this fixture cannot express where params reaches"
    )
    assert _velocity(baseline) != 0.0, (
        "the spring is at its rest length with zero velocity; its "
        "one-step residual is zero whatever the stiffness"
    )


# ------------------------------------------------------------------
# Where an injected params value does reach
# ------------------------------------------------------------------

def test_update_honours_an_injected_params_value():
    """The reference path: ``{**self.params, **params}``."""
    node = _spring()
    assert _velocity(node.update(STATE, {}, DT, params=CALIBRATED)) == pytest.approx(
        -4.0, rel=1e-6,
    )
    assert _velocity(node.update(STATE, {}, DT)) == pytest.approx(-1.0, rel=1e-6)


def test_compute_interface_correction_takes_params_at_all():
    """The base signature accepts it, which is the stated contract.

    ``SimulationNode.compute_interface_correction``'s docstring says a
    node taking ``params`` in ``update`` must take it here too.  This
    pins that the base signature really does offer the argument, so the
    inconsistency the tests below record is between *siblings in one
    class* and not a misreading of one method.
    """
    node = _spring()
    node.compute_interface_correction(STATE, {}, DT, params=CALIBRATED)


# ------------------------------------------------------------------
# MADD-ANO-018: where it does not, and cannot
# ------------------------------------------------------------------

def test_derivatives_has_no_params_argument():
    """Not "does not use it by default" -- there is nowhere to put it."""
    with pytest.raises(TypeError):
        _spring().derivatives(STATE, {}, params=CALIBRATED)  # type: ignore[call-arg]


def test_implicit_residual_has_no_params_argument():
    with pytest.raises(TypeError):
        _spring().implicit_residual(  # type: ignore[call-arg]
            STATE, STATE, {}, DT, params=CALIBRATED,
        )


def test_integrate_node_has_no_params_argument():
    with pytest.raises(TypeError):
        integrate_node(  # type: ignore[call-arg]
            _spring(), STATE, {}, DT, method="euler", params=CALIBRATED,
        )


@pytest.mark.parametrize(
    "name, run",
    [
        ("derivatives via euler_step",
         lambda n: euler_step(n.derivatives, STATE, {}, DT)),
        ("integrate_node(method='euler')",
         lambda n: integrate_node(n, STATE, {}, DT, method="euler")),
        ("implicit_euler_step",
         lambda n: implicit_euler_step(n.implicit_residual, STATE, {}, DT)[0]),
    ],
)
def test_solver_path_uses_the_constructor_stiffness_not_a_calibrated_one(name, run):
    """MADD-ANO-018: the calibration is discarded, silently.

    ``update`` reaches -4.0 with the calibrated stiffness; each of these
    reaches roughly -1.0, the constructor's answer, and no warning is
    raised.  ``implicit_euler_step`` is compared loosely because backward
    Euler does not land on the explicit value exactly -- it is the
    factor of four that matters, not the third digit.
    """
    node = _spring()
    got = _velocity(run(node))

    constructed = _velocity(node.update(STATE, {}, DT))
    calibrated = _velocity(node.update(STATE, {}, DT, params=CALIBRATED))

    # The actionable assertion first, so a future fix reports what to do
    # rather than an unexplained numeric mismatch.
    assert abs(got - calibrated) > abs(got - constructed), (
        f"{name} moved towards the calibrated stiffness ({got}, against "
        f"{constructed} constructed and {calibrated} calibrated).  If "
        f"params now reaches this path, MADD-ANO-018 is resolved: update "
        f"its resolution_status in docs/validation/known_anomalies.yaml, "
        f"drop the limitation paragraphs from the three docstrings, and "
        f"replace this module's expectation."
    )
    assert got == pytest.approx(constructed, rel=0.05), (
        f"{name} no longer matches update() without params ({got} vs "
        f"{constructed})"
    )


def test_the_two_paths_of_one_node_disagree_by_the_whole_calibration():
    """The user-visible shape of the defect, in one assertion.

    One node object, one state, one dt, two documented public entry
    points, answers a factor of four apart -- the entire calibration.
    """
    node = _spring()
    via_update = _velocity(node.update(STATE, {}, DT, params=CALIBRATED))
    via_integrator = _velocity(
        integrate_node(node, STATE, {}, DT, method="euler")
    )
    assert via_update == pytest.approx(-4.0, rel=1e-6)
    assert via_integrator == pytest.approx(-1.0, rel=1e-6)
    assert via_update / via_integrator == pytest.approx(
        K_CALIBRATED / K_CONSTRUCTED, rel=1e-6,
    )


# ------------------------------------------------------------------
# The limitation is documented, not only pinned
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, obj",
    [
        ("SimulationNode.derivatives", SimulationNode.derivatives),
        ("SimulationNode.implicit_residual", SimulationNode.implicit_residual),
        ("integrate_node", integrate_node),
    ],
)
def test_the_params_limitation_is_documented_where_a_caller_would_look(label, obj):
    """Each of the three names the anomaly in its own docstring.

    A limitation recorded only in the registry is one a caller reading
    the API never meets.  Deleting these paragraphs would otherwise be
    silent.
    """
    doc = obj.__doc__ or ""
    assert "MADD-ANO-018" in doc, (
        f"{label} does not reach an injected params value (pinned above) "
        f"but its docstring does not say so or cite MADD-ANO-018"
    )
