"""Which node entry points a calibrated ``params`` value reaches: all of them.

``update``, ``compute_interface_correction``, ``compute_boundary_fluxes``,
``derivatives`` and ``implicit_residual`` take ``params`` and apply the
``{**self.params, **params}`` rule; ``integrate_node`` and
``implicit_euler_step`` take it and forward it.  A value produced by
``maddening.sysid.fit`` therefore moves every path of a node by the same
amount, and the two documented public paths of one node object agree.

That was not true at 0.4.0.dev0.  ``derivatives`` and ``implicit_residual``
had no ``params`` at all, so ``integrate_node`` and ``implicit_euler_step``
integrated the constructor's constants while ``update`` used the
calibrated ones -- silently, and by the whole calibration.  That was
registered as ``MADD-ANO-018`` and resolved in 0.4.0.  This module used to
pin the divergence so it could not be "fixed" by accident; it now pins
the agreement, and the refusal that replaced the silence: a non-empty
``params`` for an override that cannot take it is a ``ValueError`` naming
the class and the method, never a quiet fall back to the constructor's
values.  Passing nothing still integrates the constructor's values, which
is what keeps an override declared without the keyword working.
"""

import functools
import importlib
import inspect
import pkgutil

import jax
import jax.numpy as jnp
import pytest

import maddening.nodes
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


def _spring(cls=SpringDamperNode, k: float = K_CONSTRUCTED) -> SpringDamperNode:
    return cls(
        "s", timestep=DT, stiffness=k, damping=1.0, mass=1.0,
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

    Guards every test below.  ``update`` is the reference path, so if
    injecting a different stiffness does not move its answer, the
    fixture is inert and the agreements that follow prove nothing about
    where ``params`` goes.
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


def test_update_honours_an_injected_params_value():
    """The reference path: ``{**self.params, **params}``."""
    node = _spring()
    assert _velocity(node.update(STATE, {}, DT, params=CALIBRATED)) == pytest.approx(
        -4.0, rel=1e-6,
    )
    assert _velocity(node.update(STATE, {}, DT)) == pytest.approx(-1.0, rel=1e-6)


# ------------------------------------------------------------------
# Every solver path now reaches the calibrated value
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, run",
    [
        ("derivatives(params=)",
         lambda n: {"velocity": n.derivatives(STATE, {}, params=CALIBRATED)["velocity"] * DT}),
        ("euler_step(partial(derivatives, params=))",
         lambda n: euler_step(functools.partial(n.derivatives, params=CALIBRATED), STATE, {}, DT)),
        ("integrate_node(method='euler', params=)",
         lambda n: integrate_node(n, STATE, {}, DT, method="euler", params=CALIBRATED)),
    ],
)
def test_explicit_paths_reach_the_calibrated_stiffness_exactly(name, run):
    """One forward-Euler step from rest: ``v = -k * dt`` with ``k = 400``."""
    got = _velocity(run(_spring()))
    assert got == pytest.approx(-4.0, rel=1e-6), name


@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
def test_integrate_node_moves_by_the_whole_calibration_for_every_method(method):
    """``rk4`` and ``heun`` do not land on ``-k*dt`` exactly; the
    calibrated answer must still be the answer of a node *built* with
    the calibrated stiffness, to round-off."""
    via_params = integrate_node(_spring(), STATE, {}, DT, method=method, params=CALIBRATED)
    via_constructor = integrate_node(_spring(k=K_CALIBRATED), STATE, {}, DT, method=method)
    assert _velocity(via_params) == pytest.approx(_velocity(via_constructor), rel=1e-6)
    assert _velocity(via_params) != pytest.approx(
        _velocity(integrate_node(_spring(), STATE, {}, DT, method=method)), rel=0.1,
    )


@pytest.mark.parametrize(
    "name, run",
    [
        ("implicit_euler_step(params=)",
         lambda n: implicit_euler_step(n.implicit_residual, STATE, {}, DT, params=CALIBRATED)[0]),
        ("implicit_euler_step(partial(implicit_residual, params=))",
         lambda n: implicit_euler_step(
             functools.partial(n.implicit_residual, params=CALIBRATED), STATE, {}, DT,
         )[0]),
    ],
)
def test_the_implicit_solve_reaches_the_calibrated_stiffness(name, run):
    """Backward Euler with the injected ``k`` equals backward Euler on a
    node constructed with that ``k``.  The keyword and the by-hand
    ``functools.partial`` are the same binding."""
    got = _velocity(run(_spring()))
    reference = _velocity(
        implicit_euler_step(_spring(k=K_CALIBRATED).implicit_residual, STATE, {}, DT)[0]
    )
    assert got == pytest.approx(reference, rel=1e-6), name
    assert got != pytest.approx(
        _velocity(implicit_euler_step(_spring().implicit_residual, STATE, {}, DT)[0]), rel=0.1,
    )


def test_implicit_residual_forwards_params_to_derivatives():
    """``R = x_new - x_old - dt * f(x_new; params)`` with the injected ``k``."""
    node = _spring()
    residual = node.implicit_residual(STATE, STATE, {}, DT, params=CALIBRATED)
    derivs = node.derivatives(STATE, {}, params=CALIBRATED)
    assert float(residual["velocity"]) == pytest.approx(-DT * float(derivs["velocity"]))
    assert float(residual["velocity"]) == pytest.approx(4.0, rel=1e-6)


def test_the_two_paths_of_one_node_agree_on_the_same_params():
    """The user-visible shape of the fix, in one assertion.

    One node object, one state, one dt, two documented public entry
    points, one ``params``: the same answer.
    """
    node = _spring()
    via_update = _velocity(node.update(STATE, {}, DT, params=CALIBRATED))
    via_integrator = _velocity(
        integrate_node(node, STATE, {}, DT, method="euler", params=CALIBRATED)
    )
    assert via_update == pytest.approx(-4.0, rel=1e-6)
    assert via_integrator == pytest.approx(via_update, rel=1e-6)


def test_no_params_still_integrates_the_constructor_constants():
    """``None`` and ``{}`` both mean "nothing to inject" -- the pre-0.4.0
    call, unchanged.  An empty dict is not a refusal."""
    node = _spring()
    assert _velocity(integrate_node(node, STATE, {}, DT, method="euler")) == pytest.approx(-1.0, rel=1e-6)
    assert _velocity(integrate_node(node, STATE, {}, DT, method="euler", params={})) == pytest.approx(-1.0, rel=1e-6)
    assert _velocity(implicit_euler_step(node.implicit_residual, STATE, {}, DT, params={})[0]) == pytest.approx(
        _velocity(implicit_euler_step(node.implicit_residual, STATE, {}, DT)[0]), rel=1e-6,
    )


def test_the_gradient_with_respect_to_the_injected_stiffness_reaches_both_solvers():
    """What a calibration needs: ``d(step)/dk`` through the integrators
    is finite and non-zero, not the identically-zero gradient a dropped
    ``params`` produces."""
    node = _spring()

    def explicit(p):
        return integrate_node(node, STATE, {}, DT, method="rk4", params=p)["velocity"]

    def implicit(p):
        return implicit_euler_step(node.implicit_residual, STATE, {}, DT, params=p)[0]["velocity"]

    for loss in (explicit, implicit):
        g = jax.grad(loss)(CALIBRATED)["stiffness"]
        assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_integrate_node_with_params_is_jittable():
    step = jax.jit(lambda s, p: integrate_node(_spring(), s, {}, DT, method="heun", params=p))
    eager = integrate_node(_spring(), STATE, {}, DT, method="heun", params=CALIBRATED)
    assert _velocity(step(STATE, CALIBRATED)) == pytest.approx(_velocity(eager), rel=1e-5)


# ------------------------------------------------------------------
# The refusal that replaced the silence
# ------------------------------------------------------------------

class LegacyDerivatives(SpringDamperNode):
    """A subclass whose ``derivatives`` override predates the keyword."""

    def derivatives(self, state, boundary_inputs):  # noqa: D102 - the legacy shape
        return super().derivatives(state, boundary_inputs)


class LegacyResidual(SpringDamperNode):
    """A subclass whose ``implicit_residual`` override predates the keyword."""

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt):  # noqa: D102
        return super().implicit_residual(state_new, state_old, boundary_inputs, dt)


def test_accepts_params_reports_each_entry_point_separately():
    node = _spring(LegacyDerivatives)
    assert node.accepts_params() is True                                 # update
    assert node.accepts_params(method="derivatives") is False            # the legacy override
    assert node.accepts_params(method="implicit_residual") is True       # inherited, takes it
    assert _spring().accepts_params(method="derivatives") is True
    assert _spring().accepts_params(method="no_such_method") is False


@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
def test_integrate_node_refuses_params_for_a_legacy_derivatives_override(method):
    """Named, not dropped: the class and the method are in the message."""
    with pytest.raises(ValueError, match=r"LegacyDerivatives\.derivatives\(\) takes no 'params'"):
        integrate_node(_spring(LegacyDerivatives), STATE, {}, DT, method=method, params=CALIBRATED)


def test_integrate_node_without_params_still_calls_a_legacy_override():
    """Backward compatibility is the *absence* of params, not its emptiness
    being tolerated only sometimes: both spellings call the 2-argument form."""
    node = _spring(LegacyDerivatives)
    assert _velocity(integrate_node(node, STATE, {}, DT, method="euler")) == pytest.approx(-1.0, rel=1e-6)
    assert _velocity(integrate_node(node, STATE, {}, DT, method="euler", params={})) == pytest.approx(-1.0, rel=1e-6)
    assert _velocity(implicit_euler_step(node.implicit_residual, STATE, {}, DT)[0]) == pytest.approx(
        _velocity(implicit_euler_step(_spring().implicit_residual, STATE, {}, DT)[0]), rel=1e-6,
    )


def test_implicit_euler_step_refuses_params_for_a_legacy_residual_override():
    with pytest.raises(ValueError, match=r"LegacyResidual\.implicit_residual\(\) takes no 'params'"):
        implicit_euler_step(_spring(LegacyResidual).implicit_residual, STATE, {}, DT, params=CALIBRATED)
    # and still solves without them
    new, _ = implicit_euler_step(_spring(LegacyResidual).implicit_residual, STATE, {}, DT)
    assert _velocity(new) == pytest.approx(
        _velocity(implicit_euler_step(_spring().implicit_residual, STATE, {}, DT)[0]), rel=1e-6,
    )


def test_a_legacy_derivatives_override_under_an_inherited_residual_is_loud_not_silent():
    """The inherited ``implicit_residual`` takes ``params`` and forwards it
    to a ``derivatives`` that cannot: Python's own ``TypeError`` names
    ``LegacyDerivatives.derivatives``.  Not this module's message, but
    not silence either -- and that is the property this pins."""
    with pytest.raises(TypeError, match=r"derivatives\(\).*params"):
        implicit_euler_step(_spring(LegacyDerivatives).implicit_residual, STATE, {}, DT, params=CALIBRATED)


def test_implicit_euler_step_refuses_params_for_a_free_function_without_the_keyword():
    """The solver is node-agnostic: a plain callable is inspected directly."""
    def residual(x_new, x_old, bi, dt):
        return {k: x_new[k] - x_old[k] for k in x_new}

    with pytest.raises(ValueError, match=r"residual takes no 'params'"):
        implicit_euler_step(residual, STATE, {}, DT, params=CALIBRATED)
    new, _ = implicit_euler_step(residual, STATE, {}, DT)  # no params: fine
    assert _velocity(new) == pytest.approx(0.0)


def test_implicit_euler_step_forwards_params_to_a_free_function_that_takes_it():
    seen = []

    def residual(x_new, x_old, bi, dt, *, params=None):
        seen.append(params)
        k = params["stiffness"]
        c, m = 1.0, 1.0   # the fixture's damping and mass
        return {
            "position": x_new["position"] - x_old["position"] - dt * x_new["velocity"],
            "velocity": x_new["velocity"] - x_old["velocity"]
            + dt * (k * (x_new["position"] - REST_LENGTH) + c * x_new["velocity"]) / m,
        }

    new, _ = implicit_euler_step(residual, STATE, {}, DT, params=CALIBRATED)
    assert seen and all(p is CALIBRATED for p in seen)
    reference = implicit_euler_step(_spring(k=K_CALIBRATED).implicit_residual, STATE, {}, DT)[0]
    assert _velocity(new) == pytest.approx(_velocity(reference), rel=1e-6)


# ------------------------------------------------------------------
# Every in-tree override takes the keyword
# ------------------------------------------------------------------

def _in_tree_node_classes():
    for info in pkgutil.walk_packages(maddening.nodes.__path__, "maddening.nodes."):
        importlib.import_module(info.name)

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("maddening.nodes."):
                yield sub
            yield from walk(sub)

    return sorted(set(walk(SimulationNode)), key=lambda c: f"{c.__module__}.{c.__qualname__}")


@pytest.mark.parametrize("method", ["derivatives", "implicit_residual"])
def test_every_in_tree_override_takes_the_params_keyword(method):
    """Self-maintaining: a new node that overrides either method without
    the keyword fails here before it ships, and a legacy-shaped override
    would make ``integrate_node`` refuse a calibrated graph's params."""
    overriding = [
        cls for cls in _in_tree_node_classes()
        if getattr(cls, method) is not getattr(SimulationNode, method)
    ]
    assert overriding, f"no in-tree class overrides {method}(); the pin is empty"
    missing = [
        f"{cls.__module__}.{cls.__qualname__}" for cls in overriding
        if "params" not in inspect.signature(getattr(cls, method)).parameters
    ]
    assert not missing, f"{method}() overrides without a params keyword: {missing}"


def test_the_in_tree_derivatives_overrides_are_the_six_the_fix_covered():
    """Documents the count the wiring was measured against.  Raise it
    deliberately when a node gains an ODE form; a drop is a lost override."""
    names = {
        cls.__qualname__ for cls in _in_tree_node_classes()
        if cls.derivatives is not SimulationNode.derivatives
    }
    assert names >= {
        "BallNode", "SpringDamperNode", "HeatNode", "RigidBodyNode",
        "HeartPumpNode", "LBMNode",
    }


# ------------------------------------------------------------------
# The contract is documented where a caller would look
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "label, obj",
    [
        ("SimulationNode.derivatives", SimulationNode.derivatives),
        ("SimulationNode.implicit_residual", SimulationNode.implicit_residual),
        ("integrate_node", integrate_node),
        ("implicit_euler_step", implicit_euler_step),
    ],
)
def test_the_params_contract_is_documented_where_a_caller_would_look(label, obj):
    """Each of the four states the ``{**self.params, **params}`` rule and
    cites the anomaly as history, and none still claims the value cannot
    be delivered."""
    doc = obj.__doc__ or ""
    assert "params" in doc and "MADD-ANO-018" in doc, label
    for stale in ("cannot reach", "cannot be passed", "does not reach this solve"):
        assert stale not in doc, f"{label} still says {stale!r}"
    assert "params" in inspect.signature(obj).parameters, label
