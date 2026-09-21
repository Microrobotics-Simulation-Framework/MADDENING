"""Every built-in node passes the ``verify_node`` battery.

This is the CI gate for the universal node properties (finite, structure,
determinism, jit/eager, finite state gradient) plus the params contract:
for nodes whose ``update`` takes the graph's ``params`` pytree, injected
params reproduce the baked constants and the gradient with respect to
them is finite.  Nodes that have not migrated to ``params`` get ``SKIP``
on the params checks; a node listed in ``MIGRATED`` must not skip them —
that list is the ledger of the params migration.

The sampling envelopes are the nodes' validated regimes, not the
``(-1e4, 1e4)`` default: the checks are about the update's structure,
not about whether a 1D heat cell survives a 1e4 K temperature jump in
one explicit step.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from hypothesis import given, settings

from maddening.nodes.ball import BallNode
from maddening.nodes.health_check import HealthCheckNode
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.rigid_body import RigidBodyNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from maddening.testing.verification import DEFAULT_CHECKS, make_inputs, verify_node

KW = dict(max_examples=100, derandomize=True)

N_CELLS = 8

CASES = {
    "spring": dict(
        node=lambda: SpringDamperNode("s", 0.01, stiffness=50.0, damping=2.0,
                                      mass=1.5, rest_length=1.0),
        bounds={"position": (-10.0, 10.0), "velocity": (-10.0, 10.0)},
        boundary_bounds={"anchor_position": (-10.0, 10.0)},
    ),
    "ball": dict(
        node=lambda: BallNode("b", 0.01, initial_position=1.0, elasticity=0.7),
        bounds={"position": (-10.0, 10.0), "velocity": (-20.0, 20.0)},
        boundary_bounds={"table_position": (-10.0, 10.0)},
    ),
    "heat": dict(
        node=lambda: HeatNode("h", 1e-4, n_cells=N_CELLS, thermal_diffusivity=0.1,
                              length=1.0),
        bounds={"temperature": (250.0, 400.0)},
        boundary_bounds={
            "left_temperature": (250.0, 400.0),
            "right_temperature": (250.0, 400.0),
            "heat_source": (-100.0, 100.0),
        },
    ),
    "rigid_body": dict(
        node=lambda: RigidBodyNode("r", 0.01, mass=2.0, inertia=(1.0, 2.0, 3.0)),
        bounds={
            "position": (-10.0, 10.0),
            "velocity": (-10.0, 10.0),
            # Away from the zero quaternion: ``quat_normalize`` divides
            # by the norm and the physical state is unit-norm anyway.
            "orientation": (0.25, 1.0),
            "angular_velocity": (-5.0, 5.0),
        },
        boundary_bounds={"force": (-50.0, 50.0), "torque": (-50.0, 50.0)},
    ),
    "rigid_body_2d": dict(
        node=lambda: RigidBody2DNode("r2", 0.01, mass=2.0, inertia=0.5),
        bounds={
            "x": (-10.0, 10.0), "angle": (-6.3, 6.3),
            "v": (-10.0, 10.0), "omega": (-5.0, 5.0),
        },
        boundary_bounds={"force": (-50.0, 50.0), "torque": (-50.0, 50.0)},
    ),
    "heart_pump": dict(
        node=lambda: HeartPumpNode("hp", 1e-3, resistance=1.0, compliance=1.0,
                                   heart_rate=72.0, stroke_volume=70.0,
                                   venous_pressure=5.0, systole_fraction=0.35),
        bounds={
            "arterial_pressure": (40.0, 200.0),
            "phase": (0.0, 1.0),
            "flow_rate": (0.0, 500.0),
        },
        boundary_bounds={},
        # No ``backpressure`` so the outflow reads ``venous_pressure``
        # and every Windkessel constant is exercised (the backpressure
        # path is covered by tests/nodes/test_heart_pump.py).
        kwargs=dict(boundary_inputs={}),
    ),
    "table": dict(
        node=lambda: TableNode("t", 0.01, position=0.5),
        bounds={"position": (-10.0, 10.0)},
        boundary_bounds={},
    ),
    "health_check": dict(
        node=lambda: HealthCheckNode("hc", 0.01, checks={
            "density": {"finite": True, "min": 0.0, "max": 10.0},
            "velocity": {"max_abs": 100.0},
        }),
        bounds={},
        boundary_bounds={},
        # A monitor with bool/int32 state (sampled as such) and fixed
        # boundary inputs; ``gradient_finite`` passes vacuously (no float
        # field) and ``structure`` checks the dtypes survive.
        kwargs=dict(
            boundary_inputs={"density": jnp.array([1.0, 2.0, 11.0]),
                             "velocity": jnp.array([3.0, -4.0])},
        ),
    ),
}

# Nodes that take ``params``: the params checks must run, not SKIP
# (except ``params_gradient_finite`` / ``params_effective`` on a node
# whose ``params_pytree()`` is empty or all non-trainable).
MIGRATED = {
    "spring", "ball", "heat", "rigid_body",
    "rigid_body_2d", "heart_pump", "table", "health_check",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_builtin_node_passes_battery(name):
    case = CASES[name]
    node = case["node"]()
    results = verify_node(
        node, case["bounds"], boundary_bounds=case["boundary_bounds"],
        **case.get("kwargs", {}), **KW,
    )
    bad = [str(r) for r in results.values() if not r.passed]
    assert not bad, f"{name}:\n" + "\n".join(bad)
    if name in MIGRATED:
        assert node.accepts_params()
        assert not results["params_consistent"].skipped, f"{name}: params_consistent skipped"
        assert results["params_consistent"].n_examples > 0
        pytree = node.params_pytree()
        specs = node.param_specs()
        trainable = [k for k in pytree if specs.get(k) is None or specs[k].trainable]
        if pytree:
            assert not results["params_gradient_finite"].skipped, name
        if trainable and "params_effective" in results:
            assert not results["params_effective"].skipped, name
    else:
        assert not node.accepts_params(), f"{name}: migrated but not in MIGRATED"


def test_default_checks_include_params_battery():
    assert {"params_consistent", "params_gradient_finite", "params_effective"} <= set(DEFAULT_CHECKS)


_HEAT_INPUTS = make_inputs(
    CASES["heat"]["node"](), CASES["heat"]["bounds"],
    boundary_bounds=CASES["heat"]["boundary_bounds"],
)


@given(_HEAT_INPUTS.strategy())
def test_heat_source_sampled_with_declared_shape(args):
    """The battery samples ``heat_source`` at its declared ``(n,)`` shape,
    so the check exercised the real stencil path, not a broadcast scalar."""
    _, bi, _ = args
    assert jnp.shape(bi["heat_source"]) == (N_CELLS,)


@pytest.mark.parametrize(
    "name, paths, unused",
    [
        ("spring", "update, derivatives, implicit_residual", None),
        ("heat", "update, derivatives, implicit_residual", None),
        ("heart_pump", "update, derivatives, implicit_residual", None),
        ("rigid_body", "update, derivatives, implicit_residual", None),
        # The collision-free right-hand side has no use for ``elasticity``:
        # reported, not failed -- the case that separates "not consumed"
        # from "read from self.params".
        ("ball", "update, derivatives", "not consumed by derivatives(): ['elasticity']"),
    ],
)
def test_params_effective_names_the_solver_paths_it_checked(name, paths, unused):
    case = CASES[name]
    res = verify_node(
        case["node"](), case["bounds"], boundary_bounds=case["boundary_bounds"],
        checks=["params_effective"], **case.get("kwargs", {}), **KW,
    )["params_effective"]
    assert res.passed, res.detail
    assert f"paths checked: {paths}" in res.detail, res.detail
    if unused:
        assert unused in res.detail, res.detail
