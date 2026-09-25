"""Every built-in node is rebuilt by a save/reload exactly as it was built.

A saved graph rebuilds each node as ``cls(name=, timestep=, **params)``
(``GraphManager.from_dict``, ``load_graph_from_usd``), so a constructor
argument the node keeps anywhere but ``params`` is dropped by every reload
while the reloaded graph still runs.  ``LBMNode``'s ``wall_mask`` was such
an argument until 0.4.0 (MADD-ANO-034).  Each built-in node is built here
with non-default values for its arguments -- arrays where an argument takes
one -- and reloaded through JSON and through USD; the reload must write the
same config, build the same initial state and step the same trajectory,
bit for bit.

``test_every_built_in_node_has_a_round_trip_case`` fails closed when a node
class is added to ``maddening.nodes`` or ``maddening.nodes.adaptive``
without a case here.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import warnings

import numpy as np
import pytest

import maddening.nodes as builtin_nodes
import maddening.nodes.adaptive as adaptive_nodes
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import (
    BallNode,
    HealthCheckNode,
    HeartPumpNode,
    HeatNode,
    LBMNode,
    LBMPipeNode,
    RigidBody2DNode,
    RigidBodyNode,
    SpringDamperNode,
    TableNode,
)

STEPS = 3


def _channel_walls():
    wall = np.zeros((10, 6), bool)
    wall[:, 0] = wall[:, -1] = True
    wall[4, 2] = True
    return wall


#: ``(class, timestep, kwargs)`` per case.  Arrays where the constructor
#: takes one, and every value off its default.
CASES = {
    "ball": (BallNode, 0.01, dict(initial_position=1.3, initial_velocity=0.2,
                                  elasticity=0.6, gravity=-3.0)),
    "table": (TableNode, 0.01, dict(position=0.4)),
    "spring": (SpringDamperNode, 0.01, dict(stiffness=50.0, damping=0.3, mass=2.0,
                                            rest_length=0.5, initial_position=0.9,
                                            initial_velocity=0.1)),
    "heart_pump": (HeartPumpNode, 0.01, dict(resistance=1.2, compliance=0.9, heart_rate=60.0,
                                             stroke_volume=60.0, venous_pressure=2.0,
                                             systole_fraction=0.3, initial_pressure=70.0)),
    "health_check": (HealthCheckNode, 0.01, dict(checks={"x": {"finite": True, "min": -1.0,
                                                               "max": 2.0}})),
    "heat_non_uniform": (HeatNode, 0.01, dict(
        n_cells=8, length=2.0, thermal_diffusivity=0.001, stencil_order=2,
        grid_points=np.linspace(0.0, 1.0, 8) ** 2,
        initial_temperature=np.linspace(0.0, 1.0, 8).tolist())),
    "heat_fourth_order": (HeatNode, 0.01, dict(n_cells=8, stencil_order=4,
                                               thermal_diffusivity=0.001,
                                               initial_temperature=1.0)),
    "lbm_walled_2d": (LBMNode, 1.0, dict(grid_shape=(10, 6), lattice="D2Q9", viscosity=0.1,
                                         wall_mask=_channel_walls(), inlet_face="y_min",
                                         outlet_face="y_max")),
    "lbm_pipe": (LBMPipeNode, 1.0, dict(nx=8, ny=6, nz=6, tau=0.7, pipe_radius=0.8,
                                        propeller_x=3, propeller_strength=0.001,
                                        G=-5.0, fill_fraction=0.6)),
    "rigid_body": (RigidBodyNode, 0.01, dict(
        mass=2.0, inertia=np.array([1.0, 2.0, 3.0]), gravity=np.array([0.0, -1.0, -9.81]),
        constraints={"z": 0.0}, initial_position=np.array([0.1, 0.2, 0.3]),
        initial_velocity=(1.0, 0.0, 0.0), initial_orientation=(1.0, 0.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.1, 0.0))),
    "rigid_body_2d": (RigidBody2DNode, 0.01, dict(mass=2.0, inertia=0.5,
                                                  gravity=np.array([0.0, -2.0]), initial_x=0.3,
                                                  initial_vy=0.1, initial_omega=0.2)),
}

#: Built-in node classes whose round trip is tested elsewhere, and where.
#: The wavelet node compiles too slowly for a per-reload case here.
COVERED_ELSEWHERE: dict[type, str] = {
    adaptive_nodes.WaveletAdaptiveNode: (
        "tests/nodes/adaptive/test_wavelet_node.py::"
        "test_a_config_round_trip_of_a_graph_preserves_the_trajectory and "
        "::test_an_explicit_dtype_survives_a_config_round_trip"),
}

REGISTRY = {cls.__name__: cls for cls, _, _ in CASES.values()}


def _build(case: str) -> SimulationNode:
    cls, dt, kwargs = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # RigidBody2DNode
        return cls(case, dt, **kwargs)


def _graph(case: str) -> GraphManager:
    gm = GraphManager()
    gm.add_node(_build(case))
    gm.compile()
    return gm


def _trajectory(gm: GraphManager, name: str) -> dict:
    gm.run(STEPS)
    return {k: np.asarray(v) for k, v in gm.get_node_state(name).items()}


def _initial_state(gm: GraphManager, name: str) -> dict:
    return {k: np.asarray(v) for k, v in gm.get_node(name).initial_state().items()}


@functools.lru_cache(maxsize=None)
def _original(case: str):
    """The original graph, its config, initial state and trajectory, built
    and compiled once per case for both reloads.  The trajectory is taken on
    the freshly compiled graph; a config and a stage carry no state, so
    reloading from the graph after it has stepped is the same reload."""
    gm = _graph(case)
    config = json.dumps(gm.to_dict(), sort_keys=True)
    return gm, config, _initial_state(gm, case), _trajectory(gm, case)


def _assert_identical(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k].dtype == want[k].dtype, k
        np.testing.assert_array_equal(got[k], want[k], err_msg=k)


def _reload_json(gm: GraphManager) -> GraphManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)


def _reload_usd(gm: GraphManager) -> GraphManager:
    from pxr import Usd

    from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return load_graph_from_usd(stage, node_registry=REGISTRY)


def _usd_installed() -> bool:
    try:
        return importlib.util.find_spec("pxr") is not None
    except (ImportError, ValueError):
        return False


usd_only = pytest.mark.skipif(not _usd_installed(), reason="usd-core not installed")


#: Cases whose compile alone is over the per-test budget on a CI runner
#: (the part-full two-phase pipe, ~9 s here on 3 cores): the slow lane runs
#: their config reload.  Their USD reload is not slow-marked, and is a test
#: of its own so that nothing marks it: only `test-usd` installs usd-core
#: and it runs no slow test (nor judges test time), so a slow mark there
#: would run it nowhere.  The other lanes skip it.
SLOW_CASES = {"lbm_pipe"}


def _check_reload(case: str, reload) -> None:
    original, config, initial, trajectory = _original(case)
    reloaded = reload(original)
    reloaded.compile()

    assert json.dumps(reloaded.to_dict(), sort_keys=True) == config
    _assert_identical(_initial_state(reloaded, case), initial)
    _assert_identical(_trajectory(reloaded, case), trajectory)


@pytest.mark.parametrize("case", [
    pytest.param(case, marks=pytest.mark.slow) if case in SLOW_CASES else case
    for case in sorted(CASES)
])
def test_a_config_reloaded_built_in_node_writes_builds_and_steps_as_the_original(case):
    _check_reload(case, _reload_json)


@usd_only
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_usd_reloaded_built_in_node_writes_builds_and_steps_as_the_original(case):
    _check_reload(case, _reload_usd)


def test_every_built_in_node_has_a_round_trip_case():
    """Fails closed: a node class exported by ``maddening.nodes`` or
    ``maddening.nodes.adaptive`` with no case above (and no entry in
    ``COVERED_ELSEWHERE``) is a node whose reload nobody checks."""
    exported = {
        obj for module in (builtin_nodes, adaptive_nodes) for name in module.__all__
        if inspect.isclass(obj := getattr(module, name))
        and issubclass(obj, SimulationNode) and not inspect.isabstract(obj)
    }
    assert len(exported) >= 11, sorted(c.__name__ for c in exported)
    cased = {cls for cls, _, _ in CASES.values()} | set(COVERED_ELSEWHERE)
    assert exported <= cased, f"no round-trip case for {sorted(c.__name__ for c in exported - cased)}"
