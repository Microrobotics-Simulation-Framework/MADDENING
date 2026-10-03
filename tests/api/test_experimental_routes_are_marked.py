"""The surrogate-training routes and the state streams are experimental in
0.4.0, and say so wherever a level is read.

They are to be hardened in 0.5.0.  None carried a level before (neither
``SimulationServer`` nor ``maddening.viz`` was tagged), so experimental is
their first level, not a demotion.  A route has no Python name for
``@stability`` to tag, so ``ROUTE_STABILITY`` records it, the stability
registry (and so the report) lists it as ``maddening.api.server:<route>``,
and an HTTP route carries ``x-maddening-stability`` in ``/openapi.json``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient

from maddening.api.server import ROUTE_STABILITY, SimulationServer
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import (
    _STABILITY_REGISTRY,
    register_route_stability,
)
from maddening.viz.relay import StateRelay

EXPERIMENTAL = {
    "POST /surrogate/train", "GET /surrogate/status/{job_id}",
    "POST /surrogate/activate/{job_id}", "POST /surrogate/deactivate/{node_name}",
    "WS /ws/state", "WS /ws/state/binary", "WS /ws/render",
}


def _app_routes() -> set[str]:
    app = SimulationServer({}).create_app()
    out = set()
    for route in app.routes:
        if isinstance(route, APIWebSocketRoute):
            out.add(f"WS {route.path}")
        elif isinstance(route, APIRoute):
            out.update(f"{m} {route.path}" for m in route.methods - {"HEAD"})
    return out


def test_the_training_routes_and_the_streams_are_experimental():
    assert {r for r, lvl in ROUTE_STABILITY.items()
            if lvl is StabilityLevel.EXPERIMENTAL} == EXPERIMENTAL


def test_every_marked_route_is_served_and_every_surrogate_or_stream_route_is_marked():
    served = _app_routes()
    assert set(ROUTE_STABILITY) <= served, set(ROUTE_STABILITY) - served
    family = {r for r in served if r.startswith("WS ") or " /surrogate/" in r}
    assert family <= set(ROUTE_STABILITY), family - set(ROUTE_STABILITY)


def test_the_marked_routes_are_in_the_stability_registry():
    for route, level in ROUTE_STABILITY.items():
        assert _STABILITY_REGISTRY.get(f"maddening.api.server:{route}") is level, route


def test_an_http_routes_level_is_published_in_the_openapi_document():
    schema = TestClient(SimulationServer({}).create_app()).get("/openapi.json").json()
    for route, level in ROUTE_STABILITY.items():
        method, path = route.split(" ", 1)
        if method == "WS":
            continue
        op = schema["paths"][path][method.lower()]
        assert op.get("x-maddening-stability") == level.value, route
    assert "x-maddening-stability" not in schema["paths"]["/sim/step"]["post"]


def test_the_relay_the_streams_read_is_experimental():
    assert StateRelay._stability_level is StabilityLevel.EXPERIMENTAL
    assert _STABILITY_REGISTRY["maddening.viz.relay.StateRelay"] is StabilityLevel.EXPERIMENTAL


def test_a_route_cannot_be_registered_stable():
    with pytest.raises(ValueError, match="cannot be registered stable"):
        register_route_stability("maddening.api.server", "GET /graph", StabilityLevel.STABLE)
    assert "maddening.api.server:GET /graph" not in _STABILITY_REGISTRY
