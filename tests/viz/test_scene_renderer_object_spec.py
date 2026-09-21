"""A scene object's ``x`` may name a state field, and setup must survive it.

``MatplotlibSceneRenderer``'s documented object spec says ``"x"`` is a
"fixed x centre (or a state field name)", and ``update()`` handles both.
``setup()`` did not: it passed the value straight to ``patches.Circle``,
so a field name reached matplotlib as a coordinate and setup died with
``ConversionError`` before a single frame was drawn.  The typed
``SceneObjectSpec`` (``x: float | str``) is what surfaced it.
"""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from maddening.viz.backends.matplotlib_renderer import (  # noqa: E402
    MatplotlibSceneRenderer,
)
from maddening.viz.relay import StateRelay  # noqa: E402
from maddening.viz.renderer import GraphInfo  # noqa: E402


def _graph_info() -> GraphInfo:
    return GraphInfo(
        node_names=["ball"],
        node_params={"ball": {}},
        node_state_fields={"ball": ["position", "lateral"]},
        edges=[],
        timestep=0.01,
    )


def _setup(obj) -> tuple[MatplotlibSceneRenderer, object]:
    renderer = MatplotlibSceneRenderer(StateRelay(), {"objects": [obj]})
    renderer.setup(_graph_info())
    return renderer, renderer._artists[0][1]


def test_a_state_field_name_as_x_starts_the_circle_at_the_origin():
    renderer, circle = _setup(
        {"type": "circle", "node": "ball", "y": "position", "x": "lateral",
         "radius": 0.25},
    )
    try:
        # Not the field name: there is no state yet at setup time.
        assert circle.center == (0.0, 0.25)
    finally:
        renderer.teardown()


def test_a_numeric_x_still_places_the_circle_there():
    """The branch the fix must not have broken."""
    renderer, circle = _setup(
        {"type": "circle", "node": "ball", "y": "position", "x": 1.5,
         "radius": 0.25},
    )
    try:
        assert circle.center == (1.5, 0.25)
    finally:
        renderer.teardown()


def test_update_resolves_a_state_field_name_as_x():
    """``update()`` already did this; pin it so the two stay consistent."""
    renderer, circle = _setup(
        {"type": "circle", "node": "ball", "y": "position", "x": "lateral",
         "radius": 0.25},
    )
    try:
        renderer.update(0.1, {"ball": {"position": 2.0, "lateral": 3.0}})
        assert circle.center == (3.0, 2.25)
    finally:
        renderer.teardown()
