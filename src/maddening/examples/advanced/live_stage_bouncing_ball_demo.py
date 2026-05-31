"""Live-stage bouncing-ball demo — proves :class:`LiveStage` works
without a MIME dependency (v0.3.0 §A3 acceptance gate).

Spins up a 2-node MADDENING graph (a :class:`BallNode` falling under
gravity, coupled with a :class:`TableNode` that catches it), wires
the ball's position to a USD sphere via
:func:`maddening.usd.live_stage.make_translate_updater`, and exports
the stage to disk.

Run with::

    python -m maddening.examples.advanced.live_stage_bouncing_ball_demo \\
        --out /tmp/bouncing_ball.usda

The resulting ``.usda`` file can be loaded by MICROROBOTICA (or any
USD-capable viewer) to verify the non-MIME live-stage path works
end-to-end.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional


def run(out_path: Optional[Path] = None, n_steps: int = 100) -> Path:
    """Run the demo and write a time-sampled .usda file.

    Parameters
    ----------
    out_path : Path, optional
        Where to write the .usda.  Defaults to
        ``./bouncing_ball.usda`` in the current working directory.
    n_steps : int
        Number of timesteps to simulate.

    Returns
    -------
    Path
        The path the stage was exported to.
    """
    # USD bindings are optional — fail loudly with a clear message rather
    # than at import time, so the rest of the example registry stays
    # importable.
    try:
        from pxr import Usd, UsdGeom  # type: ignore  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "live_stage_bouncing_ball_demo needs usd-core: pip install usd-core",
        ) from exc

    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.ball import BallNode
    from maddening.nodes.table import TableNode
    from maddening.usd.live_stage import LiveStage, make_translate_updater

    if out_path is None:
        out_path = Path("bouncing_ball.usda")

    # Build the graph: BallNode under gravity, with a stationary
    # TableNode feeding its position back to the ball for bounce
    # collision (this is the canonical 2-node example used in the
    # MADDENING quickstart).
    gm = GraphManager()
    ball = BallNode(
        name="ball", timestep=1e-2,
        initial_position=1.0, initial_velocity=0.0,
    )
    table = TableNode(name="table", timestep=1e-2)
    gm.add_node(ball)
    gm.add_node(table)
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()

    # Build the stage.
    stage = LiveStage()
    stage.add_dome_light()
    stage.add_ground_plane(normal="Y", offset=0.0, size=2.0)

    sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Ball")
    sphere.GetRadiusAttr().Set(0.05)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp()

    stage.register_prim(
        node_name="ball", prim_path="/World/Ball",
        updater=make_translate_updater(field="position"),
    )

    # Convert ball's 1-D position into a 3-D translate (x = 0, y = pos, z = 0).
    # Wrap the updater to do this conversion since BallNode emits scalar y.
    base_updater = make_translate_updater(field="position")
    def adapt_1d_to_3d(stage, prim_path, node_state, time_code=None):
        pos = node_state.get("position")
        if pos is None:
            return
        import numpy as np  # noqa: PLC0415
        # BallNode position is a scalar (the y-coordinate above the ground).
        adapted = np.array([0.0, float(pos), 0.0])
        base_updater(stage, prim_path,
                     {"position": adapted}, time_code=time_code)

    # Replace the registered updater with the 1D-aware adapter.
    stage._dynamic_prims[-1].updater = adapt_1d_to_3d

    # Run the simulation, writing time-sampled positions.  step() does
    # the bookkeeping for us — the returned dict is the user-facing
    # state ({node_name: {field: value}}).
    for t in range(n_steps):
        state = gm.step()
        stage.update(state, time_code=Usd.TimeCode(float(t)))

    # Set time-axis metadata so the viewer knows the range.
    stage.stage.SetStartTimeCode(0.0)
    stage.stage.SetEndTimeCode(float(n_steps - 1))
    stage.stage.SetFramesPerSecond(60.0)

    stage.export(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .usda path (default: ./bouncing_ball.usda)",
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of timesteps (default: 100)",
    )
    args = parser.parse_args()
    path = run(out_path=args.out, n_steps=args.steps)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
