"""
Matplotlib-based renderers for MADDENING simulations.

Two renderers are provided:

- ``MatplotlibTimeSeriesRenderer``: live time-series plots of state fields.
- ``MatplotlibSceneRenderer``: 2D animated scene with shapes mapped to state.

Both poll a ``StateRelay`` and are driven by ``matplotlib.animation.FuncAnimation``.
Use ``run_matplotlib()`` to drive one or more renderers from a single event loop.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from maddening.viz.renderer import Renderer, GraphInfo
from maddening.viz.relay import StateRelay


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def run_matplotlib(*renderers, interval_ms: int = 33) -> None:
    """Drive one or more matplotlib renderers and block on ``plt.show()``.

    Each renderer must already have had ``setup()`` called.  This function
    creates a ``FuncAnimation`` for each renderer, then enters the
    matplotlib event loop.

    Parameters
    ----------
    *renderers
        One or more ``MatplotlibTimeSeriesRenderer`` or
        ``MatplotlibSceneRenderer`` instances.
    interval_ms : int
        Milliseconds between animation frames (default 33, ~30 fps).
    """
    for r in renderers:
        r.start_animation(interval_ms=interval_ms)
    plt.show()


# ------------------------------------------------------------------
# Time-series renderer
# ------------------------------------------------------------------

class MatplotlibTimeSeriesRenderer(Renderer):
    """Live time-series line plots of selected state fields.

    Parameters
    ----------
    relay : StateRelay
        The snapshot buffer to poll for new data.
    plot_config : dict, optional
        Configuration dictionary.  Supported keys:

        - ``"fields"``: ``{node_name: [field1, ...]}`` -- which fields
          to plot.  Defaults to all state fields from all nodes.
        - ``"window"``: int -- number of data points to keep visible
          (0 = show all history).
        - ``"title"``: str -- figure title.
        - ``"figsize"``: tuple -- matplotlib figure size.
    """

    def __init__(self, relay: StateRelay, plot_config: dict = None):
        self._relay = relay
        self._config = plot_config or {}
        self._fig = None
        self._anim = None
        self._last_sim_time = 0.0

    def setup(self, graph_info: GraphInfo) -> None:
        fields = self._config.get("fields", None)
        if fields is None:
            fields = graph_info.node_state_fields

        self._tracked = []
        for node, field_list in fields.items():
            for f in field_list:
                self._tracked.append((node, f))

        n_plots = len(self._tracked)
        if n_plots == 0:
            raise ValueError("No fields to plot.")

        figsize = self._config.get("figsize", (10, 3 * n_plots))
        self._fig, axes = plt.subplots(n_plots, 1, figsize=figsize, squeeze=False)
        self._axes = [ax for row in axes for ax in row]

        self._lines = {}
        self._time_buffers = {}
        self._data_buffers = {}
        self._window = self._config.get("window", 0)

        for i, (node, field) in enumerate(self._tracked):
            key = (node, field)
            line, = self._axes[i].plot([], [], "b-", linewidth=0.8)
            self._axes[i].set_ylabel(f"{node}.{field}")
            self._lines[key] = line
            self._time_buffers[key] = []
            self._data_buffers[key] = []

        self._axes[-1].set_xlabel("Time (s)")
        title = self._config.get("title", "MADDENING — Time Series")
        self._fig.suptitle(title)
        self._fig.tight_layout()

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        for node, field in self._tracked:
            key = (node, field)
            if node in state and field in state[node]:
                self._time_buffers[key].append(sim_time)
                self._data_buffers[key].append(float(state[node][field]))

                t_buf = self._time_buffers[key]
                d_buf = self._data_buffers[key]
                if self._window > 0 and len(t_buf) > self._window:
                    t_buf = t_buf[-self._window:]
                    d_buf = d_buf[-self._window:]

                self._lines[key].set_data(t_buf, d_buf)

        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()

    def start_animation(self, interval_ms: int = 33) -> None:
        """Create the ``FuncAnimation``.  Call before ``plt.show()``."""
        def _tick(frame):
            sim_time, snapshot = self._relay.latest_snapshot()
            if snapshot is not None and sim_time != self._last_sim_time:
                self._last_sim_time = sim_time
                self.update(sim_time, snapshot)

        self._anim = animation.FuncAnimation(
            self._fig, _tick, interval=interval_ms, blit=False,
            cache_frame_data=False,
        )

    def run_event_loop(self, interval_ms: int = 33) -> None:
        """Convenience: start animation and block on ``plt.show()``."""
        self.start_animation(interval_ms)
        plt.show()

    def teardown(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)

    def requested_fields(self):
        return self._config.get("fields", None)


# Backwards-compatible alias
MatplotlibRenderer = MatplotlibTimeSeriesRenderer


# ------------------------------------------------------------------
# 2D Scene renderer
# ------------------------------------------------------------------

class MatplotlibSceneRenderer(Renderer):
    """2D animated scene with shapes driven by simulation state.

    Parameters
    ----------
    relay : StateRelay
        The snapshot buffer to poll for new data.
    scene_config : dict
        Scene description.  Keys:

        - ``"title"``: str -- figure title.
        - ``"figsize"``: tuple -- matplotlib figure size (default (5, 7)).
        - ``"xlim"``: (float, float) -- x-axis limits.
        - ``"ylim"``: (float, float) -- y-axis limits.
        - ``"aspect"``: str -- axis aspect ratio (default ``"equal"``).
        - ``"objects"``: list[dict] -- visual objects.

    Object types
    ~~~~~~~~~~~~~

    **circle** -- a filled circle whose position is driven by state.

    .. code-block:: python

        {
            "type": "circle",
            "node": "ball",
            "y": "position",          # state field -> y centre
            "x": 0.0,                 # fixed x centre (or a state field name)
            "radius": 0.2,
            "color": "#DD4444",
            "edgecolor": "#991111",   # optional
        }

    **surface** -- a horizontal surface at a state-driven height with a
    filled rectangle underneath.

    .. code-block:: python

        {
            "type": "surface",
            "node": "table",
            "y": "position",          # state field -> surface height
            "depth": 0.5,             # filled region below the line
            "color": "#8B7355",
            "linecolor": "black",     # optional, default "black"
            "linewidth": 1.5,         # optional
        }

    **hline** -- a simple horizontal line at a state-driven height.

    .. code-block:: python

        {
            "type": "hline",
            "node": "table",
            "y": "position",
            "color": "black",
            "linewidth": 1.0,
        }
    """

    def __init__(self, relay: StateRelay, scene_config: dict):
        self._relay = relay
        self._config = scene_config
        self._fig = None
        self._ax = None
        self._anim = None
        self._artists = []  # list of (obj_spec, artist_or_artists) tuples
        self._time_text = None
        self._last_sim_time = 0.0

    def setup(self, graph_info: GraphInfo) -> None:
        figsize = self._config.get("figsize", (5, 7))
        self._fig, self._ax = plt.subplots(figsize=figsize)

        ax = self._ax
        ax.set_xlim(self._config.get("xlim", (-1, 1)))
        ax.set_ylim(self._config.get("ylim", (-1, 6)))
        ax.set_aspect(self._config.get("aspect", "equal"))
        ax.set_xticks([])
        ax.set_ylabel("Height (m)")

        title = self._config.get("title", "MADDENING — Scene")
        self._fig.suptitle(title)

        # Time label
        self._time_text = ax.text(
            0.02, 0.97, "t = 0.00 s", transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
        )

        # Create artists for each object
        for obj in self._config.get("objects", []):
            obj_type = obj["type"]

            if obj_type == "circle":
                y0 = 0.0
                radius = obj.get("radius", 0.2)
                circle = patches.Circle(
                    (obj.get("x", 0.0), y0 + radius), radius,
                    facecolor=obj.get("color", "red"),
                    edgecolor=obj.get("edgecolor", obj.get("color", "red")),
                    linewidth=obj.get("linewidth", 1.5),
                    zorder=5,
                )
                ax.add_patch(circle)
                self._artists.append((obj, circle))

            elif obj_type == "surface":
                xlim = self._config.get("xlim", (-1, 1))
                depth = obj.get("depth", 0.5)
                rect = patches.Rectangle(
                    (xlim[0], -depth), xlim[1] - xlim[0], depth,
                    facecolor=obj.get("color", "#8B7355"),
                    linewidth=0, zorder=1,
                )
                ax.add_patch(rect)
                line_artist = ax.axhline(
                    0, color=obj.get("linecolor", "black"),
                    linewidth=obj.get("linewidth", 1.5), zorder=2,
                )
                self._artists.append((obj, (rect, line_artist)))

            elif obj_type == "hline":
                line_artist = ax.axhline(
                    0, color=obj.get("color", "black"),
                    linewidth=obj.get("linewidth", 1.0), zorder=2,
                )
                self._artists.append((obj, line_artist))

        self._fig.tight_layout()

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        self._time_text.set_text(f"t = {sim_time:.2f} s")

        for obj, artist in self._artists:
            node = obj["node"]
            if node not in state:
                continue
            node_state = state[node]
            y_field = obj.get("y")
            y_val = float(node_state[y_field]) if y_field and y_field in node_state else 0.0

            obj_type = obj["type"]

            if obj_type == "circle":
                radius = obj.get("radius", 0.2)
                x_spec = obj.get("x", 0.0)
                if isinstance(x_spec, str) and x_spec in node_state:
                    x_val = float(node_state[x_spec])
                else:
                    x_val = float(x_spec)
                artist.set_center((x_val, y_val + radius))

            elif obj_type == "surface":
                rect, line_artist = artist
                depth = obj.get("depth", 0.5)
                xlim = self._config.get("xlim", (-1, 1))
                rect.set_xy((xlim[0], y_val - depth))
                line_artist.set_ydata([y_val, y_val])

            elif obj_type == "hline":
                artist.set_ydata([y_val, y_val])

    def start_animation(self, interval_ms: int = 33) -> None:
        """Create the ``FuncAnimation``.  Call before ``plt.show()``."""
        def _tick(frame):
            sim_time, snapshot = self._relay.latest_snapshot()
            if snapshot is not None and sim_time != self._last_sim_time:
                self._last_sim_time = sim_time
                self.update(sim_time, snapshot)

        self._anim = animation.FuncAnimation(
            self._fig, _tick, interval=interval_ms, blit=False,
            cache_frame_data=False,
        )

    def run_event_loop(self, interval_ms: int = 33) -> None:
        """Convenience: start animation and block on ``plt.show()``."""
        self.start_animation(interval_ms)
        plt.show()

    def teardown(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)

    def requested_fields(self):
        fields = {}
        for obj in self._config.get("objects", []):
            node = obj["node"]
            y = obj.get("y")
            if y:
                fields.setdefault(node, [])
                if y not in fields[node]:
                    fields[node].append(y)
            x = obj.get("x")
            if isinstance(x, str):
                fields.setdefault(node, [])
                if x not in fields[node]:
                    fields[node].append(x)
        return fields or None
