"""
TerminalRenderer -- live-updating text display of simulation state.

Uses `rich <https://rich.readthedocs.io/>`_ for flicker-free in-place
terminal updates.  Works over SSH, in any terminal, with no GUI
dependencies.
"""

import threading
from typing import Optional

try:
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError as _exc:
    raise ImportError(
        "TerminalRenderer requires 'rich'. "
        "Install with:  pip install maddening[terminal]"
    ) from _exc

from maddening.viz.renderer import Renderer, GraphInfo
from maddening.viz.relay import StateRelay


class TerminalRenderer(Renderer):
    """Render simulation state as a live-updating table in the terminal.

    Parameters
    ----------
    relay : StateRelay
        The snapshot buffer to poll for new data.
    config : dict, optional
        Configuration dictionary.  Supported keys:

        - ``"fields"``: ``{node_name: [field1, ...]}`` -- which fields
          to display.  Defaults to all fields.
        - ``"precision"``: int -- decimal places for floats (default 4).
        - ``"title"``: str -- header text.
        - ``"refresh_hz"``: float -- display refresh rate (default 20).
    """

    def __init__(self, relay: StateRelay, config: dict = None):
        self._relay = relay
        self._config = config or {}
        self._tracked: list[tuple[str, str]] = []
        self._graph_info: Optional[GraphInfo] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._live: Optional[Live] = None
        self._last_sim_time = 0.0

    def setup(self, graph_info: GraphInfo) -> None:
        self._graph_info = graph_info
        fields = self._config.get("fields", None)
        if fields is None:
            fields = graph_info.node_state_fields

        self._tracked = []
        for node, field_list in fields.items():
            for f in field_list:
                self._tracked.append((node, f))

        self._precision = self._config.get("precision", 4)

    def _build_table(self, sim_time: float, state: dict[str, dict]) -> Table:
        """Build a Rich Table from the current state."""
        p = self._precision
        title = self._config.get("title", "MADDENING — Terminal Monitor")

        table = Table(title=title, show_header=True, min_width=44)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green", no_wrap=True)

        table.add_row("t", f"{sim_time:.{p}f} s")
        table.add_section()

        for node, field in self._tracked:
            if node in state and field in state[node]:
                val = float(state[node][field])
                table.add_row(f"{node}.{field}", f"{val:.{p}f}")

        return table

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        if self._live is not None:
            self._live.update(self._build_table(sim_time, state))

    def start_background(self, interval_ms: int = 50) -> None:
        """Poll the relay on a background thread with a Rich Live display.

        Call this when running alongside a matplotlib renderer whose
        event loop occupies the main thread.
        """
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, args=(interval_ms,), daemon=True,
        )
        self._thread.start()

    def run_event_loop(self, interval_ms: int = 50) -> None:
        """Block on the main thread, continuously refreshing.

        Press Ctrl-C to stop.
        """
        try:
            self._poll_loop(interval_ms)
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        """Signal the background polling thread to stop."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def teardown(self) -> None:
        self.stop()

    def requested_fields(self):
        return self._config.get("fields", None)

    # -- internal --

    def _poll_loop(self, interval_ms: int) -> None:
        interval_s = interval_ms / 1000.0
        refresh_hz = self._config.get("refresh_hz", 20)

        with Live(self._build_table(0.0, {}), refresh_per_second=refresh_hz) as live:
            self._live = live
            while not self._stop.is_set():
                sim_time, snapshot = self._relay.latest_snapshot()
                if snapshot is not None and sim_time != self._last_sim_time:
                    self._last_sim_time = sim_time
                    self.update(sim_time, snapshot)
                self._stop.wait(timeout=interval_s)
            self._live = None
