"""
RealtimeRunner -- run a simulation on a background daemon thread,
paced to wall-clock time.

Optionally reads external inputs from a ``CommandReceiver`` and
injects them into the simulation each step.
"""

import time
import threading
from typing import Optional


class RealtimeRunner:
    """Drive a ``GraphManager`` in real time on a background thread.

    Parameters
    ----------
    graph_manager
        A compiled (or compilable) ``GraphManager``.
    relay
        A ``StateRelay`` (or ``NetworkRelay``) attached to the graph manager.
    time_scale : float
        Ratio of sim-time to wall-time.  1.0 = real time, 2.0 = double
        speed, 0.5 = half speed, etc.
    steps_per_frame : int
        Number of physics steps to execute in a batch before pacing to
        wall clock.  Default 1.  Increasing this amortises sleep/wake
        overhead for fast-stepping simulations (e.g. dt=0.0001 physics
        rendered at 60 fps → steps_per_frame≈167).
    command_receiver : optional
        A ``CommandReceiver`` whose ``latest_commands()`` provides
        external inputs each step.  If ``None``, no external inputs
        are injected.
    """

    def __init__(
        self,
        graph_manager,
        relay,
        time_scale: float = 1.0,
        steps_per_frame: int = 1,
        command_receiver=None,
    ):
        self._gm = graph_manager
        self._relay = relay
        self._time_scale = time_scale
        self._steps_per_frame = max(1, steps_per_frame)
        self._cmd_recv = command_receiver
        self._paused = threading.Event()
        self._paused.set()  # starts unpaused
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sim_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start (or restart) the background simulation thread."""
        if self._gm._dirty or self._gm._compiled_step is None:
            self._gm.compile()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause the simulation.  The thread stays alive but blocks."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume a paused simulation."""
        self._paused.set()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop.set()
        self._paused.set()  # unblock if paused so the thread can exit
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset_time(self) -> None:
        """Reset simulation time to zero (call after resetting graph state)."""
        self._sim_time = 0.0

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        self._time_scale = max(0.01, value)

    @property
    def steps_per_frame(self) -> int:
        return self._steps_per_frame

    @steps_per_frame.setter
    def steps_per_frame(self, value: int) -> None:
        self._steps_per_frame = max(1, int(value))

    @property
    def sim_time(self) -> float:
        return self._sim_time

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main loop executed on the daemon thread."""
        dt = self._gm.timestep
        wall_start = time.perf_counter()
        sim_start = self._sim_time

        while not self._stop.is_set():
            self._paused.wait()
            if self._stop.is_set():
                break

            # Read external inputs from command receiver (if any)
            ext_inputs = None
            if self._cmd_recv is not None:
                ext_inputs = self._cmd_recv.latest_commands()

            # Batch-step: execute multiple physics steps before sleeping.
            # The relay (observer) still captures every step, but sleep
            # overhead is amortised.
            for _ in range(self._steps_per_frame):
                self._gm.step(external_inputs=ext_inputs)
                self._sim_time += dt

            # Pace to wall clock
            target_wall = wall_start + (self._sim_time - sim_start) / self._time_scale
            now = time.perf_counter()
            sleep_time = target_wall - now
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)
