"""
RealtimeRunner -- run a simulation on a background daemon thread,
paced to wall-clock time.

Optionally reads external inputs from a ``CommandReceiver`` and
injects them into the simulation each step.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager
    from maddening.viz.network import CommandReceiver, NetworkRelay
    from maddening.viz.relay import StateRelay

logger = logging.getLogger(__name__)


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
        graph_manager: GraphManager,
        relay: StateRelay | NetworkRelay,
        time_scale: float = 1.0,
        steps_per_frame: int = 1,
        command_receiver: CommandReceiver | None = None,
    ) -> None:
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
        #: How many command dicts this runner has rejected (see
        #: :meth:`_accepted_commands`).  Readable from the control thread.
        self.rejected_commands: int = 0
        self._last_command_error: Optional[str] = None

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

    def _accepted_commands(self, commands):
        """*commands* if the graph will take them, else ``None``, logged.

        A command dict comes off the wire from a remote controller, so
        its node and field names are whatever the other end sent.  Since
        v0.4.0 ``GraphManager.step`` rejects a name the graph does not
        declare instead of dropping it in silence -- which is right, a
        control input that quietly does nothing is the worse failure --
        but a rejected input must not take the session with it.  A dead
        runner thread costs the operator every subsequent frame and says
        only that a thread died, whereas this says exactly which name was
        wrong and keeps stepping.

        Rejection falls back to ``None``, which is the documented
        "zeros for every declared input": the same thing the runner does
        when no receiver is attached, rather than a half-applied command.
        Repeats of the same complaint are logged once, so a controller
        stuck on a bad name cannot flood the log at frame rate.

        Only the validation is caught.  An exception from the physics is
        not an input problem and still stops the thread, because a step
        that cannot run is not something the next frame recovers from.
        """
        if commands is None:
            return None
        try:
            self._gm._resolve_external_inputs(commands)  # noqa: SLF001
        except (ValueError, TypeError, KeyError) as exc:
            self.rejected_commands += 1
            message = f"{type(exc).__name__}: {exc}"
            if message != self._last_command_error:
                self._last_command_error = message
                logger.error(
                    "rejected external command %r from the command receiver; "
                    "stepping with zeros for this frame instead.  %s",
                    commands, message,
                )
            return None
        self._last_command_error = None
        return commands

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
                ext_inputs = self._accepted_commands(
                    self._cmd_recv.latest_commands()
                )

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
