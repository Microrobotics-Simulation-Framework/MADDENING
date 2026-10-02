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
from typing import TYPE_CHECKING, Any, Optional

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
        rendered at 60 fps → steps_per_frame≈167).  The stop and pause
        flags are checked between every step of a batch, so a large value
        does not delay :meth:`stop` or :meth:`pause` beyond one step.
    command_receiver : optional
        A ``CommandReceiver`` whose ``latest_commands()`` provides
        external inputs each step.  If ``None``, no external inputs
        are injected.
    lock : optional
        A lock (``threading.Lock`` / ``RLock``, anything with
        ``acquire(timeout=...)`` and ``release()``) held around every
        step, so that whoever else uses the graph -- the REST server's
        routes -- never sees it half-way through one.  It is taken one
        step at a time, never across a pause or a pacing sleep, and the
        thread waits for it in short slices that also watch the stop
        flag, so :meth:`stop` is answered within a step even while
        someone else holds it.  ``None`` (the default) takes no lock.
    max_catch_up : float
        How far behind its schedule, in wall-clock seconds, the runner
        catches up by stepping without sleeping.  Further behind -- after
        a pause, a long first compile, or a wait for *lock* -- it moves
        its schedule to the present instead, so the simulation does not
        burst through the time it lost.  Default 0.25 s.

    Attributes
    ----------
    error : str or None
        Why the thread stopped, when a step raised (``"ValueError: ..."``);
        ``None`` while it runs or after an ordinary :meth:`stop`.
    """

    def __init__(
        self,
        graph_manager: GraphManager,
        relay: StateRelay | NetworkRelay,
        time_scale: float = 1.0,
        steps_per_frame: int = 1,
        command_receiver: CommandReceiver | None = None,
        lock: Any = None,
        max_catch_up: float = 0.25,
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
        self._lock = lock
        self._max_catch_up = max(0.0, float(max_catch_up))
        self.error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start (or restart) the background simulation thread.

        Raises
        ------
        RuntimeError
            If the previous run's thread is still alive (a :meth:`stop`
            that timed out): two threads would step one graph.
        """
        if self.is_alive:
            raise RuntimeError(
                "RealtimeRunner: the previous run's thread is still alive "
                "(stop() timed out while it finished a step); call stop() "
                "again before starting"
            )
        if self._gm._dirty or self._gm._compiled_step is None:
            self._gm.compile()
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause the simulation.  The thread stays alive but blocks."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume a paused simulation."""
        self._paused.set()

    def stop(self, timeout: Optional[float] = 2.0) -> bool:
        """Signal the background thread to stop and wait for it.

        The thread checks the flag between every step, so it exits within
        one step of the signal (one step can still be long: the first one
        after a change compiles the graph).

        Parameters
        ----------
        timeout : float or None, optional
            Seconds to wait for the thread; ``None`` waits for as long as
            it takes.

        Returns
        -------
        bool
            ``True`` when no thread of this runner is alive any more;
            ``False`` when it is still finishing a step after *timeout*.
            The signal stays set, so it will exit; until then the graph is
            still being stepped, and a caller must not report the run
            stopped or write the state it steps.
        """
        self._stop.set()
        self._paused.set()  # unblock if paused so the thread can exit
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return not self.is_alive

    @property
    def is_alive(self) -> bool:
        """Whether this runner's thread is alive (running, paused, or still
        finishing a step after :meth:`stop`)."""
        return self._thread is not None and self._thread.is_alive()

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

    def _run(self) -> None:
        """The thread's target: :meth:`_loop`, with a step that raised
        recorded in :attr:`error` and logged, rather than left to
        ``threading.excepthook`` -- whoever drives the runner can then say
        why it stopped instead of reporting a runner that no longer
        exists as running."""
        try:
            self._loop()
        except BaseException as exc:  # noqa: BLE001 - recorded, then the thread ends
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("RealtimeRunner: the simulation thread stopped")

    def _acquire(self) -> bool:
        """Take :attr:`_lock` for one step; ``False`` when :meth:`stop` was
        asked for while waiting for it (the lock is then not held)."""
        if self._lock is None:
            return True
        while not self._lock.acquire(timeout=0.05):
            if self._stop.is_set():
                return False
        return True

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.release()

    def _loop(self) -> None:
        """Main loop executed on the daemon thread."""
        wall_start = time.perf_counter()
        sim_start = self._sim_time

        while not self._stop.is_set():
            if not self._paused.is_set():
                self._paused.wait()
                # Paced from the resume, not from the start: counted from
                # the start, the time spent paused read as time the
                # simulation had fallen behind, and a 3 s pause of a
                # real-time run was followed by ~3 s of simulated time in
                # a burst.
                wall_start = time.perf_counter()
                sim_start = self._sim_time
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
            # overhead is amortised.  The stop and pause flags are read
            # between steps: read only between frames, a batch of 1e8 steps
            # outlived stop()'s wait and the server reported "stopped"
            # while this thread kept stepping over the state a reset wrote.
            for _ in range(self._steps_per_frame):
                if self._stop.is_set() or not self._paused.is_set():
                    break
                if not self._acquire():
                    break
                try:
                    # Read again under the lock: a stop asked for while
                    # this thread waited for it must not cost a step.
                    if self._stop.is_set() or not self._paused.is_set():
                        break
                    self._gm.step(external_inputs=ext_inputs)
                    # What that step advanced, read after it: ``step()``
                    # recompiles a graph edited since the last frame (a
                    # node added over the REST API mid-run), so a value
                    # read once at start-up would go stale.
                    # ``gm.timestep`` is the step compile() schedules -- a
                    # sub-cycling group at its largest member timestep --
                    # and costs microseconds.
                    self._sim_time += self._gm.timestep
                finally:
                    self._release()

            # Pace to wall clock
            target_wall = wall_start + (self._sim_time - sim_start) / self._time_scale
            now = time.perf_counter()
            sleep_time = target_wall - now
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)
            elif -sleep_time > self._max_catch_up:
                # Too far behind to catch up without a visible burst:
                # schedule from here.
                wall_start = now
                sim_start = self._sim_time
