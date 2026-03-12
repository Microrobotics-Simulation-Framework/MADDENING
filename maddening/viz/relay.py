"""
StateRelay -- thread-safe snapshot buffer between simulation and renderers.

Hooks into ``GraphManager`` via its observer pattern.  The simulation
thread writes snapshots (fast, lock-protected reference swap); the
renderer polls ``latest_snapshot()`` at its own cadence.
"""

import threading
from typing import Optional


class StateRelay:
    """Thread-safe one-slot buffer for the latest simulation state.

    Parameters
    ----------
    stride : int
        Only capture every *stride*-th step.  Default 1 (every step).
        Increasing stride reduces observer overhead for fast-stepping
        simulations where intermediate states are not needed.
    """

    def __init__(self, stride: int = 1):
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._timestep: float = 0.0
        self._stride: int = max(1, stride)

    @property
    def stride(self) -> int:
        return self._stride

    @stride.setter
    def stride(self, value: int) -> None:
        self._stride = max(1, int(value))

    def attach(self, graph_manager) -> None:
        """Register as an observer on *graph_manager*.

        Extracts the common timestep so we can compute ``sim_time``
        from the step count.
        """
        self._timestep = graph_manager.timestep
        graph_manager.add_observer(self._on_event)

    def _on_event(self, event: str, data) -> None:
        """Observer callback -- invoked on the simulation thread."""
        if event == "step":
            self._step_count += 1
            if self._step_count % self._stride != 0:
                return
            with self._lock:
                # Shallow copy -- JAX arrays are immutable, safe to share
                self._snapshot = {
                    node: dict(fields) for node, fields in data.items()
                }
                self._sim_time = self._step_count * self._timestep

    def latest_snapshot(self) -> tuple[float, Optional[dict]]:
        """Return ``(sim_time, state_dict_or_None)``.

        Called from the renderer thread.  Returns the most recent
        snapshot, or ``(0.0, None)`` if no step has been observed yet.
        """
        with self._lock:
            return (self._sim_time, self._snapshot)
