"""
StateRelay -- thread-safe snapshot buffer between simulation and renderers.

Hooks into ``GraphManager`` via its observer pattern.  The simulation
thread writes snapshots (fast, lock-protected reference swap); the
renderer polls ``latest_snapshot()`` at its own cadence.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


def _step_advance(graph_manager: Optional[GraphManager], last: float) -> float:
    """The simulated time the step a relay just observed advanced.

    The graph's :attr:`~maddening.core.graph_manager.GraphManager.timestep`
    read at the step -- the step ``compile()`` schedules, a sub-cycling
    group at its largest member timestep -- or *last* when there is no
    graph to ask.  Read per step rather than once at attach time, because a
    graph edited between runs (a node of another timestep added over the
    REST API) is recompiled by its next step and steps by a different
    amount from then on.
    """
    if graph_manager is None:
        return last
    try:
        return float(graph_manager.timestep)
    except RuntimeError:     # every node removed since the step was taken
        return last


class StateRelay:
    """Thread-safe one-slot buffer for the latest simulation state.

    Parameters
    ----------
    stride : int
        Only capture every *stride*-th step.  Default 1 (every step).
        Increasing stride reduces observer overhead for fast-stepping
        simulations where intermediate states are not needed.
    """

    def __init__(self, stride: int = 1) -> None:
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._timestep: float = 0.0
        # Simulated time of the observed steps, summed step by step: what
        # each step advanced, not ``step_count * <a step read once>``.
        self._elapsed: float = 0.0
        self._gm: Optional[GraphManager] = None
        self._stride: int = max(1, stride)

    @property
    def stride(self) -> int:
        return self._stride

    @stride.setter
    def stride(self, value: int) -> None:
        self._stride = max(1, int(value))

    def attach(self, graph_manager: GraphManager) -> None:
        """Register as an observer on *graph_manager*.

        ``sim_time`` is then the simulated time of the steps observed:
        each step adds the graph's :attr:`~GraphManager.timestep` as it
        stands at that step, so a sub-cycling group counts at its largest
        member timestep and a graph edited between runs keeps time.

        Raises
        ------
        RuntimeError
            If the graph has no nodes yet (it has no step to time).
        """
        self._timestep = graph_manager.timestep
        self._gm = graph_manager
        graph_manager.add_observer(self._on_event)

    def reset(self) -> None:
        """Forget every observed step: ``sim_time`` and the step count go
        back to zero and the snapshot to ``None`` (call it after resetting
        the graph's state)."""
        with self._lock:
            self._step_count = 0
            self._elapsed = 0.0
            self._sim_time = 0.0
            self._snapshot = None

    def _on_event(self, event: str, data) -> None:
        """Observer callback -- invoked on the simulation thread."""
        if event == "step":
            self._step_count += 1
            self._timestep = _step_advance(self._gm, self._timestep)
            self._elapsed += self._timestep
            if self._step_count % self._stride != 0:
                return
            with self._lock:
                # Shallow copy -- JAX arrays are immutable, safe to share
                self._snapshot = {
                    node: dict(fields) for node, fields in data.items()
                }
                self._sim_time = self._elapsed

    def latest_snapshot(self) -> tuple[float, Optional[dict]]:
        """Return ``(sim_time, state_dict_or_None)``.

        Called from the renderer thread.  Returns the most recent
        snapshot, or ``(0.0, None)`` if no step has been observed yet.
        """
        with self._lock:
            return (self._sim_time, self._snapshot)
