"""
HistoryLogger -- an observer that accumulates state snapshots.

Attach to a :class:`GraphManager` via ``add_observer`` to record every
``"step"`` event.  The resulting :attr:`history` property returns arrays
with the same nested structure as :meth:`GraphManager.run_scan_with_history`,
making it a drop-in replacement when you need per-step callbacks (e.g.
dynamic external inputs) but still want stacked history arrays.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import jax.numpy as jnp

from maddening.core.graph_manager import EVENT_STEP


class HistoryLogger:
    """Observer that records state snapshots on each ``"step"`` event.

    Parameters
    ----------
    fields : dict[str, list[str]], optional
        If provided, only record the specified node/field combinations.
        For example, ``{"ball": ["position", "velocity"]}`` records only
        the ``position`` and ``velocity`` fields of the ``ball`` node.
        If ``None`` (default), all fields of all nodes are recorded.

    Usage::

        logger = HistoryLogger()
        gm.add_observer(logger)
        gm.run(100)
        history = logger.history  # dict[str, dict[str, jnp.ndarray]]
    """

    def __init__(
        self,
        fields: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._fields = fields
        self._records: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    # ------------------------------------------------------------------
    # Observer protocol
    # ------------------------------------------------------------------

    def __call__(self, event: str, data) -> None:
        """Called by :class:`GraphManager` as ``callback(event, data)``."""
        if event != EVENT_STEP:
            return

        state: dict[str, dict] = data

        for node_name, node_state in state.items():
            # Strip internal metadata
            if node_name == "_meta":
                continue

            if self._fields is not None and node_name not in self._fields:
                continue

            for field_name, value in node_state.items():
                if self._fields is not None:
                    if field_name not in self._fields[node_name]:
                        continue

                # Convert JAX scalar to Python float for efficient
                # accumulation (avoids holding many tiny device arrays).
                self._records[node_name][field_name].append(float(value))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def history(self) -> dict[str, dict[str, jnp.ndarray]]:
        """Return accumulated history as stacked JAX arrays.

        Returns the same nested-dict structure as
        :meth:`GraphManager.run_scan_with_history`:
        ``{node_name: {field_name: jnp.ndarray}}``, where each array
        has shape ``(n_steps,)`` (or ``(n_steps, *field_shape)`` for
        non-scalar fields).
        """
        result: dict[str, dict[str, jnp.ndarray]] = {}
        for node_name, fields in self._records.items():
            result[node_name] = {}
            for field_name, values in fields.items():
                result[node_name][field_name] = jnp.array(values)
        return result

    def reset(self) -> None:
        """Clear all recorded history so the logger can be reused."""
        self._records = defaultdict(lambda: defaultdict(list))

    def __len__(self) -> int:
        """Number of recorded timesteps (0 if nothing recorded)."""
        # All fields should have the same length; pick the first.
        for node_fields in self._records.values():
            for values in node_fields.values():
                return len(values)
        return 0

    def __repr__(self) -> str:
        n = len(self)
        nodes = list(self._records.keys())
        return f"HistoryLogger({n} steps, nodes={nodes})"
