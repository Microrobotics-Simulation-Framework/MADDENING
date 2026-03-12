"""
Checkpoint / restore -- save and load simulation state to disk.

State is persisted as a NumPy ``.npz`` archive with flat keys of the
form ``node_name/field_name``.  Internal multi-rate metadata lives
under the ``_meta/`` prefix.  JAX arrays are converted to NumPy on
save and back to JAX on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager

# Key used by GraphManager for internal multi-rate bookkeeping.
_META_KEY = "_meta"


def save_state(graph_manager: "GraphManager", path: str | Path) -> Path:
    """Persist all node states (and ``_meta``) to an ``.npz`` file.

    Parameters
    ----------
    graph_manager : GraphManager
        The graph whose state should be saved.
    path : str or Path
        Destination file.  A ``.npz`` suffix is appended automatically by
        ``numpy.savez`` if not already present.

    Returns
    -------
    Path
        The resolved path of the written file (always ends in ``.npz``).
    """
    path = Path(path)

    arrays: dict[str, np.ndarray] = {}

    # Node states
    for node_name in graph_manager.node_names:
        node_state = graph_manager.get_node_state(node_name)
        for field_name, value in node_state.items():
            key = f"{node_name}/{field_name}"
            arrays[key] = np.asarray(value)

    # Internal _meta state (multi-rate step counter, etc.)
    # Access the raw internal state dict directly.
    raw_state = graph_manager._state  # noqa: SLF001
    if _META_KEY in raw_state:
        for field_name, value in raw_state[_META_KEY].items():
            key = f"{_META_KEY}/{field_name}"
            arrays[key] = np.asarray(value)

    np.savez(path, **arrays)

    # numpy.savez appends .npz if not already present
    resolved = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return resolved


def load_state(graph_manager: "GraphManager", path: str | Path) -> None:
    """Restore node states (and ``_meta``) from an ``.npz`` file.

    Parameters
    ----------
    graph_manager : GraphManager
        The graph whose state will be overwritten.
    path : str or Path
        Source file.  If *path* has no ``.npz`` extension and the file
        does not exist, the function retries with ``.npz`` appended.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found.
    ValueError
        If the saved state does not match the current graph structure
        (different node names or field names).
    """
    path = Path(path)
    if not path.exists() and path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    data = np.load(path, allow_pickle=False)

    # Separate meta keys from node keys.
    meta_keys: dict[str, np.ndarray] = {}
    node_keys: dict[str, dict[str, np.ndarray]] = {}

    for flat_key in data.files:
        parts = flat_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected key format in checkpoint: '{flat_key}' "
                f"(expected 'node_name/field_name')"
            )
        prefix, field = parts
        if prefix == _META_KEY:
            meta_keys[field] = data[flat_key]
        else:
            node_keys.setdefault(prefix, {})[field] = data[flat_key]

    # ---- Validate against current graph structure ----
    current_nodes = set(graph_manager.node_names)
    saved_nodes = set(node_keys.keys())

    if current_nodes != saved_nodes:
        missing = current_nodes - saved_nodes
        extra = saved_nodes - current_nodes
        parts = []
        if missing:
            parts.append(f"missing from checkpoint: {sorted(missing)}")
        if extra:
            parts.append(f"extra in checkpoint: {sorted(extra)}")
        raise ValueError(
            f"Checkpoint node mismatch. {'; '.join(parts)}"
        )

    for node_name in current_nodes:
        current_fields = set(graph_manager.get_node_state(node_name).keys())
        saved_fields = set(node_keys[node_name].keys())
        if current_fields != saved_fields:
            raise ValueError(
                f"Field mismatch for node '{node_name}': "
                f"current={sorted(current_fields)}, "
                f"saved={sorted(saved_fields)}"
            )

    # ---- Apply loaded state ----
    for node_name in current_nodes:
        new_state = {
            field: jnp.array(arr) for field, arr in node_keys[node_name].items()
        }
        graph_manager.set_node_state(node_name, new_state)

    # Restore _meta if present in the checkpoint.
    raw_state = graph_manager._state  # noqa: SLF001
    if meta_keys:
        raw_state[_META_KEY] = {
            field: jnp.array(arr) for field, arr in meta_keys.items()
        }
    else:
        # If checkpoint had no meta but graph currently has it, reset.
        raw_state.pop(_META_KEY, None)
