"""
Weight serialization for trained surrogate models.

Save and load surrogate weights to/from NPZ files, preserving the
architecture metadata needed to reconstruct a ``SurrogateNode``.

The format stores each weight array as a separate NPZ entry (``w0``,
``w1``, ...) plus a JSON metadata blob.  No pickle is used -- only
numpy arrays and JSON, ensuring safe cross-platform portability.
"""

import json
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.surrogates.architecture import SurrogateArchitecture


def save_weights(
    path: str,
    weights: Any,
    architecture: SurrogateArchitecture,
    state_spec: dict[str, tuple],
    boundary_spec: dict[str, tuple],
    metadata: Optional[dict] = None,
) -> None:
    """Save surrogate weights and metadata to an NPZ file.

    Parameters
    ----------
    path : str
        Output file path (e.g. ``"model.surr.npz"``).
    weights : PyTree
        The ``(arrays, static)`` weight tuple from training.
    architecture : SurrogateArchitecture
        The architecture instance (used for type name and mode).
    state_spec : dict
        State field specifications ``{field: shape}``.
    boundary_spec : dict
        Boundary input specifications ``{field: shape}``.
    metadata : dict, optional
        Extra metadata to store (e.g. losses, epoch number).
    """
    arrays, _static = weights
    leaves = jax.tree.leaves(arrays)

    save_dict = {f"w{i}": np.asarray(a) for i, a in enumerate(leaves)}

    meta = {
        "architecture_type": type(architecture).__name__,
        "architecture_mode": architecture.mode,
        "state_spec": {k: list(v) for k, v in state_spec.items()},
        "boundary_spec": {k: list(v) for k, v in boundary_spec.items()},
        "n_weights": len(leaves),
    }

    # Store architecture-specific config if available
    arch_config = {}
    if hasattr(architecture, "hidden_sizes"):
        arch_config["hidden_sizes"] = list(architecture.hidden_sizes)
    if hasattr(architecture, "n_basis"):
        arch_config["n_basis"] = architecture.n_basis
    if hasattr(architecture, "n_modes"):
        arch_config["n_modes"] = list(architecture.n_modes)
    if hasattr(architecture, "n_layers"):
        arch_config["n_layers"] = architecture.n_layers
    if hasattr(architecture, "hidden_dim"):
        arch_config["hidden_dim"] = architecture.hidden_dim
    if arch_config:
        meta["architecture_config"] = arch_config

    if metadata:
        meta["extra"] = metadata

    save_dict["_metadata"] = np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8,
    )
    np.savez(path, **save_dict)


def load_weights(
    path: str,
    architecture: SurrogateArchitecture,
    rng_key=None,
) -> tuple[Any, dict]:
    """Load surrogate weights from an NPZ file.

    Parameters
    ----------
    path : str
        Input file path.
    architecture : SurrogateArchitecture
        An architecture instance of the same type used during saving.
        Used to reconstruct the weight tree structure via ``init_params``.
    rng_key : PRNGKey, optional
        Random key for ``init_params`` (only used for tree structure;
        values are overwritten by loaded weights).

    Returns
    -------
    weights : PyTree
        The ``(arrays, static)`` weight tuple.
    metadata : dict
        Parsed metadata dict from the checkpoint file.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    data = np.load(path, allow_pickle=False)

    raw_meta = bytes(data["_metadata"])
    meta = json.loads(raw_meta)

    state_spec = {k: tuple(v) for k, v in meta["state_spec"].items()}
    boundary_spec = {k: tuple(v) for k, v in meta["boundary_spec"].items()}

    # Initialise dummy weights to get the tree structure
    dummy_weights = architecture.init_params(rng_key, state_spec, boundary_spec)
    dummy_arrays, static = dummy_weights

    # Load saved arrays and reconstruct the tree
    n_weights = meta["n_weights"]
    loaded_leaves = [jnp.array(data[f"w{i}"]) for i in range(n_weights)]
    tree_structure = jax.tree.structure(dummy_arrays)
    loaded_arrays = jax.tree.unflatten(tree_structure, loaded_leaves)

    return (loaded_arrays, static), meta


def load_train_result(
    path: str,
    architecture: SurrogateArchitecture,
    rng_key=None,
):
    """Load weights and return a ``TrainResult``.

    Convenience wrapper around :func:`load_weights` that returns a
    ``TrainResult`` ready for ``to_node()``.

    Parameters
    ----------
    path : str
        Input file path.
    architecture : SurrogateArchitecture
        Architecture instance.
    rng_key : PRNGKey, optional
        Random key for tree structure initialization.

    Returns
    -------
    TrainResult
    """
    from maddening.surrogates.trainer import TrainResult

    weights, meta = load_weights(path, architecture, rng_key)
    state_spec = {k: tuple(v) for k, v in meta["state_spec"].items()}
    boundary_spec = {k: tuple(v) for k, v in meta["boundary_spec"].items()}

    extra = meta.get("extra", {})
    return TrainResult(
        weights=weights,
        architecture=architecture,
        train_losses=extra.get("train_losses", []),
        val_losses=extra.get("val_losses", []),
        state_spec=state_spec,
        boundary_spec=boundary_spec,
    )
