"""StaticArray — typed wrapper for arrays carried via the static_data channel.

v0.2 #3 follow-up.  An array stored in
:attr:`~maddening.core.node.SimulationNode.static_data` is a closure
target for the JIT'd step function, but unlike state arrays it is
NOT automatically sharded when the node is sharded across devices.
The node author has to declare *how* to handle the array under
sharding — replicate it across every device, or slice it along a
specific axis.

``StaticArray`` carries that declaration alongside the array
itself.  :class:`~maddening.cloud.multigpu.sharded_node.ShardedStencilNode`
materialises the per-device slice (via ``jax.device_put`` +
``NamedSharding``) and halo-exchanges it before calling
:meth:`~maddening.core.node.SimulationNode.update_padded`.  The
sliced + padded slab arrives as ``static_padded[<key>]`` inside
``update_padded``.  ``shard_axis`` is the array's own axis (not
a mesh axis), and must match one of the spatial axes the wrapping
``ShardedStencilNode`` shards via its ``axis_map.values()``.

Design contract (settled v0.2.0):

* **Required for array values**.  If you put a plain ndarray /
  jnp.ndarray in static_data, the framework emits a FutureWarning
  and coerces it to ``StaticArray(value=arr)`` with default
  ``replication="replicate"``.  In v0.3 the coercion path is
  removed; bare arrays will raise.
* **Optional for scalars / strings / tuples**.  Non-array values
  stay bare in static_data — they don't carry a sharding decision.
* **Nested structures are unsupported**.  Lists of arrays, dicts
  of arrays, sequences-of-StaticArray: all raise at construction.
  Unfold them into multiple top-level keys.
* **Not checkpointed**.  Sharding metadata (``replication``,
  ``shard_axis``) is part of the node's __init__ logic and
  reconstructed from ``self.params`` on reload.  The .npz
  manifest carries no record of it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.STABLE)
@dataclass(frozen=True)
class StaticArray:
    """A static (non-state) array with an explicit sharding policy.

    Parameters
    ----------
    value : numpy or jax array
        The array contents.  Built once at node construction time and
        held by reference; do not mutate after wrapping.
    replication : {"replicate", "shard"}
        How the GraphManager materialises this array under sharding.
        ``"replicate"`` (default): every device gets the full array.
        ``"shard"``: each device gets a slice along ``shard_axis``.
    shard_axis : int, optional
        Required when ``replication == "shard"``.  The axis of
        ``value`` to slice across the device mesh.  Must be ``None``
        when ``replication == "replicate"``.

    Raises
    ------
    TypeError
        If ``value`` is not array-like (no ``shape`` attribute), or
        is itself a sequence of arrays / mapping.
    ValueError
        If ``replication`` is unknown, or the ``replication`` /
        ``shard_axis`` combination is inconsistent.

    Examples
    --------
    Pure lookup table that every shard needs in full::

        @property
        def static_data(self):
            return {"weights": StaticArray(self._w)}  # replicate is default

    Grid coordinates sliced along the first axis when the node is
    sharded::

        @property
        def static_data(self):
            return {
                "grid_x": StaticArray(self._grid_x_array,
                                      replication="shard", shard_axis=0),
            }
    """

    value: Any
    replication: Literal["replicate", "shard"] = "replicate"
    shard_axis: Optional[int] = None

    def __post_init__(self) -> None:
        if not hasattr(self.value, "shape") or not hasattr(self.value, "dtype"):
            raise TypeError(
                f"StaticArray.value must be array-like with `shape` and "
                f"`dtype` (got {type(self.value).__name__}).  If you need "
                f"to carry a list or dict of arrays, unfold into multiple "
                f"static_data keys."
            )
        # Reject sequences / mappings explicitly even when they happen
        # to have a shape attribute (e.g. some array-of-array hybrids).
        if isinstance(self.value, (list, tuple, dict, set, frozenset)):
            raise TypeError(
                f"StaticArray does not accept nested {type(self.value).__name__}; "
                f"unfold into multiple keys in static_data."
            )

        if self.replication not in ("replicate", "shard"):
            raise ValueError(
                f"StaticArray.replication must be 'replicate' or 'shard' "
                f"(got {self.replication!r})."
            )

        if self.replication == "shard":
            if self.shard_axis is None:
                raise ValueError(
                    "StaticArray with replication='shard' requires an "
                    "explicit shard_axis (no default).",
                )
            if self.shard_axis < 0 or self.shard_axis >= len(self.value.shape):
                raise ValueError(
                    f"StaticArray.shard_axis={self.shard_axis} is out of "
                    f"range for an array with shape {tuple(self.value.shape)}.",
                )
        else:  # replicate
            if self.shard_axis is not None:
                raise ValueError(
                    "StaticArray.shard_axis must be None when "
                    "replication='replicate'.",
                )

    @property
    def shape(self) -> tuple:
        """The wrapped array's shape (proxy for `value.shape`)."""
        return tuple(int(d) for d in self.value.shape)

    @property
    def dtype(self) -> Any:
        """The wrapped array's dtype (proxy for `value.dtype`)."""
        return self.value.dtype


def coerce_static_data_value(value: Any, *, node_name: str, key: str) -> Any:
    """Wrap a bare array in ``StaticArray`` with a FutureWarning.

    Called from :func:`maddening.core.node._iter_static_data_for_hash`
    on every static_data access.  Scalars, strings, and tuples pass
    through unchanged.  Bare arrays emit a one-time FutureWarning
    (per node × key combination) and get coerced to
    ``StaticArray(value=arr)``.

    Removed in v0.3 — bare arrays in static_data will raise then.
    """
    if isinstance(value, StaticArray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        warnings.warn(
            f"Bare array in {node_name!r}.static_data[{key!r}] is "
            f"deprecated; wrap with StaticArray(value=..., "
            f"replication='replicate' or 'shard', shard_axis=...).  "
            f"Bare arrays will raise in v0.3.",
            FutureWarning,
            stacklevel=3,
        )
        return StaticArray(value=value)
    return value
