"""JAX device mesh creation for multi-GPU."""

from __future__ import annotations

import jax
from jax.sharding import Mesh


def create_device_mesh(n_devices: int | None = None) -> Mesh:
    """Create a 1-D JAX device mesh over available GPUs.

    Parameters
    ----------
    n_devices : int, optional
        Number of devices to use.  If ``None``, uses all available.

    Returns
    -------
    Mesh
        A 1-D mesh with axis name ``"devices"``.
    """
    devices = jax.devices()
    if n_devices is not None:
        if n_devices > len(devices):
            raise ValueError(
                f"Requested {n_devices} devices but only {len(devices)} available"
            )
        devices = devices[:n_devices]

    return Mesh(devices, axis_names=("devices",))
