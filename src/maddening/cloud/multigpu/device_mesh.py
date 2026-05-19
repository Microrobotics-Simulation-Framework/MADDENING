"""JAX device mesh creation for multi-GPU.

Supports 1-D meshes (slab decomposition, the v0.1 path) and N-D meshes
(pencil decomposition, the v0.2 path).  A 2-D mesh with axes
``("spatial_y", "spatial_z")`` is the canonical pencil layout: a 3-D
field with shape ``(nx, ny, nz)`` gets sharded along the two trailing
spatial axes, keeping ``nx`` (the streamwise direction for pressure-
driven LBM flow) contiguous on each device.
"""

from __future__ import annotations

from math import isqrt

import jax
import numpy as np
from jax.sharding import Mesh


def factor_devices(n: int) -> tuple[int, int]:
    """Return a square-ish 2-D factoring ``(py, pz)`` with ``py * pz == n``.

    Chooses the pair whose ratio is closest to 1 (e.g. 8 -> (2, 4),
    16 -> (4, 4), 12 -> (3, 4)).  Useful as the default pencil shape
    when the caller doesn't specify one.
    """
    if n < 1:
        raise ValueError(f"factor_devices: need n >= 1 (got {n})")
    py = isqrt(n)
    while py > 1 and n % py != 0:
        py -= 1
    pz = n // py
    return py, pz


def create_device_mesh(
    n_devices: int | None = None,
    *,
    shape: tuple[int, ...] | None = None,
    axis_names: tuple[str, ...] | None = None,
) -> Mesh:
    """Create a JAX device mesh over available devices.

    Parameters
    ----------
    n_devices : int, optional
        Number of devices to use.  If ``None``, derived from ``shape``
        when given, otherwise uses all available devices.
    shape : tuple[int, ...], optional
        Mesh shape (e.g. ``(2, 4)`` for an 8-device pencil mesh).  The
        product of ``shape`` must equal ``n_devices`` (or all available
        devices if ``n_devices`` is also ``None``).  When omitted the
        mesh is 1-D with all devices on a single axis.
    axis_names : tuple[str, ...], optional
        Names for each mesh axis.  Length must match ``len(shape)``.
        Defaults: ``("devices",)`` for 1-D; ``("spatial_y", "spatial_z")``
        for 2-D; ``("axis_0", "axis_1", ...)`` for higher dimensions.

    Returns
    -------
    Mesh
        A JAX ``Mesh`` with the requested shape and axis names.

    Examples
    --------
    1-D slab over all devices (legacy)::

        mesh = create_device_mesh()

    8-device pencil mesh (2x4)::

        mesh = create_device_mesh(shape=(2, 4))

    Custom axis names::

        mesh = create_device_mesh(
            shape=(4, 4), axis_names=("py", "pz")
        )
    """
    available = jax.devices()

    if shape is not None:
        product = 1
        for dim in shape:
            if dim < 1:
                raise ValueError(
                    f"create_device_mesh: shape={shape} has non-positive dim"
                )
            product *= dim
        if n_devices is None:
            n_devices = product
        elif n_devices != product:
            raise ValueError(
                f"create_device_mesh: n_devices={n_devices} does not match "
                f"prod(shape={shape})={product}"
            )

    if n_devices is None:
        n_devices = len(available)

    if n_devices > len(available):
        raise ValueError(
            f"Requested {n_devices} devices but only {len(available)} available"
        )

    devices = available[:n_devices]

    if shape is None:
        shape = (n_devices,)
        if axis_names is None:
            axis_names = ("devices",)
    elif axis_names is None:
        if len(shape) == 1:
            axis_names = ("devices",)
        elif len(shape) == 2:
            axis_names = ("spatial_y", "spatial_z")
        else:
            axis_names = tuple(f"axis_{i}" for i in range(len(shape)))

    if len(axis_names) != len(shape):
        raise ValueError(
            f"create_device_mesh: len(axis_names)={len(axis_names)} != "
            f"len(shape)={len(shape)}"
        )

    if len(set(axis_names)) != len(axis_names):
        raise ValueError(
            f"create_device_mesh: axis_names={axis_names} are not unique"
        )

    device_array = np.asarray(devices).reshape(shape)
    return Mesh(device_array, axis_names=axis_names)
