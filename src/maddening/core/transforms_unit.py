"""
Unit conversion transform factories for LBM <-> SI coupling.

Each factory takes physical parameters at construction time and returns
a JAX-traceable pure function suitable for ``EdgeSpec.transform``.  The
returned callable is a scalar multiplication -- fully differentiable,
JIT-compatible, and usable inside ``jax.lax.scan`` / ``fori_loop``.

All factories are re-exported from ``maddening.core.transforms`` for
discoverability::

    from maddening.core.transforms import lbm_to_si_force

The factories do **not** register transforms automatically.  If you need
a named, serializable transform for USD round-tripping, register the
returned callable yourself::

    from maddening.core.transforms import register_transform, lbm_to_si_force

    my_transform = lbm_to_si_force(dx=0.001, dt=1e-6, rho=1060.0)
    register_transform("my_lbm_force_to_si")(my_transform)

LBM lattice unit conversions
-----------------------------
LBM uses lattice units where ``dx = dt_lbm = 1``.  Physical conversion
factors (dimensional analysis):

==========  ======================================================
Quantity    Conversion factor (multiply LBM value to get SI)
==========  ======================================================
length      ``dx_physical``
time        ``dt_physical``
velocity    ``dx_physical / dt_physical``
force       ``rho_physical * dx_physical**4 / dt_physical**2``
torque      ``rho_physical * dx_physical**5 / dt_physical**2``
pressure    ``rho_physical * (dx_physical / dt_physical)**2``
==========  ======================================================
"""

from __future__ import annotations


def lbm_to_si_force(
    dx_physical: float,
    dt_physical: float,
    rho_physical: float,
):
    """Create a transform that converts LBM force to SI [N].

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].
    rho_physical : float
        Reference fluid density [kg/m^3].

    Returns
    -------
    callable
        ``(f_lbm,) -> f_si`` -- pure JAX function.
    """
    factor = float(rho_physical * dx_physical ** 4 / dt_physical ** 2)

    def _convert(f_lbm):
        return f_lbm * factor

    _convert.__qualname__ = (
        f"lbm_to_si_force(dx={dx_physical}, dt={dt_physical}, "
        f"rho={rho_physical})"
    )
    return _convert


def si_to_lbm_force(
    dx_physical: float,
    dt_physical: float,
    rho_physical: float,
):
    """Create a transform that converts SI [N] force to LBM lattice units.

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].
    rho_physical : float
        Reference fluid density [kg/m^3].

    Returns
    -------
    callable
        ``(f_si,) -> f_lbm`` -- pure JAX function.
    """
    factor = float(dt_physical ** 2 / (rho_physical * dx_physical ** 4))

    def _convert(f_si):
        return f_si * factor

    _convert.__qualname__ = (
        f"si_to_lbm_force(dx={dx_physical}, dt={dt_physical}, "
        f"rho={rho_physical})"
    )
    return _convert


def lbm_to_si_torque(
    dx_physical: float,
    dt_physical: float,
    rho_physical: float,
):
    """Create a transform that converts LBM torque to SI [N*m].

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].
    rho_physical : float
        Reference fluid density [kg/m^3].

    Returns
    -------
    callable
        ``(t_lbm,) -> t_si`` -- pure JAX function.
    """
    factor = float(rho_physical * dx_physical ** 5 / dt_physical ** 2)

    def _convert(t_lbm):
        return t_lbm * factor

    _convert.__qualname__ = (
        f"lbm_to_si_torque(dx={dx_physical}, dt={dt_physical}, "
        f"rho={rho_physical})"
    )
    return _convert


def lbm_to_si_velocity(
    dx_physical: float,
    dt_physical: float,
):
    """Create a transform that converts LBM velocity to SI [m/s].

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].

    Returns
    -------
    callable
        ``(v_lbm,) -> v_si`` -- pure JAX function.
    """
    factor = float(dx_physical / dt_physical)

    def _convert(v_lbm):
        return v_lbm * factor

    _convert.__qualname__ = (
        f"lbm_to_si_velocity(dx={dx_physical}, dt={dt_physical})"
    )
    return _convert


def lbm_to_si_pressure(
    dx_physical: float,
    dt_physical: float,
    rho_physical: float,
):
    """Create a transform that converts LBM pressure to SI [Pa].

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].
    dt_physical : float
        Physical timestep per LBM step [s].
    rho_physical : float
        Reference fluid density [kg/m^3].

    Returns
    -------
    callable
        ``(p_lbm,) -> p_si`` -- pure JAX function.
    """
    factor = float(rho_physical * (dx_physical / dt_physical) ** 2)

    def _convert(p_lbm):
        return p_lbm * factor

    _convert.__qualname__ = (
        f"lbm_to_si_pressure(dx={dx_physical}, dt={dt_physical}, "
        f"rho={rho_physical})"
    )
    return _convert


def lbm_to_si_length(dx_physical: float):
    """Create a transform that converts LBM length to SI [m].

    Parameters
    ----------
    dx_physical : float
        Physical lattice spacing [m].

    Returns
    -------
    callable
        ``(x_lbm,) -> x_si`` -- pure JAX function.
    """
    factor = float(dx_physical)

    def _convert(x_lbm):
        return x_lbm * factor

    _convert.__qualname__ = f"lbm_to_si_length(dx={dx_physical})"
    return _convert
