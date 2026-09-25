"""
LBMPipeNode -- 3D Lattice Boltzmann fluid in a cylindrical pipe with propeller.

Implements the D3Q19 lattice Boltzmann method for incompressible flow in a
pipe geometry.  A propeller is modelled as an actuator disc (body force at a
fixed axial cross-section) rather than as moving geometry.

The entire ``update`` uses ``jnp`` operations (no Python control flow that
depends on array values), so it is fully JAX-traceable and JIT-compilable.

Lattice units are used internally: dx = dt_lbm = 1.  The kinematic viscosity
is controlled by the BGK relaxation time tau via::

    nu = (tau - 0.5) / 3

Multiphase mode
---------------
When ``G != 0``, the node uses the **Shan-Chen pseudopotential** method for
multiphase flow.  An inter-particle interaction force drives spontaneous
phase separation into high-density (liquid) and low-density (gas) phases.
Surface tension, waves, and ripples emerge naturally from the interaction
force gradients at the interface.

The pseudopotential uses the Yuan-Schaefer form for better thermodynamic
consistency::

    psi(rho) = rho_0 * (1 - exp(-rho / rho_0))

When ``G == 0`` (default), the node behaves identically to the original
single-phase LBM with passive scalar tracer.

State fields
------------
f : (nx, ny, nz, 19) float32
    D3Q19 particle distribution functions.
density : (nx, ny, nz) float32
    Macroscopic density (sum of f over 19 directions).
velocity : (nx, ny, nz, 3) float32
    Macroscopic velocity (x, y, z components).
tracer : (nx, ny, nz) float32
    Phase indicator (0=gas, 1=liquid).  In multiphase mode this is derived
    from density; in single-phase mode it is advected via D3Q7 LBM.
tracer_f : (nx, ny, nz, 7) float32
    D3Q7 distribution for the passive tracer (single-phase mode only;
    frozen in multiphase mode).

Boundary inputs
---------------
propeller_force : scalar, optional
    Override the propeller body-force strength.  Positive = flow in +x.
"""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec

# ── D3Q19 lattice constants ─────────────────────────────────────────
# Use numpy (not jnp) so these remain concrete values inside JIT/scan.
# JAX auto-promotes them when used in jnp operations.

# 19 velocity vectors: (ex, ey, ez)
# Order: rest, ±x, ±y, ±z, then 12 edge-diagonals
_E = np.array([
    [0, 0, 0],    # 0  rest
    [1, 0, 0],    # 1  +x
    [-1, 0, 0],   # 2  -x
    [0, 1, 0],    # 3  +y
    [0, -1, 0],   # 4  -y
    [0, 0, 1],    # 5  +z
    [0, 0, -1],   # 6  -z
    [1, 1, 0],    # 7
    [-1, 1, 0],   # 8
    [1, -1, 0],   # 9
    [-1, -1, 0],  # 10
    [1, 0, 1],    # 11
    [-1, 0, 1],   # 12
    [1, 0, -1],   # 13
    [-1, 0, -1],  # 14
    [0, 1, 1],    # 15
    [0, -1, 1],   # 16
    [0, 1, -1],   # 17
    [0, -1, -1],  # 18
], dtype=np.int32)

# Weights
_W = np.array([
    1.0 / 3.0,                          # rest
    1.0 / 18.0, 1.0 / 18.0,             # ±x
    1.0 / 18.0, 1.0 / 18.0,             # ±y
    1.0 / 18.0, 1.0 / 18.0,             # ±z
    1.0 / 36.0, 1.0 / 36.0,             # edge xy
    1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0,             # edge xz
    1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0,             # edge yz
    1.0 / 36.0, 1.0 / 36.0,
], dtype=np.float32)

# Opposite direction index for each of the 19 directions (for bounce-back)
_OPP = np.array([
    0,   # rest → rest
    2, 1,   # +x ↔ -x
    4, 3,   # +y ↔ -y
    6, 5,   # +z ↔ -z
    10, 9, 8, 7,       # edge xy opposites
    14, 13, 12, 11,    # edge xz opposites
    18, 17, 16, 15,    # edge yz opposites
], dtype=np.int32)

# Speed of sound squared in lattice units
_CS2 = 1.0 / 3.0


# ── D3Q7 lattice constants (passive scalar transport) ──────────────

_E7 = np.array([
    [0, 0, 0],    # 0  rest
    [1, 0, 0],    # 1  +x
    [-1, 0, 0],   # 2  -x
    [0, 1, 0],    # 3  +y
    [0, -1, 0],   # 4  -y
    [0, 0, 1],    # 5  +z
    [0, 0, -1],   # 6  -z
], dtype=np.int32)

_W7 = np.array([
    1.0 / 4.0,
    1.0 / 8.0, 1.0 / 8.0,
    1.0 / 8.0, 1.0 / 8.0,
    1.0 / 8.0, 1.0 / 8.0,
], dtype=np.float32)

_OPP7 = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int32)


# ── Helper: equilibrium distribution ────────────────────────────────

def _equilibrium(density, velocity):
    """Compute equilibrium distribution f_eq for D3Q19.

    Parameters
    ----------
    density : (nx, ny, nz) float32
    velocity : (nx, ny, nz, 3) float32

    Returns
    -------
    f_eq : (nx, ny, nz, 19) float32
    """
    # e_dot_u: (nx, ny, nz, 19)  —  velocity @ E^T
    e_dot_u = velocity @ _E.astype(np.float32).T
    u_sq = jnp.sum(velocity ** 2, axis=-1, keepdims=True)  # (..., 1)

    # f_eq_q = w_q * rho * (1 + e·u/cs2 + (e·u)^2/(2*cs4) - u·u/(2*cs2))
    f_eq = _W * density[..., None] * (
        1.0
        + e_dot_u / _CS2
        + e_dot_u ** 2 / (2.0 * _CS2 ** 2)
        - u_sq / (2.0 * _CS2)
    )
    return f_eq


def _compute_macroscopic(f):
    """Extract density and velocity from distribution functions.

    Parameters
    ----------
    f : (nx, ny, nz, 19) float32

    Returns
    -------
    density : (nx, ny, nz) float32
    velocity : (nx, ny, nz, 3) float32
    """
    density = jnp.sum(f, axis=-1)
    # momentum = sum_q f_q * e_q  —  f @ E
    momentum = f @ _E.astype(np.float32)
    # Avoid division by zero for empty cells (shouldn't happen in practice)
    velocity = momentum / jnp.maximum(density[..., None], 1e-10)
    return density, velocity


def _stream(f):
    """Streaming step: shift each f_q by its lattice velocity.

    Uses ``jnp.roll`` along each axis.  The Python ``for`` loop over
    19 directions unrolls at JAX trace time.

    Parameters
    ----------
    f : (nx, ny, nz, 19) float32

    Returns
    -------
    f_streamed : (nx, ny, nz, 19) float32
    """
    slices = []
    for q in range(19):
        fq = f[..., q]
        ex, ey, ez = int(_E[q, 0]), int(_E[q, 1]), int(_E[q, 2])
        if ex != 0:
            fq = jnp.roll(fq, ex, axis=0)
        if ey != 0:
            fq = jnp.roll(fq, ey, axis=1)
        if ez != 0:
            fq = jnp.roll(fq, ez, axis=2)
        slices.append(fq)
    return jnp.stack(slices, axis=-1)


def _guo_forcing(density, velocity, force, tau):
    """Guo forcing term for the BGK collision operator.

    Implements the Guo et al. (2002) forcing scheme::

        S_q = (1 - 1/(2*tau)) * w_q * [
            (e_q - u)/cs2 + (e_q · u)/(cs4) * e_q
        ] · F

    Parameters
    ----------
    density : (nx, ny, nz) float32
    velocity : (nx, ny, nz, 3) float32
    force : (nx, ny, nz, 3) float32
        Body force per unit volume.
    tau : float

    Returns
    -------
    S : (nx, ny, nz, 19) float32
    """
    e_f = _E.astype(np.float32)  # (19, 3)

    # (e_q - u): broadcast to (nx, ny, nz, 19, 3)
    e_minus_u = e_f[None, None, None, :, :] - velocity[..., None, :]
    # (e_q · u): (nx, ny, nz, 19)  —  velocity @ E^T
    e_dot_u = velocity @ e_f.T
    # e_q scaled by e·u / cs4: (nx, ny, nz, 19, 3)
    e_scaled = e_f[None, None, None, :, :] * (e_dot_u / (_CS2 ** 2))[..., None]

    bracket = (e_minus_u / _CS2 + e_scaled)  # (nx, ny, nz, 19, 3)
    # Dot with force: (nx, ny, nz, 19)
    S = (1.0 - 0.5 / tau) * _W * jnp.sum(bracket * force[..., None, :], axis=-1)
    return S


# ── D3Q7 passive scalar helpers ────────────────────────────────────

def _tracer_equilibrium(tracer, velocity):
    """Compute D3Q7 equilibrium for a passive scalar field.

    Parameters
    ----------
    tracer : (nx, ny, nz) float32
        Scalar concentration.
    velocity : (nx, ny, nz, 3) float32
        Flow velocity (from the main D3Q19 LBM).

    Returns
    -------
    g_eq : (nx, ny, nz, 7) float32
    """
    e_dot_u = velocity @ _E7.astype(np.float32).T  # (nx, ny, nz, 7)
    return _W7 * tracer[..., None] * (1.0 + e_dot_u / _CS2)


def _stream7(g):
    """D3Q7 streaming for the passive scalar distribution.

    Parameters
    ----------
    g : (nx, ny, nz, 7) float32

    Returns
    -------
    g_streamed : (nx, ny, nz, 7) float32
    """
    slices = []
    for q in range(7):
        gq = g[..., q]
        ex, ey, ez = int(_E7[q, 0]), int(_E7[q, 1]), int(_E7[q, 2])
        if ex != 0:
            gq = jnp.roll(gq, ex, axis=0)
        if ey != 0:
            gq = jnp.roll(gq, ey, axis=1)
        if ez != 0:
            gq = jnp.roll(gq, ez, axis=2)
        slices.append(gq)
    return jnp.stack(slices, axis=-1)


# ── Shan-Chen pseudopotential helpers ──────────────────────────────

def _psi(density, rho_0):
    """Yuan-Schaefer pseudopotential.

    ``psi(rho) = rho_0 * (1 - exp(-rho / rho_0))``

    Better thermodynamic consistency and numerical stability than the
    original Shan-Chen ``exp(-rho_0 / rho)`` form.

    Parameters
    ----------
    density : array  — macroscopic density field.
    rho_0 : float    — reference density parameter.

    Returns
    -------
    psi : same shape as *density*.
    """
    return rho_0 * (1.0 - jnp.exp(-density / rho_0))


def _shan_chen_force(density, G, rho_0, wall_mask, rho_wall):
    """Compute Shan-Chen interaction force on the D3Q19 lattice.

    ``F_int(x) = -G * psi(x) * sum_q  w_q * psi(x + e_q) * e_q``

    Parameters
    ----------
    density : (nx, ny, nz) float32
    G : float
        Interaction strength (negative → attraction → phase separation).
    rho_0 : float
        Pseudopotential reference density.
    wall_mask : (nx, ny, nz) bool
        True at solid wall cells.
    rho_wall : float
        Density used for the pseudopotential at wall cells.  Controls
        wall wetting behaviour.

    Returns
    -------
    force : (nx, ny, nz, 3) float32
    """
    e_f = _E.astype(np.float32)  # (19, 3)

    # Pseudopotential field — set wall cells to psi(rho_wall) to control
    # wetting and prevent spurious interface forces at pipe boundaries.
    # jnp, not np: rho_0 / rho_wall are traced graph parameters.
    psi_wall = rho_0 * (1.0 - jnp.exp(-rho_wall / rho_0))
    psi_field = jnp.where(wall_mask, psi_wall, _psi(density, rho_0))

    # Weighted sum of shifted psi * e_q.
    # Note: roll(-ex) gives psi[i+ex] = psi(x+e_q).  This is the
    # OPPOSITE sign from the streaming roll (which uses +ex to pick up
    # the incoming particle from x - e_q).
    grad = jnp.zeros(density.shape + (3,), dtype=jnp.float32)
    for q in range(19):
        w_q = float(_W[q])
        if w_q == 0:
            continue
        psi_shifted = psi_field
        ex, ey, ez = int(_E[q, 0]), int(_E[q, 1]), int(_E[q, 2])
        if ex != 0:
            psi_shifted = jnp.roll(psi_shifted, -ex, axis=0)
        if ey != 0:
            psi_shifted = jnp.roll(psi_shifted, -ey, axis=1)
        if ez != 0:
            psi_shifted = jnp.roll(psi_shifted, -ez, axis=2)
        # w_q * psi(x+e_q) * e_q  — accumulate into (nx, ny, nz, 3)
        grad = grad + (w_q * psi_shifted)[..., None] * e_f[q]

    # F_int = -G * psi(x) * grad
    force = -G * psi_field[..., None] * grad

    # Zero force at wall cells
    force = jnp.where(wall_mask[..., None], 0.0, force)
    return force


def _eos_pressure(density, G, rho_0):
    """Shan-Chen equation of state pressure.

    ``P = rho * cs^2 + G * cs^2 / 2 * psi(rho)^2``

    Useful for computing coexistence densities via the Maxwell
    equal-area construction, but not used in the update loop.
    """
    psi = _psi(density, rho_0)
    return density * _CS2 + G * _CS2 / 2.0 * psi ** 2


# ── LBMPipeNode ─────────────────────────────────────────────────────

@stability(StabilityLevel.EXPERIMENTAL)
class LBMPipeNode(SimulationNode):
    """3D Lattice Boltzmann fluid in a cylindrical pipe with propeller.

    Uses the D3Q19 lattice with BGK collision and bounce-back wall
    boundaries.  The pipe is aligned along the x-axis with a circular
    cross-section.  Flow is driven by an actuator-disc propeller (body
    force at a specified axial plane).  Periodic boundary conditions
    in x.

    When ``G != 0``, the **Shan-Chen pseudopotential** method is enabled
    for multiphase flow with surface tension, waves, and phase
    separation.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep (seconds).  One LBM step per timestep.
    nx, ny, nz : int
        Grid dimensions.  ``nx`` is the pipe length (flow direction).
    tau : float
        BGK relaxation time.  Must be finite and > 0.5 for stability.
        Kinematic viscosity: ``nu = (tau - 0.5) / 3``.
    pipe_radius : float
        Pipe radius as fraction of ``min(ny, nz) / 2`` (0-1).
    propeller_x : int
        Axial position of the propeller disc (grid index,
        ``0 <= propeller_x < nx``).  The default, 10, therefore needs
        ``nx > 10``; a position outside the grid is refused (until 0.4.0 it
        built no disc, and the pipe ran with no propeller).
    propeller_radius : float
        Propeller disc radius as fraction of pipe radius (0-1).  A disc
        that covers no cell of the cross-section is refused.
    propeller_strength : float
        Body force magnitude applied at the propeller disc.
    initial_velocity : float
        Initial x-velocity throughout the pipe (lattice units).
    gravity : float
        Body force in the -z direction applied to all fluid cells.
        Useful for partially-filled pipes.  Default 0.0 (no gravity).
    fill_fraction : float
        Fraction of the pipe cross-section initially filled with liquid
        (0-1).  Default 1.0 (fully filled).
    tau_tracer : float
        BGK relaxation time for the passive scalar D3Q7 transport.
        Only used in single-phase mode (``G == 0``).  Default 0.6.
    G : float
        Shan-Chen interaction strength.  ``0.0`` = single-phase (default).
        Negative values cause phase separation.  For BGK collision with
        Yuan-Schaefer pseudopotential (rho_0=1), the critical G is -4.0.
        Recommended range: -4.2 to -5.0 (higher magnitude = larger
        density ratio, but harder to stabilise).
    rho_liquid : float
        Liquid-phase density (multiphase mode).  Default 1.0.  Should be
        close to the EOS coexistence density for the chosen G (roughly
        0.8-1.5 for G in [-4.2, -5.0] with rho_0=1).  The step reads it
        as the liquid reference of the phase indicator ``tracer``, and
        it is a trainable parameter there.  It also supplies the
        defaults of ``initial_rho_liquid`` and ``rho_wall``, resolved
        once, here.
    rho_gas : float
        Gas-phase density (multiphase mode).  Default 0.25.  Should be
        close to the EOS coexistence density for the chosen G (roughly
        0.15-0.45 for G in [-4.2, -5.0] with rho_0=1).  The gas
        reference of ``tracer`` (trainable), and the default of
        ``initial_rho_gas``.
    rho_0 : float
        Pseudopotential reference density.  Default 1.0.
    rho_wall : float or None
        Wall pseudopotential density for wetting control.  ``None``
        defaults to ``rho_liquid`` (fully wetted / hydrophilic wall).  The
        default is resolved at construction and stored, so a later change
        to ``rho_liquid`` through the graph's params does not move it.
    initial_rho_liquid, initial_rho_gas : float or None
        The liquid and gas densities of the initial condition (multiphase
        mode): the two ends of the ``tanh`` fill profile, and the density
        of wall cells (``initial_rho_gas``).  ``None`` takes
        ``rho_liquid`` / ``rho_gas``.  Initial conditions, not dynamics
        constants: ``initial_state`` reads them, the step never does, and
        they are not trainable.

    Notes
    -----
    Why the initial densities are separate leaves.  ``initial_state``
    runs when the graph compiles, from the constructor's values; a value
    set later through ``GraphManager.params`` reaches the step only.
    When ``rho_liquid`` / ``rho_gas`` were both the initial-condition
    recipe and a trainable step constant, a graph calibrated through
    ``params`` and the graph its saved config rebuilt started from
    different initial densities (0.30 apart after three steps, with no
    warning).  The initial condition now reads only the ``initial_*``
    leaves, which ``to_dict`` saves alongside the calibrated constants, so
    the two graphs are the same graph.
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-006",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="D3Q19 Lattice Boltzmann in cylindrical pipe with propeller actuator disc",
        governing_equations="BGK collision: f_i = f_i - (f_i - f_eq_i)/τ; streaming: f_i(x+e_i, t+1) = f_i(x, t)",
        # No order is claimed.  The only convergence study (MADD-VER-013)
        # is outside the asymptotic range -- its two triples measure 1.49
        # and 3.35 -- and bounce-back on the staircased circular wall is
        # 1st-order, so "2nd-order in space and time", which this said
        # until 0.4.0, was the bulk scheme's textbook order and not a
        # property anything measured of this node.
        discretization=(
            "Lattice Boltzmann (D3Q19, BGK, explicit; one lattice step per "
            "timestep).  No order of accuracy is claimed: the one "
            "convergence study (MADD-VER-013) is not in the asymptotic range"
        ),
        assumptions=(
            "Incompressible flow (Mach number << 1)",
            "BGK single-relaxation-time collision operator",
            "Rigid, impermeable pipe walls (bounce-back)",
            "Periodic boundary conditions in flow direction",
            "Actuator disc model for propeller (body force, not geometry)",
        ),
        limitations=(
            "Compressibility errors at high Mach number (Ma > 0.1)",
            "BGK is less stable than MRT for high Reynolds numbers",
            "No turbulence model — results unreliable above Re ~2000",
            "Wall bounce-back is 1st-order at curved boundaries",
            "No accuracy verdict: the one convergence study (MADD-VER-013) "
            "is not in the asymptotic range, and no test compares the node "
            "against an analytical solution with a pass criterion",
            "The verified configuration is single-phase, fully filled and "
            "uniformly forced (MADD-VER-013).  No verification benchmark "
            "covers the Shan-Chen multiphase mode, the passive tracer, "
            "partial fill, gravity or a localised actuator disc",
        ),
        validated_regimes=(
            ValidatedRegime(
                "tau", 0.8, 0.8,
                notes=(
                    "The only value any verification runs at: MADD-VER-013 "
                    "uses tau = 0.8 (nu = 0.1).  The constructor accepts any "
                    "tau > 0.5, which BGK needs to be stable, but nothing "
                    "verifies the node at another value; tau near 0.5 is "
                    "prone to instability and tau >> 1 adds numerical "
                    "diffusion.  0.501 to 2.0, the range this declared "
                    "until 0.4.0, was never run"
                ),
            ),
            ValidatedRegime(
                "Reynolds number", 0, 0.03,
                notes=(
                    "Re = u_mean * 2R / nu.  The only verification runs, "
                    "MADD-VER-013's uniformly forced pipe flow (tau = 0.8, "
                    "body force 1e-6), are in the Stokes regime: Re <= 0.013 "
                    "on the 12/16/24 ladder and 0.030 at 32.  The shape "
                    "factor u_max/u_mean approaches the Hagen-Poiseuille "
                    "value 2 from below (1.79, 1.86, 1.92, 1.93), but no "
                    "test holds the node to an analytical solution"
                ),
            ),
            ValidatedRegime(
                "grid", 12, 32,
                notes=(
                    "Cross-section cells per side (ny = nz, nx = 1) of "
                    "MADD-VER-013.  The 12/16/24 ladder converges "
                    "monotonically (observed order 1.49, GCI 10.7% at "
                    "Fs = 3.0); adding 32 shows it is not in the asymptotic "
                    "range (per-triple orders 1.49 and 3.35), so this is a "
                    "convergence statement, not an accuracy one.  No "
                    "convergence study runs above 32 per side"
                ),
            ),
        ),
        hazard_hints=(
            "Behaviour uncharacterised above Re ~0.03 and at any tau but 0.8; verified only in the Stokes regime, at tau = 0.8, on 12 to 32 cells per side (MADD-VER-013), not across the laminar range",
            "No turbulence model — do not use above laminar-turbulent transition (Re ~2000)",
            "Wall bounce-back assumes rigid, impermeable walls; deformable or porous walls not modelled",
            "Gravity applied uniformly — no spatially varying body forces",
            "Passive scalar tracer uses D3Q7 with separate tau — accuracy degrades at high Peclet number",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        nx: int = 64,
        ny: int = 32,
        nz: int = 32,
        tau: float = 0.8,
        pipe_radius: float = 0.9,
        propeller_x: int = 10,
        propeller_radius: float = 0.8,
        propeller_strength: float = 0.0005,
        initial_velocity: float = 0.0,
        gravity: float = 0.0,
        fill_fraction: float = 1.0,
        tau_tracer: float = 0.6,
        G: float = 0.0,
        rho_liquid: float = 1.0,
        rho_gas: float = 0.25,
        rho_0: float = 1.0,
        rho_wall: float | None = None,
        initial_rho_liquid: float | None = None,
        initial_rho_gas: float | None = None,
    ):
        # ``tau <= 0.5`` alone let a non-finite value through: inf passes
        # it (no collision at all) and nan passes every comparison.
        if not np.isfinite(tau):
            raise ValueError(f"tau must be a finite number > 0.5 (got {tau}).")
        if tau <= 0.5:
            raise ValueError(
                f"tau must be > 0.5 for stability (got {tau}). "
                f"nu = (tau - 0.5) / 3 = {(tau - 0.5) / 3:.4f}"
            )
        if not np.isfinite(tau_tracer):
            raise ValueError(
                f"tau_tracer must be a finite number > 0.5 (got {tau_tracer})."
            )
        if tau_tracer <= 0.5:
            raise ValueError(
                f"tau_tracer must be > 0.5 for stability (got {tau_tracer})."
            )
        if not 0.0 < fill_fraction <= 1.0:
            raise ValueError(
                f"fill_fraction must be in (0, 1] (got {fill_fraction})."
            )
        # The actuator disc is one x-plane of the grid.  An index outside
        # it used to build no disc at all (see the mask check below); the
        # default, 10, is outside any pipe of 10 cells or fewer.
        try:
            prop_plane = operator.index(propeller_x)
        except TypeError:
            raise ValueError(
                f"propeller_x must be an integer grid index along x (got "
                f"{propeller_x!r})."
            ) from None
        if not 0 <= prop_plane < nx:
            raise ValueError(
                f"propeller_x={propeller_x} is outside the grid: the pipe has "
                f"nx={nx} planes along x, indexed 0 to {nx - 1}, so the "
                "propeller disc would cover no cell and exert no force.  Pass "
                f"a propeller_x in [0, {nx})"
                + (" (the default, 10, needs nx > 10)." if propeller_x == 10
                   else ".")
            )
        if initial_rho_liquid is None:
            initial_rho_liquid = rho_liquid
        if initial_rho_gas is None:
            initial_rho_gas = rho_gas
        if G != 0.0:
            for liquid_name, liquid, gas_name, gas in (
                ("rho_liquid", rho_liquid, "rho_gas", rho_gas),
                ("initial_rho_liquid", initial_rho_liquid,
                 "initial_rho_gas", initial_rho_gas),
            ):
                if liquid <= gas:
                    raise ValueError(
                        f"{liquid_name} must be > {gas_name} "
                        f"(got {liquid} <= {gas})."
                    )
                if gas <= 0.0:
                    raise ValueError(f"{gas_name} must be > 0 (got {gas}).")
            if rho_0 <= 0.0:
                raise ValueError(f"rho_0 must be > 0 (got {rho_0}).")

        if rho_wall is None:
            rho_wall = rho_liquid

        super().__init__(
            name,
            timestep,
            nx=nx, ny=ny, nz=nz,
            tau=tau,
            pipe_radius=pipe_radius,
            propeller_x=propeller_x,
            propeller_radius=propeller_radius,
            propeller_strength=propeller_strength,
            initial_velocity=initial_velocity,
            gravity=gravity,
            fill_fraction=fill_fraction,
            tau_tracer=tau_tracer,
            G=G,
            rho_liquid=rho_liquid,
            rho_gas=rho_gas,
            rho_0=rho_0,
            rho_wall=rho_wall,
            initial_rho_liquid=initial_rho_liquid,
            initial_rho_gas=initial_rho_gas,
        )

        # Pre-compute masks and store as JAX arrays
        self._wall_mask = self._build_wall_mask(nx, ny, nz, pipe_radius)
        self._propeller_mask = self._build_propeller_mask(
            nx, ny, nz, propeller_x, propeller_radius, pipe_radius,
        )
        # The disc must cover a cell.  ``mask.at[propeller_x]`` drops an
        # index outside the grid without a word, and a radius that takes in
        # no cell centre builds an empty disc, so either ran a pipe with no
        # propeller at all: propeller_x=99 on nx=8 left max|u| at 3.7e-9
        # after 10 steps, where propeller_x=4 reached 5.2e-3.
        if not bool(jnp.any(self._propeller_mask)):
            raise ValueError(
                f"LBMPipeNode {name!r}: the propeller disc (propeller_radius="
                f"{propeller_radius} of the pipe radius, pipe_radius="
                f"{pipe_radius}) covers no cell of the {ny}x{nz} cross-section, "
                "so the propeller would exert no force.  Increase "
                "propeller_radius (a fraction of the pipe radius, 0-1) or use "
                "a larger cross-section."
            )
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._tau = tau
        self._tau_tracer = tau_tracer
        self._gravity = gravity
        self._fill_fraction = fill_fraction
        self._G = G
        # The initial condition's densities.  ``rho_liquid`` / ``rho_gas``
        # themselves are read by the step (from the injected params) and
        # never by ``initial_state``.
        self._initial_rho_liquid = initial_rho_liquid
        self._initial_rho_gas = initial_rho_gas
        self._rho_0 = rho_0
        self._rho_wall = rho_wall

    @staticmethod
    def _build_wall_mask(nx, ny, nz, pipe_radius):
        """Boolean mask: True at solid wall cells."""
        y = jnp.arange(ny, dtype=jnp.float32)
        z = jnp.arange(nz, dtype=jnp.float32)
        yy, zz = jnp.meshgrid(y, z, indexing="ij")
        cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
        r = jnp.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
        max_r = pipe_radius * min(ny, nz) / 2.0
        cross_section_wall = r >= max_r  # (ny, nz)
        # Broadcast to 3D: (nx, ny, nz)
        return jnp.broadcast_to(cross_section_wall[None, :, :], (nx, ny, nz))

    @staticmethod
    def _build_propeller_mask(nx, ny, nz, prop_x, prop_radius, pipe_radius):
        """Boolean mask: True at propeller disc cells."""
        y = jnp.arange(ny, dtype=jnp.float32)
        z = jnp.arange(nz, dtype=jnp.float32)
        yy, zz = jnp.meshgrid(y, z, indexing="ij")
        cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
        r = jnp.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
        max_r = prop_radius * pipe_radius * min(ny, nz) / 2.0
        prop_cross = r < max_r  # (ny, nz)
        # Single axial slice
        mask = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
        mask = mask.at[prop_x, :, :].set(prop_cross)
        return mask

    @property
    def static_data(self) -> dict:
        """The two geometry masks :meth:`update` reads, built in ``__init__``.

        Published so that :meth:`static_data_deps` can name the parameters
        they are built from.  Nothing rebuilds them when those parameters
        are written later.
        """
        from maddening.core.static_data import StaticArray
        return {
            "wall_mask": StaticArray(value=self._wall_mask, replication="replicate"),
            "propeller_mask": StaticArray(value=self._propeller_mask,
                                          replication="replicate"),
        }

    def static_data_deps(self) -> dict[str, tuple[str, ...]]:
        """The geometry each mask is built from, when the node is constructed.

        Declared because :meth:`initial_state` reads ``pipe_radius`` again
        (the fill mask, when ``fill_fraction < 1``), and a parameter the
        running node reads anywhere used to pass ``PUT /graph/params`` as
        "takes effect at the next reset": the write answered 200 and the
        reset rebuilt the fill for the new radius while the step kept the
        wall mask of the old one, and ``to_dict()`` saved a pipe the server
        was not running (MADD-ANO-024).  A declared dependency is refused
        outright on every write surface.  All three are
        ``ParamSpec(trainable=False)`` or structural, so ``compile()`` has
        nothing to object to.
        """
        return {
            "wall_mask": ("pipe_radius",),
            "propeller_mask": ("pipe_radius", "propeller_x", "propeller_radius"),
        }

    def _compute_fill_mask(self):
        """Compute a boolean mask for the initially-filled liquid region.

        Uses the z-coordinate percentile of interior cells so that
        approximately ``fill_fraction`` of the pipe interior is liquid.
        Cells near the wall at the fill level are pulled inward to
        prevent the isosurface from clipping through the pipe wall.

        Returns
        -------
        liquid_mask : (nx, ny, nz) bool — True where liquid should be.
        """
        ny, nz = self._ny, self._nz
        pipe_radius = self.params["pipe_radius"]
        cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
        max_r = pipe_radius * min(ny, nz) / 2.0

        y = np.arange(ny, dtype=np.float32)
        z = np.arange(nz, dtype=np.float32)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        interior = ~np.asarray(self._wall_mask[0, :, :])
        z_interior = zz[interior]
        z_cutoff = float(np.percentile(z_interior, self._fill_fraction * 100))

        # Distance from pipe axis
        r = np.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
        wall_dist = max_r - r  # positive inside pipe

        # Pull the liquid surface inward near the pipe wall to prevent
        # the isosurface from clipping through the wall geometry.
        # Cells within `margin` of the wall AND within `band` of the
        # fill level are excluded.
        margin = 1.5  # lattice units from wall
        band = 2.0    # lattice units from fill level
        z_from_surface = np.abs(zz - z_cutoff)
        near_wall_and_surface = (wall_dist < margin) & (z_from_surface < band)

        liquid_cross = (zz <= z_cutoff) & interior & ~near_wall_and_surface
        return jnp.broadcast_to(
            jnp.array(liquid_cross)[None, :, :],
            (self._nx, ny, nz),
        )

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units."""
        return (self.params["tau"] - 0.5) / 3.0

    def halo_width(self) -> dict[int, int]:
        """One ghost cell per side on each of the three spatial axes.

        D3Q19 streaming and the D3Q7 tracer both have unit-step neighbour
        reads; Shan-Chen multiphase forces also stay within one cell.
        """
        return {0: 1, 1: 1, 2: 1}

    def initial_state(self) -> dict:
        nx = self.params["nx"]
        ny = self.params["ny"]
        nz = self.params["nz"]
        u0 = self.params["initial_velocity"]
        fluid_mask = ~self._wall_mask

        # Initial velocity: uniform x-flow inside the pipe, zero in walls
        velocity = jnp.zeros((nx, ny, nz, 3), dtype=jnp.float32)
        velocity = velocity.at[:, :, :, 0].set(
            jnp.where(fluid_mask, u0, 0.0)
        )

        if self._G != 0.0:
            # ── Multiphase: density-based initialization ──
            # Use a smooth tanh profile at the liquid-gas interface
            # to avoid enormous Shan-Chen forces from a step function.
            ny_val, nz_val = self._ny, self._nz
            cy, cz = (ny_val - 1) / 2.0, (nz_val - 1) / 2.0
            pipe_r = self.params["pipe_radius"] * min(ny_val, nz_val) / 2.0

            # Compute z-cutoff for fill fraction
            z = np.arange(nz_val, dtype=np.float32)
            y = np.arange(ny_val, dtype=np.float32)
            _, zz_np = np.meshgrid(y, z, indexing="ij")
            interior = ~np.asarray(self._wall_mask[0, :, :])
            z_interior = zz_np[interior]
            if self._fill_fraction >= 1.0:
                z_cutoff = float(nz_val)  # above all cells
            else:
                z_cutoff = float(np.percentile(
                    z_interior, self._fill_fraction * 100,
                ))

            # Build smooth density field with tanh interface
            # interface_width controls the transition width in cells
            interface_width = 5.0
            zz = jnp.arange(nz_val, dtype=jnp.float32)
            # Distance from fill level: positive = below surface (liquid)
            z_dist = z_cutoff - zz  # (nz,)
            # tanh profile: 1 deep in liquid, 0 deep in gas
            phase = 0.5 * (1.0 + jnp.tanh(z_dist / interface_width))
            # Broadcast to (nx, ny, nz)
            phase_3d = jnp.broadcast_to(
                phase[None, None, :], (nx, ny_val, nz_val),
            )
            rho_l0, rho_g0 = self._initial_rho_liquid, self._initial_rho_gas
            density = rho_g0 + (rho_l0 - rho_g0) * phase_3d
            # Wall cells: use the gas density for well-formed bounce-back
            density = jnp.where(self._wall_mask, rho_g0, density)

            f = _equilibrium(density, velocity)

            # Tracer: the initial phase fraction, from the same recipe (so
            # the initial state reads no trainable leaf).
            tracer = jnp.clip(
                (density - rho_g0) / (rho_l0 - rho_g0),
                0.0, 1.0,
            )
            tracer = jnp.where(self._wall_mask, 0.0, tracer)
            # tracer_f is a placeholder (not evolved in multiphase mode)
            tracer_f = _tracer_equilibrium(tracer, velocity)
        else:
            # ── Single-phase: original behaviour ──
            density = jnp.ones((nx, ny, nz), dtype=jnp.float32)

            # Initialize distributions to equilibrium (including wall cells —
            # bounce-back will manage walls during update; zeroing them would
            # inject mass loss at boundary fluid cells on the first step).
            f = _equilibrium(density, velocity)

            if self._fill_fraction >= 1.0:
                tracer = jnp.where(fluid_mask, 1.0, 0.0)
            else:
                tracer = jnp.where(self._compute_fill_mask(), 1.0, 0.0)
            tracer_f = _tracer_equilibrium(tracer, velocity)

        return {
            "f": f,
            "density": density,
            "velocity": velocity,
            "tracer": tracer,
            "tracer_f": tracer_f,
        }

    def param_specs(self) -> dict[str, ParamSpec]:
        multiphase = self._G != 0.0
        return {
            **super().param_specs(),
            # BGK stability needs tau > 0.5 strictly.
            "tau": ParamSpec(bounds=(0.5, None), transform="log"),
            "tau_tracer": ParamSpec(bounds=(0.5, None), transform="log",
                                    trainable=not multiphase),
            "propeller_strength": ParamSpec(units="lattice force"),
            "gravity": ParamSpec(units="lattice acceleration"),
            # Shan-Chen constants: read only on the multiphase branch,
            # which ``G != 0`` at construction selects (structural).
            "G": ParamSpec(trainable=multiphase),
            "rho_0": ParamSpec(bounds=(0.0, None), transform="log", trainable=multiphase),
            "rho_wall": ParamSpec(bounds=(0.0, None), transform="log", trainable=multiphase),
            "rho_liquid": ParamSpec(bounds=(0.0, None), transform="log", trainable=multiphase),
            "rho_gas": ParamSpec(bounds=(0.0, None), transform="log", trainable=multiphase),
            # Geometry (masks built at construction) and initial fill.
            "pipe_radius": ParamSpec(trainable=False, description="geometry"),
            "propeller_radius": ParamSpec(trainable=False, description="geometry"),
            "fill_fraction": ParamSpec(trainable=False, description="initial fill"),
        }

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """One collide-stream step (D3Q19 + D3Q7 tracer or Shan-Chen).

        Relaxation times, forcing and the Shan-Chen constants come from
        the injected ``params`` when the graph supplies them; the pipe /
        propeller masks and the single- vs multiphase branch are
        structural and fixed at construction.
        """
        f = state["f"]
        p = self.params if params is None else {**self.params, **params}
        tau = p["tau"]
        prop_strength = boundary_inputs.get(
            "propeller_force", p["propeller_strength"],
        )
        fluid_mask = ~self._wall_mask

        # 1. Compute macroscopic from current f
        density, velocity_raw = _compute_macroscopic(f)

        # 2. Compute ALL body forces before collision
        #    (needed for velocity correction in Shan-Chen mode)
        force = jnp.zeros_like(velocity_raw)
        # Propeller
        force = force.at[:, :, :, 0].set(
            jnp.where(self._propeller_mask, prop_strength, 0.0)
        )
        # Gravity
        gravity = p.get("gravity", 0.0)
        force = force.at[:, :, :, 2].set(
            force[:, :, :, 2] + jnp.where(fluid_mask, gravity, 0.0)
        )

        if self._G != 0.0:
            # ── Multiphase: Exact Difference Method (EDM) ──
            # EDM is more stable than Guo forcing for Shan-Chen because
            # it exactly captures the force effect on equilibrium without
            # a Taylor expansion that breaks down at large interface forces.
            sc_force = _shan_chen_force(
                density, p["G"], p["rho_0"],
                self._wall_mask, p["rho_wall"],
            )
            total_force = force + sc_force

            # Density floor prevents division by near-zero gas density
            density_safe = jnp.maximum(density, 0.01)

            # EDM: Δf_i = f_eq_i(ρ, u+F/ρ) - f_eq_i(ρ, u)
            f_eq_bare = _equilibrium(density, velocity_raw)
            u_shifted = velocity_raw + total_force / density_safe[..., None]
            # Clamp shifted velocity to prevent f_eq breakdown
            # (equilibrium becomes non-physical when |u| approaches c_s).
            # Floor inside the sqrt: d/du sqrt(sum u^2) is NaN at u == 0
            # (a uniform lattice), and the forward is unchanged wherever
            # |u| > 1e-10, which the maximum() below already assumes.
            u_mag = jnp.sqrt(jnp.maximum(
                jnp.sum(u_shifted ** 2, axis=-1, keepdims=True), 1e-20,
            ))
            scale = jnp.minimum(
                1.0, 0.25 / jnp.maximum(u_mag, 1e-10),
            )
            u_shifted = u_shifted * scale
            f_eq_shifted = _equilibrium(density, u_shifted)

            # BGK collision + EDM forcing
            f_post = f - (f - f_eq_bare) / tau + (f_eq_shifted - f_eq_bare)

            # Macroscopic velocity: half-force correction (time-centering)
            velocity_eq = velocity_raw + total_force / (
                2.0 * density_safe[..., None]
            )
        else:
            # ── Single-phase: Guo forcing scheme (unchanged) ──
            velocity_eq = velocity_raw
            f_eq = _equilibrium(density, velocity_eq)
            f_post = f - (f - f_eq) / tau
            f_post = f_post + _guo_forcing(density, velocity_eq, force, tau)

        # 5. Streaming
        f_streamed = _stream(f_post)

        # 6. Bounce-back at walls
        f_bounced = f_streamed[..., _OPP]
        wall_3d = self._wall_mask[..., None]
        f_bc = jnp.where(wall_3d, f_bounced, f_streamed)

        # 7. Extract macroscopic for output
        density_new, velocity_new = _compute_macroscopic(f_bc)

        # Density floor in multiphase: prevent gas cells from reaching
        # zero density (which causes NaN on the next step).  Scale the
        # distributions so they sum to at least rho_floor.
        if self._G != 0.0:
            rho_floor = 0.01
            rho_ratio = jnp.maximum(density_new, rho_floor) / jnp.maximum(
                density_new, 1e-20,
            )
            needs_fix = density_new < rho_floor
            f_bc = jnp.where(
                needs_fix[..., None], f_bc * rho_ratio[..., None], f_bc,
            )
            density_new = jnp.where(needs_fix, rho_floor, density_new)

        # Zero velocity in walls (cosmetic, for cleaner rendering)
        velocity_new = jnp.where(self._wall_mask[..., None], 0.0, velocity_new)

        # 8. Phase indicator / tracer
        if self._G != 0.0:
            # Multiphase: derive tracer from density
            tracer_new = jnp.clip(
                (density_new - p["rho_gas"])
                / (p["rho_liquid"] - p["rho_gas"]),
                0.0, 1.0,
            )
            tracer_new = jnp.where(self._wall_mask, 0.0, tracer_new)
            # tracer_f frozen — just pass through
            tracer_f_out = state["tracer_f"]
        else:
            # Single-phase: D3Q7 passive scalar transport
            tracer_f = state["tracer_f"]
            tracer = jnp.sum(tracer_f, axis=-1)
            g_eq = _tracer_equilibrium(tracer, velocity_new)
            g_post = jnp.where(
                self._wall_mask[..., None],
                tracer_f,  # no collision at walls
                tracer_f - (tracer_f - g_eq) / p["tau_tracer"],
            )
            g_streamed = _stream7(g_post)
            g_bounced = g_streamed[..., _OPP7]
            g_bc = jnp.where(
                self._wall_mask[..., None], g_bounced, g_streamed,
            )
            tracer_new = jnp.sum(g_bc, axis=-1)
            tracer_new = jnp.where(self._wall_mask, 0.0, tracer_new)
            tracer_f_out = g_bc

        return {
            "f": f_bc,
            "density": density_new,
            "velocity": velocity_new,
            "tracer": tracer_new,
            "tracer_f": tracer_f_out,
        }
