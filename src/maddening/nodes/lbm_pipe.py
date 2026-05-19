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

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability

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
    psi_wall = rho_0 * (1.0 - np.exp(-rho_wall / rho_0))
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
        BGK relaxation time.  Must be > 0.5 for stability.
        Kinematic viscosity: ``nu = (tau - 0.5) / 3``.
    pipe_radius : float
        Pipe radius as fraction of ``min(ny, nz) / 2`` (0-1).
    propeller_x : int
        Axial position of the propeller disc (grid index).
    propeller_radius : float
        Propeller disc radius as fraction of pipe radius (0-1).
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
        Initial liquid-phase density (multiphase mode).  Default 1.0.
        Should be close to the EOS coexistence density for the chosen G
        (roughly 0.8-1.5 for G in [-4.2, -5.0] with rho_0=1).
    rho_gas : float
        Initial gas-phase density (multiphase mode).  Default 0.25.
        Should be close to the EOS coexistence density for the chosen G
        (roughly 0.15-0.45 for G in [-4.2, -5.0] with rho_0=1).
    rho_0 : float
        Pseudopotential reference density.  Default 1.0.
    rho_wall : float or None
        Wall pseudopotential density for wetting control.  ``None``
        defaults to ``rho_liquid`` (fully wetted / hydrophilic wall).
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-006",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="D3Q19 Lattice Boltzmann in cylindrical pipe with propeller actuator disc",
        governing_equations="BGK collision: f_i = f_i - (f_i - f_eq_i)/τ; streaming: f_i(x+e_i, t+1) = f_i(x, t)",
        discretization="Lattice Boltzmann (D3Q19, explicit, 2nd-order in space and time for low Ma)",
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
            "CUDA 12.2 + jaxlib 0.5.1 causes segfault on GPU (MADD-ANO-001)",
            "Wall bounce-back is 1st-order at curved boundaries",
        ),
        validated_regimes=(
            ValidatedRegime("tau", 0.501, 2.0, notes="tau > 0.5 required for stability; tau >> 1 causes numerical diffusion"),
            ValidatedRegime("Reynolds number", 0, 100, notes="Validated against Poiseuille analytical solution"),
            ValidatedRegime("grid", 8, 128, notes="Per-dimension; convergence verified up to 128³"),
        ),
        hazard_hints=(
            "Behaviour uncharacterised at Re > 100; validated only for laminar flow",
            "No turbulence model — do not use above laminar-turbulent transition (Re ~2000)",
            "Wall bounce-back assumes rigid, impermeable walls; deformable or porous walls not modelled",
            "Gravity applied uniformly — no spatially varying body forces",
            "CUDA 12.2 + jaxlib 0.5.1 causes segfault on GPU (MADD-ANO-001); CPU unaffected",
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
    ):
        if tau <= 0.5:
            raise ValueError(
                f"tau must be > 0.5 for stability (got {tau}). "
                f"nu = (tau - 0.5) / 3 = {(tau - 0.5) / 3:.4f}"
            )
        if tau_tracer <= 0.5:
            raise ValueError(
                f"tau_tracer must be > 0.5 for stability (got {tau_tracer})."
            )
        if not 0.0 < fill_fraction <= 1.0:
            raise ValueError(
                f"fill_fraction must be in (0, 1] (got {fill_fraction})."
            )
        if G != 0.0:
            if rho_liquid <= rho_gas:
                raise ValueError(
                    f"rho_liquid must be > rho_gas "
                    f"(got {rho_liquid} <= {rho_gas})."
                )
            if rho_gas <= 0.0:
                raise ValueError(f"rho_gas must be > 0 (got {rho_gas}).")
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
        )

        # Pre-compute masks and store as JAX arrays
        self._wall_mask = self._build_wall_mask(nx, ny, nz, pipe_radius)
        self._propeller_mask = self._build_propeller_mask(
            nx, ny, nz, propeller_x, propeller_radius, pipe_radius,
        )
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._tau = tau
        self._tau_tracer = tau_tracer
        self._gravity = gravity
        self._fill_fraction = fill_fraction
        self._G = G
        self._rho_liquid = rho_liquid
        self._rho_gas = rho_gas
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
            density = (
                self._rho_gas
                + (self._rho_liquid - self._rho_gas) * phase_3d
            )
            # Wall cells: use rho_gas for well-formed bounce-back
            density = jnp.where(self._wall_mask, self._rho_gas, density)

            f = _equilibrium(density, velocity)

            # Tracer derived from density
            tracer = jnp.clip(
                (density - self._rho_gas)
                / (self._rho_liquid - self._rho_gas),
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

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        f = state["f"]
        tau = self.params["tau"]
        prop_strength = boundary_inputs.get(
            "propeller_force", self.params["propeller_strength"],
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
        gravity = self.params.get("gravity", 0.0)
        force = force.at[:, :, :, 2].set(
            force[:, :, :, 2] + jnp.where(fluid_mask, gravity, 0.0)
        )

        if self._G != 0.0:
            # ── Multiphase: Exact Difference Method (EDM) ──
            # EDM is more stable than Guo forcing for Shan-Chen because
            # it exactly captures the force effect on equilibrium without
            # a Taylor expansion that breaks down at large interface forces.
            sc_force = _shan_chen_force(
                density, self._G, self._rho_0,
                self._wall_mask, self._rho_wall,
            )
            total_force = force + sc_force

            # Density floor prevents division by near-zero gas density
            density_safe = jnp.maximum(density, 0.01)

            # EDM: Δf_i = f_eq_i(ρ, u+F/ρ) - f_eq_i(ρ, u)
            f_eq_bare = _equilibrium(density, velocity_raw)
            u_shifted = velocity_raw + total_force / density_safe[..., None]
            # Clamp shifted velocity to prevent f_eq breakdown
            # (equilibrium becomes non-physical when |u| approaches c_s)
            u_mag = jnp.sqrt(jnp.sum(u_shifted ** 2, axis=-1, keepdims=True))
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
                (density_new - self._rho_gas)
                / (self._rho_liquid - self._rho_gas),
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
                tracer_f - (tracer_f - g_eq) / self._tau_tracer,
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
