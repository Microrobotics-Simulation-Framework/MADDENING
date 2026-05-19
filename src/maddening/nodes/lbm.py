"""
LBMNode -- general 3D Lattice Boltzmann node on arbitrary wall-mask domains.

A reusable LBM node that operates on any domain defined by a boolean wall
mask, with Zou-He pressure boundary conditions at configurable inlet/outlet
faces.  Supports D3Q19 (3D) and D2Q9 (2D) lattices.

The entire ``update`` uses ``jnp`` operations (no Python control flow that
depends on array values), so it is fully JAX-traceable and JIT-compilable.

Lattice units are used internally: dx = dt_lbm = 1.  The kinematic viscosity
is controlled by the BGK relaxation time tau via::

    nu = cs2 * (tau - 0.5)       [= (tau - 0.5) / 3 for standard lattices]

State fields
------------
f : (*grid_shape, Q) float32
    Lattice distribution functions.
density : grid_shape float32
    Macroscopic density (sum of f over Q directions).
velocity : (*grid_shape, D) float32
    Macroscopic velocity.
pressure : grid_shape float32
    Macroscopic pressure (= density * cs2).

Boundary inputs
---------------
inlet_pressure : scalar
    Zou-He pressure BC at the inlet face.
outlet_pressure : scalar
    Zou-He pressure BC at the outlet face.
body_force : (*grid_shape, D)
    External body force field (Guo forcing).
wall_mask_update : grid_shape bool
    Runtime wall mask override (e.g., for clot injection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


# ═══════════════════════════════════════════════════════════════════════
# Lattice descriptors
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LatticeDescriptor:
    """Immutable description of a lattice Boltzmann velocity set.

    All array fields use **numpy** (not jnp) so they remain concrete
    values inside JIT/scan -- a lesson learned from LBMPipeNode.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``"D3Q19"``.
    e : numpy.ndarray, shape (Q, D)
        Discrete velocity vectors.
    w : numpy.ndarray, shape (Q,)
        Lattice weights (must sum to 1).
    opp : numpy.ndarray, shape (Q,)
        Index of the opposite direction for each velocity.
    cs2 : float
        Speed of sound squared (1/3 for standard lattices).
    D : int
        Spatial dimensions.
    Q : int
        Number of discrete velocities.
    """
    name: str
    e: np.ndarray
    w: np.ndarray
    opp: np.ndarray
    cs2: float
    D: int
    Q: int


def d3q19() -> LatticeDescriptor:
    """Factory for the D3Q19 lattice (3D, 19 velocities)."""
    e = np.array([
        [0, 0, 0],     # 0  rest
        [1, 0, 0],     # 1  +x
        [-1, 0, 0],    # 2  -x
        [0, 1, 0],     # 3  +y
        [0, -1, 0],    # 4  -y
        [0, 0, 1],     # 5  +z
        [0, 0, -1],    # 6  -z
        [1, 1, 0],     # 7
        [-1, 1, 0],    # 8
        [1, -1, 0],    # 9
        [-1, -1, 0],   # 10
        [1, 0, 1],     # 11
        [-1, 0, 1],    # 12
        [1, 0, -1],    # 13
        [-1, 0, -1],   # 14
        [0, 1, 1],     # 15
        [0, -1, 1],    # 16
        [0, 1, -1],    # 17
        [0, -1, -1],   # 18
    ], dtype=np.int32)

    w = np.array([
        1.0 / 3.0,                           # rest
        1.0 / 18.0, 1.0 / 18.0,              # +/- x
        1.0 / 18.0, 1.0 / 18.0,              # +/- y
        1.0 / 18.0, 1.0 / 18.0,              # +/- z
        1.0 / 36.0, 1.0 / 36.0,              # edge xy
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,              # edge xz
        1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0,              # edge yz
        1.0 / 36.0, 1.0 / 36.0,
    ], dtype=np.float64)

    opp = np.array([
        0,           # rest -> rest
        2, 1,        # +x <-> -x
        4, 3,        # +y <-> -y
        6, 5,        # +z <-> -z
        10, 9, 8, 7,         # edge xy
        14, 13, 12, 11,      # edge xz
        18, 17, 16, 15,      # edge yz
    ], dtype=np.int32)

    return LatticeDescriptor(
        name="D3Q19", e=e, w=w, opp=opp, cs2=1.0 / 3.0, D=3, Q=19,
    )


def d2q9() -> LatticeDescriptor:
    """Factory for the D2Q9 lattice (2D, 9 velocities)."""
    e = np.array([
        [0, 0],      # 0  rest
        [1, 0],      # 1  +x
        [-1, 0],     # 2  -x
        [0, 1],      # 3  +y
        [0, -1],     # 4  -y
        [1, 1],      # 5  +x+y
        [-1, 1],     # 6  -x+y
        [1, -1],     # 7  +x-y
        [-1, -1],    # 8  -x-y
    ], dtype=np.int32)

    w = np.array([
        4.0 / 9.0,                          # rest
        1.0 / 9.0, 1.0 / 9.0,               # +/- x
        1.0 / 9.0, 1.0 / 9.0,               # +/- y
        1.0 / 36.0, 1.0 / 36.0,             # diagonals
        1.0 / 36.0, 1.0 / 36.0,
    ], dtype=np.float64)

    opp = np.array([
        0,        # rest -> rest
        2, 1,     # +x <-> -x
        4, 3,     # +y <-> -y
        8, 7, 6, 5,  # diagonals
    ], dtype=np.int32)

    return LatticeDescriptor(
        name="D2Q9", e=e, w=w, opp=opp, cs2=1.0 / 3.0, D=2, Q=9,
    )


# ═══════════════════════════════════════════════════════════════════════
# Pure-function LBM kernels
# ═══════════════════════════════════════════════════════════════════════

def _equilibrium(density, velocity, e, w, cs2):
    """Compute equilibrium distribution f_eq.

    Parameters
    ----------
    density : (*grid_shape,) float32
    velocity : (*grid_shape, D) float32
    e : (Q, D) numpy int array -- lattice velocities
    w : (Q,) numpy float array -- lattice weights
    cs2 : float -- speed of sound squared

    Returns
    -------
    f_eq : (*grid_shape, Q) float32
    """
    e_f = e.astype(np.float64)
    # e_dot_u: (*grid_shape, Q)
    e_dot_u = velocity @ e_f.T
    u_sq = jnp.sum(velocity ** 2, axis=-1, keepdims=True)  # (..., 1)

    f_eq = w * density[..., None] * (
        1.0
        + e_dot_u / cs2
        + e_dot_u ** 2 / (2.0 * cs2 ** 2)
        - u_sq / (2.0 * cs2)
    )
    return f_eq


def _compute_macroscopic(f, e):
    """Extract density and velocity from distribution functions.

    Parameters
    ----------
    f : (*grid_shape, Q) float32
    e : (Q, D) numpy int array

    Returns
    -------
    density : (*grid_shape,) float32
    velocity : (*grid_shape, D) float32
    """
    density = jnp.sum(f, axis=-1)
    e_f = e.astype(np.float64)
    momentum = f @ e_f  # (*grid_shape, D)
    velocity = momentum / jnp.maximum(density[..., None], 1e-10)
    return density, velocity


def _stream(f, e, ndim):
    """Streaming step: shift each f_q by its lattice velocity via jnp.roll.

    Parameters
    ----------
    f : (*grid_shape, Q) float32
    e : (Q, D) numpy int array
    ndim : int -- number of spatial dimensions (2 or 3)

    Returns
    -------
    f_streamed : (*grid_shape, Q) float32
    """
    Q = e.shape[0]
    slices = []
    for q in range(Q):
        fq = f[..., q]
        for d in range(ndim):
            shift = int(e[q, d])
            if shift != 0:
                fq = jnp.roll(fq, shift, axis=d)
        slices.append(fq)
    return jnp.stack(slices, axis=-1)


def _guo_forcing(density, velocity, force, tau, e, w, cs2):
    """Guo forcing term for the BGK collision operator.

    Implements Guo et al. (2002)::

        S_q = (1 - 1/(2*tau)) * w_q *
              [(e_q - u)/cs2 + (e_q . u)/(cs4) * e_q] . F

    Parameters
    ----------
    density : (*grid_shape,) float32
    velocity : (*grid_shape, D) float32
    force : (*grid_shape, D) float32
    tau : float
    e : (Q, D) numpy int array
    w : (Q,) numpy float array
    cs2 : float

    Returns
    -------
    S : (*grid_shape, Q) float32
    """
    e_f = e.astype(np.float64)
    ndim = e.shape[1]
    Q = e.shape[0]

    # Build extra dimensions for broadcasting: prepend len(grid_shape) Nones
    # We use explicit expansion: e_f -> (1,...,1, Q, D), etc.
    # Simpler approach: compute per-q and stack.
    e_dot_u = velocity @ e_f.T  # (*grid_shape, Q)

    # (e_q - u): (*grid_shape, Q, D)
    # Expand e_f: (Q, D) -> broadcast with velocity (..., 1, D)
    # velocity: (*grid_shape, D) -> (*grid_shape, 1, D)
    n_spatial = len(velocity.shape) - 1  # number of spatial dims in shape
    expand = (None,) * n_spatial + (slice(None), slice(None))
    e_expanded = e_f[expand]  # (1,...,1, Q, D)
    vel_expanded = velocity[..., None, :]  # (*grid_shape, 1, D)

    e_minus_u = e_expanded - vel_expanded  # (*grid_shape, Q, D)
    e_scaled = e_expanded * (e_dot_u / (cs2 ** 2))[..., None]  # (*grid_shape, Q, D)

    bracket = (e_minus_u / cs2 + e_scaled)  # (*grid_shape, Q, D)
    # Dot with force: (*grid_shape, Q)
    force_expanded = force[..., None, :]  # (*grid_shape, 1, D)
    S = (1.0 - 0.5 / tau) * w * jnp.sum(bracket * force_expanded, axis=-1)
    return S


# ═══════════════════════════════════════════════════════════════════════
# Zou-He pressure boundary conditions
# ═══════════════════════════════════════════════════════════════════════

def _get_opp_map(e):
    """Build the opposite-direction map from velocity vectors.

    For each direction q, find opp_q such that e[opp_q] == -e[q].

    Parameters
    ----------
    e : (Q, D) numpy int array

    Returns
    -------
    opp : (Q,) numpy int array
    """
    Q = e.shape[0]
    opp = np.zeros(Q, dtype=np.int32)
    for q in range(Q):
        for p in range(Q):
            if np.all(e[p] == -e[q]):
                opp[q] = p
                break
    return opp


def _classify_directions(e, face_axis, face_side):
    """Classify lattice directions as known, unknown, or tangential for a face.

    Parameters
    ----------
    e : (Q, D) numpy int array
    face_axis : int -- 0 for x, 1 for y, 2 for z
    face_side : str -- "min" or "max"

    Returns
    -------
    known : list of int -- indices pointing INTO the domain (known after streaming)
    unknown : list of int -- indices pointing OUT of the domain (need reconstruction)
    tangential : list of int -- indices with zero component on face_axis
    """
    Q = e.shape[0]
    # At a "min" face (e.g. x=0), distributions pointing in +axis are unknown
    # (they would come from outside the domain). Distributions pointing in -axis
    # are known (they were streamed from the interior).
    # At a "max" face (e.g. x=nx-1), distributions pointing in -axis are unknown.
    known = []
    unknown = []
    tangential = []
    for q in range(Q):
        comp = int(e[q, face_axis])
        if face_side == "min":
            if comp > 0:
                unknown.append(q)
            elif comp < 0:
                known.append(q)
            else:
                tangential.append(q)
        else:  # "max"
            if comp < 0:
                unknown.append(q)
            elif comp > 0:
                known.append(q)
            else:
                tangential.append(q)
    return known, unknown, tangential


def _zou_he_pressure_face(f, prescribed_density, e, w, cs2,
                          face_axis, face_side, wall_mask):
    """Apply Zou-He pressure BC on a flat face.

    Non-equilibrium bounce-back method: set unknown distributions so that
    the prescribed density (pressure / cs2) is satisfied.

    Parameters
    ----------
    f : (*grid_shape, Q) float32
    prescribed_density : scalar float -- rho = P / cs2
    e : (Q, D) numpy int array
    w : (Q,) numpy float array
    cs2 : float
    face_axis : int -- 0, 1, or 2
    face_side : str -- "min" or "max"
    wall_mask : (*grid_shape,) bool -- True at wall cells

    Returns
    -------
    f_updated : (*grid_shape, Q) float32
    """
    ndim = e.shape[1]
    known, unknown, tangential = _classify_directions(e, face_axis, face_side)

    # Build slice for the face
    face_slices = [slice(None)] * ndim
    if face_side == "min":
        face_slices[face_axis] = 0
    else:
        face_slices[face_axis] = -1
    face_sl = tuple(face_slices)

    # Extract face distributions: shape (*face_shape, Q)
    f_face = f[face_sl]
    wall_face = wall_mask[face_sl]

    # Sum of known and tangential distributions at the face
    sum_known = jnp.zeros_like(f_face[..., 0])
    for q in known:
        sum_known = sum_known + f_face[..., q]
    sum_tang = jnp.zeros_like(f_face[..., 0])
    for q in tangential:
        sum_tang = sum_tang + f_face[..., q]

    # Normal velocity from known distributions
    # For min face: u_n = 1 - (sum_known + sum_tangential) / rho_prescribed
    # For max face: u_n = -1 + (sum_known + sum_tangential) / rho_prescribed
    # (sign convention: velocity pointing into the domain is positive for min,
    #  negative for max)
    rho_p = prescribed_density
    if face_side == "min":
        u_normal = 1.0 - (sum_known + sum_tang) / jnp.maximum(rho_p, 1e-10)
    else:
        u_normal = -1.0 + (sum_known + sum_tang) / jnp.maximum(rho_p, 1e-10)

    # Non-equilibrium bounce-back for each unknown direction
    # f_q = f_{opp(q)} + (f_eq_q - f_eq_{opp(q)}) at the prescribed state
    # Simplified: for each unknown q, its opposite opp_q is known.
    # The non-equilibrium part: f_q = f_{opp(q)} + rho_p * w_q * (e_q . u) / cs2 * 2
    # This is the standard Zou-He non-equilibrium bounce-back.

    # Build the velocity vector on the face (only normal component from Zou-He)
    vel_face = jnp.zeros(f_face.shape[:-1] + (ndim,), dtype=f_face.dtype)
    vel_face = vel_face.at[..., face_axis].set(u_normal)

    # Compute equilibrium at the face for known velocity/density
    f_eq_face = _equilibrium(
        jnp.full_like(f_face[..., 0], rho_p), vel_face, e, w, cs2,
    )

    # For unknown directions, use non-equilibrium bounce-back:
    # f_q = f_opp(q) + f_eq(q) - f_eq(opp(q))
    opp_map = _get_opp_map(e)

    f_face_new = f_face
    for q in unknown:
        opp_q = opp_map[q]
        f_q_new = f_face[..., opp_q] + f_eq_face[..., q] - f_eq_face[..., opp_q]
        # Only apply to fluid cells, not wall cells
        f_q_corrected = jnp.where(wall_face, f_face[..., q], f_q_new)
        f_face_new = f_face_new.at[..., q].set(f_q_corrected)

    # Write back to the full array
    f = f.at[face_sl].set(f_face_new)
    return f


# ═══════════════════════════════════════════════════════════════════════
# Face-axis mapping
# ═══════════════════════════════════════════════════════════════════════

_FACE_MAP = {
    "x_min": (0, "min"),
    "x_max": (0, "max"),
    "y_min": (1, "min"),
    "y_max": (1, "max"),
    "z_min": (2, "min"),
    "z_max": (2, "max"),
}


# ═══════════════════════════════════════════════════════════════════════
# LBMNode
# ═══════════════════════════════════════════════════════════════════════

@stability(StabilityLevel.EXPERIMENTAL)
class LBMNode(SimulationNode):
    """General 3D/2D Lattice Boltzmann node on an arbitrary wall-mask domain.

    Uses the BGK single-relaxation-time collision operator with bounce-back
    wall boundaries and Zou-He pressure boundary conditions at configurable
    inlet/outlet faces.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep (seconds).
    grid_shape : tuple of int
        Grid dimensions, e.g. ``(64, 32, 32)`` for 3D or ``(64, 32)`` for 2D.
    viscosity : float
        Kinematic viscosity in lattice units.  Determines tau via
        ``tau = 0.5 + viscosity / cs2``.
    lattice : str
        Lattice type: ``"D3Q19"`` (default) or ``"D2Q9"``.
    wall_mask : numpy.ndarray or None
        Boolean array of shape ``grid_shape``, True = wall.  If None, no walls.
    inlet_face : str
        Face for pressure inlet BC: ``"x_min"`` (default), ``"x_max"``, etc.
    outlet_face : str
        Face for pressure outlet BC: ``"x_max"`` (default).
    geometry_source : str or None
        USD prim path for geometry sourcing.
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-007",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "General LBM node with BGK collision, Zou-He pressure BCs, "
            "and arbitrary wall-mask domains"
        ),
        governing_equations=(
            "BGK collision: f_i = f_i - (f_i - f_eq_i)/tau + S_guo; "
            "streaming: f_i(x+e_i, t+1) = f_i(x, t)"
        ),
        discretization="Lattice Boltzmann (D3Q19/D2Q9, explicit, 2nd-order)",
        assumptions=(
            "Incompressible flow (Mach number << 1)",
            "BGK single-relaxation-time collision operator",
            "Rigid, impermeable walls (mid-link bounce-back)",
            "Zou-He pressure boundary conditions at inlet/outlet",
        ),
        limitations=(
            "Compressibility errors at high Mach number (Ma > 0.1)",
            "BGK is less stable than MRT for high Reynolds numbers",
            "No turbulence model",
            "Wall bounce-back is 1st-order at curved boundaries",
        ),
        validated_regimes=(
            ValidatedRegime(
                "tau", 0.501, 2.0,
                notes="tau > 0.5 required; tau >> 1 causes numerical diffusion",
            ),
            ValidatedRegime(
                "Reynolds number", 0, 100,
                notes="Validated against Poiseuille analytical solution",
            ),
        ),
        hazard_hints=(
            "Behaviour uncharacterised at Re > 100",
            "No turbulence model -- do not use above Re ~2000",
            "Wall bounce-back assumes rigid, impermeable walls",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        grid_shape: tuple = (64, 32, 32),
        viscosity: float = 0.1,
        lattice: str = "D3Q19",
        wall_mask: Optional[np.ndarray] = None,
        inlet_face: str = "x_min",
        outlet_face: str = "x_max",
        geometry_source: Optional[str] = None,
    ):
        # Validate lattice choice
        if lattice.upper() == "D3Q19":
            lat = d3q19()
        elif lattice.upper() == "D2Q9":
            lat = d2q9()
        else:
            raise ValueError(
                f"Unknown lattice '{lattice}'. Supported: 'D3Q19', 'D2Q9'."
            )

        # Validate grid shape vs lattice dimensionality
        if len(grid_shape) != lat.D:
            raise ValueError(
                f"grid_shape has {len(grid_shape)} dims but lattice {lat.name} "
                f"requires {lat.D} dims."
            )

        # Compute tau from viscosity
        tau = 0.5 + viscosity / lat.cs2
        if tau <= 0.5:
            raise ValueError(
                f"tau must be > 0.5 (got {tau}). Increase viscosity."
            )

        # Validate faces
        for face_name, face_label in [("inlet_face", inlet_face),
                                       ("outlet_face", outlet_face)]:
            if face_label not in _FACE_MAP:
                raise ValueError(
                    f"Unknown {face_name} '{face_label}'. "
                    f"Supported: {list(_FACE_MAP.keys())}"
                )
            face_axis, _ = _FACE_MAP[face_label]
            if face_axis >= lat.D:
                raise ValueError(
                    f"{face_name} '{face_label}' uses axis {face_axis} "
                    f"but lattice {lat.name} only has {lat.D} dimensions."
                )

        super().__init__(
            name,
            timestep,
            grid_shape=grid_shape,
            viscosity=viscosity,
            lattice=lattice,
            inlet_face=inlet_face,
            outlet_face=outlet_face,
            geometry_source=geometry_source,
        )

        self._lat = lat
        self._grid_shape = tuple(grid_shape)
        self._D = lat.D
        self._Q = lat.Q
        self._tau = tau
        self._cs2 = lat.cs2
        self._inlet_face = inlet_face
        self._outlet_face = outlet_face

        # Wall mask: convert to JAX array for JIT compatibility
        if wall_mask is not None:
            if wall_mask.shape != tuple(grid_shape):
                raise ValueError(
                    f"wall_mask shape {wall_mask.shape} != grid_shape "
                    f"{tuple(grid_shape)}"
                )
            self._wall_mask = jnp.asarray(wall_mask, dtype=jnp.bool_)
        else:
            self._wall_mask = jnp.zeros(grid_shape, dtype=jnp.bool_)

    @property
    def tau(self) -> float:
        """BGK relaxation time."""
        return self._tau

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units."""
        return self._cs2 * (self._tau - 0.5)

    @property
    def lattice(self) -> LatticeDescriptor:
        """The lattice descriptor."""
        return self._lat

    # ------------------------------------------------------------------
    # SimulationNode interface
    # ------------------------------------------------------------------

    def halo_width(self) -> dict[int, int]:
        """One ghost cell per side on every spatial axis.

        D3Q19 and D2Q9 both have unit-step neighbour reads (max ``|e_q|=1``
        in every component), so a single ghost cell per side is enough for
        the streaming step.  Axis count tracks the lattice dimension
        (3 for D3Q19, 2 for D2Q9).
        """
        return {axis: 1 for axis in range(self._D)}

    def initial_state(self) -> dict:
        shape = self._grid_shape
        D = self._D
        Q = self._Q

        density = jnp.ones(shape, dtype=jnp.float32)
        velocity = jnp.zeros(shape + (D,), dtype=jnp.float32)

        # Initialize distributions to equilibrium (not zeros!) to prevent
        # mass loss at boundary fluid cells.
        f = _equilibrium(density, velocity, self._lat.e, self._lat.w,
                         self._cs2)

        pressure = density * self._cs2

        return {
            "f": f,
            "density": density,
            "velocity": velocity,
            "pressure": pressure,
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        f = state["f"]
        tau = self._tau
        lat = self._lat
        e = lat.e
        w = lat.w
        cs2 = lat.cs2
        Q = lat.Q
        D = self._D
        ndim = D

        # Runtime wall mask override (e.g. clot injection)
        wall_mask = self._wall_mask
        if "wall_mask_update" in boundary_inputs:
            wall_mask = boundary_inputs["wall_mask_update"].astype(jnp.bool_)

        fluid_mask = ~wall_mask

        # 1. Compute macroscopic from current f
        density, velocity = _compute_macroscopic(f, e)

        # 2. Body force (Guo forcing)
        force = boundary_inputs.get(
            "body_force",
            jnp.zeros(self._grid_shape + (D,), dtype=jnp.float32),
        )

        # 3. BGK collision with Guo forcing
        f_eq = _equilibrium(density, velocity, e, w, cs2)
        f_post = f - (f - f_eq) / tau
        f_post = f_post + _guo_forcing(density, velocity, force, tau, e, w, cs2)

        # 4. Streaming
        f_streamed = _stream(f_post, e, ndim)

        # 5. Bounce-back at wall cells
        opp = lat.opp
        f_bounced = f_streamed[..., opp]
        wall_expanded = wall_mask[..., None]
        f_bc = jnp.where(wall_expanded, f_bounced, f_streamed)

        # 6. Zou-He pressure BCs at inlet and outlet
        inlet_pressure = boundary_inputs.get("inlet_pressure", None)
        outlet_pressure = boundary_inputs.get("outlet_pressure", None)

        if inlet_pressure is not None:
            inlet_axis, inlet_side = _FACE_MAP[self._inlet_face]
            inlet_rho = inlet_pressure / cs2
            f_bc = _zou_he_pressure_face(
                f_bc, inlet_rho, e, w, cs2,
                inlet_axis, inlet_side, wall_mask,
            )

        if outlet_pressure is not None:
            outlet_axis, outlet_side = _FACE_MAP[self._outlet_face]
            outlet_rho = outlet_pressure / cs2
            f_bc = _zou_he_pressure_face(
                f_bc, outlet_rho, e, w, cs2,
                outlet_axis, outlet_side, wall_mask,
            )

        # 7. Compute macroscopic for output
        density_new, velocity_new = _compute_macroscopic(f_bc, e)

        # Zero velocity in wall cells (cosmetic, for cleaner output)
        velocity_new = jnp.where(wall_expanded, 0.0, velocity_new)

        pressure_new = density_new * cs2

        return {
            "f": f_bc,
            "density": density_new,
            "velocity": velocity_new,
            "pressure": pressure_new,
        }

    def derivatives(self, state: dict, boundary_inputs: dict) -> dict:
        """Not applicable for LBM (discrete update, not an ODE)."""
        raise NotImplementedError(
            "LBMNode uses a discrete lattice Boltzmann update, "
            "not a continuous ODE. derivatives() is not applicable."
        )

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        """Expose average pressure at the outlet face for coupling."""
        pressure = state["pressure"]
        outlet_axis, outlet_side = _FACE_MAP[self._outlet_face]

        # Build slice for the outlet face
        face_slices = [slice(None)] * self._D
        if outlet_side == "min":
            face_slices[outlet_axis] = 0
        else:
            face_slices[outlet_axis] = -1
        face_sl = tuple(face_slices)

        p_face = pressure[face_sl]
        # Mask out wall cells at the outlet face
        wall_face = self._wall_mask[face_sl]
        fluid_count = jnp.sum(~wall_face)
        p_sum = jnp.sum(jnp.where(wall_face, 0.0, p_face))
        outlet_pressure_avg = p_sum / jnp.maximum(fluid_count, 1.0)

        return {"outlet_pressure_avg": outlet_pressure_avg}

    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {
            "inlet_pressure": BoundaryInputSpec(
                shape=(), description="Zou-He pressure at inlet face",
            ),
            "outlet_pressure": BoundaryInputSpec(
                shape=(), description="Zou-He pressure at outlet face",
            ),
            "body_force": BoundaryInputSpec(
                shape=(*self._grid_shape, self._D),
                coupling_type="additive",
                description="External body force field",
            ),
            "wall_mask_update": BoundaryInputSpec(
                shape=self._grid_shape,
                description="Runtime wall mask override (True=wall)",
            ),
        }
