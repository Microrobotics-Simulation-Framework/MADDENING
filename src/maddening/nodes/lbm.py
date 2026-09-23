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
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import (
    DiscretizationOrder,
    NodeMeta,
    StabilityLevel,
    ValidatedRegime,
)
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec


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


def _compute_macroscopic(f, e, force=None):
    """Extract density and velocity from distribution functions.

    Implements the Guo et al. (2002) body-force scheme: when a body
    force is present, the recovered velocity carries a ``F/2`` half-step
    correction (``rho*u = Sum_q f_q e_q + F/2``).  Without this
    correction the macroscopic velocity is biased low by ``F/(2*rho)``
    per step under body-force-driven flow, and Hagen-Poiseuille profiles
    end up scaled by ``1 - 1/(2 tau)`` instead of matching the textbook
    formula.

    Parameters
    ----------
    f : (*grid_shape, Q) float32
    e : (Q, D) numpy int array
    force : (*grid_shape, D) float32 or None
        Optional body force per cell (Guo half-correction).  Pass the
        same force used by the source term in the same step.

    Returns
    -------
    density : (*grid_shape,) float32
    velocity : (*grid_shape, D) float32
    """
    density = jnp.sum(f, axis=-1)
    e_f = e.astype(np.float64)
    momentum = f @ e_f  # (*grid_shape, D)
    if force is not None:
        momentum = momentum + 0.5 * force
    velocity = momentum / jnp.maximum(density[..., None], 1e-10)
    return density, velocity


def _stream(f, e, ndim):
    """Streaming step: shift each f_q by its lattice velocity via jnp.roll.

    Periodic wrap on every axis -- used by the unsharded :meth:`LBMNode.update`
    path.  For halo-aware sharded streaming see :func:`_stream_padded`.

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


def _stream_padded(f_pad, e, ndim, halo):
    """Halo-aware streaming step on a padded distribution.

    Replaces the ``jnp.roll`` wrap with explicit indexed slicing.  Each
    direction ``q`` reads from neighbour cells that live in the halo
    region for boundary cells; the halo cells came from either neighbour
    shards (:func:`halo_exchange`) or a local periodic-pad fallback.

    Parameters
    ----------
    f_pad : (*grid_padded, Q) float32
        Halo-padded distribution.  Each spatial axis ``d`` has size
        ``grid_local[d] + 2*halo``.
    e : (Q, D) numpy int array
    ndim : int
        Number of spatial dimensions (2 or 3).
    halo : int
        Halo width on each side of every spatial axis (1 for D3Q19 / D2Q9).

    Returns
    -------
    f_streamed : (*grid_local, Q) float32
        Streamed distributions on the interior cells (halo stripped).
    """
    Q = e.shape[0]
    slices = []
    for q in range(Q):
        fq = f_pad[..., q]
        for d in range(ndim):
            shift = int(e[q, d])
            n_d = fq.shape[d] - 2 * halo
            start = halo - shift
            fq = jax.lax.slice_in_dim(fq, start, start + n_d, axis=d)
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


def _zou_he_face_closure(e, w, cs2, face_axis, face_side):
    """Static (numpy) data of the Zou-He closure on one flat face.

    Classifies the lattice directions for the face and checks, at trace
    time, the lattice identities the closure in
    :func:`_zou_he_pressure_face` relies on.  Each one holds for D2Q9 and
    D3Q19 on every face; a velocity set for which one fails would make
    the closure impose a density or a tangential velocity other than the
    one it claims, so it is refused instead of run.

    Returns
    -------
    known, unknown, tangential : list of int
        As :func:`_classify_directions`.
    sigma : int
        ``+1`` on a ``"min"`` face, ``-1`` on a ``"max"`` face: the sign of
        the face-normal component of every *unknown* direction, i.e. the
        direction pointing into the domain.
    tang_axes : list of int
        The spatial axes other than ``face_axis``.
    tang_weight : dict[int, float]
        ``1 / sum_{q in unknown} e_{q,t}^2`` for each tangential axis
        ``t`` -- the transverse-momentum coefficient (``1/2`` for both
        supported lattices).
    opp : numpy.ndarray
        Opposite-direction map.
    """
    e = np.asarray(e)
    w = np.asarray(w, dtype=np.float64)
    ndim = e.shape[1]
    known, unknown, tangential = _classify_directions(e, face_axis, face_side)
    sigma = 1 if face_side == "min" else -1
    opp = _get_opp_map(e)
    tang_axes = [a for a in range(ndim) if a != face_axis]

    problems = []
    if sorted(int(opp[q]) for q in unknown) != sorted(known):
        problems.append("the opposites of the unknown directions are not "
                        "exactly the known directions")
    if np.any(np.abs(e) > 1):
        problems.append("a velocity component exceeds one lattice unit")
    e_u = e[unknown].astype(np.float64)
    w_u = w[unknown]
    # Normal-momentum bounce-back term sums to rho * u_n over the unknowns.
    if not np.isclose(2.0 * np.sum(w_u * e_u[:, face_axis] ** 2) / cs2, 1.0):
        problems.append("2 sum_{unknown} w_q e_qn^2 / cs2 != 1")
    tang_weight: dict[int, float] = {}
    for t in tang_axes:
        # The transverse correction must leave density and normal momentum
        # alone and act on one tangential axis at a time.
        if not np.isclose(np.sum(e_u[:, t]), 0.0):
            problems.append(f"sum_{{unknown}} e_q{t} != 0")
        if not np.isclose(np.sum(w_u * e_u[:, face_axis] * e_u[:, t]), 0.0):
            problems.append(f"sum_{{unknown}} w_q e_qn e_q{t} != 0")
        for t2 in tang_axes:
            if t2 != t and not np.isclose(np.sum(e_u[:, t] * e_u[:, t2]), 0.0):
                problems.append(f"sum_{{unknown}} e_q{t} e_q{t2} != 0")
        s_tt = float(np.sum(e_u[:, t] ** 2))
        if s_tt <= 0.0:
            problems.append(f"no unknown direction has a component on axis {t}")
        else:
            tang_weight[t] = 1.0 / s_tt
    if problems:
        raise ValueError(
            "the Zou-He pressure closure does not apply to this velocity set "
            f"on face (axis {face_axis}, {face_side}): " + "; ".join(problems)
        )
    return known, unknown, tangential, sigma, tang_axes, tang_weight, opp


def _zou_he_pressure_face(f, prescribed_density, e, w, cs2,
                          face_axis, face_side, wall_mask):
    """Apply the Zou-He pressure (density) boundary condition on a flat face.

    After this call every fluid cell of the face has exactly the
    prescribed density ``rho_p`` (to rounding) and zero tangential
    velocity; the known and tangential populations are not modified, and
    wall cells on the face are left untouched.

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

    Notes
    -----
    The closure of Zou & He (1997) for D2Q9 and Hecht & Harting (2010)
    for D3Q19.  With ``n`` the face axis, ``sigma = +1`` on a ``"min"`` face and
    ``-1`` on a ``"max"`` face, the unknown set ``U`` (directions with
    ``e_qn = sigma``, arriving from outside), the known set ``K``
    (``e_qn = -sigma``, streamed out of the interior) and the tangential
    set ``T`` (``e_qn = 0``, rest included)::

        rho_p        = S_T + S_K + S_U
        rho_p * u_n  = sigma * (S_U - S_K)
        =>  u_n      = sigma * (1 - (S_T + 2 S_K) / rho_p)

    and, for each unknown ``q`` with opposite ``qbar``, non-equilibrium
    bounce-back of the normal part plus the transverse-momentum
    correction that makes the tangential velocity zero::

        f_q = f_qbar + (2 w_q / cs2) * rho_p * e_qn * u_n
                     - sum_{t != n} e_qt * N_t / (sum_{p in U} e_pt^2),
        N_t = sum_{p in T} f_p e_pt.

    On D2Q9 ``x_min`` (this module's numbering) that is
    ``f1 = f2 + 2/3 rho u``, ``f5 = f8 + 1/6 rho u - 1/2 (f3 - f4)``,
    ``f7 = f6 + 1/6 rho u + 1/2 (f3 - f4)``: Zou & He's equations with
    ``u_y = 0``.  The factor 2 on ``S_K`` is what makes the rebuilt
    density equal ``rho_p``; without it the face density comes out as
    ``rho_p + S_K`` (MADD-ANO-020).  The lattice identities the closure
    rests on are checked by :func:`_zou_he_face_closure`.
    """
    ndim = e.shape[1]
    (known, unknown, tangential, sigma, tang_axes, tang_weight,
     opp_map) = _zou_he_face_closure(e, w, cs2, face_axis, face_side)

    # Build slice for the face
    face_slices: list[Any] = [slice(None)] * ndim
    if face_side == "min":
        face_slices[face_axis] = 0
    else:
        face_slices[face_axis] = -1
    face_sl = tuple(face_slices)

    # Extract face distributions: shape (*face_shape, Q)
    f_face = f[face_sl]
    wall_face = wall_mask[face_sl]

    # Sums of the known and tangential populations at the face.
    sum_known = jnp.zeros_like(f_face[..., 0])
    for q in known:
        sum_known = sum_known + f_face[..., q]
    sum_tang = jnp.zeros_like(f_face[..., 0])
    for q in tangential:
        sum_tang = sum_tang + f_face[..., q]

    # Face-normal velocity (component along +face_axis) from the density
    # and normal-momentum moments.  Known populations count twice: once in
    # the density and once, reflected, in the momentum.
    rho_p = prescribed_density
    u_normal = sigma * (
        1.0 - (sum_tang + 2.0 * sum_known) / jnp.maximum(rho_p, 1e-10)
    )

    # Tangential momentum carried by the tangential populations; the
    # unknowns are corrected so that the face's tangential momentum is 0.
    n_t = {}
    for t in tang_axes:
        acc = jnp.zeros_like(f_face[..., 0])
        for q in tangential:
            if int(e[q, t]) != 0:
                acc = acc + float(e[q, t]) * f_face[..., q]
        n_t[t] = acc

    f_face_new = f_face
    for q in unknown:
        opp_q = int(opp_map[q])
        f_q_new = (
            f_face[..., opp_q]
            + (2.0 * float(w[q]) / cs2) * rho_p * float(e[q, face_axis]) * u_normal
        )
        for t in tang_axes:
            if int(e[q, t]) != 0:
                f_q_new = f_q_new - float(e[q, t]) * tang_weight[t] * n_t[t]
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
        # 1.1.0: Zou-He closure corrected (MADD-ANO-020); see the guide.
        algorithm_version="1.1.0",
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
        discretization_order=DiscretizationOrder(
            spatial=2.0,
            temporal=None,
            notes=(
                "2nd order in the grid spacing under diffusive scaling "
                "(lattice viscosity held fixed, velocity scaled with 1/N), "
                "measured by MADD-VER-007.  No temporal order is declared "
                "because there is no independent timestep: the lattice fixes "
                "dx = dt = 1 and ``update`` ignores its ``dt`` argument, so "
                "refining time *is* refining the grid.  Degrades to 1st "
                "order at bounce-back walls (see ``limitations``); "
                "the measurement is on a wall-free periodic domain."
            ),
        ),
        assumptions=(
            "Incompressible flow (Mach number << 1)",
            "BGK single-relaxation-time collision operator",
            "Rigid, impermeable walls (bounce-back in wall cells)",
            "Zou-He pressure boundary conditions at inlet/outlet: the face "
            "carries the prescribed density and zero tangential velocity",
        ),
        limitations=(
            "Compressibility errors at high Mach number (Ma > 0.1)",
            "BGK is less stable than MRT for high Reynolds numbers",
            "No turbulence model",
            "Wall bounce-back is 1st order, straight walls included: wall "
            "cells collide as well as reflect, and the hydrodynamic wall sits "
            "about 0.1 lattice units from the wall node rather than mid-link "
            "(measured by MADD-VER-016)",
            "Pressure faces reflect acoustic waves, so a pressure-driven flow "
            "settles more slowly than its viscous time scale",
        ),
        validated_regimes=(
            ValidatedRegime(
                "tau", 0.501, 2.0,
                notes="tau > 0.5 required; tau >> 1 causes numerical diffusion",
            ),
            ValidatedRegime(
                "Reynolds number", 0, 100,
                notes=(
                    "Validated against Poiseuille analytical solutions, "
                    "body-force (MADD-VER-003) and pressure-driven "
                    "(MADD-VER-016)"
                ),
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

        # Wall mask: convert to JAX array for JIT compatibility.
        # Track at Python level whether any walls are present so the
        # sharded `update_padded` can guard cheaply outside of tracing.
        if wall_mask is not None:
            if wall_mask.shape != tuple(grid_shape):
                raise ValueError(
                    f"wall_mask shape {wall_mask.shape} != grid_shape "
                    f"{tuple(grid_shape)}"
                )
            self._has_walls = bool(np.any(np.asarray(wall_mask)))
            self._wall_mask = jnp.asarray(wall_mask, dtype=jnp.bool_)
        else:
            self._has_walls = False
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

        # wall_mask is part of state so that ShardedStencilNode shards it
        # per device alongside f / density / velocity / pressure.  Stored
        # as uint8 (not bool) so coupling/adaptive code paths that take
        # state differences (state_new - state_old) keep working -- bool
        # subtraction is rejected by JAX.  Cast to bool at the BC site.
        return {
            "f": f,
            "density": density,
            "velocity": velocity,
            "pressure": pressure,
            "wall_mask": self._wall_mask.astype(jnp.uint8),
        }

    def update_padded(
        self,
        state_padded: dict,
        boundary_inputs: dict,
        dt: float,
        *,
        static_padded: dict | None = None,
        shard_info: dict | None = None,
        params=None,
    ) -> dict:
        """Halo-aware LBM step.

        ``state_padded`` carries halo-padded fields along every spatial
        axis declared in :meth:`halo_width`.  The streaming step reads
        across the halo via :func:`_stream_padded`; collision runs on
        the entire padded array (cheap and keeps the post-stream values
        consistent with what the neighbour shard would compute); wall
        bounce-back and Zou-He pressure BCs apply on the *interior*
        after streaming.

        Wall bounce-back uses ``state_padded["wall_mask"]`` (which has
        been halo-exchanged the same way as ``f``), so each shard sees
        its own wall topology.  Zou-He pressure BCs at
        ``inlet_face`` / ``outlet_face`` operate on the local slice of
        the face plane and are valid as long as the inlet/outlet axis
        is *not* sharded in the pencil mesh.  For the canonical
        Hagen-Poiseuille setup we shard ``(spatial_y, spatial_z)`` and
        leave ``x`` (streamwise) replicated -- which puts every shard
        in possession of the full inlet/outlet face.
        """
        f_pad = state_padded["f"]
        # Same contract as ``update``: viscosity from the injected params
        # when the (sharded) graph supplies them.
        p = self.params if params is None else {**self.params, **params}
        tau = 0.5 + p["viscosity"] / self._cs2
        lat = self._lat
        e = lat.e
        w = lat.w
        cs2 = lat.cs2
        D = self._D
        ndim = D
        halo = 1  # D3Q19/D2Q9 unit-step neighbours

        interior_slicer = tuple(
            slice(halo, -halo) if d < ndim else slice(None)
            for d in range(f_pad.ndim)
        )
        interior_slicer_scalar = interior_slicer[:ndim]

        # Wall mask: prefer the padded one from state, fall back to
        # boundary_input override, fall back to legacy node attribute.
        # Stored as uint8 in state -- cast to bool for the BC paths.
        if "wall_mask_update" in boundary_inputs:
            wall_pad = boundary_inputs["wall_mask_update"].astype(jnp.bool_)
        elif "wall_mask" in state_padded:
            wall_pad = state_padded["wall_mask"].astype(jnp.bool_)
        else:
            wall_pad = self._wall_mask

        # Body force first (needed for the Guo half-correction below).
        # Three input shapes are accepted:
        #   (D,)                      -- uniform, broadcast across grid
        #   density_pad.shape + (D,)  -- already padded local field
        #   density_pad.interior shape (e.g. (nx,ny,nz,D)) -- pad locally
        # The first form is the right call under sharding: send a
        # constant vector, broadcast trivially per device.
        target_density_shape = f_pad.shape[:-1]  # padded spatial shape
        force = boundary_inputs.get("body_force", None)
        if force is None:
            force = jnp.zeros(target_density_shape + (D,), dtype=jnp.float32)
        elif force.ndim == 1 and force.shape == (D,):
            force = jnp.broadcast_to(
                force, target_density_shape + (D,),
            ).astype(jnp.float32)
        elif force.shape != target_density_shape + (D,):
            pad_widths = [
                (halo, halo) if d < ndim else (0, 0)
                for d in range(force.ndim)
            ]
            force = jnp.pad(force, pad_widths)

        # BGK collision is purely local, so we collide on the *entire*
        # padded array -- halo cells included -- so that the post-stream
        # values read across the halo match what the neighbour shard
        # would produce.  Halo cells carry the neighbour's pre-collision
        # distribution from halo_exchange; colliding them locally is
        # algebraically identical to colliding them on the neighbour.
        density_pad, velocity_pad = _compute_macroscopic(f_pad, e, force=force)

        f_eq_pad = _equilibrium(density_pad, velocity_pad, e, w, cs2)
        f_post_pad = f_pad - (f_pad - f_eq_pad) / tau
        f_post_pad = f_post_pad + _guo_forcing(
            density_pad, velocity_pad, force, tau, e, w, cs2,
        )

        # Streaming reads post-collision values from the halo; output
        # has the interior shape.
        f_streamed_interior = _stream_padded(f_post_pad, e, ndim, halo)

        # Wall bounce-back on the interior using the interior slice of
        # the wall mask.
        wall_interior = wall_pad[interior_slicer_scalar]
        opp = lat.opp
        f_bounced = f_streamed_interior[..., opp]
        wall_expanded = wall_interior[..., None]
        f_bc_interior = jnp.where(wall_expanded, f_bounced, f_streamed_interior)

        # Zou-He pressure BCs on the inlet / outlet face (interior shape).
        # Valid under sharding provided the inlet/outlet axis is replicated
        # (not in axis_map); each shard then owns the full face and
        # applies the BC on its local slice of the face plane.
        inlet_pressure = boundary_inputs.get("inlet_pressure", None)
        outlet_pressure = boundary_inputs.get("outlet_pressure", None)
        if inlet_pressure is not None:
            inlet_axis, inlet_side = _FACE_MAP[self._inlet_face]
            inlet_rho = inlet_pressure / cs2
            f_bc_interior = _zou_he_pressure_face(
                f_bc_interior, inlet_rho, e, w, cs2,
                inlet_axis, inlet_side, wall_interior,
            )
        if outlet_pressure is not None:
            outlet_axis, outlet_side = _FACE_MAP[self._outlet_face]
            outlet_rho = outlet_pressure / cs2
            f_bc_interior = _zou_he_pressure_face(
                f_bc_interior, outlet_rho, e, w, cs2,
                outlet_axis, outlet_side, wall_interior,
            )

        # Macroscopic output: apply Guo half-correction with the
        # interior slice of the body force.
        force_interior = force[interior_slicer]
        density_new, velocity_new = _compute_macroscopic(
            f_bc_interior, e, force=force_interior,
        )
        # Zero velocity in wall cells (cosmetic, for cleaner output).
        velocity_new = jnp.where(wall_expanded, 0.0, velocity_new)
        pressure_new = density_new * cs2

        # Re-pad the outputs so the wrapper can strip uniformly.  Halo
        # entries are placeholders; values inside the interior are the
        # final post-stream result.
        f_new_pad = f_pad.at[interior_slicer].set(f_bc_interior)
        density_pad_out = state_padded["density"].at[interior_slicer_scalar].set(density_new)
        velocity_pad_out = state_padded["velocity"].at[interior_slicer].set(velocity_new)
        pressure_pad_out = state_padded["pressure"].at[interior_slicer_scalar].set(pressure_new)

        result: dict = {
            "f": f_new_pad,
            "density": density_pad_out,
            "velocity": velocity_pad_out,
            "pressure": pressure_pad_out,
        }
        if "wall_mask" in state_padded:
            result["wall_mask"] = state_padded["wall_mask"]
        return result

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            **super().param_specs(),
            # tau = 0.5 + nu / cs2 must stay > 0.5, i.e. nu > 0 strictly.
            "viscosity": ParamSpec(
                bounds=(0.0, None), transform="log", units="lattice",
                description="kinematic viscosity; tau = 0.5 + nu / cs2",
            ),
        }

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """One collide-stream step.  ``viscosity`` comes from the injected
        ``params`` when the graph supplies them (so ``tau`` is a traced,
        differentiable constant); the lattice, faces and wall geometry are
        structural and always come from the node."""
        f = state["f"]
        p = self.params if params is None else {**self.params, **params}
        tau = 0.5 + p["viscosity"] / self._cs2
        lat = self._lat
        e = lat.e
        w = lat.w
        cs2 = lat.cs2
        Q = lat.Q
        D = self._D
        ndim = D

        wall_mask = self._runtime_wall_mask(state, boundary_inputs)

        # 1. Body force (Guo forcing).  Accept either the full grid shape
        # or a uniform (D,) vector which we broadcast.
        force = boundary_inputs.get("body_force", None)
        if force is None:
            force = jnp.zeros(self._grid_shape + (D,), dtype=jnp.float32)
        elif force.ndim == 1 and force.shape == (D,):
            force = jnp.broadcast_to(
                force, self._grid_shape + (D,),
            ).astype(jnp.float32)

        # 2. Macroscopic from current f, with Guo F/2 half-correction
        density, velocity = _compute_macroscopic(f, e, force=force)

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

        # 7. Compute macroscopic for output (with Guo half-correction)
        density_new, velocity_new = _compute_macroscopic(f_bc, e, force=force)

        # Zero velocity in wall cells (cosmetic, for cleaner output)
        velocity_new = jnp.where(wall_expanded, 0.0, velocity_new)

        pressure_new = density_new * cs2

        result: dict = {
            "f": f_bc,
            "density": density_new,
            "velocity": velocity_new,
            "pressure": pressure_new,
        }
        if "wall_mask" in state:
            result["wall_mask"] = state["wall_mask"]
        return result

    def _runtime_wall_mask(self, state: dict, boundary_inputs: dict):
        """The wall mask ``update`` applies, as a bool array.

        Precedence: ``boundary_inputs["wall_mask_update"]`` (explicit
        override), then ``state["wall_mask"]`` (the stateful path, stored as
        uint8), then the constructor's mask (a state built before the mask
        joined it).  :meth:`compute_boundary_fluxes` reads the same mask, so
        the outlet average excludes exactly the cells the step treated as
        walls.
        """
        if "wall_mask_update" in boundary_inputs:
            return boundary_inputs["wall_mask_update"].astype(jnp.bool_)
        if "wall_mask" in state:
            return state["wall_mask"].astype(jnp.bool_)
        return self._wall_mask

    def derivatives(self, state: dict, boundary_inputs: dict, *, params=None) -> dict:
        """Not applicable for LBM (discrete update, not an ODE).

        Declared with the ``params`` keyword so the signature matches the
        contract even though it only raises."""
        raise NotImplementedError(
            "LBMNode uses a discrete lattice Boltzmann update, "
            "not a continuous ODE. derivatives() is not applicable."
        )

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Expose average pressure at the outlet face for coupling.

        The average is over the fluid cells of the outlet face under the
        *runtime* wall mask -- the one :meth:`update` applied
        (``wall_mask_update``, else the ``wall_mask`` state field, else the
        constructor's) -- so a wall injected at run time is excluded here
        too.  Every such fluid cell carries the imposed outlet pressure
        after a step, so with an ``outlet_pressure`` input this returns it.

        Reads no constants, so ``params`` is accepted for the contract only."""
        pressure = state["pressure"]
        outlet_axis, outlet_side = _FACE_MAP[self._outlet_face]

        # Build slice for the outlet face
        # `list[Any]`: the entries start as slices and the outlet face is
        # then replaced by an index, which narrows the element type away.
        face_slices: list[Any] = [slice(None)] * self._D
        if outlet_side == "min":
            face_slices[outlet_axis] = 0
        else:
            face_slices[outlet_axis] = -1
        face_sl = tuple(face_slices)

        p_face = pressure[face_sl]
        # Mask out wall cells at the outlet face: the runtime mask, as update.
        wall_face = self._runtime_wall_mask(state, boundary_inputs)[face_sl]
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
