"""
HeatNode -- 1D heat diffusion on a rod using explicit finite differences.

Models the 1D heat equation on a uniform or non-uniform grid::

    dT/dt = alpha * d^2T/dx^2 + source

with Dirichlet boundary conditions at both ends.  The entire ``update``
uses ``jnp`` operations (no Python loops), so it is fully JAX-traceable
and JIT-compilable.

Supports:
- 2nd-order (default) and 4th-order central difference stencils
  (``stencil_order=2`` or ``stencil_order=4``)
- Non-uniform grids via ``grid_points`` parameter
- USD geometry source via ``geometry_source`` attribute

Grid and boundary convention
----------------------------
The grid is **cell centred**: cell ``i`` sits at ``x_i = (i + 1/2) dx``
with ``dx = length / n_cells``, so the rod ends ``x = 0`` and ``x = L``
lie half a cell outside the first and last cell centre.  The Dirichlet
data ``left_temperature`` / ``right_temperature`` is the temperature
**at those rod ends**, which is what ``boundary_input_spec`` has always
documented, and it is imposed through a ghost cell rather than by
overwriting the end cells.

Before 0.4.0 the node instead wrote the Dirichlet value straight into
the first and last cell, which placed it half a cell inside the rod and
cost a full order of accuracy (MADD-ANO-007), and the 4th-order
stencil's ghosts were built at the wrong positions (MADD-ANO-008).
Both are fixed here; see ``docs/algorithm_guide/nodes/heat_node.md``.
"""

from typing import Optional

import jax.numpy as jnp

from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import (
    DiscretizationOrder,
    NodeMeta,
    StabilityLevel,
    ValidatedRegime,
)
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec


#: Largest Fourier number ``dt * alpha / dx**2`` at which the explicit
#: update is stable, per ``stencil_order``.
#:
#: Both entries are the von Neumann / spectral bound of the *whole*
#: discrete operator, boundary rows included, not of the interior
#: symbol alone -- the boundary closure is what sets the 4th-order
#: figure.
#:
#: * ``2`` -> exactly 1/2.  The mirror ghost ``2*T_b - T[0]`` is exact
#:   on the eigenvectors ``sin(m pi x / L)`` of the cell-centred
#:   operator, so the closure adds nothing to the interior spectrum and
#:   the classical bound survives untouched.  Verified numerically for
#:   n = 5..320.
#: * ``4`` -> 5/16 = 0.3125, conservative.  The 5-point interior symbol
#:   alone would allow 3/8, but the cubic ghost closure that makes the
#:   stencil actually 4th-order accurate pushes the spectral radius up.
#:   The sharp bound depends on ``n_cells``: 0.3169 at n = 5, rising
#:   monotonically to 0.3249 as n -> infinity.  5/16 sits below all of
#:   them, so one number is safe at every resolution the node accepts.
#:
#: To re-derive either figure: form the operator matrix column by column
#: (one ``_compute_laplacian`` call per unit vector, with T_b = 0) and
#: bisect on ``max|1 + Fo*lambda| <= 1`` over its eigenvalues.
#: ``tests/verification/test_mms_order.py`` pins both against runs of the
#: node itself, on either side of the bound.
MAX_FOURIER_NUMBER = {2: 0.5, 4: 5.0 / 16.0}


def _laplacian_2nd_order_uniform(T_padded, dx):
    """2nd-order central difference Laplacian on a uniform grid."""
    return (T_padded[2:] - 2.0 * T_padded[1:-1] + T_padded[:-2]) / (dx * dx)


def _dirichlet_ghosts_2nd_order(T, T_boundary):
    """The single ghost value that puts ``T_boundary`` on the rod end.

    The ghost cell centre is at ``-dx/2`` and the boundary at ``x = 0``
    is the midpoint between it and the first cell centre, so linear
    reconstruction gives ``T_ghost = 2*T_b - T[0]``.

    This is the conservative (finite-volume) closure: the resulting
    first row is ``[(T[1] - T[0])/dx - (T[0] - T_b)/(dx/2)] / dx``, a
    flux difference across the cell, and the boundary flux is the one
    evaluated *at the rod end*.  Its local truncation error is O(1) --
    the face gradient is only 1st-order accurate at ``x = 0`` -- but
    the conservative form recovers a globally 2nd-order scheme, which
    is the standard supraconvergence result for cell-centred grids and
    is what ``tests/verification/test_mms_order.py`` measures (2.000).

    Parameters
    ----------
    T : array, shape (n,)
        Temperature field, ordered from the boundary inwards.
    T_boundary : scalar
        Dirichlet datum at the rod end.

    Returns
    -------
    scalar
        The ghost value at ``-dx/2``.
    """
    return 2.0 * T_boundary - T[0]


def _dirichlet_ghosts_4th_order(T, T_boundary):
    """The two ghost values the 5-point stencil reads, from the rod end.

    Both are the cubic through ``(0, T_b)`` and the three nearest cell
    centres ``(dx/2, T[0])``, ``(3dx/2, T[1])``, ``(5dx/2, T[2])``,
    evaluated at the ghost cell centres ``-dx/2`` and ``-3dx/2``::

        T(-dx/2)  = (16*T_b - 15*T[0] +  5*T[1] -    T[2]) / 5
        T(-3dx/2) = (64*T_b - 90*T[0] + 40*T[1] -  9*T[2]) / 5

    A cubic is the lowest degree that works.  Its ghost error is
    O(dx^4); the 5-point stencil divides ghost errors by ``12 dx^2``,
    so a lower-degree extrapolation leaves a truncation error at the
    first interior cells that the conservative form can only partly
    absorb.  Measured on three manufactured solutions, linear
    extrapolation caps the scheme at 2.00 and quadratic at 2.99, while
    this cubic reaches 3.95-3.98.  A quartic is accurate enough but
    spectrally unstable -- its Fourier bound is 0.273 and it diverges
    on the same ladder.

    The linear case is worth a sentence of its own, because it is how
    this docstring was wrong on its first draft.  On a manufactured
    solution whose curvature vanishes at the rod ends -- which the one
    in ``tests/verification/test_mms_order.py`` did until 0.4.0 --
    linear extrapolation measures 4.08 and looks correct.  The
    boundary rows' error term is proportional to ``u''`` at the end,
    so a flat-ended profile cannot see it.  Any re-derivation of this
    choice has to use a profile curved at both ends.

    One consequence worth recording: under this closure the 5-point
    and 3-point forms are *algebraically identical* at cells 0 and
    n-1, both reducing to ``(16*T_b - 25*T[0] + 10*T[1] - T[2]) / (5
    dx^2)``.  The 2nd-order fallback in
    :func:`_laplacian_4th_order_uniform` therefore costs nothing at
    all any more; it used to cost an order.

    Parameters
    ----------
    T : array, shape (n,)
        Temperature field, ordered from the boundary inwards; at least
        three cells are read.
    T_boundary : scalar
        Dirichlet datum at the rod end.

    Returns
    -------
    tuple of scalar
        ``(ghost at -3dx/2, ghost at -dx/2)`` -- outermost first, so the
        pair can be concatenated onto the field directly.
    """
    near = (16.0 * T_boundary - 15.0 * T[0] + 5.0 * T[1] - T[2]) / 5.0
    far = (64.0 * T_boundary - 90.0 * T[0] + 40.0 * T[1] - 9.0 * T[2]) / 5.0
    return far, near


def _lagrange_gradient_weights(offsets):
    """Weights ``w`` such that ``p'(0) = sum_j w[j] * v[j]``.

    ``p`` is the polynomial through ``(offsets[j], v[j])``; ``offsets``
    are measured *from the point the gradient is wanted at* -- the rod
    end -- may include ``0.0`` itself, which is where the Dirichlet
    datum goes, and must be distinct.  With ``k + 1`` points the
    interpolant has degree ``k`` and the gradient is ``O(h**k)``.

    **Pure Python arithmetic, deliberately.**  A cell-centred grid's
    offsets are known before anything is traced: on a uniform grid they
    are the fixed multiples ``(2j+1)/2`` of ``dx``, and on a
    non-uniform one they are entries of ``grid_points``, which is
    ``ParamSpec(trainable=False)``.  So the whole Lagrange construction
    folds away at graph-build time and the traced program is one dot
    product -- ``k + 1`` multiplies, ``k`` adds and, on the uniform
    grid, one divide by ``dx``.

    Writing the same construction as traced arithmetic costs O(k^3)
    jaxpr primitives per rod end, which is not hypothetical: it took
    ``heat_chain`` (two 64-cell rods, ``stencil_order=4``) from 243
    jaxpr primitives to 557 in the first version of this reconstruction,
    for a program XLA then constant-folded back to the same 287 HLO
    ops.  The lowered graph was identical and the graph *builder* did
    2.3x the work -- invisible to an HLO count and squarely what
    ``benchmarks/compile_counts_baseline.json`` exists to catch.

    Because the weights carry the offsets' units, the uniform grid
    passes dimensionless offsets and divides the result by ``dx``
    afterwards.  That keeps ``length`` -- which is trainable -- in the
    traced program as the single factor the flux actually depends on,
    exactly as the closed form says it should.

    Parameters
    ----------
    offsets : sequence of float
        Signed distances from the evaluation point, all distinct.

    Returns
    -------
    tuple of float
        One weight per offset, in the same order.
    """
    k = len(offsets)
    weights = []
    for j in range(k):
        denom = 1.0
        for m in range(k):
            if m != j:
                denom *= offsets[j] - offsets[m]
        numer = 0.0
        for i in range(k):
            if i == j:
                continue
            term = 1.0
            for m in range(k):
                if m == j or m == i:
                    continue
                term *= -offsets[m]
            numer += term
        weights.append(numer / denom)
    return tuple(weights)


def _rod_end_gradient(values, T_boundary, offsets, scale=None):
    """``dT/ds`` at a rod end, ``s`` measured inwards from that end.

    ``values`` are cell-centre temperatures ordered *outwards from the
    end*, and ``offsets`` their distances from it, so the caller
    reverses both for the right-hand end.  ``T_boundary`` is the
    Dirichlet datum at the end, or ``None`` when no boundary input
    supplies one.  ``scale`` divides the result: the uniform grid
    passes ``dx`` because its offsets are in units of ``dx``, and the
    non-uniform grid passes ``None`` because its offsets are in metres.

    Why this and not ``(T[1] - T[0]) / dx``: that quotient is the
    gradient of the straight line through the first two cell centres,
    which for a cell-centred grid sits at ``x = dx``, a full cell
    inside the end the flux is named for.  It converges to the rod-end
    gradient at 1st order however good the interior stencil is, which
    caps any flux-coupled solve at 1st order through the flux alone.
    Measured on ``T = exp(x)``, ``alpha = 1``, ``n = 10``: it reports
    -1.1056 where the rod-end flux is -1.0, and refines at order 1.005.

    With the datum the polynomial is anchored at the rod end and
    ``stencil_order`` cells make it accurate to ``stencil_order``
    (measured 1.999 and 3.993 -- see
    ``tests/verification/test_mms_order.py::TestTheReportedBoundaryFluxIsAtTheRodEnd``,
    which pins both).  Without it the reconstruction is a pure
    extrapolation; it reads three cells and is 2nd order, and a
    higher-degree extrapolant is not worth its conditioning (the
    coefficient L1 norm per ``dx`` is 6 for three cells against 28.3
    for five) when there is no boundary condition for the flux to be
    consistent with.

    One consequence worth recording: the flux this returns is *not*
    the discrete face flux the conservative update uses at the first
    cell, which is the 1st-order ``-alpha (T[0] - T_b) / (dx/2)``.  The
    two differ by ``O(dx**stencil_order)`` and agree in the limit; the
    reported number is the physical rod-end flux the spec names, which
    is what a coupled neighbour needs, rather than the one that closes
    this node's own discrete energy balance to the last bit.
    """
    if T_boundary is None:
        weights = _lagrange_gradient_weights(tuple(offsets))
        terms = list(values)
    else:
        weights = _lagrange_gradient_weights((0.0, *offsets))
        terms = [T_boundary, *values]
    grad = weights[0] * terms[0]
    for weight, term in zip(weights[1:], terms[1:]):
        grad = grad + weight * term
    return grad if scale is None else grad / scale


def _laplacian_4th_order_pure(T_padded, dx):
    """4th-order central-difference Laplacian, no boundary fallback.

    Suitable when ``T_padded`` has *real* ghost cells on each side
    (e.g. coming from a halo exchange), so no cell needs a lower-order
    fallback near the edge.
    """
    dx2 = dx * dx
    return (
        -T_padded[:-4]
        + 16.0 * T_padded[1:-3]
        - 30.0 * T_padded[2:-2]
        + 16.0 * T_padded[3:-1]
        - T_padded[4:]
    ) / (12.0 * dx2)


def _laplacian_4th_order_uniform(T_padded, dx):
    """4th-order central difference Laplacian on a uniform grid.

    Interior: (-T[i+2] + 16*T[i+1] - 30*T[i] + 16*T[i-1] - T[i-2]) / (12*dx^2)
    Boundary cells (i=0 and i=n-1 only): fall back to 2nd-order.  Cell 1
    and cell n-2 keep the 5-point form -- with two ghosts per side they
    have a full stencil -- and under the cubic ghost closure the
    fallback is free anyway, because the two forms are algebraically
    identical at cells 0 and n-1 (see
    :func:`_dirichlet_ghosts_4th_order`).

    ``T_padded`` has shape ``(n+4,)`` with two ghost cells on each side.
    """
    n = T_padded.shape[0] - 4  # original cell count
    dx2 = dx * dx

    lap_4th = _laplacian_4th_order_pure(T_padded, dx)

    # 2nd-order for fallback (centred on the same cells)
    lap_2nd = (
        T_padded[3:-1] - 2.0 * T_padded[2:-2] + T_padded[1:-3]
    ) / dx2

    # Use 2nd-order for the first and last cells, 4th-order elsewhere
    idx = jnp.arange(n)
    use_4th = (idx >= 1) & (idx <= n - 2)
    lap = jnp.where(use_4th, lap_4th, lap_2nd)
    return lap


def _laplacian_nonuniform(T_padded, x_padded):
    """2nd-order Laplacian on a non-uniform grid.

    d^2T/dx^2 ~ 2 * [(T[i+1]-T[i])/(x[i+1]-x[i]) - (T[i]-T[i-1])/(x[i]-x[i-1])]
                    / (x[i+1] - x[i-1])

    ``T_padded`` has shape ``(n+2,)`` with one ghost cell on each side.
    ``x_padded`` has shape ``(n+2,)`` with matching coordinates.
    """
    dx_right = x_padded[2:] - x_padded[1:-1]  # x[i+1] - x[i]
    dx_left = x_padded[1:-1] - x_padded[:-2]  # x[i] - x[i-1]
    dx_sum = x_padded[2:] - x_padded[:-2]     # x[i+1] - x[i-1]

    grad_right = (T_padded[2:] - T_padded[1:-1]) / dx_right
    grad_left = (T_padded[1:-1] - T_padded[:-2]) / dx_left

    # Avoid division by zero
    dx_sum_safe = jnp.where(dx_sum > 0, dx_sum, 1.0)
    return 2.0 * (grad_right - grad_left) / dx_sum_safe


@stability(StabilityLevel.STABLE)
class HeatNode(SimulationNode):
    """1D heat diffusion on a rod with Dirichlet boundary conditions.

    Solves the heat equation using explicit finite differences on a
    uniform or non-uniform grid of *n_cells* cells spanning a rod of
    the given *length*.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    n_cells : int
        Number of grid cells (default 10).
    length : float
        Physical length of the rod (default 1.0).
    thermal_diffusivity : float
        Thermal diffusivity alpha in m^2/s (default 0.01).
    initial_temperature : float or array-like
        Uniform initial temperature (scalar) or per-cell initial
        temperature array of length *n_cells* (default 0.0).
    stencil_order : int
        Order of the finite difference stencil (2 or 4, default 2).
        4th-order requires at least 5 cells and falls back to
        2nd-order at boundary cells.
    grid_points : array-like or None
        Optional non-uniform grid point coordinates of shape
        ``(n_cells,)``.  When provided, the node uses variable-dx
        finite differences.  ``length`` is ignored.
    geometry_source : str or None
        Optional SdfPath to a USD prim from which to read grid
        coordinates.  Populate via :func:`load_grid_from_usd`.

    Boundary inputs
    ---------------
    left_temperature : scalar
        Dirichlet BC at the left end.  Defaults to the current left
        cell temperature if not supplied.
    right_temperature : scalar
        Dirichlet BC at the right end.  Defaults to the current right
        cell temperature if not supplied.
    heat_source : scalar or array of shape (n_cells,)
        Volumetric heat source term.  A scalar is broadcast to all
        cells.  Defaults to 0.0 if not supplied.
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-005",
        algorithm_version="2.0.0",
        stability=StabilityLevel.STABLE,
        description="1D heat diffusion on a rod with Dirichlet BCs",
        governing_equations="dT/dt = alpha * d^2T/dx^2 + S",
        discretization="Explicit finite difference, 2nd or 4th-order central in space, 1st-order forward Euler in time",
        # The default (``stencil_order=2``) claim; the per-instance
        # ``discretization_order()`` below reports the configured stencil.
        discretization_order=DiscretizationOrder(
            spatial=2.0,
            temporal=1.0,
            notes=(
                "Central differences of order ``stencil_order`` in space, "
                "forward Euler in time.  Dirichlet data is the temperature "
                "at the rod ends (x = 0 and x = L) and is imposed through "
                "the ghost cells, so the claim holds for the boundary "
                "convention boundary_input_spec documents.  Measured by the "
                "Method of Manufactured Solutions: 2.000 for stencil_order=2 "
                "over a 10/20/40/80/160 ladder (MADD-VER-005) and 3.957 for "
                "stencil_order=4 over the same ladder."
            ),
        ),
        assumptions=(
            "Constant thermal diffusivity (no temperature dependence)",
            "1D geometry (rod)",
            "Dirichlet boundary conditions at both ends, imposed at the rod ends x=0 and x=L",
        ),
        limitations=(
            "Stability limit depends on the stencil: Fourier number dt*alpha/dx^2 < 1/2 for stencil_order=2, < 5/16 for stencil_order=4 (MADD-ANO-009).  The constructor rejects a configuration above its limit; a dt or alpha supplied later to update() is not checked (MADD-ANO-002)",
            "1st-order in time -- temporal accuracy is O(dt)",
            "No convection or radiation terms",
            "Non-uniform grids are 2nd-order only; stencil_order=4 requires a uniform grid",
        ),
        validated_regimes=(
            ValidatedRegime("thermal_diffusivity", 1e-6, 1.0, "m^2/s"),
            ValidatedRegime("n_cells", 4, 1000, notes="Convergence verified up to 1000 cells"),
            ValidatedRegime(
                "CFL", 0.0, 0.5,
                notes=(
                    "dt * alpha / dx^2 < 1/2 for the default stencil_order=2 "
                    "(exact spectral bound).  For stencil_order=4 the limit "
                    "is 5/16 = 0.3125, not 1/2 -- see MADD-ANO-009 and "
                    "maddening.nodes.heat.MAX_FOURIER_NUMBER"
                ),
            ),
        ),
        hazard_hints=(
            "CFL is checked only against the constructor's timestep, thermal_diffusivity and length; a calibrated or externally supplied dt/alpha can still go unstable silently (MADD-ANO-002)",
            "No runtime validation of thermal_diffusivity > 0",
        ),
        implementation_map={
            "alpha * d^2T/dx^2 (diffusion)": "maddening.nodes.heat.HeatNode._compute_laplacian",
            "S (source term)": "maddening.nodes.heat.HeatNode.update",
            "Time integration (dT/dt)": "maddening.nodes.heat.HeatNode.update",
            "Boundary conditions": "maddening.nodes.heat._dirichlet_ghosts_2nd_order",
        },
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        n_cells: int = 10,
        length: float = 1.0,
        thermal_diffusivity: float = 0.01,
        initial_temperature: float = 0.0,
        stencil_order: int = 2,
        grid_points=None,
        geometry_source: Optional[str] = None,
    ):
        if stencil_order not in (2, 4):
            raise ValueError(
                f"stencil_order must be 2 or 4, got {stencil_order}"
            )
        if stencil_order == 4 and n_cells < 5:
            raise ValueError(
                "4th-order stencil requires at least 5 cells, "
                f"got n_cells={n_cells}"
            )

        # Reject a configuration that is unconditionally unstable.  The
        # explicit scheme above its Fourier limit does not degrade, it
        # diverges to NaN in tens of steps with no warning, and the
        # limit for stencil_order=4 (5/16) is not the one anybody would
        # guess from the literature's 1/2 -- that combination is
        # MADD-ANO-009 and it is why this is an error rather than a
        # documented caveat.  See MAX_FOURIER_NUMBER.
        #
        # This covers the constructor's own numbers only.  ``dt`` passed
        # to ``update()``, and ``thermal_diffusivity`` / ``length``
        # injected by a calibration run (both are trainable), bypass it,
        # and cannot be checked without a host callback inside a traced
        # step.  MADD-ANO-002 stays open for that reason.
        try:
            concrete = (
                float(timestep), float(length), float(thermal_diffusivity),
            )
        except (TypeError, ValueError):
            # A traced or otherwise non-concrete constructor argument.
            # Skip rather than fail: the check is a convenience, and
            # refusing to build the node would be worse than not
            # checking it.
            concrete = None
        if (
            grid_points is None
            and concrete is not None
            and n_cells > 0
            and all(v > 0 for v in concrete)
        ):
            timestep_f, length_f, alpha_f = concrete
            dx = length_f / n_cells
            fourier = timestep_f * alpha_f / (dx * dx)
            limit = MAX_FOURIER_NUMBER[stencil_order]
            if fourier > limit:
                raise ValueError(
                    f"timestep {timestep!r} is unstable for this rod: the "
                    f"Fourier number dt*alpha/dx^2 is {fourier:.4g}, above "
                    f"the {limit:g} limit of the order-{stencil_order} "
                    f"stencil (dx = length/n_cells = {dx:.6g}, alpha = "
                    f"{thermal_diffusivity!r}).  The explicit update "
                    f"diverges to NaN there rather than losing accuracy "
                    f"gracefully.  Use timestep <= "
                    f"{limit * dx * dx / alpha_f:.6g}, or more "
                    f"cells, or a smaller thermal_diffusivity."
                )

        # Process grid_points: convert to list for serialisation
        gp_list = None
        if grid_points is not None:
            import numpy as np
            gp_list = list(float(x) for x in np.asarray(grid_points).ravel())
            if len(gp_list) != n_cells:
                raise ValueError(
                    f"grid_points length ({len(gp_list)}) must match "
                    f"n_cells ({n_cells})"
                )

        super().__init__(
            name,
            timestep,
            n_cells=n_cells,
            length=length,
            thermal_diffusivity=thermal_diffusivity,
            initial_temperature=initial_temperature,
            stencil_order=stencil_order,
            grid_points=gp_list,
            geometry_source=geometry_source,
        )

        # v0.2 #3: build the grid coordinates once and expose via
        # static_data so JAX bakes them as constants in the JIT'd
        # step.  Previously ``_grid_x`` rebuilt a fresh JAX array on
        # every property access, which forced retracing whenever the
        # closure object identity changed.
        if gp_list is not None:
            self._grid_x_array = jnp.array(gp_list, dtype=jnp.float32)
        else:
            dx = length / n_cells
            self._grid_x_array = jnp.linspace(
                dx / 2, length - dx / 2, n_cells, dtype=jnp.float32,
            )

    @property
    def static_data(self) -> dict:
        """Reconstructable arrays closed over by :meth:`update`.

        ``grid_x`` is built once in ``__init__`` from ``self.params``
        (specifically ``length``, ``n_cells``, and the optional
        ``grid_points`` list) so a checkpoint/restore round-trip
        rebuilds it from the persisted params alone — see DESIGN.md
        §2 "Static-data channel".

        Sharding policy: ``replication="shard"`` along axis 0 (the
        cell axis), so when the heat rod is sharded across a device
        mesh each shard gets only its slice of the grid coordinates.
        v0.2 #3 follow-up — the GraphManager materialises the per-shard
        slice before the JIT closure captures it.
        """
        from maddening.core.static_data import StaticArray
        return {
            "grid_x": StaticArray(
                value=self._grid_x_array,
                replication="shard",
                shard_axis=0,
            ),
        }

    def static_data_deps(self) -> dict[str, tuple[str, ...]]:
        """Where ``grid_x``'s contents come from, per grid.

        The provenance genuinely differs between the two branches of
        ``__init__``, so the declaration does too.

        * **Non-uniform** (``grid_points`` given): ``grid_x`` *is*
          ``grid_points``, and the variable-dx Laplacian reads it.
          ``grid_points`` is ``ParamSpec(trainable=False)`` -- the grid
          fixes the stencil and is never fitted -- so this is a legal
          dependency, and declaring it is what will let the rebuild hook
          (D10 steps 4 and 5) know the array has to be reconstructed
          when the geometry is rewritten.
        * **Uniform** (the default): nothing is declared, because
          nothing is read.  ``grid_x`` is built here from ``length`` and
          ``n_cells``, but ``_compute_laplacian`` takes ``dx = length /
          n_cells`` from the *traced* ``length`` and never touches
          ``grid_x``, so no value derived from ``length`` is baked into
          the step and the gradient through ``length`` is complete.

        That separation is what keeps ``length`` trainable.  Were the
        uniform path to start reading ``grid_x``, this method would have
        to name ``length`` and ``compile()`` would refuse the graph --
        correctly, because the gradient would at that point be missing
        the term through the grid.  See
        :meth:`~maddening.core.node.SimulationNode.static_data_deps`.
        """
        if self._is_nonuniform:
            return {"grid_x": ("grid_points",)}
        return {}

    def discretization_order(self) -> DiscretizationOrder:
        """Order of accuracy claimed for *this* instance's stencil.

        ``stencil_order`` is a constructor argument, so the claim is
        per-instance and the class-level
        :attr:`NodeMeta.discretization_order` can only carry the
        default.  :func:`maddening.testing.mms.declared_order` prefers
        this hook for exactly that reason.

        Returns
        -------
        DiscretizationOrder
            ``spatial`` is the configured ``stencil_order``;
            ``temporal`` is 1 (forward Euler).

        Notes
        -----
        Both claims are met as of 0.4.0, measured by the Method of
        Manufactured Solutions in
        ``tests/verification/test_mms_order.py``: 2.000 for
        ``stencil_order=2`` and 3.957 for ``stencil_order=4``, over a
        10/20/40/80/160 ladder with the Dirichlet data supplied at the
        rod ends.  Before 0.4.0 the 4th-order stencil measured 0.954
        (MADD-ANO-008) and the documented boundary convention measured
        1.001 (MADD-ANO-007).
        """
        meta = type(self).meta
        assert meta is not None  # every shipped node declares NodeMeta
        return DiscretizationOrder(
            spatial=float(self.params.get("stencil_order", 2)),
            temporal=1.0,
            notes=meta.discretization_order.notes,
        )

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            **super().param_specs(),
            "thermal_diffusivity": ParamSpec(
                bounds=(0.0, None), transform="log", units="m^2/s",
            ),
            "length": ParamSpec(bounds=(0.0, None), transform="log", units="m"),
            # Geometry of the non-uniform grid: read from ``self.params``
            # (it fixes the stencil), never fitted.
            "grid_points": ParamSpec(trainable=False, description="grid geometry"),
        }

    def halo_width(self) -> dict[int, int]:
        """One ghost cell per side per FD stencil radius on axis 0.

        2nd-order central difference needs one neighbour (halo=1).
        4th-order 5-point stencil needs two neighbours (halo=2).
        """
        order = int(self.params.get("stencil_order", 2))
        radius = 1 if order == 2 else 2
        return {0: radius}

    @property
    def _is_nonuniform(self) -> bool:
        return self.params.get("grid_points") is not None

    @property
    def _grid_x(self):
        """Grid point coordinates as a JAX array (alias of static_data['grid_x'])."""
        return self._grid_x_array

    def initial_state(self) -> dict:
        n = self.params["n_cells"]
        T0 = self.params["initial_temperature"]

        # Support both scalar (broadcast) and array initial conditions.
        T0_arr = jnp.asarray(T0, dtype=jnp.float32)
        temperature = jnp.broadcast_to(T0_arr, (n,)).copy()

        return {"temperature": temperature}

    def _compute_laplacian(self, T, T_left, T_right, length=None):
        """Compute the Laplacian d^2T/dx^2 using the configured stencil.

        Parameters
        ----------
        T : array, shape (n,)
            Current temperature field.
        T_left, T_right : scalar
            Dirichlet boundary values.
        length : scalar, optional
            Domain length for the uniform grid; ``None`` reads
            ``self.params["length"]``.  ``update`` passes the injected
            (traced) value so ``length`` is a differentiable parameter.

        Returns
        -------
        array, shape (n,)
            The Laplacian.
        """
        n = self.params["n_cells"]

        if self._is_nonuniform:
            # Non-uniform grid: always use 2nd-order variable-dx stencil
            x = self._grid_x
            # Ghost coordinates: reflect the first/last cell centre
            # through the end face.  The face sits midway between the
            # ghost and the first cell, at x[0] - (x[1] - x[0])/2, which
            # for the uniform grid is exactly x = 0.
            x_left = 2.0 * x[0] - x[1]
            x_right = 2.0 * x[-1] - x[-2]
            x_padded = jnp.concatenate([
                jnp.array([x_left], dtype=x.dtype),
                x,
                jnp.array([x_right], dtype=x.dtype),
            ])
            # Same Dirichlet closure as the uniform path: the datum is
            # the value at the end face, not at the first cell centre.
            T_padded = jnp.concatenate([
                jnp.array([_dirichlet_ghosts_2nd_order(T, T_left)],
                          dtype=T.dtype),
                T,
                jnp.array([_dirichlet_ghosts_2nd_order(T[::-1], T_right)],
                          dtype=T.dtype),
            ])
            return _laplacian_nonuniform(T_padded, x_padded)

        # Uniform grid
        L = self.params["length"] if length is None else length
        dx = L / n
        stencil_order = self.params.get("stencil_order", 2)

        if stencil_order == 4:
            # Two ghost cells per side, at x = -dx/2 and x = -3dx/2 (and
            # their mirrors), built by cubic extrapolation through the
            # rod end.  Before 0.4.0 the outer ghost held
            # ``2*T_left - T[1]`` and the inner one held ``T_left``
            # itself -- values for positions one cell further in than
            # the ones the stencil reads them at -- which is
            # MADD-ANO-008.
            ghost_left_2, ghost_left_1 = _dirichlet_ghosts_4th_order(T, T_left)
            ghost_right_2, ghost_right_1 = _dirichlet_ghosts_4th_order(
                T[::-1], T_right
            )

            T_padded = jnp.concatenate([
                jnp.array([ghost_left_2, ghost_left_1], dtype=T.dtype),
                T,
                jnp.array([ghost_right_1, ghost_right_2], dtype=T.dtype),
            ])  # shape (n+4,)
            return _laplacian_4th_order_uniform(T_padded, dx)
        else:
            # 2nd-order.  The ghost puts the Dirichlet datum on the rod
            # end; before 0.4.0 it held the datum itself, which placed
            # the boundary condition half a cell inside the rod
            # (MADD-ANO-007).
            T_padded = jnp.concatenate([
                jnp.array([_dirichlet_ghosts_2nd_order(T, T_left)],
                          dtype=T.dtype),
                T,
                jnp.array([_dirichlet_ghosts_2nd_order(T[::-1], T_right)],
                          dtype=T.dtype),
            ])  # shape (n+2,)
            return _laplacian_2nd_order_uniform(T_padded, dx)

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
        """Sharded update from a halo-padded temperature field.

        ``state_padded["temperature"]`` is expected to be the local
        shard's interior with ``halo_width()[0]`` ghost cells prepended
        and appended on axis 0 (filled by
        :class:`ShardedStencilNode`).  The Laplacian is computed on
        the interior; ghost cells are passed through unchanged so the
        wrapper can strip them.

        The boundary mode used by the wrapper drives Dirichlet/Neumann
        semantics at the *global* boundary:
        ``boundary="zero"`` ↔ Dirichlet T=0,
        ``boundary="edge"`` ↔ zero-gradient (Neumann).
        For non-zero Dirichlet temperatures or per-shard BC overrides,
        plug into the coupling system (M8) rather than this primitive.
        Non-uniform grids are not yet supported under sharding.

        Same params contract as :meth:`update`: ``thermal_diffusivity``
        and ``length`` come from the injected ``params`` when the
        (sharded) graph supplies them, so a ``ShardedStencilNode``
        wrapping a HeatNode is calibratable like the unsharded node.
        """
        if self._is_nonuniform:
            raise NotImplementedError(
                "HeatNode.update_padded does not yet support non-uniform "
                "grids under sharding.  Use the unsharded path."
            )

        T_pad = state_padded["temperature"]
        halo = int(self.halo_width()[0])
        if halo == 0:
            return self.update(state_padded, boundary_inputs, dt, params=params)

        p = self.params if params is None else {**self.params, **params}
        alpha = p["thermal_diffusivity"]
        L = p["length"]
        n_global = self.params["n_cells"]
        dx = L / n_global
        stencil_order = self.params.get("stencil_order", 2)

        if stencil_order == 4:
            assert halo == 2, "4th-order stencil expects halo=2"
            lap = _laplacian_4th_order_pure(T_pad, dx)
        else:
            assert halo == 1, "2nd-order stencil expects halo=1"
            lap = _laplacian_2nd_order_uniform(T_pad, dx)

        T_interior = T_pad[halo:-halo]
        n_local = T_interior.shape[0]

        source = boundary_inputs.get(
            "heat_source", jnp.zeros(n_local, dtype=T_pad.dtype)
        )
        source = jnp.asarray(source, dtype=T_pad.dtype)
        if source.ndim == 1 and source.shape[0] == n_local + 2 * halo:
            # a grid-shaped input arrives halo-padded from ShardedStencilNode
            source = source[halo:-halo]
        source = jnp.broadcast_to(source, (n_local,))

        T_new_interior = T_interior + alpha * dt * lap + source * dt

        return {"temperature": jnp.concatenate(
            [T_pad[:halo], T_new_interior, T_pad[-halo:]], axis=0
        )}

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Explicit finite-difference update for the 1D heat equation.

        Dirichlet BCs are enforced entirely through the ghost values the
        stencil reads, so ``left_temperature`` / ``right_temperature``
        are the temperatures at the rod ends ``x = 0`` and ``x = L`` --
        what :meth:`boundary_input_spec` documents.  Every cell,
        including the first and last, is then advanced by the scheme.

        Until 0.4.0 the end cells were instead *overwritten* with the
        Dirichlet data after the update.  Cell centres are at ``dx/2``
        and ``L - dx/2``, so that imposed the boundary value half a cell
        inside the rod and made the scheme globally 1st-order
        (MADD-ANO-007).  Callers who were relying on
        ``T_new[0] == left_temperature`` exactly should read the rod-end
        value they supplied, not the first cell.

        ``n_cells`` is structural and always comes from ``self.params``;
        ``thermal_diffusivity`` comes from the injected ``params`` when
        the graph supplies them.
        """
        n = self.params["n_cells"]
        p = self.params if params is None else {**self.params, **params}
        alpha = p["thermal_diffusivity"]
        length = p["length"]

        T = state["temperature"]  # shape (n,)

        # --- Boundary conditions ---
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])

        # --- Heat source ---
        source = boundary_inputs.get("heat_source", jnp.zeros(n, dtype=T.dtype))
        source = jnp.broadcast_to(jnp.asarray(source, dtype=T.dtype), (n,))

        # --- Laplacian ---
        laplacian = self._compute_laplacian(T, T_left, T_right, length)

        T_new = T + alpha * dt * laplacian + source * dt

        return {"temperature": T_new}

    def derivatives(self, state, boundary_inputs, *, params=None):
        """dT/dt = alpha * d^2T/dx^2 + source.

        ``thermal_diffusivity`` and ``length`` come from the injected
        ``params`` when the caller supplies them, exactly as in
        :meth:`update`; ``n_cells`` is structural and always read from
        ``self.params``.
        """
        n = self.params["n_cells"]
        p = self.params if params is None else {**self.params, **params}
        alpha = p["thermal_diffusivity"]
        length = p["length"]

        T = state["temperature"]
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])
        source = boundary_inputs.get(
            "heat_source", jnp.zeros(n, dtype=T.dtype)
        )
        source = jnp.broadcast_to(
            jnp.asarray(source, dtype=T.dtype), (n,)
        )

        laplacian = self._compute_laplacian(T, T_left, T_right, length)

        return {"temperature": alpha * laplacian + source}

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt, *, params=None):
        """Backward Euler residual: T_new - T_old - dt * f(T_new)."""
        # Forward ``params`` only when given, so a subclass whose
        # ``derivatives`` override predates the keyword still works for
        # every caller that passes none.
        derivs = (
            self.derivatives(state_new, boundary_inputs) if params is None
            else self.derivatives(state_new, boundary_inputs, params=params)
        )
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    def interface_dof_indices(self):
        return {
            "left_temperature": ("temperature", 0),
            "right_temperature": ("temperature", -1),
        }

    def compute_interface_correction(self, pre_state, boundary_inputs, dt, *, params=None):
        """Recompute boundary-cell temperatures from the FD stencil.

        Since 0.4.0 ``update()`` no longer overwrites T[0] and T[-1]
        with the Dirichlet data -- the data is imposed through the
        ghost cells instead (MADD-ANO-007) -- so the value this returns
        is the one ``update()`` already produced and applying it is an
        identity.  The override is kept because the coupling system
        asks every node that declares :meth:`interface_dof_indices` for
        a correction, and answering with the stencil value keeps that
        contract true regardless of how the BC is enforced internally.
        Same constants as ``update``.
        """
        p = self.params if params is None else {**self.params, **params}
        n = self.params["n_cells"]
        alpha = p["thermal_diffusivity"]

        T = pre_state["temperature"]
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])
        source = boundary_inputs.get(
            "heat_source", jnp.zeros(n, dtype=T.dtype)
        )
        source = jnp.broadcast_to(
            jnp.asarray(source, dtype=T.dtype), (n,)
        )

        laplacian = self._compute_laplacian(T, T_left, T_right)
        corrections: list[tuple[int, jnp.ndarray]] = []

        if "left_temperature" in boundary_inputs:
            T_new_0 = T[0] + alpha * dt * laplacian[0] + source[0] * dt
            corrections.append((0, T_new_0))

        if "right_temperature" in boundary_inputs:
            T_new_last = T[-1] + alpha * dt * laplacian[-1] + source[-1] * dt
            corrections.append((-1, T_new_last))

        if corrections:
            return {"temperature": corrections}
        return {}

    def boundary_input_spec(self):
        n = self.params["n_cells"]
        return {
            "left_temperature": BoundaryInputSpec(
                shape=(), description="Dirichlet BC at left end",
                expected_units="K",
            ),
            "right_temperature": BoundaryInputSpec(
                shape=(), description="Dirichlet BC at right end",
                expected_units="K",
            ),
            "heat_source": BoundaryInputSpec(
                shape=(n,), description="Volumetric heat source",
                coupling_type="additive",
                expected_units="K/s",
            ),
        }

    def boundary_flux_spec(self):
        """The two rod-end fluxes, in ``K*m/s``.

        The units are **not** ``W/m^2``, which this declared until
        0.4.0.  ``compute_boundary_fluxes`` returns ``-alpha dT/dx``
        with ``alpha`` in m^2/s and ``T`` in K, so the quantity is
        ``K*m/s``.  The conductive flux is ``-k dT/dx =
        -rho*c_p*alpha*dT/dx``; ``rho`` and ``c_p`` are not parameters
        of this node, so it cannot report W/m^2 and a consumer that
        needs them must multiply by ``rho*c_p`` itself (~4.2e6 for
        water).  Nothing in-tree converts using this field -- edge unit
        checks only warn -- but it is copied into descriptions and
        exports, so it should say what the number is.
        """
        return {
            "left_heat_flux": BoundaryFluxSpec(
                shape=(),
                description="Heat flux at the left rod end (x = 0)",
                output_units="K*m/s",
            ),
            "right_heat_flux": BoundaryFluxSpec(
                shape=(),
                description="Heat flux at the right rod end (x = L)",
                output_units="K*m/s",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        """Conductive flux ``-alpha dT/dx`` **at the two rod ends**.

        Both entries are the x-component of the flux, so on a rod
        heated from the left both come out positive.  The gradient is
        reconstructed at ``x = 0`` and ``x = L`` -- not at the first
        interior face -- by :func:`_rod_end_gradient`, from the
        Dirichlet datum in ``boundary_inputs`` when one is supplied
        plus the nearest ``stencil_order`` cells, and from the nearest
        three cells alone when it is not.  Before 0.4.0 this returned
        the gradient between the first two cell *centres*, i.e. the
        flux at ``x = dx``, a full cell inside the end it names; that
        was 1st order at the rod end whatever the stencil, and this is
        ``stencil_order`` (measured 2.00 / 3.93-4.13) with the datum
        and 2nd order without it.

        The units are ``K*m/s``: ``-alpha dT/dx`` is the conductive
        flux divided by ``rho * c_p``, neither of which is a parameter
        of this node.  See :meth:`boundary_flux_spec`.

        Parameters
        ----------
        state : dict
            Must contain ``"temperature"``, shape ``(n_cells,)``.
        boundary_inputs : dict
            ``left_temperature`` / ``right_temperature`` are read when
            present; each end falls back to extrapolation on its own.
        dt : float
            Unused -- the flux is a function of the state alone.
        params : dict, optional
            Overrides for the trainable constants, as in ``update``.

        Returns
        -------
        dict
            ``left_heat_flux`` and ``right_heat_flux``, both scalars.
        """
        # Same constants as ``update`` (see SpringDamperNode).
        p = self.params if params is None else {**self.params, **params}
        T = state["temperature"]
        n = self.params["n_cells"]
        alpha = p["thermal_diffusivity"]
        # The non-uniform path is 2nd order whatever ``stencil_order``
        # says, because ``_compute_laplacian`` falls back to the
        # variable-dx 3-point stencil there.
        order = 2 if self._is_nonuniform else self.params.get("stencil_order", 2)

        T_left = boundary_inputs.get("left_temperature")
        T_right = boundary_inputs.get("right_temperature")
        # Cells read, outwards from each end.  The datum anchors the
        # polynomial at the rod end, so ``order`` cells reach order
        # ``order``; without it three cells reach 2nd order.
        k_left = min(n, order if T_left is not None else 3)
        k_right = min(n, order if T_right is not None else 3)

        # Offsets are plain Python floats in both branches, so
        # ``_rod_end_gradient`` folds the whole Lagrange construction
        # before tracing and emits one dot product.  See
        # :func:`_lagrange_gradient_weights` for what writing it the
        # other way cost.
        if self._is_nonuniform:
            # ``self.params["grid_points"]`` and not ``self._grid_x``:
            # the latter is the float32 static-data copy the sharded
            # stencil reads, and a coordinate rounded to float32 puts a
            # ~2e-5 relative error on ``dx``, which floors the
            # reconstruction above its own 2nd-order error from n = 40
            # up.  ``grid_points`` is ``ParamSpec(trainable=False)`` --
            # geometry, not a fitted constant -- and is stored as
            # Python floats, so reading it is free and keeps the
            # reported flux at the order it advertises.  The stencil
            # itself still reads the float32 copy; that is the wider
            # dtype question TODO records under "AdaptiveNode dtype
            # policy".
            x = self.params["grid_points"]
            # The end faces sit midway between the first/last cell
            # centre and the ghost that mirrors it -- the same
            # placement ``_compute_laplacian`` uses.
            face_left = 1.5 * x[0] - 0.5 * x[1]
            face_right = 1.5 * x[-1] - 0.5 * x[-2]
            off_left = [x[j] - face_left for j in range(k_left)]
            off_right = [face_right - x[n - 1 - j] for j in range(k_right)]
            # Already in metres.
            scale_left = scale_right = None
        else:
            # Dimensionless, in units of ``dx``; the single division
            # below is where ``length`` -- which is trainable -- enters.
            off_left = [(2 * j + 1) / 2 for j in range(k_left)]
            off_right = [(2 * j + 1) / 2 for j in range(k_right)]
            scale_left = scale_right = p["length"] / n

        grad_left = _rod_end_gradient(
            [T[j] for j in range(k_left)], T_left, off_left, scale_left
        )
        # The right end measures inwards, so ``dT/dx = -dT/ds`` there
        # and the sign of the flux flips back to match the left end.
        grad_right = _rod_end_gradient(
            [T[n - 1 - j] for j in range(k_right)], T_right, off_right,
            scale_right,
        )
        return {
            "left_heat_flux": -alpha * grad_left,
            "right_heat_flux": alpha * grad_right,
        }
