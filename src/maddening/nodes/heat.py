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
"""

import jax.numpy as jnp

from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


def _laplacian_2nd_order_uniform(T_padded, dx):
    """2nd-order central difference Laplacian on a uniform grid."""
    return (T_padded[2:] - 2.0 * T_padded[1:-1] + T_padded[:-2]) / (dx * dx)


def _laplacian_4th_order_uniform(T_padded, dx):
    """4th-order central difference Laplacian on a uniform grid.

    Interior: (-T[i+2] + 16*T[i+1] - 30*T[i] + 16*T[i-1] - T[i-2]) / (12*dx^2)
    Boundary cells (i=0,1,n-2,n-1): fall back to 2nd-order.

    ``T_padded`` has shape ``(n+4,)`` with two ghost cells on each side.
    """
    n = T_padded.shape[0] - 4  # original cell count
    dx2 = dx * dx

    # 4th-order for all cells (using the 5-point stencil over T_padded)
    lap_4th = (
        -T_padded[:-4]
        + 16.0 * T_padded[1:-3]
        - 30.0 * T_padded[2:-2]
        + 16.0 * T_padded[3:-1]
        - T_padded[4:]
    ) / (12.0 * dx2)

    # 2nd-order for fallback (centred on the same cells)
    lap_2nd = (
        T_padded[3:-1] - 2.0 * T_padded[2:-2] + T_padded[1:-3]
    ) / dx2

    # Use 2nd-order for the first and last cells, 4th-order elsewhere
    # Build a mask: 1 for interior (4th-order), 0 for boundary (2nd-order)
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
        assumptions=(
            "Constant thermal diffusivity (no temperature dependence)",
            "1D geometry (rod)",
            "Dirichlet boundary conditions at both ends",
        ),
        limitations=(
            "CFL stability limit: dt < dx^2 / (2*alpha) -- violating this produces silently incorrect results (MADD-ANO-002)",
            "1st-order in time -- temporal accuracy is O(dt)",
            "No convection or radiation terms",
            "4th-order stencil falls back to 2nd-order at boundary cells",
        ),
        validated_regimes=(
            ValidatedRegime("thermal_diffusivity", 1e-6, 1.0, "m^2/s"),
            ValidatedRegime("n_cells", 4, 1000, notes="Convergence verified up to 1000 cells"),
            ValidatedRegime("CFL", 0.0, 0.5, notes="dt * alpha / dx^2 must be < 0.5 for stability"),
        ),
        hazard_hints=(
            "CFL stability not enforced at runtime -- unstable timesteps silently produce incorrect results (MADD-ANO-002)",
            "No runtime validation of thermal_diffusivity > 0",
        ),
        implementation_map={
            "alpha * d^2T/dx^2 (diffusion)": "maddening.nodes.heat.HeatNode.update",
            "S (source term)": "maddening.nodes.heat.HeatNode.update",
            "Time integration (dT/dt)": "maddening.nodes.heat.HeatNode.update",
            "Boundary conditions": "maddening.nodes.heat.HeatNode.update",
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
        geometry_source: str = None,
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
        """Return grid point coordinates as a JAX array."""
        gp = self.params.get("grid_points")
        if gp is not None:
            return jnp.array(gp, dtype=jnp.float32)
        n = self.params["n_cells"]
        L = self.params["length"]
        dx = L / n
        return jnp.linspace(dx / 2, L - dx / 2, n)

    def initial_state(self) -> dict:
        n = self.params["n_cells"]
        T0 = self.params["initial_temperature"]

        # Support both scalar (broadcast) and array initial conditions.
        T0_arr = jnp.asarray(T0, dtype=jnp.float32)
        temperature = jnp.broadcast_to(T0_arr, (n,)).copy()

        return {"temperature": temperature}

    def _compute_laplacian(self, T, T_left, T_right):
        """Compute the Laplacian d^2T/dx^2 using the configured stencil.

        Parameters
        ----------
        T : array, shape (n,)
            Current temperature field.
        T_left, T_right : scalar
            Dirichlet boundary values.

        Returns
        -------
        array, shape (n,)
            The Laplacian.
        """
        n = self.params["n_cells"]

        if self._is_nonuniform:
            # Non-uniform grid: always use 2nd-order variable-dx stencil
            x = self._grid_x
            # Ghost coordinates: extrapolate linearly
            x_left = 2.0 * x[0] - x[1]
            x_right = 2.0 * x[-1] - x[-2]
            x_padded = jnp.concatenate([
                jnp.array([x_left], dtype=jnp.float32),
                x,
                jnp.array([x_right], dtype=jnp.float32),
            ])
            T_padded = jnp.concatenate([
                jnp.array([T_left], dtype=jnp.float32),
                T,
                jnp.array([T_right], dtype=jnp.float32),
            ])
            return _laplacian_nonuniform(T_padded, x_padded)

        # Uniform grid
        L = self.params["length"]
        dx = L / n
        stencil_order = self.params.get("stencil_order", 2)

        if stencil_order == 4:
            # Need 2 ghost cells on each side for the 5-point stencil.
            # Left ghosts: reflect T through the Dirichlet BC
            # ghost[-2] = 2*T_left - T[1], ghost[-1] = T_left
            # (linear extrapolation from the BC)
            ghost_left_2 = 2.0 * T_left - T[1]
            ghost_left_1 = T_left
            ghost_right_1 = T_right
            ghost_right_2 = 2.0 * T_right - T[-2]

            T_padded = jnp.concatenate([
                jnp.array([ghost_left_2, ghost_left_1], dtype=jnp.float32),
                T,
                jnp.array([ghost_right_1, ghost_right_2], dtype=jnp.float32),
            ])  # shape (n+4,)
            return _laplacian_4th_order_uniform(T_padded, dx)
        else:
            # 2nd-order
            T_padded = jnp.concatenate([
                jnp.array([T_left], dtype=jnp.float32),
                T,
                jnp.array([T_right], dtype=jnp.float32),
            ])  # shape (n+2,)
            return _laplacian_2nd_order_uniform(T_padded, dx)

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Explicit finite-difference update for the 1D heat equation.

        Dirichlet BCs are enforced by setting the boundary ghost values
        before computing the stencil, and overwriting the boundary cells
        after the update.
        """
        n = self.params["n_cells"]
        alpha = self.params["thermal_diffusivity"]

        T = state["temperature"]  # shape (n,)

        # --- Boundary conditions ---
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])

        # --- Heat source ---
        source = boundary_inputs.get("heat_source", jnp.zeros(n, dtype=jnp.float32))
        source = jnp.broadcast_to(jnp.asarray(source, dtype=jnp.float32), (n,))

        # --- Laplacian ---
        laplacian = self._compute_laplacian(T, T_left, T_right)

        T_new = T + alpha * dt * laplacian + source * dt

        # Enforce Dirichlet BCs on the boundary cells.
        T_new = T_new.at[0].set(T_left)
        T_new = T_new.at[-1].set(T_right)

        return {"temperature": T_new}

    def derivatives(self, state, boundary_inputs):
        """dT/dt = alpha * d^2T/dx^2 + source."""
        n = self.params["n_cells"]
        alpha = self.params["thermal_diffusivity"]

        T = state["temperature"]
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])
        source = boundary_inputs.get(
            "heat_source", jnp.zeros(n, dtype=jnp.float32)
        )
        source = jnp.broadcast_to(
            jnp.asarray(source, dtype=jnp.float32), (n,)
        )

        laplacian = self._compute_laplacian(T, T_left, T_right)

        return {"temperature": alpha * laplacian + source}

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt):
        """Backward Euler residual: T_new - T_old - dt * f(T_new)."""
        derivs = self.derivatives(state_new, boundary_inputs)
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    def interface_dof_indices(self):
        return {
            "left_temperature": ("temperature", 0),
            "right_temperature": ("temperature", -1),
        }

    def compute_interface_correction(self, pre_state, boundary_inputs, dt):
        """Recompute boundary-cell temperatures from the FD stencil.

        HeatNode's ``update()`` enforces Dirichlet BCs by overwriting
        T[0] and T[-1] after the FD update.  When those BCs come from
        coupling, this overwrites the physically meaningful stencil
        value.  This method recomputes the stencil value so the
        coupling system can restore it.
        """
        n = self.params["n_cells"]
        alpha = self.params["thermal_diffusivity"]

        T = pre_state["temperature"]
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])
        source = boundary_inputs.get(
            "heat_source", jnp.zeros(n, dtype=jnp.float32)
        )
        source = jnp.broadcast_to(
            jnp.asarray(source, dtype=jnp.float32), (n,)
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
        return {
            "left_heat_flux": BoundaryFluxSpec(
                shape=(), description="Heat flux at left boundary",
                output_units="W/m^2",
            ),
            "right_heat_flux": BoundaryFluxSpec(
                shape=(), description="Heat flux at right boundary",
                output_units="W/m^2",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        T = state["temperature"]
        alpha = self.params["thermal_diffusivity"]
        if self._is_nonuniform:
            x = self._grid_x
            dx_left = x[1] - x[0]
            dx_right = x[-1] - x[-2]
        else:
            dx_left = self.params["length"] / self.params["n_cells"]
            dx_right = dx_left
        return {
            "left_heat_flux": -alpha * (T[1] - T[0]) / dx_left,
            "right_heat_flux": -alpha * (T[-1] - T[-2]) / dx_right,
        }
