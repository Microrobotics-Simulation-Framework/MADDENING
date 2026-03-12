"""
HeatNode -- 1D heat diffusion on a rod using explicit finite differences.

Models the 1D heat equation on a uniform grid::

    dT/dt = alpha * d^2T/dx^2 + source

with Dirichlet boundary conditions at both ends.  The entire ``update``
uses ``jnp`` operations (no Python loops), so it is fully JAX-traceable
and JIT-compilable.
"""

import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.core.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.stability import stability


@stability(StabilityLevel.STABLE)
class HeatNode(SimulationNode):
    """1D heat diffusion on a rod with Dirichlet boundary conditions.

    Solves the heat equation using explicit finite differences on a
    uniform grid of *n_cells* cells spanning a rod of the given
    *length*.

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
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="1D heat diffusion on a rod with Dirichlet BCs",
        governing_equations="∂T/∂t = α∇²T + S",
        discretization="Explicit finite difference, 2nd-order central in space, 1st-order forward Euler in time",
        assumptions=(
            "Uniform grid spacing",
            "Constant thermal diffusivity (no temperature dependence)",
            "1D geometry (rod)",
            "Dirichlet boundary conditions at both ends",
        ),
        limitations=(
            "CFL stability limit: dt < dx² / (2α) — violating this produces silently incorrect results (MADD-ANO-002)",
            "1st-order in time — temporal accuracy is O(dt)",
            "No convection or radiation terms",
        ),
        validated_regimes=(
            ValidatedRegime("thermal_diffusivity", 1e-6, 1.0, "m²/s"),
            ValidatedRegime("n_cells", 4, 1000, notes="Convergence verified up to 1000 cells"),
            ValidatedRegime("CFL", 0.0, 0.5, notes="dt * alpha / dx² must be < 0.5 for stability"),
        ),
        hazard_hints=(
            "CFL stability not enforced at runtime — unstable timesteps silently produce incorrect results (MADD-ANO-002)",
            "No runtime validation of thermal_diffusivity > 0",
        ),
        implementation_map={
            "α∇²T (diffusion)": "maddening.nodes.heat.HeatNode.update",
            "S (source term)": "maddening.nodes.heat.HeatNode.update",
            "Time integration (∂T/∂t)": "maddening.nodes.heat.HeatNode.update",
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
    ):
        super().__init__(
            name,
            timestep,
            n_cells=n_cells,
            length=length,
            thermal_diffusivity=thermal_diffusivity,
            initial_temperature=initial_temperature,
        )

    def initial_state(self) -> dict:
        n = self.params["n_cells"]
        T0 = self.params["initial_temperature"]

        # Support both scalar (broadcast) and array initial conditions.
        T0_arr = jnp.asarray(T0, dtype=jnp.float32)
        temperature = jnp.broadcast_to(T0_arr, (n,)).copy()

        return {"temperature": temperature}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Explicit finite-difference update for the 1D heat equation.

        The stencil is::

            T_new[i] = T[i] + alpha * dt / dx^2 * (T[i+1] - 2*T[i] + T[i-1])
                       + source[i] * dt

        Dirichlet BCs are enforced by setting the boundary ghost values
        before computing the stencil, and overwriting the boundary cells
        after the update.
        """
        n = self.params["n_cells"]
        L = self.params["length"]
        alpha = self.params["thermal_diffusivity"]
        dx = L / n

        T = state["temperature"]  # shape (n,)

        # --- Boundary conditions ---
        T_left = boundary_inputs.get("left_temperature", T[0])
        T_right = boundary_inputs.get("right_temperature", T[-1])

        # --- Heat source ---
        source = boundary_inputs.get("heat_source", jnp.zeros(n, dtype=jnp.float32))
        # Broadcast scalar source to all cells.
        source = jnp.broadcast_to(jnp.asarray(source, dtype=jnp.float32), (n,))

        # --- Finite-difference stencil ---
        # Pad T with the Dirichlet boundary values as ghost cells.
        T_padded = jnp.concatenate([
            jnp.array([T_left], dtype=jnp.float32),
            T,
            jnp.array([T_right], dtype=jnp.float32),
        ])  # shape (n+2,)

        # Second-order central difference: T[i+1] - 2*T[i] + T[i-1]
        laplacian = T_padded[2:] - 2.0 * T_padded[1:-1] + T_padded[:-2]  # shape (n,)

        coeff = alpha * dt / (dx * dx)
        T_new = T + coeff * laplacian + source * dt

        # Enforce Dirichlet BCs on the boundary cells.
        T_new = T_new.at[0].set(T_left)
        T_new = T_new.at[-1].set(T_right)

        return {"temperature": T_new}
