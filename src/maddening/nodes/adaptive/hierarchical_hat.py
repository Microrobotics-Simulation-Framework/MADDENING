"""HierarchicalHatAdaptiveNode — toy 2, local dyadic hat basis.

Same physical problem as
:class:`maddening.nodes.adaptive.topk.TopKAdaptiveNode`
(1D Poisson + Gaussian source on ``(0, 1)`` with Dirichlet BCs), but
the basis is now a hierarchical dyadic hat basis instead of sine
eigenfunctions.  This subclass exists for three reasons:

1. **Demonstrates the framework is basis-agnostic.**  The same
   ``AdaptiveNode`` base class handles a local basis (this toy) and a
   non-local one (the TopK toy) with no framework-level changes.
2. **Exercises the round-4 locality theorem.**  Each dyadic hat
   ``phi_λ(x_sensor)`` is single-signed on its support — either zero
   (sensor not in support) or strictly positive (sensor in support).
   The wrong-sign failure mode that ``TopKAdaptiveNode`` with
   ``selection_quantity='b'`` reproduces at boundary-θ is structurally
   impossible here.
3. **Exercises the BCOO sparse-operator path through
   ``ift_linear_solve``.**  The active-set restriction
   ``M_haar[mask, mask]`` is represented as a
   :class:`jax.experimental.sparse.BCOO` matrix and passed to the
   solver primitive.  This is the round-6 Investigation 3
   compatibility test made permanent.

Basis structure
---------------

The basis has ``N_levels + 1`` levels ``ℓ = 0, 1, …, N_levels``.
Level ``ℓ`` contains ``2^ℓ`` hat functions that tile ``[0, 1]``:

* Level 0: 1 hat with support ``[0, 1]``, peak at ``x = 0.5``.
* Level 1: 2 hats with supports ``[0, 0.5]`` and ``[0.5, 1]``.
* Level ``ℓ``: hats with width ``2^{-ℓ}``, centered at
  ``(2 p + 1) / 2^{ℓ+1}`` for ``p = 0, …, 2^ℓ - 1``.

Index ordering: index 0 is the level-0 hat, then indices ``2^ℓ`` to
``2^{ℓ+1} - 1`` are the level-``ℓ`` hats.  Total
``N_max = 2^{N_levels + 1} - 1``.

PDE projection
--------------

We discretise on a uniform fine grid of ``N_grid = 2 * N_max`` points,
form the FD Dirichlet operator ``A_phys = -d^2/dx^2 + I``, and Galerkin-
project to the hat basis: ``M = Phi^T A_phys Phi`` where ``Phi`` is
the (``N_grid``, ``N_max``) hat evaluation matrix.  The Galerkin
matrix ``M`` is symmetric positive-definite, so masked solves use
CG via :func:`ift_linear_solve`.

The sensor objective is ``J = u(x_sensor) = (Phi @ c)[sensor_idx]``,
matching the TopK convention.

Examples
--------

>>> from maddening.nodes.adaptive.hierarchical_hat \\
...     import HierarchicalHatAdaptiveNode
>>> node = HierarchicalHatAdaptiveNode(N_levels=5, theta_init=0.42)
>>> state = node.initial_state()
>>> bool(state['mask'][0])  # level-0 hat always active
True
"""

from __future__ import annotations

from typing import ClassVar, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.base import AdaptiveNode


def _level_of_index(N_levels: int):
    """Map flat hat index -> (level, position-within-level).

    Convention: index 0 is the single level-0 hat.  Indices
    ``2^ℓ`` … ``2^{ℓ+1} - 1`` are the ``2^ℓ`` hats at level ``ℓ``
    for ``ℓ ≥ 1``.
    """
    N_max = 2 ** (N_levels + 1) - 1
    levels = []
    positions = []
    for level in range(N_levels + 1):
        n_hats = 2 ** level if level > 0 else 1
        for p in range(n_hats):
            levels.append(level)
            positions.append(p)
    assert len(levels) == N_max
    return jnp.asarray(levels), jnp.asarray(positions)


def _build_hat_matrix(N_levels: int, x_grid: jnp.ndarray) -> jnp.ndarray:
    """Build the (N_grid, N_max) hat evaluation matrix."""
    levels, positions = _level_of_index(N_levels)
    N_max = int(levels.shape[0])
    cols = []
    for k in range(N_max):
        level = int(levels[k])
        p = int(positions[k])
        if level == 0:
            center = 0.5
            half_width = 0.5
        else:
            # 2^level hats tile [0, 1].  Hat p has support
            # [p / 2^level, (p+1) / 2^level] and centre at the midpoint.
            n_hats = 2 ** level
            a = p / n_hats
            b = (p + 1) / n_hats
            center = 0.5 * (a + b)
            half_width = 0.5 * (b - a)
        d = jnp.abs(x_grid - center)
        col = jnp.maximum(0.0, 1.0 - d / half_width)
        cols.append(col)
    return jnp.stack(cols, axis=1)


@stability(StabilityLevel.STABLE)
class HierarchicalHatAdaptiveNode(AdaptiveNode):
    """Dyadic hat basis on a Galerkin-projected 1D Poisson problem.

    Parameters
    ----------
    N_levels : int, default 7
        Number of refinement levels.  ``N_max = 2^{N_levels + 1} - 1``
        hat basis functions.  Default ``N_levels = 7`` gives
        ``N_max = 255``.
    K : int, default 16
        Active-set budget.  Cold-start always includes the level-0 root.
    sigma : float, default 0.04
        Gaussian source width.
    sensor_x : float, default 0.333
        Sensor location in domain units.
    theta_init : float, default 0.42
        Initial parameter.
    """

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-ADAPTIVE-HHAT",
        algorithm_version="0.4.0",
        stability=StabilityLevel.STABLE,
        description=(
            "1D Galerkin-projected Poisson on a hierarchical dyadic hat "
            "basis; demonstrates the locality theorem and BCOO + lineax "
            "sparse-operator path"
        ),
        governing_equations=(
            "(-d^2/dx^2 + 1) u(x) = exp(-((x - theta) / sigma)^2) on (0, 1) "
            "with Dirichlet BCs; Galerkin projection to dyadic hat basis"
        ),
        discretization=(
            "Dyadic hierarchical hat basis with N_levels + 1 levels; "
            "level-0 root always active; Galerkin matrix M = Phi^T A_phys Phi"
        ),
        assumptions=(
            "1D domain, Dirichlet boundary conditions",
            "FD Laplacian + I as the physical-space operator",
        ),
        limitations=(
            "Toy: not intended as a production solver; demonstrates the "
            "AdaptiveNode framework in a local basis",
            "BCOO conversion of the masked operator is O(N^2) at "
            "construction time (acceptable for the small toy sizes; "
            "WaveletAdaptiveNode will use a more efficient path)",
        ),
    )

    def __init__(
        self,
        *,
        name: str = "hat_adaptive",
        timestep: float = 1.0,
        N_levels: int = 7,
        K: int = 16,
        sigma: float = 0.04,
        sensor_x: float = 1.0 / 3.0,
        theta_init: float = 0.42,
        **kw,
    ):
        N_max = 2 ** (N_levels + 1) - 1
        super().__init__(name=name, timestep=timestep, N_max=N_max, **kw)
        self.N_levels = int(N_levels)
        self.N_max = int(N_max)
        self.K = int(K)
        self.sigma = float(sigma)
        self.sensor_x = float(sensor_x)
        self._theta_init = float(theta_init)

        # Fine grid for the Galerkin projection.
        n_grid = 2 * self.N_max
        x_grid = jnp.linspace(0.0, 1.0, n_grid, dtype=jnp.float64)
        dx = float(x_grid[1] - x_grid[0])
        self._x_grid = x_grid
        self._dx = dx

        # FD operator (-d^2/dx^2 + I) on the fine grid (Dirichlet).
        n = n_grid
        e = jnp.ones(n)
        eye = jnp.eye(n)
        # Build A_phys explicitly: diag(2/dx^2 + 1) - off-diag(1/dx^2)
        main = (2.0 / dx ** 2 + 1.0) * eye
        off = (1.0 / dx ** 2) * (
            jnp.diag(jnp.ones(n - 1), 1) + jnp.diag(jnp.ones(n - 1), -1)
        )
        A_phys = main - off
        self._A_phys = A_phys

        # Hat evaluation matrix Phi : (N_grid, N_max).
        self._phi = _build_hat_matrix(self.N_levels, x_grid)

        # Galerkin matrix M = Phi^T A_phys Phi : (N_max, N_max).
        self._M = self._phi.T @ A_phys @ self._phi

        # Index of the grid point closest to x_sensor.
        self._sensor_idx = int(round(self.sensor_x * (n_grid - 1)))

        # Levels of each index (for the always-include-level-0 rule).
        self._levels, _ = _level_of_index(self.N_levels)

    # ---- theta accessors ----
    def _get_theta(self, state):
        return state["theta"]

    def _set_theta(self, state, theta_new):
        return {**state, "theta": jnp.atleast_1d(theta_new)}

    # ---- RHS in the basis ----
    def _rhs_coeffs(self, theta) -> jax.Array:
        theta_s = jnp.squeeze(theta)
        f = jnp.exp(-((self._x_grid - theta_s) / self.sigma) ** 2)
        # Galerkin RHS: Phi^T @ f.  (No dx factor -- M is built without
        # the integration weight as well, so they cancel for the
        # ratio.)
        return self._phi.T @ f

    # ---- selection ----
    def compute_active_set(self, state, *, prev=None, is_cold_start=False):
        del prev
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        # Score by |b|; force the level-0 root hat to always be included.
        score = jnp.abs(b)
        # Boost the level-0 score so it always wins.
        boost = jnp.where(self._levels == 0,
                          jnp.full_like(score, jnp.inf), 0.0)
        boosted = score + boost
        threshold = jnp.sort(boosted)[-self.K]
        return jax.lax.stop_gradient(boosted >= threshold)

    # ---- inner solve via BCOO + ift_linear_solve ----
    def solve_frozen(self, state, mask):
        theta = self._get_theta(state)
        b = self._rhs_coeffs(theta)
        # Build the masked operator: M restricted to (mask, mask) on
        # active rows/cols, identity elsewhere.  Convert to BCOO so we
        # exercise the round-6 BCOO + lineax compatibility path.
        outer = mask[:, None] & mask[None, :]
        A_eff = jnp.where(outer, self._M,
                          jnp.eye(self.N_max, dtype=self._M.dtype))
        A_bcoo = jsparse.BCOO.fromdense(A_eff)
        b_eff = jnp.where(mask, b, 0.0)

        def operator_fn(v):
            return A_bcoo @ v

        c = ift_linear_solve(
            operator_fn, b_eff, solver="cg",
            rtol=1e-10, atol=1e-12,
        )
        return {**state, "c": c, "mask": mask}

    # ---- sensor functional ----
    def _sensor(self, state) -> jax.Array:
        c = state["c"]
        # u_grid = Phi @ c; sensor reads u_grid[sensor_idx]
        return self._phi[self._sensor_idx] @ c

    # ---- full-basis gradient ----
    def compute_full_basis_gradient(self, state) -> jax.Array:
        def J_full(theta):
            b = self._rhs_coeffs(theta)
            # Direct dense solve on the full N_max x N_max system.
            c = jnp.linalg.solve(self._M, b)
            return self._phi[self._sensor_idx] @ c

        return jax.grad(lambda t: jnp.squeeze(J_full(t)))(self._get_theta(state))

    # ---- cold-start state ----
    def _initial_state_impl(self) -> dict:
        theta = jnp.atleast_1d(jnp.asarray(self._theta_init, dtype=jnp.float64))
        empty = {
            "c": jnp.zeros(self.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(self.N_max, dtype=bool),
            "theta": theta,
        }
        mask = self.compute_active_set(empty, prev=None, is_cold_start=True)
        return self.solve_frozen({**empty, "mask": mask}, mask)
