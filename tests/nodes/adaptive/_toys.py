"""Concrete ``AdaptiveNode`` subclasses used by the adaptive-node tests.

Test fixtures, not public API.  Two toys:

``PoissonSineTopKNode``
    The spike's 1-D problem: ``(-d^2/dx^2 + 1) u = exp(-((x - theta)/sigma)^2)``
    on ``(0, 1)`` with Dirichlet ends, in the sine eigenbasis
    ``phi_k(x) = sqrt(2) sin(k pi x)`` where the operator is diagonal
    (``lambda_k = (k pi)^2 + 1``).  Active set: top-K modes by ``|b_k|``
    (the spike's selection, so its measured blindness numbers apply) or
    by ``|b_k / lambda_k|``.  Objective: the sensor reading
    ``u(x_s) = sum_k c_k phi_k(x_s)``.  Known points (spike rounds 3-7,
    top-|b|, K=16): theta=0.42 blindness ~0.86 (good), theta=0.48
    ~0.17 (partial), theta=0.5 exactly 0 (Palais trap: every odd mode
    has db_k/dtheta = 0 by the reflection symmetry about x=1/2 and the
    even modes have b_k = 0, so top-|b| selects only blind modes).

``MaskedDenseNode``
    A small dense SPD system ``(A0 + theta A1) c = b0 + theta b1`` with a
    non-diagonal operator, so the frozen solve exercises a real Krylov
    adjoint through ``ift_linear_solve`` and can be checked against
    ``jnp.linalg.solve`` on the active sub-block.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.params import ParamSpec
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import AdaptiveNode


class PoissonSineTopKNode(AdaptiveNode):
    """1-D Helmholtz--Poisson toy on the sine eigenbasis with top-K selection.

    Parameters
    ----------
    theta : float
        Gaussian source centre (trainable, in ``(0, 1)``).
    sigma : float
        Gaussian source width (declared non-trainable so the problem is
        one-dimensional in its trainable parameters, as in the spike).
    n : int
        Number of sine modes (``n_max``).
    k : int
        Active-set budget.
    selection : {"b", "c"}
        Score modes by ``|b_k|`` or by ``|b_k / lambda_k|``.
    sensor_x : float
        Sensor location for the objective.
    solver : {"dense", "cg", "gmres"}
        Backend handed to ``ift_linear_solve``.  ``"dense"`` by default:
        the operator is diagonal, and an eager Krylov solve recompiles
        its while-loop on every call, which dominated the suite's run
        time.  The gradient tests exercise ``"cg"`` explicitly.
    """

    def __init__(
        self,
        name: str = "adaptive",
        timestep: float = 1.0,
        *,
        theta: float = 0.42,
        sigma: float = 0.04,
        n: int = 256,
        k: int = 16,
        selection: str = "b",
        sensor_x: float = 1.0 / 3.0,
        solver: str = "dense",
        **kw,
    ):
        if selection not in ("b", "c"):
            raise ValueError(f"selection must be 'b' or 'c', got {selection!r}")
        # ``n`` is this subclass's own structural parameter: the base
        # class keeps ``n_max`` out of ``self.params`` (it would collide
        # with this call on a round-trip reconstruction), so a subclass
        # that wants its basis size to survive serialisation declares it.
        super().__init__(
            name, timestep, n_max=n, n=int(n), theta=theta, sigma=sigma,
            k=int(k), selection=selection, sensor_x=sensor_x, solver=solver,
            **kw,
        )
        dt = self.dtype
        n_grid = 2 * int(n)
        ks = jnp.arange(1, int(n) + 1, dtype=dt)
        x = jnp.linspace(0.0, 1.0, n_grid, dtype=dt)
        self._x = x
        self._dx = x[1] - x[0]
        self._phi = jnp.sqrt(2.0) * jnp.sin(jnp.pi * jnp.outer(x, ks))   # (n_grid, n)
        self._lambdas = (ks * jnp.pi) ** 2 + 1.0
        self._phi_sensor = jnp.sqrt(2.0) * jnp.sin(jnp.pi * ks * float(sensor_x))

    def param_specs(self):
        return {
            **super().param_specs(),
            "theta": ParamSpec(bounds=(0.0, 1.0), transform="logit",
                               description="Gaussian source centre"),
            "sigma": ParamSpec(trainable=False, bounds=(0.0, None),
                               description="Gaussian source width"),
            "sensor_x": ParamSpec(trainable=False, description="sensor location"),
        }

    # -- problem pieces ----------------------------------------------------

    def rhs(self, params: dict) -> jax.Array:
        """Sine-basis coefficients ``b_k`` of the Gaussian source."""
        f = jnp.exp(-((self._x - params["theta"]) / params["sigma"]) ** 2)
        return self._dx * (self._phi.T @ f)

    def full_solution_coefficients(self, params: dict) -> jax.Array:
        """Closed-form full-basis solution ``c_k = b_k / lambda_k``."""
        return self.rhs(params) / self._lambdas

    def field(self, c: jax.Array) -> jax.Array:
        """``u(x)`` on the internal grid from coefficients ``c``."""
        return self._phi @ c

    # -- AdaptiveNode hooks --------------------------------------------------

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        del prev, is_cold_start
        b = self.rhs(params)
        score = jnp.abs(b) if self.params["selection"] == "b" else jnp.abs(b) / self._lambdas
        threshold = jnp.sort(score)[-self.params["k"]]
        return score >= threshold

    def solve_frozen(self, state, mask, params):
        # The operator is diagonal, so ``c = b / lambda`` would do; route
        # through ift_linear_solve (identity on inactive rows keeps the
        # buffer size fixed) so the tests exercise the adjoint path the
        # framework prescribes.
        diag = jnp.where(mask, self._lambdas, 1.0)
        rhs = jnp.where(mask, self.rhs(params), 0.0)
        c = ift_linear_solve(
            lambda v: diag * v, rhs, solver=self.params["solver"],
            preconditioner=lambda v: v / diag, rtol=1e-12, atol=1e-14,
        )
        return {"c": c}

    def objective(self, state, params):
        return self._phi_sensor @ state["c"]


class MaskedDenseNode(AdaptiveNode):
    """Dense SPD toy: ``(A0 + theta A1) c = b0 + theta b1``, top-K by ``|b|``.

    ``solver`` selects the ``ift_linear_solve`` backend; the gradient tests
    run the Krylov ones, everything else uses ``"dense"`` (see
    :class:`PoissonSineTopKNode` for why).
    """

    def __init__(self, name: str = "dense", timestep: float = 1.0, *,
                 theta: float = 0.3, n: int = 24, k: int = 6, seed: int = 0,
                 solver: str = "dense", **kw):
        super().__init__(name, timestep, n_max=n, n=int(n), theta=theta, k=int(k),
                         seed=int(seed), solver=solver, **kw)
        rng = np.random.default_rng(seed)
        m0 = rng.standard_normal((n, n))
        m1 = rng.standard_normal((n, n))
        self.A0 = jnp.asarray(m0 @ m0.T + n * np.eye(n), dtype=self.dtype)
        self.A1 = jnp.asarray(m1 @ m1.T, dtype=self.dtype)
        self.b0 = jnp.asarray(rng.standard_normal(n), dtype=self.dtype)
        self.b1 = jnp.asarray(rng.standard_normal(n), dtype=self.dtype)
        self.s = jnp.asarray(rng.standard_normal(n), dtype=self.dtype)

    def param_specs(self):
        return {**super().param_specs(),
                "theta": ParamSpec(bounds=(0.0, None), transform="log")}

    def matrix(self, params):
        return self.A0 + params["theta"] * self.A1

    def rhs(self, params):
        return self.b0 + params["theta"] * self.b1

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        del prev, is_cold_start
        score = jnp.abs(self.rhs(params))
        return score >= jnp.sort(score)[-self.params["k"]]

    def solve_frozen(self, state, mask, params):
        A = self.matrix(params)
        b = jnp.where(mask, self.rhs(params), 0.0)

        def op(v):
            return jnp.where(mask, A @ jnp.where(mask, v, 0.0), v)

        return {"c": ift_linear_solve(op, b, solver=self.params["solver"],
                                      rtol=1e-12, atol=1e-14)}

    def objective(self, state, params):
        return self.s @ state["c"]
