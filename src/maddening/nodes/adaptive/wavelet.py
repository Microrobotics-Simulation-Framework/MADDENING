"""WaveletAdaptiveNode -- an adaptive interpolating-wavelet solver for ``(-Laplacian + m) u = f``.

The first concrete :class:`~maddening.nodes.adaptive.base.AdaptiveNode`
subclass with a real basis.  It solves the steady elliptic problem

    (-Laplacian + m) u(x) = f(x; theta, sigma)    on [0, 1)^d, d in {1, 2, 3},

periodic or with homogeneous Dirichlet walls, for an isotropic Gaussian
source of width ``sigma`` centred at ``theta`` on axis 0 (the other axes
centred), and reports the sensor reading ``J = u(x_s)``.  Both
``theta`` and ``sigma`` are leaves of the graph parameter pytree, so
``jax.grad``, ``sysid.fit`` and ``fim`` reach them through the frozen
active-set adjoint.

How the pieces map onto the base class
--------------------------------------

* **Basis.**  The Deslauriers-Dubuc interpolating wavelet basis of
  order 4 on the isotropic Mallat multiresolution
  (:mod:`~maddening.nodes.adaptive.wavelets.transform`), L2-normalised.
  The operator is assembled once, eagerly, as ``A = Wn^T A_phys Wn``
  with ``A_phys`` the second-order central-difference discretisation
  of ``(-Laplacian + m)`` -- an exact change of basis, so the **full**
  wavelet solve *is* the finite-difference solve and the scheme is
  second order in ``h``.  The operator is preconditioned symmetrically
  by a diagonal ``D`` (hybrid Jacobi by default).
* **Selection** (:meth:`compute_active_set`): Cohen-Dahmen-DeVore
  bulk chasing (:mod:`~maddening.nodes.adaptive.wavelets.cdd`) grown
  from the coarse level to a budget ``k``.  It is a function of the
  parameters alone, never of the previous state, so it never empties
  (level 0 is the seed), never exceeds ``k`` (the seed is validated to
  fit at construction and every marking step is capped at the room
  left), and does not chatter between steps at fixed parameters.
* **Frozen solve** (:meth:`solve_frozen`): the ``k`` active functions
  gathered into a dense ``k x k`` block and solved directly
  (``frozen_solver="gather"``, the default), or the masked full-size
  operator handed to :func:`~maddening.core.solver_utils.ift_linear_solve`
  with CG (``frozen_solver="cg"``).  Both are differentiable; the
  gathered solve is the one that realises the adaptivity speed-up.
* **Full-basis gradient** (:meth:`compute_full_basis_gradient`):
  overridden with a dense solve on ``A``, because the gathered frozen
  solve is only valid for ``|mask| <= k`` and the base class's
  all-true mask would silently truncate the set.

What is baked, and what flows
-----------------------------

``theta`` and ``sigma`` enter only through the right-hand side, which
:meth:`solve_frozen` recomputes from the injected ``params`` on every
call, so their gradients are exact within an active-set region.
``mass`` and ``sensor`` are baked into the operator and the sensor row
at construction; both are declared ``ParamSpec(trainable=False)`` and
published through :attr:`static_data` with :meth:`static_data_deps`, so
``compile()`` refuses a graph that tries to train either.

Every constant is built on the host from static settings (NumPy under
``jax.ensure_compile_time_eval``), so the node may be constructed
*inside* a ``jax.jit`` trace -- a ``residual_fn`` for
:func:`maddening.sysid.fim` that builds a fresh graph per call, say --
with ``blindness_gate=False``: the gate's diagnostics return host
floats and cannot run traced.

Examples
--------
>>> from maddening.nodes.adaptive import WaveletAdaptiveNode
>>> node = WaveletAdaptiveNode("wavelet", 1.0, n_levels=5, theta=0.42)
>>> state = node.initial_state()
>>> bool(state["mask"][0])            # the coarse level is always active
True
>>> int(state["mask"].sum()) <= node.params["k"]
True
>>> sorted(node.params_pytree())     # what a fit sees; mass/sensor are frozen
['mass', 'sensor', 'sigma', 'theta']
"""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import (
    DiscretizationOrder, NodeMeta, Reference, StabilityLevel,
)
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.base import AdaptiveNode, _positive_int
from maddening.nodes.adaptive.wavelets import cdd as _cdd
from maddening.nodes.adaptive.wavelets import dirichlet as _dir
from maddening.nodes.adaptive.wavelets import operator as _op
from maddening.nodes.adaptive.wavelets import precond as _pc
from maddening.nodes.adaptive.wavelets import transform as _tr

__all__ = ["WaveletAdaptiveNode"]

_DEFAULT_SENSOR: dict[int, tuple[float, ...]] = {
    1: (0.30,), 2: (0.30, 0.40), 3: (0.30, 0.40, 0.60),
}

#: Periodic image offsets summed by the default source on a periodic axis.
PERIODIC_IMAGES: tuple[int, ...] = (-2, -1, 0, 1, 2)

#: Largest ``kappa(A_hat) * eps(dtype)`` the constructor accepts, where
#: ``kappa(A_hat)`` is :func:`~maddening.nodes.adaptive.wavelets.operator.condition_estimate`
#: of the preconditioned operator.  A backward-stable solve returns a
#: relative error up to about ``kappa * eps``; measured in float32 against
#: the float64 solve of the same node (1-D 128 and 256 points, 2-D 8^2,
#: 3-D 4^3, order 6 on 48 points; mass 1 to 1e-5; full and default budget;
#: jaxlib 0.11.0), the sensor-reading error was 0.002 to 0.58 times
#: ``kappa * eps`` and never above it.  ``1e-3`` keeps the solver's share
#: an order of magnitude inside the ``1e-2`` truncation accuracy the node
#: documents at its default budget.  The periodic operator's smallest
#: eigenvalue is proportional to ``mass`` (the constant function), so
#: ``kappa`` grows like ``1 / mass``: in float32 this refuses a periodic
#: mass below about ``2e-3`` (1-D) to ``4e-3`` (3-D, order 6); in float64
#: below about ``1e-11``.  The Dirichlet operator is not affected.
CONDITION_LIMIT: float = 1e-3


# ---------------------------------------------------------------------------
# Jitted kernels.
#
# The AdaptiveNode base class calls the hooks *eagerly* -- the cold start,
# gradient_capture_ratio and frozen_gradient_vanishes_at all run outside
# any jit -- and an eager JAX function pays a separate XLA compile for
# every primitive it meets on first use (measured on a 192-point basis:
# 4.9 s for one eager gathered solve and 6.3 s for one eager frozen
# gradient, against 0.5 s and 0.8 s jitted).  The kernels are module-level
# pure functions with the node's constant arrays as *arguments* rather
# than per-instance closures, so the compile cache is shared by every
# instance of the same size and no n_max x n_max constant is baked into
# an executable.  Inside the graph's own jit they are inlined at no cost.
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("k",))
def _select_kernel(Ah: jax.Array, Ah_sparse: Any, coarse: jax.Array,
                   b_hat: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    """``(mask, outer_iterations)``; the node's hook returns the mask alone."""
    def solve(mask: jax.Array, rhs: jax.Array) -> jax.Array:
        return _op.gather_solve(Ah, mask, rhs, k)

    mask, _, n_outer = _cdd.cdd_select_with_iterations(
        lambda v: Ah_sparse @ v, solve, b_hat, coarse, k,
    )
    return mask, n_outer


@partial(jax.jit, static_argnames=("k",))
def _gather_kernel(Ah: jax.Array, mask: jax.Array, b_hat: jax.Array, *, k: int) -> jax.Array:
    return _op.gather_solve(Ah, mask, b_hat, k)


@partial(jax.jit, static_argnames=("rtol", "atol"))
def _cg_kernel(Ah_sparse: Any, mask: jax.Array, b_hat: jax.Array, *,
               rtol: float, atol: float) -> jax.Array:
    operator_fn = _op.make_masked_operator(Ah_sparse, mask)
    return ift_linear_solve(
        operator_fn, jnp.where(mask, b_hat, 0.0), solver="cg", rtol=rtol, atol=atol,
    )


_dense_solve = jax.jit(jnp.linalg.solve)


@stability(StabilityLevel.EXPERIMENTAL)
class WaveletAdaptiveNode(AdaptiveNode):
    """Adaptive interpolating-wavelet solver for ``(-Laplacian + m) u = f`` with a frozen-set adjoint.

    Parameters
    ----------
    name : str
        Node name.
    timestep : float
        Carried by :class:`~maddening.core.node.SimulationNode`; the
        node is a steady elliptic solve and never reads it.
    dim : {1, 2, 3}
        Spatial dimension.
    n_levels : int
        Refinement levels.  The grid has ``n_coarse * 2**n_levels``
        points per axis (periodic) or ``(n_coarse + 1) * 2**n_levels - 1``
        interior points (Dirichlet); ``n_max`` is that to the power ``dim``.
    n_coarse : int
        Coarse points per axis.
    order : {2, 4, 6}
        Interpolating order of the basis (4 by default).
    k : int, optional
        Active-set budget.  Must satisfy ``seed <= k <= n_max``, where
        ``seed`` is the CDD seed size: every level-0 function, i.e. the
        coarse block *plus the first detail band*, ``(2 n_coarse)**dim``
        periodic or ``(2 n_coarse + 1)**dim`` Dirichlet.  Default
        ``min(n_max, max(seed, 8, n_max // 16))``, which always satisfies
        the bound.  ``k == n_max`` turns adaptivity off (every function
        active).
    theta : float
        Source centre on axis 0.  Trainable, bounded to ``(0, 1)``.
    sigma : float
        Source width.  Trainable, positive.
    mass : float
        Coefficient ``m`` of the zeroth-order term.  Positive.  Baked
        into the operator: ``ParamSpec(trainable=False)`` and declared
        through :meth:`static_data_deps`.
    sensor : tuple of float, optional
        Sensor location, one coordinate per axis in ``[0, 1]``, snapped
        to the nearest grid point -- on the circle for a periodic axis,
        so ``1.0`` is grid point ``0``; among the interior points for a
        Dirichlet axis, so ``0.0`` and ``1.0`` snap to the points next to
        the walls.  Baked into the sensor row; ``trainable=False``.
    preconditioner : {"hybrid", "full", "level", "dk"}
        Diagonal scaling; see :mod:`~maddening.nodes.adaptive.wavelets.precond`.
    boundary : {"periodic", "dirichlet"}
    frozen_solver : {"gather", "cg"}
        How :meth:`solve_frozen` solves on the active set.
    **kw
        Forwarded to :class:`~maddening.nodes.adaptive.base.AdaptiveNode`
        (``blindness_gate``, ``on_blind``, ``dtype``, the diagnostic
        constants).

    Notes
    -----
    Every structural setting above is stored in ``self.params`` so that
    ``cls(name=..., timestep=..., **node.params)`` rebuilds the node --
    the round trip every serialisation path relies on.  ``n_max`` is
    derived and is not stored.

    The state is exactly ``{"c", "mask"}``: the node declares no extra
    state, so nothing can carry a stale value across an active-set
    change.
    """

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-010",
        algorithm_version="1.0.1",
        stability=StabilityLevel.EXPERIMENTAL,
        description=(
            "Adaptive interpolating-wavelet (Deslauriers-Dubuc) solver for "
            "(-Laplacian + m) u = f on the unit cube: Cohen-Dahmen-DeVore "
            "bulk-chasing active set, hybrid-Jacobi scaling, gathered dense "
            "frozen solve with the AdaptiveNode frozen-active-set adjoint"
        ),
        governing_equations=(
            "(-Laplacian + m) u(x) = f(x; theta, sigma) on [0, 1)^d, periodic or "
            "homogeneous Dirichlet; f = exp(-|x - x_theta|^2 / sigma^2) with "
            "x_theta = (theta, 1/2, ...); J = u(x_s).  Active set M by CDD "
            "residual bulk marking (Doerfler theta_D = 0.5) from the coarse "
            "level to the budget k"
        ),
        discretization=(
            "Second-order central-difference (-Laplacian + m) with lumped mass "
            "h^d, Galerkin-projected onto the L2-normalised DD-4 interpolating "
            "wavelet basis on the isotropic Mallat multiresolution: "
            "A = Wn^T A_phys Wn (an exact change of basis).  Symmetric diagonal "
            "preconditioning A_hat = D^-1 A D^-1.  Frozen solve on the gathered "
            "k x k block (dense) or the masked full operator (CG)"
        ),
        discretization_order=DiscretizationOrder(
            spatial=2.0,
            temporal=None,
            notes=(
                "The full-basis solve is the second-order central-difference "
                "solution in a different basis, so the order in h is the "
                "stencil's: 2, for both boundary types.  Measured by MMS on the "
                "full basis (k = n_max).  Adaptive truncation at k < n_max adds "
                "an error controlled by the budget, not by h; no order in k is "
                "claimed.  Steady problem: no temporal order."
            ),
        ),
        assumptions=(
            "Unit cube, uniform dyadic grid, constant coefficients: the "
            "operator is assembled once from (n_levels, n_coarse, order, dim, "
            "mass, boundary) and mass is therefore not trainable",
            "The source is an isotropic Gaussian centred on axis 0 unless "
            "source_field is overridden",
            "The active-set budget k is at least the CDD seed size -- every "
            "level-0 function, (2 n_coarse)**dim periodic or (2 n_coarse + 1)**dim "
            "Dirichlet, counted from the assembled basis -- and at most n_max "
            "(validated at construction; the default k is sized from it), so the "
            "seed fits and the gathered solve never truncates the set; an "
            "oversized mask is refused eagerly and poisoned with NaN under jit",
            "Inherits the AdaptiveNode assumptions: the returned gradient is "
            "exact within an active-set region and ignores the set's "
            "dependence on the parameters (MADD-ANO-003)",
        ),
        limitations=(
            "EXPERIMENTAL: validated on the unit cube up to 256 points (1-D), "
            "64^2 (2-D) and 16^3 (3-D); the operator is dense n_max x n_max "
            "and is a compile-time constant of the step",
            "Steady elliptic solve only: update ignores dt and boundary_inputs "
            "(the AdaptiveNode hooks do not receive them), so the node is an "
            "edge source and cannot time-step or consume an edge",
            "The adaptive sensor reading is within about 1e-2 of the full-basis "
            "one at the default budget k = n_max / 16; no convergence rate in "
            "k is claimed and the error is not monotone in k",
            "Dirichlet bases are built dense in NumPy at construction and the "
            "multi-D Dirichlet basis is a tensor product; sizes beyond the "
            "validated range need a matrix-free transform",
            "Each selection runs up to 30 CDD iterations, each with one "
            "gathered k x k solve; the selection cost is O(30 k^3 + 30 nnz), "
            "paid on every update.  For k above about n_max / 2 the iteration "
            "bound is reached before the budget (128-point basis: k = 64 and "
            "k = 96 both stop at |mask| = 54, sensor-reading error 3e-11); "
            "selection_diagnostics() reports outer_iterations and budget_reached",
        ),
        references=(
            Reference("DeslauriersDubuc1989", "Interpolating subdivision (the basis)"),
            Reference("CohenDahmenDeVore2001", "Adaptive wavelet methods; bulk chasing"),
            Reference("Doerfler1996", "Bulk marking criterion"),
            Reference("DahmenKunoth1992", "Diagonal (level) preconditioning"),
            Reference("Blondel2022", (
                "IFT adjoint of the masked-CG frozen solve (frozen_solver='cg', "
                "through ift_linear_solve); the default gathered solve is plain "
                "reverse-mode through jnp.linalg.solve"
            )),
        ),
        hazard_hints=(
            "A gradient step across an active-set change misses a jump in the "
            "objective (inherited from AdaptiveNode, MADD-ANO-003)",
            "The adaptive solution is a truncation of the full one: at a small "
            "budget the sensor reading can be off by more than the 1e-2 "
            "measured at k = n_max / 16, and the error is not monotone in k",
            "mass and sensor are baked into constants; changing either on a "
            "built node (rather than rebuilding it) leaves the operator and "
            "sensor row stale, and compile() refuses to train them",
            "The periodic operator is singular at mass = 0; the constructor "
            "refuses a non-positive mass",
        ),
        implementation_map={
            "Basis synthesis u = Wn c": "maddening.nodes.adaptive.wavelets.transform.synthesis",
            "Operator assembly A = Wn^T A_phys Wn": "maddening.nodes.adaptive.wavelets.operator.assemble_operator",
            "Diagonal preconditioning D": "maddening.nodes.adaptive.wavelets.precond.diagonal_scaling",
            "Source f(x; theta, sigma)": "maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.source_field",
            "Active set by CDD bulk chasing": "maddening.nodes.adaptive.wavelets.cdd.cdd_select",
            "Frozen solve (gathered block)": "maddening.nodes.adaptive.wavelets.operator.gather_solve",
            "Frozen solve (masked CG)": "maddening.core.solver_utils.ift_linear_solve",
            "Sensor functional J = u(x_s)": "maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.objective",
            "Full-basis gradient (dense solve)": "maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.compute_full_basis_gradient",
            "Selection diagnostics (iterations, budget reached)": "maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.selection_diagnostics",
        },
    )

    #: Accepted ``frozen_solver`` values.
    FROZEN_SOLVERS: ClassVar[tuple[str, ...]] = ("gather", "cg")

    def __init__(
        self,
        name: str = "wavelet",
        timestep: float = 1.0,
        *,
        dim: int = 1,
        n_levels: int = 6,
        n_coarse: int = 2,
        order: int = 4,
        k: Optional[int] = None,
        theta: float = 0.42,
        sigma: float = 0.10,
        mass: float = 1.0,
        sensor: Optional[tuple[float, ...]] = None,
        preconditioner: str = "hybrid",
        boundary: str = "periodic",
        frozen_solver: str = "gather",
        **kw: Any,
    ):
        # Structural counts are validated, never int()-truncated: n_levels=6.9
        # used to build a 64-point basis and dim=2.5 a 2-D node, silently.
        dim = _positive_int(dim, "dim")
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {dim!r}")
        n_levels = _positive_int(n_levels, "n_levels")
        n_coarse = _positive_int(n_coarse, "n_coarse")
        order = _positive_int(order, "order")
        if order not in _tr.DD_ORDERS:
            raise ValueError(f"order must be one of {_tr.DD_ORDERS}, got {order!r}")
        if boundary not in _op.BOUNDARIES:
            raise ValueError(f"boundary must be one of {_op.BOUNDARIES}, got {boundary!r}")
        if preconditioner not in _pc.PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {_pc.PRECONDITIONERS}, got {preconditioner!r}"
            )
        if frozen_solver not in self.FROZEN_SOLVERS:
            raise ValueError(
                f"frozen_solver must be one of {self.FROZEN_SOLVERS}, got {frozen_solver!r}"
            )
        mass = float(mass)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                f"mass must be a positive float (the periodic operator is singular "
                f"at mass = 0), got {mass!r}"
            )
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be a positive float, got {sigma!r}")

        if boundary == "dirichlet":
            side = _dir.dirichlet_side(n_levels, n_coarse)
            labels = _dir._level_labels_nd(n_levels, n_coarse, dim)
        else:
            side = _tr.side_length(n_levels, n_coarse)
            labels = _tr._level_labels_np(n_levels, n_coarse, dim)
        n_max = side ** dim
        # The CDD seed is every function labelled level 0 -- the coarse
        # block AND the first detail band, (2 n_coarse)**dim periodic or
        # (2 n_coarse + 1)**dim Dirichlet -- not the n_coarse**dim coarse
        # block alone.  Counted from the labels the seed itself is built
        # from (and re-checked against it after assembly), so the bound k
        # is validated against cannot drift from the seed the node uses.
        seed_size = int(np.sum(labels == labels.min()))
        if k is None:
            k = min(n_max, max(seed_size, 8, n_max // 16))
        k = _positive_int(k, "k")
        if not seed_size <= k <= n_max:
            band = "(2 n_coarse)" if boundary == "periodic" else "(2 n_coarse + 1)"
            raise ValueError(
                f"k must satisfy seed <= k <= n_max, got k={k!r} with seed={seed_size} "
                f"and n_max={n_max} (boundary={boundary!r}, dim={dim}, "
                f"n_coarse={n_coarse}): the CDD seed is every level-0 function -- "
                f"the coarse block plus the first detail band, {band}**dim = "
                f"{seed_size} -- and the gathered frozen solve holds exactly k "
                f"functions.  Pass k={seed_size} or larger, or leave k unset for "
                f"the default max(seed, 8, n_max // 16)"
            )

        if sensor is None:
            sensor = _DEFAULT_SENSOR[dim]
        sensor_t = tuple(float(s) for s in sensor)
        if len(sensor_t) != dim or not all(0.0 <= s <= 1.0 for s in sensor_t):
            raise ValueError(
                f"sensor must be {dim} coordinate(s) in [0, 1], got {sensor!r}"
            )

        super().__init__(
            name, timestep, n_max=n_max,
            dim=dim, n_levels=n_levels, n_coarse=n_coarse, order=order, k=k,
            theta=float(theta), sigma=sigma, mass=mass, sensor=sensor_t,
            preconditioner=str(preconditioner), boundary=str(boundary),
            frozen_solver=str(frozen_solver), **kw,
        )
        self.dim = dim
        self.side = int(side)
        self.k = k
        # The numeric parameters, which ``_rhs`` casts to the node's dtype.
        self._leaf_names = frozenset(self.params_pytree())

        # -- operator, preconditioner, coarse seed (structural constants) --
        op = _op.assemble_operator(
            n_levels, n_coarse, order=order, dim=dim, mass=mass,
            boundary=boundary, dtype=self.dtype, preconditioner=preconditioner,
        )
        if op.condition_number is None:          # asked for above: cannot happen
            raise RuntimeError("assemble_operator returned no condition estimate")
        #: :func:`~maddening.nodes.adaptive.wavelets.operator.condition_estimate`
        #: of the preconditioned operator every frozen solve is a block of.
        self.condition_number = float(op.condition_number)
        self._refuse_ill_conditioned(mass=mass, boundary=boundary)
        self._op = op
        self._A = op.A
        self._D = _pc.diagonal_scaling(
            op.diagonal, op.levels, preconditioner, dtype=self.dtype,
        )
        self._Ah = (op.A / self._D[:, None]) / self._D[None, :]
        rows, cols = op.A_sparse.indices[:, 0], op.A_sparse.indices[:, 1]
        self._Ah_sparse = jsparse.BCOO(
            (op.A_sparse.data / (self._D[rows] * self._D[cols]), op.A_sparse.indices),
            shape=op.A_sparse.shape,
        )
        lev = np.asarray(op.levels)
        coarse_np = lev == lev.min()
        if int(coarse_np.sum()) != seed_size:      # same labels: cannot happen
            raise RuntimeError(
                f"WaveletAdaptiveNode: the assembled seed has {int(coarse_np.sum())} "
                f"functions but k was validated against {seed_size}"
            )
        self._coarse = jnp.asarray(coarse_np)
        self._seed_size = seed_size
        self._h = float(op.h)

        # -- grid and sensor row --
        if boundary == "dirichlet":
            coords1d = np.arange(1, self.side + 1) / (self.side + 1)
        else:
            coords1d = np.arange(self.side) / self.side
        mesh = np.meshgrid(*([coords1d] * dim), indexing="ij")
        self._grid = tuple(jnp.asarray(m.reshape(-1), dtype=self.dtype) for m in mesh)
        sidx = 0
        for d in range(dim):
            dist = np.abs(coords1d - sensor_t[d])
            if boundary == "periodic":
                # distance on the circle: sensor 1.0 is grid point 0, not
                # the last point (side - 1) / side
                dist = np.minimum(dist, 1.0 - dist)
            sidx = sidx * self.side + int(np.argmin(dist))
        self._sensor_index = int(sidx)
        self._sensor_row = op.Wn[self._sensor_index]

    def _refuse_ill_conditioned(self, *, mass: float, boundary: str) -> None:
        """Refuse an operator whose conditioning the node's dtype cannot carry.

        See :data:`CONDITION_LIMIT`.  Every frozen solve is on a principal
        block of ``A_hat``, whose condition number is at most
        ``kappa(A_hat)`` (eigenvalue interlacing), so this bounds the
        adaptive solves, the masked-CG path and the full-basis gradient
        alike.  It used to be unchecked: in float32 on 128 points,
        ``mass=1e-6`` read ``J = 6.7e5`` against ``1.77e5`` and
        ``mass=1e-8`` read ``-2.9e16``, with no warning -- or with one
        blaming the active-set budget at ``k = n_max``.
        """
        eps = float(jnp.finfo(self.dtype).eps)
        kappa = self.condition_number
        if kappa * eps <= CONDITION_LIMIT:
            return
        eps64 = float(np.finfo(np.float64).eps)
        if self.dtype != jnp.float64 and kappa * eps64 <= CONDITION_LIMIT:
            dtype_hint = (
                f"Build the node in float64, which carries kappa up to "
                f"{CONDITION_LIMIT / eps64:.1e}: jax.config.update('jax_enable_x64', "
                f"True) and dtype=jnp.float64."
            )
        else:
            dtype_hint = "No floating dtype carries it."
        if boundary == "periodic":
            cause = (
                "The periodic operator's smallest eigenvalue is proportional to "
                "mass (its eigenvector is the constant function), so kappa grows "
                "like 1 / mass."
            )
            mass_hint = (
                f"  Or raise mass to at least about "
                f"{mass * kappa * eps / CONDITION_LIMIT:.1e} for {self.dtype}."
            )
        else:
            cause, mass_hint = "", ""
        raise ValueError(
            f"{type(self).__name__} {self.name!r}: the preconditioned operator's "
            f"condition number is about {kappa:.2e} (mass={mass!r}, "
            f"boundary={boundary!r}, n_max={self.n_max}), so a {self.dtype} solve "
            f"(eps {eps:.1e}) can be wrong by a relative {kappa * eps:.1e} -- above "
            f"the limit CONDITION_LIMIT = {CONDITION_LIMIT:.0e}.  {cause}  "
            f"{dtype_hint}{mass_hint}"
        )

    # ------------------------------------------------------------------
    # Parameters and static data
    # ------------------------------------------------------------------

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            **super().param_specs(),
            "theta": ParamSpec(bounds=(0.0, 1.0), transform="logit",
                               description="source centre on axis 0"),
            "sigma": ParamSpec(bounds=(0.0, None), transform="log",
                               description="source width"),
            "mass": ParamSpec(trainable=False, bounds=(0.0, None),
                              description="zeroth-order coefficient; baked into the operator"),
            "sensor": ParamSpec(trainable=False,
                                description="sensor location; baked into the sensor row"),
        }

    @property
    def static_data(self) -> dict:
        """The two constants ``update`` / ``objective`` read that derive from parameters.

        ``scaling`` is the diagonal preconditioner (a function of ``mass``
        through the operator's diagonal) and ``sensor_row`` the row of
        ``Wn`` at the sensor's grid point (a function of ``sensor``).
        Publishing them is what lets :meth:`static_data_deps` make
        ``compile()`` refuse a graph that trains either parameter.
        """
        from maddening.core.static_data import StaticArray
        return {
            "scaling": StaticArray(value=self._D, replication="replicate"),
            "sensor_row": StaticArray(value=self._sensor_row, replication="replicate"),
        }

    def static_data_deps(self) -> dict[str, tuple[str, ...]]:
        return {"scaling": ("mass",), "sensor_row": ("sensor",)}

    # ------------------------------------------------------------------
    # Problem pieces
    # ------------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """``(side,) * dim``: the shape :meth:`field` reshapes to."""
        return (self.side,) * self.dim

    def grid_coordinates(self) -> tuple[jax.Array, ...]:
        """One flattened coordinate array per axis (``x_i = i / side`` periodic,
        ``i / (side + 1)`` interior for Dirichlet)."""
        return self._grid

    def source_field(self, params: dict) -> jax.Array:
        """The forcing ``f`` sampled on the grid, flattened row-major.

        The default is an isotropic Gaussian of width ``params["sigma"]``
        centred at ``(params["theta"], 1/2, ...)``.  On a periodic domain
        it is **periodised** -- summed over its periodic images, which for
        the separable Gaussian is a product over axes of
        ``sum_n exp(-(d + n)**2 / sigma**2)`` with ``d`` the distance to
        the centre wrapped into ``[-1/2, 1/2)`` and ``n`` in
        :data:`PERIODIC_IMAGES` -- so the problem is translation-invariant
        on the circle, like the operator and the sensor snapping.  The
        truncated sum omits images at distance ``>= 2.5``, an error below
        ``2 exp(-6.25 / sigma**2)`` of the peak per axis (under ``1e-16``
        for ``sigma <= 0.41``).  It used to be the plain Gaussian on
        ``[0, 1)``: a source near the seam lost the part that should wrap
        round, and the same problem shifted across the seam read a 28%
        different ``J``.  With Dirichlet walls the plain Gaussian is used.

        This is the override point for a different forcing -- a
        manufactured-solution study subclasses the node and returns its
        source here -- and it must read every parameter it depends on
        from ``params``, never from ``self.params``, or the graph's
        injected values are ignored.

        Parameters
        ----------
        params : dict
            Merged parameters (``self.params`` overlaid with the graph's
            pytree), as the :class:`AdaptiveNode` hooks receive them.

        Returns
        -------
        jax.Array
            Shape ``(n_max,)``.
        """
        centre = [params["theta"]] + [0.5] * (self.dim - 1)
        periodic = self.params["boundary"] == "periodic"
        factors = []
        for axis in range(self.dim):
            d = self._grid[axis] - centre[axis]
            if periodic:
                d = d - jnp.round(d)
                g = sum(jnp.exp(-(d + n) ** 2 / params["sigma"] ** 2) for n in PERIODIC_IMAGES)
            else:
                g = jnp.exp(-d ** 2 / params["sigma"] ** 2)
            factors.append(g)
        f = factors[0]
        for g in factors[1:]:
            f = f * g
        return f

    def field(self, state: dict) -> jax.Array:
        """``u = Wn c`` on the grid, flattened row-major (``reshape(grid_shape)`` for an image)."""
        return self._op.Wn @ state["c"]

    def _rhs(self, params: dict) -> jax.Array:
        """Wavelet coefficients of the source, ``h^d Wn^T f``.

        Every parameter leaf is cast to the node's dtype *before*
        :meth:`source_field` sees it, and the source is cast again after.
        The first cast is what makes the same value in any spelling give
        the same bits: a Python float ``0.1`` squared on the host is not
        ``float32(0.1)`` squared, and the ~1e-7 difference in ``b`` used
        to be enough to change the selected set.  Under
        ``jax_enable_x64`` the graph injects float64 leaves whatever the
        node was built with, so without the casts a float32 node's
        ``grid - theta`` would promote and a float64 value would be
        scattered into the float32 coefficient buffer.  The casts are
        differentiable; the tangent comes back in the leaf's own dtype.
        """
        cast = {
            key: (jnp.asarray(value, dtype=self.dtype) if key in self._leaf_names else value)
            for key, value in params.items()
        }
        f = jnp.asarray(self.source_field(cast), dtype=self.dtype)
        return (self._h ** self.dim) * (self._op.Wn.T @ f)

    def _scaled_rhs(self, params: dict) -> jax.Array:
        return self._rhs(params) / self._D

    # ------------------------------------------------------------------
    # AdaptiveNode hooks
    # ------------------------------------------------------------------

    def compute_active_set(
        self, state: dict, params: dict, *,
        prev: Optional[jax.Array] = None, is_cold_start: bool = False,
    ) -> jax.Array:
        """CDD bulk chasing from the coarse level to the budget ``k``.

        A function of ``params`` alone: ``state``, ``prev`` and
        ``is_cold_start`` are accepted for the contract and ignored.  The
        coarse level is always in the set (so it is never empty, at a
        cold start or later), the set never exceeds ``k`` (the seed is
        validated to fit at construction and every marking step is
        capped at the room left; on the eager path the result is
        re-checked and an oversized set is refused), and at
        ``k == n_max`` every function is active.  The result is a boolean
        array; the base class commits it under ``stop_gradient``.
        """
        del state, prev, is_cold_start
        if self.k >= self.n_max:
            return jnp.ones(self.n_max, dtype=bool)
        # The selection must carry no tangent (the base class stops the
        # gradient on the mask too); stopping it on the input as well is
        # what lets CDD run as a ``while_loop`` under ``jax.grad``.
        b_hat = jax.lax.stop_gradient(self._scaled_rhs(params))
        mask, _ = _select_kernel(self._Ah, self._Ah_sparse, self._coarse, b_hat, k=self.k)
        self._refuse_oversized_mask(mask, "compute_active_set")
        return mask

    def solve_frozen(self, state: dict, mask: jax.Array, params: dict) -> dict:
        """Solve on the frozen ``mask``; the differentiable half.

        In the preconditioned coordinates ``A_hat c_hat = b_hat`` with
        ``b_hat = D^-1 h^d Wn^T f(params)``; the returned coefficients are
        ``c = D^-1 c_hat``, exactly zero off the mask.  Nothing here is
        singular on inactive entries (``D > 0`` everywhere), so no
        ``mask_safe`` sanitising is needed.
        """
        del state
        b_hat = self._scaled_rhs(params)
        if self.params["frozen_solver"] == "gather":
            self._refuse_oversized_mask(mask, "solve_frozen")
            c_hat = _gather_kernel(self._Ah, mask, b_hat, k=self.k)
        else:
            tight = bool(jnp.finfo(self.dtype).eps < 1e-10)
            c_hat = _cg_kernel(
                self._Ah_sparse, mask, b_hat,
                rtol=1e-10 if tight else 1e-6, atol=1e-12 if tight else 1e-8,
            )
        return {"c": c_hat / self._D}

    def objective(self, state: dict, params: dict) -> jax.Array:
        """The sensor reading ``J = u(x_s) = Wn[s] . c``."""
        del params
        return self._sensor_row @ state["c"]

    def _refuse_oversized_mask(self, mask: jax.Array, where: str) -> None:
        """Refuse a *concrete* mask with more than ``k`` entries.

        The gathered solve holds exactly ``k`` functions.  No mask the
        node produces is larger -- the seed is validated against ``k``
        at construction and the selection never grows past it -- so this
        is the host-side assertion of that proof on every eager path
        (the cold start, the diagnostics, a direct call), with a message.
        Under a trace the mask cannot be read and
        :func:`~maddening.nodes.adaptive.wavelets.operator.gather_solve`
        poisons an oversized block with ``NaN`` instead.
        """
        concrete = self._concrete_mask(mask)
        if concrete is None:
            return
        n_active = int(concrete.sum())
        if n_active > self.k:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}.{where}: the active set has "
                f"{n_active} functions but the gathered frozen solve holds exactly "
                f"k = {self.k}; the excess would be dropped, not solved.  A mask "
                f"the node selects itself never exceeds k (its seed of "
                f"{self._seed_size} is validated to fit); a mask supplied from "
                f"outside must fit too, or use frozen_solver='cg', which solves "
                f"on any mask"
            )

    def _merged(self, params: Optional[dict]) -> dict:
        """``self.params`` overlaid with ``params``, refusing a key it does not have.

        The base class's diagnostics refuse an unknown key through
        ``_pytree``, but ``update`` and :meth:`selection_diagnostics`
        overlay through here, where ``{"thetta": 0.9}`` used to be merged
        and then never read -- the call silently returned the answer at
        the constructor's ``theta``.  Keys are static under a trace, so
        the check costs nothing there.
        """
        if params is not None:
            unknown = sorted(set(params) - set(self.params))
            if unknown:
                raise ValueError(
                    f"{type(self).__name__} {self.name!r}: unknown parameter "
                    f"key(s) {unknown} -- not constructor parameters "
                    f"({sorted(self.params)}); the leaves a graph trains are "
                    f"{sorted(self.params_pytree())}."
                )
        return super()._merged(params)

    def selection_diagnostics(self, params: Optional[dict] = None) -> dict:
        """How the CDD selection at ``params`` ended: the budget, the rounding floor or the bound.

        A host-side diagnostic returning Python scalars (not traceable),
        like :meth:`gradient_capture_ratio`.  The selection is a function
        of the parameters alone, so no state is involved, and it is the
        same function the compiled graph runs: ties are broken by a fixed
        index order (:func:`~maddening.nodes.adaptive.wavelets.cdd.cdd_select_with_iterations`),
        so this eager evaluation selects the set a jitted ``update`` or a
        graph step selects at the same parameters.

        Parameters
        ----------
        params : dict, optional
            Parameter leaves to overlay on the constructor's; ``None``
            evaluates at the constructor constants.  An unknown key is
            refused.

        Returns
        -------
        dict
            ``active`` (``|mask|``), ``k``, ``outer_iterations`` (the CDD
            loop count), ``max_outer`` (its bound,
            :data:`~maddening.nodes.adaptive.wavelets.cdd.MAX_OUTER`),
            ``budget_reached`` (``active >= k``) and ``resolved``: ``True``
            when the loop stopped because no inactive function's residual
            was above the rounding floor
            (:func:`~maddening.nodes.adaptive.wavelets.cdd.rounding_floor`)
            -- nothing distinguishable from round-off was left to mark.
            ``budget_reached`` and ``resolved`` both ``False`` with
            ``outer_iterations == max_outer`` means the bound ended the
            loop -- the documented case for ``k`` above about
            ``n_max / 2`` in float64.  Every exit leaves a valid active
            set; the budget is a ceiling, not a target.
        """
        p = self._merged(params)
        if self.k >= self.n_max:
            return {"active": self.n_max, "k": self.k, "outer_iterations": 0,
                    "max_outer": _cdd.MAX_OUTER, "budget_reached": True,
                    "resolved": False}
        b_hat = jax.lax.stop_gradient(self._scaled_rhs(p))
        mask, n_outer = _select_kernel(
            self._Ah, self._Ah_sparse, self._coarse, b_hat, k=self.k,
        )
        active = int(np.asarray(mask, dtype=bool).sum())
        n_outer = int(n_outer)
        budget = active >= self.k
        return {"active": active, "k": self.k, "outer_iterations": n_outer,
                "max_outer": _cdd.MAX_OUTER, "budget_reached": budget,
                "resolved": (not budget) and n_outer < _cdd.MAX_OUTER}

    def compute_full_basis_gradient(self, state: dict, params: Optional[dict] = None) -> dict:
        """``grad J`` with every function active, by a dense solve on ``A_hat``.

        Overrides the base default, which would run :meth:`solve_frozen`
        with an all-true mask: the gathered frozen solve holds exactly
        ``k`` functions and would silently truncate the full set.  The
        solve is the preconditioned one, ``c = D^-1 A_hat^-1 D^-1 b``,
        like every frozen solve: the unscaled ``A`` is worse conditioned
        by the spread of its diagonal (about ``4**n_levels``), and in
        float32 that error alone moved the gradient-capture ratio at
        ``k = n_max`` -- where the frozen set *is* the full set -- off 1.
        Same return convention as the base -- the leaves of
        :meth:`params_pytree`, zero for non-trainable ones.
        """
        del state
        pt = self._pytree(params)

        def J(tree: dict) -> jax.Array:
            merged = {**self.params, **tree}
            c = _dense_solve(self._Ah, self._scaled_rhs(merged)) / self._D
            return jnp.squeeze(self._sensor_row @ c)

        g = jax.grad(J)(pt)
        return {
            key: (v if self._trainable(key) else jnp.zeros_like(v))
            for key, v in g.items()
        }
