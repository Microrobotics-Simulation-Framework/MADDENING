"""AdaptiveNode -- basis-agnostic frozen-active-set framework.

An *adaptive* solver represents its solution in a basis of ``n_max``
candidate functions but only ever solves on an **active set** -- a
data-dependent subset chosen by thresholding, top-K selection, residual
bulk-chasing or any other selection rule.  Hard selection has no useful
derivative, which appears to make adaptive solvers incompatible with
end-to-end autodiff.  The frozen-active-set pattern resolves this:

1. the selection step is committed as a non-differentiable forward-pass
   operation (``jax.lax.stop_gradient`` on a boolean mask), and
2. the adjoint flows through the *frozen-basis* linear solve via the
   implicit function theorem (:func:`maddening.core.solver_utils.ift_linear_solve`).

On each open region of parameter space where the active set is constant
the resulting gradient is the **exact** derivative of the frozen-set
objective; across the kinks where the active set changes it is a
one-sided (Clarke) subgradient.  See
``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`` (rounds 1-7) for the
measurements behind every constant in this module.

State contract
--------------

* ``c`` -- coefficient vector of shape ``(n_max,)``; zero outside the
  active set (the base class enforces this after every solve).
* ``mask`` -- boolean active set of shape ``(n_max,)``, the set the
  current ``c`` was solved on.  Fixed size: adaptivity changes which
  entries are ``True``, never the array shape, so ``update`` traces once
  and runs under ``jax.jit`` / ``jax.lax.scan`` unchanged.

Subclasses may add fields via :meth:`AdaptiveNode.extra_initial_state`.
The physical parameters the solve depends on (a source position, a
material constant) are **not** state: they live in the graph parameter
pytree (``params``) like every other node's constants, so ``jax.grad``
and system identification reach them.

Selection-Equivariance Theorem (spike round 6)
----------------------------------------------

Let ``G`` be a compact group acting on the basis index set and on the
parameter space, ``Fix(G)`` its fixed-point set, ``A(theta) c = b(theta)``
the discretised problem and ``J_frozen(theta; M) = s(A_M^{-1} b_M)`` the
objective on the active set ``M``.  If at ``theta_* in Fix(G)`` the
operator is ``G``-equivariant, the source is ``G``-symmetric, the
selection rule scores modes by a ``G``-invariant functional of
``(A, b)`` and ``M`` is locally constant, then
``grad J_frozen(theta_*) in T Fix(G)``: the frozen-set gradient has no
component transverse to the symmetric manifold.  This is Palais'
principle of symmetric criticality [Palais1979] applied to the
frozen-set objective, and it means **no selection rule can escape such a
trap on its own**.  The optimiser is "blind" to the escape direction.

The base class therefore provides a diagnostic and a mitigation:

* :meth:`AdaptiveNode.blindness_ratio` -- ``|grad J_frozen| / |grad J_full|``
  (``~1`` when the frozen adjoint is trustworthy, ``~0`` at a trap);
* :meth:`AdaptiveNode.is_trapped_at` -- a cheaper binary check;
* :meth:`AdaptiveNode.symmetry_break` -- an anisotropic perturbation of
  the parameters along the *full-basis* gradient, which (unlike
  isotropic noise) leaves the fixed-point set in one step.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import (
    NodeMeta, Reference, StabilityLevel,
)
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode


@stability(StabilityLevel.STABLE)
class AdaptiveNodeBlindnessError(RuntimeError):
    """The parameters sit at a Palais fixed point of the problem's symmetry.

    Raised by :meth:`AdaptiveNode.initial_state` (when the blindness
    gate is on) and by :meth:`AdaptiveNode.cold_start` when the
    blindness ratio stays below :attr:`AdaptiveNode.blindness_threshold`
    even after one :meth:`AdaptiveNode.symmetry_break` perturbation.
    Recover by perturbing the constructor's parameters (the source
    position, typically) and retrying; the frozen-set gradient at such a
    point is structurally blind to the direction an optimiser needs.
    """


@stability(StabilityLevel.STABLE)
class AdaptiveNode(SimulationNode):
    """Base class for adaptive solvers with a frozen-active-set adjoint.

    Subclasses implement the selection rule and the frozen-basis solve;
    the base class wires them into a JAX-traceable :meth:`update`, keeps
    the fixed-size ``(c, mask)`` state consistent, routes the adjoint
    through the frozen solve, and provides the blindness diagnostics.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Timestep carried by the :class:`SimulationNode` base.
    n_max : int
        Size of the padded coefficient buffer (the candidate basis).
        Every state array has shape ``(n_max,)``; the active set is a
        boolean mask over it.
    blindness_threshold, blindness_break_delta, D_threshold : optional
        Per-instance overrides of the class-level constants below.
    blindness_gate : bool, default True
        Run :meth:`blindness_ratio` in :meth:`initial_state` and raise
        :class:`AdaptiveNodeBlindnessError` when the constructor
        parameters are blind.  Costs two gradient evaluations (one of
        them full-basis) per ``initial_state`` call.  Turn off for
        subclasses that do not implement :meth:`objective`.
    dtype : optional
        Floating dtype of ``c``.  Default: JAX's canonical float
        (``float64`` under ``jax_enable_x64``, ``float32`` otherwise).
    **params
        Physical constants of the subclass, stored on ``self.params`` and
        exposed through :meth:`params_pytree` / :meth:`param_specs` like
        any other node.

    Attributes
    ----------
    blindness_threshold : float
        ``0.7``.  Below this ratio a state is considered blind (spike
        round 6: minimises expected cost of a false escape versus a
        missed trap over the 1-D and 2-D test problems).
    blindness_break_delta : float
        ``0.05``.  Magnitude of the :meth:`symmetry_break` perturbation
        (spike round 7: one perturbation of this size escaped every 1-D
        and 2-D trap tested with no drift back; the 1-D minimum was
        0.03).
    D_threshold : int
        ``5``.  Number of trainable parameters above which the cold-start
        check alone is insufficient and :meth:`is_trapped_at` should be
        run between optimiser steps as well (spike round 5).  The base
        class does not enforce a monitoring policy; subclasses and
        optimisation loops read this constant to decide theirs.

    Notes
    -----
    The three hooks a subclass provides:

    * :meth:`compute_active_set` -- the selection rule (non-differentiable).
    * :meth:`solve_frozen` -- the solve on a frozen mask (differentiable;
      typically one :func:`~maddening.core.solver_utils.ift_linear_solve`).
    * :meth:`objective` -- the scalar the blindness diagnostics
      differentiate.  Only needed for the diagnostics.
    """

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-009",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description=(
            "Basis-agnostic adaptive-solver base class with a "
            "frozen-active-set implicit-function-theorem adjoint and "
            "Palais-trap (blindness) diagnostics"
        ),
        governing_equations=(
            "A_M(theta) c_M = b_M(theta) on the active set M = "
            "compute_active_set(state, theta); c = 0 off M; "
            "dJ/dtheta = adjoint through the frozen solve (M held fixed)"
        ),
        discretization=(
            "Subclass-defined basis of n_max candidates, padded to fixed "
            "size with a boolean active-set mask; the selection step runs "
            "under stop_gradient, the frozen solve through ift_linear_solve"
        ),
        assumptions=(
            "The active set is locally constant in the parameters: the "
            "frozen-set gradient is exact on each such region and a "
            "one-sided (Clarke) subgradient at the kinks between them",
            "solve_frozen with an all-True mask is the full-basis solve "
            "(used by the default full-basis gradient)",
            "The objective used by the blindness diagnostics is a scalar "
            "function of the solved coefficients",
        ),
        limitations=(
            "Abstract: a subclass must supply compute_active_set, "
            "solve_frozen and (for the diagnostics) objective",
            "The frozen-set gradient is blind at Palais fixed points of "
            "the problem's symmetry group; the blindness diagnostics "
            "detect this at cold start but routine monitoring is the "
            "caller's policy (recommended above D_threshold parameters)",
            "blindness_ratio and symmetry_break cost a full-basis "
            "gradient (the expensive solve adaptivity exists to avoid); "
            "they are host-side diagnostics, not traceable",
            "The padded buffer costs n_max memory and FLOPs per step "
            "regardless of how many entries are active",
        ),
        references=(
            Reference("Palais1979", "Principle of symmetric criticality (trap mechanism)"),
            Reference("CohenDahmenDeVore2001", "Adaptive wavelet methods; active-set framing"),
            Reference("Blondel2022", "Implicit differentiation through a linear solve"),
        ),
        hazard_hints=(
            "A gradient step taken across an active-set change is a "
            "subgradient, not the derivative: optimisers relying on "
            "smoothness (line searches, quasi-Newton) can stall or "
            "oscillate at the kink",
            "At a Palais fixed point jax.grad returns a plausible-looking "
            "gradient that is exactly zero in the escape direction; "
            "nothing in the forward pass signals this",
            "No runtime validation that the masked operator handed to "
            "the frozen solve is non-singular on the active set",
        ),
        implementation_map={
            "Active-set selection (stop_gradient)": "maddening.nodes.adaptive.base.AdaptiveNode.update",
            "Frozen solve A_M c_M = b_M": "maddening.nodes.adaptive.base.AdaptiveNode.solve_frozen",
            "c = 0 off the active set": "maddening.nodes.adaptive.base.AdaptiveNode.update",
            "Adjoint through the frozen solve": "maddening.core.solver_utils.ift_linear_solve",
            "Blindness ratio": "maddening.nodes.adaptive.base.AdaptiveNode.blindness_ratio",
            "Symmetry break": "maddening.nodes.adaptive.base.AdaptiveNode.symmetry_break",
        },
    )

    # Spike-finalised constants (rounds 5-7); per-instance overrides via
    # the constructor.  Deliberately *not* on ``self.params``: they steer
    # host-side diagnostics, they are not physics a fit could identify.
    blindness_threshold: float = 0.7
    blindness_break_delta: float = 0.05
    D_threshold: int = 5

    def __init__(
        self,
        name: str,
        timestep: float,
        *,
        n_max: int,
        blindness_threshold: Optional[float] = None,
        blindness_break_delta: Optional[float] = None,
        D_threshold: Optional[int] = None,
        blindness_gate: bool = True,
        dtype: Any = None,
        **params: Any,
    ):
        if int(n_max) < 1:
            raise ValueError(f"n_max must be a positive integer, got {n_max!r}")
        super().__init__(name, timestep, n_max=int(n_max), **params)
        self.n_max = int(n_max)
        self.blindness_gate = bool(blindness_gate)
        self.dtype = jnp.zeros((), dtype=dtype).dtype
        if blindness_threshold is not None:
            self.blindness_threshold = float(blindness_threshold)
        if blindness_break_delta is not None:
            self.blindness_break_delta = float(blindness_break_delta)
        if D_threshold is not None:
            self.D_threshold = int(D_threshold)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def compute_active_set(
        self,
        state: dict,
        params: dict,
        *,
        prev: Optional[jax.Array] = None,
        is_cold_start: bool = False,
    ) -> jax.Array:
        """Select the active set for the next solve.

        Must be JAX-traceable with a fixed output shape: any top-K,
        threshold or hysteresis rule expressed with ``jnp`` operations
        (``jnp.sort``, comparisons, ``jnp.where``).  The base class wraps
        the result in ``jax.lax.stop_gradient``; implementations should
        still avoid differentiable surrogates (softmax scores, soft
        thresholds) on this path, because a leaked tangent here is the
        one silent failure the framework cannot detect.

        Parameters
        ----------
        state : dict
            Current state (``c`` and ``mask`` from the previous solve).
        params : dict
            Merged constants: ``self.params`` overlaid with the graph's
            injected pytree.  Read the physical parameters from here.
        prev : jax.Array or None
            The previous active mask, for rolling or hysteresis rules
            (add above ``eps_add``, remove below ``eps_remove < eps_add``).
            ``None`` at cold start.
        is_cold_start : bool
            ``True`` on the call from :meth:`initial_state`.

        Returns
        -------
        jax.Array
            Boolean mask of shape ``(n_max,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override compute_active_set"
        )

    def solve_frozen(self, state: dict, mask: jax.Array, params: dict) -> dict:
        """Solve on the frozen active set ``mask``.

        The differentiable half of the pattern.  Build the masked
        operator (identity on inactive rows keeps the buffer size fixed)
        and the masked right-hand side, call
        :func:`maddening.core.solver_utils.ift_linear_solve`, and return
        the coefficients.  ``jax.grad`` through this method is the
        frozen-set adjoint.

        Parameters
        ----------
        state : dict
            Current state.
        mask : jax.Array
            Boolean active set of shape ``(n_max,)``, already under
            ``stop_gradient``.
        params : dict
            Merged constants (see :meth:`compute_active_set`).

        Returns
        -------
        dict
            At least ``{"c": coefficients}``; extra keys are merged into
            the state.  The base class zeroes ``c`` off the mask and
            stores ``mask`` itself.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override solve_frozen"
        )

    def objective(self, state: dict, params: dict) -> jax.Array:
        """Scalar objective ``J`` the blindness diagnostics differentiate.

        Typically a sensor reading or integral of the solved field.
        Required by :meth:`blindness_ratio`, :meth:`is_trapped_at`,
        :meth:`symmetry_break` and the cold-start gate; :meth:`update`
        never calls it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override objective to use the "
            "blindness diagnostics (or construct with blindness_gate=False)"
        )

    def extra_initial_state(self) -> dict:
        """Additional state fields beyond ``c`` and ``mask``.  Default: none."""
        return {}

    def compute_full_basis_gradient(self, state: dict, params: Optional[dict] = None) -> dict:
        """``grad_theta J`` with every basis function active.

        Default: :meth:`solve_frozen` with an all-``True`` mask,
        differentiated through :meth:`objective` with respect to the
        trainable leaves of the parameter pytree.  Override when a
        cheaper full-basis solve exists.

        Returns
        -------
        dict
            Gradient pytree with the leaves of :meth:`params_pytree`;
            non-trainable leaves are zero.
        """
        full = jnp.ones(self.n_max, dtype=bool)
        return self._objective_gradient(state, params, full)

    # ------------------------------------------------------------------
    # SimulationNode contract
    # ------------------------------------------------------------------

    def state_fields(self) -> list[str]:
        return ["c", "mask", *self.extra_initial_state().keys()]

    def initial_state(self) -> dict:
        """Cold-start state at the constructor parameters.

        Selects the active set with ``is_cold_start=True``, solves on
        it, and -- when ``blindness_gate`` is on -- raises
        :class:`AdaptiveNodeBlindnessError` if :meth:`blindness_ratio`
        is below :attr:`blindness_threshold`.  Use :meth:`cold_start`
        for the variant that perturbs the parameters once before
        giving up.
        """
        state = self._cold_start_state(self.params)
        if self.blindness_gate:
            ratio = self.blindness_ratio(state)
            if ratio < self.blindness_threshold:
                raise AdaptiveNodeBlindnessError(
                    f"{type(self).__name__} {self.name!r}: blindness ratio "
                    f"{ratio:.3f} is below the threshold "
                    f"{self.blindness_threshold:.3f} at the constructor "
                    "parameters.  The frozen-set gradient is blind here "
                    "(a Palais fixed point of the problem's symmetry): "
                    "perturb the parameters, or use cold_start() / "
                    "symmetry_break() to move off the fixed-point set."
                )
        return state

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """One adaptation step: select, freeze, solve.

        ``params`` is the node's entry of the graph parameter pytree;
        ``None`` (a direct call) uses the constructor constants.  The
        mask returned by :meth:`compute_active_set` is committed under
        ``stop_gradient``; ``jax.grad`` of anything downstream reaches
        ``params`` only through :meth:`solve_frozen`.
        """
        del boundary_inputs, dt  # the base class uses neither
        p = self._merged(params)
        mask = self.compute_active_set(state, p, prev=state["mask"])
        mask = jax.lax.stop_gradient(jnp.asarray(mask, dtype=bool))
        if mask.shape != (self.n_max,):
            raise ValueError(
                f"{type(self).__name__}.compute_active_set returned shape "
                f"{mask.shape}; expected ({self.n_max},)"
            )
        return self._solve_and_pack(state, mask, p)

    # ------------------------------------------------------------------
    # Diagnostics and mitigation
    # ------------------------------------------------------------------

    def cold_start(self, params: Optional[dict] = None) -> tuple[dict, dict]:
        """Gated cold start with one automatic escape attempt.

        1. Build the cold-start state at ``params`` and measure
           :meth:`blindness_ratio`.
        2. If it is at least :attr:`blindness_threshold`, return.
        3. Otherwise apply one :meth:`symmetry_break` of
           :attr:`blindness_break_delta`, rebuild the state there and
           re-measure.
        4. Raise :class:`AdaptiveNodeBlindnessError` if still blind.

        Cost: at most two blindness ratios and one symmetry break.
        Spike round 4 found no cheap substitute for the full diagnostic
        at cold start; round 7 found one perturbation sufficient.

        Parameters
        ----------
        params : dict, optional
            Parameter pytree (``gm.params["nodes"][name]``); ``None``
            means the constructor constants.

        Returns
        -------
        (state, params)
            The cold-start state and the (possibly perturbed) parameter
            pytree that produced it.  Assign the latter back into
            ``gm.params`` so the graph runs at the escaped point.
        """
        pt = self._pytree(params)
        state = self._cold_start_state(self._merged(pt))
        if self.blindness_ratio(state, pt) >= self.blindness_threshold:
            return state, pt
        pt = self.symmetry_break(state, pt)
        state = self._cold_start_state(self._merged(pt))
        ratio = self.blindness_ratio(state, pt)
        if ratio >= self.blindness_threshold:
            return state, pt
        raise AdaptiveNodeBlindnessError(
            f"{type(self).__name__} {self.name!r}: blindness ratio "
            f"{ratio:.3f} is still below the threshold "
            f"{self.blindness_threshold:.3f} after one symmetry_break of "
            f"delta={self.blindness_break_delta}.  The parameters appear "
            "to be bound to a Palais fixed point; perturb them and retry."
        )

    def blindness_ratio(self, state: dict, params: Optional[dict] = None) -> float:
        """``|grad J_frozen| / |grad J_full|`` at ``state``.

        The gradient of :meth:`objective` with respect to the trainable
        parameters through the current frozen mask, over the same
        gradient with every basis function active.  ``~1`` means the
        frozen adjoint reproduces the full one; ``0`` is the signature
        of a Palais fixed point (the active set is structurally blind to
        the escape direction); values above ``1`` are over-amplified but
        direction-accurate.  Returns ``1.0`` as a sentinel when the
        full gradient itself is negligible (an interior extremum of
        ``J``), where the ratio is undefined.

        Cost: two gradient evaluations, one of them full-basis.  A
        host-side diagnostic (returns a Python float), not traceable.
        """
        g_frozen = self._objective_gradient(state, params, state["mask"])
        g_full = self.compute_full_basis_gradient(state, params)
        n_full = _tree_norm(g_full)
        scale = 1.0 + _tree_norm(self._pytree(params))
        if n_full < 1e-12 * scale:
            return 1.0
        return float(_tree_norm(g_frozen) / n_full)

    def is_trapped_at(
        self, state: dict, params: Optional[dict] = None, *, eps: float = 1e-3,
    ) -> bool:
        """Cheap binary Palais-trap check (re-thresholded finite difference).

        Perturbs the parameters by ``eps`` along the full-basis gradient,
        re-selects the active set there, and compares the frozen
        gradient at the two points.  At a trap both frozen gradients are
        (near) zero while their variation rate is not, so the proxy
        ``|g_0| / (|g_eps - g_0| / eps)`` collapses; the check fires
        below ``1e-2``.  Spike round 7: reliable for exact traps, not a
        continuous estimator of partial blindness -- use
        :meth:`blindness_ratio` for that.

        Returns
        -------
        bool
        """
        pt = self._pytree(params)
        g0 = self._objective_gradient(state, pt, state["mask"])
        direction = self._escape_direction(state, pt)
        pt_eps = jax.tree.map(lambda p, d: p + eps * d, pt, direction)
        mask_eps = self.compute_active_set(state, self._merged(pt_eps), prev=state["mask"])
        mask_eps = jnp.asarray(mask_eps, dtype=bool)
        g_eps = self._objective_gradient(state, pt_eps, mask_eps)
        rate = _tree_norm(jax.tree.map(lambda a, b: a - b, g_eps, g0)) / eps
        proxy = _tree_norm(g0) / (rate + 1e-30)
        return bool(proxy < 1e-2)

    def symmetry_break(
        self, state: dict, params: Optional[dict] = None, *, delta: Optional[float] = None,
    ) -> dict:
        """Perturb the trainable parameters transversely to the fixed-point set.

        ``theta <- theta + delta * g_full / |g_full|`` with ``g_full`` the
        full-basis gradient of :meth:`objective`.  By the
        Selection-Equivariance Theorem the frozen gradient at a trap
        lies in the symmetric manifold, while the full-basis gradient
        does not; moving along it is the anisotropic step that isotropic
        noise cannot provide.  Falls back to a uniform direction over
        the trainable leaves when ``g_full`` vanishes.

        Parameters
        ----------
        state : dict
        params : dict, optional
            Parameter pytree; ``None`` means the constructor constants.
        delta : float, optional
            Step size; default :attr:`blindness_break_delta`.  ``0.0``
            returns the parameters unchanged.

        Returns
        -------
        dict
            New parameter pytree (same leaves as :meth:`params_pytree`).
            Non-trainable leaves are untouched.
        """
        d = self.blindness_break_delta if delta is None else float(delta)
        pt = self._pytree(params)
        direction = self._escape_direction(state, pt)
        return jax.tree.map(lambda p, u: p + d * u, pt, direction)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merged(self, params: Optional[dict]) -> dict:
        """``self.params`` overlaid with an injected pytree."""
        return self.params if params is None else {**self.params, **params}

    def _pytree(self, params: Optional[dict]) -> dict:
        """Full parameter pytree: the defaults overlaid with ``params``."""
        base = self.params_pytree()
        if params is None:
            return base
        return {**base, **{k: v for k, v in params.items() if k in base}}

    def _trainable(self, key: str) -> bool:
        spec = self.param_specs().get(key)
        return spec is None or spec.trainable

    def _cold_start_state(self, merged: dict) -> dict:
        empty = {
            "c": jnp.zeros(self.n_max, dtype=self.dtype),
            "mask": jnp.zeros(self.n_max, dtype=bool),
            **self.extra_initial_state(),
        }
        mask = self.compute_active_set(empty, merged, prev=None, is_cold_start=True)
        mask = jax.lax.stop_gradient(jnp.asarray(mask, dtype=bool))
        if mask.shape != (self.n_max,):
            raise ValueError(
                f"{type(self).__name__}.compute_active_set returned shape "
                f"{mask.shape}; expected ({self.n_max},)"
            )
        return self._solve_and_pack(empty, mask, merged)

    def _solve_and_pack(self, state: dict, mask: jax.Array, merged: dict) -> dict:
        new = self.solve_frozen(state, mask, merged)
        c = jnp.asarray(new["c"])
        if c.shape != (self.n_max,):
            raise ValueError(
                f"{type(self).__name__}.solve_frozen returned c of shape "
                f"{c.shape}; expected ({self.n_max},)"
            )
        c = jnp.where(mask, c, jnp.zeros((), dtype=c.dtype))
        return {**state, **new, "c": c, "mask": mask}

    def _objective_gradient(
        self, state: dict, params: Optional[dict], mask: jax.Array,
    ) -> dict:
        """``grad_theta objective`` through ``solve_frozen`` on ``mask``.

        Differentiates with respect to the float leaves of the parameter
        pytree; non-trainable leaves get a zero gradient so the
        diagnostics and the escape direction respect :meth:`param_specs`.
        """
        pt = self._pytree(params)
        mask = jax.lax.stop_gradient(jnp.asarray(mask, dtype=bool))

        def J(tree):
            merged = {**self.params, **tree}
            out = self._solve_and_pack(state, mask, merged)
            return jnp.squeeze(jnp.asarray(self.objective(out, merged)))

        g = jax.grad(J)(pt)
        return {
            k: (v if self._trainable(k) else jnp.zeros_like(v))
            for k, v in g.items()
        }

    def _escape_direction(self, state: dict, pt: dict) -> dict:
        """Unit full-basis gradient over the trainable leaves."""
        g_full = self.compute_full_basis_gradient(state, pt)
        n_full = _tree_norm(g_full)
        if n_full > 1e-12:
            return jax.tree.map(lambda v: v / n_full, g_full)
        # Flat everywhere: fall back to a uniform unit direction over the
        # trainable leaves (the caller will typically raise afterwards).
        trainable = {k: v for k, v in pt.items() if self._trainable(k)}
        size = sum(int(jnp.size(v)) for v in trainable.values()) or 1
        return {
            k: (jnp.ones_like(v) / jnp.sqrt(size) if k in trainable else jnp.zeros_like(v))
            for k, v in pt.items()
        }


def _tree_norm(tree: Any) -> float:
    leaves = [jnp.asarray(v, dtype=float).ravel() for v in jax.tree.leaves(tree)]
    if not leaves:
        return 0.0
    return float(jnp.linalg.norm(jnp.concatenate(leaves)))
