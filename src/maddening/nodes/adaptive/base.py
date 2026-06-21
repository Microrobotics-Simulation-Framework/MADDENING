"""AdaptiveNode base class — basis-agnostic adaptive PDE framework.

This module provides the framework primitive specified by the
seven-round AdaptiveNode spike series
(``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md``).  The base class
itself is basis-agnostic: it carries no opinion about wavelet
families, selection criteria, or preconditioners.  Concrete subclasses
in :mod:`maddening.nodes.adaptive.topk` and
:mod:`maddening.nodes.adaptive.hierarchical_hat` demonstrate
non-local and local bases respectively.

Design summary
==============

State schema:

* ``c`` — coefficient vector of shape ``(N_max,)``.  Entries outside
  the active set are zero.
* ``mask`` — boolean active set of shape ``(N_max,)``.
* ``theta`` — the parameter vector the PDE depends on.  Shape is
  subclass-defined (scalar in 1D, ``(d,)`` in d-D parameter space).

Subclass hooks (abstract):

* :meth:`AdaptiveNode.compute_active_set` — given the current state,
  return the boolean mask for the next solve.
* :meth:`AdaptiveNode.solve_frozen` — given a state and mask, perform
  the inner solve and return the next state dict.
* :meth:`AdaptiveNode.compute_full_basis_gradient` — return
  ``∇_θ J_full(state)``, the full-basis gradient at this state.
  Used by the blindness diagnostic and ``symmetry_break``.
* :meth:`AdaptiveNode._initial_state_impl` — the subclass's "raw"
  initial state, BEFORE the base class applies the cold-start
  blindness gate.
* :meth:`AdaptiveNode._get_theta`,
  :meth:`AdaptiveNode._set_theta` — accessors over the theta field
  in state.  Subclasses define the convention.

Base-class machinery:

* :meth:`AdaptiveNode.initial_state` — runs ``_initial_state_impl``,
  checks blindness, applies one ``symmetry_break`` if blind, raises
  :exc:`AdaptiveNodeBlindnessError` on persistent trap.  (M4.)
* :meth:`AdaptiveNode.blindness_ratio` — full diagnostic.
* :meth:`AdaptiveNode.is_trapped_at` — cheap binary check.
* :meth:`AdaptiveNode.symmetry_break` — anisotropic perturbation in
  the unit ``g_full`` direction.

Configuration constants (overridable as subclass class-attrs or
constructor kwargs):

* ``blindness_threshold = 0.7`` — round-6 finalised default.  States
  with ``blindness_ratio < blindness_threshold`` trigger
  ``symmetry_break``.
* ``blindness_break_delta = 0.05`` — round-7 finalised default.  The
  perturbation magnitude.
* ``D_threshold = 5`` — round-5.  Dimensionality above which routine
  runtime monitoring (not just cold-start) becomes recommended; the
  base class does not enforce monitoring, but subclasses can read
  this constant to decide their own policy.

Selection-Equivariance Theorem
==============================

Round-6 of the spike series produced the following extension of
Palais 1979 to ``J_frozen``:

    Let ``G`` be a compact Lie group acting orthogonally on
    ``R^N`` via a permutation-of-indices representation ``ρ``, with
    fixed-point set ``Fix(G) ⊂ Θ``.  Let ``A(θ) ∈ R^{N×N}``,
    ``b(θ) ∈ R^N`` define the discretised PDE
    ``A(θ) c(θ) = b(θ)``, and let ``ℳ : Θ → 2^{1,…,N}`` be a
    selection map.  Define
    ``J_frozen(θ; M) = s^⊤ (A_M(θ)^{-1} b_M(θ))``
    where ``A_M``, ``b_M`` are the active-set restrictions and ``s``
    is a ``G``-invariant sensor functional.

    Assume at ``θ_* ∈ Fix(G)``:
    (1) ``ρ(g) A(θ_*) = A(θ_*) ρ(g)`` for all ``g ∈ G`` (operator
    equivariance);
    (2) ``b(θ_*) ∈ V_G`` (source symmetry);
    (3) ``ℳ`` scores modes by a ``G``-invariant functional of
    ``(A, b)``, so the active subspace ``V_{M_*}`` is ``G``-stable
    (selection equivariance);
    (4) ``θ ↦ (A, b, ℳ)`` is smooth in a ``G``-invariant neighbourhood
    of ``θ_*`` with ``ℳ`` locally constant.

    *Then* ``∇_θ J_frozen(θ_*; M_*) ∈ T_{θ_*} Fix(G)``; the
    transverse component vanishes.

The consequence: no selection criterion that scores modes by a
``G``-invariant functional of ``(A, b)`` can escape the trap.
Trap mitigation must be at the optimizer level — an **anisotropic**
perturbation transverse to ``Fix(G)`` (Chen-Ziyin 2023: isotropic
SGD noise cannot escape Type-II saddles).  The
:meth:`AdaptiveNode.symmetry_break` method implements this by
perturbing ``θ`` in the direction of the **full-basis** gradient,
which by Palais lies in ``T_θ Fix(G)`` at ``θ ∈ Fix(G)`` — the
perturbation moves transversely.

Reference: ``plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md``,
round-6 subagent + round-7 Investigation 2.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode


@stability(StabilityLevel.STABLE)
class AdaptiveNodeBlindnessError(RuntimeError):
    """Raised by :meth:`AdaptiveNode.initial_state` on persistent trap.

    Indicates the cold-start state is at a Palais fixed point of the
    problem's symmetry group, and one ``symmetry_break`` perturbation
    of ``blindness_break_delta`` did not move it into a region with
    ``blindness_ratio >= blindness_threshold``.

    Recover by perturbing the constructor's ``theta_init`` (or
    equivalent) manually and retrying.  This is a structural
    condition: see the Selection-Equivariance Theorem in
    :mod:`maddening.nodes.adaptive.base`.
    """


@stability(StabilityLevel.STABLE)
class AdaptiveNode(SimulationNode):
    """Basis-agnostic adaptive PDE framework.

    Subclasses must implement:

    * :meth:`compute_active_set` — return the boolean mask of shape
      ``(N_max,)``.
    * :meth:`solve_frozen` — run the inner solve and return the next
      state dict.  Typically calls
      :func:`maddening.core.solver_utils.ift_linear_solve`.
    * :meth:`compute_full_basis_gradient` — return ``∇_θ J_full(state)``.
    * :meth:`_initial_state_impl` — the subclass's pre-gate initial state.
    * :meth:`_get_theta`, :meth:`_set_theta` — accessors over the
      theta field in state.

    Parameters
    ----------
    name : str
        Node name.
    timestep : float
        ``delta_t`` carried by the :class:`SimulationNode` base.
    N_max : int
        Maximum number of basis functions in the padded buffer.  All
        state arrays are shape ``(N_max,)``; the active mask has at
        most ``N_max`` True entries.
    blindness_threshold : float, optional
        Override the class-level default of ``0.7``.
    blindness_break_delta : float, optional
        Override the class-level default of ``0.05``.
    D_threshold : int, optional
        Override the class-level default of ``5``.
    """

    # ---- finalised configuration constants ----
    blindness_threshold: ClassVar[float] = 0.7
    blindness_break_delta: ClassVar[float] = 0.05
    D_threshold: ClassVar[int] = 5

    # ---- construction ----
    def __init__(
        self,
        *,
        name: str = "adaptive",
        timestep: float = 1.0,
        N_max: int,
        blindness_threshold: Optional[float] = None,
        blindness_break_delta: Optional[float] = None,
        D_threshold: Optional[int] = None,
        **params: Any,
    ):
        super().__init__(name=name, timestep=timestep, **params)
        self.N_max = int(N_max)
        if blindness_threshold is not None:
            self.blindness_threshold = float(blindness_threshold)
        if blindness_break_delta is not None:
            self.blindness_break_delta = float(blindness_break_delta)
        if D_threshold is not None:
            self.D_threshold = int(D_threshold)

    # ---- subclass hooks (abstract) ----
    def compute_active_set(
        self,
        state: dict,
        *,
        prev: Optional[dict] = None,
        is_cold_start: bool = False,
    ) -> jax.Array:
        """Return the boolean active mask for the next solve.

        Shape: ``(N_max,)``, dtype ``bool``.  Implementations should
        wrap any non-differentiable selection (top-k, threshold, etc.)
        in :func:`jax.lax.stop_gradient` to make the
        non-differentiability explicit (round-1 Q4: ``stop_gradient``
        is documentation when the underlying operation is naturally
        non-differentiable).

        Parameters
        ----------
        state : dict
            Current state.  May be the cold-start state or a state
            carried from a previous step.
        prev : dict or None, optional
            The previous step's state, if available.  Subclasses may
            use it for rolling selection (e.g., top-|c_prev|).
            ``None`` at cold start.
        is_cold_start : bool, default False
            ``True`` only on the very first call from
            :meth:`initial_state`.  Subclasses can branch on this to
            apply a different protocol (e.g., coarse-then-fine).

        Returns
        -------
        jax.Array
            Boolean array of shape ``(N_max,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override compute_active_set"
        )

    def solve_frozen(self, state: dict, mask: jax.Array) -> dict:
        """Inner solve with the active set ``mask`` frozen.

        Typical implementation: build the masked operator, call
        :func:`maddening.core.solver_utils.ift_linear_solve` with the
        operator and the masked right-hand side, and assemble the
        returned coefficient vector into the state dict.

        Returns
        -------
        dict
            Next-state dict containing at least ``c`` and ``mask``
            (and ``theta`` carried forward unchanged for a pure inner
            solve).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override solve_frozen"
        )

    def compute_full_basis_gradient(self, state: dict) -> jax.Array:
        """Return ``∇_θ J_full(state)``: the gradient of the sensor
        objective w.r.t. ``θ`` under the **full** basis (no mask).

        Used by :meth:`blindness_ratio` and :meth:`symmetry_break`.
        Cost: typically one full-basis forward solve + one
        adjoint solve.  This is "the expensive solve the adaptive
        framework exists to avoid" — invoked sparingly, at cold
        start and at runtime restart points only.

        Returns
        -------
        jax.Array
            Same shape as ``theta``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override compute_full_basis_gradient"
        )

    def _initial_state_impl(self) -> dict:
        """Subclass-provided initial state, BEFORE the cold-start
        blindness gate.

        Returns
        -------
        dict
            State dict containing at least ``c``, ``mask``, ``theta``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _initial_state_impl"
        )

    def _get_theta(self, state: dict) -> jax.Array:
        """Extract ``θ`` from the state dict."""
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_theta"
        )

    def _set_theta(self, state: dict, theta_new: jax.Array) -> dict:
        """Return a new state dict with ``θ`` replaced by ``theta_new``."""
        raise NotImplementedError(
            f"{type(self).__name__} must override _set_theta"
        )

    # ---- state schema ----
    def state_fields(self) -> list[str]:
        """Return the state field names.

        The base class declares ``c``, ``mask``, ``theta`` as the
        canonical fields.  Subclasses adding extra state should
        override this method to extend the list.
        """
        return ["c", "mask", "theta"]

    # ---- default update() ----
    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Default update: compute active set, then solve frozen.

        Subclasses with more elaborate flows (e.g., CDD outer loop)
        can override this method.  The default flow is one
        ``compute_active_set`` call followed by one ``solve_frozen``.
        """
        del boundary_inputs, dt  # base class makes no use of either
        mask = self.compute_active_set(
            state, prev=state, is_cold_start=False,
        )
        return self.solve_frozen(state, mask)

    # ---- initial_state ----
    # M2 stub: returns the subclass's raw initial state.  The
    # cold-start blindness gate is added in M4.
    def initial_state(self) -> dict:
        """Return the initial state.

        M2 placeholder: returns ``_initial_state_impl()`` directly.
        The cold-start blindness gate (full diagnostic + one
        ``symmetry_break`` perturbation if blind, raising
        :exc:`AdaptiveNodeBlindnessError` on persistent trap) lands in
        M4.
        """
        return self._initial_state_impl()

    # ---- diagnostic / mitigation surface (M3) ----

    def _compute_frozen_gradient(self, state: dict) -> jax.Array:
        """Compute ``∇_θ J_frozen`` at the current state.

        Default implementation: build a closure
        ``J(theta) = sensor(solve_frozen(set_theta(state, theta)))``
        and call ``jax.grad`` on it.  Subclasses with cheaper
        path (e.g., a precomputed adjoint) can override.

        Returns
        -------
        jax.Array
            Same shape as ``theta``.
        """
        # Default frozen-gradient: jax.grad through one frozen solve at
        # the CURRENT active mask.  Subclasses may override if they have
        # a cheaper closed-form path.
        mask = state["mask"]

        def frozen_J(theta: jax.Array) -> jax.Array:
            s = self._set_theta(state, theta)
            out = self.solve_frozen(s, mask)
            return self._sensor(out)

        return jax.grad(frozen_J)(self._get_theta(state))

    def _sensor(self, state: dict) -> jax.Array:
        """Scalar sensor functional of the state.  Subclasses override.

        Default: returns the first entry of ``c`` -- a placeholder so
        the framework is testable without forcing a non-trivial
        subclass.  Concrete subclasses (M5, M6) override.
        """
        return state["c"][0]

    def blindness_ratio(self, state: dict) -> float:
        """``|∇_θ J_frozen(state)| / |∇_θ J_full(state)|``.

        Cost: two gradient evaluations (one frozen, one full).  The
        full-basis evaluation is "the expensive solve adaptivity
        exists to avoid"; this diagnostic is invoked at cold start
        and at runtime restarts, not per-update.

        Returns
        -------
        float
            ``0.0`` at exact Palais fixed points where the active set
            is structurally blind to the gradient direction.  ``1.0``
            (or near) when the frozen-set adjoint matches the
            full-basis gradient.  Values above ``1.0`` indicate the
            frozen-set gradient is over-amplified (still direction-
            accurate in 1D; round-7 Inv 3 confirms generally).

            When the full-basis gradient itself is below
            ``1e-12 * ||theta||`` (an interior extremum of J or a
            zero-source problem), returns ``1.0`` as a sentinel to
            avoid the 0 / 0 ambiguity.
        """
        g_frozen = self._compute_frozen_gradient(state)
        g_full = self.compute_full_basis_gradient(state)
        n_frozen = jnp.linalg.norm(g_frozen)
        n_full = jnp.linalg.norm(g_full)
        # Sentinel: at an interior extremum of J both numerator and
        # denominator are near-zero; the ratio is undefined.  Return
        # 1.0 (treated as "well-behaved") rather than dividing by zero.
        theta = self._get_theta(state)
        theta_scale = jnp.linalg.norm(jnp.atleast_1d(theta)) + 1.0
        if float(n_full) < 1e-12 * float(theta_scale):
            return 1.0
        return float(n_frozen / n_full)

    def is_trapped_at(
        self,
        state: dict,
        *,
        eps: float = 1e-3,
        rng_key: Any = None,
    ) -> bool:
        """Cheap binary trap check (round-5/round-7 re-thresholded FD).

        Re-thresholds the mask at a perturbed θ and compares the frozen
        gradient there to the gradient at the original θ.  If both are
        near-zero (the Palais signature), the proxy
        ``|g_frozen| / |Δg_frozen|/eps`` drops to ~0 and the check fires.

        Round-7 Investigation 1: this is reliable for **exact-trap
        detection** (proxy = 0 across all (r, ε) at the trap) but
        unreliable for partial-blindness classification (~46% best
        across the round-5 sweep).  Use as a binary check only.

        Parameters
        ----------
        eps : float, default 1e-3
            Perturbation magnitude.  Larger values catch broader
            partial-blindness zones at the cost of more false positives.
        rng_key : optional
            Reserved; the current implementation uses a deterministic
            ``+1`` perturbation direction.  Multi-direction variants
            would consume this.

        Returns
        -------
        bool
            ``True`` if the proxy ratio is below ``1e-2``: a strong
            signature of a Palais fixed point.
        """
        theta = self._get_theta(state)
        # Frozen gradient at theta.
        g0 = self._compute_frozen_gradient(state)
        # Perturb theta in a deterministic direction (sign(g_full) so
        # the perturbation is anisotropic transverse to Fix(G); falls
        # back to +1 if g_full is also blind).
        g_full = self.compute_full_basis_gradient(state)
        n_full = jnp.linalg.norm(g_full)
        if float(n_full) > 1e-12:
            direction = g_full / n_full
        else:
            direction = jnp.ones_like(theta) / jnp.sqrt(theta.size)
        theta_p = theta + eps * direction
        state_p = self._set_theta(state, theta_p)
        # Re-compute mask at the perturbed theta (the "re-thresholded"
        # piece) so any selection-blindness manifests in the perturbed
        # frozen gradient too.
        mask_p = self.compute_active_set(state_p, prev=state, is_cold_start=False)
        state_p = {**state_p, "mask": mask_p}
        g_p = self._compute_frozen_gradient(state_p)
        # est = |Δg_frozen| / eps -- the variation rate of g_frozen.
        est = jnp.linalg.norm(g_p - g0) / eps
        n_g0 = jnp.linalg.norm(g0)
        proxy = float(n_g0 / (est + 1e-30))
        return proxy < 1e-2

    def symmetry_break(self, state: dict, delta: float) -> dict:
        """Anisotropic perturbation transverse to ``Fix(G)``.

        Sets ``θ ← θ + delta * g_full / ||g_full||``.  By the
        Selection-Equivariance Theorem (module docstring), at
        ``θ ∈ Fix(G)`` the full-basis gradient lies in ``T_θ Fix(G)``
        and is therefore tangent to the trap manifold.  Perturbing in
        the unit gradient direction moves transversely to the trap by
        ``delta``.

        Chen-Ziyin 2023: this is the **only** valid escape direction
        — isotropic noise cannot escape Type-II saddles like the
        Palais fixed points the framework treats here.

        Parameters
        ----------
        state : dict
        delta : float
            Perturbation magnitude.  ``0.0`` returns the state
            unchanged.  Round-7 Inv 2: a single perturbation of
            ``blindness_break_delta = 0.05`` suffices in 1D and 2D
            test cases.

        Returns
        -------
        dict
            New state dict with ``θ`` replaced.
        """
        theta = self._get_theta(state)
        g_full = self.compute_full_basis_gradient(state)
        n_full = jnp.linalg.norm(g_full)
        # Guard against zero g_full (both J_frozen and J_full are flat):
        # fall back to a uniform direction.  In practice this case is
        # already a "persistent trap" and initial_state will raise.
        if float(n_full) < 1e-12:
            direction = jnp.ones_like(theta) / jnp.sqrt(theta.size)
        else:
            direction = g_full / n_full
        theta_new = theta + delta * direction
        return self._set_theta(state, theta_new)
