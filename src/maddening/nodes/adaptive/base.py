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

What the gradient is (and is not)
---------------------------------

On each open region of parameter space where the active set is constant
the returned gradient is the **exact** derivative of the frozen-set
objective.  It ignores the dependence of the active set itself on the
parameters.  Across a region boundary -- where two candidates swap rank
and the set changes -- the frozen-set objective **jumps**: it is not
continuous there, so it is not locally Lipschitz and no Clarke
subgradient exists.  Gradient-based optimisation therefore sees a
*first-order* error whenever it crosses a boundary, equal to the sum of
the jumps crossed.

Measured on the suite's 1-D sine toy (``n_max = 256``, top-``|b|``,
``k = 16``): integrating the returned gradient over ``theta`` in
``[0.40, 0.50]`` gives ``-1.5808e-3`` against a true objective change of
``-2.3598e-3`` -- a 33 % shortfall, accounted for to six digits by the
sum of the 27 jumps crossed in that window.  The error is negligible in
the large-basis-budget regime the pattern is designed for: summed over
the switches in ``theta in [0.40, 0.42]``, the jumps are ``2.5e-1`` of
``|J|`` at ``k = 8`` and ``1.2e-8`` at ``k = 64`` (float64;
``tests/nodes/adaptive/test_active_set_switch.py::
test_the_omitted_term_decays_with_the_active_set_budget`` asserts the
decrease and a ``1e-6`` bound at ``k = 64``).  Registered as anomaly
``MADD-ANO-003``.

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

Selection-Equivariance Theorem
------------------------------

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

* :meth:`AdaptiveNode.gradient_capture_ratio` --
  ``|grad J_frozen| / |grad J_full|``.  **What it measures is
  active-set-budget adequacy**: how much of the full-basis gradient the
  frozen set reproduces.  A symmetry trap drives it to ``0``, but so
  does a budget too small for the objective, and the ratio alone cannot
  tell the two apart on its own.  (Its earlier name,
  ``blindness_ratio``, is kept as a deprecated alias.)
* :meth:`AdaptiveNode.frozen_gradient_vanishes_at` -- a binary check
  that the frozen gradient is negligible against its own rate of
  change.  A Palais fixed point implies that, so a ``False`` rules a
  trap out; a ``True`` does not establish one, because an ordinary
  stationary point looks the same.  ``is_trapped_at`` is kept as a
  deprecated alias of it;
* :meth:`AdaptiveNode.symmetry_break` -- an anisotropic perturbation of
  the parameters along the *full-basis* gradient, which (unlike
  isotropic noise) leaves the fixed-point set in one step.  It helps at
  a symmetry trap **only**; against a budget-limited ratio it makes
  matters worse (audit A3: 0.565 -> 0.060).

Stability
---------

``AdaptiveNode``, ``AdaptiveNodeBlindnessError`` and the module-level
switches are ``@stability(EVOLVING)``, not ``STABLE``: the hook
signatures and the diagnostic's contract have open questions (see
"Open questions for the 0.4.0 API freeze" in
``docs/developer_guide/adaptive_node.md``).  The 0.4.0 API freeze picks
the final level; nothing has shipped, so lowering the promise now is
free and raising it later is not.
"""

from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import (
    NodeMeta, Reference, StabilityLevel,
)
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode

_FALSEY = {"0", "false", "no", "off"}

_DIAGNOSTICS_ENABLED: bool = (
    os.environ.get("MADDENING_ADAPTIVE_DIAGNOSTICS", "1").strip().lower()
    not in _FALSEY
)


@stability(StabilityLevel.EVOLVING)
def set_adaptive_diagnostics(enabled: bool) -> bool:
    """Globally enable or disable the ``AdaptiveNode`` cold-start diagnostic.

    The diagnostic costs two gradient evaluations (one of them
    full-basis) and runs from :meth:`AdaptiveNode.initial_state`, which
    the framework calls from ``GraphManager.add_node`` /
    ``reset_state``, the profiler, the REST API, the sharded-node
    paths, the FMI model description and the hypothesis strategies.
    Turn it off process-wide when none of those calls should pay for
    it.  The initial value comes from the environment variable
    ``MADDENING_ADAPTIVE_DIAGNOSTICS`` (``0``/``false``/``no``/``off``
    disables it).

    Parameters
    ----------
    enabled : bool

    Returns
    -------
    bool
        The previous setting, so callers can restore it.
    """
    global _DIAGNOSTICS_ENABLED
    previous = _DIAGNOSTICS_ENABLED
    _DIAGNOSTICS_ENABLED = bool(enabled)
    return previous


@stability(StabilityLevel.EVOLVING)
def adaptive_diagnostics_enabled() -> bool:
    """Whether the ``AdaptiveNode`` cold-start diagnostic runs at all."""
    return _DIAGNOSTICS_ENABLED


@stability(StabilityLevel.EVOLVING)
class AdaptiveNodeBlindnessError(RuntimeError):
    """The frozen-set gradient cannot be trusted at these parameters.

    Raised by :meth:`AdaptiveNode.initial_state` (through
    :meth:`AdaptiveNode.check_gradient_capture`) in exactly two cases:

    * the gradient-capture ratio is below
      :attr:`AdaptiveNode.gradient_capture_threshold` and the node was
      constructed with ``on_blind="raise"``; or
    * :meth:`AdaptiveNode.cold_start` could not raise the ratio above
      the threshold with one :meth:`AdaptiveNode.symmetry_break`.

    Under the default ``on_blind="warn"`` a low ratio only ever
    **warns**, whatever its cause.  Until 0.4.0 the trap branch raised
    even under ``"warn"``, on the premise that
    :meth:`AdaptiveNode.frozen_gradient_vanishes_at` *established* a
    Palais fixed point.  It does not -- it is a necessary condition, and
    it fires at any stationary point of the frozen objective, including
    the optimum a successful fit converges to.  Refusing there would
    hard-error a user through the escape hatch they had chosen.
    """


@stability(StabilityLevel.EVOLVING)
class AdaptiveNode(SimulationNode):
    """Base class for adaptive solvers with a frozen-active-set adjoint.

    Subclasses implement the selection rule and the frozen-basis solve;
    the base class wires them into a JAX-traceable :meth:`update`, keeps
    the fixed-size ``(c, mask)`` state consistent, routes the adjoint
    through the frozen solve, and provides the cold-start diagnostics.

    ``EVOLVING``, not ``STABLE``: see the module docstring.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Timestep carried by the :class:`SimulationNode` base.
    n_max : int
        Size of the padded coefficient buffer (the candidate basis).
        Every state array has shape ``(n_max,)``; the active set is a
        boolean mask over it.  **Structural**, and deliberately *not*
        stored on ``self.params``: a subclass that wants its basis size
        to survive a config / USD round trip declares its own integer
        parameter for it and forwards it here (see
        ``docs/developer_guide/adaptive_node.md``).
    gradient_capture_threshold, blindness_break_delta, D_threshold : optional
        Per-instance overrides of the class-level constants below.
        ``blindness_threshold=`` is accepted as a deprecated alias of
        ``gradient_capture_threshold=``.
    blindness_gate : bool, default True
        Run :meth:`gradient_capture_ratio` in :meth:`initial_state`.
        Costs two gradient evaluations (one of them full-basis) per
        distinct parameter point; the result is cached per instance, so
        the many framework paths that call ``initial_state()`` pay once.
        Turn off for subclasses that do not implement :meth:`objective`.
        Recorded in ``self.params`` so it survives a round trip.
    on_blind : {"warn", "raise", "ignore"}, default "warn"
        What :meth:`check_gradient_capture` does when the ratio is below
        the threshold.  ``"warn"`` emits a :class:`UserWarning` naming
        the measured ratio, the threshold, which of the two causes the
        evidence points at and the remedies -- and does **not** raise,
        whichever cause it is.  ``"raise"`` raises for a low ratio of
        any cause; it is the strict setting, and the only one that
        refuses to construct.
        ``"ignore"`` skips the diagnostic entirely.  Recorded in
        ``self.params`` so it survives a round trip.
    dtype : optional
        Floating dtype of ``c``.  When given explicitly it is *enforced*
        on the coefficients :meth:`solve_frozen` returns; a non-floating
        dtype is rejected.  The default is JAX's canonical float
        resolved at construction time (``float64`` under
        ``jax_enable_x64``, ``float32`` otherwise) and types the
        cold-start buffer without forcing a cast.  An explicit dtype is
        recorded in ``self.params`` by name (``"float32"``) so it
        survives a round trip; the default records nothing, and a node
        rebuilt without the key follows the canonical float again.
    **params
        Physical constants of the subclass, stored on ``self.params`` and
        exposed through :meth:`params_pytree` / :meth:`param_specs` like
        any other node.

    Attributes
    ----------
    gradient_capture_threshold : float
        ``0.7``.  Below this ratio the active set is judged not to
        reproduce the full-basis gradient (chosen during the design study
        to minimise the expected cost of a false escape versus a missed
        trap over its 1-D and 2-D test problems; on the 1-D sine toy the
        three regimes measure about 0.86, 0.17 and 0.0, bracketed by
        ``tests/nodes/adaptive/test_blindness_diagnostics.py``).
        ``blindness_threshold`` is a
        deprecated alias kept in sync with it.
    blindness_break_delta : float
        ``0.05``.  Magnitude of the :meth:`symmetry_break` perturbation
        (in the design study one perturbation of this size escaped every
        1-D and 2-D trap tested with no drift back; the 1-D minimum was
        0.03).
    D_threshold : int
        ``5``.  Number of trainable parameters above which the cold-start
        check alone is insufficient and
        :meth:`frozen_gradient_vanishes_at` should be run between
        optimiser steps as well (a design-study choice).  The base
        class does not enforce a monitoring policy; subclasses and
        optimisation loops read this constant to decide theirs.

    Notes
    -----
    The three hooks a subclass provides:

    * :meth:`compute_active_set` -- the selection rule (non-differentiable).
    * :meth:`solve_frozen` -- the solve on a frozen mask (differentiable;
      typically one :func:`~maddening.core.solver_utils.ift_linear_solve`).
    * :meth:`objective` -- the scalar the cold-start diagnostics
      differentiate.  Only needed for the diagnostics.
    """

    meta: ClassVar[NodeMeta] = NodeMeta(
        algorithm_id="MADD-NODE-009",
        algorithm_version="1.1.0",
        stability=StabilityLevel.EVOLVING,
        description=(
            "Basis-agnostic adaptive-solver base class with a "
            "frozen-active-set implicit-function-theorem adjoint and "
            "gradient-capture / Palais-trap cold-start diagnostics"
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
            "returned gradient is exact within such a region and ignores "
            "the set's dependence on the parameters.  Across a region "
            "boundary the frozen-set objective jumps, so it is not "
            "locally Lipschitz there and no Clarke subgradient exists; "
            "gradient-based optimisation sees a first-order error equal "
            "to the sum of the jumps it crosses (measured: 33% of the "
            "objective change over theta in [0.40, 0.50] at n=256, k=16; "
            "4e-8 at k=64).  See MADD-ANO-003",
            "solve_frozen with an all-True mask is the full-basis solve "
            "(used by the default full-basis gradient)",
            "The objective used by the blindness diagnostics is a scalar "
            "function of the solved coefficients",
        ),
        limitations=(
            "Abstract: a subclass must supply compute_active_set, "
            "solve_frozen and (for the diagnostics) objective",
            "The frozen-set gradient is blind at Palais fixed points of "
            "the problem's symmetry group; the cold-start diagnostics "
            "detect this at the parameters they are handed but routine "
            "monitoring is the caller's policy (recommended above "
            "D_threshold parameters), and the cold-start call sees the "
            "constructor parameters, not a graph's live pytree",
            "gradient_capture_ratio measures active-set-budget adequacy, "
            "not symmetry alone: a low ratio means either too small a "
            "budget or a trap; frozen_gradient_vanishes_at rules a trap "
            "out when it is False, but a True does not establish one",
            "gradient_capture_ratio and symmetry_break cost a full-basis "
            "gradient (the expensive solve adaptivity exists to avoid); "
            "they are host-side diagnostics, not traceable",
            "The base class cannot repair a NaN gradient produced by a "
            "singular expression that solve_frozen evaluates on "
            "masked-out entries: masking the output protects the value, "
            "not the tangent.  Use mask_safe on the input of the unsafe "
            "operation (the double-where idiom)",
            "The padded buffer costs n_max memory and FLOPs per step "
            "regardless of how many entries are active",
        ),
        references=(
            Reference("Palais1979", "Principle of symmetric criticality (trap mechanism)"),
            Reference("CohenDahmenDeVore2001", "Adaptive wavelet methods; active-set framing"),
            Reference("Blondel2022", "Implicit differentiation through a linear solve"),
        ),
        hazard_hints=(
            "A gradient step taken across an active-set change misses a "
            "jump in the objective: the returned gradient is neither the "
            "derivative nor a subgradient there (the objective is "
            "discontinuous), so line searches and quasi-Newton methods "
            "see an inconsistent objective/gradient pair and first-order "
            "methods accumulate the sum of the crossed jumps as bias",
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
            "Gradient-capture ratio": "maddening.nodes.adaptive.base.AdaptiveNode.gradient_capture_ratio",
            "Cold-start diagnostic policy": "maddening.nodes.adaptive.base.AdaptiveNode.check_gradient_capture",
            "Symmetry break": "maddening.nodes.adaptive.base.AdaptiveNode.symmetry_break",
        },
    )

    # Constants fixed by the design study (the class docstring gives the
    # reason for each); per-instance overrides via
    # the constructor.  Kept out of :meth:`params_pytree`: they steer
    # host-side diagnostics, they are not physics a fit could identify.
    gradient_capture_threshold: float = 0.7
    blindness_threshold: float = 0.7  # deprecated alias, kept in sync
    blindness_break_delta: float = 0.05
    D_threshold: int = 5

    #: Constructor keywords recorded in ``self.params`` (so they survive
    #: a config / USD round trip) but excluded from the differentiable
    #: parameter pytree.
    DIAGNOSTIC_SETTINGS: ClassVar[tuple[str, ...]] = (
        "blindness_gate", "on_blind", "gradient_capture_threshold",
        "blindness_break_delta", "D_threshold",
    )

    ON_BLIND_POLICIES: ClassVar[tuple[str, ...]] = ("warn", "raise", "ignore")

    def __init__(
        self,
        name: str,
        timestep: float,
        *,
        n_max: int,
        gradient_capture_threshold: Optional[float] = None,
        blindness_threshold: Optional[float] = None,
        blindness_break_delta: Optional[float] = None,
        D_threshold: Optional[int] = None,
        blindness_gate: bool = True,
        on_blind: str = "warn",
        dtype: Any = None,
        **params: Any,
    ):
        n_max = _positive_int(n_max, "n_max")
        if blindness_threshold is not None:
            warnings.warn(
                "AdaptiveNode(blindness_threshold=...) is deprecated: the "
                "diagnostic measures active-set-budget adequacy, not "
                "blindness alone.  Use gradient_capture_threshold=.",
                DeprecationWarning, stacklevel=2,
            )
            if gradient_capture_threshold is None:
                gradient_capture_threshold = blindness_threshold
        if on_blind not in self.ON_BLIND_POLICIES:
            raise ValueError(
                f"on_blind must be one of {self.ON_BLIND_POLICIES!r}, "
                f"got {on_blind!r}"
            )

        settings: dict[str, Any] = {
            "blindness_gate": bool(blindness_gate),
            "on_blind": str(on_blind),
        }
        if gradient_capture_threshold is not None:
            settings["gradient_capture_threshold"] = _non_negative_float(
                gradient_capture_threshold, "gradient_capture_threshold",
            )
        if blindness_break_delta is not None:
            settings["blindness_break_delta"] = _non_negative_float(
                blindness_break_delta, "blindness_break_delta",
            )
        if D_threshold is not None:
            settings["D_threshold"] = _positive_int(D_threshold, "D_threshold")

        dt = jnp.zeros((), dtype=dtype).dtype
        if not jnp.issubdtype(dt, jnp.floating):
            raise ValueError(
                f"dtype must be a floating dtype (it types the coefficient "
                f"vector c), got {dtype!r} -> {dt}"
            )
        if dtype is not None:
            # By name, the spelling a config and a USD stage can carry.  It
            # used to be kept off ``self.params``, so a round trip rebuilt
            # the node at the canonical float: a float32 node reloaded as
            # float64 under x64, and lost the enforcement below either way.
            # The name asked for, not the one resolved: a reload resolves it
            # the same way under the same x64 setting.
            settings["dtype"] = np.dtype(dtype).name

        # ``n_max`` is structural and stays out of ``self.params``: it is
        # not a parameter a fit could identify, and putting it there made
        # every round-tripped subclass receive it twice (audit A4).
        super().__init__(name, timestep, **settings, **params)
        self.n_max = n_max
        self.blindness_gate = settings["blindness_gate"]
        self.on_blind = settings["on_blind"]

        self.dtype = dt
        # Only an explicitly requested dtype is *enforced* on the solve
        # output.  The default is JAX's canonical float resolved at
        # construction time, and forcing that would silently downcast a
        # float64 solve in a process that enabled x64 afterwards.
        self._enforce_dtype = dtype is not None

        # Resolve the class-level constants, honouring a subclass that
        # overrode the deprecated ``blindness_threshold`` in its body.
        cls_new = type(self).gradient_capture_threshold
        cls_old = type(self).blindness_threshold
        if cls_old != cls_new and cls_new == AdaptiveNode.gradient_capture_threshold:
            cls_new = cls_old
        threshold = settings.get("gradient_capture_threshold", cls_new)
        self.gradient_capture_threshold = float(threshold)
        self.blindness_threshold = float(threshold)
        if "blindness_break_delta" in settings:
            self.blindness_break_delta = settings["blindness_break_delta"]
        if "D_threshold" in settings:
            self.D_threshold = settings["D_threshold"]

        # ``initial_state()`` is called from add_node, reset_state, the
        # profiler, the REST API, the sharded-node paths, the FMI model
        # description and the hypothesis strategies; the diagnostic is
        # a pure function of the parameters, so memoise it (audit A13).
        self._capture_cache: dict[Any, float] = {}
        self._trapped_cache: dict[Any, bool] = {}

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
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

    @abstractmethod
    def solve_frozen(self, state: dict, mask: jax.Array, params: dict) -> dict:
        """Solve on the frozen active set ``mask``.

        The differentiable half of the pattern.  Build the masked
        operator (identity on inactive rows keeps the buffer size fixed)
        and the masked right-hand side, call
        :func:`maddening.core.solver_utils.ift_linear_solve`, and return
        the coefficients.  ``jax.grad`` through this method is the
        frozen-set adjoint.

        .. warning::
           Masking the *output* protects the value, not the tangent.  If
           this method evaluates an expression that is singular on the
           inactive entries (``jnp.sqrt`` of something negative there, a
           division by a zeroed diagonal), the forward pass is clean and
           ``jax.grad`` returns ``NaN``.  Sanitise the **input** of the
           unsafe operation with :meth:`mask_safe` -- the double-``where``
           idiom -- which the base class cannot do for you.

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
        """Scalar objective ``J`` the cold-start diagnostics differentiate.

        Typically a sensor reading or integral of the solved field.
        Required by :meth:`gradient_capture_ratio`,
        :meth:`frozen_gradient_vanishes_at`, :meth:`symmetry_break` and
        the cold-start diagnostic; :meth:`update` never calls it.

        Deliberately **not** ``@abstractmethod`` (unlike
        :meth:`compute_active_set` and :meth:`solve_frozen`): it is
        optional for a subclass that runs with ``blindness_gate=False``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override objective to use the "
            "cold-start diagnostics (or construct with blindness_gate=False)"
        )

    @staticmethod
    def mask_safe(mask: jax.Array, x: jax.Array, fill: float = 1.0) -> jax.Array:
        """Inner half of the double-``where`` idiom: sanitise an operand.

        Returns ``x`` where ``mask`` is true and ``fill`` elsewhere, so
        that an operation which is singular on the inactive entries is
        never evaluated there -- neither in the forward pass nor in the
        tangent.  Use it on the **input** of the unsafe operation inside
        :meth:`solve_frozen`::

            d = self.mask_safe(mask, diagonal, fill=1.0)   # never 0 off-mask
            c = ift_linear_solve(lambda v: d * v, rhs, solver="cg")

        Masking only the result (which the base class does for you) keeps
        the value finite but leaves the gradient ``NaN``.

        **Rank.**  This is a plain ``jnp.where``, so the mask broadcasts
        against the **last** axis of ``x``.  That is what you want for a
        1-D operand of shape ``(n_max,)`` and for a batch of shape
        ``(..., n_max)``.  It is *not* what you want for a square
        operator of shape ``(n_max, n_max)``: the mask then fills whole
        **columns** and leaves the rows untouched, silently and with no
        error.  Mask an operator's rows explicitly::

            A = self.mask_safe(mask[:, None], A, fill=0.0)   # rows
            A = self.mask_safe(mask, A, fill=0.0)            # columns

        Parameters
        ----------
        mask : jax.Array
            Boolean active set, shape ``(n_max,)``.  Broadcast against
            the last axis of ``x``; reshape it yourself (``mask[:,
            None]``) to mask another one.
        x : jax.Array
            Operand to sanitise.  Any rank, subject to the broadcasting
            rule above.
        fill : float, default 1.0
            Value substituted on the inactive entries.  Must be safe for
            the operation that follows (``1.0`` for division, square
            roots and logarithms).

        Returns
        -------
        jax.Array
        """
        arr = jnp.asarray(x)
        return jnp.where(mask, arr, jnp.asarray(fill, dtype=arr.dtype))

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

    def params_pytree(self) -> dict:
        """Differentiable parameters, without the diagnostic settings.

        The diagnostic knobs live in ``self.params`` so that a config /
        USD round trip rebuilds the node with the settings it was built
        with, but they are host-side policy, not physics: a fit must
        never see them as leaves.
        """
        return {
            k: v for k, v in super().params_pytree().items()
            if k not in self.DIAGNOSTIC_SETTINGS
        }

    def initial_state(self) -> dict:
        """Cold-start state at the constructor parameters.

        Selects the active set with ``is_cold_start=True``, solves on
        it, and -- when ``blindness_gate`` is on -- runs
        :meth:`check_gradient_capture` at the constructor parameters.
        A low ratio *warns* under the default ``on_blind="warn"``,
        whatever its cause, and raises only under ``on_blind="raise"``.

        The check sees the **constructor** parameters.  Once the node is
        in a graph the live values live in ``gm.params["nodes"][name]``;
        call ``node.check_gradient_capture(gm.params["nodes"][name])``
        after seeding them.
        """
        state = self._cold_start_state(self.params)
        self.check_gradient_capture(state=state)
        return state

    def check_gradient_capture(
        self,
        params: Optional[dict] = None,
        *,
        state: Optional[dict] = None,
        on_blind: Optional[str] = None,
    ) -> Optional[float]:
        """Run the cold-start diagnostic at ``params`` and apply the policy.

        Unlike :meth:`initial_state`, this accepts the parameters
        actually in use -- pass ``gm.params["nodes"][name]`` to check the
        point a graph is running at, which is what ``sysid.fit``
        optimises and what the constructor values need not be.

        Parameters
        ----------
        params : dict, optional
            Parameter pytree; ``None`` means the constructor constants.
        state : dict, optional
            Cold-start state to reuse; rebuilt at ``params`` if omitted.
        on_blind : {"warn", "raise", "ignore"}, optional
            Override the instance's policy for this call.

        Returns
        -------
        float or None
            The measured ratio, or ``None`` when the diagnostic did not
            run (gate off, policy ``"ignore"``, diagnostics globally
            disabled, or nothing trainable to differentiate).

        Raises
        ------
        AdaptiveNodeBlindnessError
            When the ratio is below :attr:`gradient_capture_threshold`
            **and** the policy is ``"raise"``.  Under ``"warn"`` a low
            ratio only warns, whatever its cause.
        """
        policy = self.on_blind if on_blind is None else on_blind
        if policy not in self.ON_BLIND_POLICIES:
            raise ValueError(
                f"on_blind must be one of {self.ON_BLIND_POLICIES!r}, "
                f"got {policy!r}"
            )
        # Only evaluate when the answer can change the outcome.
        if (
            not _DIAGNOSTICS_ENABLED
            or not self.blindness_gate
            or policy == "ignore"
            or not any(self._trainable(k) for k in self.params_pytree())
        ):
            return None

        pt = self._pytree(params)
        key = _fingerprint(pt)
        ratio = self._capture_cache.get(key) if key is not None else None
        if ratio is None:
            if state is None:
                state = self._cold_start_state(self._merged(params))
            ratio = self.gradient_capture_ratio(state, params)
            if key is not None:
                self._capture_cache[key] = ratio
        if ratio >= self.gradient_capture_threshold:
            return ratio

        if state is None:
            state = self._cold_start_state(self._merged(params))
        trapped = self._trapped_cache.get(key) if key is not None else None
        if trapped is None:
            trapped = self.frozen_gradient_vanishes_at(state, params)
            if key is not None:
                self._trapped_cache[key] = trapped
        where = "the constructor parameters" if params is None else "the supplied parameters"
        head = (
            f"{type(self).__name__} {self.name!r}: gradient-capture ratio "
            f"{ratio:.3f} is below the threshold "
            f"{self.gradient_capture_threshold:.3f} at {where}."
        )
        if trapped:
            message = (
                f"{head}  frozen_gradient_vanishes_at() is True: the "
                "frozen-set gradient is negligible against its own rate of "
                "change along the escape direction, so it carries no usable "
                "descent information here.  That is what a Palais fixed "
                "point of the problem's symmetry looks like -- but it is a "
                "necessary condition, not a sufficient one, and an ordinary "
                "stationary point of the objective produces it too.  If the "
                "operator, source and objective do share a symmetry that "
                "fixes these parameters, the remedy is cold_start() (one "
                "symmetry_break along the full-basis gradient) and seeding "
                "gm.params with the pytree it returns, or perturbing the "
                "parameters yourself.  If they do not, check instead "
                "whether you have simply converged (|grad J_full| small "
                "too) or whether the active set is degenerate; neither is "
                "helped by a perturbation.  Pass on_blind='ignore' (or "
                "blindness_gate=False) to silence this check."
            )
        else:
            message = (
                f"{head}  frozen_gradient_vanishes_at() is False, which rules "
                "a symmetry trap out: the active-set budget is too small to "
                "reproduce the "
                "full-basis gradient here.  The ratio tracks the budget (on the "
                "1-D sine toy at n_max=256 it measures 0.16 at k=4, 0.57 at "
                "k=8, 0.85 at k=16 and 1.00 from k=32, at every n). Remedies: "
                "raise the active-set budget; or accept a frozen gradient that "
                "captures this fraction of the full one and silence the check "
                "with gradient_capture_threshold=<lower>, on_blind='ignore' or "
                "blindness_gate=False.  cold_start() / symmetry_break() do "
                "*not* help here -- they move along the full-basis gradient, "
                "which at a budget-limited point lowers the ratio further "
                "(measured 0.565 -> 0.060)."
            )
        # ``"warn"`` warns, whatever the cause.  It used to raise on the
        # trap branch even here, on the grounds that a Palais fixed point
        # was the one cause the diagnostic could *establish* -- and that
        # premise is false (see frozen_gradient_vanishes_at).  The check
        # also fires at an ordinary interior optimum, so the old behaviour
        # hard-errored a user at the moment their fit converged, through
        # the escape hatch they had explicitly chosen, with a message
        # asserting a symmetry that was not there.  ``on_blind="raise"``
        # is the strict setting and still raises for either cause.
        if policy == "raise":
            raise AdaptiveNodeBlindnessError(message)
        warnings.warn(message, UserWarning, stacklevel=2)
        return ratio

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """One adaptation step: select, freeze, solve.

        ``params`` is the node's entry of the graph parameter pytree;
        ``None`` (a direct call) uses the constructor constants.  The
        mask returned by :meth:`compute_active_set` is committed under
        ``stop_gradient``; ``jax.grad`` of anything downstream reaches
        ``params`` only through :meth:`solve_frozen`, and is therefore
        the derivative of the branch this call selected -- exact within
        the active-set region and blind to the region's boundary (see
        the module docstring and ``MADD-ANO-003``).

        ``boundary_inputs`` and ``dt`` are ignored, so an ``AdaptiveNode``
        is an edge *source* only; whether the hooks should receive them
        is an open question for the 0.4.0 API freeze.
        """
        del boundary_inputs, dt  # the base class uses neither
        p = self._merged(params)
        mask = self.compute_active_set(state, p, prev=state["mask"])
        mask = self._validated_mask(mask, is_cold_start=False)
        return self._solve_and_pack(state, mask, p)

    # ------------------------------------------------------------------
    # Diagnostics and mitigation
    # ------------------------------------------------------------------

    def cold_start(self, params: Optional[dict] = None) -> tuple[dict, dict]:
        """Cold start with one automatic **symmetry-trap** escape attempt.

        1. Build the cold-start state at ``params`` and measure
           :meth:`gradient_capture_ratio`.
        2. If it is at least :attr:`gradient_capture_threshold`, return.
        3. Otherwise apply one :meth:`symmetry_break` of
           :attr:`blindness_break_delta`, rebuild the state there and
           re-measure.
        4. Raise :class:`AdaptiveNodeBlindnessError` if the ratio is
           still low.

        This is the remedy for a **Palais trap** (rule one out with
        :meth:`frozen_gradient_vanishes_at` first, then confirm the
        symmetry from the problem itself).  It is *not* a remedy for a ratio
        held down by too small an active-set budget: the perturbation
        moves along the full-basis gradient, which in the audited
        budget-limited case lowered the ratio from 0.565 to 0.060.

        Cost: at most two ratios and one symmetry break.  The design
        study found no cheaper substitute for the full diagnostic at cold
        start, and found one perturbation sufficient.

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
        if self.gradient_capture_ratio(state, pt) >= self.gradient_capture_threshold:
            return state, pt
        pt = self.symmetry_break(state, pt)
        state = self._cold_start_state(self._merged(pt))
        ratio = self.gradient_capture_ratio(state, pt)
        if ratio >= self.gradient_capture_threshold:
            return state, pt
        raise AdaptiveNodeBlindnessError(
            f"{type(self).__name__} {self.name!r}: gradient-capture ratio "
            f"{ratio:.3f} is still below the threshold "
            f"{self.gradient_capture_threshold:.3f} after one "
            f"symmetry_break of delta={self.blindness_break_delta}.  Either "
            "the parameters are bound to a Palais fixed point (rule one "
            "out with frozen_gradient_vanishes_at) -- perturb them and "
            "retry -- or the active-set "
            "budget, not the symmetry, is what holds the ratio down, and "
            "no perturbation will fix that."
        )

    def gradient_capture_ratio(
        self, state: dict, params: Optional[dict] = None,
    ) -> float:
        """``|grad J_frozen| / |grad J_full|`` at ``params``.

        The fraction of the full-basis gradient magnitude that the
        frozen active set reproduces: the gradient of :meth:`objective`
        with respect to the trainable parameters through the active set
        **selected at** ``params``, over the same gradient with every
        basis function active.

        What it measures is **active-set-budget adequacy**.  ``~1`` means
        the frozen adjoint reproduces the full one.  A low value has two
        possible causes, which this number cannot separate:

        * the budget is too small for the objective at these parameters
          (the common case -- on the suite's 1-D toy the ratio is a
          function of ``k`` alone, 0.16/0.57/0.85/1.00 at k=4/8/16/32,
          at every ``n``); or
        * the parameters sit at a Palais fixed point of the problem's
          symmetry (:meth:`frozen_gradient_vanishes_at` rules this one
          out when it is ``False``; a ``True`` is consistent with it but
          does not establish it).

        Values above ``1`` are over-amplified but direction-accurate.
        Returns ``1.0`` as a sentinel when the full gradient itself is
        negligible (an interior extremum of ``J``), where the ratio is
        undefined.  On the way there it warns if the full-basis gradient
        with respect to a **trainable** leaf is *bitwise* zero -- see
        :meth:`_warn_on_bitwise_zero_full_gradient`, the only cheap oracle
        for a parameter baked into a constant in ``__init__``.

        The active set is re-selected at ``params`` rather than read from
        ``state["mask"]``: a mask chosen at a healthy point otherwise
        reports a healthy ratio at a trap (audit A5 measured 1.005 at the
        exact trap with a stale mask).

        Cost: two gradient evaluations, one of them full-basis.  A
        host-side diagnostic (returns a Python float), not traceable.
        """
        mask = self._selected_mask(state, params)
        g_frozen = self._objective_gradient(state, params, mask)
        g_full = self.compute_full_basis_gradient(state, params)
        self._warn_on_bitwise_zero_full_gradient(g_full)
        n_full = _tree_norm(g_full)
        scale = 1.0 + _tree_norm(self._pytree(params))
        if n_full < 1e-12 * scale:
            return 1.0
        return float(_tree_norm(g_frozen) / n_full)

    def _warn_on_bitwise_zero_full_gradient(self, g_full: dict) -> None:
        """Flag a trainable leaf whose full-basis gradient is *exactly* zero.

        A parameter the objective is genuinely flat in returns a **small**
        gradient; a *bitwise* ``0.0`` means the parameter is not in the
        computation graph at all.  Far and away the commonest cause in an
        ``AdaptiveNode`` is a basis array built in ``__init__`` from that
        parameter: the array is then a Python constant, holding the
        constructor's float rather than the traced value the graph
        injects, and nothing downstream of it is a function of the
        parameter.  ``jax.grad`` returns ``0.0`` and a central finite
        difference returns ``0.0`` too -- both read the same baked numbers
        -- so the usual oracle agrees with the wrong answer and
        ``compile()`` has nothing to object to.  ``static_data_deps``
        cannot see it either: that guard only walks arrays published
        through ``static_data``, and only ones whose provenance the
        subclass declared.

        An exact zero is not proof: a problem whose objective is *exactly*
        symmetric in a parameter produces one honestly.  The warning says
        so and names both readings.

        Eager-only, like :meth:`_warn_on_non_finite_off_mask`: under a
        trace there is nothing to read.  Non-trainable leaves are skipped
        -- :meth:`_objective_gradient` zeroes those by construction.
        """
        if not _DIAGNOSTICS_ENABLED:
            return
        dead = []
        for key in sorted(g_full):
            if not self._trainable(key):
                continue
            try:
                arr = np.asarray(g_full[key])
            except (
                jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError,
                TypeError, ValueError,
            ):
                return  # traced: nothing to inspect
            if arr.size and bool(np.all(arr == 0.0)):
                dead.append(key)
        if not dead:
            return
        warnings.warn(
            f"{type(self).__name__} {self.name!r}: the full-basis gradient "
            f"with respect to trainable parameter(s) {dead} is exactly 0.0 "
            "(bitwise, not merely small).  A parameter the objective is "
            "flat in gives a small gradient, not a bitwise zero, so this "
            "usually means the parameter is not in the computation at all: "
            "an array used by solve_frozen or objective was built in "
            "__init__ from it and is now a constant holding the "
            "constructor's value.  jax.grad and a finite difference both "
            "return 0.0 there -- they read the same baked numbers -- so "
            "this check is the only cheap way to see it.  Recompute that "
            "array inside solve_frozen / objective from the params "
            "argument, or declare the parameter ParamSpec(trainable=False) "
            "if it is genuinely not fitted.  If the objective really is "
            "exactly symmetric in "
            f"{dead}, this is honest and you can silence it with "
            "blindness_gate=False or set_adaptive_diagnostics(False).",
            UserWarning, stacklevel=3,
        )

    @stability(StabilityLevel.DEPRECATED)
    def blindness_ratio(self, state: dict, params: Optional[dict] = None) -> float:
        """Deprecated alias of :meth:`gradient_capture_ratio`.

        The name asserted a cause (symmetry blindness) the number cannot
        establish; see :meth:`gradient_capture_ratio`.
        """
        warnings.warn(
            "AdaptiveNode.blindness_ratio() is deprecated: the number "
            "measures active-set-budget adequacy, not blindness alone.  "
            "Use gradient_capture_ratio().",
            DeprecationWarning, stacklevel=2,
        )
        return self.gradient_capture_ratio(state, params)

    def frozen_gradient_vanishes_at(
        self, state: dict, params: Optional[dict] = None, *, eps: float = 1e-3,
    ) -> bool:
        """Is the frozen-set gradient negligible against its own rate of change?

        Perturbs the parameters by ``eps`` along the full-basis gradient,
        re-selects the active set there, and compares the frozen gradient
        at the two points.  Returns ``True`` when the proxy
        ``|g_0| / (|g_eps - g_0| / eps)`` falls below ``1e-2``: the frozen
        gradient is flat where its own variation is not.  Reliable for
        exact traps, not a continuous estimator of partial blindness --
        use :meth:`gradient_capture_ratio` for that.

        What a ``True`` establishes, and what it does not
        ------------------------------------------------
        A Palais fixed point of the problem's symmetry produces exactly
        this signature, so a ``False`` **rules a trap out** -- which is
        what makes the check worth running.  The converse does not hold.
        Nothing in the expression looks at the mask's contents, at a group
        action or at a fixed-point set, and the same collapse appears at
        *any* stationary point of the frozen objective, symmetric or not
        -- including the interior optimum a successful fit ends at
        (``dJ_frozen/dtheta = 2.6e-14`` at ``theta = 0.343973`` on the 1-D
        sine toy, whose only reflection fixed point is ``theta = 0.5``).
        It is a necessary condition for a trap, never a sufficient one.

        Read a ``True`` as "the frozen gradient carries no usable descent
        information at these parameters", and establish the *cause*
        separately -- from the symmetry of the operator, source and
        objective, which is a property of the problem and not of this
        number.  Before deciding it is a trap, rule out the two ordinary
        explanations: the optimiser has converged (``|grad J_full|`` is
        small too -- :meth:`gradient_capture_ratio` returns its ``1.0``
        sentinel there), or the active set is degenerate.  An *empty*
        active set produced the same signature until 0.4.0; it is now
        refused by the ``compute_active_set`` contract before it can reach
        this check.

        Returns
        -------
        bool
        """
        pt = self._pytree(params)
        g0 = self._objective_gradient(state, pt, self._selected_mask(state, pt))
        direction = self._escape_direction(state, pt)
        pt_eps = jax.tree.map(lambda p, d: p + eps * d, pt, direction)
        mask_eps = self.compute_active_set(state, self._merged(pt_eps), prev=state["mask"])
        mask_eps = jnp.asarray(mask_eps, dtype=bool)
        g_eps = self._objective_gradient(state, pt_eps, mask_eps)
        rate = _tree_norm(jax.tree.map(lambda a, b: a - b, g_eps, g0)) / eps
        proxy = _tree_norm(g0) / (rate + 1e-30)
        return bool(proxy < 1e-2)

    @stability(StabilityLevel.DEPRECATED)
    def is_trapped_at(
        self, state: dict, params: Optional[dict] = None, *, eps: float = 1e-3,
    ) -> bool:
        """Deprecated alias of :meth:`frozen_gradient_vanishes_at`.

        The name asserted a cause the measurement cannot establish: a
        Palais symmetry trap implies a vanishing frozen gradient, but an
        ordinary stationary point produces one too.  See
        :meth:`frozen_gradient_vanishes_at`.
        """
        warnings.warn(
            "AdaptiveNode.is_trapped_at() is deprecated: it measures "
            "whether the frozen-set gradient is negligible against its own "
            "rate of change, which a Palais trap implies but which does not "
            "imply a Palais trap -- an ordinary stationary point, including "
            "a converged optimum, fires it too.  Use "
            "frozen_gradient_vanishes_at().",
            DeprecationWarning, stacklevel=2,
        )
        return self.frozen_gradient_vanishes_at(state, params, eps=eps)

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
        """``self.params`` overlaid with an injected pytree, refusing a key it does not have.

        ``update``, the cold start and every diagnostic overlay through
        here.  A key that is not a constructor parameter used to be merged
        and then never read, so ``update(..., params={"thetta": 0.9})``
        returned the answer at the constructor's ``theta``, eagerly and
        under ``jit`` alike, for every subclass.  Keys are static under a
        trace, so the check costs nothing there.
        """
        if params is None:
            return self.params
        unknown = sorted(set(params) - set(self.params), key=str)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: unknown parameter "
                f"key(s) {unknown} -- not constructor parameters "
                f"({sorted(self.params)}); the leaves a graph trains are "
                f"{sorted(self.params_pytree())}."
            )
        return {**self.params, **params}

    def _pytree(self, params: Optional[dict]) -> dict:
        """Full parameter pytree: the defaults overlaid with ``params``.

        Keys that are neither pytree leaves nor structural entries of
        ``self.params`` are rejected rather than silently dropped: a
        typo'd parameter name used to be ignored by every diagnostic
        (audit A12).
        """
        base = self.params_pytree()
        if params is None:
            return base
        unknown = sorted(set(params) - set(base) - set(self.params))
        if unknown:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: unknown parameter "
                f"key(s) {unknown} -- not leaves of params_pytree() "
                f"({sorted(base)}) nor constructor parameters."
            )
        return {**base, **{k: v for k, v in params.items() if k in base}}

    def _selected_mask(self, state: dict, params: Optional[dict]) -> jax.Array:
        """The active set the selection rule picks **at** ``params``.

        The diagnostics evaluate a function of the parameters, so they
        must not reuse a mask that was selected somewhere else.
        """
        mask = self.compute_active_set(
            state, self._merged(params), prev=state.get("mask"),
        )
        return jnp.asarray(mask, dtype=bool)

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
        mask = self._validated_mask(mask, is_cold_start=True)
        return self._solve_and_pack(empty, mask, merged)

    def _validated_mask(self, mask: Any, *, is_cold_start: bool) -> jax.Array:
        """Check a :meth:`compute_active_set` result against the contract,
        then commit it under ``stop_gradient``.

        Three ways a selection rule goes wrong, all of which used to be
        accepted, and two of which no diagnostic can see afterwards:

        * **wrong shape** -- the buffer is fixed size, so this is fatal;
        * **a non-boolean dtype** -- scores or an ``argsort`` become
          all-true under a bool cast and the node silently degenerates
          into a full-basis solver (see finding 4 of the 0.4.0 adaptive
          audit);
        * **an empty active set** -- solves to ``c = 0`` with an exactly
          zero gradient, which the diagnostics then misread as a Palais
          symmetry trap.

        Shape and dtype are static, so they are checked on every call,
        traced or not.  Emptiness is a *value*, so it is checked only
        when the mask is concrete -- eagerly, and in particular on the
        cold-start path every ``initial_state()`` takes.  Under ``jit``
        or ``vmap`` the mask is a tracer and there is nothing to read.
        """
        arr = jnp.asarray(mask)
        if arr.shape != (self.n_max,):
            raise ValueError(
                f"{type(self).__name__}.compute_active_set returned shape "
                f"{arr.shape}; expected ({self.n_max},)"
            )
        if not jnp.issubdtype(arr.dtype, jnp.bool_):
            raise ValueError(
                f"{type(self).__name__}.compute_active_set returned dtype "
                f"{arr.dtype}; expected a boolean array.  The mask is the "
                "active set itself, not a score: a float score array or an "
                "argsort permutation is truthy on every non-zero entry, so "
                "the node would quietly become a full-basis O(n_max) solver "
                "while gradient_capture_ratio reported 1.00 -- because the "
                "frozen set would *be* the full set -- and nothing else "
                "would object.  Return the comparison rather than the "
                "scores: `scores >= threshold`, or scatter the top-k "
                "indices into a boolean buffer with "
                "`jnp.zeros(n_max, bool).at[idx].set(True)`.  A 0/1 integer "
                "mask is refused for the same reason; write `!= 0` if that "
                "is what you meant."
            )
        concrete = self._concrete_mask(arr)
        if concrete is not None and not concrete.any():
            where = (
                "at the cold start (is_cold_start=True)" if is_cold_start
                else "during update"
            )
            raise ValueError(
                f"{type(self).__name__} {self.name!r}.compute_active_set "
                f"returned an empty active set {where}.  An empty set is "
                "refused rather than solved: the frozen solve returns c = 0 "
                "with an exactly zero gradient, and every diagnostic then "
                "misreads it -- gradient_capture_ratio measures 0.0 and "
                "frozen_gradient_vanishes_at() is True, so a problem with no "
                "symmetry at all is reported as a Palais trap and none of "
                "the remedies that error names can help (the set stays empty "
                "at every theta).  At a cold start the coefficients are all "
                "zero, so a rule that thresholds |c| selects nothing: branch "
                "on the is_cold_start argument and seed a non-empty set "
                "there (the coarsest scale, or the top-k by score).  On "
                "later calls keep it non-empty through the prev argument -- "
                "the hysteresis idiom (start from prev, add above eps_add, "
                "remove below eps_remove < eps_add) never empties a set that "
                "started non-empty."
            )
        return jax.lax.stop_gradient(arr)

    @staticmethod
    def _concrete_mask(mask: jax.Array) -> Optional[np.ndarray]:
        """``mask`` as a NumPy bool array, or ``None`` when it is traced."""
        try:
            return np.asarray(mask, dtype=bool)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
            TypeError, ValueError,
        ):
            return None

    def _solve_and_pack(self, state: dict, mask: jax.Array, merged: dict) -> dict:
        new = self.solve_frozen(state, mask, merged)
        c = jnp.asarray(new["c"])
        if c.shape != (self.n_max,):
            raise ValueError(
                f"{type(self).__name__}.solve_frozen returned c of shape "
                f"{c.shape}; expected ({self.n_max},)"
            )
        self._warn_on_non_finite_off_mask(c, mask)
        c = jnp.where(mask, c, jnp.zeros((), dtype=c.dtype))
        if self._enforce_dtype and c.dtype != self.dtype:
            c = c.astype(self.dtype)
        return {**state, **new, "c": c, "mask": mask}

    def _warn_on_non_finite_off_mask(self, c: jax.Array, mask: jax.Array) -> None:
        """Flag the ``jnp.where``-gradient trap while it is still cheap.

        A non-finite coefficient on an inactive entry is erased by the
        mask in the forward pass, so the value looks right and only
        ``jax.grad`` goes ``NaN``.  The base class cannot repair that
        (the tangent is poisoned inside the subclass's own expression),
        but it can say so.  Eager-only: under ``jit``/``grad`` the
        arrays are tracers and the check is skipped.
        """
        if not _DIAGNOSTICS_ENABLED:
            return
        try:
            values = np.asarray(c)
            active = np.asarray(mask, dtype=bool)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
            TypeError, ValueError,
        ):
            return  # traced: nothing to inspect, and nowhere to report it
        if bool(np.all(np.isfinite(values))):
            return
        if not bool(np.all(np.isfinite(np.where(active, values, 0.0)))):
            return  # the active block is non-finite too: a loud failure
        warnings.warn(
            f"{type(self).__name__} {self.name!r}: solve_frozen returned "
            "non-finite coefficients that the active-set mask then erased. "
            "The forward pass is clean but jax.grad through this solve will "
            "be NaN: the tangent of the singular expression is poisoned "
            "before the mask is applied.  Sanitise the *input* of the "
            "unsafe operation with AdaptiveNode.mask_safe(mask, x, fill) "
            "(the double-where idiom) instead of masking only its result.",
            UserWarning, stacklevel=3,
        )

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


def _positive_int(value: Any, name: str) -> int:
    """Validate a structural count: a positive integer, exactly.

    Integral floats are accepted (a JSON round trip turns ``64`` into
    ``64.0``); ``2.7`` and ``"8"`` are not.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, float) or (
        hasattr(value, "dtype") and jnp.issubdtype(getattr(value, "dtype"), jnp.floating)
    ):
        as_float = float(value)
        if not as_float.is_integer():
            raise ValueError(
                f"{name} must be a whole number, got {value!r} "
                f"(silently truncating it would shrink the basis)"
            )
        value = as_float
    elif not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer, got {value!r} of type "
            f"{type(value).__name__}"
        )
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out


def _non_negative_float(value: Any, name: str) -> float:
    """Validate a diagnostic constant: a finite, non-negative float."""
    if isinstance(value, bool) or isinstance(value, str):
        raise ValueError(f"{name} must be a non-negative float, got {value!r}")
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative float, got {value!r}")
    return out


def _fingerprint(tree: Any) -> Optional[tuple]:
    """Hashable digest of a concrete parameter pytree, or ``None``.

    ``None`` for anything traced or otherwise not convertible, which
    simply disables the diagnostic cache for that call.
    """
    try:
        return tuple(
            sorted(
                (str(k), np.asarray(v).dtype.str, np.asarray(v).tobytes())
                for k, v in tree.items()
            )
        )
    except (
        AttributeError, TypeError, ValueError,
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None
