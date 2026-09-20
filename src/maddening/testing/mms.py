"""Method of Manufactured Solutions: the measured order of convergence.

:mod:`maddening.testing.verification` checks the contracts every node
shares — finite outputs, preserved structure, determinism, JIT/eager
agreement, finite gradients.  Every one of those compares the code to
itself, so none of them can see a *wrong discretisation*: a Laplacian
with the wrong weight at the boundary is finite, deterministic,
JIT-consistent and perfectly differentiable.  This module is the other
half of the battery.  It compares the code to the mathematics.

The Method of Manufactured Solutions does that without needing a
closed-form solution of the PDE.  Pick any smooth field ``u*``,
substitute it into the equation the node claims to solve, and whatever
does not cancel is the source term ``S`` that makes ``u*`` an exact
solution of the forced problem.  Run the node with that source and the
exact boundary data; the difference from ``u*`` is the discretisation
error and nothing else.

What the harness then measures is the **observed order of
convergence**: the rate at which that error falls under refinement,
not its size.  The distinction is the whole point.  A wrong stencil
weight, a mishandled boundary or an off-by-one in a flux normally
leaves the absolute error looking perfectly acceptable on the one grid
a threshold test runs on — it shows up as order 1 where order 2 was
claimed.  Pointwise error tests pass on subtly wrong code; order tests
do not.  Both defects this module found in MADDENING's own nodes sat
comfortably inside the thresholds of the tests that already covered
them (see ``docs/developer_guide/verification.md``).

Typical use, mirroring :func:`~maddening.testing.verification.assert_node_verified`::

    from maddening.testing.mms import (
        ManufacturedSolution, RefinementAxis, assert_node_order_verified,
        diffusion_operator,
    )

    sol = ManufacturedSolution(
        exact=lambda x, t: jnp.sin(2 * jnp.pi * x) + 0.5 * x + 1.0,
        operator=diffusion_operator(alpha),
    )

    def error_at(n_cells):
        ...                       # build the node at this resolution,
        return l2_relative_error  # run it, return one error

    assert_node_order_verified(
        node, axis=RefinementAxis.SPACE, error_at=error_at,
        levels=(10, 20, 40, 80),
    )

Three things the caller is responsible for, because the harness cannot
know them:

**Refine one axis at a time.**  Refining space and time together
measures the *minimum* of the two orders, so a first-order time
integrator hides a second-order stencil.  Hold the other axis fixed, or
make its error vanish identically — for a scheme whose spatial operator
is exact on quadratics, a manufactured solution quadratic in ``x``
leaves only the temporal error, and a solution with no time dependence
run to steady state leaves only the spatial error.  Say in the test
which one is being measured; :class:`RefinementAxis` records it in the
result.

**Stay above the arithmetic noise floor.**  The observed order is a
ratio of small numbers, and float32 runs out of signal long before a
refinement ladder does.  Under float32 the ladders in
``tests/verification/test_mms_order.py`` measure a clean order up to
about 80 cells and then turn over; in float64 the same ladders stay
clean well past that.  :attr:`OrderMeasurement.monotone` is the guard:
an error that stops falling means the ladder has left the asymptotic
range, and the harness reports that as its own failure rather than as a
wrong order.

**Check that the node takes a source at all.**  MMS needs to inject
``S``.  If a node has no forcing input, the finding is that MMS needs a
hook the node does not expose — not that the study should be replaced
by something that does not test the discretisation.

Grid convergence: the fallback for nodes MMS cannot reach
---------------------------------------------------------

MMS can only test a node that has somewhere to put ``S``.  That rules
out :class:`~maddening.nodes.lbm_pipe.LBMPipeNode` (a generic source
term is not even well defined for a lattice Boltzmann collision
operator: the forcing scheme changes the order of accuracy, so the
study would measure the scheme rather than the operator), every node
with no natural forcing input, and most nodes a *user* writes.  The
alternative considered — an optional manufactured-source hook on
:class:`~maddening.core.node.SimulationNode` — was rejected; see
``TODO.md``, "DECIDED AGAINST: a manufactured-source convention on
``SimulationNode``".

The second half of this module is what made rejecting it acceptable.
:func:`measure_gci` runs the *same refinement ladder* as
:func:`measure_order`, but compares the solutions to **each other**
rather than to a manufactured exact field.  Three refinements of one
scalar functional determine the observed order of convergence, the
Richardson-extrapolated limit, and the Grid Convergence Index — an
error band on the finest solution (Richardson 1911; Roache 1994;
Celik et al. 2008).  It needs nothing from the node but the ability to
refine::

    from maddening.testing.mms import (
        RefinementAxis, assert_node_gci_verified,
    )

    def flow_shape_at(n_cells):
        ...                       # build the node at this resolution,
        return u_max / u_mean     # run it, return ONE scalar

    assert_node_gci_verified(
        node, axis=RefinementAxis.SPACE, solution_at=flow_shape_at,
        levels=(16, 24, 32), max_gci=0.60,
    )

What it buys, and what it does not: GCI never sees the exact answer, so
it cannot catch a scheme that converges cleanly to the *wrong* limit.
MMS can.  Where a node has a natural forcing input, prefer MMS and use
GCI as the uncertainty statement on top of it.

Requires ``jax`` and ``numpy``; imported from ``maddening.testing``,
which needs the ``[verify]`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import DiscretizationOrder
from maddening.testing.verification import VerificationResult

__all__ = [
    "RefinementAxis",
    "ManufacturedSolution",
    "OrderMeasurement",
    "UndeclaredOrderError",
    "DEFAULT_ORDER_SHORTFALL",
    "DEFAULT_ORDER_EXCESS",
    "diffusion_operator",
    "manufactured_acceleration",
    "measure_order",
    "check_order",
    "declared_order",
    "verify_node_order",
    "assert_node_order_verified",
    # Grid convergence / Richardson extrapolation
    "ConvergenceRegime",
    "GridConvergenceStudy",
    "InconclusiveStudyError",
    "DEFAULT_SAFETY_FACTOR",
    "CAUTIOUS_SAFETY_FACTOR",
    "DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE",
    "DEFAULT_STAGNATION_RTOL",
    "RECOMMENDED_MIN_REFINEMENT_RATIO",
    "MIN_APPARENT_ORDER",
    "MAX_APPARENT_ORDER",
    "ApparentOrder",
    "apparent_order",
    "richardson_study",
    "measure_gci",
    "check_gci",
    "verify_node_gci",
    "assert_node_gci_verified",
]


class RefinementAxis(Enum):
    """Which axis a convergence study refined.

    Recorded on the measurement so a result can never be read as a
    claim about the other axis.  ``SPACE`` refines the grid spacing
    ``h`` at fixed timestep (or with the temporal error made to vanish);
    ``TIME`` refines ``dt`` on a fixed grid.
    """

    SPACE = "space"
    TIME = "time"

    @property
    def order_attribute(self) -> str:
        """Name of the :class:`DiscretizationOrder` field for this axis.

        Doubles as the adjective used in every message the harness
        writes ("spatial order", "temporal order"), so a result can be
        read without knowing the enum.
        """
        return "spatial" if self is RefinementAxis.SPACE else "temporal"

    @property
    def symbol(self) -> str:
        return "h" if self is RefinementAxis.SPACE else "dt"


class UndeclaredOrderError(Exception):
    """A node was asked for an order it has not declared.

    Raised by :func:`assert_node_order_verified` instead of quietly
    passing.  A test that legitimately cannot measure the node turns
    this into an explicit skip::

        try:
            assert_node_order_verified(node, ...)
        except UndeclaredOrderError as exc:
            pytest.skip(str(exc))
    """


# ---------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------


def diffusion_operator(alpha: Any) -> Callable[[Callable, Any, Any], Any]:
    """Spatial operator of the 1D diffusion equation, ``alpha * d2u/dx2``.

    Parameters
    ----------
    alpha : float or array
        Diffusivity.

    Returns
    -------
    callable
        ``(u, x, t) -> alpha * d2u/dx2`` at ``(x, t)``, where ``u`` is a
        scalar function of two scalars.  Differentiated with
        :func:`jax.grad`, so the second derivative is exact to
        round-off rather than a finite difference of the manufactured
        field — which would put the very discretisation error under
        test into the source term.
    """

    def operator(u: Callable, x: Any, t: Any) -> Any:
        d2u = jax.grad(jax.grad(u, argnums=0), argnums=0)
        return alpha * d2u(x, t)

    return operator


@dataclass(frozen=True)
class ManufacturedSolution:
    """An analytic field plus the source term that makes it exact.

    The author writes only ``exact``; the source is derived by
    automatic differentiation of it through the declared PDE operator,
    which removes the step that is easiest to get wrong by hand.  For
    ``du/dt = L[u] + S``, the source that makes ``exact`` a solution is

    .. math::  S(x, t) = \\partial_t u^*(x, t) - L[u^*](x, t)

    Parameters
    ----------
    exact : callable
        ``(x, t) -> u``, all three scalars.  Must be written in
        ``jax.numpy`` so it can be differentiated and traced.
    operator : callable
        ``(u, x, t) -> L[u](x, t)``, the spatial operator the node
        claims to discretise, taking the *function* ``u`` so it can
        differentiate it.  :func:`diffusion_operator` builds the
        diffusion case.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> sol = ManufacturedSolution(
    ...     exact=lambda x, t: jnp.sin(x),
    ...     operator=diffusion_operator(2.0),
    ... )
    >>> bool(jnp.allclose(sol.source(1.0, 0.0), 2.0 * jnp.sin(1.0)))
    True
    """

    exact: Callable[[Any, Any], Any]
    operator: Callable[[Callable, Any, Any], Any]

    def source(self, x: Any, t: Any) -> Any:
        """``S(x, t)``, the forcing that makes :attr:`exact` a solution."""
        dudt = jax.grad(self.exact, argnums=1)(
            jnp.asarray(x, dtype=jnp.result_type(float)),
            jnp.asarray(t, dtype=jnp.result_type(float)),
        )
        return dudt - self.operator(self.exact, x, t)

    def field(self, xs: Any, t: Any) -> Any:
        """:attr:`exact` evaluated over an array of positions."""
        return jax.vmap(lambda x: self.exact(x, t))(jnp.asarray(xs))

    def source_field(self, xs: Any, t: Any) -> Any:
        """:meth:`source` evaluated over an array of positions."""
        return jax.vmap(lambda x: self.source(x, t))(jnp.asarray(xs))


def manufactured_acceleration(
    trajectory: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    """``d2/dt2`` of a manufactured trajectory, for ODE nodes.

    The ODE analogue of :class:`ManufacturedSolution`: for a body
    obeying ``m x'' = F``, the force that makes ``trajectory`` exact is
    ``m`` times the returned function.

    Parameters
    ----------
    trajectory : callable
        ``t -> x``, scalar ``t``, ``x`` scalar or vector, written in
        ``jax.numpy``.

    Returns
    -------
    callable
        ``t -> d2x/dt2``, by forward-mode automatic differentiation.
    """
    return jax.jacfwd(jax.jacfwd(trajectory))


# ---------------------------------------------------------------------------
# Order measurement
# ---------------------------------------------------------------------------

#: How far below its declared order a node's measurement may fall
#: before :func:`check_order` fails it.
#:
#: Measured, not chosen by intuition.  On the three ladders in
#: ``tests/verification/test_mms_order.py`` the observed order over the
#: finest pair lands within 0.02 of theory (HeatNode 1.982 against 2,
#: LBMNode 1.998 against 2, RigidBodyNode 0.999 against 1), while the
#: *coarsest* pair of the same ladders sits as much as 0.16 low (1.847)
#: because the asymptotic range has not been reached.  0.25 clears that
#: coarse-grid wobble with room to spare and still rejects both defects
#: this harness found, each of which falls a full order or more short —
#: a margin of 4x between what the band admits and what it catches.
DEFAULT_ORDER_SHORTFALL = 0.25

#: How far *above* its declared order a measurement may go before
#: :func:`check_order` fails it.
#:
#: An order well above theory is normally a broken study rather than a
#: pleasant surprise: the classic cause is a manufactured solution that
#: the scheme happens to represent exactly (a quadratic profile under a
#: second-order central difference), which makes the truncation error
#: vanish identically and measures nothing.  1.0 is wide enough for the
#: genuine superconvergence seen here — the corrected fourth-order
#: stencil measures 5.02 over one pair of a fourth-order ladder — while
#: still catching a study that has fallen through into the exact case.
DEFAULT_ORDER_EXCESS = 1.0


@dataclass(frozen=True)
class OrderMeasurement:
    """Errors from a refinement ladder, and the orders they imply.

    Attributes
    ----------
    axis : RefinementAxis
        Which axis was refined.  A measurement carries this so it can
        never be quoted as the other order.
    levels : tuple
        The refinement levels as the caller named them (cell counts,
        step counts, ...), coarsest first.
    h : tuple of float
        The refinement parameter for each level, strictly decreasing.
    errors : tuple of float
        One error per level, in the same order.
    """

    axis: RefinementAxis
    levels: tuple[Any, ...]
    h: tuple[float, ...]
    errors: tuple[float, ...]

    @property
    def pairwise_orders(self) -> tuple[float, ...]:
        """``log(e_i / e_i+1) / log(h_i / h_i+1)`` for each adjacent pair."""
        out = []
        for i in range(len(self.errors) - 1):
            e0, e1 = self.errors[i], self.errors[i + 1]
            h0, h1 = self.h[i], self.h[i + 1]
            if e0 <= 0.0 or e1 <= 0.0:
                out.append(math.nan)
            else:
                out.append(math.log(e0 / e1) / math.log(h0 / h1))
        return tuple(out)

    @property
    def observed(self) -> float:
        """The order over the *finest* pair.

        The asymptotic estimate, and what :func:`check_order` gates on:
        the order converges to theory as ``h -> 0``, so the finest pair
        is the least polluted by the coarse-grid wobble that drags
        :attr:`fitted` down.  It is also the most exposed to arithmetic
        noise, which is why :attr:`monotone` has to hold for the
        measurement to mean anything.
        """
        orders = self.pairwise_orders
        return orders[-1] if orders else math.nan

    @property
    def fitted(self) -> float:
        """Least-squares slope of ``log(error)`` against ``log(h)``.

        Uses every level, so it is steadier than :attr:`observed` and
        biased low by any coarse level outside the asymptotic range.
        Reported alongside, never gated on.
        """
        if len(self.errors) < 2 or any(e <= 0.0 for e in self.errors):
            return math.nan
        x = np.log(np.asarray(self.h, dtype=float))
        y = np.log(np.asarray(self.errors, dtype=float))
        return float(np.polyfit(x, y, 1)[0])

    @property
    def monotone(self) -> bool:
        """Whether the error fell at every refinement.

        A ladder that stops improving has left the asymptotic range —
        almost always because it has reached the arithmetic noise floor
        — and any order read off it is meaningless.
        """
        return all(
            self.errors[i + 1] < self.errors[i]
            for i in range(len(self.errors) - 1)
        )

    def table(self) -> str:
        """A human-readable refinement table for a failure message."""
        lines = [
            f"  {'level':>10}  {self.axis.symbol:>12}  {'error':>12}  {'order':>7}",
        ]
        orders = ("",) + tuple(f"{o:7.3f}" for o in self.pairwise_orders)
        for level, h, e, o in zip(self.levels, self.h, self.errors, orders):
            lines.append(f"  {level!s:>10}  {h:12.6g}  {e:12.6g}  {o:>7}")
        lines.append(
            f"  observed (finest pair) = {self.observed:.3f}, "
            f"least-squares fit = {self.fitted:.3f}"
        )
        return "\n".join(lines)


def measure_order(
    error_at: Callable[[Any], float],
    levels: Sequence[Any],
    *,
    axis: RefinementAxis,
    h_of: Callable[[Any], float] | None = None,
) -> OrderMeasurement:
    """Run a refinement ladder and return the errors and implied orders.

    Parameters
    ----------
    error_at : callable
        ``level -> error``.  One scalar per level; how the node is
        built and driven at that level is the caller's business.  The
        error should be a norm of the difference from the manufactured
        solution, normalised the same way at every level (a *relative*
        norm, so the ladder is not measuring a changing scale).
    levels : sequence
        Refinement levels, coarsest first: cell counts for
        :attr:`RefinementAxis.SPACE`, step counts for
        :attr:`RefinementAxis.TIME`.  At least two; three or more is
        what makes the trend legible.
    axis : RefinementAxis
        Which axis this ladder refines.  Required, and recorded on the
        result: "order 2" means nothing without it.
    h_of : callable, optional
        ``level -> h``.  Defaults to ``1 / level``, which is right for
        both a cell count and a step count.  Pass an explicit mapping
        when the levels are already timesteps or spacings.

    Returns
    -------
    OrderMeasurement

    Raises
    ------
    ValueError
        Fewer than two levels, or ``h`` not strictly decreasing (levels
        given in the wrong order, which would silently flip the sign of
        every measured order).
    """
    levels = tuple(levels)
    if len(levels) < 2:
        raise ValueError(
            f"a convergence study needs at least two refinement levels, got {len(levels)}"
        )
    h_fn = h_of if h_of is not None else (lambda level: 1.0 / float(level))
    h = tuple(float(h_fn(level)) for level in levels)
    if any(h[i + 1] >= h[i] for i in range(len(h) - 1)):
        raise ValueError(
            f"refinement levels must be given coarsest first, so that {axis.symbol} "
            f"strictly decreases; got {axis.symbol} = {h}"
        )
    errors = tuple(float(error_at(level)) for level in levels)
    return OrderMeasurement(axis=axis, levels=levels, h=h, errors=errors)


def check_order(
    measurement: OrderMeasurement,
    expected: float,
    *,
    name: str = "order",
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Judge a measurement against a declared order.

    Parameters
    ----------
    measurement : OrderMeasurement
    expected : float
        The theoretical order being claimed.
    name : str
        Check name carried on the result.
    shortfall, excess : float
        Half-widths of the acceptance band around ``expected``; see
        :data:`DEFAULT_ORDER_SHORTFALL` and :data:`DEFAULT_ORDER_EXCESS`
        for where the defaults come from.

    Returns
    -------
    VerificationResult
        ``PASS`` if :attr:`OrderMeasurement.observed` is inside the
        band, ``FAIL`` otherwise.  A non-monotone ladder fails with its
        own message: the study is inconclusive rather than the node
        being wrong, and the two must not be confused.
    """
    table = measurement.table()
    if not measurement.monotone:
        return VerificationResult(
            name, "FAIL", n_examples=len(measurement.errors),
            detail=(
                f"the {measurement.axis.order_attribute} refinement ladder is not "
                f"converging — the error stopped falling, so no order can be "
                f"read from it.  Usually the finest levels have reached the "
                f"arithmetic noise floor (try float64, or stop the ladder "
                f"earlier); occasionally the run has not reached the state "
                f"the error is measured at.\n{table}"
            ),
        )
    observed = measurement.observed
    if not math.isfinite(observed):
        return VerificationResult(
            name, "FAIL", n_examples=len(measurement.errors),
            detail=(
                f"the observed {measurement.axis.order_attribute} order is not finite "
                f"(an error was zero or negative, so the scheme is exact on "
                f"this manufactured solution and the study measures "
                f"nothing).\n{table}"
            ),
        )
    low, high = expected - shortfall, expected + excess
    if observed < low:
        return VerificationResult(
            name, "FAIL", n_examples=len(measurement.errors),
            detail=(
                f"observed {measurement.axis.order_attribute} order {observed:.3f} is "
                f"below the declared {expected:g} (band [{low:.2f}, "
                f"{high:.2f}]).  A shortfall of a whole order is a wrong "
                f"discretisation — a stencil weight, a boundary closure or "
                f"an off-by-one — not a tolerance to widen.\n{table}"
            ),
        )
    if observed > high:
        return VerificationResult(
            name, "FAIL", n_examples=len(measurement.errors),
            detail=(
                f"observed {measurement.axis.order_attribute} order {observed:.3f} is "
                f"above the declared {expected:g} (band [{low:.2f}, "
                f"{high:.2f}]).  That normally means the study is not "
                f"exercising the scheme: a manufactured solution the "
                f"discretisation represents exactly makes the truncation "
                f"error vanish and measures nothing.\n{table}"
            ),
        )
    return VerificationResult(
        name, "PASS", n_examples=len(measurement.errors),
        detail=f"observed {observed:.3f} against declared {expected:g}\n{table}",
    )


# ---------------------------------------------------------------------------
# Node-level entry points
# ---------------------------------------------------------------------------


def declared_order(node: Any) -> DiscretizationOrder | None:
    """The order a node claims, or ``None`` if it claims none.

    Looks first for an instance hook ``node.discretization_order()`` —
    a node whose order depends on how it was constructed (a selectable
    stencil, say) can only answer per instance — and falls back to the
    class-level :attr:`NodeMeta.discretization_order
    <maddening.core.compliance.metadata.NodeMeta.discretization_order>`.

    Parameters
    ----------
    node : SimulationNode

    Returns
    -------
    DiscretizationOrder or None
    """
    hook = getattr(node, "discretization_order", None)
    if callable(hook):
        return hook()
    meta = getattr(node, "meta", None)
    return getattr(meta, "discretization_order", None) if meta is not None else None


def _undeclared_detail(node: Any, axis: RefinementAxis) -> str:
    return (
        f"{type(node).__name__} declares no {axis.order_attribute} order of "
        f"accuracy, so there is nothing to measure against.  Set "
        f"NodeMeta(discretization_order=DiscretizationOrder("
        f"{axis.order_attribute}=...)) on the class, or override "
        f"discretization_order() on the instance when the order depends on "
        f"how the node was constructed."
    )


def verify_node_order(
    node: Any,
    *,
    axis: RefinementAxis,
    error_at: Callable[[Any], float],
    levels: Sequence[Any],
    h_of: Callable[[Any], float] | None = None,
    expected: float | None = None,
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Measure one node's order of convergence and judge it.

    The order-of-accuracy counterpart to
    :func:`~maddening.testing.verification.verify_node`, and it returns
    the same :class:`~maddening.testing.verification.VerificationResult`
    so the two can be reported together.

    Parameters
    ----------
    node : SimulationNode
        Only its declared order is read; ``error_at`` decides how the
        node is actually built and driven at each level (a spatial
        ladder has to construct a new node per resolution).
    axis : RefinementAxis
        Which axis ``error_at`` refines.
    error_at, levels, h_of
        As for :func:`measure_order`.
    expected : float, optional
        Override the declared order.  Leave it unset in a node's own
        test: taking the claim from the node is what makes a wrong
        claim visible.
    shortfall, excess : float
        Acceptance band; see :data:`DEFAULT_ORDER_SHORTFALL`.

    Returns
    -------
    VerificationResult
        ``SKIP`` — explicitly, never a silent pass — when the node
        declares no order for this axis and ``expected`` was not given.
    """
    check_name = f"{axis.order_attribute}_order"
    if expected is None:
        declared = declared_order(node)
        value = (
            getattr(declared, axis.order_attribute) if declared is not None else None
        )
        if value is None:
            return VerificationResult(
                check_name, "SKIP", detail=_undeclared_detail(node, axis),
            )
        expected = float(value)
    measurement = measure_order(error_at, levels, axis=axis, h_of=h_of)
    return check_order(
        measurement, expected, name=check_name,
        shortfall=shortfall, excess=excess,
    )


def assert_node_order_verified(node: Any, **kwargs: Any) -> None:
    """:func:`verify_node_order` that raises on a shortfall.

    A one-line pytest body, matching
    :func:`~maddening.testing.verification.assert_node_verified`::

        def test_heat_is_second_order_in_space():
            assert_node_order_verified(
                node, axis=RefinementAxis.SPACE,
                error_at=error_at, levels=(10, 20, 40, 80),
            )

    Raises
    ------
    AssertionError
        The measured order is outside the band, or the ladder did not
        converge.  The message carries the whole refinement table.
    UndeclaredOrderError
        The node declares no order for this axis.  Raised rather than
        passed over, so a node that has never declared one cannot look
        verified; a test that means to tolerate that catches this and
        skips.
    """
    result = verify_node_order(node, **kwargs)
    if result.skipped:
        raise UndeclaredOrderError(result.detail)
    if not result.passed:
        raise AssertionError(
            f"{type(node).__name__} failed the {result.name} check:\n{result.detail}"
        )


# ---------------------------------------------------------------------------
# Grid convergence / Richardson extrapolation
# ---------------------------------------------------------------------------

#: Safety factor for a three-or-more-grid study that is demonstrably in
#: the asymptotic range.
#:
#: Roache (1994, §4; 1998) proposes the Grid Convergence Index as an
#: error band with a deliberate margin over the Richardson estimate, and
#: recommends ``Fs = 1.25`` when the observed order has been *measured*
#: from three or more grids and agrees with the formal order, against
#: ``Fs = 3.0`` when it has been assumed.  The ASME V&V 20 procedure
#: (Celik et al. 2008, §3) uses 1.25 throughout its three-grid recipe.
DEFAULT_SAFETY_FACTOR = 1.25

#: Safety factor for every other case: two grids, or a three-grid study
#: whose asymptotic range could not be demonstrated.
#:
#: Roache's own conditions for 1.25 are not met when the apparent order
#: cannot be checked against anything, so this module falls back to 3.0
#: rather than quoting the narrower band on trust.  A node that declares
#: no order therefore gets a wider, honest band instead of a confident,
#: unsupported one.
CAUTIOUS_SAFETY_FACTOR = 3.0

#: How far the apparent order may sit from the formal order before the
#: solutions are declared *not* to be in the asymptotic range.
#:
#: The same 0.25 as :data:`DEFAULT_ORDER_SHORTFALL`, applied
#: symmetrically: Celik et al. (2008, §3) note that an apparent order
#: well *above* the formal order is as much a sign of a study outside
#: the asymptotic range as one below it — it usually means the coarse
#: grid is so far off that the differences are not yet following a
#: single power law.
DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE = 0.25

#: Relative size below which a solution difference counts as no
#: difference at all.
#:
#: The failure this exists for: a node that ignores its refinement
#: parameter returns the *same* number three times, and every formula
#: below then divides by zero and produces ``nan`` or, worse, a
#: plausible-looking order from round-off.  ``1e-12`` is about
#: ``1e4 * eps`` in float64 — far below any real discretisation
#: difference, far above the round-off of a well-conditioned reduction.
#: **Raise it for a float32 study**, where the noise floor is ``1e-7``
#: relative rather than ``1e-16``.
DEFAULT_STAGNATION_RTOL = 1e-12

#: Bounds of the interval the apparent-order equation is solved on.
#:
#: Below :data:`MIN_APPARENT_ORDER` a study is not converging in any
#: useful sense; above :data:`MAX_APPARENT_ORDER` the exponentials in
#: the implicit equation overflow long before the answer means
#: anything.  A root outside the interval is reported as an
#: inconclusive study, never clipped to the edge.
MIN_APPARENT_ORDER = 1e-3
MAX_APPARENT_ORDER = 40.0

#: Grid refinement ratio below which Celik et al. (2008, §2) advise
#: against reading a three-grid study at all: the differences become
#: comparable to the round-off and iterative error, and the apparent
#: order turns erratic.  Reported as an advisory on the study, not
#: gated on — there are problems where a finer ladder is unaffordable.
RECOMMENDED_MIN_REFINEMENT_RATIO = 1.3


class InconclusiveStudyError(Exception):
    """A grid convergence study could not reach a verdict.

    Raised by :func:`assert_node_gci_verified` where
    :func:`assert_node_order_verified` raises
    :exc:`UndeclaredOrderError`.  The two are deliberately distinct: an
    undeclared node is a gap in the *node's* metadata, while this is a
    gap in the *study* — the solutions oscillated, diverged, or did not
    move at all — and neither may be mistaken for a pass.
    """


class ConvergenceRegime(Enum):
    """How a triple of successively refined solutions behaves.

    Classified from the convergence ratio ``R = eps_fine /
    eps_coarse``, where ``eps_fine`` is the change over the finest pair
    of solutions and ``eps_coarse`` the change over the pair before it
    (Stern et al. 2001, §3.2).  The *sign* of ``R`` says whether the
    solutions approach a limit from one side or straddle it; the
    *magnitude* says whether they are getting closer to it at all.

    Only :attr:`MONOTONE` admits a Richardson extrapolation or a Grid
    Convergence Index.  Every other member is a study that has not
    reached a verdict, and the harness reports it as such rather than
    averaging it into an order that would look like one.
    """

    #: ``0 < R < 1``.  The solutions approach a limit from one side.
    MONOTONE = "monotone"
    #: ``-1 < R < 0``.  The solutions straddle a limit, and the
    #: oscillation is shrinking.  A limit may well exist, but the
    #: three-grid power-law model does not describe the approach to it,
    #: so no order can be read off.
    OSCILLATORY = "oscillatory"
    #: ``R >= 1``.  Successive refinements change the answer by at
    #: least as much as the refinement before, from one side.
    DIVERGENT = "divergent"
    #: ``R <= -1``.  Straddling a limit and getting worse.
    OSCILLATORY_DIVERGENT = "oscillatory-divergent"
    #: A difference vanished.  The usual cause is a node that ignored
    #: the refinement and returned the identical solution at every
    #: level, which every formula here would otherwise divide by.
    STAGNANT = "stagnant"
    #: A solution was not finite.
    INVALID = "invalid"

    @property
    def supports_extrapolation(self) -> bool:
        """Whether Richardson extrapolation and a GCI are defined here."""
        return self is ConvergenceRegime.MONOTONE

    @property
    def description(self) -> str:
        """One clause naming the regime, for a message."""
        return {
            ConvergenceRegime.MONOTONE: "monotone convergence",
            ConvergenceRegime.OSCILLATORY: "oscillatory convergence",
            ConvergenceRegime.DIVERGENT: "monotone divergence",
            ConvergenceRegime.OSCILLATORY_DIVERGENT: "oscillatory divergence",
            ConvergenceRegime.STAGNANT: "no change under refinement",
            ConvergenceRegime.INVALID: "a non-finite solution",
        }[self]


def _ln_pow_minus_s(r: float, p: float, s: float) -> float:
    """``log(r**p - s)`` without overflowing for large ``p``.

    ``r**p`` overflows a float64 at ``p * log(r) > 709``, which the
    root search below reaches routinely on a badly conditioned pair of
    refinement ratios.  Factoring the exponential out first —
    ``log(r**p - s) = p log r + log1p(-s exp(-p log r))`` — keeps every
    intermediate in range and stays accurate as ``p -> 0``, where the
    naive form loses the whole answer to cancellation.
    """
    x = p * math.log(r)
    if x <= 0.0:
        return -math.inf if s > 0 else math.log(2.0) if x == 0.0 else math.nan
    inner = -s * math.exp(-x)
    if inner <= -1.0:
        return -math.inf
    return x + math.log1p(inner)


def _order_q(p: float, r_fine: float, r_coarse: float, s: float) -> float:
    """``q(p) = log((r_fine**p - s) / (r_coarse**p - s))``, overflow-free."""
    a = _ln_pow_minus_s(r_fine, p, s)
    b = _ln_pow_minus_s(r_coarse, p, s)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return a - b


def _order_residual(
    p: float, ln_ratio: float, r_fine: float, r_coarse: float, s: float,
) -> float:
    """``p log(r_fine) - log|eps_c/eps_f| - q(p)``; zero at the answer.

    The *signed* form of the implicit equation, without the absolute
    value the fixed-point form carries.  That matters: the ``|...|``
    folds the function about its axis and manufactures spurious roots
    that a bracketing search would otherwise return.
    """
    q = _order_q(p, r_fine, r_coarse, s)
    if not math.isfinite(q):
        return math.nan
    return p * math.log(r_fine) - ln_ratio - q


@dataclass(frozen=True)
class ApparentOrder:
    """The solution of the observed-order equation, and how it was got.

    Attributes
    ----------
    value : float
        The observed (apparent) order of convergence, or ``nan`` when
        the equation had no solution in
        ``[MIN_APPARENT_ORDER, MAX_APPARENT_ORDER]``.
    method : str
        ``"closed-form"`` (constant refinement ratio), ``"fixed-point"``
        (Celik et al.'s iteration converged), ``"bracketed"`` (it did
        not, and a bisection on the signed residual did), or
        ``"no-solution"``.
    iterations : int
    detail : str
        Empty on success; why there is no answer otherwise.
    """

    value: float
    method: str
    iterations: int = 0
    detail: str = ""


def apparent_order(
    eps_fine: float,
    eps_coarse: float,
    r_fine: float,
    r_coarse: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ApparentOrder:
    """Observed order of convergence from three solutions, no exact answer.

    Implements the apparent-order equation of the ASME V&V 20 procedure
    (Celik et al. 2008, eqs. 2-3), which for solutions ``phi_1``
    (finest), ``phi_2``, ``phi_3`` on grids ``h_1 < h_2 < h_3`` reads

    .. math::

        p = \\frac{1}{\\ln r_{21}}
            \\left| \\ln\\left|\\frac{\\epsilon_{32}}{\\epsilon_{21}}\\right|
                    + q(p) \\right|,
        \\qquad
        q(p) = \\ln \\frac{r_{21}^{p} - s}{r_{32}^{p} - s},
        \\qquad
        s = \\operatorname{sgn}
            \\left(\\frac{\\epsilon_{32}}{\\epsilon_{21}}\\right)

    with ``eps_21 = phi_2 - phi_1``, ``eps_32 = phi_3 - phi_2``,
    ``r_21 = h_2/h_1`` and ``r_32 = h_3/h_2``.

    **Non-integer and non-constant refinement ratios are the reason
    this is a solve and not a formula.**  When ``r_21 == r_32`` the
    ``q`` term vanishes identically and the order is the textbook
    ``log(eps_32/eps_21) / log(r)``.  When the two ratios differ — a
    ladder of 16, 24, 32 cells, say — ``q`` depends on ``p`` and the
    equation is implicit.  Assuming a constant ratio anyway is the
    classic way to get a GCI wrong, so this function never does: it
    detects the constant case, and otherwise solves.

    Two solvers, in order.  Celik et al.'s fixed-point iteration is
    tried first, because it is the published procedure and converges in
    a handful of steps on well-conditioned data.  Its iteration map has
    derivative ``q'(p) / ln r_21``, which exceeds 1 in magnitude for
    ratio pairs that are far apart — ``r_21 = 1.4`` against
    ``r_32 = 2.7`` diverges to an overflow within twenty steps — so
    when it fails, a bisection on :func:`_order_residual` takes over.
    The residual is scanned for sign changes across the whole admissible
    interval first, rather than assuming the endpoints bracket a root:
    for that same pair the residual is negative at *both* ends and the
    root sits in between.

    Parameters
    ----------
    eps_fine : float
        ``phi_2 - phi_1``: the change over the finest pair.  Must not
        be zero; the caller is expected to have classified a stagnant
        study before getting here.
    eps_coarse : float
        ``phi_3 - phi_2``: the change over the next pair out.
    r_fine, r_coarse : float
        ``h_2/h_1`` and ``h_3/h_2``, both strictly greater than 1.

    Returns
    -------
    ApparentOrder

    Raises
    ------
    ValueError
        A refinement ratio was not greater than 1, or a difference was
        zero or non-finite.

    Examples
    --------
    A clean second-order ladder, refined by a constant factor of 2:

    >>> phi = [1.0 + 0.5 * h**2 for h in (0.025, 0.05, 0.1)]
    >>> round(apparent_order(phi[1] - phi[0], phi[2] - phi[1], 2.0, 2.0).value, 6)
    2.0

    The same order recovered from a ladder whose ratio is neither
    constant nor an integer:

    >>> h = [0.1 / (1.37 * 2.11), 0.1 / 1.37, 0.1]
    >>> phi = [1.0 + 0.5 * x**2 for x in h]
    >>> got = apparent_order(phi[1] - phi[0], phi[2] - phi[1], 2.11, 1.37)
    >>> round(got.value, 6), got.method
    (2.0, 'fixed-point')
    """
    if not (r_fine > 1.0 and r_coarse > 1.0):
        raise ValueError(
            f"refinement ratios must exceed 1 (coarsest first), got "
            f"r_fine={r_fine!r}, r_coarse={r_coarse!r}"
        )
    if not (math.isfinite(eps_fine) and math.isfinite(eps_coarse)):
        raise ValueError(
            f"solution differences must be finite, got eps_fine={eps_fine!r}, "
            f"eps_coarse={eps_coarse!r}"
        )
    if eps_fine == 0.0:
        raise ValueError(
            "the two finest solutions are identical, so no order is defined; "
            "classify the study with richardson_study() first"
        )

    ratio = eps_coarse / eps_fine
    if ratio == 0.0:
        raise ValueError(
            "the two coarsest solutions are identical, so no order is defined; "
            "classify the study with richardson_study() first"
        )
    ln_ratio = math.log(abs(ratio))
    ln_r_fine = math.log(r_fine)
    s = math.copysign(1.0, ratio)

    # Constant ratio: q vanishes and the equation is explicit.
    if abs(r_fine - r_coarse) <= 1e-12 * r_fine:
        value = abs(ln_ratio) / ln_r_fine
        if MIN_APPARENT_ORDER <= value <= MAX_APPARENT_ORDER:
            return ApparentOrder(value, "closed-form")
        return ApparentOrder(
            math.nan, "no-solution",
            detail=(
                f"the observed order {value:.4g} is outside "
                f"[{MIN_APPARENT_ORDER:g}, {MAX_APPARENT_ORDER:g}]"
            ),
        )

    # Celik et al.'s fixed-point iteration.
    p = min(max(abs(ln_ratio) / ln_r_fine, MIN_APPARENT_ORDER), MAX_APPARENT_ORDER)
    for k in range(1, max_iter + 1):
        q = _order_q(p, r_fine, r_coarse, s)
        if not math.isfinite(q):
            break
        nxt = abs(ln_ratio + q) / ln_r_fine
        if not math.isfinite(nxt) or not (
            MIN_APPARENT_ORDER <= nxt <= MAX_APPARENT_ORDER
        ):
            break
        if abs(nxt - p) <= tol * max(1.0, nxt):
            return ApparentOrder(nxt, "fixed-point", iterations=k)
        p = nxt

    # Bracketed bisection on the signed residual.  Scanning for sign
    # changes rather than trusting the endpoints: the residual is
    # negative at both ends for some ratio pairs and still has a root.
    n_scan = 257
    grid = [
        MIN_APPARENT_ORDER
        * (MAX_APPARENT_ORDER / MIN_APPARENT_ORDER) ** (i / (n_scan - 1))
        for i in range(n_scan)
    ]
    residuals = [_order_residual(x, ln_ratio, r_fine, r_coarse, s) for x in grid]
    brackets = [
        (grid[i], grid[i + 1])
        for i in range(n_scan - 1)
        if math.isfinite(residuals[i])
        and math.isfinite(residuals[i + 1])
        and residuals[i] * residuals[i + 1] <= 0.0
    ]
    if not brackets:
        return ApparentOrder(
            math.nan, "no-solution",
            detail=(
                f"the apparent-order equation has no solution in "
                f"[{MIN_APPARENT_ORDER:g}, {MAX_APPARENT_ORDER:g}] for "
                f"r_fine={r_fine:g}, r_coarse={r_coarse:g}.  Refinement "
                f"ratios this far apart do not determine an order from three "
                f"grids; use a constant ratio, or add a level."
            ),
        )
    if len(brackets) > 1:
        # Fail closed.  No configuration in the sweep of
        # ``test_the_order_equation_has_at_most_one_root_over_the_ratios_swept``
        # reaches this -- roughly 3,600 ratio pairs from 1.02 to 30 at both
        # signs of s -- so it is a guard against a formulation change rather
        # than an observed case, and that test is what would notice if the
        # case started arising.  Returning the first root would be a
        # plausible-looking number for an equation that does not determine
        # one, which is the failure this whole module exists to avoid.
        return ApparentOrder(
            math.nan, "no-solution",
            detail=(
                f"the apparent-order equation has {len(brackets)} solutions in "
                f"[{MIN_APPARENT_ORDER:g}, {MAX_APPARENT_ORDER:g}] for "
                f"r_fine={r_fine:g}, r_coarse={r_coarse:g}, so the observed "
                f"order is not determined by these three grids"
            ),
        )
    lo, hi = brackets[0]
    f_lo = _order_residual(lo, ln_ratio, r_fine, r_coarse, s)
    iterations = 0
    for iterations in range(1, 201):
        mid = 0.5 * (lo + hi)
        f_mid = _order_residual(mid, ln_ratio, r_fine, r_coarse, s)
        if not math.isfinite(f_mid):
            return ApparentOrder(
                math.nan, "no-solution", iterations=iterations,
                detail="the apparent-order residual went non-finite mid-solve",
            )
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo <= 1e-14 * max(1.0, hi):
            break
    return ApparentOrder(0.5 * (lo + hi), "bracketed", iterations=iterations)


@dataclass(frozen=True)
class GridConvergenceStudy:
    """A refinement ladder judged against itself, with no exact answer.

    The output of :func:`richardson_study` and :func:`measure_gci`.
    Every derived quantity is computed once, at construction, over the
    **finest three** levels of the ladder — the standard three-grid
    procedure of Celik et al. (2008) — while :attr:`triplet_orders`
    keeps the apparent order of every consecutive triple so a longer
    ladder can show whether the order is settling.

    All relative quantities (:attr:`gci_fine`, :attr:`gci_coarse`,
    :attr:`extrapolated_relative_error`) are **fractions, not
    percentages**: ``0.03`` is 3%.  The literature quotes GCI as a
    percentage; :meth:`table` prints it that way and the stored value
    never is, so a comparison against a tolerance cannot be out by 100.

    Attributes
    ----------
    axis : RefinementAxis
        Which axis was refined.  Carried so a result can never be
        quoted as the other one.
    levels : tuple
        Refinement levels as the caller named them, coarsest first.
    h : tuple of float
        Refinement parameter per level, strictly decreasing.
    values : tuple of float
        The scalar functional at each level, in the same order.
    regime : ConvergenceRegime
        How the finest three solutions behave.  Everything below is
        ``nan`` unless this is :attr:`~ConvergenceRegime.MONOTONE`.
    ratio : float
        The convergence ratio ``R = eps_fine / eps_coarse`` the regime
        was read from.  Signed, and meaningful in every regime.
    order : float
        Observed (apparent) order of convergence.
    order_method : str
        How :func:`apparent_order` got it.
    extrapolated : float
        The Richardson-extrapolated limit ``phi_ext`` — the value the
        ladder is heading for, accurate to one order beyond the finest
        solution.
    gci_fine : float
        Grid Convergence Index on the finest solution: the error band,
        as a fraction of that solution.
    gci_coarse : float
        The same on the middle solution, used by the asymptotic check.
    safety_factor : float
        The ``Fs`` actually applied.
    formal_order : float or None
        The order the node claims, if one was supplied.
    in_asymptotic_range : bool or None
        Whether the solutions are demonstrably in the asymptotic range.
        ``None`` means it could not be decided — see
        :attr:`asymptotic_detail`.  A GCI quoted outside the asymptotic
        range is not wrong, but it is not the 1.25-factor band either,
        which is why this drives :attr:`safety_factor`.
    asymptotic_ratio : float
        Roache's ``GCI_coarse / (r_fine**p * GCI_fine)``, which should
        be near 1.  **Read the caveat on**
        :attr:`asymptotic_detail`: for a three-grid study at a constant
        refinement ratio this quantity reduces algebraically to
        ``|phi_fine| / |phi_medium|`` and is therefore near 1 whatever
        the solutions do.  It is reported because the literature quotes
        it, and it is not what :attr:`in_asymptotic_range` is decided
        on.
    asymptotic_detail : str
        Why :attr:`in_asymptotic_range` came out as it did.
    triplet_orders : tuple of float
        Apparent order over each consecutive triple, coarsest first.
        One entry for a three-level ladder.
    detail : str
        Empty for a clean monotone study; what went wrong otherwise.
    """

    axis: RefinementAxis
    levels: tuple[Any, ...]
    h: tuple[float, ...]
    values: tuple[float, ...]
    regime: ConvergenceRegime
    ratio: float
    order: float
    order_method: str
    extrapolated: float
    gci_fine: float
    gci_coarse: float
    safety_factor: float
    formal_order: float | None
    in_asymptotic_range: bool | None
    asymptotic_ratio: float
    asymptotic_detail: str
    triplet_orders: tuple[float, ...]
    detail: str

    @property
    def refinement_ratios(self) -> tuple[float, ...]:
        """``h_i / h_i+1`` for each adjacent pair, coarsest pair first."""
        return tuple(
            self.h[i] / self.h[i + 1] for i in range(len(self.h) - 1)
        )

    @property
    def differences(self) -> tuple[float, ...]:
        """``phi_i - phi_i+1``: coarser minus finer, for each pair."""
        return tuple(
            self.values[i] - self.values[i + 1]
            for i in range(len(self.values) - 1)
        )

    @property
    def constant_ratio(self) -> bool:
        """Whether every refinement used the same ratio."""
        ratios = self.refinement_ratios
        return all(
            abs(r - ratios[0]) <= 1e-12 * ratios[0] for r in ratios
        )

    @property
    def approximate_relative_error(self) -> float:
        """``|(phi_fine - phi_medium) / phi_fine|``, the raw fine-pair change.

        The quantity the GCI scales.  Reported alongside because the
        gap between the two is exactly the correction the extrapolation
        makes: quoting this on its own as an error estimate is the
        mistake the GCI exists to prevent.
        """
        fine = self.values[-1]
        if fine == 0.0:
            return math.nan
        return abs((fine - self.values[-2]) / fine)

    @property
    def extrapolated_relative_error(self) -> float:
        """``|(phi_ext - phi_fine) / phi_ext|``: distance to the limit."""
        if not math.isfinite(self.extrapolated) or self.extrapolated == 0.0:
            return math.nan
        return abs((self.extrapolated - self.values[-1]) / self.extrapolated)

    @property
    def band(self) -> tuple[float, float]:
        """``(low, high)`` absolute bracket the GCI puts on the finest value."""
        if not math.isfinite(self.gci_fine):
            return (math.nan, math.nan)
        width = self.gci_fine * abs(self.values[-1])
        return (self.values[-1] - width, self.values[-1] + width)

    @property
    def ratios_adequate(self) -> bool:
        """Whether every refinement ratio clears the advised minimum.

        Celik et al. (2008, §2) advise ``r >= 1.3``; below that the
        solution differences approach the round-off and iterative
        error and the apparent order turns erratic.  Advisory: this is
        reported, never gated on, because there are problems where a
        coarser ladder is all that is affordable.
        """
        return all(r >= RECOMMENDED_MIN_REFINEMENT_RATIO
                   for r in self.refinement_ratios)

    def table(self) -> str:
        """A human-readable convergence table for a message."""
        symbol = self.axis.symbol
        lines = [
            f"  {'level':>10}  {symbol:>12}  {'value':>14}  {'change':>12}",
        ]
        changes = ("",) + tuple(f"{d:12.6g}" for d in self.differences)
        for level, h, v, d in zip(self.levels, self.h, self.values, changes):
            lines.append(f"  {level!s:>10}  {h:12.6g}  {v:14.8g}  {d:>12}")
        ratios = ", ".join(f"{r:g}" for r in self.refinement_ratios)
        lines.append(
            f"  refinement ratios = {ratios}"
            f"{'' if self.constant_ratio else ' (non-constant)'}"
            f"{'' if self.ratios_adequate else ' [below the advised 1.3]'}"
        )
        lines.append(
            f"  R = {self.ratio:.6g}  ->  {self.regime.description}"
        )
        if self.regime.supports_extrapolation:
            lines.append(
                f"  observed order p = {self.order:.4f} ({self.order_method})"
                + (
                    f", declared {self.formal_order:g}"
                    if self.formal_order is not None else ""
                )
            )
            if len(self.triplet_orders) > 1:
                per = ", ".join(f"{o:.3f}" for o in self.triplet_orders)
                lines.append(f"  per-triple orders (coarsest first) = {per}")
            lines.append(
                f"  Richardson limit = {self.extrapolated:.8g}  "
                f"(finest solution {self.values[-1]:.8g})"
            )
            low, high = self.band
            lines.append(
                f"  GCI(fine) = {self.gci_fine * 100:.3f}% with Fs = "
                f"{self.safety_factor:g}  ->  [{low:.8g}, {high:.8g}]"
            )
            lines.append(f"  asymptotic range: {self.asymptotic_detail}")
        else:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


def _classify(
    eps_fine: float, eps_coarse: float, values: Sequence[float], rtol: float,
) -> tuple[ConvergenceRegime, float, str]:
    """Regime, convergence ratio R, and a message, from two differences."""
    if not all(math.isfinite(v) for v in values):
        return (
            ConvergenceRegime.INVALID, math.nan,
            "a solution was not finite, so nothing can be read from the "
            "ladder; the run diverged or produced nan before the functional "
            "was taken",
        )
    scale = max(abs(v) for v in values[-3:])
    floor = rtol * scale
    stagnant_fine = abs(eps_fine) <= floor
    stagnant_coarse = abs(eps_coarse) <= floor
    if stagnant_fine or stagnant_coarse:
        which = (
            "every solution is" if stagnant_fine and stagnant_coarse
            else "the two finest solutions are" if stagnant_fine
            else "the two coarsest solutions are"
        )
        return (
            ConvergenceRegime.STAGNANT, math.nan,
            f"{which} identical to within {rtol:g} relative — the ladder did "
            f"not respond to refinement, so there is no convergence to index. "
            f"The usual cause is a node, or a driver, that ignores the "
            f"refinement parameter: check that the level actually reaches the "
            f"node.  (A genuine study at the arithmetic noise floor looks the "
            f"same; raise the precision, or widen the ladder.)",
        )
    ratio = eps_fine / eps_coarse
    if 0.0 < ratio < 1.0:
        return ConvergenceRegime.MONOTONE, ratio, ""
    if -1.0 < ratio < 0.0:
        return (
            ConvergenceRegime.OSCILLATORY, ratio,
            f"the solution differences change sign (R = {ratio:.4g}), so the "
            f"three solutions straddle their limit instead of approaching it "
            f"from one side.  The oscillation is shrinking, so a limit "
            f"plausibly exists, but the single power law a Richardson "
            f"extrapolation assumes does not describe this ladder and no "
            f"order may be read from it.  Refine further, or widen the "
            f"refinement ratio, until the sign settles",
        )
    if ratio >= 1.0:
        return (
            ConvergenceRegime.DIVERGENT, ratio,
            f"the solutions are diverging (R = {ratio:.4g} >= 1): refining "
            f"changed the answer by at least as much as the refinement "
            f"before it, so there is no limit to extrapolate to and no error "
            f"band to quote",
        )
    return (
        ConvergenceRegime.OSCILLATORY_DIVERGENT, ratio,
        f"the solutions are oscillating and diverging (R = {ratio:.4g} <= -1): "
        f"each refinement overshoots the last by more than the last "
        f"overshot.  Nothing can be extrapolated from this",
    )


def richardson_study(
    values: Sequence[float],
    h: Sequence[float],
    *,
    axis: RefinementAxis,
    levels: Sequence[Any] | None = None,
    formal_order: float | None = None,
    safety_factor: float | None = None,
    stagnation_rtol: float = DEFAULT_STAGNATION_RTOL,
    asymptotic_tolerance: float = DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
) -> GridConvergenceStudy:
    """Classify a refinement ladder and index its convergence.

    The arithmetic of the whole GCI mode, separated from the running of
    the ladder so it can be exercised on solutions the harness never
    produced — which is the only way to test that it *rejects* what it
    should.

    Parameters
    ----------
    values : sequence of float
        One scalar functional per level, coarsest first.  At least
        three: two solutions determine a difference but not a rate, and
        the point of this mode is that the rate is measured rather than
        assumed.
    h : sequence of float
        Refinement parameter per level, strictly decreasing.
    axis : RefinementAxis
        Which axis the ladder refined.
    levels : sequence, optional
        The levels as the caller named them, for the table.  Defaults
        to ``h``.
    formal_order : float, optional
        The order the scheme claims.  Supplying it is what lets the
        asymptotic range be decided at all from three grids; see
        :attr:`GridConvergenceStudy.in_asymptotic_range`.
    safety_factor : float, optional
        Override ``Fs``.  The default chooses
        :data:`DEFAULT_SAFETY_FACTOR` (1.25) when the study is in the
        asymptotic range and :data:`CAUTIOUS_SAFETY_FACTOR` (3.0)
        otherwise, following Roache (1994, 1998).
    stagnation_rtol : float
        See :data:`DEFAULT_STAGNATION_RTOL`.
    asymptotic_tolerance : float
        See :data:`DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE`.

    Returns
    -------
    GridConvergenceStudy

    Raises
    ------
    ValueError
        Fewer than three levels, mismatched lengths, or ``h`` not
        strictly decreasing.

    Examples
    --------
    >>> hs = [0.1, 0.05, 0.025]
    >>> vals = [1.0 + 0.5 * x**2 for x in hs]
    >>> study = richardson_study(vals, hs, axis=RefinementAxis.SPACE,
    ...                          formal_order=2.0)
    >>> study.regime is ConvergenceRegime.MONOTONE
    True
    >>> round(study.order, 6), round(study.extrapolated, 9)
    (2.0, 1.0)
    """
    values = tuple(float(v) for v in values)
    h = tuple(float(x) for x in h)
    if len(values) != len(h):
        raise ValueError(
            f"got {len(values)} solutions for {len(h)} refinement levels"
        )
    if len(values) < 3:
        raise ValueError(
            f"a grid convergence study needs at least three refinement "
            f"levels — the observed order is measured from the differences, "
            f"not assumed — got {len(values)}"
        )
    if any(h[i + 1] >= h[i] for i in range(len(h) - 1)):
        raise ValueError(
            f"refinement levels must be given coarsest first, so that "
            f"{axis.symbol} strictly decreases; got {axis.symbol} = {h}"
        )
    levels = tuple(levels) if levels is not None else h

    eps_coarse = values[-3] - values[-2]
    eps_fine = values[-2] - values[-1]
    regime, ratio, detail = _classify(
        eps_fine, eps_coarse, values, stagnation_rtol,
    )

    nan = math.nan
    if not regime.supports_extrapolation:
        return GridConvergenceStudy(
            axis=axis, levels=levels, h=h, values=values, regime=regime,
            ratio=ratio, order=nan, order_method="not-attempted",
            extrapolated=nan, gci_fine=nan, gci_coarse=nan,
            safety_factor=(
                safety_factor if safety_factor is not None
                else CAUTIOUS_SAFETY_FACTOR
            ),
            formal_order=formal_order, in_asymptotic_range=False,
            asymptotic_ratio=nan,
            asymptotic_detail=f"no — {regime.description}",
            triplet_orders=(), detail=detail,
        )

    r_fine = h[-2] / h[-1]
    r_coarse = h[-3] / h[-2]
    solved = apparent_order(eps_fine, eps_coarse, r_fine, r_coarse)
    if not math.isfinite(solved.value):
        return GridConvergenceStudy(
            axis=axis, levels=levels, h=h, values=values, regime=regime,
            ratio=ratio, order=nan, order_method=solved.method,
            extrapolated=nan, gci_fine=nan, gci_coarse=nan,
            safety_factor=(
                safety_factor if safety_factor is not None
                else CAUTIOUS_SAFETY_FACTOR
            ),
            formal_order=formal_order, in_asymptotic_range=False,
            asymptotic_ratio=nan,
            asymptotic_detail="no — the observed order is not determined",
            triplet_orders=(), detail=solved.detail,
        )

    p = solved.value
    triplet_orders = []
    for i in range(len(values) - 2):
        ec = values[i] - values[i + 1]
        ef = values[i + 1] - values[i + 2]
        # Only a monotonically converging triple has an order at all; a
        # sub-triple that oscillates or diverges contributes ``nan``,
        # which _assess_asymptotic_range reads as evidence against the
        # asymptotic range rather than dropping from the comparison.
        if ef == 0.0 or ec == 0.0 or not 0.0 < ef / ec < 1.0:
            triplet_orders.append(nan)
            continue
        got = apparent_order(ef, ec, h[i + 1] / h[i + 2], h[i] / h[i + 1])
        triplet_orders.append(got.value)

    rp_fine = r_fine**p
    rp_coarse = r_coarse**p
    extrapolated = (rp_fine * values[-1] - values[-2]) / (rp_fine - 1.0)

    in_range, asym_detail = _assess_asymptotic_range(
        p, formal_order, tuple(triplet_orders), asymptotic_tolerance,
    )
    fs = safety_factor
    if fs is None:
        fs = DEFAULT_SAFETY_FACTOR if in_range else CAUTIOUS_SAFETY_FACTOR

    e_fine = abs(eps_fine / values[-1]) if values[-1] != 0.0 else nan
    e_coarse = abs(eps_coarse / values[-2]) if values[-2] != 0.0 else nan
    gci_fine = fs * e_fine / (rp_fine - 1.0)
    gci_coarse = fs * e_coarse / (rp_coarse - 1.0)
    asymptotic_ratio = (
        gci_coarse / (rp_fine * gci_fine)
        if math.isfinite(gci_fine) and gci_fine != 0.0 else nan
    )

    return GridConvergenceStudy(
        axis=axis, levels=levels, h=h, values=values, regime=regime,
        ratio=ratio, order=p, order_method=solved.method,
        extrapolated=extrapolated, gci_fine=gci_fine, gci_coarse=gci_coarse,
        safety_factor=fs, formal_order=formal_order,
        in_asymptotic_range=in_range, asymptotic_ratio=asymptotic_ratio,
        asymptotic_detail=asym_detail,
        triplet_orders=tuple(triplet_orders), detail="",
    )


def _assess_asymptotic_range(
    p: float,
    formal_order: float | None,
    triplet_orders: tuple[float, ...],
    tolerance: float,
) -> tuple[bool | None, str]:
    """Decide whether the solutions are in the asymptotic range.

    Roache's published check — ``GCI_coarse / (r**p GCI_fine) ~ 1`` —
    is **reported but not used here**, because for a three-grid study
    at a constant refinement ratio it is very nearly a tautology.
    Substituting the GCI definitions and the ``p`` that was solved from
    the same three solutions, the ratio collapses to
    ``|phi_fine| / |phi_medium|``: it is within a per cent of 1 for any
    ladder whose solutions are close to each other, converging or not.
    (Verified on the LBM pipe ladder in ``test_gci_order.py``, where it
    reads 0.98 on a study that is emphatically *not* in the asymptotic
    range.)

    What does carry information:

    1. **The apparent order against the formal order.**  The asymptotic
       range is by definition where the leading truncation term
       dominates, so ``p`` approaching the formal order is the direct
       evidence.  This is the criterion ASME V&V 20 and Celik et al.
       (2008, §3) use, and it needs the node to have declared an order.
    2. **Agreement between independent triples**, when the ladder has
       four or more levels.  An order that is still moving as the grid
       refines is not yet asymptotic, and this needs nothing declared.

    With three levels and no declared order neither test is available,
    and the honest answer is ``None`` — not ``True``.  The caller sees
    a wider safety factor and a message saying why.
    """
    stable = [o for o in triplet_orders if math.isfinite(o)]
    checks: list[str] = []
    verdicts: list[bool] = []

    if formal_order is not None:
        ok = abs(p - formal_order) <= tolerance
        verdicts.append(ok)
        checks.append(
            f"observed p = {p:.3f} against the declared {formal_order:g} "
            f"({'within' if ok else 'outside'} +/-{tolerance:g})"
        )
    if len(triplet_orders) > 1:
        unusable = len(triplet_orders) - len(stable)
        if unusable:
            verdicts.append(False)
            checks.append(
                f"{unusable} of {len(triplet_orders)} triples do not converge "
                f"monotonically, so the order is not settling"
            )
        else:
            spread = max(stable) - min(stable)
            ok = spread <= tolerance
            verdicts.append(ok)
            checks.append(
                f"per-triple orders span {spread:.3f} "
                f"({'within' if ok else 'outside'} {tolerance:g})"
            )

    if not verdicts:
        return None, (
            f"undetermined — p = {p:.3f}, but with three levels and no "
            f"declared order there is nothing to compare it against.  "
            f"Declare the node's order, or add a fourth level, to decide "
            f"this; meanwhile Fs = {CAUTIOUS_SAFETY_FACTOR:g} is used rather "
            f"than {DEFAULT_SAFETY_FACTOR:g}"
        )
    joined = "; ".join(checks)
    return (all(verdicts), ("yes — " if all(verdicts) else "no — ") + joined)


def measure_gci(
    solution_at: Callable[[Any], float],
    levels: Sequence[Any],
    *,
    axis: RefinementAxis,
    h_of: Callable[[Any], float] | None = None,
    formal_order: float | None = None,
    safety_factor: float | None = None,
    stagnation_rtol: float = DEFAULT_STAGNATION_RTOL,
    asymptotic_tolerance: float = DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
) -> GridConvergenceStudy:
    """Run a refinement ladder and index its convergence.

    The same ladder :func:`measure_order` runs, with one difference
    that changes what it can cover: the callback returns the
    **solution**, not the error, so nothing outside the node is needed
    — no manufactured source, no exact field, no reference run.  That
    is what lets this reach nodes MMS cannot: see the module docstring.

    Parameters
    ----------
    solution_at : callable
        ``level -> phi``.  One scalar per level: a *functional* of the
        solution, not the field.  Anything that converges will do — a
        drag coefficient, a peak value, a flux, the L2 norm of the
        state — as long as it is computed the same way at every level.
        A functional that is dimensionless, or normalised identically
        at each level, makes the relative quantities below comparable;
        one whose definition drifts with the grid measures the drift.
    levels : sequence
        Refinement levels, coarsest first.  **At least three**, because
        the order is measured rather than assumed;
        :func:`measure_order` accepts two, and this deliberately does
        not.
    axis : RefinementAxis
        Which axis this ladder refines.
    h_of : callable, optional
        ``level -> h``.  Defaults to ``1 / level``.
    formal_order : float, optional
        The order the scheme claims, used for the asymptotic-range
        check.  :func:`verify_node_gci` fills this in from the node.
    safety_factor, stagnation_rtol, asymptotic_tolerance
        As for :func:`richardson_study`.

    Returns
    -------
    GridConvergenceStudy

    Raises
    ------
    ValueError
        Fewer than three levels, or ``h`` not strictly decreasing.
    """
    levels = tuple(levels)
    if len(levels) < 3:
        raise ValueError(
            f"a grid convergence study needs at least three refinement "
            f"levels, got {len(levels)}"
        )
    h_fn = h_of if h_of is not None else (lambda level: 1.0 / float(level))
    h = tuple(float(h_fn(level)) for level in levels)
    if any(h[i + 1] >= h[i] for i in range(len(h) - 1)):
        raise ValueError(
            f"refinement levels must be given coarsest first, so that "
            f"{axis.symbol} strictly decreases; got {axis.symbol} = {h}"
        )
    values = tuple(float(solution_at(level)) for level in levels)
    return richardson_study(
        values, h, axis=axis, levels=levels, formal_order=formal_order,
        safety_factor=safety_factor, stagnation_rtol=stagnation_rtol,
        asymptotic_tolerance=asymptotic_tolerance,
    )


def check_gci(
    study: GridConvergenceStudy,
    *,
    expected: float | None = None,
    max_gci: float | None = None,
    name: str = "grid_convergence",
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Judge a grid convergence study.

    Judges the *study*, not the node: it has no view on whether a node
    that merely converges consistently is verified.  That policy lives
    in :func:`verify_node_gci`, which is the node-level entry point and
    the one that refuses to return a pass when there is no criterion to
    judge against.

    In order:

    1. **A study that has not converged fails**, whatever else is
       asked of it — oscillatory, divergent, stagnant, non-finite, or
       an apparent order the implicit equation could not determine.
       Each reports its own regime by name.  This is the case that
       must never look like a pass, and it is checked first so that a
       node which silently ignores refinement cannot slip past by
       having no declared order to compare against.
    2. **A study positively shown to be outside the asymptotic range
       fails**, because outside it the GCI is not an error band.  This
       is not a formality: on the LBM pipe ladder in
       ``tests/verification/test_gci_order.py`` a study outside the
       asymptotic range quotes 1.2% around a solution that is 3.6%
       from the answer.  A study whose asymptotic range could not be
       *decided* — three levels and no declared order — is not failed,
       but it is given the cautious safety factor and says so.
    3. **The observed order against ``expected``**, if given, on the
       same band :func:`check_order` uses.
    4. **The GCI against ``max_gci``**, if given: the error band on the
       finest solution, as a fraction (``0.05`` is 5%).

    Parameters
    ----------
    study : GridConvergenceStudy
    expected : float, optional
        An order to hold the observed order to.
    max_gci : float, optional
        The widest acceptable error band on the finest solution, as a
        fraction of it.
    name : str
        Check name carried on the result.
    shortfall, excess : float
        Half-widths of the order band; see
        :data:`DEFAULT_ORDER_SHORTFALL`.

    Returns
    -------
    VerificationResult
        ``PASS`` or ``FAIL``.  Never ``SKIP``: a study always has an
        outcome, even when the outcome is that it could not converge.
    """
    table = study.table()
    n = len(study.values)
    if not study.regime.supports_extrapolation:
        return VerificationResult(
            name, "FAIL", n_examples=n,
            detail=(
                f"the {study.axis.order_attribute} refinement ladder shows "
                f"{study.regime.description}, so no Grid Convergence Index "
                f"exists for it and the finest solution carries no error "
                f"band.  {study.detail}\n{table}"
            ),
        )
    if not math.isfinite(study.order):
        return VerificationResult(
            name, "FAIL", n_examples=n,
            detail=(
                f"the observed {study.axis.order_attribute} order could not be "
                f"determined from this ladder, so there is no Grid "
                f"Convergence Index.  {study.detail}\n{table}"
            ),
        )

    if study.in_asymptotic_range is False:
        return VerificationResult(
            name, "FAIL", n_examples=n,
            detail=(
                f"the solutions are not in the asymptotic range, so the Grid "
                f"Convergence Index on the finest one is not an error band "
                f"and is not reported as one.  This is an inconclusive study, "
                f"not a wrong node, and the two must not be confused.  "
                f"{study.asymptotic_detail}.  Measured on the LBM pipe ladder "
                f"in tests/verification/test_gci_order.py, a study outside "
                f"the asymptotic range quoted a band of 1.2% around a "
                f"solution that was 3.6% from the answer — three times too "
                f"narrow, in the confident direction.\n{table}"
            ),
        )

    failures = []
    if expected is not None:
        low, high = expected - shortfall, expected + excess
        if study.order < low:
            failures.append(
                f"observed {study.axis.order_attribute} order "
                f"{study.order:.3f} is below the declared {expected:g} "
                f"(band [{low:.2f}, {high:.2f}]).  Measured without a "
                f"reference solution, so a shortfall here is the scheme's "
                f"own convergence rate falling short — a stencil weight, a "
                f"boundary closure, or a geometry that does not refine "
                f"smoothly"
            )
        elif study.order > high:
            failures.append(
                f"observed {study.axis.order_attribute} order "
                f"{study.order:.3f} is above the declared {expected:g} "
                f"(band [{low:.2f}, {high:.2f}]).  An apparent order well "
                f"above the formal one normally means the coarsest level is "
                f"nowhere near the asymptotic range, not that the scheme is "
                f"better than claimed"
            )
    if max_gci is not None and study.gci_fine > max_gci:
        failures.append(
            f"the Grid Convergence Index on the finest solution is "
            f"{study.gci_fine * 100:.3f}%, above the {max_gci * 100:g}% "
            f"asked of it (Fs = {study.safety_factor:g})"
        )
    if failures:
        return VerificationResult(
            name, "FAIL", n_examples=n,
            detail="; ".join(failures) + f"\n{table}",
        )
    summary = (
        f"observed order {study.order:.3f}, Richardson limit "
        f"{study.extrapolated:.6g}, GCI {study.gci_fine * 100:.3f}% "
        f"(Fs = {study.safety_factor:g})"
    )
    if expected is None and max_gci is None:
        summary += (
            " — the study converged; no order or band was asked of it"
        )
    return VerificationResult(name, "PASS", n_examples=n,
                              detail=f"{summary}\n{table}")


def _no_criterion_detail(node: Any, axis: RefinementAxis) -> str:
    return (
        f"{type(node).__name__} declares no {axis.order_attribute} order of "
        f"accuracy and the study was given no max_gci, so the ladder has "
        f"nothing to be judged against.  It ran and converged — the table "
        f"below is the measurement — but a convergent ladder on its own is "
        f"not a verified node, and this harness will not report one as if it "
        f"were.  Give it something to check: set NodeMeta("
        f"discretization_order=DiscretizationOrder({axis.order_attribute}="
        f"...)) on the class, override discretization_order() on the "
        f"instance, pass expected=, or pass max_gci= to hold the error band "
        f"to a width."
    )


def verify_node_gci(
    node: Any,
    *,
    axis: RefinementAxis,
    solution_at: Callable[[Any], float],
    levels: Sequence[Any],
    h_of: Callable[[Any], float] | None = None,
    expected: float | None = None,
    max_gci: float | None = None,
    safety_factor: float | None = None,
    stagnation_rtol: float = DEFAULT_STAGNATION_RTOL,
    asymptotic_tolerance: float = DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Run a grid convergence study on one node and judge it.

    The counterpart to :func:`verify_node_order` for nodes the Method
    of Manufactured Solutions cannot reach, returning the same
    :class:`~maddening.testing.verification.VerificationResult` so the
    two report together.

    **Order of judgement, and why it is this way round.**
    :func:`verify_node_order` decides to skip an undeclared node
    *before* running the ladder; this one runs the ladder *first* and
    only then looks at what the node declares.  A study is the evidence
    that the node refines at all, and that evidence is exactly what a
    node with no declared order would otherwise escape producing.  So a
    node that ignores its refinement parameter ``FAIL``\\ s here, and is
    never skipped into looking fine.  The cost is that an undeclared
    node pays for three runs before it skips.

    Parameters
    ----------
    node : SimulationNode
        Only its declared order is read, and only to check the
        asymptotic range and supply ``expected``.  ``solution_at``
        decides how the node is built and driven at each level.
    axis : RefinementAxis
    solution_at, levels, h_of
        As for :func:`measure_gci`.
    expected : float, optional
        Override the declared order.  Leave it unset in a node's own
        test: taking the claim from the node is what makes a wrong
        claim visible.
    max_gci : float, optional
        The widest acceptable error band on the finest solution, as a
        fraction of it.  The criterion that needs nothing declared —
        the one a user-written node can use on day one.
    safety_factor, stagnation_rtol, asymptotic_tolerance
        As for :func:`richardson_study`.
    shortfall, excess : float
        Order band; see :data:`DEFAULT_ORDER_SHORTFALL`.

    Returns
    -------
    VerificationResult
        ``FAIL`` when the ladder did not converge or was shown to be
        outside the asymptotic range, whatever the node declares.
        ``SKIP`` — explicitly, never a silent pass — when the ladder
        *did* converge but the node declares no order and no
        ``expected`` or ``max_gci`` was given, so there is no criterion
        to judge it by.  ``PASS`` otherwise.
    """
    check_name = f"{axis.order_attribute}_grid_convergence"
    declared = declared_order(node)
    formal = (
        getattr(declared, axis.order_attribute) if declared is not None else None
    )
    formal = float(formal) if formal is not None else None

    study = measure_gci(
        solution_at, levels, axis=axis, h_of=h_of,
        formal_order=formal if expected is None else float(expected),
        safety_factor=safety_factor, stagnation_rtol=stagnation_rtol,
        asymptotic_tolerance=asymptotic_tolerance,
    )
    criterion = expected if expected is not None else formal
    if criterion is None and max_gci is None:
        result = check_gci(study, name=check_name)
        if result.failed:
            return result
        return VerificationResult(
            check_name, "SKIP", n_examples=len(study.values),
            detail=f"{_no_criterion_detail(node, axis)}\n{study.table()}",
        )
    return check_gci(
        study, expected=criterion, max_gci=max_gci, name=check_name,
        shortfall=shortfall, excess=excess,
    )


def assert_node_gci_verified(node: Any, **kwargs: Any) -> None:
    """:func:`verify_node_gci` that raises rather than returning.

    A one-line pytest body, matching
    :func:`assert_node_order_verified`::

        def test_the_pipe_flow_profile_converges_under_refinement():
            assert_node_gci_verified(
                node, axis=RefinementAxis.SPACE,
                solution_at=flow_shape_at, levels=(16, 24, 32),
                max_gci=0.60,
            )

    Raises
    ------
    AssertionError
        The ladder did not converge, the observed order missed the
        band, or the Grid Convergence Index was wider than asked.  The
        message carries the whole convergence table.
    InconclusiveStudyError
        The ladder converged but there was nothing to judge it
        against.  Raised rather than passed over, so a node with no
        declared order and no requested band cannot look verified; a
        test that means to tolerate that catches this and skips::

            try:
                assert_node_gci_verified(node, ...)
            except InconclusiveStudyError as exc:
                pytest.skip(str(exc))
    """
    result = verify_node_gci(node, **kwargs)
    if result.skipped:
        raise InconclusiveStudyError(result.detail)
    if not result.passed:
        raise AssertionError(
            f"{type(node).__name__} failed the {result.name} check:\n"
            f"{result.detail}"
        )
