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
