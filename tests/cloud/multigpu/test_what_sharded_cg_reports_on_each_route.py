"""What ``sharded_cg`` reports on each of its routes, as it stands in 0.4.0.

Two measured behaviours of a STABLE function are open in the registry and
pinned here as what happens today, each beside the way out its entry and
the docstring give, which holds:

* **MADD-ANO-263.**  On the loop backend (``backend="loop"``, and
  ``"auto"`` with a preconditioner) ``converged`` and ``residual_norm``
  are the residual the iteration updates (``r <- r - alpha * A p``), not
  ``b - A x``.  In float32 the two drift apart by about ``eps * kappa``:
  at the function's default ``rtol=1e-6`` the flag is True and the norm
  reads under the tolerance on solves whose true residual is 11.7 and 115
  times the tolerance.  Under ``differentiable=True`` the same solve is
  answered ``converged=False`` with the true residual, which is the way
  out (or one product, ``||b - matvec(x)||``).
* **MADD-ANO-264.**  The default route without a preconditioner is
  lineax's CG, which sets its step to NaN when
  ``|p.A p| <= 100 * rcond * |r.r|`` with ``rcond = 2 * eps * n``.  In
  float32 that is a quotient ``p.A p / r.r`` under ``2.4e-5 * n``: a
  smooth right-hand side fails it at 4096 unknowns and white noise at 1e5,
  on a system whose eigenvalues are of order one.  The answer is NaN in
  every entry with ``converged=False`` (loud).  ``backend="loop"`` and
  float64 both solve the same systems.

Neither pin is a wish.  When one stops being true -- the loop reports the
residual of the system, or the default route answers in float32 -- the
test that fails here is to be rewritten with the registry entry it names
and the sentences in ``sharded_cg``'s docstring that cite it.

The system is the 1-D Dirichlet Laplacian shifted on its diagonal by ``s``
(``(2 + s) x[i] - x[i-1] - x[i+1]``, eigenvalues in ``(s, 4 + s)``,
condition number about ``(4 + s) / s``).  Every true residual is
``||b - A x|| / ||b||`` in float64 on the host, with the diagonal as the
solve's dtype holds it.  Measured on jax 0.10.2, 0.11.0 and 0.11.2 (CPU,
lineax 0.0.7); the three agree to the digits quoted in each test.  No
iteration count of the loop is pinned.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.iterative_solver import (
    jacobi_preconditioner,
    sharded_cg,
)

EPS32 = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------
# The system, its right-hand sides, and a solve read back on the host
# ---------------------------------------------------------------------------


def _matvec(shift: float):
    def matvec(x):
        left = jnp.concatenate([jnp.zeros((1,), x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.zeros((1,), x.dtype)])
        return (2 + shift) * x - left - right

    return matvec


def _apply64(x, diagonal: float) -> np.ndarray:
    """The same operator in float64 on the host, written a second time."""
    x = np.asarray(x, np.float64)
    ax = diagonal * x
    ax[1:] -= x[:-1]
    ax[:-1] -= x[1:]
    return ax


def _white_noise(n: int, dtype) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n).astype(dtype)


def _smooth(n: int, dtype) -> np.ndarray:
    return (np.sin(np.linspace(0.0, 6.0, n)) + 0.1).astype(dtype)


@contextlib.contextmanager
def _x64(on: bool):
    """``jax_enable_x64`` set to *on* for the block, restored after it."""
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@dataclass(frozen=True)
class Answer:
    value: np.ndarray
    converged: bool
    iters: int
    reported: float     # ``residual_norm / ||b||``, as the solver reports it
    true: float         # ``||b - A x|| / ||b||`` in float64 on the host


def _solve(shift: float, b_host: np.ndarray, **kw) -> Answer:
    """One jitted ``sharded_cg`` solve in ``b_host``'s dtype (the caller
    holds ``jax_enable_x64`` to match)."""
    matvec = _matvec(shift)

    def solve(b):
        # ``SharedSolveResult`` is not a JAX type: unpack it inside the jit.
        r = sharded_cg(matvec, b, **kw)
        return r.value, r.converged, r.iters, r.residual_norm

    value, converged, iters, residual_norm = jax.jit(solve)(jnp.asarray(b_host))
    assert value.dtype == b_host.dtype, (value.dtype, b_host.dtype)
    value = np.asarray(value)
    b64 = b_host.astype(np.float64)
    b_norm = float(np.linalg.norm(b64))
    diagonal = float(b_host.dtype.type(2 + shift))
    true = float(np.linalg.norm(b64 - _apply64(value, diagonal)) / b_norm)
    return Answer(value=value, converged=bool(converged), iters=int(iters),
                  reported=float(residual_norm) / b_norm, true=true)


# ---------------------------------------------------------------------------
# MADD-ANO-263: the loop backend reports its recursively updated residual
# ---------------------------------------------------------------------------

#: ``sharded_cg``'s own default.
DEFAULT_RTOL = 1e-6
LOOP_UNKNOWNS = 4096
LOOP_BUDGET = 3000
#: ``shift``: the condition number is about ``(4 + shift) / shift``, 401 and 4001.
LOOP_SHIFTS = (1e-2, 1e-3)


def _loop_route(route: str, shift: float) -> dict:
    """The two ways a caller reaches the loop: by name, and through the
    default ``backend="auto"`` with a preconditioner (Jacobi here)."""
    if route == "loop":
        return {"backend": "loop"}
    assert route == "auto with a Jacobi preconditioner"
    diag = jnp.full((LOOP_UNKNOWNS,), 2 + shift, jnp.float32)
    return {"preconditioner": jacobi_preconditioner(diag)}


@pytest.mark.parametrize("shift", LOOP_SHIFTS)
@pytest.mark.parametrize("route", ["loop", "auto with a Jacobi preconditioner"])
def test_the_loop_answers_converged_on_a_solve_whose_true_residual_is_far_over_rtol(
        route, shift):
    """MADD-ANO-263 as it stands.  The loop stops on its own test with
    iterations to spare, says ``converged=True`` and reports a norm under
    ``rtol * ||b||``; the residual of the system is more than five times
    the tolerance.  Measured: reported 9.58e-7 and 9.92e-7 against a true
    1.17e-5 and 1.15e-4 (11.7 and 115 times ``rtol``) on the named loop,
    1.16e-5 and 1.17e-4 behind the preconditioner."""
    b = _white_noise(LOOP_UNKNOWNS, np.float32)
    got = _solve(shift, b, rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET,
                 **_loop_route(route, shift))
    assert got.converged, "the loop no longer answers converged=True here: MADD-ANO-263"
    assert 0 < got.iters < LOOP_BUDGET
    assert got.reported <= DEFAULT_RTOL, got.reported
    assert got.true > 5 * DEFAULT_RTOL, (
        f"true residual {got.true:.3e}: if the loop now reports the residual of "
        "the system, close MADD-ANO-263 and rewrite this test")
    # Not a failed solve either: the gap is float32's ``eps * kappa``, and
    # the same solve in float64 has none (the control below).
    assert got.true < 1e3 * DEFAULT_RTOL, got.true


@pytest.mark.parametrize("shift", LOOP_SHIFTS)
@pytest.mark.parametrize("route", ["loop", "auto with a Jacobi preconditioner"])
def test_differentiable_true_answers_not_converged_on_the_same_solve_with_the_true_residual(
        route, shift):
    """The way out MADD-ANO-263 gives, and the third answer of one
    function: under ``differentiable=True`` the flag and the norm come
    from one extra product on the returned value, so the same solve reads
    ``converged=False`` with a norm within 20 % of the float64 one
    (measured: within 0.4 %)."""
    b = _white_noise(LOOP_UNKNOWNS, np.float32)
    kw = dict(rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET, **_loop_route(route, shift))
    loop = _solve(shift, b, **kw)
    checked = _solve(shift, b, differentiable=True, **kw)
    assert loop.converged and not checked.converged, (loop.converged, checked.converged)
    assert checked.iters == -1            # the documented price of the flag
    assert checked.reported == pytest.approx(checked.true, rel=0.2)
    assert checked.reported > 5 * DEFAULT_RTOL >= 5 * loop.reported
    # One solve, read twice: the two values have the same true residual.
    assert checked.true == pytest.approx(loop.true, rel=0.2)


def test_the_loop_reports_the_residual_of_the_system_in_float64():
    """The control: in float64 the recursive residual and the true one
    agree (measured: to four digits, 9.58e-7), so the flag means what it
    says.  ``differentiable=True`` reports the same number."""
    with _x64(True):
        b = _white_noise(LOOP_UNKNOWNS, np.float64)
        kw = dict(backend="loop", rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET)
        loop = _solve(LOOP_SHIFTS[0], b, **kw)
        checked = _solve(LOOP_SHIFTS[0], b, differentiable=True, **kw)
    assert loop.converged and loop.reported <= DEFAULT_RTOL
    assert loop.reported == pytest.approx(loop.true, rel=1e-3)
    assert checked.reported == pytest.approx(loop.true, rel=1e-3)
    assert checked.converged


def test_the_differentiable_flag_is_its_own_norm_against_rtol_with_no_allowance():
    """The strict side of the same flag, stated as what it is rather than
    by which way one solve falls: ``converged`` under ``differentiable=True``
    is exactly ``residual_norm <= rtol * ||b||`` on the float32 residual of
    the returned value.  A loop solve that stopped on its tolerance with a
    true residual just over it therefore reads False (measured at 1024
    unknowns, ``s = 1e-2``, ``rtol = 1e-4``, a smooth right-hand side: the
    loop stops after 75 iterations reporting 9.52e-5, the true residual is
    1.06e-4, the flag is False).  Which side of ``rtol`` that solve lands
    on is rounding and is not asserted."""
    rtol = 1e-4
    b = _smooth(1024, np.float32)
    kw = dict(backend="loop", rtol=rtol, max_iters=LOOP_BUDGET)
    loop = _solve(1e-2, b, **kw)
    checked = _solve(1e-2, b, differentiable=True, **kw)
    assert loop.converged and loop.reported <= rtol
    assert checked.reported == pytest.approx(checked.true, rel=0.2)
    assert 0.5 * rtol < checked.true < 2 * rtol       # at the tolerance, either side
    # Not within float32's rounding of the tolerance itself, where the
    # comparison below would be rounding too (measured: 6 % from it).
    assert abs(checked.reported / rtol - 1.0) > 1e-4, checked.reported
    assert checked.converged == (checked.reported <= rtol), (
        checked.converged, checked.reported)


# ---------------------------------------------------------------------------
# MADD-ANO-264: the default route returns NaN in float32
# ---------------------------------------------------------------------------

DEFAULT_ROUTE_RTOL = 1e-4
DEFAULT_ROUTE_BUDGET = 500


def breakdown_threshold(n: int, eps: float = EPS32) -> float:
    """What lineax's CG holds ``p.A p / r.r`` above: it sets the step to
    NaN unless ``|p.A p| > 100 * rcond * |r.r|`` with ``rcond = 2 * eps * n``
    (lineax 0.0.7, ``_solver/cg.py`` and ``_misc.py::resolve_rcond``)."""
    return 100 * (2 * eps * n)


@dataclass(frozen=True)
class BreakdownCase:
    unknowns: int
    shift: float
    rhs: str            # "smooth" or "white noise"
    #: The step on which the quotient is first under the threshold.
    fails_on_step: int

    def b(self, dtype) -> np.ndarray:
        make = _smooth if self.rhs == "smooth" else _white_noise
        return make(self.unknowns, dtype)


#: The cheap case (condition number 401, one step) and the one at scale
#: (condition number 9, an easy system; two steps).
BREAKDOWN_CASES = {
    "a smooth right-hand side at 4096 unknowns":
        BreakdownCase(unknowns=4096, shift=0.01, rhs="smooth", fails_on_step=1),
    "white noise at 1e5 unknowns":
        BreakdownCase(unknowns=100_000, shift=0.5, rhs="white noise", fails_on_step=2),
}
_breakdown_cases = pytest.mark.parametrize(
    "case", list(BREAKDOWN_CASES.values()), ids=list(BREAKDOWN_CASES))


def _quotients(case: BreakdownCase, steps: int) -> list[float]:
    """``p.A p / r.r`` of the first CG steps from a zero start, in float64
    on the host: the number lineax compares with its threshold.  It is
    ``1 / alpha``, between the operator's extreme eigenvalues; on step 1 it
    is the right-hand side's Rayleigh quotient ``b.A b / b.b``."""
    diagonal = float(np.float32(2 + case.shift))
    r = case.b(np.float32).astype(np.float64)
    p, gamma, out = r.copy(), float(r @ r), []
    for _ in range(steps):
        ap = _apply64(p, diagonal)
        p_ap = float(p @ ap)
        out.append(p_ap / gamma)
        r = r - (gamma / p_ap) * ap
        gamma_next = float(r @ r)
        p = r + (gamma_next / gamma) * p
        gamma = gamma_next
    return out


@_breakdown_cases
@pytest.mark.parametrize("backend", ["auto", "lineax"])
def test_the_default_route_returns_nan_in_float32(case, backend):
    """MADD-ANO-264 as it stands: ``backend="auto"`` without a
    preconditioner, and ``backend="lineax"``, answer NaN in every entry, a
    NaN ``residual_norm`` and ``converged=False`` after one or two steps
    (measured: 1 at 4096 unknowns, 2 at 1e5)."""
    got = _solve(case.shift, case.b(np.float32), backend=backend,
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isnan(got.value).all(), (
        f"{int(np.isfinite(got.value).sum())} finite entries: if the default route "
        "now answers in float32, close MADD-ANO-264 and rewrite this test")
    assert math.isnan(got.reported)
    assert not got.converged              # loud: nobody is told it converged
    assert 1 <= got.iters <= 2, got.iters


@_breakdown_cases
def test_the_route_breaks_down_on_the_step_whose_quotient_is_under_the_threshold(case):
    """The arithmetic of MADD-ANO-264, so that the reason is on record and
    a lineax release that moves it is noticed.  The threshold is
    ``200 * eps32 * n``: 0.098 at 4096 unknowns, 2.38 at 1e5, 23.8 at 1e6.

    * smooth, 4096, ``s = 0.01``: the right-hand side's Rayleigh quotient
      is 0.0100 (about ``s``), under 0.098: NaN on step 1;
    * white noise, 1e5, ``s = 0.5``: the Rayleigh quotient is 2.51, over
      2.38, so the first step is taken; the second direction's quotient is
      1.70: NaN on step 2.

    And at 1e6 unknowns the threshold is over the operator's largest
    eigenvalue, which the quotient cannot exceed: no right-hand side
    passes."""
    threshold = breakdown_threshold(case.unknowns)
    quotients = _quotients(case, steps=case.fails_on_step)
    passed, failed = quotients[:-1], quotients[-1]
    assert all(q > 1.03 * threshold for q in passed), (quotients, threshold)
    assert failed < 0.8 * threshold, (quotients, threshold)
    got = _solve(case.shift, case.b(np.float32),
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isnan(got.value).all() and got.iters == case.fails_on_step, got.iters
    # The quotient lies between the extreme eigenvalues, ``s`` and ``4 + s``
    # (1 % allowed for the float32 diagonal), and at 1e6 unknowns the
    # threshold is above the upper one.
    assert all(0.99 * case.shift < q < 4 + case.shift for q in quotients), quotients
    assert breakdown_threshold(1_000_000) > 4 + case.shift


def test_the_default_route_answers_in_float32_where_every_quotient_passes():
    """The other side of the same arithmetic: white noise at 4096 unknowns
    with ``s = 0.5`` has quotients between 1.7 and 2.5 against a threshold
    of 0.098, and the default route converges (measured: 14 steps, true
    residual 8.5e-5).  So the NaN above is the threshold's, not the
    route's on every float32 system."""
    case = BreakdownCase(unknowns=4096, shift=0.5, rhs="white noise", fails_on_step=0)
    assert min(_quotients(case, steps=6)) > 10 * breakdown_threshold(case.unknowns)
    got = _solve(case.shift, case.b(np.float32),
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert got.converged and np.isfinite(got.value).all()
    assert got.true <= 2 * DEFAULT_ROUTE_RTOL, got.true


@_breakdown_cases
def test_the_same_call_in_float64_converges(case):
    """The first way out MADD-ANO-264 gives: the identical call in float64
    (threshold ``200 * eps64 * n``, under 5e-9 here) converges.  Measured
    true residuals: 3.3e-6 and 8.6e-5."""
    assert breakdown_threshold(case.unknowns, float(np.finfo(np.float64).eps)) < case.shift
    with _x64(True):
        got = _solve(case.shift, case.b(np.float64),
                     rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert got.converged and np.isfinite(got.value).all()
    assert got.reported == pytest.approx(got.true, rel=1e-3)
    assert got.true <= 2 * DEFAULT_ROUTE_RTOL, got.true


@_breakdown_cases
def test_the_loop_backend_converges_in_float32_on_the_same_system(case):
    """The second way out: ``backend="loop"`` in float32 solves both
    systems, the true residual checked on the host (measured: 1.01e-4 at
    4096 unknowns, where the loop reported 9.7e-5 -- MADD-ANO-263's gap at
    a condition number of 401 -- and 8.6e-5 at 1e5)."""
    got = _solve(case.shift, case.b(np.float32), backend="loop",
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert got.converged and np.isfinite(got.value).all()
    assert 0 < got.iters < DEFAULT_ROUTE_BUDGET
    assert got.true <= 2 * DEFAULT_ROUTE_RTOL, got.true
