"""What ``sharded_cg`` and ``sharded_gmres`` report on each of their routes.

Two released behaviours of these STABLE functions were corrected in 0.4.0,
and this file pins the corrected ones:

* **MADD-ANO-263.**  Every loop route reports the residual of the system it
  was given: ``residual_norm`` is ``||b - A x||`` of the returned value
  from one extra product, and ``converged`` is that norm against
  ``max(atol, rtol * ||b||)`` with no allowance for rounding.  Until 0.4.0
  the CG loop (``backend="loop"``, and ``"auto"`` with a preconditioner)
  reported the residual its iteration updates (``r <- r - alpha * A p``),
  which in float32 drifts from ``b - A x`` by about ``eps * kappa``: at the
  default ``rtol=1e-6`` it said ``converged=True`` with a norm under the
  tolerance on solves whose true residual was 11.7 and 115 times the
  tolerance.  The GMRES loop behind a preconditioner reported the
  left-preconditioned residual.  WHEN the loops stop was not changed: the
  value and the iteration count are what they were, and a float32 solve the
  loop stopped early on now reads ``converged=False`` with its true residual.
* **MADD-ANO-264.**  lineax's CG sets its step to NaN when
  ``|p.A p| <= 100 * rcond * |r.r|`` with ``rcond = 2 * eps * n``.  In
  float32 that is a quotient ``p.A p / r.r`` under ``2.4e-5 * n``: a
  smooth right-hand side fails it at 4096 unknowns and white noise at 1e5,
  on a system whose eigenvalues are of order one, and every entry of
  lineax's answer is NaN.  Until 0.4.0 that was the answer of the default
  call.  The default route (``backend="auto"`` without a preconditioner)
  now solves such a system on the loop backend and reports as the loop
  does; where lineax's value is finite its result is returned bit for bit.
  ``backend="lineax"``, asked for by name, still returns the NaN, and that
  is pinned here too (the entry stays open for it).

The system is the 1-D Dirichlet Laplacian shifted on its diagonal by ``s``
(``(2 + s) x[i] - x[i-1] - x[i+1]``, eigenvalues in ``(s, 4 + s)``,
condition number about ``(4 + s) / s``).  Every true residual is
``||b - A x|| / ||b||`` in float64 on the host, with the diagonal as the
solve's dtype holds it.  Measured on jax 0.10.2, 0.11.0 and 0.11.2 (CPU,
lineax 0.0.7); the three agree to the digits quoted in each test.  No
iteration count of the loop is pinned.
"""

# No ``from __future__ import annotations`` here: the device-policy test of
# this directory executes every test module without registering it in
# ``sys.modules``, where a dataclass with string annotations cannot be built.
import contextlib
import math
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.iterative_solver import (
    block_jacobi_preconditioner,
    jacobi_preconditioner,
    sharded_cg,
    sharded_gmres,
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


def _matvec_on_a_mesh(shift: float, mesh):
    """The same operator under ``shard_map`` on a 1-D mesh: each device holds
    a contiguous block and gets its neighbours' end values by ``ppermute``."""
    devices = mesh.devices.shape[0]

    def on_a_shard(x):
        from_left = lax.ppermute(x[-1], "devices", [(i, (i + 1) % devices) for i in range(devices)])
        from_right = lax.ppermute(x[0], "devices", [(i, (i - 1) % devices) for i in range(devices)])
        me = lax.axis_index("devices")
        from_left = jnp.where(me == 0, 0.0, from_left)
        from_right = jnp.where(me == devices - 1, 0.0, from_right)
        left = jnp.concatenate([jnp.asarray([from_left], x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.asarray([from_right], x.dtype)])
        return (2 + shift) * x - left - right

    return shard_map(on_a_shard, mesh=mesh, in_specs=(P("devices"),), out_specs=P("devices"))


def _four_devices() -> dict:
    """``mesh=`` and ``in_specs=`` of a solve on four (virtual CPU) devices."""
    return {"mesh": create_device_mesh(shape=(4,)), "in_specs": P("devices")}


def _thomas64(rhs, diagonal: float) -> np.ndarray:
    """A float64 direct solve of the system on the host (the Thomas
    algorithm): the reference the derivatives are held to."""
    rhs = np.asarray(rhs, np.float64)
    n = rhs.shape[0]
    c, d = np.empty(n), np.empty(n)
    c[0], d[0] = -1.0 / diagonal, rhs[0] / diagonal
    for i in range(1, n):
        pivot = diagonal + c[i - 1]
        c[i] = -1.0 / pivot
        d[i] = (rhs[i] + d[i - 1]) / pivot
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _fields(matvec, b, solver=sharded_cg, jit: bool = True, **kw) -> tuple:
    """``(value, converged, iters, residual_norm)`` of one solve, as host
    arrays: what "bit for bit" is compared on."""
    def solve(b):
        # ``SharedSolveResult`` is not a JAX type: unpack it inside the jit.
        r = solver(matvec, b, **kw)
        return r.value, r.converged, r.iters, r.residual_norm

    return tuple(np.asarray(a) for a in (jax.jit(solve) if jit else solve)(jnp.asarray(b)))


def _same_bits(got: tuple, want: tuple) -> list:
    """The names of the fields of two ``_fields`` results that differ in any
    bit (NaN equal to NaN)."""
    names = ("value", "converged", "iters", "residual_norm")
    return [name for name, a, b in zip(names, got, want)
            if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f")]


def _solve(shift: float, b_host: np.ndarray, jit: bool = True, on_mesh: bool = False,
           **kw) -> Answer:
    """One ``sharded_cg`` solve in ``b_host``'s dtype (the caller holds
    ``jax_enable_x64`` to match), jitted unless ``jit=False``."""
    where = _four_devices() if on_mesh else {}
    matvec = _matvec_on_a_mesh(shift, where["mesh"]) if on_mesh else _matvec(shift)
    value, converged, iters, residual_norm = _fields(matvec, b_host, jit=jit, **where, **kw)
    assert value.dtype == b_host.dtype, (value.dtype, b_host.dtype)
    b64 = b_host.astype(np.float64)
    b_norm = float(np.linalg.norm(b64))
    diagonal = float(b_host.dtype.type(2 + shift))
    true = float(np.linalg.norm(b64 - _apply64(value, diagonal)) / b_norm)
    return Answer(value=value, converged=bool(converged), iters=int(iters),
                  reported=float(residual_norm) / b_norm, true=true)


def _the_iteration_as_released(matvec, b, *, rtol: float, max_iters: int, preconditioner=None):
    """The CG loop as 0.3.0 and 0.3.1 shipped it, written a second time:
    ``(x, iters, ||r||)`` with ``r`` the residual the iteration updates,
    which is what those releases reported.  From a zero start, with the
    purely relative test (``atol=None``)."""
    apply_m = preconditioner if preconditioner is not None else (lambda r: r)
    tol2 = (rtol * jnp.linalg.norm(b)) ** 2
    x0 = jnp.zeros_like(b)
    r0 = b - matvec(x0)
    z0 = apply_m(r0)
    rho0 = jnp.vdot(r0, z0).real

    def unfinished(state):
        _, r, _, _, _, iters = state
        return jnp.logical_and(jnp.vdot(r, r).real > tol2, iters < max_iters)

    def step(state):
        x, r, _, p, rho, iters = state
        ap = matvec(p)
        alpha = rho / jnp.maximum(jnp.vdot(p, ap).real, jnp.finfo(b.dtype).tiny)
        x_new = x + alpha * p
        r_new = r - alpha * ap
        z_new = apply_m(r_new)
        rho_new = jnp.vdot(r_new, z_new).real
        beta = rho_new / jnp.maximum(rho, jnp.finfo(b.dtype).tiny)
        return (x_new, r_new, z_new, z_new + beta * p, rho_new, iters + 1)

    x, r, _, _, _, iters = lax.while_loop(unfinished, step, (x0, r0, z0, z0, rho0, jnp.int32(0)))
    return x, iters, jnp.linalg.norm(r)


# ---------------------------------------------------------------------------
# MADD-ANO-263: every loop route reports the residual of the system
# ---------------------------------------------------------------------------

#: ``sharded_cg``'s own default.
DEFAULT_RTOL = 1e-6
LOOP_UNKNOWNS = 4096
LOOP_BUDGET = 3000
#: ``shift``: the condition number is about ``(4 + shift) / shift``, 401 and 4001.
LOOP_SHIFTS = (1e-2, 1e-3)


def _loop_route(route: str, shift: float) -> dict:
    """The two ways a caller asks for the loop: by name, and through the
    default ``backend="auto"`` with a preconditioner (Jacobi here)."""
    if route == "loop":
        return {"backend": "loop"}
    assert route == "auto with a Jacobi preconditioner"
    diag = jnp.full((LOOP_UNKNOWNS,), 2 + shift, jnp.float32)
    return {"preconditioner": jacobi_preconditioner(diag)}


@pytest.mark.parametrize("shift", LOOP_SHIFTS)
@pytest.mark.parametrize("route", ["loop", "auto with a Jacobi preconditioner"])
def test_the_loop_answers_not_converged_with_the_true_residual_where_it_stopped_early(
        route, shift):
    """MADD-ANO-263, fixed.  At the default ``rtol=1e-6`` in float32 the
    loop stops on its own test with iterations to spare, at a residual of
    the system of 1.17e-5 ``||b||`` (condition number 401) and 1.15e-4
    (4001): 11.7 and 115 times the tolerance.  It now says so:
    ``converged=False`` and a ``residual_norm`` within 5 % of the float64
    residual (measured: within 0.7 %).  Until 0.4.0 it said
    ``converged=True`` and reported 9.58e-7 and 9.92e-7."""
    b = _white_noise(LOOP_UNKNOWNS, np.float32)
    got = _solve(shift, b, rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET,
                 **_loop_route(route, shift))
    assert 0 < got.iters < LOOP_BUDGET        # its own test stopped it, not the cap
    assert got.true > 5 * DEFAULT_RTOL, got.true
    assert not got.converged, "the loop vouches for a solve that missed rtol: MADD-ANO-263"
    assert got.reported == pytest.approx(got.true, rel=0.05), (got.reported, got.true)
    # Not a failed solve: the value is the float32 answer the loop reached,
    # and the same solve in float64 meets the tolerance (the control below).
    assert got.true < 1e3 * DEFAULT_RTOL, got.true


@pytest.mark.parametrize("on_mesh", [False, True], ids=["one device", "four devices"])
@pytest.mark.parametrize("preconditioned", [False, True], ids=["plain", "Jacobi"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_when_the_loop_stops_did_not_change_only_what_it_reports(dtype, preconditioned,
                                                                on_mesh):
    """The fix changed the report and nothing else: ``value`` and ``iters``
    are, bit for bit, those of the iteration as 0.3.0 shipped it (written
    a second time above).  And the two reports side by side: in float32
    that iteration's own residual reads under the tolerance (what the
    releases reported, with ``converged=True``) while ``residual_norm`` is
    more than five times it; in float64 the two agree and the flag is True.

    The right-hand side is scaled so that its largest entry is in
    ``[0.5, 1)``, where the solver's power-of-two frame is 1 and the second
    implementation needs none."""
    shift = LOOP_SHIFTS[0]
    with _x64(dtype is np.float64):
        b = _white_noise(LOOP_UNKNOWNS, dtype)
        b = (b * dtype(0.75 / np.max(np.abs(b)))).astype(dtype)
        assert 0.5 <= float(np.max(np.abs(b))) < 1.0
        where = _four_devices() if on_mesh else {}
        matvec = _matvec_on_a_mesh(shift, where["mesh"]) if on_mesh else _matvec(shift)
        pc = (jacobi_preconditioner(jnp.full((LOOP_UNKNOWNS,), 2 + shift, dtype))
              if preconditioned else None)
        value, converged, iters, residual_norm = _fields(
            matvec, b, backend="loop", rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET,
            preconditioner=pc, **where)
        x, steps, recursive = (np.asarray(a) for a in jax.jit(
            lambda b: _the_iteration_as_released(matvec, b, rtol=DEFAULT_RTOL,
                                                 max_iters=LOOP_BUDGET, preconditioner=pc))(
            jnp.asarray(b)))
    assert int(iters) == int(steps) and 0 < int(iters) < LOOP_BUDGET, (iters, steps)
    assert np.array_equal(value, x), float(np.max(np.abs(value - x)))
    tolerance = DEFAULT_RTOL * float(np.linalg.norm(b.astype(np.float64)))
    assert float(recursive) <= tolerance          # what the releases reported
    if dtype is np.float32:
        assert float(residual_norm) > 5 * tolerance and not bool(converged)
    else:
        assert float(residual_norm) == pytest.approx(float(recursive), rel=1e-3)
        assert bool(converged)


@pytest.mark.parametrize("shift", LOOP_SHIFTS)
@pytest.mark.parametrize("route", ["loop", "auto with a Jacobi preconditioner"])
def test_the_loop_and_differentiable_true_give_one_answer_for_one_solve(route, shift):
    """Until 0.4.0 one function gave two answers to "did it converge": the
    loop said True and the same call with ``differentiable=True``, which
    takes one extra product on the returned value, said False.  Both now
    make that statement, on the same value."""
    b = _white_noise(LOOP_UNKNOWNS, np.float32)
    kw = dict(rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET, **_loop_route(route, shift))
    loop = _solve(shift, b, **kw)
    checked = _solve(shift, b, differentiable=True, **kw)
    assert not loop.converged and not checked.converged
    assert checked.iters == -1            # the documented price of ``differentiable``
    assert np.array_equal(loop.value, checked.value)
    assert loop.reported == pytest.approx(checked.reported, rel=1e-3)
    assert checked.reported == pytest.approx(checked.true, rel=0.05)


def test_the_loop_reports_the_residual_of_the_system_in_float64():
    """The control: in float64 the recursive residual and the true one
    agree (measured: to four digits, 9.58e-7), so the same solve meets the
    default tolerance and the flag is True.  ``differentiable=True``
    reports the same number."""
    with _x64(True):
        b = _white_noise(LOOP_UNKNOWNS, np.float64)
        kw = dict(backend="loop", rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET)
        loop = _solve(LOOP_SHIFTS[0], b, **kw)
        checked = _solve(LOOP_SHIFTS[0], b, differentiable=True, **kw)
    assert loop.converged and loop.reported <= DEFAULT_RTOL
    assert loop.reported == pytest.approx(loop.true, rel=1e-3)
    assert checked.reported == pytest.approx(loop.true, rel=1e-3)
    assert checked.converged


@pytest.mark.parametrize("differentiable", [False, True])
def test_the_flag_is_the_reported_norm_against_rtol_with_no_allowance(differentiable):
    """What the flag is, stated as that rather than by which way one solve
    falls: ``converged`` is exactly ``residual_norm <= rtol * ||b||`` on the
    float32 residual of the returned value, on the loop and under
    ``differentiable=True`` alike.  A loop solve that stopped on its
    tolerance with a true residual just over it therefore reads False
    (measured at 1024 unknowns, ``s = 1e-2``, ``rtol = 1e-4``, a smooth
    right-hand side: the loop stops after 75 iterations, the true residual
    is 1.06e-4, the flag is False).  Which side of ``rtol`` that solve
    lands on is rounding and is not asserted."""
    rtol = 1e-4
    b = _smooth(1024, np.float32)
    got = _solve(1e-2, b, backend="loop", rtol=rtol, max_iters=LOOP_BUDGET,
                 differentiable=differentiable)
    assert got.reported == pytest.approx(got.true, rel=0.2)
    assert 0.5 * rtol < got.true < 2 * rtol           # at the tolerance, either side
    # Not within float32's rounding of the tolerance itself, where the
    # comparison below would be rounding too (measured: 6 % from it).
    assert abs(got.reported / rtol - 1.0) > 1e-4, got.reported
    assert got.converged == (got.reported <= rtol), (got.converged, got.reported)


@pytest.mark.parametrize("route", ["loop", "the default route where lineax breaks down"])
@pytest.mark.parametrize("larger, rtol, atol", [("atol", 1e-6, 1e-5),
                                                ("rtol * ||b||", 1e-3, 1e-12)],
                         ids=["atol the larger", "rtol times b the larger"])
def test_the_tolerance_of_the_flag_is_the_larger_of_atol_and_rtol_times_b(larger, rtol, atol,
                                                                          route):
    """``converged`` is against ``max(atol, rtol * ||b||)``, the tolerance
    the loop stops on, whichever of the two is the larger.  With ``atol``
    the larger (by 150 times) the loop stops at a residual under ``atol``
    and far over ``rtol * ||b||`` (measured: 5e-6 against 6.4e-8); with
    ``rtol * ||b||`` the larger it stops under that and far over ``atol``.
    Both are converged solves, and each would read False against the
    smaller of the two."""
    b = (_smooth(4096, np.float32) * np.float32(1e-3))
    b_norm = float(np.linalg.norm(b.astype(np.float64)))
    tolerance, the_other = max(atol, rtol * b_norm), min(atol, rtol * b_norm)
    assert tolerance > 100 * the_other
    assert tolerance == (atol if larger == "atol" else rtol * b_norm)
    kw = {"backend": "loop"} if route == "loop" else {}
    got = _solve(1e-2, b, rtol=rtol, atol=atol, max_iters=LOOP_BUDGET, **kw)
    assert got.iters > 2                      # the loop, on either route (lineax: 1 step, NaN)
    assert the_other < got.reported * b_norm <= tolerance, got.reported * b_norm
    assert got.converged


def _scaled_laplacian(n: int, spread: float) -> np.ndarray:
    """``D A D`` for the 1-D Laplacian ``A`` and ``D`` a diagonal from 1 to
    *spread*: SPD, and badly scaled without a preconditioner."""
    a = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    d = np.diag(np.geomspace(1.0, spread, n))
    return (d @ a @ d).astype(np.float32)


def test_the_gmres_loop_behind_a_preconditioner_reports_the_system_not_the_preconditioned_one():
    """The GMRES loop's cycles test the left-preconditioned residual,
    ``||M (b - A x)||`` against ``rtol * ||M b||``, and until 0.4.0 that
    pair was the report.  On a diagonally scaled Laplacian of 32 unknowns
    behind block-Jacobi at ``rtol=1e-6`` the preconditioned test passed
    (0.3.1 answers ``converged=True`` and a ``residual_norm`` of 1.06e-5)
    while ``||b - A x||`` is 1.9e-3, 18 times ``rtol * ||b||`` and 80 times
    the preconditioned norm.  The report is now of the system given:
    False, and a norm within 10 % of the float64 residual (measured: 4 %)."""
    n, block, rtol = 32, 4, 1e-6
    a = _scaled_laplacian(n, 100.0)
    b = (np.arange(n) + 1.0).astype(np.float32)
    blocks = jnp.stack([jnp.asarray(a[i:i + block, i:i + block]) for i in range(0, n, block)])
    apply_m = block_jacobi_preconditioner(blocks)
    a_dev = jnp.asarray(a)
    value, converged, iters, residual_norm = _fields(
        lambda x: a_dev @ x, b, solver=sharded_gmres, backend="loop", rtol=rtol,
        atol=1e-8, max_iters=2000, preconditioner=apply_m)
    a64, b64, x64 = a.astype(np.float64), b.astype(np.float64), value.astype(np.float64)
    true = float(np.linalg.norm(b64 - a64 @ x64))
    preconditioned = float(np.linalg.norm(np.asarray(apply_m(jnp.asarray(b64 - a64 @ x64,
                                                                         jnp.float32)))))
    assert preconditioned < 0.1 * true              # the two norms are not one number
    assert true > 5 * rtol * float(np.linalg.norm(b64)), true
    assert not bool(converged)
    assert float(residual_norm) == pytest.approx(true, rel=0.1)


@pytest.mark.parametrize("budget, solved", [(5, False), (200, True)],
                         ids=["one cycle of five", "to the tolerance"])
def test_the_gmres_loop_reports_alike_whatever_the_scale_of_its_preconditioner(budget, solved):
    """A preconditioner ``M = c I`` changes nothing about the system or, for
    ``c`` a power of two, about the iteration: the value and the count are
    the same bit for bit.  So must the report be, and it was not while the
    report was of the preconditioned system (``c`` times the residual
    against ``c`` times the tolerance is the same test, but the NUMBER
    reported was ``c`` times the residual).  Both sides of the tolerance:
    one cycle of five steps leaves the residual at 6e-3 ``||b||``
    (``converged=False``); 200 steps reach 2e-6 (True) at ``rtol=1e-4``."""
    shift, rtol, n = 0.5, 1e-4, 1024
    b = _white_noise(n, np.float32)
    b_norm = float(np.linalg.norm(b.astype(np.float64)))
    got = {c: _fields(_matvec(shift), b, solver=sharded_gmres, backend="loop", rtol=rtol,
                      restart=5, max_iters=budget,
                      preconditioner=lambda r, c=c: np.float32(c) * r)
           for c in (1.0, 1024.0, 1 / 1024.0)}
    for c in (1024.0, 1 / 1024.0):
        assert _same_bits(got[c], got[1.0]) == [], (c, _same_bits(got[c], got[1.0]))
    value, converged, _, residual_norm = got[1.0]
    true = float(np.linalg.norm(b.astype(np.float64)
                                - _apply64(value, float(np.float32(2 + shift)))))
    assert float(residual_norm) == pytest.approx(true, rel=0.2 if solved else 0.01)
    assert bool(converged) is solved
    # Within a factor 1024 of the tolerance on its side, so that a report
    # scaled by ``c`` or tested against ``c`` times the tolerance moves the flag.
    ratio = true / (rtol * b_norm)
    assert (1 / 1024 < ratio < 1) if solved else (1 < ratio < 1024), ratio


def test_a_diagonal_system_on_four_devices_is_solved_exactly_and_reads_converged():
    """The smallest contract a caller of the loop relies on, unchanged by
    the fix: ``A = 2 I`` under ``shard_map`` on four devices and ``b`` ones
    is solved in one step, ``value`` is exactly 0.5 and the solve reads
    converged (its residual is exactly zero)."""
    where = _four_devices()
    matvec = shard_map(lambda x: 2.0 * x, mesh=where["mesh"], in_specs=(P("devices"),),
                       out_specs=P("devices"))
    for jit in (True, False):
        value, converged, iters, residual_norm = _fields(
            matvec, np.ones(64, np.float32), jit=jit, max_iters=50, backend="loop", **where)
        assert bool(converged) and np.all(value == 0.5), jit
        assert int(iters) == 1 and float(residual_norm) == 0.0, jit


# ---------------------------------------------------------------------------
# MADD-ANO-264: the default route answers where lineax's CG breaks down
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
def test_the_lineax_backend_by_name_still_returns_nan_in_float32(case):
    """What MADD-ANO-264 is still open for: ``backend="lineax"``, asked for
    by name, has no fallback.  It answers NaN in every entry, a NaN
    ``residual_norm`` and ``converged=False`` after one or two steps
    (measured: 1 at 4096 unknowns, 2 at 1e5).  Until 0.4.0 the default
    call answered the same."""
    got = _solve(case.shift, case.b(np.float32), backend="lineax",
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isnan(got.value).all(), (
        f"{int(np.isfinite(got.value).sum())} finite entries: if lineax's CG now answers "
        "in float32, the fallback of the default route is no longer reached here: "
        "rewrite this file's MADD-ANO-264 half and the entry")
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
    got = _solve(case.shift, case.b(np.float32), backend="lineax",
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isnan(got.value).all() and got.iters == case.fails_on_step, got.iters
    # The quotient lies between the extreme eigenvalues, ``s`` and ``4 + s``
    # (1 % allowed for the float32 diagonal), and at 1e6 unknowns the
    # threshold is above the upper one.
    assert all(0.99 * case.shift < q < 4 + case.shift for q in quotients), quotients
    assert breakdown_threshold(1_000_000) > 4 + case.shift


@_breakdown_cases
@pytest.mark.parametrize("jit", [True, False], ids=["jitted", "eager"])
def test_the_default_route_answers_in_float32_where_lineax_breaks_down(case, jit):
    """MADD-ANO-264, fixed for the default call: where lineax's value is
    NaN (the test above), ``backend="auto"`` without a preconditioner
    returns the loop backend's answer -- the same iteration from the same
    start, so the same value and count, bit for bit -- and reports the
    residual of the system as the loop does.  Measured: at 4096 unknowns
    68 iterations and a true residual of 1.014e-4 ``||b||`` against
    ``rtol=1e-4`` (so ``converged=False``, by 1.5 %: the loop stopped on
    its tolerance within the float32 floor of it); at 1e5 unknowns 14
    iterations and 8.64e-5 (``converged=True``).  Jitted, the choice is a
    ``lax.cond``; eager, it is taken in Python."""
    rtol = DEFAULT_ROUTE_RTOL
    kw = dict(jit=jit, rtol=rtol, max_iters=DEFAULT_ROUTE_BUDGET)
    got = _solve(case.shift, case.b(np.float32), **kw)
    loop = _solve(case.shift, case.b(np.float32), backend="loop", **kw)
    assert np.isfinite(got.value).all(), "the default route is NaN again: MADD-ANO-264"
    assert got.true <= 2 * rtol, got.true                 # a usable answer
    # The report is the truth about the system ...
    assert got.reported == pytest.approx(got.true, rel=0.05), (got.reported, got.true)
    assert abs(got.reported / rtol - 1.0) > 1e-4, got.reported    # not at rounding of rtol
    assert got.converged == (got.reported <= rtol), (got.converged, got.reported)
    if case.unknowns == 100_000:
        assert got.converged, got.reported                # 14 % inside the tolerance
    # ... and the answer is the loop backend's.
    assert 2 < got.iters == loop.iters < DEFAULT_ROUTE_BUDGET, (got.iters, loop.iters)
    assert np.array_equal(got.value, loop.value)
    assert got.reported == pytest.approx(loop.reported, rel=1e-5)
    assert got.converged == loop.converged


@pytest.mark.parametrize("shift, times_rtol", [(1e-2, 29), (1e-3, 510)],
                         ids=["condition number 401", "condition number 4001"])
def test_the_fallback_reports_the_residual_of_the_system_not_the_one_the_loop_updates(
        shift, times_rtol):
    """The fallback is new in 0.4.0 and starts honest.  At the default
    ``rtol=1e-6`` in float32, on a smooth right-hand side at 4096 unknowns
    (where lineax breaks down on its first step), the loop stops on its own
    test at a residual it has updated to under 1e-6 ``||b||``, while the
    residual of the system is 2.9e-5 (condition number 401) and 5.1e-4
    (4001).  The default route answers with that value, ``converged=False``
    and the true norm: a usable solution and no claim that it met
    ``rtol``."""
    b = _smooth(LOOP_UNKNOWNS, np.float32)
    got = _solve(shift, b, max_iters=LOOP_BUDGET)          # every default: rtol 1e-6
    assert 2 < got.iters < LOOP_BUDGET                      # the loop, stopped by its own test
    recursive = float(jax.jit(lambda b: _the_iteration_as_released(
        _matvec(shift), b, rtol=DEFAULT_RTOL, max_iters=LOOP_BUDGET)[2])(jnp.asarray(b)))
    assert recursive <= DEFAULT_RTOL * float(np.linalg.norm(b.astype(np.float64)))
    assert got.reported == pytest.approx(got.true, rel=0.05), (got.reported, got.true)
    assert 0.5 * times_rtol < got.reported / DEFAULT_RTOL < 2 * times_rtol, got.reported
    assert not got.converged


def test_the_fallback_starts_from_the_initial_guess():
    """"The same system from the same start": the loop the default route
    falls back to is given ``x0``.  From the solution itself (a float64
    direct solve, rounded to float32) it has nothing to do: no iteration,
    the value is ``x0``, and its residual is within the tolerance.  lineax's
    CG takes no start, and by name still breaks down on this system."""
    case = BREAKDOWN_CASES["a smooth right-hand side at 4096 unknowns"]
    b = case.b(np.float32)
    x0 = _thomas64(b, float(np.float32(2 + case.shift))).astype(np.float32)
    kw = dict(rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET, x0=jnp.asarray(x0))
    got = _solve(case.shift, b, **kw)
    assert got.iters == 0 and np.array_equal(got.value, x0)
    assert got.converged and got.reported <= DEFAULT_ROUTE_RTOL
    assert np.isnan(_solve(case.shift, b, backend="lineax", **kw).value).all()


def test_the_default_route_answers_in_float32_where_every_quotient_passes():
    """The other side of the same arithmetic: white noise at 4096 unknowns
    with ``s = 0.5`` has quotients between 1.7 and 2.5 against a threshold
    of 0.098, and lineax's CG converges (measured: 14 steps, true residual
    8.5e-5).  The default route's answer there is lineax's."""
    case = BreakdownCase(unknowns=4096, shift=0.5, rhs="white noise", fails_on_step=0)
    assert min(_quotients(case, steps=6)) > 10 * breakdown_threshold(case.unknowns)
    kw = dict(rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    got = _solve(case.shift, case.b(np.float32), **kw)
    assert got.converged and np.isfinite(got.value).all()
    assert got.true <= 2 * DEFAULT_ROUTE_RTOL, got.true
    by_name = _solve(case.shift, case.b(np.float32), backend="lineax", **kw)
    assert np.array_equal(got.value, by_name.value) and got.iters == by_name.iters


# Where lineax answers, the default route returns what lineax returns.

def _a_start(n: int, dtype) -> np.ndarray:
    return (0.3 * _white_noise(n, dtype)[::-1]).astype(dtype)


#: ``name: (shift, unknowns, right-hand side, keywords)``.  The first three
#: are the small systems of ``test_iterative_solver.py``'s default-route
#: tests; lineax's value is finite on every one, in float32 and float64
#: (asserted), whether or not it converged.
LINEAX_ANSWERS = {
    "the Laplacian of 32 unknowns, b = 1..32":
        (0.0, 32, lambda n, dt: (np.arange(n) + 1.0).astype(dt), dict(max_iters=500)),
    "the Laplacian of 64 unknowns, b ones":
        (0.0, 64, lambda n, dt: np.ones(n, dt), {}),
    "a zero right-hand side":
        (0.5, 64, lambda n, dt: np.zeros(n, dt), {}),
    "white noise at 4096 unknowns, rtol 1e-4":
        (0.5, 4096, _white_noise, dict(rtol=1e-4, max_iters=500)),
    "the same with an absolute floor":
        (0.5, 4096, lambda n, dt: (1e-3 * _white_noise(n, dt)).astype(dt), dict(atol=1e-5)),
    "the same from a start":
        (0.5, 4096, _white_noise, dict(rtol=1e-5, max_iters=500, x0=_a_start)),
    "condition number 401, stopped by the cap":
        (1e-2, 4096, _white_noise, dict(max_iters=50)),
    "a right-hand side near 1e-20":
        (0.5, 64, lambda n, dt: (1e-20 * _white_noise(n, dt)).astype(dt), {}),
}


@pytest.mark.parametrize("on_mesh", [False, True], ids=["one device", "four devices"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("name", list(LINEAX_ANSWERS))
def test_the_default_route_is_the_lineax_route_bit_for_bit_where_lineax_answers(
        name, dtype, on_mesh):
    """Nothing the default call returned before 0.4.0 changed where lineax's
    value is finite: ``backend="auto"`` and ``backend="lineax"`` (which has
    no fallback and was not touched) return the same ``value``,
    ``converged``, ``iters`` and ``residual_norm``, bit for bit -- jitted
    (where the fallback is the branch of a ``lax.cond`` that is not taken),
    under ``differentiable=True``, and eagerly."""
    shift, n, rhs, kw = LINEAX_ANSWERS[name]
    with _x64(dtype is np.float64):
        kw = {k: (jnp.asarray(v(n, dtype)) if callable(v) else v) for k, v in kw.items()}
        b = rhs(n, dtype)
        where = _four_devices() if on_mesh else {}
        matvec = _matvec_on_a_mesh(shift, where["mesh"]) if on_mesh else _matvec(shift)
        variants = [dict(jit=True), dict(jit=True, differentiable=True)]
        if not on_mesh:
            variants.append(dict(jit=False))
        for variant in variants:
            auto = _fields(matvec, b, **where, **kw, **variant)
            lineax = _fields(matvec, b, backend="lineax", **where, **kw, **variant)
            assert np.isfinite(lineax[0]).all(), (name, variant)   # lineax answered
            assert _same_bits(auto, lineax) == [], (name, variant, _same_bits(auto, lineax))


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
def test_the_loop_backend_solves_the_same_system_in_float32(case):
    """What the default route falls back to, asked for by name:
    ``backend="loop"`` in float32 solves both systems, the true residual
    checked on the host (measured: 1.014e-4 at 4096 unknowns -- the loop
    stopped on its tolerance and the residual of the system is 1.5 % over
    it, so the flag reads False -- and 8.64e-5 at 1e5, True)."""
    got = _solve(case.shift, case.b(np.float32), backend="loop",
                 rtol=DEFAULT_ROUTE_RTOL, max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isfinite(got.value).all()
    assert 0 < got.iters < DEFAULT_ROUTE_BUDGET
    assert got.true <= 2 * DEFAULT_ROUTE_RTOL, got.true
    assert got.reported == pytest.approx(got.true, rel=0.05)
    assert got.converged == (got.reported <= DEFAULT_ROUTE_RTOL)


def _max_rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("on_mesh", [False, True], ids=["one device", "four devices"])
def test_grad_and_jvp_through_a_default_route_solve_that_falls_back_match_a_direct_solve(
        on_mesh):
    """``differentiable=True`` sends the forward solve, the tangent solve
    and the cotangent solve through the same dispatch, so each gets the
    fallback.  At 1e5 unknowns in float32 (``s = 0.5``, an easy system)
    lineax's CG breaks down on all three right-hand sides -- asserted --
    and until 0.4.0 the default call's derivatives were NaN.  They are now
    the derivatives: reverse and forward mode with respect to ``b`` within
    ``4 * rtol`` of a float64 direct solve on the host (measured: 1.3e-4
    and 8.6e-5 at ``rtol=1e-4``), and the derivative of ``sum(x**2)`` with
    respect to the operator's shift, ``-2 x.A^-1 x``, within 1e-3
    (measured: 6e-7).  The same on a mesh of four devices."""
    n, shift, rtol = 100_000, 0.5, DEFAULT_ROUTE_RTOL
    diagonal = float(np.float32(2 + shift))
    b_host = _white_noise(n, np.float32)
    v_host = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    b, v = jnp.asarray(b_host), jnp.asarray(v_host)
    where = _four_devices() if on_mesh else {}

    def operator(s):
        return _matvec_on_a_mesh(s, where["mesh"]) if on_mesh else _matvec(s)

    kw = dict(rtol=rtol, max_iters=DEFAULT_ROUTE_BUDGET, **where)

    def solve(bb, s=shift):
        return sharded_cg(operator(s), bb, differentiable=True, **kw).value

    g = jax.jit(jax.grad(lambda bb: jnp.sum(solve(bb) ** 2)))(b)
    x, t = jax.jit(lambda bb, vv: jax.jvp(solve, (bb,), (vv,)))(b, v)
    d_shift = jax.jit(jax.grad(lambda s, bb: jnp.sum(solve(bb, s) ** 2)))(jnp.float32(shift), b)

    x64 = _thomas64(b_host, diagonal)
    assert _max_rel(x, x64) <= 4 * rtol, _max_rel(x, x64)
    assert _max_rel(g, _thomas64(2 * x64, diagonal)) <= 4 * rtol
    assert _max_rel(t, _thomas64(v_host, diagonal)) <= 4 * rtol
    exact_d_shift = float(-2 * x64 @ _thomas64(x64, diagonal))
    assert float(d_shift) == pytest.approx(exact_d_shift, rel=1e-3)

    # Each of the three solves was one lineax breaks down on.
    by_name = jax.jit(lambda rhs: sharded_cg(operator(shift), rhs, backend="lineax", **kw).value)
    for rhs in (b, v, 2 * x):
        assert np.isnan(np.asarray(by_name(rhs))).all()


def test_under_vmap_each_row_is_its_own_solve_whichever_route_answered_it():
    """Under ``vmap`` the ``cond`` is a select: both solvers run on every
    row and each row keeps the answer of its own route.  Two right-hand
    sides at 4096 unknowns with ``s = 0.01``: the smooth one breaks lineax
    down (the loop answers: 68 iterations) and white noise does not (lineax
    answers: 101 steps, converged).  Each row is, bit for bit, the
    unbatched default-route solve of that right-hand side."""
    shift = 0.01
    rows = np.stack([_smooth(4096, np.float32), _white_noise(4096, np.float32)])

    def solve(b):
        r = sharded_cg(_matvec(shift), b, rtol=DEFAULT_ROUTE_RTOL,
                       max_iters=DEFAULT_ROUTE_BUDGET)
        return r.value, r.converged, r.iters, r.residual_norm

    batched = tuple(np.asarray(a) for a in jax.jit(jax.vmap(solve))(jnp.asarray(rows)))
    one = jax.jit(solve)
    by_name = jax.jit(lambda b: sharded_cg(_matvec(shift), b, backend="lineax",
                                           rtol=DEFAULT_ROUTE_RTOL,
                                           max_iters=DEFAULT_ROUTE_BUDGET).value)
    broke_down = [bool(np.isnan(np.asarray(by_name(jnp.asarray(row)))).all()) for row in rows]
    assert broke_down == [True, False]
    for k, row in enumerate(rows):
        alone = tuple(np.asarray(a) for a in one(jnp.asarray(row)))
        assert np.isfinite(alone[0]).all() and int(alone[2]) > 2
        assert _same_bits(tuple(a[k] for a in batched), alone) == [], k
    assert bool(batched[1][1]) and not np.isnan(batched[0]).any()


@pytest.mark.parametrize("jit", [True, False], ids=["jitted", "eager"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_a_non_finite_right_hand_side_is_still_nan_on_the_default_route(bad, jit):
    """MADD-ANO-155's answer is not the fallback's to change.  A ``b`` with
    a NaN or infinite entry is handed to the backend as zeros, lineax
    returns zeros (finite: no fallback), and the result is NaN in every
    entry, a NaN ``residual_norm`` and ``converged=False`` -- here on the
    system where a finite ``b`` takes the fallback."""
    case = BREAKDOWN_CASES["a smooth right-hand side at 4096 unknowns"]
    b = case.b(np.float32)
    b[7] = bad
    value, converged, _, residual_norm = _fields(
        _matvec(case.shift), b, jit=jit, rtol=DEFAULT_ROUTE_RTOL,
        max_iters=DEFAULT_ROUTE_BUDGET)
    assert np.isnan(value).all() and not bool(converged) and np.isnan(residual_norm)


@pytest.mark.parametrize("jit", [True, False], ids=["jitted", "eager"])
def test_a_system_neither_route_can_solve_is_not_reported_converged(jit):
    """The fallback's report is of the system, so it cannot vouch for what
    the loop could not solve.  The negative of the operator is not positive
    definite: lineax's CG breaks down on it as on the operator itself, and
    the loop diverges.  The default route answers ``converged=False`` and a
    ``residual_norm`` that is not within the tolerance (measured: NaN, on a
    NaN value)."""
    case = BREAKDOWN_CASES["a smooth right-hand side at 4096 unknowns"]
    b = case.b(np.float32)
    plus = _matvec(case.shift)
    value, converged, _, residual_norm = _fields(
        lambda x: -plus(x), b, jit=jit, rtol=DEFAULT_ROUTE_RTOL,
        max_iters=DEFAULT_ROUTE_BUDGET)
    tolerance = DEFAULT_ROUTE_RTOL * float(np.linalg.norm(b.astype(np.float64)))
    assert not bool(converged)
    assert not float(residual_norm) <= tolerance, float(residual_norm)
