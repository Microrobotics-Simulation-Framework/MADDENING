"""``ift_linear_solve`` never returns a value that is not the solution in
silence, whatever floor its stopping test is given.

lineax stops a CG or GMRES solve where ``max_i |r_i| / (floor + rtol |b_i|)``
(and the same quotient of the update) is not above one.  With a floor of zero
the quotient is ``0 / 0 = NaN`` at every entry where the residual and the
right-hand side are both exactly zero -- every entry the Krylov space of a
local right-hand side behind a stencil has not reached yet -- the maximum is
NaN, ``NaN > 1`` is False, and lineax stops and reports success, so its
``throw=True`` does not fire.  The same for a floor under the dtype's smallest
normal number (XLA's CPU arithmetic flushes it), and a negative or NaN floor
stops lineax at once.  lineax 0.0.7 and 0.1.1 alike, CG and GMRES alike.
Until 0.4.0's fix ``ift_linear_solve`` returned such a value: measured in
float32 behind ``2.5 x[i] - x[i-1] - x[i+1]`` (64 unknowns, ``b`` one unit
vector), CG with ``atol=0.0``, ``rtol=0`` or ``rtol=1e-44`` at a relative
error of 0.53, and GMRES (restart 50) behind ``2.001 x[i] - ...`` on 128
unknowns at 0.36; the derivative solves the same.

Now, where the floor (``atol``, or ``rtol * max|rhs|`` without one, in the
frame of ``rhs``) is not a normal positive number, the solve is judged by the
norm rule of ``sharded_cg``, ``||rhs - A x|| <= max(atol, rtol * ||rhs||)``,
and a value that does not meet it raises on lineax's channel for a failed
solve.  Where the floor is normal nothing is added (``atol=None`` with a
concrete ``rtol`` of at least twice the smallest normal number, the coupling
adjoint's form) or the value passes through with its bits.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import solver_utils
from maddening.core._pow2_frame import pow2_frame
from maddening.core.coupling import _ift
from maddening.core.solver_utils import ift_linear_solve

#: A distinctive part of what the rule raises.
RAISED = "floor of lineax's stopping test"


def _operator(diagonal):
    def matvec(x):
        zero = jnp.zeros((1,), x.dtype)
        return (diagonal * x - jnp.concatenate([zero, x[:-1]])
                - jnp.concatenate([x[1:], zero]))
    return matvec


def _unit(n, scale=1.0):
    e0 = np.zeros(n, np.float32)
    e0[0] = np.float32(scale)
    return e0


#: ``(diagonal, unknowns)`` of a system on which each solver stops on its
#: ``0 / 0`` far from the solution: CG after one step (relative error 0.53);
#: GMRES after its first cycle of 50, which on 128 unknowns behind a diagonal
#: of 2.001 has not reached the far entries (0.36).
SYSTEMS = {"cg": (2.5, 64), "gmres": (2.001, 128)}
#: Every tolerance field given a floor that is not a normal positive number:
#: zero, under the smallest normal (directly, or once rescaled with a large
#: right-hand side), negative and NaN.
NOT_NORMAL = {
    "atol=0.0": ({"atol": 0.0}, 1.0),
    "rtol=0": ({"rtol": 0.0}, 1.0),
    "rtol=0 and atol=0.0": ({"rtol": 0.0, "atol": 0.0}, 1.0),
    "rtol=1e-44, which float32 flushes": ({"rtol": 1e-44}, 1.0),
    "atol=1e-12 beside a b of 2^100, which the frame makes subnormal": ({"atol": 1e-12}, 2.0 ** 100),
    "atol=-1": ({"atol": -1.0}, 1.0),
    "atol=NaN": ({"atol": float("nan")}, 1.0),
    "rtol=NaN": ({"rtol": float("nan")}, 1.0),
}


@pytest.mark.parametrize("tolerance", list(NOT_NORMAL))
@pytest.mark.parametrize("solver", list(SYSTEMS))
def test_a_floor_that_is_not_normal_raises_where_lineax_stopped_short(solver, tolerance):
    """Under ``jit``.  Until the fix every case returned a value 0.36 to
    1.0 off (relative error) with no error."""
    diagonal, n = SYSTEMS[solver]
    kw, scale = NOT_NORMAL[tolerance]
    solve = jax.jit(lambda b: ift_linear_solve(_operator(diagonal), b, solver=solver, **kw))
    with pytest.raises(Exception, match=RAISED):
        jax.block_until_ready(solve(jnp.asarray(_unit(n, scale))))


@pytest.mark.parametrize("tolerance", ["atol=0.0", "rtol=0"])
@pytest.mark.parametrize("solver", list(SYSTEMS))
def test_the_same_when_the_call_is_not_jitted(solver, tolerance):
    diagonal, n = SYSTEMS[solver]
    kw, _ = NOT_NORMAL[tolerance]
    with pytest.raises(Exception, match=RAISED):
        jax.block_until_ready(ift_linear_solve(
            _operator(diagonal), jnp.asarray(_unit(n)), solver=solver, **kw))


def test_a_floor_of_zero_that_is_traced_raises_too_and_a_solvable_one_is_solved():
    """The tolerances are arguments of the jitted function, so nothing is
    known of the floor when the call is traced.  On white noise every entry
    of ``b`` is non-zero, lineax's test with ``atol=0.0`` is the stricter
    entrywise one, and the solve meets the norm rule."""
    mv = _operator(2.5)
    solve = jax.jit(lambda b, rtol, atol: ift_linear_solve(mv, b, solver="cg", rtol=rtol, atol=atol))
    relative = jax.jit(lambda b, rtol: ift_linear_solve(mv, b, solver="cg", rtol=rtol))
    with pytest.raises(Exception, match=RAISED):
        jax.block_until_ready(solve(jnp.asarray(_unit(64)), jnp.float32(1e-6), jnp.float32(0.0)))
    with pytest.raises(Exception, match=RAISED):
        jax.block_until_ready(relative(jnp.asarray(_unit(64)), jnp.float32(0.0)))
    noise = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    want = np.linalg.solve(np.asarray(jax.jacfwd(mv)(jnp.zeros(64)), np.float64), noise)
    got = np.asarray(solve(jnp.asarray(noise), jnp.float32(1e-6), jnp.float32(0.0)), np.float64)
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-5


@pytest.mark.parametrize("mode", ["grad", "jvp"])
def test_the_derivative_solves_are_held_to_the_same_rule(mode):
    """With an explicit ``atol=0.0`` the tangent and adjoint solves are
    handed a floor of zero too; a cotangent or tangent that is one unit
    vector made them stop on ``0 / 0``.  Until the fix: a derivative 0.53
    off (relative error), returned with no error, eagerly and under
    ``jit``."""
    mv = _operator(2.5)
    b = jnp.asarray(np.random.default_rng(0).standard_normal(64).astype(np.float32))
    unit = jnp.asarray(_unit(64))
    if mode == "grad":
        derivative = jax.jit(jax.grad(lambda bb: ift_linear_solve(mv, bb, solver="cg", atol=0.0)[0]))
    else:
        derivative = jax.jit(lambda bb: jax.jvp(
            lambda c: ift_linear_solve(mv, c, solver="cg", atol=0.0), (bb,), (unit,))[1])
    with pytest.raises(Exception, match=RAISED):
        jax.block_until_ready(derivative(b))


@pytest.mark.parametrize("solver", list(SYSTEMS))
def test_an_exact_solve_meets_every_tolerance(solver):
    """A zero right-hand side is solved by the zero start, exactly: with
    ``rtol=0`` the floor is zero, and with ``rtol=inf`` it is ``inf * 0``,
    NaN; both are judged, and a residual of exactly zero meets them."""
    diagonal, n = SYSTEMS[solver]
    for kw in ({"rtol": 0.0}, {"rtol": float("inf")}, {"atol": 0.0}):
        got = jax.jit(lambda b: ift_linear_solve(_operator(diagonal), b, solver=solver, **kw))(
            jnp.zeros(n, jnp.float32))
        assert np.array_equal(np.asarray(got), np.zeros(n, np.float32)), kw


def _solve_as_before_the_rule(operator_fn, rhs, *, solver, rtol, atol):
    """``ift_linear_solve``'s Krylov path as it was before the rule: the
    same lineax calls, with nothing after them."""
    import lineax as lx

    n = int(rhs.shape[0])

    def framed_solve(mv, b):
        p = pow2_frame(b)
        finite = jnp.all(jnp.isfinite(b))
        b_hat = jnp.where(finite, b * p, jnp.zeros_like(b))
        atol_hat = rtol * jnp.max(jnp.abs(b_hat)) if atol is None else atol * p
        shape = jax.eval_shape(lambda: b)
        if solver == "cg":
            op = lx.FunctionLinearOperator(
                mv, shape, tags=(lx.positive_semidefinite_tag, lx.symmetric_tag))
            solver_obj = lx.CG(rtol=rtol, atol=atol_hat, max_steps=max(4 * n, 200))
        else:
            op = lx.FunctionLinearOperator(mv, shape)
            restart = min(n, 50)
            solver_obj = lx.GMRES(rtol=rtol, atol=atol_hat, restart=restart,
                                  max_steps=max(4 * restart, 100))
        x = lx.linear_solve(op, b_hat, solver=solver_obj, options={}).value / p
        return jnp.where(finite, x, jnp.full_like(x, jnp.nan))

    if solver == "cg":
        return jax.lax.custom_linear_solve(operator_fn, rhs, framed_solve, symmetric=True)
    return jax.lax.custom_linear_solve(operator_fn, rhs, framed_solve, transpose_solve=framed_solve)


#: Floors that are normal positive numbers, in each of the three forms: the
#: default (the rule is not in the program), an explicit positive ``atol``
#: (the wavelet node's masked CG passes ``rtol=1e-6, atol=1e-8``; the rule is
#: in the program and lets the value through), and ``atol=0.0`` on a
#: right-hand side with no zero entry (the rule is in the program, judges,
#: and the solve meets it).  GMRES at ``rtol=1e-3`` where an ``atol`` is
#: given: at ``1e-6`` lineax's entrywise test without the relative floor
#: stagnates on this system in float32 (lineax's own error, before and after
#: the rule alike).
NORMAL = {"default": {"cg": {"rtol": 1e-6, "atol": None}, "gmres": {"rtol": 1e-6, "atol": None}},
          "an explicit positive atol": {"cg": {"rtol": 1e-6, "atol": 1e-8},
                                        "gmres": {"rtol": 1e-3, "atol": 1e-8}},
          "atol=0.0 on white noise": {"cg": {"rtol": 1e-6, "atol": 0.0},
                                      "gmres": {"rtol": 1e-3, "atol": 0.0}}}


@pytest.mark.parametrize("tolerance", list(NORMAL))
@pytest.mark.parametrize("solver", ["cg", "gmres"])
def test_a_floor_that_is_normal_keeps_the_bits_of_the_solve_and_its_derivatives(solver, tolerance):
    """The value, a JVP and a gradient, under ``jit``, bit for bit those of
    the program before the rule (written out above)."""
    kw = NORMAL[tolerance][solver]
    mv = _operator(2.5)
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(64).astype(np.float32))
    d = jnp.asarray(rng.standard_normal(64).astype(np.float32))
    def solve(c):
        return ift_linear_solve(mv, c, solver=solver, **kw)

    def before(c):
        return _solve_as_before_the_rule(mv, c, solver=solver, **kw)

    for name, of in (("value", lambda f: f),
                     ("jvp", lambda f: lambda c: jax.jvp(f, (c,), (d,))[1]),
                     ("grad", lambda f: jax.grad(lambda c: jnp.vdot(d, f(c))))):
        now = np.asarray(jax.jit(of(solve))(b))
        then = np.asarray(jax.jit(of(before))(b))
        assert now.tobytes() == then.tobytes(), name


def test_the_rule_is_not_in_the_program_where_the_floor_is_normal_for_every_rhs(monkeypatch):
    """The default call of ``ift_linear_solve`` and the coupling adjoint's
    solve (``coupling._ift._ift_linear_solve``, whose floor is the fixed
    positive ``rtol * max|b_hat|`` with ``rtol = max(1e-6, 100 eps)``) do
    not reach the rule: with the rule replaced by a function that raises
    they return what they return without it, bit for bit.  The replacement
    is shown to take effect on a call with ``atol=0.0``."""
    mv = _operator(2.5)
    b = jnp.asarray(np.random.default_rng(0).standard_normal(64).astype(np.float32))
    adjoint = np.asarray(_ift._ift_linear_solve(mv, b, "gmres"))
    defaults = {s: np.asarray(ift_linear_solve(mv, b, solver=s)) for s in ("cg", "gmres")}

    def reached(*args, **kwargs):
        raise AssertionError("the rule was reached")

    monkeypatch.setattr(solver_utils, "_judged_by_the_norm_test_where_the_floor_is_not_normal", reached)
    with pytest.raises(AssertionError, match="the rule was reached"):
        ift_linear_solve(mv, b, solver="cg", atol=0.0)
    assert np.asarray(_ift._ift_linear_solve(mv, b, "gmres")).tobytes() == adjoint.tobytes()
    for s, value in defaults.items():
        assert np.asarray(ift_linear_solve(mv, b, solver=s)).tobytes() == value.tobytes(), s
    for dtype in (jnp.float32, jnp.float64):
        adjoint_rtol = max(1e-6, 100.0 * float(jnp.finfo(dtype).eps))
        assert solver_utils._floor_is_normal_for_every_rhs(adjoint_rtol, None, dtype)
    tiny = float(np.finfo(np.float32).tiny)
    # ``max|b_hat|`` can be 0.5: an ``rtol`` under twice the smallest normal
    # number can give a floor under it, so the rule stays in the program.
    assert not solver_utils._floor_is_normal_for_every_rhs(1.5 * tiny, None, jnp.float32)
    assert solver_utils._floor_is_normal_for_every_rhs(2.0 * tiny, None, jnp.float32)
    assert not solver_utils._floor_is_normal_for_every_rhs(1e-6, 1e-8, jnp.float32)
