"""Derivatives through ``sharded_cg`` and ``sharded_gmres``: refused without
``differentiable=True``, and with it those of the linear solve.

Three behaviours of these STABLE functions, corrected in 0.4.0:

* **MADD-ANO-266.**  A call without ``differentiable=True`` that is
  differentiated (``jax.jvp``, ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``,
  with respect to ``b`` or to a value the operator closes over, on any
  backend, also under ``jit`` and ``vmap``) raises a ``ValueError`` that says
  what to pass.  Until then such a derivative was computed by
  differentiating the iteration: lineax's tangent and transpose solves
  reuse the primal's tolerance, absolute in the primal right-hand side's
  frame, so a tangent or cotangent of another size came back wrong in
  silence (float32, 256 unknowns, unit direction: off 6.4e-4 at ``|b|``
  about 1e3, 0.65 at 1e10; ``sharded_gmres`` returned zeros at 1e10).  A
  call that is not differentiated is untouched (same bits, same lowered
  program), and so is one whose inputs do not depend on what is
  differentiated.
* **MADD-ANO-267.**  With ``differentiable=True`` and an ``x0``, the tangent
  and the cotangent solves start from zero.  They started from the primal's
  ``x0`` rescaled to their own right-hand side: with ``max_iters=0`` the
  derivative WAS ``x0``, and at ``|b|`` about 1e5 with ``x0`` the solution a
  unit-direction JVP was off 2.6e-2 on the CG loop.
* **MADD-ANO-268.**  With ``differentiable=True`` an ``atol`` bounds the
  primal solve only; the tangent and cotangent solves are purely relative,
  so a derivative is linear in its direction.  At ``atol=1e-3`` a direction
  of norm 1e-5 gave a zero derivative on the CG loop and half of it on
  lineax.

The system is the 1-D Dirichlet Laplacian shifted on its diagonal by ``s``
(``(2 + s) x[i] - x[i-1] - x[i+1]``), 64 unknowns, float32; every reference
is a dense float64 solve on the host.  No iteration count is pinned.
"""

# No ``from __future__ import annotations`` here: the device-policy test of
# this directory executes every test module without registering it in
# ``sys.modules``.
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu import iterative_solver
from maddening.cloud.multigpu.iterative_solver import (
    jacobi_preconditioner,
    sharded_cg,
    sharded_gmres,
)

UNKNOWNS = 64
SHIFT = 0.5
#: What a float32 derivative at ``rtol=1e-6`` is held to against float64
#: (measured: 5e-7 to 1e-6 on every route and at every scale).
DERIVATIVE_RTOL = 2e-5


def _matvec(shift):
    def matvec(x):
        left = jnp.concatenate([jnp.zeros((1,), x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.zeros((1,), x.dtype)])
        return (2 + shift) * x - left - right

    return matvec


def _dense64(shift: float = SHIFT) -> np.ndarray:
    """The operator in float64, with the diagonal as float32 holds it."""
    a = np.diag(np.full(UNKNOWNS, float(np.float32(2 + shift))))
    a -= np.diag(np.ones(UNKNOWNS - 1), 1) + np.diag(np.ones(UNKNOWNS - 1), -1)
    return a


def _vector(seed: int, scale: float = 1.0) -> np.ndarray:
    return (scale * np.random.default_rng(seed).standard_normal(UNKNOWNS)).astype(np.float32)


def _jacobi():
    return jacobi_preconditioner(jnp.full((UNKNOWNS,), 2 + SHIFT, jnp.float32))


#: Every route of the two solvers: the function, and the keywords that
#: select the route.  ``"auto"`` with a preconditioner is the loop backend.
ROUTES = {
    "cg default": (sharded_cg, {}),
    "cg lineax": (sharded_cg, {"backend": "lineax"}),
    "cg loop": (sharded_cg, {"backend": "loop"}),
    "cg auto with Jacobi": (sharded_cg, {"preconditioner": "jacobi"}),
    "gmres default": (sharded_gmres, {}),
    "gmres lineax": (sharded_gmres, {"backend": "lineax"}),
    "gmres loop": (sharded_gmres, {"backend": "loop"}),
    "gmres auto with Jacobi": (sharded_gmres, {"preconditioner": "jacobi"}),
}
_routes = pytest.mark.parametrize("route", list(ROUTES))


def _call(route: str, b, *, shift=SHIFT, **kw):
    solver, selects = ROUTES[route]
    selects = dict(selects)
    if selects.get("preconditioner") == "jacobi":
        selects["preconditioner"] = _jacobi()
    return solver(_matvec(shift), b, **selects, **kw)


def _max_rel(got, want) -> float:
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _bits(result) -> tuple:
    return tuple(np.asarray(f).tobytes() for f in (
        result.value, result.converged, result.iters, result.residual_norm))


# ---------------------------------------------------------------------------
# MADD-ANO-266: without differentiable=True a derivative is refused
# ---------------------------------------------------------------------------

REFUSAL = "was differentiated without differentiable=True"


@_routes
def test_a_call_without_the_flag_refuses_a_derivative_with_respect_to_b(route):
    b, c = jnp.asarray(_vector(0)), jnp.asarray(_vector(1))
    value = lambda bb: _call(route, bb).value                      # noqa: E731
    loss = lambda bb: jnp.vdot(c, _call(route, bb).value)          # noqa: E731
    with pytest.raises(ValueError, match=REFUSAL):
        jax.jvp(value, (b,), (c,))
    with pytest.raises(ValueError, match=REFUSAL):
        jax.grad(loss)(b)
    with pytest.raises(ValueError, match=REFUSAL):
        jax.jacfwd(value)(b)
    with pytest.raises(ValueError, match=REFUSAL):
        jax.jacrev(value)(b)


@_routes
def test_a_call_without_the_flag_refuses_a_derivative_with_respect_to_the_operator(route):
    """The operator closes over the shift, and the shift is differentiated."""
    b, c = jnp.asarray(_vector(0)), jnp.asarray(_vector(1))
    loss = lambda s: jnp.vdot(c, _call(route, b, shift=s).value)   # noqa: E731
    with pytest.raises(ValueError, match=REFUSAL):
        jax.grad(loss)(SHIFT)
    with pytest.raises(ValueError, match=REFUSAL):
        jax.jvp(loss, (SHIFT,), (1.0,))


@pytest.mark.parametrize("field", ["value", "residual_norm"])
@pytest.mark.parametrize("route", ["cg default", "cg loop", "gmres default"])
def test_the_refusal_holds_under_jit_and_vmap_and_for_either_floating_field(route, field):
    b = jnp.asarray(_vector(0))
    loss = lambda bb: jnp.sum(getattr(_call(route, bb), field))    # noqa: E731
    with pytest.raises(ValueError, match=REFUSAL):
        jax.jit(jax.grad(loss))(b)
    with pytest.raises(ValueError, match=REFUSAL):
        jax.vmap(jax.grad(loss))(jnp.stack([b, 2 * b]))
    with pytest.raises(ValueError, match=REFUSAL):
        jax.grad(jax.jit(loss))(b)


@pytest.mark.parametrize("solver", [sharded_cg, sharded_gmres])
def test_the_refusal_names_the_function_and_says_what_to_do(solver):
    b = jnp.asarray(_vector(0))
    with pytest.raises(ValueError) as raised:
        jax.grad(lambda bb: jnp.sum(solver(_matvec(SHIFT), bb).value))(b)
    message = str(raised.value)
    assert message.startswith(f"{solver.__name__} was differentiated without")
    for said in ("Pass differentiable=True", ".iters reads -1",
                 "stop the gradient on what goes INTO the call",
                 "a stop_gradient on the result comes too late"):
        assert said in message, said


@_routes
def test_a_stop_gradient_on_the_result_is_too_late_and_one_on_the_inputs_is_not(route):
    """The solve as a constant of a differentiation: the gradient is
    stopped on ``b`` before the call.  The loss ``b . x(stop_gradient(b))``
    then has the gradient ``x``."""
    b = jnp.asarray(_vector(0))
    with pytest.raises(ValueError, match=REFUSAL):
        jax.grad(lambda bb: jnp.vdot(bb, jax.lax.stop_gradient(_call(route, bb).value)))(b)
    gradient = jax.grad(
        lambda bb: jnp.vdot(bb, _call(route, jax.lax.stop_gradient(bb)).value))(b)
    assert np.asarray(gradient).tobytes() == np.asarray(_call(route, b).value).tobytes()


@pytest.mark.parametrize("route", ["cg default", "cg loop", "gmres loop"])
def test_a_value_that_does_not_depend_on_what_is_differentiated_is_not_refused(route):
    """The solve is in the differentiated function, and nothing it is given
    depends on the differentiated argument: ``d/dt (t * sum(x)) = sum(x)``,
    eagerly, under ``jit``, and with ``b`` an argument that is not
    differentiated."""
    b = jnp.asarray(_vector(0))
    total = float(jnp.sum(_call(route, b).value))
    loss = lambda t, bb: t * jnp.sum(_call(route, bb).value)       # noqa: E731
    assert float(jax.grad(loss)(2.0, b)) == pytest.approx(total, rel=1e-6)
    assert float(jax.jit(jax.grad(loss))(2.0, b)) == pytest.approx(total, rel=1e-6)
    assert float(jax.jvp(lambda t: loss(t, b), (2.0,), (1.0,))[1]) == pytest.approx(
        total, rel=1e-6)


@_routes
def test_a_call_that_is_not_differentiated_has_the_same_bits_and_the_same_program(
        route, monkeypatch):
    """What the refusal adds to a call nobody differentiates: nothing.  The
    lowered program and the four fields are compared with the refusal
    taken out."""
    b = jnp.asarray(_vector(0))

    def solve(bb):
        r = _call(route, bb)
        return r.value, r.converged, r.iters, r.residual_norm

    with_refusal = jax.jit(solve).lower(b).as_text()
    fields = [np.asarray(f).tobytes() for f in jax.jit(solve)(b)]
    monkeypatch.setattr(iterative_solver, "_refusing_a_derivative",
                        lambda result, name: result)
    assert jax.jit(solve).lower(b).as_text() == with_refusal
    assert [np.asarray(f).tobytes() for f in jax.jit(solve)(b)] == fields


# ---------------------------------------------------------------------------
# differentiable=True: the derivative of the linear solve
# ---------------------------------------------------------------------------


def _derivatives(route: str, b, v, c, **kw) -> tuple:
    """``(value, jvp with respect to b along v, gradient of c . x)``."""
    solve = lambda bb: _call(route, bb, differentiable=True, **kw).value   # noqa: E731
    value, tangent = jax.jit(lambda bb, vv: jax.jvp(solve, (bb,), (vv,)))(b, v)
    gradient = jax.jit(jax.grad(lambda bb: jnp.vdot(c, solve(bb))))(b)
    return np.asarray(value), np.asarray(tangent), np.asarray(gradient)


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10], ids=["1e-10", "1", "1e10"])
@_routes
def test_with_the_flag_a_derivative_is_right_whatever_the_size_of_b(route, scale):
    """The derivative of a linear solve does not depend on ``|b|``; the
    direction and the loss weight are of unit size."""
    a = _dense64()
    b_host, v_host, c_host = _vector(0, scale), _vector(1), _vector(2)
    _, tangent, gradient = _derivatives(
        route, jnp.asarray(b_host), jnp.asarray(v_host), jnp.asarray(c_host))
    assert _max_rel(tangent, np.linalg.solve(a, v_host.astype(np.float64))) < DERIVATIVE_RTOL
    assert _max_rel(gradient, np.linalg.solve(a.T, c_host.astype(np.float64))) < DERIVATIVE_RTOL


@pytest.mark.parametrize("route", ["cg loop", "cg auto with Jacobi"])
def test_with_an_x0_the_tangent_and_cotangent_solves_start_from_zero(route):
    """MADD-ANO-267.  With ``max_iters=0`` the CG loop returns its start:
    the value is ``x0``, and the derivatives are the starts of the tangent
    and the cotangent solve.  They are zero; they were ``x0``."""
    b, v, c = (jnp.asarray(_vector(k)) for k in range(3))
    x0 = jnp.asarray(_vector(3))
    value, tangent, gradient = _derivatives(route, b, v, c, x0=x0, max_iters=0)
    assert value.tobytes() == np.asarray(x0).tobytes()
    assert not np.any(tangent)
    assert not np.any(gradient)


def _cosine(u, w) -> float:
    u, w = np.asarray(u, np.float64), np.asarray(w, np.float64)
    return float(abs(u @ w) / (np.linalg.norm(u) * np.linalg.norm(w)))


@pytest.mark.parametrize("preconditioned", [False, True], ids=["plain", "Jacobi"])
def test_with_an_x0_the_gmres_tangent_and_cotangent_solves_start_from_zero(preconditioned):
    """MADD-ANO-267 on the GMRES loop, which always runs a cycle: with
    ``restart=1`` a cycle is one step along its start's residual.  From a
    zero start that residual is the right-hand side, so the tangent is a
    multiple of the direction and the gradient a multiple of the loss
    weight (the Jacobi preconditioner here is a multiple of the identity).
    From the primal's ``x0`` the tangent was not."""
    b, v, c = (jnp.asarray(_vector(k)) for k in range(3))
    x0 = jnp.asarray(_vector(3))
    route = "gmres auto with Jacobi" if preconditioned else "gmres loop"
    _, tangent, gradient = _derivatives(route, b, v, c, x0=x0, restart=1, max_iters=1)
    assert np.any(tangent) and np.any(gradient)
    assert _cosine(tangent, v) > 1 - 1e-6
    assert _cosine(gradient, c) > 1 - 1e-6
    assert _cosine(x0, v) < 0.5     # the start is not along the direction


@pytest.mark.parametrize("start", ["the solution", "noise of the size of b"])
@_routes
def test_with_an_x0_a_derivative_is_as_accurate_as_without(route, start):
    """MADD-ANO-267 at scale: ``|b|`` about 1e5, unit direction.  Started
    from the primal's ``x0`` the CG loop's tangent solve stopped on its
    recursive residual with a true error of 2.6e-2 (JVP) and 1.3e-2
    (gradient)."""
    a = _dense64()
    b_host, v_host, c_host = _vector(0, 1e5), _vector(1), _vector(2)
    x0_host = (np.linalg.solve(a, b_host.astype(np.float64)) if start == "the solution"
               else _vector(3, 1e5)).astype(np.float32)
    _, tangent, gradient = _derivatives(
        route, jnp.asarray(b_host), jnp.asarray(v_host), jnp.asarray(c_host),
        x0=jnp.asarray(x0_host))
    assert _max_rel(tangent, np.linalg.solve(a, v_host.astype(np.float64))) < DERIVATIVE_RTOL
    assert _max_rel(gradient, np.linalg.solve(a.T, c_host.astype(np.float64))) < DERIVATIVE_RTOL


@pytest.mark.parametrize("keywords", ["x0", "atol", "x0 and atol"])
@_routes
def test_with_an_x0_or_an_atol_the_value_is_the_ordinary_calls_bit_for_bit(route, keywords):
    """The primal solve of ``differentiable=True`` is the ordinary call's,
    from the caller's ``x0`` and with the caller's ``atol``; the solve that
    attaches the derivative adds an exact zero to it."""
    b = jnp.asarray(_vector(0))
    kw = {}
    if "x0" in keywords:
        kw["x0"] = jnp.asarray(_vector(3))
    if "atol" in keywords:
        kw["atol"] = 1e-4
    ordinary = jax.jit(lambda bb: _call(route, bb, **kw).value)(b)
    with_flag = jax.jit(lambda bb: _call(route, bb, differentiable=True, **kw).value)(b)
    assert np.all(np.isfinite(np.asarray(with_flag)))
    assert np.asarray(with_flag).tobytes() == np.asarray(ordinary).tobytes()


#: A power of two: scaling a right-hand side by it scales every number of
#: its solve exactly (the solvers work in a power-of-two frame of it).
SMALL = 2.0 ** -17


@_routes
def test_with_an_atol_a_derivative_is_linear_in_its_direction(route):
    """MADD-ANO-268.  ``atol=1e-3`` with ``|b|`` about 16: a direction (and
    a loss weight) 2**-17 times smaller, norm about 6e-5, is under the
    floor.  Its derivative is 2**-17 times the derivative of the direction
    itself, bit for bit, and right against float64.  It was all zero on
    the CG loop and half the size on lineax."""
    a = _dense64()
    b_host, v_host, c_host = _vector(0, 2.0), _vector(1), _vector(2)
    b, v, c = jnp.asarray(b_host), jnp.asarray(v_host), jnp.asarray(c_host)
    _, tangent, gradient = _derivatives(route, b, v, c, atol=1e-3)
    _, small_tangent, small_gradient = _derivatives(route, b, SMALL * v, SMALL * c, atol=1e-3)
    assert np.any(small_tangent) and np.any(small_gradient)
    assert small_tangent.tobytes() == (np.float32(SMALL) * tangent).tobytes()
    assert small_gradient.tobytes() == (np.float32(SMALL) * gradient).tobytes()
    assert _max_rel(tangent, np.linalg.solve(a, v_host.astype(np.float64))) < DERIVATIVE_RTOL
    assert _max_rel(gradient, np.linalg.solve(a.T, c_host.astype(np.float64))) < DERIVATIVE_RTOL


@pytest.mark.parametrize("route", ["cg loop", "gmres loop"])
def test_an_atol_above_the_right_hand_side_stops_the_primal_solve_and_not_its_derivative(route):
    """The floor is the caller's statement about the primal ``b``: above
    ``||b||`` the loop backends accept the zero start (CG at once, GMRES
    after its first cycle, which solves this system), and the tangent solve
    is still carried to ``rtol``."""
    a = _dense64()
    b_host, v_host, c_host = _vector(0), _vector(1), _vector(2)
    value, tangent, gradient = _derivatives(
        route, jnp.asarray(b_host), jnp.asarray(v_host), jnp.asarray(c_host), atol=1e3)
    if route == "cg loop":
        assert not np.any(value)
    assert _max_rel(tangent, np.linalg.solve(a, v_host.astype(np.float64))) < DERIVATIVE_RTOL
    assert _max_rel(gradient, np.linalg.solve(a.T, c_host.astype(np.float64))) < DERIVATIVE_RTOL


@_routes
def test_with_an_x0_and_an_atol_the_derivative_with_respect_to_the_operator_is_right(route):
    """``d/ds c . A(s)^-1 b = -c . A^-1 A^-1 b`` (``dA/ds`` is the
    identity), forward and reverse, through the route that keeps the
    primal's ``x0`` and ``atol`` out of the derivative solves."""
    a = _dense64()
    b_host, c_host = _vector(0), _vector(2)
    b, c, x0 = jnp.asarray(b_host), jnp.asarray(c_host), jnp.asarray(_vector(3))
    loss = lambda s: jnp.vdot(c, _call(                            # noqa: E731
        route, b, shift=s, differentiable=True, x0=x0, atol=1e-7).value)
    x = np.linalg.solve(a, b_host.astype(np.float64))
    want = -float(np.linalg.solve(a.T, c_host.astype(np.float64)) @ x)
    assert float(jax.jit(jax.grad(loss))(SHIFT)) == pytest.approx(want, rel=1e-4)
    assert float(jax.jit(lambda s: jax.jvp(loss, (s,), (1.0,))[1])(SHIFT)) == pytest.approx(
        want, rel=1e-4)


@pytest.mark.parametrize("route", ["cg loop", "gmres loop", "cg lineax"])
def test_with_an_x0_and_an_atol_a_second_derivative_is_right(route):
    """``d2/ds2 c . A(s)^-1 b = 2 c . A^-1 A^-1 A^-1 b``.  The value is
    ``x_p + A^-1 (r - stop_gradient(r))`` with ``x_p`` a constant: as a
    function of ``s`` that is ``A(s)^-1 (b - c0)`` with ``c0`` the residual
    the primal left, so its derivatives of every order are those of a
    solve."""
    a = _dense64()
    b_host, c_host = _vector(0), _vector(2)
    b, c, x0 = jnp.asarray(b_host), jnp.asarray(c_host), jnp.asarray(_vector(3))
    loss = lambda s: jnp.vdot(c, _call(                            # noqa: E731
        route, b, shift=s, differentiable=True, x0=x0, atol=1e-7).value)
    inverse = np.linalg.inv(a)
    want = 2 * float(c_host.astype(np.float64) @ inverse @ inverse @ inverse
                     @ b_host.astype(np.float64))
    assert float(jax.jit(jax.grad(jax.grad(loss)))(SHIFT)) == pytest.approx(want, rel=1e-3)


@pytest.mark.parametrize("route", ["cg loop", "cg auto with Jacobi", "gmres auto with Jacobi"])
def test_the_start_and_the_preconditioner_carry_no_derivative(route):
    """The solution does not depend on where the iteration starts or on
    what preconditions it.  Both are differentiated here, in reverse mode
    through the loop backends, whose ``while_loop`` has no transpose: the
    primal solve is a constant of the differentiation, and the gradient
    with respect to the scale of ``x0`` and of the preconditioner is zero."""
    b, c = jnp.asarray(_vector(0)), jnp.asarray(_vector(2))
    x0 = jnp.asarray(_vector(3))
    solver, selects = ROUTES[route]

    def loss(t):
        kw = {"backend": selects.get("backend", "auto")}
        if "preconditioner" in selects:
            kw["preconditioner"] = jacobi_preconditioner(
                t * jnp.full((UNKNOWNS,), 2 + SHIFT, jnp.float32))
        return jnp.vdot(c, solver(_matvec(SHIFT), b, x0=t * x0, differentiable=True,
                                  **kw).value)

    assert float(jax.jit(jax.grad(loss))(1.0)) == 0.0
    assert float(jax.jvp(loss, (1.0,), (1.0,))[1]) == 0.0
