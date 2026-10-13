"""``jax.grad`` with respect to a moving geometry is the derivative of the step.

A geometry-dependent mapping reads its geometry from node state, so the
geometry is differentiated like any other state: through a plain step,
through the fixed point of a coupling group (where a value the pass closes
over is a constant of the implicit-function rule and a member's own field
is part of the iterate), and through the sub-steps of a sub-cycled member.
That holds only while the geometry is read *inside* the pass, from the
same dictionaries and tracers the value is read from.  Reading it from
the graph's host state, or behind a ``stop_gradient``, leaves every state
exactly right and the gradient exactly wrong, so nothing but a gradient
oracle sees it.

Every case here is the two-body graph of
:mod:`tests.property.geometry_graphs` in float64 (``jax_enable_x64``),
three steps in a ``lax.scan``, jitted, with a geometry that depends on the
coupling iterate.  ``jax.grad`` of a loss on the outputs is compared with
central differences (step ``1e-6`` of the grid spacing for positions,
``1e-6`` for matrices and scalars), relative error below ``1e-6``, with
respect to:

* the **initial geometry** of each geometry edge's holder;
* the **rate** that moves it, a parameter;
* a **value-side parameter**, as the control.

The paths: a plain step with a forward and with a back edge; a group under
``solver="ift"`` with the GMRES and the dense linear solve, Gauss-Seidel
and Jacobi, with and without acceleration, under ``solver="fori"`` and at
``max_iterations=1``; a sub-cycled group under linear and constant
interpolation whose sub-cycled member is the target of one geometry edge
and the source of the other.  Both anchors; both modes of the library's
``multilinear_grid`` kind (its points start at index fractions between 0.2
and 0.8, and the test asserts that no step carries one within 0.02 of a
lattice plane, so no difference straddles a kink) and the test kind whose
geometry is the matrix.

Forward mode (``jax.jacfwd``) is taken once per path: it is what an
integer tracer carried into a coupling pass breaks.  One float32 run of
the plain path is held to the float64 gradient, relative ``1e-3``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.property import geometry_graphs as gg
from tests.property.geometry_graphs import DT, case

RELATIVE = 1e-6
_TIGHT = dict(max_iterations=400, tolerance=1e-13)
_SUB = dict(_TIGHT, subcycling=True)

P_HOLDS = dict(down="target", up="source")
F_HOLDS = dict(down="source", up="target")
TARGETS = dict(down="target", up="target")
SOURCES = dict(down="source", up="source")


def _f64(label, **kw):
    return case(label, dtype="float64", adv=0.3, **kw)


#: ``(case, also take jacfwd)``
PER_PUSH = [
    (_f64("plain: forward and back edge, multilinear", kind="multilinear", **P_HOLDS), True),
    (_f64("plain: back and forward edge, matrix", order=("P", "F"), **F_HOLDS), False),
    (_f64("group: Gauss-Seidel, GMRES, matrix", group=_TIGHT, **TARGETS), False),
]
SLOW = [
    (_f64("plain: back and forward edge, multilinear", kind="multilinear", order=("P", "F"),
          **F_HOLDS), False),
    (_f64("plain: forward and back edge, matrix", **SOURCES), True),
    (_f64("plain: multilinear 2-D", kind="multilinear", d=2, **TARGETS), False),
    (_f64("group: Gauss-Seidel, GMRES, multilinear", kind="multilinear", group=_TIGHT,
          **P_HOLDS), True),
    (_f64("group: Jacobi, dense, multilinear", kind="multilinear", **F_HOLDS,
          group=dict(_TIGHT, iteration_mode="jacobi", linear_solver="dense")), False),
    (_f64("group: dense, matrix", group=dict(_TIGHT, linear_solver="dense"), **SOURCES), False),
    (_f64("group: Aitken, multilinear", kind="multilinear", **SOURCES,
          group=dict(_TIGHT, acceleration="aitken")), False),
    (_f64("group: IQN-ILS, Jacobi, matrix", **P_HOLDS,
          group=dict(_TIGHT, acceleration="iqn-ils", iteration_mode="jacobi")), False),
    (_f64("group: fori, multilinear", kind="multilinear", **TARGETS,
          group=dict(max_iterations=150, tolerance=1e-13, solver="fori")), False),
    (_f64("group: one pass, multilinear", kind="multilinear", group=dict(max_iterations=1),
          **P_HOLDS), False),
    (_f64("group: one pass, matrix, F holds", group=dict(max_iterations=1), **F_HOLDS), False),
    (_f64("sub-cycled: linear, multilinear", kind="multilinear", dt_p=DT / 2, **P_HOLDS,
          group=dict(_SUB, boundary_interpolation="linear")), True),
    (_f64("sub-cycled: constant, multilinear, F holds", kind="multilinear", dt_p=DT / 2,
          **F_HOLDS, group=dict(_SUB, boundary_interpolation="constant")), False),
    (_f64("sub-cycled: linear, matrix, sources", dt_p=DT / 2, **SOURCES,
          group=dict(_SUB, boundary_interpolation="linear")), False),
    (_f64("sub-cycled: linear, matrix, targets, Jacobi", dt_f=DT / 2, **TARGETS,
          group=dict(_SUB, boundary_interpolation="linear", iteration_mode="jacobi")), False),
    (_f64("sub-cycled: constant, matrix, grid side fast", dt_f=DT / 2, **P_HOLDS,
          group=dict(_SUB, boundary_interpolation="constant")), False),
]


def _directions(shape) -> list:
    """Unit steps along every entry of a small array, six fixed sign patterns of a large one."""
    size = int(np.prod(shape, dtype=int)) if shape else 1
    if size <= 8:
        return [np.eye(size)[k].reshape(shape) for k in range(size)]
    rng = np.random.default_rng(size)
    return [rng.choice([-1.0, 1.0], size=shape) / np.sqrt(size) for _ in range(6)]


def _step_size(c: gg.Case, key) -> float:
    if key[0] == "state" and c.kind == "multilinear":
        return 1e-6 * min(gg.GRIDS[c.d][1])
    return 1e-6


def _assert_clear_of_the_lattice(c: gg.Case, gm) -> None:
    """No point of a multilinear case comes within 0.02 of a lattice plane
    (or leaves the hull) in the steps the gradient covers."""
    if c.kind != "multilinear":
        return
    origin, spacing, shape = gg.GRIDS[c.d]
    states = [gg.snapshot(gm)] + gg.run_steps(gm, c.steps)
    gm.reset_state()
    for state in states:
        for body in ("F", "P"):
            u = (np.asarray(state[body]["pos"], np.float64) - np.asarray(origin)) / np.asarray(
                spacing)
            frac = u - np.floor(u)
            assert np.all((frac > 0.02) & (frac < 0.98)), (c.label, body, frac)
            assert np.all((u > 0) & (u < np.asarray(shape) - 1)), (c.label, body, u)


def assert_gradient_is_the_derivative(c: gg.Case, forward_mode: bool) -> dict:
    """``jax.grad`` of *c*'s output loss against central differences; the gradient."""
    assert c.dtype == "float64" and c.adv, "premise: float64, a geometry that follows the iterate"
    with gg.x64(True):
        gm = gg.build(gg.two_body(c))
        _assert_clear_of_the_lattice(c, gm)
        loss, theta = gg.loss_of(gm, c)
        value = jax.jit(loss)
        grad = jax.jit(jax.grad(loss))(theta)
        assert any(k[0] == "state" for k in theta) and any(k[2] == "rate" for k in theta)
        for key in sorted(theta):
            g = np.asarray(grad[key], np.float64)
            base = np.asarray(theta[key], np.float64)
            h = _step_size(c, key)
            fd, ad = [], []
            for v in _directions(base.shape):
                up = {**theta, key: jnp.asarray(base + h * v)}
                down = {**theta, key: jnp.asarray(base - h * v)}
                fd.append((float(value(up)) - float(value(down))) / (2 * h))
                ad.append(float(np.vdot(g, v)))
            fd, ad = np.asarray(fd), np.asarray(ad)
            scale = float(np.max(np.abs(fd)))
            # The premise: the loss depends on this leaf, so a dropped
            # gradient is not a correct zero.
            assert scale > 0, f"{c.label}: the loss does not depend on {key}"
            assert np.max(np.abs(ad - fd)) <= RELATIVE * scale, (
                f"{c.label}: d loss / d {key}: jax.grad is "
                f"{float(np.max(np.abs(ad - fd))) / scale:.3e} (relative) from central "
                f"differences (grad {ad}, differences {fd})")
        if forward_mode:
            fwd = jax.jit(jax.jacfwd(loss))(theta)
            for key in sorted(theta):
                a, b = np.asarray(fwd[key], np.float64), np.asarray(grad[key], np.float64)
                scale = float(np.max(np.abs(b)))
                assert np.max(np.abs(a - b)) <= 1e-8 * scale, (
                    f"{c.label}: jacfwd and grad of d loss / d {key} differ by "
                    f"{float(np.max(np.abs(a - b))) / scale:.3e} (relative)")
        return {key: np.asarray(v, np.float64) for key, v in grad.items()}


def _params(cases) -> list:
    return [pytest.param(c, forward_mode, id=c.label) for c, forward_mode in cases]


@pytest.mark.parametrize("c, forward_mode", _params(PER_PUSH))
def test_the_gradient_with_respect_to_a_moving_geometry_is_the_derivative(c, forward_mode):
    """Per push; slow sibling :func:`test_the_geometry_gradient_is_the_derivative_on_every_path`."""
    assert_gradient_is_the_derivative(c, forward_mode)


# Slow: a gradient, a loss and sometimes a forward-mode Jacobian compiled
# per case, through three steps of a coupling group.
# Per push: tests/core/test_geometry_gradients.py::test_the_gradient_with_respect_to_a_moving_geometry_is_the_derivative
@pytest.mark.slow
@pytest.mark.parametrize("c, forward_mode", _params(SLOW))
def test_the_geometry_gradient_is_the_derivative_on_every_path(c, forward_mode):
    assert_gradient_is_the_derivative(c, forward_mode)


def test_the_cases_cover_the_paths_the_module_claims():
    cases = [c for c, _fwd in PER_PUSH + SLOW]
    groups = [c.knobs for c in cases if c.group is not None]
    plain = [c for c in cases if c.group is None]
    assert {c.order for c in plain} == {("F", "P"), ("P", "F")}
    assert {g.get("linear_solver", "gmres") for g in groups} == {"gmres", "dense"}
    assert {g.get("iteration_mode", "gauss-seidel") for g in groups} == {"gauss-seidel",
                                                                          "jacobi"}
    assert {g.get("solver", "ift") for g in groups} == {"ift", "fori"}
    assert {g.get("acceleration", "none") for g in groups} >= {"none", "aitken", "iqn-ils"}
    assert any(g.get("max_iterations") == 1 for g in groups)
    sub = [c for c in cases if c.group is not None and c.knobs.get("subcycling")]
    assert {c.knobs["boundary_interpolation"] for c in sub} == {"linear", "constant"}
    assert any(c.dt_p < c.dt_f for c in sub) and any(c.dt_f < c.dt_p for c in sub)
    for some in (plain, [c for c in cases if c.group is not None], sub):
        assert {c.kind for c in some} == {"multilinear", "geom_matrix"}
        assert {(c.down, c.up) for c in some} >= {("target", "source"), ("source", "target")}
    for path in ("plain", "group", "sub-cycled"):
        assert any(fwd for c, fwd in PER_PUSH + SLOW if c.label.startswith(path)), path


def test_the_float32_gradient_of_a_plain_step_is_the_float64_one():
    """The same plain path in float32, against the float64 gradient: relative ``1e-3``."""
    kw = dict(kind="multilinear", adv=0.3, **P_HOLDS)
    reference = assert_gradient_is_the_derivative(
        case("plain float64", dtype="float64", **kw), False)
    c = case("plain float32", **kw)
    gm = gg.build(gg.two_body(c))
    loss, theta = gg.loss_of(gm, c)
    grad = jax.jit(jax.grad(loss))(theta)
    assert sorted(grad) == sorted(reference)
    for key in sorted(reference):
        got = np.asarray(grad[key], np.float64)
        assert np.asarray(grad[key]).dtype == np.float32
        scale = float(np.max(np.abs(reference[key])))
        assert np.max(np.abs(got - reference[key])) <= 1e-3 * scale, (
            key, float(np.max(np.abs(got - reference[key]))) / scale)
