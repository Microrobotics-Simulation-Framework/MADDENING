"""``sharded_cg``'s default call answers wherever ``backend="loop"`` can be
traced, and where lineax answers it is ``backend="lineax"`` bit for bit.

The default route (``backend="auto"`` without a preconditioner) is lineax's
CG, and the hand-rolled loop where lineax's value is not all finite
(MADD-ANO-264: in float32 lineax's CG sets its step to NaN when
``p.A p / r.r <= 200 * eps * n``).  Under a trace the choice is made
without a ``lax.cond``: the loop is run with an iteration cap of
``max_iters`` where lineax broke down and of zero where it answered, and
each of the four fields is selected with ``jnp.where``.  This file pins
what that construction has to give:

* **inside a ``shard_map`` body** (one solve per device, on that device's
  block) the default call answers, under ``jit`` and eagerly, with and
  without ``differentiable=True``, an ``x0`` and an ``atol``.  The first
  form of the fallback (a ``lax.cond`` holding a
  ``lax.custom_linear_solve``) raised ``Primitive linear_solve requires
  varying manual axes to match`` on every such call;
* the choice is **per solve**: in one ``shard_map`` call the devices whose
  system lineax answers get lineax's four fields bit for bit, and the
  devices whose system it breaks down on get the loop's four fields bit
  for bit;
* **what it costs**: under a trace a default call that does not fall back
  executes exactly two operator products more than ``backend="lineax"``
  (the loop's ``b - A x0`` and the product of its report; no iteration);
  called eagerly it executes none.

The system is the periodic 1-D Laplacian shifted on its diagonal by ``s``
(``(2 + s) x[i] - x[i-1] - x[i+1]``, indices modulo the size), 256
unknowns per solve, float32.  With ``s = 0.5`` and white noise lineax
answers.  With ``s = 1e-3`` and the operator's lowest mode (a constant) as
the right-hand side the quotient of lineax's first step is the lowest
eigenvalue, 1e-3, under the threshold ``200 * eps * 256 = 6.1e-3``: lineax
returns NaN, and the loop solves it in a step or two.  Both are asserted before they are relied on.  No iteration
count is pinned; every comparison is of one call against another in the
same process.  Under vmap and on a mesh the route is pinned by
``test_what_sharded_cg_reports_on_each_route.py``.
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
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.iterative_solver import sharded_cg

UNKNOWNS = 256
DEVICES = 4
RTOL = 1e-4
#: Diagonal shifts: lineax answers at the first, breaks down at the second
#: (on the right-hand sides of ``_rhs``).
ANSWERS, BREAKS_DOWN = 0.5, 1e-3
EPS32 = float(np.finfo(np.float32).eps)


def _answered(x):
    return (2 + ANSWERS) * x - jnp.roll(x, 1) - jnp.roll(x, -1)


def _broken_down(x):
    return (2 + BREAKS_DOWN) * x - jnp.roll(x, 1) - jnp.roll(x, -1)


def _matvec(shift: float):
    """The periodic 1-D Laplacian shifted on its diagonal, as a function
    that CLOSES OVER NOTHING (module-level, its numbers read from module
    globals).  Inside a ``shard_map`` body the lineax route can trace no
    other kind: lineax converts a closure once, on an abstract vector that
    carries no mesh axis, and applying the converted function to a
    device's block then fails (``Primitive mul requires varying manual
    axes to match``), on the default route as on ``backend="lineax"``, and
    before 0.4.0 as now.  ``backend="loop"`` takes a closure there
    (``test_the_loop_backend_takes_a_closure_inside_a_shard_map_body``)."""
    return {ANSWERS: _answered, BREAKS_DOWN: _broken_down}[shift]


def _lowest_mode() -> np.ndarray:
    """The eigenvector of the eigenvalue ``shift``: constant."""
    return np.ones(UNKNOWNS, np.float32)


def _highest_mode() -> np.ndarray:
    """The eigenvector of the eigenvalue ``4 + shift``: alternating."""
    return np.where(np.arange(UNKNOWNS) % 2 == 0, 1.0, -1.0).astype(np.float32)


def _rhs(shift: float, seed: int = 0) -> np.ndarray:
    """White noise where lineax is meant to answer, the operator's lowest
    mode (times a factor that differs per seed) where it is meant to break
    down."""
    if shift == ANSWERS:
        return np.random.default_rng(seed).standard_normal(UNKNOWNS).astype(np.float32)
    return (1.0 + seed) * _lowest_mode()


def _fields(result) -> tuple:
    return tuple(np.asarray(f) for f in (
        result.value, result.converged, result.iters, result.residual_norm))


def _bits_differ(got: tuple, want: tuple) -> list:
    names = ("value", "converged", "iters", "residual_norm")
    return [n for n, g, w in zip(names, got, want)
            if g.dtype != w.dtype or g.tobytes() != w.tobytes()]


def test_lineax_answers_the_first_system_and_breaks_down_on_the_second():
    """The two systems are what every test below takes them for."""
    threshold = 200 * EPS32 * UNKNOWNS
    assert BREAKS_DOWN < threshold / 4 < 4 + BREAKS_DOWN   # the lowest and the highest eigenvalue

    answered = sharded_cg(_matvec(ANSWERS), jnp.asarray(_rhs(ANSWERS)),
                          rtol=RTOL, backend="lineax")
    assert np.all(np.isfinite(np.asarray(answered.value)))
    assert bool(answered.converged)

    broke = sharded_cg(_matvec(BREAKS_DOWN), jnp.asarray(_rhs(BREAKS_DOWN)),
                       rtol=RTOL, backend="lineax")
    assert np.all(np.isnan(np.asarray(broke.value)))
    assert not bool(broke.converged)

    by_loop = sharded_cg(_matvec(BREAKS_DOWN), jnp.asarray(_rhs(BREAKS_DOWN)),
                         rtol=RTOL, backend="loop")
    assert bool(by_loop.converged)


# ---------------------------------------------------------------------------
# Inside a shard_map body: one solve per device, each on its own route
# ---------------------------------------------------------------------------


#: What each of the four devices solves, per case: the operator's shift (one
#: operator for the whole ``shard_map`` call, see ``_matvec``) and each
#: device's right-hand side.  At the shift 1e-3 the lowest mode breaks lineax down on its first
#: step (quotient 1e-3) and the highest mode does not (quotient 4).
CASES = {
    "lineax answers on every device": (
        ANSWERS, [_rhs(ANSWERS, seed=d) for d in range(DEVICES)]),
    "lineax breaks down on devices 1 and 3": (
        BREAKS_DOWN, [(1.0 + d) * (_highest_mode() if d % 2 == 0 else _lowest_mode())
                      for d in range(DEVICES)]),
}
KEYWORDS = {
    "plain": {},
    "x0": {"start": True},
    "atol": {"atol": 1e-7},
    "differentiable": {"differentiable": True},
    "differentiable with x0 and atol": {"differentiable": True, "start": True, "atol": 1e-7},
}


def _per_device(backend: str, jit: bool, case: str, start: bool = False, **kw) -> tuple:
    """The four fields of one ``sharded_cg`` call per device, made inside a
    ``shard_map`` body; row ``d`` of each field is device ``d``'s."""
    mesh = create_device_mesh(shape=(DEVICES,))
    shift, blocks = CASES[case]
    b = jnp.asarray(np.concatenate(blocks).astype(np.float32))

    def body(block):
        guess = {"x0": 0.25 * block} if start else {}
        r = sharded_cg(_matvec(shift), block, rtol=RTOL, backend=backend, **guess, **kw)
        return r.value, r.converged[None], r.iters[None], r.residual_norm[None]

    f = shard_map(body, mesh=mesh, in_specs=(P("devices"),), out_specs=(P("devices"),) * 4)
    value, converged, iters, residual_norm = (jax.jit(f) if jit else f)(b)
    return (np.asarray(value).reshape(DEVICES, UNKNOWNS), np.asarray(converged),
            np.asarray(iters), np.asarray(residual_norm))


def _row(fields: tuple, device: int) -> tuple:
    return tuple(np.asarray(f[device]) for f in fields)


def _each_device_gets_its_route(case: str, keywords: str, jit: bool) -> None:
    kw = KEYWORDS[keywords]
    default = _per_device("auto", jit, case, **kw)
    by_lineax = _per_device("lineax", jit, case, **kw)
    by_loop = _per_device("loop", jit, case, **kw)
    broke_down = [bool(np.all(np.isnan(by_lineax[0][d]))) for d in range(DEVICES)]
    assert broke_down == [case.endswith("1 and 3") and d % 2 == 1 for d in range(DEVICES)]
    for device in range(DEVICES):
        got = _row(default, device)
        want = _row(by_loop if broke_down[device] else by_lineax, device)
        assert _bits_differ(got, want) == [], device
        assert np.all(np.isfinite(got[0])), device
        assert bool(got[1]), (device, float(got[3]))


@pytest.mark.parametrize("keywords", list(KEYWORDS))
@pytest.mark.parametrize("case", list(CASES))
def test_inside_a_shard_map_body_each_device_gets_the_route_its_system_needs(case, keywords):
    """One jitted ``shard_map`` call, four solves.  The default call gives
    each device whose system lineax answers what ``backend="lineax"`` gives
    it, and each device whose system lineax breaks down on what
    ``backend="loop"`` gives it: all four fields, bit for bit.  Which
    devices are which is read from lineax by name (NaN where it broke
    down), and asserted against the case."""
    _each_device_gets_its_route(case, keywords, jit=True)


# Per push: tests/cloud/multigpu/test_sharded_cg_default_route_in_every_context.py::test_inside_a_shard_map_body_each_device_gets_the_route_its_system_needs
# (the same comparison jitted) and
# tests/cloud/multigpu/test_sharded_cg_default_route_in_every_context.py::test_the_default_call_answers_inside_a_shard_map_body_that_is_not_jitted
# (one eager call; three of them, which the comparison needs, take 8 s).
@pytest.mark.slow
@pytest.mark.parametrize("keywords", list(KEYWORDS))
@pytest.mark.parametrize("case", list(CASES))
def test_inside_a_shard_map_body_that_is_not_jitted_each_device_gets_its_route(case, keywords):
    """The same, with the ``shard_map`` call made eagerly."""
    _each_device_gets_its_route(case, keywords, jit=False)


def test_the_default_call_answers_inside_a_shard_map_body_that_is_not_jitted():
    """The eager call, once per push: devices 1 and 3 fall back, and every
    device's value solves its system (the residual in float64 on the host,
    with twice ``rtol`` for the float32 solve)."""
    case = "lineax breaks down on devices 1 and 3"
    value, converged, _, _ = _per_device("auto", False, case)
    assert converged.tolist() == [True] * DEVICES
    for device, block in enumerate(CASES[case][1]):
        x = np.asarray(value[device], np.float64)
        rhs = np.asarray(block, np.float64)
        ax = (2 + np.float32(BREAKS_DOWN)) * x - np.roll(x, 1) - np.roll(x, -1)
        assert np.linalg.norm(rhs - ax) <= 2 * RTOL * np.linalg.norm(rhs), device


def test_the_loop_backend_takes_a_closure_inside_a_shard_map_body():
    """What to call there when the operator closes over a value (every
    operator with a coefficient does): ``backend="loop"``."""
    mesh = create_device_mesh(shape=(DEVICES,))
    b = jnp.asarray(np.concatenate([_rhs(ANSWERS, seed=d) for d in range(DEVICES)]))

    def body(block, shift):
        def matvec(x):
            return (2 + shift[0]) * x - jnp.roll(x, 1) - jnp.roll(x, -1)

        r = sharded_cg(matvec, block, rtol=RTOL, backend="loop")
        return r.value, r.converged[None]

    f = shard_map(body, mesh=mesh, in_specs=(P("devices"), P("devices")),
                  out_specs=(P("devices"),) * 2)
    shifts = jnp.asarray([0.5, 1.0, 2.0, 4.0], jnp.float32)
    value, converged = jax.jit(f)(b, shifts)
    assert np.asarray(converged).tolist() == [True] * DEVICES
    x = np.asarray(value, np.float64).reshape(DEVICES, UNKNOWNS)
    rhs = np.asarray(b, np.float64).reshape(DEVICES, UNKNOWNS)
    for device in range(DEVICES):
        ax = ((2 + float(shifts[device])) * x[device] - np.roll(x[device], 1)
              - np.roll(x[device], -1))
        assert np.linalg.norm(rhs[device] - ax) <= 2 * RTOL * np.linalg.norm(rhs[device])


# ---------------------------------------------------------------------------
# What the default route costs, in operator products
# ---------------------------------------------------------------------------


def _products(shift: float, backend: str, jit: bool) -> int:
    """How many times the operator is APPLIED (not traced) in one solve."""
    applied = []
    plain = _matvec(shift)

    def counted(x):
        # The callback takes an entry of ``x`` so that it depends on the
        # loop's state and cannot be moved out of a loop body.
        jax.debug.callback(lambda _: applied.append(1), x[0])
        return plain(x)

    def solve(b):
        return sharded_cg(counted, b, rtol=RTOL, backend=backend).value

    value = (jax.jit(solve) if jit else solve)(jnp.asarray(_rhs(shift)))
    jax.block_until_ready(value)
    jax.effects_barrier()
    return len(applied)


def test_a_default_call_that_does_not_fall_back_costs_two_products_under_a_trace():
    """Under ``jit`` the loop is in the program with a cap of zero
    iterations: its set-up (``b - A x0``) and its report (``b - A x``) are
    executed, its iteration is not."""
    assert _products(ANSWERS, "auto", jit=True) == _products(ANSWERS, "lineax", jit=True) + 2


def test_a_default_call_that_does_not_fall_back_costs_nothing_eagerly():
    """Called eagerly the predicate is a number: where lineax answered the
    loop is not run at all."""
    assert _products(ANSWERS, "auto", jit=False) == _products(ANSWERS, "lineax", jit=False)


@pytest.mark.parametrize("jit", [True, False], ids=["jitted", "eager"])
def test_a_default_call_that_falls_back_costs_the_lineax_attempt_and_the_loop(jit):
    """Where lineax breaks down the products are those of its attempt and
    those of ``backend="loop"``, nothing more: the loop is run once."""
    assert _products(BREAKS_DOWN, "auto", jit) == (
        _products(BREAKS_DOWN, "lineax", jit) + _products(BREAKS_DOWN, "loop", jit))
