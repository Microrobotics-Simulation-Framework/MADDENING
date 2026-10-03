"""Four configuration rows of the coupling inventory, in every numeric domain.

Rows of ``docs/validation/coupling_claims.yaml`` verified in float32 on
single-rate graphs, stated once here over the memoryless pair of
:mod:`tests.core.coupling_domains` and run in the domains their
conditions left narrowed:

* CPL-014: ``acceleration="iqn-imvj"`` with ``jacobian_reuse=0`` is
  IQN-ILS, to the bit, step after step;
* CPL-015: ``acceleration="fixed"`` relaxes by ``x + omega (F(x) - x)``,
  and ``omega = 1`` is no relaxation: the unaccelerated group, to the bit;
* CPL-016: one pass under ``"gauss-seidel"`` reads the in-pass value of the
  member before it, under ``"jacobi"`` the previous iterate;
* CPL-142: the IFT rule gives the initial guess -- the members' pre-step
  state, which a memoryless member reads through nothing else -- a zero
  derivative, where the deprecated ``"fori"`` at a small cap does not.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import fixed_relaxation
from tests.core import coupling_domains as cd

SEQ = 4
EVERY = list(cd.EVERY) + [pytest.param(cd.ADAPTIVE, marks=pytest.mark.slow)]


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _tol(domain) -> float:
    return 0.05 if _sixteen(domain) else 1e-5


def _sequenced(domain) -> bool:
    return domain.predictor or domain.restart


def _runs(domain, gm, seq, *, x0=None):
    if _sequenced(domain):
        out = cd.run_sequence(domain, gm, seq, x0=x0)
    else:
        out = cd.run(domain, gm, seq, x0=x0)
    cd.assert_in_domain(domain, gm, out)
    return out


def _same(a, b, where):
    for k, (s, t) in enumerate(zip(a, b)):
        for n in ("a", "b"):
            assert cd.bitwise(s.x(n), t.x(n)), f"{where} step {k}: {n}.x {s.x(n)} != {t.x(n)}"
        for key in ("iterations", "converged", "residual"):
            assert s.report[key] == t.report[key] or (
                math.isnan(s.report[key]) and math.isnan(t.report[key])), (
                f"{where} step {k}: {key} {s.report[key]} != {t.report[key]}")


# ---------------------------------------------------------------------------
# CPL-014
# ---------------------------------------------------------------------------
# Per push: tests/core/test_coupling_configuration_in_every_domain.py::test_imvj_without_reuse_is_iqn_ils
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_imvj_without_reuse_is_iqn_ils(label):
    """CPL-014: "0 means no reuse (same as IQN-ILS)" -- on a two-entry pair
    with independent modes, every step of a run whose forcing moves, both
    solvers: states and reports to the bit."""
    d = cd.DOMAINS[label]
    g = ((0.8, -0.6), (0.7, 0.9))
    with cd.entered(d):
        for solver in ("ift", "fori"):
            extra = dict(diagnostics=True) if solver == "fori" else {}
            runs = []
            for acc in (dict(acceleration="iqn-imvj", jacobian_reuse=0),
                        dict(acceleration="iqn-ils")):
                gm = cd.pair(d, n=2, g=g, c=((1.0, 0.5), (0.0, -0.25)), solver=solver,
                             tolerance=_tol(d), max_iterations=40, **acc, **extra)
                seq = cd.moving(gm, SEQ, g=g, c0=((1.0, 0.5), (0.0, -0.25)),
                                dc=((1.0, 1.0), (0.5, 0.5)))
                runs.append(_runs(d, gm, seq))
            _same(*runs, f"{label}/{solver}")
            assert max(s.report["iterations"] for s in runs[0]) > 2, (
                f"{label}/{solver}: fixture premise: the secant system is entered")


# ---------------------------------------------------------------------------
# CPL-015
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", cd.DTYPES + ("vmap",))
def test_fixed_relaxation_is_the_documented_combination_in_every_dtype(label):
    """CPL-015 in the formula itself: ``x + omega (F(x) - x)`` in each member
    dtype, exact on dyadic operands (``omega = 0.5``); ``omega = 1`` returns
    ``F(x)``.  Batched under ``jax.vmap`` in the batch domain."""
    d = cd.DOMAINS[label]
    with cd.entered(d):
        for dtype in dict.fromkeys(d.dtypes):
            old = jnp.asarray([[1.0, 2.0], [-0.5, 0.25]], dtype)
            raw = jnp.asarray([[2.0, 4.0], [0.5, -0.75]], dtype)
            fn = (jax.vmap(lambda o, r, w: fixed_relaxation(o, r, w), in_axes=(0, 0, None))
                  if d.vmap else (lambda o, r, w: jnp.stack([fixed_relaxation(o[i], r[i], w)
                                                             for i in range(2)])))
            half = np.asarray(fn(old, raw, 0.5), np.float64)
            np.testing.assert_array_equal(half, [[1.5, 3.0], [0.0, -0.25]])
            one = fn(old, raw, 1.0)
            assert one.dtype == jnp.dtype(dtype)
            np.testing.assert_array_equal(np.asarray(one, np.float64),
                                          np.asarray(raw, np.float64))


@pytest.mark.parametrize("label", cd.DTYPES + ("vmap",))
def test_relaxation_one_is_no_relaxation(label):
    """CPL-015: "1.0 is no relaxation" -- ``acceleration="fixed",
    relaxation=1.0`` steps a moving run as the unaccelerated group does, to
    the bit."""
    d = cd.DOMAINS[label]
    with cd.entered(d):
        runs = []
        for acc in (dict(acceleration="fixed", relaxation=1.0), dict(acceleration="none")):
            gm = cd.pair(d, g=(0.8, 0.9), tolerance=_tol(d), max_iterations=200, **acc)
            runs.append(_runs(d, gm, cd.moving(gm, SEQ, g=(0.8, 0.9))))
        _same(*runs, label)


# ---------------------------------------------------------------------------
# CPL-016
# ---------------------------------------------------------------------------
_ONE_PASS = ["mixed_dtype", "bfloat16", "float16", "vmap", "multi_rate", "checkpoint_restart"]


@pytest.mark.parametrize("label", _ONE_PASS)
def test_one_pass_reads_the_iterate_its_mode_documents(label):
    """CPL-016: ``max_iterations=1`` from a known state, ``a`` scheduled
    before ``b``.  Gauss-Seidel: ``a' = g_a b + c_a``, then ``b' = g_b a' + c_b``
    (the in-pass value); Jacobi: ``b' = g_b a + c_b`` (the previous iterate).
    Against the float64 sweep from the state each solve started from, to the
    members' rounding."""
    d = cd.DOMAINS[label]
    g, c = (0.5, -0.75), (1.0, 2.0)
    x0 = (0.3, -0.6)
    with cd.entered(d):
        for mode in ("gauss-seidel", "jacobi"):
            gm = cd.pair(d, g=g, c=c, iteration_mode=mode, max_iterations=1)
            assert [n for n in gm.schedule if n in "ab"] == ["a", "b"]
            seq = cd.moving(gm, SEQ, g=g, c0=c)
            for s in _runs(d, gm, seq, x0=x0):
                ga, gb = s.gains()
                ca, cb = s.forcing()
                pa = float(np.asarray(s.pre["a"]["x"], np.float64)[0])
                pb = float(np.asarray(s.pre["b"]["x"], np.float64)[0])
                a_new = ga * pb + float(ca[0])
                b_new = gb * (a_new if mode == "gauss-seidel" else pa) + float(cb[0])
                eps = float(cd.finfo(d.coarsest).eps)
                for name, want in (("a", a_new), ("b", b_new)):
                    got = float(np.asarray(s.x(name), np.float64)[0])
                    assert abs(got - want) <= 4 * eps * max(1.0, abs(want)), (
                        label, mode, name, got, want)


# ---------------------------------------------------------------------------
# CPL-142
# ---------------------------------------------------------------------------
_GUESS = ["f64", "mixed_dtype", "jit", "vmap", "multi_rate", "sub_cycled",
          "predictors_warm_starts", "checkpoint_restart"]


def _firing_state(domain, gm, p):
    """A state the next raw step solves the group from: on a multi-rate graph,
    the base step the group fires on (the members move)."""
    gm.reset_state()
    for _ in range(3):
        before = cd._members(gm)
        trial = gm._raw_step_fn(gm._state, gm._default_external_inputs(), p)
        if not domain.multirate or not cd.bitwise(np.asarray(trial["a"]["x"]),
                                                  before["a"]["x"]):
            return gm._state
        gm.step(params=p)
    raise AssertionError("the group never fired")


def _guess_paths(domain, state):
    """Where the solve's initial guess comes from: the members' pre-step state,
    or with a predictor the converged states it extrapolates (``_meta``)."""
    if domain.predictor:
        paths = [("_meta", k) for k in state["_meta"] if "_pred_" in k and not
                 k.endswith("_count")]
        assert int(state["_meta"]["coupling_a+b_pred_count"]) >= 3, "premise: a full history"
        return paths
    return [("a", "x"), ("b", "x")]


# Slow: a reverse- and a forward-mode compile of the IFT and the unrolled
# step per domain (8-17 s on three cores).  The zero derivative of the
# initial guess stays on every push in float32 at the solve itself.
# Per push: tests/core/test_coupling_claims_helpers.py::test_the_ift_solve_gives_the_initial_guess_a_zero_derivative
@pytest.mark.slow
@pytest.mark.parametrize("label", _GUESS)
def test_the_initial_guess_gets_a_zero_derivative(label):
    """CPL-142: ``x0`` receives a zero derivative.  A memoryless member reads its
    pre-step state only as the solve's initial guess -- with a predictor, the
    converged states it extrapolates in ``_meta`` -- so the derivative of one
    step's ``x_a`` with respect to it is exactly zero under the IFT rule,
    forward and reverse; the deprecated ``"fori"``, which differentiates the
    iterate it returned, does not give zero at a cap of three passes -- the
    check can fail.  ``jit`` takes the derivative inside ``jax.jit``; the
    batch domain under ``jax.vmap`` over three guesses; the restart domain
    from a run saved and loaded."""
    domain = "f32" if label == "jit" else label
    d = cd.DOMAINS[domain]
    with cd.entered(d):
        for solver, cap in (("ift", 60), ("fori", 3)):
            extra = dict(diagnostics=True) if solver == "fori" else {}
            gm = cd.pair(d, n=2, g=((0.8, -0.6), (0.7, 0.9)), c=((1.0, 0.5), (0.0, -0.25)),
                         solver=solver, tolerance=_tol(d), max_iterations=cap, **extra)
            seq = cd.moving(gm, 5, g=((0.8, -0.6), (0.7, 0.9)),
                            c0=((1.0, 0.5), (0.0, -0.25)), dc=((1.0, 1.0), (0.5, 0.5)))
            p = seq[-1]
            if d.predictor or d.restart:
                cd.run_sequence(d, gm, seq[:-1])       # history, or a save and a load
                state = gm._state
            else:
                state = _firing_state(d, gm, p)
            step, ext = gm._raw_step_fn, gm._default_external_inputs()
            paths = _guess_paths(d, state)

            def xa(leaves, state=state, paths=paths):
                s = {k: dict(v) if isinstance(v, dict) else v for k, v in state.items()}
                for (owner, key), leaf in zip(paths, leaves):
                    s[owner][key] = leaf
                return jnp.sum(step(s, ext, p)["a"]["x"].astype(jnp.float32))

            leaves = tuple(state[o][k] for o, k in paths)
            rev_fn = jax.grad(xa)
            if label == "jit":
                rev_fn = jax.jit(rev_fn)
            if d.vmap:
                batch = tuple(jnp.stack([v, v * 1.5, v - 0.25]) for v in leaves)
                rev = jax.vmap(rev_fn)(batch)
            else:
                rev = rev_fn(leaves)
            tangent = tuple(jnp.ones_like(v) for v in leaves)
            fwd = jax.jvp(xa, (leaves,), (tangent,))[1]
            values = [np.asarray(v, np.float64) for v in rev] + [np.asarray(fwd, np.float64)]
            if solver == "ift":
                assert all(not np.any(v) for v in values), (label, values)
            else:
                assert any(np.any(v) for v in values), (
                    f"{label}: fori at three passes gave zero too: the check cannot fail")
