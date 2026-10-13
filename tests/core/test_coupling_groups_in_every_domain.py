"""Two coupling groups in one graph keep to themselves, in every numeric domain.

CPL-079 of ``docs/validation/coupling_claims.yaml``: "groups in one graph
keep independent schedules, iteration counts and convergence flags", and
the graph steps deterministically.  Verified in float32 on the
benchmark fixtures' two-group graph; here on two memoryless pairs
(:mod:`tests.core.coupling_domains`' members), ``a <-> b`` under
Gauss-Seidel and ``c <-> d`` under Jacobi, with no edge between the groups,
in every domain: float64 and a float32 member beside a float64 one under
x64, bfloat16 and float16, a ``jax.vmap`` of the step, a multi-rate graph,
sub-cycled groups, predictors, a checkpoint restart and (slow)
``run_adaptive``.

Three statements, each the claim's:

* **independent**: each group's states and report in the two-group graph
  are, bit for bit, those of the same group compiled alone;
* **its own schedule**: under-converged (a cap of three passes and a
  criterion no pass meets, so the iterate shows the schedule), flipping
  the second group's mode moves its iterate and leaves the first group's
  bit-identical;
* **deterministic**: two builds of the graph step to the same bits and
  the same iteration counts.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from tests.core import coupling_domains as cd

PAIRS = (("a", "b"), ("c", "d"))
KEYS = ("a+b", "c+d")
N = 2
STEPS = 4
#: Gains and forcings per node: two contractions with different rates, so
#: the two groups need different pass counts.
_G = {"a": (0.5, -0.3), "b": (0.6, 0.4), "c": (0.9, 0.2), "d": (-0.7, 0.8)}
_C = {"a": (1.0, 0.5), "b": (0.0, -0.25), "c": (2.0, 1.0), "d": (0.5, 0.0)}


def _build(domain, modes, *, pairs=PAIRS, cap=40, tol=None):
    """The pairs in *pairs* as coupling groups of the given *modes*, in *domain*."""
    gm = GraphManager()
    da, db = domain.dtypes
    tb = cd.DT / 2 if domain.subcycled else cd.DT
    if tol is None:
        tol = 0.05 if jnp.dtype(domain.coarsest).itemsize == 2 else 1e-5
    for (p, q), mode in zip(pairs, modes):
        gm.add_node(cd.Lin(p, cd.DT, da, N, g=_G[p], c=_C[p]))
        gm.add_node(cd.Lin(q, tb, db, N, g=_G[q], c=_C[q]))
        if da == db:
            gm.add_edge(q, p, "x", "u")
            gm.add_edge(p, q, "x", "u")
        else:
            gm.add_edge(q, p, "x", "u", transform=lambda v: v.astype(da))
            gm.add_edge(p, q, "x", "u", transform=lambda v: v.astype(db))
    if domain.multirate:
        gm.add_node(cd.Ticker("tick", cd.DT / 2))
    for (p, q), mode in zip(pairs, modes):
        gm.add_coupling_group([p, q], **cd.group_kwargs(
            domain, iteration_mode=mode, max_iterations=cap, tolerance=tol))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate notice
        gm.compile()
    return gm


def _params(gm, k):
    """Step *k*'s parameters: every forcing moved irregularly, so a predictor's
    guess is never exact."""
    p = jax.tree.map(lambda v: v, gm.params)
    shift = sum(cd._MOVES[:k + 1])
    for name, leaves in p["nodes"].items():
        if name in _C:
            leaves["c"] = jnp.asarray(np.asarray(_C[name]) * (1.0 + 0.5 * shift),
                                      leaves["c"].dtype)
    return p


def _record(gm, state=None):
    """``({node: x}, {key: report})`` for *state* (the graph's own by default)."""
    full = gm._state if state is None else state
    saved = gm._state
    gm._state = full
    try:
        diag = gm.coupling_diagnostics()
    finally:
        gm._state = saved
    xs = {n: np.asarray(full[n]["x"]) for n in "abcd" if n in full}
    return xs, {k: dict(diag[k]) for k in KEYS if k in diag}


_VMAPPED: dict = {}


def _run(domain, gm):
    """``[({node: x}, {key: report})]`` for each checked step of one run."""
    gm.reset_state()
    if domain.vmap:
        if id(gm) not in _VMAPPED:
            _VMAPPED[id(gm)] = (gm, jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, 0))))
        step = _VMAPPED[id(gm)][1]
        params = [_params(gm, k) for k in range(3)]       # three members
        state = jax.tree.map(lambda *xs: jnp.stack(xs), *([gm._state] * 3))
        state = step(state, gm._default_external_inputs(),
                     jax.tree.map(lambda *xs: jnp.stack(xs), *params))
        return [_record(gm, jax.tree.map(lambda v, i=i: v[i], state)) for i in range(3)]
    steps = 2 * STEPS if domain.multirate else STEPS

    def one(k):
        if domain.adaptive:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm.run_adaptive(cd.DT, params=_params(gm, k), **cd.ADAPTIVE_KW)
        else:
            gm.step(params=_params(gm, k))
        return _record(gm)

    if not domain.restart:
        return [one(k) for k in range(steps)]
    for k in range(2):
        one(k)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        gm.save_state(path)
        straight = [one(k) for k in range(2, steps)]
        gm.reset_state()
        gm.load_state(path)
        after = [one(k) for k in range(2, steps)]
    for (xa, _ra), (xb, _rb) in zip(straight, after):
        assert all(cd.bitwise(xa[n], xb[n]) for n in xa), "the restart left the straight run"
    return after


def _assert_premise(domain, gm, runs):
    for xs, _reports in runs:
        for n, dtype in zip("abcd", domain.dtypes * 2):
            assert xs[n].dtype == jnp.dtype(dtype), (domain.label, n, xs[n].dtype)
    groups = gm._committed_coupling_groups.values()
    assert bool(gm._is_multirate) == domain.multirate
    assert all(g.subcycling == domain.subcycled for g in groups)
    assert all((g.predictor != "none") == domain.predictor for g in groups)


def _same(a, b, nodes, keys, where):
    for k, ((xa, ra), (xb, rb)) in enumerate(zip(a, b)):
        for n in nodes:
            assert cd.bitwise(xa[n], xb[n]), f"{where} step {k}: {n}.x {xa[n]} != {xb[n]}"
        for key in keys:
            assert cd._same_report(ra[key], rb[key]), f"{where} step {k}: {key} {ra[key]} {rb[key]}"


_EVERY = list(cd.EVERY) + [pytest.param(cd.ADAPTIVE, marks=pytest.mark.slow)]


# Per push: tests/core/test_coupling_groups_in_every_domain.py::test_each_group_in_a_two_group_graph_is_the_group_alone
# (every domain but run_adaptive, which compiles its step on every call)
@pytest.mark.parametrize("label", _EVERY)
def test_each_group_in_a_two_group_graph_is_the_group_alone(label):
    """CPL-079: each group's states, iteration counts, verdicts and every
    other report key in the two-group graph are the same group's compiled
    alone, bit for bit; and a second build steps to the same bits."""
    d = cd.DOMAINS[label]
    with cd.entered(d):
        both = _build(d, ("gauss-seidel", "jacobi"))
        runs = _run(d, both)
        _assert_premise(d, both, runs)
        for pair, key, mode in zip(PAIRS, KEYS, ("gauss-seidel", "jacobi")):
            alone = _run(d, _build(d, (mode,), pairs=(pair,)))
            _same(runs, alone, pair, (key,), f"{label} {key} alone")
        _same(runs, _run(d, _build(d, ("gauss-seidel", "jacobi"))), "abcd", KEYS,
              f"{label} second build")
        counts = {key: [reps[key]["iterations"] for _x, reps in runs] for key in KEYS}
        assert counts["a+b"] != counts["c+d"], (
            f"{label}: fixture premise: the groups take different pass counts {counts}")


@pytest.mark.parametrize("label", cd.EVERY)
def test_flipping_one_groups_schedule_leaves_the_other_alone(label):
    """CPL-079: three passes and a criterion no pass meets, so each iterate
    shows its schedule: flipping the second group to Gauss-Seidel moves its
    iterate and leaves the first group bit-identical.

    Not under ``run_adaptive``: there one controller chooses the step for
    the whole graph from the step-doubling error of every field, and an
    under-converged group's iterate depends on how many solves it gets, so
    the second group's schedule moves the steps the first is given.  With
    both groups converged (the test above) they keep to themselves there
    too."""
    d = cd.DOMAINS[label]
    with cd.entered(d):
        base = _run(d, _build(d, ("gauss-seidel", "jacobi"), cap=3, tol=1e-30))
        flipped = _run(d, _build(d, ("gauss-seidel", "gauss-seidel"), cap=3, tol=1e-30))
        _same(base, flipped, "ab", ("a+b",), f"{label} first group")
        moved = any(not cd.bitwise(xa[n], xb[n]) for (xa, _ra), (xb, _rb) in zip(base, flipped)
                    for n in "cd")
        assert moved, f"{label}: the second group's iterate ignored its schedule"
