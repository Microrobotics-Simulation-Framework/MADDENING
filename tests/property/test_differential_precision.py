"""Differential oracle: float32 and float64 give the same verdicts and answers.

``jax_enable_x64`` is a supported mode (``fim`` recommends it for an
ill-conditioned problem), and every documented claim of the fitters, the
FIM and the coupling diagnostics is stated for both precisions.  This
oracle runs one case twice -- in float32 in the pytest process, and in
float64 in a worker process started with ``JAX_ENABLE_X64=1``
(``precision_cases.py``: x64 is process-global) -- and requires, on
problems drawn away from every decision threshold:

* **the verdicts are equal** -- ``converged``, ``excited_rank``,
  ``hold_declined``, the FIM's ``rank``, the coupling report's
  ``converged`` / ``ratio_usable`` / ``spectral_usable`` /
  ``precision_limited`` and ``gradient_bound_usable``;
* **the answers agree within a tolerance derived from the problem** (each
  stated where it is used: a forward-error count times the precision's
  ``eps``, propagated by the problem's own sensitivity ``||J^+||``, or a
  bound the library documents and reports), and where the truth is known
  each precision recovers it within its own tolerance -- so a float64 run
  that silently computed in float32 would fail.

Covered: ``fit_lm``, ``fit`` (Adam), ``fit_multiple_shooting``, ``fim``,
``windowed_loss``, ``coupling_diagnostics()``, an IFT gradient, and
float32 leaves in an x64 graph.  "Away from thresholds": truths at 20-80%
of each range, starts inside it, generous budgets, a coupling tolerance two
orders above the float32 floor and a contraction rate of 0.06-0.48.

**Known failing:** B1-H1 -- under x64 the identifiability guard misses the
spring's exact ``(k, c, m)`` scale degeneracy (rank 3 where float32 holds
the scale at rank 2), with float64 and with float32 leaves; B1-L1 --
``fit_lm`` reports ``converged=False`` at its float64 floor where float32
converges.

What it cannot see: a defect both precisions share (one code path), and
problems near a threshold, where the two may legitimately differ.
"""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import precision_cases as cases

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Seconds a float64 case may take in the worker before the oracle gives up.
WORKER_TIMEOUT = 600


class Float64Worker:
    """``precision_cases.serve()`` in a process of its own, under x64."""

    def __init__(self) -> None:
        env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        self._log = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "from tests.property.precision_cases import serve; serve()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True,
            env=env, cwd=str(REPO_ROOT))

    def send(self, case: str, args: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"case": case, "args": args}) + "\n")
        self.proc.stdin.flush()

    def receive(self) -> dict:
        assert self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], WORKER_TIMEOUT)
        line = self.proc.stdout.readline() if ready else ""
        if not line:
            self._log.seek(0)
            raise RuntimeError(f"the float64 worker gave no answer "
                               f"(exit {self.proc.poll()}): {self._log.read()[-2000:]}")
        reply = json.loads(line)
        if not reply["ok"]:
            raise RuntimeError(f"float64 case failed: {reply['error']}\n{reply['traceback']}")
        return reply["result"]

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()                    # our own child, by its handle
            self.proc.wait()
        self._log.close()


@pytest.fixture(scope="module")
def worker():
    w = Float64Worker()
    try:
        yield w
    finally:
        w.close()


def both(worker: Float64Worker, case: str, *, x64_args: dict | None = None,
         **args) -> tuple[dict, dict]:
    """The case in float32 here and in float64 in the worker, at once."""
    assert not jax.config.jax_enable_x64, "the float32 side must run without x64"
    worker.send(case, {**args, **(x64_args or {})})
    r32 = cases.run_case(case, args)
    r64 = worker.receive()
    return r32, r64


def assert_same_verdicts(r32: dict, r64: dict, what: str) -> None:
    assert r32["verdicts"] == r64["verdicts"], (
        f"{what}: float32 says {r32['verdicts']}, float64 says {r64['verdicts']}")


def test_the_float64_side_runs_under_x64(worker):
    """The worker is what it claims to be (and absorbs its own start-up)."""
    worker.send("environment", {})
    env = worker.receive()
    assert env["x64"] is True and env["dtype"] == "float64", env
    assert cases.environment()["dtype"] == "float32"


# ---------------------------------------------------------------------------
# The fitters on a closed-form problem
# ---------------------------------------------------------------------------

def floor_tolerance(r: dict) -> float:
    """How far a fit at its floor may sit from the truth: the residual's
    forward error, ``BLOCK_OPS * eps * ||g||`` (each entry is at most that
    many roundings from its value), moved into the parameters by the
    problem's sensitivity ``||J^+||``."""
    return r["pinv_norm"] * cases.BLOCK_OPS * r["eps"] * r["g_norm"]


def fit_lm_tolerance(r: dict, i: int) -> float:
    """``floor_tolerance`` plus ``fit_lm``'s own stopping rule: a step of
    at most ``step_tol`` (16 ulps of the parameter) is a converged one."""
    return floor_tolerance(r) + 16 * r["eps"] * abs(r["answers"]["truth"][i])


def check_fit_lm(kinds, truth_at, start_at) -> None:
    r32, r64 = both(_worker(), "blocks_fit_lm", kinds=kinds, truth_at=truth_at,
                    start_at=start_at)
    assert_same_verdicts(r32, r64, f"fit_lm {kinds}")
    assert r32["verdicts"]["converged"], "the problem was drawn to converge comfortably"
    for i, key in enumerate(cases.KEYS):
        t32, t64 = fit_lm_tolerance(r32, i), fit_lm_tolerance(r64, i)
        p32, p64 = r32["answers"]["params"][i], r64["answers"]["params"][i]
        assert abs(p32 - r32["answers"]["truth"][i]) <= t32, (key, p32, t32)
        assert abs(p64 - r64["answers"]["truth"][i]) <= t64, (key, p64, t64)
        assert abs(p32 - p64) <= t32 + t64 + abs(r32["answers"]["truth"][i]
                                                 - r64["answers"]["truth"][i]), (key, p32, p64)


_WORKER: list[Float64Worker] = []


def _worker() -> Float64Worker:
    return _WORKER[0]


@pytest.fixture(scope="module", autouse=True)
def _current_worker(worker):
    """The module's worker, for the property tests (Hypothesis refuses a
    function-scoped fixture)."""
    _WORKER[:] = [worker]
    yield
    _WORKER.clear()


_FIT_CASES = [
    (["clip", "log", "logit"], [0.5, 0.3, 0.6], [0.9, 0.7, 0.2]),
    (["logit", "clip", "log"], [0.35, 0.65, 0.5], [0.1, 0.2, 0.9]),
]


@pytest.mark.parametrize("kinds, truth_at, start_at", _FIT_CASES)
def test_fit_lm_agrees_across_precisions(kinds, truth_at, start_at):
    check_fit_lm(kinds, truth_at, start_at)


def test_float32_leaves_in_an_x64_graph_fit_as_float32_does():
    """A graph built from float32 values keeps them float32 under x64; its
    fit must give float32's verdicts and an answer within float32's
    tolerance of float32's (the fitter's own coordinates are float64)."""
    kinds, truth_at, start_at = _FIT_CASES[0]
    r32, rmix = both(_worker(), "blocks_fit_lm", kinds=kinds, truth_at=truth_at,
                     start_at=start_at, x64_args={"leaf_dtype": "float32"})
    assert "float32" in rmix["leaf_dtypes"], rmix["leaf_dtypes"]
    assert_same_verdicts(r32, rmix, "fit_lm, float32 leaves under x64")
    for i, key in enumerate(cases.KEYS):
        t = fit_lm_tolerance(r32, i)
        assert abs(r32["answers"]["params"][i] - rmix["answers"]["params"][i]) <= 2 * t, key


def test_adam_agrees_across_precisions():
    """``fit`` stops when the loss reaches ``tol``: each answer is within
    ``||J^+|| sqrt(2 tol)`` of the truth (``0.5 ||r||^2 <= tol``) plus its
    floor, and the two within the sum."""
    kinds, truth_at, start_at = _FIT_CASES[0]
    r32, r64 = both(_worker(), "blocks_fit", kinds=kinds, truth_at=truth_at,
                    start_at=start_at)
    assert_same_verdicts(r32, r64, "fit (Adam)")
    assert r32["verdicts"]["converged"]
    for i, key in enumerate(cases.KEYS):
        tols = [r["pinv_norm"] * math.sqrt(2 * r["tol"]) + floor_tolerance(r) for r in (r32, r64)]
        for r, t in zip((r32, r64), tols):
            assert abs(r["answers"]["params"][i] - r["answers"]["truth"][i]) <= t, (key, r, t)
        assert abs(r32["answers"]["params"][i] - r64["answers"]["params"][i]) <= sum(tols), key


# ---------------------------------------------------------------------------
# The FIM
# ---------------------------------------------------------------------------

def fim_eigenvalue_tolerance(r32: dict, r64: dict) -> float:
    """Weyl: an eigenvalue moves by at most ``||F32 - F64||_2``.  ``F =
    J^T J`` in float32 is ``N``-term dot products of entries each
    ``BLOCK_OPS`` roundings from their value, and ``eigh`` is backward
    stable to ``2 n eps ||F||``: ``(N + 2 n + 2 BLOCK_OPS) eps32
    ||J||_F^2``, and ``||J||_F^2`` is the trace, the eigenvalues' sum."""
    n = len(r64["answers"]["eigvals"])
    return ((r32["n_residuals"] + 2 * n + 2 * cases.BLOCK_OPS) * r32["eps"]
            * sum(r64["answers"]["eigvals"]))


@pytest.mark.parametrize("kinds, truth_at", [(c[0], c[1]) for c in _FIT_CASES])
def test_fim_agrees_across_precisions(kinds, truth_at):
    r32, r64 = both(_worker(), "blocks_fim", kinds=kinds, truth_at=truth_at)
    assert_same_verdicts(r32, r64, "fim")
    tol = fim_eigenvalue_tolerance(r32, r64)
    for a, b in zip(sorted(r32["answers"]["eigvals"]), sorted(r64["answers"]["eigvals"])):
        assert abs(a - b) <= tol, (a, b, tol)


# ---------------------------------------------------------------------------
# windowed_loss
# ---------------------------------------------------------------------------

#: Floating operations in one step of the spring's update (the spring force,
#: the damping, the acceleration, the velocity and the position).
SPRING_OPS = 10


def test_windowed_loss_agrees_across_precisions():
    """Exactly zero at the generating parameters in both; elsewhere within
    the forward error of a stable integrator's float32 trajectory: each
    compared sample is at most ``(window + n_steps) * SPRING_OPS``
    roundings of the state's magnitude from its float64 value (the window's
    own steps, and the record's, which came from one run), and ``sum d^2``
    moves by at most ``2 sqrt(M L) delta + M delta^2``."""
    r32, r64 = both(_worker(), "spring_windowed_loss", scale=1.1)
    assert_same_verdicts(r32, r64, "windowed_loss")
    assert r32["verdicts"]["zero_at_truth"]
    m = r32["n_compared"]
    delta = (r32["window"] + m) * SPRING_OPS * r32["eps"] * r32["max_abs_state"]
    loss = r64["answers"]["off_truth"]
    tol = 2 * math.sqrt(m * loss) * delta + m * delta ** 2
    assert abs(r32["answers"]["off_truth"] - loss) <= tol, (r32["answers"], loss, tol)


# ---------------------------------------------------------------------------
# Coupling: the report's verdicts, the state, an IFT gradient
# ---------------------------------------------------------------------------

def check_pair_coupling(ka: float, kb: float, tolerance: float) -> None:
    """One coupled step from the same state.  The verdicts are equal; each
    state is within its ``spectral_error_bound`` of the step's exact fixed
    point (the bound adds the residual's float floor), so the two are
    within the sum; ``rho_spectral`` agrees to its Arnoldi residuals plus
    ``sqrt(8 eps32)`` (a nearly defective Jacobian moves its eigenvalue by
    the square root of a perturbation; ``testing_standards.md``)."""
    r32, r64 = both(_worker(), "pair_coupling", ka=ka, kb=kb, tolerance=tolerance)
    assert_same_verdicts(r32, r64, f"coupling_diagnostics() ka={ka} kb={kb} tol={tolerance}")
    assert r32["verdicts"]["converged"] and not r32["verdicts"]["precision_limited"]
    gap = float(np.linalg.norm(np.subtract(r32["answers"]["state"], r64["answers"]["state"])))
    assert gap <= r32["spectral_error_bound"] + r64["spectral_error_bound"], (
        gap, r32["spectral_error_bound"], r64["spectral_error_bound"])
    rho_tol = r32["spectral_residual"] + r64["spectral_residual"] + math.sqrt(8 * r32["eps"])
    assert abs(r32["answers"]["rho_spectral"] - r64["answers"]["rho_spectral"]) <= rho_tol


def test_the_coupling_report_agrees_across_precisions():
    check_pair_coupling(5000.0, 4000.0, 1e-3)


def check_ift_gradient(ka: float, kb: float, tolerance: float) -> None:
    """Each gradient is within its own ``gradient_relative_error_bound`` of
    the fixed point's (documented: ``|g_k - g*| <= bound * |g_k|``), plus
    the float32 tangent's rounding: ``2 PASS_TERMS eps32`` per pass,
    amplified by the resolvent ``1 / (1 - rho)``."""
    r32, r64 = both(_worker(), "pair_ift_gradient", ka=ka, kb=kb, tolerance=tolerance)
    assert_same_verdicts(r32, r64, "IFT gradient")
    g32, g64 = r32["answers"]["gradient"], r64["answers"]["gradient"]
    rounding = 2 * cases.PASS_TERMS * r32["eps"] / (1 - r64["rho_spectral"]) * abs(g64)
    tol = (r32["gradient_relative_error_bound"] * abs(g32)
           + r64["gradient_relative_error_bound"] * abs(g64) + rounding)
    assert abs(g32 - g64) <= tol, (g32, g64, tol)


# Per push: tests/core/test_coupling_ift_gradient_in_any_units.py::test_the_tangent_and_the_adjoint_are_the_same_at_every_scale
# (the IFT derivative against its own tangent, on every push) and
# tests/property/test_differential_precision.py::test_the_coupling_report_agrees_across_precisions
# (the same pair's forward solve across precisions).
@pytest.mark.slow  # a gradient through a diagnostics group compiled on both sides: 7-8 s
def test_an_ift_gradient_agrees_across_precisions():
    check_ift_gradient(5000.0, 4000.0, 1e-3)


# ---------------------------------------------------------------------------
# fit_multiple_shooting
# ---------------------------------------------------------------------------

def check_multiple_shooting(truth_damping: float, start_damping: float) -> None:
    """Each answer within ``||J_w^+|| sqrt(best_loss)`` of the truth: the
    windowed residual is ``J_w (c - c*)`` to first order (the window states
    barely move), and its norm is the square root of the loss."""
    r32, r64 = both(_worker(), "spring_fit_multiple_shooting", truth_damping=truth_damping,
                    start_damping=start_damping)
    assert_same_verdicts(r32, r64, "fit_multiple_shooting")
    tols = [r["pinv_norm"] * math.sqrt(r["best_loss"]) + 16 * r["eps"] * truth_damping
            for r in (r32, r64)]
    for r, t in zip((r32, r64), tols):
        assert abs(r["answers"]["damping"] - truth_damping) <= t, (r, t)
    assert abs(r32["answers"]["damping"] - r64["answers"]["damping"]) <= sum(tols)


# Per push: tests/property/test_differential_precision.py::test_adam_agrees_across_precisions
# (the same Adam loop, on a closed-form problem).
@pytest.mark.slow  # a multiple-shooting fit compiled on both sides: over 5 s on CI
def test_fit_multiple_shooting_agrees_across_precisions():
    check_multiple_shooting(1.0, 3.0)


# ---------------------------------------------------------------------------
# Known failing
# ---------------------------------------------------------------------------

_H1_REASON = ("B1-H1: under x64 the identifiability guard misses the spring's exact (k, c, m) "
              "scale degeneracy: _ExcitationTracker.split (sysid.py ~2984) compares a float64 "
              "eigh's resolution with a cutoff below it, and takes eps from the promoted "
              "theta; pending fix")


@pytest.mark.parametrize("args", [
    dict(fitter="fit_lm", n_iter=10),
    dict(fitter="fit", n_iter=50, lr=0.2, noise=0.02),
], ids=["fit_lm-noiseless", "adam-noisy"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_H1_REASON)
def test_the_scale_degeneracy_is_held_in_both_precisions(args):
    r32, r64 = both(_worker(), "spring_scale_guard", **args)
    assert_same_verdicts(r32, r64, "the identifiability guard")
    for r in (r32, r64):
        assert abs(r["answers"]["scale_drift"]) <= 1e-6, r


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_H1_REASON)
def test_float32_leaves_in_an_x64_graph_hold_the_scale_degeneracy_as_float32_does():
    r32, rmix = both(_worker(), "spring_scale_guard", fitter="fit", n_iter=50, lr=0.01,
                     noise=0.02, x64_args={"leaf_dtype": "float32"})
    assert_same_verdicts(r32, rmix, "the identifiability guard, float32 leaves under x64")


@pytest.mark.parametrize("truth_damping", [1e-6, 0.0])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "B1-L1: under x64 fit_lm reports converged=False at its float64 floor: the floor rule's "
    "damping ladder (sysid.py ~4471) ends one rung short of a candidate within step_tol; "
    "pending fix"))
def test_fit_lm_is_converged_at_its_floor_in_both_precisions(truth_damping):
    r32, r64 = both(_worker(), "fit_lm_at_its_floor", truth_damping=truth_damping)
    assert_same_verdicts(r32, r64, "fit_lm at its floor")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_H1_REASON)
def test_a_coordinate_started_at_its_truth_is_counted_alike_in_both_precisions():
    """Found by the drawn ``fit_lm`` property at ``ci`` depth.  A well-posed
    fit whose stiffness (an identity coordinate) starts exactly at its
    truth: its residual block is zero throughout, so no evaluated gradient
    points along it.  float32 reports ``excited_rank=2``; float64 counts the
    never-excited direction (3), its zero eigenvalue read at ``eigh``'s
    resolution against a cutoff far below it -- B1-H1's mechanism on a problem
    with no degeneracy at all."""
    r32, r64 = both(_worker(), "blocks_fit_lm", kinds=["clip", "logit", "log"],
                    truth_at=[0.5, 0.75, 0.78125], start_at=[0.5, 0.8, 0.5])
    assert_same_verdicts(r32, r64, "fit_lm with a coordinate started at its truth")


def test_the_known_failing_cases_converge_in_float32():
    """The float32 halves of B1-H1 and B1-L1 hold today: the scale is held at
    rank 2, and the floor fit converges.  (So the strict xfails above fail
    on the float64 side, as their reasons say.)"""
    r = cases.run_case("spring_scale_guard", {"fitter": "fit_lm", "n_iter": 10})
    assert r["verdicts"]["excited_rank"] == 2 and abs(r["answers"]["scale_drift"]) <= 1e-6, r
    assert cases.run_case("fit_lm_at_its_floor", {"truth_damping": 1e-6})["verdicts"][
        "converged"]


# ---------------------------------------------------------------------------
# Slow lane: drawn problems
# ---------------------------------------------------------------------------

_AWAY = st.floats(0.2, 0.8)


@st.composite
def _starts(draw, truth_at):
    """A start at least 0.05 of each range from the truth.  A coordinate
    started *at* its truth is a decision threshold: its residual block is
    zero, so no gradient the run evaluates points along it, and whether it
    counts as excited is decided by rounding (found at ``ci`` depth; the
    float64 half of it is B1-H1's, pinned below)."""
    out = []
    for t in truth_at:
        d = draw(st.floats(0.05, 0.4))
        sign = draw(st.sampled_from([1.0, -1.0]))
        s = t + sign * d
        out.append(s if 0.02 <= s <= 0.98 else t - sign * d)
    return out


# Per push: tests/property/test_differential_precision.py::test_fit_lm_agrees_across_precisions
@pytest.mark.slow  # a fit compiled in each process per example
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(kinds=st.lists(st.sampled_from(sorted(cases.SPECS)), min_size=3, max_size=3),
       data=st.data(), truth_at=st.lists(_AWAY, min_size=3, max_size=3))
def test_fit_lm_agrees_across_precisions_on_drawn_problems(kinds, data, truth_at):
    check_fit_lm(kinds, truth_at, data.draw(_starts(truth_at), label="start_at"))


# Per push: tests/property/test_differential_precision.py::test_fim_agrees_across_precisions
@pytest.mark.slow  # a FIM traced in each process per example
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(kinds=st.lists(st.sampled_from(sorted(cases.SPECS)), min_size=3, max_size=3),
       truth_at=st.lists(_AWAY, min_size=3, max_size=3))
def test_fim_agrees_across_precisions_on_drawn_problems(kinds, truth_at):
    r32, r64 = both(_worker(), "blocks_fim", kinds=kinds, truth_at=truth_at)
    assert_same_verdicts(r32, r64, "fim")
    tol = fim_eigenvalue_tolerance(r32, r64)
    for a, b in zip(sorted(r32["answers"]["eigvals"]), sorted(r64["answers"]["eigvals"])):
        assert abs(a - b) <= tol, (a, b, tol)


# Per push: tests/property/test_differential_precision.py::test_the_coupling_report_agrees_across_precisions
@pytest.mark.slow  # a diagnostics group compiled in each process per example
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(ka=st.floats(3000.0, 8000.0), kb=st.floats(2000.0, 6000.0),
       tolerance=st.sampled_from([1e-3, 3e-3, 1e-2]))
def test_the_coupling_report_agrees_across_precisions_on_drawn_pairs(ka, kb, tolerance):
    check_pair_coupling(ka, kb, tolerance)
