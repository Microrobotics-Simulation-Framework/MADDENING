"""A search for wrong ``fit_lm`` fits, between the points a grid samples.

``test_sysid_truth_recovery.py`` holds ``fit_lm`` to the truth over a fixed
grid of the parameter guide's spring.  This module searches the same
problem with ``tests/property/targeted_search.py``: every number is drawn
from a continuous range, three scores say how wrong a fit is, and
Hypothesis climbs each of them.

The problem.  The guide's spring (stiffness 30, damping 1.9, mass 1, 100
noiseless position samples) with the stiffness under its own ``log`` spec
and the damping under the transform an example names, and drawn:

* the transform of the damping: none (clipped bounds), ``log``, ``log``
  from a non-zero lower bound, ``logit``;
* the precision (float32, x64) and whether the leaves the fit leaves alone
  are frozen by their specs or left out by ``mask=`` (the rest length then
  under a ``logit`` wider than its value, by a drawn factor of 3 to 1e6);
* the width of the damping's range (0.03 to 100 times the damping) and
  where the truth sits in it, down to a few float spacings from an edge;
* the start of each parameter, up to 100 times off either way (a
  ``logit`` or clipped damping: anywhere in its range, edges included);
* the parameters' unit, over 30 decades, and the residual's, over 24.

Numbers are arguments of the compiled programs, not constants of them.
The unit of the parameters is the *mass*: a spring of mass ``u`` with
stiffness ``30 u`` and damping ``1.9 u`` moves as the guide's does, so one
record serves every unit and the truth is known to the rounding of ``k / m``
(a part in 1e7 in float32, far inside the scores' thresholds).  The unit of
the residual is a leaf of a second, uncoupled node that the residual
multiplies by.  Both are leaves of the ``params`` tree the fitter hands its
model, so the model-side programs (``sysid._compile_model``: the residual
and its Jacobian) are compiled once for each precision and each way of
leaving leaves out and shared by every example -- **four pairs of
model-side compiles in a run of all three searches**, whatever the number
of examples.  What is not shared is the library's own: its map from the
optimiser's coordinates to the parameters is compiled once per distinct
``ParamSpec`` (the bounds are constants of a spec), three scalar programs
for each range drawn.

The scores, each 0 where there is nothing to say:

a. **converged at a wrong point**: for a fit that reports
   ``converged=True`` and gave no warning about an edge of its transform's
   range, its relative parameter error (SYS-080, SYS-082);
b. **worse than the plain fit**: where the same fit with the damping under
   ``transform=None`` and the same bounds recovers the truth, the
   transformed fit's relative error, unless it warned by name that it
   ended on (or started on) the edge of its transform's range (SYS-144);
c. **returned is not evaluated**: the relative difference between
   ``best_loss`` and the loss of the fitter's own residual program at the
   parameters returned, and -- counted as at least 1 -- any difference at
   all between a leaf the fit was not to move and the array that went in
   (SYS-054).

Per push: each search, derandomised, at the house floor of examples.  Slow:
the random hunt.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st
from jax.flatten_util import ravel_pytree

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit_lm

from tests.property.sysid_transform_grid import (
    DT, EDGE_START_WARNING, EDGE_WARNING, MASS, N_STEPS, REST_LENGTH, STIFFNESS, TRUTH,
    precision,
)
from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

TRANSFORMS = ("identity", "log", "log-from-lo", "logit")
#: A fit is at the truth within this, of each parameter (of the damping's
#: range where that is the narrower): the grid's tolerance for ``fit_lm``.
TOLERANCE = 1e-3
#: ``best_loss`` against the loss at the returned parameters, relative: the
#: tolerance ``FitResult.hold_declined`` documents for a held point, in the
#: working precision's ``eps``.
LOSS_ULPS = 2.0 ** 10
#: How near an edge a drawn position may sit, in float spacings of the
#: bounds' own size (the grid's ``EDGE_ULPS``).
EDGE_ULPS = 8.0


@dataclass(frozen=True)
class Case:
    """One drawn problem; every float is a ratio or a fraction."""

    transform: str
    x64: bool
    masked: bool
    #: The damping's range, in units of the true damping.
    width: float
    #: Where in the range the truth and the start sit: a fraction of it.
    truth_at: float
    start_at: float
    #: The start of a ``log`` damping over its truth (measured from its
    #: lower bound), and of the stiffness over its truth.
    damping_off: float
    stiffness_off: float
    #: The unit of the parameters (the mass) and of the residual.
    unit: float
    residual_scale: float
    #: ``masked``: the rest length's ``logit`` bounds over its value.
    masked_range: float

    @property
    def eps(self) -> float:
        return float(np.finfo(np.float64 if self.x64 else np.float32).eps)

    @property
    def damping(self) -> float:
        return TRUTH * self.unit

    @property
    def stiffness(self) -> float:
        return STIFFNESS * self.unit

    @property
    def bounds(self) -> tuple:
        """``(lo, hi)`` with the truth at ``truth_at`` of it; a plain
        ``log`` leaf's is ``(0, hi)`` and its ``hi`` only places a start."""
        width = self.width * self.damping
        if self.transform == "log":
            return 0.0, self.damping / self._inside(self.truth_at, 0.0, width)
        lo = self.damping - self._inside(self.truth_at, self.damping - width, width) * width
        return lo, lo + width

    def _inside(self, fraction: float, lo: float, width: float) -> float:
        """``fraction`` kept ``EDGE_ULPS`` float spacings of the bounds'
        size inside ``(0, 1)``: the interior ``ParamSpec.check`` documents."""
        gap = EDGE_ULPS * self.eps * max(abs(lo), abs(lo + width), width) / width
        return min(max(fraction, gap), 1.0 - gap)

    @property
    def start(self) -> tuple:
        """``(stiffness, damping)`` the fit starts from."""
        lo, hi = self.bounds
        if self.transform in ("log", "log-from-lo"):
            damping = lo + (self.damping - lo) * self.damping_off
        else:
            damping = lo + self._inside(self.start_at, lo, hi - lo) * (hi - lo)
        return self.stiffness * self.stiffness_off, damping

    def spec(self, control: bool = False) -> ParamSpec:
        lo, hi = self.bounds
        if self.transform == "logit" and not control:
            return ParamSpec(bounds=(lo, hi), transform="logit")
        if self.transform in ("log", "log-from-lo"):
            return ParamSpec(bounds=(lo, None), transform=None if control else "log")
        return ParamSpec(bounds=(lo, hi))


def _decades(low: float, high: float):
    return st.floats(low, high).map(lambda x: 10.0 ** x)


#: A fraction of a range: anywhere, or within 1e-2 to 1e-14 of an edge
#: (:meth:`Case._inside` keeps it the documented margin inside).
_FRACTION = st.one_of(
    st.floats(0.02, 0.98),
    st.builds(lambda gap, upper: 1.0 - gap if upper else gap, _decades(-14.0, -2.0),
              st.booleans()),
)


def cases(transforms=TRANSFORMS):
    return st.builds(
        Case, transform=st.sampled_from(transforms), x64=st.booleans(), masked=st.booleans(),
        width=_decades(-1.5, 2.0), truth_at=_FRACTION, start_at=_FRACTION,
        damping_off=_decades(-2.0, 2.0), stiffness_off=_decades(-2.0, 2.0),
        # ``fit_lm`` documents every residual and Jacobian entry normal: in
        # float32 a column is the residual's unit over the parameter's
        # times 1e-6 to 1, so 27 decades between them at most.
        unit=_decades(-15.0, 15.0), residual_scale=_decades(-12.0, 12.0),
        masked_range=_decades(0.5, 6.0))


# ---------------------------------------------------------------------------
# The spring, once per precision and per way of leaving leaves out
# ---------------------------------------------------------------------------

@dataclass
class _Problem:
    gm: GraphManager
    params: dict
    residual: object
    mask: object
    #: The leaves the fit is not to move, as ``(node, key)``.
    kept: tuple


_PROBLEMS: dict = {}
#: The model-side programs, per ``(x64, masked)`` and per the order a fit
#: asks for them in (the residual first).
_PROGRAMS: dict = {}


def _problem(x64: bool, masked: bool) -> _Problem:
    key = (x64, masked)
    if key in _PROBLEMS:
        return _PROBLEMS[key]
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, initial_position=0.5, stiffness=STIFFNESS,
                                 damping=TRUTH, mass=MASS, rest_length=REST_LENGTH))
    # Carries the residual's unit in its rest length; coupled to nothing.
    gm.add_node(SpringDamperNode("units", DT, stiffness=1.0, damping=1.0, mass=1.0,
                                 rest_length=1.0, initial_position=1.0))
    gm.compile()
    kept = [("spring", "mass"), ("spring", "rest_length")]
    kept += [("units", k) for k in gm.params["nodes"]["units"]]
    if not masked:
        for node, leaf in kept:
            gm.set_param_spec(node, leaf, ParamSpec(trainable=False))
    init = {name: {k: v[None] for k, v in gm.get_node_state(name).items()}
            for name in ("spring", "units")}

    def positions(p):
        return gm.run_sweep(N_STEPS, init, return_history=True, params=p)[1]["spring"][
            "position"][0]

    params = jax.tree.map(lambda x: x, gm.params)
    measured = positions(params)

    def residual(p):
        return p["nodes"]["units"]["rest_length"] * (positions(p) - measured)

    mask = None
    if masked:
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["spring"]["stiffness"] = True
        mask["nodes"]["spring"]["damping"] = True
    _PROBLEMS[key] = _Problem(gm, params, residual, mask, tuple(kept))
    return _PROBLEMS[key]


@contextlib.contextmanager
def _shared_programs(key):
    """One compile of each model-side program per ``key``: a fit's
    ``sysid._compile_model`` calls are answered, in the order it makes
    them, from :data:`_PROGRAMS`.  (A tree without ``_compile_model`` --
    one from before the fitters compiled their model apart from their
    transforms -- runs unshared.)"""
    real = getattr(sysid, "_compile_model", None)
    if real is None:
        yield
        return
    asked = [0]

    def compile_once(fn):
        slot = (key, asked[0])
        asked[0] += 1
        if slot not in _PROGRAMS:
            _PROGRAMS[slot] = real(fn)
        return _PROGRAMS[slot]

    sysid._compile_model = compile_once  # noqa: SLF001
    try:
        yield
    finally:
        sysid._compile_model = real  # noqa: SLF001


def _own_residual(key, problem):
    """The fitter's own compiled residual where it compiles one apart
    (the first program a fit asks for); this module's otherwise."""
    if (key, 0) not in _PROGRAMS:
        _PROGRAMS[key, 0] = jax.jit(lambda p: ravel_pytree(problem.residual(p))[0])
    return _PROGRAMS[key, 0]


@dataclass
class Outcome:
    """One fit.  ``refused``: it raised ``FloatingPointError`` (a start the
    explicit step overflows from)."""

    refused: bool = False
    converged: bool = False
    stiffness: float = float("nan")
    damping: float = float("nan")
    #: The larger relative error of the two parameters.
    error: float = float("inf")
    edge_warned: bool = False
    best_loss: float = float("nan")
    #: The fitter's residual program's loss at the returned parameters.
    loss_returned: float = float("nan")
    #: ``(node, key, went in, came back)`` for each kept leaf that differs.
    moved: tuple = ()
    n_iter: int = 0

    @property
    def recovered(self) -> bool:
        return self.error <= TOLERANCE


def run_fit(case: Case, control: bool = False) -> Outcome:
    """``fit_lm`` on ``case``; ``control``: with the damping under
    ``transform=None`` and the same bounds."""
    key = (case.x64, case.masked)
    with precision(case.x64):
        problem = _problem(*key)
        gm = problem.gm
        gm.set_param_spec("spring", "damping", case.spec(control))
        if case.masked:
            gm.set_param_spec("spring", "rest_length", ParamSpec(
                bounds=(-case.masked_range * REST_LENGTH, case.masked_range * REST_LENGTH),
                transform="logit"))
        start = jax.tree.map(lambda x: x, problem.params)

        def put(node, leaf, value):
            start["nodes"][node][leaf] = jnp.asarray(value, start["nodes"][node][leaf].dtype)

        k0, c0 = case.start
        put("spring", "stiffness", k0)
        put("spring", "damping", c0)
        put("spring", "mass", MASS * case.unit)
        put("units", "rest_length", case.residual_scale)
        try:
            with _shared_programs(key), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = fit_lm(gm, problem.residual, params=start, mask=problem.mask)
        except FloatingPointError:
            return Outcome(refused=True)
        leaves = res.params["nodes"]["spring"]
        k, c = float(leaves["stiffness"]), float(leaves["damping"])
        lo, hi = case.bounds
        # The damping's own scale: itself, or its range where that is narrower.
        scale = case.damping if case.spec(control).bounds[1] is None else min(
            case.damping, hi - lo)
        error = max(abs(k - case.stiffness) / case.stiffness, abs(c - case.damping) / scale)
        moved = []
        for node, leaf in problem.kept:
            went_in, came_back = start["nodes"][node][leaf], res.params["nodes"][node][leaf]
            a, b = np.asarray(went_in), np.asarray(came_back)
            if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                moved.append((node, leaf, a.tolist(), b.tolist()))
        loss = float(sysid._half_squared_norm(  # noqa: SLF001
            _own_residual(key, problem)(res.params)))
        texts = [str(w.message) for w in caught]
        return Outcome(
            converged=bool(res.converged), stiffness=k, damping=c,
            error=error if np.isfinite(error) else float("inf"),
            edge_warned=any(EDGE_WARNING in t or EDGE_START_WARNING in t for t in texts),
            best_loss=float(res.best_loss), loss_returned=loss, moved=tuple(moved),
            n_iter=int(res.n_iter))


# ---------------------------------------------------------------------------
# The scores
# ---------------------------------------------------------------------------

def converged_at_a_wrong_point(case: Case):
    """(a) ``converged=True`` is the truth, or an edge the fit warned of."""
    fit = run_fit(case)
    wrong = fit.converged and not fit.edge_warned
    return (fit.error if wrong else 0.0), fit


def worse_than_the_plain_fit(case: Case):
    """(b) Where the control recovers the truth, the transformed fit does,
    or warns by name about an edge of its transform's range."""
    control = run_fit(case, control=True)
    if not control.recovered:
        return 0.0, ("the control did not recover", control)
    fit = run_fit(case)
    if fit.refused:
        return float("inf"), ("refused where the control recovered", fit)
    return (0.0 if fit.edge_warned else fit.error), fit


def returned_is_not_evaluated(case: Case):
    """(c) ``best_loss`` is the loss of the parameters returned, and a leaf
    the fit was not to move is the array that went in."""
    fit = run_fit(case)
    if fit.refused:
        return 0.0, fit
    floor = max(abs(fit.best_loss), abs(fit.loss_returned))
    apart = abs(fit.best_loss - fit.loss_returned) / floor if floor > 0.0 else 0.0
    # In units of the documented tolerance, so one threshold serves both
    # precisions; a moved leaf is over it whatever the losses say.
    score = apart / (LOSS_ULPS * case.eps)
    return (max(score, 2.0) if fit.moved else score), fit


#: ``(score, strategy, threshold)``.
SEARCHES = {
    "converged-at-a-wrong-point": (converged_at_a_wrong_point, cases(), TOLERANCE),
    "worse-than-the-plain-fit": (worse_than_the_plain_fit, cases(TRANSFORMS[1:]), TOLERANCE),
    "returned-is-not-evaluated": (returned_is_not_evaluated, cases(), 1.0),
}


@pytest.mark.parametrize("name", sorted(SEARCHES))
def test_no_wrong_fit_among_the_fixed_draws(name):
    score, strategy, threshold = SEARCHES[name]
    report = targeted_search(strategy, score, threshold, profile=PER_PUSH, label=name)
    assert report.examples >= PER_PUSH.max_examples // 2, str(report)


# Per push: tests/property/test_sysid_targeted_search.py::test_no_wrong_fit_among_the_fixed_draws
@pytest.mark.slow  # hundreds of fits, two an example for the control: minutes
@pytest.mark.parametrize("name", sorted(SEARCHES))
def test_no_wrong_fit_found_by_the_search(name):
    score, strategy, threshold = SEARCHES[name]
    targeted_search(strategy, score, threshold, profile=SLOW, label=name)
