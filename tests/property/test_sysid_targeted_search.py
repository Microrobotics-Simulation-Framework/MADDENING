"""A search for wrong ``fit_lm`` fits, between the points a grid samples.

``test_sysid_truth_recovery.py`` holds ``fit_lm`` to the truth over a fixed
grid of the parameter guide's spring.  This module searches the same
problem with ``tests/property/targeted_search.py``: every number is drawn
from a continuous range, three scores say how wrong a fit is, and
Hypothesis climbs each of them.

The problem.  The guide's spring (100 noiseless position samples, mass
frozen) with the stiffness under its own ``log`` spec and the damping under
the transform an example names, and drawn:

* the truth: a stiffness of 9.5 to 95 and a damping of 0.6 to 6 per unit
  mass (the guide's are 30 and 1.9), every one of them underdamped;
* the transform of the damping: none (clipped bounds), ``log``, ``log``
  from a non-zero lower bound, ``logit``;
* the precision (float32, x64) and whether the leaves the fit leaves alone
  are frozen by their specs or left out by ``mask=`` (the rest length then
  under a ``logit`` wider than its value, by a drawn factor of 3 to 1e6);
* the width of the damping's range (0.03 to 100 times the damping) and
  where the truth sits in it, down to a few float spacings from an edge;
* the start of each parameter, up to 100 times off either way (a
  ``logit`` or clipped damping: anywhere in its range, edges included;
  the stiffness 0.1 to 30 times its truth for score (b), the span its
  claim is made over);
* the parameters' unit, over 30 decades, and the residual's, over 24.

Numbers are arguments of the compiled programs, not constants of them.
The unit of the parameters is the *mass*: a spring of mass ``u`` with
stiffness ``k u`` and damping ``c u`` moves as one of mass 1 with ``k`` and
``c`` does.  A second, uncoupled node, which no fit moves, carries the
rest: the true stiffness and damping in its own, and the residual's unit in
its rest length.  The residual is that unit times the difference of two
rollouts of the spring, one at the parameters it is handed and one with the
truth put in their place -- so the record is noiseless and the truth is
known exactly, to the bit.  All of these are leaves of the ``params`` tree
the fitter hands its model, so the model-side programs
(``sysid._compile_model``: the residual and its Jacobian) are compiled once
for each precision and each way of leaving leaves out and shared by every
example -- **four pairs of model-side compiles in a run of the three
searches of the spring** (and one for the ball's), whatever the number of examples (about 2.7 s a pair on three
cores; a fit is then 0.04 s).  What is not shared is the library's own: its
map from the optimiser's coordinates to the parameters is compiled once per
distinct ``ParamSpec`` (the bounds are constants of a spec) -- three scalar
programs, about 0.1 s together, for each range drawn.

The scores, each 0 where there is nothing to say:

a. **converged at a wrong point**: for a fit that reports
   ``converged=True``, its relative parameter error (SYS-080, SYS-082);
b. **worse than the plain fit**: where the same fit with the damping under
   ``transform=None`` and the same bounds recovers the truth in the default
   50 iterations, the transformed fit's relative error after at most four
   times as many, unless it warned by name that it ended on the edge of
   its transform's range (SYS-144, SYS-145)

   -- in both, 0 for a fit that ended at an optimum that is not the truth,
   which the spring has (another basin, from a stiffness started far too
   high; a bound, or an edge of the transform's range the fit warned of by
   name, that the loss pushes onto).  What tells them from a wrong ending
   is whether the loss still falls beside the returned point
   (:func:`_descent_nearby`);

c. **returned is not evaluated**: the relative difference between
   ``best_loss`` and the loss of the fitter's own residual program at the
   parameters returned, and -- counted as at least 1 -- any difference at
   all between a leaf the fit was not to move and the array that went in
   (SYS-054);
d. **converged beside a lower loss**, on a second problem, whose residual
   is not differentiable: the guide's ``TableNode -> BallNode`` graph (200
   float32 steps; a second ball on the same table, trained by nothing,
   carries the truth and is the record), with the truth's elasticity and
   gravity and both starts drawn.  A bounce moves by one time step as a
   parameter moves, so the loss is piecewise smooth with jumps
   (MADD-ANO-021), and Levenberg-Marquardt is drawn along a piece to its
   edge.  For a fit that reports ``converged=True``: the largest fall of
   the loss at a point 1e-4 of each parameter away from the returned one
   in the optimiser's coordinates (both parameters are fitted as they
   are), in units of what the residual's rounding can move the loss by --
   counted where the loss falls *along the piece the point is on*, a tenth
   as far at a tenth of the distance (SYS-080: ``converged`` is not
   reported where a step still lowers the loss).  A fall that does not
   shrink with the distance is a jump to another piece, and a converged
   fit beside one is at an optimum of its own piece and scores 0.  The
   hunt found two kinds: the interior minimum of a piece that is not the
   truth's (truth 0.6, started at 0.5 with the true gravity: loss 5.1e-3,
   flat to rounding out to 1e-4, the truth's piece 1e-3 away), and a fit
   within ``step_tol`` of the truth on the far side of the jump that sits
   at the truth itself (truth 0.3, started at 0.75: the parameters right
   to 1e-6, the loss 2.0e-5 where 1e-10 is 1e-6 of the elasticity away).

Per push: each search, derandomised, at the house floor of examples.  Slow:
the random hunt.  What the hunt found is pinned at the foot of the module.

That the scores can fire was shown on the tree before the fixes they are
for (``fa12c585``): (b) found the ``logit`` edge trap and (c) the leaf left
out by ``mask=`` that the model was run with moved, each within tens of
examples (the pull request that added this module gives the counts).  And
(d) on ``77a04aaf``, before ``fit_lm``'s floor rule told a jump from the
rounding floor: the per-push draws find a fit converged beside a lower loss
at their third example, and random draws on seeds 1, 2 and 3 at examples 8,
17 and 13 of 20; none after it.
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
from maddening.nodes import BallNode, TableNode
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
#: ``fit_lm``'s default ``n_iter``, which the control runs for, and how
#: many times that the transformed fit is given in score (b).  A fit that
#: is still descending when its iterations run out says ``converged=False``
#: and is right; found by this search at the default: a ``logit`` fit that
#: took 55 iterations beside a control that took 43.  That is a cap, not a
#: trap, and four times the control's allowance tells them apart.
N_ITER, PATIENCE = 50, 4


@dataclass(frozen=True)
class Case:
    """One drawn problem; every float is a ratio or a fraction."""

    transform: str
    x64: bool
    masked: bool
    #: The truth, per unit mass.
    stiffness_true: float
    damping_true: float
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
        return self.damping_true * self.unit

    @property
    def stiffness(self) -> float:
        return self.stiffness_true * self.unit

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
            # Measured from the lower bound, and kept the same few float
            # spacings above it (a start that rounds onto it is refused).
            damping = lo + max((self.damping - lo) * self.damping_off,
                               EDGE_ULPS * self.eps * max(abs(lo), self.damping))
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


#: How far off the stiffness starts, in decades either way: everywhere,
#: and within the span the differential claim is made over (SYS-144: 0.1 to
#: 30 times the truth).  From further off ``fit_lm`` often stops short on
#: the end of a range under either parametrisation, and which one does is
#: chance (:func:`test_fit_lm_from_a_far_start_does_not_stop_short_on_the_end_of_a_range`).
FAR, CLAIMED = (-2.0, 2.0), (-1.0, float(np.log10(30.0)))


def cases(transforms=TRANSFORMS, stiffness_decades=FAR):
    return st.builds(
        Case, transform=st.sampled_from(transforms), x64=st.booleans(), masked=st.booleans(),
        # Shrinks to the guide's spring.
        stiffness_true=_decades(-0.5, 0.5).map(lambda x: STIFFNESS * x),
        damping_true=_decades(-0.5, 0.5).map(lambda x: TRUTH * x),
        width=_decades(-1.5, 2.0), truth_at=_FRACTION, start_at=_FRACTION,
        damping_off=_decades(-2.0, 2.0), stiffness_off=_decades(*stiffness_decades),
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
    # Carries the truth in its stiffness and damping and the residual's
    # unit in its rest length; coupled to nothing.
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

    def residual(p):
        carried = p["nodes"]["units"]
        spring = dict(p["nodes"]["spring"], stiffness=carried["stiffness"],
                      damping=carried["damping"])
        truth = dict(p, nodes=dict(p["nodes"], spring=spring))
        return carried["rest_length"] * (positions(p) - positions(truth))

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


#: The points around a returned one where :func:`_descent_nearby` reads the
#: loss: each of the eight directions, at these fractions of each
#: parameter's scale.
_RADII = (1e-4, 1e-2)
_DIRECTIONS = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))


def _descent_nearby(case: "Case", control: bool, program, params, best_loss: float,
                    edge_warned: bool) -> float:
    """The largest fall of the loss, as a fraction of ``best_loss``, over
    the feasible points around ``params``: the returned point moved by
    :data:`_RADII` of each parameter's scale in each of the eight
    directions.  Feasible: the damping inside its bounds and, for a fit
    that warned it ended on the edge of its transform's range, no nearer
    the bound it ended beside (which that transform cannot reach).  0 for a
    loss at the residual's rounding, where a fall is noise.

    This is what separates a wrong ending from a right one that is not the
    truth.  The spring's loss has more than one basin from a stiffness
    started far too high (a fast oscillation is best fitted by damping it
    out), and a truth nearer a bound than a ``logit`` resolves cannot be
    reached: a fit that stops in another basin, or on a bound or an edge
    the loss pushes it onto, has an optimum.  One that stops where the loss
    still falls has not."""
    lo, hi = case.spec(control).bounds
    leaves = params["nodes"]["spring"]
    k, c = float(leaves["stiffness"]), float(leaves["damping"])
    scale = case.damping if hi is None else min(case.damping, hi - lo)
    if edge_warned:
        if hi is not None and hi - c <= c - lo:
            hi = c
        else:
            lo = c
    # The loss of a residual of 2**5 float spacings of the positions.
    rounding = 0.5 * N_STEPS * (2.0 ** 5 * case.eps * case.residual_scale) ** 2
    if not best_loss > rounding:
        return 0.0
    lowest = best_loss
    for radius in _RADII:
        for i, j in _DIRECTIONS:
            c_near = max(c + j * radius * scale, lo)
            if hi is not None:
                c_near = min(c_near, hi)
            near = dict(leaves, stiffness=jnp.asarray(k * (1.0 + i * radius),
                                                      leaves["stiffness"].dtype),
                        damping=jnp.asarray(c_near, leaves["damping"].dtype))
            tree = dict(params, nodes=dict(params["nodes"], spring=near))
            loss = float(sysid._half_squared_norm(program(tree)))  # noqa: SLF001
            if np.isfinite(loss):
                lowest = min(lowest, loss)
    return (best_loss - lowest) / best_loss


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
    #: For a fit that is not at the truth: how far the loss falls, as a
    #: fraction of itself, at the best of the points around the returned one
    #: (:func:`_descent_nearby`).  0: a constrained local optimum.
    descent: float = 0.0
    eps: float = 0.0

    @property
    def recovered(self) -> bool:
        return self.error <= TOLERANCE

    @property
    def wrong(self) -> float:
        """The fit's relative error -- within :data:`TOLERANCE` at the
        truth, which leaves a search something to climb -- and 0 where it
        is at another optimum: off the truth with no fall of the loss
        beside it."""
        elsewhere = not self.recovered and self.descent <= LOSS_ULPS * self.eps
        return 0.0 if elsewhere else self.error


def run_fit(case: Case, control: bool = False, **options) -> Outcome:
    """``fit_lm`` on ``case``; ``control``: with the damping under
    ``transform=None`` and the same bounds.  ``options`` go to ``fit_lm``."""
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
        put("units", "stiffness", case.stiffness)
        put("units", "damping", case.damping)
        put("units", "mass", MASS * case.unit)
        put("units", "rest_length", case.residual_scale)
        try:
            with _shared_programs(key), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = fit_lm(gm, problem.residual, params=start, mask=problem.mask,
                             **options)
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
        texts = [str(w.message) for w in caught]
        edge_warned = any(EDGE_WARNING in t or EDGE_START_WARNING in t for t in texts)
        program = _own_residual(key, problem)
        loss = float(sysid._half_squared_norm(program(res.params)))  # noqa: SLF001
        descent = 0.0
        if not error <= TOLERANCE:
            descent = _descent_nearby(case, control, program, res.params, loss, edge_warned)
        return Outcome(
            converged=bool(res.converged), stiffness=k, damping=c,
            error=error if np.isfinite(error) else float("inf"),
            edge_warned=edge_warned, eps=case.eps,
            best_loss=float(res.best_loss), loss_returned=loss, moved=tuple(moved),
            n_iter=int(res.n_iter), descent=descent)


# ---------------------------------------------------------------------------
# The scores
# ---------------------------------------------------------------------------

def converged_at_a_wrong_point(case: Case):
    """(a) ``converged=True`` is an optimum: the truth, another basin's, or
    a bound or a warned-of edge the loss pushes onto."""
    fit = run_fit(case)
    return (fit.wrong if fit.converged else 0.0), fit


def worse_than_the_plain_fit(case: Case):
    """(b) Where the control recovers the truth, the transformed fit ends at
    an optimum too, converged or not, or says by name that it ended on the
    edge of its transform's range: no silent trap on the way."""
    control = run_fit(case, control=True, n_iter=N_ITER)
    if not control.recovered:
        return 0.0, ("the control did not recover", control)
    fit = run_fit(case, n_iter=PATIENCE * N_ITER)
    if fit.refused:
        return float("inf"), ("refused where the control recovered", fit)
    return (0.0 if fit.edge_warned else fit.wrong), fit


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


# ---------------------------------------------------------------------------
# The bouncing ball: a residual with jumps
# ---------------------------------------------------------------------------

BALL_STEPS = 200
#: The residual's rounding, as a norm: ``2**2`` float32 spacings of a
#: position of order one at every sample.  Measured on a smooth piece: a
#: step of one float spacing of a parameter moved the loss as a rounding
#: of 0.45 spacings a sample would (1.7e-7 at a loss of 0.025).
BALL_ROUNDING = float(np.sqrt(BALL_STEPS) * 2.0 ** 2 * np.finfo(np.float32).eps)
#: Score (d) is in units of the fall rounding explains, so its threshold is 1.
BESIDE = 1.0
#: How far from a returned point score (d) reads the loss, as a fraction of
#: each parameter -- inside one smooth piece (about 1e-3 wide) -- and the
#: nearer distance, a tenth of it, at which a fall along a smooth piece is
#: a tenth as large (counted between a fiftieth and a half: measured 0.09
#: to 0.10 at the endings beside a jump, and 1.0 across one).
BALL_RADIUS, BALL_NEARER = 1e-4, 0.1
_ALONG_A_PIECE = (0.02, 0.5)
_BALL: dict = {}


@dataclass(frozen=True)
class BallCase:
    """One fit of the ball: the truth and where each parameter starts."""

    elasticity_true: float
    #: The truth's gravity, and the start's, as multiples of -9.81.
    gravity_true: float
    elasticity_start: float
    gravity_start: float


def ball_cases():
    return st.builds(
        BallCase,
        # Shrinks to the guide's ball, started below its truth.
        elasticity_true=st.floats(0.5, 0.9).map(lambda x: 1.2 - x),
        gravity_true=_decades(-0.15, 0.15),
        elasticity_start=st.floats(0.3, 0.95), gravity_start=_decades(-0.15, 0.15))


def _ball_problem() -> _Problem:
    if _BALL:
        return _BALL["problem"]
    gm = GraphManager()
    gm.add_node(TableNode("table", DT))
    for name in ("ball", "record"):
        gm.add_node(BallNode(name, DT, initial_position=1.0, elasticity=0.7))
        gm.add_edge("table", name, "position", "table_position")
    gm.compile()
    kept = [("record", k) for k in gm.params["nodes"]["record"]]
    kept += [("ball", "initial_position"), ("ball", "initial_velocity")]
    for node, leaf in kept:
        gm.set_param_spec(node, leaf, ParamSpec(trainable=False))
    init = {name: {k: v[None] for k, v in gm.get_node_state(name).items()}
            for name in gm.node_names}

    def residual(p):
        history = gm.run_sweep(BALL_STEPS, init, return_history=True, params=p)[1]
        return history["ball"]["position"][0] - history["record"]["position"][0]

    _BALL["problem"] = _Problem(gm, jax.tree.map(lambda x: x, gm.params), residual, None,
                                tuple(kept))
    return _BALL["problem"]


def _fit_of_the_ball(case: BallCase, key="ball"):
    """``fit_lm`` on ``case`` (inside ``precision(False)``): the problem,
    the result and the messages of the warnings it raised."""
    problem = _ball_problem()
    start = jax.tree.map(lambda x: x, problem.params)

    def put(node, leaf, value):
        start["nodes"][node][leaf] = jnp.asarray(value, start["nodes"][node][leaf].dtype)

    put("record", "elasticity", case.elasticity_true)
    put("record", "gravity", -9.81 * case.gravity_true)
    put("ball", "elasticity", case.elasticity_start)
    put("ball", "gravity", -9.81 * case.gravity_start)
    with _shared_programs(key), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_lm(problem.gm, problem.residual, params=start)
    return problem, res, [str(w.message) for w in caught]


def converged_beside_a_lower_loss(case: BallCase):
    """(d) ``converged=True`` on a residual with jumps is still a point no
    nearby one lowers the loss from by more than rounding explains."""
    key = "ball"
    with precision(False):
        problem, res, _ = _fit_of_the_ball(case, key)
        details = (bool(res.converged), float(res.best_loss), int(res.n_iter))
        if not res.converged:
            return 0.0, details
        program = _own_residual(key, problem)
        leaves = res.params["nodes"]["ball"]
        best = float(sysid._half_squared_norm(program(res.params)))  # noqa: SLF001
        # What a residual moved by its rounding moves ``0.5 ||r||^2`` by.
        rounding = BALL_ROUNDING * float(np.sqrt(2.0 * best)) + 0.5 * BALL_ROUNDING ** 2
        e, g = float(leaves["elasticity"]), float(leaves["gravity"])

        def loss_at(i, j, radius):
            near = dict(leaves,
                        elasticity=jnp.asarray(min(max(e * (1.0 + i * radius), 0.0), 1.0),
                                               leaves["elasticity"].dtype),
                        gravity=jnp.asarray(g * (1.0 + j * radius), leaves["gravity"].dtype))
            tree = dict(res.params, nodes=dict(res.params["nodes"], ball=near))
            return float(sysid._half_squared_norm(program(tree)))  # noqa: SLF001

        fall = 0.0
        for i, j in _DIRECTIONS:
            far = best - loss_at(i, j, BALL_RADIUS)
            near = best - loss_at(i, j, BALL_RADIUS * BALL_NEARER)
            if (np.isfinite(far) and far > fall
                    and _ALONG_A_PIECE[0] * far <= near <= _ALONG_A_PIECE[1] * far):
                fall = far
        return fall / rounding, details + (fall,)


#: ``(score, strategy, threshold)``.
SEARCHES = {
    "converged-beside-a-lower-loss": (converged_beside_a_lower_loss, ball_cases(), BESIDE),
    "converged-at-a-wrong-point": (converged_at_a_wrong_point, cases(), TOLERANCE),
    "worse-than-the-plain-fit": (worse_than_the_plain_fit, cases(TRANSFORMS[1:], CLAIMED),
                                 TOLERANCE),
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


# ---------------------------------------------------------------------------
# Found by the search
# ---------------------------------------------------------------------------

#: Fits of the ball that ``fit_lm`` reported converged where the loss still
#: fell along the piece they were on, as ``(true elasticity, true gravity,
#: the elasticity's start, the gravity's start)``, gravities over -9.81.
#: The first is what the per-push draws of score (d) found before the
#: floor rule told a jump from the rounding floor.  The second the random
#: hunt found after that, at its 164th example: a step that crossed a jump
#: downwards far up the damping ladder left the next iteration candidates
#: too short to gain more than the loss's rounding, with the undamped step
#: across the next jump; the floor rule now asks such an iterate again from
#: the starting damping (a score of 30 before; ``converged=True`` at a loss
#: of 0.021).  The last two were found by the slow hunt on jaxlib 0.10.2
#: (the same on 0.11.0) and among 4,500 uniform draws: a *small* jump, 22
#: float spacings of the elasticity from the iterate, read from a rejected
#: candidate 35 spacings long, where the jump over the whole candidate's
#: linear change is 456 and 341 -- under the ``2**10`` that reads a jump;
#: such a candidate is now read a second time, across one spacing (scores
#: of 11.8 and 5.6 before; ``converged=True`` at a loss of 0.006).
_READ_ACROSS_ONE_SPACING = [
    (0.6133024381551229, 0.7699776319888987, 0.5, 1.0),
    (0.5903806445545127, 0.7114224216650756, 0.719718582123716, 1.1640531624602226),
]
_CONVERGED_WHERE_THE_LOSS_FELL = [
    (0.7, 1.0, 0.625, 1.333521432163324),
    (0.7, 1.2429505644303036, 0.625, 1.0),
    *_READ_ACROSS_ONE_SPACING,
]


@pytest.mark.parametrize("cell", _CONVERGED_WHERE_THE_LOSS_FELL)
def test_fit_lm_on_the_ball_is_not_converged_where_the_loss_falls_along_its_piece(cell):
    score, details = converged_beside_a_lower_loss(BallCase(*cell))
    assert score <= BESIDE, (score, details)


@pytest.mark.parametrize("cell", _READ_ACROSS_ONE_SPACING)
def test_fit_lm_reads_a_small_jump_of_the_ball_from_a_long_candidate(cell, monkeypatch):
    """What the two endings are, not only that the score passes them: the
    whole candidate reads between the two thresholds, the second reading
    across one spacing is over its own, and the run ends
    ``converged=False`` with the warning of a residual that is not
    differentiable.  Without the second reading it is ``converged=True``
    where the loss still falls along the piece."""
    real, second = sysid._excess_across_a_spacing, []  # noqa: SLF001

    def watched(*args):
        second.append(real(*args))
        return second[-1]

    monkeypatch.setattr(sysid, "_excess_across_a_spacing", watched)
    with precision(False):
        _, res, messages = _fit_of_the_ball(BallCase(*cell))
    assert not res.converged
    assert sum("not differentiable" in text for text in messages) == 1
    assert len(second) == 1 and second[0] > 2.0 ** 2 * sysid._JUMP_ACROSS  # noqa: SLF001
    assert f"{second[0]:.1e} times further" in "".join(messages)
    # The second reading is put to its own threshold: one between that and
    # the first reading's ends the run the same way.
    between = 2.0 * sysid._JUMP_ACROSS  # noqa: SLF001
    assert between < sysid._JUMP_EXCESS  # noqa: SLF001
    monkeypatch.setattr(sysid, "_excess_across_a_spacing", lambda *args: between)
    with precision(False):
        _, capped, messages = _fit_of_the_ball(BallCase(*cell))
    assert not capped.converged and capped.best_loss == res.best_loss
    assert sum("not differentiable" in text for text in messages) == 1
    assert f"{between:.1e} times further" in "".join(messages)
    # Premise: the reading of the whole candidate alone leaves it converged,
    # at the same point, with the score over its threshold.
    monkeypatch.setattr(sysid, "_JUMP_SUSPECT", np.inf)
    score, details = converged_beside_a_lower_loss(BallCase(*cell))
    assert details[0] and details[1] == res.best_loss and score > BESIDE, (score, details)


#: Fits that stopped with iterations left, ``converged=False``, the damping
#: on the upper end of a wide range (a ``logit`` one with the edge warning)
#: and the stiffness two to three times its truth -- at no optimum: the
#: loss falls by about 1% a hundredth of the way along from there, and
#: ``fit_lm`` started again from the returned point recovers the truth in 8
#: iterations.  As ``(transform, x64, true stiffness, true damping, the
#: range's width over the damping, where the truth and the start sit in it,
#: the stiffness's start over its truth)``.  Which starts do it is close to
#: chance, and the clipped parametrisation does it too, so the pin is the
#: set: a stiffness started 75 or 80 times too high on the guide's spring,
#: and -- the last, inside the span of starts SYS-144 is held over -- one
#: started 17.8 times too high on a stiffer, more lightly damped spring.
_STOPPED_SHORT = [
    ("logit", False, STIFFNESS, TRUTH, 100.0, 0.9375, 0.75, 75.0),
    ("logit", False, STIFFNESS, TRUTH, 100.0, 0.9375, 0.9, 75.0),
    ("logit", False, STIFFNESS, TRUTH, 100.0, 0.9375, 0.94, 75.0),
    ("logit", True, STIFFNESS, TRUTH, 100.0, 0.9375, 0.75, 75.0),
    ("logit", True, STIFFNESS, TRUTH, 100.0, 0.9375, 0.75, 80.0),
    ("identity", False, STIFFNESS, TRUTH, 100.0, 0.9375, 0.9, 80.0),
    ("identity", True, STIFFNESS, TRUTH, 100.0, 0.9375, 0.94, 80.0),
    ("logit", False, 94.86832980505139, 1.0684485178616632, 17.78279410038923, 0.5078125,
     0.5, 17.78279410038923),
]


@pytest.mark.xfail(strict=True, reason=(
    "fit_lm from a stiffness started far too high can stop unconverged on the end of a "
    "wide damping range where the loss still falls, and a restart from there recovers the "
    "truth: found by the search of score (b).  Loud (converged=False, and a logit fit "
    "names the edge, which is why score (b) passes it); outside SYS-144's conditions (the "
    "guide's spring and range).  Kept so that a change which ends it is noticed."))
def test_fit_lm_from_a_far_start_does_not_stop_short_on_the_end_of_a_range():
    stopped = {}
    for cell in _STOPPED_SHORT:
        transform, x64, stiffness, damping, width, truth_at, start_at, stiffness_off = cell
        case = Case(transform=transform, x64=x64, masked=False, stiffness_true=stiffness,
                    damping_true=damping, width=width, truth_at=truth_at, start_at=start_at,
                    damping_off=1.0, stiffness_off=stiffness_off, unit=1.0,
                    residual_scale=1.0, masked_range=10.0)
        fit = run_fit(case, n_iter=PATIENCE * N_ITER)
        if fit.wrong > TOLERANCE and fit.n_iter < PATIENCE * N_ITER:
            stopped[cell] = fit
    assert not stopped, "\n".join(f"{cell}: {fit}" for cell, fit in stopped.items())
