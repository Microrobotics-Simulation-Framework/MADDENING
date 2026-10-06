"""The transform grid: one fit under a bounded transform beside its control.

Shared by ``test_sysid_truth_recovery.py`` (the grid itself) and
``test_sysid_one_evaluation.py`` (which borrows the spring).

The problem is the parameter guide's spring -- stiffness 30, a damping,
mass 1, 100 noiseless position samples, mass frozen -- with the damping
declared under the transform a cell names, and the same fit run again with
the damping under the identity transform and the same bounds (clipped): the
**control**.  The oracle is differential: wherever the control recovers
the truth, the transformed fit must, or it must say by name that it ended
on the edge of its transform's range.

The truth is one fixed damping for every cell and the *bounds* are placed
around it, so "the truth at 5% of its range" is the range ``(truth - 0.05
w, truth + 0.95 w)``.  That reaches every position the transform has -- the
transform sees only where in its bounds a value sits -- and leaves the
data, and so the model-side programs the fitters compile, the same for
every cell (:func:`shared_programs`).
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

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_lm, fit_multiple_shooting, observations_from_history

#: The guide's spring.
STIFFNESS, MASS, REST_LENGTH, N_STEPS, DT = 30.0, 1.0, 1.0, 100, 0.01
#: The damping every cell's data were generated with, and the width of the
#: range placed around it (the guide's ``(0.5, 2.0)``).
TRUTH, WIDTH = 1.9, 1.5

TRANSFORMS = ("log", "log-from-lo", "logit")
#: Where in its range a value sits: three interior fractions, and a distance
#: of a few float spacings from each edge (``EDGE_ULPS`` of the bounds' own
#: size, twice the margin ``ParamSpec.check`` documents for ``logit``).
POSITIONS = ("5%", "50%", "95%", "lower-edge", "upper-edge")
EDGE_ULPS = 8.0
#: The other parameter's start, as a multiple of its truth.
OTHER_STARTS = (0.1, 0.5, 1.5, 10.0, 30.0)
FITTERS = ("fit_lm", "fit", "fit_multiple_shooting")

#: The message of the warning a fit that ends on its transform's edge gives,
#: and of the one an Adam fit started on it gives.
EDGE_WARNING = "ended on the edge of their transform's usable range"
EDGE_START_WARNING = "start on the edge of their transform's usable range"

_FRACTION = {"5%": 0.05, "50%": 0.5, "95%": 0.95}


@contextlib.contextmanager
def precision(x64: bool):
    """Run the body in float64 (``x64``) or float32, restoring the setting."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", bool(x64))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def _eps() -> float:
    return float(np.finfo(np.float64 if jax.config.read("jax_enable_x64") else np.float32).eps)


#: What stands in for "a few float spacings from an edge" where a plain
#: ``log`` leaf has no edge at float scale (its one bound is 0, and its
#: range has no top): the truth at a thousandth of the nominal top, or at
#: all but a thousandth of it.
_LOG_EDGE_FRACTION = {"lower-edge": 1e-3, "upper-edge": 1.0 - 1e-3}


def bounds_around(transform: str, truth_at: str) -> tuple[float, float]:
    """The range ``(lo, hi)`` that puts :data:`TRUTH` at ``truth_at``.

    ``logit`` and ``log-from-lo`` move a range of :data:`WIDTH` around the
    truth; at an edge the truth is :data:`EDGE_ULPS` float spacings of the
    bounds' own size inside it.  A plain ``log`` leaf has the one bound 0,
    so its range is ``(0, hi)`` with the truth at that fraction of a nominal
    top ``hi`` that only places the starts (:data:`_LOG_EDGE_FRACTION` at
    "an edge")."""
    if transform == "log":
        fraction = _FRACTION.get(truth_at) or _LOG_EDGE_FRACTION[truth_at]
        return 0.0, TRUTH / fraction
    if truth_at in _FRACTION:
        lo = TRUTH - _FRACTION[truth_at] * WIDTH
        return lo, lo + WIDTH
    if truth_at == "lower-edge":
        gap = EDGE_ULPS * _eps() * (TRUTH + WIDTH)
        return TRUTH - gap, TRUTH - gap + WIDTH
    gap = EDGE_ULPS * _eps() * max(TRUTH, WIDTH)
    return TRUTH + gap - WIDTH, TRUTH + gap


def value_at(transform: str, bounds: tuple[float, float], position: str) -> float:
    """The value at ``position`` of ``bounds``: a fraction of the range, or
    :data:`EDGE_ULPS` float spacings of the bounds' size inside an edge
    (:data:`_LOG_EDGE_FRACTION` of a plain ``log`` leaf's nominal top, which
    has no edge at float scale)."""
    lo, hi = bounds
    if position in _FRACTION:
        return lo + _FRACTION[position] * (hi - lo)
    if transform == "log":
        return _LOG_EDGE_FRACTION[position] * hi
    gap = EDGE_ULPS * _eps() * max(abs(lo), abs(hi), hi - lo)
    return lo + gap if position == "lower-edge" else hi - gap


def spec_for(transform, bounds) -> ParamSpec:
    """The damping's spec: ``transform`` over ``bounds``, or (``None``) the
    control -- the identity transform with the bounds that transform has."""
    lo, hi = bounds
    if transform == "logit":
        return ParamSpec(bounds=(lo, hi), transform="logit")
    if transform in ("log", "log-from-lo"):
        return ParamSpec(bounds=(lo, None), transform="log")
    raise ValueError(transform)


def control_for(transform, bounds) -> ParamSpec:
    lo, hi = bounds
    return ParamSpec(bounds=(lo, hi if transform == "logit" else None))


#: The spec of the leaf a cell with ``masked=True`` leaves out by ``mask=``:
#: trainable, so only the mask keeps the fit off it, and under a ``logit`` so
#: wide that its ``constrain(unconstrain(p))`` round trip is nowhere near
#: ``p`` in float32 (2.0 comes back 2.0266 from a jitted round trip).  A fit
#: that runs the leaf through its transform fits the wrong model.
MASKED_OUT_SPEC = ParamSpec(bounds=(-1e6, 1e6), transform="logit")


@dataclass
class Problem:
    """One precision's spring, its record, and the three objectives."""

    gm: GraphManager
    truth: dict
    residual: object
    loss: object
    observations: dict
    masked: bool
    #: The damping the record was generated with.
    damping: float = TRUTH

    def set_damping_spec(self, spec: ParamSpec) -> None:
        self.gm.set_param_spec("spring", "damping", spec)

    def start(self, stiffness: float, damping: float) -> dict:
        out = jax.tree.map(lambda x: x, self.truth)
        leaves = out["nodes"]["spring"]
        leaves["stiffness"] = jnp.asarray(stiffness, leaves["stiffness"].dtype)
        leaves["damping"] = jnp.asarray(damping, leaves["damping"].dtype)
        return out

    def mask(self):
        if not self.masked:
            return None
        return self.only("stiffness", "damping")

    def only(self, *keys):
        """A mask that fits ``keys`` and nothing else."""
        mask = jax.tree.map(lambda _: False, self.gm.params)
        for key in keys:
            mask["nodes"]["spring"][key] = True
        return mask


def build_problem(masked: bool, damping: float = TRUTH, float32_leaves: bool = False) -> Problem:
    """The spring at the current precision.  ``masked``: ``rest_length`` is
    left trainable under :data:`MASKED_OUT_SPEC` and kept out of the fit by
    ``mask=``; otherwise it is frozen by its spec and the fit takes the
    graph's own trainable set.  ``damping`` is the truth the record is
    generated with (the grid's is :data:`TRUTH`).  ``float32_leaves``, under
    x64: the stiffness and the damping are float32 leaves of a float64 graph
    (``ravel_pytree`` promotes the optimiser's vector around them)."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, initial_position=0.5, stiffness=STIFFNESS,
                                 damping=damping, mass=MASS, rest_length=REST_LENGTH))
    gm.compile()
    gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))
    gm.set_param_spec("spring", "rest_length",
                      MASKED_OUT_SPEC if masked else ParamSpec(trainable=False))
    state0 = gm._user_state(gm._state)  # noqa: SLF001
    init = {"spring": {k: v[None] for k, v in gm.get_node_state("spring").items()}}

    def positions(p):
        return gm.run_sweep(N_STEPS, init, return_history=True, params=p)[1]["spring"][
            "position"][0]

    truth = jax.tree.map(lambda x: x, gm.params)
    if float32_leaves:
        for key in ("stiffness", "damping"):
            truth["nodes"]["spring"][key] = jnp.asarray(truth["nodes"]["spring"][key],
                                                        jnp.float32)
    measured = positions(truth)
    history = gm.run_scan_with_history(N_STEPS, params=truth)[1]
    gm.reset_state()

    def residual(p):
        return positions(p) - measured

    def loss(p):
        return 0.5 * jnp.sum((positions(p) - measured) ** 2)

    return Problem(gm=gm, truth=truth, residual=residual, loss=loss,
                   observations=observations_from_history(state0, history), masked=masked,
                   damping=float(damping))


#: The sharing in force (:func:`shared_programs`), as ``[cache, key, next
#: slot]``; ``None`` outside one.
_SHARING: list | None = None


@contextlib.contextmanager
def shared_programs(cache: dict, key):
    """Share the fitters' compiled model-side programs between fits that
    hand them the same model.

    A fitter compiles its residual (or loss) through
    ``sysid._compile_model``, as functions of the physical ``params`` tree
    alone: nothing of the transform, the bounds or the start is inside them.
    Every cell of the grid at one precision has the same data and the same
    fitted leaves, so those programs are the same program, and compiling
    them once per ``key`` instead of once per fit is what makes six thousand
    fits affordable.  It is also a check of that statement: a fitter that
    put anything cell-specific into a model-side program would run the
    wrong one here.  The programs a fit asks for are matched to the cached
    ones by the order it asks in, which :func:`run_fit` restarts before
    every fit.
    """
    global _SHARING
    real = sysid._compile_model  # noqa: SLF001
    outer = _SHARING
    _SHARING = [cache, key, 0]

    def compile_once(fn):
        slot = (key, _SHARING[2])
        _SHARING[2] += 1
        if slot not in cache:
            cache[slot] = real(fn)
        return cache[slot]

    sysid._compile_model = compile_once  # noqa: SLF001
    try:
        yield
    finally:
        sysid._compile_model = real  # noqa: SLF001
        _SHARING = outer


def _begin_fit() -> None:
    """Before each call of a fitter: its programs are matched from the
    first again."""
    if _SHARING is not None:
        _SHARING[2] = 0


#: Adam's schedule in the grid, as ``(lr, n_iter)`` stages, each started from
#: the last one's result.  Adam's step is about ``lr`` however small the
#: gradient, so one rate cannot both cross a range (a ``logit`` coordinate
#: started on its edge is 8 units of ``u`` out in float32, 18 in float64)
#: and settle to a thousandth of it; a caller anneals, and so does the grid,
#: for the transformed fit and its control alike.
ADAM_STAGES = ((0.3, 120), (0.05, 100), (0.008, 100))


def _position(history):
    return history["spring"]["position"]


def run_fit(problem: Problem, fitter: str, start: dict):
    """``(result, edge_warned, start_warned)`` for one fit of ``problem``
    from ``start`` (the last stage's result for the Adam fitters,
    :data:`ADAM_STAGES`); ``(None, False, False)`` for a fit that refused
    its start loudly (a damping so far out that the explicit step
    overflows: ``FloatingPointError``).  ``edge_warned``: the *last* fit
    said it ended on an edge.  ``start_warned``: the first Adam fit said it
    was started on one."""
    gm = problem.gm
    res, edge, at_start = None, False, False
    try:
        if fitter == "fit_lm":
            _begin_fit()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = fit_lm(gm, problem.residual, params=start, mask=problem.mask())
            edge = any(EDGE_WARNING in str(w.message) for w in caught)
            return res, edge, False
        params, states = start, None
        for stage, (lr, n_iter) in enumerate(ADAM_STAGES):
            _begin_fit()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if fitter == "fit":
                    res = fit(gm, problem.loss, params=params, mask=problem.mask(),
                              lr=lr, n_iter=n_iter)
                else:
                    res, states = fit_multiple_shooting(
                        gm, problem.observations, obs_fn=_position, window=20,
                        params=params, mask=problem.mask(), window_states=states,
                        lr=lr, lr_states=1e-6, n_iter=n_iter)
            edge = any(EDGE_WARNING in str(w.message) for w in caught)
            if stage == 0:
                at_start = any(EDGE_START_WARNING in str(w.message) for w in caught)
            params = res.params
    except FloatingPointError:
        return None, False, False
    return res, edge, at_start


#: How near the truth counts as recovered, as a fraction of the damping's
#: range and of the stiffness itself.  ``fit_lm`` converges to the float
#: floor; Adam ends within a step of its last rate (:data:`ADAM_STAGES`),
#: and a ``logit`` coordinate whose optimum is on its bound approaches it
#: only as fast as that rate walks ``u``.
TOLERANCE = {"fit_lm": 1e-3, "fit": 1e-2, "fit_multiple_shooting": 1e-2}


def recovered(res, truth: float, scale: float, tolerance: float) -> bool:
    """Whether a fit is at the truth: the damping within ``tolerance`` of
    ``scale`` (its range; the truth itself for a plain ``log`` leaf, whose
    range is open) and the stiffness within ``tolerance`` of itself."""
    if res is None:
        return False
    leaves = res.params["nodes"]["spring"]
    return (abs(float(leaves["damping"]) - truth) <= tolerance * scale
            and abs(float(leaves["stiffness"]) - STIFFNESS) <= tolerance * STIFFNESS)


@dataclass
class Cell:
    transform: str
    truth_at: str
    start_at: str
    other: float
    control_ok: bool
    fit_ok: bool
    edge_warned: bool
    converged: bool
    damping: float
    stiffness: float
    #: Whether the damping came back on (or beyond) an end of the range its
    #: transform resolves, read from its value.
    on_edge: bool = False
    #: Whether an Adam fit said, before it ran, that it was started on one.
    start_warned: bool = False

    @property
    def violates(self) -> bool:
        """The oracle: the control recovered the truth and the transformed
        fit neither recovered it nor said, naming the parameter, that it
        ended on an edge of its transform's range or was started on one."""
        return (self.control_ok and not self.fit_ok
                and not self.edge_warned and not self.start_warned)

    @property
    def silent_on_an_edge(self) -> bool:
        """What no cell may be, whatever its control did: on an edge of its
        transform's range with no warning."""
        return self.on_edge and not self.edge_warned

    def __str__(self) -> str:
        return (f"{self.transform} truth@{self.truth_at} start@{self.start_at} "
                f"other x{self.other:g}: control {'ok' if self.control_ok else 'off'}, fit "
                f"{'ok' if self.fit_ok else 'off'} (damping {self.damping:.9g}, stiffness "
                f"{self.stiffness:.9g}, converged={self.converged}, on an edge="
                f"{self.on_edge}, edge warning={self.edge_warned}, start warning="
                f"{self.start_warned})")


def usable_range(spec: ParamSpec, dtype) -> tuple[float, float]:
    """The values the ends of ``spec``'s optimiser interval map to at
    ``dtype``: the range a fit under that transform can return."""
    ends = []
    for u, open_end in zip(spec._optimiser_interval(dtype), (-np.inf, np.inf)):  # noqa: SLF001
        ends.append(float(spec.to_constrained(jnp.asarray(u, dtype)))
                    if np.isfinite(u) else open_end)
    return ends[0], ends[1]


def run_cell(problem: Problem, fitter: str, transform: str, truth_at: str, start_at: str,
             other: float, *, bounds=None, start_damping=None) -> Cell:
    """One cell: the transformed fit and its control, from the same start.

    ``bounds`` and ``start_damping`` override the placement, for a cell
    given by its numbers (the reproducer's)."""
    bounds = bounds_around(transform, truth_at) if bounds is None else bounds
    start_c = (value_at(transform, bounds, start_at) if start_damping is None
               else start_damping)
    start = problem.start(other * STIFFNESS, start_c)
    scale = problem.damping if transform == "log" else bounds[1] - bounds[0]
    problem.set_damping_spec(control_for(transform, bounds))
    control, _, _ = run_fit(problem, fitter, start)
    problem.set_damping_spec(spec_for(transform, bounds))
    res, edge, at_start = run_fit(problem, fitter, start)
    nan = float("nan")
    leaves = {} if res is None else res.params["nodes"]["spring"]
    tolerance = TOLERANCE[fitter]
    damping = float(leaves.get("damping", nan))
    on_edge = False
    if res is not None:
        dtype = np.asarray(leaves["damping"]).dtype
        low, high = usable_range(spec_for(transform, bounds), dtype)
        # Within a few float spacings: the fitter's compiled map and this
        # eager one may round an end differently.
        slack = 8.0 * float(np.finfo(dtype).eps) * max(abs(low), abs(damping), 1e-30)
        on_edge = damping <= low + slack or damping >= high - slack
    return Cell(transform, truth_at, start_at, other,
                recovered(control, problem.damping, scale, tolerance),
                recovered(res, problem.damping, scale, tolerance), edge,
                bool(res is not None and res.converged),
                damping, float(leaves.get("stiffness", nan)), on_edge, at_start)
