"""One evaluation: a fit's objective is evaluated at the arrays it returns.

Three audit rounds running found two evaluations of ``theta -> params``
disagreeing in their last bits: ``best_loss`` one ulp off the loss of the
parameters returned, then the same for a leaf whose ``constrain(
unconstrain(p))`` round trip is inexact, then -- under a ``logit`` whose
bounds are wide beside the value -- a fit that ran a leaf it was told to
leave alone at 2.0266 for a given 2.0 and returned a stiffness of 30.06 for
a truth of 30, ``converged=True``
(audit_040_p4_12/fmu-sysid/repro_F2_transform_grid_and_best_loss.py).  The
fitters now make one evaluation: the model-side programs they compile
(``sysid._compile_model``) take the physical ``params`` tree as their
argument, and that tree -- ``_PhysicalMap.params`` -- is what the result
holds.

So the oracle here does not compare two computations of anything.  It
stands between a fitter and its model, records the arrays every model
evaluation was handed, and requires, over drawn transforms, bounds (up to
1e6 times wider than the value), dtypes, masks and guard settings:

* **the result was evaluated**: one recorded evaluation received exactly
  the arrays of ``FitResult.params``, bit for bit, and -- wherever the
  identifiability guard held nothing -- its loss is ``best_loss``, bit for
  bit;
* **what the fit does not move, the model never sees moved**: in every
  evaluation, a leaf left out by ``mask=`` or frozen by its spec is the
  object that went in, and in the result every such leaf is that object;
* **a fit that takes no step evaluates and returns its start**, bit for
  bit, fitted leaves included;
* **``callback`` receives the arrays that were evaluated**, with their loss.

Per push: the fixed cases (each transform, a ``logit`` 1e6 times wider than
its value fitted, masked out and frozen, in float32, float64 and mixed
precision, under all three fitters) and the audit's three symptoms as a
user sees them.  Slow: the property, over drawn cases.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fit, fit_lm, fit_multiple_shooting
from maddening.warnings import PrecisionLimitWarning

from tests.conftest import EXAMPLES_COSTLY
from tests.property import sysid_transform_grid as grid

KEYS = ("stiffness", "damping", "rest_length")
#: The values the three leaves start at, and the data's.
START = {"stiffness": 45.0, "damping": 3.0, "rest_length": 1.5}
TRUTH = {"stiffness": 30.0, "damping": 2.0, "rest_length": 1.2}

_RNG = np.random.default_rng(7)
_A = _RNG.normal(size=(12, 3))

#: Under x64: every leaf float64, or the three leaves float32 in a float64
#: graph (``ravel_pytree`` promotes the optimiser's vector around them).
DTYPES = {"float32": (False, np.float32), "float64": (True, np.float64),
          "mixed": (True, np.float32)}


def grid_note(message: str) -> None:
    """``hypothesis.note`` inside a property, nothing in a plain test."""
    from hypothesis import note
    from hypothesis.errors import InvalidArgument
    try:
        note(message)
    except InvalidArgument:
        pass


def _bits(leaf) -> tuple:
    a = np.asarray(leaf)
    return (a.dtype.str, a.shape, a.tobytes())


def _same_bits(tree_a, tree_b) -> bool:
    la, lb = jax.tree.leaves(tree_a), jax.tree.leaves(tree_b)
    return len(la) == len(lb) and all(_bits(x) == _bits(y) for x, y in zip(la, lb))


def _is_concrete(tree) -> bool:
    return not any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(tree))


class Recorder:
    """Stands in for ``sysid._compile_model``: compiles as it does, and
    records ``(params, outputs)`` of every concrete call of every program a
    fitter runs its model through.  (A call made while another program is
    being traced -- the guard's Hessian-vector products reuse the compiled
    gradient -- has tracers for arguments and is not an evaluation.)"""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, object]] = []
        self._real = sysid._compile_model  # noqa: SLF001

    def __call__(self, fn):
        compiled = self._real(fn)

        def run(params, *args):
            out = compiled(params, *args)
            if _is_concrete((params, args)):
                self.calls.append((params, out))
            return out

        return run

    def losses_at(self, params, fitter: str) -> list[float]:
        """The loss of every recorded evaluation that received exactly
        ``params``' arrays, as that fitter reads a loss from its program."""
        out = []
        for seen, result in self.calls:
            if not _same_bits(seen, params):
                continue
            if fitter == "fit_lm":
                if getattr(result, "ndim", None) == 1:            # the residual, not J
                    out.append(sysid._half_squared_norm(result))  # noqa: SLF001
            else:
                out.append(float(result[0]))
        return out


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(sysid, "_compile_model", rec)
    return rec


def _spec(kind: str, value: float, ratio: float, where: float):
    """A leaf's spec: ``kind`` with bounds ``ratio`` times as wide as the
    value, the value sitting at fraction ``where`` of them."""
    width = ratio * abs(value)
    lo = value - where * width
    if kind == "free":
        return ParamSpec()
    if kind == "clip":
        return ParamSpec(bounds=(lo, lo + width))
    if kind == "log":
        return ParamSpec(bounds=(0.0, None), transform="log")
    if kind == "log-from-lo":
        return ParamSpec(bounds=(lo, None), transform="log")
    return ParamSpec(bounds=(lo, lo + width), transform="logit")


def _graph(leaves, dtype):
    """The spring as a container for three leaves, each ``(role, kind,
    ratio, where)``: a ``frozen`` leaf has ``trainable=False`` over its
    spec, a ``masked`` one is trainable and left out by ``mask=``."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, mass=1.0, **{k: START[k] for k in KEYS}))
    gm.compile()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    start = jax.tree.map(lambda x: x, gm.params)
    mask = jax.tree.map(lambda _: False, gm.params)
    for key, (role, kind, ratio, where) in zip(KEYS, leaves):
        spec = _spec(kind, START[key], ratio, where)
        if role == "frozen":
            spec = ParamSpec(trainable=False, bounds=spec.bounds, transform=spec.transform)
        gm.set_param_spec("s", key, spec)
        mask["nodes"]["s"][key] = role == "fitted"
        start["nodes"]["s"][key] = jnp.asarray(START[key], dtype)
    return gm, start, mask


def _residual(p):
    q = p["nodes"]["s"]
    x = jnp.stack([q[k] for k in KEYS])
    b = jnp.asarray(_A @ np.array([TRUTH[k] for k in KEYS]), x.dtype)
    return jnp.asarray(_A, x.dtype) @ x - b


def _loss(p):
    return 0.5 * jnp.sum(_residual(p) ** 2)


def _run(fitter, gm, start, mask, recorder, *, hold=True, n_iter=6, tol=0.0):
    seen = []

    def callback(i, loss, params):
        seen.append((loss, params))

    with warnings.catch_warnings():
        # A value its transform cannot resolve, a coordinate left on an
        # edge and a declined hold all say so; none is this oracle's.
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", PrecisionLimitWarning)
        if fitter == "fit_lm":
            res = fit_lm(gm, _residual, params=start, mask=mask, n_iter=n_iter, tol=tol,
                         hold_undetermined=hold, callback=callback)
        else:
            res = fit(gm, _loss, params=start, mask=mask, n_iter=n_iter, lr=0.05, tol=tol,
                      hold_undetermined=hold, callback=callback)
    return res, seen


def _held_nothing(res, n_fitted: int) -> bool:
    """Whether ``params`` is the selected iterate as the optimiser left it."""
    return res.excited_rank in (None, n_fitted) or bool(res.hold_declined)


def _check(fitter, leaves, dtype_name, hold, recorder, *, n_iter=6):
    x64, dtype = DTYPES[dtype_name]
    with grid.precision(x64):
        gm, start, mask = _graph(leaves, dtype)
        res, seen = _run(fitter, gm, start, mask, recorder, hold=hold, n_iter=n_iter)
        _assert_one_evaluation(fitter, leaves, start, res, seen, recorder)
        return res


def _assert_one_evaluation(fitter, leaves, start, res, seen, recorder):
    roles = {key: role for key, (role, *_rest) in zip(KEYS, leaves)}
    given_leaves = start["nodes"]["s"]
    grid_note(f"{fitter} {leaves} n_iter={res.n_iter} best_loss={res.best_loss} "
              f"rank={res.excited_rank} declined={res.hold_declined}")
    assert recorder.calls, "the fitter ran no model-side program through the seam"
    # What the fit does not move, the model never sees moved -- and it is the
    # object that went in, never a copy that passed through a transform.
    for params, _ in recorder.calls:
        for key, value in params["nodes"]["s"].items():
            if roles.get(key) != "fitted":
                assert value is given_leaves[key], (key, value, given_leaves[key])
    for key, value in res.params["nodes"]["s"].items():
        if roles.get(key) != "fitted":
            assert value is given_leaves[key], (key, value, given_leaves[key])
    # The result was evaluated: an evaluation received exactly its arrays.
    at_result = recorder.losses_at(res.params, fitter)
    assert at_result, "no model evaluation received the arrays FitResult.params holds"
    if _held_nothing(res, sum(role == "fitted" for role in roles.values())):
        assert res.best_loss in at_result, (res.best_loss, at_result)
    # ``callback`` was handed the arrays that were evaluated, with their loss.
    for loss, params in seen:
        assert loss in recorder.losses_at(params, fitter), loss
    assert res.best_loss in [loss for loss, _ in seen] or res.best_iteration == len(res.losses)


# ---------------------------------------------------------------------------
# Fixed cases, per push
# ---------------------------------------------------------------------------

_FITTED_WIDE = ("fitted", "logit", 1e6, 0.5)
_CASES = {
    # Each transform fitted, at bounds a few times the value.
    "every-transform": (("fitted", "log", 1.0, 0.5), ("fitted", "logit", 4.0, 0.3),
                        ("fitted", "clip", 4.0, 0.5)),
    "log-from-lo": (("fitted", "log-from-lo", 1e3, 1.0), ("fitted", "free", 1.0, 0.5),
                    ("frozen", "log", 1.0, 0.5)),
    # The audit's: a ``logit`` 1e6 times wider than the value, fitted ...
    "wide-logit-fitted": (("fitted", "log", 1.0, 0.5), _FITTED_WIDE,
                          ("frozen", "free", 1.0, 0.5)),
    # ... and left out by ``mask=``, where the objective ran 2.0266 for 2.0.
    "wide-logit-masked": (("fitted", "log", 1.0, 0.5), ("masked", "logit", 1e6, 0.5),
                          ("masked", "log-from-lo", 1e6, 1.0)),
    "wide-logit-frozen": (("fitted", "logit", 1e4, 0.9), ("frozen", "logit", 1e6, 0.5),
                          ("masked", "log", 1.0, 0.5)),
}


@pytest.mark.parametrize("dtype_name", sorted(DTYPES))
@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("fitter", ["fit_lm", "fit"])
def test_the_arrays_a_fit_returns_are_the_arrays_it_evaluated(fitter, case, dtype_name,
                                                              recorder):
    res = _check(fitter, _CASES[case], dtype_name, True, recorder)
    assert res.best_loss is not None


@pytest.mark.parametrize("fitter", ["fit_lm", "fit"])
def test_without_the_guard_best_loss_is_the_loss_of_exactly_the_result(fitter, recorder):
    """``hold_undetermined=False``: nothing can move ``params`` after the
    run selected it, so the pair is exact in every case."""
    for case in sorted(_CASES):
        recorder.calls.clear()
        res = _check(fitter, _CASES[case], "float32", False, recorder)
        assert res.best_loss in recorder.losses_at(res.params, fitter), case


@pytest.mark.parametrize("fitter", ["fit_lm", "fit"])
@pytest.mark.parametrize("stop", ["n_iter=0", "tol"])
def test_a_fit_that_takes_no_step_evaluates_and_returns_its_start(fitter, stop, recorder):
    """Every leaf comes back as the object that went in -- the fitted ones
    too, whose ``constrain(unconstrain(p))`` is not ``p`` under these
    bounds -- and a run stopped by ``tol`` before its first update evaluated
    exactly those values."""
    gm, start, mask = _graph(_CASES["wide-logit-fitted"], np.float32)
    kw = dict(n_iter=0) if stop == "n_iter=0" else dict(n_iter=5, tol=1e30)
    res, seen = _run(fitter, gm, start, mask, recorder, **kw)
    for key, value in res.params["nodes"]["s"].items():
        assert value is start["nodes"]["s"][key], key
    if stop == "tol":
        assert res.converged and res.n_iter == 1 and res.best_iteration == 0
        assert recorder.losses_at(start, fitter) == [res.best_loss]
        assert _same_bits(seen[0][1], start)
    else:
        assert res.best_loss is None and not seen


def test_a_multiple_shooting_fit_returns_the_arrays_it_evaluated(recorder):
    """The same for the third fitter, on the spring graph its windows need:
    ``rest_length`` left out by ``mask=`` under a ``logit`` 1e6 wide."""
    problem = grid.build_problem(masked=True)
    problem.set_damping_spec(ParamSpec(bounds=(-1e4, 1e4), transform="logit"))
    start = problem.start(45.0, 3.0)
    seen = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", PrecisionLimitWarning)
        res, _ = fit_multiple_shooting(
            problem.gm, problem.observations, obs_fn=lambda h: h["spring"]["position"],
            window=20, params=start, mask=problem.mask(), n_iter=4, lr=0.05,
            lr_states=1e-6, hold_undetermined=False,
            callback=lambda i, loss, p: seen.append((loss, p)))
    given_leaves = start["nodes"]["spring"]
    for params, _ in recorder.calls:
        for key in ("rest_length", "mass", "initial_position", "initial_velocity"):
            assert params["nodes"]["spring"][key] is given_leaves[key], key
    assert res.best_loss in recorder.losses_at(res.params, "fit_multiple_shooting")
    for loss, params in seen:
        assert loss in recorder.losses_at(params, "fit_multiple_shooting")


# ---------------------------------------------------------------------------
# The audit's three symptoms, as a user sees them
# ---------------------------------------------------------------------------


def _guide_fit(spec_c, *, mask_out_damping=False, start_c=3.0):
    """``fit_lm`` on the guide's spring (truth 30 and 2) from 45 and
    ``start_c``, the damping under ``spec_c``."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01, initial_position=0.5, stiffness=30.0,
                                 damping=2.0, mass=1.0))
    gm.compile()
    gm.set_param_spec("spring", "mass", ParamSpec(trainable=False))
    gm.set_param_spec("spring", "rest_length", ParamSpec(trainable=False))
    gm.set_param_spec("spring", "damping", spec_c)
    init = {"spring": {k: v[None] for k, v in gm.get_node_state("spring").items()}}

    def positions(p):
        return gm.run_sweep(100, init, return_history=True, params=p)[1]["spring"][
            "position"][0]

    truth = jax.tree.map(lambda x: x, gm.params)
    measured = positions(truth)
    residual = jax.jit(lambda p: positions(p) - measured)
    start = jax.tree.map(lambda x: x, truth)
    start["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    start["nodes"]["spring"]["damping"] = jnp.asarray(start_c, jnp.float32)
    mask = None
    if mask_out_damping:
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["spring"]["stiffness"] = True
    return fit_lm(gm, residual, params=start, mask=mask), residual


def test_a_leaf_left_out_by_the_mask_is_run_at_the_value_given():
    """The silent wrong result: the damping, left out by ``mask=`` at its
    true 2.0 under ``logit`` bounds ``(-1e6, 1e6)``, was run at 2.0266
    inside the objective while ``params`` returned 2.0, and the stiffness
    fitted beside it came back 30.06 for a truth of 30, ``converged=True``."""
    res, _ = _guide_fit(ParamSpec(bounds=(-1e6, 1e6), transform="logit"),
                        mask_out_damping=True, start_c=2.0)
    assert res.converged
    assert float(res.params["nodes"]["spring"]["damping"]) == 2.0
    assert float(res.params["nodes"]["spring"]["stiffness"]) == pytest.approx(30.0, rel=1e-5)


def test_best_loss_is_the_loss_of_the_parameters_returned_under_a_wide_logit():
    """``best_loss`` read 1.27e-8 for parameters whose loss, by the caller's
    own residual, is 4.5e-10 (``logit`` bounds ``(-1e4, 1e4)``): the
    objective and ``params`` sat on different points of the transform's
    grid, a factor of 28 apart.  A comparison to a factor of two, not to the
    bit: this one is between the fitter's program and the caller's (the
    recorder above holds the fitter to the bit)."""
    with pytest.warns(PrecisionLimitWarning, match="cannot be resolved"):
        res, residual = _guide_fit(ParamSpec(bounds=(-1e4, 1e4), transform="logit"))
    at_params = sysid._half_squared_norm(jnp.ravel(residual(res.params)))  # noqa: SLF001
    assert 0.5 * at_params <= res.best_loss <= 2.0 * at_params, (res.best_loss, at_params)


@pytest.mark.parametrize("bounds, coarse", [((0.5, 3.5), False), ((-1e3, 1e3), False),
                                            ((-1e4, 1e4), True), ((-1e6, 1e6), True)])
def test_a_value_its_transform_cannot_resolve_is_named_at_the_start_of_the_fit(bounds,
                                                                              coarse):
    """A float32 damping of 3 under ``logit`` bounds 1e4 wide can be placed
    to 2.4e-3 at best -- 8e-4 of itself, past ``sqrt(eps)`` -- and the fit
    says so before it runs, naming the leaf, the spacing and the remedy;
    at ordinary bounds it says nothing."""
    spec = ParamSpec(bounds=bounds, transform="logit")
    spacing = spec._resolution(np.float32)  # noqa: SLF001
    size = max(abs(bounds[0]), abs(bounds[1]), bounds[1] - bounds[0])
    assert spacing == pytest.approx(float(np.finfo(np.float32).eps) * size)
    assert (spacing > np.sqrt(np.finfo(np.float32).eps) * 3.0) == coarse
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _guide_fit(spec)
    named = [str(w.message) for w in caught if issubclass(w.category, PrecisionLimitWarning)]
    if not coarse:
        assert not named, named
        return
    assert named and "at the start of the fit" in named[0], named
    for part in ("node 'spring', parameter 'damping' = 3", "'logit'", f"{spacing:.3g} apart",
                 "Tighten the bounds", "transform=None"):
        assert part in named[0], (part, named[0])


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------

_LEAF = st.tuples(
    st.sampled_from(["fitted", "fitted", "masked", "frozen"]),
    st.sampled_from(["free", "clip", "log", "log-from-lo", "logit"]),
    st.sampled_from([1.0, 4.0, 1e2, 1e4, 1e6]),         # bound width over value
    st.sampled_from([0.05, 0.3, 0.5, 0.9]),             # where in them the value sits
)

_PROBLEM = dict(
    leaves=st.tuples(_LEAF, _LEAF, _LEAF).filter(
        lambda ls: any(role == "fitted" for role, *_ in ls)),
    dtype_name=st.sampled_from(sorted(DTYPES)),
    hold=st.booleans(),
    fitter=st.sampled_from(["fit_lm", "fit"]),
)


# Per push: tests/property/test_sysid_one_evaluation.py::test_the_arrays_a_fit_returns_are_the_arrays_it_evaluated
@pytest.mark.slow  # a fit traced and compiled per example: over 5 s on CI
@given(**_PROBLEM)
@example(leaves=_CASES["wide-logit-masked"], dtype_name="float32", hold=True, fitter="fit_lm")
@example(leaves=_CASES["wide-logit-fitted"], dtype_name="mixed", hold=False, fitter="fit")
@settings(max_examples=4 * EXAMPLES_COSTLY, deadline=None, derandomize=True)
def test_every_fitter_evaluates_what_it_returns(leaves, dtype_name, hold, fitter):
    recorder = Recorder()
    real = sysid._compile_model  # noqa: SLF001
    sysid._compile_model = recorder  # noqa: SLF001
    try:
        _check(fitter, leaves, dtype_name, hold, recorder)
    finally:
        sysid._compile_model = real  # noqa: SLF001
