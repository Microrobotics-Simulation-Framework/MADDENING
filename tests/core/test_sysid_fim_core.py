"""``fim_core``: the jitted, sync-free half of ``fim``.

The three properties this file exists to pin, in the order they are
most likely to rot:

1. **``fim`` still answers exactly what it answered before it was
   jitted.**  Not "to a tolerance" -- bit for bit, including on the
   degenerate problems whose verdicts are decided at the rounding
   floor, which is where a change of arithmetic shows up first.
   :func:`_reference_fim` is the pre-jit pipeline written out, and
   :class:`TestFimUnchangedByJitting` compares against it.
2. **Nobody reads the device.**  ``fim_core`` performs zero
   device-to-host transfers and ``fim`` performs four.  Counted, and
   asserted as an exact number: a stray ``float(x)`` on a device array
   costs a pipeline stall, is invisible in every other test, and is the
   single easiest thing to reintroduce.
3. **The rollout is traced once, not once per call.**  ``fim`` used to
   invoke ``residual_fn`` twice per call and re-trace an N-step
   ``lax.scan`` every time.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    FIMCore,
    _inverse_noise_std,
    _masked_indices,
    _param_names,
    _precision_limited,
    _rank_and_crb,
    _resolve_rank_rtol,
    fim,
    fim_core,
    observations_from_history,
)
from maddening.warnings import PrecisionLimitWarning

K_TRUE, C_TRUE = 30.0, 2.0
_EPS32 = float(np.finfo(np.float32).eps)


# ---------------------------------------------------------------------------
# The reference: what ``fim`` computed before any of this was jitted
# ---------------------------------------------------------------------------


def _reference_fim(residual_fn, params, *, scale="relative", mask=None,
                   noise_std=None, rank_rtol=None):
    """``fim``'s pipeline as it stood before the jitted core, eager
    throughout.

    Deliberately a transcription and not a call into the module: a
    reference that shares the code under test cannot detect a change to
    it.  The host-side verdict helpers *are* shared, because they are
    the part that did not change and are covered in their own right --
    what this pins is the wiring around them, which is what moved: when
    the Jacobian is built, in what order the Gram product and ``eigh``
    run, and at what precision each stage lands.
    """
    if scale not in ("relative", None):
        raise ValueError(f"scale must be 'relative' or None, got {scale!r}")
    flat, unravel = ravel_pytree(params)
    idx = _masked_indices(params, mask)
    r0 = residual_fn(params)
    inv_sigma = _inverse_noise_std(noise_std, r0)

    def _r(theta):
        full = theta if idx is None else flat.at[idx].set(theta)
        r = ravel_pytree(residual_fn(unravel(full)))[0]
        return r if inv_sigma is None else r * inv_sigma

    theta0 = flat if idx is None else flat[idx]
    J = jax.jacfwd(_r)(theta0)
    names = _param_names(params)
    if idx is not None:
        names = tuple(names[i] for i in idx)
    zero_scaled = ()
    if scale == "relative":
        zero_scaled = tuple(
            nm for nm, at_zero in zip(names, np.asarray(theta0) == 0.0)
            if bool(at_zero))
        J = J * theta0[None, :]
    F = J.T @ J
    if not bool(jnp.all(jnp.isfinite(F))):
        raise FloatingPointError("non-finite Fisher matrix")
    eigvals, eigvecs = jnp.linalg.eigh(F)
    lo, hi = float(eigvals[0]), float(eigvals[-1])
    cond = float("inf") if lo <= 0.0 else hi / lo
    m = int(J.shape[0])
    rank, crb = _rank_and_crb(eigvals, eigvecs, rank_rtol, n_residual=m)
    n = int(np.asarray(eigvals).size)
    limited = _precision_limited(
        eigvals,
        _resolve_rank_rtol(eigvals.dtype, n, rank_rtol, n_residual=m),
        _resolve_rank_rtol(eigvals.dtype, n, None, n_residual=m))
    return dict(fim=F, eigvals=eigvals, eigvecs=eigvecs, rank=rank,
                cond=cond, crb=crb, param_names=names,
                zero_scaled=zero_scaled, limited=limited is not None)


# ---------------------------------------------------------------------------
# Counting what crosses the device boundary
# ---------------------------------------------------------------------------


_ArrayImpl = type(jnp.zeros(1))
#: Every route a jax array's contents take to the host.  ``np.asarray``
#: on a jax array goes through the *buffer protocol*, not ``__array__``,
#: so a counter that watches only ``__array__`` reports 3 where the
#: truth is 11 -- which is how this was first mis-measured.
#: ``jax.transfer_guard`` is no substitute: on the CPU backend it treats
#: every transfer as free and blocks nothing.
_TRANSFER_HOOKS = ("__array__", "__buffer__", "__float__", "__int__",
                   "__bool__", "__index__")


class _CountTransfers:
    """Count device-to-host reads of jax arrays inside the block."""

    def __init__(self):
        self.count = 0
        self.by_kind: dict[str, int] = {}
        self._saved: dict[str, object] = {}

    def __enter__(self):
        for name in _TRANSFER_HOOKS:
            original = getattr(_ArrayImpl, name)
            self._saved[name] = original

            def make(name=name, original=original):
                def hook(inner_self, *a, **kw):
                    self.count += 1
                    self.by_kind[name] = self.by_kind.get(name, 0) + 1
                    return original(inner_self, *a, **kw)
                return hook

            setattr(_ArrayImpl, name, make())
        return self

    def __exit__(self, *exc):
        for name, original in self._saved.items():
            setattr(_ArrayImpl, name, original)
        self._saved.clear()
        return False


def _quiet(fn, *a, **kw):
    """Run ``fn`` recording ``PrecisionLimitWarning`` instead of raising.

    ``pyproject.toml`` sets ``filterwarnings = ["error"]``, and several
    problems here warn on purpose.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [w for w in caught
                 if isinstance(w.message, PrecisionLimitWarning)]


# ---------------------------------------------------------------------------
# Problems: the spread, including the ones decided at the rounding floor
# ---------------------------------------------------------------------------


def _spring_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=K_TRUE, damping=C_TRUE,
                                 mass=1.0, rest_length=1.0,
                                 initial_position=0.5))
    gm.compile()
    return gm


def _spring_residual(gm, names, n_steps):
    init_state = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps)
    obs = observations_from_history(init_state, hist)
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for n in names:
            p["nodes"]["s"][n] = sub[n]

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=n_steps)
        return pos - truth

    return residual, {n: gm.params["nodes"]["s"][n] for n in names}


def _linear_residual(J64):
    """``r = A @ theta``: a Fisher matrix of designed spectrum."""
    n = J64.shape[1]
    A = jnp.asarray(J64, dtype=jnp.float32)
    params = {f"p{i}": jnp.float32(1.0) for i in range(n)}
    keys = tuple(params)
    return (lambda p: A @ jnp.stack([p[k] for k in keys])), params


def _spectrum_on_cutoff(n, m, ratio_x_cutoff, seed):
    """A Jacobian whose Fisher matrix sits ``ratio_x_cutoff`` times
    ``fim``'s own ``max(n, sqrt(m)) * eps`` cutoff -- i.e. a verdict
    decided at the arithmetic's noise floor.  Same construction as
    ``tests/verification/hypothesis/test_hypothesis_sysid.py``."""
    rng = np.random.default_rng(seed)
    cutoff = max(n, np.sqrt(m)) * _EPS32
    mid = np.geomspace(1e-3, 1.0, max(n - 1, 1))
    mid[-1] = 1.0
    lam = np.sort(np.concatenate([[ratio_x_cutoff * cutoff], mid]))[:n]
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    V = q * np.sign(np.diag(r))
    U = np.linalg.qr(rng.standard_normal((m, n)))[0]
    return (U * np.sqrt(lam)) @ V.T


@pytest.fixture(scope="module")
def problems():
    """``{name: (residual_fn, params, kwargs)}``.

    Covers, deliberately: an identifiable problem; the spring's
    ``(k, c, m)`` scale direction, which is genuinely rank deficient and
    whose smallest eigenvalue is a rounding artefact either side of
    zero; a ``zero_scaled`` parameter; three problems built with their
    deciding eigenvalue ratio *on* the cutoff, which is the
    precision-limited case; ``scale=None``; both shapes of
    ``noise_std``; a raised and a zeroed ``rank_rtol``; and a mask.
    """
    gm = _spring_gm()
    res2, sub2 = _spring_residual(gm, ("stiffness", "damping"), 200)
    res3, sub3 = _spring_residual(gm, ("stiffness", "damping", "mass"), 200)
    res3l, sub3l = _spring_residual(gm, ("stiffness", "damping", "mass"), 800)
    out = {
        "identifiable_pair": (res2, sub2, {}),
        "kcm_null_direction": (res3, sub3, {}),
        "kcm_null_long_residual": (res3l, sub3l, {}),
        "absolute_scale": (res2, sub2, {"scale": None}),
        "noise_scalar": (res2, sub2, {"noise_std": 2.0}),
        "noise_per_row": (res2, sub2, {"noise_std": jnp.full((200,), 0.5)}),
        "rank_rtol_raised": (res2, sub2, {"rank_rtol": 1e-3}),
        "rank_rtol_zero": (res2, sub2, {"rank_rtol": 0.0}),
        "masked": (res2, sub2, {"mask": {"stiffness": True,
                                         "damping": False}}),
    }
    rng = np.random.default_rng(7)
    fn, params = _linear_residual(rng.standard_normal((40, 3)))
    params = dict(params, p1=jnp.float32(0.0))
    out["zero_scaled"] = (fn, params, {})
    out["zero_scaled_absolute"] = (fn, params, {"scale": None})
    for n, m, ratio, seed in [(3, 200, 1.0, 11), (4, 1024, 1.3, 23),
                              (2, 64, 0.8, 31)]:
        fn, params = _linear_residual(_spectrum_on_cutoff(n, m, ratio, seed))
        out[f"on_cutoff_n{n}_m{m}"] = (fn, params, {"scale": None})
    fn, params = _linear_residual(_spectrum_on_cutoff(4, 300, 1e6, 5))
    out["well_conditioned"] = (fn, params, {"scale": None})
    fn, params = _linear_residual(_spectrum_on_cutoff(4, 300, 0.0, 5))
    out["exactly_singular"] = (fn, params, {"scale": None})
    B = rng.standard_normal((50, 3))
    B[:, 2] = B[:, 1]
    fn, params = _linear_residual(B)
    out["duplicate_column"] = (fn, params, {"scale": None})
    return out


ALL_PROBLEMS = [
    "identifiable_pair", "kcm_null_direction", "kcm_null_long_residual",
    "absolute_scale", "noise_scalar", "noise_per_row", "rank_rtol_raised",
    "rank_rtol_zero", "masked", "zero_scaled", "zero_scaled_absolute",
    "on_cutoff_n3_m200", "on_cutoff_n4_m1024", "on_cutoff_n2_m64",
    "well_conditioned", "exactly_singular", "duplicate_column",
]


# ---------------------------------------------------------------------------


class TestFimUnchangedByJitting:
    """``fim`` publishes the same numbers it published before the core
    was jitted -- bit for bit.

    Not a tolerance.  ``rank`` and ``cond`` are decided by comparing two
    numbers at the ``max(n, sqrt(m)) * eps`` floor, so a change of one
    float32 ulp in ``F`` is enough to move a published verdict: letting
    ``jax.jit`` fold the transpose into ``J.T @ J`` moved ``cond`` on
    the spring's scale direction from ``inf`` to ``3.45e+07`` and
    ``rank`` on an on-cutoff matrix from 3 to 2.  That is why ``fim``
    jits only the Jacobian -- which *is* bit-identical jitted -- and
    leaves the Gram product, ``eigh`` and the float64 host verdict layer
    eager.  These assertions are how that stays true.
    """

    @pytest.mark.parametrize("name", ALL_PROBLEMS)
    def test_report_is_bit_identical_to_the_pre_jit_pipeline(
            self, problems, name):
        fn, params, kw = problems[name]
        got, got_warnings = _quiet(fim, fn, params, **kw)
        want, want_warnings = _quiet(_reference_fim, fn, params, **kw)

        assert got.rank == want["rank"]
        assert repr(got.cond) == repr(want["cond"]), (got.cond, want["cond"])
        assert got.param_names == want["param_names"]
        assert got.zero_scaled == want["zero_scaled"]
        for field in ("fim", "eigvals", "eigvecs", "crb"):
            a = np.asarray(getattr(got, field))
            b = np.asarray(want[field])
            assert np.array_equal(a, b, equal_nan=True), (
                f"{name}.{field} moved: max |delta| = "
                f"{np.nanmax(np.abs(a - b))}")
        # ``_reference_fim`` reports the precision band rather than
        # warning about it, so the comparison is against its verdict.
        assert len(got_warnings) == int(want["limited"])

    def test_a_precision_limited_verdict_is_in_the_spread(self, problems):
        """The equivalence above is only worth something if the spread
        reaches the cases where the arithmetic decides the answer.

        Asserted rather than assumed: were every problem here
        comfortably resolved, jitting the Gram product would pass the
        whole class and the guard would be empty.
        """
        warned = []
        for name in ALL_PROBLEMS:
            fn, params, kw = problems[name]
            _, got = _quiet(fim, fn, params, **kw)
            if got:
                warned.append(name)
        assert len(warned) >= 3, warned

    def test_a_rank_deficient_problem_is_in_the_spread(self, problems):
        """Likewise for ``crb``'s ``+inf`` branch and a non-finite
        ``cond``: the fail-closed path has to be exercised."""
        deficient = []
        for name in ALL_PROBLEMS:
            fn, params, kw = problems[name]
            report, _ = _quiet(fim, fn, params, **kw)
            if report.rank < len(report.param_names):
                assert bool(np.any(np.isinf(np.asarray(report.crb))))
                deficient.append(name)
        assert len(deficient) >= 3, deficient


class TestCoreAgreesWithReport:
    """``fim_core`` answers the same questions as ``fim``.

    Two deliberate differences, both stated in ``FIMCore``: the core's
    verdicts are computed at the matrix's own precision rather than
    widened to float64 on the host, and the whole thing is jitted, so
    ``F`` itself can differ by a float32 ulp.  ``rank`` and the
    ``+inf`` pattern of ``crb`` are still required to match exactly,
    because those are what a gate reads; the finite bounds are allowed
    the ulp.  If that ever stops holding away from the cutoff, the two
    implementations of the rule have drifted apart, which is the cost of
    having two and the reason this test exists.
    """

    @pytest.mark.parametrize("name", ALL_PROBLEMS)
    def test_rank_and_crb_infinities_match_fim(self, problems, name):
        fn, params, kw = problems[name]
        report, _ = _quiet(fim, fn, params, **kw)
        core, _ = _quiet(jax.jit(functools.partial(fim_core, fn, **kw)),
                         params)
        if name.startswith("on_cutoff"):
            # Decided at the noise floor by construction: the two
            # precisions are entitled to disagree, and saying so is what
            # ``precision_limited`` is for.
            assert bool(core.precision_limited)
            return
        assert int(core.rank) == report.rank, name
        want_inf = np.isinf(np.asarray(report.crb))
        assert np.array_equal(np.isinf(np.asarray(core.crb)), want_inf), name
        finite = ~want_inf
        if finite.any():
            np.testing.assert_allclose(
                np.asarray(core.crb)[finite],
                np.asarray(report.crb)[finite], rtol=1e-4)
        assert bool(core.finite)
        if report.rank == len(report.param_names):
            assert np.isclose(float(core.cond), report.cond, rtol=1e-4), name
        else:
            # Rank deficient.  ``cond`` is a raw ``eigvals[-1] /
            # eigvals[0]`` with no cutoff protecting it, so a null
            # eigenvalue that is a rounding artefact either side of zero
            # reads ``inf`` in one arithmetic and a huge finite number in
            # the other -- ``FIMReport`` says as much.  Both mean
            # singular; neither is a number to compare.  ``rank`` and
            # ``crb``, which *are* threshold-protected, matched above.
            assert float(core.cond) > 1.0 / core.rank_rtol, name
            assert report.cond > 1.0 / core.rank_rtol, name

    @pytest.mark.parametrize("name", ALL_PROBLEMS)
    def test_zero_scaled_mask_names_the_same_parameters(self, problems, name):
        fn, params, kw = problems[name]
        report, _ = _quiet(fim, fn, params, **kw)
        core, _ = _quiet(fim_core, fn, params, **kw)
        named = tuple(nm for nm, z in zip(core.param_names,
                                          np.asarray(core.zero_scaled))
                      if bool(z))
        assert named == report.zero_scaled
        assert core.param_names == report.param_names

    @pytest.mark.parametrize("name", ALL_PROBLEMS)
    def test_precision_limited_flag_matches_the_warning(self, problems, name):
        """The device flag and the host warning answer the same
        question, and a live loop reads the flag instead of catching the
        warning -- so they have to agree."""
        fn, params, kw = problems[name]
        _, warned = _quiet(fim, fn, params, **kw)
        core, _ = _quiet(fim_core, fn, params, **kw)
        assert bool(core.precision_limited) == bool(warned), name
        if warned:
            ratio = float(core.deciding_ratio)
            assert np.isfinite(ratio) and ratio > 0.0
        else:
            assert float(core.deciding_ratio) == 0.0

    def test_jitting_the_core_does_not_change_it(self, problems):
        """Eager and jitted ``fim_core`` must agree on every verdict.

        ``F`` may move by an ulp between them -- the compiler is free to
        schedule the Gram product differently -- but nothing that is
        *reported* may, away from the cutoff.
        """
        for name in ALL_PROBLEMS:
            if name.startswith("on_cutoff"):
                continue
            fn, params, kw = problems[name]
            eager, _ = _quiet(fim_core, fn, params, **kw)
            jitted, _ = _quiet(jax.jit(functools.partial(fim_core, fn, **kw)),
                               params)
            assert int(eager.rank) == int(jitted.rank), name
            assert np.array_equal(np.isinf(np.asarray(eager.crb)),
                                  np.isinf(np.asarray(jitted.crb))), name
            assert bool(eager.finite) == bool(jitted.finite), name
            assert bool(eager.precision_limited) == \
                bool(jitted.precision_limited), name


class TestNoHostSyncs:
    """Nothing reads the device that does not have to.

    The count is asserted as an exact number rather than a bound.  A
    bound would let the next edit spend the slack, and this is the
    property that rots silently: a ``float(x)`` on a device array is a
    pipeline stall on a GPU and costs nothing measurable on the CPU this
    runs on, so no other test in the suite would notice.
    """

    def test_fim_core_reads_nothing_back(self, problems):
        fn, params, kw = problems["kcm_null_direction"]
        core_fn = jax.jit(functools.partial(fim_core, fn, **kw))
        jax.block_until_ready(core_fn(params))       # trace and compile
        with _CountTransfers() as counter:
            core = core_fn(params)
            jax.block_until_ready(core)
        assert counter.count == 0, counter.by_kind

    def test_fim_core_reads_nothing_back_with_a_noise_model(self, problems):
        """``noise_std`` is validated on the host, which is a transfer
        -- but at *trace* time, not per call.  A loop pays it once."""
        fn, params, _ = problems["identifiable_pair"]
        core_fn = jax.jit(functools.partial(fim_core, fn, noise_std=2.0))
        jax.block_until_ready(core_fn(params))
        with _CountTransfers() as counter:
            jax.block_until_ready(core_fn(params))
        assert counter.count == 0, counter.by_kind

    def test_fim_performs_exactly_four_transfers(self, problems):
        """``eigvals``, ``eigvecs``, the finiteness flag and the
        zero-scaled mask, in one ``jax.device_get``.  Everything the
        host verdict layer does afterwards is numpy on numpy."""
        fn, params, kw = problems["kcm_null_direction"]
        _quiet(fim, fn, params, **kw)                # warm the jit cache
        with _CountTransfers() as counter:
            _quiet(fim, fn, params, **kw)
        assert counter.count == 4, counter.by_kind

    def test_a_host_side_noise_model_costs_no_extra_transfer(
            self, problems):
        """Sigma is cast and inverted in numpy, so a Python scalar never
        touches a device and validating it reads nothing back."""
        fn, params, _ = problems["identifiable_pair"]
        _quiet(fim, fn, params, noise_std=2.0)
        with _CountTransfers() as counter:
            _quiet(fim, fn, params, noise_std=2.0)
        assert counter.count == 4, counter.by_kind

    def test_a_device_resident_noise_model_costs_one_more(self, problems):
        """A sigma that is already a jax array has to come back to be
        refused -- ``F = J^T Sigma^-1 J`` is a Fisher matrix only for a
        positive sigma and a trace cannot refuse anything -- and it is
        re-read per call, because it may have changed.  That is the
        price of the check, and it is why ``fim_core`` bakes sigma in at
        trace time instead."""
        fn, params, _ = problems["identifiable_pair"]
        sigma = jnp.full((200,), 0.5)
        _quiet(fim, fn, params, noise_std=sigma)
        with _CountTransfers() as counter:
            _quiet(fim, fn, params, noise_std=sigma)
        assert counter.count == 5, counter.by_kind

    def test_the_counter_can_see_a_transfer(self, problems):
        """The counter counts ``np.asarray``, which on a jax array goes
        through the buffer protocol and *not* ``__array__``.  A counter
        blind to that reports 3 for a call that makes 11 transfers, so
        the assertions above would pass however many were added."""
        x = jnp.arange(4, dtype=jnp.float32)
        with _CountTransfers() as counter:
            np.asarray(x)
            float(x[0])
            bool(x[0] > 0)
            jax.device_get(x)
        assert counter.count == 4, counter.by_kind
        assert counter.by_kind.get("__buffer__") == 1


class TestTracedOnce:
    """The rollout is traced once per signature, not once per call."""

    def _counting_spring(self, n_steps=200):
        calls = [0]
        gm = _spring_gm()
        residual, params = _spring_residual(
            gm, ("stiffness", "damping"), n_steps)

        def counted(sub):
            calls[0] += 1
            return residual(sub)

        return counted, params, calls

    def test_repeat_calls_do_not_re_enter_residual_fn(self):
        fn, params, calls = self._counting_spring()
        _quiet(fim, fn, params)
        calls[0] = 0
        for _ in range(5):
            _quiet(fim, fn, params)
        assert calls[0] == 0, (
            f"residual_fn was entered {calls[0]} times over 5 calls; the "
            "traced Jacobian is not being reused")

    def test_the_first_call_enters_residual_fn_once_not_twice(self):
        """``fim`` used to evaluate ``residual_fn(params)`` in full for
        a structure that ``_inverse_noise_std`` discards on its first
        line when ``noise_std`` is ``None`` -- a whole extra rollout per
        call, buying nothing.  One entry now: the ``jacfwd`` trace."""
        fn, params, calls = self._counting_spring()
        _quiet(fim, fn, params)
        assert calls[0] == 1, calls[0]

    def test_a_noise_model_costs_a_trace_not_a_rollout(self):
        """``jax.eval_shape`` enters ``residual_fn`` to learn its output
        structure and executes none of it, so the first call enters
        twice: once to trace the shapes, once to trace the Jacobian.
        Neither happens again -- ``eval_shape`` is cached on the same
        avals by JAX itself."""
        fn, params, calls = self._counting_spring()
        _quiet(fim, fn, params, noise_std=2.0)
        first = calls[0]
        calls[0] = 0
        _quiet(fim, fn, params, noise_std=2.0)
        assert first == 2, first
        assert calls[0] == 0, (
            f"{calls[0]} entries on a warm call with a noise model")

    def test_changing_params_does_not_retrace(self):
        fn, params, calls = self._counting_spring()
        _quiet(fim, fn, params)
        calls[0] = 0
        moved = dict(params, stiffness=jnp.float32(1.4 * K_TRUE))
        _quiet(fim, fn, moved)
        assert calls[0] == 0


class TestCacheKeyIsComplete:
    """Every static argument that changes the answer is part of the key.

    A key missing one of these returns another call's compiled Jacobian
    and the wrong report, silently -- there is no shape or dtype error
    to catch it, because the shapes agree.
    """

    def test_scale_none_and_relative_do_not_share_an_entry(self, problems):
        fn, params, _ = problems["identifiable_pair"]
        rel, _ = _quiet(fim, fn, params)
        absolute, _ = _quiet(fim, fn, params, scale=None)
        assert not np.allclose(np.asarray(rel.fim), np.asarray(absolute.fim))
        assert not np.isclose(rel.cond, absolute.cond)

    def test_a_mask_does_not_share_an_entry_with_no_mask(self, problems):
        fn, params, _ = problems["identifiable_pair"]
        full, _ = _quiet(fim, fn, params)
        masked, _ = _quiet(fim, fn, params,
                           mask={"stiffness": True, "damping": False})
        assert full.param_names == ("['damping']", "['stiffness']")
        assert masked.param_names == ("['stiffness']",)
        assert np.asarray(masked.fim).shape == (1, 1)

    def test_two_different_residuals_do_not_share_an_entry(self, problems):
        fn_a, params_a, _ = problems["identifiable_pair"]
        fn_b, params_b, _ = problems["well_conditioned"]
        a, _ = _quiet(fim, fn_a, params_a)
        b, _ = _quiet(fim, fn_b, params_b, scale=None)
        assert a.param_names != b.param_names

    def test_an_unhashable_residual_fn_still_works(self, problems):
        """A callable object that defines ``__eq__`` without
        ``__hash__`` cannot key the cache.  That is a reason to skip the
        cache, not to refuse the call."""
        fn, params, _ = problems["identifiable_pair"]

        class Unhashable:
            __hash__ = None

            def __call__(self, p):
                return fn(p)

        got, _ = _quiet(fim, Unhashable(), params)
        want, _ = _quiet(fim, fn, params)
        assert got.rank == want.rank
        assert np.array_equal(np.asarray(got.crb), np.asarray(want.crb))


class TestCoreFailsClosed:
    """``crb`` is ``+inf`` unless finiteness was positively established
    -- on the device as on the host.

    A live gate reads ``crb < tol``.  For that to be safe the bound must
    be ``+inf``, never ``NaN`` and never ``0.0``, whenever the data did
    not determine the parameter *or* the matrix holds nothing at all.
    The natural spelling, ``where(support > n * eps, inf, crb)``, reads
    the right way round and is the wrong way round: a ``NaN`` support
    fails that comparison and the rescue never fires.
    """

    def _nan_problem(self, n=2):
        """``n >= 2`` on purpose.  ``eigh`` of a 1x1 ``NaN`` matrix
        returns the eigenvector ``[1.0]``, so ``diag(P)`` is 1 and even
        a fail-*open* ``where(support > n * eps, inf, crb)`` happens to
        answer ``+inf``.  From 2x2 up the eigenvectors come back ``NaN``
        too, ``NaN > n * eps`` is False, and the rescue never fires --
        which is precisely the inversion :func:`_rank_and_crb`'s comment
        describes, and the only size at which this test can see it."""
        keys = tuple(f"p{i}" for i in range(n))

        def residual(p):
            v = jnp.stack([p[k] for k in keys])
            return jnp.concatenate([v * jnp.nan, v])

        return residual, {k: jnp.float32(1.0) for k in keys}

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_a_nan_matrix_gives_infinite_bounds_not_nan(self, n):
        fn, params = self._nan_problem(n)
        core = fim_core(fn, params)
        assert not bool(core.finite)
        assert int(core.rank) == 0
        crb = np.asarray(core.crb)
        assert np.all(np.isinf(crb)), crb
        assert not np.any(np.isnan(crb))
        assert not bool(np.any(crb < 1e30)), "crb < tol must read False"

    def test_the_core_makes_no_nan_on_a_rank_deficient_problem(
            self, problems):
        """Under ``jax_debug_nans`` a user is asking JAX to stop at the
        first ``NaN`` so they can find their own.  A rank-deficient
        Fisher matrix has eigenvalues at or below zero by construction,
        so dividing the eigenvector squares by them -- instead of by a
        substituted 1.0 in the columns the mask throws away -- produces
        a ``NaN`` this function then discards, and stops that user's
        debugger on a value that was never used.

        Run **eagerly**, and that is load-bearing: under ``jax.jit``
        ``jax_debug_nans`` inspects only the outputs of the compiled
        function, and every ``NaN`` here is discarded before it reaches
        one.  Eager dispatch checks each operation, which is the only
        way to see an intermediate.  The spring problems are left out
        for the same reason -- op-by-op over a 200-step rollout is
        minutes, and these linear residuals reach the same code."""
        for name in ("exactly_singular", "duplicate_column", "zero_scaled"):
            fn, params, kw = problems[name]
            with jax.debug_nans(True):
                core = fim_core(fn, params, **kw)
                jax.block_until_ready(core)
            assert bool(core.finite), name

    def test_the_core_makes_no_nan_on_a_zero_fisher_matrix(self):
        """The hardest case for the masked divisions, and the one that
        turns them from defensive into load-bearing.

        A residual that does not depend on the parameters at all gives
        ``F = 0``: ``eigh`` returns zero eigenvalues and the *identity*
        as eigenvectors, so the squares being divided include exact
        zeros and the divisor is an exact zero.  Unmasked that is
        ``0 / 0``, and ``cond`` is ``0 / 0`` as well -- two ``NaN``s
        computed and immediately thrown away, which is exactly what a
        ``jax_debug_nans`` user does not want to stop on.  The verdict
        itself is unaffected either way, so nothing but this notices."""
        def residual(p):
            del p
            return jnp.arange(8, dtype=jnp.float32)

        params = {f"p{i}": jnp.float32(1.0 + i) for i in range(3)}
        with jax.debug_nans(True):
            core = fim_core(residual, params)   # eager: see above
            jax.block_until_ready(core)
        assert int(core.rank) == 0
        assert bool(jnp.all(jnp.isinf(core.crb)))
        assert bool(jnp.isinf(core.cond))
        assert bool(core.finite)

    def test_fim_still_raises_on_the_same_matrix(self):
        """The core reports it, ``fim`` refuses it.  Both, not either:
        a loop cannot afford to unwind and a report cannot be built on
        ``NaN``."""
        fn, params = self._nan_problem()
        with pytest.raises(FloatingPointError, match="non-finite Fisher"):
            fim(fn, params)

    def test_an_unresolved_direction_gives_an_infinite_bound(self, problems):
        fn, params, kw = problems["kcm_null_direction"]
        core = fim_core(fn, params, **kw)
        assert int(core.rank) == 2
        assert bool(jnp.all(jnp.isinf(core.crb)))


class TestCoreIsAPytree:
    """``FIMCore`` is registered as a pytree, so it can come out of a
    jitted function and go into a ``lax.scan`` carry or a ``lax.cond``
    -- which is what "usable in a control loop" has to mean."""

    def test_it_survives_tree_flatten_and_unflatten(self, problems):
        fn, params, kw = problems["identifiable_pair"]
        core = fim_core(fn, params, **kw)
        leaves, treedef = jax.tree.flatten(core)
        assert len(leaves) == 10
        back = jax.tree.unflatten(treedef, leaves)
        assert isinstance(back, FIMCore)
        assert back.param_names == core.param_names
        assert back.n_residual == core.n_residual

    def test_the_static_metadata_does_not_become_a_traced_leaf(
            self, problems):
        """``param_names`` and ``n_residual`` are metadata.  Were they
        data fields, ``jax.jit`` would try to make arrays of a tuple of
        strings and fail -- or worse, succeed for ``n_residual`` and
        make the residual length a traced value nothing can index by."""
        fn, params, kw = problems["identifiable_pair"]
        core = jax.jit(functools.partial(fim_core, fn, **kw))(params)
        assert isinstance(core.param_names, tuple)
        assert all(isinstance(n, str) for n in core.param_names)
        assert isinstance(core.n_residual, int)
        assert isinstance(core.rank_rtol, float)

    def test_a_gate_can_branch_on_it_without_a_transfer(self, problems):
        """The point of the whole exercise: decide, on the device,
        whether the data determine the parameters well enough to act
        on."""
        fn, params, kw = problems["identifiable_pair"]

        @jax.jit
        def gate(p):
            core = fim_core(fn, p, **kw)
            ok = core.finite & ~core.precision_limited & jnp.all(
                core.crb < 1e-2)
            return jax.lax.cond(ok, lambda: 1.0, lambda: 0.0)

        jax.block_until_ready(gate(params))
        with _CountTransfers() as counter:
            jax.block_until_ready(gate(params))
        assert counter.count == 0, counter.by_kind


class TestCoreRefusesTheSameInputs:
    """Validation did not move to the device with everything else."""

    def test_a_bad_scale_is_refused(self, problems):
        fn, params, _ = problems["identifiable_pair"]
        with pytest.raises(ValueError, match="scale must be"):
            fim_core(fn, params, scale="absolute")

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), 1e-320])
    def test_a_sigma_that_is_not_a_noise_model_is_refused(
            self, problems, sigma):
        fn, params, _ = problems["identifiable_pair"]
        with pytest.raises(ValueError, match="strictly positive"):
            fim_core(fn, params, noise_std=sigma)

    @pytest.mark.parametrize("rtol", [-1.0, float("inf"), float("nan")])
    def test_a_bad_rank_rtol_is_refused(self, problems, rtol):
        fn, params, _ = problems["identifiable_pair"]
        with pytest.raises(ValueError, match="rank_rtol must be"):
            fim_core(fn, params, rank_rtol=rtol)

    def test_a_mask_selecting_nothing_is_refused(self, problems):
        fn, params, _ = problems["identifiable_pair"]
        with pytest.raises(ValueError, match="selects no parameters"):
            fim_core(fn, params,
                     mask={"stiffness": False, "damping": False})
