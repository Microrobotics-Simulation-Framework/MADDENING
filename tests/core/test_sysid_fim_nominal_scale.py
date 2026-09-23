"""``fim(scale="nominal")``: a column scale read from ``ParamSpec.bounds``.

Relative scaling multiplies column ``j`` of the Jacobian by the value of
parameter ``j``, so a parameter sitting at exactly ``0.0`` vanishes from
``F`` however well the data determine it.  ``zero_scaled`` reports that;
``scale="nominal"`` removes it by taking the scale from the width of the
parameter's declared bounds instead.  What this file pins:

1. a parameter at ``0.0`` with finite bounds is judged on the data, on
   fixtures where the data genuinely determine it (a linear residual and
   a spring rollout whose ``initial_velocity`` is the zero);
2. every row of the policy table in :func:`fim`'s docstring, including
   the fail-visible naming in ``value_scaled``;
3. column scaling is the *only* thing the mode changes -- all-ones
   widths reproduce ``scale=None`` bit for bit, widths equal to the
   values reproduce ``scale="relative"`` bit for bit, jitted and eager
   Jacobians agree bit for bit, and ``fim_core`` carries the same
   verdicts;
4. the composition with ``noise_std`` (rows, before) and ``mask``
   (columns, selected first) is the one the docstring states;
5. the refusals: ``specs`` under the wrong scale, no ``specs`` under
   the right one, a width that is a zero or an ``inf`` column at
   the parameters' precision, and a ``specs`` tree that does not mirror
   ``params`` (a non-dict, a misspelt key, a ``to_dict()`` entry, a
   ``ParamSpec`` above a dict level) -- each of which ``_spec_for``
   would otherwise resolve to the default spec, giving a report
   bit-identical to ``scale="relative"`` under the ``"nominal"`` label;
6. a list/tuple-valued params leaf is reached by a spec keyed for it,
   either one spec over the sequence or one per position.

Every rank/cond assertion is made on two or more parameters: ``eigh`` of
a 1x1 matrix has one eigenvector, ``[1.0]``, whatever the entry, so a
single-parameter fixture cannot tell a right answer from a broken one.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import FIMCore, fim, fim_core, observations_from_history

_FIELDS = ("fim", "eigvals", "eigvecs", "crb")


def _identical(a, b, *, fields=_FIELDS):
    """Bit-for-bit equality of two reports/cores on the array fields,
    plus the verdicts."""
    for f in fields:
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        assert np.array_equal(x, y, equal_nan=True), (
            f"{f} moved: max |delta| = {np.nanmax(np.abs(x - y))}")
    assert int(a.rank) == int(b.rank)
    assert repr(float(a.cond)) == repr(float(b.cond))
    assert a.param_names == b.param_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pair_at_zero():
    """``r = (a + b, a - b)``: a 2x2 Jacobian ``[[1, 1], [1, -1]]`` that
    determines both parameters completely, with ``b`` sitting at zero.
    Square on purpose -- see :class:`TestCompositionWithNoiseAndMask`."""
    params = {"a": jnp.float32(1.0), "b": jnp.float32(0.0)}

    def residual(p):
        return jnp.stack([p["a"] + p["b"], p["a"] - p["b"]])

    return residual, params


def _linear(seed, n, m, values):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((m, n)), dtype=jnp.float32)
    keys = tuple(f"p{i}" for i in range(n))
    params = {k: jnp.float32(v) for k, v in zip(keys, values)}
    return (lambda p: A @ jnp.stack([p[k] for k in keys])), params


@pytest.fixture(scope="module")
def spring_with_zero_velocity():
    """A spring released from rest at half its rest length.

    The trajectory is non-trivial, so the position history depends on
    the initial velocity -- the data determine it -- and that velocity
    is exactly ``0.0``, which is ``SpringDamperNode``'s default.  This
    is the case the mode exists for.  Built once: ``run_scan`` advances
    the graph's own state, and the residual closes over the initial
    state it was built with.
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 mass=1.0, rest_length=1.0,
                                 initial_position=0.5, initial_velocity=0.0))
    gm.compile()
    init_state = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(200)
    obs = observations_from_history(init_state, hist)
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    truth = obs["s"]["position"][1:]
    names = ("stiffness", "initial_velocity")

    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for n in names:
            p["nodes"]["s"][n] = sub[n]
        init = {"s": {
            "position": jnp.asarray(0.5, jnp.float32),
            "velocity": jnp.asarray(p["nodes"]["s"]["initial_velocity"],
                                    jnp.float32)}}

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=200)
        return pos - truth

    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    node_specs = gm.param_specs()["nodes"]["s"]
    return residual, sub, {n: node_specs[n] for n in names}


# ---------------------------------------------------------------------------
# 1. The defect, and its removal
# ---------------------------------------------------------------------------


class TestAZeroValuedParameterIsJudgedOnTheData:

    def test_linear_pair_is_zero_scaled_under_relative_and_resolved_under_nominal(self):
        residual, params = _pair_at_zero()
        rel = fim(residual, params)                       # scale="relative"
        assert rel.zero_scaled == ("['b']",)
        assert rel.rank == 1
        assert bool(np.isinf(np.asarray(rel.crb)[1]))

        specs = {"a": ParamSpec(bounds=(0.0, 2.0)),
                 "b": ParamSpec(bounds=(-1.0, 1.0))}
        nom = fim(residual, params, scale="nominal", specs=specs)
        assert nom.rank == 2
        assert nom.zero_scaled == ()
        assert nom.value_scaled == ()
        assert np.all(np.isfinite(np.asarray(nom.crb)))
        assert np.isfinite(nom.cond)

    def test_the_spring_fixture_can_express_the_defect(
            self, spring_with_zero_velocity):
        """Guard on the fixture, not the feature: under ``scale=None``
        both parameters must already be resolved, or the nominal result
        below would be testing nothing."""
        residual, sub, _ = spring_with_zero_velocity
        assert float(sub["initial_velocity"]) == 0.0
        absolute = fim(residual, sub, scale=None)
        assert absolute.rank == 2
        assert np.all(np.isfinite(np.asarray(absolute.crb)))

    def test_spring_initial_velocity_at_zero_is_resolved_with_a_bounded_spec(
            self, spring_with_zero_velocity):
        residual, sub, specs = spring_with_zero_velocity
        rel = fim(residual, sub)
        assert rel.zero_scaled == ("['initial_velocity']",)
        assert rel.rank == 1

        bounded = dict(specs, initial_velocity=ParamSpec(
            trainable=False, bounds=(-1.0, 1.0)))
        nom = fim(residual, sub, scale="nominal", specs=bounded)
        assert nom.rank == 2
        assert nom.zero_scaled == ()
        assert np.all(np.isfinite(np.asarray(nom.crb)))
        # ``stiffness`` is declared ``(0.0, None)`` / ``"log"`` by the
        # node: no width, so it is value-scaled and the report says so.
        assert nom.value_scaled == ("['stiffness']",)

    def test_the_graphs_own_specs_leave_the_zero_named_twice(
            self, spring_with_zero_velocity):
        """With the node's own specs -- ``initial_velocity`` unbounded
        -- the column is value-scaled, and a value-scaled zero is a
        zero.  The mode must be at least as honest as ``"relative"``:
        the parameter is in ``value_scaled`` *and* ``zero_scaled``."""
        residual, sub, specs = spring_with_zero_velocity
        nom = fim(residual, sub, scale="nominal", specs=specs)
        assert nom.rank == 1
        assert nom.zero_scaled == ("['initial_velocity']",)
        assert nom.value_scaled == ("['initial_velocity']", "['stiffness']")


# ---------------------------------------------------------------------------
# 2. The policy table, row by row
# ---------------------------------------------------------------------------


def _column_scales(nominal_report, absolute_report):
    """Recover the per-column multiplier from ``F_nom = S F_abs S``,
    ``S = diag(s)``: ``s_j = sqrt(F_nom[j, j] / F_abs[j, j])``, with
    the sign from the off-diagonal.  Only a *diagonal* rescaling of
    ``F`` is consistent with column scaling, and that consistency is
    asserted too, so a row-side or one-sided multiplication is seen."""
    Fn = np.asarray(nominal_report.fim, dtype=np.float64)
    Fa = np.asarray(absolute_report.fim, dtype=np.float64)
    s = np.sqrt(np.diag(Fn) / np.diag(Fa))
    np.testing.assert_allclose(Fn, np.outer(s, s) * Fa, rtol=1e-5, atol=0.0)
    return s


class TestThePolicyTable:
    """One test per row of the table in :func:`fim`'s docstring."""

    def test_finite_bounds_scale_by_the_width_whatever_the_transform(self):
        fn, params = _linear(1, 3, 12, (0.0, 0.5, 4.0))
        specs = {"p0": ParamSpec(bounds=(-1.0, 1.0)),                 # width 2
                 "p1": ParamSpec(bounds=(0.0, 1.0), transform="logit"),  # 1
                 "p2": ParamSpec(bounds=(3.0, 8.0))}                  # width 5
        nom = fim(fn, params, scale="nominal", specs=specs)
        absolute = fim(fn, params, scale=None)
        np.testing.assert_allclose(_column_scales(nom, absolute),
                                   [2.0, 1.0, 5.0], rtol=1e-6)
        assert nom.value_scaled == ()
        assert nom.zero_scaled == ()
        assert nom.rank == 3

    def test_the_width_is_a_scale_not_a_location(self):
        """The midpoint of ``(-1, 1)`` is ``0.0`` -- the failure being
        removed.  Shifting the bounds must not change the column."""
        fn, params = _linear(2, 2, 10, (0.0, 1.0))
        centred = {"p0": ParamSpec(bounds=(-1.0, 1.0)), "p1": ParamSpec(bounds=(0.0, 3.0))}
        shifted = {"p0": ParamSpec(bounds=(9.0, 11.0)), "p1": ParamSpec(bounds=(-3.0, 0.0))}
        _identical(fim(fn, params, scale="nominal", specs=centred),
                   fim(fn, params, scale="nominal", specs=shifted))

    def test_log_with_a_lower_bound_scales_by_the_offset_value_and_is_named(self):
        fn, params = _linear(3, 2, 10, (300.0, 2.0))
        specs = {"p0": ParamSpec(bounds=(273.0, None), transform="log"),
                 "p1": ParamSpec(bounds=(0.0, 4.0))}
        nom = fim(fn, params, scale="nominal", specs=specs)
        absolute = fim(fn, params, scale=None)
        np.testing.assert_allclose(_column_scales(nom, absolute),
                                   [300.0 - 273.0, 4.0], rtol=1e-6)
        assert nom.value_scaled == ("['p0']",)
        assert nom.zero_scaled == ()

    def test_log_at_its_lower_bound_is_a_named_zero(self):
        fn, params = _linear(3, 2, 10, (273.0, 2.0))
        specs = {"p0": ParamSpec(bounds=(273.0, None), transform="log"),
                 "p1": ParamSpec(bounds=(0.0, 4.0))}
        nom = fim(fn, params, scale="nominal", specs=specs)
        assert nom.value_scaled == ("['p0']",)
        assert nom.zero_scaled == ("['p0']",)
        assert nom.rank == 1
        assert bool(np.isinf(np.asarray(nom.crb)[0]))

    def test_log_without_a_lower_bound_scales_by_the_value_and_is_named(self):
        fn, params = _linear(4, 2, 10, (7.0, 2.0))
        specs = {"p0": ParamSpec(bounds=(None, None), transform="log"),
                 "p1": ParamSpec(bounds=(0.0, 4.0))}
        nom = fim(fn, params, scale="nominal", specs=specs)
        absolute = fim(fn, params, scale=None)
        np.testing.assert_allclose(_column_scales(nom, absolute),
                                   [7.0, 4.0], rtol=1e-6)
        assert nom.value_scaled == ("['p0']",)

    @pytest.mark.parametrize("bounds", [
        (0.0, None), (None, 10.0), (None, None),
        # A non-zero lower bound is what separates "scaled by p" from
        # "scaled by p - lo", the ``"log"`` rule; with ``lo = 0`` the two
        # agree and a mutant offsetting identity columns by ``lo``
        # survived the audit.  ``p0 = 3.0`` against ``lo = 2.0``: 3, not 1.
        (2.0, None), (-5.0, None),
        # An outward infinity is "no bound", not a width of ``inf``.
        (-float("inf"), float("inf")), (2.0, float("inf")),
    ], ids=["lower_only", "upper_only", "unbounded", "lower_only_nonzero",
            "lower_only_negative", "infinite_both", "infinite_upper"])
    def test_identity_without_a_width_scales_by_the_value_and_is_named(self, bounds):
        fn, params = _linear(5, 2, 10, (3.0, 2.0))
        specs = {"p0": ParamSpec(bounds=bounds), "p1": ParamSpec(bounds=(0.0, 4.0))}
        nom = fim(fn, params, scale="nominal", specs=specs)
        absolute = fim(fn, params, scale=None)
        np.testing.assert_allclose(_column_scales(nom, absolute),
                                   [3.0, 4.0], rtol=1e-6)
        assert nom.value_scaled == ("['p0']",)
        assert nom.zero_scaled == ()

    @pytest.mark.parametrize("bounds", [(0.0, None), (None, 10.0), (None, None)],
                             ids=["lower_only", "upper_only", "unbounded"])
    def test_identity_without_a_width_at_zero_is_a_named_zero(self, bounds):
        fn, params = _linear(5, 2, 10, (0.0, 2.0))
        specs = {"p0": ParamSpec(bounds=bounds), "p1": ParamSpec(bounds=(0.0, 4.0))}
        nom = fim(fn, params, scale="nominal", specs=specs)
        assert nom.value_scaled == ("['p0']",)
        assert nom.zero_scaled == ("['p0']",)
        assert nom.rank == 1

    def test_a_leaf_without_a_spec_entry_is_value_scaled_and_named(self):
        """``_spec_for`` gives a missing entry the default spec, which
        is unbounded.  So an incomplete ``specs`` cannot pass unnoticed:
        every column it failed to reach is named.  (A key that matches
        no parameter is a different thing -- refused, see
        :class:`TestASpecsTreeMustMirrorParams`.)"""
        fn, params = _linear(6, 3, 10, (1.0, 2.0, 3.0))
        nom = fim(fn, params, scale="nominal",
                  specs={"p1": ParamSpec(bounds=(0.0, 4.0))})
        assert nom.value_scaled == ("['p0']", "['p2']")

    def test_nothing_falls_back_to_one(self):
        """``specs={}`` gives no column a width.  The result is then
        ``"relative"`` -- every column value-scaled -- and says so for
        every column; it is never ``scale=None`` in disguise."""
        fn, params = _linear(7, 3, 10, (1.5, 2.5, 0.5))
        nom = fim(fn, params, scale="nominal", specs={})
        _identical(nom, fim(fn, params, scale="relative"))
        assert nom.value_scaled == ("['p0']", "['p1']", "['p2']")
        absolute = fim(fn, params, scale=None)
        assert not np.allclose(np.asarray(nom.fim), np.asarray(absolute.fim))

    def test_a_multi_element_leaf_shares_its_leafs_width(self):
        A = jnp.asarray(np.random.default_rng(8).standard_normal((10, 3)),
                        dtype=jnp.float32)
        params = {"v": jnp.array([0.0, 1.0], dtype=jnp.float32),
                  "w": jnp.float32(2.0)}

        def fn(p):
            return A @ jnp.concatenate([p["v"], p["w"][None]])

        nom = fim(fn, params, scale="nominal",
                  specs={"v": ParamSpec(bounds=(-2.0, 2.0)),
                         "w": ParamSpec(bounds=(0.0, 3.0))})
        np.testing.assert_allclose(
            _column_scales(nom, fim(fn, params, scale=None)),
            [4.0, 4.0, 3.0], rtol=1e-6)
        assert nom.param_names == ("['v'][0]", "['v'][1]", "['w']")
        assert nom.zero_scaled == ()


# ---------------------------------------------------------------------------
# 3. Column scaling is the only difference
# ---------------------------------------------------------------------------


class TestColumnScalingIsTheOnlyDifference:

    def test_unit_widths_reproduce_scale_none_bit_for_bit(self):
        fn, params = _linear(9, 3, 40, (0.0, 2.0, -3.0))
        ones = {"p0": ParamSpec(bounds=(0.0, 1.0)),
                "p1": ParamSpec(bounds=(5.0, 6.0)),
                "p2": ParamSpec(bounds=(-0.5, 0.5))}
        _identical(fim(fn, params, scale="nominal", specs=ones),
                   fim(fn, params, scale=None))
        core_n = fim_core(fn, params, scale="nominal", specs=ones)
        core_a = fim_core(fn, params, scale=None)
        _identical(core_n, core_a)
        assert np.array_equal(np.asarray(core_n.zero_scaled),
                              np.asarray(core_a.zero_scaled))

    def test_widths_equal_to_the_values_reproduce_relative_bit_for_bit(self):
        fn, params = _linear(10, 3, 40, (1.5, 0.25, 4.0))
        same = {k: ParamSpec(bounds=(0.0, float(v))) for k, v in params.items()}
        _identical(fim(fn, params, scale="nominal", specs=same),
                   fim(fn, params, scale="relative"))
        _identical(fim_core(fn, params, scale="nominal", specs=same),
                   fim_core(fn, params, scale="relative"))

    def test_jitted_and_eager_jacobians_agree_bit_for_bit(self):
        """``fim`` jits the Jacobian when ``residual_fn`` is hashable
        and builds it eagerly when it is not.  The nominal multiplier
        is a constant either way and must give the same bits."""
        fn, params = _linear(11, 3, 40, (0.0, 2.0, -3.0))
        specs = {"p0": ParamSpec(bounds=(-1.0, 1.0)),
                 "p1": ParamSpec(bounds=(0.0, None), transform="log"),
                 "p2": ParamSpec(bounds=(-4.0, 4.0))}

        class Unhashable:
            __hash__ = None

            def __call__(self, p):
                return fn(p)

        jitted = fim(fn, params, scale="nominal", specs=specs)
        eager = fim(Unhashable(), params, scale="nominal", specs=specs)
        _identical(jitted, eager)
        assert jitted.zero_scaled == eager.zero_scaled == ()
        assert jitted.value_scaled == eager.value_scaled == ("['p1']",)

    def test_different_widths_do_not_share_a_compiled_jacobian(self):
        """``specs`` is part of the cache key.  Two specs with the same
        shape and different widths must not return each other's
        constants."""
        fn, params = _linear(12, 2, 20, (1.0, 1.0))
        a = fim(fn, params, scale="nominal",
                specs={"p0": ParamSpec(bounds=(0.0, 1.0)), "p1": ParamSpec(bounds=(0.0, 1.0))})
        b = fim(fn, params, scale="nominal",
                specs={"p0": ParamSpec(bounds=(0.0, 4.0)), "p1": ParamSpec(bounds=(0.0, 1.0))})
        s = _column_scales(b, a)
        np.testing.assert_allclose(s, [4.0, 1.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Composition with noise_std and mask
# ---------------------------------------------------------------------------


class TestCompositionWithNoiseAndMask:
    """``J[i, j] = (1 / sigma_i) * (d r_i / d theta_j) * s_j``, with
    ``mask`` selecting ``j`` first."""

    def test_the_scale_multiplies_columns_not_rows_on_a_square_jacobian(self):
        """On a rectangular ``J`` a row-side multiplication is a shape
        error; on a *square* one it is a wrong answer with the right
        shape, so that is the case to pin.  ``J = [[1, 1], [1, -1]]``,
        widths ``(2, 5)``: column scaling gives ``diag(F) = (8, 50)``,
        row scaling gives ``(29, 29)``."""
        residual, params = _pair_at_zero()
        nom = fim(residual, params, scale="nominal",
                  specs={"a": ParamSpec(bounds=(0.0, 2.0)),
                         "b": ParamSpec(bounds=(-1.0, 4.0))})
        np.testing.assert_allclose(np.diag(np.asarray(nom.fim)), [8.0, 50.0])
        np.testing.assert_allclose(
            _column_scales(nom, fim(residual, params, scale=None)), [2.0, 5.0])

    def test_per_row_noise_then_per_column_scale(self):
        """With a per-row sigma the two axes are distinguishable:
        ``F_nom(sigma) = S (J^T Sigma^-1 J) S`` and nothing else."""
        rng = np.random.default_rng(13)
        A64 = rng.standard_normal((6, 2))
        A = jnp.asarray(A64, dtype=jnp.float32)
        keys = ("p0", "p1")
        params = {"p0": jnp.float32(0.0), "p1": jnp.float32(1.0)}
        fn = lambda p: A @ jnp.stack([p[k] for k in keys])
        sigma = np.geomspace(0.5, 4.0, 6)
        widths = np.array([3.0, 0.5])
        specs = {"p0": ParamSpec(bounds=(-1.5, 1.5)), "p1": ParamSpec(bounds=(0.0, 0.5))}
        nom = fim(fn, params, scale="nominal", specs=specs,
                  noise_std=jnp.asarray(sigma, dtype=jnp.float32))
        Jw = A64 / sigma[:, None] * widths[None, :]
        np.testing.assert_allclose(np.asarray(nom.fim, dtype=np.float64),
                                   Jw.T @ Jw, rtol=1e-5)
        assert nom.rank == 2

    def test_a_masked_column_keeps_its_own_spec(self):
        """The nominal record is selected by the same indices as the
        columns, so the masked matrix is the submatrix of the unmasked
        one -- the contract ``mask`` already has under the other
        scales."""
        fn, params = _linear(14, 3, 30, (0.0, 2.0, 3.0))
        specs = {"p0": ParamSpec(bounds=(-1.0, 1.0)),
                 "p1": ParamSpec(bounds=(0.0, None)),        # value-scaled
                 "p2": ParamSpec(bounds=(0.0, 7.0))}
        full = fim(fn, params, scale="nominal", specs=specs)
        mask = {"p0": False, "p1": True, "p2": True}
        masked = fim(fn, params, scale="nominal", specs=specs, mask=mask)
        assert masked.param_names == ("['p1']", "['p2']")
        # The eager Gram product accumulates a 2x30 and a 3x30 ``J`` in
        # different orders, so the submatrix identity holds to an ulp,
        # as it does under the other scales; the *scales* are exact.
        np.testing.assert_allclose(np.asarray(masked.fim),
                                   np.asarray(full.fim)[1:, 1:], rtol=1e-6)
        assert full.value_scaled == ("['p1']",)
        assert masked.value_scaled == ("['p1']",)
        only_p2 = fim(fn, params, scale="nominal", specs=specs,
                      mask={"p0": False, "p1": False, "p2": True})
        assert only_p2.value_scaled == ()
        np.testing.assert_allclose(np.asarray(only_p2.fim),
                                   np.asarray(full.fim)[2:, 2:], rtol=1e-6)


# ---------------------------------------------------------------------------
# 5. Refusals
# ---------------------------------------------------------------------------


class TestRefusals:

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    def test_specs_under_another_scale_is_refused(self, runner):
        residual, params = _pair_at_zero()
        for scale in ("relative", None):
            with pytest.raises(ValueError, match="specs is read only under"):
                runner(residual, params, scale=scale, specs={})

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    def test_nominal_without_specs_is_refused(self, runner):
        residual, params = _pair_at_zero()
        with pytest.raises(ValueError, match="needs specs="):
            runner(residual, params, scale="nominal")

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    def test_an_unknown_scale_names_all_three(self, runner):
        residual, params = _pair_at_zero()
        with pytest.raises(ValueError, match="scale must be 'relative', 'nominal' or None"):
            runner(residual, params, scale="absolute")

    @pytest.mark.parametrize("hi", [1e-300, 1e-39, 1e39, 1e20],
                             ids=["underflows", "subnormal_square_zero",
                                  "overflows", "square_overflows"])
    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    def test_a_width_that_is_not_a_float32_scale_is_refused_by_name(
            self, runner, hi):
        """``ParamSpec`` accepts ``(0.0, 1e-300)``: ``lo < hi`` holds in
        float64.  At float32 that width is ``0.0`` and the column it
        scales is the silent zero this mode exists to remove; ``1e20``
        survives the cast and overflows ``F``.  Refused, naming the
        column and the width, in both paths."""
        residual, params = _pair_at_zero()
        specs = {"a": ParamSpec(bounds=(0.0, hi)), "b": ParamSpec(bounds=(-1.0, 1.0))}
        with pytest.raises(ValueError, match=r"width of \['a'\]'s bounds") as info:
            runner(residual, params, scale="nominal", specs=specs)
        assert repr(hi) in str(info.value) or f"{hi:g}" in str(info.value)

    def test_the_guard_is_applied_at_the_parameters_precision(self):
        """The same ``1e-39`` width that is refused for float32
        parameters is a perfectly good float64 one -- the guard reads
        the dtype rather than assuming it.  x64 is process-global, so
        the float64 half is asserted on the helper with a float64
        ``theta0`` rather than by flipping the flag mid-suite."""
        from maddening.sysid import _nominal_column_vector
        nominal = (("['p0']", 1e-39, 0.0), ("['p1']", 1.0, 0.0))
        with pytest.raises(ValueError, match="float32"):
            _nominal_column_vector(nominal, np.zeros(2, dtype=np.float32))
        col = _nominal_column_vector(nominal, np.zeros(2, dtype=np.float64))
        assert np.all(np.asarray(col, dtype=np.float64) > 0.0)


class TestASpecsTreeMustMirrorParams:
    """A ``specs`` that reaches nothing is refused, not read as ``{}``.

    ``_spec_for`` answers the default spec for every leaf it cannot
    reach, so before this check a ``gm.param_specs()`` handed to a
    sub-tree ``params``, ``ParamSpec.to_dict()`` entries, a list, a
    bare ``ParamSpec`` or a string all produced a report bit-identical
    to ``scale="relative"`` -- indistinguishable from the documented
    ``specs={}``.  Each is now a ``ValueError`` naming the key path.
    A leaf *without* an entry is still the default spec, so ``{}`` and
    an incomplete dict stay accepted (the test above pins that).
    """

    @pytest.fixture()
    def three(self):
        fn, params = _linear(9, 3, 12, (1.0, 2.0, 3.0))
        good = {k: ParamSpec(bounds=(0.0, 5.0)) for k in params}
        return fn, params, good

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    @pytest.mark.parametrize("bad", [
        pytest.param("gm.param_specs()", id="string"),
        pytest.param(ParamSpec(bounds=(0.0, 1.0)), id="bare_paramspec"),
        pytest.param([ParamSpec(bounds=(0.0, 1.0))] * 3, id="list"),
    ])
    def test_a_specs_that_is_not_a_dict_is_refused(self, runner, three, bad):
        fn, params, _ = three
        with pytest.raises(ValueError, match="specs must be a dict of ParamSpec"):
            runner(fn, params, scale="nominal", specs=bad)

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    @pytest.mark.parametrize("stray", [
        pytest.param(ParamSpec(bounds=(0.0, 5.0)), id="width"),
        pytest.param(ParamSpec(bounds=(2.0, None), transform="log"), id="log_offset"),
    ])
    def test_a_key_that_matches_no_parameter_is_refused_by_path(
            self, runner, three, stray):
        """A misspelt name is exactly the case that must be loud: the
        column it meant to reach would be value-scaled and named, but
        the entry itself would vanish without a word.  Loud whenever the
        stray spec would have changed a column -- a width, or a ``log``
        offset from a non-zero lower bound."""
        fn, params, good = three
        with pytest.raises(ValueError, match=r"specs\['p9'\] matches no parameter"):
            runner(fn, params, scale="nominal", specs={**good, "p9": stray})

    @pytest.mark.parametrize("stray", [
        pytest.param(ParamSpec(), id="default"),
        pytest.param(ParamSpec(trainable=False, description="grid geometry"),
                     id="frozen_no_bounds"),
        pytest.param(ParamSpec(bounds=(0.0, None), transform="log"), id="log_from_zero"),
        pytest.param(ParamSpec(bounds=(-3.0, None)), id="one_sided_identity"),
    ])
    def test_a_stray_key_that_could_change_no_column_is_accepted(
            self, three, stray):
        """``gm.param_specs()`` declares specs for constants that are not
        leaves of ``gm.params`` (a uniform ``HeatNode``'s
        ``grid_points=None``; any constant spelled as a Python ``int``),
        so refusing every stray key refused the documented
        ``specs=gm.param_specs()`` for those graphs.  A stray spec whose
        column record is the default's cannot have changed the report
        whichever leaf it was meant for, so it is accepted -- and the
        report is bit-identical to the one without it, which is the
        whole licence for accepting it."""
        fn, params, good = three
        with_stray = fim(fn, params, scale="nominal", specs={**good, "p9": stray})
        without = fim(fn, params, scale="nominal", specs=good)
        _identical(with_stray, without)
        assert with_stray.value_scaled == without.value_scaled
        assert with_stray.zero_scaled == without.zero_scaled

    def test_the_graphs_full_specs_against_a_sub_tree_are_refused(self):
        """The shape this file's own fixture avoids by filtering:
        ``gm.param_specs()`` is keyed ``nodes -> name -> key`` and a
        two-parameter sub-tree has no ``nodes``.  Before the check this
        was accepted and reported ``rank 1`` where the correctly nested
        specs give ``rank 2``."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                     mass=1.0, rest_length=1.0,
                                     initial_position=0.5))
        gm.compile()
        gm.set_param_spec("s", "initial_velocity",
                          ParamSpec(trainable=False, bounds=(-1.0, 1.0)))
        specs = gm.param_specs()
        sub = {k: gm.params["nodes"]["s"][k]
               for k in ("stiffness", "initial_velocity")}
        A = jnp.asarray(np.random.default_rng(0).standard_normal((6, 2)),
                        dtype=jnp.float32)

        def fn(p):
            return A @ jnp.stack([p["stiffness"], p["initial_velocity"]])

        with pytest.raises(ValueError, match=r"specs\['nodes'\] matches no parameter"):
            fim(fn, sub, scale="nominal", specs=specs)
        right = fim(fn, sub, scale="nominal",
                    specs={k: specs["nodes"]["s"][k] for k in sub})
        assert right.rank == 2
        assert right.value_scaled == ("['stiffness']",)
        # The node's whole spec dict is a superset of the sub-tree.  Its
        # unreached entries -- damping, mass, rest_length,
        # initial_position -- are one-sided or unbounded, so none of them
        # could change a column and the superset gives exactly the
        # filtered report.  Give one of them a width and it could, and
        # the superset is refused by the path of that entry.
        superset = fim(fn, sub, scale="nominal", specs=specs["nodes"]["s"])
        _identical(superset, right)
        gm.set_param_spec("s", "mass", ParamSpec(bounds=(0.5, 2.0)))
        with pytest.raises(ValueError, match=r"specs\['mass'\] matches no parameter"):
            fim(fn, sub, scale="nominal", specs=gm.param_specs()["nodes"]["s"])
        # And the documented full-tree call is untouched, ``mappings: {}``
        # mirroring ``mappings: {}`` included.
        whole = fim(lambda p: fn(p["nodes"]["s"]), gm.params,
                    scale="nominal", specs=specs)
        assert "['nodes']['s']['initial_velocity']" not in whole.value_scaled

    @pytest.mark.parametrize("runner", [fim, fim_core], ids=["fim", "fim_core"])
    def test_a_to_dict_entry_is_refused_with_the_from_dict_hint(self, runner, three):
        fn, params, good = three
        encoded = {k: v.to_dict() for k, v in good.items()}
        with pytest.raises(ValueError, match=r"specs\['p0'\] is a dict but params\['p0'\] is a leaf") as info:
            runner(fn, params, scale="nominal", specs=encoded)
        assert "ParamSpec.from_dict" in str(info.value)

    def test_a_paramspec_above_a_dict_level_is_refused(self, three):
        fn, params, _ = three
        nested = {"outer": params}
        with pytest.raises(ValueError, match=r"specs\['outer'\] is a ParamSpec but params\['outer'\] is a dict"):
            fim(lambda p: fn(p["outer"]), nested, scale="nominal",
                specs={"outer": ParamSpec(bounds=(0.0, 1.0))})

    @pytest.mark.parametrize("entry", [None, (0.0, 5.0), 3.0, "log"],
                             ids=["none", "bounds_tuple", "float", "string"])
    def test_an_entry_that_is_neither_paramspec_nor_dict_is_refused(self, three, entry):
        fn, params, good = three
        with pytest.raises(ValueError, match=r"specs\['p1'\]"):
            fim(fn, params, scale="nominal", specs={**good, "p1": entry})

    def test_a_refused_specs_is_refused_before_anything_is_traced(self, three):
        """The check is structural and runs on the host before the
        Jacobian is compiled: a residual that would fail to trace never
        gets the chance to, so the error a user sees is about ``specs``."""
        _, params, good = three

        def never(p):
            raise AssertionError("residual_fn was traced")

        with pytest.raises(ValueError, match="matches no parameter"):
            fim(never, params, scale="nominal",
                specs={**good, "typo": ParamSpec(bounds=(0.0, 1.0))})

    def test_incomplete_and_empty_specs_remain_the_documented_default(self, three):
        """Refusing what mirrors nothing must not refuse what is merely
        incomplete: a missing entry is the default spec, named."""
        fn, params, good = three
        partial = fim(fn, params, scale="nominal", specs={"p0": good["p0"]})
        assert partial.value_scaled == ("['p1']", "['p2']")
        empty = fim(fn, params, scale="nominal", specs={})
        assert empty.value_scaled == ("['p0']", "['p1']", "['p2']")


class TestTheGraphsOwnSpecsAreAccepted:
    """``specs=gm.param_specs()`` is the documented spelling, and it was
    refused for two ordinary kinds of graph: ``param_specs()`` declares a
    spec for every constant a node knows, ``params_pytree()`` leaves out
    the structural ones, and the strict mirror check called every such
    entry a misspelt key.  Those entries have no width, so they could
    not change a column; they are accepted now, and the report must be
    exactly the one the pruned specs give."""

    @staticmethod
    def _pruned(specs, params):
        return {sec: {owner: {k: v for k, v in entries.items()
                              if k in params[sec].get(owner, {})}
                      for owner, entries in specs[sec].items()}
                for sec in specs}

    def _assert_accepted_as_pruned(self, gm, residual):
        specs = gm.param_specs()
        extra = set(specs["nodes"]["n"]) - set(gm.params["nodes"]["n"])
        assert extra, "fixture no longer has a declared-but-absent constant"
        whole = fim(residual, gm.params, scale="nominal", specs=specs)
        pruned = fim(residual, gm.params, scale="nominal",
                     specs=self._pruned(specs, gm.params))
        _identical(whole, pruned)
        assert whole.value_scaled == pruned.value_scaled

    def test_a_uniform_heat_node_graph(self):
        """``grid_points=None`` on a uniform grid: declared, not a leaf."""
        from maddening.nodes.heat import HeatNode
        gm = GraphManager()
        gm.add_node(HeatNode("n", 1e-4, n_cells=4))
        gm.compile()

        def residual(p):
            h = p["nodes"]["n"]
            return jnp.stack([h["thermal_diffusivity"] * 1e3, h["length"]])

        self._assert_accepted_as_pruned(gm, residual)

    def test_a_node_built_with_an_int_constant(self):
        """``stiffness=100`` is structural to ``params_pytree()`` (that it
        drops out of ``gm.params`` at all is a separate defect); its spec
        ``(0, None), "log"`` has no width, so it cannot change a column."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode("n", 0.01, stiffness=100, damping=1.0,
                                     mass=2.0, rest_length=2.0,
                                     initial_position=0))
        gm.compile()

        def residual(p):
            s = p["nodes"]["n"]
            return jnp.stack([s["damping"], s["mass"] * 2.0, s["rest_length"]])

        self._assert_accepted_as_pruned(gm, residual)


class TestAListValuedLeafIsReachedByItsSpec:
    """``jax.tree_util`` yields a ``SequenceKey`` for a list/tuple
    position, which has ``.idx`` and no ``.key``; ``_spec_for`` used to
    stop there and hand the leaf the default spec.  The array spelling
    of the same vector was already correct, so the two must agree:
    one spec keyed at the list covers every position, exactly as it
    covers every element of an array leaf, and a list of specs is read
    by position.  ``params_pytree()`` turns lists into array leaves, so
    only a hand-built ``params`` reaches this -- hence the hand-built
    fixtures.
    """

    def _fixture(self):
        A = jnp.asarray(np.random.default_rng(11).standard_normal((8, 2)),
                        dtype=jnp.float32)
        as_list = {"v": [jnp.float32(0.0), jnp.float32(1.0)]}
        as_array = {"v": jnp.array([0.0, 1.0], dtype=jnp.float32)}
        return (A, as_list, lambda p: A @ jnp.stack(p["v"]),
                as_array, lambda p: A @ p["v"])

    def test_one_spec_over_a_list_matches_the_array_spelling(self):
        A, as_list, fn_l, as_array, fn_a = self._fixture()
        specs = {"v": ParamSpec(trainable=False, bounds=(-1.0, 1.0),
                                transform="logit")}
        listed = fim(fn_l, as_list, scale="nominal", specs=specs)
        arrayed = fim(fn_a, as_array, scale="nominal", specs=specs)
        _identical(listed, arrayed)
        assert listed.param_names == ("['v'][0]", "['v'][1]")
        assert listed.value_scaled == ()
        assert listed.zero_scaled == ()
        assert listed.rank == 2

    def test_a_list_of_specs_is_read_by_position(self):
        A, as_list, fn_l, _, _ = self._fixture()
        per = {"v": [ParamSpec(bounds=(-1.0, 1.0)), ParamSpec(bounds=(0.0, 4.0))]}
        nom = fim(fn_l, as_list, scale="nominal", specs=per)
        np.testing.assert_allclose(
            _column_scales(nom, fim(fn_l, as_list, scale=None)),
            [2.0, 4.0], rtol=1e-6)
        assert nom.value_scaled == ()

    def test_a_tuple_of_leaves_is_covered_too(self):
        A, _, _, _, _ = self._fixture()
        params = {"v": (jnp.float32(0.0), jnp.float32(1.0))}
        nom = fim(lambda p: A @ jnp.stack(p["v"]), params, scale="nominal",
                  specs={"v": ParamSpec(bounds=(-2.0, 2.0))})
        assert nom.value_scaled == ()
        assert nom.zero_scaled == ()

    def test_a_list_of_specs_of_the_wrong_length_is_refused(self):
        A, as_list, fn_l, _, _ = self._fixture()
        with pytest.raises(ValueError, match=r"specs\['v'\] has 1 entries but params\['v'\] has 2 positions"):
            fim(fn_l, as_list, scale="nominal", specs={"v": [ParamSpec()]})


# ---------------------------------------------------------------------------
# fim_core carries the same verdicts, as metadata
# ---------------------------------------------------------------------------


class TestCore:

    def test_value_scaled_is_static_metadata_and_survives_jit(self):
        fn, params = _linear(15, 3, 30, (0.0, 2.0, 3.0))
        specs = {"p0": ParamSpec(bounds=(-1.0, 1.0)),
                 "p1": ParamSpec(bounds=(0.0, None), transform="log"),
                 "p2": ParamSpec(bounds=(0.0, 7.0))}
        core = jax.jit(functools.partial(fim_core, fn, scale="nominal",
                                         specs=specs))(params)
        assert isinstance(core, FIMCore)
        assert core.value_scaled == ("['p1']",)
        assert isinstance(core.value_scaled, tuple)
        leaves, treedef = jax.tree.flatten(core)
        assert len(leaves) == 10          # unchanged: value_scaled is meta
        assert jax.tree.unflatten(treedef, leaves).value_scaled == ("['p1']",)

    def test_core_and_report_agree_on_the_zero_parameter(
            self, spring_with_zero_velocity):
        residual, sub, specs = spring_with_zero_velocity
        bounded = dict(specs, initial_velocity=ParamSpec(
            trainable=False, bounds=(-1.0, 1.0)))
        report = fim(residual, sub, scale="nominal", specs=bounded)
        core = jax.jit(functools.partial(fim_core, residual, scale="nominal",
                                         specs=bounded))(sub)
        assert int(core.rank) == report.rank == 2
        assert bool(core.finite)
        assert not bool(np.any(np.asarray(core.zero_scaled)))
        assert core.value_scaled == report.value_scaled == ("['stiffness']",)
        np.testing.assert_allclose(np.asarray(core.crb), np.asarray(report.crb),
                                   rtol=1e-4)
