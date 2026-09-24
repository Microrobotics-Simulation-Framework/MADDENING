"""An integer or boolean params leaf is named, never a silent zero.

There is no derivative with respect to an integer.  ``ravel_pytree``
used to promote such a leaf into the float vector ``fim`` differentiates
and cast it back on the way out; JAX's derivative through that cast is
identically zero, so the leaf became a zero column and read as
unidentifiable whatever the data said -- an integer ``matrix_mapping``
``H`` reported rank 0 of 16 -- with nothing in the report to say why.
``fit`` on the same leaf made trainable handed it back bit-for-bit
unchanged, with a finite loss and no word about the dtype.

Now: ``fim``/``fim_core`` with ``mask=None`` leave such a leaf out and
name it in ``integer_excluded``; a ``mask`` that selects one is refused;
the fitters refuse a trainable set that holds one.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode
from maddening.sysid import fim, fim_core, fit, fit_lm, fit_multiple_shooting

_T = jnp.linspace(0.0, 1.0, 10, dtype=jnp.float32)


def _mixed():
    """``a`` is a float parameter; ``H`` an integer one the residual reads."""
    params = {"a": jnp.float32(1.5), "H": jnp.array([2, 3], jnp.int32)}

    def residual(p):
        return p["a"] * _T + p["H"][0] * _T ** 2 + p["H"][1] * _T ** 3

    return residual, params


class TestFimLeavesAnIntegerLeafOut:

    @pytest.mark.parametrize("scale", ["relative", None])
    def test_it_is_named_and_not_a_column(self, scale):
        residual, params = _mixed()
        report = fim(residual, params, scale=scale)
        assert report.param_names == ("['a']",)
        assert report.integer_excluded == ("['H']",)
        assert report.rank == 1
        assert np.isfinite(np.asarray(report.crb)).all()

    def test_the_report_is_the_one_for_the_float_leaves_alone(self):
        """Leaving ``H`` out must be exactly "``H`` is a constant": the
        report equals the one over ``a`` with ``H`` closed over, bit for
        bit, so the exclusion moves no other number."""
        residual, params = _mixed()
        report = fim(residual, params, scale=None)
        H = params["H"]
        alone = fim(lambda p: residual({"a": p["a"], "H": H}), {"a": params["a"]},
                    scale=None)
        for field in ("fim", "eigvals", "eigvecs", "crb"):
            assert np.array_equal(np.asarray(getattr(report, field)),
                                  np.asarray(getattr(alone, field))), field
        assert report.rank == alone.rank

    def test_a_boolean_leaf_is_left_out_too(self):
        params = {"a": jnp.float32(2.0), "on": jnp.array(True)}
        report = fim(lambda p: p["a"] * _T * p["on"], params, scale=None)
        assert report.param_names == ("['a']",)
        assert report.integer_excluded == ("['on']",)

    def test_an_integer_reaches_residual_fn_exactly(self):
        """The leaf is not routed through the float vector at all.  In
        float32, ``2**24 + 1`` rounds to ``2**24``; had it been promoted
        and cast back, the Jacobian below would be ``16 t``, not
        ``17 t``."""
        n = jnp.int32(2 ** 24 + 1)
        params = {"a": jnp.float32(1.0), "n": n}

        def residual(p):
            return p["a"] * (p["n"] - 16_777_200).astype(jnp.float32) * _T

        report = fim(residual, params, scale=None)
        want = float(np.sum((17.0 * np.asarray(_T, np.float64)) ** 2))
        np.testing.assert_allclose(float(np.asarray(report.fim)[0, 0]), want,
                                   rtol=1e-6)

    def test_an_explicit_mask_selecting_it_is_refused(self):
        residual, params = _mixed()
        for runner in (fim, fim_core):
            with pytest.raises(ValueError, match=r"mask selects 1 leaf/leaves of integer or boolean dtype \(int32\).*\['H'\]"):
                runner(residual, params, mask={"a": True, "H": True})

    def test_an_explicit_mask_leaving_it_out_names_nothing(self):
        """``integer_excluded`` names what the *dtype* excluded; a leaf
        the mask left out was not a candidate."""
        residual, params = _mixed()
        report = fim(residual, params, mask={"a": True, "H": False})
        assert report.param_names == ("['a']",)
        assert report.integer_excluded == ()

    def test_params_with_no_float_leaf_is_refused(self):
        params = {"H": jnp.array([2, 3], jnp.int32)}
        with pytest.raises(ValueError, match="no floating-point leaf"):
            fim(lambda p: p["H"].astype(jnp.float32) * 1.0, params)


class TestFimCoreCarriesTheSameName:

    def test_integer_excluded_is_static_metadata_that_survives_jit(self):
        residual, params = _mixed()
        core = jax.jit(functools.partial(fim_core, residual, scale=None))(params)
        assert core.integer_excluded == ("['H']",)
        assert core.param_names == ("['a']",)
        leaves, treedef = jax.tree.flatten(core)
        assert len(leaves) == 10          # a name, not a traced leaf
        assert jax.tree.unflatten(treedef, leaves).integer_excluded == ("['H']",)

    def test_a_nominal_scale_skips_it_too(self):
        """The nominal record is built per column, so it has to skip
        the leaf the columns skip -- or every width lands one column
        off."""
        residual, params = _mixed()
        specs = {"a": ParamSpec(bounds=(0.0, 4.0)), "H": ParamSpec(bounds=(0.0, 9.0))}
        report = fim(residual, params, scale="nominal", specs=specs)
        absolute = fim(residual, params, scale=None)
        assert report.value_scaled == ()
        np.testing.assert_allclose(
            np.asarray(report.fim)[0, 0], 16.0 * np.asarray(absolute.fim)[0, 0],
            rtol=1e-6)


def _heat_pair_with_int_mapping():
    gm = GraphManager()
    gm.add_node(HeatNode("a", 1e-4, n_cells=4, initial_temperature=1.0))
    gm.add_node(HeatNode("b", 1e-4, n_cells=4, initial_temperature=0.0))
    H = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=matrix_mapping(H))
    gm.compile()
    key = next(iter(gm.params["mappings"]))
    assert gm.params["mappings"][key]["H"].dtype == jnp.int32
    return gm, key


class TestTheFittersRefuseATrainableInteger:

    def test_fit_refuses_it_when_the_spec_makes_it_trainable(self):
        gm, key = _heat_pair_with_int_mapping()
        gm.set_param_spec(key, "H", ParamSpec())
        with pytest.raises(ValueError, match=r"(?s)integer or boolean dtype.*weight 'H'.*int32"):
            fit(gm, lambda p: jnp.sum(p["nodes"]["b"]["length"] ** 2), n_iter=2)

    def test_fit_lm_and_multiple_shooting_refuse_it_too(self):
        gm, key = _heat_pair_with_int_mapping()
        gm.set_param_spec(key, "H", ParamSpec())
        with pytest.raises(ValueError, match="integer or boolean dtype"):
            fit_lm(gm, lambda p: p["nodes"]["b"]["length"][None], n_iter=2)
        init = {n: gm.get_node_state(n) for n in gm.node_names}
        obs = jax.tree.map(lambda x: jnp.stack([x, x, x]), init)
        with pytest.raises(ValueError, match="integer or boolean dtype"):
            fit_multiple_shooting(gm, obs, obs_fn=lambda h: h["b"]["temperature"],
                                  window=1, n_iter=2)

    def test_an_explicit_mask_selecting_it_is_refused(self):
        gm, key = _heat_pair_with_int_mapping()
        gm.set_param_spec(key, "H", ParamSpec())
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["mappings"][key]["H"] = True
        with pytest.raises(ValueError, match="integer or boolean dtype"):
            fit(gm, lambda p: jnp.sum(p["nodes"]["b"]["length"] ** 2),
                mask=mask, n_iter=2)

    def test_a_frozen_integer_leaf_does_not_stop_a_fit(self):
        """The default spec for a mapping weight is frozen, and a fit of
        the float constants beside it runs exactly as before -- the
        refusal is for a trainable integer, not for one being present."""
        gm, key = _heat_pair_with_int_mapping()
        res = fit(gm, lambda p: jnp.sum((p["nodes"]["b"]["length"] - 2.0) ** 2),
                  n_iter=3, lr=0.05)
        assert res.n_iter == 3
        assert np.array_equal(np.asarray(res.params["mappings"][key]["H"]),
                              np.asarray(gm.params["mappings"][key]["H"]))
