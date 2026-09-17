"""``maddening.core.params``: ParamSpec validation and the pure tree maps."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.params import (
    ParamSpec,
    check_bounds,
    constrain,
    trainable_mask,
    unconstrain,
)
from tests.conftest import EXAMPLES_CHEAP

SPECS = {
    "nodes": {
        "s": {
            "stiffness": ParamSpec(bounds=(0.0, None), transform="log"),
            "damping": ParamSpec(bounds=(0.0, None)),            # clip
            "mass": ParamSpec(trainable=False, bounds=(0.0, None), transform="log"),
            "elasticity": ParamSpec(bounds=(0.0, 1.0), transform="logit"),
            "offset": ParamSpec(),                              # free
            # "initial_position": no entry -> default spec
        },
    },
    "mappings": {},
}


def _params(k, c, m, e, off, x0):
    f = lambda v: jnp.asarray(v, jnp.float32)  # noqa: E731
    return {
        "nodes": {"s": {"stiffness": f(k), "damping": f(c), "mass": f(m),
                        "elasticity": f(e), "offset": f(off),
                        "initial_position": f(x0)}},
        "mappings": {},
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(transform="exp"),
    dict(bounds=(1.0, 1.0)),
    dict(bounds=(2.0, 1.0)),
    dict(transform="log", bounds=(0.0, 1.0)),
    dict(transform="logit", bounds=(0.0, None)),
    dict(transform="logit"),
])
def test_invalid_specs_rejected(kw):
    with pytest.raises(ValueError):
        ParamSpec(**kw)


def test_check_bounds_names_offending_leaf():
    p = _params(10.0, -1.0, 1.0, 0.5, 0.0, 0.0)
    with pytest.raises(ValueError, match=r"\['nodes'\]\['s'\]\['damping'\].*below"):
        check_bounds(p, SPECS)
    # log / logit bounds are strict; identity bounds are inclusive.
    check_bounds(_params(10.0, 0.0, 1.0, 0.5, 0.0, 0.0), SPECS)
    with pytest.raises(ValueError, match="stiffness"):
        check_bounds(_params(0.0, 0.0, 1.0, 0.5, 0.0, 0.0), SPECS)
    with pytest.raises(ValueError, match="elasticity"):
        check_bounds(_params(1.0, 0.0, 1.0, 1.0, 0.0, 0.0), SPECS)


def test_mask_mirrors_params_and_defaults_to_trainable():
    m = trainable_mask(_params(1, 1, 1, 0.5, 0, 0), SPECS)
    assert jax.tree.structure(m) == jax.tree.structure(_params(1, 1, 1, 0.5, 0, 0))
    s = m["nodes"]["s"]
    assert s["mass"] is False
    assert all(s[k] is True for k in ("stiffness", "damping", "elasticity",
                                      "offset", "initial_position"))


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

# Draw float32-representable values with no float32 subnormals: the
# leaves are cast to float32, XLA's CPU backend may flush a denormal
# (1e-45) to zero inside ``clip``, and the identity leaves are asserted
# bit-identical.  Bounds are exact binary fractions (Hypothesis refuses a
# ``width=32`` bound that float32 cannot represent).
finite = dict(allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
finite64 = dict(allow_nan=False, allow_infinity=False)


@given(
    k=st.floats(2 ** -10, 1e4, **finite), c=st.floats(0.0, 1e3, **finite),
    m=st.floats(2 ** -10, 1e3, **finite), e=st.floats(2 ** -6, 63 / 64, **finite),
    off=st.floats(-1e3, 1e3, **finite), x0=st.floats(-1e3, 1e3, **finite),
)
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_constrain_inverts_unconstrain_inside_bounds(k, c, m, e, off, x0):
    p = _params(k, c, m, e, off, x0)
    u = unconstrain(p, SPECS)
    back = constrain(u, SPECS)
    for key in p["nodes"]["s"]:
        a, b = float(p["nodes"]["s"][key]), float(back["nodes"]["s"][key])
        # logit/log round-trip through float32 exp/log costs a few ulps.
        assert np.isclose(a, b, rtol=1e-5, atol=1e-6), (key, a, b)
    # Non-trainable and identity leaves are bit-identical, not merely close.
    for key in ("mass", "offset", "initial_position", "damping"):
        assert float(u["nodes"]["s"][key]) == float(p["nodes"]["s"][key])
        assert float(back["nodes"]["s"][key]) == float(p["nodes"]["s"][key])


@given(
    u=st.lists(st.floats(-30.0, 30.0, **finite64), min_size=6, max_size=6),
)
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_constrain_lands_inside_bounds_for_any_coordinates(u):
    raw = _params(*u)                       # interpret as unconstrained coords
    p = constrain(raw, SPECS)
    s = p["nodes"]["s"]
    assert float(s["stiffness"]) > 0.0
    assert 0.0 <= float(s["damping"])
    assert 0.0 < float(s["elasticity"]) < 1.0
    # Non-trainable ``mass`` is passed through, not constrained: an
    # optimiser never writes it, so an out-of-range value there is the
    # caller's, and check_bounds must still report it.
    assert float(s["mass"]) == float(raw["nodes"]["s"]["mass"])
    p["nodes"]["s"]["mass"] = jnp.asarray(1.0, jnp.float32)
    check_bounds(p, SPECS)                  # every trainable leaf inside


@given(u=st.floats(-1e4, 1e4, **finite64))
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_unconstrain_of_constrain_is_finite_for_any_coordinate(u):
    """float32 ``exp``/``sigmoid`` saturate for |u| beyond ~17-87; the
    transforms clamp to the representable interior so the inverse map
    never returns ±inf (a saturated Adam step must stay recoverable)."""
    for spec in (ParamSpec(bounds=(0.0, None), transform="log"),
                 ParamSpec(bounds=(-2.0, 3.0), transform="logit")):
        p = spec.to_constrained(jnp.asarray(u, jnp.float32))
        spec.check(p)
        back = spec.to_unconstrained(p)
        assert bool(jnp.isfinite(back)), (spec, u, p, back)


def test_maps_are_jittable_and_differentiable():
    p = _params(10.0, 1.0, 2.0, 0.5, 0.0, 0.0)
    u = unconstrain(p, SPECS)

    def loss(u_):
        q = constrain(u_, SPECS)
        return q["nodes"]["s"]["stiffness"] + q["nodes"]["s"]["elasticity"]

    g = jax.jit(jax.grad(loss))(u)["nodes"]["s"]
    # d/du exp(u) = exp(u) = stiffness ; d/du sigmoid = e(1-e)
    assert np.isclose(float(g["stiffness"]), 10.0, rtol=1e-5)
    assert np.isclose(float(g["elasticity"]), 0.25, rtol=1e-5)
    assert float(g["mass"]) == 0.0
