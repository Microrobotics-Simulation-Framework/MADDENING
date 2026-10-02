"""A bounds check compares the value it was given, not the value XLA sees.

``ParamSpec.check`` compared through ``jnp``, and XLA's CPU backend flushes
a subnormal operand to zero: a float32 ``-1e-40`` read as ``0`` and passed a
``(0, None)`` bound -- the bound of ``SpringDamperNode``'s ``damping`` and of
every positive identity-transform parameter.  ``check_params``, the FMU's
``set`` and ``set_state`` and REST's ``PUT /graph/params`` all check through
it, so each accepted a value below the range the spec declares and reported
it back as stored.  The check now compares on the host, exactly, with the
value and the bound rounded to the dtype JAX would compare them in; under a
``log`` / ``logit`` transform, whose own arithmetic flushes, a value within
the smallest normal number of its bound is refused as on it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec, check_bounds
from maddening.nodes.spring import SpringDamperNode

TINY32 = float(np.finfo(np.float32).tiny)


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0))
    gm.compile()
    return gm


@pytest.mark.parametrize("value", [-1e-45, -1e-40, -1e-38])
def test_a_value_below_a_zero_bound_by_a_subnormal_is_refused(value):
    """By ``check``, by ``check_bounds`` and by ``gm.check_params`` -- the
    gate the FMU and REST write paths share."""
    spec = ParamSpec(bounds=(0.0, None))
    with pytest.raises(ValueError, match="below bound 0.0"):
        spec.check(np.float32(value), name="damping")
    with pytest.raises(ValueError, match="below bound 0.0"):
        check_bounds({"c": jnp.asarray([1.0, value], jnp.float32)}, {"c": spec})
    gm = _spring()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["damping"] = jnp.float32(value)
    with pytest.raises(ValueError, match=r"\['damping'\].*below bound 0.0"):
        gm.check_params(p)


@pytest.mark.parametrize("value", [1e-45, 1e-40])
def test_a_value_above_a_zero_bound_by_a_subnormal_is_refused(value):
    with pytest.raises(ValueError, match="above bound 0.0"):
        ParamSpec(bounds=(None, 0.0)).check(np.float32(value), name="x")


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-45, 1e-40])
def test_a_subnormal_or_zero_inside_an_identity_bound_is_accepted(value):
    """Inside ``[0, inf)``: an identity coordinate's bound is inclusive, and
    a subnormal is a value on the right side of it."""
    ParamSpec(bounds=(0.0, None)).check(np.float32(value), name="damping")


@pytest.mark.parametrize("spec", [ParamSpec(transform="log"),
                                  ParamSpec(bounds=(0.0, None), transform="log"),
                                  ParamSpec(bounds=(0.0, 1.0), transform="logit")],
                         ids=["log", "log-from-0", "logit"])
def test_a_transformed_value_within_the_smallest_normal_of_its_bound_is_refused(spec):
    """The step's own arithmetic flushes it onto the bound, where the
    coordinate is ``-inf`` (``jnp.log`` of a float32 ``1e-40`` is), so it is
    refused as on the bound; the smallest normal number is not."""
    assert not np.isfinite(float(jnp.log(jnp.float32(1e-40))))
    with pytest.raises(ValueError, match="below bound 0.0"):
        spec.check(np.float32(1e-40), name="k")
    spec.check(np.float32(TINY32), name="k")


def test_the_bound_is_rounded_to_the_values_dtype_as_jax_rounds_it():
    """The host comparison keeps what the ``jnp`` one got right: a float32
    ``0.1`` *is* the bound ``0.1`` in float32, so a strict (``log``) bound
    refuses it and an inclusive one accepts it; an integer leaf against a
    float bound, a bfloat16 leaf and a float64 NumPy value without x64 are
    compared as JAX compares them."""
    with pytest.raises(ValueError, match="below bound 0.1"):
        ParamSpec(bounds=(0.1, None), transform="log").check(np.float32(0.1), name="k")
    ParamSpec(bounds=(0.1, None)).check(np.float32(0.1), name="k")
    ParamSpec(bounds=(0.0, 5.0)).check(jnp.int32(5), name="n")
    with pytest.raises(ValueError, match="above bound 5.0"):
        ParamSpec(bounds=(0.0, 5.0)).check(jnp.int32(6), name="n")
    with pytest.raises(ValueError, match="below bound 0.0"):
        ParamSpec(bounds=(0.0, 5.0)).check(jnp.asarray(-1.0, jnp.bfloat16), name="b")
    # 5.0000001 is 5.0 in float32, which is what jnp.asarray makes of it.
    ParamSpec(bounds=(0.0, 5.0)).check(np.float64(5.0000001), name="x")
    with pytest.raises(ValueError, match="is not finite"):
        ParamSpec(bounds=(0.0, 5.0)).check(jnp.asarray(np.nan, jnp.float32), name="x")


@pytest.mark.parametrize("param, value, flagged", [
    ("damping", -1e-40, True), ("damping", 1e-40, False), ("damping", 2.0, False),
    ("stiffness", 1e-40, True), ("stiffness", TINY32, False)])
def test_the_params_table_flags_what_check_refuses(param, value, flagged):
    """``params_table``'s ``out_of_bounds`` applies ``check``'s rule, on the
    host, so it agrees with it about a subnormal too: below the identity
    damping's 0, and within the smallest normal of the ``log`` stiffness's."""
    gm = _spring()
    gm.params["nodes"]["s"][param] = jnp.float32(value)
    row = next(r for r in gm.params_table() if r["param"] == param)
    assert row["out_of_bounds"] is flagged
    refused = False
    try:
        gm.check_params()
    except ValueError:
        refused = True
    assert refused is flagged
