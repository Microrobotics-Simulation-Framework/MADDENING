"""A coupling group's boolean options take booleans, and ``rtol`` is positive where it divides.

Every option annotated ``bool`` was read by truth value, so
``diagnostics="off"`` turned the diagnostics on and ``to_dict()`` wrote
the string; and ``rtol=0`` under the norms that divide a field's change
by ``rtol * max|field|`` made the residual ``0/0``: the step ran to its
cap with ``residual=nan`` and ``coupling_diagnostics()`` raised
``ZeroDivisionError``.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import pytest

from maddening.core.coupling.group import CouplingGroup

BOOLEAN_OPTIONS = sorted(
    f.name for f in dataclasses.fields(CouplingGroup)
    if typing.get_type_hints(CouplingGroup)[f.name] is bool)
NOT_BOOLEANS = ["rounding", "off", "no", "", 0, 1, 0.0, 1.0, None, [], [True], np.array([1, 2]),
                np.float32(1.0), np.int64(0)]
NUMERIC_OPTIONS = ("max_iterations", "tolerance", "atol", "rtol", "relaxation",
                   "jacobian_reuse", "waveform_iterations")


def _group(**knobs):
    knobs.setdefault("solver", "ift")
    return CouplingGroup(nodes=frozenset({"a", "b"}), **knobs)


def test_the_boolean_options_are_the_three_the_documentation_names():
    """A fourth boolean option joins the battery below by being declared."""
    assert BOOLEAN_OPTIONS == ["diagnostics", "strict_convergence", "subcycling"]


@pytest.mark.parametrize("name", BOOLEAN_OPTIONS)
@pytest.mark.parametrize("value", NOT_BOOLEANS, ids=repr)
def test_a_boolean_option_refuses_anything_but_a_boolean(name, value):
    with pytest.raises(TypeError, match=f"CouplingGroup.{name}="):
        _group(**{name: value})


@pytest.mark.parametrize("name", BOOLEAN_OPTIONS)
@pytest.mark.parametrize("value", [True, False, np.True_, np.False_], ids=repr)
def test_a_boolean_option_takes_python_and_numpy_booleans(name, value):
    assert bool(getattr(_group(**{name: value}), name)) is bool(value)


@pytest.mark.parametrize("norm", ["mixed", "interface"])
@pytest.mark.parametrize("zero", [0.0, 0, -0.0, np.float32(0.0)], ids=repr)
def test_rtol_zero_is_refused_under_the_norms_that_divide_by_it(norm, zero):
    with pytest.raises(ValueError, match="CouplingGroup.rtol=.*must be > 0"):
        _group(convergence_norm=norm, rtol=zero)
    # A dead band does not make it an absolute tolerance.
    with pytest.raises(ValueError, match="CouplingGroup.rtol="):
        _group(convergence_norm=norm, rtol=zero, atol=1e-3)


def test_rtol_zero_is_inert_under_l2_and_a_positive_one_is_taken_everywhere():
    with pytest.warns(UserWarning):
        assert _group(convergence_norm="l2", rtol=0.0).rtol == 0.0
    for norm in ("mixed", "interface"):
        assert _group(convergence_norm=norm, rtol=1e-30).rtol == 1e-30


@pytest.mark.parametrize("name", NUMERIC_OPTIONS)
@pytest.mark.parametrize("value", ["1", None, [1], True, float("nan"), float("inf"), -1],
                         ids=repr)
def test_a_numeric_option_refuses_a_value_of_another_kind_at_declaration(name, value):
    """The sibling sweep: no numeric option takes a string, a bool, a list or a non-finite
    value through to the step or the report."""
    knobs = {name: value}
    if name == "rtol":
        knobs["convergence_norm"] = "mixed"
    with pytest.raises((ValueError, TypeError)):
        _group(**knobs)


@pytest.mark.parametrize("name,zero_ok", [("tolerance", True), ("atol", True),
                                          ("jacobian_reuse", True), ("max_iterations", False),
                                          ("waveform_iterations", False), ("relaxation", False)])
def test_zero_is_taken_only_where_it_has_a_meaning(name, zero_ok):
    """``tolerance=0`` and ``atol=0`` are thresholds; a count or a relaxation of zero is not."""
    value = 0 if name in ("jacobian_reuse", "max_iterations", "waveform_iterations") else 0.0
    if zero_ok:
        assert getattr(_group(**{name: value}), name) == value
    else:
        with pytest.raises(ValueError, match=f"CouplingGroup.{name}="):
            _group(**{name: value})
