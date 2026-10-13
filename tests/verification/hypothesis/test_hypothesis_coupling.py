"""Property-based tests for coupling convergence norms.

Tests algebraic properties (symmetry, triangle inequality, identity)
of the convergence norms in
:mod:`maddening.core.coupling.acceleration`.
"""


import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from maddening.core.coupling.acceleration import (
    coupling_residual_l2,
    coupling_residual_mixed,
)
from tests.conftest import EXAMPLES_CHEAP


def _make_state(arr):
    """Wrap a numpy array into the state dict format expected by norms."""
    return {"node": {"field": jnp.asarray(arr)}}


float_arrays = arrays(
    dtype=np.float32,
    shape=(8,),
    elements=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False,
                       width=32),
)


class TestL2NormProperties:
    """Algebraic properties of the L2 convergence norm."""

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_symmetry(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm_ab = float(coupling_residual_l2(s_a, s_b, ["node"]))
        norm_ba = float(coupling_residual_l2(s_b, s_a, ["node"]))
        assert abs(norm_ab - norm_ba) < 1e-5, (
            f"L2 not symmetric: {norm_ab} vs {norm_ba}"
        )

    @given(a=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_identity(self, a):
        s = _make_state(a)
        norm = float(coupling_residual_l2(s, s, ["node"]))
        assert norm == 0.0, f"L2(a, a) = {norm}, expected 0"

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_non_negative(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm = float(coupling_residual_l2(s_a, s_b, ["node"]))
        assert norm >= 0.0, f"L2 negative: {norm}"

    @given(a=float_arrays, k=st.sampled_from([1e-6, 1e-3, 1.0, 1e3, 1e6]))
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_scale_invariance(self, a, k):
        """The property the triangle inequality was traded for.

        Since 0.4.0 the L2 measure divides each field's change by that
        field's own magnitude, so a group's verdict does not depend on
        the units its quantities happen to be written in.  This is what
        stopped a ~1e-5 N force satisfying an absolute ``atol`` on pass
        one while still percent-sized from its fixed point.
        """
        assume(float(np.max(np.abs(a))) > 1e-3)
        plain = float(coupling_residual_l2(
            _make_state(a), _make_state(2.0 * a), ["node"]))
        scaled = float(coupling_residual_l2(
            _make_state(a * k), _make_state(2.0 * a * k), ["node"]))
        assert scaled == pytest.approx(plain, rel=1e-4), (
            f"verdict moved with the units: {plain} at scale 1, "
            f"{scaled} at scale {k}"
        )

    def test_the_triangle_inequality_does_not_hold(self):
        """It is a relative discrepancy measure, not a norm — pinned.

        This replaces a test that asserted the triangle inequality and
        passed until the 0.4.0 scale-aware change.  Dividing by a scale
        that depends on the pair being compared is exactly what buys
        units-invariance, and it is exactly what breaks the triangle
        inequality: the two are in direct conflict and 0.4.0 chose
        invariance deliberately.

        The failure needs a **detour through a much larger state**.
        ``d(a, b)`` and ``d(b, c)`` are each divided by ``b``'s large
        magnitude and shrink; ``d(a, c)`` is divided by the small scale
        of ``a`` and ``c`` and does not.  A converging iteration does
        not take such a detour, which is why this does not bite in
        practice — see the caveat on the error bound in
        ``graph_manager``.

        Pinned rather than deleted so that anyone who restores a
        triangle-inequality assumption finds out here rather than in a
        convergence bound.
        """
        a = np.array([-1.191, 0.934, -0.737], dtype=np.float32)
        b = np.array([-2287.253, 640.208, -571.483], dtype=np.float32)
        c = np.array([1.302, -0.730, 1.219], dtype=np.float32)
        d_ac = float(coupling_residual_l2(
            _make_state(a), _make_state(c), ["node"]))
        d_ab = float(coupling_residual_l2(
            _make_state(a), _make_state(b), ["node"]))
        d_bc = float(coupling_residual_l2(
            _make_state(b), _make_state(c), ["node"]))
        assert d_ac > d_ab + d_bc, (
            "the triangle inequality now holds on the pinned "
            f"counterexample ({d_ac} <= {d_ab} + {d_bc}); if the measure "
            "became a metric again, this test and the error-bound caveat "
            "both want revisiting"
        )


class TestMixedNormProperties:
    """Properties of the mixed abs/rel convergence norm."""

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_symmetry(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm_ab = float(coupling_residual_mixed(s_a, s_b, ["node"], 1e-8, 1e-6))
        norm_ba = float(coupling_residual_mixed(s_b, s_a, ["node"], 1e-8, 1e-6))
        assert abs(norm_ab - norm_ba) < 1e-5, (
            f"Mixed norm not symmetric: {norm_ab} vs {norm_ba}"
        )

    @given(a=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_identity(self, a):
        s = _make_state(a)
        norm = float(coupling_residual_mixed(s, s, ["node"], 1e-8, 1e-6))
        assert norm == 0.0, f"Mixed(a, a) = {norm}, expected 0"

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_non_negative(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm = float(coupling_residual_mixed(s_a, s_b, ["node"], 1e-8, 1e-6))
        assert norm >= 0.0, f"Mixed norm negative: {norm}"
