"""Property-based tests for coupling convergence norms.

Tests algebraic properties (symmetry, triangle inequality, identity)
of the convergence norms in
:mod:`maddening.core.coupling.acceleration`.
"""


import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from maddening.core.coupling.acceleration import (
    coupling_residual_l2,
    coupling_residual_mixed,
)


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
    @settings(max_examples=500)
    def test_symmetry(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm_ab = float(coupling_residual_l2(s_a, s_b, ["node"]))
        norm_ba = float(coupling_residual_l2(s_b, s_a, ["node"]))
        assert abs(norm_ab - norm_ba) < 1e-5, (
            f"L2 not symmetric: {norm_ab} vs {norm_ba}"
        )

    @given(a=float_arrays)
    @settings(max_examples=200)
    def test_identity(self, a):
        s = _make_state(a)
        norm = float(coupling_residual_l2(s, s, ["node"]))
        assert norm == 0.0, f"L2(a, a) = {norm}, expected 0"

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=500)
    def test_non_negative(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm = float(coupling_residual_l2(s_a, s_b, ["node"]))
        assert norm >= 0.0, f"L2 negative: {norm}"

    @given(a=float_arrays, b=float_arrays, c=float_arrays)
    @settings(max_examples=500)
    def test_triangle_inequality(self, a, b, c):
        s_a = _make_state(a)
        s_b = _make_state(b)
        s_c = _make_state(c)
        norm_ac = float(coupling_residual_l2(s_a, s_c, ["node"]))
        norm_ab = float(coupling_residual_l2(s_a, s_b, ["node"]))
        norm_bc = float(coupling_residual_l2(s_b, s_c, ["node"]))
        assert norm_ac <= norm_ab + norm_bc + 1e-4, (
            f"Triangle violated: {norm_ac} > {norm_ab} + {norm_bc}"
        )


class TestMixedNormProperties:
    """Properties of the mixed abs/rel convergence norm."""

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=500)
    def test_symmetry(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm_ab = float(coupling_residual_mixed(s_a, s_b, ["node"], 1e-8, 1e-6))
        norm_ba = float(coupling_residual_mixed(s_b, s_a, ["node"], 1e-8, 1e-6))
        assert abs(norm_ab - norm_ba) < 1e-5, (
            f"Mixed norm not symmetric: {norm_ab} vs {norm_ba}"
        )

    @given(a=float_arrays)
    @settings(max_examples=200)
    def test_identity(self, a):
        s = _make_state(a)
        norm = float(coupling_residual_mixed(s, s, ["node"], 1e-8, 1e-6))
        assert norm == 0.0, f"Mixed(a, a) = {norm}, expected 0"

    @given(a=float_arrays, b=float_arrays)
    @settings(max_examples=500)
    def test_non_negative(self, a, b):
        s_a = _make_state(a)
        s_b = _make_state(b)
        norm = float(coupling_residual_mixed(s_a, s_b, ["node"], 1e-8, 1e-6))
        assert norm >= 0.0, f"Mixed norm negative: {norm}"
