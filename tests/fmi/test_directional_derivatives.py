"""Tests for the FMI 3.0 directional-derivative substrate (v0.3.0 §A1)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)


# A small differentiable function to test against — exact Jacobian known.
def f(x):
    return jnp.array([
        x[0] ** 2 + x[1],
        jnp.sin(x[0]) * x[1],
        2.0 * x[0] + 3.0 * x[1],
    ])


def jacobian_at(x):
    return jnp.array([
        [2.0 * x[0], 1.0],
        [jnp.cos(x[0]) * x[1], jnp.sin(x[0])],
        [2.0, 3.0],
    ])


class TestForwardMode:

    def test_jvp_matches_dense_jacobian_product(self):
        x = jnp.array([0.5, -0.3], dtype=jnp.float32)
        v = jnp.array([1.0, 0.0], dtype=jnp.float32)

        out = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.FORWARD, x=x, v=v,
        )
        expected = jacobian_at(x) @ v
        np.testing.assert_allclose(
            jax.device_get(out), jax.device_get(expected), atol=1e-5,
        )

    def test_jvp_picks_out_correct_column(self):
        """e_i seed -> i-th Jacobian column."""
        x = jnp.array([1.0, -2.0], dtype=jnp.float32)
        for i in range(2):
            v = jnp.zeros_like(x).at[i].set(1.0)
            out = get_directional_derivative(
                f, kind=DirectionalDerivativeKind.FORWARD, x=x, v=v,
            )
            np.testing.assert_allclose(
                jax.device_get(out), jacobian_at(x)[:, i], atol=1e-5,
            )

    def test_jvp_with_pytree_input(self):
        """The FMI seed/output pytree contract has to support dicts."""
        def g(x):
            return {"y": x["a"] + 2.0 * x["b"]}

        x = {"a": jnp.array(3.0), "b": jnp.array(4.0)}
        v = {"a": jnp.array(1.0), "b": jnp.array(0.5)}

        out = get_directional_derivative(
            g, kind=DirectionalDerivativeKind.FORWARD, x=x, v=v,
        )
        # d/da y = 1; d/db y = 2.  Seed (1, 0.5) -> output (1*1 + 2*0.5) = 2.
        np.testing.assert_allclose(float(out["y"]), 2.0, atol=1e-5)


class TestReverseMode:

    def test_vjp_matches_dense_jacobian_transpose_product(self):
        x = jnp.array([0.5, -0.3], dtype=jnp.float32)
        # Cotangent: same shape as f(x) — 3-vector.
        v = jnp.array([1.0, 0.0, 0.5], dtype=jnp.float32)

        out = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.REVERSE, x=x, v=v,
        )
        expected = jacobian_at(x).T @ v
        np.testing.assert_allclose(
            jax.device_get(out), jax.device_get(expected), atol=1e-5,
        )

    def test_reverse_picks_out_row(self):
        x = jnp.array([1.0, -2.0], dtype=jnp.float32)
        for i in range(3):
            v = jnp.zeros(3, dtype=jnp.float32).at[i].set(1.0)
            out = get_directional_derivative(
                f, kind=DirectionalDerivativeKind.REVERSE, x=x, v=v,
            )
            np.testing.assert_allclose(
                jax.device_get(out), jacobian_at(x)[i, :], atol=1e-5,
            )


class TestRoundTrip:

    def test_forward_and_reverse_agree(self):
        """For a single seed pair (u, v), u . (J v) == v . (J^T u)."""
        x = jnp.array([1.5, 0.7], dtype=jnp.float32)
        u = jnp.array([0.3, -0.2, 0.7], dtype=jnp.float32)  # cotangent
        v = jnp.array([0.4, 0.6], dtype=jnp.float32)         # tangent

        forward = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.FORWARD, x=x, v=v,
        )
        reverse = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.REVERSE, x=x, v=u,
        )

        lhs = float(jnp.dot(u, forward))
        rhs = float(jnp.dot(reverse, v))
        np.testing.assert_allclose(lhs, rhs, atol=1e-5)


class TestErrors:

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown kind"):
            get_directional_derivative(
                f, kind="bogus", x=jnp.array([1.0, 2.0]),  # type: ignore
                v=jnp.array([1.0, 0.0]),
            )


class TestStabilityTagging:

    def test_get_dd_tagged_evolving(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert get_directional_derivative._stability_level == \
            StabilityLevel.EVOLVING
