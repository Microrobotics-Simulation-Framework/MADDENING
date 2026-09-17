"""``maddening.core.coupling.mapping``: patch test, conservation, adjoint.

The acceptance gates from PLAN_accuracy_and_usd.md §8c: a constant (and,
with polynomial augmentation, a linear) field crosses a non-conforming
interface exactly; a conservative transfer preserves the integral.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.coupling.interface_mapping import conservative_projection_1d
from maddening.core.coupling.mapping import (
    Mapping,
    StaticLinearMapping,
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
    rbf_matrix,
)

KERNELS = ("gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline")


def _cloud(rng, n, dim, spread=1.0):
    """Well-separated points: a jittered grid, so the kernel system is
    conditioned well enough for a float32 solve."""
    if dim == 1:
        x = np.linspace(0.0, spread, n) + rng.uniform(-0.2, 0.2, n) * spread / max(n - 1, 1)
        return x.reshape(-1, 1)
    side = int(np.ceil(np.sqrt(n)))
    g = np.stack(np.meshgrid(np.linspace(0, spread, side), np.linspace(0, spread, side)),
                 -1).reshape(-1, 2)[:n]
    return g + rng.uniform(-0.2, 0.2, g.shape) * spread / max(side - 1, 1)


def _eps(points):
    """Shape parameter tied to the mean spacing: eps * h ~ 1."""
    n, d = points.shape
    h = (points.max(0) - points.min(0)).max() / max(n ** (1.0 / d) - 1, 1)
    return float(1.0 / h)


# ---------------------------------------------------------------------------
# Patch test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("dim", [1, 2])
def test_patch_constant_and_linear_reproduced(kernel, dim):
    rng = np.random.default_rng(0)
    src, tgt = _cloud(rng, 12, dim), _cloud(rng, 7, dim, spread=0.9)
    m = rbf_mapping(src, tgt, kernel=kernel, epsilon=_eps(src), polynomial=True)
    assert isinstance(m, Mapping) and m.n_source == 12 and m.n_target == 7
    a = np.array([0.3, -1.2][:dim] + [0.0] * (2 - dim))[:dim]
    for field_fn in (lambda p: np.full(p.shape[0], 2.5),
                     lambda p: 1.0 + p @ a):
        got = np.asarray(m.apply(jnp.asarray(field_fn(src), jnp.float32)))
        want = field_fn(tgt)
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_patch_test_fails_without_polynomial_for_gaussian():
    """Documents why polynomial augmentation is the default: a plain
    Gaussian interpolant does not reproduce constants off the nodes."""
    rng = np.random.default_rng(1)
    src, tgt = _cloud(rng, 8, 1), _cloud(rng, 5, 1, spread=0.9)
    m = rbf_mapping(src, tgt, kernel="gaussian", epsilon=_eps(src), polynomial=False)
    got = np.asarray(m.apply(jnp.ones(8, jnp.float32)))
    assert np.max(np.abs(got - 1.0)) > 1e-3


@given(seed=st.integers(0, 2**31), n_src=st.integers(4, 16), n_tgt=st.integers(2, 12),
       dim=st.integers(1, 2), kernel=st.sampled_from(KERNELS))
@settings(max_examples=60, deadline=None)
def test_patch_property_float64(seed, n_src, n_tgt, dim, kernel):
    """Random clouds, float64: constants and linear fields to 1e-8."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(seed)
        src, tgt = _cloud(rng, n_src, dim), _cloud(rng, n_tgt, dim, spread=0.95)
        H = rbf_matrix(jnp.asarray(src), jnp.asarray(tgt), kernel=kernel,
                       epsilon=_eps(src), polynomial=True, ridge=1e-12)
        a = rng.uniform(-1, 1, dim)
        for f in (lambda p: np.full(p.shape[0], 3.0), lambda p: 0.5 + p @ a):
            got = np.asarray(H @ jnp.asarray(f(src)))
            np.testing.assert_allclose(got, f(tgt), rtol=1e-7, atol=1e-7)
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS)
def test_conservative_mode_preserves_total(kernel):
    rng = np.random.default_rng(2)
    src, tgt = _cloud(rng, 9, 2), _cloud(rng, 14, 2, spread=1.1)
    m = rbf_mapping(src, tgt, kernel=kernel, epsilon=_eps(tgt), mode="conservative")
    assert m.mode == "conservative" and m.H.shape == (14, 9)
    force = jnp.asarray(rng.normal(size=9), jnp.float32)
    out = m.apply(force)
    assert out.shape == (14,)
    np.testing.assert_allclose(float(jnp.sum(out)), float(jnp.sum(force)), rtol=1e-4)
    # Multi-component fields (a force vector per point) too.
    f3 = jnp.asarray(rng.normal(size=(9, 3)), jnp.float32)
    np.testing.assert_allclose(np.asarray(jnp.sum(m.apply(f3), 0)),
                               np.asarray(jnp.sum(f3, 0)), rtol=1e-4, atol=1e-5)


@given(seed=st.integers(0, 2**31), n_src=st.integers(3, 12), n_tgt=st.integers(3, 12))
@settings(max_examples=40, deadline=None)
def test_conservation_property_nearest_and_rbf(seed, n_src, n_tgt):
    rng = np.random.default_rng(seed)
    src, tgt = _cloud(rng, n_src, 1), _cloud(rng, n_tgt, 1, spread=0.9)
    v = jnp.asarray(rng.normal(size=n_src), jnp.float32)
    for m in (nearest_neighbor_mapping(src, tgt, mode="conservative"),
              rbf_mapping(src, tgt, kernel="thin_plate_spline", epsilon=1.0,
                          mode="conservative")):
        np.testing.assert_allclose(float(jnp.sum(m.apply(v))), float(jnp.sum(v)),
                                   rtol=1e-4, atol=1e-5)


def test_projection_1d_matches_closure_and_conserves_integral():
    sb, tb = jnp.linspace(0, 1, 11), jnp.linspace(0, 1, 6)
    m = projection_1d_mapping(sb, tb)
    v = jnp.sin(jnp.linspace(0.05, 0.95, 10) * jnp.pi)
    np.testing.assert_allclose(np.asarray(m.apply(v)),
                               np.asarray(conservative_projection_1d(sb, tb)(v)), rtol=1e-6)
    src_int = float(jnp.sum(v * (sb[1:] - sb[:-1])))
    tgt_int = float(jnp.sum(m.apply(v) * (tb[1:] - tb[:-1])))
    assert src_int == pytest.approx(tgt_int, abs=1e-5)


# ---------------------------------------------------------------------------
# Adjoint, weights, jit/grad, validation
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 2**31), n_src=st.integers(2, 10), n_tgt=st.integers(2, 10))
@settings(max_examples=40, deadline=None)
def test_apply_T_is_the_adjoint(seed, n_src, n_tgt):
    rng = np.random.default_rng(seed)
    m = matrix_mapping(rng.normal(size=(n_tgt, n_src)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=n_src), jnp.float32)
    w = jnp.asarray(rng.normal(size=n_tgt), jnp.float32)
    lhs = float(jnp.dot(m.apply(v), w))
    rhs = float(jnp.dot(v, m.apply_T(w)))
    # Both sides are float32 sums of O(n) products that can nearly
    # cancel, so the tolerance scales with the terms, not the result.
    scale = float(jnp.linalg.norm(m.apply(v)) * jnp.linalg.norm(w)) + 1e-6
    assert abs(lhs - rhs) <= 1e-5 * scale, (lhs, rhs, scale)
    # and matches what JAX's transpose of ``apply`` gives.  Same scaling
    # argument as above: both are float32 sums over the target index, so a
    # component that nearly cancels is accurate in absolute terms, not
    # relative ones (seed=14928, n_src=3, n_tgt=9 lands on -0.0027 with a
    # 2e-7 absolute difference, which is 7e-5 relative).
    _, vjp = jax.vjp(m.apply, v)
    by_vjp, by_apply_T = np.asarray(vjp(w)[0]), np.asarray(m.apply_T(w))
    row_scale = float(jnp.linalg.norm(m.H, axis=0).max() * jnp.linalg.norm(w))
    np.testing.assert_allclose(by_vjp, by_apply_T, rtol=1e-5, atol=1e-5 * row_scale)


@pytest.mark.parametrize("seed,n_src,n_tgt", [(14928, 3, 9)])
def test_adjoint_agrees_with_the_vjp_when_a_component_nearly_cancels(seed, n_src, n_tgt):
    """A component of the adjoint that nearly cancels is accurate in
    absolute terms, not relative ones.  Pinned: this case failed the bare
    ``rtol=1e-5`` with a 2e-7 absolute difference on a -0.0027 component."""
    test_apply_T_is_the_adjoint.hypothesis.inner_test(seed, n_src, n_tgt)


def test_weights_override_and_are_differentiable():
    rng = np.random.default_rng(3)
    src, tgt = _cloud(rng, 6, 1), _cloud(rng, 4, 1)
    m = rbf_mapping(src, tgt)
    v = jnp.asarray(rng.normal(size=6), jnp.float32)
    base = m.apply(v)
    doubled = m.apply(v, weights={"H": 2.0 * m.H})
    np.testing.assert_allclose(np.asarray(doubled), 2 * np.asarray(base), rtol=1e-6)
    assert set(m.params_pytree()) == {"H"}
    g = jax.jit(jax.grad(lambda w: jnp.sum(m.apply(v, weights=w) ** 2)))(m.params_pytree())
    assert g["H"].shape == m.H.shape and bool(jnp.all(jnp.isfinite(g["H"])))
    assert float(jnp.max(jnp.abs(g["H"]))) > 0.0


def test_describe_has_no_weights_and_validation_errors():
    m = rbf_mapping(np.linspace(0, 1, 5), np.linspace(0, 1, 3), kernel="multiquadric",
                    epsilon=2.0)
    d = m.describe()
    assert d == {"kind": "rbf", "mode": "consistent", "shape": [3, 5],
                 "kernel": "multiquadric", "epsilon": 2.0, "polynomial": True,
                 "ridge": 1e-8,
                 # small point sets are inlined into the spec; never the weights
                 "points": {"source_points": {"inline": [0.0, 0.25, 0.5, 0.75, 1.0],
                                              "dtype": "float64"},
                            "target_points": {"inline": [0.0, 0.5, 1.0],
                                              "dtype": "float64"}}}
    assert "H" not in d and "weights" not in d
    with pytest.raises(ValueError, match="Unknown kernel"):
        rbf_mapping(np.linspace(0, 1, 5), np.linspace(0, 1, 3), kernel="cubic")
    with pytest.raises(ValueError, match="mode="):
        rbf_mapping(np.linspace(0, 1, 5), np.linspace(0, 1, 3), mode="sideways")
    with pytest.raises(ValueError, match="dimension"):
        rbf_mapping(np.zeros((4, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="2-D"):
        StaticLinearMapping(jnp.zeros(3))


def test_nearest_neighbor_consistent_selects_nearest():
    src = np.array([0.0, 1.0, 2.0])
    tgt = np.array([0.1, 1.9, 0.9])
    m = nearest_neighbor_mapping(src, tgt)
    np.testing.assert_array_equal(np.asarray(m.apply(jnp.array([10.0, 20.0, 30.0]))),
                                  [10.0, 30.0, 20.0])
