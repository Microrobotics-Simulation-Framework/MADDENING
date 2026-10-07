"""``constrain`` returns a value ``check`` accepts, bounds near the smallest
normal number included.

XLA's CPU backend reads a subnormal operand as zero and stores a subnormal
result as zero; ``ParamSpec.check`` compares exactly, on the host
(MADD-ANO-136).  So where a bound is a subnormal number the two disagreed:
``ParamSpec(bounds=(None, -1.4e-45)).to_constrained(u)`` was ``-0.0`` in
float32 from every ``u`` -- above the bound -- and ``check`` refused what
``constrain`` had returned (a property test could draw it).  The same
arithmetic put a ``logit`` leaf whose interval lies within a few smallest
normals of zero *on* its bound.

The battery: every pair of bounds from a grid around zero and the smallest
normal, every transform, coordinates of every size, in float32 and float64.
"""
from __future__ import annotations

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.params import _LOGIT_MIN_WIDTH_TINIES, ParamSpec


@contextlib.contextmanager
def _precision(dtype):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", np.dtype(dtype) == np.float64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _grid(dtype):
    """``(bounds, coordinates)``: zero, the smallest subnormal and normal
    numbers and their neighbours and multiples, both signs, and ordinary
    numbers."""
    F = np.dtype(dtype).type
    tiny = float(np.finfo(F).tiny)
    least = float(np.nextafter(F(0), F(1)))
    below_tiny = float(np.nextafter(F(tiny), F(0)))
    small = [least, 2 * least, tiny / 1024, below_tiny, tiny / 2]
    multiples = [k * tiny for k in (1, 1.5, 2, 3, 4, 8, 64)]
    magnitudes = small + multiples + [1.0]
    bounds = [None, 0.0] + magnitudes + [-m for m in magnitudes]
    coordinates = ([0.0, -0.0, 1.0, -1.0, 50.0, -50.0, 800.0, -800.0, 1e4, -1e4]
                   + small + [-m for m in small] + [tiny, -tiny, 2 * tiny])
    return bounds, coordinates


def _specs(dtype, transform):
    bounds, _ = _grid(dtype)
    for lo, hi in itertools.product(bounds, bounds):
        if lo is not None and hi is not None and not lo < hi:
            continue
        try:
            yield ParamSpec(bounds=(lo, hi), transform=transform)
        except ValueError:
            continue                    # not a spec at all (log with an upper bound, ...)


@pytest.mark.parametrize("transform", [None, "log", "logit"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_constrain_lands_inside_the_bounds_check_enforces(dtype, transform):
    """From every coordinate: the value is accepted by ``check``, or the
    spec is refused by name for this dtype -- never a value outside."""
    F = np.dtype(dtype).type
    _, coordinates = _grid(dtype)
    outside, mapped, refused = [], 0, 0
    with _precision(dtype):
        us = jnp.asarray(np.asarray(coordinates, dtype=dtype))
        for spec in _specs(dtype, transform):
            try:
                values = spec.to_constrained(us)        # every coordinate at once
            except ValueError as exc:
                assert "cannot map" in str(exc), (spec, str(exc))
                refused += 1
                continue
            mapped += len(coordinates)
            assert values.dtype == np.dtype(dtype)
            try:
                spec.check(values)
            except ValueError:
                for u, value in zip(coordinates, np.asarray(values)):
                    try:
                        spec.check(value)
                    except ValueError as exc:
                        outside.append((spec.bounds, u, float(value), str(exc)[:60]))
    assert not outside, (len(outside), outside[:5])
    assert mapped > 1000 or transform == "log", mapped
    assert (refused > 0) is (transform == "logit")


def test_the_drawn_case_is_inside_its_subnormal_bound():
    """The case a property test drew: an upper bound of the smallest
    negative float32, which the clip returned as ``-0.0``."""
    spec = ParamSpec(bounds=(None, -1.4e-45), transform=None)
    bound = np.float32(-1.4e-45)
    for u in (0.0, -0.0, 1.0, 1e4, 1.4e-45, -1.4e-45, -1e-40):
        value = np.asarray(spec.to_constrained(jnp.float32(u)))
        assert value.tobytes() == bound.tobytes(), (u, value)
        spec.check(value)
    inside = np.asarray(spec.to_constrained(jnp.float32(-3.0)))
    assert inside == np.float32(-3.0)


@pytest.mark.parametrize("bounds, u, want", [
    ((1.4e-45, None), -2.0, 1.4e-45), ((1.4e-45, None), 0.0, 1.4e-45),
    ((1.4e-45, None), 2.0, 2.0), ((-1e-40, None), -2.0, -1e-40),
    ((-1e-40, None), 0.0, 0.0), ((None, 1e-40), 2.0, 1e-40), ((None, 1e-40), 0.0, 0.0),
    ((None, 1e-40), -2.0, -2.0), ((-1e-40, 1e-40), 5.0, 1e-40), ((-1e-40, 1e-40), -5.0, -1e-40),
    ((-1e-40, 3.0), 5.0, 3.0), ((-1e-40, 3.0), 2.0, 2.0), ((-3.0, 1e-40), -5.0, -3.0),
])
def test_a_subnormal_bound_clips_as_the_exact_comparison_says(bounds, u, want):
    spec = ParamSpec(bounds=bounds, transform=None)
    value = np.asarray(jax.jit(spec.to_constrained)(jnp.float32(u)))
    assert value.tobytes() == np.float32(want).tobytes(), (value, want)
    spec.check(value)


def test_the_subnormal_clip_differentiates_and_batches_as_the_ordinary_one():
    """1 where the value passes through, 0 where a bound is selected; an
    ordinary spec's value and derivative are unchanged."""
    near = ParamSpec(bounds=(None, -1.4e-45), transform=None)
    ordinary = ParamSpec(bounds=(None, -1.0), transform=None)
    for spec, inside, outside in ((near, -3.0, 3.0), (ordinary, -3.0, 3.0)):
        grad = jax.grad(lambda u, spec=spec: spec.to_constrained(u))
        assert float(grad(jnp.float32(inside))) == 1.0
        assert float(grad(jnp.float32(outside))) == 0.0
    batch = jax.jit(jax.vmap(near.to_constrained))(jnp.asarray([-3.0, 0.0, 3.0], jnp.float32))
    assert np.asarray(batch).tobytes() == np.asarray([-3.0, -1.4e-45, -1.4e-45],
                                                     np.float32).tobytes()
    assert float(ordinary.to_constrained(jnp.float32(-1.0))) == -1.0
    assert float(jax.grad(ordinary.to_constrained)(jnp.float32(-1.0))) == 1.0   # on the bound


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_a_logit_interval_narrower_than_four_smallest_normals_is_refused_by_name(dtype):
    """It holds no value the maps can return: one inside each bound by the
    smallest normal number, and a normal distance between them."""
    F = np.dtype(dtype).type
    tiny = float(np.finfo(F).tiny)
    assert _LOGIT_MIN_WIDTH_TINIES == 4
    with _precision(dtype):
        # a subnormal bound: the arithmetic would read it as another number
        for bounds in ((tiny / 2, 1.0), (-1.0, -tiny / 4), (-tiny / 2, 64 * tiny)):
            spec = ParamSpec(bounds=bounds, transform="logit")
            with pytest.raises(ValueError, match="cannot map .* is a subnormal"):
                spec.to_constrained(jnp.asarray(F(0.0)))
            with pytest.raises(ValueError, match="cannot map .* is a subnormal"):
                spec.to_unconstrained(jnp.asarray(F(0.5)))
        # The same shape as a property test drew it (float32: 0.85 of the
        # smallest normal).  Under x64 it is an ordinary bound, and mapped.
        drawn = ParamSpec(bounds=(1.0027701900272462e-38, 1.0), transform="logit")
        if np.dtype(dtype) == np.float32:
            for refused in (lambda: drawn.to_constrained(jnp.asarray(F(0.0))),
                            lambda: drawn.to_unconstrained(jnp.asarray(F(0.5))),
                            lambda: drawn.check(jnp.asarray(F(0.5)))):
                with pytest.raises(ValueError, match="cannot map .* is a subnormal"):
                    refused()
        else:
            drawn.check(drawn.to_constrained(jnp.asarray(F(0.0))))
        for lo, hi in ((0.0, tiny), (0.0, 3.5 * tiny), (-tiny, tiny), (tiny, 2 * tiny)):
            spec = ParamSpec(bounds=(lo, hi), transform="logit")
            with pytest.raises(ValueError, match="cannot map .* width .* smallest normals"):
                spec.to_constrained(jnp.asarray(F(0.0)))
            with pytest.raises(ValueError, match="cannot map"):
                spec.check(jnp.asarray(F(0.5 * (lo + hi))))
        for lo, hi in ((0.0, 4 * tiny), (-2 * tiny, 2 * tiny), (0.0, 64 * tiny)):
            spec = ParamSpec(bounds=(lo, hi), transform="logit")
            for u in (-800.0, 0.0, 800.0):
                spec.check(spec.to_constrained(jnp.asarray(F(u))))


def test_an_ordinary_logit_interval_is_mapped_as_before():
    """The floor on the interior margin binds only within ``1 / (4 eps)``
    smallest normals of zero: the limits of an ordinary interval are the
    same numbers."""
    fi = jnp.finfo(jnp.float32)
    for lo, hi in ((0.0, 1.0), (8.0, 9.0), (-1.0, 1.0), (1e-30, 1e-29), (0.0, 1e-30)):
        spec = ParamSpec(bounds=(lo, hi), transform="logit")
        inner_lo, inner_hi = spec._logit_interior(fi)                   # noqa: SLF001
        m = min(4.0 * float(fi.eps) * max(abs(lo), abs(hi), hi - lo), 0.25 * (hi - lo))
        assert inner_lo == max(lo + m, float(np.nextafter(np.float32(lo), np.float32(np.inf))))
        assert inner_hi == min(hi - m, float(np.nextafter(np.float32(hi), np.float32(-np.inf))))
