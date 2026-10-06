"""The spectral estimate says what rounding left it able to say.

``arnoldi_spectral_radius`` is the estimate behind ``rho_spectral``,
``spectral_error_bound`` and ``spectral_usable``.  Three things made it
report a wrong radius as settled, each on the matrix alone:

* **the breakdown test** discarded a new Krylov direction below 1e-5 of
  the product in every dtype -- in float64, directions eleven orders
  above the products' rounding, which a non-normal matrix turns into a
  large movement of its eigenvalues, and near-degenerate modes into one;
* **the radius of the compressed matrix** came from repeated squaring,
  which is not backward stable for a non-normal matrix in float32;
* **the settled test** read only the Arnoldi residual, which bounds a
  Ritz value's error for a normal matrix alone: nothing measured what
  rounding (or a ninth vector) does to a non-normal one's radius.

The tests hold the estimate to one statement over seeded families of
matrices: *where it reports the spectrum settled, the radius is within
the margin settled stands for*, and it still reports settled on the
matrices whose radius the dtype determines.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import acceleration
from maddening.core.coupling.acceleration import (
    SPECTRAL_SETTLED_FRACTION,
    arnoldi_spectral_radius,
    spectral_rate_settled,
)
from tests.property.sysid_transform_grid import precision

SEEDS = range(40)


def _hub(seed):
    """float32: rank two in eight dimensions, two fields 1e-4 to 1e-2 of the rest."""
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 8))
    B *= rng.uniform(0.05, 0.6) / max(abs(np.linalg.eigvals(B)))
    d = np.ones(8)
    d[:2] = 10.0 ** rng.uniform(-4, -2)
    return (B * d[None, :]) / d[:, None], "float32"


def _ring(seed):
    """float32: a graded cyclic matrix of five, every eigenvalue of one modulus."""
    rng = np.random.default_rng(seed)
    g = 10.0 ** rng.uniform(-2, 2, 5)
    g *= (rng.uniform(0.3, 0.95) ** 5 / np.prod(g)) ** (1 / 5)
    A = np.zeros((5, 5))
    for i in range(5):
        A[(i + 1) % 5, i] = g[i]
    return A, "float32"


def _normal(seed):
    """float32: symmetric, six distinct eigenvalues."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    return Q @ np.diag(rng.uniform(-0.95, 0.95, 6)) @ Q.T, "float32"


#: family -> (builder, the least number of the forty seeds it must call settled).
#: Measured on this tree: 30, 40 and 40 (the tree before read 37, 40 and 40,
#: with 8, 2 and 0 of them wrong by more than the margin).
FAMILIES = {"hub": (_hub, 24), "ring": (_ring, 36), "normal": (_normal, 40)}


@functools.lru_cache(maxsize=None)
def _compiled(noise_eps, with_extra: bool):
    """One compiled estimate per shape and dtype: the matrix is an argument."""
    def run(A, v0, extra):
        return arnoldi_spectral_radius(lambda v: A @ v, v0, noise_eps=noise_eps,
                                       v_extra=extra if with_extra else None)
    return jax.jit(run)


def _estimate(A, dtype, v0=None, v_extra=None, noise_eps=None):
    with precision(dtype == "float64"):
        Aj = jnp.asarray(A, dtype)
        if v0 is None:
            v0 = jax.random.normal(jax.random.PRNGKey(0), (A.shape[0],), Aj.dtype)
        extra = jnp.zeros(A.shape[0], dtype) if v_extra is None else jnp.asarray(v_extra, dtype)
        rho, res, amp = _compiled(noise_eps, v_extra is not None)(
            Aj, jnp.asarray(v0, dtype), extra)
        return float(rho), float(res), float(amp), bool(spectral_rate_settled(rho, res))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_settled_radius_is_within_the_margin_and_the_determined_ones_settle(family):
    """Settled means within 5% of ``1 - rho``; and the flag is not simply withdrawn."""
    build, least = FAMILIES[family]
    settled, wrong = 0, []
    for seed in SEEDS:
        A, dtype = build(seed)
        true = float(max(abs(np.linalg.eigvals(A))))
        rho, _res, _amp, ok = _estimate(A, dtype)
        settled += ok
        if ok and abs(rho - true) > SPECTRAL_SETTLED_FRACTION * (1.0 - true):
            wrong.append((seed, rho, true))
    assert not wrong, f"{family}: settled and outside the margin (seed, read, radius): {wrong}"
    assert settled >= least, f"{family}: only {settled} of {len(SEEDS)} read settled"


def test_a_direction_far_above_rounding_is_kept_in_float64():
    """A direction 3e-8 of the product is eight orders above float64's rounding.

    ``A = [[a, t], [K, a]]`` from the start ``e_2``: the first product is
    ``(t, a)``, whose part outside the start is ``t = 1e-8`` beside
    ``a = 0.3``.  Discarded, the space is the start alone and the radius
    reads ``a``; the eigenvalues are ``a +/- sqrt(t K)``, 0.4 and 0.2.
    """
    A = np.array([[0.3, 1e-8], [1e6, 0.3]])
    rho, res, _amp, ok = _estimate(A, "float64", v0=np.array([0.0, 1.0]))
    assert rho == pytest.approx(0.4, abs=1e-9), rho
    assert ok and res < 1e-6, (res, ok)


def test_near_degenerate_modes_are_not_taken_for_one():
    """Three modes within 1e-5 of each other: the resolvent covers the residual's every part.

    The second and third Krylov directions are 1e-5 and 1e-10 of the
    product.  Taken for rounding, the space was one mode (and the
    residual's own direction), and the resolvent norm on it was applied
    to a residual with a part outside.
    """
    rng = np.random.default_rng(98)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    lam = (1.0 - 1e-6) * (1.0 - 1e-5 * np.array([0.0, 5.0 / 11.0, 1.0]))
    A = Q @ np.diag(lam) @ Q.T
    worst = np.inf
    for k in range(3):
        r = Q @ np.eye(3)[k] + 1e-3 * rng.normal(size=3)
        exact = float(np.linalg.norm(np.linalg.solve(np.eye(3) - A, r)))
        _rho, res, amp, _ok = _estimate(A, "float64", v0=rng.normal(size=3), v_extra=r)
        radius_form = 1.0 / max(1.0 - (_rho + 2.0 * res), 1e-300)
        worst = min(worst, max(amp, radius_form) * float(np.linalg.norm(r)) / exact)
    assert worst >= 1.0 - 1e-6, f"the bound's factor is {worst}x the exact resolvent's"


def test_the_compressed_radius_is_the_eigenvalues_not_the_squaring(monkeypatch):
    """Five eigenvalues of modulus 0.8918, read 0.8969 by 24 float32 squarings."""
    H = np.array([[-0.0079, 0.2563, -0.9531, 4.7917, 2.661],
                  [2.3894, -0.7565, -0.7449, 2.5219, -0.4776],
                  [0.0, 3.3457, -0.0899, 1.2746, -0.6037],
                  [0.0, 0.0, 3.2836, -2.0046, -1.6416],
                  [0.0, 0.0, 0.0, 5.2947, 2.8589]])
    true = float(max(abs(np.linalg.eigvals(H))))
    H32 = jnp.asarray(H, jnp.float32)
    read = float(acceleration._spectral_radius(H32))  # noqa: SLF001
    assert read == pytest.approx(true, abs=2e-4), (read, true)
    assert np.isnan(float(acceleration._spectral_radius(H32.at[0, 0].set(jnp.nan))))  # noqa: SLF001
    # The fallback where no eigensolver lowers is the squaring, and the
    # switch is read: this is the number the estimate used to report.
    monkeypatch.setattr(acceleration, "_EIGVALS_BACKENDS", ())
    squared = float(acceleration._spectral_radius(H32))  # noqa: SLF001
    assert squared == float(acceleration._spectral_radius_small(H32))  # noqa: SLF001
    assert abs(squared - true) > 2e-3, (squared, true)


def test_what_a_breakdown_discards_is_reported_as_residual():
    """A direction below the products' rounding is discarded, and its size is not lost.

    ``noise_eps`` says the products round at 1e-3 (a 16-bit map analysed
    in float32), so a direction 1e-3 of the largest product is rounding
    to the test -- and the Arnoldi residual is at least its size.
    """
    A = np.array([[0.5, 1e-3], [0.0, 0.25]])
    rho, res, _amp, _ok = _estimate(A, "float32", v0=np.array([0.0, 1.0]), noise_eps=1e-3)
    assert rho == pytest.approx(0.25, abs=1e-6) and res >= 0.99e-3, (rho, res)
    rho, res, _amp, _ok = _estimate(A, "float32", v0=np.array([0.0, 1.0]))
    assert rho == pytest.approx(0.5, abs=1e-6) and res < 1e-5, (rho, res)


@pytest.mark.parametrize("seed", [90, 130, 131, 241, 270])
def test_one_samples_signs_do_not_settle_a_radius_rounding_moves(seed):
    """Five float32 hubs a single probe of the measured disagreement read settled and wrong.

    The disagreement of one fresh product is one sample of rounding: its
    magnitudes are the measurement, its signs are luck.  Probed along it
    alone, these radii read settled 12% to 60% of themselves off; under
    eight sign patterns each is refused or right.
    """
    A, dtype = _hub(seed)
    true = float(max(abs(np.linalg.eigvals(A))))
    rho, _res, _amp, ok = _estimate(A, dtype)
    assert not ok or abs(rho - true) <= SPECTRAL_SETTLED_FRACTION * (1.0 - true), (rho, true)


def test_the_reorthogonalisation_test_alone_stops_at_a_full_space(monkeypatch):
    """With the noise threshold off, a full space still ends the basis.

    Six dimensions, eight steps: after six the leftover of a product is
    the orthogonalisation's own rounding, which the second sweep removes
    again.  Normalised into a seventh "basis vector" it is orthogonal to
    nothing and the Ritz values are no one's.
    """
    monkeypatch.setattr(acceleration, "_ARNOLDI_NOISE_ULPS", 0.0)
    _compiled.cache_clear()
    try:
        for seed in range(8):
            A, dtype = _normal(seed)
            true = float(max(abs(np.linalg.eigvals(A))))
            rho, res, _amp, ok = _estimate(A, dtype)
            assert rho == pytest.approx(true, abs=1e-5) and ok and res < 1e-4, (seed, rho, true, res)
    finally:
        _compiled.cache_clear()


def test_a_part_of_the_residual_far_above_rounding_is_taken_into_the_space():
    """1e-7 of the residual, along a mode the start does not excite, is most of the distance.

    ``A = diag(1 - 1e-9, 0.5, 0.2)`` from a start with no first component:
    the space is the last two modes.  The residual's first component is
    1e-7 of it and is amplified 1e9 times; left outside the space as
    "rounding" (it is eight orders above float64's), the factor applied to
    the residual was 2.
    """
    A = np.diag([1.0 - 1e-9, 0.5, 0.2])
    r = np.array([1e-7, 1.0, 1.0])
    exact = float(np.linalg.norm(np.linalg.solve(np.eye(3) - A, r)))
    _rho, _res, amp, _ok = _estimate(A, "float64", v0=np.array([0.0, 1.0, 1.0]), v_extra=r)
    assert amp * float(np.linalg.norm(r)) >= exact * (1.0 - 1e-6), (amp, exact)


@pytest.mark.parametrize("family", ["hub", "ring"])
def test_the_estimate_on_a_reading_follows_the_same_rule(family):
    """``_arnoldi_through`` with the identity for a reading is the same estimate.

    The interface norm's spectrum is taken through preimages; its
    breakdown test, its one more product and what it reports are the
    state's, so on the identity the two agree: the same verdicts, and no
    settled radius outside the margin.
    """
    build, _least = FAMILIES[family]
    through = jax.jit(lambda A, v0: acceleration._arnoldi_through(  # noqa: SLF001
        lambda v: A @ v, lambda v: v, v0))
    for seed in SEEDS:
        A, dtype = build(seed)
        true = float(max(abs(np.linalg.eigvals(A))))
        Aj = jnp.asarray(A, dtype)
        v0 = jax.random.normal(jax.random.PRNGKey(0), (A.shape[0],), Aj.dtype)
        rho, res, _amp = through(Aj, v0)
        ok = bool(spectral_rate_settled(rho, res))
        assert not ok or abs(float(rho) - true) <= SPECTRAL_SETTLED_FRACTION * (1.0 - true), (
            seed, float(rho), true)
    with precision(True):
        A64 = jnp.asarray(np.array([[0.3, 1e-8], [1e6, 0.3]]), jnp.float64)
        rho, _res, _amp = acceleration._arnoldi_through(  # noqa: SLF001
            lambda v: A64 @ v, lambda v: v, jnp.asarray([0.0, 1.0], jnp.float64))
        assert float(rho) == pytest.approx(0.4, abs=1e-9)


def test_the_products_rounding_is_the_coarsest_fields_not_the_analysis(monkeypatch):
    """A 16-bit group's products round at its own ``eps``, and the estimate is told so."""
    from tests.core.test_coupling_sixteen_bit_groups import _rotation

    seen = []
    real = acceleration.arnoldi_spectral_radius

    def recording(matvec, v0, *args, noise_eps=None, **kwargs):
        seen.append(noise_eps)
        return real(matvec, v0, *args, noise_eps=noise_eps, **kwargs)

    monkeypatch.setattr(acceleration, "arnoldi_spectral_radius", recording)
    gm = _rotation(jnp.bfloat16, 9, diagnostics=True)
    gm.step()
    assert seen and all(e == float(jnp.finfo(jnp.bfloat16).eps) for e in seen), seen
