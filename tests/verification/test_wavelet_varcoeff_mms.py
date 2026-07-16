"""M23 — forward-model validation of the variable-coefficient wavelet solver.

Manufactured-solution (MMS) convergence for the magnetostatics operator
``-∇·((1+χ)∇φ) + mφ = f``.  The source ``f = L[φ_exact]`` is derived from the
**continuum** PDE (analytic derivatives), not from the wavelet code — an
independent ground truth that catches operator-assembly and RHS-scaling errors a
self-consistency check cannot (it caught exactly such an ``h^dim`` scaling bug in
the varcoeff RHS during M23).

This validates the variable-coefficient operator + RHS at contrast up to 10²
(the near-term scope; χ ≥ 10³ is the R1 research track).  float64 required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import BenchmarkType, verification_benchmark
from maddening.nodes.adaptive.wavelets import operator as OP


@pytest.fixture(autouse=True)
def _x64():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def _mms_error_1d(n_levels, A, mass=0.5, n_coarse=2):
    """Full-basis wavelet solve of -d/dx((1+χ)dφ/dx)+mφ = f vs the manufactured
    φ_exact = cos(2πx), χ = A(1-cos 2πx)/2 (χ≥0, contrast A).  Source analytic."""
    side = n_coarse * 2 ** n_levels
    h = 1.0 / side
    x = np.arange(side) / side
    tp = 2 * np.pi
    phi_ex = np.cos(tp * x)
    chi = A * (1 - np.cos(tp * x)) / 2
    a = 1.0 + chi
    dphi = -tp * np.sin(tp * x)
    d2phi = -tp ** 2 * np.cos(tp * x)
    dchi = A * (tp * np.sin(tp * x)) / 2
    f = -(dchi * dphi + a * d2phi) + mass * phi_ex          # continuum L[φ_exact]
    res = OP.assemble_wave_operator(n_levels, n_coarse, 4, 1, mass=mass,
                                    a_grid=jnp.asarray(a))
    Awave, Wn = res["A_dense"], res["Wn"]
    b = Wn.T @ jnp.asarray(f)                                # strong form: no h^dim
    phi_h = np.asarray(Wn @ jnp.linalg.solve(Awave, b))
    return side, np.linalg.norm(phi_h - phi_ex) / np.linalg.norm(phi_ex)


@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-VARCOEFF-MMS",
    description=(
        "Variable-coefficient wavelet elliptic solver (WaveletVarcoeffNode "
        "operator) vs a manufactured solution of -∇·((1+χ)∇φ)+mφ=f with the "
        "source derived analytically from the continuum PDE. Validates the "
        "varcoeff operator + RHS at contrast up to 100 (near-term scope). "
        "Full-basis solve (isolates discretisation error from CDD truncation)."
    ),
    node_type="WaveletVarcoeffNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "relL2 < 1e-3 at N=256 and convergence rate > 1.9 at contrast χ ∈ "
        "{0, 10, 100}"
    ),
)
def test_varcoeff_mms_converges():
    for A in (0.0, 10.0, 100.0):
        errs, sides = [], []
        for nl in (4, 5, 6, 7):
            s, e = _mms_error_1d(nl, A)
            sides.append(s)
            errs.append(e)
        rate = np.log2(errs[-2] / errs[-1])
        assert errs[2] < 1e-3, f"A={A}: N={sides[2]} relL2={errs[2]}"
        assert rate > 1.9, f"A={A}: convergence rate {rate}"


def test_varcoeff_mms_2d():
    """2D MMS at contrast 10 — the operator is correct in 2D too."""
    def err(nl, A=10.0, mass=0.5, nc=2):
        side = nc * 2 ** nl
        h = 1.0 / side
        c1 = np.arange(side) / side
        X, Y = np.meshgrid(c1, c1, indexing="ij")
        tp = 2 * np.pi
        phi_ex = np.cos(tp * X) * np.cos(tp * Y)
        chi = A * (1 - np.cos(tp * X) * np.cos(tp * Y)) / 2
        a = 1.0 + chi
        # continuum L[φ] = -∇·(a∇φ) + mφ, analytic
        ax = A * tp * np.sin(tp * X) * np.cos(tp * Y) / 2      # ∂χ/∂x
        ay = A * tp * np.cos(tp * X) * np.sin(tp * Y) / 2      # ∂χ/∂y
        px = -tp * np.sin(tp * X) * np.cos(tp * Y)             # ∂φ/∂x
        py = -tp * np.cos(tp * X) * np.sin(tp * Y)             # ∂φ/∂y
        lap = -2 * tp ** 2 * np.cos(tp * X) * np.cos(tp * Y)   # Δφ
        f = -(ax * px + ay * py + a * lap) + mass * phi_ex
        res = OP.assemble_wave_operator(nl, nc, 4, 2, mass=mass,
                                        a_grid=jnp.asarray(a))
        Awave, Wn = res["A_dense"], res["Wn"]
        b = Wn.T @ jnp.asarray(f.reshape(-1))
        phi_h = np.asarray(Wn @ jnp.linalg.solve(Awave, b))
        return np.linalg.norm(phi_h - phi_ex.reshape(-1)) / np.linalg.norm(phi_ex)
    e4, e5 = err(4), err(5)
    assert e5 < 5e-3, f"2D relL2={e5}"
    assert np.log2(e4 / e5) > 1.8, f"2D rate {np.log2(e4/e5)}"
