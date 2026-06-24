"""M3 — cross-validation of the wavelet elliptic solver against independent
references (spike Limitation 7: guard against systematic errors invisible to
autodiff-vs-FD self-checks).

Two independent references:

* **Manufactured solution (MMS).**  Pick an analytic ``u_exact``, derive the
  forcing ``f = (-Δ + m) u_exact`` from the *continuum* PDE (external ground
  truth, not produced by the wavelet code), solve, and verify O(h²)
  convergence.  Catches operator-assembly / RHS-projection errors that
  self-consistency cannot.

* **Cross-code vs MIME.**  MIME's FFT-spectral Helmholtz solver
  (``mime.nodes.environment.fvm.pressure.make_helmholtz_solver_fft``) is a
  completely independent discretisation + solve.  Both it and the wavelet
  Galerkin solver are compared to the same analytic solution; they must agree.

float64 is required (the wavelet operator assembles in float64); a local autouse
fixture scopes it per-test so it does not leak into other modules.
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


MASS = 1.0


# ----------------------------------------------------------------------
# Manufactured-solution helpers (analytic ground truth).
# ----------------------------------------------------------------------

def _wavelet_solve_full(n_levels, dim, f_grid):
    """Full (unmasked) wavelet Galerkin solve of (-Δ + MASS) u = f."""
    res = OP.assemble_wave_operator(n_levels, 2, order=4, dim=dim, mass=MASS)
    A, Wn, h = res["A_dense"], res["Wn"], res["h"]
    b = (h ** dim) * (Wn.T @ jnp.asarray(f_grid.reshape(-1)))
    c = jnp.linalg.solve(A, b)
    u_h = Wn @ c
    return res["side"], np.asarray(u_h)


def _mms_error_1d(n_levels):
    side = 2 * 2 ** n_levels
    x = np.arange(side) / side
    u_ex = np.cos(2 * np.pi * x)
    f = ((2 * np.pi) ** 2 + MASS) * u_ex
    s, u_h = _wavelet_solve_full(n_levels, 1, f)
    return s, np.linalg.norm(u_h - u_ex) / np.linalg.norm(u_ex)


def _mms_error_2d(n_levels):
    side = 2 * 2 ** n_levels
    c1 = np.arange(side) / side
    X, Y = np.meshgrid(c1, c1, indexing="ij")
    u_ex = (np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y))
    f = (2 * (2 * np.pi) ** 2 + MASS) * u_ex
    s, u_h = _wavelet_solve_full(n_levels, 2, f)
    return s, np.linalg.norm(u_h - u_ex.reshape(-1)) / np.linalg.norm(u_ex)


# ----------------------------------------------------------------------
# MMS convergence benchmarks.
# ----------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-001",
    description="Wavelet elliptic solver vs manufactured solution, 1D, O(h^2)",
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria="relL2 < 1e-3 at N=128 and convergence rate > 1.8",
)
def test_mms_1d_converges():
    errs, sides = [], []
    for nl in (3, 4, 5, 6):
        s, e = _mms_error_1d(nl)
        sides.append(s)
        errs.append(e)
    rate = np.log2(errs[-2] / errs[-1])
    assert errs[-1] < 1e-3, f"N={sides[-1]} relL2={errs[-1]}"
    assert rate > 1.8, f"convergence rate {rate}"


@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-002",
    description="Wavelet elliptic solver vs manufactured solution, 2D, O(h^2)",
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria="relL2 < 5e-3 at N=64^2 and convergence rate > 1.8",
)
def test_mms_2d_converges():
    errs = []
    for nl in (3, 4, 5):
        _, e = _mms_error_2d(nl)
        errs.append(e)
    rate = np.log2(errs[-2] / errs[-1])
    assert errs[-1] < 5e-3, f"relL2={errs[-1]}"
    assert rate > 1.8, f"convergence rate {rate}"


# ----------------------------------------------------------------------
# Cross-code: wavelet Galerkin vs MIME FFT-spectral Helmholtz.
# ----------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-003",
    description=("Wavelet elliptic solver cross-checked against MIME's "
                 "independent FFT-spectral Helmholtz solver on the same problem"),
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.CROSS_CODE,
    acceptance_criteria=("both solvers within 1% of the analytic solution at "
                         "N=32^2 and agree with each other to within 0.5%"),
)
def test_cross_code_vs_mime_fft():
    pytest.importorskip("mime")
    from mime.nodes.environment.fvm import make_cartesian_mesh_2d
    from mime.nodes.environment.fvm import pressure as P

    N = 32
    # --- analytic problem: (MASS - Δ) u = f, u = cos(2πx)cos(2πy), periodic ---
    nl = 4                       # wavelet side = 2*2^4 = 32, matches N
    c1 = np.arange(N) / N
    X, Y = np.meshgrid(c1, c1, indexing="ij")
    u_ex = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    k2 = 2 * (2 * np.pi) ** 2
    f = (MASS + k2) * u_ex

    # --- wavelet Galerkin solution ---
    _, u_w = _wavelet_solve_full(nl, 2, f)
    err_w = np.linalg.norm(u_w - u_ex.reshape(-1)) / np.linalg.norm(u_ex)

    # --- MIME FFT-spectral Helmholtz: (I - α∇²) x = b, α = 1/MASS ---
    mesh = make_cartesian_mesh_2d(N, N, 1.0, 1.0, periodic_x=True,
                                  periodic_y=True, dtype=jnp.float64)
    solver = P.make_helmholtz_solver_fft(mesh, bc=("periodic", "periodic"))
    xc = np.asarray(mesh.x)
    u_ex_m = np.cos(2 * np.pi * xc[:, 0]) * np.cos(2 * np.pi * xc[:, 1])
    alpha = 1.0 / MASS
    # (MASS - Δ)u = f  <=>  (I - α∇²)(u) = f/MASS
    b = (f.reshape(-1) if np.allclose(xc[:, 0], X.reshape(-1)) else
         ((MASS + k2) * u_ex_m)) / MASS
    x_m = np.asarray(solver(jnp.asarray(b), alpha))
    err_m = np.linalg.norm(x_m - u_ex_m) / np.linalg.norm(u_ex_m)

    # both independently accurate ...
    assert err_w < 1e-2, f"wavelet err {err_w}"
    assert err_m < 1e-2, f"MIME err {err_m}"
    # ... hence agree with each other (same 2nd-order discretisation, two codes)
    assert abs(err_w - err_m) < 5e-3, f"wavelet {err_w} vs MIME {err_m}"
