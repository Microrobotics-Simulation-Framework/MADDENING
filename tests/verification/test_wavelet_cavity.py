"""M5 part 2 — lid-driven cavity Ghia benchmark (slow lane).

Two tests over ``benchmarks/wavelet_cavity.py``, deliberately kept separate
because they validate **different things and carry different weight**:

1. :func:`test_cavity_re100_matches_ghia` — the registered verification
   benchmark.  It validates the NumPy/SciPy **finite-difference** ψ-ω reference
   solver against Ghia-Ghia-Shin (1982).  ``WaveletAdaptiveNode`` is *not*
   involved: convection is explicit Euler in NumPy and the ψ-solve is
   ``scipy.lu_factor``/``lu_solve``.  It is registered under the FD reference,
   not under any node.

2. :func:`test_wavelet_dirichlet_basis_reproduces_fd_psi` — a plain regression
   guard, intentionally **not** a registered benchmark.  It re-solves the
   ψ-Poisson on the already-converged vorticity in the DD-wavelet Dirichlet
   basis.  That is an exact change of basis, so agreement to machine precision
   is a linear-algebra identity, not evidence of physical correctness.  See
   its docstring.

Marked ``slow`` (run with ``pytest -m slow``); ~70 s at 47².
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from maddening.core.compliance.validation import BenchmarkType, verification_benchmark

# benchmarks/ is not a package; add it to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import wavelet_cavity as WC  # noqa: E402


@pytest.fixture(scope="module")
def converged_cavity():
    """Run the FD cavity to steady state once and share it across tests."""
    return WC.run_cavity(nl=4, nc=2, Re=100.0, dt=0.002, nsteps=30000, tol=1e-6)


@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MADD-VER-CAVITY-FD-100",
    description=(
        "Lid-driven cavity at Re=100 solved by the NumPy/SciPy "
        "finite-difference stream-function/vorticity reference solver in "
        "benchmarks/wavelet_cavity.py, compared to the Ghia-Ghia-Shin (1982) "
        "centreline velocity tabulation. Validates the FD reference only: no "
        "MADDENING node participates in this benchmark. In particular this is "
        "NOT a validation of WaveletAdaptiveNode, which is a steady scalar "
        "elliptic solver with no convection, velocity, pressure or time "
        "integration, and which is absent from the time loop. The benchmark's "
        "role is to establish that the converged vorticity field used as input "
        "by test_wavelet_dirichlet_basis_reproduces_fd_psi is physically "
        "correct."
    ),
    node_type="(none: benchmarks/wavelet_cavity.py finite-difference reference solver)",
    benchmark_type=BenchmarkType.CROSS_CODE,
    acceptance_criteria=("max |u - Ghia| on the vertical centreline < 0.02; "
                         "min centreline u within 5% of -0.2109; primary vortex "
                         "centre within 0.03 of (0.617, 0.734)"),
    references=("Ghia, Ghia & Shin (1982), J. Comput. Phys. 48(3), 387-411",),
)
def test_cavity_re100_matches_ghia(converged_cavity):
    max_err, vortex, umin = WC.ghia_comparison(converged_cavity)
    # centreline profile
    assert max_err < 0.02, f"max |u - Ghia| = {max_err}"
    # headline min velocity
    assert abs(umin - (-0.21090)) / 0.21090 < 0.05, f"min u = {umin}"
    # primary vortex location
    vx, vy = vortex
    gx, gy = WC.GHIA_RE100_VORTEX
    assert (vx - gx) ** 2 + (vy - gy) ** 2 < 0.03 ** 2, f"vortex {vortex}"


@pytest.mark.slow
def test_wavelet_dirichlet_basis_reproduces_fd_psi(converged_cavity):
    """Regression guard on the Dirichlet wavelet basis — NOT a validation.

    Re-solves ``-∇²ψ = ω`` on the converged vorticity from the FD cavity, in
    the boundary-adapted DD-wavelet Dirichlet basis, and checks it against the
    FD ψ.

    **What this can and cannot show.**  The wavelet operator here is the *same*
    FD Poisson matrix expressed in a different basis (``A = Wnᵀ L Wn``, solved
    densely and un-truncated), so reproducing the FD ψ is an exact change of
    basis — an algebraic identity that holds independently of whether either
    solver is physically right.  It therefore proves nothing about the cavity
    flow, and nothing about ``WaveletAdaptiveNode``, which is not called.

    It is still worth running: it is a real regression guard on
    ``synthesis_matrix_dirichlet`` and the L² normalisation.  A broken
    boundary-adapted basis, a non-invertible ``Wn``, or a normalisation error
    would all break the identity.  That is the whole of its scope, and it is
    deliberately left out of the verification-benchmark registry so it cannot
    be harvested as a validation claim.
    """
    assert WC.wavelet_psi_consistency(converged_cavity) < 1e-8
