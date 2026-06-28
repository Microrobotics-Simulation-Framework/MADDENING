"""M5 part 2 — lid-driven cavity Ghia benchmark (slow lane).

Drives the ψ-ω cavity at Re=100 with the wavelet Dirichlet Poisson ψ-solve
(benchmarks/wavelet_cavity.py) and checks the steady-state centreline velocity
against the Ghia-Ghia-Shin (1982) tabulation.  Marked ``slow`` (run with
``pytest -m slow``); ~70 s at 47².
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


@pytest.mark.slow
@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-CAVITY-100",
    description=("Lid-driven cavity Re=100 with a wavelet Dirichlet Poisson "
                 "ψ-solve, vs Ghia-Ghia-Shin (1982) centreline velocity"),
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.CROSS_CODE,
    acceptance_criteria=("max |u - Ghia| on the vertical centreline < 0.02; "
                         "min centreline u within 5% of -0.2109; primary vortex "
                         "centre within 0.03 of (0.617, 0.734)"),
)
def test_cavity_re100_matches_ghia():
    res = WC.run_cavity(nl=4, nc=2, Re=100.0, dt=0.002, nsteps=30000, tol=1e-6)
    max_err, vortex, umin = WC.ghia_comparison(res)
    # centreline profile
    assert max_err < 0.02, f"max |u - Ghia| = {max_err}"
    # headline min velocity
    assert abs(umin - (-0.21090)) / 0.21090 < 0.05, f"min u = {umin}"
    # primary vortex location
    vx, vy = vortex
    gx, gy = WC.GHIA_RE100_VORTEX
    assert (vx - gx) ** 2 + (vy - gy) ** 2 < 0.03 ** 2, f"vortex {vortex}"
    # the wavelet Dirichlet Poisson solver reproduces the FD ψ in-context
    assert WC.wavelet_psi_consistency(res) < 1e-8
