"""Forward-model validation of the general variable-coefficient wavelet solver.

Manufactured-solution (MMS) convergence for the general elliptic operator
``-∇·(a(x)∇u) + m u = f``.  The source ``f = L[u_exact]`` is derived from the
**continuum** PDE (analytic derivatives), not from the wavelet code — an
independent ground truth that catches operator-assembly and RHS-scaling errors a
self-consistency check cannot (it caught an ``h^dim`` RHS-scaling bug when this
was first written).

``a(x)`` is a generic varying coefficient (no physics domain); the test spans
coefficient contrast up to 10² to exercise the variable-coefficient path.  float64
required.
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


def _mms_error_1d(n_levels, contrast, mass=0.5, n_coarse=2):
    """Full-basis wavelet solve of -d/dx(a du/dx)+mu=f vs u_exact=cos(2πx),
    a(x) = 1 + C(1-cos 2πx)/2 (a ranges [1, 1+C]).  Source analytic (continuum)."""
    side = n_coarse * 2 ** n_levels
    x = np.arange(side) / side
    tp = 2 * np.pi
    u_ex = np.cos(tp * x)
    a = 1.0 + contrast * (1 - np.cos(tp * x)) / 2
    da = contrast * (tp * np.sin(tp * x)) / 2
    du = -tp * np.sin(tp * x)
    d2u = -tp ** 2 * np.cos(tp * x)
    f = -(da * du + a * d2u) + mass * u_ex                  # continuum L[u_exact]
    res = OP.assemble_wave_operator(n_levels, n_coarse, 4, 1, mass=mass,
                                    a_grid=jnp.asarray(a))
    Awave, Wn = res["A_dense"], res["Wn"]
    b = Wn.T @ jnp.asarray(f)                                # strong form: no h^dim
    u_h = np.asarray(Wn @ jnp.linalg.solve(Awave, b))
    return side, np.linalg.norm(u_h - u_ex) / np.linalg.norm(u_ex)


@verification_benchmark(
    benchmark_id="MADD-VER-WAVELET-ELLIPTIC-MMS",
    description=(
        "General variable-coefficient wavelet elliptic solver "
        "(WaveletEllipticNode operator) vs a manufactured solution of "
        "-∇·(a(x)∇u)+mu=f with a generic varying coefficient a(x) and the source "
        "derived analytically from the continuum PDE. Validates the "
        "variable-coefficient operator + RHS at coefficient contrast up to 100. "
        "Full-basis solve (isolates discretisation error from CDD truncation)."
    ),
    node_type="WaveletEllipticNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "relL2 < 1e-3 at N=256 and convergence rate > 1.9 at coefficient contrast "
        "C ∈ {0, 10, 100}"
    ),
)
def test_elliptic_mms_converges():
    for C in (0.0, 10.0, 100.0):
        errs, sides = [], []
        for nl in (4, 5, 6, 7):
            s, e = _mms_error_1d(nl, C)
            sides.append(s)
            errs.append(e)
        rate = np.log2(errs[-2] / errs[-1])
        assert errs[2] < 1e-3, f"C={C}: N={sides[2]} relL2={errs[2]}"
        assert rate > 1.9, f"C={C}: convergence rate {rate}"


def test_elliptic_mms_2d():
    """2D MMS at coefficient contrast 10 — the operator is correct in 2D too."""
    def err(nl, C=10.0, mass=0.5, nc=2):
        side = nc * 2 ** nl
        c1 = np.arange(side) / side
        X, Y = np.meshgrid(c1, c1, indexing="ij")
        tp = 2 * np.pi
        u_ex = np.cos(tp * X) * np.cos(tp * Y)
        a = 1.0 + C * (1 - np.cos(tp * X) * np.cos(tp * Y)) / 2
        ax = C * tp * np.sin(tp * X) * np.cos(tp * Y) / 2
        ay = C * tp * np.cos(tp * X) * np.sin(tp * Y) / 2
        ux = -tp * np.sin(tp * X) * np.cos(tp * Y)
        uy = -tp * np.cos(tp * X) * np.sin(tp * Y)
        lap = -2 * tp ** 2 * np.cos(tp * X) * np.cos(tp * Y)
        f = -(ax * ux + ay * uy + a * lap) + mass * u_ex
        res = OP.assemble_wave_operator(nl, nc, 4, 2, mass=mass,
                                        a_grid=jnp.asarray(a))
        Awave, Wn = res["A_dense"], res["Wn"]
        b = Wn.T @ jnp.asarray(f.reshape(-1))
        u_h = np.asarray(Wn @ jnp.linalg.solve(Awave, b))
        return np.linalg.norm(u_h - u_ex.reshape(-1)) / np.linalg.norm(u_ex)
    e4, e5 = err(4), err(5)
    assert e5 < 5e-3, f"2D relL2={e5}"
    assert np.log2(e4 / e5) > 1.8, f"2D rate {np.log2(e4/e5)}"
