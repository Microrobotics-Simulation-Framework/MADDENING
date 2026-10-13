"""MADD-VER-001: HeatNode analytical verification benchmark.

Compares the HeatNode (explicit finite difference) against the exact Fourier
series solution for 1D heat conduction on a rod with Dirichlet BCs:

    T(x,0)   = sin(pi * x / L)     (initial condition)
    T(0, t)  = 0                    (left BC)
    T(L, t)  = 0                    (right BC)

Exact solution:

    T(x,t) = sin(pi * x / L) * exp(-alpha * (pi/L)^2 * t)

**History**: until 0.4.0 the HeatNode applied Dirichlet BCs by overwriting
the boundary cell values.  Cell centres are at dx/2 and L-dx/2, not at x=0
and x=L, so that imposed the boundary condition half a cell inside the rod --
an O(dx) error in the boundary data that propagated inward and held the
global convergence rate at ~1 instead of the 2nd order of the interior
stencil.  That is MADD-ANO-007 (not MADD-ANO-002, which these tests used to
cite and which is about CFL enforcement), and it is fixed: the datum is now
imposed through the ghost cell at the rod end.

Both benchmarks here had acceptance criteria sized around the defect rather
than around the theory -- MADD-VER-001 admitted 5% where the error is now
0.0016%, and MADD-VER-002 admitted any convergence rate between 0.7 and 2.5,
which is to say it admitted the order-1 result it was measuring.  Both bands
are re-derived below from what the corrected scheme actually does, with the
margin stated.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.heat import HeatNode
from maddening.core.compliance.validation import (
    verification_benchmark,
    BenchmarkType,
    _BENCHMARK_REGISTRY,
)


# --- Analytical solution -------------------------------------------------

def heat_analytical(x: np.ndarray, t: float, L: float, alpha: float) -> np.ndarray:
    """Exact solution for T(x,0)=sin(pi*x/L), T(0,t)=T(L,t)=0."""
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * t)


# --- Benchmark tests ------------------------------------------------------


@verification_benchmark(
    benchmark_id="MADD-VER-001",
    description=(
        "HeatNode 1D diffusion vs analytical Fourier solution: "
        "sin(pi x/L) initial condition with zero Dirichlet BCs"
    ),
    node_type="HeatNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "L2 relative error < 1e-4 after 100 steps at CFL=0.25 (n=50); "
        "measured 1.59e-5, a factor of 6.3 of margin.  Was < 5% before "
        "0.4.0, when the boundary-cell overwrite (MADD-ANO-007) put the "
        "error at 1.8% -- a threshold 2.8x the defect it was covering."
    ),
    references=(
        "Crank1975: The Mathematics of Diffusion, Ch. 2",
    ),
)
def test_heat_fourier_benchmark():
    """Run HeatNode for 100 steps and compare against exact solution.

    The threshold is 1e-4 against a measured 1.59e-5.  It was 5% while
    the boundary-cell overwrite held the error at 1.8% (MADD-ANO-007);
    with the overwrite gone the error fell by a factor of 1100 and a 5%
    threshold would no longer be able to fail.
    """
    n_cells = 50
    L = 1.0
    alpha = 0.01

    dx = L / n_cells
    CFL = 0.25
    dt = CFL * dx * dx / alpha

    n_steps = 100
    t_final = n_steps * dt

    # Cell-centre positions
    x = np.linspace(dx / 2, L - dx / 2, n_cells)
    T0 = np.sin(np.pi * x / L)

    node = HeatNode(
        "heat_bench",
        timestep=dt,
        n_cells=n_cells,
        length=L,
        thermal_diffusivity=alpha,
        initial_temperature=T0,
    )

    state = node.initial_state()
    for _ in range(n_steps):
        state = node.update(
            state,
            {"left_temperature": jnp.float32(0.0), "right_temperature": jnp.float32(0.0)},
            dt,
        )

    T_numerical = np.array(state["temperature"])
    T_exact = heat_analytical(x, t_final, L, alpha)

    # L2 relative error.  With the Dirichlet data imposed at the rod ends
    # the scheme is 2nd-order globally, and at n=50 / CFL=0.25 / 100 steps
    # the error is 1.59e-5.
    l2_error = np.sqrt(np.sum((T_numerical - T_exact) ** 2) / np.sum(T_exact ** 2))
    assert l2_error < 1e-4, (
        f"L2 relative error {l2_error:.3e} exceeds the 1e-4 threshold; the "
        f"corrected scheme measures 1.59e-5 here, so anything near 1e-4 "
        f"means the boundary treatment has regressed (MADD-ANO-007)"
    )


@verification_benchmark(
    benchmark_id="MADD-VER-002",
    description=(
        "HeatNode global spatial convergence study against the Fourier "
        "solution: sin(pi x/L) decaying under zero Dirichlet BCs, grid "
        "refined at fixed CFL over 20/40/80 cells in the shipping float32 "
        "precision"
    ),
    node_type="HeatNode",
    benchmark_type=BenchmarkType.CONVERGENCE_STUDY,
    acceptance_criteria=(
        "Mean of the pairwise global L2 convergence rates over a 20/40/80 "
        "ladder at CFL=0.25 within [1.7, 2.3] of the theoretical 2.0 "
        "(measured: 1.900), and the error strictly decreasing.  Before "
        "0.4.0 this study measured ~1.0 -- the boundary-cell overwrite of "
        "MADD-ANO-007 -- and its band had been widened to [0.7, 2.5], which "
        "admitted that result; the band now rejects it by 0.7."
    ),
    references=(
        "LeVeque2007: Finite Difference Methods for ODEs and PDEs",
    ),
)
def test_heat_spatial_convergence():
    """Refine the grid and verify the node converges at 2nd order globally.

    Holds CFL fixed at 0.25 so dt shrinks as dx^2, which keeps the
    forward-Euler error O(dx^2) too and lets a single rate stand for
    the scheme.

    What this benchmark used to say, and why it is worth reading twice:
    it asserted ``0.7 < avg_rate < 2.5`` and its acceptance criteria
    claimed a rate "between 1.5 and 2.5" for the *interior* cells, while
    the code measured the *global* rate.  The three disagreed, the
    executed band was the widest of them, and it was wide enough to
    accept the ~1.0 the node was actually producing.  It also blamed
    MADD-ANO-002, which is about CFL enforcement and has nothing to do
    with a boundary placement.  A verification benchmark widened until
    it accepts a broken result is worse than no benchmark: it is cited
    as evidence.

    The band is now [1.7, 2.3] around the theoretical 2.0:

    * measured 1.900 (pairwise 1.856 and 1.944), so 0.20 of margin
      below and 0.40 above;
    * across CFL in {0.2, 0.25, 0.3, 0.4} and t_final in {0.3, 0.5,
      0.8} the mean spans [1.73, 2.10], so the band is not brittle to
      the study's incidental settings;
    * the pre-0.4.0 result of ~1.0 is rejected by 0.7, which is the
      whole point of the number;
    * on the high side it rejects a ladder that has run out of signal:
      adding n=160 makes the finest pair read 3.63 and the mean 2.48,
      because at 160 cells the error is 2.6e-7 and this study runs in
      the shipping float32 precision, whose relative noise floor for a
      unit-amplitude field is ~1e-7.  80 cells is where this ladder
      stops meaning anything; the float64 ladder that goes further is
      MADD-VER-005.
    """
    L = 1.0
    alpha = 0.01
    CFL = 0.25
    t_final = 0.5

    resolutions = [20, 40, 80]
    global_errors = []

    for n_cells in resolutions:
        dx = L / n_cells
        dt = CFL * dx * dx / alpha
        n_steps = int(t_final / dt)

        x = np.linspace(dx / 2, L - dx / 2, n_cells)
        T0 = np.sin(np.pi * x / L)

        node = HeatNode(
            "conv_test",
            timestep=dt,
            n_cells=n_cells,
            length=L,
            thermal_diffusivity=alpha,
            initial_temperature=T0,
        )
        state = node.initial_state()
        for _ in range(n_steps):
            state = node.update(
                state,
                {"left_temperature": jnp.float32(0.0), "right_temperature": jnp.float32(0.0)},
                dt,
            )

        T_num = np.array(state["temperature"])
        actual_t = n_steps * dt
        T_exact = heat_analytical(x, actual_t, L, alpha)

        l2 = np.sqrt(np.sum((T_num - T_exact) ** 2) / np.sum(T_exact ** 2))
        global_errors.append(l2)

    # Convergence rates between successive refinements
    rates = []
    for i in range(len(global_errors) - 1):
        rate = np.log(global_errors[i] / global_errors[i + 1]) / np.log(2.0)
        rates.append(rate)

    avg_rate = np.mean(rates)

    assert 1.7 < avg_rate < 2.3, (
        f"Convergence rate {avg_rate:.2f} outside [1.7, 2.3] around the "
        f"theoretical 2.0.  A rate near 1 is a boundary condition imposed "
        f"in the wrong place (MADD-ANO-007), not a tolerance to widen; a "
        f"rate well above 2 is a ladder that has reached the float32 noise "
        f"floor.  "
        f"Rates: {[f'{r:.2f}' for r in rates]}, "
        f"Errors: {[f'{e:.2e}' for e in global_errors]}"
    )

    # Errors must decrease monotonically (basic sanity)
    for i in range(len(global_errors) - 1):
        assert global_errors[i + 1] < global_errors[i], (
            f"Error did not decrease from n={resolutions[i]} to n={resolutions[i+1]}"
        )


class TestBenchmarkRegistration:
    """Verify the benchmarks were properly registered."""

    def test_ver001_registered(self):
        assert "MADD-VER-001" in _BENCHMARK_REGISTRY

    def test_ver002_registered(self):
        assert "MADD-VER-002" in _BENCHMARK_REGISTRY

    def test_ver001_metadata(self):
        bm = _BENCHMARK_REGISTRY["MADD-VER-001"]
        assert bm.node_type == "HeatNode"
        assert bm.benchmark_type == BenchmarkType.ANALYTICAL

    def test_ver002_metadata(self):
        bm = _BENCHMARK_REGISTRY["MADD-VER-002"]
        assert bm.node_type == "HeatNode"
        assert bm.benchmark_type == BenchmarkType.CONVERGENCE_STUDY
