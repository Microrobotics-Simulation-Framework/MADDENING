"""MADD-VER-016: pressure-driven Poiseuille flow through LBMNode's Zou-He faces.

MADD-VER-003 and MADD-VER-007 drive LBMNode with a body force.  Neither
passes through the pressure boundary, and the only in-tree test that did
(``TestLBMPoiseuille2D``) divided each velocity profile by its own maximum
before comparing it, so it could see the *shape* of the flow and nothing
about its size.  That is how the Zou-He closure could impose 0.4156 for a
prescribed 0.36 in every release up to 0.3.1 (MADD-ANO-020): on the
64x16 channel of that test the steady face-to-face pressure drop came out
at 0.0019 against 0.0033 imposed, and the centreline velocity at 0.48x
what the imposed drop gives.

The tests here read magnitudes, at steady state, on a 2-D channel with
bounce-back walls at ``y = 0`` and ``y = ny - 1`` (``H = ny - 2`` fluid
rows) and the inlet/outlet pressures on the ``x`` faces, ``L = nx - 1``
apart:

* the face-to-face pressure drop, and the pressure gradient the channel
  actually carries in its middle half, against the imposed ``dp / L``;
* the centreline velocity against the same flow driven by the equivalent
  body force ``F = dp / L`` on a periodic channel with the same walls --
  which isolates the pressure boundary from the wall treatment;
* the centreline velocity against Hagen-Poiseuille,
  ``u_c = (dp / L) H^2 / (8 rho nu)``, in absolute terms, under grid
  refinement.

On the last one.  LBMNode's walls are bounce-back on wall *cells*, which
the node documents as first order; measured here, the hydrodynamic wall
sits about 0.09 lattice units from the wall node (0.0968 at H = 8, 0.0877
at 16, 0.0802 at 32, at tau = 1) rather than on the half-way plane at
0.5, so at a finite ``H`` the channel is about 0.8 lattice units wider
than nominal and ``u_c`` is high by a first-order amount (+20.7% at
H = 8, +10.4% at 16, +4.9% at 32; observed order 1.00 and 1.07).  That
is a property of the walls, not of the pressure boundary, and it is
the same with body-force driving (+19.8%, +10.1%, +4.5%).  So the
absolute check extrapolates the two-level ladder at the walls' declared
first order and requires the limit to be Hagen-Poiseuille: measured
1.0001.  Before the fix the same ladder read 0.920 at H = 8 and 0.658 at
H = 16 -- moving *away* from 1 under refinement, limit 0.396 -- with
face-to-face drops of 0.80 and 0.65 of the imposed one.  A single coarse
level with a loose band would have passed it: 0.920 is within 8%.

Configuration: float32 (the node's default), tau = 1 (nu = 1/6),
L = 4H, diffusive scaling (``dp`` falls as ``1/H^2`` so the Mach number
falls as ``1/H``), 3000 steps at H = 8 and 6000 at H = 16.  The
centreline velocity has settled to about 1e-4 relative by 5000 steps at
H = 16 and wanders by about 1e-5 after that (float32 and the slowly
decaying acoustic modes between two reflecting pressure faces).  The
three steady states take about 14 s together, compilation included, on
four CPU cores; H = 32 would add about two minutes and is not run here.
jaxlib 0.11.0.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.compliance.validation import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkType,
    verification_benchmark,
)
from maddening.nodes.lbm import LBMNode  # noqa: E402

CS2 = 1.0 / 3.0
NU = 1.0 / 6.0            # tau = 1
ASPECT = 4                # L = 4 H
DRHO_AT_H8 = 0.01         # inlet/outlet density difference at H = 8
STEPS = {8: 3000, 16: 6000}


def _channel(H):
    ny, nx = H + 2, ASPECT * H + 1        # nx odd: x = nx // 2 is the midpoint
    wall = np.zeros((nx, ny), bool)
    wall[:, 0] = wall[:, -1] = True
    return nx, ny, wall


def _run(node, boundary, steps):
    state = node.initial_state()
    advance = jax.jit(
        lambda s: jax.lax.fori_loop(0, steps, lambda _, x: node.update(x, boundary, 1.0), s)
    )
    return jax.device_get(advance(state))


def _pressure_driven(H):
    nx, ny, wall = _channel(H)
    drho = DRHO_AT_H8 * (8.0 / H) ** 2
    rho_in, rho_out = 1.0 + drho / 2, 1.0 - drho / 2
    node = LBMNode("channel", 1.0, grid_shape=(nx, ny), viscosity=NU,
                   lattice="D2Q9", wall_mask=wall)
    boundary = {"inlet_pressure": jnp.float32(rho_in * CS2),
                "outlet_pressure": jnp.float32(rho_out * CS2)}
    state = _run(node, boundary, STEPS[H])
    return dict(H=H, nx=nx, ny=ny, L=nx - 1, p_in=rho_in * CS2, p_out=rho_out * CS2,
                dp=drho * CS2, state=state)


def _body_force_driven(H, dp, L):
    """The same walls and viscosity, periodic in x, driven by F = dp / L."""
    _, ny, wall = _channel(H)
    nx = 4
    force = np.zeros((nx, ny, 2), np.float32)
    force[:, 1:-1, 0] = dp / L
    node = LBMNode("periodic", 1.0, grid_shape=(nx, ny), viscosity=NU,
                   lattice="D2Q9", wall_mask=wall[:nx])
    return _run(node, {"body_force": jnp.asarray(force)}, STEPS[H])


def _centreline(u_x_row, ny):
    return float(np.interp((ny - 1) / 2.0, np.arange(ny), u_x_row))


@pytest.fixture(scope="module")
def channels():
    """Steady states shared by the tests below (module scope, at module
    level: a class-scoped fixture written as a method errors on pytest 9.1)."""
    runs = {H: _pressure_driven(H) for H in STEPS}
    fine = runs[16]
    runs["body16"] = _body_force_driven(16, fine["dp"], fine["L"])
    return runs


def _hagen_poiseuille_ratio(run):
    """Measured centreline velocity over (dp/L) H^2 / (8 rho_mid nu).

    ``rho_mid`` is the density at mid-channel: the mass flux rho*u is
    conserved along a weakly compressible channel and mu = rho*nu, so the
    pressure gradient stays uniform while u varies as 1/rho."""
    xm, ny, H = run["nx"] // 2, run["ny"], run["H"]
    rho_mid = float(np.asarray(run["state"]["density"])[xm, 1:-1].mean())
    u_c = _centreline(np.asarray(run["state"]["velocity"])[xm, :, 0], ny)
    return u_c / (run["dp"] / run["L"] * H**2 / (8.0 * rho_mid * NU))


@pytest.mark.slow  # the channels fixture these three share: 6-7 s on CI
@verification_benchmark(
    benchmark_id="MADD-VER-016",
    description=(
        "LBMNode pressure-driven Poiseuille flow: a 2-D D2Q9 channel with "
        "bounce-back walls driven only by the Zou-He inlet/outlet pressure "
        "boundary, refined under diffusive scaling, against Hagen-Poiseuille "
        "in absolute terms"
    ),
    node_type="LBMNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "Centreline velocity over Hagen-Poiseuille (dp/L) H^2 / (8 rho nu) at "
        "H = 8 and 16 (L = 4H, tau = 1, float32) falls under refinement, and "
        "its extrapolation at the walls' declared first order, 2 r(16) - "
        "r(8), is within 1% of 1 (measured r = 1.2071, 1.1036; limit "
        "1.0001).  The finite-H excess is the first-order bounce-back wall "
        "(hydrodynamic wall about 0.09 lattice units from the wall node) and "
        "is the same under body-force driving.  Before 0.4.0 the Zou-He "
        "closure imposed the wrong face density (MADD-ANO-020) and the "
        "ladder read 0.920, 0.658: away from 1, limit 0.396."
    ),
    references=(
        "ZouHe1997: On pressure and velocity boundary conditions for the lattice Boltzmann BGK model",
        "Kruger2017: The Lattice Boltzmann Method: Principles and Practice",
    ),
)
def test_pressure_driven_poiseuille_converges_to_hagen_poiseuille(channels):
    r8 = _hagen_poiseuille_ratio(channels[8])
    r16 = _hagen_poiseuille_ratio(channels[16])
    assert abs(r16 - 1.0) < abs(r8 - 1.0), (r8, r16)
    limit = 2.0 * r16 - r8
    assert limit == pytest.approx(1.0, abs=0.01), (
        f"u_c / u_HP = {r8:.5f} (H=8), {r16:.5f} (H=16); first-order limit {limit:.5f}"
    )


@pytest.mark.slow  # the channels fixture these three share: 6-7 s on CI
@pytest.mark.parametrize("H,gradient_tol", [(8, 0.02), (16, 0.01)])
def test_the_channel_carries_the_imposed_pressure_drop(channels, H, gradient_tol):
    """The face pressures are the imposed ones to float32 rounding, and the
    gradient in the middle half of the channel is dp / L to within the
    entrance effect of the faces (measured +0.91% at H = 8, +0.18% at 16).
    Before the fix, at H = 16: 0.65 of the imposed drop face to face and
    0.60 of the imposed gradient in the middle half."""
    run = channels[H]
    p = np.asarray(run["state"]["pressure"], np.float64)[:, 1:-1].mean(axis=1)
    assert p[0] == pytest.approx(run["p_in"], rel=2e-6)
    assert p[-1] == pytest.approx(run["p_out"], rel=2e-6)
    assert p[0] - p[-1] == pytest.approx(run["dp"], rel=1e-3)
    nx = run["nx"]
    mid = slice(nx // 4, 3 * nx // 4 + 1)
    slope = np.polyfit(np.arange(nx)[mid], p[mid], 1)[0]
    assert -slope * run["L"] / run["dp"] == pytest.approx(1.0, abs=gradient_tol)


@pytest.mark.slow  # the channels fixture these three share: 6-7 s on CI
def test_pressure_driving_and_the_equivalent_body_force_give_the_same_flow(channels):
    """Same walls, same viscosity: a pressure drop dp over L and a body
    force dp / L must drive the same profile.  The wall treatment cancels,
    so this isolates the pressure boundary.  Measured: centreline ratio
    1.0021 (rho-corrected), largest profile difference 0.21% of u_c."""
    run = channels[16]
    ny, xm = run["ny"], run["nx"] // 2
    rho_mid = float(np.asarray(run["state"]["density"])[xm, 1:-1].mean())
    u_p = np.asarray(run["state"]["velocity"], np.float64)[xm, :, 0] * rho_mid
    u_b = np.asarray(channels["body16"]["velocity"], np.float64)[2, :, 0]
    u_c_b = _centreline(u_b, ny)
    assert _centreline(u_p, ny) / u_c_b == pytest.approx(1.0, abs=0.01)
    assert np.max(np.abs(u_p[1:-1] - u_b[1:-1])) / u_c_b < 0.01


def test_the_benchmark_is_registered():
    benchmark = _BENCHMARK_REGISTRY["MADD-VER-016"]
    assert benchmark.node_type == "LBMNode"
    assert benchmark.benchmark_type is BenchmarkType.ANALYTICAL
    assert benchmark.test_function.endswith(
        "test_pressure_driven_poiseuille_converges_to_hagen_poiseuille"
    )
