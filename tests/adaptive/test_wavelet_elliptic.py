"""WaveletEllipticNode — general matrix-free adaptive elliptic solver.

Solves -∇·(a(x)∇u)+mu=f with the coefficient field a and source f supplied by the
caller (θ→a and/or θ→b). No physics domain appears here — the node is exercised
purely as a general differentiable solver. The conftest.py autouse fixture
provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive import WaveletEllipticNode
from maddening.nodes.adaptive.wavelets import sensors as SEN


def _frozen_J(node, theta0, sensor=None):
    """Sensor as a function of θ at a mask frozen at θ0 (the production adjoint)."""
    N = node.N_max
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, bool), "theta": theta0}
    mask = jax.lax.stop_gradient(node.compute_active_set(empty, is_cold_start=True))

    def J(theta):
        st = node.solve_frozen({**empty, "theta": theta, "mask": mask}, mask)
        obs = (sensor.observe(st["c"]) if sensor is not None
               else jnp.squeeze(node._sensor(st)))
        return jnp.sum(obs ** 2) if sensor is not None else obs

    return J


# ----------------------------------------------------------------------
# construction / cold start / update / metadata
# ----------------------------------------------------------------------

def test_construct_cold_start_and_update():
    node = WaveletEllipticNode(dim=2, n_levels=3, n_coarse=2, mass=1.0)
    s = node.initial_state()
    assert s["c"].shape == (node.N_max,)
    assert int(s["mask"].sum()) >= int(jnp.asarray(node._coarse).sum())
    s2 = node.update(s, {}, 1.0)
    assert s2["c"].shape == (node.N_max,)
    assert bool(jnp.all(jnp.isfinite(s2["c"])))


def test_metadata_is_domain_neutral():
    from maddening.core.compliance.metadata import StabilityLevel
    meta = WaveletEllipticNode.meta
    assert meta.algorithm_id == "MADD-NODE-WAVELET-ELLIPTIC"
    blob = " ".join((meta.description, meta.governing_equations,
                     meta.discretization, *meta.assumptions, *meta.limitations)).lower()
    for term in ("magnet", "hall", "suscept", "chi", "χ", "dose", "tissue", "drug"):
        assert term not in blob, f"domain term {term!r} leaked into NodeMeta"
    assert getattr(WaveletEllipticNode, "_stability_level", None) == \
        StabilityLevel.EXPERIMENTAL


def test_boundary_slots_are_explicit():
    """Non-periodic BCs are named, documented slots that raise — not absent."""
    WaveletEllipticNode(dim=1, boundary="periodic")            # ok
    for bc in ("dirichlet", "neumann"):
        with pytest.raises(NotImplementedError, match="slot"):
            WaveletEllipticNode(dim=1, boundary=bc)
    with pytest.raises(ValueError, match="boundary"):
        WaveletEllipticNode(dim=1, boundary="bogus")


# ----------------------------------------------------------------------
# θ→A: coefficient is the differentiable parameter (a = θ)
# ----------------------------------------------------------------------

def test_operator_depends_on_coefficient():
    node = WaveletEllipticNode(dim=2, n_levels=3, n_coarse=2, mass=1.0)
    N = node.N_max
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, bool), "theta": jnp.ones(N)}
    mask = jax.lax.stop_gradient(node.compute_active_set(empty, is_cold_start=True))
    c0 = node.solve_frozen({**empty, "mask": mask}, mask)["c"]
    a = jnp.asarray(1.0 + 50.0 * np.exp(-((np.arange(N) - N / 2) / (N / 8)) ** 2))
    c1 = node.solve_frozen({**empty, "theta": a, "mask": mask}, mask)["c"]
    assert float(jnp.linalg.norm(c1 - c0)) > 1e-8


@pytest.mark.parametrize("dim,nl,nc", [(1, 6, 2), (2, 3, 2)])
def test_djda_grad_vs_fd_and_jit(dim, nl, nc):
    """dJ/da through matrix-free operator assembly: grad matches FD (FD-limited)
    and jit(grad) equals eager grad to machine precision."""
    node = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=1.0)
    N = node.N_max
    a0 = jnp.asarray(1.0 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N))   # a>0
    J = _frozen_J(node, a0)
    g = jax.grad(J)(a0)
    gj = jax.jit(jax.grad(J))(a0)
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    rng = np.random.default_rng(1)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(a0.at[k].add(e)) - J(a0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


@pytest.mark.slow
def test_djda_grad_vs_fd_and_jit_3d():
    node = WaveletEllipticNode(dim=3, n_levels=3, n_coarse=1, mass=1.0)
    N = node.N_max
    a0 = jnp.asarray(1.0 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N))
    J = _frozen_J(node, a0)
    g = jax.grad(J)(a0)
    gj = jax.jit(jax.grad(J))(a0)
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    rng = np.random.default_rng(2)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(a0.at[k].add(e)) - J(a0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


# ----------------------------------------------------------------------
# caller-supplied maps: coeff_fn (θ→a) and source_fn (θ→b)
# ----------------------------------------------------------------------

def test_coeff_fn_maps_theta_to_a():
    """A caller mapping θ→a (a = 1+θ) reproduces the direct a=θ case at the
    corresponding coefficient — the node composes the caller's map, no more."""
    dim, nl, nc = 2, 3, 2
    direct = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=1.0)
    mapped = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=1.0,
                                 coeff_fn=lambda th: 1.0 + th, a_ref=1.0)
    N = direct.N_max
    theta = jnp.asarray(0.2 * np.cos(2 * np.pi * np.arange(N) / N))
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, bool), "theta": jnp.ones(N)}
    mask = jax.lax.stop_gradient(direct.compute_active_set(empty, is_cold_start=True))
    c_direct = direct.solve_frozen({**empty, "theta": 1.0 + theta, "mask": mask}, mask)["c"]
    c_mapped = mapped.solve_frozen({**empty, "theta": theta, "mask": mask}, mask)["c"]
    assert float(jnp.linalg.norm(c_direct - c_mapped)) < 1e-10


def test_source_fn_maps_theta_to_b():
    """A caller mapping θ→f (fixed a): θ drives only the RHS; dJ/dθ flows and is
    jit-stable.  This is the 'source is the design parameter' configuration."""
    dim, nl, nc = 1, 6, 2
    node = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=0.5,
                               a=1.0, source_fn=lambda th: th)
    N = node.N_max
    theta0 = jnp.asarray(np.exp(-((np.arange(N) - N / 2) / (N / 10)) ** 2))
    J = _frozen_J(node, theta0)
    g = jax.grad(J)(theta0)
    gj = jax.jit(jax.grad(J))(theta0)
    assert float(jnp.max(jnp.abs(g - gj)) / (jnp.max(jnp.abs(g)) + 1e-30)) < 1e-9
    rng = np.random.default_rng(3)
    e = 1e-4
    for k in rng.choice(N, 5, replace=False):
        k = int(k)
        fd = float((J(theta0.at[k].add(e)) - J(theta0.at[k].add(-e))) / (2 * e))
        assert abs(float(g[k]) - fd) / (abs(fd) + 1e-30) < 2e-3


# ----------------------------------------------------------------------
# sensors as configuration: gradient-at-points + field functional
# ----------------------------------------------------------------------

def test_gradient_sensor_matches_direct_field():
    """GradientSensor returns ∇u (unsigned) at probes, matching a direct
    central-difference gradient of u = Wn·c."""
    from maddening.nodes.adaptive.wavelets import matrixfree as MF
    node = WaveletEllipticNode(dim=2, n_levels=4, n_coarse=2, mass=0.5)
    side, dim, h = node.side, node.dim, node._h
    gs = SEN.GradientSensor(node._wn_apply, side, dim, h, jnp.asarray([10, 200, 555]))
    c = node.initial_state()["c"]
    B = gs.observe(c)
    assert B.shape == (3, dim)
    u = np.asarray(node._wn_apply(c)).reshape((side,) * dim)
    for j, p in enumerate([10, 200, 555]):
        for d in range(dim):
            gd = (np.roll(u, -1, axis=d) - np.roll(u, 1, axis=d))[
                tuple(np.unravel_index(p, (side,) * dim))] / (2 * h)
            assert abs(float(B[j, d]) - gd) < 1e-10


def _dense_Wn(node):
    """Materialise Wn for the field-functional weight row (test-only, small N)."""
    from maddening.nodes.adaptive.wavelets import transform as T
    W = T.synthesis_matrix(node.n_levels, node.n_coarse, node.order, dim=node.dim)
    return W / node._norms[None, :]


def test_field_functional_objective_differentiates():
    """A field-functional objective (via the M6 sensor protocol) differentiates
    w.r.t. the coefficient — a region-weighted objective as configuration."""
    dim, nl, nc = 1, 6, 2
    base = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=1.0)
    N = base.N_max
    w = jnp.asarray(np.cos(2 * np.pi * np.arange(N) / N))
    node = WaveletEllipticNode(dim=dim, n_levels=nl, n_coarse=nc, mass=1.0,
                               sensor_op=SEN.field_functional(_dense_Wn(base), w))
    a0 = jnp.asarray(1.0 + 0.3 * np.cos(2 * np.pi * np.arange(N) / N))
    J = _frozen_J(node, a0)
    g = jax.grad(J)(a0)
    assert jnp.all(jnp.isfinite(g)) and float(jnp.linalg.norm(g)) > 0
