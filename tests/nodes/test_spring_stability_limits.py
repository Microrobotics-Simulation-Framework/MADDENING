"""The two stability bounds ``SpringDamperNode`` states, checked against the node.

Semi-implicit Euler is conditionally stable, and nothing checks it.  The
node's ``NodeMeta`` and algorithm guide now state both bounds:

* **one node, fixed anchor**: the step maps ``(x, v)`` by a matrix with
  trace ``2 - (k dt^2 + c dt)/m`` and determinant ``1 - c dt/m``, stable
  exactly for ``k dt^2 + 2 c dt < 4 m``;
* **two nodes anchored on each other in a converged coupling group**: each
  is explicit in its own position and implicit in the partner's, so the
  pair's momentum is not conserved and the centre-of-mass velocity of an
  equal pair is multiplied by ``g = (m - c dt)/(m - k dt^2)`` per step.
  With ``k dt^2 < m`` (where the coupling iteration converges) and
  ``c dt < m`` that is stable only for ``c >= k dt`` (MADD-ANO-098, open:
  the numerics are unchanged and nothing warns).

The reproducer that found it: rest lengths +1/-1, ``k = 1000``, ``c = 2``,
``m = 0.5``, ``dt = 0.01``.  The single node is far inside its bound
(0.14 against 2), and the pair's centre-of-mass velocity grows by 1.2 per
step.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

DT = 0.01
N_STEPS = 40


def _single_step_matrix(k, c, m, dt):
    """One ``update`` against a fixed anchor as a matrix on ``(x, v)``, from
    the node's own Jacobian (the map is affine, so this is exact)."""
    node = SpringDamperNode("s", dt, stiffness=k, damping=c, mass=m, rest_length=0.0)

    def step(xv):
        out = node.update({"position": xv[0], "velocity": xv[1]},
                          {"anchor_position": jnp.float32(0.0)}, dt)
        return jnp.stack([out["position"], out["velocity"]])

    return np.asarray(jax.jacfwd(step)(jnp.zeros(2, jnp.float32)), dtype=np.float64)


def _single_step_radius(k, c, m, dt):
    return float(np.max(np.abs(np.linalg.eigvals(_single_step_matrix(k, c, m, dt)))))


@pytest.mark.parametrize("k, c, m", [
    (1000.0, 2.0, 0.5),       # the reproducer's node
    (40000.0, 0.0, 1.0),      # undamped: the bound is dt < 2 sqrt(m/k)
    (30000.0, 50.0, 1.0),     # damping tightens it
    (100.0, 30.0, 0.1),
])
def test_the_single_node_bound_is_exact(k, c, m):
    """On ``k dt^2 + 2 c dt = 4 m`` the step matrix has the eigenvalue -1
    (its characteristic polynomial ``1 + trace + det`` vanishes at -1); its
    spectral radius is below 1 just inside and above 1 just outside.  The
    stated bound is the true edge, not a rule of thumb.  (The radius itself
    is not compared with 1 on the edge: undamped, -1 is a double eigenvalue
    there, and float32 rounding moves a double root by its square root.)"""
    dt_edge = (-2 * c + np.sqrt(4 * c * c + 16 * k * m)) / (2 * k)   # the bound's root
    jac = _single_step_matrix(k, c, m, dt_edge)
    assert 1.0 + np.trace(jac) + np.linalg.det(jac) == pytest.approx(0.0, abs=1e-5)
    inside = _single_step_radius(k, c, m, 0.98 * dt_edge)
    # undamped, the scheme is symplectic and the eigenvalues sit on the unit
    # circle inside the bound: bounded, not decaying
    assert inside < 1.0 if c > 0 else inside == pytest.approx(1.0, abs=1e-6)
    assert _single_step_radius(k, c, m, 1.02 * dt_edge) > 1.0 + 1e-3


def test_the_reproducers_single_node_is_well_inside_its_bound():
    assert 1000.0 * DT ** 2 + 2 * 2.0 * DT < 4 * 0.5
    assert _single_step_radius(1000.0, 2.0, 0.5, DT) < 1.0


@pytest.fixture(scope="module")
def pair():
    """Two springs anchored on each other in a converged coupling group,
    started at their rest separation with equal velocities: pure
    centre-of-mass motion.  ``k``, ``c`` and ``m`` go through the params
    pytree, so every case runs on one compiled step."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", DT, rest_length=1.0, initial_position=1.0,
                                 initial_velocity=1.0))
    gm.add_node(SpringDamperNode("b", DT, rest_length=-1.0, initial_position=0.0,
                                 initial_velocity=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-9)
    gm.compile()
    return gm


def _com_growth_per_step(gm, k, c, m):
    params = jax.tree.map(lambda x: x, gm.params)
    for name in ("a", "b"):
        params["nodes"][name].update(
            stiffness=jnp.float32(k), damping=jnp.float32(c), mass=jnp.float32(m))
    gm.reset_state()
    gm.run_scan(N_STEPS, params=params)
    s = float(gm.get_node_state("a")["velocity"]) + float(gm.get_node_state("b")["velocity"])
    return (s / 2.0) ** (1.0 / N_STEPS)


def _g(k, c, m, dt=DT):
    return (m - c * dt) / (m - k * dt * dt)


@pytest.mark.parametrize("k, c, m", [
    (1000.0, 2.0, 0.5),       # the reproducer: g = 1.2
    (1000.0, 9.0, 0.5),       # just under c = k dt = 10: grows
    (1000.0, 10.0, 0.5),      # on it: neither grows nor decays
    (1000.0, 11.0, 0.5),      # just over: decays
    (100.0, 0.5, 1.0),        # the default k, damping halved
    (100.0, 1.0, 1.0),        # the defaults: exactly on the limit
    (30.0, 2.0, 1.0),         # well inside
])
def test_the_coupled_pairs_centre_of_mass_grows_by_the_stated_factor(pair, k, c, m):
    measured = _com_growth_per_step(pair, k, c, m)
    assert measured == pytest.approx(_g(k, c, m), rel=1e-5)
    # ... and the stated condition is the sign of the growth.
    if c < k * DT:
        assert measured > 1.0
    elif c > k * DT:
        assert measured < 1.0


def test_the_reproducer_drifts_off_with_finite_values_and_no_warning():
    """Rest separation, at rest, one second: the centre of mass, seeded only
    by rounding, leaves.  Every value stays finite and nothing warns."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", DT, stiffness=1000.0, damping=2.0, mass=0.5,
                                 rest_length=1.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", DT, stiffness=1000.0, damping=2.0, mass=0.5,
                                 rest_length=-1.0, initial_position=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["a", "b"], max_iterations=50, tolerance=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
        gm.run_scan(100)
    a = float(gm.get_node_state("a")["position"])
    b = float(gm.get_node_state("b")["position"])
    assert np.isfinite(a) and np.isfinite(b)
    assert a - b == pytest.approx(1.0, abs=1e-2)    # the spring itself is fine
    assert abs(a) > 1.0                              # the pair has left


def test_the_relative_motion_is_stable_where_the_pair_is_not(pair):
    """Started stretched, with no centre-of-mass motion: the separation
    settles at 1 at the reproducer's constants.  The instability is the
    drift, not the spring."""
    params = jax.tree.map(lambda x: x, pair.params)
    for name in ("a", "b"):
        params["nodes"][name].update(stiffness=jnp.float32(1000.0),
                                     damping=jnp.float32(2.0), mass=jnp.float32(0.5))
    pair.reset_state()
    pair.set_node_state("a", {"position": jnp.float32(1.5), "velocity": jnp.float32(0.0)})
    pair.set_node_state("b", {"position": jnp.float32(-0.5), "velocity": jnp.float32(0.0)})
    pair.run_scan(N_STEPS, params=params)
    a = float(pair.get_node_state("a")["position"])
    b = float(pair.get_node_state("b")["position"])
    # the stretch of 1 has decayed by |lambda|**40 = (0.96/1.2)**20 ~ 0.012
    assert abs((a - b) - 1.0) < 0.05


def test_the_node_metadata_states_both_bounds():
    meta = SpringDamperNode.meta
    text = " ".join(meta.limitations)
    assert "k*dt**2 + 2*c*dt < 4*m" in text
    assert "(m - c*dt)/(m - k*dt**2)" in text and "c >= k*dt" in text
    assert any("stiffness*dt > damping" in hint for hint in meta.hazard_hints)
