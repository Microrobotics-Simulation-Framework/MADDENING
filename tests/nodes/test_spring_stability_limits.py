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
  ``c dt < m`` that is stable only for ``c >= k dt``.  Above ``c dt = m``
  the factor is negative, and it passes -1 at ``c dt + k dt^2 = 2 m``
  (MADD-ANO-098, open: the numerics are unchanged).

The reproducer that found it: rest lengths +1/-1, ``k = 1000``, ``c = 2``,
``m = 0.5``, ``dt = 0.01``.  The single node is far inside its bound
(0.14 against 2), and the pair's centre-of-mass velocity grows by 1.2 per
step.

``compile()`` warns about such a pair (the last part of this file): it
finds the pattern in the graph's edges and judges the spectral radius of
the converged step, ``maddening.nodes.spring._anchored_pair_step_matrix``,
from the live ``gm.params`` values.  The tests here check that matrix
against the node, and the warning against the pattern.
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


def test_the_reproducer_drifts_off_with_finite_values_and_warns_only_at_compile():
    """Rest separation, at rest, one second: the centre of mass, seeded only
    by rounding, leaves.  Every value stays finite.  ``compile()`` warns
    (MADD-ANO-098) and the run itself says nothing."""
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    assert [w.category for w in caught] == [UserWarning]
    assert "MADD-ANO-098" in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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
    assert "c*dt + k*dt**2 = 2*m" in text
    assert "compile() warns" in text
    assert any("stiffness*dt > damping" in hint and "compile() only warns" in hint
               for hint in meta.hazard_hints)


def test_the_heavily_damped_side_flips_sign_and_grows():
    """Above ``c dt = m`` the factor is negative, and past
    ``c dt + k dt^2 = 2 m`` the pair's common velocity flips sign and grows,
    though ``c > k dt``.  At ``k = 6000``, ``c = 150``, ``m = 1`` each node
    is inside its own bound (3.6 against 4) and ``g = -1.25``."""
    k, c, m = 6000.0, 150.0, 1.0
    assert c > k * DT and k * DT ** 2 + 2 * c * DT < 4 * m
    assert c * DT + k * DT ** 2 > 2 * m
    gm = _fresh_pair()
    params = jax.tree.map(lambda x: x, gm.params)
    for name in ("a", "b"):
        params["nodes"][name].update(
            stiffness=jnp.float32(k), damping=jnp.float32(c), mass=jnp.float32(m))
    sums = []
    for n_steps in (N_STEPS - 1, N_STEPS):
        gm.reset_state()
        gm.run_scan(n_steps, params=params)
        sums.append(float(gm.get_node_state("a")["velocity"])
                    + float(gm.get_node_state("b")["velocity"]))
    assert sums[1] / sums[0] == pytest.approx(_g(k, c, m), rel=1e-4)
    assert _g(k, c, m) == pytest.approx(-1.25)


# ---------------------------------------------------------------------------
# The compile-time warning (MADD-ANO-098 stays open: it is not a fix)
# ---------------------------------------------------------------------------

from maddening.core.transforms import identity, negate, scale  # noqa: E402
from maddening.nodes.spring import (  # noqa: E402
    _ANCHORED_PAIR_GROWTH_TOL,
    _anchored_pair_step_matrix,
)


def _fresh_pair():
    """The module's pair, built again: defaults, compiled, nothing stepped."""
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


def _springs(gm, dt=DT, names=("a", "b"), **kwargs):
    for i, name in enumerate(names):
        gm.add_node(SpringDamperNode(name, dt, rest_length=1.0 if i % 2 == 0 else -1.0,
                                     initial_position=float(i % 2 == 0), **kwargs))
    return gm


def _anchor_each_other(gm, a="a", b="b", **edge_kwargs):
    gm.add_edge(a, b, "position", "anchor_position", **edge_kwargs)
    gm.add_edge(b, a, "position", "anchor_position", **edge_kwargs)
    return gm


def _group(gm, names=("a", "b"), **kwargs):
    kwargs = {"max_iterations": 50, "tolerance": 1e-8, **kwargs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(list(names), **kwargs)
    return gm


def _reproducer(**group_kwargs):
    gm = _springs(GraphManager(), stiffness=1000.0, damping=2.0, mass=0.5)
    return _group(_anchor_each_other(gm), **group_kwargs)


def _anomaly_issues(gm):
    return [i for i in gm.validate() if "MADD-ANO-098" in i]


def test_compile_warns_about_the_reproducer():
    gm = _reproducer()
    with pytest.warns(UserWarning, match="MADD-ANO-098") as record:
        gm.compile()
    messages = [str(w.message) for w in record if "MADD-ANO-098" in str(w.message)]
    assert len(messages) == 1
    message = messages[0]
    assert "'a' and 'b'" in message
    assert "k = 1000, c = 2 and m = 0.5" in message and "dt = 0.01" in message
    assert "g = (m - c*dt)/(m - k*dt**2) = 1.2 " in message
    assert "c < k*dt (2 < 10)" in message
    # the remedy
    assert "smaller timestep" in message and "c >= k*dt" in message
    assert "10 <= c <= 90" in message


def test_the_warning_never_refuses_the_graph():
    """A warning, not an error: ``validate()`` reports it as ``WARNING:``,
    ``compile()`` finishes, and the graph steps."""
    gm = _reproducer()
    issues = _anomaly_issues(gm)
    assert len(issues) == 1 and issues[0].startswith("WARNING: ")
    assert not [i for i in gm.validate() if i.startswith("ERROR")]
    with pytest.warns(UserWarning, match="MADD-ANO-098"):
        gm.compile()
    gm.run_scan(5)
    assert np.isfinite(float(gm.get_node_state("a")["position"]))


def test_no_warning_on_the_defaults_which_sit_on_the_limit():
    """``k = 100``, ``c = 1``, ``dt = 0.01``: ``c = k dt`` exactly, so
    ``g = 1``.  Neutral, not growing: the defaults compile quietly."""
    gm = _group(_anchor_each_other(_springs(GraphManager())))
    assert _anomaly_issues(gm) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()


def test_no_warning_exactly_at_c_equal_k_dt_and_a_warning_just_below():
    """Constants exact in binary (``dt = 1/128``, ``k = 128``): ``k dt`` is
    1 to the bit, so the boundary is the boundary.  A thousandth less
    damping is past it; "strictly" means strictly."""
    dt = 1.0 / 128.0
    on = _group(_anchor_each_other(_springs(GraphManager(), dt=dt, stiffness=128.0,
                                            damping=1.0)))
    assert _anomaly_issues(on) == []
    below = _group(_anchor_each_other(_springs(GraphManager(), dt=dt, stiffness=128.0,
                                               damping=0.999)))
    issues = _anomaly_issues(below)
    assert len(issues) == 1 and "c < k*dt (0.999 < 1)" in issues[0]


def test_no_warning_for_a_lagged_exchange():
    """Without a coupling group the positions lag a step: a different scheme
    with its own limit, which the warning does not describe."""
    gm = _anchor_each_other(_springs(GraphManager(), stiffness=1000.0, damping=2.0,
                                     mass=0.5))
    assert _anomaly_issues(gm) == []


def test_no_warning_for_a_single_staggered_pass():
    assert _anomaly_issues(_reproducer(max_iterations=1)) == []


def test_no_warning_for_a_one_way_anchor():
    """``b`` follows ``a``, and ``a``'s anchor is the origin: a leader and a
    follower, each a single node against a prescribed anchor."""
    gm = _springs(GraphManager(), stiffness=1000.0, damping=2.0, mass=0.5)
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "velocity", "anchor_position")
    assert _anomaly_issues(_group(gm)) == []


@pytest.mark.parametrize("transform", [negate, scale(0.5), lambda x: x],
                         ids=["negate", "scale", "lambda-identity"])
def test_a_transform_other_than_identity_is_not_judged(transform):
    """Documented: only a bare edge or the built-in ``identity`` passes the
    partner's position through unchanged as far as the warning knows, even
    for a lambda that does the same."""
    gm = _springs(GraphManager(), stiffness=1000.0, damping=2.0, mass=0.5)
    gm.add_edge("a", "b", "position", "anchor_position", transform=transform)
    gm.add_edge("b", "a", "position", "anchor_position")
    assert _anomaly_issues(_group(gm)) == []


def test_the_built_in_identity_is_the_pattern():
    gm = _springs(GraphManager(), stiffness=1000.0, damping=2.0, mass=0.5)
    gm.add_edge("a", "b", "position", "anchor_position", transform=identity)
    gm.add_edge("b", "a", "position", "anchor_position", transform="identity")
    assert len(_anomaly_issues(_group(gm))) == 1


def test_no_warning_for_an_additive_anchor():
    gm = _springs(GraphManager(), stiffness=1000.0, damping=2.0, mass=0.5)
    gm.add_edge("a", "b", "position", "anchor_position", additive=True)
    gm.add_edge("b", "a", "position", "anchor_position")
    assert _anomaly_issues(_group(gm)) == []


def test_no_warning_when_something_else_also_writes_an_anchor():
    """The second writer is an external input on the wired field: a graph
    ``compile()`` refuses (MADD-ANO-265).  The advisory has nothing to say
    of it, and ``validate()`` lists the refusal."""
    gm = _anchor_each_other(_springs(GraphManager(), stiffness=1000.0, damping=2.0,
                                     mass=0.5))
    gm.add_external_input("a", "anchor_position")
    assert _anomaly_issues(_group(gm)) == []
    assert any("an edge and a declared external input target the same field" in i
               for i in gm.validate())


def test_no_warning_when_only_one_of_the_pair_is_in_the_group():
    """``a`` and ``c`` are grouped, ``b`` is not: the ``a``-``b`` exchange
    lags across the group boundary."""
    gm = _springs(GraphManager(), names=("a", "b", "c"), stiffness=1000.0,
                  damping=2.0, mass=0.5)
    _anchor_each_other(gm)
    gm.add_edge("a", "c", "position", "anchor_position")
    assert _anomaly_issues(_group(gm, names=("a", "c"))) == []


def test_springs_of_different_timesteps_in_a_subcycled_group_are_not_judged():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", DT, stiffness=1000.0, damping=2.0, mass=0.5))
    gm.add_node(SpringDamperNode("b", DT / 2, stiffness=1000.0, damping=2.0, mass=0.5,
                                 rest_length=-1.0))
    _anchor_each_other(gm)
    assert _anomaly_issues(_group(gm, subcycling=True)) == []


def test_a_subclass_that_overrides_update_is_not_judged():
    class Stiffer(SpringDamperNode):
        def update(self, state, boundary_inputs, dt, *, params=None):
            return super().update(state, boundary_inputs, dt, params=params)

    gm = GraphManager()
    gm.add_node(Stiffer("a", DT, stiffness=1000.0, damping=2.0, mass=0.5))
    gm.add_node(SpringDamperNode("b", DT, stiffness=1000.0, damping=2.0, mass=0.5,
                                 rest_length=-1.0))
    assert _anomaly_issues(_group(_anchor_each_other(gm))) == []


def test_a_pair_whose_coupling_iteration_cannot_converge_is_not_judged():
    """``k dt^2 >= m``: Gauss-Seidel contracts by ``(k dt^2/m)^2`` per pass,
    so the converged step is not the one that runs (documented)."""
    gm = _springs(GraphManager(), stiffness=15000.0, damping=0.0, mass=1.0)
    assert _anomaly_issues(_group(_anchor_each_other(gm))) == []


def test_two_pairs_in_one_group_are_judged_each():
    gm = _springs(GraphManager(), names=("a", "b", "c", "d"), stiffness=1000.0,
                  damping=2.0, mass=0.5)
    _anchor_each_other(gm, "a", "b")
    _anchor_each_other(gm, "c", "d")
    issues = _anomaly_issues(_group(gm, names=("a", "b", "c", "d")))
    assert len(issues) == 2
    assert any("'a' and 'b'" in i for i in issues)
    assert any("'c' and 'd'" in i for i in issues)


def test_the_rest_lengths_do_not_matter():
    """Rest lengths both +1: a constant push on top, and the same growth."""
    gm = GraphManager()
    for name in ("a", "b"):
        gm.add_node(SpringDamperNode(name, DT, stiffness=1000.0, damping=2.0, mass=0.5,
                                     rest_length=1.0))
    assert len(_anomaly_issues(_group(_anchor_each_other(gm)))) == 1


def test_compile_warns_on_the_heavily_damped_side():
    """``c > k dt`` and still growing (the test above measures it): the
    remedy points below the upper figure, not only above ``k dt``."""
    gm = _springs(GraphManager(), stiffness=6000.0, damping=150.0, mass=1.0)
    issues = _anomaly_issues(_group(_anchor_each_other(gm)))
    assert len(issues) == 1
    assert "= -1.25 every step" in issues[0]
    assert "c*dt + k*dt**2 > 2*m (2.1 > 2)" in issues[0]
    assert "60 <= c <= 140" in issues[0]


def test_every_pair_past_the_single_node_limit_is_warned_about():
    """``k dt^2 + 2 c dt >= 4 m`` with the iteration converging: each node
    is unstable alone, nothing else says so, and the pair is past its upper
    figure too."""
    k, c, m = 3000.0, 190.0, 1.0
    assert k * DT ** 2 + 2 * c * DT >= 4 * m and k * DT ** 2 < m
    gm = _springs(GraphManager(), stiffness=k, damping=c, mass=m)
    assert len(_anomaly_issues(_group(_anchor_each_other(gm)))) == 1


# -- live parameters: the values a compile runs ------------------------------


def test_a_damping_written_into_the_graph_params_is_the_one_judged():
    """The defaults compile quietly; a calibration that halves both
    dampings puts the pair past ``c = k dt``, and the next compile says so.
    Raising them back clears it."""
    gm = _fresh_pair()
    assert _anomaly_issues(gm) == []
    for name in ("a", "b"):
        gm.params["nodes"][name]["damping"] = jnp.float32(0.5)
    issues = _anomaly_issues(gm)
    assert len(issues) == 1 and "c = 0.5" in issues[0]
    with pytest.warns(UserWarning, match="MADD-ANO-098"):
        gm.compile()
    for name in ("a", "b"):
        gm.params["nodes"][name]["damping"] = jnp.float32(1.0)
    assert _anomaly_issues(gm) == []


def test_a_stiffness_written_into_one_node_makes_an_unequal_pair_that_is_judged():
    """A calibration of one node leaves the pair unequal, and the warning
    judges the converged step's spectral radius instead of the closed
    form.  The reproducer with its damping raised to ``k dt`` is quiet;
    stiffening ``a`` by 10% puts it past the limit."""
    gm = _springs(GraphManager(), stiffness=1000.0, damping=10.0, mass=0.5)
    gm = _group(_anchor_each_other(gm))
    gm.compile()
    assert _anomaly_issues(gm) == []
    gm.params["nodes"]["a"]["stiffness"] = jnp.float32(1100.0)
    issues = _anomaly_issues(gm)
    assert len(issues) == 1
    assert "'a' has k = 1100" in issues[0] and "'b' has k = 1000" in issues[0]
    assert "spectral radius" in issues[0]


def test_a_traced_or_non_finite_value_is_not_judged():
    gm = _reproducer()
    gm.params["nodes"]["a"] = {"damping": jnp.float32(np.nan)}
    gm.params["nodes"]["b"] = {"damping": jnp.float32(np.nan)}
    assert _anomaly_issues(gm) == []


# -- the matrix the warning judges, against the node -------------------------


@pytest.mark.parametrize("a, b", [
    ((1000.0, 2.0, 0.5), (1000.0, 2.0, 0.5)),       # the reproducer
    ((1000.0, 2.0, 0.5), (1400.0, 6.0, 0.8)),       # unequal, growing
    ((300.0, 9.0, 1.0), (500.0, 4.0, 0.7)),         # unequal, decaying
    ((6000.0, 150.0, 1.0), (5000.0, 140.0, 1.1)),   # heavily damped, unequal
], ids=["reproducer", "unequal-growing", "unequal-decaying", "heavily-damped"])
def test_the_step_matrix_is_the_nodes_converged_step(pair, a, b):
    """Two runs from different states: their difference evolves by the
    linear part of the step alone (the rest lengths drop out), and after 40
    steps it is the matrix's 40th power applied to the first difference, in
    the reduced coordinates ``(x_a - x_b, v_a, v_b)``."""
    params = jax.tree.map(lambda x: x, pair.params)
    for name, (k, c, m) in zip(("a", "b"), (a, b)):
        params["nodes"][name].update(
            stiffness=jnp.float32(k), damping=jnp.float32(c), mass=jnp.float32(m))

    def run(xa, va, xb, vb):
        pair.reset_state()
        pair.set_node_state("a", {"position": jnp.float32(xa), "velocity": jnp.float32(va)})
        pair.set_node_state("b", {"position": jnp.float32(xb), "velocity": jnp.float32(vb)})
        pair.run_scan(N_STEPS, params=params)
        sa, sb = pair.get_node_state("a"), pair.get_node_state("b")
        return np.array([float(sa["position"]) - float(sb["position"]),
                         float(sa["velocity"]), float(sb["velocity"])])

    base, moved = run(1.0, 0.0, 0.0, 0.0), run(1.3, 0.7, -0.2, 1.1)
    start = np.array([1.5 - 1.0, 0.7, 1.1])    # the starts' difference, reduced
    step = _anchored_pair_step_matrix(a, b, DT)
    predicted = np.linalg.matrix_power(step, N_STEPS) @ start
    np.testing.assert_allclose(moved - base, predicted,
                               rtol=1e-4, atol=1e-4 * np.max(np.abs(predicted)))


def test_the_spectral_radius_of_an_equal_pair_is_the_closed_form():
    for k, c, m in [(1000.0, 2.0, 0.5), (100.0, 0.5, 1.0), (6000.0, 150.0, 1.0)]:
        rho = np.max(np.abs(np.linalg.eigvals(
            _anchored_pair_step_matrix((k, c, m), (k, c, m), DT))))
        assert rho == pytest.approx(abs(_g(k, c, m)), rel=1e-12)
    # Inside the limit both the centre of mass and the relative motion
    # decay (the common translation, which is neutral, is left out).
    rho = np.max(np.abs(np.linalg.eigvals(
        _anchored_pair_step_matrix((1000.0, 11.0, 0.5), (1000.0, 11.0, 0.5), DT))))
    assert rho < 1.0 - _ANCHORED_PAIR_GROWTH_TOL


def test_the_step_matrix_refuses_a_pair_whose_iteration_cannot_converge():
    assert _anchored_pair_step_matrix((10000.0, 0.0, 1.0), (10000.0, 0.0, 1.0), DT) is None
    assert _anchored_pair_step_matrix((9000.0, 0.0, 1.0), (9000.0, 0.0, 1.0), DT) is not None
