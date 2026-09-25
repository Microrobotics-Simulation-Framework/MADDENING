"""C1 (v0.4.0 plan): IQN-IMVJ cross-timestep warm start through the IFT
while_loop on a *multi-physics* fixed point — different operators per
sub-domain (a 1D heat rod and a spring-damper), not one operator twice.

The plan's concern was whether the IMVJ column-write convention inside
the while_loop is right for heterogeneous sub-domains: the group must
converge, carry V/W across steps, match the fori reference, and give a
finite gradient that agrees with finite differences.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import extract_last, register_transform
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

N_CELLS = 12


@register_transform("test_heat_to_anchor")
def _heat_to_anchor(flux):
    # A boundary heat flux drives the spring's anchor: an artificial but
    # genuinely two-operator fixed point (FD stencil <-> ODE).
    return 0.02 * flux


def _graph(solver="ift", acceleration="iqn-imvj", jacobian_reuse=3, stiffness=40.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gm = GraphManager()
        gm.add_node(HeatNode("rod", 1e-3, n_cells=N_CELLS, thermal_diffusivity=0.5,
                             initial_temperature=300.0))
        gm.add_node(SpringDamperNode("spring", 1e-3, stiffness=stiffness, damping=3.0,
                                     mass=1.0, rest_length=1.0, initial_position=0.5))
        # spring position sets the rod's right-end temperature offset;
        # the rod's left boundary flux pulls the spring's anchor.
        gm.add_edge("spring", "rod", "position", "right_temperature",
                    transform=lambda x: 300.0 + 20.0 * x)
        gm.add_edge("rod", "spring", "left_heat_flux", "anchor_position",
                    transform="test_heat_to_anchor")
        gm.add_coupling_group(["rod", "spring"], max_iterations=25, tolerance=1e-7,
                              acceleration=acceleration, jacobian_reuse=jacobian_reuse,
                              solver=solver, diagnostics=True)
        gm.compile()
    return gm


def _run(gm, n):
    for _ in range(n):
        gm.step()
    return gm


#: ``{config: (graph, [snapshot after step 1, 2, ...])}``.
_TRAJECTORIES: dict = {}


def _after(n, **config):
    """``(state, diagnostics, meta)`` of ``_graph(**config)`` after ``n`` steps.

    Memoised per configuration: the trajectory from the initial state is
    deterministic, so the state after eight steps is on the way to the
    state after ten, and three tests read the same two configurations
    (each rebuilt and recompiled them).  Each snapshot is a copy, taken
    when the graph passes that step.
    """
    key = tuple(sorted(config.items()))
    if key not in _TRAJECTORIES:
        _TRAJECTORIES[key] = (_graph(**config), [])
    gm, snaps = _TRAJECTORIES[key]
    while len(snaps) < n:
        gm.step()
        snaps.append((
            {name: dict(gm.get_node_state(name)) for name in ("rod", "spring")},
            dict(gm.coupling_diagnostics()["rod+spring"]),
            dict(gm._state["_meta"]),
        ))
    return snaps[n - 1]


def test_converges_and_warm_starts_across_steps():
    _state, d, meta = _after(10)
    assert d["converged"], d
    vw = [k for k in meta if k.startswith("coupling_rod+spring_") and ("_V" in k or "_W" in k)]
    assert vw, sorted(meta)              # V/W carried in _meta between steps
    assert any(float(jnp.max(jnp.abs(meta[k]))) > 0 for k in vw)


def test_matches_fori_reference_and_no_acceleration():
    ref, _, _ = _after(10, solver="fori")
    ift, _, _ = _after(10)
    plain, _, _ = _after(10, acceleration="none", jacobian_reuse=0)
    for n in ("rod", "spring"):
        for f, v in ift[n].items():
            np.testing.assert_allclose(np.asarray(v), np.asarray(ref[n][f]),
                                       rtol=1e-4, atol=1e-4, err_msg=f"{n}.{f} vs fori")
            np.testing.assert_allclose(np.asarray(v), np.asarray(plain[n][f]),
                                       rtol=1e-4, atol=1e-4, err_msg=f"{n}.{f} vs none")


def test_imvj_uses_fewer_iterations_than_none():
    _, ift, _ = _after(8)
    _, plain, _ = _after(8, acceleration="none", jacobian_reuse=0)
    assert ift["iterations"] <= plain["iterations"]


# Slow-marked (still run by slow-tests.yml): a gradient and two
# finite-difference solves through a 20-step scan of the heat-rod/spring
# group, 7-13 s on the CI runner.  The forward is checked on every push above,
# and the gradient against finite differences, over three steps, by
# tests/core/test_imvj_gradient_matches_finite_differences.py.
@pytest.mark.slow
def test_gradient_through_multiphysics_imvj_matches_fd():
    gm = _graph()
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def loss(k):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["spring"]["stiffness"] = k

        def body(s, _):
            s = step(s, ext, p)
            return s, None
        final, _ = jax.lax.scan(body, gm._state, None, length=20)
        return jnp.sum(final["rod"]["temperature"]) + 100.0 * final["spring"]["position"]

    k0 = jnp.asarray(40.0, jnp.float32)
    g = float(jax.grad(loss)(k0))
    assert np.isfinite(g) and g != 0.0
    h = 0.5
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2 * h)
    assert abs(g - fd) <= 5e-2 * abs(fd) + 1e-3, (g, fd)
