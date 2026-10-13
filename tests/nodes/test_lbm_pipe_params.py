"""LBMPipeNode's parameters mean the same thing injected, constructed and reloaded.

Two invariants, over every trainable leaf of both modes:

* **A calibrated graph is its reloaded graph.**  ``GraphManager.to_dict``
  writes the live ``params`` over the constructor arguments, and
  ``from_dict`` rebuilds the node from them.  When the multiphase
  ``rho_liquid`` / ``rho_gas`` were both trainable step constants *and*
  the recipe of the initial density (``initial_state`` runs at compile
  time, from the constructor's values), a graph calibrated through
  ``gm.params`` and the graph its config rebuilt started from different
  initial densities: 0.30 apart in density after three steps, no warning.
  The initial condition now reads its own non-trainable
  ``initial_rho_liquid`` / ``initial_rho_gas`` leaves.
* **An injected value is the constructed value.**  ``update(...,
  params={k: v})`` on a node built with ``k = v0`` must equal ``update``
  on a node built with ``k = v``.  A step that read ``self.params["G"]``
  instead of the injected ``G`` passed every existing test (the audit's
  mutant M12); this is the test that kills it, and the same shape of
  mistake for every other leaf.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes.lbm_pipe import LBMPipeNode  # noqa: E402

GRID = dict(nx=6, ny=8, nz=8, propeller_x=2, tau=0.8, propeller_strength=1e-3)
MODES = {
    # fill_fraction < 1 gives the tracer (single phase) and the density
    # (multiphase) an interface, so every constant has something to act on.
    "single": dict(GRID, gravity=-1e-4, fill_fraction=0.6, tau_tracer=0.7),
    "multi": dict(GRID, gravity=-1e-4, fill_fraction=0.6, G=-4.5,
                  rho_liquid=2.0, rho_gas=0.3, rho_0=1.0, rho_wall=1.8),
}


def _trainable(mode):
    node = LBMPipeNode("p", 1.0, **MODES[mode])
    specs = node.param_specs()
    return sorted(k for k in node.params_pytree() if specs.get(k) is None or specs[k].trainable)


def _cases():
    return [(mode, leaf) for mode in MODES for leaf in _trainable(mode)]


def _moved(value):
    """A different, still-physical value: 10% further from zero."""
    return float(np.float32(value * 1.1 if value != 0 else 1e-4))


def _graph(**kw):
    gm = GraphManager()
    gm.add_node(LBMPipeNode("p", 1.0, **kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def test_every_trainable_leaf_of_both_modes_is_covered():
    """The parametrisation below is derived from ``param_specs``; pin what
    it derives so a leaf cannot drop out of both tests unnoticed."""
    assert _trainable("single") == ["gravity", "propeller_strength", "tau", "tau_tracer"]
    assert _trainable("multi") == [
        "G", "gravity", "propeller_strength", "rho_0", "rho_gas", "rho_liquid",
        "rho_wall", "tau",
    ]


def test_the_initial_densities_are_non_trainable_initial_conditions():
    node = LBMPipeNode("p", 1.0, **MODES["multi"])
    specs = node.param_specs()
    for key in ("initial_rho_liquid", "initial_rho_gas"):
        assert key in node.params_pytree()
        assert specs[key].trainable is False
    # Defaults resolve to the phase densities, once, at construction.
    assert node.params["initial_rho_liquid"] == 2.0
    assert node.params["initial_rho_gas"] == 0.3


def test_the_initial_state_reads_the_initial_densities_and_not_the_phase_references():
    """Same ``initial_*``, different ``rho_liquid`` / ``rho_gas``: the same
    initial state, bit for bit.  Different ``initial_*``: a different one."""
    base = dict(MODES["multi"], initial_rho_liquid=2.0, initial_rho_gas=0.3)
    a = LBMPipeNode("p", 1.0, **base).initial_state()
    b = LBMPipeNode("p", 1.0, **dict(base, rho_liquid=2.4, rho_gas=0.36)).initial_state()
    for field in a:
        np.testing.assert_array_equal(np.asarray(a[field]), np.asarray(b[field]), err_msg=field)
    c = LBMPipeNode("p", 1.0, **dict(base, initial_rho_liquid=2.4)).initial_state()
    assert not np.array_equal(np.asarray(a["density"]), np.asarray(c["density"]))


def test_the_initial_densities_are_validated_like_the_phase_densities():
    with pytest.raises(ValueError, match="initial_rho_liquid must be > initial_rho_gas"):
        LBMPipeNode("p", 1.0, **dict(MODES["multi"], initial_rho_liquid=0.2))
    with pytest.raises(ValueError, match="initial_rho_gas must be > 0"):
        LBMPipeNode("p", 1.0, **dict(MODES["multi"], initial_rho_gas=-0.1))


@pytest.mark.parametrize("mode,leaf", _cases())
def test_a_calibrated_graph_and_its_reloaded_config_are_the_same_graph(mode, leaf):
    """Move one trainable leaf through ``gm.params``, run, save, rebuild,
    run: the initial states and the three-step results are identical.  A
    fresh graph per run, since ``run_scan`` advances the graph's state."""
    kw = MODES[mode]
    gm = _graph(**kw)
    initial = {k: np.asarray(v) for k, v in gm.get_node_state("p").items()}
    value = _moved(float(gm.params["nodes"]["p"][leaf]))
    gm.params["nodes"]["p"][leaf] = jnp.asarray(value, jnp.float32)
    calibrated = gm.run_scan(3)["p"]
    cfg = gm.to_dict()
    saved = next(n for n in cfg["nodes"] if n["name"] == "p")["params"]
    assert saved[leaf] == pytest.approx(value, rel=1e-7)

    reloaded_gm = GraphManager.from_dict(cfg, {"LBMPipeNode": LBMPipeNode})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded_gm.compile()
    reloaded_initial = reloaded_gm.get_node_state("p")
    for field, arr in initial.items():
        np.testing.assert_array_equal(np.asarray(reloaded_initial[field]), arr,
                                      err_msg=f"initial {field}")
    reloaded = reloaded_gm.run_scan(3)["p"]
    for field in calibrated:
        np.testing.assert_array_equal(np.asarray(reloaded[field]), np.asarray(calibrated[field]),
                                      err_msg=f"after 3 steps: {field}")


@pytest.mark.parametrize("mode,leaf", _cases())
def test_an_injected_leaf_is_the_constructed_leaf(mode, leaf):
    """``update(params={leaf: v})`` on a node built with the old value equals
    ``update`` on a node built with ``v``, on a state with an interface and
    a flow; and the leaf moves the result, so the equality is not vacuous.
    Derived defaults (``rho_wall``, ``initial_*``) are pinned explicitly so
    the two constructions differ in ``leaf`` alone."""
    kw = dict(MODES[mode])
    if mode == "multi":
        kw.update(initial_rho_liquid=kw["rho_liquid"], initial_rho_gas=kw["rho_gas"])
    built = LBMPipeNode("p", 1.0, **kw)
    state = built.initial_state()
    for _ in range(2):                    # develop a velocity field first
        state = built.update(state, {}, 1.0)
    value = _moved(float(built.params_pytree()[leaf]))

    injected = built.update(state, {}, 1.0, params={leaf: jnp.asarray(value, jnp.float32)})
    rebuilt = LBMPipeNode("p", 1.0, **dict(kw, **{leaf: value})).update(state, {}, 1.0)
    baseline = built.update(state, {}, 1.0)

    moved = False
    for field in injected:
        np.testing.assert_allclose(np.asarray(injected[field]), np.asarray(rebuilt[field]),
                                   rtol=1e-6, atol=1e-7, err_msg=f"{leaf}: {field}")
        moved |= not np.allclose(np.asarray(injected[field]), np.asarray(baseline[field]),
                                 rtol=1e-6, atol=1e-7)
    assert moved, f"{leaf} did not change any output field: the fixture cannot see it"
