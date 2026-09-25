"""A multi-rate coupling group's ``_meta`` comes only from solves the step keeps.

On a multi-rate graph a coupling group with a rate divider above one
solves on every base step -- the step function has one static structure
-- but its node states are kept only on the steps it fires on
(``step_count % divider == 0``).  Its ``_meta`` slots used to be merged
from *every* solve, so between firings they described solves the step
had thrown away: ``coupling_diagnostics()``, the profiler and the sysid
mask reported those, ``strict_convergence`` raised about them, and the
linear predictor and the IQN-IMVJ warm start learned from them.  The
slots now follow the node states' rule.

The graph makes the difference loud.  A pair ``a <-> b`` at 0.01 s is fed
the phase ``0..9`` of a clock at 0.001 s; the pair's gain grows with the
phase, so the solve the step applies (phase 0) contracts and the ones it
discards at phases 4-9 do not.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import profile_graph
from maddening.sysid import observations_from_history, windowed_loss

KEY = "a+b"
DIVIDER = 10


class _Phase(SimulationNode):
    """``phase <- (phase + 1) mod 10``, starting so that step 0 reads 0."""

    def __init__(self, name, timestep):
        super().__init__(name=name, timestep=timestep)

    def initial_state(self):
        return {"phase": jnp.float32(-1.0)}

    def update(self, state, boundary_inputs, dt):
        return {"phase": jnp.mod(state["phase"] + 1.0, 10.0)}


class _Gained(SimulationNode):
    """``x <- (0.5 + 0.5 * phase) * u + bias``."""

    def __init__(self, name, timestep, bias, phased):
        super().__init__(name=name, timestep=timestep, bias=bias)
        self._phased = phased

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}
        if self._phased:
            spec["phase"] = BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                              default=jnp.float32(0.0))
        return spec

    def update(self, state, boundary_inputs, dt):
        # ``.get``: the profiler times each node with no inputs at all.
        gain = 0.5 + 0.5 * boundary_inputs.get("phase", jnp.float32(0.0))
        return {"x": gain * boundary_inputs.get("u", jnp.float32(0.0))
                + self.params["bias"]}


def _graph(**group_kw):
    gm = GraphManager()
    gm.add_node(_Phase("clock", 0.001))
    gm.add_node(_Gained("a", 0.01, bias=1.0, phased=True))
    gm.add_node(_Gained("b", 0.01, bias=0.0, phased=False))
    gm.add_edge("clock", "a", "phase", "phase")
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-5,
                          **group_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate INFO notice
        gm.compile()
    assert gm.is_multirate and gm.rate_dividers[KEY[0]] == DIVIDER
    return gm


def _slots(gm):
    prefix = f"coupling_{KEY}_"
    return {k[len(prefix):]: np.asarray(v)
            for k, v in gm._state["_meta"].items() if k.startswith(prefix)}


def test_the_report_between_firings_is_the_last_applied_solve():
    """Every base step after a firing reports that firing's solve, exactly."""
    gm = _graph(diagnostics=True)
    applied = None
    for step in range(2 * DIVIDER + 2):
        gm.step()
        report = dict(gm.coupling_diagnostics()[KEY])
        slots = _slots(gm)
        if step % DIVIDER == 0:
            applied = (report, slots)
            assert report["converged"], (step, report)
            continue
        assert report == applied[0], (step, report, applied[0])
        for name, value in slots.items():
            assert value.tobytes() == applied[1][name].tobytes(), (step, name)


def test_strict_convergence_does_not_raise_about_a_discarded_solve():
    """Every applied solve converges, so nothing may raise."""
    gm = _graph(strict_convergence=True)
    for _ in range(DIVIDER + 2):
        gm.step()
    assert np.isfinite(float(gm.get_node_state("a")["x"]))


def test_strict_convergence_still_raises_about_an_applied_solve():
    """The gate is on the discarded solves only: a firing step is checked.

    The same graph with a cap too small for the applied solve (phase 0,
    Gauss-Seidel rate 0.25) to meet the tolerance raises on step 0.
    """
    gm = GraphManager()
    gm.add_node(_Phase("clock", 0.001))
    gm.add_node(_Gained("a", 0.01, bias=1.0, phased=True))
    gm.add_node(_Gained("b", 0.01, bias=0.0, phased=False))
    gm.add_edge("clock", "a", "phase", "phase")
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=2, tolerance=1e-7,
                          strict_convergence=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    with pytest.raises(Exception, match="without converging"):
        gm.step()


def test_the_linear_predictor_learns_only_from_applied_solves():
    """Two applied solves in, the extrapolated guess is still near the answer.

    Fed the discarded solves the history diverged, and the forward state
    with it: 8.7e12 at the second firing and 2.1e27 at the third.
    """
    fixed_point = 4.0 / 3.0         # a = 0.5 * (0.5 * a) + 1 at phase 0
    gm = _graph(diagnostics=True, predictor="linear")
    for step in range(2 * DIVIDER + 1):
        gm.step()
        if step % DIVIDER == 0:
            report = gm.coupling_diagnostics()[KEY]
            a = float(gm.get_node_state("a")["x"])
            assert report["converged"], (step, report)
            assert a == pytest.approx(fixed_point, rel=1e-4), (step, a)
    assert int(_slots(gm)["pred_count"]) == 2


def test_the_imvj_warm_start_is_carried_only_from_applied_solves():
    gm = _graph(diagnostics=True, acceleration="iqn-imvj", jacobian_reuse=3)
    gm.step()
    fired = _slots(gm)
    for _ in range(DIVIDER - 1):
        gm.step()
    held = _slots(gm)
    for name in ("V", "W", "iterations", "residual", "amplification"):
        assert held[name].tobytes() == fired[name].tobytes(), name
    assert np.any(held["V"] != 0.0)


def test_the_profiler_counts_each_applied_solve_once():
    gm = _graph(diagnostics=True)
    report = profile_graph(gm, n_steps=2, n_warmup=0, counts=False,
                           measure_coupling=False, n_stat_steps=2 * DIVIDER)
    stats = report.coupling_iter_stats[KEY]
    assert stats["n"] == 2, stats
    assert stats["converged_fraction"] == 1.0, stats
    assert stats["at_cap_fraction"] == 0.0, stats


def test_the_sysid_mask_keeps_a_window_whose_applied_solve_converged():
    gm = _graph()
    init = {n: dict(gm.get_node_state(n)) for n in ("clock", "a", "b")}
    _final, history = gm.run_scan_with_history(DIVIDER)
    history = {k: v for k, v in history.items() if k != "_meta"}
    # Offset the measurements so the loss is not zero.
    history["a"] = {**history["a"], "x": history["a"]["x"] + 1.0}
    obs = observations_from_history(init, history)
    kw = dict(obs_fn=lambda s: s["a"]["x"], window=DIVIDER)
    unmasked = float(windowed_loss(gm, gm.params, obs, mask_unconverged=False, **kw))
    masked = float(windowed_loss(gm, gm.params, obs, mask_unconverged=True, **kw))
    assert unmasked > 0.0
    assert masked == unmasked
