"""A node that draws random numbers keeps its key's raw data in its state.

The convention (``docs/developer_guide/node_authoring.md``, "A node that
draws random numbers"): the state holds ``jax.random.key_data(key)``, a
``uint32`` array; ``update`` wraps it (``jax.random.wrap_key_data``), splits,
draws from the sub-key and returns the new key's data.  The seed is a
constructor parameter, kept as a Python ``int`` in ``self.params``.

One test per door the state goes through, each pinning a *property* of the
stream -- a new draw each time the node fires, the same stream from the same
seed, the stream of the same node run alone -- and never a drawn value: JAX
does not promise the values across versions, so the reference stream is
computed here, with the same three calls, outside any graph.

The fact the convention rests on inside a coupling group (CPL-193): every
pass of the solve calls a member's ``update`` with the state the step
started from, and the solvers iterate floating fields only, so a member
draws one sample per step, the same on every pass, and its key advances
once however many passes the solve takes and whatever its return rule
recomputes.

The contrast is pinned too: a *typed* key leaf (``jax.random.key``) steps
and scans, and ``save_state`` raises on it (MADD-ANO-171, open in 0.4.0).

The REST doors are in
``tests/api/test_a_node_that_draws_random_numbers_over_rest.py`` and the
USD round trip in ``tests/usd/test_usd_a_node_that_draws_random_numbers.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.simulation.checkpoint import (
    load_state_with_manifest,
    save_state_with_manifest,
)
from maddening.fmi import FmuTcpBridge, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import values_of
from maddening.sysid import fim, observations_from_history, windowed_loss
from maddening.testing.verification import verify_node

DT = 0.01
SEED = 7
#: The coupled pair below: ``a = OFF + K * reading`` and ``reading = G * a +
#: AMP * noise``, a loop of gain ``K * G = 0.3`` whose fixed point for one
#: sample is ``a = (OFF + K * AMP * noise) / (1 - K * G)``.
G, K, AMP, OFF = 0.6, 0.5, 2.0, 1.0


# ---------------------------------------------------------------------------
# The nodes
# ---------------------------------------------------------------------------


class NoisySensor(SimulationNode):
    """``reading = gain * signal + amplitude * N(0, 1)``, a new sample each firing.

    The node of the guide's section, with the raw sample kept as a field of
    its own so a test can read the stream.
    """

    def __init__(self, name, timestep, seed=0, amplitude=1.0, gain=1.0):
        super().__init__(name, timestep, seed=int(seed), amplitude=float(amplitude),
                         gain=float(gain))

    def initial_state(self):
        zero = jnp.zeros(())
        return {"reading": zero, "noise": zero,
                "key": jax.random.key_data(jax.random.key(self.params["seed"]))}

    def param_specs(self):
        return {**super().param_specs(),
                "seed": ParamSpec(trainable=False, description="PRNG seed")}

    def boundary_input_spec(self):
        return {"signal": BoundaryInputSpec(shape=(), default=0.0, description="measured")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        key, sub = jax.random.split(jax.random.wrap_key_data(state["key"]))
        noise = jax.random.normal(sub, (), state["noise"].dtype)
        signal = boundary_inputs.get("signal", jnp.zeros((), noise.dtype))
        return {"reading": p["gain"] * signal + p["amplitude"] * noise,
                "noise": noise,
                "key": jax.random.key_data(key)}


@stability(StabilityLevel.STABLE)
class ExportedNoisySensor(NoisySensor):
    """The sensor as a node an FMU exports: only ``STABLE`` nodes enter one."""


class TypedKeySensor(SimulationNode):
    """The contrast: the key itself in the state, not its data."""

    def __init__(self, name, timestep, seed=0):
        super().__init__(name, timestep, seed=int(seed))

    def initial_state(self):
        return {"noise": jnp.zeros(()), "key": jax.random.key(self.params["seed"])}

    def update(self, state, boundary_inputs, dt):
        key, sub = jax.random.split(state["key"])
        return {"noise": jax.random.normal(sub, (), state["noise"].dtype), "key": key}


class ConstantSensor(SimulationNode):
    """The sensor with its draw replaced by the constant ``c``: no key leaf."""

    def __init__(self, name, timestep, c=0.0, amplitude=1.0, gain=1.0):
        super().__init__(name, timestep, c=float(c), amplitude=float(amplitude),
                         gain=float(gain))

    def initial_state(self):
        return {"reading": jnp.zeros(()), "noise": jnp.zeros(())}

    def boundary_input_spec(self):
        return {"signal": BoundaryInputSpec(shape=(), default=0.0, description="measured")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        noise = jnp.asarray(p["c"], state["noise"].dtype)
        signal = boundary_inputs.get("signal", jnp.zeros((), noise.dtype))
        return {"reading": p["gain"] * signal + p["amplitude"] * noise, "noise": noise}


class Plant(SimulationNode):
    """``a = offset + k * reading``: what the sensor measures, and reads it back."""

    def __init__(self, name, timestep, k=K, offset=OFF):
        super().__init__(name, timestep, k=float(k), offset=float(offset))

    def initial_state(self):
        return {"a": jnp.zeros(())}

    def boundary_input_spec(self):
        return {"reading": BoundaryInputSpec(shape=(), default=0.0, description="fed back")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        reading = boundary_inputs.get("reading", jnp.zeros((), state["a"].dtype))
        return {"a": p["offset"] + p["k"] * reading}


class Decay(SimulationNode):
    """``x' = -rate * x`` by explicit Euler: a parameter upstream of the sensor."""

    def __init__(self, name, timestep, rate=1.0):
        super().__init__(name, timestep, rate=float(rate))

    def initial_state(self):
        return {"x": jnp.ones(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] - dt * p["rate"] * state["x"]}


# ---------------------------------------------------------------------------
# The reference stream, and the graphs
# ---------------------------------------------------------------------------


def stream(seed, n, dtype=np.float32):
    """``(the first n draws, the key data after them)`` of *seed*'s stream.

    The convention's own three calls, outside any graph: what a node that
    follows it draws, on whatever JAX this runs on.
    """
    key = jax.random.key(seed)
    draws = []
    for _ in range(n):
        key, sub = jax.random.split(key)
        draws.append(np.asarray(jax.random.normal(sub, (), dtype)))
    return np.asarray(draws, dtype), np.asarray(jax.random.key_data(key))


def assert_draws(got, want):
    """*got* is the stream *want*, to the rounding of one compiled draw.

    A draw inside a compiled step and the same draw made eagerly are the
    same sample; XLA may round its last bit differently, and the key data,
    which is integers, is compared exactly wherever a test has it.
    """
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-6)


def assert_new_draw_each_time(draws):
    draws = np.asarray(draws).ravel()
    assert len(set(draws.tolist())) == draws.size, draws


def _alone(seed=SEED, cls=NoisySensor, **kw):
    gm = GraphManager()
    gm.add_node(cls("n", DT, seed=seed, amplitude=AMP, **kw))
    gm.compile()
    return gm


def _pair(sensor, plant_dt=DT, **group):
    gm = GraphManager()
    gm.add_node(Plant("p", plant_dt))
    gm.add_node(sensor)
    gm.add_edge("p", "n", "a", "signal")
    gm.add_edge("n", "p", "reading", "reading")
    gm.add_coupling_group(["p", "n"], **group)
    gm.compile()
    return gm


def _criterion(norm):
    """A group's stopping knobs: each norm takes its own, and warns of the other's."""
    if norm == "l2":
        return dict(convergence_norm="l2", tolerance=1e-6)
    return dict(convergence_norm=norm, atol=1e-7, rtol=1e-5)


def _group(norm="l2", **kw):
    kw = dict(max_iterations=60, **_criterion(norm), **kw)
    if kw.get("acceleration") == "fixed":
        kw.setdefault("relaxation", 0.8)
    if kw.get("acceleration") == "iqn-imvj":
        kw.setdefault("jacobian_reuse", 2)
    return kw


def _fixed_points(draws):
    """The plant's value at the fixed point of each sample."""
    return (OFF + K * AMP * np.asarray(draws, np.float64)) / (1.0 - K * G)


def _report(gm):
    (report,) = gm.coupling_diagnostics().values()
    return report


@contextlib.contextmanager
def _x64():
    """``jax_enable_x64`` on for the block, put back after it."""
    before = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", before)


# ---------------------------------------------------------------------------
# Stepping, scanning, sweeping
# ---------------------------------------------------------------------------


def test_each_step_draws_the_next_sample_of_the_seeds_stream():
    want, key_after = stream(SEED, 4)
    gm = _alone()
    initial = gm.get_node_state("n")["key"]
    assert initial.dtype == jnp.uint32 and initial.shape == (2,)
    draws = [gm.step()["n"]["noise"] for _ in range(4)]
    assert_draws(draws, want)
    assert_new_draw_each_time(draws)
    key = gm.get_node_state("n")["key"]
    assert key.dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(key), key_after)


def test_a_scan_and_its_history_hold_the_stream():
    want, key_after = stream(SEED, 12)
    final = _alone().run_scan(12)
    assert_draws(final["n"]["noise"], want[-1])
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)

    final, history = _alone().run_scan_with_history(12)
    assert_draws(history["n"]["noise"], want)
    assert_new_draw_each_time(history["n"]["noise"])
    # The history holds the key after every step: any step can be resumed.
    assert history["n"]["key"].shape == (12, 2) and history["n"]["key"].dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(history["n"]["key"][4]), stream(SEED, 5)[1])
    # The same seed, the same stream; another seed, another one.
    _, again = _alone().run_scan_with_history(12)
    np.testing.assert_array_equal(np.asarray(again["n"]["noise"]),
                                  np.asarray(history["n"]["noise"]))
    _, other = _alone(seed=SEED + 1).run_scan_with_history(12)
    assert not np.any(np.asarray(other["n"]["noise"]) == np.asarray(history["n"]["noise"]))


def test_a_sweep_runs_each_key_as_its_own_stream():
    """``run_sweep`` over a batch of keys: each is the run of its own seed."""
    seeds = (3, SEED, 11)
    gm = _alone()
    batch = {"n": {
        "reading": jnp.zeros(3), "noise": jnp.zeros(3),
        "key": jnp.stack([jax.random.key_data(jax.random.key(s)) for s in seeds]),
    }}
    final, history = gm.run_sweep(6, batch, return_history=True)
    assert history["n"]["noise"].shape == (3, 6) and history["n"]["key"].shape == (3, 6, 2)
    for i, seed in enumerate(seeds):
        want, key_after = stream(seed, 6)
        assert_draws(history["n"]["noise"][i], want)
        np.testing.assert_array_equal(np.asarray(final["n"]["key"][i]), key_after)
        _, unbatched = _alone(seed=seed).run_scan_with_history(6)
        assert_draws(history["n"]["noise"][i], np.asarray(unbatched["n"]["noise"]))
    # A sweep does not advance the graph: its own key is where it was.
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("n")["key"]),
                                  stream(SEED, 0)[1])
    # ``jax.vmap`` and ``jax.jit`` of the node's own ``update``.
    node = gm.get_node("n")
    first = jax.vmap(lambda state: node.update(state, {}, DT))(batch["n"])
    assert_draws(first["noise"], [stream(s, 1)[0][0] for s in seeds])
    assert_draws(jax.jit(node.update)(node.initial_state(), {}, DT)["noise"],
                 stream(SEED, 1)[0][0])


# ---------------------------------------------------------------------------
# Checkpoints, configs, state reads and writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("manifest", [False, True], ids=["plain", "with-a-manifest"])
def test_a_checkpoint_continues_the_stream_in_a_fresh_graph(tmp_path, manifest):
    want, _ = stream(SEED, 9)
    gm = _alone()
    gm.run_scan(5)
    if manifest:
        path, _manifest = save_state_with_manifest(gm, tmp_path / "noise.npz")
    else:
        path = gm.save_state(tmp_path / "noise.npz")
    with np.load(path) as archive:
        assert archive["n/key"].dtype == np.uint32      # the key data, as it is
    fresh = _alone(seed=SEED + 1)                        # its own seed is overwritten
    if manifest:
        load_state_with_manifest(fresh, path)
    else:
        fresh.load_state(path)
    _, history = fresh.run_scan_with_history(4)
    assert history["n"]["key"].dtype == jnp.uint32
    assert_draws(history["n"]["noise"], want[5:])


def test_a_config_carries_the_seed_and_a_rebuilt_graph_restarts_the_stream():
    want, _ = stream(SEED, 6)
    gm = _alone()
    gm.run_scan(3)                       # a config holds the node, not its state
    config = json.loads(json.dumps(gm.to_dict()))
    (node,) = config["nodes"]
    assert node["params"]["seed"] == SEED and isinstance(node["params"]["seed"], int)
    rebuilt = GraphManager.from_dict(config, {"NoisySensor": NoisySensor})
    rebuilt.compile()
    assert rebuilt.get_node("n").params["seed"] == SEED
    _, history = rebuilt.run_scan_with_history(6)
    assert_draws(history["n"]["noise"], want)
    # ``reset_state`` restarts it too: the key is rebuilt from the seed.
    gm.reset_state()
    assert_draws(gm.step()["n"]["noise"], want[0])


def test_get_and_set_node_state_carry_the_key():
    want, key_after = stream(SEED, 7)
    gm = _alone()
    gm.run_scan(3)
    state = gm.get_node_state("n")
    assert state["key"].dtype == jnp.uint32
    np.testing.assert_array_equal(np.asarray(state["key"]), stream(SEED, 3)[1])
    other = _alone(seed=SEED + 1)
    other.set_node_state("n", state)
    final, history = other.run_scan_with_history(4)
    assert_draws(history["n"]["noise"], want[3:])
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)


# ---------------------------------------------------------------------------
# The seed is a constructor parameter, and not a parameter of the graph
# ---------------------------------------------------------------------------


def test_the_seed_is_rebuilt_from_a_config_and_no_fit_can_reach_it():
    """An ``int`` in ``self.params``, declared ``trainable=False``: structural.

    It is in the config (the test above) and nowhere in ``gm.params``, so
    the trainable mask, an FIM and a fit have no leaf to move.
    """
    gm = GraphManager()
    gm.add_node(Decay("d", DT, rate=2.0))
    gm.add_node(NoisySensor("n", DT, seed=SEED, amplitude=0.05, gain=1.5))
    gm.add_edge("d", "n", "x", "signal")
    gm.compile()
    assert sorted(gm.params["nodes"]["n"]) == ["amplitude", "gain"]
    assert sorted(gm.trainable_mask()["nodes"]["n"]) == ["amplitude", "gain"]
    assert gm.param_specs()["nodes"]["n"]["seed"].trainable is False

    # A record of the noisy graph, replayed: each window restarts from the
    # recorded state, key included, so the model draws the recorded samples
    # again and the loss is exactly zero at the parameters that made it.
    initial = {name: gm.get_node_state(name) for name in gm.node_names}
    _, history = gm.run_scan_with_history(16)
    record = observations_from_history(initial, history)
    assert record["n"]["key"].dtype == jnp.uint32

    def loss(p):
        return windowed_loss(gm, p, record, obs_fn=lambda h: h["n"]["reading"], window=4)

    assert float(loss(gm.params)) == 0.0
    report = fim(lambda p: jnp.atleast_1d(loss(p)), gm.params)
    assert not any("seed" in name for name in report.param_names), report.param_names
    assert any("amplitude" in name for name in report.param_names)


@pytest.mark.parametrize("spec", [ParamSpec(), None], ids=["a-trainable-spec", "a-float-seed"])
def test_a_seed_that_reaches_the_parameters_fails_verify_node(spec):
    """The two ways to get it wrong, and the check that says so.

    The default ``ParamSpec()`` is trainable, and an integer whose key is
    declared trainable is promoted to a float leaf; a seed stored as a
    ``float`` is a float leaf from the start.  Either way a fit is handed a
    leaf no path of ``update`` reads, and ``params_effective`` fails.
    """

    class WrongSeed(NoisySensor):
        def __init__(self, name, timestep, seed=0, amplitude=1.0, gain=1.0):
            super().__init__(name, timestep, seed, amplitude, gain)
            if spec is None:
                self.params["seed"] = float(seed)

        def initial_state(self):
            zero = jnp.zeros(())
            return {"reading": zero, "noise": zero,
                    "key": jax.random.key_data(jax.random.key(int(self.params["seed"])))}

        def param_specs(self):
            return {} if spec is None else {"seed": spec}

    gm = _alone(cls=WrongSeed)
    assert "seed" in gm.params["nodes"]["n"]
    assert gm.trainable_mask()["nodes"]["n"]["seed"]
    # 25 draws per check: the verdict is the same on every draw (no path
    # reads the leaf), so the count buys nothing here.
    result = verify_node(gm.get_node("n"), checks=["params_effective"], max_examples=25,
                         derandomize=True)["params_effective"]
    assert result.status == "FAIL" and "'seed'" in result.detail, result


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------


def test_the_battery_accepts_a_key_leaf_as_it_draws_it():
    """No recipe is needed for the default generator.

    An integer field is drawn over its dtype's whole range, and every pair
    of ``uint32`` words is valid key data for JAX's default (threefry)
    generator, so each drawn state is one ``update`` can split.  The checks
    that say something about a noise node: ``deterministic`` and
    ``jit_consistent`` (the sample is a function of the state),
    ``params_effective`` (the amplitude is read, the seed is not a leaf).
    """
    node = NoisySensor("n", DT, seed=SEED, amplitude=AMP)
    # 25 draws per check: each check compiles the node once, and a key leaf
    # either splits or raises on the first draw.
    results = verify_node(node, bounds={"reading": (-5.0, 5.0), "noise": (-5.0, 5.0)},
                          max_examples=25, derandomize=True)
    assert {name: r.status for name, r in results.items()} == dict.fromkeys(results, "PASS")
    assert {"deterministic", "jit_consistent", "gradient_finite",
            "params_effective"} <= set(results)


def test_the_battery_takes_a_strategy_that_draws_keys_from_seeds():
    """The recipe for a generator whose key data is not arbitrary words."""
    node = NoisySensor("n", DT, seed=SEED, amplitude=AMP)
    floats = st.floats(-5.0, 5.0, width=32).map(lambda v: jnp.asarray(v, jnp.float32))
    states = st.fixed_dictionaries({
        "reading": floats, "noise": floats,
        "key": st.integers(0, 2**31 - 1).map(
            lambda seed: jax.random.key_data(jax.random.key(seed))),
    })
    # 25 draws: as above.
    results = verify_node(node, state_strategy=states, max_examples=25, derandomize=True,
                          checks=["finite", "structure", "deterministic", "jit_consistent"])
    assert {name: r.status for name, r in results.items()} == dict.fromkeys(results, "PASS")


# ---------------------------------------------------------------------------
# A coupling-group member draws one sample per step (CPL-193)
# ---------------------------------------------------------------------------

_MODES = ("gauss-seidel", "jacobi")
_ACCELERATIONS = ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")
_SOLVERS = ("ift", "fori")
_NORMS = ("l2", "mixed", "interface")

#: Both schedules, both solvers and the three norms on every push; the
#: whole product, and the predictors, are slow.
_PER_PUSH_CELLS = (
    dict(iteration_mode="gauss-seidel", acceleration="none", solver="ift", norm="l2"),
    dict(iteration_mode="jacobi", acceleration="aitken", solver="fori", norm="mixed"),
    dict(iteration_mode="gauss-seidel", acceleration="iqn-ils", solver="ift", norm="interface"),
)
_EVERY_CELL = tuple(
    dict(iteration_mode=mode, acceleration=acceleration, solver=solver, norm=norm)
    for mode in _MODES for acceleration in _ACCELERATIONS
    for solver in _SOLVERS for norm in _NORMS
) + (
    dict(predictor="linear", norm="interface"),
    dict(predictor="quadratic", acceleration="aitken", norm="mixed"),
    dict(predictor="linear", solver="fori", norm="l2"),
)


def _cell_id(cell):
    return "-".join(str(v) for v in cell.values())


def _assert_one_sample_per_step(cell, steps=5):
    cell = dict(cell)
    # ``"ift"`` records its pass count and residual on every step; the
    # legacy ``"fori"`` loop only with ``diagnostics=True``.
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G),
               **_group(cell.pop("norm"), diagnostics=cell.get("solver") == "fori", **cell))
    final, history = gm.run_scan_with_history(steps)
    want, key_after = stream(SEED, steps)
    # One sample per step: the stream of the same node run alone ...
    _, alone = _alone().run_scan_with_history(steps)
    assert_draws(history["n"]["noise"], np.asarray(alone["n"]["noise"]))
    assert_draws(history["n"]["noise"], want)
    assert_new_draw_each_time(history["n"]["noise"])
    # ... its key advanced once per step, whatever the solve took ...
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)
    assert final["n"]["key"].dtype == jnp.uint32
    # ... and each step converged to the fixed point of that step's sample.
    report = _report(gm)
    assert report["converged"] and report["iterations"] >= 2, report
    np.testing.assert_allclose(np.asarray(history["p"]["a"]), _fixed_points(want), rtol=1e-4)


@pytest.mark.filterwarnings("ignore:CouplingGroup solver='fori':DeprecationWarning")
@pytest.mark.parametrize("cell", _PER_PUSH_CELLS, ids=_cell_id)
def test_a_group_member_draws_one_sample_per_step(cell):
    _assert_one_sample_per_step(cell)


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_a_group_member_draws_one_sample_per_step
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:CouplingGroup solver='fori':DeprecationWarning")
@pytest.mark.parametrize("cell", _EVERY_CELL, ids=_cell_id)
def test_a_group_member_draws_one_sample_per_step_under_every_schedule_solver_and_norm(cell):
    _assert_one_sample_per_step(cell)


def test_a_step_the_interface_norm_recomputes_fields_of_advances_the_key_once():
    """The return rule of ``convergence_norm="interface"`` is one more pass.

    A converged step returns the fields the norm does not measure whole
    (here ``noise``, which no edge reads) as one more plain pass computes
    them.  That pass starts from the pre-step state like every other, so it
    draws the step's sample again: the key has advanced once, and the next
    step draws the stream's next sample.
    """
    want, _ = stream(SEED, 3)
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G),
               **_group("interface"))
    first = gm.step()
    assert _report(gm)["converged"]
    assert_draws(first["n"]["noise"], want[0])
    np.testing.assert_array_equal(np.asarray(first["n"]["key"]), stream(SEED, 1)[1])
    assert_draws(gm.step()["n"]["noise"], want[1])
    assert_draws(gm.step()["n"]["noise"], want[2])
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("n")["key"]), stream(SEED, 3)[1])


def _assert_the_report_is_the_constant_twins(group, steps=3):
    noisy = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G), **group)
    twin = _pair(ConstantSensor("n", DT, amplitude=AMP, gain=G), **group)
    for _ in range(steps):
        before = {name: noisy.get_node_state(name) for name in ("p", "n")}
        after = noisy.step()
        report = _report(noisy)
        # The twin from the same state, with the sample as a constant.
        twin.set_node_state("p", before["p"])
        twin.set_node_state("n", {f: before["n"][f] for f in ("reading", "noise")})
        params = jax.tree.map(lambda leaf: leaf, twin.params)
        params["nodes"]["n"]["c"] = after["n"]["noise"]
        twin_after = twin.step(params=params)
        twin_report = _report(twin)
        assert report["converged"] and report["iterations"] >= 2, report
        assert set(report) == set(twin_report)
        for name, value in report.items():
            # Verdicts and counts exactly; a float to the rounding of two
            # programs that differ by the draw (NaN where both withhold it).
            if isinstance(value, float):
                assert value == pytest.approx(twin_report[name], rel=1e-3, abs=1e-12,
                                              nan_ok=True), (name, report, twin_report)
            else:
                assert value == twin_report[name], (name, report, twin_report)
        np.testing.assert_allclose(np.asarray(after["p"]["a"]),
                                   np.asarray(twin_after["p"]["a"]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(after["n"]["reading"]),
                                   np.asarray(twin_after["n"]["reading"]), rtol=1e-6)


def test_the_groups_report_is_that_of_the_noise_replaced_by_the_constant_it_drew():
    """The key leaf adds nothing to a residual, a pass count or a verdict."""
    _assert_the_report_is_the_constant_twins(_group("interface"))


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_the_groups_report_is_that_of_the_noise_replaced_by_the_constant_it_drew
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:CouplingGroup solver='fori':DeprecationWarning")
@pytest.mark.parametrize("group", [
    _group("l2"),
    _group("mixed", diagnostics=True, solver="fori"),
    _group("l2", acceleration="aitken", iteration_mode="jacobi"),
    _group("interface", acceleration="iqn-ils"),
], ids=["l2", "mixed-fori", "aitken-jacobi", "interface-iqn-ils"])
def test_the_groups_report_is_the_constant_twins_under_other_norms_and_accelerations(group):
    _assert_the_report_is_the_constant_twins(group)


# ---------------------------------------------------------------------------
# Multi-rate graphs and sub-cycled groups
# ---------------------------------------------------------------------------


def test_a_slow_node_advances_its_stream_only_when_it_fires():
    """Divider 4: twelve base steps are three draws, each held four steps."""
    want, key_after = stream(SEED, 3)
    gm = GraphManager()
    gm.add_node(Plant("p", DT))
    gm.add_node(NoisySensor("n", 4 * DT, seed=SEED, amplitude=AMP))
    gm.add_edge("p", "n", "a", "signal")
    gm.compile()
    assert gm.rate_dividers == {"p": 1, "n": 4}
    final, history = gm.run_scan_with_history(12)
    assert_draws(history["n"]["noise"], np.repeat(want, 4))
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)


def test_a_slow_group_draws_one_sample_per_solve_it_applies():
    """A group at divider 2 beside a faster node solves on every other step."""
    want, key_after = stream(SEED, 4)
    gm = GraphManager()
    gm.add_node(Plant("p", 2 * DT))
    gm.add_node(NoisySensor("n", 2 * DT, seed=SEED, amplitude=AMP, gain=G))
    gm.add_node(Plant("fast", DT))
    gm.add_edge("p", "n", "a", "signal")
    gm.add_edge("n", "p", "reading", "reading")
    gm.add_edge("n", "fast", "reading", "reading")
    gm.add_coupling_group(["p", "n"], **_group("l2"))
    gm.compile()
    assert gm.rate_dividers == {"p": 2, "n": 2, "fast": 1}
    final, history = gm.run_scan_with_history(8)
    assert_draws(history["n"]["noise"], np.repeat(want, 2))
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)
    np.testing.assert_allclose(np.asarray(history["p"]["a"]), np.repeat(_fixed_points(want), 2),
                               rtol=1e-4)


@pytest.mark.parametrize("fast, draws_per_step, sweeps", [("n", 4, 1), ("p", 1, 2)],
                         ids=["the-sensor-sub-cycles", "the-plant-sub-cycles-two-sweeps"])
def test_a_sub_cycled_member_draws_once_per_sub_step_and_the_same_on_every_pass(
        fast, draws_per_step, sweeps):
    """Four sub-steps per pass are four draws per step, not four per pass;
    and a second waveform sweep of the solve draws nothing new."""
    steps = 4
    want, key_after = stream(SEED, steps * draws_per_step)
    sensor_dt, plant_dt = (DT / 4, DT) if fast == "n" else (DT, DT / 4)
    gm = _pair(NoisySensor("n", sensor_dt, seed=SEED, amplitude=AMP, gain=G), plant_dt=plant_dt,
               **_group("l2", subcycling=True, boundary_interpolation="constant",
                        waveform_iterations=sweeps))
    final, history = gm.run_scan_with_history(steps)
    report = _report(gm)
    assert report["converged"] and report["iterations"] >= 2, report
    # The state after a step holds the last of that step's draws.
    assert_draws(history["n"]["noise"], want[draws_per_step - 1::draws_per_step])
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)


# ---------------------------------------------------------------------------
# Adaptive stepping
# ---------------------------------------------------------------------------


def _draws_consumed(key_data, seed=SEED, limit=64):
    """How many times the stream of *seed* was split to reach *key_data*."""
    key = jax.random.key(seed)
    for n in range(limit + 1):
        if np.array_equal(np.asarray(jax.random.key_data(key)), np.asarray(key_data)):
            return n
        key, _ = jax.random.split(key)
    raise AssertionError(f"not a key of seed {seed}'s stream within {limit} splits")


def test_an_adaptive_step_takes_two_draws_when_it_is_kept_and_none_when_rejected():
    """What ``run_adaptive`` does to the stream; stated, not endorsed.

    Each attempt is one full step (one draw, never kept) and two half steps
    (two draws, the first the full step's own).  A kept attempt is its two
    half steps, so the key advances twice; a rejected one is discarded
    whole, so the key stays and the retry draws the same samples.  The
    error estimate compares the full step with the half steps over the
    floating fields, and a field holding the sample differs between them by
    the noise itself, which reads as truncation error: the step is rejected
    down to ``dt_min`` and accepted there by force, with a warning.
    """
    want, _ = stream(SEED, 8)
    gm = _alone()
    kept = []
    with pytest.warns(UserWarning, match="Adaptive stepper hit dt_min"):
        final, info = gm.run_adaptive(
            4 * DT, dt_initial=4 * DT, dt_min=DT, dt_max=4 * DT,
            callback=lambda t, dt, state: kept.append(state["n"]["noise"]))
    assert info["n_rejected"] >= 1 and info["n_steps"] == 4, info
    assert info["dt_history"] == [DT] * 4
    assert _draws_consumed(final["n"]["key"]) == 2 * info["n_steps"]
    # After each kept step the state holds the second of its two draws.
    assert_draws(kept, want[1::2])


def test_the_adaptive_scan_takes_two_draws_per_kept_step_too():
    final, _history, info = _alone().run_adaptive_scan(
        4 * DT, max_steps=16, dt_initial=4 * DT, dt_min=DT, dt_max=4 * DT)
    assert int(info["n_steps"]) == 4
    assert _draws_consumed(final["n"]["key"]) == 8
    assert_draws(final["n"]["noise"], stream(SEED, 8)[0][-1])


# ---------------------------------------------------------------------------
# Gradients at a fixed seed
# ---------------------------------------------------------------------------


def _plain_for_gradients():
    gm = GraphManager()
    gm.add_node(Decay("d", 0.1, rate=2.0))
    gm.add_node(NoisySensor("n", 0.1, seed=SEED, amplitude=0.3, gain=1.5))
    gm.add_edge("d", "n", "x", "signal")
    gm.compile()
    return gm, ("n", "reading"), (("n", "amplitude"), ("d", "rate"), ("n", "gain"))


def _coupled_for_gradients():
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=0.3, gain=G),
               max_iterations=200, tolerance=1e-13, solver="ift")
    return gm, ("p", "a"), (("n", "amplitude"), ("p", "k"), ("p", "offset"))


@pytest.mark.parametrize("build", [_plain_for_gradients, _coupled_for_gradients],
                         ids=["plain", "through-an-ift-group"])
def test_the_gradient_at_a_fixed_seed_is_the_finite_difference_in_float64(build):
    """The noise is differentiated as a constant: the key is not a function
    of any parameter, so ``jax.grad`` is the derivative of this realisation.

    Against central differences under ``jax_enable_x64`` (float64): with
    respect to the noise amplitude and to a parameter upstream of the sensor
    (the decay rate; in the group, the plant's gain and offset).
    """
    with _x64():
        gm, (node, field), leaves = build()
        initial = jax.tree.map(lambda leaf: leaf[None],
                               {name: gm.get_node_state(name) for name in gm.node_names})
        assert initial["n"]["key"].dtype == jnp.uint32      # under x64 as well
        assert initial["n"]["noise"].dtype == jnp.float64

        def loss(params):
            # A sweep of one: it starts from ``initial`` every time and
            # leaves the graph where it was, so each call sees the same key.
            _, history = gm.run_sweep(3, initial, return_history=True, params=params)
            return jnp.sum(history[node][field] ** 2)

        base = gm.params
        gradient = jax.grad(loss)(base)

        def at(owner, leaf, value):
            params = jax.tree.map(lambda x: x, base)
            params["nodes"][owner][leaf] = jnp.asarray(value)
            return float(loss(params))

        for owner, leaf in leaves:
            value, h = float(base["nodes"][owner][leaf]), 1e-6
            central = (at(owner, leaf, value + h) - at(owner, leaf, value - h)) / (2 * h)
            got = float(gradient["nodes"][owner][leaf])
            assert abs(central) > 1e-3, (owner, leaf, central)      # a derivative to match
            assert got == pytest.approx(central, rel=1e-6), (owner, leaf)


def test_under_x64_the_key_stays_uint32_and_the_stream_is_drawn_in_float64():
    """The draw takes the state's dtype; the same key gives other numbers in it."""
    with _x64():
        want, key_after = stream(SEED, 4, np.float64)
        final, history = _alone().run_scan_with_history(4)
        assert final["n"]["key"].dtype == jnp.uint32
        assert history["n"]["noise"].dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(history["n"]["noise"]), want, rtol=1e-12)
        np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)
        # Not the float32 stream at a higher precision: other samples.
        single, single_key = stream(SEED, 4, np.float32)
        np.testing.assert_array_equal(single_key, key_after)
        assert not np.any(np.isclose(want, single, rtol=1e-3))


# ---------------------------------------------------------------------------
# The group member in the other domains (CPL-193)
# ---------------------------------------------------------------------------


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_under_x64_the_key_stays_uint32_and_the_stream_is_drawn_in_float64
@pytest.mark.slow
def test_a_group_member_draws_one_sample_per_step_under_x64():
    with _x64():
        want, key_after = stream(SEED, 4, np.float64)
        gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G),
                   max_iterations=200, tolerance=1e-12)
        final, history = gm.run_scan_with_history(4)
        assert final["n"]["key"].dtype == jnp.uint32
        assert history["n"]["noise"].dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(history["n"]["noise"]), want, rtol=1e-12)
        np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)
        assert _report(gm)["converged"]
        np.testing.assert_allclose(np.asarray(history["p"]["a"]), _fixed_points(want),
                                   rtol=1e-9)


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_a_sweep_runs_each_key_as_its_own_stream
@pytest.mark.slow
def test_a_sweep_of_a_group_runs_each_members_key_as_its_own_stream():
    """``run_sweep`` (a ``vmap`` of the scan) over a batch of keys, through a group."""
    seeds = (3, SEED, 11)
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G), **_group("l2"))
    keys = jnp.stack([jax.random.key_data(jax.random.key(s)) for s in seeds])
    batch = {"p": {"a": jnp.zeros(3)},
             "n": {"reading": jnp.zeros(3), "noise": jnp.zeros(3), "key": keys}}
    final, history = gm.run_sweep(4, batch, return_history=True)
    for i, seed in enumerate(seeds):
        want, key_after = stream(seed, 4)
        assert_draws(history["n"]["noise"][i], want)
        np.testing.assert_array_equal(np.asarray(final["n"]["key"][i]), key_after)
        np.testing.assert_allclose(np.asarray(history["p"]["a"][i]), _fixed_points(want),
                                   rtol=1e-4)


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_an_adaptive_step_takes_two_draws_when_it_is_kept_and_none_when_rejected
@pytest.mark.slow
def test_an_adaptive_step_of_a_group_is_two_solves_of_one_sample_each():
    """``run_adaptive`` keeps two half steps: two solves, each on its own sample."""
    want, _ = stream(SEED, 8)
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G), **_group("l2"))
    kept = []
    with pytest.warns(UserWarning, match="Adaptive stepper hit dt_min"):
        final, info = gm.run_adaptive(
            4 * DT, dt_initial=4 * DT, dt_min=DT, dt_max=4 * DT,
            callback=lambda t, dt, state: kept.append((state["n"]["noise"], state["p"]["a"])))
    assert info["n_rejected"] >= 1 and info["n_steps"] == 4, info
    assert _draws_consumed(final["n"]["key"]) == 8
    assert_draws([noise for noise, _ in kept], want[1::2])
    np.testing.assert_allclose(np.asarray([a for _, a in kept]), _fixed_points(want[1::2]),
                               rtol=1e-4)


# Per push: tests/core/test_a_node_that_draws_random_numbers.py::test_a_checkpoint_continues_the_stream_in_a_fresh_graph
@pytest.mark.slow
def test_a_checkpoint_between_coupled_steps_continues_a_members_stream(tmp_path):
    """``save_state`` after two coupled steps, ``load_state`` into a fresh pair."""
    want, key_after = stream(SEED, 5)
    group = _group("l2", predictor="linear")
    gm = _pair(NoisySensor("n", DT, seed=SEED, amplitude=AMP, gain=G), **group)
    gm.run_scan(2)
    path = gm.save_state(tmp_path / "pair.npz")
    fresh = _pair(NoisySensor("n", DT, seed=SEED + 1, amplitude=AMP, gain=G), **group)
    fresh.load_state(path)
    final, history = fresh.run_scan_with_history(3)
    assert_draws(history["n"]["noise"], want[2:])
    np.testing.assert_array_equal(np.asarray(final["n"]["key"]), key_after)
    np.testing.assert_allclose(np.asarray(history["p"]["a"]), _fixed_points(want[2:]), rtol=1e-4)


# ---------------------------------------------------------------------------
# FMU export
# ---------------------------------------------------------------------------


def _bridge(gm, description):
    """The guide's wiring of a graph's compiled step into an FMU's sidecar."""
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=description.instantiation_token, step_fn=gm._compiled_step,   # noqa: SLF001
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),  # noqa: SLF001
        fixed_params=description.fixed_parameters,
        input_resolver=gm._resolve_external_inputs))                             # noqa: SLF001
    return FmuTcpBridge(sidecar, description, master_dt=gm.timestep)


def test_an_fmu_exports_the_key_as_unsigned_integers_and_steps_the_stream():
    """The key data is a ``UInt32`` output; the seed is not an FMU parameter.

    An FMU state taken after four steps and set back after two more
    resumes the stream where it was taken.
    """
    want, _ = stream(SEED, 7)
    gm = _alone(cls=ExportedNoisySensor)
    description = build_model_description(gm, model_name="Noise")
    variables = {v.name: v for v in description.variables}
    assert variables["n.key"].dtype == "uint32" and variables["n.key"].causality == "output"
    assert not any("seed" in name for name in variables), sorted(variables)
    assert {"n.params.amplitude", "n.params.gain"} <= set(variables)
    bridge = _bridge(gm, description)

    def step_and_read(k):
        reply = bridge.handle({"op": "step", "t": k * DT, "dt": DT})
        assert reply["ok"], reply
        return values_of(bridge.handle({"op": "get", "type": "Float32",
                                        "vr": [variables["n.noise"].value_reference]}))[0]

    try:
        assert_draws([step_and_read(k) for k in range(4)], want[:4])
        key = values_of(bridge.handle({"op": "get", "type": "UInt32",
                                       "vr": [variables["n.key"].value_reference]}))
        np.testing.assert_array_equal(np.asarray(key, np.uint32), stream(SEED, 4)[1])
        taken = bridge.handle({"op": "get_state"})
        assert taken["ok"], taken
        for k in (4, 5):
            step_and_read(k)
        assert bridge.handle({"op": "set_state", "state": taken["state"]})["ok"]
        assert_draws([step_and_read(k) for k in range(4, 7)], want[4:7])
    finally:
        bridge.stop()


def test_an_fmu_leaves_out_a_noise_node_that_is_not_stable():
    """A new node is ``EXPERIMENTAL``: the FMU steps it and exports none of it."""
    gm = _alone()
    description = build_model_description(gm, model_name="Noise")
    assert [v.name for v in description.variables] == ["time"]


# ---------------------------------------------------------------------------
# The contrast: a typed key leaf
# ---------------------------------------------------------------------------


def _typed():
    gm = GraphManager()
    gm.add_node(TypedKeySensor("t", DT, seed=SEED))
    gm.compile()
    return gm


def test_a_typed_key_leaf_steps_and_scans_like_its_data():
    want, key_after = stream(SEED, 5)
    final, history = _typed().run_scan_with_history(5)
    assert jax.dtypes.issubdtype(final["t"]["key"].dtype, jax.dtypes.prng_key)
    assert_draws(history["t"]["noise"], want)
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(final["t"]["key"])), key_after)


def test_save_state_refuses_a_typed_key_leaf_in_these_words(tmp_path):
    """What MADD-ANO-171 reads like today: JAX's own refusal, nothing written."""
    gm = _typed()
    gm.step()
    with pytest.raises(TypeError, match=re.escape(
            "JAX array with PRNGKey dtype cannot be converted to a NumPy array")):
        gm.save_state(tmp_path / "typed.npz")
    assert not list(tmp_path.iterdir())


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "MADD-ANO-171: save_state cannot write a typed PRNG key leaf (np.asarray of a key "
    "array raises TypeError); deferred to 0.5.0.  Until then a node keeps the key's "
    "raw data, as NoisySensor does"))
def test_a_checkpoint_continues_the_stream_of_a_typed_key_leaf(tmp_path):
    want, _ = stream(SEED, 5)
    gm = _typed()
    gm.run_scan(2)
    path = gm.save_state(tmp_path / "typed.npz")
    fresh = _typed()
    fresh.load_state(path)
    _, history = fresh.run_scan_with_history(3)
    assert_draws(history["t"]["noise"], want[2:])


def test_a_complex_state_field_steps_alone_and_is_refused_inside_a_group():
    """The other dtype a state may hold in a plain graph and not in a group."""

    class Rotor(SimulationNode):
        def initial_state(self):
            return {"z": jnp.ones((), jnp.complex64)}

        def boundary_input_spec(self):
            return {"u": BoundaryInputSpec(shape=(), dtype=jnp.complex64, default=0.0)}

        def update(self, state, boundary_inputs, dt):
            u = boundary_inputs.get("u", jnp.zeros((), jnp.complex64))
            return {"z": 0.5j * state["z"] + 0.1 * u}

    gm = GraphManager()
    gm.add_node(Rotor("a", DT))
    gm.compile()
    assert complex(gm.step()["a"]["z"]) == 0.5j

    gm = GraphManager()
    gm.add_node(Rotor("a", DT))
    gm.add_node(Rotor("b", DT))
    gm.add_edge("a", "b", "z", "u")
    gm.add_edge("b", "a", "z", "u")
    gm.add_coupling_group(["a", "b"])
    gm.compile()
    with pytest.raises(TypeError, match=re.escape(
            "cannot carry a leaf of dtype complex64 through the coupling solver; "
            "supported: floating, bool, integer, typed PRNG keys")):
        gm.step()

