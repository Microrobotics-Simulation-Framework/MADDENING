"""A registered mapping kind in each numeric domain of the claims inventory.

``docs/validation/sysid_fmu_claims.yaml`` gives every row a domain matrix
(``testing_standards.md``, "The domain matrix"), and the registry's rows
state two things that depend on the domain a graph runs in:

* a mapping of a registered kind **round-trips**: the config carries its
  recipe, a reload rebuilds every weight bit for bit, at the weight's own
  dtype, and steps exactly as the graph it was saved from;
* its **weights are parameters like any other**: the step reads the live
  ones, a checkpoint carries each under its own name and restores it into
  a reload (where it wins over the rebuilt recipe), and the run continues
  as the graph that never stopped.

Both are stated once, in :func:`check_round_trip_and_checkpoint`, over a
two-member graph joined both ways by mappings of the registered kind
``inverse_distance`` (``tests/registered_mapping_kinds.py``: a mapping
class of its own with a matrix weight and a scalar one), and run in each
domain by building that graph differently:

* ``f32``: float32 members, x64 off (the default lane);
* ``f64``: under ``jax_enable_x64``, both members and every weight float64;
* ``mixed_dtype``: under x64, member ``a`` float32 beside a float64
  member ``b`` and float64 weights;
* ``multi_rate``: member ``b`` at twice member ``a``'s timestep;
* ``sub_cycled``: a coupling group with ``subcycling=True`` whose member
  ``b`` sub-steps twice per pass;
* ``predictors_warm_starts``: a group with a quadratic predictor, short of
  convergence, so the history it carries is read by the next step;
* ``checkpoint_restart``: the plain graph, with the checkpoint taken mid
  run and the reload run ahead before it loads.

``vmap`` and ``run_adaptive`` have their own tests below; the 16-bit
dtypes and sharded graphs are not exercised for a registered kind and the
rows narrow them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray
from tests.registered_mapping_kinds import (
    INVERSE_DISTANCE,
    KINDS,
    InverseDistanceMapping,
    assert_same_weights,
)

#: Exactly representable in every float dtype.
DT = 0.125
A2B = "a.x->b.inp"
B2A = "b.x->a.inp"


class Member(SimulationNode):
    """A vector that relaxes towards its mapped input, in a chosen dtype,
    publishing the points it lives on as static data."""

    def __init__(self, name, timestep, n=4, dtype="float32", rate=0.5):
        super().__init__(name, timestep, n=n, dtype=dtype, rate=rate)
        self._points = np.linspace(0.0, 1.0, n).astype(dtype)

    @property
    def static_data(self):
        return {"points": StaticArray(self._points)}

    def initial_state(self):
        n, dtype = self.params["n"], self.params["dtype"]
        return {"x": jnp.linspace(1.0, 2.0, n).astype(dtype)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],),
                                         dtype=jnp.dtype(self.params["dtype"]),
                                         description="the other member, mapped")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        rate = (self.params if params is None else params)["rate"]
        x = state["x"]
        # A mapped input arrives at the weights' dtype; the member keeps its own.
        target = boundary_inputs.get("inp", jnp.zeros_like(x)).astype(x.dtype)
        return {"x": (x + dt * rate * (target - x)).astype(x.dtype)}


REGISTRY = {"Member": Member}


@dataclasses.dataclass(frozen=True)
class Domain:
    """One numeric domain: the members' dtypes and how the graph is stepped."""

    name: str
    dtype_a: str = "float32"
    dtype_b: str = "float32"
    x64: bool = False
    dt_b: float = DT
    group: dict | None = None
    #: Steps the reload is run ahead before it loads the checkpoint.
    ahead: int = 0

    @property
    def weight_dtype(self) -> str:
        return "float64" if self.x64 else "float32"


DOMAINS = {d.name: d for d in (
    Domain("f32"),
    Domain("f64", "float64", "float64", x64=True),
    Domain("mixed_dtype", "float32", "float64", x64=True),
    Domain("multi_rate", dt_b=2 * DT),
    Domain("sub_cycled", dt_b=DT / 2,
           group=dict(subcycling=True, max_iterations=4, tolerance=1e-6)),
    Domain("predictors_warm_starts",
           group=dict(predictor="quadratic", max_iterations=2, tolerance=1e-12)),
    Domain("checkpoint_restart", ahead=3),
)}


@contextlib.contextmanager
def x64(on: bool):
    """``jax_enable_x64`` set to *on* for the block, restored after it."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def build(domain: Domain) -> GraphManager:
    """Members ``a`` (4 points) and ``b`` (6), each mapped onto the other
    by the registered kind from node references."""
    gm = GraphManager()
    gm.add_node(Member("a", DT, n=4, dtype=domain.dtype_a))
    gm.add_node(Member("b", domain.dt_b, n=6, dtype=domain.dtype_b, rate=0.25))
    make = KINDS[INVERSE_DISTANCE].build
    points = {n: gm.get_node(n).static_data["points"].value for n in ("a", "b")}
    for source, target, mode in (("a", "b", "consistent"), ("b", "a", "conservative")):
        gm.add_edge(source, target, "x", "inp", mapping=make(
            points[source], points[target], power=3.5, mode=mode,
            source_ref={"node": source, "field": "points"},
            target_ref={"node": target, "field": "points"}))
    if domain.group is not None:
        gm.add_coupling_group(["a", "b"], **domain.group)
    gm.compile()
    return gm


def _reload(gm: GraphManager) -> GraphManager:
    with warnings.catch_warnings():
        # The recipe is what is being reloaded; that the live weights
        # differ from it is asserted where it matters.
        warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
        config = json.loads(json.dumps(gm.to_dict()))
    reloaded = GraphManager.from_dict(config, REGISTRY)
    reloaded.compile()
    return reloaded


def _state(gm: GraphManager) -> dict:
    """Every leaf of the graph's state, ``_meta`` (a predictor's history,
    the multi-rate step counter) included, as host arrays."""
    return jax.tree.map(np.asarray, dict(gm._state))


def _assert_same_state(got: dict, expected: dict, what: str) -> None:
    flat_got = jax.tree_util.tree_leaves_with_path(got)
    flat_expected = jax.tree_util.tree_leaves_with_path(expected)
    assert [p for p, _ in flat_got] == [p for p, _ in flat_expected], what
    for (path, a), (_, b) in zip(flat_got, flat_expected):
        where = f"{what}: {jax.tree_util.keystr(path)}"
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert a.tobytes() == b.tobytes(), where


def _move_weights(gm: GraphManager) -> None:
    """Both weights of both edges, away from what the factory builds, as a
    fit would leave them."""
    for slot in gm.params["mappings"].values():
        slot["gain"] = (slot["gain"] * 0.75).astype(slot["gain"].dtype)
        slot["W"] = (slot["W"] * 1.25).astype(slot["W"].dtype)


def check_round_trip_and_checkpoint(domain: Domain, tmp_path) -> None:
    gm = build(domain)
    for key in (A2B, B2A):
        assert isinstance(gm.edges[0].mapping, InverseDistanceMapping)
        assert sorted(gm.params["mappings"][key]) == ["W", "gain"]
        for leaf in gm.params["mappings"][key].values():
            assert str(leaf.dtype) == domain.weight_dtype
    assert str(gm.get_node_state("a")["x"].dtype) == domain.dtype_a
    assert str(gm.get_node_state("b")["x"].dtype) == domain.dtype_b

    # 1. The recipe round-trips: same weights bit for bit, same steps.
    reloaded = _reload(gm)
    for key in (A2B, B2A):
        assert_same_weights(reloaded.params["mappings"][key], gm.params["mappings"][key],
                            what=f"rebuilt {key}")
        assert reloaded.edges[0].mapping.spec == gm.edges[0].mapping.spec
    gm.run(4)
    reloaded.run(4)
    _assert_same_state(_state(reloaded), _state(gm), "the reload after four steps")
    unmoved = _state(gm)

    # 2. Moved weights are live parameters: the step reads them ...
    gm.reset_state()
    _move_weights(gm)
    moved = {key: {n: np.asarray(v) for n, v in slot.items()}
             for key, slot in gm.params["mappings"].items()}
    gm.run(4)
    assert _state(gm)["b"]["x"].tobytes() != unmoved["b"]["x"].tobytes()

    # ... a checkpoint carries each under its own name; the config does not.
    path = gm.save_state(tmp_path / f"{domain.name}.npz")
    with pytest.warns(UserWarning, match=r"live mapping weights \['W', 'gain'\]"):
        gm.to_dict()
    fresh = _reload(gm)
    assert fresh.params["mappings"][A2B]["gain"].tobytes() != moved[A2B]["gain"].tobytes()
    if domain.ahead:
        fresh.run(domain.ahead)
    fresh.load_state(path)
    for key in (A2B, B2A):
        assert_same_weights(fresh.params["mappings"][key], moved[key],
                            what=f"restored {key}")
    _assert_same_state(_state(fresh), _state(gm), "the state the checkpoint restored")

    # ... and the restored graph continues as the one that never stopped.
    gm.run(3)
    fresh.run(3)
    _assert_same_state(_state(fresh), _state(gm), "three steps after the restart")


@pytest.mark.parametrize("name", sorted(DOMAINS))
def test_a_registered_kind_round_trips_and_its_weights_survive_a_checkpoint(name, tmp_path):
    """SYS-137 and SYS-139 in each domain: float32; float64 under x64; a
    float32 member beside a float64 one under x64 (mixed dtype); a
    multi-rate graph; a sub-cycled coupling group; a group with a predictor
    whose warm start the next step reads; and a checkpoint restart taken
    mid run into a reload that had run ahead."""
    domain = DOMAINS[name]
    with x64(domain.x64):
        check_round_trip_and_checkpoint(domain, tmp_path)


def test_a_predictor_group_carries_history_the_restart_has_to_restore(tmp_path):
    """The fixture can express the defect: in the ``predictors_warm_starts``
    graph the next step reads the history in ``_meta``, so a restart that
    dropped it would not continue as the uninterrupted run."""
    gm = build(DOMAINS["predictors_warm_starts"])
    gm.run(4)
    warm = dict(gm._state)
    cold = {**warm, "_meta": {k: (v if k == "step_count" else jnp.zeros_like(v))
                              for k, v in warm["_meta"].items()}}
    step, ext = gm._build_step_fn(), gm._default_external_inputs()
    a, b = step(warm, ext, gm.params), step(cold, ext, gm.params)
    assert np.asarray(a["b"]["x"]).tobytes() != np.asarray(b["b"]["x"]).tobytes()


@pytest.mark.parametrize("name", ["f32", "f64", "mixed_dtype"])
def test_a_gradient_reaches_every_weight_of_a_registered_kind_at_its_dtype(name):
    """A derivative with respect to each weight, at the weight's own dtype
    (float64 under x64, beside a float32 member in the mixed case)."""
    domain = DOMAINS[name]
    with x64(domain.x64):
        gm = build(domain)
        step, ext, state0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state

        def loss(params):
            state = step(step(state0, ext, params), ext, params)
            return jnp.sum(state["b"]["x"].astype(jnp.result_type(float)) ** 2)

        grads = jax.jit(jax.grad(loss))(gm.params)["mappings"]
        for key in (A2B, B2A):
            for weight, g in grads[key].items():
                assert str(g.dtype) == domain.weight_dtype
                assert g.shape == gm.params["mappings"][key][weight].shape
                assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_a_registered_kinds_weights_batch_under_vmap_as_each_member_alone():
    """``jax.vmap`` over a batch of the scalar weight ``gain`` and of the
    matrix weight ``W``: each member of the batch is the step taken with
    that member's weights alone."""
    gm = build(DOMAINS["f32"])
    step, ext, state0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state

    def after(gain, scale):
        params = jax.tree.map(lambda x: x, gm.params)
        params["mappings"][A2B] = {"W": scale * params["mappings"][A2B]["W"],
                                   "gain": gain}
        state = step(step(state0, ext, params), ext, params)
        return state["b"]["x"]

    gains = jnp.asarray([0.5, 1.0, 1.5, -2.0], jnp.float32)
    scales = jnp.asarray([1.0, 0.25, 2.0, 1.0], jnp.float32)
    batched = np.asarray(jax.jit(jax.vmap(after))(gains, scales))
    assert batched.shape == (4, 6)
    alone = np.stack([np.asarray(jax.jit(after)(g, s)) for g, s in zip(gains, scales)])
    # The same arithmetic in another order of evaluation: equal to a few ulp.
    np.testing.assert_allclose(batched, alone, rtol=4 * np.finfo(np.float32).eps)
    assert len({row.tobytes() for row in batched}) == 4       # the weights were read


def test_run_adaptive_reads_a_registered_kinds_restored_weights(tmp_path):
    """``run_adaptive`` on a reload that restored trained weights from a
    checkpoint takes the steps, and reaches the state, of the graph the
    weights were trained in -- and not those of the rebuilt recipe."""
    kwargs = dict(dt_initial=0.05, dt_max=0.1, atol=1e-6, rtol=1e-4)
    gm = build(DOMAINS["f32"])
    _move_weights(gm)
    path = gm.save_state(tmp_path / "trained.npz")
    restored = _reload(gm)
    restored.load_state(path)
    recipe = _reload(gm)

    _, info = gm.run_adaptive(0.5, **kwargs)
    _, info_restored = restored.run_adaptive(0.5, **kwargs)
    recipe.run_adaptive(0.5, **kwargs)
    assert info["dt_history"] == info_restored["dt_history"] and info["dt_history"]
    _assert_same_state(_state(restored), _state(gm), "after run_adaptive")
    assert _state(recipe)["b"]["x"].tobytes() != _state(gm)["b"]["x"].tobytes()


# ---------------------------------------------------------------------------
# SYS-138: what add_edge accepts as a weight, by dtype
# ---------------------------------------------------------------------------

class _Weighted:
    """A mapping class of the caller's own around one weight ``H``."""

    kind = "weighted"
    mode = "consistent"
    n_source = 4
    n_target = 6
    spec = None

    def __init__(self, weight):
        self._weight = weight

    def params_pytree(self):
        return {"H": self._weight}

    def apply(self, field, weights=None, geom=None):
        return (self._weight if weights is None else weights["H"]) @ field

    def apply_T(self, field, weights=None, geom=None):
        return (self._weight if weights is None else weights["H"]).T @ field

    def __repr__(self):
        return "_Weighted()"


def _add(weight) -> GraphManager:
    gm = GraphManager()
    gm.add_node(Member("a", DT, n=4))
    gm.add_node(Member("b", DT, n=6))
    gm.add_edge("a", "b", "x", "inp", mapping=_Weighted(weight))
    return gm


#: Every floating dtype JAX holds at each setting of ``jax_enable_x64``
#: (float64 exists only with it on).
_WIDTHS = [(False, "float16"), (False, "bfloat16"), (False, "float32"),
           (True, "float16"), (True, "bfloat16"), (True, "float32"), (True, "float64")]


@pytest.mark.parametrize("enabled, dtype", _WIDTHS,
                         ids=[f"x64-{'on' if on else 'off'}-{d}" for on, d in _WIDTHS])
def test_a_floating_point_weight_of_any_width_is_accepted(enabled, dtype):
    """Every floating dtype JAX can hold at the current setting: float16,
    bfloat16 and float32 with x64 off or on (a float32 weight in an x64
    process is the mixed-dtype case), float64 under x64."""
    with x64(enabled):
        weight = jnp.ones((6, 4), dtype=dtype)
        assert str(weight.dtype) == dtype
        gm = _add(weight)
        gm.compile()
        assert gm.params["mappings"][A2B]["H"].dtype == weight.dtype
        assert np.all(np.isfinite(np.asarray(gm.step()["b"]["x"], dtype=np.float64)))


@pytest.mark.parametrize("enabled", [False, True], ids=["x64-off", "x64-on"])
@pytest.mark.parametrize("weight, message", [
    (lambda: jnp.ones((6, 4), dtype=jnp.int32), "entry 'H' has dtype int32"),
    (lambda: jnp.ones((6, 4), dtype=jnp.result_type(int)), "entry 'H' has dtype int"),
    (lambda: jnp.ones((6, 4), dtype=bool), "entry 'H' has dtype bool"),
    (lambda: jnp.ones((6, 4), dtype=jnp.result_type(complex)), "entry 'H' has dtype complex"),
    (lambda: np.ones((6, 4), dtype=np.float64), "entry 'H' is ndarray, not a JAX array"),
    (lambda: np.ones((6, 4), dtype=np.float32), "entry 'H' is ndarray, not a JAX array"),
    (lambda: jnp.full((6, 4), jnp.nan), "holds a non-finite value"),
], ids=["int32", "the default int", "bool", "the default complex", "numpy float64",
        "numpy float32", "nan"])
def test_a_weight_that_is_not_a_finite_floating_jax_array_is_refused_in_float64_too(
        weight, message, enabled):
    """The refusals do not depend on ``jax_enable_x64``: an integer (int64
    under x64), a boolean, a complex or a NumPy weight, and a NaN at
    float64, are refused where the edge is added."""
    with x64(enabled):
        with pytest.raises(ValueError, match=message):
            _add(weight())
