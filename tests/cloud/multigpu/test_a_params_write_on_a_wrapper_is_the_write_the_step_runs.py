"""A ``node.params`` write made on a node that wraps another is the write
the step runs, whichever of the two it is made on and however it is spelt.

The guide documents three spellings of a write to a node's constants
(``docs/user_guide/parameters.md``, "Live values, recompiles and partial
pytrees"): the keyed write ``node.params["k"] = v``, the assignment
``node.params = {**node.params, "k": v}``, and an element written in place
(``node.params["rates"][0] = v``).  In a sharded graph the node
``gm.get_node(name)`` returns is the wrapper.  The wrappers used to *share*
the wrapped node's mapping from construction on, so the keyed write
worked; the assignment replaced the wrapper's mapping alone, and from then
on the step ran the old value while ``node.params`` read the new one back,
``gm.params`` kept the old one, ``to_dict()`` saved the one that never ran,
and every later keyed write through the wrapper was lost with it.  The same
assignment made on the wrapped node was lost the other way round.

Now a wrapper has no mapping of its own (``SimulationNode._params_owner_attr``):
one mapping, however it is reached.  Each case here applies one spelling,
on the wrapper or on the node it wraps, before the first step or after
steps have traced the step with the old value, and holds the run that
follows to three references:

* a graph of the same wrapper around a node *constructed* with the written
  value, started from the state the written graph was in: **bit for bit**
  (the same program on the same devices with the same constant, so nothing
  may differ);
* an *unsharded* twin given the same statements: to the tolerance the
  tree's sharded-equivalence tests use (``RTOL``, ``ATOL`` of
  ``test_property_sharded_equals_unsharded.py``), because a sharded and an
  unsharded program are different XLA programs and may round differently;
* the run at the old value, which it must NOT match: the fixture can
  express a lost write (the two values move the state by a factor of 0.95
  and of 0.5 a step).

Both parameter contracts: a node whose ``update`` takes ``params`` (the
write is copied into the live ``gm.params`` leaf, no recompile) and one on
the three-argument contract (the constant is baked into the trace, so the
write marks the graph dirty and the next run recompiles): the write counter
and the staleness rule of an unsharded node.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, _ParamsDict
from maddening.core.simulation.hybrid_node import HybridNode

N = 16
DT = 0.1
#: The constructor's rate, a value in between, and the written one: all
#: exact in float32, so "the value the step runs" has one spelling.
OLD, MID, NEW = 0.5, 2.0, 5.0
STEPS_BEFORE, STEPS_AFTER = 2, 3
#: What "agree" means for a sharded run against its unsharded twin: the
#: sharded-equivalence tolerance of ``test_property_sharded_equals_unsharded.py``.
RTOL, ATOL = 1e-5, 1e-6

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


def _factor(p, dt):
    return 1.0 - dt * (p["rate"] + jnp.sum(jnp.asarray(p["extra"], jnp.float32)))


class _DecayBase(SimulationNode):
    """``x <- x * (1 - dt * (rate + sum(extra)))``, cell by cell.

    Pointwise, so the one node runs under all three sharded wrappers: its
    ``update_padded`` reads the Cartesian layout and the partition layout
    alike.  ``halo`` is what the wrapper under test needs it to declare
    (the stencil wrapper wants a halo on the sharded axis, the other two
    want none); it is set by ``_wrap``, not constructed, and no number
    depends on it.  ``extra`` is a list, for the element written in place."""

    halo: dict = {}

    def __init__(self, name="n", timestep=DT, rate=OLD, extra=(0.0,)):
        super().__init__(name, timestep, rate=rate, extra=list(extra))

    def initial_state(self):
        return {"x": jnp.arange(1, N + 1, dtype=jnp.float32)}

    def halo_width(self):
        return dict(self.halo)


class _Decay(_DecayBase):
    """On the params contract: the graph injects the constants."""

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] * _factor(p, dt)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state_padded["x"] * _factor(p, dt)}


class _BakedDecay(_DecayBase):
    """On the three-argument contract: the constants are read from
    ``self.params`` when the step is traced."""

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * _factor(self.params, dt)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        return {"x": state_padded["x"] * _factor(self.params, dt)}


CONTRACTS = {"params": _Decay, "three_argument": _BakedDecay}


def _no_correction(state, boundary_inputs, dt):
    return {}


def _layout(n):
    owner = np.repeat(np.arange(n), N // n).astype(np.int32)
    edges = np.array([[i, (i + 1) % N] for i in range(N)], dtype=np.int32)
    return build_unstructured_partition(partition_assignment=owner, edges=edges, n_devices=n)


def _global_x(state):
    return state["x"]


def _wrap(kind, inner, n):
    """``(the node the graph holds, state -> x in global cell order)``."""
    if kind == "unsharded":
        return inner, _global_x
    if kind == "hybrid":
        return HybridNode(inner, _no_correction), _global_x
    mesh = create_device_mesh(shape=(n,))
    if kind in ("pointwise", "hybrid_over_pointwise"):
        node, read = ShardedPointwiseNode(inner, mesh, shard_axes=(0,)), _global_x
    elif kind in ("stencil", "hybrid_over_stencil"):
        inner.halo = {0: 1}
        node = ShardedStencilNode(inner, mesh, {"devices": 0}, boundary="periodic")
        read = _global_x
    else:
        assert kind in ("unstructured", "hybrid_over_unstructured"), kind
        node = ShardedUnstructuredNode(inner, mesh, _layout(n))

        def read(state, node=node):
            return node.gather_global(state)["x"]
    if kind.startswith("hybrid_over_"):
        node = HybridNode(node, _no_correction)
    return node, read


def _graph(kind, n, contract, **constructed):
    """A compiled graph of one node named ``n``: ``(gm, the wrapped node,
    read)``."""
    inner = CONTRACTS[contract](**constructed)
    node, read = _wrap(kind, inner, n)
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm, inner, read


def _run(gm, read, steps):
    for _ in range(steps):
        gm.step()
    return np.asarray(read(gm.get_node_state("n")))


# -- the spellings ---------------------------------------------------------
# Each takes the graph and the WRAPPED node (the object a user who built
# ``heat = HeatNode(...)`` and wrapped it still holds), and leaves the model
# at a total rate of NEW.  ``gm.get_node("n")`` is the wrapper in a sharded
# graph and the node itself in the unsharded twin.

def _keyed(gm, inner):
    gm.get_node("n").params["rate"] = NEW


def _assigned(gm, inner):
    node = gm.get_node("n")
    node.params = {**node.params, "rate": NEW}


def _assigned_unchanged_then_keyed(gm, inner):
    node = gm.get_node("n")
    node.params = {**node.params}
    gm.get_node("n").params["rate"] = NEW


def _assigned_then_keyed_on_the_wrapped_node(gm, inner):
    node = gm.get_node("n")
    node.params = {**node.params, "rate": MID}
    inner.params["rate"] = NEW


def _keyed_on_the_wrapped_node(gm, inner):
    inner.params["rate"] = NEW


def _assigned_on_the_wrapped_node(gm, inner):
    inner.params = {**inner.params, "rate": NEW}


def _assigned_on_the_wrapped_node_then_keyed(gm, inner):
    inner.params = {**inner.params, "rate": MID}
    gm.get_node("n").params["rate"] = NEW


def _a_counting_mapping_assigned(gm, inner):
    node = gm.get_node("n")
    node.params = _ParamsDict({**node.params, "rate": NEW})


def _element_in_place(gm, inner):
    gm.get_node("n").params["extra"][0] = NEW - OLD


def _element_in_place_after_an_assignment(gm, inner):
    node = gm.get_node("n")
    node.params = {**node.params}
    gm.get_node("n").params["extra"][0] = NEW - OLD


#: ``name -> (the statements, the key written, its value afterwards)``.
SPELLINGS = {
    "keyed": (_keyed, "rate", NEW),
    "assigned": (_assigned, "rate", NEW),
    "assigned_unchanged_then_keyed": (_assigned_unchanged_then_keyed, "rate", NEW),
    "assigned_then_keyed_on_the_wrapped_node": (
        _assigned_then_keyed_on_the_wrapped_node, "rate", NEW),
    "keyed_on_the_wrapped_node": (_keyed_on_the_wrapped_node, "rate", NEW),
    "assigned_on_the_wrapped_node": (_assigned_on_the_wrapped_node, "rate", NEW),
    "assigned_on_the_wrapped_node_then_keyed": (
        _assigned_on_the_wrapped_node_then_keyed, "rate", NEW),
    "a_counting_mapping_assigned": (_a_counting_mapping_assigned, "rate", NEW),
    "element_in_place": (_element_in_place, "extra", [NEW - OLD]),
    "element_in_place_after_an_assignment": (
        _element_in_place_after_an_assignment, "extra", [NEW - OLD]),
}

#: ``(wrapper, devices)``: every wrapper that wraps a node, the three sharded
#: ones on 1, 2 and 4 devices, and a wrapper around a wrapper.
WRAPPERS = (
    [(kind, n) for kind in ("pointwise", "stencil", "unstructured") for n in (1, 2, 4)]
    + [("hybrid", 1), ("hybrid_over_pointwise", 2), ("hybrid_over_stencil", 2),
       ("hybrid_over_unstructured", 2)]
)
WHEN = ("before_the_first_step", "after_steps")

#: Per push: the assignment that was lost and the keyed write it took with
#: it, on each wrapper at two devices and through a wrapper around a
#: wrapper, both contracts, before and after steps.  The whole table is the
#: slow test below.
PER_PUSH_WRAPPERS = [("pointwise", 2), ("stencil", 2), ("unstructured", 2), ("hybrid", 1),
                     ("hybrid_over_stencil", 2)]
PER_PUSH_SPELLINGS = ["assigned", "assigned_unchanged_then_keyed",
                      "assigned_on_the_wrapped_node"]


def _assert_everything_reports(gm, inner, contract, key, value):
    """What reports the parameter says what was written: the node, the node
    it wraps, the saved config, and -- for a constant the graph carries --
    ``gm.params`` and the inspection table."""
    node = gm.get_node("n")
    assert node.params is inner.params, "one mapping, however it is reached"
    assert node.params[key] == value and inner.params[key] == value
    saved = next(d for d in gm.to_dict()["nodes"] if d["name"] == "n")["params"][key]
    assert saved == value
    if contract == "params":
        leaf = np.asarray(gm.params["nodes"]["n"][key])
        assert leaf.tolist() == np.asarray(value, np.float32).tolist()
        rows = [r for r in gm.params_table() if r["owner"] == "n" and r["param"] == key]
        assert len(rows) == 1 and rows[0]["value"] == np.float32(np.ravel(value)[0])
    else:
        assert "n" not in gm.params["nodes"] or key not in gm.params["nodes"]["n"]


def _check(kind, n, contract, spelling, when):
    write, key, value = SPELLINGS[spelling]
    gm, inner, read = _graph(kind, n, contract)
    if when == "after_steps":
        _run(gm, read, STEPS_BEFORE)          # the step is traced with the old value
    start = gm.get_node_state("n")
    start_x = np.asarray(read(start))

    write(gm, inner)
    _assert_everything_reports(gm, inner, contract, key, value)
    written = _run(gm, read, STEPS_AFTER)
    _assert_everything_reports(gm, inner, contract, key, value)

    # The same wrapper around a node constructed with the value: bit for bit.
    constructed, _, constructed_read = _graph(kind, n, contract, **{key: value})
    constructed.set_node_state("n", start)
    reference = _run(constructed, constructed_read, STEPS_AFTER)
    assert np.array_equal(written, reference), (
        f"{kind} on {n} device(s), {contract} contract, {spelling} {when}: the step did "
        f"not run the written value (max difference "
        f"{float(np.max(np.abs(written - reference))):.3e})")

    # An unsharded twin given the same statements: to the sharded tolerance.
    twin, twin_inner, twin_read = _graph("unsharded", 1, contract)
    if when == "after_steps":
        _run(twin, twin_read, STEPS_BEFORE)
    write(twin, twin_inner)
    np.testing.assert_allclose(written, _run(twin, twin_read, STEPS_AFTER),
                               rtol=RTOL, atol=ATOL)

    # And not the old value's run: a lost write would be seen.
    lost = start_x * np.float32(1.0 - DT * OLD) ** STEPS_AFTER
    assert not np.allclose(written, lost, rtol=1e-2, atol=ATOL)


@pytest.mark.parametrize("when", WHEN)
@pytest.mark.parametrize("spelling", PER_PUSH_SPELLINGS)
@pytest.mark.parametrize("contract", list(CONTRACTS))
@pytest.mark.parametrize("kind, n", PER_PUSH_WRAPPERS,
                         ids=[f"{k}-{n}" for k, n in PER_PUSH_WRAPPERS])
def test_an_assignment_on_a_wrapper_or_on_its_node_is_the_write_the_step_runs(
        kind, n, contract, spelling, when):
    _check(kind, n, contract, spelling, when)


# Per push: test_an_assignment_on_a_wrapper_or_on_its_node_is_the_write_the_step_runs
@pytest.mark.slow
@pytest.mark.parametrize("contract", list(CONTRACTS))
@pytest.mark.parametrize("kind, n", WRAPPERS, ids=[f"{k}-{n}" for k, n in WRAPPERS])
def test_every_spelling_of_a_write_on_every_wrapper_is_the_write_the_step_runs(
        kind, n, contract):
    """Every spelling, before the first step and after steps, on every
    wrapper and device count: 1, 2 and 4 devices for the three sharded
    wrappers."""
    for spelling in SPELLINGS:
        for when in WHEN:
            _check(kind, n, contract, spelling, when)


@pytest.mark.parametrize("kind, n", PER_PUSH_WRAPPERS,
                         ids=[f"{k}-{n}" for k, n in PER_PUSH_WRAPPERS])
def test_every_entry_point_runs_the_value_assigned_on_a_wrapper(kind, n):
    """A node on the three-argument contract has its constant baked into
    whichever program is traced next, so a lost or half-taken write shows
    as entry points that disagree (the shape of the unsharded defect the
    write counter was built for): after the assignment ``step``, ``run``
    and a ``run_scan`` at a length never traced all run the written
    value."""
    def after(entry):
        gm, inner, read = _graph(kind, n, "three_argument")
        _run(gm, read, STEPS_BEFORE)
        gm.run_scan(1)                                  # a scan traced with the old value
        gm.reset_state()
        node = gm.get_node("n")
        node.params = {**node.params, "rate": NEW}
        entry(gm)
        return np.asarray(read(gm.get_node_state("n")))

    want = np.arange(1, N + 1, dtype=np.float32) * np.float32(1.0 - DT * NEW) ** 2
    for label, entry in (("step", lambda gm: (gm.step(), gm.step())),
                         ("run", lambda gm: gm.run(2)),
                         ("run_scan, a length traced before", lambda gm: (gm.run_scan(1),
                                                                         gm.run_scan(1))),
                         ("run_scan, a new length", lambda gm: gm.run_scan(2))):
        np.testing.assert_allclose(after(entry), want, rtol=RTOL, atol=ATOL, err_msg=label)


@pytest.mark.parametrize("kind, n", PER_PUSH_WRAPPERS,
                         ids=[f"{k}-{n}" for k, n in PER_PUSH_WRAPPERS])
def test_a_checkpoint_carries_the_value_assigned_on_a_wrapper(kind, n, tmp_path):
    """``save_state`` after the assignment, loaded into a graph built with
    the constructor's value: the loaded graph holds, and runs, the written
    one."""
    gm, inner, read = _graph(kind, n, "params")
    node = gm.get_node("n")
    node.params = {**node.params, "rate": NEW}
    path = gm.save_state(tmp_path / "written.npz")
    written = _run(gm, read, STEPS_AFTER)

    fresh, _, fresh_read = _graph(kind, n, "params")
    fresh.load_state(path)
    assert float(fresh.params["nodes"]["n"]["rate"]) == NEW
    assert np.array_equal(_run(fresh, fresh_read, STEPS_AFTER), written)


@pytest.mark.parametrize("kind, n", PER_PUSH_WRAPPERS,
                         ids=[f"{k}-{n}" for k, n in PER_PUSH_WRAPPERS])
def test_an_assignment_on_a_wrapper_leaves_the_other_calibrated_constants_alone(kind, n):
    """The assignment is a write of the keys it changes and of no other, on
    a wrapper as on a node: a constant calibrated in ``gm.params`` beside
    the key assigned stays calibrated, and handing a key back with the
    value the node holds does not put that value over its calibration."""
    gm, inner, read = _graph(kind, n, "params")
    gm.params["nodes"]["n"]["extra"] = jnp.asarray([1.5], jnp.float32)     # calibrated
    node = gm.get_node("n")
    node.params = {**node.params, "rate": NEW}
    leaves = gm.params["nodes"]["n"]
    assert float(leaves["rate"]) == NEW and leaves["extra"].tolist() == [1.5]
    gm.params["nodes"]["n"]["rate"] = jnp.float32(MID)                      # calibrated too
    node = gm.get_node("n")
    node.params = {**node.params}                                           # changes nothing
    assert float(gm.params["nodes"]["n"]["rate"]) == MID
    gm.get_node("n").params["rate"] = NEW          # the keyed write is a write, whatever the value
    assert float(gm.params["nodes"]["n"]["rate"]) == NEW


def test_a_wrapper_has_no_mapping_of_its_own():
    """However deep the wrapping, ``params`` is the innermost node's one
    mapping, and an assignment anywhere replaces it for all of them: the
    writes counted before it are carried, and the key it changes is counted
    once."""
    inner = _Decay()
    inner.halo = {0: 1}
    stencil = ShardedStencilNode(inner, create_device_mesh(shape=(2,)), {"devices": 0},
                                 boundary="periodic")
    hybrid = HybridNode(stencil, _no_correction)
    assert hybrid.params is stencil.params is inner.params
    assert "params" not in vars(hybrid) and "params" not in vars(stencil)
    inner.params["rate"] = MID                                    # one counted write
    before = inner.params
    hybrid.params = {**hybrid.params, "rate": NEW}
    assert hybrid.params is stencil.params is inner.params is not before
    assert type(inner.params) is _ParamsDict and inner.params["rate"] == NEW
    assert inner.params._write_counts() == {"rate": 2}
    stencil.params = {**stencil.params}                           # hands every key back
    assert inner.params._write_counts() == {"rate": 2}
    inner.params = {**inner.params, "extra": [1.0]}
    assert hybrid.params["extra"] == [1.0] and stencil.params is inner.params
    assert inner.params._write_counts() == {"rate": 2, "extra": 1}


class _SharingWrapper(SimulationNode):
    """A wrapper written outside the package, sharing its node's mapping by
    the idiom the built-in ones used: ``self.params = inner.params``."""

    def __init__(self, inner):
        super().__init__(inner.name, inner.delta_t)
        self.wrapped = inner
        self.params = inner.params

    def initial_state(self):
        return self.wrapped.initial_state()

    def accepts_params(self, *, method="update"):
        return self.wrapped.accepts_params(method=method)

    def params_pytree(self):
        return self.wrapped.params_pytree()

    def update(self, state, boundary_inputs, dt, *, params=None):
        return self.wrapped.update(state, boundary_inputs, dt, params=params)


@pytest.mark.parametrize("spelling", ["assigned", "assigned_unchanged_then_keyed",
                                      "a_counting_mapping_assigned"])
def test_an_assignment_on_a_wrapper_that_shares_its_nodes_mapping_reaches_the_node(spelling):
    """A node that holds another and shares its mapping by identity: an
    assignment on it gives the same new mapping to the node it holds (and
    to whatever that one wraps), so the two go on sharing."""
    write, key, value = SPELLINGS[spelling]
    inner = _Decay()
    gm = GraphManager()
    gm.add_node(_SharingWrapper(HybridNode(inner, _no_correction)))
    gm.compile()
    write(gm, inner)
    node = gm.get_node("n")
    assert node.params is node.wrapped.params is inner.params and inner.params[key] == value
    assert float(gm.params["nodes"]["n"]["rate"]) == NEW
    got = _run(gm, _global_x, STEPS_AFTER)
    want = np.arange(1, N + 1, dtype=np.float32) * np.float32(1.0 - DT * NEW) ** STEPS_AFTER
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
