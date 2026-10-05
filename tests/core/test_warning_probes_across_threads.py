"""MADDENING's warning probes leave every other thread its warnings.

The library runs a node's code a second time, with warnings silenced, in
sixteen places: when a live parameter is written (which leaves does the
compiled step read?), when ``PUT /graph/params`` checks a value (does the
constructor take it?  does the state keep its layout?), when a sharded
wrapper checks its inner node's state, when the profiler compiles its
one-iteration variant.  Each used ``warnings.catch_warnings()`` and
``simplefilter("ignore")``, which act on the whole process
(MADD-ANO-197):

* while one thread was inside a probe, **every** thread's warnings were
  dropped;
* two probes that overlapped in two threads and left in the order they
  entered each put back the other's saved filters, and the process kept an
  ``ignore`` for every warning for good.  Four independent graphs, each
  written and stepped from its own thread, did that in 1 to 9 of 40
  rounds.

Two kinds of test here, both on real graphs and neither naming the helper
that fixed it, so both fail on a tree that still uses ``catch_warnings``
for what that does:

* the reproducer, made certain: the node's own code -- traced inside the
  probe -- holds the two probes open together and lets the first one in
  leave first;
* a battery over **every** silencing block of ``src/``, found by parsing
  the source (``tests/_quiet_block_witness.py``).  A node whose
  constructor, ``initial_state()`` and ``update()`` report from inside
  each block is driven through the public path that reaches it; inside
  every one, another thread's warning is delivered and the node's own is
  silenced.  A block added to ``src/`` later fails the battery until a
  path here reaches it.

The helper's own promises are in ``test_quiet_warnings.py``.
"""

from __future__ import annotations

import os
import threading
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from tests import _quiet_block_witness as W
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.profiler import profile_graph
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

#: How long a thread waits for the other before the test gives up.
WAIT = 30  # units: s
DT = 0.01  # units: s


def _filters() -> list:
    """The process's filters, with a compiled pattern shown as its text."""
    return [(action, getattr(msg, "pattern", msg), category, getattr(mod, "pattern", mod),
             lineno) for action, msg, category, mod, lineno in warnings.filters]


def _undamped_pair() -> GraphManager:
    """Two springs anchored on each other in a coupling group, undamped at
    ``k dt^2 / m = 0.5``: ``compile()`` warns that the pair grows every
    step (MADD-ANO-098), a warning this suite does not filter."""
    gm = GraphManager()
    for name, rest, start in (("a", 1.0, 0.0), ("b", -1.0, 5.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=0.005, stiffness=10000.0,
                                     damping=0.0, mass=0.5, rest_length=rest,
                                     initial_position=start))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6)
    return gm


def _a_maddening_warning_is_still_delivered() -> None:
    """``compile()``'s own advisory reaches the caller -- as the error this
    block makes of it, at the *back* of the filters, where an ``ignore``
    left ahead of it would win."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, append=True)
        with pytest.raises(UserWarning, match="MADD-ANO-098"):
            _undamped_pair().compile()


# ---------------------------------------------------------------------------
# The reproducer: independent graphs, written and stepped in a thread each
# ---------------------------------------------------------------------------


class _Rendezvous:
    """Holds two probes open together, and lets the first one in leave
    first -- the order in which two ``catch_warnings`` blocks put back
    each other's filters.  Called from the node's ``update`` while a probe
    traces it."""

    def __init__(self):
        self.armed = False
        self.lock = threading.Lock()
        self.inside = 0
        self.first_in: int | None = None
        self.both_inside = threading.Event()
        self.first_has_left = threading.Event()
        self.overlapped = False

    def meet(self) -> None:
        if not self.armed:
            return
        with self.lock:
            self.inside += 1
            mine = self.inside
            if mine == 1:
                self.first_in = threading.get_ident()
        if mine == 1:
            # First in: wait for the other probe to open, then leave.
            self.overlapped = self.both_inside.wait(WAIT)
        elif mine == 2:
            # Second in: stay inside until the first thread's write and
            # step have returned, its probe long closed.
            self.both_inside.set()
            assert self.first_has_left.wait(WAIT), "the first probe never closed"

    def left(self) -> None:
        """A thread's write and step have returned."""
        if self.first_in == threading.get_ident():
            self.first_has_left.set()


class _MeetingSpring(SpringDamperNode):
    """A spring whose ``update`` keeps the rendezvous when it is traced."""

    rendezvous: _Rendezvous

    def update(self, state, boundary_inputs, dt, *, params=None):
        self.rendezvous.meet()
        return super().update(state, boundary_inputs, dt, params=params)


def _stepped_graph(i: int, rendezvous: _Rendezvous) -> GraphManager:
    """A compiled graph of one spring away from its rest length, stepped
    once (so the next step compiles nothing)."""
    node = _MeetingSpring(f"s{i}", timestep=DT, rest_length=1.0, initial_position=3.0)
    node.rendezvous = rendezvous
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    gm.step()
    return gm


def _write_and_step(gm: GraphManager, i: int) -> None:
    gm.params["nodes"][f"s{i}"]["stiffness"] = jnp.float32(2.5 + i)
    gm.step()


def test_independent_graphs_written_and_stepped_in_two_threads_leave_the_filters_alone():
    """Two graphs that share nothing.  Each thread writes a live parameter
    of its own graph and steps it, so each ``step()`` asks which leaves
    its compiled step reads -- one trace of the step, warnings silenced.
    The two traces are open at once and the first one in is the first one
    out.  Afterwards the process's filters are the ones before, and
    ``compile()``'s own warning is still delivered."""
    rendezvous = _Rendezvous()
    graphs = [_stepped_graph(i, rendezvous) for i in range(2)]
    before = _filters()
    errors: list = []

    def work(i):
        try:
            _write_and_step(graphs[i], i)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            rendezvous.left()

    rendezvous.armed = True
    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(3 * WAIT)
    finally:
        rendezvous.armed = False
        rendezvous.first_has_left.set()
        rendezvous.both_inside.set()
    assert not any(t.is_alive() for t in threads), "a step never returned"
    assert not errors, [f"{type(e).__name__}: {e}" for e in errors]
    assert rendezvous.inside == 2, (
        f"the step was traced {rendezvous.inside} time(s) for two writes: a write "
        "no longer asks, once, which leaves the step reads, so this test is not "
        "testing that probe")
    assert rendezvous.overlapped, (
        "the second probe did not open while the first was open: the probes "
        "are serialised across independent graphs, so one graph's node code "
        "can hold up every other graph's parameter writes")
    left = [f for f in _filters() if f not in before]
    assert _filters() == before, f"left in the process's filters: {left}"
    _a_maddening_warning_is_still_delivered()


def test_the_written_values_are_the_ones_the_steps_ran_with():
    """The fixture can express the defect and nothing else moved: unarmed,
    the same two writes are taken, not refused, and each graph steps with
    its own new stiffness."""
    rendezvous = _Rendezvous()
    graphs = [_stepped_graph(i, rendezvous) for i in range(2)]
    for i, gm in enumerate(graphs):
        _write_and_step(gm, i)
        assert float(gm.params["nodes"][f"s{i}"]["stiffness"]) == 2.5 + i
    a, b = (np.asarray(gm.get_node_state(f"s{i}")["velocity"])
            for i, gm in enumerate(graphs))
    assert rendezvous.inside == 0
    assert not np.array_equal(a, b), "the two graphs stepped with the same stiffness"


# ---------------------------------------------------------------------------
# The battery: every silencing block of src/, from inside
# ---------------------------------------------------------------------------


class ProbedRod(HeatNode):
    """A ``HeatNode`` whose code reports from inside a silencing block."""

    def __init__(self, name, timestep, **params):
        W.witness("__init__")
        super().__init__(name, timestep, **params)

    def initial_state(self):
        W.witness("initial_state")
        return super().initial_state()

    def update(self, state, boundary_inputs, dt, *, params=None):
        W.witness("update")
        return super().update(state, boundary_inputs, dt, params=params)


class ProbedSpring(SpringDamperNode):
    """A spring that reports from the hooks ``compile()`` calls."""

    def boundary_input_spec(self):
        W.witness("boundary_input_spec")
        return super().boundary_input_spec()

    def params_pytree(self):
        W.witness("params_pytree")
        return super().params_pytree()


class ProbedCells(SimulationNode):
    """Eight cells without a stencil, for the unstructured wrapper."""

    def __init__(self, name="cells", timestep=DT):
        super().__init__(name, timestep, k=2.0)

    def initial_state(self):
        W.witness("initial_state")
        return {"x": jnp.ones(8, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return dict(state)

    def update_padded(self, state_padded, boundary_inputs, dt, *, params=None):
        return dict(state_padded)


REGISTRY = {"ProbedRod": ProbedRod}


def _rod(name="rod", **overrides) -> ProbedRod:
    params = dict(n_cells=6, length=1.0, thermal_diffusivity=0.005, initial_temperature=1.0)
    params.update(overrides)
    return ProbedRod(name, DT, **params)


def _rod_graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(_rod())
    gm.compile()
    gm.step()
    return gm


def _put(gm: GraphManager, node: str, params: dict) -> int:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    return client.put(f"/graph/params/{node}", json={"params": params}).status_code


def _a_live_leaf_the_step_reads():
    gm = _rod_graph()
    gm.params["nodes"]["rod"]["thermal_diffusivity"] = jnp.float32(0.004)
    gm.step()


def _a_live_leaf_only_the_initial_state_reads():
    gm = _rod_graph()
    gm.params["nodes"]["rod"]["initial_temperature"] = jnp.float32(2.0)
    with pytest.raises(ValueError, match="differs from the node's own value"):
        gm.step()


def _put_a_leaf_the_step_reads():
    assert _put(_rod_graph(), "rod", {"thermal_diffusivity": 0.004}) == 200


def _put_an_initial_condition():
    assert _put(_rod_graph(), "rod", {"initial_temperature": 2.0}) == 200


def _put_a_structural_value():
    assert _put(_rod_graph(), "rod", {"stencil_order": 4}) == 200


def _put_a_value_the_constructor_derives_from():
    assert _put(_rod_graph(), "rod", {"length": 1.5}) == 200


def _a_parameter_under_a_mapping_point_reference():
    gm = GraphManager()
    a, b = _rod("a"), _rod("b", n_cells=5, initial_temperature=0.5)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=rbf_mapping(
        np.asarray(a.static_data["grid_x"].value), np.asarray(b.static_data["grid_x"].value),
        source_ref={"node": "a", "field": "grid_x"},
        target_ref={"node": "b", "field": "grid_x"}))
    gm.compile()
    gm.step()
    gm.params["nodes"]["a"]["length"] = jnp.float32(1.5)
    with pytest.raises(ValueError, match="interface mapping"):
        gm.step()


def _the_loop_hazard_probe():
    # The probe runs when a sharded wrapper declares a static copied along
    # a mesh axis, which takes two devices; what it does -- one trace of
    # the step, silenced -- it does for any graph.
    assert _rod_graph()._probe_xla_loop_hazards([]) == []


def _a_sharded_stencil_step():
    gm = GraphManager()
    mesh = create_device_mesh(shape=(1,))
    gm.add_node(ShardedStencilNode(_rod(n_cells=8), mesh, {"devices": 0}))
    gm.compile()
    gm.step()


def _a_sharded_unstructured_wrapper():
    n = 8
    layout = build_unstructured_partition(
        partition_assignment=np.zeros(n, np.int32),
        edges=np.array([[i, (i + 1) % n] for i in range(n)], np.int32),
        n_devices=1)
    ShardedUnstructuredNode(ProbedCells(), create_device_mesh(shape=(1,)), layout)


def _the_profilers_one_iteration_variant():
    gm = GraphManager()
    for name, rest, start in (("a", 1.0, 0.0), ("b", -1.0, 5.0)):
        gm.add_node(ProbedSpring(name=name, timestep=0.005, stiffness=500.0, damping=5.0,
                                 mass=0.5, rest_length=rest, initial_position=start))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6)
    gm.compile()
    gm.step()
    report = profile_graph(gm, n_steps=1, n_warmup=1, counts=False)
    assert report.one_iteration_step_ms is not None


#: Each public path, and the silencing blocks it runs the node's code in.
PATHS = [
    (_a_live_leaf_the_step_reads, {
        "core/graph_manager.py::GraphManager._params_read_by_step#1"}),
    (_a_live_leaf_only_the_initial_state_reads, {
        "core/graph_manager.py::GraphManager._params_read_by_step#1",
        "core/graph_manager.py::GraphManager._node_param_reads#1"}),
    (_put_a_leaf_the_step_reads, {
        "api/server.py::_abstract_initial_state#1",
        "api/server.py::_saved_graph_write_reason.build#1",
        "core/graph_manager.py::GraphManager._params_read_by_step#1",
        "core/graph_manager.py::GraphManager._state_shape_write_reason#1"}),
    (_put_an_initial_condition, {
        "core/graph_manager.py::GraphManager._node_param_reads#1",
        "core/graph_manager.py::GraphManager._initial_state_depends#1"}),
    (_put_a_structural_value, {
        "api/server.py::_trace_hooks#1",
        "core/graph_manager.py::GraphManager._constructor_write_reason.build#1",
        "core/graph_manager.py::GraphManager._hooks_trace_depends#1"}),
    (_put_a_value_the_constructor_derives_from, {
        "api/server.py::_what_the_node_computes#1"}),
    (_a_parameter_under_a_mapping_point_reference, {
        "core/graph_manager.py::GraphManager._mapped_points_moved_by.build#1"}),
    (_the_loop_hazard_probe, {
        "core/graph_manager.py::GraphManager._probe_xla_loop_hazards#1"}),
    (_a_sharded_stencil_step, {
        "cloud/multigpu/sharded_node.py::ShardedStencilNode._check_state_matches_inner#1"}),
    (_a_sharded_unstructured_wrapper, {
        "cloud/multigpu/sharded_unstructured.py::"
        "ShardedUnstructuredNode._check_inner_cell_count#1"}),
    (_the_profilers_one_iteration_variant, {
        "core/simulation/profiler.py::_one_iteration_variant._cm#1",
        "core/simulation/profiler.py::_one_iteration_variant._cm#2"}),
]

BLOCKS = [block.name for block in W.silencing_blocks()]


@pytest.fixture(scope="module")
def sightings() -> dict:
    """Every path driven once, the witness armed: ``{block: [Sighting]}``,
    and which blocks each path reached."""
    seen: dict = {}
    reached: dict = {}
    for path, _ in PATHS:
        with W.watching() as watch:
            path()
        reached[path.__name__] = watch.visited()
        for block, found in watch.seen.items():
            seen.setdefault(block, []).extend(found)
    return {"seen": seen, "reached": reached}


def test_there_are_silencing_blocks_to_test():
    """The battery below is parametrised over what the scan finds; finding
    nothing would pass it."""
    assert len(BLOCKS) >= 16, BLOCKS


def test_every_silencing_block_of_src_is_reached_by_a_path_here(sightings):
    """A block nobody drives is a block nobody has looked inside.  A new
    one fails here until a path above reaches it."""
    assert set(sightings["seen"]) == set(BLOCKS), {
        "not reached": sorted(set(BLOCKS) - set(sightings["seen"])),
        "not a block of src/": sorted(set(sightings["seen"]) - set(BLOCKS))}


@pytest.mark.parametrize("path, blocks", PATHS, ids=[p.__name__.lstrip("_") for p, _ in PATHS])
def test_a_path_reaches_the_blocks_it_is_here_for(sightings, path, blocks):
    """Each path still runs the node's code inside the blocks it was
    written to reach (it may reach others too)."""
    assert blocks <= sightings["reached"][path.__name__], (
        sorted(blocks - sightings["reached"][path.__name__]))


@pytest.mark.parametrize("block", BLOCKS)
def test_another_thread_is_told_what_it_warns_while_the_block_is_open(sightings, block):
    """From the node's code inside the block, a warning raised in another
    thread is delivered to that thread.  ``catch_warnings`` with
    ``simplefilter("ignore")`` dropped it: for as long as any thread
    probed, no thread of the process was warned of anything."""
    found = sightings["seen"].get(block, [])
    assert found, f"no path reached {block}"
    deaf = [s.where for s in found if not s.another_thread_heard]
    assert not deaf, f"another thread's warning was dropped while {deaf} ran inside {block}"


@pytest.mark.parametrize("block", BLOCKS)
def test_the_nodes_own_warning_is_silenced_inside_the_block(sightings, block):
    """What the block is for: the node's code, run a second time to learn
    something about it, does not repeat its warnings -- and, where
    warnings are errors, does not turn the probe's answer into "cannot
    tell"."""
    found = sightings["seen"].get(block, [])
    assert found, f"no path reached {block}"
    loud = [s.where for s in found if not s.silenced_here]
    assert not loud, f"the node's own warning was raised from {loud} inside {block}"


def test_the_paths_leave_the_filters_alone_and_a_maddening_warning_is_still_delivered(
        sightings):
    """After every path has run -- each with the witness's threads warning
    from inside the blocks -- nothing is left in the process's filters,
    and ``compile()``'s own warning is delivered."""
    assert not any(action == "ignore" and category is Warning and msg is None
                   for action, msg, category, _, _ in warnings.filters), warnings.filters
    _a_maddening_warning_is_still_delivered()
