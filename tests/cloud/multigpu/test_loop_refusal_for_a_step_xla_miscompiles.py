"""A loop over a sharded step XLA miscompiles is refused (MADD-ANO-068).

On jaxlib 0.10.2, 0.11.0 and 0.11.2, a ``ShardedStencilNode`` step traced
inside a loop is compiled wrongly when the node reads a sharded
``StaticArray`` in its halo, reads another array through a window at a
``shard_info`` offset, and the static is replicated over a mesh axis of
two or more devices: silently on a mesh axis the ``axis_map`` leaves
unused, a compile failure on a pencil.  ``GraphManager`` refuses every
entry point that loops over such a step, and keeps ``step``, ``run`` and
``run_adaptive`` -- except for a node inside a coupling group, whose
iteration loops over its update within the step itself (the ``fori``
solver's ``step()`` was measured 3.4e-3 off before the refusal).

What the refusal must not catch is pinned beside what it must: a static
read only in its interior, no window, a window into the static itself
(the multi-device validation goals' 2-D field does that, inside
``run_scan`` and a coupling group), a static the step never reads
(``HeatNode``'s ``grid_x``), a node with no sharded static (``LBMNode``),
and a mesh with no axis to replicate over.  Each of those runs and
answers as the unsharded graph.  The analysis that decides it
(``maddening.cloud.multigpu._scan_hazards``) is tested on jaxprs at the
end, primitive by primitive.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from maddening.cloud.multigpu import _scan_hazards as H
from maddening.cloud.multigpu import sharded_node as sharded_node_module
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.static_data import StaticArray
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 8,
    reason="needs 8 devices (the directory conftest forces 16 virtual CPU devices)")

#: The configuration the generated harness found: ``kappa`` read in its
#: halo, the table through a halo-wide window at ``shard_info[0]``, a rod
#: on a (2, 2) mesh sharded over ``px``, so ``kappa`` is copied along ``py``.
_ROD = D.StencilConfig(
    mesh_shape=(2, 2), axis_names=("px", "py"), axis_map=(("px", 0),), shape=(4,),
    halo=(1,), fill="edge", declares=False, contract="params", integral=None,
    integral_name="a_total", integral_listed=True, reads_shard_info=True, kappa="halo",
    kappa_axis=0, table="halo", source="none", misshapen_shape=(), gain=False,
    faces=False, dtype="float32", wrapping="single", steps=1, seed=0, surface="run_scan")

_REFUSED = r"refused \(MADD-ANO-068\).*'kappa'.*mesh axis 'py'"


def _graph(cfg=_ROD, sharded=True):
    case = D.build_case(cfg)
    gm, _ = D.build_graph(case, sharded)
    return case, gm


def _f(gm, name=D.NODE_NAME):
    return np.asarray(jax.device_get(gm.get_node_state(name)["f"]))


# ---------------------------------------------------------------------------
# Where it is refused, and where it is not
# ---------------------------------------------------------------------------


def _loop_entry(entry, gm, case):
    f0 = jnp.asarray(_f(gm))
    if entry == "run_scan":
        gm.run_scan(2)
    elif entry == "run_scan_with_history":
        gm.run_scan_with_history(2)
    elif entry == "run_sweep":
        gm.run_sweep(2, {case.name: {"f": jnp.stack([f0, f0])}})
    elif entry == "run_adaptive_scan":
        gm.run_adaptive_scan(0.04, max_steps=3, dt_initial=0.02)
    elif entry == "sysid.windowed_loss":
        from maddening.sysid import windowed_loss
        windowed_loss(gm, gm.params, {case.name: {"f": jnp.stack([f0, f0, f0])}},
                      obs_fn=lambda s: s[case.name]["f"], window=2)
    else:  # a gradient through run_scan
        jax.grad(lambda r: jnp.sum(gm.run_scan(
            2, params={"nodes": {case.name: {"rate": r}}})[case.name]["f"]))(0.5)


@pytest.mark.parametrize("entry", ["run_scan", "run_scan_with_history", "run_sweep",
                                   "run_adaptive_scan", "sysid.windowed_loss", "gradient"])
def test_every_entry_point_that_loops_over_the_step_refuses(entry):
    case, gm = _graph()
    before = _f(gm)
    name = "run_scan" if entry == "gradient" else entry
    with pytest.raises(RuntimeError, match=rf"^{name}: {_REFUSED}"):
        _loop_entry(entry, gm, case)
    np.testing.assert_array_equal(_f(gm), before)


@pytest.mark.parametrize("entry", ["step", "run", "run_adaptive"])
def test_the_entry_points_that_do_not_loop_run_and_answer_as_unsharded(entry):
    results = []
    for sharded in (True, False):
        _, gm = _graph(sharded=sharded)
        if entry == "step":
            gm.step()
            gm.step()
        elif entry == "run":
            gm.run(2)
        else:
            gm.run_adaptive(0.04, dt_initial=0.02, dt_max=0.02)
        results.append(_f(gm))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-6)


class _Far(SimulationNode):
    """A replicated scalar relaxing to the field's mean, driving its gain."""

    def __init__(self, timestep=0.02):
        super().__init__(name="far", timestep=timestep)

    def state_fields(self):
        return ["u"]

    def initial_state(self):
        return {"u": jnp.asarray(0.7, jnp.float32)}

    def boundary_input_spec(self):
        return {"m": BoundaryInputSpec(shape=(), description="the field's mean")}

    def update(self, state, boundary_inputs, dt):
        m = jnp.asarray(boundary_inputs.get("m", 0.0), jnp.float32)
        return {"u": state["u"] + dt * 2.0 * (m - state["u"])}


def _coupled(cfg, solver, sharded=True):
    cfg = replace(cfg, source="scalar", gain=True)
    case = D.build_case(cfg)
    gm = GraphManager()
    gm.add_node(case.make(sharded))
    gm.add_node(_Far(timestep=0.04 if solver == "subcycling" else 0.02))
    gm.add_external_input(target_node=case.name, target_field="source", shape=(),
                          dtype=np.float32)
    gm.add_edge(case.name, "far", "f", "m", transform=jnp.mean)
    gm.add_edge("far", case.name, "u", "gain")
    kwargs = {"subcycling": True} if solver == "subcycling" else (
        {"solver": solver} if solver else {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gm.add_coupling_group([case.name, "far"], max_iterations=20, tolerance=1e-7,
                              **kwargs)
    gm.compile()
    return case, gm


@pytest.mark.parametrize("solver", [None, "fori", "subcycling"], ids=["ift", "fori", "subcycling"])
def test_a_member_of_a_coupling_group_is_refused_on_step_too(solver):
    """The group's iteration -- a while_loop, a fori_loop, a sub-cycling scan --
    loops over the member's update inside the step: no entry point is safe."""
    _, gm = _coupled(_ROD, solver)
    for call in (gm.step, lambda: gm.run(1), lambda: gm.run_scan(1)):
        with pytest.raises(RuntimeError, match=r"refused \(MADD-ANO-068\).*coupling "
                                               r"group.*'kappa'.*mesh axis 'py'"):
            call()


def test_a_coupling_group_member_without_the_pattern_steps_as_unsharded():
    cfg = replace(_ROD, kappa="interior")
    results = []
    for sharded in (True, False):
        _, gm = _coupled(cfg, "fori", sharded)
        gm.step()
        gm.run_scan(1)
        results.append(_f(gm))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("cfg", [
    replace(_ROD, kappa="interior"),                      # the static read in its interior only
    replace(_ROD, table=None),                            # no window
    replace(_ROD, reads_shard_info=False, table=None),    # no shard_info, so no offset
    replace(_ROD, mesh_shape=(2, 1)),                     # nothing to replicate over
    replace(_ROD, axis_map=(("px", 0), ("py", 1)), shape=(4, 4), halo=(1, 1),
            kappa="interior"),                            # a pencil, interior read
], ids=["interior-static", "no-window", "no-shard-info", "no-replicated-axis",
        "pencil-interior"])
def test_a_neighbouring_configuration_is_not_refused_and_scans_as_unsharded(cfg):
    assert not D.loop_refusal_expected(cfg)
    D.check_config(cfg, surfaces=("run_scan",))


@pytest.mark.parametrize("cfg, axes", [
    (replace(_ROD, axis_map=(("py", 0),)), ("px",)),
    (replace(_ROD, axis_map=(("px", 0), ("py", 1)), shape=(4, 4), halo=(1, 1),
             kappa_axis=1), ("px",)),
    (replace(_ROD, mesh_shape=(2, 2, 2), axis_names=("px", "py", "pz")), ("py", "pz")),
    (replace(_ROD, mesh_shape=(2, 2), axis_map=(("px", 0), ("py", 1)), shape=(4, 4),
             halo=(1, 1), table=None, faces=True), ("py",)),
    (replace(_ROD, table="interior"), ("py",)),
    (replace(_ROD, wrapping="nested"), ("py",)),
    (replace(_ROD, wrapping="hybrid"), ("py",)),
], ids=["used-py", "pencil-static-on-axis-1", "third-axis", "face-window",
        "block-window", "nested", "hybrid"])
def test_the_refusal_names_every_axis_the_static_is_copied_along_and_no_other(cfg, axes):
    """A refusal on the wrong axis would send the user to drop the wrong one."""
    assert D.loop_refusal_expected(cfg)
    case, gm = _graph(cfg)
    with pytest.raises(RuntimeError, match="MADD-ANO-068") as err:
        gm.run_scan(1)
    msg = str(err.value)
    listed = msg.split("replicated over mesh axis ", 1)[1].split(".  ", 1)[0]
    assert sorted(a for a in cfg.axis_names if f"'{a}'" in listed) == sorted(axes), msg


class _SelfWindow(SimulationNode):
    """A 2-D field reading its sharded static in the halo along axis 0 and
    taking its own columns out of the static at ``shard_info[1]`` -- the
    pattern of the multi-device validation goals' field -- on a pencil whose
    static is copied along axis 1's mesh axis."""

    def __init__(self):
        super().__init__(name="win", timestep=0.02)
        rng = np.random.default_rng(3)
        self._f0 = rng.standard_normal((4, 4)).astype(np.float32)
        self._k = jnp.asarray((0.5 + rng.random((4, 4))).astype(np.float32))
        self._static = {"k": StaticArray(self._k, replication="shard", shard_axis=0)}

    @property
    def static_data(self):
        return self._static

    def halo_width(self):
        return {0: 1, 1: 1}

    def state_fields(self):
        return ["f"]

    def initial_state(self):
        return {"f": jnp.asarray(self._f0)}

    @staticmethod
    def _new(f_pad, k_pad, dt):
        f = f_pad[1:-1, 1:-1]
        coef = (k_pad[:-2, 1:-1] + 2 * k_pad[1:-1, 1:-1] + k_pad[2:, 1:-1]) / 4
        lap = f_pad[:-2, 1:-1] + f_pad[2:, 1:-1] + f_pad[1:-1, :-2] + f_pad[1:-1, 2:] - 4 * f
        return f + dt * coef * lap

    def update(self, state, boundary_inputs, dt):
        f_pad = jnp.pad(state["f"], 1, mode="edge")
        return {"f": self._new(f_pad, jnp.pad(self._k, 1, mode="edge"), dt)}

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None):
        f_pad = state_padded["f"]
        k_pad = jnp.pad(static_padded["k"], ((0, 0), (1, 1)), mode="edge")
        k_pad = lax.dynamic_slice_in_dim(k_pad, shard_info[1][0], f_pad.shape[1], axis=1)
        return {"f": f_pad.at[1:-1, 1:-1].set(self._new(f_pad, k_pad, dt))}


def _built_in(kind):
    """``(node, wrap)`` on a (2, 2) mesh whose ``py`` replicates the node."""
    mesh = create_device_mesh(shape=(2, 2), axis_names=("px", "py"))
    if kind == "self-window":
        return _SelfWindow(), lambda n: ShardedStencilNode(n, mesh, {"px": 0, "py": 1})
    if kind == "heat":
        from maddening.nodes.heat import HeatNode
        return (HeatNode("rod", 0.01, n_cells=8, thermal_diffusivity=0.01,
                         initial_temperature=300.0),
                lambda n: ShardedStencilNode(n, mesh, {"px": 0}))
    from maddening.nodes.lbm import LBMNode
    return (LBMNode("lbm", 1.0, grid_shape=(8, 6), viscosity=0.1, lattice="D2Q9"),
            lambda n: ShardedStencilNode(n, mesh, {"px": 0}, boundary="periodic"))


@pytest.mark.parametrize("kind, replicated", [
    ("self-window", {"k": (("py", 2),)}),
    ("heat", {"grid_x": (("py", 2),)}),
    ("lbm", {}),
])
def test_a_built_in_or_self_windowing_node_on_a_replicating_mesh_scans_as_unsharded(
        kind, replicated):
    """Not refused: a window into the static itself; a sharded static the step
    never reads (``HeatNode``'s ``grid_x``); no sharded static (``LBMNode``).
    The first two declare a static copied along ``py``, so the analysis, not
    the declaration, is what lets them through."""
    results = []
    for sharded in (True, False):
        node, wrap = _built_in(kind)
        if sharded:
            wrapper = wrap(node)
            assert wrapper._statics_replicated_over_devices() == replicated
        gm = GraphManager()
        gm.add_node(wrapper if sharded else node)
        gm.compile()
        if kind == "heat":
            gm.set_node_state("rod", {"temperature": jnp.linspace(300.0, 340.0, 8)})
        gm.run_scan(3)
        results.append(jax.tree.map(lambda a: np.asarray(jax.device_get(a)),
                                    gm.get_node_state(node.name)))
    for key in results[1]:
        np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-6, atol=1e-6)


def test_the_verdict_follows_a_recompile_both_ways():
    """Kept per compile generation: replacing a node re-asks."""
    safe = D.build_case(replace(_ROD, kappa="interior"))
    risky = D.build_case(_ROD)
    gm, _ = D.build_graph(safe, True)
    gm.run_scan(1)
    gm.remove_node(D.NODE_NAME)
    gm.add_node(risky.make(True))
    with pytest.raises(RuntimeError, match=_REFUSED):
        gm.run_scan(1)
    gm.remove_node(D.NODE_NAME)
    gm.add_node(safe.make(True))
    gm.run_scan(1)


def test_a_sharded_static_that_appears_after_construction_is_seen():
    """The declaration half is read from the node's current ``static_data``,
    not from what the wrapper classified when it was built: a node that
    exposes its sharded static only later (a provider filling it in) is
    asked about it on the next loop entry point."""
    base = D.stencil_node_class(_ROD.contract, _ROD.reads_shard_info, _ROD.declares)

    class Late(base):
        exposed = False

        @property
        def static_data(self):
            return self._static if self.exposed else {
                k: v for k, v in self._static.items() if k != "kappa"}

    node = Late(_ROD)
    mesh = create_device_mesh(shape=(2, 2), axis_names=("px", "py"))
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(node, mesh, {"px": 0}))
    gm.compile()
    gm.run_scan(1)
    node.exposed = True
    with pytest.raises(RuntimeError, match=_REFUSED):
        gm.run_scan(1)


def test_a_wrapper_the_probe_did_not_reach_is_refused(monkeypatch):
    """Fail closed: no answer from the wrapper is not a clean answer."""
    monkeypatch.setattr(sharded_node_module, "active_probe", lambda: None)
    _, gm = _graph(replace(_ROD, kappa="interior"))
    with pytest.raises(RuntimeError, match=r"MADD-ANO-068.*could not be analysed"):
        gm.run_scan(1)


# ---------------------------------------------------------------------------
# The analysis, primitive by primitive
# ---------------------------------------------------------------------------

_TABLE = jnp.arange(10.0)


def _analyse(fn, n_static=6, n_off=1, halo=1, axis=0, static_shape=None, axis_env=None):
    """``fn(static, offset)`` analysed with one static padded by ``halo``."""
    shape = static_shape or (n_static,)
    closed = jax.make_jaxpr(fn, axis_env=axis_env)(
        [jnp.ones(shape)], [jnp.int32(0)] * n_off)
    return H.analyse_update_padded(closed, [("k", axis, halo, shape[axis])])["k"]


@pytest.mark.parametrize("fn, halo_read", [
    (lambda s, o: s[0][1:5], False),                                  # interior slice
    (lambda s, o: lax.slice_in_dim(s[0], 1, 5), False),
    (lambda s, o: lax.dynamic_slice_in_dim(s[0], 1, 4), False),       # literal start
    (lambda s, o: s[0][0:4], True),                                   # into the low halo
    (lambda s, o: s[0][2:6], True),                                   # into the high halo
    (lambda s, o: s[0] * 2.0, True),                                  # the whole array
    (lambda s, o: jnp.pad(s[0], 1), True),                            # into a jit-ed helper
    (lambda s, o: o[0] * 2, False),                                   # never read
    (lambda s, o: (s[0] * 2.0, s[0][1:5])[1], False),                 # read, but dead
    (lambda s, o: lax.dynamic_slice_in_dim(s[0], o[0], 4), True),     # at a traced start
], ids=["slice", "slice_in_dim", "dynamic-literal", "low-halo", "high-halo", "whole",
        "jit-helper", "unread", "dead", "dynamic-traced"])
def test_a_halo_read_is_any_live_use_but_an_interior_slice(fn, halo_read):
    assert _analyse(fn)[0] is halo_read


def test_an_interior_slice_on_the_other_axis_of_a_2d_static_is_still_judged_on_the_shard_axis():
    got = _analyse(lambda s, o: s[0][0:2, 1:3], static_shape=(4, 6), axis=1)
    assert got[0] is False
    got = _analyse(lambda s, o: s[0][1:3, 0:2], static_shape=(4, 6), axis=1)
    assert got[0] is True


@pytest.mark.parametrize("fn, counted", [
    (lambda s, o: lax.dynamic_slice_in_dim(_TABLE, o[0], 4), True),
    (lambda s, o: lax.dynamic_slice_in_dim(jnp.pad(_TABLE, 1), o[0], 6), True),
    (lambda s, o: jnp.take(_TABLE, o[0] + jnp.arange(3)), True),                # gather
    (lambda s, o: lax.dynamic_update_slice_in_dim(_TABLE, s[0][1:3], o[0], 0), True),
    (lambda s, o: _TABLE.at[o[0]].set(1.0), True),                              # scatter
    (lambda s, o: lax.dynamic_slice_in_dim(_TABLE, o[0] + 1, 4), True),         # derived index
    (lambda s, o: lax.dynamic_slice_in_dim(s[0] * _TABLE[:6], o[0], 4), True),  # mixed operand
    (lambda s, o: lax.dynamic_slice_in_dim(jnp.pad(s[0], 1), o[0], 4), False),  # itself
    (lambda s, o: lax.dynamic_slice_in_dim(_TABLE, 2, 4), False),               # literal index
    (lambda s, o: jnp.where(o[0] == 0, _TABLE, 0.0), False),                    # no index
    (lambda s, o: (lax.dynamic_slice_in_dim(_TABLE, o[0], 4), _TABLE)[1], False),  # dead
], ids=["table", "padded-table", "gather", "update-slice", "scatter", "derived-index",
        "mixed-operand", "own-static", "literal-index", "comparison", "dead"])
def test_a_window_is_a_live_offset_index_into_an_array_other_than_the_static(fn, counted):
    assert bool(_analyse(fn)[1]) is counted


def test_an_index_from_axis_index_counts_as_an_offset():
    got = _analyse(lambda s, o: lax.dynamic_slice_in_dim(_TABLE, lax.axis_index("i"), 4),
                   axis_env=[("i", 2)])
    assert got[1]


def test_a_window_inside_a_jit_ed_helper_is_followed_argument_by_argument():
    @jax.jit
    def helper(table, start, static):
        return lax.dynamic_slice_in_dim(table, start, 4), lax.dynamic_slice_in_dim(
            static, start, 4)

    own = _analyse(lambda s, o: helper(jnp.pad(s[0], 2), o[0], s[0])[1])
    assert own[1] == []
    other = _analyse(lambda s, o: helper(_TABLE, o[0], s[0])[0])
    assert other[1]


def test_the_message_names_the_node_the_static_and_the_window():
    hazard = H.ScanHazard(node="n", node_type="T", static="k", shard_axis=0,
                          replicated_over=(("py", 2), ("pz", 4)), window="dynamic_slice")
    text = hazard.describe()
    for part in ("T 'n'", "'k'", "'py' (2 devices)", "'pz' (4 devices)", "dynamic_slice"):
        assert part in text
