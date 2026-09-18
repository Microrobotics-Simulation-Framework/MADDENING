"""``run_scan`` and its siblings build their program once per compile.

``run_scan``, ``run_scan_with_history``, ``run_sweep`` and
``run_adaptive_scan`` used to rebuild and recompile their ``lax.scan``
on every call, because the scan body closed over the external inputs
and the params by value: a fresh closure meant a fresh jaxpr meant a
fresh XLA compile.  That is not a per-step cost but a per-*call* one,
hundreds of milliseconds upward, which is exactly wrong for a caller
driving a simulation from a Python loop, a slider or an HTTP handler.

The values are now arguments of a jitted program, and the program is
cached (``GraphManager._cached_scan``).  These tests assert the counted
quantity -- ``gm.scan_trace_count``, the number of Python traces of a
scan program, one per XLA compile -- rather than a duration, which on a
shared machine would be noise.

The other half is correctness: a cache that wrongly hits is a bug.  So
the tests also pin what must *not* be cached past -- a changed external
input, a changed parameter, a restored checkpoint, a replaced node, and
any mutation of the graph after its first scan.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode


def _spring_graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=30.0, damping=2.0, initial_position=0.5,
    ))
    gm.compile()
    return gm


def _forced_graph() -> GraphManager:
    """A spring whose anchor is driven from outside, so ``ext`` matters."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=30.0, damping=2.0, initial_position=0.5,
    ))
    gm.add_external_input("s", "anchor_position", shape=())
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# The counted invariant.
# ---------------------------------------------------------------------------


def test_repeated_run_scan_builds_its_program_once():
    gm = _spring_graph()
    gm.run_scan(5)
    assert gm.scan_trace_count == 1
    for _ in range(9):
        gm.run_scan(5)
    assert gm.scan_trace_count == 1, (
        "run_scan rebuilt (and so recompiled) its scan on a repeat call"
    )


def test_each_step_count_gets_its_own_scan_program():
    """The scan length is baked into the program, so it is in the key."""
    gm = _spring_graph()
    gm.run_scan(5)
    gm.run_scan(7)
    assert gm.scan_trace_count == 2
    gm.run_scan(5)
    gm.run_scan(7)
    assert gm.scan_trace_count == 2


def test_each_scan_entry_point_gets_its_own_program():
    gm = _spring_graph()
    gm.run_scan(4)
    gm.run_scan_with_history(4)
    gm.run_sweep(4, {"s": {"position": jnp.array([0.5, 2.0]),
                           "velocity": jnp.zeros(2)}})
    assert gm.scan_trace_count == 3
    gm.run_scan(4)
    gm.run_scan_with_history(4)
    gm.run_sweep(4, {"s": {"position": jnp.array([0.5, 2.0]),
                           "velocity": jnp.zeros(2)}})
    assert gm.scan_trace_count == 3


def test_run_sweep_history_flag_is_part_of_the_key():
    init = {"s": {"position": jnp.array([0.5, 2.0]),
                  "velocity": jnp.zeros(2)}}
    gm = _spring_graph()
    gm.run_sweep(4, init, return_history=False)
    gm.run_sweep(4, init, return_history=True)
    assert gm.scan_trace_count == 2
    finals, hist = gm.run_sweep(4, init, return_history=True)
    assert gm.scan_trace_count == 2
    assert hist["s"]["position"].shape == (2, 4)


def test_trace_count_counts_the_step_and_scan_trace_count_the_scans():
    """The two counters stay distinct -- that is why this went unnoticed."""
    gm = _spring_graph()
    gm.run_scan(3)
    assert (gm.trace_count, gm.scan_trace_count) == (0, 1)
    gm.step()
    assert (gm.trace_count, gm.scan_trace_count) == (1, 1)


# ---------------------------------------------------------------------------
# A cache that wrongly hits is a correctness bug.
# ---------------------------------------------------------------------------


def test_changed_external_inputs_reach_a_cached_scan():
    gm = _forced_graph()
    start = dict(gm.get_node_state("s"))

    quiet_x = float(
        gm.run_scan(10, {"s": {"anchor_position": jnp.array(0.0)}})["s"]["position"]
    )

    gm.set_node_state("s", start)
    pushed_x = float(
        gm.run_scan(10, {"s": {"anchor_position": jnp.array(2.5)}})["s"]["position"]
    )

    assert gm.scan_trace_count == 1, "the second call rebuilt the scan"
    assert not np.isclose(quiet_x, pushed_x), (
        "the cached scan served the first call's external force"
    )


def test_changed_params_reach_a_cached_scan():
    gm = _spring_graph()
    start = dict(gm.get_node_state("s"))

    soft_x = float(gm.run_scan(10)["s"]["position"])

    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, dtype=jnp.float32)
    gm.set_node_state("s", start)
    stiff_x = float(gm.run_scan(10)["s"]["position"])

    assert gm.scan_trace_count == 1, "a params write rebuilt the scan"
    assert not np.isclose(soft_x, stiff_x), (
        "the cached scan served the pre-write stiffness"
    )


def test_a_graph_mutated_after_its_first_scan_rebuilds_it():
    gm = _spring_graph()
    gm.run_scan(5)
    assert gm.scan_trace_count == 1

    gm.add_node(BallNode("b", 0.01, initial_position=5.0))
    out = gm.run_scan(5)

    assert gm.scan_trace_count == 1, (
        "the recompile should have cleared the cache and reset the counter"
    )
    assert "b" in out, "the new node never made it into the scan"


def test_a_checkpoint_restored_between_scans_is_not_cached_past(tmp_path):
    # ``run_scan`` returns a view on the live state, so every comparison
    # snapshots a float straight away rather than holding the dict.
    gm = _spring_graph()
    gm.run_scan(5)
    gm.save_state(tmp_path / "snap.npz")
    expected = float(gm.get_node_state("s")["position"])

    after = float(gm.run_scan(5)["s"]["position"])
    assert not np.isclose(after, expected)

    gm.load_state(tmp_path / "snap.npz")
    assert np.isclose(float(gm.get_node_state("s")["position"]), expected)
    resumed = float(gm.run_scan(5)["s"]["position"])
    assert np.isclose(resumed, after), (
        "the scan did not resume from the restored state"
    )


def test_replace_node_rebuilds_the_scan():
    from maddening.surrogates.replace import replace_node

    gm = _spring_graph()
    gm.run_scan(5)
    gm.run_scan(5)
    before = float(gm.get_node_state("s")["position"])

    replace_node(gm, "s", SpringDamperNode(
        "s", 0.01, stiffness=300.0, damping=2.0, initial_position=0.5,
    ))
    gm.compile()
    after = float(gm.run_scan(5)["s"]["position"])

    assert gm.scan_trace_count == 1     # reset by the recompile
    assert not np.isclose(before, after), (
        "the replacement node's stiffness never reached the scan"
    )


def test_run_scan_still_matches_the_step_loop():
    """Moving ext/params from closed-over constants to arguments is
    numerically inert."""
    a = _forced_graph()
    ext = {"s": {"anchor_position": jnp.array(3.0)}}
    scanned = float(a.run_scan(20, ext)["s"]["position"])

    b = _forced_graph()
    for _ in range(20):
        b.step(ext)
    stepped = float(b.get_node_state("s")["position"])

    np.testing.assert_allclose(scanned, stepped, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Adaptive scan.
# ---------------------------------------------------------------------------


def test_adaptive_scan_is_cached_and_its_tolerances_stay_live():
    gm = _spring_graph()
    loose = float(gm.run_adaptive_scan(
        t_end=0.2, max_steps=32, dt_initial=0.01, atol=1e-2, rtol=1e-1,
    )[0]["s"]["position"])
    assert gm.scan_trace_count == 1

    gm.reset_state()
    out, _, info = gm.run_adaptive_scan(
        t_end=0.2, max_steps=32, dt_initial=0.01, atol=1e-10, rtol=1e-10,
    )
    tight = float(out["s"]["position"])
    assert gm.scan_trace_count == 1, "a tolerance change rebuilt the scan"
    assert int(info["n_steps"]) > 0
    assert not np.isclose(loose, tight), (
        "the cached adaptive scan served the first call's tolerances"
    )


def test_adaptive_scan_max_steps_is_part_of_the_key():
    gm = _spring_graph()
    gm.run_adaptive_scan(t_end=0.1, max_steps=16)
    gm.reset_state()
    _, history, _ = gm.run_adaptive_scan(t_end=0.1, max_steps=24)
    assert gm.scan_trace_count == 2
    assert history["s"]["position"].shape[0] == 24


@pytest.mark.parametrize("n_steps", [3, 8])
def test_run_scan_with_history_shape_and_final_state(n_steps):
    gm = _spring_graph()
    final, history = gm.run_scan_with_history(n_steps)
    assert history["s"]["position"].shape == (n_steps,)
    assert np.isclose(float(history["s"]["position"][-1]),
                      float(final["s"]["position"]))


def test_the_scan_cache_is_bounded():
    """An HTTP handler taking ``n_steps`` from the request cannot grow it
    without limit."""
    from maddening.core.graph_manager import _SCAN_CACHE_MAX

    gm = _spring_graph()
    for n in range(1, _SCAN_CACHE_MAX + 6):
        gm.run_scan(n)
    assert len(gm._scan_cache) <= _SCAN_CACHE_MAX
