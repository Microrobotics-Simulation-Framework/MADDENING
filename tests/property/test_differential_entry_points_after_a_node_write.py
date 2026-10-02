"""Entry-point agreement: after a ``node.params`` write, every way of running
the graph runs the same model.

The oracle: write ``node.params`` after the graph has compiled and traced,
with no compile in between, then measure the map each entry point applies
from one fixed state -- ``gm.step``, ``gm.run``, ``gm.run_scan`` at a length
it traced before the write and at a new one, ``gm.run_scan_with_history``,
and ``sysid.windowed_loss`` -- and require that they all agree, and agree
with the model the write leaves.  Then ``compile()`` and require the same.

It used to fail: the value a node reads from ``self.params`` when its step
is traced was baked into whichever program was traced next, so ``gm.step``
kept the trace it had cached (the old model) while a new scan length or a
sysid loss traced the write in
(audit_040_p4_5/fmu-sysid/repro_node_params_retrace.py; the same since
v0.1.0 for a node on the three-argument contract).

Three kinds of write: every constant of a node on the three-argument
contract, a structural ``int`` of a params-taking node, and a float leaf
``gm.params`` carries.  The cases are a fixed table, so each compiles once;
the comparisons are exact (one program, one arithmetic).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.sysid import observations_from_history, windowed_loss

DT = 0.01
X0 = 1.0
CACHED_LENGTH, NEW_LENGTH = 2, 3


class _Decay3(SimulationNode):
    """Three-argument contract: ``x <- x (1 - rate dt)``, ``rate`` read from
    ``self.params`` when the step is traced."""

    def initial_state(self):
        return {"x": jnp.asarray(X0, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * (1.0 - self.params["rate"] * dt)}


class _Counted(SimulationNode):
    """``x <- x + count * gain * dt``: ``count`` structural, ``gain`` a leaf."""

    def initial_state(self):
        return {"x": jnp.asarray(X0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + int(self.params["count"]) * p["gain"] * dt}


#: (id, node factory, key written, value written, one step of the model as a
#: function of the node's params).
CASES = [
    ("three-argument constant", lambda: _Decay3("n", DT, rate=1.0), "rate", 50.0,
     lambda x, p: x * (1.0 - p["rate"] * DT)),
    ("structural int", lambda: _Counted("n", DT, count=1, gain=1.0), "count", 3,
     lambda x, p: x + p["count"] * p["gain"] * DT),
    ("gm.params leaf", lambda: _Counted("n", DT, count=1, gain=1.0), "gain", 4.0,
     lambda x, p: x + p["count"] * p["gain"] * DT),
]


def _graph(factory):
    gm = GraphManager()
    gm.add_node(factory())
    gm.compile()
    gm.step()                       # a cached trace of the step ...
    gm.run_scan(CACHED_LENGTH)      # ... and of one scan length
    return gm


def _from(gm, x0=X0):
    gm.set_node_state("n", {"x": jnp.asarray(x0, jnp.float32)})


def _x(gm):
    return float(gm.get_node_state("n")["x"])


def _trajectory(model, params, n):
    xs, x = [], np.float32(X0)
    for _ in range(n):
        x = np.float32(model(x, params))
        xs.append(float(x))
    return xs


def _every_entry_point(gm, model, params):
    """What each entry point does from ``X0``, as ``{name: final x}``, and the
    windowed loss against the model's own trajectory (0 iff it runs it).

    The loss is taken *first*, before any run method has had the chance to
    recompile the graph for it: an entry point must notice a write on its
    own."""
    expected = jnp.asarray([X0] + _trajectory(model, params, 4), jnp.float32)
    _from(gm)
    loss = float(windowed_loss(gm, gm.params, {"n": {"x": expected}},
                               obs_fn=lambda s: s["n"]["x"], window=4))
    out = {}
    for name, n, run in (
        ("step", 1, lambda: gm.step()),
        ("run(1)", 1, lambda: gm.run(1)),
        (f"run_scan({CACHED_LENGTH}) cached", CACHED_LENGTH, lambda: gm.run_scan(CACHED_LENGTH)),
        (f"run_scan({NEW_LENGTH}) new", NEW_LENGTH, lambda: gm.run_scan(NEW_LENGTH)),
        (f"run_scan_with_history({NEW_LENGTH})", NEW_LENGTH,
         lambda: gm.run_scan_with_history(NEW_LENGTH)),
    ):
        _from(gm)
        run()
        out[name] = (_x(gm), _trajectory(model, params, n)[-1])
    return out, loss


def _assert_all_run(gm, model, params, when):
    measured, loss = _every_entry_point(gm, model, params)
    for name, (got, want) in measured.items():
        assert got == pytest.approx(want, rel=1e-6, abs=1e-7), (when, name, measured)
    assert loss <= 1e-10, (when, "windowed_loss", loss, measured)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_every_entry_point_runs_the_written_model(case):
    _, factory, key, value, model = case
    gm = _graph(factory)
    node = gm.get_node("n")
    before = dict(node.params)
    _assert_all_run(gm, model, before, "before the write")

    node.params[key] = value
    after = {**before, key: value}
    _assert_all_run(gm, model, after, "after the write, no compile")

    gm.compile()
    _assert_all_run(gm, model, after, "after compile()")


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_two_writes_without_a_compile_leave_the_last(case):
    """Writes compose: the second write since a compile is the one every
    entry point runs."""
    _, factory, key, value, model = case
    gm = _graph(factory)
    node = gm.get_node("n")
    before = dict(node.params)
    node.params[key] = value
    gm.step()
    node.params[key] = before[key]           # written back to the original
    _assert_all_run(gm, model, before, "after writing it back")
