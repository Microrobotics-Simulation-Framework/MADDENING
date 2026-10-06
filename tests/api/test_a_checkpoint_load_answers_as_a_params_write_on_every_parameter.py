"""``POST /checkpoint/load`` gives a parameter value the verdict ``PUT /graph/params`` gives it.

The property, over every leaf of the params pytree of every built-in node
kind (``tests/property/node_catalogue.py``) and several pairs of values:
write ``v`` with ``PUT``, then load a checkpoint that holds ``w``.  The
load answers as ``PUT w`` does on the node as it stands; after a 200 the
graph steps, ``GET /graph/params`` shows ``w``, and the node runs ``w``
after a reset exactly as it does after the ``PUT``; a refusal changes
nothing and its text advises only routes.

A load used to write ``gm.params`` and not the node's own params, as
``PUT`` does, so the graph's guard against a leaf the step cannot read
refused every checkpoint saved before an ``initial_*`` parameter (or
``TableNode.position``) was written: six of the sixteen numeric parameters
of the four nodes the API serves by default.  The load now installs a
changed leaf through ``PUT``'s own function (``_install_param_leaves``),
and the guard is left with the one case it exists for -- a leaf the load
does not change -- which the last test pins.
"""

from __future__ import annotations

import random
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import find

from maddening.core.simulation import checkpoint as checkpoint_module
from maddening.nodes import HeatNode
from tests.property.node_catalogue import CHEAP_KINDS, COSTLY_KINDS, KINDS
from tests.property.rest_oracle import (
    assert_nothing_changed,
    detail_text,
    serve,
    snapshot,
)

#: Calls a REST client does not have: a load's refusal must name none.
NOT_REST = ("gm.", "remove_node", "add_node", "reset_params", "to_dict()", "save_state()",
            "load_state")


#: Kinds whose node has no parameter leaves: a checkpoint holds none of its
#: values.  Named, so that a kind which stops having any is noticed.
NO_LEAVES = ("health_check",)


def _kwargs(kind_name: str) -> dict:
    """One valid set of constructor arguments of the kind: the simplest
    its catalogue strategy draws, the same on every run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return find(KINDS[kind_name].kwargs, lambda _: True, random=random.Random(0))


def _like(leaf: np.ndarray, value) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=leaf.dtype), leaf.shape).copy()


def _values(kind, key: str, leaf: np.ndarray) -> dict[str, np.ndarray]:
    """Values of the leaf's shape and dtype to write and to load: two
    ordinary ones (inside the catalogue's safe range where it gives one),
    the value the graph was built with, and three that some parameters
    refuse -- a negative one, zero, and one far past any stability limit."""
    if key in kind.safe:
        lo, hi = kind.safe[key]
        moved, other = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    else:
        moved, other = 0.5 * leaf + 0.25, leaf + 1.0
    return {
        "moved": _like(leaf, moved),
        "other": _like(leaf, other),
        "built": leaf.copy(),
        "negative": _like(leaf, -(np.abs(leaf) + 1.0)),
        "zero": _like(leaf, 0.0),
        "huge": _like(leaf, 1.0e30),
    }


#: ``(v, w)``: PUT *v*, then load a checkpoint holding *w*.  The first is
#: the sequence that failed (a checkpoint saved before the write).
PAIRS_PER_PUSH = (("moved", "built"), ("moved", "other"), ("moved", "negative"))
PAIRS_SLOW = PAIRS_PER_PUSH + (("moved", "zero"), ("moved", "huge"), ("built", "moved"),
                               ("negative", "moved"))


def _save_holding(served, owner: str, key: str, value: np.ndarray, name: str) -> None:
    """A checkpoint of the served graph as it is, but for leaf *key*
    holding *value*: what a save made at another time, or by a fit, holds.
    Written by the checkpoint module itself; ``gm.save_state`` and the
    save route refuse a leaf the step cannot read."""
    leaves = served.gm.params["nodes"][owner]
    held = leaves[key]
    leaves[key] = jnp.asarray(value, dtype=held.dtype)
    try:
        checkpoint_module.save_state(served.gm, served.root / name)
    finally:
        leaves[key] = held


def _put(served, owner: str, key: str, value: np.ndarray):
    return served.client.put(f"/graph/params/{owner}",
                             json={"params": {key: np.asarray(value).tolist()}})


def _reset_and_step(served) -> tuple:
    reset = served.client.post("/sim/reset")
    step = served.client.post("/sim/step")
    assert reset.status_code == 200 and step.status_code == 200, (reset.text, step.text)
    return reset.json()["state"], step.json()


def _check_every_parameter(kind_name: str, pairs) -> int:
    kind = KINDS[kind_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        served = serve(kind.graph(_kwargs(kind_name)))
    checked = 0
    try:
        owner = served.gm.node_names[0]
        keys = sorted(served.gm.params["nodes"].get(owner, {}))
        assert bool(keys) is (kind_name not in NO_LEAVES), (kind_name, keys)
        for key in keys:
            leaf = np.asarray(served.gm.params["nodes"][owner][key])
            if leaf.dtype.kind not in "fiu":
                continue
            values = _values(kind, key, leaf)
            for v_name, w_name in pairs:
                v, w = values[v_name], values[w_name]
                what = f"{kind_name}.{key}: PUT {v_name} then load {w_name}"
                start = served.gm._transaction_snapshot()
                try:
                    _save_holding(served, owner, key, w, "w.npz")
                    put_v = _put(served, owner, key, v)
                    assert put_v.status_code < 500, (what, put_v.text)
                    # PUT's verdict on w, on the node as it stands after v
                    # -- and, taken, what the node then runs after a reset.
                    stands = served.gm._transaction_snapshot()
                    put_w = _put(served, owner, key, w)
                    assert put_w.status_code < 500, (what, put_w.text)
                    ran = _reset_and_step(served) if put_w.status_code == 200 else None
                    served.gm._transaction_restore(stands)

                    before = snapshot(served)
                    load = served.client.post("/checkpoint/load", params={"path": "w.npz"})
                    assert load.status_code < 500, (what, load.text)
                    assert (load.status_code == 200) == (put_w.status_code == 200), (
                        f"{what}: the load answered {load.status_code} "
                        f"({detail_text(load)}) where PUT answered {put_w.status_code} "
                        f"({detail_text(put_w)})")
                    if load.status_code == 200:
                        shown = served.client.get(f"/graph/params/{owner}").json()[key]
                        np.testing.assert_array_equal(
                            np.asarray(shown, dtype=leaf.dtype), w, err_msg=what)
                        assert _reset_and_step(served) == ran, (
                            f"{what}: after a reset the node does not run the loaded value "
                            "as it runs the written one")
                    else:
                        assert load.status_code == 400, (what, load.text)
                        assert_nothing_changed(before, snapshot(served), what)
                        text = detail_text(load)
                        assert key in text and "nothing was loaded" in text, (what, text)
                        assert not [call for call in NOT_REST if call in text], (what, text)
                    checked += 1
                finally:
                    served.gm._transaction_restore(start)
    finally:
        served.close()
    return checked


@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
def test_a_load_answers_as_a_params_write_on_every_parameter_of_a_node_kind(kind_name):
    assert (_check_every_parameter(kind_name, PAIRS_PER_PUSH) > 0) \
        is (kind_name not in NO_LEAVES)


# Per push: tests/api/test_a_checkpoint_load_answers_as_a_params_write_on_every_parameter.py::test_a_load_answers_as_a_params_write_on_every_parameter_of_a_node_kind
@pytest.mark.slow  # every pair on every kind; an LBM / pipe / wavelet compile and trace each
@pytest.mark.parametrize("kind_name", CHEAP_KINDS + COSTLY_KINDS)
def test_a_load_answers_as_a_params_write_on_every_value_pair_and_costly_kind(kind_name):
    assert (_check_every_parameter(kind_name, PAIRS_SLOW) > 0) is (kind_name not in NO_LEAVES)


# ---------------------------------------------------------------------------
# The case the load must still refuse: a leaf it does not change
# ---------------------------------------------------------------------------


def _rod(initial_temperature: float = 0.5, grid=None):
    from maddening.core.graph_manager import GraphManager

    gm = GraphManager()
    where = dict(length=1.0) if grid is None else dict(grid_points=grid)
    gm.add_node(HeatNode("rod", 0.01, n_cells=4, thermal_diffusivity=0.001,
                         initial_temperature=initial_temperature, **where))
    gm.compile()
    return gm


GRID, OTHER_GRID = [0.1, 0.3, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9]


@pytest.mark.parametrize("key, own, held, fixable", [
    ("initial_temperature", 0.5, 0.9, True),      # only initial_state() reads it
    ("grid_points", GRID, OTHER_GRID, False),     # consumed when the node was constructed
])
def test_a_load_that_would_leave_a_leaf_the_step_cannot_read_is_refused(key, own, held,
                                                                       fixable):
    """A leaf the graph's params already hold away from the node's own
    value, where the step cannot read it (Python code wrote ``gm.params``),
    and a checkpoint holding the same value: the load changes nothing of
    that leaf, so nothing above asks about it, and a 200 would leave a
    graph whose every step is a 400.  Refused, naming both values and the
    route that puts the node's own back; after that ``PUT`` the load is
    asked what ``PUT`` asks of the checkpoint's value."""
    def build(value):
        return _rod(value) if key == "initial_temperature" else _rod(grid=value)

    served = serve(build(own))
    try:
        build(held).save_state(served.root / "held.npz")
        leaves = served.gm.params["nodes"]["rod"]
        leaves[key] = jnp.asarray(held, dtype=leaves[key].dtype)
        before = snapshot(served)

        load = served.client.post("/checkpoint/load", params={"path": "held.npz"})

        assert load.status_code == 400, load.text
        assert_nothing_changed(before, snapshot(served), "the refused load")
        text = detail_text(load)
        assert "nothing was loaded" in text and f"PUT /graph/params/rod with {key}" in text
        assert f"{own}" in text and f"{held}" in text, text
        assert not [call for call in NOT_REST if call in text], text

        # The advice, followed: the node's own value back, then the load.
        assert _put(served, "rod", key, np.asarray(own)).status_code == 200
        assert served.client.post("/sim/step").status_code == 200
        again = served.client.post("/checkpoint/load", params={"path": "held.npz"})
        assert (again.status_code == 200) is fixable, again.text
        if fixable:
            assert served.gm._nodes["rod"].node.params[key] == pytest.approx(held)
        else:
            # PUT refuses the checkpoint's value too, and the load says so
            assert _put(served, "rod", key, np.asarray(held)).status_code == 400
            assert "PUT /graph/params/rod refuses the same" in detail_text(again)
            assert not [call for call in NOT_REST if call in detail_text(again)]
        assert served.client.post("/sim/step").status_code == 200
    finally:
        served.close()
