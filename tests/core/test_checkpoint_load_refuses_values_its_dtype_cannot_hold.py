"""``load_state`` refuses a checkpoint value the graph's dtype cannot hold.

A checkpoint written in a wider dtype than the graph runs in -- a graph run
under ``jax_enable_x64`` and loaded without it, or a file written by hand --
used to be cast with no check: a float64 ``1e39`` loaded into a float32
field as ``inf``, and ``1e-50`` as ``0.0``, its sign lost, with nothing
said (from v0.1.0, where ``jnp.array`` narrowed every float64 leaf to the
canonical float32).  The rule is the FMU's (MADD-ANO-137): a finite value
that overflows to ``inf``, or a non-zero one flushed to ``+-0``, is refused;
one that rounds to a subnormal keeps its sign and magnitude and loads; a
value already ``inf`` or ``NaN`` loads as it was; an integer that would
wrap is refused.  A refused load leaves the graph as it was.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, rest_length=1.0,
                                 initial_position=0.5))
    gm.compile()
    return gm


def _checkpoint(gm, tmp_path, **replace):
    """*gm*'s checkpoint with the named members replaced (``s/position`` and the like)."""
    path = tmp_path / "c.npz"
    gm.save_state(str(path))
    with np.load(path) as data:
        members = {k: data[k] for k in data.files}
    for key, value in replace.items():
        members[key.replace("__", "/")] = np.asarray(value)
    np.savez(path, **members)
    return path


def _snapshot(gm):
    return ({n: {k: np.asarray(v) for k, v in gm.get_node_state(n).items()}
             for n in gm.node_names},
            {k: np.asarray(v) for k, v in gm.params["nodes"]["s"].items()})


def _same(a, b):
    sa, pa = a
    sb, pb = b
    return all(np.array_equal(sa[n][k], sb[n][k], equal_nan=True) for n in sa for k in sa[n]) \
        and all(np.array_equal(pa[k], pb[k]) for k in pa)


@pytest.mark.parametrize("value", [1e39, -1e39, 1e-50, -1e-50])
def test_a_float64_value_a_float32_field_would_lose_is_refused(tmp_path, value):
    gm = _spring()
    path = _checkpoint(gm, tmp_path, s__position=np.float64(value))
    before = _snapshot(gm)
    with pytest.raises(ValueError, match=r"field 's/position' holds .* float32 cannot hold"):
        gm.load_state(str(path))
    assert _same(_snapshot(gm), before)


@pytest.mark.parametrize("value", [1e-40, -1e-40, 3.4e38, 0.0, -0.0])
def test_a_value_float32_holds_loads_a_subnormal_with_its_sign(tmp_path, value):
    gm = _spring()
    gm.load_state(str(_checkpoint(gm, tmp_path, s__position=np.float64(value))))
    got = np.asarray(gm.get_node_state("s")["position"])
    assert got.dtype == np.float32
    assert got == np.float32(value) and np.signbit(got) == np.signbit(np.float64(value))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_a_value_that_was_already_non_finite_loads_as_it_was(tmp_path, value):
    """A diverged state, a ``NaN``-seeded diagnostics slot: not a value the cast lost."""
    gm = _spring()
    gm.load_state(str(_checkpoint(gm, tmp_path, s__velocity=np.float64(value))))
    got = np.asarray(gm.get_node_state("s")["velocity"])
    assert np.array_equal(got, np.float32(value), equal_nan=True)


def test_a_params_leaf_is_held_to_the_same_rule(tmp_path):
    gm = _spring()
    path = _checkpoint(gm, tmp_path, **{"_params__s__stiffness": np.float64(1e39)})
    before = _snapshot(gm)
    with pytest.raises(ValueError, match=r"params nodes\['s'\]\['stiffness'\] holds"):
        gm.load_state(str(path))
    assert _same(_snapshot(gm), before)
    path = _checkpoint(gm, tmp_path, **{"_params__s__stiffness": np.float64(45.0)})
    gm.load_state(str(path))
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 45.0


class _Counter(SimulationNode):
    """A node with an int32 state field."""

    def __init__(self, name):
        super().__init__(name, 0.01)

    def initial_state(self):
        return {"count": jnp.asarray(0, jnp.int32)}

    def update(self, state, boundary_inputs, dt):
        return {"count": state["count"] + 1}


def test_an_integer_that_would_wrap_is_refused_and_one_that_fits_loads(tmp_path):
    gm = GraphManager()
    gm.add_node(_Counter("c"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    for bad in (np.int64(2 ** 40), np.float64(2.5), np.float64(np.nan)):
        path = tmp_path / "c.npz"
        np.savez(path, **{"c/count": np.asarray(bad)})
        with pytest.raises(ValueError, match=r"field 'c/count' holds .* int32 cannot hold"):
            gm.load_state(str(path))
        assert int(gm.get_node_state("c")["count"]) == 0
    np.savez(tmp_path / "c.npz", **{"c/count": np.asarray(np.int64(7))})
    gm.load_state(str(tmp_path / "c.npz"))
    assert int(gm.get_node_state("c")["count"]) == 7


class _Typed(SimulationNode):
    """A node with one state field of each of the given dtypes."""

    def __init__(self, name, dtypes):
        super().__init__(name, 0.01)
        self._dtypes = tuple(dtypes)

    def initial_state(self):
        return {t: jnp.zeros((), t) for t in self._dtypes}

    def update(self, state, boundary_inputs, dt):
        return dict(state)


#: ``(leaf dtype, the checkpoint's value, its dtype)``: an integer of the
#: other signedness, of the leaf's own width or wider.
SIGN_WRAPS = [
    ("uint32", -1, "int32"), ("uint32", -1, "int64"), ("uint8", -1, "int8"),
    ("uint16", -32768, "int16"), ("int32", 4_000_000_000, "uint32"),
    ("int8", 200, "uint8"), ("int16", 65535, "uint16"),
]


@pytest.mark.parametrize("leaf, value, carrier", SIGN_WRAPS)
def test_an_integer_of_the_other_signedness_is_refused_not_wrapped(tmp_path, leaf, value, carrier):
    """``-1`` for an unsigned field loaded as the type's maximum, and
    4000000000 for an ``int32`` one as -294967296: the check cast the value
    to the leaf's type and back, which between a signed and an unsigned
    type of one width always returns the value it started with."""
    gm = GraphManager()
    gm.add_node(_Typed("c", [leaf]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    path = tmp_path / "c.npz"
    np.savez(path, **{f"c/{leaf}": np.asarray(value, dtype=carrier)})
    with pytest.raises(ValueError, match=rf"field 'c/{leaf}' holds {value} .* {leaf} cannot hold"):
        gm.load_state(str(path))
    assert int(gm.get_node_state("c")[leaf]) == 0
    # the type's own extremes, carried in the other signedness, load
    info = np.iinfo(leaf)
    for ok in {max(info.min, np.iinfo(carrier).min), min(info.max, np.iinfo(carrier).max)}:
        np.savez(path, **{f"c/{leaf}": np.asarray(ok, dtype=carrier)})
        gm.load_state(str(path))
        assert int(gm.get_node_state("c")[leaf]) == ok


def test_a_complex_value_with_an_imaginary_part_is_refused_for_a_real_field(tmp_path):
    """The cast to a real dtype drops the imaginary part; ``3 + 2j`` loaded
    as ``3.0``.  A complex value that is real loads."""
    gm = _spring()
    before = _snapshot(gm)
    path = _checkpoint(gm, tmp_path, s__position=np.complex128(3 + 2j))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # numpy's own ComplexWarning on the cast
        with pytest.raises(ValueError, match=r"field 's/position' holds .* float32 cannot hold"):
            gm.load_state(str(path))
        assert _same(_snapshot(gm), before)
        gm.load_state(str(_checkpoint(gm, tmp_path, s__position=np.complex128(3 + 0j))))
    assert float(gm.get_node_state("s")["position"]) == 3.0


def test_the_rest_route_answers_the_refusal_naming_no_path(tmp_path):
    """POST /checkpoint/load passes load_state's own refusal on, without the server's
    paths."""
    from fastapi.testclient import TestClient

    from maddening.api.server import SimulationServer
    from tests.property.differential import no_cloud_launch

    gm = _spring()
    _checkpoint(gm, tmp_path, s__position=np.float64(1e-50))
    with no_cloud_launch():
        server = SimulationServer({"SpringDamperNode": SpringDamperNode}, graph_manager=gm,
                                  checkpoint_root=str(tmp_path))
        client = TestClient(server.create_app(), raise_server_exceptions=False)
        resp = client.post("/checkpoint/load", params={"path": "c.npz"})
    assert resp.status_code == 400, resp.text
    assert "float32 cannot hold" in resp.json()["detail"]
    assert str(tmp_path) not in resp.json()["detail"]
