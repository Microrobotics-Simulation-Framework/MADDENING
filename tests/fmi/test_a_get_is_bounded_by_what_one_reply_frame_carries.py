"""A ``get`` costs no more than one reply frame can carry.

One in-limit 64 MiB ``get`` frame naming a scalar output's value reference
33.5 million times used to drive the bridge past 6 GB and hold the FMU
instance for about 45 s: ``FmuTcpBridge._get`` built a small float64 array
per entry before concatenating, and only then could the reply-size check
refuse anything (the B1 round-8 audit's M5; claims FMU-041, FMU-050).

Now a ``get`` naming more value references than a reply frame holds values
(``_MAX_GET_VALUES``, 64 MiB of float64 less room for the header), or whose
variables hold more values than that in total, is refused before anything
is read; each distinct variable is read once and copied into a preallocated
reply, so a repeated reference costs eight bytes; and a JSON reply that
would pass the frame limit is abandoned while it is encoded.
"""

import tracemalloc

import jax.numpy as jnp
import numpy as np
import pytest

import maddening.fmi.tcp_bridge as tb
from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, FmuTcpBridge, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization.json_codec import dumps as json_dumps

CELLS = 1000


@pytest.fixture(scope="module")
def served():
    """A spring (scalar outputs) and a rod of ``CELLS`` temperatures."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 1e-3, initial_position=0.5))
    gm.add_node(HeatNode("h", 1e-3, n_cells=CELLS, length=float(CELLS),
                         thermal_diffusivity=0.5, initial_temperature=100.0))
    gm.compile()
    md = build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)
    bridge = FmuTcpBridge(FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,     # noqa: SLF001
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),  # noqa: SLF001
        fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs)), md, master_dt=gm.timestep)  # noqa: SLF001
    yield md, bridge
    bridge.stop()


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


def _peak_while(fn):
    """``(result, peak bytes Python allocated while fn ran)``."""
    tracemalloc.start()
    try:
        result = fn()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_cap_is_what_a_binary_reply_frame_carries():
    """Binary replies at the cap fit the frame, header included; one more
    value would not."""
    header = tb.encode_binary({"ok": True, "n": tb._MAX_GET_VALUES, "dtype": "f64"}, b"")
    assert len(header) + 8 * tb._MAX_GET_VALUES <= tb._MAX_MESSAGE
    # and it is not needlessly small: 1 KiB of header room, no more
    assert 8 * (tb._MAX_GET_VALUES + 1024 // 8 + 1) > tb._MAX_MESSAGE


@pytest.mark.parametrize("last", ["the same reference", "a float"])
def test_a_get_of_more_references_than_a_reply_frame_holds_is_refused_before_anything_is_read(
        served, last):
    """The audit's frame, at a size past the cap: refused at once,
    allocating next to nothing, where it used to build an array per entry
    (about 300 bytes each) and then refuse the reply.  Refused by its length
    alone, before any entry is looked at -- even one the entry-by-entry
    check would refuse at the end."""
    md, bridge = served
    pos = _vr(md, "s.position")
    vrs = [pos] * (tb._MAX_MESSAGE // 8) + [pos if last == "the same reference" else 2.0]
    reply, peak = _peak_while(lambda: bridge._dispatch(                   # noqa: SLF001
        {"op": "get", "type": "Float32", "vr": vrs}))
    assert reply["ok"] is False and "cannot be answered" in reply["error"], reply
    assert f"a get of {len(vrs)} value references" in reply["error"], reply
    assert "split it into several gets" in reply["error"]
    assert peak < 1 << 20, peak


def test_a_get_whose_variables_hold_more_values_than_a_reply_frame_is_refused_before_reading(
        served):
    """Few references, many values: the rod's temperatures named until they
    total more than the cap.  Refused from the variables' sizes, before any
    value is read or the reply allocated."""
    md, bridge = served
    rod = next(v for v in md.variables if v.name == "h.temperature")
    size = int(np.prod(rod.shape))
    times = tb._MAX_MESSAGE // 8 // size + 1                          # past one frame of f64
    vrs = [rod.value_reference] * times
    reply, peak = _peak_while(lambda: bridge._dispatch(                   # noqa: SLF001
        {"op": "get", "type": "Float32", "vr": vrs}))
    assert reply["ok"] is False and f"hold {times * size} values" in reply["error"], reply
    assert peak < 1 << 20, peak


def test_a_repeated_reference_is_answered_in_place_from_one_read(served):
    """FMI 3.0 does not forbid naming a variable twice in one ``fmi3Get``,
    and the answer is not ambiguous (as a repeated ``set``'s is): every
    occurrence gets the value, in the order asked, on both reply forms."""
    md, bridge = served
    pos, vel, temp, time = (_vr(md, n) for n in
                            ("s.position", "s.velocity", "h.temperature", "time"))
    rod = np.asarray(bridge._sidecar.state["h"]["temperature"], np.float64)  # noqa: SLF001
    got = bridge.handle({"op": "get", "vr": [pos, temp, pos, vel, temp, time, pos]})
    assert got["ok"], got
    expected = np.concatenate([[0.5], rod, [0.5], [0.0], rod, [0.0], [0.5]])
    assert np.array_equal(np.asarray(got["values"]), expected)
    raw = bridge._dispatch({"op": "get", "vr": [pos, pos, pos]})          # noqa: SLF001
    assert raw["values"].dtype == np.float64 and raw["values"].tolist() == [0.5] * 3
    # scalars only (the reply is filled from one value per variable)
    k, c = _vr(md, "s.params.stiffness"), _vr(md, "s.params.damping")
    params = bridge._sidecar.get_params()                                 # noqa: SLF001
    k0, c0 = (float(params[n]) for n in ("s.params.stiffness", "s.params.damping"))
    assert len({0.5, k0, c0}) == 3
    got = bridge.handle({"op": "get", "vr": [k, pos, c, k, pos, c, c]})
    assert got["values"] == [k0, 0.5, c0, k0, 0.5, c0, c0], got
    assert bridge.handle({"op": "get", "vr": []}) == {"ok": True, "values": []}


@pytest.mark.parametrize("vrs, kind, words", [
    ([10_000, "x"], KeyError, "unknown value reference 10000"),
    (["x", 10_000], ValueError, "must be an integer"),
    ([1, True], ValueError, "must be an integer, got True"),
    ([np.int64(1), 2.0], ValueError, "must be an integer, got 2.0"),
], ids=["unknown first", "string first", "boolean", "float"])
def test_a_bad_reference_is_named_in_request_order(served, vrs, kind, words):
    """The first bad entry is the one named, whichever path the request
    takes (plain JSON integers, or anything else); nothing is read."""
    md, bridge = served
    reply = bridge.handle({"op": "get", "vr": vrs})
    assert reply["ok"] is False and reply["error"].startswith(kind.__name__), reply
    assert words in reply["error"], reply


def test_a_repeated_reference_of_another_type_is_refused(served):
    md, bridge = served
    pos = _vr(md, "s.position")
    reply = bridge.handle({"op": "get", "type": "Float64", "vr": [pos, pos]})
    assert reply["ok"] is False and "fmi3GetFloat64 cannot address it" in reply["error"], reply


def test_a_json_get_reply_is_the_bytes_the_whole_reply_encodes_to():
    """The chunked JSON encoding writes, byte for byte, what encoding the
    whole reply dict writes -- non-finite values as their quoted tokens,
    and across a chunk boundary."""
    values = np.array([0.5, -0.0, 1e-300, np.nan, np.inf, -np.inf, 3.0, 123456.789],
                      dtype=np.float64)
    whole = json_dumps(FmuTcpBridge._jsonify({"ok": True, "values": values}),  # noqa: SLF001
                       separators=(",", ":")).encode("utf-8")
    assert FmuTcpBridge._json_values_body(values) == whole                  # noqa: SLF001
    many = np.random.default_rng(0).standard_normal(3 * tb._JSON_VALUES_CHUNK + 7)
    whole = json_dumps(FmuTcpBridge._jsonify({"ok": True, "values": many}),    # noqa: SLF001
                       separators=(",", ":")).encode("utf-8")
    assert FmuTcpBridge._json_values_body(many) == whole                    # noqa: SLF001
    assert FmuTcpBridge._json_values_body(np.zeros(0)) == b'{"ok":true,"values":[]}'  # noqa: SLF001


def test_a_json_reply_too_long_to_send_is_abandoned_while_it_is_encoded(monkeypatch):
    """A reply past the frame limit stops being encoded at the first chunk
    that passes it, rather than being built whole and then refused."""
    monkeypatch.setattr(tb, "_MAX_MESSAGE", 4096)
    monkeypatch.setattr(tb, "_JSON_VALUES_CHUNK", 64)
    chunks = []
    real = tb._json_dumps

    def counting(obj, **kw):
        chunks.append(len(obj))
        return real(obj, **kw)

    monkeypatch.setattr(tb, "_json_dumps", counting)
    values = np.random.default_rng(1).standard_normal(64 * 100)          # ~130 kB of JSON
    assert FmuTcpBridge._json_values_body(values) is None                 # noqa: SLF001
    assert 1 <= len(chunks) <= 4096 // (64 * 4) + 1, len(chunks)
    short = np.full(64, 0.5)
    assert FmuTcpBridge._json_values_body(short) is not None              # noqa: SLF001


def test_a_reply_at_the_cap_goes_out_as_one_binary_frame(served):
    """The largest ``get`` the bridge answers fits a binary frame: a fake
    connection receives it whole, never an error in its place."""
    md, bridge = served
    pos = _vr(md, "s.position")

    class Conn:
        frames: list = []

        def sendall(self, data):
            Conn.frames.append(len(data))

    reply = bridge._dispatch({"op": "get", "vr": [pos] * tb._MAX_GET_VALUES})  # noqa: SLF001
    assert reply["ok"] and reply["values"].size == tb._MAX_GET_VALUES
    assert bridge._send_reply(Conn(), reply, True) is True                # noqa: SLF001
    assert Conn.frames == [4 + len(tb.encode_binary(
        {"ok": True, "n": tb._MAX_GET_VALUES, "dtype": "f64"}, b"")) + 8 * tb._MAX_GET_VALUES]
    assert np.all(reply["values"] == np.float64(jnp.float32(0.5)))
