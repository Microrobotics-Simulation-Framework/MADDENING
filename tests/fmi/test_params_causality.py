"""FMI ``parameter`` causality backed by the graph parameter pytree.

* the model description lists every ``gm.params`` leaf as a tunable
  parameter with ``ParamSpec`` description/units;
* the sidecar serves / writes them (``get_params`` / ``set_params``), a
  written value changes the next step without a recompile, and the FMU
  state snapshot carries them;
* forward-mode directional derivatives with respect to a parameter go
  through the same ``jax.jvp`` as everything else.
"""

import io
import os
import pickle
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi.fmu_state import deserialize_fmu_state, serialize_fmu_state
from maddening.fmi.model_description import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


@pytest.fixture
def gm():
    g = GraphManager()
    # A table edge, so the ball's step reads ``elasticity``: the FMU exports
    # only parameters its step reads, and without a table ``elasticity`` is
    # a knob that does nothing.
    g.add_node(TableNode(name="table", timestep=1e-2))
    g.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0,
                        elasticity=0.7))
    g.add_node(SpringDamperNode(name="spring", timestep=1e-2, stiffness=30.0,
                                damping=2.0, mass=1.5))
    g.add_edge("table", "ball", "position", "table_position")
    g.compile()
    return g


def _sidecar(gm, *, allow_pickle_rpc=False):
    """A sidecar for ``gm``.

    ``allow_pickle_rpc`` is the opt-in ``FmuSidecar.handle`` needs: it
    unpickles its request, so it refuses to run without it.  Only the two
    tests that exercise that wire protocol pass it.
    """
    md = build_model_description(gm, model_name="m")
    return md, FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token,
        step_fn=gm._compiled_step,
        initial_state=gm._state,
        params=gm.params,
        param_specs=gm.param_specs(),
        allow_pickle_rpc=allow_pickle_rpc,
    ))


class TestModelDescription:

    def test_every_params_leaf_the_step_reads_is_a_tunable_parameter(self, gm):
        """And no other: an ``initial_*`` condition (the FMU's initial state
        is already built) or the table's ``position`` (read by its initial
        state only) would be a knob that does nothing."""
        md = build_model_description(gm, model_name="m")
        params = {v.name: v for v in md.variables if v.causality == "parameter"}
        reads = gm._params_read_by_step()
        expected = {f"{n}.params.{k}" for n, leaves in gm.params["nodes"].items()
                    for k in leaves if (n, k) in reads}
        assert set(params) == expected
        every = {f"{n}.params.{k}" for n, leaves in gm.params["nodes"].items()
                 for k in leaves}
        assert set(md.fixed_parameters) == every - expected
        assert {"spring.params.initial_position", "ball.params.initial_velocity",
                "table.params.position"} <= set(md.fixed_parameters)
        assert all(v.variability == "tunable" for v in params.values())
        assert params["spring.params.stiffness"].unit == "N/m"
        assert params["spring.params.mass"].unit == "kg"
        assert params["ball.params.elasticity"].description
        assert params["ball.params.gravity"].dtype == "float32"
        # FMI 3.0: parameters carry a start value (fmpy validates this)
        assert params["spring.params.stiffness"].start == "30.0"
        assert all(v.start is not None for v in params.values())
        # value references are unique across inputs/outputs/parameters
        vrs = [v.value_reference for v in md.variables]
        assert len(vrs) == len(set(vrs))

    def test_parameters_can_be_left_out_and_change_the_token(self, gm):
        with_p = build_model_description(gm, model_name="m")
        without = build_model_description(gm, model_name="m", include_parameters=False)
        assert not [v for v in without.variables if v.causality == "parameter"]
        assert with_p.instantiation_token != without.instantiation_token

    def test_parameters_appear_in_xml(self, gm):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(build_model_description(gm, model_name="m").to_xml())
        names = {el.get("name"): el for el in root.iter() if el.get("causality") == "parameter"}
        assert "spring.params.stiffness" in names
        assert names["spring.params.stiffness"].get("variability") == "tunable"


class TestSidecar:

    def test_get_params_lists_flat_names(self, gm):
        _, sc = _sidecar(gm)
        p = sc.get_params()
        assert set(p) >= {"spring.params.stiffness", "spring.params.damping", "ball.params.gravity"}
        assert float(p["spring.params.stiffness"]) == 30.0

    def test_set_params_changes_next_step_without_recompile(self, gm):
        _, sc = _sidecar(gm)
        compiled_id = id(gm._compiled_step)
        ext = gm._default_external_inputs()
        base = sc.step(ext)["spring"]["velocity"]
        sc2 = _sidecar(gm)[1]
        sc2.set_params({"spring.params.stiffness": 300.0})
        assert float(sc2.get_params()["spring.params.stiffness"]) == 300.0
        stiff = sc2.step(ext)["spring"]["velocity"]
        assert not np.isclose(float(base), float(stiff))
        assert id(gm._compiled_step) == compiled_id
        assert not gm._dirty
        # the caller's pytree is untouched
        assert float(gm.params["nodes"]["spring"]["stiffness"]) == 30.0

    def test_set_params_rejects_unknown_name_and_bad_shape(self, gm):
        _, sc = _sidecar(gm)
        with pytest.raises(KeyError, match="unknown parameter 'spring.params.stifness'"):
            sc.set_params({"spring.params.stifness": 1.0})
        with pytest.raises(KeyError):
            sc.set_params({"ghost.params.stiffness": 1.0})
        with pytest.raises(ValueError, match="shape"):
            sc.set_params({"spring.params.stiffness": [1.0, 2.0]})

    @pytest.mark.parametrize("with_specs", [False, True], ids=["no_specs", "specs"])
    @pytest.mark.parametrize("value, refusal", [
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (float("-inf"), "must be finite"),
        (1e39, "does not fit its type float32"),     # finite float64, inf as float32
        (-1e39, "does not fit its type float32"),
    ])
    def test_set_params_refuses_values_the_leaf_cannot_hold(self, gm, with_specs, value,
                                                            refusal):
        """The bridge's ``set`` refused these; ``set_params`` checked them
        only through ``ParamSpec.check``, so a sidecar built without
        ``param_specs`` stored NaN, and ``1e39`` as float32 ``inf``.  The
        refusal is atomic: the valid update beside it is not written."""
        md = build_model_description(gm, model_name="m")
        sc = FmuSidecar(SidecarConfig(
            schema_token=md.instantiation_token, step_fn=gm._compiled_step,
            initial_state=gm._state, params=gm.params,
            param_specs=gm.param_specs() if with_specs else None))
        with pytest.raises(ValueError,
                           match=rf"parameter 'spring.params.stiffness': value {refusal}"):
            sc.set_params({"spring.params.damping": 3.0, "spring.params.stiffness": value})
        got = sc.get_params()
        assert float(got["spring.params.stiffness"]) == 30.0
        assert float(got["spring.params.damping"]) == 2.0
        # the largest float32 is a value the leaf can hold, and is accepted
        big = float(np.finfo(np.float32).max)
        sc.set_params({"spring.params.stiffness": big})
        assert float(sc.get_params()["spring.params.stiffness"]) == big

    def test_set_params_refuses_what_the_bridge_set_refuses(self, gm):
        """One value check for both doors, with the same words."""
        from maddening.fmi.tcp_bridge import FmuTcpBridge

        md = build_model_description(gm, model_name="m")
        sc = FmuSidecar(SidecarConfig(
            schema_token=md.instantiation_token, step_fn=gm._compiled_step,
            initial_state=gm._state, params=gm.params))
        bridge = FmuTcpBridge(sc, md, master_dt=1e-2)
        try:
            k = next(v.value_reference for v in md.variables
                     if v.name == "spring.params.stiffness")
            for value in (1e39, float("nan")):
                reply = bridge.handle({"op": "set", "vr": [k], "values": [value]})
                assert reply["ok"] is False
                with pytest.raises(ValueError) as exc:
                    sc.set_params({"spring.params.stiffness": value})
                tail = str(exc.value).split(": ", 1)[1]
                assert reply["error"].endswith(tail), (reply["error"], str(exc.value))
        finally:
            bridge.stop()

    def test_set_params_without_params_config_errors(self):
        sc = FmuSidecar(SidecarConfig(
            schema_token="t", step_fn=lambda s, e: s,
            initial_state={"n": {"x": jnp.array(0.0)}},
        ))
        assert sc.get_params() == {}
        with pytest.raises(RuntimeError, match="no parameter variables"):
            sc.set_params({"n.x": 1.0})

    def test_fmu_state_round_trip_carries_params(self, gm):
        _, sc = _sidecar(gm)
        sc.set_params({"spring.params.stiffness": 123.0})
        ext = gm._default_external_inputs()
        sc.step(ext)
        snap = sc.get_fmu_state()

        _, fresh = _sidecar(gm)
        assert float(fresh.get_params()["spring.params.stiffness"]) == 30.0
        fresh.set_fmu_state(snap)
        assert float(fresh.get_params()["spring.params.stiffness"]) == 123.0
        np.testing.assert_array_equal(
            np.asarray(fresh.state["spring"]["position"]),
            np.asarray(sc.state["spring"]["position"]),
        )
        # and the restored sidecar steps with the restored parameter
        a = fresh.step(ext)["spring"]["velocity"]
        b = sc.step(ext)["spring"]["velocity"]
        assert float(a) == float(b)

    def test_legacy_snapshot_without_params_still_loads(self):
        snap = serialize_fmu_state(state={"n": {"x": jnp.array(2.0)}}, schema_token="t")
        state, params = deserialize_fmu_state(snap, expected_schema_token="t",
                                              return_params=True)
        assert params is None and float(state["n"]["x"]) == 2.0
        assert float(deserialize_fmu_state(snap, expected_schema_token="t")["n"]["x"]) == 2.0

    def test_wire_protocol_get_set_params(self, gm):
        _, sc = _sidecar(gm, allow_pickle_rpc=True)
        status, got = pickle.loads(sc.handle(pickle.dumps(("get_params",))))
        assert status == "ok" and float(got["spring.params.damping"]) == 2.0
        status, _ = pickle.loads(sc.handle(pickle.dumps(("set_params", {"spring.params.damping": 5.0}))))
        assert status == "ok"
        assert float(sc.get_params()["spring.params.damping"]) == 5.0
        status, err = pickle.loads(sc.handle(pickle.dumps(("set_params", {"nope": 1.0}))))
        assert status == "err" and "unknown parameter" in err


def _snapshot_with(sc, kind, owner, key, value):
    """``sc``'s current state and params as an ``FMUState``, one leaf replaced."""
    state = {n: {f: np.asarray(v) for f, v in fs.items()} for n, fs in sc.state.items()}
    params = {section: {o: {k: np.asarray(v) for k, v in leaves.items()}
                        for o, leaves in owners.items()}
              for section, owners in sc.params.items()}
    target = state[owner] if kind == "state" else params["nodes"][owner]
    target[key] = np.asarray(value)
    return serialize_fmu_state(state=state, schema_token=sc._config.schema_token,
                               params=params)


def _archive_with(bridge, kind, owner, key, value):
    """The bridge's own ``get_state`` archive, one member replaced."""
    from maddening.fmi.tcp_bridge import state_of

    member = f"s/{owner}/{key}" if kind == "state" else f"p/nodes/{owner}/{key}"
    with np.load(io.BytesIO(state_of(bridge.handle({"op": "get_state"}))),
                 allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    assert member in members
    members[member] = np.asarray(value)
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def _frozen(sc):
    return ({n: {f: np.asarray(v).copy() for f, v in fs.items()} for n, fs in sc.state.items()},
            {k: np.asarray(v).copy() for k, v in sc.get_params().items()})


def _assert_unchanged(sc, before):
    state, params = before
    for n, fs in state.items():
        for f, v in fs.items():
            np.testing.assert_array_equal(np.asarray(sc.state[n][f]), v)
    for k, v in params.items():
        np.testing.assert_array_equal(np.asarray(sc.get_params()[k]), v)


#: ``(kind, owner, key, value, refusal)``: one leaf of a snapshot, and what
#: ``set_fmu_state`` and the bridge's ``set_state`` must both say about it
#: (``None``: both accept it).
_SNAPSHOT_CASES = [
    ("state", "spring", "velocity", np.float32(np.nan),
     "FMU state spring.velocity: value must be finite"),
    ("param", "spring", "stiffness", np.float32(np.inf),
     "FMU state param spring.params.stiffness: value must be finite"),
    ("param", "spring", "stiffness", np.float64(1e39),
     "FMU state param spring.params.stiffness: value does not fit its type float32"),
    ("param", "ball", "elasticity", np.float32(1.5), "elasticity'\\]=1.5 above bound 1.0"),
    ("param", "ball", "elasticity", np.float32(0.25), None),
    ("state", "spring", "position", np.zeros(3, np.float32),
     r"^FMU state spring\.position: shape \(3,\) != \(\)$"),
    ("param", "spring", "stiffness", np.zeros(2, np.float32),
     r"^FMU state param spring\.params\.stiffness: shape \(2,\) != \(\)$"),
]


class TestSnapshotValues:
    """``FmuSidecar.set_fmu_state`` is a door into the same tree as the
    bridge's ``set_state``.  It had none of that door's value checks: a
    snapshot with a NaN state field, an ``inf`` parameter or a parameter
    outside its declared bounds was installed without a word, and the
    next step computed with it."""

    def test_a_snapshot_cannot_install_a_non_finite_state_field(self, gm):
        _, sc = _sidecar(gm)
        sc.step(gm._default_external_inputs())
        before = _frozen(sc)
        with pytest.raises(ValueError,
                           match=r"^FMU state spring\.velocity: value must be finite$"):
            sc.set_fmu_state(_snapshot_with(sc, "state", "spring", "velocity", np.nan))
        _assert_unchanged(sc, before)

    def test_a_snapshot_cannot_install_a_non_finite_parameter(self, gm):
        _, sc = _sidecar(gm)
        before = _frozen(sc)
        with pytest.raises(ValueError, match=r"^FMU state param spring\.params\.stiffness: "
                                             r"value must be finite$"):
            sc.set_fmu_state(_snapshot_with(sc, "param", "spring", "stiffness",
                                            np.float32(np.inf)))
        _assert_unchanged(sc, before)
        # and one that is finite as float64 but inf in the float32 leaf
        with pytest.raises(ValueError, match="does not fit its type float32"):
            sc.set_fmu_state(_snapshot_with(sc, "param", "spring", "stiffness",
                                            np.float64(1e39)))
        _assert_unchanged(sc, before)

    def test_a_snapshot_cannot_install_a_parameter_outside_its_bounds(self, gm):
        _, sc = _sidecar(gm)                          # carries gm.param_specs()
        before = _frozen(sc)
        with pytest.raises(ValueError, match=r"elasticity'\]=1\.5 above bound 1\.0"):
            sc.set_fmu_state(_snapshot_with(sc, "param", "ball", "elasticity",
                                            np.float32(1.5)))
        _assert_unchanged(sc, before)
        # without declared bounds there is nothing to hold it to, as on the bridge
        md = build_model_description(gm, model_name="m")
        plain = FmuSidecar(SidecarConfig(
            schema_token=md.instantiation_token, step_fn=gm._compiled_step,
            initial_state=gm._state, params=gm.params))
        plain.set_fmu_state(_snapshot_with(plain, "param", "ball", "elasticity",
                                           np.float32(1.5)))
        assert float(plain.get_params()["ball.params.elasticity"]) == 1.5

    def test_a_healthy_snapshot_still_round_trips(self, gm):
        _, sc = _sidecar(gm)
        ext = gm._default_external_inputs()
        sc.step(ext)
        snap = sc.get_fmu_state()
        want = np.asarray(sc.state["spring"]["position"]).copy()
        sc.step(ext)
        sc.set_fmu_state(snap)
        np.testing.assert_array_equal(np.asarray(sc.state["spring"]["position"]), want)
        assert np.asarray(sc.state["spring"]["position"]).dtype == want.dtype
        # a leaf that arrives wider than the live one lands in the live dtype,
        # as it does through the bridge: a float64 carry would retrace the step
        sc.set_fmu_state(_snapshot_with(sc, "state", "spring", "position", np.float64(0.3)))
        restored = np.asarray(sc.state["spring"]["position"])
        assert restored.dtype == np.float32 and restored == np.float32(0.3)
        sc.set_fmu_state(_snapshot_with(sc, "param", "spring", "stiffness", np.float64(45.0)))
        assert np.asarray(sc.get_params()["spring.params.stiffness"]).dtype == np.float32

    @pytest.mark.parametrize("kind, owner, key, value, refusal", _SNAPSHOT_CASES,
                             ids=["nan_state", "inf_param", "overflow_param",
                                  "out_of_bounds_param", "healthy_param",
                                  "misshapen_state", "misshapen_param"])
    def test_set_fmu_state_refuses_what_the_bridge_set_state_refuses(
            self, gm, kind, owner, key, value, refusal):
        """The two restore paths share their checks, so they cannot drift:
        the same leaf is accepted by both or refused by both, with the same
        words."""
        from maddening.fmi.tcp_bridge import FmuTcpBridge

        md, sc = _sidecar(gm)
        _, bridge_sc = _sidecar(gm)
        bridge = FmuTcpBridge(bridge_sc, md, master_dt=1e-2)
        try:
            reply = bridge.handle({"op": "set_state",
                                   "state": _archive_with(bridge, kind, owner, key, value)})
            snap = _snapshot_with(sc, kind, owner, key, value)
            if refusal is None:
                assert reply == {"ok": True}, reply
                sc.set_fmu_state(snap)
                got = bridge_sc.get_params()[f"{owner}.params.{key}"]
                assert float(sc.get_params()[f"{owner}.params.{key}"]) == float(got)
                return
            with pytest.raises(ValueError) as exc:
                sc.set_fmu_state(snap)
            assert reply["ok"] is False
            assert reply["error"] == f"ValueError: {exc.value}", (reply["error"], str(exc.value))
            assert re.search(refusal, str(exc.value)), str(exc.value)
        finally:
            bridge.stop()


class TestDirectionalDerivativeWrtParameter:

    def test_forward_mode_wrt_stiffness_matches_grad(self, gm):
        ext = gm._default_external_inputs()

        def f(k):
            p = jax.tree.map(lambda x: x, gm.params)
            p["nodes"]["spring"]["stiffness"] = k
            return gm._compiled_step(gm._state, ext, p)["spring"]["velocity"]

        k0 = jnp.asarray(30.0, jnp.float32)
        _, t = jax.jvp(f, (k0,), (jnp.ones_like(k0),))
        g = jax.grad(f)(k0)
        assert float(t) != 0.0
        assert np.isclose(float(t), float(g), rtol=1e-5)


class TestBoundsThroughFMI:
    """``ParamSpec.bounds`` reach the importer (XML min/max) and are
    enforced by the sidecar, the same rule as ``PUT /graph/params``."""

    def test_model_description_emits_min_max_from_param_spec(self, gm):
        import xml.etree.ElementTree as ET

        gm.set_param_spec("spring", "stiffness", ParamSpec(bounds=(1.0, 100.0)))
        md = build_model_description(gm, model_name="m")
        by_name = {v.name: v for v in md.variables}
        assert by_name["spring.params.stiffness"].min == 1.0
        assert by_name["spring.params.stiffness"].max == 100.0
        # BallNode declares elasticity in [0, 1] and gravity <= 0
        assert (by_name["ball.params.elasticity"].min,
                by_name["ball.params.elasticity"].max) == (0.0, 1.0)
        root = ET.fromstring(md.to_xml())
        el = next(v for v in root.find("ModelVariables")
                  if v.get("name") == "spring.params.stiffness")
        assert el.get("min") == "1.0" and el.get("max") == "100.0"
        half = next(v for v in root.find("ModelVariables")
                    if v.get("name") == "spring.params.damping")
        assert half.get("min") == "0.0" and half.get("max") is None
        unbounded = next(v for v in root.find("ModelVariables")
                         if v.get("name") == "ball.params.gravity")
        assert unbounded.get("min") is None and unbounded.get("max") is None

    def test_set_params_rejects_out_of_bounds_atomically(self, gm):
        _, sc = _sidecar(gm, allow_pickle_rpc=True)
        before = {k: np.asarray(v).copy() for k, v in sc.get_params().items()}
        with pytest.raises(ValueError, match="ball.params.elasticity.*above bound"):
            sc.set_params({"spring.params.damping": 9.0,
                           "ball.params.elasticity": 1.5})
        after = sc.get_params()
        for k, v in before.items():
            np.testing.assert_array_equal(np.asarray(after[k]), v)   # damping untouched
        # the wire protocol reports it as an error, not a crash
        status, err = pickle.loads(sc.handle(pickle.dumps(
            ("set_params", {"ball.params.elasticity": -0.1}))))
        assert status == "err" and "below bound" in err
        # in-bounds still works
        sc.set_params({"ball.params.elasticity": 0.9})
        assert float(sc.get_params()["ball.params.elasticity"]) == pytest.approx(0.9)

    def test_sidecar_without_specs_keeps_old_behaviour(self, gm):
        md = build_model_description(gm, model_name="m")
        sc = FmuSidecar(SidecarConfig(
            schema_token=md.instantiation_token, step_fn=gm._compiled_step,
            initial_state=gm._state, params=gm.params))
        sc.set_params({"ball.params.elasticity": 1.5})      # no specs: no check
        assert float(sc.get_params()["ball.params.elasticity"]) == 1.5
