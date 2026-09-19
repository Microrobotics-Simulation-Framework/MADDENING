"""FMI ``parameter`` causality backed by the graph parameter pytree.

* the model description lists every ``gm.params`` leaf as a tunable
  parameter with ``ParamSpec`` description/units;
* the sidecar serves / writes them (``get_params`` / ``set_params``), a
  written value changes the next step without a recompile, and the FMU
  state snapshot carries them;
* forward-mode directional derivatives with respect to a parameter go
  through the same ``jax.jvp`` as everything else.
"""

import os
import pickle

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


@pytest.fixture
def gm():
    g = GraphManager()
    g.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0,
                        elasticity=0.7))
    g.add_node(SpringDamperNode(name="spring", timestep=1e-2, stiffness=30.0,
                                damping=2.0, mass=1.5))
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

    def test_every_params_leaf_is_a_tunable_parameter(self, gm):
        md = build_model_description(gm, model_name="m")
        params = {v.name: v for v in md.variables if v.causality == "parameter"}
        expected = {f"{n}.params.{k}" for n, leaves in gm.params["nodes"].items()
                    for k in leaves}
        assert set(params) == expected
        assert all(v.variability == "tunable" for v in params.values())
        assert params["spring.params.stiffness"].unit == "N/m"
        assert params["spring.params.mass"].unit == "kg"
        assert params["spring.params.initial_position"].description == "initial condition"
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
