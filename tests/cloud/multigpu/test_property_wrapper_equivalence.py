"""A sharded wrapper answers for the node it wraps -- for every wrapper.

The worst defect of the 0.4.0 cycle was a *family* defect, not a node
defect: ``ShardedStencilNode`` and ``ShardedUnstructuredNode`` proxied
the parameter contract to their inner node and ``ShardedPointwiseNode``
proxied none of it, so ``PUT /graph/params`` on a pointwise-sharded node
answered 200, echoed the new value, and left the physics running on the
old one (whole-tree audit W2, MADD-ANO registry entry "sharded pointwise
parameter writes").  An example-based test finds that only if someone
writes the example for the wrapper that happens to be broken.

The invariant is sameness across a family, so it is stated once here and
parametrised over :data:`property_support.WRAPPER_FAMILY`.  A fourth
wrapper is covered by adding one entry to that mapping.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from tests.cloud.multigpu.property_support import (
    WRAPPER_FAMILY,
    device_counts,
    param_values,
    wrapper_labels,
)
from tests.conftest import EXAMPLES_COSTLY

_BASELINE_RATE = 0.5


def _graph_for(case) -> GraphManager:
    """A compiled single-node graph around the wrapper under test."""
    gm = GraphManager()
    gm.add_node(case.wrapped)
    for field, value in case.boundary_inputs.items():
        gm.add_external_input(target_node=case.wrapped.name,
                              target_field=field,
                              shape=tuple(jnp.shape(value)))
    gm.compile()
    return gm


def _client(gm: GraphManager, case):
    from fastapi.testclient import TestClient

    from maddening.api.server import SimulationServer

    inner_cls = type(case.inner)
    server = SimulationServer({inner_cls.__name__: inner_cls}, gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _run_graph(gm, case, steps: int) -> dict:
    external = ({case.wrapped.name: dict(case.boundary_inputs)}
                if case.boundary_inputs else None)
    for _ in range(steps):
        gm.step(external)
    return {k: np.asarray(jax.device_get(v))
            for k, v in gm.get_node_state(case.wrapped.name).items()}


# ---------------------------------------------------------------------------
# The contract the wrapper proxies
# ---------------------------------------------------------------------------


@given(label=wrapper_labels(), n_devices=device_counts())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_sharded_wrapper_reports_the_parameter_contract_of_its_inner_node(
        label, n_devices):
    """``accepts_params``, ``params_pytree`` and ``param_specs`` proxy.

    All three have to agree with each other as well as with the inner
    node: a wrapper that answers ``False`` while still listing pytree
    leaves invites every caller to write a parameter nothing reads.
    """
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    inner, wrapped = case.inner, case.wrapped

    assert wrapped.accepts_params() == inner.accepts_params()
    assert set(wrapped.params_pytree()) == set(inner.params_pytree())
    for key, leaf in inner.params_pytree().items():
        np.testing.assert_allclose(np.asarray(wrapped.params_pytree()[key]),
                                   np.asarray(leaf))
    assert wrapped.param_specs() == inner.param_specs()
    if not wrapped.accepts_params():
        assert wrapped.params_pytree() == {}
    # One node, one params dict: a write through any surface must land
    # on the dict the inner ``update`` reads.
    assert wrapped.params is inner.params


@given(label=wrapper_labels(), n_devices=device_counts())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_sharded_wrapper_reports_the_interface_of_its_inner_node(
        label, n_devices):
    """State fields, boundary inputs and the node's own identity proxy."""
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    inner, wrapped = case.inner, case.wrapped

    assert wrapped.state_fields() == inner.state_fields()
    assert wrapped.boundary_input_spec() == inner.boundary_input_spec()
    assert wrapped.name == inner.name
    assert wrapped.delta_t == inner.delta_t
    serialised = wrapped.to_dict()
    assert serialised["sharded"] is True
    # The wrapper is a wrapper, not a node type of its own: the config
    # records the physics node's type (see the round-trip module for
    # what that costs on reload).
    assert serialised["type"] == type(inner).__name__


@given(label=wrapper_labels(), n_devices=device_counts())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_compiled_graph_carries_the_parameters_of_a_sharded_node(
        label, n_devices):
    """``compile`` must not drop a sharded node from ``gm.params``.

    This is the half of W2 that silently disabled calibration: a node
    missing from ``gm.params["nodes"]`` is invisible to ``sysid``, to
    FMI parameter variables and to checkpointed constants, with no error
    anywhere.
    """
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    gm = _graph_for(case)
    entry = gm.params["nodes"].get(case.wrapped.name)
    assert entry is not None, f"{label} wrapper missing from gm.params"
    assert set(entry) == set(case.inner.params_pytree())
    assert float(entry[case.param_name]) == pytest.approx(_BASELINE_RATE)
    # ...and the graph accepts a spec override for it, which it refuses
    # for a node it believes takes no params.
    gm.set_param_spec(case.wrapped.name, case.param_name,
                      case.inner.param_specs()[case.param_name])


# ---------------------------------------------------------------------------
# A write reaches the physics
# ---------------------------------------------------------------------------


@given(label=wrapper_labels(), n_devices=device_counts(), rate=param_values(),
       steps=st.integers(min_value=1, max_value=3))
@settings(max_examples=EXAMPLES_COSTLY)
def test_an_injected_parameter_reaches_the_sharded_physics(
        label, n_devices, rate, steps):
    """Running with ``params={p: v}`` equals running a node built with ``v``.

    The strong form of "the write reached the physics": not merely that
    the trajectory changed, but that it changed into exactly the
    trajectory of the constant that was written.
    """
    assume(abs(rate - _BASELINE_RATE) > 0.05)
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    injected = case.run(steps=steps, sharded=True,
                        params={case.param_name: jnp.float32(rate)})

    built_in = WRAPPER_FAMILY[label](n_devices=n_devices, rate=rate)
    reference = built_in.run(steps=steps, sharded=False)
    baseline = case.run(steps=steps, sharded=False)

    for field in reference:
        np.testing.assert_allclose(injected[field], reference[field],
                                   rtol=1e-5, atol=1e-6, err_msg=field)
    assert any(not np.allclose(injected[f], baseline[f], rtol=1e-5, atol=1e-6)
               for f in reference), "the injected parameter changed nothing"


@given(label=wrapper_labels(), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_rest_parameter_write_to_a_sharded_node_reaches_the_physics(
        label, rate):
    """``PUT /graph/params`` answering 200 must mean the physics changed.

    The user-visible half of W2, generalised over the family: the
    endpoint echoed the new value, a following ``GET`` reported it, and
    the simulation kept running on the old constant.
    """
    assume(abs(rate - _BASELINE_RATE) > 0.05)
    n_devices = min(4, len(jax.devices()))
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    gm = _graph_for(case)

    response = _client(gm, case).put(f"/graph/params/{case.wrapped.name}",
                                     json={"params": {case.param_name: rate}})
    assert response.status_code == 200, response.text
    assert response.json()["params"][case.param_name] == pytest.approx(rate)

    written = _run_graph(gm, case, steps=2)
    reference = WRAPPER_FAMILY[label](n_devices=n_devices, rate=rate)
    expected = _run_graph(_graph_for(reference), reference, steps=2)
    for field in expected:
        np.testing.assert_allclose(written[field], expected[field],
                                   rtol=1e-5, atol=1e-6, err_msg=field)


@given(label=wrapper_labels())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_rest_write_outside_a_sharded_nodes_bounds_is_refused(label):
    """The inner node's ``ParamSpec`` bounds still guard the write.

    A wrapper that loses ``param_specs()`` loses the bounds with it, and
    the refusal that should name the parameter becomes a 200.
    """
    n_devices = min(4, len(jax.devices()))
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=_BASELINE_RATE)
    low, _high = case.inner.param_specs()[case.param_name].bounds
    assume(low is not None)
    gm = _graph_for(case)

    response = _client(gm, case).put(f"/graph/params/{case.wrapped.name}",
                                     json={"params": {case.param_name: low - 1.0}})
    assert response.status_code == 400, response.text
    assert case.param_name in response.json()["detail"]
    assert float(gm.params["nodes"][case.wrapped.name][case.param_name]) == (
        pytest.approx(_BASELINE_RATE))
