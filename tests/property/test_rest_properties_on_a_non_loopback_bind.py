"""The REST surface's two property oracles, on a server that demands the token.

``tests/property/test_differential_rest_params.py`` (REST-089, "a REST
write is a reload") and ``tests/property/test_stateful_api.py`` (REST-090,
the server against an in-process model) drive a server bound, as the
default is, to loopback, where no request carries a token.  Their claims
are stated for any bind.  These run the same oracles, the same draws, on a
server told it is bound to ``0.0.0.0`` -- the token enforced, every request
carrying ``Authorization: Bearer <token>`` -- the
``non_loopback_bind`` and ``token_enforced`` cells of
``docs/validation/rest_runpod_claims.yaml``.

Nothing here can reach a cloud provider (:func:`no_cloud_launch`; neither
oracle sends a request to ``/cloud/*``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient
from hypothesis import event, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import run_state_machine_as_test

from maddening.api.server import SimulationServer
from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_FLOOR
from tests.property import test_differential_rest_params as differential
from tests.property import test_stateful_api as stateful
from tests.property.differential import no_cloud_launch, tmp_dir
from tests.property.node_catalogue import CHEAP_KINDS, KINDS, REGISTRY, writes

TOKEN = "property-token-not-a-credential"
BEARER = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _public_client(gm, root, registry) -> TestClient:
    """The oracle's client, of a server told it is bound to 0.0.0.0, every
    request carrying the token."""
    server = SimulationServer(node_registry=registry, graph_manager=gm, checkpoint_root=root,
                              bind_host="0.0.0.0", api_token=TOKEN)
    assert server.auth.enforced
    return TestClient(server.create_app(), raise_server_exceptions=False, headers=BEARER)


@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_rest_param_write_on_a_non_loopback_bind_is_refused_whole_or_runs_as_its_reload(
        kind_name, data):
    """REST-089 on a non-loopback bind: one built-in node, one generated
    write, through the real route with the bearer token -- refused with
    nothing written, or the saved graph runs what the running graph runs
    (``check_rest_write``, the loopback oracle's own check, with its client
    swapped for one of a server that demands the token)."""
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    write = data.draw(writes(kind, kwargs), label="write")
    event(f"category={write.category}")
    gm = kind.graph(kwargs)
    calibrate = data.draw(st.sampled_from(sorted(kind.safe) + [None]), label="calibrated")
    if calibrate is not None:
        live = gm.params["nodes"][kind.name]
        lo, hi = kind.safe[calibrate]
        moved = np.clip(np.asarray(live[calibrate]) * np.float32(1.25), lo, hi)
        live[calibrate] = jnp.asarray(moved.astype(np.asarray(live[calibrate]).dtype))
    with pytest.MonkeyPatch.context() as mp, tmp_dir() as root:
        mp.setattr(differential, "_client", _public_client)
        outcome = differential.check_rest_write(gm, REGISTRY, kind.name, write, root=root,
                                                rest_reset=False)
    if write.category in ("non_finite", "oversized", "wrong_type", "unknown", "invalid",
                          "mixed"):
        assert outcome == "refused", f"{write.category} write {write.params!r} was accepted"


class _PublicMachine(stateful.SimulationServerMachine):
    """The stateful machine, its server told it is bound to 0.0.0.0 and its
    client presenting the token on every call."""

    def __init__(self) -> None:
        super().__init__()
        self.server = SimulationServer(node_registry=stateful.REGISTRY,
                                       checkpoint_root=str(self.server_root),
                                       bind_host="0.0.0.0", api_token=TOKEN)
        assert self.server.auth.enforced
        self.client = TestClient(self.server.create_app(), raise_server_exceptions=False,
                                 headers=BEARER)


def test_short_rest_sequences_keep_the_server_and_the_model_in_step_on_a_non_loopback_bind():
    """REST-090 on a non-loopback bind: the per-push depth of the loopback
    machine (six calls a sequence, the house floor of examples, drawn the
    same way every run), every call presenting the bearer token."""
    run_state_machine_as_test(
        _PublicMachine,
        settings=settings(stateful_step_count=6, max_examples=EXAMPLES_FLOOR,
                          derandomize=True),
    )
