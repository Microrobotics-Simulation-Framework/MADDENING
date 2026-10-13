"""The REST inventory's claims, in every in-process server domain.

``docs/validation/rest_runpod_claims.yaml`` gives each REST row a
``domains`` matrix.  The rows' own tests run each claim on a loopback bind,
one request at a time, on a quiescent graph; this module runs the check
``tests/api/rest_claims_support.py`` states for the row (once each, keyed
by row id) in the other domains, each on a fresh server and graph:

* ``loopback_bind`` and ``no_token``: ``bind_host="127.0.0.1"``, and no
  ``Authorization`` header on any request (the peer is TestClient's
  ``testclient``, not an IP address, unless the check names one);
* ``non_loopback_bind`` and ``token_enforced``: ``bind_host="0.0.0.0"``,
  every request carrying ``Authorization: Bearer <token>`` except the
  anonymous ones an authentication check sends on purpose;
* ``runner_active``: ``POST /sim/start`` first; the realtime runner steps
  the graph for the whole check, and must still be running at its end;
* ``sim_run_active``: a ``POST /sim/run`` of ``MAX_RUN_STEPS`` is in
  flight for the whole check, its slices slowed to one 30 ms step so a
  request can reach the graph between two, and must still be running at
  the check's end (it is then told to stop, as a shutdown does);
* ``checkpoint_restore``: the graph is run, saved by
  ``POST /checkpoint/save``, run on, and restored by
  ``POST /checkpoint/load`` before the check;
* ``wrapper_nodes``: the spring is a ``HybridNode`` and the rods ``a``
  (the mapping's source) and ``c`` are ``ShardedStencilNode`` on a
  one-device mesh, so a check writing to them writes through the wrapper;
* ``shutdown``: SIGTERM arrives -- through the handler
  ``SimulationServer`` chains ahead of the one installed -- while the
  check's first request holds the graph lock.

A check that cannot run in a domain is left out of that domain's
parameters; the inventory's cell says why (``n/a`` or a narrowed
condition).  The ``concurrent`` domain -- several copies of each check at
once over a real uvicorn server on loopback -- is
``tests/api/test_rest_claims_under_concurrent_requests.py``.

Nothing here can reach a cloud provider: no check sends a request to
``/cloud/*``, and the module runs under
:func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from tests.api import rest_claims_support as S
from tests.property.differential import no_cloud_launch


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


@pytest.mark.parametrize("row", S.rows_for("loopback"))
def test_the_claim_holds_on_a_loopback_bind_without_a_token(row, tmp_path):
    """``bind_host="127.0.0.1"``; no request carries a token."""
    S.run_check(row, "loopback", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("public"))
def test_the_claim_holds_on_a_non_loopback_bind_with_the_token(row, tmp_path):
    """``bind_host="0.0.0.0"``: the token is enforced, and every request of
    the check presents it as ``Authorization: Bearer <token>`` (S.BEARER)."""
    assert S.BEARER["Authorization"].startswith("Bearer ")
    S.run_check(row, "public", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("runner"))
def test_the_claim_holds_while_the_runner_runs(row, tmp_path):
    """``POST /sim/start`` first: the realtime runner steps the graph for
    the whole check."""
    S.run_check(row, "runner", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("sim_run"))
def test_the_claim_holds_while_a_sim_run_is_in_flight(row, tmp_path):
    """A ``POST /sim/run`` is in flight for the whole check, stepping one
    slowed slice at a time."""
    S.run_check(row, "sim_run", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("restored"))
def test_the_claim_holds_on_a_graph_restored_from_a_checkpoint(row, tmp_path):
    """Run, ``POST /checkpoint/save``, run on, ``POST /checkpoint/load``;
    then the check."""
    S.run_check(row, "restored", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("wrapper"))
def test_the_claim_holds_on_a_wrapper_node(row, tmp_path):
    """The spring a ``HybridNode``, the rods ``a`` and ``c``
    ``ShardedStencilNode`` wrappers on a one-device mesh."""
    S.run_check(row, "wrapper", tmp_path)


@pytest.mark.parametrize("row", S.rows_for("shutdown"))
def test_the_claim_holds_when_sigterm_arrives_mid_request(row, tmp_path):
    """SIGTERM, chained ahead of the installed handler as uvicorn's serve
    installs it, arrives while the check's first request holds the graph
    lock (S.mid_request_sigterm)."""
    S.run_check(row, "shutdown", tmp_path)
