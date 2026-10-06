"""What a server that demands the token answers a request without it.

A ``SimulationServer`` told that its bind is not loopback demands
``Authorization: Bearer <token>`` of every route but ``/healthz`` and the
static ``/viz/*`` pages (``maddening.api.auth.UNAUTHENTICATED_PATHS``).
Both REST oracles serve a graph in that configuration
(``rest_oracle.serve(token_enforced=True)``) and hold a token-holder's
requests to the invariants they hold a loopback client's to.  This module
is the invariant only that configuration has, written once for both:

    a request that does not present the token -- no ``Authorization``
    header, another token, the token cut short or lengthened, the token
    under another scheme or under none, the token in the query string --
    is answered **401** with ``WWW-Authenticate: Bearer``, **changes
    nothing** (the oracle's snapshot, and the graph object for object),
    starts nothing, and **reveals nothing about the graph**: its body is,
    byte for byte, the one a server with an empty graph gives another
    request.

The last is asked as an identity, not as a search for node names: a body
that does not depend on the request or on the graph cannot carry either.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from tests.property import rest_oracle as O
from tests.property.graph_fingerprint import assert_exactly_as_it_was, served_fingerprint

#: What a request presents in place of ``Authorization: Bearer <token>``:
#: ``(label, the Authorization header or None, whether the token is put in
#: the query string)``.
CREDENTIALS: tuple = (
    ("no Authorization header", None, False),
    ("another token", "Bearer " + O.TOKEN[::-1], False),
    ("the token without its last character", "Bearer " + O.TOKEN[:-1], False),
    ("the token and one character more", "Bearer " + O.TOKEN + "x", False),
    ("the token in another case", "Bearer " + O.TOKEN.upper(), False),
    ("an empty bearer credential", "Bearer ", False),
    ("the token under the Basic scheme", "Basic " + O.TOKEN, False),
    ("the token with no scheme", O.TOKEN, False),
    ("the token in the query string, and no header", None, True),
)


def send_without_the_token(served: O.Served, method: str, url: str, *,
                           authorization: Optional[str] = None, query_token: bool = False,
                           **kwargs: Any):
    """*method* *url* through the served graph's client, with *authorization*
    in place of the token the client presents (no header at all when
    ``None``)."""
    follow_redirects = kwargs.pop("follow_redirects", False)
    request = served.client.build_request(method, url, **kwargs)
    if "authorization" in request.headers:
        del request.headers["authorization"]
    if authorization is not None:
        request.headers["Authorization"] = authorization
    if query_token:
        request.url = request.url.copy_merge_params({"token": O.TOKEN})
    return served.client.send(request, follow_redirects=follow_redirects)


@functools.lru_cache(maxsize=None)
def the_refusal() -> bytes:
    """The body a server with an empty graph answers ``GET /no/such/route``
    without a token: what every refusal must be, byte for byte."""
    served = O.serve(token_enforced=True)
    try:
        resp = send_without_the_token(served, "GET", "/no/such/route")
        assert resp.status_code == 401, resp.text
        return resp.content
    finally:
        served.close()


def refusal_problems(resp) -> list[str]:
    """What is wrong with *resp* as the refusal of a request without the
    token; empty when it is the documented one."""
    problems = []
    if resp.status_code != 401:
        problems.append(f"answered {resp.status_code}, not 401: {resp.text[:200]}")
    elif resp.headers.get("www-authenticate") != "Bearer":
        problems.append(f"401 without 'WWW-Authenticate: Bearer': {dict(resp.headers)}")
    if resp.content != the_refusal():
        problems.append("its body is not the one a server with an empty graph gives "
                        f"another request: {resp.text[:300]}")
    return problems


def assert_refused_without_the_token(served: O.Served, method: str, url: str, what: str, *,
                                     credentials: tuple = CREDENTIALS,
                                     object_for_object: bool = True, **kwargs: Any) -> int:
    """Send the request once per entry of *credentials* and hold each reply
    to the invariant of this module; return how many were sent.  Without
    *object_for_object* "changes nothing" is the oracle's snapshot alone
    (the fuzzer's, which sends thousands)."""
    assert served.token_enforced
    before = O.snapshot(served)
    fingerprint = served_fingerprint(served) if object_for_object else None
    runner = served.server.runner
    was_running = runner is not None and runner.is_alive
    for label, authorization, query_token in credentials:
        resp = send_without_the_token(served, method, url, authorization=authorization,
                                      query_token=query_token, **kwargs)
        where = f"{what}, sent with {label},"
        problems = refusal_problems(resp)
        assert not problems, f"{where} " + "; ".join(problems)
        runner = served.server.runner
        assert (runner is not None and runner.is_alive) == was_running, (
            f"{where} was refused and started or stopped the runner")
    where = f"{what}, sent without the token,"
    O.assert_nothing_changed(before, O.snapshot(served), where)
    if fingerprint is not None:
        assert_exactly_as_it_was(fingerprint, served_fingerprint(served), where)
    return len(credentials)
