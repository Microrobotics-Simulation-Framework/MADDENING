"""``Authorization: Bearer <token>`` is read by RFC 9110's grammar, no more generously.

``credentials = auth-scheme 1*SP token68`` (section 11.4): the scheme is
case-insensitive, one or more spaces separate it from the token, and the
token is compared exactly.  The header's value used to be split on its
first space and the rest ``str.strip()``-ped, so a tab, a form feed or a
line tabulation beside the token was taken as padding and the credential
matched.

The parsing is tested on the function, and the rule on one route that is
not a cloud route (``/graph``); the token rule on ``/cloud/*`` is
``tests/api/test_bearer_auth.py``'s, with its launcher stubbed.
"""

from __future__ import annotations

import pytest

from maddening.api.auth import APIAuth, bearer_from_headers
from tests.property import rest_oracle as O
from tests.property.without_the_token import send_without_the_token

TOKEN = O.TOKEN

#: Header values that present the token, by the grammar.
PRESENTS = [
    f"Bearer {TOKEN}", f"bearer {TOKEN}", f"BEARER {TOKEN}", f"bEaReR {TOKEN}",
    f"Bearer   {TOKEN}",                 # 1*SP
    f"  Bearer {TOKEN}  ", f"\tBearer {TOKEN}\t",   # whitespace around the field value
    f"Bearer   {TOKEN}  ",
]
#: ... and values that do not: another character between the scheme and the
#: token or after it, no space at all, another scheme.
DOES_NOT = [
    f"Bearer\t{TOKEN}", f"Bearer \t{TOKEN}", f"Bearer {TOKEN}\x0b", f"Bearer {TOKEN}\x0c",
    f"Bearer \x0c{TOKEN}", f"Bearer {TOKEN} x", f"Bearer x {TOKEN}", f"Bearer{TOKEN}",
    f"Bearer\xa0{TOKEN}", f"Bearer {TOKEN}\xa0", f"Bearer {TOKEN},", f"Bearer \"{TOKEN}\"",
    f"Bearer: {TOKEN}", f"Bearers {TOKEN}", f"Basic {TOKEN}", f"Token {TOKEN}", TOKEN,
    "Bearer", "Bearer ", "", f"Bearer {TOKEN[:-1]}", f"Bearer {TOKEN.upper()}",
]

#: The routes a credential is sent to here: none of them a cloud route.
ROUTES = [route for route in ("/graph", "/graph/state") if not route.startswith("/cloud")]
assert ROUTES and not [route for route in ROUTES if route.startswith("/cloud")]


@pytest.fixture(scope="module")
def auth() -> APIAuth:
    return APIAuth(bind_host="0.0.0.0", token=TOKEN, environ={})


@pytest.mark.parametrize("value", PRESENTS)
def test_a_header_of_the_grammar_presents_the_token(auth, value):
    assert bearer_from_headers({"authorization": value}) == TOKEN
    assert auth.verify(bearer_from_headers({"authorization": value}))


@pytest.mark.parametrize("value", DOES_NOT)
def test_any_other_header_does_not(auth, value):
    presented = bearer_from_headers({"authorization": value})
    assert presented != TOKEN
    assert not auth.verify(presented)


def test_no_header_presents_nothing(auth):
    assert bearer_from_headers({}) == ""
    assert not auth.verify(bearer_from_headers({}))


@pytest.fixture(scope="module")
def served():
    served = O.serve(O.two_rods(), token_enforced=True)
    yield served
    served.close()


def _sendable(value: str) -> bool:
    """Whether the client can put *value* in a header at all (ASCII, and
    no control character but a tab)."""
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return not any(ch in value for ch in "\x0b\x0c\r\n\x00")


@pytest.mark.parametrize("route", ROUTES)
def test_the_route_serves_a_header_of_the_grammar_and_refuses_any_other(served, route):
    assert not route.startswith("/cloud")
    sent = 0
    for value in PRESENTS:
        reply = send_without_the_token(served, "GET", route, authorization=value)
        assert reply.status_code == 200, (value, reply.status_code)
        sent += 1
    for value in DOES_NOT:
        if not value or not _sendable(value):
            continue
        reply = send_without_the_token(served, "GET", route, authorization=value)
        assert reply.status_code == 401, (value, reply.status_code)
        sent += 1
    assert sent >= len(PRESENTS) + 12
