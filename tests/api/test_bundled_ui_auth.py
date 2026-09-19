"""The bundled pages must be able to present the API's bearer token.

The three HTML pages ``/viz/app``, ``/viz/graph`` and ``/viz/render`` are
served without a credential -- they contain none -- and then have to find
one for every call they make.  ``static/auth.js`` does that: it takes the
token from ``?token=``, scrubs it out of the URL, keeps it in
``sessionStorage`` and wraps ``window.fetch``.

These are contract tests between two languages.  Nothing else notices if
the JavaScript and the Python stop agreeing on the subprotocol name or
the base64 alphabet: every page would simply stop connecting, at runtime,
in a browser nobody runs in CI.
"""

import json
import os
import re
import shutil
import subprocess

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.api import server as server_module
from maddening.api.auth import (
    WS_BEARER_PREFIX,
    WS_SUBPROTOCOL,
    encode_ws_bearer,
    websocket_credentials,
)

STATIC = server_module._STATIC_DIR

#: The pages ``SimulationServer`` itself serves.  ``vessel_flow.html``
#: belongs to ``examples/servers/vessel_flow_server.py``, a separate app.
BUNDLED_PAGES = ["app.html", "graph.html", "render.html"]

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(
    NODE is None,
    reason=(
        "the JavaScript/Python carrier-encoding contract can only be "
        "executed with a JS engine; node is not on PATH here. The "
        "constant-level checks in this module run regardless."
    ),
)


def _page(name):
    return (STATIC / name).read_text()


@pytest.mark.parametrize("name", BUNDLED_PAGES)
def test_a_bundled_page_loads_the_token_helper_before_its_own_script(name):
    """``auth.js`` wraps ``window.fetch``; it has to win the race."""
    text = _page(name)
    helper = text.find('<script src="/viz/auth.js"></script>')
    assert helper != -1, f"{name} does not load /viz/auth.js"
    own = text.find("\n<script>\n")
    assert own != -1 and helper < own, f"{name} loads auth.js too late"


@pytest.mark.parametrize("name", BUNDLED_PAGES)
def test_a_bundled_page_never_opens_a_raw_websocket(name):
    """A raw ``new WebSocket`` offers no subprotocol and gets a 403."""
    assert "new WebSocket" not in _page(name)
    assert "mdWebSocket(" in _page(name)


@pytest.mark.parametrize("name", BUNDLED_PAGES)
def test_a_bundled_page_holds_no_credential(name):
    """Which is why the pages are exempt from the token in the first place."""
    text = _page(name)
    assert "MADDENING_API_TOKEN" not in text
    assert WS_BEARER_PREFIX not in text


def test_the_helper_scrubs_the_token_out_of_the_url():
    """A token left in the address bar reaches history and Referer."""
    js = (STATIC / "auth.js").read_text()
    assert "history.replaceState" in js
    assert 'params.delete("token")' in js
    # sessionStorage only: localStorage would leave the credential behind
    # for the next person to open this browser profile.
    assert "localStorage.setItem" not in js
    assert "localStorage.getItem" not in js
    assert "sessionStorage.setItem" in js


def test_the_helper_and_the_server_agree_on_the_subprotocol_names():
    js = (STATIC / "auth.js").read_text()
    assert f'WS_SUBPROTOCOL = "{WS_SUBPROTOCOL}"' in js
    assert f'WS_BEARER_PREFIX = "{WS_BEARER_PREFIX}"' in js


@needs_node
def test_the_helper_builds_the_subprotocols_the_server_decodes(tmp_path):
    """Run the real ``auth.js`` and compare with the Python encoder.

    A drift in the base64 alphabet or the padding rule would leave every
    bundled page unable to connect, with nothing in Python to notice.
    """
    tokens = [
        "plain-token",
        "with+slash/and=pad",
        "unicode-éè-token",
        "a" * 43,
        "comma,and space",          # illegal raw, must survive encoding
    ]
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        "import fs from 'node:fs';\n"
        "const store = {};\n"
        "globalThis.window = globalThis;\n"
        "globalThis.location = {\n"
        "  origin: 'http://h', pathname: '/viz/app', search: '', hash: '',\n"
        "  href: 'http://h/viz/app',\n"
        "};\n"
        "globalThis.history = { replaceState() {} };\n"
        "globalThis.sessionStorage = {\n"
        "  getItem: k => (k in store ? store[k] : null),\n"
        "  setItem: (k, v) => { store[k] = v; },\n"
        "};\n"
        "globalThis.prompt = () => null;\n"
        "globalThis.fetch = async () => ({ status: 200 });\n"
        f"const src = fs.readFileSync({json.dumps(str(STATIC / 'auth.js'))}, 'utf8');\n"
        "const out = [];\n"
        f"for (const token of {json.dumps(tokens)}) {{\n"
        "  store['maddening.api.token'] = token;\n"
        "  globalThis.maddeningAuth = undefined;\n"
        "  globalThis.location.search = '';\n"
        "  (0, eval)(src);\n"
        "  out.push(window.maddeningAuth.protocols());\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [NODE, str(harness)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    produced = json.loads(result.stdout)
    assert produced == [websocket_credentials(t) for t in tokens]
    # And the server decodes what the browser produced.
    for token, offered in zip(tokens, produced):
        assert offered[0] == encode_ws_bearer(token)


@needs_node
def test_the_helper_takes_the_token_from_the_query_string(tmp_path):
    """The delivery channel: open ``/viz/app?token=<token>`` once."""
    harness = tmp_path / "query.mjs"
    harness.write_text(
        "import fs from 'node:fs';\n"
        "const store = {};\n"
        "let replaced = null;\n"
        "globalThis.window = globalThis;\n"
        "globalThis.location = {\n"
        "  origin: 'http://h', pathname: '/viz/app',\n"
        "  search: '?token=sekret&fps=30', hash: '', href: 'http://h/viz/app',\n"
        "};\n"
        "globalThis.history = { replaceState: (a, b, url) => { replaced = url; } };\n"
        "globalThis.sessionStorage = {\n"
        "  getItem: k => (k in store ? store[k] : null),\n"
        "  setItem: (k, v) => { store[k] = v; },\n"
        "};\n"
        "globalThis.prompt = () => null;\n"
        "globalThis.fetch = async () => ({ status: 200 });\n"
        f"(0, eval)(fs.readFileSync({json.dumps(str(STATIC / 'auth.js'))}, 'utf8'));\n"
        "process.stdout.write(JSON.stringify({\n"
        "  token: window.maddeningAuth.token(),\n"
        "  replaced, stored: store['maddening.api.token'],\n"
        "}));\n"
    )
    result = subprocess.run(
        [NODE, str(harness)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got["token"] == "sekret"
    assert got["stored"] == "sekret"
    # Scrubbed from the URL, and the page's other parameters survive.
    assert "token" not in got["replaced"]
    assert "fps=30" in got["replaced"]


@needs_node
def test_the_wrapped_fetch_adds_the_header_to_same_origin_calls(tmp_path):
    harness = tmp_path / "fetch.mjs"
    harness.write_text(
        "import fs from 'node:fs';\n"
        "const seen = [];\n"
        "const store = { 'maddening.api.token': 'sekret' };\n"
        "globalThis.window = globalThis;\n"
        "globalThis.location = {\n"
        "  origin: 'http://h', pathname: '/viz/app', search: '', hash: '',\n"
        "  href: 'http://h/viz/app',\n"
        "};\n"
        "globalThis.history = { replaceState() {} };\n"
        "globalThis.sessionStorage = {\n"
        "  getItem: k => (k in store ? store[k] : null),\n"
        "  setItem: (k, v) => { store[k] = v; },\n"
        "};\n"
        "globalThis.prompt = () => null;\n"
        "globalThis.fetch = async (url, init) => {\n"
        "  seen.push([url, init && init.headers"
        " ? (new Headers(init.headers)).get('authorization') : null]);\n"
        "  return { status: 200 };\n"
        "};\n"
        f"(0, eval)(fs.readFileSync({json.dumps(str(STATIC / 'auth.js'))}, 'utf8'));\n"
        "await window.fetch('http://h/graph');\n"
        "await window.fetch('http://h/sim/step', { method: 'POST' });\n"
        "await window.fetch('http://elsewhere/x');\n"
        "process.stdout.write(JSON.stringify(seen));\n"
    )
    result = subprocess.run(
        [NODE, str(harness)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen[0][1] == "Bearer sekret"
    assert seen[1][1] == "Bearer sekret"
    # A cross-origin call must not leak this server's credential.
    assert seen[2][1] is None


def test_the_helper_is_served_and_is_javascript():
    """The route exists, is exempt, and says it is JavaScript."""
    from fastapi.testclient import TestClient

    from maddening.api.server import SimulationServer

    client = TestClient(
        SimulationServer(
            node_registry={}, bind_host="0.0.0.0", api_token="t",
        ).create_app()
    )
    response = client.get("/viz/auth.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/javascript")
    assert "maddeningAuth" in response.text


def test_no_bundled_page_is_left_out_of_this_module():
    """A fourth page served by ``/viz/*`` must be added here too."""
    served = set(
        re.findall(r'_STATIC_DIR / "([^"]+\.html)"',
                   (server_module.__file__ and
                    open(server_module.__file__, encoding="utf-8").read()))
    )
    assert served == set(BUNDLED_PAGES), (
        f"server.py serves {sorted(served)}, this module checks "
        f"{sorted(BUNDLED_PAGES)}"
    )
