"""Passive MITM byte-tap on the real ZMQ path: encryption ON vs OFF.

Not a raw ``socket.create_connection`` tap -- that never completes a ZMTP
handshake and captures 10 bytes of greeting whatever the mechanism is.
Here a real, authorised SUB socket connects *through* a transparent TCP
byte-pump to the relay, so the full ZMTP/CURVE handshake and every
published frame cross the tap.  The control (secure=False) must show the
payload in the captured bytes, or the secure case proves nothing.

Run:
  PYTHONPATH=<wt>/src .venv/bin/python r01_wiretap_differential.py
"""
import json, socket, threading, time
import zmq
from maddening.transport_auth import TransportAuth
from maddening.viz.network import NetworkRelay

TOKEN = "the-shared-token"
STATE = {"robot": {"joint_angle": 1.25, "secret_position": 42.0}}
MARKER = "secret_position"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tap(threading.Thread):
    """Transparent TCP pump 127.0.0.1:listen -> 127.0.0.1:target, recording."""

    def __init__(self, listen_port, target_port):
        super().__init__(daemon=True)
        self.captured = bytearray()
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", listen_port))
        self._srv.listen(4)
        self._srv.settimeout(0.5)
        self._target = target_port
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                client, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self._target))
            threading.Thread(target=self._pump, args=(client, upstream, False),
                             daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, client, True),
                             daemon=True).start()

    def _pump(self, a, b, record):
        try:
            while not self._stop.is_set():
                data = a.recv(65536)
                if not data:
                    break
                if record:
                    self.captured += data
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass

    def stop(self):
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass


class FakeGM:
    timestep = 0.01

    def __init__(self):
        self._obs = []

    def add_observer(self, cb):
        self._obs.append(cb)

    def emit(self, state):
        for cb in self._obs:
            cb("step", state)


def measure(secure):
    relay_port, tap_port = free_port(), free_port()
    relay = NetworkRelay(address=f"tcp://127.0.0.1:{relay_port}",
                         secure=secure, token=TOKEN)
    gm = FakeGM()
    relay.attach(gm)
    tap = Tap(tap_port, relay_port)
    tap.start()

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    if secure:
        TransportAuth(token=TOKEN).secure_client(sub)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://127.0.0.1:{tap_port}")

    received = 0
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        gm.emit(STATE)
        try:
            while True:
                sub.recv(zmq.NOBLOCK)
                received += 1
        except zmq.Again:
            pass
        if received:
            time.sleep(0.3)
            try:
                while True:
                    sub.recv(zmq.NOBLOCK)
                    received += 1
            except zmq.Again:
                pass
            break
        time.sleep(0.02)

    sub.close()
    ctx.term()
    tap.stop()
    relay.close()
    return received, bytes(tap.captured)


if __name__ == "__main__":
    for secure in (False, True):
        got, raw = measure(secure)
        readable = MARKER.encode() in raw
        print(f"secure={secure!s:<5} frames_received_by_sub={got:<4} "
              f"bytes_captured_by_tap={len(raw):<6} "
              f"payload_readable_in_capture={readable}")
        assert got > 0, f"secure={secure}: the authorised SUB got nothing; the tap proves nothing"
    print()
    print("Control (secure=False) must be True, secured case must be False.")
