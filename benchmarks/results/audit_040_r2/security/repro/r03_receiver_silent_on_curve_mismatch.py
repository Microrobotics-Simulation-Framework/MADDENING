"""The address rule's asymmetry is LOUD for WorkerClient and SILENT for viz.

Topology: a relay inside a container binds tcp://0.0.0.0:P (it must -- a
container bound to 127.0.0.1 is unreachable through a published port), so
transport_auth turns CURVE on.  A client on the host reaches the published
port over tcp://127.0.0.1:P, so transport_auth turns CURVE *off*.  This is
the same asymmetry WorkerClient documents and raises ConnectionError for.

NetworkReceiver and CommandReceiver hit it with no exception, no warning
log, no state flag, and no timeout: latest_snapshot() just returns None
for ever, which is indistinguishable from an idle simulation.
"""
import logging, socket, time
import zmq
from maddening.cloud.multigpu.coordinator import Coordinator
from maddening.cloud.multigpu.worker_client import WorkerClient
from maddening.viz.network import CommandPublisher, CommandReceiver, NetworkReceiver, NetworkRelay

TOKEN = "the-shared-token"
STATE = {"robot": {"joint_angle": 1.25, "secret_position": 42.0}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeGM:
    timestep = 0.01
    def __init__(self): self._obs = []
    def add_observer(self, cb): self._obs.append(cb)
    def emit(self, s):
        for cb in self._obs: cb("step", s)


class Capture(logging.Handler):
    def __init__(self): super().__init__(); self.records = []
    def emit(self, record): self.records.append(self.format(record))


def viz_case():
    cap = Capture(); cap.setLevel(logging.DEBUG)
    root = logging.getLogger(); root.addHandler(cap); root.setLevel(logging.DEBUG)
    port = free_port()
    relay = NetworkRelay(address=f"tcp://0.0.0.0:{port}", token=TOKEN)   # container
    gm = FakeGM(); relay.attach(gm)
    rx = NetworkReceiver(address=f"tcp://127.0.0.1:{port}")              # host side
    rx.start()
    print(f"  relay.secure = {relay.secure}   receiver secure = {rx._secure}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6.0:
        gm.emit(STATE); time.sleep(0.02)
    snap = rx.latest_snapshot()
    rx.stop(); relay.close()
    root.removeHandler(cap)
    print(f"  after 6s of publishing, latest_snapshot() = {snap}")
    print(f"  exceptions raised: none")
    print(f"  log records mentioning the mismatch: "
          f"{[r for r in cap.records if 'CURVE' in r or 'token' in r.lower()]}")


def command_case():
    port = free_port()
    pub = CommandPublisher(address=f"tcp://0.0.0.0:{port}", token=TOKEN)
    rx = CommandReceiver(address=f"tcp://127.0.0.1:{port}")
    rx.start()
    print(f"  publisher.secure = {pub.secure}  receiver secure = {rx._secure}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 4.0:
        pub.send({"robot": {"joint_torques": [0.1, -0.2, 0.0]}}); time.sleep(0.02)
    got = rx.latest_commands()
    rx.stop(); pub.close()
    print(f"  after 4s, latest_commands() = {got}  (control input silently dead)")


def worker_case():
    port = free_port()
    coord = Coordinator(expected_workers=["flow"], edges=[], port=port,
                        bind_host="0.0.0.0", secure=True, token=TOKEN)
    coord.start()
    client = WorkerClient(coordinator_addr=f"127.0.0.1:{port}",
                          subgraph_id="flow", address="127.0.0.1:5555")
    print(f"  coordinator.secure = {coord.secure}  worker secure = {client.secure}")
    try:
        client.register_and_wait(timeout=3)
        print("  worker: NO ERROR (unexpected)")
    except Exception as exc:
        print(f"  worker raised {type(exc).__name__}: {str(exc).splitlines()[0][:90]}...")
    finally:
        coord.shutdown(); time.sleep(1.3)


if __name__ == "__main__":
    print("NetworkRelay(0.0.0.0) <- NetworkReceiver(127.0.0.1):")
    viz_case()
    print("\nCommandPublisher(0.0.0.0) <- CommandReceiver(127.0.0.1):")
    command_case()
    print("\nCoordinator(0.0.0.0) <- WorkerClient(127.0.0.1)  [the documented one]:")
    worker_case()
