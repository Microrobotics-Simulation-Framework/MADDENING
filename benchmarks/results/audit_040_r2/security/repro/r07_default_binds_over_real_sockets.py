"""What the shipped defaults actually bind, read out of the kernel.

Not from the source: each object is constructed with no arguments and
the listening address is read back from /proc/net/tcp{,6} for this
process's own sockets.  0.0.0.0 (00000000:....) or :: would mean the
default still publishes off-box.
"""
import os, socket, struct, sys

def listening_for_this_pid():
    """{(addr, port)} this process is listening on, from /proc."""
    inodes = set()
    for fd in os.listdir(f"/proc/{os.getpid()}/fd"):
        try:
            target = os.readlink(f"/proc/{os.getpid()}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    out = set()
    for path, size in (("/proc/net/tcp", 4), ("/proc/net/tcp6", 16)):
        try:
            lines = open(path).read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            local, state, inode = f[1], f[3], f[9]
            if state != "0A" or inode not in inodes:   # 0A = LISTEN
                continue
            hexaddr, hexport = local.split(":")
            raw = bytes.fromhex(hexaddr)
            if size == 4:
                addr = socket.inet_ntop(socket.AF_INET, struct.pack("<I", struct.unpack(">I", raw)[0]))
            else:
                words = struct.unpack("<4I", raw)
                addr = socket.inet_ntop(socket.AF_INET6, struct.pack(">4I", *words))
            out.add((addr, int(hexport, 16)))
    return out


def report(label, before):
    now = listening_for_this_pid()
    new = sorted(now - before)
    for addr, port in new:
        bad = addr in ("0.0.0.0", "::")
        print(f"  {label:<28} listening on {addr}:{port}"
              f"{'   <-- OFF-BOX' if bad else '   (loopback only)'}")
    if not new:
        print(f"  {label:<28} (no new listening socket seen)")
    return now


if __name__ == "__main__":
    base = listening_for_this_pid()

    from maddening.viz.network import CommandPublisher, NetworkRelay
    try:
        relay = NetworkRelay()
        base = report("NetworkRelay()", base)
    except Exception as exc:
        print(f"  NetworkRelay(): {type(exc).__name__}: {exc}"); relay = None
    try:
        pub = CommandPublisher()
        base = report("CommandPublisher()", base)
    except Exception as exc:
        print(f"  CommandPublisher(): {type(exc).__name__}: {exc}"); pub = None

    from maddening.cloud.multigpu.coordinator import Coordinator
    coord = Coordinator(expected_workers=["a"], edges=[])
    coord.start()
    import time; time.sleep(0.5)
    base = report("Coordinator()", base)

    from maddening.fmi.model_description import build_model_description
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
    from maddening.fmi.tcp_bridge import FmuTcpBridge
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.spring import SpringDamperNode
    gm = GraphManager(); gm.add_node(SpringDamperNode(name="spring", timestep=0.01)); gm.compile()
    md = build_model_description(gm, model_name="m")
    sc = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))
    bridge = FmuTcpBridge(sc, md, master_dt=0.01)
    print(f"  FmuTcpBridge default endpoint = {bridge.endpoint}  "
          f"(constructor defaults host='127.0.0.1', port=0 -- NOT 5555, "
          f"which MADD-ANO-015 residual_risk states)")
    base = report("FmuTcpBridge(defaults)", base)

    bridge.stop(); coord.shutdown()
    if relay: relay.close()
    if pub: pub.close()
    time.sleep(1.3)
