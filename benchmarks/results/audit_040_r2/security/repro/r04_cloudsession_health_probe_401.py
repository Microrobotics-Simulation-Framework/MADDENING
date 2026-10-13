"""CloudSession's container/simulation health probes now always fail.

cloud/entrypoint.py binds the container's API to 0.0.0.0 (it must), which
turns the bearer token on for every route but /healthz and /viz/*.
CloudSession._launch_worker still probes http://<vm>:8000/graph and
/graph/state with maddening.cloud._health.probe_http, which sends no
Authorization header, so both probes get 401.  urllib raises HTTPError
(a URLError), probe_http converts it to HealthProbeError("container"),
wait_for retries for 120 s and then propagates -> CloudStage.ERROR.

_skypilot.launch_vm passes no MADDENING_API_TOKEN into the container, so
CloudSession has no token to present even if probe_http could carry one.
/healthz exists for exactly this and is not used.

Everything below runs on 127.0.0.1; the server is told bind_host="0.0.0.0"
the same way entrypoint.py tells it.
"""
import socket, threading, time
import uvicorn
from maddening.api.server import SimulationServer
from maddening.cloud._health import HealthProbeError, probe_http, wait_for
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01))
    server = SimulationServer(
        node_registry={"SpringDamperNode": SpringDamperNode},
        graph_manager=gm,
        bind_host="0.0.0.0",          # what cloud/entrypoint.py passes
        api_token="container-token",
    )
    port = free_port()
    config = uvicorn.Config(server.create_app(), host="127.0.0.1", port=port,
                            log_level="error")
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)

    base = f"http://127.0.0.1:{port}"
    for path in ("/healthz", "/graph", "/graph/state"):
        try:
            probe_http(f"{base}{path}", timeout=5)
            print(f"  probe_http({path:<14}) -> OK")
        except HealthProbeError as exc:
            print(f"  probe_http({path:<14}) -> HealthProbeError"
                  f"(stage={exc.stage!r}) {exc.detail}")

    print("\n  CloudSession stage 2 is exactly this call:")
    t0 = time.monotonic()
    try:
        wait_for(lambda: probe_http(f"{base}/graph", timeout=2),
                 timeout=6, interval=2)     # 120 s in session.py
        print("  wait_for -> succeeded (unexpected)")
    except HealthProbeError as exc:
        print(f"  wait_for -> HealthProbeError(stage={exc.stage!r}) after "
              f"{time.monotonic()-t0:.1f}s  => CloudStage.ERROR, "
              f"wait_ready().fully_ready is False")
    srv.should_exit = True
    time.sleep(0.3)


if __name__ == "__main__":
    main()
