# Cloud Examples

## Directory Structure

```
cloud/
├── config/                          # Configuration templates
│   ├── cloud_credentials.example.yaml   # API key template → ~/.maddening/
│   └── job_config.example.yaml          # Job config template (safe to commit)
├── launch/                          # VM provisioning + lifecycle
│   ├── 01_validate.py                   # Dry-run config validation
│   ├── 02_runpod_launch.py              # Real launch, status, teardown
│   └── 03_reconnect_test.py             # CloudJob.from_cluster_name() test
├── server/                          # Simulation server on cloud GPU
│   ├── 04_server_test.py                # REST API (ball+spring on RTX 4090)
│   └── 05_websocket_test.py             # JSON + binary WS streaming
└── streaming/                       # WebRTC / Selkies streaming
    ├── 06_selkies_test.py               # GStreamer pipeline on cloud GPU
    └── 07_webrtc_streaming_test.py      # Full WebRTC pipeline + profiling
```

## Setup

```bash
pip install maddening[runpod]
mkdir -p ~/.maddening
cp config/cloud_credentials.example.yaml ~/.maddening/cloud_credentials.yaml
# Edit with your RunPod API key
```

## Running

Start with validation (no cloud spend):
```bash
python launch/01_validate.py --job config/job_config.example.yaml
```

Then try a real launch:
```bash
python launch/02_runpod_launch.py
```

Full server test (provisions VM, installs deps, starts server, tests API):
```bash
python server/04_server_test.py --gpu RTX4090
```

## Security: the server these examples start is unauthenticated

`server/` and `streaming/` publish port 8000 on the VM's **public** NAT
address, and the API behind it has no authentication and no TLS.  Anyone
who finds that address can rewrite the graph, read your state, and call
`POST /cloud/launch` — which provisions more paid GPU instances with the
credentials on that host — while `/docs` lists every route for them.
These scripts are demonstrations on a throwaway VM, not a deployment
pattern.

For anything you care about:

* drop `8000` from `ports:` in your job config so the provider's firewall
  stays shut, and reach the API through an SSH tunnel instead —
  `ssh -L 8000:127.0.0.1:8000 root@<vm-ip> -p <ssh-port>`, then talk to
  `http://localhost:8000`;
* or terminate TLS and authenticate in a reverse proxy in front of it;
* set `MADDENING_STREAM_SECRET` on the VM if you want the WebRTC
  signaling server to admit a viewer: clients authenticate with
  `generate_session_token(session_id, secret)` (as
  `Authorization: Bearer <token>` or `?token=`), and without a shared
  secret every connection is rejected;
* tear the VM down when you are done (`job.teardown()`), because an idle
  VM with an open API keeps costing money and keeps being reachable.

The server logs a warning at startup whenever it binds a non-loopback
address.
