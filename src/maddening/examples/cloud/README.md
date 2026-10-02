# Cloud Examples

Most scripts here **provision billable VMs**.  The ones marked *local* below
have a mode that needs no cloud account; start there.

## Without a cloud account

```bash
# The server script 04 installs on the VM, run on this machine on a free
# loopback port, with the same endpoint checks:
python -m maddening.examples.cloud.server.04_server_test --local
# The same for the JSON and binary WebSocket checks:
python -m maddening.examples.cloud.server.05_websocket_test --local
# The multi-job rendezvous (coordinator + two workers) on loopback:
python -m maddening.examples.cloud.multijob.08_two_vm_test --local
# Field subscription and compression over the binary WebSocket (local only):
python -m maddening.examples.cloud.streaming.08_subscribe_lbm_velocity
```

None of these import the cloud launcher.

## Directory Structure

```
cloud/
├── config/                          # Configuration templates
│   ├── cloud_credentials.example.yaml   # API key template → ~/.maddening/
│   └── job_config.example.yaml          # Job config template (safe to commit)
├── launch/                          # VM provisioning + lifecycle
│   ├── 01_validate.py                   # Config + credential validation, cost guards
│   ├── 02_runpod_launch.py              # Real launch, status, teardown (RunPod)
│   ├── 03_lambda_launch.py              # The same on Lambda Labs
│   ├── 03_reconnect_test.py             # CloudJob.from_cluster_name() test
│   ├── 04_aws_launch.py                 # The same on AWS
│   └── 05_gcp_launch.py                 # The same on GCP
├── server/                          # Simulation server on a cloud GPU
│   ├── 04_server_test.py                # REST API checks           (--local)
│   └── 05_websocket_test.py             # JSON + binary WS streaming (--local)
├── multijob/
│   └── 08_two_vm_test.py                # Two-VM rendezvous         (--local)
├── multigpu/
│   └── 09_real_gpu_benchmark.py         # Single- vs multi-GPU timings, 2x RTX 4090
└── streaming/                       # WebRTC / Selkies streaming
    ├── 06_selkies_test.py               # GStreamer pipeline on a cloud GPU
    ├── 07_webrtc_streaming_test.py      # Full WebRTC pipeline + profiling
    └── 08_subscribe_lbm_velocity.py     # Binary-WS field subscription (local only)
```

## Setup

```bash
pip install "maddening[runpod]"      # or [lambda], [aws], [gcp]
mkdir -p ~/.maddening
# The templates ship with MADDENING; this prints the directory holding them:
python -c "import maddening.examples.cloud.config as c; print(c.__path__[0])"
cp <that directory>/cloud_credentials.example.yaml ~/.maddening/cloud_credentials.yaml
cp <that directory>/job_config.example.yaml job.yaml
# Edit the credentials with your API key, and job.yaml to taste
```

## Running

Start with validation:
```bash
python -m maddening.examples.cloud.launch.01_validate --job job.yaml
```

Then try a real launch (or `--dry-run` first):
```bash
python -m maddening.examples.cloud.launch.02_runpod_launch --job job.yaml
```

Full server test (provisions VM, installs deps, starts server, tests API):
```bash
python -m maddening.examples.cloud.server.04_server_test --gpu RTX4090
```

## Security: the server these examples start needs a token

The cloud image binds `0.0.0.0`, so the API behind it requires
`Authorization: Bearer <token>` on every route.  Set `MADDENING_API_TOKEN`
in your job's `envs:` and pass the same value from the client, or read the
generated token out of the container log.  There is still **no TLS**, so
the token crosses the network in cleartext.

`JobConfig.ports` no longer defaults to `[8000]`, so a default launch does
**not** open the API in the provider's firewall.  `server/` and
`streaming/` add `8000` back to `ports:` explicitly, which puts the API on
the VM's public NAT address behind the token.  These scripts are
demonstrations on a throwaway VM, not a deployment pattern.

For anything you care about:

* leave `8000` out of `ports:` so the provider's firewall stays shut, and
  reach the API through an SSH tunnel instead —
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
