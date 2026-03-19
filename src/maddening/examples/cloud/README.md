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
