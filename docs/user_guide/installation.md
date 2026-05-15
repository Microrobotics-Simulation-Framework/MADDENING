# Installation

## Base Install

MADDENING requires Python 3.10+ and installs with CPU-based {term}`JAX` by default:

```bash
pip install maddening
```

This gives you the full simulation engine (JAX + NumPy) running on CPU. All core features work: graph construction, compilation, stepping, coupling, adaptive timestepping, parameter sweeps, and checkpoint/restore.

## GPU Acceleration

For GPU acceleration, add the hardware extra matching your CUDA version:

```bash
# CUDA 12 (most common)
pip install maddening[cuda12]

# TPU (Google Cloud)
pip install maddening[tpu]
```

You can combine hardware extras with any other extra:

```bash
pip install maddening[cuda12,viz]         # GPU + matplotlib plots
pip install maddening[cuda12,server]      # GPU + FastAPI server
pip install maddening[cuda12,runpod]      # GPU + RunPod cloud deploy
```

> **Note**: The `cuda12` extra upgrades the base JAX installation to the CUDA 12 variant. You must have CUDA 12 drivers installed on your system. Check with `nvidia-smi`.

## Feature Extras

Install only what you need. Each extra adds one capability:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `viz` | Matplotlib 2D renderers | `pip install maddening[viz]` |
| `terminal` | Rich terminal renderer (works over SSH) | `pip install maddening[terminal]` |
| `network` | ZeroMQ remote transport | `pip install maddening[network]` |
| `api` | FastAPI HTTP/WebSocket server | `pip install maddening[api]` |
| `surrogates` | Neural surrogate training (equinox + optax) | `pip install maddening[surrogates]` |
| `viz3d` | PyVista 3D server-side rendering | `pip install maddening[viz3d]` |
| `gpu-viz` | pygfx GPU-accelerated 3D viewer | `pip install maddening[gpu-viz]` |
| `usd` | {term}`OpenUSD` graph serialization | `pip install maddening[usd]` |
| `streaming` | GStreamer {term}`WebRTC` streaming | `pip install maddening[streaming]` |

Mix and match freely:

```bash
pip install maddening[viz,surrogates]           # plots + neural surrogates
pip install maddening[cuda12,api,network]       # GPU server with ZMQ
```

## Cloud Deployment

Deploy simulations to cloud GPU providers via {term}`SkyPilot`. Each provider has its own extra that bundles SkyPilot with the provider's SDK:

| Extra | Provider | Install command |
|-------|----------|-----------------|
| `runpod` | RunPod | `pip install maddening[runpod]` |
| `lambda` | Lambda Labs | `pip install maddening[lambda]` |
| `aws` | Amazon Web Services | `pip install maddening[aws]` |
| `gcp` | Google Cloud Platform | `pip install maddening[gcp]` |
| `cloud` | All supported providers (RunPod + Lambda) | `pip install maddening[cloud]` |
| `cloud-all` | Every SkyPilot provider | `pip install maddening[cloud-all]` |

After installing, set up your credentials file:

```bash
mkdir -p ~/.maddening
cp src/maddening/examples/cloud/cloud_credentials.example.yaml ~/.maddening/cloud_credentials.yaml
# Edit ~/.maddening/cloud_credentials.yaml and fill in your API key(s)
# ~/.maddening/cloud_credentials.yaml is the default path. For a different location, use:
# CloudLauncher(credentials_path='path/to/cloud_credentials.yaml')
```

The credentials file holds keys for all providers in one place:

```yaml
runpod:
  api_key: "rp_xxxxxxxxxxxx"

lambda_labs:
  api_key: "llxxxxxxxxxxxxxxxxxx"
```

`CloudLauncher` reads only the block matching the `provider:` field in your job config. See `src/maddening/examples/cloud/` for complete examples.

## Task Bundles

For common workflows, use a task bundle instead of listing individual extras:

| Bundle | What it includes | Use case |
|--------|-----------------|----------|
| `server` | FastAPI + uvicorn + websockets + ZMQ + rich + matplotlib | Running a simulation server (Docker image, GPU workstation) |
| `client` | ZMQ + rich | Thin client viewing a remote simulation |

```bash
# Cloud GPU Docker image
pip install maddening[server,cuda12]

# Thin client on your laptop
pip install maddening[client,viz]
```

## Everything

```bash
pip install maddening[all]       # all features + supported cloud providers
pip install maddening[dev]       # all + pytest (for development)
```

## Using uv

All install commands work identically with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install maddening[cuda12,runpod]
uv add maddening[server]
```

## Verifying Your Installation

```python
import maddening
import jax

# Check JAX backend
print(f"JAX backend: {jax.devices()[0].platform}")  # 'cpu', 'gpu', or 'tpu'

# Check available features
try:
    from maddening.viz.backends import MatplotlibRenderer
    print("matplotlib: available")
except ImportError:
    print("matplotlib: not installed (pip install maddening[viz])")

try:
    from maddening.cloud import CloudLauncher
    print("cloud: available")
except ImportError:
    print("cloud: not installed (pip install maddening[runpod])")
```

## Missing Dependency Errors

MADDENING uses lazy imports — optional features only fail when you try to use them, not at import time. When a dependency is missing, you'll see a clear error with the exact install command:

```
ImportError: 'SurrogateTrainer' requires equinox and/or optax.
Install with:  pip install maddening[surrogates]
```

```
ImportError: TerminalRenderer requires 'rich'.
Install with:  pip install maddening[terminal]
```

Follow the suggested command to install the missing extra.
