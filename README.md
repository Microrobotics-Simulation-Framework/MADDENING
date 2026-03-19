# MADDENING

Modular Automatic Differentiation and Data-Enhanced Neural-network INteracting Graph (MADDENING). It is designed to work as a stand-alone framework, but was specifically designed to be the backbone of MIME (MIcrorobotics Multiphysics Engine).

> **Regulatory disclaimer**: MADDENING is research software. It is not a medical device as defined by EU MDR (EU 2017/745) and is not intended for clinical use. When used in a regulated product, MADDENING is classified as SOUP (Software of Unknown Provenance) under IEC 62304. See `docs/regulatory/` for details.

## What MADDENING is intended to do, and an explanation of the Acronym.
MADDENING is a framework designed for soft real-time high-fidelity simulation of complex systems. It is designed to run on HPC server environments. 
The framework works by managing a graph (hence the Graph part of the acronym), where each node focuses on simulating a specific aspect of the overall simulation (hence the Modular part of the acronym).

The edges of the graph represent the coupling between simulations.

Nodes are written in JAX (hence the Automatic Differentiation part of the acronym), and can be reduced to Physics Informed Neural Networks (PINNs) for real time simulation. The training of PINNs can also be augmented by real-world data (hence the Data-Enhanced Neural Network part of the acronym).

## Installation

```bash
pip install maddening                    # CPU (base)
pip install maddening[cuda12]            # GPU with CUDA 12
pip install maddening[cuda12,viz]        # GPU + matplotlib plots
pip install maddening[server,cuda12]     # GPU simulation server (Docker)
pip install maddening[runpod]            # Cloud deploy to RunPod
```

Install only what you need. Common extras:

| Extra | What it adds |
|-------|-------------|
| `cuda12` / `tpu` | GPU / TPU acceleration |
| `viz` | Matplotlib renderers |
| `terminal` | Rich terminal renderer |
| `api` | FastAPI HTTP/WS server |
| `network` | ZeroMQ remote transport |
| `surrogates` | Neural surrogate training |
| `runpod` / `lambda` / `aws` / `gcp` | Cloud deploy via SkyPilot |
| `server` | Bundle: FastAPI + ZMQ + rich + matplotlib |
| `cloud` | All supported cloud providers |
| `all` | Everything |

Works with both pip and [uv](https://docs.astral.sh/uv/). See [docs/user_guide/installation.md](docs/user_guide/installation.md) for the full guide.
