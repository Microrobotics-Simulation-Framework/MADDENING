#!/usr/bin/env python3
"""v0.2 #5 demo: subscribe to LBM velocity only over the binary WebSocket.

Spawns an in-process SimulationServer with a small Lattice-Boltzmann
node, opens the /ws/state/binary WebSocket as a client, sends a
``subscribe`` message that asks for just the velocity field (skipping
the 19 f-distribution arrays), and decodes a handful of frames to
verify the bandwidth math the brief calls out::

    full state    ≈ velocity (3·N) + 19·f_i (N)   = 22·N floats
    subscribed    ≈ velocity (3·N)                =  3·N floats
    reduction     ≈ 1 - 3/22                      ≈ 86 % uncompressed
                                                  ≈ 95-99 % with zstd

Compares uncompressed vs zstd vs zstd+xor wire sizes at the end.

Usage:
    python 08_subscribe_lbm_velocity.py
    python 08_subscribe_lbm_velocity.py --n-cells 32 --n-frames 30
    python 08_subscribe_lbm_velocity.py --compression zstd+xor

No cloud account, no GPU required.  Runs entirely locally as a
demonstration of the encoder + transport contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
import threading
from typing import Any

import jax.numpy as jnp
import numpy as np
import uvicorn
from websockets.asyncio.client import connect

from maddening.api.server import SimulationServer
from maddening.api.binary_encoder import BinaryStateEncoder
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode


class _ToyLBMNode(SimulationNode):
    """Tiny stand-in for an LBMNode — emits velocity + 19 f-distributions.

    Velocity dynamics: a constant drift in +x.  f-distributions are
    held at a sinusoidal equilibrium so the wire payload looks like a
    real LBM frame for the encoder benchmark.
    """

    def __init__(self, name: str, timestep: float, n_cells: int):
        super().__init__(name, timestep, n_cells=n_cells)
        self._n = n_cells

    def initial_state(self) -> dict:
        N = self._n
        state = {
            "velocity": jnp.zeros((N, 3), dtype=jnp.float32),
            **{f"f{i}": jnp.ones(N, dtype=jnp.float32) * 0.05
               for i in range(19)},
        }
        return state

    def update(self, state, boundary_inputs, dt):
        # Velocity drifts in +x; f-distributions stay roughly constant.
        new_v = state["velocity"].at[:, 0].add(0.001 * dt)
        out = {"velocity": new_v}
        for i in range(19):
            out[f"f{i}"] = state[f"f{i}"]
        return out


def _build_server(n_cells: int) -> tuple[SimulationServer, int]:
    gm = GraphManager()
    gm.add_node(_ToyLBMNode("lbm", timestep=0.01, n_cells=n_cells))
    gm.compile()
    server = SimulationServer(node_registry={}, graph_manager=gm)
    port = _pick_free_port()
    return server, port


def _pick_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _client(port: int, *, compression: str, n_frames: int) -> dict[str, Any]:
    """Connect, subscribe, decode N frames, return stats."""
    url = f"ws://127.0.0.1:{port}/ws/state/binary"
    async with connect(url) as ws:
        # Receive the initial schema (encoder built with full state).
        full_schema = json.loads(await ws.recv())
        full_frame_bytes = full_schema["frame_bytes"]

        # Subscribe to velocity only + (optional) compression.
        await ws.send(json.dumps({
            "type": "subscribe",
            "fields": {"lbm": ["velocity"]},
            "compression": compression,
        }))

        # The server *eventually* re-sends a schema reflecting the new
        # subscription, but in-flight binary frames using the old
        # encoder may arrive first.  Drop those until we see the
        # text-frame schema.
        new_schema = None
        while new_schema is None:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                continue  # stale binary frame
            try:
                new_schema = json.loads(msg)
            except json.JSONDecodeError:
                continue
        assert new_schema["type"] == "schema"
        sub_frame_bytes_uncompressed = new_schema["frame_bytes"]

        # Pull N frames; record observed wire sizes (variable when
        # compression is on).  Skip occasional text frames (e.g. a
        # second schema if the server reconstructed twice).
        observed = []
        while len(observed) < n_frames:
            frame = await ws.recv()
            if not isinstance(frame, (bytes, bytearray)):
                continue
            observed.append(len(frame))
            # Sanity: sim_time header is always plaintext.
            t = struct.unpack_from("d", frame, 0)[0]
            assert t >= 0.0

    return {
        "full_frame_bytes": full_frame_bytes,
        "subset_uncompressed_bytes": sub_frame_bytes_uncompressed,
        "observed_min": min(observed),
        "observed_max": max(observed),
        "observed_mean": int(sum(observed) / len(observed)),
        "compression": compression,
    }


def _run_server(server: SimulationServer, port: int, stop: threading.Event) -> None:
    config = uvicorn.Config(
        server.create_app(), host="127.0.0.1", port=port,
        log_level="warning",
    )
    s = uvicorn.Server(config)
    # uvicorn runs an asyncio event loop in this thread.
    s.run()


async def _drive_steps(server: SimulationServer, n: int) -> None:
    """Step the simulation N times so the binary stream has frames."""
    for _ in range(n):
        server.gm.step()
        await asyncio.sleep(0.005)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cells", type=int, default=16,
                        help="Cells per direction (state size = n³)")
    parser.add_argument("--n-frames", type=int, default=10,
                        help="Frames to decode after subscribing")
    parser.add_argument(
        "--compression", choices=("none", "zstd", "zstd+xor"),
        default="zstd",
        help="Compression mode requested via the subscribe message",
    )
    args = parser.parse_args()

    n_cells = args.n_cells ** 3
    server, port = _build_server(n_cells)

    # Quick offline reference: what would the encoder emit standalone?
    sample = server.gm._state.copy()
    sample.pop("_meta", None)
    raw_enc = BinaryStateEncoder(sample)
    sub_enc = BinaryStateEncoder(sample, fields={"lbm": ["velocity"]})
    print(f"State size: {n_cells:,} cells (n={args.n_cells}³)")
    print(f"  Full uncompressed frame:     {raw_enc.frame_bytes:>10,} B")
    print(f"  Velocity-only uncompressed:  {sub_enc.frame_bytes:>10,} B "
          f"({(1 - sub_enc.frame_bytes/raw_enc.frame_bytes) * 100:.1f}% saved)")

    # Spawn the server in a daemon thread; drive the simulation forward
    # from the main loop so /ws/state/binary has frames to send.
    stop = threading.Event()
    t = threading.Thread(target=_run_server, args=(server, port, stop), daemon=True)
    t.start()

    async def runner():
        await asyncio.sleep(0.5)  # give uvicorn time to bind
        # Step in a background task so the WS client gets fresh frames.
        stepper = asyncio.create_task(_drive_steps(server, args.n_frames * 4))
        stats = await _client(port, compression=args.compression, n_frames=args.n_frames)
        await stepper
        return stats

    stats = asyncio.run(runner())
    print()
    print(f"Subscribed to fields={{lbm:[velocity]}} compression={args.compression}")
    print(f"  Schema-reported uncompressed: {stats['subset_uncompressed_bytes']:>10,} B")
    print(f"  Observed wire size mean:      {stats['observed_mean']:>10,} B")
    print(f"  Observed wire size min/max:   {stats['observed_min']:>10,} / "
          f"{stats['observed_max']:,} B")
    full_b = stats["full_frame_bytes"]
    mean_b = stats["observed_mean"]
    reduction = 1.0 - mean_b / full_b
    print(f"  End-to-end reduction:         {reduction * 100:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
