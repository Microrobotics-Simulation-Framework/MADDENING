#!/usr/bin/env python3
"""Shard a stencil node across devices -- here four emulated CPU devices.

``ShardedStencilNode`` splits a stencil node's grid into equal slabs, one
per device of a JAX mesh, and exchanges halo cells between neighbours on
every step, so the sharded node computes what the unsharded one does.
This example wraps a ``HeatNode`` (a 1-D rod with fixed end temperatures)
and checks exactly that:

1. ``print_graph()`` shows the wrapper -- ``ShardedStencilNode(HeatNode)``,
   the mesh, which state axis is split -- and the state as
   ``float32[N@devices]``;
2. ``memory_estimate()`` reports a quarter of the rod per device, against
   the whole rod on one device unsharded;
3. after the same number of steps the two rods agree (bit for bit here,
   on the same machine), and the sharded result really is spread over the
   four devices, ``N/4`` cells on each;
4. a grid that does not divide by the device count is refused when the
   wrapper is built, naming both numbers.

**This is CPU emulation, not GPU performance.**  The four devices are
virtual CPU devices created with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``, all on this one
machine and sharing its cores, so the example says nothing about speed;
it shows the layout and the correctness, which carry over to real
devices.  On a multi-GPU machine, build the mesh over the GPUs instead.

JAX fixes its device count when its backend starts, so this script sets
``XLA_FLAGS`` itself, before JAX is imported, replacing any device count
already in the variable and keeping every other flag.  Run it as its own
process (``python -m ...``), not from a session that has already used
JAX.

Usage
-----
    python -m maddening.examples.advanced.sharding_demo
    python -m maddening.examples.advanced.sharding_demo --n-cells 64 --steps 20
"""

from __future__ import annotations

import os
import sys

DEVICES = 4
_FLAG = "--xla_force_host_platform_device_count"


def _pin_device_count(xla_flags: str, count: int) -> str:
    """*xla_flags* with the host device count forced to *count*: any
    inherited setting of the flag (``--flag=N`` or ``--flag N``) is
    dropped, every other flag is kept in order."""
    tokens, kept, i = xla_flags.split(), [], 0
    while i < len(tokens):
        if tokens[i] == _FLAG:
            i += 2
        elif tokens[i].startswith(_FLAG + "="):
            i += 1
        else:
            kept.append(tokens[i])
            i += 1
    return " ".join([*kept, f"{_FLAG}={count}"])


# Before anything imports JAX.
os.environ["XLA_FLAGS"] = _pin_device_count(os.environ.get("XLA_FLAGS", ""), DEVICES)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: E402
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes.heat import HeatNode  # noqa: E402

DX = 1e-3                     # cell size [m], the same at every --n-cells
ALPHA = 1e-4                  # thermal diffusivity [m^2/s]
DT = 0.2 * DX ** 2 / ALPHA    # Fourier number 0.2: inside the stencil's limit


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def build(n_cells: int, *, sharded: bool) -> GraphManager:
    """A rod held at 100 at its left end and 0 at its right, optionally
    wrapped in a ``ShardedStencilNode`` over a mesh of every device."""
    rod = HeatNode("rod", DT, n_cells=n_cells, length=n_cells * DX,
                   thermal_diffusivity=ALPHA, initial_temperature=20.0)
    node = (ShardedStencilNode(rod, create_device_mesh(shape=(DEVICES,)),
                               axis_map={"devices": 0})
            if sharded else rod)
    gm = GraphManager()
    gm.add_node(node)
    gm.add_external_input("rod", "left_temperature")
    gm.add_external_input("rod", "right_temperature")
    gm.compile()
    return gm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cells", type=int, default=4096,
                        help=f"Cells in the rod; a multiple of {DEVICES} (default 4096)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps run by each rod (default 500)")
    args = parser.parse_args(argv)
    if args.n_cells % DEVICES:
        parser.error(f"--n-cells must be a multiple of {DEVICES}")

    if jax.device_count() != DEVICES:
        print(f"JAX started with {jax.device_count()} devices, not {DEVICES}: it was "
              f"initialised before this script set XLA_FLAGS.  Run it as its own "
              f"process: python -m maddening.examples.advanced.sharding_demo")
        return 1
    print(f"JAX devices: {jax.devices()}  (virtual CPU devices: emulation, "
          f"not a performance measurement)")

    section(f"1. A {args.n_cells}-cell HeatNode wrapped in ShardedStencilNode")
    sharded = build(args.n_cells, sharded=True)
    plain = build(args.n_cells, sharded=False)
    sharded.print_graph()
    assert "ShardedStencilNode(HeatNode)" in sharded.format_graph()

    section("2. State memory per device")
    sharded.print_memory_estimate()
    (s_row,) = [r for r in sharded.memory_estimate() if r["node"] == "rod"]
    (p_row,) = [r for r in plain.memory_estimate() if r["node"] == "rod"]
    print(f"\n  unsharded: {p_row['per_device_bytes']} bytes on {p_row['devices']} device; "
          f"sharded: {s_row['per_device_bytes']} bytes on each of {s_row['devices']}")
    assert s_row["bytes"] == p_row["bytes"] == 4 * args.n_cells        # float32
    assert s_row["devices"] == DEVICES and s_row["per_device_bytes"] == s_row["bytes"] // DEVICES
    assert p_row["devices"] == 1 and p_row["per_device_bytes"] == p_row["bytes"]

    section(f"3. {args.steps} steps of each, from the same start")
    ext = {"rod": {"left_temperature": jnp.float32(100.0),
                   "right_temperature": jnp.float32(0.0)}}
    t_sharded = sharded.run_scan(args.steps, external_inputs=ext)["rod"]["temperature"]
    t_plain = plain.run_scan(args.steps, external_inputs=ext)["rod"]["temperature"]
    a, b = np.asarray(t_sharded), np.asarray(t_plain)
    diff = float(np.abs(a - b).max())
    print(f"  temperature near the ends: [{a[0]:.3f}, {a[1]:.3f}, ..., {a[-2]:.3f}, {a[-1]:.3f}]")
    print(f"  largest difference sharded vs unsharded: {diff:.1e}"
          + ("  (bitwise identical)" if diff == 0.0 else ""))
    assert a[0] > 20.0 > a[-1], "the held ends should have pulled the rod's ends apart"
    assert diff <= 1e-5 * float(np.abs(b).max()), diff
    shards = sorted(t_sharded.addressable_shards, key=lambda s: s.index[0].start)
    print(f"  the result's sharding: {t_sharded.sharding.spec}; cells per device: "
          f"{[s.data.shape[0] for s in shards]}")
    assert len({s.device for s in shards}) == DEVICES
    assert all(s.data.shape == (args.n_cells // DEVICES,) for s in shards)
    print("  Halo exchange gives each slab its neighbours' edge cells, so the")
    print("  stencil sees the same values it would on one device.")

    section("4. A grid the devices cannot share equally is refused")
    try:
        ShardedStencilNode(HeatNode("odd", DT, n_cells=args.n_cells + 1,
                                    length=(args.n_cells + 1) * DX,
                                    thermal_diffusivity=ALPHA),
                           create_device_mesh(shape=(DEVICES,)), axis_map={"devices": 0})
    except ValueError as exc:
        print(f"  ValueError: {str(exc)[:200]}")
    else:
        raise AssertionError(f"{args.n_cells + 1} cells over {DEVICES} devices was accepted")

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
