#!/usr/bin/env python
"""Capacity test for the multi-GPU hardware session: a sharded run larger than one card.

Runs *on the machine that has the GPUs* (or, with ``--dry-run``, on CPU
virtual devices), after the checklist of ``run_pod.py`` has closed.  It
is **not a gate**: it has its own output directory, its own JSON files
and its own schema, ``run_pod.py --summarise`` reads none of them, and
nothing it writes changes the checklist's verdict.  Like the runner it
never talks to a cloud provider; the only processes it starts are itself
(one per rung) and ``git rev-parse HEAD``.

The question.  Every goal of ``run_pod.py`` compares a sharded run with
the unsharded node on one device, so none of them can be larger than one
card.  This script asks what they cannot: does a sharded simulation that
does *not* fit on one card run, and stay right, as the cards fill up?

The problem.  A periodic D3Q19 lattice -- the library's ``LBMNode``
wrapped in ``ShardedStencilNode`` -- on the pencil mesh (``--mesh 2x2``,
spatial axes 0 and 1 split) or a 1-D mesh (``--mesh 4x1``, axis 0 split
over all four devices).

The reference, with nothing unsharded at full size: a periodic tiling.
The start is a small tile (``--tile``, 16 x 24 x 20 cells) repeated
``K = (kx, ky, kz)`` times.  By translation symmetry the large periodic
run equals the tile's own periodic run, tiled, at every step; the tile
is run unsharded on one device.  Along every axis the mesh splits,
``K`` is odd and shares no factor with the devices there, and the tile's
extent is a multiple of them and at least twice them: every shard then
starts at a different phase of the tile, and a halo taken from another
shard, from the other side or from the shard itself differs from the
right one (:func:`tiling_problem` gives the arithmetic).  A ``K`` or a
tile that breaks this is refused.

The whole field never exists on one device or on the host.  It is built
block by block, each device gathering its own block from the tile
(:func:`build_state`), and compared the same way, a few planes at a time
(:func:`compare_with_tiling`; ``--check-chunk-mb`` bounds what the
comparison itself holds).  Per rung: every state field against the tiled
reference (the limit is the runner's forward limit, ``LIMITS["forward"]``
of ``run_pod.py``), the total mass against ``prod(K)`` times the tile's,
every value finite.

Memory.  ``jax.devices()[i].memory_stats()`` after the build, after the
first (compiled) step, after the last step and after the comparison.  A
rung's fill is the largest ``peak_bytes_in_use`` over ``--device-memory-gb``
(GiB), over the devices.  Where the backend has no statistics -- CPU --
the fill is recorded as *not measured* (a check not run, never a pass).

The ramp.  ``--ramp 0.25 0.5 0.75 0.85 0.9`` gives target fills; each
rung is its own process under a time box (``--rung-timeout-s``).  The
first rung's size comes from an a-priori estimate (the state's bytes per
cell times ``APRIORI_STEP_MULTIPLE``), every later rung's from the bytes
per cell the rung before it measured.  ``--cells`` gives explicit sizes
instead (the way to run on CPU), ``--k`` one exact tiling.  A rung ends
as exactly one of::

    passed          every check that ran passed
    check failed    a number is wrong: the one outcome that is a defect
    out of memory   the device's allocator refused (resource exhausted),
                    or the process was killed (status 137, SIGKILL)
    timed out       the time box ran out
    crashed         anything else; the record keeps the end of its stderr

or never starts: ``refused by the host guard`` (the host would have to
hold more than half of its available memory).  The ramp stops at the
first rung that did not pass; the summary names the ceiling -- the
largest size that passed, its fill, and what stopped the next one.

The soak.  ``--soak-minutes M`` then repeats build, steps and comparison
at ``--soak-at`` (0.75) of the ceiling's cells until the time is up,
reading the memory after every block; growth from block to block is the
finding.

Exit status::

    0   every rung that ran to its checks passed them (an out-of-memory
        or a time-out above a passing rung is a result, and is recorded)
    1   a check failed, in a rung or in the soak
    2   an option was refused (a tiling the rules above do not allow, a
        size smaller than one tile, an ``--out`` holding goal files)
    3   ``--summarise`` only: a record cannot be read, or its recorded
        verdict disagrees with its own values
    5   this script itself raised
    6   no ceiling: the first rung did not pass (out of memory, timed
        out, crashed, refused by the host guard), or ``--summarise``
        found no rung record

Files, under ``--out``: ``capacity_rung_NN.json`` (one per rung, with
``capacity_rung_NN.stderr.log``), ``capacity_soak.json`` and
``capacity_summary.json``.  ``--summarise DIR`` prints the table again
and re-derives every verdict from the recorded values, not from the
recorded flags (no JAX needed).

Examples (``README.md`` next to this file has the session order)::

    python benchmarks/multigpu/run_capacity.py --dry-run --cells 8000 30000 100000 \\
        --out /tmp/capacity-dry
    python benchmarks/multigpu/run_capacity.py --device-memory-gb 24 \\
        --ramp 0.25 0.5 0.75 0.85 0.9 --soak-minutes 5 --out results/capacity
    python benchmarks/multigpu/run_capacity.py --summarise results/capacity
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np


def _pre_import_setup(argv: list[str]) -> None:
    """Environment that must be in place before ``import jax``: ``--dry-run``
    with no accelerator backend requested pins the CPU backend and gives it
    four virtual devices, exactly as ``run_pod.py`` does (the literal
    ``--dry-run`` only, which is why the parser takes no abbreviations)."""
    if "--dry-run" not in argv:
        return
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platforms not in ("", "cpu"):
        return
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=4").strip()


_pre_import_setup(sys.argv)


def _load_runner():
    """``run_pod.py`` next to this file, for the names read off it below and
    nothing else.  Importing it needs no JAX; its own pre-import setup does
    what the one above just did, for the same literal ``--dry-run``."""
    spec = importlib.util.spec_from_file_location(
        "_run_pod_read_by_run_capacity", Path(__file__).resolve().with_name("run_pod.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = _load_runner()
#: The limit of every comparison here: the runner's limit for a sharded
#: rollout against the unsharded node, for the same reason (float32
#: round-off between two correct programs that XLA fuses differently).
LIMIT = _RUNNER.LIMITS["forward"]
#: The runner's goal names: no file of this script may be named for one.
RUNNER_GOALS = tuple(_RUNNER.ALL_GOALS)
check = _RUNNER.check
check_that = _RUNNER.check_that
check_not_run = _RUNNER.check_not_run
expected_pass = _RUNNER.expected_pass
_git_commit = _RUNNER._git_commit                                    # noqa: SLF001

CAPACITY_SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_REFUSED = 2
EXIT_RECORD_INVALID = 3
EXIT_CRASHED = 5
EXIT_NO_CEILING = 6
#: Statuses of a rung's own process only (the ramp reads them, and never
#: exits with them): the allocator refused, and the host guard refused.
EXIT_RUNG_OUT_OF_MEMORY = 7
EXIT_RUNG_HOST_GUARD = 8

PASSED = "passed"
CHECK_FAILED = "check failed"
OUT_OF_MEMORY = "out of memory"
TIMED_OUT = "timed out"
CRASHED = "crashed"
HOST_GUARD = "refused by the host guard"
OUTCOMES = (PASSED, CHECK_FAILED, OUT_OF_MEMORY, TIMED_OUT, CRASHED, HOST_GUARD)

RUNG_KIND = "capacity rung"
SOAK_KIND = "capacity soak"
SUMMARY_KIND = "capacity summary"
SUMMARY_FILE = "capacity_summary.json"
SOAK_FILE = "capacity_soak.json"

LATTICE = "D3Q19"
LATTICE_Q, LATTICE_D = 19, 3
VISCOSITY = 0.1                 # lattice units, as the runner's lattice case
STATE_FIELDS = ("f", "density", "velocity", "pressure", "wall_mask")
#: Bytes one cell of the state holds: 19 float32 populations, a density, a
#: pressure, three velocity components, and the wall mask's byte.
STATE_BYTES_PER_CELL = 4 * LATTICE_Q + 4 + 4 + 4 * LATTICE_D + 1
#: The first rung's guess of the peak bytes per cell of a step, as a
#: multiple of the state's: the state going in, the state coming out, and
#: the padded, collided, streamed and bounced copies of the populations in
#: between.  An estimate that only has to be safe at the first target (at
#: 0.25 it may be 3.6 times too low before the rung runs out of memory);
#: every later rung is sized from what the one before it measured.
APRIORI_STEP_MULTIPLE = 6.0
DEFAULT_TILE = (16, 24, 20)
DEFAULT_RAMP = (0.25, 0.5, 0.75, 0.85, 0.9)
#: A target above this leaves the allocator no room for anything it holds
#: beside the run (its own bookkeeping, the collectives' buffers).
MAX_TARGET_FILL = 0.95
GIB = 1024 ** 3

#: The tile's start: an equilibrium of a smooth density and velocity field
#: with the tile's own period, each population rippled a little so no two
#: of them carry the same pattern.  The velocity is large (several 1e-2)
#: and the tile long (it decays as viscosity times the squared wavenumber)
#: on purpose: the velocity is a small difference of populations of about
#: 0.05, and the comparison is relative to its largest value.  At 6e-3 the
#: float32 rounding of the populations alone put the sharded and the
#: unsharded velocity 1.4e-5 apart, over the limit, after three steps; on
#: an 8 x 12 x 10 tile at 0.04 they were 2e-6 apart after 10 steps and
#: 8e-6 after 40, the velocity having decayed to a quarter (CPU, jaxlib
#: 0.11.0).
DENSITY_AMPLITUDE = 0.05
VELOCITY_AMPLITUDE = 0.1
POPULATION_RIPPLE = 0.02
_PATTERN_MAX = 2.5              # the largest |tile_pattern|: 1 + 0.6 + 0.5 + 0.4
#: Two planes of the tile along a split axis must differ by at least this
#: many times the comparison's limit (relative to the largest population),
#: so that a halo holding the wrong plane is far outside it.
PLANES_APART = 100.0

#: The allocator's environment: recorded as found, then given these
#: defaults where unset.  Preallocation off is the runner's choice (the
#: pool grows as needed).  The fraction is documented by JAX for a
#: preallocated pool, 0.75 of the card by default; its allocator is
#: understood to take the same number as the upper limit of a pool that
#: grows, which would end the ramp's last targets at 0.75.  That reading
#: has not been measured (no accelerator has run this): every rung records
#: each device's ``bytes_limit``, which is the measurement.
ALLOCATOR_VARIABLES = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                       "XLA_PYTHON_CLIENT_ALLOCATOR")
ALLOCATOR_DEFAULTS = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                      "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
_ALLOCATOR_FOUND = {name: os.environ.get(name) for name in ALLOCATOR_VARIABLES}

#: What a memory reading keeps of ``Device.memory_stats()``, where present.
MEMORY_KEYS = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "largest_alloc_size",
               "bytes_reserved", "peak_bytes_reserved", "largest_free_block_bytes",
               "pool_bytes", "peak_pool_bytes")
#: The four readings of a rung, in order.
READINGS = ("after the build", "after the first compiled step", "after the last step",
            "after the check")

#: An allocator's refusal, in an exception's text or at the end of stderr.
_OUT_OF_MEMORY = re.compile(r"RESOURCE_EXHAUSTED|out of memory|OUT_OF_MEMORY|failed to allocate",
                            re.IGNORECASE)
#: The statuses of a process the kernel killed: ``subprocess`` reports the
#: signal as -9, a shell as 137.
_KILLED = (-9, 137)
MESH_AXES = ("shard_0", "shard_1")
_UNREADABLE = (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError)


# ---------------------------------------------------------------------------
# The tiling: which tiles and counts the reference can be trusted on
# ---------------------------------------------------------------------------


def parse_mesh(label: str) -> tuple[int, int]:
    """``"AxB"`` -> ``(A, B)``: the devices along spatial axis 0 and along
    axis 1.  ``B == 1`` is a 1-D mesh splitting axis 0 only."""
    m = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", label)
    if m is None:
        raise ValueError(f"--mesh {label!r}: expected AxB, the devices along spatial axis 0 "
                         "and along axis 1 (2x2 for the pencil, 4x1 for the 1-D mesh)")
    a, b = int(m.group(1)), int(m.group(2))
    if a < 2:
        raise ValueError(f"--mesh {label}: spatial axis 0 must be split over at least two "
                         "devices (with one there is no exchange along it)")
    return a, b


def devices_per_axis(mesh_shape: tuple[int, int]) -> tuple[int, int, int]:
    """The devices each spatial axis is split over; axis 2 never is."""
    return mesh_shape[0], mesh_shape[1], 1


def _cells(tile, k) -> int:
    return math.prod(int(t) * int(n) for t, n in zip(tile, k))


def grid_shape(tile, k) -> tuple[int, int, int]:
    nx, ny, nz = (int(t) * int(n) for t, n in zip(tile, k))
    return nx, ny, nz


def tiling_problem(tile, k, per_axis) -> str | None:
    """Why the tiled reference cannot be trusted for this tile, these counts
    and these devices per axis, or ``None``.

    Along an axis split over ``D`` devices, with a tile of extent ``t =
    D * m`` repeated ``K`` times, shard ``d`` starts at tile row ``(d * K *
    m) mod t``.  The right halo of shard ``d`` must hold the first row of
    shard ``d + 1``.  It is told from:

    * the first row of another shard ``d + j`` only when ``(j - 1) * K`` is
      not a multiple of ``D`` -- for every ``j``, when ``K`` shares no
      factor with ``D``;
    * the shard's own first row (a halo that wraps inside the shard) under
      the same condition;
    * the last row of the shard on the *other* side (a halo from the wrong
      neighbour, the right edge delivered) only when ``2 * K`` is not a
      multiple of ``D``: on four devices ``K = 2`` shares a factor with
      ``D`` *and* makes the two neighbours alike, which is why ``K`` must
      be odd;
    * the last row of the left neighbour (the two halos swapped) only when
      ``m >= 2``: with one tile row per device and ``K = D - 1`` the two
      rows are the same row of the tile.

    Hence: ``t`` a multiple of ``D`` and at least ``2 * D``, ``K`` odd and
    coprime with ``D``.  And no two axes of the grid may have the same
    extent, or a quantity read off the wrong axis is the right number.
    """
    if len(tile) != 3 or len(k) != 3:
        return "the tile and the counts are three numbers each, one per spatial axis"
    for axis, (t, n, d) in enumerate(zip(tile, k, per_axis)):
        if t < 1 or n < 1:
            return f"the tile's extent and the count along axis {axis} must be at least 1"
        if d == 1:
            continue
        if t % d:
            return (f"the tile's extent along axis {axis} is {t}, which the {d} devices there "
                    "do not divide: the shards would not each start on a whole row of the "
                    f"tile at a different phase (use a multiple of {d}, at least {2 * d})")
        if t < 2 * d:
            return (f"the tile's extent along axis {axis} is {t}, less than twice the {d} "
                    "devices there: a shard's two halo rows could be the same row of the "
                    "tile, and a halo from the other side would not show")
        if n % 2 == 0 or math.gcd(n, d) != 1:
            return (f"the tile count along axis {axis} is {n}: it must be odd and share no "
                    f"factor with the {d} devices there, or two shards hold the same rows of "
                    "the tile and a halo taken from the wrong one would not show")
    nx, ny, nz = grid_shape(tile, k)
    if len({nx, ny, nz}) < 3:
        return (f"the grid would be {nx} x {ny} x {nz}: two axes of the same extent, on which "
                "a quantity read off the wrong axis is the right number")
    return None


def _counts_around(target: float, d: int) -> list[int]:
    """The counts an axis split over ``d`` devices may take on either side
    of ``target``: the largest allowed one not above it (the smallest
    allowed one where there is none) and the first allowed one above."""
    def allowed(n: int) -> bool:
        return d == 1 or (n % 2 == 1 and math.gcd(n, d) == 1)

    below = max(1, int(target))
    while below > 1 and not allowed(below):
        below -= 1
    above = max(1, int(target)) + 1
    while not allowed(above):
        above += 1
    return [below, above]


#: A grid may give up this share of the largest allowed size to be closer
#: to a cube.
_CUBE_SLACK = 0.05


def tiling_for_cells(cells: int, tile, per_axis) -> tuple[int, int, int]:
    """The counts ``K`` of an allowed grid of at most ``cells`` cells, about
    as long along every axis.

    The counts along axes 0 and 1 are the allowed ones on either side of a
    cube's; the count along axis 2, which no mesh splits and so may be any
    number, takes up the rest.  Of the grids within ``_CUBE_SLACK`` of the
    largest of those, the one nearest a cube.  Raises ``ValueError`` when
    ``cells`` holds not even one tile.
    """
    side = float(cells) ** (1.0 / 3.0)
    found = []
    for kx in _counts_around(side / tile[0], per_axis[0]):
        for ky in _counts_around(side / tile[1], per_axis[1]):
            kz = cells // (kx * tile[0] * ky * tile[1] * tile[2])
            while kz >= 1 and tiling_problem(tile, (kx, ky, kz), per_axis) is not None:
                kz -= 1
            if kz < 1:
                continue
            shape = grid_shape(tile, (kx, ky, kz))
            found.append((_cells(tile, (kx, ky, kz)), max(shape) / min(shape), (kx, ky, kz)))
    if not found:
        raise ValueError(
            f"{cells} cells hold no allowed tiling of the {tile[0]} x {tile[1]} x {tile[2]} "
            f"tile ({math.prod(tile)} cells) on this mesh")
    most = max(n for n, _, _ in found)
    near = [entry for entry in found if entry[0] >= (1.0 - _CUBE_SLACK) * most]
    return min(near, key=lambda entry: (entry[1], -entry[0]))[2]


# ---------------------------------------------------------------------------
# The host: what is free, and what a rung may take of it
# ---------------------------------------------------------------------------


def _status_bytes(key: str, path: str = "/proc/self/status") -> int | None:
    """A ``kB`` line of a ``/proc`` status file, in bytes."""
    try:
        with open(path, encoding="ascii") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def vm_hwm_bytes() -> int | None:
    """This process's own peak resident memory (``VmHWM``)."""
    return _status_bytes("VmHWM")


def vm_rss_bytes() -> int | None:
    return _status_bytes("VmRSS")


def _cgroup_headroom_bytes() -> int | None:
    """What the memory limits of this process's control groups leave: the
    least ``memory.max - memory.current`` from its own group up to the root
    (cgroup v2).  In a container ``MemAvailable`` is the machine's, and the
    container's limit is the lower one."""
    try:
        with open("/proc/self/cgroup", encoding="ascii") as f:
            lines = [line.strip() for line in f if line.startswith("0::")]
        if not lines:
            return None
        group = Path("/sys/fs/cgroup") / lines[0][3:].lstrip("/")
    except OSError:
        return None
    least = None
    while True:
        try:
            limit = (group / "memory.max").read_text(encoding="ascii").strip()
            if limit != "max":
                used = int((group / "memory.current").read_text(encoding="ascii").strip())
                room = max(int(limit) - used, 0)
                least = room if least is None else min(least, room)
        except (OSError, ValueError):
            pass
        if group == Path("/sys/fs/cgroup") or group.parent == group:
            return least
        group = group.parent


def host_available_bytes() -> tuple[int | None, str]:
    """``(bytes, where it was read)`` the host has for a new process: the
    lesser of ``MemAvailable`` and the control groups' headroom; ``None``
    where neither can be read."""
    available = _status_bytes("MemAvailable", "/proc/meminfo")
    room = _cgroup_headroom_bytes()
    if available is None and room is None:
        return None, "not readable (/proc/meminfo, /sys/fs/cgroup)"
    if room is not None and (available is None or room < available):
        return room, "the control group's memory.max less memory.current"
    return available, "MemAvailable of /proc/meminfo"


def host_bytes_needed(cells: int, n_devices: int, *, devices_are_host: bool) -> int:
    """An estimate of what the host must hold for a rung of ``cells`` cells.

    On an accelerator: one shard's block of the state and the tile.  The
    build and the comparison keep every block on its device, so this is
    an allowance for one block staged through the host, not a measurement.
    On CPU virtual devices the devices *are* the host's memory: the whole
    run, at the a-priori peak per cell."""
    tile_bytes = 16 * 1024 * 1024       # the tile, its reference, the per-chunk sums: a bound
    if devices_are_host:
        return int(cells * STATE_BYTES_PER_CELL * APRIORI_STEP_MULTIPLE) + tile_bytes
    return -(-cells // n_devices) * STATE_BYTES_PER_CELL + tile_bytes


def host_guard(cells: int, n_devices: int, *, devices_are_host: bool) -> dict:
    """Whether a rung may start: refused when the host would have to hold
    more than half of what it has available (a host out-of-memory can take
    the session's shell with it)."""
    needed = host_bytes_needed(cells, n_devices, devices_are_host=devices_are_host)
    available, source = host_available_bytes()
    out = {"needed_bytes": needed, "available_bytes": available, "available_read_from": source,
           "devices_are_host_memory": devices_are_host, "refused": False}
    if available is not None and needed > available // 2:
        out["refused"] = True
        out["why"] = (f"the host would have to hold about {needed / GIB:.2f} GiB, more than "
                      f"half of the {available / GIB:.2f} GiB it has available ({source})")
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def write_json(path: Path, doc: dict) -> None:
    """Write ``doc`` whole or not at all: a rung killed while writing must
    leave the record it wrote before, not half of a new one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def rung_file(index: int) -> str:
    return f"capacity_rung_{index:02d}.json"


def fill_of(readings, device_bytes: int | None) -> dict:
    """The fill of a rung from its memory readings: per device, the largest
    ``peak_bytes_in_use`` of any reading over ``device_bytes``; the rung's
    is the largest of those.

    ``{"fill", "peak_bytes", "by_device", "bytes_limit", "why_not"}``;
    ``fill`` is ``None`` -- not measured -- when any device of any reading
    has no peak, when there is no reading, or when the card's size was not
    given, and ``why_not`` says which.  The peak, never ``bytes_in_use``:
    between two readings a step's temporaries have come and gone.
    """
    out: dict[str, Any] = {"fill": None, "peak_bytes": None, "by_device": None,
                           "bytes_limit": None, "why_not": None}
    peaks: dict[str, int] = {}
    limits: dict[str, int] = {}
    for reading in readings or []:
        for row in reading.get("devices", []):
            peak = row.get("peak_bytes_in_use")
            if isinstance(peak, bool) or not isinstance(peak, int):
                out["why_not"] = (f"device {row.get('device')} has no peak_bytes_in_use "
                                  f"{reading.get('at')}: the backend keeps no memory statistics")
                return out
            peaks[row["device"]] = max(peaks.get(row["device"], 0), peak)
            limit = row.get("bytes_limit")
            if isinstance(limit, int) and not isinstance(limit, bool):
                limits[row["device"]] = limit
    if not peaks:
        out["why_not"] = "no memory reading was taken"
        return out
    out["peak_bytes"] = max(peaks.values())
    out["bytes_limit"] = min(limits.values()) if len(limits) == len(peaks) else None
    if not device_bytes:
        out["why_not"] = "--device-memory-gb was not given"
        return out
    out["by_device"] = {device: peak / device_bytes for device, peak in peaks.items()}
    out["fill"] = out["peak_bytes"] / device_bytes
    return out


def _ran(checks) -> list:
    return [c for c in checks if not c.get("not_run")]


def rederive(doc: dict) -> dict:
    """A rung's (or the soak's) verdict from its recorded values.

    ``{"outcome", "failed", "problems", "fill"}``: ``outcome`` is the
    recorded one unless the record's own checks say otherwise (a check
    whose value is over its limit is ``check failed`` whatever the record
    says); ``problems`` lists every place the record disagrees with itself
    -- a flag that is not what its value and limit give, an outcome its
    checks do not support, a fill that is not what its readings give, a
    size that is not its tiling's.  A record with a problem is not one
    this script wrote as it stands.
    """
    problems: list[str] = []
    failed: list[str] = []
    outcome = doc.get("outcome")
    if outcome not in OUTCOMES:
        problems.append(f"its outcome is {outcome!r}, not one of {OUTCOMES}")
    checks = doc.get("checks")
    if not isinstance(checks, list):
        checks = []
        if outcome in (PASSED, CHECK_FAILED):
            problems.append("it records no checks")
    blocks = doc.get("blocks") if doc.get("kind") == SOAK_KIND else None
    for block in blocks or []:
        checks = checks + list(block.get("checks") or [])
    for c in checks:
        if not isinstance(c, dict):
            problems.append("a check is not a record")
            continue
        if c.get("not_run"):
            continue
        derived = expected_pass(c)
        if derived is None:
            problems.append(f"check {c.get('name')!r} cannot be judged from its value and limit")
        elif derived != c.get("passed"):
            problems.append(f"check {c.get('name')!r} is recorded as "
                            f"{'passed' if c.get('passed') else 'failed'}, but its value "
                            f"{c.get('value')!r} against its limit {c.get('limit')!r} says "
                            f"{'passed' if derived else 'failed'}")
        if derived is False:
            failed.append(str(c.get("name")))
    ran = _ran([c for c in checks if isinstance(c, dict)])
    if outcome == PASSED and not ran:
        problems.append("it is recorded as passed with no check that ran")
    if outcome == PASSED and failed:
        problems.append(f"it is recorded as passed, but {len(failed)} check(s) fail by their own "
                        "values")
    if outcome == CHECK_FAILED and not failed:
        problems.append("it is recorded as a failed check, but no check fails by its own values")
    if failed:
        outcome = CHECK_FAILED
    tile, k, shape = doc.get("tile"), doc.get("k"), doc.get("shape")
    try:
        if not isinstance(shape, list) or doc.get("cells") != _cells(tile, k) \
                or shape != list(grid_shape(tile, k)):
            problems.append("its cells or shape are not its tile's times its counts")
    except _UNREADABLE:
        problems.append("its tile, counts, cells or shape cannot be read")
    recorded = doc.get("memory")
    memory: dict = recorded if isinstance(recorded, dict) else {}
    gb = memory.get("device_memory_gb")
    derived_fill = fill_of(memory.get("readings"), int(gb * GIB) if isinstance(gb, (int, float))
                           and not isinstance(gb, bool) and gb > 0 else None)["fill"]
    recorded_fill = memory.get("fill")
    if derived_fill is None or not isinstance(recorded_fill, float):
        agree = derived_fill is None and recorded_fill is None
    else:
        agree = math.isclose(recorded_fill, derived_fill, rel_tol=1e-9, abs_tol=0.0)
    if not agree:
        problems.append(f"its fill is recorded as {recorded_fill!r}, but its memory readings "
                        f"give {derived_fill!r}")
    return {"outcome": outcome, "failed": failed, "problems": problems, "fill": derived_fill}


def read_record(path: Path) -> tuple[dict | None, str | None]:
    """``(record, None)``, or ``(None, why it cannot be read)``."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(doc, dict):
        return None, f"it holds a JSON {type(doc).__name__}, not an object"
    if doc.get("kind") not in (RUNG_KIND, SOAK_KIND):
        return None, f"its kind is {doc.get('kind')!r}: not a record of this script"
    return doc, None


# ---------------------------------------------------------------------------
# The backend, loaded on first use (--summarise and the ramp need none)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    # What the names below are once :func:`_load_backend` has run.
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
    from maddening.nodes.lbm import LBMNode
else:
    jax = jnp = lax = NamedSharding = P = None
    create_device_mesh = ShardedStencilNode = LBMNode = None
#: ``LBMNode`` as this script wraps it (:func:`_make_lattice_class`).
Lattice: Any = None
_PROGRAMS: dict = {}
_TILE_STEPS: dict = {}


def _allocator_environment() -> dict:
    """The allocator's variables as found and as used.  The defaults are
    set only while JAX is still unimported: after that the allocator has
    read its environment, and a value set now would be recorded as used
    without having been."""
    if "jax" not in sys.modules:
        for name, value in ALLOCATOR_DEFAULTS.items():
            os.environ.setdefault(name, value)
        used = {name: os.environ.get(name) for name in ALLOCATOR_VARIABLES}
    else:
        used = dict(_ALLOCATOR_FOUND)
    return {"found": dict(_ALLOCATOR_FOUND), "used": used}


_ALLOCATOR: dict = {}


def _make_lattice_class(base):
    """``LBMNode`` with the two methods ``ShardedStencilNode`` calls while it
    is constructed answered without building the grid.

    The wrapper's constructor calls the node's ``initial_state()`` to read
    the field shapes, and ``state_fields()``, whose default builds the
    initial state again to list its keys: the whole grid, twice, on the
    default device, before a single block is placed.  A grid that fits the
    mesh but not one card cannot get past that.  So the names are given
    outright, and ``initial_state()`` raises: the wrapper documents that a
    node which cannot build its state at construction is not checked
    there (the tiling rules are this script's divisibility check), and the
    state is built block by block by :func:`build_state`.  The step --
    ``update_padded`` -- is the library's, untouched.
    """
    class Lattice(base):
        def state_fields(self):
            return list(STATE_FIELDS)

        def initial_state(self):
            raise RuntimeError(
                "this lattice's state is built block by block, one per device "
                "(run_capacity.build_state): its initial_state() would build the whole "
                "grid on one device")

    return Lattice


def _load_backend() -> None:
    """Import JAX and the sharding wrapper (idempotent)."""
    global jax, jnp, lax, NamedSharding, P, create_device_mesh, ShardedStencilNode
    global LBMNode, Lattice
    if jax is not None:
        return
    _ALLOCATOR.update(_allocator_environment())

    import jax as _jax
    import jax.numpy as _jnp
    from jax import lax as _lax
    from jax.sharding import NamedSharding as _NamedSharding
    from jax.sharding import PartitionSpec as _P

    from maddening.cloud.multigpu import device_mesh as _dm
    from maddening.cloud.multigpu import sharded_node as _sn
    from maddening.nodes import lbm as _lbm

    jnp, lax, NamedSharding, P = _jnp, _lax, _NamedSharding, _P
    create_device_mesh = _dm.create_device_mesh
    ShardedStencilNode = _sn.ShardedStencilNode
    LBMNode = _lbm.LBMNode
    Lattice = _make_lattice_class(LBMNode)
    jax = _jax


def environment() -> dict:
    """What the numbers were measured on.  No ``nvidia-smi``: the runbook's
    own step records it, and the operator passes the card's size."""
    _load_backend()
    devices = jax.devices()
    from jaxlib import version as jaxlib_version  # noqa: PLC0415

    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib_version.__version__,
        "platform": devices[0].platform,
        "devices": [str(d) for d in devices],
        "device_kinds": sorted({d.device_kind for d in devices}),
        "n_devices_visible": len(devices),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "allocator": dict(_ALLOCATOR),
        "git_commit": _git_commit(),
    }


def read_device_memory(devices) -> list[dict]:
    """One row per device: ``Device.memory_stats()``'s counters where the
    backend keeps them, and ``measured: False`` where it does not (CPU)."""
    rows = []
    for device in devices:
        try:
            stats = device.memory_stats()
        except Exception as exc:  # noqa: BLE001 - a backend without the call
            stats, row = None, {"device": str(device), "measured": False,
                                "why": f"{type(exc).__name__}: {exc}"}
        else:
            row = {"device": str(device), "measured": stats is not None}
        if stats is not None:
            row.update({key: int(stats[key]) for key in MEMORY_KEYS if key in stats})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# The problem: a tile, its unsharded run, and the tiled field block by block
# ---------------------------------------------------------------------------


def tile_pattern(tile, phase: float) -> np.ndarray:
    """A smooth pattern with the tile's own period along every axis and no
    shorter one: it carries a frequency-one mode along each, so no shift by
    part of the tile leaves it alone.  ``phase`` makes two patterns unlike."""
    tx, ty, tz = tile
    x = (np.arange(tx, dtype=np.float64) / tx)[:, None, None]
    y = (np.arange(ty, dtype=np.float64) / ty)[None, :, None]
    z = (np.arange(tz, dtype=np.float64) / tz)[None, None, :]
    two_pi = 2 * np.pi
    return (np.cos(two_pi * x + phase) + 0.6 * np.sin(two_pi * y + 1.7 * phase + 0.5)
            + 0.5 * np.cos(two_pi * z + 0.9 * phase + 2.0)
            + 0.4 * np.cos(two_pi * (x - 2 * y + 3 * z) + 2.3 * phase + 1.0))


def tile_start(tile, lattice) -> dict:
    """The tile's start state, on the host: see ``DENSITY_AMPLITUDE``."""
    e = lattice.e.astype(np.float64)
    w, cs2 = lattice.w.astype(np.float64), float(lattice.cs2)
    rho = 1.0 + DENSITY_AMPLITUDE * tile_pattern(tile, 0.4) / _PATTERN_MAX
    u = np.stack([VELOCITY_AMPLITUDE * tile_pattern(tile, 1.3 + 2.1 * d) / _PATTERN_MAX
                  for d in range(e.shape[1])], axis=-1)
    eu = u @ e.T
    uu = np.sum(u * u, axis=-1, keepdims=True)
    f_eq = w * rho[..., None] * (1.0 + eu / cs2 + eu ** 2 / (2.0 * cs2 ** 2) - uu / (2.0 * cs2))
    ripple = np.stack([tile_pattern(tile, 0.7 * q + 0.2) for q in range(e.shape[0])],
                      axis=-1) / _PATTERN_MAX
    f = (f_eq * (1.0 + POPULATION_RIPPLE * ripple)).astype(np.float32)
    f64 = f.astype(np.float64)
    density = f64.sum(axis=-1)
    velocity = (f64 @ e) / density[..., None]
    return {"f": f, "density": density.astype(np.float32),
            "velocity": velocity.astype(np.float32),
            "pressure": (density * cs2).astype(np.float32),
            "wall_mask": np.zeros(tuple(tile), np.uint8)}


def least_plane_difference(field: np.ndarray, axis: int) -> float:
    """The smallest largest-difference between two planes of ``field`` along
    ``axis``: zero when two of them are alike."""
    planes = np.moveaxis(np.asarray(field, np.float64), axis, 0)
    n = planes.shape[0]
    return min((float(np.max(np.abs(planes[i] - planes[j])))
                for i in range(n) for j in range(i + 1, n)), default=math.inf)


def tile_reference(tile, tile_state: dict, steps: int) -> dict:
    """The tile after ``steps`` periodic steps of the unsharded ``LBMNode``,
    on one device, as host arrays."""
    key = tuple(tile)
    if key not in _TILE_STEPS:
        node = LBMNode("tile", 1.0, grid_shape=key, viscosity=VISCOSITY, lattice=LATTICE)
        _TILE_STEPS[key] = jax.jit(lambda s: node.update(s, {}, node.delta_t))
    step = _TILE_STEPS[key]
    state = {name: jnp.asarray(value) for name, value in tile_state.items()}
    for _ in range(steps):
        state = step(state)
    return {name: np.asarray(value) for name, value in state.items()}


def build_mesh(mesh_shape: tuple[int, int]):
    """``(mesh, axis_map)``: the pencil over spatial axes 0 and 1, or the
    1-D mesh over axis 0."""
    a, b = mesh_shape
    if b == 1:
        return create_device_mesh(shape=(a,), axis_names=MESH_AXES[:1]), {MESH_AXES[0]: 0}
    return (create_device_mesh(shape=(a, b), axis_names=MESH_AXES),
            {MESH_AXES[0]: 0, MESH_AXES[1]: 1})


def field_sharding(mesh, axis_map: dict, ndim: int):
    """The sharding of a state field: split along the spatial axes of
    ``axis_map``, whole along the others -- what the wrapper compiles its
    step for, so a step moves no block between devices."""
    spec: list = [None] * ndim
    for mesh_axis, spatial_axis in axis_map.items():
        spec[spatial_axis] = mesh_axis
    return NamedSharding(mesh, P(*spec))


def _take3(tile, ix, iy, iz):
    """The rows ``ix``, ``iy``, ``iz`` of ``tile`` along its three spatial
    axes.  The indices are tile rows already (taken modulo the tile's
    extents on the host), so nothing is out of range to fill or to check."""
    def rows(a, index, axis):
        return jnp.take(a, index, axis=axis, mode="clip")
    return rows(rows(rows(tile, ix, 0), iy, 1), iz, 2)


def _tiled_state(tiles: dict, ix, iy, iz) -> dict:
    return {name: _take3(tile, ix, iy, iz) for name, tile in tiles.items()}


def _compare_state(blocks: dict, x0, tiles: dict, ix, iy, iz) -> dict:
    """``ix.shape[0]`` planes of every field of one device's block, from
    plane ``x0``, against the same planes of the tiled reference: the
    largest difference, the reference's scale, whether every value is
    finite, and the sum over each row (for the mass).

    A difference that is not a number counts as infinite: a reduction's
    ``max`` need not carry a NaN (on CPU it dropped one on the 1-D mesh and
    kept it on the pencil), and a NaN cell must never read as "no
    difference"."""
    out = {}
    for name, block in blocks.items():
        got = lax.dynamic_slice_in_dim(block, x0, ix.shape[0], axis=0).astype(jnp.float32)
        want = _take3(tiles[name], ix, iy, iz).astype(jnp.float32)
        apart = jnp.abs(got - want)
        out[name] = {"max_abs": jnp.max(jnp.where(jnp.isnan(apart), jnp.inf, apart)),
                     "scale": jnp.max(jnp.abs(want)),
                     "finite": jnp.all(jnp.isfinite(got)),
                     "rows": jnp.sum(got, axis=tuple(range(2, got.ndim)))}
    return out


def _program(name: str):
    if name not in _PROGRAMS:
        _PROGRAMS[name] = jax.jit({"tiled": _tiled_state, "compare": _compare_state}[name])
    return _PROGRAMS[name]


def _bounds(index, shape) -> list[tuple[int, int]]:
    """A device's ``(start, stop)`` along the three spatial axes."""
    out = []
    for s, n in zip(index[:3], shape):
        start, stop, step = s.indices(n)
        assert step == 1, index
        out.append((start, stop))
    return out


def _on(device, tile_state: dict, ranges, tile) -> tuple:
    """``tile_state`` and the tile rows of ``ranges``, placed on ``device``."""
    tiles = {name: jax.device_put(value, device) for name, value in tile_state.items()}
    index = [jax.device_put((np.arange(start, stop, dtype=np.int64) % t).astype(np.int32), device)
             for (start, stop), t in zip(ranges, tile)]
    return tiles, index


def tiled_block(tile_state: dict, ranges, tile, device, *, most_cells: int) -> dict:
    """The block ``ranges`` -- ``(start, stop)`` per spatial axis, in the
    grid's own indices -- of every field of the tiled state, gathered from
    the tile **on** ``device``.  Refuses a block of more than ``most_cells``
    cells: nothing here may build more than one shard's block."""
    cells = math.prod(stop - start for start, stop in ranges)
    if cells > most_cells:
        raise RuntimeError(f"asked for a block of {cells} cells, more than one shard's "
                           f"({most_cells}): the field is built one device's block at a time")
    tiles, index = _on(device, tile_state, ranges, tile)
    return _program("tiled")(tiles, *index)


def build_state(tile_state: dict, tile, shape, mesh, axis_map: dict,
                requests: list | None = None) -> dict:
    """The tiled state over ``mesh``, built block by block: each device
    gathers its own block from the tile, and the blocks are assembled with
    the sharding the wrapper steps.  The whole field is never on one device
    and never on the host.  ``requests`` collects ``(device, ranges)`` of
    every block asked for."""
    devices = list(mesh.devices.flat)
    shard_cells = math.prod(shape) // len(devices)
    shardings = {name: field_sharding(mesh, axis_map, value.ndim)
                 for name, value in tile_state.items()}
    shapes = {name: tuple(shape) + value.shape[3:] for name, value in tile_state.items()}
    blocks = {}
    for device, index in shardings["f"].devices_indices_map(shapes["f"]).items():
        ranges = _bounds(index, shape)
        if requests is not None:
            requests.append((str(device), ranges))
        blocks[device] = tiled_block(tile_state, ranges, tile, device, most_cells=shard_cells)
    state = {}
    for name in tile_state:
        placed = shardings[name].devices_indices_map(shapes[name])
        for device, index in placed.items():
            assert tuple(blocks[device][name].shape[:3]) == tuple(
                stop - start for start, stop in _bounds(index, shape)), (name, device)
        state[name] = jax.make_array_from_single_device_arrays(
            shapes[name], shardings[name], [blocks[device][name] for device in placed])
    return state


def _nan_max(a: float, b: float) -> float:
    """The larger of two differences; NaN once either is (a NaN difference
    is never "no difference")."""
    return math.nan if math.isnan(a) or math.isnan(b) else max(a, b)


def compare_with_tiling(state: dict, reference: dict, tile, shape, chunk_bytes: int) -> dict:
    """Every field of ``state`` against ``reference`` tiled, on the devices,
    a few planes at a time.

    Per device, the block is walked along spatial axis 0 in chunks of as
    many planes as hold at most ``chunk_bytes`` of the state (at least one
    plane); each chunk is compared with the same planes gathered from the
    reference tile on that device.  The comparison therefore holds, per
    device, a chunk, its reference and their difference -- a few times
    ``chunk_bytes`` -- and never a second copy of a block.

    Returns ``{"fields": {name: {max_abs, reference_scale, max_rel, exact,
    finite, cells_compared}}, "mass", "chunk_bytes", "chunk_planes",
    "chunks"}``; ``mass`` is the sum of ``f``, added up in float64 on the
    host from each row's float32 sum.
    """
    blocks: dict = {}
    ranges: dict = {}
    for name, array in state.items():
        for shard in array.addressable_shards:
            blocks.setdefault(shard.device, {})[name] = shard.data
            ranges[shard.device] = _bounds(shard.index, shape)
    fields = {name: {"max_abs": 0.0, "reference_scale": 0.0, "finite": True, "cells_compared": 0}
              for name in state}
    mass, chunks, most_planes, most_bytes = 0.0, 0, 0, 0
    compare = _program("compare")
    for device, block in blocks.items():
        (x0, x1), (y0, y1), (z0, z1) = ranges[device]
        plane_cells = (y1 - y0) * (z1 - z0)
        planes = max(1, min(x1 - x0, chunk_bytes // (plane_cells * STATE_BYTES_PER_CELL)))
        most_planes = max(most_planes, planes)
        most_bytes = max(most_bytes, planes * plane_cells * STATE_BYTES_PER_CELL)
        tiles, (_, iy, iz) = _on(device, reference, ranges[device], tile)
        done = 0
        while done < x1 - x0:
            # Every chunk has ``planes`` planes, so one program serves them
            # all; the last starts early and its repeated planes are left
            # out of the sums.
            start = min(done, (x1 - x0) - planes)
            fresh = done - start
            ix = jax.device_put(((x0 + start + np.arange(planes, dtype=np.int64)) % tile[0])
                                .astype(np.int32), device)
            got = jax.device_get(compare(block, start, tiles, ix, iy, iz))
            for name, row in got.items():
                entry = fields[name]
                entry["max_abs"] = _nan_max(entry["max_abs"], float(row["max_abs"]))
                entry["reference_scale"] = max(entry["reference_scale"], float(row["scale"]))
                entry["finite"] = entry["finite"] and bool(row["finite"])
                entry["cells_compared"] += (planes - fresh) * plane_cells
            mass += float(np.sum(np.asarray(got["f"]["rows"], np.float64)[fresh:]))
            done = start + planes
            chunks += 1
    for entry in fields.values():
        scale = entry["reference_scale"]
        entry["max_rel"] = entry["max_abs"] / scale if scale > 0 else entry["max_abs"]
        entry["exact"] = entry["max_abs"] == 0.0
    return {"fields": fields, "mass": mass, "chunk_bytes": most_bytes,
            "chunk_planes": most_planes, "chunks": chunks}


# ---------------------------------------------------------------------------
# One rung, in its own process
# ---------------------------------------------------------------------------


class Problem(NamedTuple):
    """What a rung steps and what it compares with."""
    tile: tuple
    k: tuple
    shape: tuple
    cells: int
    steps: int
    mesh: Any
    axis_map: dict
    devices: list
    sharded: Any
    dt: float
    tile_state: dict
    reference: dict
    least_plane_difference: float   # over the largest |f| of the tile
    chunk_bytes: int


def is_out_of_memory(exc: BaseException) -> bool:
    """An allocator's refusal: the host's (``MemoryError``) or the runtime's
    resource-exhausted error."""
    return isinstance(exc, MemoryError) or bool(
        _OUT_OF_MEMORY.search(f"{type(exc).__name__}: {exc}"))


def base_record(args, kind: str) -> dict:
    tile, k = tuple(args.tile), tuple(args.k)
    mesh_shape = parse_mesh(args.mesh)
    return {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "kind": kind,
        "gating": False,
        "dry_run": bool(args.dry_run),
        "rung": args.rung_index,
        "target_fill": args.target_fill,
        "mesh": {"label": args.mesh, "shape": list(mesh_shape),
                 "devices_per_spatial_axis": list(devices_per_axis(mesh_shape)),
                 "n_devices": mesh_shape[0] * mesh_shape[1]},
        "lattice": LATTICE,
        "viscosity": VISCOSITY,
        "tile": list(tile),
        "k": list(k),
        "shape": list(grid_shape(tile, k)),
        "cells": _cells(tile, k),
        "steps": args.steps,
        "state_bytes_per_cell": STATE_BYTES_PER_CELL,
        "limit": LIMIT,
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
        "phase": "starting",
        "pid": os.getpid(),
        "outcome": None,
        "memory": {"device_memory_gb": args.device_memory_gb, "readings": [], "fill": None},
    }


def prepare(args, record: dict, flush: Callable[[], None]) -> Problem | None:
    """Everything before the first block is built: the backend, the host
    guard, the tile and its unsharded run, the mesh and the wrapped
    lattice.  ``None`` when the host guard refused."""
    _load_backend()
    record["environment"] = env = environment()
    tile, k = tuple(args.tile), tuple(args.k)
    shape, cells = grid_shape(tile, k), _cells(tile, k)
    mesh_shape = parse_mesh(args.mesh)
    n_devices = mesh_shape[0] * mesh_shape[1]
    if not args.dry_run and env["platform"] != "gpu":
        print("WARNING: no GPU backend -- nothing here says anything about a card's capacity")
    record["host"] = guard = host_guard(cells, n_devices,
                                        devices_are_host=env["platform"] == "cpu")
    if guard["refused"]:
        record["outcome"], record["outcome_detail"] = HOST_GUARD, guard["why"]
        flush()
        return None
    record["phase"] = "reference"
    flush()
    tile_node = LBMNode("tile", 1.0, grid_shape=tile, viscosity=VISCOSITY, lattice=LATTICE)
    tile_state = tile_start(tile, tile_node.lattice)
    reference = tile_reference(tile, tile_state, args.steps)
    per_axis = devices_per_axis(mesh_shape)
    least = min(least_plane_difference(tile_state["f"], axis)
                for axis in range(3) if per_axis[axis] > 1) / float(np.max(tile_state["f"]))
    mesh, axis_map = build_mesh(mesh_shape)
    node = Lattice("lattice", 1.0, grid_shape=shape, viscosity=VISCOSITY, lattice=LATTICE)
    sharded = ShardedStencilNode(node, mesh, axis_map=axis_map, boundary="periodic")
    record["on_one_device"] = [{
        "what": "LBMNode's own wall mask, a bool per cell its constructor builds on the "
                "default device and the sharded step does not read",
        "bytes": cells, "device": str(jax.devices()[0])}]
    return Problem(tile=tile, k=k, shape=shape, cells=cells, steps=args.steps, mesh=mesh,
                   axis_map=axis_map, devices=list(mesh.devices.flat), sharded=sharded,
                   dt=node.delta_t, tile_state=tile_state, reference=reference,
                   least_plane_difference=least,
                   chunk_bytes=int(args.check_chunk_mb * 1024 * 1024))


def run_pass(problem: Problem, note: Callable[[str], None]) -> dict:
    """Build the tiled state, step it and compare it, once.  ``note(at)``
    is called at each of :data:`READINGS`."""
    seconds: dict = {}
    requests: list = []
    t0 = time.perf_counter()
    state = build_state(problem.tile_state, problem.tile, problem.shape, problem.mesh,
                        problem.axis_map, requests)
    jax.block_until_ready(state)
    seconds["build"] = time.perf_counter() - t0
    note(READINGS[0])
    n_devices = len(problem.devices)
    partitioned = all(len(a.sharding.device_set) == n_devices
                      and not a.sharding.is_fully_replicated for a in state.values())
    placed = {name: (a.sharding, a.shape) for name, a in state.items()}

    t0 = time.perf_counter()
    stepped = problem.sharded.update(state, {}, problem.dt)
    jax.block_until_ready(stepped)
    seconds["first_step"] = time.perf_counter() - t0
    note(READINGS[1])
    keeps_layout = all(
        stepped[name].shape == shape and stepped[name].sharding.is_equivalent_to(sharding, len(shape))
        for name, (sharding, shape) in placed.items())
    state = stepped
    del stepped
    per_step = []
    for _ in range(problem.steps - 1):
        t0 = time.perf_counter()
        state = problem.sharded.update(state, {}, problem.dt)
        jax.block_until_ready(state)
        per_step.append(time.perf_counter() - t0)
    seconds["steps_after_the_first"] = sum(per_step)
    note(READINGS[2])

    t0 = time.perf_counter()
    compared = compare_with_tiling(state, problem.reference, problem.tile, problem.shape,
                                   problem.chunk_bytes)
    seconds["check"] = time.perf_counter() - t0
    note(READINGS[3])
    del state
    shard_cells = problem.cells // n_devices
    return {
        "seconds": seconds,
        "steps_per_second": (len(per_step) / sum(per_step)) if per_step and sum(per_step) > 0
        else None,
        "builder_requests": [{"device": device, "ranges": [list(r) for r in ranges],
                              "cells": math.prod(b - a for a, b in ranges)}
                             for device, ranges in requests],
        "shard_cells": shard_cells,
        "partitioned": partitioned,
        "keeps_layout": keeps_layout,
        "compared": compared,
    }


def pass_checks(problem: Problem, result: dict) -> list:
    """The checks of one pass, from its results alone."""
    n_devices = len(problem.devices)
    requests = result["builder_requests"]
    compared = result["compared"]
    tile_mass = float(np.sum(problem.reference["f"], dtype=np.float64)) * math.prod(problem.k)
    checks = [
        check_that("the tile: no two planes along a split axis are alike",
                   problem.least_plane_difference > PLANES_APART * LIMIT,
                   f"smallest plane-to-plane difference {problem.least_plane_difference:.3e} "
                   f"of the largest population, against {PLANES_APART:g} times the limit"),
        check_that(f"the builder was asked for {n_devices} blocks, one per device, none larger "
                   "than a shard",
                   len(requests) == n_devices
                   and len({r["device"] for r in requests}) == n_devices
                   and all(r["cells"] == result["shard_cells"] for r in requests),
                   f"{[r['cells'] for r in requests]} cells against {result['shard_cells']}"),
        check_that(f"the state is partitioned over {n_devices} devices", result["partitioned"]),
        check_that("a step returns the state in the layout it was given",
                   result["keeps_layout"]),
    ]
    for name in STATE_FIELDS:
        entry = compared["fields"][name]
        checks.append(check(f"{name} vs the tiled reference max_rel", entry["max_rel"], LIMIT))
        checks.append(check_that(f"{name} finite", entry["finite"]))
    checks.append(check_that(
        "every cell of every field was compared",
        all(entry["cells_compared"] == problem.cells for entry in compared["fields"].values()),
        f"{sorted({e['cells_compared'] for e in compared['fields'].values()})} of {problem.cells}"))
    mass_rel = abs(compared["mass"] - tile_mass) / tile_mass if tile_mass > 0 else math.inf
    checks.append(check("total mass vs prod(K) times the tile's rel", mass_rel, LIMIT))
    return checks


def memory_checks(memory: dict) -> list:
    """Whether the fill was measured: a check not run where it was not."""
    name = "peak device memory read on every device (the fill)"
    if memory["fill"] is None:
        return [check_not_run(name, f"not measured: {memory['why_not']}")]
    return [check_that(name, True, f"fill {memory['fill']:.4f}")]


def _note_memory(problem: Problem, readings: list, flush) -> Callable[[str], None]:
    def note(at: str) -> None:
        readings.append({"at": at, "devices": read_device_memory(problem.devices)})
        flush()
    return note


def _fill_into(memory: dict, device_gb) -> None:
    """Store the fill and what it rests on, from the readings."""
    derived = fill_of(memory["readings"], int(device_gb * GIB) if device_gb else None)
    memory.update(fill=derived["fill"], fill_by_device=derived["by_device"],
                  peak_bytes=derived["peak_bytes"], bytes_limit=derived["bytes_limit"],
                  why_not=derived["why_not"])


def _outcome_of(checks: list) -> str:
    ran = _ran(checks)
    return PASSED if ran and all(c["passed"] for c in ran) else CHECK_FAILED


def _print_checks(label: str, checks: list) -> None:
    ran = _ran(checks)
    failed = [c for c in ran if not c["passed"]]
    print(f"[capacity] {label}: checks {len(ran) - len(failed)}/{len(ran)} passed"
          + (f", {len(checks) - len(ran)} not run" if len(checks) > len(ran) else ""))
    for c in failed:
        print(f"CHECK FAILED [capacity] {c['name']}: {c['value']} (limit {c['limit']})"
              + (f" -- {c['detail']}" if c.get("detail") else ""))
    for c in checks:
        if c.get("not_run"):
            print(f"CHECK NOT RUN [capacity] {c['name']} -- {c['detail']}")


def run_rung(args, record: dict, flush: Callable[[], None]) -> int:
    """One rung: build, step, compare, with the four memory readings."""
    t_start = time.perf_counter()
    problem = prepare(args, record, flush)
    if problem is None:
        print(f"[capacity] rung {args.rung_index}: {record['outcome_detail']}")
        return EXIT_RUNG_HOST_GUARD
    memory = record["memory"]
    print(f"[capacity] rung {args.rung_index} mesh {args.mesh} K={problem.k} grid "
          f"{'x'.join(map(str, problem.shape))} = {problem.cells} cells, {problem.steps} steps",
          flush=True)
    record["phase"] = "running"
    result = run_pass(problem, _note_memory(problem, memory["readings"], flush))
    record["phase"] = "judging"
    _fill_into(memory, args.device_memory_gb)
    # Bytes per cell as the fullest device saw it: its peak over its share.
    memory["peak_bytes_per_cell"] = None if memory["peak_bytes"] is None else (
        memory["peak_bytes"] * len(problem.devices) / problem.cells)
    checks = pass_checks(problem, result) + memory_checks(memory)
    compared = result["compared"]
    record.update(
        seconds={**result["seconds"], "total": time.perf_counter() - t_start},
        steps_per_second=result["steps_per_second"],
        cell_updates_per_second=(result["steps_per_second"] * problem.cells
                                 if result["steps_per_second"] else None),
        builder_requests=result["builder_requests"],
        check_chunk_bytes=compared["chunk_bytes"],
        check_chunk_planes=compared["chunk_planes"],
        check_chunks=compared["chunks"],
        results={"fields": compared["fields"], "mass": compared["mass"],
                 "least_plane_difference": problem.least_plane_difference},
        checks=checks,
        outcome=_outcome_of(checks),
        phase="done",
    )
    record["host"]["vm_hwm_bytes"] = vm_hwm_bytes()
    flush()
    _print_checks(f"rung {args.rung_index}", checks)
    fill = memory["fill"]
    print(f"[capacity] rung {args.rung_index}: {record['outcome']}; fill "
          + (f"{fill:.3f}" if fill is not None else "not measured")
          + f"; largest max_rel {max(e['max_rel'] for e in compared['fields'].values()):.2e}",
          flush=True)
    return EXIT_OK if record["outcome"] == PASSED else EXIT_CHECK_FAILED


def run_soak(args, record: dict, flush: Callable[[], None]) -> int:
    """Blocks of build, steps and comparison until ``--soak-minutes`` have
    passed (two blocks at least: growth needs two readings), the memory
    read after every block."""
    t_start = time.perf_counter()
    problem = prepare(args, record, flush)
    if problem is None:
        print(f"[capacity] soak: {record['outcome_detail']}")
        return EXIT_RUNG_HOST_GUARD
    memory = record["memory"]
    record["blocks"] = blocks = []
    record["phase"] = "running"
    deadline = t_start + 60.0 * args.soak_minutes
    print(f"[capacity] soak mesh {args.mesh} K={problem.k} = {problem.cells} cells, blocks of "
          f"{problem.steps} steps for {args.soak_minutes} min", flush=True)
    failed = False
    while len(blocks) < 2 or time.perf_counter() < deadline:
        readings: list = []
        result = run_pass(problem, _note_memory(problem, readings, lambda: None))
        checks = pass_checks(problem, result)
        after = readings[-1]["devices"]
        blocks.append({
            "block": len(blocks) + 1,
            "seconds": result["seconds"],
            "steps_per_second": result["steps_per_second"],
            "largest_max_rel": max(e["max_rel"] for e in result["compared"]["fields"].values()),
            "exact": all(e["exact"] for e in result["compared"]["fields"].values()),
            "checks": checks,
            "memory_after": after,
            "host_rss_bytes": vm_rss_bytes(),
        })
        memory["readings"] += [{"at": f"block {len(blocks)}, {r['at']}", "devices": r["devices"]}
                               for r in readings]
        flush()
        print(f"[capacity] soak block {len(blocks)}: {_outcome_of(checks)}, largest max_rel "
              f"{blocks[-1]['largest_max_rel']:.2e}, host RSS "
              f"{(blocks[-1]['host_rss_bytes'] or 0) / GIB:.2f} GiB", flush=True)
        if _outcome_of(checks) != PASSED:
            failed = True
            _print_checks(f"soak block {len(blocks)}", checks)
            break
    record["phase"] = "judging"
    _fill_into(memory, args.device_memory_gb)
    record["growth"] = growth = soak_growth(blocks)
    record.update(
        checks=memory_checks(memory),
        seconds={"total": time.perf_counter() - t_start},
        outcome=CHECK_FAILED if failed else PASSED,
        phase="done",
    )
    record["host"]["vm_hwm_bytes"] = vm_hwm_bytes()
    flush()
    print(f"[capacity] soak: {record['outcome']} after {len(blocks)} blocks; "
          + _growth_line(growth), flush=True)
    return EXIT_CHECK_FAILED if failed else EXIT_OK


def soak_growth(blocks: list) -> dict:
    """How the memory moved from the first block to the last, read at the
    same point of each (after its comparison): per device ``bytes_in_use``
    and ``peak_bytes_in_use`` where the backend keeps them, and the host's
    resident memory.  The first block compiles; the growth that matters is
    from the second block to the last (``after_first``; ``None`` with fewer
    than three blocks, where there is no such stretch)."""
    def series(key):
        per_device: dict = {}
        for block in blocks:
            for row in block["memory_after"]:
                value = row.get(key)
                if not isinstance(value, int) or isinstance(value, bool):
                    return None
                per_device.setdefault(row["device"], []).append(value)
        return per_device or None

    def moved(values):
        return {"first_to_last": values[-1] - values[0],
                "after_first": values[-1] - values[1] if len(values) >= 3 else None}

    out: dict = {"blocks": len(blocks)}
    for key in ("bytes_in_use", "peak_bytes_in_use"):
        per_device = series(key)
        out[key] = None if per_device is None else {d: moved(v) for d, v in per_device.items()}
    rss = [b["host_rss_bytes"] for b in blocks]
    out["host_rss_bytes"] = moved(rss) if all(isinstance(v, int) for v in rss) and rss else None
    return out


def _growth_line(growth: dict) -> str:
    def most(moves) -> tuple[int, str]:
        """The largest move, and the stretch it is over: from the second
        block on where there are three, else from the first (which holds
        the first block's compiles)."""
        key, over = ("after_first", "after the first block") \
            if all(m["after_first"] is not None for m in moves) \
            else ("first_to_last", "from the first block to the last")
        return max(m[key] for m in moves), over

    in_use = growth.get("bytes_in_use")
    if in_use is None:
        device = "device memory not measured"
    else:
        device = "device bytes in use grew by {} {}".format(*most(list(in_use.values())))
    rss = growth.get("host_rss_bytes")
    if rss is None:
        host = "host RSS not read"
    else:
        moved, over = most([rss])
        host = f"host RSS grew by {moved / (1024 * 1024):.1f} MiB {over}"
    return f"{device}; {host}"


def child_main(args) -> int:
    """A rung's or the soak's own process: the record is written at every
    phase, so one that dies leaves what it had measured."""
    kind = SOAK_KIND if args.child == "soak" else RUNG_KIND
    record = base_record(args, kind)

    def flush() -> None:
        write_json(args.record, record)

    flush()
    try:
        return (run_soak if args.child == "soak" else run_rung)(args, record, flush)
    except Exception as exc:  # noqa: BLE001 - every way a rung can end is an outcome
        detail = f"{type(exc).__name__}: {str(exc)[:2000]}"
        record["raised"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                            "phase": record.get("phase")}
        if is_out_of_memory(exc):
            record["outcome"], record["outcome_detail"] = OUT_OF_MEMORY, (
                f"while {record.get('phase')}: {detail}")
            flush()
            print(f"[capacity] OUT OF MEMORY while {record.get('phase')}: {detail[:300]}",
                  file=sys.stderr)
            return EXIT_RUNG_OUT_OF_MEMORY
        traceback.print_exc()
        record["outcome"], record["outcome_detail"] = CRASHED, (
            f"while {record.get('phase')}: {detail}")
        flush()
        return EXIT_CRASHED


# ---------------------------------------------------------------------------
# The ramp: one process per rung, sized from what the last one measured
# ---------------------------------------------------------------------------


class Launched(NamedTuple):
    """How a rung's process ended."""
    returncode: int | None
    timed_out: bool
    stderr_tail: str
    wall_s: float


def _tail(path: Path, lines: int = 20) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def launch_child(argv: list[str], timeout_s: float, stderr_path: Path) -> Launched:
    """Run this script again for one rung, under its time box.  Its stdout
    is this process's; its stderr goes to ``stderr_path``.  A rung past its
    time box is killed -- this process's own child, by its handle -- and so
    is one still running when this process is on its way out (``timeout``
    around the ramp, Ctrl-C): a rung left behind would keep the cards."""
    t0 = time.perf_counter()
    timed_out = False
    with open(stderr_path, "w", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv],
                                stderr=err)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return Launched(proc.returncode, timed_out, _tail(stderr_path), time.perf_counter() - t0)


def classify(launched: Launched, record: dict | None) -> tuple[str, str]:
    """``(outcome, detail)`` of a rung from how its process ended and the
    record it left.

    The process's own word is taken only where the record bears it out: a
    status of 0 without a record whose checks pass by their own values is
    a crash, not a pass.
    """
    try:
        derived = rederive(record) if record is not None else None
    except _UNREADABLE:
        derived = None
    rc = launched.returncode
    if launched.timed_out:
        return TIMED_OUT, f"killed at its time box, while {(record or {}).get('phase')}"
    if rc == EXIT_OK:
        if derived is not None and derived["outcome"] == PASSED and not derived["problems"]:
            return PASSED, ""
        return CRASHED, "it exited 0 without a record whose checks pass: " + (
            "; ".join(derived["problems"]) if derived else "no record")
    if rc == EXIT_CHECK_FAILED and derived is not None and derived["failed"]:
        return CHECK_FAILED, "; ".join(derived["failed"][:6])
    if rc == EXIT_RUNG_OUT_OF_MEMORY and record is not None and record.get("outcome") == OUT_OF_MEMORY:
        return OUT_OF_MEMORY, str(record.get("outcome_detail", ""))
    if rc == EXIT_RUNG_HOST_GUARD and record is not None and record.get("outcome") == HOST_GUARD:
        return HOST_GUARD, str(record.get("outcome_detail", ""))
    if rc in _KILLED:
        return OUT_OF_MEMORY, (f"the process was killed (status {rc}) while "
                               f"{(record or {}).get('phase')}: the host's out-of-memory "
                               "killer, unless something else sent the signal")
    if rc == EXIT_CRASHED and record is not None and record.get("outcome") == CRASHED:
        return CRASHED, str(record.get("outcome_detail", "")) + "\n" + launched.stderr_tail
    if _OUT_OF_MEMORY.search(launched.stderr_tail):
        return OUT_OF_MEMORY, (f"the process ended with status {rc} while "
                               f"{(record or {}).get('phase')}, its stderr naming an "
                               "allocation:\n" + launched.stderr_tail)
    return CRASHED, f"status {rc} while {(record or {}).get('phase')}\n" + launched.stderr_tail


def first_rung_cells(target: float, device_bytes: int, n_devices: int,
                     bytes_per_cell: float) -> int:
    """The cells that fill ``target`` of every card at ``bytes_per_cell``."""
    return int(target * device_bytes * n_devices / bytes_per_cell)


def next_rung_cells(target: float, measured_fill: float, measured_cells: int) -> int:
    """The cells that fill ``target`` if the peak per cell stays what the
    last rung measured: its cells scaled by ``target / measured_fill``."""
    return int(measured_cells * target / measured_fill)


def _child_argv(args, kind: str, k, index: int, target, record: Path) -> list[str]:
    argv = ["--child", kind, "--record", str(record), "--rung-index", str(index),
            "--mesh", args.mesh, "--tile", *map(str, args.tile), "--k", *map(str, k),
            "--steps", str(args.steps), "--check-chunk-mb", str(args.check_chunk_mb)]
    if target is not None:
        argv += ["--target-fill", repr(float(target))]
    if args.device_memory_gb is not None:
        argv += ["--device-memory-gb", repr(float(args.device_memory_gb))]
    if kind == "soak":
        argv += ["--soak-minutes", repr(float(args.soak_minutes))]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _row(index, target, sized_from, k, tile, record_name) -> dict:
    return {"rung": index, "file": record_name, "target_fill": target, "sized_from": sized_from,
            "k": list(k), "shape": list(grid_shape(tile, k)), "cells": _cells(tile, k)}


def _run_one(args, launch, kind: str, row: dict, timeout_s: float) -> dict:
    """Launch one rung (or the soak), classify it, and finish its record
    and ``row``."""
    path = args.out / row["file"]
    stderr_path = path.with_suffix(".stderr.log")
    launched = launch(_child_argv(args, kind, row["k"], row["rung"], row["target_fill"], path),
                      timeout_s, stderr_path)
    record, why = read_record(path) if path.exists() else (None, "no record was written")
    outcome, detail = classify(launched, record)
    if record is None:
        # The process left nothing readable: write the record it could not.
        ns = argparse.Namespace(**{**vars(args), "k": row["k"], "rung_index": row["rung"],
                                   "target_fill": row["target_fill"], "child": kind,
                                   "record": path})
        record = base_record(ns, SOAK_KIND if kind == "soak" else RUNG_KIND)
        record["phase"] = f"unknown ({why})"
    record["outcome"] = outcome
    if detail:
        record["outcome_detail"] = detail
    record["launch"] = {"returncode": launched.returncode, "timed_out": launched.timed_out,
                        "wall_s": launched.wall_s, "time_box_s": timeout_s,
                        "stderr_log": stderr_path.name,
                        "stderr_tail": "" if outcome == PASSED else launched.stderr_tail}
    # A rung that died before judging left its readings: its fill is what
    # they give (how far it had got), so the record agrees with itself.
    memory = record.setdefault("memory", {"readings": []})
    _fill_into(memory, args.device_memory_gb)
    memory.setdefault("peak_bytes_per_cell", None if memory["peak_bytes"] is None else (
        memory["peak_bytes"] * record["mesh"]["n_devices"] / record["cells"]))
    write_json(path, record)
    fields = (record.get("results") or {}).get("fields") or {}
    row.update(outcome=outcome, detail=detail, fill=memory.get("fill"),
               peak_bytes=memory.get("peak_bytes"),
               peak_bytes_per_cell=memory.get("peak_bytes_per_cell"),
               bytes_limit=memory.get("bytes_limit"), wall_s=launched.wall_s,
               steps_per_second=record.get("steps_per_second"),
               largest_max_rel=max((e["max_rel"] for e in fields.values()), default=None),
               exact=all(e["exact"] for e in fields.values()) if fields else None)
    return row


def ramp_status(rows: list, soak: dict | None) -> int:
    """The exit status of a ramp from its rungs' outcomes, in order."""
    outcomes = [r["outcome"] for r in rows] + ([soak["outcome"]] if soak else [])
    if CHECK_FAILED in outcomes:
        return EXIT_CHECK_FAILED
    if not rows or rows[0]["outcome"] != PASSED:
        return EXIT_NO_CEILING
    return EXIT_OK


def ceiling_of(rows: list) -> dict | None:
    """The largest rung that passed."""
    passed = [r for r in rows if r["outcome"] == PASSED]
    return max(passed, key=lambda r: r["cells"]) if passed else None


def run_ramp(args, launch: Callable[[list, float, Path], Launched] = launch_child) -> int:
    """The ramp (or the explicit sizes), the soak, and the summary."""
    mesh_shape = parse_mesh(args.mesh)
    per_axis = devices_per_axis(mesh_shape)
    n_devices = mesh_shape[0] * mesh_shape[1]
    tile = tuple(args.tile)
    device_bytes = int(args.device_memory_gb * GIB) if args.device_memory_gb else None
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list = []
    notes: list = []
    stopped_by: dict | None = None

    def run(index, target, sized_from, k) -> dict:
        print(f"[capacity] --- rung {index}: K={tuple(k)}, {_cells(tile, k)} cells"
              + (f", target fill {target}" if target is not None else "")
              + f" ({sized_from}); time box {args.rung_timeout_s:g} s", flush=True)
        row = _run_one(args, launch, "rung", _row(index, target, sized_from, k, tile,
                                                  rung_file(index)), args.rung_timeout_s)
        rows.append(row)
        return row

    if args.k is not None:
        plan: list = [(None, "--k", tuple(args.k))]
    elif args.cells is not None:
        plan = [(None, f"--cells {n}", tiling_for_cells(n, tile, per_axis)) for n in args.cells]
    else:
        plan = [(target, None, None) for target in args.ramp]
    last = None
    for target, sized_from, k in plan:
        if k is None:
            if device_bytes is None:        # check_option_values refuses a ramp without it
                raise SystemExit("--ramp needs --device-memory-gb")
            if last is None:
                per_cell = args.bytes_per_cell or STATE_BYTES_PER_CELL * APRIORI_STEP_MULTIPLE
                want = first_rung_cells(target, device_bytes, n_devices, per_cell)
                sized_from = (f"a priori: {per_cell:g} bytes per cell"
                              + ("" if args.bytes_per_cell else
                                 f" = {STATE_BYTES_PER_CELL} of state x {APRIORI_STEP_MULTIPLE:g}"))
            elif last["fill"] is None:
                stopped_by = {"rung": None, "outcome": None, "detail": (
                    f"rung {last['rung']}'s fill was not measured, so the next rung cannot be "
                    "sized: give explicit sizes with --cells")}
                break
            else:
                want = next_rung_cells(target, last["fill"], last["cells"])
                if last["bytes_limit"] and target * device_bytes > last["bytes_limit"]:
                    notes.append(
                        f"target {target} is above the allocator's own limit, "
                        f"{last['bytes_limit'] / device_bytes:.3f} of the card (bytes_limit of "
                        f"rung {last['rung']}): expect out of memory there, whatever the card "
                        "holds; XLA_PYTHON_CLIENT_MEM_FRACTION raises it")
                sized_from = (f"rung {last['rung']} measured "
                              f"{last['peak_bytes_per_cell']:.1f} bytes per cell, fill "
                              f"{last['fill']:.3f}")
            k = tiling_for_cells(want, tile, per_axis)
            if last is not None and _cells(tile, k) <= last["cells"]:
                notes.append(f"target {target} skipped: rung {last['rung']} already filled "
                             f"{last['fill']:.3f}, and the next allowed grid is no larger")
                continue
        elif any(tuple(r["k"]) == tuple(k) for r in rows):
            notes.append(f"{sized_from} skipped: the same grid as an earlier rung")
            continue
        last = run(len(rows) + 1, target, sized_from, k)
        if last["outcome"] != PASSED:
            stopped_by = {"rung": last["rung"], "outcome": last["outcome"],
                          "cells": last["cells"], "target_fill": last["target_fill"],
                          "detail": last["detail"]}
            break
    ceiling = ceiling_of(rows)
    soak = None
    if args.soak_minutes > 0 and ceiling is not None and ramp_status(rows, None) == EXIT_OK:
        k = tiling_for_cells(int(args.soak_at * ceiling["cells"]), tile, per_axis)
        timeout_s = 60.0 * args.soak_minutes + 2 * args.rung_timeout_s
        print(f"[capacity] --- soak: K={k}, {_cells(tile, k)} cells ({args.soak_at:g} of the "
              f"ceiling's) for {args.soak_minutes:g} min; time box {timeout_s:g} s", flush=True)
        soak = _run_one(args, launch, "soak",
                        _row(0, None, f"{args.soak_at:g} of rung {ceiling['rung']}'s cells", k,
                             tile, SOAK_FILE), timeout_s)
        soak_record, _ = read_record(args.out / SOAK_FILE)
        soak["growth"] = None if soak_record is None else soak_record.get("growth")
    status = ramp_status(rows, soak)
    summary = {
        "schema_version": CAPACITY_SCHEMA_VERSION,
        "kind": SUMMARY_KIND,
        "gating": False,
        "dry_run": bool(args.dry_run),
        "git_commit": _git_commit(),
        "mesh": args.mesh,
        "tile": list(tile),
        "device_memory_gb": args.device_memory_gb,
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
        "rungs": rows,
        "notes": notes,
        "ceiling": ceiling,
        "stopped_by": stopped_by,
        "soak": soak,
        "exit_status": status,
    }
    write_json(args.out / SUMMARY_FILE, summary)
    print()
    _print_table(rows, soak, notes, stopped_by)
    print(f"wrote {args.out / SUMMARY_FILE}; exit {status}")
    return status


# ---------------------------------------------------------------------------
# The table, and --summarise
# ---------------------------------------------------------------------------


def _fmt(value, spec: str, missing: str = "-") -> str:
    return missing if value is None else format(value, spec)


def _print_table(rows: list, soak: dict | None, notes: list, stopped_by: dict | None) -> None:
    print("rung  target        cells  K            outcome                     fill   "
          "B/cell  steps/s   max_rel")
    for r in rows:
        print(f"{r['rung']:>4}  {_fmt(r['target_fill'], '6.2f'):>6}  {r['cells']:>11}  "
              f"{'x'.join(map(str, r['k'])):<11}  {r['outcome']:<25}  "
              f"{_fmt(r.get('fill'), '5.3f', 'n/m'):>5}  "
              f"{_fmt(r.get('peak_bytes_per_cell'), '6.1f', 'n/m'):>6}  "
              f"{_fmt(r.get('steps_per_second'), '7.2f'):>7}  "
              f"{_fmt(r.get('largest_max_rel'), '8.1e'):>8}")
    for note in notes:
        print(f"note: {note}")
    ceiling = ceiling_of(rows)
    if ceiling is None:
        print("CEILING: none -- no rung passed")
    else:
        fill = ceiling.get("fill")
        print(f"CEILING: rung {ceiling['rung']}, {ceiling['cells']} cells "
              f"(K = {'x'.join(map(str, ceiling['k']))}), fill "
              + (f"{fill:.3f}" if fill is not None else "not measured (n/m)"))
        limit, peak = ceiling.get("bytes_limit"), ceiling.get("peak_bytes")
        if limit and peak:
            print(f"  the allocator's own limit there: {limit} bytes "
                  f"(the peak was {peak / limit:.3f} of it)")
    if stopped_by is None:
        print("STOPPED BY: nothing -- every planned rung passed")
    elif stopped_by.get("rung") is None:
        print(f"STOPPED BY: {stopped_by['detail']}")
    else:
        first_line = (stopped_by.get("detail") or "").splitlines()[:1]
        print(f"STOPPED BY: rung {stopped_by['rung']} ({stopped_by['cells']} cells): "
              f"{stopped_by['outcome']}" + (f" -- {first_line[0]}" if first_line else ""))
    if soak is not None:
        growth = soak.get("growth")
        print(f"SOAK: {soak['cells']} cells: {soak['outcome']}"
              + (f"; {_growth_line(growth)}" if isinstance(growth, dict) else ""))


def summarise(directory: Path) -> int:
    """Print the table from the rung records under ``directory``, every
    verdict re-derived from the recorded values (:func:`rederive`)."""
    paths = sorted(directory.glob("capacity_rung_*.json"))
    if not paths:
        print(f"no capacity record (capacity_rung_*.json) under {directory}")
        return EXIT_NO_CEILING
    soak_path = directory / SOAK_FILE
    rows: list = []
    problems: list = []
    soak = None
    for path in paths + ([soak_path] if soak_path.exists() else []):
        doc, why = read_record(path)
        if doc is None:
            problems.append(f"{path.name}: cannot be read ({why})")
            continue
        try:
            derived = rederive(doc)
            memory = doc.get("memory") or {}
            fields = (doc.get("results") or {}).get("fields") or {}
            row = {"rung": int(doc.get("rung") or 0), "file": path.name,
                   "target_fill": doc.get("target_fill"), "k": list(doc["k"]),
                   "cells": _cells(doc["tile"], doc["k"]), "outcome": derived["outcome"],
                   "detail": doc.get("outcome_detail", ""), "fill": derived["fill"],
                   "peak_bytes": memory.get("peak_bytes"),
                   "bytes_limit": memory.get("bytes_limit"),
                   "peak_bytes_per_cell": memory.get("peak_bytes_per_cell"),
                   "steps_per_second": doc.get("steps_per_second"),
                   "largest_max_rel": max((float(e["max_rel"]) for e in fields.values()),
                                          default=None),
                   "growth": doc.get("growth")}
        except _UNREADABLE as exc:
            problems.append(f"{path.name}: cannot be read ({type(exc).__name__}: {exc})")
            continue
        problems += [f"{path.name}: {p}" for p in derived["problems"]]
        if doc["kind"] == SOAK_KIND:
            soak = row
        else:
            rows.append(row)
    rows.sort(key=lambda r: r["rung"])
    first_bad = next((r for r in rows if r["outcome"] != PASSED), None)
    stopped_by = None if first_bad is None else {
        "rung": first_bad["rung"], "cells": first_bad["cells"], "outcome": first_bad["outcome"],
        "detail": first_bad["detail"]}
    _print_table(rows, soak, [], stopped_by)
    if problems:
        print(f"\nRECORDS THAT DISAGREE WITH THEMSELVES ({len(problems)}); exit "
              f"{EXIT_RECORD_INVALID}:")
        for line in problems:
            print(f"  {line}")
        return EXIT_RECORD_INVALID
    return ramp_status(rows, soak)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    # No abbreviations: the CPU pin of a dry run reads the literal --dry-run
    # before the options are parsed.
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="directory for the records (its own: not the "
                                             "runner's --out)")
    ap.add_argument("--summarise", type=Path, metavar="DIR",
                    help="print the table from DIR, every verdict re-derived from the "
                         "recorded values, and exit (no JAX needed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU, four virtual devices; proves the script, measures no card")
    ap.add_argument("--mesh", default="2x2", metavar="AxB",
                    help="devices along spatial axis 0 x along axis 1: 2x2 (the pencil, "
                         "default) or 4x1 (the 1-D mesh)")
    ap.add_argument("--device-memory-gb", type=float, metavar="GIB",
                    help="one card's memory in GiB (24 for an RTX 4090); the fill is the peak "
                         "over this.  Required by --ramp")
    sizes = ap.add_mutually_exclusive_group()
    sizes.add_argument("--ramp", type=float, nargs="+", metavar="FILL",
                       help=f"target fills, rising (default {' '.join(map(str, DEFAULT_RAMP))} "
                            "when no size is given)")
    sizes.add_argument("--cells", type=int, nargs="+",
                       help="explicit sizes, one rung each: the largest allowed grid of at "
                            "most this many cells")
    sizes.add_argument("--k", type=int, nargs=3, metavar=("KX", "KY", "KZ"),
                       help="one rung of exactly this many tiles along each axis")
    ap.add_argument("--tile", type=int, nargs=3, default=list(DEFAULT_TILE),
                    metavar=("TX", "TY", "TZ"), help="the tile's cells along each axis")
    ap.add_argument("--steps", type=int, help="steps per rung and per soak block (default 10; "
                                              "dry-run 3)")
    ap.add_argument("--rung-timeout-s", type=float, help="a rung's time box (default 300; "
                                                         "dry-run 120)")
    ap.add_argument("--bytes-per-cell", type=float,
                    help="the first rung's peak bytes per cell, in place of the a-priori "
                         f"{STATE_BYTES_PER_CELL} x {APRIORI_STEP_MULTIPLE:g}")
    ap.add_argument("--soak-minutes", type=float, default=0.0,
                    help="after the ramp, repeat build, steps and comparison for this long "
                         "(default 0: no soak)")
    ap.add_argument("--soak-at", type=float, default=0.75,
                    help="the soak's size, as a share of the ceiling's cells")
    ap.add_argument("--check-chunk-mb", type=float, default=128.0,
                    help="most state bytes one comparison chunk holds per device (at least "
                         "one plane)")
    # A rung's own process (started by the ramp, not by hand).
    ap.add_argument("--child", choices=("rung", "soak"), help=argparse.SUPPRESS)
    ap.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--rung-index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--target-fill", type=float, help=argparse.SUPPRESS)
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = _parser()
    args = ap.parse_args(argv)
    if args.summarise is not None:
        return args
    if args.child is not None:
        if args.record is None or args.k is None:
            ap.error("--child needs --record and --k (it is started by the ramp)")
    elif args.out is None:
        ap.error("--out is required unless --summarise is given")
    return args


def check_option_values(args: argparse.Namespace) -> None:
    """Refuse (``SystemExit`` with the reason: ``EXIT_REFUSED``) what no
    rung can use, before anything runs."""
    try:
        mesh_shape = parse_mesh(args.mesh)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    per_axis = devices_per_axis(mesh_shape)
    tile = tuple(args.tile)
    for option, value, least in (("--steps", args.steps, 1), ("--tile", min(tile), 1),
                                 ("--rung-index", args.rung_index, 0)):
        if value is not None and value < least:
            raise SystemExit(f"{option} {value}: must be an integer >= {least}")
    for option, value in (("--device-memory-gb", args.device_memory_gb),
                          ("--rung-timeout-s", args.rung_timeout_s),
                          ("--bytes-per-cell", args.bytes_per_cell),
                          ("--check-chunk-mb", args.check_chunk_mb)):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise SystemExit(f"{option} {value}: must be a finite number > 0")
    if not (math.isfinite(args.soak_minutes) and args.soak_minutes >= 0):
        raise SystemExit(f"--soak-minutes {args.soak_minutes}: must be a finite number >= 0")
    if not 0 < args.soak_at <= 1:
        raise SystemExit(f"--soak-at {args.soak_at}: must be in (0, 1]")
    problem = tiling_problem(tile, args.k if args.k is not None else (1, 1, 1), per_axis)
    if problem is not None:
        raise SystemExit(f"--tile {' '.join(map(str, tile))}"
                         + (f" --k {' '.join(map(str, args.k))}" if args.k is not None else "")
                         + f" on mesh {args.mesh}: {problem}")
    if args.child is not None:
        return
    if args.cells is not None:
        for n in args.cells:
            try:
                tiling_for_cells(n, tile, per_axis)
            except ValueError as exc:
                raise SystemExit(f"--cells {n}: {exc}") from None
    if args.k is None and args.cells is None:
        if args.ramp is None:
            args.ramp = list(DEFAULT_RAMP)
        if args.device_memory_gb is None:
            raise SystemExit("--ramp needs --device-memory-gb, one card's memory in GiB (24 "
                             "for an RTX 4090); on CPU give explicit sizes with --cells")
        if any(not (math.isfinite(t) and 0 < t <= MAX_TARGET_FILL) for t in args.ramp):
            raise SystemExit(f"--ramp {args.ramp}: every target must be in (0, "
                             f"{MAX_TARGET_FILL}]")
        if any(b <= a for a, b in zip(args.ramp, args.ramp[1:])):
            raise SystemExit(f"--ramp {args.ramp}: the targets must rise")
        per_cell = args.bytes_per_cell or STATE_BYTES_PER_CELL * APRIORI_STEP_MULTIPLE
        first = first_rung_cells(args.ramp[0], int(args.device_memory_gb * GIB),
                                 mesh_shape[0] * mesh_shape[1], per_cell)
        try:
            tiling_for_cells(first, tile, per_axis)
        except ValueError as exc:
            raise SystemExit(f"--ramp {args.ramp[0]} of --device-memory-gb "
                             f"{args.device_memory_gb}: {exc}") from None
    if args.out.exists() and not args.out.is_dir():
        raise SystemExit(f"--out {args.out}: exists and is not a directory")
    if args.out.is_dir():
        goal_files = sorted(p.name for goal in RUNNER_GOALS for p in args.out.glob(f"{goal}*.json"))
        if goal_files:
            raise SystemExit(f"--out {args.out}: holds the runner's goal files "
                             f"({', '.join(goal_files[:4])}).  The capacity test is not part "
                             "of the checklist: give it a directory of its own")
        earlier = sorted(p.name for p in args.out.glob("capacity_*.json"))
        if earlier:
            raise SystemExit(f"--out {args.out}: holds the records of an earlier run "
                             f"({', '.join(earlier[:4])}); a summary of the two together would "
                             "be of neither.  Give this run a directory of its own")


def _terminated(signum, frame):
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None,
         launch: Callable[[list, float, Path], Launched] = launch_child) -> int:
    args = parse_args(argv)
    if args.summarise is not None:
        return summarise(args.summarise)
    check_option_values(args)
    if args.steps is None:
        args.steps = 3 if args.dry_run else 10
    if args.rung_timeout_s is None:
        args.rung_timeout_s = 120.0 if args.dry_run else 300.0
    if args.child is not None:
        return child_main(args)
    # ``timeout`` ends this process with SIGTERM: leave as for Ctrl-C, through
    # the launcher's clean-up, so the rung in flight ends with the ramp.
    previous = signal.signal(signal.SIGTERM, _terminated)
    try:
        return run_ramp(args, launch)
    finally:
        signal.signal(signal.SIGTERM, previous)


def exit_status(argv: list[str] | None = None,
                launch: Callable[[list, float, Path], Launched] = launch_child) -> int:
    """:func:`main`'s status, with a refusal and a crash of this script told
    apart from a failed check: a refusal -- ``SystemExit`` with a message --
    prints it and is ``EXIT_REFUSED``; any other exception prints its
    traceback and is ``EXIT_CRASHED``."""
    try:
        return main(argv, launch)
    except SystemExit as exc:
        if exc.code is None or isinstance(exc.code, int):
            raise                       # argparse's 2, an explicit status
        print(exc.code, file=sys.stderr)
        return EXIT_REFUSED
    except Exception:  # noqa: BLE001 - every crash gets the one status
        traceback.print_exc()
        print(f"CRASHED: the capacity test itself raised (exit {EXIT_CRASHED})", file=sys.stderr)
        return EXIT_CRASHED


if __name__ == "__main__":
    sys.exit(exit_status())
