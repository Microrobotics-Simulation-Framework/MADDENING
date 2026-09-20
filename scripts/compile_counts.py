#!/usr/bin/env python
"""Measure and gate MADDENING's deterministic compilation counts.

MADDENING's pitch is that the whole graph compiles to one jitted step,
and compile time is what a user feels first -- but until now nothing
stopped a silent 3x compile-time regression from shipping.  A wall-clock
gate is not the answer: ``test_step_cost_does_not_scale_with_max_iterations``
compared per-step cost at two iteration caps with a 3x bound on numbers
around 1e-4 s, and failed CI at 3.27x on a pull request that changed 122
documentation files and no code at all.  The number it read was the
runner, not the solver.

So this gate reads integers instead:

retrace count
    How many Python traces -- and therefore XLA compilations -- a
    workload's step takes.  The single most valuable number here.  An
    unexpected retrace is the classic silent regression (a weak-typed
    leaf, a dtype that drifts on the second step, an argument that
    stopped being static), it costs a full compile every time, and it is
    exactly reproducible.
jaxpr primitive count
    Primitives in the step's jaxpr, recursing into ``scan`` / ``while`` /
    ``cond`` bodies.  How much work the graph builder emits.
HLO op count
    Operations in the lowered StableHLO module -- what MADDENING hands
    to XLA, before XLA optimises it.  Deliberately pre-optimisation: the
    post-fusion count is the backend's business and differs by platform,
    which would make the baseline machine-dependent.

None of these move when the box is busy, which is the whole point.

Usage
-----
::

    python scripts/compile_counts.py            # rewrite the baseline
    python scripts/compile_counts.py --check    # fail on drift
    python scripts/compile_counts.py --show     # print, write nothing

``--check`` is what ``tests/core/test_compile_counts.py`` runs, so CI
fails on a regression instead of shipping one.  When it fails it prints
a table of every changed count and names the command above.

Pinned environment
------------------
The counts depend on the backend and the device count, so this script
pins both (``JAX_PLATFORMS=cpu``, four virtual host devices) before
importing JAX, **overriding whatever the caller had set** -- an inherited
``--xla_force_host_platform_device_count`` is stripped, not respected
(:func:`pin_device_count`).  That is why the pytest gate runs it as a
*subprocess*: JAX fixes its device count when its backend initialises, so
a process that has already imported JAX cannot change it.

Overriding rather than deferring is what makes the baseline reproducible
on any machine, and it is not hypothetical.  ``tests/cloud/multigpu/
conftest.py`` appends ``--xla_force_host_platform_device_count=16`` to
``os.environ`` at *import* time, so in a whole-suite run -- which is what
CI does -- every later subprocess inherits 16 virtual devices, while
running this file's tests alone inherits none.  An earlier revision took
the caller's flag when one was present, which made the gate's device
count a function of which tests happened to be collected alongside it.
The JAX version is recorded in the baseline and drives how tight the
op-count bands are (see ``TOLERANCES``).

Regenerate under the JAX version CI pins (``.github/workflows/ci.yml``)
when you can.  The committed baseline records the version it was taken
on, and a baseline whose version matches CI's gets CI's op-count checks
at their tightest; one taken on a different version still gates, on the
wider band.  The counts themselves were identical on 0.10.2 and 0.11.0,
so this costs nothing but strictness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must precede any JAX import -- see "Pinned environment" above.
DEVICE_COUNT = 4
HOST_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def pin_device_count(xla_flags: str, count: int = DEVICE_COUNT) -> str:
    """*xla_flags* with the host-device-count flag forced to *count*.

    Every inherited setting of the flag is dropped -- in both the
    ``--flag=N`` and ``--flag N`` spellings absl accepts -- and a single
    ``--flag=count`` is appended.  The caller's other XLA flags are kept,
    in order: ``--xla_gpu_autotune_level=0`` in CI is none of this
    script's business, the device count is.

    Kept pure, and separate from the assignment below, so the pinning can
    be tested against the exact string CI inherits without spawning a
    process or importing JAX.

    Parameters
    ----------
    xla_flags : str
        The inherited ``XLA_FLAGS`` value; may be empty.
    count : int, optional
        Virtual host devices to force.  Defaults to :data:`DEVICE_COUNT`.

    Returns
    -------
    str
        The value to export as ``XLA_FLAGS``.
    """
    kept: list[str] = []
    tokens = xla_flags.split()
    i = 0
    while i < len(tokens):
        if tokens[i] == HOST_DEVICE_FLAG:            # "--flag N"
            i += 2
        elif tokens[i].startswith(f"{HOST_DEVICE_FLAG}="):   # "--flag=N"
            i += 1
        else:
            kept.append(tokens[i])
            i += 1
    return " ".join([*kept, f"{HOST_DEVICE_FLAG}={count}"])


os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = pin_device_count(os.environ.get("XLA_FLAGS", ""))

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402  (after the environment is pinned)
import jaxlib  # noqa: E402

from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: E402
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.core.simulation.profiler import compile_counts  # noqa: E402
from maddening.nodes.heat import HeatNode  # noqa: E402
from maddening.nodes.spring import SpringDamperNode  # noqa: E402

BASELINE = REPO_ROOT / "benchmarks" / "compile_counts_baseline.json"
REGENERATE = "python scripts/compile_counts.py"

# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
#
# Five shapes, chosen so that each one is the *only* member of the set
# that can see some class of regression:
#
#   single_spring     the step machinery with essentially no physics in
#                     it -- ``_meta``, params threading, state
#                     normalisation, the jit wrapper.  Eighteen jaxpr
#                     primitives, so a handful of ops added to the
#                     always-on plumbing is a visible fraction here and
#                     noise anywhere else.
#   coupled_pair      the fixed-point ``while`` solver: the most complex
#                     and most frequently changed machinery in the
#                     repository, and the one whose op count is largest.
#   multirate_coupled the rate-divider ``cond`` path and the step
#                     counter; the only shape where a node's update runs
#                     under a ``cond``.
#   heat_chain        array-valued state, a stencil, and an edge
#                     transform -- plus the only ``run_scan`` program in
#                     the set, where a per-step op regression multiplies
#                     by the scan length.
#   sharded_heat      ``ShardedStencilNode`` over a four-device mesh:
#                     the halo exchange and its collectives.  A retrace
#                     or an extra collective here is expensive and
#                     invisible in the other four.
#
# Deliberately not more.  Every change to the coupling solver moves
# coupled_pair, multirate_coupled and sharded_heat at once, so the
# baseline already churns on the busiest area of the code; near-duplicate
# shapes would multiply that churn without covering anything new, and a
# baseline people regenerate without reading is a baseline that gates
# nothing.  Deliberately not fewer: drop any one of the five and a whole
# code path -- plumbing, coupling, multi-rate, scan, sharding -- stops
# being measured.


def _single_spring() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=30.0, damping=2.0, initial_position=0.5,
    ))
    gm.compile()
    return gm


def _coupled_pair() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "a", 0.01, stiffness=30.0, damping=2.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        "b", 0.01, stiffness=30.0, damping=2.0, initial_position=3.0,
    ))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=25, tolerance=1e-8)
    gm.compile()
    return gm


def _multirate_coupled() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=3.0))
    gm.add_node(SpringDamperNode("slow", 0.02, initial_position=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_edge("b", "slow", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=15, tolerance=1e-8)
    gm.compile()
    return gm


def _heat_chain() -> GraphManager:
    gm = GraphManager()
    for i in range(2):
        gm.add_node(HeatNode(
            f"h{i}", 1e-4, n_cells=64, thermal_diffusivity=0.1, stencil_order=4,
        ))
    gm.add_edge("h0", "h1", "temperature", "left_temperature",
                transform="extract_last")
    gm.compile()
    return gm


def _sharded_heat() -> GraphManager:
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(
        HeatNode("h", 1e-4, n_cells=64, thermal_diffusivity=0.1, stencil_order=4),
        create_device_mesh(shape=(DEVICE_COUNT,)),
        axis_map={"devices": 0},
        boundary="edge",
    ))
    gm.compile()
    return gm


#: ``name -> (builder, scan_steps)``.  ``scan_steps`` of 0 means the
#: workload measures the step only.  Only ``heat_chain`` builds a scan:
#: one is enough to pin the ``run_scan`` program's structure, and each
#: extra one costs a full scan compile on every CI run.
WORKLOADS: dict[str, tuple] = {
    "single_spring": (_single_spring, 0),
    "coupled_pair": (_coupled_pair, 0),
    "multirate_coupled": (_multirate_coupled, 0),
    "heat_chain": (_heat_chain, 16),
    "sharded_heat": (_sharded_heat, 0),
}

# ---------------------------------------------------------------------------
# Gating policy
# ---------------------------------------------------------------------------
#
# Exact on the retrace counts, banded on the op counts.  The asymmetry is
# not a compromise, it is what the two kinds of number deserve:
#
# A retrace count is produced by counting Python calls
# (``GraphManager._counted_step``).  It does not depend on the JAX
# version, the backend, the optimiser or the machine.  Measured on JAX
# 0.10.2 and 0.11.0 it is 1 for every workload in this file.  An exact
# gate on a genuinely exact quantity is the strongest gate available and
# costs nothing in false alarms.
#
# Op counts are produced by JAX's own tracing and lowering, so they can
# move on a JAX upgrade without anything in this repository changing.
# CI pins ``jax==0.10.2`` while ``pyproject.toml`` allows ``>=0.10,<0.13``,
# so the two-version case is not hypothetical -- it is today.  A gate that
# fires on every JAX bump is a gate people learn to bypass, so the band
# widens when the running JAX differs from the baseline's:
#
#   same JAX version   +/- max(2 ops, 2%)
#   different version  +/- max(10 ops, 25%)
#
# Both widths were set against a measurement rather than a guess.  Every
# count in this file was taken twice, once on JAX 0.10.2 (what CI pins)
# and once on JAX 0.11.0, and all thirty-four numbers came out
# *identical* -- retraces, jaxpr primitives and HLO ops, on all five
# workloads.  So:
#
#   * The same-version band is not absorbing any known variation.  It
#     exists because CI runs Python 3.11 and 3.12 and only 3.12 was
#     measured here; two ops of slack covers an interpreter difference
#     that would otherwise wedge one matrix leg, and two ops cannot hide
#     any regression this gate is for.
#   * The cross-version band is deliberately generous relative to that
#     evidence.  0.10 -> 0.11 moved nothing, but 0.12 is inside the
#     supported range and untested, and the cost of being wrong in that
#     direction is a red CI on an unrelated dependency bump.  A quarter
#     still catches the 3x regression this gate exists for.
#
# Two-sided in both cases.  A large *drop* is as much a finding as a
# rise: it usually means a node stopped running, not that something got
# faster, and either way the committed baseline is then wrong.
TOLERANCES = {
    "same_version": {"abs": 2, "rel": 0.02},
    "other_version": {"abs": 10, "rel": 0.25},
}

#: Counts compared exactly, whatever the JAX version.
EXACT_FIELDS = ("retrace_count", "scan_retrace_count", "scan_steps")


def _jax_minor(version: str) -> str:
    """``"0.10.2"`` -> ``"0.10"``.  Patch releases share a band."""
    return ".".join(version.split(".")[:2])


def measure() -> dict:
    """Measure every workload and return the baseline document."""
    devices = jax.device_count()
    if devices != DEVICE_COUNT:
        # The pin above should make this unreachable, which is why it is
        # worth keeping: reaching it means the pin did not take, and the
        # numbers below would be silently incomparable.  The counts for
        # sharded_heat depend on the mesh size.
        raise SystemExit(
            f"expected {DEVICE_COUNT} JAX devices, got {devices}: "
            f"XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}.  This script pins "
            f"the device count before importing JAX, so a mismatch means "
            f"the pin did not take -- JAX was already imported in this "
            f"process (it fixes the device count when its backend "
            f"initialises), or the backend ignored the flag.  Run the gate "
            f"as its own process: `{REGENERATE}`."
        )
    workloads = {}
    for name, (build, scan_steps) in WORKLOADS.items():
        # ``compile_counts`` warms up for four steps by default, which is
        # what makes a retrace on the second or third step visible here;
        # see its docstring for why one step would not be enough.
        workloads[name] = compile_counts(build(), scan_steps=scan_steps).as_dict()
    return {
        "_comment": (
            "Deterministic compilation counts. Regenerate with "
            f"`{REGENERATE}`; see the script's docstring for what each "
            "count means and why the gate is exact on retraces and banded "
            "on op counts."
        ),
        "environment": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "platform": "cpu",
            # The measurement, not the constant: the file should state
            # what was actually there, so ``compare`` is checking a fact.
            "device_count": devices,
        },
        "workloads": workloads,
    }


def _band(baseline_value: int, same_version: bool) -> int:
    tol = TOLERANCES["same_version" if same_version else "other_version"]
    return max(tol["abs"], int(round(tol["rel"] * abs(baseline_value))))


def compare(baseline: dict, fresh: dict) -> list[str]:
    """Problems found comparing *fresh* counts against *baseline*.

    An empty list means the gate passes.
    """
    problems: list[str] = []
    base_env = baseline.get("environment", {})
    base_jax = str(base_env.get("jax", ""))
    same_version = _jax_minor(base_jax) == _jax_minor(fresh["environment"]["jax"])

    # Topology is not banded and never widens: sharded_heat's counts are a
    # function of the mesh size, so counts taken on a different device
    # count are not comparable at any tolerance.  Fails closed -- a
    # baseline predating this field has no device_count and is rejected
    # rather than assumed to match.
    base_devices = base_env.get("device_count")
    fresh_devices = fresh["environment"]["device_count"]
    if base_devices != fresh_devices:
        problems.append(
            f"device count: baseline {base_devices!r} -> this run "
            f"{fresh_devices!r}.  sharded_heat's counts depend on the mesh "
            f"size, so the two sets of numbers are not comparable; "
            f"regenerate with `{REGENERATE}`"
        )

    base_wl = baseline.get("workloads", {})
    fresh_wl = fresh["workloads"]

    for name in sorted(set(base_wl) ^ set(fresh_wl)):
        side = "the baseline" if name in base_wl else "this script"
        problems.append(
            f"workload {name!r} exists only in {side} -- the baseline and "
            f"WORKLOADS have diverged"
        )

    for name in sorted(set(base_wl) & set(fresh_wl)):
        want, got = base_wl[name], fresh_wl[name]
        # Fail closed on an unrecognised field rather than ignoring it:
        # a count added to CompileCounts must be gated or removed from
        # the baseline, never silently carried along unchecked.
        for field in sorted(set(want) | set(got)):
            if field not in want or field not in got:
                problems.append(
                    f"{name}.{field}: present in "
                    f"{'the baseline' if field in want else 'the measurement'} "
                    f"only -- regenerate the baseline"
                )
                continue
            if field in EXACT_FIELDS:
                if int(got[field]) != int(want[field]):
                    extra = ""
                    if field == "retrace_count" and int(got[field]) > int(want[field]):
                        extra = (
                            "  An extra retrace is an extra XLA compile on "
                            "every run: look for a leaf whose dtype or weak "
                            "type changes after the first step, or an "
                            "argument that stopped being static."
                        )
                    problems.append(
                        f"{name}.{field}: {want[field]} -> {got[field]} "
                        f"(exact match required).{extra}"
                    )
                continue
            band = _band(int(want[field]), same_version)
            delta = int(got[field]) - int(want[field])
            if abs(delta) > band:
                pct = 100.0 * delta / max(1, int(want[field]))
                problems.append(
                    f"{name}.{field}: {want[field]} -> {got[field]} "
                    f"({delta:+d}, {pct:+.1f}%), outside the "
                    f"+/-{band} band for "
                    f"{'this' if same_version else 'a different'} JAX version"
                )

    if problems and not same_version:
        problems.append(
            f"(the baseline was taken on JAX {base_jax}, this run is on "
            f"JAX {fresh['environment']['jax']}; op-count bands were widened "
            f"accordingly, retrace counts never are)"
        )
    return problems


def _render(doc: dict) -> str:
    lines = [
        f"JAX {doc['environment']['jax']} / jaxlib "
        f"{doc['environment']['jaxlib']}, {doc['environment']['platform']}, "
        f"{doc['environment']['device_count']} devices",
        "",
        f"{'workload':<20}{'retrace':>9}{'jaxpr':>9}{'hlo':>9}"
        f"{'scan':>7}{'s-retr':>8}{'s-jaxpr':>9}{'s-hlo':>8}",
    ]
    for name, c in doc["workloads"].items():
        lines.append(
            f"{name:<20}{c['retrace_count']:>9}"
            f"{c['jaxpr_primitive_count']:>9}{c['hlo_op_count']:>9}"
            f"{c.get('scan_steps', ''):>7}{c.get('scan_retrace_count', ''):>8}"
            f"{c.get('scan_jaxpr_primitive_count', ''):>9}"
            f"{c.get('scan_hlo_op_count', ''):>8}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the measured counts drift from the committed "
             "baseline, printing what moved; write nothing",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="print the measured counts and write nothing",
    )
    parser.add_argument(
        "--baseline", type=Path, default=BASELINE,
        help="baseline file to check against or write (default: the "
             "committed one).  Not a way to weaken the gate -- it exists so "
             "the gate's own mutation test can point --check at a "
             "deliberately wrong baseline and prove it fails.",
    )
    args = parser.parse_args(argv)
    baseline_path: Path = args.baseline
    try:
        shown_path = baseline_path.relative_to(REPO_ROOT)
    except ValueError:
        shown_path = baseline_path

    fresh = measure()

    if args.show:
        print(_render(fresh))
        return 0

    if args.check:
        if not baseline_path.exists():
            print(
                f"{shown_path} is missing -- create it with `{REGENERATE}`",
                file=sys.stderr,
            )
            return 1
        baseline = json.loads(baseline_path.read_text())
        problems = compare(baseline, fresh)
        if problems:
            print(f"Compilation counts drifted from {shown_path}:\n",
                  file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print(
                f"\nBaseline:\n{_render(baseline)}\n\nMeasured:\n"
                f"{_render(fresh)}\n\nIf the change is intended, regenerate "
                f"with `{REGENERATE}` and commit the diff, saying in the "
                f"commit message why the counts moved.",
                file=sys.stderr,
            )
            return 1
        print(f"{shown_path}: counts match.")
        return 0

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(fresh, indent=2) + "\n")
    print(f"wrote {shown_path}\n")
    print(_render(fresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
