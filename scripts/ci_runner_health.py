#!/usr/bin/env python3
"""What a CI runner is, how it is doing, and whether it is fit to run tests.

Both test lanes (``ci.yml``'s ``test`` job and ``slow-tests.yml``'s ``slow``
job) call this script from steps of their own.  Standard library only,
except the canary, which imports NumPy and JAX the way the tests do.

``machine``
    Logs the CPU model, the vCPU count and the memory, and starts the clock
    the canary reads the install duration from.  Run it right before the
    install steps.
``canary``
    Runs after the install steps.  Times a fixed CPU micro-benchmark (a
    NumPy matmul and a small ``jax.jit`` compile) and reads how long the
    install took, and fails the job at once, with exit code 75, when either
    is far outside what any runner has been seen to do (see "Thresholds").
    A runner that slow would take hours to finish its share and then be
    reported as a timeout, which says nothing about the code.
``before-tests`` / ``after-tests``
    Bracket the test step.  They log the memory available at each end and
    the CPU *steal time* between them (``/proc/stat``: the time the
    hypervisor gave this VM's CPUs to another guest), and ``after-tests``
    writes the sidecar JSON that goes into the durations artifact, so that
    ``scripts/slow_lane_weights.py`` and ``scripts/ci_variance_report.py``
    can read the CPU a shard's times were measured on.

Degraded runners
----------------
GitHub-hosted runners are drawn from a pool of several CPU generations
(about 1.5x apart on this suite), and now and then from a machine that is
far worse than any of them: on 2026-10-13 one per-push job ran the same
tests 3.8x slower than its sibling, and a slow-lane runner took 9 minutes
to ``pip install`` (about 1 normally) and 366 s to collect (34 s on
another), every test about 10x slower.  The canary and the install time
catch that before the tests start.  On a trip the script:

* prints ``::error title=Degraded runner::<numbers>``;
* appends a line to the step summary;
* writes the marker file (``degraded-runner.json``) that the workflow
  uploads as the artifact ``degraded-runner-<job id>``;
* sets the step output ``degraded=true``;
* exits with code ``75`` (EX_TEMPFAIL), which no other step here uses.

``.github/workflows/rerun-degraded.yml`` re-runs such a job once.

Thresholds
----------
``CANARY_LIMIT_S`` and ``INSTALL_LIMIT_S`` below, each with the data it was
derived from, by the rule *more than 3x the slowest CPU model's median*.
When the runner image or the canary changes, re-measure (a throwaway
branch that runs ``machine``, the install and ``canary`` in a matrix of
twenty jobs does it in three minutes) and change both here.  Both can be overridden by environment
(``MADDENING_CANARY_LIMIT_S``, ``MADDENING_INSTALL_LIMIT_S``) for a test.

``MADDENING_FORCE_DEGRADED`` (``canary`` or ``install``, anything else
non-empty meaning both) makes the matching check trip whatever the
numbers, to test the detection and the re-run on a throwaway branch.
Leave it unset everywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

#: Exit code of a tripped canary (EX_TEMPFAIL): distinct from pytest's 0-5,
#: from ``timeout``'s 124 and from a signal's 128+n.
DEGRADED_EXIT = 75

#: The canary's score is its NumPy and JAX-compile seconds.  Limit: three
#: times the median of the slowest CPU model's runs.  Measured 2026-10-14 on
#: 40 sampler jobs (benchmarks/results/050_p0_3b_ci/thresholds.md): the
#: slowest model, AMD EPYC 7763, scored a median of 2.83 s (max 3.09 s over
#: 18 jobs; the fastest model, EPYC 9V45, 1.43 s), so 3 x 2.83 = 8.5 s.
CANARY_LIMIT_S = 8.5
#: The seconds from ``machine`` to ``canary`` (``pip install`` of JAX and
#: the extras; Python's own setup is before it).  Same rule: the slowest
#: model's median install was 68 s (EPYC 7763; 40 sampler jobs), so
#: 3 x 68 = 204 s.  Over the 653 install steps of the last 70 runs of both
#: lanes the median is 62 s, the 99th percentile 93 s and the maximum
#: 131 s: the limit is 1.6x above the worst healthy install seen.  The
#: degraded runner of 2026-10-13 took about 540 s.
INSTALL_LIMIT_S = 204.0

#: Linux's USER_HZ, the unit of /proc/stat.  Fixed at 100 on every
#: platform GitHub offers.
USER_HZ = 100

#: Matmul size and repeats, and the shape of the compiled program.  Fixed:
#: a limit means nothing if the benchmark moves.
MATMUL_N = 2200
MATMUL_REPS = 4
JIT_LAYERS = 200
JIT_WIDTH = 96


# ---------------------------------------------------------------------------
# Reading the machine (pure functions on text, so a test can feed them)
# ---------------------------------------------------------------------------

def cpu_model(cpuinfo: str) -> str:
    """The first ``model name`` of /proc/cpuinfo, or ``unknown``."""
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip() or "unknown"
    return "unknown"


def vcpu_count(cpuinfo: str) -> int:
    return sum(1 for line in cpuinfo.splitlines() if line.startswith("processor"))


def meminfo(text: str) -> dict[str, int]:
    """``{total_kib, available_kib}`` from /proc/meminfo."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            out[{"MemTotal": "total_kib", "MemAvailable": "available_kib"}[key]] = int(rest.split()[0])
    missing = {"total_kib", "available_kib"} - set(out)
    if missing:
        raise ValueError(f"/proc/meminfo has no {sorted(missing)}")
    return out


def cpu_ticks(proc_stat: str) -> dict[str, int]:
    """``{busy_and_idle, steal}`` ticks of the aggregate ``cpu`` line of /proc/stat.

    The columns are user, nice, system, idle, iowait, irq, softirq, steal,
    guest, guest_nice; guest time is already inside user, so the total is
    the first eight.
    """
    for line in proc_stat.splitlines():
        if line.startswith("cpu "):
            fields = [int(x) for x in line.split()[1:]]
            fields += [0] * (8 - len(fields))   # a kernel older than 2.6.11 has no steal
            return {"total": sum(fields[:8]), "steal": fields[7]}
    raise ValueError("/proc/stat has no aggregate cpu line")


def steal_between(before: dict[str, int], after: dict[str, int], hz: int = USER_HZ) -> dict[str, float]:
    """The steal seconds, the CPU seconds and the steal share between two ``cpu_ticks``."""
    total = after["total"] - before["total"]
    steal = after["steal"] - before["steal"]
    return {"steal_s": steal / hz, "cpu_s": total / hz,
            "steal_pct": (100.0 * steal / total) if total > 0 else 0.0}


def _gib(kib: int) -> str:
    return f"{kib / 1048576:.1f} GiB"


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def _limit(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    return float(raw) if raw else default


def degraded_reasons(canary_s: float | None, install_s: float | None,
                     canary_limit: float, install_limit: float, force: str = "") -> list[str]:
    """Why this runner is degraded: one sentence per tripped check, empty when it is not.

    ``force`` is ``canary``, ``install`` or anything else non-empty for both.
    """
    force = force.strip().lower()
    force_canary = force in ("canary",) or (bool(force) and force != "install")
    force_install = force in ("install",) or (bool(force) and force != "canary")
    reasons = []
    if force_canary:
        reasons.append(f"the canary check was forced by MADDENING_FORCE_DEGRADED={force!r}")
    elif canary_s is not None and canary_s > canary_limit:
        reasons.append(f"the CPU canary took {canary_s:.1f} s (limit {canary_limit:.1f} s)")
    if force_install:
        reasons.append(f"the install check was forced by MADDENING_FORCE_DEGRADED={force!r}")
    elif install_s is not None and install_s > install_limit:
        reasons.append(f"the install took {install_s:.0f} s (limit {install_limit:.0f} s)")
    return reasons


# ---------------------------------------------------------------------------
# State shared between the steps of one job
# ---------------------------------------------------------------------------

def state_path() -> Path:
    explicit = os.environ.get("MADDENING_RUNNER_STATE")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "maddening-runner-health.json"


def load_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    state_path().write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


#: Where the kernel's files are; a test points it at a directory of its own.
PROC_ROOT = "/proc"


def _proc(name: str) -> str:
    return (Path(PROC_ROOT) / name).read_text(encoding="utf-8", errors="replace")


def _append(env_var: str, text: str) -> None:
    target = os.environ.get(env_var)
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)


def marker_path() -> Path:
    explicit = os.environ.get("MADDENING_DEGRADED_MARKER")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "degraded-runner.json"


# ---------------------------------------------------------------------------
# The canary
# ---------------------------------------------------------------------------

def run_canary() -> dict[str, float]:
    """Time the micro-benchmark: ``{import_s, matmul_s, jit_s, score_s}``.

    ``score_s`` (matmul plus compile) is what the limit is on; the import
    time is logged because a runner with a slow disk shows it first.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    t = time.perf_counter()
    import numpy as np
    import jax
    import jax.numpy as jnp
    import_s = time.perf_counter() - t

    rng = np.random.default_rng(0)
    a = rng.standard_normal((MATMUL_N, MATMUL_N))
    b = rng.standard_normal((MATMUL_N, MATMUL_N))
    a @ b                                    # touch the pages and the BLAS threads once
    t = time.perf_counter()
    for _ in range(MATMUL_REPS):
        a @ b
    matmul_s = time.perf_counter() - t

    w = jnp.asarray(rng.standard_normal((JIT_WIDTH, JIT_WIDTH)) / JIT_WIDTH ** 0.5, dtype=jnp.float32)

    def program(x):
        for i in range(JIT_LAYERS):
            x = jnp.tanh(x @ w + 0.01 * i)
        return x.sum()

    x = jnp.ones((JIT_WIDTH, JIT_WIDTH), dtype=jnp.float32)
    t = time.perf_counter()
    jax.jit(program).lower(x).compile()
    jit_s = time.perf_counter() - t
    return {"import_s": import_s, "matmul_s": matmul_s, "jit_s": jit_s, "score_s": matmul_s + jit_s}


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_machine(_args) -> int:
    cpuinfo, mem = _proc("cpuinfo"), meminfo(_proc("meminfo"))
    state = load_state()
    state.update(cpu_model=cpu_model(cpuinfo), vcpus=vcpu_count(cpuinfo),
                 mem_total_kib=mem["total_kib"], marks={"install_start": time.time()})
    save_state(state)
    print(f"runner: cpu '{state['cpu_model']}', {state['vcpus']} vCPU, "
          f"memory {_gib(mem['total_kib'])} total, {_gib(mem['available_kib'])} available")
    return 0


def cmd_canary(args) -> int:
    state = load_state()
    started = state.get("marks", {}).get("install_start")
    install_s = (time.time() - started) if started else None
    result = run_canary()
    canary_limit = _limit("MADDENING_CANARY_LIMIT_S", CANARY_LIMIT_S)
    install_limit = _limit("MADDENING_INSTALL_LIMIT_S", INSTALL_LIMIT_S)
    state["canary"] = {**{k: round(v, 3) for k, v in result.items()},
                       "install_s": None if install_s is None else round(install_s, 1),
                       "canary_limit_s": canary_limit, "install_limit_s": install_limit}
    save_state(state)
    install_text = "install not timed" if install_s is None else f"install {install_s:.0f} s (limit {install_limit:.0f} s)"
    line = (f"runner: canary numpy matmul {result['matmul_s']:.2f} s + jit compile {result['jit_s']:.2f} s "
            f"= {result['score_s']:.2f} s (limit {canary_limit:.1f} s), import {result['import_s']:.2f} s; {install_text}")
    print(line)
    reasons = degraded_reasons(result["score_s"], install_s, canary_limit, install_limit,
                               os.environ.get("MADDENING_FORCE_DEGRADED", ""))
    label = args.label or "job"
    if not reasons:
        _append("GITHUB_STEP_SUMMARY", f"- {label}: runner healthy; {line.removeprefix('runner: ')}\n")
        return 0
    message = (f"{'; '.join(reasons)}.  cpu '{state.get('cpu_model', 'unknown')}', {state.get('vcpus', '?')} vCPU.  "
               f"{line.removeprefix('runner: ')}.  The job stops before the tests: this runner would run them "
               "hours slower than any other, and the result would say nothing about the code.")
    print(f"::error title=Degraded runner::{message}")
    _append("GITHUB_STEP_SUMMARY", f"- {label}: **degraded runner**, stopped before the tests. {message}\n")
    _append("GITHUB_OUTPUT", "degraded=true\n")
    marker_path().write_text(json.dumps({
        "degraded": True, "label": label, "reasons": reasons, "exit_code": DEGRADED_EXIT,
        "cpu_model": state.get("cpu_model"), "vcpus": state.get("vcpus"),
        **state["canary"]}, indent=1) + "\n", encoding="utf-8")
    return DEGRADED_EXIT


def _snapshot() -> dict:
    mem = meminfo(_proc("meminfo"))
    return {"time": time.time(), "available_kib": mem["available_kib"], "total_kib": mem["total_kib"],
            "ticks": cpu_ticks(_proc("stat"))}


def cmd_before_tests(_args) -> int:
    state = load_state()
    state["before_tests"] = _snapshot()
    save_state(state)
    snap = state["before_tests"]
    print(f"runner: before the tests, memory {_gib(snap['available_kib'])} available of {_gib(snap['total_kib'])}")
    return 0


def cmd_after_tests(args) -> int:
    state = load_state()
    after = _snapshot()
    before = state.get("before_tests")
    out = {"schema": 1, "label": args.label, "cpu_model": state.get("cpu_model", "unknown"),
           "vcpus": state.get("vcpus"), "mem_total_kib": after["total_kib"],
           "canary": state.get("canary"), "degraded": False}
    line = f"runner: after the tests, memory {_gib(after['available_kib'])} available of {_gib(after['total_kib'])}"
    if before:
        steal = steal_between(before["ticks"], after["ticks"])
        wall = after["time"] - before["time"]
        out["tests"] = {"wall_s": round(wall, 1), "available_kib_start": before["available_kib"],
                        "available_kib_end": after["available_kib"],
                        "steal_s": round(steal["steal_s"], 2), "cpu_s": round(steal["cpu_s"], 1),
                        "steal_pct": round(steal["steal_pct"], 3)}
        line += (f"; cpu steal {steal['steal_s']:.1f} s of {steal['cpu_s']:.0f} cpu-s over the "
                 f"{wall / 60:.1f} min test step ({steal['steal_pct']:.2f}%)")
    else:
        line += "; no before-tests snapshot, so no steal figure"
    print(line)
    _append("GITHUB_STEP_SUMMARY", f"- {args.label or 'job'}: {line.removeprefix('runner: ')}\n")
    if args.sidecar:
        Path(args.sidecar).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"runner: wrote {args.sidecar}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("machine", help="log the machine and start the install clock").set_defaults(fn=cmd_machine)
    p = sub.add_parser("canary", help="time the benchmark, read the install time, fail with 75 when degraded")
    p.add_argument("--label", default="", help="the job, for the step summary")
    p.set_defaults(fn=cmd_canary)
    sub.add_parser("before-tests", help="snapshot memory and CPU steal").set_defaults(fn=cmd_before_tests)
    p = sub.add_parser("after-tests", help="log memory and the steal since before-tests; write the sidecar")
    p.add_argument("--label", default="", help="the job, for the step summary and the sidecar")
    p.add_argument("--sidecar", default="", help="write the JSON sidecar here")
    p.set_defaults(fn=cmd_after_tests)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
