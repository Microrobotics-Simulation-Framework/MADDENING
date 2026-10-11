"""The multi-GPU runbook's table of wrapper paths says "run" only where a seed proves it.

``benchmarks/multigpu/README.md`` has one row for every documented option
of ``ShardedStencilNode`` and ``ShardedUnstructuredNode`` (and what the
goals run of ``ShardedPointwiseNode`` and ``HybridNode``), with a status:
``run`` (a seeded fault on that path fails the goals named), ``unproved``
(on a goal's path, or perhaps; no seed) or ``not run`` (CPU tests only).
A hardware session is read against that table, so it must not claim more
than ``tests/cloud/multigpu/test_run_pod_seeded_faults.py`` holds: every
``run`` row names seeds that exist and that fail the goals it names, every
seed of the wrappers is named in a ``run`` row, and every other row names
a test file that exists (RPD-067 in
``docs/validation/rest_runpod_claims.yaml``).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.cloud.multigpu import test_run_pod_seeded_faults as seeded

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "benchmarks" / "multigpu" / "README.md"
HEADING = "### Which wrapper paths the goals run"
WRAPPERS = ("ShardedStencilNode", "ShardedUnstructuredNode", "ShardedPointwiseNode",
            "HybridNode")
STATUSES = ("run", "unproved", "not run")
COLUMNS = ["path", "status", "goals", "seeds, or the CPU test"]


def seeds_and_the_goals_each_fails() -> dict:
    """``{seed: goals it must fail}`` for every seed of a wrapper, of the
    halo exchange or of the goals' own node (the ``sharded_cg`` seeds are
    of the runner's own operator, which is no wrapper's path)."""
    out = {name: tuple(seed.fails) for name, seed in seeded._SEEDS.items()}
    out.update({name: tuple(seed.fails) for name, seed in seeded._PATH_SEEDS.items()})
    out.update({name: ("forward",) for name in seeded._COUNT_SEEDS})
    assert len(out) == (len(seeded._SEEDS) + len(seeded._PATH_SEEDS)
                        + len(seeded._COUNT_SEEDS)), "two seed tables share a name"
    return out


def coverage_rows(text: str) -> list:
    """``[(wrapper, path, status, goals, seeds, test files)]`` of the section."""
    assert text.count(HEADING) == 1, "the runbook's coverage section is gone, or doubled"
    section = text.split(HEADING)[1].split("\n### ")[0]
    rows, wrapper = [], None
    for line in section.splitlines():
        named = re.fullmatch(r"`(\w+)`:", line.strip())
        if named:
            wrapper = named.group(1)
            continue
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == len(COLUMNS), line
        if cells == COLUMNS:
            continue
        assert wrapper is not None, f"a row before any wrapper's name: {line}"
        path, status, goals, proof = cells
        rows.append((wrapper, path, status, tuple(re.findall(r"`([a-z]+)`", goals)),
                     tuple(n for n in re.findall(r"`([a-z_0-9]+)`", proof)),
                     tuple(re.findall(r"`(tests/[\w/]+\.py)`", proof))))
    return rows


def problems(text: str, seeds: dict, goals: tuple) -> list:
    """Everything the table claims that the seeded test does not hold."""
    rows, out = coverage_rows(text), []
    for wrapper in WRAPPERS:
        if not [r for r in rows if r[0] == wrapper]:
            out.append(f"no row for {wrapper}")
    paths = [(r[0], r[1]) for r in rows]
    out += [f"row twice: {p}" for p in sorted(set(paths)) if paths.count(p) > 1]
    named = set()
    for wrapper, path, status, row_goals, row_seeds, files in rows:
        where = f"{wrapper}: {path[:60]}"
        if wrapper not in WRAPPERS:
            out.append(f"{where}: not one of {WRAPPERS}")
        if status not in STATUSES:
            out.append(f"{where}: status {status!r} is not one of {STATUSES}")
        elif status == "run":
            unknown = [s for s in row_seeds if s not in seeds]
            if not row_seeds or unknown:
                out.append(f"{where}: 'run' needs seeds that exist; unknown {unknown}")
            if not row_goals or set(row_goals) - set(goals):
                out.append(f"{where}: 'run' needs goals of the runner, got {row_goals}")
            known = [s for s in row_seeds if s in seeds]
            failed = {g for s in known for g in seeds[s]}
            if set(row_goals) - failed:
                out.append(f"{where}: no seed of the row fails {sorted(set(row_goals) - failed)}")
            out += [f"{where}: seed {s} fails none of the row's goals" for s in known
                    if not set(seeds[s]) & set(row_goals)]
            if files:
                out.append(f"{where}: a 'run' row names its seeds, not a test file")
            named |= set(known)
        else:
            if row_goals or row_seeds:
                out.append(f"{where}: only a 'run' row names goals or seeds")
            missing = [f for f in files if not (REPO / f).is_file()]
            if not files or missing:
                out.append(f"{where}: needs a CPU test file that exists; missing {missing}")
    out += [f"seed {s} is in no 'run' row" for s in sorted(set(seeds) - named)]
    return out


def test_every_run_row_is_proved_by_a_seed_and_every_seed_is_in_a_run_row():
    text = RUNBOOK.read_text(encoding="utf-8")
    seeds = seeds_and_the_goals_each_fails()
    goals = seeded._runner().ALL_GOALS
    assert problems(text, seeds, goals) == []
    statuses = {status for _w, _p, status, *_ in coverage_rows(text)}
    assert "run" in statuses and "not run" in statuses


def test_the_check_of_the_table_fails_on_a_claim_no_seed_holds():
    """The guard can fail: a row promoted to ``run`` without a seed, a seed
    that does not exist, a goal no seed of the row fails, a seed left out, a
    test file that is not there, a wrapper without a row."""
    text = RUNBOOK.read_text(encoding="utf-8")
    seeds = seeds_and_the_goals_each_fails()
    goals = seeded._runner().ALL_GOALS
    rows = coverage_rows(text)
    line_of = {(r[0], r[1]): next(line for line in text.splitlines() if f"| {r[1]} |" in line)
               for r in rows}
    other = next(r for r in rows if r[2] != "run")
    run = next(r for r in rows if r[2] == "run" and r[0] == "ShardedStencilNode")
    a_seed = run[4][0]

    def edited(row, old, new):
        line = line_of[(row[0], row[1])]
        assert line.count(old) == 1, (line, old)
        return text.replace(line, line.replace(old, new))

    def said(changed):
        found = problems(changed, seeds, goals)
        assert found, "the edit was not noticed"
        return " / ".join(found)

    assert "'run' needs seeds" in said(edited(other, f"| {other[2]} |", "| run |"))
    assert "unknown ['no_such_seed']" in said(edited(run, f"`{a_seed}`", "`no_such_seed`"))
    assert "fails ['exchange']" in said(edited(run, f"| run | `{run[3][0]}`",
                                               f"| run | `exchange`, `{run[3][0]}`"))
    assert "is not one of" in said(edited(run, "| run |", "| ran |"))
    assert "needs a CPU test file" in said(edited(other, other[5][0], "tests/no_such_test.py"))
    assert "no row for HybridNode" in said(text.replace("`HybridNode`:", "`Hybrid`:"))
    left_out = next(s for s in seeds if sum(s in r[4] for r in rows) == 1)
    row = next(r for r in rows if left_out in r[4])
    gone = edited(row, f"`{left_out}`", "`static_halos_are_nan`")
    assert f"seed {left_out} is in no 'run' row" in said(gone)
