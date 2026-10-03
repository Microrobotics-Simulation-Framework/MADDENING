"""``run_pod.py --summarise`` claims, in the record domains their own tests leave out.

``docs/validation/rest_runpod_claims.yaml`` gives each RPD row three
record domains (``testing_standards.md``, "The domain matrix"):

* ``dry_run_cpu``: the records a ``--dry-run`` on CPU virtual devices
  writes, as they are (``run_pod_record/`` is one);
* ``relabelled_records``: those records relabelled as a real 4-GPU run
  (``_as_real_gpu_run``), the evidence a pod session would bring back;
* ``mixed_commits``: a directory whose files record more than one commit
  (``_items_on_two_commits``: a session that re-ran some goals on a fix
  commit and kept the earlier passing files).

Most cells cite ``test_run_pod_verdict_integrity.py`` and
``test_run_pod_dry_run.py``; these tests fill the rest: each verdict rule
asked again where two of the domains meet, or on the dry-run records
themselves.  Nothing here runs a goal (only the summary, which imports no
JAX), so nothing can reach a device or a provider.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys

import pytest

from tests.cloud.multigpu.test_run_pod_verdict_integrity import (
    _ITEMS,
    _RECORD,
    _RUNNER,
    _a_second_file_from_another_commit,
    _as_real_gpu_run,
    _first_check,
    _is_forward_parity,
    _items_on_two_commits,
    _r2_limit_loosened_in_the_record,
    _raised,
    _runner_module,
    _write,
)


@pytest.fixture(scope="module")
def rp():
    return _runner_module()


@pytest.fixture(scope="module")
def recorded(rp):
    docs = {goal: rp._load_results(_RECORD, goal) for goal in rp.ALL_GOALS}
    assert all(len(docs[g]) == 1 for g in rp.ALL_GOALS)
    return docs


def _status(rp, docs) -> dict:
    return {i: s for i, (s, _) in rp.checklist_status(docs).items()}


def _line(out: str, item: int) -> str:
    return next(ln for ln in out.splitlines() if ln.startswith(f"{item}  "))


def _older_runner(doc: dict) -> None:
    """The README's scenario: a schema-5 file without a key schema 6 added."""
    doc["schema_version"] = 5
    for r in doc["results"]:
        r.pop("stencil_axis1", None)


# ---------------------------------------------------------------------------
# relabelled records
# ---------------------------------------------------------------------------

def test_an_older_runners_relabelled_file_reads_invalid_and_the_rest_still_decide(
        rp, recorded, tmp_path, capsys):
    """RPD-010 on relabelled records: the session's file from a runner that
    has since changed reads INVALID and is left out of the tables (exit 3),
    and the items no such file decides still close."""
    docs = _as_real_gpu_run(recorded)
    _older_runner(docs["indivisible"][0])
    status = _status(rp, docs)
    decided = {i for i, (_c, gs) in rp.CHECKLIST.items() if "indivisible" in gs}
    assert {i for i, s in status.items() if s == "CLOSED"} == _ITEMS - decided
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert next(ln for ln in out.splitlines() if ln.startswith("indivisible ")).endswith("INVALID")


# ---------------------------------------------------------------------------
# mixed commits
# ---------------------------------------------------------------------------

def test_a_goal_that_raised_among_files_of_two_commits_fails_its_item_and_exits_3(
        rp, recorded, tmp_path, capsys):
    """RPD-012 in a directory of two commits: the record of a goal that
    raised still reads FAIL and fails its item, and the exit is 3 -- a
    failure, not the 4 of mixed commits alone -- with the warning printed."""
    docs, _old, _new = _items_on_two_commits(recorded)
    docs["halo"] = [_raised(rp, docs)]
    assert rp.record_problems(docs["halo"][0]) == []
    assert _status(rp, docs)[2] == "FAILED"
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert "goal raised" in out and "seeded: shard_map refused the step" in out
    assert "WARNING: MIXED COMMITS" in out


def test_a_check_not_run_among_files_of_two_commits_keeps_its_items_open_and_exits_4(
        tmp_path, capsys):
    """RPD-013 in a directory of two commits: a check not run beside a check
    that ran and passed is still no failure -- the exit is mixed commits'
    4, not a failure's 3 -- and the items its goal decides stay open.  (A
    record the runner wrote holds a check not run only below four devices,
    RPD-027, so the rule is asked of the synthetic one-check documents
    ``test_run_pod_dry_run.py`` uses for the verdict rules.)"""
    from tests.cloud.multigpu.test_run_pod_dry_run import (
        _all_goal_docs, _rules_only_runner, _summarise_without_tables, _write_checklist_docs)

    rp = _rules_only_runner()
    docs = _all_goal_docs()
    docs["halo"][0]["environment"]["git_commit"] = "d" * 40
    docs["indivisible"][0]["checks"].append(rp.check_not_run("pencil refusal", "needs 4"))
    docs["indivisible"][0]["passed"] = False
    _write_checklist_docs(tmp_path, docs)
    assert _summarise_without_tables(rp, tmp_path) == 4
    out = capsys.readouterr().out
    assert "Checks not run" in out and "Failed checks" not in out
    assert "WARNING: MIXED COMMITS" in out
    line = _line(out, 5)
    assert "INCOMPLETE" in line and "CLOSED" not in line


@pytest.mark.parametrize("corrupt", ["limit-loosened", "older-runner"])
def test_a_file_that_cannot_decide_among_files_of_two_commits_reads_invalid_and_exits_3(
        rp, recorded, tmp_path, capsys, corrupt):
    """RPD-018 and RPD-010 in a directory of two commits: a record with a
    limit loosened, or one an older runner wrote, decides nothing (INVALID,
    listed under "Records that cannot decide" or left out of the tables),
    and the exit is 3, not mixed commits' 4."""
    docs, _old, _new = _items_on_two_commits(recorded)
    if corrupt == "limit-loosened":
        _r2_limit_loosened_in_the_record(rp, docs)
        goal = "stencil"
    else:
        _older_runner(docs["indivisible"][0])
        goal = "indivisible"
    decided = {i for i, (_c, gs) in rp.CHECKLIST.items() if goal in gs}
    status = _status(rp, docs)
    assert all(status[i].startswith("open: ") for i in decided), status
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert next(ln for ln in out.splitlines() if ln.startswith(f"{goal} ")).endswith("INVALID")
    assert "WARNING: MIXED COMMITS" in out


def test_the_transport_ranking_reads_its_rows_whatever_commit_the_exchange_file_records(
        rp, recorded):
    """RPD-022 beside mixed commits: the recommendation is decided by its
    rows' rules (a real accelerator run, enough devices and cells, a file
    that reads PASS, a finite speedup) -- the same decision and reason when
    the exchange file records another commit than the rest -- and the
    commit rule keeps the checklist items, not the ranking, open."""
    docs = _as_real_gpu_run(recorded)["exchange"]
    want = rp.recommend(docs, min_cells=0)
    other = copy.deepcopy(docs)
    other[0]["environment"]["git_commit"] = "c" * 40
    got = rp.recommend(other, min_cells=0)
    assert got["decision"] == want["decision"] and got["deciding_rows"] == want["deciding_rows"]
    assert got["deciding_rows"] == 2


# ---------------------------------------------------------------------------
# the dry-run records themselves
# ---------------------------------------------------------------------------

def test_dry_run_files_of_two_commits_print_the_mixed_commits_warning_and_exit_4(
        rp, recorded, tmp_path, capsys):
    """RPD-020 on the dry-run records as written: a goal from another commit
    is the MIXED COMMITS warning, naming each commit's files, and exit 4 --
    for a dry run too, whose items stay open whatever the commits."""
    docs = copy.deepcopy(recorded)
    docs["forward"][0]["environment"]["git_commit"] = "f" * 40
    assert all(s.startswith("open: ") for s in _status(rp, docs).values())
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    out = capsys.readouterr().out
    assert "WARNING: MIXED COMMITS" in out and "forward.json" in out[out.index("WARNING"):]
    assert out.rstrip().endswith("WARNING: MIXED COMMITS -- see above; exit 4")


def test_a_dry_run_record_that_disagrees_with_its_limit_fails_and_exits_3(
        rp, recorded, tmp_path, capsys):
    """RPD-017 on the dry-run records: the summary re-derives a check of a
    dry run as of any run -- a value past its limit recorded passed fails
    the goal and the summary exits 3, naming the check."""
    docs = copy.deepcopy(recorded)
    check = _first_check(docs["stencil"][0], _is_forward_parity)
    check["value"] = 10 * rp.LIMITS["forward"]
    assert check["passed"] is True
    assert rp.goal_verdict(docs["stencil"]) == "FAIL"
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    assert "recorded passed=True, but the value fails its limit" in capsys.readouterr().out


def test_a_dry_run_record_whose_commit_is_not_a_sha_counts_as_none(rp, recorded, tmp_path,
                                                                   capsys):
    """RPD-011 on the dry-run records: "HEAD" (what git prints with no
    commit yet) is no commit, and a directory recording none exits 4 saying
    so -- a dry run included."""
    docs = copy.deepcopy(recorded)
    for goal_docs in docs.values():
        for doc in goal_docs:
            doc["environment"]["git_commit"] = "HEAD"
    assert rp._commit_of(docs["halo"][0]) is None
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    assert "no file in this directory records a git commit" in capsys.readouterr().out


def test_summarising_the_dry_run_records_imports_no_jax(tmp_path):
    """RPD-021 on a directory of dry-run records, not only an empty one:
    the summary reads all eight goals (exit 0, nothing failed) in a fresh
    interpreter and never imports jax or jaxlib."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rp', {str(_RUNNER)!r})\n"
        "rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)\n"
        f"rc = rp.main(['--summarise', {str(_RECORD)!r}])\n"
        "loaded = sorted(m for m in sys.modules if m == 'jax' or m.startswith('jax.') "
        "or m == 'jaxlib' or m.startswith('jaxlib.'))\n"
        "print(rc, loaded)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_API_KEY",)}
    env["HOME"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "0 []", out.stdout


def test_a_second_relabelled_file_of_one_goal_from_another_commit_keeps_its_item_open(
        rp, recorded):
    """RPD-019 where the two domains meet: relabelled records of which one
    goal's two files disagree on the commit -- the items that goal decides
    are not CLOSED, the others are."""
    docs = _as_real_gpu_run(recorded)
    touched = _a_second_file_from_another_commit(rp, docs)
    decided = {i for i, (_c, gs) in rp.CHECKLIST.items() if set(gs) & set(touched)}
    status = _status(rp, docs)
    assert all(status[i] != "CLOSED" for i in decided), status
    assert {i for i, s in status.items() if s == "CLOSED"} == _ITEMS - decided


@pytest.mark.parametrize("a2a_ms, ppermute_ms, want", [
    (1.05, 1.0, "ppermute"), (1.0499, 1.0, "tie"), (1.0, 1.0, "tie"), (1.0, 1.0001, "all_to_all"),
], ids=["exactly-the-margin", "just-under-it", "even", "just-slower"])
def test_the_recommendation_thresholds_hold_on_a_relabelled_exchange_record(
        rp, recorded, a2a_ms, ppermute_ms, want):
    """RPD-023 on the evidence a pod session brings back: the exchange record
    relabelled as a real 4-GPU run, every row's medians set so ppermute is
    exactly the margin faster, just under it, level, or just slower -- the
    decision the thresholds give, from a record that is still valid."""
    docs = copy.deepcopy(_as_real_gpu_run(recorded)["exchange"])
    for r in docs[0]["results"]:
        r["methods"]["all_to_all"]["median_ms"] = a2a_ms
        r["methods"]["ppermute"]["median_ms"] = ppermute_ms
        r["ppermute_speedup_median"] = a2a_ms / ppermute_ms
    assert rp.record_problems(docs[0]) == []
    rec = rp.recommend(docs, min_cells=0)
    assert rec["deciding_rows"] == len(docs[0]["results"])
    assert rec["decision"] == want, rec
