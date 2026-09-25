"""``run_pod.py --summarise`` closes an item only on evidence the runner would write.

The summary decides what the multi-GPU session proved, from JSON files
that travel from a pod to a laptop by hand.  It used to take four things
on each file's word: the limit each check was held to, the set of checks,
the schema, and where and on how many devices the file was measured.  A
file with a check's limit loosened to 1.0, results that disagreed with its
checks, all but one check deleted, the stencil cases of an older runner,
a goal from another commit, ``n_devices`` 4 on an environment that saw one
device, or ``schema_version`` 99 all read ``CLOSED``.

``run_pod_record/`` is a real ``--goal all --dry-run --cells 256 1024``
output of this runner (``environment.hostname`` and ``config.out``
replaced).  Relabelled as a real 4-GPU run it must close all six items --
the control that shows the rules are not simply refusing everything.  Each
seed corrupts it one way and must move every item the seeded files decide
off ``CLOSED``, with a status that names the file and the reason, and must
leave the other items closed.

**If you change what a goal records or how its checks are built**, the
first test fails: the recorded files are no longer what the runner would
write, which is exactly what ``--summarise`` now refuses.  Regenerate them::

    JAX_PLATFORMS=cpu python benchmarks/multigpu/run_pod.py --goal all --dry-run \\
        --cells 256 1024 --out tests/cloud/multigpu/run_pod_record

and replace ``environment.hostname`` and ``config.out`` in each file.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

import maddening

_RUNNER = Path(maddening.__file__).resolve().parents[2] / "benchmarks" / "multigpu" / "run_pod.py"
_RECORD = Path(__file__).resolve().parent / "run_pod_record"
_ITEMS = {1, 2, 3, 4, 5, 6}


def _runner_module():
    spec = importlib.util.spec_from_file_location("run_pod_verdict_under_test", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rp():
    return _runner_module()


@pytest.fixture(scope="module")
def recorded(rp):
    docs = {goal: rp._load_results(_RECORD, goal) for goal in rp.ALL_GOALS}
    assert all(len(docs[g]) == 1 for g in rp.ALL_GOALS), {g: len(v) for g, v in docs.items()}
    return docs


def _as_real_gpu_run(docs: dict) -> dict:
    """The dry run relabelled as what a real 4-GPU run writes."""
    docs = copy.deepcopy(docs)
    for goal_docs in docs.values():
        for doc in goal_docs:
            doc["dry_run"] = False
            doc["environment"]["platform"] = "gpu"
            doc["environment"]["device_kinds"] = ["NVIDIA A100-SXM4-80GB"]
    return docs


def _items_decided_by(rp, goals) -> set:
    return {i for i, (_claim, gs) in rp.CHECKLIST.items() if set(gs) & set(goals)}


def _first_check(doc: dict, pred) -> dict:
    return next(c for c in doc["checks"] if pred(c))


def _is_forward_parity(c: dict) -> bool:
    return c.get("sense") == "<=" and "forward f" in c["name"]


# --- the seeds ----------------------------------------------------------------
# Each takes (rp, docs), corrupts docs in place and returns the goals whose
# files it touched.


def _r1_value_over_limit_flag_left_true(rp, d):
    _first_check(d["stencil"][0], _is_forward_parity)["value"] = 10 * rp.LIMITS["forward"]
    return ["stencil"]


def _r2_limit_loosened_in_the_record(rp, d):
    c = _first_check(d["stencil"][0], _is_forward_parity)
    c["value"], c["limit"], c["passed"] = 100 * rp.LIMITS["forward"], 1.0, True
    return ["stencil"]


def _r3_results_disagree_with_checks(rp, d):
    d["stencil"][0]["results"][0]["forward"]["parity"]["f"]["max_rel"] = 0.5
    d["halo"][0]["results"][0]["stencil_cases"][0]["forward_max_abs"] = 7.0
    return ["stencil", "halo"]


def _r4_all_but_one_check_deleted(rp, d):
    st = d["stencil"][0]
    st["checks"] = [c for c in st["checks"] if c.get("sense") == "=="][:1]
    st["results"] = st["results"][:1]
    st["passed"] = True
    return ["stencil"]


def _r5_an_older_runner_and_commit(rp, d):
    for goal in ("stencil", "hybrid"):
        for x in d[goal]:
            x["schema_version"] = 3
            x["environment"]["git_commit"] = "0" * 40
            x["environment"]["jax"] = "0.4.30"
            for c in x["checks"]:
                c.pop("sense", None)                  # how schema 3 wrote them
            x["results"] = [r for r in x["results"]
                            if r.get("boundary", "periodic") == "periodic"]
            x["checks"] = [c for c in x["checks"]
                           if " edge" not in c["name"] and "dirichlet" not in c["name"]]
    return ["stencil", "hybrid"]


def _r6_one_goal_from_another_commit_and_host(rp, d):
    d["forward"][0]["environment"]["git_commit"] = "f" * 40
    d["forward"][0]["environment"]["hostname"] = "some-other-pod"
    return ["forward"]


def _r7_more_devices_than_the_environment_saw(rp, d):
    for goal in rp.ALL_GOALS:
        for x in d[goal]:
            x["environment"]["n_devices_visible"] = 1
            x["environment"]["devices"] = x["environment"]["devices"][:1]
    return list(rp.ALL_GOALS)


def _r8_partitioned_checks_deleted(rp, d):
    for goal in ("stencil", "hybrid", "coupled"):
        for x in d[goal]:
            x["checks"] = [c for c in x["checks"] if "partitioned" not in c["name"]]
    return ["stencil", "hybrid", "coupled"]


def _r9_schema_from_the_future(rp, d):
    for goal in rp.ALL_GOALS:
        for x in d[goal]:
            x["schema_version"] = 99
    return list(rp.ALL_GOALS)


def _config_says_fewer_sizes_than_were_run(rp, d):
    d["stencil"][0]["config"]["cells"] = [256]
    return ["stencil"]


def _a_check_the_runner_never_emits(rp, d):
    d["hybrid"][0]["checks"].append(rp.check_that("an extra reassurance", True))
    return ["hybrid"]


def _a_limit_tightened_in_the_record(rp, d):
    c = _first_check(d["coupled"][0], lambda c: c.get("sense") == "<=")
    c["limit"] = c["limit"] / 10 if c["limit"] else 1e-9
    c["passed"] = rp.expected_pass(c)
    d["coupled"][0]["passed"] = all(x["passed"] for x in d["coupled"][0]["checks"])
    return ["coupled"]


def _no_commit_recorded(rp, d):
    d["halo"][0]["environment"]["git_commit"] = None
    return ["halo"]


def _a_second_file_from_another_commit(rp, d):
    rerun = copy.deepcopy(d["stencil"][0])
    rerun["_file"] = "stencil-rerun.json"
    rerun["environment"]["git_commit"] = "a" * 40
    d["stencil"].append(rerun)
    return ["stencil"]


def _a_result_measured_on_other_devices(rp, d):
    d["coupled"][0]["results"][-1]["n_devices"] = 2
    return ["coupled"]


def _config_disagrees_on_the_device_count(rp, d):
    d["gradient"][0]["config"]["n_devices"] = 8
    return ["gradient"]


def _device_list_and_count_disagree(rp, d):
    d["indivisible"][0]["environment"]["n_devices_visible"] = 8
    return ["indivisible"]


def _a_case_the_runner_does_not_run(rp, d):
    extra = copy.deepcopy(d["hybrid"][0]["results"][0])
    extra["shape"] = [8, 8]
    d["hybrid"][0]["results"].append(extra)
    return ["hybrid"]


def _a_stencil_refusal_that_recommends_the_unstructured_wrapper(rp, d):
    """The stencil refusal as it read before 0.4.0, recommending the wrapper
    that now refuses a stencil node, with its check still recorded as passed.
    The check used to ask only for the wrapper's name, which both messages
    carry."""
    st = d["indivisible"][0]["results"][0]["stencil"]
    head = st["message"].split("  ShardedUnstructuredNode is not a way out")[0]
    assert head != st["message"], st["message"]
    st["message"] = (head.rstrip(".").replace(", or run on", ", run on")
                     + ", or use ShardedUnstructuredNode, which carries an explicit "
                     "padded layout and accepts any (device, cell) pair.")
    return ["indivisible"]


#: (seed, a phrase the status of every affected item must contain; ``None``:
#: the item must read ``FAILED``).
_SEEDS = [
    (_r1_value_over_limit_flag_left_true, None),
    (_r2_limit_loosened_in_the_record,
     "limit(s) differ from LIMITS: 'field 1d 4x1 16x16 periodic forward f vs unsharded "
     "max_rel' records limit 1.0, the runner holds it to 1e-05"),
    (_r3_results_disagree_with_checks, "check value(s) disagree with its results"),
    (_r4_all_but_one_check_deleted,
     "lacks case(s) the runner runs for cells [256, 1024] on 4 devices: field 1d 4x1 "
     "16x16 edge, field 1d 4x1 16x16 dirichlet, lbm 1d 4x1 16x16 periodic (+5 more)"),
    (_r5_an_older_runner_and_commit, "schema_version 3, not 5"),
    (_r6_one_goal_from_another_commit_and_host, "its files come from 2 commits"),
    (_r7_more_devices_than_the_environment_saw,
     "n_devices 4, but its environment saw 1 device(s)"),
    (_r8_partitioned_checks_deleted, "check(s) the runner derives from its results"),
    (_r9_schema_from_the_future, "schema_version 99, not 5"),
    (_config_says_fewer_sizes_than_were_run,
     "holds case(s) the runner does not run for its config: field 2d 2x2 32x32 periodic"),
    (_a_check_the_runner_never_emits,
     "check(s) the runner does not emit for its results: 'an extra reassurance'"),
    (_a_limit_tightened_in_the_record, "limit(s) differ from LIMITS"),
    (_no_commit_recorded, "no git commit recorded in halo.json"),
    (_a_second_file_from_another_commit, "its files come from 2 commits"),
    (_a_result_measured_on_other_devices,
     "n_devices 4, but its results were measured on 2"),
    (_config_disagrees_on_the_device_count, "n_devices 4, but its config says 8"),
    (_device_list_and_count_disagree,
     "its environment lists 4 device(s) but records n_devices_visible 8"),
    (_a_case_the_runner_does_not_run, "holds case(s) the runner does not run"),
    (_a_stencil_refusal_that_recommends_the_unstructured_wrapper,
     "check value(s) disagree with its results: 'stencil refusal names the cell count, "
     "the device count and that the unstructured wrapper is not a way out for a stencil "
     "node' records True, its results give False"),
]


def test_the_recorded_dry_run_is_evidence_this_runner_would_write(rp, recorded):
    """If this fails after a change to a runner, regenerate ``run_pod_record``
    (see the module docstring): the change made old records stale."""
    for goal, docs in recorded.items():
        assert rp.record_problems(docs[0]) == [], (goal, rp.record_problems(docs[0]))
        assert rp.goal_verdict(docs) == "PASS", goal
    assert all(s == "open: passed on CPU / dry run only"
               for s, _ in rp.checklist_status(recorded).values())


def test_the_dry_run_relabelled_as_a_four_gpu_run_closes_every_item(rp, recorded):
    status = rp.checklist_status(_as_real_gpu_run(recorded))
    assert {i: s for i, (s, _) in status.items()} == {i: "CLOSED" for i in _ITEMS}


@pytest.mark.parametrize("seed, reason", _SEEDS, ids=[s.__name__.lstrip("_") for s, _ in _SEEDS])
def test_a_corrupted_record_reopens_the_items_it_decides_and_says_why(rp, recorded, seed,
                                                                     reason):
    docs = _as_real_gpu_run(recorded)
    touched = seed(rp, docs)
    status = {i: s for i, (s, _) in rp.checklist_status(docs).items()}
    affected = _items_decided_by(rp, touched)
    assert affected, "the seed must touch a goal that decides an item"
    for item in affected:
        if reason is None:
            assert status[item] == "FAILED", (item, status[item])
        else:
            assert status[item].startswith("open: ") and reason in status[item], (
                item, status[item])
    assert {i for i, s in status.items() if s == "CLOSED"} == _ITEMS - affected, status


def test_the_order_of_a_files_checks_is_not_evidence(rp, recorded):
    """The rules compare what the checks claim, not how they are listed."""
    docs = _as_real_gpu_run(recorded)
    for goal_docs in docs.values():
        goal_docs[0]["checks"].reverse()
    assert all(s == "CLOSED" for s, _ in rp.checklist_status(docs).values())


@pytest.mark.parametrize("control, goals, want", [
    (lambda d: d["halo"][0].__setitem__("n_devices", 2), ["halo"], "open: "),
    (lambda d: d["coupled"][0].__setitem__("dry_run", True), ["coupled"],
     "open: passed on CPU / dry run only"),
    (lambda d: d["indivisible"][0]["checks"][0].__setitem__("passed", False),
     ["indivisible"], "FAILED"),
], ids=["halo_on_two_devices", "coupled_is_a_dry_run", "a_flag_flipped"])
def test_the_rules_that_already_held_still_hold(rp, recorded, control, goals, want):
    docs = _as_real_gpu_run(recorded)
    control(docs)
    status = {i: s for i, (s, _) in rp.checklist_status(docs).items()}
    affected = _items_decided_by(rp, goals)
    assert all(status[i].startswith(want) for i in affected), status
    assert {i for i, s in status.items() if s == "CLOSED"} == _ITEMS - affected


def _write(directory: Path, docs: dict) -> None:
    for goal_docs in docs.values():
        for doc in goal_docs:
            name = doc.get("_file") or f"{doc['goal']}.json"
            (directory / name).write_text(
                json.dumps({k: v for k, v in doc.items() if k != "_file"}), encoding="utf-8")


def test_summarise_lists_a_record_that_cannot_decide_and_exits_3(rp, recorded, tmp_path,
                                                                  capsys):
    docs = _as_real_gpu_run(recorded)
    _r2_limit_loosened_in_the_record(rp, docs)
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert "Records that cannot decide" in out
    assert "[stencil] stencil.json: 1 limit(s) differ from LIMITS" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("1  "))
    assert "open: stencil.json cannot decide it" in line and "[stencil INVALID" in line
    assert next(ln for ln in out.splitlines() if ln.startswith("stencil ")).endswith("INVALID")


def test_summarise_keeps_an_item_of_mixed_commits_open_without_failing(rp, recorded,
                                                                       tmp_path, capsys):
    docs = _as_real_gpu_run(recorded)
    _r6_one_goal_from_another_commit_and_host(rp, docs)
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 0          # every file is valid on its own
    out = capsys.readouterr().out
    assert "[item 1] its files come from 2 commits" in out
    assert "1  " in out and "CLOSED" not in next(
        ln for ln in out.splitlines() if ln.startswith("1  "))


def test_the_transport_ranking_uses_only_a_file_that_passes(rp, recorded):
    """A row decides the default transport only from a file that reads
    ``PASS``: not from one whose checks failed (the transports disagreed),
    nor from one that is not evidence this runner would write."""
    docs = _as_real_gpu_run(recorded)["exchange"]
    assert rp.recommend(docs, min_cells=0)["deciding_rows"] == 2
    stale = copy.deepcopy(docs)
    stale[0]["schema_version"] = 3
    rec = rp.recommend(stale, min_cells=0)
    assert rec["decision"] == "undecided" and "its file reads INVALID" in rec["reason"]
    split = copy.deepcopy(docs)
    split[0]["results"][0]["bit_identical"] = False
    split[0]["checks"][0].update(value=False, passed=False)
    split[0]["passed"] = False
    rec = rp.recommend(split, min_cells=0)
    assert rec["decision"] == "undecided" and "its file reads FAIL" in rec["reason"]
