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


def _the_pencil_mesh_cases_left_out(rp, d):
    """A stencil file as a runner that knew only the 1-D mesh would write
    it on four devices -- results and checks consistent, every one of
    them passing -- and a hybrid file relabelled onto the 1-D mesh.  The
    pencil cases are the ones a fault in the wrapper's second exchanged
    axis fails; a file without them must not decide item 1."""
    st = d["stencil"][0]
    st["results"] = [r for r in st["results"] if r["mesh"] == "1d"]
    st["checks"] = [c for c in st["checks"] if " 2d " not in c["name"]]
    hy = d["hybrid"][0]
    for r in hy["results"]:
        r["mesh"], r["mesh_shape"] = "1d", [4, 1]
    hy["checks"] = rp.hybrid_checks(hy["results"], 4)
    # (as the runner would flag it: a check not run, spatial axis 1 split
    # over three devices nowhere, is not a pass)
    hy["passed"] = bool(hy["checks"]) and all(c["passed"] for c in hy["checks"])
    return ["stencil", "hybrid"]


def _the_cases_on_spatial_axis_1_left_out(rp, d):
    """Files as the schema-5 runner wrote them on four devices: the stencil
    goal on the 1-D and the 2 x 2 mesh only, the graph goals on the pencil
    only -- results and checks consistent, every check passing.  Neither
    mesh splits spatial axis 1 over more than two devices, where a halo
    from the wrong neighbour cannot show; such files must not decide items
    1, 3, 4 or 6."""
    st = d["stencil"][0]
    st["results"] = [r for r in st["results"] if r["mesh"] in ("1d", "2d")]
    st["checks"] = rp.stencil_checks(st["results"], 4)
    for goal in ("hybrid", "coupled"):
        x = d[goal][0]
        x["results"] = [r for r in x["results"] if r["mesh"] == "2d"]
        x["checks"] = rp.GOAL_CHECKS[goal](x["results"], 4)
    for goal in ("stencil", "hybrid", "coupled"):
        x = d[goal][0]
        x["passed"] = bool(x["checks"]) and all(c["passed"] for c in x["checks"])
    return ["stencil", "hybrid", "coupled"]


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


def _is_cg_convergence_check(c: dict) -> bool:
    return (" true residual" in c["name"]
            or "stopped on its tolerance, not on the iteration cap" in c["name"])


def _as_the_runner_wrote_gradient_before_its_solves_had_to_converge(doc: dict) -> None:
    """``gradient.json`` in the layout of schema 7: no record of the system
    solved or of the solves, and the parity checks only."""
    for r in doc["results"]:
        for key in ("solve", "shift", "rtol", "max_iters"):
            del r["sharded_cg"][key]
    doc["checks"] = [c for c in doc["checks"] if not _is_cg_convergence_check(c)]
    doc["passed"] = bool(doc["checks"]) and all(c["passed"] for c in doc["checks"])


def _a_gradient_file_of_schema_7_relabelled(rp, d):
    """Until schema 7 the ``sharded_cg`` part compared two solves that had
    not converged; such a file, its schema number edited to the current
    one, must not close item 3 on its parity checks."""
    _as_the_runner_wrote_gradient_before_its_solves_had_to_converge(d["gradient"][0])
    return ["gradient"]


def _the_convergence_checks_deleted_from_gradient(rp, d):
    g = d["gradient"][0]
    g["checks"] = [c for c in g["checks"] if not _is_cg_convergence_check(c)]
    return ["gradient"]


#: (seed, a phrase the status of every affected item must contain; ``None``:
#: the item must read ``FAILED``).
_SEEDS = [
    (_r1_value_over_limit_flag_left_true, None),
    (_r2_limit_loosened_in_the_record,
     "limit(s) differ from LIMITS: 'field 1d 4x1 16x20 periodic forward f vs unsharded "
     "max_rel' records limit 1.0, the runner holds it to 1e-05"),
    (_r3_results_disagree_with_checks, "check value(s) disagree with its results"),
    (_r4_all_but_one_check_deleted,
     "lacks case(s) the runner runs for cells [256, 1024] on 4 devices: field 1d 4x1 "
     "16x20 edge, field 1d 4x1 16x20 dirichlet, lbm 1d 4x1 16x20 periodic (+13 more)"),
    (_r5_an_older_runner_and_commit, "schema_version 3, not 8"),
    (_r6_one_goal_from_another_commit_and_host, "its files come from 2 commits"),
    (_r7_more_devices_than_the_environment_saw,
     "n_devices 4, but its environment saw 1 device(s)"),
    (_r8_partitioned_checks_deleted, "check(s) the runner derives from its results"),
    (_r9_schema_from_the_future, "schema_version 99, not 8"),
    (_config_says_fewer_sizes_than_were_run,
     "holds case(s) the runner does not run for its config: field 2d 2x2 32x36 periodic"),
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
    (_the_pencil_mesh_cases_left_out, "case(s) the runner"),
    (_the_cases_on_spatial_axis_1_left_out,
     "lacks case(s) the runner runs for cells [256, 1024] on 4 devices"),
    (_a_gradient_file_of_schema_7_relabelled,
     "gradient.json cannot decide it (its results cannot be read (KeyError: 'solve'))"),
    (_the_convergence_checks_deleted_from_gradient,
     "lacks 16 check(s) the runner derives from its results: '256 dof sharded_cg sharded "
     "solve stopped on its tolerance, not on the iteration cap'"),
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


def test_summarise_keeps_an_item_of_mixed_commits_open_and_exits_4(rp, recorded, tmp_path,
                                                                    capsys):
    docs = _as_real_gpu_run(recorded)
    _r6_one_goal_from_another_commit_and_host(rp, docs)
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4          # every file is valid on its own
    out = capsys.readouterr().out
    assert "[item 1] its files come from 2 commits" in out
    assert "1  " in out and "CLOSED" not in next(
        ln for ln in out.splitlines() if ln.startswith("1  "))
    assert "WARNING: MIXED COMMITS" in out


def _items_on_two_commits(recorded) -> tuple:
    """The control run, relabelled as a real 4-GPU run, with the files of
    items 2 and 4 (``halo``, ``hybrid``) from one commit and every other
    file from another: each item's own files agree, so each item closes --
    the directory a session leaves when it re-runs some goals on a fix
    commit and keeps the earlier passing files."""
    docs = _as_real_gpu_run(recorded)
    old, new = "a" * 40, "b" * 40
    for goal, goal_docs in docs.items():
        for doc in goal_docs:
            doc["environment"]["git_commit"] = old if goal in ("halo", "hybrid") else new
    return docs, old, new


def test_summarise_names_the_commit_each_item_closed_on(rp, recorded, tmp_path, capsys):
    """Until 0.4.0 the summary read all six items CLOSED from files of two
    commits without naming either: the rule that an item's files share one
    commit is per item, so items decided by different goals could close on
    different commits, and nothing said so."""
    docs, old, new = _items_on_two_commits(recorded)
    status = {i: s for i, (s, _) in rp.checklist_status(docs).items()}
    assert status == {i: "CLOSED" for i in _ITEMS}          # each item is consistent
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    out = capsys.readouterr().out
    lines = {int(ln[0]): ln for ln in out.splitlines() if ln[:1].isdigit() and "[" in ln}
    assert set(lines) == _ITEMS
    for item, line in lines.items():
        want = old if item in (2, 4) else new
        assert "CLOSED" in line and line.endswith(f"commit {want[:12]}"), line
    warning = out[out.index("WARNING: MIXED COMMITS"):]
    assert "come from 2 commits" in warning
    assert f"commit {old[:12]}: items 2, 4; halo.json, hybrid.json" in warning
    assert f"commit {new[:12]}: items 1, 3, 5, 6; " in warning
    assert out.rstrip().endswith("WARNING: MIXED COMMITS -- see above; exit 4")
    # the run table names each file's commit, beside its verdict
    halo = next(ln for ln in out.splitlines() if ln.startswith("halo "))
    assert halo.endswith(f"{old[:12]}  PASS"), halo


def test_an_item_of_mixed_commits_names_each_commit_and_its_files(rp, recorded, tmp_path,
                                                                  capsys):
    docs = _as_real_gpu_run(recorded)
    _a_second_file_from_another_commit(rp, docs)
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("1  "))
    commit = recorded["stencil"][0]["environment"]["git_commit"][:12]
    named = line[line.rindex("  commits ") + len("  commits "):]
    assert sorted(named.split("; ")) == sorted([f"{commit} (stencil.json, forward.json)",
                                                f"{'a' * 12} (stencil-rerun.json)"]), line


def test_one_commit_and_a_failure_keep_their_exit_codes(rp, recorded, tmp_path, capsys):
    """Exit 4 is for a directory that mixes commits and nothing else: one
    commit throughout is 0, and a failed check is 3 whatever the commits."""
    one = tmp_path / "one"
    one.mkdir()
    _write(one, _as_real_gpu_run(recorded))
    assert rp.summarise(one) == 0
    assert "MIXED COMMITS" not in capsys.readouterr().out
    both = tmp_path / "both"
    both.mkdir()
    docs, _old, _new = _items_on_two_commits(recorded)
    _r1_value_over_limit_flag_left_true(rp, docs)
    _write(both, docs)
    assert rp.summarise(both) == 3
    assert "WARNING: MIXED COMMITS" in capsys.readouterr().out


# --- the checks no other test pins: each must fail on a degenerate result ----


def _check_named(checks: list, fragment: str) -> dict:
    (c,) = [c for c in checks if fragment in c["name"]]
    return c


def _rederived(rp, doc: dict, mutate) -> tuple:
    """``(check status before, after)``: ``doc``'s checks derived from its
    results, then from a copy of them that ``mutate(results)`` degraded."""
    results = copy.deepcopy(doc["results"])
    before = rp.GOAL_CHECKS[doc["goal"]](results, doc["n_devices"])
    mutate(results)
    after = rp.GOAL_CHECKS[doc["goal"]](results, doc["n_devices"])
    return before, after


def test_the_stencil_goal_fails_a_parameter_gradient_that_is_zero(rp, recorded):
    """Two zeros agree with each other to any limit: a node whose loss does
    not depend on the parameter, or an adjoint that drops it on both sides,
    passes every parity check, and only this one catches it."""
    (doc,) = recorded["stencil"]
    name = "field 1d 4x1 16x20 periodic d loss / d diffusivity is not zero"

    def zero(results):
        r = results[0]
        for side in ("sharded", "unsharded"):
            r["gradient"][side]["grad_parameter"] = 0.0
        r["gradient"]["parity_grad_parameter"] = 0.0

    before, after = _rederived(rp, doc, zero)
    assert rp.check_status(_check_named(before, name)) == "passed"
    assert rp.check_status(_check_named(after, name)) == "failed"
    assert [c["name"] for c in after if rp.check_status(c) == "failed"] == [name]


def test_the_hybrid_goal_fails_a_correction_too_small_to_matter(rp, recorded):
    """A correction below the forward limit agrees with any wrapper: the
    goal would compare the inner node with itself."""
    (doc,) = recorded["hybrid"]
    name = "1d 4x1 16x20: the correction is not negligible"
    for negligible in (0.0, 100 * rp.LIMITS["forward"]):
        before, after = _rederived(rp, doc,
                                   lambda rs, v=negligible: rs[0].__setitem__("correction_rel", v))
        assert rp.check_status(_check_named(before, name)) == "passed"
        assert rp.check_status(_check_named(after, name)) == "failed", negligible


@pytest.mark.parametrize("which", ["solve", "adjoint", "tangent"])
@pytest.mark.parametrize("side", ["unsharded", "sharded"])
@pytest.mark.parametrize("dof", [256, 1024])
def test_the_gradient_goal_fails_a_cg_solve_that_did_not_converge(rp, recorded, dof, side,
                                                                  which):
    """Two solves that failed alike agree with each other: until schema 7
    the ``sharded_cg`` part passed on parity with a true residual of 6e2 to
    2e4 on both sides.  Each of the three solves behind the two derivatives
    is held to the residual limit on each side, at every size, and the
    residual of the unshifted operator at the dry run's cap (0.26, measured)
    fails the one check that names it and no other."""
    (doc,) = recorded["gradient"]
    name = f"{dof} dof sharded_cg {side} {which} true residual"
    index = [r["sharded_cg"]["dof"] for r in doc["results"]].index(dof)

    for residual in (0.26, 1.001 * rp.LIMITS["krylov_residual"], float("nan"), float("inf")):
        def degrade(results, residual=residual):
            results[index]["sharded_cg"]["solve"][side]["true_residual"][which] = residual

        before, after = _rederived(rp, doc, degrade)
        assert rp.check_status(_check_named(before, name)) == "passed"
        assert [c["name"] for c in after if rp.check_status(c) == "failed"] == [name], residual
    # at the limit it passes: the limit is the largest residual accepted
    _, at_limit = _rederived(rp, doc, lambda rs: rs[index]["sharded_cg"]["solve"][side][
        "true_residual"].__setitem__(which, rp.LIMITS["krylov_residual"]))
    assert rp.check_status(_check_named(at_limit, name)) == "passed"


@pytest.mark.parametrize("flag", [False, None, 0, 1, "True"],
                         ids=["false", "none", "0", "1", "text"])
@pytest.mark.parametrize("side", ["unsharded", "sharded"])
def test_the_gradient_goal_fails_a_cg_solve_that_stopped_on_its_iteration_cap(rp, recorded,
                                                                             side, flag):
    """The loop's own flag: a solve the cap stopped is not a solve, whatever
    it is compared with.  Only ``true`` passes."""
    (doc,) = recorded["gradient"]
    name = f"256 dof sharded_cg {side} solve stopped on its tolerance, not on the iteration cap"

    def degrade(results):
        results[0]["sharded_cg"]["solve"][side]["converged"] = flag

    before, after = _rederived(rp, doc, degrade)
    assert rp.check_status(_check_named(before, name)) == "passed"
    assert [c["name"] for c in after if rp.check_status(c) == "failed"] == [name]


def test_a_gradient_file_written_before_schema_8_is_refused_twice_over(rp, recorded):
    """A ``gradient.json`` of schema 7 checked another system (the unshifted
    operator, on which neither side converged).  It is refused for its
    schema number, as every older file is -- and, were that number edited,
    because its results do not record the solves this runner derives its
    checks from.  Neither refusal depends on the other."""
    old = copy.deepcopy(recorded["gradient"][0])
    _as_the_runner_wrote_gradient_before_its_solves_had_to_converge(old)
    assert all(c["passed"] for c in old["checks"]) and old["passed"] is True
    assert rp.record_problems(old) == ["its results cannot be read (KeyError: 'solve')"]
    assert rp.goal_verdict([old]) == "INVALID"
    old["schema_version"] = 7
    assert rp.record_problems(old) == [
        "schema_version 7, not 8: written by another version of this runner",
        "its results cannot be read (KeyError: 'solve')"]
    # the current record with only its schema number taken back is refused for that alone
    relabelled = copy.deepcopy(recorded["gradient"][0])
    relabelled["schema_version"] = 7
    assert rp.record_problems(relabelled) == [
        "schema_version 7, not 8: written by another version of this runner"]
    assert rp.goal_verdict([relabelled]) == "INVALID"


@pytest.mark.parametrize("iterations", [0, 1, 40, 41, None, True, 2.5],
                         ids=["0", "1", "max", "over_max", "none", "bool", "float"])
@pytest.mark.parametrize("side", ["unsharded", "sharded"])
def test_the_coupled_goal_fails_a_last_step_that_did_not_iterate(rp, recorded, iterations,
                                                                 side):
    """One pass is no fixed point and the cap is no convergence: a group
    that stopped on either compares two unconverged iterates."""
    (doc,) = recorded["coupled"]
    name = "1d 4x1 16x20 ift: the last step iterated"

    def degrade(results):
        results[0]["solvers"]["ift"][side]["last_step_iterations"] = iterations

    before, after = _rederived(rp, doc, degrade)
    assert rp.check_status(_check_named(before, name)) == "passed"
    assert rp.check_status(_check_named(after, name)) == "failed"


@pytest.mark.parametrize("key, fragments", [
    ("stencil", ("stencil 17x20 on 4 devices: ValueError",
                 "stencil refusal names the cell count, the device count")),
    ("stencil_axis1", ("along spatial axis 1 on 4 devices: ValueError",
                       "axis-1 stencil refusal names")),
    ("pointwise", ("pointwise 17x20 on 4 devices: ValueError", "pointwise refusal names")),
    ("pencil", ("on a 2x2 mesh: ValueError", "pencil refusal names")),
])
def test_the_indivisible_goal_fails_when_a_shape_it_must_refuse_is_accepted(rp, recorded, key,
                                                                            fragments):
    """A grid the mesh cannot split, accepted: the wrapper would run it on
    blocks of different sizes.  Each refusal's check -- and the one that it
    names the numbers -- must fail on a result that recorded no refusal."""
    (doc,) = recorded["indivisible"]

    def accept(results):
        results[0][key]["raised"] = None
        results[0][key]["message"] = ""

    before, after = _rederived(rp, doc, accept)
    for fragment in fragments:
        assert rp.check_status(_check_named(before, fragment)) == "passed", fragment
        assert rp.check_status(_check_named(after, fragment)) == "failed", fragment
    assert sum(rp.check_status(c) == "failed" for c in after) == len(fragments)


@pytest.mark.parametrize("key", ["stencil", "stencil_axis1"])
def test_the_indivisible_goal_fails_when_the_divisible_shape_is_refused(rp, recorded, key):
    """A wrapper that refused every grid would pass each refusal check; the
    divisible shape it must accept is what tells the two apart."""
    (doc,) = recorded["indivisible"]
    fragment = {"stencil": "16x20 (divisible) is accepted",
                "stencil_axis1": "16x20 sharded along spatial axis 1 (divisible) is accepted"}[key]

    def refuse(results):
        results[0][key]["divisible_raised"] = "ValueError"
        results[0][key]["divisible_message"] = "refused"

    before, after = _rederived(rp, doc, refuse)
    assert rp.check_status(_check_named(before, fragment)) == "passed"
    assert rp.check_status(_check_named(after, fragment)) == "failed"


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


# --- a goal that raised under --keep-going, and a directory with no commit ----


def _raised(rp, recorded, goal="halo") -> dict:
    doc = copy.deepcopy(recorded[goal][0])
    return rp.record_goal_raised(doc, ValueError("seeded: shard_map refused the step"))


def test_the_record_of_a_goal_that_raised_is_valid_and_fails(rp, recorded):
    doc = _raised(rp, recorded)
    assert rp.record_problems(doc) == []
    assert rp.goal_verdict([doc]) == "FAIL"
    status = {i: s for i, (s, _) in rp.checklist_status(
        {**_as_real_gpu_run(recorded), "halo": [doc]}).items()}
    assert status[2] == "FAILED"


@pytest.mark.parametrize("tamper, problem", [
    (lambda d: d["checks"][0].update(value=True, passed=True), "disagree"),
    (lambda d: d["checks"].append(dict(d["checks"][0], name="extra")), "does not emit"),
    (lambda d: d["checks"].clear(), "lacks"),
    (lambda d: d["results"].append({"cells": 1}), "records results"),
    (lambda d: d["raised"].pop("type"), "type and message"),
    (lambda d: d.update(raised="ValueError"), "type and message"),
])
def test_a_tampered_record_of_a_goal_that_raised_cannot_decide(rp, recorded, tamper, problem):
    doc = _raised(rp, recorded)
    tamper(doc)
    problems = rp.record_problems(doc)
    assert problems and any(problem in p for p in problems), problems


def test_summarise_lists_a_goal_that_raised_and_exits_3(rp, recorded, tmp_path, capsys):
    docs = _as_real_gpu_run(recorded)
    docs["halo"] = [_raised(rp, docs)]
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert "goal raised" in out and "seeded: shard_map refused the step" in out


def test_summarise_exits_4_when_no_file_records_a_commit(rp, recorded, tmp_path, capsys):
    """A tree synced without ``.git`` records no commit in any file: every
    item stays open, and the summary used to exit 0 as if one session at
    one commit had written them.  It exits 4 and says why."""
    docs = _as_real_gpu_run(recorded)
    for goal_docs in docs.values():
        for doc in goal_docs:
            doc["environment"]["git_commit"] = None
    status = {i: s for i, (s, _) in rp.checklist_status(docs).items()}
    assert all(s.startswith("open: no git commit recorded") for s in status.values()), status
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    out = capsys.readouterr().out
    assert "no file in this directory records a git commit" in out
    assert out.rstrip().endswith("WARNING: MIXED COMMITS -- see above; exit 4")


def test_summarise_exits_4_when_one_file_records_no_commit(rp, recorded, tmp_path, capsys):
    docs = _as_real_gpu_run(recorded)
    docs["halo"][0]["environment"]["git_commit"] = None
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    assert "WARNING: MIXED COMMITS" in capsys.readouterr().out


# --- a commit the runner can trust, and a file an older runner wrote -----------


def _runner_in(directory: Path):
    import shutil

    shutil.copy(_RUNNER, directory / "run_pod.py")
    spec = importlib.util.spec_from_file_location(f"run_pod_in_{directory.name}",
                                                  directory / "run_pod.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_repository_with_no_commit_records_no_commit(tmp_path):
    """``git rev-parse HEAD`` in a repository with no commit yet prints
    ``HEAD`` and exits 128; the runner recorded ``"HEAD"`` as the commit,
    and the commit gate took it as one session's."""
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    rp_here = _runner_in(tmp_path)
    assert rp_here._git_commit() is None
    subprocess.run(["git", "-C", str(tmp_path), "add", "run_pod.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "c"], check=True)
    sha = rp_here._git_commit()
    assert isinstance(sha, str) and len(sha) == 40 and int(sha, 16) >= 0


def test_a_recorded_commit_that_is_not_a_sha_is_no_commit(rp, recorded, tmp_path, capsys):
    docs = _as_real_gpu_run(recorded)
    for goal_docs in docs.values():
        for doc in goal_docs:
            doc["environment"]["git_commit"] = "HEAD"
    assert rp._commit_of(docs["halo"][0]) is None
    status = {i: s for i, (s, _) in rp.checklist_status(docs).items()}
    assert all(s.startswith("open: no git commit recorded") for s in status.values()), status
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 4
    assert "no file in this directory records a git commit" in capsys.readouterr().out


def test_summarise_reads_an_older_runners_file_invalid_instead_of_stopping(rp, recorded,
                                                                          tmp_path, capsys):
    """The README's own scenario -- the runner changed between the session
    and the summary: a schema-5 file without a key schema 6 added.  Its
    tables stopped the summary with a KeyError traceback (exit 1, which
    means "no goal JSON"); it reads INVALID and exits 3."""
    docs = copy.deepcopy(recorded)
    old = docs["indivisible"][0]
    old["schema_version"] = 5
    for r in old["results"]:
        r.pop("stencil_axis1", None)
    _write(tmp_path, docs)
    assert rp.summarise(tmp_path) == 3
    out = capsys.readouterr().out
    assert next(ln for ln in out.splitlines() if ln.startswith("indivisible ")).endswith("INVALID")
    assert "Tables leave out 1 file(s)" in out and "indivisible.json: KeyError" in out
    # The other files' tables are still printed.
    assert "Halo exchange vs NumPy" in out
