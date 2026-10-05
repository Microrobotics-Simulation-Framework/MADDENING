"""``run_pod.py --summarise`` reads a goal file of any shape, and counts every file.

``--summarise`` ended on a traceback -- exit 1, the status the runbook gives
for "no goal JSON" -- for a goal file that was a JSON object with a field
of another type: ``environment: null``, a check that is a string,
``n_devices: [4]``, a value past the float range (an audit's fuzz found 45
kinds).  ``_load_results`` now checks each file's shape and reads it, and
routes any file the summary cannot read to the ``INVALID`` stand-in (exit
3).  And a file with no ``checks`` key (what a schema-2 runner wrote) read
"no checks" and was never counted, so the summary exited 0: every file is
asked ``record_problems`` now.

The fuzz test changes one field of one committed record to a JSON value
of any shape, or deletes it, and requires a verdict: an exit status the
runbook gives a summary (0, 3 or 4), never an exception.  Everything runs
in-process, reads JSON only, and imports no JAX.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import maddening

_REPO = Path(maddening.__file__).resolve().parents[2]
_RUNNER = _REPO / "benchmarks" / "multigpu" / "run_pod.py"
_RECORD = Path(__file__).resolve().parent / "run_pod_record"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load(_RUNNER, "run_pod_summary_reads_any_goal_file")
RECORDS = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(_RECORD.glob("*.json"))}


def _paths(tree, prefix=(), depth=0):
    """Every key path into *tree*, two entries of each list deep."""
    yield prefix
    if depth >= 8:
        return
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _paths(value, prefix + (key,), depth + 1)
    elif isinstance(tree, list):
        for i, value in enumerate(tree[:2]):
            yield from _paths(value, prefix + (i,), depth + 1)


PATHS = {goal: [p for p in _paths(doc) if p] for goal, doc in RECORDS.items()}
#: The fields the summary's own tables and verdicts read (the record's top
#: level, its environment and config, and the first check and result),
#: drawn half the time: the deep paths into ``results`` are many more.
SHALLOW = {goal: [p for p in paths if len(p) <= 2 or p[:2] in (("checks", 0), ("results", 0))]
           for goal, paths in PATHS.items()}

_LEAF = (st.none() | st.booleans() | st.integers() | st.just(10 ** 400) | st.just(-10 ** 400)
         | st.floats() | st.text(max_size=4))
_JSON = st.recursive(_LEAF, lambda children: st.lists(children, max_size=3)
                     | st.dictionaries(st.text(max_size=4), children, max_size=3),
                     max_leaves=6)
_DELETE = object()


def _summarise(docs: dict) -> int:
    with tempfile.TemporaryDirectory(prefix="summarise_fuzz_") as tmp:
        for goal, doc in docs.items():
            (Path(tmp) / f"{goal}.json").write_text(json.dumps(doc), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            return rp.summarise(Path(tmp))


def _changed(doc, path, value):
    doc = copy.deepcopy(doc)
    holder = doc
    for key in path[:-1]:
        holder = holder[key]
    if value is _DELETE:
        del holder[path[-1]]
    else:
        holder[path[-1]] = value
    return doc


@given(data=st.data())
def test_a_goal_file_of_any_shape_gets_a_verdict_not_a_traceback(data):
    goal = data.draw(st.sampled_from(sorted(RECORDS)), label="goal")
    path = data.draw(st.sampled_from(SHALLOW[goal]) | st.sampled_from(PATHS[goal]),
                     label="path")
    value = data.draw(st.one_of(_JSON, st.just(_DELETE)), label="value")
    doc = _changed(RECORDS[goal], path, value)
    assert _summarise({goal: doc}) in (0, 3, 4), (goal, path, value)


@pytest.mark.parametrize("path, value", [
    (("environment",), None), (("environment",), "dry-run-host"),
    (("environment", "jax"), None), (("environment", "device_kinds"), None),
    (("environment", "platform"), None), (("checks",), None), (("checks", 0), "halo ok"),
    (("results", 0), None), (("n_devices",), [4]), (("checks", 0, "value"), 10 ** 400),
    (("config",), [1]), (("config", "n_devices"), "4"),
])
def test_each_mistyped_field_the_audit_named_reads_invalid(path, value, capsys):
    doc = _changed(RECORDS["halo"], path, value)
    assert _summarise({"halo": doc}) == 3
    docs = rp._load_results(_write_one("halo", doc), "halo")
    assert "_unreadable" in docs[0] or rp.record_problems(docs[0]), (path, value)


def test_a_table_value_past_the_float_range_leaves_the_file_out_of_the_tables(capsys):
    """A number the tables format (``{:9.3f}``) that no float holds raised
    ``OverflowError``, which the tables' guard did not catch.  The field
    is a timing no check reads, so the file's checks still decide (exit 0)
    and the tables say which file they leave out."""
    path = ("results", 0, "methods", "all_to_all", "wrapper_step", "ms_per_step")
    doc = _changed(RECORDS["forward"], path, 10 ** 400)
    with tempfile.TemporaryDirectory(prefix="summarise_table_") as tmp:
        (Path(tmp) / "forward.json").write_text(json.dumps(doc), encoding="utf-8")
        assert rp.summarise(Path(tmp)) == 0
    out = capsys.readouterr().out
    assert "Tables leave out 1 file(s)" in out and "OverflowError" in out


def _write_one(goal: str, doc: dict) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="summarise_one_"))
    (tmp / f"{goal}.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp


@pytest.mark.parametrize("extra", [False, True])
def test_a_goal_file_with_no_checks_reads_invalid_and_the_summary_exits_3(extra, capsys):
    """A schema-2 file records no ``checks``: it is evidence of nothing this
    runner would have written, alone or beside a valid file of its goal."""
    old = {k: v for k, v in RECORDS["halo"].items() if k != "checks"}
    with tempfile.TemporaryDirectory(prefix="summarise_nochecks_") as tmp:
        directory = Path(tmp)
        for goal, doc in RECORDS.items():
            if goal != "halo" or extra:
                (directory / f"{goal}.json").write_text(json.dumps(doc), encoding="utf-8")
        (directory / ("halo_old.json" if extra else "halo.json")).write_text(
            json.dumps(old), encoding="utf-8")
        assert rp.summarise(directory) == 3
    out = capsys.readouterr().out
    assert "no results or no checks recorded" in out
    assert "INVALID" in out


def test_the_committed_record_set_still_reads_pass(capsys):
    assert _summarise(RECORDS) == 0
