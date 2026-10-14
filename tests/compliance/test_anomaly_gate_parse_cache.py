"""The anomaly gate's parse cache changes its speed and nothing else.

``maddening.compliance._validate.parse_cache`` holds one parsed tree per
test file for one validation run, and ``scripts/check_anomalies.py`` keeps
what it derives from each tree for the run (``_ModuleMemo``).  A cache that
went stale would hide an edit from the gate, so these tests hold three
things:

* a cached run and an uncached run report exactly the same findings, on
  seeded faults here and on the shipped registry (the slow lane, because
  the uncached run is the 12 s the cache exists to remove);
* a file edited between two validations in one run is seen, including an
  edit that keeps the file's size and modification time;
* the cache lives for one run: nothing is held between runs or outside one.
"""

import ast
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.compliance import _validate
from maddening.compliance._validate import (
    ParseCache,
    parse_cache,
    resolve_test_reference,
    validate_anomaly_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "_gate_check_anomalies_parse_cache",
        REPO_ROOT / "scripts" / "check_anomalies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uncached(monkeypatch, gate):
    """Turn every cache off: each lookup parses and derives afresh."""
    monkeypatch.setattr(ParseCache, "parse",
                        lambda self, path: _validate._parse_uncached(path))
    monkeypatch.setattr(gate._ModuleMemo, "_get",
                        lambda self, kind, inputs, compute: compute())


# A test module with one of each thing the gate's reference rules judge.
_PROBE = '''\
import sys

import pytest

_skip = pytest.mark.skip(reason="seeded: an aliased unconditional skip")


def test_runs():
    assert True


@pytest.mark.skip(reason="seeded")
def test_never_runs():
    assert True


@_skip
def test_skipped_through_an_alias():
    assert True


@pytest.mark.skipif(sys.platform == "no-such-platform", reason="seeded")
def test_sometimes_runs():
    assert True


@pytest.mark.xfail(strict=True, reason="seeded")
def test_expected_to_fail():
    assert False


@pytest.fixture
def needs_a_device():
    pytest.skip("seeded: a fixture every run skips in")


def test_skipped_by_its_fixture(needs_a_device):
    assert True


def _helper_not_a_test():
    pass


class TestWithInit:
    def __init__(self):
        pass

    def test_never_collected(self):
        pass


class TestFine:
    def test_method(self):
        pass
'''

# A second and a third module, so that one file's derived values reaching
# another file's citations would change the findings: a module-level skip,
# and a fixture with the probe's fixture name that does not skip.
_SKIPPED_MODULE = '''\
import pytest

pytestmark = pytest.mark.skip(reason="seeded: a module that never runs")


def test_in_a_skipped_module():
    assert True
'''

_CLEAN_MODULE = '''\
import pytest


@pytest.fixture
def needs_a_device():
    return "a device"


def test_with_a_fixture_that_runs(needs_a_device):
    assert needs_a_device
'''

_ENTRY = '''\
  - anomaly_id: "MADD-ANO-{n:03d}"
    title: "Seeded"
    description: "Seeded"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Seeded"
    resolution_status: "{status}"
{closed}    affected_versions: "{versions}"
    affected_components:
      - "maddening.nodes.heat.HeatNode"
    verification:
{refs}'''


def _registry(root, entries):
    """A registry at ``root`` whose entries cite the seeded test modules.

    ``entries`` is a list of ``(status, [test names])``; a name is cited as
    ``tests/test_probe.py::<name>`` (``""`` cites the file itself), unless
    it names its file itself (``tests/<file>.py::<name>``).
    """
    body = []
    for n, (status, names) in enumerate(entries, start=1):
        closed = status == "resolved"
        refs = "".join(
            f'      - "{name}"\n' if name.startswith("tests/") else
            f'      - "tests/test_probe.py{"::" + name if name else ""}"\n'
            for name in names)
        body.append(_ENTRY.format(
            n=n, status=status, refs=refs,
            closed='    resolution_version: "0.4.0"\n' if closed else "",
            versions=">=0.1.0, <0.4.0" if closed else ">=0.1.0"))
    path = root / "known_anomalies.yaml"
    path.write_text('schema_version: "1.0"\nmaddening_version: "0.4.0.dev0"\n'
                    'generated_date: "2026-03-12"\nanomalies:\n'
                    + "".join(body))
    return path


@pytest.fixture
def seeded(tmp_path):
    """A probe file cited many times over, by entries carrying every fault
    the reference rules know, beside a clean citation of each kind."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_probe.py").write_text(_PROBE)
    (tmp_path / "tests" / "test_skipped.py").write_text(_SKIPPED_MODULE)
    (tmp_path / "tests" / "test_clean.py").write_text(_CLEAN_MODULE)
    path = _registry(tmp_path, [
        ("resolved", ["test_runs", "TestFine::test_method", ""]),
        ("resolved", ["test_never_runs", "test_runs"]),
        ("resolved", ["test_skipped_through_an_alias"]),
        ("resolved", ["test_expected_to_fail", "TestFine"]),
        ("open", ["test_expected_to_fail", "test_sometimes_runs"]),
        ("resolved", ["test_skipped_by_its_fixture"]),
        ("resolved", ["_helper_not_a_test"]),
        ("resolved", ["TestWithInit::test_never_collected"]),
        ("resolved", ["test_no_such_test"]),
        ("open", ["TestFine::test_method", "test_runs", "test_runs"]),
        ("resolved", ["tests/test_clean.py::test_with_a_fixture_that_runs",
                      "test_runs"]),
        ("resolved", ["tests/test_skipped.py::test_in_a_skipped_module"]),
        ("resolved", ["tests/test_clean.py::test_with_a_fixture_that_runs",
                      "tests/test_probe.py"]),
    ])
    return tmp_path, path


def _gate_run(gate, capsys, *args):
    status = gate.main([*map(str, args)])
    out = capsys.readouterr()
    return status, out.out, out.err


class TestCachedAndUncachedRunsAgree:
    def test_on_seeded_faults(self, gate, seeded, capsys, monkeypatch):
        root, path = seeded
        cached = _gate_run(gate, capsys, path, "--repo-root", root)
        with monkeypatch.context() as m:
            _uncached(m, gate)
            uncached = _gate_run(gate, capsys, path, "--repo-root", root)
        assert cached == uncached
        # Not vacuous: every seeded fault is in the findings both runs agree on.
        status, _, err = cached
        assert status == 1
        for needle in (
            "MADD-ANO-002: verification entry 'tests/test_probe.py::"
            "test_never_runs': is skipped unconditionally",
            "MADD-ANO-003: verification entry 'tests/test_probe.py::"
            "test_skipped_through_an_alias': is skipped unconditionally",
            "MADD-ANO-004: verification entry 'tests/test_probe.py::"
            "test_expected_to_fail': is marked xfail",
            "MADD-ANO-006: verification entry 'tests/test_probe.py::"
            "test_skipped_by_its_fixture': is skipped unconditionally",
            "MADD-ANO-007: verification entry 'tests/test_probe.py::"
            "_helper_not_a_test': '_helper_not_a_test' is not a test",
            "class 'TestWithInit' defines __init__",
            "MADD-ANO-009: verification entry 'tests/test_probe.py::"
            "test_no_such_test': 'tests/test_probe.py' defines no "
            "'test_no_such_test' at module level",
            "MADD-ANO-010: verification lists 'tests/test_probe.py::test_runs' "
            "more than once",
            "MADD-ANO-012: verification entry 'tests/test_skipped.py::"
            "test_in_a_skipped_module': is skipped unconditionally",
        ):
            assert needle in err, needle
        for clean in ("MADD-ANO-001", "MADD-ANO-011", "MADD-ANO-013"):
            assert clean not in err, clean
        assert "MADD-ANO-005" not in err      # a strict xfail pins an open entry
        assert "test_sometimes_runs': is skipped conditionally" in cached[1]

    def test_on_the_validator_alone(self, seeded, monkeypatch):
        root, path = seeded
        cached = validate_anomaly_registry(str(path), repo_root=str(root))
        with monkeypatch.context() as m:
            m.setattr(ParseCache, "parse",
                      lambda self, p: _validate._parse_uncached(p))
            uncached = validate_anomaly_registry(str(path), repo_root=str(root))
        assert cached == uncached
        assert any("defines no 'test_no_such_test'" in e for e in cached)

    # Per push: tests/compliance/test_anomaly_gate_parse_cache.py::TestCachedAndUncachedRunsAgree::test_on_seeded_faults
    @pytest.mark.slow
    def test_on_the_shipped_registry(self, gate, capsys, monkeypatch):
        """The whole gate as CI runs it, both ways (about 2 s and 12 s)."""
        cached = _gate_run(gate, capsys, "--prefix", "MADD-ANO-")
        with monkeypatch.context() as m:
            _uncached(m, gate)
            uncached = _gate_run(gate, capsys, "--prefix", "MADD-ANO-")
        assert cached == uncached
        assert cached[0] == 0, cached[2]

    def test_a_cited_file_is_parsed_once_per_run(self, gate, seeded, capsys,
                                                 monkeypatch):
        """The speed-up is real: 18 citations of three files, three parses."""
        root, path = seeded
        parsed = []
        real = _validate._parse_source

        def counting(data, py_path):
            parsed.append(py_path)
            return real(data, py_path)

        monkeypatch.setattr(_validate, "_parse_source", counting)
        _gate_run(gate, capsys, path, "--repo-root", root)
        assert sorted(parsed) == [
            os.path.join(str(root), "tests", name)
            for name in ("test_clean.py", "test_probe.py", "test_skipped.py")]


def _rename(path, old, new):
    text = path.read_text()
    assert old in text
    stat = path.stat()
    path.write_text(text.replace(old, new))
    # Same size, same modification time: only the content says it changed.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size


class TestAnEditBetweenTwoValidationsIsSeen:
    def test_by_the_validator(self, seeded):
        root, path = seeded
        probe = root / "tests" / "test_probe.py"
        ref = "tests/test_probe.py::test_runs"
        with parse_cache():
            assert resolve_test_reference(ref, str(root)) is None
            first = validate_anomaly_registry(str(path), repo_root=str(root))
            _rename(probe, "def test_runs(", "def test_rung(")
            assert "defines no 'test_runs'" in resolve_test_reference(
                ref, str(root))
            second = validate_anomaly_registry(str(path), repo_root=str(root))
            _rename(probe, "def test_rung(", "def test_runs(")
            third = validate_anomaly_registry(str(path), repo_root=str(root))
        assert not any("'test_runs'" in e for e in first)
        assert any("defines no 'test_runs'" in e for e in second)
        assert third == first

    def test_by_the_gates_reference_rules(self, gate, seeded):
        """The memo of derived values is keyed on the tree, so a re-parsed
        file is judged afresh: a skip added between two lookups is seen."""
        root, _ = seeded
        probe = root / "tests" / "test_probe.py"
        memo = gate._ModuleMemo()
        ref = "tests/test_probe.py::test_runs"
        with parse_cache():
            before, _ = gate._reference_findings(
                "MADD-ANO-001", "resolved", ref, str(root), memo)
            text = probe.read_text()
            probe.write_text(text.replace(
                "def test_runs():", "@_skip\ndef test_runs():"))
            after, _ = gate._reference_findings(
                "MADD-ANO-001", "resolved", ref, str(root), memo)
        assert before == []
        assert len(after) == 1 and "is skipped unconditionally" in after[0]

    def test_by_two_gate_runs(self, gate, seeded, capsys):
        root, path = seeded
        probe = root / "tests" / "test_probe.py"
        first = _gate_run(gate, capsys, path, "--repo-root", root)
        _rename(probe, "def test_runs(", "def test_rung(")
        second = _gate_run(gate, capsys, path, "--repo-root", root)
        assert "defines no 'test_runs'" not in first[2]
        assert "defines no 'test_runs'" in second[2]


class TestTheCacheLivesForOneRun:
    def test_nothing_is_cached_outside_a_run(self, seeded):
        root, _ = seeded
        probe = str(root / "tests" / "test_probe.py")
        assert _validate._ACTIVE_PARSE_CACHE.get() is None
        assert _validate._parse(probe) is not _validate._parse(probe)

    def test_a_run_holds_one_tree_per_content(self, seeded):
        root, _ = seeded
        probe = str(root / "tests" / "test_probe.py")
        with parse_cache() as cache:
            tree = _validate._parse(probe)
            assert _validate._parse(probe) is tree
            assert isinstance(tree, ast.Module)
            with parse_cache() as inner:
                assert inner is cache          # a nested run shares the outer
            with parse_cache(ParseCache()) as other:
                assert other is not cache
                assert _validate._parse(probe) is not tree
            assert _validate._ACTIVE_PARSE_CACHE.get() is cache
        assert _validate._ACTIVE_PARSE_CACHE.get() is None

    def test_no_cache_survives_a_validation_or_a_gate_run(self, gate, seeded,
                                                          capsys):
        root, path = seeded
        validate_anomaly_registry(str(path), repo_root=str(root))
        assert _validate._ACTIVE_PARSE_CACHE.get() is None
        _gate_run(gate, capsys, path, "--repo-root", root)
        assert _validate._ACTIVE_PARSE_CACHE.get() is None

    def test_the_gate_restores_the_collector(self, gate, seeded, capsys):
        import gc

        root, path = seeded
        assert gc.isenabled()
        _gate_run(gate, capsys, path, "--repo-root", root)
        assert gc.isenabled()
        gc.disable()
        try:
            _gate_run(gate, capsys, path, "--repo-root", root)
            assert not gc.isenabled()
        finally:
            gc.enable()

    def test_an_unreadable_or_unparseable_file_is_none_both_ways(self,
                                                                 tmp_path):
        broken = tmp_path / "test_broken.py"
        broken.write_text("def test_(:\n")
        missing = str(tmp_path / "test_missing.py")
        for path in (str(broken), missing):
            assert _validate._parse_uncached(path) is None
            with parse_cache():
                assert _validate._parse(path) is None
                assert _validate._parse(path) is None


# ---------------------------------------------------------------------------
# The registry is loaded as it always was
# ---------------------------------------------------------------------------

_ONE_ENTRY = '''\
schema_version: "1.0"
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "One entry"
    description: >
      Seeded.
    severity: "minor"
    safety_relevance: "not_safety_relevant"
    safety_relevance_rationale: "Seeded."
    affected_components:
      - "maddening.nodes.heat.HeatNode"
    affected_versions: ">=0.1.0"
    resolution_status: "open"
    verification:
      - "tests/test_probe.py::test_runs"
'''

# Documents on which libyaml's loader and PyYAML's pure-Python loader
# disagree (the pre-merge audit's four, and one more a fuzz found): the
# first three only the C loader reads, the last two only the Python one.
_LOADERS_DISAGREE = {
    "a tab after a colon": ('severity: "minor"', 'severity:\t"minor"'),
    "a trailing tab": ('title: "One entry"', 'title: "One entry"\t'),
    "a block scalar header with a comment and no space": (
        "description: >\n", "description: >#seeded\n"),
    "a byte-order mark inside the document": (
        "generated_date:", "\ufeffgenerated_date:"),
    "a lone surrogate escape": ('title: "One entry"', 'title: "One \\ud800"'),
}


def _safe_load_refuses(text):
    import yaml

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return True
    return False


class TestTheRegistryIsLoadedAsItAlwaysWas:
    """``validate_anomaly_registry`` is public: downstreams call it on their
    own registries, and its verdict must not depend on whether PyYAML was
    built with libyaml.  It loads with ``yaml.safe_load``, as it always did,
    and so does the gate, once, for every rule: a registry or fragment the
    gate passes then loads wherever else it is read (the SOUP generator,
    the assembler), and one ``yaml.safe_load`` refuses the gate refuses."""

    @pytest.mark.parametrize("case", sorted(_LOADERS_DISAGREE))
    def test_the_public_validator_reads_what_safe_load_reads(self, tmp_path, case):
        import yaml

        old, new = _LOADERS_DISAGREE[case]
        text = _ONE_ENTRY.replace(old, new, 1)
        assert text != _ONE_ENTRY
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(text, encoding="utf-8")
        if _safe_load_refuses(text):
            with pytest.raises(yaml.YAMLError):
                validate_anomaly_registry(str(path), resolve_references=False)
        else:
            assert isinstance(
                validate_anomaly_registry(str(path), resolve_references=False), list)

    @pytest.mark.parametrize("case", sorted(_LOADERS_DISAGREE))
    def test_the_gate_refuses_what_safe_load_refuses(self, gate, tmp_path, capsys,
                                                     case):
        old, new = _LOADERS_DISAGREE[case]
        text = _ONE_ENTRY.replace(old, new, 1)
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(text, encoding="utf-8")
        status, _, err = _gate_run(gate, capsys, path, "--no-resolve")
        refused = _safe_load_refuses(text)
        assert ("does not parse as YAML" in err) is refused, err
        if refused:
            assert status == 1

    def test_the_cases_are_ones_the_loaders_disagree_on(self):
        """Else the two tests above prove nothing on this PyYAML build."""
        import yaml

        if not getattr(yaml, "__with_libyaml__", False):
            pytest.skip("PyYAML without libyaml: there is no second loader")

        def verdict(text, loader):
            try:
                return repr(yaml.load(text, Loader=loader))
            except yaml.YAMLError:
                return "refused"

        for case, (old, new) in _LOADERS_DISAGREE.items():
            text = _ONE_ENTRY.replace(old, new, 1)
            assert verdict(text, yaml.SafeLoader) != verdict(text, yaml.CSafeLoader), case

    def test_one_load_serves_the_validator_and_the_gates_rules(self, gate, seeded,
                                                               capsys, monkeypatch):
        root, path = seeded
        loads = []
        real = gate._load_registry
        monkeypatch.setattr(gate, "_load_registry",
                            lambda p: loads.append(p) or real(p))
        _gate_run(gate, capsys, path, "--repo-root", root)
        assert loads == [str(path)]
