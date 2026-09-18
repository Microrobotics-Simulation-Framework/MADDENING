"""Each compliance gate must fail on the defect it exists to catch.

The gates in ``scripts/check_*.py`` are MADDENING's IEC 62304 / MDCG
evidence that the documentation matches the code, and they are cited as
delivered coverage.  The tests that already existed check that they pass on
the repository as it stands, which a gate with an empty scope also does --
a mutation audit found 22 of 28 planted defects passing.  These tests pin
the other direction: a planted bad input has to produce a non-zero exit.

Each test names the mutation it replays.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    """Import one of the scripts/ gates as a module (they are not a package)."""
    spec = importlib.util.spec_from_file_location(
        f"_gate_{name}", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(name, *args):
    """Run a gate as CI runs it, and return the CompletedProcess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["JAX_PLATFORMS"] = "cpu"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py"), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


@pytest.fixture(scope="module")
def transforms_gate():
    return _load("check_transforms")


@pytest.fixture(scope="module")
def mapping_gate():
    return _load("check_impl_mapping")


@pytest.fixture(scope="module")
def citations_gate():
    return _load("check_citations")


# ---------------------------------------------------------------------------
# check_transforms.py
# ---------------------------------------------------------------------------

class TestTransformGate:
    def test_an_unregistered_string_transform_fails_the_gate(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "uses_a_ghost.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="no_such_transform")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_a_scope_containing_no_references_fails_the_gate(
        self, transforms_gate, tmp_path
    ):
        """The audit's headline defect: the gate verified zero references."""
        (tmp_path / "nothing_to_see.py").write_text("x = 1\n")
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_a_transform_the_same_file_registers_passes(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "registers_its_own.py").write_text(
            '@register_transform("locally_defined")\n'
            "def _t(x):\n"
            "    return x\n"
            '\n'
            'gm.add_edge("a", "b", "x", "y", transform="locally_defined")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    def test_a_transform_registered_in_another_file_does_not_count(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "registers.py").write_text(
            '@register_transform("elsewhere")\ndef _t(x):\n    return x\n'
        )
        (tmp_path / "uses.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="elsewhere")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_a_param_spec_transform_is_not_an_edge_transform(
        self, transforms_gate, tmp_path
    ):
        """``ParamSpec(transform="log")`` is a reparametrisation, not an edge."""
        (tmp_path / "param_only.py").write_text(
            'spec = ParamSpec(bounds=(0.0, None), transform="log")\n'
        )
        # Nothing in scope -> the empty-scope guard fires, not a false positive
        # about "log" being unregistered.
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_the_repository_transform_references_all_resolve(self):
        result = _run("check_transforms")
        assert result.returncode == 0, result.stdout + result.stderr
        # Regression guard on the audit finding: the gate reported
        # "OK: 0 ... verified" for the whole of v0.3 and v0.4-dev.
        assert " 0 string transform reference(s) verified" not in result.stdout


# ---------------------------------------------------------------------------
# check_impl_mapping.py
# ---------------------------------------------------------------------------

_TABLE_HEADER = (
    "## Implementation Mapping\n\n"
    "| Equation Term | Implementation | Notes |\n"
    "|---------------|---------------|-------|\n"
)


def _guide(tmp_path, *rows, name="guide.md"):
    path = tmp_path / name
    path.write_text(_TABLE_HEADER + "".join(rows) + "\n## Next Section\n")
    return path


class TestImplementationMappingGate:
    def test_an_attribute_only_the_base_class_defines_fails_the_gate(
        self, mapping_gate, tmp_path
    ):
        """Replays: renaming HeatNode.update to step still reported OK.

        ``HeatNode.to_dict`` is defined by ``SimulationNode`` alone, which is
        exactly the shape the renamed method leaves behind.
        """
        _guide(tmp_path, "| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | |\n")
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_a_row_that_declares_the_behaviour_inherited_is_allowed(
        self, mapping_gate, tmp_path
    ):
        _guide(
            tmp_path,
            "| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | "
            "inherited from SimulationNode |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 0

    def test_a_row_without_a_code_reference_fails_the_gate(
        self, mapping_gate, tmp_path
    ):
        """Replays: dropping the backticks turned 16 verified into 15, exit 0."""
        _guide(tmp_path, "| Diffusion | maddening.nodes.heat.HeatNode.update | |\n")
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_a_non_callable_target_fails_the_gate(self, mapping_gate, tmp_path):
        _guide(tmp_path, "| Docs | `maddening.nodes.heat.HeatNode.__doc__` | |\n")
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_every_symbol_in_a_row_is_checked_not_only_the_first(
        self, mapping_gate, tmp_path
    ):
        _guide(
            tmp_path,
            "| Two steps | `maddening.nodes.heat.HeatNode.update` then "
            "`maddening.nodes.heat.HeatNode.no_such_method` | |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_a_guide_that_loses_rows_fails_its_pinned_minimum(
        self, mapping_gate, tmp_path
    ):
        """Replays: deleting the whole table became 'OK: 11 verified'."""
        guide = _guide(tmp_path, "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n")
        rel = guide.name
        errors = mapping_gate.check_pinned({rel: 1}, {rel: 5}, str(tmp_path))
        assert any("at least 5 expected" in e for e in errors)

    def test_a_guide_that_disappears_fails_its_pinned_minimum(
        self, mapping_gate, tmp_path
    ):
        errors = mapping_gate.check_pinned({}, {"gone.md": 3}, str(tmp_path))
        assert any("does not exist" in e for e in errors)

    def test_every_pinned_guide_is_satisfied_by_the_repository(self, mapping_gate):
        errors = mapping_gate.check_pinned(
            {}, mapping_gate.MIN_MAPPINGS, str(REPO_ROOT)
        )
        assert errors == []

    def test_the_repository_mappings_all_resolve(self):
        result = _run("check_impl_mapping")
        assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# check_citations.py
# ---------------------------------------------------------------------------

def _bib_and_doc(tmp_path, bib_text, doc_text):
    bib = tmp_path / "bibliography.bib"
    bib.write_text(bib_text)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(doc_text)
    return bib, docs


class TestCitationGate:
    def test_a_commented_out_bibliography_entry_fails_the_gate(
        self, citations_gate, tmp_path, monkeypatch
    ):
        """Replays: '% @book{Crank1975,' reported OK with exit 0.

        BibTeX ignores a commented entry and Sphinx renders a broken ref, so
        this is the exact case the gate exists for.
        """
        bib, docs = _bib_and_doc(
            tmp_path,
            "% @book{Crank1975,\n  title = {Diffusion},\n}\n",
            "See [@Crank1975].\n",
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 1

    def test_a_duplicate_bibliography_key_fails_the_gate(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib, docs = _bib_and_doc(
            tmp_path,
            "@book{Crank1975,\n}\n@article{Crank1975,\n}\n",
            "See [@Crank1975].\n",
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 1

    def test_a_hyphenated_key_resolves_instead_of_being_mis_tokenised(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib, docs = _bib_and_doc(
            tmp_path,
            "@article{Van-Leer1979,\n  title = {Towards the ultimate scheme},\n}\n",
            "See [@Van-Leer1979].\n",
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 0

    def test_a_bibliography_with_no_citations_anywhere_fails_the_gate(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib, docs = _bib_and_doc(
            tmp_path, "@book{Crank1975,\n}\n", "No citations here.\n"
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 1

    def test_a_comment_entry_type_declares_no_key(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib, docs = _bib_and_doc(
            tmp_path,
            "@comment{Crank1975,\n}\n@book{LeVeque2007,\n}\n",
            "See [@Crank1975] and [@LeVeque2007].\n",
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 1

    def test_the_repository_citations_all_resolve(self):
        result = _run("check_citations")
        assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# check_anomalies.py
# ---------------------------------------------------------------------------

_MINIMAL_ANOMALY = """\
schema_version: "1.0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "{status}"
"""


class TestAnomalyGate:
    def test_an_unrecognised_resolution_status_exits_non_zero(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_MINIMAL_ANOMALY.format(status="probably fine tbh"))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1
        assert "resolution_status" in result.stderr

    def test_a_valid_registry_exits_zero(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_MINIMAL_ANOMALY.format(status="open"))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_repository_registry_passes_with_the_prefix_ci_uses(self):
        result = _run("check_anomalies", "--prefix", "MADD-ANO-")
        assert result.returncode == 0, result.stdout + result.stderr


class TestTransformGateConstantBinding:
    """A name bound to a string constant is still a string reference."""

    def test_a_transform_bound_to_a_module_constant_is_checked(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "indirect.py").write_text(
            'GHOST = "no_such_transform"\n'
            'gm.add_edge("a", "b", "x", "y", transform=GHOST)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_a_constant_naming_a_registered_transform_passes(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "indirect_ok.py").write_text(
            'LAST = "extract_last"\n'
            'gm.add_edge("a", "b", "x", "y", transform=LAST)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    def test_a_callable_passed_by_name_is_not_a_string_reference(
        self, transforms_gate, tmp_path
    ):
        """``transform=my_fn`` is a function object, not a registry lookup."""
        (tmp_path / "callable_arg.py").write_text(
            "def my_fn(x):\n    return x\n"
            'gm.add_edge("a", "b", "x", "y", transform=my_fn)\n'
        )
        # Nothing in scope -> the empty-scope guard, not a false positive.
        assert transforms_gate.main([str(tmp_path)]) == 1


class TestTransformAllowlist:
    """The allowlist must stay a short list of deliberate negative tests.

    Two tests so far assert that `add_edge` rejects an unregistered
    transform, so their names have to be absent from the registry.  A third
    is plausible.  These guards keep the list readable and keep a dead entry
    from sitting there looking deliberate.
    """

    def test_every_entry_carries_a_reason(self, transforms_gate):
        for key, reason in transforms_gate._ALLOWED_UNRESOLVABLE.items():
            assert isinstance(reason, str) and reason.strip(), key

    def test_the_allowlist_stays_small(self, transforms_gate):
        allowlist = transforms_gate._ALLOWED_UNRESOLVABLE
        cap = transforms_gate._MAX_ALLOWED_UNRESOLVABLE
        assert len(allowlist) <= cap, (
            f"{len(allowlist)} allowlisted transform references (cap {cap}). "
            f"Each one is a hole in the gate; fix the call site instead of "
            f"raising the cap."
        )

    def test_no_entry_is_stale(self, transforms_gate):
        """An entry whose file no longer names that transform is dead.

        A file that does not exist yet is fine: an entry may land before the
        branch that introduces the test it exempts.
        """
        for (relpath, name) in transforms_gate._ALLOWED_UNRESOLVABLE:
            path = REPO_ROOT / relpath
            if not path.is_file():
                continue
            refs, _local = transforms_gate.scan_file(path)
            assert name in {n for _lineno, n in refs}, (
                f"{relpath} no longer references transform '{name}'; remove "
                f"the allowlist entry"
            )

    def test_an_allowlisted_pair_does_not_exempt_the_same_name_elsewhere(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "other_file.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="this_does_not_exist")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
