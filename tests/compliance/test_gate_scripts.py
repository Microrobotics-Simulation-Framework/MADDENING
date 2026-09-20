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


@pytest.fixture(scope="module")
def heat_stability_gate():
    return _load("check_heat_stability")


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

    def test_a_scanned_scope_with_no_mappings_fails(self, mapping_gate, tmp_path):
        """It printed "OK: 0 implementation mapping(s) verified" and exited 0.

        The pinned minimums do still run against the repository, so the gate
        was not blind -- but the line it printed named a scope it had
        verified nothing in, and that line is what gets quoted as coverage.
        """
        (tmp_path / "not_a_guide.md").write_text("# No table here\n")
        assert mapping_gate.main([str(tmp_path)]) == 1


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

    def test_the_headline_count_excludes_the_citations_it_declined(self):
        """It reported "50 citation(s) verified" having verified 45.

        The five allowlisted syntax examples are ``continue``d before the
        existence check and were still in the headline.  Verified and
        declined are now two numbers, as check_heat_stability.py's summary
        already did for its unchecked constructions.
        """
        result = _run("check_citations")
        assert result.returncode == 0, result.stdout + result.stderr
        verified = int(
            result.stdout.split("OK: ")[1].split(" citation(s)")[0]
        )
        declined = result.stdout.count(
            "is a syntax example and was NOT checked"
        )
        assert declined > 0, "nothing was declined; the test proves nothing"
        assert f"{declined} not checked" in result.stdout

        gate = _load("check_citations")
        total = len(gate.scan_directory(str(REPO_ROOT / "docs")))
        assert verified == total - declined


class TestCitationTemplateAllowlist:
    """``_TEMPLATE_CITATIONS`` was the one allowlist with no reason, no cap
    and no staleness check.  These are the guards its two neighbours have."""

    def test_every_entry_carries_a_reason(self, citations_gate):
        for key, reason in citations_gate._TEMPLATE_CITATIONS.items():
            assert isinstance(reason, str) and reason.strip(), key

    def test_the_allowlist_stays_small(self, citations_gate):
        allowlist = citations_gate._TEMPLATE_CITATIONS
        cap = citations_gate._MAX_TEMPLATE_CITATIONS
        assert len(allowlist) <= cap, (
            f"{len(allowlist)} allowlisted dangling citations (cap {cap}).  "
            f"Each one is a citation nobody checks; add the key to the "
            f"bibliography instead of raising the cap."
        )

    def test_no_entry_is_stale(self, citations_gate):
        """An entry whose file no longer carries that citation is dead."""
        for (relpath, key) in citations_gate._TEMPLATE_CITATIONS:
            path = REPO_ROOT / relpath
            if not path.is_file():
                continue
            cited = {k for _lineno, k in citations_gate.extract_citations(str(path))}
            assert key in cited, (
                f"{relpath} no longer cites [@{key}]; remove the "
                f"_TEMPLATE_CITATIONS entry"
            )

    def test_an_allowlisted_pair_does_not_exempt_the_same_key_elsewhere(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib, docs = _bib_and_doc(
            tmp_path, "@book{Crank1975,\n}\n",
            "See [@Crank1975] and [@Key].\n",
        )
        monkeypatch.setenv("BIB_PATH", str(bib))
        assert citations_gate.main([str(docs)]) == 1

    def test_a_scope_of_nothing_but_allowlisted_citations_fails(
        self, citations_gate, tmp_path, monkeypatch
    ):
        """Verified zero is not a pass, whatever the headline would say."""
        bib = tmp_path / "bibliography.bib"
        bib.write_text("@book{Crank1975,\n}\n")
        docs = tmp_path / "docs" / "developer_guide"
        docs.mkdir(parents=True)
        (docs / "node_authoring.md").write_text("Cite as [@Key].\n")
        monkeypatch.setenv("BIB_PATH", str(bib))
        monkeypatch.setattr(citations_gate, "_REPO_ROOT", str(tmp_path))
        assert citations_gate.main([str(tmp_path / "docs")]) == 1


# ---------------------------------------------------------------------------
# check_anomalies.py
# ---------------------------------------------------------------------------

# One anomaly carrying one resolvable reference.  The reference is not
# decoration: the gate now refuses a registry whose anomalies declare no
# references at all, because that registry resolves nothing and proves
# nothing, so a fixture without one is no longer a *valid* registry.
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
    affected_components:
      - "maddening.nodes.heat.HeatNode"
"""

_ANOMALY_WITHOUT_REFERENCES = """\
schema_version: "1.0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "open"
"""

_EMPTY_REGISTRY = """\
schema_version: "1.0"
generated_date: "2026-03-12"
anomalies: []
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


class TestAnomalyGateVerifiesSomething:
    """The two guards check_heat_stability.py has and this gate claimed to.

    Its zero-scope guard sat inside ``if notes:``, and ``notes`` holds only
    references *skipped as unavailable* -- so it was empty in exactly the
    case it was meant to catch.  Both shapes below printed
    ``OK: ... is valid`` and exited 0.
    """

    def test_an_empty_registry_fails(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_EMPTY_REGISTRY)
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "no anomalies" in result.stderr

    def test_a_registry_whose_anomalies_declare_no_references_fails(
        self, tmp_path
    ):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_ANOMALY_WITHOUT_REFERENCES)
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "no affected_components and no verification" in result.stderr

    def test_the_empty_registry_guard_survives_no_resolve(self, tmp_path):
        """``--no-resolve`` turns resolution off, not counting."""
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_EMPTY_REGISTRY)
        result = _run("check_anomalies", str(path), "--repo-root",
                      str(REPO_ROOT), "--no-resolve")
        assert result.returncode == 1, result.stdout

    def test_the_summary_separates_verified_from_not_checked(self):
        """One headline count that folds in declined references is how
        "50 citations verified" came to mean 45."""
        result = _run("check_anomalies", "--prefix", "MADD-ANO-")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "reference(s) verified" in result.stdout
        assert "not checked" in result.stdout

    def test_no_resolve_does_not_claim_anything_was_verified(self):
        result = _run("check_anomalies", "--prefix", "MADD-ANO-", "--no-resolve")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "NOT resolved" in result.stdout
        assert "verified" not in result.stdout


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


def _rod(tmp_path, body, name="mod.py"):
    (tmp_path / name).write_text(
        "from maddening.nodes.heat import HeatNode\n" + body + "\n"
    )
    return tmp_path


class TestHeatStabilityGate:
    """``scripts/check_heat_stability.py``: no caller may build a rod that
    ``HeatNode.__init__`` refuses.

    The gate exists because a real one got through code review, two
    greps and a partial local test run, and was caught only by CI:
    ``tests/core/test_compile_cache.py`` built a 257-cell 4th-order rod
    at a Fourier number of 0.66 inside a ``textwrap.dedent`` string
    passed to ``python -c``.  Each test below replays a way of hiding
    such a construction, or a way the gate could claim an OK it has not
    earned.
    """

    def test_it_passes_on_the_repository_as_it_stands(self, heat_stability_gate):
        assert heat_stability_gate.main([]) == 0

    def test_a_rod_past_the_fourth_order_limit_fails(
        self, heat_stability_gate, tmp_path
    ):
        """The exact configuration CI caught: Fo = 0.66 against 5/16."""
        _rod(tmp_path, 'HeatNode("h", 1e-4, n_cells=257, '
                       'thermal_diffusivity=0.1, stencil_order=4)')
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_a_rod_past_the_second_order_limit_fails(
        self, heat_stability_gate, tmp_path
    ):
        _rod(tmp_path, 'HeatNode("h", 0.51, n_cells=10, length=1.0, '
                       'thermal_diffusivity=1.0)')
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_a_rod_hidden_in_a_dedented_string_still_fails(
        self, heat_stability_gate, tmp_path
    ):
        """The reason this gate walks the AST instead of grepping.

        ``HeatNode(`` inside a string literal is not a call to anything
        scanning for calls, and the string is indented until
        ``textwrap.dedent`` runs, so it does not even parse as source
        until the gate dedents it.  This is how the real one hid.
        """
        (tmp_path / "child.py").write_text(
            'import textwrap\n'
            '_CHILD = textwrap.dedent("""\n'
            '    from maddening.nodes.heat import HeatNode\n'
            '    def factory():\n'
            '        return HeatNode("h", 1e-4, n_cells=257,\n'
            '                        thermal_diffusivity=0.1, stencil_order=4)\n'
            '""")\n'
        )
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_a_stable_rod_passes(self, heat_stability_gate, tmp_path):
        _rod(tmp_path, 'HeatNode("h", 1e-5, n_cells=257, '
                       'thermal_diffusivity=0.1, stencil_order=4)')
        assert heat_stability_gate.main([str(tmp_path)]) == 0

    def test_it_refuses_to_pass_when_it_verified_nothing(
        self, heat_stability_gate, tmp_path
    ):
        """A computed argument is unchecked, and unchecked is not a pass.

        A cruder regex sweep of the same question produced three false
        positives by trying to read expressions like
        ``thermal_diffusivity=1.0/n_cells**2``.  This gate declines to
        evaluate them -- but a file whose every construction is
        computed means the gate verified nothing, and it says so rather
        than reporting OK.
        """
        _rod(tmp_path, 'nc = 257\n'
                       'HeatNode("h", 1.0 / 3.0, n_cells=nc, '
                       'thermal_diffusivity=0.1, stencil_order=4)')
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_an_empty_scope_fails(self, heat_stability_gate, tmp_path):
        """A gate that verifies nothing cannot fail."""
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_limits_come_from_the_node_not_a_copy(self, heat_stability_gate):
        """A second source of truth would drift from the guard silently."""
        from maddening.nodes.heat import MAX_FOURIER_NUMBER

        assert heat_stability_gate.MAX_FOURIER_NUMBER is MAX_FOURIER_NUMBER

    def test_the_allowlisted_file_still_plants_a_defect(
        self, heat_stability_gate
    ):
        """An exemption that no longer exempts anything is scope creep.

        This file is allowlisted because it must contain unstable rods
        to prove the gate rejects them.  If those fixtures ever move or
        change, the exemption stops being earned and has to go -- an
        allowlist entry left behind would silently stop checking a real
        file.
        """
        for relpath in heat_stability_gate._ALLOWED_UNSTABLE:
            path = REPO_ROOT / relpath
            if not path.is_file():
                continue
            unstable, unchecked, seen = [], [], []
            heat_stability_gate.scan_source(
                path.read_text(), relpath,
                heat_stability_gate._defaults(), unstable, unchecked, seen,
            )
            assert unstable, (
                f"{relpath} is allowlisted as deliberately containing an "
                f"unstable HeatNode construction, but no longer does; remove "
                f"the _ALLOWED_UNSTABLE entry"
            )

    def test_the_allowlist_does_not_exempt_an_ordinary_file(
        self, heat_stability_gate, tmp_path
    ):
        """The exemption is by exact path, not by resemblance to one."""
        _rod(tmp_path, 'HeatNode("h", 1e-4, n_cells=257, '
                       'thermal_diffusivity=0.1, stencil_order=4)',
             name="test_gate_scripts.py")
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_parameter_defaults_come_from_the_constructor_signature(
        self, heat_stability_gate
    ):
        """Same reason: a hard-coded default would go stale on a rename."""
        import inspect

        from maddening.nodes.heat import HeatNode

        sig = inspect.signature(HeatNode.__init__)
        defaults = heat_stability_gate._defaults()
        for key, value in defaults.items():
            assert value == sig.parameters[key].default


class TestHeatStabilityCallForms:
    """A construction the constructor accepts must be a construction the
    gate can see, however it is spelled.

    Three spellings were invisible.  Each is replayed here against a rod
    ``HeatNode.__init__`` genuinely refuses, so a regression is a gate that
    passes a build that cannot run.
    """

    def test_a_positional_stencil_order_is_judged_against_that_order(
        self, heat_stability_gate, tmp_path
    ):
        """``stencil_order`` is the 7th parameter; the list stopped at the 5th.

        Fourier 0.4 is stable at order 2 (limit 0.5) and unstable at order 4
        (limit 0.3125).  With the order invisible it defaulted to 2 and the
        gate passed a rod the constructor refuses.
        """
        _rod(tmp_path, 'HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 4)')
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_same_rod_at_order_two_still_passes(
        self, heat_stability_gate, tmp_path
    ):
        """The failure above is the order, not the widened parameter list."""
        _rod(tmp_path, 'HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 2)')
        assert heat_stability_gate.main([str(tmp_path)]) == 0

    def test_an_attribute_spelled_construction_is_seen(
        self, heat_stability_gate, tmp_path
    ):
        """``ast.Attribute`` carries ``.attr``, not ``.id``."""
        (tmp_path / "attr_form.py").write_text(
            "import maddening.nodes.heat as heat\n"
            'n = heat.HeatNode("e", timestep=1e-4, n_cells=257, length=1.0,\n'
            "                  thermal_diffusivity=0.1, stencil_order=4)\n"
        )
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_an_aliased_import_is_seen(self, heat_stability_gate, tmp_path):
        (tmp_path / "alias_form.py").write_text(
            "from maddening.nodes.heat import HeatNode as Rod\n"
            'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
            "        thermal_diffusivity=0.1)\n"
        )
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_a_rebound_name_is_seen(self, heat_stability_gate, tmp_path):
        """``Rod = HeatNode`` is the rebinding an import alias avoids."""
        (tmp_path / "rebound.py").write_text(
            "from maddening.nodes.heat import HeatNode\n"
            "Rod = HeatNode\n"
            'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
            "        thermal_diffusivity=0.1)\n"
        )
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_positional_order_is_the_constructor_signature_order(
        self, heat_stability_gate
    ):
        """A hand-maintained list is what went short by two parameters."""
        import inspect

        from maddening.nodes.heat import HeatNode

        expected = [
            name for name, param
            in inspect.signature(HeatNode.__init__).parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        assert heat_stability_gate._POSITIONAL == expected
        assert "stencil_order" in heat_stability_gate._POSITIONAL


class TestHeatStabilityCounts:
    """``seen`` is the number the summary calls "verified", so nothing may
    reach it without having been evaluated.

    ``seen.append`` ran before both ``continue``s, so a rod with a
    non-positive argument and one with an unknown stencil order were counted
    as verified having had no Fourier number computed: 132 reported against
    131 evaluated.
    """

    def _scan(self, gate, source):
        unstable, unchecked, seen = [], [], []
        gate.scan_source(source, "probe.py", gate._defaults(),
                         unstable, unchecked, seen)
        return unstable, unchecked, seen

    def test_a_non_positive_argument_is_not_counted_as_verified(
        self, heat_stability_gate
    ):
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("a", timestep=1.0, n_cells=10, length=1.0,'
            " thermal_diffusivity=0.0)\n",
        )
        assert seen == []
        assert len(unchecked) == 1
        assert "non-positive" in unchecked[0][2]

    def test_an_unknown_stencil_order_is_not_counted_as_verified(
        self, heat_stability_gate
    ):
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("c", timestep=1.0, n_cells=10, length=1.0,'
            " thermal_diffusivity=100.0, stencil_order=3)\n",
        )
        assert seen == []
        assert len(unchecked) == 1
        assert "MAX_FOURIER_NUMBER" in unchecked[0][2]

    def test_an_evaluated_rod_is_counted_as_verified(
        self, heat_stability_gate
    ):
        """The other direction: the counter still counts what it should."""
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("ok", timestep=1e-5, n_cells=10, length=1.0,'
            " thermal_diffusivity=0.01)\n",
        )
        assert len(seen) == 1 and unchecked == [] and unstable == []


class TestHeatStabilityAllowlist:
    def test_every_entry_carries_a_reason(self, heat_stability_gate):
        for path, reason in heat_stability_gate._ALLOWED_UNSTABLE.items():
            assert isinstance(reason, str) and reason.strip(), path

    def test_the_allowlist_stays_small(self, heat_stability_gate):
        allowlist = heat_stability_gate._ALLOWED_UNSTABLE
        cap = heat_stability_gate._MAX_ALLOWED_UNSTABLE
        assert len(allowlist) <= cap, (
            f"{len(allowlist)} allowlisted files (cap {cap}).  Each one is a "
            f"whole file this gate stops reading; fix the rod instead of "
            f"raising the cap."
        )
