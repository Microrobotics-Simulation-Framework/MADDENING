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
import json
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
def repo_transforms_gate_run():
    """One run of the transforms gate over the repository, as CI runs it.

    Four tests read its output.  A run costs about 5 s on the CI runner,
    nearly all of it importing the modules whose registrations it
    confirms, and it is deterministic for a given tree -- so they share
    one run rather than paying for four.  ``--allow-missing-optional``:
    the test matrix installs only ``[ci]``, so the USD test modules
    cannot be imported here.  The CI compliance job runs the gate without
    the flag, with the extras.
    """
    return _run("check_transforms", "--allow-missing-optional")


@pytest.fixture(scope="module")
def mapping_gate():
    return _load("check_impl_mapping")


@pytest.fixture(scope="module")
def citations_gate():
    return _load("check_citations")


@pytest.fixture(scope="module")
def anomalies_gate():
    return _load("check_anomalies")


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
        # The fixture has to be importable: the gate confirms a local
        # registration by importing the module.  This one used to lack the
        # import and call ``gm`` at module level, so it raised NameError on
        # import and passed only because every import failure degraded to
        # "unconfirmed" -- it never exercised the path it is named for.
        (tmp_path / "registers_its_own.py").write_text(
            "from maddening.core.transforms import register_transform\n"
            '@register_transform("locally_defined")\n'
            "def _t(x):\n"
            "    return x\n"
            '\n'
            "def wire(gm):\n"
            '    gm.add_edge("a", "b", "x", "y", transform="locally_defined")\n'
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

    def test_the_repository_transform_references_all_resolve(
        self, repo_transforms_gate_run
    ):
        result = repo_transforms_gate_run
        assert result.returncode == 0, result.stdout + result.stderr
        # Regression guard on the audit finding: the gate reported
        # "OK: 0 ... verified" for the whole of v0.3 and v0.4-dev.
        assert " 0 string transform reference(s) verified" not in result.stdout


class TestTransformPositionalForm:
    """``transform`` is a positional parameter, and a string there resolves.

    The gate read ``node.keywords`` only, so the identical call written
    positionally was invisible -- and nothing in the tree uses that form
    today, which is what makes it a trap rather than a live bug.
    """

    def test_a_positional_unregistered_transform_fails_the_gate(
        self, transforms_gate, tmp_path, capsys
    ):
        # The extract_last line keeps the scope non-empty, so the exit 1 is
        # the ghost and not the empty-scope floor.
        (tmp_path / "positional_ghost.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
            'gm.add_edge("a", "b", "x", "y", "no_such_transform")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

    def test_a_positional_registered_transform_passes(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "positional_ok.py").write_text(
            'gm.add_edge("a", "b", "x", "y", "extract_last")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    def test_a_positional_edge_spec_transform_is_seen(
        self, transforms_gate, tmp_path, capsys
    ):
        (tmp_path / "spec_positional.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
            'EdgeSpec("a", "b", "x", "y", "no_such_transform")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

    def test_a_fifth_positional_that_is_not_a_string_is_not_a_reference(
        self, transforms_gate, tmp_path
    ):
        """``add_edge(..., my_fn)`` passes a callable, not a registry key."""
        (tmp_path / "callable_positional.py").write_text(
            "def my_fn(x):\n    return x\n"
            'gm.add_edge("a", "b", "x", "y", my_fn)\n'
        )
        # Nothing in scope -> the empty-scope guard, not a false positive.
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_the_positional_index_is_the_one_both_signatures_have(
        self, transforms_gate
    ):
        """A hand-written index is what a parameter reorder would break."""
        import dataclasses
        import inspect

        from maddening.core.edge import EdgeSpec
        from maddening.core.graph_manager import GraphManager

        index = transforms_gate._TRANSFORM_POSITION
        params = [
            name for name, param
            in inspect.signature(GraphManager.add_edge).parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]
        assert params[index] == "transform"
        fields = [f.name for f in dataclasses.fields(EdgeSpec)]
        assert fields[index] == "transform"


class TestTransformLiveRegistration:
    """A lexical ``@register_transform`` is a claim; the registry is the fact.

    ``find_local_registrations`` walks the whole AST for the call
    expression, so a registration inside a function nobody calls satisfied
    the gate while the name was absent from the registry after import and
    ``add_edge`` would raise ``KeyError``.
    """

    _DEAD = (
        "from maddening.core.transforms import register_transform\n"
        "\n"
        "\n"
        "def _install_later():\n"
        '    """Never called -- e.g. a helper a fixture forgot to invoke."""\n'
        '    @register_transform("phantom_transform")\n'
        "    def _phantom(x):\n"
        "        return x\n"
        "\n"
        "\n"
        "\n"
        "def wire(gm):\n"
        '    gm.add_edge("a", "b", "x", "y", transform="phantom_transform")\n'
    )

    _LIVE = (
        "from maddening.core.transforms import register_transform\n"
        "\n"
        "\n"
        '@register_transform("really_registered_transform")\n'
        "def _t(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "\n"
        "def wire(gm):\n"
        '    gm.add_edge("a", "b", "x", "y", '
        'transform="really_registered_transform")\n'
    )

    def test_a_registration_that_never_executes_fails_the_gate(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "dead_registration.py").write_text(self._DEAD)
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_the_error_says_the_registration_never_executes(
        self, transforms_gate, tmp_path, capsys
    ):
        (tmp_path / "dead_registration.py").write_text(self._DEAD)
        transforms_gate.main([str(tmp_path)])
        assert "never executes" in capsys.readouterr().out

    def test_a_module_level_registration_passes(
        self, transforms_gate, tmp_path
    ):
        """The other direction: a real registration is confirmed, not merely
        tolerated."""
        (tmp_path / "live_registration.py").write_text(self._LIVE)
        assert transforms_gate.main([str(tmp_path)]) == 0

    _NEEDS_AN_EXTRA = (
        "import a_module_that_does_not_exist_anywhere  # noqa: F401\n"
        "from maddening.core.transforms import register_transform\n"
        "\n"
        "\n"
        '@register_transform("transform_behind_an_extra")\n'
        "def _t(x):\n"
        "    return x\n"
        "\n"
        "\n"
        "def wire(gm):\n"
        '    gm.add_edge("a", "b", "x", "y", '
        'transform="transform_behind_an_extra")\n'
    )

    def test_a_scope_whose_only_reference_is_unconfirmed_fails(
        self, transforms_gate, tmp_path, capsys
    ):
        """Nothing verified is a failure, whatever else was found.

        The floor used to be on references *in scope*, so this printed
        "OK: 0 string transform reference(s) verified, 1 not confirmed
        against the live registry" and exited 0 (audit_040_phase3_wave_d,
        T5).  ``--allow-missing-optional`` must not reopen it.
        """
        (tmp_path / "needs_an_extra.py").write_text(self._NEEDS_AN_EXTRA)
        for extra in ([], ["--allow-missing-optional"]):
            assert transforms_gate.main([*extra, str(tmp_path)]) == 1, extra
            captured = capsys.readouterr()
            assert "FAIL: 0 string transform reference(s) verified" in (
                captured.err), captured
            assert "1 not confirmed against the live registry" in captured.err
            assert "OK:" not in captured.out

    def test_an_unconfirmed_reference_fails_unless_explicitly_accepted(
        self, transforms_gate, tmp_path, capsys
    ):
        """Unconfirmed is not verified, and by default not a pass either.

        The CI job that runs this gate installs the extras precisely so
        that nothing is unconfirmed; an unconfirmed reference there means
        the gate quietly started verifying less.  A lane without the extras
        opts in, and the reference is still reported.
        """
        (tmp_path / "needs_an_extra.py").write_text(self._NEEDS_AN_EXTRA)
        (tmp_path / "verifiable.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 2
        captured = capsys.readouterr()
        assert "could not be confirmed" in captured.err
        assert "--allow-missing-optional" in captured.err
        assert "NOT confirmed against the live registry" in captured.out
        assert "a_module_that_does_not_exist_anywhere" in captured.out

        assert transforms_gate.main(
            ["--allow-missing-optional", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "NOT confirmed against the live registry" in out
        assert ("OK: 1 string transform reference(s) verified, 1 not "
                "confirmed against the live registry") in out, out

    def test_a_module_that_raises_at_import_is_broken_not_unconfirmed(
        self, transforms_gate, tmp_path, capsys
    ):
        """Only a missing third-party package is the environment's fault.

        Every import failure used to degrade to "unconfirmed", so a module
        that raises on import -- whose registrations therefore run nowhere
        -- was reported as merely unchecked.
        """
        (tmp_path / "raises.py").write_text(
            "from maddening.core.transforms import register_transform\n"
            '@register_transform("registered_before_the_crash")\n'
            "def _t(x):\n"
            "    return x\n"
            'raise RuntimeError("a module-level bug")\n'
            "def wire(gm):\n"
            '    gm.add_edge("a", "b", "x", "y", '
            'transform="registered_before_the_crash")\n'
        )
        (tmp_path / "verifiable.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        for extra in ([], ["--allow-missing-optional"]):
            assert transforms_gate.main([*extra, str(tmp_path)]) == 1, extra
            out = capsys.readouterr().out
            assert "fails to import" in out and "a module-level bug" in out

    def test_a_missing_first_party_module_is_broken_not_unconfirmed(
        self, transforms_gate, tmp_path, capsys
    ):
        """``maddening.<gone>`` is a broken import, not an optional extra."""
        (tmp_path / "stale_import.py").write_text(
            "import maddening.no_such_submodule_anywhere  # noqa: F401\n"
            "from maddening.core.transforms import register_transform\n"
            '@register_transform("behind_a_stale_import")\n'
            "def _t(x):\n"
            "    return x\n"
            "def wire(gm):\n"
            '    gm.add_edge("a", "b", "x", "y", '
            'transform="behind_a_stale_import")\n'
        )
        (tmp_path / "verifiable.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        assert transforms_gate.main(
            ["--allow-missing-optional", str(tmp_path)]) == 1
        assert "fails to import" in capsys.readouterr().out

    def test_a_subpackage_refusing_its_extra_is_unconfirmed(
        self, transforms_gate, tmp_path
    ):
        """The re-raised form ``maddening.usd`` uses is still a missing extra.

        It raises ``ImportError("... requires 'usd-core'")`` from the
        ``ModuleNotFoundError``, and carries no ``name`` of its own; the
        classification walks the chain.
        """
        exc = ImportError("maddening.usd requires 'usd-core'")
        exc.__cause__ = ModuleNotFoundError("No module named 'pxr'",
                                            name="pxr")
        assert transforms_gate.missing_optional_package(exc, REPO_ROOT) == "pxr"
        first_party = ModuleNotFoundError("No module named 'maddening.gone'",
                                          name="maddening.gone")
        assert transforms_gate.missing_optional_package(
            first_party, REPO_ROOT) is None
        assert transforms_gate.missing_optional_package(
            RuntimeError("no"), REPO_ROOT) is None

    def test_loading_the_same_probe_twice_does_not_collide(
        self, transforms_gate, tmp_path
    ):
        """A probe load re-executes the module; the registry is put back.

        Without that, the second load re-registers the name to a new
        function object, ``register_transform`` raises, and a correct probe
        is reported as broken -- which the property tests, loading many
        probes in one process, would hit at random.
        """
        (tmp_path / "live_registration.py").write_text(self._LIVE)
        assert transforms_gate.main([str(tmp_path)]) == 0
        assert transforms_gate.main([str(tmp_path)]) == 0
        from maddening.core.transforms import _TRANSFORM_REGISTRY
        assert "really_registered_transform" not in _TRANSFORM_REGISTRY

    def test_an_unconfirmed_reference_is_not_counted_in_the_verified_total(
        self, transforms_gate, tmp_path, capsys
    ):
        """Two references, one verifiable: the headline says one, not two.

        ``n_allowlisted`` ``continue``s before the in-scope counter, but
        the unconfirmed were counted *before* the live-registry loop had
        decided which references they were -- two neighbouring counters
        computed differently, so the degradation path inflated the very
        number it printed a caveat next to (audit_040_r3).  The counter it
        inflated arrived in 44250c3, the fix for this same defect class.
        """
        (tmp_path / "uses_transforms.py").write_text(
            "import a_module_that_does_not_exist_anywhere  # noqa: F401\n"
            "from maddening.core.transforms import register_transform\n"
            "\n"
            "\n"
            '@register_transform("only_lexically_registered")\n'
            "def _t(x):\n"
            "    return x\n"
            "\n"
            "\n"
            "def wire(gm):\n"
            '    gm.add_edge("a", "b", "x", "y", '
            'transform="only_lexically_registered")\n'
            '    gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        assert transforms_gate.main(
            ["--allow-missing-optional", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        # Not just the exit code, and not just a substring of the caveat:
        # the number in the headline is what a reader takes away as
        # coverage, so assert the whole clause.
        assert ("OK: 1 string transform reference(s) verified, 1 not "
                "confirmed against the live registry") in out, out
        assert "2 string transform reference(s) verified" not in out

    def test_an_allowlisted_reference_is_not_counted_in_the_verified_total(
        self, transforms_gate, tmp_path, capsys, monkeypatch
    ):
        """The other counter the headline must not absorb.

        ``n_allowlisted`` is the one that was already right, and nothing
        pinned it: adding the allowlisted reference to the in-scope total
        as well makes the repository report "33 verified, 2 allowlisted
        and not checked" for 31 verified references, and every other test
        in this file still passes.  Written because the consequence -- a
        gate overstating its own coverage in IEC 62304 evidence -- is the
        defect class this whole module exists for.
        """
        probe = tmp_path / "allowlisted_and_not.py"
        probe.write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
            'gm.add_edge("a", "b", "x", "y", transform="deliberately_absent")\n'
        )
        # A scan root outside the repository keeps its absolute path as
        # the allowlist key (`relative_to` raises and `rel` falls back).
        monkeypatch.setitem(
            transforms_gate._ALLOWED_UNRESOLVABLE,
            (str(probe), "deliberately_absent"),
            "the fixture for this test",
        )
        assert transforms_gate.main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert ("OK: 1 string transform reference(s) verified, 1 "
                "allowlisted and not checked") in out, out

    #: Test packages whose modules need an optional extra to import.  A
    #: registration in one of these is legitimately unconfirmable in a CI
    #: that installs only ``[ci]`` -- the compliance job installs
    #: ``[ci,usd]`` precisely so the gates can see what they verify, but the
    #: matrix job that runs the whole suite does not.
    _OPTIONAL_EXTRA_PACKAGES = ("tests/usd/", "tests/viz/", "tests/cloud/")

    def test_every_local_registration_outside_an_optional_extra_is_confirmed(
        self, repo_transforms_gate_run
    ):
        """A NOTE is acceptable only where an extra explains it.

        Anywhere else it means the gate degraded on a module it should have
        been able to import, which is indistinguishable from the dead
        registration this check exists to catch.
        """
        result = repo_transforms_gate_run
        assert result.returncode == 0, result.stdout + result.stderr
        unexplained = [
            line for line in result.stdout.splitlines()
            if "NOT confirmed against the live registry" in line
            and not any(pkg in line for pkg in self._OPTIONAL_EXTRA_PACKAGES)
        ]
        assert not unexplained, (
            "a local @register_transform could not be confirmed in a module "
            "that needs no optional extra; the gate degraded rather than "
            "verified:\n" + "\n".join(unexplained)
        )

    def test_the_reported_registry_size_excludes_what_the_gate_imported(
        self, repo_transforms_gate_run
    ):
        """The live check imports modules that register transforms.

        Those registrations are global, so the registry must be snapshotted
        before any of them run -- otherwise a name one module registers
        starts satisfying another module's reference, which is exactly what
        "registered in another file does not count" forbids.  The headline
        count is the visible half of that snapshot.
        """
        result = repo_transforms_gate_run
        assert result.returncode == 0, result.stdout + result.stderr
        reported = int(
            result.stdout.rsplit("(", 1)[1].split(" transforms")[0]
        )
        probe = subprocess.run(
            [sys.executable, "-c",
             "from maddening.core.transforms import _TRANSFORM_REGISTRY;"
             "print(len(_TRANSFORM_REGISTRY))"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"),
                 "JAX_PLATFORMS": "cpu"},
        )
        assert reported == int(probe.stdout.strip()), (
            f"the gate reported {reported} transforms in the registry; a "
            f"bare import registers {probe.stdout.strip()}.  The gate is "
            f"counting names its own imports added."
        )

    def test_the_summary_separates_allowlisted_from_verified(
        self, repo_transforms_gate_run
    ):
        result = repo_transforms_gate_run
        assert result.returncode == 0, result.stdout + result.stderr
        assert "allowlisted and not checked" in result.stdout


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
        self, mapping_gate, tmp_path, capsys
    ):
        _guide(
            tmp_path,
            "| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | "
            "Inherited from `SimulationNode`: the base serialises every "
            "node |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 0
        assert "inherited from SimulationNode, as the row states" in (
            capsys.readouterr().out)

    @pytest.mark.parametrize("notes", [
        "not inherited",                                    # the audit's M2
        "Not inherited from `SimulationNode`",
        "inherited from SimulationNode",                    # the old spelling
        "see below. Inherited from `SimulationNode`",       # not at the start
        "Inherited: from the base class",
    ])
    def test_prose_about_inheritance_is_not_a_marker(
        self, mapping_gate, tmp_path, notes
    ):
        """Only a Notes cell that begins ``Inherited from `Base``` opts in.

        The check was ``"inherited" in row_text.lower()``, so any mention
        -- including "not inherited" -- switched the own-class check off
        (audit_040_phase3_wave_d, M2).
        """
        _guide(
            tmp_path,
            "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n",
            f"| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | "
            f"{notes} |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_a_marker_naming_the_wrong_base_fails(
        self, mapping_gate, tmp_path, capsys
    ):
        _guide(
            tmp_path,
            "| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | "
            "Inherited from `BallNode` |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "resolves through SimulationNode" in capsys.readouterr().err

    def test_a_marker_on_a_row_whose_symbol_is_defined_on_its_class_fails(
        self, mapping_gate, tmp_path, capsys
    ):
        """The marker is a claim; an override makes it false."""
        _guide(
            tmp_path,
            "| Diffusion | `maddening.nodes.heat.HeatNode.update` | "
            "Inherited from `SimulationNode` |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "drop the marker" in capsys.readouterr().err

    def test_the_failure_says_how_to_mark_an_intended_inheritance(
        self, mapping_gate, tmp_path, capsys
    ):
        _guide(tmp_path,
               "| Serialisation | `maddening.nodes.heat.HeatNode.to_dict` | |\n")
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "Inherited from `SimulationNode`" in capsys.readouterr().err

    def test_an_implementation_span_without_the_qualifier_fails(
        self, mapping_gate, tmp_path, capsys
    ):
        """audit_040_phase3_wave_d, M3: a dropped ``maddening.`` prefix left
        a code span the gate never resolved, and within a pin's slack the
        gate stayed green."""
        _guide(
            tmp_path,
            "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n",
            "| Non-uniform Laplacian | `_laplacian_nonuniform` | |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "`_laplacian_nonuniform`" in capsys.readouterr().err

    def test_a_declared_jax_primitive_is_reported_not_verified(
        self, mapping_gate, tmp_path, capsys
    ):
        """The documented convention for a term no MADDENING function owns."""
        _guide(
            tmp_path,
            "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n",
            "| Boundary conditions | `state.at[0].set(left_T)` | "
            "JAX primitive: `jax.numpy.ndarray.at[].set()` |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "declared a JAX primitive" in out
        assert "OK: 1 implementation mapping(s) verified, 1 not checked" in out

    def test_a_scope_of_only_declared_primitives_fails(
        self, mapping_gate, tmp_path
    ):
        _guide(
            tmp_path,
            "| Boundary conditions | `state.at[0].set(left_T)` | "
            "JAX primitive: `jax.numpy.ndarray.at[].set()` |\n",
        )
        assert mapping_gate.main([str(tmp_path)]) == 1

    def test_a_code_span_in_the_notes_column_need_not_be_qualified(
        self, mapping_gate, tmp_path
    ):
        """The rule is for the Implementation column, the claim itself."""
        _guide(
            tmp_path,
            "| Diffusion | `maddening.nodes.heat.HeatNode.update` | "
            "applies `alpha * dt` |\n",
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


class TestMinMappingsRatchet:
    """``MIN_MAPPINGS`` must not be lowerable from inside one file.

    It is the only protection the Implementation Mapping tables have, and it
    lives in the file an author editing a guide is already editing: dropping
    ``heat_node.md`` from 9 to 1 and deleting 8 of its 9 rows left the gate
    green and every mapping test green with it, because
    ``test_every_pinned_guide_is_satisfied_by_the_repository`` reads the
    *current* ``MIN_MAPPINGS`` and moves with the mutation.

    The floor beside this file is the second half of the ratchet: lowering a
    pin now takes an edit to two files in opposite directions.
    """

    @staticmethod
    def _floor():
        with open(Path(__file__).parent / "min_mappings_floor.json") as fh:
            return {
                os.path.normpath(path): value
                for path, value in json.load(fh)["floor"].items()
            }

    @staticmethod
    def _pins(mapping_gate):
        return {
            os.path.normpath(path): value
            for path, value in mapping_gate.MIN_MAPPINGS.items()
        }

    def test_no_pin_is_below_its_committed_floor(self, mapping_gate):
        pins, floor = self._pins(mapping_gate), self._floor()
        lowered = {
            path: (pins[path], minimum)
            for path, minimum in floor.items()
            if path in pins and pins[path] < minimum
        }
        assert not lowered, (
            f"MIN_MAPPINGS has been lowered below its committed floor: "
            f"{lowered} (pin, floor).  A guide that legitimately shrank needs "
            f"both numbers lowered, in one commit, with the reason."
        )

    def test_every_pin_has_a_floor(self, mapping_gate):
        """Otherwise a new guide could be pinned at 1 and never ratchet."""
        missing = set(self._pins(mapping_gate)) - set(self._floor())
        assert not missing, (
            f"pinned in MIN_MAPPINGS with no entry in "
            f"min_mappings_floor.json: {sorted(missing)}"
        )

    def test_every_floor_has_a_pin(self, mapping_gate):
        """Deleting the pin must not be a way round the floor."""
        missing = set(self._floor()) - set(self._pins(mapping_gate))
        assert not missing, (
            f"floored in min_mappings_floor.json but no longer pinned in "
            f"MIN_MAPPINGS: {sorted(missing)}.  Removing a pin removes the "
            f"only check on that guide's table."
        )

    def test_the_floor_itself_is_satisfied_by_the_repository(self, mapping_gate):
        """The floor is a claim about the tree, not a number in a file."""
        assert mapping_gate.check_pinned({}, self._floor(), str(REPO_ROOT)) == []

    def test_every_pin_equals_its_committed_floor(self, mapping_gate):
        """Equal, not ``>=``: slack under a pin in the floor file lets the
        pin drop to it together with the rows."""
        pins, floor = self._pins(mapping_gate), self._floor()
        differ = {path: (pins.get(path), floor.get(path))
                  for path in set(pins) | set(floor)
                  if pins.get(path) != floor.get(path)}
        assert not differ, (
            f"MIN_MAPPINGS and min_mappings_floor.json disagree: {differ} "
            f"(pin, floor).  Move both, in one commit.")

    @staticmethod
    def _guide_counts(mapping_gate):
        counts = {}
        guide_dir = REPO_ROOT / "docs" / "algorithm_guide"
        for path in sorted(guide_dir.rglob("*.md")):
            if path.name.startswith("_"):
                continue
            rel = os.path.normpath(str(path.relative_to(REPO_ROOT)))
            n, errors, _notes, skipped = mapping_gate.check_guide(str(path), rel)
            assert not errors and not skipped, (rel, errors, skipped)
            counts[rel] = n
        return counts

    def test_every_pin_is_its_guides_count(self, mapping_gate):
        """A pin below its count is slack a row deletion hides in.

        heat, lbm and wavelet sat two below (11/13, 22/24, 24/26), so
        deleting two rows from any of them passed, against a docstring and
        a CHANGELOG saying the pins sat at the counts (audit_040_p4_1,
        M5-M8).  Raise the pin in the commit that adds the rows.
        """
        counts = self._guide_counts(mapping_gate)
        slack = {path: (pin, counts.get(path))
                 for path, pin in self._pins(mapping_gate).items()
                 if counts.get(path) != pin}
        assert not slack, f"(pin, rows found) per guide: {slack}"

    def test_every_guide_with_a_mapping_table_is_pinned(self, mapping_gate):
        """An unpinned guide is all slack: its whole table can go."""
        unpinned = {path: n for path, n in self._guide_counts(mapping_gate).items()
                    if n and path not in self._pins(mapping_gate)}
        assert not unpinned, (
            f"guides with Implementation Mapping rows and no MIN_MAPPINGS "
            f"pin: {unpinned}; pin each at its count, here and in "
            f"min_mappings_floor.json")


def _delete_heat_rows(text, n):
    needles = ["`maddening.nodes.heat._laplacian_nonuniform`",
               "`maddening.nodes.heat._laplacian_4th_order_uniform`"][:n]
    return _delete_rows_containing(text, needles)


def _delete_rows_containing(text, needles):
    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if not any(n in line for n in needles)]
    assert len(lines) - len(kept) == len(needles), "a replayed row has moved"
    return "".join(kept)


def _delete_last_single_reference_rows(text, n):
    import re

    section = re.search(r"## Implementation Mapping\s*\n(.*?)(?=\n## |\Z)",
                        text, re.S).group(1)
    rows = [line for line in section.splitlines()
            if line.startswith("| ") and line.count("`maddening.") == 1]
    return _delete_rows_containing(text, rows[-n:])


@pytest.mark.parametrize("guide, mutate", [
    pytest.param("heat_node.md", lambda t: _delete_heat_rows(t, 1), id="M5"),
    pytest.param("heat_node.md", lambda t: _delete_heat_rows(t, 2), id="M6"),
    pytest.param("wavelet_adaptive_node.md", lambda t: _delete_rows_containing(
        t, ["| Analysis $c = W^{-1} u$ |", "| Dirichlet basis |"]), id="M7"),
    pytest.param("lbm_node.md",
                 lambda t: _delete_last_single_reference_rows(t, 2), id="M8"),
])
def test_deleting_rows_from_a_pinned_guide_fails_the_gate(
    mapping_gate, tmp_path, monkeypatch, guide, mutate
):
    """audit_040_p4_1, M5-M8: row deletions inside a pin's slack passed.

    The guide is copied to a scratch repository root at its real relative
    path, so the gate's own pin for it applies, and the gate runs over that
    one guide: unmutated it passes, mutated it must exit non-zero.  The
    node-ID uniqueness scan is stubbed out: it reads all of ``src`` (about
    2 s) and is not what these mutants test.
    """
    monkeypatch.setattr(mapping_gate, "algorithm_id_errors",
                        lambda _src: (0, []))
    rel = os.path.join("docs", "algorithm_guide", "nodes", guide)
    target = tmp_path / rel
    target.parent.mkdir(parents=True)
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    monkeypatch.setattr(mapping_gate, "_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(mapping_gate, "MIN_MAPPINGS",
                        {rel: mapping_gate.MIN_MAPPINGS[rel]})

    target.write_text(text, encoding="utf-8")
    assert mapping_gate.main([str(target.parent)]) == 0

    target.write_text(mutate(text), encoding="utf-8")
    assert mapping_gate.main([str(target.parent)]) == 1


def _node_guide(tmp_path, title, module, id_lines, name="node_guide.md"):
    """A guide whose header names a node, followed by one resolvable row."""
    path = tmp_path / name
    path.write_text(
        f"# {title}\n\n**Module**: `{module}`\n" + "".join(id_lines) + "\n"
        + _TABLE_HEADER
        + "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n"
        + "\n## Next Section\n"
    )
    return path


def _src_tree(tmp_path, **modules):
    """A throwaway package for the algorithm-ID scan: ``name=source``."""
    root = tmp_path / "src_pkg"
    root.mkdir(exist_ok=True)
    for name, source in modules.items():
        (root / f"{name}.py").write_text(source)
    return str(root)


class TestNodeAlgorithmIds:
    """Every node algorithm ID is unique, and a guide's ID is its node's.

    ``LBMNode`` and ``RigidBodyNode`` both carried ``MADD-NODE-007`` from
    0.1.0 to 0.3.1.  With a second duplicate seeded (``SpringDamperNode``
    taking ``BallNode``'s ``MADD-NODE-001``) and the heat guide stating
    ``MADD-NODE-006`` for a node that carries ``MADD-NODE-005``, all seven
    gates, the SOUP ``--check`` and every compliance test passed
    (audit_040_phase3_confirm, release-record).
    """

    def test_the_repository_declares_no_algorithm_id_twice(self, mapping_gate):
        n_ids, errors = mapping_gate.algorithm_id_errors()
        assert errors == []
        # Twelve NodeMeta declarations carry an ID today; a scan that finds
        # far fewer has lost its scope, not its duplicates.
        assert n_ids >= 12, n_ids

    def test_rigid_body_keeps_madd_node_007_and_lbm_moved_to_011(self):
        """The resolution of the duplicate, pinned where it was decided.

        ``RigidBodyNode``'s ID has been pinned by a test shipped in every
        release since 0.1.0; ``LBMNode``'s was first published in 0.4.0's
        algorithm guide, so ``LBMNode`` is the one renumbered.
        """
        from maddening.nodes.lbm import LBMNode
        from maddening.nodes.rigid_body import RigidBodyNode

        assert RigidBodyNode.meta.algorithm_id == "MADD-NODE-007"
        assert LBMNode.meta.algorithm_id == "MADD-NODE-011"

    def test_a_second_node_taking_an_existing_id_fails(self, mapping_gate, tmp_path):
        """Replays the audit's seeded duplicate: spring takes ball's ID."""
        root = _src_tree(
            tmp_path,
            ball='meta = NodeMeta(algorithm_id="MADD-NODE-001")\n',
            spring='meta = NodeMeta(\n    algorithm_id="MADD-NODE-001",\n)\n',
        )
        _n, errors = mapping_gate.algorithm_id_errors(root)
        (message,) = errors
        assert "MADD-NODE-001 is declared 2 times" in message
        assert "ball.py:1" in message and "spring.py:2" in message

    def test_the_gate_run_fails_on_a_duplicate_in_its_source_tree(
        self, mapping_gate, tmp_path, monkeypatch, capsys
    ):
        """Through ``main``, whatever guide directory it was pointed at."""
        root = _src_tree(
            tmp_path,
            a='meta = NodeMeta(algorithm_id="MADD-NODE-001")\n',
            b='meta = NodeMeta(algorithm_id="MADD-NODE-001")\n',
        )
        monkeypatch.setattr(mapping_gate, "SRC_PACKAGE", root)
        _guide(tmp_path, "| Diffusion | `maddening.nodes.heat.HeatNode.update` | |\n")
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "MADD-NODE-001 is declared 2 times" in capsys.readouterr().err

    @pytest.mark.parametrize("second", [
        # NodeMeta's first positional parameter is algorithm_id.
        'meta = NodeMeta("MADD-NODE-001", "1.0.0")\n',
        # Any algorithm_id= keyword, not only NodeMeta's.
        'meta = dataclasses.replace(Base.meta, algorithm_id="MADD-NODE-001")\n',
        # An attribute-spelled constructor.
        'meta = compliance.NodeMeta(algorithm_id="MADD-NODE-001")\n',
    ])
    def test_every_spelling_of_an_id_is_read(self, mapping_gate, tmp_path, second):
        root = _src_tree(
            tmp_path, a='meta = NodeMeta(algorithm_id="MADD-NODE-001")\n',
            b=second)
        _n, errors = mapping_gate.algorithm_id_errors(root)
        assert any("MADD-NODE-001 is declared 2 times" in e for e in errors), errors

    @pytest.mark.parametrize("source, reason", [
        ('meta = NodeMeta(algorithm_id=PREFIX + "001")\n', "not a string literal"),
        ('meta = NodeMeta(**META_KWARGS)\n', "can hide an algorithm_id"),
        ('meta = NodeMeta(ID_CONSTANT)\n', "not a string literal"),
    ])
    def test_an_id_the_scan_cannot_read_fails(self, mapping_gate, tmp_path,
                                              source, reason):
        """An ID nobody can read is an ID nobody checked for uniqueness."""
        root = _src_tree(tmp_path, a=source,
                         b='meta = NodeMeta(algorithm_id="MADD-NODE-002")\n')
        _n, errors = mapping_gate.algorithm_id_errors(root)
        assert any(reason in e for e in errors), errors

    def test_the_empty_default_claims_no_id(self, mapping_gate, tmp_path):
        root = _src_tree(tmp_path, a='meta = NodeMeta(algorithm_id="")\n',
                         b='meta = NodeMeta(algorithm_id="")\n',
                         c='meta = NodeMeta(algorithm_id="MADD-NODE-001")\n')
        assert mapping_gate.algorithm_id_errors(root) == (1, [])

    def test_a_scope_with_no_ids_fails(self, mapping_gate, tmp_path):
        root = _src_tree(tmp_path, a="x = 1\n")
        _n, errors = mapping_gate.algorithm_id_errors(root)
        assert errors and "verifies nothing" in errors[0]

    def test_a_guide_stating_another_nodes_id_fails(
        self, mapping_gate, tmp_path, capsys
    ):
        """Replays the audit's heat guide: 006 stated, 005 carried."""
        _node_guide(tmp_path, "HeatNode", "maddening.nodes.heat",
                    ["**Algorithm ID**: `MADD-NODE-006`\n"])
        assert mapping_gate.main([str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "states algorithm ID MADD-NODE-006" in err
        assert "'MADD-NODE-005'" in err

    def test_a_guide_stating_its_nodes_id_passes(
        self, mapping_gate, tmp_path, capsys
    ):
        _node_guide(tmp_path, "HeatNode", "maddening.nodes.heat",
                    ["**Algorithm ID**: `MADD-NODE-005`\n"])
        assert mapping_gate.main([str(tmp_path)]) == 0
        assert "1 guide algorithm ID(s) match" in capsys.readouterr().out

    @pytest.mark.parametrize("id_lines, reason", [
        ([], "states no '**Algorithm ID**"),
        (["**Algorithm ID**: `MADD-NODE-005`\n",
          "**Algorithm ID**: `MADD-NODE-005`\n"], "states 2 algorithm IDs"),
        (["**Algorithm ID**: MADD-NODE-005\n"], "is not written"),
        (["**Algorithm ID**:`MADD-NODE-005`\n"], "is not written"),
    ])
    def test_a_node_guide_whose_id_cannot_be_compared_fails(
        self, mapping_gate, tmp_path, capsys, id_lines, reason
    ):
        _node_guide(tmp_path, "HeatNode", "maddening.nodes.heat", id_lines)
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert reason in capsys.readouterr().err

    def test_a_node_id_on_a_guide_that_names_no_node_fails(
        self, mapping_gate, tmp_path, capsys
    ):
        _node_guide(tmp_path, "NoSuchNode", "maddening.nodes.heat",
                    ["**Algorithm ID**: `MADD-NODE-005`\n"])
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "does not resolve" in capsys.readouterr().err

    def test_a_node_without_its_own_nodemeta_cannot_own_a_guide_id(
        self, mapping_gate, tmp_path, capsys
    ):
        _node_guide(tmp_path, "SimulationNode", "maddening.core.node",
                    ["**Algorithm ID**: `MADD-NODE-005`\n"])
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "defines no NodeMeta of its own" in capsys.readouterr().err

    def test_a_non_node_id_needs_no_nodemeta(self, mapping_gate, tmp_path):
        """``MADD-ALG-INT-001`` documents the integrators, not a node."""
        _node_guide(tmp_path, "Explicit", "maddening.core",
                    ["**Algorithm ID**: `MADD-ALG-TEST-001`\n"])
        assert mapping_gate.main([str(tmp_path)]) == 0

    def test_two_guides_stating_one_id_fail(self, mapping_gate, tmp_path, capsys):
        for name in ("one.md", "two.md"):
            _node_guide(tmp_path, "Explicit", "maddening.core",
                        ["**Algorithm ID**: `MADD-ALG-TEST-001`\n"], name=name)
        assert mapping_gate.main([str(tmp_path)]) == 1
        assert "is stated by 2 guides" in capsys.readouterr().err


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


_CRANK = "@book{Crank1975,\n  title = {Diffusion},\n}\n"


class TestCitationGateReadsWhatPandocReads:
    """Three citations a reader's toolchain renders as broken passed with an
    undefined key (audit_040_p4_1, C8-C10, repro_gate_check_citations_gaps).
    Each case keeps one good citation beside the bad one, so the gate has
    something to verify either way."""

    @staticmethod
    def _gate(citations_gate, tmp_path, monkeypatch, bib_text, doc_text):
        bib, docs = _bib_and_doc(tmp_path, bib_text, doc_text)
        monkeypatch.setenv("BIB_PATH", str(bib))
        return citations_gate.main([str(docs)])

    @pytest.mark.parametrize("doc", [
        pytest.param("See [@Crank1975] and [@1975Crank].\n", id="C8-digit-led-key"),
        pytest.param("See [@Crank1975] and [@_Crank].\n", id="underscore-led-key"),
        pytest.param("See [@Crank1975].\n\nAs shown in [see\n@Nobody2031, p. 3].\n",
                     id="C9-bracket-wrapped-onto-the-next-line"),
        pytest.param("See [@Crank1975; @Nobody2031\n; @Other].\n",
                     id="wrapped-multi-citation"),
        pytest.param("See [@Crank1975] and [@{Nobody 2031}].\n", id="braced-key"),
        pytest.param("See [@Crank1975] and [-@Nobody2031].\n", id="suppress-author"),
    ])
    def test_an_undefined_key_fails_however_it_is_spelled(
        self, citations_gate, tmp_path, monkeypatch, capsys, doc
    ):
        assert self._gate(citations_gate, tmp_path, monkeypatch, _CRANK, doc) == 1
        assert "not found in" in capsys.readouterr().err

    def test_a_wrapped_citation_is_reported_on_the_line_its_key_is_on(
        self, citations_gate, tmp_path
    ):
        doc = tmp_path / "g.md"
        doc.write_text("Intro.\n\nAs shown in [see\n@Nobody2031, p. 3].\n")
        assert citations_gate.extract_citations(str(doc)) == [(4, "Nobody2031")]

    def test_a_bracket_does_not_reach_across_a_blank_line(
        self, citations_gate, tmp_path
    ):
        """A paragraph break ends any citation Pandoc would read."""
        doc = tmp_path / "g.md"
        doc.write_text("An unclosed [bracket\n\n@Nobody2031 is prose].\n")
        assert citations_gate.extract_citations(str(doc)) == []

    @pytest.mark.parametrize("doc", [
        "See [@Crank1975]. Mail [me@example.org] for help.\n",
        "See [@Crank1975], [the manual](mailto:me@example.org).\n",
    ])
    def test_an_address_is_not_a_citation(
        self, citations_gate, tmp_path, monkeypatch, doc
    ):
        assert self._gate(citations_gate, tmp_path, monkeypatch, _CRANK, doc) == 0

    def test_c10_an_entry_inside_a_comment_block_is_not_defined(
        self, citations_gate, tmp_path, monkeypatch, capsys
    ):
        bib = "@comment{\n" + _CRANK + "}\n@book{Other2000,\n}\n"
        doc = "See [@Other2000] and [@Crank1975].\n"
        assert self._gate(citations_gate, tmp_path, monkeypatch, bib, doc) == 1
        assert "[@Crank1975] not found" in capsys.readouterr().err

    def test_a_parenthesised_comment_block_hides_its_entry_too(
        self, citations_gate, tmp_path
    ):
        bib = tmp_path / "b.bib"
        bib.write_text("@Comment(\n" + _CRANK + ")\n@book{Other2000,\n}\n")
        assert citations_gate.parse_bib_entries(str(bib)) == [(6, "Other2000")]

    def test_an_unterminated_comment_block_fails(
        self, citations_gate, tmp_path, monkeypatch, capsys
    ):
        bib = _CRANK + "@comment{ forgot to close\n@book{Other2000,\n}\n"
        doc = "See [@Crank1975].\n"
        assert self._gate(citations_gate, tmp_path, monkeypatch, bib, doc) == 1
        assert "unterminated @comment on line 4" in capsys.readouterr().err

    def test_entries_after_a_closed_comment_block_still_count(
        self, citations_gate, tmp_path, monkeypatch
    ):
        bib = "@comment{ a note {with braces} }\n" + _CRANK
        doc = "See [@Crank1975].\n"
        assert self._gate(citations_gate, tmp_path, monkeypatch, bib, doc) == 0


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
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "{status}"
    affected_versions: "{versions}"
    affected_components:
      - "maddening.nodes.heat.HeatNode"
"""

_ANOMALY_WITHOUT_REFERENCES = """\
schema_version: "1.0"
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "open"
    affected_versions: ">=0.1.0"
"""


def _range_for(status):
    """The ``affected_versions`` a fixture entry of ``status`` must carry.

    The gate compares every range with the registry's ``maddening_version``
    (0.4.0.dev0 in these fixtures): a ``resolved`` entry's range must leave
    it out, every other status's range must admit it.
    """
    return ">=0.1.0, <0.4.0" if status == "resolved" else ">=0.1.0"


def _minimal_anomaly(status):
    return _MINIMAL_ANOMALY.format(status=status, versions=_range_for(status))


def _resolved_with_evidence(status):
    return _RESOLVED_WITH_EVIDENCE.format(
        status=status, versions=_range_for(status)
    )


_EMPTY_REGISTRY = """\
schema_version: "1.0"
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies: []
"""


class TestAnomalyGate:
    def test_an_unrecognised_resolution_status_exits_non_zero(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="probably fine tbh"))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1
        assert "resolution_status" in result.stderr

    def test_a_valid_registry_exits_zero(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="open"))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_the_repository_registry_passes_with_the_prefix_ci_uses(self):
        result = _run("check_anomalies", "--prefix", "MADD-ANO-")
        assert result.returncode == 0, result.stdout + result.stderr


_RESOLVED_WITH_EVIDENCE = """\
schema_version: "1.0"
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "{status}"
    resolution_version: "0.4.0"
    affected_versions: "{versions}"
    affected_components:
      - "maddening.nodes.heat.HeatNode"
    verification:
      - "tests/compliance/test_soup_evidence.py::test_madd_ano_001_is_recorded_resolved_by_that_floor"
"""

_TWO_ANOMALIES_WITH_A_GAP = """\
schema_version: "1.0"
maddening_version: "0.4.0.dev0"
generated_date: "2026-03-12"
anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "open"
    affected_versions: ">=0.1.0"
    affected_components:
      - "maddening.nodes.heat.HeatNode"
  - anomaly_id: "MADD-ANO-003"
    title: "Test"
    description: "Test"
    severity: "major"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: "Test"
    resolution_status: "open"
    affected_versions: ">=0.1.0"
    affected_components:
      - "maddening.nodes.heat.HeatNode"
"""


class TestAnomalyGateFailsClosedOnEvidence:
    """The two holes the release audit of 2026-09-22 walked through.

    Stripping a resolved entry's whole ``verification`` list passed the
    gate ("126 reference(s) verified"), and deleting an entry passed it
    too (17 anomalies, OK): the validator resolves what is there and
    counts what is declared, and neither notices an absence.
    """

    @pytest.mark.parametrize("status", ["resolved", "partially_resolved"])
    def test_a_closed_entry_without_verification_fails_naming_the_entry(
        self, tmp_path, status
    ):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status=status))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-001" in result.stderr
        assert "verification list is empty or missing" in result.stderr

    def test_the_rule_survives_no_resolve(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="resolved"))
        result = _run("check_anomalies", str(path), "--repo-root",
                      str(REPO_ROOT), "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "verification list is empty or missing" in result.stderr

    @pytest.mark.parametrize("status", ["resolved", "partially_resolved"])
    def test_a_closed_entry_that_cites_its_test_passes(self, tmp_path, status):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_resolved_with_evidence(status=status))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_an_open_entry_needs_no_verification(self, tmp_path):
        """The rule is about closed entries; ``open`` with only
        ``affected_components`` is the shape MADD-ANO-002 ships in."""
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="open"))
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 0, result.stdout + result.stderr

    def test_a_gap_in_the_id_sequence_fails_naming_the_missing_id(self, tmp_path):
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_TWO_ANOMALIES_WITH_A_GAP)
        result = _run("check_anomalies", str(path), "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-002" in result.stderr
        assert "contiguous" in result.stderr

    # -- a retired ID is a recorded gap, an unrecorded gap is a deletion --
    #
    # The gap rule's message tells the author to record a retirement in
    # _RETIRED_ANOMALY_IDS; until 0.4.0 the rule never read it, so a
    # genuinely retired ID could not pass.  Each case below builds a
    # repository root holding only the file the gate reads.

    @staticmethod
    def _root_retiring(tmp_path, assignment):
        root = tmp_path / "root"
        (root / "tests" / "compliance").mkdir(parents=True)
        (root / "tests" / "compliance" / "test_soup_evidence.py").write_text(
            f"_HIGHEST_ANOMALY_ID = 3\n{assignment}\n")
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_TWO_ANOMALIES_WITH_A_GAP)
        return root, path

    def test_a_gap_recorded_as_retired_passes(self, tmp_path):
        root, path = self._root_retiring(
            tmp_path, '_RETIRED_ANOMALY_IDS: frozenset = frozenset({"MADD-ANO-002"})')
        result = _run("check_anomalies", str(path), "--repo-root", str(root))
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("assignment", [
        "_RETIRED_ANOMALY_IDS: frozenset = frozenset()",
        '_RETIRED_ANOMALY_IDS: frozenset = frozenset({"MADD-ANO-004"})',
        '_RETIRED_ANOMALY_IDS: frozenset = frozenset({"MADD-VER-002"})',
        '_RETIRED_BENCHMARK_IDS: frozenset = frozenset({"MADD-ANO-002"})',
        "",
    ], ids=["empty", "another-id", "another-prefix", "the-benchmark-set", "no-assignment"])
    def test_a_gap_not_recorded_as_retired_still_fails(self, tmp_path, assignment):
        root, path = self._root_retiring(tmp_path, assignment)
        result = _run("check_anomalies", str(path), "--repo-root", str(root))
        assert result.returncode == 1, result.stdout
        assert "missing from the contiguous range" in result.stderr
        assert "MADD-ANO-002" in result.stderr

    def test_a_retired_set_the_gate_cannot_read_excuses_nothing(self, tmp_path):
        root, path = self._root_retiring(
            tmp_path, '_RETIRED_ANOMALY_IDS = frozenset(_load_ids("MADD-ANO-002"))')
        result = _run("check_anomalies", str(path), "--repo-root", str(root))
        assert result.returncode == 1, result.stdout
        assert "is not a literal set" in result.stderr
        assert "missing from the contiguous range" in result.stderr

    def test_a_retired_id_still_in_the_registry_fails(self, tmp_path):
        root, path = self._root_retiring(
            tmp_path,
            '_RETIRED_ANOMALY_IDS: frozenset = frozenset({"MADD-ANO-002", "MADD-ANO-003"})')
        result = _run("check_anomalies", str(path), "--repo-root", str(root))
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-003 is recorded as retired" in result.stderr

    @staticmethod
    def _mutated_registry(tmp_path, mutate):
        import yaml

        registry = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
        data = yaml.safe_load(registry.read_text())
        mutate(data)
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    def test_stripping_a_shipped_resolved_entrys_evidence_fails(self, tmp_path):
        """The audit's M4, made permanent: MADD-ANO-018 without its list."""
        def strip(data):
            entry = next(a for a in data["anomalies"] if a["anomaly_id"] == "MADD-ANO-018")
            assert entry["resolution_status"] == "resolved"
            del entry["verification"]

        path = self._mutated_registry(tmp_path, strip)
        result = _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                      "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-018" in result.stderr

    def test_deleting_a_shipped_entry_fails(self, tmp_path):
        """The audit's M5, one entry in from the end so the gate itself --
        not the out-of-file high-water mark -- is what catches it."""
        def delete(data):
            data["anomalies"] = [a for a in data["anomalies"] if a["anomaly_id"] != "MADD-ANO-010"]

        path = self._mutated_registry(tmp_path, delete)
        result = _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                      "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-010" in result.stderr


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

    def test_a_registry_whose_every_reference_is_unavailable_fails(
        self, tmp_path, monkeypatch
    ):
        """The case the old guard was written for, and could not reach.

        Every optional extra is installed in most environments, so no real
        symbol produces an ``unavailable`` note here.  The note is injected
        instead: one declared reference, one note, zero verified.
        """
        gate = _load("check_anomalies")
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="open"))

        def every_reference_unavailable(
            _path, *, prefix="", repo_root=None,
            resolve_references=True, notes=None,
        ):
            if notes is not None:
                notes.append(
                    "MADD-ANO-001: affected_components entry "
                    "'maddening.nodes.heat.HeatNode' was NOT checked -- "
                    "simulated missing optional extra"
                )
            return []

        monkeypatch.setattr(
            gate, "validate_anomaly_registry", every_reference_unavailable
        )
        assert gate.main([str(path), "--repo-root", str(REPO_ROOT)]) == 1

    def test_one_available_reference_is_enough_to_pass(
        self, tmp_path, monkeypatch
    ):
        """The other direction: the guard fires on zero, not on any."""
        gate = _load("check_anomalies")
        path = tmp_path / "known_anomalies.yaml"
        path.write_text(_minimal_anomaly(status="open").replace(
            '      - "maddening.nodes.heat.HeatNode"\n',
            '      - "maddening.nodes.heat.HeatNode"\n'
            '      - "maddening.core.graph_manager.GraphManager"\n',
        ))

        def one_unavailable(
            _path, *, prefix="", repo_root=None,
            resolve_references=True, notes=None,
        ):
            if notes is not None:
                notes.append("MADD-ANO-001: one entry was NOT checked")
            return []

        monkeypatch.setattr(gate, "validate_anomaly_registry", one_unavailable)
        assert gate.main([str(path), "--repo-root", str(REPO_ROOT)]) == 0

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


def _shipped_registry_with(tmp_path, mutate):
    """A copy of the shipped registry with one seeded fault, and its path."""
    import yaml

    registry = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
    data = yaml.safe_load(registry.read_text())
    mutate(data)
    path = tmp_path / "known_anomalies.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _set_range(aid, value):
    """A mutation that sets one entry's ``affected_versions`` (or drops it)."""
    def mutate(data):
        entry = next(a for a in data["anomalies"] if a["anomaly_id"] == aid)
        if value is None:
            entry.pop("affected_versions", None)
        else:
            entry["affected_versions"] = value
    return mutate


def _status_of(aid):
    import yaml

    registry = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
    data = yaml.safe_load(registry.read_text())
    return next(a for a in data["anomalies"]
                if a["anomaly_id"] == aid)["resolution_status"]


_PROBE_TESTS = '''\
import sys

import pytest


def test_runs():
    assert True


@pytest.mark.skip(reason="fixture: evidence that never runs")
def test_never_runs():
    assert True


# A condition that depends on the machine.  This fixture used to spell a
# conditional skip ``skipif(True)``, which skips everywhere and is now
# refused as unconditional (audit_040_p4_1, A20).
@pytest.mark.skipif(sys.platform == "no-such-platform", reason="fixture")
def test_sometimes_runs():
    assert True


@pytest.mark.xfail(strict=True, reason="fixture")
def test_expected_to_fail():
    assert False


def _helper_not_a_test():
    pass


class Helpers:
    def test_in_a_non_test_class(self):
        pass


class TestWithInit:
    def __init__(self):
        pass

    def test_never_collected(self):
        pass


@pytest.mark.skip(reason="fixture")
class TestSkippedClass:
    def test_inside(self):
        pass


class TestFine:
    def test_method(self):
        pass
'''


def _registry_citing(tmp_path, status, refs, components=None, extra_files=None):
    """A one-entry registry whose ``verification`` cites ``refs``, beside a
    throwaway ``tests/`` tree the refs resolve against."""
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test_probe.py").write_text(_PROBE_TESTS)
    for name, text in (extra_files or {}).items():
        (tests / name).write_text(text)
    closed = status == "resolved"
    components = components or ["maddening.nodes.heat.HeatNode"]
    path = tmp_path / "known_anomalies.yaml"
    path.write_text(
        'schema_version: "1.0"\nmaddening_version: "0.4.0.dev0"\n'
        'generated_date: "2026-03-12"\nanomalies:\n'
        '  - anomaly_id: "MADD-ANO-001"\n    title: "Test"\n'
        '    description: "Test"\n    severity: "major"\n'
        '    safety_relevance: "context_dependent"\n'
        '    safety_relevance_rationale: "Test"\n'
        f'    resolution_status: "{status}"\n'
        + ('    resolution_version: "0.4.0"\n' if closed else "")
        + f'    affected_versions: "{">=0.1.0, <0.4.0" if closed else ">=0.1.0"}"\n'
        '    affected_components:\n'
        + "".join(f'      - "{c}"\n' for c in components)
        + '    verification:\n'
        + "".join(f'      - "{r}"\n' for r in refs)
    )
    return path


class TestAnomalyGateChecksWhatAReferenceIs:
    """A ``verification`` entry must be a test that runs, listed once.

    The gate resolved each reference to a ``def`` and nothing more, so it
    accepted a reference listed twice (and counted it twice: 230 against
    228), a reference to a skip-marked test, and one to a helper pytest
    never collects (audit_040_phase3_confirm, release-record,
    repro_gate_verification_markers.py).
    """

    @staticmethod
    def _gate(path, repo_root, *extra):
        return _load("check_anomalies").main(
            [str(path), "--repo-root", str(repo_root), *extra])

    def test_tests_that_run_pass(self, tmp_path, capsys):
        path = _registry_citing(tmp_path, "resolved", [
            "tests/test_probe.py::test_runs",
            "tests/test_probe.py::TestFine::test_method",
            "tests/test_probe.py::TestFine",
            "tests/test_probe.py",
        ])
        assert self._gate(path, tmp_path) == 0, capsys.readouterr().err
        assert "0 verification test(s) skip conditionally" in capsys.readouterr().out

    @pytest.mark.parametrize("ref, reason", [
        ("tests/test_probe.py::test_never_runs", "skipped unconditionally"),
        ("tests/test_probe.py::TestSkippedClass::test_inside",
         "skipped unconditionally"),
        ("tests/test_probe.py::_helper_not_a_test", "is not a test"),
        ("tests/test_probe.py::Helpers::test_in_a_non_test_class",
         "is not collected by pytest"),
        ("tests/test_probe.py::TestWithInit::test_never_collected",
         "defines __init__"),
        ("tests/test_probe.py::test_expected_to_fail", "marked xfail"),
    ])
    def test_evidence_that_never_runs_or_passes_fails(
        self, tmp_path, capsys, ref, reason
    ):
        path = _registry_citing(tmp_path, "resolved", [
            "tests/test_probe.py::test_runs", ref])
        assert self._gate(path, tmp_path) == 1
        err = capsys.readouterr().err
        assert reason in err and ref in err, err

    def test_a_strict_xfail_pins_an_open_defect(self, tmp_path, capsys):
        """How MADD-ANO-011/012/013/021/022 cite their evidence."""
        path = _registry_citing(tmp_path, "open", [
            "tests/test_probe.py::test_expected_to_fail"])
        assert self._gate(path, tmp_path) == 0, capsys.readouterr().err

    def test_a_conditional_skip_is_reported_not_refused(self, tmp_path, capsys):
        path = _registry_citing(tmp_path, "resolved", [
            "tests/test_probe.py::test_runs",
            "tests/test_probe.py::test_sometimes_runs"])
        assert self._gate(path, tmp_path) == 0, capsys.readouterr().err
        out = capsys.readouterr().out
        assert "test_sometimes_runs': is skipped conditionally" in out
        assert "1 verification test(s) skip conditionally" in out

    @pytest.mark.parametrize("module, reason", [
        ('import pytest\npytestmark = pytest.mark.skip(reason="x")\n'
         "def test_a():\n    pass\n", "skipped unconditionally"),
        ('import pytest\npytestmark = [pytest.mark.slow, pytest.mark.skip]\n'
         "def test_a():\n    pass\n", "skipped unconditionally"),
        ('import pytest\npytest.skip("x", allow_module_level=True)\n'
         "def test_a():\n    pass\n", "skipped unconditionally"),
        ('import pytest\nzmq = pytest.importorskip("zmq")\n'
         "def test_a():\n    pass\n", "skipped conditionally"),
        ('import pytest\ntry:\n    import zmq\nexcept ImportError:\n'
         '    pytest.skip("x", allow_module_level=True)\n'
         "def test_a():\n    pass\n", "skipped conditionally"),
    ])
    def test_a_module_level_skip_reaches_every_test_in_it(
        self, tmp_path, capsys, module, reason
    ):
        for ref in ("tests/test_module_mark.py::test_a",
                    "tests/test_module_mark.py"):
            path = _registry_citing(tmp_path, "resolved",
                                    ["tests/test_probe.py::test_runs", ref],
                                    extra_files={"test_module_mark.py": module})
            rc = self._gate(path, tmp_path)
            captured = capsys.readouterr()
            assert reason in captured.err + captured.out, (ref, captured)
            assert rc == (1 if "unconditionally" in reason else 0), captured

    @pytest.mark.parametrize("name, text, reason", [
        ("helpers.py", "def test_a():\n    pass\n", "is not a file pytest collects"),
        ("test_empty.py", "def helper():\n    pass\n", "holds no test"),
    ])
    def test_a_file_that_runs_nothing_is_not_evidence(
        self, tmp_path, capsys, name, text, reason
    ):
        ref = f"tests/{name}" + ("::test_a" if name == "helpers.py" else "")
        path = _registry_citing(tmp_path, "resolved",
                                ["tests/test_probe.py::test_runs", ref],
                                extra_files={name: text})
        assert self._gate(path, tmp_path) == 1
        assert reason in capsys.readouterr().err

    @pytest.mark.parametrize("no_resolve", [False, True])
    def test_a_reference_listed_twice_fails(self, tmp_path, capsys, no_resolve):
        """Counted twice in the summary line; structural, so it holds under
        ``--no-resolve`` too."""
        ref = "tests/test_probe.py::test_runs"
        path = _registry_citing(tmp_path, "resolved", [ref, ref])
        extra = ["--no-resolve"] if no_resolve else []
        assert self._gate(path, tmp_path, *extra) == 1
        assert f"verification lists '{ref}' more than once" in (
            capsys.readouterr().err)

    def test_a_component_listed_twice_fails(self, tmp_path, capsys):
        comp = "maddening.nodes.heat.HeatNode"
        path = _registry_citing(tmp_path, "resolved",
                                ["tests/test_probe.py::test_runs"],
                                components=[comp, comp])
        assert self._gate(path, tmp_path) == 1
        assert f"affected_components lists '{comp}' more than once" in (
            capsys.readouterr().err)

    def test_the_audits_duplicates_fail_on_the_shipped_registry(self, tmp_path):
        def duplicate(data):
            entry = next(a for a in data["anomalies"]
                         if a["anomaly_id"] == "MADD-ANO-020")
            entry["verification"].append(entry["verification"][0])
            entry["affected_components"].append(entry["affected_components"][0])
        path = _shipped_registry_with(tmp_path, duplicate)
        result = _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                      "--repo-root", str(REPO_ROOT))
        assert result.returncode == 1, result.stdout
        assert "verification lists" in result.stderr
        assert "affected_components lists" in result.stderr

    def test_pytests_collection_rules_are_the_defaults_the_gate_assumes(self):
        """The gate hard-codes pytest's default ``test_*.py`` / ``Test*`` /
        ``test*``; configuring others would make it judge by the wrong ones."""
        import tomllib

        config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        options = config["tool"]["pytest"]["ini_options"]
        overridden = {"python_files", "python_classes", "python_functions"} & set(options)
        assert not overridden, (
            f"pyproject.toml sets {sorted(overridden)}; update "
            f"scripts/check_anomalies.py's _TEST_FILE / _TEST_*_PREFIX to match")


class TestAnomalyGateReadsEveryWayATestIsSkipped:
    """A skip or xfail need not be a decorator.

    The gate read ``@pytest.mark.skip`` / ``xfail`` by the decorator's text
    and ``pytestmark`` at module level, so a ``pytest.skip()`` as the cited
    test's first statement (A19), a ``pytest.xfail()`` there (A21), a mark
    bound to a name and applied as ``@_skip`` (A22) and ``skipif(True)``
    (A20) all passed -- with pytest confirming the cited test skipped or
    xfailed (audit_040_p4_1, repro_gate_check_anomalies_imperative_skip).
    Each module below holds ``test_a``, cited by a resolved entry.
    """

    @staticmethod
    def _gate_on(tmp_path, capsys, module, status="resolved"):
        path = _registry_citing(tmp_path, status,
                                ["tests/test_probe.py::test_runs",
                                 "tests/test_skips.py::test_a"],
                                extra_files={"test_skips.py": module})
        rc = _load("check_anomalies").main(
            [str(path), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    @pytest.mark.parametrize("module", [
        pytest.param('import pytest\n\ndef test_a():\n'
                     '    pytest.skip("seeded: never runs")\n    assert True\n',
                     id="A19-imperative-skip"),
        pytest.param('import pytest\n_skip = pytest.mark.skip(reason="x")\n\n'
                     '@_skip\ndef test_a():\n    pass\n',
                     id="A22-mark-bound-to-a-name"),
        pytest.param('import pytest\n\n@pytest.mark.skipif(True, reason="x")\n'
                     'def test_a():\n    pass\n', id="A20-skipif-True"),
        pytest.param('import pytest\n\n@pytest.mark.skipif("True", reason="x")\n'
                     'def test_a():\n    pass\n', id="skipif-string-True"),
        pytest.param('import pytest\n\n@pytest.mark.skipif(1 == 1, reason="x")\n'
                     'def test_a():\n    pass\n', id="skipif-constant-comparison"),
        pytest.param('import pytest\n\n@pytest.mark.skipif(reason="x")\n'
                     'def test_a():\n    pass\n', id="skipif-with-no-condition"),
        pytest.param('from pytest import mark as m\n\n@m.skip\n'
                     'def test_a():\n    pass\n', id="from-pytest-import-mark-as"),
        pytest.param('import pytest as pt\n\n@pt.mark.skip(reason="x")\n'
                     'def test_a():\n    pass\n', id="import-pytest-as"),
        pytest.param('import pytest\nskipif = pytest.mark.skipif\n\n'
                     '@skipif(True, reason="x")\ndef test_a():\n    pass\n',
                     id="aliased-skipif-called-with-True"),
        pytest.param('from pytest import skip\n\ndef test_a():\n'
                     '    skip("x")\n', id="from-pytest-import-skip"),
        pytest.param('import pytest\n\ndef test_a():\n'
                     '    raise pytest.skip.Exception("x")\n',
                     id="raise-skip-Exception"),
        pytest.param('import pytest\n\ndef test_a():\n    with open(__file__):\n'
                     '        pytest.skip("x")\n', id="skip-inside-with"),
        pytest.param('import pytest\n\ndef test_a():\n    if True:\n'
                     '        pytest.skip("x")\n', id="skip-under-if-True"),
        pytest.param('import pytest\n\ndef _require():\n    pytest.skip("x")\n\n'
                     'def test_a():\n    _require()\n', id="skip-in-a-helper"),
        pytest.param('import pytest\n\n@pytest.fixture\ndef dev():\n'
                     '    pytest.skip("x")\n\ndef test_a(dev):\n    pass\n',
                     id="skip-in-a-requested-fixture"),
        pytest.param('import pytest\n\n@pytest.fixture(autouse=True)\n'
                     'def _always():\n    pytest.skip("x")\n\n'
                     'def test_a():\n    pass\n', id="skip-in-an-autouse-fixture"),
        pytest.param('import pytest\n\n@pytest.fixture\ndef inner():\n'
                     '    pytest.skip("x")\n\n@pytest.fixture\ndef outer(inner):\n'
                     '    return 1\n\ndef test_a(outer):\n    pass\n',
                     id="skip-in-a-fixture-a-fixture-requests"),
    ])
    def test_a_skip_every_run_reaches_is_refused(self, tmp_path, capsys, module):
        rc, _out, err = self._gate_on(tmp_path, capsys, module)
        assert rc == 1, err
        assert "tests/test_skips.py::test_a': is skipped unconditionally" in err, err

    @pytest.mark.parametrize("module", [
        pytest.param('import pytest\n\ndef test_a():\n    pytest.xfail("seeded")\n',
                     id="A21-imperative-xfail"),
        pytest.param('import pytest\n_xf = pytest.mark.xfail(reason="x")\n\n'
                     '@_xf\ndef test_a():\n    pass\n', id="xfail-bound-to-a-name"),
        pytest.param('import sys\nimport pytest\n\ndef test_a():\n'
                     '    if sys.platform == "x":\n        pytest.xfail("y")\n',
                     id="guarded-imperative-xfail"),
    ])
    def test_an_xfail_is_refused_on_a_resolved_entry(self, tmp_path, capsys, module):
        rc, _out, err = self._gate_on(tmp_path, capsys, module)
        assert rc == 1, err
        assert "is marked xfail (or calls pytest.xfail())" in err, err

    def test_an_imperative_xfail_on_an_open_entry_is_accepted_like_the_mark(
        self, tmp_path, capsys
    ):
        rc, _out, err = self._gate_on(
            tmp_path, capsys,
            'import pytest\n\ndef test_a():\n    pytest.xfail("pinned")\n',
            status="open")
        assert rc == 0, err

    @pytest.mark.parametrize("module", [
        pytest.param('import sys\nimport pytest\n\ndef test_a():\n'
                     '    if sys.platform == "no-such-platform":\n'
                     '        pytest.skip("x")\n', id="guarded-skip"),
        pytest.param('import pytest\n\ndef test_a():\n'
                     '    zmq = pytest.importorskip("zmq")\n', id="importorskip"),
        pytest.param('import pytest\n\ndef test_a():\n    try:\n'
                     '        import zmq\n    except ImportError:\n'
                     '        pytest.skip("x")\n', id="skip-in-an-except"),
        pytest.param('import sys\nimport pytest\n\ndef test_a():\n'
                     '    sys.platform == "x" and pytest.skip("y")\n',
                     id="short-circuited-skip"),
        pytest.param('import sys\nimport pytest\n\n@pytest.fixture\ndef dev():\n'
                     '    if sys.platform == "x":\n        pytest.skip("y")\n\n'
                     'def test_a(dev):\n    pass\n', id="guarded-skip-in-a-fixture"),
        pytest.param('import sys\nimport pytest\n'
                     '_needs = pytest.mark.skipif(sys.platform == "x", reason="y")\n\n'
                     '@_needs\ndef test_a():\n    pass\n',
                     id="conditional-skipif-bound-to-a-name"),
    ])
    def test_a_conditional_skip_is_reported_and_counted(self, tmp_path, capsys, module):
        """What really runs in CI -- a device count, an import -- is reported."""
        rc, out, err = self._gate_on(tmp_path, capsys, module)
        assert rc == 0, err
        assert "tests/test_skips.py::test_a': is skipped conditionally" in out
        assert "1 verification test(s) skip conditionally" in out

    @pytest.mark.parametrize("module", [
        pytest.param('import pytest\n\n@pytest.mark.skipif(False, reason="x")\n'
                     'def test_a():\n    pass\n', id="skipif-False"),
        pytest.param('import pytest\n\ndef test_a():\n    return\n'
                     '    pytest.skip("unreachable")\n', id="skip-after-return"),
        pytest.param('import pytest\n\ndef test_a():\n    def later():\n'
                     '        pytest.skip("x")\n    assert later\n',
                     id="skip-in-an-uncalled-nested-def"),
        pytest.param('import pytest\n\ndef test_a():\n    if False:\n'
                     '        pytest.skip("x")\n', id="skip-under-if-False"),
        pytest.param('import pytest\n\n@pytest.fixture\ndef unused():\n'
                     '    pytest.skip("x")\n\ndef test_a():\n    pass\n',
                     id="skip-in-a-fixture-nobody-requests"),
    ])
    def test_a_skip_no_run_reaches_is_not_reported(self, tmp_path, capsys, module):
        rc, out, err = self._gate_on(tmp_path, capsys, module)
        assert rc == 0, err
        assert "0 verification test(s) skip conditionally" in out, out

    def test_the_shipped_registry_counts_its_imperative_conditional_skips(self):
        """The two fmpy ``importorskip`` calls and the device-count skips of
        ``test_halo_global_edge_fill`` (in a test body and in two fixtures)
        were invisible to the headline; they are conditional, and run in CI.

        Read statically, reference by reference: the whole-registry run is
        the compliance job's, and costs ~10 s of imports here.
        """
        import yaml

        gate = _load("check_anomalies")
        registry = yaml.safe_load(
            (REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml").read_text())
        cited = {(a["anomaly_id"], a["resolution_status"], ref)
                 for a in registry["anomalies"]
                 for ref in a.get("verification") or []}
        for ref in (
            "tests/fmi/test_non_finite_json_tokens.py::"
            "test_the_compiled_wrapper_reads_a_diverged_get_over_json",
            "tests/cloud/multigpu/test_halo_global_edge_fill.py::"
            "test_the_wrapper_fills_sharded_and_unsharded_halo_axes_alike",
            "tests/cloud/multigpu/test_halo_global_edge_fill.py::"
            "test_a_slab_exchange_fills_every_halo_slot_as_numpy_pad_does",
        ):
            entries = [(aid, status) for aid, status, r in cited if r == ref]
            assert entries, f"{ref} is no longer cited; update this test"
            for aid, status in entries:
                errors, conditional = gate._reference_findings(
                    aid, status, ref, str(REPO_ROOT))
                assert errors == [], errors
                assert len(conditional) == 1 and (
                    "is skipped conditionally" in conditional[0]), conditional


def test_a_broken_first_party_import_fails_the_anomaly_gate(
    tmp_path, capsys, monkeypatch
):
    """audit_040_p4_1, A25: a module named by ``affected_components`` gained
    ``from <own package>.x import renamed_away``; every reference behind it
    became "NOT checked" and the gate exited 0.  A throwaway package with
    the same import shape: intact it passes, broken it must exit 1.
    """
    import importlib

    root = tmp_path / "pkgroot"
    pkg = root / "p41_firstparty"
    (pkg / "core").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core" / "__init__.py").write_text("")
    (pkg / "core" / "solver_utils.py").write_text("def helper():\n    pass\n")
    node = pkg / "heart_pump.py"
    monkeypatch.syspath_prepend(str(root))
    registry_root = tmp_path / "repo"
    registry_root.mkdir()
    path = _registry_citing(registry_root, "open",
                            ["tests/test_probe.py::test_runs"],
                            components=["p41_firstparty.heart_pump.Pump"])
    gate = _load("check_anomalies")

    def run(source):
        node.write_text(source)
        for name in [m for m in sys.modules if m.split(".")[0] == "p41_firstparty"]:
            del sys.modules[name]
        importlib.invalidate_caches()
        rc = gate.main([str(path), "--repo-root", str(registry_root)])
        return rc, capsys.readouterr()

    intact = "from p41_firstparty.core.solver_utils import helper\n\nclass Pump:\n    pass\n"
    rc, captured = run(intact)
    assert rc == 0, captured.err
    broken = intact.replace(
        "\n\nclass", "\nfrom p41_firstparty.core.solver_utils import renamed_away\n\nclass")
    rc, captured = run(broken)
    assert rc == 1, captured.out
    assert "broken first-party import" in captured.err
    assert "NOT checked" not in captured.out
    for name in [m for m in sys.modules if m.split(".")[0] == "p41_firstparty"]:
        del sys.modules[name]


class TestAnomalyGateRefusesEvidenceCINeverRuns:
    """Every CI lane runs ``pytest tests/ --ignore=tests/viz``: a resolved
    entry citing a test there passed (audit_040_p4_1, A23; latent)."""

    def test_a_test_under_tests_viz_is_refused(self, tmp_path, capsys):
        (tmp_path / "tests" / "viz").mkdir(parents=True)
        (tmp_path / "tests" / "viz" / "test_network.py").write_text(
            "def test_a():\n    pass\n")
        ref = "tests/viz/test_network.py::test_a"
        path = _registry_citing(tmp_path, "resolved",
                                ["tests/test_probe.py::test_runs", ref])
        rc = _load("check_anomalies").main(
            [str(path), "--repo-root", str(tmp_path)])
        err = capsys.readouterr().err
        assert rc == 1, err
        assert f"'{ref}': 'tests/viz/test_network.py' is under tests/viz/" in err

    def test_a_directory_that_only_shares_the_prefix_is_not(self, tmp_path, capsys):
        (tmp_path / "tests" / "vizier").mkdir(parents=True)
        (tmp_path / "tests" / "vizier" / "test_x.py").write_text(
            "def test_a():\n    pass\n")
        path = _registry_citing(tmp_path, "resolved",
                                ["tests/vizier/test_x.py::test_a"])
        rc = _load("check_anomalies").main(
            [str(path), "--repo-root", str(tmp_path)])
        assert rc == 0, capsys.readouterr().err

    @staticmethod
    def _ci_test_invocations():
        """``(targets, ignores)`` for every command in the workflows that
        runs tests: a pytest (or pytest-wrapping) command with a positional
        argument under ``tests``."""
        import shlex

        import yaml

        invocations = []
        for wf in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
            data = yaml.safe_load(wf.read_text()) or {}
            for job in (data.get("jobs") or {}).values():
                for step in (job or {}).get("steps") or []:
                    run = (step or {}).get("run")
                    if not isinstance(run, str):
                        continue
                    for command in run.replace("\\\n", " ").splitlines():
                        command = command.split(" #", 1)[0].strip()
                        if "pytest" not in command and "audit_property" not in command:
                            continue
                        try:
                            tokens = shlex.split(command)
                        except ValueError:
                            continue
                        targets = [os.path.normpath(t) for t in tokens
                                   if os.path.normpath(t).split(os.sep)[0] == "tests"]
                        ignores = [os.path.normpath(t.split("=", 1)[1]) for t in tokens
                                   if t.startswith("--ignore=")]
                        ignores += [os.path.normpath(b) for a, b in zip(tokens, tokens[1:])
                                    if a == "--ignore"]
                        if targets:
                            invocations.append((targets, ignores))
        return invocations

    def test_the_paths_the_gate_refuses_are_the_paths_no_ci_lane_runs(self):
        """Derived from ``.github/workflows`` rather than trusted: a path
        is never run when some test command ignores it and no command runs
        it or anything under it."""
        invocations = self._ci_test_invocations()
        assert invocations, "no test command found in .github/workflows"

        def within(path, root):
            return path == root or path.startswith(root + os.sep)

        ignored = {i for _targets, ignores in invocations for i in ignores}
        never = {
            path for path in ignored
            if not any(
                (any(within(path, t) for t in targets)
                 and not any(within(path, i) for i in ignores))
                or any(within(t, path) for t in targets)
                for targets, ignores in invocations)
        }
        gate = _load("check_anomalies")
        assert {p.replace("/", os.sep) for p in gate.PATHS_CI_NEVER_RUNS} == never


class TestAnomalyGateHoldsEveryRangeToTheRegistrysVersion:
    """``affected_versions`` is compared with ``maddening_version`` (PEP 440).

    Before this, the SOUP generator's only rule was
    ``affected.startswith(">=")`` and this gate never read the field, so
    ``">=0.1.0, <0.4.0"`` on a ``partially_resolved`` entry -- the shape
    MADD-ANO-016 shipped in, asserting 0.4.0 is unaffected -- and
    ``"banana"`` on a ``resolved`` one both passed.  Each case seeds one
    fault into a copy of the shipped registry and runs the gate as CI does
    (``--no-resolve`` where resolution is beside the point; the rule is
    structural and must not depend on it).
    """

    @staticmethod
    def _gate(path, *extra):
        return _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                    "--repo-root", str(REPO_ROOT), *extra)

    @pytest.mark.parametrize("aid, closed", [
        ("MADD-ANO-002", ">=0.1.0, <0.4.0"),
        ("MADD-ANO-016", ">=0.1.0, <0.4.0"),
        ("MADD-ANO-005", ">=0.1.0, <0.4.0.dev0"),
    ])
    def test_a_reachable_entry_whose_range_leaves_this_version_out_fails(
        self, tmp_path, aid, closed
    ):
        assert _status_of(aid) in ("open", "partially_resolved")
        path = _shipped_registry_with(tmp_path, _set_range(aid, closed))
        result = self._gate(path, "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert aid in result.stderr and "does not admit" in result.stderr

    @pytest.mark.parametrize("aid", ["MADD-ANO-006", "MADD-ANO-007"])
    def test_a_resolved_entry_whose_range_admits_this_version_fails(
        self, tmp_path, aid
    ):
        """MADD-ANO-006 shipped ``resolved`` in 0.4.0 with ``>=0.1.0``."""
        assert _status_of(aid) == "resolved"
        path = _shipped_registry_with(tmp_path, _set_range(aid, ">=0.1.0"))
        result = self._gate(path, "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert aid in result.stderr and "admits 0.4.0.dev0" in result.stderr

    @pytest.mark.parametrize("bad", [
        "banana",
        "0.2.0, 0.2.1, 0.3.0, 0.3.1",   # MADD-ANO-004's old explicit list
        "",                              # parses as "every version"
        "<=0.3.1",                       # MADD-ANO-001's old spelling
        "<0.4.0",                        # no stated first version
        ">=0.4.0.dev0, <0.4.0",          # admits nothing at all
        "~=0.1",
        ">=0.1.0, >=0.2.0, <0.4.0",
        ">=0.1.0, <=0.3.1",              # right shape, wrong operator
        ">=0.1.0, !=0.2.0, <0.4.0",
    ])
    def test_a_range_outside_the_convention_fails_naming_the_entry(
        self, tmp_path, bad
    ):
        path = _shipped_registry_with(tmp_path, _set_range("MADD-ANO-007", bad))
        result = self._gate(path, "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-007" in result.stderr
        assert "affected_versions" in result.stderr

    def test_a_missing_range_fails(self, tmp_path):
        path = _shipped_registry_with(tmp_path, _set_range("MADD-ANO-011", None))
        result = self._gate(path, "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-011: affected_versions is None" in result.stderr

    def test_a_registry_without_its_version_fails(self, tmp_path):
        path = _shipped_registry_with(
            tmp_path, lambda data: data.pop("maddening_version"))
        result = self._gate(path, "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "no maddening_version" in result.stderr

    def test_the_shipped_defect_fails_the_full_gate_too(self, tmp_path):
        """MADD-ANO-016 as it shipped, through the run CI actually does."""
        path = _shipped_registry_with(
            tmp_path, _set_range("MADD-ANO-016", ">=0.1.0, <0.4.0"))
        result = self._gate(path)
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-016" in result.stderr


#: MADDENING's releases before 0.4.0, as CHANGELOG.md's dated headings list
#: them -- the history ``TestTheVersionRangeRule`` is written against, so a
#: test that describes a registry at another version can say what had been
#: released by then.
_RELEASED = ("0.1.0", "0.2.0", "0.2.1", "0.3.0", "0.3.1")


class TestTheVersionRangeRule:
    """``version_range_errors`` directly: the PEP 440 edges the convention
    is written around, which the gate runs above only exercise at one
    version."""

    @staticmethod
    def _registry(version, *entries):
        return {"maddening_version": version, "anomalies": [
            {"anomaly_id": f"MADD-ANO-{i:03d}", "resolution_status": status,
             "affected_versions": rng, **extra}
            for i, (status, rng, extra) in enumerate(entries, start=1)
        ]}

    @pytest.mark.parametrize("version", ["0.4.0.dev0", "0.4.0rc1", "0.4.0"])
    def test_a_cycle_introduced_defect_is_admitted_from_its_first_dev_build(
        self, anomalies_gate, version
    ):
        """``>=0.4.0.dev0`` admits every 0.4.0 build; ``>=0.4.0`` does not."""
        good = self._registry(version, ("open", ">=0.4.0.dev0", {}))
        assert anomalies_gate.version_range_errors(good) == []
        if version != "0.4.0":
            bad = self._registry(version, ("open", ">=0.4.0", {}))
            (message,) = anomalies_gate.version_range_errors(bad)
            assert "'>=0.4.0.dev0'" in message

    @pytest.mark.parametrize("version", ["0.4.0.dev0", "0.4.0", "0.4.1"])
    def test_a_range_closed_at_the_fix_leaves_out_its_dev_builds(
        self, anomalies_gate, version
    ):
        """PEP 440: ``<0.4.0`` excludes 0.4.0's own pre-releases.

        At 0.4.1 the fix is a past release, so the release list says so."""
        registry = self._registry(
            version, ("resolved", ">=0.1.0, <0.4.0",
                      {"resolution_version": "0.4.0"}))
        released = _RELEASED + (("0.4.0",) if version == "0.4.1" else ())
        assert anomalies_gate.version_range_errors(registry, released) == []

    def test_a_resolved_range_that_admits_its_resolution_version_fails(self, anomalies_gate):
        registry = self._registry(
            "0.4.0.dev0", ("resolved", ">=0.1.0, <0.5.0",
                           {"resolution_version": "0.4.0"}))
        errors = anomalies_gate.version_range_errors(registry)
        assert any("resolution_version is 0.4.0" in e for e in errors), errors

    def test_a_resolved_range_read_by_an_older_registry_is_reachable(self, anomalies_gate):
        """The same entry, on a registry at 0.3.1, says 0.3.1 is affected.

        (A 0.3.1 registry could not name 0.4.0 as a release either; that
        refusal is pinned separately, and this one is about the status.)"""
        registry = self._registry("0.3.1", ("resolved", ">=0.1.0, <0.4.0",
                                            {"resolution_version": "0.4.0"}))
        errors = anomalies_gate.version_range_errors(registry)
        assert any("admits 0.3.1" in e for e in errors), errors

    def test_none_is_the_empty_set(self, anomalies_gate):
        ok = self._registry("0.4.0.dev0", ("resolved", "none",
                                           {"resolution_version": "0.4.0"}))
        assert anomalies_gate.version_range_errors(ok) == []
        for status in ("open", "partially_resolved", "wont_fix", "fixed?"):
            bad = self._registry("0.4.0.dev0", (status, "none", {}))
            (message,) = anomalies_gate.version_range_errors(bad)
            assert "MADD-ANO-001" in message and "does not admit" in message

    def test_a_duplicate_is_parsed_but_not_compared(self, anomalies_gate):
        ok = self._registry("0.4.0.dev0", ("duplicate", ">=0.1.0, <0.2.0", {}))
        assert anomalies_gate.version_range_errors(ok) == []
        bad = self._registry("0.4.0.dev0", ("duplicate", "banana", {}))
        assert anomalies_gate.version_range_errors(bad)

    @pytest.mark.parametrize("rng, op", [
        (">=0.1.0, <=0.3.1", "'<='"),
        (">=0.1.0, !=0.2.0, <0.4.0", "'!='"),
        (">=0.1.0, <0.4.0, ==0.3.*", "'=='"),
    ])
    def test_an_operator_outside_the_convention_is_named(self, anomalies_gate, rng, op):
        """Each of these has exactly one ``>=`` and excludes this version, so
        the operator rule is the only one that can refuse it."""
        registry = self._registry("0.4.0.dev0", ("resolved", rng, {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert f"uses {op}" in message, message

    @pytest.mark.parametrize("rng", [
        ">=0.1.0, >=0.2.0, <0.4.0",
        ">=0.1.0, <0.3.0, <0.4.0",
    ])
    def test_a_range_names_one_first_version_and_at_most_one_fix(self, anomalies_gate, rng):
        """With two ``>=`` bounds, which one is FIRST depends on set order,
        so the empty-set check would catch it only some of the time; the
        shape rule has to be pinned on its own message."""
        registry = self._registry("0.4.0.dev0", ("resolved", rng, {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert "must name exactly one '>=FIRST'" in message, message

    def test_a_range_that_starts_after_this_version_fails(self, anomalies_gate):
        registry = self._registry("0.4.0.dev0", ("resolved", ">=0.5.0, <0.6.0", {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert "starts at 0.5.0" in message

    @pytest.mark.parametrize("registry", [
        {"maddening_version": "0.4.0.dev0", "anomalies": []},
        {"maddening_version": "0.4.0.dev0"},
        [],
    ])
    def test_nothing_to_check_is_a_failure(self, anomalies_gate, registry):
        """The generator runs this function too, and an empty registry used
        to pass its ``--check`` once regenerated."""
        assert anomalies_gate.version_range_errors(registry)

    @pytest.mark.parametrize("version", [None, "", "zero point four"])
    def test_an_unusable_registry_version_is_a_failure(self, anomalies_gate, version):
        registry = self._registry("0.4.0.dev0", ("open", ">=0.1.0", {}))
        registry["maddening_version"] = version
        assert anomalies_gate.version_range_errors(registry)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_range_is_refused_before_it_is_parsed(self, anomalies_gate, blank):
        """``SpecifierSet("")`` is the set of *every* version.  The one-``>=``
        rule would refuse it too, so this guard is redundant today; it is
        pinned so that loosening that rule cannot make a blank range mean
        "affects everything" without a word."""
        registry = self._registry("0.4.0.dev0", ("resolved", blank, {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert f"MADD-ANO-001: affected_versions is {blank!r}" in message

    def test_a_non_string_range_is_a_failure(self, anomalies_gate):
        registry = self._registry("0.4.0.dev0", ("open", 0.1, {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert "MADD-ANO-001: affected_versions is 0.1" in message

    def test_the_rule_does_not_lean_on_packagings_prerelease_default(
        self, anomalies_gate, monkeypatch
    ):
        """``SpecifierSet.contains`` changed its default across packaging
        releases: 22 (pytest's floor) leaves a pre-release out unless asked,
        26 lets it in.  On 22, a rule that relied on the default would read
        ``>=0.1.0`` as not admitting ``0.4.0.dev0`` and fail every open entry
        of a development registry.  The older default is simulated here, so
        the rule has to pass ``prereleases`` itself on whichever packaging
        this runs on."""
        from packaging.specifiers import SpecifierSet

        real = SpecifierSet.contains

        def packaging_22_default(self, item, prereleases=None, **kwargs):
            if prereleases is None:
                prereleases = bool(self.prereleases)
            return real(self, item, prereleases=prereleases, **kwargs)

        monkeypatch.setattr(SpecifierSet, "contains", packaging_22_default)
        assert not SpecifierSet(">=0.1.0").contains("0.4.0.dev0")  # simulated
        registry = self._registry(
            "0.4.0.dev0",
            ("open", ">=0.1.0", {}),
            ("resolved", ">=0.1.0, <0.4.0", {"resolution_version": "0.4.0"}),
        )
        assert anomalies_gate.version_range_errors(registry) == []

    def test_without_packaging_the_rule_fails_closed(self, anomalies_gate, monkeypatch):
        """``packaging`` comes with pytest; if it is ever missing, the rule
        must say so rather than pass everything."""
        monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
        registry = self._registry("0.4.0.dev0", ("open", ">=0.1.0", {}))
        (message,) = anomalies_gate.version_range_errors(registry)
        assert "packaging" in message

    def test_the_shipped_registry_passes(self, anomalies_gate):
        import yaml

        registry = yaml.safe_load(
            (REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml").read_text())
        assert anomalies_gate.version_range_errors(registry) == []

    # -- FIRST / FIX name real releases, and <FIX is the resolution_version --
    #
    # Each case below was accepted until audit_040_phase3_confirm
    # (release-record, repro_gate_version_range.py).  Setting MADD-ANO-020 to
    # ">=0.1.0, <0.2.0" while its fix is in 0.4.0 exited 0, and the SOUP
    # --check then called the regenerated row harmless drift.

    @pytest.mark.parametrize("status, rng, extra, expected", [
        ("resolved", ">=0.1.0, <0.2.0", {"resolution_version": "0.4.0"},
         "closes at 0.2.0, but resolution_version says the fix is in 0.4.0"),
        ("resolved", ">=0.1.0, <0.3.0", {"resolution_version": "0.4.0"},
         "closes at 0.3.0, but resolution_version says the fix is in 0.4.0"),
        ("open", ">=0.0.1", {}, "FIRST 0.0.1 is not a version MADDENING released"),
        ("resolved", ">=0.1.5, <0.4.0", {"resolution_version": "0.4.0"},
         "FIRST 0.1.5 is not a version MADDENING released"),
        ("resolved", ">=0.1.0, <0.3.7", {"resolution_version": "0.4.0"},
         "FIX 0.3.7 is not a version MADDENING released"),
        ("open", ">=0.3.1.post1", {},
         "FIRST 0.3.1.post1 is not a version MADDENING released"),
        # A cycle starts at .dev0, the convention's one spelling of it.
        ("open", ">=0.3.0.dev3", {},
         "FIRST 0.3.0.dev3 is not a version MADDENING released"),
        ("partially_resolved", ">=0.1.0", {"resolution_version": "0.3.7"},
         "resolution_version 0.3.7 is not a version MADDENING released"),
        ("resolved", "none", {"resolution_version": "0.3.7"},
         "resolution_version 0.3.7 is not a version MADDENING released"),
    ])
    def test_a_bound_that_names_no_release_or_disagrees_with_the_fix_fails(
        self, anomalies_gate, status, rng, extra, expected
    ):
        registry = self._registry("0.4.0.dev0", (status, rng, extra))
        errors = anomalies_gate.version_range_errors(registry, _RELEASED)
        assert any(expected in e for e in errors), errors
        assert all(e.startswith("MADD-ANO-001: ") for e in errors), errors

    @pytest.mark.parametrize("rng", [">=0.1.0, <0.4.0", "none"])
    def test_a_resolved_entry_must_say_which_release_fixed_it(
        self, anomalies_gate, rng
    ):
        """Without ``resolution_version`` the ``<FIX`` tie cannot be checked,
        so dropping the field would be a way round it."""
        registry = self._registry("0.4.0.dev0", ("resolved", rng, {}))
        (message,) = anomalies_gate.version_range_errors(registry, _RELEASED)
        assert "no resolution_version" in message

    @pytest.mark.parametrize("version, released, rng", [
        # A cycle-introduced defect keeps its .dev0 spelling after release.
        ("0.5.0.dev0", _RELEASED + ("0.4.0",), ">=0.4.0.dev0"),
        ("0.4.0.dev0", _RELEASED, ">=0.3.0.dev0"),
        ("0.4.0.dev0", _RELEASED, ">=0.4.0.dev0"),
        ("0.4.0", _RELEASED, ">=0.4.0"),
        ("0.4.0.dev0", _RELEASED, ">=0.2.1"),
    ])
    def test_a_release_or_a_cycles_first_dev_build_is_a_valid_first(
        self, anomalies_gate, version, released, rng
    ):
        registry = self._registry(version, ("open", rng, {}))
        assert anomalies_gate.version_range_errors(registry, released) == []

    def test_the_fix_may_be_a_past_release(self, anomalies_gate):
        registry = self._registry("0.4.0.dev0", (
            "resolved", ">=0.1.0, <0.3.0", {"resolution_version": "0.3.0"}))
        assert anomalies_gate.version_range_errors(registry, _RELEASED) == []

    @pytest.mark.parametrize("released", [(), ["zero point one"]])
    def test_an_unusable_release_list_fails_closed(self, anomalies_gate, released):
        registry = self._registry("0.4.0.dev0", ("open", ">=0.1.0", {}))
        assert anomalies_gate.version_range_errors(registry, released)

    def test_the_release_list_is_the_changelogs_dated_headings(
        self, anomalies_gate, tmp_path
    ):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n"
            "### [9.9.9] - 2030-01-01\n\n"          # not a section heading
            "## [0.2.0] - 2026-05-20\n\n- x\n\n"
            "## [0.1.5]\n\n"                         # no date: not a release
            "## [0.1.0] - 2025-03-01\n"
        )
        assert anomalies_gate.released_versions(str(changelog)) == (
            ("0.2.0", "0.1.0"), None)

    @pytest.mark.parametrize("text", [None, "# Changelog\n\n## [Unreleased]\n"])
    def test_a_changelog_without_releases_fails_closed(
        self, anomalies_gate, tmp_path, text
    ):
        changelog = tmp_path / "CHANGELOG.md"
        if text is not None:
            changelog.write_text(text)
        released, problem = anomalies_gate.released_versions(str(changelog))
        assert released == () and problem

    def test_the_changelog_still_lists_every_release_this_file_assumes(
        self, anomalies_gate
    ):
        """Deleting a release heading would make its version unusable as a
        bound; this pin lives outside the file it guards."""
        released, problem = anomalies_gate.released_versions()
        assert problem is None
        assert set(_RELEASED) <= set(released), released

    def test_the_changelogs_releases_are_the_release_tags(self, anomalies_gate):
        """The CHANGELOG is the source because CI's compliance job has no
        tags; wherever tags *are* present, the two must agree -- a heading
        with no tag names a release nobody made."""
        tags = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "tag", "-l", "v[0-9]*"],
            capture_output=True, text=True,
        ).stdout.split()
        if not tags:
            pytest.skip("this checkout has no v* tags (CI's compliance job "
                        "checks out without history); the CHANGELOG headings "
                        "are compared with the tags wherever they exist")
        released, problem = anomalies_gate.released_versions()
        assert problem is None
        assert sorted(released) == sorted(t[1:] for t in tags)

    def test_the_audits_shifted_fix_fails_the_gate_run(self, tmp_path):
        """MADD-ANO-020 closed at 0.2.0 while its fix is in 0.4.0."""
        path = _shipped_registry_with(
            tmp_path, _set_range("MADD-ANO-020", ">=0.1.0, <0.2.0"))
        result = _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                      "--repo-root", str(REPO_ROOT), "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-020" in result.stderr
        assert "closes at 0.2.0" in result.stderr

    def test_dropping_the_resolution_version_fails_the_gate_run(self, tmp_path):
        def drop(data):
            entry = next(a for a in data["anomalies"]
                         if a["anomaly_id"] == "MADD-ANO-020")
            entry.pop("resolution_version")
        path = _shipped_registry_with(tmp_path, drop)
        result = _run("check_anomalies", str(path), "--prefix", "MADD-ANO-",
                      "--repo-root", str(REPO_ROOT), "--no-resolve")
        assert result.returncode == 1, result.stdout
        assert "MADD-ANO-020" in result.stderr
        assert "no resolution_version" in result.stderr


class TestTransformGateConstantBinding:
    """A name bound to a string constant is still a string reference."""

    def test_a_transform_bound_to_a_module_constant_is_checked(
        self, transforms_gate, tmp_path, capsys
    ):
        (tmp_path / "indirect.py").write_text(
            self._VERIFIABLE
            + 'GHOST = "no_such_transform"\n'
            'gm.add_edge("a", "b", "x", "y", transform=GHOST)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

    def test_a_constant_naming_a_registered_transform_passes(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "indirect_ok.py").write_text(
            'LAST = "extract_last"\n'
            'gm.add_edge("a", "b", "x", "y", transform=LAST)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    #: A reference the gate verifies, so that a failure below is the
    #: unregistered name and not the empty-scope floor.  Without it, a scan
    #: that simply did not see the name also exits 1, and the test passes
    #: over the very hole it is for -- mutation-testing caught exactly that.
    _VERIFIABLE = 'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'

    def test_a_transform_bound_to_a_function_local_is_checked(
        self, transforms_gate, tmp_path, capsys
    ):
        """audit_040_phase3_wave_d, T6: a local name hid the reference."""
        (tmp_path / "local_name.py").write_text(
            self._VERIFIABLE
            + "def wire(gm):\n"
            '    name = "no_such_transform"\n'
            '    gm.add_edge("a", "b", "x", "y", transform=name)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

    def test_a_transform_bound_to_a_function_local_that_resolves_passes(
        self, transforms_gate, tmp_path
    ):
        (tmp_path / "local_name_ok.py").write_text(
            "def wire(gm):\n"
            '    name = "extract_last"\n'
            '    gm.add_edge("a", "b", "x", "y", transform=name)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    def test_every_name_a_loop_binds_is_checked(
        self, transforms_gate, tmp_path, capsys
    ):
        (tmp_path / "loop.py").write_text(
            self._VERIFIABLE
            + "def wire(gm):\n"
            '    for name in ("extract_last", "no_such_transform"):\n'
            '        gm.add_edge("a", "b", "x", "y", transform=name)\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

    def test_a_parameter_shadows_a_module_constant_of_the_same_name(
        self, transforms_gate, tmp_path
    ):
        """The parameter is what reaches the call, not the constant."""
        (tmp_path / "shadow.py").write_text(
            'NAME = "no_such_transform"\n'
            "def wire(gm, NAME):\n"
            '    gm.add_edge("a", "b", "x", "y", transform=NAME)\n'
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 0

    @pytest.mark.parametrize("splat", [
        '**{"transform": "no_such_transform"}',
        '**dict(transform="no_such_transform")',
    ])
    def test_a_transform_passed_through_a_literal_splat_is_checked(
        self, transforms_gate, tmp_path, capsys, splat
    ):
        """audit_040_phase3_wave_d, T7: ``**{...}`` hid the reference."""
        (tmp_path / "splat.py").write_text(
            self._VERIFIABLE
            + f'gm.add_edge("a", "b", "x", "y", {splat})\n'
        )
        assert transforms_gate.main([str(tmp_path)]) == 1
        assert "'no_such_transform'" in capsys.readouterr().out

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

    def test_a_scope_of_nothing_but_allowlisted_references_fails(
        self, transforms_gate, tmp_path, monkeypatch
    ):
        """Allowlisted is not verified; a scope of only those verified nothing."""
        probe = tmp_path / "only_allowlisted.py"
        probe.write_text(
            'gm.add_edge("a", "b", "x", "y", transform="deliberately_absent")\n'
        )
        monkeypatch.setitem(
            transforms_gate._ALLOWED_UNRESOLVABLE,
            (str(probe), "deliberately_absent"),
            "the fixture for this test",
        )
        assert transforms_gate.main([str(tmp_path)]) == 1

    def test_a_scan_root_that_does_not_exist_fails_naming_it(
        self, transforms_gate, tmp_path, capsys
    ):
        (tmp_path / "ok.py").write_text(
            'gm.add_edge("a", "b", "x", "y", transform="extract_last")\n'
        )
        missing = tmp_path / "no_such_dir"
        assert transforms_gate.main([str(tmp_path), str(missing)]) == 1
        assert str(missing) in capsys.readouterr().err


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
        gate = heat_stability_gate
        for relpath, (_reason, scope) in gate._ALLOWED_UNSTABLE.items():
            path = REPO_ROOT / relpath
            assert path.is_file(), (
                f"{relpath} is allowlisted but does not exist; remove the "
                f"_ALLOWED_UNSTABLE entry")
            unstable, unchecked, seen = [], [], []
            gate.scan_source(
                path.read_text(), relpath,
                gate._defaults(), unstable, unchecked, seen,
            )
            exempt = [(o, n) for o, n, _why in unstable if gate._is_allowed(o, n)]
            assert exempt, (
                f"{relpath} is allowlisted as deliberately containing an "
                f"unstable HeatNode construction, but no longer does; remove "
                f"the _ALLOWED_UNSTABLE entry"
            )
            if scope != gate.EMBEDDED_ONLY:
                # Every listed line still holds the rod it exempts, so a
                # line number left behind by an edit cannot exempt another.
                assert {n for _o, n in exempt} == set(scope), (relpath, exempt)

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
        # ...and for the right reason: the order-4 limit, not an empty scope.
        _rod(tmp_path, 'HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 4)',
             name="probe.py")
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_same_rod_at_order_two_still_passes(
        self, heat_stability_gate, tmp_path
    ):
        """The failure above is the order, not the widened parameter list."""
        _rod(tmp_path, 'HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 2)')
        assert heat_stability_gate.main([str(tmp_path)]) == 0

    @staticmethod
    def _with_a_recognised_rod(tmp_path, name, body):
        """Write the probe beside a plainly-spelled *stable* rod.

        Without it, a gate that cannot see the probe at all fails anyway --
        on the empty-scope guard -- and the test passes for the wrong
        reason.  A mutation removing the attribute match was missed exactly
        this way.
        """
        _rod(tmp_path, 'HeatNode("stable", 1e-5, n_cells=10, length=1.0,'
                       " thermal_diffusivity=0.01)", name="baseline_rod.py")
        (tmp_path / name).write_text(body)

    def _fails_naming_the_rod(self, gate, tmp_path, capsys):
        rc = gate.main([str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 1, out
        assert "Fourier number" in out, out
        return out

    def test_an_attribute_spelled_construction_is_seen(
        self, heat_stability_gate, tmp_path, capsys
    ):
        """``ast.Attribute`` carries ``.attr``, not ``.id``."""
        self._with_a_recognised_rod(
            tmp_path, "attr_form.py",
            "import maddening.nodes.heat as heat\n"
            'n = heat.HeatNode("e", timestep=1e-4, n_cells=257, length=1.0,\n'
            "                  thermal_diffusivity=0.1, stencil_order=4)\n",
        )
        out = self._fails_naming_the_rod(heat_stability_gate, tmp_path, capsys)
        assert "attr_form.py" in out

    def test_an_aliased_import_is_seen(
        self, heat_stability_gate, tmp_path, capsys
    ):
        self._with_a_recognised_rod(
            tmp_path, "alias_form.py",
            "from maddening.nodes.heat import HeatNode as Rod\n"
            'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
            "        thermal_diffusivity=0.1)\n",
        )
        out = self._fails_naming_the_rod(heat_stability_gate, tmp_path, capsys)
        assert "alias_form.py" in out

    def test_a_rebound_name_is_seen(
        self, heat_stability_gate, tmp_path, capsys
    ):
        """``Rod = HeatNode`` is the rebinding an import alias avoids."""
        self._with_a_recognised_rod(
            tmp_path, "rebound.py",
            "from maddening.nodes.heat import HeatNode\n"
            "Rod = HeatNode\n"
            'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
            "        thermal_diffusivity=0.1)\n",
        )
        out = self._fails_naming_the_rod(heat_stability_gate, tmp_path, capsys)
        assert "rebound.py" in out

    @pytest.mark.parametrize("body", [
        # The audit's case: a trivial subclass.
        "class Rod(HeatNode):\n    pass\n\n"
        'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
        "        thermal_diffusivity=0.1)\n",
        # A subclass of a subclass, defined in either order.
        "class Deep(Rod):\n    pass\n\nclass Rod(HeatNode):\n"
        "    def update(self, *a, **k):\n        return super().update(*a, **k)\n\n"
        'n = Deep("r", timestep=1e-4, n_cells=257, length=1.0,\n'
        "         thermal_diffusivity=0.1)\n",
        # A subclass of the attribute spelling.
        "import maddening.nodes.heat as heat\n\n"
        "class Rod(heat.HeatNode):\n    pass\n\n"
        'n = Rod("r", timestep=1e-4, n_cells=257, length=1.0,\n'
        "        thermal_diffusivity=0.1)\n",
    ])
    def test_a_rod_built_through_a_local_subclass_is_seen(
        self, heat_stability_gate, tmp_path, capsys, body
    ):
        """``HeatNode.__init__`` runs, guard and all, for a subclass that
        does not override it (audit_040_phase3_confirm, release-record)."""
        self._with_a_recognised_rod(
            tmp_path, "subclass_form.py",
            "from maddening.nodes.heat import HeatNode\n" + body)
        out = self._fails_naming_the_rod(heat_stability_gate, tmp_path, capsys)
        assert "subclass_form.py" in out

    def test_a_stable_rod_through_a_subclass_is_verified(self, heat_stability_gate):
        unstable, unchecked, seen = TestHeatStabilityCounts._scan(
            None, heat_stability_gate,
            "class Rod(HeatNode):\n    pass\n\n"
            'Rod("ok", timestep=1e-5, n_cells=10, length=1.0,'
            " thermal_diffusivity=0.01)\n",
        )
        assert len(seen) == 1 and unchecked == [] and unstable == []

    @pytest.mark.parametrize("override", ["__init__", "__new__"])
    def test_a_subclass_with_its_own_constructor_is_not_evaluated(
        self, heat_stability_gate, override
    ):
        """Its arguments need not reach ``HeatNode.__init__`` as written, so
        judging them as if they did could pass a rod the guard refuses."""
        unstable, unchecked, seen = TestHeatStabilityCounts._scan(
            None, heat_stability_gate,
            f"class Rod(HeatNode):\n    def {override}(self, *a, **k):\n"
            f"        pass\n\nclass Deeper(Rod):\n    pass\n\n"
            'Rod("r", timestep=1e-5, n_cells=10, length=1.0,'
            " thermal_diffusivity=0.01)\n"
            'Deeper("r", timestep=1e-5, n_cells=10, length=1.0,'
            " thermal_diffusivity=0.01)\n",
        )
        assert seen == [] and unstable == []
        assert len(unchecked) == 2
        assert all("defines its own __init__ or __new__" in u[2] for u in unchecked)

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


class TestHeatStabilityGridPoints:
    """Any ``grid_points=`` used to skip the rod, even ``grid_points=None``
    -- the default, a uniform rod whose constructor refuses an unstable
    timestep (audit_040_p4_1, H6; latent)."""

    def _scan(self, gate, source):
        unstable, unchecked, seen = [], [], []
        gate.scan_source(source, "probe.py", gate._defaults(),
                         unstable, unchecked, seen)
        return unstable, unchecked, seen

    def test_an_unstable_rod_spelling_out_grid_points_none_fails(
        self, heat_stability_gate, tmp_path
    ):
        """The auditor's H6, as a scan root: one stable rod, one refused."""
        _rod(tmp_path,
             'HeatNode("ok", timestep=0.01, n_cells=10, length=1.0, '
             'thermal_diffusivity=0.01)\n'
             'HeatNode("r", timestep=50.0, n_cells=10, length=1.0, '
             'thermal_diffusivity=0.01, grid_points=None)')
        assert heat_stability_gate.main([str(tmp_path)]) == 1

    def test_the_constructor_refuses_the_same_rod(self):
        """So the gate is agreeing with the code, not inventing a rule."""
        from maddening.nodes.heat import HeatNode

        # Through a mapping held in a variable, which the gate reports as
        # not evaluated: this file may plant unstable rods only inside
        # string literals, and a literal call here would (rightly) fail it.
        rod = dict(timestep=50.0, n_cells=10, length=1.0,
                   thermal_diffusivity=0.01, grid_points=None)
        with pytest.raises(ValueError, match="unstable"):
            HeatNode("r", **rod)

    def test_grid_points_none_positionally_is_judged_too(self, heat_stability_gate):
        unstable, _unchecked, _seen = self._scan(
            heat_stability_gate,
            'HeatNode("r", 50.0, 10, 1.0, 0.01, 0.0, 2, None)\n')
        assert len(unstable) == 1

    def test_a_literal_grid_is_outside_the_guard(self, heat_stability_gate):
        """The constructor checks uniform rods only; so does the gate."""
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("g", timestep=50.0, n_cells=4, '
            'grid_points=[0.0, 0.1, 0.3, 0.6, 1.0])\n')
        assert (unstable, unchecked, seen) == ([], [], [])

    def test_a_computed_grid_is_not_evaluated_rather_than_skipped(
        self, heat_stability_gate
    ):
        """It may be ``None`` at run time, i.e. a guarded uniform rod."""
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("g", timestep=50.0, n_cells=4, grid_points=xs)\n')
        assert unstable == [] and seen == []
        assert len(unchecked) == 1 and "grid_points is computed" in unchecked[0][2]


class TestHeatStabilitySplats:
    """A ``**`` or ``*`` splat must not make a rod read as verified.

    The keyword map was built with ``if kw.arg``, which drops a ``**``
    splat entirely: ``HeatNode("h", timestep=1e-3,
    **{"thermal_diffusivity": 1e3})`` was judged on the *default*
    diffusivity and counted as verified (139 -> 140), while
    ``HeatNode.__init__`` refuses it at Fourier number 100
    (audit_040_phase3_wave_d, H5).
    """

    _scan = TestHeatStabilityCounts._scan

    @pytest.mark.parametrize("splat", [
        '**{"thermal_diffusivity": 1e3}',
        "**dict(thermal_diffusivity=1e3)",
    ])
    def test_an_unstable_rod_spelled_through_a_literal_splat_fails(
        self, heat_stability_gate, tmp_path, capsys, splat
    ):
        root = _rod(tmp_path, f'HeatNode("h", timestep=1e-3, {splat})')
        assert heat_stability_gate.main([str(root)]) == 1
        assert "Fourier number 100" in capsys.readouterr().out

    def test_a_stable_rod_spelled_through_a_literal_splat_is_verified(
        self, heat_stability_gate
    ):
        unstable, unchecked, seen = self._scan(
            heat_stability_gate,
            'HeatNode("ok", timestep=1e-5, **{"thermal_diffusivity": 0.01})\n',
        )
        assert len(seen) == 1 and unchecked == [] and unstable == []

    @pytest.mark.parametrize("call, reason", [
        ('HeatNode("h", timestep=1e-5, **overrides)', "**mapping"),
        ('HeatNode("h", timestep=1e-5, **{**base, "n_cells": 10})', "**mapping"),
        ("HeatNode(*positional)", "*args"),
        ('HeatNode("h", timestep=1e-5, **{"timestep": 1e-3})', "given twice"),
    ])
    def test_a_splat_the_gate_cannot_read_is_not_counted_as_verified(
        self, heat_stability_gate, call, reason
    ):
        unstable, unchecked, seen = self._scan(heat_stability_gate, call + "\n")
        assert seen == [] and unstable == []
        assert len(unchecked) == 1 and reason in unchecked[0][2], unchecked

    def test_a_scope_of_only_unreadable_splats_fails(
        self, heat_stability_gate, tmp_path, capsys
    ):
        root = _rod(tmp_path, 'HeatNode("h", timestep=1e-5, **overrides)')
        assert heat_stability_gate.main([str(root)]) == 1
        err = capsys.readouterr().err
        assert "not one of them could be evaluated" in err
        assert "**mapping" in err

    def test_the_unevaluated_are_reported_by_reason_and_listed_on_request(
        self, heat_stability_gate, tmp_path, capsys
    ):
        root = _rod(
            tmp_path,
            'HeatNode("ok", timestep=1e-5, thermal_diffusivity=0.01)\n'
            'HeatNode("h", timestep=1e-5, **overrides)\n',
        )
        assert heat_stability_gate.main([str(root)]) == 0
        out = capsys.readouterr().out
        assert "not evaluated, by reason:" in out
        assert "a **mapping the gate cannot read" in out
        assert "mod.py:3:" not in out
        assert "OK: 1 HeatNode construction(s) verified" in out
        assert heat_stability_gate.main(["--list-unevaluated", str(root)]) == 0
        assert "mod.py:3:" in capsys.readouterr().out


class TestHeatStabilityAllowlist:
    def test_every_entry_carries_a_reason(self, heat_stability_gate):
        for path, (reason, _scope) in heat_stability_gate._ALLOWED_UNSTABLE.items():
            assert isinstance(reason, str) and reason.strip(), path

    def test_every_entry_names_constructions_not_a_whole_file(
        self, heat_stability_gate
    ):
        """The test file was exempt wholesale, so a real unstable rod
        appended to it passed (audit_040_phase3_confirm, release-record)."""
        gate = heat_stability_gate
        for path, (_reason, scope) in gate._ALLOWED_UNSTABLE.items():
            assert scope == gate.EMBEDDED_ONLY or (
                isinstance(scope, frozenset) and scope
                and all(isinstance(n, int) for n in scope)), (path, scope)

    def test_a_real_construction_in_the_test_file_is_not_exempt(
        self, heat_stability_gate
    ):
        gate = heat_stability_gate
        test_file = "tests/compliance/test_gate_scripts.py"
        assert gate._is_allowed(f"{test_file}{gate._EMBEDDED}", 3)
        assert not gate._is_allowed(test_file, 3)
        probe = next(p for p, (_r, sc) in gate._ALLOWED_UNSTABLE.items()
                     if sc != gate.EMBEDDED_ONLY)
        line = min(gate._ALLOWED_UNSTABLE[probe][1])
        assert gate._is_allowed(probe, line)
        assert not gate._is_allowed(probe, line + 100)
        assert not gate._is_allowed(f"{probe}{gate._EMBEDDED}", line)

    def test_only_the_planted_fixture_in_an_allowlisted_file_is_exempt(
        self, heat_stability_gate, tmp_path, monkeypatch, capsys
    ):
        """Replays the audit: a real rod appended to the exempt file."""
        gate = heat_stability_gate
        planted = tmp_path / "planted.py"
        planted.write_text(
            "from maddening.nodes.heat import HeatNode\n"
            "OK = HeatNode('ok', timestep=1e-5, n_cells=10, length=1.0,\n"
            "              thermal_diffusivity=0.01)\n\n"
            "FIXTURE = 'HeatNode(\"h\", 0.51, n_cells=10, length=1.0, "
            "thermal_diffusivity=1.0)'\n"
        )
        monkeypatch.setitem(gate._ALLOWED_UNSTABLE, str(planted),
                            ("the fixture for this test", gate.EMBEDDED_ONLY))
        assert gate.main([str(tmp_path)]) == 0, capsys.readouterr().out
        assert "1 deliberately unstable construction(s) exempt" in (
            capsys.readouterr().out)
        with planted.open("a") as fh:
            fh.write("\n\ndef _seeded_real_use():\n"
                     "    return HeatNode('h', 0.51, n_cells=10, length=1.0,\n"
                     "                    thermal_diffusivity=1.0)\n")
        assert gate.main([str(tmp_path)]) == 1
        assert "planted.py:9: Fourier number" in capsys.readouterr().out

    def test_the_allowlist_stays_small(self, heat_stability_gate):
        allowlist = heat_stability_gate._ALLOWED_UNSTABLE
        cap = heat_stability_gate._MAX_ALLOWED_UNSTABLE
        assert len(allowlist) <= cap, (
            f"{len(allowlist)} allowlisted files (cap {cap}).  Each one is a "
            f"file whose planted rods this gate stops refusing; fix the rod "
            f"instead of raising the cap."
        )


# ---------------------------------------------------------------------------
# check_doctests.py
# ---------------------------------------------------------------------------

#: Runs ``scripts/check_doctests.py`` against a throwaway package instead of
#: ``src/maddening``, in a fresh interpreter: the gate calls ``pytest.main``
#: and ``os.chdir``, neither of which belongs inside this test process.
_DOCTEST_GATE_ON = r"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", sys.argv[1])
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.REPO_ROOT = Path(sys.argv[2])
gate.PACKAGE = Path("src/probe_pkg")
gate.EXCLUDED = (gate.PACKAGE / "examples",)
# The per-file pins name the real package's files; a probe package pins its
# own.  PROBE_PINS (JSON) when a test sets it, else the probe's own counts,
# so a test about something else is not failed by a pin it never meant.
import json, os
pins = os.environ.get("PROBE_PINS")
gate.EXAMPLES_PER_FILE = (
    json.loads(pins) if pins is not None
    else {path: n for path, (_d, n) in gate._static_counts().items()})
sys.exit(gate.main(sys.argv[3:]))
"""

_TWO_EXAMPLES = '''\
def two():
    """Two examples in one docstring.

    >>> 1 + 1
    2
    >>> 2 + 2
    4
    """
'''


def _doctest_gate(tmp_path, module_source, *args, pins=None, extra=None):
    """Run the doctest gate over a probe package holding ``mod.py``.

    ``extra`` maps further module names to their source; ``pins`` replaces
    the gate's ``EXAMPLES_PER_FILE`` (default: the probe's own counts).
    """
    pkg = tmp_path / "src" / "probe_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(module_source)
    for name, text in (extra or {}).items():
        (pkg / name).write_text(text)
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PROBE_PINS", None)
    if pins is not None:
        env["PROBE_PINS"] = json.dumps(pins)
    return subprocess.run(
        [sys.executable, "-c", _DOCTEST_GATE_ON,
         str(SCRIPTS / "check_doctests.py"), str(tmp_path), *args],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
        timeout=600,
    )


class TestDoctestGate:
    """``scripts/check_doctests.py`` counts examples, not docstrings.

    A pytest doctest item is a whole docstring, so ``# doctest: +SKIP`` on
    a wrong example inside a two-example docstring left the item passing
    and the gate at "OK: 31 docstring examples ran and passed, floor 31"
    (audit_040_phase3_wave_d, D2).
    """

    def test_both_examples_of_a_docstring_are_counted(self, tmp_path):
        """A floor of 2 over one docstring: satisfiable only by examples."""
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "2")
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("OK: 2 docstring example(s) in 1 docstring(s) ran and "
                "passed, 0 skipped") in result.stdout

    def test_a_skipped_wrong_example_inside_a_passing_docstring_fails(
        self, tmp_path
    ):
        # SEEDED FAULT (fixture): the expected output is wrong and +SKIP
        # hides it -- the defect the gate must catch.
        source = _TWO_EXAMPLES.replace(
            "    >>> 2 + 2\n    4\n",
            "    >>> 2 + 2  # doctest: +SKIP\n    5\n",
        )
        result = _doctest_gate(tmp_path, source, "--min", "1")
        assert result.returncode == 1, result.stdout + result.stderr
        assert "1 example(s) inside passing doctests did not run" in (
            result.stdout)
        assert "probe_pkg.mod.two: 1 skipped (line(s) 6)" in result.stdout

    def test_fewer_examples_than_the_floor_fails(self, tmp_path):
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "3")
        assert result.returncode == 1, result.stdout + result.stderr
        assert "2 example(s) executed in passing doctests, floor is 3" in (
            result.stdout)

    def test_a_package_with_no_examples_fails_whatever_the_floor(
        self, tmp_path
    ):
        result = _doctest_gate(tmp_path, "def f():\n    return 1\n",
                               "--min", "0")
        assert result.returncode == 1, result.stdout + result.stderr
        assert "no docstring example executed" in result.stdout

    def test_deselecting_every_doctest_fails(self, tmp_path):
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "0",
                               "--", "-k", "no_such_doctest")
        assert result.returncode == 1, result.stdout + result.stderr
        assert "no docstring example executed" in result.stdout

    def test_a_runner_without_the_count_stops_the_gate(self):
        """The instrument is not optional: no ``tries``, no verdict."""
        from types import SimpleNamespace

        gate = _load("check_doctests")
        recorder = gate._Recorder()
        item = SimpleNamespace(nodeid="m.py::m.f",
                               dtest=SimpleNamespace(examples=[], lineno=0),
                               runner=SimpleNamespace())
        recorder.pytest_runtest_setup(item)
        assert recorder.instrument_error and "tries" in recorder.instrument_error

    def test_the_recorder_credits_only_the_examples_that_ran(self):
        """Two examples, the runner executed one: one run, one skipped."""
        import doctest
        from types import SimpleNamespace

        gate = _load("check_doctests")
        recorder = gate._Recorder()
        examples = [doctest.Example("1\n", "1\n", lineno=2),
                    doctest.Example("2\n", "2\n", lineno=4,
                                    options={doctest.SKIP: True})]
        runner = SimpleNamespace(tries=10)
        item = SimpleNamespace(nodeid="m.py::m.f", runner=runner,
                               dtest=SimpleNamespace(examples=examples,
                                                     lineno=10))
        recorder.pytest_runtest_setup(item)
        runner.tries += 1
        recorder.pytest_runtest_makereport(item, SimpleNamespace(when="call"))
        recorder.pytest_runtest_logreport(SimpleNamespace(
            when="call", passed=True, nodeid="m.py::m.f"))
        assert recorder.passed == {"m.py": 1}
        assert recorder.examples_run == {"m.py": 1}
        assert recorder.skipped_examples == [("m.py::m.f", [15], 1)]

    #: The committed floor, deliberately in a second file.  Lowering
    #: ``MIN_EXAMPLES`` alone -- the one-number edit that makes a shrinking
    #: example set pass -- now fails here; the same two-file ratchet
    #: ``min_mappings_floor.json`` gives the mapping pins
    #: (audit_040_r2/gates, finding G6).  Raise both when examples are
    #: added; lowering both belongs in a commit that says why.
    COMMITTED_EXAMPLE_FLOOR = 111

    #: The committed per-file pins, the second half of ``EXAMPLES_PER_FILE``'s
    #: ratchet.  Deleting the only example in ``core/compliance/metadata.py``
    #: dropped the whole file out of the gate with "OK" (audit_040_p4_1, D6).
    COMMITTED_EXAMPLES_PER_FILE = {
        "src/maddening/api/auth.py": 7,
        "src/maddening/api/server.py": 3,
        "src/maddening/core/compliance/metadata.py": 1,
        "src/maddening/core/coupling/acceleration.py": 37,
        "src/maddening/core/simulation/calibration.py": 8,
        "src/maddening/core/solver_utils.py": 5,
        "src/maddening/nodes/adaptive/wavelet.py": 6,
        "src/maddening/nodes/adaptive/wavelets/dirichlet.py": 1,
        "src/maddening/nodes/adaptive/wavelets/transform.py": 5,
        "src/maddening/serialization/json_codec.py": 3,
        "src/maddening/surrogates/types.py": 4,
        "src/maddening/testing/mms.py": 14,
        "src/maddening/transport_auth.py": 7,
        "src/maddening/viz/backends/matplotlib_renderer.py": 6,
        "src/maddening/viz/backends/terminal_renderer.py": 2,
        "src/maddening/viz/usd_viewer.py": 2,
    }

    def test_the_floor_equals_its_committed_value(self):
        """Equal, not ``>=``: a committed floor with slack under it lets
        ``MIN_EXAMPLES`` drop to it together with the examples."""
        gate = _load("check_doctests")
        assert gate.MIN_EXAMPLES == self.COMMITTED_EXAMPLE_FLOOR, (
            f"MIN_EXAMPLES is {gate.MIN_EXAMPLES}, the committed floor is "
            f"{self.COMMITTED_EXAMPLE_FLOOR}; move both together, and an "
            f"example set that shrank needs a reason, not a lower number"
        )
        assert self.COMMITTED_EXAMPLE_FLOOR == sum(
            self.COMMITTED_EXAMPLES_PER_FILE.values())

    def test_the_floor_is_the_number_of_examples_in_the_source(self):
        """A floor below the count guards only part of the collection.

        It sat at 110 over 111 examples, so one deletion passed
        (audit_040_p4_1, D5); above the count it could only ever fail.
        Static, so it holds without running the examples; the gate's own
        run in the compliance job is the dynamic half.
        """
        gate = _load("check_doctests")
        static = sum(n for _d, n in gate._static_counts().values())
        assert static == gate.MIN_EXAMPLES, (
            f"{static} examples in the source, MIN_EXAMPLES is "
            f"{gate.MIN_EXAMPLES}: set it (and COMMITTED_EXAMPLE_FLOOR) to "
            f"the count")

    def test_every_file_is_pinned_at_its_count(self):
        """The per-file pins are the source's per-file counts, exactly."""
        gate = _load("check_doctests")
        static = {path: n for path, (_d, n) in gate._static_counts().items()}
        assert gate.EXAMPLES_PER_FILE == static, {
            path: (gate.EXAMPLES_PER_FILE.get(path), static.get(path))
            for path in set(static) | set(gate.EXAMPLES_PER_FILE)
            if gate.EXAMPLES_PER_FILE.get(path) != static.get(path)
        }

    def test_every_pin_equals_its_committed_value(self):
        gate = _load("check_doctests")
        assert gate.EXAMPLES_PER_FILE == self.COMMITTED_EXAMPLES_PER_FILE

    # -- replays of audit_040_p4_1 D5 / D6 ---------------------------------

    @staticmethod
    def _counts_with(gate, rel, mutate):
        """The tree's per-file example counts with ``rel`` mutated."""
        import doctest
        import tempfile

        counts = {path: n for path, (_d, n) in gate._static_counts().items()}
        text = mutate((REPO_ROOT / rel).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "mutated.py"
            probe.write_text(text, encoding="utf-8")
            n = sum(len(doctest.DocTestParser().get_examples(d))
                    for d in gate._docstrings(probe))
        if n:
            counts[rel] = n
        else:
            counts.pop(rel, None)
        return counts

    def test_the_real_pins_refuse_one_example_deleted_from_acceleration(self):
        """D5: one example out of ``acceleration.py``, 111 -> 110."""
        gate = _load("check_doctests")
        rel = "src/maddening/core/coupling/acceleration.py"
        example = ('    >>> float(residual_precision_floor(s, ["n"], "l2", '
                   'atol=2.0))   # dead-banded: nothing read\n    0.0\n')

        def mutate(text):
            assert example in text, "the replayed example has moved"
            return text.replace(example, "", 1)

        counts = self._counts_with(gate, rel, mutate)
        assert sum(counts.values()) >= gate.MIN_EXAMPLES - 1
        errors, _slack = gate.per_file_errors(counts, gate.EXAMPLES_PER_FILE)
        assert any(e.startswith(f"{rel}: 36 example(s)") for e in errors), errors

    def test_the_real_pins_refuse_a_file_losing_its_only_example(self):
        """D6: ``metadata.py``'s one example deleted; the file drops out."""
        gate = _load("check_doctests")
        rel = "src/maddening/core/compliance/metadata.py"

        def mutate(text):
            start = text.index("    Examples\n    --------\n    >>> Discret")
            end = text.index("\n\n", start)
            return text[:start] + text[end + 2:]

        counts = self._counts_with(gate, rel, mutate)
        assert rel not in counts, "the mutation must remove every example"
        errors, _slack = gate.per_file_errors(counts, gate.EXAMPLES_PER_FILE)
        assert any(e.startswith(f"{rel}: 0 example(s)") for e in errors), errors

    def test_a_file_that_loses_its_only_example_fails_the_gate(self, tmp_path):
        """D6 end to end: the total still clears ``--min``; the pin does not."""
        other = '"""Module."""\n\n\ndef one():\n    """One.\n\n    >>> 3\n    3\n    """\n'
        pins = {"src/probe_pkg/mod.py": 2, "src/probe_pkg/other.py": 1}
        ok = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "2", pins=pins,
                           extra={"other.py": other})
        assert ok.returncode == 0, ok.stdout + ok.stderr
        gone = other.replace("    >>> 3\n    3\n", "")
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "2",
                               pins=pins, extra={"other.py": gone})
        assert result.returncode == 1, result.stdout + result.stderr
        assert ("src/probe_pkg/other.py: 0 example(s) executed and passed, "
                "pinned at 1") in result.stdout

    def test_one_example_deleted_from_a_docstring_fails_the_gate(self, tmp_path):
        """D5 end to end, with the total floor one below the count."""
        source = _TWO_EXAMPLES.replace("    >>> 2 + 2\n    4\n", "")
        result = _doctest_gate(tmp_path, source, "--min", "1",
                               pins={"src/probe_pkg/mod.py": 2})
        assert result.returncode == 1, result.stdout + result.stderr
        assert ("src/probe_pkg/mod.py: 1 example(s) executed and passed, "
                "pinned at 2") in result.stdout

    def test_a_file_with_examples_and_no_pin_fails_the_gate(self, tmp_path):
        """Fail closed: an unpinned file's examples are guarded by nothing."""
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "2", pins={})
        assert result.returncode == 1, result.stdout + result.stderr
        assert ("src/probe_pkg/mod.py holds docstring examples and has no "
                "entry in EXAMPLES_PER_FILE") in result.stdout

    def test_a_pin_below_its_count_is_reported_as_slack(self, tmp_path):
        result = _doctest_gate(tmp_path, _TWO_EXAMPLES, "--min", "2",
                               pins={"src/probe_pkg/mod.py": 1})
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("NOTE: src/probe_pkg/mod.py: 2 example(s), pinned at 1"
                in result.stdout)

