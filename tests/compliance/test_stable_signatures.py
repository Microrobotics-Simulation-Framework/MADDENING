"""The committed STABLE-signature snapshot matches the tree, and a change is caught.

``scripts/check_stable_signatures.py`` is the mechanical half of the promise a
``STABLE`` tag makes (``docs/developer_guide/deprecation_policy.md``): the
signature does not change before the next major version.  These tests check
both directions — the current tree passes, and a signature that differs from
the snapshot fails with a message that says what to do about it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_stable_signatures.py"
SNAPSHOT = REPO_ROOT / "docs" / "developer_guide" / "stable_api.json"


def _run(*args: str) -> subprocess.CompletedProcess:
    """Run the guard in a fresh interpreter (imports must not be pre-warmed)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


@pytest.fixture(scope="module")
def guard():
    """The guard module, imported by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("_stable_sig_guard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The current tree passes
# ---------------------------------------------------------------------------

class TestTheTreeMatchesTheSnapshot:
    def test_the_committed_snapshot_is_present_and_current(self):
        assert SNAPSHOT.exists(), (
            f"{SNAPSHOT.relative_to(REPO_ROOT)} is missing; regenerate it with "
            "python scripts/check_stable_signatures.py --update"
        )
        out = _run()
        assert out.returncode == 0, out.stdout + out.stderr

    def test_the_snapshot_records_exactly_the_stable_registry(self, guard):
        """No STABLE surface is missing from the snapshot and none is stale.

        The generator's module list, not this file, decides which modules are
        imported; ``tests/compliance/test_stability.py`` guards that list.
        """
        from maddening.core.compliance.metadata import StabilityLevel

        registry, _skipped = guard.load_registry()
        expected = {
            name for name, level in registry.items()
            if level is StabilityLevel.STABLE
        }
        recorded = set(json.loads(SNAPSHOT.read_text())["surfaces"])
        assert recorded == expected

    def test_a_tag_applied_outside_the_package_is_ignored(self, guard):
        """``@stability`` writes to a process-global registry, and
        ``tests/compliance/test_stability.py`` applies it to throwaway classes
        defined inside its test functions.  Those are not surfaces the package
        promises, and they cannot even be resolved by name."""
        from maddening.core.compliance.metadata import StabilityLevel
        from maddening.core.compliance.stability import stability

        @stability(StabilityLevel.STABLE)
        class NotAPublicSurface:
            pass

        registry, _ = guard.load_registry()
        assert not [name for name in registry if "NotAPublicSurface" in name]
        assert all(name.startswith("maddening.") for name in registry)

    def test_every_recorded_surface_carries_its_module_and_kind(self):
        surfaces = json.loads(SNAPSHOT.read_text())["surfaces"]
        assert surfaces, "the snapshot records no surfaces at all"
        for name, record in surfaces.items():
            assert record["kind"] in ("class", "function"), name
            assert record["module"].startswith("maddening"), name
            assert isinstance(record["parameters"], list), name
            if record["kind"] == "class":
                assert "members" in record, name

    def test_a_stable_class_records_the_methods_it_inherits(self):
        """A tagged node promises the whole ``SimulationNode`` protocol, not
        only what it overrides, so a change to an inherited hook shows up
        against every tagged node that resolves it."""
        surfaces = json.loads(SNAPSHOT.read_text())["surfaces"]
        members = surfaces["maddening.nodes.ball.BallNode"]["members"]
        base = surfaces["maddening.core.node.SimulationNode"]["members"]
        for name in ("update", "initial_state", "state_fields", "params_pytree"):
            assert name in members, name
        # ``update`` is overridden (its own annotations), ``update_padded`` is
        # not: the inherited record is byte-identical to the base's.
        assert members["update_padded"] == base["update_padded"]
        assert set(base) <= set(members)


# ---------------------------------------------------------------------------
# A changed signature is caught
# ---------------------------------------------------------------------------

def _mutated_snapshot(tmp_path: Path, mutate) -> Path:
    """Copy the committed snapshot into *tmp_path* with *mutate* applied.

    Mutating the snapshot rather than the source tree is the same comparison
    — recorded against current — and leaves the working tree untouched.
    """
    data = json.loads(SNAPSHOT.read_text())
    mutate(data["surfaces"])
    path = tmp_path / "stable_api.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


class TestAChangedSignatureFails:
    def test_a_renamed_parameter_on_a_stable_function_fails(self, tmp_path):
        def mutate(surfaces):
            fn = surfaces["maddening.cloud.multigpu.iterative_solver.sharded_cg"]
            fn["parameters"][0]["name"] = "matrix_vector_product"

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        assert "sharded_cg: CHANGED" in out.stdout
        assert "matrix_vector_product" in out.stdout
        assert "major version" in out.stdout

    def test_a_changed_default_fails(self, tmp_path):
        def mutate(surfaces):
            for p in surfaces["maddening.nodes.ball.BallNode"]["parameters"]:
                if p["name"] == "elasticity":
                    p["default"] = "0.5"

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        assert "BallNode: CHANGED" in out.stdout

    def test_a_changed_method_signature_on_a_stable_class_fails(self, tmp_path):
        def mutate(surfaces):
            gm = surfaces["maddening.core.graph_manager.GraphManager"]["members"]
            gm["add_edge"]["parameters"].append(
                {"name": "lossy", "kind": "KEYWORD_ONLY", "default": "True"}
            )

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        assert "GraphManager.add_edge: CHANGED" in out.stdout

    def test_a_removed_surface_fails(self, tmp_path):
        def mutate(surfaces):
            surfaces["maddening.core.graph_manager.GraphManager"]["members"].pop("run_scan")

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        # The tree still has it, so relative to the mutated snapshot it reads
        # as an unrecorded addition -- which is also a failure, with the
        # "regenerate" message rather than the "major bump" one.
        assert "run_scan" in out.stdout
        assert "--update" in out.stdout

    def test_a_surface_the_snapshot_does_not_know_fails_as_an_addition(self, tmp_path):
        def mutate(surfaces):
            surfaces.pop("maddening.nodes.table.TableNode")

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        assert "not in the snapshot" in out.stdout
        assert "TableNode" in out.stdout
        assert "not a break" in out.stdout

    def test_a_vanished_surface_reads_as_a_removal(self, tmp_path, guard):
        """A surface in the snapshot that the tree no longer defines at all."""
        def mutate(surfaces):
            surfaces["maddening.nodes.gone.GhostNode"] = {
                "kind": "class", "module": "maddening.nodes.gone",
                "parameters": [], "members": {},
            }

        path = _mutated_snapshot(tmp_path, mutate)
        out = _run("--snapshot", str(path))
        assert out.returncode == 1, out.stdout + out.stderr
        assert "GhostNode: REMOVED" in out.stdout
        assert "major version" in out.stdout

    def test_a_snapshot_in_an_older_format_is_refused(self, tmp_path):
        data = json.loads(SNAPSHOT.read_text())
        data["format"] = 0
        path = tmp_path / "stable_api.json"
        path.write_text(json.dumps(data))
        out = _run("--snapshot", str(path))
        assert out.returncode == 1
        assert "--update" in out.stderr

    def test_a_missing_snapshot_is_refused_rather_than_created(self, tmp_path):
        path = tmp_path / "absent.json"
        out = _run("--snapshot", str(path))
        assert out.returncode == 1
        assert not path.exists()
        assert "--update" in out.stderr


# ---------------------------------------------------------------------------
# Accepting an intended change
# ---------------------------------------------------------------------------

class TestUpdateAcceptsAnIntendedChange:
    def test_update_heals_a_mutated_snapshot_and_the_check_then_passes(self, tmp_path):
        def mutate(surfaces):
            surfaces["maddening.core.edge.EdgeSpec"]["parameters"][0]["name"] = "src"

        path = _mutated_snapshot(tmp_path, mutate)
        assert _run("--snapshot", str(path)).returncode == 1
        updated = _run("--update", "--snapshot", str(path))
        assert updated.returncode == 0, updated.stdout + updated.stderr
        assert _run("--snapshot", str(path)).returncode == 0

    def test_update_reproduces_the_committed_snapshot_byte_for_byte(self, tmp_path):
        """Regenerating must be a no-op on an unchanged tree, or every branch
        that runs ``--update`` produces a spurious diff."""
        path = tmp_path / "stable_api.json"
        shutil.copy(SNAPSHOT, path)
        assert _run("--update", "--snapshot", str(path)).returncode == 0
        assert path.read_text() == SNAPSHOT.read_text()


# ---------------------------------------------------------------------------
# Missing optional dependencies
# ---------------------------------------------------------------------------

class TestAMissingOptionalDependency:
    """The CI ``compliance`` job installs ``.[ci]``, which has no ``usd-core``,
    so ``maddening.usd.live_stage`` does not import there.  A module that
    carries no ``STABLE`` surface must not turn into a wall of false
    removals; one that does must stop the check rather than pass it."""

    def test_a_skipped_module_with_no_stable_surface_still_passes(self, guard, monkeypatch, capsys):
        registry, _ = guard.load_registry()
        monkeypatch.setattr(
            guard, "load_registry",
            lambda: (registry, {"maddening.usd.live_stage": "No module named 'pxr'"}),
        )
        assert guard.main([]) == 0
        assert "maddening.usd.live_stage not imported" in capsys.readouterr().err

    def test_a_skipped_module_carrying_a_stable_surface_stops_the_check(self, guard, monkeypatch, capsys):
        registry, _ = guard.load_registry()
        monkeypatch.setattr(
            guard, "load_registry",
            lambda: (registry, {"maddening.nodes.ball": "No module named 'nope'"}),
        )
        assert guard.main([]) == 2
        err = capsys.readouterr().err
        assert "cannot be trusted" in err
        assert "maddening.nodes.ball.BallNode" in err


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------

class TestComparison:
    def _snap(self, record):
        return {"format": 1, "surfaces": {"pkg.mod.thing": record}}

    def test_an_identical_snapshot_reports_nothing(self, guard):
        record = {"kind": "function", "module": "pkg.mod",
                  "parameters": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD"}]}
        breaking, additions = guard.compare(self._snap(record), self._snap(record))
        assert (breaking, additions) == ([], [])

    def test_a_changed_return_annotation_is_breaking(self, guard):
        old = {"kind": "function", "module": "pkg.mod", "parameters": [],
               "returns": "dict"}
        new = dict(old, returns="dict[str, float]")
        breaking, additions = guard.compare(self._snap(old), self._snap(new))
        assert len(breaking) == 1 and additions == []
        assert "CHANGED" in breaking[0]

    def test_reordering_two_parameters_is_breaking(self, guard):
        a = {"name": "a", "kind": "POSITIONAL_OR_KEYWORD"}
        b = {"name": "b", "kind": "POSITIONAL_OR_KEYWORD"}
        old = {"kind": "function", "module": "pkg.mod", "parameters": [a, b]}
        new = {"kind": "function", "module": "pkg.mod", "parameters": [b, a]}
        breaking, _ = guard.compare(self._snap(old), self._snap(new))
        assert len(breaking) == 1

    def test_turning_a_keyword_into_positional_is_breaking(self, guard):
        old = {"kind": "function", "module": "pkg.mod",
               "parameters": [{"name": "x", "kind": "KEYWORD_ONLY", "default": "1"}]}
        new = {"kind": "function", "module": "pkg.mod",
               "parameters": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD",
                               "default": "1"}]}
        breaking, _ = guard.compare(self._snap(old), self._snap(new))
        assert len(breaking) == 1

    def test_a_new_surface_is_an_addition_not_a_break(self, guard):
        old = {"format": 1, "surfaces": {}}
        new = self._snap({"kind": "function", "module": "pkg.mod", "parameters": []})
        breaking, additions = guard.compare(old, new)
        assert breaking == [] and len(additions) == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_a_default_that_embeds_an_object_address_is_an_error(self, guard):
        """An ``object()`` sentinel default would differ on every run; that is
        an infrastructure failure, not a silent pass."""
        with pytest.raises(ValueError, match="object address"):
            guard._default(object())

    def test_pep563_and_runtime_annotations_render_the_same(self, guard):
        from typing import Callable, Optional

        assert guard._annotation(Optional[str]) == "Optional[str]"
        assert guard._annotation("Optional[str]") == "Optional[str]"
        assert guard._annotation(Optional[Callable]) == "Optional[Callable]"
        assert guard._annotation(str) == "str"


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

class TestFlatteningKeepsEveryRecord:
    """A tagged method is both a class member and a surface of its own.

    ``SimulationNode.static_data_deps`` and ``invalidate_static_cache`` on
    three classes carry their own ``@stability`` *and* appear in the members
    of the class that resolves them.  Both renderings flatten to the same
    key and are not identical -- the surface keeps ``self``, the member drops
    it -- so one used to overwrite the other on the way into the comparison,
    and the totals the two code paths printed for one snapshot disagreed by
    exactly that number.
    """

    def _dual_registered(self) -> set[str]:
        surfaces = json.loads(SNAPSHOT.read_text())["surfaces"]
        return {
            f"{name}.{member}"
            for name, record in surfaces.items()
            for member in record.get("members", {})
            if f"{name}.{member}" in surfaces
        }

    def test_the_situation_this_guards_actually_occurs(self):
        """Without a dual-registered surface the rest of this class is vacuous."""
        dual = self._dual_registered()
        assert dual, (
            "no tagged method is also a member of a tagged class, so the "
            "collision these tests pin cannot arise; if that is deliberate, "
            "delete this class rather than leaving it passing on nothing"
        )
        assert "maddening.core.node.SimulationNode.static_data_deps" in dual

    def test_no_record_is_dropped_on_the_way_into_the_comparison(self, guard):
        snapshot = json.loads(SNAPSHOT.read_text())
        flat = guard._flatten(snapshot)
        surfaces = snapshot["surfaces"]
        expected = set(surfaces) | {
            f"{name}.{member}"
            for name, record in surfaces.items()
            for member in record.get("members", {})
        }
        assert set(flat) == expected
        assert len(flat) == len(expected)

    def test_the_registered_surface_wins_over_the_member_rendering(self, guard):
        """The registry is the promise, so its record is the one compared."""
        name = "maddening.core.node.SimulationNode.static_data_deps"
        snapshot = json.loads(SNAPSHOT.read_text())
        flat = guard._flatten(snapshot)
        surface = snapshot["surfaces"][name]
        member = snapshot["surfaces"]["maddening.core.node.SimulationNode"]["members"]["static_data_deps"]
        # The two renderings really do differ, or there is nothing to choose.
        assert [p["name"] for p in surface["parameters"]] == ["self"]
        assert member["parameters"] == []
        assert flat[name] == {k: v for k, v in surface.items() if k != "members"}

    def test_a_member_record_never_overwrites_the_surface_it_shadows(self, guard):
        """A synthetic snapshot whose class is visited *after* the shadowed
        surface.  The committed snapshot happens to sort the class first, so
        on real data the surface survived by luck; this pins the rule."""
        snapshot = {"format": 1, "surfaces": {
            "pkg.mod.Thing.hook": {
                "kind": "function", "module": "pkg.mod",
                "parameters": [{"name": "self", "kind": "POSITIONAL_OR_KEYWORD"}],
            },
            "pkg.mod.Thing": {
                "kind": "class", "module": "pkg.mod", "parameters": [],
                "members": {"hook": {"kind": "method", "parameters": []}},
            },
        }}
        flat = guard._flatten(snapshot)
        assert flat["pkg.mod.Thing.hook"]["kind"] == "function"
        assert [p["name"] for p in flat["pkg.mod.Thing.hook"]["parameters"]] == ["self"]
        assert set(flat) == {"pkg.mod.Thing", "pkg.mod.Thing.hook"}

    def test_flattening_does_not_depend_on_the_order_of_the_surfaces(self, guard):
        snapshot = json.loads(SNAPSHOT.read_text())
        reversed_snapshot = {
            "format": snapshot["format"],
            "surfaces": dict(reversed(list(snapshot["surfaces"].items()))),
        }
        assert guard._flatten(reversed_snapshot) == guard._flatten(snapshot)

    def test_both_code_paths_report_the_same_totals(self, tmp_path):
        """``--update`` and the passing check counted the same snapshot
        differently (243 against 239): one summed the recorded members, the
        other the flattened keys.  A gate that miscounts its own evidence is
        the defect this release has hit twice."""
        path = tmp_path / "stable_api.json"
        shutil.copy(SNAPSHOT, path)
        updated = _run("--update", "--snapshot", str(path))
        checked = _run("--snapshot", str(path))
        assert updated.returncode == 0 and checked.returncode == 0

        written = updated.stdout.split(":", 1)[1].strip()
        confirmed = checked.stdout.split(":", 1)[1].strip().removesuffix(" unchanged")
        assert written == confirmed, (
            f"--update says {written!r}, the check says {confirmed!r}"
        )
        # and the number is the real one, not a count of nothing
        n_surfaces = len(json.loads(SNAPSHOT.read_text())["surfaces"])
        assert written.startswith(f"{n_surfaces} STABLE surface(s), ")
        assert "0 member(s)" not in written
