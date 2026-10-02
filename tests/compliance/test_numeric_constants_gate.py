"""``scripts/check_numeric_constants.py`` must be able to fail.

The gate requires every small float literal and every additive use of a dtype
constant (``finfo(...).tiny`` / ``.eps``) in the numerical core to say what it
is relative to.  Four 0.4.0 defects were one absolute constant inside a
relative computation (the IFT solve's ``atol=1e-8``, IQN's ``1e-12`` floor,
``fit_lm``'s ``1e-12`` Marquardt floor, the accelerators' flushed steps), and
the gate found four more that had shipped (the public Krylov solvers'
``atol=1e-8``, the multi-rate GCD's ``1e-9``, Adam's ``eps``, the adaptive
error norm's ``1e-300``).  Each test below plants the defect the gate exists
to catch -- an unjustified literal, a stale allowlist line, an empty scope --
and asserts a non-zero exit.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "check_numeric_constants.py"


def _load():
    spec = importlib.util.spec_from_file_location("_gate_numeric_constants", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load()


def _tree(tmp_path, modules, allowlist=""):
    """A throwaway repository: ``pkg/<name>.py`` for each module, one allowlist."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    for name, text in modules.items():
        (pkg / f"{name}.py").write_text(textwrap.dedent(text))
    (tmp_path / "allow.txt").write_text(textwrap.dedent(allowlist))
    return tmp_path


def _check(gate, root, scope=("pkg",)):
    errors, inline, listed, _entries = gate.check(root, root / "allow.txt", scope)
    return errors, inline, listed


def _tokens(hits):
    return sorted((h.qualname, h.token) for h in hits)


# ---------------------------------------------------------------------------
# The repository itself
# ---------------------------------------------------------------------------
def test_the_repository_passes_and_every_allowlist_line_is_used():
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run([sys.executable, str(GATE_PATH)], capture_output=True,
                            text=True, env=env, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: "), result.stdout


def test_the_repository_scan_reads_the_whole_scope(gate):
    files, errors = gate.scope_files(REPO_ROOT, gate.SCOPE)
    assert errors == []
    rels = {rel for _p, rel in files}
    assert "src/maddening/sysid.py" in rels
    assert any(r.startswith("src/maddening/cloud/multigpu/") for r in rels)
    assert sum(r.startswith("src/maddening/core/") for r in rels) > 30


# ---------------------------------------------------------------------------
# What counts as a hit
# ---------------------------------------------------------------------------
def test_a_small_literal_is_a_hit_and_an_ordinary_one_is_not(gate, tmp_path):
    root = _tree(tmp_path, {"m": """
        A = 1e-12
        B = 0.001
        C = 15e-4            # 1.5e-3, but written with a small exponent
        D = -1e-30
        E = 0.002
        F = 0.0
        G = 3
        H = 1e6
        I = 1e-3j            # complex: not a float literal
        def f(x, tol=1e-9):
            return x * 0.5
        """})
    errors, _inline, listed = _check(gate, root)
    assert _tokens(listed) == sorted([
        ("<module>", "1e-12"), ("<module>", "0.001"), ("<module>", "15e-4"),
        ("<module>", "1e-30"), ("f", "1e-9")])
    assert len(errors) == 5


def test_a_dtype_constant_is_a_hit_only_where_it_is_absolute(gate, tmp_path):
    root = _tree(tmp_path, {"m": """
        import jax.numpy as jnp

        def additive(x, d):
            return x + jnp.finfo(d).eps

        def floor(x, d):
            return jnp.maximum(x, jnp.finfo(d).tiny)

        def bound(x, d):
            eps = jnp.finfo(d).eps
            return x - 4 * eps

        def through_an_object(x, d):
            info = jnp.finfo(d)
            return max(x, info.tiny)

        def passed_in(x, fi):
            return x + fi.smallest_normal

        def relative(x, d):
            return x * jnp.finfo(d).eps + jnp.abs(x) * jnp.finfo(d).eps

        def compared(x, d):
            return x < jnp.finfo(d).tiny

        def scaled_by_a_quantity(x, d):
            eps = jnp.finfo(d).eps
            return x + eps * jnp.abs(x)
        """})
    _errors, _inline, listed = _check(gate, root)
    assert _tokens(listed) == sorted([
        ("additive", "finfo.eps"), ("floor", "finfo.tiny"), ("bound", "finfo.eps"),
        ("through_an_object", "finfo.tiny"), ("passed_in", "finfo.smallest_normal")])


# ---------------------------------------------------------------------------
# The two justifications
# ---------------------------------------------------------------------------
def test_an_inline_units_comment_justifies_a_hit(gate, tmp_path):
    root = _tree(tmp_path, {"m": """
        A = 1e-12  # units: dimensionless, a fraction of max|b|
        # The floor of the stopping test.
        # units: dimensionless, relative to the largest entry
        B = 1e-8
        C = 1e-6
        """})
    errors, inline, listed = _check(gate, root)
    assert _tokens(inline) == [("<module>", "1e-12"), ("<module>", "1e-8")]
    assert _tokens(listed) == [("<module>", "1e-6")]
    assert len(errors) == 1 and "pkg/m.py:6" in errors[0]


def test_an_allowlist_line_justifies_exactly_the_hits_it_counts(gate, tmp_path):
    root = _tree(tmp_path, {"m": """
        def f(x):
            return x + 1e-12 * x - 1e-12
        """}, "pkg/m.py f 1e-12*2  # dimensionless: a fraction of the iterate's own size\n")
    errors, _inline, _listed = _check(gate, root)
    assert errors == []


# ---------------------------------------------------------------------------
# Planted defects: each must fail the gate
# ---------------------------------------------------------------------------
def test_an_unjustified_literal_fails_the_gate(gate, tmp_path):
    root = _tree(tmp_path, {"m": "def f(x):\n    return max(x, 1e-12)\n"})
    errors, _i, _l = _check(gate, root)
    assert len(errors) == 1
    assert "pkg/m.py:2" in errors[0] and "1e-12" in errors[0] and "no units" in errors[0]
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, str(GATE_PATH), "--root", str(root), "--allowlist",
         str(root / "allow.txt"), "--scope", "pkg"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 1, result.stdout


def test_a_second_copy_of_an_allowlisted_literal_fails_the_gate(gate, tmp_path):
    root = _tree(tmp_path, {"m": """
        def f(x):
            return x + 1e-12 + 1e-12
        """}, "pkg/m.py f 1e-12  # dimensionless: a fraction of the iterate's own size\n")
    errors, _i, _l = _check(gate, root)
    assert len(errors) == 1 and "expects 1 hit(s) and the scope holds 2" in errors[0]


def test_a_stale_allowlist_line_fails_the_gate(gate, tmp_path):
    root = _tree(tmp_path, {"m": "def f(x):\n    return x\n"},
                 "pkg/m.py f 1e-12  # dimensionless: a fraction of the iterate's own size\n")
    errors, _i, _l = _check(gate, root)
    assert len(errors) == 1 and "stale" in errors[0]


def test_an_allowlist_line_for_a_moved_literal_is_stale_in_its_old_scope(gate, tmp_path):
    root = _tree(tmp_path, {"m": "def g(x):\n    return x + 1e-12\n"},
                 "pkg/m.py f 1e-12  # dimensionless: a fraction of the iterate's own size\n")
    errors, _i, _l = _check(gate, root)
    assert len(errors) == 2
    assert any("stale" in e for e in errors) and any("no units" in e for e in errors)


@pytest.mark.parametrize("line, why", [
    ("pkg/m.py f 1e-12\n", "does not parse"),
    ("pkg/m.py f 1e-12  # small\n", "cannot say what"),
    ("pkg/m.py f 1e-12  # dimensionless, relative to x\n"
     "pkg/m.py f 1e-12  # dimensionless, relative to x\n", "repeats line 1"),
])
def test_a_malformed_allowlist_line_fails_the_gate(gate, tmp_path, line, why):
    root = _tree(tmp_path, {"m": "def f(x):\n    return x + 1e-12\n"}, line)
    errors, _i, _l = _check(gate, root)
    assert any(why in e for e in errors), errors


def test_an_empty_scan_scope_fails_the_gate(gate, tmp_path):
    root = _tree(tmp_path, {"m": "A = 1e-12  # units: dimensionless, relative to x\n"})
    (root / "empty").mkdir()
    for scope, why in [(("nowhere",), "does not exist"), (("empty",), "holds no Python file"),
                       ((), "empty")]:
        errors, _i, _l = _check(gate, root, scope)
        assert any(why in e for e in errors), (scope, errors)
    result = subprocess.run(
        [sys.executable, str(GATE_PATH), "--root", str(root), "--allowlist",
         str(root / "allow.txt"), "--scope", "nowhere"],
        capture_output=True, text=True)
    assert result.returncode == 1, result.stdout


def test_a_repository_scope_that_moved_fails_the_gate(gate, tmp_path):
    """The default scope names real paths: if ``src/maddening/core`` moved, the
    gate must say it scanned nothing there, not report OK."""
    errors, _i, _l = gate.check(tmp_path, tmp_path / "allow.txt", gate.SCOPE)[:3]
    assert len([e for e in errors if "does not exist" in e]) == len(gate.SCOPE)
