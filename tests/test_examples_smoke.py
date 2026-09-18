"""Smoke coverage for the ~50 scripts under ``maddening.examples``.

The examples ship inside the wheel and are the first thing a new user
copies, but nothing else in the suite touches them, so an API rename can
break every one of them without turning a single test red.

Two lanes, deliberately:

* A **static lane**, in the default suite and costing seconds.  It parses
  every example and resolves the ``maddening.*`` names it imports against
  the installed library, which is the failure mode a release actually
  causes.  It also pins the two packaging invariants that were broken
  before: no example writes its output into the installed package, and no
  example puts the package directory on ``sys.path``.  Static checking is
  the *only* coverage the ``examples/cloud/`` scripts can ever have -- they
  allocate real, billable machines through SkyPilot, so the suite must
  never execute them.

* A **run lane**, marked ``slow``, which actually executes a handful of
  cheap headless examples in a subprocess and asserts they exit 0.  Run it
  with ``pytest -m 'slow or not slow'``.

The split is the point: the cheap lane is the one that must not rot, so
the expensive lane is kept small enough that it stays worth enabling.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import maddening.examples

EXAMPLES_ROOT = Path(maddening.examples.__file__).resolve().parent

# Everything below this directory launches billable cloud infrastructure.
# Never execute it; static analysis only.
CLOUD_DIR = EXAMPLES_ROOT / "cloud"


def _example_files() -> list[Path]:
    """Every example script, excluding the empty package ``__init__``s."""
    return sorted(
        p for p in EXAMPLES_ROOT.rglob("*.py") if p.name != "__init__.py"
    )


def _rel(path: Path) -> str:
    return path.relative_to(EXAMPLES_ROOT).as_posix()


EXAMPLE_FILES = _example_files()
EXAMPLE_IDS = [_rel(p) for p in EXAMPLE_FILES]

# Optional extras whose absence makes a *library* module unimportable.  When
# one is missing the example that uses it cannot be checked, and that is a
# skip with a reason, not a failure.
_EXTRA_FOR_MODULE = {
    "maddening.usd": "usd",
    "maddening.api": "api",
    "maddening.cloud": "cloud",
    "maddening.surrogates": "surrogates",
    "maddening.viz": "viz",
}


def _extra_hint(module_name: str) -> str:
    for prefix, extra in _EXTRA_FOR_MODULE.items():
        if module_name == prefix or module_name.startswith(prefix + "."):
            return f"maddening[{extra}]"
    return module_name


def _maddening_imports(tree: ast.AST) -> list[tuple[str, tuple[str, ...], int]]:
    """``(module, imported_names, lineno)`` for every ``maddening`` import."""
    found: list[tuple[str, tuple[str, ...], int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; the examples use none.
            if node.level == 0 and node.module and (
                node.module == "maddening" or node.module.startswith("maddening.")
            ):
                found.append(
                    (node.module, tuple(a.name for a in node.names), node.lineno)
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "maddening" or alias.name.startswith("maddening."):
                    found.append((alias.name, (), node.lineno))
    return found


def _unresolved_names(tree: ast.AST) -> list[str]:
    """Names an example imports from ``maddening`` that no longer exist.

    Raises ``ImportError`` if a *library* module is missing entirely, so the
    caller can turn that into a skip when it is only an uninstalled extra.
    """
    problems: list[str] = []
    for module_name, names, lineno in _maddening_imports(tree):
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                continue
            try:  # ``from pkg import submodule`` is also legal
                importlib.import_module(f"{module_name}.{name}")
            except ImportError:
                problems.append(f"line {lineno}: {module_name}.{name}")
    return problems


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_parses(path: Path) -> None:
    """Every shipped example is valid Python for the interpreter we target."""
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_imports_resolve_against_the_library(path: Path) -> None:
    """Every ``maddening`` name an example imports still exists.

    This is what catches a release renaming or moving a public symbol, and
    it works without executing the example -- the only option for the cloud
    scripts.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    try:
        problems = _unresolved_names(tree)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        pytest.skip(f"optional dependency missing ({missing}); needs {_extra_hint(missing)}")
    assert not problems, (
        f"{_rel(path)} imports names that no longer exist: " + "; ".join(problems)
    )


def _embedded_payloads(tree: ast.AST) -> list[ast.AST]:
    """Parse the remote-side scripts the cloud examples embed as strings.

    ``examples/cloud/*`` ships its payloads as string literals that are
    executed on the rented VM, so they are the part most likely to drift and
    the part no local run ever touches.  Literals that are not whole modules
    (indented comment fragments, shell snippets) simply do not parse and are
    skipped -- this checks what it can, and claims nothing more.
    """
    payloads = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        source = textwrap.dedent(node.value)
        if "\nfrom maddening" not in "\n" + source:
            continue
        try:
            payloads.append(ast.parse(source))
        except SyntaxError:
            continue
    return payloads


CLOUD_FILES = [p for p in EXAMPLE_FILES if CLOUD_DIR in p.parents]


@pytest.mark.parametrize(
    "path", CLOUD_FILES, ids=[_rel(p) for p in CLOUD_FILES]
)
def test_cloud_example_remote_payload_imports_resolve(path: Path) -> None:
    """The scripts cloud examples run on the rented VM still import cleanly."""
    tree = ast.parse(path.read_text(), filename=str(path))
    payloads = _embedded_payloads(tree)
    if not payloads:
        pytest.skip("no embedded remote payload in this script")
    problems: list[str] = []
    for payload in payloads:
        try:
            problems.extend(_unresolved_names(payload))
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            pytest.skip(
                f"optional dependency missing ({missing}); needs {_extra_hint(missing)}"
            )
    assert not problems, (
        f"{_rel(path)} remote payload imports names that no longer exist: "
        + "; ".join(problems)
    )


def _sys_path_mutations(tree: ast.AST) -> list[int]:
    """Line numbers of ``sys.path.insert``/``append`` calls in real code."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("insert", "append")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_does_not_write_into_the_installed_package(path: Path) -> None:
    """Examples write results to the working directory, not to site-packages.

    ``dirname(__file__)/../..`` is ``src/maddening`` under the src layout, so
    joining an output filename onto it drops the file next to the library --
    which is site-packages for anyone who pip-installed MADDENING, and fails
    outright when that install is read-only.
    """
    source = path.read_text()
    assert "_project_root" not in source, (
        f"{_rel(path)} resurrects the pre-src-layout _project_root: it resolves "
        "to the package directory, not to a project root. Write outputs to "
        "os.getcwd() instead."
    )
    # AST, not a substring search: the cloud examples legitimately embed
    # ``sys.path.insert`` inside the payload strings they run on the rented
    # VM, where the checkout really is somewhere sys.path does not know.
    assert not _sys_path_mutations(ast.parse(source, filename=str(path))), (
        f"{_rel(path)} puts a directory on sys.path. The examples run against "
        "an installed maddening; a path hack only makes core/nodes/viz "
        "importable as top-level packages and duplicates module objects."
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_usage_docstring_is_runnable(path: Path) -> None:
    """Usage lines name a module path that ``python -m`` can actually run."""
    source = path.read_text()
    assert "python maddening/examples/" not in source, (
        f"{_rel(path)} documents the pre-src-layout script path. Use "
        "'python -m maddening.examples.<package>.<module>'."
    )
    assert "/home/nick" not in source, (
        f"{_rel(path)} hard-codes a developer's home directory in its usage "
        "instructions."
    )


# ---------------------------------------------------------------------------
# Run lane
# ---------------------------------------------------------------------------
# Cheap, headless, non-interactive examples, chosen to cover the surfaces this
# release changed: the parameter pytree, the coupling solver default, the
# derivative rule, and interface mappings.  Each costs a few seconds, almost
# all of it JAX import and tracing.  Deliberately not the whole set: the GUI,
# server and cloud examples block forever or cost money.
RUNNABLE_EXAMPLES = [
    "maddening.examples.basics.bouncing_ball",
    "maddening.examples.basics.heat_diffusion_demo",
    "maddening.examples.basics.rigid_body_demo",
    "maddening.examples.advanced.adaptive_demo",
    "maddening.examples.advanced.parameter_sweep_demo",
    "maddening.examples.coupling.coupling_demo",
    "maddening.examples.coupling.subcycling_demo",
    "maddening.examples.coupling.spatial_interpolation_demo",
]


@pytest.mark.slow
@pytest.mark.parametrize("module", RUNNABLE_EXAMPLES)
def test_headless_example_runs_to_completion(module: str, tmp_path: Path) -> None:
    """The cheap headless examples still run end to end and exit 0.

    Run from ``tmp_path`` so the plots they save land there, which also
    proves they no longer write into the installed package.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{module} exited {result.returncode}\n"
        f"--- stdout tail ---\n{result.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
