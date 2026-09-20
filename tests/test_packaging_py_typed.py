"""The built wheel must carry the PEP 561 marker and the force-included data.

Under `PEP 561 <https://peps.python.org/pep-0561/>`_ a type checker
consuming an installed package **must ignore every annotation in it**
unless the package ships a ``py.typed`` marker inside the installed
package directory.  ``src/maddening/py.typed`` existing in the source
tree is therefore not the claim that matters: the claim is that the
*distribution* contains ``maddening/py.typed``.

Hatchling only copies non-Python files that a ``force-include`` entry
names, and a ``force-include`` entry whose source path is wrong is
silently a no-op -- the build succeeds and ships nothing.  So this
module builds a real wheel and reads its member list.  The USD schema
files are asserted alongside the marker because they are the other
force-included resources and share the same failure mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every path the wheel must contain that is not a ``.py`` file, i.e.
#: everything that depends on ``[tool.hatch.build.targets.wheel.force-include]``
#: being correct.
REQUIRED_WHEEL_MEMBERS = (
    "maddening/py.typed",
    "maddening/usd/schema/generatedSchema.usda",
    "maddening/usd/schema/plugInfo.json",
)

#: Set in CI so that a missing build backend fails the job instead of
#: quietly skipping the only test that proves the marker is shipped.
_REQUIRED_ENV = "MADDENING_REQUIRE_PACKAGING_TESTS"


def _require_or_skip(reason: str) -> None:
    if os.environ.get(_REQUIRED_ENV):
        pytest.fail(f"{_REQUIRED_ENV} is set but {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    """Build a wheel from the checkout with the configured backend."""
    try:
        import hatchling  # noqa: F401
    except ImportError:
        _require_or_skip(
            "hatchling (the build backend declared in pyproject.toml) is not "
            "installed; `pip install -e .[ci]` provides it"
        )
    out = tmp_path_factory.mktemp("wheel")
    # In a subprocess: the backend imports and warns on its own schedule,
    # and `filterwarnings = ["error"]` would turn any of that into a
    # failure of this test rather than of the build.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hatchling.build import build_wheel; "
            "print(build_wheel(sys.argv[1]))",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"wheel build failed (exit {proc.returncode}):\n{proc.stderr}")
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel in {out}, got {wheels}"
    return zipfile.ZipFile(wheels[0])


def test_py_typed_marker_exists_in_source_tree() -> None:
    """The marker file itself is present and empty.

    PEP 561 gives the *contents* of ``py.typed`` a meaning only for
    stub-only distributions (``partial\\n``).  For an inline-annotated
    package like this one the file is empty, and writing anything into
    it risks a checker reading it as a partial-stub declaration.
    """
    marker = REPO_ROOT / "src" / "maddening" / "py.typed"
    assert marker.is_file(), f"{marker} is missing; PEP 561 requires the marker"
    assert marker.read_bytes() == b"", (
        "py.typed must be empty for an inline-annotated package; contents are "
        "only meaningful for stub-only distributions"
    )


def test_wheel_ships_force_included_data(built_wheel: zipfile.ZipFile) -> None:
    """Every force-included resource is a member of the built wheel."""
    names = set(built_wheel.namelist())
    missing = [m for m in REQUIRED_WHEEL_MEMBERS if m not in names]
    assert not missing, (
        f"the built wheel is missing {missing}; check "
        "[tool.hatch.build.targets.wheel.force-include] in pyproject.toml. "
        f"Wheel contains {len(names)} members."
    )


def test_wheel_py_typed_is_empty(built_wheel: zipfile.ZipFile) -> None:
    """The shipped marker is an empty file, not a stray copy of something."""
    assert built_wheel.read("maddening/py.typed") == b""


def test_wheel_includes_package_modules(built_wheel: zipfile.ZipFile) -> None:
    """Guard the assertion above against a wheel that ships nothing at all.

    An empty or near-empty wheel would satisfy nothing, but a future
    build-configuration mistake could make the marker the *only* thing
    present; then `test_wheel_ships_force_included_data` would pass while
    the distribution was broken.
    """
    modules = [n for n in built_wheel.namelist()
               if n.startswith("maddening/") and n.endswith(".py")]
    assert len(modules) > 100, (
        f"only {len(modules)} Python modules in the wheel; the build is not "
        "packaging the source tree"
    )
    assert "maddening/__init__.py" in modules
