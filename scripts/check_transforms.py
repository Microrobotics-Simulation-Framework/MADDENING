#!/usr/bin/env python
"""Validate that all edge transforms in examples and tests use registered names.

Scans Python files under ``src/maddening/examples/`` for calls to
``add_edge(..., transform=...)`` and ``EdgeSpec(..., transform=...)``,
extracts the transform references, and verifies that any string
references resolve in the ``TransformRegistry``.

This is a CI gate for USD serialization compatibility -- transforms
referenced by string name must be importable and registered.

Exit codes:
    0 -- all referenced transforms are valid
    1 -- at least one unresolvable transform found
"""

import ast
import sys
from pathlib import Path


def _find_transform_string_refs(filepath: Path) -> list[tuple[int, str]]:
    """Find string literals used as transform= arguments in a file.

    Returns list of (line_number, string_value) pairs.
    """
    try:
        source = filepath.read_text()
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "transform" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        results.append((kw.value.lineno, kw.value.value))
    return results


def main():
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "maddening"

    # Import the registry to check against
    sys.path.insert(0, str(project_root / "src"))
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    # Scan production code (examples and core modules) for string transform refs.
    # Test files are excluded because they may intentionally use invalid names
    # for negative testing.
    scan_dirs = [
        src_dir / "examples",
        src_dir / "core",
        src_dir / "nodes",
    ]

    errors = []
    n_checked = 0

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for pyfile in scan_dir.rglob("*.py"):
            refs = _find_transform_string_refs(pyfile)
            for lineno, name in refs:
                n_checked += 1
                if name not in _TRANSFORM_REGISTRY:
                    rel = pyfile.relative_to(project_root)
                    errors.append(
                        f"  {rel}:{lineno}: transform '{name}' "
                        f"not found in TransformRegistry"
                    )

    if errors:
        print(f"FAIL: {len(errors)} unresolvable transform reference(s):")
        for err in errors:
            print(err)
        print(
            "\nFix: register each transform with "
            "@register_transform('name') from "
            "maddening.core.transforms"
        )
        sys.exit(1)

    print(
        f"OK: {n_checked} string transform reference(s) verified "
        f"({len(_TRANSFORM_REGISTRY)} transforms in registry)"
    )


if __name__ == "__main__":
    main()
