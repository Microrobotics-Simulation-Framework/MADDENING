#!/usr/bin/env python3
"""CI script — thin wrapper around the compliance validator.

Usage:
    python scripts/check_anomalies.py [path] [--prefix PREFIX]
                                      [--repo-root DIR] [--no-resolve]

If no path is given, defaults to docs/validation/known_anomalies.yaml.
``--repo-root`` is the directory that ``verification:`` test paths are
resolved against; it defaults to the repository this script lives in.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# Add src to path so this works without installation
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from maddening.compliance._validate import validate_anomaly_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join("docs", "validation", "known_anomalies.yaml"),
        help="Path to the known_anomalies.yaml to validate",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Required anomaly ID prefix (e.g. 'MADD-ANO-')",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Root that the verification: test paths are resolved against "
             "(default: this script's repository)",
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="Schema check only: do not resolve affected_components symbols "
             "or verification test paths",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root
    if repo_root is None and os.path.abspath(args.path).startswith(_REPO_ROOT):
        repo_root = _REPO_ROOT

    errors = validate_anomaly_registry(
        args.path,
        prefix=args.prefix,
        repo_root=repo_root,
        resolve_references=not args.no_resolve,
    )
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"OK: anomaly registry at {args.path} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
