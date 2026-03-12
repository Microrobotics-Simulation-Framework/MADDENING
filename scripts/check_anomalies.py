#!/usr/bin/env python3
"""CI script — thin wrapper around the compliance validator.

Usage:
    python scripts/check_anomalies.py [path] [--prefix PREFIX]

If no path is given, defaults to docs/validation/known_anomalies.yaml.
"""

import sys
import os

# Add src to path so this works without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from maddening.compliance._validate import validate_anomaly_registry


def main():
    path = "docs/validation/known_anomalies.yaml"
    prefix = ""

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--prefix" and i + 1 < len(args):
            prefix = args[i + 1]
            i += 2
        elif not args[i].startswith("--"):
            path = args[i]
            i += 1
        else:
            i += 1

    errors = validate_anomaly_registry(path, prefix=prefix)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: anomaly registry at {path} is valid")


if __name__ == "__main__":
    main()
