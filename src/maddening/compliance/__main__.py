"""
Command-line interface for MADDENING compliance tooling.

Usage::

    python -m maddening.compliance check-anomalies <path> [--prefix PREFIX]

Examples::

    # Validate MADDENING's own registry
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml

    # Validate MIME's registry with namespace enforcement
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml --prefix MIME-ANO-
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m maddening.compliance",
        description="MADDENING compliance tooling CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # check-anomalies
    ca = sub.add_parser(
        "check-anomalies",
        help="Validate a known_anomalies.yaml file",
    )
    ca.add_argument("path", help="Path to the YAML file to validate")
    ca.add_argument(
        "--prefix",
        default="",
        help="Expected anomaly ID prefix (e.g., 'MIME-ANO-')",
    )

    args = parser.parse_args()

    if args.command == "check-anomalies":
        from maddening.compliance._validate import validate_anomaly_registry
        errors = validate_anomaly_registry(args.path, prefix=args.prefix)
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print("OK: anomaly registry is valid")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
