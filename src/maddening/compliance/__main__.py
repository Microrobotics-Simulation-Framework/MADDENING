"""
Command-line interface for MADDENING compliance tooling.

Usage::

    python -m maddening.compliance check-anomalies <path> [--prefix PREFIX]
        [--version-field NAME]

A registry's top level carries ``schema_version``, ``generated_date``,
``anomalies`` and exactly one project version field: a key named
``<project>_version``, the whole key matching ``[a-z][a-z0-9_]*_version`` and
not ``schema_version`` (``maddening_version``, ``my_lib_version``), whose
value is a non-empty string.  ``--version-field NAME`` requires that key to
be exactly ``NAME``; without it any one such key is accepted.

Exit status: 0 when the registry is valid, 1 when it is not (each error on
stderr), 2 on a usage error (including a ``--version-field`` that is not
itself a project version field name).

Examples::

    # Validate MADDENING's own registry
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml \\
        --version-field maddening_version

    # Validate MIME's registry with namespace enforcement
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml \\
        --prefix MIME-ANO- --version-field mime_version
"""

import argparse
import sys


def _version_field_name(value: str) -> str:
    """argparse ``type=`` for ``--version-field``: refuse a malformed name."""
    from maddening.compliance._validate import _is_version_field_name

    if not _is_version_field_name(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a project version field name: it must be "
            f"'<project>_version' with <project> matching [a-z][a-z0-9_]*, "
            f"and not 'schema_version'"
        )
    return value


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
        description=(
            "Validate a known_anomalies.yaml file.  Its top level carries "
            "schema_version, generated_date, anomalies and exactly one "
            "project version field: a key named '<project>_version' "
            "(<project> matching [a-z][a-z0-9_]*, not 'schema_version'), "
            "whose value is a non-empty string.  Exit status 0: valid; "
            "1: invalid; 2: usage error."
        ),
    )
    ca.add_argument("path", help="Path to the YAML file to validate")
    ca.add_argument(
        "--prefix",
        default="",
        help="Expected anomaly ID prefix (e.g., 'MIME-ANO-')",
    )
    ca.add_argument(
        "--version-field",
        default=None,
        metavar="NAME",
        type=_version_field_name,
        help=(
            "Require the project version field to be exactly NAME "
            "(e.g., 'mime_version').  Default: any one key named "
            "'<project>_version' is accepted."
        ),
    )

    args = parser.parse_args()

    if args.command == "check-anomalies":
        from maddening.compliance._validate import validate_anomaly_registry
        errors = validate_anomaly_registry(
            args.path, prefix=args.prefix, version_field=args.version_field,
        )
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
