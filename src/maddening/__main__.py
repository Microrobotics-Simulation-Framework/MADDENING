"""``python -m maddening <command>``: MADDENING's command line.

Commands
--------
info
    Print the version and environment report (:func:`maddening.show_versions`):
    MADDENING, Python, JAX and dependency versions, the JAX backend and
    devices, x64, allowlisted environment variables and installed extras.
    ``--json`` prints it as JSON.

Adding a command: write ``_cmd_<name>(args) -> int`` and register it in
:func:`build_parser` with ``set_defaults(func=...)``.  Each command
imports what it needs inside its function, so ``python -m maddening
--help`` stays fast and works on an installation whose optional (or
even core) dependencies are broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def _cmd_info(args: argparse.Namespace) -> int:
    from maddening.info import collect_versions, show_versions  # noqa: PLC0415
    if args.json:
        sys.stdout.write(json.dumps(collect_versions(), indent=2, sort_keys=True,
                                    default=str) + "\n")
    else:
        show_versions()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, one subcommand per command."""
    parser = argparse.ArgumentParser(
        prog="python -m maddening",
        description="MADDENING command-line tools.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    commands.required = True
    info = commands.add_parser(
        "info", help="print version and environment information for bug reports",
        description="Print MADDENING's version and environment report "
                    "(maddening.show_versions()). No secret is printed: only an "
                    "allowlist of JAX/XLA/MADDENING environment variables is read.",
    )
    info.add_argument("--json", action="store_true",
                      help="print the report as JSON instead of text")
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
