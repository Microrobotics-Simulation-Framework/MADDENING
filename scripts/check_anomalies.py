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


#: Reference fields whose entries this gate resolves.  Both count towards
#: the scope: a registry that declares neither verifies nothing.
#:
#: Measured after the guards below landed (2026-09-20), because the scope
#: guard was suspected of being conditional on ``notes`` and is not:
#: stripping every ``affected_components`` *and* every ``verification``
#: entry from the shipped registry exits 1, while stripping
#: ``affected_components`` alone exits 0 with 57 verification entries
#: still resolving -- so the run that passed genuinely verified something.
#: The residual gap is narrower than the suspicion: losing one *whole
#: kind* of reference leaves the gate green, because the sum is what is
#: guarded and not each field.  Recorded rather than fixed; a per-field
#: floor would be a ratchet, and this registry has entries for which one
#: of the two is legitimately absent.
_REFERENCE_FIELDS = ("affected_components", "verification")


def _census(path):
    """``(anomalies, declared references)`` in the registry, without resolving.

    Counting is separate from resolving so that the zero-scope guards below
    can run unconditionally -- including under ``--no-resolve``, where
    nothing is resolved but an empty registry is still an empty registry.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    anomalies = [a for a in (data.get("anomalies") or []) if isinstance(a, dict)]
    declared = 0
    for a in anomalies:
        for field in _REFERENCE_FIELDS:
            entries = a.get(field) or []
            if isinstance(entries, str):
                entries = [entries]
            declared += len(entries)
    return len(anomalies), declared


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

    notes: list[str] = []
    errors = validate_anomaly_registry(
        args.path,
        prefix=args.prefix,
        repo_root=repo_root,
        resolve_references=not args.no_resolve,
        notes=notes,
    )

    # A symbol in an optional subpackage this environment cannot import is
    # unverified, not broken -- but an unverified reference is a hole in the
    # evidence, so say so on stdout rather than passing in silence.
    for n in notes:
        print(f"NOTE: {n}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # ...and if *nothing* could be checked, the run proves nothing.  These
    # are check_heat_stability.py's two guards -- an empty scope, and a scope
    # where nothing was evaluable -- which this gate claimed in a comment to
    # already have and did not: the only guard here was inside ``if notes``,
    # and ``notes`` is populated solely by references skipped as unavailable,
    # so it is empty exactly when nothing was skipped.  A registry with no
    # anomalies, or with every reference stripped, entered no guard at all
    # and printed OK (audit_040_r2/gates, finding G7).
    n_anomalies, declared = _census(args.path)

    if n_anomalies == 0:
        print(
            f"FAIL: {args.path} declares no anomalies at all.\n"
            "A gate that verifies nothing cannot fail.  An empty registry is "
            "a truncated file or a wrong path far more often than it is a "
            "product with no known anomalies; if it really is empty, say so "
            "by pointing this gate somewhere else.",
            file=sys.stderr,
        )
        return 1

    if declared == 0:
        print(
            f"FAIL: the {n_anomalies} anomal(ies) in {args.path} declare no "
            f"affected_components and no verification entries between them.\n"
            "A gate that verifies nothing cannot fail.  The schema permits an "
            "anomaly with neither, but a whole registry with neither is not "
            "evidence of anything -- this run resolved zero references.",
            file=sys.stderr,
        )
        return 1

    if not args.no_resolve:
        # Every declared reference either resolved or was noted as
        # unavailable, because anything else is already an error above.
        verified = declared - len(notes)
        if verified == 0:
            print(
                f"FAIL: all {declared} reference(s) in {args.path} were "
                f"skipped as unavailable; this environment can verify none of "
                f"them.  Install the extras named above before trusting the "
                f"result.",
                file=sys.stderr,
            )
            return 1
        scope = (f"{n_anomalies} anomal(ies), {verified} reference(s) "
                 f"verified, {len(notes)} not checked")
    else:
        scope = (f"{n_anomalies} anomal(ies), {declared} reference(s) "
                 f"declared and NOT resolved (--no-resolve)")

    # Verified and declined are reported separately: a single headline count
    # that folds in the references the gate declined to check is how "50
    # citations verified" came to mean 45.
    print(f"OK: anomaly registry at {args.path} is valid ({scope})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
