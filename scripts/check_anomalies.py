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

#: A closed or half-closed entry is a claim that something now prevents
#: the defect; ``verification`` is where the claim names its evidence.
_STATUSES_THAT_MUST_CITE_EVIDENCE = ("resolved", "partially_resolved")


def _evidence_errors(anomalies):
    """Registry-level rules the schema validator does not enforce.

    Both fail closed, and both were found missing by the release audit of
    2026-09-22, which stripped MADD-ANO-018's entire ``verification`` list
    (status still ``resolved``) and then deleted the entry outright: the
    gate printed ``OK`` both times.

    1. An entry whose ``resolution_status`` says the defect is (partly)
       gone must cite at least one ``verification`` test.  The validator
       resolves every entry that IS there; it has no opinion on a list
       that is empty or missing, so a resolution could lose its evidence
       and stay green.
    2. Within each ID prefix the numbers run contiguously from 001 to the
       highest present.  Both registries are contiguous by construction
       (``tests/compliance/test_soup_evidence.py`` says why), so a gap is
       a deleted entry.  An entry in this list is IEC 62304 evidence: it
       is retired by recording the retirement, never by deletion.

    What this cannot see is the deletion of the *highest* entry: a pin on
    the high-water mark has to live outside the file it guards, or it is
    edited along with the deletion.  That pin is ``_HIGHEST_ANOMALY_ID``
    in ``tests/compliance/test_soup_evidence.py``, and this gate does not
    duplicate it.
    """
    import re

    errors = []
    by_prefix: dict = {}
    for a in anomalies:
        aid = str(a.get("anomaly_id", "<missing>"))
        status = a.get("resolution_status")
        if status in _STATUSES_THAT_MUST_CITE_EVIDENCE:
            entries = a.get("verification") or []
            if isinstance(entries, str):
                entries = [entries]
            if not entries:
                errors.append(
                    f"{aid}: resolution_status is {status!r} but the "
                    f"verification list is empty or missing.  A resolution "
                    f"without a named test is a claim, not evidence: cite the "
                    f"test that pins it (tests/<file>.py::<test>), or set the "
                    f"status back to 'open'."
                )
        m = re.fullmatch(r"(.*?)(\d+)", aid)
        if m:
            by_prefix.setdefault(m.group(1), {})[int(m.group(2))] = aid
    for prefix, numbers in sorted(by_prefix.items()):
        width = max(len(str(n)) for n in numbers)
        width = max(width, 3)
        highest = max(numbers)
        missing = sorted(set(range(1, highest + 1)) - set(numbers))
        if missing:
            names = [f"{prefix}{n:0{width}d}" for n in missing]
            errors.append(
                f"anomaly ID(s) missing from the contiguous range "
                f"{prefix}{1:0{width}d}..{prefix}{highest:0{width}d}: {names}.  "
                f"An entry in this registry is IEC 62304 evidence; restore it, "
                f"or record the retirement in the _RETIRED_* frozenset in "
                f"tests/compliance/test_soup_evidence.py -- never delete it "
                f"and never reuse the number."
            )
    return errors


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
    return len(anomalies), declared, anomalies


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
    n_anomalies, declared, anomalies = _census(args.path)
    # Evidence rules run whether or not the schema passed and whether or
    # not references are resolved: a stripped verification list and a
    # deleted entry are both structural, and ``--no-resolve`` must not
    # switch them off.
    errors = list(errors) + _evidence_errors(anomalies)

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
