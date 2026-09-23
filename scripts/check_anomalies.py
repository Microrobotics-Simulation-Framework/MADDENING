#!/usr/bin/env python3
"""CI script — the compliance validator plus the registry-level rules.

Usage:
    python scripts/check_anomalies.py [path] [--prefix PREFIX]
                                      [--repo-root DIR] [--no-resolve]

If no path is given, defaults to docs/validation/known_anomalies.yaml.
``--repo-root`` is the directory that ``verification:`` test paths are
resolved against; it defaults to the repository this script lives in.

``affected_versions`` convention
--------------------------------
Every entry's ``affected_versions`` is a PEP 440 version specifier set,
evaluated with ``packaging.specifiers.SpecifierSet`` (pre-releases
admitted) against the registry header's ``maddening_version`` -- the
version of the tree the registry ships in.  :func:`version_range_errors`
enforces it, and ``scripts/generate_soup_tables.py`` calls the same
function, so the SOUP package and this gate cannot disagree about it.

* The range is ``>=FIRST`` -- the first version that carried the defect --
  optionally followed by ``, <FIX`` -- the release that carries the fix.
  No other operator is accepted, and the range must admit ``FIRST``.
* A defect first introduced during a development cycle starts at that
  cycle's first development build: ``>=0.4.0.dev0``, never ``>=0.4.0``,
  which PEP 440 orders *after* every 0.4.0 pre-release and so excludes
  the builds that carry the defect.
* ``open``, ``partially_resolved``, ``wont_fix`` and any status nobody has
  enumerated leave the defect reachable: the range must admit
  ``maddening_version``, i.e. stay open-ended.
* ``resolved``: the range must NOT admit ``maddening_version`` (nor
  ``resolution_version`` when one is given).  ``<0.4.0`` excludes 0.4.0's
  own pre-releases under PEP 440, so a registry at ``0.4.0.dev0`` -- whose
  tree carries the fix -- is correctly outside ``>=0.1.0, <0.4.0``.
* ``none`` is the empty set: a defect introduced and fixed within one
  development cycle, carried by no tagged release and not by the tree the
  registry describes.  PEP 440 has no spelling for "0.4.0.dev0 builds
  before the fix", because every build of the cycle is ``0.4.0.dev0``;
  ``>=0.4.0.dev0, <0.4.0``, which such entries used to carry, admits no
  version at all and fails here.  The description says which builds had it.
* ``duplicate``: parsed, not compared; its range lives on the other entry.

An empty or missing range, an unparseable one, a missing or non-PEP 440
``maddening_version``, a registry with no anomalies and an unimportable
``packaging`` all fail: this check fails closed.  ``packaging`` is not a
runtime dependency of MADDENING; it is present wherever this gate runs
because pytest (``packaging>=22``) and matplotlib depend on it.
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

#: ``resolution_status`` values whose defect is gone from the version the
#: registry describes: the range must NOT admit ``maddening_version``.
_STATUSES_RESOLVED = frozenset({"resolved"})

#: A duplicate's range lives on the entry it duplicates.  It is still
#: parsed -- a garbage range fails -- but not compared with the version.
_STATUSES_NOT_COMPARED = frozenset({"duplicate"})

#: Every status whose defect cannot reach the version the registry
#: describes.  Everything else -- ``open``, ``partially_resolved``,
#: ``wont_fix`` and a status nobody has enumerated -- is *reachable*, so
#: an unrecognised spelling can never close a range by accident.
#: ``generate_soup_tables.py`` counts its "reachable in this version"
#: headline with this same set, so the headline and this gate cannot come
#: to disagree about which entries are live.
UNREACHABLE_STATUSES = _STATUSES_RESOLVED | _STATUSES_NOT_COMPARED

#: The ``affected_versions`` literal that spells the empty set.
EMPTY_RANGE = "none"

#: The only specifier operators the convention uses: ``>=FIRST`` and
#: ``<FIX``.  Anything else is refused rather than interpreted.
_RANGE_OPERATORS = (">=", "<")

_CONVENTION = ("see the affected_versions convention in the header of "
               "docs/validation/known_anomalies.yaml")


def _range_errors_for(aid, status, text, current, resolved_in, sp):
    """Every way one entry's ``affected_versions`` breaks the convention.

    ``sp`` is ``(SpecifierSet, InvalidSpecifier, Version, InvalidVersion)``
    from ``packaging``, passed in so the import happens -- and fails
    closed -- in exactly one place.
    """
    SpecifierSet, InvalidSpecifier, Version, InvalidVersion = sp

    if text == EMPTY_RANGE:
        def admits(_version):
            return False

        shown = f"'{EMPTY_RANGE}' (the empty set)"
    else:
        try:
            spec = SpecifierSet(text)
        except InvalidSpecifier as exc:
            return [f"{aid}: affected_versions {text!r} is not a PEP 440 "
                    f"version specifier set ({exc}); {_CONVENTION}"]
        by_op: dict = {}
        for s in spec:
            by_op.setdefault(s.operator, []).append(s)
        unknown = sorted(set(by_op) - set(_RANGE_OPERATORS))
        if unknown:
            return [f"{aid}: affected_versions {text!r} uses "
                    f"{', '.join(repr(o) for o in unknown)}; the convention "
                    f"is '>=FIRST' or '>=FIRST, <FIX' and nothing else; "
                    f"{_CONVENTION}"]
        if len(by_op.get(">=", ())) != 1 or len(by_op.get("<", ())) > 1:
            return [f"{aid}: affected_versions {text!r} must name exactly "
                    f"one '>=FIRST' and at most one '<FIX'; {_CONVENTION}"]
        first = Version(by_op[">="][0].version)
        if not spec.contains(first, prereleases=True):
            # ">=0.4.0.dev0, <0.4.0" lands here: PEP 440's "<0.4.0" also
            # excludes 0.4.0's pre-releases, so the set is empty.
            return [f"{aid}: affected_versions {text!r} does not admit "
                    f"{first}, its own first version, so it admits no "
                    f"version at all.  A defect introduced and fixed within "
                    f"one development cycle is written '{EMPTY_RANGE}'; "
                    f"{_CONVENTION}"]
        if first > current:
            return [f"{aid}: affected_versions {text!r} starts at {first}, "
                    f"after {current}, the version this registry describes.  "
                    f"A defect introduced during a development cycle starts "
                    f"at that cycle's first development build "
                    f"('>={Version(first.base_version)}.dev0'), which PEP 440 "
                    f"orders before every pre-release of {first.base_version}; "
                    f"{_CONVENTION}"]

        def admits(version):
            return spec.contains(version, prereleases=True)

        shown = repr(text)

    if status in _STATUSES_NOT_COMPARED:
        return []

    errors = []
    if status in _STATUSES_RESOLVED:
        if admits(current):
            errors.append(
                f"{aid}: resolution_status is {status!r}, but affected_versions "
                f"{shown} admits {current}, the version this registry "
                f"describes -- the range says the defect is still here.  Close "
                f"it at the release that carries the fix ('>=FIRST, <FIX'), or "
                f"reopen the entry; {_CONVENTION}")
        if resolved_in not in (None, ""):
            try:
                fixed = Version(str(resolved_in))
            except InvalidVersion:
                errors.append(f"{aid}: resolution_version {resolved_in!r} is "
                              f"not a PEP 440 version")
            else:
                if admits(fixed):
                    errors.append(
                        f"{aid}: resolution_version is {fixed}, but "
                        f"affected_versions {shown} admits {fixed}")
        return errors

    if not admits(current):
        errors.append(
            f"{aid}: resolution_status is {status!r}, which leaves the defect "
            f"reachable, but affected_versions {shown} does not admit "
            f"{current}, the version this registry describes -- a range that "
            f"says 'you are not affected' where the entry says you are.  Keep "
            f"the range open-ended ('>=FIRST') while the defect is reachable "
            f"and let the prose carry the condition; {_CONVENTION}")
    return errors


def version_range_errors(registry):
    """Check every ``affected_versions`` against the registry's own version.

    The one implementation of the convention in this module's docstring;
    ``scripts/generate_soup_tables.py --check`` calls it too.  It replaces
    a ``startswith(">=")`` test in the SOUP generator, which let
    ``">=0.1.0, <0.4.0"`` through on a ``partially_resolved`` entry
    (MADD-ANO-016 shipped that way: a range asserting 0.4.0 is unaffected
    beside a residual risk saying the 0.4.0 behaviour was never observed
    against a provider), and this gate, which never read the field at all.

    Returns a list of messages, each naming the entry it is about; empty
    means every range agrees with its entry's status.  Fails closed: an
    empty registry, a missing or non-PEP 440 ``maddening_version``, a
    missing or unparseable range, or an unimportable ``packaging`` is an
    error, never a pass.
    """
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        return [f"affected_versions cannot be checked: the 'packaging' "
                f"library is not importable ({exc}).  Install it; do not "
                f"skip the check."]
    sp = (SpecifierSet, InvalidSpecifier, Version, InvalidVersion)

    if not isinstance(registry, dict):
        return ["the registry is not a YAML mapping, so no affected_versions "
                "range could be checked"]
    anomalies = [a for a in (registry.get("anomalies") or [])
                 if isinstance(a, dict)]
    if not anomalies:
        return ["the registry declares no anomalies, so no affected_versions "
                "range was checked; a range check over nothing verifies "
                "nothing"]

    declared = registry.get("maddening_version")
    if declared in (None, ""):
        return ["the registry header has no maddening_version, so no "
                "affected_versions range can be compared with the version "
                "the registry describes"]
    try:
        current = Version(str(declared))
    except InvalidVersion:
        return [f"maddening_version {declared!r} is not a PEP 440 version, so "
                f"no affected_versions range can be compared with it"]

    errors = []
    for a in anomalies:
        aid = str(a.get("anomaly_id", "<missing>"))
        raw = a.get("affected_versions")
        if not isinstance(raw, str) or not raw.strip():
            # An empty string would parse as a SpecifierSet that admits
            # *everything*, so it is refused before it reaches the parser.
            errors.append(
                f"{aid}: affected_versions is {raw!r}; every entry records a "
                f"PEP 440 range, or '{EMPTY_RANGE}' for the empty set; "
                f"{_CONVENTION}")
            continue
        errors += _range_errors_for(
            aid, a.get("resolution_status"), raw.strip(), current,
            a.get("resolution_version"), sp,
        )
    return errors


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
    The parsed document is returned too, for the rules that read the header.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        data = {}
    anomalies = [a for a in (data.get("anomalies") or []) if isinstance(a, dict)]
    declared = 0
    for a in anomalies:
        for field in _REFERENCE_FIELDS:
            entries = a.get(field) or []
            if isinstance(entries, str):
                entries = [entries]
            declared += len(entries)
    return len(anomalies), declared, anomalies, data


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
    n_anomalies, declared, anomalies, data = _census(args.path)
    # Evidence rules run whether or not the schema passed and whether or
    # not references are resolved: a stripped verification list and a
    # deleted entry are both structural, and ``--no-resolve`` must not
    # switch them off.  The version-range rule is structural too.  It is
    # skipped only for an empty registry, which the zero-scope guard below
    # reports in its own words (the function itself fails closed on one,
    # for the SOUP generator's sake).
    errors = list(errors) + _evidence_errors(anomalies)
    if n_anomalies:
        errors += version_range_errors(data)

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
