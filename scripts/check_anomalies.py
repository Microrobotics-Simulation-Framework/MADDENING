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
* ``FIRST`` and ``FIX`` are versions MADDENING has actually released, or
  the current cycle's own versions: ``FIRST`` may also be the first
  development build of a released or the current cycle (``X.Y.Z.dev0``),
  and ``FIX`` the release the current cycle is building towards
  (``0.4.0`` while ``maddening_version`` is ``0.4.0.dev0``).  ``>=0.1.5``,
  ``<0.3.7`` and ``>=0.3.1.post1`` name versions nobody could install.
* ``resolution_version``, where given, is such a release too, and a
  ``resolved`` entry must give it: it is the version whose fix the entry
  claims.  A ``resolved`` range's ``<FIX`` must *be* that version --
  ``>=0.1.0, <0.2.0`` on an entry fixed in 0.4.0 says 0.2.0 to 0.3.1 were
  never affected, and the SOUP table printed it as harmless drift.

The released versions are the dated section headings of ``CHANGELOG.md``
(``## [0.3.1] - 2026-06-22``), read by :func:`released_versions`.  Not
``git tag``: CI's compliance job checks out without tags, and a rule that
silently loosened there would verify less on the one machine whose result
is cited.  ``tests/compliance/test_gate_scripts.py`` holds the headings to
the ``v*`` tags wherever the tags are present.

An empty or missing range, an unparseable one, a missing or non-PEP 440
``maddening_version``, a registry with no anomalies and an unimportable
``packaging`` all fail: this check fails closed.  ``packaging`` is not a
runtime dependency of MADDENING; it is present wherever this gate runs
because pytest (``packaging>=22``) and matplotlib depend on it.
"""

import argparse
import ast
import os
import re
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

#: The release record the released versions are read from.
CHANGELOG = os.path.join(_REPO_ROOT, "CHANGELOG.md")

#: A released version's section heading: ``## [0.3.1] - 2026-06-22``.
#: ``## [Unreleased]`` carries no date and is not a release.
_RELEASE_HEADING = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$", re.M)


def released_versions(changelog=CHANGELOG):
    """The versions MADDENING has released, from ``CHANGELOG.md``.

    Returns ``(versions, error)``: the version strings of every dated
    ``## [X.Y.Z] - YYYY-MM-DD`` heading, newest first as the file lists
    them, and ``None`` -- or ``((), message)`` when the file cannot be read
    or holds no dated heading, which callers must treat as a failure: an
    empty release list would refuse every range, and a missing one must
    not be read as "anything goes".
    """
    try:
        with open(changelog, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return (), (f"the released versions cannot be read from {changelog} "
                    f"({exc}), so no affected_versions bound can be checked "
                    f"against them")
    found = tuple(m.group("version") for m in _RELEASE_HEADING.finditer(text))
    if not found:
        return (), (f"{changelog} has no dated '## [X.Y.Z] - YYYY-MM-DD' "
                    f"release heading, so no affected_versions bound can be "
                    f"checked against a released version")
    return found, None


def _range_errors_for(aid, status, text, current, resolved_in, sp, released):
    """Every way one entry's ``affected_versions`` breaks the convention.

    ``sp`` is ``(SpecifierSet, InvalidSpecifier, Version, InvalidVersion)``
    from ``packaging``, passed in so the import happens -- and fails
    closed -- in exactly one place.  ``released`` is the set of released
    :class:`~packaging.version.Version` objects (:func:`released_versions`).
    """
    SpecifierSet, InvalidSpecifier, Version, InvalidVersion = sp

    # The versions a bound may name.  FIX: a release, or the one this cycle
    # is building towards.  FIRST: those, or the first development build of
    # any of them -- a defect introduced during a cycle starts at X.Y.Z.dev0,
    # and keeps that spelling after X.Y.Z ships.
    cycle = Version(current.base_version)
    fix_ok = frozenset(released) | {cycle}
    first_ok = fix_ok | {Version(f"{v.base_version}.dev0") for v in fix_ok}
    listed = ", ".join(str(v) for v in sorted(released))

    def not_a_release(role, version, allowed):
        return (f"{aid}: {role} {version} is not a version MADDENING released "
                f"({listed}, from CHANGELOG.md's dated headings) nor {allowed}; "
                f"a bound names a version somebody could install; "
                f"{_CONVENTION}")

    fixed = None
    errors = []
    if resolved_in not in (None, ""):
        try:
            fixed = Version(str(resolved_in))
        except InvalidVersion:
            return [f"{aid}: resolution_version {resolved_in!r} is not a PEP "
                    f"440 version"]
        if fixed not in fix_ok:
            errors.append(not_a_release(
                "resolution_version", fixed,
                f"{cycle}, the release this cycle is building towards"))

    fix = None
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
        # Reported beside, not instead of, the status rules below: a bound
        # that names no release is one defect, and the range can still
        # disagree with its status as well.
        if first not in first_ok:
            errors.append(not_a_release(
                "affected_versions FIRST", first,
                f"the first development build of one or of this cycle "
                f"('X.Y.Z.dev0', e.g. '{cycle}.dev0')"))
        if by_op.get("<"):
            fix = Version(by_op["<"][0].version)
            if fix not in fix_ok:
                errors.append(not_a_release(
                    "affected_versions FIX", fix,
                    f"{cycle}, the release this cycle is building towards"))

        def admits(version):
            return spec.contains(version, prereleases=True)

        shown = repr(text)

    if status in _STATUSES_NOT_COMPARED:
        return errors

    if status in _STATUSES_RESOLVED:
        if admits(current):
            errors.append(
                f"{aid}: resolution_status is {status!r}, but affected_versions "
                f"{shown} admits {current}, the version this registry "
                f"describes -- the range says the defect is still here.  Close "
                f"it at the release that carries the fix ('>=FIRST, <FIX'), or "
                f"reopen the entry; {_CONVENTION}")
        if fixed is None:
            errors.append(
                f"{aid}: resolution_status is {status!r} but no "
                f"resolution_version says which release carries the fix, so "
                f"the range's '<FIX' cannot be checked against it; add it")
        else:
            if admits(fixed):
                errors.append(
                    f"{aid}: resolution_version is {fixed}, but "
                    f"affected_versions {shown} admits {fixed}")
            if fix is not None and fix != fixed:
                errors.append(
                    f"{aid}: affected_versions {shown} closes at {fix}, but "
                    f"resolution_version says the fix is in {fixed}.  '<FIX' "
                    f"is the release that carries the fix, so the two are one "
                    f"version; a range closing early says the releases in "
                    f"between were never affected; {_CONVENTION}")
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


def version_range_errors(registry, released=None):
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

    ``released`` is the list of version strings MADDENING has released;
    ``None`` reads it from ``CHANGELOG.md`` (:func:`released_versions`),
    which is what both callers do.  Tests pass it to describe a registry
    at another point in the project's history.  An unreadable or empty
    list fails closed.
    """
    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        return [f"affected_versions cannot be checked: the 'packaging' "
                f"library is not importable ({exc}).  Install it; do not "
                f"skip the check."]
    sp = (SpecifierSet, InvalidSpecifier, Version, InvalidVersion)

    if released is None:
        released, problem = released_versions()
        if problem:
            return [problem]
    if not released:
        return ["no released versions were given, so no affected_versions "
                "bound can be checked against a released version"]
    try:
        released_set = frozenset(Version(str(v)) for v in released)
    except InvalidVersion as exc:
        return [f"a released version is not a PEP 440 version ({exc}); fix "
                f"the CHANGELOG.md heading it came from"]

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
            a.get("resolution_version"), sp, released_set,
        )
    return errors


#: Where a retired ID is recorded -- the file the gap error below names.
#: The pin lives outside the registry it guards, next to the high-water
#: mark (``_HIGHEST_ANOMALY_ID``), which is why this gate reads a test
#: module rather than a field of the YAML.
_RETIRED_IDS_FILE = os.path.join("tests", "compliance", "test_soup_evidence.py")
_RETIRED_IDS_NAME = "_RETIRED_ANOMALY_IDS"

#: A ``partially_resolved`` entry says part of the defect is gone and part
#: is not; ``residual_risk`` is where it says which part is left.  Without
#: it the SOUP table shows "partially resolved" with nothing a reader can
#: act on (audit_040_p4_2, A9: deleting MADD-ANO-014's passed).
_STATUSES_THAT_MUST_STATE_RESIDUAL_RISK = ("partially_resolved",)


def retired_anomaly_ids(repo_root):
    """``(retired, error)``: the retired anomaly IDs recorded under ``repo_root``.

    Reads ``_RETIRED_ANOMALY_IDS`` from ``tests/compliance/
    test_soup_evidence.py`` without importing it (a test module imports
    pytest and the package).  It is a literal ``{id: reason}`` dict --
    ``{}``, or ``{"MADD-ANO-NNN": "why it was retired", ...}`` -- and
    ``retired`` is that dict.  Only a literal is accepted, because the gate
    must be able to say exactly which IDs are excused, and every ID carries
    its reason, because a retirement is a sign-off act and a bare ID in a
    set said nothing about why a number went unused (audit_040_p4_2, A2).

    Fails closed: a missing file or a missing assignment excuses nothing
    (``retired`` is empty).  An assignment that is present but is not such
    a dict -- the set form this took until 0.4.0 included -- is returned as
    ``error`` and excuses nothing either, and so is an ID whose reason is
    missing or blank.
    """
    import ast

    if not repo_root:
        return {}, None
    path = os.path.join(repo_root, _RETIRED_IDS_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError):
        return {}, None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if not any(getattr(t, "id", None) == _RETIRED_IDS_NAME for t in targets):
            continue
        bad = (f"{_RETIRED_IDS_FILE}: {_RETIRED_IDS_NAME} is not a literal "
               f"{{id: reason}} dict ({{}} or {{\"MADD-ANO-NNN\": \"why it was "
               f"retired\", ...}}), so no gap can be excused by it")
        try:
            retired = ast.literal_eval(value) if value is not None else None
        except (ValueError, TypeError, SyntaxError):
            return {}, bad
        if not isinstance(retired, dict) or not all(
                isinstance(k, str) for k in retired):
            return {}, bad
        blank = sorted(k for k, v in retired.items()
                       if not isinstance(v, str) or not v.strip())
        if blank:
            return {}, (f"{_RETIRED_IDS_FILE}: {_RETIRED_IDS_NAME} records "
                        f"{blank} with no reason.  A retirement says why the "
                        f"number went unused; until every ID carries one, "
                        f"none is excused")
        return retired, None
    return {}, None


def _git(cwd, *args):
    """``(returncode, stdout)`` of one git command, or ``(None, why)``."""
    import subprocess

    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return done.returncode, done.stdout if done.returncode == 0 else done.stderr


def _entry_in(text, aid):
    """The registry entry ``aid`` in the YAML ``text``, or ``None``."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    try:
        data = yaml.load(text, Loader=loader) or {}
    except yaml.YAMLError:
        return None
    for a in (data.get("anomalies") or []) if isinstance(data, dict) else []:
        if isinstance(a, dict) and str(a.get("anomaly_id")) == aid:
            return a
    return None


def last_committed_entry(registry_path, aid):
    """``(entry, where, error)``: ``aid`` as the registry's history last records it.

    A retired entry is gone from the tree, so the only record of what it
    said is git.  ``entry`` is the entry as the newest commit that holds
    it records it: ``HEAD``'s own copy when the deletion is not committed
    yet, otherwise the copy in the parent of the commit that removed it.
    ``entry`` is ``None`` with no error when the registry's history never
    held the ID at all -- a number allocated and withdrawn before any
    commit recorded it.  ``where`` names the commit the entry was read
    from.

    Fails closed: a registry outside a git work tree, one git does not
    track, a shallow clone (whose history may stop before the removal) and
    a history that names the ID but shows no commit removing it are each
    an ``error``, never "no such entry".
    """
    path = os.path.abspath(registry_path)
    cwd = os.path.dirname(path)
    rc, top = _git(cwd, "rev-parse", "--show-toplevel")
    if rc != 0:
        return None, None, (f"{registry_path} is not in a git work tree "
                            f"({str(top).strip()}), so what the retired entry "
                            f"last said cannot be read from its history")
    # Every command below runs at the top of the work tree: a pathspec is
    # read relative to the working directory, so from docs/validation the
    # registry's own repository-relative path named nothing, and every
    # retirement read "not tracked" (found by re-running the audit's
    # mutation harness, whose registry sits where the real one does).
    cwd = top.strip()
    rel = os.path.relpath(path, cwd).replace(os.sep, "/")
    rc, shallow = _git(cwd, "rev-parse", "--is-shallow-repository")
    if rc != 0 or shallow.strip() != "false":
        return None, None, ("this is a shallow clone, so the commit that "
                            "removed the retired entry may be outside the "
                            "history it holds; run the gate in a full clone "
                            "(actions/checkout with fetch-depth: 0)")
    rc, out = _git(cwd, "ls-files", "--error-unmatch", "--", rel)
    if rc != 0:
        return None, None, (f"{rel} is not tracked by git, so its history "
                            f"cannot say what the retired entry last said")
    rc, head = _git(cwd, "show", f"HEAD:{rel}")
    if rc == 0:
        entry = _entry_in(head, aid)
        if entry is not None:
            return entry, "HEAD", None
    rc, log = _git(cwd, "log", "--format=%H", "-S", f'anomaly_id: "{aid}"',
                   "--", rel)
    if rc != 0:
        return None, None, f"git log over {rel} failed: {str(log).strip()}"
    commits = log.split()
    if not commits:
        return None, None, None
    for commit in commits:
        rc, before = _git(cwd, "show", f"{commit}^:{rel}")
        if rc != 0:
            continue
        rc, after = _git(cwd, "show", f"{commit}:{rel}")
        after_entry = _entry_in(after, aid) if rc == 0 else None
        before_entry = _entry_in(before, aid)
        if before_entry is not None and after_entry is None:
            return before_entry, f"{commit[:12]}^", None
    return None, None, (f"the history of {rel} names {aid} "
                        f"({len(commits)} commit(s)) but no commit removing "
                        f"its entry could be found")


def retirement_errors(retired, registry_path, present=()):
    """Refuse the retirement of an entry that was still reachable.

    Retiring an ID takes it out of the registry, and so out of the SOUP
    package's list of what a user of this version is exposed to.  That is
    only honest for a defect nobody is exposed to: an entry whose last
    committed ``resolution_status`` is ``resolved`` or ``duplicate`` (the
    statuses :data:`UNREACHABLE_STATUSES` names), or a number no commit
    ever recorded.  An ``open`` or ``partially_resolved`` entry -- or any
    status nobody enumerated -- is closed, and stays in the registry as
    evidence; it is never retired.  Deleting MADD-ANO-035 (open) and
    recording it retired passed until this rule (audit_040_p4_2, A2).

    The status is read from git (:func:`last_committed_entry`), because the
    tree no longer holds the entry; where git cannot answer, the
    retirement is refused rather than trusted.  An ID still in the
    registry (``present``) is skipped: :func:`_evidence_errors` already
    refuses it, as a retirement that never happened.
    """
    errors = []
    for aid in sorted(set(retired) - set(present)):
        entry, where, problem = last_committed_entry(registry_path, aid)
        if problem:
            errors.append(
                f"{aid} is recorded as retired in {_RETIRED_IDS_FILE}, but "
                f"whether it was still reachable cannot be established: "
                f"{problem}.  A retirement that cannot be checked excuses "
                f"nothing.")
            continue
        if entry is None:
            continue
        status = entry.get("resolution_status")
        if status not in UNREACHABLE_STATUSES:
            errors.append(
                f"{aid} is recorded as retired in {_RETIRED_IDS_FILE}, but "
                f"its last committed entry ({where}) is {status!r}, which "
                f"leaves the defect reachable.  A reachable anomaly is never "
                f"retired: close it -- 'resolved' with its evidence, or "
                f"'duplicate' of the entry that carries it -- and keep it in "
                f"the registry.")
    return errors


def _evidence_errors(anomalies, retired=frozenset()):
    """Registry-level rules the schema validator does not enforce.

    Both fail closed, and both were found missing by the release audit of
    2026-09-22, which stripped MADD-ANO-018's entire ``verification`` list
    (status still ``resolved``) and then deleted the entry outright: the
    gate printed ``OK`` both times.

    1. An entry whose ``resolution_status`` says the defect is (partly)
       gone must cite at least one ``verification`` test.  The validator
       resolves every entry that IS there; it has no opinion on a list
       that is empty or missing, so a resolution could lose its evidence
       and stay green.  A ``partially_resolved`` entry must also say, in
       ``residual_risk``, what is still reachable.
    2. Within each ID prefix the numbers run contiguously from 001 to the
       highest present, except for IDs recorded as retired (``retired``,
       read by :func:`retired_anomaly_ids`; a retirement of a reachable
       entry is refused separately, by :func:`retirement_errors`).  Both
       registries are
       contiguous by construction (``tests/compliance/test_soup_evidence.py``
       says why), so an unexplained gap is a deleted entry.  An entry in
       this list is IEC 62304 evidence: it is retired by recording the
       retirement, never by deletion.  Until 0.4.0 this rule ignored the
       record its own message pointed to, so a genuinely retired ID could
       never pass.  A retired ID that is still in the registry is refused
       too: the retirement never happened.

    What this cannot see is the deletion of the *highest* entry: a pin on
    the high-water mark has to live outside the file it guards, or it is
    edited along with the deletion.  That pin is ``_HIGHEST_ANOMALY_ID``
    in ``tests/compliance/test_soup_evidence.py``, and this gate does not
    duplicate it.
    """
    import re

    retired = frozenset(retired)
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
        if status in _STATUSES_THAT_MUST_STATE_RESIDUAL_RISK:
            residual = a.get("residual_risk")
            if not isinstance(residual, str) or not residual.strip():
                errors.append(
                    f"{aid}: resolution_status is {status!r} but "
                    f"residual_risk is empty or missing.  A partial "
                    f"resolution says which part of the defect is still "
                    f"reachable, and residual_risk is where a reader looks "
                    f"for it: state it, or set the status to 'open' or "
                    f"'resolved'."
                )
        m = re.fullmatch(r"(.*?)(\d+)", aid)
        if m:
            by_prefix.setdefault(m.group(1), {})[int(m.group(2))] = aid
    present = {aid for numbers in by_prefix.values() for aid in numbers.values()}
    for rid in sorted(retired & present):
        errors.append(
            f"{rid} is recorded as retired in {_RETIRED_IDS_FILE} "
            f"({_RETIRED_IDS_NAME}) but is still in the registry: the "
            f"retirement never happened.  Remove one or the other."
        )
    for prefix, numbers in sorted(by_prefix.items()):
        width = max(len(str(n)) for n in numbers)
        width = max(width, 3)
        highest = max(numbers)
        excused = set()
        for rid in retired:
            m = re.fullmatch(re.escape(prefix) + r"(\d+)", rid)
            if m:
                excused.add(int(m.group(1)))
        missing = sorted(set(range(1, highest + 1)) - set(numbers) - excused)
        if missing:
            names = [f"{prefix}{n:0{width}d}" for n in missing]
            errors.append(
                f"anomaly ID(s) missing from the contiguous range "
                f"{prefix}{1:0{width}d}..{prefix}{highest:0{width}d}: {names}.  "
                f"An entry in this registry is IEC 62304 evidence; restore it, "
                f"or record the retirement in _RETIRED_ANOMALY_IDS in "
                f"tests/compliance/test_soup_evidence.py -- never delete it "
                f"and never reuse the number."
            )
    return errors


#: pytest's default collection rules.  ``pyproject.toml`` overrides none of
#: ``python_files`` / ``python_classes`` / ``python_functions`` (a test in
#: ``tests/compliance/test_gate_scripts.py`` pins that), so these are the
#: rules the suite is collected by.
_TEST_FILE = re.compile(r"(test_.*|.*_test)\.py")
_TEST_CLASS_PREFIX = "Test"
_TEST_FUNCTION_PREFIX = "test"


#: Test paths that no CI lane collects: every pytest run over ``tests/`` in
#: ``.github/workflows`` passes ``--ignore=tests/viz``, and no job targets
#: the directory itself.  A test there runs on nobody's machine but the
#: author's, so it evidences nothing the release can point to
#: (audit_040_p4_1, A23).  Repository-relative POSIX paths.  Kept in step
#: with the workflows by ``tests/compliance/test_gate_scripts.py``, which
#: derives the same set from them and fails when the two differ.
PATHS_CI_NEVER_RUNS = ("tests/viz",)


def _ci_never_runs(rel):
    """The entry of :data:`PATHS_CI_NEVER_RUNS` holding ``rel``, or ``None``."""
    norm = os.path.normpath(rel).replace(os.sep, "/")
    for path in PATHS_CI_NEVER_RUNS:
        if norm == path or norm.startswith(path.rstrip("/") + "/"):
            return path
    return None


def _pytest_aliases(tree):
    """Module-level names that may stand for something from pytest.

    ``import pytest as pt`` binds ``pt`` to ``"pytest"``; ``from pytest
    import mark, skip as sk`` binds ``mark`` / ``sk`` to ``"pytest.mark"``
    / ``"pytest.skip"``; any other module-level ``NAME = value`` binds
    ``NAME`` to the ``value`` expression, so ``_skip =
    pytest.mark.skip(reason=...)`` followed by ``@_skip`` is read as the
    mark it is.  Matching the decorator's text alone (``endswith(
    "mark.skip")``) passed that spelling (audit_040_p4_1, A22).  A later
    binding replaces an earlier one, as it does at run time.
    """
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and alias.name.split(".")[0] == "pytest":
                    aliases[alias.asname] = alias.name
        elif (isinstance(node, ast.ImportFrom) and not node.level
              and (node.module or "").split(".")[0] == "pytest"):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            aliases[node.targets[0].id] = node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
              and isinstance(node.target, ast.Name)):
            aliases[node.target.id] = node.value
    return aliases


def _dotted(expr, aliases, depth=0):
    """``expr``'s dotted name with module-level aliases resolved, or ``None``."""
    if depth > 20:                       # a binding cycle: ``a = b; b = a``
        return None
    if isinstance(expr, ast.Name):
        bound = aliases.get(expr.id)
        if isinstance(bound, str):
            return bound
        if isinstance(bound, (ast.Name, ast.Attribute)):
            return _dotted(bound, aliases, depth + 1)
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value, aliases, depth + 1)
        return None if base is None else f"{base}.{expr.attr}"
    return None


#: Node types a condition may be built from for :func:`_static_truth` to
#: evaluate it: literals and the operators that combine them, and nothing
#: that can name, call or compute (no ``BinOp`` -- ``9 ** 9 ** 9``).
_CONSTANT_CONDITION_NODES = (
    ast.Constant, ast.Tuple, ast.List, ast.Load, ast.UnaryOp, ast.Not,
    ast.UAdd, ast.USub, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is,
    ast.IsNot,
)


def _static_truth(expr):
    """``True`` / ``False`` when a condition's value is fixed in the source.

    ``None`` when it depends on anything at all -- a name, a call, an
    attribute -- which is every condition that means something.  pytest
    evaluates a *string* condition as Python, so a string is parsed and
    judged the same way: ``skipif("True")`` is ``skipif(True)``.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        try:
            expr = ast.parse(expr.value.strip(), mode="eval").body
        except SyntaxError:
            return None
    if not all(isinstance(n, _CONSTANT_CONDITION_NODES) for n in ast.walk(expr)):
        return None
    try:
        code = compile(ast.fix_missing_locations(ast.Expression(body=expr)),
                       "<condition>", "eval")
        return bool(eval(code, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _mark_kinds(expr, aliases=None, depth=0):
    """``"skip"`` / ``"skipif"`` / ``"xfail"`` for each pytest mark in ``expr``.

    ``expr`` is a decorator or a ``pytestmark`` value (one mark or a list).
    Names bound at module level are resolved first (:func:`_pytest_aliases`).

    A ``skipif`` whose condition is fixed in the source is not conditional:
    ``skipif(True)`` skips on every machine and is reported as ``"skip"``,
    ``skipif(False)`` never skips and is not reported at all; with no
    condition, pytest skips unconditionally.  ``xfail`` conditions are read
    the same way.  A condition that depends on anything is ``"skipif"``:
    the gate cannot know the machine CI runs on, so it reports rather than
    judges (audit_040_p4_1, A20).
    """
    aliases = aliases or {}
    if depth > 20:
        return []
    if isinstance(expr, (ast.List, ast.Tuple)):
        return [k for item in expr.elts
                for k in _mark_kinds(item, aliases, depth + 1)]
    call = expr if isinstance(expr, ast.Call) else None
    target = call.func if call is not None else expr
    if (call is None and isinstance(target, ast.Name)
            and isinstance(aliases.get(target.id), (ast.Call, ast.List, ast.Tuple))):
        # ``_skip = pytest.mark.skip(reason=...)`` applied as ``@_skip``.
        return _mark_kinds(aliases[target.id], aliases, depth + 1)
    parts = (_dotted(target, aliases) or "").split(".")
    kind = parts[-1] if len(parts) >= 2 and parts[-2] == "mark" else None
    if kind == "skip":
        return ["skip"]
    if kind not in ("skipif", "xfail"):
        return []
    conditions = list(call.args) if call is not None else []
    if call is not None:
        conditions += [kw.value for kw in call.keywords if kw.arg == "condition"]
    truths = [_static_truth(c) for c in conditions]
    if not conditions or any(t is True for t in truths):
        return ["skip" if kind == "skipif" else "xfail"]
    if all(t is False for t in truths):
        return []
    return [kind]


#: Calls that end a test where they run, by their resolved dotted name.
_IMPERATIVE_OUTCOMES = {
    "pytest.skip": "skip",
    "pytest.skip.Exception": "skip",
    "pytest.xfail": "xfail",
    "pytest.xfail.Exception": "xfail",
    "pytest.importorskip": "importorskip",
}


def _imperative_kinds(stmts, aliases, helpers, seen=frozenset()):
    """Skip / xfail outcomes that running ``stmts`` can reach imperatively.

    A mark is not the only way a test stops: ``pytest.skip()`` or
    ``pytest.xfail()`` as a test's first statement makes it skip or xfail
    on every run, and the gate read decorators only, so both passed with
    pytest reporting the cited test skipped / xfailed (audit_040_p4_1,
    A19 / A21).  So the statements are walked in execution order:

    * a ``pytest.skip()`` that every run reaches -- straight-line code,
      not under an ``if``, a loop, an ``except``, a short-circuit or a
      conditional expression -- is ``"skip"``; one under a condition is
      ``"skipif"`` (an ``if`` whose test is fixed in the source counts as
      the branch it always takes);
    * ``pytest.importorskip()`` is ``"skipif"`` wherever it is: it skips
      only where the import fails;
    * ``pytest.xfail()`` is ``"xfail"`` wherever it is, as a conditional
      ``xfail`` mark is;
    * a module-level function the statements call (``helpers``) is
      followed, once, so ``_require_devices()`` is read like its body.

    Code after a ``return`` / ``raise`` in the same block is not reached
    and is not read.  Nested ``def`` / ``class`` / ``lambda`` bodies do not
    run where they are written and are not read either.
    """
    kinds = []

    def on_call(call, straight):
        kind = _IMPERATIVE_OUTCOMES.get(_dotted(call.func, aliases) or "")
        if kind == "importorskip":
            kinds.append("skipif")
        elif kind == "skip":
            kinds.append("skip" if straight else "skipif")
        elif kind == "xfail":
            kinds.append("xfail")
        elif (isinstance(call.func, ast.Name) and call.func.id in helpers
              and call.func.id not in seen):
            name = call.func.id
            for k in _imperative_kinds(helpers[name].body, aliases, helpers,
                                       seen | {name}):
                kinds.append(k if straight or k != "skip" else "skipif")

    def expr(node, straight):
        if isinstance(node, (ast.Lambda, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.IfExp):
            truth = _static_truth(node.test)
            expr(node.test, straight)
            expr(node.body, straight and truth is True)
            expr(node.orelse, straight and truth is False)
            return
        if isinstance(node, ast.BoolOp):
            expr(node.values[0], straight)
            for value in node.values[1:]:
                expr(value, False)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            straight = False             # the body may run zero times
        if isinstance(node, ast.Call):
            on_call(node, straight)
        for child in ast.iter_child_nodes(node):
            expr(child, straight)

    def block(body, straight):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            if isinstance(stmt, ast.If):
                truth = _static_truth(stmt.test)
                expr(stmt.test, straight)
                if truth is not False:
                    block(stmt.body, straight and truth is True)
                if truth is not True:
                    block(stmt.orelse, straight and truth is False)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    expr(item.context_expr, straight)
                block(stmt.body, straight)
            elif isinstance(stmt, (ast.Try, ast.TryStar)):
                block(stmt.body, straight)
                for handler in stmt.handlers:
                    block(handler.body, False)
                block(stmt.orelse, False)
                block(stmt.finalbody, straight)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                expr(stmt.iter, straight)
                block(stmt.body, False)
                block(stmt.orelse, False)
            elif isinstance(stmt, ast.While):
                truth = _static_truth(stmt.test)
                expr(stmt.test, straight)
                block(stmt.body, straight and truth is True)
                block(stmt.orelse, False)
            elif isinstance(stmt, ast.Match):
                expr(stmt.subject, straight)
                for case in stmt.cases:
                    block(case.body, False)
            else:
                expr(stmt, straight)
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break,
                                     ast.Continue)):
                    return               # the rest of this block never runs

    block(stmts, True)
    return kinds


def _module_functions(tree):
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _fixture_defs(body, aliases):
    """``({name: def}, [autouse defs])`` for the fixtures a body defines."""
    found, autouse = {}, []
    for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            call = deco if isinstance(deco, ast.Call) else None
            if _dotted(call.func if call else deco, aliases) != "pytest.fixture":
                continue
            name = node.name
            for kw in (call.keywords if call else []):
                if (kw.arg == "name" and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    name = kw.value.value
                elif kw.arg == "autouse" and _static_truth(kw.value) is True:
                    autouse.append(node)
            found[name] = node
    return found, autouse


def _parameters(func):
    args = func.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def _test_function_kinds(func, classes, tree, aliases):
    """Imperative skips / xfails a test function reaches when it runs.

    Its own body (:func:`_imperative_kinds`), and the body of every fixture
    it requests by parameter name, the fixtures those request, and the
    ``autouse`` fixtures in scope -- defined in the same module or in a
    class enclosing the test.  A fixture that calls ``pytest.skip()``
    stops the test before its first line.
    """
    helpers = _module_functions(tree)
    kinds = _imperative_kinds(func.body, aliases, helpers)
    fixtures, autouse = _fixture_defs(tree.body, aliases)
    for cls in classes:
        inner, inner_autouse = _fixture_defs(cls.body, aliases)
        fixtures.update(inner)
        autouse += inner_autouse
    todo = [fixtures[n] for n in _parameters(func) if n in fixtures] + autouse
    done = set()
    while todo:
        fixture = todo.pop()
        if id(fixture) in done:
            continue
        done.add(id(fixture))
        kinds += _imperative_kinds(fixture.body, aliases, helpers)
        todo += [fixtures[n] for n in _parameters(fixture) if n in fixtures]
    return kinds


def _module_marks(tree, aliases=None):
    """Skip / xfail marks a test module applies to everything in it.

    ``pytestmark = ...`` (aliases resolved), and whatever the module's own
    top-level code reaches when it is imported (:func:`_imperative_kinds`):
    a top-level ``pytest.skip(..., allow_module_level=True)`` is
    unconditional, one inside an ``if`` or ``except`` is conditional, and
    a ``pytest.importorskip(...)`` anywhere is conditional.
    """
    aliases = _pytest_aliases(tree) if aliases is None else aliases
    kinds = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "pytestmark" for t in node.targets):
            kinds += _mark_kinds(node.value, aliases)
    kinds += _imperative_kinds(tree.body, aliases, _module_functions(tree))
    return kinds


def _holds_a_test(body):
    """Does this module or class body hold a test pytest would collect?"""
    import ast

    for node in body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith(_TEST_FUNCTION_PREFIX)):
            return True
        if (isinstance(node, ast.ClassDef)
                and node.name.startswith(_TEST_CLASS_PREFIX)
                and _holds_a_test(node.body)):
            return True
    return False


def _reference_findings(aid, status, ref, repo_root):
    """``(errors, conditional)`` for one ``verification`` entry.

    The validator has already said whether the entry *resolves*; this says
    whether what it resolves to is evidence that runs.  An entry that does
    not resolve yields nothing here -- the validator reports it.

    * The file must be one pytest collects, and each component a name it
      collects: a ``Test*`` class without ``__init__``, a ``test*``
      function.  A module or class entry must hold at least one test.
    * The file must not be under a path no CI lane collects
      (:data:`PATHS_CI_NEVER_RUNS`): evidence has to run where the release
      is tested.
    * An unconditional ``skip`` anywhere on the path (the function, a class
      it sits in, the module) is an error: evidence that never runs.  A
      mark (aliases resolved), a ``skipif`` whose condition is fixed in the
      source, and a ``pytest.skip()`` every run reaches -- in the test's
      body, a helper it calls, or a fixture it requests -- all count.
    * An ``xfail`` on a ``resolved`` entry's evidence is an error: a
      resolution is evidenced by a test expected to pass.  On an open entry
      a strict ``xfail`` is how the defect is pinned, and is accepted.  A
      ``pytest.xfail()`` call counts as an ``xfail``.
    * A conditional skip (``skipif``, ``importorskip``, a guarded
      ``pytest.skip``) is returned in ``conditional``: it runs where its
      condition holds, and the gate reports it rather than hiding it.

    What this cannot see: a fixture or helper defined in another module
    (``conftest.py`` included), a method reached through ``self``, a
    ``usefixtures`` mark, a collection hook, and a condition that is
    always true on CI without being fixed in the source.  Those are the
    runtime's to decide; the gate reads source.
    """
    from maddening.compliance._validate import _defined_in, _member, _parse

    parts = str(ref).split("::")
    rel = parts[0]
    where = f"{aid}: verification entry '{ref}'"
    if not os.path.isfile(os.path.join(repo_root, rel)):
        return [], []
    if not _TEST_FILE.fullmatch(os.path.basename(rel)):
        return [f"{where}: '{rel}' is not a file pytest collects (test_*.py "
                f"or *_test.py), so nothing in it runs as a test"], []
    ignored = _ci_never_runs(rel)
    if ignored:
        return [f"{where}: '{rel}' is under {ignored}/, which every CI lane "
                f"excludes (--ignore={ignored} in .github/workflows), so it "
                f"never runs where the release is tested and evidences "
                f"nothing; move the test or cite one that CI runs"], []
    tree = _parse(os.path.join(repo_root, rel))
    if tree is None:
        return [], []

    module = _defined_in(tree.body)
    aliases = _pytest_aliases(tree)
    kinds = _module_marks(tree, aliases)
    node = None
    classes = []
    for name in parts[1:]:
        node = module.get(name) if node is None else (
            _member(node, name, module) if isinstance(node, ast.ClassDef)
            else None)
        if node is None:
            return [], []
        kinds += [k for d in node.decorator_list
                  for k in _mark_kinds(d, aliases)]
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            if not node.name.startswith(_TEST_CLASS_PREFIX):
                return [f"{where}: class {node.name!r} is not collected by "
                        f"pytest (a test class's name starts with "
                        f"'{_TEST_CLASS_PREFIX}')"], []
            if any(isinstance(n, ast.FunctionDef) and n.name == "__init__"
                   for n in node.body):
                return [f"{where}: class {node.name!r} defines __init__, so "
                        f"pytest does not collect it"], []
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        getattr(t, "id", None) == "pytestmark"
                        for t in stmt.targets):
                    kinds += _mark_kinds(stmt.value, aliases)
        elif not node.name.startswith(_TEST_FUNCTION_PREFIX):
            return [f"{where}: {node.name!r} is not a test -- pytest collects "
                    f"a function whose name starts with "
                    f"'{_TEST_FUNCTION_PREFIX}', so this never runs; cite the "
                    f"test that calls it"], []
    if node is None or isinstance(node, ast.ClassDef):
        body = tree.body if node is None else node.body
        if not _holds_a_test(body):
            return [f"{where}: holds no test pytest collects"], []
    else:
        kinds += _test_function_kinds(node, classes, tree, aliases)

    errors, conditional = [], []
    if "skip" in kinds:
        errors.append(f"{where}: is skipped unconditionally (pytest.mark.skip, "
                      f"a skipif whose condition is always true, or a "
                      f"pytest.skip() every run reaches -- in the module, the "
                      f"test, a helper it calls or a fixture it requests), so "
                      f"it never runs and evidences nothing; fix it or cite a "
                      f"test that runs")
    if "xfail" in kinds and status in _STATUSES_RESOLVED:
        errors.append(f"{where}: is marked xfail (or calls pytest.xfail()), "
                      f"but the entry is {status!r}; a resolution is "
                      f"evidenced by a test expected to pass")
    if "skipif" in kinds:
        conditional.append(f"{where}: is skipped conditionally (skipif, "
                           f"importorskip, or a pytest.skip() under a "
                           f"condition) -- it evidences the entry only where "
                           f"its condition lets it run")
    return errors, conditional


def _reference_errors(anomalies, repo_root, resolve=True):
    """``(errors, conditional)`` over every entry's references.

    A reference listed twice in one entry is refused whether or not
    references are resolved: it is one piece of evidence counted twice,
    and the count is what the gate's summary line reports (230 against 228
    when the audit duplicated one ``verification`` and one
    ``affected_components`` entry, against a CHANGELOG saying the counts
    had stopped inflating).  The rest needs the files, so it runs only
    when references are resolved.
    """
    errors, conditional = [], []
    for a in anomalies:
        aid = str(a.get("anomaly_id", "<missing>"))
        for field in _REFERENCE_FIELDS:
            entries = a.get(field) or []
            if isinstance(entries, str):
                entries = [entries]
            seen = set()
            for entry in entries:
                if str(entry) in seen:
                    errors.append(f"{aid}: {field} lists '{entry}' more than "
                                  f"once; one reference is one piece of "
                                  f"evidence, however often it is listed")
                seen.add(str(entry))
        if not (resolve and repo_root):
            continue
        refs = a.get("verification") or []
        if isinstance(refs, str):
            refs = [refs]
        for ref in dict.fromkeys(str(r) for r in refs):
            errs, cond = _reference_findings(
                aid, a.get("resolution_status"), ref, repo_root)
            errors += errs
            conditional += cond
    return errors, conditional


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
    retired, retired_error = retired_anomaly_ids(repo_root or _REPO_ROOT)
    errors = list(errors) + _evidence_errors(anomalies, retired)
    if retired_error:
        errors.append(retired_error)
    errors += retirement_errors(
        retired, args.path,
        present={str(a.get("anomaly_id")) for a in anomalies})
    if n_anomalies:
        errors += version_range_errors(data)
    ref_errors, conditional = _reference_errors(
        anomalies, repo_root, resolve=not args.no_resolve)
    errors += ref_errors

    # A symbol in an optional subpackage this environment cannot import is
    # unverified, not broken -- but an unverified reference is a hole in the
    # evidence, so say so on stdout rather than passing in silence.
    for n in notes:
        print(f"NOTE: {n}")
    for c in conditional:
        print(f"NOTE: {c}")

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
                 f"verified, {len(notes)} not checked; {len(conditional)} "
                 f"verification test(s) skip conditionally")
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
