#!/usr/bin/env python3
"""CI bridge: verify Implementation Mapping tables in algorithm guides.

Parses each algorithm guide's Implementation Mapping Markdown table,
extracts every backticked ``maddening.*`` qualified name from its rows, and
verifies that each one resolves to an existing callable that the named
class actually defines.

Four things this gate has to get right, because each was a hole:

* **Own ``__dict__``, not the MRO.**  ``getattr`` walks base classes, so a
  row naming ``HeatNode.update`` kept resolving after the concrete method
  was renamed -- ``SimulationNode.update`` answered instead.  A row opts
  into inherited behaviour only with an explicit marker: a Notes cell that
  *begins* ``Inherited from `Base```, naming the class the symbol actually
  comes from.  The gate checks the named base against the one the symbol
  resolves through, and fails a marker on a row whose symbols are all
  defined on the class they name (the override the marker denies).  The
  marker used to be the substring "inherited" anywhere in the row, so
  "not inherited" switched the check off (audit_040_phase3_wave_d, M2).
* **Every symbol in a row**, not just the first, and every row must carry a
  code reference at all -- dropping the backticks used to drop the row.
* **Every code span in the Implementation column is a qualified
  ``maddening.*`` name.**  A span that lost its ``maddening.`` prefix still
  counted as "a code reference", was never resolved, and -- within the
  slack between a guide's count and its pin -- left the gate green
  (audit_040_phase3_wave_d, M3).  The one exception is the documented
  convention for a term a JAX primitive or third-party function handles:
  a Notes cell that *begins* ``JAX primitive`` or ``Third-party`` declares
  the row's unqualified spans as such.  They are reported, not verified.
* **A pinned minimum per guide**, so a table that vanishes fails instead of
  quietly lowering the count.  The pins sit at the current counts; a guide
  that gains rows should raise its pin, or the slack reopens for row
  deletions.  ``tests/compliance/test_gate_scripts.py::TestMinMappingsRatchet``
  holds each pin equal to its guide's count, so that cannot happen quietly.

Two identity checks run beside the mappings, because a mapping table is only
evidence for the node the guide is *about*:

* **Every node algorithm ID is unique across ``src/maddening``.**  Every
  ``algorithm_id=`` keyword (and ``NodeMeta``'s first positional argument)
  is read from the source with :mod:`ast`, so a module that needs an
  optional extra is still scanned.  One that is not a string literal, or a
  ``NodeMeta(**...)`` that could hide one, fails: an ID the scan cannot read
  is an ID whose uniqueness nobody checked.  ``LBMNode`` and
  ``RigidBodyNode`` both carried ``MADD-NODE-007`` from 0.1.0 to 0.3.1, and
  with a second duplicate seeded every gate still passed
  (audit_040_phase3_confirm, release-record).
* **A guide's ``**Algorithm ID**`` is its node's ``NodeMeta.algorithm_id``.**
  The guide names its node by its ``# Title`` and ``**Module**`` lines; the
  class they resolve to must define its own ``NodeMeta`` with the ID the
  guide states.  A ``MADD-NODE-`` ID on a guide whose title and module name
  no node class fails, and so does a node guide with no ID line, a second
  ID line, or one not written ``**Algorithm ID**: `ID```.  Every stated ID
  is also unique among the guides, which is the only check a non-node ID
  (``MADD-ALG-...``) gets.

Usage:
    python scripts/check_impl_mapping.py [docs/algorithm_guide/]

Exits 0 if all mappings resolve, nonzero if any are stale.
"""

import ast
import importlib
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# Add src to path
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from maddening.compliance._validate import resolve_dotted_name  # noqa: E402

DEFAULT_GUIDE_DIR = os.path.join("docs", "algorithm_guide")

#: The package whose node algorithm IDs must be unique.  Scanned whatever
#: guide directory the gate is pointed at, like the pins: uniqueness is a
#: property of the tree, not of the scope one run happened to name.
SRC_PACKAGE = os.path.join(_REPO_ROOT, "src", "maddening")

#: Prefix of the IDs ``NodeMeta.algorithm_id`` carries.  A guide stating
#: one must name the node that carries it.
NODE_ID_PREFIX = "MADD-NODE-"

# Minimum number of resolvable ``maddening.*`` references per guide, keyed by
# path relative to the repository root.  Pinned so that a deleted table, or a
# row that loses its backticks, fails instead of reporting a smaller "OK".
# Raise a number when a guide gains rows.
#
# Lowering one is not a convention any more: ``tests/compliance/
# min_mappings_floor.json`` holds a committed floor and
# ``TestMinMappingsRatchet`` asserts ``MIN_MAPPINGS[path] >= floor[path]``,
# so a pin can only go down together with an edit to another file.  It used
# to be a comment saying "never lower one to make CI pass", and dropping a
# pin from 9 to 1 while deleting 8 rows of the guide left every gate and
# every mapping test green (audit_040_r2/gates, finding G6).
#
# A pin below its guide's count is slack a row deletion hides in: the
# wavelet guide sat at 20 with 24 references, so four could go
# (audit_040_phase3_confirm).  Floors, not exact counts, so that two
# branches each adding rows to one guide do not collide on its number.
#
# The slack came back: heat, lbm and wavelet sat two below their counts
# (11/13, 22/24, 24/26), so deleting two rows from any of them passed
# while the module docstring and the CHANGELOG said the pins sat at the
# counts (audit_040_p4_1, M5-M8).  ``TestMinMappingsRatchet`` now holds
# every pin *equal* to its guide's count, and every guide with a mapping
# table to a pin, so raising a guide without raising its pin fails CI
# rather than reopening the hole.
MIN_MAPPINGS = {
    os.path.join("docs", "algorithm_guide", "nodes", "heat_node.md"): 15,
    os.path.join("docs", "algorithm_guide", "nodes", "adaptive_node.md"): 12,
    os.path.join("docs", "algorithm_guide", "nodes", "wavelet_adaptive_node.md"): 26,
    os.path.join(
        "docs", "algorithm_guide", "solvers", "explicit_integrators.md"
    ): 5,
    os.path.join("docs", "algorithm_guide", "nodes", "spring_node.md"): 9,
    os.path.join("docs", "algorithm_guide", "nodes", "ball_node.md"): 7,
    os.path.join("docs", "algorithm_guide", "nodes", "rigid_body_2d_node.md"): 7,
    os.path.join("docs", "algorithm_guide", "nodes", "heart_pump_node.md"): 9,
    os.path.join("docs", "algorithm_guide", "nodes", "lbm_node.md"): 24,
}

_QNAME = re.compile(r"`(maddening\.[^`]+)`")
_CODE_SPAN = re.compile(r"`[^`]+`")
#: The only spelling that opts a row into inherited resolution: a Notes cell
#: that *starts* with it, naming the defining base class (bare or
#: qualified).  Anchored and case-sensitive, so prose that merely mentions
#: inheritance -- "not inherited", "the inherited update" -- is not a marker.
_INHERITED_MARKER = re.compile(r"Inherited from `([A-Za-z_][\w.]*)`")
#: The only spelling that declares an Implementation span to be a JAX
#: primitive or third-party call rather than a MADDENING symbol
#: (``docs/developer_guide/documentation_standards.md``); anchored likewise.
_PRIMITIVE_MARKER = re.compile(r"(?:JAX primitive|Third-party)\b")
# A Markdown table cell may contain an escaped pipe.  Splitting on a bare
# ``|`` mangled every row holding LaTeX like ``\|g\|``, which shifted the
# Implementation column out of cell 1 and made the row invisible to the gate.
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into cells, honouring ``\\|`` escapes."""
    return [c.strip() for c in _UNESCAPED_PIPE.split(line)[1:-1]]


def extract_rows(md_path: str) -> list[list[str]]:
    """Return the data rows of the Implementation Mapping table, as cell lists."""
    with open(md_path) as f:
        content = f.read()

    section_match = re.search(
        r"## Implementation Mapping\s*\n(.*?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1)

    rows = []
    for line in section.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if len(cells) < 2:
            continue
        # Separator rows (|---|---|---|)
        if all(set(c) <= {"-", " ", ":"} for c in cells):
            continue
        # Header row
        if (cells[0].lower().startswith("equation")
                or cells[1].lower().startswith("implementation")):
            continue
        rows.append(cells)

    return rows


def extract_qualified_names(md_path: str) -> list[tuple[str, str]]:
    """Extract ``(equation_term, qualified_name)`` pairs from a guide.

    Every backticked ``maddening.*`` span in the row is returned, not only
    the first: a row may trace one equation term to two functions.
    """
    results = []
    for cells in extract_rows(md_path):
        term = cells[0]
        for match in _QNAME.finditer(" | ".join(cells)):
            results.append((term, match.group(1).rstrip("`).,( ")))
    return results


def check_guide(
    md_path: str, relpath: str
) -> tuple[int, list[str], list[str], list[str]]:
    """Check one guide.  Returns ``(n_checked, errors, notes, skipped)``.

    ``n_checked`` counts only references this environment could actually
    resolve; one that needs an uninstalled optional subpackage lands in
    ``skipped`` and is not counted as verified.
    """
    errors: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    checked = 0

    for cells in extract_rows(md_path):
        term = cells[0]
        impl = cells[1]
        # A row whose Implementation cell carries no code span at all traces
        # its equation term to nothing.  Dropping the backticks used to make
        # the row invisible to this gate.
        if not _CODE_SPAN.search(impl):
            errors.append(
                f"{relpath}: row '{term}' has no code reference in its "
                f"Implementation column"
            )
            continue

        # A code span that is not a qualified name is never resolved, so a
        # dropped ``maddening.`` prefix made the symbol invisible while the
        # row still "carried a code reference".
        declares_primitive = any(_PRIMITIVE_MARKER.match(c) for c in cells[2:])
        for span in _CODE_SPAN.findall(impl):
            # One definition of "a qualified name": the pattern the symbols
            # are extracted with, so the two can never disagree.
            if _QNAME.fullmatch(span):
                continue
            if declares_primitive:
                skipped.append(
                    f"{relpath}: {span} (for term '{term}') is declared a "
                    f"JAX primitive / third-party call and was NOT checked"
                )
            else:
                errors.append(
                    f"{relpath}: row '{term}' has {span} in its "
                    f"Implementation column, which is not a qualified "
                    f"maddening.* name, so it cannot be checked -- write the "
                    f"full dotted path, or begin the Notes cell with "
                    f"'JAX primitive' if it is one"
                )

        row_text = " | ".join(cells)
        marker = next(
            (m.group(1) for m in (_INHERITED_MARKER.match(c) for c in cells[2:])
             if m),
            None,
        )
        marker_base = marker.rsplit(".", 1)[-1] if marker else None
        row_inherited = False
        row_checked = False
        for match in _QNAME.finditer(row_text):
            qname = match.group(1).rstrip("`).,( ")
            checked += 1
            res = resolve_dotted_name(
                qname,
                require_own=marker_base is None,
                require_callable=True,
            )
            if res.unavailable:
                # An optional subpackage this environment cannot import.
                # Not checked is not the same as not there; saying "stale"
                # here would make a guide for a USD or viz node impossible
                # to keep green in a CI that installs only [ci].
                checked -= 1
                skipped.append(
                    f"{relpath}: '{qname}' (for term '{term}') was NOT "
                    f"checked -- {res.reason}"
                )
                continue
            row_checked = True
            if not res.ok:
                hint = ""
                if res.inherited_from and marker_base is None:
                    hint = (f"; if the row means the inherited behaviour, "
                            f"begin its Notes cell with "
                            f"'Inherited from `{res.inherited_from}`'")
                errors.append(
                    f"{relpath}: '{qname}' (for term '{term}') does not "
                    f"resolve: {res.reason}{hint}"
                )
            elif res.inherited_from:
                row_inherited = True
                if res.inherited_from != marker_base:
                    errors.append(
                        f"{relpath}: '{qname}' (for term '{term}') resolves "
                        f"through {res.inherited_from}, but the row says "
                        f"'Inherited from `{marker}`'"
                    )
                else:
                    notes.append(
                        f"{relpath}: '{qname}' (for term '{term}') is "
                        f"inherited from {res.inherited_from}, as the row "
                        f"states"
                    )
        if marker_base is not None and row_checked and not row_inherited:
            # The marker is a claim about the code.  When the class now
            # defines the symbol itself, the claim is stale -- and it is the
            # very override the row says does not exist.
            errors.append(
                f"{relpath}: row '{term}' says 'Inherited from `{marker}`', "
                f"but every symbol in it is defined on the class it names; "
                f"drop the marker"
            )

    return checked, errors, notes, skipped


def check_pinned(
    per_file: dict[str, int],
    min_mappings: dict[str, int],
    repo_root: str,
) -> list[str]:
    """Check the pinned per-guide minimums against the repository.

    Runs regardless of the directory that was scanned, so that a guide
    deleted outright still fails rather than dropping out of the count.
    """
    errors: list[str] = []
    for pinned, minimum in sorted(min_mappings.items()):
        abspath = os.path.join(repo_root, pinned)
        if not os.path.isfile(abspath):
            errors.append(
                f"{pinned}: pinned in MIN_MAPPINGS but the file does not exist"
            )
            continue
        found = per_file.get(pinned)
        if found is None:
            found, errs, _notes, _skipped = check_guide(abspath, pinned)
            errors.extend(errs)
        if found < minimum:
            errors.append(
                f"{pinned}: {found} implementation mapping(s) found, at least "
                f"{minimum} expected -- a row or the whole table has gone "
                f"missing (update MIN_MAPPINGS only if the guide legitimately "
                f"shrank)"
            )
    return errors


def _callee_name(call: ast.Call):
    """``NodeMeta`` for ``NodeMeta(...)`` and ``compliance.NodeMeta(...)``."""
    func = call.func
    return getattr(func, "id", None) or getattr(func, "attr", None)


def algorithm_ids(src_root: str):
    """Every node algorithm ID declared under ``src_root``, read statically.

    Returns ``(ids, errors)``: ``ids`` maps each ID to the ``path:line``
    locations that declare it, and ``errors`` lists every declaration the
    scan could not read.  Static on purpose: importing the package would
    skip whatever needs an optional extra, and a duplicate there is still a
    duplicate.

    Read as an ID: the value of any ``algorithm_id=`` keyword (``NodeMeta``,
    ``dataclasses.replace`` or anything else), and the first positional
    argument of a ``NodeMeta(...)`` call, which is ``algorithm_id``.  The
    empty string is ``NodeMeta``'s default and claims no ID.
    """
    ids: dict[str, list[str]] = {}
    errors: list[str] = []
    for root, _dirs, files in os.walk(src_root):
        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, _REPO_ROOT)
            try:
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=path)
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"{rel}: could not be parsed ({exc}), so no "
                              f"algorithm ID in it was checked")
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                values = [kw.value for kw in node.keywords
                          if kw.arg == "algorithm_id"]
                if _callee_name(node) == "NodeMeta":
                    if node.args:
                        values.append(node.args[0])
                    if any(kw.arg is None for kw in node.keywords):
                        errors.append(
                            f"{rel}:{node.lineno}: NodeMeta(**...) can hide "
                            f"an algorithm_id this check cannot read; pass it "
                            f"as a string literal keyword")
                for value in values:
                    where = f"{rel}:{getattr(value, 'lineno', node.lineno)}"
                    if not (isinstance(value, ast.Constant)
                            and isinstance(value.value, str)):
                        errors.append(
                            f"{where}: algorithm_id is not a string literal, "
                            f"so its uniqueness cannot be checked; write the "
                            f"ID out")
                    elif value.value:
                        ids.setdefault(value.value, []).append(where)
    return ids, errors


def algorithm_id_errors(src_root: str = SRC_PACKAGE):
    """``(n_ids, errors)``: every node algorithm ID must be unique."""
    ids, errors = algorithm_ids(src_root)
    for aid, where in sorted(ids.items()):
        if len(where) > 1:
            errors.append(
                f"algorithm ID {aid} is declared {len(where)} times "
                f"({', '.join(where)}).  An algorithm ID names one algorithm "
                f"in the compliance record; give all but one of them a new, "
                f"unused ID and record the old -> new mapping in the release "
                f"notes")
    if not ids and not errors:
        errors.append(
            f"no algorithm_id found under {os.path.relpath(src_root, _REPO_ROOT)}"
            f"; a uniqueness check over nothing verifies nothing -- the scan "
            f"scope is wrong")
    return len(ids), errors


#: A line that *tries* to state an ID, whatever its spelling, so that a
#: misspelt one fails instead of dropping out of the check.
_ID_LINE_ANY = re.compile(r"^\s*\*\*Algorithm ID\*\*.*$", re.M)
_ID_LINE = re.compile(r"^\*\*Algorithm ID\*\*: `([^`\s]+)`\s*$")
_TITLE_LINE = re.compile(r"^# (\S+)\s*$", re.M)
_MODULE_LINE = re.compile(r"^\*\*Module\*\*: `([\w.]+)`\s*$", re.M)


def _guide_node_class(content: str):
    """``(qualified name, class or None, reason)`` for a guide's node.

    The class is the one the guide's first ``# Title`` names inside its
    ``**Module**``.  ``class`` is ``None`` with a ``reason`` when either
    line is missing or the name does not resolve; ``reason`` starts with
    ``unavailable:`` when an optional extra is what stopped it.
    """
    from maddening.core.node import SimulationNode

    title, module = _TITLE_LINE.search(content), _MODULE_LINE.search(content)
    if not (title and module):
        return None, None, "it has no '# ClassName' and '**Module**' header"
    qname = f"{module.group(1)}.{title.group(1)}"
    res = resolve_dotted_name(qname)
    if res.unavailable:
        return qname, None, f"unavailable: {res.reason}"
    if not res.ok:
        return qname, None, f"'{qname}' does not resolve ({res.reason})"
    obj = getattr(importlib.import_module(module.group(1)), title.group(1))
    if not (isinstance(obj, type) and issubclass(obj, SimulationNode)):
        return qname, None, f"'{qname}' is not a SimulationNode subclass"
    return qname, obj, None


def guide_id_errors(md_path: str, relpath: str):
    """Check one guide's stated algorithm ID against its node's NodeMeta.

    Returns ``(stated_id or None, matched, errors, skipped)``; ``matched``
    is true when the ID was compared with a ``NodeMeta`` and agreed.
    """
    with open(md_path, encoding="utf-8") as fh:
        content = fh.read()
    lines = [m.group(0).strip() for m in _ID_LINE_ANY.finditer(content)]
    qname, cls, why_not = _guide_node_class(content)
    if why_not and why_not.startswith("unavailable:"):
        return None, False, [], [f"{relpath}: the node the guide documents "
                                 f"was NOT checked -- {why_not[13:]}"]
    if not lines:
        if cls is not None:
            return None, False, [
                f"{relpath}: documents node {qname} but states no "
                f"'**Algorithm ID**: `...`'; its NodeMeta says "
                f"{getattr(cls.__dict__.get('meta'), 'algorithm_id', None)!r}"
            ], []
        return None, False, [], []
    if len(lines) > 1:
        return None, False, [f"{relpath}: states {len(lines)} algorithm IDs "
                             f"({lines}); a guide documents one algorithm"], []
    m = _ID_LINE.match(lines[0])
    if not m:
        return None, False, [
            f"{relpath}: {lines[0]!r} is not written '**Algorithm ID**: `ID`', "
            f"so the ID cannot be checked against its node"], []
    stated = m.group(1)
    if cls is None:
        if stated.startswith(NODE_ID_PREFIX):
            return stated, False, [
                f"{relpath}: states node algorithm ID {stated}, but {why_not}, "
                f"so no NodeMeta can be compared with it"], []
        return stated, False, [], []
    meta = cls.__dict__.get("meta")
    if meta is None:
        return stated, False, [
            f"{relpath}: states algorithm ID {stated} for {qname}, which "
            f"defines no NodeMeta of its own (it inherits "
            f"{getattr(cls.meta, 'algorithm_id', None)!r}); a guide documents "
            f"the node that carries the ID"], []
    if meta.algorithm_id != stated:
        return stated, False, [
            f"{relpath}: states algorithm ID {stated}, but "
            f"{qname}.meta.algorithm_id is {meta.algorithm_id!r}"], []
    return stated, True, [], []


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    guide_dir = argv[0] if argv else os.path.join(_REPO_ROOT, DEFAULT_GUIDE_DIR)

    if not os.path.isdir(guide_dir):
        print(f"Directory not found: {guide_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    id_skipped: list[str] = []
    checked = 0
    per_file: dict[str, int] = {}
    stated_ids: dict[str, list[str]] = {}
    ids_matched = 0

    # Recursive: the scope used to be a non-recursive listdir of
    # docs/algorithm_guide/nodes/, so a guide in any other subdirectory was
    # invisible to the gate.
    for root, _dirs, files in os.walk(guide_dir):
        for fname in sorted(files):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, _REPO_ROOT)
            n, errs, ns, sk = check_guide(fpath, relpath)
            checked += n
            per_file[relpath] = n
            errors.extend(errs)
            notes.extend(ns)
            skipped.extend(sk)
            stated, matched, errs, sk = guide_id_errors(fpath, relpath)
            errors.extend(errs)
            id_skipped.extend(sk)
            ids_matched += matched
            if stated is not None:
                stated_ids.setdefault(stated, []).append(relpath)

    for aid, guides in sorted(stated_ids.items()):
        if len(guides) > 1:
            errors.append(f"algorithm ID {aid} is stated by {len(guides)} "
                          f"guides ({', '.join(guides)}); each documents one "
                          f"algorithm")

    errors.extend(check_pinned(per_file, MIN_MAPPINGS, _REPO_ROOT))
    n_node_ids, id_errors = algorithm_id_errors(SRC_PACKAGE)
    errors.extend(id_errors)

    for n in notes:
        print(f"NOTE: {n}")
    for sk in skipped + id_skipped:
        print(f"NOTE: {sk}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} problem(s) found; {checked} mapping(s) checked",
            file=sys.stderr,
        )
        return 1

    # A scanned scope with nothing in it verified nothing, whatever the pins
    # did.  The pins resolve against _REPO_ROOT rather than the scanned
    # directory, so they still ran -- but the line this gate prints names the
    # scanned scope, and that line is what gets quoted as coverage.  The same
    # guard check_heat_stability.py and check_transforms.py already have
    # (audit_040_r2/gates, finding G7).
    if checked == 0:
        print(
            f"FAIL: 0 implementation mapping(s) found in {guide_dir}"
            + (f" ({len(skipped)} reference(s) could not be checked in this "
               f"environment)" if skipped else "")
            + ".\nA gate that verifies nothing cannot fail.  Either the scan "
              "scope is wrong or every guide has lost its Implementation "
              "Mapping table; fix the scope rather than trusting the OK.",
            file=sys.stderr,
        )
        return 1

    suffix = f", {len(skipped)} not checked" if skipped else ""
    id_suffix = (f", {len(id_skipped)} not checked" if id_skipped else "")
    print(
        f"OK: {checked} implementation mapping(s) verified{suffix} "
        f"across {guide_dir}; {ids_matched} guide algorithm ID(s) match "
        f"their node's NodeMeta{id_suffix}; {n_node_ids} node algorithm "
        f"ID(s) unique across src/maddening"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
