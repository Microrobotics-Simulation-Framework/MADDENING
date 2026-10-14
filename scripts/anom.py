#!/usr/bin/env python3
"""Look entries up in the anomaly registry and its fragments without opening them.

    anom.py list [--open] [--grep TEXT] [--similar WORDS] ...
        one line per entry: id, status, severity, versions, title; entries
        still in a fragment show their label and [fragment <file>]
    anom.py show 195 [MADD-ANO-198 MADD-ANO-NEW-<stem>-<n> ...]
        the full entries, numbered or labelled
    anom.py next
        the registry's highest id and the committed high-water mark; a new
        entry gets no number here: it goes in a fragment, numbered at assembly
    anom.py refs 195 | MADD-ANO-NEW-<stem>-<n>
        every tracked file that mentions the id (or the label, and the id the
        label was assembled as)

The registry (docs/validation/known_anomalies.yaml) is ~21,000 lines; this
reads it as text, one block per entry.  Before registering an anomaly, run
`anom.py list --similar "a few words"`: it searches the fragments too.  New
entries: scripts/fragments.py new --anomaly.  Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fragments as F  # noqa: E402

_ID_LINE = re.compile(r"\s*-\s+anomaly_id:\s*[\"']?([^\"'\s]+)")
_RELEASE = re.compile(r"^## \[(\d+(?:\.\d+)*)\] - \d{4}-\d{2}-\d{2}\s*$", re.M)


class Entry:
    def __init__(self, key, where, line, lines):
        self.key, self.where, self.line, self.lines = key, where, line, lines
        m = re.fullmatch(r"MADD-ANO-(\d+)", key)
        self.number = int(m.group(1)) if m else None

    @property
    def in_fragment(self):
        return self.number is None

    def field(self, name):
        for i, ln in enumerate(self.lines):
            m = re.match(rf"\s+(?:-\s+)?{name}:\s*(.*)", ln)
            if m:
                v = m.group(1).strip()
                if v in ("", ">", "|", ">-", "|-"):
                    v = " ".join(x.strip() for x in self.lines[i + 1:i + 4])
                return v.strip("\"'")
        return ""

    def text(self):
        return "\n".join(self.lines)


def _blocks(text, where):
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if _ID_LINE.match(ln)]
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        for j in range(start + 1, end):
            if lines[j] and not lines[j][0].isspace() and not lines[j].startswith("#"):
                end = j
                break
        yield Entry(_ID_LINE.match(lines[start]).group(1), where, start + 1, lines[start:end])


def entries(root: Path) -> list[Entry]:
    """Every registry entry, then every fragment entry, in file order."""
    found = list(_blocks((root / F.REGISTRY).read_text(encoding="utf-8"), F.REGISTRY))
    folder = root / F.ANOMALIES
    if folder.is_dir():
        for path in sorted(folder.glob("*.yaml")):
            rel = path.relative_to(root).as_posix()
            found += list(_blocks(path.read_text(encoding="utf-8"), rel))
    return found


def _released(root: Path) -> list[tuple[int, ...]]:
    try:
        text = (root / F.CHANGELOG).read_text(encoding="utf-8")
    except OSError:
        return []
    return [tuple(int(x) for x in m.group(1).split(".")) for m in _RELEASE.finditer(text)]


def _release_key(version: str):
    m = re.match(r"(\d+(?:\.\d+)*)", version.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def shipped(versions: str, released) -> bool:
    """Whether ``affected_versions`` admits a version CHANGELOG.md lists as released.

    Read under the registry's own convention (``>=FIRST`` or ``>=FIRST,
    <FIX``, held by scripts/check_anomalies.py): FIRST may be a development
    build ``X.Y.Z.dev0``, which comes before X.Y.Z, so its release number is
    what is compared.  ``none`` admits nothing.
    """
    text = versions.strip()
    if not text or text.lower() == "none":
        return False
    first = fix = None
    for part in text.split(","):
        part = part.strip()
        if part.startswith(">="):
            first = _release_key(part[2:])
        elif part.startswith("<"):
            fix = _release_key(part[1:])
    if first is None:
        return False
    return any(first <= r and (fix is None or r < fix) for r in released)


def _wanted(spec: str) -> str:
    spec = spec.strip()
    if spec.isdigit():
        return F.numbered(int(spec))
    m = re.fullmatch(r"MADD-ANO-(\d+)", spec)
    return F.numbered(int(m.group(1))) if m else spec


def cmd_list(args, root):
    def words(t):
        return [w for w in re.findall(r"[a-z0-9_]+", t.lower()) if len(w) > 2]

    want_ids = None
    if args.ids:
        want_ids = set()
        for part in args.ids.split(","):
            if ".." in part:
                lo, hi = part.split("..")
                want_ids.update(range(int(lo), int(hi) + 1))
            else:
                want_ids.add(int(part))
    released = _released(root)
    rows = []
    for e in entries(root):
        low = e.text().lower()
        st, sev = e.field("resolution_status"), e.field("severity")
        ver, title = e.field("affected_versions"), e.field("title")
        if want_ids is not None and e.number not in want_ids:
            continue
        if args.open and st in F.UNREACHABLE:
            continue
        if args.status and st not in [x.strip() for x in args.status.split(",")]:
            continue
        if args.severity and sev not in [x.strip() for x in args.severity.split(",")]:
            continue
        if args.title and not all(w in title.lower() for w in args.title.lower().split()):
            continue
        if args.grep and args.grep.lower() not in low:
            continue
        if args.shipped and not shipped(ver, released):
            continue
        if args.never_shipped and shipped(ver, released):
            continue
        if args.resolved_in and e.field("resolution_version") != args.resolved_in:
            continue
        if args.fragments_only and not e.in_fragment:
            continue
        if args.component:
            m = re.search(r"\n\s+affected_components:(.*?)(?=\n    [a-z_]+:|\Z)", e.text(), re.S)
            if not m or args.component.lower() not in m.group(1).lower():
                continue
        score = 0
        if args.similar:
            q, t, d = set(words(args.similar)), set(words(title)), set(words(e.text()))
            score = 3 * len(q & t) + len(q & d)
            if score == 0:
                continue
        rows.append((score, e, st, sev, ver, title))
    if args.similar:
        rows.sort(key=lambda r: -r[0])
    if args.count:
        print(f"{len(rows)} entries ({sum(r[1].in_fragment for r in rows)} in fragments)")
        for name, k in (("status", 2), ("severity", 3)):
            c = collections.Counter(r[k] for r in rows)
            print(f"  by {name}: " + ", ".join(f"{v} {n}" for n, v in
                                               sorted(c.items(), key=lambda x: -x[1])))
        n_shipped = sum(shipped(r[4], released) for r in rows)
        print(f"  shipped: {n_shipped}, never shipped: {len(rows) - n_shipped}")
        return 0
    for score, e, st, sev, ver, title in rows:
        t = title if args.full_title else title[:110]
        lead = f"[{score:>2}] " if args.similar else ""
        ident = f"{e.number:>4}" if e.number is not None else e.key
        tail = f"  [fragment {e.where}]" if e.in_fragment else ""
        print(f"{lead}{ident} {st:<19} {sev:<8} {ver:<18} {t}{tail}")
    return 0


def cmd_show(args, root):
    every = entries(root)
    status = 0
    for spec in args.ids:
        key = _wanted(spec)
        hits = [e for e in every if e.key == key]
        if not hits:
            print(f"{key}: not in the registry or a fragment of this tree")
            status = 1
            continue
        for e in hits:
            print(f"# {e.where} line {e.line}")
            print(e.text().rstrip())
            print()
    return status


def cmd_next(args, root):
    every = entries(root)
    ids = [e.number for e in every if e.number is not None]
    labels = [e for e in every if e.in_fragment]
    hw = re.search(r"^_HIGHEST_ANOMALY_ID\s*=\s*(\d+)", (root / F.HW_FILE).read_text(
        encoding="utf-8"), re.M)
    print(f"entries: {len(ids)}; highest id in the registry: {max(ids, default=0)}; "
          f"_HIGHEST_ANOMALY_ID: {hw.group(1) if hw else '?'}; "
          f"in fragments: {len(labels)} entr(ies) in "
          f"{len({e.where for e in labels})} file(s)")
    missing = sorted(set(range(1, max(ids, default=0) + 1)) - set(ids))
    if missing:
        print("gaps:", missing[:20])
    print("A new entry takes no number by hand: write it in docs/fragments/anomalies/"
          "<stem>.yaml (scripts/fragments.py new --anomaly) under its label; "
          "scripts/fragments.py assemble numbers it.")
    return 0


def cmd_refs(args, root):
    key = _wanted(args.id)
    text = (root / F.REGISTRY).read_text(encoding="utf-8")
    recorded = F.recorded_labels(text)
    patterns = []
    if F.LABEL_RE.fullmatch(key):
        patterns.append(re.escape(key) + r"([^A-Za-z0-9_-]|$)")
        for aid in recorded.get(key, []):
            print(f"{key} was assembled as {aid}")
            patterns.append(rf"MADD-ANO-0*{int(aid.split('-')[-1])}\b")
    else:
        m = re.fullmatch(r"MADD-ANO-(\d+)", key)
        if not m:
            print(f"{args.id!r} is neither an id nor a label", file=sys.stderr)
            return 2
        patterns.append(rf"MADD-ANO-0*{int(m.group(1))}\b")
        for lab, ids in recorded.items():
            if key in ids:
                print(f"{key} came from the fragment label {lab}")
    out = subprocess.run(["git", "-C", str(root), "grep", "-n", "-E", "|".join(patterns)],
                         capture_output=True, text=True).stdout
    for ln in out.splitlines():
        print(ln[:160])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=F.REPO_ROOT,
                    help="the repository to read (default: the one holding this script)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="one line per entry, with filters", description=(
        "One line per entry: id (or label), status, severity, affected versions, title. "
        "Filters combine (AND); words are case-insensitive. Entries still in a fragment "
        "are included and marked."))
    ls.add_argument("--open", action="store_true", help="only entries still reachable (not resolved, not duplicate)")
    ls.add_argument("--status", help="comma-separated statuses, e.g. open,partially_resolved")
    ls.add_argument("--severity", help="comma-separated severities, e.g. critical,major")
    ls.add_argument("--title", help="every word must appear in the title")
    ls.add_argument("--grep", help="text that must appear anywhere in the entry")
    ls.add_argument("--component", help="text that must appear in affected_components")
    ls.add_argument("--shipped", action="store_true", help="only defects a released version carried "
                    "(the range admits a version CHANGELOG.md lists as released)")
    ls.add_argument("--never-shipped", action="store_true", help="only defects no released version carried")
    ls.add_argument("--resolved-in", help="resolution_version equals this, e.g. 0.4.0")
    ls.add_argument("--ids", help="numbered ids only: a range or list, e.g. 190..210 or 12,35")
    ls.add_argument("--fragments-only", action="store_true", help="only entries still in a fragment")
    ls.add_argument("--similar", help="rank entries by the words their title and text share with "
                    "these; use it to check whether an anomaly already exists")
    ls.add_argument("--count", action="store_true", help="totals by status and severity only")
    ls.add_argument("--full-title", action="store_true", help="do not cut titles at 110 characters")
    sh = sub.add_parser("show", help="the full entries")
    sh.add_argument("ids", nargs="+", help="195, MADD-ANO-195, or a label")
    sub.add_parser("next", help="the highest id and the high-water mark")
    rf = sub.add_parser("refs", help="every tracked file that mentions an id or a label")
    rf.add_argument("id", help="195, MADD-ANO-195, or a label")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    return {"list": cmd_list, "show": cmd_show, "next": cmd_next,
            "refs": cmd_refs}[args.cmd](args, root)


if __name__ == "__main__":
    sys.exit(main())
