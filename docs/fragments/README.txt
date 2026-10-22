Changelog and registry fragments
================================

A pull request never edits CHANGELOG.md, docs/release_notes/,
docs/validation/known_anomalies.yaml, _HIGHEST_ANOMALY_ID, the SOUP table or
the development record's index.  It adds its own files here instead, named
after its branch (lower-case, with / _ . turned into -):

  changes/<branch>.yaml     changelog bullets and release-notes text
  anomalies/<branch>.yaml   new registry entries, each labelled
                            MADD-ANO-NEW-<branch>-<n> until it is numbered

Start one with:

  python scripts/fragments.py new --changes      (or --anomaly)

and check it with:

  python scripts/check_fragments.py --base origin/<your base branch>
  python scripts/check_anomalies.py --prefix MADD-ANO-

(--base: the ratchet reads the history of the line you merge into, not your
branch's own commits before it merges.)

Name a new anomaly by its label in tests and other files.  The assembler
(scripts/fragments.py assemble, run as its own pull request) numbers the
entries, rewrites every label to its number, writes the changelog and the
release notes at each release, and deletes the fragments it took.

Pull requests merge with a merge commit only (never squashed or rebased):
the ratchet reads each line's merge history, one merge commit per pull
request.

Known limits
------------

Follow-ups, not handled yet:

- Retiring an assembled entry (_RETIRED_ANOMALY_IDS): its "# label:" comment
  leaves with it, and the ratchet then reports the label as dropped.  Keep
  the entry, closed.
- Reverting a merged pull request is refused for its fragments.  Close its
  anomaly entries instead of deleting them.  A changes fragment has no clean
  way out yet: only a release assembly removes it, so its text would ship.
- A hand-written "# label:" comment can still pass in two cases: a label the
  pull request's own commits held, and a comment above the registry's first
  entry.  Never write one.
- At the numbering hand-over (RELEASE_LINES = ("0.5",)) a fragment still on
  line "0.4" turns the gate red: re-line it to "0.5" in that commit.
- Run locally without --base, the hint about a fragment added and dropped on
  your own branch also shows where it does not apply (a release branch, a
  shallow clone).  CI passes --base on every run.
- A fix merged at a release version after the release assembly is red as its
  own pull request: merge it into the branch that runs the assembly again.
- The release assembly regenerates the SOUP table only when it numbers an
  entry; after a version bump alone, run scripts/generate_soup_tables.py.
- A .0 release's notes list each entry's title as written, not in code spans.
- Some test helpers use libyaml's loader; every gate reads the files
  themselves, and the anomaly gate its old registry versions, with
  yaml.safe_load.

This file is plain text on purpose: the documentation build reads every .md
file under docs/, and a page here would be an orphan.  The whole procedure is
in docs/developer_guide/documentation_standards.md, "Recording a change".
