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

This file is plain text on purpose: the documentation build reads every .md
file under docs/, and a page here would be an orphan.  The whole procedure is
in docs/developer_guide/documentation_standards.md, "Recording a change".
