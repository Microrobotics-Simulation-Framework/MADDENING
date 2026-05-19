---
orphan: false
---

# Versioned documentation: design and rollout

```{versionadded} v0.2
The per-version `switcher.json` scaffolding lands with v0.2.  The
companion changes to MICROROBOTICA's umbrella `conf.py` + docs CI
are required to make the switcher live on the published site —
that work is tracked under [MICROROBOTICA umbrella PR pending](#).
```

The three MSF projects (MADDENING, MIME, MICROROBOTICA) are
published from a single sphinx-multiproject build at
<https://microrobotica.org/>.  Today that build emits a single
"latest" version of each project — there's no `v0.1` URL you can
deep-link to.  When v0.2 lands as the new stable release, v0.1
documentation effectively disappears.

This page is the rollout plan for keeping every supported version
addressable at a stable URL with a UI version picker.

## The shape of the live site (target)

```
https://microrobotica.org/                       — MICROROBOTICA latest
https://microrobotica.org/v0.1/                  — MICROROBOTICA v0.1
https://microrobotica.org/maddening/             — MADDENING latest (== current main)
https://microrobotica.org/maddening/v0.2/        — MADDENING v0.2 (stable)
https://microrobotica.org/maddening/v0.1/        — MADDENING v0.1
https://microrobotica.org/mime/                  — MIME latest
https://microrobotica.org/mime/v0.2/             — MIME v0.2
```

* `/<project>/` always points at `latest` (= the most recent tagged
  release, not `main`).  `main` content lives at
  `/<project>/dev/` if someone wants pre-release docs.
* Each version is a frozen build of the docs in that release's
  `docs/` tree against the Sphinx config at that tag.  Switching
  versions doesn't switch *just* the content — the navbar, theme,
  and inventory all freeze together.

## How a user sees it: PyData theme's version switcher

PyData Sphinx Theme ships a built-in [version
switcher](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html)
that reads a JSON file at build time and renders a dropdown in the
navbar.  We use it because the umbrella site already runs PyData
theme (`MICROROBOTICA/docs/conf.py` line ~70).

Each project ships its own `switcher.json` at
`docs/_static/switcher.json`, e.g. {download}`MADDENING's
<../_static/switcher.json>`:

```json
[
  {"version": "latest", "name": "latest (main)",
   "url": "https://microrobotica.org/maddening/"},
  {"version": "v0.2",   "name": "v0.2 (stable)",
   "url": "https://microrobotica.org/maddening/v0.2/", "preferred": true},
  {"version": "v0.1",   "name": "v0.1",
   "url": "https://microrobotica.org/maddening/v0.1/"}
]
```

The umbrella `conf.py` wires it into the theme:

```python
html_theme_options.update({
    "switcher": {
        "json_url": f"{_BASEURLS[project]}_static/switcher.json",
        "version_match": os.environ.get("DOCS_VERSION", "latest"),
    },
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
})
```

`DOCS_VERSION` is set by the CI build per tag — `latest` for the
`main` build, `v0.2` for the `v0.2.0` tag, etc.

## How a build produces all the versions

```{mermaid}
flowchart LR
    Tags["git tags<br/>v0.1.0 · v0.2.0 · ..."] --> Matrix
    Main["main branch"] --> Matrix
    Matrix["docs.yml matrix:<br/>{ tag in tags, 'main' }"] --> Build
    Build["checkout @ tag/main<br/>DOCS_VERSION=tag make all<br/>OUT=_build/html/{project}/{tag}/"] --> Merge
    Merge["merge into root _build/<br/>aliases: latest → newest tag"] --> Deploy
    Deploy["github-pages deploy"]
```

The umbrella `Makefile` already builds each project sequentially
(`make all` → `_build/html/{maddening,mime,microrobotica}/`).  The
multi-version extension is:

1. Loop over `git tag --list 'v*'`.
2. For each tag, `git worktree add /tmp/wt-$tag $tag`, then `cd
   docs && DOCS_VERSION=$tag make all OUTDIR=_build/html-$tag/`.
3. For `main`, the build is the same but with `DOCS_VERSION=latest`.
4. After all builds, symlink `_build/html/latest →
   _build/html-$latest_tag/` and copy each `_build/html-$tag/` into
   `_build/html/{project}/$tag/`.
5. Each project's root `switcher.json` lives at
   `_build/html/{project}/_static/switcher.json` and is the same
   file in every version's output (so the dropdown lists every
   version regardless of which one the user is currently on).

## Why this design over alternatives

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **sphinx-multiversion** | Automatic git-tag enumeration; single command. | Each historical tag must build with the *current* `conf.py`; breaking config changes cascade across history. | Rejected — we expect `conf.py` to evolve. |
| **Read-the-Docs** | Hosted, automatic. | Not the umbrella architecture; would need a separate domain per project. | Rejected — keep the unified microrobotica.org URL. |
| **Manual switcher.json + CI matrix** *(chosen)* | Each version builds against its own `conf.py`; switcher UI freezes whatever we ship; clean rollback. | Switcher JSON has to be updated for every release. | **Chosen.** |
| **Single-version + 404 page** | Cheapest. | Loses backwards-discoverable docs. | Rejected — v0.1 users still need their docs. |

## Rollout sequence

```{note}
The first three steps are MADDENING-side; the rest require
coordinated changes on MICROROBOTICA's umbrella repo.
```

1. **(this commit)** — `docs/_static/switcher.json` in each project
   lists `latest` + every released tag.  Today only the v0.2 row
   has a target (we haven't built v0.1 with this infra yet); v0.1
   resolves to a 404 until step 4.
2. **(this commit)** — Document the design in this page so the
   handoff is in version control, not in someone's head.
3. **(MICROROBOTICA umbrella PR)** — add `switcher` +
   `navbar_end` to `html_theme_options` in `conf.py`.  Set
   `DOCS_VERSION` from env var with `latest` default.
4. **(MICROROBOTICA umbrella PR)** — extend `Makefile` with a
   `versions` target that loops over `git tag` per project and
   builds each into the right subdirectory.
5. **(MICROROBOTICA docs.yml)** — switch the workflow from
   "build once" to "matrix of (project, tag), then merge into a
   single artifact".  Cache builds aggressively because most tags
   won't change between deploys.
6. **(release process)** — every release adds a new entry to its
   project's `switcher.json` and a new git tag.  The next CI run
   picks it up automatically.

## How a developer triggers a version-specific local build

Until the CI changes land, the per-version path is manual:

```bash
# Build v0.2 docs from a checkout at the v0.2.0 tag
git worktree add /tmp/maddening-v0.2 v0.2.0
cd ../MICROROBOTICA/docs
DOCS_VERSION=v0.2 make maddening   # → _build/html/maddening/

# Same for main → _build/html-main/
git worktree add /tmp/maddening-main main
DOCS_VERSION=latest make maddening
```

The switcher will appear in the navbar; clicking the v0.1 entry
404s until v0.1 is built into the right subdir.

## What goes in a new release's switcher.json

When cutting a new release N:

1. Update `docs/_static/switcher.json`:
   * Add a new entry for version N marked `"preferred": true`.
   * Flip the previous stable's `"preferred"` to `false`.
   * `latest` always points at `/<project>/` (the unversioned
     URL, which the CI maps to the newest tag).
2. Tag the release.
3. Push.
4. CI picks up the new tag, builds it into the right subdir, and
   the switcher picks it up automatically on the next umbrella
   build.

## Open questions

* **Where does MIME's switcher.json live?**  Same path
  (`MIME/docs/_static/switcher.json`).  Not yet authored — file a
  parallel issue.
* **Does MICROROBOTICA itself have versions?**  Yes — the IDE
  ships with the same versioning lifecycle.  Same scaffolding,
  different `_BASEURLS` row.
* **Do we want a `dev/` alias?**  Probably useful for users
  tracking unstable APIs.  Easy add: `/dev/` always maps to the
  most recent `main` build alongside the tag-pinned versions.
