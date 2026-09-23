#!/usr/bin/env python
"""Generate the SOUP evidence summary tables from their machine-readable sources.

``docs/validation/soup_package.md`` and
``docs/validation/framework_verification.md`` are the IEC 62304 SOUP /
MDCG evidence set.  Both used to carry hand-maintained summary tables
sitting next to the machine-readable file they summarise --
``known_anomalies.yaml`` and the ``@verification_benchmark`` registry --
and both had drifted from it (2 of 5 anomalies, 1 of 4 benchmarks, and
a version identification two to three releases stale).  For a
regulatory artifact "it drifted" is itself the finding, so the tables
are no longer written by hand.

Every table this script owns lives between a pair of markers::

    <!-- BEGIN GENERATED: known-anomalies -->
    ...
    <!-- END GENERATED: known-anomalies -->

Text outside those markers is prose and is left alone.

Sources
-------
``pyproject.toml``
    Version, licence, Python floor, base dependencies, build backend,
    repository URL -- everything in the Software Identification table.
``CITATION.cff``
    Release date (a pre-release version has none).
``docs/validation/known_anomalies.yaml``
    The anomaly registry.  This file is the source; the script only
    reads it.
``maddening.compliance.get_benchmark_registry()``
    The verification benchmark registry, populated by importing the
    test modules in ``BENCHMARK_MODULES``.
``tests/``
    Which test packages exist.
``.github/workflows/ci.yml``
    Python matrix, JAX pin and runner -- the configuration the evidence
    was produced on.

Usage
-----
::

    python scripts/generate_soup_tables.py            # rewrite in place
    python scripts/generate_soup_tables.py --check    # fail on drift

``--check`` is what ``tests/compliance/test_soup_evidence.py`` runs, so
CI fails on a stale table instead of shipping one.

Cross-file consistency is checked in both modes: the registry header
and ``CITATION.cff`` must name the version ``pyproject.toml`` names,
every ``affected_versions`` range must agree with its entry's status
about that version (``check_anomalies.version_range_errors``, the same
function the registry gate runs), and every test module registering a
``MADD-VER-`` benchmark must be in ``BENCHMARK_MODULES`` (otherwise it
would silently drop out of the index -- the drift this script exists to
stop).
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import tomllib
from pathlib import Path

# Force CPU JAX: importing the benchmark modules pulls JAX in, and this
# runs in the docs/compliance job, not on an accelerator.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT))  # so `tests.…` imports resolve

import yaml  # noqa: E402  (after sys.path setup)

# The registry gate owns the affected_versions rule and the set of
# statuses that count as "not reachable"; this script imports both rather
# than keeping a second copy that could drift (it did: this file tested
# ``startswith(">=")`` while the gate never read the field at all).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_anomalies as _anomaly_gate  # noqa: E402

SOUP_PACKAGE = REPO_ROOT / "docs" / "validation" / "soup_package.md"
FRAMEWORK_VERIFICATION = REPO_ROOT / "docs" / "validation" / "framework_verification.md"
ANOMALY_REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
CITATION = REPO_ROOT / "CITATION.cff"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TESTS = REPO_ROOT / "tests"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: Test modules whose import populates the verification benchmark
#: registry.  ``_check_benchmark_modules_are_complete`` scans ``tests/``
#: for ``benchmark_id="MADD-VER-`` and fails if one is missing here, so
#: a newly registered benchmark cannot quietly vanish from the index.
BENCHMARK_MODULES: tuple[str, ...] = (
    "tests.verification.test_heat_analytical",
    "tests.verification.test_mms_order",
    "tests.verification.test_mms_order_ode_nodes",
    "tests.verification.test_gci_order",
    "tests.cloud.multigpu.test_lbm_poiseuille",
    "tests.nodes.adaptive.test_verification",
    "tests.verification.test_wavelet_mms_order",
)

#: Prefix that marks a benchmark as MADDENING's own (CONTRIBUTING.md).
#: ``tests/compliance/test_validation_benchmark.py`` registers
#: ``TEST-VER-*`` entries into the same global registry while it runs,
#: so the filter is what makes the generated table order-independent.
BENCHMARK_PREFIX = "MADD-VER-"

#: The ``resolution_status`` values whose defect cannot reach the version
#: the registry describes -- the complement of the §3 headline's
#: "reachable" count.  Imported from the registry gate, which uses the
#: same set to decide which ranges must admit that version, so the
#: headline and the range rule cannot disagree.  An unrecognised status
#: is outside it and so counts as reachable; it is also refused outright
#: by the schema validator (``maddening.compliance._validate``), which
#: enum-checks ``resolution_status`` against ``ResolutionStatus``.
_UNREACHABLE_STATUSES = _anomaly_gate.UNREACHABLE_STATUSES

#: What each top-level package under ``tests/`` covers.  Which packages
#: exist comes from the tree, and
#: ``_check_test_directories_are_described`` fails if a package appears
#: with no entry here -- the old hand-written table listed 7 of 12.
TEST_DIRECTORY_SCOPE: dict[str, str] = {
    "adaptive": "Adaptive timestepping: Richardson extrapolation, PI controller",
    "api": "FastAPI server, WebSocket, binary encoding, server-side rendering",
    "cloud": "Distributed execution: multi-GPU sharding, halo exchange, "
             "resume transport, checkpoint download",
    "compliance": "Compliance infrastructure: metadata, anomaly validator, "
                  "stability decorator, benchmark registry, provenance",
    "core": "Core framework: GraphManager, scheduling, coupling, params, "
            "checkpoint, sweep, solver utilities",
    "fmi": "FMI/FMU export and import: model description, binary frames, "
           "parameter variables",
    "nodes": "Physics node correctness: HeatNode, LBMNode, LBMPipeNode, "
             "RigidBody2DNode, SpringDamperNode, AdaptiveNode",
    "property": "Hypothesis property tests over generated graphs, meshes and "
                "coupling configurations",
    "security": "Transport authentication: ZMQ CURVE encryption and "
                "key derivation for the state, command and coordinator sockets",
    "surrogates": "Neural {term}`surrogate <Surrogate>` training, "
                  "architectures, dataset generation",
    "usd": "USD stage serialization and round-trips, geometry sources, "
           "interface mappings",
    "verification": "Registered verification benchmarks (analytical "
                    "comparisons, convergence studies)",
    "viz": "Visualization backends, ZMQ transport, serialization",
}

#: Facts about the package that pyproject does not carry.
FULL_NAME = (
    "Modular Automatic Differentiation and Data Enhanced "
    "Neural-network INteracting Graph"
)
INSTALL_COMMAND = "`pip install maddening`"

_PRERELEASE_RE = re.compile(r"(?:a|b|rc|\.dev|\.post)\d*$")
_JAX_PIN_RE = re.compile(r"\bjax==([0-9][^\"\'\s]*)")


# --------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------


def read_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def read_registry() -> dict:
    with ANOMALY_REGISTRY.open() as fh:
        return yaml.safe_load(fh)


def read_citation() -> dict:
    with CITATION.open() as fh:
        return yaml.safe_load(fh)


def _matrix_axis(matrix: dict, key: str) -> list[str]:
    """Every value one matrix axis takes, in workflow order.

    A GitHub matrix declares an axis either as a top-level list
    (``python-version: ["3.12"]``, crossed with the other axes) or as
    per-leg entries under ``include:``.  Reading only the first form is
    how a matrix rewrite turns into a silently empty evidence table, so
    both are read and their union returned.
    """
    values: list[str] = []
    for value in matrix.get(key) or []:
        if str(value) not in values:
            values.append(str(value))
    for leg in matrix.get("include") or []:
        if isinstance(leg, dict) and key in leg and str(leg[key]) not in values:
            values.append(str(leg[key]))
    return values


def read_ci() -> dict:
    """Python versions, JAX pins and runner, read out of the CI workflow.

    The old hand-written page said "Python: 3.12" and "JAX: 0.4+" while
    CI ran 3.11 and 3.12 against a jax==0.10.2 pin.  Nothing about the
    verified configuration is retyped here.

    Both axes of the ``test`` matrix are read.  The JAX pins are the
    union of the literal ``jax==X`` the single-lane jobs install and the
    ``jax-version`` axis the matrix lanes install from: the regex alone
    stopped seeing the test lanes the moment their pin became
    ``jax==${{ matrix.jax-version }}``, which would have dropped a
    verified point out of the SOUP package without failing anything.
    """
    with CI_WORKFLOW.open() as fh:
        workflow = yaml.safe_load(fh)

    jobs = workflow.get("jobs") or {}
    # ``or {}`` at every hop: a deleted job, strategy or matrix
    # parses as ``None``, and a traceback here is neither the
    # answer nor the honest "unknown" the rows are built to print.
    test_job = jobs.get("test") or {}
    matrix = (test_job.get("strategy") or {}).get("matrix") or {}
    runners = sorted({
        job["runs-on"] for job in jobs.values()
        if isinstance(job, dict) and isinstance(job.get("runs-on"), str)
    })
    pins = set(_JAX_PIN_RE.findall(CI_WORKFLOW.read_text()))
    pins.update(_matrix_axis(matrix, "jax-version"))
    return {
        "pythons": _matrix_axis(matrix, "python-version"),
        "runners": runners,
        "jax_pins": sorted(pins),
    }


def jax_requirement(pyproject: dict) -> str:
    """The declared ``jax`` requirement, verbatim from ``pyproject.toml``.

    ``jaxlib>=...`` does not start with ``jax>``, so this picks the
    framework requirement and not the runtime one.
    """
    return next(
        (d for d in pyproject["project"]["dependencies"] if d.startswith("jax>")),
        "",
    )


def ci_pythons(ci: dict) -> str:
    """The interpreters CI runs, or an explicit unknown.

    ``requires-python`` is what pip permits; this is what ran.  An empty
    matrix prints the unknown rather than an empty string next to the
    word "verified", which reads as a claim and is not one -- the same
    fail-closed rule ``verified_jax`` applies to the pin.
    """
    if not ci["pythons"]:
        return "**unknown** (no `python-version` matrix found in the CI workflow)"
    return ", ".join(ci["pythons"])


def verified_pythons(ci: dict) -> str:
    """The Python half of the Software Identification row."""
    if not ci["pythons"]:
        return f"verified point {ci_pythons(ci)}"
    return f"verified on {ci_pythons(ci)} (the CI matrix)"


def jax_evidence(ci: dict, jax_spec: str) -> str:
    """The JAX row of the test-suite table: what ran, never what is allowed."""
    pins = ci["jax_pins"]
    if not pins:
        return (
            "**unknown** — no `jax==` pin found in the CI workflow, so what "
            "the evidence was generated against is not recorded; "
            f"`{jax_spec}` is the *declared* range and is not evidence"
        )
    joined = ", ".join(f"`{v}`" for v in pins)
    installs = (
        "the only version CI installs" if len(pins) == 1
        else f"the {len(pins)} versions CI installs"
    )
    return (
        f"evidence generated at {joined}, {installs}; `{jax_spec}` is the "
        "*declared* range and no other point in it has been exercised"
    )


def verified_jax(ci: dict) -> str:
    """What CI actually installed, phrased so it cannot overclaim.

    The declared range is exercised at whatever points CI pins, which is
    one.  A range is not evidence for the points inside it that nobody
    ran, so this never says "verified on 0.10-0.13"; and if the pin
    cannot be found at all the row says **unknown** rather than falling
    back to the range, because a SOUP document that silently substitutes
    a permitted range for a verified one is the exact defect this row
    exists to avoid.
    """
    pins = ci["jax_pins"]
    if not pins:
        return "**verified point unknown** (no `jax==` pin found in the CI workflow)"
    if len(pins) == 1:
        return f"verified at {pins[0]} (the only version CI installs)"
    joined = ", ".join(pins)
    return f"verified at {joined} (the versions CI installs)"


def package_version(pyproject: dict) -> str:
    return str(pyproject["project"]["version"])


def is_prerelease(version: str) -> bool:
    return bool(_PRERELEASE_RE.search(version))


def load_benchmarks() -> dict:
    """Import the benchmark-bearing test modules and return the registry."""
    import importlib

    for module in BENCHMARK_MODULES:
        importlib.import_module(module)

    from maddening.compliance import get_benchmark_registry

    return {
        bid: bm
        for bid, bm in get_benchmark_registry().items()
        if bid.startswith(BENCHMARK_PREFIX)
    }


def test_packages() -> list[str]:
    """Return the names of the packages under ``tests/`` that hold tests.

    Membership only.  A file *count* per package would be exact and
    would also make every PR that adds a test file conflict on this
    document -- the CHANGELOG lesson, applied to a table.  What drifted
    was which packages were listed at all (7 of 12); that is what is
    pinned.  A per-run test count belongs in the CI job log and in the
    release notes, which are written once at release time.
    """
    out = []
    for child in sorted(TESTS.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if any(child.rglob("test_*.py")):
            out.append(child.name)
    return out


# --------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------


def _cell(text: object) -> str:
    """Render a value as a Markdown table cell."""
    s = " ".join(str(text).split())
    return s.replace("|", "\\|")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def _enum(value: object) -> str:
    """Render a schema enum verbatim, in backticks.

    Prettifying (``context_dependent`` -> "Context-dependent") makes the
    cell un-greppable against the YAML it came from, which is the one
    thing a reader checking this document against the registry wants to
    do.  The value is printed as the schema spells it.
    """
    if value in (None, ""):
        return "—"
    return f"`{value}`"


def render_software_identification(pyproject: dict, citation: dict,
                                   ci: dict) -> str:
    project = pyproject["project"]
    version = package_version(pyproject)

    if is_prerelease(version):
        release_date = "unreleased (development build)"
    else:
        released = citation.get("date-released")
        release_date = str(released) if released else "—"

    deps = ", ".join(project.get("dependencies", ()))
    backend = pyproject.get("build-system", {}).get("build-backend", "")
    repo = project.get("urls", {}).get("Repository", "")

    rows = [
        ["Name", "MADDENING"],
        ["Full Name", FULL_NAME],
        ["Version", version],
        ["Release Date", release_date],
        ["Licence", project.get("license", "")],
        ["Source Repository", repo],
        # The floor is what pip permits; it is not what was tested, and this
        # is the identification table a downstream reader treats as "what
        # this software is".  State both, the way the verification table
        # already does for JAX.
        ["Python Version", f'{project.get("requires-python", "")} permitted; '
                           f'{verified_pythons(ci)}'],
        # Same treatment as the Python row above, and for the same
        # reason.  ``Base Dependencies`` below states `jax>=0.10,<0.13`,
        # which is what pip permits; read as the verified configuration
        # it claims twenty-odd untested releases.  Both CI lanes pin one
        # version, so the declared range is exercised at exactly one
        # point, and that point -- not the range -- is the evidence.
        ["JAX Version", f'{jax_requirement(pyproject)} permitted; '
                        f'{verified_jax(ci)}'],
        ["Base Dependencies", deps],
        ["Build System", backend.split(".")[0] if backend else ""],
        ["Install", INSTALL_COMMAND],
    ]
    return _table(["Field", "Value"], rows)


def render_known_anomalies(registry: dict) -> str:
    rows = []
    for a in registry.get("anomalies", []):
        status = _enum(a.get("resolution_status"))
        resolved_in = a.get("resolution_version")
        if resolved_in:
            status = f"{status} (in {resolved_in})"
        rows.append([
            a.get("anomaly_id", "—"),
            a.get("title", "—"),
            _enum(a.get("severity")),
            _enum(a.get("safety_relevance")),
            status,
            a.get("affected_versions", "—"),
        ])
    table = _table(
        ["ID", "Title", "Severity", "Safety Relevance", "Status",
         "Affected Versions"],
        rows,
    )
    n = len(rows)
    # Two counts, because they answer different questions and the
    # headline used to give only the first.  ``open`` is a lifecycle
    # state: how many entries nobody has closed.  What a reader of a
    # SOUP document needs is how many known defects can reach them in
    # the version they are running, and ``partially_resolved`` entries
    # are in that set -- MADD-ANO-005's estimate still falls back to the
    # pre-0.4.0 residual test on a reachable path, and MADD-ANO-014's
    # own residual risk says "the degraded path is still the default and
    # still silent".  Counting only ``open`` left every
    # ``partially_resolved`` entry out of the total, and under-reported,
    # which is the dangerous direction.
    #
    # "Reachable" is deliberately the *same* predicate as the registry
    # gate's range rule (``check_anomalies.version_range_errors``): a
    # reachable entry's ``affected_versions`` must admit the version the
    # registry describes.  Both read ``_UNREACHABLE_STATUSES``, so the
    # headline and the range gate cannot come to disagree, and an
    # unrecognised status counts as reachable rather than being quietly
    # dropped from the total.
    #
    # The breakdown names every reachable status it counts.  It used to
    # print "N `open` plus M `partially_resolved`" with M computed as
    # everything reachable that was not open, which would have labelled
    # a `wont_fix` entry `partially_resolved`.
    reachable_by_status: dict[str, int] = {}
    for a in registry.get("anomalies", []):
        status = a.get("resolution_status")
        if status in _UNREACHABLE_STATUSES:
            continue
        key = str(status)
        reachable_by_status[key] = reachable_by_status.get(key, 0) + 1
    n_reachable = sum(reachable_by_status.values())
    order = ["open", "partially_resolved", "wont_fix"]
    order += sorted(set(reachable_by_status) - set(order))
    parts = []
    for status in order:
        count = reachable_by_status.get(status, 0)
        if status in ("open", "partially_resolved") or count:
            part = f"{count} `{status}`"
            if status == "partially_resolved":
                part += " whose residual risk is still live"
            parts.append(part)
    return (
        f"{table}\n\n"
        f"*{n} anomalies registered.  {n_reachable} have a defect reachable "
        f"in this version — every entry whose `resolution_status` is not "
        f"`resolved` or `duplicate`, which is {' plus '.join(parts)}.  "
        f"The Affected Versions column is a PEP 440 specifier set read "
        f"against this document's version; `{_anomaly_gate.EMPTY_RANGE}` "
        f"marks a defect "
        f"introduced and fixed within one development cycle, which no "
        f"release carried.  The convention, and the gate that holds every "
        f"range to it, are in the header of `known_anomalies.yaml`.  "
        f"Rationale, workaround, affected components and "
        f"verification evidence for each: `known_anomalies.yaml`.*"
    )


def render_benchmarks(benchmarks: dict) -> str:
    rows = []
    for bid, bm in sorted(benchmarks.items()):
        test = bm.test_function or "—"
        rows.append([
            bid,
            bm.node_type,
            _enum(bm.benchmark_type.value),
            bm.acceptance_criteria,
            f"`{test}`",
        ])
    table = _table(
        ["Benchmark ID", "Node", "Type", "Acceptance Criteria", "Test"],
        rows,
    )
    return f"{table}\n\n*{len(rows)} benchmarks registered.*"


def render_test_organization(packages: list[str]) -> str:
    rows = [
        [f"`tests/{name}/`", TEST_DIRECTORY_SCOPE.get(name, "—")]
        for name in packages
    ]
    return _table(["Directory", "Scope"], rows)


def render_test_suite(pyproject: dict, packages: list[str], ci: dict) -> str:
    jax_spec = jax_requirement(pyproject)
    # Which base dependencies CI actually pins, and which float.  The
    # row used to read "`jax>=0.10,<0.13` supported", which asserts a
    # range on evidence taken at one point inside it, and nothing said
    # anything at all about the other base dependencies -- CI installs
    # them from their ranges, so the resolved version differs run to
    # run and is recorded nowhere.  A SOUP document that cannot say what
    # its evidence was generated against is missing the part IEC 62304
    # cares most about, so this now says what is pinned, what is not,
    # and that the unpinned ones are unrecorded rather than assumed.
    pinned_names = {"jax", "jaxlib"}
    floating = [
        d for d in pyproject["project"]["dependencies"]
        if re.split(r"[<>=!~\[]", d, 1)[0].strip() not in pinned_names
    ]
    rows = [
        ["Test runner", "pytest"],
        ["CI system", "GitHub Actions"],
        ["CI runners", ", ".join(f"`{r}`" for r in ci["runners"])],
        ["Python versions", ci_pythons(ci) + " (floor: "
                            f"{pyproject['project'].get('requires-python', '')})"],
        ["JAX", jax_evidence(ci, jax_spec)],
        ["Other base dependencies", ", ".join(f"`{d}`" for d in floating)
                                    + " — installed from these ranges, "
                                      "not pinned, so the resolved version "
                                      "differs between runs and **is not "
                                      "recorded**"],
        ["Backend", "CPU (GPU tests are not run in CI — MADD-ANO-001)"],
        ["Test packages", f"{len(packages)} — listed below"],
    ]
    return _table(["Field", "Value"], rows)


# --------------------------------------------------------------------
# Consistency checks
# --------------------------------------------------------------------


def _check_versions(pyproject: dict, registry: dict, citation: dict) -> list[str]:
    version = package_version(pyproject)
    errors = []
    declared = str(registry.get("maddening_version", ""))
    if declared != version:
        errors.append(
            f"{ANOMALY_REGISTRY.relative_to(REPO_ROOT)}: maddening_version is "
            f"{declared!r}, but pyproject.toml says {version!r}"
        )
    cited = str(citation.get("version", ""))
    if cited != version:
        errors.append(
            f"{CITATION.relative_to(REPO_ROOT)}: version is {cited!r}, but "
            f"pyproject.toml says {version!r}"
        )
    if is_prerelease(version) and citation.get("date-released"):
        errors.append(
            f"{CITATION.relative_to(REPO_ROOT)}: date-released is set while "
            f"the version ({version}) is a pre-release, which has no release "
            f"date; drop the field until the release is tagged"
        )
    return errors


def _check_version_ranges(registry: dict) -> list[str]:
    """Every ``affected_versions`` must agree with its entry's status.

    The rule and its PEP 440 convention live in
    ``scripts/check_anomalies.py`` (``version_range_errors``), which CI's
    compliance job runs; this calls the same function so the SOUP table
    cannot print a range the registry gate would refuse.

    It replaced ``_check_unresolved_anomalies_are_open_ended``, whose test
    was ``affected.startswith(">=")``: ``">=0.1.0, <0.4.0"`` starts with
    ``>=`` and closes the range, so MADD-ANO-016 (``partially_resolved``)
    shipped a claim that 0.4.0 was unaffected beside a footer counting it
    as reachable.  The semantic check compares each range with the
    registry's own ``maddening_version`` instead: a reachable entry's
    range must admit it, a ``resolved`` entry's must not, and an
    unparseable range fails.  MADD-ANO-001 and MADD-ANO-002, which sat
    ``open`` with a closed range for three releases, and MADD-ANO-005,
    ``partially_resolved`` with ``<=0.3.0``, are the history it exists for.
    """
    return _anomaly_gate.version_range_errors(registry)


def _check_test_directories_are_described(packages: list[str]) -> list[str]:
    """Every test package must have a scope description, and vice versa."""
    errors = []
    for name in packages:
        if name not in TEST_DIRECTORY_SCOPE:
            errors.append(
                f"tests/{name}/ holds tests but has no entry in "
                f"TEST_DIRECTORY_SCOPE in {Path(__file__).name}; add one so it "
                f"appears in framework_verification.md"
            )
    for name in TEST_DIRECTORY_SCOPE:
        if name not in packages:
            errors.append(
                f"TEST_DIRECTORY_SCOPE describes tests/{name}/, which holds no "
                f"test files; drop the entry"
            )
    return errors


def _check_benchmark_modules_are_complete() -> list[str]:
    """Every test module registering a MADD-VER benchmark must be listed."""
    listed = {m.replace(".", "/") + ".py" for m in BENCHMARK_MODULES}
    errors = []
    for path in sorted(TESTS.rglob("test_*.py")):
        if 'benchmark_id="' + BENCHMARK_PREFIX not in path.read_text():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel not in listed:
            errors.append(
                f"{rel} registers a {BENCHMARK_PREFIX} benchmark but is not in "
                f"BENCHMARK_MODULES in {Path(__file__).name}, so it would be "
                f"missing from framework_verification.md"
            )
    return errors


# --------------------------------------------------------------------
# Marker splicing
# --------------------------------------------------------------------


def _marker_re(name: str) -> re.Pattern:
    return re.compile(
        r"(?P<begin><!-- BEGIN GENERATED: " + re.escape(name) + r"[^>]*-->\n)"
        r"(?P<body>.*?)"
        r"(?P<end><!-- END GENERATED: " + re.escape(name) + r" -->)",
        re.DOTALL,
    )


def splice(text: str, name: str, body: str, path: Path) -> str:
    pattern = _marker_re(name)
    if not pattern.search(text):
        raise SystemExit(
            f"{path.relative_to(REPO_ROOT)}: no "
            f"'<!-- BEGIN GENERATED: {name} -->' / "
            f"'<!-- END GENERATED: {name} -->' pair found"
        )
    # A function replacement is used deliberately: `re.sub` does not
    # process escapes in what a function returns, so a cell containing
    # `\|` survives intact.
    return pattern.sub(
        lambda m: m.group("begin") + body + "\n" + m.group("end"),
        text,
        count=1,
    )


def build() -> tuple[dict[Path, str], list[str]]:
    """Return the desired content of each owned file, and any errors."""
    pyproject = read_pyproject()
    registry = read_registry()
    citation = read_citation()
    benchmarks = load_benchmarks()
    packages = test_packages()
    ci = read_ci()

    errors = (
        _check_versions(pyproject, registry, citation)
        + _check_version_ranges(registry)
        + _check_test_directories_are_described(packages)
        + _check_benchmark_modules_are_complete()
    )

    soup = SOUP_PACKAGE.read_text()
    soup = splice(soup, "software-identification",
                  render_software_identification(pyproject, citation, ci),
                  SOUP_PACKAGE)
    soup = splice(soup, "known-anomalies",
                  render_known_anomalies(registry), SOUP_PACKAGE)

    fv = FRAMEWORK_VERIFICATION.read_text()
    fv = splice(fv, "test-suite",
                render_test_suite(pyproject, packages, ci),
                FRAMEWORK_VERIFICATION)
    fv = splice(fv, "test-organization",
                render_test_organization(packages), FRAMEWORK_VERIFICATION)
    fv = splice(fv, "verification-benchmarks",
                render_benchmarks(benchmarks), FRAMEWORK_VERIFICATION)

    return {SOUP_PACKAGE: soup, FRAMEWORK_VERIFICATION: fv}, errors


#: Any MADD-* evidence identifier, for naming what a drift would remove.
_EVIDENCE_ID = re.compile(r"MADD-[A-Z]+-\d+")


def describe_drift(rel, current: str, generated: str) -> str:
    """Say what *kind* of difference this is, not just that there is one.

    The remedy is opposite for the two kinds, and this message used to
    prescribe one of them for both: *"run
    ``python scripts/generate_soup_tables.py`` and commit the result"*.  That
    is right when the committed document is behind its source.  It is exactly
    wrong when the committed document has a row the source no longer
    produces, because then regenerating rewrites the document from a source
    that has lost an entry -- the two agree again, every gate goes green, and
    the entry is gone from the IEC 62304 evidence set.  That is how deleting
    an open anomaly from ``known_anomalies.yaml`` passed the whole compliance
    suite (audit_040_r2/gates, finding G5): the instruction in this very
    message was the last step of the defect.

    So: count lines only in the committed file (the source dropped them),
    lines only in the generated file (the document is behind), and lines
    changed in place, and lead with the dangerous one when it is present.

    The classification cannot be read off the ``difflib`` opcodes alone.
    ``SequenceMatcher`` emits ``delete`` only when a surviving line
    anchors the deletion on both sides; a row deleted next to a row whose
    text also changed comes back as one ``replace``, whose old lines this
    function used to file entirely under "changed in place".  Deleting
    ``MADD-ANO-012`` and rewording the anomaly after it therefore printed
    *"no committed row would be lost.  Run generate_soup_tables.py"* with
    the ``-| MADD-ANO-012 | ... | open |`` row in the diff below it
    (audit_040_r3, the residual half of G5) -- and a deletion adjacent to
    an edit is the shape of every "re-derive the registry" commit in this
    release.  So the decision is made on the *evidence IDs*: an ID the
    committed document carries and a fresh generation does not is a lost
    row whatever opcode it arrives in.
    """
    old_lines = current.splitlines(keepends=True)
    new_lines = generated.splitlines(keepends=True)

    def _ids(text: str) -> set[str]:
        return {m.group(0) for m in _EVIDENCE_ID.finditer(text)}

    #: IDs the committed document has and the regenerated one would not.
    lost_ids = _ids(current) - _ids(generated)

    removed: list[str] = []
    added: list[str] = []
    changed: list[str] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            removed.extend(old_lines[i1:i2])
        elif tag == "insert":
            added.extend(new_lines[j1:j2])
        elif tag == "replace":
            # An old line inside a `replace` is only "changed in place" if
            # nothing it identifies is disappearing from the document.
            for line in old_lines[i1:i2]:
                (removed if _ids(line) & lost_ids else changed).append(line)

    lines = [
        f"{rel} does not match a fresh generation from its sources:",
        f"  {len(removed)} line(s) only in the COMMITTED file "
        f"(the source no longer produces them)",
        f"  {len(added)} line(s) only in the GENERATED file "
        f"(the source has them, the document does not)",
        f"  {len(changed)} line(s) changed in place",
        "",
    ]

    if removed or lost_ids:
        # `or lost_ids` is a backstop and is currently unreachable on its
        # own: every old line outside an `equal` block lands in `delete`
        # or `replace`, and both now route a line carrying a lost ID into
        # `removed`.  It is kept because it holds the invariant directly
        # -- an ID the committed document has and the generated one does
        # not is a lost row -- so a future change to the opcode loop
        # cannot quietly reintroduce the defect.  A mutation that removes
        # this clause alone is therefore NOT caught by the test suite;
        # one that breaks the opcode loop is (see
        # TestDriftIsClassified).  `removed` still contributes the names,
        # so a lost row that carries no MADD-* ID at all is reported too.
        lost = sorted(lost_ids
                      | {m.group(0) for line in removed
                         for m in _EVIDENCE_ID.finditer(line)})
        named = f" ({', '.join(lost)})" if lost else ""
        lines += [
            f"DO NOT regenerate yet.  Regenerating would DELETE those "
            f"committed rows{named} from the evidence set, and the gate "
            f"would then pass.",
            "Establish why the source lost them first: an entry removed from "
            "docs/validation/known_anomalies.yaml, a @verification_benchmark "
            "decorator deleted, or two decorators sharing a benchmark_id.  "
            "Restore it; if it was genuinely retired, record the retirement "
            "explicitly (see tests/compliance/test_soup_evidence.py) rather "
            "than by deletion.",
        ]
    else:
        lines += [
            "The committed document is behind its source; no committed row "
            "would be lost.  Run `python scripts/generate_soup_tables.py` and "
            "commit the result.",
        ]

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"{rel} (committed)",
        tofile=f"{rel} (generated)",
    )
    return "\n".join(lines) + "\n\n" + "".join(diff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="exit non-zero if a committed file differs from a fresh "
             "generation, printing the diff; write nothing",
    )
    args = parser.parse_args()

    wanted, errors = build()

    failures = list(errors)
    for path, content in wanted.items():
        current = path.read_text()
        if current == content:
            continue
        rel = path.relative_to(REPO_ROOT)
        if args.check:
            failures.append(describe_drift(rel, current, content))
        else:
            path.write_text(content)
            print(f"wrote {rel}")

    if failures:
        for f in failures:
            print(f"ERROR: {f}", file=sys.stderr)
        return 1

    print("OK: SOUP evidence tables match their sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
