"""The draw-rejection audit has to be able to fail, and its model has to hold.

``scripts/audit_property_rejection.py`` measures what fraction of each
property test's Hypothesis draws is thrown away by ``assume()`` or by a
strategy ``.filter()``.  It exists because that fraction is otherwise
invisible: a test rejecting 5% of its draws and one rejecting 85% print the
same green tick and the same ``max_examples``, and the only thing that ever
says otherwise is ``HealthCheck.filter_too_much`` -- which is a sampling test
on the first few dozen draws, so it stays quiet for months and then fires for
whoever next narrows a strategy.

That happened on this tree.
``test_coupling_acceleration_agreement.py::test_accelerating_every_field_lands_on_the_same_answer_as_plain_iteration``
went red in CI with "9 inputs generated successfully, 50 filtered out" after a
change to an unrelated strategy, and the audit that followed found it had
been discarding most of its draws all along.

So this file tests the auditor, not the audited:

* the gate can fail (a synthetic over-budget test is caught, end to end,
  through the real CLI);
* the accounting is right, including the statuses that are *not* rejections;
* the health-check risk model matches both an independent computation and
  the installed Hypothesis *driven until its health checks fire*, so it
  cannot go stale silently.

Where the gate itself runs is ``.github/workflows/ci.yml``
(``verify-hypothesis``), which audits while it runs the suite it already ran.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import HealthCheck, Phase, assume, given, settings
from hypothesis import strategies as st
from hypothesis.errors import FailedHealthCheck, Unsatisfiable
from hypothesis.statistics import collector

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_property_rejection.py"


def _load_audit():
    """Import the audit script as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_audit_property_rejection", AUDIT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: ``@dataclass`` resolves annotations through
    # ``sys.modules[cls.__module__]`` and raises without it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


# ---------------------------------------------------------------------------
# The accounting
# ---------------------------------------------------------------------------
def _stats(**phases):
    """A minimal stand-in for ``ConjectureRunner.statistics``."""
    out = {"stopped-because": "settings.max_examples=200"}
    for phase, statuses in phases.items():
        out[f"{phase}-phase"] = {
            "duration-seconds": 0.0,
            "distinct-failures": 0,
            "shrinks-successful": 0,
            "test-cases": [{"status": s, "runtime": 0.0, "drawtime": 0.0,
                            "events": []} for s in statuses],
        }
    return out


def test_the_filter_rate_counts_invalid_draws_only(audit):
    """``invalid`` is what ``assume()`` and ``.filter()`` produce, and what
    ``HealthCheck.filter_too_much`` counts."""
    rec = audit.record_from_statistics(
        "t", _stats(generate=["valid"] * 3 + ["invalid"] * 5)
    )
    assert rec.drawn == 8
    assert rec.rejected == 5
    assert rec.effective_examples == 3
    assert rec.rate == pytest.approx(5 / 8)


def test_an_overrun_is_reported_apart_from_the_filter_rate(audit):
    """The distinction the first cut of this harness got wrong.

    An overrun is Hypothesis running out of entropy for a big draw; it has
    its own budget and its own health check (``data_too_large``), and the
    test did not reject anything.  Measured before this split, every
    ``hypothesis.extra.numpy.arrays`` property in
    ``tests/verification/hypothesis/test_hypothesis_coupling.py`` read as
    ~10% "rejection" with no ``assume`` written anywhere in the file.  Fold
    overruns into the filter rate and the table blames the author for the
    size of their arrays, and the real offenders sink into the noise.
    """
    rec = audit.record_from_statistics(
        "t", _stats(generate=["valid"] * 9 + ["overrun"])
    )
    assert rec.rate == 0.0, "an overrun is not a filtered draw"
    assert rec.overrun_rate == pytest.approx(0.1)
    assert rec.drawn == 10


def test_a_failing_example_is_work_done_not_a_rejection(audit):
    """``interesting`` reached the body and falsified it.

    Counting it as a rejection would make every genuinely failing test read
    as 100% rejection and bury the real offenders in the table.
    """
    rec = audit.record_from_statistics("t", _stats(generate=["valid", "interesting"]))
    assert rec.rejected == 0
    assert rec.rate == 0.0


def test_shrink_draws_do_not_count_against_the_budget(audit):
    """Shrinking only happens for an already-failing test, and its draws are
    not examples -- most of them are *meant* to be invalid."""
    rec = audit.record_from_statistics(
        "t", _stats(generate=["valid"] * 4, shrink=["invalid"] * 100)
    )
    assert rec.drawn == 4
    assert rec.rate == 0.0


def test_replayed_database_examples_count(audit):
    """The ``reuse`` phase replays saved failures; they are draws like any
    other and a test that rejects them is rejecting real work."""
    rec = audit.record_from_statistics(
        "t", _stats(reuse=["valid", "invalid"], generate=["valid"] * 2)
    )
    assert rec.drawn == 4
    assert rec.rejected == 1
    assert rec.rate == pytest.approx(0.25)


def test_several_engine_runs_under_one_node_id_accumulate(audit):
    """A test body that calls more than one ``@given`` function reports twice."""
    rec = audit.record_from_statistics("t", _stats(generate=["valid", "invalid"]))
    rec = audit.record_from_statistics("t", _stats(generate=["invalid"] * 3), into=rec)
    assert rec.runs == 2
    assert rec.drawn == 5
    assert rec.rejected == 4
    assert rec.rate == pytest.approx(0.8)


def test_a_starved_run_is_flagged(audit):
    """Below ~1% valid the engine stops early, and the test really did run
    fewer examples than it asked for.  That must not read as a clean pass."""
    stats = _stats(generate=["valid"] * 2 + ["invalid"] * 500)
    stats["stopped-because"] = (
        "settings.max_examples=200, but < 1% of test cases satisfied assumptions"
    )
    rec = audit.record_from_statistics("t", stats)
    assert rec.starved
    assert rec.as_dict()["starved"]


def test_a_test_that_drew_nothing_is_not_a_division_by_zero(audit):
    rec = audit.record_from_statistics("t", _stats())
    assert rec.rate == 0.0
    assert rec.overrun_rate == 0.0


def test_a_run_that_only_overran_does_not_report_a_filter_rate(audit):
    """Every draw discarded, none of them filtered: the denominator of the
    filter rate is empty, and 0/0 must read as "nothing filtered"."""
    rec = audit.record_from_statistics("t", _stats(generate=["overrun"] * 5))
    assert rec.rate == 0.0
    assert rec.overrun_rate == 1.0


def test_the_overrun_gate_is_separate_and_can_fire_on_its_own(audit):
    """A test that filters nothing but overruns half its draws is still
    a test whose search is being eaten, and ``data_too_large`` will
    eventually say so."""
    plugin = audit.RejectionAuditPlugin()
    plugin.records["t"] = audit.record_from_statistics(
        "t", _stats(generate=["valid"] * 4 + ["overrun"] * 6))
    assert plugin.over_budget() == list(plugin.records.values())
    assert plugin.over_budget(max_overrun=0.9) == []


# ---------------------------------------------------------------------------
# What configuration the measurement was taken in
# ---------------------------------------------------------------------------
def _fake_checkout(root, *parts):
    probe = root.joinpath(*parts, "src", "maddening", "__init__.py")
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("VERSION = '0.0.0'\n")
    return root.joinpath(*parts)


def test_the_audit_says_whether_constant_injection_was_on(audit, tmp_path):
    """A rate measured with constant injection off is a different number.

    Since the 6.16x line Hypothesis harvests the literals out of every local
    module and injects them into draws, which changes the value distribution
    and so can change a rejection rate.  Whether it is on depends on the
    **path the tree sits at**: ``is_local_module_file`` excludes any path
    with a ``test`` or ``tests`` component, so a git worktree under
    ``MADDENING-wt/test/<branch>/`` has the whole of ``src/`` classified as
    test files and injection silently off -- while CI, at
    ``/home/runner/work/MADDENING/MADDENING``, has it on.

    Measured: 0 local constants in such a worktree against 497 at a path
    without the component, same commit.  That is a difference between the
    measurement and the thing measured, and the only defence is for every
    run to print which side it was on.
    """
    clean = _fake_checkout(tmp_path, "workspace", "maddening")
    assert "ON" in audit.describe_constant_injection(clean)

    for component in ("test", "tests"):
        shadowed = _fake_checkout(tmp_path, component, "branch")
        message = audit.describe_constant_injection(shadowed)
        assert "OFF" in message, message
        assert repr(component) in message, message


def test_the_configuration_line_never_takes_the_gate_down(audit, tmp_path):
    """It reads Hypothesis internals, so it has to fail soft.

    The whole point of this branch is that a gate must not go red because a
    library moved something.  This line is a diagnostic; if it cannot be
    computed it says so and the audit still runs.
    """
    assert audit.describe_constant_injection(tmp_path / "nope")
    assert audit.describe_constant_injection(audit.REPO_ROOT)

    import builtins
    real_import = builtins.__import__

    def explode(name, *a, **k):
        if "constants_ast" in name or "conjecture" in name:
            raise ImportError("pretend hypothesis moved it")
        return real_import(name, *a, **k)

    builtins.__import__ = explode
    try:
        message = audit.describe_constant_injection(audit.REPO_ROOT)
    finally:
        builtins.__import__ = real_import
    assert "unknown" in message, message


# ---------------------------------------------------------------------------
# The risk model
# ---------------------------------------------------------------------------
# The probes below drive the installed Hypothesis until its health checks
# actually fire, and read the budgets off the failure messages.  An earlier
# version of this file asserted the same facts by grepping
# ``hypothesis/internal/conjecture/engine.py`` for
# ``state.invalid_examples == max_invalid_draws``.  That is not a test of
# behaviour, and it did not survive contact with a floating dependency:
# 6.168.0 renamed the three counters ``*_examples`` -> ``*_test_cases``
# without changing a single threshold or branch, and CI went red claiming
# "filter_too_much no longer counts only INVALID draws" -- which was false.
# Pinning a library's source text cannot work against ``hypothesis>=6.165,<7``.
# Driving the library can.
#
# These probes use ``suppress_health_check``, which the rest of this
# repository must not (see ``docs/developer_guide/testing_standards.md``).
# They are probes OF the library, not property tests of MADDENING: the whole
# point is to isolate one health check by silencing the others so that which
# check fires, and at what count, is unambiguous.  That is not a precedent
# for silencing one over a test that filters too much.

#: A strategy whose every draw exceeds Hypothesis's generation buffer.  The
#: minimum size alone is over the limit, so the overrun rate is 100% and the
#: probe is deterministic rather than a sampling argument.
_ALWAYS_OVERRUNS = st.lists(
    st.binary(min_size=500, max_size=500), min_size=20, max_size=100)

#: ``large_base_example`` fires on ``_ALWAYS_OVERRUNS`` before anything else
#: can, and is not what these probes are about.
_LARGE_BASE = getattr(HealthCheck, "large_base_example", None)

_FILTERED_RE = re.compile(
    r"(\d+) inputs were generated successfully, "
    r"while (\d+) inputs were filtered out")
_OVERRAN_RE = re.compile(
    r"(\d+) inputs were generated successfully, "
    r"while (\d+) inputs exceeded the maximum allowed entropy")


def _drive(strategy, body, suppress=(), max_examples=200):
    """Run a planted test to completion or to its first failed health check.

    Returns ``(health_check_name_or_None, message, status_counts)``.
    """
    seen: list[dict] = []

    @settings(max_examples=max_examples, deadline=None, database=None,
              phases=[Phase.generate], suppress_health_check=list(suppress))
    @given(strategy)
    def planted(value):
        body(value)

    name = message = None
    try:
        with collector.with_value(seen.append):
            planted()
    except FailedHealthCheck as exc:
        message = " ".join(str(exc).split())
        name = next((h.name for h in HealthCheck if f"HealthCheck.{h.name}" in message),
                    "unknown")
    except Unsatisfiable as exc:
        message = " ".join(str(exc).split())
    counts: Counter[str] = Counter()
    if seen:
        for case in seen[0].get("generate-phase", {}).get("test-cases", []):
            counts[case["status"] if isinstance(case, dict) else case.split(",")[0]] += 1
    return name, message, counts


def _counts_from(message, pattern, audit):
    """The two numbers a health-check message reports, or a skip.

    The *verdict* -- which health check fired -- is behaviour, and is
    asserted.  The counts have to be read out of English prose, and a
    reworded message is not a semantic change: that is the distinction this
    whole file exists to make, so a message this cannot parse skips with a
    reason instead of failing.
    """
    found = pattern.search(message or "")
    if found is None:
        pytest.skip(
            "hypothesis reworded its health-check message, so the budget "
            "cannot be read from it; the check itself still fired as "
            f"expected. Update the pattern. Message: {message!r}")
    return int(found.group(1)), int(found.group(2))


def test_filter_too_much_still_counts_only_invalid_draws(audit):
    """The semantic claim the whole audit rests on, driven rather than read.

    ``scripts/audit_property_rejection.py`` reports filtered draws and
    overrun draws in separate columns, gates them against separate budgets,
    and prices them with separate risk curves.  All of that is wrong if
    Hypothesis lumps them together.

    Two halves, because either alone is ambiguous:

    * an always-``assume(False)`` test must trip ``filter_too_much`` at
      exactly ``HEALTH_CHECK_MAX_INVALID`` filtered draws with no overruns
      anywhere -- which pins the budget and the counter;
    * a test whose every draw *overruns* must NOT trip ``filter_too_much``,
      even after far more than ``HEALTH_CHECK_MAX_INVALID`` of them.
      ``data_too_large`` is suppressed for this half precisely so that
      ``filter_too_much`` gets the chance it would take if overruns counted.
    """
    name, message, counts = _drive(st.integers(), lambda _: assume(False))
    assert name == "filter_too_much", (name, message)
    valid, filtered = _counts_from(message, _FILTERED_RE, audit)
    assert filtered == audit.HEALTH_CHECK_MAX_INVALID, message
    assert valid < audit.HEALTH_CHECK_MAX_VALID, message

    if _LARGE_BASE is None:
        pytest.skip("HealthCheck.large_base_example is gone; the overrun "
                    "probe cannot isolate data_too_large any more")
    name, message, counts = _drive(
        _ALWAYS_OVERRUNS, lambda _: None,
        suppress=[_LARGE_BASE, HealthCheck.data_too_large])
    # Order matters here.  A failed health check aborts the run before
    # Hypothesis emits statistics, so ``counts`` is empty whenever one fires
    # -- which means a ``counts``-based skip written first would turn the one
    # result this probe exists to catch into a silent skip.  The verdict is
    # read from ``name`` before anything is allowed to skip.
    assert name != "filter_too_much", (
        f"a test whose every draw overruns tripped filter_too_much: overruns "
        f"are being counted as filtered draws, and the audit's two-column "
        f"split, its two gates and its two risk curves all need re-deriving. "
        f"{message}"
    )
    if name is not None:
        pytest.skip(
            f"the overrun probe tripped {name} instead of running to "
            f"completion, so it could not give filter_too_much the chance "
            f"it needed; filter_too_much itself did not fire. {message}")
    if counts["overrun"] < audit.HEALTH_CHECK_MAX_INVALID:
        pytest.skip(
            f"the planted strategy only overran {counts['overrun']} times; "
            f"it can no longer outrun HealthCheck.filter_too_much's budget "
            f"of {audit.HEALTH_CHECK_MAX_INVALID}, so this probe proves "
            f"nothing -- make the draw bigger")
    assert counts["invalid"] == 0, counts


def test_the_overrun_budget_is_the_one_the_risk_model_prices(audit):
    """``HEALTH_CHECK_MAX_OVERRUN`` read off ``data_too_large`` itself.

    ``MAX_OVERRUN`` is set from this curve rather than from the measured
    distribution (the tree runs up to 37.6%), so the budget underneath it is
    load-bearing in a way ``MAX_REJECTION``'s is not.
    """
    if _LARGE_BASE is None:
        pytest.skip("HealthCheck.large_base_example is gone; this probe "
                    "cannot isolate data_too_large any more")
    name, message, _ = _drive(_ALWAYS_OVERRUNS, lambda _: None,
                              suppress=[_LARGE_BASE])
    assert name == "data_too_large", (name, message)
    _, overran = _counts_from(message, _OVERRAN_RE, audit)
    assert overran == audit.HEALTH_CHECK_MAX_OVERRUN, message


def test_the_valid_draw_budget_switches_the_health_check_off(audit):
    """``HEALTH_CHECK_MAX_VALID`` as the staircase it actually is.

    The risk model says "fewer than ``HEALTH_CHECK_MAX_VALID`` successes in
    the first ``HEALTH_CHECK_MAX_INVALID + HEALTH_CHECK_MAX_VALID - 1``
    trials".  The observable form of that is a step: plant a test whose first
    ``k`` draws are valid and whose every later draw is filtered, and there is
    a ``k`` above which ``filter_too_much`` can no longer fire at all.

    Asserted as a window rather than a point.  Whether the very first test
    case is counted by the health-check state is an implementation detail
    that has moved before and says nothing about the budget; *where the step
    is* is the budget.
    """
    def step(k):
        counter = itertools.count()
        name, message, _ = _drive(
            st.integers(), lambda _: assume(next(counter) < k))
        return name, message

    below, message = step(audit.HEALTH_CHECK_MAX_VALID - 1)
    assert below == "filter_too_much", (below, message)
    valid, _ = _counts_from(message, _FILTERED_RE, audit)
    assert valid < audit.HEALTH_CHECK_MAX_VALID, message

    above, message = step(audit.HEALTH_CHECK_MAX_VALID + 1)
    assert above is None, (
        f"a test whose first {audit.HEALTH_CHECK_MAX_VALID + 1} draws are "
        f"valid still tripped {above}; HEALTH_CHECK_MAX_VALID is not "
        f"{audit.HEALTH_CHECK_MAX_VALID} any more and the risk model's "
        f"success budget needs re-deriving"
    )


def test_the_risk_model_agrees_with_a_negative_binomial(audit):
    """Independent derivation of the same number.

    The model counts "fewer than 10 successes in the first 59 trials".  The
    equivalent statement is "the 50th failure arrives before the 10th
    success", i.e. a sum over negative-binomial terms.  Two ways of writing
    it, so an off-by-one in either shows up here rather than in a threshold
    nobody re-derives.
    """
    for budget in (audit.HEALTH_CHECK_MAX_INVALID, audit.HEALTH_CHECK_MAX_OVERRUN):
        for rejection in (0.1, 0.35, 0.5, 0.64, 0.8, 0.9):
            keep = 1.0 - rejection
            # P(the budget-th failure lands after k = 0..9 successes)
            nb = math.fsum(
                math.comb(budget - 1 + k, k) * keep**k * rejection**budget
                for k in range(audit.HEALTH_CHECK_MAX_VALID)
            )
            assert audit.health_check_failure_probability(
                rejection, budget=budget
            ) == pytest.approx(nb, rel=1e-9, abs=1e-18), (budget, rejection)


def test_the_risk_model_is_monotone_and_bounded(audit):
    assert audit.health_check_failure_probability(0.0) == 0.0
    assert audit.health_check_failure_probability(1.0) == 1.0
    previous = -1.0
    for i in range(21):
        p = audit.health_check_failure_probability(i / 20)
        assert p >= previous
        previous = p
    with pytest.raises(ValueError):
        audit.health_check_failure_probability(1.5)


def test_the_gate_sits_where_the_risk_turns_over(audit):
    """``MAX_REJECTION`` is not a round number someone liked.

    Two constraints fix it.  Below it, one run of one test trips
    ``filter_too_much`` with probability under 1e-11 -- so a suite of a few
    hundred property tests run on every push will not see it this decade.
    Above 70% the same probability is 7e-3, which for a suite this size is a
    red CI every few weeks.  The measured distribution of this repository's
    two property suites puts every test at or under 26.6% after the fixes in
    this branch, so the gate also leaves real headroom -- about 1.5x -- for
    sampling noise and for an honest test that drifts a little.  At 80
    examples that noise is worth several points on its own: the same four
    ``TestFIM`` tests measured 8/16/18/22% in one ci run and 16/24/26/27% in
    the next, which is why the gate is not set snugly against the maximum.
    """
    assert audit.health_check_failure_probability(audit.MAX_REJECTION) < 1e-11
    assert audit.health_check_failure_probability(0.70) > 1e-3
    assert 0.266 < audit.MAX_REJECTION < 0.70


def test_the_overrun_gate_is_set_from_the_risk_not_from_the_tree(audit):
    """``MAX_OVERRUN`` cannot be set the way ``MAX_REJECTION`` was.

    ``data_too_large`` tolerates 20 overruns before 10 valid draws rather
    than 50, so its curve turns over far earlier -- and the measured tree
    runs right up to it, at 37.6% for
    ``test_a_mask_whose_keys_differ_from_params_is_refused``.  A gate set
    from that distribution would sit in the part of the curve where the
    health check genuinely fires.  It is set from the curve instead: at
    ``MAX_OVERRUN`` one run trips the check with probability of order a few
    percent, which is where the problem stops being theoretical, and the
    worst measured today is comfortably below it.
    """
    at_gate = audit.health_check_failure_probability(
        audit.MAX_OVERRUN, budget=audit.HEALTH_CHECK_MAX_OVERRUN)
    assert 5e-3 < at_gate < 1e-1, at_gate
    # Today's worst measured overrun rate must sit under the gate, or CI is
    # red on merge rather than on a regression.
    assert audit.MAX_OVERRUN > 0.376
    # And the overrun budget must stay looser than the filter budget in
    # rate terms while being tighter in risk terms -- if that ever inverts,
    # the two gates have been conflated.
    assert at_gate > audit.health_check_failure_probability(
        audit.MAX_REJECTION)


def test_a_filtered_test_still_gets_every_example_it_asked_for(audit):
    """The claim the docs rest on, pinned against the installed Hypothesis.

    ``docs/developer_guide/testing_standards.md`` tells readers that a high
    rejection rate costs wall-clock and health-check margin but *not* search
    depth: the engine keeps drawing until it has ``max_examples`` valid
    examples, and only gives up below ~1% valid
    (``INVALID_THRESHOLD_BASE`` / ``INVALID_PER_VALID``).  That is a property
    of this Hypothesis version, not a law, and it is the whole reason the
    gate is justified on fragility rather than on lost coverage.  If it ever
    stops being true the advice has to change, so it is asserted rather than
    believed.

    50% is used rather than something dramatic because the health check's
    per-run failure probability there is 2.6e-8: deliberately far enough
    from the edge that this test cannot itself flake.
    """
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.statistics import collector

    seen: list[dict] = []

    @settings(max_examples=60, deadline=None, database=None)
    @given(st.integers(0, 10**9))
    def half_of_every_draw_is_thrown_away(x):
        assume(x % 2 == 0)

    with collector.with_value(seen.append):
        half_of_every_draw_is_thrown_away()

    record = audit.record_from_statistics("synthetic", seen[0])
    assert record.effective_examples == 60, (
        "Hypothesis no longer tops up a filtered run to max_examples valid "
        "examples; testing_standards.md says it does"
    )
    assert record.rate > 0.3, "the synthetic gate did not filter anything"
    assert not record.starved


# ---------------------------------------------------------------------------
# The gate, end to end
# ---------------------------------------------------------------------------
_SYNTHETIC_SUITE = '''
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st

_S = settings(max_examples=60, deadline=None, database=None,
              suppress_health_check=[HealthCheck.filter_too_much])

@_S
@given(st.integers(0, 10**9))
def test_wastes_four_draws_in_five(x):
    # keeps 20%: a rejection rate of ~80%
    assume(x % 10 < 2)

@_S
@given(st.integers(0, 10**9))
def test_wastes_nothing(x):
    assert x >= 0
'''


def _run_audit_cli(tmp_path, *args):
    suite = tmp_path / "test_synthetic_rejection.py"
    suite.write_text(textwrap.dedent(_SYNTHETIC_SUITE))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), str(suite), *args],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )


def test_the_gate_fails_on_a_test_that_wastes_most_of_its_draws(tmp_path):
    """The mutation this gate exists to catch, planted and caught.

    A gate that has never been shown to fail is not a gate; MADDENING's four
    compliance gates scored 6 caught against 22 missed before anyone checked.
    """
    result = _run_audit_cli(tmp_path, "--check", "--max-rejection", "0.4")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test_wastes_four_draws_in_five" in result.stdout
    assert "over a gate" in result.stdout


def test_the_gate_passes_the_same_suite_under_a_looser_budget(tmp_path):
    """The other direction: the failure above is the rate, not the harness."""
    result = _run_audit_cli(tmp_path, "--check", "--max-rejection", "0.95")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_measured_rate_is_the_rate_the_test_was_built_to_have(tmp_path):
    """End to end, the number in the table is the number in the source.

    ``assume(x % 10 < 2)`` keeps a fifth of its draws; the audit has to say
    so, or the whole table is decoration.
    """
    result = _run_audit_cli(tmp_path, "--max-rejection", "0.95")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = {
        line.split()[0].rsplit("::", 1)[-1]: line
        for line in result.stdout.splitlines()
        if "::test_wastes" in line
    }
    assert set(rows) == {"test_wastes_four_draws_in_five", "test_wastes_nothing"}
    wasteful = rows["test_wastes_four_draws_in_five"].split()
    measured = float(wasteful[3].rstrip("%!").strip()) / 100.0
    assert 0.70 <= measured <= 0.88, result.stdout
    assert rows["test_wastes_nothing"].split()[3].startswith("0.0%")


def test_the_json_report_carries_every_column_the_table_shows(tmp_path):
    """The JSON is what a trend over releases would be built from."""
    import json

    out = tmp_path / "report.json"
    result = _run_audit_cli(tmp_path, "--max-rejection", "0.95", "--json", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(out.read_text())
    assert {r["nodeid"].rsplit("::", 1)[-1] for r in payload["records"]} == {
        "test_wastes_four_draws_in_five", "test_wastes_nothing",
    }
    for record in payload["records"]:
        assert record.keys() >= {
            "drawn", "rejected", "effective_examples", "rate", "overrun",
            "overrun_rate", "health_check_risk", "starved",
        }
