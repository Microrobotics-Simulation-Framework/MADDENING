#!/usr/bin/env python3
"""Measure what fraction of each property test's draws Hypothesis throws away.

A property test that calls ``assume()`` -- or draws from a strategy with a
``.filter()`` on it -- does not get the examples it asked for for free.  Every
rejected draw is a graph built, a solve run and a result discarded, and the
test never says so: a test rejecting 5% of its draws and one rejecting 85%
report the same green tick and the same ``max_examples``.

The one place Hypothesis *does* say so is ``HealthCheck.filter_too_much``,
which aborts a run that accumulates 50 rejected draws before 10 accepted ones
(``max_invalid_draws`` / ``max_valid_draws`` in
``hypothesis.internal.conjecture.engine``).  That is a *sampling* test on the
first few dozen draws, so it is a step function with a very soft edge: a test
at a true 60% filter rate trips it about once in 28,000 runs, one at 70% about
once in 140, and one at 85% three runs in five.  Which is how a test can sit
green for months and then go red for whoever next narrows a strategy.

What a high rejection rate does NOT do, on Hypothesis 6.165.x, is reduce the
number of examples actually checked.  The engine keeps drawing until it has
``max_examples`` *valid* ones and only gives up below ~1% valid
(``INVALID_THRESHOLD_BASE`` / ``INVALID_PER_VALID``, derived for r=0.01).
Measured here against a synthetic gate: at 98% rejection the test still ran
its full 200 examples, using 9068 draws to do it.  So the cost of rejection is
wall-clock and health-check fragility, not silently shallower search -- see
``docs/developer_guide/testing_standards.md``.

Usage
-----
Run the audit over the property suites and print the table::

    MADDENING_HYPOTHESIS_PROFILE=ci PYTHONPATH=src JAX_PLATFORMS=cpu \\
        python scripts/audit_property_rejection.py \\
            tests/property tests/verification/hypothesis

Fail if any test is over the gate.  The ``verify-hypothesis`` job in
``.github/workflows/ci.yml`` runs the suite this way, so the measurement
costs nothing beyond the run it was already doing::

    python scripts/audit_property_rejection.py --check \
        --pytest-arg=-v tests/verification/hypothesis/

``tests/property/test_draw_rejection_budget.py`` tests this harness rather
than using it: the accounting, the risk model, and that the gate can fail.

Options of note: ``--json PATH`` writes the raw per-test record,
``--markdown`` emits the table as Markdown, ``--max-rejection`` overrides the
gate, ``--only-over`` trims the table to the interesting rows.

This module is also importable: ``RejectionAuditPlugin`` is a plain pytest
plugin object, so ``pytest.main([...], plugins=[RejectionAuditPlugin()])``
works from anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest
from hypothesis.statistics import collector

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
# Picked from the measured distribution of this repository's property suites,
# not from intuition.  See ``docs/developer_guide/testing_standards.md`` and
# ``tests/property/test_draw_rejection_budget.py`` for the derivation.  In
# short: over 170 tests that draw, the filter rate before this branch had a
# median of 0.0%, a p90 of 13.0% and a maximum of 42.9%; after it the worst
# is 26.6%.  0.40 is ~1.5x that worst case, so it will not fire on sampling
# noise -- which at 80 examples is worth several points on its own --
# and ``HealthCheck.filter_too_much``'s per-run probability at 0.40 is
# 1.8e-12 -- a test sitting exactly on the gate is still safe, while one at
# 0.70 is at 7e-3 and one at 0.85 at 61%.
MAX_REJECTION = 0.40

#: Overruns are a separate problem with a separate health check, so they get
#: a separate budget.  ``data_too_large`` fires at 20 overruns before 10 valid
#: draws, which is a far tighter window than ``filter_too_much``'s -- but an
#: overrun is Hypothesis running out of entropy for a big draw, not anything
#: the author wrote, so it is reported beside the filter rate rather than
#: folded into it.  Measured here: every ``hypothesis.extra.numpy.arrays``
#: test in this tree overruns a few percent of its draws with no ``assume``
#: anywhere in it.
#:
#: Unlike ``MAX_REJECTION``, this is set from the RISK curve rather than from
#: the measured distribution, because the measured distribution runs right up
#: to it: the worst in the tree is
#: ``test_a_mask_whose_keys_differ_from_params_is_refused`` at 37.6%, whose
#: per-run ``data_too_large`` probability is 6e-4.  0.50 is where that
#: probability reaches 3% -- about one red run in thirty for a single test,
#: which is no longer something to leave alone.  It is a tripwire for a
#: strategy that starts drawing much bigger, not a clean bill of health for
#: what is under it; the audit prints the column either way, and the fix for
#: a high overrun rate is a smaller draw, never a removed gate.
MAX_OVERRUN = 0.50

#: ``hypothesis.internal.conjecture.engine`` health-check constants, mirrored
#: so the risk estimates below do not depend on Hypothesis internals staying
#: importable.  ``test_draw_rejection_budget.py`` pins them against the
#: installed Hypothesis.
HEALTH_CHECK_MAX_INVALID = 50
HEALTH_CHECK_MAX_VALID = 10
HEALTH_CHECK_MAX_OVERRUN = 20

#: Phases whose draws count towards the budget.  ``shrink`` is excluded: it
#: only runs for a test that is already failing, and its draws are not
#: examples.
COUNTED_PHASES = ("reuse", "generate")


def health_check_failure_probability(
    rejection_rate: float, *, budget: int = HEALTH_CHECK_MAX_INVALID
) -> float:
    """Chance that one run at ``rejection_rate`` trips the matching health check.

    A Hypothesis health check fails when a run reaches ``budget`` rejected
    draws before ``HEALTH_CHECK_MAX_VALID`` accepted ones -- 50 for
    ``filter_too_much``, 20 for ``data_too_large``.  Draws are independent
    Bernoulli trials to a good approximation, so that is the probability of
    seeing fewer than ``HEALTH_CHECK_MAX_VALID`` successes in the first
    ``budget + HEALTH_CHECK_MAX_VALID - 1`` trials.

    This is why the health check is no substitute for the measurement: it is
    a step function with a very soft edge.  At a 60% filter rate it fires
    about once in 28000 runs, at 85% about once in three.

    Parameters
    ----------
    rejection_rate : float
        Fraction of draws rejected, in ``[0, 1]``.
    budget : int, default ``HEALTH_CHECK_MAX_INVALID``
        Rejected draws the check tolerates.  Pass
        ``HEALTH_CHECK_MAX_OVERRUN`` for ``data_too_large``.

    Returns
    -------
    float
        Probability in ``[0, 1]`` that a single run trips the health check.

    Examples
    --------
    >>> round(health_check_failure_probability(0.0), 6)
    0.0
    >>> health_check_failure_probability(0.85) > 0.3
    True
    """
    if not 0.0 <= rejection_rate <= 1.0:
        raise ValueError(f"rejection_rate must be in [0, 1], got {rejection_rate!r}")
    keep = 1.0 - rejection_rate
    if keep >= 1.0:
        return 0.0
    if keep <= 0.0:
        return 1.0
    n = budget + HEALTH_CHECK_MAX_VALID - 1
    return math.fsum(
        math.comb(n, k) * keep**k * (1.0 - keep) ** (n - k)
        for k in range(HEALTH_CHECK_MAX_VALID)
    )


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class RejectionRecord:
    """Draw accounting for one test id, summed over every ``@given`` run in it."""

    nodeid: str
    valid: int = 0
    invalid: int = 0
    overrun: int = 0
    runs: int = 0
    stopped_because: list[str] = field(default_factory=list)

    @property
    def drawn(self) -> int:
        """Total draws Hypothesis generated, accepted or not."""
        return self.valid + self.invalid + self.overrun

    @property
    def rejected(self) -> int:
        """Draws thrown away by ``assume()`` or a strategy ``.filter()``.

        Overruns are deliberately not in here.  They are Hypothesis running
        out of entropy for a large draw, not the test rejecting an input, and
        they answer to a different health check (``data_too_large``).  Folding
        them in would blame a test for the size of its arrays: every
        ``hypothesis.extra.numpy.arrays`` property in this tree overruns a few
        percent of its draws with no ``assume`` written anywhere in it.
        """
        return self.invalid

    @property
    def rate(self) -> float:
        """Fraction of *decided* draws filtered out; 0.0 for a test that drew none.

        The denominator excludes overruns, so this is exactly the ratio
        ``HealthCheck.filter_too_much`` watches.
        """
        decided = self.valid + self.invalid
        return self.invalid / decided if decided else 0.0

    @property
    def overrun_rate(self) -> float:
        """Fraction of all draws that ran out of entropy."""
        return self.overrun / self.drawn if self.drawn else 0.0

    @property
    def effective_examples(self) -> int:
        """Draws that actually reached the test body."""
        return self.valid

    @property
    def starved(self) -> bool:
        """True if Hypothesis gave up on a phase for want of valid draws."""
        return any("satisfied assumptions" in s for s in self.stopped_because)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodeid": self.nodeid,
            "drawn": self.drawn,
            "rejected": self.rejected,
            "effective_examples": self.effective_examples,
            "rate": self.rate,
            "invalid": self.invalid,
            "overrun": self.overrun,
            "overrun_rate": self.overrun_rate,
            "runs": self.runs,
            "starved": self.starved,
            "health_check_risk": health_check_failure_probability(self.rate),
            "overrun_health_check_risk": health_check_failure_probability(
                self.overrun_rate, budget=HEALTH_CHECK_MAX_OVERRUN),
            "stopped_because": self.stopped_because,
        }


def record_from_statistics(nodeid: str, stats: dict[str, Any],
                           into: RejectionRecord | None = None) -> RejectionRecord:
    """Fold one Hypothesis ``statistics`` dict into a :class:`RejectionRecord`.

    Parameters
    ----------
    nodeid : str
        The pytest node id the run belongs to.
    stats : dict
        ``ConjectureRunner.statistics``, as handed to
        ``hypothesis.statistics.collector``.
    into : RejectionRecord, optional
        Accumulate into this record instead of a fresh one, for a test that
        runs the engine more than once.

    Returns
    -------
    RejectionRecord
    """
    rec = into if into is not None else RejectionRecord(nodeid)
    rec.runs += 1
    because = stats.get("stopped-because")
    if because:
        rec.stopped_because.append(str(because))
    for phase in COUNTED_PHASES:
        cases = stats.get(f"{phase}-phase", {}).get("test-cases", [])
        counts = Counter(case["status"] for case in cases)
        rec.valid += counts["valid"]
        rec.invalid += counts["invalid"]
        rec.overrun += counts["overrun"]
        # ``interesting`` is a failing example; it reached the body, so it is
        # not a rejection, but it is not a clean pass either.  Count it as
        # work done so a failing run does not read as 100% rejection.
        rec.valid += counts["interesting"]
    return rec


# --------------------------------------------------------------------------
# The pytest plugin
# --------------------------------------------------------------------------
class RejectionAuditPlugin:
    """Collect per-test draw accounting from Hypothesis's statistics hook.

    Hypothesis calls ``hypothesis.statistics.collector`` with the engine's
    statistics at the end of every ``@given`` run, whether or not its pytest
    plugin is loaded -- which matters here, because MADDENING runs pytest with
    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``.
    """

    def __init__(self) -> None:
        self.records: dict[str, RejectionRecord] = {}

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        """Watch the Hypothesis statistics hook for the duration of one test."""
        nodeid = item.nodeid

        def observe(stats: dict[str, Any]) -> None:
            self.records[nodeid] = record_from_statistics(
                nodeid, stats, self.records.get(nodeid)
            )

        with collector.with_value(observe):
            return (yield)

    def sorted_records(self) -> list[RejectionRecord]:
        """Records worst-first, ties broken by node id for a stable table."""
        return sorted(self.records.values(), key=lambda r: (-r.rate, r.nodeid))

    def over_budget(self, max_rejection: float = MAX_REJECTION,
                    max_overrun: float = MAX_OVERRUN) -> list[RejectionRecord]:
        """Records over either budget: filtering, or entropy overruns."""
        return [r for r in self.sorted_records()
                if r.rate > max_rejection or r.overrun_rate > max_overrun]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def format_table(records: Sequence[RejectionRecord], *, markdown: bool = False,
                 max_rejection: float = MAX_REJECTION) -> str:
    """Render the audit table.

    Parameters
    ----------
    records : sequence of RejectionRecord
        Already in the order they should appear.
    markdown : bool, default False
        Emit a Markdown table (for a PR body) rather than plain columns.
    max_rejection : float
        Rows above this are flagged.

    Returns
    -------
    str
    """
    header = ("test", "drawn", "filtered", "rate", "overrun", "effective",
              "risk/run")
    rows: list[tuple[str, ...]] = []
    for rec in records:
        rows.append((
            rec.nodeid,
            str(rec.drawn),
            str(rec.rejected),
            f"{rec.rate:.1%}" + ("  !" if rec.rate > max_rejection else ""),
            f"{rec.overrun_rate:.1%}"
            + ("  !" if rec.overrun_rate > MAX_OVERRUN else ""),
            str(rec.effective_examples) + ("  STARVED" if rec.starved else ""),
            _fmt_risk(health_check_failure_probability(rec.rate)),
        ))
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] * len(header)) + "|"]
        out += ["| " + " | ".join(c.replace("|", r"\|") for c in row) + " |"
                for row in rows]
        return "\n".join(out)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip(),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
              for row in rows]
    return "\n".join(lines)


def _fmt_risk(p: float) -> str:
    if p == 0.0:
        return "0"
    if p < 1e-4:
        return f"{p:.0e}"
    return f"{p:.2%}"


def summarise(records: Sequence[RejectionRecord]) -> str:
    """One-paragraph description of the distribution, for the report header."""
    if not records:
        return "no Hypothesis runs were observed"
    rates = sorted(r.rate for r in records)
    n = len(rates)

    def pct(q: float) -> float:
        return rates[min(n - 1, int(q * n))]

    zero = sum(1 for r in rates if r == 0.0)
    return (
        f"{n} tests with Hypothesis draws; {zero} filter nothing at all. "
        f"filter rate: median {pct(0.5):.1%}, p90 {pct(0.9):.1%}, "
        f"max {rates[-1]:.1%}. "
        f"Total draws {sum(r.drawn for r in records)}, of which "
        f"{sum(r.rejected for r in records)} filtered out and "
        f"{sum(r.overrun for r in records)} overrun."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run_audit(paths: Sequence[str], *, extra_args: Sequence[str] = ()) -> tuple[RejectionAuditPlugin, int]:
    """Run pytest over ``paths`` with the audit plugin attached.

    Returns
    -------
    (RejectionAuditPlugin, int)
        The populated plugin and pytest's exit status.
    """
    plugin = RejectionAuditPlugin()
    args = [*paths, "-p", "no:cacheprovider", *extra_args]
    # Quiet only when the caller expressed no preference: the audit is meant
    # to be droppable in front of an existing CI pytest invocation without
    # changing what that invocation prints.
    if not any(a.startswith("-v") or a.startswith("-q") or a == "--verbose"
               for a in extra_args):
        args.append("-q")
    status = pytest.main(args, plugins=[plugin])
    return plugin, int(status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*",
                        default=["tests/property", "tests/verification/hypothesis"],
                        help="test paths to audit")
    parser.add_argument("--max-rejection", type=float, default=MAX_REJECTION,
                        help=f"gate, as a fraction (default {MAX_REJECTION})")
    parser.add_argument("--max-overrun", type=float, default=MAX_OVERRUN,
                        help=f"entropy-overrun gate (default {MAX_OVERRUN})")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any test is over the gate")
    parser.add_argument("--json", dest="json_path", type=Path,
                        help="write the raw per-test records here")
    parser.add_argument("--markdown", action="store_true",
                        help="emit the table as Markdown")
    parser.add_argument("--only-over", type=float, default=None, metavar="RATE",
                        help="only table rows above this rejection rate")
    parser.add_argument("--pytest-arg", action="append", default=[],
                        dest="pytest_args", help="extra argument for pytest")
    args = parser.parse_args(argv)

    plugin, status = run_audit(args.paths, extra_args=args.pytest_args)
    records = plugin.sorted_records()

    shown = records
    if args.only_over is not None:
        shown = [r for r in records
                 if r.rate > args.only_over or r.overrun_rate > args.only_over]

    profile = os.environ.get("MADDENING_HYPOTHESIS_PROFILE", "dev")
    print()
    print(f"draw-rejection audit (hypothesis profile: {profile})")
    print(summarise(records))
    print()
    print(format_table(shown, markdown=args.markdown,
                       max_rejection=args.max_rejection))

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(
            {"profile": profile, "records": [r.as_dict() for r in records]},
            indent=2) + "\n")
        print(f"\nwrote {args.json_path}")

    over = plugin.over_budget(args.max_rejection, args.max_overrun)
    if over:
        print(f"\n{len(over)} test(s) over a gate "
              f"(filter {args.max_rejection:.0%}, overrun {args.max_overrun:.0%}):")
        for rec in over:
            why = []
            if rec.rate > args.max_rejection:
                why.append(f"filters {rec.rate:.1%}")
            if rec.overrun_rate > args.max_overrun:
                why.append(f"overruns {rec.overrun_rate:.1%}")
            print(f"  {rec.nodeid}: {', '.join(why)}")
    if args.check:
        if status != 0:
            print("\npytest itself failed; the audit is not trustworthy",
                  file=sys.stderr)
            return status
        return 1 if over else 0
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
