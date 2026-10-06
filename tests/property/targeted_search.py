"""A targeted search: Hypothesis climbs a "how wrong" score towards a threshold.

A grid samples chosen points.  A search hunts between them: the caller gives
a strategy, a **score** -- a function of one drawn example that returns a
non-negative float, 0 where nothing is wrong and larger the further the
result is from what is claimed, with whatever explains it -- and the
largest score the claim allows.  :func:`targeted_search` runs the property
``score <= threshold`` and hands every score to ``hypothesis.target``, so
once the random draws are spent Hypothesis mutates the worst example it has
seen towards a worse one.  On a failure it raises an ``AssertionError``
that carries the *shrunk* example, its score and its details.

Two profiles: :data:`PER_PUSH` (derandomised, the house floor of examples:
the same draws on every run, so a red run is a change in the code) and
:data:`SLOW` (random, many examples: the hunt).  Used by
``test_sysid_targeted_search.py``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import hypothesis
from hypothesis import HealthCheck, Phase, given, settings

from tests.conftest import EXAMPLES_FLOOR


@dataclass(frozen=True)
class Profile:
    """How long a search runs and whether it draws the same examples every
    time.  ``seed``: a fixed seed for a random profile (a search that is
    being measured); ``None`` leaves it to Hypothesis."""

    max_examples: int
    derandomize: bool
    seed: Optional[int] = None
    #: ``False`` reports the first failing example as drawn (a measurement
    #: of how long a defect takes to find has no use for the shrink).
    shrink: bool = True

    def seeded(self, seed: int, shrink: bool = True) -> "Profile":
        return replace(self, seed=seed, derandomize=False, shrink=shrink)


#: Every push: the house floor of examples, the same ones on every run.
PER_PUSH = Profile(max_examples=EXAMPLES_FLOOR, derandomize=True)
#: The slow lane: a random hunt.
SLOW = Profile(max_examples=40 * EXAMPLES_FLOOR, derandomize=False)


@dataclass
class Report:
    """What a search saw.  ``found_after``: how many examples had been
    scored when the first one went over the threshold (``None``: none
    did).  ``example``, ``score`` and ``details``: the worst example seen,
    or -- after a failure -- the shrunk failing one."""

    examples: int = 0
    found_after: Optional[int] = None
    score: float = 0.0
    example: Any = None
    details: Any = None
    scores: list = field(default_factory=list)

    def __str__(self) -> str:
        return (f"score {self.score!r} for {self.example!r} ({self.details}); "
                f"{self.examples} examples scored"
                + ("" if self.found_after is None
                   else f", the first over the threshold was example {self.found_after}"))


class _OverThreshold(Exception):
    """One type, raised from one line, so Hypothesis shrinks to one example."""


def targeted_search(strategy, score: Callable[[Any], tuple], threshold: float, *,
                    profile: Profile = PER_PUSH, label: str = "score",
                    fail: bool = True) -> Report:
    """Search ``strategy`` for an example whose ``score`` exceeds ``threshold``.

    ``score(example)`` returns ``(value, details)``: ``value`` a
    non-negative float (``inf`` allowed; ``nan`` is refused, since a score
    that cannot be compared can hide anything), ``details`` anything whose
    ``repr`` explains it.  It must be a function of the example alone.

    Raises ``AssertionError`` with the shrunk example when one is found
    (``fail=False`` returns the report instead, for a test that measures
    how long a known defect takes to find); returns the :class:`Report` of
    the worst example otherwise.
    """
    report = Report()
    last = {}

    def run(example):
        value, details = score(example)
        value = float(value)
        if math.isnan(value) or value < 0.0:
            raise ValueError(f"a score is a non-negative number, not {value!r} "
                             f"(for {example!r}: {details})")
        report.examples += 1
        report.scores.append(value)
        hypothesis.target(min(value, sys.float_info.max), label=label)
        if value > threshold:
            if report.found_after is None:
                report.found_after = report.examples
            last.update(example=example, score=value, details=details)
            raise _OverThreshold
        if report.found_after is None and value >= report.score:
            report.score, report.example, report.details = value, example, details

    prop = given(strategy)(run)
    prop = settings(max_examples=profile.max_examples, derandomize=profile.derandomize,
                    database=None, deadline=None, print_blob=False,
                    phases=[p for p in Phase if profile.shrink or p is not Phase.shrink],
                    suppress_health_check=list(HealthCheck))(prop)
    if profile.seed is not None:
        prop = hypothesis.seed(profile.seed)(prop)
    try:
        prop()
    except _OverThreshold:
        # Hypothesis replays the shrunk example last: ``last`` holds it.
        report.score, report.example, report.details = (
            last["score"], last["example"], last["details"])
        if fail:
            raise AssertionError(
                f"{label} is {report.score!r}, over the threshold {threshold!r}, for the "
                f"shrunk example\n  {report.example!r}\n  {report.details}\n"
                f"({report.examples} examples scored, the first over the threshold was "
                f"example {report.found_after})") from None
    return report
