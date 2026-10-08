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
:data:`SLOW` (many examples: the hunt).  Used by
``test_sysid_targeted_search.py`` and the three coupling searches.

**A hunt is seeded.**  Every hunt runs under a seed of its own
(``SLOW.seeded(n)``), so the slow lane gives one tree one verdict; a
random profile with no seed is refused.  **Exploring** is one environment
switch, read here and nowhere else: under ``MADDENING_SEARCH_ENTROPY=fresh``
one integer of entropy is drawn for the process and mixed into the seed of
every hunt (linear, nonlinear, geometry, sysid), and each hunt prints it.
**Replaying** an exploring run is the same command with that integer in
place of ``fresh``.  The per-push searches read no switch.
``test_targeted_search.py`` holds all three, and that no hunt in the tree
is written without a seed.
"""

from __future__ import annotations

import hashlib
import math
import os
import secrets
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
    #: ``True``: a hunt, whose seed :data:`ENTROPY_VARIABLE` replaces with
    #: fresh entropy.  A per-push search draws under its own seed whatever
    #: the environment holds.
    hunt: bool = False

    def seeded(self, seed: int, shrink: bool = True) -> "Profile":
        return replace(self, seed=seed, derandomize=False, shrink=shrink)


#: Every push: the house floor of examples, the same ones on every run.
PER_PUSH = Profile(max_examples=EXAMPLES_FLOOR, derandomize=True)
#: The slow lane: a hunt.  It runs under a seed of its own,
#: ``SLOW.seeded(n)``: with none it is refused.
SLOW = Profile(max_examples=40 * EXAMPLES_FLOOR, derandomize=False, hunt=True)

#: The one switch that lets the hunts explore.  Unset (CI): every hunt
#: draws under its own seed.  ``fresh``: one integer of entropy is drawn
#: for the process, mixed into every hunt's seed and printed by each hunt.
#: An integer: that entropy again, which replays the run that printed it.
ENTROPY_VARIABLE = "MADDENING_SEARCH_ENTROPY"
_fresh_entropy: Optional[int] = None


def search_entropy() -> Optional[int]:
    """What :data:`ENTROPY_VARIABLE` asks for: ``None`` (unset or empty:
    the hunts are seeded) or the entropy of an exploring run.  Anything
    but ``fresh`` or an integer is refused: a misspelt switch must not
    read as "seeded"."""
    global _fresh_entropy
    raw = os.environ.get(ENTROPY_VARIABLE, "").strip()
    if not raw:
        return None
    if raw == "fresh":
        if _fresh_entropy is None:
            # The operating system's entropy: Hypothesis reseeds ``random``.
            _fresh_entropy = secrets.randbits(32)
        return _fresh_entropy
    if not raw.isdecimal():
        raise ValueError(f"{ENTROPY_VARIABLE} is 'fresh' or the integer an exploring run "
                         f"printed, not {raw!r}")
    return int(raw)


def seed_of(profile: Profile) -> tuple:
    """``(seed, entropy)``: the seed a search under *profile* draws with
    (``None``: derandomised) and the entropy mixed into it (``None``: the
    profile's own seed).

    A random profile with no seed is refused, switch or no switch: it
    would draw other examples on every run, and under the switch its
    examples could not be told from another hunt's."""
    if profile.derandomize:
        return profile.seed, None
    if profile.seed is None:
        raise ValueError(
            "a random search runs under a seed of its own, so that one tree gives one "
            f"verdict: pass SLOW.seeded(n), not a profile with no seed ({profile}).  "
            f"{ENTROPY_VARIABLE}=fresh is how a seeded hunt explores.")
    entropy = search_entropy() if profile.hunt else None
    if entropy is None:
        return profile.seed, None
    mixed = hashlib.sha256(f"{entropy}:{profile.seed}".encode()).digest()
    return int.from_bytes(mixed[:8], "big"), entropy


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
    #: The seed the examples were drawn under (``None``: derandomised), and
    #: the entropy of an exploring run that was mixed into it (``None``:
    #: the seed is the profile's own).
    seed: Optional[int] = None
    entropy: Optional[int] = None

    @property
    def drawn(self) -> str:
        """A fingerprint of every score in the order scored: two runs that
        drew the same examples and scored them alike print the same one."""
        return hashlib.sha256(repr(self.scores).encode()).hexdigest()[:12]

    def __str__(self) -> str:
        return (f"score {self.score!r} for {self.example!r} ({self.details}); "
                f"{self.examples} examples scored (fingerprint {self.drawn})"
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
    seed, entropy = seed_of(profile)
    report = Report(seed=seed, entropy=entropy)
    replay = ""
    if entropy is not None:
        replay = (f"exploring under {ENTROPY_VARIABLE}={entropy} (the hunt's own seed is "
                  f"{profile.seed}): the same command with that value in place of 'fresh' "
                  "draws these examples again")
        print(f"{label}: {replay}")
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
    if seed is not None:
        prop = hypothesis.seed(seed)(prop)
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
                f"example {report.found_after})" + (f"\n{replay}" if replay else "")) from None
    return report
