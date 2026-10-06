"""The targeted-search helper does what its callers rely on.

``targeted_search`` is the instrument of
``test_sysid_targeted_search.py``; an instrument that could not fail would
make every search under it pass.  On scores with a known answer: it finds
an example over the threshold and reports the shrunk one, it returns the
worst example when there is none, the per-push profile scores the same
examples on every run, and a score that cannot be compared is refused.
"""

from __future__ import annotations

import pytest
from hypothesis import strategies as st

from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

_PAIRS = st.tuples(st.integers(0, 1000), st.integers(0, 1000))


def _sum(pair):
    return float(pair[0] + pair[1]), {"pair": pair}


def test_an_example_over_the_threshold_is_found_and_reported_shrunk():
    with pytest.raises(AssertionError) as caught:
        targeted_search(_PAIRS, _sum, 1500.0, profile=SLOW.seeded(0), label="sum")
    message = str(caught.value)
    # The shrunk example is the smallest pair over the threshold: 1501.
    assert "sum is 1501.0, over the threshold 1500.0" in message, message
    assert "shrunk example" in message and "'pair'" in message, message


def test_the_report_says_how_many_examples_the_first_find_took():
    found = targeted_search(_PAIRS, _sum, 1500.0, profile=SLOW.seeded(0, shrink=False),
                            fail=False)
    assert found.found_after is not None and 1 <= found.found_after <= found.examples
    assert found.score > 1500.0 and sum(found.example) == found.score
    # Not shrunk: the scoring stopped at the find, but for Hypothesis's one
    # replay of it.
    assert found.examples <= found.found_after + 1


def test_a_clean_search_returns_the_worst_example_it_saw():
    report = targeted_search(_PAIRS, _sum, 2000.0, profile=PER_PUSH)
    assert report.found_after is None and report.examples == PER_PUSH.max_examples
    assert report.score == max(report.scores) == float(sum(report.example))
    assert report.details == {"pair": report.example}


def test_the_per_push_profile_scores_the_same_examples_every_time():
    first = targeted_search(_PAIRS, _sum, 2000.0, profile=PER_PUSH)
    again = targeted_search(_PAIRS, _sum, 2000.0, profile=PER_PUSH)
    assert first.scores == again.scores and len(set(first.scores)) > 1


@pytest.mark.parametrize("bad", [float("nan"), -1.0])
def test_a_score_that_is_not_a_non_negative_number_is_refused(bad):
    with pytest.raises(ValueError, match="non-negative number"):
        targeted_search(_PAIRS, lambda pair: (bad, None), 1.0, profile=PER_PUSH)


def test_an_infinite_score_is_over_any_threshold():
    with pytest.raises(AssertionError, match="is inf, over the threshold"):
        targeted_search(_PAIRS, lambda pair: (float("inf"), None), 1e300, profile=PER_PUSH)
