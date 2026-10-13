"""The targeted-search helper does what its callers rely on.

``targeted_search`` is the instrument of
``test_sysid_targeted_search.py``; an instrument that could not fail would
make every search under it pass.  On scores with a known answer: it finds
an example over the threshold and reports the shrunk one, it returns the
worst example when there is none, the per-push profile scores the same
examples on every run, and a score that cannot be compared is refused.

**A hunt is seeded** (the second half of this module).  Until 2026-10-08
the slow hunts of the linear coupling search and of the sysid search drew
fresh entropy on every run, so one tree could be green on one slow-lane
run and red on the next.  Held here, per push and with nothing compiled:
a seeded hunt scores the same examples every time; a random profile with
no seed is refused; no hunt in the tree is written without a seed
(:func:`hunts_of`, a scan of every test module that names the slow
profile); and the one switch that explores, ``MADDENING_SEARCH_ENTROPY``,
moves the draws of a hunt and of nothing else, prints what replays it,
and replays.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
from hypothesis import strategies as st

from tests.property import targeted_search as helper
from tests.property.targeted_search import ENTROPY_VARIABLE, PER_PUSH, SLOW, targeted_search

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


# ---------------------------------------------------------------------------
# A hunt is seeded; one switch explores
# ---------------------------------------------------------------------------

#: A hunt short enough for every push (the slow profile's 800 examples are
#: not needed to tell two sets of draws apart).
_HUNT = dataclasses.replace(SLOW, max_examples=3 * PER_PUSH.max_examples)
#: What ``fresh`` draws in these tests: the per-push lane holds no randomness.
_ENTROPY = 20261008


@pytest.fixture
def seeded(monkeypatch):
    """The environment CI runs in: no switch."""
    monkeypatch.delenv(ENTROPY_VARIABLE, raising=False)


@pytest.fixture
def exploring(monkeypatch):
    """``MADDENING_SEARCH_ENTROPY=fresh`` in a process that has drawn no
    entropy yet, the draw itself pinned to :data:`_ENTROPY`."""
    monkeypatch.setenv(ENTROPY_VARIABLE, "fresh")
    monkeypatch.setattr(helper, "_fresh_entropy", None)
    monkeypatch.setattr(helper.secrets, "randbits", lambda bits: _ENTROPY)


def _hunt(seed: int, threshold: float = 2000.0, profile=_HUNT):
    return targeted_search(_PAIRS, _sum, threshold, profile=profile.seeded(seed), label="sum")


def test_a_hunt_scores_the_same_examples_every_time(seeded):
    """The slow lane's verdict on a tree: the hunt's own seed, no entropy."""
    first, again, other = _hunt(7), _hunt(7), _hunt(8)
    assert (first.seed, first.entropy) == (7, None)
    assert first.scores == again.scores
    assert (first.drawn, first.scored) == (again.drawn, again.scored)
    assert len(set(first.scores)) > 1 and first.scores != other.scores
    assert first.drawn != other.drawn and first.scored != other.scored
    assert f"(drawn {first.drawn}, scores {first.scored})" in str(first)


def test_the_two_fingerprints_tell_the_draws_from_the_scores(seeded):
    """Two flat scores give the search nothing to climb, so it draws the
    same examples under both: ``drawn`` agrees and ``scored`` does not."""
    one, two = (targeted_search(_PAIRS, lambda pair, flat=flat: (flat, None), 2000.0,
                                profile=_HUNT.seeded(7)) for flat in (1.0, 2.0))
    assert one.examples == two.examples == _HUNT.max_examples
    assert one.drawn == two.drawn and one.scored != two.scored


@pytest.mark.parametrize("switch", [None, "fresh", str(_ENTROPY)])
def test_a_random_search_with_no_seed_is_refused(monkeypatch, switch):
    """Fresh entropy has one door, the switch on a seeded hunt: a profile
    that would draw other examples on every run does not run at all."""
    monkeypatch.delenv(ENTROPY_VARIABLE, raising=False)
    if switch is not None:
        monkeypatch.setenv(ENTROPY_VARIABLE, switch)
    scored = []

    def score(pair):
        scored.append(pair)
        return 0.0, None

    for profile in (SLOW, _HUNT, dataclasses.replace(PER_PUSH, derandomize=False),
                    dataclasses.replace(SLOW.seeded(3), seed=None)):
        with pytest.raises(ValueError, match="runs under a seed of its own"):
            targeted_search(_PAIRS, score, 1.0, profile=profile)
    assert scored == []


def test_the_switch_gives_every_hunt_fresh_entropy_and_prints_what_replays_it(exploring, capsys):
    fresh, other = _hunt(7), _hunt(8)
    printed = capsys.readouterr().out
    # One draw of entropy a process, mixed into each hunt's own seed.
    assert fresh.entropy == other.entropy == _ENTROPY
    assert len({7, 8, fresh.seed, other.seed}) == 4
    assert printed.count(f"sum: exploring under {ENTROPY_VARIABLE}={_ENTROPY} ") == 2, printed
    assert "the hunt's own seed is 7" in printed and "the hunt's own seed is 8" in printed


def test_an_exploring_run_draws_other_examples_and_its_entropy_replays_it(monkeypatch, exploring):
    fresh = _hunt(7)
    monkeypatch.delenv(ENTROPY_VARIABLE)
    own = _hunt(7)
    assert own.entropy is None and own.scores != fresh.scores and own.drawn != fresh.drawn
    # The replay: the printed integer in place of ``fresh`` (no draw of
    # entropy: one would be another integer).
    monkeypatch.setattr(helper.secrets, "randbits", lambda bits: pytest.fail("a replay draws nothing"))
    monkeypatch.setenv(ENTROPY_VARIABLE, str(_ENTROPY))
    replayed = _hunt(7)
    assert (replayed.seed, replayed.entropy) == (fresh.seed, _ENTROPY)
    assert replayed.scores == fresh.scores
    assert (replayed.drawn, replayed.scored) == (fresh.drawn, fresh.scored)


def test_fresh_entropy_is_drawn_from_the_operating_system_once_a_process(monkeypatch):
    """Not from ``random``, which Hypothesis seeds inside a test."""
    monkeypatch.setenv(ENTROPY_VARIABLE, "fresh")
    monkeypatch.setattr(helper, "_fresh_entropy", None)
    draws = iter((11, 12))
    monkeypatch.setattr(helper.secrets, "randbits", lambda bits: next(draws))
    assert [helper.search_entropy() for _ in range(3)] == [11, 11, 11]


def test_a_find_under_the_switch_says_how_to_replay_it(monkeypatch, exploring):
    with pytest.raises(AssertionError) as caught:
        _hunt(0, threshold=1500.0, profile=SLOW)
    message = str(caught.value)
    assert "sum is 1501.0, over the threshold 1500.0" in message, message
    assert f"exploring under {ENTROPY_VARIABLE}={_ENTROPY} (the hunt's own seed is 0)" in message
    monkeypatch.delenv(ENTROPY_VARIABLE)
    with pytest.raises(AssertionError) as caught:
        _hunt(0, threshold=1500.0, profile=SLOW)
    assert ENTROPY_VARIABLE not in str(caught.value)


def test_the_switch_leaves_a_per_push_search_alone(monkeypatch, seeded):
    """Derandomised, or seeded off the per-push profile as the coupling
    searches' per-push tests are: the same draws whatever the switch says."""
    every_push = dataclasses.replace(PER_PUSH, max_examples=2 * PER_PUSH.max_examples).seeded(3)

    def both():
        return [targeted_search(_PAIRS, _sum, 2000.0, profile=profile)
                for profile in (PER_PUSH, every_push)]

    without = both()
    monkeypatch.setenv(ENTROPY_VARIABLE, "fresh")
    under = both()
    assert [(r.seed, r.entropy) for r in under] == [(None, None), (3, None)]
    assert [r.scores for r in under] == [r.scores for r in without]


@pytest.mark.parametrize("value", ["Fresh", "random", "1.5", "-3", "0x10", "fresh 7"])
def test_a_switch_that_is_neither_fresh_nor_an_integer_is_refused(monkeypatch, value):
    """A misspelt switch read as "seeded" would be an exploring run that
    explored nothing."""
    monkeypatch.setenv(ENTROPY_VARIABLE, value)
    with pytest.raises(ValueError, match=ENTROPY_VARIABLE):
        _hunt(7)
    # A per-push search reads no switch.
    assert targeted_search(_PAIRS, _sum, 2000.0, profile=PER_PUSH).entropy is None


def test_an_empty_switch_is_no_switch(monkeypatch):
    monkeypatch.setenv(ENTROPY_VARIABLE, " ")
    assert (_hunt(7).seed, helper.search_entropy()) == (7, None)


# --- no hunt in the tree is written without a seed --------------------------

_TESTS = Path(__file__).resolve().parents[1]
_HELPER = "tests.property.targeted_search"
#: The modules that hunt.  The scan must find a seeded hunt in each: a scan
#: that found none anywhere would pass on anything.
HUNTING = ("property/test_coupling_targeted_search.py", "property/test_coupling_nonlinear_search.py",
           "property/test_coupling_geometry_search.py", "property/test_sysid_targeted_search.py")
#: Not scanned: the helper, which defines the profiles, and this module,
#: which hands it an unseeded one to see it refused.
_NOT_SCANNED = ("property/targeted_search.py", "property/test_targeted_search.py")


def hunts_of(source: str) -> tuple:
    """``(seeded hunts, problems)`` of one test module's source.

    A hunt is a use of the slow profile.  It is seeded where ``SLOW`` is
    the receiver of ``.seeded(n)``, itself or through
    ``dataclasses.replace(SLOW, ...)``; every other use is a problem,
    ``SLOW.max_examples`` apart (a number).  So are the other ways to a
    random profile with no seed in a module that imports the helper: a
    ``Profile`` built by hand, ``SLOW`` under another name, and a call
    that sets ``derandomize`` or ``hunt``, or ``seed`` through
    ``replace``.  ``x.SLOW`` is read as the slow profile in any module."""
    tree = ast.parse(source)
    problems, seeded, receivers, numbers = [], 0, set(), set()
    names_slow = uses_helper = False

    def is_slow(node) -> bool:
        return ((names_slow and isinstance(node, ast.Name) and node.id == "SLOW"
                 and isinstance(node.ctx, ast.Load))
                or (isinstance(node, ast.Attribute) and node.attr == "SLOW"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _HELPER:
            uses_helper = True
            for alias in node.names:
                if alias.name == "SLOW" and alias.asname in (None, "SLOW"):
                    names_slow = True
                elif alias.name in ("SLOW", "Profile", "*"):
                    problems.append(f"line {node.lineno}: imports {alias.name} "
                                    f"{'as ' + alias.asname if alias.asname else ''}: a profile "
                                    "is PER_PUSH or SLOW, under its own name")
        elif isinstance(node, ast.ImportFrom) and node.module == "tests.property":
            uses_helper |= any(alias.name == "targeted_search" for alias in node.names)
        elif isinstance(node, ast.Import):
            uses_helper |= any(alias.name == _HELPER for alias in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "max_examples":
            numbers.add(id(node.value))
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "seeded":
            inside = list(ast.walk(func.value))
            if not any(is_slow(n) for n in inside):
                continue                       # a per-push profile's seed
            first = (node.args or [kw.value for kw in node.keywords if kw.arg == "seed"] or [None])[0]
            if first is None or (isinstance(first, ast.Constant) and first.value is None):
                problems.append(f"line {node.lineno}: .seeded() with no seed")
                continue
            seeded += 1
            receivers.update(id(n) for n in inside)
        elif uses_helper:
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            for kw in node.keywords:
                if kw.arg in ("derandomize", "hunt") or (kw.arg == "seed" and called == "replace"):
                    problems.append(f"line {node.lineno}: sets {kw.arg}= in a call: what a "
                                    "profile draws is set by .seeded(n) and by nothing else")
    for node in ast.walk(tree):
        if is_slow(node) and id(node) not in receivers and id(node) not in numbers:
            problems.append(f"line {node.lineno}: the slow profile with no seed (write "
                            "SLOW.seeded(n), or dataclasses.replace(SLOW, ...).seeded(n), in one "
                            "expression)")
        elif uses_helper and ((isinstance(node, ast.Name) and node.id == "Profile")
                              or (isinstance(node, ast.Attribute) and node.attr == "Profile")):
            problems.append(f"line {node.lineno}: builds or names a Profile: a profile is "
                            "PER_PUSH or SLOW with fields replaced")
    return seeded, sorted(set(problems))


def _scanned() -> dict:
    """``{path under tests/: source}`` of every test module that could
    name the slow profile (the text ``SLOW`` beside the helper's name, or
    ``.SLOW``)."""
    found = {}
    for path in sorted(_TESTS.rglob("*.py")):
        rel = path.relative_to(_TESTS).as_posix()
        text = path.read_text(encoding="utf-8")
        if rel not in _NOT_SCANNED and ((".SLOW" in text) or ("SLOW" in text
                                                             and "targeted_search" in text)):
            found[rel] = text
    return found


def test_no_hunt_in_the_tree_is_written_without_a_seed():
    """The slow lane sets no switch (``.github/workflows/slow-tests.yml``),
    so a hunt written with a seed is a hunt that draws the same examples
    on the same tree."""
    scanned = _scanned()
    assert set(HUNTING) <= set(scanned), sorted(set(HUNTING) - set(scanned))
    found = {rel: hunts_of(source) for rel, source in scanned.items()}
    problems = [f"tests/{rel}, {problem}" for rel, (_n, each) in found.items() for problem in each]
    assert not problems, (
        "a hunt with no seed draws other examples on every slow-lane run, so one tree can be "
        "green once and red the next time (to explore, set "
        f"{ENTROPY_VARIABLE}=fresh):\n  " + "\n  ".join(problems))
    without = [rel for rel in HUNTING if found[rel][0] == 0]
    assert not without, f"the scan found no seeded hunt in {without}: it no longer reads them"


def test_the_slow_lane_sets_no_switch():
    """What the scan's verdict rests on: nothing in CI turns the hunts'
    seeds into fresh entropy.  (An exploring job is a workflow of its own,
    and joins this list's exceptions when it exists.)"""
    workflows = sorted((_TESTS.parent / ".github" / "workflows").glob("*.y*ml"))
    assert workflows
    setting = [w.name for w in workflows if ENTROPY_VARIABLE in w.read_text(encoding="utf-8")]
    assert not setting, f"{ENTROPY_VARIABLE} is set in {setting}"


_IMPORT = f"from {_HELPER} import PER_PUSH, SLOW, targeted_search\nimport dataclasses\n"
_UNSEEDED = {
    "the-slow-profile-itself": _IMPORT + "targeted_search(s, f, 1.0, profile=SLOW)\n",
    "the-slow-profile-shortened": _IMPORT + (
        "profile = dataclasses.replace(SLOW, max_examples=SLOW.max_examples // 4)\n"),
    "seeded-on-another-line": _IMPORT + (
        "profile = dataclasses.replace(SLOW, max_examples=115)\nprofile = profile.seeded(3)\n"),
    "seeded-with-none": _IMPORT + "profile = SLOW.seeded(None)\n",
    "seeded-with-nothing": _IMPORT + "profile = SLOW.seeded()\n",
    "the-seed-taken-back": _IMPORT + "p = dataclasses.replace(SLOW.seeded(3), seed=None)\n",
    "under-another-name": f"from {_HELPER} import SLOW as HUNT\nprofile = HUNT\n",
    "through-the-module": "from tests.property import targeted_search as ts\nprofile = ts.SLOW\n",
    "through-another-search": (
        "from tests.property import test_coupling_targeted_search as linear\n"
        "profile = linear.SLOW\n"),
    "a-profile-built-by-hand": (
        f"from {_HELPER} import Profile\nprofile = Profile(max_examples=800, derandomize=False)\n"),
    "a-per-push-profile-made-random": _IMPORT + (
        "profile = dataclasses.replace(PER_PUSH, derandomize=False)\n"),
    "a-seeded-hunt-beside-an-unseeded-one": _IMPORT + (
        "a = SLOW.seeded(1)\nb = dataclasses.replace(SLOW, max_examples=9)\n"),
}
_SEEDED = {
    "seeded": (_IMPORT + "targeted_search(s, f, 1.0, profile=SLOW.seeded(4000 + i))\n", 1),
    "shortened-and-seeded": (_IMPORT + (
        "profile = dataclasses.replace(SLOW, max_examples=SLOW.max_examples // 7 + 1).seeded(\n"
        "    3000 + 10 * block, shrink=False)\n"), 1),
    "through-the-module": ("from tests.property import targeted_search as ts\n"
                           "profile = ts.SLOW.seeded(seed=5)\n", 1),
    "a-number-of-the-slow-profile": (_IMPORT + "n = SLOW.max_examples // 4\n", 0),
    "a-per-push-seed": (_IMPORT + "profile = dataclasses.replace(PER_PUSH, max_examples=150).seeded(2)\n",
                        0),
    "a-list-of-slow-cases-in-a-module-without-the-helper": ("SLOW = [1, 2]\ncases = SLOW + [3]\n", 0),
}


@pytest.mark.parametrize("name", sorted(_UNSEEDED))
def test_the_scan_finds_a_hunt_written_without_a_seed(name):
    _seeded, problems = hunts_of(_UNSEEDED[name])
    assert problems, f"the scan passed:\n{_UNSEEDED[name]}"


@pytest.mark.parametrize("name", sorted(_SEEDED))
def test_the_scan_passes_a_hunt_written_with_a_seed(name):
    source, hunts = _SEEDED[name]
    assert hunts_of(source) == (hunts, [])
