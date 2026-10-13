"""The rules every entry of ``coupling_diagnostics()`` keeps about its
reason codes, as one assertion other test modules call on the reports
they already read (a helper module: it holds no test).

``assert_reason_rules(report)`` is called by the every-domain battery's
report reader (``tests/core/coupling_domains.py``), by the linear and
the geometry searches on each report they judge, and by
``tests/core/test_coupling_report_reason_codes.py``, which also seeds
the faults that show it can fail.
"""

from __future__ import annotations

import json
import math
import re

from maddening.core.coupling import reason_codes as rc

#: The two flags a ``False`` reading of which must come with a cause.
USABLE = ("spectral_usable", "gradient_bound_usable")

#: The keys of a report that are not numbers or flags: what a comparison
#: of two reports number by number leaves to :func:`same_causes`.
WORDS = ("not_usable_reason", "reason_codes")

#: The two codes of an estimate that did not settle.  Which of them an
#: entry carries follows ``compile()``'s structural count of the pass, so
#: a graph and a twin of another structure (an edge-mapped pair and its
#: node-inlined twin) may differ in it while every number agrees.
UNSETTLED = frozenset({rc.INTERFACE_TOO_WIDE, rc.SPECTRAL_SELF_CHECK_FAILED})


def assert_reason_rules(report, where="") -> None:
    """*report* (one group's entry) keeps the rules of its reason codes.

    * ``"reason_codes"`` is a dict with one list for each of
      ``reason_codes.FLAGS``, of codes of ``reason_codes.ALL``, each once,
      in that order;
    * a usable flag that is ``False`` has at least one code and the entry
      a ``"not_usable_reason"`` (a non-empty string); a flag that is
      ``True`` has none, and with both ``True`` there is no reason;
    * the gradient flag's list holds the spectral flag's codes;
    * ``"precision_limited"`` has a code exactly where
      ``"residual_precision_floor"`` is NaN, and the flag is then
      ``False``;
    * the codes are what a JSON round trip gives back.
    """
    codes = report["reason_codes"]
    assert isinstance(codes, dict) and tuple(codes) == rc.FLAGS, (where, codes)
    order = {code: i for i, code in enumerate(rc.ALL)}
    for flag, listed in codes.items():
        assert isinstance(listed, list), (where, flag, listed)
        assert all(isinstance(code, str) and code in order for code in listed), (
            where, flag, listed)
        assert listed == sorted(set(listed), key=order.__getitem__), (where, flag, listed)
    reason = report.get("not_usable_reason")
    for flag in USABLE:
        value = report[flag]
        assert isinstance(value, bool), (where, flag, value)
        if value:
            assert codes[flag] == [], (where, flag, "a True flag carries", codes[flag])
        else:
            assert codes[flag], (where, flag, "is False with no code", dict(report))
            assert isinstance(reason, str) and reason.strip(), (
                where, flag, "is False with no sentence", dict(report))
    if all(report[flag] for flag in USABLE):
        assert reason is None, (where, "both flags stand beside a reason", reason)
    if not report["gradient_bound_usable"]:
        assert set(codes["spectral_usable"]) <= set(codes["gradient_bound_usable"]), (
            where, codes)
    assert not (report["spectral_usable"] and not report["gradient_bound_usable"]
                and not set(codes["gradient_bound_usable"]) <= {
                    rc.GRADIENT_BOUND_NOT_COMPUTED, rc.GRADIENT_BOUND_NOT_CERTIFIED}), (
        where, "a gradient flag that is False by itself has a cause of its own", codes)
    floor = report["residual_precision_floor"]
    assert isinstance(floor, float), (where, floor)
    assert bool(codes["precision_limited"]) == math.isnan(floor), (
        where, "a floor that was not measured, and only that, has a code", floor, codes)
    if codes["precision_limited"]:
        assert report["precision_limited"] is False, (where, dict(report))
    assert json.loads(json.dumps(codes)) == codes, (where, codes)


def folded_codes(codes) -> dict:
    """A report's ``"reason_codes"`` with the two codes of an unsettled
    estimate (:data:`UNSETTLED`) read as one."""
    return {flag: sorted({"unsettled" if code in UNSETTLED else code for code in listed})
            for flag, listed in codes.items()}


def same_causes(a, b) -> bool:
    """Do the reports *a* and *b* give the same causes?  Their
    ``"reason_codes"`` are equal once folded (:func:`folded_codes`), and
    both have a ``"not_usable_reason"`` or neither has."""
    return (folded_codes(a["reason_codes"]) == folded_codes(b["reason_codes"])
            and ("not_usable_reason" in a) == ("not_usable_reason" in b))


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?|\binf\b|\bnan\b")


def same_words(a: str, b: str, exact: bool) -> bool:
    """Are two reasons the same sentence?  Equal; or, where the two
    reports are compared to a tolerance (not *exact*), equal once the
    numbers they quote are left out (a residual printed to three digits
    differs in its last between two programs that agree to 1e-6)."""
    return a == b or (not exact and _NUMBER.sub("#", a) == _NUMBER.sub("#", b))
