"""The guide's table of reason codes is the library's list.

``coupling_diagnostics()`` gives each ``False`` usable flag a list of
codes (``maddening.core.coupling.reason_codes``), and the developer
guide's section "The reason codes of a report" says what each means and
what to do about it.  A code added without a row, or a row left behind by
a code that was renamed, would leave a caller with a string nothing
explains.  This reads a file under ``docs/``, so it lives here: a
docs-only pull request runs ``tests/compliance`` and no other test.
"""

from __future__ import annotations

import re
from pathlib import Path

from maddening.core.coupling import reason_codes

GUIDE = Path(__file__).resolve().parents[2] / "docs/developer_guide/coupling_algorithm_guide.md"
SECTION = "### The reason codes of a report"


def _rows() -> list:
    text = GUIDE.read_text()
    assert text.count(SECTION) == 1, "the guide's section on the reason codes is gone"
    body = text.split(SECTION, 1)[1].split("\n### ", 1)[0]
    return re.findall(r"^\| `([a-z_]+)` \| [^|]+ \| [^|]+ \|$", body, re.M)


def test_every_reason_code_has_one_row_in_the_guide_s_table_and_no_row_is_left_over():
    rows = _rows()
    assert sorted(rows) == sorted(reason_codes.ALL), (
        sorted(set(reason_codes.ALL) - set(rows)), sorted(set(rows) - set(reason_codes.ALL)))
    # In the order a report lists them.
    assert rows == list(reason_codes.ALL)


def test_the_user_guide_points_at_the_table():
    inspection = (GUIDE.parents[1] / "user_guide/inspection.md").read_text()
    assert "coupling_algorithm_guide.md#the-reason-codes-of-a-report" in inspection
    assert "`reason_codes`" in inspection and "`residual_precision_floor`" in inspection
