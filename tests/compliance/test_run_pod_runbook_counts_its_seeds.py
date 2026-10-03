"""The multi-GPU runbook counts the seeded wrapper faults the seeded test holds.

``benchmarks/multigpu/README.md`` says how many faults
``tests/cloud/multigpu/test_run_pod_seeded_faults.py`` seeds into the stencil
wrapper; both it and the runner's docstring once counted seven, while the test
held eight (RPD-024 in ``docs/validation/rest_runpod_claims.yaml``).  The
runner's own count is checked beside the runner
(``tests/cloud/multigpu/test_run_pod_documented_edges.py``); the runbook is
documentation, which a docs-only change runs only the compliance tests on, so
its count is checked here, from the two files' text.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "benchmarks" / "multigpu" / "README.md"
SEEDED = REPO / "tests" / "cloud" / "multigpu" / "test_run_pod_seeded_faults.py"
WORDS = {"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def test_the_runbook_counts_every_seed_and_names_the_eighth():
    seeds = re.findall(r'^    "([a-z_0-9]+)": Seed\(', SEEDED.read_text(encoding="utf-8"), re.M)
    text = RUNBOOK.read_text(encoding="utf-8")
    said = re.search(r"test_run_pod_seeded_faults\.py` holds (\w+) faults", text)
    assert said, "the runbook no longer says how many faults the seeded test holds"
    assert WORDS[said.group(1)] == len(seeds) == len(set(seeds)), (said.group(1), seeds)
    assert "domain integrals summed over the first mesh axis only" in text.replace("\n", " ")
