"""MADD-ANO-005's key list is the coupling diagnostics' key set, not a paraphrase.

The registry entry states "the reported keys are exactly [...]".  A
sentence like that goes stale silently when a key is added, so it is read
back and compared with what ``coupling_diagnostics()`` returns.

This lives under ``tests/compliance`` because it reads
``docs/validation/known_anomalies.yaml``: a pull request that touches only
documentation runs the compliance job alone (``.github/workflows/ci.yml``,
the ``changes`` job), so an edit to the entry must meet this test there.
The graph is the one ``tests/core/test_coupling_gradient_error_bound.py``
pins the gradient bound on, imported rather than copied so the two cannot
drift apart.
"""

import re
from pathlib import Path

import yaml

from tests.core.test_coupling_gradient_error_bound import _curved_graph

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_anomaly_registry_quotes_the_reported_keys_exactly():
    registry = yaml.safe_load(
        (REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml").read_text())
    entry = next(a for a in registry["anomalies"] if a["anomaly_id"] == "MADD-ANO-005")
    match = re.search(r"reported keys are exactly\s*\[([^\]]*)\]", entry["description"])
    assert match, "MADD-ANO-005 no longer quotes the key list"
    quoted = set(re.findall(r"'([a-z_]+)'", match.group(1)))
    gm = _curved_graph("log", max_iterations=4)
    gm.step()
    assert quoted == set(gm.coupling_diagnostics()["a+b"])
