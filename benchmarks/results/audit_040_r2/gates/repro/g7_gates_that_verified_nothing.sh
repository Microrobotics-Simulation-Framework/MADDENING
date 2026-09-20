#!/bin/bash
# G7: three gates report success having verified nothing.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cd "$WT"

echo "=== check_anomalies.py on an EMPTY registry ==="
cat > $T/empty.yaml <<'YAML'
schema_version: "1.0"
maddening_version: "0.4.0"
generated_date: "2026-09-20"
anomalies: []
YAML
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_anomalies.py $T/empty.yaml --prefix MADD-ANO-
echo "rc=$?"

echo
echo "=== check_anomalies.py on a registry with every reference stripped ==="
$PY - "$T" <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open("docs/validation/known_anomalies.yaml"))
for a in d["anomalies"]:
    a.pop("affected_components", None); a.pop("verification", None)
yaml.safe_dump(d, open(sys.argv[1] + "/noref.yaml", "w"), sort_keys=False, width=10000)
PYEOF
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_anomalies.py $T/noref.yaml --prefix MADD-ANO-
echo "rc=$?"

echo
echo "=== check_impl_mapping.py on an empty scope ==="
mkdir -p $T/guides
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_impl_mapping.py $T/guides
echo "rc=$?"

echo
echo "=== audit_property_rejection.py --check where every property test skips ==="
mkdir -p $T/props
cat > $T/props/test_all_skipped.py <<'PYEOF'
import pytest
from hypothesis import given, strategies as st


@pytest.mark.skip(reason="stand-in for a suite skipped for a missing extra")
@given(st.integers())
def test_property_that_never_runs(x):
    assert x == x
PYEOF
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  MADDENING_HYPOTHESIS_PROFILE=ci $PY scripts/audit_property_rejection.py $T/props --check 2>&1 | tail -6
echo "rc=${PIPESTATUS[0]}"
echo
echo "Compare: check_heat_stability.py and check_transforms.py both fail closed here."
