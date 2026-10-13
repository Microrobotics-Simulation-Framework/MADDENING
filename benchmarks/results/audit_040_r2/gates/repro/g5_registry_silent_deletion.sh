#!/bin/bash
# G5: nothing pins the *membership* of the anomaly or benchmark registries.
# Delete an entry, run the generator the equality gate's own error message tells
# you to run, and every gate is green with the entry gone from the evidence set.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
cd "$WT"
R=docs/validation/known_anomalies.yaml
F=tests/verification/test_mms_order_ode_nodes.py
BK=$(mktemp -d); cp $R $BK/; cp $F $BK/
restore() { cp $BK/$(basename $R) $R; cp $BK/$(basename $F) $F;
            git checkout -- docs/validation/ 2>/dev/null; rm -rf "$BK"; }
trap restore EXIT

echo "=== A. delete an OPEN anomaly from the registry ==="
$PY - <<'PYEOF'
import yaml
p = "docs/validation/known_anomalies.yaml"
d = yaml.safe_load(open(p))
d["anomalies"] = [a for a in d["anomalies"] if a["anomaly_id"] != "MADD-ANO-012"]
yaml.safe_dump(d, open(p, "w"), sort_keys=False, width=10000)
PYEOF
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_anomalies.py --prefix MADD-ANO-; echo "check_anomalies rc=$?"
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/generate_soup_tables.py >/dev/null 2>&1
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/generate_soup_tables.py --check >/dev/null 2>&1
echo "soup --check after regenerating rc=$?"
echo -n "MADD-ANO-012 in soup_package.md: "; grep -c "MADD-ANO-012" docs/validation/soup_package.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src \
  $PY -m pytest tests/compliance/ -q -p no:cacheprovider 2>&1 | tail -2
restore; trap - EXIT; BK=$(mktemp -d); cp $R $BK/; cp $F $BK/
trap 'cp $BK/$(basename $R) $R; cp $BK/$(basename $F) $F; git checkout -- docs/validation/ 2>/dev/null; rm -rf "$BK"' EXIT

echo
echo "=== B. two @verification_benchmark decorators with the same id ==="
sed -i 's/benchmark_id="MADD-VER-011"/benchmark_id="MADD-VER-010"/' $F
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/generate_soup_tables.py >/dev/null 2>&1
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/generate_soup_tables.py --check >/dev/null 2>&1
echo "soup --check after regenerating rc=$?"
echo -n "MADD-VER-011 in framework_verification.md: "; grep -c "MADD-VER-011" docs/validation/framework_verification.md
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src \
  $PY -m pytest tests/compliance/test_soup_evidence.py -q -p no:cacheprovider 2>&1 | tail -2
