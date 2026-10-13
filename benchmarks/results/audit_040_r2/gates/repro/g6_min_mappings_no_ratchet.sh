#!/bin/bash
# G6: MIN_MAPPINGS is the only thing protecting the Implementation Mapping
# tables, and nothing stops it being lowered.  Drop the heat_node pin 9 -> 1,
# delete 8 of its 9 mapping rows: gate green, TestImplementationMappingGate green.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
cd "$WT"
S=scripts/check_impl_mapping.py; G=docs/algorithm_guide/nodes/heat_node.md
BK=$(mktemp -d); cp $S $BK/; cp $G $BK/
trap 'cp $BK/$(basename $S) $S; cp $BK/$(basename $G) $G; rm -rf "$BK"' EXIT
sed -i 's|"heat_node.md"): 9,|"heat_node.md"): 1,|' $S
$PY - <<'PYEOF'
p = "docs/algorithm_guide/nodes/heat_node.md"
kept, n = [], 0
for line in open(p).read().split("\n"):
    if line.startswith("|") and "`maddening." in line:
        n += 1
        if n > 1:
            continue
    kept.append(line)
open(p, "w").write("\n".join(kept))
print(f"removed {n - 1} mapping rows from heat_node.md")
PYEOF
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_impl_mapping.py; echo "gate rc=$?"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu PYTHONPATH=$WT/src \
  $PY -m pytest tests/compliance/test_gate_scripts.py -q -p no:cacheprovider -k Mapping 2>&1 | tail -2
