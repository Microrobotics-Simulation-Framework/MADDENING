#!/bin/bash
# G2: check_transforms.py cannot see a transform passed positionally.
# `transform` is the 5th positional parameter of GraphManager.add_edge and the
# 5th field of EdgeSpec, and a string there is resolved at runtime.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
cd "$WT"
probe=tests/core/_g2_probe.py
trap 'rm -f "$probe"' EXIT

echo "--- keyword form (the gate sees it) ---"
cat > $probe <<'PYEOF'
def f(gm):
    gm.add_edge("a", "b", "x", "y", transform="definitely_not_registered")
PYEOF
PYTHONPATH=$WT/src $PY scripts/check_transforms.py; echo "rc=$?"

echo "--- positional form, identical meaning (the gate does not) ---"
cat > $probe <<'PYEOF'
def f(gm):
    gm.add_edge("a", "b", "x", "y", "definitely_not_registered")
PYEOF
PYTHONPATH=$WT/src $PY scripts/check_transforms.py; echo "rc=$?   <-- 0, the bogus name is invisible"
