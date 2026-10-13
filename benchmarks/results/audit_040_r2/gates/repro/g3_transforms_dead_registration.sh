#!/bin/bash
# G3: check_transforms.py accepts a register_transform() that never executes.
# The AST scan looks for the call lexically; it does not check the name is in
# the registry after import.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
cd "$WT"
probe=tests/core/_g3_probe.py
trap 'rm -f "$probe"' EXIT
cat > $probe <<'PYEOF'
from maddening.core.transforms import register_transform


def _install_later():
    """Never called by anything -- e.g. a helper a fixture forgot to invoke."""
    @register_transform("phantom_transform")
    def _phantom(x):
        return x


def f(gm):
    gm.add_edge("a", "b", "x", "y", transform="phantom_transform")
PYEOF
PYTHONPATH=$WT/src $PY scripts/check_transforms.py; echo "gate rc=$?"
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY - <<'PYEOF'
import tests.core._g3_probe            # noqa: F401  (import fires module level)
from maddening.core.transforms import _TRANSFORM_REGISTRY
print("in registry after import?", "phantom_transform" in _TRANSFORM_REGISTRY)
PYEOF
echo "^ gate said verified; registry says no. add_edge would raise KeyError."
