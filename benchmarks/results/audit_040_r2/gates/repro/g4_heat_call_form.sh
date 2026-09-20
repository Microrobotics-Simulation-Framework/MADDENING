#!/bin/bash
# G4: check_heat_stability.py only matches `HeatNode(...)` as a bare Name.
# Two unstable rods (Fourier 0.66 > 0.5) written `heat.HeatNode(...)` and via an
# aliased import are invisible; with the repo's own calls in scope the gate is green.
set -u
WT=${1:-/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates}
PY=/home/nick/MSF/msf/.venv/bin/python
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
cat > $D/attribute_form.py <<'PYEOF'
from maddening.nodes import heat
n = heat.HeatNode("rod", timestep=1e-4, n_cells=257, length=1.0, thermal_diffusivity=0.1)
PYEOF
cat > $D/alias_form.py <<'PYEOF'
from maddening.nodes.heat import HeatNode as Rod
n = Rod("rod", timestep=1e-4, n_cells=257, length=1.0, thermal_diffusivity=0.1)
PYEOF
cd "$WT"
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY scripts/check_heat_stability.py tests "$D"
echo "rc=$?   <-- 0, with two rods in scope that HeatNode.__init__ refuses"
JAX_PLATFORMS=cpu PYTHONPATH=$WT/src $PY -c "
from maddening.nodes import heat
try:
    heat.HeatNode('rod', timestep=1e-4, n_cells=257, length=1.0, thermal_diffusivity=0.1)
except ValueError as e:
    print('runtime:', str(e)[:90])"
