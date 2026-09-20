#!/bin/bash
# Mutation-test scripts/check_heat_stability.py.  Each probe file goes in a
# scope of its own; the last one is the positive control.
WT=/home/nick/MSF/msf/MADDENING-wt/audit-r2-numerics
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
for f in a_nonpositive b_unknown_order c_positional_stencil d_attribute_call e_genuinely_unstable; do
  rm -f $TMP/*.py; cp $HERE/gate_probes/$f.py $TMP/
  echo "--- $f ---"
  ( cd $WT && PYTHONPATH=$WT/src JAX_PLATFORMS=cpu \
      /home/nick/MSF/msf/.venv/bin/python scripts/check_heat_stability.py $TMP )
  echo "    exit=$?"
done
# (c) in a mixed scope: the zero-scope guard no longer fires
rm -f $TMP/*.py; cp $HERE/gate_probes/d_attribute_call.py $TMP/
cat > $TMP/ok.py <<'PY'
from maddening.nodes.heat import HeatNode
n = HeatNode("ok", timestep=1e-4, n_cells=10, length=1.0, thermal_diffusivity=0.01)
PY
echo "--- attribute-spelled unstable rod + one valid construction ---"
( cd $WT && PYTHONPATH=$WT/src JAX_PLATFORMS=cpu \
    /home/nick/MSF/msf/.venv/bin/python scripts/check_heat_stability.py $TMP )
echo "    exit=$?"
rm -rf $TMP
