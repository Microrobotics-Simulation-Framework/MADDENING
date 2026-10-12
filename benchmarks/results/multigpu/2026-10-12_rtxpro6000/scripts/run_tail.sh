#!/bin/bash
# The stress tail (benchmarks/multigpu/README.md, "After the checklist"), gates nothing.
# Deviations agreed 2026-10-12: --device-memory-gb 96 (the maintainer; the runbook's 24 was for 4090s),
# --rung-timeout-s 360 and a 40-min box for the 2x2 ramp (cards 4x larger), no 30-min tail cap
# (the 2-h cap is lifted), and exchange --fields 5 first (the runbook's "if time allows").
source ~/session_env.sh
T=results/stress; mkdir -p $T; P=$T/progress.txt
step() { local name=$1 box=$2 log=$3; shift 3
  echo "START $name $(date -u +%H:%M:%SZ)" >> $P
  timeout $box "$@" > $log 2>&1; local rc=$?
  echo "END $name rc=$rc $(date -u +%H:%M:%SZ)" >> $P; }
big() { step "$2-$3" "$1" $T/$2-$3.log python3 benchmarks/multigpu/run_pod.py --goal "$2" --cells "$3" \
          --partition contiguous --warmup 1 --repeats 3 --out $T/cells-$3; }
step exchange-f5 10m results/multigpu-f5.log python3 benchmarks/multigpu/run_pod.py --goal exchange --fields 5 --out results/multigpu-f5
big 2m forward  10000000
big 2m exchange 10000000
big 4m gradient 10000000
big 4m forward  30000000
big 4m exchange 30000000
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
C="python3 benchmarks/multigpu/run_capacity.py --device-memory-gb 96 --rung-timeout-s 360"
step capacity-2x2 40m results/capacity-2x2.log $C --mesh 2x2 --ramp 0.25 0.5 0.75 0.85 0.9 --soak-minutes 3 --out results/capacity-2x2
step capacity-4x1 15m results/capacity-4x1.log $C --mesh 4x1 --ramp 0.25 0.85 --out results/capacity-4x1
python3 benchmarks/multigpu/run_capacity.py --summarise results/capacity-2x2 > results/capacity-2x2.txt 2>&1; echo "SUMMARY capacity-2x2 rc=$?" >> $P
python3 benchmarks/multigpu/run_capacity.py --summarise results/capacity-4x1 > results/capacity-4x1.txt 2>&1; echo "SUMMARY capacity-4x1 rc=$?" >> $P
unset XLA_PYTHON_CLIENT_MEM_FRACTION
step stencil-default-precision 10m $T/stencil-default-precision.log env -u JAX_DEFAULT_MATMUL_PRECISION \
  python3 benchmarks/multigpu/run_pod.py --goal stencil --cells 100000 --out $T/stencil-default-precision
big 9m forward 100000000
echo "ALL DONE $(date -u +%H:%M:%SZ)" >> $P
