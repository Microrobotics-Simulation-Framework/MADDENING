#!/bin/bash
# Section 2b of benchmarks/multigpu/README.md, in its order and time boxes;
# stops at the first non-zero exit (section 3).  Output as the runbook's tee.
source ~/session_env.sh
R="python3 benchmarks/multigpu/run_pod.py --out results/multigpu"
L=results/multigpu
M=results/multigpu-mesh; mkdir -p $M
run() { local name=$1 box=$2 log=$3; shift 3
  echo "START $name $(date -u +%H:%M:%SZ)" >> $L/progress.txt
  timeout $box "$@" > $log 2>&1; local rc=$?
  echo "END $name rc=$rc $(date -u +%H:%M:%SZ)" >> $L/progress.txt; return $rc; }
run indivisible 10m $L/indivisible.log $R --goal indivisible || exit 1
run halo        10m $L/halo.log        $R --goal halo        || exit 1
run coupled     25m $L/coupled.log     $R --goal coupled     || exit 1
run stencil     45m $L/stencil.log     $R --goal stencil     || exit 1
run hybrid      20m $L/hybrid.log      $R --goal hybrid      || exit 1
run exchange    10m $L/exchange.log    $R --goal exchange    || exit 1
run forward     20m $L/forward.log     $R --goal forward     || exit 1
run gradient    20m $L/gradient.log    $R --goal gradient    || exit 1
run forward-mesh 25m $M/forward.log python3 benchmarks/multigpu/run_pod.py --out $M --goal forward --mesh ~/vessel_tet_1001664.npz || exit 1
python3 benchmarks/multigpu/run_pod.py --summarise results/multigpu > $L/summary.txt 2>&1; echo "SUMMARY rc=$?" >> $L/progress.txt
python3 benchmarks/multigpu/run_pod.py --summarise $M > $M/summary.txt 2>&1; echo "SUMMARY-MESH rc=$?" >> $L/progress.txt
echo "ALL DONE $(date -u +%H:%M:%SZ)" >> $L/progress.txt
