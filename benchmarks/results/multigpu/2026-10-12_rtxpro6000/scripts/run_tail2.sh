#!/bin/bash
# Reruns of the tail steps the runbook's CPU-estimated boxes could not hold (host-side setup grows
# with the cell count), with realistic boxes and unbuffered output.  Gates nothing.
source ~/session_env.sh
T=results/stress; P=$T/progress.txt
step() { local name=$1 box=$2 log=$3; shift 3
  echo "START $name $(date -u +%H:%M:%SZ)" >> $P
  timeout $box "$@" > $log 2>&1; local rc=$?
  echo "END $name rc=$rc $(date -u +%H:%M:%SZ)" >> $P; }
big() { step "$2-$3-rerun" "$1" $T/$2-$3-rerun.log python3 -u benchmarks/multigpu/run_pod.py --goal "$2" --cells "$3" \
          --partition contiguous --warmup 1 --repeats 3 --out $T/cells-$3; }
big 15m gradient 10000000
big 15m forward  30000000
big 30m forward  100000000
echo "RERUNS DONE $(date -u +%H:%M:%SZ)" >> $P
