#!/bin/bash
# The maintainer's additions before teardown (2026-10-12), gate nothing:
#  1. the LBM matmul precision fix: the patch on a COPY of the checkout, stencil at 1e5
#     WITHOUT JAX_DEFAULT_MATMUL_PRECISION (was 348/356 at the default, 376/376 at highest);
#  2. graph vs eager at 1e7, and profile_graph(trace=True) on 4 cards sharded and 1 unsharded.
source ~/session_env.sh
P=~/extra_progress.txt
echo "START precision-fix $(date -u +%H:%M:%SZ)" >> $P
rm -rf ~/precision_fix && cp -a ~/sky_workdir ~/precision_fix && rm -rf ~/precision_fix/results
cd ~/precision_fix && git apply ~/lbm_matmul_precision.patch && git diff --stat >> $P 2>&1
mkdir -p ~/precision_fix_results
PYTHONPATH=~/precision_fix/src python3 -c "import maddening; print('imports', maddening.__file__)" >> $P 2>&1
env -u JAX_DEFAULT_MATMUL_PRECISION PYTHONPATH=~/precision_fix/src timeout 20m python3 -u \
  benchmarks/multigpu/run_pod.py --goal stencil --cells 100000 \
  --out ~/precision_fix_results/stencil-default-precision-fixed \
  > ~/precision_fix_results/stencil-default-precision-fixed.log 2>&1
echo "END precision-fix rc=$? $(date -u +%H:%M:%SZ)" >> $P
echo "START graph-vs-eager $(date -u +%H:%M:%SZ)" >> $P
cd ~/sky_workdir && mkdir -p ~/gve_results
timeout 40m python3 -u ~/graph_vs_eager.py --cells 10000000 --steps 20 --profile-steps 50 \
  --profile-dir ~/gve_results/profile --out ~/gve_results/gve.json > ~/gve_results/gve.log 2>&1
echo "END graph-vs-eager rc=$? $(date -u +%H:%M:%SZ)" >> $P
echo "ALL DONE $(date -u +%H:%M:%SZ)" >> $P
