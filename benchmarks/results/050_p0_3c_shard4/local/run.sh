#!/bin/bash
# usage: run.sh NAME CORES -- pytest args...   (runs in the worktree, CI-like env, capped)
S=/tmp/claude-1000/-home-nick-MSF-msf-MADDENING/6705ca06-9e31-48a7-911f-0df1e90c341b/scratchpad/shard4
WT=/home/nick/MSF/msf/MADDENING-wt/slow-shard4
name=$1; cores=$2; shift 3
cd $WT
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_FLAGS=--xla_gpu_autotune_level=0 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MADDENING_TEST_JAX_TIMING=1 PYTHONPATH=$WT/src HOME=$S/home
start=$(date +%s)
systemd-run --user --scope -q -p MemoryMax=12G -- taskset -c $cores \
  /home/nick/MSF/msf/.venv/bin/python $S/monitor.py $S/runs/$name.mon 5 -- \
  /home/nick/MSF/msf/.venv/bin/python -m pytest -p no:cacheprovider -m 'slow or not slow' -v -rs --tb=short \
  --durations=0 --durations-min=1.0 --junitxml=$S/runs/$name.xml -o junit_family=xunit1 "$@" > $S/runs/$name.log 2>&1
echo "exit $? wall $(( $(date +%s) - start )) s" | tee -a $S/runs/$name.log
