#!/bin/bash
# Mutation harness: applies a sed mutation to a scratch COPY of src
# (never the worktree), runs the named tests against it, and reports.
# $1 = file (relative to src), $2 = old, $3 = new, $4... = pytest args
SCR=/tmp/claude-1000/-home-nick-MSF-msf-MADDENING/6705ca06-9e31-48a7-911f-0df1e90c341b/scratchpad
WT=/home/nick/MSF/msf/MADDENING-wt/audit-r2-numerics
rm -rf $SCR/mutsrc && cp -r $WT/src $SCR/mutsrc
f="$SCR/mutsrc/$1"; old="$2"; new="$3"; shift 3
before=$(grep -cF -- "$old" "$f")
if [ "$before" -eq 0 ]; then echo "MUTATION DID NOT APPLY: '$old' not in $f"; exit 2; fi
python3 - "$f" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert old in s
open(p, "w").write(s.replace(old, new))
PY
echo "### mutation applied ($before site(s)) in $1"
cd $WT && PYTHONPATH=$SCR/mutsrc JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nick/MSF/msf/.venv/bin/python -m pytest "$@" -q -p no:cacheprovider -x --no-header 2>&1 | tail -25
