#!/bin/bash
# Mutation-test the non-finite gates.  Each mutation is applied, the gates are
# run, and the mutation is reverted with `git checkout --` (the worktree has no
# uncommitted real edits, so this is safe).
WT=/home/nick/MSF/msf/MADDENING-wt/audit-r2-serialization
cd $WT || exit 1
GATES="tests/core/test_non_finite_json_tokens.py tests/usd/test_usd_params.py tests/fmi/test_non_finite_json_tokens.py tests/fmi/test_binary_frames_properties.py"
run_gates () {
  PYTHONPATH=$WT/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    timeout 900 /home/nick/MSF/msf/.venv/bin/python -m pytest $GATES -q -p no:cacheprovider 2>&1 | tail -3
}
apply () { python3 - "$@" <<'PY'
import sys, pathlib
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path); s = p.read_text()
assert s.count(old) == 1, f"anchor count {s.count(old)} for {old!r}"
p.write_text(s.replace(old, new)); print("   applied to", path)
PY
}
mut () {
  echo "=================================================================="
  echo "MUTATION: $1"
  apply "$2" "$3" "$4" || { echo "   ANCHOR FAILED"; return; }
  run_gates
  git checkout -- "$2"
}

C=src/maddening/serialization/json_codec.py

mut "M1 encode: stop recursing into tuples (list only)" $C \
  '    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")' \
  '    if isinstance(obj, list):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")'

mut "M2 encode: drop the ambiguous-string refusal" $C \
  '        if obj in NON_FINITE_TOKENS:
            raise ValueError(' \
  '        if False and obj in NON_FINITE_TOKENS:
            raise ValueError('

mut "M3 decode: do not decode inside lists" $C \
  '    if isinstance(obj, (list, tuple)):
        out = [decode_non_finite(v) for v in obj]' \
  '    if isinstance(obj, list) and False:
        out = [decode_non_finite(v) for v in obj]'

mut "M4 dumps: allow_nan=True (bare tokens back)" $C \
  '    kwargs.pop("allow_nan", None)
    return json.dumps(encode_non_finite(obj), allow_nan=False, **kwargs)' \
  '    kwargs.pop("allow_nan", None)
    return json.dumps(obj, allow_nan=True, **kwargs)'

mut "M5 SIGN FLIP: Infinity decodes to -inf" $C \
  '_DECODE = {NAN_TOKEN: math.nan, INF_TOKEN: math.inf, NEG_INF_TOKEN: -math.inf}' \
  '_DECODE = {NAN_TOKEN: math.nan, INF_TOKEN: -math.inf, NEG_INF_TOKEN: math.inf}'

mut "M6 SIGN FLIP: -inf encodes as Infinity" $C \
  '    return INF_TOKEN if value > 0 else NEG_INF_TOKEN' \
  '    return INF_TOKEN'

mut "M7 encode: NaN encodes as the inf token" $C \
  '    if math.isnan(value):
        return NAN_TOKEN' \
  '    if math.isnan(value):
        return INF_TOKEN'

mut "M8 decode: drop the token table entirely (identity)" $C \
  '    if isinstance(obj, str):
        return _DECODE.get(obj, obj)' \
  '    if isinstance(obj, str):
        return obj'

echo "=================================================================="
echo "final git status (must be clean):"
git status --porcelain
