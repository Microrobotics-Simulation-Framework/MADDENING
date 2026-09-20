#!/bin/bash
WT=/home/nick/MSF/msf/MADDENING-wt/audit-r2-serialization
cd $WT || exit 1
GATES="tests/core/test_non_finite_json_tokens.py tests/usd/test_usd_params.py tests/fmi/test_non_finite_json_tokens.py tests/fmi/test_binary_frames_properties.py tests/fmi/test_c_unit.py"
run_gates () {
  PYTHONPATH=$WT/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    timeout 900 /home/nick/MSF/msf/.venv/bin/python -m pytest $GATES -q -p no:cacheprovider 2>&1 | tail -4
}
apply () { python3 - "$@" <<'PY'
import sys, pathlib
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path); s = p.read_text()
assert s.count(old) == 1, f"anchor count {s.count(old)}"
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
CC=src/maddening/fmi/c/maddening_fmu.c

echo "### BASELINE"; run_gates

mut "M9 encode: a tuple comes back as a list" $C \
  '        return tuple(out) if isinstance(obj, tuple) else out' \
  '        return out'

mut "M10 encode: identity shortcut always copies" $C \
  '        return obj if all(a is b for a, b in zip(out.values(), obj.values())) else out
    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")' \
  '        return out
    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")'

mut "M11 encode: identity shortcut always returns the ORIGINAL dict" $C \
  '        return obj if all(a is b for a, b in zip(out.values(), obj.values())) else out
    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")' \
  '        return obj
    if isinstance(obj, (list, tuple)):
        out = [encode_non_finite(v, _path=f"{_path}[{i}]")'

mut "M12 C parse_values: do not step over the opening quote" $CC \
  '        int quoted = (*p == '"'"'"'"'"');
        if (quoted) ++p;' \
  '        int quoted = 0;'

mut "M13 C parse_values: drop the closing-quote check" $CC \
  '            if (*p != '"'"'"'"'"') {
                inst_log(in, fmi3Error, "logStatusError",
                         "maddening_fmu: malformed quoted number in reply");
                return fmi3Error;
            }' \
  '            if (0) { return fmi3Error; }'

mut "M14 C do_set: allow non-finite through the JSON path" $CC \
  '        if (isnan(values[i]) || isinf(values[i])) {' \
  '        if (0 \&\& (isnan(values[i]) || isinf(values[i]))) {'

echo "=================================================================="
echo "final git status (must be clean):"
git status --porcelain
