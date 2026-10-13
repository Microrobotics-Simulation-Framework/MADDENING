import math, json, sys
import numpy as np
from maddening.serialization.json_codec import (
    encode_non_finite, decode_non_finite, dumps, loads)

fails = []
def rt(label, obj, *, via_json=True):
    """encode->(json)->decode must equal obj structurally."""
    try:
        enc = encode_non_finite(obj)
    except Exception as e:
        print(f"[ENC-RAISE] {label}: {type(e).__name__}: {e}")
        return
    try:
        if via_json:
            back = loads(dumps(obj))
        else:
            back = decode_non_finite(enc)
    except Exception as e:
        print(f"[RT-RAISE ] {label}: {type(e).__name__}: {e}")
        fails.append((label, 'raise', str(e)))
        return
    try:
        ok = bool(same(obj, back))
    except Exception as e:
        ok = False
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}: {obj!r} -> {enc!r} -> {back!r}")
    if not ok:
        fails.append((label, obj, back))

def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b): return True
        return a == b and math.copysign(1,a)==math.copysign(1,b)
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys())==list(b.keys()) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    import numpy as _np
    if isinstance(a,_np.ndarray) or isinstance(b,_np.ndarray):
        return type(a)==type(b) and bool(_np.array_equal(a,b,equal_nan=True))
    return type(a)==type(b) and a==b

print("== in-memory encode/decode (no json) ==")
for label, obj in [
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("-inf", float("-inf")),
    ("neg zero", -0.0),
    ("subnormal", 5e-324),
    ("nested", {"a":[{"b":(float("inf"), -0.0)}]}),
    ("np.float64 nan", np.float64("nan")),
    ("np.float32 nan", np.float32("nan")),
    ("0-d array nan", np.array(np.nan)),
    ("np array", np.array([np.nan, 1.0])),
    ("non-ascii key", {"ключ": float("nan"), "日本": -0.0}),
    ("nan as dict key", {float("nan"): 1}),
    ("inf as dict key", {float("inf"): 1}),
    ("float key finite", {1.5: "x"}),
    ("bool", {"t": True, "f": False}),
    ("long float", 0.1234567890123456789),
    ("tuple", (1.0, float("nan"))),
    ("string token", {"name": "NaN"}),
    ("string token key", {"NaN": 1.0}),
    ("lowercase nan str", {"s": "nan"}),
]:
    rt(label, obj, via_json=False)

print()
print("== through json (dumps/loads) ==")
for label, obj in [
    ("nan", {"v": float("nan")}),
    ("neg zero", {"v": -0.0}),
    ("subnormal", {"v": 5e-324}),
    ("nested", {"a":[{"b":[float("inf"), -0.0]}]}),
    ("np.float64 nan", {"v": np.float64("nan")}),
    ("np.float32 nan", {"v": np.float32("nan")}),
    ("0-d array nan", {"v": np.array(np.nan)}),
    ("nan as dict key", {float("nan"): 1}),
    ("inf as dict key", {float("inf"): 1}),
    ("float key finite", {1.5: "x"}),
    ("non-ascii key", {"ключ": float("nan")}),
    ("tuple", {"v": (1.0, float("nan"))}),
    ("lowercase nan str", {"s": "nan"}),
    ("INF str", {"s": "INF"}),
    ("nan(chars) str", {"s": "nan(0x1)"}),
]:
    rt(label, obj, via_json=True)

print()
print("FAILS:", len(fails))
for f in fails: print("  ", f)
