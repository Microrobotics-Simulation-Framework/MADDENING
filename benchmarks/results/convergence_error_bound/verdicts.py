#!/usr/bin/env python
"""Verdict-level diff of two replay_sweep runs, for the error-bound change."""
import base64, json, sys
from pathlib import Path
import numpy as np

def load(p):
    return {json.loads(l)["row"]: json.loads(l)
            for l in Path(p).read_text().splitlines() if l.strip()}

b, a = load(sys.argv[1]), load(sys.argv[2])
rows = sorted(set(b) & set(a))
n_ok = 0
step_verdicts_b = step_verdicts_a = 0
gained = lost = 0
rows_verdict_changed = []
rows_newly_never_converge = []
rows_lost_some = []
iters_b, iters_a = [], []
cap_b = cap_a = 0
rel_step1, rel_final = [], []
moved = identical = 0
for r in rows:
    rb, ra = b[r], a[r]
    if not (rb.get("ok") and ra.get("ok")):
        continue
    n_ok += 1
    row_g = row_l = 0
    caps = rb.get("caps") or {}
    for k in rb["group_keys"]:
        cb = np.asarray(rb["groups"][k]["converged"], bool)
        ca = np.asarray(ra["groups"][k]["converged"], bool)
        step_verdicts_b += int(cb.sum()); step_verdicts_a += int(ca.sum())
        row_g += int((~cb & ca).sum()); row_l += int((cb & ~ca).sum())
        ib = np.asarray(rb["groups"][k]["iterations"], float)
        ia = np.asarray(ra["groups"][k]["iterations"], float)
        iters_b.append(ib.mean()); iters_a.append(ia.mean())
        cap = caps.get(k)
        if cap:
            cap_b += int((ib >= cap - 1).sum()); cap_a += int((ia >= cap - 1).sum())
    gained += row_g; lost += row_l
    if row_g or row_l:
        rows_verdict_changed.append((rb["fixture"], rb["label"], row_g, row_l))
    any_b = any(any(rb["groups"][k]["converged"]) for k in rb["group_keys"])
    any_a = any(any(ra["groups"][k]["converged"]) for k in ra["group_keys"])
    if any_b and not any_a:
        rows_newly_never_converge.append((rb["fixture"], rb["label"]))
    elif row_l:
        rows_lost_some.append((rb["fixture"], rb["label"], row_l))
    sb = np.frombuffer(base64.b64decode(rb["step1_state"]), np.float32).astype(np.float64)
    sa = np.frombuffer(base64.b64decode(ra["step1_state"]), np.float32).astype(np.float64)
    fb = np.frombuffer(base64.b64decode(rb["final_state"]), np.float32).astype(np.float64)
    fa = np.frombuffer(base64.b64decode(ra["final_state"]), np.float32).astype(np.float64)
    for src, dst in ((sb, rel_step1), (fb, rel_final)):
        pass
    den1 = np.sqrt((sb*sb).sum()); denf = np.sqrt((fb*fb).sum())
    if den1 > 0 and np.isfinite(sa).all():
        rel_step1.append(float(np.sqrt(((sa-sb)**2).sum())/den1))
    if denf > 0 and np.isfinite(fa).all():
        rel_final.append(float(np.sqrt(((fa-fb)**2).sum())/denf))
    if rb["step_hashes"] == ra["step_hashes"]:
        identical += 1
    else:
        moved += 1

tot_steps = step_verdicts_b
print(f"rows compared (both ok)        {n_ok}")
print(f"converged step-verdicts before {step_verdicts_b}")
print(f"converged step-verdicts after  {step_verdicts_a}")
print(f"  gained (False -> True)       {gained}")
print(f"  lost   (True  -> False)      {lost}")
print(f"rows whose verdicts changed    {len(rows_verdict_changed)}")
print(f"rows that converged before and never converge now  {len(rows_newly_never_converge)}")
for f, l in rows_newly_never_converge[:25]:
    print(f"    {f:<18s} {l}")
print(f"rows that lost some steps but still converge       {len(rows_lost_some)}")
for f, l, n in sorted(rows_lost_some, key=lambda t: -t[2])[:15]:
    print(f"    {f:<18s} {l:<28s} -{n} steps")
ib, ia = np.asarray(iters_b), np.asarray(iters_a)
print(f"mean iterations/step  before {ib.mean():.3f}  after {ia.mean():.3f} "
      f"({100*(ia.mean()/ib.mean()-1):+.1f}%)")
print(f"groups whose mean iterations rose {int((ia>ib+1e-9).sum())}, "
      f"fell {int((ia<ib-1e-9).sum())}, unchanged {int((abs(ia-ib)<=1e-9).sum())}")
print(f"at-cap step count  before {cap_b}  after {cap_a}")
print(f"rows bit-identical across all steps {identical}, rows whose state moved {moved}")
for name, v in (("rel step1", rel_step1), ("rel final", rel_final)):
    v = np.asarray([x for x in v if np.isfinite(x)])
    print(f"{name}: median {np.median(v):.3e}  p90 {np.percentile(v,90):.3e}  max {v.max():.3e}")
