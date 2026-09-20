#!/usr/bin/env python3
"""G1: check_citations.py reports N citations "verified" when it verified N-5.

The five (path, key) pairs in _TEMPLATE_CITATIONS are `continue`d before the
existence check, but they are still counted in `len(citations)` on the OK line.

    PYTHONPATH=<WT>/src python g1_citations_count_conflation.py <WT>
"""
import importlib.util, os, sys

wt = sys.argv[1] if len(sys.argv) > 1 else "/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates"
spec = importlib.util.spec_from_file_location("cc", os.path.join(wt, "scripts/check_citations.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

cits = m.scan_directory(os.path.join(wt, "docs"))
tmpl = [(f, l, k) for f, l, k in cits
        if (os.path.relpath(f, wt), k) in m._TEMPLATE_CITATIONS]
print(f"gate prints          : OK: {len(cits)} citation(s) verified")
print(f"actually verified    : {len(cits) - len(tmpl)}")
print(f"skipped, but counted : {len(tmpl)}")
for t in tmpl:
    print("   ", os.path.relpath(t[0], wt), t[1], t[2])
assert len(tmpl) > 0, "no template citations -- nothing to demonstrate"
print("\nFAIL: the headline count includes references the gate declined to check.")
