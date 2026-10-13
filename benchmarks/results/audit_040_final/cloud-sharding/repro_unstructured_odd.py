"""Unstructured sharding at device counts / cell counts the suite never tries.

tests/cloud/multigpu/property_support.py pins DEVICE_COUNTS = (1, 2, 4):
3 devices, and cell counts that leave ragged shards, are never exercised.
"""
import os, sys
import numpy as np
import jax
sys.path.insert(0, os.path.join(os.environ["WT"], "tests", "cloud", "multigpu"))
from property_support import build_unstructured, build_stencil   # noqa: E402

print(f"devices: {len(jax.devices())}\n")
print("== ShardedUnstructuredNode: sharded vs unsharded, 3 steps ==")
for nd in (1, 2, 3, 4):
    for n_cells in (12, 16, 17):
        tag = f"dev={nd} cells={n_cells} divides={n_cells % nd == 0}"
        try:
            case = build_unstructured(n_devices=nd, n_cells=n_cells)
            a = case.run(steps=3, sharded=False)
            b = case.run(steps=3, sharded=True)
            errs = {k: float(np.max(np.abs(a[k] - b[k][:len(a[k])])))
                    for k in a}
            m = max(errs.values())
            print(f"  {tag:34s} max err = {m:.3e} "
                  f"{'OK' if m < 1e-5 else '*** MISMATCH ***'}")
        except Exception as exc:
            print(f"  {tag:34s} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]}")

print("\n== ShardedStencilNode via the suite's own builder, 3 steps ==")
for nd in (1, 2, 3, 4):
    for n_cells in (12, 16, 18):
        tag = f"dev={nd} cells={n_cells} divides={n_cells % nd == 0}"
        try:
            case = build_stencil(n_devices=nd, n_cells=n_cells)
            a = case.run(steps=3, sharded=False)
            b = case.run(steps=3, sharded=True)
            m = max(float(np.max(np.abs(a[k] - b[k]))) for k in a)
            print(f"  {tag:34s} max err = {m:.3e} "
                  f"{'OK' if m < 1e-5 else '*** MISMATCH ***'}")
        except Exception as exc:
            print(f"  {tag:34s} RAISED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]}")
