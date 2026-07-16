"""D4 follow-up — the correct representative key is the structural block.

d4_colnorms.py showed level-only reconstruction fails because levels_*() labels
the coarse block and the first detail level both as 0.  The right key is the
transform's block layout: coarse block, then per level a set of subband blocks
(1 in 1D, 3 in 2D, 7 in 3D), each of a known size.  Build that block-id vector
and verify a representative-per-block reconstruction is exact.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from maddening.nodes.adaptive.wavelets import operator as OP


def block_ids(dim, nl, nc):
    """Structural block id per DOF, matching the synthesis coeff layout."""
    n_sub = {1: 1, 2: 3, 3: 7}[dim]
    ids = []
    bid = 0
    ids += [bid] * (nc ** dim)          # coarse block
    cur = nc
    for _ in range(nl):
        block = cur ** dim              # size of one subband block at this level
        for _s in range(n_sub):
            bid += 1
            ids += [bid] * block
        cur *= 2
    return np.asarray(ids)


def run(dim, nl, nc, order=4):
    side = nc * 2 ** nl
    h = 1.0 / side
    norms = np.asarray(OP.column_norms(nl, nc, order, dim, h))
    bids = block_ids(dim, nl, nc)
    assert len(bids) == len(norms), (len(bids), len(norms))
    recon = np.empty_like(norms)
    for b in np.unique(bids):
        idx = np.where(bids == b)[0]
        recon[idx] = norms[idx[0]]       # one representative per block
    return side, len(norms), len(np.unique(bids)), np.max(np.abs(recon - norms))


if __name__ == "__main__":
    cases = [(1, 5, 2), (2, 3, 2), (3, 2, 2), (2, 4, 2)]
    print(f"{'dim':>3} {'side':>5} {'N':>6} {'#blocks':>8} {'recon err':>11}")
    all_pass = True
    for dim, nl, nc in cases:
        side, N, nb, err = run(dim, nl, nc)
        ok = err < 1e-12
        all_pass &= ok
        print(f"{dim:>3} {side:>5} {N:>6} {nb:>8} {err:>11.3e}  {'PASS' if ok else 'FAIL'}")
    print()
    print("Verdict:", "PASS — key representatives on structural block id (1 + nl*n_subband blocks)"
          if all_pass else "FAIL")
