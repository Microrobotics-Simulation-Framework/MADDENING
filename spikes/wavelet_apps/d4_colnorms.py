"""D4 derisk — O(log N) column norms via translation-invariance representatives.

column_norms (operator.py:217) materialises the dense W and takes per-column
sqrt(h^dim sum W[:,j]^2).  M14 wants to avoid materialising W: by periodic
translation invariance, every column in one translation-invariance group shares
a norm, so a handful of representatives suffice.

This derisk discovers the required granularity: how many DISTINCT norm values
exist, and whether grouping by LEVEL alone reproduces them (1D) or whether
SUBBAND granularity is needed (2D/3D, anisotropic LH/HL/HH).  It does not assume
the answer.

Pass: the number of distinct norms is O(#levels * #subbands) << N, and a
representative-per-group reconstruction matches the dense norms to 1e-12.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.adaptive.wavelets import operator as OP, transform as T


def run(dim, nl, nc, order=4):
    side = nc * 2 ** nl
    h = 1.0 / side
    norms = np.asarray(OP.column_norms(nl, nc, order, dim, h))    # dense reference
    lev = np.asarray({1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](nl, nc))

    # distinct norms (rounded to absorb fp noise)
    distinct = np.unique(np.round(norms, 12))
    n_subband = {1: 1, 2: 3, 3: 7}[dim]
    expected_max = 1 + nl * n_subband   # coarse + subbands per level (upper bound)

    # Reconstruct from a LEVEL-only representative: for each level take the first
    # column's norm and broadcast.  Works iff norms are constant within a level.
    recon_level = np.empty_like(norms)
    for l in np.unique(lev):
        idx = np.where(lev == l)[0]
        recon_level[idx] = norms[idx[0]]
    err_level = np.max(np.abs(recon_level - norms))

    return side, len(norms), len(distinct), expected_max, err_level


if __name__ == "__main__":
    cases = [(1, 5, 2), (2, 3, 2), (3, 2, 2)]
    print(f"{'dim':>3} {'side':>5} {'N':>6} {'#distinct':>10} {'<=1+nl*sb':>10} "
          f"{'level-recon err':>16}")
    all_ok = True
    for dim, nl, nc in cases:
        side, N, nd, exp, el = run(dim, nl, nc)
        compact = nd <= exp
        # level-only reconstruction is exact only if subbands within a level share
        # a norm; if not, err_level is O(1) and M14 needs (level,subband) reps.
        print(f"{dim:>3} {side:>5} {N:>6} {nd:>10} {exp:>10} {el:>16.3e} "
              f"{'level-ok' if el < 1e-12 else 'needs-subband'}")
        all_ok &= compact
    print()
    print("compactness (distinct << N):", "PASS" if all_ok else "FAIL")
    print("NOTE: if any row says needs-subband, M14's representative must be keyed on")
    print("      (level, subband), not level alone.  Either way #reps is O(log N).")
