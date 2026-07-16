# D4 — O(log N) column norms via translation-invariance representatives

**Verdict: PASS**, with a concrete recipe for M14. Column norms are constant within
each **structural block** (coarse; and per level, per subband), so `1 + nl·n_subband`
representatives reproduce all N norms exactly.

## Results

`d4_colnorms.py` (distinct-norm compactness):

| dim | N | #distinct norms | level-only recon err |
|---|---|---|---|
| 1 | 64 | 6 | 1.5e-1 ✗ |
| 2 | 256 | 4 | 1.6e-1 ✗ |
| 3 | 512 | 3 | 1.2e-1 ✗ |

`d4_blocks.py` (block-keyed reconstruction):

| dim | N | #blocks | recon err |
|---|---|---|---|
| 1 | 64 | 6 | **0.0** |
| 2 | 256 | 10 | **0.0** |
| 3 | 512 | 15 | **0.0** |
| 2 | 1024 | 13 | **0.0** |

## The subtlety M14 must get right

Compactness is not in doubt (3–6 distinct values for N up to 512). But **`levels_*()`
is the wrong key.** `levels_1d` labels the coarse block *and* the first detail level
both `0` (`transform.py:273-276`), and it carries no subband distinction at all. So a
level-only representative fails by O(0.15) — it conflates the coarse scaling functions
with the first detail band.

The correct key is the **structural block id** matching the synthesis coefficient
layout: block 0 = coarse (`nc**dim` DOFs), then for each level, `n_subband` blocks
(1 in 1D, 3 in 2D, 7 in 3D) of size `cur**dim`. Total `1 + nl·n_subband` blocks —
O(log N). Keyed this way, one representative per block reproduces the dense norms to
**machine zero** in 1D/2D/3D.

`#distinct ≤ #blocks` because some blocks share a norm by symmetry (e.g. LH/HL), but
keying on block is safe and exact; keying on the observed distinct values would be
fragile.

## Consequence for M14

Implement `column_norms` matrix-free by evaluating `‖W e_j‖` for **one representative
`j` per structural block** and scattering by block id — not by materialising `W`, and
not by reusing `levels_*()`. The block-id constructor is in `d4_blocks.py::block_ids`.
This does not block M13/M15; it is a self-contained replacement for `operator.py:217`.
