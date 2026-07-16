# D3 — Matrix-free transpose vs analysis

**Verdict: PASS.** `jax.linear_transpose(synthesis)` is the exact adjoint `Wᵀ`;
`analysis_*` is not. M13 may proceed.

## Result (`d3_transpose.py`, fp64)

| dim | N | \|linT − Wᵀ\| | adjoint id | \|analysis − Wᵀ\| | roundtrip |
|---|---|---|---|---|---|
| 1 | 64 | 0.0 | 7.2e-16 | 9.94e-1 | 0.0 |
| 2 | 256 | 0.0 | 4.4e-16 | 9.38e-1 | 0.0 |
| 3 | 512 | 0.0 | 5.9e-16 | 8.13e-1 | 0.0 |

## Reading

- `jax.linear_transpose(synthesis)` reproduces the dense `Wᵀ` to **machine zero**,
  and satisfies the adjoint identity `⟨W c, v⟩ = ⟨c, Wᵀ v⟩` to ~1e-16. This is the
  correct matrix-free transpose for the entire gradient path.
- `analysis_*` (the *inverse* `W⁻¹`) differs from `Wᵀ` by **~0.8–0.9** — order-unity,
  not a rounding effect. DD interpolating wavelets are non-orthogonal, so `W⁻¹ ≠ Wᵀ`.
  `analysis∘synthesis = I` to 0.0 confirms it is a correct inverse — which is exactly
  why substituting it for the transpose produces no error and no shape mismatch, just
  a wrong answer of order-unity magnitude.

## Consequence for M13

Use `jax.linear_transpose(synthesis)` for `Wᵀ`. Audit every transform call in the
backward pass (adjoint solve, RHS projection `Wᵀf`, sensor row, `dJ/da`): any
`analysis_*` where `Wᵀ` belongs makes `dJ/da` silently wrong. The trap is evidenced,
not asserted.

## Note (methodology)

First run of the script reported FAIL because the *check* was wrong, not the identity:
`vmap(transpose_fn)` over the identity stacks results as rows, giving `(Wᵀ)ᵀ = W`, so it
compared `W` to `Wᵀ`. Fixed by transposing the stack and adding the basis-independent
adjoint-identity check on random vectors. Lesson for the milestones: when a derisk fails,
rule out the harness before trusting the STOP.
