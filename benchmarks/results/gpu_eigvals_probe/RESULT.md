# GPU probe: the non-symmetric eigenvalue solve in coupling diagnostics

One-off local run, maintainer-approved, 2026-10-07. Tree: `release/0.4.0` at `1f66801c`.
Machine: NVIDIA RTX A2000 8 GB Laptop GPU. jax/jaxlib 0.11.0 with `jax-cuda12-plugin` and
`jax-cuda12-pjrt` 0.11.0 installed into a scratch `--target` directory (the shared venv has no
CUDA plugin and was not modified). Scripts: `probe.py`, `cost_split.py`; raw results
`probe_{cpu,gpu}{,_x64}.json`, logs `run_*.log`.

## Question

PR #248 put `jnp.linalg.eigvals` of a small non-symmetric matrix (at most 9 × 9) inside the
diagnostics branch of the coupled step. It had only ever run on CPU.

## Answers

1. **It runs on the GPU backend**, eagerly and under `jit`, inside `lax.scan`, `vmap` and
   `lax.cond`, in float32 and float64. Values equal the CPU's to rounding (worst relative
   difference 6e-7 in float32, 2e-15 in float64) and NumPy's float64 reference.
2. **A coupled pair with `diagnostics=True` compiles and runs on GPU** through `step` and
   `run_scan`. With diagnostics off, states equal the CPU's to 1e-7 (float32) and 4e-16 (float64).
3. **Reports agree where they are usable.** In float64 (flags True) every reported number equals
   the CPU's to rounding. In float32 this pair sits at its float floor: `spectral_usable` and
   `gradient_bound_usable` are False on both backends, and the not-usable `rho_spectral` differs
   between them (1.0e-5 vs 1.0e-5 after `step`; 2.3e-5 vs 3.9e-5 after `run_scan`), as noise does.
4. **One difference worth a line:** after 40 `step` calls in float32, with the residual exactly 0,
   `gradient_relative_error_bound` reads 4.3e-6 on CPU and **NaN on GPU**, with
   `gradient_bound_usable=False` on both. Inside the contract (the number is flagged not usable),
   but a NaN where the other backend has a number.
5. **Cost on GPU, this tiny pair, float32, per step:** diagnostics off 0.3 ms; diagnostics on
   6 to 8 ms with `eigvals`; 2 to 3.4 ms with the repeated-squaring fallback forced. So the
   eigenvalue solve (LAPACK on the host, a device round trip each call) is about 4 ms of a
   diagnostics-on step on GPU, and the rest of the diagnostics program about 2 to 3 ms. On CPU a
   diagnostics-on step of the same pair is about 0.25 to 0.5 ms.

## An environment trap, not a library defect

The first GPU pass failed in every `eigvals` call (and in `jnp.linalg.solve`) with
`Error loading CUDA libraries. GPU will not be used.` from jaxlib's GPU solver kernels, while
`eigh`, `svd`, `qr` and the whole solve path worked. Cause: the NVIDIA runtime libraries
installed as wheels in the venv were not on the loader path for those kernels. With
`LD_LIBRARY_PATH` pointing at `site-packages/nvidia/*/lib` everything runs. **On a pod, check
this before trusting a diagnostics run:** `python -c "import jax.numpy as jnp; print(jnp.linalg.eigvals(jnp.eye(3)))"`
on the GPU backend must not raise.

## What follows

- `eigvals` stays the GPU path (repeated squaring was the wrong number it replaced). The cost is
  opt-in (`diagnostics=True`) and belongs in the user guide's diagnostics section.
- The NaN in a not-usable `gradient_relative_error_bound` on GPU is a small item for the next
  coupling batch (a not-usable number should be the same kind of value on both backends).
- The one-line `eigvals` check joins the RunPod section-0 checklist.
