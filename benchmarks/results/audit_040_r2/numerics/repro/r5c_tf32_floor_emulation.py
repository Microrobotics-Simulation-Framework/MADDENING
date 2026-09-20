"""Does a reduced-precision matmul lift the J.T@J noise floor above fim's
rank cutoff max(n, sqrt(m)) * eps_float32?

Non-degenerate construction: J = U diag(s) V^T with s having an exact
zero, U/V random orthogonal, so the null direction is not aligned with a
coordinate pair and every entry of F is a distinct inner product.

Emulates the two reduced formats faithfully:
  TF32  -> operands rounded to 10 explicit mantissa bits, accumulate fp32
  BF16  -> operands rounded to  7 explicit mantissa bits, accumulate fp32
(that is what NVIDIA tensor cores do).  Reference = exact float64 Gram.
"""
import numpy as np

EPS32 = np.finfo(np.float32).eps


def round_mantissa(x, bits):
    """Round a float32 array to `bits` explicit mantissa bits."""
    x = np.asarray(x, dtype=np.float32)
    m, e = np.frexp(x.astype(np.float64))
    scale = 2.0 ** bits
    m = np.round(m * scale) / scale
    return np.ldexp(m, e).astype(np.float32)


def floor_ratio(n, m, rank_def, seed, mant_bits=None):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.ones(n)
    s[:rank_def] = 0.0            # exact null directions
    s[rank_def:] = np.geomspace(1.0, 0.3, n - rank_def)
    J64 = U @ np.diag(s) @ V.T
    J = J64.astype(np.float32)
    Jm = J if mant_bits is None else round_mantissa(J, mant_bits)
    F = (Jm.T.astype(np.float32) @ Jm).astype(np.float32)   # fp32 accumulate
    ev = np.linalg.eigvalsh(F.astype(np.float64))
    return abs(ev[0]) / ev[-1]


print(f"{'n':>3} {'m':>6} {'cutoff':>10} | {'fp32':>10} {'x':>7} | "
      f"{'TF32(10b)':>10} {'x':>7} | {'BF16(7b)':>10} {'x':>7}")
for n, m in ((2, 4000), (3, 4000), (5, 2000), (12, 800), (25, 800)):
    cutoff = max(n, np.sqrt(m)) * EPS32
    row = [f"{n:3d} {m:6d} {cutoff:10.3e} |"]
    for bits, _lab in ((None, "fp32"), (10, "tf32"), (7, "bf16")):
        rs = [floor_ratio(n, m, 1, seed, bits) for seed in range(8)]
        p = float(np.percentile(rs, 99))
        row.append(f" {p:10.3e} {p/cutoff:7.2f} |")
    print("".join(row).rstrip("|"))
print()
print("x = floor / cutoff.  >1 means the arithmetic's own noise exceeds the")
print("cutoff, so an exactly rank-deficient F is reported as full rank with a")
print("finite crb and no PrecisionLimitWarning (the pre-0.4.0 failure mode).")
