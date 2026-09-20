"""Re-derive MAX_FOURIER_NUMBER exactly as its docstring instructs.

'To re-derive either figure: form the operator matrix column by column
 (one _compute_laplacian call per unit vector, with T_b = 0) and bisect
 on max|1 + Fo*lambda| <= 1 over its eigenvalues.'
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.heat import HeatNode, MAX_FOURIER_NUMBER

L = 1.0


def operator(n, order):
    dt = 0.1 * (L / n) ** 2
    node = HeatNode("h", timestep=dt, n_cells=n, length=L,
                    thermal_diffusivity=1.0, stencil_order=order)
    dx = L / n
    cols = []
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        lap = np.asarray(node._compute_laplacian(jnp.asarray(e), 0.0, 0.0, L),
                         dtype=np.float64)
        cols.append(lap)
    A = np.stack(cols, axis=1) * dx * dx   # dimensionless: Fo * A
    return A


def sharp_fourier(n, order):
    A = operator(n, order)
    ev = np.linalg.eigvals(A)
    def stable(fo):
        return np.max(np.abs(1.0 + fo * ev)) <= 1.0 + 1e-13
    lo, hi = 0.0, 1.0
    if stable(hi):
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo


print(f"MAX_FOURIER_NUMBER = {MAX_FOURIER_NUMBER}")
for order in (2, 4):
    print(f"\n--- stencil_order={order}; declared limit {MAX_FOURIER_NUMBER[order]} ---")
    prev = None
    mono = True
    for n in (5, 6, 8, 10, 16, 20, 40, 80, 160, 320):
        if order == 4 and n < 5:
            continue
        fo = sharp_fourier(n, order)
        flag = ""
        if prev is not None and fo < prev - 1e-12:
            flag = "  <-- NOT monotone increasing"
            mono = False
        prev = fo
        print(f"  n={n:4d}  sharp Fo = {fo:.6f}{flag}")
    print(f"  monotone increasing in n: {mono}")
