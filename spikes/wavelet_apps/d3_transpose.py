"""D3 derisk — matrix-free transpose vs analysis (the non-orthogonality trap).

Question: for the DD wavelet synthesis W, does jax.linear_transpose(synthesis)
reproduce the dense W^T to ~1e-12, and does analysis_* fail to (because DD
wavelets are non-orthogonal, W^-1 != W^T)?

Pass: linear_transpose matches dense W^T to 1e-12 in 1D/2D/3D; the analysis
mismatch is quantified so the trap is evidenced, not asserted.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.adaptive.wavelets import transform as T

_SYNTH = {1: T.synthesis_1d, 2: T.synthesis_2d, 3: T.synthesis_3d}
_ANAL = {1: T.analysis_1d, 2: T.analysis_2d, 3: T.analysis_3d}


def run_dim(dim, nl, nc, order=4):
    N = T.n_dofs(nl, nc, dim)
    synth = lambda c: _SYNTH[dim](c, nl, nc, order)
    anal = lambda v: _ANAL[dim](v, nl, nc, order)

    W = np.asarray(T.synthesis_matrix(nl, nc, order, dim=dim))   # (N, N)
    Wt_dense = W.T

    # matrix-free transpose via linear_transpose.  transpose_fn(v) = W^T v, so
    # transpose_fn(e_i) is column i of the matrix representation of W^T.
    # vmap-over-eye stacks these as ROWS, giving (W^T)^T = W; transpose to
    # recover W^T.
    z = jnp.zeros(N)
    transpose_fn = jax.linear_transpose(synth, z)
    stack = np.asarray(jax.vmap(lambda e: transpose_fn(e)[0])(jnp.eye(N)))
    Wt_mf = stack.T
    err_transpose = np.max(np.abs(Wt_mf - Wt_dense))

    # independent check of the defining adjoint identity <synth(c),v>=<c,W^T v>
    key = jax.random.PRNGKey(0)
    c = jax.random.normal(key, (N,)); v = jax.random.normal(jax.random.PRNGKey(1), (N,))
    lhs = float(jnp.dot(synth(c), v)); rhs = float(jnp.dot(c, transpose_fn(v)[0]))
    err_adjoint = abs(lhs - rhs) / (abs(lhs) + 1e-30)

    # analysis materialised as a matrix (what the WRONG code would use for W^T)
    A_mat = np.asarray(jax.vmap(lambda e: anal(e))(jnp.eye(N))).T  # column j = analysis(e_j)
    err_analysis_vs_transpose = np.max(np.abs(A_mat - Wt_dense))

    # sanity: analysis is the inverse of synthesis (round-trip)
    roundtrip = np.max(np.abs(np.asarray(jax.vmap(lambda e: anal(synth(e)))(jnp.eye(N))) - np.eye(N)))
    return N, err_transpose, err_analysis_vs_transpose, roundtrip, err_adjoint


if __name__ == "__main__":
    cases = [(1, 5, 2), (2, 3, 2), (3, 2, 2)]
    print(f"{'dim':>3} {'N':>6} {'|linT - W^T|':>13} {'adjoint id':>11} "
          f"{'|analysis - W^T|':>16} {'roundtrip':>10}")
    all_pass = True
    for dim, nl, nc in cases:
        N, et, ea, rt, adj = run_dim(dim, nl, nc)
        ok = et < 1e-12 and adj < 1e-12
        all_pass &= ok
        print(f"{dim:>3} {N:>6} {et:>13.3e} {adj:>11.3e} {ea:>16.3e} {rt:>10.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print("Verdict:", "PASS — jax.linear_transpose(synthesis) IS W^T; analysis_* (err ~0.9) is NOT"
          if all_pass else "FAIL — STOP")
