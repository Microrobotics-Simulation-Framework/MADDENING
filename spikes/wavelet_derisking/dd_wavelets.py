"""Deslauriers-Dubuc (DD) interpolating wavelets -- spike construction.

SPIKE CODE. Investigative, not production. Built for the wavelet
derisking spike (plans/WAVELET_ADAPTIVE_NODE_DERISKING_SPIKE.md).
Numpy-based: the Gate 1 questions are linear-algebra (condition number,
sign patterns), so we use numpy/np.linalg rather than JAX here. JAX is
reserved for the trajectory-adjoint investigation (§6).

What this provides
------------------
* DD interpolating-subdivision prediction filters for orders
  2N in {2, 4, 6} (linear, cubic, quintic midpoint interpolation).
* A *synthesis matrix* ``W`` (N x N) whose columns are the wavelet /
  scaling basis functions sampled on the fine grid.  Built by running
  the inverse interpolating-wavelet transform on unit coefficient
  vectors -- so column j of W *is* basis function j on the grid.
* A per-column ``level`` label so the Dahmen-Kunoth diagonal
  ``D_lambda = 2^{level}`` can be assembled.
* A Haar synthesis matrix in the same framework, used to *calibrate*
  the harness against the round-7 spike baselines before trusting any
  DD number.

Two boundary modes:
* ``periodic`` -- circular prediction stencils.  Cleanest test of the
  interior Dahmen-Kunoth O(1) claim (used for the condition-number
  sweep, §2 Hyp A).
* ``dirichlet`` -- reduced-order one-sided stencils near the ends, for
  the wrong-sign / locality boundary test (§3), which is specifically
  about theta near the domain boundary.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# DD interpolating-subdivision prediction filters.
#
# A midpoint between coarse samples is predicted by polynomial
# interpolation of the 2N nearest coarse samples.  The weights below
# are the classic DD coefficients (symmetric, sum to 1).
# ----------------------------------------------------------------------

# order -> (offsets relative to left coarse neighbour, weights)
# midpoint sits between sample 0 and sample 1.
_DD_FILTERS = {
    2: (np.array([0, 1]), np.array([1 / 2, 1 / 2])),
    4: (
        np.array([-1, 0, 1, 2]),
        np.array([-1 / 16, 9 / 16, 9 / 16, -1 / 16]),
    ),
    6: (
        np.array([-2, -1, 0, 1, 2, 3]),
        np.array([3 / 256, -25 / 256, 150 / 256, 150 / 256, -25 / 256, 3 / 256]),
    ),
}


def dd_filter(order: int):
    """Return (offsets, weights) for the DD midpoint predictor."""
    if order not in _DD_FILTERS:
        raise ValueError(f"unsupported DD order {order}; have {list(_DD_FILTERS)}")
    return _DD_FILTERS[order]


# ----------------------------------------------------------------------
# One refinement step: given values on a coarse grid of `nc` points
# (periodic), produce values on the 2*nc fine grid.  Even fine indices
# copy the coarse values; odd (midpoint) indices are predicted.
# ----------------------------------------------------------------------

def _refine_periodic(coarse: np.ndarray, order: int) -> np.ndarray:
    nc = coarse.shape[0]
    offsets, weights = dd_filter(order)
    fine = np.zeros(2 * nc, dtype=float)
    fine[0::2] = coarse
    # midpoint k sits between coarse k and coarse k+1
    mids = np.zeros(nc, dtype=float)
    for off, w in zip(offsets, weights):
        mids += w * coarse[(np.arange(nc) + off) % nc]
    fine[1::2] = mids
    return fine


def _refine_dirichlet(coarse: np.ndarray, order: int) -> np.ndarray:
    """Refine with one-sided fallback near the ends.

    Coarse samples are assumed to be interior nodal values on (0,1)
    with implied zero Dirichlet data just outside.  When the 2N-point
    stencil would reach outside the available coarse samples we drop
    to the widest *centred* stencil that fits, falling back ultimately
    to linear.  This keeps the basis local and avoids fabricating data
    past the boundary.
    """
    nc = coarse.shape[0]
    fine = np.zeros(2 * nc + 1, dtype=float)
    # fine grid has 2*nc+1 points; even -> coarse (shifted by one so the
    # boundaries at index 0 and 2*nc are the zero Dirichlet nodes).
    # We model coarse[i] as living at fine index 2*i+1.
    fine[1::2] = coarse
    # midpoints: between consecutive coarse samples, and the two ends
    # (between boundary-zero and first/last coarse sample).
    padded = np.concatenate(([0.0], coarse, [0.0]))  # zeros at boundary
    for j in range(nc + 1):
        # midpoint between padded[j] and padded[j+1]
        # try centred stencil of order `order`, shrink if it doesn't fit
        o = order
        while o > 2:
            half = o // 2
            lo = j - (half - 1)
            hi = j + half + 1
            if lo >= 0 and hi <= len(padded):
                break
            o -= 2
        offsets, weights = dd_filter(o)
        half = o // 2
        val = 0.0
        for off, w in zip(offsets, weights):
            idx = j + off  # relative to padded, left neighbour = j
            if 0 <= idx < len(padded):
                val += w * padded[idx]
        fine[2 * j] = val
    return fine


# ----------------------------------------------------------------------
# Build the synthesis matrix W and level labels.
# ----------------------------------------------------------------------

def synthesis_matrix(
    n_levels: int,
    n_coarse: int = 1,
    order: int = 4,
    boundary: str = "periodic",
):
    """Construct the DD inverse-transform (synthesis) matrix.

    Parameters
    ----------
    n_levels : number of *detail* levels added on top of the coarse grid.
    n_coarse : number of coarse-scale scaling coefficients.
    order    : DD order (2, 4, or 6).
    boundary : 'periodic' or 'dirichlet'.

    Returns
    -------
    W      : (N, N) array; column j is basis function j on the fine grid.
    levels : (N,) int array; the level |lambda| of each column
             (0 = coarse scaling and coarsest detail, increasing with
             refinement).
    x      : (N,) fine-grid coordinates in (0,1).
    """
    if boundary == "periodic":
        N = n_coarse * (2 ** n_levels)
    elif boundary == "dirichlet":
        # interior points only; boundary nodes (value 0) are not DOF.
        N = n_coarse
        for _ in range(n_levels):
            N = 2 * N + 1
    else:
        raise ValueError(boundary)

    # Coefficient layout: [coarse(n_coarse), detail_0(n_coarse),
    # detail_1(2 n_coarse), ...]. Track the level of each coeff.
    coeff_levels = [0] * n_coarse  # coarse scaling -> level 0
    size = n_coarse
    for lvl in range(n_levels):
        n_detail = size  # one detail per coarse interval (periodic) ...
        coeff_levels.extend([lvl] * n_detail)
        size = size + n_detail if boundary == "dirichlet" else 2 * size
    # For periodic, detail count at each level equals current coarse count.
    # Recompute layout cleanly:
    coeff_levels = []
    if boundary == "periodic":
        coeff_levels.extend([0] * n_coarse)
        cur = n_coarse
        for lvl in range(n_levels):
            coeff_levels.extend([lvl] * cur)  # detail count == cur
            cur *= 2
    else:
        coeff_levels.extend([0] * n_coarse)
        cur = n_coarse
        for lvl in range(n_levels):
            n_detail = cur + 1
            coeff_levels.extend([lvl] * n_detail)
            cur = 2 * cur + 1
    coeff_levels = np.array(coeff_levels, dtype=int)
    assert coeff_levels.shape[0] == N, (coeff_levels.shape[0], N)

    # Synthesis of a single coefficient vector.
    def synth(coeffs: np.ndarray) -> np.ndarray:
        idx = 0
        vals = coeffs[idx : idx + n_coarse].astype(float).copy()
        idx += n_coarse
        cur = n_coarse
        for lvl in range(n_levels):
            if boundary == "periodic":
                refined = _refine_periodic(vals, order)
                n_detail = cur
                detail = coeffs[idx : idx + n_detail]
                idx += n_detail
                refined[1::2] += detail  # add detail at midpoints
                vals = refined
                cur *= 2
            else:
                refined = _refine_dirichlet(vals, order)
                n_detail = cur + 1
                detail = coeffs[idx : idx + n_detail]
                idx += n_detail
                refined[0::2] += detail  # midpoints are at even indices
                vals = refined
                cur = 2 * cur + 1
        return vals

    W = np.zeros((N, N), dtype=float)
    e = np.zeros(N, dtype=float)
    for j in range(N):
        e[:] = 0.0
        e[j] = 1.0
        W[:, j] = synth(e)

    if boundary == "periodic":
        x = np.arange(N) / N
    else:
        x = (np.arange(1, N + 1)) / (N + 1)
    return W, coeff_levels, x


# ----------------------------------------------------------------------
# Haar synthesis matrix (calibration baseline) -- same framework.
# ----------------------------------------------------------------------

def haar_synthesis_matrix(n_levels: int, n_coarse: int = 1):
    """Periodic orthonormal Haar synthesis matrix and level labels.

    Used to calibrate the condition-number harness against the round-7
    Haar baselines before trusting DD numbers.
    """
    N = n_coarse * (2 ** n_levels)
    coeff_levels = [0] * n_coarse
    cur = n_coarse
    for lvl in range(n_levels):
        coeff_levels.extend([lvl] * cur)
        cur *= 2
    coeff_levels = np.array(coeff_levels, dtype=int)

    def synth(coeffs: np.ndarray) -> np.ndarray:
        idx = 0
        vals = coeffs[idx : idx + n_coarse].astype(float).copy()
        idx += n_coarse
        cur = n_coarse
        for lvl in range(n_levels):
            detail = coeffs[idx : idx + cur]
            idx += cur
            # orthonormal Haar reconstruction: each coarse value splits
            # into two, +/- detail, scaled by 1/sqrt(2)
            new = np.zeros(2 * cur, dtype=float)
            new[0::2] = (vals + detail) / np.sqrt(2.0)
            new[1::2] = (vals - detail) / np.sqrt(2.0)
            vals = new
            cur *= 2
        return vals

    W = np.zeros((N, N), dtype=float)
    e = np.zeros(N, dtype=float)
    for j in range(N):
        e[:] = 0.0
        e[j] = 1.0
        W[:, j] = synth(e)
    x = np.arange(N) / N
    return W, coeff_levels, x


# ----------------------------------------------------------------------
# Isotropic 2D interpolating-wavelet synthesis (periodic).
#
# The *correct* basis for an isotropic PDE operator: the Mallat pyramid
# with three detail subbands (LH, HL, HH) per level, ALL at the same
# resolution level j -- NOT the full anisotropic tensor product, which
# is the hyperbolic-cross basis and is known not to be H^1-stable with
# a simple diagonal.
# ----------------------------------------------------------------------

def _refine2d_periodic(coarse: np.ndarray, order: int) -> np.ndarray:
    """Predict a 2s x 2s fine image from an s x s coarse image.

    Even/even = copy; even/odd = predict along y; odd/even = predict
    along x; odd/odd = predict along both.  Returns the prediction
    (details added by the caller).
    """
    s = coarse.shape[0]
    fine = np.zeros((2 * s, 2 * s), dtype=float)
    fine[0::2, 0::2] = coarse
    # predict along axis 1 (y) for even rows (vectorised roll-based)
    pred_y = _midpoints_along(coarse, 1, order)
    fine[0::2, 1::2] = pred_y
    # predict along axis 0 (x) for even cols
    pred_x = _midpoints_along(coarse, 0, order)
    fine[1::2, 0::2] = pred_x
    # odd/odd: predict along x of the already-y-predicted columns
    pred_xy = _midpoints_along(pred_y, 0, order)
    fine[1::2, 1::2] = pred_xy
    return fine


def synthesis_matrix_2d_isotropic(n_levels: int, n_coarse: int = 2, order: int = 4):
    """Isotropic 2D periodic DD synthesis matrix and level labels.

    Returns W2 ((N^2, N^2)), levels (N^2,), where N = n_coarse*2^n_levels.
    Column j of W2 is the j-th 2D basis function flattened (row-major).
    Level label: 0 for the coarse LL block; j for all three detail
    subbands added at refinement step j.
    """
    N = n_coarse * (2 ** n_levels)

    # Build the coefficient layout and level labels.
    blocks = []  # list of (level, kind, size) describing flat segments
    blocks.append((0, "LL", n_coarse))
    cur = n_coarse
    for lvl in range(n_levels):
        for kind in ("LH", "HL", "HH"):
            blocks.append((lvl, kind, cur))
        cur *= 2

    levels = []
    for lvl, kind, sz in blocks:
        levels.extend([lvl] * (sz * sz))
    levels = np.array(levels, dtype=int)
    assert levels.shape[0] == N * N, (levels.shape[0], N * N)

    def synth2d(flat: np.ndarray) -> np.ndarray:
        idx = 0
        # coarse block
        sz = n_coarse
        img = flat[idx : idx + sz * sz].reshape(sz, sz).astype(float).copy()
        idx += sz * sz
        cur = n_coarse
        for lvl in range(n_levels):
            dLH = flat[idx : idx + cur * cur].reshape(cur, cur); idx += cur * cur
            dHL = flat[idx : idx + cur * cur].reshape(cur, cur); idx += cur * cur
            dHH = flat[idx : idx + cur * cur].reshape(cur, cur); idx += cur * cur
            fine = _refine2d_periodic(img, order)
            fine[0::2, 1::2] += dLH
            fine[1::2, 0::2] += dHL
            fine[1::2, 1::2] += dHH
            img = fine
            cur *= 2
        return img.reshape(-1)

    M = N * N
    W2 = np.zeros((M, M), dtype=float)
    e = np.zeros(M, dtype=float)
    for j in range(M):
        e[:] = 0.0
        e[j] = 1.0
        W2[:, j] = synth2d(e)
    return W2, levels, N


# ----------------------------------------------------------------------
# Isotropic 3D interpolating-wavelet synthesis (periodic).
# 7 detail subbands per level (all L/H parities except LLL), all level j.
# ----------------------------------------------------------------------

def _midpoints_along(arr: np.ndarray, axis: int, order: int) -> np.ndarray:
    """Vectorised periodic midpoint prediction along one axis (np.roll)."""
    offsets, weights = dd_filter(order)
    out = np.zeros_like(arr)
    for off, w in zip(offsets, weights):
        out += w * np.roll(arr, -off, axis=axis)
    return out


def _refine3d_periodic(coarse: np.ndarray, order: int):
    """Return dict parity->predicted block (each s x s x s) for the 8
    parities of a 2s cube. Caller adds details to the 7 non-(0,0,0)."""
    blocks = {}
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                cur = coarse
                if px:
                    cur = _midpoints_along(cur, 0, order)
                if py:
                    cur = _midpoints_along(cur, 1, order)
                if pz:
                    cur = _midpoints_along(cur, 2, order)
                blocks[(px, py, pz)] = cur
    return blocks


_PARITIES7 = [(0, 0, 1), (0, 1, 0), (0, 1, 1),
              (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def synthesis_matrix_3d_isotropic(n_levels: int, n_coarse: int = 1, order: int = 4):
    """Isotropic 3D periodic DD synthesis matrix and level labels.

    N_side = n_coarse * 2^n_levels; returns W3 ((N^3, N^3)), levels (N^3,),
    N_side. Memory: for N_side=16, W3 is 4096x4096 (~134 MB) -- OK.
    """
    Nside = n_coarse * (2 ** n_levels)
    M = Nside ** 3

    # level labels in Mallat layout: coarse cube then 7 detail cubes per level
    levels = []
    levels.extend([0] * (n_coarse ** 3))
    cur = n_coarse
    for lvl in range(n_levels):
        for _ in _PARITIES7:
            levels.extend([lvl] * (cur ** 3))
        cur *= 2
    levels = np.array(levels, dtype=int)
    assert levels.shape[0] == M, (levels.shape[0], M)

    def synth3d(flat: np.ndarray) -> np.ndarray:
        idx = 0
        sz = n_coarse
        img = flat[idx: idx + sz ** 3].reshape(sz, sz, sz).astype(float).copy()
        idx += sz ** 3
        cur = n_coarse
        for lvl in range(n_levels):
            blocks = _refine3d_periodic(img, order)
            fine = np.zeros((2 * cur, 2 * cur, 2 * cur))
            fine[0::2, 0::2, 0::2] = blocks[(0, 0, 0)]
            for (px, py, pz) in _PARITIES7:
                d = flat[idx: idx + cur ** 3].reshape(cur, cur, cur)
                idx += cur ** 3
                fine[px::2, py::2, pz::2] = blocks[(px, py, pz)] + d
            img = fine
            cur *= 2
        return img.reshape(-1)

    W3 = np.zeros((M, M), dtype=float)
    e = np.zeros(M, dtype=float)
    for j in range(M):
        e[:] = 0.0
        e[j] = 1.0
        W3[:, j] = synth3d(e)
    return W3, levels, Nside


# ----------------------------------------------------------------------
# Physical-space operators.
# ----------------------------------------------------------------------

def laplacian_periodic(N: int, mass: bool = True) -> np.ndarray:
    """H^1 bilinear-form matrix for (-d^2/dx^2 + I), periodic.

    Stiffness S = (1/h) tridiag(-1, 2, -1) circulant; lumped mass
    M = h I.  Returns S + M when ``mass`` else S.
    """
    h = 1.0 / N
    S = np.zeros((N, N))
    idx = np.arange(N)
    S[idx, idx] = 2.0 / h
    S[idx, (idx + 1) % N] += -1.0 / h
    S[idx, (idx - 1) % N] += -1.0 / h
    if mass:
        S[idx, idx] += h
    return S


def laplacian_dirichlet(N: int, mass: bool = True) -> np.ndarray:
    """(-d^2/dx^2 + I) with homogeneous Dirichlet BC on N interior nodes."""
    h = 1.0 / (N + 1)
    S = np.zeros((N, N))
    idx = np.arange(N)
    S[idx, idx] = 2.0 / h
    for i in range(N - 1):
        S[i, i + 1] = -1.0 / h
        S[i + 1, i] = -1.0 / h
    if mass:
        S[idx, idx] += h
    return S
