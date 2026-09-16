"""Interface mappings: field transfer between non-conforming interfaces.

A :class:`Mapping` turns a field sampled on the source interface into a
field on the target interface.  The edge applies it before the scalar
``transform``; its weights live in the graph parameter pytree under
``params["mappings"][edge_key]`` — a traced input, so a mapping can be
differentiated through (``jax.grad`` of a loss with respect to the
weights) or replaced without recompiling — never in a Python closure.

:class:`StaticLinearMapping` is the first implementation: a dense matrix
``H`` (``n_target × n_source``) built once from the two point sets.
Factories:

* :func:`rbf_mapping` — radial basis function interpolation with
  polynomial augmentation (constants and linear fields reproduced to
  round-off, which is what makes the transpose *conservative*),
* :func:`nearest_neighbor_mapping`,
* :func:`projection_1d_mapping` — cell-average projection between 1D
  grids (conservative by construction),
* :func:`matrix_mapping` — bring your own matrix.

Modes follow preCICE.  ``"consistent"`` transfers a *value* field
(temperature, displacement): ``H @ v`` interpolates.  ``"conservative"``
transfers an *integral* quantity (force, heat flow) such that the total
is preserved: it is the transpose of the consistent mapping in the
opposite direction, and it preserves sums exactly when that consistent
mapping reproduces constants — hence polynomial augmentation is on by
default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

_KERNELS = ("gaussian", "multiquadric", "inverse_multiquadric", "thin_plate_spline")
_MODES = ("consistent", "conservative")


@runtime_checkable
class Mapping(Protocol):
    """What an edge needs from a mapping.

    ``apply(field, weights, geom)`` maps a source field (shape
    ``(n_source,)`` or ``(n_source, C)``) to the target; ``apply_T`` is
    the transpose (adjoint) map.  ``weights`` is the entry of
    ``params["mappings"]`` for this edge (``None`` = the mapping's own
    :meth:`params_pytree`); ``geom`` is reserved for mappings that
    depend on a moving interface (resolved like a state field) and is
    ignored by static mappings.
    """

    kind: str
    mode: str

    @property
    def n_source(self) -> int: ...

    @property
    def n_target(self) -> int: ...

    def params_pytree(self) -> dict: ...

    def apply(self, field, weights: Optional[dict] = None, geom=None): ...

    def apply_T(self, field, weights: Optional[dict] = None, geom=None): ...


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True, eq=False)
class StaticLinearMapping:
    """``target = H @ source`` with a fixed dense matrix.

    ``H`` has shape ``(n_target, n_source)`` and is the mapping's one
    parameter (``params_pytree() == {"H": H}``); the graph passes it
    back as ``weights`` on every step.  Compared and hashed by identity
    (``eq=False``): an ``EdgeSpec`` holding a mapping must stay hashable,
    and two mappings with equal matrices are still two edges' weights.
    """
    H: Any
    kind: str = "matrix"
    mode: str = "consistent"
    meta: dict = None  # type: ignore[assignment]  # hyper-parameters, for describe()

    def __post_init__(self):
        H = jnp.asarray(self.H)
        if H.ndim != 2:
            raise ValueError(f"H must be a 2-D matrix, got shape {H.shape}")
        if self.mode not in _MODES:
            raise ValueError(f"mode={self.mode!r} not in {_MODES}")
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    @property
    def n_source(self) -> int:
        return int(self.H.shape[1])

    @property
    def n_target(self) -> int:
        return int(self.H.shape[0])

    def params_pytree(self) -> dict:
        return {"H": self.H}

    def _matrix(self, weights):
        return self.H if weights is None else weights["H"]

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        return self._matrix(weights) @ field

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        return self._matrix(weights).T @ field

    def describe(self) -> dict:
        """Kind, mode, shape and hyper-parameters (never the weights)."""
        return {
            "kind": self.kind, "mode": self.mode,
            "shape": [self.n_target, self.n_source], **self.meta,
        }

    def __repr__(self) -> str:
        return (f"StaticLinearMapping({self.kind}, {self.mode}, "
                f"{self.n_target}x{self.n_source})")


# ---------------------------------------------------------------------------
# RBF
# ---------------------------------------------------------------------------


def _as_points(x) -> np.ndarray:
    """``(n, d)`` float64 NumPy array (the matrix is assembled and solved
    in double precision on the host; only the result is cast)."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _pairwise_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.sqrt(np.maximum(d2, 0.0))


def _kernel(r: np.ndarray, eps: float, name: str) -> np.ndarray:
    if name == "gaussian":
        return np.exp(-(eps * r) ** 2)
    if name == "multiquadric":
        return np.sqrt(1.0 + (eps * r) ** 2)
    if name == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + (eps * r) ** 2)
    if name == "thin_plate_spline":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 1e-300, r ** 2 * np.log(np.where(r > 1e-300, r, 1.0)), 0.0)
    raise ValueError(f"Unknown kernel {name!r}; choose from {_KERNELS}")


def _out_dtype(*arrays):
    """float64 only when the caller passed float64 under ``jax_enable_x64``;
    float32 otherwise (the graph's working precision)."""
    x64 = bool(jax.config.read("jax_enable_x64"))
    if x64 and any(np.asarray(a).dtype == np.float64 for a in arrays):
        return jnp.float64
    return jnp.float32


@stability(StabilityLevel.EVOLVING)
def rbf_matrix(
    source_points,
    target_points,
    *,
    kernel: str = "gaussian",
    epsilon: float = 1.0,
    polynomial: bool = True,
    ridge: float = 1e-8,
) -> jnp.ndarray:
    """Consistent RBF interpolation matrix ``H`` (``n_target × n_source``).

    With ``polynomial=True`` the interpolant is augmented with the linear
    polynomial ``1, x_1, ..., x_D`` (the standard well-posed form for
    thin-plate splines, and the one that reproduces constant and linear
    fields exactly — the patch test).  The system is solved, not
    inverted, and the ridge is *relative* to the kernel matrix's scale
    (``ridge * max|Φ_ss|``), so it means the same thing for a Gaussian
    (entries ≤ 1) and a thin-plate spline on a metre-sized interface.

    ``H = [Φ_ts  P_t] · A⁻¹[:, :n_source]`` with
    ``A = [[Φ_ss + λI, P_s], [P_sᵀ, 0]]``.

    The matrix is assembled and solved in float64 on the host (it is
    built once, at graph construction; the kernel systems are
    ill-conditioned enough that a float32 solve loses 3–4 digits) and
    returned as a float32 JAX array — float64 when the points were
    float64 under ``jax_enable_x64``.  Point coordinates are therefore
    static; a mapping that must follow a moving interface is the
    matrix-free ``OnTheFlyMapping`` planned for 0.5.0.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; choose from {_KERNELS}")
    dtype = _out_dtype(source_points, target_points)
    src = _as_points(source_points)
    tgt = _as_points(target_points)
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"source and target points must share a dimension, got "
            f"{src.shape[1]} and {tgt.shape[1]}"
        )
    n, d = src.shape
    phi_ss = _kernel(_pairwise_r(src, src), epsilon, kernel)
    phi_ts = _kernel(_pairwise_r(tgt, src), epsilon, kernel)
    scale = max(float(np.max(np.abs(phi_ss))), 1e-300)
    phi_ss = phi_ss + (ridge * scale) * np.eye(n)

    if not polynomial:
        H = np.linalg.solve(phi_ss.T, phi_ts.T).T        # Φ_ts Φ_ss⁻¹
        return jnp.asarray(H, dtype=dtype)

    q = 1 + d
    p_s = np.concatenate([np.ones((n, 1)), src], axis=1)              # (n, q)
    p_t = np.concatenate([np.ones((tgt.shape[0], 1)), tgt], axis=1)
    a = np.block([[phi_ss, p_s], [p_s.T, np.zeros((q, q))]])
    b = np.concatenate([np.eye(n), np.zeros((q, n))])
    coeff = np.linalg.solve(a, b)                                     # (n + q, n)
    H = np.concatenate([phi_ts, p_t], axis=1) @ coeff
    return jnp.asarray(H, dtype=dtype)


@stability(StabilityLevel.EVOLVING)
def rbf_mapping(
    source_points,
    target_points,
    *,
    kernel: str = "gaussian",
    epsilon: float = 1.0,
    polynomial: bool = True,
    ridge: float = 1e-8,
    mode: str = "consistent",
) -> StaticLinearMapping:
    """RBF mapping from ``source_points`` to ``target_points``.

    ``mode="consistent"`` interpolates values.  ``mode="conservative"``
    builds the consistent interpolant from the *target* points to the
    *source* points and applies its transpose, so a quantity summed over
    the source (a total force) is summed identically over the target —
    exactly when the reverse interpolant reproduces constants, i.e. with
    ``polynomial=True``.
    """
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    kw = dict(kernel=kernel, epsilon=epsilon, polynomial=polynomial, ridge=ridge)
    if mode == "consistent":
        H = rbf_matrix(source_points, target_points, **kw)
    else:
        H = rbf_matrix(target_points, source_points, **kw).T
    meta = dict(kernel=kernel, epsilon=float(epsilon), polynomial=bool(polynomial),
                ridge=float(ridge))
    return StaticLinearMapping(H, kind="rbf", mode=mode, meta=meta)


# ---------------------------------------------------------------------------
# Nearest neighbour / 1D projection / explicit matrix
# ---------------------------------------------------------------------------


def _nn_matrix(source_points, target_points) -> jnp.ndarray:
    src = np.asarray(_as_points(source_points))
    tgt = np.asarray(_as_points(target_points))
    d2 = np.sum((tgt[:, None, :] - src[None, :, :]) ** 2, axis=-1)
    idx = np.argmin(d2, axis=1)
    H = np.zeros((tgt.shape[0], src.shape[0]), np.float32)
    H[np.arange(tgt.shape[0]), idx] = 1.0
    return jnp.asarray(H)


@stability(StabilityLevel.EVOLVING)
def nearest_neighbor_mapping(
    source_points, target_points, *, mode: str = "consistent",
) -> StaticLinearMapping:
    """Nearest-neighbour mapping (a 0/1 selection matrix).

    ``"conservative"`` is the transpose of the reverse selection: each
    source value is *added* to the target point nearest to it, so the
    total is preserved exactly.
    """
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    if mode == "consistent":
        H = _nn_matrix(source_points, target_points)
    else:
        H = _nn_matrix(target_points, source_points).T
    return StaticLinearMapping(H, kind="nearest_neighbor", mode=mode)


@stability(StabilityLevel.EVOLVING)
def projection_1d_mapping(source_boundaries, target_boundaries) -> StaticLinearMapping:
    """Cell-average projection between two 1D grids (integral-preserving).

    ``P[i, j] = |target_i ∩ source_j| / |target_i|``.
    """
    sb = np.asarray(source_boundaries, dtype=np.float64)
    tb = np.asarray(target_boundaries, dtype=np.float64)
    n_src, n_tgt = sb.size - 1, tb.size - 1
    P = np.zeros((n_tgt, n_src), np.float64)
    for i in range(n_tgt):
        lo_t, hi_t = tb[i], tb[i + 1]
        for j in range(n_src):
            overlap = min(hi_t, sb[j + 1]) - max(lo_t, sb[j])
            if overlap > 0:
                P[i, j] = overlap / (hi_t - lo_t)
    return StaticLinearMapping(jnp.asarray(P, jnp.float32), kind="projection_1d",
                               mode="conservative")


@stability(StabilityLevel.EVOLVING)
def matrix_mapping(H, *, mode: str = "consistent", kind: str = "matrix") -> StaticLinearMapping:
    """Wrap a precomputed matrix (e.g. supermesh weights built offline)."""
    return StaticLinearMapping(jnp.asarray(H), kind=kind, mode=mode)


__all__ = [
    "Mapping",
    "StaticLinearMapping",
    "matrix_mapping",
    "nearest_neighbor_mapping",
    "projection_1d_mapping",
    "rbf_mapping",
    "rbf_matrix",
]
