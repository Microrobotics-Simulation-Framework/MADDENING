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

Every factory attaches a :class:`~maddening.core.coupling.mapping_spec.MappingSpec`
(kind, hyper-parameters, *references* to the point sets — never the
weights) so a graph config or USD stage can rebuild the mapping; see
:mod:`maddening.core.coupling.mapping_spec`.  Each is registered under its
kind in :mod:`maddening.core.coupling.mapping_registry`, and
:func:`register_mapping` (experimental) adds a kind of your own to the
same table.

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

import math

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.coupling.mapping_registry import (
    _register_builtin,
    register_mapping,
)
from maddening.core.coupling.mapping_spec import (
    MappingSpec,
    normalise_point_reference,
    reference_for_array,
)

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

    ``params_pytree()`` is snapshotted into ``gm.params["mappings"]``,
    which checkpoints, ``POST /checkpoint/load``, system identification,
    the FMU's state archive and ``to_dict`` all walk as a flat table of
    arrays.  For any class other than :class:`StaticLinearMapping`,
    ``GraphManager.add_edge`` therefore refuses one that is not a plain
    ``dict`` (empty for a mapping without weights) from Python identifiers
    to concrete, finite, floating-point JAX arrays, the same on every call
    (see :func:`~maddening.core.coupling.mapping_registry.register_mapping`).

    **Optional attributes of a geometry-dependent mapping (experimental).**
    None is part of this protocol: the graph reads each with ``getattr``
    and a static mapping has none of them.

    ``needs_geometry`` : bool
        ``True`` for a mapping that reads a moving geometry.  Its edge
        must then name one, ``add_edge(..., mapping=m, geometry=(anchor,
        field))`` with ``anchor`` ``"source"`` or ``"target"``, and
        ``apply`` is called with that state field as ``geom``, at the time
        level the edge's value has.  Absent or ``False``: ``geom`` is
        never passed and a ``geometry=`` on the edge is refused.  A
        registered kind declares the same value with
        ``register_mapping(..., needs_geometry=True)``; a factory whose
        product disagrees with its registration is refused at rebuild.
    ``geometry_shape`` : tuple of int
        The static shape of the geometry ``apply`` reads.  ``compile()``
        refuses an edge whose geometry field has another shape.
    ``accepts_geometry_shape(shape) -> bool``
        Replaces the equality test against ``geometry_shape`` for a
        mapping that takes more than one shape (a one-dimensional grid
        that reads ``(n_points,)`` as well as ``(n_points, 1)``).
    ``geometry_dtype_problems(dtype) -> (errors, warnings)``
        Two lists of sentences about a geometry of that floating dtype,
        asked once when the graph is validated: each error is an
        ``ERROR:`` issue (``compile()`` raises), each warning a
        ``WARNING:`` one.  For what only the mapping knows, such as a
        float32 coordinate that cannot resolve a cell of its grid.
    ``field_shapes() -> (source_lead, target_lead)``
        The leading axes of the field the mapping reads and of the field
        it delivers, as two tuples, for a mapping whose fields are not
        ``(n_source, ...)`` and ``(n_target, ...)`` (a grid field kept in
        the grid's own shape).  ``add_edge`` and ``validate()`` then
        compare these with the two ends instead of ``n_source`` and
        ``n_target``.

    A geometry-dependent mapping must be a pure function of ``(field,
    weights, geom)``: nothing cached on values and nothing carried
    between calls, so that a restart from a checkpoint is the
    uninterrupted run.  It is called under ``jit``, ``grad`` and ``vmap``
    with a traced ``geom``.
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

    Each delivered value is a sum over the matrix's width.  A coupling
    report's float floor does not count that sum's rounding, and
    ``coupling_diagnostics()`` withdraws ``spectral_usable`` at the
    float floor of a group with such a mapping more than ten entries
    wide on an internal edge, whatever the weights are (MADD-ANO-255).
    """
    H: Any
    kind: str = "matrix"
    mode: str = "consistent"
    meta: dict = None  # type: ignore[assignment]  # hyper-parameters, for describe()
    # How the mapping was built (set by the factories); ``None`` for a
    # hand-constructed instance, which then cannot be serialised.
    spec: Optional[MappingSpec] = None

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
        """Kind, mode, shape and hyper-parameters (never the weights).

        ``"kind"`` is always this mapping's user-facing :attr:`kind` —
        the label given to ``matrix_mapping(kind=...)`` included, so a
        dashboard keyed on it (``GET /graph``) sees ``"supermesh"``, not
        ``"matrix"``.  With a :attr:`spec` the rest is that spec's
        ``to_dict()`` (which keeps the factory kind ``"matrix"`` plus
        ``label``) and ``shape`` — what ``EdgeSpec.to_dict`` writes and
        ``MappingSpec.from_dict`` reads back.
        """
        d = {
            "kind": self.kind, "mode": self.mode,
            "shape": [self.n_target, self.n_source], **self.meta,
        }
        if self.spec is not None:
            d.update(self.spec.to_dict())
            d["kind"] = self.kind
        return d

    def __repr__(self) -> str:
        return (f"StaticLinearMapping({self.kind}, {self.mode}, "
                f"{self.n_target}x{self.n_source})")


# ---------------------------------------------------------------------------
# What params_pytree() may contain
# ---------------------------------------------------------------------------


def _params_pytree_problem(tree: Any) -> Optional[str]:
    """Why *tree* cannot be an entry of ``gm.params["mappings"]``, or
    ``None``.  Structure, key names and leaves; one call's worth."""
    if type(tree) is not dict:
        return (f"returned {type(tree).__name__}, not a plain dict of weight name -> "
                f"array")
    for key, leaf in tree.items():
        if not isinstance(key, str) or not key.isidentifier():
            return (f"has the key {key!r}; a weight name must be a Python identifier "
                    f"(it is a member name in a checkpoint archive and a key of a "
                    f"config's param_specs)")
        where = f"entry {key!r}"
        if isinstance(leaf, jax.core.Tracer):
            return (f"{where} is a traced value; the graph snapshots concrete weights, "
                    f"so build the mapping outside jit / grad / vmap")
        if not isinstance(leaf, jax.Array):
            what = ("a nested container" if isinstance(leaf, (dict, list, tuple))
                    else type(leaf).__name__)
            return (f"{where} is {what}, not a JAX array; the entry is a flat table of "
                    f"arrays, and a NumPy array or a Python number would take its "
                    f"dtype from jax_enable_x64 at the moment it was read (use "
                    f"jnp.asarray(value))")
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            return (f"{where} has dtype {leaf.dtype}; a weight is a real "
                    f"floating-point array (keep indices and other integer structure "
                    f"as attributes of the mapping, outside the parameter tree)")
        if not bool(np.all(np.isfinite(np.asarray(leaf)))):
            return f"{where} holds a non-finite value (NaN or infinity)"
    return None


def _params_contract_problem(mapping: Any) -> Optional[str]:
    """Why ``mapping.params_pytree()`` cannot back an edge, or ``None``.

    The answer starts with ``params_pytree()`` so that a caller can put
    its own subject in front.  A :class:`StaticLinearMapping` is not
    asked: its entry is ``{"H": <2-D array>}`` by construction, and what
    its matrix may hold (an integer selection matrix, say) is unchanged.

    Every other class is held to what each reader of
    ``gm.params["mappings"]`` assumes, none of which checks it for itself:

    * checkpoints and the FMU state archive store each leaf as the member
      ``<edge key>/<name>`` -- a nested dict would be pickled as an object
      array, which the loader refuses, and a ``/`` in a name is read back
      as part of the edge key and the weight silently not restored;
    * ``param_specs`` / ``set_param_spec``, the trainable mask and
      ``sysid`` address a weight as ``mappings -> edge -> name``, three
      keys deep, and fit only floating-point leaves;
    * ``POST /checkpoint/load`` judges each leaf it would install as one
      numeric array, against its bounds (``PUT /graph/params`` is per node
      and has no door to a mapping's weights);
    * ``to_dict`` warns when the live weights differ from
      ``params_pytree()``, and ``reset_params()`` restores it, so two
      calls must agree, and a NaN (unequal to itself) would warn forever.
    """
    if type(mapping) is StaticLinearMapping:
        return None
    first = mapping.params_pytree()
    problem = _params_pytree_problem(first)
    if problem is not None:
        return f"params_pytree() {problem}"
    second = mapping.params_pytree()
    if type(second) is not dict or list(second) != list(first):
        return ("params_pytree() is not the same on every call: a second call "
                f"returned the keys {list(second) if isinstance(second, dict) else second!r}"
                f" after {list(first)}")
    for key, leaf in first.items():
        again = second[key]
        if not isinstance(again, jax.Array) or isinstance(again, jax.core.Tracer) \
                or again.shape != leaf.shape or again.dtype != leaf.dtype \
                or not np.array_equal(np.asarray(again), np.asarray(leaf)):
            return (f"params_pytree() is not the same on every call: entry {key!r} "
                    f"changed between two calls.  It is what reset_params() restores "
                    f"and what to_dict() compares the live weights with, so compute "
                    f"the weights once, when the mapping is built")
    return None


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


def _finite_real(name: str, value) -> float:
    """``value`` as a float; a non-real or non-finite hyper-parameter is a
    ``ValueError`` (it is meaningless for the kernel and JSON has no
    ``Infinity`` / ``NaN``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        # An integer no float64 holds: ``math.isfinite`` raises for it.
        raise ValueError(f"{name} must be finite, got an integer of "
                         f"{len(str(abs(int(value))))} digits, which no float64 holds"
                         ) from None
    if not finite:
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


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

    Both point sets must be finite ``(n,)`` or ``(n, d)`` arrays of the
    same ``d``, and ``source_points`` must hold at least one point;
    anything else is a ``ValueError`` naming the argument and the first
    offending point (a NaN coordinate used to give a matrix of NaN, and
    an infinite target a row of zeros or of infinities, without a word).
    Coincident source points are accepted: the ridge shares the weight
    between them, and with ``ridge=0`` the solve raises
    ``numpy.linalg.LinAlgError`` (a ``ValueError``).
    """
    from maddening.core.coupling import _mapping_checks as _checks  # noqa: PLC0415

    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; choose from {_KERNELS}")
    epsilon = _finite_real("epsilon", epsilon)
    ridge = _finite_real("ridge", ridge)
    dtype = _out_dtype(source_points, target_points)
    src = _checks.checked_points("source_points", source_points)
    tgt = _checks.checked_points("target_points", target_points, allow_empty=True)
    _checks.check_same_dimension(src, tgt)
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


@_register_builtin(
    "rbf",
    arrays=("source_points", "target_points"),
    hyperparameters={"kernel": str, "epsilon": float, "polynomial": bool,
                     "ridge": float, "mode": str},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
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
    source_ref=None,
    target_ref=None,
) -> StaticLinearMapping:
    """RBF mapping from ``source_points`` to ``target_points``.

    ``mode="consistent"`` interpolates values.  ``mode="conservative"``
    builds the consistent interpolant from the *target* points to the
    *source* points and applies its transpose, so a quantity summed over
    the source (a total force) is summed identically over the target —
    exactly when the reverse interpolant reproduces constants, i.e. with
    ``polynomial=True``.

    ``source_ref`` / ``target_ref`` say where the points come from for
    serialisation (``{"node": name, "field": key}`` or
    ``{"asset": "file.npy"}``, see
    :mod:`~maddening.core.coupling.mapping_spec`); without them a set of
    at most ``INLINE_POINT_LIMIT`` points is inlined into the spec and a
    larger one leaves the mapping unserialisable.

    The point sets are checked as in :func:`rbf_matrix`, under the names
    given here: the set the interpolant is built on (``source_points`` in
    consistent mode, ``target_points`` in conservative mode) must hold at
    least one point.
    """
    from maddening.core.coupling import _mapping_checks as _checks  # noqa: PLC0415

    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    epsilon = _finite_real("epsilon", epsilon)
    ridge = _finite_real("ridge", ridge)
    # Checked here as well as in ``rbf_matrix`` so that the refusal names
    # this function's argument: conservative mode hands the two sets to
    # ``rbf_matrix`` the other way round.
    _checks.check_same_dimension(
        _checks.checked_points("source_points", source_points,
                               allow_empty=mode != "consistent"),
        _checks.checked_points("target_points", target_points,
                               allow_empty=mode == "consistent"),
    )
    # Annotated: the literal mixes `str` and `float`, so without this the
    # `**kw` expansion offers `str | float` to every keyword parameter.
    kw: dict[str, Any] = dict(kernel=kernel, epsilon=epsilon,
                              polynomial=polynomial, ridge=ridge)
    if mode == "consistent":
        H = rbf_matrix(source_points, target_points, **kw)
    else:
        H = rbf_matrix(target_points, source_points, **kw).T
    meta = dict(kernel=kernel, epsilon=float(epsilon), polynomial=bool(polynomial),
                ridge=float(ridge))
    spec = MappingSpec("rbf", {**meta, "mode": mode}, {
        "source_points": reference_for_array(source_points, source_ref, name="source_points"),
        "target_points": reference_for_array(target_points, target_ref, name="target_points"),
    })
    return StaticLinearMapping(H, kind="rbf", mode=mode, meta=meta, spec=spec)


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


@_register_builtin(
    "nearest_neighbor",
    arrays=("source_points", "target_points"),
    hyperparameters={"mode": str},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
@stability(StabilityLevel.EVOLVING)
def nearest_neighbor_mapping(
    source_points, target_points, *, mode: str = "consistent",
    source_ref=None, target_ref=None,
) -> StaticLinearMapping:
    """Nearest-neighbour mapping (a 0/1 selection matrix).

    ``"conservative"`` is the transpose of the reverse selection: each
    source value is *added* to the target point nearest to it, so the
    total is preserved exactly.  ``source_ref`` / ``target_ref`` as in
    :func:`rbf_mapping`.

    Both point sets must be finite ``(n,)`` or ``(n, d)`` arrays of the
    same ``d``, and the set the nearest point is searched in
    (``source_points`` in consistent mode, ``target_points`` in
    conservative mode) must hold at least one point; anything else is a
    ``ValueError`` naming the argument and the first offending point.  A
    NaN coordinate used to be *selected*: ``argmin`` returns a NaN
    distance, so every target read the one source that had no position.
    Among equidistant points the lowest index is taken.
    """
    from maddening.core.coupling import _mapping_checks as _checks  # noqa: PLC0415

    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    _checks.check_same_dimension(
        _checks.checked_points("source_points", source_points,
                               allow_empty=mode != "consistent"),
        _checks.checked_points("target_points", target_points,
                               allow_empty=mode == "consistent"),
    )
    if mode == "consistent":
        H = _nn_matrix(source_points, target_points)
    else:
        H = _nn_matrix(target_points, source_points).T
    spec = MappingSpec("nearest_neighbor", {"mode": mode}, {
        "source_points": reference_for_array(source_points, source_ref, name="source_points"),
        "target_points": reference_for_array(target_points, target_ref, name="target_points"),
    })
    return StaticLinearMapping(H, kind="nearest_neighbor", mode=mode, spec=spec)


@_register_builtin(
    "projection_1d",
    arrays=("source_boundaries", "target_boundaries"),
    hyperparameters={},
    references={"source_boundaries": "source_ref", "target_boundaries": "target_ref"},
)
@stability(StabilityLevel.EVOLVING)
def projection_1d_mapping(
    source_boundaries, target_boundaries, *, source_ref=None, target_ref=None,
) -> StaticLinearMapping:
    """Cell-average projection between two 1D grids (integral-preserving).

    ``P[i, j] = |target_i ∩ source_j| / |target_i|``.  ``source_ref`` /
    ``target_ref`` reference the boundary arrays for serialisation, as
    in :func:`rbf_mapping`.

    Each boundary array must be one-dimensional, hold at least two
    values, be finite and be **strictly increasing**; anything else is a
    ``ValueError`` naming the argument and the first offending index.
    The arrays are not sorted or reversed for you, because the field
    keeps its cell order.  (The overlap formula assumes increasing
    boundaries: a descending array used to give a matrix of zeros and a
    non-monotone one rows that sum to more than one, without a word.)

    The two grids need not cover the same interval.  Outside the other
    grid a cell is treated as empty, so:

    * the integral is preserved, ``sum_i |target_i| (P f)_i = sum_j
      |source_j| f_j``, when the target grid covers the source grid; a
      part of the source outside the target is dropped;
    * a constant is reproduced (a row sums to one) on every target cell
      the source grid covers; a target cell partly outside it averages
      in zeros, and one wholly outside it is zero.  Two grids that share
      no interval therefore give a matrix of zeros: check that they are
      in the same units and frame.
    """
    from maddening.core.coupling import _mapping_checks as _checks  # noqa: PLC0415

    sb = _checks.checked_boundaries("source_boundaries", source_boundaries)
    tb = _checks.checked_boundaries("target_boundaries", target_boundaries)
    n_src, n_tgt = sb.size - 1, tb.size - 1
    P = np.zeros((n_tgt, n_src), np.float64)
    for i in range(n_tgt):
        lo_t, hi_t = tb[i], tb[i + 1]
        for j in range(n_src):
            overlap = min(hi_t, sb[j + 1]) - max(lo_t, sb[j])
            if overlap > 0:
                P[i, j] = overlap / (hi_t - lo_t)
    spec = MappingSpec("projection_1d", {}, {
        "source_boundaries": reference_for_array(
            source_boundaries, source_ref, name="source_boundaries"),
        "target_boundaries": reference_for_array(
            target_boundaries, target_ref, name="target_boundaries"),
    })
    return StaticLinearMapping(jnp.asarray(P, jnp.float32), kind="projection_1d",
                               mode="conservative", spec=spec)


@stability(StabilityLevel.EVOLVING)
def matrix_mapping(
    H, *, mode: str = "consistent", kind: str = "matrix", asset: Optional[str] = None,
) -> StaticLinearMapping:
    """Wrap a precomputed matrix (e.g. supermesh weights built offline).

    A matrix is never inlined into a config: to make the mapping
    serialisable pass ``asset="<file>.npy"`` (a path relative to the
    directory the config / USD stage is saved in) and save ``H`` there
    with ``numpy.save``; ``from_dict`` / ``load_graph_from_usd`` read it
    back from ``base_dir``.  Without ``asset`` the mapping works but
    ``GraphManager.to_dict`` and the USD writer refuse it.  ``kind`` is
    a free string label: it is the mapping's ``kind`` (``describe()``,
    ``GET /graph``) and is recorded as the spec's ``label``; the spec's
    own kind stays ``"matrix"`` so the rebuild finds this factory.  The
    asset reference records the content hash of ``H``, so the file read
    back must hold exactly this matrix.

    ``H`` must be finite: a NaN or an infinity is a ``ValueError`` naming
    its index (``PUT /graph/params`` and ``POST /checkpoint/load`` refuse
    a non-finite mapping weight too).  A traced ``H`` has no values to
    check and is taken as given.
    """
    from maddening.core.coupling import _mapping_checks as _checks  # noqa: PLC0415

    if not isinstance(kind, str) or not kind:
        raise ValueError(f"matrix_mapping: kind must be a non-empty string label, got {kind!r}")
    if not _checks.is_traced(H):
        _checks.check_finite("H", H)
    hyper = {"mode": mode} if kind == "matrix" else {"mode": mode, "label": kind}
    ref = None if asset is None else normalise_point_reference(asset, name="H")
    if ref is not None and "asset" not in ref:
        raise ValueError("matrix_mapping: asset= must name a .npy/.npz file")
    if ref is not None:
        ref = reference_for_array(H, ref, name="H", inline_ok=False)
    spec = MappingSpec("matrix", hyper, {"H": ref})
    return StaticLinearMapping(jnp.asarray(H), kind=kind, mode=mode, spec=spec)


@_register_builtin(
    "matrix",
    arrays=("H",),
    hyperparameters={"mode": str, "label": str},
    references={"H": "asset"},
)
def _matrix_from_spec(H, *, mode: str = "consistent", label: str = "matrix",
                      asset=None) -> StaticLinearMapping:
    """:func:`matrix_mapping` as a ``MappingSpec`` of kind ``"matrix"``
    calls it: the spec's ``label`` hyper-parameter is the factory's
    ``kind`` argument (the user-facing label), because ``kind`` in a spec
    names the factory."""
    return matrix_mapping(H, mode=mode, kind=label, asset=asset)


__all__ = [
    "Mapping",
    "MappingSpec",
    "StaticLinearMapping",
    "matrix_mapping",
    "nearest_neighbor_mapping",
    "projection_1d_mapping",
    "rbf_mapping",
    "rbf_matrix",
    "register_mapping",
]
