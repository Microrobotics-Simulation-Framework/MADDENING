"""Static sparse interface mappings: a row structure, applied as a gather.

A :class:`~maddening.core.coupling.mapping.StaticLinearMapping` stores the
whole ``n_target x n_source`` matrix.  For two interfaces of a million
points each that matrix has ``1e12`` entries, of which a nearest-neighbour
selection uses one per row.  :class:`StaticSparseMapping` stores only the
entries a row uses: an integer index ``(n_rows, k)`` kept on the mapping,
and one floating-point weight array of the same shape, which is the
mapping's entry of ``gm.params["mappings"]``.

*Experimental in 0.4.0: everything in this module may change in a minor
release.*

Three kinds are registered (through
:func:`~maddening.core.coupling.mapping_registry.register_mapping`, like
any kind from another library, so they are held to the same contract):

``sparse_nearest_neighbor`` -- :func:`sparse_nearest_neighbor_mapping`
    The sparse form of ``nearest_neighbor``: the same matrix, entry for
    entry, found with a k-d tree instead of an ``n_target x n_source``
    distance table.
``sparse_projection_1d`` -- :func:`sparse_projection_1d_mapping`
    The sparse form of ``projection_1d``: the same entries, found with a
    sorted sweep.
``sparse_matrix`` -- :func:`sparse_matrix_mapping`
    Bring your own rows: an index array and a value array.

The index is structure, not a parameter
---------------------------------------
Only the weights ``W`` are in the parameter tree, where they are a traced
input like any other weight (differentiable, replaceable without a
recompile, carried by a checkpoint).  The index stays on the mapping as a
read-only host array and is **baked into the compiled step as a
constant**.  A weight write can therefore never change the sparsity
pattern: a new pattern is a new mapping, a new edge and a recompile.

Because the weights mean something only against the index they were built
for, a checkpoint records the pattern's digest
(:meth:`StaticSparseMapping.structure_digest`) beside the weights, and
``load_state`` refuses weights that were saved for another pattern.

Two layouts, and the conservative nearest neighbour
---------------------------------------------------
``layout="gather"`` (the default): row ``i`` lists the sources of target
``i``, and ``apply`` is ``sum_j W[i, j] * field[indices[i, j]]`` -- a
gather and a sum along each row.  Every output row is reduced on its own,
with no accumulator shared between rows.

``layout="scatter"``: row ``j`` lists the targets of *source* ``j``, and
``apply`` adds ``W[j, m] * field[j]`` into target ``indices[j, m]`` -- a
scatter-add.  It stores one slot per entry whatever the pattern looks
like, where the gather layout pads every row to the longest one.

The conservative nearest neighbour is the transpose of the reverse
selection (each source adds to its nearest target), so its rows are as
long as the number of sources that share a target.
:func:`sparse_nearest_neighbor_mapping` builds it in the gather layout by
default and refuses a pattern whose padding exceeds
:data:`MAX_SPARSE_STRUCTURE_BYTES`; ``transpose="scatter"`` builds the same
operator in the scatter layout.

What was measured about the two (jax / jaxlib 0.11.0, 2026-10-05):

* a scatter-add with repeated indices gave **one** result on the CPU, bit
  equal to the in-order sum, and **twenty different results in twenty
  runs on a GPU**;
* the gather-and-sum gave one result on each backend (not the same bits
  on the two).

So the scatter layout is not reproducible run to run on a GPU, and nothing
more than the above is claimed for either.  The reverse-mode derivative of
a gather with respect to the *field* is itself a scatter-add, so on a GPU
a gradient with respect to the source field through either layout may
differ in its last bits between runs even where the forward result does
not.

Summation order
---------------
The row sum is deterministic for one compiled program.  It is not the
dense mapping's summation order, and it is not bit-stable between a
``jax.vmap`` of a step and the step alone (neither is ``H @ field``): the
sparse and the dense mapping agree to rounding, entry by entry, not bit
for bit.  For one entry per row the two are equal as numbers when the
mapping is applied on its own.  Inside a compiled step not even that
carries over, whatever the weights: a graph with a sparse mapping and the
same graph with the dense matrix are two programs, and the compiler
evaluates the arithmetic around the mapping differently in each (a
product fused with a neighbouring addition in one and not in the other).
The two graphs step to within rounding of one another, and nothing closer
is claimed.

Limits
------
* Every referenced array is bounded by the reference resolver
  (``mapping_spec.MAX_ASSET_BYTES``), as for every kind.
* The row structure -- index plus weights -- is bounded by
  :data:`MAX_SPARSE_STRUCTURE_BYTES`, checked from the row counts before
  the padded arrays are allocated.
* A nearest-neighbour search over a degenerate point set (very many
  points at one distance from very many others) is bounded by
  :data:`TIE_CANDIDATES_PER_POINT` and :data:`TIE_CANDIDATES_FLOOR`.

Each refusal is a :class:`SparseMappingLimitError`, a ``ValueError``.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Optional, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.coupling import _mapping_checks as _checks
from maddening.core.coupling.mapping import _MODES
from maddening.core.coupling.mapping_registry import register_mapping
from maddening.core.coupling.mapping_spec import (
    MAX_ASSET_BYTES,
    MappingSpec,
    _check_element_width,
    normalise_point_reference,
    reference_for_array,
)

#: Largest row structure a builder may produce, in bytes: the padded index
#: (4 bytes a slot) plus the padded weights.  Checked from the row counts
#: before anything padded is allocated.  Module constant: raise it for a
#: genuinely large interface (``sparse_mapping.MAX_SPARSE_STRUCTURE_BYTES =
#: ...``).  At the default, 33.5 million float32 slots: 4.2 million targets
#: with eight entries each.
MAX_SPARSE_STRUCTURE_BYTES = MAX_ASSET_BYTES

#: A nearest-neighbour search re-examines, with the dense expression, every
#: point within rounding of the nearest one.  The candidates it may collect
#: in total are ``TIE_CANDIDATES_PER_POINT * n + TIE_CANDIDATES_FLOOR`` for
#: ``n`` searched points; a point set that needs more is degenerate (very
#: many points equidistant from very many others) and is refused.
TIE_CANDIDATES_PER_POINT = 8
TIE_CANDIDATES_FLOOR = 1_000_000

#: The two ways a row structure is applied; see the module docstring.
_LAYOUTS = ("gather", "scatter")

#: Bytes of one index slot (int32).
_INDEX_BYTES = 4

#: The largest size an int32 index can address.
_MAX_SIZE = 2 ** 31 - 1

#: Points are searched this many at a time (so a degenerate set is refused
#: after one chunk's search), and each batch of candidate lists of the
#: tied ones holds at most this many candidates.
_TIE_CHUNK = 4096
_TIE_BATCH_CANDIDATES = 1_000_000

#: A coordinate beyond this magnitude overflows the squared distance the
#: nearest-neighbour rule is stated in.
_COORDINATE_LIMIT = 1e150

_EPS64 = float(np.finfo(np.float64).eps)

#: How far apart, relatively, the two nearest distances a k-d tree reports
#: must be before the nearer point is taken without consulting the dense
#: expression, in units of the float64 epsilon; ``2 * dimension`` more is
#: added for the rounding of a sum of that many squares.  The tree's own
#: distances and its pruning bounds carry a rounding of a few tens of
#: epsilons at most (one or two per level of a tree at most 64 deep), and
#: so does the dense expression; this is an order of magnitude wider than
#: both together.  Wider only costs a re-examination, never a result.
_TIE_BAND_EPSILONS = 1024

#: Below this distance a squared coordinate difference is subnormal: it is
#: rounded to a multiple of the smallest float, so two ways of computing a
#: squared distance (a tree's, with a fused multiply-add on some builds,
#: and the dense expression's) can differ by far more than the band above.
#: Every point this close to the nearest one is handed to the dense
#: expression, whatever distances the tree reported for them.
_UNDERFLOW_RADIUS = 2.0 * math.sqrt(float(np.finfo(np.float64).tiny))


class SparseMappingLimitError(ValueError):
    """A sparse mapping would exceed a limit of this module: its row
    structure :data:`MAX_SPARSE_STRUCTURE_BYTES`, or its nearest-neighbour
    search the tie-candidate bound.  Raised before the allocation."""


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def _host_array(name: str, values: Any) -> np.ndarray:
    """*values* as a NumPy array; a traced value has no host data."""
    if isinstance(values, jax.core.Tracer):
        raise ValueError(
            f"{name} is a traced value; a sparse mapping is built once, on the host, "
            f"outside jit / grad / vmap"
        )
    return np.asarray(values)


def _checked_size(name: str, value: Any) -> int:
    """*value* as a size an int32 index can address, or a ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    size = int(value)
    if not 1 <= size <= _MAX_SIZE:
        raise ValueError(
            f"{name} must be between 1 and 2**31 - 1 = {_MAX_SIZE} (the index is "
            f"int32), got {size}"
        )
    return size


def _first(mask: np.ndarray) -> tuple:
    """Index of the first ``True`` of *mask*, in C order."""
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape))


@stability(StabilityLevel.EXPERIMENTAL)
@dataclass(frozen=True, eq=False)
class StaticSparseMapping:
    """A fixed sparse linear map: a row structure and one weight per slot.

    Parameters
    ----------
    indices : array of int, shape ``(n_rows, k)``
        The row structure.  In the ``"gather"`` layout row ``i`` holds the
        source indices of target ``i``; in the ``"scatter"`` layout row
        ``j`` holds the target indices of source ``j``.  Kept on the
        mapping as a read-only ``int32`` host array and baked into the
        compiled step; never part of the parameter tree.
    weights : floating-point array, shape ``(n_rows, k)``
        One weight per slot, held as a JAX array (``jnp.asarray(weights)``):
        the mapping's one parameter, ``params_pytree() == {"W": weights}``.
        A padded slot holds 0.
    n_source : int
        Size of the source field along its first axis.  It cannot be
        inferred from a gather index (the largest index used may be
        smaller); in the scatter layout it must equal ``n_rows``.
    counts : array of int, shape ``(n_rows,)``, optional
        Valid slots per row: slots ``0 .. counts[i] - 1`` of row ``i`` are
        entries, the rest are padding.  ``None`` when every row is full.
    kind, mode : str
        As for every mapping.  ``mode`` is a label here: the operator is
        what the rows say.
    meta : dict, optional
        Hyper-parameters for :meth:`describe` when there is no ``spec``.
    spec : MappingSpec, optional
        How the mapping was built (set by the factories); ``None`` for a
        hand-constructed instance, which then cannot be serialised.
    layout : ``"gather"`` or ``"scatter"``
        How the rows are applied; see the module docstring.
    n_target : int, optional
        Size of the target field along its first axis: ``n_rows`` in the
        gather layout (and may be omitted), required in the scatter one.

    Notes
    -----
    Validated once, on the host, when it is constructed: the shapes, an
    integer index with every entry in range, ``counts`` within ``[0, k]``,
    a padded slot holding index 0 and weight exactly 0, finite
    floating-point weights.  The index is then frozen.

    Padded slots are masked out of the gathered (or scattered) *values*,
    so a padded slot contributes an exact zero whatever the field holds at
    index 0, and the derivative with respect to a padded weight is zero.

    Field shapes ``(n,)`` and ``(n, C)`` are the tested domain.  A field
    whose first axis is not the mapping's size is a ``ValueError`` when the
    step is traced (a gather would otherwise clamp the index and return a
    result).

    Compared and hashed by identity, like ``StaticLinearMapping``.
    """

    indices: Any
    weights: Any
    n_source: int
    counts: Optional[Any] = None
    kind: str = "sparse_matrix"
    mode: str = "consistent"
    meta: dict = None  # type: ignore[assignment]  # hyper-parameters, for describe()
    spec: Optional[MappingSpec] = None
    layout: str = "gather"
    n_target: Optional[int] = None

    #: A static mapping: ``apply`` ignores ``geom``.
    needs_geometry = False

    def __post_init__(self):
        if self.layout not in _LAYOUTS:
            raise ValueError(f"layout={self.layout!r} not in {_LAYOUTS}")
        if self.mode not in _MODES:
            raise ValueError(f"mode={self.mode!r} not in {_MODES}")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"kind must be a non-empty string, got {self.kind!r}")

        raw = _host_array("indices", self.indices)
        if raw.dtype.kind not in "iu":
            raise ValueError(
                f"indices must be an integer array, got dtype {raw.dtype}"
            )
        if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
            raise ValueError(
                f"indices must have shape (n_rows, k) with at least one row and one "
                f"slot, got {raw.shape}"
            )
        n_rows, k = (int(s) for s in raw.shape)
        n_source = _checked_size("n_source", self.n_source)
        if self.layout == "gather":
            n_target = n_rows if self.n_target is None else self.n_target
            n_target = _checked_size("n_target", n_target)
            if n_target != n_rows:
                raise ValueError(
                    f"n_target={n_target} does not match the {n_rows} rows of the "
                    f"index (in the gather layout a row is a target)"
                )
            n_columns, columns = n_source, "n_source"
        else:
            if self.n_target is None:
                raise ValueError(
                    "the scatter layout needs n_target: a row is a source there, and "
                    "the largest target index used need not be the last target"
                )
            n_target = _checked_size("n_target", self.n_target)
            if n_source != n_rows:
                raise ValueError(
                    f"n_source={n_source} does not match the {n_rows} rows of the "
                    f"index (in the scatter layout a row is a source)"
                )
            n_columns, columns = n_target, "n_target"

        counts = None
        valid = np.ones((n_rows, k), dtype=bool)
        if self.counts is not None:
            given = _host_array("counts", self.counts)
            if given.dtype.kind not in "iu" or given.shape != (n_rows,):
                raise ValueError(
                    f"counts must be an integer array of shape ({n_rows},), got "
                    f"{given.dtype}{given.shape}"
                )
            if bool(np.any(given < 0)) or bool(np.any(given > k)):
                at = _first((given < 0) | (given > k))[0]
                raise ValueError(
                    f"counts[{at}] = {int(given[at])} is outside [0, {k}], the slots "
                    f"of a row"
                )
            if bool(np.any(given != k)):
                counts = np.array(given, dtype=np.int32, order="C")
                valid = np.arange(k)[None, :] < counts[:, None]
        out_of_range = valid & ((raw < 0) | (raw >= n_columns))
        if bool(out_of_range.any()):
            at = _first(out_of_range)
            raise ValueError(
                f"indices[{at[0]}, {at[1]}] = {int(raw[at])} is outside [0, {columns}) "
                f"= [0, {n_columns})"
            )
        stray = ~valid & (raw != 0)
        if bool(stray.any()):
            at = _first(stray)
            raise ValueError(
                f"indices[{at[0]}, {at[1]}] = {int(raw[at])} is a padded slot (row "
                f"{at[0]} has {int(cast(np.ndarray, counts)[at[0]])} entries); a padded "
                f"slot holds index 0"
            )
        indices = np.array(raw, dtype=np.int32, order="C")
        indices.setflags(write=False)
        if counts is not None:
            counts.setflags(write=False)

        if isinstance(self.weights, jax.core.Tracer):
            raise ValueError(
                "weights is a traced value; the graph snapshots concrete weights, so "
                "build the mapping outside jit / grad / vmap"
            )
        weights = jnp.asarray(self.weights)
        if not jnp.issubdtype(weights.dtype, jnp.floating):
            raise ValueError(
                f"weights must be a floating-point array, got dtype {weights.dtype}"
            )
        if tuple(weights.shape) != (n_rows, k):
            raise ValueError(
                f"weights has shape {tuple(weights.shape)}, the index has "
                f"{(n_rows, k)}: one weight per slot"
            )
        host = np.asarray(weights)
        if host.dtype.kind != "f":
            host = host.astype(np.float64)       # bfloat16: widened exactly
        if not bool(np.all(np.isfinite(host))):
            at = _first(~np.isfinite(host))
            raise ValueError(
                f"weights[{at[0]}, {at[1]}] is not finite ({host[at].item()!r})"
            )
        loaded = ~valid & (host != 0)
        if bool(loaded.any()):
            at = _first(loaded)
            raise ValueError(
                f"weights[{at[0]}, {at[1]}] = {host[at].item()!r} is in a padded slot; "
                f"a padded slot holds exactly 0"
            )

        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "n_source", n_source)
        object.__setattr__(self, "n_target", n_target)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    # -- sizes ------------------------------------------------------------

    @property
    def k(self) -> int:
        """Slots per row (the longest row's entries)."""
        return int(self.indices.shape[1])

    @property
    def nnz(self) -> int:
        """Entries: the valid slots of every row."""
        if self.counts is None:
            return int(self.indices.size)
        return int(np.sum(self.counts, dtype=np.int64))

    # -- the Mapping protocol ---------------------------------------------

    def params_pytree(self) -> dict:
        return {"W": self.weights}

    def _weights(self, weights: Optional[dict]):
        w = self.weights if weights is None else weights["W"]
        if tuple(jnp.shape(w)) != tuple(self.indices.shape):
            raise ValueError(
                f"{self!r}: the weights W have shape {tuple(jnp.shape(w))}, the row "
                f"structure has {tuple(self.indices.shape)}: one weight per slot"
            )
        return w

    def _field(self, field, size: int, what: str):
        field = jnp.asarray(field)
        if field.ndim < 1 or int(field.shape[0]) != size:
            raise ValueError(
                f"{self!r}: the field has shape {tuple(field.shape)}; its first axis "
                f"must be {what} = {size}"
            )
        return field

    def _valid(self, trailing: int):
        """The valid-slot mask, shaped to broadcast over a field's
        trailing axes, or ``None`` when every row is full."""
        if self.counts is None:
            return None
        valid = jnp.arange(self.k, dtype=jnp.int32)[None, :] < self.counts[:, None]
        return valid.reshape(valid.shape + (1,) * trailing)

    def _gather(self, field, w):
        """``out[i] = sum_j w[i, j] * field[indices[i, j]]``."""
        trailing = field.ndim - 1
        gathered = field[self.indices]
        valid = self._valid(trailing)
        if valid is not None:
            gathered = jnp.where(valid, gathered, jnp.zeros((), gathered.dtype))
        return jnp.sum(w.reshape(w.shape + (1,) * trailing) * gathered, axis=1)

    def _scatter(self, field, w, size: int):
        """``out[indices[j, m]] += w[j, m] * field[j]``, from zero."""
        trailing = field.ndim - 1
        spread = field[:, None]
        valid = self._valid(trailing)
        if valid is not None:
            spread = jnp.where(valid, spread, jnp.zeros((), spread.dtype))
        terms = w.reshape(w.shape + (1,) * trailing) * spread
        out = jnp.zeros((size,) + tuple(field.shape[1:]), terms.dtype)
        return out.at[self.indices].add(terms)

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        """The source field mapped onto the target; ``geom`` is ignored."""
        w = self._weights(weights)
        field = self._field(field, self.n_source, "n_source")
        if self.layout == "gather":
            return self._gather(field, w)
        return self._scatter(field, w, cast(int, self.n_target))

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        """The transpose map, target to source.  Not on the step path."""
        w = self._weights(weights)
        field = self._field(field, cast(int, self.n_target), "n_target")
        if self.layout == "gather":
            return self._scatter(field, w, self.n_source)
        return self._gather(field, w)

    # -- description ------------------------------------------------------

    def describe(self) -> dict:
        """Kind, mode, shape and hyper-parameters (never the index or the
        weights), exactly as ``StaticLinearMapping.describe`` writes them:
        with a :attr:`spec`, its ``to_dict()`` plus ``mode`` and ``shape``.
        """
        d = {
            "kind": self.kind, "mode": self.mode,
            "shape": [self.n_target, self.n_source], **self.meta,
        }
        if self.spec is not None:
            d.update(self.spec.to_dict())
            d["kind"] = self.kind
        return d

    def structure_digest(self) -> str:
        """SHA-256 (hex) of the row structure: layout, sizes, index, counts.

        Two mappings have the same digest exactly when the same weights
        mean the same operator on both.  A checkpoint stores it beside the
        weights and ``load_state`` refuses weights saved for another one.
        """
        cached = self.__dict__.get("_structure_digest")
        if cached is not None:
            return cached
        h = hashlib.sha256()
        h.update(b"maddening.StaticSparseMapping/1\n")
        h.update(f"{self.layout} {self.n_source} {self.n_target} "
                 f"{self.indices.shape[0]} {self.indices.shape[1]}\n".encode("ascii"))
        h.update(np.ascontiguousarray(self.indices, dtype="<i4").tobytes())
        if self.counts is None:
            h.update(b"\nfull")
        else:
            h.update(b"\ncounts")
            h.update(np.ascontiguousarray(self.counts, dtype="<i4").tobytes())
        digest = h.hexdigest()
        object.__setattr__(self, "_structure_digest", digest)
        return digest

    def __copy__(self):
        return self          # immutable: the index is frozen, the fields are too

    def __deepcopy__(self, memo):
        return self          # a copy's index would be writable under a baked step

    def __repr__(self) -> str:
        layout = "" if self.layout == "gather" else f", {self.layout}"
        return (f"StaticSparseMapping({self.kind}, {self.mode}, "
                f"{self.n_target}x{self.n_source}, k={self.k}{layout})")


# ---------------------------------------------------------------------------
# What a builder is given
# ---------------------------------------------------------------------------
#
# The geometry checks are the dense factories' own (``_mapping_checks``): a
# point set that is not an (n,) or (n, d) array of finite coordinates, and
# boundaries that do not strictly rise, are refused in the same words.  Two
# things are asked first that the dense factories leave to NumPy, and one
# is added for the tree.


def _real_array(name: str, values: Any) -> np.ndarray:
    """*values* as a host array of a real numeric dtype (bool, integer,
    float); a complex, text or object array is refused by name rather than
    coerced, and so is a traced value."""
    raw = _host_array(name, values)
    if raw.dtype.kind not in "biuf":
        raise ValueError(
            f"{name} must be a real numeric array (bool, integer or float), got dtype "
            f"{raw.dtype}"
        )
    return raw


def _checked_points(name: str, values: Any) -> np.ndarray:
    """A point set as an ``(n, d)`` float64 array, or a ``ValueError``.

    ``(n,)`` is read as *n* points on a line.  The set must hold at least
    one point with at least one coordinate, all finite (the dense
    factories' checks) and none beyond the magnitude whose square a
    float64 holds.  Both sides must hold a point: a sparse mapping has at
    least one row.
    """
    arr = _checks.checked_points(name, _real_array(name, values))
    huge = np.abs(arr) > _COORDINATE_LIMIT
    if bool(huge.any()):
        at = _first(huge)
        raise ValueError(
            f"{name} holds a coordinate of magnitude above {_COORDINATE_LIMIT:g} at "
            f"index {at[0]} ({arr[at].item()!r}); its squared distance to another "
            f"point overflows a float64, so no nearest point can be told from it"
        )
    return arr


def _checked_boundaries(name: str, values: Any) -> np.ndarray:
    """Cell boundaries of a 1-D grid as a float64 array, or a ``ValueError``:
    one-dimensional, at least two values, finite, strictly increasing (the
    dense factory's check)."""
    return _checks.checked_boundaries(name, _real_array(name, values))


# ---------------------------------------------------------------------------
# Row structures
# ---------------------------------------------------------------------------


def _check_structure_bytes(what: str, n_rows: int, k: int, weight_bytes: int, *,
                           rows: str = "n_target", detail: str = "",
                           hint: str = "") -> None:
    """Refuse (before anything is allocated) a row structure of *n_rows*
    rows of *k* slots that would exceed :data:`MAX_SPARSE_STRUCTURE_BYTES`."""
    n_bytes = n_rows * k * (_INDEX_BYTES + weight_bytes)
    if n_bytes > MAX_SPARSE_STRUCTURE_BYTES:
        raise SparseMappingLimitError(
            f"{what}: the row structure needs {n_rows} rows of {k} slots at "
            f"{_INDEX_BYTES + weight_bytes} bytes a slot = {n_bytes} bytes, more than "
            f"MAX_SPARSE_STRUCTURE_BYTES={MAX_SPARSE_STRUCTURE_BYTES} "
            f"({MAX_SPARSE_STRUCTURE_BYTES >> 20} MiB): {rows}={n_rows}, largest row "
            f"{k}{detail}.{hint}  For a genuinely large interface, raise "
            f"maddening.core.coupling.sparse_mapping.MAX_SPARSE_STRUCTURE_BYTES"
        )


def _padded_rows(what: str, n_rows: int, rows: np.ndarray, columns: np.ndarray,
                 values: Optional[np.ndarray], *, hint: str = ""):
    """``(indices, weights, counts)`` of the entries ``(rows[e], columns[e])``
    with ``values[e]`` (1 when ``None``), as float32 weights.

    *rows* must be ascending; entries of one row keep their order.  The
    byte cap is checked from the row counts before the padded arrays exist.
    """
    counts = np.bincount(rows, minlength=n_rows).astype(np.int64)
    k = max(1, int(counts.max(initial=0)))
    _check_structure_bytes(
        what, n_rows, k, np.dtype(np.float32).itemsize,
        detail=(f", median row {int(np.median(counts))}, {int(rows.size)} entries; "
                f"every row is padded to the largest one"),
        hint=hint)
    starts = np.cumsum(counts) - counts
    slots = np.arange(rows.size, dtype=np.int64) - starts[rows]
    indices = np.zeros((n_rows, k), dtype=np.int32)
    weights = np.zeros((n_rows, k), dtype=np.float32)
    indices[rows, slots] = columns
    weights[rows, slots] = 1.0 if values is None else values
    return indices, weights, counts


# ---------------------------------------------------------------------------
# Nearest neighbour
# ---------------------------------------------------------------------------


def _kdtree():
    """``scipy.spatial.KDTree``.  scipy is a base dependency; it is
    imported here, on first use, so that ``import maddening`` does not pay
    for it."""
    try:
        from scipy.spatial import KDTree  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "sparse_nearest_neighbor_mapping needs scipy (scipy.spatial.KDTree), "
            f"which could not be imported: {exc}"
        ) from exc
    return KDTree


def _unique_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of *points* and, for each, the lowest index it
    occurs at.  Only an optimisation: it stops a set of identical points
    from making every search a many-way tie."""
    if points.shape[1] == 1:
        column, first = np.unique(points[:, 0], return_index=True)
        return column.reshape(-1, 1), first.astype(np.int64)
    unique, first = np.unique(points, axis=0, return_index=True)
    return np.ascontiguousarray(unique), first.astype(np.int64)


def _nearest_lowest_index(what: str, points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """For each row of *queries*, the nearest row of *points* under the
    dense kind's rule: the **lowest index** among the points whose float64
    squared distance ``np.sum((query - point) ** 2, axis=-1)`` is minimal --
    what ``np.argmin`` over the dense distance table returns.

    A k-d tree only proposes candidates.  A query whose two nearest tree
    distances are further apart than rounding can explain takes the
    nearer; every other query collects each point within the band and the
    dense expression decides among them.  The tree never decides a tie, so
    the result does not depend on the tree or on the scipy version.

    The queries are searched :data:`_TIE_CHUNK` at a time, and the
    candidates the tied ones would collect are counted before any is
    collected.  A degenerate set -- on which a tree cannot prune, so that a
    search measures every query against every point -- is therefore
    refused after one chunk has been searched, not after all of them.
    """
    dim = points.shape[1]
    unique, first = _unique_points(points)
    n_queries = queries.shape[0]
    if unique.shape[0] == 1:
        return np.full(n_queries, first[0], dtype=np.int64)
    tree = _kdtree()(unique)
    band = (_TIE_BAND_EPSILONS + 2 * dim) * _EPS64
    limit = TIE_CANDIDATES_PER_POINT * n_queries + TIE_CANDIDATES_FLOOR
    nearest = np.empty(n_queries, dtype=np.int64)
    total = 0
    for start in range(0, n_queries, _TIE_CHUNK):
        stop = min(start + _TIE_CHUNK, n_queries)
        q = queries[start:stop]
        found = tree.query(q, k=2)
        distance, index = np.asarray(found[0]), np.asarray(found[1])
        if not bool(np.all(np.isfinite(distance))) or int(index.max()) >= unique.shape[0]:
            raise ValueError(
                f"{what}: the nearest-neighbour search found no finite distance for some "
                f"point; the coordinates are outside the range a float64 distance holds"
            )
        # units: dimensionless, a relative widening of a distance
        radius = distance[:, 0] * (1.0 + band)
        # units: the coordinates' own; the floor is where their squares go subnormal
        radius = np.maximum(radius, _UNDERFLOW_RADIUS)
        chosen = first[index[:, 0]]
        tied = np.nonzero(distance[:, 1] <= radius)[0]
        if tied.size:
            sizes = np.asarray(
                tree.query_ball_point(q[tied], radius[tied], return_length=True),
                dtype=np.int64)
            total += int(sizes.sum())
            if total > limit:
                raise SparseMappingLimitError(
                    f"{what}: the point set is degenerate: resolving which point is "
                    f"nearest needs more than {limit} candidates "
                    f"(TIE_CANDIDATES_PER_POINT={TIE_CANDIDATES_PER_POINT} for each of "
                    f"{n_queries} searched points, plus TIE_CANDIDATES_FLOOR="
                    f"{TIE_CANDIDATES_FLOOR}); after {stop} of {n_queries} searched "
                    f"points there were {total}, the largest tie being "
                    f"{int(sizes.max())} points at one distance.  Very many points are "
                    f"equidistant from very many others (points on a sphere around the "
                    f"ones they are searched from, say); perturb or thin them"
                )
            chosen[tied] = _resolve_ties(what, tree, unique, first, q[tied], radius[tied],
                                         sizes)
        nearest[start:stop] = chosen
    return nearest


def _resolve_ties(what: str, tree, unique: np.ndarray, first: np.ndarray,
                  q: np.ndarray, r: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The dense rule for the tied queries *q*: among the points within
    *r* of each, the lowest original index at the minimal dense squared
    distance.  *sizes* is how many points each will collect, already
    counted against the bound; they are collected in batches of whole
    queries, each holding a bounded number of candidates."""
    out = np.empty(q.shape[0], dtype=np.int64)
    edges = [0]
    running = 0
    for i, size in enumerate(sizes.tolist()):
        if running and running + size > _TIE_BATCH_CANDIDATES:
            edges.append(i)
            running = 0
        running += size
    edges.append(len(sizes))
    for lo, hi in zip(edges[:-1], edges[1:]):
        lists = tree.query_ball_point(q[lo:hi], r[lo:hi])
        lengths = np.fromiter((len(found) for found in lists), dtype=np.int64,
                              count=hi - lo)
        if bool(np.any(lengths == 0)):
            raise RuntimeError(
                f"{what}: the tie search lost the nearest point of a query it "
                f"had found; this is a defect, please report it"
            )
        flat = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64,
                           count=int(lengths.sum()))
        rows = np.repeat(np.arange(hi - lo), lengths)
        d2 = np.sum((q[lo:hi][rows] - unique[flat]) ** 2, axis=-1)
        original = first[flat]
        order = np.lexsort((original, d2, rows))
        out[lo:hi] = original[order[np.cumsum(lengths) - lengths]]
    return out


_TRANSPOSES = _LAYOUTS


@register_mapping(
    "sparse_nearest_neighbor",
    arrays=("source_points", "target_points"),
    hyperparameters={"mode": str, "transpose": str},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
@stability(StabilityLevel.EXPERIMENTAL)
def sparse_nearest_neighbor_mapping(
    source_points, target_points, *, mode: str = "consistent",
    transpose: str = "gather", source_ref=None, target_ref=None,
) -> StaticSparseMapping:
    """Nearest-neighbour mapping without the dense distance table.

    The same operator as
    :func:`~maddening.core.coupling.mapping.nearest_neighbor_mapping`,
    entry for entry: scattering the rows into a zero matrix gives that
    factory's ``H`` bit for bit, in both modes.  Ties go where the dense
    kind sends them: to the lowest index among the points at the minimal
    float64 squared distance.

    Parameters
    ----------
    source_points, target_points : array-like, ``(n,)`` or ``(n, d)``
        The two point sets: at least one point each, finite, of one
        dimension, no coordinate beyond ``1e150`` in magnitude.
    mode : ``"consistent"`` or ``"conservative"``
        Consistent: each target takes the value at its nearest source (one
        entry per row).  Conservative: the transpose of the reverse
        selection -- each source value is *added* to the target nearest to
        it, so the total is preserved exactly.
    transpose : ``"gather"`` or ``"scatter"``
        How the conservative operator is applied; ``mode="consistent"``
        takes only ``"gather"``.

        ``"gather"`` (the default) lists, for each target, the sources that
        add to it, padded to the longest list, and sums along each row.
        One result per compiled program on the CPU and on a GPU.  Refused
        with a :class:`SparseMappingLimitError` naming this argument when
        the padding exceeds :data:`MAX_SPARSE_STRUCTURE_BYTES`, which a
        strongly non-uniform pattern does (many sources sharing one
        nearest target).

        ``"scatter"`` stores one entry per source and applies a
        scatter-add: compact whatever the pattern.  Measured on the CPU it
        is the in-order sum, one result; measured on a GPU it gave a
        different result on every run.  Choose it when the gather form is
        refused and run-to-run reproducibility on a GPU is not needed.
    source_ref, target_ref : optional
        Where the points come from, for serialisation, as in
        :func:`~maddening.core.coupling.mapping.rbf_mapping`.

    Returns
    -------
    StaticSparseMapping
        With float32 weights of 1 (as the dense kind's are, under
        ``jax_enable_x64`` too) and a spec that records ``mode`` and
        ``transpose``, so a rebuilt mapping is applied the same way.

    Raises
    ------
    ValueError
        For a point set that is empty, not ``(n,)`` or ``(n, d)``, not real,
        not finite or beyond ``1e150``; for point sets of two dimensions;
        for an unknown ``mode`` or ``transpose``, or ``transpose="scatter"``
        with ``mode="consistent"``.
    SparseMappingLimitError
        For a row structure over :data:`MAX_SPARSE_STRUCTURE_BYTES`, or a
        point set so degenerate that resolving its ties needs more than the
        candidate bound.

    Notes
    -----
    The equality with the dense kind holds wherever the dense expression is
    itself meaningful: for coordinates up to ``1e150``.  Below a spacing of
    about ``1e-154`` the dense squared distance is subnormal and stops
    telling points apart; the rule is still the dense one there (the lowest
    index among the points it cannot tell from the nearest).
    """
    what = "sparse_nearest_neighbor_mapping"
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    if transpose not in _TRANSPOSES:
        raise ValueError(f"transpose={transpose!r} not in {_TRANSPOSES}")
    if mode == "consistent" and transpose != "gather":
        raise ValueError(
            f"transpose={transpose!r} says how the conservative operator (a "
            f"transpose) is applied; mode='consistent' has none and takes only "
            f"transpose='gather'"
        )
    source = _checked_points("source_points", source_points)
    target = _checked_points("target_points", target_points)
    _checks.check_same_dimension(source, target)
    spec = MappingSpec("sparse_nearest_neighbor", {"mode": mode, "transpose": transpose}, {
        "source_points": reference_for_array(source_points, source_ref, name="source_points"),
        "target_points": reference_for_array(target_points, target_ref, name="target_points"),
    })
    n_source, n_target = int(source.shape[0]), int(target.shape[0])
    one = np.dtype(np.float32).itemsize

    if mode == "consistent":
        _check_structure_bytes(what, n_target, 1, one, detail=" (one entry per target)")
        nearest = _nearest_lowest_index(what, source, target)
        return StaticSparseMapping(
            nearest.reshape(-1, 1), jnp.ones((n_target, 1), jnp.float32),
            n_source=n_source, kind="sparse_nearest_neighbor", mode=mode, spec=spec)

    if transpose == "scatter":
        _check_structure_bytes(what, n_source, 1, one, rows="n_source",
                               detail=" (one entry per source)")
        nearest = _nearest_lowest_index(what, target, source)
        return StaticSparseMapping(
            nearest.reshape(-1, 1), jnp.ones((n_source, 1), jnp.float32),
            n_source=n_source, n_target=n_target, layout="scatter",
            kind="sparse_nearest_neighbor", mode=mode, spec=spec)

    nearest = _nearest_lowest_index(what, target, source)
    # Stable: within a target the sources stay in ascending order.
    order = np.argsort(nearest, kind="stable")
    indices, weights, counts = _padded_rows(
        what, n_target, nearest[order], order, None,
        hint=("  This pattern is uneven: many sources share one nearest target.  Pass "
              "transpose='scatter' to apply the same operator as a scatter-add, which "
              "stores one entry per source whatever the pattern (it is not "
              "reproducible run to run on a GPU)."))
    return StaticSparseMapping(
        indices, jnp.asarray(weights), n_source=n_source, counts=counts,
        kind="sparse_nearest_neighbor", mode=mode, spec=spec)


# ---------------------------------------------------------------------------
# 1-D projection
# ---------------------------------------------------------------------------


@register_mapping(
    "sparse_projection_1d",
    arrays=("source_boundaries", "target_boundaries"),
    hyperparameters={},
    references={"source_boundaries": "source_ref", "target_boundaries": "target_ref"},
)
@stability(StabilityLevel.EXPERIMENTAL)
def sparse_projection_1d_mapping(
    source_boundaries, target_boundaries, *, source_ref=None, target_ref=None,
) -> StaticSparseMapping:
    """Cell-average projection between two 1-D grids, without the dense
    double loop.

    ``P[i, j] = |target_i ∩ source_j| / |target_i|``: the same entries as
    :func:`~maddening.core.coupling.mapping.projection_1d_mapping`, each
    evaluated by that factory's own expression, so scattering the rows
    into a zero matrix gives its ``H`` bit for bit.  A row lists the
    source cells a target cell overlaps, in ascending order; a target cell
    outside the source grid has none.

    Both boundary arrays must be **strictly increasing**; anything else is
    a ``ValueError``.  They are not sorted or reversed for you, because the
    field keeps its cell order.

    Parameters
    ----------
    source_boundaries, target_boundaries : array-like, ``(n + 1,)``
        The cell boundaries of the two grids: at least two each, finite,
        strictly increasing.
    source_ref, target_ref : optional
        References to the boundary arrays for serialisation, as in
        :func:`~maddening.core.coupling.mapping.rbf_mapping`.

    Returns
    -------
    StaticSparseMapping
        Mode ``"conservative"``, float32 weights whatever
        ``jax_enable_x64`` says: the dense kind's, bit for bit (see
        :func:`~maddening.core.coupling.mapping.projection_1d_mapping`
        for what that costs in a float64 graph;
        :func:`sparse_matrix_mapping` keeps float64 values there).

    Raises
    ------
    ValueError
        For boundaries that are not one-dimensional, hold fewer than two
        values, are not real or finite, or do not strictly increase.
    SparseMappingLimitError
        For a row structure over :data:`MAX_SPARSE_STRUCTURE_BYTES` (one
        target cell spanning very many source cells).
    """
    what = "sparse_projection_1d_mapping"
    sb = _checked_boundaries("source_boundaries", source_boundaries)
    tb = _checked_boundaries("target_boundaries", target_boundaries)
    spec = MappingSpec("sparse_projection_1d", {}, {
        "source_boundaries": reference_for_array(
            source_boundaries, source_ref, name="source_boundaries"),
        "target_boundaries": reference_for_array(
            target_boundaries, target_ref, name="target_boundaries"),
    })
    n_source, n_target = sb.size - 1, tb.size - 1
    low, high = tb[:-1], tb[1:]
    # Source cell j overlaps target cell i exactly when sb[j + 1] > low_i and
    # sb[j] < high_i -- the dense factory's ``overlap > 0``, for boundaries
    # that rise.  That is a run of consecutive cells, found by two searches.
    j_first = np.maximum(np.searchsorted(sb, low, side="right") - 1, 0)
    j_last = np.minimum(np.searchsorted(sb, high, side="left") - 1, n_source - 1)
    run = np.maximum(j_last - j_first + 1, 0).astype(np.int64)
    rows = np.repeat(np.arange(n_target, dtype=np.int64), run)
    offsets = np.cumsum(run) - run
    columns = np.arange(rows.size, dtype=np.int64) - offsets[rows] + j_first[rows]
    # The dense factory's own expression, entry by entry, in float64.
    overlap = np.minimum(high[rows], sb[columns + 1]) - np.maximum(low[rows], sb[columns])
    values = (overlap / (high - low)[rows]).astype(np.float32)
    indices, weights, counts = _padded_rows(what, n_target, rows, columns, values)
    return StaticSparseMapping(
        indices, jnp.asarray(weights), n_source=n_source, counts=counts,
        kind="sparse_projection_1d", mode="conservative", spec=spec)


# ---------------------------------------------------------------------------
# Bring your own rows
# ---------------------------------------------------------------------------


def _asset_reference(name: str, array: Any, asset: Any, keyword: str) -> Optional[dict]:
    if asset is None:
        return None
    ref = normalise_point_reference(asset, name=name)
    if "asset" not in ref:
        raise ValueError(f"sparse_matrix_mapping: {keyword}= must name a .npy/.npz file")
    return reference_for_array(array, ref, name=name, inline_ok=False)


@register_mapping(
    "sparse_matrix",
    arrays=("indices", "values"),
    hyperparameters={"n_source": int, "mode": str, "name": str},
    references={"indices": "indices_asset", "values": "values_asset"},
)
@stability(StabilityLevel.EXPERIMENTAL)
def sparse_matrix_mapping(
    indices, values, *, n_source: int, mode: str = "consistent", name: str = "",
    indices_asset: Optional[Any] = None, values_asset: Optional[Any] = None,
) -> StaticSparseMapping:
    """Wrap precomputed sparse rows (weights built offline, say).

    Row ``i`` of the operator is ``sum_j values[i, j] * e[indices[i, j]]``:
    ``target[i] = sum_j values[i, j] * source[indices[i, j]]``.

    Parameters
    ----------
    indices : array of int, shape ``(n_target, k)``
        The source index of each slot.  ``-1`` marks an unused slot (a row
        with fewer than ``k`` entries).  The same index may appear twice in
        a row; its values add.
    values : floating-point array, shape ``(n_target, k)``
        The weight of each slot, finite.  The value of an unused slot must
        be exactly 0: a weight that would be dropped is refused rather
        than dropped.
    n_source : int
        Size of the source field.  It cannot be inferred from the indices.
    mode : ``"consistent"`` or ``"conservative"``
        A label: the rows are applied as given.
    name : str
        A free label for display (``describe()``, ``GET /graph``).  The
        mapping's ``kind`` stays ``"sparse_matrix"``.
    indices_asset, values_asset : str or dict, optional
        The two arrays are never inlined into a config.  To make the
        mapping serialisable, save each with ``numpy.save`` (or both as
        members of one ``.npz``) next to the config and name the files
        here, as ``"<file>.npy"`` or ``{"asset": "<file>.npz", "key":
        "<member>"}``.  Each reference records the content hash of its
        array, so the file read back must hold exactly it.  Without both,
        the mapping works but ``GraphManager.to_dict`` and the USD writer
        refuse it.

    Returns
    -------
    StaticSparseMapping
        Each row's used slots are moved to its front, in the order given.
        The weights keep the dtype ``jnp.asarray(values)`` gives.

    Raises
    ------
    ValueError
        For arrays that are not ``(n_target, k)`` with at least one row and
        slot, of two shapes, an index that is not an integer array, below
        ``-1`` or not below ``n_source``, values that are not a
        floating-point array or not finite, a non-zero value in an unused
        slot, an unknown ``mode``, or an ``n_source`` that is not an
        integer between 1 and ``2**31 - 1``.
    SparseMappingLimitError
        For arrays over :data:`MAX_SPARSE_STRUCTURE_BYTES` together.
    """
    what = "sparse_matrix_mapping"
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    if not isinstance(name, str):
        raise ValueError(f"{what}: name must be a string label, got {name!r}")
    n_source = _checked_size("n_source", n_source)
    spec = MappingSpec("sparse_matrix", {"n_source": n_source, "mode": mode, "name": name}, {
        "indices": _asset_reference("indices", indices, indices_asset, "indices_asset"),
        "values": _asset_reference("values", values, values_asset, "values_asset"),
    })

    index = _host_array("indices", indices)
    if index.dtype.kind not in "iu":
        raise ValueError(f"{what}: indices must be an integer array, got dtype {index.dtype}")
    given = _host_array("values", values)
    if not jnp.issubdtype(given.dtype, jnp.floating):
        raise ValueError(
            f"{what}: values must be a floating-point array, got dtype {given.dtype}"
        )
    _check_element_width(given.dtype, "values")
    if values_asset is not None and given.dtype.kind != "f":
        raise ValueError(
            f"{what}: values has dtype {given.dtype}, which a .npy asset cannot hold "
            f"(NumPy stores it as raw bytes); save float32 values to name them with "
            f"values_asset=, or leave the reference out"
        )
    if index.ndim != 2 or index.shape[0] < 1 or index.shape[1] < 1:
        raise ValueError(
            f"{what}: indices must have shape (n_target, k) with at least one row and "
            f"one slot, got {index.shape}"
        )
    if given.shape != index.shape:
        raise ValueError(
            f"{what}: values has shape {given.shape}, indices has {index.shape}: one "
            f"value per slot"
        )
    n_target, k = (int(s) for s in index.shape)
    weight_bytes = np.dtype(jax.dtypes.canonicalize_dtype(given.dtype)).itemsize
    _check_structure_bytes(what, n_target, k, weight_bytes,
                           detail=" (the slots of the arrays given)")

    unused = index == -1 if index.dtype.kind == "i" else np.zeros(index.shape, dtype=bool)
    out_of_range = ~unused & ((index < 0) | (index >= n_source))
    if bool(out_of_range.any()):
        at = _first(out_of_range)
        raise ValueError(
            f"{what}: indices[{at[0]}, {at[1]}] = {int(index[at])} is neither a source "
            f"index in [0, n_source) = [0, {n_source}) nor -1 (an unused slot)"
        )
    # Judged as float64 values: a 16-bit dtype widens exactly.
    judged = given if given.dtype.kind == "f" else given.astype(np.float64)
    _checks.check_finite("values", judged)
    dropped = unused & (judged != 0)
    if bool(dropped.any()):
        at = _first(dropped)
        raise ValueError(
            f"{what}: values[{at[0]}, {at[1]}] = {judged[at].item()!r} belongs to an "
            f"unused slot (indices[{at[0]}, {at[1]}] = -1); it would be dropped, so "
            f"it must be exactly 0"
        )

    counts = None
    if bool(unused.any()):
        counts = np.sum(~unused, axis=1, dtype=np.int64)
        if bool(np.any(unused[:, :-1] & ~unused[:, 1:])):
            # An unused slot before a used one: move the used slots to the
            # front of their row, keeping their order.
            order = np.argsort(unused, axis=1, kind="stable")
            index = np.take_along_axis(index, order, axis=1)
            given = np.take_along_axis(given, order, axis=1)
            unused = np.take_along_axis(unused, order, axis=1)
        index = np.where(unused, 0, index)
        given = np.where(unused, np.zeros((), given.dtype), given)
    return StaticSparseMapping(
        index, jnp.asarray(given), n_source=n_source, counts=counts,
        kind="sparse_matrix", mode=mode, spec=spec)


__all__ = [
    "MAX_SPARSE_STRUCTURE_BYTES",
    "TIE_CANDIDATES_FLOOR",
    "TIE_CANDIDATES_PER_POINT",
    "SparseMappingLimitError",
    "StaticSparseMapping",
    "sparse_matrix_mapping",
    "sparse_nearest_neighbor_mapping",
    "sparse_projection_1d_mapping",
]
