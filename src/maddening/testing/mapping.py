"""Verification of an interface mapping: the edge between two nodes.

:func:`~maddening.testing.verification.verify_node` checks one node.  Two
verified nodes are not yet a verified model: what one hands the other
passes through an edge, and an edge that interpolates across two
discretisations can create or destroy the quantity it carries, or hand
over a value the source never held.  :func:`verify_mapping` is the
battery for that edge.  It works on any object with the members of the
:class:`~maddening.core.coupling.mapping.Mapping` protocol -- built by a
shipped factory, registered with
:func:`~maddening.core.coupling.mapping_registry.register_mapping`, or
written by hand and never registered -- and on a whole
:class:`~maddening.core.edge.EdgeSpec`, where every check is made on
what the edge *delivers* (the mapping, then the edge's ``transform``),
through the one function the step delivers an edge through.

Usage::

    from maddening.testing.mapping import assert_mapping_verified

    assert_mapping_verified(my_mapping)                    # claims: its mode
    assert_mapping_verified(my_mapping, consistent=True, polynomial_order=1,
                            source_coordinates=xs, target_coordinates=xt)

**What the mapping claims is the caller's statement.**  The battery
cannot know what a kind of your own promises and does not guess:
``consistent`` and ``conservative`` default to the mapping's ``mode``,
the polynomial degree it reproduces defaults to zero (constants, the
total), and a property that is not claimed is not checked -- its result
is a ``SKIP`` that says "not claimed", never a pass.

What the two words mean for a transfer:

``consistent``
    The transfer of a *value* (a temperature, a velocity): a field that is
    constant on the source arrives as the same constant, and, for a kind
    that claims degree ``p``, so does every polynomial of the coordinates
    up to that degree.  The degree bounds the order of the transfer: a
    kind that reproduces degree ``p`` carries a smooth field with an
    error of order ``h**(p + 1)``.
``conservative``
    The transfer of an *amount* (a force, a heat flow, a mass): the total
    over the target equals the total over the source, and, for degree
    ``p``, so do the moments up to that degree (degree 1: the first
    moment, so the centroid of what was sent).  With ``source_measure`` /
    ``target_measure`` the totals are weighted sums (a field of
    densities on cells of different sizes).

The adjoint identity, ``<apply(x), y> == <x, apply_T(y)>``, is what ties
the two together: the transpose of a consistent gather is a conservative
scatter, so a pair of edges built from one mapping and its transpose
exchanges the same work in both directions.  It is a property of the
two applications as a *pair*; neither has it alone.

Tolerances are stated in units of the rounding of the dtypes involved,
against the magnitude the mapping's own terms have (``rounding_units *
eps * gain * max|field|``, with ``gain`` the largest absolute row sum of
the operator), so they mean the same for a selection matrix and for a
kernel interpolant with large cancelling weights.

At the underflow end a tolerance stops shrinking.  The delivery flushes:
a result below the smallest normal number of its dtype (``tiny``) is
zero and an operand below it is read as zero, where the float64
reference of a check keeps both.  One flush moves one value by less than
``tiny`` *in the units of that value*, whatever the operator's gain: the
delivered value itself, after the edge's transform; each product and
partial sum of the mapping's row, which the transform then scales; each
source value as it is read, which the operator carries.  No comparison
is tighter than what those flushes can cost the values it compares, and
every comparison is exactly the rounding tolerance wherever that is the
larger of the two: at any magnitude where ``eps * gain * max|field|`` is
more than a few ``tiny``.  (The position derivative is compared with an
absolute allowance of its own, ``tiny / eps``.)

Requires ``hypothesis >= 6.165``; install the ``[verify]`` extra.
"""

from __future__ import annotations

import itertools
import json
import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from maddening.core._pow2_frame import pow2_host_factor
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.coupling.mapping import Mapping, _params_contract_problem
from maddening.core.coupling.mapping_spec import (
    MappingSpec,
    _mapping_config_dict,
    check_mapping_serialisable,
    make_point_resolver,
)
from maddening.core.edge import EdgeSpec, _delivered
from maddening.testing.strategies import _representable
from maddening.testing.verification import VerificationResult, _Refused

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    from hypothesis.errors import HypothesisException
    from hypothesis.extra.numpy import arrays
except ImportError as e:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "hypothesis is required for property-based verification. "
        "Install it with: pip install maddening[verify]"
    ) from e

__all__ = [
    "DEFAULT_MAPPING_CHECKS",
    "DEFAULT_ROUNDING_UNITS",
    "verify_mapping",
    "assert_mapping_verified",
]

#: The checks :func:`verify_mapping` runs when ``checks`` is not given, in
#: the order they are reported.  ``dtype`` expands to one result per
#: entry of ``dtypes`` (``dtype_float32``, ``dtype_float64``, ...).
DEFAULT_MAPPING_CHECKS = (
    "structure", "linearity", "consistent", "conservative", "adjoint",
    "geometry_derivative", "outside_hull", "dtype", "jit_consistent",
    "round_trip",
)

#: Default width of every comparison, in units of ``eps * gain *
#: max|field|``.  The rounding of a sum of ``k`` products is at most ``k``
#: such units and typically ``sqrt(k)``: 64 covers a row of 64 terms in the
#: worst case and of some thousands in the typical one.
DEFAULT_ROUNDING_UNITS = 64.0

#: Above this many source entries the operator's gain is estimated from
#: sign probes instead of being read off the operator applied to an
#: identity (which costs ``n_source`` channels).
_DENSE_GAIN_LIMIT = 1024

#: The sign-probe estimate of the gain is multiplied by this: a random
#: sign vector reads a row of ``k`` entries as about ``1 / sqrt(k)`` of
#: its absolute sum.
_PROBE_GAIN_SLACK = 4.0

#: A draw of the position-derivative check is on a kink when the jump of
#: the slope its stencil could hold exceeds this fraction of the slope.
_KINK_FRACTION = 1e-3  # units: fraction of the slope

#: ... and when the positions' dtype moves a stencil point by more than
#: this fraction of the step.
_UNEVEN_STEP = 1e-3  # units: fraction of the step


class _Kink(Exception):
    """A drawn position at which the kernel changes polynomial: nothing
    is compared there."""


def _eps(*dtypes: Any) -> float:
    """The coarsest rounding among the floating dtypes given."""
    out = 0.0
    for dtype in dtypes:
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.floating):
            out = max(out, float(jnp.finfo(dtype).eps))
    return out or float(np.finfo(np.float32).eps)


def _magnitude_floor(dtype: Any) -> float:
    """The smallest field magnitude a tolerance is scaled by: below
    ``tiny / eps`` the products of a field with the weights leave the
    normal range, and their rounding no longer shrinks with the field."""
    info = jnp.finfo(np.dtype(dtype))
    return float(info.tiny) / float(info.eps)


def _tiny(dtype: Any) -> float:
    """The smallest normal number of *dtype*: what one flush to zero
    costs, at most, in a value of that dtype."""
    dtype = np.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = np.dtype(np.float32)
    return float(jnp.finfo(dtype).tiny)


def _flush_cost(subject: "_Subject", gain: float, delivered: Any, field: Any, *,
                transposed: bool = False) -> float:
    """What flushes to zero can cost ONE value the edge delivers, in the
    units of that value.

    A delivered value is ``scale * sum_j w_ij x_j``, and each operation
    that computes it may flush (a result below ``tiny`` is zero, an
    operand below ``tiny`` is read as zero), which moves the value
    flushed by less than ``tiny`` of its dtype:

    * the delivered value itself, in delivered units: ``tiny``;
    * the ``n_source`` products and the ``n_source - 1`` partial sums of
      the mapping's row, which the transform then scales:
      ``|scale| * (2 n_source - 1) * tiny``;
    * each source value as it is read, in the field's dtype, which the
      operator carries: ``gain * tiny``.

    Only the last shrinks with the gain.  A floor that is the gain times
    a few ``tiny`` is therefore below one flush of the delivered value
    for an edge whose transform scales down, and below the flushes of
    the row's products for a mapping whose weights are small -- and a
    check with such a floor fails an honest edge on a draw that reaches
    the underflow end (a scalar of ``4.9e-38`` on a field of ``16``,
    through an edge that converts N to kN).

    Twice the sum is returned.  The rounding of the same value is
    allowed for separately, a tolerance is the larger of the two
    allowances and not their sum (:func:`_allowed`), and the larger of
    two is at least half of both together: it covers every flush counted
    here and, at once, a rounding of half its own allowance.

    With ``transposed`` the value is one of ``scale * apply_T(y)``: a
    column of the same operator, ``n_target`` terms, whose absolute sum
    is at most ``n_target`` times the gain.
    """
    lead = subject.target_lead if transposed else subject.source_lead
    terms = int(np.prod(lead, dtype=np.int64))
    carried = terms * gain if transposed else gain
    return 2.0 * (_tiny(delivered) * (1.0 + abs(subject.scale) * (2 * terms - 1))
                  + _tiny(field) * carried)


def _allowed(rounding: float, flushes: float) -> float:
    """The tolerance of a comparison: what rounding can cost it, and never
    less than what flushes to zero can (:func:`_flush_cost`, times the
    values compared).  Every tolerance of the battery is taken here but
    the position derivative's, which adds an absolute allowance of its
    own, so the underflow end is allowed for in one place; wherever the
    rounding is the larger -- any field of ordinary magnitude -- the
    tolerance is that value, unchanged to the last bit."""
    return max(rounding, flushes)


def _x64() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _np(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


@dataclass(frozen=True)
class _Subject:
    """What is verified: an edge, and how to call it."""

    edge: EdgeSpec
    mapping: Any
    weights: dict | None
    scale: float
    needs_geometry: bool
    source_lead: tuple[int, ...]
    target_lead: tuple[int, ...]
    label: str
    #: What the checks share: one :class:`_Gain` per sampling dtype, and
    #: the compiled delivery.
    gains: dict = field(default_factory=dict, compare=False, repr=False)

    def gain(self, dtype: Any) -> "_Gain":
        key = np.dtype(dtype)
        if key not in self.gains:
            self.gains[key] = _Gain(self, key)
        return self.gains[key]

    @property
    def mappings(self) -> dict | None:
        return None if self.weights is None else {self.edge.key: self.weights}

    def deliver(self, value: Any, geom: Any = None) -> Any:
        """What the edge hands its target: the step's own edge rule."""
        return _delivered(self.edge, value, self.mappings, geom)

    def transpose(self, value: Any, geom: Any = None) -> Any:
        if self.needs_geometry:
            return self.mapping.apply_T(value, self.weights, geom)
        return self.mapping.apply_T(value, self.weights)

    def fast(self, value: Any, geom: Any = None) -> Any:
        """:meth:`deliver` for the checks of the operator's algebra, which
        evaluate it several times a draw: compiled where the kind
        compiles (``jit_consistent`` is the check that the compiled and
        the eager delivery agree), eager otherwise."""
        if "deliver" not in self.gains:
            self.gains["deliver"] = _compiled_or_eager(self.deliver)
        return self.gains["deliver"](value, None if geom is None else jnp.asarray(geom))

    def fast_transpose(self, value: Any, geom: Any = None) -> Any:
        if "transpose" not in self.gains:
            self.gains["transpose"] = _compiled_or_eager(self.transpose)
        return self.gains["transpose"](value, None if geom is None else jnp.asarray(geom))

    def weight_dtypes(self) -> list[np.dtype]:
        tree = self.weights if self.weights is not None else self.mapping.params_pytree()
        return [np.dtype(leaf.dtype) for leaf in tree.values()]

    def what_transform(self) -> str:
        if self.edge.transform is None:
            return ""
        name = getattr(self.edge.transform, "__qualname__", repr(self.edge.transform))
        return (f"  The edge applies the transform {name} after the mapping, declared as "
                f"a scaling by scale={self.scale:g}: a transform that is not that scaling "
                f"(a clamp, an offset, another factor) is what this reports.")


def _leads(mapping: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shapes: Callable[..., Any] | None = getattr(mapping, "field_shapes", None)
    if callable(shapes):
        source, target = shapes()
        return tuple(int(n) for n in source), tuple(int(n) for n in target)
    return (int(mapping.n_source),), (int(mapping.n_target),)


def _subject(mapping: Any, transform: Any, scale: float | None, weights: dict | None) -> _Subject:
    if isinstance(mapping, EdgeSpec):
        edge = mapping
        if transform is not None:
            raise ValueError(
                "transform= is for a bare mapping; an EdgeSpec carries its own "
                "transform, and that is the one verified")
        if edge.mapping is None:
            raise ValueError(
                f"{edge!r} has no mapping: verify_mapping checks an interface mapping, "
                f"alone or on its edge")
        inner = edge.mapping
        label = f"edge {edge.key}"
    else:
        inner = mapping
        label = type(mapping).__name__
        edge = None
    missing = [name for name in ("kind", "mode", "n_source", "n_target", "params_pytree",
                                 "apply", "apply_T") if not hasattr(inner, name)]
    if missing:
        # Without these nothing can be called; the structure check says so
        # as a FAIL, and the other checks have nothing to run on.
        raise _MissingMembers(missing)
    needs_geometry = bool(getattr(inner, "needs_geometry", False))
    if edge is None:
        edge = EdgeSpec(
            "source", "target", "value", "value", transform=transform, mapping=inner,
            geometry=("target", "geometry") if needs_geometry else None)
    elif needs_geometry and edge.geometry is None:
        raise ValueError(
            f"{edge!r} carries a geometry-dependent mapping and names no geometry; "
            f"the step would call it without one")
    if scale is None:
        scale = 1.0
    elif not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale != 0):
        raise ValueError(f"scale must be a finite, non-zero number, got {scale!r}")
    if scale != 1.0 and edge.transform is None:
        raise ValueError(
            f"scale={scale!r} declares what the edge's transform multiplies by, and "
            f"there is no transform")
    source_lead, target_lead = _leads(inner)
    return _Subject(edge, inner, weights, float(scale), needs_geometry, source_lead,
                    target_lead, label)


class _MissingMembers(Exception):
    def __init__(self, names: list[str]):
        super().__init__(", ".join(names))
        self.names = names


@dataclass(frozen=True)
class _Sampling:
    """How examples are drawn: the field values and the geometry."""

    dtype: np.dtype
    bounds: tuple[float, float]
    geometry: Any
    geometry_strategy: st.SearchStrategy | None
    in_battery: bool = True

    def field(self, lead: tuple[int, ...], dtype: np.dtype | None = None) -> st.SearchStrategy:
        dtype = self.dtype if dtype is None else dtype
        # Drawn in an IEEE width Hypothesis knows; any other floating
        # dtype (bfloat16) is a cast of float32 draws.
        drawn = dtype if dtype in _IEEE else np.dtype(np.float32)
        lo, hi = _representable(self.bounds[0], self.bounds[1], drawn)
        fields = arrays(drawn, lead, elements=st.floats(
            lo, hi, width=_width(drawn), allow_nan=False, allow_infinity=False,
            allow_subnormal=False))
        return fields if drawn == dtype else fields.map(lambda a: a.astype(dtype))

    def geom(self) -> st.SearchStrategy:
        if self.geometry_strategy is None:
            return st.just(self.geometry)
        strategy = self.geometry_strategy

        @st.composite
        def drawn(draw):
            try:
                example = draw(strategy)
            except HypothesisException:  # rejection and the engine's control flow
                raise
            except Exception as exc:
                raise _Refused(
                    f"geometry_strategy raised {type(exc).__name__} while generating "
                    f"an example: {exc}") from exc
            if not (hasattr(example, "shape") and hasattr(example, "dtype")):
                raise _Refused(
                    f"geometry_strategy must yield an array of positions, got "
                    f"{type(example).__name__}")
            return example
        return drawn()


_IEEE = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))


def _width(dtype: np.dtype) -> Literal[16, 32, 64]:
    """The width Hypothesis draws floats of *dtype* in."""
    if dtype == np.float16:
        return 16
    return 32 if dtype == np.float32 else 64


def _drive(
    name: str,
    strategy: st.SearchStrategy,
    body: Callable[..., None],
    *,
    max_examples: int,
    derandomize: bool,
    notes: Callable[[], str] | None = None,
) -> VerificationResult:
    """Run ``body`` on drawn examples and package the outcome, as
    :func:`maddening.testing.verification._run` does for a node."""
    last: dict[str, Any] = {}
    count = 0

    @settings(max_examples=max_examples, deadline=None, database=None,
              derandomize=derandomize,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                                     HealthCheck.filter_too_much])
    @given(strategy)
    def prop(example):
        nonlocal count
        count += 1
        last.clear()
        last.update(example)
        body(**example)

    try:
        prop()
    except _Refused as refused:
        raise ValueError(str(refused)) from refused.__cause__
    except AssertionError as e:
        return VerificationResult(name, "FAIL", detail=str(e) or "assertion failed",
                                  counterexample=dict(last), n_examples=count)
    except HypothesisException as e:
        return VerificationResult(name, "ERROR", detail=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 - the mapping raised on some input
        return VerificationResult(
            name, "FAIL",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            counterexample=dict(last), n_examples=count)
    return VerificationResult(name, "PASS", n_examples=count,
                              detail=notes() if notes is not None else "")


def _skip(name: str, reason: str) -> VerificationResult:
    return VerificationResult(name, "SKIP", detail=reason)


# ---------------------------------------------------------------------------
# The operator's gain: the scale every tolerance is taken against
# ---------------------------------------------------------------------------


def _row_sums(delivered_identity: Any) -> float:
    out = np.abs(_np(delivered_identity).astype(np.float64))
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite")
    return float(np.max(np.sum(out, axis=-1), initial=0.0))


def _gain(subject: _Subject, dtype: np.dtype, geom: Any,
          deliver: Callable[[Any, Any], Any] | None = None) -> float:
    """The largest absolute row sum of what the edge delivers, at *geom*.

    ``|sum_j h_ij x_j| <= gain * max|x|``, and the rounding of that sum is
    a small multiple of ``eps * gain * max|x|``, so this is the magnitude
    a difference "to rounding" is measured against.  Read exactly, from
    the operator applied to an identity, where the source is small and
    the mapping takes a ``(n, C)`` field; estimated from sign probes
    otherwise.
    """
    deliver = subject.deliver if deliver is None else deliver
    tiny = float(np.finfo(np.float64).tiny)
    size = int(np.prod(subject.source_lead, dtype=np.int64))
    if size <= _DENSE_GAIN_LIMIT:
        try:
            identity = jnp.asarray(np.eye(size, dtype=dtype).reshape(subject.source_lead + (size,)))
            out = deliver(identity, geom)
            if tuple(out.shape) == subject.target_lead + (size,):
                return max(_row_sums(out), tiny)
        except Exception:  # noqa: BLE001 - a kind that takes no channel axis
            pass
    rng = np.random.default_rng(0)
    probes = [np.ones(subject.source_lead, dtype)]
    probes += [rng.choice(np.asarray([-1.0, 1.0], dtype), size=subject.source_lead)
               for _ in range(6)]
    largest = 0.0
    for probe in probes:
        out = np.abs(_np(subject.deliver(jnp.asarray(probe), geom)).astype(np.float64))
        largest = max(largest, float(np.max(out, initial=0.0)))
    return max(_PROBE_GAIN_SLACK * largest, tiny)


class _Gain:
    """:func:`_gain`, computed once for a static mapping and per geometry
    for one that reads positions (there through the compiled delivery:
    the gain only scales a tolerance)."""

    def __init__(self, subject: _Subject, dtype: np.dtype):
        self.subject, self.dtype = subject, dtype
        self._static: float | None = None

    def __call__(self, geom: Any) -> float:
        if not self.subject.needs_geometry:
            if self._static is None:
                self._static = _gain(self.subject, self.dtype, None)
            return self._static
        return _gain(self.subject, self.dtype, geom, self.subject.fast)


def _compiled_or_eager(fn: Callable[..., Any]) -> Callable[..., Any]:
    """*fn* through ``jax.jit`` for as long as that works, eagerly after
    the first failure (a kind that does not compile: ``jit_consistent``
    is the check that says so).  For the evaluations a check makes many
    of per draw."""
    state: dict[str, Any] = {"compiled": jax.jit(fn)}

    def call(*args):
        if state:
            try:
                return state["compiled"](*args)
            except Exception:  # noqa: BLE001
                state.clear()
        return fn(*args)
    return call


def _amax(*values: Any) -> float:
    return max((float(np.max(np.abs(_np(v)), initial=0.0)) for v in values), default=0.0)


def _geom_dtype(geom: Any) -> list[np.dtype]:
    return [] if geom is None else [np.dtype(geom.dtype)]


# ---------------------------------------------------------------------------
# Coordinates, polynomials and measures
# ---------------------------------------------------------------------------


def _coordinates(given_: Any, geom: Any, size: int, what: str) -> np.ndarray | None:
    """``(size, d)`` float64 coordinates from an array or a function of
    the drawn geometry, or ``None`` when none were given."""
    if given_ is None:
        return None
    value = given_(geom) if callable(given_) else given_
    value = np.asarray(_np(value), dtype=np.float64)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != size:
        raise _Refused(
            f"{what} must give one coordinate row per entry, shape ({size},) or "
            f"({size}, d); got {value.shape}")
    return value


def _monomials(d: int, degree: int) -> list[tuple[int, ...]]:
    """Exponent tuples of every monomial in ``d`` coordinates of total
    degree 0 to *degree*, constants first."""
    out = [p for p in itertools.product(range(degree + 1), repeat=d) if sum(p) <= degree]
    return sorted(out, key=lambda p: (sum(p), p))


def _monomial(coords: np.ndarray, powers: tuple[int, ...]) -> np.ndarray:
    out = np.ones(coords.shape[0], np.float64)
    for a, k in enumerate(powers):
        out = out * coords[:, a] ** k
    return out


def _measure(given_: Any, size: int, what: str) -> np.ndarray:
    if given_ is None:
        return np.ones(size, np.float64)
    value = np.asarray(_np(given_), dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"{what} must hold one weight per entry, {size} in all; got "
                         f"shape {np.shape(given_)}")
    return value


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def _mapping_structure(subject: _Subject, sampling: _Sampling, channels: int | None,
                       rounding_units: float, **kw) -> VerificationResult:
    """Members, the parameter contract, shapes, and nothing changed by a call."""
    name = "structure"
    mapping = subject.mapping
    if not isinstance(mapping, Mapping):
        return VerificationResult(name, "FAIL", detail=(
            f"{type(mapping).__name__} does not have the members of the Mapping "
            f"protocol (kind, mode, n_source, n_target, params_pytree, apply, apply_T)"))
    if not isinstance(mapping.kind, str) or not mapping.kind:
        return VerificationResult(name, "FAIL", detail=(
            f"kind must be a non-empty string, got {mapping.kind!r}"))
    for size_name in ("n_source", "n_target"):
        size = getattr(mapping, size_name)
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            return VerificationResult(name, "FAIL", detail=(
                f"{size_name} must be a positive integer, got {size!r}"))
    try:
        problem = _params_contract_problem(mapping)
    except Exception as e:  # noqa: BLE001
        problem = f"params_pytree() raised {type(e).__name__}: {e}"
    if problem is not None:
        return VerificationResult(name, "FAIL", detail=(
            f"{problem}.  GraphManager.add_edge refuses this mapping for the same reason."))
    gain = subject.gain(sampling.dtype)

    def same(before: Any, after: Any, what: str) -> None:
        assert before.shape == after.shape and before.dtype == after.dtype and \
            np.array_equal(before, after, equal_nan=True), (
                f"a call changed {what}: a mapping is a pure function of the field, "
                f"the weights and the geometry, and must leave all three as they were")

    def body(x, y, geom):
        tree_before = {k: _np(v).copy() for k, v in mapping.params_pytree().items()}
        given_weights = None if subject.weights is None else {
            k: _np(v).copy() for k, v in subject.weights.items()}
        x_host, y_host = np.array(x), np.array(y)
        geom_host = None if geom is None else np.array(geom)
        # NumPy arrays in, so that an in-place edit of an argument shows.
        out = subject.deliver(x_host, geom_host)
        back = subject.transpose(y_host, geom_host)
        same(x, x_host, "the field it was given")
        same(y, y_host, "the field apply_T was given")
        if geom is not None:
            same(np.asarray(geom), geom_host, "the geometry it was given")
        if subject.weights is not None and given_weights is not None:
            assert list(subject.weights) == list(given_weights), "a call changed the weights' keys"
            for key, before in given_weights.items():
                same(before, _np(subject.weights[key]), f"the weight {key!r}")
        tree_after = mapping.params_pytree()
        assert list(tree_after) == list(tree_before), "a call changed params_pytree()'s keys"
        for key, before in tree_before.items():
            same(before, _np(tree_after[key]), f"params_pytree()[{key!r}]")
        assert tuple(out.shape) == subject.target_lead, (
            f"the delivered field has shape {tuple(out.shape)} for a source field of "
            f"shape {subject.source_lead}; the mapping declares {subject.target_lead}")
        assert tuple(back.shape) == subject.source_lead, (
            f"apply_T returned shape {tuple(back.shape)} for a target field of shape "
            f"{subject.target_lead}; the mapping declares {subject.source_lead}")
        # Nothing carried between calls: the same call again is the same
        # result (to rounding: a scatter-add on an accelerator may differ
        # in its last bits between runs).
        again = subject.deliver(jnp.asarray(x), geom)
        g = gain(geom)
        tol = _allowed(
            rounding_units * _eps(out.dtype) * g * max(
                _amax(x), _magnitude_floor(sampling.dtype)),
            2.0 * _flush_cost(subject, g, out.dtype, sampling.dtype))
        assert float(np.max(np.abs(_np(again) - _np(out)), initial=0.0)) <= tol, (
            "two calls on the same field, weights and geometry gave different results: "
            "the mapping keeps something between calls")
        if channels:
            wide = jnp.stack([jnp.asarray(x)] * channels, axis=-1)
            out_c = subject.deliver(wide, geom)
            assert tuple(out_c.shape) == subject.target_lead + (channels,), (
                f"a source field with {channels} channels, shape "
                f"{subject.source_lead + (channels,)}, was delivered with shape "
                f"{tuple(out_c.shape)}; expected {subject.target_lead + (channels,)} "
                f"(pass channels=None for a kind that documents scalar fields only)")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "y": sampling.field(subject.target_lead),
        "geom": sampling.geom()})
    return _drive(name, strategy, body, **kw)


def _mapping_linearity(subject: _Subject, sampling: _Sampling, rounding_units: float,
                       **kw) -> VerificationResult:
    """``delivered(a x + b y) == a delivered(x) + b delivered(y)``."""
    gain = subject.gain(sampling.dtype)

    def body(x, y, a, b, geom):
        xj, yj = jnp.asarray(x), jnp.asarray(y)
        a_t, b_t = xj.dtype.type(a), xj.dtype.type(b)
        left = _np(subject.fast(a_t * xj + b_t * yj, geom)).astype(np.float64)
        dx, dy = subject.fast(xj, geom), subject.fast(yj, geom)
        right = float(a_t) * _np(dx).astype(np.float64) + float(b_t) * _np(dy).astype(np.float64)
        # The combination a x + b y is formed in the field's dtype.
        eps = _eps(dx.dtype, xj.dtype, *_geom_dtype(geom))
        size = abs(float(a_t)) * _amax(x) + abs(float(b_t)) * _amax(y)
        g = gain(geom)
        # Three deliveries are compared, two of them times a scalar.  The
        # combination is formed in arithmetic that flushes too: its two
        # products and the field values it reads cost what the three
        # deliveries are allowed for reading theirs; a scalar below tiny
        # costs more, since it is read as zero and takes its whole term
        # with it (such a draw says nothing about that term).
        weight = abs(float(a_t)) + abs(float(b_t))
        unread = sum(abs(float(c)) * _amax(v) for c, v in ((a_t, x), (b_t, y))
                     if abs(float(c)) < _tiny(xj.dtype))
        flushes = (1.0 + weight) * _flush_cost(subject, g, dx.dtype, xj.dtype) \
            + 2.0 * g * unread
        tol = _allowed(
            rounding_units * eps * g * max(size, _magnitude_floor(dx.dtype)), flushes)
        gap = float(np.max(np.abs(left - right), initial=0.0))
        assert gap <= tol, (
            f"what the edge delivers is not linear in the value: delivered(a x + b y) "
            f"differs from a delivered(x) + b delivered(y) by {gap:.3g}, "
            f"{gap / tol * rounding_units:.3g} rounding units where {rounding_units:g} "
            f"are allowed.{subject.what_transform()}")

    scalars = st.floats(-4.0, 4.0, width=32, allow_nan=False)
    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "y": sampling.field(subject.source_lead),
        "a": scalars, "b": scalars, "geom": sampling.geom()})
    return _drive("linearity", strategy, body, **kw)


def _mapping_consistent(subject: _Subject, sampling: _Sampling, rounding_units: float,
                        polynomial_order: int, source_coordinates: Any,
                        target_coordinates: Any, **kw) -> VerificationResult:
    """Constants, and polynomials up to the claimed degree, are reproduced."""
    gain = subject.gain(sampling.dtype)
    n_s = int(np.prod(subject.source_lead, dtype=np.int64))
    n_t = int(np.prod(subject.target_lead, dtype=np.int64))
    dtype = sampling.dtype

    def body(c, geom):
        g = gain(geom)
        eps = _eps(dtype, *subject.weight_dtypes(), *_geom_dtype(geom))
        constant = jnp.full(subject.source_lead, dtype.type(c), dtype)
        delivered = subject.fast(constant, geom)
        out = _np(delivered).astype(np.float64)
        expected = subject.scale * float(dtype.type(c))
        flushes = _flush_cost(subject, g, delivered.dtype, dtype)
        tol = _allowed(
            rounding_units * eps * g * max(abs(float(c)), _magnitude_floor(dtype)), flushes)
        gap = float(np.max(np.abs(out - expected), initial=0.0))
        assert gap <= tol, (
            f"a constant field {float(c):.6g} is not reproduced: the delivered values "
            f"differ from {expected:.6g} by up to {gap:.3g}, "
            f"{gap / tol * rounding_units:.3g} rounding units where {rounding_units:g} "
            f"are allowed.  A consistent transfer hands over a value field unchanged "
            f"where it is uniform.{subject.what_transform()}")
        if polynomial_order < 1:
            return
        xs = _coordinates(source_coordinates, geom, n_s, "source_coordinates")
        xt = _coordinates(target_coordinates, geom, n_t, "target_coordinates")
        assert xs is not None and xt is not None
        for powers in _monomials(xs.shape[1], polynomial_order)[1:]:
            f_s, f_t = _monomial(xs, powers), _monomial(xt, powers)
            field_ = jnp.asarray(f_s.astype(dtype).reshape(subject.source_lead))
            out = _np(subject.fast(field_, geom)).astype(np.float64).reshape(-1)
            size = max(float(np.max(np.abs(f_s))), float(np.max(np.abs(f_t))),
                       _magnitude_floor(dtype))
            # The coordinates are cast to the field's dtype before the
            # transfer, so the reference carries that rounding too.
            tol = _allowed(rounding_units * eps * max(g, abs(subject.scale)) * size, flushes)
            gap = float(np.max(np.abs(out - subject.scale * f_t), initial=0.0))
            assert gap <= tol, (
                f"the monomial with exponents {powers} of the coordinates is not "
                f"reproduced: the delivered values differ from it at the target "
                f"coordinates by up to {gap:.3g}, {gap / tol * rounding_units:.3g} "
                f"rounding units where {rounding_units:g} are allowed.  The mapping was "
                f"claimed to reproduce polynomials up to degree {polynomial_order}."
                f"{subject.what_transform()}")

    lo, hi = _representable(sampling.bounds[0], sampling.bounds[1], dtype)
    strategy = st.fixed_dictionaries({
        "c": st.floats(lo, hi, width=_width(dtype), allow_nan=False),
        "geom": sampling.geom()})
    return _drive("consistent", strategy, body, **kw)


def _mapping_conservative(subject: _Subject, sampling: _Sampling, rounding_units: float,
                          polynomial_order: int, source_coordinates: Any,
                          target_coordinates: Any, source_measure: Any, target_measure: Any,
                          **kw) -> VerificationResult:
    """The total, and the moments up to the claimed degree, are preserved."""
    gain = subject.gain(sampling.dtype)
    n_s = int(np.prod(subject.source_lead, dtype=np.int64))
    n_t = int(np.prod(subject.target_lead, dtype=np.int64))
    m_s = _measure(source_measure, n_s, "source_measure")
    m_t = _measure(target_measure, n_t, "target_measure")

    def body(x, geom):
        out = subject.fast(jnp.asarray(x), geom)
        eps = _eps(out.dtype, *subject.weight_dtypes(), *_geom_dtype(geom))
        g = gain(geom)
        flush = _flush_cost(subject, g, out.dtype, np.asarray(x).dtype)
        sent = np.asarray(x, np.float64).reshape(-1)
        received = _np(out).astype(np.float64).reshape(-1)
        xs = xt = None
        if polynomial_order >= 1:
            xs = _coordinates(source_coordinates, geom, n_s, "source_coordinates")
            xt = _coordinates(target_coordinates, geom, n_t, "target_coordinates")
            assert xs is not None and xt is not None
        d = 1 if xs is None else xs.shape[1]
        for powers in _monomials(d, polynomial_order):
            f_s = np.ones(n_s) if xs is None else _monomial(xs, powers)
            f_t = np.ones(n_t) if xt is None else _monomial(xt, powers)
            total_in = float(np.sum(m_s * f_s * sent))
            total_out = float(np.sum(m_t * f_t * received))
            size = float(np.sum(np.abs(m_t * f_t))) * g * max(
                _amax(x), _magnitude_floor(out.dtype))
            size = max(size, abs(subject.scale) * float(np.sum(np.abs(m_s * f_s * sent))))
            # Every received value enters the total with its weight.
            tol = _allowed(rounding_units * eps * size,
                           float(np.sum(np.abs(m_t * f_t))) * flush)
            gap = abs(total_out - subject.scale * total_in)
            what = "the total" if not any(powers) else (
                f"the moment with exponents {powers} of the coordinates")
            assert gap <= tol, (
                f"{what} is not preserved: {total_in:.9g} was sent"
                + (f" (times scale={subject.scale:g})" if subject.scale != 1.0 else "")
                + f" and {total_out:.9g} arrived, a difference of {gap:.3g}, "
                f"{gap / tol * rounding_units:.3g} rounding units where "
                f"{rounding_units:g} are allowed.  A conservative transfer creates and "
                f"destroys nothing.{subject.what_transform()}")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "geom": sampling.geom()})
    return _drive("conservative", strategy, body, **kw)


def _mapping_adjoint(subject: _Subject, sampling: _Sampling, rounding_units: float,
                     **kw) -> VerificationResult:
    """``<delivered(x), y> == scale * <x, apply_T(y)>``."""
    gain = subject.gain(sampling.dtype)

    def body(x, y, geom):
        xj, yj = jnp.asarray(x), jnp.asarray(y)
        out = subject.fast(xj, geom)
        back = subject.fast_transpose(yj, geom)
        # Both sides are products of two fields, and the reference is
        # float64: the product of two float64 fields near either end of
        # the range leaves it (an infinite or a zero inner product, and a
        # tolerance that is one or the other).  So x, and what was
        # delivered of it, are framed by one exact power of two, y and its
        # transpose by another: the comparison is the unframed one to the
        # last bit wherever that was in range, and stays in range.
        p = pow2_host_factor(_amax(x), np.float64)
        q = pow2_host_factor(_amax(y), np.float64)
        x_framed, y_framed = np.asarray(x, np.float64) * p, np.asarray(y, np.float64) * q
        left = float(np.sum((_np(out).astype(np.float64) * p) * y_framed))
        right = subject.scale * float(np.sum(x_framed * (_np(back).astype(np.float64) * q)))
        eps = _eps(out.dtype, back.dtype, *_geom_dtype(geom))
        floor = _magnitude_floor(out.dtype)
        g = gain(geom)
        x_sum, y_sum = float(np.sum(np.abs(x_framed))), float(np.sum(np.abs(y_framed)))
        size = g * max(_amax(x) * p, floor * p) * max(y_sum, floor * q)
        # Every delivered value enters the left side times its y, every
        # transposed value the right side times its x.
        flushes = y_sum * (_flush_cost(subject, g, out.dtype, xj.dtype) * p) + x_sum * (
            _flush_cost(subject, g, back.dtype, yj.dtype, transposed=True) * q)
        tol = _allowed(rounding_units * eps * size, flushes)
        gap = abs(left - right)
        assert gap <= tol, (
            f"apply_T is not the adjoint of what the edge delivers: <delivered(x), y> = "
            f"{left / p / q:.9g} and "
            + (f"scale * <x, apply_T(y)> = " if subject.scale != 1.0 else "<x, apply_T(y)> = ")
            + f"{right / p / q:.9g} differ by {gap / p / q:.3g}, "
            f"{gap / tol * rounding_units:.3g} "
            f"rounding units where {rounding_units:g} are allowed.  A pair of edges "
            f"built on apply and apply_T then exchanges different amounts of work in "
            f"the two directions.{subject.what_transform()}")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "y": sampling.field(subject.target_lead),
        "geom": sampling.geom()})
    return _drive("adjoint", strategy, body, **kw)


def _mapping_geometry_derivative(subject: _Subject, sampling: _Sampling,
                                 rounding_units: float, geometry_step: float | None,
                                 **kw) -> VerificationResult:
    """The derivative with respect to a position against a central
    difference, at draws that are not on a kink of the kernel.

    One coordinate of one point is moved per draw.  The functional is
    ``phi(g) = sum(r * delivered(x, g))`` for a drawn ``r``, evaluated on
    a five-point stencil: the position, and a half step and a whole step
    either side.  Where the stencil's two third differences vanish (to
    rounding, and to a small fraction of the slope), ``phi`` is one
    smooth piece over the stencil and the central difference at the half
    step is its derivative.  Anywhere else the kernel changes polynomial
    inside the stencil -- the draw is *on a kink* -- and the draw is
    counted and not compared, since a finite difference across a kink is
    not a derivative.  Any jump of the slope too small to be detected
    that way is added to the tolerance of the comparison.
    """
    name = "geometry_derivative"
    gain = subject.gain(sampling.dtype)
    counts = {"compared": 0, "kink": 0, "unresolved": 0}

    def eager(xj, rj, g):
        return jnp.sum(rj * subject.deliver(xj, g))

    # A draw costs six evaluations and a gradient.
    functional = _compiled_or_eager(eager)
    gradient = _compiled_or_eager(jax.grad(eager, argnums=2))

    def body(x, r, geom, where):
        geom_j = jnp.asarray(geom)
        if not np.issubdtype(np.dtype(geom_j.dtype), np.floating):
            raise _Refused(f"the positions must be floating-point, got {geom_j.dtype}")
        xj, rj = jnp.asarray(x), jnp.asarray(r)

        def phi(g):
            return functional(xj, rj, g)

        flat = int(where * geom_j.size) % max(int(geom_j.size), 1)
        index = np.unravel_index(flat, geom_j.shape)
        eps = _eps(geom_j.dtype, xj.dtype)
        base = _np(geom_j).astype(np.float64)
        extent = max(float(np.max(base) - np.min(base)), float(np.max(np.abs(base))),
                     _magnitude_floor(geom_j.dtype))
        h = float(geometry_step) if geometry_step is not None else eps ** (1.0 / 3.0) * extent
        direction = np.zeros(geom_j.shape, np.float64)
        direction[index] = 1.0

        def at(step: float) -> tuple[float, float]:
            moved = jnp.asarray((base + step * direction).astype(geom_j.dtype))
            # With the step actually taken, after the cast to the positions' dtype.
            return float(phi(moved)), float(_np(moved)[index]) - float(base[index])

        (f_2, t_2), (f_1, t_1) = at(h), at(h / 2)
        (f_m1, t_m1), (f_m2, t_m2) = at(-h / 2), at(-h)
        f_0 = float(phi(geom_j))
        uneven = max(abs(t_2 - h), abs(t_1 - h / 2), abs(t_m1 + h / 2), abs(t_m2 + h))
        if uneven > _UNEVEN_STEP * h:
            # The positions' dtype cannot hold this stencil at this point.
            counts["unresolved"] += 1
            raise _Kink
        size = gain(geom) * max(_amax(x), _magnitude_floor(xj.dtype)) * float(
            np.sum(np.abs(np.asarray(r, np.float64))))
        noise = rounding_units * eps * size
        central, central_half = (f_2 - f_m2) / (2 * h), (f_1 - f_m1) / h
        derivative = float(_np(gradient(xj, rj, geom_j))[index])
        # Third differences over the two four-point halves of the stencil.
        # On one smooth piece each is (h/2)**3 times the third derivative:
        # nothing for a kernel that is a polynomial of low degree in a
        # coordinate.  A jump J of the slope anywhere inside the stencil
        # leaves at least J * h / 6 in one of them, so 6 / h times the
        # larger bounds any jump the stencil could hold.
        third = max(abs(-f_m2 + 3 * f_m1 - 3 * f_0 + f_1), abs(-f_m1 + 3 * f_0 - 3 * f_1 + f_2))
        jump = 6.0 * third / h
        slope = max(abs(central_half), abs(derivative))
        if jump > 6.0 * noise / h + _KINK_FRACTION * slope:
            counts["kink"] += 1
            raise _Kink
        # The derivative arrives in the positions' dtype: below that dtype's
        # range (a float64 field of 1e-150 read at float32 positions) it is
        # zero, whatever the kernel's slope.  This allowance is absolute --
        # multiplied by neither the gain nor r -- so it is also what covers
        # the flushes of the functional's evaluations, which the noise
        # above (a floor times the gain) does not for an edge that scales
        # down; a stencil those flushes dominate is counted as a kink.
        underflow = max(_magnitude_floor(geom_j.dtype), _magnitude_floor(xj.dtype))
        tol = 4.0 * noise / h + 2.0 * abs(central - central_half) + jump + underflow
        gap = abs(derivative - central_half)
        counts["compared"] += 1
        assert gap <= tol, (
            f"the derivative with respect to position {tuple(int(i) for i in index)} "
            f"is {derivative:.9g} by automatic differentiation and "
            f"{central_half:.9g} by a central difference of step {h / 2:.3g} "
            f"(a difference of {gap:.3g} where {tol:.3g} is allowed; the draw is not "
            f"on a kink: the differences over the stencil close).  A gradient through "
            f"this edge with respect to the positions is wrong: look for a "
            f"stop_gradient, an integer cast or a host-side computation on the "
            f"positions.")

    def guarded(**example):
        try:
            body(**example)
        except _Kink:
            pass

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "r": sampling.field(subject.target_lead),
        "geom": sampling.geom(), "where": st.floats(0.0, 1.0, exclude_max=True)})
    def notes() -> str:
        text = (f"{counts['compared']} draws compared; {counts['kink']} were on a kink "
                f"of the kernel and were not")
        if counts["unresolved"]:
            text += (f"; at {counts['unresolved']} the positions' dtype cannot hold a "
                     f"stencil of this step")
        return text

    result = _drive(name, strategy, guarded, notes=notes, **kw)
    if result.status == "PASS" and counts["compared"] == 0:
        return _skip(name, (
            f"no draw could be compared ({notes()}).  Pass geometry_step= (a step well "
            f"inside one cell of the kernel, and well above the positions' rounding) or "
            f"positions away from the kinks."))
    return result


def _mapping_outside_hull(subject: _Subject, sampling: _Sampling, rounding_units: float,
                          hull: Any, **kw) -> VerificationResult:
    """A position outside the hull is its projection onto the hull
    (``outside="clamp"``), and the derivative with respect to a
    coordinate strictly outside is zero."""
    lower = np.atleast_1d(np.asarray(hull[0], np.float64))
    upper = np.atleast_1d(np.asarray(hull[1], np.float64))
    gain = subject.gain(sampling.dtype)
    gradient = _compiled_or_eager(
        jax.grad(lambda xj, g: jnp.sum(subject.deliver(xj, g)), argnums=1))

    def body(x, geom, push, far):
        geom_np = _np(geom)
        cols = geom_np if geom_np.ndim == 2 else geom_np[:, None]
        if cols.shape[1] != lower.size or lower.shape != upper.shape:
            raise _Refused(
                f"hull=(lower, upper) must give one bound per axis of the positions "
                f"({cols.shape[1]}); got {lower.shape} and {upper.shape}")
        span = np.maximum(upper - lower, 1e-300)
        choice = np.resize(np.asarray(push), cols.shape)
        moved = cols.astype(np.float64)
        moved = np.where(choice == 1, upper + far * span, moved)
        moved = np.where(choice == 2, lower - far * span, moved)
        moved = np.where(choice == 3, upper, moved)   # exactly on a face
        moved = np.where(choice == 4, lower, moved)
        outside = jnp.asarray(moved.astype(geom_np.dtype).reshape(geom_np.shape))
        clamped = jnp.asarray(np.clip(_np(outside).reshape(cols.shape), lower, upper)
                              .astype(geom_np.dtype).reshape(geom_np.shape))
        xj = jnp.asarray(x)
        out, expected = subject.fast(xj, outside), subject.fast(xj, clamped)
        assert bool(np.all(np.isfinite(_np(out)))), (
            "the delivered field is not finite for finite positions at or outside the hull")
        g = gain(clamped)
        tol = _allowed(
            rounding_units * _eps(out.dtype, geom_np.dtype) * g * max(
                _amax(x), _magnitude_floor(out.dtype)),
            2.0 * _flush_cost(subject, g, out.dtype, xj.dtype))
        gap = float(np.max(np.abs(_np(out).astype(np.float64)
                                  - _np(expected).astype(np.float64)), initial=0.0))
        assert gap <= tol, (
            f"positions outside the hull do not deliver what their projections onto "
            f"the hull deliver (a difference of {gap:.3g} where {tol:.3g} is allowed); "
            f"the kind was declared to clamp")
        grad = _np(gradient(xj, outside))
        strictly = (np.asarray(_np(outside)).reshape(cols.shape) > upper) | (
            np.asarray(_np(outside)).reshape(cols.shape) < lower)
        leak = float(np.max(np.abs(grad.reshape(cols.shape)[strictly]), initial=0.0))
        assert leak == 0.0, (
            f"the derivative with respect to a coordinate strictly outside the hull is "
            f"{leak:.3g}, not zero: a clamped position does not move the result")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "geom": sampling.geom(),
        "push": st.lists(st.integers(0, 4), min_size=1, max_size=16),
        "far": st.floats(1e-3, 8.0)})
    return _drive("outside_hull", strategy, body, **kw)


def _mapping_dtype(subject: _Subject, sampling: _Sampling, rounding_units: float,
                   dtype: np.dtype, **kw) -> VerificationResult:
    """A field of *dtype* is delivered in the dtype the field and the
    weights promote to, finite, and equal to the widest evaluation."""
    name = f"dtype_{dtype.name}"
    if dtype.itemsize == 8 and not _x64():
        return _skip(name, "float64 needs jax_enable_x64, which is off in this process")
    wide = np.dtype(np.float64 if _x64() else np.float32)
    gain = subject.gain(wide)

    def body(x, geom):
        xj = jnp.asarray(x)
        out = subject.deliver(xj, geom)
        expected = np.dtype(jnp.result_type(xj.dtype, *subject.weight_dtypes()))
        assert np.dtype(out.dtype) == expected, (
            f"a {dtype.name} field is delivered as {out.dtype}; the field and the "
            f"weights promote to {expected}.  The target node would be handed a "
            f"boundary input of another dtype than its state, which a scan refuses "
            f"or, worse, promotes.{subject.what_transform()}")
        assert bool(np.all(np.isfinite(_np(out)))), (
            f"the delivered field is not finite for a finite {dtype.name} field")
        if wide.itemsize <= dtype.itemsize:
            return
        reference = _np(subject.deliver(jnp.asarray(np.asarray(x, wide)), geom))
        g = gain(geom)
        tol = _allowed(
            rounding_units * _eps(dtype, *subject.weight_dtypes(), *_geom_dtype(geom)) * (
                g * max(_amax(x), _magnitude_floor(dtype))),
            _flush_cost(subject, g, out.dtype, dtype))
        gap = float(np.max(np.abs(_np(out).astype(np.float64)
                                  - reference.astype(np.float64)), initial=0.0))
        assert gap <= tol, (
            f"the {dtype.name} evaluation differs from the {wide.name} one by {gap:.3g}, "
            f"{gap / tol * rounding_units:.3g} rounding units of {dtype.name} where "
            f"{rounding_units:g} are allowed: precision is lost inside the mapping")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead, dtype), "geom": sampling.geom()})
    return _drive(name, strategy, body, **kw)


def _mapping_jit(subject: _Subject, sampling: _Sampling, rounding_units: float,
                 **kw) -> VerificationResult:
    """The compiled delivery agrees with the eager one, to rounding."""
    gain = subject.gain(sampling.dtype)
    if subject.needs_geometry:
        compiled = jax.jit(lambda value, geom: subject.deliver(value, geom))
    else:
        compiled = jax.jit(lambda value, geom: subject.deliver(value))

    def body(x, geom):
        xj = jnp.asarray(x)
        eager = subject.deliver(xj, geom)
        jitted = compiled(xj, None if geom is None else jnp.asarray(geom))
        assert eager.dtype == jitted.dtype and eager.shape == jitted.shape, (
            f"jit changes the result's type: {eager.dtype}{tuple(eager.shape)} eagerly, "
            f"{jitted.dtype}{tuple(jitted.shape)} compiled")
        g = gain(geom)
        tol = _allowed(
            rounding_units * _eps(eager.dtype, *_geom_dtype(geom)) * g * max(
                _amax(x), _magnitude_floor(eager.dtype)),
            2.0 * _flush_cost(subject, g, eager.dtype, xj.dtype))
        gap = float(np.max(np.abs(_np(eager).astype(np.float64)
                                  - _np(jitted).astype(np.float64)), initial=0.0))
        assert gap <= tol, (
            f"the compiled delivery differs from the eager one by {gap:.3g}, "
            f"{gap / tol * rounding_units:.3g} rounding units where {rounding_units:g} "
            f"are allowed: usually Python branching on values, or a host-side "
            f"computation that tracing freezes")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "geom": sampling.geom()})
    return _drive("jit_consistent", strategy, body, **kw)


def _mapping_round_trip(subject: _Subject, sampling: _Sampling, resolve_points: Any,
                        base_dir: Any, **kw) -> VerificationResult:
    """Saved as a config writes it and rebuilt as a config reads it, the
    mapping is the same operator, bit for bit."""
    name = "round_trip"
    mapping = subject.mapping
    try:
        check_mapping_serialisable(mapping, edge_key=subject.edge.key)
    except ValueError as e:
        return _skip(name, (
            f"not serialisable, so there is no save/load route to check: {e}  (A kind "
            f"registered with register_mapping, whose factory attaches a complete "
            f"MappingSpec, has one.)"))
    spec = mapping.spec
    needs_graph = [n for n, ref in spec.points.items() if isinstance(ref, dict) and "node" in ref]
    if needs_graph and resolve_points is None:
        return _skip(name, (
            f"the mapping's arrays {needs_graph} are references to node fields, which "
            f"only a graph resolves: pass resolve_points=gm.point_resolver(), or check "
            f"the route with GraphManager.to_dict() / from_dict()"))
    resolver = resolve_points if resolve_points is not None else make_point_resolver(
        None, base_dir)
    try:
        stored = json.loads(json.dumps(_mapping_config_dict(mapping)))
        rebuilt = MappingSpec.from_dict(stored).build(resolver)
    except Exception as e:  # noqa: BLE001
        return VerificationResult(name, "FAIL", detail=(
            f"the saved mapping could not be rebuilt: {type(e).__name__}: {e}"))
    problems = []
    for attribute in ("kind", "mode", "n_source", "n_target"):
        if getattr(rebuilt, attribute) != getattr(mapping, attribute):
            problems.append(f"{attribute} is {getattr(rebuilt, attribute)!r}, was "
                            f"{getattr(mapping, attribute)!r}")
    if bool(getattr(rebuilt, "needs_geometry", False)) != subject.needs_geometry:
        problems.append("needs_geometry changed")
    before, after = mapping.params_pytree(), rebuilt.params_pytree()
    if list(before) != list(after):
        problems.append(f"params_pytree() has the keys {list(after)}, had {list(before)}")
    else:
        for key, leaf in before.items():
            new = after[key]
            if new.shape != leaf.shape or new.dtype != leaf.dtype or not np.array_equal(
                    _np(new), _np(leaf)):
                problems.append(f"the weight {key!r} is not the saved mapping's, bit for bit")
    if problems:
        return VerificationResult(name, "FAIL", detail=(
            "the mapping rebuilt from its saved form differs: " + "; ".join(problems)
            + ".  The factory does not record everything it was given in the "
              "MappingSpec it attaches."))
    rebuilt_subject = _Subject(
        EdgeSpec(subject.edge.source_node, subject.edge.target_node,
                 subject.edge.source_field, subject.edge.target_field,
                 transform=subject.edge.transform, mapping=rebuilt,
                 geometry=subject.edge.geometry),
        rebuilt, None, subject.scale, subject.needs_geometry, subject.source_lead,
        subject.target_lead, subject.label)
    own = _Subject(subject.edge, mapping, None, subject.scale, subject.needs_geometry,
                   subject.source_lead, subject.target_lead, subject.label)

    def body(x, geom):
        xj = jnp.asarray(x)
        a, b = _np(own.deliver(xj, geom)), _np(rebuilt_subject.deliver(xj, geom))
        assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), (
            "the mapping rebuilt from its saved form delivers a different field: the "
            "factory does not record everything it was given in the MappingSpec it "
            "attaches")

    strategy = st.fixed_dictionaries({
        "x": sampling.field(subject.source_lead), "geom": sampling.geom()})
    return _drive(name, strategy, body, **kw)


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EXPERIMENTAL)
def verify_mapping(
    mapping: Any,
    *,
    checks: Sequence[str] | None = None,
    consistent: bool | None = None,
    conservative: bool | None = None,
    polynomial_order: int = 0,
    source_coordinates: Any = None,
    target_coordinates: Any = None,
    source_measure: Any = None,
    target_measure: Any = None,
    transform: Callable[[Any], Any] | None = None,
    scale: float | None = None,
    weights: dict | None = None,
    geometry: Any = None,
    geometry_strategy: st.SearchStrategy | None = None,
    geometry_step: float | None = None,
    hull: tuple[Any, Any] | None = None,
    outside: str | None = None,
    bounds: tuple[float, float] = (-100.0, 100.0),
    dtype: npt.DTypeLike = np.float32,
    dtypes: Sequence[npt.DTypeLike] = (np.float32, np.float64),
    channels: int | None = 2,
    rounding_units: float = DEFAULT_ROUNDING_UNITS,
    resolve_points: Callable[[dict], Any] | None = None,
    base_dir: Any = None,
    max_examples: int = 100,
    derandomize: bool = False,
) -> dict[str, VerificationResult]:
    """Run a battery of checks on an interface mapping, or on an edge.

    The edge counterpart of
    :func:`~maddening.testing.verification.verify_node`, returning the
    same :class:`~maddening.testing.verification.VerificationResult` per
    check.  **Experimental.**

    Parameters
    ----------
    mapping : Mapping or EdgeSpec
        Any object with the members of the
        :class:`~maddening.core.coupling.mapping.Mapping` protocol,
        registered or not; or an :class:`~maddening.core.edge.EdgeSpec`
        that carries one, in which case every check is made on what the
        edge delivers -- the mapping and then the edge's transform,
        through the function the step delivers an edge through.
    checks : sequence of str, optional
        Subset of :data:`DEFAULT_MAPPING_CHECKS`.
    consistent, conservative : bool, optional
        What the mapping claims.  ``None`` takes the claim from
        ``mapping.mode`` (``"consistent"`` claims the first,
        ``"conservative"`` the second).  A property that is not claimed
        is not checked and its result is a ``SKIP`` saying so.  Where
        ``mode`` is only a label (a matrix of your own), state both.
    polynomial_order : int
        The degree the claims hold to: polynomials of the coordinates up
        to this degree are reproduced (``consistent``), moments up to it
        preserved (``conservative``).  0, the default, is constants and
        the total.  A degree above 0 needs both sets of coordinates.
    source_coordinates, target_coordinates : array or callable, optional
        Where the source and the target entries are, ``(n,)`` or
        ``(n, d)``, in the mapping's flat order; or ``fn(geometry)``
        returning that, for the side whose positions are the geometry.
    source_measure, target_measure : array, optional
        The weight of each entry in a total (cell sizes, quadrature
        weights), for a field of densities.  Plain sums when omitted.
    transform : callable, optional
        The edge's transform, for a bare mapping.  Not with an
        ``EdgeSpec``, which carries its own.
    scale : float, optional
        The factor the transform multiplies by (a unit conversion, a
        sign): constants are then delivered times ``scale`` and the total
        is preserved times ``scale``.  Without it a transform is held to
        delivering the mapping's values unchanged, so one that is not
        linear (a clamp), or scales by an undeclared factor, fails
        ``linearity`` / ``consistent`` / ``conservative`` / ``adjoint``
        with the reason.
    weights : dict, optional
        The entry of ``params["mappings"]`` to verify with, instead of
        the mapping's own ``params_pytree()``.
    geometry : array, optional
        Sample positions for a geometry-dependent kind, used as given
        for every example.
    geometry_strategy : hypothesis.strategies.SearchStrategy, optional
        A strategy that yields position arrays (the vocabulary of
        ``verify_node``'s ``state_strategy``).  Not with ``geometry``.
        Keep the positions where the kind's claims hold (a multilinear
        gather reproduces linear fields inside its hull).
    geometry_step : float, optional
        The step of the central difference the position derivative is
        compared with, in the positions' units: well inside one cell of
        the kernel.  Default ``eps**(1/3)`` of the positions' extent.
    hull : (lower, upper), optional
        The box the kind's kernel covers, one bound per axis; with
        ``outside`` it enables the ``outside_hull`` check.
    outside : {"clamp"}, optional
        What the kind documents for a position outside ``hull``:
        ``"clamp"`` is "its projection onto the hull".  Another
        behaviour is yours to test.
    bounds : (float, float)
        Envelope of the sampled field values.
    dtype : numpy dtype
        Sampling dtype of every check but the ``dtype_*`` ones.
    dtypes : sequence of numpy dtype
        One ``dtype_<name>`` check each.  Add ``float16`` or
        ``bfloat16`` only for a kind that claims it.
    channels : int, optional
        ``structure`` also delivers a field with this many trailing
        channels, ``(n_source, C)``, which the protocol documents.
        ``None`` for a kind that documents scalar fields only.
    rounding_units : float
        Width of every comparison, in units of ``eps * gain *
        max|field|``: ``eps`` the coarsest rounding among the dtypes
        involved, ``gain`` the largest absolute row sum of the operator
        delivered.  See :data:`DEFAULT_ROUNDING_UNITS`.  Raise it for a
        kind whose documented accuracy is not rounding (a kernel
        interpolant solved with a ridge), and say so.  Whatever the
        width, no comparison is tighter than what flushes to zero can
        cost the values it compares: ``2 * tiny * (1 + |scale| * (2 *
        n_source - 1) + gain)`` for each delivered value (the module's
        notes say why), which is not in rounding units and matters only
        at the underflow end.
    resolve_points : callable, optional
        Resolver for the ``round_trip`` check's references
        (``gm.point_resolver()`` for references to node fields).
    base_dir : path-like, optional
        Directory asset references are relative to, for ``round_trip``.
    max_examples, derandomize
        As for ``verify_node``.

    Returns
    -------
    dict[str, VerificationResult]
        One entry per check:

        ``structure``
            The protocol's members; the ``params_pytree()`` contract
            ``add_edge`` enforces; the delivered and the transposed
            shapes for the declared sizes; no argument, weight or
            parameter changed by a call, and nothing kept between calls.
        ``linearity``
            ``delivered(a x + b y) == a delivered(x) + b delivered(y)``.
        ``consistent`` / ``conservative``
            As claimed; ``SKIP`` when not claimed.
        ``adjoint``
            ``<delivered(x), y> == scale * <x, apply_T(y)>``.  ``SKIP``
            when ``linearity`` failed: the identity is between two
            linear maps.
        ``geometry_derivative``
            For a geometry-dependent kind: the derivative with respect
            to a position against a central difference.  A draw on a
            kink of the kernel is detected (the one-sided slopes do not
            close with the step), counted in the result's detail and
            not compared.  ``SKIP`` for a static kind, and when no draw
            could be compared.
        ``outside_hull``
            With ``hull`` and ``outside="clamp"``: positions at and
            outside the hull deliver what their projections deliver, and
            the derivative with respect to a coordinate strictly outside
            is zero.  ``SKIP`` otherwise.
        ``dtype_<name>``
            A field of that dtype is delivered in the dtype the field
            and the weights promote to, finite, and equal to the widest
            evaluation to that dtype's rounding.  ``SKIP`` for float64
            without ``jax_enable_x64``.
        ``jit_consistent``
            ``jax.jit`` of the delivery equals the eager one.
        ``round_trip``
            For a kind with a complete
            :class:`~maddening.core.coupling.mapping_spec.MappingSpec`
            (a registered kind): written as a config writes it and
            rebuilt as a config reads it, the mapping has the same
            weights and delivers the same field, bit for bit.  ``SKIP``,
            with the reason, for an unregistered kind, a closure, or
            references this call cannot resolve.

    Raises
    ------
    ValueError
        An unknown name in ``checks``; options that contradict each
        other; a geometry-dependent kind with no positions, or positions
        for a static one; a degree above 0 without coordinates; a
        supplied strategy that raises.  None is a verdict on the
        mapping.

    See Also
    --------
    maddening.testing.verification.verify_node : the node.
    maddening.testing.coupled.verify_graph_order : the coupled graph.
    """
    selected = list(DEFAULT_MAPPING_CHECKS) if checks is None else list(checks)
    unknown = set(selected) - set(DEFAULT_MAPPING_CHECKS)
    if unknown:
        raise ValueError(
            f"unknown checks {sorted(unknown)}; valid: {sorted(DEFAULT_MAPPING_CHECKS)}")
    if outside is not None and outside != "clamp":
        raise ValueError(
            f"outside={outside!r}: only 'clamp' (a position outside the hull is its "
            f"projection onto it) can be checked here")
    if (hull is None) != (outside is None):
        raise ValueError("hull= and outside= go together: the box, and what the kind "
                         "documents beyond it")
    if isinstance(polynomial_order, bool) or not isinstance(polynomial_order, int) \
            or polynomial_order < 0:
        raise ValueError(f"polynomial_order must be a non-negative integer, got "
                         f"{polynomial_order!r}")
    if polynomial_order > 0 and (source_coordinates is None or target_coordinates is None):
        raise ValueError(
            f"polynomial_order={polynomial_order} is a claim about polynomials of the "
            f"coordinates: pass source_coordinates= and target_coordinates=")
    if geometry is not None and geometry_strategy is not None:
        raise ValueError("geometry is one fixed array of positions and geometry_strategy "
                         "draws them: pass one or the other")
    if geometry_strategy is not None and not isinstance(geometry_strategy, st.SearchStrategy):
        raise TypeError(f"geometry_strategy must be a Hypothesis strategy that yields "
                        f"positions, got {type(geometry_strategy).__name__}")
    if not (isinstance(rounding_units, (int, float)) and rounding_units > 0):
        raise ValueError(f"rounding_units must be positive, got {rounding_units!r}")

    try:
        subject = _subject(mapping, transform, scale, weights)
    except _MissingMembers as missing:
        inner = mapping.mapping if isinstance(mapping, EdgeSpec) else mapping
        failed = VerificationResult("structure", "FAIL", detail=(
            f"{type(inner).__name__} lacks {missing.names}: a mapping needs the members "
            f"of the Mapping protocol (kind, mode, n_source, n_target, params_pytree, "
            f"apply, apply_T)"))
        return {"structure": failed, **{
            name: _skip(name, "the mapping lacks protocol members; see 'structure'")
            for name in _result_names(selected, dtypes) if name != "structure"}}
    has_positions = geometry is not None or geometry_strategy is not None
    if subject.needs_geometry and not has_positions:
        raise ValueError(
            f"{subject.label} reads a moving geometry (needs_geometry is True): pass "
            f"sample positions as geometry= or a Hypothesis strategy for them as "
            f"geometry_strategy=")
    if not subject.needs_geometry and (has_positions or hull is not None
                                       or geometry_step is not None):
        raise ValueError(
            f"{subject.label} is a static mapping (it has no needs_geometry=True), so "
            f"geometry=, geometry_strategy=, geometry_step=, hull= and outside= do not "
            f"apply to it")

    mode = getattr(subject.mapping, "mode", None)
    if consistent is None:
        consistent = mode == "consistent"
    if conservative is None:
        conservative = mode == "conservative"
    sampling = _Sampling(np.dtype(dtype), (float(bounds[0]), float(bounds[1])),
                         None if geometry is None else jnp.asarray(geometry),
                         geometry_strategy)
    kw: dict[str, Any] = dict(max_examples=max_examples, derandomize=derandomize)
    unclaimed = ("not claimed: neither stated by the caller nor implied by the mapping's "
                 f"mode ({mode!r}), so it was not checked")

    results: dict[str, VerificationResult] = {}
    for name in selected:
        if name == "structure":
            results[name] = _mapping_structure(subject, sampling, channels, rounding_units,
                                               **kw)
        elif name == "linearity":
            results[name] = _mapping_linearity(subject, sampling, rounding_units, **kw)
        elif name == "consistent":
            results[name] = _mapping_consistent(
                subject, sampling, rounding_units, polynomial_order, source_coordinates,
                target_coordinates, **kw) if consistent else _skip(name, unclaimed)
        elif name == "conservative":
            results[name] = _mapping_conservative(
                subject, sampling, rounding_units, polynomial_order, source_coordinates,
                target_coordinates, source_measure, target_measure,
                **kw) if conservative else _skip(name, unclaimed)
        elif name == "adjoint":
            linear = results.get("linearity")
            if linear is not None and not linear.passed:
                results[name] = _skip(name, (
                    "the delivery is not linear (see 'linearity'), and the adjoint "
                    "identity is between two linear maps: nothing to compare"))
            else:
                results[name] = _mapping_adjoint(subject, sampling, rounding_units, **kw)
        elif name == "geometry_derivative":
            results[name] = _mapping_geometry_derivative(
                subject, sampling, rounding_units, geometry_step,
                **kw) if subject.needs_geometry else _skip(
                    name, "a static mapping: it reads no positions")
        elif name == "outside_hull":
            if not subject.needs_geometry:
                results[name] = _skip(name, "a static mapping: it reads no positions")
            elif hull is None:
                results[name] = _skip(name, (
                    "not claimed: pass hull=(lower, upper) and outside='clamp' for a "
                    "kind that documents what a position outside its kernel's box is"))
            else:
                results[name] = _mapping_outside_hull(subject, sampling, rounding_units,
                                                      hull, **kw)
        elif name == "dtype":
            for one in dtypes:
                result = _mapping_dtype(subject, sampling, rounding_units, np.dtype(one), **kw)
                results[result.name] = result
        elif name == "jit_consistent":
            results[name] = _mapping_jit(subject, sampling, rounding_units, **kw)
        elif name == "round_trip":
            results[name] = _mapping_round_trip(subject, sampling, resolve_points,
                                                base_dir, **kw)
    return results


def _result_names(selected: Sequence[str], dtypes: Sequence[npt.DTypeLike]) -> list[str]:
    names: list[str] = []
    for name in selected:
        if name == "dtype":
            names += [f"dtype_{np.dtype(one).name}" for one in dtypes]
        else:
            names.append(name)
    return names


@stability(StabilityLevel.EXPERIMENTAL)
def assert_mapping_verified(mapping: Any, *, require: Sequence[str] = (),
                            **kwargs: Any) -> None:
    """:func:`verify_mapping` that raises ``AssertionError`` listing every failure.

    A one-line pytest body::

        def test_my_kind_transfers_what_it_claims():
            assert_mapping_verified(my_mapping(xs, xt), require=("round_trip",))

    Parameters
    ----------
    mapping
        As for :func:`verify_mapping`, and every keyword of it is
        accepted and passed on.
    require : sequence of str
        Results that must have been *checked*: a ``SKIP`` of one of
        these (a claim not made, a round trip that could not be taken)
        raises too, with its reason.  A ``SKIP`` otherwise counts as
        passed, as it does for a node.

    Raises
    ------
    AssertionError
        A check failed or could not run, or a required one was skipped.
    """
    results = verify_mapping(mapping, **kwargs)
    absent = [name for name in require if name not in results]
    if absent:
        raise ValueError(f"require names {absent}, which are not among the results "
                         f"{sorted(results)}")
    bad = [r for r in results.values() if not r.passed]
    bad += [results[name] for name in require if results[name].skipped]
    if bad:
        inner = mapping.mapping if isinstance(mapping, EdgeSpec) else mapping
        raise AssertionError(
            f"{type(inner).__name__} failed {len(bad)} check(s):\n"
            + "\n".join(str(r) for r in bad))
