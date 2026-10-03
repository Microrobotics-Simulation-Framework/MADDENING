"""Per-parameter metadata for the graph parameter pytree.

:class:`ParamSpec` says, for one leaf of ``GraphManager.params``, whether
an optimiser may move it (``trainable``), what range is physical
(``bounds``) and how to reparametrise it so an unconstrained optimiser
cannot leave that range (``transform``).  Nodes declare specs for their
own constants in :meth:`SimulationNode.param_specs`; a graph can
override any of them with :meth:`GraphManager.set_param_spec`.

The three module functions are pure pytree maps, so they compose with
``jax.jit`` / ``jax.grad``:

* :func:`trainable_mask` — ``params``-shaped pytree of bools;
* :func:`unconstrain` — ``params`` → optimiser coordinates ``u``;
* :func:`constrain` — ``u`` → ``params`` (the inverse, plus clipping for
  bounded identity leaves).

Non-trainable leaves pass through both maps unchanged, so
``constrain(unconstrain(p)) == p`` on the whole tree and an optimiser
that only updates the masked leaves never touches the others.

Transforms::

    None      p = u                     (bounds enforced by clipping in constrain;
                                         on a bound, d p / d u is 1, into the range)
    "log"     p = lo + exp(u)           lo = bounds[0] or 0; p > lo strictly
    "logit"   p = lo + (hi - lo) * sigmoid(u)    both bounds required; lo < p < hi
"""

from __future__ import annotations

import functools
import math
import numbers
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

_TRANSFORMS = (None, "log", "logit")


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _clip_into_bounds(u, lo, hi):
    """``jnp.clip(u, lo, hi)`` whose derivative *at* a bound is the
    one-sided derivative into the interior.

    The bounded identity transform clips.  ``jnp.clip`` is
    ``minimum(maximum(u, lo), hi)``, and JAX splits a tie between the two
    arguments of ``maximum`` evenly, so at ``u == lo`` exactly its
    derivative was **0.5** -- half of what moving the parameter into its
    range does, and the direction a parameter sitting on its bound can
    move.  A spring held at ``damping = 0.0`` (its lower bound) gave a
    gradient of -2.64e-2 where a one-sided difference into the range gave
    -5.24e-2, and ``fit_lm``'s Jacobian column there (it differentiates
    through ``constrain``) was halved, its Gauss-Newton entry quartered.
    (``fim`` differentiates the physical parameters, where nothing is
    clipped.)  At a bound the derivative is now 1 (into the range);
    strictly outside the bounds it is 0 and strictly inside it is 1, as
    before.  The value, and the derivative everywhere but on a bound, are
    ``jnp.clip``'s bit for bit: the same expression computes the value, and
    inside the bounds the tangent is passed through unchanged, as
    ``jnp.clip``'s selects pass it.
    """
    return jnp.clip(u, lo, hi)


@_clip_into_bounds.defjvp
def _clip_into_bounds_jvp(lo, hi, primals, tangents):
    (u,), (du,) = primals, tangents
    inside = jnp.ones(jnp.shape(u), dtype=bool)
    if lo is not None:
        inside = inside & (u >= lo)
    if hi is not None:
        inside = inside & (u <= hi)
    return _clip_into_bounds(u, lo, hi), jnp.where(inside, du, jnp.zeros_like(du))


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class ParamSpec:
    """Metadata for one leaf of the graph parameter pytree.

    Parameters
    ----------
    trainable : bool
        Whether an optimiser may change the value.  Initial conditions
        and parameters known a priori are declared ``False`` so a fit
        cannot walk an unidentifiable direction through them (the
        ``(k, c, m)`` common-scale direction of a spring, for one).
    bounds : (lo, hi)
        Physical range; ``None`` on either side means unbounded.  Each
        side is ``None`` or a real number that is not ``NaN``.  An
        infinity pointing the unbounded way (``lo = -inf``,
        ``hi = +inf``) is accepted and means exactly what ``None`` means
        -- to :meth:`check`, :func:`constrain` and ``fim``'s nominal
        scale alike -- and is kept as given, so a document that stored
        ``(-inf, inf)`` round-trips; ``None`` is the canonical spelling.
        Refused: ``NaN`` on either side (every comparison with it is
        False, so :meth:`check` passed any value and :func:`constrain`
        returned ``NaN``), an infinity pointing the other way (a lower
        bound of ``+inf`` admits no value), and an infinite bound under
        a transform that needs a finite one (``"logit"`` mapped every
        value to ``-inf`` under ``(0, inf)``).
    transform : {None, "log", "logit"}
        Reparametrisation used by :func:`unconstrain` / :func:`constrain`.
        ``"log"`` needs ``hi is None`` and is measured from ``lo``, or from
        0 when ``lo`` is ``None``, so :meth:`check` requires a value above
        it either way; ``"logit"`` needs both bounds.  ``None`` with
        bounds clips in :func:`constrain`: the derivative is 1 inside the
        bounds, 0 beyond them, and 1 *on* a bound -- the one-sided
        derivative into the range, the way a parameter sitting there can
        move (``jnp.clip`` alone splits that tie and gives 0.5).
    description, units : str
        Documentation only (surfaced by FMI ``parameter`` variables).
    """
    trainable: bool = True
    bounds: tuple[Optional[float], Optional[float]] = (None, None)
    transform: Optional[str] = None
    description: str = ""
    units: str = ""

    def __post_init__(self):
        if self.transform not in _TRANSFORMS:
            raise ValueError(
                f"ParamSpec.transform={self.transform!r} not in {_TRANSFORMS}"
            )
        lo, hi = self.bounds
        for side, b, unbounded in (("lower", lo, -math.inf),
                                   ("upper", hi, math.inf)):
            if b is None:
                continue
            if isinstance(b, bool) or not isinstance(b, numbers.Real):
                raise ValueError(
                    f"ParamSpec.bounds: the {side} bound {b!r} is not a real "
                    f"number; give a number, or None for no bound")
            if math.isnan(b):
                raise ValueError(
                    f"ParamSpec.bounds: the {side} bound is NaN. Every "
                    f"comparison with NaN is False, so check() would pass any "
                    f"value and constrain() would return NaN; give a number, "
                    f"or None for no bound")
            if math.isinf(b) and b != unbounded:
                raise ValueError(
                    f"ParamSpec.bounds: a {side} bound of {b} admits no value "
                    f"at all; give a finite number, or None for no bound")
            if math.isinf(b) and self.transform is not None:
                raise ValueError(
                    f"ParamSpec.transform={self.transform!r} needs a finite "
                    f"{side} bound or None, got {b}: the transform is measured "
                    f"from its bounds, and from an infinite one every value "
                    f"maps to an infinite coordinate. Spell no bound as None")
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"ParamSpec.bounds must satisfy lo < hi, got {self.bounds}")
        if self.transform == "log" and hi is not None:
            raise ValueError("ParamSpec.transform='log' takes a lower bound only")
        if self.transform == "logit" and (lo is None or hi is None):
            raise ValueError("ParamSpec.transform='logit' needs both bounds")

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-compatible form (``bounds`` as a 2-list with ``None``)."""
        return {
            "trainable": self.trainable,
            "bounds": [self.bounds[0], self.bounds[1]],
            "transform": self.transform,
            "description": self.description,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        """The spec :meth:`to_dict` wrote (a saved graph, a USD stage).

        ``trainable`` must be a JSON boolean, and each side of ``bounds``
        ``null`` or a number; anything else is refused rather than
        coerced.  Until 0.4.0 ``trainable`` went through ``bool()``, so
        the strings ``"false"``, ``"no"`` and ``"0"`` all read as
        trainable, and a bound went through ``float()``, so ``true`` read
        as ``1.0`` and ``"2"`` as ``2.0`` past the refusals of
        :meth:`__post_init__`.

        Raises
        ------
        ValueError
            If ``trainable`` is not a boolean, or a bound is not ``None``
            or a real number (or is one :meth:`__post_init__` refuses).
        """
        trainable = d.get("trainable", True)
        if not isinstance(trainable, bool):
            raise ValueError(
                f"ParamSpec.trainable must be true or false, got "
                f"{type(trainable).__name__} {trainable!r}; a document that "
                f"means 'not trainable' must say false, not a string or a "
                f"number")
        # ``bounds`` may be serialised as JSON ``null`` (unbounded).
        lo, hi = d.get("bounds") or (None, None)

        def _bound(b):
            # A real number is widened to float as before; anything else
            # (a bool, a string) goes to ``__post_init__`` as given, which
            # refuses it by name.
            if isinstance(b, numbers.Real) and not isinstance(b, bool):
                return float(b)
            return b

        return cls(
            trainable=trainable,
            bounds=(_bound(lo), _bound(hi)),
            transform=d.get("transform"),
            description=d.get("description", ""),
            units=d.get("units", ""),
        )

    # -- leaf maps -------------------------------------------------------

    def _lo(self) -> float:
        return 0.0 if self.bounds[0] is None else float(self.bounds[0])

    def _require_floating(self, dtype) -> None:
        """A ``log`` / ``logit`` leaf has to be floating.

        :meth:`to_constrained`'s identity branch goes out of its way to
        keep a leaf's dtype -- "a leaf's dtype is part of the pytree
        contract" -- and the transform branches cannot: ``log`` of an
        ``int32`` leaf is float, nothing downstream remembers it was an
        integer, and ``constrain(unconstrain(p))`` comes back float
        (``logit`` also comes back 2.4e-7 off), which is exactly the
        identity this module's docstring states for the whole tree.  So
        refuse the combination instead of quietly changing a dtype: a
        parameter an optimiser moves continuously through ``exp`` or
        ``sigmoid`` is not an integer parameter.
        """
        if not jnp.issubdtype(dtype, jnp.floating):
            raise ValueError(
                f"ParamSpec.transform={self.transform!r} needs a floating-point "
                f"leaf, got dtype {dtype}. The transform maps the leaf through "
                "exp/sigmoid, so constrain(unconstrain(p)) cannot return the "
                "integer it started from, and this module promises that round "
                "trip over the whole tree. Either store the parameter as a "
                "float, or declare it with transform=None (bounds are still "
                "enforced, by clipping, and the dtype is preserved)."
            )

    def to_unconstrained(self, p):
        p = jnp.asarray(p)
        if self.transform in ("log", "logit"):
            self._require_floating(p.dtype)
        if self.transform == "log":
            return jnp.log(p - self._lo())
        if self.transform == "logit":
            # `__post_init__` refuses transform='logit' without both bounds.
            lo_b, hi_b = self.bounds
            assert lo_b is not None and hi_b is not None
            lo, hi = float(lo_b), float(hi_b)
            t = (p - lo) / (hi - lo)
            return jnp.log(t) - jnp.log1p(-t)
        return p

    def _log_floor(self, fi) -> float:
        """The smallest ``exp(u)`` a ``log`` leaf takes: ``2 * eps * |lo|``
        is 2-4 ulps of ``lo`` (>= 1 ulp is what makes ``lo + m > lo``
        exact); ``tiny`` keeps the ``lo == 0`` floor at the smallest
        normal."""
        return max(float(fi.tiny), 2.0 * float(fi.eps) * abs(self._lo()))

    def _logit_interior(self, fi) -> tuple[float, float]:
        """The representable interior a ``logit`` leaf is clamped to.

        Clipping the *result* (not the sigmoid) keeps rounding in
        ``lo + (hi - lo) * t`` off a bound: with ``m >= 4 ulp`` of every
        quantity involved, ``p - lo`` and ``hi - lo`` round to distinct
        floats and the inverse ``log(t) - log1p(-t)`` stays finite.  An
        interval only a few ulps wide (``(1e6, 1e6 + 0.1)`` in float32) has
        no interior to clamp to; the margin is capped so the clip never
        inverts, and ``check`` reports such a value.
        """
        lo_b, hi_b = self.bounds
        assert lo_b is not None and hi_b is not None
        lo, hi = float(lo_b), float(hi_b)
        m = 4.0 * float(fi.eps) * max(abs(lo), abs(hi), hi - lo)
        m = min(m, 0.25 * (hi - lo))
        # ``lo + m`` / ``hi - m`` can round back onto the bound for an
        # interval a few ulps wide; the next representable float inside is
        # the true limit, so take the wider of the two.
        _t = np.dtype(fi.dtype).type
        inner_lo = max(lo + m, float(np.nextafter(_t(lo), _t(np.inf))))
        inner_hi = min(hi - m, float(np.nextafter(_t(hi), _t(-np.inf))))
        if inner_lo > inner_hi:                    # no interior float at all
            inner_lo = inner_hi = 0.5 * (lo + hi)
        return inner_lo, inner_hi

    def _optimiser_interval(self, dtype) -> tuple[float, float]:
        """The optimiser coordinates ``u`` over which :meth:`to_constrained`
        is the transform itself, not a clamp.

        Outside it ``to_constrained`` returns a clamped value whose
        derivative is 0, so an optimiser that steps there has no gradient
        back: the bounds of a ``transform=None`` leaf, and for ``log`` /
        ``logit`` the coordinates where ``exp`` leaves its floor or
        overflows, or the sigmoid lands on the representable interior's
        edge.  Each end is moved inward until the value it maps to is
        strictly inside the clamp, at ``dtype``'s rounding, so a coordinate
        projected onto it keeps a non-zero derivative.  ``(-inf, inf)`` for
        an unbounded identity leaf, and for a ``logit`` interval with no
        interior float to clamp to.
        """
        lo_b, hi_b = self.bounds
        if self.transform is None:
            return (-math.inf if lo_b is None else float(lo_b),
                    math.inf if hi_b is None else float(hi_b))
        fi = np.finfo(dtype)
        if self.transform == "log":
            floor, ceiling = self._log_floor(fi), float(fi.max)
            u_lo, u_hi = math.log(floor), math.log(ceiling)
            lo = self._lo()

            def inside(u):
                e = float(jnp.exp(jnp.asarray(u, dtype)))
                return floor < e < ceiling
        else:
            inner_lo, inner_hi = self._logit_interior(fi)
            if not inner_lo < inner_hi:
                return -math.inf, math.inf
            lo, hi = float(lo_b), float(hi_b)  # type: ignore[arg-type]
            t_lo, t_hi = (inner_lo - lo) / (hi - lo), (inner_hi - lo) / (hi - lo)
            u_lo = math.log(t_lo) - math.log1p(-t_lo)
            u_hi = math.log(t_hi) - math.log1p(-t_hi)

            def inside(u):
                v = float(lo + (hi - lo) * jax.nn.sigmoid(jnp.asarray(u, dtype)))
                return inner_lo < v < inner_hi

        def inward(u, direction):
            step = 1e-6 * max(1.0, abs(u))
            for _ in range(64):
                if inside(u):
                    return u
                u += direction * step
                step *= 2.0
            return u

        return inward(u_lo, 1.0), inward(u_hi, -1.0)

    def to_constrained(self, u):
        u = jnp.asarray(u)
        # ``exp`` under/overflows and ``sigmoid`` saturates to exactly 0
        # or 1 in float32 for |u| beyond ~17-88, which would put the
        # value *on* a strict bound (or at inf) and make the inverse map
        # non-finite.  Clamp to the representable interior, so the round
        # trip stays finite for any optimiser coordinates.  The margin
        # is *relative to the bound*: ``lo + tiny`` is ``lo`` again in
        # float32 for any ``lo != 0`` (the audit's ``8.0 + exp(-15)``
        # case), so the clamp must be a few ulps of ``lo`` / ``hi`` wide
        # or ``check`` rejects the fit's own output and ``unconstrain``
        # returns ``-inf``.
        floating = jnp.issubdtype(u.dtype, jnp.floating)
        fi = jnp.finfo(u.dtype if floating else jnp.float32)
        if self.transform == "log":
            return self._lo() + jnp.clip(jnp.exp(u), self._log_floor(fi), fi.max)
        if self.transform == "logit":
            # `__post_init__` refuses transform='logit' without both bounds.
            lo_b, hi_b = self.bounds
            assert lo_b is not None and hi_b is not None
            lo, hi = float(lo_b), float(hi_b)
            inner_lo, inner_hi = self._logit_interior(fi)
            p = lo + (hi - lo) * jax.nn.sigmoid(u)
            return jnp.clip(p, inner_lo, inner_hi)
        lo, hi = self.bounds
        if lo is not None or hi is not None:
            # Python-float bounds would promote an integer leaf (integer
            # matrix-mapping weights made trainable) to float; a leaf's
            # dtype is part of the pytree contract, so keep it.  An integer
            # leaf has no derivative to take, so it keeps the plain clip.
            if not floating:
                return jnp.clip(u, lo, hi).astype(u.dtype)
            return _clip_into_bounds(u, lo, hi).astype(u.dtype)
        return u

    def check(self, p, *, name: str = "param") -> None:
        """Raise ``ValueError`` if a concrete value is non-finite or
        violates ``bounds``.

        NaN compares ``False`` against every bound, so it would pass a
        pure comparison check and only surface as a NaN state later;
        ``inf`` is likewise never a usable constant.  Under
        ``transform="log"`` a missing lower bound is the 0 the transform is
        measured from, so a value ``<= 0`` is refused.

        The comparisons are made on the host, in NumPy, with the value and
        the bound each rounded to the dtype JAX would compare them in (the
        leaf's, or float32 for an integer leaf against a float bound).
        Through ``jnp`` they were not exact: XLA's CPU backend flushes a
        subnormal operand to zero, so a float32 ``-1e-40`` passed a
        ``(0, None)`` bound -- every ``check_params``, FMU ``set`` and
        ``set_state`` behind it then accepted a value below the range the
        spec declares, and reported it back as stored.  Under a ``log`` or
        ``logit`` transform the step's own arithmetic decides whether the
        value has a coordinate, so there its distance from a bound is
        computed as that arithmetic computes it (:func:`_step_gap`:
        subnormal operands and results flushed to zero) and a distance
        that is not positive is refused as on the bound: ``log`` of a
        float32 ``1e-40`` is ``-inf`` on that backend.
        """
        lo, hi = self.bounds
        v = np.asarray(p)
        strict = self.transform in ("log", "logit")
        if np.issubdtype(v.dtype, np.complexfloating):
            self._check_through_jnp(jnp.asarray(p), name, strict)
            return
        if jnp.issubdtype(v.dtype, jnp.inexact) \
                and not bool(np.all(np.isfinite(v.astype(np.float64)))):
            raise ValueError(f"{name}={v} is not finite")
        if lo is None and self.transform == "log":
            # ``log`` is measured from ``lo``, or from 0 without one
            # (:meth:`_lo`): 0 and below have no coordinate at all --
            # ``unconstrain`` returned -inf for 0 and NaN below it, and this
            # check used to pass both.
            a, b, tiny = _bound_operands(v, 0.0)
            if bool(np.any(_step_gap(a, b, tiny) <= 0.0)):
                raise ValueError(
                    f"{name}={v} below bound 0.0 (transform='log' without a "
                    "lower bound is measured from 0, so the value must be > 0, "
                    "and by at least the smallest normal number of its dtype, "
                    "below which the arithmetic flushes it to 0)")
        if lo is not None:
            a, b, tiny = _bound_operands(v, lo)
            if bool(np.any(_step_gap(a, b, tiny) <= 0.0) if strict else np.any(a < b)):
                raise ValueError(f"{name}={v} below bound {lo}")
        if hi is not None:
            a, b, tiny = _bound_operands(v, hi)
            if bool(np.any(_step_gap(b, a, tiny) <= 0.0) if strict else np.any(a > b)):
                raise ValueError(f"{name}={v} above bound {hi}")

    def _check_through_jnp(self, v, name: str, strict: bool) -> None:
        """:meth:`check` of a complex value, as before 0.4.0: NumPy orders
        complex numbers and JAX does not, so the host comparison would
        accept what this has always refused."""
        lo, hi = self.bounds
        if not bool(jnp.all(jnp.isfinite(v))):
            raise ValueError(f"{name}={v} is not finite")
        if lo is None and self.transform == "log" and bool(jnp.any(v <= 0.0)):
            raise ValueError(f"{name}={v} below bound 0.0")
        if lo is not None and bool(jnp.any(v <= lo) if strict else jnp.any(v < lo)):
            raise ValueError(f"{name}={v} below bound {lo}")
        if hi is not None and bool(jnp.any(v >= hi) if strict else jnp.any(v > hi)):
            raise ValueError(f"{name}={v} above bound {hi}")


def _bound_operands(v: np.ndarray, bound) -> tuple[np.ndarray, float, float]:
    """``(value, bound, tiny)`` for an exact host comparison of ``v`` with
    ``bound``: each rounded to the dtype JAX compares them in (the result
    type of the leaf and a weakly typed Python number), then widened to
    float64, which holds every such value exactly -- a float32 subnormal is
    a normal float64.  ``tiny`` is that dtype's smallest normal number."""
    # ``canonicalize_dtype`` first: a float64 NumPy value without x64 is
    # compared in float32, as ``jnp.asarray`` would make it, and asking
    # ``result_type`` of float64 itself warns that it is unavailable.
    dt = np.dtype(jnp.result_type(jax.dtypes.canonicalize_dtype(v.dtype), bound))
    a = v.astype(dt).astype(np.float64)
    b = float(np.asarray(bound, dtype=dt).astype(np.float64))
    tiny = float(jnp.finfo(dt).tiny) if jnp.issubdtype(dt, jnp.floating) else 0.0
    return a, b, tiny


def _step_gap(a, b, tiny: float):
    """``a - b`` as XLA's CPU backend computes it in a dtype whose smallest
    normal number is ``tiny``: a subnormal operand is read as zero, and a
    subnormal result is stored as zero.  ``a`` and ``b`` are exact float64
    copies of values of that dtype (:func:`_bound_operands`), so the
    difference of the flushed operands is exact and only its flush is
    left to apply.  A strict bound is met when this is positive: it is the
    distance ``log`` and the logistic's inverse take."""
    def flush(x):
        return np.where(np.abs(x) < tiny, 0.0, x)
    gap = flush(np.asarray(a, dtype=np.float64)) - flush(np.asarray(b, dtype=np.float64))
    return flush(gap)


DEFAULT_SPEC = ParamSpec()


# ---------------------------------------------------------------------------
# Resolving a ParamSpec for each params leaf: one walk, two policies
# ---------------------------------------------------------------------------
#
# ``specs`` mirrors ``params``.  Which ``ParamSpec`` governs a leaf is
# decided by descending ``specs`` along the leaf's pytree path, one key at
# a time, and there is exactly one definition of a step of that descent:
# :func:`_spec_step`.  The per-path reader (:func:`_spec_for`) folds it
# over one path; the tree reader (:func:`_resolve_specs`) applies it at
# every level of the whole tree.  Before this was so, the two were
# separate walks written against different ideas of a ``namedtuple`` --
# the checker saw a tuple and accepted a list of specs for it, the reader
# saw the ``GetAttrKey`` path and read a dict -- so the spelling the
# checker accepted was never read and ``check_bounds`` passed ``k=50``
# against declared bounds ``(-5, 5)``.  A differential fuzz found 633 of
# 3100 accepted random trees resolving differently.  Sharing the step
# makes "accepted" and "read" the same statement.

#: No entry on this path: the leaves below take :data:`DEFAULT_SPEC`.
_ABSENT = object()

#: Path keys addressed by position -- a list/tuple element, a child of a
#: node registered without keys -- and so by a list/tuple of specs.
_POSITIONAL_KEYS = (jax.tree_util.SequenceKey, jax.tree_util.FlattenedIndexKey)
#: Path keys addressed by name -- a dict key, a namedtuple or dataclass
#: field -- and so by a dict of specs keyed the same way.
_NAMED_KEYS = (jax.tree_util.DictKey, jax.tree_util.GetAttrKey)


def _path_component(key):
    """The plain Python key a pytree path entry addresses: the dict key of
    a ``DictKey``, the field name of a ``GetAttrKey``, the position of a
    ``SequenceKey`` or ``FlattenedIndexKey``; ``None`` for a key type
    ``jax.tree_util`` did not have when this was written (a custom node's
    own key class), which :func:`_spec_step` refuses to address."""
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    if isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    if isinstance(key, jax.tree_util.FlattenedIndexKey):
        return key.key
    return None


def _spec_step(node, key):
    """One step of the descent: from the ``specs`` entry ``node`` toward
    the params child at path key ``key``.

    Returns ``(child, None)`` -- ``child`` may be :data:`_ABSENT` -- or
    ``(None, problem)`` with a short code naming why ``node`` cannot
    address that child.  The rules:

    * a **named** key (dict key, namedtuple/dataclass field) is read from
      a dict of specs keyed the same way; a missing key is ``_ABSENT``;
    * a **positional** key (list/tuple element, keyless custom node) is
      read from a list/tuple of specs by position;
    * a ``ParamSpec`` covers a **list/tuple** level beneath it -- one spec
      per vector, as one spec covers every element of an array leaf --
      and nothing else: a dict or a record has named parameters, each of
      which needs its own entry.
    """
    if isinstance(node, ParamSpec):
        if isinstance(key, jax.tree_util.SequenceKey):
            return node, None
        return None, "paramspec_above"
    if isinstance(key, _POSITIONAL_KEYS):
        if not isinstance(node, (list, tuple)):
            return None, "needs_sequence"
        i = (key.idx if isinstance(key, jax.tree_util.SequenceKey)
             else key.key)
        if not 0 <= i < len(node):
            return None, "too_short"
        return node[i], None
    if isinstance(key, _NAMED_KEYS):
        if not isinstance(node, dict):
            return None, "needs_dict"
        return node.get(_path_component(key), _ABSENT), None
    return None, "unaddressable"


def _leaf_spec(node):
    """``(spec, None)`` for the entry the descent reached at a params
    leaf, or ``(None, problem)`` when that entry is not a spec."""
    if node is _ABSENT:
        return DEFAULT_SPEC, None
    if isinstance(node, ParamSpec):
        return node, None
    if isinstance(node, dict):
        return None, "dict_at_leaf"
    return None, "not_a_spec"


def _spec_for(specs: dict, path) -> ParamSpec:
    """The :class:`ParamSpec` governing the params leaf at ``path``.

    :func:`_spec_step` folded over ``path``: a nested dict for every dict
    (or record) level, a list/tuple of specs by position or one covering
    ``ParamSpec`` for a list/tuple level, a ``ParamSpec`` at the leaf.  A
    missing entry means the default spec, and so -- this is the lenient
    reader -- does an entry the descent cannot read.  Every whole-tree
    consumer goes through :func:`_resolve_specs`, which runs the same
    steps and *refuses* an unreadable entry; for a ``specs`` that
    :func:`_resolve_specs` accepts, the two agree leaf for leaf by
    construction (and ``tests/property/test_param_spec_resolution.py``
    holds them to it).
    """
    node = specs
    for key in path:
        if node is _ABSENT:
            break
        node, problem = _spec_step(node, key)
        if problem is not None:
            return DEFAULT_SPEC
    spec, problem = _leaf_spec(node)
    if problem is not None or spec is None:
        return DEFAULT_SPEC
    return spec


def _child_entries(node):
    """``[(path key, child), ...]`` for one level of a pytree node, or
    ``None`` if ``node`` is a leaf.  An empty container (``{}``, ``()``,
    ``None``) is not a leaf and has no children."""
    if jax.tree_util.all_leaves([node]):
        return None
    entries = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda x: x is not node)[0]
    return [(path[0], child) for path, child in entries]


def _describe(param_node) -> str:
    """How a params node reads in an error message."""
    entries = _child_entries(param_node)
    if entries is None:
        return "a leaf"
    if not entries:
        return "an empty container"
    keys = [k for k, _ in entries]
    comps = [_path_component(k) for k in keys]
    if all(isinstance(k, jax.tree_util.DictKey) for k in keys):
        return f"a dict with keys {comps!r}"
    if all(isinstance(k, jax.tree_util.GetAttrKey) for k in keys):
        return (f"a {type(param_node).__name__} record with fields "
                f"{comps!r}")
    if all(isinstance(k, jax.tree_util.SequenceKey) for k in keys):
        return f"a sequence of {comps!r}"
    return (f"a {type(param_node).__name__} node whose {len(keys)} children "
            f"jax addresses by {type(keys[0]).__name__}")


def _step_message(problem, spec_node, param_node, key, where) -> str:
    kind = type(spec_node).__name__
    if problem == "paramspec_above":
        return (f"specs{where} is a ParamSpec but params{where} is "
                f"{_describe(param_node)}: a ParamSpec belongs at a leaf (one "
                f"may cover a list/tuple of leaves); use a nested dict with an "
                f"entry per parameter here")
    if problem == "needs_sequence":
        cover = (", or one ParamSpec covering every position"
                 if isinstance(key, jax.tree_util.SequenceKey) else "")
        return (f"specs{where} is a {kind} but params{where} is "
                f"{_describe(param_node)}: give a list/tuple of ParamSpec "
                f"read by position{cover}")
    if problem == "needs_dict":
        named = isinstance(key, jax.tree_util.GetAttrKey)
        return (f"specs{where} is a {kind} but params{where} is "
                f"{_describe(param_node)}: "
                + ("its children are addressed by field name, not position, "
                   "so give a dict keyed by field name"
                   if named else
                   "a sequence of specs is read by position and needs a "
                   "list/tuple of parameters; give a dict keyed the same way"))
    if problem == "too_short":
        n = len(_child_entries(param_node) or ())
        return (f"specs{where} has {len(spec_node)} entries but "
                f"params{where} has {n} positions")
    return (f"specs{where} is given but params{where} is "
            f"{_describe(param_node)}, and a specs tree has no way to address "
            f"children keyed like that; leave specs{where} out and its leaves "
            f"take the default spec")


def _leaf_message(problem, spec_node, where) -> str:
    if problem == "dict_at_leaf":
        hint = ""
        if set(spec_node) <= set(ParamSpec.__dataclass_fields__):
            hint = (" (this looks like ParamSpec.to_dict() output; "
                    "pass ParamSpec.from_dict(...) instead)")
        return (f"specs{where} is a dict but params{where} is a leaf, "
                f"whose entry must be a ParamSpec{hint}")
    return (f"specs{where} is {type(spec_node).__name__}; expected a "
            f"ParamSpec or a nested dict of them")


def _resolve_specs(params, specs, *, unmatched=None) -> list:
    """The :class:`ParamSpec` of every params leaf, in flatten order --
    or a ``ValueError`` naming the first entry that cannot be read.

    Walks ``params`` top down, descending ``specs`` alongside with
    :func:`_spec_step` (the step :func:`_spec_for` folds), so the spec it
    returns for a leaf is :func:`_spec_for`'s whenever it returns at all.
    What it refuses, each by its key path: a ``specs`` that is not a
    dict; an entry on a leaf's path that the step cannot read (a
    ``ParamSpec`` above a dict or record level, a list where names are
    needed or a dict where positions are, a list of specs shorter than
    the sequence it mirrors, a key type specs cannot address); and an
    entry at a leaf that is not a ``ParamSpec`` (``ParamSpec.to_dict()``
    output being the common one).  Each of those used to hand a leaf the
    default spec in silence -- ``check_bounds`` passing an out-of-bounds
    value, ``trainable_mask`` marking a frozen leaf trainable.

    ``unmatched`` decides what happens to an entry that lies on *no*
    leaf's path -- a key naming no parameter, a list longer than its
    sequence.  It is called as ``unmatched(where, entry, param_node,
    parent_where)`` and may raise; a list longer than its sequence is
    refused outright whenever ``unmatched`` is given.  ``None`` ignores them, which is what the tree maps
    need: :meth:`GraphManager.param_specs` declares a spec for every
    constant a node knows, including those ``params_pytree()`` leaves
    out as structural (a uniform ``HeatNode``'s ``grid_points=None``, a
    constant spelled as a Python ``int``), and refusing those would
    refuse every such graph.  The cost is that a misspelt key in a
    hand-built ``specs`` is not caught here; :func:`_validate_specs_mirror`
    is the strict form for a caller that must know its specs were read.
    """
    if not isinstance(specs, dict):
        raise ValueError(
            f"specs must be a dict of ParamSpec mirroring params -- "
            f"gm.param_specs() for a graph's tree, or "
            f"{{'k': ParamSpec(...)}} for a flat one -- got "
            f"{type(specs).__name__}")
    out: list = []

    def walk(spec_node, param_node, where: str) -> None:
        entries = _child_entries(param_node)
        if entries is None:
            spec, problem = _leaf_spec(spec_node)
            if problem is not None:
                raise ValueError(_leaf_message(problem, spec_node, where))
            out.append(spec)
            return
        if spec_node is _ABSENT:
            out.extend(DEFAULT_SPEC for _ in jax.tree_util.tree_leaves(param_node))
            return
        if not isinstance(spec_node, (ParamSpec, dict, list, tuple)):
            raise ValueError(_leaf_message("not_a_spec", spec_node, where))
        for key, child in entries:
            sub, problem = _spec_step(spec_node, key)
            if problem is not None:
                raise ValueError(
                    _step_message(problem, spec_node, param_node, key, where))
            walk(sub, child, where + jax.tree_util.keystr((key,)))
        if unmatched is None:
            return
        if isinstance(spec_node, dict):
            reached = {_path_component(k) for k, _ in entries
                       if isinstance(k, _NAMED_KEYS)}
            for k, v in spec_node.items():
                if k not in reached:
                    unmatched(f"{where}[{k!r}]", v, param_node, where)
        elif (isinstance(spec_node, (list, tuple))
              and len(spec_node) > len(entries)):
            raise ValueError(
                f"specs{where} has {len(spec_node)} entries but "
                f"params{where} has {len(entries)} positions")

    walk(specs, params, "")
    return out


def _validate_specs_mirror(params, specs, *, unmatched_ok=None) -> list:
    """Raise ``ValueError`` unless every entry of ``specs`` is read;
    otherwise return the per-leaf specs, as :func:`_resolve_specs` does.

    :func:`_resolve_specs`'s refusals, plus the one it leaves to its
    caller: an entry on no leaf's path -- a key that matches no
    parameter (a misspelt name is the case that must be loud), or a list
    of specs longer than its sequence.  A spec that reaches nothing
    changes nothing, so a caller that needs to know its ``specs`` were
    *read* (``fim`` under ``scale="nominal"``) asks this first.  A leaf
    *without* an entry is not refused -- it gets the default spec, and
    ``{}`` remains the explicit "no leaf has a declared width".

    ``unmatched_ok(entry) -> bool`` exempts a key that matches no
    parameter when the caller can show the entry could not have changed
    its answer whichever leaf it was meant for; ``fim`` passes "resolves
    to the same column scale as the default spec", which is what lets
    ``gm.param_specs()`` through for a graph whose nodes declare specs
    for constants that are not leaves.  Only a ``ParamSpec`` entry can be
    exempted: a whole unmatched sub-tree is the wrong level of the tree,
    not a declaration about a constant.
    """
    def refuse(where, entry, param_node, parent):
        if (unmatched_ok is not None and isinstance(entry, ParamSpec)
                and unmatched_ok(entry)):
            return
        why = ""
        if unmatched_ok is not None:
            why = (" An entry for a constant that is not a leaf of params -- "
                   "gm.param_specs() declares one for every constant a node "
                   "knows, including those params_pytree() leaves out as "
                   "structural (None, int, bool or string values) -- is "
                   "accepted only where it could not have changed the "
                   "answer, and this one could.")
        raise ValueError(
            f"specs{where} matches no parameter: params{parent} is "
            f"{_describe(param_node)}. A spec that reaches nothing changes "
            f"nothing, so it is refused rather than ignored; drop the entry "
            f"or fix the key.{why}")

    return _resolve_specs(params, specs, unmatched=refuse)


def _map_with_specs(fn, params, specs):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    per_leaf = _resolve_specs(params, specs)
    return jax.tree_util.tree_unflatten(
        treedef, [fn(s, leaf) for s, leaf in zip(per_leaf, leaves)])


@stability(StabilityLevel.EVOLVING)
def trainable_mask(params: dict, specs: dict) -> dict:
    """``params``-shaped pytree of Python bools from the specs.

    ``specs`` is resolved as :func:`check_bounds` describes: an entry on
    a leaf's path that cannot be read is a ``ValueError``, not the
    default (trainable) spec.
    """
    return _map_with_specs(lambda s, _: s.trainable, params, specs)


@stability(StabilityLevel.EVOLVING)
def unconstrain(params: dict, specs: dict) -> dict:
    """Map every trainable leaf to optimiser coordinates.

    ``specs`` is resolved as :func:`check_bounds` describes.
    """
    return _map_with_specs(
        lambda s, p: s.to_unconstrained(p) if s.trainable else p, params, specs,
    )


@stability(StabilityLevel.EVOLVING)
def constrain(u: dict, specs: dict) -> dict:
    """Inverse of :func:`unconstrain` (with clipping for bounded identity
    leaves); non-trainable leaves pass through.

    ``specs`` is resolved as :func:`check_bounds` describes.
    """
    return _map_with_specs(
        lambda s, x: s.to_constrained(x) if s.trainable else x, u, specs,
    )


@stability(StabilityLevel.EVOLVING)
def check_bounds(params: dict, specs: dict) -> None:
    """Raise ``ValueError`` naming the first leaf outside its bounds.

    This is a safety check, so it fails closed on a ``specs`` it cannot
    read: an entry on a leaf's path that the resolver cannot use -- a
    ``ParamSpec`` over a dict, a list of specs for a namedtuple or
    dataclass (whose fields are addressed by name), ``ParamSpec.to_dict()``
    output at a leaf -- is a ``ValueError`` naming its key path.  Each of
    those used to give the leaf the default, unbounded spec and pass.

    A key that matches *no* leaf is ignored.  That is deliberate and is
    the one gap: :meth:`GraphManager.param_specs` declares a spec for
    every constant a node knows, including those ``params_pytree()``
    leaves out as structural (a uniform ``HeatNode``'s
    ``grid_points=None``, a constant given as a Python ``int``), and
    ``gm.check_params`` hands that tree straight here -- refusing
    unmatched keys would refuse every such graph.  So a misspelt key in
    a hand-built ``specs`` is not caught by this function.
    """
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    for (path, leaf), spec in zip(entries, _resolve_specs(params, specs)):
        spec.check(leaf, name=jax.tree_util.keystr(path))
