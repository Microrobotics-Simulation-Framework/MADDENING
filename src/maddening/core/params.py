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

    None      p = u                     (bounds enforced by clipping in constrain)
    "log"     p = lo + exp(u)           lo = bounds[0] or 0; p > lo strictly
    "logit"   p = lo + (hi - lo) * sigmoid(u)    both bounds required; lo < p < hi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

_TRANSFORMS = (None, "log", "logit")


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
        Physical range; ``None`` on either side means unbounded.
    transform : {None, "log", "logit"}
        Reparametrisation used by :func:`unconstrain` / :func:`constrain`.
        ``"log"`` needs ``hi is None``; ``"logit"`` needs both bounds.
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
        # ``bounds`` may be serialised as JSON ``null`` (unbounded).
        lo, hi = d.get("bounds") or (None, None)
        return cls(
            trainable=bool(d.get("trainable", True)),
            bounds=(None if lo is None else float(lo), None if hi is None else float(hi)),
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
            lo = self._lo()
            # ``2 * eps * |lo|`` is 2-4 ulps of ``lo`` (>= 1 ulp is what
            # makes ``lo + m > lo`` exact); ``tiny`` keeps the lo == 0
            # floor at the smallest normal, as before.
            m = max(float(fi.tiny), 2.0 * float(fi.eps) * abs(lo))
            return lo + jnp.clip(jnp.exp(u), m, fi.max)
        if self.transform == "logit":
            # `__post_init__` refuses transform='logit' without both bounds.
            lo_b, hi_b = self.bounds
            assert lo_b is not None and hi_b is not None
            lo, hi = float(lo_b), float(hi_b)
            # Clip the *result* (not the sigmoid) so rounding in
            # ``lo + (hi - lo) * t`` cannot land on a bound either: with
            # ``m >= 4 ulp`` of every quantity involved, ``p - lo`` and
            # ``hi - lo`` round to distinct floats and the inverse
            # ``log(t) - log1p(-t)`` stays finite.
            # An interval only a few ulps wide (``(1e6, 1e6 + 0.1)`` in
            # float32) has no interior to clamp to; cap the margin so the
            # clip never inverts, and let ``check`` report such a value.
            m = 4.0 * float(fi.eps) * max(abs(lo), abs(hi), hi - lo)
            m = min(m, 0.25 * (hi - lo))
            # ``lo + m`` / ``hi - m`` can round back onto the bound for an
            # interval a few ulps wide; the next representable float
            # inside is the true limit, so take the wider of the two.
            _t = np.dtype(fi.dtype).type
            inner_lo = max(lo + m, float(np.nextafter(_t(lo), _t(np.inf))))
            inner_hi = min(hi - m, float(np.nextafter(_t(hi), _t(-np.inf))))
            if inner_lo > inner_hi:                    # no interior float at all
                inner_lo = inner_hi = 0.5 * (lo + hi)
            p = lo + (hi - lo) * jax.nn.sigmoid(u)
            return jnp.clip(p, inner_lo, inner_hi)
        lo, hi = self.bounds
        if lo is not None or hi is not None:
            # Python-float bounds would promote an integer leaf (integer
            # matrix-mapping weights made trainable) to float; a leaf's
            # dtype is part of the pytree contract, so keep it.
            return jnp.clip(u, lo, hi).astype(u.dtype)
        return u

    def check(self, p, *, name: str = "param") -> None:
        """Raise ``ValueError`` if a concrete value is non-finite or
        violates ``bounds``.

        NaN compares ``False`` against every bound, so it would pass a
        pure comparison check and only surface as a NaN state later;
        ``inf`` is likewise never a usable constant.
        """
        lo, hi = self.bounds
        v = jnp.asarray(p)
        if jnp.issubdtype(v.dtype, jnp.inexact) and not bool(jnp.all(jnp.isfinite(v))):
            raise ValueError(f"{name}={v} is not finite")
        strict = self.transform in ("log", "logit")
        if lo is not None:
            bad = bool(jnp.any(v <= lo)) if strict else bool(jnp.any(v < lo))
            if bad:
                raise ValueError(f"{name}={v} below bound {lo}")
        if hi is not None:
            bad = bool(jnp.any(v >= hi)) if strict else bool(jnp.any(v > hi))
            if bad:
                raise ValueError(f"{name}={v} above bound {hi}")


DEFAULT_SPEC = ParamSpec()


# ---------------------------------------------------------------------------
# Tree maps over (params, specs)
# ---------------------------------------------------------------------------


def _path_component(key):
    """The plain Python key a pytree path entry stands for: the dict key
    of a ``DictKey``, the attribute name of a ``GetAttrKey``, the
    position of a ``SequenceKey``; ``None`` for anything else."""
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    if isinstance(key, jax.tree_util.SequenceKey):
        return key.idx
    return None


def _spec_for(specs: dict, path) -> ParamSpec:
    """The :class:`ParamSpec` governing the params leaf at ``path``.

    ``specs`` mirrors the params tree: a nested dict for every dict (or
    attribute) level, and for a list/tuple level either a list/tuple of
    specs read by position or a single ``ParamSpec`` that covers every
    position beneath it -- one spec per vector, the way one spec covers
    every element of an array leaf.  A missing entry means the default
    spec.  This is the per-leaf, lenient read the fitters and the tree
    maps use; :func:`_validate_specs_mirror` is the strict check that a
    whole ``specs`` tree reaches what it claims to.
    """
    node = specs
    for i, key in enumerate(path):
        if isinstance(node, ParamSpec):
            covers = all(isinstance(k, jax.tree_util.SequenceKey)
                         for k in path[i:])
            return node if covers else DEFAULT_SPEC
        k = _path_component(key)
        if isinstance(key, jax.tree_util.SequenceKey):
            if not isinstance(node, (list, tuple)) or not (
                    isinstance(k, int) and 0 <= k < len(node)):
                return DEFAULT_SPEC
            node = node[k]
        elif k is not None and isinstance(node, dict):
            node = node.get(k)
        else:
            return DEFAULT_SPEC
        if node is None:
            return DEFAULT_SPEC
    return node if isinstance(node, ParamSpec) else DEFAULT_SPEC


def _children(node) -> Optional[dict]:
    """``{component: child}`` for one level of a pytree node, or ``None``
    if ``node`` is a leaf.  An empty container (``{}``, ``None``) is
    not a leaf and has no children."""
    if jax.tree_util.all_leaves([node]):
        return None
    entries = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda x: x is not node)[0]
    return {_path_component(path[0]): child for path, child in entries}


def _validate_specs_mirror(params, specs) -> None:
    """Raise ``ValueError`` unless ``specs`` is a dict tree that mirrors
    ``params`` -- every key path in it either names a params leaf or is
    a nested dict on the way to one.

    :func:`_spec_for` gives a leaf the default spec whenever the walk
    to it fails, and a spec that is never reached changes nothing, so a
    caller that needs to know its ``specs`` were *read* (``fim`` under
    ``scale="nominal"``) asks this first.  Refused, each by its key
    path: a ``specs`` that is not a dict; a key that matches no
    parameter (a misspelt name is the case that must be loud); a dict
    where a leaf needs a ``ParamSpec`` (``ParamSpec.to_dict()`` output
    is the common one); a ``ParamSpec`` above a dict level; a
    list/tuple of specs whose length differs from the params sequence;
    and an entry that is none of these.  A leaf *without* an entry is
    not refused -- it gets the default spec, and ``{}`` remains the
    explicit "no leaf has a declared width".
    """
    if not isinstance(specs, dict):
        raise ValueError(
            f"specs must be a dict of ParamSpec mirroring params -- "
            f"gm.param_specs() for a graph's tree, or "
            f"{{'k': ParamSpec(...)}} for a flat one -- got "
            f"{type(specs).__name__}")

    def describe(param_node) -> str:
        ch = _children(param_node)
        if ch is None:
            return "a leaf"
        if not ch:
            return "an empty container"
        kind = ("a sequence of" if isinstance(param_node, (list, tuple))
                else "a dict with keys")
        return f"{kind} {list(ch)!r}"

    def walk(spec_node, param_node, where: str) -> None:
        children = _children(param_node)
        if isinstance(spec_node, ParamSpec):
            for path, _ in jax.tree_util.tree_flatten_with_path(param_node)[0]:
                if not all(isinstance(k, jax.tree_util.SequenceKey)
                           for k in path):
                    raise ValueError(
                        f"specs{where} is a ParamSpec but params{where} is "
                        f"{describe(param_node)}: a ParamSpec belongs at a "
                        f"leaf (one may cover a list/tuple of leaves); use a "
                        f"nested dict with an entry per parameter here")
            return
        if isinstance(spec_node, dict):
            if children is None:
                hint = ""
                if set(spec_node) <= set(ParamSpec.__dataclass_fields__):
                    hint = (" (this looks like ParamSpec.to_dict() output; "
                            "pass ParamSpec.from_dict(...) instead)")
                raise ValueError(
                    f"specs{where} is a dict but params{where} is a leaf, "
                    f"whose entry must be a ParamSpec{hint}")
            if isinstance(param_node, (list, tuple)):
                raise ValueError(
                    f"specs{where} is a dict but params{where} is "
                    f"{describe(param_node)}: give a list/tuple of ParamSpec "
                    f"read by position, or one ParamSpec covering every "
                    f"position")
            for k, v in spec_node.items():
                if k not in children:
                    raise ValueError(
                        f"specs{where}[{k!r}] matches no parameter: "
                        f"params{where} is {describe(param_node)}. A spec "
                        f"that reaches nothing changes nothing, so it is "
                        f"refused rather than ignored; drop the entry or fix "
                        f"the key")
                walk(v, children[k], f"{where}[{k!r}]")
            return
        if isinstance(spec_node, (list, tuple)):
            if not isinstance(param_node, (list, tuple)):
                raise ValueError(
                    f"specs{where} is a {type(spec_node).__name__} but "
                    f"params{where} is {describe(param_node)}: a sequence of "
                    f"specs is read by position and needs a list/tuple of "
                    f"parameters")
            if len(spec_node) != len(param_node):
                raise ValueError(
                    f"specs{where} has {len(spec_node)} entries but "
                    f"params{where} has {len(param_node)} positions")
            for i, (sv, pv) in enumerate(zip(spec_node, param_node)):
                walk(sv, pv, f"{where}[{i}]")
            return
        raise ValueError(
            f"specs{where} is {type(spec_node).__name__}; expected a "
            f"ParamSpec or a nested dict of them")

    walk(specs, params, "")


def _map_with_specs(fn, params: dict, specs: dict):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    out = [fn(_spec_for(specs, path), leaf) for path, leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, out)


@stability(StabilityLevel.EVOLVING)
def trainable_mask(params: dict, specs: dict) -> dict:
    """``params``-shaped pytree of Python bools from the specs."""
    return _map_with_specs(lambda s, _: s.trainable, params, specs)


@stability(StabilityLevel.EVOLVING)
def unconstrain(params: dict, specs: dict) -> dict:
    """Map every trainable leaf to optimiser coordinates."""
    return _map_with_specs(
        lambda s, p: s.to_unconstrained(p) if s.trainable else p, params, specs,
    )


@stability(StabilityLevel.EVOLVING)
def constrain(u: dict, specs: dict) -> dict:
    """Inverse of :func:`unconstrain` (with clipping for bounded identity
    leaves); non-trainable leaves pass through."""
    return _map_with_specs(
        lambda s, x: s.to_constrained(x) if s.trainable else x, u, specs,
    )


@stability(StabilityLevel.EVOLVING)
def check_bounds(params: dict, specs: dict) -> None:
    """Raise ``ValueError`` naming the first leaf outside its bounds."""
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        _spec_for(specs, path).check(leaf, name=jax.tree_util.keystr(path))
