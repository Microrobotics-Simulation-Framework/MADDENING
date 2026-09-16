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

    def to_unconstrained(self, p):
        p = jnp.asarray(p)
        if self.transform == "log":
            return jnp.log(p - self._lo())
        if self.transform == "logit":
            lo, hi = (float(b) for b in self.bounds)
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
            lo, hi = (float(b) for b in self.bounds)
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
            p = lo + (hi - lo) * jax.nn.sigmoid(u)
            return jnp.clip(p, lo + m, hi - m)
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


def _spec_for(specs: dict, path) -> ParamSpec:
    """``specs`` mirrors the params tree as nested dicts of ParamSpec; a
    missing entry means the default spec."""
    node = specs
    for key in path:
        k = getattr(key, "key", None)
        if k is None or not isinstance(node, dict):
            return DEFAULT_SPEC
        node = node.get(k)
        if node is None:
            return DEFAULT_SPEC
    return node if isinstance(node, ParamSpec) else DEFAULT_SPEC


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
