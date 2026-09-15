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

_TRANSFORMS = (None, "log", "logit")


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
        lo, hi = d.get("bounds", (None, None))
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
        # trip stays finite for any optimiser coordinates.
        if self.transform == "log":
            fi = jnp.finfo(u.dtype if jnp.issubdtype(u.dtype, jnp.floating) else jnp.float32)
            return self._lo() + jnp.clip(jnp.exp(u), fi.tiny, fi.max)
        if self.transform == "logit":
            lo, hi = (float(b) for b in self.bounds)
            eps = jnp.finfo(u.dtype).eps if jnp.issubdtype(u.dtype, jnp.floating) else 1e-7
            t = jnp.clip(jax.nn.sigmoid(u), eps, 1.0 - eps)
            return lo + (hi - lo) * t
        lo, hi = self.bounds
        if lo is not None or hi is not None:
            return jnp.clip(u, lo, hi)
        return u

    def check(self, p, *, name: str = "param") -> None:
        """Raise ``ValueError`` if a concrete value violates ``bounds``."""
        lo, hi = self.bounds
        v = jnp.asarray(p)
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


def trainable_mask(params: dict, specs: dict) -> dict:
    """``params``-shaped pytree of Python bools from the specs."""
    return _map_with_specs(lambda s, _: s.trainable, params, specs)


def unconstrain(params: dict, specs: dict) -> dict:
    """Map every trainable leaf to optimiser coordinates."""
    return _map_with_specs(
        lambda s, p: s.to_unconstrained(p) if s.trainable else p, params, specs,
    )


def constrain(u: dict, specs: dict) -> dict:
    """Inverse of :func:`unconstrain` (with clipping for bounded identity
    leaves); non-trainable leaves pass through."""
    return _map_with_specs(
        lambda s, x: s.to_constrained(x) if s.trainable else x, u, specs,
    )


def check_bounds(params: dict, specs: dict) -> None:
    """Raise ``ValueError`` naming the first leaf outside its bounds."""
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        _spec_for(specs, path).check(leaf, name=jax.tree_util.keystr(path))
