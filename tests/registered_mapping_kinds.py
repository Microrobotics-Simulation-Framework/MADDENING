"""Mapping kinds registered the way another library registers them.

Importing this module registers three kinds through the public
``register_mapping``, for the rest of the process -- as importing a
library that ships a mapping does.  They are written against the public
API only (the ``Mapping`` protocol, ``MappingSpec``,
``reference_for_array``), so what the existing harnesses prove of them is
what they prove of a third party's kind:

``inverse_distance``
    A mapping class of its own with **two** weights of different rank --
    a matrix ``W`` and a scalar ``gain`` -- and one hyper-parameter of
    each declarable type (a real, a bool, a string and an integer).  This
    is the kind the property strategies draw
    (``tests/property/strategies.py``) and the parametrised harnesses run.
``linear_1d``
    Returns the library's ``StaticLinearMapping`` (one weight, ``H``), the
    short way to add a kind; its reference keywords are the default
    ``<array>_ref``.
``selection``
    A mapping with **no** weights and no hyper-parameters (a gather), whose
    entry of ``gm.params["mappings"]`` is ``{}``.

:data:`KINDS` describes each for a parametrised test: the factory, the
weights it exposes and a set of valid hyper-parameters.

:func:`temporary_kind` registers a kind for one ``with`` block and removes
it again (through the registry's private test hook: nothing public
unregisters).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import StaticLinearMapping, register_mapping
from maddening.core.coupling.mapping_registry import _unregister
from maddening.core.coupling.mapping_spec import MappingSpec, reference_for_array

INVERSE_DISTANCE = "inverse_distance"
LINEAR_1D = "linear_1d"
SELECTION = "selection"

_MODES = ("consistent", "conservative")


def _as_points(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))


# ---------------------------------------------------------------------------
# inverse_distance: a mapping class of its own, two weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class InverseDistanceMapping:
    """``target = gain * (W @ source)``.

    ``W`` (``n_target x n_source``) and the scalar ``gain`` are the
    weights; both live in ``params["mappings"]`` and both are read from
    ``weights`` when the graph passes them back.
    """

    W: Any
    gain: Any
    mode: str
    spec: Optional[MappingSpec]
    kind: str = INVERSE_DISTANCE

    @property
    def n_source(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_target(self) -> int:
        return int(self.W.shape[0])

    def params_pytree(self) -> dict:
        return {"W": self.W, "gain": self.gain}

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        w = self.params_pytree() if weights is None else weights
        return w["gain"] * (w["W"] @ field)

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        w = self.params_pytree() if weights is None else weights
        return w["gain"] * (w["W"].T @ field)

    def __repr__(self) -> str:
        return f"InverseDistanceMapping({self.mode}, {self.n_target}x{self.n_source})"


def _shepard(source: np.ndarray, target: np.ndarray, power: float,
             normalise: bool, neighbours: int) -> np.ndarray:
    """Bounded inverse-distance weights ``1 / (1 + r**power)``, each row
    scaled to sum to one when ``normalise``.  With ``neighbours > 0`` only
    that many nearest sources of each target keep a weight (the lower
    index wins a tie)."""
    w = 1.0 / (1.0 + _distances(target, source) ** power)
    if 0 < neighbours < w.shape[1]:
        nearest = np.argsort(-w, axis=1, kind="stable")[:, :neighbours]
        kept = np.zeros_like(w)
        np.put_along_axis(kept, nearest, np.take_along_axis(w, nearest, axis=1), axis=1)
        w = kept
    return w / np.sum(w, axis=1, keepdims=True) if normalise else w


@register_mapping(
    INVERSE_DISTANCE,
    arrays=("source_points", "target_points"),
    hyperparameters={"power": float, "normalise": bool, "mode": str, "neighbours": int},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
def inverse_distance_mapping(source_points, target_points, *, power: float = 2.0,
                             normalise: bool = True, mode: str = "consistent",
                             neighbours: int = 0,
                             source_ref=None, target_ref=None) -> InverseDistanceMapping:
    """Inverse-distance (Shepard) weights between two point sets, from every
    source or from each target's ``neighbours`` nearest ones."""
    if mode not in _MODES:
        raise ValueError(f"mode={mode!r} not in {_MODES}")
    if not power > 0:
        raise ValueError(f"power must be positive, got {power!r}")
    if neighbours < 0:
        raise ValueError(f"neighbours must not be negative, got {neighbours!r}")
    src, tgt = _as_points(source_points), _as_points(target_points)
    if mode == "consistent":
        W = _shepard(src, tgt, power, normalise, neighbours)
    else:
        W = _shepard(tgt, src, power, normalise, neighbours).T
    spec = MappingSpec(
        INVERSE_DISTANCE,
        {"power": float(power), "normalise": bool(normalise), "mode": mode,
         "neighbours": int(neighbours)},
        {"source_points": reference_for_array(source_points, source_ref,
                                              name="source_points"),
         "target_points": reference_for_array(target_points, target_ref,
                                              name="target_points")},
    )
    dtype = jnp.result_type(float)
    return InverseDistanceMapping(W=jnp.asarray(W, dtype=dtype),
                                  gain=jnp.asarray(1.0, dtype=dtype), mode=mode, spec=spec)


# ---------------------------------------------------------------------------
# linear_1d: the library's StaticLinearMapping, default reference keywords
# ---------------------------------------------------------------------------

@register_mapping(
    LINEAR_1D,
    arrays=("source_points", "target_points"),
    hyperparameters={"clamp": bool},
)
def linear_1d_mapping(source_points, target_points, *, clamp: bool = True,
                      source_points_ref=None, target_points_ref=None) -> StaticLinearMapping:
    """Piecewise-linear interpolation between two increasing 1-D point
    sets; outside the source range it holds the end value (``clamp``) or
    extrapolates the end segment."""
    xs = np.asarray(source_points, dtype=np.float64).ravel()
    xt = np.asarray(target_points, dtype=np.float64).ravel()
    if xs.size < 2 or not np.all(np.diff(xs) > 0):
        raise ValueError("linear_1d needs at least two increasing source points")
    H = np.zeros((xt.size, xs.size))
    for i, x in enumerate(xt):
        j = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2))
        t = (x - xs[j]) / (xs[j + 1] - xs[j])
        if clamp:
            t = min(max(t, 0.0), 1.0)
        H[i, j] += 1.0 - t
        H[i, j + 1] += t
    spec = MappingSpec(LINEAR_1D, {"clamp": bool(clamp)}, {
        "source_points": reference_for_array(source_points, source_points_ref,
                                             name="source_points"),
        "target_points": reference_for_array(target_points, target_points_ref,
                                             name="target_points"),
    })
    return StaticLinearMapping(jnp.asarray(H, dtype=jnp.result_type(float)),
                               kind=LINEAR_1D, spec=spec)


# ---------------------------------------------------------------------------
# selection: no weights at all
# ---------------------------------------------------------------------------

class SelectionMapping:
    """``target[i] = source[index[i]]``: a gather, with nothing to train.

    The index array is structure, not a parameter: it stays an attribute
    and ``params_pytree()`` is empty.
    """

    kind = SELECTION
    mode = "consistent"

    def __init__(self, index: np.ndarray, n_source: int, spec: Optional[MappingSpec]):
        self.index = jnp.asarray(index)
        self._n_source = int(n_source)
        self.spec = spec

    @property
    def n_source(self) -> int:
        return self._n_source

    @property
    def n_target(self) -> int:
        return int(self.index.shape[0])

    def params_pytree(self) -> dict:
        return {}

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        return field[self.index]

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        shape = (self._n_source,) + tuple(jnp.shape(field)[1:])
        return jnp.zeros(shape, jnp.result_type(field)).at[self.index].add(field)

    def __repr__(self) -> str:
        return f"SelectionMapping({self.n_target}<-{self.n_source})"


@register_mapping(
    SELECTION,
    arrays=("source_points", "target_points"),
    hyperparameters={},
    references={"source_points": "source_ref", "target_points": "target_ref"},
)
def selection_mapping(source_points, target_points, *, source_ref=None,
                      target_ref=None) -> SelectionMapping:
    """Each target point takes the value at its nearest source point."""
    src, tgt = _as_points(source_points), _as_points(target_points)
    index = np.argmin(_distances(tgt, src), axis=1).astype(np.int32)
    spec = MappingSpec(SELECTION, {}, {
        "source_points": reference_for_array(source_points, source_ref,
                                             name="source_points"),
        "target_points": reference_for_array(target_points, target_ref,
                                             name="target_points"),
    })
    return SelectionMapping(index, src.shape[0], spec)


# ---------------------------------------------------------------------------
# What a parametrised harness needs to know of each kind
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisteredKind:
    """One registered kind, as a parametrised test sees it.

    ``build(source, target, source_ref=None, target_ref=None, **hyper)``
    calls the factory with the two references under whatever keywords the
    kind declared, so a harness written for the built-in
    ``source_ref=`` / ``target_ref=`` spelling runs every kind.
    """

    kind: str
    factory: Callable[..., Any]
    weights: tuple[str, ...]
    source_ref: str
    target_ref: str
    #: Valid hyper-parameters to draw, as ``{name: values}``.
    hyper: dict

    def build(self, source, target, *, source_ref=None, target_ref=None, **hyper):
        return self.factory(source, target, **hyper,
                            **{self.source_ref: source_ref, self.target_ref: target_ref})

    def __repr__(self) -> str:  # the pytest id
        return self.kind


KINDS: dict[str, RegisteredKind] = {
    INVERSE_DISTANCE: RegisteredKind(
        INVERSE_DISTANCE, inverse_distance_mapping, ("W", "gain"),
        "source_ref", "target_ref",
        {"power": (1.0, 3.5), "normalise": (True, False), "mode": _MODES,
         "neighbours": (0, 2)}),
    LINEAR_1D: RegisteredKind(
        LINEAR_1D, linear_1d_mapping, ("H",),
        "source_points_ref", "target_points_ref", {"clamp": (True, False)}),
    SELECTION: RegisteredKind(
        SELECTION, selection_mapping, (), "source_ref", "target_ref", {}),
}


def weights_of(mapping) -> dict[str, np.ndarray]:
    """A mapping's own weights as NumPy arrays (``{}`` for none)."""
    return {name: np.asarray(leaf) for name, leaf in mapping.params_pytree().items()}


def assert_same_weights(got, expected, *, what: str = "weights") -> None:
    """Two weight tables are equal bit for bit: names, dtypes, shapes, bytes."""
    got = {k: np.asarray(v) for k, v in got.items()}
    expected = {k: np.asarray(v) for k, v in expected.items()}
    assert list(got) == list(expected), f"{what}: {list(got)} != {list(expected)}"
    for name, leaf in expected.items():
        assert got[name].dtype == leaf.dtype, f"{what}[{name!r}] dtype"
        assert got[name].shape == leaf.shape, f"{what}[{name!r}] shape"
        assert got[name].tobytes() == leaf.tobytes(), f"{what}[{name!r}] differs"


@contextlib.contextmanager
def temporary_kind(kind: str, factory: Callable[..., Any], *, arrays,
                   hyperparameters, references=None):
    """Register *factory* as *kind* for the ``with`` block, then remove it."""
    register_mapping(kind, arrays=arrays, hyperparameters=hyperparameters,
                     references=references)(factory)
    try:
        yield factory
    finally:
        _unregister(kind)
