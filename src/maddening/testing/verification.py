"""Property-based verification harness for SimulationNode subclasses.

Runs a battery of universal checks against a node's ``update()`` by
sampling inputs with Hypothesis and shrinking any failure to a minimal
counterexample.  The checks are the ones every physics node should
satisfy regardless of what it models: finite outputs, preserved state
structure, determinism, JIT/eager agreement, finite gradients, and —
for nodes that take the graph's ``params`` pytree — injected params
reproducing the baked constants and finite gradients with respect to
them.

Usage::

    from maddening.testing.verification import verify_node

    node = MyPhysicsNode(name="test", timestep=0.01)
    results = verify_node(node, bounds={"temperature": (200.0, 5000.0)})
    for name, r in results.items():
        print(name, r.status, r.counterexample if r.failed else "")
    assert all(r.passed for r in results.values())

Each check is also exposed as a standalone function (``node_finite``,
``node_gradient_finite``, ...) for finer control, and
:func:`node_invariant` accepts an arbitrary predicate.

Requires ``hypothesis >= 6.165``. Install via::

    pip install maddening[verify]
"""

from __future__ import annotations

import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp

import numpy as np
import numpy.typing as npt

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import _method_accepts_params
from maddening.core.compliance.stability import stability
from maddening.testing.strategies import (
    boundary_inputs_for,
    bounded_dt,
    node_states,
)

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    from hypothesis.errors import HypothesisException
except ImportError as e:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "hypothesis is required for property-based verification. "
        "Install it with: pip install maddening[verify]"
    ) from e


Bounds = dict[str, tuple[float, float]]


@stability(StabilityLevel.EXPERIMENTAL)
@dataclass
class VerificationResult:
    """Outcome of one check.

    ``status`` is ``"PASS"`` (no counterexample found in ``n_examples``
    samples), ``"FAIL"`` (a shrunk counterexample is in
    ``counterexample``), ``"ERROR"`` (the check itself could not run,
    e.g. ``update()`` is not differentiable; see ``detail``), or
    ``"SKIP"`` (the check does not apply to this node, e.g. the params
    checks on a node whose ``update`` takes no ``params``).  ``SKIP``
    counts as passed.
    """
    name: str
    status: str
    detail: str = ""
    counterexample: dict[str, Any] = field(default_factory=dict)
    n_examples: int = 0

    @property
    def passed(self) -> bool:
        return self.status in ("PASS", "SKIP")

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    @property
    def skipped(self) -> bool:
        return self.status == "SKIP"

    def __str__(self) -> str:
        head = f"{self.name}: {self.status}"
        if self.failed:
            return f"{head}\n  counterexample: {self.counterexample}\n  {self.detail}"
        if self.status in ("ERROR", "SKIP"):
            return f"{head}\n  {self.detail}"
        return f"{head} ({self.n_examples} examples)"


@dataclass(frozen=True)
class _Inputs:
    node: Any
    bounds: Bounds
    boundary_bounds: Bounds
    boundary_inputs: dict | None
    dt_range: tuple[float, float]
    dtype: np.dtype

    def strategy(self) -> st.SearchStrategy:
        if self.boundary_inputs is not None:
            bi = st.just(self.boundary_inputs)
        else:
            bi = boundary_inputs_for(
                self.node, self.boundary_bounds, dtype=self.dtype,
            )
        return st.tuples(
            node_states(self.node, self.bounds, dtype=self.dtype),
            bi,
            bounded_dt(*self.dt_range),
        )


def _run(
    name: str,
    inputs: _Inputs,
    body: Callable[[dict, dict, float], None],
    *,
    max_examples: int,
    derandomize: bool,
) -> VerificationResult:
    """Drive ``body`` under Hypothesis and package the outcome.

    Hypothesis replays the minimal failing example last, so the inputs
    recorded on the final call are the shrunk counterexample.
    """
    last: dict[str, Any] = {}
    count = 0

    @settings(
        max_examples=max_examples,
        deadline=None,
        database=None,
        derandomize=derandomize,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(inputs.strategy())
    def prop(args):
        nonlocal count
        state, bi, dt = args
        count += 1
        last.clear()
        last.update(state=state, boundary_inputs=bi, dt=dt)
        body(state, bi, dt)

    try:
        prop()
    except AssertionError as e:
        return VerificationResult(
            name, "FAIL", detail=str(e) or "assertion failed",
            counterexample=dict(last), n_examples=count,
        )
    except HypothesisException as e:
        return VerificationResult(name, "ERROR", detail=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001 - update() raised on some input
        return VerificationResult(
            name, "FAIL",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
            counterexample=dict(last), n_examples=count,
        )
    return VerificationResult(name, "PASS", n_examples=count)


def _to_np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def node_finite(inputs: _Inputs, **kw) -> VerificationResult:
    """Every output field is finite for every sampled input."""
    def body(state, bi, dt):
        out = inputs.node.update(state, bi, dt)
        for f, v in out.items():
            assert bool(jnp.all(jnp.isfinite(v))), f"non-finite value in '{f}'"
    return _run("finite", inputs, body, **kw)


def node_structure(inputs: _Inputs, **kw) -> VerificationResult:
    """Output keys, shapes and dtypes match the input state."""
    def body(state, bi, dt):
        out = inputs.node.update(state, bi, dt)
        assert set(out) == set(state), (
            f"key mismatch: {sorted(set(out) ^ set(state))}"
        )
        for f in state:
            assert out[f].shape == state[f].shape, (
                f"shape mismatch in '{f}': {out[f].shape} != {state[f].shape}"
            )
            assert out[f].dtype == state[f].dtype, (
                f"dtype mismatch in '{f}': {out[f].dtype} != {state[f].dtype}"
            )
    return _run("structure", inputs, body, **kw)


def node_deterministic(inputs: _Inputs, **kw) -> VerificationResult:
    """Two eager calls on identical inputs are bit-identical."""
    def body(state, bi, dt):
        a = inputs.node.update(state, bi, dt)
        b = inputs.node.update(state, bi, dt)
        for f in a:
            assert np.array_equal(_to_np(a[f]), _to_np(b[f]), equal_nan=True), (
                f"non-deterministic output in '{f}'"
            )
    return _run("deterministic", inputs, body, **kw)


def node_jit_consistent(inputs: _Inputs, rtol: float = 1e-5, atol: float = 1e-6, **kw):
    """``jax.jit(update)`` agrees with the eager call.

    Disagreement usually means Python-side branching on array values,
    a host-side side effect, or a dtype that changes under tracing.
    """
    jitted = jax.jit(inputs.node.update)

    def body(state, bi, dt):
        a = inputs.node.update(state, bi, dt)
        b = jitted(state, bi, dt)
        for f in a:
            x, y = _to_np(a[f]), _to_np(b[f])
            assert np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True), (
                f"jit/eager mismatch in '{f}': max abs diff "
                f"{np.max(np.abs(x - y))}"
            )
    return _run("jit_consistent", inputs, body, **kw)


def node_gradient_finite(inputs: _Inputs, **kw) -> VerificationResult:
    """``d(sum of outputs)/d(state)`` is finite for every float field.

    This is the check that catches backward-pass-only failures: NaN
    from ``lstsq``/``solve`` VJPs on rank-deficient inputs, ``sqrt`` or
    ``norm`` at zero, ``jnp.where`` guards that protect the forward but
    not the gradient.
    """
    node = inputs.node

    def body(state, bi, dt):
        diff = {f: v for f, v in state.items() if jnp.issubdtype(v.dtype, jnp.floating)}
        if not diff:
            return
        rest = {f: v for f, v in state.items() if f not in diff}

        def loss(d):
            out = node.update({**rest, **d}, bi, dt)
            return sum(jnp.sum(v) for v in out.values()
                       if jnp.issubdtype(v.dtype, jnp.floating))

        g = jax.grad(loss)(diff)
        for f, v in g.items():
            assert bool(jnp.all(jnp.isfinite(v))), f"non-finite gradient wrt '{f}'"
    return _run("gradient_finite", inputs, body, **kw)


_NO_PARAMS = object()


def _produces_fluxes(node) -> bool:
    return _implements(node, "compute_boundary_fluxes")


def _flux_accepts_params(node) -> bool:
    """The graph's question, through the one params rule."""
    return _method_accepts_params(node, "compute_boundary_fluxes")


def _outputs(node, state, bi, dt, params=_NO_PARAMS):
    """``update`` outputs plus (namespaced) boundary fluxes, both under
    the same params contract, so the params checks see what a flux edge
    delivers, not only what the node integrates."""
    if params is _NO_PARAMS:
        out = dict(node.update(state, bi, dt))
        if _produces_fluxes(node):
            out.update({f"flux:{k}": v for k, v in node.compute_boundary_fluxes(state, bi, dt).items()})
        return out
    out = dict(node.update(state, bi, dt, params=params))
    if _produces_fluxes(node):
        if _flux_accepts_params(node):
            fl = node.compute_boundary_fluxes(state, bi, dt, params=params)
        else:
            fl = node.compute_boundary_fluxes(state, bi, dt)
        out.update({f"flux:{k}": v for k, v in fl.items()})
    return out


def _node_accepts_params(node) -> bool:
    """Does the graph pass ``params=`` to ``node.update``?

    The graph's own rule, :func:`~maddening.core.node._method_accepts_params`.
    This used to answer ``False`` for any node object without an
    ``accepts_params`` method, so a duck-typed node with a ``params``
    keyword -- which the graph injects -- had every params check
    reported as ``SKIP``, and ``SKIP`` counts as passed.
    """
    return _method_accepts_params(node, "update")


def _implements(node, method: str) -> bool:
    """Does ``node`` have ``method`` other than the base-class default?

    Read off the bound attribute, not the class, so a duck-typed node
    (no base class to compare against) and a hook installed as an
    instance attribute (a ``functools.partial``, a callable object) are
    seen the way the graph sees them.
    """
    from maddening.core.node import SimulationNode as _Base  # noqa: PLC0415
    fn = getattr(node, method, None)
    if fn is None:
        return False
    return getattr(fn, "__func__", fn) is not getattr(_Base, method, None)


def _same(a, b) -> bool:
    x, y = _to_np(a), _to_np(b)
    try:
        return bool(np.array_equal(x, y, equal_nan=True))
    except TypeError:  # a dtype isnan cannot take (bool, object)
        return bool(np.array_equal(x, y))


def _close(a, b, rtol: float, atol: float) -> bool:
    """``allclose`` for floating fields, exact equality for the rest."""
    x, y = _to_np(a), _to_np(b)
    if x.shape != y.shape:
        return False
    if np.issubdtype(x.dtype, np.inexact) or np.issubdtype(y.dtype, np.inexact):
        return bool(np.allclose(x.astype(np.float64), y.astype(np.float64),
                                rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(x, y))


def _all_same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(_same(a[f], b[f]) for f in a)


def _all_close(a: dict, b: dict, rtol: float, atol: float) -> bool:
    return set(a) == set(b) and all(_close(a[f], b[f], rtol, atol) for f in a)


def _max_diff(a: dict, b: dict) -> float:
    out = 0.0
    for f in set(a) & set(b):
        x = _to_np(a[f]).astype(np.float64, copy=False)
        y = _to_np(b[f]).astype(np.float64, copy=False)
        if x.shape == y.shape and x.size:
            out = max(out, float(np.max(np.abs(x - y))))
    return out


def _missing_pytree(name: str, node) -> VerificationResult | None:
    """A node object the graph injects (its ``update`` takes ``params``) but
    that has no ``params_pytree()`` to build its ``gm.params`` entry from."""
    if callable(getattr(node, "params_pytree", None)):
        return None
    return VerificationResult(
        name, "FAIL",
        detail=(
            f"update() takes params but {type(node).__name__} has no "
            "params_pytree(): the graph cannot build this node's entry of "
            "gm.params.  Subclass SimulationNode or define params_pytree()."
        ),
    )


# ---------------------------------------------------------------------------
# params_effective: the paths, the perturbations and the references
# ---------------------------------------------------------------------------

#: Largest number of elements of one leaf the value probes perturb one at a
#: time; a larger leaf gets this many, evenly spaced.  A vector leaf is
#: probed per element because "some element acts" is not "every element
#: acts": a node reading ``g[0]`` from the injected params and ``g[1:]``
#: from ``self.params`` has a non-zero gradient wrt ``g``.
_MAX_PROBED_ELEMENTS = 8
#: The value probes (two extra evaluations per path and element) run on the
#: first this-many examples; the gradient screen runs on all of them.
_VALUE_PROBE_EXAMPLES = 20
#: A (path, element) pair seen consistent, with a visible effect, on this
#: many examples is not probed again.
_CONFIRMATIONS = 3
#: Examples kept for the post-run rebuild probe.
_KEPT_EXAMPLES = 8


def _probe_indices(size: int) -> list[int]:
    if size <= _MAX_PROBED_ELEMENTS:
        return list(range(size))
    return sorted({int(i) for i in np.linspace(0, size - 1, _MAX_PROBED_ELEMENTS)})


def _element_label(key: str, shape: tuple, idx: int) -> str:
    if not shape:
        return key
    return f"{key}[{', '.join(str(int(i)) for i in np.unravel_index(idx, shape))}]"


def _perturbed_scalar(old: float, spec) -> float:
    """A visibly different value inside the leaf's ``ParamSpec`` bounds:
    x1.5, else x0.5 (0.5 / -0.5 at zero), else half-way to a bound."""
    lo, hi = spec.bounds if spec is not None else (None, None)

    def inside(v):
        return v != old and (lo is None or v > lo) and (hi is None or v < hi)

    for v in ((old * 1.5, old * 0.5) if old != 0 else (0.5, -0.5)):
        if inside(v):
            return v
    if hi is not None and old < hi:
        return old + (hi - old) / 2
    if lo is not None and old > lo:
        return old - (old - lo) / 2
    return old + 1.0


def _as_constructor_value(old, new: np.ndarray):
    """``new`` in the container type of the constructor value ``old``, or
    ``None`` when there is no faithful spelling (the reference is then
    unavailable for this leaf).  Built from the *float32* leaf, so the
    constructed and the injected value are the same number."""
    if isinstance(old, bool):
        return None
    if isinstance(old, (int, float)) and not isinstance(old, np.generic):
        return float(new) if new.ndim == 0 else None
    if isinstance(old, (list, tuple)):
        return type(old)(np.asarray(new, dtype=np.float64).tolist())
    if isinstance(old, np.generic):
        kind = old.dtype if np.issubdtype(old.dtype, np.floating) else np.float64
        return np.asarray(new, dtype=kind)[()]
    if isinstance(old, np.ndarray):
        kind = old.dtype if np.issubdtype(old.dtype, np.floating) else np.float64
        return np.asarray(new, dtype=kind).reshape(old.shape)
    if hasattr(old, "dtype") and hasattr(old, "shape"):
        kind = old.dtype if jnp.issubdtype(old.dtype, jnp.floating) else jnp.zeros(()).dtype
        return jnp.asarray(new, dtype=kind).reshape(old.shape)
    return None


def _rebuilder(node) -> Callable[[dict], Any] | None:
    """``overrides -> type(node)(name, timestep, **{**params, **overrides})``
    from ``node.to_dict()`` -- the rebuild the config round trip already
    relies on -- or ``None`` when the node does not describe itself that
    way.  Whether the rebuild is *faithful* is checked against the node
    before it is trusted (see :func:`node_params_effective`)."""
    to_dict = getattr(node, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        d = to_dict()
        name, timestep, params = d["name"], d["timestep"], dict(d["params"])
    except Exception:  # noqa: BLE001 - no usable self-description
        return None
    cls = type(node)

    def build(overrides: dict):
        return cls(name, timestep, **{**params, **overrides})
    return build


def _flatten_corrections(out) -> dict:
    return {
        f"{field}@{idx}": value
        for field, pairs in (out or {}).items()
        for idx, value in pairs
    }


def _path_functions(node) -> dict[str, Callable]:
    """``name -> fn(target, state, bi, dt, params_or_None) -> dict``, for the
    paths through which the graph and the integrators read the node's
    constants: :meth:`update`, :meth:`compute_boundary_fluxes` (what a flux
    edge delivers), :meth:`derivatives` and :meth:`implicit_residual`
    (``integrate_node`` / ``implicit_euler_step``) and
    :meth:`compute_interface_correction` (a coupled interface).  Each is its
    own path: a summed objective let one path's sensitivity mask another's
    dead read."""
    def call(method, *args, p, target):
        fn = getattr(target, method)
        return fn(*args) if p is None else fn(*args, params=p)

    paths: dict[str, Callable] = {
        "update": lambda t, s, bi, dt, p: dict(call("update", s, bi, dt, p=p, target=t)),
    }
    if _produces_fluxes(node):
        paths["compute_boundary_fluxes"] = lambda t, s, bi, dt, p: dict(
            call("compute_boundary_fluxes", s, bi, dt, p=p, target=t))
    if _implements(node, "derivatives"):
        paths["derivatives"] = lambda t, s, bi, dt, p: dict(
            call("derivatives", s, bi, p=p, target=t))
    if _implements(node, "implicit_residual"):
        # x_new = x_old = state: R = -dt * f(state; params), whose
        # sensitivity to a constant is dt times f's.
        paths["implicit_residual"] = lambda t, s, bi, dt, p: dict(
            call("implicit_residual", s, s, bi, dt, p=p, target=t))
    iface = getattr(node, "interface_dof_indices", None)
    if _implements(node, "compute_interface_correction") and callable(iface) and iface():
        paths["compute_interface_correction"] = lambda t, s, bi, dt, p: _flatten_corrections(
            call("compute_interface_correction", s, bi, dt, p=p, target=t))
    return paths


#: What each path's missing ``params`` keyword costs, for the refusal.
_WITHOUT_KEYWORD = {
    "derivatives": ("integrate_node / implicit_euler_step refuse a calibrated "
                    "params for this node rather than integrate the "
                    "constructor's constants while update() uses the "
                    "calibrated ones"),
    "implicit_residual": ("implicit_euler_step refuses a calibrated params for "
                          "this node rather than solve with the constructor's "
                          "constants while update() uses the calibrated ones"),
    "compute_boundary_fluxes": ("a calibrated constant would change the node's "
                                "integration but not the flux it delivers over "
                                "an edge"),
    "compute_interface_correction": ("a coupled interface would be corrected "
                                     "with the constructor's constants while "
                                     "update() uses the calibrated ones"),
}


def _skip_no_params(name: str) -> VerificationResult:
    return VerificationResult(
        name, "SKIP",
        detail="update() does not take a params keyword (constants are "
               "baked into the trace; see SimulationNode.params_pytree)",
    )


def node_params_consistent(
    inputs: _Inputs, rtol: float = 1e-5, atol: float = 1e-6, **kw,
) -> VerificationResult:
    """Injected params reproduce the baked-constant step.

    ``update(state, bi, dt, params=node.params_pytree())`` must agree
    with ``update(state, bi, dt)`` to float32 round-off.  The graph runs
    the first form (traced, differentiable constants); a node that
    reads a constant from ``self.params`` on one path and from
    ``params`` on the other, or applies it differently, silently
    calibrates the wrong model.  Round-off (not exactness) is the
    contract because a traced constant can be FMA-contracted where a
    Python float is folded.

    ``SKIP`` for nodes whose ``update`` takes no ``params``.
    """
    node = inputs.node
    if not _node_accepts_params(node):
        return _skip_no_params("params_consistent")
    if _produces_fluxes(node) and not _flux_accepts_params(node):
        return VerificationResult(
            "params_consistent", "FAIL",
            detail=(
                "update() takes params but compute_boundary_fluxes() does not: "
                "a calibrated constant would change the node's integration but "
                "not the flux it delivers over an edge.  Declare "
                "compute_boundary_fluxes(self, state, boundary_inputs, dt, *, "
                "params=None) and read constants from params."
            ),
        )
    missing = _missing_pytree("params_consistent", node)
    if missing is not None:
        return missing
    injected = node.params_pytree()

    def body(state, bi, dt):
        a = _outputs(node, state, bi, dt)
        b = _outputs(node, state, bi, dt, params=injected)
        assert set(a) == set(b), f"key mismatch: {sorted(set(a) ^ set(b))}"
        for f in a:
            x, y = _to_np(a[f]), _to_np(b[f])
            assert np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True), (
                f"baked/injected params mismatch in '{f}': max abs diff "
                f"{np.max(np.abs(x - y))}"
            )
    return _run("params_consistent", inputs, body, **kw)


def node_params_gradient_finite(inputs: _Inputs, **kw) -> VerificationResult:
    """``d(sum of outputs)/d(params_pytree)`` is finite.

    The params analogue of :func:`node_gradient_finite`: calibration and
    system identification differentiate with respect to these leaves,
    so a backward-only NaN here (``sqrt`` of a parameter at zero, a
    ``where`` guard that protects only the forward) is what breaks an
    optimiser.  ``SKIP`` for nodes whose ``update`` takes no ``params``
    or whose :meth:`params_pytree` is empty.
    """
    node = inputs.node
    if not _node_accepts_params(node):
        return _skip_no_params("params_gradient_finite")
    missing = _missing_pytree("params_gradient_finite", node)
    if missing is not None:
        return missing
    base = node.params_pytree()
    if not base:
        return VerificationResult(
            "params_gradient_finite", "SKIP", detail="params_pytree() is empty",
        )

    def body(state, bi, dt):
        def loss(p):
            out = _outputs(node, state, bi, dt, params=p)
            return sum(jnp.sum(v) for v in out.values()
                       if jnp.issubdtype(v.dtype, jnp.floating))

        g = jax.grad(loss)(base)
        for f, v in g.items():
            assert bool(jnp.all(jnp.isfinite(v))), (
                f"non-finite gradient wrt param '{f}'"
            )
    return _run("params_gradient_finite", inputs, body, **kw)


#: Seed of the fixed random projection ``params_effective`` differentiates
#: (see :func:`_projected`).  Combined with the CRC-32 of the field name so
#: two fields of the same shape get different weights.
_PROBE_SEED = 20260922


def _projection_weights(name: str, shape: tuple[int, ...]) -> np.ndarray:
    """One weight per element of output field ``name``: magnitude in
    ``[0.5, 1.5]``, random sign, fixed by ``_PROBE_SEED`` and the name."""
    rng = np.random.default_rng([_PROBE_SEED, zlib.crc32(name.encode())])
    magnitude = rng.uniform(0.5, 1.5, size=shape)
    sign = rng.choice(np.array([-1.0, 1.0]), size=shape)
    return magnitude * sign


def _projected(out: dict) -> Any:
    """``sum_f <w_f, out_f>`` over the floating fields of ``out`` -- the
    scalar whose gradient ``params_effective`` inspects.

    A plain ``sum(jnp.sum(v))`` is annihilated by any conservation law
    with equal coefficients (mass moving between two compartments at an
    injected rate has ``d(sum)/d(rate) == 0`` exactly), which made the
    check FAIL a correct node.  Per-element weights of distinct
    magnitude and random sign leave no such law standing.
    """
    total = None
    for name, v in out.items():
        if not jnp.issubdtype(v.dtype, jnp.floating):
            continue
        w = jnp.asarray(_projection_weights(name, tuple(v.shape)), dtype=v.dtype)
        term = jnp.sum(v * w)
        total = term if total is None else total + term
    return jnp.zeros(()) if total is None else total


def node_params_effective(
    inputs: _Inputs, rtol: float = 1e-5, atol: float = 1e-6, **kw,
) -> VerificationResult:
    """Every *trainable* parameter acts through the injected params -- on
    every path, and the way a constructed value acts.

    A leaf of :meth:`params_pytree` that a path reads from ``self.params``
    (or from a copy made in ``__init__``) instead of the injected
    ``params`` is a silent trap: the graph passes a value, an optimiser
    moves it, nothing changes.  ``params_consistent`` cannot see it (at
    the constructor value both spellings read the same number), so this
    check perturbs each leaf and compares.

    Paths
    -----
    Each path the graph or an integrator reads the node's constants
    through is probed **on its own**: :meth:`update`,
    :meth:`compute_boundary_fluxes` (what a flux edge delivers),
    :meth:`derivatives` and :meth:`implicit_residual` (what
    ``integrate_node`` / ``implicit_euler_step`` integrate) and
    :meth:`compute_interface_correction` (a coupled interface; probed when
    the node declares interface DOFs).  Until 0.4.0 shipped ``update`` and
    the fluxes were summed into one objective, so a flux that read the
    injected value masked an ``update`` that did not, and the interface
    correction was never probed.  A path whose override has no ``params``
    keyword is a ``FAIL``; a ``derivatives`` / ``implicit_residual`` that
    raises ``NotImplementedError`` (a discrete node) is not applicable.

    Per sample, per path
    --------------------
    * a **gradient screen**: the gradient of a fixed random projection
      of the path's outputs (see :func:`_projected`; a plain sum is
      annihilated by any equal-coefficient conservation law) with respect
      to every injected leaf, element by element;
    * a **value test**, on the first ``_VALUE_PROBE_EXAMPLES`` samples:
      each element of each trainable leaf (up to ``_MAX_PROBED_ELEMENTS``
      per leaf, evenly spaced) is moved to a perturbed value inside its
      ``ParamSpec`` bounds, once through the injected params and once
      through ``node.params`` (swapped in place and restored).  Where the
      constructed value moves the output, the injected one must move it
      to the same place (``rtol`` / ``atol`` as in
      ``params_consistent``).  A difference test rather than a gradient,
      so a leaf the path reads only through a comparison (a dead-zone
      threshold, zero gradient) is not mistaken for a ``self.params``
      read, and ``0.5 * p["k"] + 0.5 * self.params["k"]`` -- non-zero
      gradient, half the effect -- is not mistaken for a correct read.

    After the run, an element that moved nothing on a path through
    either spelling, while it acts on another path, is probed once more
    against a node **rebuilt** from ``to_dict()`` with the perturbed
    value (when the node rebuilds, and a rebuild with no override
    reproduces it): a path that moves for the rebuilt node but never for
    the injected value reads a copy made at construction.

    Verdicts
    --------
    ``FAIL`` for: a path that reads an element from ``self.params``
    (constructed value moves it, injected never does); a path that
    applies the injected value differently from a constructed one; a
    path that reads a copy made at construction (rebuild probe); a
    trainable leaf that moved nothing on any path on any sample (read
    nowhere -- or declare it ``ParamSpec(trainable=False)`` -- or the
    sampled envelope never exercises it); and, failing closed, a non-
    ``update`` path on which a leaf that acts through ``update`` never
    moves, when neither ``node.params`` nor a rebuild can provide a
    reference.  ``detail`` names the paths checked and, on a ``PASS``,
    the elements each path does not consume.  ``SKIP`` for nodes without
    ``params`` and for leaves declared ``trainable=False``.

    Limitations
    -----------
    Closed in 0.4.0: all five false ``PASS``es and the false ``FAIL`` the
    release audit planted.  Still open, and documented rather than
    guessed at: a copy made at construction is caught only when the node
    rebuilds faithfully from ``to_dict()`` (a node that cannot is
    reported "not consumed" on that path, as before); the value test runs
    on the first ``_VALUE_PROBE_EXAMPLES`` samples only, so a split read
    confined to a region of the envelope those samples miss can pass; and
    a leaf larger than ``_MAX_PROBED_ELEMENTS`` is value-probed on that
    many evenly spaced elements (the gradient screen still sees all of
    them).
    """
    node = inputs.node
    if not _node_accepts_params(node):
        return _skip_no_params("params_effective")
    missing = _missing_pytree("params_effective", node)
    if missing is not None:
        return missing
    base = node.params_pytree()
    specs = node.param_specs() if callable(getattr(node, "param_specs", None)) else {}
    trainable = [k for k in base if specs.get(k) is None or specs[k].trainable]
    if not trainable:
        return VerificationResult(
            "params_effective", "SKIP", detail="no trainable parameters",
        )

    paths = _path_functions(node)
    accepts = {name: _method_accepts_params(node, name) for name in paths if name != "update"}
    shapes = {k: tuple(np.shape(_to_np(base[k]))) for k in trainable}
    sizes = {k: int(np.prod(shapes[k], dtype=np.int64)) for k in trainable}
    elements = [(k, i) for k in trainable for i in _probe_indices(sizes[k])]
    labels = {el: _element_label(el[0], shapes[el[0]], el[1]) for el in elements}
    ctor_params = getattr(node, "params", None)
    injected_for: dict = {}
    ctor_value: dict = {}
    for k, i in elements:
        leaf = _to_np(base[k])
        flat = np.array(leaf, copy=True).reshape(-1)
        flat[i] = _perturbed_scalar(float(flat[i]), specs.get(k))
        new = flat.reshape(leaf.shape).astype(leaf.dtype)
        injected_for[(k, i)] = {**base, k: jnp.asarray(new, dtype=jnp.asarray(base[k]).dtype)}
        ctor_value[(k, i)] = (
            _as_constructor_value(ctor_params[k], new)
            if isinstance(ctor_params, dict) and k in ctor_params else None
        )

    grad_any = {name: {k: np.zeros(sizes[k], dtype=bool) for k in trainable} for name in paths}
    inj_moved: dict[str, set] = {name: set() for name in paths}
    ref_moved: dict[str, set] = {name: set() for name in paths}
    confirmed: dict[str, dict] = {name: {} for name in paths}
    mismatch: dict[str, dict] = {name: {} for name in paths}
    not_applicable: set[str] = set()
    kept: list = []
    count = 0

    def body(state, bi, dt):
        nonlocal count
        count += 1
        if len(kept) < _KEPT_EXAMPLES:
            kept.append((state, bi, dt))
        for name, fn in paths.items():
            if name in not_applicable:
                continue
            try:
                baseline = fn(node, state, bi, dt, None)
            except NotImplementedError:
                if name in ("derivatives", "implicit_residual"):
                    not_applicable.add(name)
                    continue
                raise
            if name != "update" and not accepts[name]:
                raise AssertionError(
                    f"update() takes params but {name}() does not: "
                    f"{_WITHOUT_KEYWORD[name]}.  Declare {name}(..., *, "
                    "params=None) and read constants from "
                    "{**self.params, **params}."
                )
            g = jax.grad(lambda p, fn=fn: _projected(fn(node, state, bi, dt, p)))(base)
            for k in trainable:
                grad_any[name][k] |= (_to_np(g[k]) != 0).reshape(-1)
            if count > _VALUE_PROBE_EXAMPLES:
                continue
            injected_base = fn(node, state, bi, dt, base)
            for el in elements:
                if el in mismatch[name] or confirmed[name].get(el, 0) >= _CONFIRMATIONS:
                    continue
                inj = fn(node, state, bi, dt, injected_for[el])
                moved_inj = not _all_same(inj, injected_base)
                if moved_inj:
                    inj_moved[name].add(el)
                value = ctor_value[el]
                if value is None:
                    continue
                assert isinstance(ctor_params, dict)  # value is None otherwise
                key = el[0]
                old = ctor_params[key]
                ctor_params[key] = value
                try:
                    try:
                        ref = fn(node, state, bi, dt, None)
                    except Exception:  # noqa: BLE001 - a different value broke it: it reads it
                        ref = None
                finally:
                    ctor_params[key] = old
                if ref is None:
                    ref_moved[name].add(el)
                    if not moved_inj:
                        mismatch[name][el] = (float("nan"), float("nan"), False)
                    continue
                if _all_same(ref, baseline):
                    continue
                ref_moved[name].add(el)
                if _all_close(inj, ref, rtol, atol):
                    confirmed[name][el] = confirmed[name].get(el, 0) + 1
                else:
                    mismatch[name][el] = (_max_diff(inj, ref), _max_diff(baseline, ref), moved_inj)

    r = _run("params_effective", inputs, body, **kw)
    if r.status != "PASS":
        return r

    checked = [name for name in paths if name not in not_applicable]

    def acts(name, el) -> bool:
        k, i = el
        return bool(grad_any[name][k][i]) or el in inj_moved[name] or el in ref_moved[name]

    # Post-run: the rebuild probe, for an element that moved nothing on a
    # path through either spelling while it acts on another one.
    rebuilt_moved: dict[str, set] = {name: set() for name in paths}
    build = _rebuilder(node)
    rebuilt0 = None
    faithful: dict[str, bool] = {}
    rebuilt_for: dict = {}
    if build is not None and kept:
        try:
            rebuilt0 = build({})
        except Exception:  # noqa: BLE001 - the node does not rebuild
            rebuilt0 = None
    for name in checked:
        fn = paths[name]
        candidates = [
            el for el in elements
            if not acts(name, el) and ctor_value[el] is not None
            and any(acts(other, el) for other in checked if other != name)
        ]
        if not candidates or rebuilt0 is None:
            continue
        try:
            faithful[name] = all(
                _all_close(fn(rebuilt0, s, bi, dt, None), fn(node, s, bi, dt, None), rtol, atol)
                for s, bi, dt in kept
            )
        except Exception:  # noqa: BLE001 - the rebuilt node cannot run this path
            faithful[name] = False
        if not faithful[name]:
            continue
        for el in candidates:
            if el not in rebuilt_for:
                try:
                    rebuilt_for[el] = build({el[0]: ctor_value[el]})
                except Exception:  # noqa: BLE001 - that value does not construct
                    rebuilt_for[el] = None
            rebuilt1 = rebuilt_for[el]
            if rebuilt1 is None:
                continue
            try:
                if any(not _all_same(fn(rebuilt1, s, bi, dt, None), fn(rebuilt0, s, bi, dt, None))
                       for s, bi, dt in kept):
                    rebuilt_moved[name].add(el)
            except Exception:  # noqa: BLE001 - a different value broke it: it reads it
                rebuilt_moved[name].add(el)

    def names(els) -> list[str]:
        return [labels[el] for el in sorted(els, key=lambda e: (trainable.index(e[0]), e[1]))]

    problems: list[str] = []
    for name in checked:
        where = f"{name}()"
        never = [el for el, (_, _, moved) in mismatch[name].items()
                 if not moved and el not in inj_moved[name]
                 and not grad_any[name][el[0]][el[1]]]
        split = [el for el in mismatch[name] if el not in never]
        if never:
            problems.append(
                f"{where} reads {names(never)} from self.params, not from the "
                "injected params: the constructor value moves its output and "
                "the injected value never does, so "
                + _CONSEQUENCE[name]
                + ".  Read them from {**self.params, **params}."
            )
        if split:
            worst = max(
                (mismatch[name][el][0] / mismatch[name][el][1]
                 for el in split if mismatch[name][el][1] > 0),
                default=float("nan"),
            )
            problems.append(
                f"{where} does not apply the injected {names(split)} the way a "
                "constructed value is applied: at a perturbed value the injected "
                "output misses the constructed one by up to "
                f"{worst:.2f} of the perturbation's own effect (a read split "
                "between the injected params and self.params?), so "
                + _CONSEQUENCE[name] + "."
            )
        if rebuilt_moved[name]:
            problems.append(
                f"{where} ignores the injected {names(rebuilt_moved[name])}: a node "
                "constructed with a different value gives a different output and "
                "the injected value never does -- it reads a copy made at "
                "construction.  Read them from {**self.params, **params} at call "
                "time."
            )
        if name != "update":
            blind = [
                el for el in elements
                if ctor_value[el] is None and not acts(name, el)
                and el not in rebuilt_moved[name] and acts("update", el)
            ]
            if blind:
                problems.append(
                    f"{where}: the injected {names(blind)} never move its output "
                    "while update()'s do, and the constructor value could not be "
                    "varied to tell an unused constant from an ignored one (not a "
                    "plain node.params entry); failing closed"
                )
    dead = [
        k for k in trainable
        if not any(grad_any[name][k].any() for name in checked)
        and not any(acts(name, (k, i)) or (k, i) in rebuilt_moved[name]
                    for name in checked for i in _probe_indices(sizes[k]))
    ]
    if dead:
        problems.append(
            f"zero gradient on every sample wrt trainable param(s) {dead}, and "
            "neither the injected nor the constructed value moves any output: "
            "no path reads them (declare them ParamSpec(trainable=False)), or "
            "the sampled envelope never exercises them"
        )
    if problems:
        return VerificationResult(
            "params_effective", "FAIL", n_examples=r.n_examples,
            detail="; ".join(problems),
        )

    notes = [f"paths checked: {', '.join(checked)}"]
    for name in paths:
        if name in not_applicable:
            notes.append(f"{name}(): not applicable (raises NotImplementedError)")
    for name in checked:
        unused: list[str] = []
        for k in trainable:
            idx = _probe_indices(sizes[k])
            quiet = [i for i in idx if not acts(name, (k, i))]
            if not quiet:
                continue
            if len(quiet) == len(idx) and not grad_any[name][k].any():
                unused.append(k)
            elif shapes[k]:
                # A vector leaf the path reads only part of: name the rest.
                unused.extend(labels[(k, i)] for i in quiet)
        if unused:
            notes.append(f"not consumed by {name}(): {unused}")
    return VerificationResult(
        "params_effective", "PASS", n_examples=r.n_examples, detail="; ".join(notes),
    )


#: What a path that ignores the injected value does to a calibration.
_CONSEQUENCE = {
    "update": "the graph would step the constructor's constant while an optimiser moves the injected one",
    "compute_boundary_fluxes": "a flux edge would deliver the constructor's flux while update() uses the calibrated constant",
    "derivatives": ("integrate_node(..., params=...) / implicit_euler_step(..., params=...) "
                    "would integrate the constructor's constant while update() used the "
                    "calibrated one"),
    "implicit_residual": ("implicit_euler_step(..., params=...) would solve with the "
                          "constructor's constant while update() used the calibrated one"),
    "compute_interface_correction": ("a coupled interface would be corrected with the "
                                     "constructor's constant while update() uses the "
                                     "calibrated one"),
}


def node_boundedness(
    inputs: _Inputs, output_bounds: Bounds, **kw,
) -> VerificationResult:
    """Each field named in ``output_bounds`` stays within ``[lo, hi]``."""
    def body(state, bi, dt):
        out = inputs.node.update(state, bi, dt)
        for f, (lo, hi) in output_bounds.items():
            if f not in out:
                continue
            v = _to_np(out[f])
            assert np.all(v >= lo) and np.all(v <= hi), (
                f"'{f}' left [{lo}, {hi}]: min={v.min()}, max={v.max()}"
            )
    return _run("boundedness", inputs, body, **kw)


def node_energy_monotone(
    inputs: _Inputs, energy_fn: Callable[[dict], Any], rtol: float = 1e-6, **kw,
) -> VerificationResult:
    """``energy_fn(update(state)) <= energy_fn(state)`` (dissipative step).

    ``rtol`` allows float32 round-off; a genuine energy gain is orders
    of magnitude larger than that.
    """
    def body(state, bi, dt):
        e0 = float(energy_fn(state))
        e1 = float(energy_fn(inputs.node.update(state, bi, dt)))
        assert e1 <= e0 * (1 + rtol) + 1e-12, f"energy rose: {e0} -> {e1}"
    return _run("energy_monotone", inputs, body, **kw)


def node_invariant(
    inputs: _Inputs,
    predicate: Callable[[dict, dict, dict, float], bool],
    name: str = "invariant",
    **kw,
) -> VerificationResult:
    """``predicate(state_in, state_out, boundary_inputs, dt)`` holds."""
    def body(state, bi, dt):
        out = inputs.node.update(state, bi, dt)
        assert bool(predicate(state, out, bi, dt)), f"'{name}' violated"
    return _run(name, inputs, body, **kw)


# ---------------------------------------------------------------------------
# Batteries
# ---------------------------------------------------------------------------

DEFAULT_CHECKS = (
    "finite", "structure", "deterministic", "jit_consistent", "gradient_finite",
    "params_consistent", "params_gradient_finite", "params_effective",
)


@stability(StabilityLevel.EXPERIMENTAL)
def make_inputs(
    node,
    bounds: Bounds | None = None,
    *,
    boundary_bounds: Bounds | None = None,
    boundary_inputs: dict | None = None,
    dt_range: tuple[float, float] = (1e-4, 0.01),
    dtype: npt.DTypeLike = np.float32,
) -> _Inputs:
    """Bundle the sampling envelope for the standalone ``node_*`` checks."""
    return _Inputs(
        node, bounds or {}, boundary_bounds or {}, boundary_inputs,
        dt_range, np.dtype(dtype),
    )


@stability(StabilityLevel.EXPERIMENTAL)
def verify_node(
    node,
    bounds: Bounds | None = None,
    *,
    checks: list[str] | None = None,
    boundary_bounds: Bounds | None = None,
    boundary_inputs: dict | None = None,
    dt_range: tuple[float, float] = (1e-4, 0.01),
    dtype: npt.DTypeLike = np.float32,
    output_bounds: Bounds | None = None,
    energy_fn: Callable[[dict], Any] | None = None,
    invariants: dict[str, Callable[[dict, dict, dict, float], bool]] | None = None,
    max_examples: int = 200,
    derandomize: bool = False,
) -> dict[str, VerificationResult]:
    """Run a battery of property checks on ``node.update``.

    Parameters
    ----------
    node : SimulationNode
    bounds : dict
        Per-state-field ``(lo, hi)`` sampling envelope.  Unlisted fields
        default to ``(-1e4, 1e4)``.
    checks : list of str, optional
        Subset of :data:`DEFAULT_CHECKS` to run.  ``output_bounds``,
        ``energy_fn`` and ``invariants`` add their checks regardless.
    boundary_bounds : dict, optional
        Envelope for inputs declared in ``node.boundary_input_spec()``.
    boundary_inputs : dict, optional
        Fixed boundary-input dict; overrides sampling.  Use it when the
        node has no ``boundary_input_spec`` but ``update`` requires an
        input.
    dt_range : (float, float)
        Timestep envelope.
    dtype : numpy dtype
        Sampling dtype.  ``float32`` matches the execution dtype most
        nodes use, so overflow is found where it actually happens.
    output_bounds : dict, optional
        Enables :func:`node_boundedness`.
    energy_fn : callable, optional
        Enables :func:`node_energy_monotone`.
    invariants : dict[str, callable], optional
        Extra ``name -> predicate(state_in, state_out, boundary, dt)``
        checks.
    max_examples : int
        Samples per check.
    derandomize : bool
        Seed Hypothesis from the check's own structure so repeated runs
        draw identical samples (reproducible CI, weaker exploration).

    Returns
    -------
    dict[str, VerificationResult]

    See Also
    --------
    maddening.testing.mms.verify_node_order : the order-of-accuracy half
        of the battery.  Every check here compares the node to itself,
        so none of them can see a wrong discretisation — a stencil with
        the wrong weight is finite, deterministic, JIT-consistent and
        differentiable.  That one compares it to the mathematics.
    """
    inputs = make_inputs(
        node, bounds, boundary_bounds=boundary_bounds,
        boundary_inputs=boundary_inputs, dt_range=dt_range, dtype=dtype,
    )
    kw = dict(max_examples=max_examples, derandomize=derandomize)
    battery = {
        "finite": node_finite,
        "structure": node_structure,
        "deterministic": node_deterministic,
        "jit_consistent": node_jit_consistent,
        "gradient_finite": node_gradient_finite,
        "params_consistent": node_params_consistent,
        "params_gradient_finite": node_params_gradient_finite,
        "params_effective": node_params_effective,
    }
    selected = list(DEFAULT_CHECKS) if checks is None else list(checks)
    unknown = set(selected) - set(battery)
    if unknown:
        raise ValueError(
            f"unknown checks {sorted(unknown)}; valid: {sorted(battery)}"
        )

    results: dict[str, VerificationResult] = {}
    for name in selected:
        results[name] = battery[name](inputs, **kw)
    if output_bounds:
        results["boundedness"] = node_boundedness(inputs, output_bounds, **kw)
    if energy_fn is not None:
        results["energy_monotone"] = node_energy_monotone(inputs, energy_fn, **kw)
    for name, pred in (invariants or {}).items():
        results[name] = node_invariant(inputs, pred, name=name, **kw)
    return results


@stability(StabilityLevel.EXPERIMENTAL)
def assert_node_verified(node, bounds: Bounds | None = None, **kwargs) -> None:
    """``verify_node`` that raises ``AssertionError`` listing every failure.

    Convenient as a one-line pytest body::

        def test_my_node():
            assert_node_verified(my_node, bounds={"T": (200.0, 5000.0)})

    See Also
    --------
    maddening.testing.mms.assert_node_order_verified : the same one-line
        shape for the node's order of accuracy, which this battery
        cannot see.
    """
    results = verify_node(node, bounds, **kwargs)
    bad = [r for r in results.values() if not r.passed]
    if bad:
        raise AssertionError(
            f"{type(node).__name__} failed {len(bad)} check(s):\n"
            + "\n".join(str(r) for r in bad)
        )
