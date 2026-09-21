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
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp

import numpy as np
import numpy.typing as npt

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.node import _method_accepts_params, _signature_takes_params
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
    from maddening.core.node import SimulationNode as _Base  # noqa: PLC0415
    return type(node).compute_boundary_fluxes is not _Base.compute_boundary_fluxes


def _flux_accepts_params(node) -> bool:
    return _signature_takes_params(node.compute_boundary_fluxes)


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
    probe = getattr(node, "accepts_params", None)
    return bool(probe()) if callable(probe) else False


def _implements(node, method: str) -> bool:
    """Does ``type(node)`` override ``method`` at all?"""
    from maddening.core.node import SimulationNode as _Base  # noqa: PLC0415
    return getattr(type(node), method, None) is not getattr(_Base, method)


def _perturbed(value):
    """A visibly different constructor value of the same container type
    (x1.5, or 0.5 where the value is exactly zero)."""
    arr = np.asarray(value, dtype=np.float64)
    new = np.where(arr == 0, 0.5, arr * 1.5)
    if isinstance(value, (list, tuple)):
        return type(value)(new.tolist())
    if isinstance(value, np.generic):
        return type(value)(new)
    if isinstance(value, float):
        return float(new)
    return jnp.asarray(new, dtype=getattr(value, "dtype", None))


def _same(a, b) -> bool:
    x, y = _to_np(a), _to_np(b)
    try:
        return bool(np.array_equal(x, y, equal_nan=True))
    except TypeError:  # a dtype isnan cannot take (bool, object)
        return bool(np.array_equal(x, y))


def _reads_constructor_value(node, call, key, baseline):
    """Does ``call()`` -- one no-params evaluation of a path -- change
    when ``node.params[key]`` changes?

    This is what separates a constant a path *does not consume* (the
    collision-free ``derivatives`` of a bouncing ball has no use for
    ``elasticity``) from one it consumes *from the wrong place* (an
    override that reads ``self.params[key]`` and ignores the injected
    value).  Both have a zero gradient with respect to the injected leaf;
    only the second moves when the constructor value moves.  The swap is
    undone in ``finally``.  ``None`` when the probe cannot be made:
    ``node.params`` is not a plain dict, or the leaf is not a constructor
    entry.
    """
    params = getattr(node, "params", None)
    if not isinstance(params, dict) or key not in params:
        return None
    old = params[key]
    try:
        params[key] = _perturbed(old)
    except Exception:  # noqa: BLE001 - a value this cannot perturb
        params[key] = old
        return None
    try:
        probe = call()
    except Exception:  # noqa: BLE001 - it certainly read it: a different value broke it
        return True
    finally:
        params[key] = old
    return any(not _same(baseline[f], probe[f]) for f in baseline)


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


def node_params_effective(inputs: _Inputs, **kw) -> VerificationResult:
    """Every *trainable* parameter influences the output -- on every path.

    A leaf of :meth:`params_pytree` that ``update`` reads from
    ``self.params`` instead of the injected ``params`` is a silent trap:
    the graph passes a value, an optimiser moves it, nothing changes and
    the gradient is identically zero.  ``params_consistent`` cannot see
    that (both paths read the same constant), so this check aggregates
    over the sampled inputs and fails if some trainable leaf had a zero
    gradient on *every* example.  A leaf that only matters on some
    inputs (a restitution coefficient without a collision) passes as
    long as one sample exercised it.  ``SKIP`` for nodes without
    ``params`` and for leaves declared ``trainable=False``.

    The same trap has a second door.  A node that implements
    ``derivatives`` or ``implicit_residual`` is integrated through
    :func:`~maddening.core.simulation.integrators.integrate_node` and
    :func:`~maddening.core.simulation.implicit.implicit_euler_step`,
    which forward ``params`` to those methods; an override that *takes*
    the keyword and reads ``self.params`` anyway integrates the
    constructor's constant while ``update`` uses the calibrated one
    (the ``MADD-ANO-018`` shape, one method further in).  So for each
    implemented path this check also differentiates the path's output
    with respect to the injected leaves, and separates a leaf the path
    does not consume at all (zero gradient, and the output does not move
    when the constructor value is varied either -- the collision-free
    right-hand side of a bouncing ball has no use for ``elasticity``)
    from one it consumes from the wrong place (zero gradient, but the
    output moves with the constructor value).  The first is reported in
    ``detail``; the second is a ``FAIL`` naming the method and the
    leaves.  An override declared without the keyword is a ``FAIL`` too
    -- the integrators refuse a calibrated ``params`` for it -- and an
    override that raises ``NotImplementedError`` (a discrete node) is
    recorded as not applicable.  A path is probed by varying
    ``node.params[leaf]`` in place and restoring it; a leaf that is not a
    constructor entry cannot be probed and fails closed if it is alive
    through ``update`` and dead through the path.  ``detail`` names the
    paths checked either way.
    """
    node = inputs.node
    if not _node_accepts_params(node):
        return _skip_no_params("params_effective")
    base = node.params_pytree()
    specs = node.param_specs() if hasattr(node, "param_specs") else {}
    trainable = [k for k in base if specs.get(k) is None or specs[k].trainable]
    if not trainable:
        return VerificationResult(
            "params_effective", "SKIP", detail="no trainable parameters",
        )

    def via_update(state, bi, dt, p):
        return _outputs(node, state, bi, dt) if p is None else _outputs(node, state, bi, dt, params=p)

    def via_derivatives(state, bi, dt, p):
        return node.derivatives(state, bi) if p is None else node.derivatives(state, bi, params=p)

    def via_residual(state, bi, dt, p):
        # x_new = x_old = state: R = -dt * f(state; params), whose
        # sensitivity to a constant is dt times f's.
        return (node.implicit_residual(state, state, bi, dt) if p is None
                else node.implicit_residual(state, state, bi, dt, params=p))

    paths = {"update": via_update}
    if _implements(node, "derivatives"):
        paths["derivatives"] = via_derivatives
    if _implements(node, "implicit_residual"):
        paths["implicit_residual"] = via_residual
    solver_paths = [name for name in paths if name != "update"]
    accepts = {name: _method_accepts_params(node, name) for name in solver_paths}

    seen_nonzero: dict[str, set[str]] = {name: set() for name in paths}
    consumed: dict[str, set[str]] = {name: set() for name in solver_paths}
    unprobeable: dict[str, set[str]] = {name: set() for name in solver_paths}
    not_applicable: set[str] = set()

    def body(state, bi, dt):
        for name, fn in paths.items():
            if name in not_applicable:
                continue
            baseline = None
            if name != "update":
                try:
                    baseline = fn(state, bi, dt, None)
                except NotImplementedError:
                    not_applicable.add(name)
                    continue
                if not accepts[name]:
                    raise AssertionError(
                        f"update() takes params but {name}() does not: "
                        "integrate_node / implicit_euler_step refuse a calibrated "
                        "params for this node rather than integrate the "
                        "constructor's constants while update() uses the "
                        f"calibrated ones.  Declare {name}(..., *, params=None) "
                        "and read constants from {**self.params, **params}."
                    )

            def loss(p, fn=fn):
                out = fn(state, bi, dt, p)
                return sum(jnp.sum(v) for v in out.values()
                           if jnp.issubdtype(v.dtype, jnp.floating))

            g = jax.grad(loss)(base)
            for k in trainable:
                if k not in seen_nonzero[name] and bool(jnp.any(g[k] != 0)):
                    seen_nonzero[name].add(k)
            if name == "update":
                continue
            for k in trainable:
                if k in seen_nonzero[name] or k in consumed[name] or k in unprobeable[name]:
                    continue
                reads = _reads_constructor_value(
                    node, lambda fn=fn: fn(state, bi, dt, None), k, baseline,
                )
                if reads is None:
                    unprobeable[name].add(k)
                elif reads:
                    consumed[name].add(k)

    r = _run("params_effective", inputs, body, **kw)
    if r.status != "PASS":
        return r

    problems: list[str] = []
    dead = sorted(set(trainable) - seen_nonzero["update"])
    if dead:
        problems.append(
            f"zero gradient on every sample wrt trainable param(s) {dead}: "
            "update() probably reads them from self.params instead of "
            "the injected params (or declare them ParamSpec(trainable=False))"
        )
    checked = ["update"] + [n for n in solver_paths if n not in not_applicable]
    for name in checked[1:]:
        ignored = sorted(consumed[name] - seen_nonzero[name])
        if ignored:
            problems.append(
                f"{name}() reads {ignored} from self.params, not from the "
                "injected params: the constructor value moves its output and "
                "the injected value never does, so integrate_node(..., "
                "params=...) / implicit_euler_step(..., params=...) would "
                "integrate the constructor's constant while update() used the "
                "calibrated one.  Read them from {**self.params, **params}."
            )
        blind = sorted(
            k for k in unprobeable[name]
            if k in seen_nonzero["update"] and k not in seen_nonzero[name]
        )
        if blind:
            problems.append(
                f"{name}(): zero gradient on every sample wrt {blind} while "
                "update() has one, and the constructor value could not be "
                "varied to tell an unused constant from an ignored one "
                "(not a plain node.params entry); failing closed"
            )
    if problems:
        return VerificationResult(
            "params_effective", "FAIL", n_examples=r.n_examples,
            detail="; ".join(problems),
        )

    notes = [f"paths checked: {', '.join(checked)}"]
    for name in solver_paths:
        if name in not_applicable:
            notes.append(f"{name}(): not applicable (raises NotImplementedError)")
            continue
        unused = sorted(set(trainable) - seen_nonzero[name] - consumed[name])
        if unused:
            notes.append(f"not consumed by {name}(): {unused}")
    return VerificationResult(
        "params_effective", "PASS", n_examples=r.n_examples, detail="; ".join(notes),
    )


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
