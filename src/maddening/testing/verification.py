"""Property-based verification harness for SimulationNode subclasses.

Runs a battery of universal checks against a node's ``update()`` by
sampling inputs with Hypothesis and shrinking any failure to a minimal
counterexample.  The checks are the ones every physics node should
satisfy regardless of what it models: finite outputs, preserved state
structure, determinism, JIT/eager agreement, and finite gradients.

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


@dataclass
class VerificationResult:
    """Outcome of one check.

    ``status`` is ``"PASS"`` (no counterexample found in ``n_examples``
    samples), ``"FAIL"`` (a shrunk counterexample is in
    ``counterexample``), or ``"ERROR"`` (the check itself could not run,
    e.g. ``update()`` is not differentiable; see ``detail``).
    """
    name: str
    status: str
    detail: str = ""
    counterexample: dict[str, Any] = field(default_factory=dict)
    n_examples: int = 0

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"

    def __str__(self) -> str:
        head = f"{self.name}: {self.status}"
        if self.failed:
            return f"{head}\n  counterexample: {self.counterexample}\n  {self.detail}"
        if self.status == "ERROR":
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
)


def make_inputs(
    node,
    bounds: Bounds | None = None,
    *,
    boundary_bounds: Bounds | None = None,
    boundary_inputs: dict | None = None,
    dt_range: tuple[float, float] = (1e-4, 0.01),
    dtype: np.dtype = np.float32,
) -> _Inputs:
    """Bundle the sampling envelope for the standalone ``node_*`` checks."""
    return _Inputs(
        node, bounds or {}, boundary_bounds or {}, boundary_inputs,
        dt_range, np.dtype(dtype),
    )


def verify_node(
    node,
    bounds: Bounds | None = None,
    *,
    checks: list[str] | None = None,
    boundary_bounds: Bounds | None = None,
    boundary_inputs: dict | None = None,
    dt_range: tuple[float, float] = (1e-4, 0.01),
    dtype: np.dtype = np.float32,
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


def assert_node_verified(node, bounds: Bounds | None = None, **kwargs) -> None:
    """``verify_node`` that raises ``AssertionError`` listing every failure.

    Convenient as a one-line pytest body::

        def test_my_node():
            assert_node_verified(my_node, bounds={"T": (200.0, 5000.0)})
    """
    results = verify_node(node, bounds, **kwargs)
    bad = [r for r in results.values() if not r.passed]
    if bad:
        raise AssertionError(
            f"{type(node).__name__} failed {len(bad)} check(s):\n"
            + "\n".join(str(r) for r in bad)
        )
