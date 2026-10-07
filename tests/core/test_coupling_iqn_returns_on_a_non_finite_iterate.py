"""A coupling group under IQN returns from its step when its iterate leaves float range.

``acceleration="iqn-ils"`` and ``"iqn-imvj"`` solve a least-squares
problem on the secant columns each pass, through an SVD.  LAPACK's SVD
does not return on some matrices that hold an ``inf``, and a group whose
iterate overflowed before ``max_iterations`` handed it one: ``step()``
never returned (no return in 240 s where a converging step of the same
compiled graph takes 0.05 s), in every release that has IQN.  ``"none"``
and ``"aitken"`` ran to the cap and reported the state as not finite.

Every call that could hang runs in a thread of its own and is given
:data:`LIMIT` seconds, so a regression fails this file and does not stop
the run.  The fixture's first entry is ``exp`` of itself each pass (no
fixed point; it leaves float range within a few passes of any start)
beside three entries that contract, so the secant column of the pass
that overflows holds one ``inf`` and finite entries: the matrix the SVD
does not return on (an all-``inf`` column, or a NaN beside the ``inf``,
returns).
"""

from __future__ import annotations

import atexit
import math
import os
import sys
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import acceleration
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

#: Seconds a call is given (a step that returns takes well under one).
LIMIT = 30.0
IQN = ("iqn-ils", "iqn-imvj")
DTYPES = ("float32", "float64")


_STUCK = []


def _leave_without_finalising():
    """A thread that never returns from LAPACK aborts the interpreter when
    it finalises (exit -6, after the report): leave by ``os._exit`` once
    everything else at exit has run, with the failing status."""
    if _STUCK:
        return
    _STUCK.append(True)

    def leave():
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    atexit.register(leave)


def within_the_limit(fn):
    """``fn()`` from a daemon thread, or a failure after :data:`LIMIT` seconds."""
    box: dict = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(LIMIT)
    if thread.is_alive():
        _leave_without_finalising()
        pytest.fail(f"no return in {LIMIT:.0f} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


class _Relay(SimulationNode):
    """``x <- (exp(u_0), u_1 / 2 + 1, ...)``, or ``x <- u``."""

    def __init__(self, name, grows, dtype):
        super().__init__(name=name, timestep=1.0)
        self._grows, self._dtype = grows, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.asarray(_START, self._dtype)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(len(_START),), dtype=self._dtype,
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        if self._grows:
            u = jnp.concatenate([jnp.exp(u[:1]), 0.5 * u[1:] + 1.0])
        return {"x": u.astype(self._dtype)}


_START = (3.0, -5.0, 7.0, 2.0)


def _graph(acceleration_, dtype, **group):
    gm = GraphManager()
    gm.add_node(_Relay("a", True, dtype))
    gm.add_node(_Relay("b", False, dtype))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(max_iterations=40, tolerance=1e-6, acceleration=acceleration_, diagnostics=False)
    kw.update(group)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _step_reports_a_state_that_is_not_finite(acceleration_, dtype, **group):
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        gm = _graph(acceleration_, dtype, **group)
        within_the_limit(gm.step)                 # the compile is in the limit too
        (report,) = gm.coupling_diagnostics().values()
        state = np.asarray(gm.get_node_state("a")["x"])
    finally:
        jax.config.update("jax_enable_x64", x64)
    assert not np.all(np.isfinite(state)), f"premise: the iterate stayed finite ({state})"
    assert report["converged"] is False, report
    assert report["residual"] == math.inf, report
    return report


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("acceleration_", IQN)
def test_a_step_whose_iterate_leaves_float_range_under_iqn_returns(acceleration_, dtype):
    """CPL-053, CPL-084: the step returns, ``converged`` False and
    ``residual`` ``inf``, having run to its cap as the unaccelerated group
    does."""
    report = _step_reports_a_state_that_is_not_finite(acceleration_, dtype)
    assert report["iterations"] == 40, report


# Slow: four more compiles of the pair, two of them of an unrolled loop.
# Per push: tests/core/test_coupling_iqn_returns_on_a_non_finite_iterate.py::test_a_step_whose_iterate_leaves_float_range_under_iqn_returns
@pytest.mark.slow
@pytest.mark.parametrize("group", [
    dict(solver="fori", max_iterations=24),
    dict(diagnostics=True),
    dict(convergence_norm="l2"),
    dict(convergence_norm="interface"),
    dict(iteration_mode="jacobi"),
], ids=["fori", "diagnostics", "l2", "interface", "jacobi"])
@pytest.mark.parametrize("acceleration_", IQN + ("aitken", "none"))
def test_the_step_returns_under_every_solver_norm_and_schedule(acceleration_, group):
    """The same step under the unrolled solver, with the diagnostics (whose
    analysis takes singular values at the returned state), under each norm
    and both schedules; and the accelerations that always
    returned, beside it."""
    report = _step_reports_a_state_that_is_not_finite(acceleration_, "float64", **group)
    assert report["iterations"] == group.get("max_iterations", 40), report


def _poisoned(kind, n, cols, dtype, rng):
    """``(V, W, residual, previous residual)`` with *kind* written into them."""
    big = float(np.finfo(dtype).max)
    V = rng.standard_normal((n, cols)).astype(dtype)
    W = rng.standard_normal((n, cols)).astype(dtype)
    r = rng.standard_normal(n).astype(dtype)
    r_prev = rng.standard_normal(n).astype(dtype)
    if kind == "an inf in a stored column":
        V[min(3, n - 1), 1] = np.inf
    elif kind == "a -inf in a stored column":
        V[0, cols - 1] = -np.inf
    elif kind == "an inf column":
        V[:, 1] = np.inf
    elif kind == "every entry inf":
        V[:] = np.inf
    elif kind == "a NaN in a stored column":
        V[0, 1] = np.nan
    elif kind == "an inf and a NaN":
        V[0, 1], V[1, 2] = np.inf, np.nan
    elif kind == "an inf residual":             # the new column is inf
        r[0] = np.inf
    elif kind == "a NaN residual":
        r[1] = np.nan
    elif kind == "a previous residual at the largest float":   # the new column overflows
        r_prev[:] = -big
        r[:] = big
    elif kind == "columns at the largest float":
        V[:, 1], V[0, :] = big, -big
    else:
        assert kind == "finite", kind
    return V, W, r, r_prev


_POISONS = ("an inf in a stored column", "a -inf in a stored column", "an inf column",
            "every entry inf", "a NaN in a stored column", "an inf and a NaN",
            "an inf residual", "a NaN residual", "a previous residual at the largest float",
            "columns at the largest float", "finite")


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_secant_update_returns_on_every_non_finite_input(dtype):
    """``iqn_ils_update`` itself, compiled, on secant matrices and
    residuals holding each kind of value a diverging iterate leaves in
    them, at three sizes: it returns, and a matrix that is not finite
    means the quasi-Newton step is not taken (the Aitken fallback's
    iterate comes back)."""
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        rng = np.random.default_rng(0)
        for n, cols in ((4, 4), (12, 8), (16, 16)):
            update = jax.jit(lambda *a: acceleration.iqn_ils_update(*a, have_prev=True))
            for kind in _POISONS:
                V, W, r, r_prev = _poisoned(kind, n, cols, np.dtype(dtype), rng)
                x_old = rng.standard_normal(n).astype(dtype)
                with np.errstate(all="ignore"):
                    x_raw = (x_old + r).astype(dtype)
                args = (jnp.asarray(x_raw), jnp.asarray(x_old), jnp.asarray(r_prev),
                        jnp.asarray(x_old), jnp.asarray(V), jnp.asarray(W), jnp.int32(cols - 1),
                        jnp.asarray(0.5, dtype), jnp.asarray(r_prev))
                out = within_the_limit(lambda args=args: jax.block_until_ready(update(*args)))
                x_new, new_V = np.asarray(out[0]), np.asarray(out[1])
                if kind == "finite":
                    assert np.all(np.isfinite(x_new)), (kind, n, cols)
                elif not np.all(np.isfinite(new_V)):
                    aitken = np.asarray(acceleration.aitken_relaxation(
                        args[1], args[0], args[8], args[7])[0])
                    # Compiled against not: an ulp or two (a fused multiply-add).
                    np.testing.assert_allclose(x_new, aitken, rtol=16 * np.finfo(dtype).eps,
                                               err_msg=f"{kind}, {n} x {cols}")
    finally:
        jax.config.update("jax_enable_x64", x64)


@pytest.mark.parametrize("dtype", DTYPES)
def test_no_svd_of_the_coupling_analysis_is_handed_a_matrix_that_is_not_finite(dtype):
    """``_lapack_input`` withholds exactly the matrices whose squares do
    not sum to a finite number, and returns every other unchanged;
    ``_spectral_norm`` of a withheld matrix is NaN and of any other the
    SVD's."""
    x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == "float64")
    try:
        rng = np.random.default_rng(1)
        big = float(np.finfo(dtype).max)
        norm = jax.jit(acceleration._spectral_norm)                       # noqa: SLF001
        for n, m in ((4, 4), (12, 8), (16, 16)):
            A = rng.standard_normal((n, m)).astype(dtype)
            safe, ok = acceleration._lapack_input(jnp.asarray(A))        # noqa: SLF001
            assert bool(ok)
            np.testing.assert_array_equal(np.asarray(safe), A)
            np.testing.assert_array_equal(np.asarray(norm(jnp.asarray(A))),
                                          np.asarray(jnp.linalg.norm(jnp.asarray(A), ord=2)))
            for poison in (np.inf, -np.inf, np.nan, big):
                B = A.copy()
                B[min(3, n - 1), 0] = poison
                if poison == big:
                    B[:, 0], B[0, :] = big, -big
                safe, ok = acceleration._lapack_input(jnp.asarray(B))    # noqa: SLF001
                assert not bool(ok), (poison, n, m)
                assert not np.any(np.asarray(safe)), (poison, n, m)
                assert math.isnan(float(within_the_limit(lambda B=B: norm(jnp.asarray(B)))))
        # Per matrix of a batch.
        batch = rng.standard_normal((3, 4, 4)).astype(dtype)
        batch[1, 2, 2] = np.inf
        out = np.asarray(within_the_limit(lambda: norm(jnp.asarray(batch))))
        assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2]), out
    finally:
        jax.config.update("jax_enable_x64", x64)
