"""CPL-185 under ``jax_enable_x64``: a non-finite tangent or cotangent gives NaN.

``tests/core/test_krylov_solves_propagate_a_non_finite_rhs.py`` holds the
claim in float32, at two and at sixty coupled DOF.  These hold it at sixty
-- above the dense fallback, so GMRES alone answers -- in an x64 process,
with both members float64 (``float64``) and with a float32 member beside a
float64 one (``mixed``, edges cast to the reader's dtype).  float64's
tolerance ``rtol * max|b|`` is the one a non-finite entry used to turn NaN
or ``inf``, letting the zero initial guess pass before a single step.
"""

from __future__ import annotations

import contextlib
import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

N = 30                     # per member: 60 coupled DOF, above the dense fallback
BAD = (float("nan"), float("inf"), float("-inf"))
LEAVES = ("float64", "mixed")


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _dtypes(leaves):
    """Member A's dtype and member B's: both float64, or A float32 (mixed)."""
    return {"float64": (jnp.float64, jnp.float64), "mixed": (jnp.float32, jnp.float64)}[leaves]


class _Relay(SimulationNode):
    """``x <- b + g u`` on an ``N``-entry field of the given dtype."""

    def __init__(self, name, g, b, dtype):
        super().__init__(name, 1.0, g=jnp.asarray(g, dtype), b=jnp.asarray(b, dtype))
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.zeros(N, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=self._dtype,
                                       default=jnp.zeros(N, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["b"] + p["g"] * boundary_inputs["u"]}


@functools.lru_cache(maxsize=None)
def _coupled(leaves, linear_solver):
    """``(jvp_fn, grad_fn)`` of the pair under x64, jitted once: the tangent of
    both fields along ``t`` in A's bias, and the gradient in every parameter of
    ``sum((x_A - obs)**2)``."""
    da, db = _dtypes(leaves)
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.5, 1.0, da))
    gm.add_node(_Relay("B", 0.8, 0.5, db))
    gm.add_edge("A", "B", "x", "u", transform=lambda v: v.astype(db))
    gm.add_edge("B", "A", "x", "u", transform=lambda v: v.astype(da))
    gm.add_coupling_group(["A", "B"], max_iterations=400, tolerance=1e-12, solver="ift",
                          linear_solver=linear_solver)
    gm.compile()
    step, state, ext, params = (gm._compiled_step, gm._state,           # noqa: SLF001
                                gm._default_external_inputs(), gm.params)  # noqa: SLF001

    @jax.jit
    def jvp_fn(t):
        tangent = jax.tree.map(jnp.zeros_like, params)
        tangent["nodes"]["A"]["b"] = t.astype(da)
        _, out = jax.jvp(lambda p: step(state, ext, p), (params,), (tangent,))
        return jnp.concatenate([out["A"]["x"].astype(jnp.float64), out["B"]["x"]])

    @jax.jit
    def grad_fn(obs):
        g = jax.grad(lambda p: jnp.sum((step(state, ext, p)["A"]["x"] - obs) ** 2))(params)
        return jnp.stack([g["nodes"][nm][k].astype(jnp.float64)
                          for nm in ("A", "B") for k in ("b", "g")])

    return jvp_fn, grad_fn


@pytest.mark.parametrize("leaves", LEAVES)
def test_a_non_finite_tangent_or_cotangent_gives_nan_at_sixty_dof_under_x64(leaves):
    """CPL-185: under the default GMRES a NaN or infinite tangent gives a NaN
    tangent in every entry, and a NaN or infinite observation (a cotangent
    entry) a NaN gradient in every parameter -- never an exactly zero one --
    where ``linear_solver="dense"`` is non-finite too; finite inputs agree
    with dense to the tolerance of the leaves' precision."""
    with _x64():
        jvp_g, grad_g = _coupled(leaves, "gmres")
        jvp_d, grad_d = _coupled(leaves, "dense")
        rtol = 1e-9 if leaves == "float64" else 1e-4
        one = jnp.asarray(1.0, jnp.float64)
        np.testing.assert_allclose(np.asarray(jvp_g(one)), np.asarray(jvp_d(one)), rtol=rtol)
        obs = jnp.full((N,), 3.0, _dtypes(leaves)[0])
        np.testing.assert_allclose(np.asarray(grad_g(obs)), np.asarray(grad_d(obs)), rtol=rtol)
        for bad in BAD:
            t = jnp.asarray(bad, jnp.float64)
            got, ref = np.asarray(jvp_g(t)), np.asarray(jvp_d(t))
            assert np.all(np.isnan(got)), (leaves, bad, got)
            assert not np.all(np.isfinite(ref)), (leaves, bad, "the dense reference is finite")
            o = obs.at[N - 1].set(bad)
            got, ref = np.asarray(grad_g(o)), np.asarray(grad_d(o))
            assert np.all(np.isnan(got)), (leaves, bad, got)
            assert not np.all(np.isfinite(ref)), (leaves, bad, "the dense reference is finite")
