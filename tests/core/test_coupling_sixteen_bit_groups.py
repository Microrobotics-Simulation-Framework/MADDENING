"""A bfloat16 or float16 coupling group: its pass count and its diagnostics.

Two defects of 16-bit groups under the default ``solver="ift"``:

* the pass count was cast to the group's floating dtype for the
  diagnostics carry, which rounds every count above 256 (bfloat16) or
  2048 (float16) to that dtype's grid -- a group that ran its whole budget
  of 257 passes reported 256, and the documented cap check ``iterations >=
  max_iterations`` was false at the cap (``converged`` stayed right).  The
  count is now carried as the int32 it is;
* ``diagnostics=True`` raised ``NotImplementedError`` from inside the step:
  the spectral and gradient bounds call LAPACK, which has no 16-bit
  kernels.  Their analysis now runs in float32 while the map is still
  evaluated in the group's own dtype.

The fixture is a rotation coupled to an identity relay (``|x|`` preserved,
the residual O(1)), which never meets ``tolerance=1e-30`` and so always
runs to its cap, and a contracting pair for the bound itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
_THETA = 0.3
_ROTATION = np.array([[np.cos(_THETA), -np.sin(_THETA)], [np.sin(_THETA), np.cos(_THETA)]])


class _Linear(SimulationNode):
    """``x <- G @ u + b`` in a fixed dtype."""

    def __init__(self, name, G, b, x0, dtype):
        super().__init__(name, 1.0, G=jnp.asarray(G, dtype), b=jnp.asarray(b, dtype))
        self._x0 = np.asarray(x0)
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=self._dtype,
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt):
        x = self.params["G"] @ boundary_inputs["u"] + self.params["b"]
        return {"x": x.astype(self._dtype)}


def _pair(dtype, Ga, ba, Gb, bb, x0, **group):
    gm = GraphManager()
    gm.add_node(_Linear("a", Ga, ba, x0, dtype))
    gm.add_node(_Linear("b", Gb, bb, x0, dtype))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], **group)
    gm.compile()
    return gm


def _rotation(dtype, cap, diagnostics):
    return _pair(dtype, _ROTATION, [0.0, 0.0], np.eye(2), [0.0, 0.0], [1.0, 0.0],
                 max_iterations=cap, tolerance=1e-30, diagnostics=diagnostics)


@pytest.mark.parametrize("dtype, cap", [(jnp.bfloat16, 257), (jnp.bfloat16, 301),
                                        (jnp.float16, 2049), (jnp.float32, 257)])
def test_the_cap_check_holds_at_the_cap_in_any_float_dtype(dtype, cap):
    """``iterations == max_iterations`` exactly when the budget ran out, at any count."""
    gm = _rotation(dtype, cap, diagnostics=False)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["iterations"] == cap and d["iterations"] >= cap, dict(d)
    assert not d["converged"]


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_diagnostics_run_on_a_sixteen_bit_group(dtype):
    """``diagnostics=True`` computes the spectral keys instead of raising.

    The rotation's spectral radius is one: nothing contracts, so the bound
    is not usable, and it says so.  The cap check holds with diagnostics on
    as well.
    """
    gm = _rotation(dtype, 257, diagnostics=True)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["iterations"] == 257, dict(d)
    assert d["rho_spectral"] == pytest.approx(1.0, abs=2e-2), dict(d)
    assert not d["spectral_usable"], dict(d)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_a_usable_sixteen_bit_spectral_bound_is_not_below_the_true_distance(dtype):
    """A contracting 16-bit pair stopped early: the bound against the exact fixed point.

    ``a <- 0.6 u + (1, 2)``, ``b <- 0.9 u``; the fixed point of the map with
    its gains rounded to the dtype, by a float64 solve, and the distance in
    the group's relative L2 norm.
    """
    Ga, Gb = np.eye(2) * 0.6, np.eye(2) * 0.9
    ba, bb = np.array([1.0, 2.0]), np.zeros(2)
    gm = _pair(dtype, Ga, ba, Gb, bb, [0.0, 0.0], max_iterations=4, tolerance=1e-30,
               diagnostics=True)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["iterations"] == 4 and np.isfinite(d["rho_spectral"]), dict(d)
    r = lambda v: np.asarray(jnp.asarray(v, dtype), np.float64)  # noqa: E731
    Ga_, Gb_, ba_, bb_ = r(Ga), r(Gb), r(ba), r(bb)
    # a = Ga b + ba, b = Gb a + bb  =>  a = Ga (Gb a + bb) + ba
    a = np.linalg.solve(np.eye(2) - Ga_ @ Gb_, Ga_ @ bb_ + ba_)
    exact = {"a": a, "b": Gb_ @ a + bb_}
    dist2 = 0.0
    for nm in ("a", "b"):
        got = np.asarray(gm.get_node_state(nm)["x"], np.float64)
        ref = max(np.max(np.abs(got)), np.max(np.abs(exact[nm])))
        dist2 += float(np.sum(((got - exact[nm]) / ref) ** 2))
    dist = dist2 ** 0.5
    assert dist > 0, "fixture premise: stopped short of the fixed point"
    if d["spectral_usable"]:
        assert d["spectral_error_bound"] >= dist, (d["spectral_error_bound"], dist, dict(d))


def _contracting(dtype, **group):
    return _pair(dtype, np.eye(2) * 0.6, [1.0, 2.0], np.eye(2) * 0.9, [0.0, 0.0], [0.0, 0.0],
                 max_iterations=60, tolerance=1e-2, **group)


@pytest.mark.parametrize("acceleration", ["aitken", "iqn-ils", "iqn-imvj"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_fori_and_ift_take_the_same_passes_on_an_accelerated_sixteen_bit_group(dtype, acceleration):
    """``solver="fori"`` with an accelerator runs on a 16-bit group, and agrees with ``"ift"``.

    The fori loop's carries were seeded in at least float32 (Aitken's
    ``omega`` and residual, IQN's ``V``/``W`` and secant state) while the
    accelerator returned them in the group's dtype, and ``lax.fori_loop``
    raised ``TypeError`` ("carry input and carry output must have equal
    types") at trace time -- in 0.3.x as well.  The returned values are now
    cast to their carries' dtypes.  ``CouplingGroup.solver`` documents that
    the two solvers run the same passes; two steps exercise the IMVJ warm
    start read back from ``_meta``.
    """
    reports = {}
    for solver in ("fori", "ift"):
        gm = _contracting(dtype, acceleration=acceleration, solver=solver, diagnostics=True)
        gm.step()
        gm.step()
        reports[solver] = gm.coupling_diagnostics()[KEY]
    assert reports["fori"]["iterations"] == reports["ift"]["iterations"], reports
    assert reports["fori"]["converged"] == reports["ift"]["converged"], reports
