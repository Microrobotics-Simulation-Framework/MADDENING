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
  evaluated in the group's own dtype;
* ``jax.grad`` and ``jax.jvp`` through the group raised the same
  ``NotImplementedError`` (lineax's QR ``TypeError`` on jax 0.11.2): the
  IFT rule's linear solve ran in the group's dtype (MADD-ANO-161).  It now
  runs in float32 on the group's own 16-bit operator and casts back;
* the spectral slots were stored in the group's dtype, so ``rho_spectral``
  read to bfloat16's ``2**-8`` and the bounds beside it were rounded too
  (CPL-087).  They are kept in the analysis's float32.

The fixture is a rotation coupled to an identity relay (``|x|`` preserved,
the residual O(1)), which never meets ``tolerance=1e-30`` and so always
runs to its cap, and a contracting pair for the bound itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
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


class _Affine(SimulationNode):
    """``x <- g * u + b`` entry by entry, ``g`` and ``b`` graph parameters."""

    def __init__(self, name, g, b, n, dtype):
        super().__init__(name, 1.0, g=jnp.asarray(g, dtype), b=jnp.full((n,), b, dtype))
        self._n, self._dtype = n, dtype

    def initial_state(self):
        return {"x": jnp.zeros((self._n,), self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros((self._n,), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": (p["g"] * boundary_inputs["u"] + p["b"]).astype(self._dtype)}


def _affine_pair(dtype, n, **group):
    """``a <- 0.6 u + 1``, ``b <- 0.9 u + 0.5``, each reading the other: the
    coupled sensitivity of ``x_a`` to ``b_a`` is ``1 / (1 - g_a g_b)`` per entry."""
    gm = GraphManager()
    gm.add_node(_Affine("a", 0.6, 1.0, n, dtype))
    gm.add_node(_Affine("b", 0.9, 0.5, n, dtype))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], tolerance=1e-2, max_iterations=200, **group)
    gm.compile()
    return gm


def _sensitivity(gm, dtype):
    """``1 / (1 - g_a g_b)`` in float64 from the gains as the dtype holds them."""
    r = lambda v: float(np.asarray(jnp.asarray(v, dtype), np.float64))  # noqa: E731
    return 1.0 / (1.0 - r(0.6) * r(0.9))


def _xa_total(gm):
    def f(p):
        gm.reset_state()
        return jnp.sum(gm.run_scan(2, params=p)["a"]["x"].astype(jnp.float32))
    return f


@pytest.mark.parametrize("linear_solver", ["gmres", "dense"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_ift_gradient_of_a_sixteen_bit_group_is_its_coupled_sensitivity(dtype, linear_solver):
    """``jax.grad`` and ``jax.jvp`` through a 16-bit group under ``solver="ift"``
    give the IFT sensitivity to the dtype's resolution, and agree with each
    other; they raised ``NotImplementedError`` (MADD-ANO-161).

    The tolerance is four ``eps`` of the dtype times the conditioning
    ``1 / (1 - g_a g_b)``: the operator is the 16-bit map's own, its products
    rounded to the dtype, and the answer is returned in it."""
    gm = _affine_pair(dtype, 1, linear_solver=linear_solver)
    p = jax.tree.map(lambda v: v, gm.params)
    want = _sensitivity(gm, dtype)
    eps = float(jnp.finfo(dtype).eps)
    tol = 4 * eps * want * want
    g = jax.grad(_xa_total(gm))(p)
    got = float(np.asarray(g["nodes"]["a"]["b"], np.float64).sum())
    assert abs(got - want) <= tol, (got, want, tol)
    t = jax.tree.map(jnp.zeros_like, p)
    t["nodes"]["a"]["b"] = jnp.ones_like(t["nodes"]["a"]["b"])
    fwd = float(jax.jvp(_xa_total(gm), (p,), (t,))[1])
    assert abs(fwd - want) <= tol, (fwd, want, tol)


class _Mixing(SimulationNode):
    """``x <- G @ u + b``: a dense coupling, so the adjoint's Krylov space has no
    shortcut (every entry reads every other)."""

    def __init__(self, name, G, b, dtype):
        super().__init__(name, 1.0, G=jnp.asarray(G, dtype), b=jnp.asarray(b, dtype))
        self._n, self._dtype = len(b), dtype

    def initial_state(self):
        return {"x": jnp.zeros((self._n,), self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros((self._n,), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": (p["G"] @ boundary_inputs["u"] + p["b"]).astype(self._dtype)}


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_a_sixteen_bit_group_above_the_dense_fallback_differentiates_by_gmres(dtype):
    """128 coupled entries, above the 50 at which a failed GMRES re-solves
    densely, so the float32 GMRES must meet its criterion on the 16-bit
    operator by itself.  ``a <- diag(g) u + b_a`` reads a dense ``b <- G u + b_b``
    (radius 0.5).  A float32 criterion (100 float32 ``eps``) is unreachable on
    an operator whose products are rounded to 16 bits and raised "the coupling
    adjoint solve did not converge"; a criterion of 100 units of the 16-bit
    dtype (0.78 in bfloat16) stopped GMRES at its first iterate.  The gradient
    of ``sum(x_a)`` in ``b_a`` is ``1^T (I - diag(g) G)^{-1}`` of the gains as
    the dtype holds them, to two ``eps`` of the dtype in norm -- the 16-bit
    output's own resolution."""
    n = 64
    rng = np.random.default_rng(7)
    g = rng.uniform(0.3, 0.7, n)
    G = rng.standard_normal((n, n))
    G *= 0.5 / np.max(np.abs(np.linalg.eigvals(np.diag(g) @ G)))
    gm = GraphManager()
    gm.add_node(_Affine("a", g, 1.0, n, dtype))
    gm.add_node(_Mixing("b", G, rng.uniform(-1, 1, n), dtype))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], tolerance=1e-2, max_iterations=200)
    gm.compile()
    p = jax.tree.map(lambda v: v, gm.params)
    got = np.asarray(jax.grad(_xa_total(gm))(p)["nodes"]["a"]["b"], np.float64)
    r = lambda v: np.asarray(jnp.asarray(v, dtype), np.float64)  # noqa: E731
    want = np.linalg.solve((np.eye(n) - np.diag(r(g)) @ r(G)).T, np.ones(n))
    eps = float(jnp.finfo(dtype).eps)
    assert np.all(np.isfinite(got)), got
    assert np.linalg.norm(got - want) <= 2 * eps * np.linalg.norm(want), (
        np.linalg.norm(got - want) / np.linalg.norm(want))


_SPECTRAL_SLOTS = ("rho_spectral", "spectral_residual", "spectral_amplification",
                   "gradient_relative_error_bound", "pass_evaluations")


def _slot_dtypes(gm):
    meta = gm._state["_meta"]
    return {s: jnp.asarray(meta[f"coupling_{KEY}_{s}"]).dtype for s in _SPECTRAL_SLOTS}


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_the_spectral_slots_are_kept_in_the_analysis_dtype(dtype):
    """A 16-bit group's spectral report is the float32 analysis's, not rounded to
    the group's dtype (CPL-087): seeded, written by ``step`` and put back by
    ``reset_state`` as float32.  ``rho_spectral``
    is the pass map's radius to about one ``eps`` of the group's dtype times
    the Jacobian's norm, as documented."""
    gm = _affine_pair(dtype, 1, diagnostics=True)
    want = {s: jnp.dtype(jnp.float32) for s in _SPECTRAL_SLOTS}
    assert _slot_dtypes(gm) == want
    gm.step()
    assert _slot_dtypes(gm) == want
    d = gm.coupling_diagnostics()[KEY]
    resid = float(gm._state["_meta"][f"coupling_{KEY}_spectral_residual"])
    r = lambda v: float(np.asarray(jnp.asarray(v, dtype), np.float64))  # noqa: E731
    # Gauss-Seidel: b reads the updated a, so dF/dx = [[0, g_a], [0, g_a g_b]].
    jac = np.array([[0.0, r(0.6)], [0.0, r(0.6) * r(0.9)]])
    radius = r(0.6) * r(0.9)
    eps = float(jnp.finfo(dtype).eps)
    # Read whether or not the bound is usable: the rank-two spectrum is
    # resolved either way (the residual it reports is the slack).
    assert np.isfinite(d["rho_spectral"]), dict(d)
    assert abs(d["rho_spectral"] - radius) <= (
        eps * max(1.0, np.linalg.norm(jac, 2)) + 2 * resid), dict(d)
    gm.reset_state()
    assert _slot_dtypes(gm) == want
