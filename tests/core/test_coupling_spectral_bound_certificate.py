"""The spectral bound applies its resolvent to a residual that is in the space.

``spectral_error_bound`` multiplies the residual by the resolvent norm of
the Krylov-compressed Jacobian.  That norm bounds ``(I - J)^{-1} r`` only
for an ``r`` inside the invariant space it was measured on, and two
fixtures showed the residual outside it with ``spectral_usable=True``:

* **a repeated eigenvalue.**  ``A = B (x) I_2`` -- the same coupling on
  both components of a vector field -- has minimal polynomial of degree
  two, so one start vector's Krylov space broke down (``h = 0``,
  "invariant", "settled") at dimension 2 while the range has dimension
  4.  A zero Arnoldi residual was read as certifying the residual lay in
  the space; the bound read 0.92x the true distance.
* **a dead-banded field on the loop.**  The spectrum was taken in the
  group's norm coordinates, where a field inside the dead band has
  weight zero -- which zeroed its row and column and cut the coupling
  loop out of the spectrum: ``rho_spectral=0.000`` for a radius of 0.9,
  and the bound 0.15-0.29x the true distance of the field the norm keeps.

The Krylov space is now continued from the residual at every breakdown
(and whatever it never absorbs is reported unresolved), and a
dead-banded field keeps its own magnitude's weight in the spectrum,
with its share of the residual -- which ``residual`` does not contain --
measured and folded into the factor.  Both fixtures are linear, so the
fixed point is exact in float64.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import arnoldi_spectral_radius
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Gain(SimulationNode):
    """``x <- g * u + c`` on a length-``n`` field."""

    def __init__(self, name, g, c):
        super().__init__(name=name, timestep=1.0)
        self._g = np.float32(g)
        self._c = jnp.asarray(c, jnp.float32)
        self._n = int(self._c.shape[0])

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._g * boundary_inputs["u"] + self._c}


def _pair(ga, ca, gb, cb, **group_kw):
    gm = GraphManager()
    gm.add_node(_Gain("a", ga, ca))
    gm.add_node(_Gain("b", gb, cb))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


def _group_l2(pairs):
    """The group's L2 norm over ``(got, want)`` field pairs."""
    total = 0.0
    for got, want in pairs:
        ref = max(np.max(np.abs(got)), np.max(np.abs(want)))
        if ref > 0.0:
            total += float(np.sum(((got - want) / ref) ** 2))
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# A repeated eigenvalue
# ---------------------------------------------------------------------------

_ALPHA, _BETA = 23.6584, 0.0024


@pytest.mark.parametrize("c,cap", [
    ((-1.434, 0.824, -0.107, 0.129), 4),
    ((-1.434, 0.824, -0.107, 0.129), 2),
    ((1.365, -1.708, 0.191, -0.078), 4),
])
def test_a_repeated_eigenvalue_does_not_leave_the_residual_outside_the_space(c, cap):
    """Jacobi on ``x_a <- 23.66 x_b + c_a``, ``x_b <- 0.0024 x_a + c_b``, both in R^2."""
    c = np.asarray(c, np.float32)
    gm = _pair(_ALPHA, c[:2], _BETA, c[2:], iteration_mode="jacobi",
               max_iterations=cap, tolerance=1e-9)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    A = np.kron(np.array([[0.0, float(np.float32(_ALPHA))],
                          [float(np.float32(_BETA)), 0.0]]), np.eye(2))
    x_star = np.linalg.solve(np.eye(4) - A, c.astype(np.float64))
    xa, xb = (np.asarray(gm.get_node_state(n)["x"], np.float64) for n in ("a", "b"))
    distance = _group_l2([(xa, x_star[:2]), (xb, x_star[2:])])
    assert distance > 1e-3, "fixture premise: five orders above the float32 floor"
    assert d["rho_spectral"] == pytest.approx(math.sqrt(_ALPHA * _BETA), abs=1e-4)
    assert d["spectral_usable"] is True, d
    assert d["spectral_error_bound"] >= distance, (
        f"bound {d['spectral_error_bound']:.4e} under the true distance "
        f"{distance:.4e} (ratio {d['spectral_error_bound'] / distance:.3f})"
    )


def test_arnoldi_continues_from_the_residual_at_a_breakdown():
    """The mechanism, on the matrix: without the residual the space stops at two.

    With ``v_extra`` the space picks up the residual's missing part at
    the breakdown, breaks down again at four -- the whole space -- and
    the compressed resolvent then bounds ``(I - A)^{-1} r`` exactly as
    the docstring promises.
    """
    A = jnp.kron(jnp.array([[0.0, 2.0], [0.02, 0.0]]), jnp.eye(2))
    v0 = jnp.array([1.0, 2.0, 0.5, -1.0])
    r = jnp.array([1.0, 0.0, 0.0, 1.0])
    exact = float(jnp.linalg.norm(jnp.linalg.solve(jnp.eye(4) - A, r)))
    _rho, res, amp = arnoldi_spectral_radius(lambda v: A @ v, v0)
    assert float(res) == 0.0, "fixture premise: the space breaks down, 'settled'"
    assert float(amp) * float(jnp.linalg.norm(r)) < exact, (
        "fixture premise: the one-vector space's resolvent understates on r"
    )
    _rho, res, amp = arnoldi_spectral_radius(lambda v: A @ v, v0, v_extra=r)
    assert float(res) < 1e-5
    assert float(amp) * float(jnp.linalg.norm(r)) >= exact


def test_a_residual_the_space_never_absorbs_reads_as_unresolved():
    """With no breakdown to continue from, the missed part is reported, not ignored.

    Ten distinct eigenvalues in a two-step space: the residual is not in
    it, and its outside fraction raises the reported residual, so the
    space cannot read as settled on the strength of ``h`` alone.
    """
    lam = jnp.linspace(-0.5, 0.5, 10)
    A = jnp.diag(lam)
    v0 = jnp.ones(10)
    r = jnp.zeros(10).at[3].set(1.0)
    _rho, res_plain, _ = arnoldi_spectral_radius(lambda v: A @ v, v0, n_steps=2)
    _rho, res, _ = arnoldi_spectral_radius(lambda v: A @ v, v0, n_steps=2, v_extra=r)
    assert float(res) >= float(res_plain)
    assert float(res) > 0.5, "most of r is outside a two-dimensional space"


# ---------------------------------------------------------------------------
# A dead-banded field on the loop
# ---------------------------------------------------------------------------

#: ``x_a <- G x_b + 1``, ``x_b <- h x_a``: loop gain ``G h = 0.9``.  With
#: ``h = 1e-9`` the relay field sits at ~1e-9-1e-8 -- a displacement in
#: metres feeding a stiffness of 1e9 -- inside a dead band of 1e-7.
_H, _G = 1e-9, 0.9e9


@pytest.mark.parametrize("norm,knob", [("l2", dict(tolerance=1e-6)),
                                       ("mixed", dict(rtol=1e-4))])
@pytest.mark.parametrize("cap", (3, 10))
def test_a_dead_banded_field_on_the_loop_stays_in_the_spectrum(norm, knob, cap):
    gm = _pair(_G, [1.0], _H, [0.0], max_iterations=cap, convergence_norm=norm,
               atol=1e-7, **knob)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert float(np.max(np.abs(gm.get_node_state("b")["x"]))) <= 1e-7, (
        "fixture premise: the relay field is inside the dead band"
    )
    x_star = 1.0 / (1.0 - _G * _H)
    xa = float(gm.get_node_state("a")["x"][0])
    scale = 1.0 if norm == "l2" else 1e-4
    distance = abs(xa - x_star) / max(abs(xa), x_star) / scale  # the kept field
    assert d["rho_spectral"] == pytest.approx(_G * _H, abs=1e-3), (
        f"rho_spectral={d['rho_spectral']} for a loop gain of {_G * _H}"
    )
    assert d["spectral_usable"] is True, d
    assert d["spectral_error_bound"] >= distance, (
        f"bound {d['spectral_error_bound']:.4e} under the kept field's true "
        f"distance {distance:.4e}"
    )


def test_a_loop_the_dead_band_hides_from_the_residual_is_seen_by_the_bound():
    """Jacobi from rest: the residual reads zero, the kept field is 90% off.

    The first two Jacobi passes leave ``x_a`` at 1 (``x_b`` was 0 in
    both), and ``x_b``'s move from 0 to ``1e-9`` is inside the dead
    band, so the group's residual is exactly zero and it reports
    converged -- the dead band's documented blindness, which no
    ``tolerance`` can contradict.  The spectral bound cannot rescue it
    through the resolvent alone (the residual it multiplies is zero);
    it sees it because the dead-banded field's share of the residual,
    which ``residual`` does not contain, is measured and folded into its
    factor.  Without that share it read 4.3e-4 against a distance of 0.9.
    """
    gm = _pair(_G, [1.0], _H, [0.0], iteration_mode="jacobi", max_iterations=3,
               tolerance=1e-6, atol=1e-7)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    x_star = 1.0 / (1.0 - _G * _H)
    xa = float(gm.get_node_state("a")["x"][0])
    distance = abs(xa - x_star) / max(abs(xa), x_star)
    assert d["converged"] is True and d["error_estimate"] < 1e-6, (
        f"fixture premise: the residual cannot see the loop: {d}"
    )
    assert distance > 0.5, "fixture premise: the kept field is far off"
    assert d["spectral_error_bound"] >= distance, (
        f"bound {d['spectral_error_bound']:.3e} under the kept field's true "
        f"distance {distance:.3e}"
    )
