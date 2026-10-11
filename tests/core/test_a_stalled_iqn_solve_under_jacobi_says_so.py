"""IQN under the Jacobi schedule can stall where the plain Jacobi loop converges, and it says so.

The pair: two float32 relays ``x <- 0.3 x_pre + tanh(G u + b)`` of 2 and 4
entries, each reading the other through a matrix mapping (one edge
negated), default norm and solver.  An ordinary contraction: at a
tolerance of 1e-6 plain Jacobi stops in 12 passes and plain Gauss-Seidel
in 6.  Under ``iteration_mode="jacobi"`` with ``acceleration="iqn-ils"``
or ``"iqn-imvj"`` the same solve runs to its cap of 50 passes and returns
a state 0.14 from the fixed point, with a residual of 0.18.

What is held:

* the stall is **reported**: ``converged=False`` with
  ``iterations == max_iterations`` and a residual above the tolerance, and
  ``strict_convergence=True`` raises -- nothing presents the returned
  state as a result;
* what the guide sends a user to instead converges on the same pair:
  Gauss-Seidel with either IQN, and Jacobi with no acceleration or with
  Aitken;
* a strict xfail asks for what the accelerator ought to do -- converge
  where the plain loop does -- so a change that ends the stall is noticed
  (the open entry on IQN under the Jacobi schedule in
  ``docs/validation/known_anomalies.yaml``).

How firmly the pair stalls (measured 2026-10-11 on four pinned cores of an
x86-64 laptop).  On jax 0.10.2, 0.11.0 and 0.11.2 the residual at the cap
is 0.177 at tolerances of 1e-4 and 1e-6: 1.8e3 and 1.8e5 tolerances.  With
every constant of the pair multiplied entrywise by ``1 + s N(0, 1)`` for
``s`` = 1e-7, 1e-6, 1e-4 and 1e-2 (12 draws of each, jax 0.11.0) neither
IQN acceleration converged on any of the 48 pairs -- residuals at the cap
from 2.8e-4 to 0.99, and from 0.092 for the two smallest ``s`` -- and
plain Jacobi converged on all of them in 11 or 12 passes.  So the verdict
does not hang on one machine's last bit.  The residual does (0.09 to 0.38
at ``s`` = 1e-7), and it is not pinned: the tests ask for the verdict, the
pass count at the cap, and a residual above the tolerance.
"""

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F = jnp.float32
CAP = 50
TOLERANCE = 1e-6
ALPHA = 0.3

# The pair's constants, as the float32 values the graph holds.
GA = [[1.0204595, -1.2778325],
      [0.20904942, -0.2838848]]
GB = [[-0.22632465, -0.107798584, -1.0099931, -0.115966186],
      [-0.43260655, 1.6614997, 0.112893306, -0.1763154],
      [-0.14064372, -0.33402318, -0.52757525, -0.19540049],
      [0.2409727, -0.11927681, 0.47887936, -0.099901065]]
BA = [0.024259565, 1.5458208]
BB = [0.5451055, -0.50522876, -0.18283898, 0.54052514]
#: ``a -> b``: 2 entries onto 4.
H_AB = [[0.4, 0.2], [-0.5, 0.3], [0.8, 0.0], [0.0, -0.6]]
#: ``b -> a``: 4 entries onto 2; the edge negates what it delivers.
H_BA = [[0.064061694, 0.46907306, -1.9797136, 0.7149148],
        [-0.6717513, -1.1680338, 0.19351204, 0.49038142]]

IQN = ("iqn-ils", "iqn-imvj")
#: The loops that converge on the pair: the plain ones, and what the guide
#: recommends in place of IQN under Jacobi.
CONVERGING = (("jacobi", "none"), ("jacobi", "aitken"),
              ("gauss-seidel", "iqn-ils"), ("gauss-seidel", "iqn-imvj"))


class Relay(SimulationNode):
    """``x <- alpha x + tanh(G u + b)``, declaring one evaluation per update."""

    def __init__(self, name, G, b):
        G = jnp.asarray(G, F)
        super().__init__(name, 0.01, G=G, b=jnp.asarray(b, F), alpha=jnp.asarray(ALPHA, F))
        self._n, self._m = G.shape

    def initial_state(self):
        return {"x": jnp.linspace(0.1, 0.4, self._n, dtype=F)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._m,), dtype=F,
                                       default=jnp.zeros(self._m, F))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.zeros(self._m, F))
        return {"x": p["alpha"] * state["x"] + jnp.tanh(p["G"] @ u + p["b"])}

    def update_evaluations(self):
        return 1


def _pair(mode, acceleration, **group):
    gm = GraphManager()
    gm.add_node(Relay("a", GA, BA))
    gm.add_node(Relay("b", GB, BB))
    gm.add_edge("a", "b", "x", "u", mapping=matrix_mapping(jnp.asarray(H_AB, F)))
    gm.add_edge("b", "a", "x", "u", mapping=matrix_mapping(jnp.asarray(H_BA, F)),
                transform="negate")
    gm.add_coupling_group(["a", "b"], max_iterations=CAP, tolerance=TOLERANCE,
                          iteration_mode=mode, acceleration=acceleration, **group)
    gm.compile()
    return gm


def _fixed_point(a0, b0):
    """The step's fixed point in float64, by plain passes in NumPy."""
    Ga, Gb = np.asarray(GA, np.float32).astype(np.float64), np.asarray(GB, np.float32).astype(np.float64)
    ba, bb = np.asarray(BA, np.float32).astype(np.float64), np.asarray(BB, np.float32).astype(np.float64)
    Hab = np.asarray(H_AB, np.float32).astype(np.float64)
    Hba = np.asarray(H_BA, np.float32).astype(np.float64)
    alpha = float(np.float32(ALPHA))
    a, b = a0.copy(), b0.copy()
    for _ in range(2000):
        a_new = alpha * a0 + np.tanh(Ga @ -(Hba @ b) + ba)
        b_new = alpha * b0 + np.tanh(Gb @ (Hab @ a) + bb)
        moved = max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b)))
        a, b = a_new, b_new
        if moved < 1e-15:
            return a, b
    raise AssertionError("premise: the pair's plain iteration converges in float64")


@functools.lru_cache(maxsize=None)
def _solved(mode, acceleration):
    """One step of the pair: its report, and how far the returned state is
    from the fixed point.  Plain numbers only (no graph is kept)."""
    gm = _pair(mode, acceleration)
    a_ref, b_ref = _fixed_point(np.asarray(gm.get_node_state("a")["x"], np.float64),
                                np.asarray(gm.get_node_state("b")["x"], np.float64))
    gm.step()
    report = gm.coupling_diagnostics()["a+b"]
    error = max(np.max(np.abs(np.asarray(gm.get_node_state("a")["x"], np.float64) - a_ref)),
                np.max(np.abs(np.asarray(gm.get_node_state("b")["x"], np.float64) - b_ref)))
    return {"iterations": int(report["iterations"]), "converged": bool(report["converged"]),
            "residual": float(report["residual"]), "error": float(error)}


@pytest.mark.parametrize("mode, acceleration", CONVERGING)
def test_the_plain_loop_and_the_recommended_combinations_converge_on_the_pair(mode, acceleration):
    """The pair is an ordinary one: every loop here is within the tolerance
    well inside the cap (measured: 10 to 35 passes of 50) and a few 1e-7
    from the fixed point."""
    solved = _solved(mode, acceleration)
    assert solved["converged"] and solved["iterations"] < CAP, solved
    assert solved["residual"] <= TOLERANCE, solved
    assert solved["error"] < 1e-5, solved


@pytest.mark.parametrize("acceleration", IQN)
def test_iqn_under_jacobi_reports_the_solve_it_did_not_finish(acceleration):
    """At the cap, with a residual above the tolerance and a state that is
    not the fixed point: ``converged`` is ``False``."""
    solved = _solved("jacobi", acceleration)
    assert solved["converged"] is False and solved["iterations"] == CAP, solved
    assert solved["residual"] > TOLERANCE, solved
    # Not a false alarm: the state is far outside what the loops above reach.
    assert solved["error"] > 1e3 * TOLERANCE, solved


@pytest.mark.parametrize("acceleration", IQN)
def test_strict_convergence_raises_on_the_stalled_solve(acceleration):
    """With the message for a finite state that ran out of passes (the one
    for a non-finite state reads "without converging: its state ...")."""
    gm = _pair("jacobi", acceleration, strict_convergence=True)
    with pytest.raises(Exception, match=f"max_iterations={CAP} without converging; "):
        gm.step()


@pytest.mark.xfail(strict=True, reason=(
    "IQN under the Jacobi schedule stalls on this pair (the open entry on it in "
    "docs/validation/known_anomalies.yaml): no safeguard returns to a plain pass "
    "when the accelerated residual grows"))
@pytest.mark.parametrize("acceleration", IQN)
def test_iqn_under_jacobi_converges_where_plain_jacobi_does(acceleration):
    """What the accelerator ought to do.  Strict: a change that ends the
    stall fails here until this mark and the registry entry are revisited."""
    assert _solved("jacobi", "none")["converged"], "premise: the plain loop converges"
    solved = _solved("jacobi", acceleration)
    assert solved["converged"] and solved["error"] < 1e-5, solved
