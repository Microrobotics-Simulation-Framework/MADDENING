"""IQN-IMVJ warm-starts from the same secant columns under both solvers.

``acceleration="iqn-imvj"`` carries the first ``jacobian_reuse`` secant
columns of one step into the next.  Under ``solver="ift"`` the loop
stops on the pass that meets the criterion, so what it carries are the
columns that pass left.  The legacy ``"fori"`` loop runs to
``max_iterations`` whatever happens and freezes the state once
converged; every pass after that measures the same residual and the same
raw output, and it used to shift a zero secant column in each time.  By
the cap the warm-start window was all zeros: ``jacobian_reuse`` did
nothing under ``"fori"``, and from the second step on the two solvers
took different passes, reported different ``iterations`` and returned
different states -- against the documented parity (migrating a graph
between them "does not move the answer or the verdict").

The graph is affine in its coupled state, so the previous step's secant
columns describe this step's Jacobian exactly and a working warm start
visibly shortens the solve.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
N = 3
REUSE = 4
STEPS = 6

_M_A = np.array([[0.6, 0.2, -0.1], [0.1, 0.5, 0.2], [-0.2, 0.1, 0.7]], np.float32)
_M_B = np.array([[0.8, -0.1, 0.1], [0.2, 0.7, 0.0], [0.0, 0.2, 0.9]], np.float32)


class _Drive(SimulationNode):
    """``t <- t + dt``; the forcing ``f = (sin t, cos t, sin 2t)`` of node ``a``."""

    def __init__(self, name, timestep):
        super().__init__(name=name, timestep=timestep)

    def initial_state(self):
        return {"t": jnp.float32(0.0), "f": jnp.zeros(N, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        t = state["t"] + dt
        return {"t": t, "f": jnp.stack([jnp.sin(t), jnp.cos(t), jnp.sin(2.0 * t)])}


class _Linear(SimulationNode):
    """``x <- M @ u (+ f)``."""

    def __init__(self, name, timestep, matrix, forced):
        super().__init__(name=name, timestep=timestep)
        self._matrix = matrix
        self._forced = forced

    def initial_state(self):
        return {"x": jnp.zeros(N, jnp.float32)}

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(N,), dtype=jnp.float32,
                                       default=jnp.zeros(N, jnp.float32))}
        if self._forced:
            spec["f"] = BoundaryInputSpec(shape=(N,), dtype=jnp.float32,
                                          default=jnp.zeros(N, jnp.float32))
        return spec

    def update(self, state, boundary_inputs, dt):
        x = jnp.asarray(self._matrix) @ boundary_inputs.get("u", jnp.zeros(N))
        if self._forced:
            x = x + boundary_inputs.get("f", jnp.zeros(N))
        return {"x": x.astype(jnp.float32)}


def _graph(solver):
    gm = GraphManager()
    gm.add_node(_Drive("drive", 0.1))
    gm.add_node(_Linear("a", 0.1, _M_A, forced=True))
    gm.add_node(_Linear("b", 0.1, _M_B, forced=False))
    gm.add_edge("drive", "a", "f", "f")
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], acceleration="iqn-imvj",
                          jacobian_reuse=REUSE, max_iterations=30,
                          tolerance=1e-5, solver=solver, diagnostics=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # solver="fori" is deprecated
        gm.compile()
    return gm


@pytest.fixture(scope="module")
def runs():
    """``{solver: [(report, state vector, V) after step 1..STEPS]}``."""
    out = {}
    for solver in ("ift", "fori"):
        gm = _graph(solver)
        rows = []
        for _ in range(STEPS):
            gm.step()
            state = np.concatenate([np.asarray(gm.get_node_state(n)["x"])
                                    for n in ("a", "b")])
            rows.append((dict(gm.coupling_diagnostics()[KEY]), state,
                         np.asarray(gm._state["_meta"][f"coupling_{KEY}_V"])))
        out[solver] = rows
    return out


def test_the_solvers_take_the_same_passes_every_step(runs):
    for step, (ift, fori) in enumerate(zip(runs["ift"], runs["fori"]), start=1):
        assert fori[0]["iterations"] == ift[0]["iterations"], (
            step, fori[0]["iterations"], ift[0]["iterations"])
        assert fori[0]["converged"] and ift[0]["converged"], step


def test_the_solvers_return_the_same_state(runs):
    """To float32 round-off: two ulps of the largest entry.

    Bit-identical on jaxlib 0.11.0; 2e-07 to 1.3e-06 apart, relative to
    the largest entry, from the second step on while ``"fori"`` warmed
    from zeros.
    """
    eps = float(np.finfo(np.float32).eps)
    for step, (ift, fori) in enumerate(zip(runs["ift"], runs["fori"]), start=1):
        gap = float(np.max(np.abs(fori[1] - ift[1])) / np.max(np.abs(ift[1])))
        assert gap <= 2 * eps, (step, gap)


def test_fori_carries_the_latching_passs_columns(runs):
    """The warm-start window holds real columns, the ones ``"ift"`` carries."""
    for step, (ift, fori) in enumerate(zip(runs["ift"], runs["fori"]), start=1):
        window_fori = fori[2][:, :REUSE]
        assert np.any(window_fori != 0.0), (step, window_fori)
        np.testing.assert_allclose(window_fori, ift[2][:, :REUSE],
                                   rtol=1e-4, atol=1e-6, err_msg=f"step {step}")


def test_the_warm_start_shortens_the_solve(runs):
    """The point of ``jacobian_reuse``: later steps need fewer passes."""
    first = runs["fori"][0][0]["iterations"]
    later = [row[0]["iterations"] for row in runs["fori"][1:]]
    assert max(later) < first, (first, later)
