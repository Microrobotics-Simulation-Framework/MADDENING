"""A coupling group whose member holds a zero-size floating field (an empty
contact set, a list of none) steps under every solver and norm, with or
without the diagnostics.

A field with no entries carries no norm and no weight: the residual norms
skip it.  The report's analysis under ``solver="ift"`` with
``diagnostics=True`` does not yet (MADD-ANO-238).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F32 = jnp.float32
NORMS = ("l2", "mixed", "interface")


class _Relay(SimulationNode):
    """``x <- 1 + u0 / 4``, beside a field that has no entries."""

    def initial_state(self):
        return {"x": jnp.full((2,), 0.5, F32), "empty": jnp.zeros((0,), F32)}

    def boundary_input_spec(self):
        return {"u0": BoundaryInputSpec(shape=(2,), dtype=F32, default=jnp.zeros((2,), F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": (0.25 * boundary_inputs["u0"] + 1.0).astype(F32), "empty": state["empty"]}


def _stepped(norm, solver, diagnostics):
    gm = GraphManager()
    gm.add_node(_Relay("a", 1.0))
    gm.add_node(_Relay("b", 1.0))
    gm.add_edge("b", "a", "x", "u0")
    gm.add_edge("a", "b", "x", "u0")
    gm.add_coupling_group(["a", "b"], convergence_norm=norm, solver=solver,
                          diagnostics=diagnostics, rtol=1e-6, tolerance=1e-6)
    gm.compile()
    gm.step()
    return gm


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("solver, diagnostics", [("fori", True), ("ift", False)])
def test_a_group_with_an_empty_field_steps(norm, solver, diagnostics):
    gm = _stepped(norm, solver, diagnostics)
    assert gm.coupling_diagnostics()["a+b"]["converged"] is True
    assert float(jnp.max(jnp.abs(gm.get_node_state("a")["x"] - 4.0 / 3.0))) < 1e-4


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "MADD-ANO-238: a zero-size floating field in a coupling group fails to trace under "
    "solver='ift' with diagnostics=True (a reduction over no entries); pending fix"))
@pytest.mark.parametrize("norm", NORMS)
def test_a_group_with_an_empty_field_steps_under_ift_with_diagnostics(norm):
    gm = _stepped(norm, "ift", True)
    assert gm.coupling_diagnostics()["a+b"]["converged"] is True
