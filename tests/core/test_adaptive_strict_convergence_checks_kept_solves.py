"""``strict_convergence`` under the adaptive steppers checks only the solves they keep.

``run_adaptive`` and ``run_adaptive_scan`` estimate each attempt's error by
step doubling: one full step and two half steps, of which only the half
steps are kept, and only when the attempt is accepted.  The strict check
used to run inside every one of those solves, so an attempt whose full
step (or whose rejected half steps) did not converge raised -- although
rejecting that attempt is exactly how the controller recovers, and every
solve the run keeps converges.  The principle is MADD-ANO-044's: strict
checks a step that keeps the solve.

The fixture couples two stiff springs both ways; the coupling gain
``dt**2 k / m`` is 3 at ``dt = 0.01`` (Gauss-Seidel diverges) and 0.75 at
the half step.  The first attempt's full step therefore exhausts its cap
unconverged and is discarded.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

KW = dict(dt_initial=0.01, dt_max=0.01, atol=1e-3, rtol=1e-3)
T_END = 0.004


def _pair(strict, max_iterations=30):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30000.0, damping=50.0,
                                 initial_position=-1.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=30000.0, damping=50.0,
                                 initial_position=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=max_iterations, tolerance=1e-5,
                          strict_convergence=strict)
    gm.compile()
    return gm


def _leaves(state):
    return {f"{n}.{f}": np.asarray(v) for n, d in state.items() for f, v in d.items()}


def test_the_discarded_full_step_does_not_converge():
    """The precondition, checked rather than assumed."""
    gm = _pair(False)
    full = jax.jit(gm._build_dt_step_fn())(gm._state, gm._default_external_inputs(),
                                           np.float32(0.01), gm.params)
    iterations = int(full["_meta"]["coupling_a+b_iterations"])
    assert iterations == 30, iterations


def test_run_adaptive_completes_when_every_kept_solve_converges():
    lenient, info = _pair(False).run_adaptive(T_END, **KW)
    assert info["n_rejected"] >= 1, info
    strict_gm = _pair(True)
    strict, strict_info = strict_gm.run_adaptive(T_END, **KW)
    assert strict_info["dt_history"] == info["dt_history"]
    for leaf, value in _leaves(lenient).items():
        np.testing.assert_array_equal(_leaves(strict)[leaf], value, err_msg=leaf)
    assert strict_gm.coupling_diagnostics()["a+b"]["converged"] is True


def test_run_adaptive_scan_completes_when_every_kept_solve_converges():
    lenient, _, info = _pair(False).run_adaptive_scan(T_END, max_steps=12, **KW)
    strict, _, strict_info = _pair(True).run_adaptive_scan(T_END, max_steps=12, **KW)
    assert float(strict_info["final_t"]) == float(info["final_t"]) > 0.0
    for leaf, value in _leaves(lenient).items():
        np.testing.assert_array_equal(_leaves(strict)[leaf], value, err_msg=leaf)


def test_run_adaptive_still_raises_about_a_kept_solve_that_did_not_converge():
    """A cap of two passes leaves even the half steps unconverged."""
    gm = _pair(True, max_iterations=2)
    before = _leaves({n: gm.get_node_state(n) for n in gm.node_names})
    with pytest.raises(RuntimeError, match="without converging"):
        gm.run_adaptive(T_END, **KW)
    after = {n: gm.get_node_state(n) for n in gm.node_names}
    for leaf, value in before.items():
        np.testing.assert_array_equal(_leaves(after)[leaf], value, err_msg=leaf)


def test_run_adaptive_scan_still_raises_about_a_kept_solve_that_did_not_converge():
    gm = _pair(True, max_iterations=2)
    with pytest.raises(Exception, match="without converging"):
        _, _, info = gm.run_adaptive_scan(T_END, max_steps=12, **KW)
        jax.block_until_ready(info["n_steps"])
