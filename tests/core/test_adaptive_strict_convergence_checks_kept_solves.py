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
``dt**2 k / m`` is 12 at ``dt = 0.02``, 3 at 0.01 and 0.75 at 0.005, and
Gauss-Seidel diverges above 1.  From ``dt_initial = 0.02`` with a cap of
ten passes, the first attempt's full step *and* its two half steps exhaust
the cap unconverged (finite, far off); the attempt is rejected and every
solve the run goes on to keep converges.  With ``dt_min = dt_initial =
0.01`` the controller cannot shrink, the stepper accepts the attempt, and
its half steps -- 21 passes to converge at 0.005 -- are kept unconverged.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

#: Recovers by rejection: nothing kept is unconverged.
KW = dict(dt_initial=0.02, dt_max=0.02, atol=1e-3, rtol=1e-3)
#: Cannot shrink: the unconverged half steps are kept.
KW_FORCED = dict(dt_initial=0.01, dt_max=0.01, dt_min=0.01, atol=1e-3, rtol=1e-3)
T_END = 0.004
CAP = 10


def _pair(strict, max_iterations=CAP):
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


def test_the_first_attempt_is_discarded_unconverged():
    """The precondition, checked rather than assumed: its full step and both
    half steps stop at the cap, finite, and the run rejects the attempt."""
    gm = _pair(False)
    step = jax.jit(gm._build_dt_step_fn())
    ext = gm._default_external_inputs()
    full = step(gm._state, ext, np.float32(0.02), gm.params)
    half = step(gm._state, ext, np.float32(0.01), gm.params)
    for state in (full, half):
        assert int(state["_meta"]["coupling_a+b_iterations"]) == CAP
        assert np.isfinite(float(state["_meta"]["coupling_a+b_residual"]))
        assert float(state["_meta"]["coupling_a+b_residual"]) > 1e-5
    _, info = _pair(False).run_adaptive(T_END, **KW)
    assert info["n_rejected"] >= 1 and info["dt_history"][0] < 0.02, info


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
    gm = _pair(True)
    before = _leaves({n: gm.get_node_state(n) for n in gm.node_names})
    with pytest.raises(RuntimeError, match="without converging"):
        gm.run_adaptive(T_END, **KW_FORCED)
    # Raised before the step was accepted: the graph did not move.
    after = _leaves({n: gm.get_node_state(n) for n in gm.node_names})
    for leaf, value in before.items():
        np.testing.assert_array_equal(after[leaf], value, err_msg=leaf)


def test_run_adaptive_accepts_the_same_steps_without_strict():
    """The control: those half steps really are kept when strict is off."""
    gm = _pair(False)
    with pytest.warns(UserWarning, match="hit dt_min"):
        gm.run_adaptive(T_END, **KW_FORCED)
    assert gm.coupling_diagnostics()["a+b"]["converged"] is False


def test_run_adaptive_scan_still_raises_about_a_kept_solve_that_did_not_converge():
    gm = _pair(True)
    with pytest.raises(Exception, match="without converging"):
        _, _, info = gm.run_adaptive_scan(T_END, max_steps=4, **KW_FORCED)
        jax.block_until_ready(info["n_steps"])
