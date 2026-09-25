"""``gradient_relative_error_bound`` under ``jax.vmap``: rounding, not a defect.

Batched over two states of one multi-rate graph, the bound of the step
that fires read 1.84e-06 where the same step unbatched read 2.01e-06
(jaxlib 0.11.0).  The cause is the bound's curvature factor: a difference
of two Jacobian-vector products of the map, here twelve float32 ulps of
the products, which the batched program rounds one ulp differently.  So
the bound's leading digit is at its own float resolution -- about
``amplification * eps`` relative, the resolution of the gradient itself --
and both values still bound the true error.

The map is bilinear in the pair's state and the clock's phase, so the
true error is exact: the IFT tangent in the phase direction is
proportional to ``b`` (the Jacobian in the state does not depend on the
state), and its relative error at the returned iterate is
``|b_k / b* - 1|`` with ``b* = 2/3``.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

# Compiling the spectral and gradient-bound machinery twice (the step and
# its vmap) costs 17 s on a three-core slice, over the per-test budget.
# Per push, the bound read through one vmapped compile of a simpler group,
# member by member against the closed-form error:
# tests/core/test_coupling_gradient_bound_through_a_vmapped_step.py::test_the_gradient_bound_holds_for_each_member_of_a_vmapped_batch
pytestmark = pytest.mark.slow

KEY = "a+b"
BOUND = f"coupling_{KEY}_gradient_relative_error_bound"
B_STAR = 2.0 / 3.0          # a = 0.5 * b + 1, b = 0.5 * a at phase 0


class _Phase(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name=name, timestep=timestep)

    def initial_state(self):
        return {"phase": jnp.float32(-1.0)}

    def update(self, state, boundary_inputs, dt):
        return {"phase": jnp.mod(state["phase"] + 1.0, 10.0)}


class _Gained(SimulationNode):
    """``x <- (0.5 + 0.5 * phase) * u + bias``."""

    def __init__(self, name, timestep, bias, phased):
        super().__init__(name=name, timestep=timestep, bias=bias)
        self._phased = phased

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}
        if self._phased:
            spec["phase"] = BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                              default=jnp.float32(0.0))
        return spec

    def update(self, state, boundary_inputs, dt):
        gain = 0.5 + 0.5 * boundary_inputs.get("phase", jnp.float32(0.0))
        return {"x": gain * boundary_inputs.get("u", jnp.float32(0.0))
                + self.params["bias"]}


@pytest.fixture(scope="module")
def steps():
    """``(one_at_a_time, batched)`` for the firing step (count 10) and a
    non-firing one (count 15, where the discarded solve diverges)."""
    gm = GraphManager()
    gm.add_node(_Phase("clock", 0.001))
    gm.add_node(_Gained("a", 0.01, bias=1.0, phased=True))
    gm.add_node(_Gained("b", 0.01, bias=0.0, phased=False))
    gm.add_edge("clock", "a", "phase", "phase")
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=10, tolerance=1e-5,
                          diagnostics=True, predictor="linear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate INFO notice
        gm.compile()
    states = []
    for count in range(16):
        if count in (10, 15):
            states.append(gm._state)
        gm.step()
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()
    single = gm._compiled_step(states[0], ext)
    batched = jax.jit(jax.vmap(step, in_axes=(0, None)))(
        jax.tree.map(lambda a, b: jnp.stack([a, b]), *states), ext)
    return single, jax.tree.map(lambda leaf: leaf[0], batched)


def test_the_forward_and_its_report_are_identical_under_vmap(steps):
    """The state and the loop's own numbers to the bit; the spectrum to ulps.

    The Arnoldi runs batched linear algebra, which rounds differently:
    ``rho_spectral`` read 0.24999999 batched against 0.25 (two ulps).
    """
    single, batched = steps
    for name in ("a", "b"):
        assert (np.asarray(batched[name]["x"]).tobytes()
                == np.asarray(single[name]["x"]).tobytes()), name
    for suffix in ("iterations", "residual", "amplification"):
        slot = f"coupling_{KEY}_{suffix}"
        assert (np.asarray(batched["_meta"][slot]).tobytes()
                == np.asarray(single["_meta"][slot]).tobytes()), slot
    for suffix in ("rho_spectral", "spectral_amplification"):
        slot = f"coupling_{KEY}_{suffix}"
        np.testing.assert_allclose(np.asarray(batched["_meta"][slot]),
                                   np.asarray(single["_meta"][slot]),
                                   rtol=8 * float(np.finfo(np.float32).eps),
                                   err_msg=slot)


def test_both_bounds_hold_and_agree_to_their_own_resolution(steps):
    single, batched = steps
    b_k = float(single["b"]["x"])
    true = abs(b_k / B_STAR - 1.0)
    assert true > 0.0                   # the forward stopped short: a real error
    unbatched_bound = float(single["_meta"][BOUND])
    batched_bound = float(batched["_meta"][BOUND])
    assert unbatched_bound >= true, (unbatched_bound, true)
    assert batched_bound >= true, (batched_bound, true)
    # One or two ulps of the products in a twelve-ulp difference: the two
    # programs may disagree by that much and no more.
    assert abs(batched_bound - unbatched_bound) <= 0.25 * max(
        batched_bound, unbatched_bound), (batched_bound, unbatched_bound)
