"""Reverse-mode AD through a coupling group with a sharded member, on every push.

``test_coupling_group_with_sharded_and_replicated_members.py`` differentiates
eight steps of a group -- a :class:`ShardedStencilNode` field iterated to
convergence together with a replicated far-field scalar -- under both
solvers, sharded and unsharded, against a float64 model of the implicitly
coupled step.  Its two adjoint tests are slow-marked for the cost of
compiling that (about 21 s of fixture on three cores), so without this
module no push would differentiate through a coupling group with a
partitioned member.

This is the same property at its smallest: the same two nodes, one step,
the default solver (a ``while_loop`` forward and an implicit-function-theorem
adjoint), the field split over two devices, differentiated once with respect
to a trained parameter of each member and the coupling coefficient, and
held to central differences of the same float64 model.

It can fail: an adjoint that treated the converged pass as a constant, or
differentiated only the last pass (the explicitly lagged step), gives a
gradient that the float64 model of that step matches and the implicit one
does not -- the test asserts the two models are far apart at this
tolerance, so the check can tell them apart.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from tests.cloud.multigpu.test_coupling_group_with_sharded_and_replicated_members import (
    DT, EPS, MAX_ITERATIONS, THETA, THETA_F32, FarField, SlabField, _as_f32)

N_DEV = 2
#: Measured worst over the three parameters: see the module docstring of
#: the slow sibling for the float32 error model; one step of it is smaller.
GRAD_RTOL = 16 * EPS

pytestmark = pytest.mark.skipif(len(jax.devices()) < N_DEV,
                                reason=f"needs {N_DEV} CPU-virtual devices")


def _model(theta, *, implicit: bool = True):
    """``loss`` after one step, in float64: implicit (the fixed point) or lagged."""
    d, c, h = (float(t) for t in theta)
    dt = _as_f32(DT)
    f = np.asarray(SlabField().initial_state()["f"], np.float64)
    u = np.float64(np.asarray(FarField().initial_state()["u"]))
    if implicit:
        _, u_new = np.linalg.solve(np.array([[1.0, -dt * h], [-dt * c, 1.0]]),
                                   np.array([f.mean() * (1.0 - dt * h), u * (1.0 - dt * c)]))
    else:                               # one Jacobi pass: each reads the other's old value
        u_new = u + dt * c * (f.mean() - u)
        f_new = f + dt * (d * (np.roll(f, 1) - 2 * f + np.roll(f, -1)) + h * (u - f))
        return float(np.sum(f_new ** 2) + u_new ** 2)
    f = f + dt * (d * (np.roll(f, 1) - 2 * f + np.roll(f, -1)) + h * (u_new - f))
    return float(np.sum(f ** 2) + u_new ** 2)


def _model_gradient(theta, *, implicit: bool = True) -> np.ndarray:
    theta = np.asarray(theta, np.float64)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6 * theta[i]
        grad[i] = (_model(theta + step, implicit=implicit)
                   - _model(theta - step, implicit=implicit)) / (2 * step[i])
    return grad


def test_reverse_mode_ad_through_a_coupling_group_with_a_sharded_member():
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(SlabField(), create_device_mesh(shape=(N_DEV,)),
                                   axis_map={"devices": 0}, boundary="periodic"))
    gm.add_node(FarField())
    gm.add_edge("field", "far", "f", "field_mean", transform=jnp.mean)
    gm.add_edge("far", "field", "u", "ambient")
    group = gm.add_coupling_group(["field", "far"], max_iterations=MAX_ITERATIONS,
                                  tolerance=1e-7)
    assert group.solver == "ift", "the default solver moved; re-derive the tolerance"
    gm.compile()
    step_fn = gm._build_step_fn()  # noqa: SLF001 -- the step sysid differentiates
    ext = gm._resolve_external_inputs(None)  # noqa: SLF001
    state0 = gm._state  # noqa: SLF001

    def loss(theta):
        params = jax.tree.map(lambda x: x, gm.params)
        params["nodes"]["field"]["diffusivity"] = theta[0]
        params["nodes"]["far"]["conductance"] = theta[1]
        params["nodes"]["field"]["exchange"] = theta[2]
        final = step_fn(state0, ext, params)
        value = jnp.sum(final["field"]["f"] ** 2) + final["far"]["u"] ** 2
        return value, final["field"]["f"]

    (value, field), grad = jax.jit(jax.value_and_grad(loss, has_aux=True))(
        jnp.asarray(THETA, jnp.float32))

    # The field really is split across the devices inside the traced step.
    assert len(field.sharding.device_set) == N_DEV and not field.sharding.is_fully_replicated
    want = _model_gradient(THETA_F32)
    got = np.asarray(grad, np.float64)
    rel = np.abs(got - want) / np.abs(want)
    assert np.all(rel <= GRAD_RTOL), (
        f"relative error {np.array2string(rel / EPS, precision=2)} eps, allowed "
        f"{GRAD_RTOL / EPS:.0f} eps (got {got}, want {want})")
    assert float(value) == pytest.approx(_model(THETA_F32), rel=4 * EPS)
    # Non-vacuity: the lagged (one-pass) step's gradient is far outside the
    # tolerance, so an adjoint that saw only one pass would fail above.
    lagged = _model_gradient(THETA_F32, implicit=False)
    assert np.max(np.abs(lagged - want) / np.abs(want)) > 1e2 * GRAD_RTOL, (lagged, want)
