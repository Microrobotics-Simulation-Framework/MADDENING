"""A coupling group with one sharded member and one replicated member.

This is the production shape of a sharded field solver: the field is
partitioned across the devices by :class:`ShardedStencilNode` and is
iterated to convergence, inside one coupling group, together with a small
unsharded solver it trades interface values with on every pass -- here a
far-field value that the field relaxes towards, and the field's mean that
drives the far field back.

The invariant: the group's forward rollout, and ``jax.grad`` of a loss on
it with respect to trained parameters of *both* members, are those of the
same group with the sharded member replaced by the node it wraps.  Under
the default solver (``"ift"``: a ``while_loop`` forward, an
implicit-function-theorem adjoint solved by GMRES) and under the
deprecated ``"fori"`` solver.  Both paths are also held to an independent
float64 NumPy model of the implicitly coupled step, so "agrees with the
unsharded group" cannot pass on two copies of the same wrong answer.

Before this module nothing under ``tests/`` put a partitioned array inside
a coupling group.  ``test_sharded_wrapper_coupling_hooks.py`` couples a
``ShardedPointwiseNode`` to a relay under ``solver="ift"``, forward only;
its wrapped spring's state was 0-d, living whole on one device, until the
wrapper began refusing a node with nothing to shard, and it is now a batch
of independent springs, one per device, with nothing exchanged between
shards.  ``test_coupled_sharded.py`` couples two sharded
nodes by staggered edges, with no group and a 10% tolerance; the graph
property in ``test_property_sharded_equals_unsharded.py`` drives a sharded
node through a one-way edge; ``test_graph_multigpu.py`` places whole nodes
on devices.

Tolerances, in float32 machine epsilon, are ~4x the worst difference
measured over 2 and 4 virtual CPU devices, 16 and 64 cells, and 1, 8 and 32
steps, for both solvers, on jaxlib 0.10.2, 0.11.0 and 0.11.2 (the three
gave identical numbers):

* sharded against unsharded: state 0.95 eps of the field's scale, loss
  1.97 eps, gradient 2.34 eps (``"ift"``) and 4.58 eps (``"fori"``);
* either against the float64 model: state and loss 2.6 eps, gradient
  5.7 eps (``"ift"``) and 19.6 eps (``"fori"``, which differentiates the
  iterate it returned rather than the fixed point, so its distance from
  the exact derivative is set by the group's tolerance, not by round-off).

The differences come from the field's mean -- a cross-device reduction on
the sharded path -- and from the convergence test, which is a reduction
over the partitioned state too: on 2 devices the sharded group took 4
passes where the unsharded one took 3, and the states still agreed to
0.6 eps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec

N_DEV = 4
N_CELLS = 16
STEPS = 8
DT = 0.05
#: (field diffusivity, far-field conductance, field exchange coefficient):
#: a trained parameter of the sharded member, one of the replicated member
#: and the coefficient of the coupling itself.
THETA = (0.3, 2.0, 0.8)
GROUP = "far+field"
MAX_ITERATIONS = 40

EPS = float(np.finfo(np.float32).eps)
#: Sharded against unsharded (measured worst in the module docstring).
STATE_RTOL = 4 * EPS
LOSS_RTOL = 8 * EPS
GRAD_RTOL = 16 * EPS
#: Either path against the float64 model of the coupled step.
REFERENCE_STATE_RTOL = 16 * EPS
REFERENCE_GRAD_RTOL = {"ift": 32 * EPS, "fori": 128 * EPS}

#: ``None`` is CouplingGroup's default solver, passed as nothing at all so
#: that the default is what this covers; ``"fori"`` is the deprecated one.
SOLVERS = [None, "fori"]
SOLVER_IDS = ["default", "fori"]

pytestmark = pytest.mark.skipif(len(jax.devices()) < N_DEV,
                                reason=f"needs {N_DEV} CPU-virtual devices")


class SlabField(SimulationNode):
    """1-D periodic diffusion relaxing towards a far-field value.

    ``f <- f + dt * (diffusivity * lap(f) + exchange * (ambient - f))``.
    The halo-1 stencil makes it a :class:`ShardedStencilNode` member, and
    with periodic halos its ``update_padded`` is exactly its ``update``.
    """

    def __init__(self, name: str = "field", n_cells: int = N_CELLS,
                 diffusivity: float = THETA[0], exchange: float = THETA[2]) -> None:
        super().__init__(name=name, timestep=DT, diffusivity=diffusivity,
                         exchange=exchange)
        self._n = int(n_cells)

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def state_fields(self) -> list[str]:
        return ["f"]

    def initial_state(self) -> dict:
        x = np.linspace(0.0, 1.0, self._n, endpoint=False, dtype=np.float32)
        f = np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x) + 1.0
        return {"f": jnp.asarray(f, jnp.float32)}

    def boundary_input_spec(self) -> dict:
        return {"ambient": BoundaryInputSpec(shape=(), description="far-field value")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "diffusivity": ParamSpec(bounds=(0.0, None), transform="log"),
                "exchange": ParamSpec(bounds=(0.0, None), transform="log")}

    @staticmethod
    def _step(f_pad, ambient, p, dt):
        f = f_pad[1:-1]
        lap = f_pad[2:] - 2 * f + f_pad[:-2]
        return f + dt * (p["diffusivity"] * lap + p["exchange"] * (ambient - f))

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        f = state["f"]
        ambient = jnp.asarray(boundary_inputs.get("ambient", 0.0), jnp.float32)
        return {"f": self._step(jnp.concatenate([f[-1:], f, f[:1]]), ambient, p, dt)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        f_pad = state_padded["f"]
        ambient = jnp.asarray(boundary_inputs.get("ambient", 0.0), jnp.float32)
        return {"f": jnp.concatenate([f_pad[:1], self._step(f_pad, ambient, p, dt),
                                      f_pad[-1:]])}


class FarField(SimulationNode):
    """A replicated scalar ``u <- u + dt * conductance * (field_mean - u)``."""

    def __init__(self, name: str = "far", conductance: float = THETA[1]) -> None:
        super().__init__(name=name, timestep=DT, conductance=conductance)

    def state_fields(self) -> list[str]:
        return ["u"]

    def initial_state(self) -> dict:
        return {"u": jnp.asarray(0.2, jnp.float32)}

    def boundary_input_spec(self) -> dict:
        return {"field_mean": BoundaryInputSpec(shape=(), description="mean of the field")}

    def param_specs(self) -> dict:
        return {**super().param_specs(),
                "conductance": ParamSpec(bounds=(0.0, None), transform="log")}

    def update(self, state, boundary_inputs, dt, *, params=None) -> dict:
        p = self.params if params is None else {**self.params, **params}
        mean = jnp.asarray(boundary_inputs.get("field_mean", 0.0), jnp.float32)
        return {"u": state["u"] + dt * p["conductance"] * (mean - state["u"])}


def _graph(*, sharded: bool, solver: str | None) -> GraphManager:
    """The group, with the field sharded over ``N_DEV`` devices or not."""
    field = SlabField()
    gm = GraphManager()
    if sharded:
        gm.add_node(ShardedStencilNode(field, create_device_mesh(shape=(N_DEV,)),
                                       axis_map={"devices": 0}, boundary="periodic"))
    else:
        gm.add_node(field)
    gm.add_node(FarField())
    gm.add_edge("field", "far", "f", "field_mean", transform=jnp.mean)
    gm.add_edge("far", "field", "u", "ambient")
    if solver is None:
        group = gm.add_coupling_group(["field", "far"], max_iterations=MAX_ITERATIONS,
                                      tolerance=1e-7)
        assert group.solver == "ift", "the default solver moved; re-derive the tolerances"
    else:
        with pytest.warns(DeprecationWarning, match="fori"):
            gm.add_coupling_group(["field", "far"], max_iterations=MAX_ITERATIONS,
                                  tolerance=1e-7, solver=solver)
    gm.compile()
    return gm


def _host(tree) -> dict:
    return jax.tree.map(lambda a: np.asarray(jax.device_get(a)), tree)


# ---------------------------------------------------------------------------
# The float64 model: the fixed point of one coupling pass, solved exactly
# ---------------------------------------------------------------------------


def _as_f32(x) -> float:
    """The float32 value the graph actually uses, widened exactly to float64."""
    return float(np.float32(x))


#: ``THETA`` as the graph sees it: the model below is evaluated at these.
THETA_F32 = tuple(_as_f32(t) for t in THETA)


def reference(theta, steps: int = STEPS):
    """``(f, u, loss)`` of the implicitly coupled rollout, in float64.

    At convergence each step solves both updates at once, each node reading
    the *other's* new value: ``f' = f + dt (D lap f + h (u' - f))`` and
    ``u' = u + dt c (mean f' - u)``.  The periodic Laplacian has zero mean,
    so ``mean f'`` and ``u'`` satisfy a 2x2 linear system, and ``f'`` follows.
    """
    d, c, h = (float(t) for t in theta)
    dt = _as_f32(DT)
    f = np.asarray(SlabField().initial_state()["f"], np.float64)
    u = np.float64(np.asarray(FarField().initial_state()["u"]))
    for _ in range(steps):
        m = f.mean()
        m_new, u_new = np.linalg.solve(
            np.array([[1.0, -dt * h], [-dt * c, 1.0]]),
            np.array([m * (1.0 - dt * h), u * (1.0 - dt * c)]))
        f = f + dt * (d * (np.roll(f, 1) - 2 * f + np.roll(f, -1)) + h * (u_new - f))
        u = u_new
    return f, float(u), float(np.sum(f ** 2) + u ** 2)


def reference_gradient(theta) -> np.ndarray:
    """Central differences of the float64 model's loss (relative step 1e-6)."""
    theta = np.asarray(theta, np.float64)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = 1e-6 * theta[i]
        grad[i] = (reference(theta + step)[2] - reference(theta - step)[2]) / (2 * step[i])
    return grad


def _assert_close(got, want, rtol: float, what: str) -> None:
    """``max|got - want| <= rtol * max|want|``, the error in eps in the message."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = float(np.max(np.abs(want)))
    err = float(np.max(np.abs(got - want)))
    assert err <= rtol * scale, (
        f"{what}: max|diff| = {err:.3e} = {err / scale / EPS:.2f} eps of the scale "
        f"{scale:.3e}; allowed {rtol / EPS:.0f} eps")


def _assert_close_each(got, want, rtol: float, what: str) -> None:
    """Componentwise relative agreement, for gradients of unlike size."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    rel = np.abs(got - want) / np.abs(want)
    assert np.all(rel <= rtol), (
        f"{what}: relative error {np.array2string(rel / EPS, precision=2)} eps, "
        f"allowed {rtol / EPS:.0f} eps (got {got}, want {want})")


# ---------------------------------------------------------------------------
# Forward: the public run_scan, once per solver and path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=SOLVERS, ids=SOLVER_IDS)
def forward(request):
    solver = request.param
    out = {}
    for sharded in (True, False):
        gm = _graph(sharded=sharded, solver=solver)
        final = gm.run_scan(STEPS)
        out[sharded] = {
            "field": final["field"]["f"],          # device array: keeps its sharding
            "state": _host({n: final[n] for n in ("field", "far")}),
            # The pass count the default solver records on every step
            # (``coupling_diagnostics()`` reads the same entry but costs
            # seconds of compile per call).  ``"fori"`` records none
            # without ``diagnostics=True``: it runs every pass regardless.
            "iterations": gm._state.get("_meta", {}).get(  # noqa: SLF001
                f"coupling_{GROUP}_iterations"),
        }
    return solver, out


def test_the_forward_rollout_matches_the_group_with_the_unsharded_inner_node(forward):
    solver, out = forward
    sharded, unsharded = out[True]["state"], out[False]["state"]
    _assert_close(sharded["field"]["f"], unsharded["field"]["f"], STATE_RTOL,
                  f"{solver}: field")
    _assert_close(sharded["far"]["u"], unsharded["far"]["u"], STATE_RTOL,
                  f"{solver}: far field")


def test_the_forward_rollout_is_the_implicitly_coupled_step(forward):
    """Both paths against the float64 model -- and the model is not the
    uncoupled rollout, so the check sees the coupling itself."""
    solver, out = forward
    f_ref, u_ref, _ = reference(THETA_F32)
    for sharded in (True, False):
        state = out[sharded]["state"]
        label = f"{solver}, {'sharded' if sharded else 'unsharded'}"
        _assert_close(state["field"]["f"], f_ref, REFERENCE_STATE_RTOL, f"{label}: field")
        _assert_close(state["far"]["u"], u_ref, REFERENCE_STATE_RTOL, f"{label}: far field")
    # With the far field frozen at its initial value the field's mean would
    # end elsewhere by orders of magnitude more than any tolerance above.
    f_frozen, _, _ = reference((THETA_F32[0], 0.0, THETA_F32[2]))
    assert abs(f_frozen.mean() - f_ref.mean()) > 1e3 * REFERENCE_STATE_RTOL * np.abs(f_ref).max()


def test_the_sharded_member_is_partitioned_and_the_group_iterates(forward):
    """What makes the two checks above mean something: the field really is
    split across the devices on the sharded path, and under the default
    solver the last step took more than one coupling pass on both paths
    and stopped before its budget (``iterations == max_iterations`` exactly
    when it ran out)."""
    solver, out = forward
    sharding = out[True]["field"].sharding
    assert len(sharding.device_set) == N_DEV and not sharding.is_fully_replicated, sharding
    assert len(out[False]["field"].sharding.device_set) == 1
    if solver is not None:
        return
    for sharded in (True, False):
        iterations = int(out[sharded]["iterations"])
        assert 2 <= iterations < MAX_ITERATIONS, (sharded, iterations)


# ---------------------------------------------------------------------------
# Adjoint: jax.grad through a scan of the compiled step
# ---------------------------------------------------------------------------


def _value_and_grad(gm: GraphManager):
    """``theta -> ((loss, final_state), d loss / d theta)`` over ``STEPS`` steps.

    ``theta`` is written into a copy of ``gm.params``: the sharded member's
    diffusivity and exchange coefficient and the replicated member's
    conductance.  The same pure step the graph compiles, scanned, as
    ``maddening.sysid`` does.
    """
    step_fn = gm._build_step_fn()  # noqa: SLF001
    ext = gm._resolve_external_inputs(None)  # noqa: SLF001
    state0 = gm._state  # noqa: SLF001

    def loss(theta):
        params = jax.tree.map(lambda x: x, gm.params)
        params["nodes"]["field"]["diffusivity"] = theta[0]
        params["nodes"]["far"]["conductance"] = theta[1]
        params["nodes"]["field"]["exchange"] = theta[2]

        def body(state, _):
            return step_fn(state, ext, params), None

        final, _ = jax.lax.scan(body, state0, None, length=STEPS)
        value = jnp.sum(final["field"]["f"] ** 2) + final["far"]["u"] ** 2
        return value, {n: final[n] for n in ("field", "far")}

    return jax.jit(jax.value_and_grad(loss, has_aux=True))


@pytest.fixture(scope="module", params=SOLVERS, ids=SOLVER_IDS)
def adjoint(request):
    solver = request.param
    theta = jnp.asarray(THETA, jnp.float32)
    out = {}
    for sharded in (True, False):
        (value, final), grad = _value_and_grad(_graph(sharded=sharded, solver=solver))(theta)
        out[sharded] = {"loss": float(value), "grad": np.asarray(grad),
                        "state": _host(final)}
    return solver, out


@pytest.mark.slow
def test_the_adjoint_matches_the_group_with_the_unsharded_inner_node(adjoint):
    solver, out = adjoint
    sharded, unsharded = out[True], out[False]
    assert np.all(np.abs(unsharded["grad"]) > 1e-2), unsharded["grad"]   # not degenerate
    _assert_close_each(sharded["grad"], unsharded["grad"], GRAD_RTOL, f"{solver}: gradient")
    _assert_close(sharded["loss"], unsharded["loss"], LOSS_RTOL, f"{solver}: loss")
    _assert_close(sharded["state"]["field"]["f"], unsharded["state"]["field"]["f"],
                  STATE_RTOL, f"{solver}: field under the gradient trace")


@pytest.mark.slow
def test_the_adjoint_is_the_derivative_of_the_implicitly_coupled_step(adjoint):
    """Both gradients against central differences of the float64 model."""
    solver, out = adjoint
    want = reference_gradient(THETA_F32)
    _, _, loss_ref = reference(THETA_F32)
    for sharded in (True, False):
        label = f"{solver}, {'sharded' if sharded else 'unsharded'}"
        _assert_close_each(out[sharded]["grad"], want,
                           REFERENCE_GRAD_RTOL[solver or "ift"], f"{label}: gradient")
        _assert_close(out[sharded]["loss"], loss_ref, REFERENCE_STATE_RTOL, f"{label}: loss")
