"""The conditioning guard on the masked-CG path says what it bounds, and a CG
solve that does not converge says why.

``CONDITION_LIMIT`` models a direct solve (``kappa * eps``).  Its docstring
claimed it bounded "the masked-CG path" too, but ``frozen_solver="cg"``
stops at a relative residual of ``rtol`` (1e-6 in float32), and at
conditionings the guard accepted lineax's CG stagnated or broke down --
float32 at 256 points already at ``mass = 0.3`` -- and surfaced as an
opaque equinox error ("try increasing ``max_steps``", which the node does
not expose; more steps do not help).  Loud, never a wrong number, but
neither the claim nor the message was right.

Now: on the CG path the guard's solve term is ``kappa * max(eps, rtol)``,
which bounds the accuracy of a CG solve that converges; and an eager CG
solve that does not converge is re-raised as a ``ValueError`` naming the
conditioning, the tolerance and the fixes.  The suite's ``conftest.py``
enables float64 per test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maddening.nodes.adaptive.wavelet as wavelet_module
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive.wavelet import CONDITION_LIMIT


def _node(dtype, n_levels, mass, solver, **kw):
    return WaveletAdaptiveNode("w", 1.0, n_levels=n_levels, mass=mass,
                               k=2 ** (n_levels + 1), dtype=dtype,
                               blindness_gate=False, frozen_solver=solver, **kw)


# -- what the guard bounds on the CG path ------------------------------------

@pytest.mark.parametrize("dtype,mass,rtol", [
    (jnp.float32, 1e-2, 1e-6), (jnp.float64, 1e-6, 1e-10),
], ids=["float32", "float64"])
def test_the_cg_bound_uses_the_cg_tolerance_and_refuses_what_it_cannot_carry(
        dtype, mass, rtol):
    """kappa = 1.9e3 (float32 case) and 1.9e7 (float64 case): the direct
    solve is inside the limit, a CG stopping at ``rtol`` is not.  The
    auditor's float32 mass 1e-2 used to be accepted and then fail inside
    lineax; now it is refused at construction, and the message points at
    the gathered solve, which carries it."""
    gather = _node(dtype, 6, mass, "gather")
    assert gather.solve_error_bound() <= CONDITION_LIMIT
    with pytest.raises(ValueError) as info:
        _node(dtype, 6, mass, "cg")
    message = str(info.value)
    assert f"a masked-CG solve stopping at a relative residual of {rtol:.0e}" in message
    assert "frozen_solver='gather', a direct solve bounded by kappa * eps" in message
    assert ", accepted)" in message
    kappa = gather.condition_number
    eps64 = float(np.finfo(np.float64).eps)
    expected = kappa * rtol + gather.physical_condition_number * eps64
    assert f"can be wrong by a relative {expected:.1e}" in message


def test_the_bound_is_unchanged_for_the_gathered_solve_and_for_a_well_conditioned_cg():
    gather = _node(jnp.float32, 6, 1.0, "gather")
    cg = _node(jnp.float32, 6, 1.0, "cg")
    eps32 = float(jnp.finfo(jnp.float32).eps)
    eps64 = float(np.finfo(np.float64).eps)
    assert gather.solve_error_bound() == pytest.approx(
        gather.condition_number * eps32 + gather.physical_condition_number * eps64)
    assert cg.solve_error_bound() == pytest.approx(
        cg.condition_number * 1e-6 + cg.physical_condition_number * eps64)
    # and in float64 the CG term is rtol = 1e-10, not eps
    assert cg.solve_error_bound(np.float64) == pytest.approx(
        cg.condition_number * 1e-10 + cg.physical_condition_number * eps64)


@pytest.mark.parametrize("dtype,n_levels,mass", [
    (jnp.float32, 6, 1.0), (jnp.float32, 6, 0.1), (jnp.float64, 7, 1e-4),
])
def test_an_accepted_cg_solve_that_converges_is_within_the_bound_of_the_direct_one(
        dtype, n_levels, mass):
    """The bound is honest where CG converges: the CG reading is within
    ``solve_error_bound()`` of the float64 direct reading (measured errors
    are 2e-7 to 5e-6 in float32 and 3e-8 in float64)."""
    cg = _node(dtype, n_levels, mass, "cg")
    ref = _node(jnp.float64, n_levels, mass, "gather")
    J = float(cg.objective(cg.initial_state(), {}))
    J_ref = float(ref.objective(ref.initial_state(), {}))
    assert abs(J - J_ref) / abs(J_ref) <= cg.solve_error_bound()


# -- a CG solve that does not converge ---------------------------------------

#: Accepted by the bound, and measured to stagnate or break down below the
#: CG tolerance (jaxlib 0.11.0; lineax 0.0.7 and 0.1.1): at 1, 4 and 16
#: times the step budget for the first two.
_NON_CONVERGING = [
    (jnp.float32, 7, 0.3),      # kappa 65
    (jnp.float64, 7, 1e-5),     # kappa 1.9e6
    (jnp.float32, 7, 0.1),      # kappa 194, breaks down into NaN
]


def _outcome(dtype, n_levels, mass):
    node = _node(dtype, n_levels, mass, "cg")
    assert node.solve_error_bound() <= CONDITION_LIMIT       # accepted
    try:
        state = node.initial_state()
    except ValueError as err:
        return "refused", str(err), node
    return "converged", float(node.objective(state, {})), node


def test_an_eager_cg_solve_that_does_not_converge_is_refused_with_the_conditioning_and_the_fix():
    """Every configuration either converges to within the bound or is
    refused by the node's own message -- never lineax's "increase
    ``max_steps``", never a wrong number -- and at least one of them
    really exercises the refusal on whatever jaxlib runs this."""
    refused = 0
    for dtype, n_levels, mass in _NON_CONVERGING:
        kind, value, node = _outcome(dtype, n_levels, mass)
        if kind == "converged":
            ref = _node(jnp.float64, n_levels, mass, "gather")
            J_ref = float(ref.objective(ref.initial_state(), {}))
            assert abs(value - J_ref) / abs(J_ref) <= node.solve_error_bound()
            continue
        refused += 1
        message = value
        assert "the masked-CG frozen solve (frozen_solver='cg')" in message
        assert f"condition number is about {node.condition_number:.2e}" in message
        assert f"max(4 n_max, 200) = {max(4 * node.n_max, 200)} steps" in message
        assert "more steps do not help" in message
        assert "bounds the accuracy of a CG solve that converges, not whether it does" \
            in message
        assert "use frozen_solver='gather'" in message
        assert "raise mass" in message
        assert ("build the node in float64" in message) == (dtype == jnp.float32)
        assert "max_steps`" not in message          # lineax's advice is not repeated
    assert refused >= 1


def test_the_gathered_solve_carries_every_configuration_the_cg_path_refused():
    """The first fix the message names really works."""
    for dtype, n_levels, mass in _NON_CONVERGING:
        node = _node(dtype, n_levels, mass, "gather")
        J = float(node.objective(node.initial_state(), {}))
        assert np.isfinite(J)


def test_only_a_lineax_convergence_failure_is_translated(monkeypatch):
    """The catch recognises lineax's own wording and nothing else: an
    unrelated error propagates unchanged, and the translated one chains
    the original as ``__cause__``."""
    node = _node(jnp.float32, 5, 1.0, "cg")
    lineax_says = ("_EquinoxRuntimeError: The maximum number of solver steps was "
                   "reached. Try increasing `max_steps`.")

    def fails_like_lineax(*args, **kwargs):
        raise RuntimeError(lineax_says)

    monkeypatch.setattr(wavelet_module, "_cg_kernel", fails_like_lineax)
    with pytest.raises(ValueError, match="ran out of steps") as info:
        node.initial_state()
    assert isinstance(info.value.__cause__, RuntimeError)

    def fails_non_finite(*args, **kwargs):
        raise RuntimeError("The linear solver returned non-finite (NaN or inf) output.")

    monkeypatch.setattr(wavelet_module, "_cg_kernel", fails_non_finite)
    with pytest.raises(ValueError, match="broke down into non-finite values"):
        node.initial_state()

    def fails_otherwise(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    monkeypatch.setattr(wavelet_module, "_cg_kernel", fails_otherwise)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        node.initial_state()


def test_an_error_that_surfaces_only_when_the_result_is_awaited_is_still_translated(
        monkeypatch):
    """JAX dispatches asynchronously: lineax's runtime error can surface when
    the result is awaited rather than at the call (on this CPU build it
    surfaces at the call, so nothing else here can tell the difference).
    The eager path blocks on the result inside the catch, so a late error
    is translated too, not raised later from wherever ``c`` is next read."""
    node = _node(jnp.float32, 5, 1.0, "cg")
    real_block = jax.block_until_ready

    def late_failure(x):
        real_block(x)
        raise RuntimeError("_EquinoxRuntimeError: The maximum number of solver steps "
                           "was reached. Try increasing `max_steps`.")

    monkeypatch.setattr(jax, "block_until_ready", late_failure)
    with pytest.raises(ValueError, match="ran out of steps"):
        node.initial_state()


def test_under_a_trace_the_cg_solve_is_not_blocked_on():
    """Under ``jit`` the solve has not run when ``solve_frozen`` returns, so
    there is nothing to catch and nothing may be forced: the traced update
    still compiles and agrees with the eager one."""
    node = _node(jnp.float32, 5, 1.0, "cg")
    state = node.initial_state()
    eager = node.update(state, {}, 1.0)
    traced = jax.jit(lambda s: node.update(s, {}, 1.0))(state)
    np.testing.assert_allclose(np.asarray(traced["c"]), np.asarray(eager["c"]),
                               rtol=1e-5, atol=1e-7)
