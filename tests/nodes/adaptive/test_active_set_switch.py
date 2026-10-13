"""Behaviour at and across an active-set switch -- the regime that makes
this node different from an ordinary one, and the one the first version
of the suite never visited (audit A2).

The frozen-set objective is piecewise smooth **with jumps**: where two
candidates swap rank, their coefficients and their sensor weights differ,
so ``J_frozen`` steps.  It is therefore not locally Lipschitz there, no
Clarke subgradient exists, and the returned gradient -- which is exact
*within* a region and ignores the set's dependence on the parameter --
misses a *first-order* contribution equal to the sum of the jumps a step
crosses.  These tests assert that, rather than hiding it.

The in-region gradient check does **not** use a finite difference of the
mask-recomputing objective: the oracle is the smooth branch the forward
pass actually selected, and the step is verified to stay inside the
region.  Registered as anomaly MADD-ANO-003.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.nodes.adaptive._toys import PoissonSineTopKNode

INTERVAL = (0.40, 0.50)


class _Toy:
    """The 1-D sine toy with jitted helpers (mask, J, grad J)."""

    def __init__(self, k=16, n=256, theta=0.42):
        self.node = PoissonSineTopKNode(theta=theta, n=n, k=k, blindness_gate=False)
        self.state = self.node.initial_state()
        self._J = jax.jit(self._objective)
        self._grad = jax.jit(jax.grad(self._objective))
        self._mask = jax.jit(self._select)

    def _params(self, theta):
        return {**self.node.params, "theta": theta}

    def _select(self, theta):
        return self.node.compute_active_set(self.state, self._params(theta))

    def _objective(self, theta):
        out = self.node.update(self.state, {}, 1.0, params={"theta": theta})
        return self.node.objective(out, self._params(theta))

    def J(self, theta):
        return float(self._J(jnp.asarray(float(theta))))

    def grad(self, theta):
        return float(self._grad(jnp.asarray(float(theta))))

    def mask(self, theta):
        return np.asarray(self._mask(jnp.asarray(float(theta))))

    def branch_objective(self, mask):
        """``J`` with the active set held at ``mask`` -- the smooth branch."""
        frozen = jnp.asarray(mask, dtype=bool)

        def J(theta):
            params = self._params(theta)
            out = self.node._solve_and_pack(self.state, frozen, params)
            return self.node.objective(out, params)

        return J

    def switches(self, lo, hi, samples=2001, refine=50):
        """Every theta in ``[lo, hi]`` where the active set changes."""
        grid = np.linspace(lo, hi, samples)
        found, previous = [], self.mask(lo)
        step = (hi - lo) / (samples - 1)
        for theta in grid[1:]:
            current = self.mask(theta)
            if np.array_equal(current, previous):
                continue
            left, right = theta - step, theta
            for _ in range(refine):
                mid = 0.5 * (left + right)
                if np.array_equal(self.mask(mid), previous):
                    left = mid
                else:
                    right = mid
            found.append(0.5 * (left + right))
            previous = current
        return found


@pytest.fixture(scope="module")
def toy():
    # Module-scoped fixtures are built before the per-test x64 fixture in
    # ``conftest.py`` runs, and these measurements need float64.
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield _Toy()
    finally:
        jax.config.update("jax_enable_x64", prior)


# -- the objective is discontinuous, not kinked -----------------------------------

def test_frozen_objective_jump_at_an_active_set_switch_does_not_vanish_with_the_step(toy):
    """A kink would make ``J(ts+d) - J(ts-d)`` go to zero with ``d``; a
    jump converges to a non-zero constant.  It is a jump."""
    theta_s = toy.switches(0.4209, 0.4211, samples=3)[0]
    left, right = toy.mask(theta_s - 1e-9), toy.mask(theta_s + 1e-9)
    assert int((left & ~right).sum()) == 1 and int((right & ~left).sum()) == 1

    jumps = [toy.J(theta_s + d) - toy.J(theta_s - d) for d in (1e-6, 1e-8, 1e-10)]
    scale = abs(toy.J(theta_s))
    assert min(abs(j) for j in jumps) > 1e-3 * scale, jumps
    for j in jumps[1:]:
        assert j == pytest.approx(jumps[0], rel=1e-2), jumps
    # The two one-sided slopes, by contrast, nearly agree: the "kink" part
    # is negligible and the jump is the whole error.
    slopes = (toy.grad(theta_s - 1e-9), toy.grad(theta_s + 1e-9))
    assert slopes[1] == pytest.approx(slopes[0], rel=0.1)


def test_central_finite_difference_across_a_switch_diverges_as_one_over_h(toy):
    """Finite differences are not a valid oracle at a switch -- they blow
    up like ``jump / (2h)`` -- while the returned gradient stays finite.
    The suite documents this rather than avoiding the point."""
    theta_s = toy.switches(0.4209, 0.4211, samples=3)[0]
    gradient = toy.grad(theta_s)
    assert np.isfinite(gradient) and abs(gradient) < 1.0
    jump = toy.J(theta_s + 1e-9) - toy.J(theta_s - 1e-9)

    differences = {}
    for h in (1e-5, 1e-6, 1e-7):
        differences[h] = (toy.J(theta_s + h) - toy.J(theta_s - h)) / (2 * h)
        assert not np.array_equal(toy.mask(theta_s - h), toy.mask(theta_s + h))
        # fd = jump / (2h) + the smooth slope: the divergence is exactly
        # 1/h, not noise, and it is the jump that drives it.
        assert h * (differences[h] - gradient) == pytest.approx(jump / 2, rel=2e-2)
    assert abs(differences[1e-7]) > 1000 * abs(gradient)
    assert abs(differences[1e-7]) > 10 * abs(differences[1e-6]) > 10 * abs(differences[1e-5])


# -- inside a region the returned gradient is exact --------------------------------

def test_gradient_inside_an_active_set_region_matches_the_selected_branch(toy):
    """Strictly inside a region the returned gradient is the derivative of
    the branch the forward pass selected, and a finite difference *of that
    branch* (never of the mask-recomputing objective -- spike
    recommendation 3) confirms it, with a step verified to stay inside."""
    switches = toy.switches(0.415, 0.435, samples=801)
    assert len(switches) >= 2, switches
    widest = max(zip(switches, switches[1:]), key=lambda pair: pair[1] - pair[0])
    base = 0.5 * (widest[0] + widest[1])
    gap = min(abs(base - s) for s in switches)
    h = min(gap / 10.0, 1e-5)
    assert h < gap, (h, gap)

    mask = toy.mask(base)
    assert np.array_equal(toy.mask(base - h), mask)
    assert np.array_equal(toy.mask(base + h), mask)   # no switch within the step

    branch = toy.branch_objective(mask)
    gradient = toy.grad(base)
    assert gradient == pytest.approx(
        float(jax.grad(branch)(jnp.asarray(base))), rel=1e-12,
    )
    theta = jnp.asarray(base)
    fd = float((branch(theta + h) - branch(theta - h)) / (2 * h))
    assert abs(gradient - fd) / abs(fd) < 1e-7, (gradient, fd, h)


# -- the size of the omitted first-order term ---------------------------------------

def test_integrated_gradient_plus_switch_jumps_reconstructs_the_objective_change(toy):
    """The headline measurement: over theta in [0.40, 0.50] at n=256, k=16
    the integral of the returned gradient is ~-1.58e-3 against a true
    change of ~-2.36e-3 -- a 33 % shortfall, accounted for by the sum of
    the 27 jumps crossed."""
    lo, hi = INTERVAL
    grid = np.linspace(lo, hi, 801)
    integral = float(np.trapezoid([toy.grad(t) for t in grid], grid))
    true_change = toy.J(hi) - toy.J(lo)

    switches = toy.switches(lo, hi)
    jumps = [toy.J(s + 1e-10) - toy.J(s - 1e-10) for s in switches]

    assert len(switches) == 27, len(switches)
    assert integral == pytest.approx(-1.5808e-3, rel=1e-3)
    assert true_change == pytest.approx(-2.3598e-3, rel=1e-3)
    shortfall = abs(true_change - integral) / abs(true_change)
    assert shortfall == pytest.approx(0.33, abs=0.01), shortfall
    # The shortfall *is* the crossed jumps: adding them back recovers the
    # objective change to two orders of magnitude better than the raw
    # integral does.
    reconstructed = integral + sum(jumps)
    assert reconstructed == pytest.approx(true_change, rel=2e-2)
    assert abs(reconstructed - true_change) < 0.02 * abs(integral - true_change)


def test_the_omitted_term_decays_with_the_active_set_budget():
    """The regime where the frozen gradient is trustworthy: the jumps
    shrink with the budget, because the coefficient that swaps rank
    shrinks.  ~1e-8 of |J| by k=64 (the spike measured 4e-8)."""
    relative = {}
    for k in (8, 16, 32, 64):
        toy = _Toy(k=k)
        switches = toy.switches(0.40, 0.42, samples=401, refine=45)
        total = sum(abs(toy.J(s + 1e-10) - toy.J(s - 1e-10)) for s in switches)
        relative[k] = total / abs(toy.J(0.41))
    assert relative[8] > relative[16] > relative[32] > relative[64], relative
    assert relative[64] < 1e-6, relative
