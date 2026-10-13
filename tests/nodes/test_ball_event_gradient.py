"""Gradients through BallNode's bounce omit the event-time derivative.

MADD-ANO-021.  ``BallNode.update`` tests for contact once per step and,
on contact, pins the position to the table and reflects the velocity.
``jax.grad`` differentiates the branch the step took and never sees the
moment of contact, so the derivative of the contact time is missing
from every gradient taken across a bounce.  With the clamp the gradient
of the post-bounce height with respect to the drop height is *exactly*
zero: the position is set to a constant, and the impact velocity
depends on how many whole steps the fall took rather than on the drop
height, so everything after the bounce is flat in the drop height
within each one-step window and jumps between windows.

The reference is the exact continuous solution of the same problem: a
drop from rest at height ``h`` onto a table at 0, contact at
``t_c = sqrt(2 h / g)``, rebound speed ``e g t_c``, free flight for the
remaining ``T - t_c``.  ``T = 0.8 s`` holds exactly one bounce for every
drop height used here.

The strict xfail is the entry's acceptance criterion and flips when
event localisation lands (planned for 0.5.0).  The plain tests pin
today's behaviour exactly, so a partial change -- a gradient that stops
being zero without becoming right -- is visible as a failure too, and
they pin the boundary of the defect: the values are right, and so is
the gradient with respect to a parameter the contact time does not
depend on.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.ball import BallNode

G = 9.81
E = 0.8
DT = 1e-3
N_STEPS = 800                      # T = 0.8 s: one bounce, before the second
T_FINAL = N_STEPS * DT
TABLE = {"table_position": jnp.float32(0.0)}
DROP_HEIGHTS = (1.0, 1.005, 1.0101, 1.015)

#: Tolerance of the acceptance criterion.  The tick model's *values* are
#: within one step of travel of the exact solution (about 3e-3 on a
#: rebound height of order 0.1), so an event-localised gradient is
#: expected to agree with the continuous derivative to O(dt), i.e. far
#: inside 2 %.  Today's error is 100 %.
REL_TOL = 0.02


def _final_height(h, *, dt=DT, n_steps=N_STEPS, elasticity=E, gravity=G):
    """Height after ``n_steps`` of the node's own ``update`` from rest at ``h``."""
    node = BallNode("ball", dt, elasticity=E, gravity=-G, initial_position=1.0)
    params = {"elasticity": elasticity, "gravity": -gravity}
    s0 = {"position": jnp.asarray(h, jnp.float32), "velocity": jnp.float32(0.0)}
    final, _ = jax.lax.scan(
        lambda s, _: (node.update(s, TABLE, dt, params=params), None),
        s0, None, length=n_steps,
    )
    return final["position"]


def _exact_final_height(h, elasticity=E, gravity=G):
    """The continuous solution: one bounce at ``t_c = sqrt(2 h / g)``."""
    t_c = jnp.sqrt(2.0 * h / gravity)
    rebound = elasticity * gravity * t_c
    tau = T_FINAL - t_c
    return rebound * tau - 0.5 * gravity * tau ** 2


def _grad_tick(h, **kw):
    return float(jax.jit(jax.grad(lambda x: _final_height(x, **kw)))(jnp.float32(h)))


def _grad_exact(h):
    return float(jax.grad(_exact_final_height)(jnp.float32(h)))


def test_one_bounce_happens_inside_the_window_for_every_drop_height():
    """The fixture can express the defect: every drop lands once, before T,
    and the rebound peaks before the second contact."""
    for h in DROP_HEIGHTS:
        t_c = np.sqrt(2 * h / G)
        t_second = t_c + 2 * E * G * t_c / G
        assert t_c < T_FINAL < t_second, h


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "MADD-ANO-021: BallNode's per-step contact returns the gradient of the "
        "branch the step took and omits the derivative of the contact time, so "
        "d(height after the bounce)/d(drop height) is exactly 0.0 against an "
        "exact +0.5892.  This is a framework limitation (any node that branches "
        "on its own state inside update() has it); it flips when event "
        "localisation with an implicit-function-theorem event-time derivative "
        "lands (planned for 0.5.0).  When it XPASSes, resolve MADD-ANO-021 and "
        "drop the exact-zero pin below."
    ),
)
@pytest.mark.parametrize("h", DROP_HEIGHTS)
def test_the_bounce_gradient_matches_the_exact_event_time_derivative(h):
    got, want = _grad_tick(h), _grad_exact(h)
    assert abs(got - want) <= REL_TOL * abs(want), (
        f"h={h}: grad through the bounce {got:+.5f}, exact {want:+.5f}"
    )


@pytest.mark.parametrize("dt,n_steps", [(1e-3, 800), (2.5e-4, 3200), (1e-4, 8000)])
@pytest.mark.parametrize("h", DROP_HEIGHTS)
def test_the_bounce_gradient_with_respect_to_the_drop_height_is_exactly_zero(h, dt, n_steps):
    """Today's value, pinned exactly -- and not a precision artefact: a
    finer step narrows the flat windows without making the gradient
    non-zero.  A change that moves it off zero without meeting the
    xfail above is a partial fix and should be seen."""
    assert _grad_tick(h, dt=dt, n_steps=n_steps) == 0.0


def test_the_exact_derivative_the_criterion_compares_against():
    """The reference figure quoted in the registry and the docstring."""
    assert _grad_exact(1.0) == pytest.approx(0.5892, abs=5e-5)


def test_the_bounce_value_is_within_one_tick_of_travel_of_the_exact_solution():
    """The defect is in the gradient only: the trajectory looks right.
    One tick of travel at the rebound speed is ``e * g * t_c * dt``,
    about 3.5e-3 here; the measured worst case is 3.0e-3."""
    hs = np.linspace(1.0, 1.02, 201)
    tick = np.asarray(jax.jit(jax.vmap(_final_height))(jnp.asarray(hs, jnp.float32)))
    exact = np.asarray(jax.vmap(_exact_final_height)(jnp.asarray(hs, jnp.float32)))
    one_tick = E * G * np.sqrt(2 * hs.max() / G) * DT
    assert np.max(np.abs(tick - exact)) <= one_tick


def test_the_gravity_gradient_misses_the_event_time_term():
    """A parameter that moves the contact time gets a gradient that is
    non-zero and wrong: about 13 times the exact value, because the
    omitted event-time term nearly cancels the rest."""
    got = float(jax.grad(lambda g: _final_height(jnp.float32(1.0), gravity=g))(jnp.float32(G)))
    want = float(jax.grad(lambda g: _exact_final_height(jnp.float32(1.0), gravity=g))(
        jnp.float32(G)))
    assert want == pytest.approx(0.00510, abs=5e-5)
    assert got == pytest.approx(0.0651, abs=5e-4)
    assert got / want > 10.0


def test_the_elasticity_gradient_is_right_because_the_contact_time_does_not_depend_on_it():
    """The boundary of the defect: the omitted term is the event time's
    derivative, and the contact time of the first bounce is independent
    of the restitution coefficient."""
    got = float(jax.grad(lambda e: _final_height(jnp.float32(1.0), elasticity=e))(
        jnp.float32(E)))
    want = float(jax.grad(lambda e: _exact_final_height(jnp.float32(1.0), elasticity=e))(
        jnp.float32(E)))
    assert want == pytest.approx(1.5436, abs=5e-4)
    assert got == pytest.approx(want, rel=1e-3)


def test_nothing_in_maddening_warns_about_the_missing_term():
    """The registry says the wrong gradient arrives silently.  If a
    warning is ever added, this fails so the entry is updated with it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _grad_tick(1.0)
    ours = [w for w in caught if "maddening" in (w.filename or "")]
    assert ours == [], [str(w.message) for w in ours]
