"""The floor's gain of a same-pass read that differences entries of one source field.

With ``diagnostics=True`` the float floor of a coupling pass weights every
group-internal read by its relative gain, measured by one JVP of the
reading member's update *along the source's own state*
(``PRECISION_FLOOR_ULPS``, CPL-102).  That direction moves every entry of
the source field by the same relative amount, so a read that takes a
difference of two nearly equal entries of one field -- a mapping row
``[1, -1]`` here -- returns a small response and is counted as a gain of
about one.  The entries round independently: the difference of two
float32 numbers near ``L`` carries a rounding of ``eps * L``, which is
``L / |difference|`` times the floor's unit in what the reader receives.

Under Jacobi the report does not rest on that count -- the two entries
are separate coordinates of the iterate, and the resolvent the bound
applies carries their cross gain.  Under Gauss-Seidel the read is of the
same pass, the pass's Jacobian has it folded in, and the count is all
that carries the rounding: the bound reads below the true distance with
``spectral_usable=True`` (MADD-ANO-212, open; the measured, opt-in
``diagnostics="rounding"`` level planned for 0.5.0 is what closes it).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

F32 = jnp.float32
KEY = "A+B"
#: The magnitude of ``A``'s two entries; their difference is of order one.
LARGE = 1000.0
GAIN, FORCING = 0.99, 0.01


class _Offset(SimulationNode):
    """``u = [L + t, L]``: two entries near ``L`` whose difference is the input."""

    def __init__(self):
        super().__init__("A", 1.0)

    def initial_state(self):
        return {"u": jnp.asarray([LARGE, LARGE], F32)}

    def update(self, state, boundary_inputs, dt):
        t = jnp.ravel(jnp.asarray(boundary_inputs["inp"]))[0]
        return {"u": jnp.stack([jnp.float32(LARGE) + t, jnp.float32(LARGE)])}

    def update_evaluations(self):
        return 1.0


class _Relay(SimulationNode):
    """``u = g * d + c`` on the one entry it is handed."""

    def __init__(self):
        super().__init__("B", 1.0)

    def initial_state(self):
        return {"u": jnp.asarray([0.0], F32)}

    def update(self, state, boundary_inputs, dt):
        d = jnp.ravel(jnp.asarray(boundary_inputs["inp"]))[0]
        return {"u": jnp.stack([jnp.float32(GAIN) * d + jnp.float32(FORCING)])}

    def update_evaluations(self):
        return 1.0


def _stalled(mode):
    """``(report, true distance)`` of the pair run to its float32 stall.

    ``A -> B`` delivers ``u[0] - u[1]`` through a mapping; the loop
    ``t <- g t + c`` has the fixed point ``c / (1 - g)``.  The distance is
    the L2 norm's, each field over its own magnitude, to the fixed point
    of the map the float32 constants define.
    """
    gm = GraphManager()
    gm.add_node(_Offset())
    gm.add_node(_Relay())
    gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(np.array([[1.0, -1.0]], np.float32)))
    gm.add_edge("B", "A", "u", "inp")
    gm.add_coupling_group(["A", "B"], max_iterations=6000, convergence_norm="l2",
                          tolerance=1e-9, diagnostics=True, iteration_mode=mode)
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    u_a = np.asarray(gm.get_node_state("A")["u"], np.float64)
    u_b = np.asarray(gm.get_node_state("B")["u"], np.float64)
    large, gain, forcing = (float(np.float32(v)) for v in (LARGE, GAIN, FORCING))
    t_star = forcing / (1.0 - gain)
    terms = np.concatenate([np.abs(u_a - [large + t_star, large]) / np.max(np.abs(u_a)),
                            np.abs(u_b - t_star) / np.max(np.abs(u_b))])
    return d, float(np.sqrt(np.sum(terms ** 2)))


def test_a_jacobi_group_reading_a_difference_within_one_field_is_bounded():
    """The control: under Jacobi the entries are coordinates of the iterate,
    the resolvent carries their cross gain, and the bound covers the stall.

    The flag is withdrawn on it all the same, for the radius: the weighted
    Jacobian has norm 1 400 beside a radius of 0.99, float32 reads 0.9951
    (ten of the flag's margins off, on the bound's safe side), and the
    spectral estimate's rounding probe measures that it is not settled.
    """
    d, true = _stalled("jacobi")
    assert d["precision_limited"] and true > 1e-4, (dict(d), true)
    assert d["spectral_error_bound"] >= true, (d["spectral_error_bound"] / true, dict(d))
    assert not d["spectral_usable"] and abs(d["rho_spectral"] - 0.99) > 5e-4, dict(d)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-212: the floor's gain of a same-pass read is measured along the source's "
    "own state, which a difference of two entries of that field cancels, so a "
    "Gauss-Seidel group stalled behind such a read reports a usable bound below the "
    "true distance; open, deferred to 0.5.0"))
def test_a_gauss_seidel_group_reading_a_difference_within_one_field_is_bounded():
    """Under Gauss-Seidel the read is of the same pass and only the counted
    floor carries its rounding: the group stalls 0.3% from its fixed point at
    ``residual=0.0`` and the bound reads 0.054x that distance, usable."""
    d, true = _stalled("gauss-seidel")
    assert d["precision_limited"] and d["spectral_usable"] and true > 1e-4, (dict(d), true)
    assert d["spectral_error_bound"] >= true, (d["spectral_error_bound"] / true, dict(d))
