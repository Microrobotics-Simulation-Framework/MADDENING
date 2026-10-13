"""The interface norm's float floor of a value an edge delivers by cancellation.

Under ``convergence_norm="interface"`` the residual, its float floor and
the spectral analysis read what each internal edge *delivers*, and the
floor is ``PRECISION_FLOOR_ULPS`` units of ``eps * max|delivered field|``
an entry (CPL-100): the delivered field's own magnitude.  A delivered
value is no finer than the field it was computed from.  A mapping row
``[1, -1]`` on two entries near ``L`` delivers their difference rounded at
``eps * L``, which is ``L / |difference|`` of the floor's unit, and the
norm itself reads that difference: no schedule puts the two entries'
rounding anywhere the report counts it.  A pair stalled there reports
``residual=0.0``, ``precision_limited`` and a usable bound, and is
``L / |difference|`` of that bound's floor from its fixed point in the
norm's own units (MADD-ANO-247, open).

The pair is that of
``test_coupling_floor_gain_of_a_difference_within_one_field.py``
(MADD-ANO-212, the count of a same-pass read under Gauss-Seidel), with the
entries' magnitude a parameter.  Under Jacobi its report holds where the
norm reads the fields themselves -- the L2 norm in that module, the mixed
norm here -- and where the difference has nothing to cancel; it does not
hold where the norm reads the difference of two large entries.  The same
digits on jaxlib 0.10.2 and 0.11.2 (CPU), at ``L`` = 0, 1, 10, 100, 1000
and 1e5: the exact residual of the returned state is 0.044, 0.098, 0.71,
**5.7, 45 and 9445** times the floor beside a reported residual of 0.0
under either schedule (0.04 to 0.06 under the mixed norm at every ``L``),
and the bound over the true distance is 32, 11, 1.9, **0.25, 0.031 and
1.5e-4** under Jacobi and 46, 15, 2.7, 0.35, 0.044 and 2.1e-4 under
Gauss-Seidel.

The targeted search allows its scores this factor on the cells whose norm
reads a mapped edge (``_reading_cancellation`` in
``tests/property/test_coupling_targeted_search.py``): what it then no
longer asks is asked here, with no allowance.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import PRECISION_FLOOR_ULPS
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

F32 = jnp.float32
KEY = "A+B"
GAIN, FORCING = 0.99, 0.01
#: Tight enough that the pair runs to its float32 stall under every norm.
RTOL = 1e-9  # units: dimensionless (a relative tolerance)


class DeliveredDifferenceUnderItsFloor(AssertionError):
    """MADD-ANO-247: a usable bound under the true distance of a group
    whose interface norm reads a difference an edge delivers."""


class _Offset(SimulationNode):
    """``u = [L + t, L]``: two entries near ``L`` whose difference is the input."""

    def __init__(self, large: float):
        super().__init__("A", 1.0)
        self._large = float(large)

    def initial_state(self):
        return {"u": jnp.asarray([self._large, self._large], F32)}

    def update(self, state, boundary_inputs, dt):
        t = jnp.ravel(jnp.asarray(boundary_inputs["inp"]))[0]
        return {"u": jnp.stack([jnp.float32(self._large) + t, jnp.float32(self._large)])}

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


def _rms(*scaled) -> float:
    entries = np.concatenate([np.atleast_1d(np.asarray(s, np.float64)) for s in scaled])
    return float(np.sqrt(np.mean(entries ** 2)))


def _stalled(mode: str, norm: str, large: float) -> dict:
    """The pair run to its float32 stall: its report, the true distance to
    the fixed point and the exact residual of the returned state, both in
    the group's own norm at that state.

    ``A -> B`` delivers ``u[0] - u[1]`` through a mapping and ``B -> A``
    delivers ``B``'s entry; the loop ``t <- g t + c`` has the fixed point
    ``c / (1 - g)`` of the float32 constants.  ``"interface"`` reads the
    two delivered values, each over ``rtol`` times its own magnitude;
    ``"mixed"`` reads the two fields, each over ``rtol`` times its largest
    entry; both pool by the root mean square.
    """
    gm = GraphManager()
    gm.add_node(_Offset(large))
    gm.add_node(_Relay())
    gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(np.array([[1.0, -1.0]], np.float32)))
    gm.add_edge("B", "A", "u", "inp")
    gm.add_coupling_group(["A", "B"], max_iterations=6000, convergence_norm=norm, rtol=RTOL,
                          diagnostics=True, iteration_mode=mode)
    gm.compile()
    gm.step()
    d = dict(gm.coupling_diagnostics()[KEY])
    u_a = np.asarray(gm.get_node_state("A")["u"], np.float64)
    u_b = float(np.asarray(gm.get_node_state("B")["u"], np.float64)[0])
    lg, gain, forcing = (float(np.float32(v)) for v in (large, GAIN, FORCING))
    t_star = forcing / (1.0 - gain)
    delivered = float(u_a[0] - u_a[1])                  # exact: both are float32 numbers
    # One exact pass from the returned state: A's two entries and B's one.
    passed_a, passed_b = np.asarray([lg + u_b, lg]), gain * delivered + forcing
    if norm == "interface":
        distance = _rms((delivered - t_star) / abs(delivered), (u_b - t_star) / abs(u_b))
        residual = _rms((u_b - delivered) / max(abs(delivered), abs(u_b)),
                        (passed_b - u_b) / max(abs(u_b), abs(passed_b)))
    else:
        assert norm == "mixed", norm
        scale_a = float(np.max(np.abs(u_a)))
        distance = _rms((u_a - [lg + t_star, lg]) / scale_a, (u_b - t_star) / abs(u_b))
        residual = _rms((passed_a - u_a) / scale_a, (passed_b - u_b) / abs(u_b))
    floor = PRECISION_FLOOR_ULPS * float(np.finfo(np.float32).eps) / RTOL    # one evaluation's
    return dict(report=d, distance=distance / RTOL, residual=residual / RTOL, floor=floor,
                cancels=float(np.max(np.abs(u_a))) / abs(delivered))


def _held(seen: dict) -> None:
    d = seen["report"]
    assert d["precision_limited"] and d["spectral_usable"] and seen["distance"] > 0.0, seen
    assert d["spectral_error_bound"] >= seen["distance"], (
        d["spectral_error_bound"] / seen["distance"], seen)


def test_a_jacobi_pair_is_bounded_where_the_norm_reads_the_fields():
    """The control: the mixed norm reads ``A``'s field at its own
    magnitude, ``eps * L`` is its unit there, and the bound covers the
    stall (31 times the distance, usable)."""
    seen = _stalled("jacobi", "mixed", 100.0)
    assert seen["cancels"] > 50.0, seen
    _held(seen)


def test_a_jacobi_pair_is_bounded_where_the_delivered_difference_cancels_nothing():
    """The other control: the same edge and the same norm with ``L = 0``.
    The difference is of an entry and zero, it rounds at its own magnitude
    and the bound holds (32 times the distance, usable)."""
    seen = _stalled("jacobi", "interface", 0.0)
    assert seen["cancels"] == 1.0, seen
    _held(seen)
    assert seen["residual"] <= seen["report"]["residual"] + seen["floor"], seen


@pytest.mark.xfail(strict=True, raises=DeliveredDifferenceUnderItsFloor, reason=(
    "MADD-ANO-247: under the interface norm the float floor of a value an edge delivers is "
    "taken at the delivered value's own magnitude, and a difference of two entries of one "
    "field rounds at the entries'; a group stalled there reports a usable bound below the "
    "true distance under either schedule; open, deferred to 0.5.0"))
@pytest.mark.parametrize("large", (100.0, 1000.0))
def test_a_jacobi_pair_whose_norm_reads_a_delivered_difference_is_bounded(large):
    """Under Jacobi the two entries are coordinates of the iterate, and
    that is what bounds this pair under the L2 and the mixed norm.  The
    interface norm reads their difference, at the difference's own
    magnitude: the pair stalls at ``residual=0.0`` with the exact residual
    of the state it returns 5.7 (``L = 100``) and 45 (``L = 1000``) times
    the floor the report adds -- 0.056 of the floor per unit of ``L /
    |difference|``, from 10 to 1e5 -- and the bound reads 0.25 and 0.031
    of the true distance, ``spectral_usable=True``."""
    seen = _stalled("jacobi", "interface", large)
    d = seen["report"]
    assert seen["cancels"] > 50.0, seen
    assert d["precision_limited"] and d["spectral_usable"] and d["residual"] == 0.0, seen
    if (seen["residual"] > d["residual"] + seen["floor"]
            or d["spectral_error_bound"] < seen["distance"]):
        raise DeliveredDifferenceUnderItsFloor(
            f"the bound is {d['spectral_error_bound'] / seen['distance']:.3g} of the true "
            f"distance, usable; the exact residual of the returned state is "
            f"{seen['residual'] / seen['floor']:.3g} of the floor; the delivered difference "
            f"is 1/{seen['cancels']:.4g} of the entries it is taken of: {seen}")
