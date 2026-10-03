"""A bfloat16 or float16 group's change is measured, summed and held in float32.

Every coupling norm forms ``|dx| / (rtol * max|v|)`` per entry, squares it
and sums the squares; the precision floor sums ``(eps / rtol)**2`` per
entry.  For a 16-bit group all of that ran in the group's own dtype:

* the running sum was a weakly typed ``jnp.array(0.0)``, which took the
  field's float16, and the int32 entry count it was divided by became
  float16 ``inf`` above 65 504 entries -- the RMS of a finite sum read
  exactly ``0.0``, and a group 4% from its fixed point stopped with
  ``converged=True`` under the mixed and interface norms;
* at the default ``rtol=1e-6`` a change of one float16 ulp is a ratio of
  about 977, whose square is past float16's 65 504 -- the residual read
  ``inf`` on a finite state, and ``strict_convergence`` blamed divergence
  ("no larger max_iterations would help") where the cap was the cause;
* the floor overflowed alike: ``inf`` for every ``rtol`` below about 3.8e-6,
  and above 65 504 entries under every norm.

Now each field is widened to float32 before anything is formed from it (the
16-bit value as the state holds it -- see ``acceleration._widened``), the
residual and its report slots are float32, and ``strict_convergence``
decides "non-finite state" from the state.  The oracles are the closed forms
in float64; the reproducers are the round-6 coupling audit's.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    coupling_residual_interface,
    coupling_residual_l2,
    coupling_residual_mixed,
    residual_precision_floor,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.profiler import _meta_converged

SIXTEEN = (jnp.float16, jnp.bfloat16)
KEY = "A+B"
#: More active entries than float16 can count (its largest finite value is
#: 65 504): two fields of this many.
BIG = 40_000


def _pair_states(n, dtype, old_value, new_value):
    old = {nm: {"u": jnp.full((n,), old_value, dtype)} for nm in ("A", "B")}
    new = {nm: {"u": jnp.full((n,), new_value, dtype)} for nm in ("A", "B")}
    return new, old


_EDGES = [EdgeSpec("A", "B", "u", "inp"), EdgeSpec("B", "A", "u", "inp")]


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_a_sixteen_bit_norm_over_more_than_65504_entries_reads_its_rms(dtype):
    """``2 * BIG`` entries each moved by one ulp: every norm reads the closed form."""
    eps = float(jnp.finfo(dtype).eps)
    new, old = _pair_states(BIG, dtype, 1.0, 1.0 + eps)
    rel = eps / (1.0 + eps)
    rtol = 1e-2
    assert float(coupling_residual_mixed(new, old, ["A", "B"], 0.0, rtol)) == pytest.approx(
        rel / rtol, rel=1e-5)
    assert float(coupling_residual_interface(new, old, _EDGES, 0.0, rtol)) == pytest.approx(
        rel / rtol, rel=1e-5)
    assert float(coupling_residual_l2(new, old, ["A", "B"])) == pytest.approx(
        rel * math.sqrt(2 * BIG), rel=1e-5)


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_a_sixteen_bit_norm_at_the_default_rtol_is_finite_on_a_finite_change(dtype):
    """A one-pass change of order one at ``rtol=1e-6``: about 1e6, not ``inf``."""
    old = {"A": {"u": jnp.zeros((1,), dtype)}, "B": {"u": jnp.zeros((1,), dtype)}}
    new = {"A": {"u": jnp.asarray([0.1], dtype)}, "B": {"u": jnp.asarray([0.19], dtype)}}
    # Each entry moved by its whole magnitude: |dx| / (rtol |x|) = 1 / rtol.
    for got in (coupling_residual_mixed(new, old, ["A", "B"], 0.0, 1e-6),
                coupling_residual_interface(new, old, _EDGES, 0.0, 1e-6)):
        assert jnp.asarray(got).dtype == jnp.float32
        assert float(got) == pytest.approx(1e6, rel=1e-6)


@pytest.mark.parametrize("dtype", SIXTEEN)
@pytest.mark.parametrize("rtol", [1e-6, 3e-6, 4e-6, 1e-3])
def test_the_sixteen_bit_floor_is_four_eps_over_rtol(dtype, rtol):
    """The floor is the documented ``4 eps / rtol`` at every rtol, not ``inf``."""
    s = {"A": {"u": jnp.full((3,), 0.998, dtype)}, "B": {"u": jnp.full((2,), 0.998, dtype)}}
    eps = float(jnp.finfo(dtype).eps)
    for norm in ("mixed", "interface"):
        got = float(residual_precision_floor(s, ["A", "B"], norm, 0.0, rtol, _EDGES))
        assert got == pytest.approx(PRECISION_FLOOR_ULPS * eps / rtol, rel=1e-6), norm


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_the_sixteen_bit_floor_counts_more_than_65504_entries(dtype):
    """``4 eps sqrt(n)`` under "l2" and ``4 eps / rtol`` under "mixed" at n = 70 000."""
    n = 70_000
    s = {"n": {"u": jnp.ones((n,), dtype)}}
    eps = float(jnp.finfo(dtype).eps)
    assert float(residual_precision_floor(s, ["n"], "l2")) == pytest.approx(
        PRECISION_FLOOR_ULPS * eps * math.sqrt(n), rel=1e-6)
    assert float(residual_precision_floor(s, ["n"], "mixed", 0.0, 0.1)) == pytest.approx(
        PRECISION_FLOOR_ULPS * eps / 0.1, rel=1e-6)


class _Field(SimulationNode):
    """``u <- g * inp + c`` elementwise, on an ``n``-entry field of a chosen dtype."""

    def __init__(self, name, g, c, x0, n, dtype):
        super().__init__(name, 1.0)
        self._a = (g, c, x0, n, dtype)

    def initial_state(self):
        _g, _c, x0, n, dt = self._a
        return {"u": jnp.full((n,), x0, dt)}

    def update(self, state, boundary_inputs, dt_):
        g, c, _x0, n, dt = self._a
        inp = boundary_inputs.get("inp", jnp.zeros((n,), dt))
        return {"u": (jnp.asarray(g, dt) * inp + jnp.asarray(c, dt)).astype(dt)}

    def update_evaluations(self):
        return 1.0


def _field_pair(dtype, n, *, x0=0.8, **group):
    """``u_A = 0.9 u_B + 0.1``, ``u_B = 0.9 u_A + 0.1``: fixed point 1, GS rate 0.81."""
    gm = GraphManager()
    gm.add_node(_Field("A", 0.9, 0.1, x0, n, dtype))
    gm.add_node(_Field("B", 0.9, 0.1, x0, n, dtype))
    gm.add_edge("A", "B", "u", "inp")
    gm.add_edge("B", "A", "u", "inp")
    gm.add_coupling_group(["A", "B"], **group)
    gm.compile()
    return gm


@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_a_float16_group_above_65504_entries_does_not_stop_on_a_zero_residual(norm):
    """2 x 40 000 float16 entries: the group iterates as the float32 group does.

    Before, the count overflowed, the residual read 0.0 after 8 passes and
    the group stopped 4.1e-2 from its fixed point with ``converged=True``,
    where the float32 group runs 15 passes to 9.4e-3.
    """
    out = {}
    for dtype in (jnp.float16, jnp.float32):
        gm = _field_pair(dtype, BIG, convergence_norm=norm, rtol=1e-2, max_iterations=50)
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        dist = float(np.max(np.abs(np.asarray(gm.get_node_state("A")["u"], np.float64) - 1.0)))
        out[jnp.dtype(dtype).name] = (d, dist)
    d16, dist16 = out["float16"]
    d32, dist32 = out["float32"]
    assert d16["residual"] > 0.0, dict(d16)
    assert d16["converged"] and d16["iterations"] >= d32["iterations"] - 2, (dict(d16), dict(d32))
    assert dist16 <= 2.0 * dist32, (dist16, dist32)


class _Relay(SimulationNode):
    """``x <- g * inp + c`` in a chosen dtype, one entry."""

    def __init__(self, name, g, c, x0, dtype):
        super().__init__(name, 1.0)
        self._g, self._c, self._x0, self._dt = g, c, x0, dtype

    def initial_state(self):
        return {"x": jnp.asarray([self._x0], self._dt)}

    def update(self, state, boundary_inputs, dt):
        inp = boundary_inputs.get("inp", jnp.zeros((1,), self._dt))
        return {"x": (jnp.asarray(self._g, self._dt) * inp
                      + jnp.asarray(self._c, self._dt)).astype(self._dt)}

    def update_evaluations(self):
        return 1.0


def _relay_pair(dtype, *, g=0.9, c=0.1, **group):
    gm = GraphManager()
    gm.add_node(_Relay("A", g, c, 0.0, dtype))
    gm.add_node(_Relay("B", g, c, 0.0, dtype))
    gm.add_edge("A", "B", "x", "inp")
    gm.add_edge("B", "A", "x", "inp")
    gm.add_coupling_group(["A", "B"], **group)
    gm.compile()
    return gm


@pytest.mark.parametrize("dtype", SIXTEEN)
@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_a_sixteen_bit_residual_at_the_default_rtol_is_finite_and_held_in_float32(dtype, norm):
    """Five passes from zero at ``rtol=1e-6``: a finite residual near the float32 group's.

    Before, it read ``inf`` with ``spectral_error_bound=nan`` on a finite
    state.  The slot it is stored in is float32 too: kept in float16 a
    finite residual above 65 504 overflowed on its way into the report.
    """
    reports = {}
    for dt in (dtype, jnp.float32):
        gm = _relay_pair(dt, convergence_norm=norm, max_iterations=5, diagnostics=True)
        gm.step()
        assert gm._state["_meta"][f"coupling_{KEY}_residual"].dtype == jnp.float32
        reports[dt] = gm.coupling_diagnostics()[KEY]
    d = reports[dtype]
    assert math.isfinite(d["residual"]) and not d["converged"], dict(d)
    # The 16-bit iterate differs from the float32 one by its own rounding.
    assert d["residual"] == pytest.approx(reports[jnp.float32]["residual"], rel=0.05)
    assert d["spectral_usable"] and math.isfinite(d["spectral_error_bound"]), dict(d)


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_every_reader_gives_a_sixteen_bit_groups_verdict(dtype):
    """The report and the profiler read the float32 slots as the loop decided them."""
    for cap, expected in ((60, True), (3, False)):
        gm = _relay_pair(dtype, convergence_norm="mixed", rtol=1e-2, max_iterations=cap)
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        assert d["converged"] is expected, (cap, dict(d))
        assert _meta_converged(gm._state["_meta"], f"coupling_{KEY}_residual",
                               f"coupling_{KEY}_amplification", 1.0, 1.0) is expected


def test_strict_convergence_names_the_cap_on_a_finite_float16_state():
    """A float16 group at its cap on a finite state is told to raise the cap.

    Before, the norm overflowed to ``inf`` on that finite state and the
    check read it as divergence: "its state is non-finite ... no larger
    max_iterations would help" -- while 40 passes converge.
    """
    gm = _relay_pair(jnp.float16, convergence_norm="mixed", max_iterations=5,
                     strict_convergence=True)
    with pytest.raises(Exception, match="Raise max_iterations") as info:
        gm.step()
    assert "non-finite" not in str(info.value)


def test_strict_convergence_names_the_cap_where_the_estimate_alone_overflows():
    """A float32 group whose estimate overflows on a finite, measurable state.

    At ``rtol=1e-25`` an order-one relative change is a ratio of 1e25 and
    its square is past float32's range: the residual is ``inf`` while every
    field is finite and measurable.  That is an unconverged exit whose
    remedy is the tolerance, and the check now says so; it used to read the
    estimate's ``inf`` as a non-finite state.
    """
    gm = _relay_pair(jnp.float32, convergence_norm="mixed", rtol=1e-25, max_iterations=3,
                     strict_convergence=True)
    with pytest.raises(Exception, match="loosen the tolerance") as info:
        gm.step()
    assert "non-finite" not in str(info.value)


def test_strict_convergence_names_a_state_beyond_its_dtypes_range():
    """A finite state too large for the L2 norm to measure a change at is non-finite.

    ``a = 0.5 b + 2e38``, ``b = 0.5 a``: the fixed point, 2.67e38, is a
    finite float32, but its scale's reciprocal is subnormal, so the L2 norm
    cannot evaluate a change there (``_scaled_change``).  Deciding from the
    state keeps that case where the message puts it: "beyond the range its
    dtype can measure a change at".
    """
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.5, 2e38, 0.0, jnp.float32))
    gm.add_node(_Relay("B", 0.5, 0.0, 0.0, jnp.float32))
    gm.add_edge("A", "B", "x", "inp")
    gm.add_edge("B", "A", "x", "inp")
    gm.add_coupling_group(["A", "B"], max_iterations=40, strict_convergence=True)
    gm.compile()
    with pytest.raises(Exception, match="state is non-finite") as info:
        gm.step()
    assert "Raise max_iterations" not in str(info.value)
