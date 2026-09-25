"""Two ``HeatNode`` rods coupled end to end have a lower Fourier limit.

Each rod's end-cell temperature is the other's Dirichlet datum, and a
coupling group converges the exchange within the step.  The converged pair
is not one continuous rod.  Each datum becomes an implicit value, the other
rod's end cell at the new time, while the interiors stay explicit, and the pair has an
interface mode that alternates in sign every step.  Its amplification
reaches -1 at a Fourier number below the single-rod limit each constructor
checks:

* ``stencil_order=2``: exactly 3/8 (the mirror ghost ``2*T_b - T[0]``);
  -1.5 at 0.4 and -4 at 0.45.  The single-rod limit is 1/2.
* ``stencil_order=4``: 0.2261266 as ``n_cells`` goes to infinity, 0.2261215
  at ``n_cells = 5``.  The cubic ghost's larger interface gain, 16/5 of the
  datum where the mirror ghost has 2, brings it well under the order-4
  single-rod limit of 5/16.  The node records 0.226.

MADD-ANO-050 (open).  Nothing refuses such a graph; ``compile()`` warns.
The derivation is in the comment on
``maddening.nodes.heat._COUPLED_PAIR_MAX_FOURIER_NUMBER``.

These tests pin what stays open, so that a fix, or a drifted figure, fails
here.  They cover the growth rate on either side of each limit, the
limits themselves as properties of the node's own operator, the single rod's
stability at the same Fourier number with fixed data, and the warning.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import extract_first, extract_last
from maddening.nodes.heat import (
    _COUPLED_PAIR_MAX_FOURIER_NUMBER,
    MAX_FOURIER_NUMBER,
    HeatNode,
)

N_CELLS = 8
LENGTH = 1.0
ALPHA = 0.1


def _dt(fourier, n_cells=N_CELLS, alpha=ALPHA):
    dx = LENGTH / n_cells
    return fourier * dx * dx / alpha


def _pair_graph(fourier, order=2, *, grouped=True, max_iterations=100):
    dt = _dt(fourier)
    gm = GraphManager()
    gm.add_node(HeatNode("a", dt, n_cells=N_CELLS, length=LENGTH,
                         thermal_diffusivity=ALPHA, initial_temperature=300.0,
                         stencil_order=order))
    gm.add_node(HeatNode("b", dt, n_cells=N_CELLS, length=LENGTH,
                         thermal_diffusivity=ALPHA, initial_temperature=360.0,
                         stencil_order=order))
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=extract_last)
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=extract_first)
    if grouped:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.add_coupling_group(["a", "b"], max_iterations=max_iterations,
                                  tolerance=1e-9)
    return gm


def _pair_history(fourier, steps, order=2):
    """Temperatures of both rods, ``(steps, 2 * N_CELLS)``, from a converged
    end-to-end exchange started at 300 K / 360 K.

    ``compile()`` warns about the pair exactly when it is past the limit
    (MADD-ANO-050), so every run here also checks the warning at its own
    Fourier number.
    """
    gm = _pair_graph(fourier, order)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    flagged = [w for w in caught if "MADD-ANO-050" in str(w.message)]
    assert bool(flagged) == (fourier > _COUPLED_PAIR_MAX_FOURIER_NUMBER[order]), (
        fourier, order, [str(w.message) for w in caught])
    out = gm.run_scan_with_history(steps)
    hist = out[1] if isinstance(out, tuple) else out
    return np.concatenate([np.asarray(hist["a"]["temperature"]),
                           np.asarray(hist["b"]["temperature"])],
                          axis=1).astype(np.float64)


def _alternating_part(history):
    """The part of the field that alternates in time, per step, at the cell
    where it is largest (signed)."""
    part = history[1:-1] - 0.5 * (history[:-2] + history[2:])
    at = np.argmax(np.abs(np.nan_to_num(part)), axis=1)
    return part[np.arange(len(part)), at]


def _alternating_ratio(history):
    """Per-step ratio of the part of the field that alternates in time,
    over the steps where it is measurable (above rounding, below overflow)."""
    signed = _alternating_part(history)
    size = np.abs(signed)
    usable = np.isfinite(size) & (size > 1e-2) & (size < 1e30)
    pairs = np.where(usable[:-1] & usable[1:])[0]
    return signed[pairs + 1] / signed[pairs]


# ---------------------------------------------------------------------------
# stencil_order=2: 3/8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fourier, amplification", [(0.40, -1.5), (0.45, -4.0)],
                         ids=["fo-0.40", "fo-0.45"])
def test_a_converged_end_to_end_pair_diverges_above_three_eighths(fourier, amplification):
    history = _pair_history(fourier, 60)
    ratios = _alternating_ratio(history)
    assert len(ratios) >= 5
    np.testing.assert_allclose(ratios[-5:], amplification, rtol=1e-3)
    assert np.nanmax(np.abs(history[-1] - 330.0)) > 1e5 or not np.all(np.isfinite(history[-1]))


def test_the_same_pair_below_three_eighths_stays_bounded():
    history = _pair_history(0.36, 60)
    assert np.all(np.isfinite(history))
    # The maximum principle of the continuous problem: nothing leaves the
    # initial range.
    assert history.min() >= 300.0 - 1e-3 and history.max() <= 360.0 + 1e-3


def test_one_rod_with_fixed_data_is_stable_at_the_same_fourier_number():
    """The node's own limit is 1/2 for fixed Dirichlet data, which is what
    its constructor checks, and it holds at Fo = 0.45."""
    gm = GraphManager()
    gm.add_node(HeatNode("a", _dt(0.45), n_cells=N_CELLS, length=LENGTH,
                         thermal_diffusivity=ALPHA, initial_temperature=360.0))
    # Both ends held at 0 K (an external input defaults to zero).
    gm.add_external_input("a", "left_temperature")
    gm.add_external_input("a", "right_temperature")
    gm.compile()
    out = gm.run_scan_with_history(60)
    hist = out[1] if isinstance(out, tuple) else out
    temperature = np.asarray(hist["a"]["temperature"])
    assert np.all(np.isfinite(temperature))
    assert temperature.min() >= -1e-3 and temperature.max() <= 360.0 + 1e-3


# ---------------------------------------------------------------------------
# stencil_order=4: 0.226, not 3/8 and not the constructor's 5/16
# ---------------------------------------------------------------------------


#: The interface mode's amplification per step for the order-4 pair: the
#: leading eigenvalue of the pair's amplification matrix built from
#: ``_compute_laplacian`` (float64, the same at 8 and 20 cells).
_ORDER4_AMPLIFICATION = {0.25: -1.865520, 0.28: -4.719616}


@pytest.mark.parametrize("fourier", sorted(_ORDER4_AMPLIFICATION),
                         ids=lambda f: f"fo-{f:.2f}")
def test_a_converged_order_four_pair_diverges_below_its_constructor_limit(fourier):
    """Both Fourier numbers are inside 5/16, the limit the order-4
    constructor checks, and inside 3/8, the figure MADD-ANO-050 quoted
    before it was derived per order.  The measured growth is the predicted
    interface mode (audit_040_p4_2, release-record, H2)."""
    assert fourier < MAX_FOURIER_NUMBER[4]
    history = _pair_history(fourier, 60, order=4)
    ratios = _alternating_ratio(history)
    assert len(ratios) >= 5
    np.testing.assert_allclose(ratios[-5:], _ORDER4_AMPLIFICATION[fourier], rtol=1e-3)
    assert np.nanmax(np.abs(history[-1] - 330.0)) > 1e5 or not np.all(np.isfinite(history[-1]))


def test_the_same_order_four_pair_below_its_limit_stays_bounded():
    """At Fo = 0.22 the interface mode decays (amplification -0.848).  The
    order-4 scheme has no discrete maximum principle, so the field may
    overshoot the initial range by a little (1.95 K measured at 0.20) but
    must not grow."""
    history = _pair_history(0.22, 300, order=4)
    assert np.all(np.isfinite(history))
    assert history.min() >= 300.0 - 5.0 and history.max() <= 360.0 + 5.0
    assert np.max(np.abs(_alternating_part(history)[-20:])) < 1e-3


def _pair_amplification_limit(order, n_cells):
    """Largest Fo at which the converged pair's amplification matrix has
    spectral radius <= 1, built from the node's own ``_compute_laplacian``.

    The step is ``(I - Fo Q) T' = (I + Fo P) T``: ``P`` the two rods'
    operators with their far ends on the default datum (their own end
    cell), ``Q`` the implicit cross terms through the exchanged data.
    """
    node = HeatNode("r", 1e-9, n_cells=n_cells, length=float(n_cells),
                    thermal_diffusivity=1.0, stencil_order=order)

    def columns(left_default, right_default):
        """Laplacian of each unit vector, one rod end on its default datum
        (its own end cell), the other end's datum zero."""
        def lap(e):
            left = e[0] if left_default else 0.0
            right = e[-1] if right_default else 0.0
            return node._compute_laplacian(e, left, right, length=float(n_cells))
        rows = jax.vmap(lap)(jnp.eye(n_cells, dtype=jnp.float32))
        return np.asarray(rows, dtype=np.float64).T

    def datum_response(left, right):
        zero = jnp.zeros(n_cells, dtype=jnp.float32)
        return np.asarray(node._compute_laplacian(zero, left, right,
                                                  length=float(n_cells)),
                          dtype=np.float64)

    n = n_cells
    P = np.zeros((2 * n, 2 * n))
    Q = np.zeros((2 * n, 2 * n))
    P[:n, :n] = columns(True, False)          # rod a: right datum from b
    P[n:, n:] = columns(False, True)          # rod b: left datum from a
    Q[:n, n] = datum_response(0.0, 1.0)       # a's right datum is b'[0]
    Q[n:, n - 1] = datum_response(1.0, 0.0)   # b's left datum is a'[-1]
    eye = np.eye(2 * n)

    def stable(fo):
        amp = np.linalg.solve(eye - fo * Q, eye + fo * P)
        return np.max(np.abs(np.linalg.eigvals(amp))) <= 1.0 + 1e-6

    # ``I - Fo Q`` is singular at the single-rod limit itself (the exchange's
    # fixed-point gain, 2 Fo or 16/5 Fo, reaches 1 there), so bracket inside it.
    lo, hi = 0.05, 0.95 * MAX_FOURIER_NUMBER[order]
    assert stable(lo) and not stable(hi)
    while hi - lo > 1e-7:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


@pytest.mark.parametrize("order, sharp", [(2, 0.375), (4, 0.2261266)],
                         ids=["order-2", "order-4"])
@pytest.mark.parametrize("n_cells", [5, 8, 20])
def test_the_recorded_pair_limit_sits_under_the_pair_operator_limit(order, sharp, n_cells):
    """The constant the warning uses, against the operator it describes.

    At order 2 the limit is 3/8 from above (0.3750085 at 5 cells); at
    order 4 it is 0.2261266 from below (0.2261215 at 5 cells), and 0.226
    is under every one.  A change to either ghost closure moves the
    operator and fails here before it can make the warning wrong.
    """
    limit = _pair_amplification_limit(order, n_cells)
    recorded = _COUPLED_PAIR_MAX_FOURIER_NUMBER[order]
    assert recorded <= limit + 1e-6
    assert limit - recorded < 2e-4
    assert abs(limit - sharp) < (1e-5 if n_cells >= 8 else 1e-4)


# ---------------------------------------------------------------------------
# The compile-time warning
# ---------------------------------------------------------------------------


def _anomaly_issues(gm):
    return [i for i in gm.validate() if "MADD-ANO-050" in i]


_EPS = 5e-4


@pytest.mark.parametrize("order, fourier", [
    (2, 0.40),
    (4, _COUPLED_PAIR_MAX_FOURIER_NUMBER[4] + _EPS),
], ids=["order-2-fo-0.40", "order-4-just-above"])
def test_compile_warns_about_a_pair_past_its_limit(order, fourier):
    gm = _pair_graph(fourier, order)
    with pytest.warns(UserWarning, match="MADD-ANO-050") as record:
        gm.compile()
    message = " ".join(str(w.message) for w in record
                       if "MADD-ANO-050" in str(w.message))
    limit = _COUPLED_PAIR_MAX_FOURIER_NUMBER[order]
    assert "'a'" in message and "'b'" in message
    assert f"Fo = {fourier:.4g}" in message
    assert f"limit {limit:g}" in message
    assert f"stencil_order={order}" in message
    assert "both are above the limit" in message


@pytest.mark.parametrize("order, fourier", [
    (2, 0.36),
    (4, _COUPLED_PAIR_MAX_FOURIER_NUMBER[4] - _EPS),
], ids=["order-2-fo-0.36", "order-4-just-below"])
def test_no_warning_for_a_pair_below_its_limit(order, fourier):
    assert _anomaly_issues(_pair_graph(fourier, order)) == []


def test_the_order_four_threshold_is_not_the_order_two_one():
    """0.30 is inside 3/8 and inside 5/16, so a warning keyed on either
    figure would stay silent.  The order-4 pair is past its own limit."""
    assert len(_anomaly_issues(_pair_graph(0.30, 4))) == 1
    assert _anomaly_issues(_pair_graph(0.30, 2)) == []


def test_no_warning_for_a_lagged_exchange():
    """Without a coupling group the data lags a step and the pair stays
    bounded at 0.45 (MADD-ANO-050's own control)."""
    assert _anomaly_issues(_pair_graph(0.45, grouped=False)) == []


def test_no_warning_for_a_single_staggered_pass():
    """``max_iterations=1`` is one staggered pass, the lagged exchange in
    effect, not a converged one."""
    assert _anomaly_issues(_pair_graph(0.45, max_iterations=1)) == []


def test_no_warning_for_a_single_rod():
    gm = GraphManager()
    gm.add_node(HeatNode("a", _dt(0.45), n_cells=N_CELLS, thermal_diffusivity=ALPHA))
    gm.add_external_input("a", "left_temperature")
    assert _anomaly_issues(gm) == []


def _rods(gm, fouriers, orders=None):
    """Rods ``a``, ``b``, ... on one timestep, each at its own Fourier number
    (set through its diffusivity)."""
    orders = orders or [2] * len(fouriers)
    dt = _dt(fouriers[0])
    for i, (fo, order) in enumerate(zip(fouriers, orders)):
        alpha = ALPHA * fo / fouriers[0]
        gm.add_node(HeatNode(chr(ord("a") + i), dt, n_cells=N_CELLS,
                             thermal_diffusivity=alpha, stencil_order=order))
    return gm


def _group(gm, names, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(names, max_iterations=50, tolerance=1e-8, **kw)


def _couple(gm, left, right):
    """``left``'s right end against ``right``'s left end, both ways."""
    gm.add_edge(left, right, "temperature", "left_temperature", transform=extract_last)
    gm.add_edge(right, left, "temperature", "right_temperature", transform=extract_first)


def test_the_mirrored_orientation_is_the_same_pair():
    """``a``'s left end against ``b``'s right end: the same interface."""
    gm = _rods(GraphManager(), [0.40, 0.40])
    gm.add_edge("a", "b", "temperature", "right_temperature", transform=extract_first)
    gm.add_edge("b", "a", "temperature", "left_temperature", transform=extract_last)
    _group(gm, ["a", "b"])
    issues = _anomaly_issues(gm)
    assert len(issues) == 1 and "left end cell of 'a'" in issues[0]


def test_no_warning_when_the_ends_do_not_meet():
    """``a``'s last cell sets ``b``'s left datum, but ``b``'s first cell
    sets ``a``'s *left* datum, the end the first edge does not come from.
    That is no interface: the implicit dependence does not close."""
    gm = _rods(GraphManager(), [0.45, 0.45])
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=extract_last)
    gm.add_edge("b", "a", "temperature", "left_temperature", transform=extract_first)
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_no_warning_for_a_one_way_feed():
    gm = _rods(GraphManager(), [0.45, 0.45])
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=extract_last)
    gm.add_edge("b", "a", "temperature", "heat_source")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_a_transform_it_does_not_recognise_is_not_judged():
    """Documented blind spot: only the built-in extract_first/extract_last
    are read, not a lambda that happens to do the same thing."""
    gm = _rods(GraphManager(), [0.45, 0.45])
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=lambda t: t[-1])
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=extract_first)
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_no_warning_when_something_else_also_writes_the_datum():
    gm = _rods(GraphManager(), [0.45, 0.45])
    _couple(gm, "a", "b")
    gm.add_external_input("a", "right_temperature")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_no_warning_for_an_additive_exchange():
    gm = _rods(GraphManager(), [0.45, 0.45])
    gm.add_edge("a", "b", "temperature", "left_temperature",
                transform=extract_last, additive=True)
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=extract_first)
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_no_warning_for_an_edge_outside_the_group():
    """``b``-``c`` exchange lagged across the group boundary, however high
    ``c``'s Fourier number.  The ``a``-``b`` pair inside is below its limit."""
    gm = _rods(GraphManager(), [0.30, 0.30, 0.45])
    _couple(gm, "a", "b")
    _couple(gm, "b", "c")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_no_warning_for_non_uniform_rods():
    """The pair limit is derived for the uniform stencil, so a rod on
    ``grid_points`` is not judged (documented)."""
    gm = GraphManager()
    points = ((np.arange(N_CELLS) + 0.5) / N_CELLS).tolist()
    for name in ("a", "b"):
        gm.add_node(HeatNode(name, _dt(0.45), n_cells=N_CELLS,
                             thermal_diffusivity=ALPHA, grid_points=points))
    _couple(gm, "a", "b")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_each_interface_of_a_chain_is_judged():
    gm = _rods(GraphManager(), [0.40, 0.40, 0.40])
    _couple(gm, "a", "b")
    _couple(gm, "b", "c")
    _group(gm, ["a", "b", "c"])
    issues = _anomaly_issues(gm)
    assert len(issues) == 2
    assert any("'a' and 'b'" in i for i in issues)
    assert any("'b' and 'c'" in i for i in issues)


def test_one_rod_past_its_own_limit_is_named_alone():
    """Mixed orders: each rod against the limit of its own stencil.  At
    0.30 (order 2) and 0.23 (order 4) only the order-4 rod is past it."""
    gm = _rods(GraphManager(), [0.30, 0.23], orders=[2, 4])
    _couple(gm, "a", "b")
    _group(gm, ["a", "b"])
    issues = _anomaly_issues(gm)
    assert len(issues) == 1 and "'b' is above the limit" in issues[0]


def test_a_pair_below_both_of_its_own_limits_is_not_flagged():
    """0.37 at order 2 and 0.225 at order 4: each under its own limit."""
    gm = _rods(GraphManager(), [0.37, 0.225], orders=[2, 4])
    _couple(gm, "a", "b")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []


def test_a_diffusivity_written_into_the_graph_params_is_the_one_judged():
    """``compile()`` runs the live ``gm.params`` leaves, so the warning
    reads them: a rod built at 0.30 with its diffusivity written up to
    Fo = 0.45 is past the limit."""
    gm = _rods(GraphManager(), [0.30, 0.30])
    _couple(gm, "a", "b")
    _group(gm, ["a", "b"])
    assert _anomaly_issues(gm) == []
    gm.params["nodes"]["a"] = {"thermal_diffusivity": jnp.float32(ALPHA * 1.5)}
    issues = _anomaly_issues(gm)
    assert len(issues) == 1 and "'a' is above the limit" in issues[0]
    assert "Fo = 0.45" in issues[0]


def test_rods_of_different_timesteps_in_a_subcycled_group_are_not_judged():
    """Sub-cycling steps the fast rod several times on one datum, a
    different scheme from the one the limit is derived for."""
    gm = GraphManager()
    dt = _dt(0.45)
    gm.add_node(HeatNode("a", dt, n_cells=N_CELLS, thermal_diffusivity=ALPHA))
    gm.add_node(HeatNode("b", dt / 2, n_cells=N_CELLS, thermal_diffusivity=2 * ALPHA))
    _couple(gm, "a", "b")
    _group(gm, ["a", "b"], subcycling=True)
    assert _anomaly_issues(gm) == []
