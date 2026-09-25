"""Two ``HeatNode`` rods coupled end to end are unstable above Fo = 3/8.

Each rod's end-cell temperature is the other's Dirichlet datum, and a
coupling group converges the exchange within the step.  ``HeatNode``
imposes the datum through the ghost ``2*T_b - T[0]``, so the converged
pair is not one continuous rod: it has an interface mode whose per-step
amplification is exactly -1 at Fo = 3/8, -1.5 at 0.4 and -4 at 0.45 --
while the constructor accepts each rod up to its own fixed-data limit of
1/2.  Nothing refuses such a graph (MADD-ANO-046, open).

These tests pin what stays open, so that a fix -- or a drifted figure --
fails here: the growth rate on either side of the limit, and the single
rod's stability at the same Fourier number with fixed data.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import extract_first, extract_last
from maddening.nodes.heat import HeatNode

N_CELLS = 8
LENGTH = 1.0
ALPHA = 0.1


def _dt(fourier):
    dx = LENGTH / N_CELLS
    return fourier * dx * dx / ALPHA


def _pair_history(fourier, steps):
    """Temperatures of both rods, ``(steps, 2 * N_CELLS)``, from a converged
    end-to-end exchange started at 300 K / 360 K."""
    dt = _dt(fourier)
    gm = GraphManager()
    gm.add_node(HeatNode("a", dt, n_cells=N_CELLS, length=LENGTH,
                         thermal_diffusivity=ALPHA, initial_temperature=300.0))
    gm.add_node(HeatNode("b", dt, n_cells=N_CELLS, length=LENGTH,
                         thermal_diffusivity=ALPHA, initial_temperature=360.0))
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=extract_last)
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=extract_first)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["a", "b"], max_iterations=100, tolerance=1e-9)
    gm.compile()
    out = gm.run_scan_with_history(steps)
    hist = out[1] if isinstance(out, tuple) else out
    return np.concatenate([np.asarray(hist["a"]["temperature"]),
                           np.asarray(hist["b"]["temperature"])],
                          axis=1).astype(np.float64)


def _alternating_ratio(history):
    """Per-step ratio of the part of the field that alternates in time,
    over the steps where it is measurable (above rounding, below overflow)."""
    part = history[1:-1] - 0.5 * (history[:-2] + history[2:])
    at = np.argmax(np.abs(np.nan_to_num(part)), axis=1)
    signed = part[np.arange(len(part)), at]
    size = np.abs(signed)
    usable = np.isfinite(size) & (size > 1e-2) & (size < 1e30)
    pairs = np.where(usable[:-1] & usable[1:])[0]
    return signed[pairs + 1] / signed[pairs]


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
