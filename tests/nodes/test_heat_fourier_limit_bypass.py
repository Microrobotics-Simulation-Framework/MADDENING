"""MADD-ANO-002: HeatNode's Fourier limit is checked at construction only.

Since 0.4.0 the constructor refuses a ``timestep`` above its stencil's
Fourier limit (MADD-ANO-009's check, pinned in
``tests/verification/test_mms_order.py::TestHeatStabilityBound``).  What
MADD-ANO-002 keeps open is every route around that check: a ``dt`` handed
to ``update()`` and a ``thermal_diffusivity`` injected through ``params``
both reach an unstable Fourier number, diverge, and say nothing.

These tests pin the limitation, the way ``test_x64_graph_scan_limitation.py``
pins MADD-ANO-017, so they fail in both directions:

* if ``update()`` starts refusing or warning on an unstable ``dt`` or
  injected parameter, the non-finite / no-warning assertions fail and say
  MADD-ANO-002 can be narrowed or closed;
* if the rod the registry entry measures stops producing the figures it
  quotes, the stable-end assertions fail and say the entry's numbers are
  stale.  (An earlier revision of the entry quoted 0.516 and 0.437 for two
  of them; they reproduced under no reading and were withdrawn, so the
  figures it quotes now are pinned here.)

The rod is the entry's own: 20 cells, length 1.0, thermal_diffusivity 0.01,
a half-sine initial profile, and a node constructed at a stable Fo = 0.4.
Float32, the framework default.
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import pytest

from maddening.nodes.heat import HeatNode

_ANOMALY = "MADD-ANO-002"
_UPDATE_THE_ENTRY = (
    f"Update {_ANOMALY} in docs/validation/known_anomalies.yaml in the same "
    f"commit: its description records this behaviour."
)

N_CELLS, LENGTH, ALPHA = 20, 1.0, 0.01
DX = LENGTH / N_CELLS
STEPS = 60


def _dt(fourier: float) -> float:
    return fourier * DX * DX / ALPHA


def _node() -> HeatNode:
    """The entry's node, constructed at a Fourier number the check accepts."""
    return HeatNode("h", timestep=_dt(0.4), n_cells=N_CELLS, length=LENGTH,
                    thermal_diffusivity=ALPHA)


def _half_sine() -> dict:
    x = (jnp.arange(N_CELLS) + 0.5) * DX
    return {"temperature": jnp.sin(math.pi * x / LENGTH).astype(jnp.float32)}


_DIRICHLET_ZERO = {
    "left_temperature": jnp.float32(0.0),
    "right_temperature": jnp.float32(0.0),
}


def _run(node, dt, boundary, params=None):
    """Advance ``STEPS`` steps and return (max |T|, warnings raised)."""
    state = _half_sine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(STEPS):
            if params is None:
                state = node.update(state, boundary, dt)
            else:
                state = node.update(state, boundary, dt, params=params)
    return float(jnp.max(jnp.abs(state["temperature"]))), list(caught)


def test_the_constructor_does_refuse_the_same_fourier_number():
    """The contrast that makes the two bypasses below a defect: Fo = 5 on
    this rod is refused when it arrives through the constructor."""
    with pytest.raises(ValueError, match=r"Fourier number .* is 5"):
        HeatNode("h", timestep=_dt(5.0), n_cells=N_CELLS, length=LENGTH,
                 thermal_diffusivity=ALPHA)


@pytest.mark.parametrize("boundary", [{}, _DIRICHLET_ZERO],
                         ids=["no-boundary-data", "dirichlet-zero"])
def test_an_unstable_dt_passed_to_update_diverges_silently(boundary):
    peak, caught = _run(_node(), _dt(5.0), boundary)
    assert not math.isfinite(peak), (
        f"update() with dt at Fo = 5 stayed finite (max|T| = {peak}).  "
        f"{_UPDATE_THE_ENTRY}"
    )
    assert caught == [], (
        f"update() now warns on an unstable dt: {[str(w.message) for w in caught]}.  "
        f"{_UPDATE_THE_ENTRY}"
    )


@pytest.mark.parametrize("boundary", [{}, _DIRICHLET_ZERO],
                         ids=["no-boundary-data", "dirichlet-zero"])
def test_an_unstable_injected_thermal_diffusivity_diverges_silently(boundary):
    """0.125 at the node's own Fo = 0.4 timestep is Fo = 5 again."""
    peak, caught = _run(_node(), _dt(0.4), boundary,
                        params={"thermal_diffusivity": 0.125})
    assert not math.isfinite(peak), (
        f"an injected thermal_diffusivity at Fo = 5 stayed finite "
        f"(max|T| = {peak}).  {_UPDATE_THE_ENTRY}"
    )
    assert caught == [], (
        f"update() now warns on an unstable injected parameter: "
        f"{[str(w.message) for w in caught]}.  {_UPDATE_THE_ENTRY}"
    )


@pytest.mark.parametrize(
    "fourier, boundary, recorded",
    [
        (0.4, {}, 0.675313),
        (0.5, {}, 0.657864),
        (0.4, _DIRICHLET_ZERO, 0.550473),
        (0.5, _DIRICHLET_ZERO, 0.474083),
    ],
    ids=["fo0.4-no-bc", "fo0.5-no-bc", "fo0.4-dirichlet", "fo0.5-dirichlet"],
)
def test_the_stable_end_of_the_rod_decays_to_the_recorded_figures(
    fourier, boundary, recorded
):
    peak, caught = _run(_node(), _dt(fourier), boundary)
    assert caught == []
    # rel=1e-5 is float32 accumulation over 60 steps with room for a
    # different XLA; it is far tighter than the withdrawn figures' error.
    assert peak == pytest.approx(recorded, rel=1e-5), (
        f"max|T| after {STEPS} steps at Fo = {fourier} is {peak:.6f}, not the "
        f"{recorded} {_ANOMALY} records.  {_UPDATE_THE_ENTRY}"
    )
