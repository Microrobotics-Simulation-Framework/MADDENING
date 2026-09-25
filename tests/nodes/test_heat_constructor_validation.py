"""``HeatNode`` refuses constants no rod can have.

Each case below used to construct, compile and step with no refusal and no
warning, and to answer wrongly (audit_040_p4_3, sharding-params; the same
at v0.3.1, checked on a ``git archive`` of the tag):

* ``length=-1``: the Laplacian reads ``dx**2`` and is unchanged, but both
  rod-end fluxes change sign (+12.87 where the rod reports -12.87);
* ``grid_points`` with two points swapped: finite and wrong, because the
  variable-spacing stencil replaces a non-positive span with 1.0;
* ``timestep < 0`` or ``thermal_diffusivity < 0``: the heat equation run
  backwards (anti-diffusion).  The Fourier check was skipped for both;
* a non-finite value of any of these.

A zero diffusivity is a rod that does not conduct, and is kept.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.heat import HeatNode

T0 = [300.0 + 10.0 * i for i in range(10)]
MONOTONE = [0.05, 0.15, 0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 0.97, 0.99]


@pytest.mark.parametrize("length", [-1.0, 0.0, float("inf"), float("nan")],
                         ids=["negative", "zero", "inf", "nan"])
def test_a_length_that_is_not_a_finite_positive_number_is_refused(length):
    with pytest.raises(ValueError, match=r"length must be a finite number > 0"):
        HeatNode("h", 1e-3, n_cells=10, length=length, initial_temperature=T0)


def test_the_negative_length_this_used_to_accept_flipped_both_fluxes():
    """Why the refusal: the same rod at +length reports the flux the
    physics gives, and nothing about -length is a rod.  Read through the
    flux formula (the Laplacian cannot see the sign), for the record."""
    rod = HeatNode("h", 1e-3, n_cells=10, length=1.0, initial_temperature=T0)
    ends = {"left_temperature": jnp.float32(250.0),
            "right_temperature": jnp.float32(400.0)}
    state = rod.initial_state()
    forward = rod.compute_boundary_fluxes(state, ends, 1e-3)
    flipped = rod.compute_boundary_fluxes(state, ends, 1e-3, params={"length": -1.0})
    assert float(forward["left_heat_flux"]) == pytest.approx(
        -float(flipped["left_heat_flux"]))
    assert float(forward["left_heat_flux"]) < 0.0


@pytest.mark.parametrize("timestep", [-1e-3, 0.0, float("inf"), float("nan")],
                         ids=["negative", "zero", "inf", "nan"])
def test_a_timestep_that_is_not_a_finite_positive_number_is_refused(timestep):
    with pytest.raises(ValueError, match=r"timestep must be a finite number > 0"):
        HeatNode("h", timestep, n_cells=10, initial_temperature=T0)


@pytest.mark.parametrize("alpha", [-0.01, float("inf"), float("nan")],
                         ids=["negative", "inf", "nan"])
def test_a_diffusivity_that_is_negative_or_not_finite_is_refused(alpha):
    with pytest.raises(ValueError, match=r"thermal_diffusivity must be a finite number >= 0"):
        HeatNode("h", 1e-3, n_cells=10, thermal_diffusivity=alpha, initial_temperature=T0)


def test_a_zero_diffusivity_is_a_rod_that_does_not_conduct():
    rod = HeatNode("h", 1e-3, n_cells=10, thermal_diffusivity=0.0,
                   initial_temperature=np.sin(np.arange(10.0)).tolist())
    state = rod.initial_state()
    after = rod.update(state, {"left_temperature": jnp.float32(5.0)}, 1e-3)
    np.testing.assert_array_equal(np.asarray(after["temperature"]),
                                  np.asarray(state["temperature"]))


@pytest.mark.parametrize("points, where", [
    ([0.05, 0.3, 0.15, 0.5, 0.65, 0.8, 0.9, 0.95, 0.97, 0.99], "indices 1 and 2"),
    (MONOTONE[::-1], "indices 0 and 1"),
    ([0.05, 0.15, 0.15, 0.5, 0.65, 0.8, 0.9, 0.95, 0.97, 0.99], "indices 1 and 2"),
], ids=["two-swapped", "reversed", "repeated"])
def test_grid_points_out_of_order_are_refused(points, where):
    with pytest.raises(ValueError, match=f"strictly increasing.*{where}"):
        HeatNode("h", 1e-4, n_cells=10, grid_points=points, initial_temperature=T0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_grid_points_that_are_not_finite_are_refused(bad):
    points = list(MONOTONE)
    points[4] = bad
    with pytest.raises(ValueError, match=r"grid_points must be finite.*index 4"):
        HeatNode("h", 1e-4, n_cells=10, grid_points=points, initial_temperature=T0)


def test_an_ordinary_rod_on_either_grid_is_still_built():
    HeatNode("h", 1e-3, n_cells=10, length=1.0, thermal_diffusivity=0.01)
    HeatNode("h", 1e-4, n_cells=10, grid_points=MONOTONE, initial_temperature=T0)


def test_a_traced_constant_is_not_judged():
    """The checks read concrete numbers only; a rod built inside a trace
    (its diffusivity a tracer) is left to run, as the Fourier check always
    was."""

    def temperature_after_one_step(alpha):
        rod = HeatNode("h", 1e-3, n_cells=10, thermal_diffusivity=alpha,
                       initial_temperature=T0)
        return rod.update(rod.initial_state(), {}, 1e-3)["temperature"]

    out = jax.jit(temperature_after_one_step)(jnp.float32(0.01))
    assert np.all(np.isfinite(np.asarray(out)))
