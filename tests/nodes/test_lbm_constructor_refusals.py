"""The LBM constructors refuse values that would run a different model.

* ``LBMNode(viscosity=inf)`` passed ``tau > 0.5`` (``tau = inf``) and the
  node ran with no collision at all; ``nan`` passes every comparison.
* ``LBMPipeNode`` had the same gap for ``tau`` and ``tau_tracer``.
* ``LBMPipeNode(propeller_x=99)`` on an 8-plane pipe built no actuator
  disc -- ``mask.at[99]`` drops an out-of-range index without a word -- and
  the pipe ran with no propeller: ``max|u|`` 3.7e-9 after 10 steps, where
  ``propeller_x=4`` reached 5.2e-3.  The default, 10, is outside every pipe
  of 10 planes or fewer.  A disc radius that takes in no cell centre did
  the same.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maddening.nodes.lbm import LBMNode
from maddening.nodes.lbm_pipe import LBMPipeNode

NON_FINITE = [math.inf, -math.inf, math.nan]
PIPE = dict(nx=8, ny=8, nz=8, propeller_strength=0.01)


@pytest.mark.parametrize("value", NON_FINITE, ids=["inf", "-inf", "nan"])
def test_lbm_node_refuses_a_non_finite_viscosity(value):
    with pytest.raises(ValueError, match="viscosity must be a finite number"):
        LBMNode("l", 1.0, grid_shape=(8, 8), lattice="D2Q9", viscosity=value)


def test_lbm_node_takes_a_finite_viscosity():
    node = LBMNode("l", 1.0, grid_shape=(8, 8), lattice="D2Q9", viscosity=0.1)
    assert node.tau == pytest.approx(0.8)


@pytest.mark.parametrize("key", ["tau", "tau_tracer"])
@pytest.mark.parametrize("value", NON_FINITE, ids=["inf", "-inf", "nan"])
def test_lbm_pipe_refuses_a_non_finite_relaxation_time(key, value):
    with pytest.raises(ValueError, match=f"{key} must be a finite number"):
        LBMPipeNode("p", 1.0, propeller_x=4, **PIPE, **{key: value})


@pytest.mark.parametrize("propeller_x", [8, 99, -1, -8])
def test_lbm_pipe_refuses_a_propeller_outside_the_grid(propeller_x):
    with pytest.raises(ValueError, match=rf"propeller_x={propeller_x} is outside the grid"):
        LBMPipeNode("p", 1.0, propeller_x=propeller_x, **PIPE)


def test_the_default_propeller_position_needs_a_pipe_longer_than_ten_planes():
    with pytest.raises(ValueError, match=r"the default, 10, needs nx > 10"):
        LBMPipeNode("p", 1.0, **PIPE)
    LBMPipeNode("p", 1.0, **{**PIPE, "nx": 11})


@pytest.mark.parametrize("propeller_x", [2.5, "4", None])
def test_lbm_pipe_refuses_a_propeller_position_that_is_not_an_index(propeller_x):
    with pytest.raises(ValueError, match="propeller_x must be an integer grid index"):
        LBMPipeNode("p", 1.0, propeller_x=propeller_x, **PIPE)


@pytest.mark.parametrize("radius", [0.0, -0.5, 0.01, math.nan])
def test_lbm_pipe_refuses_a_propeller_disc_that_covers_no_cell(radius):
    with pytest.raises(ValueError, match="covers no cell"):
        LBMPipeNode("p", 1.0, propeller_x=4, propeller_radius=radius, **PIPE)


@pytest.mark.parametrize("propeller_x", [0, 7, np.int64(3)])
def test_a_propeller_inside_the_grid_has_its_disc_on_that_plane(propeller_x):
    node = LBMPipeNode("p", 1.0, propeller_x=propeller_x, **PIPE)
    mask = np.asarray(node._propeller_mask)
    planes = np.flatnonzero(mask.any(axis=(1, 2)))
    assert planes.tolist() == [int(propeller_x)]

