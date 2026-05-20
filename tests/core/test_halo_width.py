"""Tests for the ``halo_width`` contract and ``requires_halo`` compat shim.

Covers M1 of the v0.2 halo-exchange roadmap.  Verifies:

1. Every in-tree node declares ``halo_width`` correctly.
2. The ``requires_halo`` property derives from ``halo_width`` for new nodes.
3. Legacy subclasses that override ``requires_halo`` instead of
   ``halo_width`` emit a ``DeprecationWarning`` at class creation time.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.health_check import HealthCheckNode
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.lbm import LBMNode
from maddening.nodes.lbm_pipe import LBMPipeNode
from maddening.nodes.rigid_body import RigidBodyNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


# ---------------------------------------------------------------------------
# In-tree node halo widths
# ---------------------------------------------------------------------------


def _make_lbm_3d() -> LBMNode:
    return LBMNode(
        name="lbm3d", timestep=1.0, grid_shape=(8, 8, 8), viscosity=0.1,
        lattice="D3Q19",
    )


def _make_lbm_2d() -> LBMNode:
    return LBMNode(
        name="lbm2d", timestep=1.0, grid_shape=(8, 8), viscosity=0.1,
        lattice="D2Q9",
    )


def _make_lbm_pipe() -> LBMPipeNode:
    return LBMPipeNode(
        name="pipe", timestep=1.0, nx=8, ny=8, nz=8, tau=0.6,
        initial_velocity=0.01, propeller_strength=0.0,
    )


def _make_heat(stencil_order: int = 2) -> HeatNode:
    return HeatNode(
        name="heat", timestep=0.01, n_cells=16, length=1.0,
        thermal_diffusivity=0.1, stencil_order=stencil_order,
    )


@pytest.mark.parametrize("factory,expected", [
    (lambda: BallNode(name="b", timestep=0.01, initial_position=0.0,
                       initial_velocity=0.0), {}),
    (lambda: TableNode(name="t", timestep=0.01), {}),
    (lambda: SpringDamperNode(name="s", timestep=0.01, stiffness=1.0,
                              damping=0.1, rest_length=1.0), {}),
    (lambda: RigidBodyNode(name="rb", timestep=0.01, mass=1.0), {}),
    (lambda: RigidBody2DNode(name="rb2d", timestep=0.01, mass=1.0,
                              inertia=1.0), {}),
    (lambda: HealthCheckNode(name="hc", timestep=0.01), {}),
    (lambda: HeartPumpNode(name="hp", timestep=0.01), {}),
    (lambda: _make_heat(stencil_order=2), {0: 1}),
    (lambda: _make_heat(stencil_order=4), {0: 2}),
    (_make_lbm_3d, {0: 1, 1: 1, 2: 1}),
    (_make_lbm_2d, {0: 1, 1: 1}),
    (_make_lbm_pipe, {0: 1, 1: 1, 2: 1}),
])
def test_in_tree_node_halo_width(factory, expected):
    node = factory()
    assert node.halo_width() == expected
    assert node.requires_halo is bool(expected)


# ---------------------------------------------------------------------------
# Compat shim
# ---------------------------------------------------------------------------


class _NewStyleNode(SimulationNode):
    """Subclass that uses the new ``halo_width`` API."""

    def halo_width(self) -> dict[int, int]:
        return {0: 3}

    def initial_state(self) -> dict:
        return {"x": jnp.zeros(4)}

    def update(self, state, boundary_inputs, dt):
        return state


def test_new_style_subclass_no_warning():
    """A subclass that overrides ``halo_width`` must not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class _NoWarn(SimulationNode):
            def halo_width(self) -> dict[int, int]:
                return {}

            def initial_state(self):
                return {"x": jnp.zeros(1)}

            def update(self, state, boundary_inputs, dt):
                return state


def test_new_style_subclass_property_derives_correctly():
    node = _NewStyleNode(name="n", timestep=0.1)
    assert node.halo_width() == {0: 3}
    assert node.requires_halo is True


def test_legacy_subclass_emits_future_warning():
    """Subclasses overriding only ``requires_halo`` warn at class creation.

    v0.2 escalates this from ``DeprecationWarning`` to
    :class:`FutureWarning` so it surfaces in notebooks where
    DeprecationWarning is silenced by default.  In v0.3 the warning
    becomes a :class:`maddening.warnings.MigrationError`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class _Legacy(SimulationNode):
            @property
            def requires_halo(self) -> bool:
                return True

            def initial_state(self):
                return {"x": jnp.zeros(1)}

            def update(self, state, boundary_inputs, dt):
                return state

        future_warnings = [
            w for w in caught if issubclass(w.category, FutureWarning)
        ]
        assert future_warnings, "expected a FutureWarning at class creation"
        msg = str(future_warnings[0].message)
        assert "_Legacy" in msg
        assert "halo_width" in msg
        assert "MigrationError" in msg  # forward-link to v0.3 behaviour


def test_legacy_subclass_requires_halo_still_works():
    """The compat shim leaves legacy subclasses functional under v0.2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        class _Legacy(SimulationNode):
            @property
            def requires_halo(self) -> bool:
                return True

            def initial_state(self):
                return {"x": jnp.zeros(1)}

            def update(self, state, boundary_inputs, dt):
                return state

        node = _Legacy(name="n", timestep=0.1)

    assert node.requires_halo is True
    # The legacy subclass did NOT override halo_width, so it returns the
    # base default (empty). M4 will decide whether to refuse to shard such
    # nodes or imply a default halo from the bool.


def test_future_warning_message_names_replacement_and_removal_release():
    """FutureWarning message must give the user enough to migrate
    without grepping the source."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        class _Legacy(SimulationNode):
            @property
            def requires_halo(self) -> bool:
                return True

            def initial_state(self):
                return {"x": jnp.zeros(1)}

            def update(self, state, boundary_inputs, dt):
                return state

        msg = str([w for w in caught if issubclass(w.category, FutureWarning)][0].message)
    assert "halo_width" in msg          # the replacement API
    assert "v0.3" in msg                # the removal release
