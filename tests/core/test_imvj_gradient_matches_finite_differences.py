"""The gradient through a multi-physics IQN-IMVJ group agrees with finite differences.

``tests/core/test_coupling_multiphysics_imvj.py::test_gradient_through_multiphysics_imvj_matches_fd``
checks this through a 20-step scan and is slow-marked; the forward of the
same group is checked on every push there, and IMVJ's gradient is compared
with other solvers' gradients (not with finite differences) in
``test_coupling_ift_iqn_imvj.py``.  This is the finite-difference check on
every push: the same heat-rod/spring fixed point (a stencil and an ODE,
not one operator twice), three steps of a scan so that IMVJ's V/W history
is carried from step to step, and a larger timestep so that three steps
move the spring enough for a float32 central difference to resolve.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

DT = 5e-3          # Fourier number 0.25 on the 12-cell rod: inside the stencil's limit
STEPS = 3


@register_transform("test_imvj_fd_heat_to_anchor")
def _heat_to_anchor(flux):
    return 0.02 * flux


def _graph() -> GraphManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gm = GraphManager()
        gm.add_node(HeatNode("rod", DT, n_cells=12, thermal_diffusivity=0.5,
                             initial_temperature=300.0))
        gm.add_node(SpringDamperNode("spring", DT, stiffness=40.0, damping=3.0, mass=1.0,
                                     rest_length=1.0, initial_position=0.5))
        # The spring sets the rod's right-end temperature; the rod's left
        # boundary flux moves the spring's anchor.
        gm.add_edge("spring", "rod", "position", "right_temperature",
                    transform=lambda x: 300.0 + 20.0 * x)
        gm.add_edge("rod", "spring", "left_heat_flux", "anchor_position",
                    transform="test_imvj_fd_heat_to_anchor")
        gm.add_coupling_group(["rod", "spring"], max_iterations=25, tolerance=1e-7,
                              acceleration="iqn-imvj", jacobian_reuse=3, solver="ift")
        gm.compile()
    return gm


def test_the_gradient_through_a_multiphysics_imvj_group_matches_finite_differences():
    gm = _graph()
    step = gm._build_step_fn()               # noqa: SLF001
    ext = gm._default_external_inputs()      # noqa: SLF001

    def loss(k):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["spring"]["stiffness"] = k
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state, None,  # noqa: SLF001
                                length=STEPS)
        return jnp.sum(final["rod"]["temperature"]) + 100.0 * final["spring"]["position"]

    # One program for the gradient and the finite-difference values: the
    # cost here is tracing the IFT solve and its adjoint (about 2.5 s on
    # three cores, which no compilation cache removes), so it is paid once.
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    k0 = jnp.asarray(40.0, jnp.float32)
    h = 2.0
    g = float(value_and_grad(k0)[1])
    fd = (float(value_and_grad(k0 + h)[0]) - float(value_and_grad(k0 - h)[0])) / (2 * h)
    assert np.isfinite(g) and g != 0.0
    # The slow test's tolerance; measured here: 0.3% apart.
    assert abs(g - fd) <= 5e-2 * abs(fd) + 1e-3, (g, fd)
