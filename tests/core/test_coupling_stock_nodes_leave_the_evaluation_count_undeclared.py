"""No node in ``maddening.nodes`` declares ``update_evaluations()`` in 0.4.0, and why.

A group whose residual is at its float floor reports
``spectral_usable=False`` unless every node declares its evaluation
count, because there the floor is the whole bound.  A spring's update
is a single explicit step, so declaring ``1`` on it would change no
number -- an undeclared node already counts as one evaluation -- and
would only let the flag be set at the floor.

It is not declared, because the first group it was tried on then read a
wrong radius with its flag set.  Two ``SpringDamperNode`` s anchored on
each other (``k = 6000``, ``c = 90``, ``dt = 0.01``, float32,
Gauss-Seidel, the default tolerance) settle to rest, and their
velocities fall under a hundredth of the positions that drive them: the
regime of MADD-ANO-230, where a direction the Arnoldi breakdown test
takes for rounding can carry the dominant mode.  With the two nodes
declaring one evaluation, 2 of 1 376 steps over six configurations read
settled outside the margin (``rho_spectral`` 0.462 for 0.36 on the sixth
step; CPU, jaxlib 0.11.0); undeclared, none of them reads usable at the
floor.  The declaration waits for that entry.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

import maddening.nodes as stock
from maddening import GraphManager
from maddening.core.coupling.acceleration import SPECTRAL_SETTLED_FRACTION
from maddening.core.node import SimulationNode
from maddening.nodes import SpringDamperNode

STOCK = sorted(
    (cls for _, cls in inspect.getmembers(stock, inspect.isclass)
     if issubclass(cls, SimulationNode) and cls is not SimulationNode),
    key=lambda cls: cls.__name__)


def test_the_stock_nodes_are_found():
    assert len(STOCK) >= 8 and SpringDamperNode in STOCK, STOCK


@pytest.mark.parametrize("cls", STOCK, ids=lambda cls: cls.__name__)
def test_a_stock_node_does_not_declare_its_evaluation_count(cls):
    """Declaring it is a decision about MADD-ANO-230 (see the module docstring), not a tidy-up."""
    assert cls.update_evaluations is SimulationNode.update_evaluations, cls


def test_a_settling_stock_spring_pair_is_never_usable_with_a_radius_outside_the_margin():
    """The pair's Gauss-Seidel radius is 0.36 whatever the state: ``(k dt**2 / m)**2``."""
    k, dt, exact = 6000.0, 0.01, 0.36
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", dt, stiffness=k, damping=90.0, mass=1.0,
                                 rest_length=-1.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("b", dt, stiffness=k, damping=90.0, mass=1.0,
                                 rest_length=1.0, initial_position=1.2))
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_edge("a", "b", "position", "anchor_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["a", "b"], iteration_mode="gauss-seidel", tolerance=1e-6,
                              max_iterations=60, diagnostics=True)
        gm.compile()
    at_the_floor = 0
    for _ in range(12):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.step()
            d = gm.coupling_diagnostics()["a+b"]
        if d["precision_limited"]:
            at_the_floor += 1
            assert d["spectral_usable"] is False, dict(d)
        if d["spectral_usable"]:
            margin = SPECTRAL_SETTLED_FRACTION * (1.0 - d["rho_spectral"])
            assert abs(d["rho_spectral"] - exact) <= margin, dict(d)
    # The premise: at the default tolerance this float32 pair is at its floor.
    assert at_the_floor >= 6, at_the_floor
