"""Graphs of every kind the inspection methods must handle, built once each.

``graph(kind)`` builds and caches; the cache is safe to share across tests
*only because* the inspection methods are read-only, which is exactly
what ``test_inspection_read_only.py`` proves for every kind here.  A test
that needs to change a graph builds its own with ``build(kind)``.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


def _spring(name: str, dt: float = 0.01, x0: float = 1.5) -> SpringDamperNode:
    return SpringDamperNode(name, dt, stiffness=30.0, damping=2.0, initial_position=x0)


def _pair(**group) -> GraphManager:
    gm = GraphManager()
    gm.add_node(_spring("a", x0=1.5))
    gm.add_node(_spring("b", x0=0.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], **{"max_iterations": 8, "tolerance": 1e-6, **group})
    return gm


def _single_uncompiled() -> GraphManager:
    gm = GraphManager()
    gm.add_node(_spring("s"))
    return gm


def _single() -> GraphManager:
    gm = _single_uncompiled()
    gm.compile()
    gm.step()
    return gm


def _stale() -> GraphManager:
    gm = _single()
    gm.add_node(_spring("t", x0=0.5))          # dirty: modified since the compile
    return gm


def _coupled() -> GraphManager:
    gm = _pair()                               # solver="ift", diagnostics off
    gm.compile()
    gm.step()
    return gm


def _coupled_diagnostics() -> GraphManager:
    gm = _pair(diagnostics=True)
    gm.compile()
    gm.step()
    return gm


def _coupled_fori() -> GraphManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        gm = _pair(solver="fori")              # diagnostics off: no report at all
    gm.compile()
    gm.step()
    return gm


def _coupled_capped() -> GraphManager:
    gm = _pair(max_iterations=1)               # always at the cap; ratio unusable
    gm.compile()
    gm.step()
    return gm


def _multirate() -> GraphManager:
    gm = GraphManager()
    gm.add_node(_spring("fast", dt=0.01))
    gm.add_node(_spring("slow", dt=0.02, x0=0.0))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.compile()
    gm.run_scan(4)
    return gm


def _subcycled() -> GraphManager:
    gm = GraphManager()
    gm.add_node(_spring("coarse", dt=0.02))
    gm.add_node(_spring("fine", dt=0.01, x0=0.0))
    gm.add_edge("coarse", "fine", "position", "anchor_position")
    gm.add_edge("fine", "coarse", "position", "anchor_position")
    gm.add_coupling_group(["coarse", "fine"], max_iterations=8, tolerance=1e-6,
                          subcycling=True)
    gm.compile()
    gm.step()
    return gm


def _mapped() -> GraphManager:
    """A mapping, a registered and an anonymous transform, an additive edge,
    a flux edge and an external input."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod_a", 1e-4, n_cells=6, thermal_diffusivity=0.1,
                         initial_temperature=300.0))
    gm.add_node(HeatNode("rod_b", 1e-4, n_cells=12, thermal_diffusivity=0.1,
                         initial_temperature=350.0))
    H = np.zeros((12, 6), np.float32)
    H[np.arange(12), np.arange(12) // 2] = 1.0
    gm.add_edge("rod_a", "rod_b", "temperature", "heat_source",
                mapping=matrix_mapping(H), additive=True)
    gm.add_edge("rod_a", "rod_b", "right_heat_flux", "left_temperature", transform="negate")
    gm.add_edge("rod_b", "rod_a", "temperature", "heat_source",
                transform=lambda x: 1e-3 * x[::2])
    gm.add_external_input("rod_a", "left_temperature")
    gm.compile()
    gm.step()
    return gm


def _after_run_scan() -> GraphManager:
    gm = _pair()
    gm.compile()
    gm.run_scan(5)
    return gm


def _after_grad() -> GraphManager:
    """Left holding the tracers of ``jax.grad`` through ``run_scan``."""
    gm = _pair()
    gm.compile()
    gm.step()

    def loss(p):
        return jnp.sum(gm.run_scan(3, params=p)["a"]["position"] ** 2)

    jax.grad(loss)(gm.params)
    assert gm._state_traced     # noqa: SLF001 - the premise of this kind
    return gm


def _empty() -> GraphManager:
    return GraphManager()


BUILDERS = {
    "empty": _empty,
    "single_uncompiled": _single_uncompiled,
    "single": _single,
    "stale": _stale,
    "coupled": _coupled,
    "coupled_diagnostics": _coupled_diagnostics,
    "coupled_fori": _coupled_fori,
    "coupled_capped": _coupled_capped,
    "multirate": _multirate,
    "subcycled": _subcycled,
    "mapped": _mapped,
    "after_run_scan": _after_run_scan,
    "after_grad": _after_grad,
}

#: Kinds that have never been compiled.
UNCOMPILED = frozenset({"empty", "single_uncompiled"})


def build(kind: str) -> GraphManager:
    """A fresh graph of this kind (for a test that changes it)."""
    return BUILDERS[kind]()


@functools.lru_cache(maxsize=None)
def graph(kind: str) -> GraphManager:
    """The shared graph of this kind (read it, never change it)."""
    return build(kind)
