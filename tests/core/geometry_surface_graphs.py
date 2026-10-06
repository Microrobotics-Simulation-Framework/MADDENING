"""A grid and a row of moving markers, joined by the shipped ``multilinear_grid`` kind.

The graph the tests of everything *around* a geometry edge share (USD,
checkpoints, the REST surrogate routes, the inspection text, the report
printers, the FMU family): small, built from plain constructor arguments
so that it round-trips through a config and a stage, and with a geometry
that moves every step, so that a surface which dropped or froze it gives
other numbers.

``grid.x`` (``N_GRID`` values on the lattice ``0, 0.5, ..``) is gathered
at the markers' positions into ``markers.sampled`` (the geometry is the
*target's* ``pos``), and ``markers.x`` is scattered back onto the grid as
``grid.deposit`` (the geometry is the *source's* ``pos``).  The markers
drift at ``speed``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

N_GRID, N_MARKERS = 6, 4
DT = 0.01
SPACING = 0.5
GATHER, SCATTER = "grid.x->markers.sampled", "markers.x->grid.deposit"
GROUP = "grid+markers"


class GridField(SimulationNode):
    """``x <- decay x + 0.1 deposit`` on ``n`` lattice points."""

    def __init__(self, name, timestep, n=N_GRID, decay=0.5):
        super().__init__(name, timestep, n=n, decay=decay)

    def initial_state(self):
        n = int(self.params["n"])
        return {"x": jnp.linspace(1.0, 2.0, n, dtype=jnp.float32)}

    def boundary_input_spec(self):
        n = int(self.params["n"])
        return {"deposit": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                             default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        deposit = boundary_inputs.get("deposit", jnp.zeros_like(state["x"]))
        return {"x": jnp.float32(self.params["decay"]) * state["x"]
                + jnp.float32(0.1) * deposit}


class Markers(SimulationNode):
    """``x <- 0.5 x + sampled`` at ``n`` points that drift at ``speed``.

    ``pos`` has shape ``(n, 1)``: the geometry of both edges.
    """

    def __init__(self, name, timestep, n=N_MARKERS, speed=0.5, start=0.3):
        super().__init__(name, timestep, n=n, speed=speed, start=start)

    def initial_state(self):
        n = int(self.params["n"])
        pos = float(self.params["start"]) + 0.4 * np.arange(n)
        return {"x": jnp.zeros(n, jnp.float32),
                "pos": jnp.asarray(pos.reshape(n, 1), jnp.float32)}

    def boundary_input_spec(self):
        n = int(self.params["n"])
        return {"sampled": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                             default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        sampled = boundary_inputs.get("sampled", jnp.zeros_like(state["x"]))
        return {"x": jnp.float32(0.5) * state["x"] + sampled,
                "pos": state["pos"] + jnp.float32(dt) * jnp.float32(self.params["speed"])}


class StillMarkers(Markers):
    """Markers that hold no ``pos``: what a replacement must not be."""

    def initial_state(self):
        return {"x": super().initial_state()["x"]}

    def update(self, state, boundary_inputs, dt):
        sampled = boundary_inputs.get("sampled", jnp.zeros_like(state["x"]))
        return {"x": jnp.float32(0.5) * state["x"] + sampled}


REGISTRY = {"GridField": GridField, "Markers": Markers}


def mappings():
    """``(gather, scatter)``: grid to markers, markers to grid."""
    kw = dict(origin=[0.0], spacing=[SPACING], shape=[N_GRID], n_points=N_MARKERS)
    return (multilinear_grid_mapping(mode="consistent", **kw),
            multilinear_grid_mapping(mode="conservative", **kw))


def static_matrices():
    """The two mappings' matrices at the markers' first positions."""
    gather, scatter = mappings()
    pos = Markers("m", DT).initial_state()["pos"]
    g = np.stack([np.asarray(gather.apply(jnp.asarray(e), None, pos))
                  for e in np.eye(N_GRID, dtype=np.float32)], axis=1)
    s = np.stack([np.asarray(scatter.apply(jnp.asarray(e), None, pos))
                  for e in np.eye(N_MARKERS, dtype=np.float32)], axis=1)
    return g, s


def graph(*, group: bool = False, geometry: bool = True, compile: bool = True,
          edges: bool = True, **group_kw) -> GraphManager:
    """The two-node graph; with ``geometry=False`` its static twin (the
    matrices of the first positions on ordinary mapped edges); with
    ``edges=False`` the two nodes alone."""
    gm = GraphManager()
    gm.add_node(GridField("grid", DT))
    gm.add_node(Markers("markers", DT))
    if not edges:
        pass
    elif geometry:
        gather, scatter = mappings()
        gm.add_edge("grid", "markers", "x", "sampled", mapping=gather,
                    geometry=("target", "pos"))
        gm.add_edge("markers", "grid", "x", "deposit", mapping=scatter,
                    geometry=("source", "pos"))
    else:
        g, s = static_matrices()
        gm.add_edge("grid", "markers", "x", "sampled", mapping=matrix_mapping(g))
        gm.add_edge("markers", "grid", "x", "deposit", mapping=matrix_mapping(s))
    if group:
        gm.add_coupling_group(["grid", "markers"],
                              **{"max_iterations": 30, "diagnostics": True, **group_kw})
    if compile:
        gm.compile()
    return gm


def states(gm: GraphManager) -> dict:
    """Every node's state as NumPy arrays."""
    return {name: {k: np.asarray(v) for k, v in gm.get_node_state(name).items()}
            for name in ("grid", "markers")}


def assert_same_states(a: GraphManager, b: GraphManager, what: str) -> None:
    sa, sb = states(a), states(b)
    for name in sa:
        assert sa[name].keys() == sb[name].keys(), (what, name)
        for field in sa[name]:
            assert np.array_equal(sa[name][field], sb[name][field]), (what, name, field)
