"""The dead band of a *geometry* edge read at its source asks both of its quantities.

The rule of ``test_the_dead_band_of_an_edge_read_at_its_source.py``, on
an edge whose mapping reads a geometry: a ``multilinear_grid`` scatter
(points to grid, ``mode="conservative"``) delivers more entries than its
source field holds, so ``convergence_norm="interface"`` reads it at its
source, and with ``atol > 0`` that reading leaves the norm only where
**both** the source field and what the edge delivers (the mapping at the
geometry the step uses, then the transform) are at or below ``atol``
(``acceleration._kept_by_what_is_delivered``).

**The pair** (the audited construction on a geometry edge; float64)::

    p (3 markers)   p.x <- size * (b + A u)        p.pos: 3 positions, held still
    q (30 cells)    q.x <- c + 0.5 q_pre + u / (size * gain)
    p.x -> q.u   multilinear_grid, points to grid, geometry=("source", "pos"),
                 then ``transform = lambda v: v * gain``     [read at its source]
    q.x -> p.u   a static 30 -> 3 gather                     [read as delivered]

The positions put each marker 0.4 of a spacing into its cell, so the
stencil is the static pair's ``H`` (weights 0.6 and 0.4) and the loop is
the same for every ``size`` and ``gain``.

**The reference** is numerical (``tests/property/coupling_reference.py``):
the one-pass map of a float64 twin of the same graph, and the fixed
point of that map by Newton from the returned iterate.  Nothing of the
criterion is in it.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property import coupling_reference as cr

M, N = 3, 30
KEY = "p+q"
ATOL = 1e-6
RTOL = 1e-6
CELLS = np.array([4, 13, 22])
POSITIONS = (CELLS + 0.4).reshape(M, 1)
G = np.zeros((M, N))
G[np.arange(M), CELLS] = 0.6
G[np.arange(M), CELLS + 1] = 0.4
_A = np.random.default_rng(5).normal(size=(M, M))
A = _A * (0.7 / np.max(np.abs(np.linalg.eigvals(_A @ G @ G.T))))
B = np.array([1.0, 1.5, 2.0])
C = np.linspace(0.5, 1.0, N)
Q0 = np.linspace(1.0, 3.0, N)

#: ``(size, gain)`` by quadrant of the band at ``atol = 1e-6``: is the
#: source field above it, is what the edge delivers above it.  In all
#: three the reading is kept.
KEPT = {
    "field inside, delivered above": (1e-9, 1e9),      # the audited case
    "field above, delivered inside": (1.0, 1e-9),      # the reverse
    "field above, delivered above": (1.0, 1.0),
}
#: What a converged step is held to, in tolerances of each field's own
#: magnitude (the static pair's promise: it measured 0.9 to 7.5 with no
#: dead band).
PROMISE = 25.0


class Markers(SimulationNode):
    """``x <- size * (b + A u)``; ``pos`` is returned as it was."""

    def __init__(self, name, timestep, size):
        super().__init__(name, timestep)
        self._size = size

    def initial_state(self):
        return {"x": jnp.asarray(self._size * B), "pos": jnp.asarray(POSITIONS)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(M,), dtype=jnp.float64, default=jnp.zeros(M))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._size * (jnp.asarray(B) + jnp.asarray(A) @ boundary_inputs["u"]),
                "pos": state["pos"]}

    def update_evaluations(self):
        return 1


class Grid(SimulationNode):
    """``x <- c + 0.5 x_pre + scale * u``."""

    def __init__(self, name, timestep, scale):
        super().__init__(name, timestep)
        self._scale = scale

    def initial_state(self):
        return {"x": jnp.asarray(Q0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=jnp.float64, default=jnp.zeros(N))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.asarray(C) + 0.5 * state["x"] + self._scale * boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


def build(size, gain, **knobs) -> GraphManager:
    """The pair under *knobs* (run inside ``cr.x64()``)."""
    gm = GraphManager()
    gm.add_node(Markers("p", 1.0, size))
    gm.add_node(Grid("q", 1.0, 1.0 / (size * gain)))
    gm.add_edge("p", "q", "x", "u",
                mapping=multilinear_grid_mapping([0.0], [1.0], [N], n_points=M,
                                                 mode="conservative"),
                geometry=("source", "pos"), transform=lambda v: v * gain)
    gm.add_edge("q", "p", "x", "u", mapping=matrix_mapping(jnp.asarray(G)))
    gm.add_coupling_group(["p", "q"], convergence_norm="interface", solver="ift", **knobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _members(gm) -> dict:
    return {name: {f: np.asarray(v) for f, v in gm.get_node_state(name).items()}
            for name in ("p", "q")}


def held_steps(size, gain, schedule, steps=4) -> list:
    """Step the pair under the dead band and, after every step, find the
    fixed point of the twin's pass from the returned iterate.  One row
    per step: the report, and each value field's largest error over its
    own magnitude in tolerances."""
    rows = []
    with cr.x64():
        knobs = {"iteration_mode": schedule}
        gm = build(size, gain, rtol=RTOL, atol=ATOL, max_iterations=400, **knobs)
        twin = build(size, gain, **cr.twin_knobs(knobs))
        reference = cr.PassReference.of(twin)
        for _ in range(steps):
            pre = _members(gm)
            gm.step()
            report = dict(gm.coupling_diagnostics()[KEY])
            returned = _members(gm)
            for name, fields in pre.items():
                twin.set_node_state(name, {f: jnp.asarray(v) for f, v in fields.items()})
            bound = reference.at(twin._state, twin.params)      # noqa: SLF001
            x = bound.flat(returned)
            fixed = bound.fixed_point(x)
            assert fixed.converged, (fixed.ulps, fixed.history)
            errors = {}
            for node, field in (("p", "x"), ("q", "x")):
                star = np.asarray(bound.field(fixed.x, node, field))
                got = np.asarray(bound.field(x, node, field))
                errors[node] = float(np.max(np.abs(got - star)) / np.max(np.abs(star)) / RTOL)
            # The quadrant is the one its name says, at the fixed point.
            delivered = gain * float(np.max(np.abs(G.T @ np.asarray(
                bound.field(fixed.x, "p", "x")))))
            rows.append({"report": report, "errors": errors,
                         "field": float(np.max(np.abs(bound.field(fixed.x, "p", "x")))),
                         "delivered": delivered})
    return rows


def _holds(quadrant, schedule) -> None:
    size, gain = KEPT[quadrant]
    rows = held_steps(size, gain, schedule)
    for row in rows:
        assert (row["field"] > ATOL) == quadrant.startswith("field above"), row
        assert (row["delivered"] > ATOL) == quadrant.endswith("delivered above"), row
        assert row["report"]["converged"], row
        assert max(row["errors"].values()) < PROMISE, row
    # A kept reading is solved for: no step is accepted after a pass or two.
    assert min(int(row["report"]["iterations"]) for row in rows) > 10, rows


@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_small_force_spread_onto_a_grid_in_other_units_is_held_to_the_criterion(schedule):
    """The audited case on a geometry edge: forces of about 1e-8 that the
    multilinear scatter and its transform hand on as tens to hundreds,
    under a dead band of 1e-6.  Every converged step leaves both fields
    within the promise of the reference's fixed point, on both schedules."""
    _holds("field inside, delivered above", schedule)


# Per push: tests/core/test_the_dead_band_of_a_geometry_edge_read_at_its_source.py::test_a_small_force_spread_onto_a_grid_in_other_units_is_held_to_the_criterion
@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
@pytest.mark.parametrize("quadrant", ["field above, delivered inside",
                                      "field above, delivered above"])
def test_a_geometry_edge_read_at_its_source_is_kept_where_either_quantity_is_above_the_band(
        quadrant, schedule):
    """The reverse (the field above the band, what it delivers inside
    it) and the plain case: kept, and held to the same promise."""
    _holds(quadrant, schedule)
