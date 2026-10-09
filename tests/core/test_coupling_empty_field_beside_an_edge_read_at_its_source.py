"""A field with no entries beside a mapped edge the interface norm reads at
its source: the group steps, and returns and reports what its twin without
the field does.

Three rules meet on one group here, and each was tested without the others:

* ``convergence_norm="interface"`` reads an internal edge whose static
  mapping delivers more entries than its source holds **at its source**
  (``_interface_plan._norm_side``;
  ``test_the_interface_norm_reads_a_mapped_edge_on_its_compact_side.py``);
* **a field with no entries is not read** by any norm, weight, floor or
  gain (``acceleration._has_entries``;
  ``test_coupling_diagnostics_with_an_empty_field.py``, whose mapped
  shapes carry a tie, read as delivered);
* a solve returns the fields the norm measures whole as the accepted
  iterate holds them and the others one plain pass on
  (``test_coupling_interface_norm_answers_for_the_state_it_returns.py``).

**The graphs.**  Member ``a`` holds two entries and member ``b`` five; the
edge ``a.x -> b.u`` scatters two onto five through a dense static mapping
and is read at its source.  The edge back is a plain one (``plain``: every
reading is a field as it is, so the report's analysis is taken in the
state's weights) or a mapping of five onto two, read as delivered
(``gather``: the analysis is taken on the reading).  The field with no
entries sits on ``a``, the member the scattered edge reads:

* ``unread``: no edge reads it;
* ``read``: a plain edge reads it, and so delivers no entries;
* ``transformed``: an edge reads it through a transform.  With the plain
  edge back this is the cell both rules keep in the state's weights: the
  scattered edge's reading is its field as it is only because it is read
  at its source, and the transformed edge does not make the reading
  another norm only because it reads no entries (``entry_fields``);
* ``scattered``: an edge reads it through a static mapping of none onto
  three.  More entries are delivered than the source holds, so the edge
  is read at its source, which has no entries: it is not read.  (Read as
  delivered it would be three zeros, a reading with entries.)

**What is asserted.**  The premises, on the library's own enumeration
(``_interface_readings``): the scattered edge is read at two entries, the
edge that carries the field with no entries is not read, and the graph's
readings are its twin's.  Then the state after two steps, every number of
``coupling_diagnostics()`` and (slow cells) the gradient through a step
equal the twin's **to the bit**, and the field is handed back with no
entries.  To the bit although the two programs are not one text: they
differ by operands with no entries, and in ``scattered`` by a sum of three
delivered zeros added to ``b.x``.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import _interface_readings
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import empty_field_graphs as eg

F32 = jnp.float32
N_A, N_B, N_W = 2, 5, 3

#: Two entries onto five (rows sum to one), and five onto two.
_SCATTER = np.asarray([[1.0, 0.0], [0.7, 0.3], [0.5, 0.5], [0.2, 0.8], [0.0, 1.0]], np.float32)
_GATHER = np.asarray([[0.5, 0.3, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 0.4, 0.5]], np.float32)

BACK = ("plain", "gather")
PLACES = ("unread", "read", "transformed", "scattered")


def _double(v):
    return 2.0 * v


class _Member(SimulationNode):
    """``x <- x_pre / 2 + u / 4 + u**2 / 50 + c`` on ``n`` entries.

    The port ``u`` has ``n_in`` entries; where that is not ``n`` (member
    ``a`` behind the plain edge) it is folded onto ``n`` by ``_GATHER``
    inside the node.  ``holds`` adds a field ``none`` with no entries;
    ``port`` names a second input (``(name, entries)``) whose sum is added
    to ``x``.
    """

    def __init__(self, name, *, n, n_in, c, holds=False, port=None):
        super().__init__(name, 1.0)
        self._n, self._n_in, self._c = int(n), int(n_in), float(c)
        self._holds, self._port = bool(holds), port

    def initial_state(self):
        state = {"x": jnp.linspace(0.5, -0.25, self._n).astype(F32)}
        if self._holds:
            state["none"] = jnp.zeros((0,), F32)
        return state

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(self._n_in,), dtype=F32,
                                       default=jnp.zeros((self._n_in,), F32))}
        if self._port is not None:
            name, entries = self._port
            spec[name] = BoundaryInputSpec(shape=(entries,), dtype=F32,
                                           default=jnp.zeros((entries,), F32))
        return spec

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"].astype(F32)
        if self._n_in != self._n:
            u = jnp.asarray(_GATHER) @ u
        x = (jnp.asarray(0.5, F32) * state["x"] + jnp.asarray(0.25, F32) * u
             + jnp.asarray(0.02, F32) * u * u + jnp.asarray(self._c, F32))
        if self._port is not None:
            x = x + jnp.sum(boundary_inputs[self._port[0]]).astype(F32)
        out = {"x": x.astype(F32)}
        if self._holds:
            out["none"] = state["none"]
        return out

    def update_evaluations(self):
        return 1


def _build(back: str, place, mode: str) -> GraphManager:
    """The compiled graph; ``place=None`` is the twin, without the field,
    the port that reads it and the edge that carries it."""
    port = {"read": ("e", 0), "transformed": ("e", 0), "scattered": ("w", N_W)}.get(place)
    gm = GraphManager()
    gm.add_node(_Member("a", n=N_A, n_in=N_B if back == "plain" else N_A, c=1.0,
                        holds=place is not None))
    gm.add_node(_Member("b", n=N_B, n_in=N_B, c=0.4, port=port))
    gm.add_edge("a", "b", "x", "u", mapping=matrix_mapping(jnp.asarray(_SCATTER)))
    if back == "plain":
        gm.add_edge("b", "a", "x", "u")
    else:
        gm.add_edge("b", "a", "x", "u", mapping=matrix_mapping(jnp.asarray(_GATHER)))
    if place in ("read", "transformed"):
        gm.add_edge("a", "b", "none", "e",
                    transform=_double if place == "transformed" else None)
    elif place == "scattered":
        gm.add_edge("a", "b", "none", "w",
                    mapping=matrix_mapping(jnp.zeros((N_W, 0), F32)))
    gm.add_coupling_group(["a", "b"], convergence_norm="interface", solver="ift",
                          diagnostics=True, iteration_mode=mode, max_iterations=40,
                          rtol=1e-6, tolerance=1e-6)
    gm.compile()
    return gm


def _readings(gm) -> list:
    """``(edge, entries read)`` for every edge the interface norm reads."""
    state = gm._state  # noqa: SLF001
    return [(edge.key, int(np.size(value)))
            for edge, _dtype, value in _interface_readings(gm._edges, state)]  # noqa: SLF001


@functools.lru_cache(maxsize=None)
def _run(back, place, mode, gradient):
    """``(readings, outcome, the field was handed back)`` of one graph,
    compiled and stepped once per session (the cells share twins)."""
    gm = _build(back, place, mode)
    readings = _readings(gm)
    out = eg.outcome(gm, with_gradient=gradient)
    kept = eg.none_fields_are_kept(gm, "float32") if place is not None else None
    return readings, out, kept


def _assert_the_twin_converges_and_reports_numbers(back, mode, gradient):
    """The comparison is not between two failures (two outcomes that are
    both empty, or both NaN, would also agree), and the twin's scattered
    edge is read at its source."""
    readings, twin, _ = _run(back, None, mode, gradient)
    assert readings == [("a.x->b.u", N_A), ("b.x->a.u", N_B if back == "plain" else N_A)], readings
    report = twin["report"]
    assert report["converged"] is True, report
    assert report["spectral_usable"] is True, report
    assert math.isfinite(report["spectral_error_bound"]), report
    assert all(np.all(np.isfinite(v)) for fields in twin["state"].values()
               for v in fields.values())


def _assert_reports_what_its_twin_does(back, place, mode, gradient):
    _assert_the_twin_converges_and_reports_numbers(back, mode, gradient)
    readings, got, kept = _run(back, place, mode, gradient)
    twin_readings, twin, _ = _run(back, None, mode, gradient)

    # The premises: one edge read at its source, the other as delivered,
    # and nothing read of the field with no entries.
    assert readings == [("a.x->b.u", N_A), ("b.x->a.u", N_B if back == "plain" else N_A)], (
        "the scattered edge is read at its two source entries and the edge "
        f"that carries no entries is not read: {readings}")
    assert readings == twin_readings

    assert kept, "the field with no entries was not handed back as it was"
    assert eg.differences(got, twin) == []
    if gradient:
        assert np.all(np.isfinite(got["gradient"])) and np.any(got["gradient"] != 0)


#: Every push: the analysis in the state's weights, which both rules must
#: hold for this group to keep (see ``transformed`` above), and the
#: analysis on the reading with the field behind a mapping of none onto
#: three (the edge both rules decide).
PER_PUSH = [("plain", "transformed", "gauss-seidel"), ("gather", "scattered", "jacobi")]


@pytest.mark.parametrize("back, place, mode", PER_PUSH,
                         ids=["-".join(cell) for cell in PER_PUSH])
def test_the_twin_without_the_field_converges_and_reports_numbers(back, place, mode):
    """Compiles the twin the comparison below reads (one graph a test)."""
    _assert_the_twin_converges_and_reports_numbers(back, mode, False)


@pytest.mark.parametrize("back, place, mode", PER_PUSH,
                         ids=["-".join(cell) for cell in PER_PUSH])
def test_a_group_with_an_edge_read_at_its_source_reports_what_its_twin_without_the_field_does(
        back, place, mode):
    _assert_reports_what_its_twin_does(back, place, mode, False)


# Per push: tests/core/test_coupling_empty_field_beside_an_edge_read_at_its_source.py::test_a_group_with_an_edge_read_at_its_source_reports_what_its_twin_without_the_field_does[plain-transformed-gauss-seidel]
@pytest.mark.slow
@pytest.mark.parametrize("mode", ("gauss-seidel", "jacobi"))
@pytest.mark.parametrize("place", PLACES)
@pytest.mark.parametrize("back", BACK)
def test_every_place_of_the_field_reports_what_its_twin_does_with_the_gradient(back, place, mode):
    _assert_reports_what_its_twin_does(back, place, mode, True)
