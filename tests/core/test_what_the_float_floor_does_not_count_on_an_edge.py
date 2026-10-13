"""Two things on an internal edge that the float floor of a report does not count.

The float floor of a coupling report (and with it ``precision_limited``
and the two usable flags) counts the coarsest floating dtype the group's
pass goes through, what each internal edge delivers included (CPL-100),
and withdraws the flags behind a static mapping whose rows add up many
entries (MADD-ANO-257).  Two idioms are outside both counts, stated in
the user guide beside each rule, and open:

* **an internal edge whose source is a boundary flux** (a key of
  ``compute_boundary_fluxes``, not a state field): what it delivers is
  not read, so a transform on it that narrows the dtype is not counted
  (MADD-ANO-259).  The same value carried as a state field is counted;
* **a sum written inside an edge transform** (a scatter-add of many
  terms into one entry): the row guard reads static mappings only
  (MADD-ANO-260).  The same operator declared as a static sparse mapping
  is counted, and the flags are withdrawn.

Each is pinned here as what happens today (a usable flag beside a bound
far below the distance to the fixed point), next to the way out the
documents give, which holds.  Neither is a wish: when one of these pins
stops being true its registry entry and the guide's sentence are to be
rewritten.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.sparse_mapping import sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64


def _distance(fields, rtol: float) -> float:
    """The ``"mixed"`` norm of the distance to the fixed point, in
    tolerances: *fields* is ``[(returned, fixed point), ...]``, each
    entry's miss over its field's own magnitude, pooled."""
    total, count = 0.0, 0
    for x, fixed in fields:
        x, fixed = np.asarray(x, np.float64).ravel(), np.asarray(fixed, np.float64).ravel()
        total += float(np.sum(((x - fixed) / np.max(np.abs(x))) ** 2))
        count += x.size
    return math.sqrt(total / count) / rtol


def _group(gm: GraphManager, members, rtol: float, schedule: str, norm: str = "mixed") -> None:
    gm.add_coupling_group(list(members), convergence_norm=norm, rtol=rtol, atol=0.0,
                          iteration_mode=schedule, max_iterations=3000, solver="ift",
                          diagnostics=True, acceleration="none")


# ---------------------------------------------------------------------------
# A narrowing transform on an edge that carries a boundary flux
# ---------------------------------------------------------------------------

N = 3
A_GAIN, B_GAIN = 0.9, 0.8
A_LOAD = np.array([1.0, 1.3, 0.7])
B_LOAD = np.array([0.3, 0.1, 0.2])
NARROW_RTOL = 1e-12


class Affine(SimulationNode):
    """``x <- load + gain * u`` in float64; ``u`` is declared in
    *in_dtype*.  With *offers_flux* the value is also handed out as the
    boundary flux ``f``."""

    def __init__(self, name: str, gain: float, load, in_dtype: str, offers_flux: bool = False):
        super().__init__(name, 1.0)
        self._gain, self._load, self._in = gain, np.asarray(load), jnp.dtype(in_dtype)
        self._offers_flux = offers_flux

    def initial_state(self):
        return {"x": jnp.zeros(N, jnp.float64)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=self._in, default=jnp.zeros(N, self._in))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.asarray(self._load, jnp.float64)
                + self._gain * boundary_inputs["u"].astype(jnp.float64)}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"f": state["x"]} if self._offers_flux else {}

    def update_evaluations(self):
        return 1


def _narrowing_pair(source: str, schedule: str, norm: str = "mixed") -> GraphManager:
    """Two float64 members; the edge A -> B narrows to float32, its
    source A's state field ``x`` or the same value as the flux ``f``."""
    gm = GraphManager()
    gm.add_node(Affine("A", A_GAIN, A_LOAD, "float64", offers_flux=source == "flux"))
    gm.add_node(Affine("B", B_GAIN, B_LOAD, "float32"))
    gm.add_edge("A", "B", "f" if source == "flux" else "x", "u",
                transform=lambda v: v.astype(jnp.float32))
    gm.add_edge("B", "A", "x", "u")
    _group(gm, ["A", "B"], NARROW_RTOL, schedule, norm)
    return gm


def _narrowing_report(source: str, schedule: str) -> tuple:
    """``(report, distance in tolerances)`` after four steps, in x64."""
    with x64(True):
        gm = _narrowing_pair(source, schedule)
        for _ in range(4):
            gm.step()
        (report,) = (dict(r) for r in gm.coupling_diagnostics().values())
        fixed_a = (A_LOAD + A_GAIN * B_LOAD) / (1.0 - A_GAIN * B_GAIN)
        fixed_b = B_LOAD + B_GAIN * fixed_a
        distance = _distance([(gm.get_node_state("A")["x"], fixed_a),
                              (gm.get_node_state("B")["x"], fixed_b)], NARROW_RTOL)
    return report, distance


@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
def test_a_narrowing_transform_on_a_flux_edge_is_not_in_the_float_floor(schedule):
    """MADD-ANO-259 as it stands: the pair stalls at float32's rounding
    1e5 tolerances from its fixed point, and the report says converged
    and ``spectral_usable`` beside a bound a millionth of the distance
    (measured 5.6e-8 under Jacobi and 7.5e-8 under Gauss-Seidel, on jax
    0.10.2, 0.11.0 and 0.11.2)."""
    report, distance = _narrowing_report("flux", schedule)
    assert distance > 1e4, distance
    assert report["converged"] and report["spectral_usable"], report
    assert report["spectral_error_bound"] < 1e-3 * distance, (report, distance)


@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
def test_the_same_value_carried_as_a_state_field_is_counted(schedule):
    """What the guide says to do instead: with the narrowed value a state
    field the floor is float32's, and the bound is above the distance
    (measured 29.8 and 38.8 times it)."""
    report, distance = _narrowing_report("state", schedule)
    assert distance > 1e4, distance
    assert report["spectral_usable"], report
    assert report["spectral_error_bound"] >= distance, (report, distance)


def test_the_interface_norm_refuses_the_flux_edge_at_compile():
    """Under ``"interface"`` the norm reads what the edges carry, and a
    flux edge is refused before any report exists."""
    with x64(True):
        gm = _narrowing_pair("flux", "gauss-seidel", norm="interface")
        with pytest.raises(RuntimeError, match="boundary flux"):
            gm.compile()


# ---------------------------------------------------------------------------
# A sum of many terms written inside an edge transform
# ---------------------------------------------------------------------------

CELLS = 4
#: The terms added into one entry.  300 keeps the pins quick; measured
#: behind 3000 the transform's bound is 0.034 of the distance and the
#: mapping's 15 times it (the same on jax 0.10.2, 0.11.0 and 0.11.2).
MARKERS = 300
LOOP_GAIN = 0.9
SUM_RTOL = 1e-7
G_LOAD, P_LOAD = 0.1, 1.0
#: The grid's response to the sum it reads: the loop gain over the sum's
#: weight (each marker hands half its value to grid entries 1 and 2).
G_GAIN = LOOP_GAIN / (0.5 * MARKERS)


class Grid(SimulationNode):
    """``x <- G_LOAD + G_GAIN * u`` on the four grid entries, float32."""

    def initial_state(self):
        return {"x": jnp.zeros(CELLS, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(CELLS,), dtype=jnp.dtype("float32"),
                                       default=jnp.zeros(CELLS, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.float32(G_LOAD) + jnp.float32(G_GAIN) * boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


class Markers(SimulationNode):
    """``x[i] <- P_LOAD + u[1]``: every marker reads grid entry 1."""

    def initial_state(self):
        return {"x": jnp.zeros(MARKERS, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(CELLS,), dtype=jnp.dtype("float32"),
                                       default=jnp.zeros(CELLS, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.float32(P_LOAD) + jnp.broadcast_to(boundary_inputs["u"][1], (MARKERS,))}

    def update_evaluations(self):
        return 1


def _summing_report(how: str) -> tuple:
    """``(report, distance in tolerances)`` after three steps of the pair
    whose edge markers -> grid adds half of every marker into grid
    entries 1 and 2: *how* is ``"transform"`` (a scatter-add written in
    the edge's transform) or ``"mapping"`` (the same operator as a static
    sparse mapping)."""
    gm = GraphManager()
    gm.add_node(Grid("G", 1.0))
    gm.add_node(Markers("P", 1.0))
    gm.add_edge("G", "P", "x", "u")
    if how == "transform":
        cells = jnp.asarray(np.concatenate([np.ones(MARKERS, np.int32),
                                            np.full(MARKERS, 2, np.int32)]))
        gm.add_edge("P", "G", "x", "u",
                    transform=lambda v: jnp.zeros(CELLS, v.dtype).at[cells].add(
                        jnp.concatenate([0.5 * v, 0.5 * v])))
    else:
        index = np.full((CELLS, MARKERS), -1, np.int64)
        weight = np.zeros((CELLS, MARKERS), np.float32)
        index[1] = index[2] = np.arange(MARKERS)
        weight[1] = weight[2] = 0.5
        gm.add_edge("P", "G", "x", "u",
                    mapping=sparse_matrix_mapping(index, weight, n_source=MARKERS))
    _group(gm, ["G", "P"], SUM_RTOL, "gauss-seidel")
    for _ in range(3):
        gm.step()
    (report,) = (dict(r) for r in gm.coupling_diagnostics().values())
    g_gain, g_load, p_load = (float(np.float32(v)) for v in (G_GAIN, G_LOAD, P_LOAD))
    g = (g_load + g_gain * 0.5 * MARKERS * p_load) / (1.0 - g_gain * 0.5 * MARKERS)
    distance = _distance([(gm.get_node_state("G")["x"], np.array([g_load, g, g, g_load])),
                          (gm.get_node_state("P")["x"], np.full(MARKERS, p_load + g))], SUM_RTOL)
    return report, distance


def test_a_sum_written_inside_an_edge_transform_is_not_counted_by_the_row_guard():
    """MADD-ANO-260 as it stands: 300 float32 terms added into one entry
    by the edge's transform, the residual at its float floor, and
    ``spectral_usable`` is set beside a bound under the distance
    (measured 0.283 of it on jax 0.10.2, 0.11.0 and 0.11.2)."""
    report, distance = _summing_report("transform")
    assert report["converged"] and report["precision_limited"], report
    assert report["spectral_usable"], report
    assert report["spectral_error_bound"] < 0.5 * distance, (report, distance)


def test_the_same_sum_declared_as_a_static_sparse_mapping_withdraws_the_flags():
    """What the guide says to do instead: as a mapping the row of 300
    entries is counted, both flags are withdrawn, and the bound is above
    the distance (measured 3.8 times it)."""
    report, distance = _summing_report("mapping")
    assert not report["spectral_usable"] and not report["gradient_bound_usable"], report
    assert report["spectral_error_bound"] >= distance, (report, distance)
