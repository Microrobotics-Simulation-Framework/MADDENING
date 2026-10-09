"""The dead band of an interface edge read at its source asks both of its quantities.

``atol`` is "how small is indistinguishable from zero", in the interface
quantity's own units.  The interface norm reads an edge whose static
mapping delivers more entries than its source field holds at the
*source* (the compact-side rule), and such an edge has two quantities:
the field, and what the mapping and the transform make of it.  With the
band asked of the source field alone, three forces of about 1e-8 that an
edge hands on as 17 to 520 (a unit conversion, ``transform=lambda f: f *
1e9``) left the norm at ``atol=1e-6``, and a Jacobi pair accepted after
one pass 34% to 74% from its fixed point with ``converged=True``.

The rule (``acceleration._kept_by_what_is_delivered``, the one place it
is decided): **a reading taken at the source is dropped only where both
the source field and what the edge delivers are at or below ``atol``.**
The reading, its scale and its float floor stay the source field's.

**The pair** (the audited construction, with the two scales as
parameters; every node an affine relay)::

    p (3 values)    p.x <- size * (b + A u)
    q (30 values)   q.x <- c + 0.5 q_pre + u / (size * gain)
    p.x -> q.u   a 3 -> 30 scatter (dense, or sparse in the scatter layout),
                 then ``transform = lambda v: v * gain``     [read at its source]
    q.x -> p.u   a 30 -> 3 gather                            [read as delivered]

so the loop and its fixed point are the same for every ``size`` and
``gain``, the source field is of order ``size`` and what the scatter
delivers of order ``size * gain``.  The fixed point is a closed form.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _interface_plan, acceleration
from maddening.core.coupling.acceleration import (
    _interface_readings,
    _kept_by_what_is_delivered,
    coupling_residual_interface,
    residual_precision_floor,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

M, N = 3, 30
KEY = "p+q"
ATOL = 1e-6
CELLS = np.array([4, 13, 22])
G = np.zeros((M, N))
G[np.arange(M), CELLS] = 0.6
G[np.arange(M), CELLS + 1] = 0.4
H = G.T.copy()
_A = np.random.default_rng(5).normal(size=(M, M))
A = _A * (0.7 / np.max(np.abs(np.linalg.eigvals(_A @ G @ H))))
B = np.array([1.0, 1.5, 2.0])
C = np.linspace(0.5, 1.0, N)
Q0 = np.linspace(1.0, 3.0, N)

#: ``(size, gain)`` by quadrant of the band at ``atol = 1e-6``: is the
#: source field above it, is what the edge delivers above it.
QUADRANTS = {
    "field above, delivered above": (1.0, 1.0),
    "field inside, delivered above": (1e-9, 1e9),      # the audited case
    "field above, delivered inside": (1.0, 1e-9),
    "field inside, delivered inside": (1e-9, 1.0),
}
KEPT = {name: name != "field inside, delivered inside" for name in QUADRANTS}


class Markers(SimulationNode):
    """``p.x <- size * (b + A u)``: three values, a relay.  With
    *samples_itself* it is handed the whole grid field and applies the
    gather ``G`` inside its update (``u`` is then thirty values)."""

    def __init__(self, name, timestep, size, dtype, samples_itself=False):
        super().__init__(name, timestep)
        self._size, self._dt = size, jnp.dtype(dtype)
        self._matrix, self._port = (A @ G, N) if samples_itself else (A, M)

    def initial_state(self):
        return {"x": jnp.zeros(M, self._dt)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._port,), dtype=self._dt,
                                       default=jnp.zeros(self._port, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        out = jnp.asarray(B, self._dt) + jnp.asarray(self._matrix, self._dt) @ boundary_inputs["u"]
        return {"x": jnp.asarray(self._size, self._dt) * out}

    def update_evaluations(self):
        return 1


class Grid(SimulationNode):
    """``q.x <- c + 0.5 q.x + u * back``: thirty values, carrying state."""

    def __init__(self, name, timestep, back, dtype):
        super().__init__(name, timestep)
        self._back, self._dt = back, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.asarray(Q0, self._dt)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=self._dt,
                                       default=jnp.zeros(N, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"] * jnp.asarray(self._back, self._dt)
        return {"x": jnp.asarray(C, self._dt) + 0.5 * state["x"] + u}

    def update_evaluations(self):
        return 1


def scatter(kind: str, dtype):
    """The 3 -> 30 mapping ``H``, dense or sparse in the scatter layout."""
    if kind == "dense":
        return matrix_mapping(jnp.asarray(H, dtype))
    return StaticSparseMapping(np.stack([CELLS, CELLS + 1], axis=1),
                               jnp.asarray(np.tile([0.6, 0.4], (M, 1)), dtype),
                               n_source=M, n_target=N, layout="scatter")


def edges(gain: float, kind: str, dtype) -> list:
    return [EdgeSpec("p", "q", "x", "u", mapping=scatter(kind, dtype),
                     transform=lambda v: v * gain),
            EdgeSpec("q", "p", "x", "u", mapping=matrix_mapping(jnp.asarray(G, dtype)))]


def build(size, gain, *, schedule, atol, kind="dense", dtype="float64", rtol=1e-6,
          diagnostics=False, plain_back=False) -> GraphManager:
    """The pair.  With *plain_back* the grid field returns whole through a
    plain edge (the markers sample it themselves), so the scatter is the
    group's only mapped edge."""
    gm = GraphManager()
    gm.add_node(Markers("p", 1.0, size, dtype, samples_itself=plain_back))
    gm.add_node(Grid("q", 1.0, 1.0 / (size * gain), dtype))
    for edge in edges(gain, kind, dtype):
        back = edge.source_node == "q"
        gm.add_edge(edge.source_node, edge.target_node, edge.source_field, edge.target_field,
                    mapping=None if (back and plain_back) else edge.mapping,
                    transform=edge.transform)
    gm.add_coupling_group(["p", "q"], convergence_norm="interface", rtol=rtol, atol=atol,
                          iteration_mode=schedule, max_iterations=400, solver="ift",
                          diagnostics=diagnostics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def fixed_point(size, q_pre):
    """``p = size s``, ``q = d + H s`` with ``(I - A G H) s = b + A G d``
    and ``d = c + q_pre / 2``, in float64."""
    d = C + 0.5 * np.asarray(q_pre, np.float64)
    s = np.linalg.solve(np.eye(M) - A @ G @ H, B + A @ G @ d)
    return size * s, d + H @ s


def errors_in_tolerances(gm, size, q_pre, rtol) -> tuple:
    """Each field's largest error over its own magnitude, in tolerances."""
    p_star, q_star = fixed_point(size, q_pre)
    p = np.asarray(gm.get_node_state("p")["x"], np.float64)
    q = np.asarray(gm.get_node_state("q")["x"], np.float64)
    return (float(np.max(np.abs(p - p_star)) / np.max(np.abs(p_star)) / rtol),
            float(np.max(np.abs(q - q_star)) / np.max(np.abs(q_star)) / rtol))


def iterates(size, dtype):
    """Two successive iterates of the pair near its fixed point: every
    field moving by about 1e-3 of itself."""
    p_star, q_star = fixed_point(size, Q0)
    old = {"p": {"x": jnp.asarray(p_star, dtype)}, "q": {"x": jnp.asarray(q_star, dtype)}}
    new = {"p": {"x": jnp.asarray(p_star * 1.001, dtype)},
           "q": {"x": jnp.asarray(q_star * 0.999, dtype)}}
    return new, old


def _bits(value) -> bytes:
    return np.asarray(value).tobytes()


# ---------------------------------------------------------------------------
# The decision, on the residual and on its floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["dense", "sparse-scatter"])
@pytest.mark.parametrize("quadrant", sorted(QUADRANTS))
def test_a_source_reading_leaves_the_norm_only_where_field_and_delivered_are_both_inside(
        quadrant, kind, dtype):
    """Four quadrants of the band.  Kept: the residual and the floor are,
    to the bit, those of the same pair with no dead band.  Dropped (both
    inside): they are those of the other edge alone."""
    size, gain = QUADRANTS[quadrant]
    with x64(dtype == "float64"):
        both = edges(gain, kind, dtype)
        new, old = iterates(size, dtype)
        names = ["p", "q"]
        residual = coupling_residual_interface(new, old, both, ATOL, 1e-3)
        floor = residual_precision_floor(new, names, "interface", ATOL, 1e-3, both)
        if KEPT[quadrant]:
            want = coupling_residual_interface(new, old, both, 0.0, 1e-3)
            want_floor = residual_precision_floor(new, names, "interface", 0.0, 1e-3, both)
        else:
            want = coupling_residual_interface(new, old, both[1:], 0.0, 1e-3)
            want_floor = residual_precision_floor(new, names, "interface", 0.0, 1e-3, both[1:])
        assert float(residual) > 0.1 and float(floor) > 0.0
        assert _bits(residual) == _bits(want), (float(residual), float(want))
        assert _bits(floor) == _bits(want_floor), (float(floor), float(want_floor))


def test_the_quadrants_are_the_ones_their_names_say():
    """The fixture can express the rule: at each quadrant the two
    magnitudes lie on the sides of ``atol`` its name gives."""
    for name, (size, gain) in QUADRANTS.items():
        p_star, _ = fixed_point(size, Q0)
        field, delivered = np.max(np.abs(p_star)), np.max(np.abs(gain * (H @ p_star)))
        assert (field > ATOL) == ("field above" in name), (name, field)
        assert (delivered > ATOL) == ("delivered above" in name), (name, delivered)


def test_one_function_decides_and_only_for_a_reading_taken_at_a_source():
    """What the decision is taken on: with a band declared, the reading of
    the scatter carries what the edge delivers and the decision is that
    its magnitude is above ``atol``; the gather, read as delivered, has no
    second quantity; and with no band declared nothing is carried."""
    with x64(True):
        new, old = iterates(1e-9, "float64")
        both = edges(1e9, "dense", "float64")
        banded = list(_interface_readings(both, new, old, band=True))
        assert [r.part.what for r in banded] == ["source", "delivered"]
        assert banded[0].delivered is not None and banded[1].delivered is None
        assert np.asarray(banded[0].delivered[0]).shape == (N,)
        assert bool(_kept_by_what_is_delivered(banded[0], ATOL)) is True
        assert bool(_kept_by_what_is_delivered(banded[0], 1e6)) is False
        assert _kept_by_what_is_delivered(banded[1], ATOL) is None
        plain = list(_interface_readings(both, new, old))
        assert all(r.delivered is None for r in plain)
        assert _kept_by_what_is_delivered(plain[0], ATOL) is None


def test_a_group_without_a_dead_band_applies_no_mapping_for_the_band(monkeypatch):
    """``atol == 0`` (the default) builds nothing: the residual and the
    floor of the pair apply the scatter not at all (it is read at its
    source), and with a band declared once per state read."""
    calls = []
    real = _interface_plan._delivered

    def counting(edge, value, mappings=None, geom=None):
        calls.append((edge.source_node, edge.target_node))
        return real(edge, value, mappings, geom)

    monkeypatch.setattr(_interface_plan, "_delivered", counting)
    with x64(True):
        new, old = iterates(1e-9, "float64")
        both = edges(1e9, "dense", "float64")
        coupling_residual_interface(new, old, both, 0.0, 1e-3)
        residual_precision_floor(new, ["p", "q"], "interface", 0.0, 1e-3, both)
        assert ("p", "q") not in calls and calls.count(("q", "p")) == 3
        calls.clear()
        coupling_residual_interface(new, old, both, ATOL, 1e-3)
        assert calls.count(("p", "q")) == 2
        calls.clear()
        residual_precision_floor(new, ["p", "q"], "interface", ATOL, 1e-3, both)
        assert calls.count(("p", "q")) == 1


def test_a_source_field_at_exactly_zero_stays_out_whatever_is_delivered():
    """The band's own guard is untouched: a field with no magnitude has
    no scale to be measured against, so it contributes nothing even where
    a transform delivers something from it."""
    with x64(True):
        new, old = iterates(1.0, "float64")
        zero = {**new, "p": {"x": jnp.zeros(M)}}
        offset = [EdgeSpec("p", "q", "x", "u", mapping=scatter("dense", "float64"),
                           transform=lambda v: v + 5.0),
                  EdgeSpec("q", "p", "x", "u", mapping=matrix_mapping(jnp.asarray(G)))]
        got = coupling_residual_interface(zero, zero | {"q": old["q"]}, offset, ATOL, 1e-3)
        want = coupling_residual_interface(zero, zero | {"q": old["q"]}, offset[1:], 0.0, 1e-3)
        assert _bits(got) == _bits(want)


# ---------------------------------------------------------------------------
# The verdict of a solve
# ---------------------------------------------------------------------------

#: What a converged step of this pair is held to here, in tolerances of
#: each field's own magnitude.  Measured with no dead band on jax 0.10.2,
#: 0.11.0 and 0.11.2: 0.9 to 7.5 (float64) over six steps.
PROMISE = 25.0


@pytest.mark.parametrize("kind", ["dense", "sparse-scatter"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_small_force_handed_on_in_other_units_is_held_to_the_criterion(schedule, kind):
    """The audited pair (forces of 1e-8 delivered as 17 to 520, a dead
    band of 1e-6): every converged step leaves both fields within the
    promise of the fixed point, as the same pair with no dead band does.
    Under Jacobi it accepted after one pass, 3.4e5 to 7.4e5 tolerances
    off, on every other step."""
    size, gain = QUADRANTS["field inside, delivered above"]
    with x64(True):
        gm = build(size, gain, schedule=schedule, atol=ATOL, kind=kind)
        seen = []
        for _ in range(4):
            q_pre = np.asarray(gm.get_node_state("q")["x"])
            gm.step()
            report = gm.coupling_diagnostics()[KEY]
            assert report["converged"], report
            seen.append(int(report["iterations"]))
            assert max(errors_in_tolerances(gm, size, q_pre, 1e-6)) < PROMISE, report
        assert min(seen) > 10, seen


# Per push: tests/core/test_the_dead_band_of_an_edge_read_at_its_source.py::test_a_small_force_handed_on_in_other_units_is_held_to_the_criterion
@pytest.mark.slow
@pytest.mark.parametrize("dtype, rtol", [("float32", 1e-4), ("float64", 1e-6)])
@pytest.mark.parametrize("kind", ["dense", "sparse-scatter"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
@pytest.mark.parametrize("quadrant", sorted(q for q in QUADRANTS if KEPT[q]))
def test_a_kept_reading_holds_both_fields_to_the_promise_in_every_quadrant(
        quadrant, schedule, kind, dtype, rtol):
    """Every quadrant in which the reading is kept, both schedules, both
    scatter layouts, float32 and float64: ``converged=True`` only with
    both fields within the promise of the closed-form fixed point."""
    size, gain = QUADRANTS[quadrant]
    with x64(dtype == "float64"):
        gm = build(size, gain, schedule=schedule, atol=ATOL, kind=kind, dtype=dtype, rtol=rtol)
        for _ in range(4):
            q_pre = np.asarray(gm.get_node_state("q")["x"])
            gm.step()
            report = gm.coupling_diagnostics()[KEY]
            if report["converged"]:
                assert max(errors_in_tolerances(gm, size, q_pre, rtol)) < PROMISE, report
        assert report["converged"], report


def test_the_report_takes_the_decision_the_residual_took():
    """With diagnostics on, the audited pair's kept reading is in the
    spectrum's weights and in the floor: the bound is a few tolerances,
    as with no dead band (asked of the source alone, the weights put the
    whole of ``p`` into the dead-banded share and the bound read 1.9e6 to
    3.7e6), and the floor the step recorded is the one the report uses."""
    size, gain = QUADRANTS["field inside, delivered above"]
    with x64(True):
        reports = {}
        for atol in (ATOL, 0.0):
            gm = build(size, gain, schedule="jacobi", atol=atol, diagnostics=True)
            gm.step()
            gm.step()
            reports[atol] = dict(gm.coupling_diagnostics()[KEY])
        banded, plain = reports[ATOL], reports[0.0]
        assert banded["spectral_usable"] and plain["spectral_usable"]
        assert banded["iterations"] == plain["iterations"]
        assert banded["spectral_error_bound"] == pytest.approx(
            plain["spectral_error_bound"], rel=1e-6)
        assert banded["spectral_error_bound"] < 1e3


def test_a_banded_group_whose_only_mapped_edge_is_read_at_its_source_records_its_floor():
    """Who owns the ``reading_floor`` slot.  The band asks what the
    scatter delivers, with the weights the step ran with, so the floor of
    such a group cannot be taken from the returned state alone and the
    step records it.  With no dead band declared the same group has no
    such slot, as before: its ``_meta`` and its program are unchanged."""
    size, gain = QUADRANTS["field inside, delivered above"]
    with x64(True):
        for atol in (ATOL, 0.0):
            gm = build(size, gain, schedule="jacobi", atol=atol, plain_back=True)
            gm.step()
            meta = gm._state["_meta"]                               # noqa: SLF001
            assert (f"coupling_{KEY}_reading_floor" in meta) == (atol > 0), sorted(meta)
            assert gm.coupling_diagnostics()[KEY]["converged"]


def test_the_recorded_floor_is_the_floor_of_the_returned_state():
    """The slot a banded group gains holds ``residual_precision_floor``
    of the state the step returned, per evaluation, by the same rule."""
    size, gain = QUADRANTS["field inside, delivered above"]
    with x64(True):
        gm = build(size, gain, schedule="jacobi", atol=ATOL)
        gm.step()
        state = {name: gm.get_node_state(name) for name in ("p", "q")}
        recorded = gm._state["_meta"][f"coupling_{KEY}_reading_floor"]      # noqa: SLF001
        want = residual_precision_floor(state, ["p", "q"], "interface", ATOL, 1e-6,
                                        edges(gain, "dense", "float64"))
        assert float(recorded) == pytest.approx(float(want), rel=1e-12)
        assert float(recorded) > 0.0
    assert acceleration._declares_a_band(ATOL) and not acceleration._declares_a_band(0.0)
