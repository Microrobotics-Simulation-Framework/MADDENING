"""On three members a dead band hides a change under Gauss-Seidel too.

A field at or below ``atol`` leaves the residual (CPL-011), and the loop
accepts on the first residual it measures, between its first two passes.
A change that has to cross such a field is therefore not in the residual
until it reaches a field the norm keeps.  Under Gauss-Seidel a member
swept before the one it reads still reads the previous pass, so on a
ring of three swept against its data flow the change needs three passes
to arrive, and the loop has accepted after one.

This is the dead band's own behaviour under every norm, with and without
acceleration (MADD-ANO-254, open; the redesign of ``atol`` is 0.5.0's).
What is held here:

* the defect stays visible: a strict xfail of the claim "a converged
  step leaves the kept field within ``PROMISE`` tolerances of its fixed
  point" on the ring below, and its premise stated as what happens today;
* its controls pass: the same ring with no dead band; the same ring
  swept along its data flow, and a PAIR of the same two kinds of member
  in both sweep orders (measured instances, not rules: other pairs fail,
  ``test_the_dead_band_of_a_pair_and_of_one_member.py``);
* ``compile()`` warns for the ring and for the pair with the band
  declared, and every build here expects exactly that.

**The ring** (float32; data flow A -> C -> B -> A; members added, and so
swept, A, B, C)::

    A.x (3 displacements, O(10))   <- 1e9 * u         reads B.x
    C.x (36 forces, about 1e-8)    <- load + 0.5e-9 u reads A.x  (the load grows each step)
    B.x (36 forces, about 1e-8)    <- u               reads C.x

    A -> C  nearest neighbour, 3 onto 36     B -> A  nearest neighbour, 36 onto 3

Loop gain 0.5, ``rtol = 1e-4``, ``atol = 1e-6``: both force fields are
inside the band, ``A.x`` is the one field every norm keeps.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout
from maddening.core.coupling.sparse_mapping import sparse_nearest_neighbor_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

NA, NF = 3, 36
DTYPE = "float32"
RTOL = 1e-4
ATOL = 1e-6
STIFFNESS = 1e9
COMPLIANCE = 0.5e-9
NORMS = ("interface", "mixed", "l2")
#: The norm the per-push tests run: the one whose ring is the closest to
#: its fixed point of the three (4.6e3 tolerances against 1.8e4 and 8e3).
PER_PUSH_NORM = "mixed"
ACCELERATIONS = ("none", "aitken", "iqn-ils")

#: The claim, in tolerances: a converged step leaves the kept field this
#: close to its fixed point.  Measured without the band on jax 0.11.0:
#: 0.16 to 3.0 on the ring under the three norms, 0.75 on the pair.
PROMISE = 25.0

#: What the banded ring is pinned at: more than this many tolerances off
#: on some converged one-pass step.  Measured (CPU; the same four digits
#: on jax 0.10.2, 0.11.0 and 0.11.2) with no acceleration, Aitken and
#: IQN-ILS: 4.6e3 under "mixed" in all three, 8.0e3 under "l2" in all
#: three, and 1.8e4, 4.6e3 and 1.8e4 under "interface".  1e3 is a loose
#: lower bound on the smallest of them: a factor 4.6 of room for another
#: jax or platform, and still 40 promises.
PINNED_OFF = 1e3


class Relay(SimulationNode):
    """``x <- load + gain * u`` entrywise; ``load`` is an external input
    of the one member that is *loaded* (zero for the others)."""

    def __init__(self, name: str, size: int, gain: float, loaded: bool = False):
        super().__init__(name, 1.0)
        self._n, self._gain, self._loaded = size, gain, loaded

    def initial_state(self):
        return {"x": jnp.zeros(self._n, DTYPE)}

    def boundary_input_spec(self):
        zero = jnp.zeros(self._n, DTYPE)
        spec = {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.dtype(DTYPE), default=zero)}
        if self._loaded:
            spec["load"] = BoundaryInputSpec(shape=(self._n,), dtype=jnp.dtype(DTYPE),
                                             default=zero)
        return spec

    def update(self, state, boundary_inputs, dt, *, params=None):
        x = jnp.asarray(self._gain, DTYPE) * boundary_inputs["u"]
        return {"x": boundary_inputs["load"] + x if self._loaded else x}

    def update_evaluations(self):
        return 1


def _positions(count: int) -> np.ndarray:
    return (np.arange(count) + 0.5) / count


def _spread() -> np.ndarray:
    """The 36 x 3 matrix of the A -> C edge (and its transpose's pattern
    for B -> A, which reads the nearest of the 36)."""
    return np.asarray(sparse_nearest_neighbor_mapping(_positions(NA), _positions(NF)).apply(
        jnp.eye(NA, dtype=jnp.float32), None), np.float64)


def _gather() -> np.ndarray:
    return np.asarray(sparse_nearest_neighbor_mapping(_positions(NF), _positions(NA)).apply(
        jnp.eye(NF, dtype=jnp.float32), None), np.float64)


def _compile(gm: GraphManager, advised: bool) -> None:
    """Compile, expecting the dead-band advisory exactly where it is due."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    ours = [w for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
    assert len(ours) == (1 if advised else 0), [str(w.message) for w in caught]


def _group(gm: GraphManager, members, norm: str, atol: float, acceleration: str,
           schedule: str = "gauss-seidel") -> None:
    tolerance = {"tolerance": RTOL} if norm == "l2" else {"rtol": RTOL}
    gm.add_coupling_group(list(members), convergence_norm=norm, atol=atol,
                          iteration_mode=schedule, max_iterations=200, solver="ift",
                          acceleration=acceleration, **tolerance)


def ring_graph(norm: str, atol: float, order: str = "ABC", acceleration: str = "none",
               schedule: str = "gauss-seidel") -> GraphManager:
    """The ring, not compiled; *order* is the order the members are added
    in (the sweep's)."""
    nodes = {"A": Relay("A", NA, STIFFNESS), "B": Relay("B", NF, 1.0),
             "C": Relay("C", NF, COMPLIANCE, loaded=True)}
    gm = GraphManager()
    for name in order:
        gm.add_node(nodes[name])
    gm.add_external_input("C", "load", shape=(NF,), dtype=jnp.dtype(DTYPE))
    gm.add_edge("A", "C", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NA), _positions(NF)))
    gm.add_edge("C", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NF), _positions(NA)))
    _group(gm, order, norm, atol, acceleration, schedule)
    return gm


def ring(norm: str, atol: float, order: str = "ABC", acceleration: str = "none") -> GraphManager:
    """The ring under Gauss-Seidel, compiled: advised on exactly where it
    declares the band."""
    gm = ring_graph(norm, atol, order, acceleration)
    _compile(gm, advised=atol > 0)
    return gm


def pair(norm: str, atol: float, order: str) -> GraphManager:
    """The same displacement and force with no relay between them:
    ``A.x <- 1e9 u`` reads ``C.x``, ``C.x <- load + 0.5e-9 u`` reads
    ``A.x``.  Advised on wherever it declares the band, as every group
    is."""
    nodes = {"A": Relay("A", NA, STIFFNESS), "C": Relay("C", NF, COMPLIANCE, loaded=True)}
    gm = GraphManager()
    for name in order:
        gm.add_node(nodes[name])
    gm.add_external_input("C", "load", shape=(NF,), dtype=jnp.dtype(DTYPE))
    gm.add_edge("A", "C", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NA), _positions(NF)))
    gm.add_edge("C", "A", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NF), _positions(NA)))
    _group(gm, order, norm, atol, "none")
    _compile(gm, advised=atol > 0)
    return gm


def steps(gm: GraphManager, norm: str, count: int = 6) -> list:
    """``(iterations, converged, distance of A.x in tolerances)`` for each
    step after the first (which starts from zero), the load on C growing
    by half each step.  The fixed point is the float64 solve of
    ``A = 1e9 G (load + 0.5e-9 S A)`` with the constants as float32 holds
    them."""
    spread, gather = _spread(), _gather()
    stiffness = float(np.float32(STIFFNESS))
    compliance = float(np.float32(COMPLIANCE))
    gm.reset_state()
    out, size = [], 1e-8
    for step in range(count):
        load = (size * (1.0 + 0.3 * np.arange(NF) / NF)).astype(np.float32)
        gm.step(external_inputs={"C": {"load": jnp.asarray(load)}})
        (report,) = gm.coupling_diagnostics().values()
        if step:
            # The first step starts from zero and, with the band declared,
            # returns A.x at zero: it has no scale to be measured against.
            x = np.asarray(gm.get_node_state("A")["x"], np.float64)
            system = np.eye(NA) - stiffness * compliance * (gather @ spread)
            fixed = np.linalg.solve(system, stiffness * (gather @ load.astype(np.float64)))
            relative = np.abs(x - fixed) / np.max(np.abs(x))
            # A.x is the one kept field: "interface" and "mixed" pool the
            # mean square over its entries, "l2" the sum.
            off = np.sqrt(np.sum(relative ** 2) if norm == "l2"
                          else np.mean(relative ** 2)) / RTOL
            out.append((int(report["iterations"]), bool(report["converged"]), float(off)))
        size *= 1.5
    return out


def _held(run: list) -> bool:
    """Every converged step left the kept field within the promise."""
    return all(off < PROMISE for _iterations, converged, off in run if converged)


def _pinned(run: list) -> bool:
    """Some converged step accepted after ONE pass far outside the promise."""
    return any(converged and iterations == 1 and off > PINNED_OFF
               for iterations, converged, off in run)


_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-254 (open): on three members under Gauss-Seidel a change that has to cross "
    "two dead-banded fields is not seen until it reaches the kept one; the ring accepts "
    "after one pass 4.6e3 to 1.8e4 tolerances off"))


# ---------------------------------------------------------------------------
# The claim that does not hold today, and its premise
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def banded_run() -> list:
    """Five steps of the banded ring under the per-push norm (the numbers
    only: no compiled graph outlives the fixture's body)."""
    return steps(ring(PER_PUSH_NORM, ATOL), PER_PUSH_NORM)


@_XFAIL
def test_a_converged_step_of_the_banded_ring_holds_its_kept_field(banded_run):
    """What CPL-011's oracle asks of the kept field, and today does not
    hold on the ring swept against its data flow."""
    run = banded_run
    assert all(converged for _i, converged, _off in run), run
    assert _held(run), run


def test_the_banded_ring_accepts_after_one_pass_far_from_its_fixed_point(banded_run):
    """The premise of the strict xfail, as what happens today.  When this
    stops being true the xfail above turns into a pass and both are to be
    rewritten."""
    run = banded_run
    assert all(converged for _i, converged, _off in run), run
    assert _pinned(run), run


# Per push: tests/core/test_the_dead_band_of_a_ring_of_three.py::test_a_converged_step_of_the_banded_ring_holds_its_kept_field
@pytest.mark.slow
@_XFAIL
@pytest.mark.parametrize("acceleration", ACCELERATIONS)
@pytest.mark.parametrize("norm", NORMS)
def test_a_converged_step_of_the_banded_ring_holds_its_kept_field_under_every_norm(
        norm, acceleration):
    """The same claim under each norm, with no acceleration, with Aitken
    and with IQN-ILS: none holds."""
    run = steps(ring(norm, ATOL, acceleration=acceleration), norm)
    assert all(converged for _i, converged, _off in run), run
    assert _held(run), run


# Per push: tests/core/test_the_dead_band_of_a_ring_of_three.py::test_the_banded_ring_accepts_after_one_pass_far_from_its_fixed_point
@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ACCELERATIONS)
@pytest.mark.parametrize("norm", NORMS)
def test_the_banded_ring_accepts_after_one_pass_under_every_norm(norm, acceleration):
    """The premise in every cell of the slow xfail above."""
    run = steps(ring(norm, ATOL, acceleration=acceleration), norm)
    assert all(converged for _i, converged, _off in run), run
    assert _pinned(run), run


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------

def test_the_same_ring_with_no_dead_band_is_held():
    """``atol = 0`` keeps the two force fields of 1e-8 in the residual
    (only a field that is exactly zero leaves it), and every step takes
    its passes to the fixed point."""
    run = steps(ring(PER_PUSH_NORM, 0.0), PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in run), run
    assert _held(run) and min(i for i, _c, _off in run) > 3, run


@pytest.mark.parametrize("order", ["AC", "CA"], ids=["the force read from the previous pass",
                                                     "the force read in the same sweep"])
def test_a_pair_holds_with_the_force_inside_the_dead_band_in_both_sweep_orders(order):
    """This pair held with the band declared, in both sweep orders: its
    one loop has one lagged read.  A measured instance and not a rule (a
    pair whose loop passes that read twice does not hold), so the
    advisory is given for it too, as ``pair()`` expects."""
    run = steps(pair(PER_PUSH_NORM, ATOL, order), PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in run), run
    assert _held(run) and min(i for i, _c, _off in run) > 3, run


def test_the_banded_ring_swept_along_its_data_flow_held():
    """One measured instance, not a rule: swept C, B, A each member reads
    the one before it from the same pass, and this ring held."""
    run = steps(ring(PER_PUSH_NORM, ATOL, order="CBA"), PER_PUSH_NORM)
    assert all(converged for _i, converged, _off in run), run
    assert _held(run), run


# Per push: tests/core/test_the_dead_band_of_a_ring_of_three.py::test_the_same_ring_with_no_dead_band_is_held
@pytest.mark.slow
@pytest.mark.parametrize("norm", [n for n in NORMS if n != PER_PUSH_NORM])
def test_the_controls_hold_under_the_other_norms(norm):
    """No dead band; the pair in both sweep orders; the ring swept along
    its data flow (a measured instance)."""
    for name, gm in (("no band", ring(norm, 0.0)), ("pair AC", pair(norm, ATOL, "AC")),
                     ("pair CA", pair(norm, ATOL, "CA")),
                     ("ring CBA", ring(norm, ATOL, order="CBA"))):
        run = steps(gm, norm)
        assert all(converged for _i, converged, _off in run), (name, run)
        assert _held(run), (name, run)


# ---------------------------------------------------------------------------
# The advisory on three members
# ---------------------------------------------------------------------------

def _advisories(gm: GraphManager) -> list:
    return [issue for issue in gm.validate() if _group_layout._DEAD_BAND_ADVISORY in issue]


@pytest.mark.parametrize("norm", NORMS)
def test_validate_advises_on_a_dead_band_declared_on_three_members(norm):
    """One ``WARNING:`` line under Gauss-Seidel, naming the group, its
    ``atol``, the member count, the measured ring, the measured pairs,
    that no group is excepted and the one way out.  It does not offer
    Gauss-Seidel as a remedy: that is the schedule the group already
    runs."""
    lines = _advisories(ring_graph(norm, ATOL))
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("WARNING: coupling group ['A', 'B', 'C'] declares a dead band on 3 members")
    assert "atol=1e-06" in line and "iteration_mode='gauss-seidel'" in line
    assert "a ring of three members under Gauss-Seidel" in line and "4.6e3 to 1.8e4" in line
    assert "under Gauss-Seidel a pair of 2-vectors" in line and "4.3e3 to 7.7e3" in line
    assert "is not affected" not in line and "No group is excepted" in line
    assert "Set atol=0.0 (the default) on this group" in line and "MADD-ANO-254" in line
    assert "Use " not in line and "under Jacobi" not in line


def test_three_members_under_jacobi_get_one_advisory_that_names_both_cases():
    lines = _advisories(ring_graph(PER_PUSH_NORM, ATOL, schedule="jacobi"))
    assert len(lines) == 1
    line = lines[0]
    assert _group_layout._DEAD_BAND_UNDER_JACOBI in line and "3 members" in line
    assert "a pair under Jacobi" in line and "a ring of three members" in line
    assert "Set atol=0.0 (the default) on this group" in line


@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("norm", NORMS)
def test_a_group_with_no_dead_band_gets_no_advisory(norm, schedule):
    """The default ``atol`` stays silent, on three members and under
    Jacobi: ``validate()`` has no such line and ``compile()`` no such
    warning."""
    gm = ring_graph(norm, 0.0, schedule=schedule)
    assert _advisories(gm) == []
    _compile(gm, advised=False)


def test_compile_warns_once_for_a_banded_group_of_three():
    """``compile()`` emits the advisory as one ``UserWarning``, under the
    opening that names the member count (the test configuration filters
    both openings; every build in this module records the advisory
    itself and counts it)."""
    with pytest.warns(UserWarning, match=_group_layout._DEAD_BAND_ADVISORY) as caught:
        ring_graph(PER_PUSH_NORM, ATOL).compile()
    ours = [w for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
    assert len(ours) == 1
    assert _group_layout._DEAD_BAND_UNDER_JACOBI not in str(ours[0].message)


def test_at_no_dead_band_only_a_field_at_exactly_zero_leaves_the_residual():
    """What the advisory says ``atol=0.0`` does, read from the residual's
    own helper: a field with no magnitude contributes nothing and is not
    active; a field of 1e-30 is active and its change is measured against
    its own magnitude."""
    from maddening.core.coupling.acceleration import _scaled_change

    zero = jnp.zeros(3, DTYPE)
    scaled, active = _scaled_change(zero, zero, 0.0, RTOL)
    assert not bool(active) and float(jnp.max(scaled)) == 0.0
    new = jnp.asarray([1e-30, 2e-30, 3e-30], DTYPE)
    old = new * jnp.asarray(1.5, DTYPE)
    scaled, active = _scaled_change(new, old, 0.0, RTOL)
    assert bool(active)
    np.testing.assert_allclose(np.asarray(scaled), 0.5 * np.asarray(new) / (RTOL * 4.5e-30),
                               rtol=1e-4)
    # ... and the same small field inside a declared band is not.
    assert not bool(_scaled_change(new, old, ATOL, RTOL)[1])
