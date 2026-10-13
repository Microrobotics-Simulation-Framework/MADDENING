"""Under Jacobi a field inside the dead band un-holds the fields computed from it.

A field at or below ``atol`` leaves the residual (CPL-011).  Under Jacobi
every member reads the previous iterate, so with ``p`` dropped the
residual of a pair ``p <- f(q)``, ``q <- g(p)`` is ``|g(p_k) - q_k|``: it
tests that ``q`` agrees with the ``p`` it was computed from, and nothing
in that pass tests ``p = f(q)``.  A kept field can then be returned with
``converged=True`` far from its fixed point.  Gauss-Seidel reads the
member swept before it from the same pass, so on THIS pair (one loop,
one lagged read) its residual is the whole loop's; a pair whose loop
passes the lagged read twice fails under Gauss-Seidel as well
(``test_the_dead_band_of_a_pair_and_of_one_member.py``).

This is the dead band's own behaviour, on plain edges, under every norm
(MADD-ANO-254, open: no rule that only delays the exit is proved for
every loop, see the registry entry).  What is held here:

* the defect stays visible: a strict xfail on the plain pair below;
* its controls pass: the same pair with no dead band, and under
  Gauss-Seidel in both sweep orders (the dropped field read in the same
  sweep, and read from the previous pass): one measured instance, not a
  rule, and the pair is advised on under Gauss-Seidel too;
* ``validate()`` says so for a group that declares a dead band, under
  either schedule, and ``compile()`` warns.

**The pair** (plain edges; the audited construction with the scatter and
the amplification inside the grid node)::

    p (3 values, about 1e-8)   p.x <- 1e-9 (b + A G u)
    q (30 values)              q.x <- c + 0.5 q_pre + 1e9 H u
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec
from tests.core.test_the_dead_band_of_an_edge_read_at_its_source import (
    ATOL,
    C,
    H,
    KEY,
    M,
    PROMISE,
    Grid,
    Markers,
    errors_in_tolerances,
)
from tests.sparse_mapping_support import x64

SIZE = 1e-9
RTOL = 1e-6
NORMS = ("interface", "mixed", "l2")


class Spreading(Grid):
    """``q.x <- c + 0.5 q.x + back * (H u)``: handed the three forces
    through a plain edge, it spreads and amplifies them itself."""

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(M,), dtype=self._dt,
                                       default=jnp.zeros(M, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = (jnp.asarray(H, self._dt) @ boundary_inputs["u"]) * jnp.asarray(self._back, self._dt)
        return {"x": jnp.asarray(C, self._dt) + 0.5 * state["x"] + u}


def build(schedule: str, atol: float, norm: str, order: str = "pq") -> GraphManager:
    """The plain pair; *order* is the order the members are added in
    (the sweep's)."""
    nodes = {"p": Markers("p", 1.0, SIZE, "float64", samples_itself=True),
             "q": Spreading("q", 1.0, 1.0 / SIZE, "float64")}
    gm = GraphManager()
    for name in order:
        gm.add_node(nodes[name])
    gm.add_edge("p", "q", "x", "u")
    gm.add_edge("q", "p", "x", "u")
    tolerance = {"tolerance": RTOL} if norm == "l2" else {"rtol": RTOL}
    gm.add_coupling_group(["p", "q"], convergence_norm=norm, atol=atol,
                          iteration_mode=schedule, max_iterations=400, solver="ift", **tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def steps(gm: GraphManager, count: int = 4) -> list:
    """``(iterations, converged, worst error in tolerances)`` per step."""
    out = []
    for _ in range(count):
        q_pre = np.asarray(gm.get_node_state("q")["x"])
        gm.step()
        report = gm.coupling_diagnostics()[KEY]
        out.append((int(report["iterations"]), bool(report["converged"]),
                    max(errors_in_tolerances(gm, SIZE, q_pre, RTOL))))
    return out


def _held(run: list) -> bool:
    """Every converged step left both fields within the promise."""
    return all(error < PROMISE for _iterations, converged, error in run if converged)


@pytest.mark.xfail(strict=True, reason=(
    "MADD-ANO-254 (open): under Jacobi a field inside the dead band un-holds the fields "
    "computed from it; the pair accepts after one pass, 4.5e5 to 7.4e5 tolerances off"))
@pytest.mark.parametrize("norm", NORMS)
def test_a_converged_jacobi_step_holds_the_fields_computed_from_a_dead_banded_one(norm):
    """What CPL-011's oracle asks (the distance of the kept fields to
    their fixed point) and today does not hold: every other step accepts
    after one pass."""
    with x64(True):
        run = steps(build("jacobi", ATOL, norm))
    assert all(converged for _i, converged, _e in run), run
    assert _held(run), run


@pytest.mark.parametrize("norm", NORMS)
def test_the_same_pair_with_no_dead_band_is_held_under_jacobi(norm):
    """The control: ``atol = 0`` keeps ``p`` in the residual, and every
    step takes its dozens of passes to the fixed point."""
    with x64(True):
        run = steps(build("jacobi", 0.0, norm))
    assert all(converged for _i, converged, _e in run), run
    assert _held(run) and min(i for i, _c, _e in run) > 10, run


@pytest.mark.parametrize("order", ["pq", "qp"], ids=["read in the same sweep",
                                                     "read from the previous pass"])
@pytest.mark.parametrize("norm", NORMS)
def test_gauss_seidel_holds_the_pair_with_the_field_inside_the_dead_band(norm, order):
    """Gauss-Seidel with the dead band declared, the dropped member swept
    first (the kept one reads it in the same sweep) and swept second (the
    kept one reads it from the previous pass): both hold.  A measured
    instance, not a rule: this pair's loop has one lagged read, the
    advisory is given for it all the same, and other pairs fail."""
    with x64(True):
        run = steps(build("gauss-seidel", ATOL, norm, order))
    assert all(converged for _i, converged, _e in run), run
    assert _held(run) and min(i for i, _c, _e in run) > 10, run


def test_the_fixture_reaches_the_one_pass_acceptance():
    """The premise of the strict xfail, stated as what happens today: with
    the band declared the Jacobi pair accepts some step after one pass,
    far outside the promise.  When this stops being true the xfail above
    turns into a pass and both are to be rewritten."""
    with x64(True):
        run = steps(build("jacobi", ATOL, "mixed"))
    one_pass = [(i, e) for i, converged, e in run if converged and i == 1]
    assert one_pass and all(e > 1e3 * PROMISE for _i, e in one_pass), run


# ---------------------------------------------------------------------------
# The advisory
# ---------------------------------------------------------------------------

def _advisories(gm: GraphManager) -> list:
    return [issue for issue in gm.validate() if _group_layout._DEAD_BAND_ADVISORY in issue]


def _uncompiled(schedule: str, atol: float, norm: str = "mixed") -> GraphManager:
    gm = GraphManager()
    gm.add_node(Markers("p", 1.0, SIZE, "float32", samples_itself=True))
    gm.add_node(Spreading("q", 1.0, 1.0 / SIZE, "float32"))
    gm.add_edge("p", "q", "x", "u")
    gm.add_edge("q", "p", "x", "u")
    tolerance = {"tolerance": 1e-4} if norm == "l2" else {"rtol": 1e-4}
    gm.add_coupling_group(["p", "q"], convergence_norm=norm, atol=atol,
                          iteration_mode=schedule, **tolerance)
    return gm


@pytest.mark.parametrize("norm", NORMS)
def test_validate_advises_on_a_dead_band_declared_under_jacobi(norm):
    """One ``WARNING:`` line naming the group, its ``atol``, the measured
    pairs under both schedules, that no group is excepted and the one way
    out (``atol=0.0``); one for the same pair under Gauss-Seidel, under
    the other opening; none at the default ``atol`` under either
    schedule.  (Three or more members:
    ``test_the_dead_band_of_a_ring_of_three.py``.)"""
    lines = _advisories(_uncompiled("jacobi", ATOL, norm))
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("WARNING: coupling group ['p', 'q'] " + _group_layout._DEAD_BAND_UNDER_JACOBI)
    assert "atol=1e-06" in line and "2 members" in line
    assert "a pair under Jacobi" in line and "4.5e5 to 7.4e5" in line
    assert "a ring of three" not in line
    assert "under Gauss-Seidel a pair of 2-vectors" in line and "4.3e3 to 7.7e3" in line
    assert "is not affected" not in line and "No group is excepted" in line
    assert "Set atol=0.0 (the default) on this group" in line and "MADD-ANO-254" in line
    (swept,) = _advisories(_uncompiled("gauss-seidel", ATOL, norm))
    assert swept.startswith("WARNING: coupling group ['p', 'q'] declares a dead band on 2 members")
    assert "iteration_mode='gauss-seidel'" in swept and "under Jacobi" not in swept
    assert "is not affected" not in swept and "No group is excepted" in swept
    assert "Set atol=0.0 (the default) on this group" in swept
    for schedule in ("gauss-seidel", "jacobi"):
        assert _advisories(_uncompiled(schedule, 0.0, norm)) == []


def test_compile_warns_once_for_such_a_group():
    """``compile()`` emits the advisory as a ``UserWarning``, as it does
    every advisory ``validate()`` returns (the test configuration filters
    it for the suites that run such groups on purpose): once for the pair
    under Jacobi, once for the same pair under Gauss-Seidel, and not at
    the default ``atol``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _uncompiled("jacobi", ATOL).compile()
    ours = [w for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
    assert len(ours) == 1 and issubclass(ours[0].category, UserWarning)
    assert _group_layout._DEAD_BAND_UNDER_JACOBI in str(ours[0].message)
    for schedule, atol, due in (("gauss-seidel", ATOL, 1), ("gauss-seidel", 0.0, 0),
                                ("jacobi", 0.0, 0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _uncompiled(schedule, atol).compile()
        ours = [w for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
        assert len(ours) == due, (schedule, atol, [str(w.message) for w in caught])
        assert all(_group_layout._DEAD_BAND_ON_MEMBERS + " 2 members" in str(w.message)
                   for w in ours)


def test_the_configured_filter_matches_the_advisory_under_both_openings_and_no_other():
    """The test configuration ignores ``compile()``'s copy of the advisory
    for every group that earns it (the advisory is due wherever ``atol >
    0``, and the suite's batteries and searches build such groups on
    purpose): its pattern is the two openings, of one member, a pair and
    three members under either schedule, and it matches no other
    advisory ``validate()`` returns for the same graphs, nor another
    sentence about a dead band."""
    import re
    import tomllib
    from pathlib import Path

    from maddening.core.coupling.group import CouplingGroup

    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    ours = [f for f in config["tool"]["pytest"]["ini_options"]["filterwarnings"]
            if "dead band" in f]
    assert len(ours) == 1
    action, pattern, category = ours[0].split(":")
    assert action == "ignore" and category == "UserWarning"
    # The pattern is the two openings and nothing after them.
    assert re.fullmatch(pattern, "WARNING: ... " + _group_layout._DEAD_BAND_UNDER_JACOBI)
    assert re.fullmatch(pattern, "WARNING: ... " + _group_layout._DEAD_BAND_ON_MEMBERS + " 2 member")

    def line(members, schedule):
        (out,) = _group_layout._dead_band_advisories(
            CouplingGroup(frozenset(members), atol=ATOL, iteration_mode=schedule))
        return out

    for members in ("p", "pq", "pqr"):
        for schedule in ("jacobi", "gauss-seidel"):
            assert re.match(pattern, line(members, schedule)), (members, schedule)
    assert "declares a dead band on 1 member (" in line("p", "gauss-seidel")
    assert "1 member)" in line("p", "jacobi")
    # The warnings machinery matches from the start of the message, as
    # ``re.match`` does: another advisory, or another sentence that names
    # a dead band, is not ignored.
    others = [issue for issue in _uncompiled("gauss-seidel", ATOL).validate()
              if _group_layout._DEAD_BAND_ADVISORY not in issue]
    others += ["WARNING: coupling group ['p', 'q'] declares a dead band of 1e-06",
               "WARNING: a dead band on 2 members hides a change"]
    assert not [text for text in others if re.match(pattern, text)]
