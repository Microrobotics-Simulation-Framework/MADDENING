"""The interface norm and its bound on mapped edges, in every numeric domain.

Under ``convergence_norm="interface"`` a coupling group measures what its
internal edges *deliver*: each edge's source field through the edge's
interface mapping, with the weights the step ran with, and then its
transform (CPL-041).  The spectral bound is taken in that reading (CPL-088).
Those rows' own tests run float32 pairs; here both claims are stated once,
on a pair with one internal edge of each kind -- ``b -> a`` through a
mapping, ``a -> b`` plain -- and run in every
domain of :mod:`tests.core.coupling_domains` -- float64, a float32 member
beside a float64 one under x64 (where a float64 matrix applied to a float32
field delivers float64 numbers that carry float32 rounding), bfloat16 and
float16, a ``jax.vmap`` of the step, a multi-rate graph, a sub-cycled
group, a predictor, a checkpoint restart, (slow) ``run_adaptive`` and, from
``tests/cloud/multigpu``, a member sharded over four devices.

**The fixture.**  ``x_a <- g_a (H x_b) + c_a``, ``x_b <- g_b x_a + c_b`` on
two entries a member, memoryless, so every domain solves the same fixed
point.  Each member's first entry is a constant 32.  The mapping weights it
by a few thousandths, so what the mapped edge delivers is of order one and
moves with ``b``'s second entry, while its source field's magnitude is the
32: read on the source field, the same change is an order of magnitude
smaller -- more than every dtype here resolves, bfloat16's eight bits
included.  (The plain edge delivers its source field, the 32 with it, and
is read as it always was.)  The mapping *object* holds a decoy matrix and
the real one reaches each step through ``params["mappings"]``, as a fitted
or a per-step weight does: a reading taken with the mapping's own weights
is a reading of another edge.

**The oracles** are in float64 NumPy from the parameters and matrices as
each dtype stores them, and call nothing of the library's reading:

* CPL-041: a group stopped at its cap unconverged reports the residual of
  the iterate its loop stopped on, ``||F(x) - x||`` with ``F`` one
  Gauss-Seidel pass -- here the RMS, over every entry the two edges
  deliver, of the change of the delivered value over ``rtol`` times its
  edge's largest magnitude.  (The step returns that iterate with the
  field read only through the mapping one pass on, CPL-191, so the
  iterate is read with the return rule switched off);
* CPL-088: where ``spectral_usable`` is True, ``spectral_error_bound`` is
  at least the distance to the exact fixed point in the same norm at the
  returned state;
* CPL-188: the residual's float floor the report adds is the one the step
  measured, with the weights it ran with: the ``reading_floor`` slot is
  ``residual_precision_floor`` of the returned state under those weights,
  and not under the mapping object's own.
"""

from __future__ import annotations

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    residual_precision_floor,
)
from tests.core import coupling_domains as cd
from tests.property import coupled_graphs as cg

#: Every domain, ``run_adaptive`` slow (it compiles its step on every call).
EVERY = list(cd.EVERY) + [pytest.param(cd.ADAPTIVE, marks=pytest.mark.slow)]

#: The matrix the steps run with on ``b -> a``: not a selection, a scale or
#: symmetric.  Every number here is exact in bfloat16.
H_BA = np.array([[2.0 ** -8, 1.0], [-(2.0 ** -8), 0.875]])
#: What the mapping object holds instead.
DECOY = np.array([[0.125, 0.25], [0.5, -0.125]])
#: A member's first entry has no gain: it stays at its forcing, 32.  The
#: second entries close the loop at a gain of 0.875.
GAINS = ((0.0, 1.0), (0.0, 1.0))
FORCING = ((32.0, -0.5), (32.0, 0.75))
#: ``(b -> a, a -> b)``: one mapped edge, one plain.
MAPPINGS = (H_BA, None)

_GRAPHS: dict = {}


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _rtol(domain) -> float:
    """A tolerance the dtype resolves, far below the pair's distance at the cap."""
    return 0.02 if _sixteen(domain) else 1e-5


def _graph(label, accepted=False):
    """The mapped pair in *label*'s domain: two passes with the report's
    analysis on, compiled once per module.  *accepted*: a second graph, to
    be stepped with the return rule switched off (:func:`_solves`)."""
    if (label, accepted) not in _GRAPHS:
        d = cd.DOMAINS[label]
        with cd.entered(d):
            _GRAPHS[label, accepted] = cd.pair(
                d, n=2, g=GAINS, c=FORCING, mappings=(DECOY, None),
                convergence_norm="interface", rtol=_rtol(d), max_iterations=2,
                diagnostics=True)
    return _GRAPHS[label, accepted]


def _scenarios(domain, gm) -> list:
    """Steps' parameters: the real matrix, and the forcing moving from
    step to step where the domain runs a sequence."""
    moves = (0.0, 0.25, -0.125, 0.5)
    seq = [cd.params_with(gm, g=GAINS, mappings=MAPPINGS,
                          c=tuple(np.asarray(c) * (1.0 + m) for c in FORCING)) for m in moves]
    return [seq] if (domain.predictor or domain.restart) else [seq[0]]


#: The graphs already traced with the return rule switched off.
_TRACED_AT_THE_ACCEPTED_ITERATE: set = set()


def _solves(label, accepted=False) -> list:
    """The domain's solves.  *accepted*: of the iterate each loop stopped on
    -- the state its report is of -- read by stepping a graph of its own
    with the interface norm's return rule switched off
    (:func:`tests.property.coupled_graphs.accepted_iterate`)."""
    d = cd.DOMAINS[label]
    with contextlib.ExitStack() as stack:
        asked = stack.enter_context(cg.accepted_iterate()) if accepted else None
        gm = _graph(label, accepted)
        stack.enter_context(cd.entered(d))  # x64 on in the float64 and mixed-dtype domains
        out = []
        for scenario in _scenarios(d, gm):
            if isinstance(scenario, list):
                out.extend(cd.run_sequence(d, gm, scenario))
            else:
                out.extend(cd.run(d, gm, [scenario]))
        cd.assert_in_domain(d, gm, out)
    if accepted and label not in _TRACED_AT_THE_ACCEPTED_ITERATE:
        assert asked, "the step never asked the return rule: the patch is on the wrong name"
        _TRACED_AT_THE_ACCEPTED_ITERATE.add(label)
    return out


def _delivered(s, state) -> list:
    """What the two edges deliver at *state* (``{"a": x_a, "b": x_b}``), float64."""
    h_ba, h_ab = s.mapping_matrices()
    return [h_ba @ state["b"], h_ab @ state["a"]]


def _one_pass(gm, s, x) -> dict:
    """One Gauss-Seidel pass of the pair from *x*, in the group's sweep order."""
    ga, gb = (np.asarray(g, np.float64) for g in s.gains())
    ca, cb = s.forcing()
    h_ba, h_ab = s.mapping_matrices()
    cur = dict(x)
    for name in [n for n in gm.schedule if n in ("a", "b")]:
        if name == "a":
            cur["a"] = ga * (h_ba @ cur["b"]) + ca
        else:
            cur["b"] = gb * (h_ab @ cur["a"]) + cb
    return cur


def _interface_norm(new: list, old: list, rtol, weights_from_new=False) -> float:
    """The RMS over every delivered entry of its change over ``rtol`` times
    its field's largest magnitude (over the pair, or at *new* alone)."""
    terms = []
    for a, b in zip(new, old):
        ref = np.max(np.abs(a)) if weights_from_new else max(np.max(np.abs(a)),
                                                             np.max(np.abs(b)))
        terms.append(np.abs(a - b) / (rtol * ref))
    return float(np.sqrt(np.mean(np.concatenate(terms) ** 2)))


def _state(s) -> dict:
    return {n: np.asarray(s.x(n), np.float64) for n in ("a", "b")}


# Per push: tests/core/test_coupling_mapped_edges_in_every_domain.py::test_the_interface_norm_measures_what_mapped_edges_deliver
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_the_interface_norm_measures_what_mapped_edges_deliver(label):
    """CPL-041 on a mapped internal edge beside a plain one.

    At its cap of two passes the group is far from converged, so the
    residual it reports is that of the iterate its loop stopped on: the
    change, over one more pass, of what the two edges deliver -- the mapped
    one with the matrix the step ran with.  Read on the source fields -- or
    with the mapping object's own weights -- the same iterate gives another
    number.  (The step returns that iterate with ``b``, which is read only
    through the mapping, one pass on; the iterate itself is read with the
    return rule switched off.)
    """
    d = cd.DOMAINS[label]
    gm = _graph(label, accepted=True)
    eps = float(cd.finfo(d.coarsest).eps)
    for s in _solves(label, accepted=True):
        r = s.report
        x = _state(s)
        fx = _one_pass(gm, s, x)
        want = _interface_norm(_delivered(s, fx), _delivered(s, x), _rtol(d))
        # run_adaptive's controller repeats the capped solve on shrinking
        # steps, so its state nears the fixed point; the residual is checked
        # all the same.
        assert not r["converged"] or d.adaptive, (label, r)
        # The one more pass and the delivered values are evaluated in the
        # members' dtype: each entry's change carries a rounding of a few eps
        # of its field's magnitude, which is ``eps / rtol`` in the norm's
        # units whatever the change.  Two of those (the measured worst is
        # 0.19, on this fixture's numbers, exact in every dtype here).
        tolerance = 2 * eps / _rtol(d)
        assert abs(r["residual"] - want) <= tolerance, (
            f"{label}: reported {r['residual']:.6e}, the delivered values moved {want:.6e} "
            f"(tolerance {tolerance:.2e})")
        # The fixture premise: this dtype tells the two readings apart.
        raw = _interface_norm([fx["b"], fx["a"]], [x["b"], x["a"]], _rtol(d))
        assert abs(raw - want) > 4 * tolerance or d.adaptive, (
            f"{label}: the source fields read {raw:.4e}, within the tolerance of {want:.4e}")


# Per push: tests/core/test_coupling_mapped_edges_in_every_domain.py::test_a_usable_bound_covers_the_distance_in_what_mapped_edges_deliver
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_a_usable_bound_covers_the_distance_in_what_mapped_edges_deliver(label):
    """CPL-088 on a mapped internal edge beside a plain one.

    The spectrum is taken on the reading the residual is in, so the bound
    covers the distance to the exact fixed point measured in what the edges
    deliver, each over its own magnitude at the returned state.
    """
    d = cd.DOMAINS[label]
    for s in _solves(label):
        r = s.report
        x = _state(s)
        xa_star, xb_star = cd.mapped_fixed_point(s)
        dist = _interface_norm(_delivered(s, x), _delivered(s, {"a": xa_star, "b": xb_star}),
                               _rtol(d), weights_from_new=True)
        assert dist > 1.0 or d.adaptive, (
            f"{label}: fixture premise: outside the tolerance at the cap ({dist})")
        assert r["spectral_usable"] is True, (label, r)
        assert r["spectral_error_bound"] >= dist, (
            f"{label}: bound {r['spectral_error_bound']:.4e} under the distance {dist:.4e} "
            f"in the delivered values ({r})")


# Per push: tests/core/test_coupling_mapped_edges_in_every_domain.py::test_the_reports_floor_is_the_one_the_step_measured_with_its_weights
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_the_reports_floor_is_the_one_the_step_measured_with_its_weights(label):
    """CPL-188: a group whose norm reads a mapped edge carries the floor the
    step measured, in every domain.

    The ``reading_floor`` slot is ``residual_precision_floor`` of the
    returned state with the parameters the step ran with, to the bit, and
    it counts each delivered value at the coarser of its own dtype's eps and
    its source field's: in the mixed-dtype domain the plain edge casts a
    float32 field to float64 and is counted at float32's, so the floor is
    ``4 eps32 / rtol`` there and not the ``1 / sqrt(2)`` of it that float64's
    eps on that edge would give.  (That the weights are the step's and not
    the mapping object's is told apart by a dead band, in
    ``test_coupling_interface_reading_is_what_the_edge_delivers.py``: both
    sets of weights keep this pair's edges in the norm.)
    """
    d = cd.DOMAINS[label]
    gm = _graph(label)
    group = gm._committed_coupling_groups[cd.KEY]
    edges = [e for e in gm._edges if {e.source_node, e.target_node} <= {"a", "b"}]
    eps = float(cd.finfo(d.coarsest).eps)
    for s in _solves(label):
        slot = np.asarray(s.meta[f"coupling_{cd.KEY}_reading_floor"])
        with cd.entered(d):
            state = {n: {"x": jnp.asarray(s.x(n))} for n in ("a", "b")}
            want = np.asarray(residual_precision_floor(
                state, ["a", "b"], "interface", group.atol, group.rtol, edges,
                evaluations=1.0, mappings=s.params["mappings"]))
        assert float(slot) == float(want.astype(slot.dtype)), (label, slot, want)
        assert float(slot) == pytest.approx(PRECISION_FLOOR_ULPS * eps / _rtol(d), rel=1e-6), (
            f"{label}: every delivered entry at the coarsest member's eps")
