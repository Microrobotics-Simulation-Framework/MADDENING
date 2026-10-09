"""The interface norm on a mapped edge that expands, and one that reduces, in every numeric domain.

Under ``convergence_norm="interface"`` an internal edge whose static mapping
delivers **more** entries than its source field holds is read at its
source value -- the compact side -- and every other edge as delivered
(CPL-188); a group that reports ``converged=True`` is then within ``K``
tolerances of its fixed point in those readings, ``K`` from the loop's own
operator and blind to the size of the large field (CPL-192).  Those rows'
own tests run float32 and float64 pairs, one rate, unbatched.  The pairs
of :mod:`tests.core.coupling_domains` have one size for both members, so
in the battery of every other domain a mapped edge is a tie, read as
delivered: the rule itself had not run there.

Here it does.  :mod:`tests.core.coupling_domain_sizes` builds a small field
(three entries) coupled to a large one (thirty-six) in each domain --
float32, float64, a float32 member beside a float64 one, bfloat16 and
float16, a ``jax.vmap`` of the step, a multi-rate graph, a sub-cycled
group, a predictor, a checkpoint restart, (slow) ``run_adaptive`` and, from
``tests/cloud/multigpu``, a member sharded over four devices -- two ways
(one edge expands, one reduces: either member the small one), with both
edges expanding and with both reducing, through the dense kind and both
sparse layouts, under Gauss-Seidel and Jacobi, plain and accelerated.

**The oracles** call nothing of the library's reading:

* the exact float64 reference of the pair under the compact rule
  (``interface_side_graphs.Reference`` with every number as its member
  stores it): the residual, the verdict and the pass count of the plain
  iteration, the fields the norm reads at their source, and the distance
  to the exact fixed point at exit against ``K``.  The same reference
  under the other rule (every edge as delivered) stops elsewhere: each
  cell with a scatter shows it could tell;
* the marker-side twin: the same pair with the scatter applied inside the
  large member and a plain edge carrying the small field, whose norm reads
  the compact side by construction.  The state, every key of the report
  and every ``_meta`` slot, to the bit wherever the two steps are one
  program (every cell of every push outside the mixed-dtype domain), and
  to rounding elsewhere, for the reasons :func:`_same_solve` gives;
* the float floor from the dtypes and sizes of the two readings.

**What each domain adds** is asserted where it is: a member of a batch
against its own unbatched solve; the base step of a multi-rate graph on
which the group does not fire; a sub-cycled group against the same group
at one rate (the reading is taken once per pass, of the members' fields as
the pass leaves them, the fast member's after its sub-steps); the pass
count of a predictor's guess; the graph a checkpoint brings back -- the
report, the floor's slot or its absence, and the next step.

**The state a solve returns** under the interface norm is the iterate its
loop accepted with every field the norm does not measure whole one plain
pass on (the return rule; ``interface_side_graphs.measured_whole``).  The
source of the edge that expands is read at its source, measured whole and
kept; the source of the gather is recomputed.  So a state is compared with
the reference only on the fields the norm reads at their source (the
accepted iterate), and with the twin everywhere: the twin's step has the
same rule and keeps the same field -- except in the mixed-dtype domain,
where the twin's plain edge casts, its small field is read through a
transform and recomputed, and the edge-mapped pair's is compared with the
iterate the twin accepted (:func:`_kept_by_the_pair_alone`).  A residual
that can only be restated from the state it is of (``run_adaptive``, whose
last half step starts where the harness does not see) is read on a graph
of its own, stepped with the return rule switched off
(``coupled_graphs.accepted_iterate``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import PRECISION_FLOOR_ULPS
from tests.core import coupling_domain_sizes as cs
from tests.core import coupling_domains as cd
from tests.property import coupled_graphs as cg
from tests.property import geometry_graphs as gg
from tests.property import geometry_interface_graphs as gi
from tests.property import interface_side_graphs as sg

#: A predicted pass count is asserted where every estimate the reference's
#: loop compared was at least this factor from the threshold (the residual's
#: float32 rounding is 2 eps / rtol = 2.4e-3 of a tolerance).
EXIT_MARGIN = 1.05
#: ``(loop gain, sign of the round trip)`` of a cell's scenarios, each on
#: its own seed (its own mapping weights and biases).  Two: under ``vmap``
#: a batch with fewer members than the small field has entries.
GAINS = ((0.6, 1.0), (0.3, -1.0))
#: What the biases are multiplied by, step after step, where a domain runs
#: a sequence (a predictor's history, a restart): irregular, so the
#: quadratic predictor's guess is never the fixed point.
MOVES = (1.0, 1.25, 0.875, 1.5, 1.3)


def _c(label, kind, form, schedule, acceleration="none", small="a", **kw) -> cs.Cell:
    return cs.Cell(label, kind, form, schedule, acceleration, small, **kw)


#: The cells of every push: in each domain a two-way pair with the small
#: field on ``a`` and one with it on ``b`` (the member a domain sub-steps,
#: or holds in float64 beside a float32 one) and a pair whose edges both
#: expand or both reduce, the mapping forms and the schedules rotated over
#: the domains; and each stock acceleration once, in four domains (every
#: domain under every acceleration is the slow lane's product).  The
#: 16-bit domains stop at a cap of two passes, as the tie's cells do there.
PUSH = (
    _c("f32", "two-way", "matrix", "gauss-seidel"),
    _c("f32", "two-way", "sparse", "jacobi", small="b"),
    _c("f32", "scatter-only", "sparse-transposed", "gauss-seidel"),
    _c("f64", "two-way", "sparse", "gauss-seidel"),
    _c("f64", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("f64", "gather-only", "matrix", "jacobi"),
    _c("mixed_dtype", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("mixed_dtype", "two-way", "matrix", "jacobi", small="b"),
    _c("mixed_dtype", "scatter-only", "sparse", "jacobi"),
    _c("mixed_dtype", "two-way", "sparse", "gauss-seidel", "fixed", small="b"),
    _c("bfloat16", "two-way", "matrix", "gauss-seidel"),
    _c("bfloat16", "two-way", "sparse", "jacobi", small="b"),
    _c("bfloat16", "gather-only", "sparse-transposed", "gauss-seidel"),
    _c("float16", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("float16", "two-way", "matrix", "jacobi", small="b"),
    _c("float16", "scatter-only", "sparse", "jacobi"),
    _c("vmap", "two-way", "sparse", "gauss-seidel"),
    _c("vmap", "two-way", "matrix", "jacobi", small="b"),
    _c("vmap", "gather-only", "sparse", "gauss-seidel"),
    _c("multi_rate", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("multi_rate", "two-way", "sparse", "jacobi", small="b"),
    _c("multi_rate", "scatter-only", "matrix", "gauss-seidel"),
    _c("multi_rate", "two-way", "matrix", "jacobi", "aitken", small="b"),
    _c("sub_cycled", "two-way", "matrix", "gauss-seidel"),
    _c("sub_cycled", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("sub_cycled", "gather-only", "sparse", "jacobi"),
    _c("sub_cycled", "two-way", "sparse", "gauss-seidel", "iqn-ils"),
    _c("predictors_warm_starts", "two-way", "sparse", "gauss-seidel"),
    _c("predictors_warm_starts", "two-way", "matrix", "jacobi", small="b"),
    _c("predictors_warm_starts", "scatter-only", "sparse-transposed", "jacobi"),
    _c("predictors_warm_starts", "two-way", "sparse-transposed", "gauss-seidel", "iqn-imvj",
       small="b"),
    _c("checkpoint_restart", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("checkpoint_restart", "two-way", "sparse", "jacobi", small="b"),
    _c("checkpoint_restart", "scatter-only", "matrix", "gauss-seidel"),
)
#: The two-way cells whose marker-side twin is compiled on every push: one
#: in each domain, the small field on ``a`` and on ``b`` in turn, and one
#: accelerated (IQN-IMVJ's secant history and its Jacobian carried from
#: step to step under a predictor).  The twins of the others run in the slow
#: lane (:func:`test_the_other_pairs_of_every_push_report_what_their_twins_report`).
TWINNED = (
    _c("f32", "two-way", "matrix", "gauss-seidel"),
    _c("f64", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("mixed_dtype", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("bfloat16", "two-way", "sparse", "jacobi", small="b"),
    _c("float16", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("vmap", "two-way", "matrix", "jacobi", small="b"),
    _c("multi_rate", "two-way", "sparse-transposed", "gauss-seidel"),
    _c("sub_cycled", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("predictors_warm_starts", "two-way", "sparse", "gauss-seidel"),
    _c("predictors_warm_starts", "two-way", "sparse-transposed", "gauss-seidel", "iqn-imvj",
       small="b"),
    _c("checkpoint_restart", "two-way", "sparse", "jacobi", small="b"),
)
assert set(TWINNED) <= set(PUSH)
#: With ``diagnostics=True`` (the spectral keys of the report), on every
#: push: the small field on the member a sub-cycled group sub-steps.  Two
#: graphs compiled with the report's analysis, eight seconds on eight
#: cores; the other domains' are slow.
DIAGNOSED = (
    _c("sub_cycled", "two-way", "sparse", "gauss-seidel", small="b", diagnostics=True),
)
DIAGNOSED_SLOW = (
    _c("f32", "two-way", "sparse-transposed", "gauss-seidel", diagnostics=True),
    _c("f64", "two-way", "matrix", "jacobi", diagnostics=True),
    _c("mixed_dtype", "two-way", "sparse", "gauss-seidel", small="b", diagnostics=True),
    _c("mixed_dtype", "two-way", "matrix", "jacobi", diagnostics=True),
    _c("bfloat16", "two-way", "sparse", "gauss-seidel", diagnostics=True),
    _c("float16", "two-way", "matrix", "jacobi", small="b", diagnostics=True),
    _c("vmap", "two-way", "sparse", "gauss-seidel", diagnostics=True),
    _c("multi_rate", "two-way", "matrix", "gauss-seidel", small="b", diagnostics=True),
    _c("sub_cycled", "two-way", "sparse-transposed", "jacobi", diagnostics=True),
    _c("predictors_warm_starts", "two-way", "sparse", "jacobi", small="b", diagnostics=True),
    _c("checkpoint_restart", "two-way", "sparse", "gauss-seidel", diagnostics=True),
)

#: The slow lane's product: every domain, every stock acceleration, both
#: schedules, and the four pairs -- two ways with the small field on either
#: member, both edges expanding, both reducing -- with the mapping form
#: rotated over them.  This file's own tuples: no other table's length
#: enters a cell.
LABELS = ("f32", "f64", "mixed_dtype", "bfloat16", "float16", "vmap", "multi_rate",
          "sub_cycled", "predictors_warm_starts", "checkpoint_restart")
ACCELERATIONS = ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")
SCHEDULES = ("gauss-seidel", "jacobi")
PAIRS = (("two-way", "a"), ("two-way", "b"), ("scatter-only", "a"), ("gather-only", "a"))
FORMS = ("matrix", "sparse", "sparse-transposed")


def _product(labels, pairs=PAIRS) -> list:
    cells = []
    for i, label in enumerate(labels):
        for j, acceleration in enumerate(ACCELERATIONS):
            for k, schedule in enumerate(SCHEDULES):
                for m, (kind, small) in enumerate(pairs):
                    cells.append(_c(label, kind, FORMS[(i + j + k + m) % len(FORMS)], schedule,
                                    acceleration, small))
    return cells


PRODUCT = _product(LABELS)
#: ``run_adaptive`` compiles its step on every call: the two-way pairs only.
ADAPTIVE = _product((cd.ADAPTIVE,), PAIRS[:2])
#: Larger fields (the report's Krylov steps do not resolve these loops; the
#: criterion, the floor and the twin do not ask them to).
LARGER = (
    _c("f32", "two-way", "sparse", "gauss-seidel", n_small=6, n_large=600),
    _c("f64", "two-way", "matrix", "jacobi", small="b", n_small=6, n_large=600),
    _c("mixed_dtype", "two-way", "sparse-transposed", "jacobi", small="b", n_small=20,
       n_large=2000),
    _c("vmap", "two-way", "sparse", "jacobi", n_small=20, n_large=2000),
    _c("multi_rate", "scatter-only", "sparse", "gauss-seidel", n_small=20, n_large=2000),
    _c("sub_cycled", "two-way", "sparse", "gauss-seidel", small="b", n_small=20, n_large=2000),
    _c("predictors_warm_starts", "two-way", "sparse-transposed", "gauss-seidel", n_small=20,
       n_large=2000),
    _c("checkpoint_restart", "gather-only", "sparse", "jacobi", n_small=20, n_large=2000),
    _c("bfloat16", "two-way", "sparse", "gauss-seidel", small="b", n_small=6, n_large=600),
    _c("float16", "two-way", "matrix", "jacobi", n_small=6, n_large=600),
)
#: The sharded domain's cells (run from ``tests/cloud/multigpu``): member
#: ``b`` over four devices, every field four copies, the dense kind (four
#: diagonal copies of the matrix).  With the small field on ``b`` the norm
#: reads a sharded field at its source; with it on ``a`` the expanding edge
#: delivers onto the sharded one.
SHARDED = (
    _c(cd.SHARDED, "two-way", "matrix", "gauss-seidel"),
    _c(cd.SHARDED, "two-way", "matrix", "jacobi", small="b"),
    _c(cd.SHARDED, "two-way", "matrix", "gauss-seidel", "aitken", small="b"),
    _c(cd.SHARDED, "two-way", "matrix", "jacobi", "iqn-ils"),
)
#: A ``vmap`` batch with more members than the large field has entries.
BATCH = 40


def _ids(cells):
    return [c.id for c in cells]


def _plain(cells):
    return [c for c in cells if c.acceleration == "none"]


def _two_way(cells):
    return [c for c in cells if c.kind == "two-way"]


# ---------------------------------------------------------------------------
# Running a cell
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Run:
    built: cs.Built
    #: One reference per solve.
    refs: list
    solves: list
    #: The restart domain's: the uninterrupted solves, and the graph as the
    #: checkpoint was written and as the load left it.
    straight: list
    saved: object = None
    loaded: object = None


_RUNS: dict = {}


def _sequenced(cell) -> bool:
    return cell.domain.predictor or cell.domain.restart


#: The steps of a sequence whose pass count must be predicted, at least.
PREDICTED_STEPS = 2


def _chain_margins(cell, draw) -> list:
    """The margins of a sequence's steps on the reference's own chain: each
    step started where the reference's last one stopped (through the
    predictor's guess where the domain has one).  The graph's steps start
    from its own states, which are these to rounding; the margins of the
    steps the domain checks (a restart's: those after the checkpoint)."""
    states, margins = [], []
    for k, move in enumerate(MOVES):
        ref = cs.Stored(cell, draw, scale=move)
        before = states[-1] if states else ref.start
        start = _guess(k, states, before) if cell.domain.predictor else before
        exit_ = ref.started_at(start).plain_exit("compact")
        states.append(exit_["state"])
        margins.append(exit_["margin"])
    return margins[2:] if cell.domain.restart else margins


def _decided(cell, draw) -> bool:
    """Is *draw*'s exit decided with margin (in the reference's own float64
    arithmetic: the same answer on every platform)?  From the graph's
    initial state; and, for a sequence, on :data:`PREDICTED_STEPS` of the
    steps the domain checks, with a little to spare for the graph's own
    rounding of the states they start from."""
    if cell.sixteen or cell.acceleration != "none":
        return True
    if cs.Stored(cell, draw).plain_exit("compact")["margin"] < EXIT_MARGIN:
        return False
    if not _sequenced(cell):
        return True
    spare = 1.01 * EXIT_MARGIN
    return sum(m >= spare for m in _chain_margins(cell, draw)) >= PREDICTED_STEPS


def _draws(cell, count=None) -> list:
    """The cell's scenarios: for each loop gain the first seed whose exit
    under the compact rule is decided with margin (:func:`_decided`).  A
    sequence runs the first; *count* asks for more scenarios than
    :data:`GAINS` has."""
    gains = GAINS if count is None else [GAINS[i % len(GAINS)] for i in range(count)]
    gains = gains[:1] if _sequenced(cell) else gains
    out, seed = [], 0
    for gain, sign in gains:
        while True:
            draw = sg.Draw(seed, gain, sign)
            seed += 1
            if _decided(cell, draw):
                break
            assert seed < 60 * len(gains), f"{cell.id}: no seed decides its exit with margin"
        out.append(draw)
    return out


def _run(cell, twin: bool = False, count=None, accepted: bool = False) -> Run:
    """The cell's graph (or its twin's), built and stepped once per session.

    *accepted*: a graph of its own, built and stepped with the interface
    norm's return rule switched off (``coupled_graphs.accepted_iterate``),
    so each solve returns the iterate its loop accepted -- the state its
    report is of.  Not for a sequence, whose later steps start from the
    states the earlier ones returned.
    """
    key = (cell, twin, count, accepted)
    if key in _RUNS:
        return _RUNS[key]
    d = cell.domain
    assert not (accepted and _sequenced(cell)), cell
    with contextlib.ExitStack() as stack:
        asked = stack.enter_context(cg.accepted_iterate()) if accepted else None
        stack.enter_context(cd.entered(d))
        draws = _draws(cell, count)
        if _sequenced(cell):
            refs = [cs.Stored(cell, draws[0], scale=move) for move in MOVES]
        else:
            refs = [cs.Stored(cell, draw) for draw in draws]
        built = cs.build_twin(cell) if twin else cs.build(cell)
        params = [cs.params_of(ref, built) for ref in refs]
        straight, saved, loaded = [], None, None
        if d.restart:
            straight, solves, saved, loaded = cs.restart_pairs(built, params)
            refs = refs[len(refs) - len(solves):]
        elif d.predictor:
            solves = cd.run_sequence(d, built.gm, params)
        else:
            solves = cd.run(d, built.gm, params)
        cs.assert_sized(built, solves)
    if accepted:
        assert asked, "the step never asked the return rule: the patch is on the wrong name"
    _RUNS[key] = Run(built, refs, solves, straight, saved, loaded)
    return _RUNS[key]


def _forget(cell) -> None:
    for key in [k for k in _RUNS if k[0] == cell]:
        del _RUNS[key]


def _eps(dtype) -> float:
    return float(cd.finfo(dtype).eps)


def _resolution(cell) -> float:
    """What a reported residual resolves, in tolerances: ``4 eps / rtol`` at
    the coarsest member's eps.

    Each of the two values a reading differences is the rounded result of
    a few operations (a gather's two products and their sum, a gain, a
    bias): up to two eps of its magnitude, so their difference carries up
    to four, ``4 eps / rtol`` in the norm's units whatever the change.
    That is the float floor the library documents for one evaluation
    (``PRECISION_FLOOR_ULPS``).  Measured: 2.5 ``eps / rtol`` on one
    float32 scenario in forty (the residual reads 0.1730 where the same
    graph in float64 and the reference read 0.1700), 0.6 on the others.
    """
    return 4.0 * _eps(cell.domain.coarsest) / cell.rtol


def _guess(k: int, returned: list, before: dict) -> dict:
    """The iterate step *k* of a sequence starts from under
    ``predictor="quadratic"``, as the group documents it, from the states
    the earlier steps *returned*: none until two are stored (the state
    *before* the step), linear ``2 x_n - x_{n-1}`` with two, quadratic
    ``3 x_n - 3 x_{n-1} + x_{n-2}`` from the fourth step on."""
    if k < 2:
        return before
    if k == 2:
        return {n: 2.0 * returned[1][n] - returned[0][n] for n in ("p", "q")}
    return {n: 3.0 * returned[k - 1][n] - 3.0 * returned[k - 2][n] + returned[k - 3][n]
            for n in ("p", "q")}


def _starts(cell, run: Run) -> list:
    """The iterate each solve's loop started from, as the reference names it.

    The state before the step, or a predictor's guess (:func:`_guess`)
    from the states the earlier steps returned.  ``None`` under
    ``run_adaptive``, whose reported solve is its last half step's,
    started where the harness does not see.
    """
    if cell.domain.adaptive:
        return [None] * len(run.solves)
    before = [cs.state_of(cell, s, pre=True) for s in run.solves]
    if not cell.domain.predictor:
        return before
    returned = [cs.state_of(cell, s) for s in run.solves]
    return [_guess(k, returned, before[k]) for k in range(len(run.solves))]


def _source_read(cell) -> tuple:
    """The reference's nodes whose field the compact rule reads at its source."""
    return {"two-way": ("p",), "scatter-only": ("p", "q"), "gather-only": ()}[cell.kind]


# ---------------------------------------------------------------------------
# The checks, each over one cell
# ---------------------------------------------------------------------------
def _check_reference(cell, count=None) -> None:
    """The plain iteration of *cell* against the exact reference under the compact rule."""
    assert cell.acceleration == "none", cell
    run = _run(cell, count=count)
    d = cell.domain
    resolution = _resolution(cell)
    predicted, told_apart = 0, False
    # run_adaptive: the iterate each reported solve accepted, on a graph of
    # its own with the return rule switched off.
    at_accepted = _run(cell, count=count, accepted=True).solves if d.adaptive else None
    assert at_accepted is None or len(at_accepted) == len(run.solves)
    for k, (s, ref, start) in enumerate(zip(run.solves, run.refs, _starts(cell, run))):
        r = s.report
        x = cs.state_of(cell, s)
        where = f"{cell.id}, solve {k}"
        reported = r["residual"]
        if start is None:
            # run_adaptive: the claim, of the state it returns; and the
            # residual, which is of the iterate the loop accepted.  The
            # returned state is that iterate with the gather's source one
            # plain pass on and does not determine it, and the reference's
            # own loop cannot be started where the last half step was.  So
            # the residual is restated from the state a step returns with
            # the return rule switched off, which is its accepted iterate.
            assert r["converged"] is True, (where, r)
            assert ref.distance(x, "compact") <= ref.K, (where, ref.distance(x), ref.K)
            z = at_accepted[k]
            assert z.report["converged"] is True, (where, z.report)
            xz, reported = cs.state_of(cell, z), z.report["residual"]
            want = ref.in_tolerances(ref.residual(ref.one_pass(xz), xz, "compact"))
            other = ref.in_tolerances(ref.residual(ref.one_pass(xz), xz, "delivered"))
            iterate = None
        else:
            exit_ = ref.started_at(start).plain_exit("compact")
            other_exit = ref.plain_exit("delivered")
            iterate = exit_["state"]
            if cell.sixteen:
                # Stopped at the cap, far from converged: the residual is that
                # of the iterate the loop holds, over one more pass.
                assert r["converged"] is False and r["iterations"] == cs.CAP_16, (where, r)
                want = ref.in_tolerances(ref.residual(ref.one_pass(iterate), iterate, "compact"))
                other = ref.in_tolerances(
                    ref.residual(ref.one_pass(iterate), iterate, "delivered"))
                assert want > 1.0 + resolution, f"{where}: fixture premise: outside the tolerance"
            else:
                assert exit_["converged"], f"{where}: fixture premise: the reference converges"
                assert r["converged"] is True, (where, r)
                want, other = exit_["residual"], other_exit["residual"]
                if exit_["margin"] >= EXIT_MARGIN:
                    predicted += 1
                    assert r["iterations"] == exit_["iterations"], (
                        f"{where}: the step took {r['iterations']} passes; the plain loop "
                        f"on the compact readings stops after {exit_['iterations']} (margin "
                        f"{exit_['margin']:.3g}; read as delivered, after "
                        f"{other_exit['iterations']})")
                told_apart |= other_exit["iterations"] != exit_["iterations"]
                distance = ref.distance(x, "compact")
                assert distance <= ref.K, (
                    f"{where}: converged=True at {distance:.3f} tolerances from the fixed "
                    f"point in the compact readings; K = {ref.K:.3f}")
        assert abs(reported - want) <= resolution, (
            f"{where}: reported residual {reported!r}; the compact readings give "
            f"{want!r} (resolution {resolution:.2e}; read as delivered: {other!r})")
        told_apart |= abs(other - want) > 2.0 * resolution
        for name in _source_read(cell) if iterate is not None else ():
            # A field the norm reads whole is the accepted iterate.  The
            # passes are evaluated in the members' dtypes, each value of
            # the loop passing through both: a few eps of the coarser a
            # pass, which a contraction sums to at most K of them.
            bound = 16.0 * _eps(d.coarsest) * (1.0 + ref.K) * np.max(np.abs(iterate[name]))
            worst = float(np.max(np.abs(x[name] - iterate[name])))
            assert worst <= bound, (
                f"{where}: {cell.names[name]}.x is {worst:.3e} from the reference's iterate "
                f"(bound {bound:.3e})")
    if cell.kind != "gather-only":
        assert told_apart, (
            f"{cell.id}: fixture premise: the reference read as delivered stops where the "
            f"compact one does, to the residual's resolution")
    if not (cell.sixteen or d.adaptive):
        # Every scenario started from the graph's initial state was chosen
        # with margin; so were some of a sequence's steps (``_decided``).
        need = PREDICTED_STEPS if _sequenced(cell) else len(run.solves)
        assert predicted >= need, f"{cell.id}: {predicted} pass counts predicted, of {need}"


def _numbers_agree(a, b, rel: float) -> bool:
    if isinstance(a, (bool, np.bool_, str, type(None))) or isinstance(b, (str, type(None))):
        return a == b
    a, b = float(a), float(b)
    if (math.isnan(a) and math.isnan(b)) or a == b:
        return True
    return rel > 0.0 and abs(a - b) <= rel * max(abs(a), abs(b))


def _one_program(cell) -> bool:
    """Do the edge-mapped pair and its twin evaluate the same numbers in the
    same order?

    Not in the mixed-dtype domain (:func:`_same_solve`).  Elsewhere, at
    the sizes of every push, where the scatter is held in its natural
    layout (the mapping and the twin's node are the same scatter-add), or
    no cell of the large field takes the contributions of two markers
    (every sum the scatter makes has one term, in any order).  A dense
    matrix or the other layout sums two markers' contributions to one cell
    in an order of its own, and the passes then differ in the last bit.
    And not with more than three values a reading: the two graphs' states
    can still agree to the bit, but each compiles its own sum of a
    reading's squares, and with twenty terms the two float32 sums of the
    same numbers were measured a bit apart on an earlier pass (the
    amplification 2.3629119 against 2.3629122).
    """
    if cell.domain.dtype_a != cell.domain.dtype_b or cell.n_small > cs.N_SMALL:
        return False
    cells = sg.edges_of(cell.shape)[0].cells
    touched = np.concatenate([cells, cells + 1])
    return cell.form == "sparse" or len(np.unique(touched)) == len(touched)


def _kept_by_the_pair_alone(cell, mapped: Run, twin: Run) -> tuple:
    """The members whose field the edge-mapped pair returns as its accepted
    iterate holds it, and its twin one plain pass on.

    Under the interface norm a solve returns a field the norm measures
    whole as the accepted iterate holds it and every other field one plain
    pass on (``interface_side_graphs.measured_whole``, restated from each
    graph's own edges).  The edge-mapped pair reads its small field at its
    source, before the mapping and the cast: measured whole, kept.  The
    twin reads it by a plain edge, which measures it whole too -- except
    in the mixed-dtype domain, where that edge casts: the field is read
    through a transform, and recomputed.  The large field is read through
    the gather in both graphs and recomputed in both.
    """
    members = set(cell.names.values())
    kept = set(sg.measured_whole(mapped.built.gm)) & members
    kept_by_twin = set(sg.measured_whole(twin.built.gm)) & members
    d = cell.domain
    assert kept == {cell.names["p"]}, (cell.id, kept)
    assert kept_by_twin == (set() if d.dtype_a != d.dtype_b else kept), (cell.id, kept_by_twin)
    return tuple(sorted(kept - kept_by_twin))


def _same_solve(cell, mapped, twin, what: str, twin_accepted=None, apart=()) -> None:
    """*mapped* (the edge-mapped graph's) is *twin*'s: the state, the report, ``_meta``.

    A field in *apart* (:func:`_kept_by_the_pair_alone`) is the accepted
    iterate's in the edge-mapped graph and one plain pass on in the twin:
    it is compared with *twin_accepted*'s, the same solve of the twin
    stepped with the return rule switched off -- the iterate the twin
    accepted -- to the bound the two returned fields were held to.

    To the bit where the two steps are one program (:func:`_one_program`):
    the same passes on the same numbers, and the same values read.

    Otherwise to rounding: the verdict and the pass count the same, the
    state to 64 eps of the coarser member, the residual and the estimates
    made of it to the residual's resolution or 2% (the numbers derived
    from residual ratios: ``test_coupling_interface_side.TWIN_RTOL`` has 2%
    for float32).  **The mixed-dtype domain is never one program**: the
    twin's plain edge casts the small field to the large member's dtype
    and the norm reads the cast value (a float32 field widened to float64
    and squared there; a float64 field rounded to float32, whose changes
    below float32's resolution it does not see), while the edge-mapped
    graph reads the stored field itself; and with the small field in
    float64 the twin scatters its float32 rounding where the mapping
    scatters the float64 value and rounds the sum.  The floor's slot is
    compared wherever both graphs count the same eps: not with the small
    field in float64 beside a float32 member, where the twin's cast
    reading is counted at float32's eps (:func:`_floor` states the
    edge-mapped graph's).
    """
    d = cell.domain
    exact = _one_program(cell)
    where = f"{cell.id}, {what}"
    assert not (exact and apart), (where, apart)
    for name in ("a", "b"):
        xa, xb = mapped.x(name), twin.x(name)
        of = "the twin's"
        if name in apart:
            xb, of = twin_accepted.x(name), "the iterate the twin accepted"
        if exact:
            assert cd.bitwise(xa, xb), f"{where}: {name}.x differs from the twin's"
        else:
            xa, xb = xa.astype(np.float64), xb.astype(np.float64)
            bound = 64.0 * _eps(d.coarsest) * np.max(np.abs(xb))
            assert np.max(np.abs(xa - xb)) <= bound, (
                f"{where}: {name}.x is {np.max(np.abs(xa - xb)):.3e} from {of}")
    ra, rb = mapped.report, twin.report
    assert sorted(ra) == sorted(rb), (where, sorted(ra), sorted(rb))
    differ = {}
    for key in ra:
        same = _numbers_agree(ra[key], rb[key], 0.0 if exact else 2e-2)
        if not same and not exact and key in ("residual", "error_estimate",
                                              "gradient_error_estimate"):
            same = abs(float(ra[key]) - float(rb[key])) <= _resolution(cell)
        if not same:
            differ[key] = (ra[key], rb[key])
    assert not differ, f"{where}: (edge-mapped, twin) {differ}"
    assert sorted(mapped.meta) == sorted(twin.meta), (where, sorted(mapped.meta))
    if exact:
        slots = [key for key in mapped.meta if not cd.bitwise(mapped.meta[key], twin.meta[key])]
        assert not slots, f"{where}: _meta slots differ from the twin's: {slots}"
    elif d.dtype_a == d.dtype_b or cell.small == "a":
        assert cd.bitwise(cs.slot_of(mapped), cs.slot_of(twin)), (
            where, cs.slot_of(mapped), cs.slot_of(twin))


def _check_twin(cell, count=None) -> None:
    """The edge-mapped pair reports what its marker-side twin reports."""
    mapped, twin = _run(cell, count=count), _run(cell, twin=True, count=count)
    apart = _kept_by_the_pair_alone(cell, mapped, twin)
    accepted = (_run(cell, twin=True, count=count, accepted=True).solves if apart
                else [None] * len(twin.solves))
    assert len(mapped.solves) == len(twin.solves) == len(accepted) > 0
    for k, (s, t, z) in enumerate(zip(mapped.solves, twin.solves, accepted)):
        _same_solve(cell, s, t, f"solve {k}", z, apart)
    if cell.domain.restart:
        assert not apart, cell
        _same_solve(cell, mapped.saved, twin.saved, "the checkpointed graph")
        _same_solve(cell, mapped.loaded, twin.loaded, "the loaded graph")


def _check_claim(cell, count=None) -> None:
    """CPL-192 on *cell*: every solve converges, within ``K`` tolerances."""
    run = _run(cell, count=count)
    for k, (s, ref) in enumerate(zip(run.solves, run.refs)):
        assert s.report["converged"] is True, (cell.id, k, s.report)
        distance = ref.distance(cs.state_of(cell, s), "compact")
        assert distance <= ref.K, (
            f"{cell.id}, solve {k}: converged=True at {distance:.3f} tolerances from the "
            f"fixed point in the compact readings; K = {ref.K:.3f}")


def _floor(cell) -> float:
    """The float floor of the pair's residual per evaluation, from the rule:
    ``4 eps / rtol`` pooled over the entries the norm reads -- an edge that
    expands at its source field's entries and that field's own eps, any
    other at the entries it delivers and the coarser of the delivered
    dtype's eps and its source's."""
    d, shape = cell.domain, cell.shape
    dtype = {sg_name: (d.dtype_a if name == "a" else d.dtype_b)
             for sg_name, name in cell.names.items()}
    sizes = sg.node_sizes(shape)
    total = count = 0.0
    for e in sg.edges_of(shape):
        n_source, n_delivered = sizes[e.src][0], sizes[e.dst][1]
        if sg.side_of(n_source, n_delivered) == "source":
            n, eps = n_source, _eps(dtype[e.src])
        else:
            n, eps = n_delivered, max(_eps(dtype[e.dst]), _eps(dtype[e.src]))
        total += n * eps * eps
        count += n
    return PRECISION_FLOOR_ULPS * math.sqrt(total / count) / cell.rtol


def _check_slot(cell, count=None) -> None:
    """Only a pair with an edge read through its mapping owns the floor's
    slot, and it holds the floor of the compact readings."""
    run = _run(cell, count=count)
    key = f"coupling_{cd.KEY}_reading_floor"
    seen = list(run.solves) + list(run.straight) + [s for s in (run.saved, run.loaded) if s]
    for s in seen:
        if cell.kind == "scatter-only":
            assert key not in s.meta, (
                f"{cell.id}: every mapped edge is read at its source, and the group "
                f"holds {key} = {s.meta[key]!r}")
            continue
        assert key in s.meta, f"{cell.id}: a gather is read through its mapping; no {key}"
        assert float(s.meta[key]) == pytest.approx(_floor(cell), rel=1e-6), (
            f"{cell.id}: the slot holds {float(s.meta[key])!r}; the compact readings' "
            f"floor is {_floor(cell)!r}")


def _slots_agree(a, b, rel: float) -> bool:
    """Two ``_meta`` slots: a floating one to *rel* of its largest entry
    (NaN beside NaN), any other to the bit."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind != "f" and a.dtype.kind != "V":
        return cd.bitwise(a, b)
    a, b = a.astype(np.float64), b.astype(np.float64)
    scale = max(float(np.max(np.abs(np.nan_to_num(a)), initial=0.0)),
                float(np.max(np.abs(np.nan_to_num(b)), initial=0.0)))
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b) | (np.abs(a - b) <= rel * scale)))


def _same_graph(a, b) -> list:
    """What differs between two snapshots of one graph: members (every
    field each holds), report, ``_meta``."""
    out = [f"{n}.{f}" for n in ("a", "b") for f in sorted(set(a.state[n]) | set(b.state[n]))
           if f not in a.state[n] or f not in b.state[n]
           or not cd.bitwise(a.state[n][f], b.state[n][f])]
    out += [f"report[{k}]" for k in sorted(set(a.report) | set(b.report))
            if k not in a.report or k not in b.report
            or not _numbers_agree(a.report[k], b.report[k], 0.0)]
    out += [f"_meta[{k}]" for k in sorted(set(a.meta) | set(b.meta))
            if k not in a.meta or k not in b.meta or not cd.bitwise(a.meta[k], b.meta[k])]
    return out


def _group_differs(a, b) -> list:
    """:func:`_same_graph` without what is the graph's and not the group's:
    a multi-rate graph counts its base steps (``_meta["step_count"]``), and
    a group that sub-cycles reports its sweeps' total beside the largest."""
    return [w for w in _same_graph(a, b)
            if w != "_meta[step_count]" and "total_iterations" not in w]


def _check_restart(cell) -> None:
    """A checkpoint brings back the report, the floor's slot (or its
    absence) and the next step of the uninterrupted run."""
    run = _run(cell)
    assert run.saved is not None and run.straight, cell.id
    assert not _same_graph(run.saved, run.loaded), (
        f"{cell.id}: the loaded graph differs from the checkpointed one: "
        f"{_same_graph(run.saved, run.loaded)}")
    for k, (a, b) in enumerate(zip(run.straight, run.solves)):
        assert not _same_graph(a, b), (
            f"{cell.id}: step {k} after the restart differs from the uninterrupted run's: "
            f"{_same_graph(a, b)}")


def _same_member(where: str, member, alone, ulps: float = 0.0, numbers: float = 1e-5) -> None:
    """A member of a ``vmap`` batch is its own unbatched solve.

    The state, the verdict, the pass count and every integer slot to the
    bit.  The report's float numbers to 1e-5: a batched reduction sums a
    reading's squares in another order than the unbatched one, so a
    residual of an earlier pass can differ in its last bit and the
    amplification made of three of them with it (measured 5e-7 with
    twenty values a reading; the cells of every push agree to the bit).
    A member that kept iterating with its batch, or stopped with it,
    would differ by passes, not by bits.

    *ulps* (the pairs with geometry edges): the state to that many ``eps``
    of each field's largest entry instead of to the bit, the verdict and
    the pass count still exact.  The batched step and the unbatched one
    are two programs, and a member that adds a product to a bias, or an
    edge that sums two weighted samples, is compiled with a fused
    multiply-add in one and not in the other (measured: one float32 ulp
    of one entry of ``x`` on the two gather-only pairs of the seven
    ``vmap`` cells, the other five to the bit; jaxlib 0.11.0, CPU; the
    same pass counts and the same residual to the bit).
    """
    for name in ("a", "b"):
        assert sorted(member.state[name]) == sorted(alone.state[name]), where
        for field in member.state[name]:
            got, want = member.state[name][field], alone.state[name][field]
            if not ulps:
                assert cd.bitwise(got, want), (
                    f"{where}: {name}.{field} differs from its unbatched solve's")
                continue
            assert got.dtype == want.dtype and got.shape == want.shape, (where, name, field)
            bound = ulps * float(cd.finfo(want.dtype).eps) * float(np.max(np.abs(want)))
            worst = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
            assert worst <= bound, (
                f"{where}: {name}.{field} is {worst:.3e} from its unbatched solve's "
                f"(bound {bound:.3e})")
    for key in ("converged", "iterations"):
        assert member.report[key] == alone.report[key], (where, key)
    assert sorted(member.report) == sorted(alone.report), where
    assert sorted(member.meta) == sorted(alone.meta), where
    differ = {key: (member.report[key], alone.report[key]) for key in member.report
              if not _numbers_agree(member.report[key], alone.report[key], numbers)}
    differ.update({key: (member.meta[key], alone.meta[key]) for key in member.meta
                   if not _slots_agree(member.meta[key], alone.meta[key], numbers)})
    assert not differ, f"{where} differs from its unbatched solve: {differ}"


def _check_batch(cell, count=None) -> None:
    """Each member of the ``vmap`` domain's batch is its own unbatched solve."""
    run = _run(cell, count=count)
    gm = run.built.gm
    passes = set()
    with cd.entered(cell.domain):
        for k, s in enumerate(run.solves):
            (alone,) = cd.run(cd.DOMAINS["f32"], gm, [s.params])
            # To a few roundings, not to the bit: the batched step and the
            # unbatched one are two compiled programs, and which of them
            # fuses a multiply-add depends on the processor as well (CPL-096
            # claims "a few ulps").  A slow-lane runner returned one float32
            # ulp of one entry on the sparse two-way pair under jaxlib 0.10.2
            # where every other run had the pair to the bit.  The verdict
            # and the pass count stay exact.  The report's numbers follow
            # the state: a rounding of an entry is ``eps / rtol`` of a
            # tolerance in the residual (0.29356 for 0.29346 on that
            # runner), so they are held to that many roundings as the
            # residual resolves them (:func:`_resolution`), not to 1e-5.
            _same_member(f"{cell.id}: member {k} of the batch", s, alone, ulps=4.0,
                         numbers=max(1e-5, 4.0 * _resolution(cell)))
            passes.add(s.report["iterations"])
    if cell.acceleration == "none":
        assert len(passes) > 1, (
            f"{cell.id}: fixture premise: the members of the batch stop on different passes")


def _check_batched_floor(cell) -> None:
    """A ``vmap`` of the mixed-dtype pair records each member's own floor.

    The solve's loop is traced on unbatched values whatever wraps the
    step, so its residual cannot see a batch; the floor is measured after
    the loop, on the batched state.  With the small field in float64
    beside a float32 member the two sides of the expanding edge give two
    floors (:func:`_floor`: ``4 eps32 / (sqrt(2) rtol)`` at the source,
    ``4 eps32 / rtol`` as delivered), so this batch shows which side was
    read there: each member's slot is the compact readings', and the
    member is its own unbatched solve.
    """
    d = cell.domain
    assert d.dtype_a != d.dtype_b and cell.small == "b" and not d.vmap, cell
    run = _run(cell)
    with cd.entered(d):
        batch = cd.run(cd.DOMAINS["vmap"], run.built.gm, [s.params for s in run.solves])
    assert len(batch) == len(run.solves) > 1
    for k, (member, alone) in enumerate(zip(batch, run.solves)):
        _same_member(f"{cell.id}: member {k} of a batch", member, alone)
        assert float(cs.slot_of(member)) == pytest.approx(_floor(cell), rel=1e-6), (
            f"{cell.id}: member {k} of a batch holds the floor {float(cs.slot_of(member))!r}; "
            f"the compact readings' is {_floor(cell)!r}")


def _check_one_rate(cell) -> None:
    """A sub-cycled or a multi-rate pair is the same pair at one rate, to the bit.

    The members are memoryless, so a pass is the same map whatever the
    step: the reading is taken once per pass, of the members' fields as
    the pass leaves them -- the fast member's after its two sub-steps --
    and nothing of the report, the state or the floor's slot moves.
    """
    run, plain = _run(cell), _run(dataclasses.replace(cell, label="f32"))
    for k, (s, t) in enumerate(zip(run.solves, plain.solves)):
        assert not _group_differs(s, t), (
            f"{cell.id}: solve {k} differs from the one-rate pair's: {_group_differs(s, t)}")


def _check_firing(cell) -> None:
    """A multi-rate graph reads the pair on the base steps its group fires on.

    The clock outside the group runs at half the group's timestep, so the
    group solves on every other base step.  On the step between, the
    members, the report and every ``_meta`` slot of the group -- the
    floor's among them -- are the last solve's, to the bit.
    """
    run = _run(cell)
    gm, p = run.built.gm, run.solves[0].params
    with cd.entered(cell.domain):
        gm.reset_state()
        shots = []
        for _ in range(6):
            gm.step(params=p)
            shots.append(cd._solve(gm, p))  # noqa: SLF001
    fired = [bool(_group_differs(a, b)) for a, b in zip(shots, shots[1:])]
    # The group solves on the first base step and on every second one after
    # it (each later solve starts at the last one's fixed point and takes
    # another pass, so its report differs).
    assert fired == [False, True, False, True, False], (cell.id, fired)
    assert not _same_graph(shots[1], run.solves[0]), (
        f"{cell.id}: the pair of base steps the harness reads holds another solve: "
        f"{_same_graph(shots[1], run.solves[0])}")


def _check_diagnosed(cell) -> None:
    """The spectral analysis reads the side the criterion reads.

    The twin's report holds the same spectral radius and the same bound
    (to the rounding of two analyses of one problem, as
    ``test_the_interface_norm_reads_a_mapped_edge_on_its_compact_side``
    has them; to 2% where the two are not one program, :func:`_same_solve`), and
    a usable bound covers the distance to the exact fixed point in the
    compact readings.  (How loose it may be is not held here: measured
    2.0 to 2.8 ``K`` residuals, the same number in both graphs.)

    ``gradient_relative_error_bound`` is compared where the two graphs hold
    the scatter's weights the same way: the sparse kind in its natural
    layout, one weight per entry in the mapping as in the twin's node.
    The bound takes one probe per entry of every floating constant of at
    most 64 entries and one probe of a larger constant as a whole
    (``_bounds.GRADIENT_PROBE_ENTRY_LIMIT``), so a dense matrix or the
    other layout (108 slots here, most of them zeros) is probed along one
    direction where the twin's six weights are probed one by one, and the
    two bounds of one problem differ (measured: 1.27e-4 against 1.41e-4).
    That is the probe plan's, on either side of the norm.
    """
    mapped, twin = _run(cell), _run(cell, twin=True)
    skipped = () if cell.form == "sparse" else ("gradient_relative_error_bound",)
    usable = 0
    for k, (s, t, ref) in enumerate(zip(mapped.solves, twin.solves, mapped.refs)):
        ra, rb = s.report, t.report
        where = f"{cell.id}, solve {k}"
        assert ra["converged"] is not cell.sixteen, (where, ra)
        differ = {key: (ra[key], rb[key]) for key in ra if key not in skipped
                  and not _numbers_agree(ra[key], rb[key], 1e-6 if _one_program(cell) else 2e-2)}
        assert sorted(ra) == sorted(rb) and not differ, f"{where}: (edge-mapped, twin) {differ}"
        if not ra["spectral_usable"]:
            continue
        usable += 1
        distance = ref.in_tolerances(ref.distance(cs.state_of(cell, s), "compact"))
        bound = float(ra["spectral_error_bound"])
        assert 0.0 < distance <= bound * (1.0 + 1e-6), (where, distance, bound)
    assert usable, f"{cell.id}: fixture premise: no solve reports a usable spectrum"


def check_everything(cell) -> None:
    """Every check a cell admits."""
    d = cell.domain
    if cell.acceleration == "none":
        _check_reference(cell)
    elif not cell.sixteen:
        _check_claim(cell)
    _check_slot(cell)
    if cell.kind == "two-way" and not (d.sharded and cell.small == "a"):
        # (The twin's large member applies a scatter: not a pointwise node,
        # so it cannot be the sharded one.)
        _check_twin(cell)
    if d.restart:
        _check_restart(cell)
    if d.vmap:
        _check_batch(cell)
    if d.multirate and cell.acceleration == "none":
        _check_firing(cell)


# ---------------------------------------------------------------------------
# On every push
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell", _plain(PUSH), ids=_ids(_plain(PUSH)))
def test_a_pair_of_two_sizes_stops_where_the_compact_reference_does(cell):
    """CPL-188 in every domain: the residual, the verdict and the pass count
    of the plain iteration are the exact reference's under the compact
    rule, the fields read at their source are its iterate, and a converged
    pair is within ``K`` tolerances (CPL-192).  In the 16-bit domains, at
    the cap: the residual of the iterate the loop holds."""
    _check_reference(cell)


@pytest.mark.parametrize("cell", TWINNED, ids=_ids(TWINNED))
def test_a_pair_of_two_sizes_reports_what_its_marker_side_twin_reports(cell):
    """The state, every key of the report and every ``_meta`` slot of the
    edge-mapped pair are those of the pair whose large member applies the
    scatter itself, in every domain, plain and accelerated."""
    _check_twin(cell)


_ACCELERATED = [c for c in PUSH if c.acceleration != "none"]


@pytest.mark.parametrize("cell", _ACCELERATED, ids=_ids(_ACCELERATED))
def test_a_converged_accelerated_pair_of_two_sizes_is_within_K_tolerances(cell):
    """CPL-192 under each stock acceleration, in the domains that run them."""
    _check_claim(cell)


@pytest.mark.parametrize("cell", PUSH, ids=_ids(PUSH))
def test_only_a_pair_with_an_edge_read_through_its_mapping_owns_the_floor_slot(cell):
    """CPL-188's consequence in every domain: a pair with a gather records
    the floor of its compact readings; a pair whose edges both expand
    records none."""
    _check_slot(cell)


_BATCHED = [c for c in PUSH if c.domain.vmap]


@pytest.mark.parametrize("cell", _BATCHED, ids=_ids(_BATCHED))
def test_each_member_of_a_batch_of_pairs_of_two_sizes_is_its_own_unbatched_solve(cell):
    _check_batch(cell)


#: The cells at two rates whose one-rate pair is a float32 cell of every push.
_MIXED = [c for c in PUSH if c.label == "mixed_dtype" and c.small == "b"
          and c.acceleration == "none"]
assert len(_MIXED) == 1


@pytest.mark.parametrize("cell", _MIXED, ids=_ids(_MIXED))
def test_each_member_of_a_batch_of_mixed_dtype_pairs_records_the_floor_of_its_compact_readings(
        cell):
    _check_batched_floor(cell)


_RATES = [c for c in PUSH if (c.domain.multirate or c.domain.subcycled)
          and dataclasses.replace(c, label="f32") in PUSH]
assert {c.label for c in _RATES} == {"multi_rate", "sub_cycled"}


@pytest.mark.parametrize("cell", _RATES, ids=_ids(_RATES))
def test_a_pair_of_two_sizes_at_two_rates_is_the_pair_at_one(cell):
    _check_one_rate(cell)


_CLOCKED = [c for c in PUSH if c.domain.multirate and c.acceleration == "none"]


@pytest.mark.parametrize("cell", _CLOCKED, ids=_ids(_CLOCKED))
def test_a_multi_rate_graph_reads_a_pair_of_two_sizes_on_the_steps_its_group_fires(cell):
    _check_firing(cell)


_RESTARTED = [c for c in PUSH if c.domain.restart]


@pytest.mark.parametrize("cell", _RESTARTED, ids=_ids(_RESTARTED))
def test_a_restart_brings_back_the_report_the_floor_slot_and_the_next_step(cell):
    _check_restart(cell)


@pytest.mark.parametrize("cell", DIAGNOSED, ids=_ids(DIAGNOSED))
def test_a_diagnosed_pair_of_two_sizes_reports_the_spectrum_of_its_twin(cell):
    _check_diagnosed(cell)


# ---------------------------------------------------------------------------
# The slow lane
# ---------------------------------------------------------------------------
_UNTWINNED = [c for c in _two_way(PUSH) if c not in TWINNED]


# Slow: the twins of the other two-way cells, a graph each.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_of_two_sizes_reports_what_its_marker_side_twin_reports
@pytest.mark.slow
@pytest.mark.parametrize("cell", _UNTWINNED, ids=_ids(_UNTWINNED))
def test_the_other_pairs_of_every_push_report_what_their_twins_report(cell):
    _check_twin(cell)


# Slow: two graphs compiled with the report's analysis per cell.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_diagnosed_pair_of_two_sizes_reports_the_spectrum_of_its_twin
@pytest.mark.slow
@pytest.mark.parametrize("cell", DIAGNOSED_SLOW, ids=_ids(DIAGNOSED_SLOW))
def test_a_diagnosed_pair_of_two_sizes_reports_the_spectrum_of_its_twin_in_every_domain(cell):
    _check_diagnosed(cell)
    _forget(cell)


# Slow: four hundred cells, one or two graphs compiled for each.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_of_two_sizes_stops_where_the_compact_reference_does
# tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_converged_accelerated_pair_of_two_sizes_is_within_K_tolerances
# tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_only_a_pair_with_an_edge_read_through_its_mapping_owns_the_floor_slot
@pytest.mark.slow
@pytest.mark.parametrize("cell", PRODUCT, ids=_ids(PRODUCT))
def test_every_configuration_of_a_pair_of_two_sizes_in_every_domain(cell):
    """Domain x acceleration x schedule x pair: every check the cell admits."""
    check_everything(cell)
    _forget(cell)


# Slow: run_adaptive compiles its step on every call.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_of_two_sizes_stops_where_the_compact_reference_does
# (every domain but run_adaptive, the same check)
@pytest.mark.slow
@pytest.mark.parametrize("cell", ADAPTIVE, ids=_ids(ADAPTIVE))
def test_run_adaptive_steps_a_pair_of_two_sizes_as_a_plain_step_does(cell):
    """Under ``run_adaptive`` the reported solve is its last half step's: the
    residual of the state it returns in the compact readings, the claim,
    the floor's slot and the twin."""
    check_everything(cell)
    _forget(cell)


# Slow: fields of 600 and 2000 entries.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_of_two_sizes_stops_where_the_compact_reference_does
@pytest.mark.slow
@pytest.mark.parametrize("cell", LARGER, ids=_ids(LARGER))
def test_a_larger_pair_of_two_sizes_in_every_domain(cell):
    check_everything(cell)
    _forget(cell)


_BATCHES = [c for c in PUSH if c.domain.vmap and c.acceleration == "none"]


# Slow: forty scenarios, each also solved alone.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_each_member_of_a_batch_of_pairs_of_two_sizes_is_its_own_unbatched_solve
@pytest.mark.slow
@pytest.mark.parametrize("cell", _BATCHES, ids=_ids(_BATCHES))
def test_a_batch_with_more_members_than_the_large_field_has_entries(cell):
    """The side of an edge is its mapping's, whatever the batch: with forty
    members the batch axis is longer than either field."""
    assert BATCH > cell.n_large
    _check_reference(cell, count=BATCH)
    _check_batch(cell, count=BATCH)
    if cell.kind == "two-way":
        _check_twin(cell, count=BATCH)
    _forget(cell)


# ---------------------------------------------------------------------------
# A pair coupled through a multilinear_grid scatter and gather (experimental)
# ---------------------------------------------------------------------------
# The pairs above are coupled through static mappings.  Here the markers
# and the grid of ``tests/property/geometry_interface_graphs.py``: the
# mapping reads positions that move with the iterate, and under the
# interface norm a scatter is read at its inputs -- the marker values, and
# for a source anchor the positions in grid spacings -- and a gather as
# delivered (MAP-050).  That module runs float32 and float64 pairs at one
# rate, unbatched; this section runs the pair in every domain the norm
# accepts it in (``coupling_domain_sizes.GEO_ACCEPTED``) and asserts the
# refusal in the one it does not (a sub-cycled group).
#
# The oracles are that module's, with every number as its member stores it
# (``coupling_domain_sizes.GeoStored``): the exact float64 statement of the
# reading -- the pass a plain iteration stops on, its residual, the fields
# a solve returns as its accepted iterate holds them and the others one
# plain pass on, the distance to the fixed point against ``K`` -- and the
# marker-side twin, whose plain edges carry the marker values and the
# positions in spacings.  What a domain adds is asserted as for the static
# pairs: a member of a batch against its own unbatched solve, the base
# steps a multi-rate graph's group fires on and the same pair at one rate,
# a predictor's guess as the loop's first iterate, the graph a checkpoint
# brings back.
#
# Unlike the static pairs' members these have a memory: the positions
# integrate from the pre-step state.  So each solve's reference is handed
# the state its step started from (``GeoStored.from_state``), and the
# markers are written after ``compile()`` (as every step's state is; the
# advisory about positions a dtype cannot resolve is asked of the state
# ``compile()`` sees, and has a test of its own below).


def _g(label, kind="two-way", anchors=("source", "target"), schedule="gauss-seidel",
       acceleration="none", small="a", **kw) -> cs.GeoCell:
    return cs.GeoCell(label, kind, anchors, schedule, acceleration, small, **kw)


#: The geometry cells of every push: the two-way pair (a scatter read at
#: its source with its positions, a gather anchored at its target) once in
#: each accepted domain, the member that holds the markers' values and the
#: schedule rotated.  Every other kind, anchor and acceleration in every
#: domain is the slow lane's product.
GEO_PUSH = (
    _g("f32"),
    _g("mixed_dtype", schedule="jacobi"),
    _g("vmap"),
    _g("multi_rate"),
    _g("predictors_warm_starts", schedule="jacobi", small="b"),
    _g("checkpoint_restart", anchors=("source", "source"), small="b"),
)
#: Whose marker-side twin is compiled on every push.
GEO_TWINNED = (_g("f32"),)
assert set(GEO_TWINNED) <= set(GEO_PUSH)
# The batch's and the multi-rate graph's cells are the float32 cell in
# another domain: each is compared with that cell's own solves.
assert {dataclasses.replace(c, label="f32") for c in GEO_PUSH
        if c.domain.vmap or c.domain.multirate} == {GEO_PUSH[0]}
GEO_ANCHORS = (("source", "target"), ("target", "source"), ("source", "source"),
               ("target", "target"))
GEO_ACCELERATIONS = ("aitken", "iqn-ils")


def _geo_product() -> list:
    """The slow lane's cells: every accepted domain x kind x schedule, the
    anchors and the member that holds ``p`` rotated; this file's own
    tuples, and no other table's length."""
    cells = []
    for i, label in enumerate(cs.GEO_ACCEPTED):
        for j, kind in enumerate(gi.KINDS):
            for k, schedule in enumerate(SCHEDULES):
                cells.append(_g(label, kind, GEO_ANCHORS[(i + j + k) % len(GEO_ANCHORS)], schedule,
                                small="ab"[(i + j) % 2]))
    return [c for c in cells if c not in GEO_PUSH]


GEO_PRODUCT = _geo_product()
#: The claim under an acceleration, slow: in each accepted domain, one of
#: the two (a quasi-Newton history carried through a predictor's sequence
#: and a restart among them).
GEO_ACCELERATED = tuple(
    _g(label, "two-way", GEO_ANCHORS[i % 2], SCHEDULES[i % 2],
       GEO_ACCELERATIONS[i % len(GEO_ACCELERATIONS)], "ab"[i % 2])
    for i, label in enumerate(cs.GEO_ACCEPTED))
#: The twins of the slow lane: a domain each (the sequences of a predictor
#: and of a restart are not twinned: :func:`_check_geo_twin`).
GEO_TWINNED_SLOW = (
    _g("f64", anchors=("source", "source"), schedule="jacobi", small="b"),
    _g("f64", anchors=("target", "target")),
    _g("vmap", small="b"),
    _g("multi_rate"),
    _g("mixed_dtype", schedule="jacobi"),
)
#: What the reference's residual and a returned field are held to: tight
#: where every member holds float64; elsewhere the float32 allowances of
#: ``test_coupling_geometry_interface_norm.py``.
GEO_TIGHT = 1e-8
GEO_ALLOWED = {True: 1.02, False: 1.10}


@dataclasses.dataclass
class GeoRun:
    built: cs.GeoBuilt
    #: One reference per solve, each started where its solve started.
    refs: list
    solves: list
    straight: list
    saved: object = None
    loaded: object = None


_GEO_RUNS: dict = {}


#: A pass count is asserted where every estimate the reference's loop
#: compared was at least this factor from the threshold: beside a float32
#: member the residual carries the positions' rounding (half the float
#: floor is what it is held to), which the margin must clear.  The same
#: number for a float64 pair, so that every domain runs the same scenarios.
GEO_MARGIN = 1.3
#: ``(loop gain of the values with the positions held, sign of the round
#: trip)`` of a geometry cell's scenarios.  Lower than :data:`GAINS`: a
#: plain loop's estimates fall by its contraction from pass to pass, so
#: the margin of its exit is at most the root of one over it, and under
#: Jacobi a loop gain of 0.6 (0.77 a pass) cannot reach :data:`GEO_MARGIN`.
GEO_GAINS = ((0.25, 1.0), (0.15, -1.0))


def _geo_chain_margins(cell, draw) -> list:
    """The margins of a sequence's steps on the reference's own chain: each
    step started from the state the reference returns for the last
    (through the predictor's guess where the domain has one); those of the
    steps the domain checks (a restart's: the ones after the checkpoint)."""
    whole = cs.geo_measured_whole(cell)
    returned, margins = [], []
    for k, move in enumerate(MOVES):
        ref = cs.GeoStored(cell, draw, scale=move)
        if returned:
            ref.from_state(returned[-1])
        start = _geo_guess(k, returned, ref.pre) if cell.domain.predictor else None
        exit_ = ref.plain_exit(start=start)
        if not exit_["converged"]:
            return []
        returned.append(ref.returned(exit_["state"], whole))
        margins.append(exit_["margin"])
    return margins[2:] if cell.domain.restart else margins


def _geo_decided(cell, draw) -> bool:
    """Is *draw*'s exit decided with margin, in the reference's own float64
    arithmetic (the same answer on every platform)?  From the pre-step
    state; and, for a sequence, on :data:`PREDICTED_STEPS` of the steps
    the domain checks, with a little to spare for the graph's own rounding
    of the states they start from."""
    if cs.GeoStored(cell, draw).plain_exit()["margin"] < GEO_MARGIN:
        return False
    if not _sequenced(cell):
        return True
    return sum(m >= 1.01 * GEO_MARGIN for m in _geo_chain_margins(cell, draw)) >= PREDICTED_STEPS


def _geo_draws(cell) -> list:
    """The cell's scenarios: for each loop gain of :data:`GEO_GAINS` (the
    first, for a sequence) the first candidate whose exit is decided with
    margin (:func:`_geo_decided`).  The margin of a plain loop is set by
    its contraction more than by its seed, so the candidates step the gain
    down from the nominal one by 3% at a time, a seed each."""
    out = []
    for gain, sign in (GEO_GAINS[:1] if _sequenced(cell) else GEO_GAINS):
        for seed in range(36):
            draw = gi.Draw(seed, gain * (1.0 - 0.03 * (seed % 9)), cs.GEO_PULL, sign)
            if _geo_decided(cell, draw):
                break
        else:
            raise AssertionError(f"{cell.id}: no candidate decides its exit with margin")
        out.append(draw)
    return out


def _geo_batch(gm, refs, params) -> list:
    """Every scenario as one member of a ``jax.vmap`` of the step, each
    started from its own reference's pre-step state."""
    import jax  # noqa: PLC0415

    if id(gm) not in cd._VMAPPED:  # noqa: SLF001
        cd._VMAPPED[id(gm)] = (  # noqa: SLF001
            gm, jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, 0))))  # noqa: SLF001
    step = cd._VMAPPED[id(gm)][1]  # noqa: SLF001
    starts = []
    for ref in refs:
        ref.start(gm)
        # A tree of its own: the graph writes the next scenario's start
        # into the containers it holds.
        starts.append(jax.tree.map(lambda v: v, gm._state))  # noqa: SLF001
    for name in ("a", "b"):
        assert not cd.bitwise(starts[0][name]["x"], starts[-1][name]["x"]), (
            "fixture premise: the scenarios of a batch start from their own states")
    new = step(cd._stack(starts), gm._default_external_inputs(), cd._stack(params))  # noqa: SLF001
    return [cd._solve(gm, p, jax.tree.map(lambda v, i=i: v[i], new),  # noqa: SLF001
                      pre=cd._members(gm, starts[i]))  # noqa: SLF001
            for i, p in enumerate(params)]


def _geo_run(cell, twin: bool = False) -> GeoRun:
    """The cell's graph (or its twin's), built and stepped once per session."""
    key = (cell, twin)
    if key in _GEO_RUNS:
        return _GEO_RUNS[key]
    d = cell.domain
    with cd.entered(d):
        draws = _geo_draws(cell)
        if _sequenced(cell):
            refs = [cs.GeoStored(cell, draws[0], scale=move) for move in MOVES]
        else:
            refs = [cs.GeoStored(cell, draw) for draw in draws]
        first = refs[0]
        built = (cs.build_geometry_twin(cell, first.pre["p"]["pos"]) if twin
                 else cs.build_geometry(cell))
        # The markers are not in the state ``compile()`` saw: nothing to say.
        assert not built.advisories, built.advisories
        gm = built.gm
        params = [ref.params(gm) for ref in refs]
        straight, saved, loaded = [], None, None
        if d.restart:
            straight, solves, saved, loaded = cs.restart_pairs(built, params, start=first.write)
            refs = refs[len(refs) - len(solves):]
        elif d.predictor:
            first.start(gm)
            solves = [cd._one(d, gm, p) for p in params]  # noqa: SLF001
        elif d.vmap:
            solves = _geo_batch(gm, refs, params)
        else:
            solves = []
            for ref, p in zip(refs, params):
                ref.start(gm)
                solves.append(cd._one(d, gm, p))  # noqa: SLF001
        cd.assert_in_domain(d, gm, solves)
        for ref, s in zip(refs, solves):
            ref.from_state(cs.geo_state_of(cell, s, pre=True))
    # The premise: the pair is the cell's, its two edges of the kind.
    mapped = [e for e in gm._edges if e.mapping is not None]  # noqa: SLF001
    assert [e.mapping.kind for e in mapped] == ["multilinear_grid"] * (1 if twin else 2), cell.id
    order = [n for n in gm.schedule if n in ("a", "b")]
    assert order == [cell.names["p"], cell.names["q"]], (cell.id, order)
    _GEO_RUNS[key] = GeoRun(built, refs, solves, straight, saved, loaded)
    return _GEO_RUNS[key]


def _geo_forget(cell) -> None:
    for key in [k for k in _GEO_RUNS if k[0] == cell]:
        del _GEO_RUNS[key]


def _geo_guess(k: int, returned: list, before: dict) -> dict:
    """:func:`_guess` for members of two fields: the predictor
    extrapolates every field the earlier steps returned."""
    if k < 2:
        return before
    if k == 2:
        weights = ((2.0, 1), (-1.0, 0))
    else:
        weights = ((3.0, k - 1), (-3.0, k - 2), (1.0, k - 3))
    return {n: {f: sum(w * returned[i][n][f] for w, i in weights) for f in ("x", "pos")}
            for n in ("p", "q")}


def _geo_starts(cell, run: GeoRun) -> list:
    """The iterate each solve's loop started from: the state before the
    step (``None``: the reference's own pre-step state), or a predictor's
    guess from the states the earlier steps returned."""
    if not cell.domain.predictor:
        return [None] * len(run.solves)
    returned = [cs.geo_state_of(cell, s) for s in run.solves]
    before = [cs.geo_state_of(cell, s, pre=True) for s in run.solves]
    return [_geo_guess(k, returned, before[k]) for k in range(len(run.solves))]


def _geo_scale(cell, field: str, value) -> float:
    """What a field is compared against: a value its own size, a position one spacing."""
    return min(cell.shape.spacing) if field == "pos" else float(np.max(np.abs(value)))


def _check_geo_reference(cell) -> None:
    """The plain iteration of a geometry *cell* against the exact reference
    of the rule: the pass it stops on, its residual, the state it returns,
    and the claim."""
    assert cell.acceleration == "none", cell
    run = _geo_run(cell)
    exact, whole = cell.exact, cs.geo_measured_whole(cell)
    predicted, told_apart, kept_apart = 0, set(), False
    for k, (s, ref, start) in enumerate(zip(run.solves, run.refs, _geo_starts(cell, run))):
        r, x = s.report, cs.geo_state_of(cell, s)
        where = f"{cell.id}, solve {k}"
        plain = ref.plain_exit(start=start)
        assert plain["converged"], f"{where}: fixture premise: the reference converges"
        assert r["converged"] is True, (where, r)
        if plain["margin"] >= GEO_MARGIN:
            predicted += 1
            assert r["iterations"] == plain["iterations"], (
                f"{where}: the step took {r['iterations']} passes; the plain loop on the "
                f"rule's readings stops after {plain['iterations']} (margin "
                f"{plain['margin']:.3g})")
        else:
            assert abs(r["iterations"] - plain["iterations"]) <= 1, (where, r, plain)
        accepted = plain["state"]
        after = ref.one_pass(accepted)
        if r["iterations"] == plain["iterations"]:
            allowed = (GEO_TIGHT * plain["residual"] if exact
                       else 0.05 * plain["residual"] + 0.5 * ref.floor(accepted))
            assert abs(r["residual"] - plain["residual"]) <= allowed, (
                f"{where}: reported residual {r['residual']!r}; the rule's readings give "
                f"{plain['residual']!r} (allowed {allowed:.2e})")
            for rule in ("delivered", "no-positions", "own-magnitude"):
                other = ref.residual(after, accepted, rule)
                if abs(other - plain["residual"]) > 1.5 * allowed:
                    told_apart.add(rule)
            want = ref.returned(accepted, whole)
            for name in ("p", "q"):
                for field in ("x", "pos"):
                    bound = (GEO_TIGHT if exact else 2e-5) * _geo_scale(
                        cell, field, want[name][field])
                    worst = float(np.max(np.abs(x[name][field] - want[name][field])))
                    assert worst <= bound, (
                        f"{where}: {cell.names[name]}.{field} is {worst:.3e} from what the "
                        f"rule returns (bound {bound:.3e}; kept whole: {whole})")
                    gap = float(np.max(np.abs(after[name][field] - accepted[name][field])))
                    kept_apart |= (name, field) in whole and gap > 100.0 * bound
        distance, K = ref.distance(x), ref.K(whole)
        assert distance <= GEO_ALLOWED[exact] * K, (
            f"{where}: converged=True at {distance:.3f} tolerances from the fixed point in "
            f"the rule's readings; K = {K:.3f}")
    # Premise: the residual of the readings the rule is not differs by more
    # than the comparison allows.  Every edge read as delivered, wherever
    # there is a scatter; and, where every member holds float64 and a
    # scatter is anchored at its source, the scatter without its positions
    # and the positions over their own magnitude.  (Beside a float32
    # member the allowance is half the float floor, and the positions'
    # share of this family's residual at exit is under it: those two are
    # told apart by ``test_coupling_geometry_interface_norm.py``, edge by
    # edge.)
    ways = list(zip(gi.WAYS[cell.kind], cell.anchors))
    needed = {"delivered"} if any(way == "scatter" for way, _anchor in ways) else set()
    if exact and ("scatter", "source") in ways:
        needed |= {"no-positions", "own-magnitude"}
    assert needed <= told_apart, (
        f"{cell.id}: fixture premise: the readings {sorted(needed - told_apart)} give the "
        f"residual of the rule, to what the comparison allows")
    if whole and exact:
        assert kept_apart, (
            f"{cell.id}: fixture premise: a kept field's accepted iterate and one pass on "
            f"agree to what the comparison allows")
    # Every scenario started from the pre-step state was chosen with margin;
    # so were some of a sequence's steps (``_geo_decided``).
    need = PREDICTED_STEPS if _sequenced(cell) else len(run.solves)
    assert predicted >= need, f"{cell.id}: {predicted} pass counts predicted, of {need}"


def _check_geo_claim(cell) -> None:
    """MAP-050's claim on *cell*: every solve converges, within ``K`` tolerances."""
    run = _geo_run(cell)
    whole = cs.geo_measured_whole(cell)
    for k, (s, ref) in enumerate(zip(run.solves, run.refs)):
        assert s.report["converged"] is True, (cell.id, k, s.report)
        distance, K = ref.distance(cs.geo_state_of(cell, s)), ref.K(whole)
        assert distance <= GEO_ALLOWED[cell.exact] * K, (
            f"{cell.id}, solve {k}: converged=True at {distance:.3f} tolerances from the "
            f"fixed point in the rule's readings; K = {K:.3f}")


def _check_geo_slot(cell) -> None:
    """The step records the floor of a gather anchored at its target (its
    positions are the pre-step state, which the returned state does not
    hold), with the dtypes of the members in it; a pair with no such edge
    owns no slot."""
    run = _geo_run(cell)
    key = f"coupling_{cd.KEY}_reading_floor"
    owned = any(way == "gather" and anchor == "target"
                for way, anchor in zip(gi.WAYS[cell.kind], cell.anchors))
    for ref, s in zip(run.refs, run.solves):
        if not owned:
            assert key not in s.meta, (cell.id, key, s.meta[key])
            continue
        want = ref.floor(cs.geo_state_of(cell, s))
        assert key in s.meta and float(s.meta[key]) == pytest.approx(want, rel=1e-4), (
            f"{cell.id}: the slot holds {s.meta.get(key)!r}; the rule's floor is {want!r}")
    for s in list(run.straight) + [s for s in (run.saved, run.loaded) if s]:
        assert (key in s.meta) == owned, (cell.id, key)


def _check_geo_twin(cell) -> None:
    """The edge-mapped pair reports what its marker-side twin reports:
    the verdict and the pass count, the residual, and the state -- but the
    positions the scatter reads, which the edge-mapped graph's norm
    measures whole and keeps while the twin carries them through a
    transform and returns them one pass on: within the last pass's step.

    Not asked of a sequence (a predictor's, a restart's): after their
    first step the two graphs start from positions a tolerance apart,
    and from then on they are two problems.  Each step of a sequence is
    held to the reference, started where the graph's own last step
    stopped.
    """
    assert cell.kind == "two-way" and not _sequenced(cell), cell
    mapped, twin = _geo_run(cell), _geo_run(cell, twin=True)
    exact = cell.exact
    assert len(mapped.solves) == len(twin.solves) > 0
    for k, (s, t, ref) in enumerate(zip(mapped.solves, twin.solves, mapped.refs)):
        where = f"{cell.id}, solve {k}"
        ra, rb = s.report, t.report
        assert ra["converged"] is True and rb["converged"] is True, (where, ra, rb)
        assert ra["iterations"] == rb["iterations"] > 2, (where, ra, rb)
        xa, xb = cs.geo_state_of(cell, s), cs.geo_state_of(cell, t)
        allowed = (1e-9 * ra["residual"] if exact
                   else 0.05 * ra["residual"] + 0.5 * ref.floor(xa))
        assert abs(ra["residual"] - rb["residual"]) <= allowed, (where, ra, rb)
        for name in ("p", "q"):
            for field in ("x", "pos"):
                scale = _geo_scale(cell, field, xb[name][field])
                worst = float(np.max(np.abs(xa[name][field] - xb[name][field])))
                if (name, field) == ("p", "pos") and cell.anchors[0] == "source":
                    assert worst <= 10.0 * gi.RTOL * scale, (where, worst)
                    continue
                assert worst <= (1e-9 if exact else 1e-3) * scale, (
                    f"{where}: {cell.names[name]}.{field} is {worst:.3e} from the twin's")
        # The pinned marker: the unit entry of the twin's position edge.
        assert np.all(xa["p"]["pos"][0] == ref.pre["p"]["pos"][0]), where


def _check_geo_batch(cell) -> None:
    """Each member of the ``vmap`` domain's batch is its own unbatched
    solve: the same scenario of the same pair, stepped alone (the float32
    domain's cell, a graph of its own built the same way)."""
    run, plain = _geo_run(cell), _geo_run(dataclasses.replace(cell, label="f32"))
    assert [r.draw for r in run.refs] == [r.draw for r in plain.refs], cell.id
    passes = set()
    for k, (s, alone) in enumerate(zip(run.solves, plain.solves)):
        _same_member(f"{cell.id}: member {k} of the batch", s, alone, ulps=4.0)
        passes.add(s.report["iterations"])
    if cell.acceleration == "none":
        assert len(passes) > 1, (
            f"{cell.id}: fixture premise: the members of the batch stop on different passes")


def _check_geo_one_rate(cell) -> None:
    """A multi-rate graph's pair is the same pair at one rate, to the bit."""
    run, plain = _geo_run(cell), _geo_run(dataclasses.replace(cell, label="f32"))
    for k, (s, t) in enumerate(zip(run.solves, plain.solves)):
        assert not _group_differs(s, t), (
            f"{cell.id}: solve {k} differs from the one-rate pair's: {_group_differs(s, t)}")


def _check_geo_firing(cell) -> None:
    """:func:`_check_firing` for a pair whose markers were written after
    ``compile()``: on the base step between two solves the members -- the
    positions among them -- the report and every slot are the last solve's."""
    run = _geo_run(cell)
    gm, p = run.built.gm, run.solves[0].params
    with cd.entered(cell.domain):
        run.refs[0].start(gm)
        shots = []
        for _ in range(6):
            gm.step(params=p)
            shots.append(cd._solve(gm, p))  # noqa: SLF001
    fired = [bool(_group_differs(a, b)) for a, b in zip(shots, shots[1:])]
    assert fired == [False, True, False, True, False], (cell.id, fired)
    assert not _same_graph(shots[1], run.solves[0]), (
        f"{cell.id}: the pair of base steps the harness reads holds another solve: "
        f"{_same_graph(shots[1], run.solves[0])}")
    # The positions moved with each solve (the members have a memory here).
    moved = [name for name in ("a", "b")
             if not cd.bitwise(shots[1].state[name]["pos"], shots[3].state[name]["pos"])]
    assert moved, f"{cell.id}: fixture premise: a second solve moves the markers"


def _check_geo_restart(cell) -> None:
    """A checkpoint brings back the positions with the rest: the report,
    the floor's slot (or its absence) and the next step."""
    run = _geo_run(cell)
    assert run.saved is not None and run.straight, cell.id
    assert not _same_graph(run.saved, run.loaded), (
        f"{cell.id}: the loaded graph differs from the checkpointed one: "
        f"{_same_graph(run.saved, run.loaded)}")
    for k, (a, b) in enumerate(zip(run.straight, run.solves)):
        assert not _same_graph(a, b), (
            f"{cell.id}: step {k} after the restart differs from the uninterrupted run's: "
            f"{_same_graph(a, b)}")
    # Premise: the checkpoint was written away from where the run started.
    started = run.refs[0].layout["q"]["index"] * np.asarray(cell.shape.spacing)
    held = np.asarray(run.saved.state[cell.names["q"]]["pos"], np.float64)
    assert np.max(np.abs(held - started)) > 1e-3 * min(cell.shape.spacing), cell.id


def check_geometry(cell) -> None:
    """Every check a geometry cell admits."""
    d = cell.domain
    if cell.acceleration == "none":
        _check_geo_reference(cell)
    else:
        _check_geo_claim(cell)
    _check_geo_slot(cell)
    if d.restart:
        _check_geo_restart(cell)
    if d.vmap:
        _check_geo_batch(cell)
    if d.multirate and cell.acceleration == "none":
        _check_geo_firing(cell)


@pytest.mark.parametrize("cell", GEO_PUSH, ids=_ids(GEO_PUSH))
def test_a_pair_with_geometry_edges_stops_where_the_rules_reference_does(cell):
    """MAP-050 in every domain that accepts the pair: the pass count, the
    residual and the state a solve returns are the exact reference's under
    the rule (a scatter at its source value and its source's positions in
    grid spacings, a gather as delivered), and a converged pair is within
    ``K`` tolerances.  With what the domain adds: the floor's slot, a
    member of a batch against its unbatched solve, the base steps a group
    fires on, a predictor's guess, a restart."""
    check_geometry(cell)


@pytest.mark.parametrize("cell", GEO_TWINNED, ids=_ids(GEO_TWINNED))
def test_a_pair_with_geometry_edges_reports_what_its_marker_side_twin_reports(cell):
    _check_geo_twin(cell)


_GEO_RATES = [c for c in GEO_PUSH if c.domain.multirate
              and dataclasses.replace(c, label="f32") in GEO_PUSH]
assert len(_GEO_RATES) == 1


@pytest.mark.parametrize("cell", _GEO_RATES, ids=_ids(_GEO_RATES))
def test_a_multi_rate_graphs_pair_with_geometry_edges_is_the_pair_at_one_rate(cell):
    _check_geo_one_rate(cell)


def _refused_graph(cell, **group_kw):
    return cs.geometry_group(cell, cs.geometry_graph(cell), **group_kw)


@pytest.mark.parametrize("kind", gi.KINDS)
def test_a_sub_cycled_pair_with_geometry_edges_is_refused_under_the_interface_norm(kind):
    """The one domain of the battery the norm does not read such a pair
    in: ``compile()`` refuses, naming both edges and the reason.  The same
    graph under ``"mixed"`` compiles, and so does the pair at one rate
    under the interface norm (every other cell of this section)."""
    (label,) = cs.GEO_REFUSED
    cell = _g(label, kind, GEO_ANCHORS[gi.KINDS.index(kind)])
    with cd.entered(cell.domain):
        refused = _refused_graph(cell)
        gg.assert_interface_norm_refused(refused.compile, list(cell.keys), "sub-cycled")
        other = _refused_graph(cell, convergence_norm="mixed")
        other.compile()
        assert other._committed_coupling_groups[cd.KEY].subcycling  # noqa: SLF001


def _geo_owed(cell, pre: dict) -> dict:
    """``{library edge key: member that stores the positions}``: the
    advisories ``compile()`` owes *cell* with its markers at *pre*.

    The rule restated (``geometry_interface_graphs.positions_behind`` and
    ``positions_floor``): an edge whose reading rests on stored positions
    -- a scatter anchored at its source, a gather at either anchor --
    where four roundings per evaluation of the farthest coordinate, in the
    dtype of the member that stores it, are the tolerance or more."""
    shape = cell.shape
    E = gi.EVALUATIONS[shape.schedule]
    owed = {}
    for i, (src, dst) in enumerate(gi.EDGES):
        behind = gi.positions_behind(shape).get(f"{src}.x->{dst}.u")
        if behind is None:
            continue
        holder = behind[0]
        dtype = str(jnp.dtype(cell.dtypes[holder]))
        if E * gi.positions_floor(pre[holder]["pos"], shape.spacing, dtype) >= 1.0:
            owed[cell.keys[i]] = cell.names[holder]
    return owed


#: The pairs the advisory is asked of in each domain: the two-way pair with
#: the markers' values on either member; two gathers each anchored at its
#: source (so each member's positions are behind one edge); and a scatter
#: anchored at its target (no reading of positions: never warned of)
#: beside a gather anchored at its source, both on the positions of ``a``.
_GEO_ADVISED = (("two-way", ("source", "target"), "a"), ("two-way", ("source", "target"), "b"),
                ("gather-only", ("source", "source"), "a"),
                ("two-way", ("target", "source"), "b"))


@pytest.mark.parametrize("label", cs.GEO_ACCEPTED)
def test_compile_warns_of_far_markers_by_the_dtype_of_the_member_that_stores_them(label):
    """In every domain: with the markers in the state ``compile()`` sees,
    three thousand spacings from zero, it warns once for each edge whose
    reading rests on positions a float32 member stores -- a part of a
    scatter's reading, or what a gather's value is delivered at -- naming
    the member, and says nothing of a float64 member's (the mixed-dtype
    domain has one of each), of a scatter anchored at its target, nor of
    any pair near zero.  The graph is compiled either way."""
    seen = set()
    for kind, anchors, small in _GEO_ADVISED:
        for origin in (cs.GEO_ORIGIN, 3000.0):
            cell = _g(label, kind, anchors, small=small, origin=origin)
            with cd.entered(cell.domain):
                ref = cs.GeoStored(cell, gi.Draw(0, GEO_GAINS[0][0], cs.GEO_PULL))
                built = cs.build_geometry(cell, placed={n: ref.pre[n]["pos"] for n in "pq"})
            owed = _geo_owed(cell, ref.pre)
            said = {}
            for text in built.advisories:
                (key,) = [k for k in cell.keys if repr(k) in text]
                assert key not in said, built.advisories
                said[key] = text
            assert sorted(said) == sorted(owed), (cell.id, sorted(said), owed)
            for key, member in owed.items():
                assert f"float32 positions {member}.pos" in said[key], (cell.id, said[key])
            assert built.gm._compiled_step is not None, cell.id  # noqa: SLF001
            if origin == cs.GEO_ORIGIN:
                assert not owed, (cell.id, owed)
            seen.add(len(owed))
    # Premise: pairs with both edges warned of and with one (the scatter
    # anchored at its target is silent; so is a float64 member's edge in
    # the mixed-dtype domain); none where every member holds float64.
    d = cd.DOMAINS[label]
    every_float64 = all(jnp.dtype(t) == jnp.dtype(jnp.float64) for t in d.dtypes)
    assert seen == ({0} if every_float64 else {0, 1, 2}), (label, seen)


# Slow: a graph compiled per cell (two for a batch), a hundred and twenty cells.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_with_geometry_edges_stops_where_the_rules_reference_does
@pytest.mark.slow
@pytest.mark.parametrize("cell", GEO_PRODUCT, ids=_ids(GEO_PRODUCT))
def test_every_kind_anchor_and_schedule_of_a_pair_with_geometry_edges_in_every_domain(cell):
    """Domain x kind x schedule, the anchors rotated: every check the cell admits."""
    check_geometry(cell)
    _geo_forget(cell)


# Slow: a graph compiled per cell.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_with_geometry_edges_stops_where_the_rules_reference_does
@pytest.mark.slow
@pytest.mark.parametrize("cell", GEO_ACCELERATED, ids=_ids(GEO_ACCELERATED))
def test_a_converged_accelerated_pair_with_geometry_edges_is_within_K_tolerances(cell):
    check_geometry(cell)
    _geo_forget(cell)


# Slow: two graphs compiled per cell.
# Per push: tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py::test_a_pair_with_geometry_edges_reports_what_its_marker_side_twin_reports
@pytest.mark.slow
@pytest.mark.parametrize("cell", GEO_TWINNED_SLOW, ids=_ids(GEO_TWINNED_SLOW))
def test_a_pair_with_geometry_edges_reports_what_its_twin_reports_in_the_other_domains(cell):
    _check_geo_twin(cell)
    _geo_forget(cell)
