"""The interface norm on a mapped edge that expands, and one that reduces, in every numeric domain.

Under ``convergence_norm="interface"`` an internal edge whose static mapping
delivers **more** entries than its source field holds is read at its
source value -- the compact side -- and every other edge as delivered
(CPL-188); a group that reports ``converged=True`` is then within ``K``
tolerances of its fixed point in those readings, ``K`` from the loop's own
operator and blind to the size of the large field (CPL-191).  Those rows'
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
  and every ``_meta`` slot, to the bit (in the mixed-dtype domain to
  float32 rounding, for the reason :func:`_same_solve` gives);
* the float floor from the dtypes and sizes of the two readings.

**What each domain adds** is asserted where it is: a member of a batch
against its own unbatched solve; the base step of a multi-rate graph on
which the group does not fire; a sub-cycled group against the same group
at one rate (the reading is taken once per pass, of the members' fields as
the pass leaves them, the fast member's after its sub-steps); the pass
count of a predictor's guess; the graph a checkpoint brings back -- the
report, the floor's slot or its absence, and the next step.

A state is compared with the reference only on the fields the norm reads
at their source (the accepted iterate, whichever state a step returns for
the other fields), and with the twin everywhere (the twin's step has the
same rule).
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import PRECISION_FLOOR_ULPS
from tests.core import coupling_domain_sizes as cs
from tests.core import coupling_domains as cd
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
#: or holds in float64 beside a float32 one), a pair whose edges both
#: expand or both reduce, and an accelerated pair; the mapping forms, the
#: schedules and the accelerations rotated over the domains.  The 16-bit
#: domains stop at a cap of two passes, as the tie's cells do there, where
#: an acceleration has nothing to show.
PUSH = (
    _c("f32", "two-way", "matrix", "gauss-seidel"),
    _c("f32", "two-way", "sparse", "jacobi", small="b"),
    _c("f32", "scatter-only", "sparse-transposed", "gauss-seidel"),
    _c("f32", "two-way", "sparse-transposed", "jacobi", "aitken"),
    _c("f64", "two-way", "sparse", "gauss-seidel"),
    _c("f64", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("f64", "gather-only", "matrix", "jacobi"),
    _c("f64", "two-way", "matrix", "gauss-seidel", "iqn-ils", small="b"),
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
    _c("vmap", "two-way", "sparse-transposed", "gauss-seidel", "iqn-imvj"),
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
    _c("checkpoint_restart", "two-way", "matrix", "gauss-seidel", "fixed"),
)
#: The two-way cells whose marker-side twin is compiled on every push: one
#: in each domain, the small field on ``a`` and on ``b`` in turn, and two
#: accelerated (IQN's secant history, and its Jacobian carried from step
#: to step under a predictor).  The twins of the others run in the slow
#: lane (:func:`test_the_other_pairs_of_every_push_report_what_their_twins_report`).
TWINNED = (
    _c("f32", "two-way", "matrix", "gauss-seidel"),
    _c("f64", "two-way", "sparse-transposed", "jacobi", small="b"),
    _c("f64", "two-way", "matrix", "gauss-seidel", "iqn-ils", small="b"),
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


def _draws(cell, count=None) -> list:
    """The cell's scenarios: for each loop gain the first seed whose exit
    under the compact rule is decided with margin, in the reference's own
    float64 arithmetic (the same seeds on every platform).  A sequence runs
    the first; *count* asks for more scenarios than :data:`GAINS` has."""
    gains = GAINS if count is None else [GAINS[i % len(GAINS)] for i in range(count)]
    out, seed = [], 0
    for gain, sign in gains:
        while True:
            draw = sg.Draw(seed, gain, sign)
            seed += 1
            if cell.sixteen or cell.acceleration != "none":
                break
            if cs.Stored(cell, draw).plain_exit("compact")["margin"] >= EXIT_MARGIN:
                break
            assert seed < 40 * len(gains), f"{cell.id}: no seed decides its exit with margin"
        out.append(draw)
    return out[:1] if _sequenced(cell) else out


def _run(cell, twin: bool = False, count=None) -> Run:
    key = (cell, twin, count)
    if key in _RUNS:
        return _RUNS[key]
    d = cell.domain
    with cd.entered(d):
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
    _RUNS[key] = Run(built, refs, solves, straight, saved, loaded)
    return _RUNS[key]


def _forget(cell) -> None:
    for key in [k for k in _RUNS if k[0] == cell]:
        del _RUNS[key]


def _eps(dtype) -> float:
    return float(cd.finfo(dtype).eps)


def _resolution(cell) -> float:
    """What a reported residual resolves, in tolerances: each entry's change
    carries a rounding of a few eps of its field's magnitude, ``eps / rtol``
    in the norm's units whatever the change.  Two of those, at the coarsest
    member's eps (the bound the tie's cells of this battery use)."""
    return 2.0 * _eps(cell.domain.coarsest) / cell.rtol


def _starts(cell, run: Run) -> list:
    """The iterate each solve's loop started from, as the reference names it.

    The state before the step; under a predictor the guess the group
    documents, from the states the earlier steps returned: none before two
    are stored (and at the second step: the count is 1), linear ``2 x_n -
    x_{n-1}`` with two, quadratic ``3 x_n - 3 x_{n-1} + x_{n-2}`` from the
    third on.  ``None`` under ``run_adaptive``, whose reported solve is its
    last half step's, started where the harness does not see.
    """
    if cell.domain.adaptive:
        return [None] * len(run.solves)
    before = [cs.state_of(cell, s, pre=True) for s in run.solves]
    if not cell.domain.predictor:
        return before
    returned = [cs.state_of(cell, s) for s in run.solves]
    starts = []
    for k in range(len(run.solves)):
        if k < 2:
            starts.append(before[k])
        elif k == 2:
            starts.append({n: 2.0 * returned[1][n] - returned[0][n] for n in ("p", "q")})
        else:
            starts.append({n: 3.0 * returned[k - 1][n] - 3.0 * returned[k - 2][n]
                           + returned[k - 3][n] for n in ("p", "q")})
    return starts


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
    for k, (s, ref, start) in enumerate(zip(run.solves, run.refs, _starts(cell, run))):
        r = s.report
        x = cs.state_of(cell, s)
        where = f"{cell.id}, solve {k}"
        if start is None:
            # run_adaptive: the residual of the state it returns, and the claim.
            want = ref.in_tolerances(ref.residual(ref.one_pass(x), x, "compact"))
            other = ref.in_tolerances(ref.residual(ref.one_pass(x), x, "delivered"))
            assert r["converged"] is True, (where, r)
            assert ref.distance(x, "compact") <= ref.K, (where, ref.distance(x), ref.K)
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
        assert abs(r["residual"] - want) <= resolution, (
            f"{where}: reported residual {r['residual']!r}; the compact readings give "
            f"{want!r} (resolution {resolution:.2e}; read as delivered: {other!r})")
        told_apart |= abs(other - want) > 4.0 * resolution
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
        # with margin; a sequence's later steps start where the earlier
        # ones left them, and at least one of them is decided with margin.
        need = 1 if _sequenced(cell) else len(run.solves)
        assert predicted >= need, f"{cell.id}: {predicted} pass counts predicted, of {need}"


def _numbers_agree(a, b, rel: float) -> bool:
    if isinstance(a, (bool, np.bool_, str, type(None))) or isinstance(b, (str, type(None))):
        return a == b
    a, b = float(a), float(b)
    if (math.isnan(a) and math.isnan(b)) or a == b:
        return True
    return rel > 0.0 and abs(a - b) <= rel * max(abs(a), abs(b))


def _same_solve(cell, mapped, twin, what: str) -> None:
    """*mapped* (the edge-mapped graph's) is *twin*'s: the state, the report, ``_meta``.

    To the bit: the two steps evaluate the same passes on the same numbers
    and read the same values.  **Not in the mixed-dtype domain**, where
    they are not one program: the twin's plain edge casts the small field
    to the large member's dtype and the norm reads the cast value (a
    float32 field widened to float64 and squared there; a float64 field
    rounded to float32, at float32's eps), while the edge-mapped graph
    reads the stored field itself; and with the small field in float64 the
    twin scatters its float32 rounding where the mapping scatters the
    float64 value and rounds the sum.  There: the verdict and the pass
    count the same, the state to float32 rounding, the residual to its
    resolution, the numbers derived from residual ratios to 2% (as
    ``test_coupling_interface_side.TWIN_RTOL`` has for float32), and the
    floor's slot only where the small field is the float32 one (the twin's
    cast reading of a float64 field is counted at float32's eps:
    :func:`_floor` states the edge-mapped graph's).
    """
    mixed = cell.domain.dtype_a != cell.domain.dtype_b
    where = f"{cell.id}, {what}"
    for name in ("a", "b"):
        xa, xb = mapped.x(name), twin.x(name)
        if not mixed:
            assert cd.bitwise(xa, xb), f"{where}: {name}.x differs from the twin's"
        else:
            xa, xb = xa.astype(np.float64), xb.astype(np.float64)
            bound = 64.0 * _eps(jnp.float32) * np.max(np.abs(xb))
            assert np.max(np.abs(xa - xb)) <= bound, (
                f"{where}: {name}.x is {np.max(np.abs(xa - xb)):.3e} from the twin's")
    ra, rb = mapped.report, twin.report
    assert sorted(ra) == sorted(rb), (where, sorted(ra), sorted(rb))
    differ = {}
    for key in ra:
        if not mixed:
            same = _numbers_agree(ra[key], rb[key], 0.0)
        elif key == "residual":
            same = abs(float(ra[key]) - float(rb[key])) <= _resolution(cell)
        else:
            same = _numbers_agree(ra[key], rb[key], 2e-2)
        if not same:
            differ[key] = (ra[key], rb[key])
    assert not differ, f"{where}: (edge-mapped, twin) {differ}"
    assert sorted(mapped.meta) == sorted(twin.meta), (where, sorted(mapped.meta))
    if not mixed:
        slots = [key for key in mapped.meta if not cd.bitwise(mapped.meta[key], twin.meta[key])]
        assert not slots, f"{where}: _meta slots differ from the twin's: {slots}"
    elif cell.small == "a":
        assert cd.bitwise(cs.slot_of(mapped), cs.slot_of(twin)), (
            where, cs.slot_of(mapped), cs.slot_of(twin))


def _check_twin(cell, count=None) -> None:
    """The edge-mapped pair reports what its marker-side twin reports."""
    mapped, twin = _run(cell, count=count), _run(cell, twin=True, count=count)
    assert len(mapped.solves) == len(twin.solves) > 0
    for k, (s, t) in enumerate(zip(mapped.solves, twin.solves)):
        _same_solve(cell, s, t, f"solve {k}")
    if cell.domain.restart:
        _same_solve(cell, mapped.saved, twin.saved, "the checkpointed graph")
        _same_solve(cell, mapped.loaded, twin.loaded, "the loaded graph")


def _check_claim(cell, count=None) -> None:
    """CPL-191 on *cell*: every solve converges, within ``K`` tolerances."""
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


def _same_graph(a, b) -> list:
    """What differs between two snapshots of one graph: members, report, ``_meta``."""
    out = [f"{n}.x" for n in ("a", "b") if not cd.bitwise(a.x(n), b.x(n))]
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


def _check_batch(cell, count=None) -> None:
    """Each member of a ``vmap`` batch is its own unbatched solve, to the bit."""
    run = _run(cell, count=count)
    gm = run.built.gm
    passes = set()
    with cd.entered(cell.domain):
        for k, s in enumerate(run.solves):
            (alone,) = cd.run(cd.DOMAINS["f32"], gm, [s.params])
            assert not _same_graph(s, alone), (
                f"{cell.id}: member {k} of the batch differs from its unbatched solve: "
                f"{_same_graph(s, alone)}")
            passes.add(s.report["iterations"])
    if cell.acceleration == "none":
        assert len(passes) > 1, (
            f"{cell.id}: fixture premise: the members of the batch stop on different passes")


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
    has them; to 2% in the mixed-dtype domain, :func:`_same_solve`), and
    a usable bound covers the distance to the exact fixed point in the
    compact readings without being looser than the loop's own
    amplification allows.

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
    mixed = cell.domain.dtype_a != cell.domain.dtype_b
    skipped = () if cell.form == "sparse" else ("gradient_relative_error_bound",)
    usable = 0
    for k, (s, t, ref) in enumerate(zip(mapped.solves, twin.solves, mapped.refs)):
        ra, rb = s.report, t.report
        where = f"{cell.id}, solve {k}"
        assert ra["converged"] is not cell.sixteen, (where, ra)
        differ = {key: (ra[key], rb[key]) for key in ra if key not in skipped
                  and not _numbers_agree(ra[key], rb[key], 2e-2 if mixed else 1e-6)}
        assert sorted(ra) == sorted(rb) and not differ, f"{where}: (edge-mapped, twin) {differ}"
        if not ra["spectral_usable"]:
            continue
        usable += 1
        distance = ref.in_tolerances(ref.distance(cs.state_of(cell, s), "compact"))
        bound = float(ra["spectral_error_bound"])
        assert 0.0 < distance <= bound * (1.0 + 1e-6), (where, distance, bound)
        if ra["converged"]:
            assert bound <= 2.0 * ref.K * float(ra["residual"]), (where, bound, ref.K, ra)
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
    pair is within ``K`` tolerances (CPL-191).  In the 16-bit domains, at
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
    """CPL-191 under each stock acceleration, in the domains that run them."""
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


# Slow: sixteen more graphs compiled.
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
