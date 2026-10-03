"""Differential oracles over a strength-3 covering array of the coupling group's knobs.

Coupling's defects have repeatedly lived in *interactions* between
settings -- the deprecated ``"fori"`` loop with Aitken on a 16-bit group,
``"fori"`` with IQN-IMVJ dropping ``jacobian_reuse``, the interface norm
with a flux edge, a multi-rate group's predictor history -- which a table
of one knob at a time never reaches.  This module draws its configurations
from a covering array (:mod:`tests.property.covering_array`): every value
of every pair of knobs, and of every triple, appears in some row.

**The knobs** (:data:`DOMAINS`): ``solver`` (ift, fori) x
``iteration_mode`` x ``acceleration`` (all five) x ``convergence_norm``
(all three) x ``linear_solver`` (gmres, dense; ``"bicgstab"`` is not an
option, CPL-039) x ``predictor`` (all three) x ``subcycling`` x
``boundary_interpolation`` (linear, constant) x ``waveform_iterations``
(1, 2) x ``diagnostics`` x ``strict_convergence`` x the dtype (float32 in
process, float64 in a subprocess under ``jax_enable_x64``) x a budget
(``max_iterations`` 60 at rate 0.6, which converges, or 3 at rate 0.9,
which mostly does not -- the strict-convergence oracle needs both).
Constraints: under ``"fori"`` ``linear_solver`` and ``strict_convergence``
are inert (they warn), and without sub-cycling so are
``waveform_iterations`` and ``boundary_interpolation``.  The array has
:data:`ROWS` rows and covers every valid triple
(``test_the_array_covers_every_valid_triple``).

**The structure** (:func:`structure`) is one group of three relays of two
entries each (a cycle and a chord), a driver upstream and a reader
downstream, non-float leaves on four nodes (a typed PRNG key on the
driver), and, with sub-cycling, one member at half the macro timestep.

**The oracles**, per row, over four steps from a drawn state:

1. *The monolithic float64 reference* (:class:`~tests.property.coupled_topologies.LinearModel`):
   every node's defect within its rounding, the group's within the bound
   its reported residual gives, the reported residual equal to
   ``||F(x) - x||`` of the returned state to its rounding, a converged
   group within the tolerance its threshold gives of its exact fixed
   point, and the whole state within the propagated allowance of the
   one-linear-system solve.
2. *fori == ift*, in lock step (the solver twin takes the row's whole
   state before every step): equal passes and verdicts above the float
   floor, states to the round-off of that step's passes.
3. *diagnostics on == off*: states, ``_meta`` slots, passes and verdicts
   bit for bit (the diagnostics twin).
4. *strict_convergence agrees with the report*: the strict twin raises on
   exactly the steps the report calls unconverged (with
   ``waveform_iterations > 1``, also on a step whose earlier sweep hit the
   cap, CPL-052), and otherwise returns the same bits.
5. *Every usable bound bounds*: ``spectral_error_bound`` is at least the
   distance to the exact fixed point in the returned state's weights, and
   ``gradient_relative_error_bound`` at least the true relative error of
   the gradient in every scalar gain and bias of the group (slow: the
   Jacobian compiles the step again).

Budget: per push, the float32 rows of a one-way slice of the array
(:func:`~tests.property.covering_array.one_way_slice`: every value of
every knob at least once) -- each test one compiled graph; in the slow
lane, every row, and the float64 rows in subprocesses.  Every seed is the
row's index.

What this cannot see: a non-linear group (no closed form), multi-rate and
adaptive stepping (the schedule harness's business,
``test_differential_schedules.py``), and any fault in the definition of
"fixed point of the evaluated map" that the reference shares.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.group import CouplingGroup
from tests.core.test_coupling_solver_equivalence import residual_noise_floor
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property.covering_array import coverage, ipog, one_way_slice

# ---------------------------------------------------------------------------
# The knob space and the array
# ---------------------------------------------------------------------------

#: Every knob and its values.  ``budget`` is ``max_iterations`` and the
#: drawn rate together (see the module docstring).
DOMAINS = {
    "solver": ("ift", "fori"),
    "iteration_mode": ("gauss-seidel", "jacobi"),
    "acceleration": ("none", "fixed", "aitken", "iqn-ils", "iqn-imvj"),
    "convergence_norm": ("l2", "mixed", "interface"),
    "linear_solver": ("gmres", "dense"),
    "predictor": ("none", "linear", "quadratic"),
    "subcycling": (False, True),
    "boundary_interpolation": ("linear", "constant"),
    "waveform_iterations": (1, 2),
    "diagnostics": (False, True),
    "strict_convergence": (False, True),
    "dtype": ("float32", "float64"),
    "budget": ("ample", "starved"),
}


def valid(a: dict) -> bool:
    """The configurations ``CouplingGroup`` accepts quietly (no inert knob set)."""
    if a.get("solver") == "fori":
        if a.get("linear_solver", "gmres") != "gmres":
            return False
        if a.get("strict_convergence", False):
            return False
    if a.get("subcycling") is False:
        if a.get("waveform_iterations", 1) != 1:
            return False
        if a.get("boundary_interpolation", "linear") != "linear":
            return False
    return True


#: The strength-3 array.
ROWS = ipog(DOMAINS, 3, valid)
#: The per-push slice: the float32 rows in which every knob takes every value.
PER_PUSH = one_way_slice(ROWS, keep=lambda r: r["dtype"] == "float32")

_STEPS = 4
_TOLERANCE = 1e-4      # the l2 threshold: 43x the float32 floor of 6 entries
_RTOL = 1e-4           # the mixed / interface rtol: floor 9.5e-3 of the threshold 1
_BUDGET = {"ample": (60, 0.6), "starved": (3, 0.9)}


def structure(subcycling: bool) -> ct.Topology:
    """The covering array's graph: ``drv -> {g0, g1, g2} -> sink``, two entries a node."""
    nodes = (
        ct.TNode("drv", 2, 0, alpha=1.0, beta=1.0, leaves=("count", "key")),
        ct.TNode("g0", 2, 2, alpha=0.5, beta=1.0, leaves=("count", "tag", "flag")),
        ct.TNode("g1", 2, 1, alpha=-0.25, beta=0.0, leaves=("flag",),
                 timestep=0.5 if subcycling else 1.0),
        ct.TNode("g2", 2, 2, alpha=0.0, beta=-0.5, leaves=("tag",)),
        ct.TNode("sink", 2, 1, alpha=0.5, leaves=("tag",)),
    )
    edges = (ct.TEdge("g2", "g0", 0), ct.TEdge("drv", "g0", 1), ct.TEdge("g0", "g1", 0),
             ct.TEdge("g1", "g2", 0), ct.TEdge("g0", "g2", 1), ct.TEdge("g2", "sink", 0))
    return ct.Topology(nodes, edges, (("g0", "g1", "g2"),), "interactions")


def knobs(row: dict, **override) -> dict:
    """The ``CouplingGroup`` configuration of *row*, with *override* applied."""
    cap, _rho = _BUDGET[row["budget"]]
    g = {k: row[k] for k in ("solver", "iteration_mode", "acceleration", "convergence_norm",
                             "linear_solver", "predictor", "subcycling",
                             "boundary_interpolation", "waveform_iterations", "diagnostics",
                             "strict_convergence")}
    g["max_iterations"] = cap
    g.update(override)
    if g["solver"] == "fori":
        g["linear_solver"] = "gmres"
        g["strict_convergence"] = False
    if g["convergence_norm"] == "l2":
        g["tolerance"] = _TOLERANCE
    else:
        g["rtol"] = _RTOL
    if g["acceleration"] == "fixed":
        g["relaxation"] = 0.7
    if g["acceleration"] == "iqn-imvj":
        g["jacobian_reuse"] = 2
    return cg.live_knobs(g)


def _group_cfg(g: dict) -> dict:
    return dict(iteration_mode=g.get("iteration_mode", "gauss-seidel"),
                subcycling=bool(g.get("subcycling", False)),
                boundary_interpolation=g.get("boundary_interpolation", "linear"),
                convergence_norm=g.get("convergence_norm", "l2"),
                rtol=g.get("rtol", 1e-6), tolerance=g.get("tolerance", 1e-6))


def _threshold(g: dict) -> float:
    return float(g.get("tolerance", 1e-6)) if g.get("convergence_norm", "l2") == "l2" else 1.0


@functools.lru_cache(maxsize=None)
def _values(index: int) -> dict:
    row = ROWS[index]
    _cap, rho = _BUDGET[row["budget"]]
    topo = structure(row["subcycling"])
    return ct.draw_values(topo, np.random.default_rng(index), rho, nonnormal=bool(index % 2),
                          dtype=row["dtype"], group_cfgs=[_group_cfg(knobs(row))])


@functools.lru_cache(maxsize=6)
def _built(index: int, items: tuple) -> ct.Built:
    row = ROWS[index]
    return ct.build(structure(row["subcycling"]), dict(items), dtype=row["dtype"])


#: What every oracle but the strict one runs: the row with
#: ``strict_convergence`` off (a starved strict row raises; oracle 4 holds
#: the strict graph to this one, bit for bit, wherever it does not).
_BASE = {"strict_convergence": False}


def built_for(index: int, **override) -> tuple:
    g = knobs(ROWS[index], **{**_BASE, **override})
    return _built(index, tuple(sorted(g.items()))), g


def _run(index: int, **override) -> tuple:
    built, g = built_for(index, **override)
    return built, g, ct.run(built, _values(index), _STEPS)


def _model(index: int, g: dict) -> ct.LinearModel:
    row = ROWS[index]
    return ct.LinearModel(structure(row["subcycling"]), _values(index), dtype=row["dtype"],
                          group_cfgs=[_group_cfg(g)])


# ---------------------------------------------------------------------------
# The oracles
# ---------------------------------------------------------------------------


def assert_reproduces_the_monolithic_reference(index: int) -> None:
    """Oracle 1: every step against the float64 reference (and the leaves)."""
    built, g, traj = _run(index)
    model = _model(index, g)
    for k, step in enumerate(traj, start=1):
        where = f"row {index} step {k} {g}"
        model.check_step(step.pre, step.state, step.reports, thresholds=[_threshold(g)],
                         where=where)
        ct.check_leaves(built.topo, step.state, k, model.divider, where=where)


def _floor(g: dict, dtype: str, n_float: int) -> float:
    """The norm's float floor (``residual_noise_floor``) at *dtype*'s epsilon."""
    scale = float(np.finfo(np.dtype(dtype)).eps) / float(np.finfo(np.float32).eps)
    return scale * residual_noise_floor(g.get("convergence_norm", "l2"),
                                        g.get("rtol", 1e-6), n_float)


def _parity_report(rep: dict, g: dict) -> dict:
    """``coupling_diagnostics()``'s entry in the shape ``criterion_is_resolved`` reads."""
    amp = rep.get("amplification", float("nan"))
    return {"residual": rep["residual"],
            "amplification": amp if np.isfinite(amp) else 0.0,
            "step_scale": (float(g.get("relaxation", 1.0)) if g.get("acceleration") == "fixed"
                           else 1.0),
            "threshold": _threshold(g)}


def _rounding_per_pass(model: ct.LinearModel, pre: dict, state: dict) -> float:
    """Twice the worst node's relative rounding of one evaluation (two programs, one each).

    :meth:`LinearModel.rounding` at the returned state, over ``max|x|``:
    the forward error of each node's sum, ``T eps sum|term|`` -- what
    ``coupled_graphs.rounding_bound`` counts for float32, at the graph's
    own epsilon.
    """
    worst = 0.0
    for nd in model.topo.nodes:
        read = {}
        for i, _K in model.terms[nd.name][2]:
            e = model.topo.edges[i]
            src = pre if i in model.back else state
            read[(i,)] = np.asarray(src[e.src]["x"], np.float64)
        eps_i = model.rounding(nd.name, pre, read)
        ref = max(float(np.max(np.abs(np.asarray(state[nd.name]["x"], np.float64)))), 1e-300)
        worst = max(worst, float(np.max(eps_i)) / ref)
    return 2.0 * worst


def _full_state(gm) -> dict:
    """A shallow copy of the graph's whole state, ``_meta`` included (arrays are immutable)."""
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in gm._state.items()}  # noqa: SLF001


def _synced(target, source: dict) -> dict:
    """*target*'s state with every node state and every shared ``_meta`` slot from *source*.

    The slots both solvers keep -- the predictor history, IQN-IMVJ's
    ``V`` / ``W`` warm start -- mean the same on both (CPL-068), so a
    step from the synced state is a step of both solvers from one input.
    """
    state = _full_state(target)
    for k, v in source.items():
        if k != "_meta":
            state[k] = dict(v)
    meta = dict(state.get("_meta", {}))
    for k, v in source.get("_meta", {}).items():
        if k in meta and np.shape(meta[k]) == np.shape(v) and \
                np.asarray(meta[k]).dtype == np.asarray(v).dtype:
            meta[k] = v
    if meta:
        state["_meta"] = meta
    return state


def assert_fori_and_ift_agree(index: int) -> None:
    """Oracle 2: the solver twin takes the same passes and returns the same state.

    Lock-step: before each step the fori graph is given the ift graph's
    whole pre-step state (:func:`_synced`), so every step compares the
    two solvers from one input -- the documented parity (CPL-063) is a
    statement about one solve -- and a step on which they may differ
    (below) does not excuse the steps after it.  Equal ``iterations``
    and ``converged`` wherever the threshold clears the norm's float
    floor by 4x and neither estimate lies within its own rounding of the
    threshold (``criterion_is_resolved``); with equal passes, states to
    the round-off of that step's passes.
    """
    row = ROWS[index]
    if row["solver"] == "ift":
        b_i, g_ift = built_for(index)
        b_f, g_fori = built_for(index, solver="fori", diagnostics=True)
    else:
        b_f, g_fori = built_for(index, diagnostics=True)
        b_i, g_ift = built_for(index, solver="ift", diagnostics=False)
    values = _values(index)
    model = _model(index, g_ift)
    key = b_i.topo.group_key(0)
    n_float = sum(model.topo.node(m).n for m in model.topo.groups[0])
    floor = _floor(g_ift, row["dtype"], n_float)
    sweeps = int(g_ift.get("waveform_iterations", 1)) if g_ift.get("subcycling") else 1
    ct.set_initial(b_i, values)
    ct.set_initial(b_f, values)
    p_i, p_f = ct.params_for(b_i, values), ct.params_for(b_f, values)
    compared = 0
    for k in range(1, _STEPS + 1):
        b_f.gm._store_state(_synced(b_f.gm, _full_state(b_i.gm)))   # noqa: SLF001
        pre = ct._snapshot(b_i.gm, {})                                # noqa: SLF001
        b_i.gm.step(params=p_i)
        b_f.gm.step(params=p_f)
        d_i = dict(b_i.gm.coupling_diagnostics()[key])
        d_f = dict(b_f.gm.coupling_diagnostics()[key])
        s_i, s_f = ct._snapshot(b_i.gm, {}), ct._snapshot(b_f.gm, {})  # noqa: SLF001
        same = d_f["iterations"] == d_i["iterations"]
        resolved = (cg.criterion_is_resolved(_parity_report(d_f, g_fori), floor)
                    and cg.criterion_is_resolved(_parity_report(d_i, g_ift), floor))
        if _threshold(g_ift) >= 4.0 * floor and (same or resolved):
            assert same, (f"row {index} step {k}: from one state fori took "
                          f"{d_f['iterations']} passes, ift {d_i['iterations']}, above the "
                          f"float floor ({d_f}, {d_i})")
            assert d_f["converged"] == d_i["converged"], f"row {index} step {k}: verdicts"
        if not same:
            continue    # below the floor: adjacent passes, documented
        passes = int(d_f["iterations"]) * sweeps
        bound = passes * _rounding_per_pass(model, pre, s_f)
        gap = cg.relative_gap(s_f, s_i)
        assert gap <= bound, (f"row {index} step {k}: states {gap:.3e} apart relatively "
                              f"after {passes} identical passes from one state "
                              f"(bound {bound:.3e})")
        compared += 1
    assert compared, f"row {index}: no step took the same passes under both solvers"


def assert_diagnostics_are_inert(index: int) -> None:
    """Oracle 3: the diagnostics twin returns the same bits, ``_meta`` slots and verdicts."""
    row = ROWS[index]
    _b, _g, a = _run(index)
    _b, _g, b = _run(index, diagnostics=not row["diagnostics"])
    off, on = (a, b) if not row["diagnostics"] else (b, a)
    for k, (s0, s1) in enumerate(zip(off, on), start=1):
        moved = cg.bitwise_differences(s0.state, s1.state)
        assert not moved, f"row {index} step {k}: diagnostics=True moved {moved}"
        changed = [s for s in s0.metas[0] if s not in s1.metas[0]
                   or s0.metas[0][s].tobytes() != s1.metas[0][s].tobytes()]
        assert not changed, f"row {index} step {k}: diagnostics=True changed slots {changed}"
        if s0.reports and s1.reports:
            assert (s0.reports[0]["iterations"], s0.reports[0]["converged"]) == (
                s1.reports[0]["iterations"], s1.reports[0]["converged"]), (
                index, k, s0.reports[0], s1.reports[0])


def assert_strict_convergence_agrees_with_the_report(index: int) -> None:
    """Oracle 4: the strict twin raises exactly where the report says it must.

    ``strict_convergence`` raises when a solve exits at its cap with its
    estimate outside the threshold (CPL-033), so it raises on a step
    exactly when that step's report says ``converged=False`` -- and, with
    ``waveform_iterations > 1``, also on a step whose *earlier* sweep hit
    the cap while the last converged (the report's ``iterations`` is then
    the cap: CPL-052, CPL-122).  Where it does not raise it changes no bit.
    """
    row = ROWS[index]
    if row["solver"] == "fori":
        # Under "fori" the knob is read nowhere: it must say so at
        # construction rather than turn silently (CPL-004).
        with pytest.warns(UserWarning, match="strict_convergence"):
            CouplingGroup(nodes=frozenset(structure(False).groups[0]), solver="fori",
                          strict_convergence=True)
        return
    # The row's own budget, and two passes -- which leaves a group at rate
    # 0.6 or 0.9 short of its threshold -- so both outcomes are seen.  Per
    # push the strict twin runs without diagnostics (their spectral
    # machinery is most of a compile, 11.7 s on CI for the row that has
    # them); the slow lane holds the strict twin with the row's own.
    own = {} if index not in PER_PUSH else {"diagnostics": False}
    for budget in (own, {"max_iterations": 2, "diagnostics": False}):
        _strict_against_the_report(index, **budget)


def _strict_against_the_report(index: int, **override) -> None:
    row = ROWS[index]
    _b, g_loose, loose = _run(index, **override)
    built, _g = built_for(index, strict_convergence=True, **override)
    values = _values(index)
    ct.set_initial(built, values)
    params = ct.params_for(built, values)
    waveform = int(g_loose.get("waveform_iterations", 1)) if g_loose.get("subcycling") else 1
    for k, step in enumerate(loose, start=1):
        rep = step.reports[0]
        try:
            built.gm.step(params=params)
        except Exception as exc:                     # noqa: BLE001 - equinox's error_if
            assert "without converging" in str(exc), exc
            capped = rep["iterations"] >= g_loose["max_iterations"]
            assert (not rep["converged"]) or (waveform > 1 and capped), (
                f"row {index} step {k}: strict_convergence raised where the report says "
                f"converged ({rep})")
            return
        assert rep["converged"], (
            f"row {index} step {k}: the report says unconverged ({rep}) and "
            "strict_convergence did not raise")
        moved = cg.bitwise_differences(step.state, ct._snapshot(built.gm, {}))  # noqa: SLF001
        assert not moved, f"row {index} step {k}: strict_convergence moved {moved}"


def assert_the_spectral_bound_bounds(index: int) -> None:
    """Oracle 5a: a usable ``spectral_error_bound`` >= the distance, returned-state weights.

    Only ``solver="ift"`` with ``diagnostics=True`` computes the bound; every
    other row must leave every bound unusable (CPL-012, CPL-092), and a
    ``"fori"`` row without diagnostics must report nothing at all.
    """
    row = ROWS[index]
    _b, g, traj = _run(index)
    model = _model(index, g)
    computes = row["solver"] == "ift" and row["diagnostics"]
    for k, step in enumerate(traj, start=1):
        if row["solver"] == "fori" and not row["diagnostics"]:
            assert not step.reports, f"row {index} step {k}: fori reported {step.reports}"
            continue
        d = step.reports[0]
        if not computes:
            assert not d["spectral_usable"] and not d["gradient_bound_usable"], (
                f"row {index} step {k}: a bound claims usability where none was computed: {d}")
            continue
        if not d.get("spectral_usable"):
            continue
        dist = model.returned_weight_distance(0, step.pre, step.state)
        assert d["spectral_error_bound"] >= dist, (
            f"row {index} step {k}: spectral_error_bound {d['spectral_error_bound']:.4e} "
            f"below the true distance {dist:.4e} ({d})")


def assert_the_gradient_bound_bounds(index: int) -> None:
    """Oracle 5b: a usable gradient bound >= the true relative error, every member constant.

    One step from the drawn state.  The IFT gradient the step returns is
    ``jax.jacfwd`` of the compiled step; the truth is the derivative of
    the group's exact fixed point (:meth:`LinearModel.group_fixed_point`)
    by central differences in the reference's precision, ``h`` a
    millionth of the constant (exact for a bias, ``O(h**2)`` for a gain).
    Measured as CPL-093 reads it: ``||D (g_k - t*)|| <= bound ||D g_k||``,
    ``D`` the group's norm at the returned state.
    """
    row = ROWS[index]
    if not (row["solver"] == "ift" and row["diagnostics"]):
        return          # no gradient bound is computed; oracle 5a holds it unusable
    built, g = built_for(index)
    values = _values(index)
    ct.set_initial(built, values)
    params = ct.params_for(built, values)
    gm = built.gm
    pre = ct._snapshot(gm, {})                                  # noqa: SLF001
    step, state = gm._compiled_step, gm._state                  # noqa: SLF001
    ext = gm._resolve_external_inputs(None)                     # noqa: SLF001
    members = list(built.topo.groups[0])

    def group_x(p):
        out = step(state, ext, p)
        return jnp.concatenate([out[m]["x"] for m in members])

    jac = jax.jacfwd(group_x)(params)
    gm.step(params=params)
    d = gm.coupling_diagnostics()[built.topo.group_key(0)]
    if not d.get("gradient_bound_usable"):
        return
    after = ct._snapshot(gm, {})                                # noqa: SLF001
    model = ct.LinearModel(built.topo, values, dtype=row["dtype"], group_cfgs=[_group_cfg(g)])
    S, w, _rt, _rms = model.norm_parts(0, after)
    worst, where = 0.0, None
    for m in members:
        nd = built.topo.node(m)
        for leaf in [f"G{j}" for j in range(nd.ports)] + ["b"]:
            J = np.asarray(jac["nodes"][m][leaf], np.float64)
            base = np.asarray(values["nodes"][m]["G"][int(leaf[1:])] if leaf != "b"
                              else values["nodes"][m]["b"], np.float64)
            for entry in np.ndindex(base.shape):
                h = 1e-6 * max(1.0, abs(float(base[entry])))
                xs = []
                for sign in (1.0, -1.0):
                    m2 = ct.LinearModel(built.topo, _perturbed(values, m, leaf, entry, sign * h),
                                        dtype=row["dtype"], group_cfgs=[_group_cfg(g)],
                                        exact=True)
                    fp = m2.group_fixed_point(0, pre, after)
                    xs.append(np.concatenate([np.asarray(fp[mm], np.float64) for mm in members]))
                t_star = (xs[0] - xs[1]) / (2.0 * h)
                g_k = J[(slice(None),) + entry]
                den = float(np.linalg.norm(w * (S @ g_k)))
                if den == 0.0:
                    continue
                rel = float(np.linalg.norm(w * (S @ (g_k - t_star)))) / den
                if rel > worst:
                    worst, where = rel, (m, leaf, entry)
    assert d["gradient_relative_error_bound"] >= worst, (
        f"row {index}: gradient bound {d['gradient_relative_error_bound']:.3e} under the true "
        f"relative error {worst:.3e} of the gradient in {where} ({d})")


def _perturbed(values: dict, node: str, leaf: str, entry: tuple, h: float) -> dict:
    out = {"nodes": {k: {"G": [np.asarray(G, np.float64) for G in v["G"]],
                         "b": np.asarray(v["b"], np.float64), "x0": v["x0"]}
                     for k, v in values["nodes"].items()},
           "H": dict(values["H"])}
    target = out["nodes"][node]["b"] if leaf == "b" else out["nodes"][node]["G"][int(leaf[1:])]
    target[entry] += h
    return out


#: The oracles, by name, for the float64 subprocess.
ORACLES = {
    "reference": assert_reproduces_the_monolithic_reference,
    "solvers": assert_fori_and_ift_agree,
    "diagnostics": assert_diagnostics_are_inert,
    "strict": assert_strict_convergence_agrees_with_the_report,
    "spectral": assert_the_spectral_bound_bounds,
    "gradient": assert_the_gradient_bound_bounds,
}


def oracles_for(row: dict) -> list:
    return list(ORACLES)


# ---------------------------------------------------------------------------
# The array itself
# ---------------------------------------------------------------------------


def test_the_array_covers_every_valid_triple():
    """Every valid pair and triple of knob values appears in some row; no row is invalid.

    The array is regenerated here from the committed generator, so a
    change to the knob space or to the generator that loses coverage
    fails this test rather than thinning the search silently.
    """
    assert all(valid(r) for r in ROWS)
    for t in (1, 2, 3):
        covered, required, missing = coverage(ROWS, DOMAINS, t, valid)
        assert covered == required, f"strength {t}: {len(missing)} missing, e.g. {missing[:3]}"
    assert 40 <= len(ROWS) <= 80, len(ROWS)
    assert ipog(DOMAINS, 3, valid) == ROWS, "the generator is not deterministic"


def test_the_coverage_check_fails_on_a_thinned_array():
    """The check above can fail: drop any one row and some valid triple goes missing."""
    for drop in range(len(ROWS)):
        thinned = ROWS[:drop] + ROWS[drop + 1:]
        covered, required, _missing = coverage(thinned, DOMAINS, 3, valid)
        if covered < required:
            return
    pytest.fail("no single row of the array is needed for its triple coverage")


def test_the_per_push_slice_takes_every_value_of_every_knob():
    """The per-push rows hold every value of every knob, float32's dtype aside."""
    for knob, options in DOMAINS.items():
        seen = {ROWS[i][knob] for i in PER_PUSH}
        want = {"float32"} if knob == "dtype" else set(options)
        assert seen == want, (knob, seen)


# ---------------------------------------------------------------------------
# The per-row tests
# ---------------------------------------------------------------------------


def _row_params(slow_everywhere: bool = False, also_slow=frozenset()):
    """The float32 rows: the per-push slice unmarked, every other row slow.

    Every per-row test takes this one list, so ``scope="module"`` groups
    the tests by row: one row's graphs are built once and serve every
    oracle before the next row's are built (pytest groups by the
    parameter's index, which is only shared when the lists are equal).
    *also_slow* moves rows of the slice to the slow lane for one test.
    """
    for i, row in enumerate(ROWS):
        if row["dtype"] != "float32":
            continue
        slow = slow_everywhere or i not in PER_PUSH or i in also_slow
        yield pytest.param(i, id=f"r{i:02d}", marks=(pytest.mark.slow,) if slow else ())


#: Per-push rows under ``"ift"`` without diagnostics, whose diagnostics twin
#: compiles the spectral machinery (4-8 s cold on CI): the first keeps the
#: oracle on every push, the others run it in the slow lane.
_DIAGNOSTICS_TWIN_SLOW = frozenset(
    [i for i in PER_PUSH if ROWS[i]["solver"] == "ift" and not ROWS[i]["diagnostics"]][1:])


# Slow: the rows outside the per-push slice, each one compiled graph (1-5 s
# on CI, the spectral machinery the most of it under ift with diagnostics).
# Per push: tests/property/test_differential_coupling_interactions.py::test_every_row_reproduces_the_monolithic_reference
@pytest.mark.parametrize("index", list(_row_params()), scope="module")
def test_every_row_reproduces_the_monolithic_reference(index):
    assert_reproduces_the_monolithic_reference(index)


# Slow: as above; the solver twin compiles a second graph.
# Per push: tests/property/test_differential_coupling_interactions.py::test_every_row_takes_the_same_passes_under_either_solver
@pytest.mark.parametrize("index", list(_row_params()), scope="module")
def test_every_row_takes_the_same_passes_under_either_solver(index):
    assert_fori_and_ift_agree(index)


# Slow: as above; the diagnostics twin compiles a second graph, and for an
# ift row without diagnostics that twin is the spectral machinery
# (_DIAGNOSTICS_TWIN_SLOW).
# Per push: tests/property/test_differential_coupling_interactions.py::test_every_row_returns_the_same_bits_with_diagnostics_on_or_off
@pytest.mark.parametrize("index", list(_row_params(also_slow=_DIAGNOSTICS_TWIN_SLOW)),
                         scope="module")
def test_every_row_returns_the_same_bits_with_diagnostics_on_or_off(index):
    assert_diagnostics_are_inert(index)


# Slow: as above; the strict twin compiles a second graph.
# Per push: tests/property/test_differential_coupling_interactions.py::test_strict_convergence_raises_exactly_where_the_report_says_unconverged
@pytest.mark.parametrize("index", list(_row_params()), scope="module")
def test_strict_convergence_raises_exactly_where_the_report_says_unconverged(index):
    assert_strict_convergence_agrees_with_the_report(index)


# Slow: as above (the row's own graph, already built by the tests above).
# Per push: tests/property/test_differential_coupling_interactions.py::test_a_usable_spectral_bound_bounds_the_distance_in_every_row
@pytest.mark.parametrize("index", list(_row_params()), scope="module")
def test_a_usable_spectral_bound_bounds_the_distance_in_every_row(index):
    assert_the_spectral_bound_bounds(index)


# Slow: the Jacobian of the step compiles it a third time (5-10 s on CI).
# Per push: tests/property/test_differential_gradient_bound.py::test_a_usable_gradient_bound_covers_every_scalar_constant_at_an_early_exit
@pytest.mark.parametrize("index", list(_row_params(slow_everywhere=True)), scope="module")
def test_a_usable_gradient_bound_bounds_the_error_in_every_row(index):
    assert_the_gradient_bound_bounds(index)


# ---------------------------------------------------------------------------
# float64, in a subprocess
# ---------------------------------------------------------------------------

_F64_ROWS = [i for i, r in enumerate(ROWS) if r["dtype"] == "float64"]
_F64_CHUNKS = [_F64_ROWS[k:k + 6] for k in range(0, len(_F64_ROWS), 6)]


def _x64_main(indices: list) -> None:
    """Run every oracle of every row in *indices*; called in a subprocess under x64."""
    assert jax.config.jax_enable_x64, "the float64 rows need jax_enable_x64"
    failures = []
    for i in indices:
        for name in oracles_for(ROWS[i]):
            try:
                ORACLES[name](i)
            except AssertionError as exc:
                failures.append(f"row {i} {name}: {exc}")
    print(json.dumps({"rows": indices, "failures": failures}))
    if failures:
        raise SystemExit(1)


# Slow: a subprocess per chunk (jax under jax_enable_x64), ~6 rows of up to
# four compiled graphs each.
# Per push: tests/property/test_differential_coupling_interactions.py::test_every_row_reproduces_the_monolithic_reference
@pytest.mark.slow
@pytest.mark.parametrize("chunk", range(len(_F64_CHUNKS)), ids=lambda c: f"chunk{c}")
def test_the_float64_rows_hold_every_oracle_in_a_subprocess(chunk):
    """Every oracle on the float64 rows, under ``jax_enable_x64`` in a child process.

    ``jax_enable_x64`` is process-global, so the rows run in a child (the
    reference's precision must be finer than float64 there:
    ``coupled_topologies.reference_precision_ok``).
    """
    if not ct.reference_precision_ok():
        pytest.skip("numpy's longdouble is float64 on this platform: the reference "
                    "cannot resolve a float64 graph's rounding")
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu",
               PYTHONPATH=os.pathsep.join(sys.path))
    code = ("import sys; from tests.property.test_differential_coupling_interactions "
            f"import _x64_main; _x64_main({_F64_CHUNKS[chunk]!r})")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=1500)
    assert proc.returncode == 0, (proc.stdout[-4000:] + proc.stderr[-4000:])
