"""``convergence_norm="interface"`` reads a static mapped edge on its compact side.

The interface norm pools every entry it reads into one RMS.  Read as
delivered, an internal edge whose mapping scatters a small field onto a
large one (30 marker forces onto a grid of ``N`` cells) puts ``N`` entries
into the pool of which a few dozen change, and the criterion is diluted: a
converged group's marker forces were 23, 39, 134 and 459 tolerances from
their fixed point at ``N`` = 1e3 to 1e6
(``benchmarks/results/interface_norm_dilution``).

**The rule.**  An internal edge whose static mapping delivers *more*
entries than its source field holds is read at its **source value**: the
field itself, before the mapping and so before the transform (the step
applies the mapping, then the transform).  A mapping onto fewer entries,
**a tie**, and an edge with no mapping are read as delivered.  The sizes
are the ones the mapping declares.

What is held here, on the library's own functions and on compiled graphs
(the property, its exact reference and the search's score are in
``tests/property/test_coupling_interface_side.py`` and
``test_coupling_targeted_search.py``):

* the residual, the float floor and the floor's eps each read the
  prescribed side -- the value, the entry count and the magnitude -- on
  the dense form, both sparse layouts and a mapping class of the caller's
  own, expanding, reducing and tied, with and without a transform, in
  float32 and float64;
* a group owns the ``reading_floor`` slot exactly where an edge is read
  *through* its mapping, the step writes the slot it was compiled with,
  and the three plans a graph builds of one group read every edge on the
  same side;
* a checkpoint written by a tree that recorded the slot for a group that
  no longer owns it loads, loudly, and reports;
* a diagnosed edge-mapped pair reports every number its marker-side twin
  reports, where the spectral analysis settles.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import itertools
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _coupled_block
from maddening.core.coupling import _interface_plan as ip
from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    _interface_readings,
    coupling_residual_interface,
    residual_precision_floor,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping, sparse_matrix_mapping
from maddening.core.edge import EdgeSpec, _delivered
from maddening.core.simulation.checkpoint import load_state, save_state
from tests.property import interface_side_graphs as sg
from tests.property.sysid_transform_grid import precision

RTOL = 1e-3


# ---------------------------------------------------------------------------
# The library's functions, edge by edge (no graph)
# ---------------------------------------------------------------------------


class _OwnStatic:
    """A static mapping class of the caller's own: ``target[i] = w *
    (source[i % n_source] + source[(i + 1) % n_source])``, its two sizes
    declared as ``add_edge`` requires."""

    kind = "own_static"
    mode = "consistent"

    def __init__(self, n_source, n_target, dtype):
        self.n_source, self.n_target = int(n_source), int(n_target)
        self._w = jnp.asarray(0.75, dtype)

    def params_pytree(self):
        return {"w": self._w}

    def apply(self, field, weights=None, geom=None):
        w = self._w if weights is None else weights["w"]
        rows = jnp.arange(self.n_target)
        return w * (field[rows % self.n_source] + field[(rows + 1) % self.n_source])

    def apply_T(self, field, weights=None, geom=None):
        raise NotImplementedError


def _matrix(n_source, n_target, dtype, scale=1.0):
    rng = np.random.default_rng(100 * n_source + n_target)
    return (scale * (0.5 + rng.random((n_target, n_source)))).astype(dtype)


def _dense(n_source, n_target, dtype, scale=1.0):
    return matrix_mapping(_matrix(n_source, n_target, dtype, scale))


def _sparse_gather(n_source, n_target, dtype, scale=1.0):
    """Each target row lists two sources (padded gather)."""
    index = np.stack([np.arange(n_target) % n_source, (np.arange(n_target) + 1) % n_source], axis=1)
    return sparse_matrix_mapping(index, _matrix(2, n_target, dtype, scale), n_source=n_source)


def _sparse_scatter(n_source, n_target, dtype, scale=1.0):
    """Each source row lists two targets (scatter layout): the weights'
    shape is ``(n_source, 2)`` whatever the target's size."""
    index = np.stack([np.arange(n_source) % n_target, (np.arange(n_source) + 1) % n_target], axis=1)
    return StaticSparseMapping(index, jnp.asarray(_matrix(2, n_source, dtype, scale)),
                               n_source=n_source, n_target=n_target,
                               counts=np.full(n_source, 2), layout="scatter")


FORMS = {
    "dense": _dense,
    "sparse-gather": _sparse_gather,
    "sparse-scatter": _sparse_scatter,
    "own-static": lambda n, m, dtype, scale=1.0: _OwnStatic(n, m, dtype),
}
#: ``(n_source, n_target)``: expanding, reducing, and a tie.
SIZES = {"expanding": (3, 7), "reducing": (7, 3), "tie": (4, 4)}


def _offset(v):
    return v + jnp.asarray(8.0, v.dtype)


TRANSFORMS = {"plain": None, "offset": _offset}
CELLS = list(itertools.product(sorted(FORMS), SIZES, TRANSFORMS, ("float32", "float64")))


def _cell(form, size, transform, dtype):
    n_source, n_target = SIZES[size]
    mapping = FORMS[form](n_source, n_target, dtype)
    edge = EdgeSpec("a", "b", "x", "u", mapping=mapping, transform=TRANSFORMS[transform])
    rng = np.random.default_rng(7)
    old = jnp.asarray(1.0 + rng.random(n_source), dtype)
    new = jnp.asarray(np.asarray(old) * (1.0 + 1e-3 * rng.standard_normal(n_source)), dtype)
    return edge, new, old


def _as_plain(*values):
    """The states and the one plain edge that read *values* as they are."""
    return [{"r": {"x": v}} for v in values], [EdgeSpec("r", "b", "x", "u")]


@pytest.mark.parametrize("form,size,transform,dtype", CELLS, ids=["-".join(c) for c in CELLS])
def test_the_residual_reads_each_static_mapped_edge_on_its_compact_side(
        form, size, transform, dtype):
    """The residual of one mapped edge is the residual of a plain edge over
    the prescribed reading -- the source field where the mapping expands,
    the delivered value otherwise -- to the bit: the value, the entry count
    and the magnitude it is divided by.  And it is not the other side's
    wherever the two differ."""
    with precision(dtype == "float64"):
        edge, new, old = _cell(form, size, transform, dtype)
        at_source = size == "expanding"
        sides = {"source": (new, old), "delivered": (_delivered(edge, new), _delivered(edge, old))}
        got = float(coupling_residual_interface({"a": {"x": new}}, {"a": {"x": old}}, [edge],
                                                0.0, RTOL))
        want = {}
        for side, values in sides.items():
            states, plain = _as_plain(*values)
            want[side] = float(coupling_residual_interface(*states, plain, 0.0, RTOL))
        assert got == want["source" if at_source else "delivered"], (got, want)
        assert got != want["delivered" if at_source else "source"], "the cell can tell the sides apart"
        assert got > 0.0
        (read,) = _interface_readings([edge], {"a": {"x": new}})
        assert read[0] is edge and read[1] == jnp.dtype(dtype)
        assert np.shape(read[2]) == ((SIZES[size][0],) if at_source else (SIZES[size][1],))
        if at_source:
            assert read[2] is new, "the stored field: no mapping, no transform"


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("size", sorted(SIZES))
def test_the_pool_counts_the_entries_of_the_side_that_is_read(form, size):
    """Pooled with a second edge that does not move: the mapped edge
    weighs in with the entry count of its prescribed side, not the
    other's (a right value with the large side's count is the dilution)."""
    with precision(True):
        edge, new, old = _cell(form, size, "plain", "float64")
        still = jnp.ones(5, jnp.float64)
        rest = EdgeSpec("c", "b", "x", "v")
        got = float(coupling_residual_interface(
            {"a": {"x": new}, "c": {"x": still}}, {"a": {"x": old}, "c": {"x": still}},
            [edge, rest], 0.0, RTOL))
        alone = float(coupling_residual_interface({"a": {"x": new}}, {"a": {"x": old}}, [edge],
                                                  0.0, RTOL))
        n_source, n_target = SIZES[size]
        n_read = n_source if size == "expanding" else n_target
        assert got == pytest.approx(alone * np.sqrt(n_read / (n_read + 5.0)), rel=1e-9)


def _narrow(v):
    return v.astype(jnp.float32)


@pytest.mark.parametrize("form", sorted(FORMS))
def test_the_float_floor_takes_its_eps_on_the_side_that_is_read(form):
    """Under x64, a float64 field whose edge delivers float32 (a narrowing transform):
    read at its source the reading is the stored float64 field, at that
    dtype's eps; read as delivered it is the float32 value, at the coarser
    eps.  A floor left on the delivered side of an expanding edge is 5e8
    times too large."""
    with precision(True):
        floors = {}
        for size, (n_source, n_target) in SIZES.items():
            edge = EdgeSpec("a", "b", "x", "u", mapping=FORMS[form](n_source, n_target, "float64"),
                            transform=_narrow)
            x = jnp.asarray(1.0 + np.arange(n_source), jnp.float64)
            assert _delivered(edge, x).dtype == jnp.float32
            floors[size] = float(residual_precision_floor(
                {"a": {"x": x}}, ["a"], "interface", 0.0, RTOL, [edge]))
        eps32, eps64 = (float(np.finfo(t).eps) for t in (np.float32, np.float64))
        assert floors["expanding"] == pytest.approx(PRECISION_FLOOR_ULPS * eps64 / RTOL, rel=1e-6)
        assert floors["reducing"] == pytest.approx(PRECISION_FLOOR_ULPS * eps32 / RTOL, rel=1e-6)
        assert floors["tie"] == floors["reducing"]


@pytest.mark.parametrize("form", ["dense", "sparse-gather", "sparse-scatter"])
def test_the_dead_band_is_asked_of_the_side_that_is_read(form):
    """Weights of 1e-6 deliver values under the dead band from a field
    above it.  An expanding edge is read at its source, which is active:
    the floor counts it.  A reducing edge and a tie are read as delivered,
    which the dead band excludes: nothing is read and the floor is 0."""
    atol = 1e-3
    with precision(True):
        floors, residuals = {}, {}
        for size, (n_source, n_target) in SIZES.items():
            edge = EdgeSpec("a", "b", "x", "u",
                            mapping=FORMS[form](n_source, n_target, "float64", scale=1e-6))
            x = jnp.asarray(1.0 + np.arange(n_source), jnp.float64)
            assert float(jnp.max(jnp.abs(_delivered(edge, x)))) < atol < float(jnp.max(x))
            floors[size] = float(residual_precision_floor(
                {"a": {"x": x}}, ["a"], "interface", atol, RTOL, [edge]))
            residuals[size] = float(coupling_residual_interface(
                {"a": {"x": 1.01 * x}}, {"a": {"x": x}}, [edge], atol, RTOL))
        eps64 = float(np.finfo(np.float64).eps)
        assert floors["expanding"] == pytest.approx(PRECISION_FLOOR_ULPS * eps64 / RTOL, rel=1e-6)
        assert floors["reducing"] == floors["tie"] == 0.0
        assert residuals["expanding"] > 1.0 and residuals["reducing"] == residuals["tie"] == 0.0


def test_a_source_side_reading_takes_no_weights_the_step_ran_with():
    """The weights a caller hands the step for one edge move a delivered
    reading and leave a source-side one where it was."""
    with precision(True):
        other = {"a.x->b.u": {"H": jnp.full((7, 3), 5.0, jnp.float64)}}
        edge, new, old = _cell("dense", "expanding", "plain", "float64")
        states = ({"a": {"x": new}}, {"a": {"x": old}})
        assert float(coupling_residual_interface(*states, [edge], 0.0, RTOL, mappings=other)) == (
            float(coupling_residual_interface(*states, [edge], 0.0, RTOL)))
        other = {"a.x->b.u": {"H": jnp.asarray(np.eye(3, 7), jnp.float64)}}
        edge, new, old = _cell("dense", "reducing", "plain", "float64")
        states = ({"a": {"x": new}}, {"a": {"x": old}})
        assert float(coupling_residual_interface(*states, [edge], 0.0, RTOL, mappings=other)) != (
            float(coupling_residual_interface(*states, [edge], 0.0, RTOL)))


# ---------------------------------------------------------------------------
# Compiled graphs: who owns the floor's slot, and the three plans of a group
# ---------------------------------------------------------------------------

#: ``kind -> (shape, sides of p -> q and q -> p, owns the reading_floor slot)``.
#: The mapping layouts and the dtypes are rotated over the kinds; the last
#: is a tie (both mappings square: delivered under the rule).
GRAPHS = {
    "gather-only": (sg.Shape("gather-only", 60, 5, "sparse", "jacobi", "float32"),
                    ("delivered", "delivered"), True),
    "two-way": (sg.Shape("two-way", 60, 5, "matrix", "gauss-seidel", "float64"),
                ("source", "delivered"), True),
    "scatter-only": (sg.Shape("scatter-only", 60, 5, "sparse-transposed", "jacobi", "float64"),
                     ("source", "source"), False),
    "tie": (sg.Shape("two-way", 6, 6, "sparse", "gauss-seidel", "float32"),
            ("delivered", "delivered"), True),
}
SLOT = "coupling_p+q_reading_floor"
DRAW = sg.Draw(seed=3, gain=0.6)


def _meta(gm) -> dict:
    return gm._state["_meta"]  # noqa: SLF001


def _stepped(shape, built):
    ref = sg.Reference(shape, DRAW)
    gm = built.gm
    gm.reset_state()
    for name in ("p", "q"):
        gm.set_node_state(name, {"x": jnp.asarray(ref.start[name], shape.dtype)})
    gm.step(params=ref.params(built))
    return ref


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def graph(request):
    """``(kind, shape, the built graph, the plans each site built of its group)``."""
    shape, _sides, _owns = GRAPHS[request.param]
    plans = []
    real = ip.interface_plan

    def spy(site):
        def _spied(*args, **kwargs):
            plan = real(*args, **kwargs)
            plans.append((site, plan))
            return plan
        return _spied

    patch = pytest.MonkeyPatch()
    # ``graph_manager`` reads the function off the module; the coupled block
    # imported the name.
    patch.setattr(ip, "interface_plan", spy("graph_manager"))
    patch.setattr(_coupled_block, "interface_plan", spy("coupled_block"))
    try:
        with precision(shape.dtype == "float64"):
            built = sg.build(shape)
            seeded = set(_meta(built.gm))
            ref = _stepped(shape, built)
            after_step = dict(_meta(built.gm))
            (report,) = built.gm.coupling_diagnostics().values()
    finally:
        patch.undo()
    return dict(kind=request.param, shape=shape, built=built, plans=plans, seeded=seeded,
                after_step=after_step, report=dict(report), ref=ref)


def test_every_plan_a_graph_builds_of_a_group_reads_each_edge_on_the_same_side(graph):
    """``validate()``, ``compile()`` and the trace of the group's block each
    build the group's plan; the side is the edge's own, so the three agree
    -- and it is the one the two ends' sizes prescribe."""
    _shape, sides, _owns = GRAPHS[graph["kind"]]
    want = {"p.x->q.u": sides[0], "q.x->p.u": sides[1]}
    sites = [site for site, _plan in graph["plans"]]
    assert sites.count("graph_manager") >= 2 and sites.count("coupled_block") >= 1, (
        f"both patches took effect: {sites}")
    for site, plan in graph["plans"]:
        assert {r.key: r.norm_side for r in plan.internal} == want, site
    sizes = sg.node_sizes(graph["shape"])
    for record in graph["plans"][-1][1].internal:
        n_source, n_target = sizes[record.source[0]][0], sizes[record.target[0]][1]
        assert record.norm_side == sg.side_of(n_source, n_target)
        assert (record.mapping.n_source, record.mapping.n_target) == (n_source, n_target)


def test_a_group_owns_the_floor_slot_where_an_edge_is_read_through_its_mapping(graph):
    """``compile()`` seeds ``reading_floor`` for a group with an edge read as
    delivered through a mapping, and for no other: a group whose mapped
    edges are all read at their source reads no weights.  The step writes
    exactly the slots it was compiled with (a slot seeded and never
    written, or written and never seeded, is two plans disagreeing), and
    a scan carries them."""
    _shape, _sides, owns = GRAPHS[graph["kind"]]
    assert (SLOT in graph["seeded"]) == owns
    assert set(graph["after_step"]) == graph["seeded"]
    if owns:
        assert np.isfinite(np.asarray(graph["after_step"][SLOT])), "the step measured it"
    built, shape = graph["built"], graph["shape"]
    with precision(shape.dtype == "float64"):
        built.gm.run_scan(2, params=graph["ref"].params(built))
        assert set(_meta(built.gm)) == graph["seeded"]


def test_the_report_of_a_converged_pair_is_within_K_tolerances_in_the_compact_readings(graph):
    """The claim, on each kind: converged, and the distance to the exact
    fixed point in the compact readings is at most ``K`` tolerances.  (One
    jitted step of the compiled graph; the float64 kinds run under x64.)"""
    shape, built = graph["shape"], graph["built"]
    with precision(shape.dtype == "float64"):
        ref = _stepped(shape, built)
        (report,) = built.gm.coupling_diagnostics().values()
        state = {name: np.asarray(built.gm.get_node_state(name)["x"], np.float64)
                 for name in ("p", "q")}
    assert bool(report["converged"])
    assert 0.0 < ref.distance(state, "compact") <= ref.K
    restated = ref.residual(ref.one_pass(state), state, "compact")
    tight = 1e-9 if shape.dtype == "float64" else 2e-3
    assert abs(float(report["residual"]) - restated) <= tight * restated


# ---------------------------------------------------------------------------
# A checkpoint from a tree that recorded the slot for such a group
# ---------------------------------------------------------------------------


def _assert_same_report(a, b, rel=0.0):
    assert sorted(a) == sorted(b)
    for key, value in a.items():
        if isinstance(value, (bool, np.bool_, str, type(None))):
            assert value == b[key], key
        else:
            assert float(value) == pytest.approx(float(b[key]), rel=rel, abs=0.0, nan_ok=True), key


def test_a_checkpoint_that_carries_the_slot_of_a_group_that_no_longer_owns_it_loads_loudly(
        tmp_path):
    """Before the rule, every interface-norm group with a mapped internal
    edge recorded ``reading_floor``.  An archive written then for a group
    whose mapped edges are now all read at their source carries a slot
    this graph does not have: the load says so, drops it, and the report
    takes the floor from the loaded state (which is all a source-side
    reading depends on)."""
    shape = GRAPHS["scatter-only"][0]
    with precision(True):
        built = sg.build(shape)
        _stepped(shape, built)
        gm = built.gm
        (before,) = gm.coupling_diagnostics().values()
        path = save_state(gm, tmp_path / "now.npz")
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        assert f"_meta/{SLOT}" not in arrays
        # As a development tree before the rule wrote it: the slot, finite.
        arrays[f"_meta/{SLOT}"] = np.asarray(4.0 * np.finfo(np.float64).eps / sg.RTOL)
        old = tmp_path / "before_the_rule.npz"
        np.savez(old, **arrays)

        fresh = sg.build(shape).gm
        with pytest.warns(RuntimeWarning,
                          match=rf"not present in this graph: \['{re.escape(SLOT)}'\]"):
            load_state(fresh, old)
        assert SLOT not in _meta(fresh)
        (after,) = fresh.coupling_diagnostics().values()
        _assert_same_report(after, before)
        fresh.step(params=sg.Reference(shape, DRAW).params(built))
        assert SLOT not in _meta(fresh)


def test_a_checkpoint_of_a_group_that_owns_the_slot_brings_it_back(tmp_path):
    """The other half: where an edge is still read through its mapping
    the slot round-trips, silently."""
    shape = GRAPHS["two-way"][0]
    with precision(True):
        built = sg.build(shape)
        _stepped(shape, built)
        (before,) = built.gm.coupling_diagnostics().values()
        path = save_state(built.gm, tmp_path / "two_way.npz")
        fresh = sg.build(shape).gm
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_state(fresh, path)
        assert np.asarray(_meta(fresh)[SLOT]) == np.asarray(_meta(built.gm)[SLOT])
        (after,) = fresh.coupling_diagnostics().values()
        _assert_same_report(after, before)


# ---------------------------------------------------------------------------
# The diagnostics: one side for the criterion, the floor and the spectrum
# ---------------------------------------------------------------------------

#: Four markers: the loop closes on eight numbers, which the report's eight
#: Krylov steps resolve, so ``spectral_usable`` holds and the bounds are
#: finite numbers the comparison can see move.
SETTLED = sg.Shape("two-way", 40, 4, "matrix", "jacobi", "float64", diagnostics=True)


def test_a_diagnosed_edge_mapped_pair_reports_every_number_of_its_marker_side_twin():
    """The twin applies the scatter inside the grid node, so both of its
    internal edges carry marker-sized values and its norm reads the compact
    side by construction.  The criterion, the float floor behind
    ``precision_limited`` and the spectral analysis each read the edges for
    themselves; with all three on one side the two reports are the same
    numbers -- the spectral radius and both bounds included."""
    with precision(True):
        ref = sg.Reference(SETTLED, DRAW)
        reports = {}
        for which, built in (("edge-mapped", sg.build(SETTLED)),
                             ("twin", sg.marker_side_twin(SETTLED, ref))):
            _stepped(SETTLED, built)
            (report,) = built.gm.coupling_diagnostics().values()
            reports[which] = dict(report)
    mapped, twin = reports["edge-mapped"], reports["twin"]
    assert sorted(mapped) == sorted(twin)
    assert mapped["converged"] and mapped["spectral_usable"] and twin["spectral_usable"]
    assert np.isfinite(mapped["spectral_error_bound"])
    _assert_same_report(mapped, twin, rel=1e-6)
