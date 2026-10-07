"""Twins of an edge-mapped graph whose interface reading is its own.

``convergence_norm="interface"`` reads what each internal edge delivers:
the source field through the edge's mapping and then its transform.  The
node-inlined twin of ``tests/property/test_differential_geometry_edges.py``
moves a mapping into the *target* node, so its edges deliver the raw
source field and its interface norm is another norm: reports of the two
graphs under that norm can agree only loosely.  The two twins here keep
the reading (``tests/property/geometry_graphs.py``):

* :func:`~tests.property.geometry_graphs.transform_twin` writes a static
  mapping as its edge's transform.  Same state, same reading: **every
  number of the report but the gradient bound is the edge-mapped graph's**
  (measured bit for bit on jaxlib 0.11.0, under both schedules, Aitken
  and IQN-ILS, in float32 and float64; held here to :data:`ULPS` of the
  dtype's ``eps``, since they are two programs).  The gradient bound
  probes every parameter of the step, and the mapping's weights are
  parameters there and constants of a transform here.
* :func:`~tests.property.geometry_graphs.relay_twin` moves the mapping
  into a relay on the *source*, whose state field **is** the mapped
  value: a plain edge delivers it, so the reading is the same entry for
  entry, and a geometry the mapping reads can sit inside the relay.  The
  state has the relay's field in it, so what is equal is what the reading
  decides -- every iterate of an unaccelerated solve, the residual, the
  pass count, the verdict, ``rho_spectral`` where the spectrum is
  resolved, the flags -- and ``spectral_error_bound`` is equal only to
  :data:`RELAY_BOUND_GAP`: its factor is the norm of a resolvent
  compressed onto a Krylov basis of the *state*, which is another basis
  with the relay's field in it (measured 1.1% to 3.4% apart under
  Gauss-Seidel).  The float floor is not compared (13% apart: the norm
  counts an inner product for a mapping it evaluates and none for a field
  it reads as stored), nor the gradient bound (its norm is the raw source
  fields, which are other fields here).  Under Jacobi the relay doubles
  the scalars the pass carries (18 for 9 on the two-body graph), past
  what eight Krylov steps resolve, so the relay comparison is made under
  Gauss-Seidel.

**What this proves on today's tree** (no geometry): the library reads a
static mapped internal edge as the step delivers it, and both twins say
so; a fault seeded in the library's reading (the edge read without its
mapping) breaks both equalities (measured when this file was added: the
two residuals 8.8% apart; ``docs/developer_guide/testing_standards.md``).  And that the relay
twin is the edge-mapped graph where it has a source-anchored *moving*
geometry: on the solve path, which 0.4.0 supports, the two step bit for
bit.

**What waits** (:data:`~tests.property.geometry_graphs.RELAY_INTERFACE_CASES`,
in the harness's phase-1 block): the interface norm over a geometry edge
is refused at compile in 0.4.0.  Each such case asserts the refusal and
that its relay twin compiles and reports today; when
``DIAGNOSTICS_READ_GEOMETRY`` is set the slow test compares the two
reports as the static cases are compared.
"""

from __future__ import annotations

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from tests.property import coupled_graphs as cg
from tests.property import geometry_graphs as gg
from tests.property import test_coupling_targeted_search as linear

KEY = "F+P"
#: Float ``eps`` of the group's dtype two programs of the same arithmetic
#: may put between a reported number and its twin's.  Zero was measured;
#: an Arnoldi process turns one rounding of a product into a few of the
#: radius, and 2**10 of them (1.2e-4 in float32) is still far under what
#: a reading without its mapping moves (the seeded fault: the residual
#: 8.8% apart on the per-push case).
ULPS = 2.0 ** 10
#: How far the relay twin's ``spectral_error_bound`` factor may be from
#: the edge-mapped graph's (module docstring): three times the largest gap
#: measured.
RELAY_BOUND_GAP = 0.10
#: Every number of a report but the gradient bound's two keys.
NUMBERS = ("residual", "amplification", "error_estimate", "gradient_error_estimate",
           "rho_spectral", "spectral_error_bound")
FLAGS = ("converged", "ratio_usable", "spectral_usable", "precision_limited")

_P_HOLDS = dict(down="target", up="source")
_SOURCES = dict(down="source", up="source")


def _interface(dtype: str, **knobs) -> gg.Case:
    return gg.case(f"{dtype} {knobs}", dtype=dtype, adv=0.3, **_P_HOLDS,
                   group=dict(convergence_norm="interface", rtol=1e-6, diagnostics=True,
                              **knobs))


#: A solve stopped at a cap of three passes (a residual far above its
#: float floor, so it is a number and not a rounding).  Per push: one
#: case, whose edge-mapped graph both twins are compared with (a compile
#: of a group with diagnostics is seconds; :func:`_edge` keeps it).  The
#: other dtype, the other schedule and the accelerations: the slow lane.
STATIC_PER_PUSH = [_interface("float32", max_iterations=3)]
STATIC_SLOW = [
    _interface("float64", max_iterations=3),
    _interface("float64", max_iterations=3, iteration_mode="jacobi"),
    _interface("float32", max_iterations=3, iteration_mode="jacobi"),
    _interface("float64", max_iterations=6, acceleration="aitken"),
    _interface("float32", max_iterations=6, acceleration="iqn-ils"),
]
#: The relay twin's comparison is the slow lane's (a third compile); per
#: push the relay twin is held to the edge-mapped graph on the solve path
#: (:data:`MOVING`) and the seeded reading fault to the transform twin.
RELAY_SLOW = [_interface("float32", max_iterations=3), _interface("float64", max_iterations=3)]


@functools.lru_cache(maxsize=2)
def _edge(c: gg.Case):
    """The compiled edge-mapped graph of static case *c* (two static
    mapped internal edges)."""
    with gg.x64(c.needs_x64):
        static = gg.static_twin(c)
        assert all(e.geometry is None for e in static.edges) and sum(
            e.mapping is not None for e in static.edges) == 2, "premise: two static mapped edges"
        return gg.build(static)


def _report(gm) -> dict:
    """``coupling_diagnostics()`` of the two-body group, with the floor it used."""
    report = dict(gm.coupling_diagnostics()[KEY])
    report["floor"] = linear._reported_floor(gm, KEY, cg.group_meta(gm, KEY), report)  # noqa: SLF001
    return report


def _close(a: float, b: float, eps: float, what) -> None:
    a, b = float(a), float(b)
    assert np.isfinite(a) and np.isfinite(b), (what, a, b)
    assert abs(a - b) <= ULPS * eps * max(abs(a), abs(b)), (what, a, b)


def _same_states(c: gg.Case, a: dict, b: dict, step: int) -> None:
    """Every field the edge-mapped graph holds is the twin's, within rounding."""
    for name in a:
        for field, x in a[name].items():
            y = b[name][field]
            assert x.dtype == y.dtype and x.shape == y.shape, (name, field)
            scale = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
            assert float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) <= (
                8 * float(np.finfo(x.dtype).eps) * scale), (c.label, step, name, field)


def compare(c: gg.Case, edge, twin, *, relay: bool, steps: int = 3) -> None:
    """The reports of *edge* and *twin* after each of *steps* steps."""
    eps = float(np.finfo(np.dtype(c.dtype)).eps)
    edge.reset_state()
    twin.reset_state()
    for step in range(1, steps + 1):
        edge.step()
        twin.step()
        _same_states(c, gg.snapshot(edge), gg.snapshot(twin), step)
        ra, rb = _report(edge), _report(twin)
        where = (c.label, step)
        assert "not_usable_reason" not in ra and "not_usable_reason" not in rb, where
        assert int(ra["iterations"]) == int(rb["iterations"]), (where, ra, rb)
        for flag in FLAGS:
            assert bool(ra[flag]) is bool(rb[flag]), (where, flag, ra, rb)
        assert bool(ra["spectral_usable"]), ("premise: a usable report", where, ra)
        if relay:
            for name in ("residual", "amplification", "error_estimate", "rho_spectral"):
                _close(ra[name], rb[name], eps, (where, name))
            fa = float(ra["spectral_error_bound"]) / (float(ra["residual"]) + ra["floor"])
            fb = float(rb["spectral_error_bound"]) / (float(rb["residual"]) + rb["floor"])
            assert abs(fa - fb) <= RELAY_BOUND_GAP * max(fa, fb), (where, fa, fb)
            continue
        for name in NUMBERS + ("floor",):
            _close(ra[name], rb[name], eps, (where, name))


def assert_reports_as_its_transform_twin(c: gg.Case) -> None:
    with gg.x64(c.needs_x64):
        compare(c, _edge(c), gg.build(gg.transform_twin(gg.static_twin(c))), relay=False)


def assert_reports_as_its_relay_twin(c: gg.Case) -> None:
    with gg.x64(c.needs_x64):
        compare(c, _edge(c), gg.build(gg.relay_twin(gg.static_twin(c))), relay=True)


@pytest.mark.parametrize("c", STATIC_PER_PUSH, ids=repr)
def test_a_static_mapped_edge_under_the_interface_norm_reports_a_usable_spectrum(c):
    """The premise of the comparisons, and the compile of the edge-mapped
    graph they share (a test's seconds here are a compile each)."""
    with gg.x64(c.needs_x64):
        edge = _edge(c)
        edge.reset_state()
        edge.step()
        report = _report(edge)
    assert bool(report["spectral_usable"]) and not bool(report["converged"]), report
    assert float(report["residual"]) > 2.0 ** 10 * report["floor"], (
        "premise: a residual far above its float floor", report)


@pytest.mark.parametrize("c", STATIC_PER_PUSH, ids=repr)
def test_a_static_mapped_edge_under_the_interface_norm_reports_as_its_transform_twin(c):
    """Per push; slow sibling
    :func:`test_every_static_mapped_edge_reports_as_its_transform_twin`."""
    assert_reports_as_its_transform_twin(c)


# Slow: two graphs with diagnostics compiled per case.
# Per push: tests/property/test_interface_reading_twins.py::test_a_static_mapped_edge_under_the_interface_norm_reports_as_its_transform_twin
@pytest.mark.slow
@pytest.mark.parametrize("c", STATIC_SLOW, ids=repr)
def test_every_static_mapped_edge_reports_as_its_transform_twin(c):
    assert_reports_as_its_transform_twin(c)


# Slow: two graphs with diagnostics compiled per case.
# Per push: tests/property/test_interface_reading_twins.py::test_a_static_mapped_edge_under_the_interface_norm_reports_as_its_transform_twin
@pytest.mark.slow
@pytest.mark.parametrize("c", RELAY_SLOW, ids=repr)
def test_a_static_mapped_edge_under_the_interface_norm_reports_as_its_relay_twin(c):
    assert_reports_as_its_relay_twin(c)


def test_the_relay_twin_delivers_the_mapped_value_over_plain_edges_only():
    """The twin's structure: the same nodes and groups, no mapping on any
    edge, and each formerly mapped edge reading the relay's field, whose
    initial value is the mapping of the source field beside it."""
    c = STATIC_PER_PUSH[0]
    with gg.x64(c.needs_x64):
        static, twin = gg.static_twin(c), gg.relay_twin(gg.static_twin(c))
        assert [nd.name for nd in twin.nodes] == [nd.name for nd in static.nodes]
        assert twin.groups == static.groups and len(twin.edges) == len(static.edges)
        for e, t in zip(static.edges, twin.edges):
            assert t.mapping is None and t.geometry is None, t
            assert (t.src, t.dst, t.tf, t.transform, t.additive) == (
                e.src, e.dst, e.tf, e.transform, e.additive)
            assert (t.sf == e.sf) is (e.mapping is None), (e, t)
            if e.mapping is not None:
                state = twin.node(e.src).initial_state()
                np.testing.assert_array_equal(
                    np.asarray(state[t.sf]), np.asarray(e.mapping.apply(state[e.sf], None)))
        assert all(e.mapping is None and e.transform is not None
                   for e in gg.transform_twin(gg.static_twin(c)).edges)


def test_a_target_anchored_geometry_has_no_relay_twin():
    """The relay would read the target's geometry over an internal edge of
    its own, which the interface norm would read: refused, not built."""
    c = gg.case("target anchors", adv=0.3, down="target", up="target")
    with pytest.raises(NotImplementedError, match="target-anchored geometry has no relay twin"):
        gg.relay_twin(gg.two_body(c))


# ---------------------------------------------------------------------------
# The relay twin of a moving geometry, on the solve path
# ---------------------------------------------------------------------------

#: Source-anchored geometry edges whose geometry moves with the input
#: (inside a group, with the iterate), under the norms 0.4.0 supports.
MOVING = [
    gg.case("plain step", adv=0.3, **_SOURCES),
    # Two passes, not a converged solve: at a fixed point the iterate and
    # the pass agree, and a geometry read from the wrong one of them is the
    # same number.
    gg.case("group, Gauss-Seidel, two passes", adv=0.3, **_SOURCES,
            group=dict(max_iterations=2)),
    pytest.param(gg.case("group, Gauss-Seidel, converged", adv=0.3, **_SOURCES,
                         group=dict(max_iterations=200, tolerance=1e-5)),
                 marks=pytest.mark.slow),
    pytest.param(gg.case("plain step, multilinear", kind="multilinear", adv=0.3, **_SOURCES),
                 marks=pytest.mark.slow),
    pytest.param(gg.case("group, Jacobi, three passes", adv=0.3, **_SOURCES,
                         group=dict(max_iterations=3, iteration_mode="jacobi")),
                 marks=pytest.mark.slow),
    pytest.param(gg.case("group, multilinear, float64", kind="multilinear", dtype="float64",
                         adv=0.3, **_SOURCES, group=dict(max_iterations=3)),
                 marks=pytest.mark.slow),
    pytest.param(gg.case("group, P before F", order=("P", "F"), adv=0.3, **_SOURCES,
                         group=dict(max_iterations=3)), marks=pytest.mark.slow),
]


# Slow (the marked cases): two graphs compiled per case.
# Per push: tests/property/test_interface_reading_twins.py::test_the_relay_twin_of_a_moving_source_anchored_geometry_steps_as_the_edge_mapped_graph
@pytest.mark.parametrize("c", MOVING, ids=repr)
def test_the_relay_twin_of_a_moving_source_anchored_geometry_steps_as_the_edge_mapped_graph(c):
    """The relay reads the geometry field beside the value, both as its
    update has just produced them -- the time level a source-anchored
    geometry edge reads ("from the dict the value is read from").  Every
    field the edge-mapped graph holds, after every step, within rounding
    (zero was measured), with the geometry moved and the pass count the
    same."""
    with gg.x64(c.needs_x64):
        edge, twin = gg.build(gg.two_body(c)), gg.build(gg.relay_twin(gg.two_body(c)))
        before = gg.snapshot(edge)
        for step in range(1, c.steps + 1):
            edge.step()
            twin.step()
            _same_states(c, gg.snapshot(edge), gg.snapshot(twin), step)
            if c.group is not None:
                assert int(edge.coupling_diagnostics()[KEY]["iterations"]) == int(
                    twin.coupling_diagnostics()[KEY]["iterations"]), (c.label, step)
        after = gg.snapshot(edge)
    field = "pos" if c.kind == "multilinear" else "A"
    assert np.any(after["F"][field] != before["F"][field]), "premise: the geometry moved"


# ---------------------------------------------------------------------------
# What waits for diagnostics that read a geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", gg.RELAY_INTERFACE_CASES, ids=repr)
def test_the_interface_norm_over_a_geometry_edge_is_refused_and_its_relay_twin_builds(c):
    """PHASE 1 (see ``geometry_graphs``), per push: the interface norm over
    a geometry edge is refused at compile, and the relay twin -- plain
    edges only -- is accepted.  Slow sibling:
    :func:`test_the_interface_norm_over_a_geometry_edge_reports_as_its_relay_twin`."""
    with gg.x64(c.needs_x64):
        keys = [e.key for e in gg.build(gg.two_body(c), compile=False).edges
                if e.geometry is not None]
        assert len(keys) == 2, keys
        if gg.DIAGNOSTICS_READ_GEOMETRY:
            gg.build(gg.two_body(c))        # accepted, once diagnostics read a geometry
        else:
            gg.assert_interface_norm_refused(lambda: gg.build(gg.two_body(c)), keys)
        twin = gg.build(gg.relay_twin(gg.two_body(c)))
    assert all(e.geometry is None and e.mapping is None for e in twin.edges)


# Slow: a relay twin with diagnostics compiled per case (and, once
# diagnostics read a geometry, the edge-mapped graph beside it).
# Per push: tests/property/test_interface_reading_twins.py::test_the_interface_norm_over_a_geometry_edge_is_refused_and_its_relay_twin_builds
@pytest.mark.slow
@pytest.mark.parametrize("c", gg.RELAY_INTERFACE_CASES, ids=repr)
def test_the_interface_norm_over_a_geometry_edge_reports_as_its_relay_twin(c):
    """PHASE 1 (see ``geometry_graphs``): the relay twin of a refused case
    reports a usable spectrum today -- the report the edge-mapped graph is
    held to once ``DIAGNOSTICS_READ_GEOMETRY`` is set, by the comparison of
    the static cases."""
    with gg.x64(c.needs_x64):
        twin = gg.build(gg.relay_twin(gg.two_body(c)))
        if not gg.DIAGNOSTICS_READ_GEOMETRY:
            twin.step()
            report = twin.coupling_diagnostics()[KEY]
            assert bool(report["spectral_usable"]) and np.isfinite(
                float(report["spectral_error_bound"])), report
            return
        compare(c, gg.build(gg.two_body(c)), twin, relay=True)
