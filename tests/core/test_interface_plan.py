"""A coupling group's edges are described once, and every reader of them
reads a view of that description (``core/coupling/_interface_plan.py``).

The first half pins the description itself: one record per edge, on every
shape of edge a group can hold.  The second half pins each place where
two views of it *differ*.  Those differences are today's behaviour, kept
on purpose and reconciled one at a time elsewhere; a view that quietly
started returning what its neighbour returns would fail here.

The side the interface norm reads an edge on is part of the description:
the compact side, decided in one function from the sizes the edge's
mapping declares.
"""

from __future__ import annotations

import dataclasses
import types

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout as layout
from maddening.core.coupling import _interface_plan as ip
from maddening.core.coupling.acceleration import (
    _interface_readings,
    coupling_residual_interface,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping, sparse_matrix_mapping
from maddening.core.edge import EdgeSpec, _delivered
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F32 = jnp.float32


class _Plain:
    """A node that defines no fluxes."""

    compute_boundary_fluxes = SimulationNode.compute_boundary_fluxes


class _Fluxy:
    """A node that defines ``compute_boundary_fluxes``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": state["x"]}


class _Geometry:
    """Stands for a geometry-dependent mapping: only what the plan asks of one."""

    kind = "multilinear_grid"
    needs_geometry = True


class _Custom:
    """Stands for a registered kind's own static mapping class: what the
    plan asks of one is its kind and the two sizes it declares."""

    kind = "custom"

    def __init__(self, n_source=2, n_target=2):
        self.n_source, self.n_target = n_source, n_target


class _Shaped(_Custom):
    """A static mapping that declares its fields' leading axes."""

    def __init__(self, source_lead, target_lead):
        super().__init__(int(np.prod(source_lead)), int(np.prod(target_lead)))
        self._leads = (tuple(source_lead), tuple(target_lead))

    def field_shapes(self):
        return self._leads


def _double(v):
    return 2.0 * v


DENSE = matrix_mapping(np.eye(2, dtype=np.float32))
#: A dense mapping onto more entries than it reads (2 -> 3).
WIDE_DENSE = matrix_mapping(np.ones((3, 2), np.float32))
SPARSE = sparse_matrix_mapping(np.array([[0], [1]]), np.ones((2, 1), np.float32), n_source=2)
GEOM = _Geometry()

STATE = {
    "a": {"x": jnp.ones(2, F32), "n": jnp.int32(3), "g": jnp.zeros((2, 1), F32)},
    "b": {"x": jnp.ones(2, F32), "y": jnp.ones(2, F32)},
    "c": {"x": jnp.ones(2, F32), "z": jnp.zeros((0,), F32)},
    "o": {"x": jnp.ones(2, F32)},
    "p": {"x": jnp.ones(2, F32)},
}
NODES = {name: types.SimpleNamespace(node=node) for name, node in {
    "a": _Fluxy(), "b": _Plain(), "c": _Plain(), "o": _Plain(), "p": _Fluxy()}.items()}
MEMBERS = frozenset("abc")
#: The sweep: neither the order of the names nor that of the edges below.
ORDER = ("c", "a", "b")

#: One edge of every shape, in the order they are declared.
EDGES = (
    EdgeSpec("b", "a", "x", "u0"),                                        # a plain field edge
    EdgeSpec("a", "b", "x", "u0", mapping=DENSE),                         # a static dense mapping
    EdgeSpec("a", "c", "x", "u0", mapping=SPARSE),                        # a static sparse mapping
    EdgeSpec("a", "b", "x", "u1", transform=_double),                     # a transform
    EdgeSpec("b", "c", "y", "u1", mapping=DENSE, transform=_double),      # a mapping, then a transform
    EdgeSpec("a", "b", "q", "u2"),                                        # a flux
    EdgeSpec("a", "c", "n", "u2"),                                        # a non-floating source
    EdgeSpec("c", "a", "x", "u1"),                                        # a field read whole ...
    EdgeSpec("c", "b", "x", "u3", mapping=_Custom()),                     # ... and through a mapping
    EdgeSpec("a", "b", "x", "u4", mapping=GEOM, geometry=("source", "g")),
    EdgeSpec("b", "a", "x", "u2", mapping=GEOM, geometry=("target", "g")),
    EdgeSpec("c", "b", "z", "u5"),                                        # an edge delivering no entries
    EdgeSpec("c", "b", "gone", "u6"),                                     # neither a field nor a flux
    EdgeSpec("o", "a", "x", "u3"),                                        # inbound
    EdgeSpec("p", "b", "q", "u7"),                                        # an inbound flux
    EdgeSpec("a", "o", "q", "u0"),                                        # an outbound flux
    EdgeSpec("o", "a", "x", "u4", mapping=GEOM, geometry=("target", "g")),
    EdgeSpec("o", "p", "x", "u0"),                                        # not the group's
)


def _plan(edges=EDGES, order=ORDER, members=MEMBERS, state=None, nodes=None):
    return ip.interface_plan(members, edges, order, STATE if state is None else state,
                             NODES if nodes is None else nodes)


def _keys(records):
    return [r.key for r in records]


# ---------------------------------------------------------------------------
# The description
# ---------------------------------------------------------------------------
def test_each_edge_with_an_end_in_the_group_has_one_record_of_what_it_is():
    """Role, source kind, mapping form, anchor, transform and side, edge by edge."""
    plan = _plan()
    table = {r.key: (r.role, r.source_kind, r.mapping_form, r.anchor, r.has_transform, r.norm_side)
             for r in (*plan.internal, *plan.crossing)}
    assert table == {
        "b.x->a.u0": ("internal", "floating", "none", None, False, "delivered"),
        "a.x->b.u0": ("internal", "floating", "static_dense", None, False, "delivered"),
        "a.x->c.u0": ("internal", "floating", "static_sparse", None, False, "delivered"),
        "a.x->b.u1": ("internal", "floating", "none", None, True, "delivered"),
        "b.y->c.u1": ("internal", "floating", "static_dense", None, True, "delivered"),
        "a.q->b.u2": ("internal", "flux", "none", None, False, "delivered"),
        "a.n->c.u2": ("internal", "non_floating", "none", None, False, "delivered"),
        "c.x->a.u1": ("internal", "floating", "none", None, False, "delivered"),
        "c.x->b.u3": ("internal", "floating", "static", None, False, "delivered"),
        "a.x->b.u4": ("internal", "floating", "needs_geometry", ("source", "g"), False, "delivered"),
        "b.x->a.u2": ("internal", "floating", "needs_geometry", ("target", "g"), False, "delivered"),
        "c.z->b.u5": ("internal", "floating", "none", None, False, "delivered"),
        "c.gone->b.u6": ("internal", "absent", "none", None, False, "delivered"),
        "o.x->a.u3": ("inbound", "floating", "none", None, False, "delivered"),
        "p.q->b.u7": ("inbound", "flux", "none", None, False, "delivered"),
        "a.q->o.u0": ("outbound", "flux", "none", None, False, "delivered"),
        "o.x->a.u4": ("inbound", "floating", "needs_geometry", ("target", "g"), False, "delivered"),
    }
    by_key = {r.key: r for r in (*plan.internal, *plan.crossing)}
    for declared, edge in enumerate(EDGES[:-1]):
        record = by_key[edge.key]
        assert record.edge is edge and record.declared == declared
        assert record.source == (edge.source_node, edge.source_field)
        assert record.target == (edge.target_node, edge.target_field)
        assert record.mapping is edge.mapping
    assert by_key["a.x->b.u4"].mapping_kind == "multilinear_grid"
    assert by_key["b.x->a.u0"].mapping_kind is None
    assert plan.members == MEMBERS and plan.order == ORDER
    assert plan.flux_members == frozenset("a"), "p defines fluxes too, and is no member"
    assert plan.member_fields == {"a": ("x", "n", "g"), "b": ("x", "y"), "c": ("x", "z")}


def test_the_internal_edges_are_held_in_the_order_the_norm_sums_them():
    """By the source's place in the sweep, then the source field, the target's
    place, the target field and the ordinal: not as declared."""
    plan = _plan()
    assert _keys(plan.internal) == [
        "c.gone->b.u6", "c.x->a.u1", "c.x->b.u3", "c.z->b.u5",
        "a.n->c.u2", "a.q->b.u2", "a.x->c.u0", "a.x->b.u0", "a.x->b.u1", "a.x->b.u4",
        "b.x->a.u0", "b.x->a.u2", "b.y->c.u1",
    ]
    assert all(plan.is_internal(r.edge) for r in plan.internal)
    assert not any(plan.is_internal(r.edge) for r in plan.crossing)
    assert not plan.is_internal(EDGES[-1])


def test_the_side_the_norm_reads_is_one_field_that_the_reading_branches_on(monkeypatch):
    """Delivered, for every edge of the table (its mappings are ties); and
    the delivered reading is the step's edge rule."""
    plan = _plan()
    assert {r.norm_side for r in (*plan.internal, *plan.crossing)} == {ip.DELIVERED}
    calls = []

    def spy(edge, value, mappings=None, geom=None):
        calls.append(edge.key)
        return _delivered(edge, value, mappings, geom)

    monkeypatch.setattr(ip, "_delivered", spy)
    by_key = {r.key: r for r in plan.internal}
    x = jnp.asarray([1.0, 3.0], F32)
    np.testing.assert_array_equal(by_key["b.y->c.u1"].reading(x), 2.0 * x)
    np.testing.assert_array_equal(
        by_key["a.x->b.u0"].reading(x, {"a.x->b.u0": DENSE.params_pytree()}), x)
    assert calls == ["b.y->c.u1", "a.x->b.u0"], "the patch of the module that reads it took effect"
    elsewhere = dataclasses.replace(by_key["b.x->a.u0"], norm_side="somewhere")
    with pytest.raises(ValueError, match="no reading on side 'somewhere'"):
        elsewhere.reading(x)
    assert not elsewhere.reads_source_as_is
    as_is = {r.key for r in plan.internal if r.reads_source_as_is}
    assert as_is == {"b.x->a.u0", "a.q->b.u2", "a.n->c.u2", "c.x->a.u1", "c.z->b.u5",
                     "c.gone->b.u6"}, "no mapping and no transform"
    # The other side: the source value itself, through neither the mapping
    # nor the transform, and with no call of the edge rule.
    expanding = ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=WIDE_DENSE,
                                         transform=_double), STATE)
    assert expanding.norm_side == ip.SOURCE
    del calls[:]
    assert expanding.reading(x, {"a.x->b.u0": WIDE_DENSE.params_pytree()}) is x
    assert calls == []
    assert expanding.reads_source_as_is and not expanding.reads_through_mapping
    assert by_key["a.x->b.u0"].reads_through_mapping, "a tie is read through its mapping"
    assert not by_key["b.x->a.u0"].reads_through_mapping, "no mapping to read through"


def _sparse(n_source, n_target, layout):
    """A sparse mapping of those sizes with one weight per row of its layout."""
    if layout == "gather":
        return sparse_matrix_mapping(np.zeros((n_target, 1), np.int32),
                                     np.ones((n_target, 1), np.float32), n_source=n_source)
    return StaticSparseMapping(np.zeros((n_source, 1), np.int32),
                               jnp.ones((n_source, 1), F32), n_source=n_source,
                               n_target=n_target, counts=np.ones(n_source, np.int64),
                               layout="scatter")


#: ``(form, how to build a mapping of (n_source, n_target))``: the dense
#: kind, the sparse kind in either layout (its weights are ``(n_target, k)``
#: in one and ``(n_source, k)`` in the other), a registered kind's own
#: class, and one that declares its fields' leading axes.
SIDE_FORMS = {
    "dense": lambda n, m: matrix_mapping(np.ones((m, n), np.float32)),
    "sparse-gather": lambda n, m: _sparse(n, m, "gather"),
    "sparse-scatter": lambda n, m: _sparse(n, m, "scatter"),
    "registered": lambda n, m: _Custom(n, m),
    "shaped": lambda n, m: _Shaped((n,), (m,)),
}


@pytest.mark.parametrize("form", sorted(SIDE_FORMS))
@pytest.mark.parametrize("n_source,n_target,side", [
    (2, 3, "source"), (2, 600, "source"), (3, 2, "delivered"), (600, 2, "delivered"),
    (2, 2, "delivered"), (1, 1, "delivered")])
@pytest.mark.parametrize("transform", [None, _double])
def test_a_static_mapping_onto_more_entries_is_read_at_its_source_and_a_tie_as_delivered(
        form, n_source, n_target, side, transform):
    """The rule, on every static form: more entries delivered than the field
    holds is ``"source"``; fewer, **and a tie**, ``"delivered"``.  Decided by
    the sizes the mapping declares, whatever its weights' shape and whatever
    else the edge carries."""
    mapping = SIDE_FORMS[form](n_source, n_target)
    edge = EdgeSpec("a", "b", "x", "u0", mapping=mapping, transform=transform)
    assert ip._norm_side(edge) == side
    record = ip._edge_record(edge)
    assert record.norm_side == side
    assert record.reads_source_as_is == (side == "source")
    assert record.reads_through_mapping == (side == "delivered")
    assert record.has_transform == (transform is not None)
    if side == "source":
        value = jnp.arange(n_source, dtype=F32)
        assert record.reading(value) is value, "before the mapping and the transform"


def test_the_side_counts_entries_where_a_mapping_declares_its_fields_leading_axes():
    """``field_shapes()`` replaces the two sizes: the entries are the
    products of the leading axes."""
    for source_lead, target_lead, side in (((2, 3), (7,), "source"), ((2, 3), (6,), "delivered"),
                                           ((7,), (2, 3), "delivered"), ((5,), (2, 3), "source")):
        edge = EdgeSpec("a", "b", "x", "u0", mapping=_Shaped(source_lead, target_lead))
        assert ip._norm_side(edge) == side, (source_lead, target_lead)
    record = ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=_Shaped((2, 3), (7,))))
    grid = jnp.ones((2, 3, 4), F32)
    assert record.reading(grid) is grid, "further axes pass through"


def test_the_side_is_the_edges_own_wherever_it_is_described():
    """One function of the edge: a plan built from any state or order, a
    plan built without a state, and a bare edge described for the report
    all read each edge on the same side."""
    edges = (EdgeSpec("a", "b", "x", "u0", mapping=WIDE_DENSE),
             EdgeSpec("b", "a", "y", "u0", mapping=_sparse(2, 3, "gather")),
             EdgeSpec("b", "a", "x", "u1", mapping=matrix_mapping(np.ones((1, 2), np.float32))),
             EdgeSpec("a", "b", "x", "u1", mapping=DENSE),
             EdgeSpec("b", "a", "x", "u2"))
    want = ["source", "source", "delivered", "delivered", "delivered"]
    sides = lambda records: [r.norm_side for r in sorted(records, key=lambda r: edges.index(r.edge))]  # noqa: E731
    wide = {"a": {"x": np.ones(2, np.float64)}, "b": {"x": jnp.ones(2, F32), "y": jnp.ones(2, F32)}}
    for order in (("a", "b"), ("b", "a")):
        for state in (STATE, wide, None):
            plan = ip.interface_plan(frozenset("ab"), edges, order, state, NODES)
            assert sides(plan.internal) == want, (order, state is None)
    assert sides(ip.interface_records(list(edges), STATE)) == want
    assert sides(ip.interface_records(list(edges))) == want


def test_a_geometry_mapping_is_left_on_the_delivered_side():
    """A geometry-dependent kind that declares no length scale has no
    reading at its source (a position is measured in the kind's own
    length), so it is left on the delivered side whatever its sizes, and
    ``compile()`` refuses it inside a group under the interface norm."""
    assert not hasattr(GEOM, "geometry_length_scale")
    for anchor in ("source", "target"):
        record = ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=GEOM,
                                          geometry=(anchor, "g")))
        assert (record.mapping_form, record.norm_side) == ("needs_geometry", "delivered")
    plan = _pair(EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("source", "g")),
                 EdgeSpec("b", "a", "x", "u0"))
    assert _keys(plan.geometry_edges()) == ["a.x->b.u0"]
    group = types.SimpleNamespace(convergence_norm="interface", nodes=frozenset("ab"),
                                  subcycling=False)
    assert layout._geometry_edge_coupling_errors(group, {}, plan) == [], (
        "the kind's name is what compile() asks; this stand-in bears the accepted one")
    other = types.SimpleNamespace(kind="some_other_kind", needs_geometry=True, n_source=2,
                                  n_target=2)
    plan = _pair(EdgeSpec("a", "b", "x", "u0", mapping=other, geometry=("source", "g")),
                 EdgeSpec("b", "a", "x", "u0"))
    (error,) = layout._geometry_edge_coupling_errors(group, {}, plan)
    assert error.startswith("ERROR:") and "only for the 'multilinear_grid' kind" in error
    assert "'some_other_kind'" in error and "geometry source.g" in error


def test_a_static_mapping_that_declares_no_sizes_cannot_be_placed():
    """``add_edge`` refuses such a mapping; a bare edge that carries one is
    refused here rather than read on a side nobody chose."""
    class _Sizeless:
        kind = "custom"

    with pytest.raises(TypeError, match="cannot tell which side of it is the compact one"):
        ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=_Sizeless()))


def test_a_source_side_reading_refuses_a_field_of_another_size_than_the_mapping_declares():
    """The side was decided on the declared sizes: a field that does not have
    them is refused, not read on a side its own size would not have chosen."""
    record = ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=WIDE_DENSE))
    with pytest.raises(ValueError, match=r"leading axes \(2,\).*has shape \(5,\)"):
        record.reading(jnp.ones(5, F32))
    np.testing.assert_array_equal(record.reading(jnp.ones((2, 4), F32)), jnp.ones((2, 4), F32))
    # A scalar stands for a field of one entry, as ``add_edge`` reads it.
    one = ip._edge_record(EdgeSpec("a", "b", "x", "u0",
                                   mapping=matrix_mapping(np.ones((3, 1), np.float32))))
    assert one.norm_side == ip.SOURCE and float(one.reading(jnp.float32(2.0))) == 2.0


def test_a_bare_sequence_of_edges_is_read_in_the_order_given_from_the_state_handed():
    """What a direct call of the norm reads: no group, so no canonical order
    and no flux hook to ask."""
    edges = [EDGES[3], EDGES[0], EDGES[5], EDGES[6]]
    records = ip.interface_records(edges, STATE)
    assert [r.edge for r in records] == edges
    assert [r.source_kind for r in records] == ["floating", "floating", "absent", "non_floating"]
    plan = _plan()
    assert ip.interface_records(plan) is plan.internal


def test_the_three_derivations_the_later_stages_need_are_one_line_each():
    """Nothing here consumes them yet, so they are not views: each is a line
    over the records."""
    plan = _plan()
    measured_whole = {r.source for r in plan.internal
                      if r.source_kind == ip.FLOATING and r.reads_source_as_is}
    assert measured_whole == {("b", "x"), ("c", "x"), ("c", "z")}
    floating = {(nn, f) for nn in plan.members for f, v in STATE[nn].items()
                if jnp.issubdtype(v.dtype, jnp.floating)}
    unread = floating - plan.source_fields()
    assert unread == {("a", "g")}
    # (ii) with "reads a geometry" standing in for the geometry stage's
    # rule; the static rule is the record's own side.
    compact = {r.key: ((r.source, r.anchor) if r.anchor is not None else r.key)
               for r in plan.internal if r.source_kind == ip.FLOATING}
    assert compact["a.x->b.u4"] == (("a", "x"), ("source", "g"))
    assert compact["b.x->a.u0"] == "b.x->a.u0"
    # (i) under the compact rule: a field an expanding mapping reads is
    # measured whole, like one a plain edge reads.
    pair = _pair(EdgeSpec("a", "b", "x", "u0", mapping=WIDE_DENSE),
                 EdgeSpec("b", "a", "x", "u0", mapping=DENSE), EdgeSpec("b", "a", "y", "u1"))
    assert {r.source for r in pair.internal
            if r.source_kind == ip.FLOATING and r.reads_source_as_is} == {("a", "x"), ("b", "y")}


# ---------------------------------------------------------------------------
# Where two views differ
# ---------------------------------------------------------------------------
def _pair(*edges, state=None, nodes=None, order=("a", "b")):
    return _plan(edges, order=order, members=frozenset("ab"), state=state, nodes=nodes)


def _iqn_group(**kwargs):
    return types.SimpleNamespace(acceleration="iqn-ils", accelerated_fields=None,
                                 nodes=frozenset("ab"), **kwargs)


def test_the_accelerator_counts_a_source_anchored_geometry_and_the_spectrum_does_not():
    """IQN's default set holds the geometry an edge reads from its producer;
    the spectrum's weights hold the edges' source fields alone."""
    back = EdgeSpec("b", "a", "x", "u0")
    source = _pair(EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("source", "g")), back)
    assert source.iqn_fields() == {"a": ("g", "x"), "b": ("x",)}
    assert source.source_fields() == {("a", "x"), ("b", "x")}
    assert layout._group_accel_fields(_iqn_group(), source, STATE) == {"a": ("g", "x"), "b": ("x",)}
    # A geometry read from the *target* is the target's own pre-step state.
    target = _pair(EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("target", "y")), back)
    assert target.iqn_fields() == {"a": ("x",), "b": ("x",)}


def test_a_flux_edge_is_its_producers_whole_state_to_the_accelerator_only():
    """IQN takes every field of the producer; to the spectrum's weights the
    flux's name is a pair no state field matches."""
    plan = _pair(EdgeSpec("a", "b", "q", "u0"), EdgeSpec("b", "a", "x", "u0"))
    assert plan.iqn_fields() == {"a": ("g", "n", "x"), "b": ("x",)}
    assert plan.source_fields() == {("a", "q"), ("b", "x")}
    assert "q" not in plan.member_fields["a"]
    # Only floating fields are flattened, whatever the default set names.
    assert layout._group_accel_fields(_iqn_group(), plan, STATE) == {"a": ("g", "x"), "b": ("x",)}
    assert _keys(plan.flux_edges()) == ["a.q->b.u0"]
    assert not plan.norm_reads_mapping_weights()
    mapped_flux = _pair(EdgeSpec("a", "b", "q", "u0", mapping=DENSE), EdgeSpec("b", "a", "x", "u0"))
    assert not mapped_flux.norm_reads_mapping_weights(), "a flux is not read from the state"
    assert mapped_flux.mapped_keys() == ("a.q->b.u0",)


def test_a_field_two_edges_read_is_two_readings_and_one_field():
    """The norm sums over edges; the accelerator and the spectrum's weights
    count fields."""
    twice = (EdgeSpec("a", "b", "x", "u0"), EdgeSpec("a", "b", "x", "u1"),
             EdgeSpec("b", "a", "x", "u0"))
    plan = _pair(*twice)
    assert [r.source for r in ip.interface_records(plan)] == [("a", "x"), ("a", "x"), ("b", "x")]
    assert plan.source_fields() == {("a", "x"), ("b", "x")}
    assert plan.iqn_fields() == {"a": ("x",), "b": ("x",)}
    floats = {"a": ("g", "x"), "b": ("x", "y")}
    assert not layout._reading_is_the_fields(plan, floats)
    once = _pair(twice[0], twice[2])
    assert layout._reading_is_the_fields(once, floats)
    # The residual counts the field read twice, twice: a.x moves by ``d`` of
    # itself and b.x not at all.
    rtol, d = 1e-3, 1e-2
    old = {"a": {"x": jnp.ones(2, F32)}, "b": {"x": jnp.ones(2, F32)}}
    new = {"a": {"x": jnp.full(2, 1.0 + d, F32)}, "b": {"x": jnp.ones(2, F32)}}
    r_twice = float(coupling_residual_interface(new, old, plan, 0.0, rtol))
    r_once = float(coupling_residual_interface(new, old, once, 0.0, rtol))
    assert r_twice == pytest.approx(d / (1.0 + d) / rtol * np.sqrt(2.0 / 3.0), rel=1e-4)
    assert r_once == pytest.approx(d / (1.0 + d) / rtol * np.sqrt(1.0 / 2.0), rel=1e-4)
    assert len(list(_interface_readings(plan, new))) == 3


def test_the_norm_decides_what_is_floating_on_the_state_it_is_handed():
    """A record's kind is that of the state its plan was built from; the
    reading asks the state in front of it."""
    edge = EdgeSpec("a", "b", "n", "u0")
    plan = _pair(edge, EdgeSpec("b", "a", "x", "u0"))
    assert [r.source_kind for r in plan.internal] == ["non_floating", "floating"]
    assert [e.key for e, *_ in _interface_readings(plan, STATE)] == ["b.x->a.u0"]
    floated = {**STATE, "a": {**STATE["a"], "n": jnp.float32(3.0)}}
    assert [e.key for e, *_ in _interface_readings(plan, floated)] == ["a.n->b.u0", "b.x->a.u0"]
    built_floating = _pair(edge, EdgeSpec("b", "a", "x", "u0"), state=floated)
    assert [r.source_kind for r in built_floating.internal] == ["floating", "floating"]
    assert [e.key for e, *_ in _interface_readings(built_floating, STATE)] == ["b.x->a.u0"]


def test_an_edge_that_delivers_no_entries_is_a_field_and_not_a_reading():
    """The norm skips what delivers nothing; the field is still named to the
    accelerator and to the spectrum's weights."""
    plan = _plan((EdgeSpec("c", "b", "z", "u0"), EdgeSpec("b", "c", "x", "u0")),
                 order=("b", "c"), members=frozenset("bc"))
    assert [e.key for e, *_ in _interface_readings(plan, STATE)] == ["b.x->c.u0"]
    assert plan.source_fields() == {("c", "z"), ("b", "x")}
    assert plan.iqn_fields() == {"b": ("x",), "c": ("z",)}


def test_only_an_edge_read_through_its_mapping_makes_the_floor_depend_on_weights():
    """A floating source, a mapping, and the delivered side: an edge the
    norm reads at its source is the stored field, whatever the weights."""
    back = EdgeSpec("b", "a", "x", "u0")
    interface = types.SimpleNamespace(convergence_norm="interface", atol=0.0)
    mixed = types.SimpleNamespace(convergence_norm="mixed", atol=0.0)
    mapped = _pair(EdgeSpec("a", "b", "x", "u0", mapping=DENSE), back)
    assert mapped.norm_reads_mapping_weights()
    assert layout._reads_mapping_weights(interface, mapped)
    assert not layout._reads_mapping_weights(mixed, mapped)
    assert not _pair(EdgeSpec("a", "b", "x", "u0"), back).norm_reads_mapping_weights()
    assert not _pair(EdgeSpec("a", "b", "n", "u0", mapping=DENSE), back).norm_reads_mapping_weights()
    # An inbound mapped edge is not the group's interface.
    inbound = _pair(EdgeSpec("o", "b", "x", "u0", mapping=DENSE), back)
    assert not inbound.norm_reads_mapping_weights() and inbound.mapped_keys() == ()
    # Read at its source: no weights in the reading, no slot, no key.
    scatter = EdgeSpec("a", "b", "x", "u0", mapping=WIDE_DENSE)
    expanding = _pair(scatter, back)
    assert not expanding.norm_reads_mapping_weights() and expanding.mapped_keys() == ()
    assert not layout._reads_mapping_weights(interface, expanding)
    assert layout._reading_is_the_fields(expanding, {"a": ("x",), "b": ("x",)})
    # With a dead band declared, the band of an edge read at its source is
    # asked of what the edge delivers too, through the weights the step ran
    # with: the group then owns the slot.  Only under the interface norm,
    # only for an edge read at its source, and never at the default atol.
    banded = types.SimpleNamespace(convergence_norm="interface", atol=1e-6)
    assert expanding.band_reads_beyond_the_state()
    assert layout._reads_mapping_weights(banded, expanding)
    assert not layout._reads_mapping_weights(
        types.SimpleNamespace(convergence_norm="mixed", atol=1e-6), expanding)
    assert not _pair(EdgeSpec("a", "b", "x", "u0"), back).band_reads_beyond_the_state()
    assert not mapped.band_reads_beyond_the_state()
    assert not _pair(EdgeSpec("a", "b", "n", "u0", mapping=WIDE_DENSE),
                     back).band_reads_beyond_the_state()
    # One edge of each side: the group reads the weights of the gather alone.
    gather = EdgeSpec("b", "a", "x", "u0", mapping=matrix_mapping(np.ones((1, 2), np.float32)))
    two_way = _pair(scatter, gather)
    assert two_way.norm_reads_mapping_weights() and two_way.mapped_keys() == ("b.x->a.u0",)
    assert not layout._reading_is_the_fields(two_way, {"a": ("x",), "b": ("x",)})
    # A field read at its source by two edges is still read twice.
    twice = _pair(scatter, EdgeSpec("a", "b", "x", "u1", mapping=WIDE_DENSE), back)
    assert not layout._reading_is_the_fields(twice, {"a": ("x",), "b": ("x",)})


def test_the_geometry_views_differ_in_whether_an_inbound_edge_counts():
    """The refusal under the interface norm is about internal edges; what the
    pass resolves, and so what the diagnostics must read, includes the edges
    into a member from outside, and never one out of the group."""
    internal = EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("source", "g"))
    inbound = EdgeSpec("o", "a", "x", "u1", mapping=_Custom(), geometry=("target", "g"))
    outbound = EdgeSpec("a", "o", "x", "u0", mapping=GEOM, geometry=("source", "g"))
    plan = _pair(inbound, outbound, internal, EdgeSpec("b", "a", "x", "u0"))
    assert _keys(plan.geometry_edges()) == ["a.x->b.u0"]
    assert _keys(plan.resolved_geometry_edges()) == ["o.x->a.u1", "a.x->b.u0"], "as declared"
    # Each field once, with the first mapping that reads it.
    assert plan.geometry_holders() == [("a", "g", inbound.mapping)]
    assert plan.reads_own_geometry("a", "g")
    assert not plan.reads_own_geometry("b", "g") and not plan.reads_own_geometry("a", "x")
    assert not _pair(internal, EdgeSpec("b", "a", "x", "u0")).reads_own_geometry("a", "g"), (
        "a source anchor is not the holder reading its own state")


def test_the_positions_a_pass_reads_from_the_iterate_are_listed_once_per_mapping():
    """What differs between the returned iterate and the fixed point: the
    positions a source-anchored internal edge reads with its value, and
    those a member that computes fluxes reads again for its flux hook.  A
    member's pre-step positions (a target anchor of ``update``) and an
    outside node's are constants of the pass and are not listed."""
    back = EdgeSpec("b", "a", "x", "u0")
    other = _Geometry()
    source = EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("source", "g"))
    again = EdgeSpec("a", "b", "x", "u1", mapping=other, geometry=("source", "g"))
    assert _pair(source, again, back).geometry_iterate_reads() == [
        ("a", "g", GEOM), ("a", "g", other)]
    # A target anchor of ``update``: the member's pre-step state.
    target = EdgeSpec("a", "b", "x", "u0", mapping=GEOM, geometry=("target", "g"))
    plain = {name: types.SimpleNamespace(node=_Plain()) for name in "abo"}
    assert _pair(target, back, nodes=plain).geometry_iterate_reads() == []
    # The same edge into a member that computes fluxes is resolved again
    # for the hook, from the member's in-pass state.
    fluxy = {**plain, "b": types.SimpleNamespace(node=_Fluxy())}
    assert _pair(target, back, nodes=fluxy).geometry_iterate_reads() == [("b", "g", GEOM)]
    # From outside the group: the outside node's positions are a constant,
    # the member's own are not where it computes fluxes.
    outside = EdgeSpec("o", "a", "x", "u1", mapping=GEOM, geometry=("source", "g"))
    inbound = EdgeSpec("o", "b", "x", "u1", mapping=GEOM, geometry=("target", "g"))
    assert _pair(outside, back, nodes=plain).geometry_iterate_reads() == []
    assert _pair(inbound, back, nodes=plain).geometry_iterate_reads() == []
    assert _pair(inbound, back, nodes=fluxy).geometry_iterate_reads() == [("b", "g", GEOM)]
    # Never an edge out of the group.
    outbound = EdgeSpec("a", "o", "x", "u0", mapping=GEOM, geometry=("source", "g"))
    assert _pair(outbound, back, nodes=plain).geometry_iterate_reads() == []


def test_a_flux_across_the_groups_boundary_counts_for_the_pass_and_not_for_the_refusals():
    """Whether a pass computes fluxes at all counts every edge with an end in
    the group; the refusals read the internal ones."""
    back = (EdgeSpec("a", "b", "x", "u0"), EdgeSpec("b", "a", "x", "u0"))
    assert not _pair(*back).resolves_a_flux()
    outbound = _pair(*back, EdgeSpec("a", "o", "q", "u0"))
    assert outbound.resolves_a_flux() and outbound.flux_edges() == []
    assert not outbound.flux_producer_reads_a_flux()
    inbound = _pair(*back, EdgeSpec("p", "b", "q", "u1"))
    assert inbound.resolves_a_flux() and inbound.flux_edges() == []
    assert not inbound.flux_producer_reads_a_flux(), "b reads it, and b produces none"
    into_a_producer = _pair(*back, EdgeSpec("p", "a", "q", "u1"))
    assert into_a_producer.flux_producer_reads_a_flux()
    # A source that is neither a field nor a hook's is refused by validate();
    # it is no flux to the pass, and is "not read from the state" to the refusal.
    missing = _pair(*back, EdgeSpec("b", "a", "gone", "u1"))
    assert not missing.resolves_a_flux() and _keys(missing.flux_edges()) == ["b.gone->a.u1"]


def test_which_inputs_are_coupled_and_who_reads_whom():
    plan = _pair(EdgeSpec("a", "b", "x", "u0"), EdgeSpec("a", "b", "n", "u1"),
                 EdgeSpec("b", "a", "x", "u0"), EdgeSpec("a", "a", "x", "u1"),
                 EdgeSpec("o", "a", "x", "u2"))
    assert plan.coupled_inputs() == {"a": {"u0", "u1"}, "b": {"u0", "u1"}}, "not the inbound u2"
    assert plan.member_reads() == {"a": {"a", "b"}, "b": {"a"}}, "itself included"
    assert ip.member_reads(plan.declared_edges(), frozenset("ab")) == plan.member_reads()
    assert [e.key for e in ip.internal_edges(EDGES, MEMBERS)] == [
        e.key for e in _plan().declared_edges()]


# ---------------------------------------------------------------------------
# The order the report's fallback floor reads the edges in, on a compiled graph
# ---------------------------------------------------------------------------
class _Relay(SimulationNode):
    """``x <- b + g * u0``."""

    def __init__(self, name, gain, bias):
        super().__init__(name, 1.0, g=jnp.asarray(gain, F32), b=jnp.asarray(bias, F32))

    def initial_state(self):
        return {"x": jnp.asarray(0.5, F32)}

    def boundary_input_spec(self):
        return {"u0": BoundaryInputSpec(shape=(), dtype=F32, default=jnp.asarray(0.0, F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": (p["b"] + p["g"] * boundary_inputs["u0"]).astype(F32)}


def test_the_reports_fallback_floor_reads_the_edges_in_the_order_the_norm_sums_them():
    """The residual sums a group's edges in the canonical order, and the
    report's fallback floor is handed them in that order
    (``_committed_floor_inputs``), which is not the declared one here."""
    gm = GraphManager()
    gm.add_node(_Relay("a", 0.5, 1.0))
    gm.add_node(_Relay("b", 0.5, 0.0))
    gm.add_edge("b", "a", "x", "u0")
    gm.add_edge("a", "b", "x", "u0")
    gm.add_coupling_group(["a", "b"], convergence_norm="interface", rtol=1e-6)
    gm.compile()
    _evaluations, _declared, internal = gm._committed_floor_inputs["a+b"]   # noqa: SLF001
    assert [e.key for e in internal] == ["a.x->b.u0", "b.x->a.u0"]
    plan = ip.interface_plan(frozenset("ab"), gm._edges, gm.schedule,       # noqa: SLF001
                             gm._state, gm._nodes)                          # noqa: SLF001
    assert list(gm.schedule) == ["a", "b"]
    assert _keys(plan.internal) == ["a.x->b.u0", "b.x->a.u0"]
    assert internal == plan.norm_edges() == tuple(r.edge for r in plan.internal)
    assert [e.key for e in plan.declared_edges()] == ["b.x->a.u0", "a.x->b.u0"]


# ---------------------------------------------------------------------------
# A geometry-dependent mapping: the parts of its reading (experimental)
# ---------------------------------------------------------------------------
#
# The decision the plan is held to: a gather (and a tie) is read as
# delivered, at the geometry the step uses; a scatter onto more entries is
# read at its inputs, the source value and -- for a source anchor -- the
# positions in units of the kind's own length scale (the grid spacing per
# axis).  A target-anchored geometry is the target's pre-step state: a
# constant of the solve, and no part of a reading.

_SPACING = (0.5, 0.125)


def _grid(mode, n_points=3, shape=(4, 2), layout="flat"):
    from maddening.core.coupling.grid_mapping import multilinear_grid_mapping  # noqa: PLC0415

    return multilinear_grid_mapping([0.0] * len(shape), _SPACING[:len(shape)], shape,
                                    n_points=n_points, mode=mode, layout=layout)


def _geometry_record(mode, anchor, **grid):
    return ip._edge_record(EdgeSpec("a", "b", "x", "u0", mapping=_grid(mode, **grid),
                                    geometry=(anchor, "pos")))


@pytest.mark.parametrize("mode, n_points, shape, side", [
    ("conservative", 3, (4, 2), "source"),      # a scatter: 3 points onto 8 cells
    ("consistent", 3, (4, 2), "delivered"),     # its gather: 8 cells at 3 points
    ("conservative", 8, (4, 2), "delivered"),   # a tie
    ("consistent", 8, (4, 2), "delivered"),     # a tie
    ("conservative", 5, (3,), "delivered"),     # 5 points onto 3 cells: fewer entries
    ("consistent", 5, (3,), "source"),          # 3 cells at 5 points: more entries
])
@pytest.mark.parametrize("layout", ["flat", "shaped"])
def test_a_geometry_mapping_is_placed_by_the_sizes_it_declares(mode, n_points, shape, side,
                                                               layout):
    """The compact side, by the rule of a static mapping: the mode's name
    decides nothing, and neither does the grid's layout."""
    for anchor in ("source", "target"):
        record = _geometry_record(mode, anchor, n_points=n_points, shape=shape, layout=layout)
        assert (record.mapping_form, record.norm_side) == ("needs_geometry", side)


def test_the_parts_of_a_geometry_edges_reading():
    """What is read, against what, and which fields it holds whole."""
    scatter = _geometry_record("conservative", "source")
    assert scatter.parts == (
        ip.ReadingPart("source", ("a", "x"), True, "own_magnitude"),
        ip.ReadingPart("geometry", ("a", "pos"), True, "kernel_length"))
    assert scatter.measured_whole == (("a", "x"), ("a", "pos"))
    assert not scatter.reads_source_as_is, "a second part, in another unit"
    assert not scatter.reads_through_mapping and not scatter.reads_pre_step_geometry

    anchored_at_target = _geometry_record("conservative", "target")
    assert anchored_at_target.parts == (
        ip.ReadingPart("source", ("a", "x"), True, "own_magnitude"),)
    assert anchored_at_target.measured_whole == (("a", "x"),)
    assert anchored_at_target.reads_source_as_is, "the pre-step positions are no reading"

    for anchor in ("source", "target"):
        gather = _geometry_record("consistent", anchor)
        assert gather.parts == (ip.ReadingPart("delivered", ("a", "x"), False, "own_magnitude"),)
        assert gather.measured_whole == () and not gather.reads_source_as_is
        assert gather.reads_through_mapping and not gather.reads_weights_of_the_step
        assert gather.reads_pre_step_geometry == (anchor == "target")


def test_a_position_is_read_in_the_kinds_own_length_on_each_axis():
    """The grid spacing of the axis: 0.5 on the first and 0.125 on the
    second here, so a wrong axis is another number.  The value is read as
    stored."""
    assert _grid("conservative").geometry_length_scale() == _SPACING
    x = jnp.asarray([1.0, -2.0, 0.5], F32)
    pos = jnp.asarray([[0.75, 0.0625], [1.25, 0.03125], [0.5, 0.09375]], F32)
    record = _geometry_record("conservative", "source")
    value, positions = record.read(({"a": {"x": x, "pos": pos}},))
    assert value[0] is record.edge and value[1] == jnp.dtype(F32) and value[2] is x
    assert value.part.what == "source" and value.positions == (None,)
    assert positions.part.unit == "kernel_length" and positions[1] == jnp.dtype(F32)
    np.testing.assert_array_equal(positions[2], np.asarray(pos) / np.asarray(_SPACING))
    assert np.asarray(positions[2]).dtype == np.float32, "in the geometry's own dtype"
    swapped = np.asarray(pos) / np.asarray(_SPACING[::-1])
    assert not np.allclose(positions[2], swapped)
    # One axis, and the flat ``(points,)`` geometry a one-dimensional grid takes.
    line = _geometry_record("conservative", "source", shape=(6,))
    for flat in (pos[:, :1], pos[:, 0]):
        _value, got = line.read(({"a": {"x": x, "pos": flat}},))
        np.testing.assert_array_equal(got[2], np.asarray(flat) / _SPACING[0])
        assert got[2].shape == flat.shape


def test_a_readings_geometry_is_at_the_time_level_the_step_uses():
    """A source-anchored geometry is the field of the same state the value
    is read from; a target-anchored one is the target's pre-step state,
    the same at every state, and outside a step it cannot be read."""
    field = jnp.arange(8.0, dtype=F32)
    new_pos = jnp.asarray([[0.25, 0.03], [0.80, 0.06], [1.20, 0.10]], F32)
    old_pos = new_pos + 0.05
    pre_pos = new_pos - 0.07
    new = {"a": {"x": field, "pos": new_pos}, "b": {"pos": new_pos + 0.3}}
    old = {"a": {"x": 2.0 * field, "pos": old_pos}, "b": {"pos": old_pos + 0.3}}
    mapping = _grid("consistent")

    (at_source,) = _geometry_record("consistent", "source").read((new, old))
    np.testing.assert_array_equal(at_source[2], mapping.apply(field, None, new_pos))
    np.testing.assert_array_equal(at_source[3], mapping.apply(2.0 * field, None, old_pos))
    assert not np.allclose(at_source[3], mapping.apply(2.0 * field, None, new_pos))
    np.testing.assert_array_equal(
        at_source.positions[1], np.asarray(old_pos) / np.asarray(_SPACING, np.float32))

    record = _geometry_record("consistent", "target")
    for pre_step in ({"b": {"pos": pre_pos}}, lambda name: {"b": {"pos": pre_pos}}[name]):
        (at_target,) = record.read((new, old), pre_step=pre_step)
        np.testing.assert_array_equal(at_target[2], mapping.apply(field, None, pre_pos))
        np.testing.assert_array_equal(at_target[3], mapping.apply(2.0 * field, None, pre_pos))
        assert not np.allclose(at_target[2], mapping.apply(field, None, new["b"]["pos"]))
    with pytest.raises(ValueError, match="pre-step b.pos .a target-anchored geometry."):
        record.read((new,))
    np.testing.assert_array_equal(record.geometry_at(new, {"b": {"pos": pre_pos}}), pre_pos)
    assert _geometry_record("consistent", "source").geometry_at(new) is new_pos


def _geometry_pair(mode, anchor, back=None):
    state = {"a": {"x": jnp.ones(8 if mode == "consistent" else 3, F32),
                   "pos": jnp.zeros((3, 2), F32)},
             "b": {"x": jnp.ones(2, F32), "pos": jnp.zeros((3, 2), F32)}}
    return _pair(EdgeSpec("a", "b", "x", "u0", mapping=_grid(mode), geometry=(anchor, "pos")),
                 back or EdgeSpec("b", "a", "x", "u0"), state=state)


def test_the_step_records_the_floor_where_the_returned_state_cannot_repeat_the_reading():
    """Who owns ``reading_floor``: a group read through weights the step
    ran with, or at a target's pre-step geometry.  A geometry-dependent
    mapping holds no weights; read at the returned state's own geometry,
    or at its source, it needs no record."""
    owners = {(mode, anchor): _geometry_pair(mode, anchor).norm_reads_beyond_the_state()
              for mode in ("consistent", "conservative") for anchor in ("source", "target")}
    assert owners == {("consistent", "target"): True, ("consistent", "source"): False,
                      ("conservative", "source"): False, ("conservative", "target"): False}
    for mode in ("consistent", "conservative"):
        for anchor in ("source", "target"):
            assert not _geometry_pair(mode, anchor).norm_reads_mapping_weights()
    tie = _pair(EdgeSpec("a", "b", "x", "u0", mapping=DENSE), EdgeSpec("b", "a", "x", "u0"))
    assert tie.norm_reads_mapping_weights() and tie.norm_reads_beyond_the_state()
    group = types.SimpleNamespace(convergence_norm="interface", atol=0.0)
    assert layout._reads_mapping_weights(group, _geometry_pair("consistent", "target"))
    assert not layout._reads_mapping_weights(group, _geometry_pair("consistent", "source"))
    # The report's fallback is asked of the bare edges it keeps.
    for anchor, needs in (("target", True), ("source", False)):
        edges = _geometry_pair("consistent", anchor).norm_edges()
        assert layout._floor_needs_the_step(group, edges) is needs
        assert not layout._floor_needs_the_step(
            types.SimpleNamespace(convergence_norm="mixed", atol=0.0), edges)
    assert not layout._floor_needs_the_step(group, tie.norm_edges()), (
        "weights: the graph's own are the fallback")


def test_the_fields_measured_whole_follow_the_parts_and_the_spectrums_weights_do_not_yet():
    """The return rule keeps what a part holds whole: the source of a
    scatter and the positions it reads from its source.  The accelerator
    already counts those positions.  **The spectrum's weights
    (``source_fields``) still leave them out**: the bounds of such a group
    are reported not usable until the stage that analyses this reading,
    and that stage starts here."""
    plan = _geometry_pair("conservative", "source")
    assert plan.measured_whole() == {("a", "x"), ("a", "pos"), ("b", "x")}
    assert plan.iqn_fields() == {"a": ("pos", "x"), "b": ("x",)}
    assert ("a", "pos") not in plan.source_fields()
    assert _geometry_pair("conservative", "target").measured_whole() == {("a", "x"), ("b", "x")}
    assert _geometry_pair("consistent", "source").measured_whole() == {("b", "x")}
    group = types.SimpleNamespace(nodes=frozenset("ab"), convergence_norm="interface")
    state = {"a": {"x": jnp.ones(3, F32), "pos": jnp.zeros((3, 2), F32), "w": jnp.ones(1, F32)},
             "b": {"x": jnp.ones(2, F32), "pos": jnp.zeros((3, 2), F32)}}
    assert layout._fields_the_interface_norm_misses(group, plan, ["a", "b"], state) == {
        "a": ("w",), "b": ("pos",)}
    assert layout._fields_the_interface_norm_misses(
        group, _geometry_pair("conservative", "target"), ["a", "b"], state) == {
            "a": ("pos", "w"), "b": ("pos",)}
    assert layout._fields_the_interface_norm_misses(
        group, _geometry_pair("consistent", "source"), ["a", "b"], state) == {
            "a": ("pos", "w", "x"), "b": ("pos",)}
    floats = {"a": ("pos", "x"), "b": ("pos", "x")}
    assert not layout._reading_is_the_fields(plan, floats)
    assert layout._reading_is_the_fields(_geometry_pair("conservative", "target"), floats)


def test_the_interface_norm_refuses_a_sub_cycled_group_and_another_kind_by_name():
    """The two narrowings, each with its reason; a single-rate group of
    the ``multilinear_grid`` kind is accepted, and so is every group under
    another norm."""
    plan = _geometry_pair("conservative", "source",
                          back=EdgeSpec("b", "a", "x", "u0", mapping=_grid("consistent"),
                                        geometry=("target", "pos")))
    one_rate = {"a": types.SimpleNamespace(timestep=0.1), "b": types.SimpleNamespace(timestep=0.1)}
    two_rates = {"a": types.SimpleNamespace(timestep=0.1), "b": types.SimpleNamespace(timestep=0.05)}

    def group(norm="interface", subcycling=False):
        return types.SimpleNamespace(convergence_norm=norm, nodes=frozenset("ab"),
                                     subcycling=subcycling)

    assert layout._geometry_edge_coupling_errors(group(), one_rate, plan) == []
    assert layout._geometry_edge_coupling_errors(group(subcycling=True), one_rate, plan) == []
    errors = layout._geometry_edge_coupling_errors(group(subcycling=True), two_rates, plan)
    assert len(errors) == 2 and all(e.startswith("ERROR:") for e in errors)
    for error, key in zip(errors, ("a.x->b.u0", "b.x->a.u0")):
        assert key in error and "in a sub-cycled group" in error and "['b']" in error
        assert "Use convergence_norm='mixed' or 'l2'" in error
    for norm in ("l2", "mixed"):
        assert layout._geometry_edge_coupling_errors(group(norm, True), two_rates, plan) == []
