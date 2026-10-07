"""What a geometry edge accepts, what it refuses, and what a config carries of it.

``add_edge(..., mapping=m, geometry=(anchor, field))`` names the moving
geometry a geometry-dependent mapping reads: a state field of the edge's
own source (``"source"``) or target (``"target"``) node.  Every way of
getting that wrong is refused where it is made, with a message that says
what to do, and never runs:

====  ====================  ===============================================
G1    ``add_edge``          a geometry without a mapping
G2    ``add_edge``          a geometry on a static mapping (it would be
                            ignored)
G3    ``add_edge``          a geometry-dependent mapping without a geometry
G4    ``add_edge``          anything but ``("source" | "target", <field>)``,
                            another node's name included
G5    ``compile``           the field is not in the anchor's state
G6    ``compile``           the field is a boundary flux of the anchor
G7    ``compile``           the field is not float32 or float64
G8    ``compile``           the field's shape is not the mapping's
                            ``geometry_shape``
G9    ``compile``           either end is sharded
G10   adaptive steppers     any geometry edge in the graph
G11   ``replace_node``      the replacement drops the geometry field
G12   ``DatasetGenerator``  the target receives a geometry edge
G13   ``build_mapping``     a factory's product disagrees with its kind's
                            ``needs_geometry``
====  ====================  ===============================================

(G14, a non-floating field through the ``multilinear_grid`` kind, is with
that kind's tests in ``test_multilinear_grid_kernel.py``.)

``add_edge`` raises ``ValueError``; ``validate()`` returns the compile-time
ones as ``ERROR:`` issues and ``compile()`` raises them as
``RuntimeError``.  Each test asserts the exception type and the phrase of
the message that tells the user what is wrong, and -- where the refusal
promises it -- that nothing was changed.  Each has a control beside it:
the same call with the one thing put right goes through.
"""

from __future__ import annotations

import inspect
import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping, nearest_neighbor_mapping, register_mapping
from maddening.core.coupling.mapping_registry import _unregister
from maddening.core.coupling.mapping_spec import MappingRebuildError, MappingSpec
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.replace import replace_node

from tests.property import geometry_graphs as gg

KEY = "a.x->b.u"
N_SOURCE, N_TARGET = 3, 2
SHAPE = (N_TARGET, N_SOURCE)


class Holder(SimulationNode):
    """``x <- 0.5 x + u`` and a kept field ``g`` of shape ``(rows, cols)``.

    Its constructor arguments are plain integers, so a graph of these
    round-trips through ``to_dict`` / ``from_dict``.
    """

    g_dtype = "float32"

    def __init__(self, name, timestep, n=3, rows=N_TARGET, cols=N_SOURCE):
        super().__init__(name, timestep, n=n, rows=rows, cols=cols)

    def _geometry(self):
        shape = (int(self.params["rows"]), int(self.params["cols"]))
        size = shape[0] * shape[1]
        if self.g_dtype == "key":
            return jax.random.split(jax.random.key(0), size).reshape(shape)
        # Different at the two ends of an edge, so the anchor matters.
        values = np.arange(1, size + 1).reshape(shape) + 3 * (ord(self.name[0]) - ord("a"))
        if self.g_dtype == "bool":
            return jnp.asarray(values % 2 == 0)
        dtype = jnp.dtype(self.g_dtype)
        return jnp.asarray(values / 8 if jnp.issubdtype(dtype, jnp.floating) else values, dtype)

    def initial_state(self):
        n = int(self.params["n"])
        return {"x": jnp.arange(1, n + 1, dtype=jnp.float32), "g": self._geometry()}

    def boundary_input_spec(self):
        n = int(self.params["n"])
        return {"u": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                       default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.zeros_like(state["x"]))
        return {"x": jnp.float32(0.5) * state["x"] + u, "g": state["g"]}


class FluxHolder(Holder):
    """A :class:`Holder` with fluxes: ``q`` and ``gq``, which has the geometry's shape."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": jnp.float32(2.0) * state["x"], "gq": jnp.float32(2.0) * state["g"]}


def _holder_of(dtype: str) -> type:
    return type(f"Holder_{dtype}", (Holder,), {"g_dtype": dtype})


REGISTRY = {"Holder": Holder}


def _graph(source=Holder, target=Holder, **edge):
    """``a (3) -> b (2)`` with the edge given by *edge* (none when empty)."""
    gm = GraphManager()
    gm.add_node(source("a", 1.0, n=N_SOURCE))
    gm.add_node(target("b", 1.0, n=N_TARGET))
    if edge:
        gm.add_edge("a", "b", "x", "u", **edge)
    return gm


def _geom():
    return gg.geom_matrix_mapping(N_TARGET, N_SOURCE)


def _static():
    return matrix_mapping(np.ones(SHAPE, np.float32))


# ---------------------------------------------------------------------------
# What add_edge accepts
# ---------------------------------------------------------------------------


def test_geometry_is_the_last_keyword_only_argument_of_add_edge_and_defaults_to_none():
    params = inspect.signature(GraphManager.add_edge).parameters
    assert list(params)[-1] == "geometry"
    assert params["geometry"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["geometry"].default is None
    gm = _graph(mapping=_static())
    assert gm.edges[0].geometry is None


def test_an_edge_spec_holds_the_geometry_as_its_last_field_and_carries_it_through_a_re_add():
    names = [f.name for f in EdgeSpec.__dataclass_fields__.values()]
    assert names[-2:] == ["ordinal", "geometry"]
    gm = _graph(mapping=_geom(), geometry=("target", "g"))
    edge = gm.edges[0]
    assert edge.geometry == ("target", "g") and type(edge.geometry) is tuple
    hash(edge)
    assert edge.add_edge_kwargs()["geometry"] == ("target", "g")
    # The geometry is not part of an edge's identity.
    assert edge.key == KEY
    gm.remove_edge("a", "b", "x", "u")
    assert gm.edges == []
    gm.add_edge(**edge.add_edge_kwargs())
    assert gm.edges[0].geometry == ("target", "g")


@pytest.mark.parametrize("spelling", [("source", "g"), ["source", "g"],
                                      {"anchor": "source", "field": "g"},
                                      {"field": "g", "anchor": "source"}],
                         ids=["tuple", "list", "dict", "dict-reordered"])
def test_every_accepted_spelling_is_normalised_to_the_tuple(spelling):
    gm = _graph(mapping=_geom(), geometry=spelling)
    assert gm.edges[0].geometry == ("source", "g") and type(gm.edges[0].geometry) is tuple


@pytest.mark.parametrize("anchor", ["source", "target"])
def test_a_well_formed_geometry_edge_compiles_and_steps(anchor):
    """The control for every refusal below."""
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    assert not [i for i in gm.validate() if i.startswith("ERROR")]
    gm.compile()
    gm.step()
    held = gm.get_node_state("a" if anchor == "source" else "b")["g"]
    assert np.any(np.asarray(gm.get_node_state("a")["g"]) != np.asarray(
        gm.get_node_state("b")["g"])), "premise: the two ends hold different geometries"
    # ``a`` runs first and has no input: ``b`` reads ``0.5 * a.x0``.
    want = np.asarray(held) @ (0.5 * np.arange(1.0, N_SOURCE + 1))
    assert gm.params["mappings"][KEY] == {}
    np.testing.assert_allclose(
        np.asarray(gm.get_node_state("b")["x"]), 0.5 * np.arange(1.0, N_TARGET + 1) + want,
        rtol=1e-6)


# ---------------------------------------------------------------------------
# G1 to G4: add_edge
# ---------------------------------------------------------------------------


def _refused_add_edge(phrases, **edge):
    gm = _graph()
    with pytest.raises(ValueError) as refused:
        gm.add_edge("a", "b", "x", "u", **edge)
    message = str(refused.value)
    for phrase in (f"add_edge({KEY})", *phrases):
        assert phrase in message, (phrase, message)
    assert gm.edges == [], "a refused edge was added"
    return message


class _TwoArgumentMapping:
    """A mapping written before ``geom`` existed: ``apply`` takes the field
    and the weights and nothing else."""

    kind = "test_two_argument"
    mode = "consistent"
    n_source, n_target = N_SOURCE, N_TARGET

    def params_pytree(self):
        return {}

    def apply(self, field, weights):
        return field[:N_TARGET] * jnp.float32(2.0)

    def apply_T(self, field, weights):
        return jnp.concatenate([field * jnp.float32(2.0), jnp.zeros(N_SOURCE - N_TARGET)])


def test_an_edge_without_a_geometry_still_makes_the_two_argument_call():
    """``apply(value, weights)``, as before the feature: a mapping whose
    ``apply`` has no third parameter steps on an ordinary edge.  (The
    program-text gate cannot see this: the shipped kinds accept
    ``geom=None``, and passing it changes no program.)"""
    gm = _graph(mapping=_TwoArgumentMapping())
    gm.compile()
    gm.step()
    # b.x <- 0.5 * [1, 2] + 2 * a.x[:2], with a.x read after a's own update.
    a = np.asarray(gm.get_node_state("a")["x"])
    assert np.array_equal(np.asarray(gm.get_node_state("b")["x"]),
                          np.float32(0.5) * np.arange(1, N_TARGET + 1, dtype=np.float32)
                          + np.float32(2.0) * a[:N_TARGET])


def test_g1_a_geometry_without_a_mapping_is_refused():
    _refused_add_edge(["was given without a mapping", "pass mapping=", "drop geometry="],
                      geometry=("source", "g"))


@pytest.mark.parametrize("make", [_static, lambda: nearest_neighbor_mapping(
    np.linspace(0, 1, N_SOURCE), np.linspace(0, 1, N_TARGET))], ids=["matrix", "nearest"])
def test_g2_a_geometry_on_a_static_mapping_is_refused(make):
    message = _refused_add_edge(["is static (it reads no geometry)", "would be ignored"],
                                mapping=make(), geometry=("source", "g"))
    assert "('source', 'g')" in message


def test_g3_a_geometry_dependent_mapping_without_a_geometry_is_refused():
    _refused_add_edge(["reads a moving geometry and none was given",
                       'geometry=("source", <state field>)', 'geometry=("target", <state field>)'],
                      mapping=_geom())


@pytest.mark.parametrize("bad", [
    "g", "source", ("g",), ("source",), ("source", "g", "extra"), ("a", "g"), ("b", "g"),
    ("other", "g"), ("Source", "g"), ("SOURCE", "g"), (" target", "g"), ("source", ""),
    ("", "g"), ("source", None), (None, "g"), ("source", 3), (0, "g"), ("g", "source"),
    {"anchor": "source"}, {"field": "g"}, {"anchor": "a", "field": "g"},
    {"anchor": "source", "field": ""}, {"anchor": "source", "field": "g", "node": "a"},
    {"node": "a", "field": "g"}, 5, 2.5, True, {"source", "g"}, [("source", "g")], (),
], ids=repr)
def test_g4_a_malformed_geometry_or_one_on_another_node_is_refused(bad):
    """Not ``("source" | "target", <field>)`` in one of the accepted
    spellings: a bare field name, a node's name as the anchor (a geometry
    held by another node is not supported), a wrong length, an empty or
    non-string part, a dict with other keys."""
    message = _refused_add_edge(
        ['geometry must be ("source", <field>) or ("target", <field>)',
         "A geometry held by any other node is not supported"],
        mapping=_geom(), geometry=bad)
    assert repr(bad) in message


# ---------------------------------------------------------------------------
# G5 to G9: compile
# ---------------------------------------------------------------------------


def _refused_compile(gm, phrases):
    errors = [i for i in gm.validate() if i.startswith("ERROR")]
    hits = [i for i in errors if all(p in i for p in phrases)]
    assert len(hits) == 1, (phrases, errors)
    assert hits[0].startswith(f"ERROR: edge {KEY}: "), hits[0]
    with pytest.raises(RuntimeError) as refused:
        gm.compile()
    for phrase in phrases:
        assert phrase in str(refused.value), (phrase, str(refused.value))
    return hits[0]


@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g5_a_geometry_field_the_anchor_does_not_hold_is_refused(anchor, node):
    gm = _graph(mapping=_geom(), geometry=(anchor, "positions"))
    issue = _refused_compile(gm, [f"geometry field 'positions' is not in the state of its "
                                  f"{anchor} node {node!r}", "State fields:"])
    assert "'g'" in issue and "'x'" in issue


@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g6_a_boundary_flux_as_the_geometry_is_refused(anchor, node):
    """``gq`` has the geometry's shape and dtype: it is refused for being a
    flux (recomputed within a step, no single time level), not for either."""
    gm = _graph(source=FluxHolder, target=FluxHolder, mapping=_geom(),
                geometry=(anchor, "gq"))
    _refused_compile(gm, [f"geometry field 'gq' of {node!r} is a boundary flux",
                          "not a state field", f"Hold the geometry in {node!r}'s state"])


@pytest.mark.parametrize("dtype", ["int32", "uint32", "bool", "float16", "bfloat16", "key"])
@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g7_a_geometry_that_is_not_float32_or_float64_is_refused(anchor, node, dtype):
    cls = _holder_of(dtype)
    gm = _graph(source=cls, target=cls, mapping=_geom(), geometry=(anchor, "g"))
    _refused_compile(gm, [f"geometry field {node}.g has dtype",
                          "a geometry must be a float32 or float64 array"])


def test_g7_a_float64_geometry_is_accepted():
    with gg.x64(True):
        cls = _holder_of("float64")
        gm = _graph(source=cls, target=cls, mapping=_geom(), geometry=("source", "g"))
        assert not [i for i in gm.validate() if i.startswith("ERROR")]
        gm.compile()
        assert gm.get_node_state("a")["g"].dtype == jnp.float64


@pytest.mark.parametrize("rows, cols", [(N_SOURCE, N_TARGET), (N_TARGET, N_SOURCE + 1),
                                        (N_TARGET * N_SOURCE, 1), (1, N_TARGET * N_SOURCE)])
@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g8_a_geometry_of_another_shape_is_refused(anchor, node, rows, cols):
    gm = GraphManager()
    gm.add_node(Holder("a", 1.0, n=N_SOURCE, rows=rows, cols=cols))
    gm.add_node(Holder("b", 1.0, n=N_TARGET, rows=rows, cols=cols))
    gm.add_edge("a", "b", "x", "u", mapping=_geom(), geometry=(anchor, "g"))
    _refused_compile(gm, [f"geometry field {node}.g has shape {(rows, cols)}",
                          f"reads a geometry of shape {SHAPE}"])


class _Wrapper(Holder):
    """A node that holds another as ``_inner``, as a wrapper does."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)
        self._inner = Holder(name, timestep, n=n)


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("x",))


@pytest.mark.parametrize("wrapped", [False, True], ids=["sharded", "wraps-a-sharded-node"])
@pytest.mark.parametrize("anchor", ["source", "target"])
@pytest.mark.parametrize("which, node", [("source", "a"), ("target", "b")])
def test_g9_a_sharded_end_is_refused(which, node, anchor, wrapped):
    """Either end, whichever end holds the geometry: a node carrying a
    device mesh (the ``Sharded*Node`` convention: a non-``None``
    ``_mesh``), or wrapping one that does."""
    gm = GraphManager()
    for name, n in (("a", N_SOURCE), ("b", N_TARGET)):
        nd = (_Wrapper if wrapped else Holder)(name, 1.0, n=n)
        if name == node:
            (nd._inner if wrapped else nd)._mesh = _mesh()     # noqa: SLF001
        gm.add_node(nd)
    gm.add_edge("a", "b", "x", "u", mapping=_geom(), geometry=(anchor, "g"))
    errors = [i for i in gm.validate() if i.startswith("ERROR")]
    hits = [i for i in errors if f"its {which} node {node!r} is sharded" in i]
    assert len(hits) == 1 and hits[0].startswith(f"ERROR: edge {KEY}: "), errors
    assert "not supported on an edge with a sharded end" in hits[0]
    assert "Use an unsharded node on both ends" in hits[0]


def test_every_compile_refusal_is_about_geometry_edges_only():
    """The controls: the same nodes with a static mapping, or with no edge
    at all, raise none of G5 to G9 (a flux producer, an integer field and a
    field of another shape are ordinary state there)."""
    for cls in (FluxHolder, _holder_of("int32"), _holder_of("key")):
        gm = _graph(source=cls, target=cls, mapping=_static())
        assert not [i for i in gm.validate() if "geometry" in i]
        gm.compile()


# ---------------------------------------------------------------------------
# G10: the adaptive steppers
# ---------------------------------------------------------------------------


class _Counting(gg.GeomMatrixMapping):
    calls = 0

    def apply(self, field, weights=None, geom=None):
        type(self).calls += 1
        return super().apply(field, weights, geom)


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan", "_build_dt_step_fn"])
def test_g10_the_adaptive_steppers_refuse_a_graph_with_a_geometry_edge(entry):
    """Before any trace: the time level a geometry is read at across half
    steps and rejected attempts is not defined."""
    mapping = _Counting(N_TARGET, N_SOURCE, _geom().spec)
    gm = _graph(mapping=mapping, geometry=("target", "g"))
    gm.compile()
    before = (_Counting.calls, gm.trace_count)
    call = {"run_adaptive": lambda: gm.run_adaptive(0.5, dt_initial=0.1),
            "run_adaptive_scan": lambda: gm.run_adaptive_scan(0.5, max_steps=8, dt_initial=0.1),
            "_build_dt_step_fn": lambda: gm._build_dt_step_fn()}[entry]    # noqa: SLF001
    with pytest.raises(RuntimeError) as refused:
        call()
    message = str(refused.value)
    assert message.startswith(f"{entry}: "), message
    for phrase in ("geometry-dependent mapping", KEY,
                   "the adaptive steppers do not support in 0.4.0", "Use step / run_scan"):
        assert phrase in message, (phrase, message)
    assert (_Counting.calls, gm.trace_count) == before, "the refusal came after a trace"


def test_g10_the_adaptive_steppers_still_run_the_same_graph_with_a_static_mapping():
    gm = _graph(mapping=_static())
    gm.compile()
    with warnings.catch_warnings():
        # (The holder's update is not a time integrator, so the stepper
        # complains about its error estimate; that is not what is asked.)
        warnings.simplefilter("ignore", UserWarning)
        gm.run_adaptive(0.2, dt_initial=0.1)
        gm._build_dt_step_fn()      # noqa: SLF001


# ---------------------------------------------------------------------------
# G11: replace_node
# ---------------------------------------------------------------------------


class _NoGeometry(Holder):
    def initial_state(self):
        return {"x": super().initial_state()["x"]}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"]}


@pytest.mark.parametrize("replacement", [
    _NoGeometry, _holder_of("int32"), _holder_of("float16"),
    lambda name, dt, n: Holder(name, dt, n=n, rows=N_SOURCE, cols=N_TARGET),
], ids=["no-field", "integer", "float16", "another-shape"])
@pytest.mark.parametrize("anchor, node, n", [("source", "a", N_SOURCE), ("target", "b", N_TARGET)])
def test_g11_replacing_the_node_that_holds_a_geometry_with_one_that_does_not_is_refused(
        anchor, node, n, replacement):
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    nodes, edges = dict(gm._nodes), list(gm.edges)       # noqa: SLF001
    with pytest.raises(ValueError) as refused:
        replace_node(gm, node, replacement(node, 1.0, n=n))
    message = str(refused.value)
    for phrase in (f"replace_node({node!r})", f"edge {KEY} reads its geometry from {node}.g",
                   f"shape {SHAPE}", "float32", "does not hold", "Nothing was changed."):
        assert phrase in message, (phrase, message)
    assert dict(gm._nodes) == nodes and gm.edges == edges     # noqa: SLF001
    assert all(a is b for a, b in zip(gm.edges, edges))
    gm.step()       # and the graph still runs


@pytest.mark.parametrize("anchor, other, n", [("source", "b", N_TARGET),
                                               ("target", "a", N_SOURCE)])
def test_g11_replacing_the_other_end_of_a_geometry_edge_is_unaffected(anchor, other, n):
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    replace_node(gm, other, _NoGeometry(other, 1.0, n=n))
    assert gm.edges[0].geometry == (anchor, "g") and gm.edges[0].key == KEY
    gm.compile()
    gm.step()


@pytest.mark.parametrize("anchor, node, n", [("source", "a", N_SOURCE), ("target", "b", N_TARGET)])
def test_g11_a_replacement_that_holds_the_geometry_keeps_the_edge_s_geometry(anchor, node, n):
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    replace_node(gm, node, Holder(node, 1.0, n=n))
    assert gm.edges[0].geometry == (anchor, "g")
    gm.compile()
    gm.step()


# ---------------------------------------------------------------------------
# G12: the surrogate dataset generator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["source", "target"])
def test_g12_the_dataset_generator_refuses_a_target_fed_through_a_geometry_edge(anchor):
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    with pytest.raises(ValueError) as refused:
        DatasetGenerator.from_graph(gm, "b", 4)
    message = str(refused.value)
    for phrase in ("DatasetGenerator", "node 'b' receives edge", KEY,
                   "geometry-dependent mapping",
                   "cannot be rebuilt from a state history"):
        assert phrase in message, (phrase, message)
    with pytest.raises(ValueError, match="cannot be rebuilt from a state history"):
        DatasetGenerator.from_sweep(gm, "b", 4, {
            name: {k: jnp.stack([v, v]) for k, v in gm.get_node_state(name).items()}
            for name in ("a", "b")})


def test_g12_the_dataset_generator_still_serves_the_source_of_a_geometry_edge():
    """The other end receives nothing through the edge: its inputs are rebuilt as before."""
    gm = _graph(mapping=_geom(), geometry=("source", "g"))
    gm.compile()
    data = DatasetGenerator.from_graph(gm, "a", 4)
    assert data is not None


# ---------------------------------------------------------------------------
# G13: a kind's registration and its factory must agree
# ---------------------------------------------------------------------------


def _config(mapping_dict, geometry=None) -> dict:
    edge = {"source_node": "a", "target_node": "b", "source_field": "x", "target_field": "u",
            "mapping": mapping_dict}
    if geometry is not None:
        edge["geometry"] = geometry
    return {"nodes": [{"type": "Holder", "name": "a", "timestep": 1.0,
                       "params": {"n": N_SOURCE}},
                      {"type": "Holder", "name": "b", "timestep": 1.0,
                       "params": {"n": N_TARGET}}],
            "edges": [edge], "external_inputs": []}


@pytest.mark.parametrize("declared", [True, False])
def test_g13_a_factory_whose_product_disagrees_with_its_registration_is_refused(declared):
    """A kind registered as geometry-dependent whose factory returns a static
    mapping, and the reverse: a config could then be accepted or refused by
    the flag and run by the object.  Refused when the mapping is rebuilt,
    naming the kind."""
    kind = f"test_disagrees_{'geometry' if declared else 'static'}"

    class Product(gg.GeomMatrixMapping):
        needs_geometry = not declared

    def factory(*, n_target, n_source):
        m = Product(n_target, n_source, MappingSpec(kind, {"n_target": int(n_target),
                                                           "n_source": int(n_source)}, {}))
        m.kind = kind
        return m

    register_mapping(kind, arrays=(), hyperparameters={"n_target": int, "n_source": int},
                     needs_geometry=declared)(factory)
    try:
        spec = {"kind": kind, "n_target": N_TARGET, "n_source": N_SOURCE, "points": {}}
        with pytest.raises(ValueError) as refused:
            MappingSpec.from_dict(spec).build(lambda ref: None)
        assert kind in str(refused.value), str(refused.value)
        geometry = {"anchor": "source", "field": "g"} if declared else None
        with pytest.raises(MappingRebuildError) as wrapped:
            GraphManager.from_dict(_config(spec, geometry), REGISTRY)
        assert kind in str(wrapped.value) and "a.x -> b.u" in str(wrapped.value)
    finally:
        _unregister(kind)


def test_g13_the_test_kind_s_registration_and_product_agree():
    """The control: ``test_geom_matrix`` rebuilds from its spec."""
    mapping = _geom()
    rebuilt = mapping.spec.build(lambda ref: None)
    assert type(rebuilt) is gg.GeomMatrixMapping and rebuilt.needs_geometry is True
    assert rebuilt.geometry_shape == SHAPE


# ---------------------------------------------------------------------------
# What a config carries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["source", "target"])
def test_a_config_carries_the_geometry_and_from_dict_restores_it(anchor):
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    config = json.loads(json.dumps(gm.to_dict()))
    assert config["edges"][0]["geometry"] == {"anchor": anchor, "field": "g"}
    assert config["edges"][0]["mapping"]["kind"] == gg.GEOM_MATRIX
    loaded = GraphManager.from_dict(config, REGISTRY)
    assert loaded.edges[0].geometry == (anchor, "g")
    assert type(loaded.edges[0].mapping) is gg.GeomMatrixMapping
    assert json.loads(json.dumps(loaded.to_dict())) == config
    loaded.compile()
    gm.compile()
    loaded.step()
    gm.step()
    for name in ("a", "b"):
        assert np.array_equal(loaded.get_node_state(name)["x"], gm.get_node_state(name)["x"])


def test_a_config_without_a_geometry_edge_has_no_geometry_key():
    """A graph that uses none of this writes the config it always wrote."""
    nearest = nearest_neighbor_mapping(np.linspace(0, 1, N_SOURCE), np.linspace(0, 1, N_TARGET))
    for edge in ({}, dict(mapping=nearest), dict(transform="negate")):
        config = _graph(**edge).to_dict()
        assert all("geometry" not in e for e in config["edges"]), config["edges"]


@pytest.mark.parametrize("bad", ["g", ["a", "g"], {"anchor": "source"}, {"anchor": "b",
                                                                         "field": "g"}, 7],
                         ids=repr)
def test_a_malformed_geometry_in_a_config_is_a_value_error_naming_the_edge(bad):
    spec = _geom().spec.to_dict()
    with pytest.raises(ValueError) as refused:
        GraphManager.from_dict(_config(spec, bad), REGISTRY)
    assert KEY in str(refused.value) and "geometry" in str(refused.value)


def test_a_config_whose_geometry_dependent_mapping_lost_its_geometry_is_refused():
    """G3 at the door a file comes through."""
    with pytest.raises(ValueError, match="reads a moving geometry and none was given"):
        GraphManager.from_dict(_config(_geom().spec.to_dict()), REGISTRY)


# ---------------------------------------------------------------------------
# The reference kind's own compile-time check: positions the geometry's dtype
# cannot resolve
# ---------------------------------------------------------------------------

_F32_EPS = float(np.finfo(np.float32).eps)


def _grid_graph(origin, spacing, *, dtype="float32"):
    """``a`` (a 3-point grid along axis 0) gathered to ``b``'s two points,
    whose positions are ``b.g`` (shape ``(2, 3)``)."""
    mapping = gg.multilinear((origin, 0.0, 0.0), (spacing, 1.0, 1.0), (N_SOURCE, 1, 1),
                             n_points=N_TARGET, mode="consistent")
    cls = _holder_of(dtype)
    return _graph(source=cls, target=cls, mapping=mapping, geometry=("target", "g"))


def test_a_grid_far_from_the_origin_of_a_float32_geometry_is_refused_at_compile():
    """Origin 1e6, spacing 1e-3: a float32 position resolves about 119 cells.
    Refused with the number, and the same grid under a float64 geometry
    compiles."""
    gm = _grid_graph(1.0e6, 1.0e-3)
    issue = _refused_compile(gm, ["a float32 geometry locates a point on axis 0",
                                  "of a cell", "hold the geometry in float64"])
    assert "only to 119 of a cell" in issue, issue
    with gg.x64(True):
        fine = _grid_graph(1.0e6, 1.0e-3, dtype="float64")
        assert not [i for i in fine.validate() if "geometry" in i]
        fine.compile()


def test_a_grid_a_float32_geometry_resolves_only_coarsely_warns_at_compile():
    """Origin 1e4, spacing 1: 1.2e-3 of a cell, between 1/1024 and 1/16: a
    warning naming the edge, and the graph compiles and steps."""
    gm = _grid_graph(1.0e4, 1.0)
    issues = [i for i in gm.validate() if "geometry" in i]
    assert len(issues) == 1 and issues[0].startswith(f"WARNING: edge {KEY}: a float32 geometry")
    with pytest.warns(UserWarning, match="locates a point on axis 0"):
        gm.compile()
    gm.step()
    quiet = _grid_graph(100.0, 1.0)
    assert not [i for i in quiet.validate() if "geometry" in i]


@pytest.mark.parametrize("points, errors, warned", [
    (2**19 + 1, 1, 0),      # exactly 1/16 of a cell at the far end: refused
    (2**19, 0, 1),          # just under: a warning
    (2**13 + 1, 0, 1),      # exactly 1/1024: a warning
    (2**13, 0, 0),          # just under: nothing
])
def test_the_resolution_thresholds_are_a_sixteenth_and_a_1024th_of_a_cell(points, errors,
                                                                          warned):
    """With the origin at zero and unit spacing the far end of an axis of
    ``n`` points is located to ``eps * (n - 1)`` cells, exactly."""
    mapping = gg.multilinear((0.0,), (1.0,), (points,), n_points=2, mode="consistent")
    assert _F32_EPS * (2**19) == 1 / 16 and _F32_EPS * (2**13) == 1 / 1024
    got_errors, got_warnings = mapping.geometry_dtype_problems(np.float32)
    assert (len(got_errors), len(got_warnings)) == (errors, warned), (got_errors, got_warnings)
    assert mapping.geometry_dtype_problems(np.float64) == ([], [])


@pytest.mark.parametrize("origin, spacing", [(0.0, 1e-40), (1e-40, 1.0), (0.0, 1e39),
                                             (1e39, 1.0)])
def test_a_grid_outside_the_normal_range_of_the_geometry_s_dtype_is_refused(origin, spacing):
    """A spacing or an origin that is subnormal, or beyond the largest
    number, in float32 is read as zero or infinity there: refused for a
    float32 geometry, fine for a float64 one.  An origin of exactly zero
    is fine."""
    mapping = gg.multilinear((origin,), (spacing,), (4,), n_points=2, mode="consistent")
    errors, _warnings = mapping.geometry_dtype_problems(np.float32)
    assert len(errors) == 1 and "outside the normal range of a float32 geometry" in errors[0]
    # (float64 holds both numbers; an origin of 1e39 with unit spacing is
    # then refused for its resolution instead, which is right.)
    in_float64, _ = mapping.geometry_dtype_problems(np.float64)
    assert not any("normal range" in e for e in in_float64)
    assert (in_float64 == []) == (origin < 1e30)
    assert gg.multilinear((0.0,), (1.0,), (4,), n_points=2,
                          mode="consistent").geometry_dtype_problems(np.float32) == ([], [])


# ---------------------------------------------------------------------------
# Phase 1: what a group with a geometry edge reports, and what it refuses
# ---------------------------------------------------------------------------


def _ring(norm: str, tolerance: float = 1e-6, **edge):
    """``a <-> b`` in a coupling group; ``a -> b`` is the edge given."""
    gm = GraphManager()
    gm.add_node(Holder("a", 1.0, n=N_SOURCE))
    gm.add_node(Holder("b", 1.0, n=N_TARGET))
    gm.add_edge("a", "b", "x", "u", **edge)
    gm.add_edge("b", "a", "x", "u",
                mapping=matrix_mapping(np.full((N_SOURCE, N_TARGET), 0.05, np.float32)))
    live = {"tolerance": tolerance} if norm == "l2" else {"rtol": tolerance}
    gm.add_coupling_group(["a", "b"], convergence_norm=norm, max_iterations=50,
                          diagnostics=True, **live)
    return gm


@pytest.mark.parametrize("anchor", ["source", "target"])
def test_the_interface_norm_on_a_group_with_a_geometry_edge_is_refused_at_compile(anchor):
    gm = _ring("interface", mapping=_geom(), geometry=(anchor, "g"))
    gg.assert_interface_norm_refused(gm.compile, [KEY])
    errors = [i for i in gm.validate() if i.startswith("ERROR")]
    assert len(errors) == 1 and f"geometry {anchor}.g" in errors[0], errors
    # The controls: the same group with a static mapping compiles under the
    # interface norm, and the geometry edge compiles under the other two.
    _ring("interface", mapping=_static()).compile()
    for norm in ("l2", "mixed"):
        _ring(norm, mapping=_geom(), geometry=(anchor, "g")).compile()


@pytest.mark.parametrize("anchor, norm", [("source", "l2"), ("target", "mixed")])
def test_a_group_with_a_geometry_edge_reports_its_solve_and_no_bound_with_the_reason(anchor,
                                                                                    norm):
    """Diagnostics do not read the moving geometry of a mapping kind other
    than ``multilinear_grid`` in 0.4.0: the report keeps the solve's
    outcome, every bound is NaN, every ``*_usable`` flag False, and
    ``not_usable_reason`` names the edge and the kind.  The same group with a static
    mapping reports its bounds as it always did, and has no such key."""
    gm = _ring(norm, mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    assert gm.coupling_diagnostics() == {}
    gm.step()
    report = gm.coupling_diagnostics()["a+b"]
    gg.assert_not_diagnosed(report, [KEY], "kind")
    assert "'test_geom_matrix'" in report["not_usable_reason"]
    assert bool(report["converged"])
    # The premise: every entry the report withholds would otherwise have
    # said something.  Read the same slots as a group without a geometry
    # edge would be read -- at a loose tolerance every flag is usable and
    # every bound finite; at a tight one the residual is at its floor.
    loose = _ring(norm, 1e-3, mapping=_geom(), geometry=(anchor, "g"))
    loose.compile()
    loose.step()
    gg.assert_not_diagnosed(loose.coupling_diagnostics()["a+b"], [KEY], "kind")
    for graph in (gm, loose):
        graph._committed_geometry_edges = {}          # noqa: SLF001
    unnarrowed, tight = loose.coupling_diagnostics()["a+b"], gm.coupling_diagnostics()["a+b"]
    assert "not_usable_reason" not in unnarrowed
    for flag in ("ratio_usable", "spectral_usable", "gradient_bound_usable"):
        assert unnarrowed[flag] is True, (flag, dict(unnarrowed))
    for bound in ("amplification", "error_estimate", "gradient_error_estimate",
                  "rho_spectral", "spectral_error_bound", "gradient_relative_error_bound"):
        assert np.isfinite(unnarrowed[bound]), (bound, dict(unnarrowed))
    assert tight["precision_limited"] is True, dict(tight)
    static = _ring(norm, mapping=_static())
    static.compile()
    static.step()
    plain = static.coupling_diagnostics()["a+b"]
    assert "not_usable_reason" not in plain
    assert bool(plain["ratio_usable"]) or bool(plain["spectral_usable"]), plain


def test_a_geometry_edge_into_a_group_from_outside_also_leaves_it_undiagnosed():
    """The pass resolves the edge, so the group's report is narrowed too; a
    geometry edge *out of* a group leaves the group's report alone."""
    def build(into: bool):
        gm = GraphManager()
        rows, cols = (N_TARGET, N_SOURCE) if into else (N_SOURCE, N_TARGET)
        gm.add_node(Holder("a", 1.0, n=N_SOURCE, rows=rows, cols=cols))
        gm.add_node(Holder("b", 1.0, n=N_TARGET))
        gm.add_node(Holder("c", 1.0, n=N_TARGET))
        weak = matrix_mapping(np.full((N_TARGET, N_TARGET), 0.05, np.float32))
        gm.add_edge("b", "c", "x", "u", mapping=weak)
        gm.add_edge("c", "b", "x", "u", mapping=weak)
        if into:
            gm.add_edge("a", "b", "x", "u", mapping=_geom(), geometry=("source", "g"),
                        additive=True)
        else:
            gm.add_edge("b", "a", "x", "u", mapping=gg.geom_matrix_mapping(N_SOURCE, N_TARGET),
                        geometry=("target", "g"))
        gm.add_coupling_group(["b", "c"], max_iterations=50)
        gm.compile()
        gm.step()
        return gm.coupling_diagnostics()["b+c"]

    gg.assert_not_diagnosed(build(True), ["a.x->b.u"], "kind")
    assert "not_usable_reason" not in build(False)


# ---------------------------------------------------------------------------
# The dtype rules hold for a geometry written after compile()
# ---------------------------------------------------------------------------
# ``validate()`` asks them of the state ``compile()`` is given.  A state
# write is not a recompile: a geometry compiled as float64 and written as
# float32 used to reach the kernel unasked, and on a grid float32 cannot
# resolve every sample came back 0.0.  They are asked again where a
# program is traced, which a change of dtype always causes.

_ENTRIES = {
    "step": lambda gm: gm.step(),
    "run": lambda gm: gm.run(2),
    "run_scan": lambda gm: gm.run_scan(2),
    "run_scan_with_history": lambda gm: gm.run_scan_with_history(2),
    "resolve_boundary_inputs": lambda gm: gm.resolve_boundary_inputs("b"),
}


def _rewritten(gm, node, dtype):
    state = dict(gm.get_node_state(node))
    state["g"] = jnp.asarray(np.asarray(state["g"]), dtype=dtype)
    gm.set_node_state(node, state)
    assert str(gm.get_node_state(node)["g"].dtype) == dtype


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_a_float32_geometry_written_after_compile_meets_the_resolution_rule(
        anchor, node, entry):
    """Origin 1e6, spacing 1e-3, compiled with a float64 geometry and then
    handed a float32 one by ``set_node_state``: refused with the number
    ``compile()`` gives, at whichever entry point traces first; and the
    float64 geometry written back runs."""
    with gg.x64(True):
        cls = _holder_of("float64")
        if anchor == "target":
            mapping = gg.multilinear((1.0e6, 0.0, 0.0), (1.0e-3, 1.0, 1.0), (N_SOURCE, 1, 1),
                                     n_points=N_TARGET, mode="consistent")
        else:
            mapping = gg.multilinear((1.0e6, 0.0), (1.0e-3, 1.0), (N_TARGET, 1),
                                     n_points=N_SOURCE, mode="conservative")
        gm = GraphManager()
        gm.add_node(cls("a", 1.0, n=N_SOURCE, rows=N_SOURCE, cols=2))
        gm.add_node(cls("b", 1.0, n=N_TARGET))
        gm.add_edge("a", "b", "x", "u", mapping=mapping, geometry=(anchor, "g"))
        gm.compile()
        gm.step()
        _rewritten(gm, node, "float32")
        with pytest.raises(ValueError) as refused:
            _ENTRIES[entry](gm)
        message = str(refused.value)
        assert f"edge {KEY}: a float32 geometry locates a point on axis 0" in message
        assert "only to 119 of a cell" in message and "'g' has dtype float32" in message
        _rewritten(gm, node, "float64")
        _ENTRIES[entry](gm)


def test_a_float32_geometry_the_grid_resolves_is_accepted_after_compile():
    """The rule is about resolution, not about a changed dtype: on a grid
    at the origin the float32 write retraces and steps."""
    with gg.x64(True):
        gm = _grid_graph(0.0, 1.0, dtype="float64")
        gm.compile()
        gm.step()
        _rewritten(gm, "b", "float32")
        gm.step()
        assert str(gm.get_node_state("b")["g"].dtype) == "float32"


@pytest.mark.parametrize("dtype", ["int32", "uint32", "bool", "float16", "bfloat16"])
@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g7_holds_for_a_geometry_written_after_compile(anchor, node, dtype):
    """``compile()`` refuses a geometry that is not float32 or float64
    (G7); so does the first step after a write that makes it one."""
    gm = _graph(mapping=_geom(), geometry=(anchor, "g"))
    gm.compile()
    gm.step()
    _rewritten(gm, node, dtype)
    with pytest.raises(TypeError, match=f"edge {KEY}: its geometry field 'g' now has "
                                        f"dtype {dtype}"):
        gm.step()


def test_the_kernel_refuses_a_geometry_dtype_that_cannot_resolve_its_grid_when_called_directly():
    """Outside any graph: ``apply`` with positions in a dtype the grid's
    coordinates are too coarse in raises instead of returning samples."""
    mapping = gg.multilinear((1.0e6,), (1.0e-3,), (6,), n_points=2, mode="consistent")
    field = jnp.arange(6, dtype=jnp.float32)
    points = np.asarray([[1.0e6 + 0.0013], [1.0e6 + 0.0027]])
    with pytest.raises(ValueError, match="multilinear_grid: a float32 geometry locates a "
                                         "point on axis 0 .* only to 119 of a cell"):
        mapping.apply(field, None, jnp.asarray(points, jnp.float32))
    with gg.x64(True):
        got = mapping.apply(field, None, jnp.asarray(points, jnp.float64))
        np.testing.assert_allclose(np.asarray(got), [1.3, 2.7], rtol=1e-6)


def _written_ring(anchor, *, subcycled, iteration_mode):
    """``a -> b`` through the geometry edge and ``b -> a`` through a static
    one, in a coupling group; *subcycled* halves ``b``'s timestep, so its
    inputs are interpolated between two iterates."""
    gm = GraphManager()
    gm.add_node(Holder("a", 1.0, n=N_SOURCE))
    gm.add_node(Holder("b", 0.5 if subcycled else 1.0, n=N_TARGET))
    gm.add_edge("a", "b", "x", "u", mapping=_geom(), geometry=(anchor, "g"))
    gm.add_edge("b", "a", "x", "u", mapping=matrix_mapping(
        np.full((N_SOURCE, N_TARGET), 1 / 16, np.float32)))
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-5,
                          subcycling=subcycled, iteration_mode=iteration_mode,
                          **({"boundary_interpolation": "linear"} if subcycled else {}))
    return gm


@pytest.mark.parametrize("iteration_mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("subcycled", [False, True], ids=["one-rate", "sub-cycled"])
@pytest.mark.parametrize("anchor, node", [("source", "a"), ("target", "b")])
def test_g7_holds_after_compile_for_a_geometry_edge_inside_a_coupling_group(
        anchor, node, subcycled, iteration_mode):
    """Every read of a geometry inside a coupled solve asks the same rule,
    the interpolated read of a sub-cycled member's source included."""
    gm = _written_ring(anchor, subcycled=subcycled, iteration_mode=iteration_mode)
    gm.compile()
    gm.step()
    _rewritten(gm, node, "float16")
    with pytest.raises(TypeError, match=f"edge {KEY}: its geometry field 'g' now has "
                                        f"dtype float16"):
        gm.step()
