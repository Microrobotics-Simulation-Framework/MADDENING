"""Hypothesis strategies that build *valid* MADDENING graphs.

The contract of this module is deliberately narrow: every recipe it
draws must build, compile and step without raising and without emitting
a warning the suite treats as an error.  A property that fails on a
recipe from here is therefore a defect in the framework, never a
malformed input -- which is the only way a round-trip property is worth
running.

A draw produces a :class:`GraphRecipe`, not a :class:`GraphManager`:
several properties need *two* graphs that are identical by construction
(a reference rollout and a reloaded one, an interrupted run and an
uninterrupted one), and a recipe is the reproducible, shrinkable,
printable description that :meth:`GraphRecipe.build` turns into as many
identical graphs as a test wants.

What is drawn
-------------
node classes (ball, table, spring, heat, rigid-body-2D), timesteps that
are commensurate so multi-rate graphs are legal, node names including
the awkward-but-legal ones (``"a-b"``, ``"a.b"``, ``"1st"``, a name with
a space, a non-ASCII name), edges with and without a registered
transform, additive edges, edge units, live parameter overrides (a
"calibrated" graph), :class:`ParamSpec` overrides including trainable
mapping weights, external inputs, and interface mappings built by the
real factories (RBF, nearest neighbour, 1-D projection) from either a
node-field point reference or an inlined point set.

What is deliberately *not* drawn
--------------------------------
* **Coupling groups.**  ``GraphManager.to_dict`` has no field for them
  (the USD writer does), so every generated graph with one would fail
  the config round trip for a reason that is a known format gap rather
  than a defect this suite found.
* **Initial-condition parameter overrides.**  ``initial_*`` leaves (and
  ``TableNode.position``) are read by ``initial_state`` only, never by
  ``update``.  Writing one into ``gm.params`` changes what ``to_dict``
  serialises but not the live graph's state, so a config round trip
  legitimately produces a different trajectory.  That asymmetry is by
  design (state belongs to checkpoints); only ``trainable`` dynamics
  constants are perturbed here.
* **Geometry parameters** (``HeatNode.length``, ``grid_points``).  They
  define the point sets a ``MappingSpec`` references by content hash, so
  perturbing one would make ``to_dict`` correctly refuse to write a
  reference that no longer resolves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from hypothesis import strategies as st

from maddening.core.coupling.mapping import (
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.core.transforms import scale
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

# ``scale(f)`` registers ``"scale_<f>"`` on first call and returns the
# cached callable afterwards, so importing this module makes the two
# names the edge strategy draws resolvable by ``add_edge(transform=str)``
# and by the USD / config readers.
scale(2.0)
scale(0.5)

#: ``{type name: class}`` for ``GraphManager.from_dict``.
NODE_REGISTRY: dict[str, type] = {
    "BallNode": BallNode,
    "TableNode": TableNode,
    "SpringDamperNode": SpringDamperNode,
    "HeatNode": HeatNode,
    "RigidBody2DNode": RigidBody2DNode,
}

#: Node names, including every kind of legal-but-awkward one.  ``add_node``
#: rejects only ``""``, ``"/"``, ``"#"`` and ``"->"`` (they delimit
#: checkpoint keys, mapping slots and edge keys), so everything here is a
#: name a user may really have.  ``"a-b"`` and ``"a.b"`` are both in the
#: pool on purpose: they are distinct graph nodes that a naive USD prim
#: name would collapse onto the same ``a_b``.
NODE_NAME_POOL = (
    "rod",
    "node_1",
    "Ball",
    "a-b",
    "a.b",
    "with space",
    "1st",
    "Ünïcode",
    "_leading",
    "co2(aq)",
)

#: Registered transforms that turn an ``(n,)`` source into a scalar.
_SCALARISING = ("extract_first", "extract_last", "extract_second", "extract_second_last")
#: Registered transforms that preserve shape.
_SHAPE_PRESERVING = ("negate", "identity", "scale_2.0", "scale_0.5")


def _f32(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Floats exactly representable in float32.

    Every comparison in this suite is exact equality, and a float64
    literal that is rounded on its way into a ``float32`` node parameter
    would make "the reloaded graph has the same parameters" depend on
    the rounding rather than on the serialiser.
    """
    # Snap the bounds themselves to float32 too: Hypothesis refuses a
    # ``width=32`` range whose endpoint is not exactly representable.
    return st.floats(min_value=float(np.float32(lo)), max_value=float(np.float32(hi)),
                     allow_nan=False, allow_infinity=False, width=32)


# ---------------------------------------------------------------------------
# Node kinds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Port:
    shape: tuple[int, ...]
    units: Optional[str] = None
    additive: bool = False


@dataclass(frozen=True)
class NodeRecipe:
    """One node: its class, name, timestep and constructor arguments."""

    type_name: str
    name: str
    timestep: float
    kwargs: tuple[tuple[str, Any], ...]

    @property
    def params(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.kwargs}

    def build(self):
        return NODE_REGISTRY[self.type_name](
            name=self.name, timestep=self.timestep, **self.params
        )

    # -- interface -------------------------------------------------------

    @property
    def outputs(self) -> dict[str, _Port]:
        """State fields this node can be an edge *source* of."""
        p = self.params
        if self.type_name == "BallNode":
            return {"position": _Port((), "m"), "velocity": _Port((), "m/s")}
        if self.type_name == "TableNode":
            return {"position": _Port((), "m")}
        if self.type_name == "SpringDamperNode":
            return {"position": _Port((), "m"), "velocity": _Port((), "m/s")}
        if self.type_name == "HeatNode":
            return {"temperature": _Port((int(p["n_cells"]),), "K")}
        if self.type_name == "RigidBody2DNode":
            return {"x": _Port((2,), "m"), "v": _Port((2,), "m/s"),
                    "angle": _Port((), "rad"), "omega": _Port((), "rad/s")}
        raise AssertionError(self.type_name)

    @property
    def inputs(self) -> dict[str, _Port]:
        """Declared boundary inputs, with the units the node expects."""
        p = self.params
        if self.type_name == "BallNode":
            return {"table_position": _Port((), "m")}
        if self.type_name == "TableNode":
            return {}
        if self.type_name == "SpringDamperNode":
            return {"anchor_position": _Port((), "m")}
        if self.type_name == "HeatNode":
            n = int(p["n_cells"])
            return {"left_temperature": _Port((), "K"),
                    "right_temperature": _Port((), "K"),
                    "heat_source": _Port((n,), "K/s", additive=True)}
        if self.type_name == "RigidBody2DNode":
            return {"force": _Port((2,), "N", additive=True),
                    "torque": _Port((), "N*m", additive=True)}
        raise AssertionError(self.type_name)

    @property
    def perturbable(self) -> tuple[str, ...]:
        """Trainable dynamics constants -- the leaves a "calibration" may
        move.  Initial conditions and geometry are excluded (module
        docstring)."""
        return {
            "BallNode": ("elasticity", "gravity"),
            "TableNode": (),
            "SpringDamperNode": ("stiffness", "damping", "mass", "rest_length"),
            "HeatNode": ("thermal_diffusivity",),
            "RigidBody2DNode": ("mass", "inertia", "gravity"),
        }[self.type_name]


# Constructor-argument strategies.  Ranges keep every generated graph
# numerically well behaved over the handful of steps the properties run:
# the heat rod stays far inside its explicit-Euler stability limit, the
# spring far below its own, and nothing overflows float32.
_KWARGS: dict[str, st.SearchStrategy] = {
    "BallNode": st.fixed_dictionaries({
        "initial_position": _f32(0.0, 3.0),
        "initial_velocity": _f32(-1.0, 1.0),
        "elasticity": _f32(0.0, 1.0),
        "gravity": _f32(-12.0, -1.0),
    }),
    "TableNode": st.fixed_dictionaries({"position": _f32(-1.0, 1.0)}),
    "SpringDamperNode": st.fixed_dictionaries({
        "stiffness": _f32(1.0, 50.0),
        "damping": _f32(0.0, 2.0),
        "mass": _f32(0.5, 2.0),
        "rest_length": _f32(0.0, 2.0),
        "initial_position": _f32(-1.0, 1.0),
        "initial_velocity": _f32(-1.0, 1.0),
    }),
    "HeatNode": st.fixed_dictionaries({
        "n_cells": st.integers(min_value=3, max_value=7),
        "length": st.sampled_from([1.0, 2.0]),
        "thermal_diffusivity": _f32(1e-3, 1e-2),
        "initial_temperature": _f32(0.0, 5.0),
    }),
    "RigidBody2DNode": st.fixed_dictionaries({
        "mass": _f32(0.5, 2.0),
        "inertia": _f32(0.5, 2.0),
        "gravity": st.tuples(_f32(-1.0, 1.0), _f32(-12.0, -1.0)),
        "initial_x": _f32(-1.0, 1.0),
        "initial_y": _f32(-1.0, 1.0),
        "initial_vx": _f32(-1.0, 1.0),
        "initial_vy": _f32(-1.0, 1.0),
        "initial_angle": _f32(-1.0, 1.0),
        "initial_omega": _f32(-1.0, 1.0),
    }),
}

NODE_KINDS = tuple(_KWARGS)


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MappingRecipe:
    """How to rebuild one interface mapping with the real factories.

    ``source_ref`` / ``target_ref`` are ``(node, field)`` pairs naming a
    node's ``static_data`` entry, or ``None`` for a synthetic point set
    that the factory inlines into the :class:`MappingSpec`.  Both paths
    are serialisable, and both are exercised: the node reference is the
    one ``to_dict`` re-resolves and hash-checks on every write.
    """

    kind: str
    n_source: int
    n_target: int
    source_ref: Optional[tuple[str, str]] = None
    target_ref: Optional[tuple[str, str]] = None
    kernel: str = "gaussian"
    epsilon: float = 1.0
    polynomial: bool = True
    ridge: float = 1e-6
    mode: str = "consistent"

    @staticmethod
    def _points(gm: GraphManager, ref, n: int):
        if ref is None:
            return np.linspace(0.0, 1.0, n, dtype=np.float64)
        node, field = ref
        return gm.get_node(node).static_data[field].value

    def build(self, gm: GraphManager):
        src = self._points(gm, self.source_ref, self.n_source)
        tgt = self._points(gm, self.target_ref, self.n_target)
        src_ref = None if self.source_ref is None else {
            "node": self.source_ref[0], "field": self.source_ref[1]}
        tgt_ref = None if self.target_ref is None else {
            "node": self.target_ref[0], "field": self.target_ref[1]}
        if self.kind == "rbf":
            return rbf_mapping(src, tgt, kernel=self.kernel, epsilon=self.epsilon,
                               polynomial=self.polynomial, ridge=self.ridge,
                               mode=self.mode, source_ref=src_ref, target_ref=tgt_ref)
        if self.kind == "nearest_neighbor":
            return nearest_neighbor_mapping(src, tgt, mode=self.mode,
                                            source_ref=src_ref, target_ref=tgt_ref)
        if self.kind == "projection_1d":
            # Cell boundaries, not cell centres: n cells need n+1 edges.
            return projection_1d_mapping(
                np.linspace(0.0, 1.0, self.n_source + 1, dtype=np.float64),
                np.linspace(0.0, 1.0, self.n_target + 1, dtype=np.float64),
            )
        raise AssertionError(self.kind)


# ---------------------------------------------------------------------------
# Edges, external inputs, overrides
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeRecipe:
    source: str
    target: str
    source_field: str
    target_field: str
    transform: Optional[str] = None
    additive: bool = False
    units: Optional[str] = None
    mapping: Optional[MappingRecipe] = None


@dataclass(frozen=True)
class ExternalInputRecipe:
    target_node: str
    target_field: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SpecOverride:
    """One ``set_param_spec`` call.  ``owner`` is a node name or, for a
    mapped edge's weights, the edge key -- which only exists once the
    edge does, so these are applied last."""

    owner: str
    key: str
    trainable: bool = True
    bounds: tuple[Optional[float], Optional[float]] = (None, None)
    transform: Optional[str] = None
    description: str = ""
    units: str = ""

    @property
    def spec(self) -> ParamSpec:
        return ParamSpec(trainable=self.trainable, bounds=self.bounds,
                         transform=self.transform, description=self.description,
                         units=self.units)


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphRecipe:
    """Everything needed to build one graph, as many times as wanted."""

    nodes: tuple[NodeRecipe, ...]
    edges: tuple[EdgeRecipe, ...] = ()
    external_inputs: tuple[ExternalInputRecipe, ...] = ()
    #: ``(node, key, factor)`` -- the live value is multiplied by
    #: ``factor``, which is what a calibration would have left behind.
    param_overrides: tuple[tuple[str, str, float], ...] = ()
    spec_overrides: tuple[SpecOverride, ...] = ()
    #: Scales ``params["mappings"][key]["H"]`` after compile, standing in
    #: for weights moved by ``sysid``.  ``1.0`` leaves them alone.
    mapping_weight_scale: float = 1.0

    # -- introspection ---------------------------------------------------

    @property
    def registry(self) -> dict[str, type]:
        return {n.type_name: NODE_REGISTRY[n.type_name] for n in self.nodes}

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)

    @property
    def has_mapping(self) -> bool:
        return any(e.mapping is not None for e in self.edges)

    # -- construction ----------------------------------------------------

    def build(self) -> GraphManager:
        """A compiled ``GraphManager``.  Deterministic: two calls give two
        graphs that are equal leaf for leaf."""
        gm = GraphManager()
        for node in self.nodes:
            gm.add_node(node.build())
        for edge in self.edges:
            gm.add_edge(
                source=edge.source,
                target=edge.target,
                source_field=edge.source_field,
                target_field=edge.target_field,
                transform=edge.transform,
                additive=edge.additive,
                source_units=edge.units,
                target_units=edge.units,
                mapping=None if edge.mapping is None else edge.mapping.build(gm),
            )
        for ext in self.external_inputs:
            gm.add_external_input(ext.target_node, ext.target_field, shape=ext.shape)
        gm.compile()
        for node_name, key, factor in self.param_overrides:
            leaf = gm.params["nodes"][node_name][key]
            gm.params["nodes"][node_name][key] = (leaf * factor).astype(leaf.dtype)
        for override in self.spec_overrides:
            gm.set_param_spec(override.owner, override.key, override.spec)
        if self.mapping_weight_scale != 1.0:
            for slot in gm.params["mappings"].values():
                for key, value in slot.items():
                    slot[key] = (value * self.mapping_weight_scale).astype(value.dtype)
        return gm


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------

def _timesteps(draw, count: int) -> list[float]:
    """Commensurate timesteps, so a multi-rate graph has an exact GCD."""
    base = draw(st.sampled_from([0.01, 0.005, 0.02]))
    return [base * draw(st.sampled_from([1, 1, 1, 2, 4])) for _ in range(count)]


def _grid_ref(node: NodeRecipe, field: str) -> Optional[tuple[str, str]]:
    """The ``static_data`` point set matching ``field``, when there is one.

    Only :class:`HeatNode` publishes one (``grid_x``, one coordinate per
    cell), and only its cell-valued fields line up with it.
    """
    if node.type_name == "HeatNode" and field in ("temperature", "heat_source"):
        return (node.name, "grid_x")
    return None


@st.composite
def _mapping(draw, source: NodeRecipe, source_field: str,
             target: NodeRecipe, target_field: str) -> MappingRecipe:
    n_source = source.outputs[source_field].shape[0]
    n_target = target.inputs[target_field].shape[0]
    kind = draw(st.sampled_from(["rbf", "nearest_neighbor", "projection_1d"]))
    if kind == "projection_1d":
        # Built from cell boundaries, which no node publishes; always
        # inlined, and always conservative.
        return MappingRecipe("projection_1d", n_source, n_target)
    src_ref = _grid_ref(source, source_field)
    tgt_ref = _grid_ref(target, target_field)
    return MappingRecipe(
        kind,
        n_source,
        n_target,
        source_ref=src_ref if draw(st.booleans()) else None,
        target_ref=tgt_ref if draw(st.booleans()) else None,
        kernel=draw(st.sampled_from(["gaussian", "multiquadric"])),
        epsilon=draw(st.sampled_from([1.0, 2.0, 4.0])),
        polynomial=draw(st.booleans()),
        mode=draw(st.sampled_from(["consistent", "conservative"])),
    )


@st.composite
def _edges(draw, nodes: tuple[NodeRecipe, ...], *, allow_mappings: bool,
           require_mapping: bool) -> tuple[EdgeRecipe, ...]:
    """Edges over ``nodes``, at most one replacive edge per target field."""
    candidates: list[tuple[NodeRecipe, str, NodeRecipe, str, str]] = []
    for src in nodes:
        for tgt in nodes:
            if src.name == tgt.name:
                continue
            for sf, sp in src.outputs.items():
                for tf, tp in tgt.inputs.items():
                    if sp.shape == tp.shape:
                        candidates.append((src, sf, tgt, tf, "direct"))
                    elif sp.shape and not tp.shape:
                        candidates.append((src, sf, tgt, tf, "scalarise"))
                    if allow_mappings and sp.shape and tp.shape:
                        candidates.append((src, sf, tgt, tf, "mapped"))
    if not candidates:
        return ()
    mapped = [c for c in candidates if c[4] == "mapped"]
    if require_mapping and not mapped:
        return ()

    chosen: list[tuple] = []
    if require_mapping:
        chosen.append(draw(st.sampled_from(mapped)))
    n_extra = draw(st.integers(min_value=0 if chosen else 1, max_value=3))
    for _ in range(n_extra):
        chosen.append(draw(st.sampled_from(candidates)))

    edges: list[EdgeRecipe] = []
    used: dict[tuple[str, str], int] = {}
    for src, sf, tgt, tf, how in chosen:
        port = tgt.inputs[tf]
        slot = (tgt.name, tf)
        additive = port.additive and draw(st.booleans())
        # A second edge onto the same field is only well defined when both
        # are additive; otherwise the later one silently wins.
        if used.get(slot):
            if not (port.additive and all(
                    e.additive for e in edges
                    if (e.target, e.target_field) == slot)):
                continue
            additive = True
        if used.get(slot, 0) >= 2:
            continue
        mapping = None
        transform = None
        if how == "scalarise":
            pool = _SCALARISING if src.outputs[sf].shape[0] >= 2 else ("extract_first",)
            transform = draw(st.sampled_from(list(pool)))
        elif how == "mapped":
            mapping = draw(_mapping(src, sf, tgt, tf))
        else:
            transform = draw(st.sampled_from([None, *_SHAPE_PRESERVING]))
        # Units are only ever declared as the ones the target expects: a
        # mismatch is a documented ``UnitMismatchWarning``, i.e. an invalid
        # input, and this generator makes only valid ones.
        units = port.units if draw(st.booleans()) else None
        edges.append(EdgeRecipe(src.name, tgt.name, sf, tf, transform=transform,
                                additive=additive, units=units, mapping=mapping))
        used[slot] = used.get(slot, 0) + 1
    return tuple(edges)


@st.composite
def _external_inputs(draw, nodes, edges) -> tuple[ExternalInputRecipe, ...]:
    taken = {(e.target, e.target_field) for e in edges}
    free = [(n, f, p) for n in nodes for f, p in n.inputs.items()
            if (n.name, f) not in taken]
    if not free or not draw(st.booleans()):
        return ()
    node, field, port = draw(st.sampled_from(free))
    return (ExternalInputRecipe(node.name, field, port.shape),)


@st.composite
def _param_overrides(draw, nodes) -> tuple[tuple[str, str, float], ...]:
    leaves = [(n.name, k) for n in nodes for k in n.perturbable]
    if not leaves:
        return ()
    picked = draw(st.lists(st.sampled_from(leaves), max_size=3, unique=True))
    return tuple((name, key, draw(st.sampled_from([0.5, 0.75, 1.25, 2.0])))
                 for name, key in picked)


def _edge_key(edge: EdgeRecipe, ordinal: int) -> str:
    base = f"{edge.source}.{edge.source_field}->{edge.target}.{edge.target_field}"
    return base if not ordinal else f"{base}#{ordinal}"


@st.composite
def _spec_overrides(draw, nodes, edges) -> tuple[SpecOverride, ...]:
    owners: list[tuple[str, str]] = [
        (n.name, k) for n in nodes for k in n.perturbable
    ]
    seen: dict[str, int] = {}
    for edge in edges:
        if edge.mapping is None:
            continue
        base = f"{edge.source}.{edge.source_field}->{edge.target}.{edge.target_field}"
        owners.append((_edge_key(edge, seen.get(base, 0)), "H"))
        seen[base] = seen.get(base, 0) + 1
    if not owners:
        return ()
    picked = draw(st.lists(st.sampled_from(owners), max_size=3, unique=True))
    out = []
    for owner, key in picked:
        trainable = draw(st.booleans())
        bounded = draw(st.booleans())
        # Bounds are metadata, not a constraint the graph enforces, but a
        # spec that cannot be constructed is not a valid input: keep
        # ``lo < hi`` and keep ``log`` lower-bounded only.
        bounds = (0.0, None) if bounded else (None, None)
        transform = "log" if bounded and draw(st.booleans()) else None
        out.append(SpecOverride(owner, key, trainable=trainable, bounds=bounds,
                                transform=transform,
                                description=draw(st.sampled_from(["", "learned"])),
                                units=draw(st.sampled_from(["", "SI"]))))
    return tuple(out)


@st.composite
def graph_recipes(
    draw,
    *,
    min_nodes: int = 2,
    max_nodes: int = 4,
    kinds: tuple[str, ...] = NODE_KINDS,
    allow_mappings: bool = True,
    require_mapping: bool = False,
    train_mapping_weights: bool = False,
) -> GraphRecipe:
    """Valid :class:`GraphRecipe`\\ s.

    Parameters
    ----------
    min_nodes, max_nodes : int
        Graph size.  Two to four by default: JAX compilation dominates
        the runtime of every property here, so a handful of structurally
        varied small graphs buys far more than a few large ones.
    kinds : tuple of str
        Node classes to draw from.
    allow_mappings, require_mapping : bool
        Whether an edge may / must carry an interface mapping.
    train_mapping_weights : bool
        Scale ``params["mappings"]`` away from what the ``MappingSpec``
        rebuilds, standing in for weights moved by ``sysid``.  Implies
        ``require_mapping``.
    """
    require_mapping = require_mapping or train_mapping_weights
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    names = draw(st.lists(st.sampled_from(NODE_NAME_POOL),
                          min_size=n, max_size=n, unique=True))
    if require_mapping:
        # A mapped edge needs an array-valued source and an array-valued
        # target interface; guarantee two nodes that have them.
        kinds_with_arrays = tuple(k for k in kinds if k in ("HeatNode", "RigidBody2DNode"))
        if not kinds_with_arrays:
            kinds_with_arrays = ("HeatNode",)
        picked = [draw(st.sampled_from(kinds_with_arrays)) for _ in range(2)]
        picked += [draw(st.sampled_from(kinds)) for _ in range(n - 2)]
    else:
        picked = [draw(st.sampled_from(kinds)) for _ in range(n)]
    steps = _timesteps(draw, n)
    nodes = tuple(
        NodeRecipe(kind, name, dt, tuple(draw(_KWARGS[kind]).items()))
        for kind, name, dt in zip(picked, names, steps)
    )
    edges = draw(_edges(nodes, allow_mappings=allow_mappings,
                        require_mapping=require_mapping))
    return GraphRecipe(
        nodes=nodes,
        edges=edges,
        external_inputs=draw(_external_inputs(nodes, edges)),
        param_overrides=draw(_param_overrides(nodes)),
        spec_overrides=draw(_spec_overrides(nodes, edges)),
        mapping_weight_scale=(
            draw(st.sampled_from([0.5, 1.5, 3.0])) if train_mapping_weights else 1.0
        ),
    )
