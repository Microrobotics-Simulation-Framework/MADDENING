"""Coupling groups that hold a floating field with no entries, and their twins without it.

A member of a coupling group may hold a floating field of size zero (an
empty contact set, a list with none this configuration), and an edge may
deliver one.  Such a field is not read: it has no magnitude, so it carries
no norm, no weight, no floor term and no gain
(``maddening.core.coupling.acceleration._has_entries``).  What that means
for a user is one sentence, and it is what the tests built on this module
assert: **a group with such a field steps, and returns and reports what the
same group without the field does.**

:func:`build` makes one graph of :data:`SHAPES` (where the field with no
entries sits) or, with ``empty=False``, its *twin*: the same nodes, edges
and group with the field ``none``, the ports that read it and the edges
that carry it left out.  :func:`outcome` steps a graph and returns
everything the comparison reads; :func:`differences` compares two of them.
"""

from __future__ import annotations

import contextlib
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

#: Where the field with no entries sits.
SHAPES = {
    "unread": "a member holds it and no edge reads it",
    "source": "an edge reads it, and so delivers no entries",
    "both_ways": "each member holds one and reads the other's",
    "read_twice": "two edges read it",
    "transformed": "an edge reads it through a transform",
    "only_field": "it is the only floating field of one member",
    "unread_mapped": "no edge reads it, and the edge beside it carries a mapping",
    "source_mapped": "an edge reads it, and the edge beside it carries a mapping",
    "only_edges": "the members exchange nothing else: every internal edge delivers no entries",
    "no_entries": "it is every floating field the group has",
}
#: The two shapes whose group has nothing to iterate on: no entry crosses
#: an edge, or no member holds one.
DEGENERATE = ("only_edges", "no_entries")
#: The shape whose twin each shape's twin is: without the field, the first
#: five are one graph and the two with a mapped edge another.
TWIN = {
    "unread": "unread", "source": "unread", "both_ways": "unread", "read_twice": "unread",
    "transformed": "unread", "only_field": "only_field",
    "unread_mapped": "unread_mapped", "source_mapped": "unread_mapped",
    "only_edges": "only_edges", "no_entries": "no_entries",
}
NORMS = ("l2", "mixed", "interface")
ACCELERATIONS = ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")
MODES = ("gauss-seidel", "jacobi")

#: The ports that read the field with no entries, by shape.
_PORTS = {"source": ("e",), "both_ways": ("e",), "read_twice": ("e", "e2"),
          "transformed": ("e",), "source_mapped": ("e",), "only_edges": ("e",)}


@contextlib.contextmanager
def x64(on: bool):
    """``jax_enable_x64`` set to *on* for the block, then put back."""
    before = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", bool(on))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", before)


class Member(SimulationNode):
    """``x <- x_pre / 2 + u / 4 + u**2 / 50 + c`` on two entries.

    Mildly nonlinear, so the gradient bound has a curvature to measure, and
    it reads its own pre-step state, so a gradient with respect to the
    state the step started from is not zero.

    ``holds`` is the dtype of a field ``none`` with no entries the member
    also holds (``None``: no such field); ``ports`` names input ports with
    no entries (of dtype ``port_dtype``), whose values (a sum over none)
    are added to ``x``; ``counted`` adds an int32 port ``k`` the member
    receives and ignores.
    """

    def __init__(self, name, timestep, *, dtype, c, holds=None, ports=(), port_dtype=None,
                 counted=False):
        super().__init__(name, timestep)
        self._dtype = jnp.dtype(dtype)
        self._c = tuple(float(v) for v in c)
        self._holds = None if holds is None else jnp.dtype(holds)
        self._ports = tuple(ports)
        self._port_dtype = jnp.dtype(port_dtype or dtype)
        self._counted = bool(counted)

    def initial_state(self):
        state = {"x": jnp.asarray([0.5, -0.25], self._dtype)}
        if self._holds is not None:
            state["none"] = jnp.zeros((0,), self._holds)
        return state

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(2,), dtype=self._dtype,
                                       default=jnp.zeros((2,), self._dtype))}
        for port in self._ports:
            spec[port] = BoundaryInputSpec(shape=(0,), dtype=self._port_dtype,
                                           default=jnp.zeros((0,), self._port_dtype))
        if self._counted:
            spec["k"] = BoundaryInputSpec(shape=(), dtype=jnp.int32,
                                          default=jnp.zeros((), jnp.int32))
        return spec

    def update(self, state, boundary_inputs, dt, *, params=None):
        d = self._dtype
        u = boundary_inputs.get("u", jnp.zeros((2,), d)).astype(d)
        x = (jnp.asarray(0.5, d) * state["x"] + jnp.asarray(0.25, d) * u
             + jnp.asarray(0.02, d) * u * u + jnp.asarray(self._c, d))
        for port in self._ports:
            x = x + jnp.sum(boundary_inputs[port]).astype(d)
        out = {"x": x.astype(d)}
        if self._holds is not None:
            out["none"] = state["none"]
        return out

    def update_evaluations(self):
        return 1


class Bystander(SimulationNode):
    """A member with a counter ``n`` and, with ``holds``, a field ``none``
    with no entries: then its only floating field.  It reads a member's
    ``x`` and counts the passes that handed it a positive first entry."""

    def __init__(self, name, timestep, *, dtype, holds=None):
        super().__init__(name, timestep)
        self._dtype = jnp.dtype(dtype)
        self._holds = None if holds is None else jnp.dtype(holds)

    def initial_state(self):
        state = {"n": jnp.zeros((), jnp.int32)}
        if self._holds is not None:
            state["none"] = jnp.zeros((0,), self._holds)
        return state

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=self._dtype,
                                       default=jnp.zeros((2,), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        out = {"n": state["n"] + (boundary_inputs["u"][0] > 0).astype(jnp.int32)}
        if self._holds is not None:
            out["none"] = state["none"]
        return out

    def update_evaluations(self):
        return 1


def _double(v):
    return 2.0 * v


def build(shape: str, knobs: dict, *, empty: bool = True, dtype="float32",
          none_dtype=None) -> GraphManager:
    """The compiled graph of *shape* under the group options *knobs*.

    ``empty=False`` is the twin: no field ``none``, no port that reads it
    and no edge that carries it.  ``none_dtype`` is the dtype of the field
    with no entries where it is not the members' own.
    """
    from maddening.core.coupling.mapping import matrix_mapping  # noqa: PLC0415

    assert shape in SHAPES, shape
    held = jnp.dtype(none_dtype or dtype) if empty else None
    reads = _PORTS.get(shape, ()) if empty else ()
    both = shape in ("both_ways", "only_edges")
    only = shape == "only_field"

    gm = GraphManager()
    if shape == "no_entries":
        # Two members that hold a counter and the field; the node that
        # feeds them is outside the group.
        gm.add_node(Bystander("a", 1.0, dtype=dtype, holds=held))
        gm.add_node(Member("b", 1.0, dtype=dtype, c=(0.4, 0.9)))
        gm.add_node(Bystander("c", 1.0, dtype=dtype, holds=held))
        gm.add_edge("b", "a", "x", "u")
        gm.add_edge("b", "c", "x", "u")
        return _grouped(gm, ["a", "c"], knobs)
    gm.add_node(Member("a", 1.0, dtype=dtype, c=(1.0, 0.7),
                       holds=None if only else held, ports=reads if both else (),
                       port_dtype=held, counted=only))
    gm.add_node(Member("b", 1.0, dtype=dtype, c=(0.4, 0.9),
                       holds=held if both else None, ports=reads, port_dtype=held))
    members = ["a", "b"]
    if only:
        gm.add_node(Bystander("c", 1.0, dtype=dtype, holds=held))
        members.append("c")
    if shape == "only_edges":
        pass                    # no entry crosses an edge
    elif shape.endswith("_mapped"):
        H = np.asarray([[0.6, 0.4], [0.2, 0.8]], np.float64)
        gm.add_edge("b", "a", "x", "u")
        gm.add_edge("a", "b", "x", "u", mapping=matrix_mapping(jnp.asarray(H, jnp.dtype(dtype))))
    else:
        gm.add_edge("b", "a", "x", "u")
        gm.add_edge("a", "b", "x", "u")
    if only:
        gm.add_edge("a", "c", "x", "u")
        gm.add_edge("c", "a", "n", "k")
    for port in reads:
        gm.add_edge("a", "b", "none", port,
                    transform=_double if shape == "transformed" else None)
        if both:
            gm.add_edge("b", "a", "none", port)
    return _grouped(gm, members, knobs)


def _grouped(gm: GraphManager, members: list, knobs: dict) -> GraphManager:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "CouplingGroup solver='fori' is deprecated", DeprecationWarning)
        gm.add_coupling_group(members, **knobs)
        gm.compile()
    return gm


def members_of(gm: GraphManager) -> list:
    (group,) = gm._coupling_groups
    return sorted(group.nodes)


def gradient(gm: GraphManager) -> np.ndarray:
    """``d loss / d x`` of the first node that holds an ``x``, at the state
    the graph holds, through one compiled step: the loss is the sum of
    squares of every node's ``x`` after the step."""
    compiled = gm._compiled_step
    assert compiled is not None, "step the graph once first"
    start = gm._state
    holders = [name for name in gm.node_names if "x" in start[name]]

    def loss_of(x_first):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in start.items()}
        state[holders[0]] = {**state[holders[0]], "x": x_first}
        stepped = compiled(state, {})
        return sum(jnp.sum(jnp.square(stepped[name]["x"])) for name in holders)

    return np.asarray(jax.grad(loss_of)(start[holders[0]]["x"]))


def outcome(gm: GraphManager, *, steps: int = 2, with_gradient: bool = True) -> dict:
    """Step *gm* and return what a user can observe of it.

    ``{"state": {node: {field: array}}, "report": {...}, "gradient": array}``
    with the fields ``none`` left out of ``state`` (:func:`none_fields_are_kept`
    checks those).  The gradient is taken at the state the first step left;
    the report is the last step's.
    """
    gm.step()
    grad = gradient(gm) if with_gradient else None
    for _ in range(steps - 1):
        gm.step()
    state = {name: {f: np.asarray(v) for f, v in gm.get_node_state(name).items() if f != "none"}
             for name in gm.node_names}
    return {"state": state, "gradient": grad,
            "report": dict(gm.coupling_diagnostics().get("+".join(members_of(gm)), {}))}


def none_fields_are_kept(gm: GraphManager, dtype) -> bool:
    """Every field ``none`` still has no entries and the dtype it was given."""
    found = [np.asarray(gm.get_node_state(name)["none"]) for name in members_of(gm)
             if "none" in gm.get_node_state(name)]
    return bool(found) and all(v.size == 0 and v.dtype == np.dtype(dtype) for v in found)


def _same(a, b) -> bool:
    if isinstance(a, (bool, str, type(None))) or isinstance(b, (bool, str, type(None))):
        return type(a) is type(b) and a == b
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=True))


def differences(got: dict, twin: dict) -> list:
    """Everything in which two outcomes differ, to the bit: ``[(what, got, twin), ...]``."""
    out = []
    for name in sorted(set(got["state"]) | set(twin["state"])):
        for field in sorted(set(got["state"].get(name, {})) | set(twin["state"].get(name, {}))):
            a = got["state"].get(name, {}).get(field)
            b = twin["state"].get(name, {}).get(field)
            if a is None or b is None or not _same(a, b):
                out.append((f"state {name}.{field}", a, b))
    for key in sorted(set(got["report"]) | set(twin["report"])):
        a, b = got["report"].get(key, "<absent>"), twin["report"].get(key, "<absent>")
        if not _same(a, b):
            out.append((f"report {key}", a, b))
    if (got["gradient"] is None) != (twin["gradient"] is None) or (
            got["gradient"] is not None and not _same(got["gradient"], twin["gradient"])):
        out.append(("gradient", got["gradient"], twin["gradient"]))
    return out
