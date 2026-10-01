"""Generated coupled graphs of synthetic nodes, for the differential tests.

The ``test_differential_*`` modules state each of their oracles over graphs
built here.  The library nodes are physics with opinions -- a spring pins
its own contraction rate, a heat slab its Fourier limit -- so a property
drawn over them explores the corner of configuration space those nodes
happen to occupy.  The synthetic nodes below put every property a
coupling solver can be sensitive to under the strategy's control instead:

* **the spectrum** -- the group's coupling operator is assembled from
  drawn gain matrices and rescaled to a drawn spectral radius, near 1
  included, optionally non-normal (a strongly upper-triangular part) or of
  rank one (a single coupling mode, where the error estimate is exact);
* **non-linearity** -- ``tanh`` on every input;
* **leaf types** -- a wide ``int32`` counter, a ``uint32`` linear
  congruential tag and a ``bool`` flag, all recomputed from the pre-step
  state; a float field nothing reads (``unread``); a clock ``t <- t + dt``;
* **edge kinds** -- state edges and flux edges (``q = 2 x``, a field that
  is not in the producer's state);
* **topology** -- a cycle through the group plus drawn chords, and an
  optional driver upstream and sink downstream of the group.

Every node's ``update`` is ``x <- alpha x_pre + sum_j G_j f(u_j) + b +
beta dt``, so the group's map is affine whenever ``f`` is the identity and
its fixed point is a float64 linear solve
(:func:`exact_fixed_point`).  ``G_j`` and ``b`` are node *parameters*:
they reach the compiled step as traced arguments, so one compiled graph
serves every drawn spectrum of one structure (:func:`params_for`).

Nothing here is a test.  The module is imported by the differential
modules and by nothing in ``src/``.
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from maddening.core.coupling.group import _FIELD_DEFAULTS, _INERT_RULES
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F32 = np.float32
EPS32 = float(np.finfo(np.float32).eps)

#: Leaves a node may carry beside ``x``, all recomputed from the pre-step
#: state (or, for ``clock``, from the pre-step state and ``dt``) and so
#: independent of the coupling iterate.  ``"sign"`` is *not* here: it reads
#: a coupled input, and is opted into explicitly where that is the point.
LEAF_POOL = ("count", "tag", "flag", "unread", "clock")

#: Initial values of the non-float leaves.  ``2**24 + 1`` and
#: ``0xDEADBEEF`` are the values float32 cannot hold, so a leaf that took a
#: round trip through the accelerator's floating vector comes back changed.
COUNT0 = 2**24 + 1
TAG0 = 0xDEADBEEF
TAG_MUL = 1664525
TAG_ADD = 1013904223


class Relay(SimulationNode):
    """``x <- alpha * x_pre + sum_j G_j @ f(u_j) + b + beta * dt``.

    ``f`` is the identity, or ``tanh`` when *nonlinear*.  With *ode* the
    node is instead explicit Euler on ``x' = alpha x + sum_j G_j f(u_j) +
    b`` -- ``x <- x_pre + dt (...)`` -- a consistent time discretisation,
    which is what an adaptive stepper's error estimate assumes (the
    algebraic form is a different map at every ``dt``).  Each input port
    ``u{j}`` takes at most one edge, so an external input can stand in for
    any edge without the additive/replacive question arising (the
    hand-unrolled multi-rate reference relies on that).
    """

    def __init__(self, name, timestep, *, n, n_inputs, alpha=0.0, beta=0.0,
                 nonlinear=False, leaves=(), x0=None, ode=False):
        params = {f"G{j}": jnp.zeros((n, n), jnp.float32) for j in range(n_inputs)}
        params["b"] = jnp.zeros(n, jnp.float32)
        super().__init__(name, timestep, **params)
        self._n = int(n)
        self._k = int(n_inputs)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._nl = bool(nonlinear)
        self._ode = bool(ode)
        self._leaves = tuple(leaves)
        self._x0 = np.zeros(n, F32) if x0 is None else np.asarray(x0, F32)

    def initial_state(self):
        s = {"x": jnp.asarray(self._x0, jnp.float32)}
        if "count" in self._leaves:
            s["count"] = jnp.asarray(COUNT0, jnp.int32)
        if "tag" in self._leaves:
            s["tag"] = jnp.asarray(TAG0, jnp.uint32)
        if "flag" in self._leaves:
            s["flag"] = jnp.asarray(True)
        if "unread" in self._leaves:
            s["unread"] = jnp.asarray(0.25, jnp.float32)
        if "clock" in self._leaves:
            s["clock"] = jnp.asarray(0.0, jnp.float32)
        if "sign" in self._leaves:
            s["sign"] = jnp.asarray(False)
        return s

    def boundary_input_spec(self):
        return {f"u{j}": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                           default=jnp.zeros(self._n, jnp.float32))
                for j in range(self._k)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        dt32 = jnp.asarray(dt, jnp.float32)
        if self._ode:
            f = jnp.float32(self._alpha) * state["x"] + p["b"]
        else:
            f = jnp.float32(self._alpha) * state["x"] + p["b"] + jnp.float32(self._beta) * dt32
        for j in range(self._k):
            u = boundary_inputs.get(f"u{j}", jnp.zeros(self._n, jnp.float32))
            if self._nl:
                u = jnp.tanh(u)
            f = f + p[f"G{j}"] @ u
        x = state["x"] + dt32 * f if self._ode else f
        out = {"x": x.astype(jnp.float32)}
        if "count" in self._leaves:
            out["count"] = state["count"] + jnp.int32(1)
        if "tag" in self._leaves:
            out["tag"] = state["tag"] * jnp.uint32(TAG_MUL) + jnp.uint32(TAG_ADD)
        if "flag" in self._leaves:
            out["flag"] = jnp.logical_not(state["flag"])
        if "unread" in self._leaves:
            out["unread"] = jnp.float32(0.5) * state["unread"] + jnp.float32(1.0)
        if "clock" in self._leaves:
            out["clock"] = state["clock"] + jnp.asarray(dt, jnp.float32)
        if "sign" in self._leaves:
            u0 = boundary_inputs.get("u0", jnp.zeros(self._n, jnp.float32))
            out["sign"] = jnp.sum(u0) > 0
        return out

    def update_evaluations(self):
        return 1


class FluxRelay(Relay):
    """A :class:`Relay` that also produces the boundary flux ``q = 2 x``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": jnp.float32(2.0) * state["x"]}


class SubStepped(SimulationNode):
    """*inner* advanced ``d`` times per step at ``dt / d``, inputs held.

    The uniform-rate reference for a sub-cycled coupling group under
    ``boundary_interpolation="constant"``: a sub-cycled member takes
    ``d`` sub-steps of its own timestep per coupling pass, each reading
    the in-pass state of its sources, which is exactly this node's update
    at the group's macro timestep -- written out by hand, with no
    sub-cycling machinery under it.
    """

    def __init__(self, inner: Relay, d: int):
        super().__init__(inner.name, inner.delta_t * d, **inner.params)
        self._inner = inner
        self._d = int(d)

    def initial_state(self):
        return self._inner.initial_state()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def update(self, state, boundary_inputs, dt, *, params=None):
        sub = jnp.asarray(dt, jnp.float32) / jnp.float32(self._d)
        for _ in range(self._d):
            state = self._inner.update(state, boundary_inputs, sub, params=params)
        return state

    def update_evaluations(self):
        return self._d


# ---------------------------------------------------------------------------
# Structure: what is static in the compiled step
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NodeDef:
    """One synthetic node: everything about it the compiled step bakes in."""

    name: str
    n_inputs: int
    timestep: float = 1.0
    alpha: float = 0.0
    beta: float = 0.0
    nonlinear: bool = False
    leaves: tuple = ()
    flux: bool = False
    ode: bool = False


@dataclasses.dataclass(frozen=True)
class EdgeDef:
    """``src.field -> dst.u{port}``; ``field`` is ``"x"`` or the flux ``"q"``."""

    src: str
    dst: str
    port: int
    field: str = "x"


@dataclasses.dataclass(frozen=True)
class GraphDef:
    """A graph's structure: nodes, edges, which of them form the group."""

    n: int
    nodes: tuple
    edges: tuple
    group_nodes: tuple

    def node(self, name: str) -> NodeDef:
        return next(nd for nd in self.nodes if nd.name == name)

    @property
    def key(self) -> str:
        return "+".join(sorted(self.group_nodes))

    @property
    def internal_edges(self) -> tuple:
        g = set(self.group_nodes)
        return tuple(e for e in self.edges if e.src in g and e.dst in g)

    def with_timesteps(self, timesteps: dict) -> "GraphDef":
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, timestep=timesteps.get(nd.name, nd.timestep))
            for nd in self.nodes))

    def as_ode(self, alpha: float = -1.0) -> "GraphDef":
        """Every node as explicit Euler (``ode=True``) with decay *alpha*."""
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, ode=True, alpha=alpha) for nd in self.nodes))

    def without_leaves(self, drop: tuple) -> "GraphDef":
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, leaves=tuple(lf for lf in nd.leaves if lf not in drop))
            for nd in self.nodes))

    def with_leaves(self, leaves: tuple, names=None) -> "GraphDef":
        names = set(self.group_nodes if names is None else names)
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, leaves=tuple(leaves)) if nd.name in names else nd
            for nd in self.nodes))


def make_node(nd: NodeDef, n: int, x0=None) -> Relay:
    cls = FluxRelay if nd.flux else Relay
    return cls(nd.name, nd.timestep, n=n, n_inputs=nd.n_inputs, alpha=nd.alpha,
               beta=nd.beta, nonlinear=nd.nonlinear, leaves=nd.leaves, x0=x0, ode=nd.ode)


def live_knobs(knobs: dict) -> dict:
    """*knobs* with every setting the rest of the configuration ignores reset.

    ``CouplingGroup`` warns about a deliberately set knob its configuration
    never reads, which ``filterwarnings = ["error"]`` makes fatal; the
    rules come from the library's own table so a gate added there needs no
    second edit here.  Only inert knobs are touched, and only back to
    their declared default, so this cannot change what is solved.
    """
    view = SimpleNamespace(**{**_FIELD_DEFAULTS, **knobs})
    for rule in _INERT_RULES:
        if rule.live(view):
            continue
        for name in rule.fields:
            setattr(view, name, _FIELD_DEFAULTS[name])
    return {name: getattr(view, name) for name in knobs
            if name in _FIELD_DEFAULTS}


def build_graph(gdef: GraphDef, group: dict, *, x0: Optional[dict] = None,
                substep: Optional[dict] = None, compile: bool = True) -> GraphManager:
    """A :class:`GraphManager` for *gdef* with one coupling group.

    *group* is the ``CouplingGroup`` configuration; knobs it leaves inert
    are dropped (:func:`live_knobs`).  *substep* maps a node name to a
    sub-step count: that node is wrapped in :class:`SubStepped` (the
    sub-cycling reference).  Parameters are the zeros :class:`Relay`
    seeds; the values come in through :func:`params_for` at run time.
    """
    gm = GraphManager()
    substep = substep or {}
    for nd in gdef.nodes:
        node = make_node(nd, gdef.n, None if x0 is None else x0.get(nd.name))
        if nd.name in substep:
            node = SubStepped(node, substep[nd.name])
        gm.add_node(node)
    for e in gdef.edges:
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "CouplingGroup solver='fori' is deprecated", DeprecationWarning)
        gm.add_coupling_group(list(gdef.group_nodes), **live_knobs(group))
    if compile:
        with warnings.catch_warnings():
            # The multi-rate INFO notice is a UserWarning on purpose; the
            # graphs here are multi-rate on purpose.
            warnings.filterwarnings("ignore", ".*multi-rate.*")
            gm.compile()
    return gm


# ---------------------------------------------------------------------------
# Values: what reaches the compiled step as an argument
# ---------------------------------------------------------------------------


def coupling_matrix(gdef: GraphDef, values: dict) -> np.ndarray:
    """The group's coupling operator ``M`` (float64), block ``(dst, src)``.

    ``x = M x + c`` is the group's fixed-point equation when every node is
    linear: block ``(i, j)`` is the gain node ``i`` applies to node ``j``'s
    output (twice it across a flux edge, ``q = 2 x``).  Its spectral radius
    is the Jacobi rate.
    """
    names = list(gdef.group_nodes)
    n = gdef.n
    M = np.zeros((n * len(names), n * len(names)))
    for e in gdef.internal_edges:
        G = np.asarray(values[e.dst]["G"][e.port], np.float32).astype(np.float64)
        fac = 2.0 if e.field == "q" else 1.0
        i, j = names.index(e.dst), names.index(e.src)
        M[i * n:(i + 1) * n, j * n:(j + 1) * n] += fac * G
    return M


def draw_values(rng: np.random.Generator, gdef: GraphDef, rho: float, *,
                nonnormal: bool = False, rank_one: bool = False,
                bias_scale: float = 1.0) -> dict:
    """Gains, biases and initial states, the group rescaled to rate *rho*.

    By default the internal gains are drawn, assembled into
    :func:`coupling_matrix` and multiplied by one common factor so its
    spectral radius -- the Jacobi rate -- is ``|rho|``.  *nonnormal* adds
    a strongly upper-triangular part to every gain.

    *rank_one* asks for a group with a **single coupling mode**: the
    internal edges must form a pure cycle, the gain on the edge into the
    cycle's first node is drawn of rank one, and it is scaled so the cycle
    product ``P`` (the gain once round the loop) has trace ``rho``.  ``P``
    then has exactly one non-zero eigenvalue, ``rho`` (negative allowed),
    and a Gauss-Seidel sweep in the cycle's own order *is* ``x <- P x +
    c`` on that node -- one mode, so the residual sequence is a clean
    geometric decay and the error estimate has no excuse.
    """
    n = gdef.n
    values: dict = {}
    for nd in gdef.nodes:
        Gs = []
        for _ in range(nd.n_inputs):
            G = rng.normal(size=(n, n))
            if nonnormal and n > 1:
                G = np.triu(G) + np.triu(rng.normal(size=(n, n)) * 4.0, 1)
            Gs.append(G)
        values[nd.name] = {"G": Gs, "b": rng.normal(size=n) * bias_scale,
                           "x0": rng.normal(size=n)}
    names = list(gdef.group_nodes)
    group = set(names)
    internal = gdef.internal_edges
    if rank_one:
        cycle = [next(e for e in internal if e.dst == nm) for nm in names]
        assert len(internal) == len(names) and all(
            sum(1 for e in internal if e.dst == nm) == 1 for nm in names), (
            "rank_one needs the group's internal edges to be a pure cycle")
        first = cycle[0]
        u, v = rng.normal(size=n), rng.normal(size=n)
        if nonnormal:
            v = v + 3.0 * rng.normal(size=n)
        values[first.dst]["G"][first.port] = np.outer(u, v)
        # P = G_in(first) * G_in(prev) * ... once round the loop.
        P = np.eye(n)
        node = first.dst
        for _ in range(len(names)):
            e = next(e for e in internal if e.dst == node)
            fac = 2.0 if e.field == "q" else 1.0
            P = P @ (fac * np.asarray(values[node]["G"][e.port], np.float64))
            node = e.src
        tr = float(np.trace(P))
        values[first.dst]["G"][first.port] = (
            values[first.dst]["G"][first.port] * (rho / tr if tr != 0.0 else 0.0))
    else:
        M = coupling_matrix(gdef, {k: {**v_, "G": [np.asarray(g, np.float64) for g in v_["G"]]}
                                   for k, v_ in values.items()})
        r = float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0
        scale = abs(rho) / r if r > 0 else 0.0
        for e in internal:
            values[e.dst]["G"][e.port] = np.asarray(values[e.dst]["G"][e.port]) * scale
    for nd in gdef.nodes:
        if nd.name not in group:
            # Outside gains stay O(1/n): a driver or sink is not part of
            # the spectrum and must not dominate it.
            values[nd.name]["G"] = [np.asarray(G) / max(n, 1) for G in values[nd.name]["G"]]
        values[nd.name]["G"] = [np.asarray(G, F32) for G in values[nd.name]["G"]]
        values[nd.name]["b"] = np.asarray(values[nd.name]["b"], F32)
        values[nd.name]["x0"] = np.asarray(values[nd.name]["x0"], F32)
    return values


def params_for(gm: GraphManager, values: dict) -> dict:
    """``gm.params`` with every node's gains and bias replaced by *values*."""
    base = gm.params
    nodes = {name: dict(p) for name, p in base["nodes"].items()}
    for name, v in values.items():
        if name not in nodes:
            continue
        for j, G in enumerate(v["G"]):
            nodes[name][f"G{j}"] = jnp.asarray(G, jnp.float32)
        nodes[name]["b"] = jnp.asarray(v["b"], jnp.float32)
    return {**base, "nodes": nodes}


def recover(gm: GraphManager) -> None:
    """Put *gm* back after a transform left tracers in it, quietly.

    ``jax.grad`` of a loss that calls ``run_scan`` leaves the traced final
    state in the graph.  Every entry point but ``reset_state`` puts it
    back first (with a ``RuntimeWarning``); ``reset_state`` does not, and
    raises on a group with a predictor -- a strict xfail in
    ``test_differential_coupling_solvers.py``.  So go through an entry
    point that does.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
        gm.coupling_diagnostics()


def set_initial(gm: GraphManager, values: dict) -> None:
    """Reset *gm* (state and coupling seeds) and write every node's ``x0``."""
    recover(gm)
    gm.reset_state()
    for name, v in values.items():
        if name in gm.node_names:
            s = dict(gm.get_node_state(name))
            s["x"] = jnp.asarray(v["x0"], jnp.float32)
            gm.set_node_state(name, s)


def snapshot(gm: GraphManager) -> dict:
    """Every node's state as numpy arrays (no ``_meta``)."""
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()}
            for n in gm.node_names}


def group_meta(gm: GraphManager, key: str) -> dict:
    """The ``_meta`` slots group *key* owns, as numpy arrays."""
    prefix = f"coupling_{key}_"
    return {k[len(prefix):]: np.asarray(v)
            for k, v in gm._state.get("_meta", {}).items()  # noqa: SLF001
            if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def bitwise_differences(a: dict, b: dict) -> list[str]:
    """``node.field`` (or slot) names whose bytes differ between *a* and *b*."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) or isinstance(y, dict):
            out += [f"{k}.{f}" for f in bitwise_differences(x or {}, y or {})]
        elif x is None or y is None or np.asarray(x).tobytes() != np.asarray(y).tobytes() \
                or np.asarray(x).dtype != np.asarray(y).dtype:
            out.append(str(k))
    return out


def relative_gap(a: dict, b: dict, nodes=None, floor: float = 0.0) -> float:
    """Largest ``max|x - y| / max(|x|, |y|, floor)`` over float fields.

    Non-float fields must match exactly; a mismatch is ``inf``.
    """
    worst = 0.0
    for node in (nodes if nodes is not None else a):
        for f in a[node]:
            x = np.asarray(a[node][f])
            y = np.asarray(b[node][f])
            if not np.issubdtype(x.dtype, np.floating):
                if x.tobytes() != y.tobytes():
                    return float("inf")
                continue
            x = x.astype(np.float64)
            y = y.astype(np.float64)
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                if x.tobytes() != y.tobytes():
                    return float("inf")
                continue
            scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), floor)
            if scale == 0.0:
                continue
            worst = max(worst, float(np.max(np.abs(x - y))) / scale)
    return worst


def expected_leaves(nd: NodeDef, updates: int, clock: Optional[float] = None) -> dict:
    """The closed-form non-float leaves after *updates* calls of ``update``."""
    out = {}
    if "count" in nd.leaves:
        out["count"] = np.int32(COUNT0 + updates)
    if "tag" in nd.leaves:
        t = TAG0
        for _ in range(updates):
            t = (t * TAG_MUL + TAG_ADD) & 0xFFFFFFFF
        out["tag"] = np.uint32(t)
    if "flag" in nd.leaves:
        out["flag"] = np.bool_(updates % 2 == 0)
    return out


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def graph_defs(draw, *, min_group: int = 2, max_group: int = 4,
               n: Optional[int] = None, allow_flux: bool = True,
               allow_nonlinear: bool = True, allow_outside: bool = True,
               leaves: Optional[tuple] = None, chords: bool = True):
    """A group of 2-4 synthetic nodes in a cycle, plus chords and neighbours.

    The cycle visits the group in a drawn order (which decides the
    Gauss-Seidel sweep's dependence pattern); each ordered pair not on it
    gets a chord with a drawn probability.  A flux edge replaces one
    internal edge; a driver feeds the group's first node and a sink reads
    its last.
    """
    m = draw(st.integers(min_group, max_group))
    n = n if n is not None else draw(st.integers(1, 3))
    names = [f"g{i}" for i in range(m)]
    order = draw(st.permutations(range(m)))
    edges: list = []
    ports = {nm: 0 for nm in names}

    def add(src, dst, field="x"):
        edges.append(EdgeDef(src, dst, ports[dst], field))
        ports[dst] += 1

    for i in range(m):
        add(names[order[i]], names[order[(i + 1) % m]])
    if chords:
        for i in range(m):
            for j in range(m):
                if i != j and not any(e.src == names[i] and e.dst == names[j] for e in edges) \
                        and draw(st.booleans()) and draw(st.booleans()):
                    add(names[i], names[j])
    flux_node = None
    if allow_flux and draw(st.booleans()) and draw(st.booleans()):
        k = draw(st.integers(0, len(edges) - 1))
        e = edges[k]
        edges[k] = EdgeDef(e.src, e.dst, e.port, "q")
        flux_node = e.src
    outside = allow_outside and draw(st.booleans())
    nonlinear = allow_nonlinear and draw(st.booleans()) and draw(st.booleans())
    nodes = []
    for nm in names:
        lv = leaves if leaves is not None else tuple(
            leaf for leaf in LEAF_POOL if draw(st.booleans()))
        alpha = draw(st.sampled_from([0.0, 0.5, -0.25]))
        beta = draw(st.sampled_from([0.0, 1.0, -0.5]))
        nodes.append(NodeDef(nm, ports[nm] + (1 if outside and nm == names[0] else 0),
                             alpha=alpha, beta=beta, nonlinear=nonlinear,
                             leaves=lv, flux=nm == flux_node))
    if outside:
        edges.append(EdgeDef("drv", names[0], ports[names[0]]))
        edges.append(EdgeDef(names[-1], "sink", 0))
        nodes = ([NodeDef("drv", 0, alpha=1.0, beta=1.0,
                          leaves=leaves if leaves is not None else ("count",))]
                 + nodes
                 + [NodeDef("sink", 1, alpha=0.5,
                            leaves=leaves if leaves is not None else ("tag",))])
    return GraphDef(n=n, nodes=tuple(nodes), edges=tuple(edges), group_nodes=tuple(names))


#: Thresholds drawn for the live tolerance knob: one at the float32 floor
#: of an O(1) relative norm (a criterion made of rounding), one just above
#: it, and two well above it.
THRESHOLDS = (2e-7, 1e-6, 1e-4, 1e-2)


@st.composite
def group_configs(draw, gdef: GraphDef, *, accelerations=None, norms=None,
                  modes=None, caps=None, predictors=None, thresholds=THRESHOLDS):
    """A coupling-group configuration with only live knobs set.

    The interface norm is not drawn for a graph with a flux edge
    (``compile()`` refuses it there: MADD-ANO-060).
    """
    has_flux = any(e.field == "q" for e in gdef.edges)
    norms = norms or (("l2", "mixed") if has_flux else ("l2", "mixed", "interface"))
    norms = tuple(nm for nm in norms if not (has_flux and nm == "interface"))
    cfg = dict(
        convergence_norm=draw(st.sampled_from(norms)),
        acceleration=draw(st.sampled_from(
            accelerations or ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj"))),
        iteration_mode=draw(st.sampled_from(modes or ("gauss-seidel", "jacobi"))),
        max_iterations=draw(st.sampled_from(caps or (1, 2, 3, 5, 12, 40))),
        predictor=draw(st.sampled_from(predictors or ("none", "linear", "quadratic"))),
    )
    thr = draw(st.sampled_from(thresholds))
    if cfg["convergence_norm"] == "l2":
        cfg["tolerance"] = thr
    else:
        cfg["rtol"] = thr
    if cfg["acceleration"] == "fixed":
        cfg["relaxation"] = draw(st.sampled_from([0.4, 0.8, 1.3]))
    if cfg["acceleration"] == "iqn-imvj":
        cfg["jacobian_reuse"] = draw(st.integers(0, 4))
    return live_knobs(cfg)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def exact_fixed_point(gdef: GraphDef, values: dict, pre: dict, inputs: dict,
                      dt: float) -> dict:
    """The group's fixed point in float64, for linear nodes.

    The map the graph *evaluates* is solved: gains and biases rounded to
    float32 first, then ``(I - M) x = c`` in float64, with ``c`` holding
    each node's ``alpha x_pre + b + beta dt`` and the contributions of
    edges from outside the group (*inputs*: ``{(dst, port): value}``, the
    float32 values the step fed in).  Returns ``{node: x}``.
    """
    names = list(gdef.group_nodes)
    n = gdef.n
    M = coupling_matrix(gdef, values)
    c = np.zeros(n * len(names))
    for i, nm in enumerate(names):
        nd = gdef.node(nm)
        v = values[nm]
        ci = (np.float64(F32(nd.alpha)) * np.asarray(pre[nm]["x"], np.float64)
              + np.asarray(v["b"], F32).astype(np.float64)
              + np.float64(F32(nd.beta)) * np.float64(F32(dt)))
        for (dst, port), val in inputs.items():
            if dst == nm:
                G = np.asarray(v["G"][port], F32).astype(np.float64)
                ci = ci + G @ np.asarray(val, np.float64)
        c[i * n:(i + 1) * n] = ci
    x = np.linalg.solve(np.eye(len(c)) - M, c)
    return {nm: x[i * n:(i + 1) * n] for i, nm in enumerate(names)}


def back_edges(gdef: GraphDef, schedule) -> set:
    """The edges a step reads from the previous step's state.

    The library's rule (``identify_back_edges``): an edge whose source is
    not scheduled before its target.  Edges inside the coupling group are
    excluded -- the group iterates them as forward edges.
    """
    pos = {nm: i for i, nm in enumerate(schedule)}
    group = set(gdef.group_nodes)
    return {e for e in gdef.edges
            if pos[e.src] >= pos[e.dst] and not (e.src in group and e.dst in group)}


class HandUnrolledMultirate:
    """A multi-rate graph's semantics written out as a Python loop.

    The library's multi-rate step runs every node and the coupling group
    under a ``step_count % divider`` gate inside one compiled program.
    This reference does the same scheduling by hand, with no gate in
    sight: at base step ``k`` each block of the schedule either fires --
    a node through its own jitted ``update``, the group through a
    *uniform-rate* graph of its members alone, its outside inputs fed in
    as external inputs -- or is skipped, and a skipped block's state is
    simply not touched.  Back edges read the state at the start of the
    base step, forward edges the state already updated in it.

    The group's ``_meta`` slots (report, predictor history, IMVJ warm
    start) live in the sub-graph, which only steps when the group fires:
    that is the rule ``coupling_diagnostics()`` documents for a
    multi-rate group, held here by construction rather than by a
    ``lax.cond``.
    """

    def __init__(self, gdef: GraphDef, group: dict, schedule, base_dt: float):
        self.gdef = gdef
        self.schedule = list(schedule)
        self.base_dt = float(base_dt)
        self.back = back_edges(gdef, schedule)
        gnames = set(gdef.group_nodes)
        self.divider = {nd.name: int(round(nd.timestep / base_dt)) for nd in gdef.nodes}
        sub = GraphManager()
        for nd in gdef.nodes:
            if nd.name in gnames:
                sub.add_node(make_node(nd, gdef.n))
        for e in gdef.internal_edges:
            sub.add_edge(e.src, e.dst, e.field, f"u{e.port}")
        self.feeds = [e for e in gdef.edges if e.dst in gnames and e.src not in gnames]
        for e in self.feeds:
            sub.add_external_input(e.dst, f"u{e.port}", shape=(gdef.n,))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "CouplingGroup solver='fori' is deprecated", DeprecationWarning)
            sub.add_coupling_group(list(gdef.group_nodes), **live_knobs(group))
        sub.compile()
        self.sub = sub
        self.nodes = {nd.name: make_node(nd, gdef.n) for nd in gdef.nodes
                      if nd.name not in gnames}
        self._updates = {nm: jax.jit(node.update) for nm, node in self.nodes.items()}

    def _inputs(self, nm, new, full):
        bi = {}
        for e in self.gdef.edges:
            if e.dst != nm:
                continue
            src = full if e in self.back else new
            bi[f"u{e.port}"] = jnp.asarray(src[e.src][e.field])
        return bi

    def run(self, values: dict, n_steps: int, params_of=None):
        """``(trajectory, group_meta_trajectory)`` over *n_steps* base steps.

        The trajectory holds every node's state after each base step; the
        meta trajectory the group's ``_meta`` slots after each base step.
        """
        sub = self.sub
        sub.reset_state()
        state = {}
        for nd in self.gdef.nodes:
            s = dict(make_node(nd, self.gdef.n).initial_state())
            s["x"] = jnp.asarray(values[nd.name]["x0"], jnp.float32)
            state[nd.name] = s
        sub_params = params_for(sub, values)
        node_params = {nm: {**{f"G{j}": jnp.asarray(G, jnp.float32)
                               for j, G in enumerate(values[nm]["G"])},
                            "b": jnp.asarray(values[nm]["b"], jnp.float32)}
                       for nm in self.nodes}
        gnames = list(self.gdef.group_nodes)
        traj, metas = [], []
        group_done = False
        for k in range(n_steps):
            full = {nm: dict(s) for nm, s in state.items()}
            new = {nm: dict(s) for nm, s in state.items()}
            group_done = False
            for nm in self.schedule:
                if nm in gnames:
                    if group_done:
                        continue
                    group_done = True
                    if k % self.divider[gnames[0]] != 0:
                        continue
                    for g in gnames:
                        sub.set_node_state(g, new[g])
                    ext = {}
                    for e in self.feeds:
                        src = full if e in self.back else new
                        ext.setdefault(e.dst, {})[f"u{e.port}"] = jnp.asarray(src[e.src][e.field])
                    sub.step(ext, params=sub_params)
                    for g in gnames:
                        new[g] = dict(sub.get_node_state(g))
                    continue
                if k % self.divider[nm] != 0:
                    continue
                nd = self.gdef.node(nm)
                new[nm] = dict(self._updates[nm](
                    new[nm], self._inputs(nm, new, full), nd.timestep,
                    params=node_params[nm]))
            state = new
            traj.append({nm: {f: np.asarray(v) for f, v in s.items()} for nm, s in state.items()})
            metas.append(group_meta(sub, self.gdef.key))
        return traj, metas


@st.composite
def drawn_values(draw, gdef: GraphDef, *, rhos=(0.3, 0.9, 0.99, 0.999),
                 rank_one: bool = False, bias_scales=(1e-3, 1.0, 1e3)):
    """Values for *gdef*: a drawn rate, normality, seed and field scale.

    The gains themselves come from a NumPy generator seeded by the draw
    (the shrinkable part is the rate, the normality and the scale), so
    one draw is one reproducible spectrum.
    """
    rho = draw(st.sampled_from(rhos))
    nonnormal = draw(st.booleans())
    seed = draw(st.integers(0, 2**32 - 1))
    scale = draw(st.sampled_from(bias_scales))
    return draw_values(np.random.default_rng(seed), gdef, rho, nonnormal=nonnormal,
                       rank_one=rank_one, bias_scale=scale)


def report(gm: GraphManager, gdef: GraphDef) -> Optional[dict]:
    """The loop's own verdict from the ``_meta`` slots, or ``None``.

    ``iterations``, ``total_iterations``, ``residual`` and ``converged``
    exactly as ``coupling_diagnostics()`` derives them -- ``converged``
    through the same :func:`reported_converged` and
    :func:`convergence_criterion` it calls -- without the float-floor and
    spectral computations the full report adds, which cost more than the
    step being measured.  ``None`` where the group reports nothing (a
    ``"fori"`` group without ``diagnostics``).
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        convergence_criterion,
        reported_converged,
    )

    meta = group_meta(gm, gdef.key)
    if "iterations" not in meta or int(meta["iterations"]) == 0:
        return None
    group = gm._coupling_groups[0]  # noqa: SLF001
    threshold, scale = convergence_criterion(group)
    it = int(meta["iterations"])
    amp = float(meta.get("amplification", 0.0))
    return {
        "iterations": it,
        "total_iterations": max(int(meta.get("total_iterations", it)), it),
        "residual": float(meta["residual"]),
        "amplification": amp,
        "step_scale": scale,
        "threshold": threshold,
        "converged": reported_converged(meta["residual"], meta.get("amplification", 0.0),
                                        scale, threshold),
    }


def criterion_is_resolved(rep: dict, floor: float) -> bool:
    """Is this report's verdict further from the threshold than its own rounding?

    The verdict compares ``r max(omega amp, 1)`` with the threshold.  A
    residual carries up to *floor* of rounding (it is a cancellation), and
    ``amp = 1 / (1 - r_k / r_{k-1})`` turns that into ``2 floor / r`` on
    the rate and ``2 omega floor amp**2`` on the estimate (derived in
    ``test_differential_fixed_point.py``).  Two differently compiled
    programs that round an ulp apart can stop on different passes only
    when the estimate is within that of the threshold.
    """
    amp = rep["amplification"] if rep["amplification"] >= 1.0 else 1.0
    omega = rep["step_scale"]
    est = rep["residual"] * max(omega * amp, 1.0)
    resolution = floor * max(omega * amp, 1.0) + 2.0 * omega * floor * amp ** 2
    return abs(est - rep["threshold"]) > resolution


def trajectory(gm: GraphManager, gdef: GraphDef, values: dict, steps: int) -> list:
    """``[(state, group_meta, report or None)]`` after each of *steps* steps.

    Starts from :func:`set_initial`, so the graph's previous state does
    not leak into the measurement.
    """
    set_initial(gm, values)
    params = params_for(gm, values)
    out = []
    for _ in range(steps):
        gm.step(params=params)
        out.append((snapshot(gm), group_meta(gm, gdef.key), report(gm, gdef)))
    return out


def _cycle(m: int, n: int, *, chords=(), flux_edge: Optional[int] = None,
           outside: bool = True, leaves=LEAF_POOL, alpha=0.5, beta=1.0,
           nonlinear: bool = False) -> GraphDef:
    """A fixed structure: the cycle ``g0 -> g1 -> ... -> g0`` plus *chords*."""
    names = [f"g{i}" for i in range(m)]
    ports = {nm: 0 for nm in names}
    edges: list = []
    for src, dst in [(names[i], names[(i + 1) % m]) for i in range(m)] + [
            (names[a], names[b]) for a, b in chords]:
        edges.append(EdgeDef(src, dst, ports[dst]))
        ports[dst] += 1
    if flux_edge is not None:
        e = edges[flux_edge]
        edges[flux_edge] = EdgeDef(e.src, e.dst, e.port, "q")
    flux_src = {e.src for e in edges if e.field == "q"}
    nodes = [NodeDef(nm, ports[nm] + (1 if outside and nm == names[0] else 0),
                     alpha=alpha, beta=beta, nonlinear=nonlinear, leaves=tuple(leaves),
                     flux=nm in flux_src) for nm in names]
    if outside:
        edges.append(EdgeDef("drv", names[0], ports[names[0]]))
        edges.append(EdgeDef(names[-1], "sink", 0))
        nodes = ([NodeDef("drv", 0, alpha=1.0, beta=1.0, leaves=tuple(leaves))] + nodes
                 + [NodeDef("sink", 1, alpha=0.5, leaves=tuple(leaves))])
    return GraphDef(n=n, nodes=tuple(nodes), edges=tuple(edges), group_nodes=tuple(names))


#: The per-push structures.  ``TRIANGLE`` carries every leaf kind and an
#: outside driver and sink; ``FLUX_PAIR`` couples through a flux edge;
#: ``NONLINEAR_RING`` is a four-node ring of ``tanh`` relays with a chord.
TRIANGLE = _cycle(3, 2, chords=((0, 2),))
FLUX_PAIR = _cycle(2, 1, flux_edge=0)
NONLINEAR_RING = _cycle(4, 1, chords=((1, 3),), nonlinear=True)
STRUCTURES = {"triangle": TRIANGLE, "flux-pair": FLUX_PAIR, "nonlinear-ring": NONLINEAR_RING}


def steer_around_known_crashes(gdef: GraphDef, group: dict) -> dict:
    """*group*, moved off a configuration known to raise at trace.

    A Jacobi group in which a flux producer reads another node's flux
    raises ``KeyError`` (the Jacobi pass resolves producers' inputs before
    any flux exists; the Gauss-Seidel pass has a two-sweep seed for it).
    Pinned as a strict xfail in ``test_differential_leaves.py``; oracles
    that are about something else are steered to Gauss-Seidel instead.
    """
    producers = {nd.name for nd in gdef.nodes if nd.flux}
    if group.get("iteration_mode") == "jacobi" and any(
            e.field == "q" and e.dst in producers for e in gdef.edges):
        return dict(group, iteration_mode="gauss-seidel")
    return group


#: Leaves the predictor's unflatten drops from the group's starting iterate.
NON_FLOAT_LEAVES = ("count", "tag", "flag", "sign")


def steer_leaves_around_known_crashes(gdef: GraphDef, group: dict) -> GraphDef:
    """*gdef* without non-float group leaves where they are known to raise.

    A predictor (``"linear"`` / ``"quadratic"``) under the ``"mixed"``
    norm raises ``KeyError`` on a group node holding an integer or boolean
    leaf: the predictor rebuilds the starting iterate from its floating
    fields alone, and the mixed norm reads every field of it.  Pinned as a
    strict xfail in ``test_differential_leaves.py``.
    """
    if group.get("predictor", "none") != "none" and group.get("convergence_norm") == "mixed":
        return dataclasses.replace(gdef, nodes=tuple(
            dataclasses.replace(nd, leaves=tuple(lf for lf in nd.leaves
                                                 if lf not in NON_FLOAT_LEAVES))
            if nd.name in gdef.group_nodes else nd for nd in gdef.nodes))
    return gdef


def gauss_seidel_matrix(gdef: GraphDef, values: dict) -> np.ndarray:
    """The error map of one Gauss-Seidel sweep, float64, for linear nodes.

    Sweeps the group in ``gdef.group_nodes`` order (the schedule a group
    with no outside predecessor gets: insertion order), each node reading
    the already-updated value of every node before it and the previous
    iterate of every node after it.  Column ``j`` is the sweep applied to
    the ``j``-th unit error, with every constant zeroed.
    """
    names = list(gdef.group_nodes)
    n = gdef.n
    dim = n * len(names)
    J = np.zeros((dim, dim))
    for col in range(dim):
        old = np.zeros(dim)
        old[col] = 1.0
        new = old.copy()
        for i, nm in enumerate(names):
            acc = np.zeros(n)
            for e in gdef.internal_edges:
                if e.dst != nm:
                    continue
                j = names.index(e.src)
                fac = 2.0 if e.field == "q" else 1.0
                G = np.asarray(values[nm]["G"][e.port], F32).astype(np.float64)
                acc = acc + fac * G @ new[j * n:(j + 1) * n]
            new[i * n:(i + 1) * n] = acc
        J[:, col] = new
    return J


def group_distance(gm: GraphManager, gdef: GraphDef, group: dict, got: dict,
                   exact: dict) -> float:
    """Distance between *got* and *exact* in the group's own convergence norm.

    A float64 NumPy restatement of the three documented norms (each
    field's change divided by ``rtol`` times the field's magnitude, the
    larger of its two ``max |v|``; a field whose magnitude does not exceed
    ``atol`` leaves the norm; ``"l2"`` the root sum of squares at
    ``rtol = 1``, ``"mixed"`` the RMS over every active entry,
    ``"interface"`` the RMS over the internal edges' source values) --
    written out here rather than called, so a fault in the library's norm
    is a disagreement and not a shared assumption.  Only ``x`` (and the
    flux ``q = 2 x``) can differ from the fixed point: every other
    floating field is independent of the iterate.
    """
    nodes = list(gdef.group_nodes)
    norm = group.get("convergence_norm", "l2")
    atol = float(group.get("atol", 0.0))
    rtol = 1.0 if norm == "l2" else float(group.get("rtol", 1e-6))

    def scaled(a, b):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        ref = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if not (ref > atol and ref > 0.0):
            return np.zeros(0)
        return np.abs(a - b) / (rtol * ref)

    if norm == "interface":
        parts = []
        for e in gdef.internal_edges:
            fac = 2.0 if e.field == "q" else 1.0
            parts.append(scaled(fac * np.asarray(got[e.src]["x"], np.float64),
                                fac * np.asarray(exact[e.src], np.float64)))
    else:
        parts = [scaled(got[nm]["x"], exact[nm]) for nm in nodes]
    flat = np.concatenate(parts) if parts else np.zeros(0)
    if norm == "l2":
        return float(np.sqrt(np.sum(flat ** 2)))
    return float(np.sqrt(np.sum(flat ** 2) / max(flat.size, 1)))


def note(message: str) -> None:
    """``hypothesis.note`` inside a property; nothing in a plain test."""
    from hypothesis import note as _note  # noqa: PLC0415
    from hypothesis.errors import InvalidArgument  # noqa: PLC0415

    try:
        _note(message)
    except InvalidArgument:
        pass
