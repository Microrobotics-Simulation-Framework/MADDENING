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

    ``f`` is the identity, or ``tanh`` when *nonlinear*.  Each input port
    ``u{j}`` takes at most one edge, so an external input can stand in for
    any edge without the additive/replacive question arising (the
    hand-unrolled multi-rate reference relies on that).
    """

    def __init__(self, name, timestep, *, n, n_inputs, alpha=0.0, beta=0.0,
                 nonlinear=False, leaves=(), x0=None):
        params = {f"G{j}": jnp.zeros((n, n), jnp.float32) for j in range(n_inputs)}
        params["b"] = jnp.zeros(n, jnp.float32)
        super().__init__(name, timestep, **params)
        self._n = int(n)
        self._k = int(n_inputs)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._nl = bool(nonlinear)
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
        x = (jnp.float32(self._alpha) * state["x"] + p["b"]
             + jnp.float32(self._beta) * jnp.asarray(dt, jnp.float32))
        for j in range(self._k):
            u = boundary_inputs.get(f"u{j}", jnp.zeros(self._n, jnp.float32))
            if self._nl:
                u = jnp.tanh(u)
            x = x + p[f"G{j}"] @ u
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
        super().__init__(inner.name, inner.timestep * d, **inner.params)
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

    def with_leaves(self, leaves: tuple, names=None) -> "GraphDef":
        names = set(self.group_nodes if names is None else names)
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, leaves=tuple(leaves)) if nd.name in names else nd
            for nd in self.nodes))


def make_node(nd: NodeDef, n: int, x0=None) -> Relay:
    cls = FluxRelay if nd.flux else Relay
    return cls(nd.name, nd.timestep, n=n, n_inputs=nd.n_inputs, alpha=nd.alpha,
               beta=nd.beta, nonlinear=nd.nonlinear, leaves=nd.leaves, x0=x0)


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

    The internal gains are drawn, assembled into :func:`coupling_matrix`
    and multiplied by one common factor so its spectral radius is
    ``|rho|``.  *nonnormal* adds a strongly upper-triangular part to every
    gain; *rank_one* draws the internal gains so ``M`` has rank one (one
    coupling mode, eigenvalue ``rho`` exactly, which may be negative), the
    regime in which the residual sequence is a clean geometric decay and
    the error estimate has no excuse.
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
    if rank_one:
        # M = u v^T with v^T u = rho: block (i, j) = u_i v_j^T.  Each
        # internal edge (dst <- src) carries the block u_dst v_src^T; a
        # pair of nodes joined by two edges splits it between them.
        dim = n * len(names)
        u = rng.normal(size=dim)
        v = rng.normal(size=dim)
        if nonnormal:
            v = v + 3.0 * rng.normal(size=dim)
        blocks: dict = {}
        for e in gdef.internal_edges:
            blocks.setdefault((e.dst, e.src), []).append(e)
        for (dst, src), edges in blocks.items():
            i, j = names.index(dst), names.index(src)
            blk = np.outer(u[i * n:(i + 1) * n], v[j * n:(j + 1) * n])
            for e in edges:
                fac = 2.0 if e.field == "q" else 1.0
                values[dst]["G"][e.port] = blk / (len(edges) * fac)
        # Pairs with no edge contribute nothing; rescale the realised M.
        M = coupling_matrix(gdef, {k: {**v_, "G": [np.asarray(g, np.float64) for g in v_["G"]]}
                                   for k, v_ in values.items()})
        lam = np.linalg.eigvals(M)
        top = lam[np.argmax(np.abs(lam))]
        scale = rho / top.real if abs(top) > 0 else 0.0
    else:
        M = coupling_matrix(gdef, {k: {**v_, "G": [np.asarray(g, np.float64) for g in v_["G"]]}
                                   for k, v_ in values.items()})
        r = float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0
        scale = abs(rho) / r if r > 0 else 0.0
    group = set(names)
    for e in gdef.edges:
        if e.src in group and e.dst in group:
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


def set_initial(gm: GraphManager, values: dict) -> None:
    """Reset *gm* (state and coupling seeds) and write every node's ``x0``."""
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
