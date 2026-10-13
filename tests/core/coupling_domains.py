"""Domain fixtures: one coupled graph run in each numeric domain of the coupling inventory.

``docs/validation/coupling_claims.yaml`` gives every row a domain matrix
(``testing_standards.md``, "The domain matrix").  A claim verified in
float32 under ``jit`` is *narrowed* in a domain no test runs it in.  This
module lets a test state its claim once, over a two-member group built by
:func:`pair`, and run it in each domain by wrapping the graph and its step
rather than writing the claim again:

* ``f32``: float32 members, x64 off (the default lane);
* ``f64``: under ``jax_enable_x64``, both members float64;
* ``mixed_dtype``: under x64, member ``a`` float32 and member ``b``
  float64, joined by casting edges (the graph refuses an edge between
  dtypes without a transform);
* ``bfloat16`` and ``float16`` (the ``16bit`` domain): both members in
  the 16-bit dtype; each has its own ``finfo`` edges, so a claim about
  an edge is held at each;
* ``vmap``: the step under ``jax.vmap``, each member of the batch a
  different scenario (parameters and state batched together), and each
  member's report derived from its own slots by the host code the graph
  uses for its own state;
* ``multi_rate``: a clock outside the group at half the group's
  timestep, so the group fires on every other base step; each solve is
  read after the pair of base steps that holds it;
* ``sub_cycled``: ``subcycling=True``, member ``b`` at half the group's
  timestep and sub-stepped twice per pass;
* ``predictors_warm_starts``: ``predictor="quadratic"`` (and, where the
  claim's acceleration is IQN-IMVJ, ``jacobian_reuse=2``), over a run
  whose forcing moves every step so the predictor's guess is not the
  fixed point;
* ``checkpoint_restart``: the run saved after two steps, run on, reset,
  loaded and run again; the step after the load is checked, and it
  must reproduce the uninterrupted run's to the bit;
* ``adaptive``: each solve taken by ``run_adaptive`` over one step of
  ``DT`` (slow: the stepper compiles its step on every call);
* ``sharded``: member ``b`` partitioned over four CPU virtual devices by
  ``ShardedPointwiseNode``, every field four copies of the test's entries,
  so the group's norm, floor and spectrum read a sharded field through
  cross-device reductions (run from ``tests/cloud/multigpu``).

**The members are memoryless**: ``x <- g * u + c`` with ``g`` a scalar
and ``c`` a vector, both parameters, so one compiled graph serves every
scenario of a test and a step's fixed point depends only on the
parameters it ran with -- not on the step size, the sub-cycling, the
rate divider, the predictor's guess or the restart.  Its closed form is
in :func:`fixed_point`, in float64 from the parameters as each member's
dtype rounds them.

A run gives one :class:`Solve` per checked step: the members' state on
the host, the report ``coupling_diagnostics()`` gives for it, and the
parameters it ran with.

Both members of :func:`pair` have one size, so a mapped edge between them
is a tie.  :mod:`tests.core.coupling_domain_sizes` builds, in the same
domains and for the same :func:`run` and :func:`run_sequence`, a small
field coupled to a large one: a mapped edge that expands and one that
reduces.
"""

from __future__ import annotations

import contextlib
import dataclasses
import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core.coupling_reason_rules import assert_reason_rules

KEY = "a+b"
#: The library's keys of the pair's two edges, ``b -> a`` and ``a -> b``:
#: where a mapped edge's weights live in ``params["mappings"]``.
EDGE_KEYS = ("b.x->a.u", "a.x->b.u")
#: Exactly representable in every float dtype.
DT = 0.125


@dataclasses.dataclass(frozen=True)
class Domain:
    """One numeric domain: the members' dtypes and how the graph is stepped."""

    label: str                      # the pytest id
    name: str                       # the inventory's domain name
    dtype_a: object = jnp.float32
    dtype_b: object = jnp.float32
    x64: bool = False
    vmap: bool = False
    multirate: bool = False
    subcycled: bool = False
    predictor: bool = False
    restart: bool = False
    adaptive: bool = False
    sharded: bool = False

    @property
    def dtypes(self) -> tuple:
        return (self.dtype_a, self.dtype_b)

    @property
    def coarsest(self):
        """The member dtype with the larger ``eps``."""
        return max(self.dtypes, key=lambda d: float(jnp.finfo(d).eps))


DOMAINS = {d.label: d for d in (
    Domain("f32", "f32"),
    Domain("f64", "f64", jnp.float64, jnp.float64, x64=True),
    Domain("mixed_dtype", "mixed_dtype", jnp.float32, jnp.float64, x64=True),
    Domain("bfloat16", "16bit", jnp.bfloat16, jnp.bfloat16),
    Domain("float16", "16bit", jnp.float16, jnp.float16),
    Domain("vmap", "vmap", vmap=True),
    Domain("multi_rate", "multi_rate", multirate=True),
    Domain("sub_cycled", "sub_cycled", subcycled=True),
    Domain("predictors_warm_starts", "predictors_warm_starts", predictor=True),
    Domain("checkpoint_restart", "checkpoint_restart", restart=True),
    Domain("adaptive", "adaptive", adaptive=True),
    Domain("sharded", "sharded", sharded=True),
)}
#: The dtype domains, and the graph domains (float32 members, stepped differently).
DTYPES = ("f32", "f64", "mixed_dtype", "bfloat16", "float16")
GRAPHS = ("vmap", "multi_rate", "sub_cycled", "predictors_warm_starts", "checkpoint_restart")
EVERY = DTYPES + GRAPHS
#: ``run_adaptive`` compiles its dt-parameterised step on every call (about
#: two seconds with diagnostics on three cores), so its cells are slow tests.
ADAPTIVE = "adaptive"
#: ``run_adaptive``'s settings: one accepted step of ``DT`` per call.  The
#: members are memoryless, so the step-doubling error is zero and the
#: step is accepted at ``dt_max``.
ADAPTIVE_KW = dict(dt_initial=DT, dt_max=DT, dt_min=DT / 64, rtol=1e-2, atol=1e-6)
#: The sharded domain partitions member ``b`` over this many devices, and
#: every field is this many copies of the entries a test asks for (so the
#: claim and its oracle are the test's, entry by entry).  Needs the CPU
#: virtual devices ``tests/cloud/multigpu``'s conftest provides; the sharded
#: cells run from there.
N_SHARD = 4
SHARDED = "sharded"


@contextlib.contextmanager
def x64(on: bool):
    """``jax_enable_x64`` set to *on* for the block, restored after it."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def entered(domain: Domain):
    """The context a domain's graphs are built and stepped in."""
    return x64(domain.x64)


def finfo(dtype):
    """``jnp.finfo`` (numpy's does not know bfloat16)."""
    return jnp.finfo(dtype)


# ---------------------------------------------------------------------------
# The members
# ---------------------------------------------------------------------------
class Lin(SimulationNode):
    """``x <- g * u + c`` on a length-``n`` field, ``g`` and ``c`` parameters."""

    def __init__(self, name, timestep, dtype, n, *, g, c):
        c = np.broadcast_to(np.asarray(c, np.float64), (n,))
        # ``g`` a scalar, or one gain per entry (independent modes).
        super().__init__(name, timestep, g=jnp.asarray(np.asarray(g, np.float64), dtype),
                         c=jnp.asarray(c, dtype))
        self._dtype, self._n = dtype, n

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros(self._n, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = jnp.asarray(boundary_inputs["u"]).astype(self._dtype)
        return {"x": (p["g"] * u + p["c"]).astype(self._dtype)}

    def update_evaluations(self):
        return 1


class Flux(Lin):
    """A :class:`Lin` that also produces the flux ``q = 2 x`` (a function of
    its state alone), for flux edges inside the group."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": jnp.asarray(2.0, self._dtype) * state["x"]}


class Ticker(SimulationNode):
    """A clock outside the group; at half its timestep the graph is multi-rate."""

    def initial_state(self):
        return {"t": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"t": state["t"] + jnp.asarray(dt, jnp.float32)}


def group_kwargs(domain: Domain, **kw) -> dict:
    """*kw* with the domain's group settings added."""
    kw = dict(kw)
    if domain.subcycled:
        kw.update(subcycling=True, boundary_interpolation="constant")
    if domain.predictor:
        kw["predictor"] = "quadratic"
        if kw.get("acceleration") == "iqn-imvj":
            kw.setdefault("jacobian_reuse", 2)
    return kw


def pair(domain: Domain, *, n=1, g=(0.5, 0.5), c=(1.0, 0.0), x0=None, node=Lin,
         fields=("x", "x"), mappings=None, extra_edges=(), **group_kw) -> GraphManager:
    """``x_a <- g_a x_b + c_a``, ``x_b <- g_b x_a + c_b`` in *domain*, compiled.

    Build it inside :func:`entered`.  *group_kw* go to the group, with the
    domain's settings added by :func:`group_kwargs`.  *node* is the
    members' class (:class:`Lin` or :class:`Flux`) and *fields* the source
    fields of the edges ``b -> a`` and ``a -> b`` (``"q"`` is a flux edge).

    *mappings* is ``(H_ba, H_ab)``: an ``n x n`` matrix for the interface
    mapping on ``b -> a`` and on ``a -> b`` (``None`` for an edge without
    one), so the pair is ``x_a <- g_a (H_ba x_b) + c_a``, ``x_b <- g_b
    (H_ab x_a) + c_b``.  Each mapping is built in the dtype of the member
    its edge feeds, and the edge's cast (the mixed-dtype domain) follows it.
    The matrices are the mapping objects' own weights; :func:`params_with`
    passes others for a step.

    *extra_edges* are further edges between the two members, each
    ``(source, target, source field, target port)``, added after the pair's
    own with the same cast: a field that a second internal edge reads, into
    a port *node* declares beside ``"u"``.
    """
    gm = GraphManager()
    da, db = domain.dtypes
    tb = DT / 2 if domain.subcycled else DT
    maps = tuple(
        None if H is None else matrix_mapping(jnp.asarray(_mapping_matrix(domain, H), dtype))
        for H, dtype in zip(mappings or (None, None), (da, db)))
    if domain.sharded:
        n, g, c = n * N_SHARD, tuple(_tiled(v) for v in g), tuple(_tiled(v) for v in c)
    gm.add_node(node("a", DT, da, n, g=g[0], c=c[0]))
    member_b = node("b", tb, db, n, g=g[1], c=c[1])
    if domain.sharded:
        from maddening.cloud.multigpu.device_mesh import create_device_mesh
        from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
        member_b = ShardedPointwiseNode(member_b, create_device_mesh(shape=(N_SHARD,)),
                                        shard_axes=0)
    gm.add_node(member_b)
    if da == db:
        gm.add_edge("b", "a", fields[0], "u", mapping=maps[0])
        gm.add_edge("a", "b", fields[1], "u", mapping=maps[1])
    else:
        gm.add_edge("b", "a", fields[0], "u", transform=lambda v: v.astype(da), mapping=maps[0])
        gm.add_edge("a", "b", fields[1], "u", transform=lambda v: v.astype(db), mapping=maps[1])
    for source, target, field, port in extra_edges:
        to = da if target == "a" else db
        cast = {} if da == db else {"transform": lambda v, to=to: v.astype(to)}
        gm.add_edge(source, target, field, port, **cast)
    if domain.multirate:
        gm.add_node(Ticker("tick", DT / 2))
    with warnings.catch_warnings():
        # The deprecated solver="fori" says so; a test that asks for it knows.
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["a", "b"], **group_kwargs(domain, **group_kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate notice
        gm.compile()
    if x0 is not None:
        set_x(gm, x0)
    return gm


def _tiled(v):
    """*v* (a scalar, or one value per entry) as ``N_SHARD`` copies of its entries."""
    a = np.asarray(v, np.float64)
    return a if a.ndim == 0 else np.tile(a, N_SHARD)


def _mapping_matrix(domain: Domain, H) -> np.ndarray:
    """*H* in float64, as ``N_SHARD`` diagonal copies where the sharded domain
    multiplied the entries (each copy of the field mapped by itself)."""
    H = np.asarray(H, np.float64)
    return np.kron(np.eye(N_SHARD), H) if domain.sharded else H


def _fit(v, shape):
    """*v* broadcast to *shape*, tiled first where the sharded domain multiplied
    the entries."""
    a = np.asarray(v, np.float64)
    if a.ndim and shape and a.shape[-1] != shape[-1]:
        a = np.tile(a, shape[-1] // a.shape[-1])
    return np.broadcast_to(a, shape)


def set_x(gm: GraphManager, x0) -> None:
    """Start both members at *x0* (a pair of arrays or scalars)."""
    for name, v in zip(("a", "b"), x0):
        cur = gm._state[name]["x"]
        new = jnp.asarray(_fit(v, cur.shape), cur.dtype)
        sharding = getattr(cur, "sharding", None)
        if sharding is not None and len(sharding.device_set) > 1:
            new = jax.device_put(new, sharding)         # keep a sharded field sharded
        gm._state[name] = {**gm._state[name], "x": new}


def params_with(gm: GraphManager, *, g=None, c=None, mappings=None) -> dict:
    """The graph's parameter pytree with ``g`` and ``c`` replaced, in each leaf's dtype.

    *mappings* is ``(H_ba, H_ab)``: the weights of the pair's mapped edges
    for the step (``None`` keeps an edge's own), as :func:`pair` takes them.
    """
    p = jax.tree.map(lambda v: v, gm.params)
    for i, name in enumerate(("a", "b")):
        leaves = p["nodes"][name]
        if g is not None:
            leaves["g"] = jnp.asarray(_fit(g[i], leaves["g"].shape), leaves["g"].dtype)
        if c is not None:
            leaves["c"] = jnp.asarray(_fit(c[i], leaves["c"].shape), leaves["c"].dtype)
    for key, H in zip(EDGE_KEYS, mappings or ()):
        if H is None:
            continue
        leaf = p["mappings"][key]["H"]
        H = np.asarray(H, np.float64)
        if H.shape != leaf.shape:           # the sharded domain's copies
            H = np.kron(np.eye(leaf.shape[0] // H.shape[0]), H)
        p["mappings"][key]["H"] = jnp.asarray(H, leaf.dtype)
    return p


# ---------------------------------------------------------------------------
# Running a domain
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Solve:
    """One checked step: the members' state, the report, the parameters."""

    state: dict             # {"a": {"x": ...}, "b": {"x": ...}} host arrays
    meta: dict
    report: dict
    params: dict
    pre: dict | None = None  # the members' state the solve started from

    def x(self, name) -> np.ndarray:
        return np.asarray(self.state[name]["x"])

    def gains(self) -> tuple:
        """``(g_a, g_b)`` in float64 as stored: floats, or arrays for per-entry gains."""
        out = []
        for n in ("a", "b"):
            g = np.asarray(self.params["nodes"][n]["g"], np.float64)
            out.append(float(g) if g.ndim == 0 else g)
        return tuple(out)

    def forcing(self) -> tuple:
        return tuple(np.asarray(self.params["nodes"][n]["c"], np.float64) for n in ("a", "b"))

    def mapping_matrices(self) -> tuple:
        """``(H_ba, H_ab)`` in float64 as the step held them; the identity
        for an edge without a mapping."""
        maps = self.params.get("mappings", {})
        n = self.x("a").shape[0]
        return tuple(np.asarray(maps[key]["H"], np.float64) if key in maps else np.eye(n)
                     for key in EDGE_KEYS)


def fixed_point(s: Solve) -> tuple:
    """``(x_a*, x_b*)`` in float64 from the step's parameters, as stored."""
    ga, gb = s.gains()
    ca, cb = s.forcing()
    xa = (ca + ga * cb) / (1.0 - ga * gb)
    return xa, gb * xa + cb


def mapped_fixed_point(s: Solve) -> tuple:
    """``(x_a*, x_b*)`` of a pair with mapped edges, in float64 from the
    step's parameters and mapping weights as stored: the solution of
    ``x_a = g_a (H_ba x_b) + c_a``, ``x_b = g_b (H_ab x_a) + c_b``."""
    ga, gb = (np.broadcast_to(np.asarray(g, np.float64), s.x("a").shape) for g in s.gains())
    ca, cb = s.forcing()
    h_ba, h_ab = s.mapping_matrices()
    n = len(ca)
    system = np.block([[np.eye(n), -ga[:, None] * h_ba], [-gb[:, None] * h_ab, np.eye(n)]])
    x = np.linalg.solve(system, np.concatenate([ca, cb]))
    return x[:n], x[n:]


def _host(tree):
    return jax.tree.map(np.asarray, tree)


def report_of(gm: GraphManager, state=None) -> dict:
    """``coupling_diagnostics()[KEY]`` for *state* (the graph's own by default).

    Every report the battery reads keeps the rules of its reason codes
    (``assert_reason_rules``): in each domain, and for each member of a
    vmapped step, whose codes come from that member's slots."""
    if state is None:
        report = dict(gm.coupling_diagnostics()[KEY])
    else:
        saved, gm._state = gm._state, state
        try:
            report = dict(gm.coupling_diagnostics()[KEY])
        finally:
            gm._state = saved
    assert_reason_rules(report, KEY)
    return report


def _solve(gm, params, state=None, pre=None) -> Solve:
    full = gm._state if state is None else state
    return Solve(_host({n: full[n] for n in ("a", "b")}), _host(full.get("_meta", {})),
                 report_of(gm, state), params, pre)


def _members(gm, state=None) -> dict:
    full = gm._state if state is None else state
    return _host({n: full[n] for n in ("a", "b")})


def _stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def bitwise(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


_VMAPPED: dict = {}


def run(domain: Domain, gm: GraphManager, params_seq: list, *, x0=None) -> list:
    """One :class:`Solve` per entry of *params_seq*, each from a fresh start.

    Each scenario starts from the graph's initial state (or *x0*); the
    domain decides how the step that solves it is taken.  Run it inside
    :func:`entered`.
    """
    if domain.vmap:
        return _run_vmap(gm, params_seq, x0)
    out = []
    for p in params_seq:
        gm.reset_state()
        if x0 is not None:
            set_x(gm, x0)
        out.append(_one(domain, gm, p))
    return out


def _one(domain: Domain, gm: GraphManager, p) -> Solve:
    pre = _members(gm)
    if domain.adaptive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.run_adaptive(DT, params=p, **ADAPTIVE_KW)
        return _solve(gm, p, pre=pre)
    if domain.multirate:
        # The group fires on one of every two base steps; between firings
        # its report and its members are the last applied solve's, so the
        # pair holds exactly one solve, started from the state before it.
        gm.step(params=p)
        gm.step(params=p)
        return _solve(gm, p, pre=pre)
    gm.step(params=p)
    return _solve(gm, p, pre=pre)


def _run_vmap(gm, params_seq, x0):
    """Every scenario as one member of a ``jax.vmap`` of the step."""
    if id(gm) not in _VMAPPED:
        _VMAPPED[id(gm)] = (gm, jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, 0))))
    step = _VMAPPED[id(gm)][1]
    gm.reset_state()
    if x0 is not None:
        set_x(gm, x0)
    state = _stack([gm._state] * len(params_seq))
    new = step(state, gm._default_external_inputs(), _stack(params_seq))
    out = []
    for i, p in enumerate(params_seq):
        member = jax.tree.map(lambda v, i=i: v[i], new)
        out.append(_solve(gm, p, member, pre=_members(gm, jax.tree.map(lambda v, i=i: v[i],
                                                                       state))))
    return out


#: Increments of the forcing between steps: irregular, so no polynomial the
#: quadratic predictor extrapolates passes through them and its guess is
#: never the fixed point.
_MOVES = (0.0, 0.31, -0.17, 0.52, 0.11, -0.29, 0.43, 0.07)


def moving(gm: GraphManager, steps: int, *, g=(0.5, 0.5), c0=(1.0, 0.0), dc=(1.0, -0.5),
           scale: float = 1.0) -> list:
    """Parameters for *steps* steps whose forcing moves every step, times *scale*.

    Step ``k`` has ``c = scale * (c0 + dc * sum(_MOVES[:k + 1]))``: a forcing
    that moves irregularly, so a predictor's guess is never exact.
    """
    out, acc = [], 0.0
    for k in range(steps):
        acc += _MOVES[k % len(_MOVES)]
        out.append(params_with(gm, g=g, c=tuple(
            scale * (np.asarray(c0[i], np.float64) + acc * np.asarray(dc[i], np.float64))
            for i in range(2))))
    return out


def run_sequence(domain: Domain, gm: GraphManager, params_seq: list, *, x0=None) -> list:
    """One :class:`Solve` per step of one continuous run over *params_seq*.

    The predictor domain needs the run's history (its guess extrapolates
    the last converged states); the restart domain saves after two
    steps, runs on, resets, loads and runs again, and returns the steps
    after the load -- each checked against the uninterrupted run's to
    the bit.  Other domains step through the sequence.
    """
    gm.reset_state()
    if x0 is not None:
        set_x(gm, x0)
    if domain.vmap:
        raise ValueError("a sequence has no batch axis; use run()")
    if not domain.restart:
        return [_one(domain, gm, p) for p in params_seq]
    start = min(2, len(params_seq) - 1)
    for p in params_seq[:start]:
        _one(domain, gm, p)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        gm.save_state(path)
        straight = [_one(domain, gm, p) for p in params_seq[start:]]
        gm.reset_state()
        gm.load_state(path)
        restarted = [_one(domain, gm, p) for p in params_seq[start:]]
    for a, b in zip(straight, restarted):
        for n in ("a", "b"):
            assert bitwise(a.x(n), b.x(n)), (
                f"the run after the checkpoint restart left the uninterrupted one: "
                f"{n}.x {b.x(n)!r} against {a.x(n)!r}")
        assert a.report == b.report or _same_report(a.report, b.report), (a.report, b.report)
    return restarted


def _same_report(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                return False
        elif x != y:
            return False
    return True


def assert_in_domain(domain: Domain, gm: GraphManager, solves: list) -> None:
    """The premise: the graph and the solves are in *domain*.

    A domain fixture that silently fell back to the default lane (a member
    promoted to float32, a group that did not sub-cycle, a graph that is
    not multi-rate) would verify the default lane twice.
    """
    assert solves, "no solve to check"
    for s in solves:
        for name, dtype in zip(("a", "b"), domain.dtypes):
            assert s.x(name).dtype == jnp.dtype(dtype), (
                f"{domain.label}: member {name} holds {s.x(name).dtype}, not "
                f"{jnp.dtype(dtype)} (x64 {jax.config.jax_enable_x64})")
    group = gm._committed_coupling_groups[KEY]
    assert bool(gm._is_multirate) == domain.multirate, "multi-rate premise"
    assert group.subcycling == domain.subcycled, "sub-cycling premise"
    if domain.subcycled:
        assert gm.get_node("b").delta_t == DT / 2, "member b sub-steps twice per pass"
    assert (group.predictor != "none") == domain.predictor, "predictor premise"
    field = gm._state["b"]["x"]
    devices = len(field.sharding.device_set) if hasattr(field, "sharding") else 1
    assert (devices == N_SHARD) == domain.sharded, f"sharding premise: {devices} devices"
