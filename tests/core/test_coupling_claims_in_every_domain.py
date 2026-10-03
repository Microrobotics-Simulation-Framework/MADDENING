"""The coupling inventory's report, bound and gradient claims, in every numeric domain.

``docs/validation/coupling_claims.yaml`` gives each row a ``domains``
matrix: the numeric domains its conditions cover, and the test that
exercises the claim in each.  The rows' own tests run float32 graphs in
the default lane; this module states the same claims once each, as a
check function keyed by its row id, and runs every check in the other
domains:

* ``f64``: under ``jax_enable_x64``, both members float64;
* ``mixed_dtype``: under x64, one member float32 and one float64;
* ``16bit``: both members bfloat16;
* ``vmap``: the step under ``jax.vmap`` over three states, each member's
  report derived from its own slots by the code the graph uses for its
  own state;
* ``multi_rate``: a node outside the group at half the group's
  timestep, so the group fires on every other base step;
* ``sub_cycled``: ``subcycling=True``, one member at half the other's
  timestep, sub-stepped twice per pass;
* ``predictors_warm_starts``: ``predictor="quadratic"`` with
  ``acceleration="iqn-imvj"`` and ``jacobian_reuse=2``;
* ``adaptive``: ``run_adaptive``, the report read after each call, and
  gradients through ``run_adaptive_scan``;
* ``checkpoint_restart``: every run saved after two steps, run on,
  loaded and run again; every claim is checked on the steps after the
  load.

**The fixture makes every claim checkable exactly.**  Two scalar members,
``x_a <- g_a u_a + c_a`` and ``x_b <- g_b u_b + c_b``, each reading the
other, with ``c = b0 + b1 * clock`` computed from the node's own clock
and recorded in its state.  The map is affine and *memoryless*: a step's
fixed point is ``x* = (I - M)^{-1} c`` with ``M = [[0, g_a], [g_b, 0]]``,
whatever the step size, the sub-cycling, the predictor or the restart,
and it is computed here in float64 from the returned state's own ``c``.
``g`` and ``b0`` are parameters, so one compiled graph serves every
scenario (two spectra, a forcing that stops moving, a non-finite
forcing, a power-of-two rescaling), and the derivatives of one step with
respect to ``b0_a`` (additive) and ``g_a`` (multiplicative) have closed
forms: the IFT rule linearised at the returned iterate is
``(I - M)^{-1} dF/d theta (x_k)``.

**Tolerances.**  A comparison against the float64 oracle allows the
evaluated map's own rounding: the floor the report adds, ``4 m eps``
per entry in the norm's units, recomputed here from its documented
formula (``FLOOR`` below), and ``64 eps`` relative times the conditioning
``||(I - M)^{-1}||`` for a derivative.  Comparisons between programs the
documentation says compute the same arithmetic (diagnostics on and off;
the IFT gradient in an additive parameter of an affine map under two
accelerations; a power-of-two rescaling; a scaled cotangent) are bit for
bit.

The sharded domain is in
``tests/cloud/multigpu/test_coupling_claims_on_a_sharded_graph.py``, which
needs the virtual devices that directory's conftest provides.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import residual_precision_floor
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
#: Exactly representable in every float dtype, bfloat16 included.
DT = 0.125
#: Steps per scenario (base steps are twice as many on the multi-rate graph).
STEPS = 6
#: The report's floor: ``PRECISION_FLOOR_ULPS`` units of ``eps`` per entry
#: per evaluation (``acceleration.py``), restated rather than imported.
FLOOR_ULPS = 4.0


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------
class _Forced(SimulationNode):
    """``x <- g * u + c`` with ``c = b0 + b1 * clock``, the clock advanced first.

    Memoryless: ``x`` does not read its own previous value, so a step's
    coupled fixed point depends only on the gains and the ``c`` recorded
    beside it.
    """

    def __init__(self, name, timestep, dtype, *, g, b0, b1):
        super().__init__(name, timestep, g=jnp.asarray(g, dtype), b0=jnp.asarray(b0, dtype),
                         b1=jnp.asarray(b1, dtype))
        self._dtype = dtype

    def initial_state(self):
        z = jnp.zeros((), self._dtype)
        return {"x": z, "c": z, "clock": z}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=self._dtype,
                                       default=jnp.zeros((), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        clock = state["clock"] + jnp.asarray(dt, self._dtype)
        c = (p["b0"] + p["b1"] * clock).astype(self._dtype)
        u = jnp.asarray(boundary_inputs.get("u", jnp.zeros((), self._dtype))).astype(self._dtype)
        return {"x": (p["g"] * u + c).astype(self._dtype), "c": c, "clock": clock}

    def update_evaluations(self):
        return 1


class _Ticker(SimulationNode):
    """A clock outside the group; at half its timestep the graph is multi-rate."""

    def initial_state(self):
        return {"t": jnp.zeros((), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"t": state["t"] + jnp.asarray(dt, jnp.float32)}


@contextlib.contextmanager
def _x64(on: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


#: Parameters per scenario.  ``moving``: the forcing moves every step and
#: the spectral radius is 0.6; ``slow``: radius 0.95, so the early steps
#: exhaust the cap; ``steady``: the forcing stops moving, so the iterate
#: settles onto its fixed point and the residual onto its float floor;
#: ``nan``: a non-finite forcing.
SCENARIOS = {
    "moving": dict(g_a=0.6, g_b=0.6, b0_a=1.0, b0_b=0.5, b1_a=0.5, b1_b=-0.25),
    "slow": dict(g_a=0.95, g_b=0.95, b0_a=1.0, b0_b=0.5, b1_a=0.5, b1_b=-0.25),
    "steady": dict(g_a=0.6, g_b=0.6, b0_a=1.0, b0_b=0.5, b1_a=0.0, b1_b=0.0),
    "nan": dict(g_a=0.6, g_b=0.6, b0_a=float("nan"), b0_b=0.5, b1_a=0.5, b1_b=-0.25),
}
#: A power of two: every quantity of the group scales exactly, so the
#: rescaled run must be the unscaled one to the bit (CPL-009, CPL-010).
#: 2**-8 keeps every float16 and bfloat16 value normal.
SCALE = 2.0 ** -8


@dataclasses.dataclass
class Config:
    """One domain: dtypes, group settings and the graph's shape."""

    name: str
    dtype_a: object
    dtype_b: object
    x64: bool = False
    rtol: float = 1e-2
    tolerance: float = 1e-3
    acceleration: str = "aitken"
    extra: dict = dataclasses.field(default_factory=dict)
    multirate: bool = False
    subcycled: bool = False
    runner: str = "steps"          # "steps", "vmap", "adaptive", "restart"
    #: Run the strict_convergence scenario in this process.  The sharded
    #: domain cannot: its raise aborts the process (see its module).
    strict: bool = True

    def group(self, kind: str) -> dict:
        if kind == "main":
            g = dict(acceleration=self.acceleration, convergence_norm="mixed", rtol=self.rtol,
                     max_iterations=8)
            g.update(self.extra)
        elif kind == "plain":
            g = dict(convergence_norm="l2", tolerance=self.tolerance, max_iterations=8)
        elif kind == "single":
            g = dict(convergence_norm="l2", tolerance=self.tolerance, max_iterations=1)
        elif kind == "strict":
            g = dict(convergence_norm="l2", tolerance=self.tolerance, max_iterations=8,
                     strict_convergence=True)
        else:
            raise KeyError(kind)
        if self.subcycled:
            g.update(subcycling=True, boundary_interpolation="constant")
        return g


CONFIGS = {
    "f64": Config("f64", jnp.float64, jnp.float64, x64=True),
    "mixed_dtype": Config("mixed_dtype", jnp.float32, jnp.float64, x64=True),
    # bfloat16 resolves 2**-8, so a criterion of a percent would sit at
    # the floor: the group asks for a quarter.
    "16bit": Config("16bit", jnp.bfloat16, jnp.bfloat16, rtol=0.25, tolerance=0.25),
    "vmap": Config("vmap", jnp.float32, jnp.float32, runner="vmap"),
    "multi_rate": Config("multi_rate", jnp.float32, jnp.float32, multirate=True),
    "sub_cycled": Config("sub_cycled", jnp.float32, jnp.float32, subcycled=True),
    "predictors_warm_starts": Config(
        "predictors_warm_starts", jnp.float32, jnp.float32, acceleration="iqn-imvj",
        extra=dict(jacobian_reuse=2, predictor="quadratic")),
    "adaptive": Config("adaptive", jnp.float32, jnp.float32, runner="adaptive"),
    "checkpoint_restart": Config("checkpoint_restart", jnp.float32, jnp.float32,
                                 runner="restart"),
}


def build(cfg: Config, kind: str, diagnostics: bool) -> GraphManager:
    gm = GraphManager()
    s = SCENARIOS["moving"]
    tb = DT / 2 if cfg.subcycled else DT
    gm.add_node(_Forced("a", DT, cfg.dtype_a, g=s["g_a"], b0=s["b0_a"], b1=s["b1_a"]))
    gm.add_node(_Forced("b", tb, cfg.dtype_b, g=s["g_b"], b0=s["b0_b"], b1=s["b1_b"]))
    if cfg.dtype_a == cfg.dtype_b:
        gm.add_edge("b", "a", "x", "u")
        gm.add_edge("a", "b", "x", "u")
    else:
        # The graph refuses an edge between dtypes without a transform.
        gm.add_edge("b", "a", "x", "u", transform=lambda v: v.astype(cfg.dtype_a))
        gm.add_edge("a", "b", "x", "u", transform=lambda v: v.astype(cfg.dtype_b))
    if cfg.multirate:
        gm.add_node(_Ticker("tick", DT / 2))
    gm.add_coupling_group(["a", "b"], diagnostics=diagnostics, **cfg.group(kind))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate notice
        gm.compile()
    return gm


def params_for(gm: GraphManager, scenario: str, scale: float = 1.0) -> dict:
    """The graph's parameter pytree with *scenario*'s values, in each leaf's dtype."""
    s = SCENARIOS[scenario]
    p = jax.tree.map(lambda v: v, gm.params)
    for node in ("a", "b"):
        leaves = p["nodes"][node]
        dt = leaves["g"].dtype
        # A vector forcing (the sharded member's) varies along the vector
        # with mean one, so its mean is the scenario's value; SCALE is a
        # power of two, so every rescaled entry is exact.
        n = int(np.size(leaves["b0"]))
        profile = (np.ones(()) if n == 1 and np.ndim(leaves["b0"]) == 0
                   else 0.5 + np.arange(n) / max(n - 1, 1))
        leaves["g"] = jnp.asarray(s[f"g_{node}"], dt)
        leaves["b0"] = jnp.asarray(s[f"b0_{node}"] * scale * profile, dt)
        leaves["b1"] = jnp.asarray(s[f"b1_{node}"] * scale * profile, dt)
    return p


# ---------------------------------------------------------------------------
# What a run records, and the oracle
# ---------------------------------------------------------------------------
def _host(tree):
    return jax.tree.map(np.asarray, tree)


def _eps(dtype) -> float:
    return float(jnp.finfo(dtype).eps)      # numpy's finfo does not know bfloat16


@dataclasses.dataclass
class Record:
    """One step: the state it returned, its slots, its report and its inputs."""

    label: str            # the scenario, or "scaled"
    step: int
    state: dict           # {"a": {...}, "b": {...}}, host arrays in their dtypes
    meta: dict
    report: dict | None
    gains: tuple          # (g_a, g_b) as the step held them
    group: object         # the CouplingGroup the step was built from
    evaluations: float
    pre: dict | None = None

    @property
    def scenario(self) -> str:
        return self.label

    @property
    def finite(self) -> bool:
        return all(np.all(np.isfinite(np.asarray(v, np.float64)))
                   for n in ("a", "b") for v in self.state[n].values())

    def slot(self, name):
        v = self.meta.get(f"coupling_{KEY}_{name}")
        return None if v is None else np.asarray(v)

    def x(self) -> np.ndarray:
        """``(x_a, mean x_b)``: member ``a`` reads ``b`` through its mean, which
        for the scalar pair is ``b`` itself (the sharded module's ``b`` is a
        vector)."""
        return np.array([_mean(self.state["a"]["x"]), _mean(self.state["b"]["x"])])

    def matrix(self) -> np.ndarray:
        return np.array([[0.0, self.gains[0]], [self.gains[1], 0.0]])

    def oracle(self):
        """``(x*, (I - M)^{-1})`` in float64 from the gains and the recorded
        forcing, for ``(x_a, mean x_b)``: ``mean x_b = g_b x_a + mean c_b`` holds
        entry by entry averaged, so the pair's algebra is the scalar one."""
        inv = np.linalg.inv(np.eye(2) - self.matrix())
        c = np.array([_mean(self.state["a"]["c"]), _mean(self.state["b"]["c"])])
        return inv @ c, inv

    @property
    def eps(self) -> float:
        """The coarsest float resolution among the group's fields."""
        return max(_eps(np.asarray(v).dtype) for n in ("a", "b") for v in self.state[n].values())

    @property
    def threshold(self) -> float:
        return 1.0 if self.group.convergence_norm == "mixed" else float(self.group.tolerance)


def record_of(gm: GraphManager, label, step, gains, pre=None, *, state=None) -> Record:
    full = gm._state if state is None else state
    host = _host({n: full[n] for n in ("a", "b")})
    meta = _host(full.get("_meta", {}))
    ev, _declared, _edges = gm._committed_floor_inputs[KEY]
    measured = float(np.asarray(meta.get(f"coupling_{KEY}_pass_evaluations", np.nan)))
    evaluations = max(float(ev), measured) if math.isfinite(measured) else float(ev)
    if state is not None:
        saved, gm._state = gm._state, state
        try:
            report = gm.coupling_diagnostics().get(KEY)
        finally:
            gm._state = saved
    else:
        report = gm.coupling_diagnostics().get(KEY)
    return Record(label, step, host, meta, None if report is None else dict(report),
                  gains, gm._committed_coupling_groups[KEY], evaluations, pre)


def _gains(p) -> tuple:
    return tuple(float(np.asarray(p["nodes"][n]["g"], np.float64)) for n in ("a", "b"))


# ---------------------------------------------------------------------------
# The group's norm, its floor and its distance, restated in float64
# ---------------------------------------------------------------------------
def _mean(v) -> float:
    return float(np.mean(np.asarray(v, np.float64)))


def _f64(v) -> np.ndarray:
    return np.asarray(v, np.float64)


def _norm(group, new: dict, old: dict, weights_from_new: bool = False) -> float:
    """The group's norm of ``new - old``: each field's change over ``rtol``
    times the field's magnitude -- the larger of its two ``max |v|``, as the
    residual takes it, or with *weights_from_new* ``new``'s alone, the
    returned state's weights the bounds are stated in (MADD-ANO-146) -- the
    L2 norm the root sum of squares at ``rtol = 1``, the mixed norm the RMS
    over every active entry."""
    l2 = group.convergence_norm == "l2"
    rtol = 1.0 if l2 else float(group.rtol)
    total, count = 0.0, 0
    for n in ("a", "b"):
        for f in new[n]:
            a, b = _f64(new[n][f]), _f64(old[n][f])
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return math.inf
            ref = float(np.max(np.abs(a)))
            if not weights_from_new:
                ref = max(ref, float(np.max(np.abs(b))))
            if not ref > float(group.atol):
                continue
            total += float(np.sum((np.abs(a - b) / (rtol * ref)) ** 2))
            count += a.size
    return math.sqrt(total) if l2 else math.sqrt(total / max(count, 1))


def floor_of(r: Record) -> float:
    """``FLOOR_ULPS * m`` units of each field's own ``eps`` per entry, in the
    norm's units: ``sqrt(sum eps**2)`` under L2, the RMS of ``eps / rtol``
    under the mixed norm."""
    l2 = r.group.convergence_norm == "l2"
    rtol = 1.0 if l2 else float(r.group.rtol)
    total, count = 0.0, 0
    for n in ("a", "b"):
        for v in r.state[n].values():
            if not float(np.max(np.abs(_f64(v)))) > float(r.group.atol):
                continue
            size = np.asarray(v).size
            total += size * (_eps(np.asarray(v).dtype) / rtol) ** 2
            count += size
    if count == 0:
        return 0.0
    return FLOOR_ULPS * r.evaluations * math.sqrt(total if l2 else total / count)


def _with(r: Record, xa, xb) -> dict:
    """The returned state with the members' ``x`` replaced."""
    return {"a": {**r.state["a"], "x": _f64(xa)}, "b": {**r.state["b"], "x": _f64(xb)}}


def distance(r: Record) -> float:
    """The returned state's distance to the step's fixed point, in the group's
    norm at the returned state: each field over its own ``max |field|`` there."""
    x_star, _ = r.oracle()
    xb_star = r.gains[1] * x_star[0] + _f64(r.state["b"]["c"])
    return _norm(r.group, r.state, _with(r, x_star[0], xb_star), weights_from_new=True)


def recomputed_residual(r: Record) -> float:
    """``||F(x) - x||`` of one pass from the returned state, in the group's sweep order."""
    xa, xb_mean = r.x()
    a = r.gains[0] * xb_mean + _mean(r.state["a"]["c"])
    src = a if r.group.iteration_mode == "gauss-seidel" else xa
    b = r.gains[1] * src + _f64(r.state["b"]["c"])
    return _norm(r.group, _with(r, a, b), r.state)


# ---------------------------------------------------------------------------
# Running a domain
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DomainRun:
    """Everything one domain's checks read.

    A gradient that raised is kept as the exception, in ``errors`` under
    its family's name, and re-raised by every check that reads that family:
    a domain whose gradients cannot be taken still runs its forward checks.
    """

    cfg: Config
    main: list            # diagnostics=True, the main group, every scenario
    main_off: list        # the same group with diagnostics=False
    plain: list           # acceleration="none", the L2 norm, diagnostics=False
    single: list          # max_iterations=1, diagnostics=True
    scaled: dict          # {"main": (unscaled, scaled), "plain": (unscaled, scaled)}
    grads: list = dataclasses.field(default_factory=list)        # (main record, gradient)
    grads_plain: list = dataclasses.field(default_factory=list)  # (plain record, gradient)
    grads_single: list = dataclasses.field(default_factory=list)  # (single record, gradient)
    jvp_vs_grad: list = dataclasses.field(default_factory=list)  # (record, forward, reverse)
    cotangent: list = dataclasses.field(default_factory=list)    # (record, g, g at 2**-40)
    strict: dict = dataclasses.field(default_factory=dict)       # {"raised", "quiet"}
    errors: dict = dataclasses.field(default_factory=dict)
    #: Gradient families not yet computed, by name: each compiles its own
    #: programs, so it is taken by the first check that reads it, which
    #: spreads a domain's cost over its tests (the time budget is per test).
    lazy: dict = dataclasses.field(default_factory=dict)

    def get(self, family: str):
        if family in self.lazy:
            fn = self.lazy.pop(family)
            with _x64(self.cfg.x64):
                try:
                    setattr(self, family, fn())
                except _UNSUPPORTED_DTYPE as exc:
                    self.errors[family] = exc
        if family in self.errors:
            raise self.errors[family]
        return getattr(self, family)


def _one_step_fns(gm):
    """``x_a`` one step after a given state: the gradient of ``k * x_a`` (``k``
    traced, so one compile serves a unit and a scaled cotangent) and the
    forward-mode tangent in ``g_a``."""
    step = gm._raw_step_fn
    ext = gm._default_external_inputs()

    def xa(p, s):
        return step(s, ext, p)["a"]["x"]

    cot = jax.jit(jax.grad(lambda p, s, k: k * xa(p, s)))
    jvp = jax.jit(lambda p, s, t: jax.jvp(lambda q: xa(q, s), (p,), (t,))[1])
    return cot, jvp


def _unit(p):
    return jnp.asarray(1.0, p["nodes"]["a"]["g"].dtype)


def _tangent_on_ga(p):
    t = jax.tree.map(jnp.zeros_like, p)
    t["nodes"]["a"]["g"] = jnp.ones_like(t["nodes"]["a"]["g"])
    return t


def _as_floats(g, i=None):
    pick = (lambda v: v) if i is None else (lambda v: v[i])
    return {"g": float(np.asarray(pick(g["nodes"]["a"]["g"]), np.float64)),
            "b0": float(np.asarray(pick(g["nodes"]["a"]["b0"]), np.float64))}


def _stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


class _Steps:
    """``gm.step`` per record; with *tmp*, every run is saved after two steps,
    run on, loaded and run again, and only the steps after the load are
    recorded."""

    batched = False

    def __init__(self, tmp=None):
        self.tmp = tmp

    def run(self, gm, label, scenario, steps, scale=1.0):
        gm.reset_state()
        p = params_for(gm, scenario, scale)
        out, start = [], 0
        if self.tmp is not None:
            start = min(2, steps - 1)
            for _ in range(start):
                gm.step(params=p)
            path = self.tmp / f"{id(gm)}-{label}-{scenario}.npz"
            gm.save_state(path)
            for _ in range(start, steps):
                gm.step(params=p)
            gm.reset_state()
            gm.load_state(path)
        for k in range(start, steps):
            pre = gm._state
            gm.step(params=p)
            out.append(record_of(gm, label, k, _gains(p), pre))
        return out

    def grads(self, gm, records):
        cot, _jvp = _one_step_fns(gm)
        out = []
        for r in records:
            if _fired(r):
                p = params_for(gm, _base(r))
                out.append((r, _as_floats(cot(p, r.pre, _unit(p)))))
        return out


class _Vmap(_Steps):
    """Three states through one ``jax.vmap`` of the raw step; each member's
    report is derived from its own slots by the host code the graph uses
    for its own state."""

    batched = True

    def __init__(self):
        super().__init__(None)
        self._steps: dict = {}

    def run(self, gm, label, scenario, steps, scale=1.0):
        # One jitted vmap per graph: a fresh wrapper per call recompiles.
        if id(gm) not in self._steps:
            self._steps[id(gm)] = (gm, jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, None))))
        step = self._steps[id(gm)][1]
        ext = gm._default_external_inputs()
        gm.reset_state()
        p = params_for(gm, scenario, scale)
        members = []
        for _ in range(3):          # three phases of the same run
            members.append(gm._state)
            gm.step(params=p)
        state = _stack(members)
        out = []
        for k in range(steps):
            pre, state = state, step(state, ext, p)
            for m in range(3):
                member = jax.tree.map(lambda v, m=m: v[m], state)
                member_pre = jax.tree.map(lambda v, m=m: v[m], pre)
                out.append(record_of(gm, label, k, _gains(p), member_pre, state=member))
        return out

    def grads(self, gm, records):
        cot, _jvp = _one_step_fns(gm)
        vcot = jax.jit(jax.vmap(cot, in_axes=(None, 0, None)))
        out = []
        records = [r for r in records if _fired(r)]
        for label in dict.fromkeys(_base(r) for r in records):
            rs = [r for r in records if _base(r) == label]
            p = params_for(gm, label)
            g = vcot(p, _stack([r.pre for r in rs]), _unit(p))
            out += [(r, _as_floats(g, i)) for i, r in enumerate(rs)]
        return out


class _Adaptive:
    """``run_adaptive`` windows; gradients through ``run_adaptive_scan``."""

    batched = False
    KW = dict(dt_initial=DT, dt_max=2 * DT, dt_min=DT / 64, rtol=1e-2, atol=1e-6)

    def run(self, gm, label, scenario, steps, scale=1.0):
        gm.reset_state()
        p = params_for(gm, scenario, scale)
        out = []
        for k in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gm.run_adaptive(3 * DT, params=p, **self.KW)
            out.append(record_of(gm, label, k, _gains(p)))
        return out

    def grads(self, gm, records):
        """``d x_a`` at ``t_end`` through ``run_adaptive_scan`` on *gm* (diagnostics
        off: compiling the scan with them costs a minute), beside the record of
        the first ``run_adaptive`` window from the same start, which reaches the
        same ``t_end``.  The two steppers take one acceptance rule (CPL-134),
        so the two forwards are one state; that is asserted, not assumed."""
        out = []
        for scenario in ("moving", "steady"):
            def xa(p):
                gm.reset_state()
                final, _h, _i = gm.run_adaptive_scan(3 * DT, max_steps=16, params=p, **self.KW)
                return final["a"]["x"]

            p = params_for(gm, scenario)
            g = jax.grad(xa)(p)
            gm.reset_state()
            final, _h, _i = gm.run_adaptive_scan(3 * DT, max_steps=16, params=p, **self.KW)
            r = next(r for r in records if r.label == scenario and r.step == 0)
            for n in ("a", "b"):
                assert _bitwise(final[n]["x"], r.state[n]["x"]), (
                    f"{scenario}: run_adaptive_scan and run_adaptive reached different "
                    f"states ({final[n]['x']!r} against {r.state[n]['x']!r})")
            out.append((r, _as_floats(g)))
        return out


def _base(r: Record) -> str:
    return "moving" if r.label == "scaled" else r.label


def _fired(r: Record) -> bool:
    """Whether the step solved the group: on a multi-rate graph a base step the
    group does not fire on leaves its members (and their clocks) alone, and
    its one-step gradient is zero."""
    return r.pre is None or not _bitwise(r.pre["a"]["clock"], r.state["a"]["clock"])


def _bitwise(a, b) -> bool:
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def _strict(cfg: Config, runner, steps: int, builder) -> dict:
    gm = builder(cfg, "strict", diagnostics=False)
    out = {}
    for scenario, key in (("slow", "raised"), ("steady", "quiet")):
        try:
            runner.run(gm, scenario, scenario, steps)
            jax.block_until_ready(gm._state)
            out[key] = False
        except Exception as exc:        # equinox.error_if raises at run time
            if "strict_convergence" not in str(exc) and "max_iterations" not in str(exc):
                raise
            out[key] = True
    return out


#: What a linear solve in a dtype LAPACK has no kernel for raises: jaxlib's
#: NotImplementedError, or lineax's QR refusing the dtype with a TypeError
#: (lineax 0.1 under jax 0.11.2, CI's lane).
_UNSUPPORTED_DTYPE = (NotImplementedError, TypeError)


def _collect(run: DomainRun, family: str, fn):
    """``run.<family>``, computed by *fn* when a check first reads it; the
    exception a gradient raised is kept in ``errors`` and re-raised."""
    run.lazy[family] = fn


def _jvp_and_cotangent(run: DomainRun, runner, plain, main_off):
    """Forward mode against reverse mode (plain), and a 2**-40 cotangent
    against a unit one (main), on every firing step of ``moving``."""
    pcot, pjvp = _one_step_fns(plain)
    mcot, _mj = _one_step_fns(main_off)
    if runner.batched:
        pcot = jax.jit(jax.vmap(pcot, in_axes=(None, 0, None)))
        pjvp = jax.jit(jax.vmap(pjvp, in_axes=(None, 0, None)))
        mcot = jax.jit(jax.vmap(mcot, in_axes=(None, 0, None)))

    def batches(records):
        rs = [r for r in records if r.label == "moving" and _fired(r)]
        if runner.batched:
            return [(rs, _stack([r.pre for r in rs]))]
        return [([r], r.pre) for r in rs]

    def pick(v, i):
        return v[i] if runner.batched else v

    def jvp_vs_grad():
        out = []
        p = params_for(plain, "moving")
        for rs, pre in batches(run.plain):
            f = pjvp(p, pre, _tangent_on_ga(p))
            b = pcot(p, pre, _unit(p))["nodes"]["a"]["g"]
            out += [(r, float(np.asarray(pick(f, i), np.float64)),
                     float(np.asarray(pick(b, i), np.float64))) for i, r in enumerate(rs)]
        return out

    def cotangent():
        out = []
        p = params_for(main_off, "moving")
        k = jnp.asarray(2.0 ** -40, p["nodes"]["a"]["g"].dtype)
        for rs, pre in batches(run.main):
            g1 = mcot(p, pre, _unit(p))["nodes"]["a"]["b0"]
            gk = mcot(p, pre, k)["nodes"]["a"]["b0"] / k
            out += [(r, np.asarray(pick(g1, i)), np.asarray(pick(gk, i)))
                    for i, r in enumerate(rs)]
        return out

    _collect(run, "jvp_vs_grad", jvp_vs_grad)
    _collect(run, "cotangent", cotangent)


def domain_run(name: str, tmp_path_factory, *, cfg: Config | None = None,
               builder=None) -> DomainRun:
    """Every record and gradient one domain's checks read.  *cfg* and *builder*
    default to the domain's entry in ``CONFIGS`` and :func:`build`; the sharded
    module passes its own."""
    cfg = CONFIGS[name] if cfg is None else cfg
    builder = build if builder is None else builder
    steps = 2 * STEPS if cfg.multirate else STEPS
    with _x64(cfg.x64):
        runner = {"vmap": _Vmap(), "adaptive": _Adaptive(),
                  "restart": _Steps(tmp_path_factory.mktemp("restart")),
                  "steps": _Steps()}[cfg.runner]
        main = builder(cfg, "main", diagnostics=True)
        main_off = builder(cfg, "main", diagnostics=False)
        plain = builder(cfg, "plain", diagnostics=False)
        single = builder(cfg, "single", diagnostics=True)
        rec = {k: [] for k in ("main", "main_off", "plain", "single")}
        for scenario in SCENARIOS:
            rec["main"] += runner.run(main, scenario, scenario, steps)
            rec["main_off"] += runner.run(main_off, scenario, scenario, steps)
            rec["plain"] += runner.run(plain, scenario, scenario, steps)
            rec["single"] += runner.run(single, scenario, scenario, min(steps, 3))
        scaled = {
            "main": (runner.run(main, "moving", "moving", steps),
                     runner.run(main, "scaled", "moving", steps, SCALE)),
            "plain": (runner.run(plain, "moving", "moving", steps),
                      runner.run(plain, "scaled", "moving", steps, SCALE)),
        }
        run = DomainRun(cfg, rec["main"], rec["main_off"], rec["plain"], rec["single"], scaled)
        adaptive = cfg.runner == "adaptive"
        # The diagnostics graph's records, beside the gradients of the same
        # forwards without diagnostics (bit-identical, CPL-091).
        _collect(run, "grads", lambda: runner.grads(main_off, rec["main"]))
        _collect(run, "grads_plain", lambda: runner.grads(plain, rec["plain"]))
        if not adaptive:
            _collect(run, "grads_single", lambda: runner.grads(single, rec["single"]))
            _jvp_and_cotangent(run, runner, plain, main_off)
        if cfg.strict:
            _collect(run, "strict", lambda: _strict(cfg, runner, steps, builder))
    return run

# ---------------------------------------------------------------------------
# The claims, one check per row
# ---------------------------------------------------------------------------
CHECKS: dict = {}


def check(*rows):
    def deco(fn):
        for row in rows:
            CHECKS[row] = fn
        return fn
    return deco


def _reported(records):
    return [r for r in records if r.report is not None]


def _fail(problems, what):
    assert not problems, f"{what}:\n  " + "\n  ".join(problems[:12])


def _where(r: Record) -> str:
    return f"{r.label} step {r.step}"


@check("CPL-006")
def _single_pass_gradient(run: DomainRun):
    """max_iterations=1: the IFT gradient is the single pass's derivative,
    ``d x_a / d g_a = x_b`` at the iterate's start and ``d x_a / d b0_a = 1``."""
    problems = []
    for r, g in run.get("grads_single"):
        if not r.finite:
            continue
        xb0 = _mean(r.pre["b"]["x"])
        tol = 8 * r.eps * max(1.0, abs(xb0))
        if abs(g["g"] - xb0) > tol or abs(g["b0"] - 1.0) > 8 * r.eps:
            problems.append(f"{_where(r)}: {g} against (g={xb0}, b0=1)")
    assert run.get("grads_single"), "no single-pass gradient was taken"
    _fail(problems, "the single pass's IFT gradient is not the pass's derivative")


@check("CPL-007")
def _single_pass_verdict(run: DomainRun):
    """One pass: one iteration, no ratio, and the raw residual test decides."""
    problems = []
    for r in _reported(run.single):
        d = r.report
        raw = np.asarray(r.slot("residual"))
        verdict = bool(raw <= np.asarray(r.threshold, raw.dtype))
        if d["iterations"] != 1 or d["ratio_usable"] or d["converged"] != verdict \
                or not math.isnan(d["amplification"]):
            problems.append(f"{_where(r)}: {d}")
    assert _reported(run.single), "the single-pass group reported nothing"
    _fail(problems, "max_iterations=1 is not the raw residual test")


def _residual_problems(records):
    problems = []
    for r in _reported(records):
        if not r.finite:
            continue
        got, want = r.report["residual"], recomputed_residual(r)
        if abs(got - want) > 2 * floor_of(r) + 16 * r.eps * abs(want):
            problems.append(f"{_where(r)}: reported {got:.6e}, recomputed {want:.6e}, "
                            f"floor {floor_of(r):.3e}")
    return problems


@check("CPL-008")
def _l2_is_the_relative_norm(run: DomainRun):
    """The L2 norm divides each field's change by the field's largest magnitude."""
    assert any(r.finite for r in _reported(run.plain))
    _fail(_residual_problems(run.plain), "the L2 residual is not the documented relative norm")


@check("CPL-040")
def _mixed_is_the_rms(run: DomainRun):
    """The mixed norm is the RMS of each entry's change over rtol times its field's magnitude."""
    assert any(r.finite for r in _reported(run.main))
    _fail(_residual_problems(run.main), "the mixed residual is not the documented RMS")


@check("CPL-057")
def _residual_is_the_returned_states(run: DomainRun):
    """Recomputing ``||F(x) - x||`` on the returned state reproduces ``residual``."""
    _fail(_residual_problems(run.main) + _residual_problems(run.plain),
          "the residual is not the returned state's")


def _scaled_problems(unscaled, scaled):
    problems = []
    assert len(unscaled) == len(scaled) and unscaled
    for u, s in zip(unscaled, scaled):
        du, ds = u.report, s.report
        if (du is None) != (ds is None):
            problems.append(f"step {u.step}: a report on one side only")
            continue
        for key in ("iterations", "converged", "residual", "amplification"):
            if not _bitwise(np.float64(du[key]), np.float64(ds[key])):
                problems.append(f"step {u.step}: {key} {du[key]!r} against {ds[key]!r}")
        for n in ("a", "b"):
            want = np.asarray(u.state[n]["x"]) * np.asarray(SCALE, np.asarray(u.state[n]["x"]).dtype)
            if not _bitwise(want, s.state[n]["x"]):
                problems.append(f"step {u.step}: {n}.x scaled {s.state[n]['x']!r}, want {want!r}")
    return problems


@check("CPL-009")
def _verdict_in_any_units(run: DomainRun):
    """A power-of-two rescaling leaves passes, verdict and residual bit-identical (none, L2)."""
    _fail(_scaled_problems(*run.scaled["plain"]), "the unaccelerated group depends on its units")


@check("CPL-010")
def _accelerated_in_any_units(run: DomainRun):
    """The same under the domain's accelerator and the mixed norm."""
    _fail(_scaled_problems(*run.scaled["main"]), "the accelerated group depends on its units")


@check("CPL-024")
def _edges_read_the_current_iterate(run: DomainRun):
    """A converged state is at the implicit fixed point ``x = M x + c``, within
    the spectral bound, and nearer it than the staggered update is."""
    problems, seen = [], 0
    for r in _reported(run.main):
        d = r.report
        if not (d["converged"] and r.finite and d["spectral_usable"]):
            continue
        seen += 1
        dist = distance(r)
        if dist > d["spectral_error_bound"]:
            problems.append(f"{_where(r)}: distance {dist:.3e} > bound {d['spectral_error_bound']:.3e}")
        if r.pre is not None and r.label == "moving" and r.step > 0 and _fired(r):
            stag_a = r.gains[0] * _mean(r.pre["b"]["x"]) + _mean(r.state["a"]["c"])
            stag_b = r.gains[1] * _mean(r.pre["a"]["x"]) + _f64(r.state["b"]["c"])
            if _norm(r.group, r.state, _with(r, stag_a, stag_b)) <= dist:
                problems.append(f"{_where(r)}: no nearer the implicit fixed point than "
                                "the staggered update")
    assert seen, "no converged step with a usable bound"
    _fail(problems, "a converged step is not at the implicit fixed point")


def _ift_at(r: Record) -> dict:
    """The IFT rule at the returned iterate: ``(I - M)^{-1} dF/d theta (x_k)``."""
    _, inv = r.oracle()
    x = r.x()
    return {"g": float((inv @ np.array([x[1], 0.0]))[0]), "b0": float(inv[0, 0])}


def _star(r: Record) -> dict:
    """The fixed point's own sensitivity."""
    x_star, inv = r.oracle()
    return {"g": float((inv @ np.array([x_star[1], 0.0]))[0]), "b0": float(inv[0, 0])}


def _cond(r: Record) -> float:
    return float(np.linalg.norm(r.oracle()[1], 2))


def _gradient_problems(pairs, want, only_converged=False):
    problems, seen = [], 0
    for r, g in pairs:
        if not r.finite or (only_converged and not (r.report or {}).get("converged", True)):
            continue
        seen += 1
        ref = want(r)
        tol = 64 * r.eps * _cond(r)
        for key in ("g", "b0"):
            if not abs(g[key] - ref[key]) <= tol * max(1.0, abs(ref[key])):
                problems.append(f"{_where(r)} d/d{key}: {g[key]!r} against {ref[key]!r}")
    assert seen, "no gradient to check"
    return problems


@check("CPL-030")
def _forward_and_reverse_modes(run: DomainRun):
    """jvp and grad through the step agree, and both are the IFT rule at the iterate."""
    problems = []
    if run.cfg.runner == "adaptive":
        problems += _gradient_problems(run.get("grads"), _ift_at)
    for r, fwd, rev in run.get("jvp_vs_grad"):
        tol = 64 * r.eps * _cond(r) * max(1.0, abs(rev))
        want = _ift_at(r)["g"]
        if abs(fwd - rev) > tol or abs(rev - want) > tol:
            problems.append(f"{_where(r)}: jvp {fwd!r}, grad {rev!r}, IFT {want!r}")
    assert run.get("jvp_vs_grad") or run.cfg.runner == "adaptive"
    _fail(problems, "forward and reverse mode disagree or miss the IFT rule")


@check("CPL-033")
def _strict_convergence(run: DomainRun):
    """strict_convergence raises on a capped, unconverged solve and not on a converged one."""
    strict = run.get("strict")
    assert strict == {"raised": True, "quiet": False}, strict


@check("CPL-051")
def _converged_is_the_estimate_against_the_threshold(run: DomainRun):
    """``converged``: the estimate ``residual * max(omega * amp, 1)``, taken in the
    residual's dtype, meets the threshold rounded to it."""
    problems = []
    for r in _reported(run.main + run.plain + run.single):
        res = r.slot("residual")
        amp = np.asarray(r.slot("amplification"), res.dtype)
        omega = np.asarray(float(r.group.relaxation) if r.group.acceleration == "fixed" else 1.0,
                           res.dtype)
        with np.errstate(all="ignore"):
            est = res * np.maximum(omega * amp, np.ones_like(amp))
            verdict = bool(est <= np.asarray(r.threshold, res.dtype))
        d = r.report
        if d["converged"] != verdict or not _bitwise(np.float64(d["error_estimate"]),
                                                     np.float64(est)):
            problems.append(f"{_where(r)}: converged={d['converged']} estimate="
                            f"{d['error_estimate']!r}; recomputed {verdict} {float(est)!r}")
    _fail(problems, "converged is not the estimate against the threshold")


@check("CPL-053")
def _unconverged_means_capped_or_not_finite(run: DomainRun):
    problems = []
    for r in _reported(run.main + run.plain):
        d = r.report
        if not d["converged"] and r.finite and d["iterations"] != r.group.max_iterations:
            problems.append(f"{_where(r)}: unconverged at {d['iterations']} of "
                            f"{r.group.max_iterations} on a finite state")
    assert any(not r.report["converged"] for r in _reported(run.main)), "nothing unconverged"
    _fail(problems, "converged=False with neither the cap nor a non-finite state")


@check("CPL-055")
def _ratio_usable(run: DomainRun):
    """ratio_usable is the amplification slot's validity, and nothing else."""
    problems = []
    for r in _reported(run.main + run.plain + run.single):
        d, amp = r.report, float(r.slot("amplification"))
        usable = amp >= 1.0
        if d["ratio_usable"] != usable:
            problems.append(f"{_where(r)}: ratio_usable={d['ratio_usable']}, slot {amp}")
        elif usable and d["amplification"] != amp:
            problems.append(f"{_where(r)}: amplification {d['amplification']} != slot {amp}")
        elif not usable and not (math.isnan(d["amplification"])
                                 and d["error_estimate"] == d["residual"]):
            problems.append(f"{_where(r)}: a rejected ratio still feeds the estimate: {d}")
    if any(r.report["ratio_usable"] for r in _reported(run.single)):
        problems.append("a single pass reported a usable ratio")
    _fail(problems, "ratio_usable is not the slot's validity")


@check("CPL-060")
def _iterations_is_a_cap_check(run: DomainRun):
    problems = []
    for r in _reported(run.main + run.plain):
        it, cap = r.report["iterations"], r.group.max_iterations
        if not 1 <= it <= cap:
            problems.append(f"{_where(r)}: iterations {it} outside [1, {cap}]")
        if not r.report["converged"] and r.finite and it != cap:
            problems.append(f"{_where(r)}: unconverged at {it} of {cap}")
    capped = [r for r in _reported(run.main + run.plain) if r.finite
              and r.report["iterations"] == r.group.max_iterations]
    assert capped, "no finite step reached the cap: the fixture premise"
    _fail(problems, "iterations is not a cap check")


@check("CPL-083")
def _derived_keys_follow_their_slots(run: DomainRun):
    problems = []
    for r in _reported(run.main + run.plain + run.single):
        d = r.report
        if math.isnan(d["amplification"]) == d["ratio_usable"]:
            problems.append(f"{_where(r)}: amplification {d['amplification']} with "
                            f"ratio_usable={d['ratio_usable']}")
        g = d["gradient_error_estimate"]
        if d["ratio_usable"] and g != d["error_estimate"]:
            problems.append(f"{_where(r)}: gradient_error_estimate {g} != {d['error_estimate']}")
        if not d["ratio_usable"] and g != math.inf:
            problems.append(f"{_where(r)}: gradient_error_estimate {g} without a ratio")
        if not math.isfinite(d["residual"]) and d["precision_limited"]:
            problems.append(f"{_where(r)}: a non-finite residual read precision_limited")
    _fail(problems, "a derived key does not follow its slots")


@check("CPL-084")
def _a_non_finite_state(run: DomainRun):
    """residual is inf on a non-finite state, and the verdict is False."""
    nan = [r for r in _reported(run.main + run.plain + run.single) if r.label == "nan"]
    assert nan and all(not r.finite for r in nan), "the nan scenario stayed finite"
    bad = [f"{_where(r)}: {r.report}" for r in nan
           if r.report["residual"] != math.inf or r.report["converged"]]
    _fail(bad, "a non-finite state did not read residual=inf, converged=False")


def _spectral_tolerance(r: Record) -> float:
    """The resolution ``rho_spectral`` is documented to: "exact ... to
    float32" for a float32 or wider group (64 float32 ``eps``; a float64
    group's reads ~2e-8 off, measured), and for a bfloat16 or float16 group
    "to about one ``eps`` of that dtype times the Jacobian's norm": its
    analysis and its slot are float32 (CPL-072, CPL-087), but the
    Jacobian-vector products are the map's own, rounded to its dtype."""
    if r.eps > _eps(np.float32):
        return r.eps * max(1.0, float(np.linalg.norm(pass_jacobian(r), 2)))
    return 64 * _eps(np.float32)


def pass_jacobian(r: Record) -> np.ndarray:
    """One pass's Jacobian ``dF/dx`` in ``(x_a, x_b)``: ``M`` under Jacobi; under
    Gauss-Seidel ``b`` reads the updated ``x_a``, so its row is ``g_b`` times
    ``a``'s."""
    m = r.matrix()
    if r.group.iteration_mode == "gauss-seidel":
        return np.array([m[0], m[1, 0] * m[0]])
    return m


def pass_radius(r: Record) -> float:
    """The spectral radius of one pass's Jacobian ``dF/dx``.  A Gauss-Seidel
    pass reads the updated ``x_a``, so ``dF/dx = [[0, g_a], [0, g_a g_b]]``
    and the radius is ``|g_a g_b|``; a Jacobi pass is ``M`` itself, radius
    ``sqrt|g_a g_b|``."""
    prod = abs(r.gains[0] * r.gains[1])
    return prod if r.group.iteration_mode == "gauss-seidel" else math.sqrt(prod)


@check("CPL-087")
def _rho_spectral(run: DomainRun):
    """Eight Arnoldi steps resolve a rank-two Jacobian: ``rho_spectral`` is the
    pass map's spectral radius where usable; NaN where nothing was computed."""
    problems, seen = [], 0
    for r in _reported(run.main):
        d = r.report
        if not d["spectral_usable"]:
            continue
        seen += 1
        rho = pass_radius(r)
        tol = _spectral_tolerance(r) + 2 * float(r.slot("spectral_residual"))
        if abs(d["rho_spectral"] - rho) > tol:
            problems.append(f"{_where(r)}: rho_spectral {d['rho_spectral']!r}, true {rho!r}")
    for r in _reported(run.single):
        if not math.isnan(r.report["rho_spectral"]):
            problems.append(f"{_where(r)} (one pass): rho_spectral {r.report['rho_spectral']}")
    assert seen, "no usable spectrum"
    _fail(problems, "rho_spectral is not the Jacobian's spectral radius")


@check("CPL-088")
def _spectral_bound_holds(run: DomainRun):
    problems, seen = [], 0
    for r in _reported(run.main):
        d = r.report
        if not (d["spectral_usable"] and r.finite):
            continue
        seen += 1
        dist = distance(r)
        if not d["spectral_error_bound"] >= dist:
            problems.append(f"{_where(r)}: bound {d['spectral_error_bound']:.4e} below the "
                            f"distance {dist:.4e}")
    assert seen, "no usable spectral bound"
    _fail(problems, "a usable spectral bound is below the true distance")


@check("CPL-089")
def _spectral_bound_formula(run: DomainRun):
    """``(residual + floor) * max(amp, 1/(1 - rho_safe))``: never below
    ``residual + floor`` where finite, inf where ``rho_safe >= 1``, NaN where
    ``rho`` is NaN."""
    problems = []
    for r in _reported(run.main + run.single):
        d = r.report
        rho = float(r.slot("rho_spectral")) if r.slot("rho_spectral") is not None else math.nan
        res = float(r.slot("spectral_residual")) if r.slot("spectral_residual") is not None \
            else math.nan
        bound = d["spectral_error_bound"]
        if math.isnan(rho) or not math.isfinite(d["residual"]):
            if not math.isnan(bound):
                problems.append(f"{_where(r)}: bound {bound} where nothing was computed")
            continue
        if rho + 2 * res >= 1.0:
            if bound != math.inf:
                problems.append(f"{_where(r)}: bound {bound} at rho_safe {rho + 2 * res}")
            continue
        floor = floor_of(r)
        if not bound >= (d["residual"] + floor) * (1 - 4 * _eps(np.float32)):
            problems.append(f"{_where(r)}: bound {bound:.4e} below residual + floor "
                            f"{d['residual'] + floor:.4e}")
    _fail(problems, "the spectral bound does not follow its formula")


@check("CPL-091")
def _diagnostics_leave_the_forward_alone(run: DomainRun):
    problems = []
    assert len(run.main) == len(run.main_off)
    for on, off in zip(run.main, run.main_off):
        for n in ("a", "b"):
            for f in ("x", "c", "clock"):
                if not _bitwise(on.state[n][f], off.state[n][f]):
                    problems.append(f"{_where(on)}: {n}.{f} {on.state[n][f]!r} != {off.state[n][f]!r}")
        for s in ("iterations", "residual", "amplification"):
            if not _bitwise(on.slot(s), off.slot(s)):
                problems.append(f"{_where(on)}: slot {s} {on.slot(s)!r} != {off.slot(s)!r}")
    _fail(problems, "diagnostics=True moved the forward")


@check("CPL-092")
def _spectral_usable(run: DomainRun):
    """Usable means a finite bound from a settled Arnoldi space (residual at
    most 5% of ``1 - rho``); never where nothing was computed."""
    problems = []
    for r in _reported(run.main + run.single):
        d = r.report
        if d["spectral_usable"]:
            rho, res = float(r.slot("rho_spectral")), float(r.slot("spectral_residual"))
            if not (math.isfinite(d["spectral_error_bound"]) and res <= 0.05 * (1 - rho)):
                problems.append(f"{_where(r)}: usable with bound {d['spectral_error_bound']}, "
                                f"residual {res}, rho {rho}")
    for r in _reported(run.main):
        d = r.report
        # Both members declare their evaluation count, so a resolved
        # rank-two spectrum well inside the unit circle is usable wherever
        # it was computed.
        if r.finite and not d["spectral_usable"] and math.isfinite(d["spectral_error_bound"]) \
                and d["rho_spectral"] < 0.9:
            problems.append(f"{_where(r)}: a resolved rank-two spectrum read unusable: {d}")
    if any(r.report["spectral_usable"] for r in _reported(run.single)):
        problems.append("a single pass read spectral_usable")
    if any(r.report["spectral_usable"] for r in _reported(run.main) if not r.finite):
        problems.append("a non-finite state read spectral_usable")
    _fail(problems, "spectral_usable does not read its conditions")


@check("CPL-093")
def _gradient_bound_holds(run: DomainRun):
    """``|g_k - g*| <= bound * |g_k|`` for the gradient in the multiplicative ``g_a``."""
    problems, seen = [], 0
    for r, g in run.get("grads"):
        d = r.report
        if not (d and d["gradient_bound_usable"] and r.finite):
            continue
        seen += 1
        star = _star(r)["g"]
        true = abs(g["g"] - star) / abs(g["g"]) if g["g"] else math.inf
        slack = 64 * r.eps * _cond(r)
        if not true <= d["gradient_relative_error_bound"] + slack:
            problems.append(f"{_where(r)}: relative error {true:.4e} above the bound "
                            f"{d['gradient_relative_error_bound']:.4e}")
    assert seen, "no usable gradient bound"
    _fail(problems, "a usable gradient bound is below the true error")


@check("CPL-094")
def _additive_parameter_is_exact_from_any_iterate(run: DomainRun):
    """In an additive parameter of a map affine in its state the IFT gradient is
    the fixed point's from any iterate, converged or not."""
    pairs = [(r, {"g": _star(r)["g"], "b0": g["b0"]}) for r, g in run.get("grads")]
    _fail(_gradient_problems(pairs, _star), "the additive gradient depends on the iterate")


@check("CPL-095")
def _gradient_bound_usable(run: DomainRun):
    problems = []
    for r in _reported(run.main + run.single):
        d = r.report
        want = bool(d["spectral_usable"] and math.isfinite(d["gradient_relative_error_bound"]))
        if d["gradient_bound_usable"] != want:
            problems.append(f"{_where(r)}: gradient_bound_usable={d['gradient_bound_usable']}")
    for r in _reported(run.single):
        if not math.isnan(r.report["gradient_relative_error_bound"]):
            problems.append(f"{_where(r)} (one pass): bound {r.report['gradient_relative_error_bound']}")
    _fail(problems, "gradient_bound_usable does not read its conditions")


@check("CPL-097")
def _precision_limited(run: DomainRun):
    """precision_limited is the residual at or below the floor recomputed from the state."""
    problems, limited = [], 0
    for r in _reported(run.main + run.plain + run.single):
        d, floor = r.report, floor_of(r)
        want = floor > 0.0 and math.isfinite(d["residual"]) and d["residual"] <= floor
        limited += want
        if d["precision_limited"] != want and abs(d["residual"] - floor) > 1e-3 * floor:
            problems.append(f"{_where(r)}: precision_limited={d['precision_limited']} at "
                            f"residual {d['residual']:.3e}, floor {floor:.3e}")
    assert limited, "no step reached its floor: the fixture premise"
    _fail(problems, "precision_limited is not the residual against its floor")


@check("CPL-100")
def _the_floor_formula(run: DomainRun):
    """The floor is 4 m eps per entry in the norm's units, each field at its own dtype."""
    problems = []
    for r in _reported(run.main + run.plain):
        if not r.finite:
            continue
        g = r.group
        state = {n: {f: jnp.asarray(v) for f, v in r.state[n].items()} for n in ("a", "b")}
        lib = residual_precision_floor(state, ["a", "b"], g.convergence_norm, g.atol,
                                       g.rtol, (), evaluations=r.evaluations)
        mine = floor_of(r)
        # The library takes the floor in the fields' own dtype (bfloat16
        # rounds it to 2**-8 relative); the formula is restated in float64.
        if not abs(float(lib) - mine) <= 4 * _eps(lib.dtype) * mine:
            problems.append(f"{_where(r)}: floor {float(lib)!r}, formula {mine!r}")
    _fail(problems, "the floor is not the documented formula")


@check("CPL-140", "CPL-146")
def _ift_gradient(run: DomainRun):
    """The gradient is the IFT rule linearised at the returned iterate (CPL-140)
    and it is finite on the default path (CPL-146)."""
    problems = _gradient_problems(run.get("grads"), _ift_at) + _gradient_problems(run.get("grads_plain"), _ift_at)
    for r, g in run.get("grads") + run.get("grads_plain"):
        if r.finite and not all(math.isfinite(v) for v in g.values()):
            problems.append(f"{_where(r)}: a non-finite gradient {g}")
    _fail(problems, "the IFT gradient is not the rule at the returned iterate")


@check("CPL-141", "CPL-142")
def _additive_gradient_under_every_acceleration(run: DomainRun):
    """In an additive parameter of an affine map the IFT gradient is the same,
    to the bit, under the domain's accelerator (and its predictor) and under
    none: the rule is F's, whatever path reached the iterate."""
    problems = []
    by = {}
    for r, g in run.get("grads_plain"):
        if r.finite:
            by.setdefault(r.label, set()).add(np.float64(g["b0"]).tobytes())
    for r, g in run.get("grads"):
        if r.finite and r.label in by and np.float64(g["b0"]).tobytes() not in by[r.label]:
            seen = sorted(float(np.frombuffer(b)[0]) for b in by[r.label])
            problems.append(f"{_where(r)}: d/db0 {g['b0']!r} not among the unaccelerated "
                            f"group's {seen}")
    assert by, "no gradient to compare"
    _fail(problems, "the additive gradient depends on the accelerator")


@check("CPL-143")
def _cotangent_scale(run: DomainRun):
    """The adjoint's criterion is relative: a cotangent scaled by 2**-40 gives
    the same gradient, to the bit, once divided back."""
    problems = [f"{_where(r)}: {a!r} against {b!r}" for r, a, b in run.get("cotangent")
                if r.finite and not _bitwise(a, b)]
    if run.cfg.runner != "adaptive":
        assert run.get("cotangent"), "no cotangent compared"
    _fail(problems, "a scaled cotangent changes the gradient")


# ---------------------------------------------------------------------------
# The tests: one per domain, parametrised by the rows it covers
# ---------------------------------------------------------------------------
ROWS = sorted(CHECKS)
#: Rows a domain cannot express, with the reason; every other row runs.
SKIP: dict = {
    # run_adaptive's steps are the stepper's, not one compiled step from a
    # recorded state: the single-pass closed form and the jvp of one step
    # have no stepper analogue, and run_adaptive_scan's gradients stand in.
    "adaptive": {"CPL-006", "CPL-143"},
    # strict_convergence's raise aborts a process stepping a sharded member;
    # the sharded module checks it in a subprocess.
    "sharded": {"CPL-033"},
}
_RUNS: dict = {}


def _run(name, tmp_path_factory) -> DomainRun:
    if name not in _RUNS:
        try:
            _RUNS[name] = domain_run(name, tmp_path_factory)
        except Exception as exc:        # one failed build fails every row, once
            _RUNS[name] = exc
    if isinstance(_RUNS[name], Exception):
        raise _RUNS[name]
    return _RUNS[name]


#: Rows checked in some domains only: CPL-142 is the predictor's initial
#: guess, which only the predictor domain varies.
ONLY: dict = {"CPL-142": {"predictors_warm_starts"}}


#: The cells where the tree does not meet the claim: ``(domain, row) ->
#: (exception, reason)``.  Each is a strict xfail whose reason starts with
#: the row's id, and the row is ``failing`` in the inventory, its finding
#: carrying the reproducer.  Empty since the 16-bit adjoint solve
#: (MADD-ANO-161), the 16-bit spectral slots (CPL-087) and the adaptive
#: scan's gradient at an exact error estimate (MADD-ANO-160) were fixed.
KNOWN_FAILING: dict = {}


def _rows(domain):
    out = []
    for r in ROWS:
        if r in SKIP.get(domain, set()) or domain not in ONLY.get(r, {domain}):
            continue
        if (domain, r) in KNOWN_FAILING:
            exc, reason = KNOWN_FAILING[(domain, r)]
            out.append(pytest.param(r, marks=pytest.mark.xfail(strict=True, raises=exc,
                                                               reason=reason)))
        else:
            out.append(r)
    return out


def _check(domain, row, tmp_path_factory):
    run = _run(domain, tmp_path_factory)
    with _x64(run.cfg.x64):
        CHECKS[row](run)


@pytest.mark.parametrize("row", _rows("f64"))
def test_the_claim_holds_in_float64(row, tmp_path_factory):
    """Under ``jax_enable_x64``, both members float64."""
    _check("f64", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("mixed_dtype"))
def test_the_claim_holds_with_a_float32_member_under_x64(row, tmp_path_factory):
    """Under x64, member ``a`` float32 and member ``b`` float64."""
    _check("mixed_dtype", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("16bit"))
def test_the_claim_holds_on_a_sixteen_bit_group(row, tmp_path_factory):
    """Both members bfloat16."""
    _check("16bit", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("vmap"))
def test_the_claim_holds_for_each_member_of_a_vmapped_step(row, tmp_path_factory):
    """Three states through one ``jax.vmap`` of the step, each member checked."""
    _check("vmap", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("multi_rate"))
def test_the_claim_holds_on_a_multirate_graph(row, tmp_path_factory):
    """The group fires on every other base step; between firings its report
    and its members' state are the last applied solve's."""
    _check("multi_rate", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("sub_cycled"))
def test_the_claim_holds_on_a_subcycled_group(row, tmp_path_factory):
    """``subcycling=True``, member ``b`` sub-stepped twice per pass."""
    _check("sub_cycled", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("predictors_warm_starts"))
def test_the_claim_holds_with_a_predictor_and_a_warm_start(row, tmp_path_factory):
    """``predictor="quadratic"``, ``acceleration="iqn-imvj"``, ``jacobian_reuse=2``."""
    _check("predictors_warm_starts", row, tmp_path_factory)


# Slow: run_adaptive compiles its dt-parameterised step again on every call
# (with diagnostics, ~2 s each on three cores), and the gradients compile
# run_adaptive_scan and its transpose: ~60 s for the domain.  The adaptive
# report and its verdict stay on every push in the adaptive steppers' own
# tests.
# Per push: tests/core/test_adaptive_report_covers_both_kept_half_steps.py::test_the_report_and_strict_convergence_give_one_verdict
# tests/core/test_adaptive_strict_convergence_checks_kept_solves.py::test_run_adaptive_scan_completes_when_every_kept_solve_converges
@pytest.mark.slow
@pytest.mark.parametrize("row", _rows("adaptive"))
def test_the_claim_holds_after_run_adaptive(row, tmp_path_factory):
    """The report after ``run_adaptive``; gradients through ``run_adaptive_scan``."""
    _check("adaptive", row, tmp_path_factory)


@pytest.mark.parametrize("row", _rows("checkpoint_restart"))
def test_the_claim_holds_after_a_checkpoint_restart(row, tmp_path_factory):
    """Every run saved after two steps, run on, loaded and run again."""
    _check("checkpoint_restart", row, tmp_path_factory)
