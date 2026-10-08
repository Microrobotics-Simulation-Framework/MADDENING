"""The lattice-plane table of a group with a geometry edge: an independent
reference of its pass, a constructive placement of the fixed point beside a
lattice plane, and the side of that plane each point the diagnostics rest on
is on.

**The pair.**  A grid-side member ``F`` and a point-side member ``P`` joined
by a ``multilinear_grid`` gather (``F -> P``) and scatter (``P -> F``), each
reading the positions of its source from the iterate or of its target from
the pre-step state (``anchors``), the positions moving with the member's
mapped input (``kP``, ``kF``) or not.

**The reference is independent of** ``src/``: a NumPy float64 multilinear
kernel of its own, written to take complex positions and fields, so the
pass's Jacobian -- the dependence through the positions included -- is a
complex-step derivative, exact to rounding inside a lattice cell.  Its
fixed point is the plain iteration's from the returned iterate, finished by
Newton on the polynomial piece it is in.  Only :func:`build` imports the
library: it makes the graph under test.

**The placement is constructive** (:func:`placed_on_a_plane`): for a drawn
pair the initial position of one marker is bisected until the *fixed point*
has that marker on a lattice plane; a shift of that initial position then
puts the fixed point at a chosen signed distance either side of it, from a
few float resolutions to a tenth of a spacing.  Nothing is hoped to drift
into a window that can be 1e-12 of a spacing wide.

**The sides** (:func:`sides`).  Three points matter to the gradient's flag:
the returned iterate ``x_k``, the Newton point ``x_N`` (the other point the
Newton-Kantorovich check takes a Jacobian at) and the fixed point ``x*``.
Relative to the fixed point's lattice cells the other two are each in them
or not: four rows.  ``k!=*`` with ``N=*`` is the case of MADD-ANO-242 (the
check sees the plane), ``k!=*`` with ``N!=*`` the case of MADD-ANO-251 (it
cannot), ``k=*`` with ``N!=*`` a Newton point that overshoots a plane the
other two are short of.  The positions the pass *builds* at the iterate
(what a Gauss-Seidel sweep reads after its holder's update) are a fourth
point, reported beside the row.

Lifted from the reproducers of the audit that found MADD-ANO-251
(``benchmarks/results/audit_040_p4_17/mapping/repro_reference.py`` and
``repro_scan_plane_distance.py``).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

#: A position within this many float resolutions of a lattice plane is on
#: it (``_bounds.GEOMETRY_PLANE_ULPS``, restated: the reference does not
#: read the library's constant).
PLANE_RESOLUTIONS = 8.0


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclasses.dataclass
class Cfg:
    N: tuple = (6,)              # lattice shape (1 or 2 axes)
    m: int = 2                   # markers
    origin: tuple = (0.0,)
    spacing: tuple = (0.5,)
    dtype: str = "float32"
    anchors: tuple = ("target", "source")   # (gather anchor, scatter anchor)
    # F: x' = aF*x + bF*deposit + sF ; F.pos' = F.pos + vF + kF*deposit[sel]
    aF: float = 0.5
    bF: float = 0.1
    # P: x' = aP*x + bP*sampled + cP ; P.pos' = P.pos + vP + kP*sampled
    aP: float = 0.5
    bP: float = 1.0
    cP: float = 0.0
    vP: tuple = (0.0,)           # per-axis drift of P.pos
    kP: float = 0.0              # P.pos' += kP * sampled (first axis)
    vF: tuple = (0.0,)
    kF: float = 0.0
    quad: float = 0.0            # smooth nonlinearity in P.x: + quad*sampled**2
    # initial state
    xF0: tuple = ()
    xP0: tuple = ()
    posP0: tuple = ()            # (m, d)
    posF0: tuple = ()
    # group
    order: tuple = ("F", "P")    # add_node order
    mode: str = "gauss-seidel"
    norm: str = "l2"
    tolerance: float = 1e-4
    rtol: float = 1e-4
    max_iterations: int = 60
    acceleration: str = "none"
    relaxation: float = 1.0
    flux: bool = False           # P computes a flux (unused by anyone)
    predictor: str = "none"
    sweep: tuple = ()            # the order the group sweeps its members (found empirically)
    group: bool = True           # False: no coupling group (a plain step; the back edge is staggered)

    @property
    def d(self):
        return len(self.N)

    @property
    def size(self):
        return int(np.prod(self.N))

    def holds(self, node):
        """Does *node* hold a 'pos' field?"""
        g, s = self.anchors
        if node == "P":
            return g == "target" or s == "source"
        return g == "source" or s == "target"


# --------------------------------------------------------------------------
# independent reference kernel (numpy, complex-capable)
# --------------------------------------------------------------------------
def _stencil(cfg: Cfg, pos):
    """indices (m, 2**d) and weights (m, 2**d) of own multilinear stencil."""
    pos = np.asarray(pos).reshape(cfg.m, cfg.d)
    lo, hi, fr = [], [], []
    for a in range(cfg.d):
        n = cfg.N[a]
        u = (pos[:, a] - cfg.origin[a]) / cfg.spacing[a]
        ur = u.real
        top = n - 1
        cl = np.where(ur < 0, 0.0, np.where(ur > top, float(top), u))
        base = np.clip(np.floor(cl.real), 0, max(n - 2, 0))
        fr.append(cl - base)
        i0 = base.astype(int)
        lo.append(i0)
        hi.append(np.minimum(i0 + 1, n - 1))
    strides = [int(np.prod(cfg.N[a + 1:])) for a in range(cfg.d)]
    idx, w = [], []
    for corner in itertools.product((0, 1), repeat=cfg.d):
        flat = np.zeros(cfg.m, int)
        ww = np.ones(cfg.m, dtype=pos.dtype)
        for a in range(cfg.d):
            flat = flat + strides[a] * (hi[a] if corner[a] else lo[a])
            ww = ww * (fr[a] if corner[a] else (1 - fr[a]))
        idx.append(flat)
        w.append(ww)
    return np.stack(idx, 1), np.stack(w, 1)


def ref_gather(cfg, f, pos):
    idx, w = _stencil(cfg, pos)
    return np.sum(w * np.asarray(f)[idx], axis=1)


def ref_scatter(cfg, y, pos):
    idx, w = _stencil(cfg, pos)
    y = np.asarray(y)
    out = np.zeros(cfg.size, dtype=np.result_type(w.dtype, y.dtype))
    np.add.at(out, idx, w * y[:, None])
    return out


def cells_of(cfg, pos):
    """Lattice cell signature of every coordinate (which polynomial piece)."""
    pos = np.asarray(pos).real.reshape(cfg.m, cfg.d)
    out = []
    for a in range(cfg.d):
        n = cfg.N[a]
        u = (pos[:, a] - cfg.origin[a]) / cfg.spacing[a]
        c = np.where(u < 0, -1, np.where(u > n - 1, n, np.clip(np.floor(u), 0, max(n - 2, 0))))
        out.append(c.astype(int))
    return np.stack(out, 1)


def plane_distance(cfg, pos):
    pos = np.asarray(pos).real.reshape(cfg.m, cfg.d)
    out = []
    for a in range(cfg.d):
        n = cfg.N[a]
        u = (pos[:, a] - cfg.origin[a]) / cfg.spacing[a]
        fr = u - np.floor(u)
        inside = np.minimum(fr, 1 - fr)
        dist = np.where(u < 0, -u, np.where(u > n - 1, u - (n - 1), inside))
        out.append(dist * cfg.spacing[a])
    return np.stack(out, 1)


# --------------------------------------------------------------------------
# the reference pass
# --------------------------------------------------------------------------
def fields(cfg: Cfg):
    """[(node, field, size)] in a fixed order of my own."""
    out = [("F", "x", cfg.size)]
    if cfg.holds("F"):
        out.append(("F", "pos", cfg.m * cfg.d))
    out.append(("P", "x", cfg.m))
    if cfg.holds("P"):
        out.append(("P", "pos", cfg.m * cfg.d))
    return out


def pack(cfg, st):
    return np.concatenate([np.ravel(np.asarray(st[n][f])) for n, f, _ in fields(cfg)])


def unpack(cfg, x):
    out, k = {"F": {}, "P": {}}, 0
    for n, f, s in fields(cfg):
        out[n][f] = x[k:k + s]
        k += s
    return out


def field_ids(cfg):
    return np.concatenate([np.full(s, i) for i, (_n, _f, s) in enumerate(fields(cfg))])


def _sel(cfg):
    return np.arange(cfg.m) % cfg.size


def _upd_F(cfg, pre, deposit):
    new = {"x": cfg.aF * pre["x"] + cfg.bF * deposit}
    if cfg.holds("F"):
        drift = np.zeros((cfg.m, cfg.d), dtype=complex)
        drift = drift + np.asarray(cfg.vF)[None, :]
        drift[:, 0] = drift[:, 0] + cfg.kF * deposit[_sel(cfg)]
        new["pos"] = pre["pos"] + drift.ravel()
    return new


def _upd_P(cfg, pre, sampled):
    new = {"x": cfg.aP * pre["x"] + cfg.bP * sampled + cfg.cP + cfg.quad * sampled * sampled}
    if cfg.holds("P"):
        drift = np.zeros((cfg.m, cfg.d), dtype=complex)
        drift = drift + np.asarray(cfg.vP)[None, :]
        drift[:, 0] = drift[:, 0] + cfg.kP * sampled
        new["pos"] = pre["pos"] + drift.ravel()
    return new


def ref_pass(cfg: Cfg, x, c, sweep=None):
    """One pass of the group: x the incoming iterate (flat), c the pre-step
    state (flat).  Time levels as the algorithm guide's table."""
    it, pre = unpack(cfg, x), unpack(cfg, c)
    g_anchor, s_anchor = cfg.anchors
    sweep = sweep or cfg.sweep or cfg.order
    cur = {"F": dict(it["F"]), "P": dict(it["P"])}      # what a reader sees

    def do_F(view):
        geom = view["P"]["pos"] if s_anchor == "source" else pre["F"]["pos"]
        return _upd_F(cfg, pre["F"], ref_scatter(cfg, view["P"]["x"], geom))

    def do_P(view):
        geom = view["F"]["pos"] if g_anchor == "source" else pre["P"]["pos"]
        return _upd_P(cfg, pre["P"], ref_gather(cfg, view["F"]["x"], geom))

    if cfg.mode == "jacobi":
        new = {"F": do_F(cur), "P": do_P(cur)}
    else:
        new = {}
        for node in sweep:
            new[node] = do_F(cur) if node == "F" else do_P(cur)
            cur[node] = new[node]
    return pack(cfg, new)


def jac(cfg, x, c, wrt="x"):
    """dF/dx (or dF/dc) by complex step: exact inside a lattice cell."""
    n = x.size
    h = 1e-30
    cols = []
    for j in range(n):
        if wrt == "x":
            xp = x.astype(complex)
            xp[j] += 1j * h
            cols.append(ref_pass(cfg, xp, c.astype(complex)).imag / h)
        else:
            cp = c.astype(complex)
            cp[j] += 1j * h
            cols.append(ref_pass(cfg, x.astype(complex), cp).imag / h)
    return np.stack(cols, 1)


def fixed_point(cfg, x0, c, iters=200000, tol=1e-15, warm=60):
    """The fixed point the plain iteration from x0 goes to: *warm* plain
    passes, then Newton on the polynomial piece the iterate is in (a step is
    kept only where it lowers the residual; a plain pass otherwise)."""
    x = x0.astype(float).copy()
    n = x.size
    eye = np.eye(n)

    def res(v):
        return float(np.max(np.abs(ref_pass(cfg, v, c).real - v)))

    for _ in range(warm):
        xn = ref_pass(cfg, x, c).real
        if not np.all(np.isfinite(xn)):
            return x, False
        x = xn
    r_now = res(x)
    scale = max(1.0, float(np.max(np.abs(x))))
    for _ in range(iters):
        if r_now <= tol * scale:
            break
        stepped = False
        try:
            dx = np.linalg.solve(eye - jac(cfg, x, c), ref_pass(cfg, x, c).real - x)
            xn = x + dx
            rn = res(xn)
            if np.all(np.isfinite(xn)) and rn < 0.5 * r_now:
                x, r_now, stepped = xn, rn, True
        except np.linalg.LinAlgError:
            pass
        if not stepped:
            for _ in range(25):
                x = ref_pass(cfg, x, c).real
            if not np.all(np.isfinite(x)):
                return x, False
            rn = res(x)
            if not rn < r_now * 0.999 and rn > tol * scale:
                # not contracting here: no fixed point the iteration goes to
                if rn > 1e-9 * scale:
                    return x, False
            r_now = rn
            iters -= 25
            if iters <= 0:
                break
    return x, bool(r_now <= 1e-11 * scale)


def weights(cfg, x):
    """D: 1 / max|field| of each entry's field at x (0 for a zero field)."""
    ids = field_ids(cfg)
    w = np.zeros(x.size)
    for i in range(ids.max() + 1):
        top = np.max(np.abs(x[ids == i]))
        w[ids == i] = 1.0 / top if top > 0 else 0.0
    return w


def norm_unit(cfg, x):
    if cfg.norm == "l2":
        return 1.0
    return 1.0 / (cfg.rtol * math.sqrt(int(np.sum(weights(cfg, x) > 0))))


def pos_slices(cfg):
    out, k = {}, 0
    for n, f, s in fields(cfg):
        if f == "pos":
            out[n] = slice(k, k + s)
        k += s
    return out


# --------------------------------------------------------------------------
# the graph under test (coefficients are node params: one compile per cell)
# --------------------------------------------------------------------------
P_KEYS = {"F": ("aF", "bF", "kF", "vF"), "P": ("aP", "bP", "cP", "quad", "kP", "vP")}


def p_keys(cfg):
    """The coefficients each node reads (a position's drift only where the
    node holds positions: the graph refuses a parameter no hook reads)."""
    return {"F": ("aF", "bF") + (("kF", "vF") if cfg.holds("F") else ()),
            "P": ("aP", "bP", "cP", "quad") + (("kP", "vP") if cfg.holds("P") else ())}


def rounded(cfg: Cfg) -> Cfg:
    """cfg with every coefficient and initial value rounded to the graph's
    dtype: the float64 reference evaluates the map the group evaluates."""
    T = np.dtype(cfg.dtype)

    def r(v):
        a = np.asarray(v, np.float64).astype(T).astype(np.float64)
        return float(a) if a.ndim == 0 else tuple(a.ravel().tolist())

    kw = {k: r(getattr(cfg, k)) for ks in P_KEYS.values() for k in ks}
    for k in ("xF0", "xP0", "posP0", "posF0"):
        kw[k] = r(getattr(cfg, k)) if len(getattr(cfg, k)) else ()
    return dataclasses.replace(cfg, **kw)


def build(cfg: Cfg, diagnostics=True):
    import jax.numpy as jnp
    from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import BoundaryInputSpec, BoundaryFluxSpec, SimulationNode

    T = jnp.dtype(cfg.dtype)
    N, m, d = cfg.size, cfg.m, cfg.d
    sel = np.asarray(_sel(cfg))
    holdsF, holdsP = cfg.holds("F"), cfg.holds("P")

    class Fnode(SimulationNode):
        def initial_state(self):
            st = {"x": jnp.asarray(np.asarray(cfg.xF0, np.float64), T)}
            if holdsF:
                st["pos"] = jnp.asarray(np.asarray(cfg.posF0, np.float64).reshape(m, d), T)
            return st

        def boundary_input_spec(self):
            return {"deposit": BoundaryInputSpec(shape=(N,), dtype=T)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = {**self.params, **(params or {})}
            dep = boundary_inputs.get("deposit", jnp.zeros_like(state["x"]))
            new = {"x": p["aF"] * state["x"] + p["bF"] * dep}
            if holdsF:
                drift = jnp.zeros((m, d), T) + jnp.asarray(p["vF"], T)[None, :]
                drift = drift.at[:, 0].add(p["kF"] * dep[sel])
                new["pos"] = state["pos"] + drift
            return new

    class Pnode(SimulationNode):
        def initial_state(self):
            st = {"x": jnp.asarray(np.asarray(cfg.xP0, np.float64), T)}
            if holdsP:
                st["pos"] = jnp.asarray(np.asarray(cfg.posP0, np.float64).reshape(m, d), T)
            return st

        def boundary_input_spec(self):
            return {"sampled": BoundaryInputSpec(shape=(m,), dtype=T)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = {**self.params, **(params or {})}
            s = boundary_inputs.get("sampled", jnp.zeros_like(state["x"]))
            new = {"x": p["aP"] * state["x"] + p["bP"] * s + p["cP"] + p["quad"] * s * s}
            if holdsP:
                drift = jnp.zeros((m, d), T) + jnp.asarray(p["vP"], T)[None, :]
                drift = drift.at[:, 0].add(p["kP"] * s)
                new["pos"] = state["pos"] + drift
            return new

    lattice = dict(origin=list(cfg.origin), spacing=list(cfg.spacing), shape=list(cfg.N),
                   n_points=m)
    gm = GraphManager()

    def arr(v):
        return np.asarray(v, np.float64).astype(np.dtype(cfg.dtype))

    keys = p_keys(cfg)
    made = {
        "F": Fnode("F", 0.01, **{k: arr(getattr(cfg, k)) for k in keys["F"]}),
        "P": Pnode("P", 0.01, **{k: arr(getattr(cfg, k)) for k in keys["P"]}),
    }
    for name in cfg.order:
        gm.add_node(made[name])
    gm.add_edge("F", "P", "x", "sampled",
                mapping=multilinear_grid_mapping(mode="consistent", **lattice),
                geometry=(cfg.anchors[0], "pos"))
    gm.add_edge("P", "F", "x", "deposit",
                mapping=multilinear_grid_mapping(mode="conservative", **lattice),
                geometry=(cfg.anchors[1], "pos"))
    kw = dict(convergence_norm=cfg.norm, diagnostics=diagnostics, acceleration=cfg.acceleration,
              iteration_mode=cfg.mode, predictor=cfg.predictor)
    if cfg.norm == "l2":
        kw["tolerance"] = cfg.tolerance
    else:
        kw["rtol"] = cfg.rtol
    if cfg.acceleration == "fixed":
        kw["relaxation"] = cfg.relaxation
    if cfg.group:
        gm.add_coupling_group(["F", "P"], max_iterations=cfg.max_iterations, **kw)
    return gm


def initial_flat(cfg):
    T = np.dtype(cfg.dtype)
    st = {"F": {"x": np.asarray(cfg.xF0, np.float64).astype(T)},
          "P": {"x": np.asarray(cfg.xP0, np.float64).astype(T)}}
    if cfg.holds("F"):
        st["F"]["pos"] = np.asarray(cfg.posF0, np.float64).astype(T).ravel()
    if cfg.holds("P"):
        st["P"]["pos"] = np.asarray(cfg.posP0, np.float64).astype(T).ravel()
    return pack(cfg, st).astype(np.float64)


def state_flat(cfg, gm):
    st = {n: {f: np.asarray(v, np.float64).ravel() for f, v in gm.get_node_state(n).items()}
          for n in ("F", "P")}
    return pack(cfg, st)


def load(cfg: Cfg, gm):
    """Write cfg's coefficients and initial state into a compiled graph of the
    same cell (no recompile): gm.params leaves and set_node_state."""
    import jax.numpy as jnp
    T = jnp.dtype(cfg.dtype)
    tree = gm.params
    for node, keys in p_keys(cfg).items():
        for k in keys:
            old = tree["nodes"][node][k]
            tree["nodes"][node][k] = jnp.asarray(
                np.asarray(getattr(cfg, k), np.float64).reshape(np.shape(old)), old.dtype)
    gm.params = tree
    stF = {"x": jnp.asarray(np.asarray(cfg.xF0, np.float64), T)}
    stP = {"x": jnp.asarray(np.asarray(cfg.xP0, np.float64), T)}
    if cfg.holds("F"):
        stF["pos"] = jnp.asarray(np.asarray(cfg.posF0, np.float64).reshape(cfg.m, cfg.d), T)
    if cfg.holds("P"):
        stP["pos"] = jnp.asarray(np.asarray(cfg.posP0, np.float64).reshape(cfg.m, cfg.d), T)
    gm.set_node_state("F", stF)
    gm.set_node_state("P", stP)


def step_report(cfg, gm):
    c = state_flat(cfg, gm)
    gm.step()
    x = state_flat(cfg, gm)
    rep = dict(next(iter(gm.coupling_diagnostics().values())))
    return x, c, rep


# --------------------------------------------------------------------------
# scoring against the reference
# --------------------------------------------------------------------------
def real_pass(cfg, x, c):
    return ref_pass(cfg, x, c).real


def param_list(cfg):
    out = []
    for ks in p_keys(cfg).values():
        for k in ks:
            v = getattr(cfg, k)
            if isinstance(v, tuple):
                out.extend((k, a) for a in range(len(v)))
            else:
                out.append((k, None))
    return out


def jac_params(cfg, x, c):
    """dF/dparam, one column per scalar coefficient (complex step)."""
    h = 1e-30
    cols = []
    for k, a in param_list(cfg):
        v = getattr(cfg, k)
        if a is None:
            moved = dataclasses.replace(cfg, **{k: complex(v, h)})
        else:
            vv = [complex(t) for t in v]
            vv[a] += 1j * h
            moved = dataclasses.replace(cfg, **{k: tuple(vv)})
        cols.append(ref_pass(moved, x.astype(complex), c.astype(complex)).imag / h)
    return np.stack(cols, 1)


def param_sizes(cfg):
    """The size each probe moves its constant by (its magnitude; a zero entry
    at its constant's largest, or 1)."""
    out = []
    for k, a in param_list(cfg):
        v = np.atleast_1d(np.asarray(getattr(cfg, k), float))
        mag = abs(v[a or 0])
        top = np.max(np.abs(v))
        out.append(mag if mag > 0 else (top if top > 0 else 1.0))
    return np.asarray(out)


def state_sizes(cfg, c):
    ids = field_ids(cfg)
    out = np.abs(c).copy()
    for i in range(ids.max() + 1):
        sel = ids == i
        top = np.max(np.abs(c[sel]))
        out[sel] = np.where(out[sel] > 0, out[sel], top if top > 0 else 1.0)
    return out


def op2(A):
    return float(np.linalg.norm(A, 2)) if A.size else 0.0


def score(cfg: Cfg, x, c, rep, full=True):
    """Every reported number against the float64 reference at the returned
    iterate x (pre-step state c)."""
    out = {}
    n = x.size
    J = jac(cfg, x, c)
    D = weights(cfg, x)
    live = D > 0
    Dl = np.where(live, D, 1.0)
    unit = norm_unit(cfg, x)
    ev = np.linalg.eigvals(J[np.ix_(live, live)] * (D[live][:, None] / D[live][None, :]))
    out["rho_ref"] = float(np.max(np.abs(ev))) if ev.size else 0.0
    xs, ok = fixed_point(cfg, x, c)
    out["fp_ok"] = bool(ok)
    out["dist"] = float(unit * np.linalg.norm(D * (x - xs)))
    out["res_ref"] = float(unit * np.linalg.norm(D * (real_pass(cfg, x, c) - x)))
    ps = pos_slices(cfg)
    crossed, near, moved = False, np.inf, 0.0
    for node, sl in ps.items():
        crossed = crossed or bool(np.any(cells_of(cfg, x[sl]) != cells_of(cfg, xs[sl])))
        near = min(near, float(np.min(plane_distance(cfg, x[sl]))))
        moved = max(moved, float(np.max(np.abs(x[sl] - xs[sl]))))
    out["crossed"] = crossed
    out["plane_dist_min"] = near
    # in spacings: the returned iterate's and the fixed point's nearest plane
    hmin = min(cfg.spacing)
    out["xk_plane_frac"] = near / hmin
    out["fp_plane_frac"] = min(float(np.min(plane_distance(cfg, xs[sl]))) for sl in ps.values()) / hmin if ps else float("inf")
    out["pos_moved"] = moved
    # the positions the pass builds from x (a Gauss-Seidel sweep reads them
    # after their holder's update), and the Newton point's
    fx = real_pass(cfg, x, c)
    eps = float(np.finfo(np.dtype(cfg.dtype)).eps)
    ulps = np.inf
    ip_crossed = False
    for node, sl in ps.items():
        ip_crossed = ip_crossed or bool(np.any(cells_of(cfg, fx[sl]) != cells_of(cfg, xs[sl])))
        for v in (x[sl], fx[sl], xs[sl]):
            dist_v = plane_distance(cfg, v).ravel()
            ulp = eps * np.maximum(np.abs(v), 1e-300)
            ulps = min(ulps, float(np.min(dist_v / ulp)))
    out["ip_crossed"] = ip_crossed
    out["min_plane_ulps"] = ulps
    try:
        xn = x + np.linalg.solve(np.eye(n) - J, fx - x)
        out["newton_same_cell_as_fp"] = bool(all(
            np.all(cells_of(cfg, xn[sl]) == cells_of(cfg, xs[sl])) for sl in ps.values()))
        out["newton_same_cell_as_xk"] = bool(all(
            np.all(cells_of(cfg, xn[sl]) == cells_of(cfg, x[sl])) for sl in ps.values()))
    except np.linalg.LinAlgError:
        pass
    if not full:
        return out
    try:
        Js = jac(cfg, xs, c)
        out["rho_fp"] = float(np.max(np.abs(np.linalg.eigvals(Js))))
        Rk = np.linalg.inv(np.eye(n) - J)
        # h: how far the Jacobian moves between the iterate and the fixed point
        E = Rk @ (Js - J)
        out["h_ref"] = op2((Dl[:, None] * E / Dl[None, :])[np.ix_(live, live)])
        Jc_k = np.concatenate([jac(cfg, x, c, "c"), jac_params(cfg, x, c)], 1)
        Jc_s = np.concatenate([jac(cfg, xs, c, "c"), jac_params(cfg, xs, c)], 1)
        sizes = np.concatenate([state_sizes(cfg, c), param_sizes(cfg)])
        tk = Rk @ Jc_k
        ts = np.linalg.solve(np.eye(n) - Js, Jc_s)
        num = np.linalg.norm(D[:, None] * (tk - ts), axis=0)
        den = np.linalg.norm(D[:, None] * tk, axis=0)
        rhs = np.linalg.norm(D[:, None] * Jc_k * sizes[None, :], axis=0)
        eps = float(np.finfo(np.dtype(cfg.dtype)).eps)
        resolved = rhs > 1e3 * eps * math.sqrt(n)
        rel = np.where(resolved & (den > 0), num / np.where(den > 0, den, 1), 0.0)
        out["grad_err"] = float(np.max(rel)) if rel.size else 0.0
        out["grad_arg"] = int(np.argmax(rel)) if rel.size else -1
    except np.linalg.LinAlgError:
        out["grad_err"] = float("nan")
    return out


# --------------------------------------------------------------------------
# the sides of a lattice plane
# --------------------------------------------------------------------------
ROWS = ("k=* N=*", "k=* N!=*", "k!=* N=*", "k!=* N!=*")


def plane_resolution(cfg: Cfg, pos):
    """``eps`` of the graph's dtype times the largest coordinate magnitude of
    the lattice's axis, or the coordinate's own (or its offset's from the
    origin) where that is larger; shape ``(m, d)``."""
    pos = np.asarray(pos).real.reshape(cfg.m, cfg.d)
    eps = float(np.finfo(np.dtype(cfg.dtype)).eps)
    out = []
    for a in range(cfg.d):
        top = cfg.origin[a] + (cfg.N[a] - 1) * cfg.spacing[a]
        scale = max(abs(cfg.origin[a]), abs(top))
        out.append(eps * np.maximum(np.maximum(np.abs(pos[:, a]), np.abs(pos[:, a] - cfg.origin[a])),
                                    scale))
    return np.stack(out, 1)


def sides(cfg: Cfg, x, c) -> dict:
    """Which lattice cells the positions are in at the returned iterate, at
    the Newton point, in the pass at the iterate and at the fixed point, by
    the reference; the row of the table; and the nearest any of them comes
    to a plane, in float resolutions and in spacings."""
    n = x.size
    xs, ok = fixed_point(cfg, x, c)
    fx = real_pass(cfg, x, c)
    J = jac(cfg, x, c)
    try:
        xn = x + np.linalg.solve(np.eye(n) - J, fx - x)
    except np.linalg.LinAlgError:
        xn = fx
    k_same = n_same = built_same = True
    nearest_res = nearest_frac = np.inf
    fp_frac = np.inf
    for _node, sl in pos_slices(cfg).items():
        at_fp = cells_of(cfg, xs[sl])
        k_same = k_same and bool(np.all(cells_of(cfg, x[sl]) == at_fp))
        n_same = n_same and bool(np.all(cells_of(cfg, xn[sl]) == at_fp))
        built_same = built_same and bool(np.all(cells_of(cfg, fx[sl]) == at_fp))
        for v in (x[sl], xn[sl], fx[sl], xs[sl]):
            dist = plane_distance(cfg, v)
            nearest_res = min(nearest_res, float(np.min(dist / plane_resolution(cfg, v))))
            nearest_frac = min(nearest_frac, float(np.min(dist / np.asarray(cfg.spacing)[None, :])))
        fp_frac = min(fp_frac, float(np.min(
            plane_distance(cfg, xs[sl]) / np.asarray(cfg.spacing)[None, :])))
    row = f"k{'=' if k_same else '!='}* N{'=' if n_same else '!='}*"
    return {"fp_ok": bool(ok), "row": row, "k_same": k_same, "n_same": n_same,
            "built_same": built_same, "one_piece": k_same and n_same and built_same,
            "nearest_resolutions": nearest_res, "nearest_spacings": nearest_frac,
            "fixed_point_spacings": fp_frac,
            "on_a_plane": nearest_res < PLANE_RESOLUTIONS}


def wrong_numbers(cfg: Cfg, x, c, rep, where: dict) -> list:
    """Every reported number whose flag is set, against the reference: the
    failures, as text (empty where all hold).

    * ``spectral_usable``: ``rho_spectral`` within 5% of ``1 - rho`` of the
      reference's radius at the returned iterate (off a plane: on one, which
      cell's Jacobian the floats evaluated is rounding's, MADD-ANO-239);
      the distance to the fixed point within ``spectral_error_bound / (1 -
      h)`` in one cell (``h`` how far the Jacobian moves on the way) and
      within twice the bound across a plane (MAP-049);
    * ``gradient_bound_usable``: the implicit derivative's relative error,
      worst resolved constant, within ``gradient_relative_error_bound``;
      and never set unless the iterate, the Newton point, the positions the
      pass builds and the fixed point are in one lattice cell, off every
      plane (MADD-ANO-251).
    """
    sc = score(cfg, x, c, rep)
    bad = []
    if not sc["fp_ok"]:
        return bad
    bound, rho = float(rep["spectral_error_bound"]), float(rep["rho_spectral"])
    grad = float(rep["gradient_relative_error_bound"])
    if rep["spectral_usable"] and not where["on_a_plane"]:
        if abs(rho - sc["rho_ref"]) > 0.05 * (1.0 - rho) + 1e-4:
            bad.append(f"rho_spectral {rho:.6g}, the reference's radius {sc['rho_ref']:.6g}")
        h = sc.get("h_ref", math.nan)
        if where["k_same"] and where["built_same"] and h < 1 and sc["dist"] > bound / (1 - h) * 1.001:
            bad.append(f"spectral_error_bound {bound:.6g} (h {h:.3g}), the distance {sc['dist']:.6g}")
        if not (where["k_same"] and where["built_same"]) and sc["dist"] > 2 * bound * 1.001:
            bad.append(f"twice spectral_error_bound {bound:.6g} across a plane, the distance "
                       f"{sc['dist']:.6g}")
    if rep["gradient_bound_usable"] and moving(cfg):
        # The rule's own statement: one lattice cell, off every plane.  (A
        # plane between two points that are both within the window of it
        # is rounding's to place: not asked.)
        if not where["one_piece"] and not where["on_a_plane"]:
            bad.append(f"gradient_bound_usable is set on row {where['row']} (the positions the "
                       f"pass builds in the fixed point's cell: {where['built_same']})")
        if where["nearest_resolutions"] < PLANE_RESOLUTIONS / 2:
            bad.append(f"gradient_bound_usable is set with a position "
                       f"{where['nearest_resolutions']:.3g} float resolutions from a plane")
    if rep["gradient_bound_usable"] and (where["one_piece"] or not moving(cfg)):
        if sc["grad_err"] > grad * 1.02 + 1e-9:
            bad.append(f"gradient_relative_error_bound {grad:.6g}, the reference's error "
                       f"{sc['grad_err']:.6g} ({sc['grad_err'] / grad:.1f}x)")
    return bad


def moving(cfg: Cfg) -> bool:
    """Does the pass read a position from the iterate (a source anchor)?  With
    two target anchors every position is a constant of the pass."""
    return "source" in cfg.anchors


# --------------------------------------------------------------------------
# the constructive placement
# --------------------------------------------------------------------------
def drawn(structure: Cfg, rng, *, quad: bool = True) -> Cfg:
    """*structure* with drawn coefficients and a drawn initial state."""
    N, m, d = structure.size, structure.m, structure.d
    span = [structure.origin[a] + (structure.N[a] - 1) * structure.spacing[a] for a in range(d)]

    def positions():
        p = np.stack([rng.uniform(structure.origin[a], span[a], m) for a in range(d)], 1)
        return tuple(p.ravel().tolist())

    def log_uniform(lo, hi):
        return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))

    def sign():
        return float(rng.choice([-1.0, 1.0]))

    kw = dict(
        aF=float(rng.uniform(-0.8, 0.8)), bF=sign() * log_uniform(0.05, 1.5),
        aP=float(rng.uniform(-0.8, 0.8)), bP=sign() * log_uniform(0.05, 1.5),
        cP=float(rng.uniform(-1, 1)),
        quad=(sign() * log_uniform(0.01, 0.3) if quad and rng.random() < 0.5 else 0.0),
        vP=tuple(rng.uniform(-0.1, 0.1, d).tolist()), vF=tuple(rng.uniform(-0.1, 0.1, d).tolist()),
        kP=sign() * log_uniform(0.01, 1.0), kF=sign() * log_uniform(0.01, 1.0),
        xF0=tuple(rng.uniform(0.3, 2.0, N).tolist()) if rng.random() < 0.5
        else tuple(rng.uniform(-2.0, 2.0, N).tolist()),
        xP0=tuple((rng.uniform(0.3, 2.0, m) * rng.choice([-1, 1], m)).tolist()),
        posP0=positions() if structure.holds("P") else (),
        posF0=positions() if structure.holds("F") else (),
    )
    return dataclasses.replace(structure, **kw)


def shifted(cfg: Cfg, key: str, j: int, a: int, value: float) -> Cfg:
    """*cfg* with coordinate ``(j, a)`` of the initial positions *key* set."""
    pos = np.asarray(getattr(cfg, key), float).reshape(cfg.m, cfg.d).copy()
    pos[j, a] = value
    return dataclasses.replace(cfg, **{key: tuple(pos.ravel().tolist())})


def _fixed_point_coordinate(cfg: Cfg, sl, j: int, a: int):
    c0 = initial_flat(cfg)
    xs, ok = fixed_point(cfg, c0, c0, iters=3000, tol=1e-15)
    return xs[sl].reshape(cfg.m, cfg.d)[j, a] if ok else None


def placed_on_a_plane(structure: Cfg, rng, *, tries: int = 30):
    """A drawn pair and the initial position of one of its markers at which
    the *fixed point* has that marker on a lattice plane.

    Returns ``(cfg, key, j, a, on_plane)``: ``shifted(cfg, key, j, a,
    on_plane + s)`` has its fixed point about ``s`` past the plane (the
    fixed point follows the initial position at a rate near one).  The
    marker is one whose position the pass reads from the iterate where
    there is one.  ``None`` after *tries* draws without a fixed point that
    brackets a plane.
    """
    for _ in range(tries):
        cfg = drawn(structure, rng)
        g, s = cfg.anchors
        from_iterate = [n for n in ("P", "F") if cfg.holds(n)
                        and ((n == "P" and s == "source") or (n == "F" and g == "source"))]
        node = str(rng.choice(from_iterate or [n for n in ("P", "F") if cfg.holds(n)]))
        key = "posP0" if node == "P" else "posF0"
        j, a = int(rng.integers(cfg.m)), int(rng.integers(cfg.d))
        h = cfg.spacing[a]
        sl = pos_slices(cfg)[node]
        try:
            p0 = np.asarray(getattr(cfg, key), float).reshape(cfg.m, cfg.d)[j, a]
            at = _fixed_point_coordinate(cfg, sl, j, a)
            if at is None:
                continue
            index = int(np.clip(np.round((at - cfg.origin[a]) / h), 0, cfg.N[a] - 1))
            plane = cfg.origin[a] + index * h
            lo, hi = p0 - 0.75 * h, p0 + 0.75 * h
            f_lo = _fixed_point_coordinate(shifted(cfg, key, j, a, lo), sl, j, a)
            f_hi = _fixed_point_coordinate(shifted(cfg, key, j, a, hi), sl, j, a)
            if f_lo is None or f_hi is None or (f_lo - plane) * (f_hi - plane) > 0:
                continue
            for _bisection in range(70):
                mid = 0.5 * (lo + hi)
                f_mid = _fixed_point_coordinate(shifted(cfg, key, j, a, mid), sl, j, a)
                if f_mid is None:
                    break
                if (f_mid - plane) * (f_lo - plane) > 0:
                    lo, f_lo = mid, f_mid
                else:
                    hi = mid
            else:
                return cfg, key, j, a, 0.5 * (lo + hi)
        except (IndexError, ValueError, FloatingPointError, OverflowError):
            continue
    return None


def signed_distances(cfg: Cfg, on_plane: float, a: int, rng, count: int) -> list:
    """*count* shifts of the initial position, half each side of the plane:
    magnitudes log-uniform from four float resolutions of the position to a
    tenth of a spacing."""
    eps = float(np.finfo(np.dtype(cfg.dtype)).eps)
    top = cfg.origin[a] + (cfg.N[a] - 1) * cfg.spacing[a]
    low = 4.0 * eps * max(abs(on_plane), abs(cfg.origin[a]), abs(top))
    high = 0.1 * cfg.spacing[a]
    sizes = 10 ** rng.uniform(math.log10(low), math.log10(high), count)
    return [float(s) * (1.0 if i % 2 == 0 else -1.0) for i, s in enumerate(sizes)]


def as_cfg(d: dict) -> Cfg:
    return Cfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
