"""A geometry group whose fixed point is placed beside a lattice plane *by
solving for its constants*: the pair of the audit that found the spectral
flag standing on a bound far under the distance (MADD-ANO-248).

**The pair.**  A grid member ``G`` and one marker ``M``.  ``G.x`` is the
marker's multilinear deposit at the marker's position, read from the iterate
(or from the same sweep): ``x' = scatter(M.y at M.pos)``.  The marker moves
along a fixed direction with a read-out of the grid field through a plain
edge: ``pos' = pos_pre + v + (w . x + q (u . x)**2) dirn``, ``y' = y_pre``.
So the marker's position along its line obeys a one-dimensional map ``g``
whose form is chosen cell by cell through ``w``:

* **before a chosen lattice plane** ``g(tau) = s + a_s tau + Q tau**2``
  (``tau`` the signed distance past the plane): a polynomial whose fixed
  point is at about ``s / (1 - a_s)``, *past* the plane for ``s > 0``.  The
  curvature ``Q`` comes from the kernel alone on a two-dimensional lattice
  (the ``t_x t_y`` term of the stencil, linear members) or from a member's
  smooth quadratic response on a one-dimensional one (``q``);
* **past it**, cell by cell, straight lines of chosen slopes (``after``): a
  next cell that expands (1.2) sends the iterate on to the cell after; one
  that contracts at 0.999 or 0.99 holds a fixed point of its own.

The first pass lands ``e1`` before the plane.  With ``s`` under the Newton
step's second-order miss the returned iterate *and* the Newton point are
both before the plane, in one cell, and the Newton-Kantorovich check reads
that cell's small ``h``; the cell's polynomial has its fixed point past the
plane, where the pass is another polynomial.  The pass has no fixed point
where the bound says.

**The reference** is :mod:`tests.property.plane_sides`'s NumPy kernel (it
does not import the library's), with the pass's Jacobian by complex step,
the fixed point the plain iteration goes to, and the fixed point of the
polynomial piece at the iterate (the stencil of the iterate's cell
continued past its planes).  Only :func:`build` imports the library.

Lifted from ``benchmarks/results/audit_040_p4_19/plane_margin/``
(``repro_lib.py``, ``repro_f1_spectral_flag_newton_point_short_of_plane.py``
and ``repro_placed_fixed_points.py``).
"""
from __future__ import annotations

import dataclasses
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from tests.property import plane_sides as ps

KEY = "G+M"
#: The coefficients a compiled graph takes as node parameters: one compile
#: per structure (:data:`STRUCTURE`), every case loaded into it.
PARAMS = ("w", "u", "q", "v", "dirn")
STRUCTURE = ("shape", "origin", "spacing", "dtype", "order", "mode", "norm", "tolerance",
             "rtol", "acceleration", "max_iterations")


@dataclasses.dataclass(frozen=True)
class Model:
    shape: tuple = (6,)
    origin: tuple = (0.0,)
    spacing: tuple = (0.5,)
    dtype: str = "float32"
    order: tuple = ("M", "G")           # add_node order: the sweep's
    mode: str = "gauss-seidel"
    norm: str = "l2"
    tolerance: float = 1e-2
    rtol: float = 1e-2
    acceleration: str = "none"
    max_iterations: int = 60
    x0: tuple = ()                      # (N,)
    pos0: tuple = ()                    # (d,)
    w: tuple = ()                       # (N,): the read-out that moves the marker
    u: tuple = ()                       # (N,): the read-out of the quadratic response
    q: float = 0.0
    v: tuple = (0.0,)                   # (d,)
    dirn: tuple = (1.0,)                # (d,)
    plane: float = 0.0                  # the first coordinate of the chosen lattice plane

    @property
    def d(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def lattice(self) -> ps.Cfg:
        """The lattice, as the reference kernel takes it (one marker)."""
        return ps.Cfg(N=tuple(self.shape), m=1, origin=tuple(self.origin),
                      spacing=tuple(self.spacing), dtype=self.dtype)

    def structure(self) -> tuple:
        return tuple((name, getattr(self, name)) for name in STRUCTURE)


# --------------------------------------------------------------------------
# the reference pass: fields G.x (N), M.pos (d), M.y (1), in that order
# --------------------------------------------------------------------------
def pos_slice(mo: Model) -> slice:
    return slice(mo.size, mo.size + mo.d)


def ref_pass(mo: Model, x, c, piece=None):
    """One pass at the iterate *x* with the pre-step state *c*.  With
    *piece* (the marker's cell signature) the deposit is that cell's
    polynomial continued past its planes."""
    n, d = mo.size, mo.d
    lattice = mo.lattice
    w, u = np.asarray(mo.w, float), np.asarray(mo.u, float)

    def grid(pos, y):
        return ps.ref_scatter(lattice, y, pos, piece)

    def marker(field):
        read = u @ field
        drive = w @ field + mo.q * read * read
        return c[n:n + d] + np.asarray(mo.v) + drive * np.asarray(mo.dirn)

    y_new = c[n + d:]
    if mo.mode == "jacobi":
        return np.concatenate([grid(x[n:n + d], x[n + d:]), marker(x[:n]), y_new])
    if mo.order[0] == "M":
        pos = marker(x[:n])
        return np.concatenate([grid(pos, y_new), pos, y_new])
    field = grid(x[n:n + d], x[n + d:])
    return np.concatenate([field, marker(field), y_new])


def jac(mo: Model, x, c, piece=None):
    cols = []
    for j in range(x.size):
        xp = x.astype(complex)
        xp[j] += 1e-30j
        cols.append(ref_pass(mo, xp, c.astype(complex), piece).imag / 1e-30)
    return np.stack(cols, 1)


def fixed_point(mo: Model, x, c, passes=4000):
    """``(x*, ok)``: where the plain iteration from *x* goes, finished by
    Newton on the polynomial piece it arrived in."""
    y = np.asarray(x, float).copy()
    for _ in range(passes):
        z = ref_pass(mo, y, c).real
        if not np.all(np.isfinite(z)):
            return y, False
        moved = float(np.max(np.abs(z - y)))
        y = z
        if moved < 1e-13:
            break
    eye = np.eye(y.size)
    for _ in range(4):
        try:
            step = np.linalg.solve(eye - jac(mo, y, c), ref_pass(mo, y, c).real - y)
        except np.linalg.LinAlgError:
            return y, False
        if cells(mo, y + step) != cells(mo, y):
            break
        y = y + step
    return y, bool(np.max(np.abs(ref_pass(mo, y, c).real - y)) < 1e-10)


def cell_fixed_point(mo: Model, x, c, iters=80):
    """``(x_p, ok)``: the fixed point of the polynomial piece the pass is at
    *x*, by Newton on that polynomial, wherever it lies."""
    piece = ps.cells_of(mo.lattice, x[pos_slice(mo)])
    y = np.asarray(x, float).copy()
    eye = np.eye(y.size)
    for _ in range(iters):
        try:
            step = np.linalg.solve(eye - jac(mo, y, c, piece),
                                   ref_pass(mo, y, c, piece).real - y)
        except np.linalg.LinAlgError:
            return y, False
        y = y + step
        if not np.all(np.isfinite(y)):
            return y, False
        if float(np.max(np.abs(step))) < 1e-13:
            return y, True
    return y, False


def cells(mo: Model, x) -> tuple:
    return tuple(int(v) for v in np.ravel(ps.cells_of(mo.lattice, x[pos_slice(mo)])))


def weights(mo: Model, x):
    """The norm's weights at the returned state: each field over its own
    largest magnitude."""
    out = np.zeros(x.size)
    for sl in (slice(0, mo.size), pos_slice(mo), slice(mo.size + mo.d, x.size)):
        top = float(np.max(np.abs(x[sl])))
        out[sl] = 1.0 / top if top > 0 else 0.0
    return out


def located(mo: Model, x, c) -> dict:
    """Where the four points lie: the returned iterate ``x_k``, the Newton
    point ``x_N``, the fixed point ``x_p`` of the polynomial the pass is at
    the iterate, and the fixed point ``x*`` the pass goes to; the distance
    from the iterate to the last in the report's norm."""
    n = x.size
    fx = ref_pass(mo, x, c).real
    xn = x + np.linalg.solve(np.eye(n) - jac(mo, x, c), fx - x)
    xp, p_ok = cell_fixed_point(mo, x, c)
    xs, ok = fixed_point(mo, x, c)
    weight = weights(mo, x)
    unit = 1.0 if mo.norm == "l2" else 1.0 / (mo.rtol * math.sqrt(int(np.sum(weight > 0))))
    first = pos_slice(mo).start
    resolution = float(np.min(ps.plane_resolution(mo.lattice, x[pos_slice(mo)])))
    nearest = min(float(np.min(ps.plane_distance(mo.lattice, v[pos_slice(mo)])))
                  for v in (x, xn, fx, xs))
    return {
        "fp_ok": bool(ok), "p_ok": bool(p_ok),
        "cells": {"k": cells(mo, x), "N": cells(mo, xn), "built": cells(mo, fx),
                  "p": cells(mo, xp), "*": cells(mo, xs)},
        # the first coordinate of each, past the chosen plane
        "past": {"k": float(x[first] - mo.plane), "N": float(xn[first] - mo.plane),
                 "p": float(xp[first] - mo.plane), "*": float(xs[first] - mo.plane)},
        "p_in_cell": bool(p_ok and cells(mo, xp) == cells(mo, x)),
        "one_cell": cells(mo, x) == cells(mo, xn) == cells(mo, fx) == cells(mo, xs),
        "distance": float(unit * np.linalg.norm(weight * (x - xs))),
        "nearest_resolutions": nearest / resolution,
        "rho": float(np.max(np.abs(np.linalg.eigvals(jac(mo, x, c))))),
        "rho_at_fixed_point": float(np.max(np.abs(np.linalg.eigvals(jac(mo, xs, c))))),
    }


def wrong_numbers(report: dict, where: dict) -> list:
    """Every flagged number of *report* that does not hold at *where*
    (:func:`located`), as text.  A flag is a statement about the fixed point
    the pass goes to: the distance to it within twice
    ``spectral_error_bound`` (the Newton-Kantorovich radius of the bound's
    own argument), and no flag at all with the fixed point of the iterate's
    polynomial past a lattice plane."""
    bad = []
    if not where["fp_ok"]:
        return bad
    bound = float(report["spectral_error_bound"])
    for flag in ("spectral_usable", "gradient_bound_usable"):
        if not report[flag]:
            continue
        if where["distance"] > 2.0 * bound * 1.001:
            bad.append(f"{flag} is set with the fixed point {where['distance'] / bound:.4g} "
                       "times spectral_error_bound away")
        if where["nearest_resolutions"] > ps.PLANE_RESOLUTIONS:
            if where["p_ok"] and not where["p_in_cell"]:
                bad.append(f"{flag} is set with the fixed point of the iterate's polynomial "
                           f"past a lattice plane (cells {where['cells']})")
            if not where["one_cell"]:
                bad.append(f"{flag} is set with a lattice plane between the points it rests "
                           f"on (cells {where['cells']})")
    return bad


# --------------------------------------------------------------------------
# the two constructions
# --------------------------------------------------------------------------
def _lines_after(w, k, after, step):
    """Read-out weights past index *k*: one straight line per cell, of the
    slopes *after*, then flat."""
    for slope in after:
        k += 1
        if k >= w.shape[0]:
            return
        w[k] = w[k - 1] + slope * step
    w[k + 1:] = w[k]


def kernel_curvature(*, s=2e-6, e1=0.01, a_s=0.5, after=(1.2, 0.5), Q=0.5, kappa=0.25,
                     dtype="float64", tolerance=0.03, iplane=2, c0=0.3, **knobs) -> Model:
    """A 6 x 3 lattice of spacing 0.5 and **linear members**: the marker
    moves along ``(1, kappa)`` and the curvature of its map before the
    plane is the kernel's own ``t_x t_y`` term.  ``s``: how far past the
    plane (at ``x = 1``) the first cell's polynomial has its fixed point,
    along the line; ``e1``: how far before it the first pass lands."""
    shape, h = (6, 3), 0.5
    cross = Q * h * h / kappa
    table = np.zeros(shape)
    start = -(a_s - 0.5 * cross / h) * h
    table[iplane - 1, 0], table[iplane - 1, 1] = start, start - cross
    # Past the plane the two rows the marker reads are equal: straight lines.
    column = table[:, 0].copy()
    _lines_after(column, iplane, after, h)
    table[iplane + 1:, 0] = table[iplane + 1:, 1] = column[iplane + 1:]
    table[:iplane - 1, 0], table[:iplane - 1, 1] = table[iplane - 1, 0], table[iplane - 1, 1]
    table[:, 2] = table[:, 1]
    w = (table - c0).ravel()
    lattice = ps.Cfg(N=shape, m=1, origin=(0.0, 0.0), spacing=(h, h), dtype=dtype)
    on_plane = np.array([iplane * h, 0.5 * h])
    line = np.array([1.0, kappa])

    def read(tau):
        return float(w @ ps.ref_scatter(lattice, np.ones(1), (on_plane + tau * line)[None, :]))

    shift = s - read(0.0)
    alpha = (-e1 - s) / (read(-e1) - read(0.0))
    x0 = (alpha * ps.ref_scatter(lattice, np.ones(1), (on_plane - e1 * line)[None, :])
          + (1 - alpha) * ps.ref_scatter(lattice, np.ones(1), on_plane[None, :]))
    p0 = on_plane - e1 * line
    return Model(shape=shape, origin=(0.0, 0.0), spacing=(h, h), dtype=dtype,
                 tolerance=tolerance, rtol=tolerance, x0=tuple(x0), pos0=tuple(p0),
                 w=tuple(w), u=tuple(np.zeros(w.size)), q=0.0,
                 v=tuple(on_plane + shift * line - p0), dirn=tuple(line),
                 plane=float(on_plane[0]), **knobs)


def member_curvature(*, s=1e-6, e1=0.002, a_s=0.5, after=(1.2, 0.5), Q=2.0, dtype="float64",
                     tolerance=0.01, h=0.5, origin=0.0, n=6, iplane=2, c0=0.2,
                     **knobs) -> Model:
    """A lattice of *n* points and a member with a **smooth quadratic
    response**: ``g(p) = c + s - a_s z + Q z**2`` with ``z = c - p`` before
    the plane ``c``, the lines of *after* past it."""
    c = origin + iplane * h
    w, u = np.zeros(n), np.zeros(n)
    w[iplane], w[iplane - 1] = a_s * h - c0, -c0
    w[:iplane - 1] = w[iplane - 1]
    _lines_after(w, iplane, after, h)
    u[iplane - 1] = 1.0
    shift = c + s - w[iplane]
    p0 = c - e1
    alpha = (p0 - shift - w[iplane + 1]) / (w[iplane] - w[iplane + 1])
    x0 = np.zeros(n)
    x0[iplane], x0[iplane + 1] = alpha, 1 - alpha
    return Model(shape=(n,), origin=(origin,), spacing=(h,), dtype=dtype, tolerance=tolerance,
                 rtol=tolerance, x0=tuple(x0), pos0=(p0,), w=tuple(w), u=tuple(u), q=Q * h * h,
                 v=(shift - p0,), dirn=(1.0,), plane=float(c), **knobs)


# --------------------------------------------------------------------------
# the graph under test
# --------------------------------------------------------------------------
def _params(mo: Model) -> dict:
    kind = np.dtype(mo.dtype)
    return {"w": np.asarray(mo.w, kind), "u": np.asarray(mo.u, kind),
            "q": np.asarray(mo.q, kind), "v": np.asarray(mo.v, kind),
            "dirn": np.asarray(mo.dirn, kind)}


def build(mo: Model, *, diagnostics: bool = True):
    """The pair as a graph: one ``multilinear_grid`` scatter read at the
    marker's position (source-anchored), one plain edge back."""
    import jax.numpy as jnp
    from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import BoundaryInputSpec, SimulationNode

    kind = jnp.dtype(mo.dtype)
    n, d = mo.size, mo.d

    class Grid(SimulationNode):
        def initial_state(self):
            return {"x": jnp.asarray(mo.x0, kind)}

        def boundary_input_spec(self):
            return {"deposit": BoundaryInputSpec(shape=(n,), dtype=kind)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            return {"x": boundary_inputs.get("deposit", state["x"]) + 0 * state["x"]}

    class Marker(SimulationNode):
        def initial_state(self):
            return {"pos": jnp.asarray(mo.pos0, kind).reshape(1, d), "y": jnp.ones((1,), kind)}

        def boundary_input_spec(self):
            return {"field": BoundaryInputSpec(shape=(n,), dtype=kind)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = {**self.params, **(params or {})}
            field = boundary_inputs.get("field", jnp.zeros((n,), kind))
            read = p["u"] @ field
            drive = p["w"] @ field + p["q"] * read * read
            return {"pos": state["pos"] + p["v"][None, :] + drive * p["dirn"][None, :],
                    "y": state["y"]}

    gm = GraphManager()
    made = {"G": Grid("G", 0.01), "M": Marker("M", 0.01, **_params(mo))}
    for name in mo.order:
        gm.add_node(made[name])
    gm.add_edge("G", "M", "x", "field")
    gm.add_edge("M", "G", "y", "deposit", geometry=("source", "pos"),
                mapping=multilinear_grid_mapping(
                    mode="conservative", origin=list(mo.origin), spacing=list(mo.spacing),
                    shape=list(mo.shape), n_points=1))
    knobs = {"tolerance": mo.tolerance} if mo.norm == "l2" else {"rtol": mo.rtol}
    gm.add_coupling_group(["G", "M"], solver="ift", diagnostics=diagnostics,
                          convergence_norm=mo.norm, iteration_mode=mo.mode,
                          acceleration=mo.acceleration, max_iterations=mo.max_iterations,
                          **knobs)
    return gm


def load(mo: Model, gm) -> None:
    """*mo*'s coefficients and initial state into a compiled graph of its
    structure (no recompile)."""
    import jax.numpy as jnp

    kind = jnp.dtype(mo.dtype)
    tree = gm.params
    for name, value in _params(mo).items():
        tree["nodes"]["M"][name] = jnp.asarray(value, kind)
    gm.params = tree
    gm.set_node_state("G", {"x": jnp.asarray(mo.x0, kind)})
    gm.set_node_state("M", {"pos": jnp.asarray(mo.pos0, kind).reshape(1, mo.d),
                            "y": jnp.ones((1,), kind)})


def state_flat(mo: Model, gm):
    grid, marker = gm.get_node_state("G"), gm.get_node_state("M")
    return np.concatenate([np.asarray(grid["x"], np.float64).ravel(),
                           np.asarray(marker["pos"], np.float64).ravel(),
                           np.asarray(marker["y"], np.float64).ravel()])


def step_report(mo: Model, gm):
    """``(x, c, report, slots)`` of one step: the returned state, the
    pre-step state, the group's report and its two lattice-plane slots."""
    c = state_flat(mo, gm)
    gm.step()
    x = state_flat(mo, gm)
    report = dict(next(iter(gm.coupling_diagnostics().values())))
    meta = gm._state["_meta"]                                          # noqa: SLF001
    slots = {name: float(meta[f"coupling_{KEY}_geometry_{name}"])
             for name in ("plane_limit", "plane_margin")}
    return x, c, report, slots
