"""The spectral bound of an interface-norm group whose field several edges read.

``coupling_residual_interface`` sums over a group's internal *edges*: a
field that ``k`` of them read is counted ``k`` times.  The spectral analysis
behind ``spectral_error_bound`` took its weights from the *fields*, each
once, wherever no edge carried a mapping or a transform.  A residual
measured in one norm times a resolvent norm measured in another bounds
nothing: on a star whose hub's field every leaf reads -- the shape the
coupling guide's "wide star" row recommends the interface norm for -- the
residual sits on the leaves (counted once each) and the error it implies on
the hub (counted once per leaf), and the bound read 0.73, 0.52, 0.37 and
0.26 of the true distance at 2, 4, 8 and 16 leaves with
``spectral_usable=True`` (Jacobi, float64, jaxlib 0.11.0; 0.29 in float32 at
``rtol=1e-4``).  With an identity transform on every edge the same star took
the analysis on the reading and read 1.005 (CPL-088, MADD-ANO-213).

Such a group's report is now taken on the reading too
(``graph_manager._reading_is_the_fields`` decides, statically).  The oracle
is the float64 fixed point of the affine map as the step holds it, and the
distance is the interface norm's: every internal edge's source value, its
change over ``rtol`` times its own ``max|v|`` at the returned state, the RMS
over every edge.
"""

from __future__ import annotations

import math
import os
import types
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import graph_manager as gmod
from maddening.core.coupling.acceleration import coupling_residual_interface
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import resolve_transform
from tests.core import coupling_domains as cd

F32, F64 = jnp.float32, jnp.float64
RTOL = 1e-6


# ---------------------------------------------------------------------------
# A star: the hub's field is read by every leaf
# ---------------------------------------------------------------------------
class _Relay(SimulationNode):
    """``x <- b + sum_j g[j] * u_j``: one scalar field, one evaluation."""

    def __init__(self, name, gains, bias, x0, dtype):
        super().__init__(name, 1.0, g=jnp.asarray(np.asarray(gains), dtype),
                         b=jnp.asarray(bias, dtype))
        self._x0, self._n, self._dtype = x0, len(gains), dtype

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, self._dtype)}

    def boundary_input_spec(self):
        return {f"u{j}": BoundaryInputSpec(shape=(), dtype=self._dtype,
                                           default=jnp.asarray(0.0, self._dtype))
                for j in range(self._n)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        acc = p["b"]
        for j in range(self._n):
            acc = acc + p["g"][j] * boundary_inputs[f"u{j}"]
        return {"x": acc.astype(self._dtype)}

    def update_evaluations(self):
        return 1.0


def _star(leaves, dtype):
    """``(A, b, x*, x0)`` in float64, as *dtype* holds the gains and biases.

    ``h = b_0 + sum_i c_i l_i``, ``l_i = b_i + g_i h``.  The hub responds
    strongly to leaf 0, which responds weakly to it (a loop gain of 0.5),
    and the fixed point is 1 on every field (to the dtype's rounding of
    the biases), so the relative gains are the gains.
    """
    c = np.full(leaves, 0.01)
    c[0] = 5.0
    g = np.full(leaves, 0.001)
    g[0] = 0.1
    A = np.zeros((leaves + 1, leaves + 1))
    A[0, 1:], A[1:, 0] = c, g
    b = (np.eye(leaves + 1) - A) @ np.ones(leaves + 1)
    held = np.dtype(dtype)
    A, b = A.astype(held).astype(np.float64), b.astype(held).astype(np.float64)
    star = np.linalg.solve(np.eye(leaves + 1) - A, b)
    start = star[1:] * (1.0 + 1e-2 * np.random.default_rng(1).standard_normal(leaves))
    return A, b, star, np.concatenate([[A[0, 1:] @ start + b[0]], start])


def _names(leaves):
    return ["h"] + [f"l{i}" for i in range(leaves)]


_STARS: dict = {}


def _star_graph(leaves, dtype=F64, *, mode="jacobi", norm="interface", acceleration="none",
                cap=200, rtol=RTOL, identity=False, cached=True):
    """The star, compiled once per configuration (``cached=False``: a fresh one)."""
    key = (leaves, np.dtype(dtype).name, mode, norm, acceleration, cap, rtol, identity)
    if cached and key in _STARS:
        return _STARS[key]
    A, b, _star_point, x0 = _star(leaves, dtype)
    with cd.x64(dtype == F64):
        gm = GraphManager()
        gm.add_node(_Relay("h", A[0, 1:], b[0], x0[0], dtype))
        for i in range(leaves):
            gm.add_node(_Relay(f"l{i}", [A[1 + i, 0]], b[1 + i], x0[1 + i], dtype))
        transform = resolve_transform("identity") if identity else None
        for i in range(leaves):
            gm.add_edge(f"l{i}", "h", "x", f"u{i}", transform=transform)
            gm.add_edge("h", f"l{i}", "x", "u0", transform=transform)
        scale = {"tolerance": rtol} if norm == "l2" else {"rtol": rtol}
        gm.add_coupling_group(_names(leaves), iteration_mode=mode, convergence_norm=norm,
                              acceleration=acceleration, max_iterations=cap,
                              diagnostics=True, **scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
    if cached:
        _STARS[key] = gm
    return gm


def _stepped(gm, leaves, dtype=F64):
    """``(report, returned state as float64)`` of one step from the star's start."""
    with cd.x64(dtype == F64):
        gm.reset_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.step()
            report = dict(gm.coupling_diagnostics()["+".join(sorted(_names(leaves)))])
        x = np.array([float(gm.get_node_state(n)["x"]) for n in _names(leaves)], np.float64)
    return report, x


def _per_edge(dx, x, leaves, rtol, pair=None):
    """The interface norm on a star: the hub once per leaf that reads it,
    each leaf once, over ``rtol`` times the field's magnitude (over *x* and
    *pair* where the norm compares two iterates)."""
    ref = np.abs(x) if pair is None else np.maximum(np.abs(x), np.abs(pair))
    q = (dx / (rtol * ref)) ** 2
    return float(np.sqrt((leaves * q[0] + q[1:].sum()) / (2 * leaves)))


def _per_field(dx, x, rtol):
    """The same RMS with every field counted once: the mixed norm's."""
    return float(np.sqrt(np.mean((dx / (rtol * np.abs(x))) ** 2)))


def _assert_bound_covers(report, true, label):
    assert report["spectral_usable"] is True, (label, report)
    assert math.isfinite(report["spectral_error_bound"]), (label, report)
    assert report["spectral_error_bound"] >= true, (
        f"{label}: bound {report['spectral_error_bound']:.4e} is "
        f"{report['spectral_error_bound'] / true:.3f} of the distance {true:.4e} ({report})")


def test_the_interface_norm_counts_a_field_once_for_every_edge_that_reads_it():
    """What the bound must be a bound in: on a star of four leaves the
    residual holds the hub's change four times and each leaf's once."""
    leaves = 4
    gm = _star_graph(leaves)
    internal = [e for e in gm._edges]                                       # noqa: SLF001
    assert len(internal) == 2 * leaves
    rng = np.random.default_rng(7)
    old = 1.0 + 0.1 * rng.standard_normal(leaves + 1)
    new = old * (1.0 + 1e-5 * rng.standard_normal(leaves + 1))

    def state(v):
        return {n: {"x": jnp.asarray(v[i], F64)} for i, n in enumerate(_names(leaves))}

    with cd.x64(True):
        got = float(coupling_residual_interface(state(new), state(old), internal, 0.0, RTOL))
    want = _per_edge(new - old, new, leaves, RTOL, pair=old)
    assert got == pytest.approx(want, rel=1e-5)
    # ... which is not the fields once each: the two norms differ here.
    once = float(np.sqrt(np.mean(((new - old) / (RTOL * np.maximum(abs(new), abs(old)))) ** 2)))
    assert abs(want - once) > 1e-2 * once, "premise: the multiplicity is visible"


#: ``(leaves, dtype, sweep, acceleration, cap, rtol)``.  Per push: the
#: converged star at two and four leaves, the same star capped early (the
#: bound is claimed for any iterate) and float32 at a tolerance its float
#: floor does not cover.
_STAR_CELLS = [
    (2, "f64", "jacobi", "none", 200, RTOL),
    (4, "f64", "jacobi", "none", 200, RTOL),
    (4, "f64", "jacobi", "none", 4, RTOL),
    (4, "f32", "jacobi", "none", 200, 1e-4),
    pytest.param(16, "f64", "jacobi", "none", 200, RTOL, marks=pytest.mark.slow),
    pytest.param(16, "f64", "jacobi", "none", 4, RTOL, marks=pytest.mark.slow),
    pytest.param(8, "f32", "jacobi", "none", 200, 1e-4, marks=pytest.mark.slow),
    pytest.param(4, "f64", "gauss-seidel", "none", 200, RTOL, marks=pytest.mark.slow),
    pytest.param(4, "f64", "jacobi", "aitken", 200, RTOL, marks=pytest.mark.slow),
    pytest.param(4, "f64", "jacobi", "iqn-ils", 200, RTOL, marks=pytest.mark.slow),
]


# Per push: tests/core/test_coupling_spectral_bound_where_edges_share_a_field.py::test_a_usable_bound_covers_the_distance_on_a_star
# (two and four leaves, capped early, float32; the slow cells are wider stars, the other sweep and the accelerations)
@pytest.mark.parametrize("leaves, dtype, mode, acceleration, cap, rtol", _STAR_CELLS)
def test_a_usable_bound_covers_the_distance_on_a_star(leaves, dtype, mode, acceleration, cap,
                                                      rtol):
    """CPL-088 where the hub's field is read by every leaf: the bound covers
    the distance to the exact fixed point in the norm the residual is in,
    the hub counted once per edge that reads it."""
    dt = {"f64": F64, "f32": F32}[dtype]
    _A, _b, star, _x0 = _star(leaves, dt)
    gm = _star_graph(leaves, dt, mode=mode, acceleration=acceleration, cap=cap, rtol=rtol)
    report, x = _stepped(gm, leaves, dt)
    true = _per_edge(x - star, x, leaves, rtol)
    # Plain iteration stops within a few thresholds of the fixed point; a
    # quasi-Newton group lands on it (a distance of 2e-9 thresholds).
    assert true > (0.5 if acceleration == "none" else 0.0), (
        f"fixture premise: a distance worth bounding ({true})")
    _assert_bound_covers(report, true, (leaves, dtype, mode, acceleration, cap))
    if acceleration == "none" and mode == "jacobi" and dtype == "f64":
        # Plain Jacobi on an affine map: the bound is the distance, to the
        # analysis' own margin (a fix that only inflated it would pass above).
        assert report["spectral_error_bound"] <= 1.1 * true, (leaves, cap, report, true)


# Per push: tests/core/test_coupling_spectral_bound_where_edges_share_a_field.py::test_an_identity_transform_on_every_edge_of_a_star_changes_no_bound
# (Jacobi, the sweep the bound fell short under; Gauss-Seidel is the same check)
@pytest.mark.parametrize("mode", ["jacobi", pytest.param("gauss-seidel", marks=pytest.mark.slow)])
def test_an_identity_transform_on_every_edge_of_a_star_changes_no_bound(mode):
    """A registered ``"identity"`` on every edge delivers exactly the raw
    fields, so the star reads the same bound, radius and flags with it and
    without.  (Without it the analysis was the fields' and read half as
    much: 1.55 against 3.01 thresholds at four leaves.)"""
    leaves = 4
    plain, _x = _stepped(_star_graph(leaves, mode=mode), leaves)
    through, _x = _stepped(_star_graph(leaves, mode=mode, identity=True), leaves)
    assert plain["spectral_usable"] is True and through["spectral_usable"] is True
    assert plain["rho_spectral"] == pytest.approx(through["rho_spectral"], rel=1e-9)
    assert plain["spectral_error_bound"] == pytest.approx(through["spectral_error_bound"],
                                                          rel=1e-9)
    # The gradient bound keeps the state's own analysis, with or without.
    assert plain["gradient_bound_usable"] == through["gradient_bound_usable"]
    if plain["gradient_bound_usable"]:
        assert plain["gradient_relative_error_bound"] == pytest.approx(
            through["gradient_relative_error_bound"], rel=1e-9)


# Per push: tests/core/test_coupling_spectral_bound_where_edges_share_a_field.py::test_a_usable_bound_covers_the_distance_on_a_star
# (the same star under the interface norm; these are the two norms that count a field once)
@pytest.mark.slow
@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_the_norms_that_count_a_field_once_keep_their_bound_on_a_star(norm):
    """The mixed and L2 norms read each field once whatever reads it: their
    bound is in the fields' own weights and covers the distance in them."""
    leaves = 4
    _A, _b, star, _x0 = _star(leaves, F64)
    report, x = _stepped(_star_graph(leaves, norm=norm), leaves)
    if norm == "mixed":
        true = _per_field(x - star, x, RTOL)
    else:
        true = float(np.sqrt(np.sum(((x - star) / np.abs(x)) ** 2)))
    _assert_bound_covers(report, true, norm)


# ---------------------------------------------------------------------------
# Which groups take the analysis on the reading
# ---------------------------------------------------------------------------
def _edge(source, field="x", transform=None, mapping=None):
    return types.SimpleNamespace(source_node=source, source_field=field,
                                 transform=transform, mapping=mapping)


def test_the_reading_is_the_fields_only_where_each_is_read_once_as_it_is():
    """The static rule: the state's weights are the interface norm's exactly
    when every internal edge delivers its floating source field unchanged
    and no field is read twice."""
    is_fields = gmod._reading_is_the_fields                                  # noqa: SLF001
    floats = {"a": ("x", "y"), "b": ("x",), "c": ("x",)}
    assert is_fields([], floats)
    assert is_fields([_edge("a"), _edge("b")], floats), "a pair"
    assert is_fields([_edge("a"), _edge("b"), _edge("c")], floats), "a ring"
    assert is_fields([_edge("a", "x"), _edge("a", "y"), _edge("b")], floats), (
        "two fields of one node, each read once")
    assert not is_fields([_edge("a"), _edge("a"), _edge("b")], floats), "a field read twice"
    assert not is_fields([_edge("b"), _edge("a"), _edge("c"), _edge("a")], floats), (
        "read twice, the two reads apart")
    assert not is_fields([_edge("a", transform=lambda v: v), _edge("b")], floats), "a transform"
    assert not is_fields([_edge("a", mapping=object()), _edge("b")], floats), "a mapping"
    # A field the norm does not read (not floating) is in neither count.
    assert is_fields([_edge("a", "n"), _edge("a", "n"), _edge("b")], floats)
    assert is_fields([_edge("a", "n", transform=lambda v: v), _edge("b")], floats)


def _pair_graph(identity=False):
    """A plain float32 pair under the interface norm with the report's analysis on."""
    transform = resolve_transform("identity") if identity else None
    gm = GraphManager()
    gm.add_node(_Relay("a", [0.5], 1.0, 0.0, F32))
    gm.add_node(_Relay("b", [0.5], 0.0, 0.0, F32))
    gm.add_edge("b", "a", "x", "u0", transform=transform)
    gm.add_edge("a", "b", "x", "u0", transform=transform)
    gm.add_coupling_group(["a", "b"], convergence_norm="interface", rtol=RTOL,
                          max_iterations=3, diagnostics=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


_SECOND_SPECTRUM = {
    "pair": (lambda: _pair_graph(), False),
    "star": (lambda: _star_graph(2, F32, cap=3, cached=False), True),
    "pair-identity": (lambda: _pair_graph(identity=True), True),
    "star-mixed": (lambda: _star_graph(2, F32, norm="mixed", cap=3, cached=False), False),
}


# Per push: tests/core/test_coupling_spectral_bound_where_edges_share_a_field.py::test_only_a_group_whose_reading_is_not_its_fields_takes_the_second_spectrum
# (the pair and the star; the slow cells are the transform and the mixed norm, the same count)
@pytest.mark.parametrize("which", ["pair", "star",
                                   pytest.param("pair-identity", marks=pytest.mark.slow),
                                   pytest.param("star-mixed", marks=pytest.mark.slow)])
def test_only_a_group_whose_reading_is_not_its_fields_takes_the_second_spectrum(which,
                                                                               monkeypatch):
    """The step traces the analysis on the reading for a star under the
    interface norm (and for a transformed edge), and not for a pair whose
    two fields are each read once: that group's report, and what its
    diagnostics cost (CPL-013), are what they were."""
    build, expected = _SECOND_SPECTRUM[which]
    calls = []
    real = gmod._interface_spectral_rate_at                                 # noqa: SLF001

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gmod, "_interface_spectral_rate_at", counted)
    gm = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.step()
    assert bool(calls) is expected, (which, len(calls))


# ---------------------------------------------------------------------------
# One node reading a field twice, in every numeric domain
# ---------------------------------------------------------------------------
class _TwoPorts(cd.Lin):
    """``x <- g * (u + v) + c``: a :class:`~tests.core.coupling_domains.Lin`
    with a second port (absent from the inputs of a member nothing feeds
    through it)."""

    def boundary_input_spec(self):
        spec = super().boundary_input_spec()
        return {**spec, "v": spec["u"]}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = jnp.asarray(boundary_inputs["u"]).astype(self._dtype)
        if "v" in boundary_inputs:
            u = u + jnp.asarray(boundary_inputs["v"]).astype(self._dtype)
        return super().update(state, {**boundary_inputs, "u": u}, dt, params=params)


#: Every domain, ``run_adaptive`` slow (it compiles its step on every call).
EVERY = list(cd.EVERY) + [pytest.param(cd.ADAPTIVE, marks=pytest.mark.slow)]
#: ``x_a <- 5 x_b - 4``, ``x_b <- 0.0625 (x_a + x_a) + 0.875``: ``a`` responds
#: strongly to ``b``, which reads ``a`` twice and responds weakly (a loop
#: gain of 0.625); the fixed point is 1 on both.  Exact in bfloat16.
TWICE_GAINS = (5.0, 0.0625)
TWICE_FORCING = (-4.0, 0.875)
#: ``b`` starts at what it reads of ``a``, so the Jacobi passes move the two
#: members in turn, and after three the change the loop measured and the
#: error it implies sit on different members: counted once per field, the
#: bound read 0.741 of the distance in float32 and float64.
TWICE_START = (2.0, 1.125)
TWICE_PASSES = 3
_TWICE: dict = {}


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _rtol(domain) -> float:
    """A tolerance the dtype resolves, far below the pair's distance at the cap."""
    return 0.02 if _sixteen(domain) else 1e-5


def _twice_graph(label):
    """The pair whose ``a`` is read by two edges, in *label*'s domain: three
    Jacobi passes with the report's analysis on, compiled once per module."""
    if label not in _TWICE:
        d = cd.DOMAINS[label]
        with cd.entered(d):
            _TWICE[label] = cd.pair(
                d, g=TWICE_GAINS, c=TWICE_FORCING, node=_TwoPorts,
                extra_edges=(("a", "b", "x", "v"),), convergence_norm="interface",
                iteration_mode="jacobi", rtol=_rtol(d), max_iterations=TWICE_PASSES,
                diagnostics=True)
    return _TWICE[label]


def _twice_solves(label) -> list:
    d = cd.DOMAINS[label]
    gm = _twice_graph(label)
    with cd.entered(d):
        moves = (0.0, 0.25, -0.125, 0.5)
        seq = [cd.params_with(gm, g=TWICE_GAINS,
                              c=tuple(np.asarray(c) * (1.0 + m) for c in TWICE_FORCING))
               for m in moves]
        if d.predictor or d.restart:
            out = cd.run_sequence(d, gm, seq, x0=TWICE_START)
        else:
            out = cd.run(d, gm, [seq[0]], x0=TWICE_START)
        cd.assert_in_domain(d, gm, out)
    return out


# Per push: tests/core/test_coupling_spectral_bound_where_edges_share_a_field.py::test_a_usable_bound_covers_the_distance_where_a_node_reads_a_field_twice
# (every domain but run_adaptive, the same check)
@pytest.mark.parametrize("label", EVERY)
def test_a_usable_bound_covers_the_distance_where_a_node_reads_a_field_twice(label):
    """CPL-088 on a pair whose ``a`` feeds two ports of ``b``, in every
    numeric domain: the norm counts ``a`` twice and ``b`` once, and the
    bound covers the distance to the exact fixed point counted so."""
    d = cd.DOMAINS[label]
    assert len(_twice_graph(label)._edges) == 3                             # noqa: SLF001
    for s in _twice_solves(label):
        r = s.report
        ga, gb = s.gains()
        ca, cb = s.forcing()
        xa_star = (ca + ga * cb) / (1.0 - 2.0 * ga * gb)
        xb_star = 2.0 * gb * xa_star + cb
        xa, xb = (np.asarray(s.x(n), np.float64) for n in ("a", "b"))
        # The three edges: b -> a, and a -> b twice.
        terms = np.concatenate([np.abs(xb - xb_star) / np.max(np.abs(xb)),
                                np.abs(xa - xa_star) / np.max(np.abs(xa)),
                                np.abs(xa - xa_star) / np.max(np.abs(xa))]) / _rtol(d)
        dist = float(np.sqrt(np.mean(terms ** 2)))
        assert dist > 1.0 or d.adaptive, (
            f"{label}: fixture premise: outside the tolerance at the cap ({dist})")
        assert r["spectral_usable"] is True, (label, r)
        assert r["spectral_error_bound"] >= dist, (
            f"{label}: bound {r['spectral_error_bound']:.4e} is "
            f"{r['spectral_error_bound'] / dist:.3f} of the distance {dist:.4e} with a "
            f"field counted for both edges that read it ({r})")
