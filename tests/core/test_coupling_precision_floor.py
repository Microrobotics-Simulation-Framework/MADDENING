"""A residual of ``0.0`` is a statement about float32, not about the fixed point.

The report's two bound keys, ``spectral_error_bound`` and
``gradient_relative_error_bound``, are built on the residual the group
measured, ``N(F~(x) - x)`` for the one-pass map as float arithmetic
evaluates it.  A distance bound needs the residual of the *exact* map,
and the two differ by the map's evaluation error.  On a slow group the
difference is everything: once ``(1 - rho) * |x - x*|`` is below half an
ulp a pass changes nothing, ``F~(x) == x`` bitwise, the residual is
exactly ``0.0`` -- and before this was fixed both keys read ``0.0`` with
their usable flags ``True`` on a state 38 348 ulps from its fixed point.

So each key adds the residual's float resolution,
:func:`~maddening.core.coupling.acceleration.residual_precision_floor`,
before amplifying it, and the report says when the residual is at that
floor (``precision_limited``).  ``converged`` and the criterion are
deliberately unchanged: a precision floor in the criterion would make a
tight float32 tolerance unreachable and move iteration counts across the
suite.  What is pinned here is that the bound keys are where a stalled
group shows.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    residual_precision_floor,
    spectral_error_bound,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

_EPS = float(np.finfo(np.float32).eps)


class _Map(SimulationNode):
    """``x <- a + g * u`` (``"affine"``) or ``x <- a + g * u**2`` (``"square"``).

    One evaluation per update, and it says so (``update_evaluations``):
    the fixtures here are about the floor's value, and an undeclared
    node's group is not reported usable at the floor (see
    ``test_an_undeclared_sub_stepped_node_is_not_usable_at_the_floor``).
    """

    def __init__(self, name, kind, a, g, x0):
        super().__init__(name=name, timestep=1.0, a=jnp.float32(a), g=jnp.float32(g))
        self._kind = kind
        self._x0 = x0

    def initial_state(self):
        return {"x": jnp.float32(self._x0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        return {"x": p["a"] + p["g"] * (u if self._kind == "affine" else u * u)}

    def update_evaluations(self):
        return 1


class _Relay(SimulationNode):
    """``x <- u``; one evaluation, declared."""

    def __init__(self, name, x0):
        super().__init__(name=name, timestep=1.0)
        self._x0 = x0

    def initial_state(self):
        return {"x": jnp.float32(self._x0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


def _relay_graph(kind, a, g, x0, **group_kw):
    gm = GraphManager()
    gm.add_node(_Map("a", kind, a, g, x0))
    gm.add_node(_Relay("b", x0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, max_iterations=50, tolerance=1e-6)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _l2_distance(gm, x_star):
    """The group's L2 norm between the returned state and ``x_star``."""
    total = 0.0
    for node in ("a", "b"):
        got = float(gm.get_node_state(node)["x"])
        ref = max(abs(got), abs(x_star))
        total += ((got - x_star) / ref) ** 2
    return math.sqrt(total)


# The linear relay: ``x_a <- g x_b + 1`` with ``g = 0.99999``.  Every
# condition ``spectral_error_bound`` lists holds (linear map, rank one,
# resolved spectrum); the only thing wrong with the old number was the
# float32 floor.
_SLOW_GAIN = 0.99999
_SLOW_G32 = float(np.float32(_SLOW_GAIN))
_SLOW_FIXED_POINT = 1.0 / (1.0 - _SLOW_G32)
#: Started 0.3% short of the fixed point, as a time-stepped run whose
#: fixed point moves slowly is: 38 348 ulps, and ``(1 - g) * |x - x*|``
#: is a quarter of an ulp, so the first pass changes nothing.
_SLOW_START = float(np.float32(_SLOW_FIXED_POINT * (1.0 - 3e-3)))


def test_a_stalled_float32_iterate_is_not_reported_at_its_fixed_point():
    """Residual ``0.0``, ``converged=True`` -- and a bound that still covers it.

    The criterion is unchanged and this pins that it is: one pass, a
    residual of exactly ``0.0``, ``converged=True``.  What changed is
    the bound: it is the float resolution of the residual times the
    amplification, which covers the true distance of 4.2e-3 (it read
    ``0.0`` before), and ``precision_limited`` says the residual is at
    that resolution.
    """
    gm = _relay_graph("affine", 1.0, _SLOW_GAIN, _SLOW_START)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    ulps = abs(float(gm.get_node_state("a")["x"]) - _SLOW_FIXED_POINT) / float(
        np.spacing(np.float32(_SLOW_FIXED_POINT)))
    assert ulps > 3e4, f"fixture premise: {ulps:.0f} ulps from the fixed point"
    # The criterion's behaviour on a stalled iterate, unchanged by design.
    assert d["residual"] == 0.0
    assert d["iterations"] == 1
    assert d["converged"] is True
    # The bound keys are where it shows.
    distance = _l2_distance(gm, _SLOW_FIXED_POINT)
    assert distance == pytest.approx(4.24e-3, rel=0.02), "fixture premise"
    assert d["spectral_usable"] is True
    assert d["spectral_error_bound"] >= distance, (
        f"bound {d['spectral_error_bound']:.3e} under the true distance "
        f"{distance:.3e} on a stalled iterate"
    )
    assert d["precision_limited"] is True
    # And it is the floor that carries it: the residual is zero, so the
    # bound is exactly the floor times the amplification the key applies.
    floor = PRECISION_FLOOR_ULPS * _EPS * math.sqrt(2.0)
    assert d["spectral_error_bound"] >= floor / (1.0 - d["rho_spectral"])


def test_a_stalled_linear_iterate_does_not_read_its_gradient_as_exact():
    """The gradient bound's ``distance == 0`` branch is no longer reached.

    ``d x*/d g = 1 / (1 - g)**2`` at the fixed point; the IFT rule
    evaluates the same expression at the iterate it was handed, which is
    3e-3 short, so the gradient is 3e-3 off.  The bound read ``0.0``.
    """
    gm = _relay_graph("affine", 1.0, _SLOW_GAIN, _SLOW_START)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]

    def fixed_point_of(params):
        g = _relay_graph("affine", 1.0, _SLOW_GAIN, _SLOW_START)
        return g.run_scan(1, params=params)["a"]["x"]

    params = _relay_graph("affine", 1.0, _SLOW_GAIN, _SLOW_START).params
    got = float(jax.grad(fixed_point_of)(params)["nodes"]["a"]["g"])
    exact = 1.0 / (1.0 - _SLOW_G32) ** 2
    true_error = abs(got - exact) / abs(got)
    assert true_error > 1e-3, "fixture premise: the gradient is measurably off"
    assert d["gradient_bound_usable"] is True
    assert d["gradient_relative_error_bound"] >= true_error, (
        f"gradient bound {d['gradient_relative_error_bound']:.3e} under the "
        f"true relative error {true_error:.3e}"
    )


def _square_fixed_point(a32, g32):
    """The attracting root of ``x = a + g x**2``, Newton-polished in float64."""
    x = (1.0 - math.sqrt(1.0 - 4.0 * g32 * a32)) / (2.0 * g32)
    for _ in range(50):
        x -= (a32 + g32 * x * x - x) / (2.0 * g32 * x - 1.0)
    return x


@pytest.mark.parametrize("offset", (-6e-5, 3e-5))
def test_a_stalled_nonlinear_iterate_does_not_read_its_gradient_as_exact(offset):
    """``x_a <- a + g u**2`` with ``F'(x*) = 0.999``: 1.5-3% off, and the bound read ``0.0``.

    Here the IFT gradient is off because the *slope* differs between the
    iterate and the fixed point, and the slope is amplified a
    thousandfold.  The honest answer at float32 is that no bound exists:
    the distance float32 can resolve about a group this slow (the
    floor times the amplification, ~9e-4) is a distance over which the
    Jacobian moves by about the gap ``1 - F'`` itself, so the
    Newton-Kantorovich check fails (``h`` near 1) and nothing measured
    at the returned iterate bounds the resolvent at the fixed point.
    The bound reads ``inf`` and is unusable -- where it read ``0.0``,
    usable, before.
    """
    slope, g = 0.999, 0.25
    a = (1.0 - (1.0 - slope) ** 2) / (4.0 * g)
    a32, g32 = float(np.float32(a)), float(np.float32(g))
    x_star = _square_fixed_point(a32, g32)
    x0 = float(np.float32(x_star + offset))
    gm = _relay_graph("square", a, g, x0)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["residual"] == 0.0 and d["converged"] is True, "fixture premise"

    def fixed_point_of(params):
        return _relay_graph("square", a, g, x0).run_scan(1, params=params)["a"]["x"]

    params = _relay_graph("square", a, g, x0).params
    got = float(jax.grad(fixed_point_of)(params)["nodes"]["a"]["a"])
    exact = 1.0 / (1.0 - 2.0 * g32 * x_star)
    true_error = abs(got - exact) / abs(got)
    assert true_error > 1e-2, "fixture premise: the gradient is percent-off"
    assert d["precision_limited"] is True
    assert d["spectral_error_bound"] > 0.0
    assert d["gradient_relative_error_bound"] != 0.0
    assert math.isinf(d["gradient_relative_error_bound"]), d
    assert d["gradient_bound_usable"] is False, d


def test_a_residual_above_the_floor_is_not_precision_limited():
    """The flag is about the residual's resolution, not about convergence.

    The ``rho = 0.25`` cycle stopped at ``tolerance=1e-2`` has a residual
    thousands of ulps above float32's resolution: converged, and not
    precision-limited.  Beside it the arithmetic of the key: the floor
    is added to the residual, not compared with it.
    """
    gm = _relay_graph("affine", 1.0, 0.25, 0.0, tolerance=1e-2)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["converged"] is True
    assert d["precision_limited"] is False
    floor = PRECISION_FLOOR_ULPS * _EPS * math.sqrt(2.0)
    assert d["residual"] > 100 * floor


def test_the_floor_is_added_to_the_residual_before_it_is_amplified():
    """``(residual + floor) * amplification`` -- the exact map's residual bound."""
    got = float(spectral_error_bound(1e-4, 0.9, 0.0, 12.0, floor=1e-5))
    assert got == pytest.approx((1e-4 + 1e-5) * 12.0, rel=1e-6)
    got = float(spectral_error_bound(1e-4, 0.95, 0.0, 12.0, floor=1e-5))
    assert got == pytest.approx((1e-4 + 1e-5) * 20.0, rel=1e-6), "radius form wins"
    assert float(spectral_error_bound(0.0, 0.5, 0.0, 1.0, floor=0.0)) == 0.0, (
        "a norm that reads nothing has no floor and bounds nothing"
    )
    assert math.isnan(float(spectral_error_bound(float("nan"), 0.5, 0.0, 1.0, floor=1e-5)))


def _state(**fields):
    return {"n": dict(fields), "m": {"y": jnp.full((3,), 2.0, jnp.float32)}}


def test_the_precision_floor_is_what_each_norm_can_resolve():
    """One unit of ``eps * max|field|`` per read entry, in the norm's own units.

    L2 sums over the entries it reads, the RMS norms do not; the dead
    band removes a field exactly as the residual's norm does, and an
    integer field is never read.
    """
    s = _state(x=jnp.array([1.0, -3.0], jnp.float32), k=jnp.int32(7))
    c = PRECISION_FLOOR_ULPS * _EPS
    assert float(residual_precision_floor(s, ["n", "m"], "l2")) == pytest.approx(
        c * math.sqrt(5.0), rel=1e-6)
    assert float(residual_precision_floor(s, ["n", "m"], "mixed", rtol=1e-4)) == pytest.approx(
        c / 1e-4, rel=1e-6)
    # ``y`` (max 2.0) inside a dead band of 2.5: only ``x`` (max 3.0) is read.
    assert float(residual_precision_floor(s, ["n", "m"], "l2", atol=2.5)) == pytest.approx(
        c * math.sqrt(2.0), rel=1e-6)
    assert float(residual_precision_floor(s, ["n", "m"], "l2", atol=10.0)) == 0.0
    # The interface norm reads edge sources only, through their transforms.
    edges = [EdgeSpec(source_node="m", source_field="y", target_node="n",
                      target_field="u")]
    assert float(residual_precision_floor(
        s, ["n", "m"], "interface", rtol=1e-3, interface_edges=edges)) == pytest.approx(
        c / 1e-3, rel=1e-6)
    assert float(residual_precision_floor(
        s, ["n", "m"], "interface", rtol=1e-3, interface_edges=())) == 0.0


#: The recorded disagreement between two compilations of the same pass,
#: in units of ``eps * max|field|``: the largest gap between the ``"ift"``
#: and ``"fori"`` residuals over the 480-cell sweep behind
#: ``test_coupling_solver_equivalence.py``, as ``PRECISION_FLOOR_ULPS``'s
#: derivation quotes it.  Too expensive to re-run here.
_CROSS_COMPILATION_ULPS = 0.82


def test_the_floor_has_headroom_over_a_dense_updates_evaluation_error():
    """The constant is calibrated, not arbitrary: twice both things it must cover.

    ``PRECISION_FLOOR_ULPS`` is justified by what it must cover -- the
    difference between a residual float32 computes and the exact map's.
    Its derivation names two parts: the evaluation error of one
    evaluation, measured here the way the docstring states it (a dense
    update ``A @ u + c`` in float32 against the same float32 operands in
    float64, near the fixed point, over random normal contractions of
    dimension 2-6, in units of ``eps * max|field|`` and in the L2 norm
    of the relay group), and the disagreement between two compilations
    of the same pass (recorded, 0.82).  Twice their *sum* is required:
    asking only for twice the first let ``PRECISION_FLOOR_ULPS = 1.5``
    pass -- a floor that one evaluation plus one recompilation already
    exceeds.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        n = int(rng.integers(2, 7))
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        A = ((Q * rng.uniform(-0.98, 0.98, n)) @ Q.T).astype(np.float32)
        c = rng.uniform(-2.0, 2.0, n).astype(np.float32)
        x_star = np.linalg.solve(np.eye(n) - A.astype(np.float64), c.astype(np.float64))
        u = (x_star * (1.0 + 1e-6 * rng.normal(size=n))).astype(np.float32)
        f32 = np.asarray(jnp.asarray(A) @ jnp.asarray(u) + jnp.asarray(c), np.float64)
        f64 = A.astype(np.float64) @ u.astype(np.float64) + c.astype(np.float64)
        ref = max(np.max(np.abs(f32)), np.max(np.abs(u)))
        err = (f32 - f64) / ref
        worst = max(worst, math.sqrt(2.0 * np.sum(err ** 2)) / (_EPS * math.sqrt(2.0 * n)))
    assert 0.0 < worst < 1.0, f"fixture premise: measured {worst:.3f}"
    covered = worst + _CROSS_COMPILATION_ULPS
    assert PRECISION_FLOOR_ULPS >= 2.0 * covered, (
        f"the floor ({PRECISION_FLOOR_ULPS} units) has less than 2x headroom "
        f"over one evaluation ({worst:.3f} units, measured) plus one "
        f"recompilation ({_CROSS_COMPILATION_ULPS} units, recorded)"
    )


class _Map16(SimulationNode):
    """``x <- a + g * u`` held in float16, reading a float32 partner; one evaluation, declared."""

    def update_evaluations(self):
        return 1

    def __init__(self, name, a, g, x0):
        super().__init__(name=name, timestep=1.0, a=jnp.float32(a), g=jnp.float32(g))
        self._x0 = x0

    def initial_state(self):
        return {"x": jnp.float16(self._x0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": (p["a"] + p["g"] * boundary_inputs["u"]).astype(jnp.float16)}


def _mixed_dtype_graph(x0):
    gm = GraphManager()
    gm.add_node(_Map16("a", 1.0, 0.99, x0))
    gm.add_node(_Relay("b", x0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u",
                transform=lambda v: v.astype(jnp.float32))
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=50,
                          tolerance=1e-6)
    gm.compile()
    return gm


def test_a_float16_field_beside_a_float32_one_is_floored_at_float16():
    """The gradient bound's floor, and the step it probes across, at each field's own resolution.

    ``x_a`` (float16) ``<- 1 + 0.99 u``, ``x_b`` (float32) ``<- x_a``,
    started at the lowest float16 value the map leaves where it is:
    3.1% short of ``x* = 100``, so the IFT gradient ``d x*/dg`` it
    returns, ``x_b / (1 - g)``, is 3.2% off ``a / (1 - g)**2``.  The
    flat fixed-point vector is float32, and the gradient bound took its
    floor from float32's eps -- 8192 times finer than the float16
    field's -- and probed the curvature across a step no float16 output
    could register: it read ``0.0`` with ``gradient_bound_usable=True``.
    The spectral bound beside it, whose floor is taken field by field,
    was right all along and is the control.
    """
    g32, a32 = float(np.float32(0.99)), 1.0
    x_star = a32 / (1.0 - g32)
    stalled = [
        float(v) for v in (np.nextafter(np.float16(x_star), np.float16(0))
                           - np.float16(0.0625) * np.float16(k) for k in range(64))
        if np.float16(np.float32(a32 + g32 * float(v))) == v
    ]
    x0 = min(stalled)
    gm = _mixed_dtype_graph(x0)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["residual"] == 0.0 and d["iterations"] == 1, f"fixture premise: stalled: {d}"
    assert (x_star - x0) / x_star > 0.03, f"fixture premise: {x0} is 3% short"
    distance = _l2_distance(gm, x_star)
    assert d["spectral_error_bound"] >= distance, "the control: per-field floor"
    x_b = float(gm.get_node_state("b")["x"])
    ift = x_b / (1.0 - g32)                   # the IFT tangent at the returned iterate
    exact = a32 / (1.0 - g32) ** 2            # d x*/dg at the fixed point
    true_error = abs(ift - exact) / abs(ift)
    assert true_error > 0.03, "fixture premise: the gradient is 3% off"
    assert d["gradient_relative_error_bound"] >= true_error, (
        f"gradient bound {d['gradient_relative_error_bound']:.3e} under the true "
        f"relative error {true_error:.3e} on a float16 field"
    )
    assert d["gradient_bound_usable"] is True, d


# ---------------------------------------------------------------------------
# A composite map: the floor is per evaluation
# ---------------------------------------------------------------------------

_HALF_ULP_OF_ONE = 2.0 ** -24


class _Relax(SimulationNode):
    """``dx/dt = (G u + C - x) / T`` by explicit Euler in ``substeps`` steps of ``h``.

    ``declare`` is what ``update_evaluations`` returns: ``None`` (not
    declared), or a count.
    """

    def __init__(self, name, substeps, h, c, *, timestep=1.0, declare=None):
        super().__init__(name=name, timestep=timestep, g=jnp.float32(0.5),
                         c=jnp.float32(c))
        self._n, self._h, self._declare = substeps, h, declare

    def initial_state(self):
        return {"x": jnp.float32(1.0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        x = state["x"]
        target = p["g"] * boundary_inputs["u"] + p["c"]
        h = jnp.float32(self._h)
        for _ in range(self._n):
            x = x + h * (target - x)
        return {"x": x}

    def update_evaluations(self):
        return self._declare


def _stalled_substep_graph(n, *, framework=False, declare=None):
    """A relay whose node integrates ``n`` explicit Euler sub-steps, started stalled at 1.0.

    ``h = 1/n`` and ``C`` put every increment ``h (T - x)`` at 0.95 of
    half an ulp of 1, so each sub-step is rounded away and the float32
    map returns its input -- while the exact map moves.  ``framework``
    has the graph sub-cycle a one-step node ``n`` times per pass
    (``subcycling=True``, node timestep ``1/n`` of its partner's)
    instead of the node looping inside ``update``.  Returns the graph
    and the exact fixed point (float64 closed form on the float32
    constants).
    """
    h = float(np.float32(1.0 / n))
    c = float(np.float32(0.5 + 0.95 * _HALF_ULP_OF_ONE / h))
    gm = GraphManager()
    if framework:
        gm.add_node(_Relax("a", 1, h, c, timestep=1.0 / n, declare=declare))
        kw = {"subcycling": True}
    else:
        gm.add_node(_Relax("a", n, h, c, declare=declare))
        kw = {}
    gm.add_node(_Relay("b", 1.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=50,
                          tolerance=1e-6, **kw)
    gm.compile()
    alpha = (1.0 - h) ** n
    x_star = (alpha + (1.0 - alpha) * c) / (1.0 - (1.0 - alpha) * float(np.float32(0.5)))
    return gm, x_star


@pytest.mark.parametrize("framework", (False, True), ids=("internal", "subcycled"))
@pytest.mark.parametrize("n", (20, 100))
def test_a_sub_stepped_node_is_floored_per_evaluation(n, framework):
    """``n`` sub-steps, counted: the bound covers the distance and says it is usable.

    Before the floor was counted per evaluation it was four units
    whatever the node did, and this group -- residual ``0.0``,
    ``converged=True`` -- read a bound 0.80x (``n = 20``) and 0.14x
    (``n = 100``) its true distance with ``spectral_usable=True``.  The
    framework's sub-cycling is counted without a declaration; a node
    that loops inside ``update`` declares ``n``.  Both forms here
    declare, so the flag holds.
    """
    gm, x_star = _stalled_substep_graph(n, framework=framework,
                                        declare=1 if framework else n)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["residual"] == 0.0 and d["iterations"] == 1, f"fixture premise: stalled: {d}"
    distance = _l2_distance(gm, x_star)
    flat = PRECISION_FLOOR_ULPS * _EPS * math.sqrt(2.0) / (1.0 - d["rho_spectral"])
    assert distance > flat, (
        f"fixture premise: {distance:.3e} is beyond the bound a flat floor "
        f"gives ({flat:.3e})"
    )
    assert d["precision_limited"] is True
    assert d["spectral_usable"] is True, d
    assert d["spectral_error_bound"] >= distance, (
        f"bound {d['spectral_error_bound']:.3e} under the true distance "
        f"{distance:.3e} for {n} sub-steps"
    )
    # The floor is the per-evaluation one times the count.
    floor = n * PRECISION_FLOOR_ULPS * _EPS * math.sqrt(2.0)
    assert d["spectral_error_bound"] >= floor / (1.0 - d["rho_spectral"]) * (1 - 1e-6)


@pytest.mark.parametrize("n", (20, 100))
def test_an_undeclared_sub_stepped_node_is_not_usable_at_the_floor(n):
    """The same node looping inside ``update`` without saying so: the flag is withheld.

    Nothing outside ``update`` can count its sub-steps, so the floor
    takes it as one evaluation, and the bound built on that reads below
    the true distance -- the premise asserted here.  Where the residual
    is at the floor (``precision_limited``) that unverified count is
    the whole bound, and neither ``spectral_usable`` nor
    ``gradient_bound_usable`` may say otherwise.
    """
    gm, x_star = _stalled_substep_graph(n)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _l2_distance(gm, x_star)
    assert d["residual"] == 0.0, "fixture premise: stalled"
    assert d["spectral_error_bound"] < distance, (
        "fixture premise: counted as one evaluation, the bound reads low"
    )
    assert d["precision_limited"] is True
    assert d["spectral_usable"] is False, d
    assert d["gradient_bound_usable"] is False, d
    # Declared, the same group is covered and usable (the control).
    declared, _ = _stalled_substep_graph(n, declare=n)
    declared.step()
    dd = declared.coupling_diagnostics()["a+b"]
    assert dd["spectral_usable"] is True
    assert dd["spectral_error_bound"] >= _l2_distance(declared, x_star)
    # With a zero residual both bounds are the floor carried through, so
    # declaring ``n`` evaluations scales each by exactly ``n`` -- the
    # gradient bound's floor inside the step as well as the report's.
    assert dd["spectral_error_bound"] / d["spectral_error_bound"] == pytest.approx(n, rel=1e-5)
    assert dd["gradient_relative_error_bound"] / d["gradient_relative_error_bound"] == (
        pytest.approx(n, rel=1e-3))


@pytest.mark.parametrize("n", (1, 4, 10, 20, 50, 100))
def test_the_floor_covers_explicit_euler_per_declared_sub_step(n):
    """The model behind the count: ``n`` Euler sub-steps stay under ``n`` floors.

    ``x <- x + h (T - x)``, ``h = 1/n``, evaluated in float32 against the
    same float32 operands in float64, in units of ``eps * max|field|``,
    over targets from 1e-7 to 1e-3 relative -- including the ones whose
    every increment rounds away.  The worst case must stay under
    ``PRECISION_FLOOR_ULPS * n``, and from ``n = 20`` on it exceeds the
    flat ``PRECISION_FLOOR_ULPS`` the floor used to be.
    """
    h = np.float32(1.0 / n)
    step = jax.jit(lambda x0, t: jax.lax.fori_loop(0, n, lambda i, x: x + h * (t - x), x0))
    rng = np.random.default_rng(n)
    worst = 0.0
    for _ in range(400):
        x0 = np.float32(rng.uniform(0.5, 2.0))
        t = np.float32(x0 * (1.0 + rng.choice((-1.0, 1.0)) * 10.0 ** rng.uniform(-7, -3)))
        got = float(step(jnp.float32(x0), jnp.float32(t)))
        x = float(x0)
        for _ in range(n):
            x = x + float(h) * (float(t) - x)
        worst = max(worst, abs(got - x) / (_EPS * max(abs(got), abs(x), float(x0))))
    assert worst <= PRECISION_FLOOR_ULPS * n, (
        f"{n} sub-steps: {worst:.2f} units, over {PRECISION_FLOOR_ULPS * n:.0f}"
    )
    if n >= 20:
        assert worst > PRECISION_FLOOR_ULPS, (
            f"premise: {n} sub-steps ({worst:.2f} units) exceed one evaluation's floor"
        )


def test_precision_limited_is_the_residual_at_or_below_the_floor():
    """The flag's threshold is the floor itself, read on the stored residual.

    The report derives the flag on the host from the residual the step
    stored.  Rewriting that one number walks it across the threshold:
    at the floor and at three quarters of it the residual is rounding,
    just above it it is not.  (Moving the threshold to half the floor
    went unnoticed by every other test.)
    """
    gm = _relay_graph("affine", 1.0, 0.25, 0.0, tolerance=1e-2)
    gm.step()
    floor = float(residual_precision_floor(gm._state, ["a", "b"], "l2"))
    assert floor == pytest.approx(PRECISION_FLOOR_ULPS * _EPS * math.sqrt(2.0), rel=1e-6)
    slot = "coupling_a+b_residual"
    dtype = gm._state["_meta"][slot].dtype
    for value, limited in ((0.75 * floor, True), (floor, True),
                           (float(np.nextafter(np.float32(floor), np.float32(1.0))), False),
                           (2.0 * floor, False)):
        stored = jnp.asarray(value, dtype)
        gm._state["_meta"][slot] = stored
        d = gm.coupling_diagnostics()["a+b"]
        assert d["residual"] == float(stored)
        assert d["precision_limited"] is limited, (value / floor, d)


@pytest.mark.parametrize("bad", (0.5, 0, float("nan"), float("inf"), True, "2"))
def test_an_invalid_evaluation_count_is_refused_at_compile(bad):
    """A count below one would shrink the floor under one evaluation's rounding."""
    gm = GraphManager()
    gm.add_node(_Relax("a", 2, 0.5, 0.5, declare=bad))
    gm.add_node(_Relay("b", 1.0))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=5, tolerance=1e-6)
    with pytest.raises(RuntimeError, match="update_evaluations"):
        gm.compile()
