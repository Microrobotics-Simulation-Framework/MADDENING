"""Under ``convergence_norm="interface"`` the spectral bound is in the norm's own reading.

The interface norm measures each internal edge's source value *after* the
edge's transform, each over its own magnitude (``coupling_residual_interface``).
The spectral analysis behind ``spectral_error_bound`` took its weights from
the raw source fields, so the bound multiplied a residual measured in one
set of coordinates by a resolvent norm measured in another: with an affine
offset on an edge (a unit conversion's 273.15) or a registered selection
(``"extract_last"``) it read 0.0014-0.098x the true distance with
``spectral_usable=True`` (CPL-088, round-6 coupling audit).  The report's
spectral triple is now taken on the reading itself
(``graph_manager._interface_spectral_rate_at``).

The oracle is the float64 fixed point of the affine map, and the distance is
the interface norm's: every internal edge's transformed value, its change
over ``rtol`` times its own ``max|v|`` at the returned state, the RMS over
every entry.
"""

from __future__ import annotations

import math
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import resolve_transform
from tests.conftest import EXAMPLES_COSTLY
from tests.core import coupling_domains as cd

F32 = np.float32
RTOL = 1e-6
KEY = "A+B"


# ---------------------------------------------------------------------------
# The audit's two fixtures: constants baked into the nodes
# ---------------------------------------------------------------------------


class _LinVec(SimulationNode):
    """``u = g * inp + c`` (vector field ``u``, scalar input ``inp``)."""

    def __init__(self, name, g, c, u0):
        super().__init__(name, 1.0)
        self._g = np.asarray(g, F32)
        self._c = np.asarray(c, F32)
        self._u0 = np.asarray(u0, F32)

    def initial_state(self):
        return {"u": jnp.asarray(self._u0)}

    def update(self, state, boundary_inputs, dt):
        inp = boundary_inputs.get("inp", jnp.float32(0.0))
        return {"u": jnp.asarray(self._g) * inp + jnp.asarray(self._c)}

    def update_evaluations(self):
        return 1.0


#: ``name -> (transform on A -> B, its float64 reading, gA, cA, gB, cB, uA0, uB0)``;
#: B -> A reads ``u[0]`` through ``"extract_first"``.
AUDIT_FIXTURES = {
    # A.u = [10, u1]; B reads u[-1]: fixed point s = u1 = 1.
    "extract-last": ("extract_last", lambda v: v[-1], [0.0, 50.0], [10.0, -49.0],
                     [0.01], [0.99], [10.0, 0.5], [0.99]),
    # A unit conversion's offset: uA = 0.7 s + 0.3, s = 0.7 (uA + 1000) + 0.3 - 700.
    "offset": (lambda v: v[0] + 1000.0, lambda v: v[0] + 1000.0, [0.7], [0.3], [0.7],
               [0.3 - 700.0], [0.9], [0.93]),
}


def _audit_graph(fixture, acceleration, cap, norm="interface", mode="jacobi"):
    tj, _tn, gA, cA, gB, cB, uA0, uB0 = AUDIT_FIXTURES[fixture]
    gm = GraphManager()
    gm.add_node(_LinVec("A", gA, cA, uA0))
    gm.add_node(_LinVec("B", gB, cB, uB0))
    gm.add_edge("A", "B", "u", "inp", transform=tj)
    gm.add_edge("B", "A", "u", "inp", transform="extract_first")
    gm.add_coupling_group(["A", "B"], max_iterations=cap, convergence_norm=norm, rtol=RTOL,
                          diagnostics=True, iteration_mode=mode, acceleration=acceleration)
    gm.compile()
    return gm


def _audit_exact(fixture, held_in=np.float64):
    """The float64 fixed point of the affine map, its constants as *held_in* rounds them."""
    _tj, tn, gA, cA, gB, cB, _uA0, _uB0 = AUDIT_FIXTURES[fixture]
    gA, cA, gB, cB = (np.asarray(v, held_in).astype(np.float64) for v in (gA, cA, gB, cB))
    t0 = tn(cA)
    slope = tn(gA + cA) - t0
    s = (gB[0] * t0 + cB[0]) / (1.0 - gB[0] * slope)
    uA = gA * s + cA
    return uA, gB * tn(uA) + cB


def _rms_over_own_magnitude(pairs, rtol=RTOL):
    """``sqrt(mean((|a - b| / (rtol * max|a|))**2))`` over every entry of every pair."""
    terms = np.concatenate([np.abs(a - b) / (rtol * np.max(np.abs(a))) for a, b in pairs])
    return float(np.sqrt(np.mean(terms ** 2)))


def _audit_distance(fixture, gm, held_in=np.float64):
    _tj, tn, *_ = AUDIT_FIXTURES[fixture]
    uA_s, uB_s = _audit_exact(fixture, held_in)
    uA = np.asarray(gm.get_node_state("A")["u"], np.float64)
    uB = np.asarray(gm.get_node_state("B")["u"], np.float64)
    return _rms_over_own_magnitude([
        (np.atleast_1d(tn(uA)), np.atleast_1d(tn(uA_s))),
        (np.atleast_1d(uB[0]), np.atleast_1d(uB_s[0])),
    ])


def _assert_bound_holds(d, true, label):
    assert math.isfinite(d["spectral_error_bound"]) or not d["spectral_usable"], (label, dict(d))
    if d["spectral_usable"] and true > 0:
        assert d["spectral_error_bound"] >= true, (
            label, d["spectral_error_bound"] / true, d["spectral_error_bound"], true, dict(d))


@pytest.mark.parametrize("cap", [2, 3])
@pytest.mark.parametrize("acceleration", ["none", "aitken"])
@pytest.mark.parametrize("fixture", sorted(AUDIT_FIXTURES))
def test_the_bound_holds_in_the_transformed_reading_on_the_audit_fixtures(fixture, acceleration,
                                                                         cap):
    """``spectral_error_bound >= ||x - x*||`` in the interface norm, wherever usable.

    Before, ``extract_last`` read 0.078x and the offset 0.0014x at ``cap=2``.
    """
    gm = _audit_graph(fixture, acceleration, cap)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["spectral_usable"], dict(d)        # the fixture premise: a resolved spectrum
    _assert_bound_holds(d, _audit_distance(fixture, gm), (fixture, acceleration, cap))


def test_a_stalled_transformed_pair_is_bounded_through_the_readings_float_floor():
    """Where the residual reads zero the bound is the reading's float floor, amplified.

    Gauss-Seidel with Aitken lands the ``extract_last`` pair on its float32
    fixed point by the third pass: the residual is exactly 0.0, the state
    is still a rounding away from the fixed point of the map its float32
    constants define (0.66 in the norm's units), and what bounds that is
    the floor taken on the reading (each transformed value at its own
    eps), which the residual is added to before it is amplified.
    """
    gm = _audit_graph("extract-last", "aitken", 3, mode="gauss-seidel")
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    true = _audit_distance("extract-last", gm, held_in=F32)
    assert d["residual"] == 0.0 and d["precision_limited"] and true > 0, (dict(d), true)
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, "stalled")


@pytest.mark.parametrize("gain", [0.9, 0.999])
def test_the_bound_holds_in_the_returned_readings_weights_on_a_growing_pair(gain):
    """``A <- B + 1``, ``B <- gain * A`` from zero, read through a selection on each edge.

    The residual divides each transformed value by the larger of its
    magnitudes over the pair the pass compared, the bound's factor is in
    the returned state's weights, and on a pair still growing toward its
    fixed point the two differ: the factor carries their ratio on the
    reading as it does on the state (MADD-ANO-146; without it the
    untransformed pair read 0.94x at this cap).
    """
    gm = GraphManager()
    gm.add_node(_LinVec("A", [1.0], [1.0], [0.0]))
    gm.add_node(_LinVec("B", [gain], [0.0], [0.0]))
    gm.add_edge("B", "A", "u", "inp", transform="extract_first")
    gm.add_edge("A", "B", "u", "inp", transform="extract_first")
    gm.add_coupling_group(["A", "B"], max_iterations=2, convergence_norm="interface", rtol=RTOL,
                          diagnostics=True)
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    g32 = float(F32(gain))
    a_star = 1.0 / (1.0 - g32)
    true = _rms_over_own_magnitude([
        (np.asarray(gm.get_node_state("A")["u"], np.float64), np.asarray([a_star])),
        (np.asarray(gm.get_node_state("B")["u"], np.float64), np.asarray([g32 * a_star])),
    ])
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, ("growing", gain))


def test_the_arnoldi_through_an_identity_reading_is_the_plain_arnoldi():
    """``_arnoldi_through`` with the reading the state itself is ``arnoldi_spectral_radius``.

    Radius, Arnoldi residual and resolvent norm agree to rounding, where
    the start vector's Krylov space breaks down too: the start lies in one
    invariant block (radius 0.5), the residual in the other (0.9), and the
    space continues from the residual's direction, carried with its
    preimage.  Without the residual only the start's block is seen.
    """
    from maddening.core.coupling.acceleration import _arnoldi_through, arnoldi_spectral_radius

    A = np.zeros((5, 5), F32)
    A[:2, :2] = [[0.5, 0.2], [0.0, 0.3]]
    A[2:, 2:] = [[0.9, 0.3, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, -0.4]]
    A = jnp.asarray(A)
    start = jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0], jnp.float32)
    residual = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0], jnp.float32)

    def matvec(v):
        return A @ v

    plain = arnoldi_spectral_radius(matvec, start, v_extra=residual)
    through = _arnoldi_through(matvec, lambda v: v, start, extra=(residual, residual))
    for name, a, b in zip(("rho", "residual", "amplification"), plain, through):
        assert float(b) == pytest.approx(float(a), rel=1e-5, abs=1e-6), name
    assert float(through[0]) == pytest.approx(0.9, rel=1e-5)
    alone = _arnoldi_through(matvec, lambda v: v, start)
    assert float(alone[0]) == pytest.approx(0.5, rel=1e-5)


#: ``(gA, cA, gB, cB, a0, b0)``: B's value stays inside the dead band
#: (``|b| <= atol``) and drives A through a large gain; the loop gain is 0.5
#: and 0.8.  In the second, A's first pass returns its start exactly.
_DEAD_BANDED = {
    "moving": (1e4, 0.5, 5e-5, 5e-5, 0.5, 1e-5),
    "read-edge-at-rest": (2e3, 1.0, 4e-4, 1e-4, 3.0, 1e-3),
}
_DEAD_BAND = 1e-2


@pytest.mark.parametrize("case", sorted(_DEAD_BANDED))
def test_a_dead_banded_transformed_edge_is_folded_into_the_bound(case):
    """An edge the dead band excludes still drives the edge the norm reads.

    ``|b| <= atol`` takes B's value out of the interface norm, but A is
    ``gA * b + cA`` with a large gain, so B's unread change is most of A's
    distance to its fixed point.  The analysis folds the unread share of
    the residual into the factor, on the reading as on the state's fields:
    against the read residual, or, where that is exactly zero (A has not
    moved between the two passes the norm compared), against the reading's
    float floor.  Without the fold the bound reads 0.23x the distance of
    the edge the norm reads; without the floor it is ``inf``.
    """
    gA, cA, gB, cB, a0, b0 = _DEAD_BANDED[case]
    gm = GraphManager()
    gm.add_node(_LinVec("A", [gA], [cA], [a0]))
    gm.add_node(_LinVec("B", [gB], [cB], [b0]))
    gm.add_edge("A", "B", "u", "inp", transform="extract_first")
    gm.add_edge("B", "A", "u", "inp", transform="extract_first")
    gm.add_coupling_group(["A", "B"], max_iterations=2, convergence_norm="interface", rtol=RTOL,
                          atol=_DEAD_BAND, diagnostics=True, iteration_mode="jacobi")
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    gA, cA, gB, cB = (float(F32(v)) for v in (gA, cA, gB, cB))
    a_star = (gA * cB + cA) / (1.0 - gA * gB)
    a = np.asarray(gm.get_node_state("A")["u"], np.float64)
    b = float(np.asarray(gm.get_node_state("B")["u"], np.float64)[0])
    assert abs(b) <= _DEAD_BAND < abs(a[0]), (a, b)          # the fixture premise
    true = _rms_over_own_magnitude([(a, np.asarray([a_star]))])
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, ("dead band", case))


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_the_bound_holds_in_the_transformed_reading_on_the_audit_fixtures
@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ["none", "fixed", "aitken", "iqn-ils"])
@pytest.mark.parametrize("fixture", sorted(AUDIT_FIXTURES))
def test_the_bound_holds_in_the_transformed_reading_at_every_cap(fixture, acceleration):
    """The audit's whole sweep: every acceleration, caps 2 to 6."""
    for cap in (2, 3, 4, 5, 6):
        gm = _audit_graph(fixture, acceleration, cap)
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        _assert_bound_holds(d, _audit_distance(fixture, gm), (fixture, acceleration, cap))


class _LinVecIn(SimulationNode):
    """:class:`_LinVec` in a chosen dtype."""

    def __init__(self, name, g, c, u0, dtype):
        super().__init__(name, 1.0)
        self._p = (np.asarray(g, np.float64), np.asarray(c, np.float64),
                   np.asarray(u0, np.float64), dtype)

    def initial_state(self):
        return {"u": jnp.asarray(self._p[2], self._p[3])}

    def update(self, state, boundary_inputs, dt):
        g, c, _u0, dtype = self._p
        inp = jnp.asarray(boundary_inputs.get("inp", 0.0), dtype)
        return {"u": (jnp.asarray(g, dtype) * inp + jnp.asarray(c, dtype)).astype(dtype)}

    def update_evaluations(self):
        return 1.0


#: ``domain -> (dtype, x64, rtol)``: the reading's analysis outside float32.
#: A 16-bit group asks for what its fields resolve.
READING_DOMAINS = {
    "f64": (jnp.float64, True, RTOL),
    "float16": (jnp.float16, False, 1e-2),
    "bfloat16": (jnp.bfloat16, False, 1e-1),
}


def _assert_the_bound_holds_in(domain, mode):
    """The audit's ``extract_last`` pair in *domain*'s dtype under *mode*, two passes.

    The oracle is the float64 fixed point of the affine map with its
    constants as the dtype rounds them, the distance the interface norm's.
    """
    dtype, x64, rtol = READING_DOMAINS[domain]
    tj, tn, gA, cA, gB, cB, uA0, uB0 = AUDIT_FIXTURES["extract-last"]
    with cd.x64(x64):
        gm = GraphManager()
        gm.add_node(_LinVecIn("A", gA, cA, uA0, dtype))
        gm.add_node(_LinVecIn("B", gB, cB, uB0, dtype))
        gm.add_edge("A", "B", "u", "inp", transform=tj)
        gm.add_edge("B", "A", "u", "inp", transform="extract_first")
        gm.add_coupling_group(["A", "B"], max_iterations=2, convergence_norm="interface",
                              rtol=rtol, diagnostics=True, iteration_mode=mode)
        gm.compile()
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        uA = np.asarray(gm.get_node_state("A")["u"], np.float64)
        uB = np.asarray(gm.get_node_state("B")["u"], np.float64)
        gA, cA, gB, cB = (np.asarray(jnp.asarray(np.asarray(v, np.float64), dtype), np.float64)
                          for v in (gA, cA, gB, cB))
    t0 = tn(cA)
    s_star = (gB[0] * t0 + cB[0]) / (1.0 - gB[0] * (tn(gA + cA) - t0))
    uA_s = gA * s_star + cA
    uB_s = gB * tn(uA_s) + cB
    true = _rms_over_own_magnitude([
        (np.atleast_1d(tn(uA)), np.atleast_1d(tn(uA_s))),
        (np.atleast_1d(uB[0]), np.atleast_1d(uB_s[0])),
    ], rtol)
    assert d["spectral_usable"] and true > 0, (dict(d), true)     # the fixture premise
    _assert_bound_holds(d, true, (domain, mode))


@pytest.mark.parametrize("domain", ["f64", "float16"])
def test_the_bound_holds_in_the_transformed_reading_in_float64_and_in_float16(domain):
    """The reading's analysis outside float32: the bound read 0.078x (float64) and 0.85x (float16)."""
    _assert_the_bound_holds_in(domain, "jacobi")


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_the_bound_holds_in_the_transformed_reading_in_float64_and_in_float16
@pytest.mark.slow
@pytest.mark.parametrize("domain, mode", [("bfloat16", "jacobi"), ("f64", "gauss-seidel"),
                                          ("float16", "gauss-seidel"),
                                          ("bfloat16", "gauss-seidel")])
def test_the_bound_holds_in_the_transformed_reading_in_every_float_dtype_and_sweep(domain, mode):
    """bfloat16, and Gauss-Seidel in each of the three dtypes."""
    _assert_the_bound_holds_in(domain, mode)


def test_the_mixed_norm_with_the_same_transforms_keeps_the_raw_fields():
    """Under "mixed" the norm reads the raw fields whatever the edges transform.

    The transformed reading is the interface norm's alone: the same graph
    under "mixed" is bounded in the raw fields' norm, as it always was.
    """
    fixture = "offset"
    gm = _audit_graph(fixture, "none", 2, norm="mixed")
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    uA_s, uB_s = _audit_exact(fixture)
    true = _rms_over_own_magnitude([
        (np.asarray(gm.get_node_state("A")["u"], np.float64), uA_s),
        (np.asarray(gm.get_node_state("B")["u"], np.float64), uB_s),
    ])
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, "mixed")
    # The report's note states the bound with its conditions (CPL-088).
    notes = " ".join(gm.coupling_report().notes)
    assert "spectral_usable is True" in notes and "transformed" in notes, notes


# ---------------------------------------------------------------------------
# An identity transform takes the reading's path: it must agree with none
# ---------------------------------------------------------------------------


class _Mat(SimulationNode):
    """``u <- G @ inp + c``; ``G`` and ``c`` are parameters, so one compile serves every draw."""

    def __init__(self, name, n, k, u0=None):
        super().__init__(name, 1.0, G=jnp.zeros((n, k), jnp.float32),
                         c=jnp.zeros((n,), jnp.float32))
        self._n, self._k = int(n), int(k)
        self._u0 = np.zeros(n, F32) if u0 is None else np.asarray(u0, F32)

    def initial_state(self):
        return {"u": jnp.asarray(self._u0)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self._k,), dtype=jnp.float32,
                                         default=jnp.zeros(self._k, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"u": p["G"] @ boundary_inputs["inp"] + p["c"]}

    def update_evaluations(self):
        return 1.0


#: ``kind -> (transform, its linear part S and offset o as float64 functions of n, output size)``.
#: Offsets, scales and selections, alone and combined -- what an edge between
#: two physics codes carries (a unit conversion, a sign, a boundary value).
TRANSFORMS = {
    "none": (None, lambda n: (np.eye(n), np.zeros(n)), lambda n: n),
    "offset": (lambda v: v + 273.15, lambda n: (np.eye(n), np.full(n, 273.15)), lambda n: n),
    "scale": (lambda v: -2.5 * v, lambda n: (-2.5 * np.eye(n), np.zeros(n)), lambda n: n),
    "last": (lambda v: v[-1:], lambda n: (np.eye(n)[-1:], np.zeros(1)), lambda n: 1),
    "unit-of-first": (lambda v: 1000.0 * v[:1] + 3.0,
                      lambda n: (1000.0 * np.eye(n)[:1], np.full(1, 3.0)), lambda n: 1),
    "identity": (resolve_transform("identity"), lambda n: (np.eye(n), np.zeros(n)), lambda n: n),
}
N = 2


def _mat_graph(ab, ba, *, mode="jacobi", acceleration="none", cap=3):
    gm = GraphManager()
    gm.add_node(_Mat("A", N, TRANSFORMS[ba][2](N)))
    gm.add_node(_Mat("B", N, TRANSFORMS[ab][2](N)))
    gm.add_edge("A", "B", "u", "inp", transform=TRANSFORMS[ab][0])
    gm.add_edge("B", "A", "u", "inp", transform=TRANSFORMS[ba][0])
    gm.add_coupling_group(["A", "B"], max_iterations=cap, convergence_norm="interface",
                          rtol=RTOL, diagnostics=True, iteration_mode=mode,
                          acceleration=acceleration)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # a lambda transform is not serialisable: noted
        gm.compile()
    return gm


def _run(gm, values):
    """Reset, write the drawn start, step once with the drawn parameters."""
    gm.reset_state()
    for nm in ("A", "B"):
        gm.set_node_state(nm, {"u": jnp.asarray(values[nm]["u0"], jnp.float32)})
    p = gm.params
    nodes = {nm: {**p["nodes"][nm], "G": jnp.asarray(values[nm]["G"], jnp.float32),
                  "c": jnp.asarray(values[nm]["c"], jnp.float32)} for nm in ("A", "B")}
    gm.step(params={**p, "nodes": {**p["nodes"], **nodes}})
    return gm.coupling_diagnostics()[KEY]


def _exact_and_distance(gm, ab, ba, values):
    """The float64 fixed point of the float32-rounded affine map, and the interface distance."""
    S_ab, o_ab = TRANSFORMS[ab][1](N)
    S_ba, o_ba = TRANSFORMS[ba][1](N)
    GA = np.asarray(values["A"]["G"], F32).astype(np.float64)
    GB = np.asarray(values["B"]["G"], F32).astype(np.float64)
    cA = np.asarray(values["A"]["c"], F32).astype(np.float64)
    cB = np.asarray(values["B"]["c"], F32).astype(np.float64)
    M = np.block([[np.zeros((N, N)), GA @ S_ba], [GB @ S_ab, np.zeros((N, N))]])
    k = np.concatenate([GA @ o_ba + cA, GB @ o_ab + cB])
    x_star = np.linalg.solve(np.eye(2 * N) - M, k)
    uA = np.asarray(gm.get_node_state("A")["u"], np.float64)
    uB = np.asarray(gm.get_node_state("B")["u"], np.float64)
    true = _rms_over_own_magnitude([
        (S_ab @ uA + o_ab, S_ab @ x_star[:N] + o_ab),
        (S_ba @ uB + o_ba, S_ba @ x_star[N:] + o_ba),
    ])
    return x_star, true


def _values(rng, ab, ba, rho):
    """Gains contracting at about *rho* through both transforms' linear parts."""
    S_ab, _ = TRANSFORMS[ab][1](N)
    S_ba, _ = TRANSFORMS[ba][1](N)
    GA = rng.normal(size=(N, S_ba.shape[0]))
    GB = rng.normal(size=(N, S_ab.shape[0]))
    M = np.block([[np.zeros((N, N)), GA @ S_ba], [GB @ S_ab, np.zeros((N, N))]])
    r = max(abs(np.linalg.eigvals(M)))
    s = rho / r if r > 0 else 1.0       # M scales with s: Jacobi radius rho
    return {"A": {"G": GA * s, "c": rng.normal(size=N) * 3.0, "u0": rng.normal(size=N)},
            "B": {"G": GB * s, "c": rng.normal(size=N) * 3.0, "u0": rng.normal(size=N)}}


@pytest.mark.parametrize("mode", ["jacobi", "gauss-seidel"])
def test_an_identity_transform_reads_the_bound_an_untransformed_edge_does(mode):
    """The reading's analysis and the state's agree where the reading is the state.

    A registered ``"identity"`` on both edges makes the interface norm read
    exactly the raw fields, so the analysis taken through it must give the
    raw fields' bound, radius and flag, to float32 rounding.
    """
    values = _values(np.random.default_rng(4), "none", "none", 0.8)
    plain = _run(_mat_graph("none", "none", mode=mode), values)
    through = _run(_mat_graph("identity", "identity", mode=mode), values)
    assert plain["spectral_usable"] == through["spectral_usable"] is True
    assert through["rho_spectral"] == pytest.approx(plain["rho_spectral"], rel=1e-4)
    assert through["spectral_error_bound"] == pytest.approx(plain["spectral_error_bound"],
                                                            rel=1e-4)
    # The gradient bound keeps the state's own analysis on either path.
    assert through["gradient_bound_usable"] == plain["gradient_bound_usable"] is True
    assert through["gradient_relative_error_bound"] == pytest.approx(
        plain["gradient_relative_error_bound"], rel=1e-5)


_STRUCTURES = [(ab, ba) for ab in ("offset", "scale", "last", "unit-of-first")
               for ba in ("none", "offset", "last")]
_GRAPHS: dict = {}


def _cached_graph(ab, ba, mode, acceleration, cap):
    key = (ab, ba, mode, acceleration, cap)
    if key not in _GRAPHS:
        _GRAPHS[key] = _mat_graph(ab, ba, mode=mode, acceleration=acceleration, cap=cap)
    return _GRAPHS[key]


def test_a_usable_bound_holds_on_a_transformed_pair():
    """A selection, a scale and an offset on one edge, a selection on the other.

    One compiled structure, three drawn pairs; the property below draws
    every structure.
    """
    rng = np.random.default_rng(11)
    ab, ba = "unit-of-first", "last"
    gm = _cached_graph(ab, ba, "jacobi", "none", 3)
    for _ in range(3):
        values = _values(rng, ab, ba, 0.9)
        d = _run(gm, values)
        _x, true = _exact_and_distance(gm, ab, ba, values)
        assert d["spectral_usable"], (ab, ba, dict(d))
        _assert_bound_holds(d, true, (ab, ba))


def test_on_the_reading_path_the_gradient_bound_stands_only_on_the_states_settled_spectrum(
        monkeypatch):
    """The gradient bound rests on the state's own spectral triple, not the reading's.

    ``gradient_bound_usable`` is read off the report's ``spectral_usable``,
    which is the reading's on this path; the gradient bound takes its
    distance and resolvent from the state's triple, in the weights its own
    norms use, so it is withdrawn (NaN) where that one did not settle.  The
    two operators share their non-zero spectrum and their rank (the state's
    Jacobian factors through the reading), so the triples part only where
    the reading has more scalars than the Arnoldi takes steps, or in
    rounding; here the state's Arnoldi residual is inflated to four times
    the settled fraction, which leaves its bound finite and contracting, to
    reach the gate.
    """
    from maddening.core import graph_manager as gm_mod
    from maddening.core.coupling.acceleration import SPECTRAL_SETTLED_FRACTION

    ab, ba = "unit-of-first", "last"
    values = _values(np.random.default_rng(11), ab, ba, 0.9)
    plain = _run(_mat_graph(ab, ba), values)
    # The control: the same pair, unpatched, carries a usable gradient bound.
    assert plain["spectral_usable"] and plain["gradient_bound_usable"], dict(plain)

    real = gm_mod._spectral_rate_at

    def state_triple_unsettled(*args, **kwargs):
        out = real(*args, **kwargs)
        if "field_reference" in kwargs:     # the untransformed path's call: untouched
            return out
        rho, resid, amp = out
        return rho, jnp.maximum(resid, 4.0 * SPECTRAL_SETTLED_FRACTION * (1.0 - rho)), amp

    monkeypatch.setattr(gm_mod, "_spectral_rate_at", state_triple_unsettled)
    d = _run(_mat_graph(ab, ba), values)
    assert d["spectral_usable"], dict(d)                    # the reading's triple settled
    assert d["spectral_error_bound"] == plain["spectral_error_bound"], dict(d)
    assert math.isnan(d["gradient_relative_error_bound"]), dict(d)
    assert not d["gradient_bound_usable"], dict(d)


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_a_usable_bound_holds_on_a_transformed_pair
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_bound_holds_on_random_transformed_internal_edges(data):
    """Random offsets, scales and selections on the internal edges; drawn gains and starts.

    The structure (which transforms, schedule, accelerator, cap) is drawn
    from a fixed set and each is compiled once; the gains, offsets and
    starting state reach the compiled step as values.
    """
    ab, ba = data.draw(st.sampled_from(_STRUCTURES), label="transforms")
    mode = data.draw(st.sampled_from(["jacobi", "gauss-seidel"]), label="mode")
    acceleration = data.draw(st.sampled_from(["none", "aitken", "iqn-ils"]), label="acceleration")
    cap = data.draw(st.sampled_from([2, 3, 5]), label="cap")
    rho = data.draw(st.sampled_from([0.3, 0.8, 0.95]), label="rho")
    seed = data.draw(st.integers(0, 2**31 - 1), label="seed")
    values = _values(np.random.default_rng(seed), ab, ba, rho)
    gm = _cached_graph(ab, ba, mode, acceleration, cap)
    d = _run(gm, values)
    _x, true = _exact_and_distance(gm, ab, ba, values)
    note(f"report {dict(d)}; true {true}")
    _assert_bound_holds(d, true, (ab, ba, mode, acceleration, cap, rho))
