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
(``coupling._bounds._interface_spectral_rate_at``).

The oracle is the float64 fixed point of the affine map, and the distance is
the interface norm's: every internal edge's transformed value, its change
over ``rtol`` times its own ``max|v|`` at the returned state, the RMS over
every entry.

**An interface mapping is part of the reading.**  The norm reads what each
internal edge *delivers*: the source value through the edge's mapping
(``EdgeSpec.mapping``, with the weights the step ran with) and then its
transform.  While the norm and this analysis left the mapping out, the
audit's ``extract_last`` pair with the selection written as a
``matrix_mapping`` read 0.064x, 0.318x and 0.074x the true distance in the
delivered values at caps 2 to 4, ``spectral_usable=True``.  The fixtures
and the drawn structures below therefore come with a mapping, alone and
before a transform, and are held to the same oracle.
"""

from __future__ import annotations

import functools
import math
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from maddening.core.coupling.mapping import matrix_mapping
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
    # "extract-last" with the selection written as an interface mapping.
    "selection-as-mapping": (None, lambda v: v[-1], [0.0, 50.0], [10.0, -49.0],
                             [0.01], [0.99], [10.0, 0.5], [0.99]),
    # A mapping, then a transform written for what the mapping delivers:
    # A.u = [10, 0.7 s + 0.3], the mapping selects the second entry and the
    # transform adds the offset.
    "mapping-then-offset": (lambda v: v[0] + 1000.0, lambda v: v[-1] + 1000.0, [0.0, 0.7],
                            [10.0, 0.3], [0.7], [0.3 - 700.0], [10.0, 0.9], [0.93]),
}
#: The interface mapping on ``A -> B`` of the fixtures that have one, applied
#: before the transform.
AUDIT_MAPPINGS = {"selection-as-mapping": [[0.0, 1.0]], "mapping-then-offset": [[0.0, 1.0]]}
#: The fixtures the norm read on the source field before: a mapped edge.
MAPPED_AUDIT_FIXTURES = sorted(AUDIT_MAPPINGS)


def _audit_mapping(fixture, dtype=F32):
    """The fixture's mapping on ``A -> B`` in *dtype*, or ``None``."""
    H = AUDIT_MAPPINGS.get(fixture)
    return None if H is None else matrix_mapping(jnp.asarray(np.asarray(H, np.float64), dtype))


def _audit_graph(fixture, acceleration, cap, norm="interface", mode="jacobi"):
    """The fixture's pair, handed back at its initial state.  The last
    sixteen are kept compiled: the per-push tests share theirs, and the slow
    sweep's eighty do not pile up."""
    gm = _build_audit_graph(fixture, acceleration, cap, norm, mode)
    gm.reset_state()
    return gm


@functools.lru_cache(maxsize=16)
def _build_audit_graph(fixture, acceleration, cap, norm, mode):
    tj, _tn, gA, cA, gB, cB, uA0, uB0 = AUDIT_FIXTURES[fixture]
    gm = GraphManager()
    gm.add_node(_LinVec("A", gA, cA, uA0))
    gm.add_node(_LinVec("B", gB, cB, uB0))
    gm.add_edge("A", "B", "u", "inp", transform=tj, mapping=_audit_mapping(fixture))
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


#: Each fixture under plain iteration and under Aitken at caps 2 and 3; a
#: mapped fixture's Aitken cells are slow (one more compiled pair each).
_AUDIT_CELLS = [
    pytest.param(fixture, acceleration, cap, marks=(
        [pytest.mark.slow] if fixture in AUDIT_MAPPINGS and acceleration == "aitken" else []))
    for fixture in sorted(AUDIT_FIXTURES) for acceleration in ("none", "aitken")
    for cap in (2, 3)]


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_the_bound_holds_in_the_transformed_reading_on_the_audit_fixtures
# (every fixture under plain iteration, and the transform fixtures under Aitken)
@pytest.mark.parametrize("fixture, acceleration, cap", _AUDIT_CELLS)
def test_the_bound_holds_in_the_transformed_reading_on_the_audit_fixtures(fixture, acceleration,
                                                                         cap):
    """``spectral_error_bound >= ||x - x*||`` in the interface norm, wherever usable.

    Before, ``extract_last`` read 0.078x and the offset 0.0014x at ``cap=2``;
    with the selection written as a mapping, 0.064x (the norm and the
    analysis read the source field, not what the mapping delivers).
    """
    gm = _audit_graph(fixture, acceleration, cap)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    assert d["spectral_usable"], dict(d)        # the fixture premise: a resolved spectrum
    _assert_bound_holds(d, _audit_distance(fixture, gm), (fixture, acceleration, cap))


@pytest.mark.parametrize("fixture", ["extract-last", "selection-as-mapping"])
def test_a_stalled_transformed_pair_is_bounded_through_the_readings_float_floor(fixture):
    """Where the residual reads zero the bound is the reading's float floor, amplified.

    Gauss-Seidel with Aitken lands the ``extract_last`` pair on its float32
    fixed point by the third pass: the residual is exactly 0.0, the state
    is still a rounding away from the fixed point of the map its float32
    constants define (0.66 in the norm's units), and what bounds that is
    the floor taken on the reading (each transformed value at its own
    eps), which the residual is added to before it is amplified.
    """
    gm = _audit_graph(fixture, "aitken", 3, mode="gauss-seidel")
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    true = _audit_distance(fixture, gm, held_in=F32)
    assert d["residual"] == 0.0 and d["precision_limited"] and true > 0, (dict(d), true)
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, ("stalled", fixture))


def _one_entry_edge(kind) -> dict:
    """``add_edge`` arguments for an edge that delivers its one-entry source
    unchanged, through a transform or through an interface mapping."""
    if kind == "transform":
        return {"transform": "extract_first"}
    return {"mapping": matrix_mapping(np.ones((1, 1), F32))}


@pytest.mark.parametrize("kind", ["transform", "mapping"])
@pytest.mark.parametrize("gain", [0.9, 0.999])
def test_the_bound_holds_in_the_returned_readings_weights_on_a_growing_pair(gain, kind):
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
    gm.add_edge("B", "A", "u", "inp", **_one_entry_edge(kind))
    gm.add_edge("A", "B", "u", "inp", **_one_entry_edge(kind))
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
    _assert_bound_holds(d, true, ("growing", gain, kind))


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


@pytest.mark.parametrize("kind", ["transform", "mapping"])
@pytest.mark.parametrize("case", sorted(_DEAD_BANDED))
def test_a_dead_banded_transformed_edge_is_folded_into_the_bound(case, kind):
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
    gm.add_edge("A", "B", "u", "inp", **_one_entry_edge(kind))
    gm.add_edge("B", "A", "u", "inp", **_one_entry_edge(kind))
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
    _assert_bound_holds(d, true, ("dead band", case, kind))


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


def _assert_the_bound_holds_in(domain, mode, fixture="extract-last", settled=True):
    """The audit's ``extract_last`` pair in *domain*'s dtype under *mode*, two passes.

    The oracle is the float64 fixed point of the affine map with its
    constants as the dtype rounds them, the distance the interface norm's.
    *fixture* is the spelling of the selection: the transform, or an
    interface mapping in the members' dtype.
    """
    dtype, x64, rtol = READING_DOMAINS[domain]
    tj, tn, gA, cA, gB, cB, uA0, uB0 = AUDIT_FIXTURES[fixture]
    with cd.x64(x64):
        gm = GraphManager()
        gm.add_node(_LinVecIn("A", gA, cA, uA0, dtype))
        gm.add_node(_LinVecIn("B", gB, cB, uB0, dtype))
        gm.add_edge("A", "B", "u", "inp", transform=tj, mapping=_audit_mapping(fixture, dtype))
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
    assert true > 0, (dict(d), true)                               # the fixture premise
    if not settled:
        # The dtype's rounding does not determine the radius to the flag's
        # margin (see the caller), and the number is still no smaller than
        # the distance.  Whether the flag is withdrawn is the rounding
        # certificate's verdict on the rounding it MEASURES, and a 16-bit
        # dtype's rounding is the processor's: half-precision arithmetic is
        # native on some and goes through float32 on others.  float16 under
        # Gauss-Seidel reads refused on most machines and usable on some
        # (two of fourteen CI runners on 2026-10-08, one on each jaxlib;
        # refused on a workstation at 1, 2, 4 and 8 threads).  So the
        # verdict is not pinned; what it stands for is: where the flag is
        # set, the radius read is within the flag's margin of the pair's
        # (5% of 1 - rho, CPL-087; the pair's loop gain in closed form,
        # its square root under Jacobi) and the bound is over the distance.
        assert d["spectral_error_bound"] >= true, (dict(d), true)
        if d["spectral_usable"]:
            gain = abs(float(gB[0] * (tn(gA + cA) - t0)))
            rho = gain if mode == "gauss-seidel" else math.sqrt(gain)
            read = float(d["rho_spectral"])
            assert abs(read - rho) <= 0.05 * (1.0 - read), (
                f"spectral_usable is set on a radius outside its margin: read {read!r}, "
                f"the pair's {rho!r} ({domain}, {mode}, {fixture}); bound "
                f"{float(d['spectral_error_bound'])!r}, distance {true!r}")
            _assert_bound_holds(d, true, (domain, mode, fixture))
        return
    assert d["spectral_usable"], (dict(d), true)
    _assert_bound_holds(d, true, (domain, mode, fixture))


@pytest.mark.parametrize("fixture", ["extract-last", "selection-as-mapping"])
@pytest.mark.parametrize("domain", ["f64", "float16"])
def test_the_bound_holds_in_the_transformed_reading_in_float64_and_in_float16(domain, fixture):
    """The reading's analysis outside float32: the bound read 0.078x (float64) and 0.85x (float16).

    With the selection written as a mapping in the members' dtype the reading
    is the same numbers, and so is what the bound must cover.
    """
    _assert_the_bound_holds_in(domain, "jacobi", fixture)


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_the_bound_holds_in_the_transformed_reading_in_float64_and_in_float16
@pytest.mark.slow
@pytest.mark.parametrize("fixture", ["extract-last", "selection-as-mapping"])
@pytest.mark.parametrize("domain, mode", [("bfloat16", "jacobi"), ("f64", "gauss-seidel"),
                                          ("float16", "gauss-seidel"),
                                          ("bfloat16", "gauss-seidel")])
def test_the_bound_holds_in_the_transformed_reading_in_every_float_dtype_and_sweep(domain, mode,
                                                                                  fixture):
    """bfloat16, and Gauss-Seidel in each of the three dtypes, both spellings of the selection.

    The pair's weighted Jacobian is far from normal (gains 50 and 0.01),
    and only float64 determines its radius to the flag's margin under
    every sweep:

    * **bfloat16** (``eps = 2**-7``): the second Krylov direction of the
      reading is 0.6% of the product, under one bfloat16 rounding of it,
      and the breakdown test must take it for one.  Discarded, the
      reading's radius reads 1.06 for 0.707 (Jacobi) -- a direction that
      small returns with a gain of 90 -- so the bound is ``inf`` and
      ``spectral_usable`` is False.  Kept (the rule before the threshold
      was scaled to the products' rounding) the number was right by
      trusting a direction the dtype cannot tell from noise.
    * **float16 under Gauss-Seidel**: the fresh product disagrees with
      the Arnoldi relation by 1.2e-4 in the row a gain of 66 multiplies,
      which can move a radius of 0.5 by 0.09 against a margin of 0.025:
      the rounding certificate refuses it.  The radius read is right
      (0.5008) and the bound covers the distance 72 times over.  That
      refusal is the certificate's verdict on one machine's half-precision
      rounding and is not asserted (``_assert_the_bound_holds_in``): on a
      machine whose rounding is finer the flag is set, on the same radius.
    """
    settled = domain == "f64"
    _assert_the_bound_holds_in(domain, mode, fixture, settled=settled)


@pytest.mark.parametrize("kind", ["transform", "mapping"])
def test_a_stalled_float32_pair_behind_edges_that_widen_is_bounded_at_float32_resolution(kind):
    """A delivered value is no finer than the field it was computed from.

    Float32 members under ``jax_enable_x64`` whose internal edges deliver
    float64 -- a transform that returns float64, or a float64 mapping
    matrix, which is what a NumPy matrix becomes there.  The pair stalls
    at ``residual=0.0`` a float32 rounding from its fixed point (0.66 in
    the norm's units), so the bound is the reading's float floor,
    amplified.  Taken at the delivered dtype's eps that floor was
    ``2**-29`` of float32's, and behind the widening transform the bound
    read 6.7e-7 of the true distance with ``spectral_usable=True``.
    """
    _tj, tn, gA, cA, gB, cB, uA0, uB0 = AUDIT_FIXTURES["extract-last"]
    with cd.x64(True):
        gm = GraphManager()
        gm.add_node(_LinVecIn("A", gA, cA, uA0, jnp.float32))
        gm.add_node(_LinVecIn("B", gB, cB, uB0, jnp.float32))
        if kind == "transform":
            gm.add_edge("A", "B", "u", "inp", transform=lambda v: v[-1:].astype(jnp.float64))
            gm.add_edge("B", "A", "u", "inp", transform=lambda v: v[:1].astype(jnp.float64))
        else:
            gm.add_edge("A", "B", "u", "inp",
                        mapping=matrix_mapping(np.array([[0.0, 1.0]], np.float64)))
            gm.add_edge("B", "A", "u", "inp", mapping=matrix_mapping(np.ones((1, 1), np.float64)))
        gm.add_coupling_group(["A", "B"], max_iterations=3, convergence_norm="interface",
                              rtol=RTOL, diagnostics=True, iteration_mode="gauss-seidel",
                              acceleration="aitken")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # a lambda transform is not serialisable: noted
            gm.compile()
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        uA = np.asarray(gm.get_node_state("A")["u"])
        uB = np.asarray(gm.get_node_state("B")["u"])
    assert uA.dtype == uB.dtype == np.float32           # the premise: float32 members
    gA, cA, gB, cB = (np.asarray(v, F32).astype(np.float64) for v in (gA, cA, gB, cB))
    s_star = (gB[0] * tn(cA) + cB[0]) / (1.0 - gB[0] * (tn(gA + cA) - tn(cA)))
    uA_s = gA * s_star + cA
    true = _rms_over_own_magnitude([
        (np.atleast_1d(tn(uA.astype(np.float64))), np.atleast_1d(tn(uA_s))),
        (uB.astype(np.float64)[:1], (gB * tn(uA_s) + cB)[:1]),
    ])
    assert d["residual"] == 0.0 and d["precision_limited"] and true > 0, (dict(d), true)
    assert d["spectral_usable"], dict(d)
    _assert_bound_holds(d, true, ("widened", kind))


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
    assert "spectral_usable is True" in notes and "each edge delivers" in notes, notes


# ---------------------------------------------------------------------------
# An identity transform takes the reading's path: it must agree with none
# ---------------------------------------------------------------------------


class _Mat(SimulationNode):
    """``u <- G @ inp + c``; ``G`` and ``c`` are parameters, so one compile serves every draw."""

    def __init__(self, name, n, k, u0=None, declare=True):
        super().__init__(name, 1.0, G=jnp.zeros((n, k), jnp.float32),
                         c=jnp.zeros((n,), jnp.float32))
        self._n, self._k = int(n), int(k)
        self._u0 = np.zeros(n, F32) if u0 is None else np.asarray(u0, F32)
        self._declare = declare

    def initial_state(self):
        return {"u": jnp.asarray(self._u0)}

    def boundary_input_spec(self):
        # ``add_edge`` holds a mapping's ``n_target`` to the declared shape
        # of the input, whatever transform follows the mapping; a node fed
        # through a mapping *and* a transform that changes the size leaves
        # its input undeclared.
        if not self._declare:
            return {}
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
#: Edge kinds that carry an interface mapping before their transform:
#: ``kind -> (rows of the mapping matrix, the transform applied to what it
#: delivers)``.  The matrix is ``rows x N``; its entries are drawn with the
#: gains and reach the step through ``params["mappings"]``, as a fitted or
#: per-step weight does, while the mapping *object* holds a decoy
#: (:data:`_DECOY`): a reading taken with the object's own weights is wrong.
MAPPED = {
    "mapped": (2, "none"),
    "mapped-wide": (1, "none"),
    "mapped-tall": (3, "none"),
    "mapped-then-offset": (2, "offset"),
    "mapped-then-last": (3, "last"),
    "mapped-then-unit": (2, "unit-of-first"),
}
_DECOY = 0.37
_EDGE_KEYS = {"ab": "A.u->B.inp", "ba": "B.u->A.inp"}


def _transform_of(kind):
    return TRANSFORMS[MAPPED[kind][1] if kind in MAPPED else kind][0]


def _mapping_of(kind):
    """The edge's mapping object (holding the decoy), or ``None``."""
    if kind not in MAPPED:
        return None
    return matrix_mapping(np.full((MAPPED[kind][0], N), _DECOY, F32))


def _out_size(kind) -> int:
    """How many entries an edge of *kind* delivers from a source of ``N``."""
    if kind in MAPPED:
        rows, after = MAPPED[kind]
        return TRANSFORMS[after][2](rows)
    return TRANSFORMS[kind][2](N)


def _resized_after_mapping(kind) -> bool:
    """Does *kind*'s transform change the size of what its mapping delivers?"""
    return kind in MAPPED and _out_size(kind) != MAPPED[kind][0]


def _reading_of(kind, H=None):
    """``(S, o)`` in float64: an edge of *kind* delivers ``S @ u + o``.

    For a mapped kind ``S`` is the transform's linear part times the
    mapping matrix *H* as float32 holds it.
    """
    if kind in MAPPED:
        rows, after = MAPPED[kind]
        S_t, o = TRANSFORMS[after][1](rows)
        return S_t @ np.asarray(H, F32).astype(np.float64), o
    return TRANSFORMS[kind][1](N)


def _mat_graph(ab, ba, *, mode="jacobi", acceleration="none", cap=3, own_weights=None):
    """The pair with edges of kinds *ab* and *ba*.  *own_weights* builds the
    mapping objects from ``{"ab": H, "ba": H}`` instead of the decoy."""
    gm = GraphManager()
    gm.add_node(_Mat("A", N, _out_size(ba), declare=not _resized_after_mapping(ba)))
    gm.add_node(_Mat("B", N, _out_size(ab), declare=not _resized_after_mapping(ab)))
    maps = {"ab": _mapping_of(ab), "ba": _mapping_of(ba)}
    for side, H in (own_weights or {}).items():
        maps[side] = matrix_mapping(np.asarray(H, F32))
    gm.add_edge("A", "B", "u", "inp", transform=_transform_of(ab), mapping=maps["ab"])
    gm.add_edge("B", "A", "u", "inp", transform=_transform_of(ba), mapping=maps["ba"])
    gm.add_coupling_group(["A", "B"], max_iterations=cap, convergence_norm="interface",
                          rtol=RTOL, diagnostics=True, iteration_mode=mode,
                          acceleration=acceleration)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # a lambda transform is not serialisable: noted
        gm.compile()
    return gm


def _run(gm, values, *, pass_weights=True):
    """Reset, write the drawn start, step once with the drawn parameters.

    The drawn mapping matrices (``values["H"]``) go in through
    ``params["mappings"]`` unless *pass_weights* is off, which leaves the
    mapping objects' own.
    """
    gm.reset_state()
    for nm in ("A", "B"):
        gm.set_node_state(nm, {"u": jnp.asarray(values[nm]["u0"], jnp.float32)})
    p = gm.params
    nodes = {nm: {**p["nodes"][nm], "G": jnp.asarray(values[nm]["G"], jnp.float32),
                  "c": jnp.asarray(values[nm]["c"], jnp.float32)} for nm in ("A", "B")}
    maps = {k: dict(v) for k, v in p.get("mappings", {}).items()}
    if pass_weights:
        for side, H in values.get("H", {}).items():
            maps[_EDGE_KEYS[side]]["H"] = jnp.asarray(H, jnp.float32)
    gm.step(params={**p, "nodes": {**p["nodes"], **nodes}, "mappings": maps})
    return gm.coupling_diagnostics()[KEY]


def _exact_and_distance(gm, ab, ba, values):
    """The float64 fixed point of the float32-rounded affine map, and the interface distance."""
    S_ab, o_ab = _reading_of(ab, values.get("H", {}).get("ab"))
    S_ba, o_ba = _reading_of(ba, values.get("H", {}).get("ba"))
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
    """Gains contracting at about *rho* through both edges' linear parts.

    A mapped edge's matrix is drawn first (an unmapped pair draws what it
    always drew), rounded to float32 as the step holds it.
    """
    H = {side: rng.normal(size=(MAPPED[kind][0], N)).astype(F32)
         for side, kind in (("ab", ab), ("ba", ba)) if kind in MAPPED}
    S_ab, _ = _reading_of(ab, H.get("ab"))
    S_ba, _ = _reading_of(ba, H.get("ba"))
    GA = rng.normal(size=(N, S_ba.shape[0]))
    GB = rng.normal(size=(N, S_ab.shape[0]))
    M = np.block([[np.zeros((N, N)), GA @ S_ba], [GB @ S_ab, np.zeros((N, N))]])
    r = max(abs(np.linalg.eigvals(M)))
    s = rho / r if r > 0 else 1.0       # M scales with s: Jacobi radius rho
    out = {"A": {"G": GA * s, "c": rng.normal(size=N) * 3.0, "u0": rng.normal(size=N)},
           "B": {"G": GB * s, "c": rng.normal(size=N) * 3.0, "u0": rng.normal(size=N)}}
    if H:
        out["H"] = H
    return out


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
    # The coupled block reads its own binding of the name, so that is the one patched.
    from maddening.core.coupling import _coupled_block as block_mod
    from maddening.core.coupling.acceleration import SPECTRAL_SETTLED_FRACTION

    ab, ba = "unit-of-first", "last"
    values = _values(np.random.default_rng(11), ab, ba, 0.9)
    plain = _run(_mat_graph(ab, ba), values)
    # The control: the same pair, unpatched, carries a usable gradient bound.
    assert plain["spectral_usable"] and plain["gradient_bound_usable"], dict(plain)

    real = block_mod._spectral_rate_at

    def state_triple_unsettled(*args, **kwargs):
        out = real(*args, **kwargs)
        if "field_reference" in kwargs:     # the untransformed path's call: untouched
            return out
        rho, resid, amp = out
        return rho, jnp.maximum(resid, 4.0 * SPECTRAL_SETTLED_FRACTION * (1.0 - rho)), amp

    monkeypatch.setattr(block_mod, "_spectral_rate_at", state_triple_unsettled)
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


# ---------------------------------------------------------------------------
# Mapped internal edges: the reading is what the mapping delivers
# ---------------------------------------------------------------------------


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_a_selection_written_as_a_mapping_is_bounded_as_the_same_selection_as_a_transform
# (caps 2 and 3, on the pairs the fixtures' test compiled)
@pytest.mark.parametrize("cap", [2, 3, pytest.param(4, marks=pytest.mark.slow)])
def test_a_selection_written_as_a_mapping_is_bounded_as_the_same_selection_as_a_transform(cap):
    """The audit's ``extract_last`` pair, the selection as a ``matrix_mapping``.

    The two edges deliver the same number, so the group must report the
    same solve: the bound at or above the distance in the delivered values
    (it read 0.064x, 0.318x and 0.074x of it at these caps while the norm
    and the analysis read the source field), and the residual, the radius
    and the bound of the transform's spelling.
    """
    seen = {}
    for fixture in ("extract-last", "selection-as-mapping"):
        gm = _audit_graph(fixture, "none", cap)
        gm.step()
        d = gm.coupling_diagnostics()[KEY]
        true = _audit_distance(fixture, gm)
        assert d["spectral_usable"] and true > 0, (fixture, dict(d), true)
        assert d["spectral_error_bound"] >= true, (
            fixture, cap, d["spectral_error_bound"] / true, dict(d))
        seen[fixture] = (d, true)
    (as_transform, true_t), (as_mapping, true_m) = seen["extract-last"], seen["selection-as-mapping"]
    assert true_m == pytest.approx(true_t, rel=1e-5)
    assert as_mapping["iterations"] == as_transform["iterations"]
    assert as_mapping["converged"] == as_transform["converged"]
    for key in ("residual", "rho_spectral", "spectral_error_bound"):
        assert as_mapping[key] == pytest.approx(as_transform[key], rel=1e-4), (key, cap)


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_an_identity_mapping_reads_the_bound_an_unmapped_edge_does
# (under Jacobi; Gauss-Seidel is the same check on another compiled pair)
@pytest.mark.parametrize("mode", ["jacobi", pytest.param("gauss-seidel",
                                                         marks=pytest.mark.slow)])
def test_an_identity_mapping_reads_the_bound_an_unmapped_edge_does(mode):
    """A mapping that delivers its source unchanged takes the reading's path
    and must give the raw fields' bound, radius and flags, to float32 rounding."""
    values = _values(np.random.default_rng(4), "none", "none", 0.8)
    plain = _run(_cached_graph("none", "none", mode, "none", 3), values)
    eye = {"ab": np.eye(N), "ba": np.eye(N)}
    through = _run(_mat_graph("mapped", "mapped", mode=mode, own_weights=eye), values)
    assert plain["spectral_usable"] == through["spectral_usable"] is True
    assert through["iterations"] == plain["iterations"]
    assert through["residual"] == pytest.approx(plain["residual"], rel=1e-5)
    assert through["rho_spectral"] == pytest.approx(plain["rho_spectral"], rel=1e-4)
    assert through["spectral_error_bound"] == pytest.approx(plain["spectral_error_bound"],
                                                            rel=1e-4)
    assert through["gradient_bound_usable"] == plain["gradient_bound_usable"] is True
    assert through["gradient_relative_error_bound"] == pytest.approx(
        plain["gradient_relative_error_bound"], rel=1e-5)


def test_a_usable_bound_holds_on_a_mapped_pair():
    """A mapping on each edge and no transform anywhere: three entries
    delivered from two on one edge, one from two on the other.

    With no transform in the group, only the mappings put the report on the
    reading's analysis; on the source fields it is another norm's bound.
    One compiled structure, three drawn pairs (the matrices drawn with the
    gains and passed through ``params``); the audit fixtures hold a mapping
    before a transform, and the property below draws every structure.
    """
    rng = np.random.default_rng(23)
    ab, ba = "mapped-tall", "mapped-wide"
    gm = _cached_graph(ab, ba, "jacobi", "none", 3)
    for _ in range(3):
        values = _values(rng, ab, ba, 0.9)
        d = _run(gm, values)
        _x, true = _exact_and_distance(gm, ab, ba, values)
        assert d["spectral_usable"], (ab, ba, dict(d))
        _assert_bound_holds(d, true, (ab, ba))


def test_weights_passed_for_one_step_are_read_as_the_mapping_objects_own_are():
    """The reading is taken with the weights the step ran with.

    A graph whose mapping objects hold a decoy, stepped with the real
    matrices in ``params["mappings"]``, reports what a graph built from the
    real matrices reports: the residual, the radius, the bound, the
    gradient bound and every flag.  Stepped with its own (decoy) weights it
    reports another solve.
    """
    ab, ba = "mapped-tall", "mapped-wide"       # the pair above: compiled once
    values = _values(np.random.default_rng(5), ab, ba, 0.85)
    gm = _cached_graph(ab, ba, "jacobi", "none", 3)
    passed = dict(_run(gm, values))
    _x, true = _exact_and_distance(gm, ab, ba, values)
    assert passed["spectral_usable"], passed
    _assert_bound_holds(passed, true, "passed weights")
    own = dict(_run(_mat_graph(ab, ba, own_weights=values["H"]), values, pass_weights=False))
    for key in ("iterations", "converged", "spectral_usable", "precision_limited",
                "gradient_bound_usable"):
        assert passed[key] == own[key], (key, passed, own)
    for key in ("residual", "rho_spectral", "spectral_error_bound",
                "gradient_relative_error_bound"):
        assert passed[key] == pytest.approx(own[key], rel=1e-4), (key, passed, own)
    decoy = dict(_run(gm, values, pass_weights=False))
    assert abs(decoy["residual"] - passed["residual"]) > 1e-2 * passed["residual"], (
        "fixture premise: the decoy weights are another solve", decoy, passed)


_MAPPED_STRUCTURES = [(ab, ba) for ab in sorted(MAPPED)
                      for ba in ("none", "last", "mapped", "mapped-then-offset")]


# Per push: tests/core/test_coupling_spectral_bound_on_the_transformed_reading.py::test_a_usable_bound_holds_on_a_mapped_pair
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_bound_holds_on_random_mapped_internal_edges(data):
    """Random mapping matrices on the internal edges, alone and before a transform.

    The structure (which edges are mapped, which transform follows, the
    schedule, the accelerator, the cap) is drawn from a fixed set and each
    is compiled once; the matrices, the gains, the offsets and the starting
    state reach the compiled step as values.
    """
    ab, ba = data.draw(st.sampled_from(_MAPPED_STRUCTURES), label="edges")
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
