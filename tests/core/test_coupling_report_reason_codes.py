"""Every usable flag that is ``False`` in ``coupling_diagnostics()`` says why, in a code.

A report's ``spectral_usable`` and ``gradient_bound_usable`` are ``False``
for causes that mean different things: a pass wider than the estimate's
eight Krylov steps (expected, nothing is wrong), an estimate whose check
of itself failed (a worry), a residual at its float floor (switch to
float64, or loosen the tolerance).  A plain float32 pair at the default
norm and tolerance had both flags ``False`` and no reason at all, and a
test cannot branch on a sentence.  So every entry carries
``"reason_codes"`` (the constants of ``maddening.core.coupling.reason_codes``)
and ``"residual_precision_floor"``, and a ``False`` usable flag always has
a ``"not_usable_reason"``.

Here: one constructed case per code, each asserting the code and not the
sentence; the rules every report keeps (``assert_reason_rules``, which
the every-domain battery and the two searches also call on each report
they read); the pinned measurement of what the eight steps resolve (a
loop closing through six scalars beside a member of 3000 entries); the
doors a report goes through; and that the codes are host bookkeeping:
the same state gives the same numbers whatever they say.

The geometry rules' codes are asserted where those groups are built
already (``tests/property/geometry_graphs.py::assert_not_diagnosed`` and
the tests that call it; the long-row and the checkpoint modules), and
here on the functions that decide them, with no graph.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout
from maddening.core.coupling import reason_codes as rc
from maddening.core.coupling.acceleration import SPECTRAL_KRYLOV_STEPS
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.serialization import json_codec
from tests.core.coupling_reason_rules import USABLE, assert_reason_rules

KEY = "body+field"
REPO = Path(__file__).resolve().parents[2]


class _Lin(SimulationNode):
    """``x <- M u + c``; ``c`` is a parameter (so the gradient bound has a
    constant to probe) unless *bare*."""

    def __init__(self, name, n_out, n_in, seed, gain=0.8, *, bare=False, declares=True):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((n_out, n_in))
        c = jnp.asarray(rng.standard_normal(n_out), jnp.float32)
        super().__init__(name, 1.0, **({} if bare else {"c": c}))
        self._c = c
        self._M = jnp.asarray(gain * M / np.linalg.norm(M, 2), jnp.float32)
        self._shape = (n_out, n_in)
        self._declares = declares

    def initial_state(self):
        return {"x": jnp.zeros(self._shape[0], jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._shape[1],), dtype=jnp.float32,
                                       default=jnp.zeros(self._shape[1], jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        c = self._c if "c" not in self.params else (
            self.params if params is None else {**self.params, **params})["c"]
        return {"x": self._M @ boundary_inputs["u"] + c}

    def update_evaluations(self):
        return 1 if self._declares else None


def _pair(k, n, *, first="body", bare=False, declares=True, gain=0.8, **group):
    """A ``body`` of *k* scalars and a ``field`` of *n* entries, each reading
    the other whole, stepped once.  Under Gauss-Seidel the loop closes
    through the member swept second."""
    nodes = {"body": _Lin("body", k, n, 1, gain, bare=bare, declares=declares),
             "field": _Lin("field", n, k, 2, gain, bare=bare, declares=declares)}
    gm = GraphManager()
    for name in (first, "field" if first == "body" else "body"):
        gm.add_node(nodes[name])
    gm.add_edge("body", "field", "x", "u")
    gm.add_edge("field", "body", "x", "u")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["body", "field"], **group)
        gm.compile()
        gm.step()
    return gm


def _report(gm) -> dict:
    report = dict(gm.coupling_diagnostics()[KEY])
    assert_reason_rules(report)
    return report


def _report_with(gm, **slots) -> dict:
    """The report of *gm*'s state with the named ``_meta`` slots of its
    group replaced (what a step that measured them would have left)."""
    kept = gm._state                                                        # noqa: SLF001
    meta = dict(kept["_meta"])
    for suffix, value in slots.items():
        slot = f"coupling_{KEY}_{suffix}"
        meta[slot] = np.asarray(value, np.asarray(meta[slot]).dtype)
    gm._state = {**kept, "_meta": meta}                                     # noqa: SLF001
    try:
        return _report(gm)
    finally:
        gm._state = kept                                                    # noqa: SLF001


#: A diagnosed pair whose loop closes through six scalars, at a tolerance
#: above its float floor, with parameters and declared evaluation counts:
#: both flags stand.
DIAGNOSED = dict(diagnostics=True, convergence_norm="mixed", rtol=1e-4, max_iterations=60)


@pytest.fixture(scope="module")
def narrow():
    gm = _pair(6, 40, **DIAGNOSED)
    report = _report(gm)
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    return gm


@pytest.fixture(scope="module")
def wide():
    """The same pair with a body of nine scalars: wider than the steps span."""
    return _pair(9, 40, **DIAGNOSED)


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------

def test_the_codes_are_lower_case_constants_each_in_one_kind_and_in_the_guide_s_table():
    """Each code is a module constant spelt as its own lower-case value,
    listed once in ``ALL``, in exactly one of the three kinds, and has one
    row in the guide's table."""
    constants = {name: value for name, value in vars(rc).items()
                 if name.isupper() and isinstance(value, str)}
    assert sorted(constants.values()) == sorted(rc.ALL) and len(set(rc.ALL)) == len(rc.ALL)
    for name, value in constants.items():
        assert value == name.lower() and re.fullmatch(r"[a-z][a-z_]*[a-z]", value), name
        assert name in rc.__all__, name
    kinds = (rc.EXPECTED, rc.CONFIGURATION, rc.WORRY)
    assert set().union(*kinds) == set(rc.ALL) and sum(map(len, kinds)) == len(rc.ALL)
    assert rc.FLAGS == ("spectral_usable", "gradient_bound_usable", "precision_limited")
    guide = (REPO / "docs/developer_guide/coupling_algorithm_guide.md").read_text()
    for code in rc.ALL:
        assert len(re.findall(rf"^\| `{code}` \|", guide, re.M)) == 1, code
    documented = GraphManager.coupling_diagnostics.__doc__
    assert '"reason_codes"' in documented and '"residual_precision_floor"' in documented


# ---------------------------------------------------------------------------
# One constructed case per code
# ---------------------------------------------------------------------------

def test_a_usable_report_has_no_code_no_reason_and_a_measured_floor(narrow):
    report = _report(narrow)
    assert report["reason_codes"] == {flag: [] for flag in rc.FLAGS}
    assert "not_usable_reason" not in report
    assert 0.0 < report["residual_precision_floor"] < report["residual"]
    assert report["precision_limited"] is False
    # At its float floor too: every member declares its evaluation count,
    # so the floor is the bound, the flags stand, and no code is attached
    # to a flag that is True.
    stalled = _report_with(narrow, residual=0.0)
    assert stalled["precision_limited"] and stalled["spectral_usable"], stalled
    assert stalled["reason_codes"] == {flag: [] for flag in rc.FLAGS}
    assert "not_usable_reason" not in stalled


def test_the_plain_float32_pair_at_the_default_norm_and_tolerance_says_why():
    """The report that had both flags ``False`` and no reason at all: a
    diagnosed pair of undeclared float32 relays under the default ``l2``
    norm and tolerance converges to its float floor.  ``rho_spectral`` is
    right; the flag is down for the floor alone, and says so."""
    gm = _pair(6, 40, declares=False, diagnostics=True, max_iterations=60)
    report = _report(gm)
    assert report["converged"] and report["precision_limited"], report
    assert not report["spectral_usable"] and not report["gradient_bound_usable"]
    assert report["reason_codes"]["spectral_usable"] == [rc.AT_FLOAT_FLOOR]
    assert report["reason_codes"]["gradient_bound_usable"] == [rc.AT_FLOAT_FLOOR]
    assert report["residual"] <= report["residual_precision_floor"]
    assert math.isfinite(report["rho_spectral"]) and report["rho_spectral"] < 1.0


@pytest.mark.parametrize("group, code", [
    (dict(), rc.DIAGNOSTICS_OFF),
    (dict(diagnostics=True, solver="fori"), rc.SOLVER_NOT_IFT),
    (dict(diagnostics=True, max_iterations=1), rc.SINGLE_PASS),
], ids=["diagnostics-off", "fori", "one-pass"])
def test_a_step_that_computes_no_estimate_says_which_setting_it_was(group, code):
    report = _report(_pair(6, 12, convergence_norm="mixed", rtol=1e-4, **group))
    assert math.isnan(report["rho_spectral"])
    for flag in USABLE:
        assert report["reason_codes"][flag] == [code], (flag, report["reason_codes"])
    # The floor is measured all the same: no code for ``precision_limited``.
    assert report["reason_codes"]["precision_limited"] == []
    assert report["residual_precision_floor"] > 0.0


def test_each_cause_the_slots_can_show_has_its_code(narrow):
    """The report is host arithmetic on what the step stored, so each
    cause is constructed by storing what a step that met it stores, on one
    compiled graph.  Each asserts the code; none reads the sentence."""
    nan, inf = math.nan, math.inf
    cases = {
        # No estimate in a state whose group asks for one.
        rc.ESTIMATE_NOT_RECORDED: (dict(rho_spectral=nan), "spectral_usable"),
        # What a step leaves at a state that is not finite.
        rc.STATE_NOT_FINITE: (dict(rho_spectral=nan, spectral_residual=nan,
                                   spectral_amplification=nan, residual=inf,
                                   amplification=0.0), "spectral_usable"),
        rc.BOUND_NOT_EVALUATED: (dict(residual=inf, amplification=0.0), "spectral_usable"),
        rc.NOT_CONTRACTING: (dict(rho_spectral=1.25), "spectral_usable"),
        rc.SPECTRAL_SELF_CHECK_FAILED: (dict(spectral_residual=0.3), "spectral_usable"),
        rc.GRADIENT_BOUND_NOT_COMPUTED: (dict(gradient_relative_error_bound=nan),
                                         "gradient_bound_usable"),
        rc.GRADIENT_BOUND_NOT_CERTIFIED: (dict(gradient_relative_error_bound=inf),
                                          "gradient_bound_usable"),
    }
    for code, (slots, flag) in cases.items():
        report = _report_with(narrow, **slots)
        assert report[flag] is False and code in report["reason_codes"][flag], (code, report)
        if flag == "gradient_bound_usable":
            # The gradient flag's own cause: the spectral flag stands, with no code.
            assert report["spectral_usable"] and report["reason_codes"][flag] == [code]
        else:
            assert report["reason_codes"]["spectral_usable"] == [code], (code, report)
    # A singular compressed resolvent reads as the pass not contracting too.
    report = _report_with(narrow, spectral_amplification=inf)
    assert report["reason_codes"]["spectral_usable"] == [rc.NOT_CONTRACTING]
    # A state that is not finite has no float floor either, and says so.
    report = _report_with(narrow, **cases[rc.STATE_NOT_FINITE][0])
    assert report["precision_limited"] is False


def test_a_solve_that_left_float_range_says_so():
    """End to end: relays of gain three diverge to a state that is not
    finite.  The step computes nothing there, and the report says which
    cause it is.  (The float floor under this norm is a count of ``eps``,
    the same at any state, so it is still a number; ``precision_limited``
    is ``False`` because the residual is not finite.)"""
    gm = _pair(4, 6, gain=3.0, max_iterations=200, strict_convergence=False,
               diagnostics=True, convergence_norm="mixed", rtol=1e-4)
    report = _report(gm)
    assert not report["converged"] and not math.isfinite(report["residual"]), report
    for flag in USABLE:
        assert report["reason_codes"][flag] == [rc.STATE_NOT_FINITE], report["reason_codes"]
    assert report["precision_limited"] is False


def test_a_floor_that_could_not_be_measured_has_a_code_and_reads_nan(narrow, monkeypatch):
    """``precision_limited`` reads ``False`` in two ways, and the codes tell
    them apart: measured and above the floor (no code), and not measured
    (a code, and NaN under ``residual_precision_floor``).  Constructed by
    making the floor's own function return NaN, as it does for a state the
    norm cannot read; the bound built on it is then not a number either."""
    from maddening.core import graph_manager as module  # noqa: PLC0415

    measured = _report(narrow)
    assert measured["precision_limited"] is False
    assert measured["reason_codes"]["precision_limited"] == []
    monkeypatch.setattr(module, "residual_precision_floor",
                        lambda *args, **kwargs: jnp.asarray(jnp.nan, jnp.float32))
    report = _report(narrow)
    assert math.isnan(report["residual_precision_floor"]), "the patch takes effect"
    assert report["precision_limited"] is False
    assert report["reason_codes"]["precision_limited"] == [rc.BOUND_NOT_EVALUATED]
    assert report["reason_codes"]["spectral_usable"] == [rc.BOUND_NOT_EVALUATED]


def test_several_causes_at_once_are_all_listed_in_the_table_s_order(narrow):
    """Not settled, at the float floor with a count that is not declared,
    and a gradient bound that did not certify: three codes, none picked."""
    kept = narrow._committed_floor_inputs                                   # noqa: SLF001
    evaluations, _declared, edges = kept[KEY]
    narrow._committed_floor_inputs = {KEY: (evaluations, False, edges)}     # noqa: SLF001
    try:
        report = _report_with(narrow, spectral_residual=0.3, residual=0.0,
                              gradient_relative_error_bound=math.inf)
    finally:
        narrow._committed_floor_inputs = kept                               # noqa: SLF001
    assert report["precision_limited"] is True
    assert report["reason_codes"]["spectral_usable"] == [
        rc.SPECTRAL_SELF_CHECK_FAILED, rc.AT_FLOAT_FLOOR]
    assert report["reason_codes"]["gradient_bound_usable"] == [
        rc.SPECTRAL_SELF_CHECK_FAILED, rc.AT_FLOAT_FLOOR, rc.GRADIENT_BOUND_NOT_CERTIFIED]
    reason = report["not_usable_reason"]
    for said in ("did not settle", "float floor", "Newton-Kantorovich"):
        assert said in reason, (said, reason)


def test_an_estimate_that_did_not_settle_is_too_wide_only_where_the_pass_is(narrow, wide):
    """The split the codes exist for.  Nine scalars close the wide pair's
    loop: the estimate cannot settle, which is expected, and the code says
    so.  The narrow pair's six are within the steps, so the same reading
    of its slot is the estimate's check of itself, a worry, and never the
    expected code.  And a state of eight entries is whole: within the
    steps whatever closes its loop."""
    report = _report(wide)
    assert not report["spectral_usable"] and math.isfinite(report["rho_spectral"])
    assert report["reason_codes"]["spectral_usable"] == [rc.INTERFACE_TOO_WIDE]
    assert rc.INTERFACE_TOO_WIDE in report["reason_codes"]["gradient_bound_usable"]
    assert "more independent interface scalars" in report["not_usable_reason"]
    assert wide._committed_pass_widths[KEY] == (9, 49)                      # noqa: SLF001
    assert narrow._committed_pass_widths[KEY] == (6, 46)                    # noqa: SLF001
    unsettled = _report_with(narrow, spectral_residual=0.3)
    assert unsettled["reason_codes"]["spectral_usable"] == [rc.SPECTRAL_SELF_CHECK_FAILED]
    assert "at most 6 scalar(s) in a state of 46 entries" in unsettled["not_usable_reason"]
    assert "more independent interface scalars" not in unsettled["not_usable_reason"]
    # The margin of an unsettled estimate can make the bound inf by itself:
    # the cause is still the estimate, and the report does not say that the
    # pass fails to contract.
    over = _report_with(wide, spectral_residual=0.6)
    assert over["spectral_error_bound"] == math.inf and over["rho_spectral"] < 1.0
    assert over["reason_codes"]["spectral_usable"] == [rc.INTERFACE_TOO_WIDE]
    assert "does not contract" not in over["not_usable_reason"]
    assert "spectral_error_bound is inf because" in over["not_usable_reason"]


@pytest.mark.parametrize("mode, sizes, order, width", [
    # Gauss-Seidel: what the loop closes through, in either sweep order.
    ("gauss-seidel", (6, 300), "body", (6, 306)),
    ("gauss-seidel", (6, 300), "field", (6, 306)),
    ("gauss-seidel", (7, 300), "body", (7, 307)),
    ("gauss-seidel", (8, 300), "field", (8, 308)),
    ("gauss-seidel", (13, 300), "body", (13, 313)),
    # Jacobi: both directions count; a whole state of eight is within.
    ("jacobi", (6, 300), "body", (306, 306)),
    ("jacobi", (4, 4), "body", (8, 8)),
    ("jacobi", (5, 5), "body", (10, 10)),
])
def test_the_width_of_a_pass_is_what_it_reads_from_the_previous_one(mode, sizes, order, width):
    """``_pass_width`` on graphs that are built and never compiled: the
    smaller of the entries read from the previous pass and of the members
    that read them, and the entries of the whole state."""
    k, n = sizes
    gm = GraphManager()
    nodes = {"body": _Lin("body", k, n, 1), "field": _Lin("field", n, k, 2)}
    for name in (order, "field" if order == "body" else "body"):
        gm.add_node(nodes[name])
    gm.add_edge("body", "field", "x", "u")
    gm.add_edge("field", "body", "x", "u")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["body", "field"], iteration_mode=mode)
    from maddening.core.coupling import _interface_plan  # noqa: PLC0415

    group = gm._coupling_groups[0]                                          # noqa: SLF001
    state = {name: node.initial_state() for name, node in nodes.items()}
    sweep = [name for name in gm.node_names if name in group.nodes]
    plan = _interface_plan.interface_plan(group.nodes, gm._edges, sweep,    # noqa: SLF001
                                          state, gm._nodes)                 # noqa: SLF001
    assert _group_layout._pass_width(group, gm._nodes, plan, state) == width   # noqa: SLF001
    spans = width[1] <= SPECTRAL_KRYLOV_STEPS or width[0] <= SPECTRAL_KRYLOV_STEPS - 1
    code, _clause = _group_layout._unsettled_cause(                         # noqa: SLF001
        width, rho=0.5, arnoldi_residual=0.3, fraction=0.05, steps=SPECTRAL_KRYLOV_STEPS)
    assert code == (rc.SPECTRAL_SELF_CHECK_FAILED if spans else rc.INTERFACE_TOO_WIDE)


def test_a_long_mapped_row_is_named_beside_whatever_else_holds_the_flags_down(narrow):
    """The row rule's code, on a report whose flags stand (it withdraws
    them) and on one whose flags another cause already holds down (it
    withdraws nothing, and is listed all the same: the floor the entry
    shows does not count the row's rounding)."""
    kept = narrow._committed_mapped_rows                                    # noqa: SLF001
    narrow._committed_mapped_rows = {                                       # noqa: SLF001
        KEY: (("field.x->body.u", "a dense matrix mapping, counted at the matrix's width,",
               400),)}
    try:
        report = _report(narrow)
        unsettled = _report_with(narrow, spectral_residual=0.3)
    finally:
        narrow._committed_mapped_rows = kept                                # noqa: SLF001
    honest = _report(narrow)
    assert report["reason_codes"]["spectral_usable"] == [rc.LONG_MAPPED_ROW]
    assert report["reason_codes"]["gradient_bound_usable"] == [rc.LONG_MAPPED_ROW]
    assert unsettled["reason_codes"]["spectral_usable"] == [
        rc.SPECTRAL_SELF_CHECK_FAILED, rc.LONG_MAPPED_ROW]
    for name in set(honest) - {*USABLE, "not_usable_reason", "reason_codes"}:
        assert np.asarray(report[name]).tobytes() == np.asarray(honest[name]).tobytes(), name


def test_the_geometry_rules_give_their_codes_where_they_decide():
    """The functions that decide a geometry group's report, with no graph:
    a group that solves positions; one whose positions are constants of
    its pass, whose causes are a plain group's; a missing record; the two
    readings of the self-check."""
    common = dict(bound=1e-3, gradient_bound=2e-2, rho=0.5, arnoldi_residual=1e-9,
                  settled=True, precision_limited=False, declared=False, limit=math.inf,
                  margin=math.inf, fraction=0.05, steps=SPECTRAL_KRYLOV_STEPS)
    keys = ["M.y->G.deposit"]

    def decided(**changed):
        usable, gradient, reason, spectral, own = _group_layout._geometry_flag_causes(  # noqa: SLF001
            keys, **{**common, **changed})
        codes = [code for code, _clause in spectral], [code for code, _clause in own]
        assert usable == (not codes[0]) and gradient == (usable and not codes[1])
        return usable, gradient, reason, *codes

    assert decided(solved=())[3:] == ([], [])
    usable, gradient, reason, spectral, own = decided(solved=("M.pos",))
    assert (usable, gradient, spectral, own) == (False, False, [rc.GEOMETRY_POSITIONS_SOLVED], [])
    assert decided(solved=None)[3] == [rc.GEOMETRY_RECORD_MISSING]
    assert decided(solved=(), margin=3.0)[3] == [rc.GEOMETRY_RECORD_MISSING]
    assert decided(solved=(), precision_limited=True)[3] == [rc.AT_FLOAT_FLOOR]
    assert decided(solved=(), gradient_bound=math.nan)[3:] == (
        [], [rc.GRADIENT_BOUND_NOT_COMPUTED])
    narrow_width = decided(solved=(), settled=False, arnoldi_residual=0.3, width=(6, 40))
    assert narrow_width[3] == [rc.SPECTRAL_SELF_CHECK_FAILED]
    assert decided(solved=(), settled=False, arnoldi_residual=0.3, width=(9, 40))[3] == [
        rc.INTERFACE_TOO_WIDE]
    check = _group_layout._geometry_self_check_code                         # noqa: SLF001
    assert check(0.4) == rc.GEOMETRY_SELF_CHECK_FAILED
    assert check(math.nan) == rc.GEOMETRY_SELF_CHECK_NOT_EVALUATED


# ---------------------------------------------------------------------------
# What the eight steps resolve (the guide's statement, measured)
# ---------------------------------------------------------------------------

def _true_radius(gm) -> float:
    body, field = gm.get_node("body"), gm.get_node("field")
    loop = np.asarray(body._M, np.float64) @ np.asarray(field._M, np.float64)   # noqa: SLF001
    return float(np.abs(np.linalg.eigvals(loop)).max())


# Per push: tests/core/test_coupling_report_reason_codes.py::test_an_estimate_that_did_not_settle_is_too_wide_only_where_the_pass_is
@pytest.mark.slow
@pytest.mark.parametrize("k, n", [(3, 3), (3, 300), (6, 6), (6, 300), (6, 3000), (7, 300)])
@pytest.mark.parametrize("first", ["body", "field"])
def test_a_loop_closing_through_a_few_scalars_is_resolved_whatever_is_on_the_other_side(k, n,
                                                                                         first):
    """The measurement behind the guide: a pair whose loop closes through
    3, 6 or 7 scalars, with a second member of up to 3000 entries, in
    either sweep order, has ``spectral_usable=True`` and the pass's
    radius; and with members that declare a parameter the gradient flag
    follows it."""
    gm = _pair(k, n, first=first, **DIAGNOSED)
    report = _report(gm)
    assert report["spectral_usable"] and report["gradient_bound_usable"], (k, n, first, report)
    assert report["rho_spectral"] == pytest.approx(_true_radius(gm), rel=2e-3)
    assert gm._committed_pass_widths[KEY][0] == k                           # noqa: SLF001


# Per push: tests/core/test_coupling_report_reason_codes.py::test_an_estimate_that_did_not_settle_is_too_wide_only_where_the_pass_is
@pytest.mark.slow
@pytest.mark.parametrize("k, n, mode", [(8, 300, "gauss-seidel"), (13, 300, "gauss-seidel"),
                                        (6, 300, "jacobi"), (5, 5, "jacobi")])
def test_a_wider_loop_is_not_and_its_report_says_that_is_expected(k, n, mode):
    """A member of eight scalars beside a larger one, a rigid body's full
    state (thirteen), and under Jacobi a pair whose two directions add up
    past the limit: the estimate does not settle, and the code is the
    expected one, never the worry."""
    report = _report(_pair(k, n, iteration_mode=mode, **DIAGNOSED))
    assert not report["spectral_usable"], (k, n, mode, report)
    assert report["reason_codes"]["spectral_usable"] == [rc.INTERFACE_TOO_WIDE]


# Per push: tests/core/test_coupling_report_reason_codes.py::test_a_usable_report_has_no_code_no_reason_and_a_measured_floor
@pytest.mark.slow
def test_members_that_declare_no_parameter_keep_the_spectral_flag_and_say_why_the_gradient_s_is_down():
    """What the gradient flag follows: with members that declare no
    parameter (and read no pre-step state) there is no constant for the
    fixed point to respond to, the gradient bound is NaN, and its flag is
    down for a cause of its own, beside a spectral flag that stands."""
    report = _report(_pair(6, 300, bare=True, **DIAGNOSED))
    assert report["spectral_usable"] and not report["gradient_bound_usable"], report
    assert report["reason_codes"]["gradient_bound_usable"] == [rc.GRADIENT_BOUND_NOT_COMPUTED]
    assert report["reason_codes"]["spectral_usable"] == []


# Per push: tests/core/test_coupling_report_reason_codes.py::test_a_usable_report_has_no_code_no_reason_and_a_measured_floor
@pytest.mark.slow
def test_a_whole_state_of_eight_entries_is_within_the_steps_under_jacobi():
    report = _report(_pair(4, 4, iteration_mode="jacobi", **DIAGNOSED))
    assert report["spectral_usable"] and report["gradient_bound_usable"], report


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------

def test_the_codes_and_the_floor_go_through_json_as_they_are(narrow, wide):
    """Plain ``json`` and the library's strict codec: the codes come back
    equal, lists of strings in a dict, and the floor a number (NaN where
    it was not measured, which the strict codec carries too)."""
    for report in (_report(narrow), _report(wide),
                   _report_with(narrow, rho_spectral=math.nan, spectral_residual=math.nan,
                                spectral_amplification=math.nan, residual=math.inf,
                                amplification=0.0)):
        for back in (json.loads(json.dumps(report)),
                     json_codec.loads(json_codec.dumps(report))):
            assert back["reason_codes"] == report["reason_codes"]
            assert type(back["reason_codes"]["spectral_usable"]) is list
            assert_reason_rules(back)
        json.loads(json_codec.dumps(report), parse_constant=lambda token: pytest.fail(token))


def test_a_checkpoint_brings_the_codes_back_and_one_saved_after_a_write_says_so(tmp_path):
    """A checkpoint holds the slots, and the codes are derived from them:
    loaded into a fresh state the report is the one that was saved, codes
    included.  One saved after the group's state was written has no state
    to measure the floor on, and lists that beside the cause the group
    already had (two codes, none picked), for all three flags' lists."""
    gm = _pair(6, 12, convergence_norm="mixed", rtol=1e-4)
    stepped = _report(gm)
    assert stepped["reason_codes"]["spectral_usable"] == [rc.DIAGNOSTICS_OFF]
    clean = gm.save_state(tmp_path / "stepped.npz")
    gm.set_node_state("body", {"x": jnp.zeros(6, jnp.float32)})
    written = gm.save_state(tmp_path / "written.npz")
    gm.reset_state()
    gm.load_state(clean)
    assert _report(gm)["reason_codes"] == stepped["reason_codes"]
    gm.reset_state()
    gm.load_state(written)
    loaded = _report(gm)
    both = [rc.DIAGNOSTICS_OFF, rc.WRITTEN_BEFORE_SAVE]
    assert loaded["reason_codes"] == {"spectral_usable": both, "gradient_bound_usable": both,
                                      "precision_limited": [rc.WRITTEN_BEFORE_SAVE]}
    assert math.isnan(loaded["residual_precision_floor"])
    assert loaded["not_usable_reason"].startswith("this report was loaded from a checkpoint")
    gm.step()
    assert _report(gm)["reason_codes"] == stepped["reason_codes"]


# ---------------------------------------------------------------------------
# Nothing else moves
# ---------------------------------------------------------------------------

#: Every key a report had before it carried its causes, less the reason.
NUMBERS = ("iterations", "total_iterations", "residual", "amplification", "error_estimate",
           "ratio_usable", "gradient_error_estimate", "converged", "rho_spectral",
           "spectral_error_bound", "spectral_usable", "gradient_relative_error_bound",
           "gradient_bound_usable", "precision_limited")


def test_the_codes_read_the_report_and_move_nothing_in_it(narrow, wide, monkeypatch):
    """With the two functions that derive the causes made to return
    nothing, every number and every flag of a report is the same to the
    bit: they are read from the report, never the other way."""
    def numbers(gm, **slots):
        report = dict(_report_with(gm, **slots)) if slots else dict(gm.coupling_diagnostics()[KEY])
        return {name: np.asarray(report[name]).tobytes() for name in NUMBERS}

    cases = [(narrow, {}), (wide, {}), (narrow, dict(spectral_residual=0.3)),
             (narrow, dict(gradient_relative_error_bound=math.inf)),
             (narrow, dict(rho_spectral=1.25))]
    with_codes = [numbers(gm, **slots) for gm, slots in cases]
    monkeypatch.setattr(_group_layout, "_flag_causes", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(_group_layout, "_causes_sentence", lambda *args, **kwargs: None)
    for (gm, slots), want in zip(cases, with_codes):
        kept = gm._state                                                    # noqa: SLF001
        if slots:
            meta = dict(kept["_meta"])
            for suffix, value in slots.items():
                slot = f"coupling_{KEY}_{suffix}"
                meta[slot] = np.asarray(value, np.asarray(meta[slot]).dtype)
            gm._state = {**kept, "_meta": meta}                             # noqa: SLF001
        try:
            bare = dict(gm.coupling_diagnostics()[KEY])
        finally:
            gm._state = kept                                                # noqa: SLF001
        assert {name: np.asarray(bare[name]).tobytes() for name in NUMBERS} == want
    # The premise: the patch takes effect (a False flag is left with no code).
    assert bare["spectral_usable"] is False and bare["reason_codes"]["spectral_usable"] == []


def test_the_table_of_coupling_report_quotes_only_the_reasons_it_always_did(narrow):
    """``coupling_report()`` is read by people and by tests of its own: a
    flag that is ``False`` for want of diagnostics, or for a cause its
    numbers show, gets the lines it always had, not the new sentence."""
    quiet = _pair(6, 12, convergence_norm="mixed", rtol=1e-4)
    assert "not_usable_reason" in _report(quiet)
    (row,) = list(quiet.coupling_report())
    assert not any("diagnostics=False" in flag for flag in row["flags"]), row["flags"]
    kept = narrow._state                                                    # noqa: SLF001
    meta = dict(kept["_meta"])
    slot = f"coupling_{KEY}_spectral_residual"
    meta[slot] = np.asarray(0.3, np.asarray(meta[slot]).dtype)
    narrow._state = {**kept, "_meta": meta}                                 # noqa: SLF001
    try:
        (row,) = list(narrow.coupling_report())
    finally:
        narrow._state = kept                                                # noqa: SLF001
    assert "spectral_usable=False: the spectral bound is not settled or not finite" in row["flags"]
