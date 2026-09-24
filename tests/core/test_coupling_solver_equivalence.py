"""What ``solver="fori"`` and ``solver="ift"`` owe each other.

``solver`` is advertised as a migration knob: ``"fori"`` is deprecated
and removed in the next minor, so a graph is expected to move to the
``"ift"`` default and get the same physics.  On an exit that ran out of
iterations the two have always agreed bit-for-bit -- they spend the same
number of passes by construction.  On an exit that *met the convergence
criterion* they used not to, and the gap was not round-off: ``fori``
froze on the iterate whose residual passed while ``ift`` returned that
iterate advanced by one more pass, so the two answers differed by one
whole residual -- 2.14% on the affine cycle below, the figure measured
on MIME's D2 two-scale Schwarz group
(``plans/MIME_vs_MADDENING_040_BASELINE.md`` §4.2), with both solvers
reporting ``converged=True``.

**Both solvers now return the iterate that passed** (decision D2 in
``plans/MADDENING_040_DECISIONS.md``).  Two things follow, and this
module holds both:

*The forward solve agrees.*  Whatever the norm, the acceleration and
the leaf types, a converged exit hands back the same state from either
solver, and that state is the one whose residual the reported flag is
about.  Relative to the value returned the gap is now zero, not small.

*The adjoints still differ, by design, and the difference is
bounded.*  ``fori`` differentiates straight through its iterates, so it
returns the derivative of the truncated iterate it computed.  ``ift``
applies the implicit function theorem, so it returns the derivative of
``F``'s *fixed point*, linearised at the state handed back.  Those are
different questions, and they converge on the same answer exactly as
the criterion tightens: a finite difference of ``ift``'s own forward is
not the quantity its adjoint computes, and the disagreement is of order
the residual the group reported.  Tighten the criterion and both
gradients land on the fixed point's.

Neither ``max_iterations`` nor ``tolerance`` is the knob that moves any
of this.  An exit on the criterion happens before the cap, so raising
the cap changes nothing; and ``tolerance`` is *not read at all* under
the ``"interface"`` and ``"mixed"`` norms, whose threshold is hard-coded
to ``1.0`` -- there the live knobs are ``atol`` and ``rtol``.  That is
why the two controls run against MIME's D2 group left every digit in
place.

Reproduction and the full ingredient sweep:
``benchmarks/results/repro_ift-fori-divergence/REPORT.md``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.group import _FIELD_DEFAULTS, _INERT_RULES
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import scale as _register_scale
from maddening.nodes.rigid_body_2d import RigidBody2DNode

#: Gauss-Seidel contraction factor of the two-node cycle below.  The
#: fixed point is ``scale / (1 - _RHO)``; one extra pass moves the
#: answer by ``_RHO`` of itself, which is the 2.14% quoted above.
_RHO = 0.0214

#: An L2 tolerance looser than the group's first-pass step, so the
#: criterion is met immediately at O(1) state values.  Everything then
#: stays comfortably inside float32, and the test does not depend on
#: whether the session enabled x64.  The same early exit happens with
#: the *default* tolerances whenever the interface quantity is small in
#: the units it is written in -- see ``_SMALL`` below.
_LOOSE = 10.0

#: The unforced route to the same early exit: an interface field three
#: decades below the default ``atol`` of 1e-8, which is where MIME's D2
#: group sits (drag force ~1.7e-05 N).  Nothing is loosened here; the
#: default criterion is simply absolute at this magnitude.
_SMALL = 1e-9


class _Flow(SimulationNode):
    """``tau <- gain * scale + disp``, optionally carrying an int32 leaf."""

    def __init__(self, name, timestep, gain=1.0, scale=1.0, int_leaf=False):
        super().__init__(name, timestep, gain=gain)
        self._scale = scale
        self._int_leaf = int_leaf

    def initial_state(self):
        state = {"tau": jnp.asarray(0.0)}
        if self._int_leaf:
            state["i_step"] = jnp.asarray(0, jnp.int32)
        return state

    def boundary_input_spec(self):
        return {"disp": BoundaryInputSpec(shape=(), description="displacement")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        disp = boundary_inputs.get("disp", jnp.asarray(0.0))
        out = {"tau": p["gain"] * self._scale + disp}
        if self._int_leaf:
            out["i_step"] = state["i_step"] + jnp.asarray(1, jnp.int32)
        return out


class _Struct(SimulationNode):
    """``disp <- b * tau``."""

    def __init__(self, name, timestep, b=_RHO):
        super().__init__(name, timestep, b=b)

    def initial_state(self):
        return {"disp": jnp.asarray(0.0)}

    def boundary_input_spec(self):
        return {"tin": BoundaryInputSpec(shape=(), description="traction")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        tin = boundary_inputs.get("tin", jnp.asarray(0.0))
        return {"disp": p["b"] * tin}


def _graph(solver, *, scale=1.0, norm="l2", tolerance=None, atol=None,
           rtol=None, acceleration="none", max_iterations=8, int_leaf=False):
    """A two-node affine cycle that meets its criterion on pass one.

    Only the tolerance knobs ``norm`` actually reads are forwarded, plus
    any the caller named outright.  ``CouplingGroup`` warns about a
    knob its norm ignores, so forwarding all three unconditionally
    would make every non-L2 cell here raise under
    ``filterwarnings = ["error"]`` -- and the one test that *wants* a
    dead ``tolerance``
    (:func:`test_the_cap_and_the_tolerance_are_both_inert_under_the_interface_norm`)
    still gets one, and expects the warning that comes with it.
    """
    knobs: dict[str, float] = {}
    if norm == "l2":
        knobs["tolerance"] = _LOOSE if tolerance is None else tolerance
    else:
        knobs["atol"] = 1e-8 if atol is None else atol
        knobs["rtol"] = 1e-6 if rtol is None else rtol
    for name, value in (("tolerance", tolerance), ("atol", atol),
                        ("rtol", rtol)):
        if value is not None:
            knobs[name] = value
    gm = GraphManager()
    gm.add_node(_Flow("flow", 0.01, scale=scale, int_leaf=int_leaf))
    gm.add_node(_Struct("struct", 0.01))
    gm.add_edge("flow", "struct", "tau", "tin")
    gm.add_edge("struct", "flow", "disp", "disp")
    gm.add_coupling_group(
        ["flow", "struct"], max_iterations=max_iterations,
        convergence_norm=norm, acceleration=acceleration,
        solver=solver, diagnostics=True, strict_convergence=False, **knobs,
    )
    gm.compile()
    return gm


def _tau(solver, **kw):
    """One step; returns ``(tau, that group's diagnostics)``."""
    gm = _graph(solver, **kw)
    gm.step()
    return (float(gm.get_node_state("flow")["tau"]),
            gm.coupling_diagnostics()["flow+struct"])


@functools.lru_cache(maxsize=None)
def _tau_default(solver):
    """``_tau(solver)`` on the default graph, once per module.

    Four tests below read the same two one-step solves; each rebuilt and
    recompiled both.  The result is a pure function of the solver, so
    it is computed once and shared.
    """
    return _tau(solver)


@functools.lru_cache(maxsize=None)
def _grad_and_fd(solver, *, h=1e-2, **kw):
    """``(analytic d tau / d gain, central difference of the same)``.

    The traced evaluation builds its own ``GraphManager``: ``run_scan``
    writes the new state back onto the manager, so reusing one across a
    traced and an untraced call leaks a tracer out of the trace.  The two
    untraced evaluations share one, reset to its initial state before
    each (``reset_state`` restores the coupling seeds as well), so the
    difference is taken with one compiled scan rather than two.  ``h`` is
    large because the map is affine in ``gain`` -- a central difference
    of an affine function is exact for any step, and a large one keeps
    the subtraction well away from float32 round-off.

    Cached per configuration: it is a pure function of its arguments,
    and two tests ask for the default one.
    """
    def loss(p, gm=None):
        gm = gm if gm is not None else _graph(solver, **kw)
        return jnp.sum(gm.run_scan(1, params=p)["flow"]["tau"])

    fd_graph = _graph(solver, **kw)
    base = fd_graph.params
    analytic = float(jax.jit(jax.grad(loss))(base)["nodes"]["flow"]["gain"])

    def shifted(delta):
        p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p["nodes"]["flow"]["gain"] = base["nodes"]["flow"]["gain"] + delta
        fd_graph.reset_state()
        return float(loss(p, fd_graph))

    return analytic, (shifted(h) - shifted(-h)) / (2 * h)


# ---------------------------------------------------------------------------
# The forward solve: one state, from either solver
# ---------------------------------------------------------------------------

def test_the_two_solvers_return_the_same_state_on_a_converged_exit():
    """The contract ``solver`` is advertised under.

    Was a strict xfail: the two returned states used to be one whole
    pass of the group map apart (2.14% here).  Both solvers now stop on
    the iterate whose residual met the criterion, so migrating a graph
    from the deprecated ``fori`` to the ``ift`` default cannot move a
    digit.
    """
    fori, _ = _tau_default("fori")
    ift, _ = _tau_default("ift")
    assert ift == pytest.approx(fori, rel=1e-6)


def test_the_state_returned_is_the_one_measured_not_its_successor():
    """Of the two adjacent iterates, the measured one is the contract.

    The group's Gauss-Seidel map on this graph is
    ``tau -> scale + _RHO * tau``.  The pass that satisfied the
    criterion had already computed its successor by the time the loop
    stopped; that successor -- ``1 + _RHO * tau``, which is what ``ift``
    used to hand back -- is discarded, because nothing measured it.
    What the caller gets is ``tau``, and the residual reported alongside
    it is a measurement of ``tau`` itself.
    """
    ift, diag = _tau_default("ift")
    successor = 1.0 + _RHO * ift
    assert ift == pytest.approx(1.0, rel=1e-6), "the measured iterate"
    assert successor != pytest.approx(ift, rel=1e-4), (
        "fixture premise: the discarded update is not round-off away"
    )
    # ``residual`` is the group's L2 norm of one pass applied to the
    # returned state, i.e. the distance to that discarded successor.
    # The norm divides each field's change by that field's own
    # magnitude (scale-aware since 0.4.0), and both fields here move by
    # the same *relative* amount, so the two contributions are equal.
    rel = (successor - ift) / successor
    assert diag["residual"] == pytest.approx(
        (2.0 * rel ** 2) ** 0.5, rel=1e-4,
    )


def test_both_solvers_report_the_same_verdict_about_the_same_state():
    """The report and the state it describes now travel together.

    ``coupling_diagnostics`` reported the same residual and
    ``converged=True`` for both solvers even while their answers were
    2.14% apart, which is what made the divergence silent.  The reports
    still agree; now the states they describe do too.
    """
    fori, d_fori = _tau_default("fori")
    ift, d_ift = _tau_default("ift")
    assert d_fori["converged"] is True and d_ift["converged"] is True
    assert d_fori["residual"] == pytest.approx(d_ift["residual"], rel=1e-6)
    assert abs(ift - fori) / abs(fori) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("max_iterations", [2, 8, 40])
@pytest.mark.parametrize("tolerance", [1e-4, 1e-14])
def test_the_cap_and_the_tolerance_are_both_inert_under_the_interface_norm(
    max_iterations, tolerance,
):
    """The two controls a reader reaches for first prove nothing here.

    The exit is on the criterion, before the cap, so ``max_iterations``
    cannot matter; and under ``convergence_norm="interface"`` the
    threshold is hard-coded to ``1.0``, so ``tolerance`` is never read.
    A 20x cap increase and a ten-order tolerance tightening leave every
    digit of both answers where it was.  This is the configuration that
    made MIME's D2 group look like two solvers converging to different
    fixed points: both controls were no-ops, so the invariance they
    showed was not evidence of anything.  It is recorded here because
    that reasoning error is easier to repeat than to spot -- the
    solvers now agree, but they still agree on an under-converged
    state, and neither knob is the one that changes that (see
    :func:`test_tightening_the_live_threshold_reaches_the_fixed_point`).
    """
    kw = dict(scale=_SMALL, norm="interface", max_iterations=max_iterations,
              tolerance=tolerance)
    # Setting the dead knob is the point of the test, and the group now
    # says so at construction -- which is the whole remedy for the day
    # this configuration cost.
    with pytest.warns(UserWarning, match=r"CouplingGroup\.tolerance"):
        fori, d = _tau("fori", **kw)
        ift, _ = _tau("ift", **kw)
    assert d["converged"] is True
    assert fori == pytest.approx(_SMALL, rel=1e-5)
    assert ift == pytest.approx(_SMALL, rel=1e-5)


def test_tightening_the_live_threshold_reaches_the_fixed_point():
    """``tolerance`` under the L2 norm *is* live.

    Both solvers agree on a loose criterion now, but they agree on an
    iterate that is one residual short of the fixed point.  Tightening
    the knob that is actually read puts them on the fixed point itself
    -- which is the only thing that makes the reported state, the
    reported residual and the IFT adjoint all describe the same point.
    """
    kw = dict(tolerance=1e-6, max_iterations=200)
    fori, d = _tau("fori", **kw)
    ift, _ = _tau("ift", **kw)
    assert d["converged"] is True
    assert ift == pytest.approx(fori, rel=1e-5)
    assert fori == pytest.approx(1.0 / (1.0 - _RHO), rel=1e-5)


@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
@pytest.mark.parametrize("acceleration", ["none", "fixed", "aitken", "iqn-ils"])
@pytest.mark.parametrize("int_leaf", [False, True])
def test_the_solvers_agree_under_every_norm_acceleration_and_leaf_type(
    norm, acceleration, int_leaf,
):
    """Parity is not a property of one lucky configuration.

    The divergence this matrix used to measure at a flat 2.14% needed
    none of the exotic ingredients the standing hypothesis named --
    neither ``accelerated_fields``, nor an ``"interface"`` convergence
    norm, nor an int32 leaf in a node's state -- so the fix must not
    need them absent either.  The int32 leaf is the interesting cell:
    the ``ift`` path holds integer leaves out of its fixed-point vector
    while ``fori`` carries them, and the two still agree because every
    node integrates from the pre-step state, so an integer field's
    value cannot depend on the iterate.
    """
    # The mixed and interface norms ignore ``tolerance``; their
    # threshold is 1.0 against a residual scaled by ``atol + rtol*|v|``,
    # so the first-pass step of ~1.0 needs ``atol`` loosened to match
    # what ``_LOOSE`` does for the L2 norm.
    kw = dict(norm=norm, acceleration=acceleration, int_leaf=int_leaf)
    if norm != "l2":
        kw["atol"] = _LOOSE
    fori, d_fori = _tau("fori", **kw)
    ift, d_ift = _tau("ift", **kw)
    assert d_fori["converged"] is True and d_ift["converged"] is True, (
        "fixture premise: this cell must exit on the criterion, which is "
        "the only exit the two solvers used to disagree on"
    )
    assert ift == pytest.approx(fori, rel=1e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# The adjoints: the same state, two different derivatives
# ---------------------------------------------------------------------------

def test_the_ift_adjoint_is_the_sensitivity_of_the_fixed_point():
    """``ift``'s gradient answers a different question from ``fori``'s.

    This is the part the forward fix deliberately does *not* change.
    The implicit function theorem gives ``d x*/d theta`` for the fixed
    point ``x*`` of the group map, here ``scale/(1 - rho)``, so the
    adjoint is ``1/(1 - rho)`` whatever iterate the forward stopped on
    and however loose the criterion was.  ``fori`` differentiates
    straight through its iterates, so it gives the derivative of the
    truncated iterate it actually returned -- exactly ``1.0`` after one
    pass of ``tau = gain * scale + disp`` from ``disp = 0``.  Both are
    correct answers; they are answers to different questions, and a
    user picking ``solver`` is picking between them.
    """
    ift_g, _ = _grad_and_fd("ift")
    fori_g, _ = _grad_and_fd("fori")
    assert ift_g == pytest.approx(1.0 / (1.0 - _RHO), rel=1e-5)
    assert fori_g == pytest.approx(1.0, rel=1e-5)


def test_the_ift_adjoint_disagrees_with_its_forward_by_about_the_residual():
    """The documented error bound, asserted rather than assumed.

    Was a strict xfail asserting the opposite -- that ``ift``'s adjoint
    should match a central difference of ``ift``'s own forward.  It
    cannot, and no choice of which adjacent iterate to return would
    make it: a finite difference of the forward differentiates the
    truncated iterate, the adjoint differentiates the fixed point, and
    the gap between the two is of order how far the iterate is from
    that fixed point.  ``_ift_solve_impl`` documents it as
    ``residual * cond(I - dF/dx)``; here ``cond`` is O(1), so the
    reported residual bounds it directly.  ``fori``, differentiating
    what it computed, matches its own difference to round-off.

    Returning the measured iterate made this gap *larger*, from
    ``rho^2/(1+rho)`` to ``rho``, because the discarded successor was
    the nearer of the two to the fixed point.  That is the price of a
    convergence flag that describes the state it is attached to, and
    the way to pay less of it is to tighten the live criterion -- see
    :func:`test_both_gradients_agree_once_the_forward_has_converged`.
    """
    fori_g, fori_fd = _grad_and_fd("fori")
    assert fori_g == pytest.approx(fori_fd, rel=1e-5), "fori is the control"

    ift_g, ift_fd = _grad_and_fd("ift")
    _tau_value, diag = _tau_default("ift")
    assert diag["converged"] is True
    assert abs(ift_g - ift_fd) <= 2.0 * diag["residual"], (
        f"analytic {ift_g} vs finite difference {ift_fd}: the adjoint may "
        f"only be as wrong as the reported residual ({diag['residual']}) "
        "times the conditioning of (I - dF/dx), which is O(1) here"
    )


def test_both_gradients_agree_once_the_forward_has_converged():
    """Tighten the live criterion and the two questions have one answer.

    At ``tolerance=1e-6`` the forward is on the fixed point, so the
    derivative of the iterate and the derivative of the fixed point are
    the same number, and each solver agrees with its own central
    difference to float32 round-off.  This is the configuration a user
    who cares about gradients should be in.
    """
    kw = dict(tolerance=1e-6, max_iterations=200)
    exact = 1.0 / (1.0 - _RHO)
    for solver in ("fori", "ift"):
        analytic, fd = _grad_and_fd(solver, **kw)
        assert analytic == pytest.approx(fd, rel=1e-4), solver
        assert analytic == pytest.approx(exact, rel=1e-5), solver


# ---------------------------------------------------------------------------
# The residual's own resolution: what "the same report" can mean in float32
# ---------------------------------------------------------------------------
#
# The fixtures above live at O(1) state values with a loosened criterion, so
# their residuals are O(1e-2) and the two solvers agree on them to six digits.
# That is not the regime a converged multi-rate group is in.  There the
# residual is a *cancellation* -- every norm divides ``F(x) - x`` by a scale,
# and near the fixed point the numerator is the difference of two nearly equal
# float32 states -- so one unit in the last place of either state is a
# full-size contribution to it.
#
# The two solvers cannot avoid rounding differently: ``"ift"`` runs its passes
# in a ``lax.while_loop`` (it has to, to exit early) and ``"fori"`` in a
# ``lax.fori_loop``, and XLA compiles the two bodies to differently rounded
# arithmetic.  Measured on ``_multirate_graph`` below: the *same* map applied
# to the *same* input vector moves two of ten float32 components by one ulp
# under one compilation (1.8626e-09, which is exactly
# ``np.spacing(float32(node_1.angle))``) and leaves all ten bit-identical
# under the other.  That is the difference between reporting 1.04e-05 and
# reporting exactly 0.0, with both solvers returning the same state and both
# reporting ``converged=True``.
#
# So the contract is: same passes, same state, same verdict; and the reported
# residual agrees to the measurement's own resolution, derived below.  It is
# not "the same number".  That overstatement is what
# ``tests/property/test_coupling_error_bound.py`` used to assert, at an
# absolute tolerance of 1e-9 against a floor three thousand times larger.

#: float32 machine epsilon -- the relative size of one unit in the last
#: place, and so of the smallest state change the norms can resolve.
_F32_EPS = float(np.finfo(np.float32).eps)

#: Ulps of slack on the floor.  One ulp is the *minimum* two differently
#: compiled copies of a pass can differ by; a subcycled pass is a scan over
#: four coupled updates, so a few can accumulate.
#:
#: Measured, not guessed: over a 480-cell sweep of the fixture below (caps
#: 1-6, one to three waveform iterations, four accelerations, both iteration
#: modes, both comparable norms, one and two steps) the worst solver-to-solver
#: residual disagreement was **0.82 of a one-ulp floor** -- 82 cells disagreed
#: by more than ``rel=1e-4`` and not one exceeded a single ulp.  Eight is an
#: order of magnitude above that, which is the margin a float32 cancellation
#: needs to survive a different CPU or a different XLA release.
#:
#: It stays a real gate at that width.  Under the mixed norm it is
#: ``8 * eps / rtol``, ~1e-3 of that norm's threshold of 1.0; adding 0.01 to
#: one solver's reported residual fails 25 of the 26 cases below.  Under the
#: L2 norm it is ``8 * eps * sqrt(n)``, which for a small group is a few 1e-6
#: and so can *exceed* a tight ``tolerance``: an L2 group whose tolerance is
#: at or below the norm's float32 resolution has a criterion made of rounding,
#: and no comparison of two such residuals can say more than that.
_ULP_SLACK = 8.0


def residual_noise_floor(norm, rtol, n_float_entries):
    """How much of a reported ``residual`` is float32 rounding.

    One ulp of state is ``_F32_EPS * |x|`` in a norm's numerator, and
    the norm's own scale divides the ``|x|`` back out:

    * ``"mixed"`` and ``"interface"`` are the RMS of
      ``|dx| / (rtol * |x|)`` over the active fields, so one ulp per
      field is ``eps / rtol`` however many fields there are;
    * ``"l2"`` sums ``(|dx| / |x|)**2`` over every float entry and
      applies no ``rtol``, so one ulp per entry is ``eps * sqrt(n)``.

    Parameters
    ----------
    norm : str
        The group's ``convergence_norm``.
    rtol : float
        The group's ``rtol`` (unread by the L2 norm).
    n_float_entries : int
        Float scalars in the group's state, for the L2 norm's sum.

    Returns
    -------
    float
        The absolute residual difference below which two reports are
        the same measurement.
    """
    if norm == "l2":
        return _ULP_SLACK * _F32_EPS * math.sqrt(max(n_float_entries, 1))
    return _ULP_SLACK * _F32_EPS / rtol


#: The shrunk Hypothesis counterexample from
#: ``test_the_bound_is_the_same_on_both_solvers``, as literals.  The blob
#: replays only while ``tests/property/strategies.py`` is untouched; this
#: does not.
_MULTIRATE_GROUP = dict(
    max_iterations=2, tolerance=1e-06, convergence_norm="mixed", atol=1e-08,
    rtol=1e-03, diagnostics=True, acceleration="aitken", relaxation=1.0,
    iteration_mode="jacobi", accelerated_fields=None, subcycling=True,
    boundary_interpolation="constant", jacobian_reuse=0,
    waveform_iterations=2, predictor="linear", strict_convergence=False,
)

_MULTIRATE_NODES = ("node_1", "rod")


def _quiet_knobs(knobs):
    """*knobs* with every field this configuration does not read reset.

    The parametrisation below flips ``convergence_norm`` and ``solver``,
    which are gates: ``rtol`` is live under ``"mixed"`` and dead under
    ``"l2"``, ``linear_solver`` is live under ``"ift"`` and dead under
    ``"fori"``.  Leaving a stranded value behind makes
    :class:`CouplingGroup` warn, and ``filterwarnings = ["error"]``
    turns that into a failure about the fixture rather than about the
    solvers.

    The rules come from the library's own ``_INERT_RULES`` table rather
    than being restated, so a gate added there is respected here with no
    second edit -- the same argument
    :func:`tests.property.strategies.without_inert_knobs` makes.  Only
    knobs the table calls inert are touched, and only back to their
    declared default, so this cannot change what any cell measures.
    """
    view = SimpleNamespace(**{**_FIELD_DEFAULTS, **knobs})
    for rule in _INERT_RULES:
        if rule.live(view):
            continue
        for name in rule.fields:
            setattr(view, name, _FIELD_DEFAULTS[name])
    return {name: getattr(view, name) for name in knobs}


def _multirate_graph(solver, **overrides):
    """Two rigid bodies at 4:1 timesteps in a three-edge coupled cycle.

    Every constant is from the shrunk counterexample.  The ingredients
    that matter, and that a hand-built grid of two-node fixtures missed:
    a *three*-edge cycle (two edges rod->node_1, one back), an external
    input on a coupled field, subcycling at a 4:1 ratio, and a cap of
    two -- which together put the group inside its threshold of 1.0 at
    a residual of ~1e-05, i.e. at the norm's float32 resolution.
    """
    # ``scale(0.5)`` registers ``"scale_0.5"`` and returns the callable.
    # The recipe named it by string; the callable it resolves to is passed
    # instead, because a factory-registered name exists only once something
    # has called the factory and ``scripts/check_transforms.py`` -- which
    # reads the live registry -- correctly cannot see it statically.  The
    # edge is the same edge either way.
    half = _register_scale(0.5)
    gm = GraphManager()
    gm.add_node(RigidBody2DNode(
        name="rod", timestep=0.04, mass=1.5548670291900635,
        initial_vy=0.371927946805954, inertia=0.7841625809669495,
        initial_omega=-0.10158340632915497, initial_y=1.1920928955078125e-07,
        initial_angle=-0.33835136890411377,
        gravity=(0.4851800799369812, -7.662717819213867),
        initial_vx=0.0, initial_x=0.0,
    ))
    gm.add_node(RigidBody2DNode(
        name="node_1", timestep=0.01, mass=0.5, initial_y=-0.0,
        gravity=(-0.6507397294044495, -8.365083694458008),
        initial_vy=0.45580893754959106, initial_omega=0.62205970287323,
        initial_vx=-0.16179445385932922, initial_x=-0.0, inertia=0.5,
        initial_angle=0.0,
    ))
    gm.add_edge(source="rod", target="node_1", source_field="angle",
                target_field="torque", transform="identity", additive=True,
                source_units="N*m", target_units="N*m")
    gm.add_edge(source="rod", target="node_1", source_field="x",
                target_field="force", transform=half, additive=False)
    gm.add_edge(source="node_1", target="rod", source_field="x",
                target_field="force", transform="identity", additive=True)
    gm.add_external_input("rod", "torque", shape=())
    knobs = dict(_MULTIRATE_GROUP, solver=solver, **overrides)
    if solver == "ift":
        # Read by the IFT path alone; passing it to "fori" would strand
        # it and the group would warn, fatally under filterwarnings.
        knobs["linear_solver"] = "gmres"
    gm.add_coupling_group(["node_1", "rod"], **_quiet_knobs(knobs))
    gm.compile()
    return gm


#: ``{(solver, knobs): (graph, [(diagnostics, state) after step 1, 2, ...])}``.
_MULTIRATE_RUNS: dict = {}


def _multirate_run(solver, steps=1, **overrides):
    """``(diagnostics, state)`` for the group after ``steps`` steps.

    Memoised per solver and effective configuration: the step sequence
    from the initial state is deterministic, so the state after two
    steps is the state after one, stepped once more, and the tests below
    ask for the same configurations many times (the two tests of the
    counterexample and two cells of the class sweep are one
    configuration; every sweep cell is asked at one and at two steps).
    Each distinct configuration used to be rebuilt and retraced per ask,
    ~2.5 s of tracing per graph on the CI runner.  The graph is kept at
    the last step it reached and advanced only when a later step is
    asked for.
    """
    knobs = tuple(sorted({**_MULTIRATE_GROUP, **overrides}.items()))
    key = (solver, knobs)
    if key not in _MULTIRATE_RUNS:
        _MULTIRATE_RUNS[key] = (_multirate_graph(solver, **overrides), [])
    gm, history = _MULTIRATE_RUNS[key]
    while len(history) < steps:
        gm.step()
        history.append((
            dict(gm.coupling_diagnostics()["node_1+rod"]),
            {n: dict(gm.get_node_state(n)) for n in _MULTIRATE_NODES},
        ))
    return history[steps - 1]


#: Floor on the scale a state difference is divided by.  ``rod`` starts at
#: ``initial_y=1.19e-07`` and ``initial_x=0.0``, where a pure relative
#: measure reads one subnormal of difference as O(1).  Below this the check
#: is absolute instead, which at the 1e-5 tolerance it is used with is
#: 1e-11 -- orders above the few ulps float32 can put there.
_STATE_SCALE_FLOOR = 1e-6


def _state_gap(a, b):
    """Largest relative difference between two returned states."""
    worst = 0.0
    for node in _MULTIRATE_NODES:
        for field in a[node]:
            x = np.asarray(a[node][field], np.float64)
            y = np.asarray(b[node][field], np.float64)
            scale = max(np.max(np.abs(x)), np.max(np.abs(y)),
                        _STATE_SCALE_FLOOR)
            worst = max(worst, float(np.max(np.abs(x - y)) / scale))
    return worst


def _n_float_entries(state):
    return sum(
        int(np.asarray(v).size)
        for node in _MULTIRATE_NODES for v in state[node].values()
        if np.issubdtype(np.asarray(v).dtype, np.floating)
    )


def test_a_converged_multirate_group_returns_one_state_from_either_solver():
    """The counterexample, reduced to the claim that actually matters.

    ``coupling_diagnostics()`` reported ``residual=0.0`` under ``ift``
    and ``1.04e-05`` under ``fori`` on this graph, with
    ``converged=True`` on both.  The alarming reading -- that one
    solver had stopped somewhere the other had not -- is wrong: the
    state handed back is the same to float32 round-off, and it was
    measured bit-for-bit identical on the reference platform.  Only the
    *measurement* of how far that state is from the fixed point moved,
    because at 1e-05 in this norm the measurement is rounding.

    What this cannot see, and does not claim to: D2 itself.  This group
    is converged to float32, so the successor ``ift`` used to return
    instead of the measured iterate is ~1e-08 away relatively -- below
    any state tolerance worth writing.  Reverting D2 leaves every
    assertion here green and fails 33 of the affine-cycle tests above,
    which is where that contract is guarded.  This fixture guards the
    *reporting* contract in the regime the affine cycle never reaches.
    """
    d_ift, s_ift = _multirate_run("ift")
    d_fori, s_fori = _multirate_run("fori")

    assert d_ift["converged"] is True and d_fori["converged"] is True
    assert d_ift["iterations"] == d_fori["iterations"]
    assert _state_gap(s_ift, s_fori) <= 1e-5, (
        "the two solvers returned materially different states, which is a "
        "solver defect and not the reporting artefact this module documents"
    )


def test_the_residual_at_the_fixed_point_is_reported_to_its_own_resolution():
    """Below the noise floor, ``residual`` is a rounding measurement.

    The premise half of this test is as important as the assertion: if
    the fixture ever stops landing at the norm's float32 resolution it
    stops covering the defect, and the first assertion says so out
    loud rather than passing vacuously.
    """
    d_ift, s_ift = _multirate_run("ift")
    d_fori, _ = _multirate_run("fori")
    floor = residual_noise_floor(
        _MULTIRATE_GROUP["convergence_norm"], _MULTIRATE_GROUP["rtol"],
        _n_float_entries(s_ift),
    )

    assert max(d_ift["residual"], d_fori["residual"]) <= floor, (
        "fixture premise: this group must converge to within the norm's "
        "float32 resolution, which is the regime the two solvers cannot "
        "agree to more digits in"
    )
    assert abs(d_ift["residual"] - d_fori["residual"]) <= floor, (
        f"ift {d_ift['residual']} vs fori {d_fori['residual']}: the two "
        f"reports differ by more than the measurement's own resolution "
        f"({floor}), so this is a real disagreement and not round-off"
    )


def _multirate_cells():
    """The sweep's cells, ids as ``max_iterations-waveform-norm-steps``.

    Slow-marked except the cells in the counterexample's own
    configuration: every other cell is a pair of graphs traced and
    compiled for it alone, 4-10 s on the CI runner.  Those two run on
    every push beside the two tests of the counterexample itself, whose
    solves they share (``_multirate_run``); the rest of the class runs
    in slow-tests.yml.
    """
    own = (_MULTIRATE_GROUP["max_iterations"],
           _MULTIRATE_GROUP["waveform_iterations"],
           _MULTIRATE_GROUP["convergence_norm"])
    for max_iterations in (2, 3):
        for waveform_iterations in (1, 2, 3):
            for norm in ("mixed", "l2"):
                for steps in (1, 2):
                    marks = (() if (max_iterations, waveform_iterations, norm) == own
                             else (pytest.mark.slow,))
                    yield pytest.param(
                        max_iterations, waveform_iterations, norm, steps,
                        marks=marks,
                        id=f"{max_iterations}-{waveform_iterations}-{norm}-{steps}",
                    )


@pytest.mark.parametrize(
    "max_iterations,waveform_iterations,norm,steps", list(_multirate_cells()),
)
def test_the_multirate_solvers_agree_across_the_configuration_class(
    max_iterations, waveform_iterations, norm, steps,
):
    """Parity over the knobs the counterexample turned out to need.

    The hand-built grid that preceded this found zero disagreements in
    ~3,900 configurations, because none of them combined subcycling,
    waveform relaxation and a cap small enough to stop at the norm's
    resolution.  Every cell here does, and the invariant is the one
    that survives float32: same verdict, same state, and a residual
    agreeing to the floor derived above.

    ``max_iterations=1`` is excluded deliberately -- it returns before
    the solver branch, so both paths run identical code and the cell
    would prove nothing about the two loops.
    """
    over = dict(max_iterations=max_iterations,
                waveform_iterations=waveform_iterations,
                convergence_norm=norm)
    d_ift, s_ift = _multirate_run("ift", steps=steps, **over)
    d_fori, s_fori = _multirate_run("fori", steps=steps, **over)

    assert d_ift["converged"] == d_fori["converged"]
    assert _state_gap(s_ift, s_fori) <= 1e-5
    floor = residual_noise_floor(norm, _MULTIRATE_GROUP["rtol"],
                                  _n_float_entries(s_ift))
    assert abs(d_ift["residual"] - d_fori["residual"]) <= floor, (
        f"ift {d_ift['residual']} vs fori {d_fori['residual']} "
        f"(floor {floor})"
    )
