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

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

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


def _grad_and_fd(solver, *, h=1e-2, **kw):
    """``(analytic d tau / d gain, central difference of the same)``.

    Each evaluation builds its own ``GraphManager``: ``run_scan`` writes
    the new state back onto the manager, so reusing one across a traced
    and an untraced call leaks a tracer out of the trace.  ``h`` is
    large because the map is affine in ``gain`` -- a central difference
    of an affine function is exact for any step, and a large one keeps
    the subtraction well away from float32 round-off.
    """
    def loss(p, gm=None):
        gm = gm if gm is not None else _graph(solver, **kw)
        return jnp.sum(gm.run_scan(1, params=p)["flow"]["tau"])

    base = _graph(solver, **kw).params
    analytic = float(jax.grad(loss)(base)["nodes"]["flow"]["gain"])

    def shifted(delta):
        p = {"nodes": {n: dict(v) for n, v in base["nodes"].items()}}
        p["nodes"]["flow"]["gain"] = base["nodes"]["flow"]["gain"] + delta
        return float(loss(p))

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
    fori, _ = _tau("fori")
    ift, _ = _tau("ift")
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
    ift, diag = _tau("ift")
    successor = 1.0 + _RHO * ift
    assert ift == pytest.approx(1.0, rel=1e-6), "the measured iterate"
    assert successor != pytest.approx(ift, rel=1e-4), (
        "fixture premise: the discarded update is not round-off away"
    )
    # ``residual`` is the group's L2 norm of one pass applied to the
    # returned state, i.e. the distance to that discarded successor.
    assert diag["residual"] == pytest.approx(
        ((successor - ift) ** 2 + (_RHO * successor - _RHO * ift) ** 2) ** 0.5,
        rel=1e-4,
    )


def test_both_solvers_report_the_same_verdict_about_the_same_state():
    """The report and the state it describes now travel together.

    ``coupling_diagnostics`` reported the same residual and
    ``converged=True`` for both solvers even while their answers were
    2.14% apart, which is what made the divergence silent.  The reports
    still agree; now the states they describe do too.
    """
    fori, d_fori = _tau("fori")
    ift, d_ift = _tau("ift")
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
    _tau_value, diag = _tau("ift")
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
