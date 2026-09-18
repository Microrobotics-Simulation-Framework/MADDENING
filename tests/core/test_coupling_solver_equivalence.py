"""What ``solver="fori"`` and ``solver="ift"`` owe each other.

``solver`` is advertised as a migration knob: ``"fori"`` is deprecated
and removed in the next minor, so a graph is expected to move to the
``"ift"`` default and get the same physics.  On an exit that ran out of
iterations the two do agree bit-for-bit -- they spend the same number
of passes by construction.  On an exit that *met the convergence
criterion* they do not, and the gap is not round-off:

``fori`` freezes on the iterate whose residual passed; ``ift`` returns
that iterate advanced by one more pass.  The gap is therefore one whole
residual, and relative to the returned value it is small only when the
criterion is relatively tight.  Every convergence criterion here is
*absolute* at small magnitudes -- the L2 norm compares ``||dx||``
against ``tolerance`` outright, and the mixed / interface norms scale
by ``atol + rtol*|v|``, which is just ``atol`` once ``|v|`` is below
``atol/rtol`` -- so a group can satisfy its criterion on the first pass
while still being percent-sized away from its fixed point.  Both
solvers then report ``converged=True`` and hand back answers that
differ by percent.

Two consequences, pinned as strict xfails below:

*The forward solve differs.*  Reproduced here at 2.14% on a two-node
affine cycle, the figure measured on MIME's D2 two-scale Schwarz group
(``plans/MIME_vs_MADDENING_040_BASELINE.md`` §4.2).  The gap is exactly
one application of the group's own Gauss-Seidel map -- not a different
set of leaves in the fixed-point vector, not a different order, not a
dtype promotion.

*The IFT gradient stops matching its own forward.*  The implicit
function theorem is applied at a point that is not a fixed point, so
the analytic adjoint disagrees with a central difference of the very
function ``ift`` computes.  ``fori``, which differentiates straight
through its iterates, agrees with its own finite difference whether or
not the group converged.

Neither ``max_iterations`` nor ``tolerance`` is the knob that moves
this.  An exit on the criterion happens before the cap, so raising the
cap changes nothing; and ``tolerance`` is *not read at all* under the
``"interface"`` and ``"mixed"`` norms, whose threshold is hard-coded to
``1.0`` -- there the live knobs are ``atol`` and ``rtol``.  That is why
the two controls run against MIME's D2 group left every digit in place.

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


def _graph(solver, *, scale=1.0, norm="l2", tolerance=_LOOSE, atol=1e-8,
           rtol=1e-6, acceleration="none", max_iterations=8, int_leaf=False):
    """A two-node affine cycle that meets its criterion on pass one."""
    gm = GraphManager()
    gm.add_node(_Flow("flow", 0.01, scale=scale, int_leaf=int_leaf))
    gm.add_node(_Struct("struct", 0.01))
    gm.add_edge("flow", "struct", "tau", "tin")
    gm.add_edge("struct", "flow", "disp", "disp")
    gm.add_coupling_group(
        ["flow", "struct"], max_iterations=max_iterations,
        tolerance=tolerance, convergence_norm=norm, acceleration=acceleration,
        solver=solver, atol=atol, rtol=rtol, diagnostics=True,
        strict_convergence=False,
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
# The mechanism, and the controls that do not move it
# ---------------------------------------------------------------------------

def test_the_solver_gap_is_exactly_one_more_pass_of_the_group_map():
    """``ift``'s answer is ``fori``'s answer put through ``F`` once more.

    That is the whole of the difference.  The group's Gauss-Seidel map
    on this graph is ``tau -> scale + _RHO * tau``, and applying it to
    what ``fori`` returns reproduces what ``ift`` returns.  Neither path
    is solving a different system: they are reporting the same
    iteration one step apart.
    """
    fori, _ = _tau("fori")
    ift, _ = _tau("ift")
    assert ift == pytest.approx(1.0 + _RHO * fori, rel=1e-6)


def test_both_solvers_report_converged_at_the_states_they_disagree_about():
    """Nothing warns.  ``coupling_diagnostics`` reports the same
    residual and ``converged=True`` for both, at answers 2.14% apart."""
    fori, d_fori = _tau("fori")
    ift, d_ift = _tau("ift")
    assert d_fori["converged"] is True and d_ift["converged"] is True
    assert d_fori["residual"] == pytest.approx(d_ift["residual"], rel=1e-6)
    assert abs(ift - fori) / abs(fori) == pytest.approx(_RHO, rel=1e-5)


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
    fixed points: both controls were no-ops.
    """
    kw = dict(scale=_SMALL, norm="interface", max_iterations=max_iterations,
              tolerance=tolerance)
    fori, d = _tau("fori", **kw)
    ift, _ = _tau("ift", **kw)
    assert d["converged"] is True
    assert fori == pytest.approx(_SMALL, rel=1e-5)
    assert ift == pytest.approx(_SMALL * (1.0 + _RHO), rel=1e-5)


def test_tightening_the_live_threshold_closes_the_gap():
    """``tolerance`` under the L2 norm *is* live, and closing the
    criterion closes the gap.

    That is the diagnostic which separates this defect from the two
    paths genuinely solving different systems: iterate to the real
    fixed point and both solvers land on it.
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
def test_the_gap_needs_no_int_leaf_no_interface_norm_and_no_acceleration(
    norm, acceleration, int_leaf,
):
    """Every exotic ingredient the standing hypothesis named is innocent.

    That hypothesis was that ``accelerated_fields``, an ``"interface"``
    convergence norm and an int32 leaf in a node's state together made
    the two paths flatten different fixed-point vectors.  They do not.
    The gap is the same 2.14% with the plain L2 norm, no acceleration
    and no integer leaf, and adding the integer leaf does not move a
    digit -- the ``ift`` path holds integer leaves out of its
    fixed-point vector, but on this graph (as on any graph whose nodes
    integrate from the pre-step state) their value does not depend on
    the iterate, so holding them out costs nothing.
    """
    # The mixed and interface norms ignore ``tolerance``; their
    # threshold is 1.0 against a residual scaled by ``atol + rtol*|v|``,
    # so the first-pass step of ~1.0 needs ``atol`` loosened to match
    # what ``_LOOSE`` does for the L2 norm.
    kw = dict(norm=norm, acceleration=acceleration, int_leaf=int_leaf)
    if norm != "l2":
        kw["atol"] = _LOOSE
    fori, _ = _tau("fori", **kw)
    ift, _ = _tau("ift", **kw)
    assert abs(ift - fori) / abs(fori) == pytest.approx(_RHO, rel=1e-4)


# ---------------------------------------------------------------------------
# The two defects
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, pre-existing on release/0.4.0.  On an exit that met the "
        "convergence criterion the two solvers return states one whole "
        "pass apart -- fori freezes on the iterate whose residual "
        "passed, ift returns that iterate advanced once more -- so "
        "moving a graph off the deprecated fori onto the ift default "
        "silently changes the answer by one residual.  Here that is "
        "2.14%, the figure measured on MIME's D2 group, and both "
        "solvers report converged=True.  Closing it means choosing one "
        "of the two states as the contract (fori's measured iterate, "
        "or ift's further update plus a re-measured residual) and "
        "making both solvers return it."
    ),
)
def test_the_two_solvers_return_the_same_state_on_a_converged_exit():
    fori, _ = _tau("fori")
    ift, _ = _tau("ift")
    assert ift == pytest.approx(fori, rel=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known, pre-existing on release/0.4.0.  The IFT adjoint is "
        "valid only at a fixed point, but the state ift returns on a "
        "criterion exit is one pass short of one and the criterion that "
        "licensed the exit is absolute at this magnitude.  So the "
        "analytic gradient (scale/(1-rho)) disagrees with a central "
        "difference of the function ift itself computes "
        "(scale*(1+rho)) by rho^2/(1+rho) = 4.5e-04, while fori's "
        "straight-through gradient matches its own to round-off.  "
        "strict_convergence does not catch it: it re-tests the same "
        "residual that already passed."
    ),
)
def test_the_ift_gradient_matches_its_own_finite_difference():
    fori_g, fori_fd = _grad_and_fd("fori")
    assert fori_g == pytest.approx(fori_fd, rel=1e-5), "fori is the control"
    ift_g, ift_fd = _grad_and_fd("ift")
    assert ift_g == pytest.approx(ift_fd, rel=1e-5)
