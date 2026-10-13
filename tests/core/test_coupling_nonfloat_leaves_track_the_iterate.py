"""A coupled node's integer and boolean fields describe the state returned.

A node may compute a non-floating field from a boundary input -- a contact
flag reading a gap.  Such a field depends on the coupling iterate, and the
two solvers disagreed on it: ``solver="fori"`` recomputes it on every pass,
while ``solver="ift"`` kept only floating fields in its fixed-point vector
and restored every other field from the *first* pass, on the premise that
"its first-pass value is already the converged one" (MADD-ANO-059's
resolution and ``_floating_accel_fields``).  So the ift state carried a flag
computed from an input the solve had long left, beside floating fields at
the fixed point.  ``_run_ift_forward`` now recomputes the non-floating
fields from the returned floating ones with one more evaluation of the
pass, kept out of the linearisation.

Neighbouring cases pinned here, beyond the harness's reproducer: every
acceleration, both iteration modes, a predictor under every norm (the
predictor used to drop these fields from the iterate it starts from, and
the mixed norm then raised ``KeyError``), a sub-cycled member, waveform
sweeps, the IFT gradient, the retrace count, ``diagnostics`` bit-identity,
and a counter that reads no input.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property.coupled_graphs import live_knobs

#: ``a: x = k u + c`` and ``b: x = u``: the fixed point is ``c / (1 - k)``.
K, C = -0.5, -0.1
X_STAR = C / (1.0 - K)          # -1/15: a's input is negative at the fixed point


class Contact(SimulationNode):
    """``x = k u + c``; ``touching = u > 0`` reads the coupled input; ``n`` counts.

    ``touching`` is the field the solvers disagreed on: it depends on the
    iterate.  ``n`` reads the pre-step state only, the case the old
    premise was true for.  ``leaves=False`` builds the same node without
    either, the reference for the floating state and the gradient.
    """

    def __init__(self, name, timestep, *, leaves=True):
        super().__init__(name, timestep, k=K, c=C)
        self._leaves = leaves

    def initial_state(self):
        s = {"x": jnp.asarray(0.0, jnp.float32)}
        if self._leaves:
            s["touching"] = jnp.asarray(False)
            s["n"] = jnp.asarray(0, jnp.int32)
        return s

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.asarray(0.0, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.asarray(0.0, jnp.float32))
        out = {"x": (p["k"] * u + p["c"]).astype(jnp.float32)}
        if self._leaves:
            out["touching"] = u > 0
            out["n"] = state["n"] + jnp.int32(1)
        return out


class Follower(SimulationNode):
    """``x = u``, starting at ``+1`` so ``a``'s first input is positive."""

    def initial_state(self):
        return {"x": jnp.asarray(1.0, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.asarray(0.0, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs.get("u", jnp.asarray(0.0, jnp.float32))}


def _pair(*, leaves=True, a_dt=1.0, **group):
    gm = GraphManager()
    gm.add_node(Contact("a", a_dt, leaves=leaves))
    gm.add_node(Follower("b", 1.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    # Jacobi contracts this pair at sqrt(0.5) per pass, rotating, and
    # over-relaxation at 0.8: a cap that lets every configuration converge.
    cfg = dict(max_iterations=120, tolerance=1e-6)
    cfg.update(group)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        # Knobs the configuration leaves inert dropped (they warn).
        gm.add_coupling_group(["a", "b"], **live_knobs(cfg))
    gm.compile()
    return gm


def _assert_consistent(gm, *, updates):
    """``touching`` is what ``a`` computes from the returned ``b``; ``n`` counted."""
    a, b = gm.get_node_state("a"), gm.get_node_state("b")
    assert float(b["x"]) == pytest.approx(X_STAR, rel=1e-4)
    assert bool(a["touching"]) is (float(b["x"]) > 0) is False
    assert a["touching"].dtype == jnp.bool_
    assert a["n"].dtype == jnp.int32 and int(a["n"]) == updates


_ACCELERATIONS = [
    dict(acceleration="none"),
    dict(acceleration="aitken"),
    dict(acceleration="fixed", relaxation=0.6),
    dict(acceleration="fixed", relaxation=1.1),
    dict(acceleration="iqn-ils"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2),
]


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("accel", _ACCELERATIONS, ids=lambda a: "-".join(map(str, a.values())))
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_an_iterate_dependent_flag_is_the_one_the_returned_state_computes(solver, accel, mode):
    gm = _pair(solver=solver, iteration_mode=mode, diagnostics=True, **accel)
    gm.step()
    assert gm.coupling_diagnostics()["a+b"]["converged"]
    _assert_consistent(gm, updates=1)


@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
@pytest.mark.parametrize("predictor", ["linear", "quadratic"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_predictor_starts_from_an_iterate_that_keeps_the_non_floating_fields(
        solver, predictor, norm):
    """The predictor extrapolates floating fields and merges them back.

    It used to *replace* each node's state with them, so the iterate the
    solve started from had no ``touching`` or ``n``; the mixed norm looked
    every field of the new iterate up in the old one and raised
    ``KeyError`` on the first residual.  The l2 and interface norms happen
    not to look, and are held here too.
    """
    knobs = dict(tolerance=1e-6) if norm == "l2" else dict(rtol=1e-5)
    gm = _pair(solver=solver, predictor=predictor, convergence_norm=norm,
               diagnostics=True, **knobs)
    for k in range(1, 5):
        gm.step()
        _assert_consistent(gm, updates=k)


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_subcycled_member_and_waveform_sweeps_keep_the_flag_consistent(solver):
    """``a`` takes two sub-steps per pass, and the solve runs twice per step."""
    gm = _pair(solver=solver, a_dt=0.5, subcycling=True, boundary_interpolation="constant",
               waveform_iterations=2, diagnostics=True)
    for k in range(1, 3):
        gm.step()
        _assert_consistent(gm, updates=2 * k)


def test_the_flag_agrees_between_the_solvers_and_the_floats_with_the_leafless_graph():
    """fori == ift on the flag; the leaves move no floating value."""
    with_leaves = {s: _pair(solver=s) for s in ("ift", "fori")}
    without = _pair(leaves=False)
    for _ in range(3):
        for gm in (*with_leaves.values(), without):
            gm.step()
    ift, fori = (with_leaves[s].get_node_state("a") for s in ("ift", "fori"))
    assert bool(ift["touching"]) == bool(fori["touching"])
    for name in ("a", "b"):
        assert np.asarray(with_leaves["ift"].get_node_state(name)["x"]).tobytes() == \
            np.asarray(without.get_node_state(name)["x"]).tobytes()


def test_the_ift_gradient_is_the_fixed_points_with_the_leaves_present():
    """The extra pass is outside the linearisation: ``dx*/dk``, ``dx*/dc`` exact.

    ``x* = c / (1 - k)``, so ``dx*/dk = c / (1 - k)**2`` and
    ``dx*/dc = 1 / (1 - k)``.  The pair is algebraic, so one step from any
    start lands on the fixed point and ``run_scan(2)`` returns it.
    """
    grads = {}
    for leaves in (True, False):
        gm = _pair(leaves=leaves, tolerance=1e-7)

        def loss(p, gm=gm):
            return gm.run_scan(2, params=p)["b"]["x"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
            g = jax.grad(loss)(gm.params)["nodes"]["a"]
            gm.reset_state()
        grads[leaves] = (float(g["k"]), float(g["c"]))
    assert grads[True] == pytest.approx((C / (1.0 - K) ** 2, 1.0 / (1.0 - K)), rel=1e-4)
    assert grads[True] == pytest.approx(grads[False], rel=1e-6)


def test_the_extra_pass_does_not_retrace_the_step():
    gm = _pair()
    for _ in range(4):
        gm.step()
    assert gm.trace_count == 1


def test_diagnostics_leave_the_flag_and_the_state_bit_identical():
    gms = {d: _pair(diagnostics=d, acceleration="aitken") for d in (False, True)}
    for _ in range(3):
        for gm in gms.values():
            gm.step()
    for name in ("a", "b"):
        for field, value in gms[False].get_node_state(name).items():
            assert np.asarray(value).tobytes() == \
                np.asarray(gms[True].get_node_state(name)[field]).tobytes(), (name, field)


# ---------------------------------------------------------------------------
# A non-floating field carried across a group-internal edge
# ---------------------------------------------------------------------------


class Gate(SimulationNode):
    """``x = k u + c`` and ``open = u < 0.5``: a flag another member reads."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep, k=K, c=C)

    def initial_state(self):
        return {"x": jnp.asarray(0.0, jnp.float32), "open": jnp.asarray(False)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.asarray(0.0, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.asarray(0.0, jnp.float32))
        return {"x": (p["k"] * u + p["c"]).astype(jnp.float32), "open": u < 0.5}


class Biased(Follower):
    """``x = u + 0.2 * open``: reads the flag across the edge."""

    def boundary_input_spec(self):
        return {**super().boundary_input_spec(),
                "open": BoundaryInputSpec(shape=(), dtype=jnp.bool_,
                                          default=jnp.asarray(False))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        bias = jnp.where(boundary_inputs["open"], 0.2, 0.0).astype(jnp.float32)
        return {"x": boundary_inputs["u"] + bias}


def _gated(solver, mode, reader_first, **group):
    """The only consistent fixed point has the gate open: ``x_b = 1/15``.

    ``b`` starts at ``+1``, so the gate is shut on the first pass; shut,
    the pair would settle at ``-1/15``, where the gate must be open.  The
    reader takes the flag from the iterate under Jacobi, and under
    Gauss-Seidel when it is scheduled first.
    """
    gm = GraphManager()
    nodes = [Gate("a", 1.0), Biased("b", 1.0)]
    for node in (nodes[::-1] if reader_first else nodes):
        gm.add_node(node)
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("a", "b", "open", "open")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["a", "b"], max_iterations=80, tolerance=1e-6, solver=solver,
                              iteration_mode=mode, diagnostics=True, **group)
    gm.compile()
    return gm


@pytest.mark.parametrize("mode, reader_first", [
    ("jacobi", False), ("gauss-seidel", True), ("gauss-seidel", False)])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_flag_read_across_an_internal_edge_iterates_with_the_solve(solver, mode, reader_first):
    """``"ift"`` held the flag at its first-pass value inside the loop.

    The reader then iterated the map with the gate shut, and ``"ift"``
    returned ``x_b = -1/15`` with ``converged=True`` -- the fixed point
    of a map the graph does not define -- where ``"fori"``, carrying the
    whole state, found ``1/15``.
    """
    gm = _gated(solver, mode, reader_first)
    gm.step()
    assert gm.coupling_diagnostics()["a+b"]["converged"]
    assert float(gm.get_node_state("b")["x"]) == pytest.approx(1.0 / 15.0, rel=1e-4)
    assert float(gm.get_node_state("a")["x"]) == pytest.approx(-2.0 / 15.0, rel=1e-4)
    assert bool(gm.get_node_state("a")["open"]) is True


def test_the_ift_gradient_through_a_flag_edge_is_the_open_branchs():
    """``x_b* = (c + 0.2) / (1 - k)`` on the open branch: ``dx_b*/dc = 1 / (1 - k)``."""
    gm = _gated("ift", "jacobi", False)

    def loss(p):
        return gm.run_scan(2, params=p)["b"]["x"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
        g = jax.grad(loss)(gm.params)["nodes"]["a"]
    assert float(g["c"]) == pytest.approx(1.0 / (1.0 - K), rel=1e-4)
    assert float(g["k"]) == pytest.approx((C + 0.2) / (1.0 - K) ** 2, rel=1e-4)


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_predictor_keeps_an_edge_carried_flag_in_the_iterate_it_starts_from(solver):
    """The predictor's start keeps every non-floating field, not only for the norms.

    Under Jacobi the reader takes the flag from the iterate the solve
    starts from.  The predictor used to rebuild that iterate from its
    floating fields alone, so the flag was missing and the first pass
    raised ``KeyError: 'open'``; the norms checking a field is floating
    before reading the old iterate do not help here.
    """
    gm = _gated(solver, "jacobi", False, predictor="linear")
    for _ in range(4):
        gm.step()
        assert gm.coupling_diagnostics()["a+b"]["converged"]
        assert float(gm.get_node_state("b")["x"]) == pytest.approx(1.0 / 15.0, rel=1e-4)
        assert bool(gm.get_node_state("a")["open"]) is True
