"""After ``run_adaptive*`` the coupling report covers both kept half steps.

An accepted adaptive attempt keeps two solves per coupling group, its two
half steps, and ``strict_convergence`` checks both.  The ``_meta`` report
slots used to be whatever the second half step wrote, so
``coupling_diagnostics()`` said ``converged=True`` about an accepted step
whose first half step had exited at ``max_iterations`` unconverged -- the
step ``strict_convergence=True`` refuses.  The report is now folded
(``_fold_kept_half_step_reports``): ``iterations`` the larger half's count,
``total_iterations`` the sum where the group owns that slot, and the
per-solve keys from the half whose verdict is the step's.

The fixture is two relays ``a <-> b`` of gain 0.5 each way (Gauss-Seidel
rate 0.25) that ignore ``dt``, so each half step is the same fixed-point
problem from a different start: from zero the first runs out of its four
passes, and the second, starting from the first's result, converges.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
#: One accepted attempt of the whole interval, accepted on its error.
ADAPTIVE = dict(dt_initial=0.1, dt_min=0.1, dt_max=0.1, atol=1e3, rtol=1.0)


class _Relay(SimulationNode):
    def __init__(self, name, bias, dt=1.0):
        super().__init__(name, dt, bias=jnp.asarray(bias, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.float32(0.5) * boundary_inputs["u"] + self.params["bias"]}


def _graph(cap=4, strict=False, waveform=1):
    gm = GraphManager()
    gm.add_node(_Relay("a", [1.0, 2.0]))
    # Half the timestep when the group must sub-cycle (waveform sweeps run
    # only in a sub-cycling group); the relay ignores dt either way.
    gm.add_node(_Relay("b", [0.0, 1.0], dt=0.5 if waveform > 1 else 1.0))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    kw = dict(subcycling=True, waveform_iterations=waveform) if waveform > 1 else {}
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=1e-3,
                          strict_convergence=strict, **kw)
    gm.compile()
    return gm


def _run(entry, gm):
    if entry == "run_adaptive":
        gm.run_adaptive(0.1, **ADAPTIVE)
    else:
        gm.run_adaptive_scan(0.1, max_steps=1, **ADAPTIVE)
    return gm.coupling_diagnostics()[KEY]


@functools.lru_cache(maxsize=None)
def _halves(cap, waveform=1):
    """Each kept half step's own slots, through the jitted dt-parameterised step."""
    gm = _graph(cap, waveform=waveform)
    fn = jax.jit(gm._build_dt_step_fn(collect_strict=True))
    state, ext = gm._state, gm._resolve_external_inputs(None)
    out = []
    for _half in (1, 2):
        state, _verdicts = fn(state, ext, jnp.asarray(0.05), gm.params)
        meta = state["_meta"]
        out.append({k[len(f"coupling_{KEY}_"):]: np.asarray(v) for k, v in meta.items()
                    if k.startswith(f"coupling_{KEY}_")})
    return out


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan"])
def test_a_kept_unconverged_first_half_is_reported_unconverged(entry):
    """The finding: ``converged=True`` beside a first half stopped at the cap."""
    first, second = _halves(4)
    assert int(first["iterations"]) == 4 and int(second["iterations"]) < 4, (
        "fixture premise: the first half caps, the second converges")
    d = _run(entry, _graph(4))
    assert not d["converged"], dict(d)
    assert d["iterations"] == 4, dict(d)
    # The per-solve keys are the first half's, together.
    assert d["residual"] == pytest.approx(float(first["residual"]), rel=1e-6), dict(d)
    assert d["amplification"] == pytest.approx(float(first["amplification"]), rel=1e-6)


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan"])
def test_the_report_and_strict_convergence_give_one_verdict(entry):
    """Where the report says unconverged the strict check raises, and vice versa."""
    for cap in (4, 30):
        d = _run(entry, _graph(cap))
        if d["converged"]:
            _run(entry, _graph(cap, strict=True))
        else:
            with pytest.raises(Exception, match="without converging"):
                _run(entry, _graph(cap, strict=True))


def test_both_halves_converged_reports_the_second():
    """With both kept solves converged the report is the second half's, as before."""
    first, second = _halves(30)
    d = _run("run_adaptive", _graph(30))
    assert d["converged"], dict(d)
    assert d["iterations"] == max(int(first["iterations"]), int(second["iterations"]))
    assert d["residual"] == float(second["residual"]), dict(d)


def test_a_waveform_group_reports_the_sum_of_both_halves_passes():
    """``total_iterations`` sums the halves (and their sweeps) where the slot exists."""
    first, second = _halves(4, waveform=2)
    assert "total_iterations" in first, "fixture premise: the group owns the sum's slot"
    d = _run("run_adaptive", _graph(4, waveform=2))
    assert d["total_iterations"] == int(first["total_iterations"]) + int(
        second["total_iterations"]), dict(d)
    assert d["iterations"] == max(int(first["iterations"]), int(second["iterations"]))


def test_a_one_sweep_group_reports_the_larger_half_as_its_total():
    """No slot to hold a sum: ``total_iterations`` reads ``iterations`` (documented)."""
    d = _run("run_adaptive", _graph(4))
    assert d["total_iterations"] == d["iterations"] == 4, dict(d)


@pytest.mark.parametrize("entry", ["run_adaptive", "run_adaptive_scan"])
def test_the_fold_leaves_the_state_alone(entry):
    """Only report slots move: the user state is the second half step's, bit for bit."""
    gm = _graph(4)
    fn = jax.jit(gm._build_dt_step_fn(collect_strict=True))
    state, ext = gm._state, gm._resolve_external_inputs(None)
    for _half in (1, 2):
        state, _v = fn(state, ext, jnp.asarray(0.05), gm.params)
    _run(entry, gm)
    for nm in ("a", "b"):
        np.testing.assert_array_equal(np.asarray(gm.get_node_state(nm)["x"]),
                                      np.asarray(state[nm]["x"]))
