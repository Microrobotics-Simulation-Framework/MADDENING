"""The report of a member of a ``jax.vmap`` of the step, under Jacobi.

A batch of states stepped through ``jax.vmap`` of the compiled step is
another compiled program than the step alone.  What a member's report
shares with the same member stepped alone, and what it does not
(measured on thirty graphs with a mapped internal edge, four members and
two steps each, and on plain pairs and rings; CPU, jax 0.11.0):

* the verdict, the pass count and every flag are equal, and the state to
  a few roundings;
* ``gradient_relative_error_bound`` is the bound's own arithmetic run by
  the batched program, and is good to its own float resolution: equal to
  4e-6 on plain edges under either schedule; on a Jacobi group with a
  mapped edge 0.03% to 0.8% apart above the float floor and up to a
  factor of two apart where the report is at its float floor
  (``precision_limited``).  The member's number does not depend on the
  other members of the batch, on their order or on its size (a batch of
  one gives it), so nothing of a member is updated after its own loop
  has ended.

The two existing tests of the bound under ``jax.vmap`` run Gauss-Seidel
groups (``test_coupling_gradient_bound_through_a_vmapped_step.py`` per
push, ``test_coupling_gradient_bound_under_vmap.py`` slow).  Held here,
under Jacobi: the equal parts on a pair with a mapped edge; and on the
curved pair whose gradient is closed form, the bound of each member of a
batch against the true error, beside the member's alone.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.sparse_mapping import sparse_nearest_neighbor_mapping
from maddening.core.graph_manager import GraphManager
from tests.core.test_coupling_gradient_error_bound import (
    _CURVED, _PHI, _Curved, _Relay, _analytic_curved)
from tests.core.test_the_dead_band_of_a_ring_of_three import NA, NF, Relay, _positions

FLAGS = ("converged", "iterations", "spectral_usable", "gradient_bound_usable",
         "precision_limited")


def _report(gm: GraphManager, state) -> dict:
    """The report ``coupling_diagnostics()`` gives of *state* (a member's
    state put in the graph's place, and the graph's own put back)."""
    saved = gm._state  # noqa: SLF001
    gm._state = state  # noqa: SLF001
    try:
        (report,) = (dict(r) for r in gm.coupling_diagnostics().values())
        return report
    finally:
        gm._state = saved  # noqa: SLF001


def _batch_and_alone(gm: GraphManager, members: list) -> list:
    """``[(member of the batch, the member alone), ...]`` after one step."""
    ext, params = gm._default_external_inputs(), gm.params  # noqa: SLF001
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *members)
    batch = jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, None)))(  # noqa: SLF001
        stacked, ext, params)
    step = jax.jit(gm._raw_step_fn)  # noqa: SLF001
    return [(jax.tree.map(lambda leaf, i=i: leaf[i], batch), step(member, ext, params))
            for i, member in enumerate(members)]


# Two compiles of a group of 39 entries with its diagnostics (the batch and
# the step alone): 10 s on an eight-core slice.
# Per push: tests/core/test_coupling_report_of_a_batch_member_under_jacobi.py::test_the_gradient_bound_of_a_batch_member_under_jacobi_holds_as_the_members_alone
@pytest.mark.slow
def test_the_verdict_the_passes_and_the_flags_of_a_batch_member_are_the_members_alone():
    """A Jacobi pair with a sparse mapped edge each way (3 displacements
    and 36 forces, loop gain 0.5, ``atol = 0``), three states a step
    apart: in the batch each member has the verdict, the pass count and
    the flags it has alone, and its state to four roundings."""
    gm = GraphManager()
    gm.add_node(Relay("A", NA, 1e9))
    gm.add_node(Relay("C", NF, 0.5e-9, loaded=True))
    gm.add_external_input("C", "load", shape=(NF,), dtype=jnp.dtype("float32"))
    gm.add_edge("A", "C", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NA), _positions(NF)))
    gm.add_edge("C", "A", "x", "u",
                mapping=sparse_nearest_neighbor_mapping(_positions(NF), _positions(NA)))
    gm.add_coupling_group(["A", "C"], convergence_norm="mixed", rtol=1e-4, atol=0.0,
                          iteration_mode="jacobi", max_iterations=200, solver="ift",
                          diagnostics=True)
    gm.compile()
    load = jnp.asarray(1e-8 * (1.0 + 0.3 * np.arange(NF) / NF), jnp.float32)
    members = []
    for _ in range(3):
        members.append(gm._state)  # noqa: SLF001
        gm.step(external_inputs={"C": {"load": load}})
    passes = set()
    for i, (in_batch, alone) in enumerate(_batch_and_alone(gm, members)):
        got, want = _report(gm, in_batch), _report(gm, alone)
        assert {k: got[k] for k in FLAGS} == {k: want[k] for k in FLAGS}, (i, got, want)
        for name in ("A", "C"):
            a = np.asarray(in_batch[name]["x"], np.float64)
            b = np.asarray(alone[name]["x"], np.float64)
            assert np.max(np.abs(a - b)) <= 4 * np.finfo(np.float32).eps * np.max(np.abs(b)), (
                i, name)
        passes.add(want["iterations"])
    # The batch is not three copies of one loop: the members stop on
    # different passes.
    assert len(passes) > 1, passes


def test_the_gradient_bound_of_a_batch_member_under_jacobi_holds_as_the_members_alone():
    """The curved pair ``u <- a + g u**2`` under Jacobi, capped at eight
    passes from four starts.  Each member's bound in the batch is within
    1e-4 of the member's alone (measured 0 to 4e-6: the two programs
    round the bound's arithmetic differently, and on plain edges far
    from the float floor that is all the difference there is), its flag
    is set in both, and both bound the true relative error of the
    gradient at the returned iterate (measured 1.3 times it)."""
    kind, starts = "square", (0.0, 0.5, 0.25, 1.0)
    a, g = _CURVED[kind]
    phi, dphi = _PHI[kind]
    _u_star, exact = _analytic_curved(kind)
    gm = GraphManager()
    gm.add_node(_Curved("a", kind, a, g))
    gm.add_node(_Relay("b"))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=8, tolerance=1e-7,
                          iteration_mode="jacobi")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the cap is reached on purpose
        gm.compile()
    state0 = gm._state  # noqa: SLF001
    members = [{**state0, "a": {**state0["a"], "x": jnp.float32(start)},
                "b": {**state0["b"], "x": jnp.float32(start)}} for start in starts]
    bounds = set()
    for i, (in_batch, alone) in enumerate(_batch_and_alone(gm, members)):
        got, want = _report(gm, in_batch), _report(gm, alone)
        assert {k: got[k] for k in FLAGS} == {k: want[k] for k in FLAGS}, (i, got, want)
        assert want["gradient_bound_usable"] and want["iterations"] == 8, (i, want)
        # The tangent of the returned state: phi is evaluated at the
        # relay's value.
        u = float(alone["b"]["x"])
        tangent = {"a": 1.0 / (1.0 - g * dphi(u)), "g": phi(u) / (1.0 - g * dphi(u))}
        true = max(abs(tangent[c] - exact[c]) / abs(tangent[c]) for c in ("a", "g"))
        assert true > 1e-2, (i, true)          # the forward stopped visibly short
        for which, report in (("in the batch", got), ("alone", want)):
            bound = float(report["gradient_relative_error_bound"])
            assert true <= bound <= 4.0 * true, (i, which, bound, true)
        np.testing.assert_allclose(got["gradient_relative_error_bound"],
                                   want["gradient_relative_error_bound"], rtol=1e-4)
        bounds.add(float(got["gradient_relative_error_bound"]))
    assert len(bounds) == len(starts), "one member's bound broadcast to the batch"
