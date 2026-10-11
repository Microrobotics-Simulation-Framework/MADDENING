"""A change of the units of ONE field of a coupling group, under each acceleration.

A coupling group's convergence norms divide each field's change by that
field's own magnitude, so the *measurement* of a change does not depend on
the units a field is written in.  The run that is measured is another
matter.  With one common factor on every field it is the same run under
every acceleration, bit for bit (``tests/property/test_differential_coupling_units.py``,
``tests/core/test_coupling_verdict_under_a_decimal_change_of_units.py``).
With a factor on one field alone -- a force rewritten in micronewtons
beside a displacement left in metres -- it is the same run only where the
step treats each field by itself:

* ``acceleration="none"`` and ``"fixed"``: bit-identical, pinned here;
* ``"aitken"``, ``"iqn-ils"`` and ``"iqn-imvj"``: their step is formed from
  sums over all the accelerated fields together (one power-of-two frame for
  the whole vector), so a field in much smaller units carries no weight in
  it and the iteration takes another path.  Open, MADD-ANO-274; documented
  on ``CouplingGroup.convergence_norm``.  The second test pins that the
  dependence is there, in the one form that does not depend on rounding:
  the pass counts are not the unscaled run's.  **When the accelerators
  scale each field by itself, that test must fail and is then replaced by
  the equality the first one asserts.**

The fixture is affine and exactly equivariant: every constant is rescaled
by exact powers of two, and the node code is the same program.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

N = 3
STEPS = 4
#: ``b`` written in units ``2**20`` (about a million) smaller; ``a`` left.
ONE_FIELD = (1.0, 2.0 ** 20)


class Affine(SimulationNode):
    """``x <- G u + D x_pre + c``."""

    def __init__(self, name, G, D, c, x0):
        super().__init__(name=name, timestep=1.0)
        self._G, self._D = jnp.asarray(G, jnp.float32), jnp.asarray(D, jnp.float32)
        self._c, self._x0 = jnp.asarray(c, jnp.float32), jnp.asarray(x0, jnp.float32)

    def initial_state(self):
        return {"x": self._x0}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=jnp.float32,
                                       default=jnp.zeros(N, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._G @ boundary_inputs["u"] + self._D @ state["x"] + self._c}


def _run(scale_a, scale_b, **group):
    """``[(passes, converged, residual, x_a / scale_a, x_b / scale_b)]`` over ``STEPS`` steps.

    A contraction of spectral radius 0.8 between two members of three
    entries each, with ``a`` in units ``scale_a`` and ``b`` in ``scale_b``
    (powers of two, so the division on the way out is exact).
    """
    rng = np.random.default_rng(3)
    g_ab, g_ba = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    radius = max(abs(np.linalg.eigvals(
        np.block([[np.zeros((N, N)), g_ab], [g_ba, np.zeros((N, N))]]))))
    g_ab, g_ba = g_ab * 0.8 / radius, g_ba * 0.8 / radius
    d_a, d_b = 0.3 * rng.standard_normal((N, N)), 0.3 * rng.standard_normal((N, N))
    c_a, c_b = rng.standard_normal(N), rng.standard_normal(N)
    gm = GraphManager()
    gm.add_node(Affine("a", g_ab * (scale_a / scale_b), d_a, c_a * scale_a,
                       0.1 * scale_a * np.ones(N)))
    gm.add_node(Affine("b", g_ba * (scale_b / scale_a), d_b, c_b * scale_b,
                       -0.2 * scale_b * np.ones(N)))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    with warnings.catch_warnings():
        # A field in small units beside one at order one is what this is about.
        warnings.simplefilter("ignore", UserWarning)
        gm.add_coupling_group(["a", "b"], iteration_mode="jacobi", max_iterations=25,
                              tolerance=1e-5, **group)
        gm.compile()
        out = []
        for _ in range(STEPS):
            gm.step()
            report = gm.coupling_diagnostics()["a+b"]
            out.append((int(report["iterations"]), bool(report["converged"]),
                        float(report["residual"]),
                        np.asarray(gm.get_node_state("a")["x"]) / np.float32(scale_a),
                        np.asarray(gm.get_node_state("b")["x"]) / np.float32(scale_b)))
    return out


@pytest.mark.parametrize("group", [
    {"acceleration": "none"}, {"acceleration": "fixed", "relaxation": 0.7}],
    ids=lambda g: g["acceleration"])
def test_without_a_coupled_step_one_fields_units_do_not_change_the_run(group):
    """``"none"`` and ``"fixed"`` step each field by itself: the same run, to the bit."""
    reference, rescaled = _run(1.0, 1.0, **group), _run(*ONE_FIELD, **group)
    for (i0, c0, r0, a0, b0), (i1, c1, r1, a1, b1) in zip(reference, rescaled):
        assert (i0, c0, r0) == (i1, c1, r1)
        np.testing.assert_array_equal(a0, a1)
        np.testing.assert_array_equal(b0, b1)


def test_under_iqn_one_field_in_other_units_is_another_run():
    """The open dependence (MADD-ANO-274), pinned by pass counts alone.

    With both fields at order one ``"iqn-ils"`` converges on every step.
    With ``b`` in units ``2**20`` smaller the same problem does not take
    those passes (measured: 25, 12, 16, 16 against 8, 8, 8, 8, the first
    step unconverged at the cap).  Only "not the same passes" is asserted:
    the counts themselves depend on the least-squares rounding.
    """
    reference = _run(1.0, 1.0, acceleration="iqn-ils")
    rescaled = _run(*ONE_FIELD, acceleration="iqn-ils")
    assert all(converged for _passes, converged, *_rest in reference)
    assert [step[0] for step in rescaled] != [step[0] for step in reference]
    assert sum(step[0] for step in rescaled) > sum(step[0] for step in reference)
