"""The profiler's one-iteration variant warns about nothing the caller wrote.

``profile_graph(measure_coupling=True)`` rebuilds every coupling group with
``max_iterations=1`` through ``dataclasses.replace``, which re-runs
``CouplingGroup.__post_init__``: every accelerated group then warned that
its acceleration was "ignored under max_iterations=1" -- a cap the caller
never set -- and the warning was attributed to ``dataclasses.py``, because
the stack walk stopped at the first frame outside the package, which was
the standard library's.  Fatal under ``-W error``.
"""

import dataclasses
import warnings

import jax.numpy as jnp
import pytest

from maddening.core.coupling.group import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import profile_graph


class _Affine(SimulationNode):
    def __init__(self, name, gain, bias):
        super().__init__(name=name, timestep=0.01, gain=gain, bias=bias)

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.float32(0.0))
        return {"x": self.params["gain"] * u + self.params["bias"]}


def _pair(**kw):
    gm = GraphManager()
    gm.add_node(_Affine("a", 0.5, 1.0))
    gm.add_node(_Affine("b", 0.5, 0.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("kw", [
    dict(acceleration="aitken"),
    dict(acceleration="fixed", relaxation=0.7),
    dict(acceleration="iqn-ils"),
], ids=["aitken", "fixed", "iqn-ils"])
def test_measuring_coupling_overhead_warns_about_nothing(kw):
    gm = _pair(**kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = profile_graph(gm, n_steps=2, n_warmup=1, counts=False,
                               measure_coupling=True)
    assert report.one_iteration_step_ms > 0.0
    inert = [str(w.message) for w in caught if "max_iterations=1" in str(w.message)]
    assert not inert, inert
    # The caller's own groups are back, untouched.
    assert gm._coupling_groups[0].max_iterations == 20


def test_a_warning_from_a_replaced_group_names_the_line_that_replaced_it():
    """Standard-library frames are skipped: the warning lands here."""
    group = CouplingGroup(nodes=frozenset({"a", "b"}), acceleration="aitken")
    with pytest.warns(UserWarning, match="max_iterations=1") as record:
        dataclasses.replace(group, max_iterations=1)
    assert [w.filename for w in record] == [__file__], [
        (w.filename, w.lineno) for w in record]
