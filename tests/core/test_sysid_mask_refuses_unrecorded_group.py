"""``windowed_loss(mask_unconverged=True)`` refuses a group it cannot read.

The mask reads each coupling group's residual slot in ``_meta``.
``solver="fori"`` with ``diagnostics=False`` writes none, and the mask used
to skip such a group (``if key in meta``), so an unconverged window was
never masked and nothing said so.  It now raises, and every configuration
that does record a verdict still masks.
"""

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.sysid import observations_from_history, windowed_loss


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


def _unconverged_pair(**kw):
    """Two passes cannot bring a rate-0.9 pair to 1e-8: every step is unconverged."""
    gm = GraphManager()
    gm.add_node(_Affine("a", 0.95, 1.0))
    gm.add_node(_Affine("b", 0.95, 0.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=2, tolerance=1e-8, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # solver="fori" is deprecated
        gm.compile()
    return gm


def _loss(gm, mask):
    init = {n: dict(gm.get_node_state(n)) for n in ("a", "b")}
    truth = {n: {f: jnp.asarray(v)[None] + 1.0 for f, v in init[n].items()}
             for n in ("a", "b")}
    obs = observations_from_history(init, truth)
    return float(windowed_loss(gm, gm.params, obs, obs_fn=lambda s: s["a"]["x"],
                               window=1, mask_unconverged=mask))


def test_a_fori_group_without_diagnostics_is_refused():
    gm = _unconverged_pair(solver="fori", diagnostics=False)
    with pytest.raises(ValueError, match=r"cannot mask coupling group \['a', 'b'\]"
                                         r".*solver='fori' with diagnostics=False"):
        _loss(gm, mask=True)


def test_the_same_group_is_fine_without_the_mask():
    gm = _unconverged_pair(solver="fori", diagnostics=False)
    assert _loss(gm, mask=False) > 0.0


@pytest.mark.parametrize("kw", [
    dict(solver="fori", diagnostics=True),
    dict(solver="ift", diagnostics=False),
    dict(solver="ift", diagnostics=True),
], ids=["fori-diagnostics", "ift", "ift-diagnostics"])
def test_every_group_that_records_a_verdict_is_masked(kw):
    gm = _unconverged_pair(**kw)
    assert _loss(gm, mask=False) > 0.0
    assert _loss(gm, mask=True) == 0.0
