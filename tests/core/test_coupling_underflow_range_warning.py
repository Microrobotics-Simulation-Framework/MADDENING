"""A coupled group whose fields sit in the subnormal range says so, once.

Below ``finfo(dtype).tiny / finfo(dtype).eps`` (about ``9.9e-32`` in float32)
a change of one ulp of a field is smaller than the smallest normal number,
which XLA's CPU backend flushes to zero.  MADDENING's own coupling arithmetic
works in power-of-two frames and keeps its resolution down to ``tiny``; a
node's ``update`` does not, and below ``tiny`` the group's norm reads the field
as zero.  ``GraphManager`` checks each coupled group on the first step after
``compile()`` and warns once per group with the field, its magnitude and the
remedy.  An exactly zero field -- zero is not a small unit -- never warns.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.coupling._reports import _underflow_range_fields
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.warnings import PrecisionLimitWarning, UnderflowRangeWarning

F32 = jnp.finfo(jnp.float32)
THRESHOLD = float(F32.tiny) / float(F32.eps)        # 2**-103, about 9.9e-32


class _Relay(SimulationNode):
    """``x <- g * u + c``, plus a field ``z`` that stays exactly zero."""

    def __init__(self, name, g, c, dtype=jnp.float32):
        super().__init__(name, 1.0, c=jnp.asarray(c, dtype))
        self._g, self._dtype = g, dtype

    def initial_state(self):
        return {"x": jnp.zeros(1, self._dtype), "z": jnp.zeros(3, self._dtype),
                "k": jnp.zeros((), jnp.int32)}

    def state_fields(self):
        return ["x", "z", "k"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=self._dtype, description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        x = (self._g * boundary_inputs["u"] + p["c"]).astype(self._dtype)
        return {"x": x, "z": state["z"], "k": state["k"] + 1}


def _pair(c, dtype=jnp.float32):
    gm = GraphManager()
    gm.add_node(_Relay("a", 0.5, c, dtype))
    gm.add_node(_Relay("b", 1.0, 0.0, dtype))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-5)
    gm.compile()
    return gm


def _recorded(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if issubclass(w.category, UnderflowRangeWarning)]


def test_a_group_in_the_subnormal_range_warns_once_with_the_field_and_the_remedy():
    gm = _pair(1e-33)                     # fixed point 2e-33, below 9.9e-32
    first = _recorded(gm.step)
    assert len(first) == 1
    msg = str(first[0].message)
    assert "'a+b'" in msg and "a.x" in msg and "Rescale" in msg and "float32" in msg
    assert "1 more field(s)" in msg      # b.x is there too; z (zero) is not counted
    assert issubclass(UnderflowRangeWarning, PrecisionLimitWarning)
    # Once per group: later steps, and a recompile, do not repeat it.
    assert _recorded(lambda: [gm.step() for _ in range(3)]) == []
    gm.compile()
    assert _recorded(gm.step) == []


def test_an_exactly_zero_group_never_warns():
    gm = _pair(0.0)
    assert _recorded(lambda: [gm.step() for _ in range(3)]) == []
    assert float(jnp.max(jnp.abs(gm.get_node_state("a")["x"]))) == 0.0


@pytest.mark.parametrize("c", [1.0, 1e-20, 2.0 ** -100])
def test_a_group_above_the_threshold_does_not_warn(c):
    gm = _pair(c)                         # fixed point 2c, at or above 2**-99
    assert _recorded(gm.step) == []


def test_the_warning_waits_for_a_step_outside_a_transform():
    gm = _pair(1e-33)

    def loss(p):
        out = gm.run_scan(1, params=p)
        return jnp.sum(out["a"]["x"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jax.grad(loss)(gm.params)         # stores tracers: no check, still pending
    assert len(_recorded(gm.step)) == 1


def test_the_field_scan_lists_exactly_the_nonzero_finite_fields_below_tiny_over_eps():
    class _G:
        nodes = ("n",)

    def listed(**fields):
        state = {"n": {k: jnp.asarray(v[0], v[1]) for k, v in fields.items()}}
        return [(f, d.name if hasattr(d, "name") else str(d))
                for _n, f, _m, d in _underflow_range_fields([_G()], state).get("n", [])]

    f32_thr = THRESHOLD
    assert listed(a=([0.5 * f32_thr, 0.0], jnp.float32)) == [("a", "float32")]
    assert listed(a=([2.0 * f32_thr], jnp.float32)) == []
    assert listed(a=([0.0, 0.0], jnp.float32)) == []
    assert listed(a=([np.nan, 1e-35], jnp.float32), b=([np.inf], jnp.float32)) == []
    assert listed(a=([3], jnp.int32)) == []
    # A typed PRNG key cannot become a numpy array; it is skipped, not read.
    state = {"n": {"key": jax.random.key(0), "x": jnp.asarray([0.5 * f32_thr], jnp.float32)}}
    assert [f for _n, f, _m, _d in _underflow_range_fields([_G()], state)["n"]] == ["x"]
    # Each field at its own dtype's threshold: float16's is 0.0625.
    assert listed(h=([0.01], jnp.float16), g=([0.1], jnp.float16)) == [("h", "float16")]
    bf_thr = float(jnp.finfo(jnp.bfloat16).tiny) / float(jnp.finfo(jnp.bfloat16).eps)
    assert listed(b=([0.5 * bf_thr], jnp.bfloat16)) == [("b", "bfloat16")]
    assert listed(b=([4.0 * bf_thr], jnp.bfloat16)) == []
