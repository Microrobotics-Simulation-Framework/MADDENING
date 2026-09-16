"""Integer, boolean and PRNG-key state leaves survive coupled steps
bit-exactly, inside and outside the coupling group, for the IFT and fori
solvers, with a predictor and with IQN acceleration; gradients still flow
through the group; the predictor extrapolates floating fields only.

The IFT solver carries such leaves through ``closure_convert`` as float32
images (16-bit limbs for wide integers, uint32 data for typed keys), so
these tests pin the exactness of that encoding end to end.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.spring import SpringDamperNode



class KeyHolder(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32),
                "key": jnp.array([0xDEADBEEF, 0x12345678], jnp.uint32),
                "big": jnp.array(2**24 + 1, jnp.int32),
                "neg": jnp.array(-(2**31), jnp.int32),
                "flag": jnp.array(True)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {**s, "y": s["y"] + dt * (x - s["y"])}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


class TypedKeyHolder(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32), "key": jax.random.key(0)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"]), "key": s["key"]}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


def _coupled(holder_cls, extra=None, **kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(holder_cls("k", 0.01))
    if extra is not None:
        gm.add_node(extra)
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=5, tolerance=1e-8, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("kw", [dict(), dict(solver="fori"), dict(predictor="linear"),
                                dict(acceleration="iqn-ils")])
def test_wide_integer_leaves_survive_a_coupled_step(kw):
    gm = _coupled(KeyHolder, **kw)
    out = gm.run_scan(3)
    np.testing.assert_array_equal(np.asarray(out["k"]["key"]),
                                  np.array([0xDEADBEEF, 0x12345678], np.uint32))
    assert int(out["k"]["big"]) == 2**24 + 1
    assert int(out["k"]["neg"]) == -(2**31)
    assert bool(out["k"]["flag"]) is True
    assert out["k"]["key"].dtype == jnp.uint32 and out["k"]["big"].dtype == jnp.int32


def test_typed_prng_key_inside_and_outside_group():
    gm = _coupled(TypedKeyHolder)
    out = gm.run_scan(2)
    assert jax.dtypes.issubdtype(out["k"]["key"].dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(out["k"]["key"])),
                                  np.asarray(jax.random.key_data(jax.random.key(0))))
    gm2 = _coupled(KeyHolder, extra=TypedKeyHolder("free", 0.01))
    out2 = gm2.run_scan(2)
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(out2["free"]["key"])),
                                  np.asarray(jax.random.key_data(jax.random.key(0))))


def test_gradient_still_flows_with_wide_integer_leaves_in_the_group():
    gm = _coupled(KeyHolder)

    def loss(p):
        return jnp.sum(gm.run_scan(4, params=p)["k"]["y"] ** 2)

    g = jax.grad(loss)(gm.params)["nodes"]["s"]["stiffness"]
    assert np.isfinite(float(g)) and float(g) != 0.0




def test_predictor_leaves_integer_fields_alone():
    class Counter(SimulationNode):
        def initial_state(self):
            return {"y": jnp.array(0.0, jnp.float32), "n": jnp.array(0, jnp.int32),
                    "flag": jnp.array(False)}

        def update(self, s, bi, dt, *, params=None):
            x = bi.get("x", jnp.array(0.0, jnp.float32))
            return {"y": s["y"] + dt * (x - s["y"]), "n": s["n"] + 1, "flag": ~s["flag"]}

        def boundary_input_spec(self):
            return {"x": BoundaryInputSpec(shape=(), description="drive")}

    gm = _coupled(Counter, predictor="quadratic")
    out = gm.run_scan(6)
    assert int(out["k"]["n"]) == 6 and out["k"]["n"].dtype == jnp.int32
    assert bool(out["k"]["flag"]) is False and out["k"]["flag"].dtype == jnp.bool_
