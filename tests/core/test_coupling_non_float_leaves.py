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
import warnings

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


@pytest.mark.parametrize("kw", [
    dict(), dict(solver="fori"), dict(predictor="linear"),
    dict(acceleration="iqn-ils"),
    # ``solver="fori"`` relaxed *every* field under these two, and the
    # integer leaves came back rounded through float32 (0xdeadbeef as
    # 0xdeadbf00, 2**24 + 1 as 2**24): MADD-ANO-056.
    dict(solver="fori", acceleration="aitken"),
    dict(solver="fori", acceleration="fixed", relaxation=0.7),
    dict(acceleration="aitken"),
    dict(acceleration="fixed", relaxation=0.7),
    dict(solver="fori", acceleration="iqn-ils"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2),
], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()) or "default")
def test_wide_integer_leaves_survive_a_coupled_step(kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori'", DeprecationWarning)
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


# ---------------------------------------------------------------------------
# An integer field in the quasi-Newton selection (MADD-ANO-056)
# ---------------------------------------------------------------------------


class Counter(SimulationNode):
    """A float field driven by the edge, and a step counter beside it."""

    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32), "n": jnp.array(0, jnp.int32)}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"]), "n": s["n"] + 1}


class FluxCounter(Counter):
    """A ``Counter`` whose coupling output is a boundary flux, not a state field."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"pull": 2.0 * state["y"]}


def _counter_pair(solver="ift", **kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(Counter("k", 0.01))
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori'", DeprecationWarning)
        gm.add_coupling_group(["s", "k"], max_iterations=8, tolerance=1e-6,
                              solver=solver, **kw)
    return gm


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_an_integer_field_named_in_accelerated_fields_is_left_out(solver, acceleration):
    """The counter is dropped from the quasi-Newton problem, not relaxed.

    Under ``solver="ift"`` the secant matrices were sized for the integer
    entry while the iterate's index map left it out, and the first step
    died in a broadcasting error inside the traced loop; under ``"fori"``
    the counter was relaxed through float32 with the rest.  Naming it
    must give the step that naming only the floating field gives.
    """
    named = _counter_pair(solver, acceleration=acceleration,
                          accelerated_fields={"k": ("y", "n"), "s": ("position",)})
    floating = _counter_pair(solver, acceleration=acceleration,
                             accelerated_fields={"k": ("y",), "s": ("position",)})
    a, b = named.run_scan(3), floating.run_scan(3)
    assert int(a["k"]["n"]) == 3 and a["k"]["n"].dtype == jnp.int32
    for node in ("s", "k"):
        for field, value in b[node].items():
            np.testing.assert_array_equal(np.asarray(a[node][field]), np.asarray(value),
                                          err_msg=f"{node}.{field}")


@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_a_flux_edge_from_a_node_with_an_integer_field_accelerates_its_floats(acceleration):
    """Auto-detection maps a flux edge to its producer's whole state: floats only.

    A flux is a function of the producer's state, so the producer's state
    stands in for it in the quasi-Newton problem -- every *floating*
    field of it.  The counter used to be counted into the secant
    matrices, and the step died in a broadcasting error.
    """
    from maddening.core.coupling.helpers import add_flux_coupling

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(FluxCounter("k", 0.01))
    gm.add_edge("s", "k", "position", "x")
    add_flux_coupling(gm, "k", "s", "pull", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=8, tolerance=1e-6,
                          acceleration=acceleration)
    out = gm.run_scan(3)
    assert int(out["k"]["n"]) == 3
    assert gm.coupling_diagnostics()["k+s"]["converged"] is True


def test_accelerated_fields_naming_no_floating_field_is_refused_at_compile():
    """Nothing would be left to accelerate: refused by name, before the trace."""
    gm = _counter_pair(acceleration="iqn-ils", accelerated_fields={"k": ("n",)})
    with pytest.raises(ValueError, match="names no floating-point field"):
        gm.compile()
