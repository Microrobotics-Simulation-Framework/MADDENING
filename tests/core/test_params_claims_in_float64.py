"""SYS-101 to SYS-105 and SYS-109 under ``jax_enable_x64``.

``ParamSpec``'s transforms and checks are stated for any floating value:
a leaf on its bound has derivative one into its range, ``log`` keeps a
value strictly above its bound, ``logit`` strictly inside its interval,
``constrain`` always lands inside, and ``check`` refuses anything outside
-- "however small", which in float64 reaches the subnormals near
``5e-324``.  The float32 versions are in ``test_param_spec_edge_cases.py``,
``test_params_spec.py``, ``test_params_bounded_identity_derivative.py``
and ``test_sysid_claims_edges.py``; these hold the same claims at float64's
own edges, and at float32's for a float32 leaf in an x64 process.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec, check_bounds, constrain, unconstrain
from maddening.nodes.spring import SpringDamperNode

F64 = np.float64
TINY64 = float(np.finfo(np.float64).tiny)
#: The smallest float64 subnormal.
SUB64 = float(np.nextafter(F64(0.0), F64(1.0)))


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


SPECS = {
    "nodes": {"s": {
        "stiffness": ParamSpec(bounds=(0.0, None), transform="log"),
        "damping": ParamSpec(bounds=(0.0, None)),
        "elasticity": ParamSpec(bounds=(0.0, 1.0), transform="logit"),
        "offset": ParamSpec(),
    }},
    "mappings": {},
}


def _params(k, c, e, off, dtype=jnp.float64):
    f = lambda v: jnp.asarray(v, dtype)  # noqa: E731
    return {"nodes": {"s": {"stiffness": f(k), "damping": f(c), "elasticity": f(e),
                            "offset": f(off)}}, "mappings": {}}


@pytest.mark.parametrize("bounds, u, expected", [
    ((0.0, 2.0), 0.0, 1.0), ((0.0, 2.0), 2.0, 1.0), ((0.0, 2.0), 1.0, 1.0),
    ((0.0, 2.0), -0.5, 0.0), ((0.0, None), 0.0, 1.0), ((None, 3.0), 3.0, 1.0),
    # A bound float32 cannot hold: the float64 leaf on it is on it.
    ((0.1, 0.2), 0.1, 1.0), ((0.1, 0.2), float(np.nextafter(F64(0.1), F64(0.0))), 0.0),
])
def test_the_clip_derivative_at_a_bound_is_one_in_float64(bounds, u, expected):
    """SYS-101: on a bound, derivative 1 into the range, in both modes; off the
    bounds the clip is ``jnp.clip`` bit for bit; the leaf stays float64."""
    with _x64():
        spec = ParamSpec(bounds=bounds)
        x = jnp.asarray(u, jnp.float64)
        assert spec.to_constrained(x).dtype == jnp.float64
        assert float(jax.grad(spec.to_constrained)(x)) == expected
        _, tangent = jax.jvp(spec.to_constrained, (x,), (jnp.float64(1.0),))
        assert float(tangent) == expected
        lo, hi = bounds
        clipped = jnp.clip(x, -jnp.inf if lo is None else lo, jnp.inf if hi is None else hi)
        assert float(spec.to_constrained(x)) == float(clipped)


@pytest.mark.parametrize("bad", [0.0, -TINY64, -SUB64, -1.0])
def test_a_log_spec_refuses_a_value_at_or_below_its_bound_in_float64(bad):
    """SYS-102: ``log`` is strictly above ``lo`` (0 when None) and ``check``
    refuses anything else, float64's subnormals included; a value it can
    unconstrain is accepted and comes back finite."""
    with _x64():
        spec = ParamSpec(transform="log")
        with pytest.raises(ValueError, match="below bound 0.0"):
            spec.check(np.float64(bad), name="k")
        with pytest.raises(ValueError, match="below bound 0.0"):
            check_bounds({"k": jnp.asarray(bad, jnp.float64)}, {"k": spec})
        for good in (2.0, TINY64):
            spec.check(np.float64(good), name="k")
            assert np.isfinite(float(spec.to_unconstrained(jnp.asarray(good, jnp.float64))))
        # A non-zero bound: the next float64 above it is inside, the bound is not.
        lo = 1e3
        bounded = ParamSpec(bounds=(lo, None), transform="log")
        with pytest.raises(ValueError, match="below bound"):
            bounded.check(np.float64(lo), name="k")
        bounded.check(np.nextafter(F64(lo), F64(np.inf)), name="k")
        # Large negative coordinates stay strictly inside and re-unconstrain.
        for u in (-800.0, -100.0, -30.0, 0.0, 50.0):
            for b in (0.0, lo, -lo):
                s = ParamSpec(bounds=(b, None), transform="log")
                p = s.to_constrained(jnp.asarray(u, jnp.float64))
                assert p.dtype == jnp.float64
                s.check(p)
                assert np.isfinite(float(s.to_unconstrained(p)))


@pytest.mark.parametrize("lo, width_ulps", [(-1e3, 8), (0.0, 1e6), (1.0, 16), (5.0, 1e12)])
def test_logit_constrain_is_strictly_inside_in_float64(lo, width_ulps):
    """SYS-103: ``lo < p < hi`` for any coordinate, an interval a few float64
    ulps wide included (one float32 could not tell apart from a point)."""
    with _x64():
        hi = float(F64(lo) + width_ulps * np.spacing(F64(max(abs(lo), 1.0))))
        assert hi > lo
        spec = ParamSpec(bounds=(lo, hi), transform="logit")
        for u in (-1e4, -40.0, -1.0, 0.0, 1.0, 40.0, 1e4):
            p = spec.to_constrained(jnp.asarray(u, jnp.float64))
            assert p.dtype == jnp.float64
            assert lo < float(p) < hi, (u, float(p), lo, hi)
            spec.check(p)
            assert np.isfinite(float(spec.to_unconstrained(p)))


def test_constrain_lands_inside_bounds_for_any_coordinate_in_float64():
    """SYS-104: ``constrain`` is always inside, ``unconstrain`` inverts it inside
    the bounds, and both are jittable and differentiable, in float64."""
    with _x64():
        rng = np.random.default_rng(1729)
        for u in [(-1e4,) * 4, (1e4,) * 4, (-745.0, -745.0, -745.0, 0.0)] + \
                [tuple(rng.uniform(-60.0, 60.0, 4)) for _ in range(40)]:
            p = constrain(_params(*u), SPECS)
            s = p["nodes"]["s"]
            assert all(v.dtype == jnp.float64 for v in s.values())
            assert float(s["stiffness"]) > 0.0 and float(s["damping"]) >= 0.0
            assert 0.0 < float(s["elasticity"]) < 1.0
            check_bounds(p, SPECS)
        inside = _params(3.0, 0.5, 0.25, -2.0)
        back = constrain(unconstrain(inside, SPECS), SPECS)
        for k, v in inside["nodes"]["s"].items():
            assert float(back["nodes"]["s"][k]) == pytest.approx(float(v), rel=1e-12)
        f = jax.jit(lambda q: jax.tree.reduce(jnp.add, constrain(q, SPECS)))
        g = jax.grad(f)(_params(0.3, 0.2, -0.1, 0.0))
        assert all(np.isfinite(float(v)) for v in g["nodes"]["s"].values())


def test_check_params_names_the_leaf_out_of_range_in_float64():
    """SYS-105: the first leaf out of range is named, a non-finite value is
    refused even without bounds."""
    with _x64():
        with pytest.raises(ValueError, match=r"\['nodes'\]\['s'\]\['damping'\].*below"):
            check_bounds(_params(10.0, -1e-300, 0.5, 0.0), SPECS)
        with pytest.raises(ValueError, match="elasticity"):
            check_bounds(_params(10.0, 0.0, 1.0, 0.0), SPECS)
        with pytest.raises(ValueError):
            ParamSpec().check(np.float64(np.nan), name="offset")
        check_bounds(_params(10.0, 0.0, 0.5, 0.0), SPECS)


def _spring_graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", timestep=0.01, stiffness=10.0, damping=0.5, mass=1.0))
    gm.compile()
    return gm


def test_a_value_below_a_zero_bound_by_a_float64_subnormal_is_refused():
    """SYS-109: ``check`` and ``gm.check_params`` refuse ``-5e-324`` below a zero
    bound, which a comparison that flushed subnormals would read as zero."""
    with _x64():
        with pytest.raises(ValueError, match="below bound"):
            ParamSpec(bounds=(0.0, None)).check(np.float64(-SUB64), name="damping")
        gm = _spring_graph()
        p = jax.tree.map(lambda x: x, gm.params)
        assert p["nodes"]["s"]["damping"].dtype == jnp.float64
        p["nodes"]["s"]["damping"] = jnp.asarray(-SUB64, jnp.float64)
        with pytest.raises(ValueError, match=r"\['damping'\].*below bound 0.0"):
            gm.check_params(p)


def test_a_float32_value_below_a_zero_bound_by_a_subnormal_is_refused_under_x64():
    """SYS-109 for a float32 leaf in an x64 process: ``-1e-40`` is still refused."""
    with _x64():
        with pytest.raises(ValueError, match="below bound"):
            ParamSpec(bounds=(0.0, None)).check(np.float32(-1e-40), name="damping")
        gm = _spring_graph()
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["s"]["damping"] = jnp.asarray(-1e-40, jnp.float32)
        with pytest.raises(ValueError, match=r"\['damping'\].*below bound 0.0"):
            gm.check_params(p)


def _checkpoint_with(gm, path, member, value):
    """*gm*'s checkpoint at *path* with one member replaced by *value*."""
    gm.save_state(str(path))
    with np.load(path) as data:
        members = {k: data[k] for k in data.files}
    members[member] = np.asarray(value)
    np.savez(path, **members)
    return path


def test_load_state_into_a_float32_field_of_an_x64_graph_refuses_what_float32_cannot_hold(tmp_path):
    """SYS-123 with mixed dtypes: the built-in spring keeps float32 state under
    x64, and a float64 checkpoint value it cannot hold -- ``1e39``, ``-1e-50`` --
    is refused, leaving the graph as it was; ``1e-40`` loads with its sign."""
    with _x64():
        gm = _spring_graph()
        assert np.asarray(gm.get_node_state("s")["position"]).dtype == np.float32
        before = np.asarray(gm.get_node_state("s")["position"]).copy()
        for value in (1e39, -1e-50):
            path = _checkpoint_with(gm, tmp_path / "c.npz", "s/position", np.float64(value))
            with pytest.raises(ValueError, match="float32 cannot hold"):
                gm.load_state(str(path))
            assert np.array_equal(np.asarray(gm.get_node_state("s")["position"]), before)
        gm.load_state(str(_checkpoint_with(gm, tmp_path / "c.npz", "s/position",
                                           np.float64(-1e-40))))
        got = np.asarray(gm.get_node_state("s")["position"])
        assert got.dtype == np.float32 and got == np.float32(-1e-40) and np.signbit(got)


def test_load_state_into_a_float64_field_loads_what_float32_could_not(tmp_path):
    """SYS-123 at float64: a float64 field holds ``1e39`` and ``-1e-50``, so they
    load bit for bit; a value already non-finite loads as it was."""
    from tests.core.test_coupling_claims_in_every_domain import CONFIGS, build

    with _x64():
        gm = build(CONFIGS["f64"], "plain", diagnostics=False)
        assert np.asarray(gm.get_node_state("a")["x"]).dtype == np.float64
        for value in (1e39, -1e-50, float("inf")):
            gm.load_state(str(_checkpoint_with(gm, tmp_path / "c.npz", "a/x", np.float64(value))))
            got = np.asarray(gm.get_node_state("a")["x"])
            assert got.dtype == np.float64 and got.tobytes() == np.float64(value).tobytes()
