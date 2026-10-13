"""sysid claims in the numeric domains their rows' conditions cover.

``docs/validation/sysid_fmu_claims.yaml`` gives each row a ``domains``
matrix.  The rows' own tests use a float32 spring; these run the same
claims on the coupled pair of
``tests/core/test_coupling_claims_in_every_domain.py`` (two memoryless
members, each reading the other) built in each domain:

* SYS-004, "``gm._state`` is not modified", by a value and a gradient of
  ``windowed_loss``: under x64 (float64, and a float32 member beside a
  float64 one), inside a user's ``jax.jit``, under ``jax.vmap`` over
  parameters, on a multi-rate graph, on a sub-cycled group and on a group
  carrying a predictor and IQN-IMVJ warm starts;
* SYS-006, the teacher-forced loss is zero at the generating parameters
  and positive elsewhere, and SYS-013, its gradient is finite and
  informative: under ``jax.vmap`` over a batch of parameters, each member
  as its own call;
* SYS-039, ``fim_core`` fails closed on a NaN matrix inside ``jax.jit``.

The loss reads member ``a``'s ``x``; ``g`` and ``b0`` are the parameters.
Tolerances are stated where they are taken.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.sysid import fim_core, observations_from_history, windowed_loss
from tests.core import test_coupling_claims_in_every_domain as battery

N, WINDOW = 12, 4


def _obs_fn(h):
    return h["a"]["x"]


def _recorded(cfg_name: str, kind: str = "plain"):
    """A graph of *cfg_name*'s domain and a record it made at its parameters."""
    cfg = battery.CONFIGS[cfg_name]
    gm = battery.build(cfg, kind, diagnostics=False)
    p = battery.params_for(gm, "moving")
    gm.params = p
    init = {name: gm.get_node_state(name) for name in gm.node_names}
    steps = 2 * N if cfg.multirate else N
    _, hist = gm.run_scan_with_history(steps)
    obs = observations_from_history(init, hist)
    return cfg, gm, obs


def _snapshot(gm):
    return [np.array(x) for x in jax.tree.leaves(gm._state)]


def _untouched(gm, obs, loss_and_grad):
    before = _snapshot(gm)
    loss_and_grad()
    after = _snapshot(gm)
    assert len(before) == len(after)
    assert all(a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
               for a, b in zip(before, after))


def _perturbed(gm):
    p = jax.tree.map(lambda v: v, gm.params)
    p["nodes"]["a"]["g"] = (p["nodes"]["a"]["g"] * 1.25).astype(p["nodes"]["a"]["g"].dtype)
    return p


def _loss(gm, obs, window=WINDOW, **kw):
    return lambda p: windowed_loss(gm, p, obs, obs_fn=_obs_fn, window=window, **kw)


#: (domain config, group kind, sample_every and start_step for the loss)
_SYS_004 = {
    "f64": ("f64", "plain", {}),
    "mixed_dtype": ("mixed_dtype", "plain", {}),
    "multi_rate": ("multi_rate", "plain", {"sample_every": 2, "start_step": 0}),
    "sub_cycled": ("sub_cycled", "plain", {}),
    "predictors_warm_starts": ("predictors_warm_starts", "main", {}),
}


def _sys_004(domain):
    name, kind, kw = _SYS_004[domain]
    cfg = battery.CONFIGS[name]
    with battery._x64(cfg.x64):
        cfg, gm, obs = _recorded(name, kind)
        if "sample_every" in kw:
            obs = jax.tree.map(lambda v: v[::kw["sample_every"]], obs)
        loss = _loss(gm, obs, **kw)
        p = _perturbed(gm)
        _untouched(gm, obs, lambda: (loss(gm.params), jax.grad(loss)(p)))
        assert float(loss(p)) > 0.0      # the loss read the graph: not a no-op


@pytest.mark.parametrize("row", ["SYS-004"])
def test_the_claim_holds_in_float64(row):
    """Under x64, both members float64."""
    _sys_004("f64")


@pytest.mark.parametrize("row", ["SYS-004"])
def test_the_claim_holds_with_a_float32_node_under_x64(row):
    """Under x64, member ``a`` float32 and member ``b`` float64."""
    _sys_004("mixed_dtype")


@pytest.mark.parametrize("row", ["SYS-004"])
def test_the_claim_holds_on_a_multirate_graph(row):
    """A node outside the group at half its timestep: the group fires every other step."""
    _sys_004("multi_rate")


@pytest.mark.parametrize("row", ["SYS-004"])
def test_the_claim_holds_on_a_subcycled_group(row):
    """``subcycling=True``, member ``b`` sub-stepped twice per pass."""
    _sys_004("sub_cycled")


@pytest.mark.parametrize("row", ["SYS-004"])
def test_the_claim_holds_with_a_predictor_and_a_warm_start(row):
    """``predictor="quadratic"``, ``acceleration="iqn-imvj"``, ``jacobian_reuse=2``."""
    _sys_004("predictors_warm_starts")


@pytest.fixture(scope="module")
def float32_pair():
    return _recorded("vmap")[1:]


def _jit_untouched(gm, obs):
    loss = _loss(gm, obs)
    p = _perturbed(gm)
    _untouched(gm, obs, lambda: (jax.jit(loss)(gm.params), jax.jit(jax.grad(loss))(p)))
    assert float(jax.jit(loss)(p)) > 0.0


def _fim_core_nan_under_jit():
    p = {"a": jnp.float32(1.0), "b": jnp.float32(2.0)}
    core = jax.jit(lambda q: fim_core(lambda r: jnp.stack([r["a"] * jnp.nan, r["b"]]), q))(p)
    assert np.isinf(float(core.cond)) and not bool(core.finite)
    assert int(core.rank) == 0 and np.isinf(np.asarray(core.crb)).all()
    ok = jax.jit(lambda q: fim_core(lambda r: jnp.stack([2.0 * r["a"], r["b"] - r["a"],
                                                         r["b"]]), q))(p)
    assert bool(ok.finite) and not bool(ok.precision_limited)
    assert float(ok.deciding_ratio) == 0.0


@pytest.mark.parametrize("row", ["SYS-004", "SYS-039"])
def test_the_claim_holds_under_jit(row, float32_pair):
    """SYS-004: a user's ``jax.jit`` of the loss and of its gradient leaves the
    graph alone.  SYS-039: ``fim_core`` jitted fails closed on a NaN matrix,
    and reads ``deciding_ratio == 0.0`` on a resolved one."""
    if row == "SYS-004":
        _jit_untouched(*float32_pair)
    else:
        _fim_core_nan_under_jit()


def _batched_params(gm):
    """The generating parameters, then two perturbations, stacked on a new axis."""
    ps = [gm.params, _perturbed(gm)]
    q = jax.tree.map(lambda v: v, gm.params)
    q["nodes"]["b"]["b0"] = (q["nodes"]["b"]["b0"] + 0.25).astype(q["nodes"]["b"]["b0"].dtype)
    ps.append(q)
    return ps, jax.tree.map(lambda *xs: jnp.stack(xs), *ps)


@pytest.mark.parametrize("row", ["SYS-004", "SYS-006", "SYS-013"])
def test_the_claim_holds_under_vmap(row, float32_pair):
    """``jax.vmap`` of the loss and its gradient over three parameter sets.

    SYS-004: the graph is left alone.  SYS-006: the generating member reads
    exactly 0.0 (each window replays the record's own arithmetic) and the
    others are positive and equal their unbatched losses to 1e-6 relative
    (a batched program may round its reductions differently).  SYS-013: each
    member's gradient is finite, the perturbed ones non-zero, and equal to
    the unbatched gradient to 1e-5 relative.
    """
    gm, obs = float32_pair
    loss = _loss(gm, obs)
    singles, batch = _batched_params(gm)
    if row == "SYS-004":
        _untouched(gm, obs, lambda: (jax.vmap(loss)(batch), jax.vmap(jax.grad(loss))(batch)))
        return
    if row == "SYS-006":
        losses = np.asarray(jax.vmap(loss)(batch), np.float64)
        assert losses[0] == 0.0, losses
        for i in (1, 2):
            one = float(loss(singles[i]))
            assert losses[i] > 0.0 and losses[i] == pytest.approx(one, rel=1e-6), (i, losses[i], one)
        return
    grads = jax.vmap(jax.grad(loss))(batch)
    for i in (1, 2):
        one = jax.grad(loss)(singles[i])
        for node in ("a", "b"):
            for key in ("g", "b0"):
                got = float(grads["nodes"][node][key][i])
                want = float(one["nodes"][node][key])
                assert np.isfinite(got)
                assert got == pytest.approx(want, rel=1e-5, abs=1e-12), (i, node, key, got, want)
        assert any(float(grads["nodes"]["a"][k][i]) != 0.0 for k in ("g", "b0"))
