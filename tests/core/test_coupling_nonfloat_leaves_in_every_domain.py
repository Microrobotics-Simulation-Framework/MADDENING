"""Non-floating leaves in a coupling group, differentiated through, in every domain.

CPL-145 of ``docs/validation/coupling_claims.yaml``: with an integer or
boolean leaf in a coupling group -- and a typed PRNG key, which travels as
its uint32 data -- reverse and forward mode through ``run_scan`` over the
coupled step work, and give the fixed point's derivative.  Verified in
float32 on single-rate graphs; here the pair ``a: x = k u + c`` (with a
contact flag ``touching = u > 0`` read from the iterate, a step counter
``n`` and a typed key folded every update) and ``b: x = u`` runs in more
domains: float64 and a float32 member beside a float64 one under x64, a
multi-rate graph (a clock at half the group's step), a sub-cycled member
(``a`` at half the step, sub-stepped twice per pass), a quadratic
predictor, a checkpoint restart (the counter and the flag; a typed key
cannot be saved, MADD-ANO-168), and ``jax.vmap`` of the gradient of a
``lax.scan`` of the raw step over three parameter sets.

The pair is algebraic, so every solve lands on ``x* = c / (1 - k)`` from
any start: ``dx*/dk = c / (1 - k)**2`` and ``dx*/dc = 1 / (1 - k)``, the
oracle, in float64.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import coupling_domains as cd

K, C = -0.5, -0.1
STEPS = 4


class Contact(SimulationNode):
    """``x = k u + c``, ``touching = u > 0``, ``n += 1`` and the key folded, per update."""

    def __init__(self, name, timestep, dtype, *, key=True):
        super().__init__(name, timestep, k=jnp.asarray(K, dtype), c=jnp.asarray(C, dtype))
        self._dtype, self._key = dtype, key

    def initial_state(self):
        s = {"x": jnp.zeros((), self._dtype), "touching": jnp.asarray(False),
             "n": jnp.asarray(0, jnp.int32)}
        if self._key:
            s["key"] = jax.random.key(0)
        return s

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=self._dtype,
                                       default=jnp.zeros((), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = jnp.asarray(boundary_inputs["u"]).astype(self._dtype)
        out = {"x": (p["k"] * u + p["c"]).astype(self._dtype), "touching": u > 0,
               "n": state["n"] + jnp.int32(1)}
        if self._key:
            out["key"] = jax.random.fold_in(state["key"], 1)
        return out


class Follower(SimulationNode):
    """``x = u``, starting at +1 so ``a``'s first input is positive."""

    def __init__(self, name, timestep, dtype):
        super().__init__(name, timestep)
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.ones((), self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=self._dtype,
                                       default=jnp.zeros((), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.asarray(boundary_inputs["u"]).astype(self._dtype)}


def _pair(domain, *, key=True):
    gm = GraphManager()
    da, db = domain.dtypes
    gm.add_node(Contact("a", cd.DT / 2 if domain.subcycled else cd.DT, da, key=key))
    gm.add_node(Follower("b", cd.DT, db))
    if da == db:
        gm.add_edge("b", "a", "x", "u")
        gm.add_edge("a", "b", "x", "u")
    else:
        gm.add_edge("b", "a", "x", "u", transform=lambda v: v.astype(da))
        gm.add_edge("a", "b", "x", "u", transform=lambda v: v.astype(db))
    if domain.multirate:
        gm.add_node(cd.Ticker("tick", cd.DT / 2))
    gm.add_coupling_group(["a", "b"], **cd.group_kwargs(domain, max_iterations=120,
                                                         tolerance=_tol(domain)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate notice
        gm.compile()
    return gm


def _exact():
    return {"k": C / (1.0 - K) ** 2, "c": 1.0 / (1.0 - K)}


def _tolerance(domain) -> float:
    """Relative: the IFT rule is linearised at the returned iterate (CPL-140),
    within ``tolerance * amp`` of the fixed point (``amp = 1 / (1 - k)``), and
    rounds at the coarsest member's resolution."""
    return max(64 * float(cd.finfo(domain.coarsest).eps), 8 * _tol(domain) / (1.0 - K))


def _tol(domain) -> float:
    return 1e-12 if domain.x64 and domain.dtype_a == jnp.float64 else 1e-7


def _scan_loss(gm, steps):
    def loss(p):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
            return gm.run_scan(steps, params=p)["b"]["x"]
    return loss


def _start(gm, restart_from):
    """The run's start, set *outside* the transform (resetting inside a
    differentiated loss fails on a predictor group: MADD-ANO-169)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
        gm.reset_state()
        if restart_from is not None:
            gm.load_state(restart_from)


def _derivatives(gm, steps, *, restart_from=None):
    """``(reverse, forward)`` derivatives of ``b.x`` after *steps* in ``k`` and ``c``."""
    loss = _scan_loss(gm, steps)
    p = gm.params
    _start(gm, restart_from)
    rev = jax.grad(loss)(p)["nodes"]["a"]
    tangent = jax.tree.map(jnp.zeros_like, p)
    tangent["nodes"]["a"]["k"] = jnp.ones_like(tangent["nodes"]["a"]["k"])
    _start(gm, restart_from)
    fwd = jax.jvp(loss, (p,), (tangent,))[1]
    cg_recover(gm)
    return ({"k": float(rev["k"]), "c": float(rev["c"])}, float(fwd))


def cg_recover(gm):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
        gm.coupling_diagnostics()


def _assert_exact(domain, rev, fwd, where):
    want, tol = _exact(), _tolerance(domain)
    for name in ("k", "c"):
        assert abs(rev[name] - want[name]) <= tol * abs(want[name]), (where, name, rev, want)
    assert abs(fwd - want["k"]) <= tol * abs(want["k"]), (where, fwd, want)


def _assert_leaves(domain, gm, updates):
    a = gm.get_node_state("a")
    assert a["touching"].dtype == jnp.bool_ and bool(a["touching"]) is False
    assert a["n"].dtype == jnp.int32 and int(a["n"]) == updates, (domain.label, int(a["n"]))
    assert a["x"].dtype == jnp.dtype(domain.dtype_a)


_LABELS = ("f64", "mixed_dtype", "multi_rate", "sub_cycled", "predictors_warm_starts")


@pytest.mark.parametrize("label", _LABELS)
def test_reverse_and_forward_mode_through_run_scan_with_non_float_leaves(label):
    """CPL-145 in *label*: ``jax.grad`` and ``jax.jvp`` through ``run_scan`` of the
    pair with a flag, a counter and a typed key in the group give the fixed
    point's derivative, and the leaves hold their closed form after the run."""
    d = cd.DOMAINS[label]
    steps = 2 * STEPS if d.multirate else STEPS
    with cd.entered(d):
        gm = _pair(d)
        rev, fwd = _derivatives(gm, steps)
        _assert_exact(d, rev, fwd, label)
        gm.reset_state()
        gm.run_scan(steps)
        fired = STEPS                           # the group's solves in the run
        _assert_leaves(d, gm, fired * (2 if d.subcycled else 1))
        assert "key" in gm.get_node_state("a"), "premise: the typed key is in the group"
        assert bool(gm._is_multirate) == d.multirate
        g = gm._committed_coupling_groups["a+b"]
        assert g.subcycling == d.subcycled and (g.predictor != "none") == d.predictor


def test_reverse_and_forward_mode_from_a_checkpoint_restart_with_non_float_leaves():
    """CPL-145 after a restart: two steps saved, reset and loaded, then the
    derivatives through ``run_scan`` from the loaded state are the fixed
    point's, and the counter carries on from the checkpoint.  Without the
    typed key, which ``save_state`` cannot write (MADD-ANO-168)."""
    d = cd.DOMAINS["checkpoint_restart"]
    gm = _pair(d, key=False)
    gm.run_scan(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        gm.save_state(path)
        rev, fwd = _derivatives(gm, STEPS, restart_from=path)
        gm.reset_state()
        gm.load_state(path)
    _assert_exact(d, rev, fwd, "checkpoint_restart")
    gm.run_scan(STEPS)
    _assert_leaves(d, gm, 2 + STEPS)


def test_vmap_of_the_gradient_through_a_scan_with_non_float_leaves():
    """CPL-145 under ``jax.vmap``: the gradient of a ``lax.scan`` of the raw step
    (the program ``run_scan`` runs) batched over three parameter sets, each
    member's derivative its own fixed point's, the leaves in the carry."""
    d = cd.DOMAINS["vmap"]
    gm = _pair(d)
    step = gm._raw_step_fn
    ext = gm._default_external_inputs()
    gm.reset_state()
    s0 = gm._state

    def loss(p):
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), s0, None, length=STEPS)
        return final["b"]["x"]

    ks = (-0.5, 0.25, 0.6)
    params = [jax.tree.map(lambda v: v, gm.params) for _ in ks]
    for p, k in zip(params, ks):
        p["nodes"]["a"]["k"] = jnp.asarray(k, jnp.float32)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *params)
    grads = jax.jit(jax.vmap(jax.grad(loss)))(batched)["nodes"]["a"]
    for i, k in enumerate(ks):
        want = {"k": C / (1.0 - k) ** 2, "c": 1.0 / (1.0 - k)}
        for name in ("k", "c"):
            got = float(grads[name][i])
            assert abs(got - want[name]) <= _tolerance(d) / (1.0 - k) * abs(want[name]), (
                k, name, got, want)


@pytest.mark.xfail(strict=True, raises=jax.errors.UnexpectedTracerError, reason=(
    "MADD-ANO-169: reset_state() inside a differentiated loss raises UnexpectedTracerError "
    "on a predictor group once an earlier transform has left tracers in the graph; "
    "deferred to 0.5.0"))
def test_a_loss_that_resets_a_predictor_group_differentiates_twice():
    """``loss(p) = (reset_state(); run_scan(n, params=p))`` -- each evaluation from
    the initial state -- differentiated twice.  Without a predictor both
    gradients are ``dx*/dc``; with one the second raises: the first transform
    left its tracers in the graph, ``reset_state`` puts nothing back inside a
    transform, and it sizes the predictor history's seed from the traced slot.
    Resetting outside the loss, or calling any entry point between the two,
    works (``_start`` above)."""
    d = cd.DOMAINS["predictors_warm_starts"]
    gm = _pair(d, key=False)

    def loss(p):
        gm.reset_state()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
            return gm.run_scan(STEPS, params=p)["b"]["x"]

    for _ in range(2):
        g = jax.grad(loss)(gm.params)["nodes"]["a"]["c"]
        assert abs(float(g) - _exact()["c"]) <= _tolerance(d) * _exact()["c"]
