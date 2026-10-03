"""The cases of the float32 / float64 differential, and the float64 worker.

JAX's ``jax_enable_x64`` is process-global, and flipping it inside a test
process changes the dtype of every array built afterwards, the compile
cache's keys included.  So the float64 side of
``test_differential_precision.py`` runs here, in a separate process started
with ``JAX_ENABLE_X64=1``, and the float32 side in the pytest process.  Each
case is one function, run unchanged in both: it builds its graph in the
process's default float dtype and returns plain JSON -- the verdicts the
oracle compares exactly, the answers it compares within a derived
tolerance, and the quantities that tolerance is derived from.

``serve()`` is the worker: one JSON request per line on stdin
(``{"case": name, "args": {...}}``), one JSON reply per line on stdout
(``{"ok": true, "result": ...}`` or ``{"ok": false, "error": ...}``).
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
import traceback
import warnings
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes import SpringDamperNode
from maddening.sysid import (
    fim,
    fit,
    fit_lm,
    fit_multiple_shooting,
    observations_from_history,
    windowed_loss,
)


def float_dtype():
    """The process's default float dtype: float64 under x64, else float32."""
    return jnp.asarray(1.0).dtype


def _f(x) -> float:
    return float(np.asarray(x))


def _floats(tree) -> list[float]:
    return [float(v) for v in np.asarray(tree, np.float64).ravel()]


@contextlib.contextmanager
def _quiet():
    """Warnings off: a guard declining a hold, a multi-rate assumption, a
    compile-time pattern warning -- the verdicts carry what the oracle
    compares."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# A closed-form problem: one monotone, nonlinear block of residuals per
# coordinate (``tests/property/test_sysid_truth_recovery.py``), here in the
# process's own dtype.
# ---------------------------------------------------------------------------

T_GRID = np.linspace(0.1, 2.0, 8)
KEYS = ("stiffness", "damping", "rest_length")
SPECS = {
    "clip": ((0.0, 5.0), None),
    "log": ((0.0, None), "log"),
    "logit": ((0.5, 2.0), "logit"),
}
RANGES = {"clip": (0.0, 5.0), "log": (0.05, 20.0), "logit": (0.5, 2.0)}
#: The most floating operations between a parameter and one residual entry
#: of ``_blocks`` (``rest * t + 0.2 * rest**2 * t``, then the data
#: subtracted): the forward error of an entry is at most this many units of
#: ``eps`` times its magnitude, to first order.
BLOCK_OPS = 6


def _blocks_graph(kinds, leaf_dtype=None):
    gm = GraphManager()
    kw = dict(stiffness=30.0, damping=2.0, mass=1.0, rest_length=1.0, initial_position=0.2)
    if leaf_dtype is not None:
        kw = {k: np.asarray(v, leaf_dtype) for k, v in kw.items()}
    gm.add_node(SpringDamperNode("s", 0.01, **kw))
    for key, kind in zip(KEYS, kinds):
        bounds, transform = SPECS[kind]
        gm.set_param_spec("s", key, ParamSpec(bounds=bounds, transform=transform))
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.compile()
    return gm


def _blocks(p):
    q = p["nodes"]["s"]
    k, c, rest = (q[key] for key in KEYS)
    t = jnp.asarray(T_GRID, jnp.result_type(k))
    return jnp.concatenate([jnp.log1p(c * t), jnp.sqrt(k + 0.1) * t,
                            rest * t + 0.2 * rest ** 2 * t])


def _value(kind, fraction):
    lo, hi = RANGES[kind]
    return lo + fraction * (hi - lo)


def _point(gm, kinds, fractions):
    p = jax.tree.map(lambda x: x, gm.params)
    for key, kind, fraction in zip(KEYS, kinds, fractions):
        leaf = p["nodes"]["s"][key]
        p["nodes"]["s"][key] = jnp.asarray(_value(kind, fraction), jnp.result_type(leaf))
    return p


def _mask(gm):
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for key in KEYS:
        mask["nodes"]["s"][key] = True
    return mask


def _coords(p) -> list[float]:
    return [_f(p["nodes"]["s"][key]) for key in KEYS]


def _sensitivity(truth) -> dict:
    """``||J^+||_2`` of the residual in the physical coordinates at the
    truth, ``||g(truth)||_2`` and the leaf's ``eps``: what the oracle's
    tolerance is made of."""
    def g(x):
        p = jax.tree.map(lambda v: v, truth)
        for i, key in enumerate(KEYS):
            p["nodes"]["s"][key] = x[i]
        return _blocks(p)
    x0 = jnp.stack([truth["nodes"]["s"][key] for key in KEYS])
    J = np.asarray(jax.jacfwd(g)(x0), np.float64)
    s = np.linalg.svd(J, compute_uv=False)
    return {"pinv_norm": float(1.0 / s[-1]), "g_norm": float(np.linalg.norm(
        np.asarray(g(x0), np.float64))), "eps": float(jnp.finfo(x0.dtype).eps),
        "n_residuals": int(J.shape[0])}


def blocks_fit_lm(kinds, truth_at, start_at, n_iter=60, leaf_dtype=None):
    with _quiet():
        gm = _blocks_graph(kinds, leaf_dtype)
        truth = _point(gm, kinds, truth_at)
        data = _blocks(truth)
        res = fit_lm(gm, lambda p: _blocks(p) - data, params=_point(gm, kinds, start_at),
                     mask=_mask(gm), n_iter=n_iter)
        return {"verdicts": {"converged": bool(res.converged),
                             "excited_rank": res.excited_rank,
                             "hold_declined": bool(res.hold_declined)},
                "answers": {"params": _coords(res.params), "truth": _coords(truth)},
                "leaf_dtypes": sorted({str(jnp.result_type(v))
                                       for v in res.params["nodes"]["s"].values()}),
                **_sensitivity(truth)}


def blocks_fit(kinds, truth_at, start_at, n_iter=800, lr=0.05, rel_tol=1e-8):
    with _quiet():
        gm = _blocks_graph(kinds)
        truth = _point(gm, kinds, truth_at)
        data = _blocks(truth)
        loss = jax.jit(lambda p: 0.5 * jnp.sum((_blocks(p) - data) ** 2))
        start = _point(gm, kinds, start_at)
        tol = rel_tol * _f(loss(start))
        res = fit(gm, loss, params=start, mask=_mask(gm), n_iter=n_iter, lr=lr, tol=tol)
        return {"verdicts": {"converged": bool(res.converged),
                             "excited_rank": res.excited_rank,
                             "hold_declined": bool(res.hold_declined)},
                "answers": {"params": _coords(res.params), "truth": _coords(truth)},
                "tol": tol, **_sensitivity(truth)}


def blocks_fim(kinds, truth_at, noise_std=0.01):
    with _quiet():
        gm = _blocks_graph(kinds)
        truth = _point(gm, kinds, truth_at)
        data = _blocks(truth)
        report = fim(lambda p: _blocks(p) - data, truth, mask=_mask(gm),
                     noise_std=noise_std)
        return {"verdicts": {"rank": int(report.rank),
                             "param_names": list(report.param_names)},
                "answers": {"eigvals": _floats(report.eigvals), "cond": float(report.cond)},
                **_sensitivity(truth)}


# ---------------------------------------------------------------------------
# The spring graph: windowed loss, multiple shooting, the scale degeneracy
# ---------------------------------------------------------------------------

def _spring(stiffness=30.0, damping=2.0, mass=1.0, leaf_dtype=None):
    gm = GraphManager()
    kw = dict(stiffness=stiffness, damping=damping, mass=mass)
    if leaf_dtype is not None:
        kw = {k: np.asarray(v, leaf_dtype) for k, v in kw.items()}
    gm.add_node(SpringDamperNode("spring", 0.01, initial_position=0.5, **kw))
    gm.compile()
    return gm


def _widen(gm):
    """Every float state leaf in the process's float dtype.  A node seeds
    its state in float32 and its first update under x64 promotes it, which
    a graph scan refuses (MADD-ANO-017); a run started from widened state
    scans."""
    for n in gm.node_names:
        st = gm.get_node_state(n)
        gm.set_node_state(n, {k: (jnp.asarray(v, float_dtype())
                                  if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating) else v)
                              for k, v in st.items()})


def _record(gm, n_steps):
    _widen(gm)
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps)
    gm.reset_state()
    _widen(gm)
    return observations_from_history(init, hist)


def _position(h):
    return h["spring"]["position"]


def spring_windowed_loss(scale, window=8, n_steps=40):
    """The teacher-forced loss at the truth and at the stiffness times
    ``scale``, and what the tolerance on the latter is made of."""
    with _quiet():
        gm = _spring()
        obs = _record(gm, n_steps)
        at_truth = _f(windowed_loss(gm, gm.params, obs, obs_fn=_position, window=window))
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["spring"]["stiffness"] = p["nodes"]["spring"]["stiffness"] * scale
        off = _f(windowed_loss(gm, p, obs, obs_fn=_position, window=window))
        return {"verdicts": {"zero_at_truth": at_truth == 0.0},
                "answers": {"at_truth": at_truth, "off_truth": off},
                "max_abs_state": float(max(np.max(np.abs(np.asarray(v)))
                                           for v in jax.tree.leaves(obs))),
                "n_compared": int(n_steps), "window": window,
                "eps": float(jnp.finfo(float_dtype()).eps)}


def spring_fit_multiple_shooting(truth_damping, start_damping, n_steps=40, window=10,
                                 tol=1e-7):
    """Damping alone, by multiple shooting from the noiseless record, and
    ``||J_w^+||``: the sensitivity of the windowed simulation (each window
    from its recorded start) to the damping at the truth."""
    with _quiet():
        source = _spring(damping=truth_damping)
        obs = _record(source, n_steps)
        gm = _spring(damping=start_damping)
        _widen(gm)
        mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
        mask["nodes"]["spring"]["damping"] = True
        # The window states barely move (lr_states), so the answer is the
        # damping's, not a trade against freed window starts.
        res, _ = fit_multiple_shooting(gm, obs, obs_fn=_position, window=window, mask=mask,
                                       n_iter=300, lr=0.1, lr_states=1e-9, tol=tol)
        step = source._build_step_fn()                                # noqa: SLF001
        ext = source._resolve_external_inputs(None)                   # noqa: SLF001

        def windowed(c):
            p = jax.tree.map(lambda x: x, source.params)
            p["nodes"]["spring"]["damping"] = c
            out = []
            for w in range(n_steps // window):
                s = {"spring": {k: v[w * window] for k, v in obs["spring"].items()}}

                def body(s, _):
                    s = step(s, ext, p)
                    return s, s["spring"]["position"]
                _, xs = jax.lax.scan(body, s, None, length=window)
                out.append(xs)
            return jnp.concatenate(out)
        J = np.asarray(jax.jacfwd(windowed)(jnp.asarray(truth_damping, float_dtype())),
                       np.float64)
        return {"verdicts": {"converged": bool(res.converged)},
                "answers": {"damping": _f(res.params["nodes"]["spring"]["damping"])},
                "best_loss": float(res.best_loss), "tol": tol,
                "pinv_norm": float(1.0 / max(float(np.linalg.norm(J)), 1e-300)),
                "eps": float(jnp.finfo(float_dtype()).eps)}


def spring_scale_guard(fitter="fit_lm", n_iter=10, lr=0.2, noise=0.0, leaf_dtype=None):
    """H1's case: the spring's ``(k, c, m)`` scale is undetermined by its
    data (log on all three); the guard holds it at the start, whose
    geometric mean is returned relative to the start's."""
    with _quiet():
        gm = _spring(leaf_dtype=leaf_dtype)
        for key in ("stiffness", "damping", "mass"):
            gm.set_param_spec("spring", key, ParamSpec(bounds=(0.0, None), transform="log"))
        gm.set_param_spec("spring", "rest_length", ParamSpec(trainable=False))
        st = gm.get_node_state("spring")
        init = {"spring": {k: jnp.asarray(v, float_dtype())[None] for k, v in st.items()}}

        def positions(p):
            _, hh = gm.run_sweep(120, init, return_history=True, params=p)
            return hh["spring"]["position"][0]
        clean = positions(gm.params)
        data = clean + jnp.asarray(np.random.default_rng(1).normal(0, noise, 120),
                                   clean.dtype) if noise else clean
        start = jax.tree.map(lambda x: x, gm.params)
        for key, v in (("stiffness", 45.0), ("damping", 3.0), ("mass", 1.3)):
            start["nodes"]["spring"][key] = jnp.asarray(
                v, jnp.result_type(start["nodes"]["spring"][key]))

        def scale(p):
            s = p["nodes"]["spring"]
            return math.exp(np.mean([math.log(_f(s[k])) for k in ("stiffness", "damping",
                                                                   "mass")]))
        if fitter == "fit_lm":
            res = fit_lm(gm, lambda p: positions(p) - data, params=start, n_iter=n_iter)
        else:
            res = fit(gm, jax.jit(lambda p: jnp.sum((positions(p) - data) ** 2)),
                      params=start, n_iter=n_iter, lr=lr)
        return {"verdicts": {"excited_rank": res.excited_rank,
                             "hold_declined": bool(res.hold_declined)},
                "answers": {"scale_drift": scale(res.params) / scale(start) - 1.0}}


def fit_lm_at_its_floor(truth_damping):
    """L1's case: a noiseless spring fit whose damping is weakly
    determined; ``converged`` at the fitter's own floor."""
    with _quiet():
        def build(c):
            gm = _spring(damping=c)
            gm.set_param_spec("spring", "mass", ParamSpec(trainable=False, bounds=(0.0, None),
                                                          transform="log"))
            return gm
        gt = build(truth_damping)
        init = {"spring": {k: jnp.asarray(v, float_dtype())[None]
                           for k, v in gt.get_node_state("spring").items()}}
        data = gt.run_sweep(150, init, return_history=True)[1]["spring"]["position"][0]
        gm = build(2.0)
        resid = jax.jit(lambda p: gm.run_sweep(150, init, return_history=True, params=p)[1][
            "spring"]["position"][0] - data)
        start = jax.tree.map(lambda x: x, gm.params)
        start["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, float_dtype())
        start["nodes"]["spring"]["damping"] = jnp.asarray(4.0, float_dtype())
        res = fit_lm(gm, resid, params=start, n_iter=100)
        return {"verdicts": {"converged": bool(res.converged)},
                "answers": {"stiffness": _f(res.params["nodes"]["spring"]["stiffness"]),
                            "n_iter": int(res.n_iter)}}


# ---------------------------------------------------------------------------
# A coupled pair: coupling_diagnostics() verdicts and an IFT gradient
# ---------------------------------------------------------------------------

#: Floating multiply-adds between an entry of the coupled state and its
#: next value in one pass of the pair's map (a spring's update: the spring
#: force, the damping, the velocity and the position): the forward error of
#: a pass is at most ``2 * PASS_TERMS * eps`` relative per entry (the
#: coupling kit's ``rounding_bound``), amplified by the resolvent
#: ``1 / (1 - rho)`` in a converged solve or its tangent.
PASS_TERMS = 8


def _pair(ka, kb, tolerance):
    """Two springs anchored on each other: stiff enough that the pair's
    coupled map contracts at ``rho = (ka * kb)**0.5 * dt**2`` (0.06-0.48
    over the drawn range), so the solve takes several passes and stops on
    its estimate, well above the float32 floor -- away from the
    precision-limited regime, where float32 and float64 legitimately read
    differently."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=ka, rest_length=0.5,
                                 initial_position=0.5, damping=0.5))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=kb, rest_length=0.3,
                                 initial_position=0.2, damping=0.1))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=40, tolerance=tolerance, diagnostics=True)
    gm.compile()
    return gm


def _pair_state(gm) -> list[float]:
    return [float(v) for n in ("a", "b") for k in ("position", "velocity")
            for v in np.asarray(gm.get_node_state(n)[k], np.float64).ravel()]


def pair_coupling(ka, kb, tolerance):
    """One coupled step from the compiled state: the verdicts, the state,
    and the bounds on its distance to the step's exact fixed point."""
    with _quiet():
        gm = _pair(ka, kb, tolerance)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        meta = gm._state["_meta"]                                      # noqa: SLF001
        return {"verdicts": {k: bool(d[k]) for k in ("converged", "ratio_usable",
                                                     "spectral_usable", "precision_limited")},
                "answers": {"state": _pair_state(gm), "rho_spectral": float(d["rho_spectral"])},
                "spectral_error_bound": float(d["spectral_error_bound"]),
                "spectral_residual": _f(meta["coupling_a+b_spectral_residual"]),
                "residual": float(d["residual"]),
                "eps": float(jnp.finfo(jnp.asarray(gm.get_node_state("a")["position"]).dtype).eps)}


def pair_ift_gradient(ka, kb, tolerance):
    """``d (x_a + x_b) / d k_a`` through one coupled step from the compiled
    state, under the IFT solver, and the step's own bound on that
    gradient's relative error (``gradient_relative_error_bound``)."""
    with _quiet():
        gm = _pair(ka, kb, tolerance)
        state0 = {n: dict(f) for n, f in gm._state.items()}           # noqa: SLF001
        step = gm._build_step_fn()                                    # noqa: SLF001
        ext = gm._resolve_external_inputs(None)                       # noqa: SLF001

        def total(k):
            p = jax.tree.map(lambda x: x, gm.params)
            p["nodes"]["a"]["stiffness"] = k
            s = step(state0, ext, p)
            return s["a"]["position"] + s["b"]["position"]
        k0 = jnp.asarray(gm.params["nodes"]["a"]["stiffness"], float_dtype())
        g = _f(jax.grad(total)(k0))
        gm.step()                                  # the same step, for its report
        d = gm.coupling_diagnostics()["a+b"]
        return {"verdicts": {"converged": bool(d["converged"]),
                             "gradient_bound_usable": bool(d["gradient_bound_usable"])},
                "answers": {"gradient": g},
                "gradient_relative_error_bound": float(d["gradient_relative_error_bound"]),
                "rho_spectral": float(d["rho_spectral"]),
                "eps": float(jnp.finfo(k0.dtype).eps)}


def environment():
    return {"x64": bool(jax.config.jax_enable_x64), "dtype": str(float_dtype()),
            "jax": jax.__version__}


CASES: dict[str, Callable[..., Any]] = {f.__name__: f for f in (
    blocks_fit_lm, blocks_fit, blocks_fim, spring_windowed_loss,
    spring_fit_multiple_shooting, spring_scale_guard, fit_lm_at_its_floor,
    pair_coupling, pair_ift_gradient, environment)}


def run_case(name: str, args: dict) -> Any:
    return CASES[name](**args)


def serve() -> None:
    """The float64 worker's loop (see the module docstring)."""
    out = sys.stdout
    sys.stdout = sys.stderr            # anything a case prints stays off the channel
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            reply = {"ok": True, "result": run_case(request["case"], request.get("args", {}))}
        except Exception as exc:                                      # noqa: BLE001
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()}
        out.write(json.dumps(reply, allow_nan=True) + "\n")
        out.flush()


if __name__ == "__main__":
    serve()
