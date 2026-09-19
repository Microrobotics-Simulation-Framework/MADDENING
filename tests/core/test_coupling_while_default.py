"""The early-exit while_loop solver is the default for every group config.

``solver="ift"`` (default) must reproduce the legacy fori path's converged
state for every acceleration, iteration mode, convergence norm, and with
diagnostics on; must actually exit early; must report / enforce
non-convergence; and must be differentiable in both modes for configs
the old gate excluded (jacobi, diagnostics, fixed, iqn-ils).
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


def _group_kw(**group_kw) -> dict:
    """The fixtures' group settings, minus any the chosen norm ignores.

    ``tolerance=1e-8`` is the fixture default because these graphs are
    held against the legacy ``fori`` path under the L2 norm, which reads
    it.  ``"mixed"`` and ``"interface"`` test a residual already scaled
    by ``atol`` / ``rtol`` against a fixed threshold of 1.0 and never
    read ``tolerance``, so carrying the default into those cells put a
    dead number in the group -- which ``CouplingGroup`` now warns about.
    A ``tolerance`` a test names itself is passed through regardless;
    dropping that one would hide the setting rather than the warning.
    """
    kw = dict(max_iterations=30, tolerance=1e-8)
    kw.update(group_kw)
    if kw.get("convergence_norm", "l2") != "l2" and "tolerance" not in group_kw:
        kw.pop("tolerance")
    return kw


def _springs(**group_kw) -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="spring_a", timestep=0.001, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="spring_b", timestep=0.001, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=2.0,
    ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    kw = _group_kw(**group_kw)
    gm.add_coupling_group(["spring_a", "spring_b"], **kw)
    return gm


def _slow_springs(**group_kw) -> GraphManager:
    """The same 2-cycle, coupled strongly enough to still be iterating.

    ``_springs`` is so weakly coupled (``k * dt**2 / m = 5e-5``) that
    its second pass lands on the fixed point to the last bit of
    float32: its residual is *exactly* zero from pass two on.  That is
    fine for "does it converge", but it cannot express "this group ran
    out of iterations" -- a group whose returned state has a zero
    residual has converged, whatever the cap said.  This one needs six
    passes at 1e-12, so a cap of two genuinely leaves it short.
    """
    gm = GraphManager()
    for name, pos in (("spring_a", 0.0), ("spring_b", 2.0)):
        gm.add_node(SpringDamperNode(
            name=name, timestep=0.05, stiffness=100.0, damping=1.0,
            mass=1.0, rest_length=1.0, initial_position=pos,
        ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    kw = _group_kw(**group_kw)
    gm.add_coupling_group(["spring_a", "spring_b"], **kw)
    return gm


def _rods(**group_kw) -> GraphManager:
    gm = GraphManager()
    for name, T in (("rod_a", 100.0), ("rod_b", 0.0)):
        gm.add_node(HeatNode(
            name=name, timestep=0.001, n_cells=10, thermal_diffusivity=0.01,
            length=1.0, initial_temperature=T,
        ))
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[0])
    kw = _group_kw(**group_kw)
    gm.add_coupling_group(["rod_a", "rod_b"], **kw)
    return gm


def _run(gm: GraphManager, n: int = 5) -> dict:
    s = None
    for _ in range(n):
        s = gm.step()
    return s


def _assert_states_close(a: dict, b: dict, nodes, atol=1e-5):
    for nn in nodes:
        for fld, va in a[nn].items():
            np.testing.assert_allclose(
                va, b[nn][fld], atol=atol, rtol=1e-5,
                err_msg=f"{nn}.{fld}",
            )


CONFIGS = [
    dict(acceleration="none"),
    dict(acceleration="fixed", relaxation=0.7),
    dict(acceleration="aitken"),
    dict(acceleration="iqn-ils"),
    dict(acceleration="iqn-imvj", jacobian_reuse=0),
    dict(acceleration="iqn-imvj", jacobian_reuse=3),
    dict(acceleration="none", iteration_mode="jacobi"),
    dict(acceleration="aitken", iteration_mode="jacobi"),
    dict(acceleration="iqn-ils", iteration_mode="jacobi"),
    dict(acceleration="none", diagnostics=True),
    dict(acceleration="aitken", diagnostics=True),
    dict(acceleration="iqn-imvj", jacobian_reuse=3, diagnostics=True),
    dict(acceleration="none", convergence_norm="mixed", atol=1e-8, rtol=1e-6),
    dict(acceleration="aitken", convergence_norm="mixed", atol=1e-8, rtol=1e-6),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_default_matches_fori_over_five_steps(cfg):
    s_new = _run(_springs(**cfg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s_old = _run(_springs(solver="fori", **cfg))
    _assert_states_close(s_new, s_old, ("spring_a", "spring_b"))


@pytest.mark.parametrize("norm", ["l2", "interface"])
def test_default_matches_fori_on_heat_rods(norm):
    kw = dict(convergence_norm=norm, atol=1e-8, rtol=1e-6)
    s_new = _run(_rods(**kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s_old = _run(_rods(solver="fori", **kw))
    _assert_states_close(s_new, s_old, ("rod_a", "rod_b"), atol=1e-4)


def test_imvj_warm_start_persists_secant_matrices():
    gm = _springs(acceleration="iqn-imvj", jacobian_reuse=3, diagnostics=True)
    _run(gm, 3)
    meta = gm._state["_meta"]
    V = meta["coupling_spring_a+spring_b_V"]
    assert V.shape[1] == 29
    assert float(jnp.abs(V).sum()) > 0.0, "V/W were not persisted across steps"


def test_default_is_ift_and_fori_is_deprecated():
    assert CouplingGroup(nodes=frozenset({"a", "b"})).solver == "ift"
    with pytest.warns(DeprecationWarning, match="solver='fori' is deprecated"):
        CouplingGroup(nodes=frozenset({"a", "b"}), solver="fori")


def test_exits_early_and_reports_convergence():
    gm = _springs(max_iterations=50, tolerance=1e-6, diagnostics=True)
    gm.step()
    d = gm.coupling_diagnostics()["spring_a+spring_b"]
    assert d["converged"] is True
    assert 0 < d["iterations"] < 50
    assert d["residual"] <= 1e-6


def test_unconverged_is_reported_not_raised_by_default():
    gm = _slow_springs(max_iterations=2, tolerance=1e-12, diagnostics=True)
    gm.step()
    d = gm.coupling_diagnostics()["spring_a+spring_b"]
    assert d["converged"] is False
    assert d["residual"] > 1e-12
    # A group that exhausted its budget reports the budget: two passes
    # ran and two are reported, so ``iterations >= max_iterations``
    # detects the cap.  This used to read ``1`` -- the while loop's body
    # count, one short of the passes -- which is the defect, not the
    # contract; ``fori`` always reported ``2`` here.
    assert d["iterations"] == 2


def test_strict_convergence_raises_at_cap():
    gm = _slow_springs(max_iterations=2, tolerance=1e-12,
                       strict_convergence=True)
    with pytest.raises(Exception, match="without converging"):
        gm.step()


def test_strict_convergence_silent_when_converged():
    gm = _springs(max_iterations=50, tolerance=1e-6, strict_convergence=True)
    s = gm.step()
    assert bool(jnp.isfinite(s["spring_a"]["position"]))


@pytest.mark.parametrize("cfg", [
    dict(acceleration="none", iteration_mode="jacobi"),
    dict(acceleration="none", diagnostics=True),
    dict(acceleration="fixed", relaxation=0.7),
    dict(acceleration="iqn-ils"),
    dict(acceleration="iqn-imvj", jacobian_reuse=3),
], ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_forward_and_reverse_ad_through_previously_excluded_configs(cfg):
    gm = _springs(**cfg)
    gm.step()
    compiled = gm._compiled_step
    base = gm._state

    def f(p):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        state["spring_a"]["position"] = p
        out = compiled(state, {})
        return out["spring_b"]["position"]

    p0 = jnp.array(0.1, dtype=jnp.float32)
    g = jax.grad(f)(p0)
    _, t = jax.jvp(f, (p0,), (jnp.ones_like(p0),))
    assert bool(jnp.isfinite(g)) and bool(jnp.isfinite(t))
    np.testing.assert_allclose(g, t, rtol=1e-4, atol=1e-6)
    assert float(g) != 0.0


def test_the_passes_run_do_not_scale_with_max_iterations():
    """Early exit: raising the cap on an easily converging group must not
    raise the work done (the fori path ran the cap out every step).

    Asserted on the pass count rather than the clock.  The claim is that
    the solver stops when it has converged, and the reported iteration
    count *is* that claim -- the fori path this replaced reports the cap,
    so a regression to it fails here.  A wall-clock ratio measured the
    same thing indirectly and could be defeated by a busy runner: on a
    shared CI machine it once read 3.27x against a 3.0x bound while the
    idle-box ratio is 0.97-1.31.
    """
    counts = {}
    for max_iterations in (3, 60, 200):
        gm = _rods(max_iterations=max_iterations, tolerance=1e-6,
                   diagnostics=True)
        for _ in range(3):
            gm.step()
        diag = gm.coupling_diagnostics()
        assert diag, "the group reports no diagnostics"
        for key, value in diag.items():
            assert value["converged"], (max_iterations, key, value)
            counts.setdefault(key, {})[max_iterations] = int(value["iterations"])

    for key, by_cap in counts.items():
        assert len(set(by_cap.values())) == 1, (key, by_cap)
        # Guard the assertion itself: a group that exits on its first pass
        # would satisfy cap-invariance without exercising early exit.
        assert 1 < next(iter(by_cap.values())) < 3, (key, by_cap)


@pytest.mark.slow
def test_step_cost_does_not_scale_with_max_iterations():
    """The same claim on the clock, kept as a coarse smoke check.

    Marked slow and given a wide bound because it measures ~1e-4 s per
    step, where one scheduler hiccup dominates the number.  It is here to
    catch a catastrophic regression, not to measure anything; the precise
    claim is tested above.
    """
    import time

    def timed(max_iterations):
        gm = _rods(max_iterations=max_iterations, tolerance=1e-6)
        for _ in range(3):
            gm.step()
        t0 = time.perf_counter()
        for _ in range(20):
            s = gm.step()
        jax.block_until_ready(s["rod_a"]["temperature"])
        return (time.perf_counter() - t0) / 20

    t_small, t_large = timed(3), timed(60)
    assert t_large < 8.0 * t_small, (t_small, t_large)
