"""Persistent compilation cache: enable/env/warm_cache, and a real
cross-process hit (the only test that proves persistence)."""

import os
import subprocess
import sys
import textwrap

import pytest

from maddening.core.simulation import compile_cache as cc

_CHILD = textwrap.dedent("""
    import os, sys, time
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.heat import HeatNode
    from maddening.core.simulation.compile_cache import enable, warm_cache
    enable(sys.argv[1])
    def factory():
        gm = GraphManager()
        for i in range(3):
            gm.add_node(HeatNode(f"h{i}", 1e-4, n_cells=257, thermal_diffusivity=0.1,
                                 stencil_order=4))
        gm.add_edge("h0", "h1", "temperature", "left_temperature", transform="extract_last")
        gm.add_edge("h1", "h2", "temperature", "left_temperature", transform="extract_last")
        gm.add_coupling_group(["h0", "h1", "h2"], max_iterations=8, tolerance=1e-6)
        return gm
    r = warm_cache(factory, n_steps=2, scan_steps=5)
    print(f"{r['compile_s']:.4f} {r['scan_compile_s']:.4f}")
""")


def _child(cache_dir):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONWARNINGS": "ignore"}
    out = subprocess.run([sys.executable, "-c", _CHILD, str(cache_dir)],
                         capture_output=True, text=True, env=env, check=True)
    compile_s, scan_s = map(float, out.stdout.strip().split()[-2:])
    return compile_s, scan_s


@pytest.mark.slow
def test_second_process_hits_the_persistent_cache(tmp_path):
    cold, cold_scan = _child(tmp_path)
    assert any(tmp_path.iterdir()), "no cache entries were written"
    warm, warm_scan = _child(tmp_path)
    # A cache hit skips XLA compilation: comfortably faster than cold.
    assert warm < 0.6 * cold, (cold, warm)
    assert warm_scan < 0.6 * cold_scan, (cold_scan, warm_scan)


def test_enable_is_idempotent_and_expands_user(tmp_path, monkeypatch):
    d = cc.enable(str(tmp_path / "xla"))
    assert d == str(tmp_path / "xla") and os.path.isdir(d)
    assert cc.enabled_dir() == d
    assert cc.enable(str(tmp_path / "xla")) == d
    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "env"))
    assert cc.enable_from_env() == str(tmp_path / "env")
    monkeypatch.delenv(cc.ENV_VAR)
    assert cc.enable_from_env() is None


def test_warm_cache_reports(tmp_path):
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.spring import SpringDamperNode

    def factory():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", 0.01))
        return gm

    r = cc.warm_cache(factory, n_steps=3, scan_steps=4, cache_dir=str(tmp_path))
    assert r["cache_dir"] == str(tmp_path)
    assert r["compile_s"] > 0 and r["step_ms"] >= 0 and r["scan_compile_s"] > 0
