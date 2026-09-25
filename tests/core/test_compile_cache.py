"""Persistent compilation cache: enable/env/warm_cache, and a real
cross-process hit (the only test that proves persistence).

``enable`` and ``warm_cache`` switch JAX's persistent cache on for the
whole process, and JAX builds its cache object once, at the first compile
that finds a directory set.  Left alone, the two in-process tests here
turned a persistent cache on in pytest's temporary directory for every test
that ran after them: in the slow lane, whose runs are the "cache off"
timings that triage is based on, 858 of one shard's 1503 tests ran with a
cache.  Every test here therefore runs inside ``_compile_cache_restored``.
"""

import contextlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax._src import compilation_cache as _jax_cache_state
from jax.experimental.compilation_cache import compilation_cache as jax_cache

from maddening.core.simulation import compile_cache as cc

#: The process-wide settings ``cc.enable`` changes.
_JAX_SETTINGS = ("jax_compilation_cache_dir", "jax_persistent_cache_min_compile_time_secs",
                 "jax_persistent_cache_min_entry_size_bytes")


def _snapshot():
    return {k: getattr(jax.config, k) for k in _JAX_SETTINGS}, cc.enabled_dir()


@contextlib.contextmanager
def _compile_cache_restored():
    """Undo everything ``enable`` / ``warm_cache`` change process-wide.

    The three JAX settings and ``compile_cache``'s own record are put back,
    and JAX's cache object is dropped on the way in and out: JAX never
    re-reads the directory once it has built the object, so without the
    reset on entry a test's ``enable(tmp)`` would silently keep writing to
    the directory the process started with, and without the one on exit
    every later compile would keep writing to the test's.
    """
    settings, enabled = _snapshot()
    jax_cache.reset_cache()
    try:
        yield
    finally:
        for key, value in settings.items():
            jax.config.update(key, value)
        cc._enabled_dir = enabled  # noqa: SLF001 -- restoring what enable() set
        jax_cache.reset_cache()


@pytest.fixture(scope="module", autouse=True)
def _module_start():
    """The settings when this module started; checked again when it ends."""
    start = _snapshot()
    yield start
    assert _snapshot() == start, "a test in this module left a compilation cache switched on"


@pytest.fixture(autouse=True)
def _no_cache_leaks_out():
    with _compile_cache_restored():
        yield

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
            # timestep 1e-5, not the 1e-4 this used to pass: at 257 cells
            # and alpha=0.1 that is a Fourier number of 0.66, which the
            # 4th-order stencil's 5/16 limit refuses since 0.4.0.  This
            # test measures compile *time* and never looks at a value, so
            # only the timestep moves -- cell count, stencil order and
            # graph shape, which are what set the compile cost, are as
            # they were.
            gm.add_node(HeatNode(f"h{i}", 1e-5, n_cells=257, thermal_diffusivity=0.1,
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


def _entries(cache_dir):
    return sorted(p.name for p in cache_dir.rglob("*") if p.is_file())


# Not slow-marked: two child processes, 4.3 s on the CI runner -- under
# the 5 s line -- and the only test that proves persistence.  (Its old
# @pytest.mark.slow sat on ``_entries`` above, where a mark does nothing,
# after a helper was inserted between it and this test.)
def test_second_process_hits_the_persistent_cache(tmp_path):
    cold, cold_scan = _child(tmp_path)
    entries = _entries(tmp_path)
    assert entries, "no cache entries were written"
    warm, warm_scan = _child(tmp_path)
    # The proof of a hit is that the second process compiled nothing new:
    # the cache holds exactly the entries the first one wrote.  (A wall-
    # clock ratio was the old check; it is unreliable under CPU load.)
    assert _entries(tmp_path) == entries, "the warm process added cache entries"
    # and a hit is never slower than a cold compile
    assert warm <= cold and warm_scan <= cold_scan, (cold, warm, cold_scan, warm_scan)


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


def test_the_restore_undoes_enable_and_forgets_the_cache_it_built(tmp_path):
    """``enable`` plus a compile builds a cache in ``tmp_path``; afterwards nothing writes there."""
    before = _snapshot()
    target = tmp_path / "xla"
    with _compile_cache_restored():
        cc.enable(str(target))
        jax.jit(lambda x: x * 2.0 + 1.0)(jnp.float32(1.0)).block_until_ready()
        written = _entries(target)
        assert written, "enable() and a compile wrote no cache entry: the check below proves nothing"
    assert _snapshot() == before
    # A program this process has never compiled: had the cache object
    # survived, it would be written to ``target`` too.
    jax.jit(lambda x: x * 3.0 - 7.0)(jnp.float32(2.0)).block_until_ready()
    assert _entries(target) == written, "a compile after the restore still wrote to the test's cache"


# Keep this last: it checks the state every test above left behind.
def test_the_module_leaves_the_process_cache_as_it_found_it(_module_start):
    settings, enabled = _snapshot()
    assert (settings, enabled) == _module_start
    built = _jax_cache_state._cache  # noqa: SLF001 -- the object JAX writes through
    if built is not None:
        # Rebuilt since the last reset: only ever from the starting directory.
        assert settings["jax_compilation_cache_dir"] is not None
        assert Path(built._path) == Path(settings["jax_compilation_cache_dir"])  # noqa: SLF001
