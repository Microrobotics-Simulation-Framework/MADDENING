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
                 "jax_persistent_cache_min_entry_size_bytes", "jax_enable_compilation_cache")


def _snapshot():
    return {k: getattr(jax.config, k) for k in _JAX_SETTINGS}, cc._enabled_dir  # noqa: SLF001


def _cache_left_behind(settings):
    """Where JAX's cache object points, if not at the directory ``settings`` name.

    JAX reads and writes through this object whatever the settings say
    once it exists: with the settings restored but the object kept, every
    later compile still looks programs up in, and (above the restored
    minimum compile time) writes them to, the test's directory.
    """
    built = _jax_cache_state._cache  # noqa: SLF001 -- the object JAX writes through
    if built is None:
        return None
    want = settings["jax_compilation_cache_dir"]
    if want is not None and Path(built._path) == Path(want):  # noqa: SLF001
        return None
    return str(built._path)  # noqa: SLF001


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
    assert _cache_left_behind(start[0]) is None, "a test in this module left JAX's cache object behind"


@pytest.fixture(autouse=True)
def _no_cache_leaks_out():
    with _compile_cache_restored():
        yield


@pytest.fixture
def no_directory_enabled(monkeypatch):
    """Start from a process in which no directory is enabled and the
    variable is unset (the autouse fixture puts both back afterwards)."""
    monkeypatch.setattr(cc, "_enabled_dir", None)
    monkeypatch.delenv(cc.ENV_VAR, raising=False)


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
    # The variable never re-points a directory code enabled.
    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "env"))
    assert cc.enable_from_env() == d
    monkeypatch.delenv(cc.ENV_VAR)
    assert cc.enable_from_env() is None
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cc.enable("~/xla") == str(tmp_path / "home" / "xla") == cc.enabled_dir()


def test_the_variable_then_the_home_directory_are_the_defaults(tmp_path, monkeypatch, no_directory_enabled):
    assert cc.enable_from_env() is None and cc.enabled_dir() is None
    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "env"))
    assert cc.enable_from_env() == str(tmp_path / "env") == cc.enabled_dir()
    monkeypatch.setattr(cc, "_enabled_dir", None)
    monkeypatch.delenv(cc.ENV_VAR)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cc.enable() == str(tmp_path / "home" / ".cache" / "maddening" / "xla") == cc.enabled_dir()


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
    """``enable`` plus a compile builds a cache in the test's directory;
    afterwards nothing uses it, and the cache the process had is back.

    The process starts the test with a cache already in use, as a CI lane
    does (``JAX_COMPILATION_CACHE_DIR``); the autouse fixture undoes that.
    """
    outer = tmp_path / "outer"
    jax.config.update("jax_compilation_cache_dir", str(outer))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.jit(lambda x: x - 5.0)(jnp.float32(1.0)).block_until_ready()
    assert _entries(outer), "the outer cache is not in use: the checks below prove nothing"
    before = _snapshot()
    target = tmp_path / "xla"
    with _compile_cache_restored():
        cc.enable(str(target))
        jax.jit(lambda x: x * 2.0 + 1.0)(jnp.float32(1.0)).block_until_ready()
        written = _entries(target)
        assert written, "the compile went to the cache the process already had, not enable()'s"
        assert _cache_left_behind(before[0]) == str(target)     # ...so the check can fire
    assert _snapshot() == before
    assert _cache_left_behind(before[0]) is None
    in_outer = len(_entries(outer))
    # A program this process has never compiled: had the cache object
    # survived, it would be looked up in ``target`` and written there.
    jax.jit(lambda x: x * 3.0 - 7.0)(jnp.float32(2.0)).block_until_ready()
    assert _cache_left_behind(before[0]) is None
    assert _entries(target) == written, "a compile after the restore still wrote to the test's cache"
    assert len(_entries(outer)) > in_outer, "the process's own cache did not come back"


# ---------------------------------------------------------------------------
# The directory the module reports is the directory JAX writes to, and an
# explicit directory wins over MADDENING_COMPILATION_CACHE_DIR.
#
# JAX opens its persistent cache once, at the first compile that finds a
# directory set, and ignores a later change of the setting.  Every entry
# point that enables the cache is held to the rule below: enable(dir),
# enable() without a directory, enable_from_env(), GraphManager.compile()
# (which calls it, and through which the REST server and the cloud entry
# point compile), and warm_cache() with and without cache_dir.
# ---------------------------------------------------------------------------

def _compile_new_program(scale):
    """Compile a program this process has not compiled before (``scale`` is
    baked into it), so the persistent cache is looked up and written."""
    jax.jit(lambda x: x * scale + scale ** 2)(jnp.float32(1.0)).block_until_ready()


def _spring_graph():
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    return gm


def test_a_move_after_the_cache_opened_writes_to_the_new_directory(tmp_path, no_directory_enabled):
    a, b = tmp_path / "a", tmp_path / "b"
    assert cc.enable(str(a)) == str(a)
    _compile_new_program(3.25)
    in_a = _entries(a)
    assert in_a, "the first compile wrote nothing: the checks below prove nothing"
    assert cc.enable(str(b)) == str(b)
    assert cc.enabled_dir() == str(b)
    _compile_new_program(5.75)
    assert _entries(b), "enable(b) reported b while JAX kept writing elsewhere"
    assert _entries(a) == in_a, "a compile after enable(b) still wrote to a"
    assert cc.enabled_dir() == str(b)


def test_an_explicit_directory_wins_over_the_variable_in_graph_compile(
        tmp_path, monkeypatch, no_directory_enabled):
    env, mine = tmp_path / "env", tmp_path / "mine"
    monkeypatch.setenv(cc.ENV_VAR, str(env))
    assert cc.enable(str(mine)) == str(mine)
    gm = _spring_graph()
    gm.compile()
    gm.step()
    assert cc.enabled_dir() == str(mine)
    assert _entries(mine), "the graph's step was not written to the enabled directory"
    assert not _entries(env), "GraphManager.compile() re-pointed the cache at the variable"


def test_an_explicit_directory_wins_after_an_eager_op_opened_the_cache(
        tmp_path, monkeypatch, no_directory_enabled):
    """The case whose outcome used to depend on whether anything had
    compiled between ``enable`` and ``compile()``."""
    env, mine = tmp_path / "env", tmp_path / "mine"
    monkeypatch.setenv(cc.ENV_VAR, str(env))
    cc.enable(str(mine))
    _compile_new_program(7.125)          # opens JAX's cache at ``mine``
    gm = _spring_graph()
    gm.compile()
    gm.step()
    _compile_new_program(9.375)
    assert cc.enabled_dir() == str(mine)
    assert not _entries(env)


def test_warm_cache_writes_to_the_directory_it_was_given_whatever_the_variable_says(
        tmp_path, monkeypatch, no_directory_enabled):
    env, mine = tmp_path / "env", tmp_path / "mine"
    monkeypatch.setenv(cc.ENV_VAR, str(env))
    r = cc.warm_cache(_spring_graph, n_steps=2, cache_dir=str(mine))
    assert r["cache_dir"] == str(mine) == cc.enabled_dir()
    assert _entries(mine) and not _entries(env)


def test_warm_cache_without_a_directory_keeps_the_one_enabled(
        tmp_path, monkeypatch, no_directory_enabled):
    env, mine = tmp_path / "env", tmp_path / "mine"
    monkeypatch.setenv(cc.ENV_VAR, str(env))
    cc.enable(str(mine))
    r = cc.warm_cache(_spring_graph, n_steps=2)
    assert r["cache_dir"] == str(mine) == cc.enabled_dir()
    assert _entries(mine) and not _entries(env)


def test_enable_without_a_directory_keeps_the_one_enabled(tmp_path, monkeypatch, no_directory_enabled):
    mine = tmp_path / "mine"
    cc.enable(str(mine))
    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "env"))
    assert cc.enable() == str(mine)
    assert cc.enable_from_env() == str(mine)
    assert cc.enabled_dir() == str(mine)
    assert not (tmp_path / "env").exists()


def test_graph_compile_moves_an_already_open_cache_to_the_variable(
        tmp_path, monkeypatch, no_directory_enabled):
    """With the variable set and nothing enabled, ``compile()`` enables the
    variable's directory, also when JAX had already opened a cache of its
    own (a CI lane's ``JAX_COMPILATION_CACHE_DIR``): JAX then writes there."""
    outer, env = tmp_path / "outer", tmp_path / "env"
    jax.config.update("jax_compilation_cache_dir", str(outer))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    _compile_new_program(11.5)
    in_outer = _entries(outer)
    assert in_outer, "the outer cache is not in use: the checks below prove nothing"
    monkeypatch.setenv(cc.ENV_VAR, str(env))
    gm = _spring_graph()
    gm.compile()
    gm.step()
    assert cc.enabled_dir() == str(env)
    assert _entries(env), "compile() reported the variable's directory while JAX wrote elsewhere"
    assert _entries(outer) == in_outer


def test_enabled_dir_never_names_a_directory_jax_is_not_using(tmp_path, no_directory_enabled):
    mine, other = tmp_path / "mine", tmp_path / "other"
    cc.enable(str(mine))
    # Re-pointed before anything opened the cache: the next compile opens ``other``.
    jax.config.update("jax_compilation_cache_dir", str(other))
    assert cc.enabled_dir() is None
    _compile_new_program(13.25)
    assert _entries(other) and not _entries(mine)
    assert cc.enabled_dir() is None
    # enable() again wins, and moves JAX's open cache back.
    assert cc.enable(str(mine)) == str(mine) == cc.enabled_dir()
    _compile_new_program(15.75)
    assert _entries(mine)
    # Re-pointed after the cache opened: JAX keeps writing to ``mine``, and says so.
    jax.config.update("jax_compilation_cache_dir", str(other))
    in_other = _entries(other)
    _compile_new_program(17.125)
    assert _entries(other) == in_other
    assert cc.enabled_dir() == str(mine)
    # A cache switched off is no directory at all, and enable() switches it on.
    jax.config.update("jax_enable_compilation_cache", False)
    assert cc.enabled_dir() is None
    in_mine = _entries(mine)
    assert cc.enable(str(mine)) == str(mine) == cc.enabled_dir()
    _compile_new_program(29.5)
    assert len(_entries(mine)) > len(in_mine), "enable() left the cache switched off"


def test_enable_moves_a_cache_open_elsewhere_even_when_the_setting_already_names_its_directory(
        tmp_path, no_directory_enabled):
    """JAX's setting can name the directory while the cache JAX opened is
    somewhere else (the setting was changed after the cache opened):
    ``enable`` must go by the open cache, not by the setting."""
    outer, mine = tmp_path / "outer", tmp_path / "mine"
    jax.config.update("jax_compilation_cache_dir", str(outer))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    _compile_new_program(25.5)
    in_outer = _entries(outer)
    assert in_outer, "the outer cache is not in use: the checks below prove nothing"
    jax.config.update("jax_compilation_cache_dir", str(mine))   # the open cache stays at outer
    assert cc.enable(str(mine)) == str(mine) == cc.enabled_dir()
    _compile_new_program(27.25)
    assert _entries(mine) and _entries(outer) == in_outer


def test_the_open_cache_probe_reads_this_jax(tmp_path, no_directory_enabled):
    """``_open_cache_dir`` reads private JAX state.  On a JAX whose layout
    differs it answers "unknown" and the module falls back to the setting;
    this test makes that a failure on every JAX the lanes run, not a quiet
    loss of the check."""
    assert cc._open_cache_dir() is None  # noqa: SLF001 -- the autouse fixture reset it
    cc.enable(str(tmp_path))
    _compile_new_program(19.5)
    assert cc._open_cache_dir() == str(tmp_path)  # noqa: SLF001


def test_a_jax_that_cannot_reopen_its_cache_refuses_the_move(tmp_path, monkeypatch, no_directory_enabled):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setattr(cc, "_reset_jax_cache", None)
    cc.enable(str(a))                    # nothing opened yet: no reset needed
    _compile_new_program(21.25)
    with pytest.raises(RuntimeError, match="before the process compiles anything"):
        cc.enable(str(b))
    assert cc.enabled_dir() == str(a)
    assert jax.config.jax_compilation_cache_dir == str(a)
    assert cc.enable(str(a)) == str(a)   # staying put is not a move


def test_warm_cache_refuses_to_report_a_directory_it_did_not_write_to(tmp_path, no_directory_enabled):
    mine, other = tmp_path / "mine", tmp_path / "other"

    def factory():
        jax.config.update("jax_compilation_cache_dir", str(other))   # behind the module's back
        return _spring_graph()

    with pytest.raises(RuntimeError, match="re-pointed it during the warm-up"):
        cc.warm_cache(factory, cache_dir=str(mine))


def test_the_threshold_given_is_the_threshold_applied(tmp_path, monkeypatch, no_directory_enabled):
    mine = tmp_path / "mine"
    secs = "jax_persistent_cache_min_compile_time_secs"
    cc.enable(str(mine))
    assert getattr(jax.config, secs) == 0.0
    cc.enable(str(mine), min_compile_time_secs=2.5)     # same directory, new threshold
    assert getattr(jax.config, secs) == 2.5
    cc.enable(str(mine))                                 # no threshold given: kept
    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "env"))
    cc.enable_from_env()
    assert getattr(jax.config, secs) == 2.5


def test_a_relative_directory_is_reported_absolute(tmp_path, monkeypatch, no_directory_enabled):
    monkeypatch.chdir(tmp_path)
    assert cc.enable("xla") == str(tmp_path / "xla")
    monkeypatch.chdir(tmp_path / "xla")
    _compile_new_program(23.75)
    assert _entries(tmp_path / "xla") and cc.enabled_dir() == str(tmp_path / "xla")


def test_only_compile_cache_points_jax_at_a_cache_directory():
    """The class guard: one module writes JAX's cache settings, and the
    only other code that enables a cache is ``GraphManager.compile()``,
    through ``enable_from_env``.  A route that pointed JAX at a directory
    itself (a server start-up, an entry point) would bypass the rule the
    tests above hold."""
    src = Path(cc.__file__).resolve().parents[2]       # src/maddening
    own = Path(cc.__file__).resolve()
    writers, enablers = [], []
    for path in sorted(src.rglob("*.py")):
        # The examples are user code: they call the public API like anyone.
        if path.resolve() == own or "examples" in path.relative_to(src).parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(src))
        if any(t in text for t in ("jax_compilation_cache_dir", "reset_cache(", "set_cache_dir(",
                                   "jax_enable_compilation_cache")):
            writers.append(rel)
        if any(t in text for t in ("enable_from_env(", "compile_cache.enable(", "warm_cache(")):
            enablers.append(rel)
    assert writers == [], writers
    assert enablers == ["core/graph_manager.py"], enablers


# Keep this last: it checks the state every test above left behind.
def test_the_module_leaves_the_process_cache_as_it_found_it(_module_start):
    assert _snapshot() == _module_start
    # Rebuilt since the last reset, if at all: only ever from the starting directory.
    assert _cache_left_behind(_module_start[0]) is None
