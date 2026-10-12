"""Persistent XLA compilation cache and cache warming (PERF-2).

A MADDENING step compiles once per process — around a second for a
small coupled graph on a GPU, tens of seconds for a large LBM graph —
and every experiment run, test process or REST server restart pays it
again.  JAX can persist compiled executables to disk; this module turns
that on with settings that make it useful for MADDENING's short
compiles (JAX's defaults skip anything under a second), and offers
:func:`warm_cache` to populate the cache ahead of time.

Enable it in one of three ways::

    export MADDENING_COMPILATION_CACHE_DIR=~/.cache/maddening/xla   # env
    maddening.core.simulation.compile_cache.enable("~/.cache/...")   # code
    gm.compile()   # picks the env var up automatically, once per process

Which directory is used
-----------------------
* An explicit directory wins: ``enable(dir)`` and
  ``warm_cache(..., cache_dir=dir)`` use ``dir`` whatever
  ``MADDENING_COMPILATION_CACHE_DIR`` says.
* The environment variable is the default.  ``GraphManager.compile()``
  enables it only while no directory is enabled yet, so it never
  re-points a cache that code chose.
* ``enable()`` without a directory keeps the one already enabled, then
  falls back to the variable, then to ``~/.cache/maddening/xla``.

The directory this module reports is the directory JAX writes to.  JAX
opens its persistent cache once, at the first compile that finds a
directory set, and ignores a later change of the setting; so when
``enable`` moves the cache after that, it also drops JAX's open cache
(``jax.experimental.compilation_cache.compilation_cache.reset_cache``),
and the next compile opens the new directory.  Programs the process has
already compiled stay in memory and are not written to the new
directory.  On a JAX without that reset, a move after the cache has
opened is refused with a ``RuntimeError`` rather than reported and
ignored.  :func:`enabled_dir` reads JAX's own state, and answers
``None`` rather than a directory JAX is not using.

The cache key includes the XLA version, backend and the traced program,
so a JAX upgrade or a changed graph structure simply misses; stale
entries are never reused.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import jax

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

try:
    # Public (experimental) API, present on every JAX the CI lanes test.
    from jax.experimental.compilation_cache.compilation_cache import (
        reset_cache as _reset_jax_cache,
    )
except ImportError:  # pragma: no cover - a JAX without the public reset
    _reset_jax_cache = None

if TYPE_CHECKING:
    # Type-checking only: importing the graph module at runtime would
    # make the dependency circular.
    from maddening.core.graph_manager import GraphManager

ENV_VAR = "MADDENING_COMPILATION_CACHE_DIR"

_enabled_dir: Optional[str] = None
# One enable at a time: a REST server compiles from several threads, and
# each compile asks enable_from_env() whether a directory is enabled yet.
_lock = threading.Lock()


class _Unknown:
    """The answer when this JAX's private cache state cannot be read."""

    def __repr__(self) -> str:
        return "<unknown>"


_UNKNOWN = _Unknown()


def _setting(name: str):
    """A JAX setting's current value (``jax.config.read`` refuses the
    settings that have a context manager, as these do)."""
    return getattr(jax.config, name)


def _open_cache_dir() -> Union[str, None, _Unknown]:
    """The directory of the cache JAX has opened, ``None`` if it has opened
    none, ``_UNKNOWN`` if this JAX does not say.

    Reads private JAX state (``jax._src.compilation_cache._cache`` and its
    ``_path``), the only record of where JAX reads and writes once it has
    opened its cache.  The layout is the same on every JAX the CI lanes
    test (``tests/core/test_compile_cache.py`` holds it to that); any
    other layout reads as ``_UNKNOWN``, never as a directory.
    """
    try:
        from jax._src import compilation_cache as jax_state  # noqa: PLC0415
    except ImportError:  # pragma: no cover - a JAX with another layout
        return _UNKNOWN
    if not hasattr(jax_state, "_cache"):
        return _UNKNOWN
    opened = jax_state._cache  # noqa: SLF001 -- the object JAX writes through
    if opened is None:
        return None
    path = getattr(opened, "_path", None)
    if path is None:
        return _UNKNOWN
    return os.path.abspath(str(path))


def _jax_dir_in_use() -> Optional[str]:
    """Where JAX's persistent cache reads and writes: the cache it has
    opened, else the directory its next compile will open; ``None`` when
    the cache is off."""
    if not _setting("jax_enable_compilation_cache"):
        return None
    opened = _open_cache_dir()
    if isinstance(opened, str):
        return opened
    # Nothing opened yet (or a JAX that does not say): the setting is
    # where the next compile opens it.  enable() drops the open cache
    # whenever it moves the setting, so through this module the two agree.
    setting = _setting("jax_compilation_cache_dir")
    return os.path.abspath(setting) if setting else None


def _resolve(cache_dir: Optional[str]) -> str:
    """The absolute directory ``enable(cache_dir)`` uses (lock held)."""
    if cache_dir is None:
        if _enabled_dir is not None:
            return _enabled_dir  # already enabled: keep that directory
        cache_dir = os.environ.get(ENV_VAR)
    target = cache_dir or str(Path.home() / ".cache" / "maddening" / "xla")
    # Absolute, so a later chdir cannot move where JAX writes away from
    # the directory reported here.
    return os.path.abspath(os.path.expanduser(target))


def _point_jax_at(target: str, min_compile_time_secs: Optional[float]) -> str:
    """Make ``target`` the directory JAX reads and writes (lock held)."""
    global _enabled_dir
    Path(target).mkdir(parents=True, exist_ok=True)
    opened = _open_cache_dir()
    if opened is _UNKNOWN and not _setting("jax_compilation_cache_dir"):
        opened = None  # no directory was ever set: JAX cannot have opened one
    must_reopen = opened is not None and opened != target
    reset = _reset_jax_cache
    if must_reopen and reset is None:
        raise RuntimeError(
            f"JAX's persistent compilation cache is already open at {opened!r}, and "
            f"this JAX ({jax.__version__}) offers no way to re-open it: JAX would keep "
            f"writing to {opened!r} while the cache was reported at {target!r}.  Call "
            f"compile_cache.enable({target!r}) before the process compiles anything, or "
            f"set {ENV_VAR} (or JAX_COMPILATION_CACHE_DIR) before it starts.")
    if min_compile_time_secs is not None:
        secs = float(min_compile_time_secs)
    elif _enabled_dir is not None:
        # Keep the threshold an earlier enable() chose.
        secs = float(_setting("jax_persistent_cache_min_compile_time_secs"))
    else:
        secs = 0.0  # JAX's own default (1 s) would skip most MADDENING steps
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", target)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", secs)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    if must_reopen and reset is not None:
        # After the setting moved, so the next compile (in any thread)
        # opens the new directory.
        reset()
    _enabled_dir = target
    return target


@stability(StabilityLevel.EVOLVING)
def enable(cache_dir: Optional[str] = None, *,
           min_compile_time_secs: Optional[float] = None) -> str:
    """Point JAX's persistent compilation cache at ``cache_dir``.

    ``cache_dir`` defaults to the directory already enabled, then the
    ``MADDENING_COMPILATION_CACHE_DIR`` environment variable, then
    ``~/.cache/maddening/xla``; an explicit ``cache_dir`` wins over the
    variable.  Sets ``jax_persistent_cache_min_compile_time_secs`` to
    ``min_compile_time_secs`` (default: the threshold an earlier call set,
    else 0, since MADDENING steps often compile in well under JAX's 1 s
    default) and disables the minimum entry size.

    If JAX has already opened its cache in another directory, that cache
    is dropped and the next compile opens ``cache_dir``; programs already
    compiled in this process stay in memory and are not written there.  A
    JAX that cannot re-open its cache refuses the move with a
    ``RuntimeError``.  Re-pointing while another thread is compiling may
    let that compile use either directory; :func:`enabled_dir` then says
    which (``None`` if not this one).

    Idempotent; returns the directory in use (absolute).
    """
    with _lock:
        return _point_jax_at(_resolve(cache_dir), min_compile_time_secs)


def enabled_dir() -> Optional[str]:
    """The directory :func:`enable` set, while JAX's cache uses it.

    ``None`` before :func:`enable`, and once JAX's persistent cache reads
    and writes somewhere else (after a direct ``jax.config.update``, say):
    never a directory JAX is not using.
    """
    d = _enabled_dir
    if d is None or _jax_dir_in_use() != d:
        return None
    return d


@stability(StabilityLevel.EVOLVING)
def enable_from_env() -> Optional[str]:
    """:func:`enable` the ``MADDENING_COMPILATION_CACHE_DIR`` directory, if
    the variable is set and no directory is enabled yet.

    ``GraphManager.compile()`` calls this on every compile; once a
    directory is enabled, by code or by an earlier call, it changes
    nothing, so an explicit :func:`enable` wins over the variable.
    Returns :func:`enabled_dir`, or ``None`` when the variable is unset.
    """
    if not os.environ.get(ENV_VAR):
        return None
    with _lock:
        if _enabled_dir is None:
            _point_jax_at(_resolve(None), None)
    return enabled_dir()


@stability(StabilityLevel.EVOLVING)
def warm_cache(
    gm_factory: Callable[[], "GraphManager"],
    *,
    external_inputs: Optional[dict] = None,
    n_steps: int = 1,
    scan_steps: int = 0,
    cache_dir: Optional[str] = None,
) -> dict:
    """Compile a graph's step (and optionally its ``run_scan``) so the
    persistent cache holds the executables before the real run.

    Parameters
    ----------
    gm_factory : callable
        Builds and returns the graph (same structure as the run will use;
        the cache is keyed on the traced program).
    external_inputs : dict, optional
        Inputs for the steps (``None`` = the graph's zero defaults).
    n_steps : int
        Steps to run through ``gm.step`` (at least 1 to trigger the
        compile; the second step is what proves no retrace).
    scan_steps : int
        If > 0, also compile ``run_scan(scan_steps)``.
    cache_dir : str, optional
        Passed to :func:`enable`: an explicit directory wins over
        ``MADDENING_COMPILATION_CACHE_DIR``; ``None`` keeps the directory
        already enabled, then the variable, then the default.

    Returns
    -------
    dict
        ``{"cache_dir", "compile_s", "step_ms", "scan_compile_s"}``;
        ``cache_dir`` is the directory the programs were written to.

    Raises
    ------
    RuntimeError
        If JAX's cache was re-pointed elsewhere during the warm-up, so the
        programs are not in ``cache_dir``.
    """
    d = enable(cache_dir)
    gm = gm_factory()
    t0 = time.perf_counter()
    gm.compile()
    gm.step(external_inputs)
    jax.block_until_ready(jax.tree.leaves(gm._state))  # noqa: SLF001
    compile_s = time.perf_counter() - t0
    step_ms = 0.0
    for _ in range(max(0, n_steps - 1)):
        t0 = time.perf_counter()
        gm.step(external_inputs)
        jax.block_until_ready(jax.tree.leaves(gm._state))  # noqa: SLF001
        step_ms = (time.perf_counter() - t0) * 1000
    scan_s = 0.0
    if scan_steps > 0:
        t0 = time.perf_counter()
        jax.block_until_ready(jax.tree.leaves(gm.run_scan(scan_steps, external_inputs)))
        scan_s = time.perf_counter() - t0
    if enabled_dir() != d:
        raise RuntimeError(
            f"warm_cache enabled {d!r}, but JAX's persistent cache now reads and "
            f"writes {_jax_dir_in_use()!r}: something re-pointed it during the "
            f"warm-up, and the compiled programs are not in {d!r}.")
    return {"cache_dir": d, "compile_s": compile_s, "step_ms": step_ms,
            "scan_compile_s": scan_s}


__all__ = ["ENV_VAR", "enable", "enable_from_env", "enabled_dir", "warm_cache"]
