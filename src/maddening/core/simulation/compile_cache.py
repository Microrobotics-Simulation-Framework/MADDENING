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

The cache key includes the XLA version, backend and the traced program,
so a JAX upgrade or a changed graph structure simply misses; stale
entries are never reused.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import jax

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

if TYPE_CHECKING:
    # Type-checking only: importing the graph module at runtime would
    # make the dependency circular.
    from maddening.core.graph_manager import GraphManager

ENV_VAR = "MADDENING_COMPILATION_CACHE_DIR"

_enabled_dir: Optional[str] = None


@stability(StabilityLevel.EVOLVING)
def enable(cache_dir: Optional[str] = None, *, min_compile_time_secs: float = 0.0) -> str:
    """Point JAX's persistent compilation cache at ``cache_dir``.

    ``cache_dir`` defaults to the ``MADDENING_COMPILATION_CACHE_DIR``
    environment variable, then ``~/.cache/maddening/xla``.  Sets
    ``jax_persistent_cache_min_compile_time_secs`` to
    ``min_compile_time_secs`` (default 0: MADDENING steps often compile
    in well under JAX's 1 s default threshold) and disables the minimum
    entry size.  Idempotent; returns the directory in use.
    """
    global _enabled_dir
    if cache_dir is None:
        env = os.environ.get(ENV_VAR)
        if env is None and _enabled_dir is not None:
            return _enabled_dir      # already enabled: keep that directory
        cache_dir = env
    target = cache_dir or str(Path.home() / ".cache" / "maddening" / "xla")
    target = str(Path(target).expanduser())
    Path(target).mkdir(parents=True, exist_ok=True)
    if _enabled_dir == target:
        return target
    jax.config.update("jax_compilation_cache_dir", target)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time_secs))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    _enabled_dir = target
    return target


def enabled_dir() -> Optional[str]:
    """The cache directory :func:`enable` set, or ``None``."""
    return _enabled_dir


@stability(StabilityLevel.EVOLVING)
def enable_from_env() -> Optional[str]:
    """:func:`enable` if ``MADDENING_COMPILATION_CACHE_DIR`` is set; else no-op."""
    if os.environ.get(ENV_VAR):
        return enable()
    return None


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
        Passed to :func:`enable`.

    Returns
    -------
    dict
        ``{"cache_dir", "compile_s", "step_ms", "scan_compile_s"}``.
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
    return {"cache_dir": d, "compile_s": compile_s, "step_ms": step_ms,
            "scan_compile_s": scan_s}


__all__ = ["ENV_VAR", "enable", "enable_from_env", "enabled_dir", "warm_cache"]
