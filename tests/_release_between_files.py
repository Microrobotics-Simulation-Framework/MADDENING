"""Release what a test file compiled once its last test has run.

The slow lane runs each shard's files in one process, and nothing else
bounds what JAX keeps across them: every compiled program stays in JAX's
caches, and a test file that keeps graphs in a module-level
``functools.lru_cache`` keeps them, and their programs, for the rest of
the process.  Files are dealt onto shards by measured time, not memory,
so a deal that puts several such files on one shard runs the runner out
of memory: XLA aborts the process inside a compile (SIGABRT) and the
shard reports nothing after it (2026-10-09, shard 6 of 8; 2026-10-12,
shard 6 of 10, in ``test_differential_coupling_topologies.py`` after
``test_converged_groups_are_at_their_fixed_point.py`` and
``test_differential_coupling_solvers.py``).

When a file's last test has been torn down (its module fixtures with
it), :func:`release`

* clears every ``functools.lru_cache`` / ``functools.cache`` defined in a
  module under ``tests/`` (the test file's own, and those of helper
  modules other files share: they rebuild on their next call);
* collects, then clears JAX's compiled programs (``jax.clear_caches()``);
* hands the freed heap back to the OS (glibc ``malloc_trim``), so that
  the shard's resident size measures what it still holds.

A cache imported from the library is left alone: only one whose
``__module__`` is the module it sits in is cleared.  Nothing inside a
file changes: the release runs between files, never between a file's
tests, so a file's tests still share their module fixtures and caches.

Registered from ``tests/conftest.py`` only when
``MADDENING_TEST_RELEASE_BETWEEN_FILES=1`` (the slow lane sets it), so
local runs and the per-push lane are untouched.  Each release prints one
line (the file, the caches cleared, the resident size before and after),
so a shard's log shows where its memory goes.
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from pathlib import Path

import pytest

ENV_VAR = "MADDENING_TEST_RELEASE_BETWEEN_FILES"

_TESTS = Path(__file__).resolve().parent


def _defined_under_tests(module) -> bool:
    path = getattr(module, "__file__", None)
    if not path:
        return False
    try:
        return Path(path).resolve().is_relative_to(_TESTS)
    except (OSError, ValueError):
        return False


def clear_test_caches(modules=None) -> int:
    """Clear each lru_cache defined in a module under ``tests/``; return how many.

    ``modules`` is ``{name: module}``, ``sys.modules`` by default.
    """
    cleared = 0
    for name, module in list((sys.modules if modules is None else modules).items()):
        if module is None or not _defined_under_tests(module):
            continue
        for obj in list(vars(module).values()):
            if (callable(getattr(obj, "cache_clear", None))
                    and callable(getattr(obj, "cache_info", None))
                    and getattr(obj, "__module__", None) == name):
                obj.cache_clear()
                cleared += 1
    return cleared


def _malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc: the memory is free for reuse, if not returned


def resident_bytes() -> int | None:
    """This process's resident set size, or None where /proc is absent."""
    try:
        with open("/proc/self/statm", encoding="ascii") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def release() -> int:
    """Release the test files' caches and JAX's compiled programs; return the caches cleared."""
    cleared = clear_test_caches()
    gc.collect()
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
    gc.collect()
    _malloc_trim()
    return cleared


def _gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.2f} GB"


class Plugin:
    """Calls :func:`release` after the teardown of each file's last test."""

    def __init__(self, config, release=release):
        self.config = config
        self.release = release

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        # trylast: after pytest's own teardown, which has finalised the
        # module's fixtures by now when the next test is in another file.
        if nextitem is not None and nextitem.path == item.path:
            return
        before = resident_bytes()
        cleared = self.release()
        after = resident_bytes()
        reporter = self.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"[release after {item.path.name}] {cleared} test caches cleared; "
                f"resident {_gb(before)} -> {_gb(after)}")


def register(config):
    """Register the plugin; called from conftest."""
    config.pluginmanager.register(Plugin(config), "maddening-release-between-files")
