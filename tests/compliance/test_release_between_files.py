"""The slow lane releases each test file's compiled programs when the file ends.

``tests/_release_between_files.py`` is what keeps a shard's memory bounded
by its largest file instead of by the sum of its files.  These tests hold
its three parts: what it clears, when it runs, and that the slow lane
switches it on.
"""

from __future__ import annotations

import functools
import re
import sys
import types
from pathlib import Path

import jax

from tests import _release_between_files as rb

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    return module


def _cache_in(module, name):
    @functools.lru_cache(maxsize=None)
    def cached(x):
        return [x]
    cached.__module__ = module.__name__
    setattr(module, name, cached)
    cached(1)
    cached(2)
    return cached


def test_a_cache_a_test_file_defines_is_cleared_and_others_are_not():
    """Cleared: an lru_cache defined in a module under tests/.  Kept: one
    imported into it from elsewhere, and one in a module outside tests/."""
    under = _module("tests._release_probe_under", rb._TESTS / "_release_probe_under.py")
    outside = _module("_release_probe_outside", REPO_ROOT / "src" / "_release_probe_outside.py")
    own = _cache_in(under, "own")
    elsewhere = _cache_in(outside, "elsewhere")
    under.imported = elsewhere  # a library cache seen through a test file's namespace

    # Synthetic modules only: the real ones belong to the run this test is in.
    assert rb.clear_test_caches({m.__name__: m for m in (under, outside)}) == 1

    assert own.cache_info().currsize == 0
    assert elsewhere.cache_info().currsize == 2


def test_the_release_clears_the_caches_then_jax_compiled_programs(monkeypatch):
    """Stubbed, so the run this test is in keeps its programs.  That
    ``jax.clear_caches()`` empties a jitted function's cache was checked on
    jax 0.10.2, 0.11.0 and 0.11.2 (``f._cache_size()`` 1 -> 0)."""
    calls = []
    monkeypatch.setattr(rb, "clear_test_caches", lambda: calls.append("test caches") or 3)
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append("jax"))

    assert rb.release() == 3

    assert calls == ["test caches", "jax"]


class _Item:
    def __init__(self, path):
        self.path = Path(path)


class _Config:
    class pluginmanager:  # noqa: N801 - stands in for pytest's attribute
        @staticmethod
        def get_plugin(name):
            return None


def test_the_release_runs_when_a_file_ends_and_never_between_its_tests():
    calls = []
    plugin = rb.Plugin(_Config(), release=lambda: calls.append(1) or 0)
    a1, a2, b1 = _Item("tests/a/test_one.py"), _Item("tests/a/test_one.py"), _Item("tests/b/test_two.py")

    plugin.pytest_runtest_teardown(a1, a2)
    assert calls == []
    plugin.pytest_runtest_teardown(a2, b1)
    assert calls == [1]
    plugin.pytest_runtest_teardown(b1, None)
    assert calls == [1, 1]


def test_the_slow_lane_switches_it_on_and_the_conftest_registers_it():
    workflow = (REPO_ROOT / ".github" / "workflows" / "slow-tests.yml").read_text(encoding="utf-8")
    assert re.search(rf'^\s+{rb.ENV_VAR}: "1"$', workflow, re.MULTILINE), (
        f"slow-tests.yml no longer sets {rb.ENV_VAR}: a shard's memory is then the sum of its files'")
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert f'os.environ.get("{rb.ENV_VAR}") == "1"' in conftest
    assert "_release_between_files.register(config)" in conftest
