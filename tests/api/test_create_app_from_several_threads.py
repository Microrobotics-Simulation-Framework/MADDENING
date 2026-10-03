"""Apps built in several threads at once leave the warnings filters as they were.

FastAPI builds each route's fields inside ``warnings.catch_warnings()``,
which saves the process's warnings filters on the way in and puts them
back on the way out.  The filters are process-wide and the block is not
thread-safe, so when two threads each ran :meth:`SimulationServer.create_app`
at once, each put back the filters it had saved -- which could be the other
thread's, mid-block.  Two things followed (MADD-ANO-174):

* a warning FastAPI silences while it builds a route
  (pydantic's ``UnsupportedFieldAttributeWarning``) was shown, and raised
  where warnings are errors -- as under this suite's filter;
* ``ignore::UserWarning``, which FastAPI sets for a moment, could be left
  in the filters for good, so every MADDENING warning (each derives from
  ``UserWarning``) was silently dropped from then on.

``create_app`` now builds one app at a time.  The race is real but narrow,
so these tests widen it: every ``catch_warnings`` block is held open a
little longer before it puts the filters back, which makes the blocks of
apps built at once overlap on every run.  Nothing here sends a request.
"""

from __future__ import annotations

import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.warnings import UnitMismatchWarning

#: Threads building an app at once, and how long each ``catch_warnings``
#: block is held open before it puts the filters back.
THREADS = 4
HOLD = 0.001  # units: s


def _filters() -> list:
    return [(action, getattr(msg, "pattern", msg), category, getattr(mod, "pattern", mod),
             lineno) for action, msg, category, mod, lineno in warnings.filters]


@pytest.fixture
def held_open(monkeypatch):
    """Every ``warnings.catch_warnings`` block, in any thread, held open
    :data:`HOLD` seconds longer before it puts the filters back."""
    real_exit = warnings.catch_warnings.__exit__

    def exit_later(self, *exc_info):
        time.sleep(HOLD)
        return real_exit(self, *exc_info)

    monkeypatch.setattr(warnings.catch_warnings, "__exit__", exit_later)


def _build_at_once(servers) -> list:
    """Each server's ``create_app()`` in its own thread, released
    together; the exceptions they raised."""
    barrier = threading.Barrier(len(servers))
    errors: list = []

    def build(server):
        try:
            barrier.wait(30)
            server.create_app()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=build, args=(s,)) for s in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads), "an app was never built"
    return errors


def test_apps_built_in_several_threads_at_once_leave_the_warnings_filters_as_they_were(
        tmp_path, held_open):
    """Four apps built at once, three times over: no thread raises (a
    warning FastAPI silences is not shown, so not raised under this
    suite's error filter) and the process's filters afterwards are the
    ones before, with no ``ignore`` left behind."""
    servers = [SimulationServer({}, checkpoint_root=str(tmp_path / str(i)))
               for i in range(THREADS)]
    before = _filters()
    for _ in range(3):
        errors = _build_at_once(servers)
        assert not errors, [f"{type(e).__name__}: {e}" for e in errors]
        assert _filters() == before, [f for f in _filters() if f not in before]


def test_a_maddening_warning_still_reaches_the_caller_after_apps_built_at_once(
        tmp_path, held_open):
    """What a filter left behind would cost: after apps built at once, a
    MADDENING warning is still shown to the caller -- here as the error the
    suite's filter makes it -- not dropped by an ``ignore::UserWarning``
    nobody set."""
    servers = [SimulationServer({}, checkpoint_root=str(tmp_path / str(i)))
               for i in range(THREADS)]
    for _ in range(3):
        _build_at_once(servers)
    with warnings.catch_warnings():
        # The suite's filters, as the process holds them now, plus "error"
        # for this one category at the back: an ``ignore`` ahead of it
        # (left behind by the race) would still win.
        warnings.filterwarnings("error", category=UnitMismatchWarning, append=True)
        with pytest.raises(UnitMismatchWarning):
            warnings.warn(UnitMismatchWarning("probe: a unit mismatch"))


def test_create_app_holds_the_build_lock_while_it_builds(tmp_path, monkeypatch):
    """The lock the apps are built under is held for the whole build and
    let go after it, also when the build raises."""
    server = SimulationServer({}, checkpoint_root=str(tmp_path))
    seen: list = []

    def try_take():
        got = server_module._CREATE_APP_LOCK.acquire(blocking=False)
        if got:
            server_module._CREATE_APP_LOCK.release()
        seen.append(got)

    def build(self):
        # An RLock this thread holds cannot be taken by another thread.
        other = threading.Thread(target=try_take)
        other.start()
        other.join(10)
        raise RuntimeError("build failed")

    monkeypatch.setattr(SimulationServer, "_build_app", build)
    with pytest.raises(RuntimeError, match="build failed"):
        server.create_app()
    assert seen == [False], "another thread could take the lock mid-build"
    assert server_module._CREATE_APP_LOCK.acquire(blocking=False)
    server_module._CREATE_APP_LOCK.release()
