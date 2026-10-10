"""Shared scaffolding for the differential state-and-I/O harness.

Every oracle in ``test_differential_*.py`` compares two paths that must agree
**bit for bit**: the same compiled computation reached two ways (a write and
a reload, a checkpoint and an uninterrupted run, a graph and its
serialisation, an FMU transport and the graph it serves).  The comparisons
here are therefore exact and NaN-safe: two leaves agree when they have the
same dtype, the same shape and the same bytes, so a NaN that both paths
compute the same way agrees and a ``-0.0`` against ``0.0`` does not.  No
oracle in the harness takes a tolerance except where a docstring says so
and why.

Nothing here may reach a cloud provider.  :func:`no_cloud_launch` is the
module-scoped guard every harness module applies: ``HOME`` is an empty
scratch directory, cloud credentials are unset, and every launcher the REST
server could reach raises if called.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterator, Optional

import jax
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager

#: Environment variables that could hand a launcher credentials.
CLOUD_CREDENTIALS = ("RUNPOD_API_KEY", "LAMBDA_API_KEY", "AWS_ACCESS_KEY_ID",
                     "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                     "GOOGLE_APPLICATION_CREDENTIALS", "SKYPILOT_API_SERVER_ENDPOINT")


class CloudLaunchAttempted(AssertionError):
    """Raised by the stubbed launchers: the harness must never get here."""


@contextlib.contextmanager
def no_cloud_launch() -> Iterator[None]:
    """Empty ``HOME``, no credentials, every launcher stubbed to raise.

    The REST route catches whatever a launcher raises and answers 500, so a
    stub that only raised could be swallowed; every attempt is also recorded
    and the guard fails on exit if there was one.
    """
    attempts: list[str] = []

    def refuse(where: str):
        def stub(*args, **kwargs):
            attempts.append(where)
            raise CloudLaunchAttempted(
                f"the differential harness reached {where}; it must not")
        return stub

    with tempfile.TemporaryDirectory(prefix="maddening-diff-home-") as home, \
            pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", home)
        for var in CLOUD_CREDENTIALS:
            mp.delenv(var, raising=False)
        for target in ("maddening.cloud._skypilot.launch_vm",
                       "maddening.cloud.session.CloudSession.__init__",
                       "maddening.cloud.session.CloudSession.launch"):
            with contextlib.suppress(ImportError, AttributeError):
                mp.setattr(target, refuse(target))
        yield
    assert not attempts, f"cloud launch attempted: {attempts}"


def note(message: str) -> None:
    """``hypothesis.note`` inside a property, ``print`` outside one: the
    oracles here are called from pinned reproducers too, where Hypothesis
    refuses a note (and a strict xfail would then pass for the wrong
    reason)."""
    from hypothesis import note as hypothesis_note
    from hypothesis.errors import InvalidArgument

    try:
        hypothesis_note(message)
    except InvalidArgument:
        print(message)


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------

def _host(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def leaves_identical(expected: Any, actual: Any) -> Optional[str]:
    """``None`` when two array leaves are identical (dtype, shape, bytes),
    else a one-line description of the difference."""
    e, a = _host(expected), _host(actual)
    if e.dtype != a.dtype:
        return f"dtype {e.dtype} vs {a.dtype}"
    if e.shape != a.shape:
        return f"shape {e.shape} vs {a.shape}"
    if e.tobytes() == a.tobytes():
        return None
    with np.errstate(all="ignore"):
        if np.issubdtype(e.dtype, np.floating):
            diff = np.nanmax(np.abs(e.astype(np.float64) - a.astype(np.float64))) \
                if e.size else 0.0
            return f"max |difference| {diff:.6g}; expected {e!r}, got {a!r}"
    return f"expected {e!r}, got {a!r}"


def assert_trees_identical(expected: dict, actual: dict, *, what: str) -> None:
    """Two ``{owner: {field: array}}`` trees are the same tree, bit for bit."""
    assert set(expected) == set(actual), (
        f"{what}: owners differ -- only expected {sorted(set(expected) - set(actual))}, "
        f"only actual {sorted(set(actual) - set(expected))}")
    for owner in expected:
        e, a = expected[owner], actual[owner]
        assert set(e) == set(a), f"{what}[{owner!r}]: fields {sorted(e)} vs {sorted(a)}"
        for key in e:
            problem = leaves_identical(e[key], a[key])
            assert problem is None, f"{what}[{owner!r}][{key!r}]: {problem}"


def states(gm: GraphManager) -> dict:
    """Every node's user-visible state, copied to the host."""
    return {name: {k: _host(v) for k, v in gm.get_node_state(name).items()}
            for name in gm.node_names}


def full_state(gm: GraphManager) -> dict:
    """``gm._state`` including ``_meta``, copied to the host."""
    return {name: {k: _host(v) for k, v in fields.items()}
            for name, fields in gm._state.items()  # noqa: SLF001
            if isinstance(fields, dict)}


def params_tree(gm: GraphManager) -> dict:
    """``gm.params`` flattened to ``{"<section>/<owner>": {key: array}}``."""
    return {f"{section}/{owner}": {k: _host(v) for k, v in leaves.items()}
            for section, owners in gm.params.items() if isinstance(owners, dict)
            for owner, leaves in owners.items() if isinstance(leaves, dict)}


def rollout(gm: GraphManager, n_steps: int) -> dict:
    """``n_steps`` of the jitted single step, and the state reached.

    ``run`` re-enters the same compiled step each time, so a rollout split
    anywhere is bit-identical to an unsplit one; a ``run_scan`` of another
    length is another XLA program and may land an ulp away
    (``tests/property/test_round_trips.py::test_a_split_rollout_is_step_for_step_identical``).
    """
    gm.run(n_steps)
    return states(gm)


# ---------------------------------------------------------------------------
# Reloads
# ---------------------------------------------------------------------------

def json_config(gm: GraphManager) -> dict:
    """``gm.to_dict()`` through JSON text, as a saved file would carry it."""
    return json.loads(json.dumps(gm.to_dict(), allow_nan=True))


#: ``compile()``'s advisory for a group that declares a dead band on three
#: or more members (MADD-ANO-254), as a pattern for a warnings filter: the
#: pattern of ``strategies.DRAWN_DEAD_BAND``, kept here so that this module
#: imports no strategy.  A drawn graph can carry such a group, and whoever
#: compiles it a second time is advised a second time.
DEAD_BAND_ON_MEMBERS = r"(?s).*declares a dead band on \d+ members"


def reload_from_config(config: dict, registry: dict) -> GraphManager:
    """A compiled graph rebuilt from ``config``.

    The config carries a group's ``atol`` like every other field, so a
    group that declares a dead band on three or more members is advised
    on again when its reload compiles: expected here by name, as
    ``strategies.GraphRecipe.build`` expects it at the first build.
    Nothing else is filtered."""
    gm = GraphManager.from_dict(config, registry)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=DEAD_BAND_ON_MEMBERS, category=UserWarning)
        gm.compile()
    return gm


def checkpoint_path(directory: str | Path, stem: str = "checkpoint") -> Path:
    return Path(directory) / f"{stem}.npz"


# ---------------------------------------------------------------------------
# What a refused write must leave alone
# ---------------------------------------------------------------------------

def deep_copy_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: deep_copy_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(deep_copy_jsonable(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def graph_snapshot(gm: GraphManager) -> dict:
    """Everything a write could touch: every node's ``params``, the live
    ``gm.params`` leaves, the full state, the dirty flag and the compile
    generation."""
    return {
        "node_params": {name: deep_copy_jsonable(dict(spec.node.params))
                        for name, spec in gm._nodes.items()},  # noqa: SLF001
        "params": params_tree(gm),
        "state": full_state(gm),
        "dirty": gm._dirty,  # noqa: SLF001
    }


def _same_python(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return leaves_identical(np.asarray(a), np.asarray(b)) is None
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_python(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            _same_python(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return type(a) is type(b) and a == b


def assert_nothing_written(gm: GraphManager, before: dict, *, what: str) -> None:
    after = graph_snapshot(gm)
    for name, params in before["node_params"].items():
        now = after["node_params"].get(name)
        assert now is not None and _same_python(params, now), (
            f"{what}: node {name!r} params changed: {params!r} -> {now!r}")
    assert_trees_identical(before["params"], after["params"], what=f"{what}: gm.params")
    assert_trees_identical(before["state"], after["state"], what=f"{what}: state")
    assert after["dirty"] == before["dirty"], (
        f"{what}: the dirty flag moved {before['dirty']} -> {after['dirty']}")


def canonical(value: Any) -> Any:
    """JSON-decoded ``value`` with NaN replaced by a sentinel, for ``==``."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float) and value != value:
        return "<nan>"
    return value


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Warnings off for a reload the oracle has already judged.

    Used only around steps whose warnings are not the property under test
    (a reload of a config whose own warnings are asserted elsewhere).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def tmp_dir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="maddening-diff-")


def env_report() -> str:
    """The configuration a measurement is taken in (common brief)."""
    import jaxlib

    return (f"jax {jax.__version__}, jaxlib {jaxlib.__version__}, "
            f"affinity {sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'}")
