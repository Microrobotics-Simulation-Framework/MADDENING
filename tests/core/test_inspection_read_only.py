"""Every inspection method leaves every kind of graph exactly as it found it.

``format_graph`` / ``print_graph``, ``to_mermaid`` / ``to_dot``,
``state_summary``, ``params_table``, ``coupling_report``,
``memory_estimate`` and their ``print_*`` forms promise to be strictly
read-only: no write to the state, ``_meta``, ``params``, node
parameters, the dirty / compiled flags, the schedule or a cache, and no
compile or trace of the step.  The guard (``inspection_guard_support``)
fingerprints every attribute of the graph around each call and records
JAX's compile events; the second half of this file shows the guard fails
on each way a method could break the promise.

Two methods read through code that dispatches eager ``jax.numpy``
operations, which JAX compiles once per shape: ``coupling_report``
(through ``coupling_diagnostics``' float floor) and ``params_table`` on a
never-compiled graph (through each node's ``params_pytree()``).  Their
*first* call may fire compile events and nothing else; every method's
*second* call must fire none, with one exception.

The exception: on a coupling group one of whose internal edges has a
transform or a mapping, **every** call of ``coupling_report`` traces that
edge's rule once, under ``jax.eval_shape``, for the shape and dtype it
delivers (the float floor reads them).  The step is not compiled or
traced again and no attribute of the graph changes, but a transform is
user code and runs as Python once per report, so one with a side effect
shows it.  That is held here as it is
(``test_coupling_report_traces_an_internal_edges_rule_once_a_call``, and
the ``coupled_transformed`` kind of the matrix): a cache of what an edge
delivers would turn that test over, and with it the sentence in
``docs/user_guide/inspection.md``.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core import inspection
from tests.core.inspection_graphs import (
    BUILDERS,
    TRACES_AN_EDGE,
    TRANSFORM_RUNS,
    UNCOMPILED,
    build,
    graph,
)
from tests.core.inspection_guard_support import (
    INSPECTION_CALLS,
    TRACE_EVENT,
    assert_read_only,
    compile_events,
    eager_first_call,
    read_only_call,
    unavailable,
)


METHODS = INSPECTION_CALLS


def _eager_first_call(method: str, kind: str) -> bool:
    return eager_first_call(method, uncompiled=kind in UNCOMPILED)


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_inspection_method_changes_nothing_and_compiles_nothing(kind, method):
    if (reason := unavailable(method)) is not None:
        pytest.skip(reason)
    gm = graph(kind)
    call = METHODS[method]
    assert_read_only(gm, call, allow_eager_compile=_eager_first_call(method, kind))
    if kind in TRACES_AN_EDGE and "coupling_report" in method:
        # Warm, and the one exception: the edges' rules are traced again,
        # and nothing is lowered or compiled.
        _result, events = read_only_call(gm, call)
        assert events and all(e.endswith(TRACE_EVENT) for e in events), sorted(set(events))
    else:
        assert_read_only(gm, call)   # warm: no exception for anyone else


def test_coupling_report_traces_an_internal_edges_rule_once_a_call():
    """What the guide says of ``coupling_report()`` on a group with a
    transform or a mapping on an internal edge, pinned as it is: each call
    runs each such edge's transform once as Python (a trace of the edge's
    rule: a transform on one edge here, a mapping then a transform on the
    other), fires trace events and nothing else, and leaves the graph and
    the step's trace count alone.  No other inspection method runs a
    transform.  A cache of what an edge delivers makes the first assertion
    fail: change it and the guide's sentence together."""
    gm = graph("coupled_transformed")
    step_traces = gm.trace_count
    assert step_traces == 1
    METHODS["coupling_report"](gm)                       # the first call's eager compiles
    for _ in range(3):
        before = dict(TRANSFORM_RUNS)
        rows, events = read_only_call(gm, METHODS["coupling_report"])
        assert {k: TRANSFORM_RUNS[k] - before[k] for k in before} == {"last": 1, "first": 1}
        assert events and all(e.endswith(TRACE_EVENT) for e in events), sorted(set(events))
        assert len(rows) == 1 and gm.trace_count == step_traces
    before = dict(TRANSFORM_RUNS)
    for name, call in METHODS.items():
        if "coupling_report" not in name and unavailable(name) is None:
            assert_read_only(gm, call, allow_eager_compile=eager_first_call(name, uncompiled=False))
            assert TRANSFORM_RUNS == before, name
    # A group with plain edges fires nothing on a warm report.
    plain = graph("coupled")
    METHODS["coupling_report"](plain)
    assert read_only_call(plain, METHODS["coupling_report"])[1] == []


def test_a_traced_graph_stays_traced_after_every_method():
    """The escaped-tracer graph is reported as it stands, never put back:
    putting it back is a write (and ``coupling_diagnostics`` would do it)."""
    gm = graph("after_grad")
    for name, call in METHODS.items():
        if unavailable(name) is not None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # the recovery warning would fire here
            call(gm)
        assert gm._state_traced, name          # noqa: SLF001


def test_an_uncompiled_graph_stays_uncompiled_after_every_method():
    gm = graph("single_uncompiled")
    for name, call in METHODS.items():
        if unavailable(name) is not None:
            continue
        call(gm)
        assert gm._compiled_step is None and gm._dirty, name     # noqa: SLF001
        assert gm.params == {"nodes": {}, "mappings": {}}, name


# ----------------------------------------------------------------------
# The guard can fail: each way a method could break the promise
# ----------------------------------------------------------------------

@pytest.fixture
def fresh_single():
    return build("single")


def _caught(gm, mutant, *, allow_eager_compile=False) -> str:
    with pytest.raises(AssertionError) as info:
        assert_read_only(gm, mutant, allow_eager_compile=allow_eager_compile)
    return str(info.value)


def test_guard_catches_a_state_leaf_written(fresh_single):
    new = jnp.float32(9.0)
    message = _caught(fresh_single, lambda gm: gm._state["s"].__setitem__("position", new))
    assert "_state" in message


def test_guard_catches_a_state_leaf_replaced_by_an_equal_copy(fresh_single):
    """Same bytes, different array: still a write to the graph."""
    copy = jnp.array(fresh_single._state["s"]["position"], copy=True)
    message = _caught(fresh_single, lambda gm: gm._state["s"].__setitem__("position", copy))
    assert "_state" in message


def test_guard_catches_the_dirty_flag_flipped(fresh_single):
    message = _caught(fresh_single, lambda gm: setattr(gm, "_dirty", True))
    assert "_dirty" in message


def test_guard_catches_a_compile(fresh_single):
    message = _caught(fresh_single, lambda gm: gm.compile(), allow_eager_compile=True)
    assert "_compile_generation" in message and "_compiled_step" in message


def test_guard_catches_a_step(fresh_single):
    message = _caught(fresh_single, lambda gm: gm.step(), allow_eager_compile=True)
    assert "_state" in message


def test_guard_catches_a_trace_counter_bumped(fresh_single):
    message = _caught(fresh_single, lambda gm: setattr(gm, "_n_traces", gm._n_traces + 1))
    assert "<trace_count>" in message


def test_guard_catches_a_fresh_jit_compile(fresh_single):
    """Nothing on the graph moves; the compile event alone fails it."""
    x = jnp.arange(7.0)
    message = _caught(fresh_single, lambda gm: jax.jit(lambda v: v * 3.0 + 1.0)(x))
    assert "compiled or traced" in message


def test_guard_catches_a_compile_where_only_a_trace_is_allowed(fresh_single):
    """The allowance for a traced edge admits a trace and nothing more."""
    x = jnp.arange(11.0)
    message = _caught_tracing(fresh_single, lambda gm: jax.jit(lambda v: v * 5.0 - 1.0)(x))
    assert "compiled or lowered" in message
    assert_read_only(fresh_single, lambda gm: jax.eval_shape(lambda v: v[-1], x), allow_trace=True)
    message = _caught_tracing(fresh_single, lambda gm: setattr(gm, "_n_traces", gm._n_traces + 1))
    assert "<trace_count>" in message


def _caught_tracing(gm, mutant) -> str:
    with pytest.raises(AssertionError) as info:
        assert_read_only(gm, mutant, allow_trace=True)
    return str(info.value)


def test_guard_catches_a_params_leaf_coerced_in_place():
    gm = build("single")
    gm.params["nodes"]["s"]["damping"] = 3.0          # a Python float, as a user writes it
    # ``check_params`` coerces such a leaf in place (``_params_or_default``):
    # a write the read-only methods must not make.
    message = _caught(gm, lambda g: g.check_params(), allow_eager_compile=True)
    assert "params" in message


def test_guard_catches_a_node_params_dict_edited(fresh_single):
    message = _caught(fresh_single,
                      lambda gm: gm.get_node("s").params.__setitem__("extra", 1.0))
    assert "_nodes" in message


def test_guard_catches_the_escaped_tracers_put_back():
    """``coupling_diagnostics`` recovers a traced graph -- a write."""
    gm = build("after_grad")

    def recover(g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g.coupling_diagnostics()

    message = _caught(gm, recover, allow_eager_compile=True)
    assert "_state" in message and "_state_traced" in message


def test_guard_catches_a_coupling_report_that_ignores_the_tracers(monkeypatch):
    """The regression the report's tracer check exists for: without it,
    ``coupling_report`` reads ``coupling_diagnostics`` on a traced graph,
    which puts the graph back."""
    gm = build("after_grad")
    real = inspection._status
    monkeypatch.setattr(inspection, "_status",
                        lambda g: real(g).__class__(real(g).ever_compiled, real(g).stale, False))

    def report(g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g.coupling_report()

    message = _caught(gm, report, allow_eager_compile=True)
    assert "_state_traced" in message


def test_compile_event_recorder_sees_eager_and_jitted_compiles():
    """The recorder the guard leans on is live: a never-seen shape compiles."""
    with compile_events() as events:
        jnp.cumsum(jnp.arange(13.0) * 2.5)
    assert events
    with compile_events() as events:
        pass
    assert not events
