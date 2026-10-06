"""An external input runs in the dtype it was declared with, at every door.

``add_external_input(node, field)`` declares ``float32`` unless told
otherwise, also under ``jax_enable_x64``.  The declaration was applied to
the zeros of an omitted input, written to the saved configuration and
exported by the FMU as the variable's type -- and ignored by the step for a
value handed in: an x64 graph ran ``0.1`` as a float64 where its own FMU
ran ``float32(0.1)`` (0.5908093400306381 against 0.5908093402561657 after
ten steps of a float64 spring, nothing said).

Every step program now casts an external input to its declared dtype, and
a value the cast changes is reported once per input.  The battery here:
every entry point that takes ``external_inputs``, on a uniform, a coupled
and a multi-rate graph; and what is, and is not, reported.
"""
from __future__ import annotations

import contextlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

CAST_WARNING = ".*is declared .* does not hold as given"


@contextlib.contextmanager
def _x64(enabled: bool = True):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@stability(StabilityLevel.STABLE)
class _Lag(SimulationNode):
    """``x' = drive + ext - x`` in JAX's default float (float64 under x64):
    ``drive`` comes by an edge, ``ext`` and ``count`` from outside."""

    def initial_state(self):
        return {"x": jnp.asarray(0.25), "seen": jnp.asarray(0.0)}

    def boundary_input_spec(self):
        return {"drive": BoundaryInputSpec(shape=()), "ext": BoundaryInputSpec(shape=()),
                "count": BoundaryInputSpec(shape=())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        x = state["x"]
        ext = boundary_inputs.get("ext", 0.0)
        rate = boundary_inputs.get("drive", 0.0) + ext - x
        return {"x": x + dt * rate / 3, "seen": ext + jnp.zeros_like(x)}


def _graph(kind: str = "uniform", **declared) -> GraphManager:
    gm = GraphManager()
    gm.add_node(_Lag("a", 0.01))
    if kind != "single":
        gm.add_node(_Lag("b", 0.02 if kind == "multirate" else 0.01))
        gm.add_edge("a", "b", "x", "drive")
        gm.add_edge("b", "a", "x", "drive")
        gm.add_external_input("b", "ext", **declared)
    if kind == "coupled":
        gm.add_coupling_group(["a", "b"], max_iterations=4)
    gm.add_external_input("a", "ext", **declared)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="WARNING. cycle detected")
        gm.compile()
    return gm


def _ext(gm: GraphManager, value) -> dict:
    return {ei.target_node: {ei.target_field: value} for ei in gm._external_inputs}  # noqa: SLF001


def _bits(tree) -> dict:
    return {f"{n}.{f}": (np.asarray(v).dtype.name, np.asarray(v).tobytes())
            for n, fields in tree.items() if not n.startswith("_")
            for f, v in fields.items()}


ENTRY_POINTS = {
    "step": lambda gm, ext: [gm.step(external_inputs=ext) for _ in range(4)][-1],
    "run": lambda gm, ext: (gm.run(4, external_inputs=ext), gm._user_state(gm._state))[1],  # noqa: SLF001
    "run_scan": lambda gm, ext: gm.run_scan(4, external_inputs=ext),
    "run_scan_with_history": lambda gm, ext: gm.run_scan_with_history(4, external_inputs=ext)[1],
    "run_sweep": lambda gm, ext: gm.run_sweep(
        3, {n: {f: jnp.stack([v, v + 1]) for f, v in fields.items()}
            for n, fields in gm._user_state(gm._state).items()},  # noqa: SLF001
        external_inputs=ext),
    "run_adaptive": lambda gm, ext: gm.run_adaptive(0.04, external_inputs=ext),
    "run_adaptive_scan": lambda gm, ext: gm.run_adaptive_scan(
        0.04, max_steps=16, external_inputs=ext),
    "compiled step": lambda gm, ext: gm._compiled_step(gm._state, ext, gm.params),  # noqa: SLF001
}

CASES = ([("uniform", e) for e in ENTRY_POINTS]
         + [(k, e) for k in ("coupled", "multirate", "single")
            for e in ("step", "run_scan", "compiled step")])


def _result(out):
    """The state an entry point answered with (some answer a tuple or an
    object carrying it)."""
    for attr in ("final_state", "state"):
        if hasattr(out, attr):
            out = getattr(out, attr)
    if isinstance(out, tuple):
        out = out[0]
    return _bits(out)


def _run(kind: str, entry: str, value, **declared) -> dict:
    gm = _graph(kind, **declared)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=CAST_WARNING)
        return _result(ENTRY_POINTS[entry](gm, _ext(gm, value)))


@pytest.mark.parametrize("kind, entry", CASES)
def test_an_x64_graph_runs_a_supplied_value_in_the_declared_dtype(kind, entry):
    """The default (float32) declaration on a float64 graph: ``0.1`` handed
    in as a Python float, a ``numpy.float64`` or a float64 array runs as
    ``float32(0.1)`` -- what the FMU's Float32 variable holds -- and not as
    the float64 a float64 declaration runs.  It used to run as the
    float64 at every one of these doors."""
    with _x64():
        narrowed = _run(kind, entry, np.float32(0.1))
        for value in (0.1, np.float64(0.1), jnp.asarray(0.1, jnp.float64)):
            assert _run(kind, entry, value) == narrowed, type(value).__name__
        wide = _run(kind, entry, 0.1, dtype=jnp.float64)
        assert wide != narrowed
        assert wide == _run(kind, entry, np.float64(0.1), dtype=jnp.float64)
        # the state is float64 either way: only the input was narrowed
        assert {dtype for dtype, _ in narrowed.values()} == {"float64"}


@pytest.mark.parametrize("kind", ["uniform", "coupled", "multirate"])
def test_the_step_reads_the_value_the_declared_dtype_holds(kind):
    """The node sees the cast value itself: ``seen`` is the input as the
    update received it."""
    with _x64():
        gm = _graph(kind)
        with pytest.warns(UserWarning, match=CAST_WARNING):
            out = gm.step(external_inputs=_ext(gm, 0.1))
        assert float(out["a"]["seen"]) == float(np.float32(0.1)) != 0.1
        gm64 = _graph(kind, dtype=jnp.float64)
        assert float(gm64.step(external_inputs=_ext(gm64, 0.1))["a"]["seen"]) == 0.1


def test_a_value_the_cast_changes_is_reported_once_per_input():
    with _x64():
        gm = _graph("uniform")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                gm.step(external_inputs=_ext(gm, 0.1))
            gm.run_scan(2, external_inputs=_ext(gm, 0.3))
        said = [str(w.message) for w in caught if "does not hold as given" in str(w.message)]
        assert len(said) == 2, said                      # one per input, not per step
        assert sorted(m.split("'")[1] for m in said) == ["a.ext", "b.ext"]
        for message in said:
            assert "declared float32" in message and "a float64 value" in message
            assert "dtype=" in message and "before 0.4.0" in message


@pytest.mark.parametrize("value", [0.5, 0.0, -2.0, 3, True, float("inf"), float("nan"),
                                   np.float32(0.1), np.float64(0.25),
                                   float(np.float32(0.1))])
def test_a_value_the_declared_dtype_holds_exactly_is_not_reported(value):
    """Nothing changed for such a value, so nothing is said: the cast is
    judged by the value, not by the dtype it arrived in."""
    with _x64():
        gm = _graph("single")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = gm.step(external_inputs=_ext(gm, value))
        np.testing.assert_array_equal(np.asarray(out["a"]["seen"]), np.float64(value))


@pytest.mark.parametrize("value", [0.1, np.float64(0.1), 1e39, 16777217, 1 + 2j])
def test_without_x64_only_a_value_jax_would_not_have_narrowed_itself_is_reported(value):
    """Without x64 JAX takes a float64 ``0.1`` in as float32 before the
    graph sees it, as it always did: no number changed, nothing is said.
    An integer float32 cannot hold, or a complex number, is another
    matter."""
    with _x64(False):
        gm = _graph("single")
        changed = value in (16777217, 1 + 2j)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gm.step(external_inputs=_ext(gm, value))
        said = [w for w in caught if "does not hold as given" in str(w.message)]
        assert bool(said) is changed, [str(w.message) for w in caught]


@pytest.mark.parametrize("dtype, value, changed", [
    (jnp.int32, 2, False), (jnp.int32, 2.0, False), (jnp.int32, 0.5, True),
    (jnp.int32, 2 ** 31, True), (jnp.bool_, True, False), (jnp.bool_, 1, False),
    (jnp.bool_, 2, True), (jnp.bool_, 0.5, True), (jnp.float32, 16777216, False),
    (jnp.float32, 16777217, True), (jnp.float64, 2 ** 53 + 1, True),
    (jnp.float64, 0.1, False), (jnp.float16, 0.1, True), (jnp.float16, 0.5, False),
])
def test_every_declared_kind_is_cast_and_reports_only_a_changed_value(dtype, value, changed):
    """Integer, Boolean and float declarations of every width: the update
    receives the declared dtype, and the report is exactly "the cast
    changed the value"."""
    with _x64():
        gm = _graph("single", dtype=dtype)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            traced = jax.make_jaxpr(
                lambda e: gm._compiled_step(gm._state, e, gm.params))(  # noqa: SLF001
                    _ext(gm, value))
            gm._resolve_external_inputs(_ext(gm, value))                # noqa: SLF001
        said = [w for w in caught if "does not hold as given" in str(w.message)]
        assert bool(said) is changed, [str(w.message) for w in caught]
        # converted in the program, unless it arrived in the dtype already
        assert (f"new_dtype={jnp.dtype(dtype).name}" in str(traced)) is (
            not isinstance(value, bool))


def test_a_traced_value_is_judged_by_its_dtype():
    """Under a transformation there is no number to compare: a traced
    float64 into a float32 input is reported, a traced float32 is not."""
    with _x64():
        for dtype, changed in ((jnp.float64, True), (jnp.float32, False)):
            gm = _graph("single")

            def loss(v, gm=gm):
                out = gm._compiled_step(                                # noqa: SLF001
                    gm._state, gm._resolve_external_inputs(_ext(gm, v)), gm.params)  # noqa: SLF001
                return out["a"]["x"]

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                grad = jax.grad(loss)(jnp.asarray(0.1, dtype))
            said = [w for w in caught if "does not hold as given" in str(w.message)]
            assert bool(said) is changed and (not changed or "a traced float64" in str(said[0].message))
            assert grad.dtype == jnp.dtype(dtype) and float(grad) > 0


def test_an_input_of_the_declared_dtype_adds_nothing_to_the_step_program():
    """The cast is free where it changes nothing: stepped with arrays of
    the declared dtype (the zeros of an omitted input among them) the
    program holds no conversion of the input."""
    gm = _graph("single")
    ext = gm._resolve_external_inputs(None)                             # noqa: SLF001
    text = str(jax.make_jaxpr(lambda e: gm._compiled_step(gm._state, e, gm.params))(ext))  # noqa: SLF001
    assert "convert_element_type" not in text, text


def test_an_omitted_and_an_undeclared_input_are_handled_as_before():
    with _x64():
        gm = _graph("uniform")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            partial = gm._resolve_external_inputs({"a": {"ext": np.float32(0.1)}})  # noqa: SLF001
        assert partial["b"]["ext"].dtype == jnp.float32 and float(partial["b"]["ext"]) == 0.0
        with pytest.raises(ValueError, match="does not declare"):
            gm.step(external_inputs={"a": {"nope": 0.1}})


def test_a_reloaded_graph_casts_as_the_graph_it_was_saved_from():
    """The saved configuration carries the declared dtype, so the reloaded
    graph runs the same number."""
    with _x64():
        for declared in ({}, {"dtype": jnp.float64}):
            gm = _graph("single", **declared)
            again = GraphManager.from_dict(gm.to_dict(), node_registry={"_Lag": _Lag})
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=CAST_WARNING)
                want = _bits(gm.step(external_inputs=_ext(gm, 0.1)))
                assert _bits(again.step(external_inputs=_ext(again, 0.1))) == want
