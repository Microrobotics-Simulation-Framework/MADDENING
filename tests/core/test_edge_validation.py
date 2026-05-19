"""Tests for v0.2 #4 compile-time edge validation.

These exercise the shape/dtype/units checks added to
:meth:`GraphManager.validate` and the routing of warnings through
:class:`maddening.warnings.EdgeValidationWarning` subclasses in
:meth:`GraphManager.compile`.
"""

from __future__ import annotations

import warnings as _w

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.warnings import (
    DtypeMismatchWarning,
    EdgeValidationWarning,
    ShapeMismatchWarning,
    UnitMismatchWarning,
)


# ---------------------------------------------------------------------------
# Helper nodes
# ---------------------------------------------------------------------------


class _ScalarSourceNode(SimulationNode):
    """Emits a single scalar in its state."""

    def initial_state(self):
        return {"out": jnp.array(1.0, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state


class _VectorSourceNode(SimulationNode):
    """Emits a vector of length 3."""

    def __init__(self, name, timestep, length=3, dtype=jnp.float32):
        super().__init__(name, timestep, length=length)
        self._len = length
        self._dt = dtype

    def initial_state(self):
        return {"out": jnp.zeros(self._len, dtype=self._dt)}

    def update(self, state, boundary_inputs, dt):
        return state


class _ScalarSinkNode(SimulationNode):
    """Declares a scalar BoundaryInputSpec."""

    def initial_state(self):
        return {"value": jnp.array(0.0, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        v = boundary_inputs.get("in", jnp.array(0.0))
        return {"value": v}

    def boundary_input_spec(self):
        return {
            "in": BoundaryInputSpec(
                shape=(), dtype=jnp.float32, expected_units="N",
            ),
        }


class _Vec3SinkNode(SimulationNode):
    """Declares a length-3 vector input."""

    def initial_state(self):
        return {"u": jnp.zeros(3, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("vec", jnp.zeros(3))
        return {"u": u}

    def boundary_input_spec(self):
        return {
            "vec": BoundaryInputSpec(shape=(3,), dtype=jnp.float32),
        }


# ---------------------------------------------------------------------------
# Default behaviour: clean graph passes silently
# ---------------------------------------------------------------------------


class TestCleanGraphSilent:
    def test_matching_shape_and_dtype_no_warnings(self):
        gm = GraphManager()
        gm.add_node(_ScalarSourceNode("src", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in")
        with _w.catch_warnings():
            _w.simplefilter("error", EdgeValidationWarning)
            gm.compile()  # would raise if any EdgeValidationWarning fired

    def test_matching_vector_shape(self):
        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=3))
        gm.add_node(_Vec3SinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "vec")
        with _w.catch_warnings():
            _w.simplefilter("error", EdgeValidationWarning)
            gm.compile()


# ---------------------------------------------------------------------------
# Shape mismatch
# ---------------------------------------------------------------------------


class TestShapeMismatch:
    def test_vector_into_scalar_warns(self):
        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=3))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in")
        with pytest.warns(ShapeMismatchWarning, match="shape"):
            gm.compile()

    def test_wrong_vector_length_warns(self):
        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=5))
        gm.add_node(_Vec3SinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "vec")
        with pytest.warns(ShapeMismatchWarning, match=r"\(5,\).*\(3,\)"):
            gm.compile()

    def test_transform_suppresses_shape_warning(self):
        # When a transform is provided we can't validate shape at
        # compile time (transform may reshape), so no warning.
        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=5))
        gm.add_node(_Vec3SinkNode("sink", timestep=0.01))
        gm.add_edge(
            "src", "sink", "out", "vec",
            transform=lambda x: x[:3],   # trim to 3
        )
        with _w.catch_warnings():
            _w.simplefilter("error", ShapeMismatchWarning)
            gm.compile()  # would raise if a ShapeMismatchWarning fired

    def test_warning_is_subclass_of_edge_validation(self):
        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=3))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in")
        with pytest.warns(EdgeValidationWarning):  # via the parent class
            gm.compile()


# ---------------------------------------------------------------------------
# Dtype mismatch
# ---------------------------------------------------------------------------


class TestDtypeMismatch:
    def test_int_into_float32_warns(self):
        # Source emits int32, sink expects float32
        class _IntSource(SimulationNode):
            def initial_state(self):
                return {"out": jnp.array(0, dtype=jnp.int32)}
            def update(self, s, b, dt): return s

        gm = GraphManager()
        gm.add_node(_IntSource("src", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in")
        with pytest.warns(DtypeMismatchWarning, match="dtype"):
            gm.compile()

    def test_transform_suppresses_dtype_warning(self):
        class _IntSource(SimulationNode):
            def initial_state(self):
                return {"out": jnp.array(0, dtype=jnp.int32)}
            def update(self, s, b, dt): return s

        gm = GraphManager()
        gm.add_node(_IntSource("src", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge(
            "src", "sink", "out", "in",
            transform=lambda x: x.astype(jnp.float32),
        )
        with _w.catch_warnings():
            _w.simplefilter("error", DtypeMismatchWarning)
            gm.compile()


# ---------------------------------------------------------------------------
# Unit mismatch — uses the new class
# ---------------------------------------------------------------------------


class TestUnitMismatch:
    def test_unit_mismatch_uses_specific_warning_class(self):
        gm = GraphManager()
        gm.add_node(_ScalarSourceNode("src", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in", source_units="kN")
        with pytest.warns(UnitMismatchWarning):
            gm.compile()


# ---------------------------------------------------------------------------
# Aggregation: every problem is surfaced in a single compile() pass
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_all_edge_problems_emitted_in_one_pass(self):
        """A graph with shape, dtype, and unit problems all on different
        edges should emit one warning per problem, not stop at the first."""
        class _BadSource(SimulationNode):
            def initial_state(self):
                return {"out": jnp.array(0, dtype=jnp.int32)}  # wrong dtype
            def update(self, s, b, dt): return s

        gm = GraphManager()
        gm.add_node(_BadSource("src1", timestep=0.01))
        gm.add_node(_VectorSourceNode("src2", timestep=0.01, length=5))  # wrong shape
        gm.add_node(_ScalarSourceNode("src3", timestep=0.01))  # ok shape; wrong units
        gm.add_node(_ScalarSinkNode("sink1", timestep=0.01))
        gm.add_node(_Vec3SinkNode("sink2", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink3", timestep=0.01))

        gm.add_edge("src1", "sink1", "out", "in")          # dtype
        gm.add_edge("src2", "sink2", "out", "vec")         # shape
        gm.add_edge("src3", "sink3", "out", "in", source_units="kN")  # units

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always", EdgeValidationWarning)
            gm.compile()

        kinds = {type(w.message).__name__ for w in caught
                 if issubclass(w.category, EdgeValidationWarning)}
        assert "ShapeMismatchWarning" in kinds
        assert "DtypeMismatchWarning" in kinds
        assert "UnitMismatchWarning" in kinds


# ---------------------------------------------------------------------------
# Errors continue to be hard failures
# ---------------------------------------------------------------------------


class TestHardErrors:
    def test_unknown_target_node_raises(self):
        gm = GraphManager()
        gm.add_node(_ScalarSourceNode("src", timestep=0.01))
        gm.add_edge("src", "ghost", "out", "in")
        with pytest.raises(RuntimeError, match="non-existent target"):
            gm.compile()

    def test_unknown_source_field_raises(self):
        gm = GraphManager()
        gm.add_node(_ScalarSourceNode("src", timestep=0.01))
        gm.add_node(_ScalarSinkNode("sink", timestep=0.01))
        gm.add_edge("src", "sink", "missing_field", "in")
        with pytest.raises(RuntimeError, match="not in state"):
            gm.compile()


# ---------------------------------------------------------------------------
# Spec dtype set to None: no dtype check fires
# ---------------------------------------------------------------------------


class TestSpecDtypeNone:
    def test_no_dtype_warning_when_spec_dtype_none(self):
        class _Sink(SimulationNode):
            def initial_state(self):
                return {"x": jnp.array(0.0)}
            def update(self, s, b, dt): return s
            def boundary_input_spec(self):
                return {"in": BoundaryInputSpec(shape=(), dtype=None)}

        class _IntSource(SimulationNode):
            def initial_state(self):
                return {"out": jnp.array(0, dtype=jnp.int32)}
            def update(self, s, b, dt): return s

        gm = GraphManager()
        gm.add_node(_IntSource("src", timestep=0.01))
        gm.add_node(_Sink("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "in")
        with _w.catch_warnings():
            _w.simplefilter("error", DtypeMismatchWarning)
            gm.compile()  # would raise if dtype warning fired


# ---------------------------------------------------------------------------
# Spec shape () means scalar — vector source warns; spec shape (-1,) leaves
# the dimension symbolic so no warning is generated.
# ---------------------------------------------------------------------------


class TestSymbolicShape:
    def test_negative_dim_in_spec_skips_shape_check(self):
        class _SymSink(SimulationNode):
            def initial_state(self):
                return {"u": jnp.zeros(0, dtype=jnp.float32)}
            def update(self, s, b, dt): return s
            def boundary_input_spec(self):
                return {"vec": BoundaryInputSpec(shape=(-1,), dtype=jnp.float32)}

        gm = GraphManager()
        gm.add_node(_VectorSourceNode("src", timestep=0.01, length=7))
        gm.add_node(_SymSink("sink", timestep=0.01))
        gm.add_edge("src", "sink", "out", "vec")
        with _w.catch_warnings():
            _w.simplefilter("error", ShapeMismatchWarning)
            gm.compile()  # symbolic dim shouldn't warn


# ---------------------------------------------------------------------------
# Backwards-compatible: old generic "WARNING:" still goes through UserWarning
# ---------------------------------------------------------------------------


class TestGenericWarning:
    def test_disconnected_node_still_emits_userwarning(self):
        class _Standalone(SimulationNode):
            def initial_state(self): return {"x": jnp.array(0.0)}
            def update(self, s, b, dt): return s

        gm = GraphManager()
        gm.add_node(_Standalone("alone", timestep=0.01))
        with pytest.warns(UserWarning, match="disconnected"):
            gm.compile()
