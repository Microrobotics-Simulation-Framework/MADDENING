"""Tests for the TransformRegistry and built-in transforms."""

import jax.numpy as jnp
import pytest

from maddening.core.transforms import (
    _TRANSFORM_REGISTRY,
    UnregisteredTransformError,
    extract_first,
    extract_last,
    extract_second,
    extract_second_last,
    get_transform_name,
    identity,
    is_registered,
    list_transforms,
    negate,
    register_transform,
    resolve_transform,
    scale,
    validate_all_registered,
)
from maddening.core.edge import EdgeSpec


class TestRegistration:
    """Tests for @register_transform decorator."""

    def test_register_and_resolve_by_name(self):
        @register_transform("test_reg_1")
        def my_fn(x):
            return x + 1
        assert resolve_transform("test_reg_1") is my_fn

    def test_register_sets_attribute(self):
        @register_transform("test_reg_2")
        def my_fn(x):
            return x * 2
        assert my_fn._transform_name == "test_reg_2"

    def test_duplicate_same_fn_ok(self):
        @register_transform("test_reg_3")
        def my_fn(x):
            return x
        # Re-registering the same function is fine
        register_transform("test_reg_3")(my_fn)

    def test_duplicate_different_fn_raises(self):
        @register_transform("test_reg_4")
        def fn_a(x):
            return x
        with pytest.raises(ValueError, match="already registered"):
            @register_transform("test_reg_4")
            def fn_b(x):
                return x + 1

    def test_resolve_unknown_name_raises(self):
        with pytest.raises(KeyError, match="not found"):
            resolve_transform("nonexistent_transform_xyz")

    def test_resolve_none_returns_none(self):
        assert resolve_transform(None) is None

    def test_resolve_callable_returns_callable(self):
        fn = lambda x: x
        assert resolve_transform(fn) is fn

    def test_get_transform_name_registered(self):
        assert get_transform_name(extract_first) == "extract_first"

    def test_get_transform_name_unregistered(self):
        assert get_transform_name(lambda x: x) is None

    def test_is_registered(self):
        assert is_registered(extract_first)
        assert not is_registered(lambda x: x)

    def test_list_transforms(self):
        transforms = list_transforms()
        assert "extract_first" in transforms
        assert "extract_last" in transforms
        assert "negate" in transforms
        assert "identity" in transforms


class TestBuiltinTransforms:
    """Tests for built-in registered transforms."""

    def test_extract_first(self):
        arr = jnp.array([10.0, 20.0, 30.0])
        assert float(extract_first(arr)) == 10.0

    def test_extract_last(self):
        arr = jnp.array([10.0, 20.0, 30.0])
        assert float(extract_last(arr)) == 30.0

    def test_extract_second(self):
        arr = jnp.array([10.0, 20.0, 30.0])
        assert float(extract_second(arr)) == 20.0

    def test_extract_second_last(self):
        arr = jnp.array([10.0, 20.0, 30.0])
        assert float(extract_second_last(arr)) == 20.0

    def test_negate(self):
        assert float(negate(jnp.array(5.0))) == -5.0

    def test_identity(self):
        x = jnp.array(42.0)
        assert float(identity(x)) == 42.0

    def test_scale_factory(self):
        fn = scale(2.5)
        assert float(fn(jnp.array(4.0))) == 10.0
        assert is_registered(fn)
        assert get_transform_name(fn) == "scale_2.5"

    def test_scale_reuse(self):
        fn1 = scale(3.0)
        fn2 = scale(3.0)
        assert fn1 is fn2  # Same factor returns same function


class TestValidation:
    """Tests for validate_all_registered."""

    def test_all_registered(self):
        edges = [
            EdgeSpec("a", "b", "x", "y", transform=extract_first),
            EdgeSpec("a", "b", "x", "y", transform=negate),
            EdgeSpec("a", "b", "x", "y", transform=None),
        ]
        warnings = validate_all_registered(edges)
        assert warnings == []

    def test_unregistered_warned(self):
        edges = [
            EdgeSpec("a", "b", "x", "y", transform=lambda x: x + 1),
        ]
        warnings = validate_all_registered(edges)
        assert len(warnings) == 1
        assert "unregistered" in warnings[0].lower()

    def test_mixed_registered_and_unregistered(self):
        edges = [
            EdgeSpec("a", "b", "x", "y", transform=extract_first),
            EdgeSpec("a", "b", "x", "y", transform=lambda x: x),
            EdgeSpec("a", "b", "x", "y", transform=None),
        ]
        warnings = validate_all_registered(edges)
        assert len(warnings) == 1


class TestGraphManagerIntegration:
    """Test that string transform names work in GraphManager.add_edge."""

    def test_string_transform_in_add_edge(self):
        from maddening.core.graph_manager import GraphManager
        from maddening.nodes.heat import HeatNode

        gm = GraphManager()
        gm.add_node(HeatNode("a", 0.001, n_cells=5,
                              initial_temperature=100.0))
        gm.add_node(HeatNode("b", 0.001, n_cells=5,
                              initial_temperature=0.0))
        # Use string name — should resolve to the registered function
        gm.add_edge("a", "b", "temperature", "left_temperature",
                     transform="extract_last")
        gm.add_edge("b", "a", "temperature", "right_temperature",
                     transform="extract_first")
        gm.add_coupling_group(["a", "b"], max_iterations=10,
                               tolerance=1e-8)
        gm.compile()
        state = gm.run_scan(10)
        assert jnp.all(jnp.isfinite(state["a"]["temperature"]))

    def test_unknown_string_transform_raises(self):
        from maddening.core.graph_manager import GraphManager
        from maddening.nodes.spring import SpringDamperNode

        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01))
        gm.add_node(SpringDamperNode("b", 0.01))
        with pytest.raises(KeyError, match="not found"):
            gm.add_edge("a", "b", "position", "anchor_position",
                         transform="this_does_not_exist")
