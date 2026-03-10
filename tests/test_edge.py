"""Tests for EdgeSpec."""

import jax.numpy as jnp

from maddening.core.edge import EdgeSpec


class TestEdgeSpec:
    def test_basic_creation(self):
        e = EdgeSpec("a", "b", "x", "y")
        assert e.source_node == "a"
        assert e.target_node == "b"
        assert e.source_field == "x"
        assert e.target_field == "y"
        assert e.transform is None

    def test_frozen(self):
        e = EdgeSpec("a", "b", "x", "y")
        import pytest
        with pytest.raises(AttributeError):
            e.source_node = "c"

    def test_with_transform(self):
        fn = lambda x: x * 2
        e = EdgeSpec("a", "b", "x", "y", transform=fn)
        assert e.transform is fn

    def test_to_dict_no_transform(self):
        e = EdgeSpec("a", "b", "x", "y")
        d = e.to_dict()
        assert d == {
            "source_node": "a",
            "target_node": "b",
            "source_field": "x",
            "target_field": "y",
        }
        assert "transform" not in d

    def test_to_dict_with_transform(self):
        def my_transform(x):
            return x
        e = EdgeSpec("a", "b", "x", "y", transform=my_transform)
        d = e.to_dict()
        assert "transform" in d
        assert "my_transform" in d["transform"]

    def test_repr(self):
        e = EdgeSpec("nodeA", "nodeB", "pos", "surface")
        r = repr(e)
        assert "nodeA.pos" in r
        assert "nodeB.surface" in r

    def test_equality(self):
        e1 = EdgeSpec("a", "b", "x", "y")
        e2 = EdgeSpec("a", "b", "x", "y")
        assert e1 == e2

    def test_hashable(self):
        e1 = EdgeSpec("a", "b", "x", "y")
        e2 = EdgeSpec("a", "b", "x", "y")
        assert hash(e1) == hash(e2)
        s = {e1, e2}
        assert len(s) == 1
