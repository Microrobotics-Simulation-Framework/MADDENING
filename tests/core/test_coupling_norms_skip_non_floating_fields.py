"""Every coupling norm decides a field is floating before reading the old iterate.

``coupling_residual_mixed`` and ``coupling_residual_interface`` looked a
field up in the *old* iterate before skipping it as non-floating; the L2
norm checked first.  An old iterate rebuilt from its floating fields -- the
predictor's starting iterate did exactly that -- then raised ``KeyError``
naming an integer or boolean field the norm was about to ignore.  The
predictor now keeps those fields; the norms no longer depend on it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from maddening.core.coupling.acceleration import (
    coupling_residual_interface,
    coupling_residual_l2,
    coupling_residual_mixed,
)


def _states():
    new = {"a": {"x": jnp.array([1.0, 2.0], jnp.float32), "n": jnp.int32(3),
                 "flag": jnp.asarray(True)},
           "b": {"x": jnp.float32(0.5), "tag": jnp.uint32(7)}}
    old_full = {"a": {"x": jnp.array([1.01, 1.98], jnp.float32), "n": jnp.int32(2),
                      "flag": jnp.asarray(False)},
                "b": {"x": jnp.float32(0.49), "tag": jnp.uint32(6)}}
    old_floats = {"a": {"x": old_full["a"]["x"]}, "b": {"x": old_full["b"]["x"]}}
    return new, old_full, old_floats


def _edge(source_node, source_field):
    """What the interface norm reads of an ``EdgeSpec``: its source, and the
    mapping and transform the edge rule applies to it (none here)."""
    return SimpleNamespace(source_node=source_node, source_field=source_field,
                           mapping=None, transform=None)


_EDGES = [_edge("a", "x"), _edge("b", "x"), _edge("a", "n"), _edge("b", "tag")]


@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
def test_an_old_iterate_without_its_non_floating_fields_measures_the_same(norm):
    new, old_full, old_floats = _states()
    nodes = ["a", "b"]

    def measure(old):
        if norm == "l2":
            return coupling_residual_l2(new, old, nodes, 0.0)
        if norm == "mixed":
            return coupling_residual_mixed(new, old, nodes, 0.0, 1e-3)
        return coupling_residual_interface(new, old, _EDGES, 0.0, 1e-3)

    full = measure(old_full)
    assert float(measure(old_floats)) == float(full) > 0.0
