"""``replace_node`` must not change what any edge *does*.

The unit tests in ``tests/surrogates/test_replace.py`` pin one dropped
attribute each.  This is the general statement they are instances of:
for an arbitrary valid graph and an arbitrary node of it, swapping that
node for a surrogate that holds the same state leaves every boundary
input every node receives numerically unchanged.  It is stated over
*effects* rather than over ``EdgeSpec`` fields on purpose -- an edge
whose ``additive`` comes back but whose value does not is still wrong,
and the value is the only thing a user sees.

The surrogate here is a stand-in, not a fit: it holds the replaced
node's state exactly, so any difference in a downstream boundary input
is the replacement machinery losing something, never approximation
error.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import SurrogateNode
from maddening.surrogates.replace import replace_node

from tests.property.strategies import graph_recipes


class _Hold(SurrogateArchitecture):
    """Returns the state it was given -- the replaced node, frozen."""

    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {}

    def forward(self, params, state, boundary_inputs, dt):
        return dict(state)


def _stand_in(gm, name):
    """A :class:`SurrogateNode` carrying ``name``'s current state."""
    node = gm.get_node(name)
    state = dict(gm._state[name])
    return SurrogateNode(
        name=name,
        timestep=gm._nodes[name].timestep,
        architecture=_Hold(),
        weights={},
        state_spec={k: tuple(jnp.shape(v)) for k, v in state.items()},
        boundary_spec={
            k: tuple(getattr(v, "shape", ()) or ())
            for k, v in node.boundary_input_spec().items()
        },
        initial_values=state,
    )


def _boundary_inputs(gm, names):
    return {
        n: {k: np.asarray(v) for k, v in gm.resolve_boundary_inputs(n).items()}
        for n in names
    }


# Coupling groups are excluded: they change how ``step`` iterates, not
# what an edge delivers, and ``resolve_boundary_inputs`` -- the effect
# this property is about -- does not consult them.  They would only make
# every example slower to build.
_RECIPES = st.one_of(
    graph_recipes(allow_coupling_groups=False),
    # A second draw with the mapping weights moved away from what the
    # ``MappingSpec`` rebuilds, standing in for a sysid fit: those live
    # in ``gm.params["mappings"]``, which ``remove_node`` discards.
    graph_recipes(allow_coupling_groups=False, train_mapping_weights=True),
)


@given(recipe=_RECIPES, data=st.data())
def test_replace_node_leaves_every_edge_effect_unchanged(recipe, data):
    gm = recipe.build()
    names = recipe.node_names
    name = data.draw(st.sampled_from(names), label="replaced node")

    # A surrogate holds its state as float32 (``SurrogateNode.initial_state``),
    # so a node with a non-floating state field cannot be stood in for
    # exactly and the property would be measuring the cast, not the swap.
    assume(all(
        jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
        for v in gm._state[name].values()
    ))

    before = _boundary_inputs(gm, names)

    replace_node(gm, name, _stand_in(gm, name))
    gm.compile()

    after = _boundary_inputs(gm, names)
    assert set(after) == set(before)
    for node_name in before:
        assert set(after[node_name]) == set(before[node_name]), node_name
        for field, want in before[node_name].items():
            np.testing.assert_allclose(
                after[node_name][field], want, rtol=1e-5, atol=1e-6,
                err_msg=f"{node_name}.{field} changed across replace_node",
            )


@given(recipe=_RECIPES, data=st.data())
def test_replace_node_preserves_every_edge_key(recipe, data):
    """``EdgeSpec.key`` carries ``ordinal``, and names the
    ``params["mappings"]`` slot an edge's weights live in."""
    gm = recipe.build()
    name = data.draw(st.sampled_from(recipe.node_names), label="replaced node")
    assume(all(
        jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
        for v in gm._state[name].values()
    ))
    before = [e.key for e in gm._edges]

    replace_node(gm, name, _stand_in(gm, name))

    assert sorted(e.key for e in gm._edges) == sorted(before)
