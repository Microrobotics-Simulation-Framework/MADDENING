"""Contract properties for the static-data dependency declaration (D10).

``tests/core/test_static_data.py`` covers the declaration by example --
``HeatNode``'s two grids, a wrapper, the message a user reads.  This
module covers the *rule*, over generated nodes rather than chosen ones:

    ``compile()`` succeeds **iff** no declared dependency names a
    trainable parameter.

Both directions matter and each fails a different way.  A rule that
never fires is the latent hazard D10 describes, unchanged; a rule that
fires too often refuses graphs that are perfectly differentiable -- and
the very first candidate, "any declared dependency on a parameter with
``trainable=True``", is that rule, because it would refuse every node
deriving a static from an ``int`` such as ``n_cells``.  What makes a
parameter dangerous is that the graph *differentiates* it: it has to be
a leaf of :meth:`~maddening.core.node.SimulationNode.params_pytree` as
well as trainable.  Since 0.4.0 an ``int`` whose node declares it
trainable *is* such a leaf -- ``params_pytree`` promotes it, because
``stiffness=100`` is a stiffness and not a grid size -- so the ``int``
that stays structural is the one declared frozen (or not declared at
all, like ``n_cells``).  The generated nodes mix ``float`` and ``int``
parameters, trainable and frozen, with dependency declarations that also
name parameters the node does not have, so both halves of the condition
are searched rather than asserted.

The third property is composition: wrapping a node must not change the
verdict.  ``static_data`` forwards to wrapped nodes and so does the
declaration, so a wrapper is exactly as differentiable as what it wraps
-- it can neither rescue a violation nor invent one.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, static_data_dep_violations
from maddening.core.params import ParamSpec
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.core.static_data import StaticArray
from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY

_PARAM_NAMES = ("alpha", "beta", "gamma")
_STATIC_KEYS = ("table", "mesh", "weights")
#: A name no generated node ever declares as a parameter, so a
#: declaration may point at nothing -- which must be ignored, not
#: guessed at.
_ABSENT = "not_a_parameter"


class _GeneratedNode(SimulationNode):
    """A node built from a drawn recipe.

    Every static is built in ``__init__`` and every ``float`` parameter
    also reaches ``update`` traced, so a node whose declaration is legal
    is genuinely differentiable rather than merely accepted.
    """

    def __init__(self, name, timestep, *, kinds, trainable, statics, deps):
        super().__init__(
            name,
            timestep,
            **{k: (3 if kind == "int" else 1.5) for k, kind in kinds.items()},
        )
        self._kinds = dict(kinds)
        self._trainable = dict(trainable)
        self._deps = {k: tuple(v) for k, v in deps.items()}
        self._tables = {
            key: jnp.arange(2, dtype=jnp.float32) + i
            for i, key in enumerate(statics)
        }

    @property
    def static_data(self) -> dict:
        return {k: StaticArray(v) for k, v in self._tables.items()}

    def static_data_deps(self) -> dict:
        return dict(self._deps)

    def param_specs(self) -> dict:
        return {
            **super().param_specs(),
            **{k: ParamSpec(trainable=self._trainable[k]) for k in self._kinds},
        }

    def initial_state(self) -> dict:
        return {"y": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        acc = state["y"]
        for table in self._tables.values():
            acc = acc + table[0] * dt          # the baked half
        for key, kind in self._kinds.items():
            if kind == "float":
                acc = acc + p[key] * dt        # the traced half
        return {"y": acc}


@st.composite
def _recipes(draw):
    """``(kinds, trainable, statics, deps)`` for one generated node."""
    names = draw(st.lists(
        st.sampled_from(_PARAM_NAMES), unique=True, min_size=1, max_size=3,
    ))
    kinds = {n: draw(st.sampled_from(("float", "int"))) for n in names}
    trainable = {n: draw(st.booleans()) for n in names}
    statics = draw(st.lists(
        st.sampled_from(_STATIC_KEYS), unique=True, min_size=1, max_size=3,
    ))
    deps = {}
    for key in statics:
        named = draw(st.lists(
            st.sampled_from([*names, _ABSENT]), unique=True, max_size=3,
        ))
        if named:
            deps[key] = tuple(named)
    return kinds, trainable, statics, deps


def _forbidden(kinds, trainable, deps) -> set[tuple[str, str]]:
    """``(static_key, param_key)`` pairs the rule must refuse.

    Spelled out independently of the implementation: a dependency is
    forbidden when the parameter is one the graph differentiates *and*
    nothing has frozen it.  Every generated parameter carries a declared
    spec, so a trainable one is differentiated whether it is spelled as a
    float or as an int (``params_pytree`` promotes the int spelling of a
    declared-trainable constant); a frozen int stays structural.
    """
    return {
        (static_key, param_key)
        for static_key, param_keys in deps.items()
        for param_key in param_keys
        if kinds.get(param_key) in ("float", "int") and trainable.get(param_key, True)
    }


def _node(recipe, name="generated"):
    kinds, trainable, statics, deps = recipe
    return _GeneratedNode(
        name, 0.01,
        kinds=kinds, trainable=trainable, statics=statics, deps=deps,
    )


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(recipe=_recipes())
def test_compile_refuses_a_graph_iff_a_dependency_is_trainable(recipe):
    """The whole rule, in both directions, on a one-node graph."""
    kinds, trainable, _statics, deps = recipe
    forbidden = _forbidden(kinds, trainable, deps)

    gm = GraphManager()
    gm.add_node(_node(recipe))
    if not forbidden:
        gm.compile()
        gm.step()               # and it is a working graph, not just a quiet one
        return

    with pytest.raises(ValueError) as excinfo:
        gm.compile()
    message = str(excinfo.value)
    # Whichever pair it reports has to be one of the forbidden ones, and
    # the message has to name all three things a reader needs.
    reported = [
        (s, p) for (s, p) in forbidden
        if f"'{s}'" in message and f"'{p}'" in message
    ]
    assert reported, f"message names no forbidden pair: {message}"
    assert "'generated'" in message


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(recipe=_recipes())
def test_wrapping_a_node_does_not_change_the_verdict(recipe):
    """A wrapper is exactly as differentiable as what it wraps.

    ``HybridNode`` forwards ``param_specs`` and ``params_pytree``; the
    bare wrapper below forwards neither, so the check has to reach the
    inner node itself rather than lean on the wrapper's manners.
    """
    kinds, trainable, _statics, deps = recipe
    refused = bool(_forbidden(kinds, trainable, deps))

    class _BareWrapper(SimulationNode):
        def __init__(self, inner):
            super().__init__(inner.name, inner.delta_t)
            self.inner = inner

        def initial_state(self):
            return self.inner.initial_state()

        def update(self, state, boundary_inputs, dt):
            return self.inner.update(state, boundary_inputs, dt)

    for wrap in (
        lambda n: HybridNode(n, lambda s, b, d: {}),
        _BareWrapper,
        lambda n: HybridNode(_BareWrapper(n), lambda s, b, d: {}),
    ):
        gm = GraphManager()
        gm.add_node(wrap(_node(recipe)))
        if refused:
            with pytest.raises(ValueError):
                gm.compile()
        else:
            gm.compile()


@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
@given(recipe=_recipes())
def test_the_violations_found_are_exactly_the_forbidden_pairs(recipe):
    """No graph, so the rule itself is searched wide rather than deep.

    ``compile`` reports the first violation and stops; this is the whole
    set, which is what pins "iff" at the level of individual pairs
    rather than of the graph as a whole.
    """
    kinds, trainable, _statics, deps = recipe
    node = _node(recipe)
    found = {(s, p) for _owner, s, p in static_data_dep_violations(node)}
    assert found == _forbidden(kinds, trainable, deps)
    # The declaration never invents a static, wrapped or not.
    assert set(node.static_data_deps()) <= set(node.static_data)


@st.composite
def _recipes_with_overrides(draw):
    """A recipe plus the graph-level ``set_param_spec`` calls over it."""
    recipe = draw(_recipes())
    kinds = recipe[0]
    # ``set_param_spec`` reaches the leaves of ``params_pytree()`` only.
    # A float always is one; an int is one only while its node declares it
    # trainable, and an override freezing it would take it back out of the
    # pytree the override was validated against -- so the overrides stay
    # on the floats.
    overrides = {
        name: draw(st.booleans())
        for name, kind in kinds.items()
        if kind == "float" and draw(st.booleans())
    }
    return recipe, overrides


def _effective_trainable(trainable, overrides) -> dict:
    """What ``gm.trainable_mask()`` reports: the node's own specs with
    the graph's overrides written over them."""
    return {**trainable, **overrides}


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(case=_recipes_with_overrides())
def test_the_rule_sees_the_same_specs_the_optimiser_does(case):
    """``compile()`` refuses iff ``trainable_mask()`` says so.

    ``GraphManager.param_specs()`` merges ``set_param_spec`` overrides
    over the node's own specs, and ``trainable_mask``, ``unconstrain``,
    ``check_params`` and ``maddening.sysid`` all optimise against that
    merged view.  The rule resolved the node alone, so it disagreed in
    both directions: a graph-level unfreeze walked past the refusal into
    a gradient missing the term through the static, and a graph-level
    freeze -- the remedy the error message names first -- did not clear
    the refusal.  This property is the agreement, not either half.
    """
    recipe, overrides = case
    kinds, trainable, _statics, deps = recipe
    forbidden = _forbidden(kinds, _effective_trainable(trainable, overrides), deps)

    gm = GraphManager()
    gm.add_node(_node(recipe, name="n"))
    for key, is_trainable in overrides.items():
        gm.set_param_spec("n", key, ParamSpec(trainable=is_trainable))

    # The mask is the optimiser's view; the rule has to agree with it.
    mask = gm.param_specs()["nodes"]["n"]
    for key, is_trainable in overrides.items():
        assert mask[key].trainable is is_trainable

    if not forbidden:
        gm.compile()
        gm.step()
        return
    with pytest.raises(ValueError) as excinfo:
        gm.compile()
    message = str(excinfo.value)
    assert [(s, p) for (s, p) in forbidden
            if f"'{s}'" in message and f"'{p}'" in message], message


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(case=_recipes_with_overrides())
def test_an_override_on_a_declared_parameter_re_runs_the_rule(case):
    """A spec change that moves the verdict has to reach the next run.

    ``set_param_spec`` is documented not to dirty the graph -- specs are
    optimiser-side metadata.  A key a ``static_data_deps()`` entry names
    is the exception: it decides whether the graph compiles at all, so a
    graph already compiled must not go on running the accepted step.
    """
    recipe, overrides = case
    kinds, trainable, _statics, deps = recipe
    if _forbidden(kinds, trainable, deps):
        return                      # never compiled in the first place
    declared = {p for keys in deps.values() for p in keys}

    gm = GraphManager()
    gm.add_node(_node(recipe, name="n"))
    gm.compile()
    assert not gm._dirty

    for key, is_trainable in overrides.items():
        gm.set_param_spec("n", key, ParamSpec(trainable=is_trainable))
        assert gm._dirty == (key in declared), key
        gm._dirty = False
