"""Metamorphic oracle: a node's stability level changes exactly the
documented part of its FMU export.

``build_model_description`` exports a node's surfaces -- the external
inputs it reads, its state fields as outputs, its ``gm.params`` leaves as
tunable parameters, and its rate as a ``<Clock>`` -- only if its class is
tagged ``STABLE``, or ``EVOLVING`` / ``PROVISIONAL`` with
``include_evolving=True`` ("only STABLE-tagged sources/sinks contribute",
``model_description.py``).  So changing one class's level changes the
export by exactly that class's surfaces, and nothing else.

The oracle states it as a filter.  The **full** export is the description
built with every class in the graph tagged ``STABLE``.  For a drawn
assignment of levels (every level, and untagged), ``include_evolving`` and
``multi_clock``, the description built under it must equal the full export
filtered by the rule:

* an input is kept iff its target node is exported;
* an output iff its node is exported;
* a parameter iff its node is exported -- otherwise it is listed in
  ``fixed_parameters`` with a reason naming the stability level, while an
  exported node keeps the full export's own reasons for the leaves its step
  cannot read;
* a clock carries exactly the exported nodes on its interval, and a clock
  none is left on is gone; every kept variable keeps its clock;
* every kept variable keeps every attribute but its value reference (they
  are renumbered in order).

Changing one class's level between two assignments is then the metamorphic
statement: the two descriptions differ by that class's surfaces only.

Each surface is its own test, so a defect in one is pinned without hiding
the others.  **Known failing: M1** -- inputs are not filtered.

Tolerance: none (names and attributes compared exactly).

What it cannot see: a defect in the full export itself (which the FMU
differential oracles check against the graph), and a class that is
exported for a reason other than its level (there is none).
"""

from __future__ import annotations

import contextlib
import re
import warnings
from typing import Iterator

import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import _STABILITY_REGISTRY
from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.nodes import BallNode, HeatNode, SpringDamperNode, TableNode
from maddening.nodes.heart_pump import HeartPumpNode

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD

#: Every level a class can carry, and ``None`` for untagged.
LEVELS = (StabilityLevel.STABLE, StabilityLevel.EVOLVING, StabilityLevel.PROVISIONAL,
          StabilityLevel.EXPERIMENTAL, StabilityLevel.INTERNAL, StabilityLevel.DEPRECATED, None)

SURFACES = ("inputs", "outputs", "parameters", "clocks")


def exported(level, include_evolving: bool) -> bool:
    """The documented rule (``build_model_description``, ``include_evolving``)."""
    return level is StabilityLevel.STABLE or (
        include_evolving and level in (StabilityLevel.EVOLVING, StabilityLevel.PROVISIONAL))


def class_key(node) -> str:
    return f"{type(node).__module__}.{type(node).__name__}"


@contextlib.contextmanager
def levels(assignment: dict[str, object]) -> Iterator[None]:
    """The stability registry with each class in ``assignment`` at its
    level (``None``: untagged), restored exactly afterwards."""
    saved = {k: _STABILITY_REGISTRY[k] for k in assignment if k in _STABILITY_REGISTRY}
    try:
        for key, level in assignment.items():
            if level is None:
                _STABILITY_REGISTRY.pop(key, None)
            else:
                _STABILITY_REGISTRY[key] = level
        yield
    finally:
        for key in assignment:
            _STABILITY_REGISTRY.pop(key, None)
        _STABILITY_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# A description, as the oracle compares it
# ---------------------------------------------------------------------------

_CLOCK_NODES = re.compile(r"ticks when nodes (.*) update\.$")


class Export:
    """A description without its value references: each surface keyed by
    variable name, every attribute but the value reference kept, and a
    variable's clocks named by their intervals."""

    def __init__(self, md) -> None:
        interval = {v.value_reference: v.interval_decimal for v in md.variables if v.is_clock}
        self.surfaces: dict[str, dict] = {s: {} for s in SURFACES}
        self.owner: dict[str, str] = {}
        for v in md.variables:
            if v.is_clock:
                nodes = _CLOCK_NODES.search(v.description)
                assert nodes is not None, v.description
                self.surfaces["clocks"][v.interval_decimal] = frozenset(
                    nodes.group(1).split(", "))
                continue
            if v.causality == "independent":
                continue
            surface = {"input": "inputs", "output": "outputs",
                       "parameter": "parameters"}[v.causality]
            self.surfaces[surface][v.name] = (
                v.causality, v.variability, v.dtype, v.shape,
                tuple(sorted(interval[c] for c in v.clocks)), v.start, v.min, v.max,
                v.unit, v.description, v.node, v.field)
            self.owner[v.name] = v.node
        self.fixed = dict(md.fixed_parameters)


def describe(gm: GraphManager, *, include_evolving: bool, multi_clock: bool) -> Export:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Export(build_model_description(gm, model_name="Meta",
                                              include_evolving=include_evolving,
                                              multi_clock=multi_clock))


def full_export(gm: GraphManager, *, multi_clock: bool) -> Export:
    with levels({class_key(spec.node): StabilityLevel.STABLE for spec in gm._nodes.values()}):  # noqa: SLF001
        return describe(gm, include_evolving=False, multi_clock=multi_clock)


def expected_surface(full: Export, gm: GraphManager, kept: set[str], surface: str):
    """The full export's ``surface`` filtered to the exported nodes ``kept``."""
    if surface == "clocks":
        return {dt: nodes & kept for dt, nodes in full.surfaces["clocks"].items()
                if nodes & kept}
    rows = {name: row for name, row in full.surfaces[surface].items()
            if full.owner[name] in kept}
    if surface != "parameters":
        return rows
    fixed = {}
    for name, reason in full.fixed.items():
        node = name.split(".params.")[0]
        if node in kept:
            fixed[name] = reason
    for node, leaves in (gm.params.get("nodes") or {}).items():
        if node not in kept:
            for key in leaves:
                fixed[f"{node}.params.{key}"] = "<stability>"
    return rows, fixed


def actual_surface(export: Export, surface: str):
    if surface != "parameters":
        return export.surfaces[surface]
    fixed = {name: ("<stability>" if "stability level" in reason else reason)
             for name, reason in export.fixed.items()}
    return export.surfaces["parameters"], fixed


def check_filter(gm: GraphManager, assignment: dict, include_evolving: bool,
                 multi_clock: bool, surface: str, full: Export | None = None) -> None:
    full = full or full_export(gm, multi_clock=multi_clock)
    kept = {name for name, spec in gm._nodes.items()                           # noqa: SLF001
            if exported(assignment.get(class_key(spec.node)), include_evolving)}
    with levels(assignment):
        got = describe(gm, include_evolving=include_evolving, multi_clock=multi_clock)
    want = expected_surface(full, gm, kept, surface)
    have = actual_surface(got, surface)
    assert have == want, (
        f"{surface} under {_show(assignment)} include_evolving={include_evolving} "
        f"multi_clock={multi_clock} (exported nodes {sorted(kept)}): "
        f"only in the description {_diff(have, want)}, missing {_diff(want, have)}")


def _show(assignment: dict) -> dict:
    return {k.rsplit(".", 1)[-1]: (v.name if v is not None else "untagged")
            for k, v in assignment.items()}


def _diff(a, b):
    if isinstance(a, tuple):
        return [_diff(x, y) for x, y in zip(a, b)]
    return sorted(str(k) for k in set(a) - set(b)) + sorted(
        str(k) for k in set(a) & set(b) if a[k] != b[k])


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def _plant() -> GraphManager:
    """Four classes on two rates, each with a surface of every kind it has:
    a spring and a heart pump read external inputs, the heat rod an array
    input on the slower rate, the ball a table edge."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=30.0, rest_length=0.4))
    gm.add_node(HeatNode("rod", 0.02, n_cells=5, thermal_diffusivity=0.01))
    gm.add_node(TableNode("table", 0.01, position=0.25))
    gm.add_node(BallNode("ball", 0.01, initial_position=1.5))
    gm.add_node(HeartPumpNode("pump", 0.02))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("spring", "rod", "position", "left_temperature")
    gm.add_external_input("spring", "anchor_position")
    gm.add_external_input("rod", "heat_source", shape=(5,))
    gm.add_external_input("pump", "backpressure")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # the pump is EXPERIMENTAL
        gm.compile()
    return gm


_PLANT: list[GraphManager] = []


def plant() -> GraphManager:
    if not _PLANT:
        _PLANT.append(_plant())
    return _PLANT[0]


@st.composite
def assignments(draw, gm: GraphManager, *, flip_an_input_target: bool = False):
    """Levels for every class in ``gm``.  ``flip_an_input_target``: one
    class that some external input targets is drawn exported under the
    other choices and is then made not exported, so its input must go."""
    classes = sorted({class_key(spec.node) for spec in gm._nodes.values()})   # noqa: SLF001
    assignment = {c: draw(st.sampled_from(LEVELS), label=c.rsplit(".", 1)[-1]) for c in classes}
    include_evolving = draw(st.booleans(), label="include_evolving")
    if flip_an_input_target:
        targets = sorted({class_key(gm.get_node(ei.target_node))
                          for ei in gm._external_inputs})                     # noqa: SLF001
        target = draw(st.sampled_from(targets), label="dropped input target")
        assignment[target] = draw(st.sampled_from(
            [lvl for lvl in LEVELS if not exported(lvl, include_evolving)]))
    return assignment, include_evolving


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surface", ["outputs", "parameters", "clocks"])
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_a_class_level_filters_exactly_its_own_surfaces(surface, data):
    gm = plant()
    assignment, include_evolving = data.draw(assignments(gm), label="levels")
    multi_clock = data.draw(st.booleans(), label="multi_clock")
    event(f"exported classes: {sum(exported(v, include_evolving) for v in assignment.values())}")
    check_filter(gm, assignment, include_evolving, multi_clock, surface)


@pytest.mark.xfail(strict=True, reason=(
    "M1: build_model_description's stability filter is not applied to external inputs "
    "(the inputs loop, model_description.py ~949, never calls "
    "_ensure_stable_only_or_opt_in); pending fix"))
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_a_class_level_filters_exactly_its_own_inputs(data):
    gm = plant()
    assignment, include_evolving = data.draw(
        assignments(gm, flip_an_input_target=True), label="levels")
    check_filter(gm, assignment, include_evolving, data.draw(st.booleans()), "inputs")


def test_inputs_of_exported_classes_are_all_kept():
    """The half of the input rule that holds today: with every class
    exported, every declared input is an FMU input."""
    gm = plant()
    every = {class_key(spec.node): StabilityLevel.STABLE for spec in gm._nodes.values()}  # noqa: SLF001
    check_filter(gm, every, False, True, "inputs")


def test_changing_one_class_changes_only_its_own_variables():
    """The metamorphic statement on its own: the spring's class between
    STABLE and EXPERIMENTAL, everything else STABLE.  Outputs, parameters
    and clocks differ by the spring's surfaces only (its input is M1's)."""
    gm = plant()
    base = {class_key(spec.node): StabilityLevel.STABLE for spec in gm._nodes.values()}  # noqa: SLF001
    spring = class_key(gm.get_node("spring"))
    with levels(base):
        before = describe(gm, include_evolving=False, multi_clock=True)
    with levels({**base, spring: StabilityLevel.EXPERIMENTAL}):
        after = describe(gm, include_evolving=False, multi_clock=True)
    for surface in ("outputs", "parameters"):
        gone = set(before.surfaces[surface]) - set(after.surfaces[surface])
        assert gone and all(before.owner[n] == "spring" for n in gone), (surface, gone)
        assert set(after.surfaces[surface]) <= set(before.surfaces[surface])
        for name in after.surfaces[surface]:
            assert after.surfaces[surface][name] == before.surfaces[surface][name], name
    assert {k for k in after.fixed if k.startswith("spring.")} == {
        f"spring.params.{key}" for key in gm.params["nodes"]["spring"]}
    # the spring shares its rate with the table and the ball: the clock stays
    assert after.surfaces["clocks"] == {
        dt: nodes - {"spring"} for dt, nodes in before.surfaces["clocks"].items()}


# Per push: tests/property/test_metamorphic_fmu_stability_filter.py::test_a_class_level_filters_exactly_its_own_surfaces
@pytest.mark.slow  # a graph compiled and two descriptions built per example
@pytest.mark.parametrize("surface", ["outputs", "parameters", "clocks"])
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_class_level_filters_exactly_its_own_surfaces_on_generated_graphs(surface, data):
    from tests.property.strategies import ALL_NODE_KINDS, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=4,
                                     allow_mappings=False), label="recipe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gm = recipe.build()
    assignment, include_evolving = data.draw(assignments(gm), label="levels")
    check_filter(gm, assignment, include_evolving, data.draw(st.booleans()), surface)
