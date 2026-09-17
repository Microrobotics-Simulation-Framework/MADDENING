"""A coupling group survives a config round trip with every field intact.

``to_dict`` used to write nodes, edges, external inputs and parameter
specs and nothing about coupling groups, so a graph whose cyclic
subsystem was iterated to a converged fixed point each timestep came
back from its own config as a graph that takes one staggered pass —
same nodes, same edges, different answer, no warning.
``test_the_gap_this_closes_a_group_dropped_changes_the_answer`` pins
that difference, and the rest of this module is about carrying *all*
nineteen fields rather than the handful that happen to be interesting:
a group reloaded without its acceleration or its iteration cap is still
a group, and still solves differently.
"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from maddening.core.coupling.group import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

REGISTRY = {"HeatNode": HeatNode}

#: A non-default value for every field but ``nodes``.  Non-default is the
#: point: a serialiser that dropped a field and let the constructor
#: default fill it back in would pass a round trip written with defaults.
NON_DEFAULT = {
    "max_iterations": 17,
    "tolerance": 3e-7,
    "convergence_norm": "mixed",
    "atol": 2e-8,
    "rtol": 3e-6,
    "diagnostics": True,
    "acceleration": "iqn-ils",
    "relaxation": 0.75,
    "iteration_mode": "jacobi",
    "accelerated_fields": {"rod_a": ("temperature",), "rod_b": ("temperature",)},
    "subcycling": True,
    "boundary_interpolation": "quadratic",
    "jacobian_reuse": 3,
    "waveform_iterations": 2,
    "predictor": "quadratic",
    # 'fori' is deprecated (and the deprecation warning is filtered in
    # pyproject.toml), but it is a legal stored value until it is removed
    # and a config that loses it silently changes the solver.
    "solver": "fori",
    "strict_convergence": True,
    "linear_solver": "dense",
}


def _two_rods(dt_b: float = 0.01) -> GraphManager:
    """Two heat rods exchanging boundary temperatures — a real cycle."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod_a", 0.01, n_cells=5, initial_temperature=1.0))
    gm.add_node(HeatNode("rod_b", dt_b, n_cells=5, initial_temperature=0.0))
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform="extract_last")
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform="extract_first")
    return gm


def _reload(gm: GraphManager) -> GraphManager:
    """Through the config, as JSON text: a dict that is not JSON is not a
    config, whatever ``to_dict`` returns."""
    return GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)


# ---------------------------------------------------------------------------
# Every field
# ---------------------------------------------------------------------------

def test_to_dict_writes_every_field_of_the_dataclass():
    """The written key set *is* the field set.  A field added to
    ``CouplingGroup`` and not to the writer fails here, rather than in
    somebody's reloaded experiment."""
    written = set(CouplingGroup(nodes=frozenset({"a"})).to_dict())
    assert written == {f.name for f in fields(CouplingGroup)}


def test_every_field_round_trips_through_a_config():
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], **NON_DEFAULT)

    reloaded = _reload(gm)

    assert len(reloaded._coupling_groups) == 1
    before, after = gm._coupling_groups[0], reloaded._coupling_groups[0]
    for f in fields(CouplingGroup):
        assert getattr(after, f.name) == getattr(before, f.name), f.name
    assert after == before


def test_the_non_default_fixture_is_non_default_in_every_field():
    """Guards the guard: if one of these values ever became the default,
    the round trip above would stop testing that field."""
    default = CouplingGroup(nodes=frozenset({"rod_a", "rod_b"}))
    for name, value in NON_DEFAULT.items():
        assert getattr(default, name) != value, name
    # ... and the fixture covers everything but the node set.
    assert set(NON_DEFAULT) | {"nodes"} == {f.name for f in fields(CouplingGroup)}


def test_the_config_is_json_and_survives_a_text_round_trip():
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], **NON_DEFAULT)

    config = gm.to_dict()
    stored = config["coupling_groups"][0]
    assert stored["nodes"] == ["rod_a", "rod_b"]          # sorted list, not a set
    assert stored["accelerated_fields"] == {              # dict of lists, not tuples
        "rod_a": ["temperature"], "rod_b": ["temperature"],
    }
    assert json.loads(json.dumps(config)) == config


def test_to_dict_is_idempotent_through_from_dict():
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], **NON_DEFAULT)

    once = gm.to_dict()
    assert _reload(gm).to_dict() == once


# ---------------------------------------------------------------------------
# Several groups, and none
# ---------------------------------------------------------------------------

def test_two_groups_of_different_configuration_keep_their_own_settings():
    gm = GraphManager()
    for name in ("a1", "a2", "b1", "b2"):
        gm.add_node(HeatNode(name, 0.01, n_cells=4))
    gm.add_edge("a1", "a2", "temperature", "left_temperature", transform="extract_last")
    gm.add_edge("a2", "a1", "temperature", "right_temperature", transform="extract_first")
    gm.add_edge("b1", "b2", "temperature", "left_temperature", transform="extract_last")
    gm.add_edge("b2", "b1", "temperature", "right_temperature", transform="extract_first")
    gm.add_coupling_group(["a1", "a2"], max_iterations=4, acceleration="aitken")
    gm.add_coupling_group(["b1", "b2"], max_iterations=30, acceleration="fixed",
                          relaxation=0.4, iteration_mode="jacobi")

    reloaded = _reload(gm)

    by_nodes = {g.nodes: g for g in reloaded._coupling_groups}
    assert by_nodes[frozenset({"a1", "a2"})] == gm._coupling_groups[0]
    assert by_nodes[frozenset({"b1", "b2"})] == gm._coupling_groups[1]
    # Not one group's settings applied to both.
    assert by_nodes[frozenset({"a1", "a2"})].acceleration == "aitken"
    assert by_nodes[frozenset({"b1", "b2"})].relaxation == 0.4


def test_a_graph_with_no_groups_writes_no_key_and_reloads_with_none():
    """Absent, not ``[]``: an uncoupled graph writes the config it wrote
    before this key existed, which is what every reader of one (MIME's
    graph inspector among them) is already parsing."""
    gm = _two_rods()
    config = gm.to_dict()

    assert "coupling_groups" not in config
    assert GraphManager.from_dict(config, REGISTRY)._coupling_groups == []


def test_a_config_written_before_the_key_existed_still_loads():
    """Backward compatibility, stated as the old file itself: a config
    with exactly the three keys the writer used to emit."""
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], acceleration="aitken")
    old = {k: v for k, v in gm.to_dict().items() if k != "coupling_groups"}
    assert set(old) == {"nodes", "edges", "external_inputs"}

    reloaded = GraphManager.from_dict(old, REGISTRY)

    assert reloaded._coupling_groups == []
    assert reloaded.node_names == ["rod_a", "rod_b"]


# ---------------------------------------------------------------------------
# Hand-edited files
# ---------------------------------------------------------------------------

def _config_with(**overrides) -> dict:
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"])
    config = gm.to_dict()
    config["coupling_groups"][0].update(overrides)
    return config


@pytest.mark.parametrize(
    "field, bad",
    [
        ("acceleration", "aitkin"),
        ("convergence_norm", "mixxed"),
        ("solver", "for"),
        ("iteration_mode", "jacopi"),
        ("boundary_interpolation", "lineer"),
        ("predictor", "qaudratic"),
        ("linear_solver", "gmrs"),
    ],
)
def test_a_misspelled_enum_names_the_key_the_value_and_the_group(field, bad):
    """``CouplingGroup.__post_init__`` already rejects these; the loader's
    job is to say *which* group of the file it was reading, because that
    is what makes a hand-edited config actionable."""
    with pytest.raises(ValueError) as exc:
        GraphManager.from_dict(_config_with(**{field: bad}), REGISTRY)

    msg = str(exc.value)
    assert field in msg, msg
    assert repr(bad) in msg, msg
    assert "expected one of" in msg, msg
    assert "coupling_groups[0]" in msg, msg
    assert "rod_a" in msg and "rod_b" in msg, msg


def test_an_unknown_key_is_rejected_rather_than_ignored():
    """A typo in a *key* is the same silent misconfiguration as a typo in
    a value, so the dataclass constructor's ``TypeError`` is surfaced."""
    with pytest.raises(ValueError, match="accelaration"):
        GraphManager.from_dict(_config_with(accelaration="aitken"), REGISTRY)


def test_a_group_naming_a_node_the_config_does_not_have_is_rejected():
    with pytest.raises(ValueError) as exc:
        GraphManager.from_dict(_config_with(nodes=["rod_a", "ghost"]), REGISTRY)

    msg = str(exc.value)
    assert "ghost" in msg, msg
    assert "coupling_groups[0]" in msg, msg


def test_a_group_with_no_nodes_key_is_rejected_by_name():
    config = _config_with()
    del config["coupling_groups"][0]["nodes"]

    with pytest.raises(ValueError, match=r"coupling_groups\[0\].*'nodes'"):
        GraphManager.from_dict(config, REGISTRY)


def test_a_string_node_list_is_not_five_one_letter_nodes():
    """``list("rod_a")`` is a list of five names, and the failure that
    follows it talks about a node called ``'r'``."""
    with pytest.raises(ValueError, match="not the string 'rod_a'"):
        GraphManager.from_dict(_config_with(nodes="rod_a"), REGISTRY)


def test_two_groups_that_share_a_node_are_refused_on_load():
    """The graph refuses overlapping groups when they are built by hand
    (``add_coupling_group``), so it refuses them from a file too — naming
    the second group, which is the one to delete."""
    config = _config_with()
    config["coupling_groups"].append(dict(config["coupling_groups"][0]))

    with pytest.raises(ValueError) as exc:
        GraphManager.from_dict(config, REGISTRY)

    msg = str(exc.value)
    assert "coupling_groups[1]" in msg, msg
    assert "already belong to a coupling group" in msg, msg


# ---------------------------------------------------------------------------
# What the fields are *for*
# ---------------------------------------------------------------------------

def test_a_reloaded_group_reaches_the_same_state():
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], max_iterations=12, tolerance=1e-9,
                          acceleration="aitken")
    reloaded = _reload(gm)

    expected = gm.run_scan(3)
    actual = reloaded.run_scan(3)

    for node, fieldset in expected.items():
        for name, value in fieldset.items():
            # Exact: the reloaded graph is meant to be the same graph, and
            # a tolerance here would hide a dropped acceleration setting.
            assert bool((value == actual[node][name]).all()), f"{node}.{name}"


def test_the_gap_this_closes_a_group_dropped_changes_the_answer():
    """Why the key exists.  The same graph, reloaded from a config that
    does not mention its coupling group, is a *staggered* graph: one pass
    per timestep instead of iteration to a fixed point.  If this ever
    stops being true, the round trip above has stopped proving anything.
    """
    gm = _two_rods()
    gm.add_coupling_group(["rod_a", "rod_b"], max_iterations=20, tolerance=1e-12)
    config = gm.to_dict()
    without = {k: v for k, v in config.items() if k != "coupling_groups"}

    coupled = GraphManager.from_dict(config, REGISTRY).run_scan(3)
    staggered = GraphManager.from_dict(without, REGISTRY).run_scan(3)

    assert float(coupled["rod_b"]["temperature"][0]) != float(
        staggered["rod_b"]["temperature"][0]
    )
