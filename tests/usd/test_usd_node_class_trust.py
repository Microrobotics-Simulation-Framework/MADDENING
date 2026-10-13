"""A USD stage may only name node classes the caller allows.

A ``.usda`` carries the fully qualified Python class of every node it
holds, and until 0.4.0 the reader resolved an unregistered one with
``importlib.import_module``.  Opening an untrusted stage therefore ran the
named module's import-time code -- before the ``TypeError`` that followed,
and with ``sys.path[0]`` being the running script's directory, so a
hostile ``.usda`` shipped next to a ``.py`` was plain code execution.
``GraphManager.from_dict`` has always required an explicit registry; the
USD reader now does too.

Written from the independent audit of 2026-09-19 (``params-io``; H3,
reproducer ``r15_usd_import.py``).
"""

import sys
import textwrap

import pytest
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.usd.serialization import (
    load_graph_from_usd,
    register_node_class,
    save_graph_to_usd,
)

_STAGE = """#usda 1.0
def MaddeningSimulationGraph "Simulation" {{
    def Scope "nodes" {{
        def MaddeningNode "n" {{
            custom string maddening:nodeType = "{node_type}"
            custom string maddening:nodeName = "n"
            custom double maddening:timestep = 0.01
            custom string maddening:paramsJson = "{{}}"
        }}
    }}
}}
"""


@pytest.fixture
def importable_module(tmp_path, monkeypatch):
    """A module on ``sys.path`` that records having been imported.

    Stands in for the hostile payload: the point is only that importing it
    has an observable effect, which a load must not cause.
    """
    marker = tmp_path / "imported.txt"
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "usd_trust_payload.py").write_text(textwrap.dedent(f"""
        import pathlib
        pathlib.Path({str(marker)!r}).write_text("imported")


        class Payload:
            def __init__(self, name, timestep, **kwargs):
                self.name = name
                self.timestep = timestep
    """))
    monkeypatch.syspath_prepend(str(pkg))
    monkeypatch.delitem(sys.modules, "usd_trust_payload", raising=False)
    yield marker
    sys.modules.pop("usd_trust_payload", None)


def _stage_naming(tmp_path, node_type):
    path = tmp_path / "graph.usda"
    path.write_text(_STAGE.format(node_type=node_type))
    return Usd.Stage.Open(str(path))


def test_opening_a_stage_does_not_import_the_module_it_names(
        tmp_path, importable_module):
    stage = _stage_naming(tmp_path, "usd_trust_payload.Payload")

    with pytest.raises(KeyError):
        load_graph_from_usd(stage)

    assert not importable_module.exists(), (
        "opening a stage must not run the code of a module it names"
    )
    assert "usd_trust_payload" not in sys.modules


def test_the_refusal_says_exactly_what_to_pass(tmp_path, importable_module):
    stage = _stage_naming(tmp_path, "usd_trust_payload.Payload")

    with pytest.raises(KeyError) as exc:
        load_graph_from_usd(stage)

    msg = str(exc.value)
    assert "usd_trust_payload.Payload" in msg
    assert "node_registry" in msg
    assert "register_node_class" in msg
    assert "allow_import=True" in msg


def test_allow_import_is_the_opt_in_for_the_old_behaviour(
        tmp_path, importable_module):
    stage = _stage_naming(tmp_path, "usd_trust_payload.Payload")

    # The stub is not a real SimulationNode, so the load fails after the
    # import -- which is the point: the import is what allow_import buys.
    with pytest.raises(Exception):
        load_graph_from_usd(stage, allow_import=True)

    assert importable_module.exists()
    assert "usd_trust_payload" in sys.modules


def test_a_node_registry_is_honoured_by_qualified_and_by_bare_name(tmp_path):
    for key in ("tests.usd.test_usd_node_class_trust.SpringDamperNode",
                "SpringDamperNode"):
        stage = _stage_naming(
            tmp_path, "tests.usd.test_usd_node_class_trust.SpringDamperNode")
        gm = load_graph_from_usd(stage, node_registry={key: SpringDamperNode})
        assert isinstance(gm.get_node("n"), SpringDamperNode)


def test_a_registry_keeps_a_stage_written_by_a_third_party_node_loadable(tmp_path):
    """The migration path for a caller the new default would break."""

    class OutOfTreeNode(SpringDamperNode):
        pass

    qualname = f"{OutOfTreeNode.__module__}.{OutOfTreeNode.__qualname__}"
    stage = _stage_naming(tmp_path, qualname)

    with pytest.raises(KeyError):
        load_graph_from_usd(stage)

    gm = load_graph_from_usd(stage, node_registry={qualname: OutOfTreeNode})
    assert isinstance(gm.get_node("n"), OutOfTreeNode)

    # register_node_class() is the other way in, and is process-wide.
    register_node_class(OutOfTreeNode)
    try:
        assert isinstance(load_graph_from_usd(_stage_naming(tmp_path, qualname))
                          .get_node("n"), OutOfTreeNode)
    finally:
        from maddening.usd.serialization import _NODE_CLASS_REGISTRY
        _NODE_CLASS_REGISTRY.pop(qualname, None)


def test_a_graph_of_built_in_nodes_still_round_trips_with_no_registry():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.compile()
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    reloaded = load_graph_from_usd(stage)
    assert isinstance(reloaded.get_node("s"), SpringDamperNode)


def test_allow_import_does_not_register_the_class_for_later_loads(
        tmp_path, importable_module):
    """Opting in applies to the call that opted in, not to the process.

    Caching the imported class would let the *next* load of an untrusted
    stage instantiate a class that load never allowed.
    """
    stage = _stage_naming(tmp_path, "usd_trust_payload.Payload")
    with pytest.raises(Exception):
        load_graph_from_usd(stage, allow_import=True)
    assert "usd_trust_payload" in sys.modules       # the import did happen

    with pytest.raises(KeyError):
        load_graph_from_usd(_stage_naming(tmp_path, "usd_trust_payload.Payload"))
