"""One function turns what was read at an edge's source into what its target is handed.

``maddening.core.edge._delivered`` is the step's edge rule: the interface
mapping, with the weights the step ran with, then the transform.  The
step's boundary resolution calls it, and so does everything that reads
"the value an edge carries": the helpers outside the step
(``GraphManager._boundary_inputs_from``), and the interface norm, its
float floor and the spectral analysis taken on its reading
(``acceleration._interface_readings``).

It has been written twice before.  Three readers outside the step applied
the transform and left the mapping out (MADD-ANO-193), and so did the
interface norm and its floor (MADD-ANO-195): each time one copy was fixed
the next was found.  So the rule is pinned where it cannot be copied
quietly: no other module of ``src/maddening`` calls an edge's
``transform`` or its mapping's ``apply``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "maddening"
#: The one module that may apply an edge.
RULE = SRC / "core" / "edge.py"


def edge_applications(source: str) -> list:
    """``[(line, call), ...]``: every call in *source* that applies an edge.

    A call of a method named ``transform`` (``edge.transform(v)``), and a
    call of ``apply`` on something named ``mapping`` (``edge.mapping.apply(v,
    w)``, ``mapping.apply(v)``).  Reading the attributes, comparing them
    (``e.transform is None``), a transform's registered *name* and a
    mapping's transpose ``apply_T`` are not applications.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        named = owner.attr if isinstance(owner, ast.Attribute) else getattr(owner, "id", "")
        if node.func.attr == "transform":
            found.append((node.lineno, f"{ast.unparse(owner)}.transform(...)"))
        elif node.func.attr == "apply" and "mapping" in str(named).lower():
            found.append((node.lineno, f"{ast.unparse(owner)}.apply(...)"))
    return found


def test_only_the_edge_rule_applies_an_edges_mapping_or_its_transform():
    """Every reader of what an edge delivers goes through ``_delivered``."""
    found, scanned = [], 0
    for path in sorted(SRC.rglob("*.py")):
        scanned += 1
        if path == RULE:
            continue
        found += [f"{path.relative_to(REPO)}:{line} {call}"
                  for line, call in edge_applications(path.read_text(encoding="utf-8"))]
    assert scanned > 150, f"the scan read {scanned} files: its scope no longer exists"
    assert found == [], (
        "apply an edge with maddening.core.edge._delivered (mapping, then transform, with "
        "the step's weights), not by hand:\n" + "\n".join(found))


def test_the_edge_rule_is_where_the_scan_says_it_is():
    """The premise: the rule's module applies a mapping and a transform, once
    each, inside ``_delivered``."""
    source = RULE.read_text(encoding="utf-8")
    calls = sorted(call for _line, call in edge_applications(source))
    assert calls == ["edge.mapping.apply(...)", "edge.transform(...)"], calls
    rule = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "_delivered")
    inside = {line for line, _call in edge_applications(source)}
    assert all(rule.lineno < line <= rule.end_lineno for line in inside), (rule.lineno, inside)


def test_the_scan_finds_a_second_copy_of_the_rule_and_nothing_else():
    """The scan can fail: each way the rule has been copied is found, and
    what merely reads an edge is not."""
    copies = (
        "def norm(edges, s):\n    for edge in edges:\n        v = s[edge.source_node]\n"
        "        if edge.transform is not None:\n            v = edge.transform(v)\n",
        "def read(e, v, w):\n    return e.mapping.apply(v, w)\n",
        "def read(mapping, v):\n    return mapping.apply(v)\n",
        "def read(spec, v):\n    return spec.edge.transform(v)\n",
    )
    for source in copies:
        assert edge_applications(source), source
    harmless = (
        "def name(edge):\n    return edge.transform.__qualname__ if edge.transform else None\n",
        "def transposed(mapping, f):\n    return mapping.apply_T(f)\n",
        "def has(e):\n    return e.transform is None and e.mapping is None\n",
        "import jax\n\ndef f(tree):\n    return jax.tree.map(abs, tree)\n",
        "def fit(spec):\n    return spec.transform in ('log', 'logit')\n",
    )
    for source in harmless:
        assert edge_applications(source) == [], source
