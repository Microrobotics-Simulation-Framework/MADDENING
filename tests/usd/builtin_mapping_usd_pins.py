"""The USD half of ``tests/core/builtin_mapping_pins.py``.

:func:`capture_usd` records the ``maddening:mappingSpecJson`` attribute
each mapped edge of the two weight-pinned graphs writes to a stage, and the
weights a stage rebuilds.  Its output on the tree before the mapping kinds
became a registry is the ``usd`` section of
``tests/core/data/builtin_mapping_pins.json``::

    python -m tests.usd.builtin_mapping_usd_pins tests/core/data/builtin_mapping_pins.json

It needs ``usd-core``, so it lives here, with the tests the USD job runs.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from pxr import Usd

from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd
from tests.core.builtin_mapping_pins import (
    REGISTRY,
    _weights,
    rods_with_node_references,
    vectors_with_inline_and_asset_references,
)


def capture_usd() -> dict:
    """The ``maddening:mappingSpecJson`` text of every mapped edge of the
    two weight-pinned graphs, and the weights a stage rebuilds."""
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        graphs = {
            "rods_with_node_references": rods_with_node_references(),
            "vectors_with_inline_and_asset_references":
                vectors_with_inline_and_asset_references(base),
        }
        for name, gm in graphs.items():
            stage = Usd.Stage.CreateNew(str(base / f"{name}.usda"))
            save_graph_to_usd(gm, stage)
            stage.GetRootLayer().Save()
            attrs = {}
            for prim in stage.GetPrimAtPath("/Simulation/edges").GetChildren():
                attr = prim.GetAttribute("maddening:mappingSpecJson")
                attrs[prim.GetName()] = attr.Get() if attr else None
            reloaded = load_graph_from_usd(Usd.Stage.Open(str(base / f"{name}.usda")),
                                           node_registry=REGISTRY)
            out[name] = {"attributes": attrs, "rebuilt_weights": _weights(reloaded)}
    return out


def main(path: str) -> None:
    """Write the ``usd`` section, keeping the rest of the file."""
    target = Path(path)
    pins = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    pins["usd"] = capture_usd()
    target.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
