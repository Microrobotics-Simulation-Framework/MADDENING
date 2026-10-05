"""What the four built-in mapping kinds write, rebuild and refuse, as data.

:func:`capture` builds fixed graphs over ``rbf``, ``nearest_neighbor``,
``projection_1d`` and ``matrix`` mappings and records, for each: the exact
JSON text of ``GraphManager.to_dict()``, the weights of every mapped edge
and of the same edge after ``from_dict``, and the message of every refusal
a malformed spec draws.  :func:`capture_usd` records the
``maddening:mappingSpecJson`` attribute each edge writes to a USD stage.

``tests/core/data/builtin_mapping_pins.json`` is the output of this module
on the tree *before* the mapping kinds became a registry
(``release/0.4.0`` at ``a7e69509``)::

    python -m tests.core.builtin_mapping_pins tests/core/data/builtin_mapping_pins.json

``tests/core/test_mapping_registry.py`` and
``tests/usd/test_usd_mapping_spec.py`` run it again on the current tree and
compare, so a change to what a built-in kind writes or rebuilds fails with
the two values side by side.

The weights are compared exactly, so the cases are chosen for arithmetic
that does not depend on the machine: selection and projection matrices
(``nearest_neighbor``, ``projection_1d``) and a given matrix are exact by
construction, and the ``rbf`` cases use a Gaussian kernel narrow against the
point spacing, whose system is close to the identity (a LAPACK build that
differs in the last place of a float64 solve still rounds to the same
float32).  The other kernels are pinned through their config text only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.coupling.mapping_spec import (
    MappingSpec,
    make_point_resolver,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode

PINS = Path(__file__).parent / "data" / "builtin_mapping_pins.json"


class Vec(SimulationNode):
    """n-vector integrating its boundary input (a stand-in interface)."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


REGISTRY = {"HeatNode": HeatNode, "Vec": Vec}

#: The matrix of the ``matrix`` cases, saved as ``H.npy`` beside the config.
MATRIX = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.5, 0.5, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.5, 0.5, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.25, 0.75]], np.float32)


def _grid(gm: GraphManager, name: str):
    return gm.get_node(name).static_data["grid_x"].value


def rods_with_node_references() -> GraphManager:
    """Two rods mapped both ways from their own grids, by node reference."""
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=6, thermal_diffusivity=0.1))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=12, thermal_diffusivity=0.1))
    xc, xf = _grid(gm, "coarse"), _grid(gm, "fine")
    gm.add_edge("coarse", "fine", "temperature", "heat_source",
                mapping=rbf_mapping(xc, xf, kernel="gaussian", epsilon=24.0,
                                    source_ref={"node": "coarse", "field": "grid_x"},
                                    target_ref={"node": "fine", "field": "grid_x"}))
    gm.add_edge("fine", "coarse", "temperature", "heat_source",
                mapping=nearest_neighbor_mapping(
                    xf, xc, mode="conservative",
                    source_ref={"node": "fine", "field": "grid_x"},
                    target_ref={"node": "coarse", "field": "grid_x"}))
    return gm


def vectors_with_inline_and_asset_references(base_dir: Path) -> GraphManager:
    """Two vector nodes: a projection and a nearest-neighbour selection from
    inlined points, a labelled and an unlabelled matrix from an asset, and
    two Gaussian interpolants (with and without the polynomial)."""
    np.save(base_dir / "H.npy", MATRIX)
    np.savez(base_dir / "grids.npz", four=np.linspace(0.0, 1.0, 5),
             six=np.linspace(0.0, 1.0, 7))
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=4))
    gm.add_node(Vec("b", 1.0, n=6))
    x4 = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    x6 = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    gm.add_edge("a", "b", "v", "inp",
                mapping=projection_1d_mapping(np.linspace(0.0, 1.0, 5),
                                              np.linspace(0.0, 1.0, 7)))
    gm.add_edge("a", "b", "v", "inp", additive=True,
                mapping=projection_1d_mapping(
                    np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 7),
                    source_ref={"asset": "grids.npz", "key": "four"},
                    target_ref={"asset": "grids.npz", "key": "six"}))
    gm.add_edge("a", "b", "v", "inp", additive=True,
                mapping=matrix_mapping(MATRIX, kind="supermesh", mode="conservative",
                                       asset="H.npy"))
    gm.add_edge("a", "b", "v", "inp", additive=True,
                mapping=matrix_mapping(MATRIX, asset="H.npy"))
    gm.add_edge("b", "a", "v", "inp",
                mapping=nearest_neighbor_mapping(x6, x4))
    gm.add_edge("b", "a", "v", "inp", additive=True,
                mapping=rbf_mapping(x6, x4, kernel="gaussian", epsilon=16.0,
                                    polynomial=False, ridge=1e-6))
    gm.add_edge("b", "a", "v", "inp", additive=True,
                mapping=rbf_mapping(x6, x4, kernel="gaussian", epsilon=16.0,
                                    mode="conservative"))
    return gm


def vectors_with_every_kernel() -> GraphManager:
    """One edge per ``rbf`` kernel that the exact weight pins leave out;
    only the config text of this graph is pinned."""
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=4))
    gm.add_node(Vec("b", 1.0, n=6))
    x4 = np.linspace(0.0, 1.0, 4)
    x6 = np.linspace(0.0, 1.0, 6)
    for i, kernel in enumerate(("multiquadric", "inverse_multiquadric",
                                "thin_plate_spline")):
        gm.add_edge("a", "b", "v", "inp", additive=bool(i),
                    mapping=rbf_mapping(x4, x6, kernel=kernel, epsilon=2.0 + i,
                                        polynomial=bool(i % 2), ridge=1e-8,
                                        mode="conservative" if i == 2 else "consistent"))
    return gm


def _weights(gm: GraphManager) -> dict:
    out = {}
    for edge in gm.edges:
        if edge.mapping is None:
            continue
        out[edge.key] = {
            name: {"dtype": str(np.asarray(leaf).dtype),
                   "shape": list(np.shape(leaf)),
                   "values": np.asarray(leaf).tolist()}
            for name, leaf in edge.mapping.params_pytree().items()
        }
    return out


def _config_case(gm: GraphManager, base_dir=None, *, weights: bool = True) -> dict:
    text = json.dumps(gm.to_dict())
    case = {"config": text}
    if weights:
        reloaded = GraphManager.from_dict(json.loads(text), REGISTRY, base_dir=base_dir)
        case["weights"] = _weights(gm)
        case["rebuilt_weights"] = _weights(reloaded)
        case["rebuilt_config"] = json.dumps(reloaded.to_dict())
    return case


INLINE3 = {"inline": [0.0, 0.5, 1.0], "dtype": "float64"}
_PAIR = {"source_points": INLINE3, "target_points": INLINE3}

#: Malformed specs of the built-in kinds, each refused by ``from_dict``.
#: None of the messages lists the registered kinds, so they are the same in
#: a process that has registered others.
REFUSED_SPECS: dict[str, dict] = {
    "no points": {"kind": "rbf", "mode": "consistent"},
    "points not a dict": {"kind": "nearest_neighbor", "points": [INLINE3, INLINE3]},
    "unknown hyper-parameter": {"kind": "rbf", "sigma": 2.0, "points": _PAIR},
    "hyper-parameter another kind takes": {"kind": "nearest_neighbor", "epsilon": 1.0,
                                           "points": _PAIR},
    "real given a string": {"kind": "rbf", "epsilon": "abc", "points": _PAIR},
    "real given None": {"kind": "rbf", "ridge": None, "points": _PAIR},
    "real given a bool": {"kind": "rbf", "epsilon": True, "points": _PAIR},
    "bool given a number": {"kind": "rbf", "polynomial": 1, "points": _PAIR},
    "str given a number": {"kind": "rbf", "kernel": 3, "points": _PAIR},
    "label not a str": {"kind": "matrix", "label": 5, "points": {"H": {"asset": "H.npy"}}},
    "wrong point sets": {"kind": "projection_1d", "points": {"source_points": INLINE3}},
    "extra point set": {"kind": "matrix", "points": {"H": {"asset": "H.npy"},
                                                     "G": INLINE3}},
    "missing reference": {"kind": "rbf", "points": {"source_points": None,
                                                    "target_points": INLINE3}},
    "missing matrix reference": {"kind": "matrix", "points": {"H": None}},
    "factory refuses the mode": {"kind": "nearest_neighbor", "mode": "sideways",
                                 "points": _PAIR},
    "factory refuses the kernel": {"kind": "rbf", "kernel": "cubic", "points": _PAIR},
    "recorded shape differs": {"kind": "nearest_neighbor", "shape": [3, 9],
                               "points": _PAIR},
}


def _refusals(base_dir: Path) -> dict:
    out = {}
    for name, mapping in REFUSED_SPECS.items():
        config = {"nodes": [{"type": "Vec", "name": "a", "timestep": 1.0, "params": {"n": 3}},
                            {"type": "Vec", "name": "b", "timestep": 1.0, "params": {"n": 3}}],
                  "edges": [{"source_node": "a", "target_node": "b", "source_field": "v",
                             "target_field": "inp", "mapping": mapping}],
                  "external_inputs": []}
        try:
            GraphManager.from_dict(config, REGISTRY, base_dir=base_dir)
        except Exception as exc:  # noqa: BLE001 - the type is what is recorded
            out[name] = {"type": type(exc).__name__, "message": str(exc),
                         "cause": type(exc.__cause__).__name__}
        else:
            out[name] = {"type": None}
    return out


def _unserialisable() -> dict:
    """The write-side refusals: a point set too large to inline, a matrix
    without an asset, a mapping without a spec."""
    out = {}
    big = np.linspace(0.0, 1.0, 65)
    cases = {
        "large source without a reference": (65, 2, rbf_mapping(big, [0.0, 1.0])),
        "matrix without an asset": (4, 6, matrix_mapping(MATRIX, kind="supermesh")),
    }
    for name, (n_source, n_target, mapping) in cases.items():
        gm = GraphManager()
        gm.add_node(Vec("a", 1.0, n=n_source))
        gm.add_node(Vec("b", 1.0, n=n_target))
        gm.add_edge("a", "b", "v", "inp", mapping=mapping)
        record = {"shown": json.dumps(gm.to_dict(strict_mappings=False)["edges"])}
        try:
            gm.to_dict()
        except ValueError as exc:
            record["message"] = str(exc)
        try:
            mapping.spec.build(make_point_resolver())
        except ValueError as exc:
            record["build"] = f"{type(exc).__name__}: {exc}"
        out[name] = record
    return out


def _spec_forms() -> dict:
    """``MappingSpec`` itself: what ``from_dict`` makes of the forms
    ``describe()`` writes, and what ``to_dict`` gives back."""
    forms = {
        "defaults filled by the factory, not the spec":
            {"kind": "rbf", "epsilon": 2, "points": _PAIR},
        "mode dropped for a kind with a fixed mode":
            {"kind": "projection_1d", "mode": "conservative", "shape": [2, 2],
             "points": {"source_boundaries": INLINE3, "target_boundaries": INLINE3}},
        "a user label names the matrix factory":
            {"kind": "supermesh", "label": "supermesh", "mode": "consistent",
             "points": {"H": {"asset": "H.npy"}}},
        "a label that is another kind's name is still a matrix":
            {"kind": "rbf", "label": "rbf", "mode": "consistent",
             "points": {"H": {"asset": "H.npy"}}},
    }
    return {name: json.dumps(MappingSpec.from_dict(d).to_dict()) for name, d in forms.items()}


def capture() -> dict:
    """Everything the config side pins, as JSON-able data."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        return {
            "rods_with_node_references": _config_case(rods_with_node_references()),
            "vectors_with_inline_and_asset_references": _config_case(
                vectors_with_inline_and_asset_references(base), base),
            "vectors_with_every_kernel": _config_case(vectors_with_every_kernel(),
                                                      weights=False),
            "refusals": _refusals(base),
            "unserialisable": _unserialisable(),
            "spec_forms": _spec_forms(),
        }


def capture_usd() -> dict:
    """The ``maddening:mappingSpecJson`` text of every mapped edge of the
    two weight-pinned graphs, and the weights a stage rebuilds."""
    from pxr import Usd  # noqa: PLC0415

    from maddening.usd.serialization import (  # noqa: PLC0415
        load_graph_from_usd,
        save_graph_to_usd,
    )

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
    pins = capture()
    pins["usd"] = capture_usd()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
