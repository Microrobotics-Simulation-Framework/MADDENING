"""
USD graph serialization -- save and load MADDENING graphs to/from USD.

This module provides :func:`save_graph_to_usd` to serialize a
``GraphManager`` (topology, parameters, coupling) to a USD stage,
and :func:`load_graph_from_usd` to reconstruct a ``GraphManager``
from a USD stage.

Edge transforms are serialized by their registered name (see
:mod:`maddening.core.transforms`).  Unregistered transforms raise
an error during serialization.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

import numpy as np
from pxr import Sdf, Usd, Vt

from maddening.core.transforms import (
    UnregisteredTransformError,
    get_transform_name,
    resolve_transform,
)

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


# ------------------------------------------------------------------
# Node type registry for deserialization
# ------------------------------------------------------------------

_NODE_CLASS_REGISTRY: dict[str, type] = {}


def register_node_class(cls: type) -> type:
    """Register a SimulationNode subclass for USD deserialization.

    The class is stored under its fully qualified name
    (``module.qualname``).  This is what ``save_graph_to_usd``
    writes as the ``maddening:nodeType`` attribute.
    """
    key = f"{cls.__module__}.{cls.__qualname__}"
    _NODE_CLASS_REGISTRY[key] = cls
    return cls


def _ensure_builtins_registered():
    """Lazily register all built-in MADDENING nodes."""
    if _NODE_CLASS_REGISTRY:
        return
    from maddening.nodes.ball import BallNode
    from maddening.nodes.heat import HeatNode
    from maddening.nodes.spring import SpringDamperNode
    from maddening.nodes.table import TableNode
    from maddening.nodes.rigid_body_2d import RigidBody2DNode
    from maddening.nodes.health_check import HealthCheckNode

    for cls in [
        BallNode, HeatNode, SpringDamperNode, TableNode,
        RigidBody2DNode, HealthCheckNode,
    ]:
        register_node_class(cls)


def _resolve_node_class(qualified_name: str) -> type:
    """Resolve a qualified name to a node class."""
    _ensure_builtins_registered()
    if qualified_name in _NODE_CLASS_REGISTRY:
        return _NODE_CLASS_REGISTRY[qualified_name]
    # Try importing the module and getting the class
    parts = qualified_name.rsplit(".", 1)
    if len(parts) == 2:
        module_path, class_name = parts
        import importlib
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            register_node_class(cls)
            return cls
        except (ImportError, AttributeError):
            pass
    raise KeyError(
        f"Node class '{qualified_name}' not found. "
        f"Register it with register_node_class() or ensure it is importable."
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

def save_graph_to_usd(
    gm: "GraphManager",
    stage: Usd.Stage,
    root_path: str = "/Simulation",
) -> None:
    """Serialize a GraphManager's topology and parameters to a USD stage.

    Creates a hierarchy of typed prims under *root_path*::

        /Simulation                         (MaddeningSimulationGraph)
        /Simulation/nodes/ball              (MaddeningNode)
        /Simulation/nodes/spring            (MaddeningNode)
        /Simulation/edges/e0                (MaddeningEdge)
        /Simulation/coupling_groups/cg0     (MaddeningCouplingGroup)
        /Simulation/external_inputs/ext0    (MaddeningExternalInput)

    Parameters
    ----------
    gm : GraphManager
        The graph manager to serialize.
    stage : Usd.Stage
        The target USD stage.
    root_path : str
        Path for the root prim.
    """
    # Validate edge transforms before writing (fail early)
    for edge in gm._edges:
        if edge.transform is not None:
            tname = get_transform_name(edge.transform)
            if tname is None:
                raise UnregisteredTransformError(
                    f"Edge {edge.source_node}.{edge.source_field} -> "
                    f"{edge.target_node}.{edge.target_field} has an "
                    f"unregistered transform "
                    f"({edge.transform.__qualname__}). "
                    f"Use @register_transform for USD serialization."
                )

    # Phase 1: create all typed prims (cannot be inside Sdf.ChangeBlock
    # for codeless schemas in usd-core 26.x)
    root_prim = stage.DefinePrim(root_path, "MaddeningSimulationGraph")

    nodes_path = root_path + "/nodes"
    node_prims = {}
    for node_name in gm._nodes:
        safe_name = _safe_prim_name(node_name)
        node_prims[node_name] = stage.DefinePrim(
            f"{nodes_path}/{safe_name}", "MaddeningNode"
        )

    edges_path = root_path + "/edges"
    edge_prims = []
    for i in range(len(gm._edges)):
        edge_prims.append(
            stage.DefinePrim(f"{edges_path}/e{i}", "MaddeningEdge")
        )

    cg_path = root_path + "/coupling_groups"
    cg_prims = []
    for i in range(len(gm._coupling_groups)):
        cg_prims.append(
            stage.DefinePrim(f"{cg_path}/cg{i}", "MaddeningCouplingGroup")
        )

    ext_path = root_path + "/external_inputs"
    ext_prims = []
    for i in range(len(gm._external_inputs)):
        ext_prims.append(
            stage.DefinePrim(f"{ext_path}/ext{i}", "MaddeningExternalInput")
        )

    # Phase 2: set attributes (can use ChangeBlock for efficiency)
    with Sdf.ChangeBlock():
        # Root attributes
        base_dt = getattr(gm, "_base_dt", None)
        if base_dt is None and gm._nodes:
            base_dt = min(s.timestep for s in gm._nodes.values())
        root_prim.GetAttribute("maddening:baseDt").Set(
            float(base_dt or 0.01)
        )
        root_prim.GetAttribute("maddening:isMultirate").Set(
            bool(gm._is_multirate)
        )

        # Node attributes
        for node_name, spec in gm._nodes.items():
            prim = node_prims[node_name]
            node_obj = spec.node
            prim.GetAttribute("maddening:nodeType").Set(
                f"{type(node_obj).__module__}.{type(node_obj).__qualname__}"
            )
            prim.GetAttribute("maddening:timestep").Set(
                float(spec.timestep)
            )
            prim.GetAttribute("maddening:paramsJson").Set(
                json.dumps(
                    _params_to_serializable(node_obj.params), default=str
                )
            )

        # Edge attributes
        for i, edge in enumerate(gm._edges):
            prim = edge_prims[i]
            prim.GetAttribute("maddening:sourceNode").Set(edge.source_node)
            prim.GetAttribute("maddening:targetNode").Set(edge.target_node)
            prim.GetAttribute("maddening:sourceField").Set(
                edge.source_field
            )
            prim.GetAttribute("maddening:targetField").Set(
                edge.target_field
            )
            prim.GetAttribute("maddening:additive").Set(edge.additive)

            if edge.transform is not None:
                tname = get_transform_name(edge.transform)
                prim.GetAttribute("maddening:transformName").Set(tname)

        # Coupling group attributes
        for i, group in enumerate(gm._coupling_groups):
            prim = cg_prims[i]
            prim.GetAttribute("maddening:nodes").Set(
                Vt.StringArray(sorted(group.nodes))
            )
            prim.GetAttribute("maddening:maxIterations").Set(
                group.max_iterations
            )
            prim.GetAttribute("maddening:tolerance").Set(
                float(group.tolerance)
            )
            prim.GetAttribute("maddening:convergenceNorm").Set(
                group.convergence_norm
            )
            prim.GetAttribute("maddening:acceleration").Set(
                group.acceleration
            )
            prim.GetAttribute("maddening:relaxation").Set(
                float(group.relaxation)
            )
            prim.GetAttribute("maddening:iterationMode").Set(
                group.iteration_mode
            )
            prim.GetAttribute("maddening:subcycling").Set(
                group.subcycling
            )
            prim.GetAttribute("maddening:boundaryInterpolation").Set(
                group.boundary_interpolation
            )
            prim.GetAttribute("maddening:diagnostics").Set(
                group.diagnostics
            )
            prim.GetAttribute("maddening:predictor").Set(
                group.predictor
            )

        # External input attributes
        for i, ext in enumerate(gm._external_inputs):
            prim = ext_prims[i]
            prim.GetAttribute("maddening:targetNode").Set(ext.target_node)
            prim.GetAttribute("maddening:targetField").Set(ext.target_field)
            prim.GetAttribute("maddening:shape").Set(
                Vt.IntArray(list(ext.shape))
            )


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

def load_graph_from_usd(
    stage: Usd.Stage,
    root_path: str = "/Simulation",
) -> "GraphManager":
    """Reconstruct a GraphManager from a USD stage.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage containing a MADDENING simulation graph.
    root_path : str
        Path of the root ``MaddeningSimulationGraph`` prim.

    Returns
    -------
    GraphManager
        A new graph manager with nodes, edges, coupling groups,
        and external inputs restored from the USD stage.
    """
    from maddening.core.graph_manager import GraphManager

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"No prim at {root_path}")

    gm = GraphManager()

    # --- Nodes ---
    nodes_prim = stage.GetPrimAtPath(root_path + "/nodes")
    node_name_map: dict[str, str] = {}  # safe_name -> original_name
    if nodes_prim.IsValid():
        for child in nodes_prim.GetChildren():
            node_type = child.GetAttribute("maddening:nodeType").Get()
            timestep = child.GetAttribute("maddening:timestep").Get()
            params_json = child.GetAttribute("maddening:paramsJson").Get()

            if not node_type:
                continue

            params = json.loads(params_json) if params_json else {}
            cls = _resolve_node_class(node_type)

            # Reconstruct the node name from the prim name
            # (we stored the safe name as the prim name; the original
            # name was used as the node name)
            node_name = child.GetName()

            # Create the node
            node = cls(name=node_name, timestep=timestep, **params)
            gm.add_node(node)
            node_name_map[child.GetName()] = node_name

    # --- Edges ---
    edges_prim = stage.GetPrimAtPath(root_path + "/edges")
    if edges_prim and edges_prim.IsValid():
        for child in edges_prim.GetChildren():
            source_node = child.GetAttribute("maddening:sourceNode").Get()
            target_node = child.GetAttribute("maddening:targetNode").Get()
            source_field = child.GetAttribute("maddening:sourceField").Get()
            target_field = child.GetAttribute("maddening:targetField").Get()
            transform_name = child.GetAttribute(
                "maddening:transformName"
            ).Get()
            additive = child.GetAttribute("maddening:additive").Get()

            transform = None
            if transform_name:
                transform = resolve_transform(transform_name)

            gm.add_edge(
                source_node,
                target_node,
                source_field,
                target_field,
                transform=transform,
                additive=bool(additive) if additive is not None else False,
            )

    # --- Coupling groups ---
    cg_prim = stage.GetPrimAtPath(root_path + "/coupling_groups")
    if cg_prim and cg_prim.IsValid():
        for child in cg_prim.GetChildren():
            nodes_attr = child.GetAttribute("maddening:nodes").Get()
            if not nodes_attr:
                continue
            node_names = list(nodes_attr)
            kwargs = {}
            for attr_name, param_name, converter in [
                ("maddening:convergenceNorm", "convergence_norm", str),
                ("maddening:acceleration", "acceleration", str),
                ("maddening:relaxation", "relaxation", float),
                ("maddening:iterationMode", "iteration_mode", str),
                ("maddening:subcycling", "subcycling", bool),
                (
                    "maddening:boundaryInterpolation",
                    "boundary_interpolation",
                    str,
                ),
                ("maddening:diagnostics", "diagnostics", bool),
                ("maddening:predictor", "predictor", str),
            ]:
                val = child.GetAttribute(attr_name).Get()
                if val is not None:
                    kwargs[param_name] = converter(val)

            max_iters_val = child.GetAttribute(
                "maddening:maxIterations"
            ).Get()
            tol_val = child.GetAttribute("maddening:tolerance").Get()

            gm.add_coupling_group(
                node_names,
                max_iterations=int(max_iters_val)
                if max_iters_val is not None
                else 10,
                tolerance=float(tol_val) if tol_val is not None else 1e-6,
                **kwargs,
            )

    # --- External inputs ---
    ext_prim = stage.GetPrimAtPath(root_path + "/external_inputs")
    if ext_prim and ext_prim.IsValid():
        for child in ext_prim.GetChildren():
            target_node = child.GetAttribute("maddening:targetNode").Get()
            target_field = child.GetAttribute("maddening:targetField").Get()
            shape_arr = child.GetAttribute("maddening:shape").Get()
            shape = tuple(shape_arr) if shape_arr else ()
            gm.add_external_input(target_node, target_field, shape=shape)

    return gm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_prim_name(name: str) -> str:
    """Convert a node name to a valid SdfPath element."""
    tokens = Sdf.Path.TokenizeIdentifier(name)
    if tokens:
        return "_".join(tokens)
    return name.replace("-", "_").replace(" ", "_").replace(".", "_")


def _params_to_serializable(params: dict) -> dict:
    """Convert node params dict to JSON-serializable form."""
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif hasattr(v, "tolist"):
            result[k] = np.asarray(v).tolist()
        else:
            result[k] = v
    return result
