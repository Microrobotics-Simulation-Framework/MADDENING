"""
USD graph serialization -- save and load MADDENING graphs to/from USD.

This module provides :func:`save_graph_to_usd` to serialize a
``GraphManager`` (topology, parameters, coupling) to a USD stage,
and :func:`load_graph_from_usd` to reconstruct a ``GraphManager``
from a USD stage.

Edge transforms are serialized by their registered name (see
:mod:`maddening.core.transforms`).  Unregistered transforms raise
an error during serialization.  Edge interface mappings are serialized
as their :class:`~maddening.core.coupling.mapping_spec.MappingSpec`
(JSON in ``maddening:mappingSpecJson``) and rebuilt on load; the
weights themselves live in checkpoints, never in the stage.

A node's prim name is a mangled, de-duplicated USD identifier -- node
names may legally contain characters (``-``, ``.``, spaces, parentheses)
that a prim name may not, and may legally start with a digit.  The node's
own name is therefore written to ``maddening:nodeName`` and restored from
there; the prim name is only a path element.

Backward compatibility
----------------------

Two attributes were added in 0.4.0 -- ``maddening:dtype`` on an external
input (its declared dtype, which no stage carried before, so an ``int32``
or ``bool`` input reloaded as ``float32``) and
``maddening:paramArrayShapesJson`` on a node (the shapes of zero-size
array params, which ``tolist()`` flattens away).  A stage written without
either loads exactly as it did before: the dtype falls back to the
declaration default and the param keeps the shape its nested lists imply.

Trust boundary
--------------

**A ``.usda`` / ``.usdc`` stage is untrusted input**, exactly like an FMI
frame or a mapping asset: it names the Python class of every node it
carries, and a stage can name any class at all.  :func:`load_graph_from_usd`
therefore instantiates only classes the *caller* has allowed -- the
built-ins, whatever :func:`register_node_class` registered, and whatever
the ``node_registry`` argument passes in -- the same rule
:meth:`GraphManager.from_dict` has always applied to a config.

Until 0.4.0 the reader fell back to ``importlib.import_module`` on the
stage's own string, so merely *opening* an untrusted stage ran the named
module's import-time code (the ``TypeError`` that followed arrived far too
late).  That fallback is now off unless the caller passes
``allow_import=True``, which is only ever appropriate for a stage from a
source you would run a script from.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
from pxr import Sdf, Usd, Vt

from maddening.core.coupling.group import coupling_group_kwargs
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


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered():
    """Lazily register all built-in MADDENING nodes.

    Guarded by its own flag rather than by the registry being empty: a
    caller that runs ``register_node_class`` for one of its own classes
    before the first load would otherwise keep the built-ins out of the
    registry for ever.  That went unnoticed while an unregistered class
    was resolved by importing it.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
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


def _resolve_node_class(
    qualified_name: str,
    node_registry: Optional[dict[str, type]] = None,
    *,
    allow_import: bool = False,
) -> type:
    """Resolve a stage's ``maddening:nodeType`` string to a node class.

    The string comes off the stage, so it is untrusted: only classes the
    caller has allowed are instantiated.  In order, that is
    ``node_registry`` (by qualified name, then by bare class name, so a
    registry written for :meth:`GraphManager.from_dict` works here too),
    then the process-wide registry of built-ins and anything
    :func:`register_node_class` was called with.

    ``allow_import`` restores the pre-0.4.0 behaviour of importing the
    module the stage names.  **Importing runs that module's top-level
    code**, so it is opt-in and belongs only to a caller who trusts the
    stage as much as a script.
    """
    _ensure_builtins_registered()
    if node_registry:
        cls = node_registry.get(qualified_name)
        if cls is None:
            cls = node_registry.get(qualified_name.rsplit(".", 1)[-1])
        if cls is not None:
            return cls
    if qualified_name in _NODE_CLASS_REGISTRY:
        return _NODE_CLASS_REGISTRY[qualified_name]
    parts = qualified_name.rsplit(".", 1)
    if allow_import and len(parts) == 2:
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
        f"Node class '{qualified_name}' is not allowed by this load. "
        f"A USD stage is untrusted input and names its own Python classes, "
        f"so only classes the caller allows are instantiated.  Pass it in: "
        f"load_graph_from_usd(stage, node_registry={{'{qualified_name}': "
        f"{parts[-1]}}}), or call register_node_class({parts[-1]}) first.  "
        f"To go back to importing whatever the stage names -- which runs "
        f"'{parts[0] if len(parts) == 2 else qualified_name}' at import time "
        f"and is safe only for a stage you trust as much as a script -- pass "
        f"allow_import=True."
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

#: ``(USD attribute, CouplingGroup field, reader conversion)`` for every
#: scalar field of a coupling group.  The writer and the reader both walk
#: this one table, so the stage cannot come to carry a field one of them
#: does not know about — the way it carried eleven of nineteen until the
#: config grew a ``coupling_groups`` key and the two formats were made to
#: say the same thing.  ``nodes`` (a string array) and
#: ``accelerated_fields`` (JSON, since USD has no dict-of-arrays type) are
#: handled beside it; ``tests/usd/test_usd_serialization.py`` asserts the
#: three together cover ``dataclasses.fields(CouplingGroup)``, so a field
#: added to the group fails that test until it is written here too.
_COUPLING_GROUP_ATTRS: tuple[tuple[str, str, type], ...] = (
    ("maddening:maxIterations", "max_iterations", int),
    ("maddening:tolerance", "tolerance", float),
    ("maddening:convergenceNorm", "convergence_norm", str),
    ("maddening:atol", "atol", float),
    ("maddening:rtol", "rtol", float),
    ("maddening:diagnostics", "diagnostics", bool),
    ("maddening:acceleration", "acceleration", str),
    ("maddening:relaxation", "relaxation", float),
    ("maddening:iterationMode", "iteration_mode", str),
    ("maddening:subcycling", "subcycling", bool),
    ("maddening:boundaryInterpolation", "boundary_interpolation", str),
    ("maddening:jacobianReuse", "jacobian_reuse", int),
    ("maddening:waveformIterations", "waveform_iterations", int),
    ("maddening:predictor", "predictor", str),
    ("maddening:solver", "solver", str),
    ("maddening:strictConvergence", "strict_convergence", bool),
    ("maddening:linearSolver", "linear_solver", str),
)

#: The two fields ``_COUPLING_GROUP_ATTRS`` cannot describe.
_COUPLING_GROUP_NODES_ATTR = "maddening:nodes"
_COUPLING_GROUP_FIELDS_ATTR = "maddening:acceleratedFieldsJson"


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
    # Validate edge transforms / mappings before writing (fail early)
    from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
        check_mapping_serialisable,
    )
    _resolve_points = gm.point_resolver()
    for edge in gm._edges:
        if edge.mapping is not None:
            check_mapping_serialisable(edge.mapping, edge_key=edge.key,
                                       resolve_points=_resolve_points)
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
    prim_names = _prim_names(gm._nodes)
    for node_name in gm._nodes:
        node_prims[node_name] = stage.DefinePrim(
            f"{nodes_path}/{prim_names[node_name]}", "MaddeningNode"
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
            # The prim name is a mangled, de-duplicated identifier; the
            # node's own name is the one every edge, coupling group and
            # external input refers to, so it is written out verbatim.
            prim.CreateAttribute(
                "maddening:nodeName", Sdf.ValueTypeNames.String,
            ).Set(node_name)
            prim.GetAttribute("maddening:timestep").Set(
                float(spec.timestep)
            )
            # Effective params: constructor args with the live (possibly
            # calibrated) ``gm.params`` values written over them, so the
            # reloaded node is the one that was calibrated.
            node_params = (
                gm.effective_node_params(node_name)
                if spec.accepts_params else node_obj.params
            )
            prim.GetAttribute("maddening:paramsJson").Set(
                _params_json(node_params, node_name)
            )
            lost = _degenerate_shapes(node_params)
            if lost:
                # ``np.zeros((0, 3)).tolist()`` is ``[]``: every axis after a
                # zero-length one vanishes, and the param reloads as shape
                # ``(0,)``.  Record those shapes beside the JSON (only when
                # there are any, so no existing stage changes) and reshape on
                # load.  An older reader ignores the attribute and behaves as
                # it did before.
                attr = prim.CreateAttribute(
                    _PARAM_SHAPES_ATTR, Sdf.ValueTypeNames.String, custom=True,
                )
                attr.Set(json.dumps(lost, sort_keys=True))
            overrides = gm.param_spec_overrides().get(node_name)
            if overrides:
                attr = prim.CreateAttribute(
                    "maddening:paramSpecOverridesJson",
                    Sdf.ValueTypeNames.String,
                )
                attr.Set(json.dumps({k: s.to_dict() for k, s in overrides.items()}))

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
            # Declared units are part of the EdgeSpec and of the config
            # form; without them a reloaded graph stops unit-checking an
            # edge the original was checking.
            for attr_name, value in (("maddening:sourceUnits", edge.source_units),
                                     ("maddening:targetUnits", edge.target_units)):
                if value is not None:
                    prim.CreateAttribute(
                        attr_name, Sdf.ValueTypeNames.String,
                    ).Set(value)
            if edge.mapping is not None:
                # The spec (kind, hyper-parameters, point references) plus
                # the shape ``describe()`` adds, as one JSON string — the
                # idiom used for ParamSpec overrides above.
                attr = prim.CreateAttribute(
                    "maddening:mappingSpecJson", Sdf.ValueTypeNames.String,
                )
                attr.Set(json.dumps(edge.mapping.describe()))
                # ParamSpec overrides keyed by the edge key (a mapping whose
                # weights were made trainable for sysid) belong to the edge,
                # not to any node prim, so they are written here.
                edge_overrides = gm.param_spec_overrides().get(edge.key)
                if edge_overrides:
                    ov_attr = prim.CreateAttribute(
                        "maddening:paramSpecOverridesJson", Sdf.ValueTypeNames.String,
                    )
                    ov_attr.Set(json.dumps(
                        {k: s.to_dict() for k, s in edge_overrides.items()}))

        # Coupling group attributes.  ``CouplingGroup.to_dict`` is the
        # same plain-data view the config writes, so the stage and the
        # config carry one field set from one source.
        for i, group in enumerate(gm._coupling_groups):
            prim = cg_prims[i]
            stored = group.to_dict()
            prim.GetAttribute(_COUPLING_GROUP_NODES_ATTR).Set(
                Vt.StringArray(stored["nodes"])
            )
            for attr_name, field_name, _convert in _COUPLING_GROUP_ATTRS:
                prim.GetAttribute(attr_name).Set(stored[field_name])
            accelerated = stored["accelerated_fields"]
            prim.GetAttribute(_COUPLING_GROUP_FIELDS_ATTR).Set(
                "" if accelerated is None
                else json.dumps(accelerated, sort_keys=True)
            )

        # External input attributes
        for i, ext in enumerate(gm._external_inputs):
            prim = ext_prims[i]
            prim.GetAttribute("maddening:targetNode").Set(ext.target_node)
            prim.GetAttribute("maddening:targetField").Set(ext.target_field)
            prim.GetAttribute("maddening:shape").Set(
                Vt.IntArray(list(ext.shape))
            )
            # The declared dtype, which the stage did not carry at all: an
            # int32 or bool external input reloaded as float32.  Written as
            # the dtype's name beside the shape, since the two are the same
            # piece of information about the same array.
            prim.CreateAttribute(
                _EXT_DTYPE_ATTR, Sdf.ValueTypeNames.String, custom=True,
            ).Set(_dtype_name(ext.dtype))


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

def load_graph_from_usd(
    stage: Usd.Stage,
    root_path: str = "/Simulation",
    base_dir=None,
    *,
    node_registry: Optional[dict[str, type]] = None,
    allow_import: bool = False,
) -> "GraphManager":
    """Reconstruct a GraphManager from a USD stage.

    A stage is untrusted input: it names the Python class of every node
    it carries.  Only classes the caller allows are instantiated -- see
    ``node_registry`` and ``allow_import``, and the *Trust boundary*
    section of this module.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage containing a MADDENING simulation graph.
    root_path : str
        Path of the root ``MaddeningSimulationGraph`` prim.
    base_dir : path-like, optional
        Directory that ``{"asset": ...}`` point references of edge
        mappings are relative to.  Defaults to the directory of the
        stage's root layer when it is a file, else the working
        directory.
    node_registry : dict, optional
        Node classes this load may instantiate, keyed by qualified name
        (``"maddening.nodes.ball.BallNode"``) or by bare class name
        (``"BallNode"``), so a registry written for
        :meth:`GraphManager.from_dict` works unchanged.  Searched before
        the built-ins and the :func:`register_node_class` registry, which
        remain available whether or not this is given.
    allow_import : bool, default False
        Import the module a stage names when no registered class matches.
        **Importing executes that module**, so this is only for a stage
        you trust as much as a script; before 0.4.0 it was the
        unconditional behaviour.

    Returns
    -------
    GraphManager
        A new graph manager with nodes, edges, coupling groups,
        and external inputs restored from the USD stage.

    Raises
    ------
    KeyError
        If the stage names a node class that is neither registered nor in
        ``node_registry``.  The message names the class and says what to
        pass.
    """
    from maddening.core.graph_manager import GraphManager

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"No prim at {root_path}")

    gm = GraphManager()
    if base_dir is None:
        real = stage.GetRootLayer().realPath
        if real:
            from pathlib import Path  # noqa: PLC0415
            base_dir = Path(real).parent

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
            params = _restore_param_shapes(child, params)
            cls = _resolve_node_class(node_type, node_registry,
                                      allow_import=allow_import)

            # The node's own name, which need not be a legal prim name
            # (``"a-b"``, ``"1st"``).  Stages written before
            # ``maddening:nodeName`` existed only have the prim name.
            name_attr = child.GetAttribute("maddening:nodeName")
            node_name = (name_attr.Get() if name_attr else None) or child.GetName()

            # Create the node
            node = cls(name=node_name, timestep=timestep, **params)
            gm.add_node(node)
            node_name_map[child.GetName()] = node_name

            overrides_attr = child.GetAttribute("maddening:paramSpecOverridesJson")
            overrides_json = overrides_attr.Get() if overrides_attr else None
            if overrides_json:
                import warnings  # noqa: PLC0415

                from maddening.core.params import ParamSpec  # noqa: PLC0415
                for key, spec_dict in json.loads(overrides_json).items():
                    try:
                        gm.set_param_spec(node_name, key, ParamSpec.from_dict(spec_dict))
                    except (KeyError, ValueError) as exc:
                        # The node class changed since the stage was
                        # written (parameter renamed, node no longer takes
                        # params): keep loading, say what was dropped.
                        warnings.warn(
                            f"ignoring ParamSpec override {node_name}.{key} from "
                            f"the USD stage: {exc}", RuntimeWarning, stacklevel=2,
                        )

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

            mapping = None
            spec_attr = child.GetAttribute("maddening:mappingSpecJson")
            spec_json = spec_attr.Get() if spec_attr else None
            if spec_json:
                where = (f"{source_node}.{source_field} -> "
                         f"{target_node}.{target_field}")
                try:
                    spec_dict = json.loads(spec_json)
                except json.JSONDecodeError as exc:
                    # A hand-edited / truncated attribute must name its edge
                    # like every other rebuild failure does.
                    from maddening.core.coupling.mapping_spec import (  # noqa: PLC0415
                        MappingRebuildError,
                    )
                    raise MappingRebuildError(where, None, exc) from exc
                mapping = gm._rebuild_mapping(  # noqa: SLF001
                    {"source_node": source_node, "target_node": target_node,
                     "source_field": source_field, "target_field": target_field,
                     "mapping": spec_dict},
                    gm.point_resolver(base_dir),
                )

            units = {}
            for key, attr_name in (("source_units", "maddening:sourceUnits"),
                                   ("target_units", "maddening:targetUnits")):
                attr = child.GetAttribute(attr_name)
                units[key] = attr.Get() if attr else None

            gm.add_edge(
                source_node,
                target_node,
                source_field,
                target_field,
                transform=transform,
                additive=bool(additive) if additive is not None else False,
                mapping=mapping,
                **units,
            )

            edge_overrides_attr = child.GetAttribute("maddening:paramSpecOverridesJson")
            edge_overrides_json = (
                edge_overrides_attr.Get() if edge_overrides_attr else None
            )
            if edge_overrides_json and gm.edges:
                import warnings  # noqa: PLC0415

                from maddening.core.params import ParamSpec  # noqa: PLC0415
                edge_key = gm.edges[-1].key
                for key, ov_dict in json.loads(edge_overrides_json).items():
                    try:
                        gm.set_param_spec(edge_key, key, ParamSpec.from_dict(ov_dict))
                    except (KeyError, ValueError) as exc:
                        # The mapping changed since the stage was written
                        # (different factory, different weight names): keep
                        # loading, say what was dropped.
                        warnings.warn(
                            f"ignoring ParamSpec override {edge_key}.{key} from "
                            f"the USD stage: {exc}", RuntimeWarning, stacklevel=2,
                        )

    # --- Coupling groups ---
    cg_prim = stage.GetPrimAtPath(root_path + "/coupling_groups")
    if cg_prim and cg_prim.IsValid():
        for child in cg_prim.GetChildren():
            nodes_attr = child.GetAttribute(_COUPLING_GROUP_NODES_ATTR).Get()
            if not nodes_attr:
                continue
            # Rebuilt as the same plain-data dict the config stores, then
            # through the same ``add_coupling_group``: one loader for two
            # formats.  An attribute a stage does not author reads back as
            # the schema fallback, which is the dataclass default, so a
            # stage written before a field existed loads as it always did.
            stored: dict = {"nodes": list(nodes_attr)}
            for attr_name, field_name, convert in _COUPLING_GROUP_ATTRS:
                val = child.GetAttribute(attr_name).Get()
                if val is not None:
                    stored[field_name] = convert(val)
            accelerated = child.GetAttribute(_COUPLING_GROUP_FIELDS_ATTR).Get()
            if accelerated:
                stored["accelerated_fields"] = json.loads(accelerated)

            nodes, kwargs = coupling_group_kwargs(stored)
            try:
                gm.add_coupling_group(nodes, **kwargs)
            except (KeyError, TypeError, ValueError) as exc:
                # A stage is as hand-editable as a config, and the
                # constructor's complaint names the field but not the prim
                # it came from.  Name it, as the config loader names the
                # index of the group it was reading.
                detail = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
                raise ValueError(
                    f"{child.GetPath()} cannot be rebuilt: {detail}"
                ) from exc

    # --- External inputs ---
    ext_prim = stage.GetPrimAtPath(root_path + "/external_inputs")
    if ext_prim and ext_prim.IsValid():
        for child in ext_prim.GetChildren():
            target_node = child.GetAttribute("maddening:targetNode").Get()
            target_field = child.GetAttribute("maddening:targetField").Get()
            shape_arr = child.GetAttribute("maddening:shape").Get()
            shape = tuple(shape_arr) if shape_arr else ()
            dtype = _ext_dtype(child)
            if dtype is None:
                # A stage written before the attribute existed says nothing
                # about the dtype, so it gets the declaration default, which
                # is what such a stage has always loaded as.
                gm.add_external_input(target_node, target_field, shape=shape)
            else:
                gm.add_external_input(target_node, target_field, shape=shape,
                                      dtype=dtype)

    return gm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_prim_name(name: str) -> str:
    """A node name as a **valid** ``SdfPath`` element.

    Node names are only forbidden ``/``, ``#`` and ``->`` (see
    ``GraphManager.add_node``), so ``"a-b"``, ``"with space"``,
    ``"co2(aq)"`` and ``"1st"`` are all legal nodes and none of them is a
    legal prim name.  The result is always a valid identifier -- the
    previous version could return one that was not (``"1st"``), which
    made ``DefinePrim`` raise on an ill-formed path.

    The mangling is lossy and not injective, so it is only ever the prim
    *name*: the node's own name is written to ``maddening:nodeName`` and
    that is what the reader restores.
    """
    tokens = Sdf.Path.TokenizeIdentifier(name)
    candidate = "_".join(tokens) if tokens else re.sub(r"\W", "_", name, flags=re.UNICODE)
    if not Sdf.Path.IsValidIdentifier(candidate):
        # A leading digit (or an empty result) is the remaining case.
        candidate = f"n_{candidate}"
    if not Sdf.Path.IsValidIdentifier(candidate):  # pragma: no cover - belt and braces
        candidate = "node"
    return candidate


def _prim_names(node_names) -> dict[str, str]:
    """``{node name: prim name}``, with collisions broken by a suffix.

    ``"a-b"`` and ``"a.b"`` are two different nodes that mangle to the
    same ``a_b``; without this the second one overwrote the first and the
    reload silently lost a node.
    """
    out: dict[str, str] = {}
    used: set[str] = set()
    for name in node_names:
        base = _safe_prim_name(name)
        candidate, suffix = base, 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        out[name] = candidate
    return out


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


#: Shapes of zero-size array params, which ``tolist()`` cannot carry.
_PARAM_SHAPES_ATTR = "maddening:paramArrayShapesJson"

#: The declared dtype of an external input.
_EXT_DTYPE_ATTR = "maddening:dtype"


def _allowed_dtypes() -> dict[str, np.dtype]:
    """The dtypes an external input may declare, by name.

    An allowlist rather than ``np.dtype(name)`` on the stage's own string:
    a stage is untrusted input, and ``np.dtype`` accepts far more than a
    boundary array can sensibly be (object arrays, structured records).
    """
    import jax.numpy as jnp  # noqa: PLC0415 - keeps module import light
    names: dict[str, np.dtype] = {}
    for dt in (np.bool_, np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64,
               np.float16, np.float32, np.float64, jnp.bfloat16):
        names[np.dtype(dt).name] = np.dtype(dt)
    return names


def _dtype_name(dtype) -> str:
    """``dtype`` as the name the stage stores (``"int32"``, ``"bool"``)."""
    return np.dtype(dtype).name


def _ext_dtype(prim):
    """The external input's declared dtype, or ``None`` for a stage that
    does not carry one (every stage written before 0.4.0)."""
    attr = prim.GetAttribute(_EXT_DTYPE_ATTR)
    name = attr.Get() if attr else None
    if not name:
        return None
    dtype = _allowed_dtypes().get(str(name))
    if dtype is None:
        import warnings  # noqa: PLC0415
        warnings.warn(
            f"ignoring external-input dtype {name!r} from the USD stage: not "
            f"a dtype a boundary array may declare; using the default",
            RuntimeWarning, stacklevel=2,
        )
    return dtype


def _degenerate_shapes(params: dict) -> dict[str, list[int]]:
    """``{key: shape}`` for every array param whose shape ``tolist()``
    loses -- an array with a zero-length axis and more than one axis."""
    out: dict[str, list[int]] = {}
    for k, v in params.items():
        shape = getattr(v, "shape", None)
        if shape is not None and len(shape) > 1 and 0 in tuple(shape):
            out[k] = [int(d) for d in shape]
    return out


def _restore_param_shapes(prim, params: dict) -> dict:
    """Undo :func:`_degenerate_shapes` for a stage that recorded them."""
    attr = prim.GetAttribute(_PARAM_SHAPES_ATTR)
    raw = attr.Get() if attr else None
    if not raw:
        return params
    for key, shape in json.loads(raw).items():
        if key in params:
            params[key] = np.asarray(params[key]).reshape(tuple(shape))
    return params


def _params_json(params: dict, node_name: str) -> str:
    """The ``maddening:paramsJson`` text for one node's params.

    No ``default=str``: a param the JSON encoder cannot represent used to
    be written as its ``repr`` and reload as that string -- a
    ``static_data_provider`` object came back as
    ``"<Provider /data/mesh.vtu>"`` -- while ``GraphManager.to_dict`` +
    ``json.dumps`` raised a ``TypeError`` for the same graph.  The two
    serialisers now agree, and they agree on the answer that fails at save
    time rather than on the one that fails much later somewhere else.
    """
    try:
        return json.dumps(_params_to_serializable(params))
    except TypeError as exc:
        bad = sorted(
            k for k, v in _params_to_serializable(params).items()
            if not _json_representable(v)
        )
        raise TypeError(
            f"node {node_name!r}: parameter(s) {bad} cannot be written to a "
            f"USD stage ({exc}).  A param that is not JSON-representable has "
            f"to be rebuilt by the node's constructor from something that "
            f"is, or kept out of ``params``; writing its repr() would reload "
            f"it as a string."
        ) from exc


def _json_representable(value) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True
