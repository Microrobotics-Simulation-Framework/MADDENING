"""
USDWriter -- write MADDENING simulation frames to a USD stage.

Each call to ``write_frame`` stores one frame of simulation state
as time-sampled attributes on typed ``MaddeningNode`` prims.

Usage::

    import maddening.usd  # register schemas first
    from pxr import Usd

    stage = Usd.Stage.CreateNew("sim.usda")
    writer = USDWriter(stage, gm)
    for t in range(100):
        state = gm.step(state)
        writer.write_frame(state, float(t))
    stage.Save()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pxr import Sdf, Usd, Vt

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


class USDWriter:
    """Writes simulation state to a USD stage as time-sampled attributes.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage to write into.
    gm : GraphManager
        The compiled graph manager (used for node metadata).
    root_path : str
        SdfPath of the root prim under which node prims are created.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        gm: "GraphManager",
        root_path: str = "/Simulation",
    ):
        self._stage = stage
        self._gm = gm
        self._root_path = root_path
        self._prim_cache: dict[str, Usd.Prim] = {}
        self._attr_cache: dict[tuple[str, str], Usd.Attribute] = {}

        # Create root prim as MaddeningSimulationGraph
        root_prim = stage.DefinePrim(root_path, "MaddeningSimulationGraph")
        root_prim.GetAttribute("maddening:baseDt").Set(
            float(gm._base_dt if hasattr(gm, "_base_dt") else 0.01)
        )
        root_prim.GetAttribute("maddening:isMultirate").Set(
            bool(gm._is_multirate)
        )

        # Create nodes container
        self._nodes_path = root_path + "/nodes"

    def _get_or_create_prim(self, node_name: str) -> Usd.Prim:
        """Get or create a MaddeningNode prim for the given node."""
        if node_name in self._prim_cache:
            return self._prim_cache[node_name]

        # Sanitize node name for SdfPath (replace invalid chars)
        safe_name = Sdf.Path.TokenizeIdentifier(node_name)
        if safe_name:
            safe_name = "_".join(safe_name)
        else:
            safe_name = node_name.replace("-", "_").replace(" ", "_")

        prim_path = f"{self._nodes_path}/{safe_name}"
        prim = self._stage.DefinePrim(prim_path, "MaddeningNode")

        # Set static metadata
        if node_name in self._gm._nodes:
            spec = self._gm._nodes[node_name]
            node_obj = spec.node
            prim.GetAttribute("maddening:nodeType").Set(
                f"{type(node_obj).__module__}.{type(node_obj).__qualname__}"
            )
            prim.GetAttribute("maddening:timestep").Set(float(spec.timestep))
            import json
            prim.GetAttribute("maddening:paramsJson").Set(
                json.dumps(
                    _params_to_serializable(node_obj.params),
                    default=str,
                )
            )

        self._prim_cache[node_name] = prim
        return prim

    def _get_or_create_attr(
        self, prim: Usd.Prim, field: str, value: np.ndarray
    ) -> Usd.Attribute:
        """Get or create a time-sampled attribute for a state field."""
        prim_path = prim.GetPath().pathString
        key = (prim_path, field)
        if key in self._attr_cache:
            return self._attr_cache[key]

        # Determine SdfValueType from numpy array
        attr_name = f"state:{field}"
        np_val = np.asarray(value)

        if np_val.ndim == 0:
            sdf_type = Sdf.ValueTypeNames.Double
        elif np_val.ndim == 1:
            sdf_type = Sdf.ValueTypeNames.FloatArray
        else:
            # Higher-dimensional: flatten and store shape as a separate attr
            sdf_type = Sdf.ValueTypeNames.FloatArray
            shape_attr_name = f"state:{field}:shape"
            if not prim.GetAttribute(shape_attr_name).IsValid():
                shape_attr = prim.CreateAttribute(
                    shape_attr_name, Sdf.ValueTypeNames.IntArray
                )
                shape_attr.Set(Vt.IntArray(list(np_val.shape)))

        attr = prim.CreateAttribute(attr_name, sdf_type)
        self._attr_cache[key] = attr
        return attr

    def write_frame(self, state: dict, time: float) -> None:
        """Write one frame of simulation state as time-sampled attributes.

        Parameters
        ----------
        state : dict
            Full state dict ``{node_name: {field: value, ...}, ...}``.
            The special ``"_meta"`` key is skipped.
        time : float
            USD time code for this frame.
        """
        # Phase 1: ensure all prims and attributes exist (typed
        # prim creation via DefinePrim cannot happen inside
        # Sdf.ChangeBlock for codeless schemas in usd-core 26.x)
        prim_attr_pairs = []
        for node_name, node_state in state.items():
            if node_name == "_meta":
                continue
            if not isinstance(node_state, dict):
                continue
            prim = self._get_or_create_prim(node_name)
            for field, value in node_state.items():
                attr = self._get_or_create_attr(prim, field, value)
                prim_attr_pairs.append((attr, value))

        # Phase 2: write time-sampled values (ChangeBlock is safe here)
        with Sdf.ChangeBlock():
            for attr, value in prim_attr_pairs:
                np_val = np.asarray(value)
                if np_val.ndim == 0:
                    attr.Set(float(np_val), time)
                elif np_val.ndim == 1:
                    attr.Set(Vt.FloatArray(np_val.tolist()), time)
                else:
                    attr.Set(
                        Vt.FloatArray(np_val.ravel().tolist()), time
                    )


def _params_to_serializable(params: dict) -> dict:
    """Convert node params dict to JSON-serializable form."""
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif hasattr(v, "tolist"):
            # JAX arrays
            result[k] = np.asarray(v).tolist()
        else:
            result[k] = v
    return result
