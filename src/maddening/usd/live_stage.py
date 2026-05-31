"""LiveStage — generic per-timestep USD writer pulled into MADDENING (v0.3.0 §A3).

The plan deliberately *demoted* A3 from "substantial new feature" to
"small consolidation": the live-stage architecture already existed in
MIME's :mod:`mime.viz.stage_bridge`.  MICROROBOTICA already rendered
those stages.  What was missing was a domain-neutral generalisation
that non-MIME MADDENING consumers (the bouncing-ball demo, future
non-microrobotics frameworks) could use without depending on MIME.

This module pulls the generic core into MADDENING.  The
domain-specific bits — registering robots with cylinder defaults,
registering magnetic-field arrows, flow-field cross-sections — stay
in MIME's ``viz.stage_bridge`` (now reframed as a thin extension that
imports :class:`LiveStage` and adds domain registrations).

Design split

Generic (this module)
    - Stage creation (in-memory or on a pre-existing
      :class:`pxr.Usd.Stage`).
    - Stage metadata (up-axis, metersPerUnit).
    - Default ``/World`` scope + default camera.
    - Dynamic-prim registry: any caller can register a
      :class:`PrimUpdater` callback per prim path; the
      registered updaters fire each timestep wrapped in
      ``Sdf.ChangeBlock`` for batched notifications.
    - Materials (UsdPreviewSurface), dome lights, ground planes.
    - Reference / payload geometry loading.
    - Export to disk.

Domain-specific (stays in MIME)
    - ``register_robot`` with cylinder / sphere / mesh defaults
      driven by ``mime.core.geometry.GeometrySource``.
    - ``register_field`` with arrow-specific scaling.
    - Flow-field cross-section register + per-frame colormap update.

Stability
~~~~~~~~~
The generic surface is tagged ``@stability(EVOLVING)`` — the wire
format (USD stage layout) is settled, but new generic primitives may
be added.  v0.4.0+ will harden specific shapes; the freeze happens at
the M4 (v0.9.0) stability gate per STACK_V1.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


logger = logging.getLogger(__name__)


# USD imports are optional — the viz layer should not break the core package.
try:
    from pxr import Usd, UsdGeom, Gf, Sdf  # type: ignore
    try:
        from pxr import UsdShade, UsdLux  # type: ignore
        _HAS_USD_SHADE = True
    except ImportError:
        _HAS_USD_SHADE = False
    _HAS_USD = True
except ImportError:  # pragma: no cover — handled at first use
    _HAS_USD = False
    _HAS_USD_SHADE = False


def _require_usd() -> None:
    if not _HAS_USD:
        raise ImportError(
            "LiveStage requires OpenUSD Python bindings (pxr). "
            "Install with: pip install usd-core",
        )


# ---------------------------------------------------------------------------
# Utility converters — exported for domain extensions to reuse.
# ---------------------------------------------------------------------------


def quat_wxyz_to_gf(q: np.ndarray) -> "Gf.Quatf":
    """Convert a ``[w, x, y, z]`` numpy quaternion to :class:`Gf.Quatf`."""
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def vec3_to_gf(v: np.ndarray) -> "Gf.Vec3d":
    """Convert a 3-element numpy vector to :class:`Gf.Vec3d`."""
    return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))


def quat_from_z_to_direction(direction: np.ndarray) -> np.ndarray:
    """Return ``[w, x, y, z]`` rotating the +Z axis to ``direction``.

    ``direction`` does not have to be unit-length — it's normalised
    inside.  Used by MIME's arrow rendering and any other consumer
    that wants to align a Z-axis-aligned prim with a vector field.
    """
    direction = np.asarray(direction, dtype=np.float64)
    mag = np.linalg.norm(direction)
    if mag < 1e-30:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction = direction / mag
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(dot)
    half = angle / 2.0
    return np.array([
        math.cos(half),
        axis[0] * math.sin(half),
        axis[1] * math.sin(half),
        axis[2] * math.sin(half),
    ])


# ---------------------------------------------------------------------------
# PrimUpdater protocol — the extension point domain extensions implement.
# ---------------------------------------------------------------------------


class PrimUpdater(Protocol):
    """Callback that writes per-frame attributes to one registered prim.

    The ``update(stage, prim_path, node_state, time_code)`` callback
    is invoked once per :meth:`LiveStage.update` call for each
    registered prim.  Implementations should read the relevant
    fields from ``node_state`` (the inner node's state dict) and
    write them to the USD prim using ``Sdf.ChangeBlock`` — the
    enclosing :meth:`LiveStage.update` already opens a single
    change block, so callers don't need to nest.
    """

    def __call__(
        self,
        stage: "Usd.Stage",
        prim_path: str,
        node_state: dict[str, Any],
        time_code: Optional[Any] = None,
    ) -> None: ...


@dataclass
class _RegisteredPrim:
    node_name: str
    prim_path: str
    updater: PrimUpdater


# ---------------------------------------------------------------------------
# LiveStage — generic per-timestep USD writer.
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EVOLVING)
class LiveStage:
    """Per-timestep USD stage writer (generic — domain-neutral).

    Parameters
    ----------
    stage : Usd.Stage, optional
        Pre-existing stage to write to.  If ``None``, an in-memory
        stage is created.
    up_axis : {"Z", "Y"}
        Stage up axis.  Default ``"Z"``.
    meters_per_unit : float
        Stage metersPerUnit.  Default ``1.0`` (SI metres).

    Notes
    -----
    A non-MIME consumer (e.g. the bouncing-ball quickstart) typically
    looks like::

        stage = LiveStage()
        stage.add_dome_light()
        stage.add_ground_plane()
        stage.register_prim(
            node_name="ball",
            prim_path="/World/Ball",
            updater=_translate_only_updater,
        )

    where ``_translate_only_updater`` is a small function the caller
    writes — see :func:`make_translate_updater` for the typical case.
    """

    def __init__(
        self,
        stage: Optional[Any] = None,
        up_axis: str = "Z",
        meters_per_unit: float = 1.0,
    ) -> None:
        _require_usd()
        if up_axis not in ("Z", "Y"):
            raise ValueError(
                f"LiveStage: up_axis must be 'Z' or 'Y' (got {up_axis!r})",
            )

        if stage is None:
            self._stage = Usd.Stage.CreateInMemory()
        else:
            self._stage = stage

        UsdGeom.SetStageUpAxis(
            self._stage,
            UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y,
        )
        UsdGeom.SetStageMetersPerUnit(self._stage, meters_per_unit)

        self._world = UsdGeom.Xform.Define(self._stage, "/World")
        self._dynamic_prims: list[_RegisteredPrim] = []

        self._camera_path = "/World/Camera"
        self._setup_default_camera()

    @property
    def stage(self) -> Any:
        """The live USD stage."""
        return self._stage

    # -- Default scene dressing --------------------------------------------

    def _setup_default_camera(self) -> None:
        cam = UsdGeom.Camera.Define(self._stage, self._camera_path)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.0001, 100.0))
        cam.GetFocalLengthAttr().Set(50.0)
        xform = UsdGeom.Xformable(cam.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.005, -0.01, 0.005))

    def add_dome_light(
        self,
        prim_path: str = "/Lights/Dome",
        intensity: float = 500.0,
    ) -> None:
        """Add an ambient dome light for scene illumination."""
        if not _HAS_USD_SHADE:
            return
        light = UsdLux.DomeLight.Define(self._stage, prim_path)
        light.GetIntensityAttr().Set(intensity)

    def add_ground_plane(
        self,
        prim_path: str = "/World/Environment/Ground",
        size: float = 0.05,
        offset: float = -0.006,
        normal: str = "Y",
        color: tuple = (0.82, 0.79, 0.75),
    ) -> None:
        """Add a ground-plane mesh.

        Parameters
        ----------
        normal : {"X", "Y", "Z"}
            Which axis the plane is perpendicular to.  The plane is
            offset by ``offset`` along this axis.  Default ``"Y"``
            (offset = -0.006 m below origin), which is what MIME's
            scene uses (and what was hard-coded into the old
            ``StageBridge.add_ground_plane``).
        """
        s = size
        if normal == "Y":
            pts = [
                Gf.Vec3f(-s, offset, -s),
                Gf.Vec3f(s, offset, -s),
                Gf.Vec3f(s, offset, s),
                Gf.Vec3f(-s, offset, s),
            ]
        elif normal == "Z":
            pts = [
                Gf.Vec3f(-s, -s, offset),
                Gf.Vec3f(s, -s, offset),
                Gf.Vec3f(s, s, offset),
                Gf.Vec3f(-s, s, offset),
            ]
        elif normal == "X":
            pts = [
                Gf.Vec3f(offset, -s, -s),
                Gf.Vec3f(offset, s, -s),
                Gf.Vec3f(offset, s, s),
                Gf.Vec3f(offset, -s, s),
            ]
        else:
            raise ValueError(f"normal must be X/Y/Z (got {normal!r})")
        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        mesh.GetPointsAttr().Set(pts)
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # -- Materials ---------------------------------------------------------

    def create_material(
        self,
        name: str,
        diffuse_color: tuple = (0.8, 0.8, 0.8),
        opacity: float = 1.0,
        roughness: float = 0.5,
        metallic: float = 0.0,
        ior: float = 1.5,
        specular_color: tuple = (0.5, 0.5, 0.5),
    ) -> str:
        """Create a UsdPreviewSurface material and return its prim path."""
        if not _HAS_USD_SHADE:
            return ""
        mat_path = f"/Materials/{name}"
        mat = UsdShade.Material.Define(self._stage, mat_path)
        shader = UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f,
        ).Set(Gf.Vec3f(*diffuse_color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
        shader.CreateInput(
            "specularColor", Sdf.ValueTypeNames.Color3f,
        ).Set(Gf.Vec3f(*specular_color))
        mat.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface",
        )
        return mat_path

    def bind_material(self, prim_path: str, material_path: str) -> None:
        if not _HAS_USD_SHADE:
            return
        prim = self._stage.GetPrimAtPath(prim_path)
        mat = UsdShade.Material(self._stage.GetPrimAtPath(material_path))
        if prim.IsValid() and mat.GetPrim().IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    # -- References / payloads --------------------------------------------

    def add_reference_geometry(
        self,
        usd_path: str,
        prim_path: str = "/World/Anatomy",
        as_reference: bool = False,
    ) -> None:
        """Load a USD file as a payload (deferred) or reference (immediate)."""
        prim = self._stage.DefinePrim(prim_path)
        if as_reference:
            prim.GetReferences().AddReference(usd_path)
        else:
            prim.GetPayloads().AddPayload(usd_path)

    # -- Dynamic prim registration + update loop ---------------------------

    def register_prim(
        self,
        *,
        node_name: str,
        prim_path: str,
        updater: PrimUpdater,
    ) -> None:
        """Register a dynamic prim with a per-timestep update callback.

        Parameters
        ----------
        node_name : str
            The name of the :class:`SimulationNode` whose state
            drives this prim.  Used to look up ``state[node_name]``
            in :meth:`update`.
        prim_path : str
            USD prim path.  Must already exist on the stage —
            callers are responsible for defining the prim (whether
            as :class:`UsdGeom.Sphere`, ``Cylinder``, ``Mesh``, etc.)
            before registering it.
        updater : PrimUpdater
            Callback invoked each timestep with the registered prim's
            node state.

        Notes
        -----
        The callback is invoked inside a
        :class:`Sdf.ChangeBlock`, so no need to open another one.
        """
        self._dynamic_prims.append(_RegisteredPrim(
            node_name=node_name, prim_path=prim_path, updater=updater,
        ))

    def update(
        self,
        state: dict[str, dict[str, Any]],
        time_code: Optional[Any] = None,
    ) -> None:
        """Write current dynamic state to the USD stage.

        Each registered :class:`PrimUpdater` is invoked with its
        prim's ``node_state``.  All writes happen inside a single
        :class:`Sdf.ChangeBlock` so the stage emits one notification
        for the entire frame.

        Parameters
        ----------
        state : dict
            Full graph state: ``{node_name: {field_name: value, ...},
            ...}``.  Values are JAX or numpy arrays.
        time_code : Usd.TimeCode, optional
            If given, registered updaters write time-sampled values
            at this code; otherwise they write at the default time.
        """
        with Sdf.ChangeBlock():
            for reg in self._dynamic_prims:
                node_state = state.get(reg.node_name)
                if node_state is None:
                    continue
                reg.updater(
                    self._stage, reg.prim_path, node_state, time_code,
                )

    # -- Export ------------------------------------------------------------

    def export(self, path: str) -> None:
        """Export the current stage to a USD file on disk."""
        self._stage.GetRootLayer().Export(path)


# ---------------------------------------------------------------------------
# Off-the-shelf updaters for the common cases.
# ---------------------------------------------------------------------------


def make_translate_updater(
    field: str = "position",
) -> PrimUpdater:
    """Build an updater that translates a prim using ``state[field]``.

    Suitable for any prim that needs only position tracking (the
    bouncing-ball demo, a 1-DOF mass on a spring, etc.).  The
    prim must have a single ``AddTranslateOp`` xform op already.
    """
    def updater(stage, prim_path, node_state, time_code=None):
        pos = node_state.get(field)
        if pos is None:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if not ops:
            return
        val = vec3_to_gf(np.asarray(pos))
        if time_code is not None:
            ops[0].Set(val, time_code)
        else:
            ops[0].Set(val)
    return updater


def make_translate_orient_updater(
    position_field: str = "position",
    orientation_field: str = "orientation",
) -> PrimUpdater:
    """Build an updater that translates + orients a prim.

    Suitable for rigid-body nodes whose state carries
    ``position: (3,)`` and ``orientation: (4,)`` (``[w, x, y, z]``
    quaternion).  Prim must have ``AddTranslateOp`` + ``AddOrientOp``
    in that order.
    """
    def updater(stage, prim_path, node_state, time_code=None):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if len(ops) < 2:
            return
        pos = node_state.get(position_field)
        if pos is not None:
            val = vec3_to_gf(np.asarray(pos))
            if time_code is not None:
                ops[0].Set(val, time_code)
            else:
                ops[0].Set(val)
        orient = node_state.get(orientation_field)
        if orient is not None:
            val = quat_wxyz_to_gf(np.asarray(orient))
            if time_code is not None:
                ops[1].Set(val, time_code)
            else:
                ops[1].Set(val)
    return updater


__all__ = [
    "LiveStage",
    "PrimUpdater",
    "make_translate_updater",
    "make_translate_orient_updater",
    "quat_wxyz_to_gf",
    "vec3_to_gf",
    "quat_from_z_to_direction",
]
