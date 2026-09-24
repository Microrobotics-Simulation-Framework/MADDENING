"""Sharded wrappers for :class:`SimulationNode`.

Two flavours:

- :class:`ShardedPointwiseNode` wraps a pointwise node and shards its
  state along a single device-mesh axis.  No halo exchange.
- :class:`ShardedStencilNode` wraps a stencil node (non-empty
  ``halo_width``), pads each state field with halo cells from
  neighbouring shards via :func:`halo_exchange`, calls
  :meth:`SimulationNode.update_padded`, and strips halos from the
  result.

The legacy ``ShardedNode`` alias was removed in v0.3.0 (per the v0.2.x
deprecation cycle).  Use :class:`ShardedPointwiseNode` for pointwise
sharding or :class:`ShardedStencilNode` for stencil sharding.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.halo import (
    _BOUNDARY_MODES,
    _global_edge_halos,
    halo_exchange,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import (
    BoundaryFluxSpec,  # noqa: F401 - named by boundary_flux_spec's annotation
    SimulationNode,
    _method_accepts_params,
    _signature_takes_keyword,
)
from maddening.core.static_data import StaticArray, coerce_static_data_value

#: Mesh axis :class:`ShardedPointwiseNode` shards over (the 1-D default
#: name of :func:`~maddening.cloud.multigpu.device_mesh.create_device_mesh`).
_MESH_AXIS = "devices"


def _check_shard_divisible(
    *,
    wrapper: str,
    owner: str,
    what: str,
    spatial_axis: int,
    extent: int,
    mesh_axis: str,
    n_devices: int,
) -> None:
    """Refuse a shard that JAX would split unevenly, in the caller's terms.

    ``jax.device_put`` raises :class:`jax.errors.IndivisibleError` from
    inside the sharding machinery, naming neither the node, the field,
    the cell count nor the device count.  Both Cartesian wrappers
    validate up front instead, so the failure arrives at construction
    with the two numbers that have to agree and the ways to make them.

    Parameters
    ----------
    wrapper, owner, what : str
        Wrapper class name, node name, and what is being sharded (e.g.
        ``"state field 'f'"``), for the message.
    spatial_axis, extent : int
        The array axis being sharded and how many cells it has.
    mesh_axis : str
        Name of the mesh axis it is sharded over.
    n_devices : int
        How many devices that mesh axis has.

    Raises
    ------
    ValueError
        When *extent* is not a multiple of *n_devices*.
    """
    if n_devices <= 0 or extent % n_devices == 0:
        return
    nearest = ((extent // n_devices) + 1) * n_devices
    divisors = sorted(d for d in range(1, n_devices + 1) if extent % d == 0)
    raise ValueError(
        f"{wrapper} cannot shard {what} of node {owner!r}: spatial axis "
        f"{spatial_axis} has {extent} cells and mesh axis {mesh_axis!r} has "
        f"{n_devices} devices, which does not divide it "
        f"({extent} % {n_devices} == {extent % n_devices}).  A sharded axis "
        f"is split evenly across the devices, so the cell count must be a "
        f"multiple of the device count: resize that axis to a multiple of "
        f"{n_devices} (the next one up is {nearest}), run on one of the "
        f"device counts that do divide {extent} ({divisors}), or use "
        "ShardedUnstructuredNode, which carries an explicit padded layout "
        "and accepts any (device, cell) pair."
    )


def _declared_halo_boundary(node: SimulationNode) -> Optional[str]:
    """The halo fill ``node`` declares through an optional ``halo_boundary()``.

    A stencil node whose ``update_padded`` reproduces its own ``update``
    only under one fill of the halos at the edges of the global grid says
    which by defining ``halo_boundary()`` (e.g.
    :meth:`~maddening.nodes.lbm.LBMNode.halo_boundary`, ``"periodic"``,
    because its unsharded streaming wraps).  Duck-typed, like
    ``domain_integral_axes``: a node without the method declares nothing
    and returns ``None``.

    Raises
    ------
    ValueError
        When the declared value is not one of the halo-exchange modes.
    """
    hook = getattr(node, "halo_boundary", None)
    if hook is None:
        return None
    declared = hook() if callable(hook) else hook
    if declared is None:
        return None
    if declared not in _BOUNDARY_MODES:
        raise ValueError(
            f"{type(node).__name__} {node.name!r}.halo_boundary() returned "
            f"{declared!r}; it must be one of {_BOUNDARY_MODES} or None."
        )
    return str(declared)


def _accepts_params(node: SimulationNode) -> bool:
    """True when ``node.update(..., params=x)`` would deliver ``x``.

    :func:`~maddening.core.node._method_accepts_params`, the one params
    rule: the node's own :meth:`SimulationNode.accepts_params` when it has
    one, the signature (explicit keyword or ``**kwargs``) otherwise, so
    the wrapper answers exactly what the graph would have answered for
    the unwrapped node.  The duck-typed fallback here used to accept only
    the explicit keyword.
    """
    return _method_accepts_params(node, "update")


class _ForwardsCouplingHooks:
    """The flux and interface-correction hooks, forwarded to ``self._inner``.

    Both wrappers that use this keep the graph-level state in the inner
    node's own global view -- each field is the inner node's array, placed
    with a ``NamedSharding`` -- so the inner node's
    ``compute_boundary_fluxes`` and ``compute_interface_correction`` read
    it exactly as they would unwrapped, and an index from
    ``interface_dof_indices`` names the same cell.  Without the
    forwarding a wrapped node published no fluxes (a flux edge from it
    failed to compile) and no interface DOFs (a coupled interface was
    silently left uncorrected).  ``params`` reaches the inner hook under
    the one params rule, as in
    :class:`~maddening.core.simulation.hybrid_node.HybridNode`.

    :class:`~maddening.cloud.multigpu.sharded_unstructured.ShardedUnstructuredNode`
    does not use it: its state is in partition layout, where the inner
    node's global indices name different cells.
    """

    _inner: SimulationNode

    def boundary_flux_spec(self) -> dict[str, "BoundaryFluxSpec"]:
        return self._inner.boundary_flux_spec()

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        if params is not None and _method_accepts_params(
                self._inner, "compute_boundary_fluxes"):
            return self._inner.compute_boundary_fluxes(
                state, boundary_inputs, dt, params=params)
        return self._inner.compute_boundary_fluxes(state, boundary_inputs, dt)

    def interface_dof_indices(self) -> dict[str, tuple[str, int]]:
        return self._inner.interface_dof_indices()

    def update_evaluations(self) -> Optional[float]:
        """The wrapped node's declaration: sharding does not change how often it rounds."""
        return self._inner.update_evaluations()

    def compute_interface_correction(
        self,
        pre_state: dict,
        boundary_inputs: dict,
        dt: float,
        *,
        params=None,
    ) -> dict[str, list[tuple[int, Any]]]:
        if params is not None and _method_accepts_params(
                self._inner, "compute_interface_correction"):
            return self._inner.compute_interface_correction(
                pre_state, boundary_inputs, dt, params=params)
        return self._inner.compute_interface_correction(
            pre_state, boundary_inputs, dt)


@stability(StabilityLevel.STABLE)
class ShardedPointwiseNode(_ForwardsCouplingHooks, SimulationNode):
    """Data-parallel wrapper for a pointwise :class:`SimulationNode`.

    Only nodes with empty ``halo_width()`` can be wrapped; stencil nodes
    must use :class:`ShardedStencilNode`.

    Parameters
    ----------
    node : SimulationNode
        The pointwise node to wrap.
    mesh : Mesh
        JAX device mesh (1-D, axis name ``"devices"``).
    shard_axes : int or tuple[int, ...]
        Which axis of the state arrays to shard, as a 1-tuple (an ``int``
        is accepted and wrapped).  Any single axis is honoured: the
        partition spec puts the mesh axis at that position and replicates
        the rest.  A state field with too few dimensions to have that
        axis is replicated.  Multi-axis sharding raises
        :class:`NotImplementedError`.

    Raises
    ------
    ValueError
        If *node* is a stencil node, if the mesh has no ``"devices"``
        axis, if *shard_axes* is not a non-negative axis index, or if the
        sharded axis of a state field is not divisible by the device
        count.

    Notes
    -----
    ``initial_state`` is where the placement happens: it places every
    state field onto the mesh with ``jax.device_put``.  ``update``
    deliberately does neither a ``device_put`` nor a ``shard_map`` --
    the operation is pointwise, so
    XLA's SPMD propagation keeps a sharded input sharded through it, and
    forcing a placement would insert a resharding collective on every
    step.  The consequence is that the wrapper follows the sharding of
    the state it is *given*: feed it an unsharded array (a state that
    round-tripped through a checkpoint, a host array from a REST write)
    and the step runs on one device, silently and correctly.  Start from
    ``initial_state`` -- ``GraphManager`` does -- or ``device_put`` the
    state yourself with :attr:`sharding`.
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        shard_axes: int | tuple[int, ...] = (0,),
    ) -> None:
        if node.halo_width():
            raise ValueError(
                f"{type(node).__name__} declares halo_width="
                f"{node.halo_width()} -- use ShardedStencilNode for "
                "stencil sharding.  ShardedPointwiseNode only wraps "
                "pointwise nodes (halo_width() == {})."
            )

        if isinstance(shard_axes, int):
            shard_axes = (shard_axes,)
        shard_axes = tuple(shard_axes)

        if len(shard_axes) > 1:
            raise NotImplementedError(
                f"Multi-axis pointwise sharding (axes={shard_axes}) is "
                "not yet implemented. Use a single shard axis."
            )
        if len(shard_axes) != 1 or int(shard_axes[0]) != shard_axes[0] \
                or shard_axes[0] < 0:
            raise ValueError(
                f"shard_axes={shard_axes!r} must be a single non-negative "
                "axis index (an int or a 1-tuple)."
            )
        shard_axis = int(shard_axes[0])
        if _MESH_AXIS not in mesh.axis_names:
            raise ValueError(
                f"ShardedPointwiseNode shards over a mesh axis named "
                f"{_MESH_AXIS!r}, which mesh.axis_names={mesh.axis_names} "
                "does not have.  Build the mesh with "
                "create_device_mesh(shape=(n,)) (its 1-D default axis name), "
                "or use ShardedStencilNode, which takes an axis_map."
            )

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        self._inner = node
        # One node, one params dict: the wrapper and the node it wraps
        # share it, so a write through any surface (REST, sysid, a
        # recompile) reaches the ``self.params`` the inner ``update``
        # actually reads.  A copy silently strands the write on the
        # wrapper.
        self.params = node.params
        self._mesh = mesh
        self._shard_axes = shard_axes
        self._shard_axis = shard_axis
        # The requested axis, not always axis 0: ``P("devices")`` names
        # the *first* array axis, so a caller asking for axis 1 used to
        # get axis 0 sharded and an IndivisibleError blaming an axis they
        # had not chosen.  Leading axes are replicated (``None``).
        self._sharding = NamedSharding(
            mesh, P(*([None] * shard_axis + [_MESH_AXIS])),
        )
        self._n_devices = int(mesh.shape[_MESH_AXIS])
        self._validate_state_divisible(node)
        # Graph parameter contract: the wrapper is a params node exactly
        # when the node it wraps is one.
        self._inner_accepts_params = _accepts_params(node)

    def _validate_state_divisible(self, node: SimulationNode) -> None:
        """Refuse at construction what ``device_put`` would refuse later.

        A node that cannot build its initial state yet (one waiting for
        a ``static_data_provider``, say) is left alone: the same check
        runs again in :meth:`initial_state`, where the arrays are real.
        """
        try:
            state = node.initial_state()
        except Exception:      # noqa: BLE001 - construction must not depend on it
            return
        self._check_state_divisible(state)

    def _check_state_divisible(self, state: dict) -> None:
        for field, arr in state.items():
            if jnp.ndim(arr) <= self._shard_axis:
                continue       # too few axes to shard: replicated
            _check_shard_divisible(
                wrapper=type(self).__name__, owner=self._inner.name,
                what=f"state field {field!r}", spatial_axis=self._shard_axis,
                extent=int(jnp.shape(arr)[self._shard_axis]),
                mesh_axis=_MESH_AXIS, n_devices=self._n_devices,
            )

    @property
    def sharding(self) -> NamedSharding:
        """The placement ``initial_state`` gives every sharded field."""
        return self._sharding

    def halo_width(self) -> dict[int, int]:
        """ShardedPointwiseNode only wraps pointwise nodes (no halo)."""
        return {}

    def initial_state(self) -> dict:
        state = self._inner.initial_state()
        self._check_state_divisible(state)
        sharded = {}
        for field, arr in state.items():
            if arr.ndim > self._shard_axis:
                sharded[field] = jax.device_put(arr, self._sharding)
            else:
                sharded[field] = arr
        return sharded

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Delegate to the wrapped node, forwarding injected ``params``.

        ``params`` (the node's entry of ``GraphManager.params``) is
        handed to an inner ``update(..., params=)``; a node on the
        3-argument contract is called without it.
        """
        if self._inner_accepts_params and params is not None:
            # `_inner_accepts_params` was computed from the inner node's
            # signature; `SimulationNode.update` does not declare it.
            return self._inner.update(
                state, boundary_inputs, dt,
                params=params)  # pyright: ignore[reportCallIssue]
        return self._inner.update(state, boundary_inputs, dt)

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    # -- graph parameter contract (proxied to the inner node) -----------

    def accepts_params(self, *, method: str = "update") -> bool:
        if method != "update":
            return super().accepts_params(method=method)
        return self._inner_accepts_params

    def params_pytree(self) -> dict:
        return self._inner.params_pytree() if self._inner_accepts_params else {}

    def param_specs(self) -> dict:
        return self._inner.param_specs() if self._inner_accepts_params else {}

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, "to_dict") else {}
        d["sharded"] = True
        d["shard_axes"] = self._shard_axes
        return d


def _params_signature(params) -> tuple:
    """Cache-key component for a params pytree (structure + shapes)."""
    if not params:
        return ()
    return tuple(
        (jax.tree_util.keystr(path), tuple(jnp.shape(leaf)), str(jnp.asarray(leaf).dtype))
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    )


@stability(StabilityLevel.STABLE)
class ShardedStencilNode(_ForwardsCouplingHooks, SimulationNode):
    """Pencil-decomposition wrapper for a stencil :class:`SimulationNode`.

    On each step, every state field listed in the node's
    ``halo_width()`` is halo-exchanged along the relevant mesh axis,
    :meth:`SimulationNode.update_padded` is called on the padded state,
    and halos are stripped before the result is returned.

    Parameters
    ----------
    node : SimulationNode
        The stencil node to wrap.  Must override ``update_padded`` and
        declare a non-empty ``halo_width()``.
    mesh : Mesh
        JAX device mesh whose axes match ``axis_map``.
    axis_map : dict[str, int]
        Maps each mesh axis name to the spatial axis index of the
        node's state arrays it shards.  Example for a 3-D LBM under a
        2-D pencil mesh::

            {"spatial_y": 1, "spatial_z": 2}

        Spatial axes not appearing as values of ``axis_map`` are
        replicated on every device; their halos do not need exchange.
    boundary : str
        How the halos at the edges of the *global* grid are filled
        (``"periodic"``, ``"edge"`` or ``"zero"``, exactly as
        :func:`~maddening.cloud.multigpu.halo.halo_exchange` documents);
        interior halos always come from the neighbouring shard.  The same
        fill is used on a mesh axis of one device, and on a halo axis
        ``axis_map`` leaves unsharded.  Default ``"edge"`` -- each edge
        cell repeated across its halo -- for a node that applies its
        physical boundary conditions in ``update_padded`` after the
        exchange.  A node that declares ``halo_boundary()`` must be given
        exactly that mode (see the note below).  One string for every
        axis; anything else is refused at construction.

    Raises
    ------
    ValueError
        If *node* is pointwise, if ``axis_map`` names a mesh axis the
        mesh does not have or a spatial axis with no declared halo, if a
        sharded extent is not divisible by the devices on its mesh axis
        (see the note below), if *boundary* is not one of the three modes
        (a per-axis dict included), or if *boundary* -- the default
        ``"edge"`` included -- differs from the mode the node declares.

    Notes
    -----
    **A node may declare its halo boundary.**  Some stencil nodes impose
    no condition at the edges of their grid in ``update_padded``: what
    arrives from the halo *is* the boundary condition.
    :class:`~maddening.nodes.lbm.LBMNode` is one -- its unsharded
    ``update`` streams periodically (``jnp.roll``), so its sharded step is
    the same model only with periodic halos.  Such a node defines
    ``halo_boundary()``, and a *boundary* that differs from it -- the
    default ``"edge"`` included -- is refused at construction, because it
    would make the sharded node compute something the unsharded node does
    not.  Wrap an ``LBMNode`` with ``boundary="periodic"``.  (Before 0.4.0
    an ``LBMNode`` wrapped with the default ``"edge"`` ran, and a walled
    channel's centreline velocity moved by 0.64% against the unsharded
    node.)  :class:`~maddening.nodes.heat.HeatNode` declares ``"edge"``
    for the opposite reason: it closes its rod ends itself, from its
    ``left_temperature`` / ``right_temperature`` inputs, so any other fill
    would be ignored, and the refusal says to pass those inputs instead.

    **Each sharded extent must divide by the devices on its mesh axis.**
    A pencil decomposition gives every device the same slab, so a 17-cell
    axis over 3 devices has no layout; construction refuses it, naming
    the cell count and the device count.  Pad the grid to a multiple,
    choose a device count that divides it, or use
    :class:`~maddening.cloud.multigpu.sharded_unstructured.ShardedUnstructuredNode`,
    which carries an explicit padded layout and takes any (device, cell)
    pair.  This is a property of the stencil path only.
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        axis_map: dict[str, int],
        boundary: str = "edge",
    ) -> None:
        halo = node.halo_width()
        if not halo:
            raise ValueError(
                f"{type(node).__name__} has empty halo_width() -- use "
                "ShardedPointwiseNode for pointwise sharding."
            )

        # One mode for every axis, and a known one.  Until 0.4.0 nothing
        # checked: an unknown string failed only when the first step was
        # traced (inside halo_exchange), and on a halo axis axis_map
        # leaves unsharded anything but "periodic"/"edge" -- a typo, or a
        # per-axis dict -- was silently filled with zeros.
        if not isinstance(boundary, str):
            raise ValueError(
                f"ShardedStencilNode: boundary must be one mode for every "
                f"axis, one of {_BOUNDARY_MODES}; got "
                f"{type(boundary).__name__} {boundary!r}.  Per-mesh-axis "
                "modes are a halo_exchange feature: the wrapper also fills "
                "halo axes axis_map leaves unsharded, which have no mesh "
                "axis to key a mode by."
            )
        if boundary not in _BOUNDARY_MODES:
            raise ValueError(
                f"ShardedStencilNode: unknown boundary mode {boundary!r}; "
                f"expected one of {_BOUNDARY_MODES}."
            )

        # The halo fill at the global edges.  A node that declares one
        # (``halo_boundary()``) must be given exactly that one -- the
        # default ``"edge"`` included.  For LBMNode the halo is the
        # boundary condition, so another fill is another model; for
        # HeatNode, which closes its rod ends itself, another fill would be
        # ignored without a word.  A node may add what to do instead
        # (``halo_boundary_hint()``).  A node that declares nothing is
        # wrapped exactly as before.
        declared = _declared_halo_boundary(node)
        if declared is not None and boundary != declared:
            hint = getattr(node, "halo_boundary_hint", None)
            hint = hint() if callable(hint) else hint
            raise ValueError(
                f"ShardedStencilNode: {type(node).__name__} {node.name!r} declares "
                f"halo_boundary() == {declared!r}, the one fill of the halos at "
                "the edges of the global grid it takes, but was given "
                f"boundary={boundary!r}"
                f"{' (the default)' if boundary == 'edge' else ''}.  With another "
                "fill the sharded node would silently compute a different model "
                "from the unsharded one, or silently ignore the fill.  Pass "
                f"boundary={declared!r}." + (f"  {hint}" if hint else "")
            )

        # Validate axis_map keys against the mesh and warn on covered axes
        for mesh_axis in axis_map:
            if mesh_axis not in mesh.axis_names:
                raise ValueError(
                    f"axis_map references mesh axis {mesh_axis!r} not "
                    f"present in mesh.axis_names={mesh.axis_names}"
                )

        # Every spatial axis the node sharding covers must have a declared
        # halo width (else it would not be a stencil axis).
        for spatial_axis in axis_map.values():
            if spatial_axis not in halo:
                raise ValueError(
                    f"axis_map shards spatial axis {spatial_axis} but "
                    f"node {type(node).__name__} reports halo_width="
                    f"{halo} with no entry for that axis."
                )

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        self._inner = node
        # Share the inner node's params dict rather than copying it, so a
        # write through any surface reaches the code that reads it.
        self.params = node.params
        self._mesh = mesh
        self._axis_map = dict(axis_map)
        self._boundary = boundary

        # axes for halo_exchange: list of (mesh_axis, spatial_axis, halo)
        self._exchange_axes: list[tuple[str, int, int]] = [
            (ma, sa, halo[sa]) for ma, sa in self._axis_map.items()
        ]

        sharded_spatial = set(self._axis_map.values())
        self._replicated_halo_axes = {
            sa: h for sa, h in halo.items() if sa not in sharded_spatial
        }

        # Classify the inner node's static_data: pick out every
        # StaticArray declared with ``replication="shard"`` and validate
        # that its shard_axis lines up with one of the spatial axes this
        # wrapper actually shards.
        self._sharded_static: dict[str, StaticArray] = (
            self._classify_sharded_static(node.static_data)
        )
        # Per-device materialisation of those statics, cached across
        # calls.  See ``_materialise_sharded_statics``.
        self._static_device_cache: Optional[tuple] = None

        # A pencil decomposition splits each sharded axis evenly, so the
        # grid extent has to be a multiple of the devices on that axis.
        # Unvalidated, that surfaced as an IndivisibleError from inside
        # ``device_put`` on the first ``initial_state`` or first step,
        # naming neither the node nor either number.
        self._check_divisible_extents()

        # Probe inner.update_padded's signature once.  Nodes ported to
        # v0.2.1 accept `static_padded=` and `shard_info=`; v0.2-era
        # nodes do not.  If the node has sharded statics declared but
        # its signature does not accept `static_padded`, that is a
        # contract violation and we raise here rather than at first
        # trace.
        self._inner_accepts_static_padded = _signature_takes_keyword(
            node.update_padded, "static_padded",
        )
        self._inner_accepts_shard_info = _signature_takes_keyword(
            node.update_padded, "shard_info",
        )
        # Graph parameter contract on the sharded path: an inner
        # ``update_padded(..., params=None)`` receives the node's entry of
        # ``GraphManager.params`` (replicated across shards).
        #
        # The one params rule (explicit keyword or ``**kwargs``), asked
        # through the inner node's own probe.  A ``**kwargs`` node that
        # did not get ``params`` silently fell back to its constructor
        # constant, so the injected leaf never entered the trace and
        # d(loss)/d(param) came back exactly 0.0 with no error anywhere.
        self._inner_accepts_params = _method_accepts_params(node, "update_padded")
        if self._sharded_static and not self._inner_accepts_static_padded:
            raise ValueError(
                f"{type(node).__name__} declares sharded static_data "
                f"({sorted(self._sharded_static)}) but its update_padded "
                "signature does not accept 'static_padded'. Update the "
                "signature to '(self, state_padded, boundary_inputs, dt, "
                "*, static_padded=None, shard_info=None)'."
            )

        # Cache for shard_map-wrapped update functions, keyed by state
        # shape signature.  Built once per shape, then reused so the JAX
        # trace cache hits on every subsequent step.
        self._local_update_fn = self._build_local_update()
        self._sharded_cache: dict[tuple, Any] = {}

    def halo_width(self) -> dict[int, int]:
        """Same as the wrapped node -- sharding does not change the stencil."""
        return self._inner.halo_width()

    def _check_divisible_extents(self, state: Optional[dict] = None) -> None:
        """Refuse a grid this mesh cannot split evenly, naming both numbers.

        Checks every state field and every sharded ``StaticArray`` on
        each axis of ``axis_map``.  A node that cannot build its initial
        state at construction time (one waiting for a
        ``static_data_provider``) is skipped here and checked again in
        :meth:`initial_state`, where the arrays are real.
        """
        if state is None:
            try:
                state = self._inner.initial_state()
            except Exception:  # noqa: BLE001 - construction must not depend on it
                state = {}
        for mesh_axis, spatial_axis in self._axis_map.items():
            n_devices = int(self._mesh.shape[mesh_axis])
            for field, arr in state.items():
                if jnp.ndim(arr) <= spatial_axis:
                    continue
                _check_shard_divisible(
                    wrapper=type(self).__name__, owner=self._inner.name,
                    what=f"state field {field!r}", spatial_axis=spatial_axis,
                    extent=int(jnp.shape(arr)[spatial_axis]),
                    mesh_axis=mesh_axis, n_devices=n_devices,
                )
            for key, static in self._sharded_static.items():
                if static.shard_axis != spatial_axis:
                    continue
                shape = tuple(jnp.shape(static.value))
                if spatial_axis >= len(shape):
                    continue
                _check_shard_divisible(
                    wrapper=type(self).__name__, owner=self._inner.name,
                    what=f"static array {key!r}", spatial_axis=spatial_axis,
                    extent=int(shape[spatial_axis]),
                    mesh_axis=mesh_axis, n_devices=n_devices,
                )

    def initial_state(self) -> dict:
        state = self._inner.initial_state()
        self._check_divisible_extents(state)
        return {
            field: jax.device_put(arr, self._sharding_for_field(arr))
            for field, arr in state.items()
        }

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def update_padded(
        self,
        state_padded,
        boundary_inputs,
        dt,
        *,
        static_padded=None,
        shard_info=None,
    ):
        kwargs = {}
        if self._inner_accepts_static_padded and static_padded is not None:
            kwargs["static_padded"] = static_padded
        if self._inner_accepts_shard_info and shard_info is not None:
            kwargs["shard_info"] = shard_info
        return self._inner.update_padded(
            state_padded, boundary_inputs, dt, **kwargs
        )

    def domain_integral_fields(self) -> set[str]:
        """Proxy to the wrapped node's declaration."""
        return self._inner.domain_integral_fields()

    # -- graph parameter contract (proxied to the inner node) -----------

    def accepts_params(self, *, method: str = "update") -> bool:
        if method != "update":
            return super().accepts_params(method=method)
        return self._inner_accepts_params

    def params_pytree(self) -> dict:
        return self._inner.params_pytree() if self._inner_accepts_params else {}

    def param_specs(self) -> dict:
        return self._inner.param_specs() if self._inner_accepts_params else {}

    def domain_integral_axes(self) -> dict[str, tuple[str, ...]]:
        """Proxy to the wrapped node's declaration."""
        return dict(getattr(self._inner, "domain_integral_axes", dict)())

    def _integral_reduction(self, key: str):
        """``(reduce_axes, unreduced_axes)`` for a domain-integral key."""
        mesh_axes = tuple(self._mesh.axis_names)
        axes = self.domain_integral_axes().get(key)
        if axes is None:
            return mesh_axes, ()
        axes = tuple(axes)
        unknown = [a for a in axes if a not in mesh_axes]
        if unknown:
            raise ValueError(
                f"domain_integral_axes[{key!r}] names mesh axes {unknown} "
                f"not in mesh.axis_names={mesh_axes}"
            )
        return axes, tuple(a for a in mesh_axes if a not in axes)

    # ------------------------------------------------------------------
    # update path
    # ------------------------------------------------------------------

    def _build_local_update(self):
        """Build the stable inner update function (closure on static data).

        The returned function takes ``(local_state, boundary_inputs, dt,
        local_static)`` and is reused across every step -- only its in/out
        specs depend on the input shapes, so the shard_map trace caches
        cleanly.
        """
        inner = self._inner
        mesh = self._mesh
        exchange_axes = self._exchange_axes
        boundary = self._boundary
        replicated_halo_axes = self._replicated_halo_axes
        strip_fn = self._strip_halos
        field_needs_halo = self._field_needs_halo
        sharded_static = self._sharded_static
        axis_map = self._axis_map  # {mesh_axis: spatial_axis}
        halo_widths = inner.halo_width()
        state_set = set(inner.state_fields())
        integrals = set(inner.domain_integral_fields())
        integral_reduction = {k: self._integral_reduction(k) for k in integrals}
        accepts_static_padded = self._inner_accepts_static_padded
        accepts_shard_info = self._inner_accepts_shard_info
        accepts_params = self._inner_accepts_params
        mesh_axis_tup = tuple(mesh.axis_names)

        # Pre-compute per-static halo-exchange descriptors. A sharded
        # static gets halo-exchanged only on the single mesh axis that
        # maps to its shard_axis, with ``boundary="edge"`` (statics
        # don't evolve in time -- periodic wrap would be wrong even when
        # the state uses periodic).
        static_exchange: dict[str, list[tuple[str, int, int]]] = {}
        for k, sa in sharded_static.items():
            descriptors: list[tuple[str, int, int]] = []
            if sa.shard_axis in halo_widths:
                h = halo_widths[sa.shard_axis]
                for ma, mapped_sax in axis_map.items():
                    if mapped_sax == sa.shard_axis:
                        descriptors.append((ma, sa.shard_axis, h))
                        break
            static_exchange[k] = descriptors

        def _pad_replicated(arr):
            out = arr
            for sa in sorted(replicated_halo_axes):
                h = int(replicated_halo_axes[sa])
                if h == 0 or sa >= out.ndim:
                    continue
                n = out.shape[sa]
                left = jax.lax.slice_in_dim(out, 0, h, axis=sa)
                right = jax.lax.slice_in_dim(out, n - h, n, axis=sa)
                # An unsharded axis is the whole axis, so both its halos
                # are global halos -- filled exactly as halo_exchange
                # fills them on a sharded one (the same helper).
                left_halo, right_halo = _global_edge_halos(
                    left, right, spatial_axis=sa, halo=h, boundary=boundary,
                )
                out = jnp.concatenate([left_halo, out, right_halo], axis=sa)
            return out

        def _pad_like_state(arr):
            arr2 = _pad_replicated(arr)
            if exchange_axes and field_needs_halo(arr):
                arr2 = halo_exchange(
                    arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,
                )
            return arr2

        def _local_update(local_state, local_bi, local_dt, local_static,
                          local_params, *, grid_bi=frozenset()):
            # 1. Halo-pad state.
            padded = {f: _pad_like_state(arr) for f, arr in local_state.items()}

            # 1b. Grid-shaped boundary inputs (a per-cell body force, a
            #     wall-mask update) arrive as this shard's slice and are
            #     padded exactly like a state field, so the inner sees
            #     them at the padded local shape it expects.  Everything
            #     else (scalars, a uniform (D,) vector) is replicated.
            local_bi = {
                k: (_pad_like_state(v) if k in grid_bi else v)
                for k, v in local_bi.items()
            }

            # 2. Halo-pad sharded statics (boundary="edge").
            padded_static: dict[str, Any] = {}
            for k, arr in local_static.items():
                descriptors = static_exchange[k]
                if descriptors:
                    padded_static[k] = halo_exchange(
                        arr, mesh=mesh, axes=descriptors, boundary="edge",
                    )
                else:
                    padded_static[k] = arr

            # 3. Compute shard_info: {spatial_axis: (global_offset,
            #    local_extent)} for every spatial axis the node shards.
            #    ``global_offset`` is a traced JAX scalar — usable in
            #    dynamic_slice, not in Python integer slicing.
            shard_info: dict[int, tuple[Any, int]] = {}
            for ma, sax in axis_map.items():
                extent = None
                for arr in local_state.values():
                    if sax < arr.ndim:
                        extent = arr.shape[sax]
                        break
                if extent is None:
                    for arr in local_static.values():
                        if sax < arr.ndim:
                            extent = arr.shape[sax]
                            break
                if extent is None:
                    continue
                offset = lax.axis_index(ma) * extent
                shard_info[sax] = (offset, extent)

            # 4. Dispatch.
            extra_kwargs = {}
            if accepts_static_padded and padded_static:
                extra_kwargs["static_padded"] = padded_static
            if accepts_shard_info and shard_info:
                extra_kwargs["shard_info"] = shard_info
            if accepts_params and local_params:
                extra_kwargs["params"] = local_params
            new_padded = inner.update_padded(
                padded, local_bi, local_dt, **extra_kwargs
            )

            # 5. Classify outputs: state fields → strip halos; declared
            #    integrals → psum across the full mesh; otherwise raise
            #    (the out_specs build below would also catch it).
            out: dict[str, Any] = {}
            for k, v in new_padded.items():
                if k in state_set:
                    out[k] = strip_fn(v, original=local_state[k])
                elif k in integrals:
                    reduce_axes, unreduced = integral_reduction[k]
                    red = lax.psum(v, axis_name=reduce_axes) if reduce_axes else v
                    # One leading axis per unreduced mesh axis: the local
                    # value is that shard's slice of the stacked result.
                    out[k] = red[(None,) * len(unreduced)] if unreduced else red
                else:
                    raise ValueError(
                        f"{type(inner).__name__}.update_padded returned "
                        f"key {k!r} that is neither in state_fields() "
                        "nor in domain_integral_fields()."
                    )
            return out

        return _local_update

    def _state_signature(self, state: dict) -> tuple:
        return tuple(sorted(
            (f, tuple(arr.shape), str(arr.dtype)) for f, arr in state.items()
        ))

    def _bi_signature(self, boundary_inputs: dict) -> tuple:
        return tuple(sorted(
            (k, tuple(jnp.asarray(v).shape), str(jnp.asarray(v).dtype))
            for k, v in boundary_inputs.items()
        ))

    def _static_signature(self, static: dict) -> tuple:
        return tuple(sorted(
            (k, tuple(a.shape), str(a.dtype)) for k, a in static.items()
        ))

    def _get_sharded_fn(
        self, state: dict, boundary_inputs: dict, static: dict, params=None,
    ):
        key = (
            self._state_signature(state),
            self._bi_signature(boundary_inputs),
            self._static_signature(static),
            self._inner.static_data_hash(),
            _params_signature(params),
        )
        fn = self._sharded_cache.get(key)
        if fn is not None:
            return fn

        state_specs = {f: self._spec_for_field(arr) for f, arr in state.items()}
        grid_bi = self._grid_shaped_boundary_inputs(state, boundary_inputs)
        bi_specs = {
            k: (self._spec_for_field(jnp.asarray(v)) if k in grid_bi else P())
            for k, v in boundary_inputs.items()
        }
        static_specs = {
            k: self._spec_for_static_key(k, arr)
            for k, arr in static.items()
        }
        # Outputs: state fields keep their per-shard specs (halos are
        # stripped to original shape); declared domain integrals are
        # replicated after lax.psum.  Anything else would have raised
        # inside _local_update; we leave the out_specs key absent here
        # so shard_map's pytree consistency check catches it too.
        out_specs = dict(state_specs)
        for k in self._inner.domain_integral_fields():
            _, unreduced = self._integral_reduction(k)
            out_specs[k] = P(*unreduced) if unreduced else P()

        params_specs = jax.tree.map(lambda _: P(), params if params else {})
        sm = shard_map(
            functools.partial(self._local_update_fn, grid_bi=grid_bi),
            mesh=self._mesh,
            in_specs=(state_specs, bi_specs, P(), static_specs, params_specs),
            out_specs=out_specs,
        )
        # Bare shard_map outside jit incurs ~250ms/call of Python dispatch
        # overhead on CPU; wrapping it in jit reduces that to microseconds.
        # When ShardedStencilNode is used inside GraphManager's jitted step
        # function the outer jit would absorb this anyway, but the eager
        # path (standalone .update calls in tests) needs the explicit jit.
        fn = jax.jit(sm)
        self._sharded_cache[key] = fn
        return fn

    def _classify_sharded_static(self, static_data) -> dict[str, StaticArray]:
        """Pick the ``replication="shard"`` entries out of ``static_data``.

        Validates that every such array's ``shard_axis`` is one of the
        spatial axes this wrapper actually shards.  Called from
        ``__init__`` and again whenever the inner node's static data
        changes identity (see :meth:`_materialise_sharded_statics`).
        """
        sharded_spatial = set(self._axis_map.values())
        out: dict[str, StaticArray] = {}
        for k, v in static_data.items():
            v_coerced = coerce_static_data_value(
                v, node_name=self._inner.name, key=k,
            )
            if isinstance(v_coerced, StaticArray) and v_coerced.replication == "shard":
                if v_coerced.shard_axis not in sharded_spatial:
                    raise ValueError(
                        f"StaticArray {k!r} declares shard_axis="
                        f"{v_coerced.shard_axis} but "
                        f"{type(self._inner).__name__} "
                        f"shards spatial axes {sorted(sharded_spatial)} via "
                        "ShardedStencilNode (per axis_map.values())."
                    )
                out[k] = v_coerced
        return out

    @staticmethod
    def _static_cache_key(sharded: dict[str, StaticArray]) -> tuple:
        """Identity key for a classified sharded-static dict.

        ``id(sa.value)`` is the change signal: a node that rebuilds its
        static arrays (a checkpoint restore through a
        ``static_data_provider``, a ``replace_node`` that brings a new
        mesh) hands back a *different* array object and the cache
        misses.  Shape, dtype and ``shard_axis`` ride along so a
        same-object-different-view case cannot slip through.  The
        materialised dict keeps a reference to those very objects, so
        an ``id`` can never be recycled while it is a live cache key.

        What this deliberately does *not* see is a mutation of an array
        in place.  ``SimulationNode.static_data`` requires the values to
        be stable for a node instance, and ``static_data_hash`` (the
        framework's existing invalidation signal) does not hash contents
        either; a node that really does rewrite a static array in place
        must call :meth:`invalidate_static_cache`.
        """
        return tuple(
            (k, id(sharded[k].value), sharded[k].shape,
             str(sharded[k].dtype), sharded[k].shard_axis)
            for k in sorted(sharded)
        )

    @stability(StabilityLevel.STABLE)
    def invalidate_static_cache(self) -> None:
        """Drop the cached per-device copy of the sharded static arrays.

        Call this after rewriting a sharded ``StaticArray``'s buffer in
        place; replacing the array object is detected automatically.
        The ``super()`` call forwards to the wrapped node, so a cache
        further in is dropped too.
        """
        self._static_device_cache = None
        super().invalidate_static_cache()

    def _materialise_sharded_statics(self) -> dict:
        """Per-device materialisation of every sharded StaticArray.

        Each StaticArray with ``replication="shard"`` is placed onto
        the device mesh via ``jax.device_put`` + ``NamedSharding`` whose
        PartitionSpec puts the matching mesh-axis at the array's
        ``shard_axis``.  This is the "3a materialisation" step from the
        v0.2.1 plan -- v0.2.0 only stored ``shard_axis`` as metadata.

        The result is cached: the statics do not change from step to
        step, so re-partitioning and re-copying them on every public
        ``update`` was pure host overhead on the interactive path (one
        ``device_put`` per sharded array per frame).  The cache is keyed
        on :meth:`_static_cache_key`, so a node that hands back a
        different array -- or a different set of sharded keys -- is
        picked up on the next call.
        """
        sharded = self._classify_sharded_static(self._inner.static_data)
        key = self._static_cache_key(sharded)
        cached = self._static_device_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        # The inner node's sharded statics changed.  If their *structure*
        # changed (a key appeared/disappeared, or moved axis) the closure
        # built by ``_build_local_update`` and every shard_map compiled
        # against it are stale too.
        struct = tuple((k, sharded[k].shard_axis) for k in sorted(sharded))
        prev_struct = tuple(
            (k, self._sharded_static[k].shard_axis)
            for k in sorted(self._sharded_static)
        )
        self._sharded_static = sharded
        if struct != prev_struct:
            self._local_update_fn = self._build_local_update()
            self._sharded_cache.clear()

        # ``ensure_compile_time_eval`` keeps the placement concrete even
        # when this runs inside a trace (GraphManager compiles the step
        # with the node's ``update`` in it).  Without it the cached value
        # would be a tracer that escapes its trace -- and the statics are
        # compile-time constants anyway, so evaluating them eagerly is
        # exactly right.
        out: dict[str, jax.Array] = {}
        with jax.ensure_compile_time_eval():
            for k, sa in sharded.items():
                arr = jnp.asarray(sa.value)
                spec = self._spec_for_static_key(k, arr)
                sharding = NamedSharding(self._mesh, spec)
                out[k] = jax.device_put(arr, sharding)
        # ``sharded`` is retained so the ``id()``s in ``key`` stay pinned.
        self._static_device_cache = (key, out, sharded)
        return out

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Halo-pad every state field, call ``update_padded``, strip halos.

        Sharded spatial axes get halo cells from neighbour shards via
        :func:`halo_exchange`.  Spatial axes with halo but no sharding
        (replicated axes) get their halos filled locally according to
        ``boundary``.  This lets a node with halo on every axis run
        under a partial-pencil mesh without the stencil needing to
        know which axes are sharded.

        Sharded static arrays declared by the inner node (via
        :class:`~maddening.core.static_data.StaticArray` with
        ``replication="shard"``) are materialised per-device and
        halo-exchanged with ``boundary="edge"`` before being passed
        through as ``static_padded`` to :meth:`update_padded`.

        The shard_map wrapper is built once per
        ``(state_shape, bi_shape, static_shape, static_data_hash)``
        signature and cached so repeated steps hit JAX's compile cache.
        """
        static_materialised = self._materialise_sharded_statics()
        fn = self._get_sharded_fn(state, boundary_inputs, static_materialised, params)
        return fn(
            state,
            boundary_inputs,
            jnp.asarray(dt, dtype=jnp.float32),
            static_materialised,
            params if params else {},
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _grid_shaped_boundary_inputs(
        self, state: dict, boundary_inputs: dict,
    ) -> frozenset[str]:
        """Names of the boundary inputs that are per-cell fields.

        A boundary input is grid-shaped when it has at least ``rank``
        dimensions and its leading ``rank`` dims equal the state fields'
        grid dims.  Those are sharded and halo-padded like state; anything
        else (a scalar pressure, a uniform ``(D,)`` force vector, a
        per-column profile along an unsharded axis) is replicated.

        The grid rank is bracketed by what the inner node declares --
        at least ``max(spatial axes in axis_map, halo_width keys) + 1``
        -- and by the state itself: no grid field can have fewer dims
        than the grid, so the smallest state field's ``ndim`` is an upper
        bound.  We take the larger of the two.  Matching only the sharded
        axes' extents (the previous rule) let an ``(n,)`` input on an
        ``n x n`` grid through by coincidence; requiring the full leading
        shape rules that out.  The residual ambiguity (every state field
        carries component dims *and* the grid has undeclared spatial
        axes) cannot be resolved from shapes alone.
        """
        halo = self._inner.halo_width()
        declared = list(self._axis_map.values()) + list(halo.keys())
        if not declared:
            return frozenset()
        rank_lb = max(int(a) for a in declared) + 1
        integrals = set(self._inner.domain_integral_fields())
        grid_fields = [
            arr for k, arr in state.items()
            if k not in integrals and jnp.ndim(arr) >= rank_lb
        ]
        if not grid_fields:
            return frozenset()
        rank = max(rank_lb, min(int(jnp.ndim(a)) for a in grid_fields))
        ref = tuple(int(n) for n in jnp.shape(grid_fields[0])[:rank])
        out = set()
        for k, v in boundary_inputs.items():
            shape = tuple(jnp.shape(v))
            if len(shape) >= rank and tuple(shape[:rank]) == ref:
                out.add(k)
        return frozenset(out)

    def _spec_for_field(self, arr: jax.Array) -> P:
        """PartitionSpec for a single state field array.

        Only the spatial axes referenced by ``axis_map`` are sharded;
        all others are replicated.
        """
        spec: list[Optional[str]] = [None] * arr.ndim
        for mesh_axis, spatial_axis in self._axis_map.items():
            if spatial_axis < arr.ndim:
                spec[spatial_axis] = mesh_axis
        return P(*spec)

    def _sharding_for_field(self, arr: jax.Array) -> NamedSharding:
        return NamedSharding(self._mesh, self._spec_for_field(arr))

    def _spec_for_static_key(self, key: str, arr: jax.Array) -> P:
        """PartitionSpec for a sharded StaticArray.

        Only the array's own ``shard_axis`` gets a mesh-axis assignment;
        every other axis is replicated.  Picks the (unique) mesh axis
        whose ``axis_map`` entry maps to that spatial axis.
        """
        sa = self._sharded_static[key]
        spec: list[Optional[str]] = [None] * arr.ndim
        for mesh_axis, spatial_axis in self._axis_map.items():
            if spatial_axis == sa.shard_axis and spatial_axis < arr.ndim:
                spec[spatial_axis] = mesh_axis
                break
        return P(*spec)

    def _field_needs_halo(self, arr: jax.Array) -> bool:
        """True if this field has at least one sharded spatial axis."""
        for spatial_axis in self._axis_map.values():
            if spatial_axis < arr.ndim:
                return True
        return False

    def _strip_halos(self, arr: jax.Array, *, original: jax.Array) -> jax.Array:
        """Strip every halo axis (sharded **and** replicated)."""
        halo = self._inner.halo_width()
        out = arr
        for spatial_axis in sorted(halo):
            if spatial_axis >= out.ndim:
                continue
            h = int(halo[spatial_axis])
            if h == 0:
                continue
            target = original.shape[spatial_axis]
            out = jax.lax.slice_in_dim(out, h, h + target, axis=spatial_axis)
        return out

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, "to_dict") else {}
        d["sharded"] = True
        d["sharded_stencil"] = True
        d["axis_map"] = self._axis_map
        d["boundary"] = self._boundary
        return d
