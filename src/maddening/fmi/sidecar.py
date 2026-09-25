"""Sidecar for the MADDENING FMU shim, and its (pickle) in-process protocol.

Architecture
~~~~~~~~~~~~

The FMU itself is a compiled DLL/dylib/so loaded into a host
co-simulation tool (FMPy / Simulink / OpenModelica).  The host calls
FMI 3.0 C functions on it; the DLL marshals each call into a
length-prefixed frame and forwards it over TCP to
:class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, which drives a
long-running Python :class:`FmuSidecar` holding the JAX-JITted graph.
This is the only way to avoid paying XLA's startup cost on every FMU
instantiation.  No ZeroMQ is involved.

Wire format
~~~~~~~~~~~

.. warning::

   :meth:`FmuSidecar.handle` speaks a **pickled** request/response
   protocol, and unpickling a request executes whatever the sender put
   in it.  It is therefore **off by default** since 0.4.0: a sidecar has
   to be built with ``SidecarConfig(allow_pickle_rpc=True)`` before
   ``handle`` will answer at all, and that opt-in is only ever
   appropriate for an in-process caller whose bytes you already trust as
   much as your own code.  Nothing in MADDENING calls it: the shipped
   transport is :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, which
   speaks length-prefixed JSON and raw arrays and never unpickles
   anything.

Every message is a length-prefixed bytes blob; the payload is a
Python pickle.  Two message kinds:

* Request  (host → sidecar): ``("step",      external_inputs)``
                              ``("get_dd",   kind, x, v)``
                              ``("get_state",)``
                              ``("set_state", payload, token)``
* Response (sidecar → host): ``("ok",        result)``
                              ``("err",       traceback)``

Stability
~~~~~~~~~

The protocol is tagged ``@stability(EVOLVING)`` — settled enough that
the FMU's C wrapper can be written against it, but additions
(per-clock event signalling, FMU-state caching) may grow before M4.

Reference implementation
~~~~~~~~~~~~~~~~~~~~~~~~

The Python sidecar lives in this module as :class:`FmuSidecar`.  The
C wrapper that ships in the FMU (``maddening/fmi/c/maddening_fmu.c``)
talks to it through :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`, a
TCP transport carrying length-prefixed JSON (no libzmq needed on the
importer's side); the pickle protocol below stays for in-process use
and Python clients.  Tests can call the sidecar directly without a
socket — see ``tests/fmi/test_sidecar.py``.
"""

from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)
from maddening.fmi.fmu_state import (
    FMUState,
    deserialize_fmu_state,
    serialize_fmu_state,
)


def _copy_tree(tree: Any) -> Any:
    """Shallow-copy the dict spine so ``set_params`` never mutates the
    caller's pytree (leaves are immutable arrays)."""
    if isinstance(tree, dict):
        return {k: _copy_tree(v) for k, v in tree.items()}
    return tree


def _not_tunable_error(name: str, reason: str, current: Any) -> ValueError:
    """The refusal for a new value of a parameter the step cannot read.

    Shared by :meth:`FmuSidecar.set_params`, :meth:`FmuSidecar.set_fmu_state`
    and the bridge's ``set_state``, so every door into the parameter tree
    says the same thing.
    """
    return ValueError(
        f"parameter {name!r} is not tunable: {reason}.  The FMU would report "
        f"the new value while every step kept computing with "
        f"{np.array2string(np.asarray(current), threshold=8)}; nothing was "
        "written"
    )


def _same_leaf(a: Any, b: Any) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return x.shape == y.shape and bool(np.array_equal(x, y))


@stability(StabilityLevel.EVOLVING)
@dataclass(frozen=True)
class SidecarConfig:
    """Static configuration handed to :class:`FmuSidecar` at startup.

    Attributes
    ----------
    schema_token : str
        The instantiation token of the
        :class:`maddening.fmi.model_description.ModelDescription` this
        FMU was built from.  Used to validate snapshots on SetFMUState.
    step_fn : callable
        ``step_fn(state, external_inputs) -> new_state``.  Typically
        :meth:`GraphManager.step` bound to a particular graph.
    initial_state : dict
        The seed state at FMU instantiation time.
    unknown_fn : callable, optional
        ``unknown_fn(x) -> y`` for directional derivative requests.
        If ``None``, ``get_dd`` requests will error.
    params : dict, optional
        The graph parameter pytree (``GraphManager.params`` layout).
        When given, ``step_fn`` is called as
        ``step_fn(state, external_inputs, params)`` — the compiled
        step's contract — and the FMI ``parameter`` variables
        (``<node>.params.<key>``) are served from / written into it by
        :meth:`FmuSidecar.get_params` / :meth:`FmuSidecar.set_params`
        and carried in the FMU state snapshot.
    param_specs : dict, optional
        ``GraphManager.param_specs()`` for the same graph.  When given,
        :meth:`FmuSidecar.set_params` rejects a value outside a leaf's
        declared ``ParamSpec.bounds`` (the ``min`` / ``max`` the model
        description advertises), so an importer cannot drive the step
        with a constant the graph declares invalid.  The FMU-state
        archive path in
        :class:`maddening.fmi.tcp_bridge.FmuTcpBridge` checks against the
        same declarations, so neither door into the parameter tree is
        wider than the other.
    fixed_params : mapping of str to str, optional
        ``{"<node>.params.<key>": reason}`` for parameters the compiled step
        cannot read -- pass :attr:`ModelDescription.fixed_parameters
        <maddening.fmi.model_description.ModelDescription.fixed_parameters>`.
        :meth:`FmuSidecar.set_params` and :meth:`FmuSidecar.set_fmu_state`
        refuse a *new* value for any of them (re-setting the current value
        is accepted): it would be reported by :meth:`FmuSidecar.get_params`
        and ignored by every step.  A sidecar holds only the compiled step,
        so it cannot work this out alone; :class:`FmuTcpBridge
        <maddening.fmi.tcp_bridge.FmuTcpBridge>` fills it from the model
        description it serves, whatever was passed here.
    allow_pickle_rpc : bool, default False
        Let :meth:`FmuSidecar.handle` serve the pickled RPC protocol.
        Unpickling a request runs arbitrary code from whoever supplied
        the bytes, so ``handle`` refuses unless this is set; the TCP
        bridge does not use it and does not need it.
    """
    schema_token: str
    step_fn: Callable[..., dict]
    initial_state: dict[str, dict[str, Any]]
    unknown_fn: Optional[Callable[[Any], Any]] = None
    params: Optional[dict] = None
    param_specs: Optional[dict] = None
    fixed_params: Optional[Mapping[str, str]] = None
    allow_pickle_rpc: bool = False


@stability(StabilityLevel.EVOLVING)
class FmuSidecar:
    """In-process FMU sidecar — handles FMI 3.0 RPC messages.

    Holds the JAX-JITted graph in long-running memory.  A real
    deployment runs it in a Python process of its own, apart from the
    importer's, behind :class:`~maddening.fmi.tcp_bridge.FmuTcpBridge`;
    tests instantiate it directly and call :meth:`handle` to exercise the
    protocol without involving a real socket.
    """

    def __init__(self, config: SidecarConfig) -> None:
        self._config = config
        self._state = dict(config.initial_state)
        self._params = (
            None if config.params is None else _copy_tree(config.params)
        )
        self._fixed: dict[str, str] = dict(config.fixed_params or {})

    def _refuse_new_values_for(self, fixed: Mapping[str, str]) -> None:
        """Add ``{"<node>.params.<key>": reason}`` to the parameters whose
        new values are refused (the bridge's model-description contract)."""
        for name, reason in fixed.items():
            self._fixed.setdefault(name, reason)

    @property
    def fixed_params(self) -> dict[str, str]:
        """The parameters this sidecar refuses a new value for, with why."""
        return dict(self._fixed)

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return self._state

    @property
    def params(self) -> Optional[dict]:
        return self._params

    @property
    def param_specs(self) -> Optional[dict]:
        """The ``ParamSpec`` tree this sidecar validates against, if any.

        ``{"nodes": {name: {key: ParamSpec}}, "mappings": {...}}``, the
        layout :meth:`GraphManager.param_specs` returns.  Exposed so that
        every writer into the parameter tree -- ``set_params`` here and
        the FMU-state archive in
        :meth:`maddening.fmi.tcp_bridge.FmuTcpBridge._decode_state` --
        checks against the same declarations.
        """
        return self._config.param_specs

    # -- High-level handlers -------------------------------------------------

    def step(self, external_inputs: dict[str, dict[str, Any]]) -> dict:
        if self._params is None:
            self._state = self._config.step_fn(self._state, external_inputs)
        else:
            self._state = self._config.step_fn(
                self._state, external_inputs, self._params,
            )
        return self._state

    def get_params(self) -> dict[str, Any]:
        """``{"<node>.params.<key>": value}`` for every FMI ``parameter``
        variable (the names ``build_model_description`` emits)."""
        if self._params is None:
            return {}
        return {
            f"{node}.params.{key}": value
            for node, leaves in self._params.get("nodes", {}).items()
            for key, value in leaves.items()
        }

    def set_params(self, updates: dict[str, Any]) -> None:
        """Write FMI ``parameter`` variables (``fmi3SetFloat*`` on a
        parameter value reference) into the pytree the next step uses.

        Keys are ``"<node>.params.<key>"``; an unknown name, a shape
        that differs from the current leaf, a new value for a parameter the
        step cannot read (``SidecarConfig.fixed_params``), or a value
        outside the leaf's ``ParamSpec.bounds`` (when the config carries
        ``param_specs``) is an error, so an importer cannot silently
        tune a constant the step never reads or declares invalid.  The
        call is atomic: nothing is written unless every update is valid.
        """
        if self._params is None:
            raise RuntimeError(
                "Sidecar wasn't configured with a params pytree; the FMU "
                "exposes no parameter variables.",
            )
        nodes = self._params.get("nodes", {})
        spec_nodes = (self._config.param_specs or {}).get("nodes", {})
        staged: list[tuple[str, str, Any]] = []
        for name, value in updates.items():
            node, sep, key = name.partition(".params.")
            if not sep or node not in nodes or key not in nodes[node]:
                raise KeyError(
                    f"unknown parameter {name!r}; known: {sorted(self.get_params())}",
                )
            current = nodes[node][key]
            new = jnp.asarray(value, dtype=current.dtype)
            if new.shape != current.shape:
                raise ValueError(
                    f"parameter {name!r} has shape {current.shape}, got {new.shape}",
                )
            if name in self._fixed and not _same_leaf(new, current):
                raise _not_tunable_error(name, self._fixed[name], current)
            spec = spec_nodes.get(node, {}).get(key)
            if spec is not None:
                spec.check(new, name=name)
            staged.append((node, key, new))
        for node, key, new in staged:
            nodes[node][key] = new

    def get_directional_derivative(
        self, kind: DirectionalDerivativeKind, x: Any, v: Any,
    ) -> Any:
        if self._config.unknown_fn is None:
            raise RuntimeError(
                "Sidecar wasn't configured with an unknown_fn; cannot "
                "answer get_directional_derivative requests.",
            )
        return get_directional_derivative(
            self._config.unknown_fn, kind=kind, x=x, v=v,
        )

    def get_fmu_state(self) -> FMUState:
        return serialize_fmu_state(
            state=self._state,
            schema_token=self._config.schema_token,
            params=self._params,
        )

    def set_fmu_state(self, fmu_state: FMUState) -> None:
        state, params = deserialize_fmu_state(
            fmu_state, expected_schema_token=self._config.schema_token,
            return_params=True,
        )
        new_params = None
        if params is not None and self._params is not None:
            new_params = {
                k: ({n: {kk: jnp.asarray(vv) for kk, vv in leaves.items()}
                     for n, leaves in v.items()} if isinstance(v, dict) else v)
                for k, v in params.items()
            }
            # A snapshot is a door into the parameter tree like set_params:
            # it may not install a new value for a parameter the step cannot
            # read.  Checked before anything is committed.
            live_nodes = self._params.get("nodes", {})
            for name, reason in self._fixed.items():
                node, _, key = name.partition(".params.")
                if key not in live_nodes.get(node, {}):
                    continue
                current = live_nodes[node][key]
                restored = new_params.get("nodes", {}).get(node, {}).get(key, current)
                if not _same_leaf(restored, current):
                    raise _not_tunable_error(name, reason, current)
        self._state = state
        if new_params is not None:
            self._params = new_params

    # -- Wire-level RPC -----------------------------------------------------

    def handle(self, request: bytes) -> bytes:
        """Parse one wire request and produce one wire response.

        Both directions are pickled tuples.  Errors are caught and
        returned as ``("err", traceback_string)`` so the C wrapper
        can surface the failure to the FMI runtime via
        ``fmi3Status`` without losing the Python traceback.

        .. warning::

           ``pickle.loads`` on the request executes whatever produced the
           bytes.  This method therefore refuses to run unless the
           sidecar was built with
           ``SidecarConfig(allow_pickle_rpc=True)``.  The shipped
           transport, :class:`maddening.fmi.tcp_bridge.FmuTcpBridge`,
           does not go through here; prefer it.

        Raises
        ------
        RuntimeError
            If the sidecar was not built with ``allow_pickle_rpc=True``.
        """
        if not self._config.allow_pickle_rpc:
            raise RuntimeError(
                "FmuSidecar.handle speaks a pickled protocol, and unpickling "
                "a request executes whatever produced it.  It is disabled "
                "unless the sidecar is built with "
                "SidecarConfig(..., allow_pickle_rpc=True), which is only "
                "appropriate for an in-process caller you trust as much as "
                "your own code.  The shipped FMU transport is "
                "maddening.fmi.tcp_bridge.FmuTcpBridge, which never "
                "unpickles anything.",
            )
        try:
            payload = pickle.loads(request)
        except Exception as exc:  # pragma: no cover — pickle errors
            return pickle.dumps(("err", f"unpickle failed: {exc!r}"))

        try:
            kind = payload[0]
            if kind == "step":
                result = self.step(payload[1])
                return pickle.dumps(("ok", result))
            if kind == "get_dd":
                _, dd_kind, x, v = payload
                result = self.get_directional_derivative(dd_kind, x, v)
                return pickle.dumps(("ok", result))
            if kind == "get_state":
                fmu_state = self.get_fmu_state()
                return pickle.dumps(("ok", fmu_state))
            if kind == "get_params":
                return pickle.dumps(("ok", {
                    k: np.asarray(v) for k, v in self.get_params().items()
                }))
            if kind == "set_params":
                _, updates = payload
                self.set_params(updates)
                return pickle.dumps(("ok", None))
            if kind == "set_state":
                _, fmu_state = payload
                self.set_fmu_state(fmu_state)
                return pickle.dumps(("ok", None))
            return pickle.dumps((
                "err", f"unknown request kind {kind!r}",
            ))
        except Exception:  # noqa: BLE001 — broad catch by design
            return pickle.dumps(("err", traceback.format_exc()))


__all__ = [
    "FmuSidecar",
    "SidecarConfig",
]
