"""ZMQ sidecar protocol for the MADDENING FMU shim.

Architecture
~~~~~~~~~~~~

The FMU itself is a compiled DLL/dylib/so loaded into a host
co-simulation tool (FMPy / Simulink / OpenModelica).  The host calls
FMI 3.0 C functions on it; the DLL marshals each call into a ZMQ
message and forwards it to a long-running Python sidecar process
that holds the JAX-JITted graph.  This is the only way to avoid
paying XLA's startup cost on every FMU instantiation.

Wire format
~~~~~~~~~~~

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
C wrapper that ships in the FMU itself is out of scope for v0.3.0
(it's a v0.4.0 / MIME v0.5.0 deliverable).  Tests can call the
sidecar directly without going through a real ZMQ socket — see
``tests/fmi/test_sidecar.py``.
"""

from __future__ import annotations

import pickle
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
        with a constant the graph declares invalid.
    """
    schema_token: str
    step_fn: Callable[..., dict]
    initial_state: dict[str, dict[str, Any]]
    unknown_fn: Optional[Callable[[Any], Any]] = None
    params: Optional[dict] = None
    param_specs: Optional[dict] = None


@stability(StabilityLevel.EVOLVING)
class FmuSidecar:
    """In-process FMU sidecar — handles FMI 3.0 RPC messages.

    Holds the JAX-JITted graph in long-running memory.  A real
    deployment runs this as a separate process behind a ZMQ socket;
    tests instantiate it directly and call :meth:`handle` to
    exercise the protocol without involving a real socket.
    """

    def __init__(self, config: SidecarConfig) -> None:
        self._config = config
        self._state = dict(config.initial_state)
        self._params = (
            None if config.params is None else _copy_tree(config.params)
        )

    @property
    def state(self) -> dict[str, dict[str, Any]]:
        return self._state

    @property
    def params(self) -> Optional[dict]:
        return self._params

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
        that differs from the current leaf, or a value outside the
        leaf's ``ParamSpec.bounds`` (when the config carries
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
        self._state = state
        if params is not None and self._params is not None:
            self._params = {
                k: ({n: {kk: jnp.asarray(vv) for kk, vv in leaves.items()}
                     for n, leaves in v.items()} if isinstance(v, dict) else v)
                for k, v in params.items()
            }

    # -- Wire-level RPC -----------------------------------------------------

    def handle(self, request: bytes) -> bytes:
        """Parse one wire request and produce one wire response.

        Both directions are pickled tuples.  Errors are caught and
        returned as ``("err", traceback_string)`` so the C wrapper
        can surface the failure to the FMI runtime via
        ``fmi3Status`` without losing the Python traceback.
        """
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
