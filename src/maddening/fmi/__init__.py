"""FMI 3.0 substrate for MADDENING (added in v0.3.0).

The *generic* FMI 3.0 machinery lives here in MADDENING.  MIME's
planned ``mime-fmi`` is a thin selector that picks specific
subgraphs out of MIME and re-emits them as named, citeable FMUs.
The architectural work is here.

Why FMI 3.0, not 2.0
~~~~~~~~~~~~~~~~~~~~

The load-bearing reason is ``fmi3GetDirectionalDerivative``: MADDENING
is built on JAX autodiff, and ``jax.jvp`` / ``jax.vjp`` already produce
exact directional derivatives.  Exposing them through FMI 3 is a thin
shim (see :mod:`maddening.fmi.directional_derivatives`; the shipped FMU
binary does not reach it yet, see below).  FMI 2.0 has no equivalent —
we'd be deliberately discarding our most differentiating capability.

Other FMI 3 features that map well onto MADDENING:

* **Dynamic arrays** — map onto :class:`StaticArray`'s shape/dtype
  contract without fixed-size workarounds.
* **Opaque binary** — maps onto :class:`BinaryStateEncoder`'s schema
  for graph state that doesn't decompose cleanly into FMI scalars.
* **Scheduled execution (clocks)** — maps onto MADDENING's
  multi-rate scheduler.  Since 0.4.0 ``modelDescription.xml`` can
  declare one clock per node timestep
  (``build_model_description(multi_clock=True)``); the shipped C wrapper
  still refuses scheduled execution.
* **Co-simulation** — the interface type the shipped FMU implements;
  the C wrapper refuses model exchange.

What ships in 0.4.0
~~~~~~~~~~~~~~~~~~~

* :mod:`maddening.fmi.model_description` — builds
  ``modelDescription.xml`` from a :class:`GraphManager`'s public surface
  + the ``@stability`` audit registry.
* :mod:`maddening.fmi.package` — compiles the C wrapper
  (``maddening/fmi/c/maddening_fmu.c``; libc and the FMI 3.0 headers
  only) and packages the ``.fmu`` (:func:`build_fmu_binary`,
  :func:`write_fmu`).
* :mod:`maddening.fmi.tcp_bridge` — :class:`FmuTcpBridge`, the Python
  end of the C wrapper's wire protocol: length-prefixed JSON frames (and
  binary frames under protocol 2) over a plain TCP socket.  ZeroMQ is not
  involved.  The bridge authenticates no caller, so keep it on its
  default loopback bind (MADD-ANO-023).
* :mod:`maddening.fmi.sidecar` — :class:`FmuSidecar`, which holds the
  JAX-JITted graph the bridge drives.  Its pickle RPC (``handle``) is off
  by default, and nothing in MADDENING calls it.
* :mod:`maddening.fmi.fmu_state` — round-trips full graph state via
  the integrity manifest already shipped under v0.2 #8.
* :mod:`maddening.fmi.directional_derivatives` — wraps ``jax.jvp`` /
  ``jax.vjp`` behind a small ``fmi3GetDirectionalDerivative``-shaped
  Python API, for Python callers.  The FMU cannot reach it: the C
  wrapper's ``fmi3GetDirectionalDerivative`` returns ``fmi3Error``, and
  the bridge protocol has no derivative request.

Out of scope until later
~~~~~~~~~~~~~~~~~~~~~~~~

* FMU **import** + SSP — both later.
* MIME-side ``mime-fmi`` package + named-experiment FMUs +
  Simulink-workstation acceptance — MIME's, not MADDENING's.
  MADDENING ships the substrate; MIME picks specific subgraphs and
  emits citeable FMUs.
* Scheduled execution through the FMU binary, and directional
  derivatives through it — both refused by the C wrapper today.

Public API
~~~~~~~~~~

The substrate is tagged ``@stability(EVOLVING)``.  Promotion is
decided surface by surface in the stability freeze rounds that lead
to 1.0 (``docs/developer_guide/api_freeze_proposal.md``).  Within
evolving, additions are allowed but signatures are stable.
"""

from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)
from maddening.fmi.fmu_state import (
    FMUState,
    deserialize_fmu_state,
    serialize_fmu_state,
)
from maddening.fmi.model_description import (
    FMIVariable,
    ModelDescription,
    build_model_description,
)
from maddening.fmi.package import (
    MODEL_IDENTIFIER,
    build_fmu_binary,
    write_fmu,
)
from maddening.fmi.tcp_bridge import FmuTcpBridge

__all__ = [
    "DirectionalDerivativeKind",
    "FMIVariable",
    "FMUState",
    "FmuTcpBridge",
    "MODEL_IDENTIFIER",
    "ModelDescription",
    "build_fmu_binary",
    "build_model_description",
    "deserialize_fmu_state",
    "get_directional_derivative",
    "serialize_fmu_state",
    "write_fmu",
]
