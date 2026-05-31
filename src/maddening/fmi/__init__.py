"""FMI 3.0 substrate for MADDENING (v0.3.0 §A1).

Per STACK_V1 §3 M2 and the v0.3.0 plan: the *generic* FMI 3.0
machinery lives here in MADDENING.  MIME's ``mime-fmi`` (shipping in
v0.5.0 / STACK_V1 M3) is a thin selector that picks specific
subgraphs out of MIME and re-emits them as named, citeable FMUs.
The architectural work is here.

Why FMI 3.0, not 2.0
~~~~~~~~~~~~~~~~~~~~

The load-bearing reason is ``fmi3GetDirectionalDerivative``: MADDENING
is built on JAX autodiff, and ``jax.jvp`` / ``jax.vjp`` already produce
exact directional derivatives.  Exposing them through FMI 3 is a thin
shim (see :mod:`maddening.fmi.directional_derivatives`).  FMI 2.0 has
no equivalent — we'd be deliberately discarding our most
differentiating capability.

Other FMI 3 features that map well onto MADDENING:

* **Dynamic arrays** — map onto :class:`StaticArray`'s shape/dtype
  contract without fixed-size workarounds.
* **Opaque binary** — maps onto :class:`BinaryStateEncoder`'s schema
  for graph state that doesn't decompose cleanly into FMI scalars.
* **Scheduled execution (clocks)** — maps onto MADDENING's
  multi-rate scheduler.  v0.3.0 ships single-clock support; per-effect
  multi-rate is a later extension within FMI 3.
* **Co-simulation** — same contract as FMI 2.0; both supported.

Scope for v0.3.0
~~~~~~~~~~~~~~~~

* :mod:`maddening.fmi.model_description` — builds
  ``modelDescription.xml`` from a :class:`GraphManager`'s public surface
  + the ``@stability`` audit registry.
* :mod:`maddening.fmi.directional_derivatives` — wraps ``jax.jvp`` /
  ``jax.vjp`` behind a small ``fmi3GetDirectionalDerivative``-shaped
  Python API.  The FMU C shim calls into this via ZMQ.
* :mod:`maddening.fmi.fmu_state` — round-trips full graph state via
  the integrity manifest already shipped under v0.2 #8.
* :mod:`maddening.fmi.sidecar` — out-of-process ZMQ shim that the FMU
  C wrapper marshals into.  v0.3.0 ships the protocol + a Python
  reference implementation; the C wrapper itself is a v0.4.0 / MIME
  v0.5.0 deliverable.

Out of scope until later
~~~~~~~~~~~~~~~~~~~~~~~~

* FMU **import** + SSP — both later (per ADR-2026-FMI Rev 3).
* MIME-side ``mime-fmi`` package + named-experiment FMUs +
  Simulink-workstation acceptance — STACK_V1 M3 (MIME v0.5.0).
  MADDENING ships the substrate in v0.3.0; MIME picks specific
  subgraphs and emits citeable FMUs in M3.
* Per-effect multi-rate clock-based FMU export — designed-in
  (the FMI 3 clock concept), single-clock-only for v0.3.0.

Public API
~~~~~~~~~~

The substrate is tagged ``@stability(EVOLVING)`` until M4 (v0.9.0)
when STACK_V1 freezes the surface.  Within evolving, additions are
allowed but signatures are stable.
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

__all__ = [
    "DirectionalDerivativeKind",
    "FMIVariable",
    "FMUState",
    "ModelDescription",
    "build_model_description",
    "deserialize_fmu_state",
    "get_directional_derivative",
    "serialize_fmu_state",
]
