"""
HealthCheckNode -- execution-layer fault detection (Section 9.8).

A SimulationNode subclass designed to be instantiated and configured by
downstream libraries.  The HealthCheck node participates in the graph as
a regular node: it receives state from monitored nodes via edges, performs
checks, and writes diagnostic outputs (pass/fail flags, check values)
into its own state.

The HealthCheck node does NOT halt the simulation — halting is a control
decision that belongs to the downstream application layer.
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.EXPERIMENTAL)
class HealthCheckNode(SimulationNode):
    """Base health monitor that downstream libraries configure.

    Receives state from one or more monitored nodes via edges.
    Performs configurable checks and writes results to its own state.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    checks : dict
        Mapping of field names to check configurations::

            {"density": {"finite": True, "min": 0.0, "max": 10.0},
             "velocity": {"finite": True, "max_abs": 100.0}}

        Supported check types per field:
        - ``finite``: bool — check for NaN/Inf
        - ``min``: float — check value >= min
        - ``max``: float — check value <= max
        - ``max_abs``: float — check |value| <= max_abs
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-HEALTH-001",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Base health monitor for execution-layer fault detection",
        assumptions=(
            "Monitored state is received via boundary inputs (edges)",
            "Check results are outputs only — halting logic belongs to the application layer",
        ),
        limitations=(
            "Cannot detect errors that produce in-range but incorrect values",
            "Moment checks require historical state — first step has no baseline",
        ),
        hazard_hints=(
            "Reliability depends on using numerical primitives independent of the "
            "monitored node — if both share the same XLA compilation path, a compiler "
            "bug may corrupt both simultaneously (see Section 9.8 Algorithmic Diversity "
            "Principle)",
            "Does not halt the simulation — check results are outputs only; halting "
            "logic belongs to the application layer",
            "Cannot detect errors that occur before the monitored state is passed "
            "via boundary inputs",
        ),
    )

    def __init__(self, name: str, timestep: float, checks: dict | None = None):
        super().__init__(name, timestep, checks=checks or {})

    @property
    def requires_halo(self) -> bool:
        """Pointwise (no spatial neighbor access)."""
        return False

    def initial_state(self) -> dict:
        checks = self.params.get("checks", {})
        n_checks = len(checks)
        return {
            "all_passed": jnp.array(True),
            "n_checks": jnp.array(n_checks, dtype=jnp.int32),
            "n_passed": jnp.array(n_checks, dtype=jnp.int32),
            "n_failed": jnp.array(0, dtype=jnp.int32),
            "check_results": jnp.ones(max(n_checks, 1), dtype=jnp.bool_),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Run configured checks on monitored fields from boundary inputs.

        Uses only simple JAX primitives (jnp.isfinite, jnp.sum, jnp.all,
        comparisons) — fundamentally different from any physics node's
        numerical operations, per the Algorithmic Diversity Principle.
        """
        checks = self.params.get("checks", {})
        if not checks:
            return state

        results = []
        for field_name, cfg in checks.items():
            field_data = boundary_inputs.get(field_name, None)
            if field_data is None:
                results.append(jnp.array(True))
                continue

            field_ok = jnp.array(True)

            # NaN/Inf check
            if cfg.get("finite", False):
                field_ok = field_ok & jnp.all(jnp.isfinite(field_data))

            # Min bound check
            if "min" in cfg:
                field_ok = field_ok & jnp.all(field_data >= cfg["min"])

            # Max bound check
            if "max" in cfg:
                field_ok = field_ok & jnp.all(field_data <= cfg["max"])

            # Max absolute value check
            if "max_abs" in cfg:
                field_ok = field_ok & jnp.all(jnp.abs(field_data) <= cfg["max_abs"])

            results.append(field_ok)

        check_arr = jnp.stack(results) if results else jnp.ones(1, dtype=jnp.bool_)
        n_passed = jnp.sum(check_arr.astype(jnp.int32))
        n_failed = jnp.array(len(results), dtype=jnp.int32) - n_passed

        return {
            "all_passed": jnp.all(check_arr),
            "n_checks": jnp.array(len(results), dtype=jnp.int32),
            "n_passed": n_passed,
            "n_failed": n_failed,
            "check_results": check_arr,
        }
