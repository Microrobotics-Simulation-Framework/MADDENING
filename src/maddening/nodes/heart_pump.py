"""
HeartPumpNode -- 2-element Windkessel model for arterial pressure.

Generates physiologically plausible arterial pressure waveforms using
a lumped-parameter model of arterial compliance::

    dP_art/dt = (Q_heart(t) - Q_out) / C

where:
    P_art = arterial pressure
    Q_heart(t) = cardiac output waveform (pulsatile, sinusoidal systole)
    Q_out = (P_art - P_downstream) / R  (outflow through vascular resistance)
    C = arterial compliance
    R = peripheral resistance

The cardiac output waveform ``Q_heart(phase)`` is sinusoidal during
systole and zero during diastole.  ``phase`` advances from 0 to 1
each heartbeat at a rate of ``heart_rate / 60``.
"""

import jax.numpy as jnp

from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.compliance.metadata import NodeMeta, StabilityLevel, ValidatedRegime
from maddening.core.compliance.stability import stability


def _cardiac_output(phase, systole_fraction, q_max):
    """Compute instantaneous cardiac output from cycle phase.

    Uses ``jnp.where`` for JAX traceability (no Python if/else).

    Parameters
    ----------
    phase : scalar
        Position in cardiac cycle [0, 1).
    systole_fraction : scalar
        Fraction of cycle that is systole.
    q_max : scalar
        Peak flow rate during systole.

    Returns
    -------
    scalar
        Instantaneous cardiac output.
    """
    # Sinusoidal flow during systole, zero during diastole
    q_systole = q_max * jnp.sin(jnp.pi * phase / systole_fraction)
    return jnp.where(phase < systole_fraction, q_systole, 0.0)


@stability(StabilityLevel.EXPERIMENTAL)
class HeartPumpNode(SimulationNode):
    """2-element Windkessel heart pump model.

    Models pulsatile arterial pressure using a lumped-parameter
    Windkessel circuit with sinusoidal cardiac output during systole.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    resistance : float
        Peripheral vascular resistance R (default 1.0).
    compliance : float
        Arterial compliance C (default 1.0).
    heart_rate : float
        Heart rate in beats per minute (default 72).
    stroke_volume : float
        Volume ejected per beat (default 70, in consistent units).
    venous_pressure : float
        Downstream venous pressure (default 0).
    systole_fraction : float
        Fraction of cardiac cycle that is systole (default 0.35).
    initial_pressure : float
        Starting arterial pressure (default 80).

    Boundary inputs
    ---------------
    backpressure : scalar
        Pressure feedback from downstream domain (e.g. LBM outlet).
        When provided, replaces ``venous_pressure`` in the outflow
        computation.
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-008",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="2-element Windkessel heart pump model",
        governing_equations="dP/dt = (Q_heart - (P-P_v)/R) / C",
        discretization="Forward Euler (explicit, 1st-order)",
        assumptions=(
            "Lumped parameters (spatially uniform arterial compartment)",
            "Rigid arterial walls (compliance is constant)",
            "Sinusoidal systolic flow waveform",
            "Instantaneous valve opening/closing (no valve dynamics)",
            "No inertial effects (2-element model, no inductance)",
        ),
        limitations=(
            "Forward Euler is only 1st-order -- large timesteps cause pressure drift",
            "No wave propagation (lumped model)",
            "Compliance and resistance are constant (no autoregulation)",
            "No coronary circulation or venous return dynamics",
        ),
        validated_regimes=(
            ValidatedRegime("heart_rate", 40.0, 180.0, "bpm"),
            ValidatedRegime("systole_fraction", 0.2, 0.5),
            ValidatedRegime("resistance", 0.01, 100.0),
            ValidatedRegime("compliance", 0.001, 100.0),
        ),
        hazard_hints=(
            "Forward Euler can produce negative pressures with large dt or low compliance",
            "No stability check on dt relative to R*C time constant",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        resistance: float = 1.0,
        compliance: float = 1.0,
        heart_rate: float = 72.0,
        stroke_volume: float = 70.0,
        venous_pressure: float = 0.0,
        systole_fraction: float = 0.35,
        initial_pressure: float = 80.0,
    ):
        super().__init__(
            name,
            timestep,
            resistance=resistance,
            compliance=compliance,
            heart_rate=heart_rate,
            stroke_volume=stroke_volume,
            venous_pressure=venous_pressure,
            systole_fraction=systole_fraction,
            initial_pressure=initial_pressure,
        )

    def _compute_q_max(self):
        """Compute peak systolic flow rate from stroke volume.

        The integral of Q_max * sin(pi * phase / sf) over phase in
        [0, sf] equals Q_max * sf * 2 / pi (in phase-time).
        Converting to real time: integral = Q_max * sf * (1/f) * 2/pi
        where f = heart_rate/60.  Setting this equal to stroke_volume:

            Q_max = stroke_volume * pi / (2 * sf / f)
                  = stroke_volume * pi * f / (2 * sf)
        """
        sv = self.params["stroke_volume"]
        hr = self.params["heart_rate"]
        sf = self.params["systole_fraction"]
        freq = hr / 60.0  # beats per second
        return sv * jnp.pi * freq / (2.0 * sf)

    @property
    def requires_halo(self) -> bool:
        """Pointwise (no spatial neighbor access)."""
        return False

    def initial_state(self) -> dict:
        return {
            "arterial_pressure": jnp.array(
                self.params["initial_pressure"], dtype=jnp.float32
            ),
            "phase": jnp.array(0.0, dtype=jnp.float32),
            "flow_rate": jnp.array(0.0, dtype=jnp.float32),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Forward Euler update for the 2-element Windkessel model.

        1. Advance cardiac cycle phase.
        2. Compute cardiac output from waveform.
        3. Compute outflow through peripheral resistance.
        4. Update arterial pressure.
        """
        R = self.params["resistance"]
        C = self.params["compliance"]
        hr = self.params["heart_rate"]
        sf = self.params["systole_fraction"]
        P_venous = self.params["venous_pressure"]

        P_art = state["arterial_pressure"]
        phase = state["phase"]

        # 1. Advance phase
        phase_new = jnp.fmod(phase + dt * hr / 60.0, 1.0)

        # 2. Compute cardiac output
        q_max = self._compute_q_max()
        Q_heart = _cardiac_output(phase_new, sf, q_max)

        # 3. Compute outflow (use backpressure if provided)
        P_downstream = boundary_inputs.get("backpressure", P_venous)
        # Ensure P_downstream is a JAX array for traceability
        P_downstream = jnp.asarray(P_downstream, dtype=jnp.float32)
        Q_out = (P_art - P_downstream) / R

        # 4. Update pressure
        dP_dt = (Q_heart - Q_out) / C
        P_art_new = P_art + dP_dt * dt

        return {
            "arterial_pressure": P_art_new,
            "phase": phase_new,
            "flow_rate": Q_heart,
        }

    def derivatives(self, state, boundary_inputs):
        """Time derivatives for higher-order integration.

        Note: flow_rate derivative is not continuous (waveform has
        a discontinuity at systole/diastole transition), so we return
        an approximate zero for it.
        """
        R = self.params["resistance"]
        C = self.params["compliance"]
        hr = self.params["heart_rate"]
        sf = self.params["systole_fraction"]
        P_venous = self.params["venous_pressure"]

        P_art = state["arterial_pressure"]
        phase = state["phase"]

        q_max = self._compute_q_max()
        Q_heart = _cardiac_output(phase, sf, q_max)

        P_downstream = boundary_inputs.get("backpressure", P_venous)
        P_downstream = jnp.asarray(P_downstream, dtype=jnp.float32)
        Q_out = (P_art - P_downstream) / R

        dP_dt = (Q_heart - Q_out) / C

        return {
            "arterial_pressure": dP_dt,
            "phase": jnp.array(hr / 60.0, dtype=jnp.float32),
            "flow_rate": jnp.array(0.0, dtype=jnp.float32),
        }

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt):
        """Backward Euler residual for the pressure ODE.

        R(x_new) = x_new - x_old - dt * f(x_new, boundary_inputs)
        """
        derivs = self.derivatives(state_new, boundary_inputs)
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        """Expose arterial pressure as inlet_pressure for downstream coupling."""
        return {
            "inlet_pressure": state["arterial_pressure"],
        }

    def boundary_input_spec(self):
        return {
            "backpressure": BoundaryInputSpec(
                shape=(),
                description="Downstream pressure feedback",
            ),
        }
