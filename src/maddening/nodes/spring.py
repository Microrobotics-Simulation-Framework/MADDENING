"""
SpringDamperNode -- a spring-damper connecting two attachment points.

Models a linear spring with viscous damping.  One end of the spring is
the node's own ``position``; the other end arrives via
``boundary_inputs["anchor_position"]``.

The entire ``update`` uses ``jnp`` operations so it is fully
JAX-traceable and JIT-compilable.
"""

import math

import jax.numpy as jnp
import numpy as np

from maddening.core.node import (
    BoundaryFluxSpec,
    BoundaryInputSpec,
    SimulationNode,
    _method_with_params,
)
from maddening.core.compliance.metadata import (
    DiscretizationOrder,
    NodeMeta,
    StabilityLevel,
    ValidatedRegime,
)
from maddening.core.compliance.stability import stability
from maddening.core.params import ParamSpec


@stability(StabilityLevel.STABLE)
class SpringDamperNode(SimulationNode):
    """A spring-damper connecting two attachment points.

    Models a linear spring with damping::

        F = -k * (position - anchor_position - rest_length) - c * velocity
        acceleration = F / mass

    The node tracks its own position (one end of the spring).
    The other end comes via ``boundary_inputs["anchor_position"]``.

    Integration uses semi-implicit Euler (velocity updated first, then
    position uses the new velocity) for better energy behaviour.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    stiffness : float
        Spring constant *k* (N/m).  Default 100.0.
    damping : float
        Damping coefficient *c* (N s/m).  Default 1.0.
    mass : float
        Point mass at this end of the spring (kg).  Default 1.0.
    rest_length : float
        Natural (unstretched) length of the spring.  Default 1.0.
    initial_position : float
        Starting position of this end.  Default 0.0.
    initial_velocity : float
        Starting velocity of this end.  Default 0.0.
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-003",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="Linear spring-damper connecting two attachment points",
        governing_equations="F = -k*(x - anchor - rest) - c*v; a = F/m",
        discretization="Semi-implicit Euler (1st-order, better energy conservation than forward Euler)",
        discretization_order=DiscretizationOrder(
            spatial=None,
            temporal=1.0,
            notes=(
                "Semi-implicit (symplectic) Euler -- the velocity is "
                "advanced first and the position uses the already-updated "
                "velocity -- so the scheme is 1st order globally in both "
                "state fields.  Unlike a body under a state-independent "
                "force, the position does not gain an order here: the "
                "spring force depends on the position itself, so the "
                "leading error term does not cancel.  No spatial order -- "
                "the node integrates an ODE.  Measured by MADD-VER-009."
            ),
        ),
        assumptions=(
            "Linear spring (Hooke's law)",
            "Viscous damping (linear in velocity)",
            "Point mass (no rotational dynamics)",
        ),
        limitations=(
            "1st-order integration — energy drift over long simulations",
            "No nonlinear spring behaviour (hardening, softening)",
            "No collision detection with other objects",
            "Conditionally stable, and nothing checks it: against a fixed "
            "or prescribed anchor the scheme is stable only for "
            "k*dt**2 + 2*c*dt < 4*m (dt < 2*sqrt(m/k) undamped)",
            "Two nodes anchored on each other in a converged coupling group "
            "are each explicit in their own position and implicit in the "
            "partner's, so the action-reaction pair does not cancel: the "
            "centre-of-mass velocity of an equal pair (same k, c, m) is "
            "multiplied by (m - c*dt)/(m - k*dt**2) every step.  Where the "
            "coupling iteration converges (k*dt**2 < m) and c*dt < m, the "
            "pair is stable only for c >= k*dt, far inside the single-node "
            "limit; past it the pair drifts off with growing speed.  With "
            "heavier damping it is stable up to c*dt + k*dt**2 = 2*m, past "
            "which its common velocity flips sign and grows (MADD-ANO-098).  "
            "compile() warns about such a pair; it does not refuse it",
        ),
        validated_regimes=(
            ValidatedRegime("stiffness", 0.01, 1e6, "N/m", "Tested range; very stiff springs need small dt"),
            ValidatedRegime("damping", 0.0, 1e4, "N·s/m"),
        ),
        hazard_hints=(
            "Very stiff springs (k > 1e4) with large dt can cause numerical instability",
            "Zero mass causes division by zero",
            "Two springs anchored on each other in a coupling group gain "
            "momentum without bound when stiffness*dt > damping (equal "
            "pair, stiffness*dt**2 < mass), and compile() only warns: keep "
            "damping >= stiffness*dt, or the timestep below "
            "damping/stiffness, without raising damping*dt + "
            "stiffness*dt**2 above 2*mass",
        ),
    )

    def __init__(
        self,
        name: str,
        timestep: float,
        stiffness: float = 100.0,
        damping: float = 1.0,
        mass: float = 1.0,
        rest_length: float = 1.0,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ):
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        super().__init__(
            name,
            timestep,
            stiffness=stiffness,
            damping=damping,
            mass=mass,
            rest_length=rest_length,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )

    def halo_width(self) -> dict[int, int]:
        """Pointwise (no spatial neighbour access)."""
        return {}

    @staticmethod
    def _coupling_group_advisories(**context) -> list[str]:
        """Warnings about this node's coupling group (for ``GraphManager.validate``).

        Private on purpose, like ``HeatNode._coupling_group_advisories``:
        it is an internal convention between the built-in nodes and
        ``GraphManager``, not yet part of the node contract.  See
        :func:`_anchored_pair_advisories` for what it warns about.
        """
        return _anchored_pair_advisories(**context)

    def param_specs(self) -> dict[str, ParamSpec]:
        return {
            **super().param_specs(),
            "stiffness": ParamSpec(bounds=(0.0, None), transform="log", units="N/m"),
            "damping": ParamSpec(bounds=(0.0, None), units="N*s/m"),
            "mass": ParamSpec(bounds=(0.0, None), transform="log", units="kg"),
            "rest_length": ParamSpec(units="m"),
        }

    def initial_state(self) -> dict:
        """Position and velocity, in the dtype :meth:`update` produces from
        this node's constants: float32 by default, float64 under
        ``jax_enable_x64`` for constants given as Python floats (which
        :meth:`params_pytree` places at float64 there), float32 again for
        constants given as float32 -- never narrower than float32.

        Seeded in float32 outright until 0.4.0 shipped, so under x64 the
        first update promoted the state to float64 and every graph scan --
        ``run_scan``, ``run_scan_with_history``, ``run_sweep``, and the
        parameter guide's system-identification recipe on this node --
        refused the carry (MADD-ANO-017's spring surface).  With x64 off
        nothing changes.
        """
        dtype = jnp.result_type(jnp.float32, *self.params_pytree().values())
        return {
            "position": jnp.array(self.params["initial_position"], dtype=dtype),
            "velocity": jnp.array(self.params["initial_velocity"], dtype=dtype),
        }

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Semi-implicit Euler integration of spring-damper dynamics.

        If ``anchor_position`` is not supplied the anchor defaults to the
        origin (0.0), so the node still produces sensible behaviour when
        tested in isolation.  ``params`` (injected by the graph) overrides
        the constants in ``self.params`` with traced, differentiable
        values.
        """
        p = self.params if params is None else {**self.params, **params}
        k = p["stiffness"]
        c = p["damping"]
        m = p["mass"]
        rest = p["rest_length"]

        position = state["position"]
        velocity = state["velocity"]

        anchor = boundary_inputs.get("anchor_position", jnp.array(0.0, dtype=jnp.float32))

        # Spring-damper force: F = -k*(x - anchor - rest) - c*v
        force = -k * (position - anchor - rest) - c * velocity

        # Semi-implicit Euler: update velocity first, then use new velocity
        acceleration = force / m
        velocity = velocity + acceleration * dt
        position = position + velocity * dt

        return {"position": position, "velocity": velocity}

    def derivatives(self, state, boundary_inputs, *, params=None):
        """dx/dt = v, dv/dt = F/m.

        Same constants as ``update`` (``{**self.params, **params}``): a
        calibrated stiffness drives ``integrate_node`` and the implicit
        solve as it drives the explicit step.
        """
        p = self.params if params is None else {**self.params, **params}
        k = p["stiffness"]
        c = p["damping"]
        m = p["mass"]
        rest = p["rest_length"]
        anchor = boundary_inputs.get(
            "anchor_position", jnp.array(0.0, dtype=jnp.float32)
        )
        force = -k * (state["position"] - anchor - rest) - c * state["velocity"]
        return {
            "position": state["velocity"],
            "velocity": force / m,
        }

    def implicit_residual(self, state_new, state_old, boundary_inputs, dt, *, params=None):
        """Backward Euler residual: x_new - x_old - dt * f(x_new)."""
        # Through the shared binder: empty ``params`` calls the two-argument
        # form (a ``derivatives`` override that predates the keyword keeps
        # working for every caller that passes none), and a non-empty one
        # for such an override is the documented ValueError naming the
        # class and the method -- not Python's TypeError from one call
        # deeper (MADD-ANO-018's refusal contract, one method in).
        derivs = _method_with_params(self, "derivatives", params)(
            state_new, boundary_inputs,
        )
        return {
            k: state_new[k] - state_old[k] - dt * derivs[k]
            for k in derivs
        }

    def boundary_input_spec(self):
        return {
            "anchor_position": BoundaryInputSpec(
                shape=(), description="Position of the other end of the spring",
                expected_units="m",
            ),
        }

    def boundary_flux_spec(self):
        return {
            "spring_force": BoundaryFluxSpec(
                shape=(), description="Spring-damper force",
                output_units="N",
            ),
        }

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        # Same constants as ``update``: a calibrated stiffness must change
        # the force this node delivers over a flux edge, not only its own
        # integration.
        p = self.params if params is None else {**self.params, **params}
        anchor = boundary_inputs.get(
            "anchor_position", jnp.array(0.0, dtype=jnp.float32)
        )
        k = p["stiffness"]
        c = p["damping"]
        rest = p["rest_length"]
        force = -k * (state["position"] - anchor - rest) - c * state["velocity"]
        return {"spring_force": force}


#: Growth per step at or below which the anchored-pair warning stays silent:
#: one part in a million.  It absorbs the rounding of constants that sit on
#: the limit ``c = k*dt`` (the defaults ``k = 100``, ``c = 1``,
#: ``dt = 0.01`` do, exactly), and a mode growing more slowly than that
#: needs a million steps to grow by ``e``.
_ANCHORED_PAIR_GROWTH_TOL = 1e-6


def _live_spring_constants(node, live):
    """``(k, c, m)`` a compile would use, or ``None`` if they cannot be judged.

    ``live`` is the node's entry of ``GraphManager.params["nodes"]``, whose
    leaves override the constructor's at compile time.  ``None`` when a
    value is not a concrete finite number (a traced leaf, say), or is
    outside the range the derivation assumes (``k > 0``, ``c >= 0``,
    ``m > 0``), so that the caller judges nothing rather than something
    wrong.
    """
    p = {**node.params, **(live or {})}
    try:
        k = float(p["stiffness"])
        c = float(p["damping"])
        m = float(p["mass"])
    except (TypeError, ValueError, KeyError):
        return None
    if not all(math.isfinite(v) for v in (k, c, m)):
        return None
    if k <= 0 or c < 0 or m <= 0:
        return None
    return k, c, m


def _anchored_pair_step_matrix(a, b, dt):
    """The converged step of two springs anchored on each other, or ``None``.

    ``a`` and ``b`` are ``(k, c, m)``.  At the converged fixed point each
    node's force reads its own position at the old time and its partner's
    at the new one, so the new velocities solve
    ``[[1, -alpha_a], [-alpha_b, 1]] v' = R s`` with
    ``alpha = k*dt**2/m``.  The step depends on the positions only through
    their difference, so the pair's common translation (an eigenvalue of
    exactly 1, and the reason the full 4x4 map has a Jordan block on the
    limit) is left out: the state is ``s = (x_a - x_b, v_a, v_b)``.

    ``None`` when ``alpha_a * alpha_b`` is 1 or more (or within a millionth
    of 1): Gauss-Seidel contracts by that product per pass, so the coupling
    iteration does not converge and the step that runs is not this one.  ``tests/nodes/
    test_spring_stability_limits.py`` checks the matrix against the node.
    """
    (ka, ca, ma), (kb, cb, mb) = a, b
    alpha_a = ka * dt * dt / ma
    alpha_b = kb * dt * dt / mb
    det = 1.0 - alpha_a * alpha_b
    if det <= _ANCHORED_PAIR_GROWTH_TOL:
        return None
    rhs = np.array([
        [-ka * dt / ma, 1.0 - ca * dt / ma, 0.0],
        [kb * dt / mb, 0.0, 1.0 - cb * dt / mb],
    ])
    v_new = np.array([[1.0, alpha_a], [alpha_b, 1.0]]) @ rhs / det
    d_new = np.array([1.0, 0.0, 0.0]) + dt * (v_new[0] - v_new[1])
    return np.vstack([d_new, v_new])


def _fmt(value):
    return f"{value:.4g}"


def _fmt_growth(value):
    """A growth factor, with the digits that tell it from 1: the warning
    fires from one part in a million."""
    return f"{value:.7g}"


def _anchored_pair_advisories(*, group, nodes, timesteps, edges, feeds,
                              live_params):
    """``WARNING:`` strings for two springs anchored on each other that grow.

    MADD-ANO-098.  When a coupling group converges an exchange in which
    one ``SpringDamperNode``'s ``position`` is the other's
    ``anchor_position`` and the other way round, each node is explicit in
    its own position and implicit in its partner's.  The spring forces on
    the two then do not cancel, and the pair's momentum is not conserved.
    For an equal pair (the same ``k``, ``c`` and ``m``) the centre-of-mass
    velocity is multiplied by ``g = (m - c*dt)/(m - k*dt**2)`` every
    step.  Where the coupling iteration converges (``k*dt**2 < m``) that
    is stable exactly for ``k*dt <= c <= (2*m - k*dt**2)/dt``: below
    ``c = k*dt`` the pair drifts off together with a growing speed, and
    above the upper figure, where each spring's damping relaxes its
    velocity within a step, the common velocity flips sign every step and
    grows.  Every pair past its own single-node limit
    ``k*dt**2 + 2*c*dt < 4*m`` is past the upper figure too.  This finds
    the pattern in the edges and warns when the converged step grows.  It
    warns and never refuses.

    **What counts as the pattern.**  Detection is conservative: it would
    rather miss a pair than warn about one that is not there.  Two
    ``SpringDamperNode``\\ s ``a`` and ``b`` in the group, whose class
    keeps ``SpringDamperNode.update``, qualify when there are:

    * an edge inside the group from ``a.position`` into
      ``b.anchor_position``, and
    * the reverse edge, from ``b.position`` into ``a.anchor_position``,

    and every one of these holds:

    * neither edge has a transform, other than the built-in ``identity``;
    * neither edge is additive or mapped;
    * nothing else, whether an edge or an external input, writes to either
      anchor;
    * the two nodes have the same timestep;
    * the group's ``max_iterations`` is at least 2.

    The rest lengths do not matter.  They add a constant force, which
    leaves the growth factor unchanged; a pair whose rest lengths are not
    opposite is pushed one way as well.

    **What is judged.**  The spectral radius of the converged step (see
    :func:`_anchored_pair_step_matrix`), from the live ``gm.params``
    values a compile uses.  It warns when that exceeds
    ``1 + _ANCHORED_PAIR_GROWTH_TOL``.  For an equal pair the growing mode
    is the centre of mass and the warning gives ``g`` in closed form; for
    an unequal pair, which is what a calibration of either node leaves
    behind, it gives the spectral radius.

    **What it does not see**, on purpose or by construction:

    * an edge through any other transform, including a lambda that passes
      the value through unchanged;
    * a one-way anchor (the pair is then a leader and a follower, each
      stable inside the single-node limit);
    * a ``SpringDamperNode`` wrapped in another node, or a subclass that
      overrides ``update``;
    * a group with ``max_iterations=1``.  That is a single staggered pass,
      the lagged exchange, which is a different scheme (MADD-ANO-098 notes
      it diverges at the reproducer's constants too);
    * two springs of different timesteps in a sub-cycled group;
    * ``k*dt**2 >= m`` on an equal pair (``alpha_a * alpha_b >= 1`` in
      general).  The coupling iteration does not converge there, so the
      converged step is not the one that runs: without acceleration the
      state leaves float range, and ``coupling_diagnostics()`` reports
      ``converged=False``;
    * a stiffness or mass that is not positive, a negative damping, a
      value that is not a concrete number at compile time, and anything
      supplied after compile: a ``dt`` passed at run time, or a parameter
      injected into a run.

    A cap of a few coupling passes stops the exchange short of
    convergence, and a loose tolerance does too.  The warning judges the
    converged step and does not model either.
    """
    from maddening.core.transforms import identity

    if getattr(group, "max_iterations", 2) <= 1:
        return []
    springs = {
        name: node for name, node in nodes.items()
        if isinstance(node, SpringDamperNode)
        and type(node).update is SpringDamperNode.update
    }
    if len(springs) < 2:
        return []

    # target spring -> the spring whose position is its anchor
    anchored_on = {}
    for e in edges:
        if (
            e.source_node in springs and e.target_node in springs
            and e.source_node != e.target_node
            and e.source_field == "position"
            and e.target_field == "anchor_position"
            and (e.transform is None or e.transform is identity)
            and not e.additive and e.mapping is None
            and feeds.get((e.target_node, e.target_field), 0) == 1
        ):
            anchored_on[e.target_node] = e.source_node
    pairs = sorted(
        (a, b) for a, b in anchored_on.items()
        if anchored_on.get(b) == a and a < b
    )

    issues = []
    for a, b in pairs:
        try:
            dt = float(timesteps[a])
            dt_b = float(timesteps[b])
        except (TypeError, ValueError, KeyError):
            continue
        if dt != dt_b or not math.isfinite(dt) or dt <= 0:
            continue
        consts_a = _live_spring_constants(springs[a], live_params.get(a))
        consts_b = _live_spring_constants(springs[b], live_params.get(b))
        if consts_a is None or consts_b is None:
            continue
        step = _anchored_pair_step_matrix(consts_a, consts_b, dt)
        if step is None:
            continue
        rho = float(np.max(np.abs(np.linalg.eigvals(step))))
        if not rho > 1.0 + _ANCHORED_PAIR_GROWTH_TOL:
            continue
        issues.append(
            _anchored_pair_message(group, a, b, consts_a, consts_b, dt, rho))
    return issues


def _anchored_pair_message(group, a, b, consts_a, consts_b, dt, rho):
    (ka, ca, ma), (kb, cb, mb) = consts_a, consts_b
    equal = all(
        math.isclose(x, y, rel_tol=1e-6) for x, y in zip(consts_a, consts_b)
    )
    head = (
        f"WARNING: SpringDamperNodes {a!r} and {b!r} are anchored on each "
        f"other in the coupling group {sorted(group.nodes)}: each one's "
        f"position is the other's anchor_position, and the group converges "
        f"that exchange within the step.  Each node is then explicit in its "
        f"own position and implicit in its partner's, and the step that "
        f"results has a stability limit of its own, tighter than either "
        f"node's.  "
    )
    if equal:
        k, c, m = ka, ca, ma
        g = (m - c * dt) / (m - k * dt * dt)
        lo, hi = k * dt, (2.0 * m - k * dt * dt) / dt
        if g > 1.0:
            why = (
                f"since c < k*dt ({_fmt(c)} < {_fmt(lo)}): the pair drifts "
                f"off together with a speed that grows every step, while "
                f"the spring between them can look right"
            )
        elif g < -1.0:
            why = (
                f"since c*dt + k*dt**2 > 2*m ({_fmt(c * dt + k * dt * dt)} "
                f"> {_fmt(2.0 * m)}): the pair's common velocity flips sign "
                f"and grows every step"
            )
        else:
            why = None
        if why is not None and math.isclose(abs(g), rho, rel_tol=1e-6):
            growth = (
                f"Both have k = {_fmt(k)}, c = {_fmt(c)} and m = {_fmt(m)}, "
                f"with dt = {_fmt(dt)}.  The spring forces on the two do not "
                f"cancel, and the converged step multiplies the pair's "
                f"centre-of-mass velocity by g = (m - c*dt)/(m - k*dt**2) = "
                f"{_fmt_growth(g)} every step, {why} (MADD-ANO-098).  "
            )
        else:
            growth = (
                f"Both have k = {_fmt(k)}, c = {_fmt(c)} and m = {_fmt(m)}, "
                f"with dt = {_fmt(dt)}, and the converged step grows by "
                f"g = {_fmt_growth(rho)} per step, its spectral radius "
                f"(MADD-ANO-098).  "
            )
        remedy = (
            f"Such a pair is stable for k*dt <= c <= (2*m - k*dt**2)/dt, "
            f"here {_fmt(lo)} <= c <= {_fmt(hi)}.  Use a smaller timestep, "
            f"which widens that range, or keep c >= k*dt without passing "
            f"the upper figure."
        )
    else:
        growth = (
            f"{a!r} has k = {_fmt(ka)}, c = {_fmt(ca)} and m = {_fmt(ma)}, "
            f"{b!r} has k = {_fmt(kb)}, c = {_fmt(cb)} and m = {_fmt(mb)}, "
            f"with dt = {_fmt(dt)}, and the converged step grows by "
            f"g = {_fmt_growth(rho)} per step, its spectral radius once the "
            f"pair's common translation is left out, so the pair moves off "
            f"without bound while the spring between them can look right "
            f"(MADD-ANO-098).  "
        )
        remedy = (
            f"An equal pair (the same k, c and m) is stable for "
            f"k*dt <= c <= (2*m - k*dt**2)/dt.  Use a smaller timestep, or "
            f"keep each spring's damping c >= k*dt ({_fmt(ka * dt)} for "
            f"{a!r}, {_fmt(kb * dt)} for {b!r}) without passing that upper "
            f"figure."
        )
    return head + growth + remedy
