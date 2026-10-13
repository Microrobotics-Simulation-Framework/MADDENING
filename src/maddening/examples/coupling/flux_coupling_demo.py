"""
Flux Coupling Demo
==================

Demonstrates these coupling features:

1. **Correct DD coupling** on heat rods (interior cell interface)
2. **Flux conservation monitoring** verifying energy balance
3. **Additive inputs** (multiple springs on a rigid body)
4. **IQN-IMVJ** vs IQN-ILS acceleration comparison
5. **Interface residual norm** as convergence criterion
6. **Subcycling interpolation** modes (constant / linear / quadratic)
7. **Repeated sweeps** (``waveform_iterations``) of a subcycled group

Sections 6 and 7 show what the sub-cycling options do today, which is
less than their names say (MADD-ANO-027): ``waveform_iterations`` re-runs
the same fixed-point solve rather than relaxing a boundary waveform, and
``boundary_interpolation="quadratic"`` is exactly ``"linear"``.

Sections 4-7 couple two ``SpringDamperNode`` nodes anchored to each other.
``SpringDamperNode`` pulls its end towards ``anchor + rest_length``, so
the two ends get rest lengths +1 and -1: one spring between two masses,
with a rest state.  (With the same sign at both ends there is none and
the pair drifts away together.)

Run with::

    JAX_PLATFORMS=cpu python -m maddening.examples.coupling.flux_coupling_demo
    JAX_PLATFORMS=cpu python -m maddening.examples.coupling.flux_coupling_demo --sections 1,3
"""

import argparse
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.coupling.helpers import (
    add_symmetric_value_coupling,
    check_conservation,
)
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.rigid_body import RigidBodyNode


# ======================================================================
# 1. Correct DD coupling of heat rods
# ======================================================================

def demo_dd_heat_coupling():
    """Two heat rods coupled via Dirichlet-Dirichlet value exchange.

    Key point: use INTERIOR cells (T[-2], T[1]) as the interface
    values, not the end cells (T[-1], T[0]).  Feeding an end cell
    straight back as the neighbour's Dirichlet datum couples each rod
    to its own boundary layer, which used to be a hard "cold lock"
    when HeatNode overwrote its end cells with the datum, and is still
    the wrong thing to exchange now that it does not (MADD-ANO-007):
    the end cell is the one the boundary closure has just acted on.
    Using interior cells gives proper energy-conserving coupling.
    """
    print("=" * 60)
    print("1. Correct DD heat coupling (interior cell interface)")
    print("=" * 60)

    dt = 0.001
    n_cells = 20
    n_steps = 500

    gm = GraphManager()
    gm.add_node(HeatNode("rod_a", dt, n_cells=n_cells,
                          thermal_diffusivity=0.01,
                          initial_temperature=100.0))
    gm.add_node(HeatNode("rod_b", dt, n_cells=n_cells,
                          thermal_diffusivity=0.01,
                          initial_temperature=0.0))
    # Interior cells as interface values (see the docstring)
    add_symmetric_value_coupling(
        gm, "rod_a", "rod_b",
        field_a="temperature", input_a="right_temperature",
        field_b="temperature", input_b="left_temperature",
        transform_a_to_b=lambda T: T[-2],
        transform_b_to_a=lambda T: T[1],
    )
    gm.add_coupling_group(
        ["rod_a", "rod_b"], max_iterations=20, tolerance=1e-10,
    )
    gm.compile()
    state = gm.run_scan(n_steps)

    T_a = state["rod_a"]["temperature"]
    T_b = state["rod_b"]["temperature"]
    total_energy = float(jnp.sum(T_a) + jnp.sum(T_b))
    expected = n_cells * 100.0 + n_cells * 0.0  # initial total

    print(f"  T_a mean: {float(jnp.mean(T_a)):.2f} K")
    print(f"  T_b mean: {float(jnp.mean(T_b)):.2f} K  (heat transferred!)")
    print(f"  Total energy: {total_energy:.1f}  (expected {expected:.1f})")
    assert abs(total_energy - expected) < 1.0, "Energy not conserved!"
    print("  Energy conservation: PASS")


# ======================================================================
# 2. Flux conservation monitoring
# ======================================================================

def demo_conservation():
    """Verify that heat flux is continuous at the coupling interface.

    With interior-cell DD coupling and matched grid/material, the
    two rods are mirror images across the interface, so the rod-end
    fluxes they report should agree.

    Since 0.4.0 ``compute_boundary_fluxes`` reconstructs the flux **at
    the rod end** (``x = 0`` / ``x = L``) rather than between the first
    two cell centres, so what is compared here is the flux at two
    points that are genuinely one cell apart in this overlapping
    decomposition.  The old reading made the imbalance identically
    zero by symmetry -- the two rods' interior differences were the
    same subtraction -- which flattered the check; the residual now
    printed is float32 round-off, and the units are K*m/s, not W/m^2
    (see :meth:`HeatNode.boundary_flux_spec`).
    """
    print()
    print("=" * 60)
    print("2. Flux conservation monitoring")
    print("=" * 60)

    dt = 0.001
    n_cells = 20

    gm = GraphManager()
    gm.add_node(HeatNode("rod_a", dt, n_cells=n_cells,
                          thermal_diffusivity=0.01,
                          initial_temperature=100.0))
    gm.add_node(HeatNode("rod_b", dt, n_cells=n_cells,
                          thermal_diffusivity=0.01,
                          initial_temperature=0.0))
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-2])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[1])
    gm.add_coupling_group(
        ["rod_a", "rod_b"], max_iterations=20, tolerance=1e-10,
    )
    gm.compile()
    state = gm.run_scan(500)

    # Compare fluxes at the interface — should match
    conservation = check_conservation(
        gm, state,
        [("rod_a", "right_heat_flux", "rod_b", "left_heat_flux")],
    )
    for name, imbalance in conservation.items():
        print(f"  {name}: imbalance = {imbalance:.2e}")
        assert abs(imbalance) < 0.01, "Flux not conserved!"
    print("  Flux conservation: PASS")


# ======================================================================
# 3. Additive inputs: multiple force sources on a rigid body
# ======================================================================

def demo_additive_inputs():
    """Two independent springs feed additive forces to a rigid body.

    Each spring oscillates independently (feedforward, no coupling
    group needed).  Both connect to the body's "force" input with
    additive=True, so the body receives their vector sum.  With
    symmetric initial conditions, the net force is zero and the
    body stays at the origin.
    """
    print()
    print("=" * 60)
    print("3. Additive inputs (two springs on rigid body)")
    print("=" * 60)

    dt = 0.001
    gm = GraphManager()

    # Rigid body at the origin, no gravity, held in the x-y plane
    gm.add_node(RigidBodyNode("body", dt, mass=1.0,
                              gravity=(0.0, 0.0, 0.0),
                              constraints={"z": 0.0, "rx": 0.0, "ry": 0.0}))

    # Two symmetric springs oscillating independently
    gm.add_node(SpringDamperNode("spring_r", dt, stiffness=10.0,
                                  damping=1.0, rest_length=0.0,
                                  initial_position=2.0))
    gm.add_node(SpringDamperNode("spring_l", dt, stiffness=10.0,
                                  damping=1.0, rest_length=0.0,
                                  initial_position=-2.0))

    # Both spring positions → body force (ADDITIVE)
    # Transform: scalar position → force vector [Fx, 0, 0]
    gm.add_edge("spring_r", "body", "position", "force",
                transform=lambda p: jnp.array([5.0 * p, 0.0, 0.0]),
                additive=True)
    gm.add_edge("spring_l", "body", "position", "force",
                transform=lambda p: jnp.array([5.0 * p, 0.0, 0.0]),
                additive=True)

    gm.compile()
    state = gm.run_scan(1000)

    body_x = float(state["body"]["position"][0])
    s_r = float(state["spring_r"]["position"])
    s_l = float(state["spring_l"]["position"])
    net_force = 5.0 * (s_r + s_l)

    print(f"  Spring R: {s_r:+.4f}")
    print(f"  Spring L: {s_l:+.4f}")
    print(f"  Net force (sum): {net_force:.6f}  (expected 0)")
    print(f"  Body x-position: {body_x:.6f}  (expected 0)")
    assert abs(body_x) < 0.01, f"Body drifted: {body_x}"
    print("  Symmetry check: PASS")


# ======================================================================
# 4. IQN-IMVJ vs IQN-ILS acceleration
# ======================================================================

def demo_imvj_vs_ils():
    """Compare IQN-ILS and IQN-IMVJ on the same coupled springs.

    IQN-IMVJ reuses the Jacobian approximation from previous
    timesteps, which should reduce iterations on later steps.
    """
    print()
    print("=" * 60)
    print("4. IQN-IMVJ vs IQN-ILS acceleration")
    print("=" * 60)

    dt = 0.01
    n_steps = 100

    def _build(accel, **kwargs):
        gm = GraphManager()
        # Damping 2 > stiffness * dt = 1, which the coupled explicit pair
        # needs to stay stable.
        gm.add_node(SpringDamperNode("sa", dt, stiffness=100.0,
                                      damping=2.0, rest_length=1.0,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode("sb", dt, stiffness=100.0,
                                      damping=2.0, rest_length=-1.0,
                                      initial_position=5.0))
        gm.add_edge("sa", "sb", "position", "anchor_position")
        gm.add_edge("sb", "sa", "position", "anchor_position")
        gm.add_coupling_group(
            ["sa", "sb"], max_iterations=15, tolerance=1e-10,
            acceleration=accel, **kwargs,
        )
        gm.compile()
        return gm

    gm_ils = _build("iqn-ils")
    gm_ils.run_scan(n_steps)
    d_ils = gm_ils.coupling_diagnostics()

    gm_imvj = _build("iqn-imvj", jacobian_reuse=5)
    gm_imvj.run_scan(n_steps)
    d_imvj = gm_imvj.coupling_diagnostics()

    # Both should reach the same answer
    pos_ils = float(gm_ils.get_node_state("sa")["position"])
    pos_imvj = float(gm_imvj.get_node_state("sa")["position"])

    print(f"  IQN-ILS  final iters: {d_ils['sa+sb']['iterations']}  "
          f"residual: {d_ils['sa+sb']['residual']:.2e}  "
          f"pos_a: {pos_ils:.6f}")
    print(f"  IQN-IMVJ final iters: {d_imvj['sa+sb']['iterations']}  "
          f"residual: {d_imvj['sa+sb']['residual']:.2e}  "
          f"pos_a: {pos_imvj:.6f}")
    print(f"  Position match: |delta| = {abs(pos_ils - pos_imvj):.2e}")


# ======================================================================
# 5. Interface residual norm
# ======================================================================

def demo_interface_norm():
    """Use convergence_norm='interface' to check only the coupling
    edge values rather than the full state."""
    print()
    print("=" * 60)
    print("5. Interface residual norm")
    print("=" * 60)

    dt = 0.01
    n_steps = 50

    def _build(norm, **kwargs):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("sa", dt, stiffness=50.0,
                                      rest_length=1.0,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode("sb", dt, stiffness=50.0,
                                      rest_length=-1.0,
                                      initial_position=3.0))
        gm.add_edge("sa", "sb", "position", "anchor_position")
        gm.add_edge("sb", "sa", "position", "anchor_position")
        gm.add_coupling_group(
            ["sa", "sb"], max_iterations=15,
            convergence_norm=norm, **kwargs,
        )
        gm.compile()
        return gm

    gm_l2 = _build("l2", tolerance=1e-8)
    gm_l2.run_scan(n_steps)
    d_l2 = gm_l2.coupling_diagnostics()

    # ``atol`` stays at its default of 0.0: a group that sets it is advised
    # on at compile() (a dead band can hide a change, MADD-ANO-254).
    gm_iface = _build("interface", rtol=1e-8)
    gm_iface.run_scan(n_steps)
    d_iface = gm_iface.coupling_diagnostics()

    # Both should converge to the same physics
    pos_l2 = float(gm_l2.get_node_state("sa")["position"])
    pos_if = float(gm_iface.get_node_state("sa")["position"])

    print(f"  L2 norm:        iters={d_l2['sa+sb']['iterations']}  "
          f"pos_a={pos_l2:.6f}")
    print(f"  Interface norm: iters={d_iface['sa+sb']['iterations']}  "
          f"pos_a={pos_if:.6f}")
    print(f"  Position match: |delta| = {abs(pos_l2 - pos_if):.2e}")


# ======================================================================
# 6. Subcycling interpolation comparison
# ======================================================================

def demo_subcycling_interpolation():
    """Run the constant, linear and quadratic boundary interpolation modes
    of a subcycled group beside a uniform-rate reference.

    ``"quadratic"`` is never given its third value and is exactly
    ``"linear"``, and both ends of the interpolation are end-of-step
    estimates (MADD-ANO-027), so no mode is expected to be more accurate
    than another here.
    """
    print()
    print("=" * 60)
    print("6. Subcycling interpolation modes (constant / linear / quadratic)")
    print("=" * 60)

    dt_fast, dt_slow = 0.001, 0.005
    k, c, m = 50.0, 1.0, 1.0
    n_slow_steps = 100
    n_fast_steps = n_slow_steps * round(dt_slow / dt_fast)

    # Reference: both nodes at fast rate (no subcycling needed)
    gm_ref = GraphManager()
    gm_ref.add_node(SpringDamperNode("fast", dt_fast, stiffness=k,
                                      damping=c, mass=m,
                                      initial_position=0.0))
    gm_ref.add_node(SpringDamperNode("slow", dt_fast, stiffness=k,
                                      damping=c, mass=m, rest_length=-1.0,
                                      initial_position=3.0))
    gm_ref.add_edge("fast", "slow", "position", "anchor_position")
    gm_ref.add_edge("slow", "fast", "position", "anchor_position")
    gm_ref.add_coupling_group(["fast", "slow"],
                               max_iterations=20, tolerance=1e-10)
    gm_ref.compile()
    s_ref = gm_ref.run_scan(n_fast_steps)
    ref_pos = float(s_ref["fast"]["position"])

    results = {}
    for interp in ("constant", "linear", "quadratic"):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("fast", dt_fast, stiffness=k,
                                      damping=c, mass=m,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode("slow", dt_slow, stiffness=k,
                                      damping=c, mass=m, rest_length=-1.0,
                                      initial_position=3.0))
        gm.add_edge("fast", "slow", "position", "anchor_position")
        gm.add_edge("slow", "fast", "position", "anchor_position")
        gm.add_coupling_group(
            ["fast", "slow"], max_iterations=20, tolerance=1e-10,
            subcycling=True, boundary_interpolation=interp,
        )
        gm.compile()
        s = gm.run_scan(n_slow_steps)
        pos = float(s["fast"]["position"])
        err = abs(pos - ref_pos)
        results[interp] = (pos, err)
        print(f"  {interp:10s}: pos_fast = {pos:.6f}  "
              f"error vs ref = {err:.2e}")

    print(f"  {'reference':10s}: pos_fast = {ref_pos:.6f}")
    print("  ('quadratic' is exactly 'linear' (MADD-ANO-027), and 'linear'")
    print("   differs from 'constant' only by the in-pass change of a source")
    print("   scheduled before the sub-cycled node.  The error against the")
    print("   reference is the slow node's time discretisation, which no")
    print("   interpolation mode reduces.)")


# ======================================================================
# 7. Repeated sweeps (``waveform_iterations``) of a subcycled group
# ======================================================================

def demo_repeated_sweeps():
    """Run a subcycled group with ``waveform_iterations`` 1, 2 and 3.

    This is not waveform relaxation (MADD-ANO-027): each sweep re-solves
    the same fixed point from where the previous one stopped, with no
    boundary waveform over the sub-step window, so it cannot improve the
    boundary data.
    """
    print()
    print("=" * 60)
    print("7. Repeated sweeps (waveform_iterations) of a subcycled group")
    print("=" * 60)

    dt_fast, dt_slow = 0.001, 0.005
    k, c, m = 50.0, 1.0, 1.0
    n_steps = 100

    # Reference at fast rate
    gm_ref = GraphManager()
    gm_ref.add_node(SpringDamperNode("fast", dt_fast, stiffness=k,
                                      damping=c, mass=m,
                                      initial_position=0.0))
    gm_ref.add_node(SpringDamperNode("slow", dt_fast, stiffness=k,
                                      damping=c, mass=m, rest_length=-1.0,
                                      initial_position=3.0))
    gm_ref.add_edge("fast", "slow", "position", "anchor_position")
    gm_ref.add_edge("slow", "fast", "position", "anchor_position")
    gm_ref.add_coupling_group(["fast", "slow"],
                               max_iterations=20, tolerance=1e-10)
    gm_ref.compile()
    n_fast_total = n_steps * round(dt_slow / dt_fast)
    s_ref = gm_ref.run_scan(n_fast_total)
    ref_pos = float(s_ref["fast"]["position"])

    for wf_iters in (1, 2, 3):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("fast", dt_fast, stiffness=k,
                                      damping=c, mass=m,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode("slow", dt_slow, stiffness=k,
                                      damping=c, mass=m, rest_length=-1.0,
                                      initial_position=3.0))
        gm.add_edge("fast", "slow", "position", "anchor_position")
        gm.add_edge("slow", "fast", "position", "anchor_position")
        gm.add_coupling_group(
            ["fast", "slow"], max_iterations=20, tolerance=1e-10,
            subcycling=True, waveform_iterations=wf_iters,
        )
        gm.compile()
        s = gm.run_scan(n_steps)
        pos = float(s["fast"]["position"])
        err = abs(pos - ref_pos)
        print(f"  waveform_iterations={wf_iters}: "
              f"pos_fast = {pos:.6f}  error vs ref = {err:.2e}")

    print(f"  {'reference':>23s}: pos_fast = {ref_pos:.6f}")
    print("  (Each sweep re-solves the same fixed point (MADD-ANO-027).  After")
    print("   a converged first sweep a later one applies at least one more")
    print("   pass, moving the state by about one residual: nothing here,")
    print("   where tolerance=1e-10 leaves the pair stationary in float32.)")


# ======================================================================

SECTIONS = {
    1: demo_dd_heat_coupling,
    2: demo_conservation,
    3: demo_additive_inputs,
    4: demo_imvj_vs_ils,
    5: demo_interface_norm,
    6: demo_subcycling_interpolation,
    7: demo_repeated_sweeps,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Flux coupling demo")
    parser.add_argument(
        "--sections", default=",".join(str(k) for k in SECTIONS),
        help="Comma-separated section numbers to run (default: all, 1-7)",
    )
    args = parser.parse_args(argv)
    chosen = [int(x) for x in args.sections.split(",") if x.strip()]
    unknown = sorted(set(chosen) - set(SECTIONS))
    if unknown:
        parser.error(f"unknown section(s) {unknown}; choose from 1-7")
    for number in chosen:
        SECTIONS[number]()
    print()
    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
