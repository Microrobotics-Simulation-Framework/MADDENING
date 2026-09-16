"""``verify_node`` battery for the LBM nodes and the surrogate node.

Companion of ``test_builtin_nodes_verified.py`` for the heavier nodes:
same battery, same params-migration ledger (``MIGRATED``), small lattices
so the Hypothesis runs stay fast.

Sampling envelopes: distributions are drawn as small positive values so
the macroscopic density stays away from zero (velocity = momentum /
density) and the Mach number stays low; pressures around the lattice
reference ``rho * cs2 = 1/3``; body forces small.  The battery checks the
update's structure (finite, deterministic, jit-consistent, differentiable
in state and params, every trainable parameter effective), not the
physics of a random lattice state.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.nodes.lbm import LBMNode
from maddening.nodes.lbm_pipe import LBMPipeNode
from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.node import SurrogateNode
from maddening.testing.verification import verify_node

KW = dict(max_examples=30, derandomize=True)

F_BOUNDS = (0.02, 0.2)          # per-direction populations, rho ~ 0.4-3.8
P_BOUNDS = (0.3, 0.4)           # pressure = rho * cs2, rho ~ 1
FORCE_BOUNDS = (-1e-3, 1e-3)


class _AffineDirect(SurrogateArchitecture):
    """new_state = state * w["scale"] + w["bias"] — two float weights."""
    mode = "direct"

    def init_params(self, rng_key, state_spec, boundary_spec):
        return {"scale": jnp.array(0.9, jnp.float32),
                "bias": jnp.array(0.1, jnp.float32)}

    def forward(self, params, state, boundary_inputs, dt):
        return {k: v * params["scale"] + params["bias"] for k, v in state.items()}


def _surrogate():
    arch = _AffineDirect()
    return SurrogateNode(
        "sur", 0.01, architecture=arch,
        weights=arch.init_params(None, {"x": (3,)}, {}),
        state_spec={"x": (3,)}, boundary_spec={}, initial_values={"x": [1.0, 2.0, 3.0]},
    )


CASES = {
    "lbm_d2q9": dict(
        node=lambda: LBMNode("l2", 1.0, grid_shape=(8, 6), viscosity=0.1, lattice="D2Q9"),
        bounds={"f": F_BOUNDS, "wall_mask": (0.0, 1.0)},
        boundary_bounds={
            "inlet_pressure": P_BOUNDS, "outlet_pressure": P_BOUNDS,
            "body_force": FORCE_BOUNDS, "wall_mask_update": (0.0, 1.0),
        },
    ),
    "lbm_d3q19": dict(
        node=lambda: LBMNode("l3", 1.0, grid_shape=(5, 4, 4), viscosity=0.2),
        bounds={"f": F_BOUNDS, "wall_mask": (0.0, 1.0)},
        boundary_bounds={
            "inlet_pressure": P_BOUNDS, "outlet_pressure": P_BOUNDS,
            "body_force": FORCE_BOUNDS, "wall_mask_update": (0.0, 1.0),
        },
    ),
    "lbm_pipe": dict(
        node=lambda: LBMPipeNode("pipe", 1.0, nx=6, ny=5, nz=5, tau=0.8,
                                 propeller_x=2, propeller_strength=1e-3,
                                 gravity=-1e-4),
        bounds={"f": F_BOUNDS, "tracer_f": (0.0, 0.2)},
        boundary_bounds={},
    ),
    "lbm_pipe_multiphase": dict(
        node=lambda: LBMPipeNode("pipe2", 1.0, nx=6, ny=5, nz=5, tau=0.8,
                                 propeller_x=2, propeller_strength=1e-3,
                                 G=-5.0, rho_liquid=1.0, rho_gas=0.25,
                                 rho_0=1.0, fill_fraction=0.6),
        bounds={"f": F_BOUNDS, "tracer_f": (0.0, 0.2)},
        boundary_bounds={},
    ),
    "surrogate": dict(
        node=_surrogate,
        bounds={"x": (-10.0, 10.0)},
        boundary_bounds={},
    ),
}

# Nodes that take ``params``: the params checks must run, not SKIP.
MIGRATED = {"lbm_d2q9", "lbm_d3q19", "lbm_pipe", "lbm_pipe_multiphase", "surrogate"}


# D3Q19 dominates the wall-clock (~3 min on CPU): slow lane.
_PARAMS = [
    pytest.param(n, marks=pytest.mark.slow) if n == "lbm_d3q19" else n
    for n in sorted(CASES)
]


@pytest.mark.parametrize("name", _PARAMS)
def test_node_passes_battery(name):
    case = CASES[name]
    node = case["node"]()
    results = verify_node(
        node, case["bounds"], boundary_bounds=case["boundary_bounds"], **KW,
    )
    bad = [str(r) for r in results.values() if not r.passed]
    assert not bad, f"{name}:\n" + "\n".join(bad)
    if name in MIGRATED:
        assert node.accepts_params()
        for check in ("params_consistent", "params_gradient_finite", "params_effective"):
            assert not results[check].skipped, f"{name}: {check} skipped"
            assert results[check].n_examples > 0


def test_lbm_pipe_multiphase_constants_are_trainable_only_in_multiphase():
    single = LBMPipeNode("a", 1.0, nx=6, ny=5, nz=5)
    multi = LBMPipeNode("b", 1.0, nx=6, ny=5, nz=5, G=-5.0)
    for key in ("G", "rho_0", "rho_wall", "rho_liquid", "rho_gas"):
        assert single.param_specs()[key].trainable is False
        assert multi.param_specs()[key].trainable is True
    assert single.param_specs()["tau_tracer"].trainable is True
    assert multi.param_specs()["tau_tracer"].trainable is False
    for key in ("pipe_radius", "propeller_radius", "fill_fraction", "initial_velocity"):
        assert single.param_specs()[key].trainable is False


def test_surrogate_weights_are_flat_params_leaves():
    node = _surrogate()
    leaves = node.params_pytree()
    assert set(leaves) == {"weights['bias']", "weights['scale']"}
    state = node.initial_state()
    injected = dict(leaves)
    injected["weights['scale']"] = jnp.asarray(2.0, jnp.float32)
    out = node.update(state, {}, 0.01, params=injected)
    assert jnp.allclose(out["x"], state["x"] * 2.0 + 0.1)
    # Unmodified injection reproduces the baked weights exactly.
    same = node.update(state, {}, 0.01, params=leaves)
    assert jnp.array_equal(same["x"], node.update(state, {}, 0.01)["x"])
