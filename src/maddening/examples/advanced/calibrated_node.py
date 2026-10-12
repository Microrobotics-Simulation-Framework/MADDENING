#!/usr/bin/env python3
"""A node of your own, made calibratable: ``params=``, ``param_specs()``,
``static_data_deps()``, then checked and fitted.

``HeatedBody`` (about forty lines, below) is a lumped thermal mass: a body
of heat capacity ``C`` loses heat to the room through a conductance ``h``
and is warmed by a thermostat whose power curve ``P(T)`` is tabulated once,
in ``__init__``, from two of the node's constants::

    dT/dt = (P(T) - h (T - T_room)) / C

It shows the three things a node author writes so that ``maddening.sysid``
can calibrate the node:

1. **``update(..., *, params=None)``** reads every float constant from the
   injected ``params``, so the graph's parameter pytree reaches it, traced
   and differentiable.  Every constructor argument goes to
   ``super().__init__``: ``params`` is all a saved graph carries.
2. **``param_specs()``** says what an optimiser may do with each constant:
   ``h`` and ``C`` are positive (``bounds`` and a ``log`` transform); the
   heater's ``heater_power`` and ``setpoint`` are ``trainable=False``.
3. **``static_data_deps()``** says that the power table the step reads was
   computed from ``heater_power`` and ``setpoint``.  The table is baked into
   the compiled step as a constant, so a gradient cannot pass through it:
   with the declaration, ``compile()`` refuses a graph that makes either of
   them trainable, where without it a fit would move the constant while the
   step went on reading the old table.

Then, on a graph holding the node:

* ``gm.nodes_without_params()`` is empty: the node's constants are in
  ``gm.params``.  A node on the old three-argument ``update`` is listed;
* :func:`~maddening.testing.verify_node` passes (the battery's
  ``params_*`` checks are the ones that catch a constant read from
  ``self.params`` instead of the injected ``params``);
* making ``heater_power`` trainable makes ``compile()`` raise, naming the
  static and the parameter;
* :func:`~maddening.sysid.fit_lm` recovers ``h`` and ``C`` from a noisy
  temperature record, from a start 50% off, and each lands within four
  Cramér-Rao standard deviations of the value that made the data (the
  bound comes from :func:`~maddening.sysid.fim` with the noise level, so
  the tolerance is the one the data itself allows: under Gaussian noise
  an unbiased estimate misses by more than four of them with probability
  6e-5).

Needs the ``verify`` extra (Hypothesis) for the ``verify_node`` part.  See
``docs/developer_guide/node_authoring.md`` ("Where a constant belongs")
and ``docs/user_guide/parameters.md``.

Usage
-----
    python -m maddening.examples.advanced.calibrated_node
    python -m maddening.examples.advanced.calibrated_node --samples 120 --max-examples 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray
from maddening.sysid import fim, fit_lm


class HeatedBody(SimulationNode):
    """A lumped thermal mass, cooled by the room and warmed by a thermostat."""

    def __init__(self, name, timestep, conductance=2.0, heat_capacity=40.0,
                 room_temperature=20.0, heater_power=150.0, setpoint=60.0,
                 initial_temperature=20.0):
        # Every constant goes to the base class, which keeps it in
        # self.params: that is what a saved graph carries.
        super().__init__(name, timestep, conductance=conductance,
                         heat_capacity=heat_capacity,
                         room_temperature=room_temperature,
                         heater_power=heater_power, setpoint=setpoint,
                         initial_temperature=initial_temperature)
        # The thermostat's power curve [W] against temperature [deg C],
        # tabulated once from two of the constants above.
        self._grid = jnp.linspace(0.0, 120.0, 121)
        self._power = heater_power * jax.nn.sigmoid((setpoint - self._grid) / 2.0)

    def initial_state(self):
        return {"temperature": jnp.asarray(self.params["initial_temperature"], jnp.float32)}

    def param_specs(self):
        positive = ParamSpec(bounds=(0.0, None), transform="log")
        baked = ParamSpec(trainable=False, description="tabulated into power_table")
        return {**super().param_specs(),        # initial_* entries: frozen
                "conductance": positive, "heat_capacity": positive,
                "heater_power": baked, "setpoint": baked}

    @property
    def static_data(self):
        return {"temperature_grid": StaticArray(self._grid, replication="replicate"),
                "power_table": StaticArray(self._power, replication="replicate")}

    def static_data_deps(self):
        # The grid is built from literals and declares nothing.
        return {"power_table": ("heater_power", "setpoint")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        t = state["temperature"]
        heating = jnp.interp(t, self._grid, self._power)
        loss = p["conductance"] * (t - p["room_temperature"])
        return {"temperature": t + dt * (heating - loss) / p["heat_capacity"]}


class OldContractBody(SimulationNode):
    """The same cooling on the three-argument ``update``: no ``params``."""

    def __init__(self, name, timestep, conductance=2.0, heat_capacity=40.0):
        super().__init__(name, timestep, conductance=conductance,
                         heat_capacity=heat_capacity)

    def initial_state(self):
        return {"temperature": jnp.asarray(60.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        t = state["temperature"]
        k = self.params["conductance"] / self.params["heat_capacity"]
        return {"temperature": t - dt * k * (t - 20.0)}


TRUE = {"conductance": 2.0, "heat_capacity": 40.0}
START = {"conductance": 3.0, "heat_capacity": 60.0}   # 50% off, both
NOISE_STD = 0.2      # thermometer noise [deg C]
DT = 0.5             # [s]; the body's time constant C/h is 20 s


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def with_values(params: dict, **values) -> dict:
    """A copy of *params* with the body's leaves set to *values*."""
    out = jax.tree.map(lambda x: x, params)
    for key, value in values.items():
        out["nodes"]["body"][key] = jnp.asarray(value, dtype=jnp.float32)
    return out


def select(params: dict, *keys: str) -> dict:
    """A mask (params-shaped pytree of bools) selecting the body's *keys*."""
    mask = jax.tree.map(lambda _: False, params)
    for key in keys:
        mask["nodes"]["body"][key] = True
    return mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=240,
                        help="Recorded samples, one per 0.5 s step (default 240)")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Hypothesis examples per verify_node check (default 50)")
    args = parser.parse_args(argv)
    if args.samples < 80:
        # Under two time constants of the warm-up the record does not
        # separate h from C well enough for the bounds to be meaningful.
        parser.error("--samples must be at least 80")
    n = args.samples

    gm = GraphManager()
    gm.add_node(HeatedBody("body", DT, **TRUE))
    gm.compile()

    # -- 1. the node's constants are graph parameters ----------------------
    section("1. The node's constants are in gm.params")
    gm.print_params_table()
    print(f"\n  nodes_without_params() = {gm.nodes_without_params()}")
    assert gm.nodes_without_params() == []
    leaves = set(gm.params["nodes"]["body"])
    assert {"conductance", "heat_capacity", "heater_power", "setpoint"} <= leaves
    trainable = gm.trainable_mask()["nodes"]["body"]
    assert trainable["conductance"] and trainable["heat_capacity"]
    assert not trainable["heater_power"] and not trainable["setpoint"]

    old = GraphManager()
    old.add_node(OldContractBody("old", DT))
    old.compile()
    print(f"  ...and for a node on the three-argument update: "
          f"{old.nodes_without_params()}")
    assert old.nodes_without_params() == ["old"]
    assert "old" not in old.params["nodes"]
    print("  Its constants are baked into the trace: nothing can fit them.")

    # -- 2. verify_node ----------------------------------------------------
    section("2. verify_node on the node")
    from maddening.testing.verification import verify_node   # the [verify] extra

    results = verify_node(HeatedBody("probe", DT), {"temperature": (0.0, 120.0)},
                          max_examples=args.max_examples, derandomize=True)
    for name, result in results.items():
        print(f"  {name:24s} {result.status}")
    failed = [str(r) for r in results.values() if not r.passed]
    assert not failed, "\n".join(failed)
    assert results["params_effective"].status == "PASS"
    print("  Every check passes: each trainable constant moves the step through "
          "the injected params.")

    # -- 3. what static_data_deps buys -------------------------------------
    section("3. A trainable constant behind a baked table is refused")
    gm.set_param_spec("body", "heater_power", ParamSpec())   # trainable
    try:
        gm.compile()
    except ValueError as exc:
        print(f"  compile() refuses it: {str(exc)[:150]}...")
        assert "power_table" in str(exc) and "heater_power" in str(exc)
    else:
        raise AssertionError("a trainable dependency of a static was accepted")
    gm.set_param_spec("body", "heater_power",
                      ParamSpec(trainable=False, description="tabulated into power_table"))
    gm.compile()
    print("  Frozen again, the graph compiles.")

    # -- 4. calibrate h and C from a noisy record ---------------------------
    section("4. Recover h and C with fit_lm from a noisy record")
    # run_sweep runs a batch of simulations from the states it is given (a
    # batch of one here), takes params=, and writes nothing back to the graph.
    init = {"body": {k: v[None] for k, v in gm.get_node_state("body").items()}}

    def temperatures(params: dict) -> jnp.ndarray:
        _, history = gm.run_sweep(n, init, return_history=True, params=params)
        return history["body"]["temperature"][0]

    truth = gm.params
    rng = np.random.default_rng(0)
    measured = temperatures(truth) + jnp.asarray(
        rng.normal(0.0, NOISE_STD, n), dtype=jnp.float32)

    @jax.jit
    def residual(params: dict) -> jnp.ndarray:
        return temperatures(params) - measured

    print(f"Recorded {n} samples ({n * DT:.0f} s) of the temperature, "
          f"noise sigma {NOISE_STD} deg C; the body warms from 20 to "
          f"{float(measured[-1]):.1f} deg C.")
    start = with_values(truth, **START)
    mask = select(start, "conductance", "heat_capacity")
    lm = fit_lm(gm, residual, params=start, mask=mask, noise_std=NOISE_STD)
    print(lm)
    assert lm.excited_rank == 2 and lm.hold_declined is False

    bounds = fim(residual, lm.params, mask=mask, noise_std=NOISE_STD, scale=None)
    sigma = dict(zip(bounds.param_names, np.sqrt(np.asarray(bounds.crb))))
    for key in ("conductance", "heat_capacity"):
        s = float(sigma[f"['nodes']['body']['{key}']"])
        got, true = float(lm.params["nodes"]["body"][key]), TRUE[key]
        print(f"  {key:14s} start {START[key]:5.1f}  fitted {got:8.4f} +- {s:.4f} "
              f"(1 sigma), true {true}: {abs(got - true) / s:.1f} sigma away")
        assert math.isfinite(s) and abs(got - true) < 4 * s, (key, got, s)
    # The frozen constants come back exactly as they went in.
    for key in ("heater_power", "setpoint", "room_temperature"):
        assert float(lm.params["nodes"]["body"][key]) == float(start["nodes"]["body"][key])
    gm.params = lm.params
    print("\nAfter the fit (gm.params holds it, and a save of the graph carries it):")
    gm.print_params_table()

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
