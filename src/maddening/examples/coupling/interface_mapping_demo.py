#!/usr/bin/env python
"""An edge between two grids: ``add_edge(mapping=...)``, saved and reloaded.

Two heat rods discretised differently -- a coarse 8-cell "heater" and a
fine 24-cell rod -- joined by one edge whose **interface mapping** turns
the heater's temperature field into a heat-source field on the rod's
grid.  The mapping is an RBF interpolant (thin-plate spline with linear
polynomial augmentation) built from each rod's own cell centres.

What it shows, and checks:

1. **The mapped edge.**  ``print_graph()`` names the mapping and the slot
   that holds its weights, ``gm.params["mappings"][<edge key>]["H"]`` (a
   traced input, like a node constant).  The weights pass the *patch
   test*: a constant field and a linear field come through exactly, to
   float32 round-off.  After one step, the rod's temperature is exactly
   ``dt * H @ T_heater``: the edge delivers the mapped field.
2. **Save and reload.**  ``gm.to_dict()`` writes the mapping's *recipe*
   (a ``MappingSpec``: kind, hyper-parameters and point references with
   their sha256), never the weights.  ``GraphManager.from_dict`` rebuilds
   the mapping by calling the same factory on the same points, so the
   rebuilt ``H`` is bitwise equal, and the reloaded graph steps exactly
   as the original does.
3. **The refusal when a mapped coordinate would move (MADD-ANO-063).**
   A uniform ``HeatNode`` derives its ``grid_x`` -- the points the
   mapping was built from -- from ``length``, and its own step reads
   ``length`` too.  Writing a new ``length`` into ``gm.params`` would move
   the rod while the mapping kept the old points' weights, so the next
   run refuses it with a ``ValueError`` naming the edge and the field, and
   so does ``to_dict()``.  Nothing was stepped; restoring the value makes
   the graph whole again.  To really change the geometry, rebuild the
   node and the mapped edge from the new points (done here, too).

One case is *not* refused, and is documented as MADD-ANO-022: an explicit
or traced ``params=`` pytree (a fit, ``run_scan(params=...)``) still
computes with the constructor's geometry, because no write is made for
anything to refuse.  Freeze such a parameter
(``ParamSpec(trainable=False)``) or give the mapping explicit
coordinates; see ``docs/user_guide/parameters.md``.

The older closure-based maps (``transform=`` from
``maddening.core.coupling.interface_mapping``) are compared in
``spatial_interpolation_demo.py``; prefer ``mapping=``, whose weights the
graph can differentiate, replace and serialise.

Usage
-----
    python -m maddening.examples.coupling.interface_mapping_demo
    python -m maddening.examples.coupling.interface_mapping_demo --steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

DT = 0.01
EDGE = "heater.temperature->rod.heat_source"
REGISTRY = {"HeatNode": HeatNode}


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def build(heater_length: float = 1.0) -> GraphManager:
    """The coarse heater mapped onto the fine rod.

    The point sets are each rod's ``grid_x`` (cell centres), passed both
    as arrays (what the factory computes with) and as references
    (``{"node": ..., "field": ...}``, what a saved config records).
    """
    heater = HeatNode("heater", DT, n_cells=8, length=heater_length,
                      thermal_diffusivity=0.005,
                      initial_temperature=np.linspace(1.0, 2.0, 8).tolist())
    rod = HeatNode("rod", DT, n_cells=24, length=1.0,
                   thermal_diffusivity=0.005, initial_temperature=0.0)
    gm = GraphManager()
    gm.add_node(heater)
    gm.add_node(rod)
    gm.add_edge("heater", "rod", "temperature", "heat_source", mapping=rbf_mapping(
        np.asarray(heater.static_data["grid_x"].value),
        np.asarray(rod.static_data["grid_x"].value),
        kernel="thin_plate_spline", mode="consistent",
        source_ref={"node": "heater", "field": "grid_x"},
        target_ref={"node": "rod", "field": "grid_x"},
    ))
    gm.compile()
    return gm


def states(gm: GraphManager) -> dict:
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()}
            for n in gm.node_names}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=200,
                        help="Steps run before and after the round trip (default 200)")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    section("1. A mapped edge between an 8-cell and a 24-cell rod")
    gm = build()
    gm.print_graph()

    H = np.asarray(gm.params["mappings"][EDGE]["H"], dtype=np.float64)
    x_heater = np.asarray(gm.get_node("heater").static_data["grid_x"].value, np.float64)
    x_rod = np.asarray(gm.get_node("rod").static_data["grid_x"].value, np.float64)
    const_err = np.abs(H @ np.ones(8) - 1.0).max()
    linear_err = np.abs(H @ x_heater - x_rod).max()
    print(f"\n  params['mappings'][{EDGE!r}]['H']: shape {H.shape} (target x source)")
    print(f"  patch test: constant field error {const_err:.1e}, "
          f"linear field error {linear_err:.1e}")
    assert H.shape == (24, 8)
    assert const_err < 1e-6 and linear_err < 1e-6

    gm.step()
    heater_t = np.asarray(gm.get_node_state("heater")["temperature"], np.float64)
    rod_t = np.asarray(gm.get_node_state("rod")["temperature"], np.float64)
    # The rod started uniform at 0, so diffusion adds nothing on step one:
    # its temperature is the delivered source times dt.  The edge reads the
    # heater's temperature after the heater's own update (schedule order).
    delivered_err = np.abs(rod_t - DT * H @ heater_t).max()
    print(f"  after one step, rod temperature vs dt * H @ T_heater: "
          f"max difference {delivered_err:.1e}")
    assert delivered_err < 1e-6 * np.abs(rod_t).max()
    gm.run(args.steps - 1)
    print(f"  after {args.steps} steps the rod's mean temperature is "
          f"{float(np.mean(gm.get_node_state('rod')['temperature'])):.4f}")

    # ------------------------------------------------------------------
    section("2. Save the graph, reload it, and step both")
    config = json.loads(json.dumps(gm.to_dict()))        # through real JSON
    spec = config["edges"][0]["mapping"]
    refs = {k: f"{v['node']}.{v['field']} (sha256 {v['sha256'][:12]}...)"
            for k, v in spec["points"].items()}
    print(f"  the config's mapping: kind={spec['kind']!r}, mode={spec['mode']!r}, "
          f"kernel={spec['kernel']!r}, shape={spec['shape']}")
    for name, ref in refs.items():
        print(f"    {name}: {ref}")
    assert "H" not in json.dumps(spec), "the config carries the recipe, not the weights"

    fresh = GraphManager.from_dict(config, REGISTRY)
    fresh.compile()
    H_fresh = np.asarray(fresh.params["mappings"][EDGE]["H"])
    bitwise = np.array_equal(H_fresh, np.asarray(gm.params["mappings"][EDGE]["H"]))
    print(f"  rebuilt H bitwise equal to the original: {bitwise}")
    assert bitwise

    for name, fields in states(gm).items():                 # same starting state
        fresh.set_node_state(name, {f: jnp.asarray(v) for f, v in fields.items()})
    gm.run(args.steps)
    fresh.run(args.steps)
    a, b = states(gm), states(fresh)
    identical = all(np.array_equal(a[n][f], b[n][f]) for n in a for f in a[n])
    print(f"  {args.steps} more steps on each: states identical: {identical}")
    assert identical

    # ------------------------------------------------------------------
    section("3. A write that would move the mapped points is refused")
    before = states(gm)
    gm.params["nodes"]["heater"]["length"] = jnp.float32(1.5)
    for label, call in (("gm.run(1)", lambda: gm.run(1)), ("gm.to_dict()", gm.to_dict)):
        try:
            call()
        except ValueError as exc:
            message = str(exc)
            print(f"  {label} refused: {message[:230]}...")
            assert EDGE in message and "heater.grid_x" in message
        else:
            raise AssertionError(f"{label} accepted a length the mapping cannot follow")
    unchanged = all(np.array_equal(before[n][f], v)
                    for n, fields in states(gm).items() for f, v in fields.items())
    print(f"  the state is untouched: {unchanged}")
    assert unchanged

    gm.params["nodes"]["heater"]["length"] = jnp.float32(1.0)      # drop the edit
    gm.run(1)
    print("  with the value restored the graph runs again.")

    # The supported way: rebuild the node, and the mapping from its new points.
    longer = build(heater_length=1.5)
    H_longer = np.asarray(longer.params["mappings"][EDGE]["H"], dtype=np.float64)
    x_longer = np.asarray(longer.get_node("heater").static_data["grid_x"].value, np.float64)
    print(f"  rebuilt at length 1.5: the heater's cells now span "
          f"[{x_longer[0]:.4f}, {x_longer[-1]:.4f}] and H was rebuilt from them "
          f"(linear patch error {np.abs(H_longer @ x_longer - x_rod).max():.1e}); "
          f"{len(json.dumps(longer.to_dict()))} bytes of config, saved without complaint.")
    assert not np.array_equal(H_longer, H)
    assert np.abs(H_longer @ x_longer - x_rod).max() < 1e-5

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
