#!/usr/bin/env python3
"""Export a graph as an FMI 3.0 FMU: describe it, package it, validate it, run it.

MADDENING's FMU is a thin C wrapper that forwards every FMI call over a
loopback TCP connection to a Python *sidecar* holding the compiled graph
(``docs/user_guide/fmu_export.md`` has the architecture and the wire
protocol).  This example builds one, checks it, and runs it in a master:

1. :func:`~maddening.fmi.build_model_description` -- outputs
   ``<node>.<field>``, the external input ``spring.anchor_position``, and
   one tunable parameter ``<node>.params.<key>`` for each constant the
   compiled step reads, with its ``ParamSpec`` bounds as ``min`` /
   ``max``.  A constant the step cannot read is *not* exported: here the
   spring's initial conditions, which only ``initial_state()`` reads, are
   listed in ``md.fixed_parameters`` with the reason.
2. :func:`~maddening.fmi.build_fmu_binary` compiles the wrapper (a C
   compiler is needed; libc only) and :func:`~maddening.fmi.write_fmu`
   packages ``modelDescription.xml`` and the binary into a ``.fmu`` in a
   temporary directory.
3. FMPy's ``validate_fmu`` checks the package against the FMI 3.0 schema
   and reports no problems.
4. The sidecar, driven in-process: a parameter set through it changes the
   next steps exactly as ``params=`` does on the graph, and a value outside
   its bounds, or a new value for a fixed parameter, is refused.
5. A multi-rate graph described with ``multi_clock=True``: one FMI 3.0
   ``<Clock>`` per node timestep, and every output and input tagged with
   its node's clock, so an importer knows that the spring, on a five times
   coarser step, changes only every fifth communication step.  The same
   graph described without it has no clock at all.
6. The two packaged FMUs, run in a master: FMPy's ``simulate_fmu`` loads
   each ``.fmu``, whose binary connects to an :class:`~maddening.fmi.FmuTcpBridge`
   serving the graph on ``127.0.0.1`` (a port the OS chooses).  Both were
   packaged without an endpoint, so the binary reads it from
   ``MADDENING_FMU_ENDPOINT``.  A parameter set as a start value and a
   driven input give the graph's own ``run_scan`` result, and the
   multi-clock FMU's spring output holds for five steps at a time.

Each stage that needs something optional -- a C compiler for 2 and 6,
FMPy for 3 and 6 -- is skipped with the reason when it is missing, and
the rest still runs.  Nothing is launched besides the compiler, and the
only socket is the bridge's, on the loopback interface.  MADDENING has
no FMU importer of its own, and FMPy is not one of its runtime
dependencies: ``pip install fmpy`` to run stages 3 and 6.

Usage
-----
    python -m maddening.examples.advanced.fmu_export_demo
    python -m maddening.examples.advanced.fmu_export_demo --steps 5
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.fmi import (
    MODEL_IDENTIFIER, FmuTcpBridge, build_fmu_binary, build_model_description, write_fmu,
)
from maddening.fmi.package import find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

DT = 0.01
#: The multi-rate graph's spring steps once for every five graph steps.
SLOW_DT = 5 * DT
#: Where the FMU binary looks for the bridge when the package names none.
ENDPOINT_VARIABLE = "MADDENING_FMU_ENDPOINT"


def build() -> GraphManager:
    """A spring-damper whose anchor is an external input."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def build_multirate() -> GraphManager:
    """A ball bouncing on a table every 0.01 s, and a spring stepped every 0.05 s."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=DT))
    gm.add_node(BallNode(name="ball", timestep=DT, initial_position=1.0))
    gm.add_node(SpringDamperNode(name="spring", timestep=SLOW_DT, stiffness=30.0,
                                 damping=2.0, initial_position=0.5))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def sidecar_for(gm: GraphManager, md) -> FmuSidecar:
    """The sidecar the FMU user guide wires up: the graph's compiled step and
    seed state, completing a step's inputs the way ``GraphManager.step``
    does.  ``_compiled_step``, ``_state`` and ``_resolve_external_inputs`` are
    GraphManager internals, read here exactly as that guide reads them."""
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs,
    ))


def simulate_in_fmpy(fmu: Path, gm: GraphManager, md, **kwargs):
    """Serve *gm* on a loopback port and run *fmu* in FMPy's ``simulate_fmu``.

    Returns FMPy's result and how many requests the bridge answered.
    """
    from fmpy import simulate_fmu

    with FmuTcpBridge(sidecar_for(gm, md), md, master_dt=gm.timestep) as bridge:
        previous = os.environ.get(ENDPOINT_VARIABLE)
        os.environ[ENDPOINT_VARIABLE] = bridge.endpoint
        try:
            result = simulate_fmu(str(fmu), **kwargs)
        finally:
            if previous is None:
                del os.environ[ENDPOINT_VARIABLE]
            else:
                os.environ[ENDPOINT_VARIABLE] = previous
        return result, bridge.requests_served


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps the sidecar and the FMPy runs take (default 50)")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="maddening_fmu_demo_") as tmp:
        run(args, Path(tmp))
    print()
    print("All checks passed.")
    return 0


def run(args, tmp: Path) -> None:
    gm = build()

    section("1. The model description")
    md = build_model_description(gm, model_name="Spring",
                                 model_identifier=MODEL_IDENTIFIER)
    by_name = {v.name: v for v in md.variables}
    for causality in ("output", "input", "parameter"):
        names = [v for v in md.variables if v.causality == causality]
        print(f"  {causality}s ({len(names)}):")
        for v in names:
            extra = ""
            if causality == "parameter":
                extra = f"  min={v.min}  max={v.max}" if (v.min, v.max) != (None, None) else ""
            print(f"    {v.name:28s} {v.dtype:8s} {v.variability}{extra}")
    print("  not exported (md.fixed_parameters):")
    for name, reason in md.fixed_parameters.items():
        print(f"    {name}: {reason.split(';')[0][:110]}")
    assert by_name["spring.anchor_position"].causality == "input"
    assert by_name["spring.position"].causality == "output"
    stiffness, damping = by_name["spring.params.stiffness"], by_name["spring.params.damping"]
    assert stiffness.causality == damping.causality == "parameter"
    # ParamSpec bounds as min / max: damping >= 0; stiffness has a log
    # transform, strictly above 0, so its min is the smallest positive float32.
    assert damping.min == 0.0 and 0.0 < stiffness.min < 1e-30
    assert "spring.params.initial_position" in md.fixed_parameters
    assert "spring.params.initial_position" not in by_name

    section("2. Package the FMU")
    cc = find_c_compiler()
    fmu = tmp / "Spring.fmu"
    if cc is None:
        print("  skipped: no C compiler found (set $CC or install gcc/clang); "
              "writing a description-only FMU instead")
        binary = None
        write_fmu(md, fmu)
    else:
        binary = build_fmu_binary(tmp / "build", cc=cc)
        print(f"  compiled the wrapper with {cc}: {binary.name}")
        # No endpoint: the wrapper reads MADDENING_FMU_ENDPOINT at run time
        # (or resources/endpoint.txt, written when one is passed here).
        write_fmu(md, fmu, binary=binary)
    with zipfile.ZipFile(fmu) as zf:
        members = zf.namelist()
    print(f"  wrote {fmu.name} ({fmu.stat().st_size} bytes) into a temporary directory:")
    for name in members:
        print(f"    {name}")
    assert "modelDescription.xml" in members
    assert (cc is None) or any(n.startswith("binaries/") for n in members)

    section("3. Validate it with FMPy")
    try:
        from fmpy import read_model_description
        from fmpy.validation import validate_fmu
    except ImportError:
        print("  skipped: FMPy is not installed (pip install fmpy)")
    else:
        problems = validate_fmu(str(fmu))
        description = read_model_description(str(fmu))
        print(f"  validate_fmu: {len(problems)} problems; FMPy reads FMI "
              f"{description.fmiVersion}, {len(description.modelVariables)} variables, "
              f"co-simulation: {description.coSimulation is not None}")
        assert problems == [], problems
        assert len(description.modelVariables) == len(md.variables)

    section("4. The sidecar, driven in-process")
    sidecar = sidecar_for(gm, md)
    sidecar.set_params({"spring.params.stiffness": 45.0})
    anchor = {"spring": {"anchor_position": jnp.float32(0.25)}}
    for _ in range(args.steps):
        state = sidecar.step(anchor)
    got = float(state["spring"]["position"])

    reference = build()
    p = reference.params
    p["nodes"]["spring"]["stiffness"] = jnp.float32(45.0)
    want = float(reference.run_scan(args.steps, external_inputs=anchor, params=p)
                 ["spring"]["position"])
    print(f"  stiffness set to 45 through the sidecar, anchor at 0.25, {args.steps} steps:")
    print(f"    sidecar spring.position {got:.6f}; graph run_scan with params= {want:.6f}")
    assert np.isclose(got, want, rtol=1e-6, atol=1e-7), (got, want)

    for update, why in (({"spring.params.stiffness": -1.0}, "below its bound"),
                        ({"spring.params.initial_position": 2.0}, "a fixed parameter")):
        try:
            sidecar.set_params(update)
        except (ValueError, KeyError) as exc:
            print(f"  refused, {why}: {str(exc).splitlines()[0][:120]}")
        else:
            raise AssertionError(f"the sidecar accepted {update}")
    assert float(sidecar.params["nodes"]["spring"]["stiffness"]) == 45.0

    section("5. A multi-rate graph: one clock per rate")
    multirate = build_multirate()
    md_clocked = build_model_description(multirate, model_name="MultiRate",
                                         model_identifier=MODEL_IDENTIFIER,
                                         multi_clock=True)
    clocks = md_clocked.clocks()
    clock_name = {c.value_reference: c.name for c in clocks}
    print(f"  graph step {md_clocked.graph_timestep} s; {len(clocks)} clocks:")
    for c in clocks:
        print(f"    {c.name}: every {c.interval_decimal} s -- {c.description}")
    print("  outputs and inputs, with their clocks:")
    for v in md_clocked.variables:
        if v.causality in ("output", "input") and not v.is_clock:
            print(f"    {v.name:24s} {v.causality:7s} {v.variability:9s} "
                  f"{', '.join(clock_name[r] for r in v.clocks)}")
    clocked = {v.name: v for v in md_clocked.variables}
    assert [c.interval_decimal for c in clocks] == [DT, SLOW_DT]
    assert clocked["spring.position"].clocks == (clocks[1].value_reference,)
    assert clocked["ball.position"].clocks == (clocks[0].value_reference,)
    assert clocked["spring.anchor_position"].clocks == (clocks[1].value_reference,)
    assert clocked["spring.position"].variability == "discrete"
    single = build_model_description(multirate, model_name="MultiRate",
                                     model_identifier=MODEL_IDENTIFIER)
    print(f"  without multi_clock=True: {len(single.clocks())} clocks, and "
          f"spring.position is {next(v.variability for v in single.variables if v.name == 'spring.position')}")
    assert single.clocks() == [] and "<Clock" not in single.to_xml()

    fmu_clocked = tmp / "MultiRate.fmu"
    write_fmu(md_clocked, fmu_clocked, binary=binary)
    print(f"  wrote {fmu_clocked.name} ({fmu_clocked.stat().st_size} bytes)"
          + ("" if binary else ", description only (no C compiler)"))

    section("6. Run the FMUs in a master (FMPy)")
    if binary is None:
        print("  skipped: no C compiler, so neither FMU has a binary to load")
        return
    if importlib.util.find_spec("fmpy") is None:
        print("  skipped: FMPy is not installed (pip install fmpy); any FMI 3.0 "
              "co-simulation master loads the same files")
        return
    from fmpy.validation import validate_fmu

    assert validate_fmu(str(fmu_clocked)) == [], "the multi-clock FMU is valid FMI 3.0"
    n = args.steps
    signal = np.array([(0.0, 0.25), (n * DT, 0.25)],
                      dtype=[("time", np.float64), ("spring.anchor_position", np.float64)])
    res, served = simulate_in_fmpy(
        fmu, gm, md, start_time=0.0, stop_time=n * DT, step_size=DT,
        output_interval=DT, start_values={"spring.params.stiffness": 45.0},
        input=signal, output=["spring.position", "spring.params.stiffness"])
    from_fmu = float(res["spring.position"][-1])
    print(f"  {fmu.name}: {len(res)} samples over {n * DT:.2f} s, stiffness "
          f"{res['spring.params.stiffness'][-1]:g} as a start value, anchor at 0.25 as an input")
    print(f"    FMPy spring.position {from_fmu:.6f}; graph run_scan with params= {want:.6f} "
          f"({served} requests over the loopback port)")
    assert res["spring.params.stiffness"][-1] == 45.0
    assert np.isclose(from_fmu, want, rtol=1e-5, atol=1e-6), (from_fmu, want)

    n_clocked = max(n, 2 * round(SLOW_DT / DT))
    res, served = simulate_in_fmpy(
        fmu_clocked, multirate, md_clocked, start_time=0.0, stop_time=n_clocked * DT,
        step_size=DT, output_interval=DT, output=["ball.position", "spring.position"])
    ball_changes = int(np.count_nonzero(np.diff(res["ball.position"])))
    spring_changes = int(np.count_nonzero(np.diff(res["spring.position"])))
    print(f"  {fmu_clocked.name}: {n_clocked} communication steps of {DT} s; "
          f"ball.position changed on {ball_changes}, spring.position on {spring_changes}")
    final = build_multirate().run_scan(n_clocked)
    for name in ("ball", "spring"):
        got, ref = float(res[f"{name}.position"][-1]), float(final[name]["position"])
        print(f"    {name}.position: FMPy {got:.6f}; graph run_scan {ref:.6f}")
        assert np.isclose(got, ref, rtol=1e-5, atol=1e-6), (name, got, ref)
    # The spring updates on the first graph step and then on every fifth.
    assert 1 <= spring_changes <= -(-n_clocked // round(SLOW_DT / DT)) < ball_changes


if __name__ == "__main__":
    sys.exit(main())
