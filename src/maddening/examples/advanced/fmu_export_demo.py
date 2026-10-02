#!/usr/bin/env python3
"""Export a graph as an FMI 3.0 FMU: describe it, package it, validate it.

MADDENING's FMU is a thin C wrapper that forwards every FMI call over a
loopback TCP connection to a Python *sidecar* holding the compiled graph
(``docs/user_guide/fmu_export.md`` has the architecture and the wire
protocol).  This example builds one and checks it, without starting any
simulation tool:

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

Each stage that needs something optional -- a C compiler for 2, FMPy for
3 -- is skipped with the reason when it is missing, and the rest still
runs.  Nothing is launched besides the compiler; no socket is opened.

Usage
-----
    python -m maddening.examples.advanced.fmu_export_demo
    python -m maddening.examples.advanced.fmu_export_demo --steps 5
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, build_fmu_binary, build_model_description, write_fmu
from maddening.fmi.package import find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.spring import SpringDamperNode

DT = 0.01


def build() -> GraphManager:
    """A spring-damper whose anchor is an external input."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=50,
                        help="Steps the in-process sidecar takes (default 50)")
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
    # The wiring the FMU user guide gives: the sidecar holds the graph's
    # compiled step and seed state.  ``_compiled_step`` and ``_state`` are
    # GraphManager internals, read here exactly as that guide reads them.
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        fixed_params=md.fixed_parameters,
    ))
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


if __name__ == "__main__":
    sys.exit(main())
