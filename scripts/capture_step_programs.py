#!/usr/bin/env python3
"""Capture, or check, the compiled programs of graphs that use no geometry-dependent mapping.

Geometry-dependent mappings add a path to the code that resolves every
edge of every step.  The claim that has to survive it is "a graph that
uses none is unchanged", and the strongest form of that claim is about
the *program*, not about results: result bits depend on the CPU's
instruction set and on jaxlib, so a committed capture of results does not
travel to another machine, while the lowered program text depends only on
the jax version -- and identical text gives identical results anywhere.

For each of a fixed set of graphs (:func:`gate_graphs`) this lowers five
programs and records the sha256 of their StableHLO text:

``step``
    what ``GraphManager.step()`` compiles (parameters traced);
``step_baked_params``
    the pure step with ``params=None`` (parameters baked in as constants);
``scan3``
    what ``run_scan(3)`` compiles;
``sweep2``
    what ``run_sweep(3, <batch of 2>)`` compiles;
``adaptive``
    the adaptive steppers' step function (``_build_dt_step_fn``).

A program the library refuses to build (the adaptive step of a multi-rate
graph, say) is recorded as the refusal: its exception type and the digest
of its message.

The graphs: the four named coupled topologies at two configurations each;
the same four with ragged sparse mappings and one with a scatter-layout
sparse mapping; a multi-rate graph, a sub-cycled group, a group with a
predictor and a group under Aitken relaxation (its two-pass exit); a flux edge
outside any group; an uncoupled pair joined by an ``rbf`` mapping; and a
graph with a sharded node on two virtual CPU devices (which is why this script sets
``--xla_force_host_platform_device_count`` before it imports jax).

The gate is strict on purpose.  The text changes with any change to what
is traced -- a numerically neutral ``* 1.0``, two existing operations in
another order -- so a change that leaves it alone has, provably, changed
nothing for these graphs.

**Capturing.**  The capture must be taken on the commit the change starts
from, with a clean tree, once per jax version that is tested::

    python scripts/capture_step_programs.py --write

writes ``tests/core/data/step_program_digests.json``: the commit, and
under ``captures`` one entry per ``jax.__version__``.  Run it again under
another jax (``pip install --target <dir> jax==<v> jaxlib==<v>`` and put
``<dir>`` first on ``PYTHONPATH``) to add that version's entry; an entry
is only ever added to a file captured on the same commit.  After the
first commit that follows, the file is not edited again
(``tests/compliance/test_step_program_digests_are_frozen.py``).

**Checking.**  ``--check`` recomputes and compares, and exits non-zero on
a difference or on a jax version with no entry (it never passes for lack
of a capture).  ``tests/core/test_step_program_digests.py`` is the same
comparison under pytest, with the gate's own mutation tests.

``--print`` writes this process's digests as JSON; ``--only NAME`` (may
repeat) restricts any mode to some graphs.

The graphs are built by ``tests/property/coupled_topologies.py``: a change
there that alters what a relay traces changes these digests as surely as
a change to the library does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIGESTS = REPO_ROOT / "tests" / "core" / "data" / "step_program_digests.json"
PROGRAMS = ("step", "step_baked_params", "scan3", "sweep2", "adaptive")
HOST_FLAG = "--xla_force_host_platform_device_count"
#: Virtual CPU devices the sharded graph needs.
SHARDED_DEVICES = 2
SHARDED = "sharded"
#: The graph whose group exits on Aitken's two-pass rule.
AITKEN = "aitken/chain-into-ring"
FORMAT = 1


class _Captured(Exception):
    """Raised in place of running a compiled program: carries the program."""


# ---------------------------------------------------------------------------
# The gate graphs
# ---------------------------------------------------------------------------


def _ct():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests.property import coupled_topologies as ct  # noqa: PLC0415
    return ct


def _built(topo, knobs, **kw):
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ct().build(topo, knobs, **kw).gm


def _named(name: str, choice: int, **kw):
    ct = _ct()
    topo = ct.named_topologies()[name]
    return _built(topo, ct.topology_knobs(topo, choice), **kw)


def _multi_rate():
    """``chain-into-ring`` beside an isolated relay at half the step: a multi-rate graph."""
    import dataclasses  # noqa: PLC0415

    ct = _ct()
    topo = ct.named_topologies()["chain-into-ring"]
    tick = ct.TNode("tick", n=1, alpha=1.0, beta=1.0, timestep=0.5)
    topo = dataclasses.replace(topo, nodes=topo.nodes + (tick,))
    return _built(topo, ct.topology_knobs(topo, 0))


def _sub_cycled():
    """``chain-into-ring`` with the last member of its group at half the group's step."""
    ct = _ct()
    topo = ct.named_topologies()["chain-into-ring"]
    topo = topo.with_timesteps({members[-1]: 0.5 for members in topo.groups})
    knobs = [dict(g, subcycling=True, boundary_interpolation="constant")
             for g in ct.topology_knobs(topo, 0)]
    return _built(topo, knobs)


def _predictor():
    """``chain-into-ring`` with a quadratic predictor on its group."""
    ct = _ct()
    topo = ct.named_topologies()["chain-into-ring"]
    return _built(topo, [dict(g, predictor="quadratic") for g in ct.topology_knobs(topo, 0)])


def _aitken():
    """``chain-into-ring`` with Aitken relaxation on its group, under the
    default solver: the one acceleration that must meet the threshold on
    two consecutive passes before the loop exits."""
    ct = _ct()
    topo = ct.named_topologies()["chain-into-ring"]
    return _built(topo, [dict(acceleration="aitken", iteration_mode="gauss-seidel",
                              convergence_norm="l2", tolerance=1e-6, max_iterations=200)
                         for _ in topo.groups])


def _flux_outside_a_group():
    """An ungrouped ring whose forward edge carries a boundary flux."""
    ct = _ct()
    b = ct.TopologyBuilder()
    b.node("a", 1, alpha=0.5, flux=True)
    b.node("b", 1, alpha=0.25, beta=1.0)
    b.edge("a", "b", field="q")
    b.edge("b", "a")
    return _built(b.build("flux-outside-a-group"), [], node_order=("a", "b"))


def _rbf_edge():
    """Two uncoupled relays of different sizes joined by an ``rbf`` mapping."""
    import numpy as np  # noqa: PLC0415

    from maddening.core.coupling.mapping import rbf_mapping  # noqa: PLC0415
    from maddening.core.graph_manager import GraphManager  # noqa: PLC0415

    ct = _ct()
    gm = GraphManager()
    gm.add_node(ct.TRelay("a", 1.0, n=5, ports=0, alpha=0.5, beta=1.0))
    gm.add_node(ct.TRelay("b", 1.0, n=3, ports=1, alpha=0.25))
    gm.add_edge("a", "b", "x", "u0",
                mapping=rbf_mapping(np.linspace(0.0, 1.0, 5), np.linspace(0.1, 0.9, 3)))
    gm.compile()
    return gm


def _sharded():
    """A heat rod sharded over two devices."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: PLC0415
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode  # noqa: PLC0415
    from maddening.core.graph_manager import GraphManager  # noqa: PLC0415
    from maddening.nodes.heat import HeatNode  # noqa: PLC0415

    if len(jax.devices()) < SHARDED_DEVICES:
        raise RuntimeError(
            f"the sharded gate graph needs {SHARDED_DEVICES} devices and this process has "
            f"{len(jax.devices())}: set XLA_FLAGS={HOST_FLAG}={SHARDED_DEVICES} before jax "
            f"is imported (running this script does)")
    n_cells, length, alpha = 16, 1.0, 0.01
    dx = length / n_cells
    x = np.linspace(dx / 2, length - dx / 2, n_cells)
    rod = HeatNode("heat", timestep=0.25 * dx * dx / alpha, n_cells=n_cells, length=length,
                   thermal_diffusivity=alpha,
                   initial_temperature=np.sin(np.pi * x / length).astype(np.float32))
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(rod, create_device_mesh(shape=(SHARDED_DEVICES,)),
                                   axis_map={"devices": 0}))
    gm.compile()
    return gm


def has_sparse_mappings() -> bool:
    """Does this tree's topology harness build sparse mappings (tier 1)?"""
    import inspect  # noqa: PLC0415

    return "mapping_kind" in inspect.signature(_ct().build).parameters


def gate_graphs() -> dict:
    """``{name: builder}``: every gate graph of this tree, in a fixed order."""
    import functools  # noqa: PLC0415

    names = sorted(_ct().named_topologies())
    graphs: dict = {}
    for name in names:
        for choice in (0, 1):
            graphs[f"named/{name}/choice-{choice}"] = functools.partial(_named, name, choice)
    if has_sparse_mappings():
        for name in names:
            for choice in (0, 1):
                graphs[f"sparse-ragged/{name}/choice-{choice}"] = functools.partial(
                    _named, name, choice, mapping_kind="sparse-ragged")
        graphs["sparse-scatter/chain-into-ring/choice-0"] = functools.partial(
            _named, "chain-into-ring", 0, mapping_kind="sparse-scatter")
    graphs["multi-rate/chain-into-ring"] = _multi_rate
    graphs["sub-cycled/chain-into-ring"] = _sub_cycled
    graphs["predictor/chain-into-ring"] = _predictor
    graphs[AITKEN] = _aitken
    graphs["flux-outside-a-group"] = _flux_outside_a_group
    graphs["rbf-edge"] = _rbf_edge
    graphs[SHARDED] = _sharded
    return graphs


# ---------------------------------------------------------------------------
# Programs and digests
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _refusal(exc: BaseException) -> str:
    """A refused program, as a digest: the exception type and its message
    (object addresses removed, so the digest is the same in every process)."""
    import re  # noqa: PLC0415

    message = re.sub(r"0x[0-9a-fA-F]+", "0x", str(exc))
    return f"refused:{type(exc).__name__}:{_sha(message)[:16]}"


def _library_program(gm, entry: str, *args, **kwargs):
    """``(jitted function, arguments)`` of what ``gm.<entry>(...)`` would run.

    Every compiled entry point hands its program to
    ``_call_surfacing_strict``; this takes it from there, so the text
    lowered is the library's own program and not a restatement of it, and
    nothing is compiled or run.
    """
    def grab(fn, *call_args):
        raise _Captured(fn, call_args)

    gm._call_surfacing_strict = grab            # noqa: SLF001
    try:
        getattr(gm, entry)(*args, **kwargs)
    except _Captured as captured:
        return captured.args
    finally:
        del gm._call_surfacing_strict           # noqa: SLF001
    raise RuntimeError(f"GraphManager.{entry} did not reach _call_surfacing_strict")


def program_texts(gm) -> dict:
    """``{program: lowered StableHLO text, or the refusal}`` of a compiled graph."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    def lowered(make) -> str:
        try:
            fn, args = make()
            return fn.lower(*args).as_text()
        except _Captured:
            raise
        except Exception as exc:     # noqa: BLE001 -- a refusal is a recorded outcome
            return _refusal(exc)

    def baked():
        step = gm._raw_step_fn                               # noqa: SLF001
        ext = gm._resolve_external_inputs(None)              # noqa: SLF001
        return jax.jit(lambda state, e: step(state, e, None)), (gm._state, ext)  # noqa: SLF001

    def sweep():
        batch = {name: {f: jnp.stack([v, v]) for f, v in gm.get_node_state(name).items()}
                 for name in gm.node_names}
        return _library_program(gm, "run_sweep", 3, batch)

    def adaptive():
        fn = gm._build_dt_step_fn()                          # noqa: SLF001
        ext = gm._resolve_external_inputs(None)              # noqa: SLF001
        return jax.jit(fn), (gm._state, ext, jnp.asarray(0.01, jnp.float32), gm.params)  # noqa: SLF001

    gm.reset_state()
    return {
        "step": lowered(lambda: _library_program(gm, "step")),
        "step_baked_params": lowered(baked),
        "scan3": lowered(lambda: _library_program(gm, "run_scan", 3)),
        "sweep2": lowered(sweep),
        "adaptive": lowered(adaptive),
    }


def graph_digests(builder) -> dict:
    """``{program: sha256 of its text, or the refusal}`` of one gate graph."""
    texts = program_texts(builder())
    assert tuple(texts) == PROGRAMS
    return {name: (text if text.startswith("refused:") else _sha(text))
            for name, text in texts.items()}


def all_digests(only=None) -> dict:
    graphs = gate_graphs()
    unknown = sorted(set(only or ()) - set(graphs))
    if unknown:
        raise SystemExit(f"unknown gate graph(s) {unknown}; choose from {sorted(graphs)}")
    return {name: graph_digests(builder) for name, builder in graphs.items()
            if not only or name in only}


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def jax_version() -> str:
    import jax  # noqa: PLC0415

    return str(jax.__version__)


def load(path: Path = DIGESTS) -> dict:
    """The digests file, or an empty capture when there is none."""
    if not path.exists():
        return {"format": FORMAT, "commit": None, "captures": {}}
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT or not isinstance(data.get("captures"), dict):
        raise SystemExit(f"{path}: not a step-program digests file of format {FORMAT}")
    return data


def differences(captured: dict, current: dict) -> list:
    """Why *current* (``{graph: {program: digest}}``) is not what *captured* holds
    for this jax version; empty when it is."""
    version = jax_version()
    entry = captured.get("captures", {}).get(version)
    if entry is None:
        have = sorted(captured.get("captures", {}))
        return [f"no capture for jax {version} (the file holds {have or 'none'}): take one on "
                f"the base commit with `python scripts/capture_step_programs.py --write`"]
    out = []
    for graph, programs in current.items():
        if graph not in entry:
            out.append(f"{graph}: no captured entry for jax {version}")
            continue
        for program in PROGRAMS:
            want, got = entry[graph].get(program), programs[program]
            if want != got:
                out.append(f"{graph}: {program} is {got[:24]}, captured {str(want)[:24]}")
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def write(path: Path = DIGESTS, only=None) -> dict:
    """Add this jax version's capture to *path* (created if missing)."""
    dirty = _git("status", "--porcelain", "--", "src", "tests/property/coupled_topologies.py",
                 "tests/property/coupled_graphs.py", "scripts/capture_step_programs.py")
    if dirty:
        raise SystemExit(
            "refusing to capture from a tree with uncommitted changes to what the "
            f"programs are built from:\n{dirty}\nCommit (or discard) them first: a capture "
            "is of a commit.")
    commit = _git("rev-parse", "HEAD")
    data = load(path)
    if data["captures"] and data["commit"] != commit:
        raise SystemExit(
            f"{path} was captured on {data['commit']}, and HEAD is {commit}: an entry is "
            f"only added to a capture of the same commit.  Check that commit out, or delete "
            f"the file to start a new capture.")
    if only:
        raise SystemExit("--write captures every gate graph; --only is for --print and --check")
    data["commit"] = commit
    data["captures"][jax_version()] = all_digests()
    data["captures"] = dict(sorted(data["captures"].items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    return data


def _force_virtual_devices() -> None:
    """Give this process the virtual CPU devices the sharded graph needs.

    Must run before jax is imported.  A device count the caller already
    set is left alone.
    """
    if "jax" in sys.modules:
        raise SystemExit("capture_step_programs must set XLA_FLAGS before jax is imported")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    flags = os.environ.get("XLA_FLAGS", "")
    if HOST_FLAG not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {HOST_FLAG}={SHARDED_DEVICES}".strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="add this jax version's capture")
    mode.add_argument("--check", action="store_true", help="recompute and compare")
    mode.add_argument("--print", action="store_true", help="print this process's digests")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="a gate graph (may repeat)")
    parser.add_argument("--file", type=Path, default=DIGESTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    _force_virtual_devices()
    if args.write:
        data = write(args.file, args.only)
        print(f"wrote {args.file}: jax {jax_version()} at {data['commit']}, "
              f"{len(data['captures'][jax_version()])} graphs")
        return 0
    current = all_digests(args.only)
    if args.print:
        print(json.dumps({"jax": jax_version(), "graphs": current}, indent=1))
        return 0
    problems = differences(load(args.file), current)
    for line in problems:
        print(line)
    print(f"{len(current)} graphs, jax {jax_version()}: "
          + ("unchanged" if not problems else f"{len(problems)} difference(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
