"""Compilation counts, and the regression gate built on them.

MADDENING had no performance regression gate at all.  The obvious one to
build is a wall-clock gate, and this repository already knows what that
costs: ``test_step_cost_does_not_scale_with_max_iterations`` compared
per-step cost at two iteration caps with a 3x bound on quantities around
1e-4 s, and failed CI at 3.27x on a pull request that changed 122
documentation files and no code.  PR 75 replaced it with an assertion on
a count.

So the gate counts things.  ``CompileCounts`` reports retraces (XLA
compilations of the step), jaxpr primitives and lowered StableHLO ops --
integers that reproduce exactly on any machine, under any load.
``scripts/compile_counts.py`` measures five workloads and compares them
to ``benchmarks/compile_counts_baseline.json``.

Half of this module tests the counts.  The other half tests **that the
gate can fail**, which is the half that matters: MADDENING shipped four
compliance gates that could not fail, and one of them was verifying
*zero* references while passing CI for months.  The mutation here is a
real one -- a node whose state aval changes after the first step, so the
step really is compiled twice -- driven through the real script, whose
exit code and message are then asserted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.profiler import (
    CompileCounts,
    compile_counts,
    count_jaxpr_primitives,
    profile_graph,
)
from maddening.nodes.spring import SpringDamperNode

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "compile_counts.py"
BASELINE = REPO_ROOT / "benchmarks" / "compile_counts_baseline.json"


def _single() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=30.0, damping=2.0, initial_position=0.5,
    ))
    gm.compile()
    return gm


def _coupled() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        "a", 0.01, stiffness=30.0, damping=2.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        "b", 0.01, stiffness=30.0, damping=2.0, initial_position=3.0,
    ))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=25, tolerance=1e-8)
    gm.compile()
    return gm


class AvalChangesOnce(SimulationNode):
    """A node that forces exactly one extra compile of the step.

    The first step is traced against a ``(1,)`` leaf and returns a
    ``(2,)`` one, so the second step's argument has a different aval and
    JAX must trace and compile the whole step again.  From the third
    step on the aval is stable, so the retrace count settles at 2 rather
    than growing without bound -- the mutation has to be as
    deterministic as the thing it is testing.

    This is the shape of the real bug the counts exist to catch: nothing
    here is wrong enough to raise, the results are all correct, and the
    only symptom is a second XLA compile on every run.
    """

    def initial_state(self):
        return {"x": jnp.zeros(1), "n": jnp.array(0, jnp.int32)}

    def state_fields(self):
        return ["x", "n"]

    def update(self, state, boundary_inputs, dt, *, params=None):
        x = state["x"] if state["x"].shape[0] == 2 else jnp.zeros(2)
        return {"x": x + dt, "n": state["n"] + 1}


# ---------------------------------------------------------------------------
# The counts themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_single, _coupled], ids=["single", "coupled"])
def test_a_healthy_graph_compiles_its_step_exactly_once(build):
    """One trace is one XLA compile; a healthy graph needs exactly one."""
    gm = build()
    counts = compile_counts(gm)
    assert counts.retrace_count == 1
    assert counts.jaxpr_primitive_count > 0
    assert counts.hlo_op_count > 0


def test_measuring_the_counts_does_not_itself_retrace_the_graph():
    """Reading the counts must not move the number being read.

    ``compile_counts`` lowers the step and builds its jaxpr, both of
    which re-enter ``jax.jit``.  On today's JAX those are cache hits, so
    this assertion is redundant -- which is the point of writing it.  If
    a JAX upgrade ever stops caching them, the measurement would start
    inflating its own retrace count and every baseline in the repository
    would move at once, with nothing to say why.
    """
    gm = _coupled()
    gm.step()
    before = gm.trace_count
    counts = compile_counts(gm)
    assert gm.trace_count == before == counts.retrace_count == 1
    # and the graph is still usable, on the same compilation
    gm.step()
    assert gm.trace_count == 1


def test_counts_are_reproducible_for_two_identical_graphs():
    """The property the whole gate rests on: same graph, same integers.

    No wall-clock quantity in ``ProfileReport`` can promise this, which
    is why none of them gate.
    """
    assert compile_counts(_coupled()).as_dict() == compile_counts(_coupled()).as_dict()


def test_counts_grow_with_the_graph():
    """Guard the assertions above against passing vacuously.

    A ``compile_counts`` that returned a constant -- or counted only the
    outermost jaxpr, missing every ``scan``/``while``/``cond`` body --
    would satisfy every reproducibility check in this file.  The coupled
    pair puts a fixed-point ``while`` loop in the step, so it must count
    far higher than a lone spring.
    """
    single = compile_counts(_single())
    coupled = compile_counts(_coupled())
    assert coupled.jaxpr_primitive_count > 10 * single.jaxpr_primitive_count
    assert coupled.hlo_op_count > 10 * single.hlo_op_count


def test_subjaxprs_are_counted_but_only_once_each():
    """A ``scan`` body counts once, not once per iteration.

    Otherwise the number would describe the trip count rather than the
    program, and a 16-step and a 1600-step scan of the same graph could
    not be compared.
    """
    def body(carry, _):
        return carry * 1.5 + 1.0, None

    def scan_10(x):
        out, _ = jax.lax.scan(body, x, None, length=10)
        return out

    def scan_1000(x):
        out, _ = jax.lax.scan(body, x, None, length=1000)
        return out

    x = jnp.float32(1.0)
    n10 = count_jaxpr_primitives(jax.make_jaxpr(scan_10)(x))
    n1000 = count_jaxpr_primitives(jax.make_jaxpr(scan_1000)(x))
    assert n10 == n1000
    # and the body really was descended into: scan alone would be 1
    assert n10 > 1


def test_scan_counts_are_recorded_only_when_a_scan_was_measured():
    """A workload that measured no scan must not record a scan count of
    zero, which a baseline reader could not tell from a real zero."""
    gm = _single()
    assert "scan_retrace_count" not in compile_counts(gm).as_dict()

    gm = _single()
    with_scan = compile_counts(gm, scan_steps=8).as_dict()
    assert with_scan["scan_steps"] == 8
    assert with_scan["scan_retrace_count"] == 1
    assert with_scan["scan_jaxpr_primitive_count"] > 0
    assert with_scan["scan_hlo_op_count"] > 0


def test_the_scan_measurement_leaves_the_graph_where_it_found_it():
    """A scan measurement runs the graph ``scan_steps`` forward
    internally; the caller did not ask for that, so it is rolled back.

    Warmup is a different matter and is *not* rolled back -- it is the
    documented cost of getting a meaningful retrace count -- so this
    turns it off to isolate the scan.
    """
    gm = _single()
    gm.step()
    before = float(gm.get_node_state("s")["position"])
    compile_counts(gm, scan_steps=8, warmup_steps=0)
    assert float(gm.get_node_state("s")["position"]) == before


def test_warmup_steps_advance_the_graph_and_are_not_rolled_back():
    """The side effect, pinned rather than left implicit.

    Four steps is what makes a retrace on the second or third step
    visible, so it has to really happen; a caller that cannot afford it
    passes ``warmup_steps=0``.
    """
    gm = _single()
    start = float(gm.get_node_state("s")["position"])
    compile_counts(gm, warmup_steps=4)
    assert float(gm.get_node_state("s")["position"]) != start

    gm = _single()
    compile_counts(gm, warmup_steps=0)
    # one step is still the floor: a graph that never ran has no
    # compiled program to lower and a retrace count of 0
    assert float(gm.get_node_state("s")["position"]) != start


def test_profile_graph_reports_the_counts_and_prints_them():
    gm = _coupled()
    report = profile_graph(gm, n_steps=5, n_warmup=1)
    assert isinstance(report.counts, CompileCounts)
    assert report.counts.retrace_count == 1
    text = str(report)
    assert "Compile counts" in text
    assert f"{report.counts.hlo_op_count}" in text


def test_a_changing_state_aval_is_reported_as_an_extra_compile():
    """The mutation itself, measured directly: the counts see it."""
    gm = GraphManager()
    gm.add_node(AvalChangesOnce("g", 0.01))
    gm.compile()
    for _ in range(4):
        gm.step()
    assert gm.trace_count == 2, "the mutation did not actually retrace"
    assert compile_counts(gm).retrace_count == 2


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
#
# The script pins ``JAX_PLATFORMS`` and the device count before importing
# JAX, because both change the counts.  Under pytest JAX is already
# imported (with whatever device count the conftest chose), so the gate
# has to run in a child process for its numbers to mean anything.


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_the_committed_baseline_matches_what_the_code_compiles_to():
    """The gate itself.  A failure here is the regression it exists for;
    the message says how to regenerate the baseline if it is intended."""
    done = _run("--check")
    assert done.returncode == 0, done.stderr or done.stdout


def test_every_baseline_workload_records_one_retrace():
    """Read the committed file directly rather than through the script.

    The gate would pass just as happily against a baseline that recorded
    a retrace count of 3, so something has to state the healthy value.
    Any change that makes this fail is a real extra XLA compile per run,
    whatever the baseline says.
    """
    doc = json.loads(BASELINE.read_text())
    assert doc["workloads"], "the baseline records no workloads"
    for name, counts in doc["workloads"].items():
        assert counts["retrace_count"] == 1, (name, counts)
        if "scan_retrace_count" in counts:
            assert counts["scan_retrace_count"] == 1, (name, counts)


# The child imports the script as a module, replaces its workload set
# with the retracing node above, and drives the real ``main``.  Nothing
# about the gate is stubbed: it measures, compares and returns an exit
# code exactly as it does in CI.
_CHILD = textwrap.dedent("""
    import importlib.util, sys
    from pathlib import Path

    repo, baseline, argv = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    spec = importlib.util.spec_from_file_location(
        "compile_counts_script", repo / "scripts" / "compile_counts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sys.path.insert(0, str(repo / "tests"))
    from core.test_compile_counts import AvalChangesOnce
    from maddening.core.graph_manager import GraphManager

    def build():
        gm = GraphManager()
        gm.add_node(AvalChangesOnce("g", 0.01))
        gm.compile()
        return gm

    mod.WORKLOADS.clear()
    mod.WORKLOADS["retracing"] = (build, 0)
    sys.exit(mod.main([*argv, "--baseline", baseline]))
""")


def _child(baseline: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(REPO_ROOT), str(baseline), *argv],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


@pytest.fixture
def mutant_baseline(tmp_path) -> Path:
    """A baseline generated from the deliberately retracing workload.

    Generated rather than hand-written: the numbers in it are a real
    measurement of code that really compiles its step twice.
    """
    path = tmp_path / "mutant_baseline.json"
    done = _child(path)
    assert done.returncode == 0, done.stderr
    doc = json.loads(path.read_text())
    assert doc["workloads"]["retracing"]["retrace_count"] == 2, (
        "the mutation did not produce an extra compile, so nothing below "
        "is testing anything: " + json.dumps(doc["workloads"])
    )
    return path


def _rewrite(path: Path, **fields) -> None:
    doc = json.loads(path.read_text())
    doc["workloads"]["retracing"].update(fields)
    path.write_text(json.dumps(doc, indent=2))


def test_the_gate_fails_on_an_extra_retrace_and_names_it(mutant_baseline):
    """The mutation test the gate has to survive to be worth having.

    The baseline is set back to the one trace the code had *before* the
    mutation; the measurement still sees two.  That is exactly the
    silent regression this gate exists for -- correct results, one extra
    XLA compile on every run -- and the gate must fail on it, name the
    field, and say what to do.
    """
    _rewrite(mutant_baseline, retrace_count=1)
    done = _child(mutant_baseline, "--check")
    assert done.returncode != 0, "the gate passed on an extra retrace"
    assert "retracing.retrace_count" in done.stderr
    assert "1 -> 2" in done.stderr
    assert "extra XLA compile" in done.stderr
    assert "python scripts/compile_counts.py" in done.stderr


def test_the_gate_fails_on_a_large_op_count_jump(mutant_baseline):
    """Op counts are banded, not exact -- but a 3x jump is the very
    regression the gate was asked for, so it must still fire."""
    doc = json.loads(mutant_baseline.read_text())
    real = doc["workloads"]["retracing"]["jaxpr_primitive_count"]
    _rewrite(mutant_baseline, jaxpr_primitive_count=real * 3)
    done = _child(mutant_baseline, "--check")
    assert done.returncode != 0, "the gate passed on a 3x op-count jump"
    assert "retracing.jaxpr_primitive_count" in done.stderr
    assert "outside the" in done.stderr


def test_the_gate_tolerates_a_one_op_difference(mutant_baseline):
    """The other half of a band: it has to *not* fire inside it.

    Documented deliberately.  The band is two ops on a matching JAX
    version, so a one-op difference passes; that is the price of not
    failing CI over a lowering detail nobody changed, and it is nowhere
    near enough to hide a regression worth gating.
    """
    doc = json.loads(mutant_baseline.read_text())
    real = doc["workloads"]["retracing"]["hlo_op_count"]
    _rewrite(mutant_baseline, hlo_op_count=real + 1)
    done = _child(mutant_baseline, "--check")
    assert done.returncode == 0, done.stderr


def test_the_gate_fails_closed_on_a_count_it_does_not_recognise(mutant_baseline):
    """A field in the baseline that the measurement does not produce is
    a failure, not something to skip over.

    Where a gate in this repository has been wrong before, it has been
    wrong by quietly ignoring what it did not recognise.  A count added
    to ``CompileCounts`` and never gated, or one removed and left in the
    baseline, has to be visible.
    """
    _rewrite(mutant_baseline, some_future_count=17)
    done = _child(mutant_baseline, "--check")
    assert done.returncode != 0, "the gate ignored an unrecognised count"
    assert "some_future_count" in done.stderr


def test_the_gate_fails_when_a_workload_disappears(mutant_baseline):
    """Dropping a workload from the script must not silently shrink what
    is gated."""
    doc = json.loads(mutant_baseline.read_text())
    doc["workloads"]["a_workload_the_script_no_longer_builds"] = {
        "retrace_count": 1, "jaxpr_primitive_count": 1, "hlo_op_count": 1,
    }
    mutant_baseline.write_text(json.dumps(doc, indent=2))
    done = _child(mutant_baseline, "--check")
    assert done.returncode != 0
    assert "a_workload_the_script_no_longer_builds" in done.stderr


def test_the_gate_fails_on_a_missing_baseline(tmp_path):
    missing = tmp_path / "not_here.json"
    done = _child(missing, "--check")
    assert done.returncode != 0
    assert "python scripts/compile_counts.py" in done.stderr


def test_regenerating_the_baseline_is_the_documented_one_command(tmp_path):
    """``--check`` must be satisfiable by the command it prints.

    A gate whose fix instruction does not work is a gate people route
    around.  Generate into a scratch path, check against it, and the
    exit code has to be 0.
    """
    scratch = tmp_path / "regenerated.json"
    assert _run("--baseline", str(scratch)).returncode == 0
    assert _run("--check", "--baseline", str(scratch)).returncode == 0
    # ... over the same workload set the committed baseline covers.  The
    # *values* are deliberately not compared to the committed ones: that
    # would be an exact cross-version gate by the back door, and the
    # banding in the script exists precisely so a JAX upgrade that shifts
    # a count does not wedge CI.
    assert set(json.loads(scratch.read_text())["workloads"]) == \
        set(json.loads(BASELINE.read_text())["workloads"])
