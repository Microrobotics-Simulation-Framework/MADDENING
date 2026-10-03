"""Smoke coverage for the ~50 scripts under ``maddening.examples``.

The examples ship inside the wheel and are the first thing a new user
copies, but nothing else in the suite touches them, so an API rename can
break every one of them without turning a single test red -- and an
example that runs can still print a claim its numbers contradict, which
only running it (and checking what it asserts) catches.

Three lanes:

* A **static lane**, costing seconds.  It parses every example and
  resolves the ``maddening.*`` names it imports against the installed
  library, which is the failure mode a release actually causes.  It also
  pins the packaging invariants: no example writes its output into the
  installed package, and no example puts the package directory on
  ``sys.path``.  Static checking is the *only* coverage the provisioning
  scripts under ``examples/cloud/`` can ever have -- they allocate real,
  billable machines, so the suite must never execute them.

* A **run lane**.  Every example is classified (``run``, ``loopback``,
  ``gui``, ``cloud``, ``helper``) and the classification is checked to
  cover the directory.  Headless examples run per push at a small size,
  through their size arguments; examples whose default is too slow for
  every push also run at the default in the slow lane, beside a named
  per-push sibling.

* A **loopback lane**.  The server examples run on 127.0.0.1 with an
  OS-chosen port: their documented routes must answer and Ctrl-C must
  stop them cleanly.  ``remote_viz_client --local`` and the ``--local``
  modes of the cloud examples run end to end without a cloud account.

Every example subprocess runs under a guard (``_GUARD``) that refuses
cloud SDK imports, MADDENING's provisioning modules and any traffic off
loopback, and that turns a deprecation *issued by MADDENING* into an
error.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import maddening.examples

EXAMPLES_ROOT = Path(maddening.examples.__file__).resolve().parent

# Everything below this directory launches billable cloud infrastructure.
# Never execute it; static analysis only.
CLOUD_DIR = EXAMPLES_ROOT / "cloud"


def _example_files() -> list[Path]:
    """Every example script, excluding the empty package ``__init__``s."""
    return sorted(
        p for p in EXAMPLES_ROOT.rglob("*.py") if p.name != "__init__.py"
    )


def _rel(path: Path) -> str:
    return path.relative_to(EXAMPLES_ROOT).as_posix()


EXAMPLE_FILES = _example_files()
EXAMPLE_IDS = [_rel(p) for p in EXAMPLE_FILES]

# Optional extras whose absence makes a *library* module unimportable.  When
# one is missing the example that uses it cannot be checked, and that is a
# skip with a reason, not a failure.
_EXTRA_FOR_MODULE = {
    "maddening.usd": "usd",
    "maddening.api": "api",
    "maddening.cloud": "cloud",
    "maddening.surrogates": "surrogates",
    "maddening.viz": "viz",
}


def _extra_hint(module_name: str) -> str:
    for prefix, extra in _EXTRA_FOR_MODULE.items():
        if module_name == prefix or module_name.startswith(prefix + "."):
            return f"maddening[{extra}]"
    return module_name


def _maddening_imports(tree: ast.AST) -> list[tuple[str, tuple[str, ...], int]]:
    """``(module, imported_names, lineno)`` for every ``maddening`` import."""
    found: list[tuple[str, tuple[str, ...], int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; the examples use none.
            if node.level == 0 and node.module and (
                node.module == "maddening" or node.module.startswith("maddening.")
            ):
                found.append(
                    (node.module, tuple(a.name for a in node.names), node.lineno)
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "maddening" or alias.name.startswith("maddening."):
                    found.append((alias.name, (), node.lineno))
    return found


def _unresolved_names(tree: ast.AST) -> list[str]:
    """Names an example imports from ``maddening`` that no longer exist.

    Raises ``ImportError`` if a *library* module is missing entirely, so the
    caller can turn that into a skip when it is only an uninstalled extra.
    """
    problems: list[str] = []
    for module_name, names, lineno in _maddening_imports(tree):
        module = importlib.import_module(module_name)
        for name in names:
            if hasattr(module, name):
                continue
            try:  # ``from pkg import submodule`` is also legal
                importlib.import_module(f"{module_name}.{name}")
            except ImportError:
                problems.append(f"line {lineno}: {module_name}.{name}")
    return problems


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_parses(path: Path) -> None:
    """Every shipped example is valid Python for the interpreter we target."""
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_imports_resolve_against_the_library(path: Path) -> None:
    """Every ``maddening`` name an example imports still exists.

    This is what catches a release renaming or moving a public symbol, and
    it works without executing the example -- the only option for the cloud
    scripts.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    try:
        problems = _unresolved_names(tree)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        pytest.skip(f"optional dependency missing ({missing}); needs {_extra_hint(missing)}")
    assert not problems, (
        f"{_rel(path)} imports names that no longer exist: " + "; ".join(problems)
    )


def _embedded_payloads(tree: ast.AST) -> list[ast.AST]:
    """Parse the remote-side scripts the cloud examples embed as strings.

    ``examples/cloud/*`` ships its payloads as string literals that are
    executed on the rented VM, so they are the part most likely to drift and
    the part no local run ever touches.  Literals that are not whole modules
    (indented comment fragments, shell snippets) simply do not parse and are
    skipped -- this checks what it can, and claims nothing more.
    """
    payloads = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        source = textwrap.dedent(node.value)
        if "\nfrom maddening" not in "\n" + source:
            continue
        try:
            payloads.append(ast.parse(source))
        except SyntaxError:
            continue
    return payloads


CLOUD_FILES = [p for p in EXAMPLE_FILES if CLOUD_DIR in p.parents]


@pytest.mark.parametrize(
    "path", CLOUD_FILES, ids=[_rel(p) for p in CLOUD_FILES]
)
def test_cloud_example_remote_payload_imports_resolve(path: Path) -> None:
    """The scripts cloud examples run on the rented VM still import cleanly."""
    tree = ast.parse(path.read_text(), filename=str(path))
    payloads = _embedded_payloads(tree)
    if not payloads:
        pytest.skip("no embedded remote payload in this script")
    problems: list[str] = []
    for payload in payloads:
        try:
            problems.extend(_unresolved_names(payload))
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            pytest.skip(
                f"optional dependency missing ({missing}); needs {_extra_hint(missing)}"
            )
    assert not problems, (
        f"{_rel(path)} remote payload imports names that no longer exist: "
        + "; ".join(problems)
    )


def _sys_path_mutations(tree: ast.AST) -> list[int]:
    """Line numbers of ``sys.path.insert``/``append`` calls in real code."""
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("insert", "append")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_does_not_write_into_the_installed_package(path: Path) -> None:
    """Examples write results to the working directory, not to site-packages.

    ``dirname(__file__)/../..`` is ``src/maddening`` under the src layout, so
    joining an output filename onto it drops the file next to the library --
    which is site-packages for anyone who pip-installed MADDENING, and fails
    outright when that install is read-only.
    """
    source = path.read_text()
    assert "_project_root" not in source, (
        f"{_rel(path)} resurrects the pre-src-layout _project_root: it resolves "
        "to the package directory, not to a project root. Write outputs to "
        "os.getcwd() instead."
    )
    # AST, not a substring search: the cloud examples legitimately embed
    # ``sys.path.insert`` inside the payload strings they run on the rented
    # VM, where the checkout really is somewhere sys.path does not know.
    assert not _sys_path_mutations(ast.parse(source, filename=str(path))), (
        f"{_rel(path)} puts a directory on sys.path. The examples run against "
        "an installed maddening; a path hack only makes core/nodes/viz "
        "importable as top-level packages and duplicates module objects."
    )


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_usage_docstring_is_runnable(path: Path) -> None:
    """Usage lines name a module path that ``python -m`` can actually run."""
    source = path.read_text()
    assert "python maddening/examples/" not in source, (
        f"{_rel(path)} documents the pre-src-layout script path. Use "
        "'python -m maddening.examples.<package>.<module>'."
    )
    assert "/home/nick" not in source, (
        f"{_rel(path)} hard-codes a developer's home directory in its usage "
        "instructions."
    )


# ---------------------------------------------------------------------------
# Run lane
# ---------------------------------------------------------------------------
# Every example is classified below, and the classification is checked to
# cover the directory exactly, so a new example cannot slip in unrun.
#
#   run       -- headless; executed by test_example_runs_headless (per push,
#                at a small size where the example takes a size argument)
#                and, when its default size is too slow for every push, by
#                test_example_runs_at_default_size (slow lane).
#   loopback  -- a server, or a client/server pair; executed on 127.0.0.1
#                with an OS-chosen port by the loopback tests further down.
#   gui       -- needs a display.  Run under MPLBACKEND=Agg when that still
#                exercises it, otherwise import-checked only.
#   cloud     -- provisions billable VMs.  NEVER executed; import-checked,
#                and their remote payloads checked, by the static lane.
#   helper    -- a module other examples import, not a script.


@dataclass(frozen=True)
class Run:
    """One example invocation: ``python -m maddening.examples.<module> <args>``."""

    module: str
    args: tuple[str, ...] = ()
    #: Modules that must be importable; otherwise the test skips, naming them.
    needs: tuple[str, ...] = ()
    #: Substrings the combined output must contain.
    expect: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return " ".join((self.module, *self.args))


# Per push: each under ~5 s warm on CI's runner (3-4 s on a loaded 3-core
# slice of the development machine).
PER_PUSH = [
    Run("basics.bouncing_ball", expect=("All checks passed",)),
    Run("basics.heat_diffusion_demo", expect=("All checks passed",)),
    Run("basics.rigid_body_demo", expect=("All checks passed",)),
    Run("basics.bouncing_ball_terminal", ("--duration", "1"), needs=("rich",),
        expect=("Stopped at sim_time",)),
    # GUI under Agg: the window loop returns at once, so these cover
    # building the graph, the renderers and the teardown -- not drawing.
    Run("basics.bouncing_ball_scene", needs=("matplotlib",), expect=("Stopped at",)),
    Run("basics.bouncing_ball_combined", needs=("matplotlib", "rich"),
        expect=("Stopped at",)),
    Run("advanced.adaptive_demo", expect=("All checks passed",)),
    Run("advanced.differentiable_optimization", expect=("All checks passed",)),
    Run("advanced.external_inputs_demo", expect=("All checks passed",)),
    Run("advanced.multirate_demo", expect=("All checks passed",)),
    Run("advanced.parameter_sweep_demo", expect=("All checks passed",)),
    Run("advanced.scan_performance", ("--steps", "2000"), expect=("All checks passed",)),
    Run("advanced.profile_lbm_step", ("--n-steps", "20"),
        expect=("Perfetto JSON written to",)),
    Run("advanced.profiling_demo", ("--n-cells", "64", "--n-steps", "20"),
        expect=("All checks passed",)),
    Run("advanced.sysid_demo", ("--samples", "100", "--n-iter", "150"),
        expect=("All checks passed",)),
    Run("advanced.checkpoint_resume_demo", ("--warmup", "20", "--steps", "20"),
        expect=("All checks passed",)),
    # Pins its own device count (4) before importing JAX, whatever
    # XLA_FLAGS it inherits.
    Run("advanced.sharding_demo", ("--n-cells", "64", "--steps", "20"),
        expect=("All checks passed",)),
    # Skips its packaging / validation stages, saying why, when the C
    # compiler or FMPy is missing; CI has both.
    Run("advanced.fmu_export_demo", ("--steps", "5"), expect=("All checks passed",)),
    Run("advanced.surrogate_demo",
        ("--epochs", "10", "--train-steps", "100", "--compare-steps", "20"),
        needs=("equinox", "optax"), expect=("Done!",)),
    Run("coupling.coupling_demo", expect=("All checks passed",)),
    Run("coupling.coupled_spring_ball", expect=("All checks passed",)),
    Run("coupling.subcycling_demo", expect=("subcycling IMPROVES accuracy",)),
    Run("coupling.acceleration_comparison", ("--steps", "10"),
        expect=("Takeaway",)),
    Run("coupling.convergence_diagnostics_demo", ("--steps", "5"),
        expect=("All demos complete",)),
    Run("coupling.jacobi_vs_gauss_seidel", ("--steps", "40"), expect=("Takeaway",)),
    Run("coupling.spatial_interpolation_demo", ("--heat-steps", "2000"),
        expect=("All demos complete",)),
    Run("coupling.flux_coupling_demo", ("--sections", "1,2,3"),
        expect=("All demos completed successfully",)),
    Run("coupling.interface_mapping_demo", ("--steps", "20"),
        expect=("All checks passed",)),
    # In-process TestClient: no port, no server process.
    Run("servers.rest_params_demo", ("--steps", "5"), needs=("fastapi", "httpx"),
        expect=("All checks passed",)),
    Run("cloud.streaming.08_subscribe_lbm_velocity", ("--n-frames", "5"),
        needs=("fastapi", "uvicorn", "websockets", "zstandard"),
        expect=("End-to-end reduction",)),
]

# Default size, slow lane.  Each names the per-push sibling above that
# runs the same code at a smaller size.
SLOW = [
    (Run("advanced.surrogate_demo", needs=("equinox", "optax"), expect=("Done!",)),
     "100 training epochs: ~9 s", "advanced.surrogate_demo --epochs 10 ..."),
    (Run("advanced.scan_performance", expect=("All checks passed",)),
     "10 000-step python loop baseline", "advanced.scan_performance --steps 2000"),
    (Run("coupling.acceleration_comparison", expect=("Takeaway",)),
     "60-step runs read coupling_diagnostics() every step (~6 ms a call)",
     "coupling.acceleration_comparison --steps 10"),
    (Run("coupling.convergence_diagnostics_demo", expect=("All demos complete",)),
     "nine graphs, diagnostics read every step", "--steps 5"),
    (Run("coupling.jacobi_vs_gauss_seidel", expect=("Takeaway", "Settled")),
     "1000 steps x 7 graphs; also checks the cycle comes to rest",
     "coupling.jacobi_vs_gauss_seidel --steps 40"),
    (Run("coupling.spatial_interpolation_demo", expect=("All demos complete",)),
     "20 000 coupled heat steps", "--heat-steps 2000"),
    (Run("coupling.flux_coupling_demo", expect=("All demos completed successfully",)),
     "all seven sections, 13 graphs", "--sections 1,2,3"),
    (Run("servers.lbm_pipe_interactive", ("--frames", "2"), needs=("pyvista",),
         expect=("Rendered 2 frames",)),
     "3-D LBM plus off-screen VTK rendering", "none: needs pyvista, absent in CI"),
]

# Need usd-core, which only CI's test-usd job installs; that job runs this
# test by node id (and no slow tests, so none of these may be slow).
USD_RUNS = [
    Run("advanced.live_stage_bouncing_ball_demo", ("--steps", "20"),
        needs=("pxr",), expect=("Wrote bouncing_ball.usda",)),
    Run("coupling.vessel_bifurcation", ("--steps", "500"), needs=("pxr",),
        expect=("Done.",)),
]

NOT_RUN = {
    "servers/lbm_pipe_replay.py":
        "gui: GPU replay viewer (pygfx/wgpu) and a display; no headless path",
    "coupling/vessel_bifurcation_live.py":
        "gui: live PyVista window driven by PyVistaLiveRenderer.run_live; "
        "no off-screen mode",
    "coupling/vessel_flow_helpers.py":
        "helper: imported by servers/vessel_flow_server.py, which the "
        "loopback lane runs",
    "cloud/launch/01_validate.py":
        "cloud: constructs CloudLauncher and reads credentials -- never run",
    "cloud/launch/02_runpod_launch.py": "cloud: provisions a RunPod VM -- never run",
    "cloud/launch/03_lambda_launch.py": "cloud: provisions a Lambda VM -- never run",
    "cloud/launch/03_reconnect_test.py": "cloud: provisions a RunPod VM -- never run",
    "cloud/launch/04_aws_launch.py": "cloud: provisions an AWS VM -- never run",
    "cloud/launch/05_gcp_launch.py": "cloud: provisions a GCP VM -- never run",
    "cloud/multigpu/09_real_gpu_benchmark.py":
        "cloud: provisions a 2-GPU VM; its payload times multi-GPU speed-up, "
        "which a CPU run cannot measure",
    "cloud/streaming/06_selkies_test.py":
        "cloud: provisions a VM; its payload needs GStreamer and PyGObject",
    "cloud/streaming/07_webrtc_streaming_test.py":
        "cloud: provisions a VM; its payload needs GStreamer and PyGObject",
}

# Run by the loopback tests below.
LOOPBACK = {
    "servers/remote_sim_server.py", "servers/remote_viz_client.py",
    "servers/api_server.py", "servers/interactive_graph_server.py",
    "servers/launch_app.py", "servers/launch_server_render.py",
    "servers/lbm_pipe_server.py", "servers/vessel_flow_server.py",
    "cloud/server/04_server_test.py", "cloud/server/05_websocket_test.py",
    "cloud/multijob/08_two_vm_test.py",
}


def _module_file(module: str) -> str:
    return module.replace(".", "/") + ".py"


def test_every_example_is_classified_exactly_once() -> None:
    """New examples must be run, or excluded with a reason -- never forgotten."""
    run = {_module_file(r.module) for r in PER_PUSH + USD_RUNS} | {
        _module_file(r.module) for r, _, _ in SLOW
    }
    groups = [run, LOOPBACK, set(NOT_RUN)]
    everything = set(EXAMPLE_IDS)
    covered = set().union(*groups)
    assert covered == everything, (
        f"unclassified: {sorted(everything - covered)}; "
        f"stale: {sorted(covered - everything)}"
    )
    assert not (run & LOOPBACK) and not (run & set(NOT_RUN)) and not (
        LOOPBACK & set(NOT_RUN)
    ), "an example is in more than one class"
    for rel, reason in NOT_RUN.items():
        if rel.startswith("cloud/"):
            assert reason.startswith("cloud:"), rel


# A sitecustomize for every example subprocess.  Belt and braces around
# examples that sit next to billable infrastructure:
#  * importing a cloud SDK or MADDENING's provisioning modules fails;
#  * any socket connect/sendto or DNS lookup off loopback fails;
#  * a DeprecationWarning (or Pending/Future) *issued by MADDENING itself*
#    is raised, whatever stacklevel attributes it to -- third-party
#    deprecations are left alone.
_GUARD = textwrap.dedent('''
    import importlib.abc, ipaddress, socket, sys, warnings

    _BLOCKED = ("sky", "runpod", "boto3", "botocore", "google.cloud",
                "googleapiclient", "azure", "maddening.cloud.launcher",
                "maddening.cloud.session", "maddening.cloud._skypilot",
                "maddening.cloud.providers")

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            for b in _BLOCKED:
                if name == b or name.startswith(b + "."):
                    raise ImportError(
                        f"example run must not import {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Block())

    def _local(host):
        if host in ("localhost", "", None):
            return True
        if isinstance(host, bytes):
            host = host.decode()
        try:
            ip = ipaddress.ip_address(str(host).split("%")[0])
        except ValueError:
            return False
        return ip.is_loopback or ip.is_unspecified

    def _check(addr):
        if isinstance(addr, tuple) and addr and not _local(addr[0]):
            raise PermissionError(f"example run must stay on loopback: {addr!r}")

    _connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex
    _sendto, _gai = socket.socket.sendto, socket.getaddrinfo
    socket.socket.connect = lambda s, a: (_check(a), _connect(s, a))[1]
    socket.socket.connect_ex = lambda s, a: (_check(a), _connect_ex(s, a))[1]
    socket.socket.sendto = lambda s, d, *a: (_check(a[-1]), _sendto(s, d, *a))[1]

    def _getaddrinfo(host, *a, **k):
        if not _local(host):
            raise PermissionError(f"example run must stay on loopback: {host!r}")
        return _gai(host, *a, **k)

    socket.getaddrinfo = _getaddrinfo

    _warn = warnings.warn
    def _strict_warn(message, category=None, stacklevel=1, *a, **k):
        cat = category or (type(message) if isinstance(message, Warning)
                           else UserWarning)
        if issubclass(cat, (DeprecationWarning, PendingDeprecationWarning,
                            FutureWarning)):
            caller = sys._getframe(1).f_code.co_filename.replace("\\\\", "/")
            if "/maddening/" in caller and "/maddening/examples/" not in caller:
                raise cat(str(message))
        return _warn(message, category, stacklevel + 1, *a, **k)
    warnings.warn = _strict_warn
''')

_CLOUD_ENV = ("RUNPOD_API_KEY", "LAMBDA_API_KEY", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
              "GOOGLE_APPLICATION_CREDENTIALS", "CLOUDSDK_CONFIG",
              "SKYPILOT_CONFIG", "AZURE_CLIENT_SECRET")


@pytest.fixture(scope="module")
def example_env(tmp_path_factory) -> dict[str, str]:
    """Environment for example subprocesses: CPU, Agg, guarded, no credentials."""
    root = tmp_path_factory.mktemp("example_env")
    guard = root / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(_GUARD)
    home = root / "home"
    home.mkdir()
    # Headless: no display either, so VTK renders off-screen rather than
    # reaching for an X server it may not be allowed to use.
    env = {k: v for k, v in os.environ.items()
           if k not in _CLOUD_ENV and k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    env.update({
        "JAX_PLATFORMS": "cpu",
        "MPLBACKEND": "Agg",
        # One font cache for the module: an empty one costs ~3 s per run.
        "MPLCONFIGDIR": str(root / "mpl"),
        # No credentials or SkyPilot state can be read from here.
        "HOME": str(home),
        "PYTHONPATH": os.pathsep.join(
            [str(guard)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        ),
    })
    return env


def _skip_unless_importable(needs: tuple[str, ...]) -> None:
    missing = [m for m in needs if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {', '.join(missing)} (optional dependency not installed)")


def _run_example(run: Run, cwd: Path, env: dict[str, str], timeout: float = 300):
    _skip_unless_importable(run.needs)
    result = subprocess.run(
        [sys.executable, "-m", f"maddening.examples.{run.module}", *run.args],
        cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"{run.id} exited {result.returncode}\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-3000:]}"
    )
    for text in run.expect:
        assert text in output, f"{run.id}: expected {text!r} in its output"
    return output


@pytest.mark.parametrize("run", PER_PUSH, ids=[r.id for r in PER_PUSH])
def test_example_runs_headless(run: Run, tmp_path: Path, example_env) -> None:
    """The example runs end to end at a small size, exits 0, says it passed.

    Run from ``tmp_path``, so anything it saves lands there -- which also
    proves it does not write into the installed package.
    """
    _run_example(run, tmp_path, example_env)


@pytest.mark.parametrize("run", USD_RUNS, ids=[r.id for r in USD_RUNS])
def test_usd_example_runs_headless(run: Run, tmp_path: Path, example_env) -> None:
    """The USD examples run end to end and write their stage into ``tmp_path``."""
    _run_example(run, tmp_path, example_env)
    assert list(tmp_path.glob("*.usda")), f"{run.id} wrote no .usda file"


@pytest.mark.slow
@pytest.mark.parametrize(
    "run, reason, sibling", SLOW, ids=[r.id for r, _, _ in SLOW],
)
def test_example_runs_at_default_size(run, reason, sibling, tmp_path, example_env):
    """The example's user-facing default runs too (slow: see *reason*)."""
    _run_example(run, tmp_path, example_env, timeout=900)


# ---------------------------------------------------------------------------
# Loopback lane: servers and client/server pairs on 127.0.0.1
# ---------------------------------------------------------------------------

def _stop(proc: subprocess.Popen, grace: float = 20.0) -> int:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
    return proc.wait()


class _Served:
    """A server example started with ``--port 0``; its URL once it answers."""

    def __init__(self, module, args, env, cwd, ready_path, startup=180.0):
        import httpx

        self.proc = subprocess.Popen(
            [sys.executable, "-m", f"maddening.examples.{module}",
             "--port", "0", *args],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.lines: list[str] = []
        self._port: list[int] = []
        found = threading.Event()

        def drain():
            for line in self.proc.stdout:
                self.lines.append(line)
                m = re.search(r"Serving on http://127\.0\.0\.1:(\d+)", line)
                if m and not self._port:
                    self._port.append(int(m.group(1)))
                    found.set()
            found.set()

        self._drain = threading.Thread(target=drain, daemon=True)
        self._drain.start()
        found.wait(startup)
        if not self._port:
            self.stop()
            pytest.fail(f"{module} never said where it serves:\n{self.output}")
        self.url = f"http://127.0.0.1:{self._port[0]}"
        deadline = time.monotonic() + startup
        while True:
            try:
                if httpx.get(self.url + ready_path, timeout=5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline or self.proc.poll() is not None:
                self.stop()
                pytest.fail(f"{module} did not answer {ready_path}:\n{self.output}")
            time.sleep(0.2)

    @property
    def output(self) -> str:
        return "".join(self.lines[-60:])

    def stop(self) -> int:
        """Ctrl-C the server and close its output.  The pipe is closed
        here, after the drain has read it to the end: left to the garbage
        collector, its ``ResourceWarning`` landed in whichever later test
        the collector ran in -- once inside pytest's own import of
        ``tracemalloc``, which failed that test instead."""
        code = _stop(self.proc)
        self._drain.join(30)
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        return code


def _ws_messages(url: str, n: int) -> list:
    import asyncio

    import websockets

    async def go():
        async with websockets.connect(url, open_timeout=30) as ws:
            return [await asyncio.wait_for(ws.recv(), timeout=30) for _ in range(n)]

    return asyncio.run(go())


# (module, extra args, needs, readiness path, probes).  A probe is
# (method, path) -- an HTTP request that must answer 200 -- or
# ("WS", path, n) -- a WebSocket that must deliver n messages.  Never
# /cloud/*: SimulationServer serves a launch route, and nothing here may
# reach it.
HTTP_SERVERS = [
    ("servers.api_server", (), ("fastapi", "uvicorn", "websockets"), "/healthz",
     [("GET", "/graph"), ("POST", "/sim/step"), ("POST", "/sim/run?n_steps=100"),
      ("GET", "/graph/state"), ("POST", "/sim/start"), ("WS", "/ws/state", 3),
      ("POST", "/sim/stop")]),
    ("servers.interactive_graph_server", (), ("fastapi", "uvicorn"), "/healthz",
     [("GET", "/viz/graph"), ("GET", "/graph"), ("POST", "/sim/step")]),
    ("servers.launch_app", ("--no-browser",), ("fastapi", "uvicorn"), "/healthz",
     [("GET", "/viz/app"), ("POST", "/sim/step"), ("GET", "/graph/state")]),
    ("servers.launch_server_render", (), ("fastapi", "uvicorn", "websockets",
                                          "matplotlib"), "/healthz",
     [("GET", "/viz/render"), ("POST", "/sim/start"), ("WS", "/ws/render", 2),
      ("POST", "/sim/stop")]),
    ("servers.lbm_pipe_server", ("--grid", "24", "12", "12"),
     ("fastapi", "uvicorn", "websockets", "pyvista"), "/healthz",
     [("GET", "/viz/render"), ("POST", "/sim/start"), ("WS", "/ws/render", 2),
      ("POST", "/sim/stop")]),
    ("servers.vessel_flow_server", ("--grid", "32", "16", "16"),
     ("fastapi", "uvicorn", "websockets"), "/sim/vitals",
     [("GET", "/"), ("GET", "/sim/geometry"), ("PUT", "/sim/heart_rate?bpm=120"),
      ("POST", "/sim/inject_clot?x=10&y=8&z=8&radius=2"), ("WS", "/ws/state", 2),
      ("POST", "/sim/clear_clot"), ("POST", "/sim/stop")]),
]


@pytest.mark.parametrize(
    "module, args, needs, ready, probes", HTTP_SERVERS,
    ids=[m for m, *_ in HTTP_SERVERS],
)
def test_http_example_server_serves_on_a_free_port(
    module, args, needs, ready, probes, tmp_path, example_env,
) -> None:
    """``--port 0`` binds an OS-chosen loopback port, the documented routes
    answer, and Ctrl-C shuts the server down cleanly (exit 0)."""
    _skip_unless_importable(("httpx", *needs))
    import httpx

    served = _Served(module, args, example_env, tmp_path, ready)
    try:
        for probe in probes:
            assert not probe[1].startswith("/cloud")
            if probe[0] == "WS":
                msgs = _ws_messages(served.url.replace("http", "ws") + probe[1],
                                    probe[2])
                assert len(msgs) == probe[2]
            else:
                r = httpx.request(probe[0], served.url + probe[1], timeout=60)
                assert r.status_code == 200, (probe, r.status_code, r.text[:300])
    finally:
        code = served.stop()
    assert code == 0, f"{module} exited {code} on Ctrl-C:\n{served.output}"


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else (pid reuse)
        return True
    return True


def test_remote_viz_client_local_mode_streams_frames_and_stops_its_server(
    tmp_path, example_env,
) -> None:
    """``remote_viz_client --local``: one command starts the server on a free
    loopback port, frames arrive, and the server subprocess is gone after."""
    _skip_unless_importable(("zmq",))
    output = _run_example(
        Run("servers.remote_viz_client",
            ("--local", "--mode", "print", "--frames", "5", "--timeout", "120"),
            expect=("Received 5 of 5 frames",)),
        tmp_path, example_env,
    )
    pid = int(re.search(r"Started local server \(pid (\d+)\)", output).group(1))
    assert re.search(rf"Local server \(pid {pid}\) stopped, exit code 0", output)
    assert re.search(r"on tcp://127\.0\.0\.1:\d+", output)
    assert not _pid_exists(pid), f"remote_sim_server (pid {pid}) outlived the client"


def test_remote_viz_client_reports_missing_frames(tmp_path, example_env) -> None:
    """A run that cannot get its frames in time fails, and still stops the server."""
    _skip_unless_importable(("zmq",))
    result = subprocess.run(
        [sys.executable, "-m", "maddening.examples.servers.remote_viz_client",
         "--local", "--mode", "print", "--frames", "1000000", "--timeout", "2"],
        cwd=tmp_path, env=example_env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-2000:]
    pid = int(re.search(r"Started local server \(pid (\d+)\)", result.stdout).group(1))
    assert not _pid_exists(pid)


# The cloud examples' --local modes run their remote payload on loopback.
# The guard above makes any import of the provisioning modules fail, so a
# --local path that ever reached for them would fail here rather than
# provision.
CLOUD_LOCAL = [
    Run("cloud.server.04_server_test", ("--local",),
        needs=("fastapi", "uvicorn"), expect=("All server tests passed!",)),
    Run("cloud.server.05_websocket_test", ("--local",),
        needs=("fastapi", "uvicorn", "websockets"),
        expect=("All WebSocket tests passed!",)),
    Run("cloud.multijob.08_two_vm_test", ("--local",), needs=("zmq",),
        expect=("ALL MULTI-JOB TESTS PASSED!",)),
]


@pytest.mark.parametrize("run", CLOUD_LOCAL, ids=[r.id for r in CLOUD_LOCAL])
def test_cloud_example_local_mode_runs_on_loopback(run, tmp_path, example_env):
    """The cloud examples' ``--local`` mode needs no cloud account."""
    output = _run_example(run, tmp_path, example_env)
    m = re.search(r"Local server \(pid (\d+)\) stopped, exit code (-?\d+)", output)
    if m:  # the two server tests
        assert m.group(2) == "0", output[-2000:]
        assert not _pid_exists(int(m.group(1)))


def test_example_guard_refuses_cloud_imports_and_off_loopback_traffic(
    example_env, tmp_path,
) -> None:
    """The guard the run lane relies on can actually fail."""
    probe = textwrap.dedent('''
        import socket, warnings
        for name in ("sky", "maddening.cloud.launcher"):
            try:
                __import__(name)
            except ImportError:
                print("blocked", name)
        try:
            socket.create_connection(("192.0.2.1", 80), timeout=1)
        except PermissionError:
            print("blocked network")
        from maddening.nodes.rigid_body_2d import RigidBody2DNode
        try:
            RigidBody2DNode("b", 0.01)
        except DeprecationWarning:
            print("blocked deprecation")
    ''')
    result = subprocess.run(
        [sys.executable, "-c", probe], env=example_env, cwd=tmp_path,
        capture_output=True, text=True, timeout=120,
    )
    out = result.stdout
    for line in ("blocked sky", "blocked maddening.cloud.launcher",
                 "blocked network", "blocked deprecation"):
        assert line in out, (line, out, result.stderr[-2000:])
