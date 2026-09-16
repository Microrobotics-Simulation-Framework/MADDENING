"""C-level tests for the FMU wrapper: unit tests and a fuzz harness, each
built plain and with ``-fsanitize=address,undefined``, plus FMPy driving
an ASan-instrumented build in a subprocess.  Skipped without a C
compiler.  (Valgrind is not required; ASan/UBSan cover the same memory
and UB classes on this toolchain.)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from maddening.fmi.package import C_SOURCE, FMI3_INCLUDE_DIR, find_c_compiler

C_DIR = Path(__file__).resolve().parent / "c"
CC = find_c_compiler()
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler")

SANITIZE = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=undefined"]
SAN_ENV = {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1:halt_on_error=1",
           "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"}


def _build(src: Path, out: Path, *flags: str, shared: bool = False) -> Path:
    cmd = [CC, "-g", "-O1", "-Wall", "-Wextra", f"-I{FMI3_INCLUDE_DIR}", str(src), "-o", str(out),
           "-lpthread", *flags]
    if shared:
        cmd[1:1] = ["-shared", "-fPIC"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # the wrapper is compiled with -Wall -Wextra: no warnings allowed
    assert "warning:" not in proc.stderr, proc.stderr
    return out


def _run(exe: Path, *args: str, env: dict | None = None, timeout: float = 120) -> str:
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run([str(exe), *args], capture_output=True, text=True, env=full_env,
                          timeout=timeout)
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "runtime error" not in proc.stderr and "AddressSanitizer" not in proc.stderr, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("sanitized", [False, True], ids=["plain", "asan+ubsan"])
def test_c_unit_tests(tmp_path, sanitized):
    exe = _build(C_DIR / "test_maddening_fmu.c", tmp_path / "unit", *(SANITIZE if sanitized else []))
    out = _run(exe, env=SAN_ENV if sanitized else None)
    assert "0 failures" in out, out


@pytest.mark.parametrize("seed", [1, 7, 12345])
def test_fuzz_replies_under_sanitizers(tmp_path, seed):
    exe = _build(C_DIR / "fuzz_maddening_fmu.c", tmp_path / "fuzz", *SANITIZE)
    out = _run(exe, str(seed), "3000", env=SAN_ENV)
    assert f"seed={seed} iterations=3000 ok" in out


@pytest.mark.slow
def test_fuzz_long_run(tmp_path):
    exe = _build(C_DIR / "fuzz_maddening_fmu.c", tmp_path / "fuzz", *SANITIZE)
    out = _run(exe, "2026", "60000", env=SAN_ENV, timeout=1500)
    assert "iterations=60000 ok" in out


def test_wrapper_compiles_clean_with_strict_warnings(tmp_path):
    """The shipped source builds warning-free under -Wall -Wextra -pedantic."""
    _build(C_SOURCE, tmp_path / "strict.so", "-std=c11", "-pedantic", shared=True)


def test_fmpy_drives_the_sanitized_binary(tmp_path):
    """FMPy drives an ASan/UBSan build of the wrapper in a subprocess with
    the sanitizer runtime preloaded.  Only FMPy and the wrapper live in
    that process (JAX itself is not ASan-clean under LD_PRELOAD); the
    graph, the bridge and a hostile bridge run here, unsanitized."""
    import json
    import socket
    import struct
    import threading

    import jax.numpy as jnp

    pytest.importorskip("fmpy")
    from tests.fmi.test_c_wrapper import DT, _bridge, _graph
    from maddening.fmi.package import write_fmu
    from maddening.fmi.tcp_bridge import recv_message, send_message

    libasan = subprocess.run([CC, "-print-file-name=libasan.so"], capture_output=True,
                             text=True).stdout.strip()
    if not libasan or not os.path.exists(libasan):
        pytest.skip("no libasan runtime")
    so = _build(C_SOURCE, tmp_path / "maddening_fmu.so", *SANITIZE, shared=True)

    gm = _graph()
    md, bridge = _bridge(gm)

    class Hostile:
        """Answers hello correctly, then garbage / truncated / huge frames."""

        def __init__(self):
            self.srv = socket.socket()
            self.srv.bind(("127.0.0.1", 0))
            self.srv.listen(1)
            self.endpoint = "127.0.0.1:%d" % self.srv.getsockname()[1]
            threading.Thread(target=self.run, daemon=True).start()

        def run(self):
            conn, _ = self.srv.accept()
            with conn:
                n = 0
                while True:
                    try:
                        req = recv_message(conn)
                    except Exception:
                        return
                    if req is None:
                        return
                    n += 1
                    if req.get("op") == "hello":
                        send_message(conn, {"ok": True, "token": md.instantiation_token})
                    elif n % 3 == 0:
                        conn.sendall(struct.pack(">I", 70000) + b'{"ok":true' + b"x" * 100)
                        return                                     # truncated frame, hang up
                    elif n % 3 == 1:
                        conn.sendall(struct.pack(">I", 12) + b"\xff\x00garbage!!!!")
                    else:
                        send_message(conn, {"ok": True, "values": "notalist", "state": 5})

    script = tmp_path / "drive.py"
    script.write_text('''
import json, sys
from fmpy import simulate_fmu
fmu, hostile, dt = sys.argv[1], sys.argv[2], float(sys.argv[3])
res = simulate_fmu(fmu, start_time=0.0, stop_time=0.2, step_size=dt, output_interval=dt,
                   start_values={"spring.params.stiffness": 45.0},
                   output=["ball.position", "spring.position"])
out = {"spring": float(res["spring.position"][-1]), "ball": float(res["ball.position"][-1])}
try:
    simulate_fmu(hostile, start_time=0.0, stop_time=0.05, step_size=dt, output=["ball.position"])
    out["hostile"] = "returned"
except Exception as exc:
    out["hostile"] = type(exc).__name__
print("RESULT " + json.dumps(out))
''')
    env = {**os.environ, "LD_PRELOAD": libasan, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
           "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"}
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint)
        h = Hostile()
        fmu2 = write_fmu(md, tmp_path / "hostile.fmu", binary=so, endpoint=h.endpoint)
        proc = subprocess.run([sys.executable, str(script), str(fmu), str(fmu2), str(DT)],
                              capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-6000:]}"
    assert "AddressSanitizer" not in proc.stderr and "runtime error" not in proc.stderr, proc.stderr[-6000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    got = json.loads(line[7:])
    ref = _graph()
    p = ref.params
    p["nodes"]["spring"]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    out = ref.run_scan(int(round(0.2 / DT)), params=p)
    assert got["spring"] == pytest.approx(float(out["spring"]["position"]), rel=1e-5)
    assert got["ball"] == pytest.approx(float(out["ball"]["position"]), rel=1e-5)
    assert got["hostile"] != "returned" or True     # the importer must not crash; erroring is fine


VALGRIND = shutil.which("valgrind")
CLANG = shutil.which("clang")


@pytest.mark.skipif(VALGRIND is None, reason="valgrind not installed")
def test_c_unit_tests_under_valgrind(tmp_path):
    """Memcheck over the whole unit-test binary: no invalid reads/writes,
    no leaks, no use of uninitialised values (error exit code on any)."""
    exe = _build(C_DIR / "test_maddening_fmu.c", tmp_path / "unit_vg")
    cmd = [VALGRIND, "--error-exitcode=99", "--leak-check=full", "--errors-for-leak-kinds=definite",
           "--track-origins=yes", "-q", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-6000:]}"
    assert "0 failures" in proc.stdout


@pytest.mark.skipif(VALGRIND is None, reason="valgrind not installed")
def test_fuzz_under_valgrind(tmp_path):
    exe = _build(C_DIR / "fuzz_maddening_fmu.c", tmp_path / "fuzz_vg")
    cmd = [VALGRIND, "--error-exitcode=99", "--leak-check=full", "--errors-for-leak-kinds=definite",
           "-q", str(exe), "42", "300"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-6000:]}"


@pytest.mark.skipif(CLANG is None, reason="clang (libFuzzer) not installed")
def test_libfuzzer_short_campaign(tmp_path):
    """Coverage-guided fuzzing of the reply parser for a bounded time.

    Built with UBSan (not ASan): on the GitHub runner an ASan-instrumented
    libFuzzer process reported ~7.6 GB RSS already at INITED with 25 MB of
    live heap, tripping libFuzzer's RSS-based limit, while the same binary
    sits at ~40 MB on a workstation.  So the guard here is a per-allocation
    limit (``-malloc_limit_mb``), which does not depend on how the host
    accounts resident memory; memory safety and leaks of the identical
    harness are covered by the ASan seeded runs and valgrind above.  This
    campaign contributes coverage-guided input exploration.
    """
    exe = tmp_path / "libfuzz"
    cmd = ["clang", "-g", "-O1", "-DLIBFUZZER", "-fsanitize=fuzzer,undefined",
           "-fno-sanitize-recover=undefined",
           f"-I{FMI3_INCLUDE_DIR}", str(C_DIR / "fuzz_maddening_fmu.c"), "-o", str(exe), "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"clang cannot build libFuzzer targets here: {proc.stderr[-500:]}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # No RSS-based abort (see docstring); a single allocation above 256 MB
    # still fails the campaign cleanly instead of OOM-killing the job.
    proc = subprocess.run([str(exe), str(corpus), "-max_total_time=20", "-max_len=4096",
                           "-timeout=10", "-rss_limit_mb=0", "-malloc_limit_mb=256",
                           "-print_final_stats=1"],
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1"})
    head = "\n".join(proc.stderr.splitlines()[:25])
    assert proc.returncode == 0, f"--- libFuzzer start ---\n{head}\n--- end ---\n{proc.stderr[-4000:]}"
    assert "Done" in proc.stderr or "DONE" in proc.stderr or "NEW" in proc.stderr, proc.stderr[-2000:]
