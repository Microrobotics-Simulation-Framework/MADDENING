"""Scratch probe (not committed): the control checklist child with extra XLA flags, through the harness."""
import os
import time
from pathlib import Path

from tests.cloud.multigpu import run_pod_support as rps
from tests.cloud.multigpu.test_run_pod_seeded_faults import _RUNNER, _N_DEV, _scratch_library

OUT = Path(os.environ["PROBE_OUT"])


def test_probe(tmp_path, monkeypatch):
    extra = os.environ.get("PROBE_XLA", "")
    real = rps.offline_env

    def env(pythonpath, n_devices=4):
        e = real(pythonpath, n_devices)
        e["XLA_FLAGS"] = (e["XLA_FLAGS"] + " " + extra).strip()
        return e

    monkeypatch.setattr(rps, "offline_env", env)
    src = _scratch_library(tmp_path, None)
    t0 = time.perf_counter()
    proc = rps.run_pod(_RUNNER, ["--goal", "checklist", "--dry-run", "--keep-going", "--cells", "256",
                                 "--out", tmp_path / "out"],
                       pythonpath=str(src), timeout=1800, n_devices=_N_DEV)
    OUT.write_text(f"wall {time.perf_counter() - t0:.1f} rc {proc.returncode} flags {extra!r}\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
    assert proc.returncode == 0
