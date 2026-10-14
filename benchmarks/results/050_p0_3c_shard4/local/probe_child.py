"""Scratch probe (not committed): one checklist child through the harness, its stderr kept."""
import os
from pathlib import Path

from tests.cloud.multigpu.run_pod_support import run_pod
from tests.cloud.multigpu.test_run_pod_seeded_faults import _RUNNER, _N_DEV, _scratch_library

OUT = Path(os.environ["PROBE_OUT"])


def test_probe(tmp_path):
    src = _scratch_library(tmp_path, None)
    goal = os.environ.get("PROBE_GOAL", "checklist")
    proc = run_pod(_RUNNER, ["--goal", goal, "--dry-run", "--keep-going", "--cells", "256",
                             "--out", tmp_path / "out"],
                   pythonpath=str(src), timeout=1800, n_devices=_N_DEV)
    OUT.write_text(proc.stdout + "\n=====STDERR=====\n" + proc.stderr)
    assert proc.returncode == 0
