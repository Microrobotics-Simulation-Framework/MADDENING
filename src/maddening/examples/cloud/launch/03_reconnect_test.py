#!/usr/bin/env python3
"""Test CloudJob.from_cluster_name() reconnect.

Launches a VM, saves the cluster name, creates a NEW CloudJob handle
from just the name (simulating a script restart), verifies status()
still works, then tears down.

Usage:
    python 03_reconnect_test.py
"""

import sys
import time

from maddening.cloud.launcher import (
    CloudJob,
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


def main():
    config = JobConfig(
        provider="runpod",
        gpu_type="RTX5090",
        use_spot=False,  # on-demand for reliable availability
        cost=CostPolicy(
            max_cost_per_hour=2.0,
            max_total_budget=5.0,
            autostop_minutes=5,
            auto_teardown=True,
        ),
    )

    launcher = CloudLauncher()

    # --- Phase 1: Launch and get cluster name ---
    print("Phase 1: Launching...")
    try:
        job = launcher.launch(config)
    except LaunchError as e:
        print(f"Launch failed: {e}")
        sys.exit(1)

    cluster_name = job.cluster_name
    vm_ip = job.vm_ip
    print(f"  Cluster: {cluster_name}")
    print(f"  VM IP: {vm_ip}")
    print(f"  Phase: {job.phase.value}")

    # --- Phase 2: Simulate script restart ---
    print()
    print("Phase 2: Simulating restart (creating new handle from cluster name)...")
    del job  # throw away the original handle

    # Reconnect with credentials (enables teardown)
    job2 = CloudJob.from_cluster_name(
        cluster_name,
        credentials_path="~/.maddening/cloud_credentials.yaml",
    )
    print(f"  Reconnected to: {job2.cluster_name}")
    print(f"  Phase: {job2.phase.value}")

    # --- Phase 3: Verify status works ---
    print()
    print("Phase 3: Status check on reconnected handle...")
    status = job2.status()
    print(f"  Status: {status}")
    assert status["cluster_status"] == "UP", f"Expected UP, got {status['cluster_status']}"
    assert status["vm_ip"] is not None, "Expected VM IP"
    print(f"  VM IP from status: {status['vm_ip']}")
    print("  PASS: status() works on reconnected handle")

    # --- Phase 4: Teardown via reconnected handle ---
    print()
    print("Phase 4: Tearing down via reconnected handle...")
    job2.teardown()
    print(f"  Phase after teardown: {job2.phase.value}")
    assert job2.is_done(), "Expected is_done() == True after teardown"
    print("  PASS: teardown works on reconnected handle")

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
