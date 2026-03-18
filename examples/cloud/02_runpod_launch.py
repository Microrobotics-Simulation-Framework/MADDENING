#!/usr/bin/env python3
"""Launch a MADDENING job on RunPod, stream logs, then tear down.

Usage:
    python 02_runpod_launch.py
    python 02_runpod_launch.py --job job_config.example.yaml
    python 02_runpod_launch.py --dry-run

Requires:
    - ~/.maddening/cloud_credentials.yaml with your RunPod API key
    - pip install "skypilot[runpod]"
"""

import argparse
import sys
import time
from pathlib import Path

from maddening.cloud.launcher import (
    CloudLauncher,
    CostLimitError,
    CredentialError,
    LaunchError,
)


def main():
    parser = argparse.ArgumentParser(description="Launch on RunPod")
    parser.add_argument(
        "--job", default="job_config.example.yaml",
        help="Path to job config YAML",
    )
    parser.add_argument(
        "--creds", default=None,
        help="Path to credentials YAML",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and resolve resources without provisioning",
    )
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job config not found: {job_path}")
        sys.exit(1)

    try:
        launcher = CloudLauncher(credentials_path=args.creds)

        # Validate first
        print("Validating configuration...")
        result = launcher.validate(job_path)
        print(f"  Instance: {result['instance_type']}")
        print(f"  Hourly cost: ${result['hourly_cost']:.2f}")
        print(f"  Budget used: ${result['budget_used']:.2f}")
        print(f"  Budget remaining: ${result['budget_remaining']:.2f}")

        # Launch
        print(f"\nLaunching {'(dry run)' if args.dry_run else ''}...")
        job = launcher.launch(job_path, dry_run=args.dry_run)
        print(f"  Cluster: {job.cluster_name}")
        print(f"  Phase: {job.phase.value}")

        if args.dry_run:
            print("\nDry run complete — no resources provisioned.")
            return

        # Stream logs (blocks until provisioning + job start)
        print("\nStreaming logs (Ctrl+C to detach)...")
        try:
            job.stream_logs()
        except KeyboardInterrupt:
            print("\nDetached from logs.")

        # Check status
        print(f"\nJob phase: {job.phase.value}")
        print(f"VM IP: {job.vm_ip}")
        status = job.status()
        print(f"Status: {status}")
        print(f"Cost so far: ${job.cost_so_far():.2f}")

        # Prompt before teardown
        input("\nPress Enter to tear down the cluster...")
        print("Tearing down...")
        job.teardown()
        print("Done.")

    except CredentialError as e:
        print(f"Credential error: {e}")
        sys.exit(1)
    except CostLimitError as e:
        print(f"Cost limit: {e}")
        sys.exit(1)
    except LaunchError as e:
        print(f"Launch error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
