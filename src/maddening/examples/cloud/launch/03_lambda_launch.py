#!/usr/bin/env python3
"""Launch a MADDENING job on Lambda Labs, stream logs, then tear down.

This example mirrors ``02_runpod_launch.py`` but selects the
``lambda_labs`` provider via the job config.

Usage:
    python 03_lambda_launch.py
    python 03_lambda_launch.py --job job_config.example.yaml
    python 03_lambda_launch.py --dry-run

Requires:
    - ~/.maddening/cloud_credentials.yaml with your Lambda Labs API key
    - pip install "skypilot[lambda]"

Set the job config's ``provider:`` field to ``lambda_labs`` (or use
``--cloud lambda`` on the CLI; see SkyPilot docs).
"""

import argparse
import sys
from pathlib import Path

from maddening.cloud.launcher import (
    CloudLauncher,
    CostLimitError,
    CredentialError,
    LaunchError,
)


def main():
    parser = argparse.ArgumentParser(description="Launch on Lambda Labs")
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

        print("Validating configuration...")
        result = launcher.validate(job_path)
        print(f"  Instance: {result['instance_type']}")
        print(f"  Hourly cost: ${result['hourly_cost']:.2f}")
        print(f"  Budget remaining: ${result['budget_remaining']:.2f}")

        print(f"\nLaunching {'(dry run)' if args.dry_run else ''}...")
        job = launcher.launch(job_path, dry_run=args.dry_run)
        print(f"  Cluster: {job.cluster_name}")
        print(f"  Phase: {job.phase.value}")

        if args.dry_run:
            print("\nDry run complete — no resources provisioned.")
            return

        print("\nStreaming logs (Ctrl+C to detach)...")
        try:
            job.stream_logs()
        except KeyboardInterrupt:
            print("\nDetached from logs.")

        print(f"\nJob phase: {job.phase.value}")
        print(f"VM IP: {job.vm_ip}")
        print(f"Status: {job.status()}")

    except CredentialError as exc:
        print(f"Credential error: {exc}")
        sys.exit(2)
    except CostLimitError as exc:
        print(f"Cost guard tripped: {exc}")
        sys.exit(3)
    except LaunchError as exc:
        print(f"Launch failed: {exc}")
        sys.exit(4)


if __name__ == "__main__":
    main()
