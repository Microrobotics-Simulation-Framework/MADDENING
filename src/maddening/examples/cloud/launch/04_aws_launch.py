#!/usr/bin/env python3
"""Launch a MADDENING job on AWS via SkyPilot.

Uses the v0.2 #7 ``AWSProvider`` to materialise credentials in
``~/.aws/credentials`` for the duration of the launch.

Usage:
    python 04_aws_launch.py
    python 04_aws_launch.py --job job_config.example.yaml --dry-run

Credentials YAML structure expected (``~/.maddening/cloud_credentials.yaml``)::

    aws:
      aws_access_key_id: "AKIA..."
      aws_secret_access_key: "..."
      region: "us-east-1"            # optional; written to ~/.aws/config
      # aws_session_token: "..."     # optional; STS temporary creds
      # profile: "default"           # optional; defaults to "default"

Requires:
    - pip install "skypilot[aws]"
    - An IAM user / role with EC2 launch permissions.
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
    parser = argparse.ArgumentParser(description="Launch on AWS")
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
            return

        print("\nStreaming logs (Ctrl+C to detach)...")
        try:
            job.stream_logs()
        except KeyboardInterrupt:
            print("\nDetached from logs.")
        print(f"\nJob phase: {job.phase.value}  IP={job.vm_ip}")
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
