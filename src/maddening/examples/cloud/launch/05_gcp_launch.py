#!/usr/bin/env python3
"""Launch a MADDENING job on Google Cloud Platform via SkyPilot.

Uses the v0.2 #7 ``GCPProvider`` to materialise credentials in
``~/.config/gcloud/application_default_credentials.json``.

Usage:
    python 05_gcp_launch.py
    python 05_gcp_launch.py --job job_config.example.yaml --dry-run

Credentials YAML structure expected
(``~/.maddening/cloud_credentials.yaml``).  Provide ONE of:

    # Service-account key JSON (recommended for headless / CI)
    gcp:
      service_account_json:
        type: "service_account"
        project_id: "..."
        private_key: "-----BEGIN PRIVATE KEY-----\\n...\\n-----END..."
        client_email: "..."
        ...
      project_id: "my-cloud-project"      # optional override

    # OR user credentials from `gcloud auth application-default login`
    gcp:
      application_default_credentials:
        client_id: "..."
        client_secret: "..."
        refresh_token: "..."
      project_id: "my-cloud-project"

Requires:
    - pip install "skypilot[gcp]"
    - A GCP project with Compute Engine enabled.
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
    parser = argparse.ArgumentParser(description="Launch on GCP")
    parser.add_argument("--job", default="job_config.example.yaml")
    parser.add_argument("--creds", default=None)
    parser.add_argument("--dry-run", action="store_true")
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

        print(f"\nLaunching {'(dry run)' if args.dry_run else ''}...")
        job = launcher.launch(job_path, dry_run=args.dry_run)
        print(f"  Cluster: {job.cluster_name}")
        if args.dry_run:
            return

        print("\nStreaming logs (Ctrl+C to detach)...")
        try:
            job.stream_logs()
        except KeyboardInterrupt:
            pass
        print(f"\nJob phase: {job.phase.value}  IP={job.vm_ip}")

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
