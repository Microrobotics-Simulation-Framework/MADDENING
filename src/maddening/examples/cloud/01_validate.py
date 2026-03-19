#!/usr/bin/env python3
"""Validate cloud configuration without spending money.

Parses the job config and credentials, resolves the GPU type to an
instance type, checks cost guards, and prints the result.

Usage:
    python 01_validate.py
    python 01_validate.py --job job_config.example.yaml
    python 01_validate.py --creds ~/.maddening/cloud_credentials.yaml
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
    parser = argparse.ArgumentParser(description="Validate cloud config")
    parser.add_argument(
        "--job", default="job_config.example.yaml",
        help="Path to job config YAML",
    )
    parser.add_argument(
        "--creds", default=None,
        help="Path to credentials YAML (default: ~/.maddening/cloud_credentials.yaml)",
    )
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job config not found: {job_path}")
        sys.exit(1)

    try:
        launcher = CloudLauncher(credentials_path=args.creds)
        result = launcher.validate(job_path)
    except CredentialError as e:
        print(f"Credential error: {e}")
        sys.exit(1)
    except CostLimitError as e:
        print(f"Cost limit exceeded: {e}")
        print(f"  guard_type={e.guard_type}, limit={e.limit}, actual={e.actual}")
        sys.exit(1)
    except LaunchError as e:
        print(f"Launch error: {e}")
        sys.exit(1)

    print("Validation passed:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
