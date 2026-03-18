"""Cloud provider credential management.

Each ``CloudProvider`` subclass knows how to write and delete the
credential file that SkyPilot expects for a specific cloud, and how to
set the corresponding environment variables.

Adding a new provider is a one-class extension — no changes to
``CloudLauncher`` required.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class CloudProvider(ABC):
    """Base class for cloud provider credential and identity handling."""

    @abstractmethod
    def write_credentials(self, creds: dict) -> None:
        """Write credentials to the provider-specific file path.

        Creates parent directories if needed.
        """
        ...

    @abstractmethod
    def delete_credentials(self) -> None:
        """Remove the credential file written by ``write_credentials()``.

        No-op if the file doesn't exist.
        """
        ...

    @abstractmethod
    def credential_file_path(self) -> Path:
        """Return the absolute path to the credential file SkyPilot expects."""
        ...

    @abstractmethod
    def skypilot_cloud_name(self) -> str:
        """Return the string SkyPilot uses to identify this cloud."""
        ...

    @abstractmethod
    def validate_creds_dict(self, creds: dict) -> None:
        """Raise ``ValueError`` if required keys are missing from *creds*."""
        ...

    def set_env_vars(self, creds: dict) -> dict[str, Optional[str]]:
        """Set provider-specific env vars.  Returns previous values.

        Previous values are ``None`` if the var was not set before.
        Subclasses override to set provider-specific vars.
        """
        return {}

    def restore_env_vars(self, saved: dict[str, Optional[str]]) -> None:
        """Restore env vars saved by ``set_env_vars()``."""
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class RunPodProvider(CloudProvider):
    """RunPod credential handling.

    Credential file: ``~/.runpod/config.toml``
    Format: TOML with ``[default]`` profile containing ``api_key``.

    Also sets ``RUNPOD_API_KEY`` env var.  Both the file and the env var
    are required: SkyPilot's credential validator checks the file
    (``sky/clouds/runpod.py:361``), and the adaptor layer falls back to
    the env var for actual API calls (``sky/adaptors/runpod.py:26``).

    ``gpu_type`` values are SkyPilot accelerator names for the RunPod
    catalog.  Common examples::

        RTXA4000, A40, RTX4090, L4, A100-80GB, H100-SXM

    Run ``sky show-gpus --cloud runpod --all`` to list all types with
    pricing.
    """

    def credential_file_path(self) -> Path:
        return Path.home() / ".runpod" / "config.toml"

    def skypilot_cloud_name(self) -> str:
        return "runpod"

    def validate_creds_dict(self, creds: dict) -> None:
        if "api_key" not in creds:
            raise ValueError(
                "RunPod credentials must contain 'api_key'. "
                "Get yours at https://www.runpod.io/console/user/settings"
            )

    def write_credentials(self, creds: dict) -> None:
        self.validate_creds_dict(creds)
        path = self.credential_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'[default]\napi_key = "{creds["api_key"]}"\n',
            encoding="utf-8",
        )

    def delete_credentials(self) -> None:
        path = self.credential_file_path()
        if path.exists():
            path.unlink()

    def set_env_vars(self, creds: dict) -> dict[str, Optional[str]]:
        saved = {"RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY")}
        os.environ["RUNPOD_API_KEY"] = creds["api_key"]
        return saved


class LambdaLabsProvider(CloudProvider):
    """Lambda Labs credential handling.

    **STUB** — credential write/delete implemented, but launch has not
    been validated end-to-end.

    Credential file: ``~/.lambda_cloud/lambda_keys``
    Format: plain text, ``api_key = YOUR_KEY`` (one entry per line).

    ``gpu_type`` values are SkyPilot accelerator names for the Lambda
    catalog.  Common examples::

        A100-80GB, H100-SXM, A10

    Run ``sky show-gpus --cloud lambda`` to list all available types.
    """

    def credential_file_path(self) -> Path:
        return Path.home() / ".lambda_cloud" / "lambda_keys"

    def skypilot_cloud_name(self) -> str:
        return "lambda"

    def validate_creds_dict(self, creds: dict) -> None:
        if "api_key" not in creds:
            raise ValueError(
                "Lambda Labs credentials must contain 'api_key'. "
                "Get yours at https://cloud.lambdalabs.com/api-keys"
            )

    def write_credentials(self, creds: dict) -> None:
        self.validate_creds_dict(creds)
        path = self.credential_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'api_key = {creds["api_key"]}\n',
            encoding="utf-8",
        )

    def delete_credentials(self) -> None:
        path = self.credential_file_path()
        if path.exists():
            path.unlink()

    def set_env_vars(self, creds: dict) -> dict[str, Optional[str]]:
        # Lambda SDK doesn't use an env var — credentials are file-only.
        return {}


PROVIDERS: dict[str, CloudProvider] = {
    "runpod": RunPodProvider(),
    "lambda_labs": LambdaLabsProvider(),
}
