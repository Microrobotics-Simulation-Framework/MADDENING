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

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.EVOLVING)
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


class AWSProvider(CloudProvider):
    """AWS credential handling (v0.2 #7).

    Credential file: ``~/.aws/credentials``
    Format: INI-style with one section per profile.  By default the
    provider writes the ``[default]`` profile; pass a different name
    via ``profile`` to write a named section that AWS_PROFILE selects.

    Required keys in *creds*::

        aws_access_key_id
        aws_secret_access_key

    Optional::

        aws_session_token   — STS temporary credentials
        region              — written to ``~/.aws/config`` as the
                              default region for the profile

    Also sets ``AWS_PROFILE`` (so subsequent ``boto3`` /
    ``aws cli`` / SkyPilot calls in the same process pick the right
    section) and, when supplied, ``AWS_DEFAULT_REGION``.

    ``gpu_type`` values are SkyPilot accelerator names for the AWS
    catalog (e.g. ``A100``, ``H100``, ``A10G``, ``T4``, ``V100``).
    Run ``sky show-gpus --cloud aws`` for the live list.
    """

    def __init__(self, profile: str = "default"):
        self._profile = profile

    def credential_file_path(self) -> Path:
        return Path.home() / ".aws" / "credentials"

    def config_file_path(self) -> Path:
        return Path.home() / ".aws" / "config"

    def skypilot_cloud_name(self) -> str:
        return "aws"

    def validate_creds_dict(self, creds: dict) -> None:
        missing = [
            k for k in ("aws_access_key_id", "aws_secret_access_key")
            if k not in creds
        ]
        if missing:
            raise ValueError(
                f"AWS credentials must contain {missing}. "
                "Get them from the IAM console under 'Security credentials'."
            )

    def _emit_credentials_block(self, creds: dict) -> str:
        lines = [
            f"[{self._profile}]",
            f"aws_access_key_id = {creds['aws_access_key_id']}",
            f"aws_secret_access_key = {creds['aws_secret_access_key']}",
        ]
        if "aws_session_token" in creds:
            lines.append(f"aws_session_token = {creds['aws_session_token']}")
        lines.append("")
        return "\n".join(lines)

    def _emit_config_block(self, region: str) -> str:
        # The config-file section name is "profile X" except for default.
        section = (
            "default" if self._profile == "default"
            else f"profile {self._profile}"
        )
        return f"[{section}]\nregion = {region}\n"

    def write_credentials(self, creds: dict) -> None:
        self.validate_creds_dict(creds)
        creds_path = self.credential_file_path()
        creds_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge with any existing profiles: append-or-replace this one
        # section, leave the rest of the file alone.
        existing = (
            creds_path.read_text(encoding="utf-8") if creds_path.exists() else ""
        )
        new_block = self._emit_credentials_block(creds)
        merged = _replace_or_append_ini_section(
            existing, self._profile, new_block,
        )
        creds_path.write_text(merged, encoding="utf-8")
        creds_path.chmod(0o600)

        if "region" in creds:
            config_path = self.config_file_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            existing_cfg = (
                config_path.read_text(encoding="utf-8")
                if config_path.exists() else ""
            )
            section_name = (
                "default" if self._profile == "default"
                else f"profile {self._profile}"
            )
            new_cfg = self._emit_config_block(creds["region"])
            merged_cfg = _replace_or_append_ini_section(
                existing_cfg, section_name, new_cfg,
            )
            config_path.write_text(merged_cfg, encoding="utf-8")

    def delete_credentials(self) -> None:
        creds_path = self.credential_file_path()
        if creds_path.exists():
            existing = creds_path.read_text(encoding="utf-8")
            cleaned = _delete_ini_section(existing, self._profile)
            if cleaned.strip():
                creds_path.write_text(cleaned, encoding="utf-8")
            else:
                creds_path.unlink()

        config_path = self.config_file_path()
        if config_path.exists():
            section_name = (
                "default" if self._profile == "default"
                else f"profile {self._profile}"
            )
            existing_cfg = config_path.read_text(encoding="utf-8")
            cleaned_cfg = _delete_ini_section(existing_cfg, section_name)
            if cleaned_cfg.strip():
                config_path.write_text(cleaned_cfg, encoding="utf-8")
            else:
                config_path.unlink()

    def set_env_vars(self, creds: dict) -> dict[str, Optional[str]]:
        saved = {"AWS_PROFILE": os.environ.get("AWS_PROFILE")}
        os.environ["AWS_PROFILE"] = self._profile
        if "region" in creds:
            saved["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION")
            os.environ["AWS_DEFAULT_REGION"] = creds["region"]
        return saved


class GCPProvider(CloudProvider):
    """GCP credential handling (v0.2 #7).

    Credential file:
        ``~/.config/gcloud/application_default_credentials.json``

    Format: JSON containing one of two service-account / user
    credential shapes.  The provider writes whichever the caller
    supplies — typically a service-account key JSON or the output of
    ``gcloud auth application-default login``.

    Required keys in *creds* — either of:

        service_account_json : dict
            Full service-account key JSON (the file you download from
            the IAM console).  Stored verbatim.

        application_default_credentials : dict
            User credentials shape (``client_id`` + ``client_secret``
            + ``refresh_token``).  Stored verbatim.

    Optional::

        project_id : str    — written to ``GOOGLE_CLOUD_PROJECT`` env
                              var and used by SkyPilot.

    Sets ``GOOGLE_APPLICATION_CREDENTIALS`` to the file path and
    ``GOOGLE_CLOUD_PROJECT`` to the project id when supplied.

    Run ``sky show-gpus --cloud gcp`` for available accelerators.
    """

    def credential_file_path(self) -> Path:
        return (
            Path.home()
            / ".config" / "gcloud"
            / "application_default_credentials.json"
        )

    def skypilot_cloud_name(self) -> str:
        return "gcp"

    def validate_creds_dict(self, creds: dict) -> None:
        if not (
            "service_account_json" in creds
            or "application_default_credentials" in creds
        ):
            raise ValueError(
                "GCP credentials must contain either "
                "'service_account_json' (recommended for headless / CI) "
                "or 'application_default_credentials' (from "
                "`gcloud auth application-default login`). "
                "Get a service-account key at "
                "https://console.cloud.google.com/iam-admin/serviceaccounts"
            )

    def write_credentials(self, creds: dict) -> None:
        import json as _json
        self.validate_creds_dict(creds)
        path = self.credential_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = creds.get("service_account_json") or creds.get(
            "application_default_credentials",
        )
        path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        path.chmod(0o600)

    def delete_credentials(self) -> None:
        path = self.credential_file_path()
        if path.exists():
            path.unlink()

    def set_env_vars(self, creds: dict) -> dict[str, Optional[str]]:
        saved = {
            "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS",
            ),
        }
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            self.credential_file_path()
        )
        if "project_id" in creds:
            saved["GOOGLE_CLOUD_PROJECT"] = os.environ.get(
                "GOOGLE_CLOUD_PROJECT",
            )
            os.environ["GOOGLE_CLOUD_PROJECT"] = str(creds["project_id"])
        return saved


# ---------------------------------------------------------------------------
# Helpers for INI-file merge/delete (AWS-style credentials)
# ---------------------------------------------------------------------------


def _replace_or_append_ini_section(text: str, section: str, new_block: str) -> str:
    """Replace ``[section]`` in *text* (preserving other sections) or
    append ``new_block`` to the end if it doesn't exist.

    ``new_block`` is expected to start with ``[section]``.
    """
    sections = _split_ini_sections(text)
    if section in sections:
        sections[section] = new_block.rstrip("\n") + "\n"
    else:
        sections[section] = new_block.rstrip("\n") + "\n"
    out_parts: list[str] = []
    for name, block in sections.items():
        out_parts.append(block.rstrip("\n"))
    return "\n".join(out_parts) + "\n"


def _delete_ini_section(text: str, section: str) -> str:
    sections = _split_ini_sections(text)
    sections.pop(section, None)
    if not sections:
        return ""
    out_parts = [b.rstrip("\n") for b in sections.values()]
    return "\n".join(out_parts) + "\n"


def _split_ini_sections(text: str) -> dict[str, str]:
    """Crude INI parser: returns ``{section_name: full_block_text}``.

    Order-preserving (Python 3.7+ dict).  Comment lines / blank lines
    before the first section are dropped (they shouldn't exist in
    well-formed AWS files).
    """
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(buf) + "\n"
            current = stripped[1:-1].strip()
            buf = [line]
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf) + "\n"
    return sections


PROVIDERS: dict[str, CloudProvider] = {
    "runpod": RunPodProvider(),
    "lambda_labs": LambdaLabsProvider(),
    "aws": AWSProvider(),
    "gcp": GCPProvider(),
}
