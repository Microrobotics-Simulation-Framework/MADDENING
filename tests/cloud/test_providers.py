"""Tests for cloud provider credential management."""

import os
from pathlib import Path

import pytest

from maddening.cloud.providers import (
    PROVIDERS,
    CloudProvider,
    LambdaLabsProvider,
    RunPodProvider,
)
from maddening.cloud.launcher import _credential_context


# -- Test-isolated providers that write to tmp_path -------------------

class _TmpRunPodProvider(RunPodProvider):
    """RunPodProvider that writes to a temp directory."""

    def __init__(self, base: Path):
        self._base = base

    def credential_file_path(self) -> Path:
        return self._base / ".runpod" / "config.toml"


class _TmpLambdaLabsProvider(LambdaLabsProvider):
    """LambdaLabsProvider that writes to a temp directory."""

    def __init__(self, base: Path):
        self._base = base

    def credential_file_path(self) -> Path:
        return self._base / ".lambda_cloud" / "lambda_keys"


# -- RunPodProvider tests ---------------------------------------------

class TestRunPodProvider:
    def test_write_credentials(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        prov.write_credentials({"api_key": "rp_test123"})
        path = prov.credential_file_path()
        assert path.exists()
        content = path.read_text()
        assert "[default]" in content
        assert 'api_key = "rp_test123"' in content

    def test_write_creates_parent_dirs(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        prov.write_credentials({"api_key": "rp_x"})
        assert prov.credential_file_path().parent.is_dir()

    def test_delete_credentials(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        prov.write_credentials({"api_key": "rp_x"})
        assert prov.credential_file_path().exists()
        prov.delete_credentials()
        assert not prov.credential_file_path().exists()

    def test_delete_no_file_is_noop(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        prov.delete_credentials()  # should not raise

    def test_validate_missing_api_key(self):
        prov = RunPodProvider()
        with pytest.raises(ValueError, match="api_key"):
            prov.validate_creds_dict({})

    def test_validate_success(self):
        prov = RunPodProvider()
        prov.validate_creds_dict({"api_key": "rp_xxx"})  # no exception

    def test_skypilot_cloud_name(self):
        assert RunPodProvider().skypilot_cloud_name() == "runpod"

    def test_set_env_vars(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        prov = RunPodProvider()
        saved = prov.set_env_vars({"api_key": "rp_env_test"})
        assert os.environ["RUNPOD_API_KEY"] == "rp_env_test"
        assert saved["RUNPOD_API_KEY"] is None  # was not set before
        prov.restore_env_vars(saved)
        assert "RUNPOD_API_KEY" not in os.environ

    def test_set_env_vars_preserves_existing(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "original")
        prov = RunPodProvider()
        saved = prov.set_env_vars({"api_key": "new_key"})
        assert os.environ["RUNPOD_API_KEY"] == "new_key"
        assert saved["RUNPOD_API_KEY"] == "original"
        prov.restore_env_vars(saved)
        assert os.environ["RUNPOD_API_KEY"] == "original"


# -- LambdaLabsProvider tests -----------------------------------------

class TestLambdaLabsProvider:
    def test_write_credentials(self, tmp_path):
        prov = _TmpLambdaLabsProvider(tmp_path)
        prov.write_credentials({"api_key": "ll_test456"})
        path = prov.credential_file_path()
        assert path.exists()
        content = path.read_text()
        assert "api_key = ll_test456" in content

    def test_delete_credentials(self, tmp_path):
        prov = _TmpLambdaLabsProvider(tmp_path)
        prov.write_credentials({"api_key": "ll_x"})
        prov.delete_credentials()
        assert not prov.credential_file_path().exists()

    def test_delete_no_file_is_noop(self, tmp_path):
        prov = _TmpLambdaLabsProvider(tmp_path)
        prov.delete_credentials()

    def test_validate_missing_api_key(self):
        prov = LambdaLabsProvider()
        with pytest.raises(ValueError, match="api_key"):
            prov.validate_creds_dict({})

    def test_skypilot_cloud_name(self):
        assert LambdaLabsProvider().skypilot_cloud_name() == "lambda"


# -- Credential context manager tests ---------------------------------

class TestCredentialContext:
    def test_writes_and_deletes(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        creds = {"api_key": "rp_ctx"}
        with _credential_context(prov, creds):
            assert prov.credential_file_path().exists()
        assert not prov.credential_file_path().exists()

    def test_cleanup_on_exception(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        creds = {"api_key": "rp_exc"}
        with pytest.raises(RuntimeError):
            with _credential_context(prov, creds):
                assert prov.credential_file_path().exists()
                raise RuntimeError("boom")
        assert not prov.credential_file_path().exists()

    def test_preserves_preexisting_file(self, tmp_path):
        prov = _TmpRunPodProvider(tmp_path)
        # Pre-create the file (user-managed)
        prov.write_credentials({"api_key": "rp_preexisting"})
        assert prov.credential_file_path().exists()
        with _credential_context(prov, {"api_key": "rp_new"}):
            # File still exists (we didn't overwrite)
            assert prov.credential_file_path().exists()
            content = prov.credential_file_path().read_text()
            assert "rp_preexisting" in content  # original preserved
        # File NOT deleted (we didn't write it)
        assert prov.credential_file_path().exists()

    def test_env_var_restored_after_exit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        prov = _TmpRunPodProvider(tmp_path)
        creds = {"api_key": "rp_env"}
        with _credential_context(prov, creds):
            assert os.environ.get("RUNPOD_API_KEY") == "rp_env"
        assert "RUNPOD_API_KEY" not in os.environ


# -- PROVIDERS registry -----------------------------------------------

class TestProvidersRegistry:
    def test_contains_runpod(self):
        assert "runpod" in PROVIDERS
        assert isinstance(PROVIDERS["runpod"], RunPodProvider)

    def test_contains_lambda_labs(self):
        assert "lambda_labs" in PROVIDERS
        assert isinstance(PROVIDERS["lambda_labs"], LambdaLabsProvider)

    def test_contains_aws(self):
        from maddening.cloud.providers import AWSProvider
        assert "aws" in PROVIDERS
        assert isinstance(PROVIDERS["aws"], AWSProvider)

    def test_contains_gcp(self):
        from maddening.cloud.providers import GCPProvider
        assert "gcp" in PROVIDERS
        assert isinstance(PROVIDERS["gcp"], GCPProvider)

    def test_all_four_providers_registered(self):
        assert set(PROVIDERS.keys()) == {"runpod", "lambda_labs", "aws", "gcp"}


# ---------------------------------------------------------------------------
# AWSProvider (v0.2 #7)
# ---------------------------------------------------------------------------


class _TmpAWSProvider:
    """AWS provider variant that writes credentials/config under tmp_path."""

    def __new__(cls, base, profile="default"):
        from maddening.cloud.providers import AWSProvider
        inst = AWSProvider.__new__(AWSProvider)
        inst._profile = profile
        inst._base = base
        inst.credential_file_path = lambda: base / ".aws" / "credentials"
        inst.config_file_path = lambda: base / ".aws" / "config"
        return inst


class TestAWSProviderCredentialFile:
    def test_writes_default_profile(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "AKIA_TEST_ID",
            "aws_secret_access_key": "secret/example",
        })
        path = prov.credential_file_path()
        content = path.read_text()
        assert "[default]" in content
        assert "aws_access_key_id = AKIA_TEST_ID" in content
        assert "aws_secret_access_key = secret/example" in content

    def test_writes_named_profile(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path, profile="dev")
        prov.write_credentials({
            "aws_access_key_id": "AKIA_DEV",
            "aws_secret_access_key": "dev_secret",
        })
        content = prov.credential_file_path().read_text()
        assert "[dev]" in content
        assert "[default]" not in content

    def test_writes_session_token(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "AKIA_STS",
            "aws_secret_access_key": "s",
            "aws_session_token": "token-string",
        })
        assert "aws_session_token = token-string" in prov.credential_file_path().read_text()

    def test_writes_region_to_config_file(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "A",
            "aws_secret_access_key": "S",
            "region": "us-west-2",
        })
        cfg = prov.config_file_path().read_text()
        assert "[default]" in cfg  # default profile uses bare name
        assert "region = us-west-2" in cfg

    def test_writes_region_to_named_config_section(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path, profile="dev")
        prov.write_credentials({
            "aws_access_key_id": "A",
            "aws_secret_access_key": "S",
            "region": "eu-west-1",
        })
        cfg = prov.config_file_path().read_text()
        assert "[profile dev]" in cfg
        assert "region = eu-west-1" in cfg

    def test_preserves_other_profiles_when_merging(self, tmp_path):
        prov_a = _TmpAWSProvider(tmp_path, profile="prod")
        prov_a.write_credentials({
            "aws_access_key_id": "PROD_KEY",
            "aws_secret_access_key": "p",
        })
        prov_b = _TmpAWSProvider(tmp_path, profile="dev")
        prov_b.write_credentials({
            "aws_access_key_id": "DEV_KEY",
            "aws_secret_access_key": "d",
        })
        content = prov_b.credential_file_path().read_text()
        assert "PROD_KEY" in content
        assert "DEV_KEY" in content
        assert content.count("[prod]") == 1
        assert content.count("[dev]") == 1

    def test_overwriting_same_profile_does_not_duplicate(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "A1", "aws_secret_access_key": "s",
        })
        prov.write_credentials({
            "aws_access_key_id": "A2", "aws_secret_access_key": "s",
        })
        content = prov.credential_file_path().read_text()
        assert "A1" not in content
        assert "A2" in content
        assert content.count("[default]") == 1

    def test_credential_file_chmod_600(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "A", "aws_secret_access_key": "S",
        })
        mode = prov.credential_file_path().stat().st_mode & 0o777
        assert mode == 0o600


class TestAWSProviderValidation:
    def test_missing_access_key_raises(self):
        from maddening.cloud.providers import AWSProvider
        with pytest.raises(ValueError, match="aws_access_key_id"):
            AWSProvider().validate_creds_dict({"aws_secret_access_key": "s"})

    def test_missing_secret_raises(self):
        from maddening.cloud.providers import AWSProvider
        with pytest.raises(ValueError, match="aws_secret_access_key"):
            AWSProvider().validate_creds_dict({"aws_access_key_id": "k"})

    def test_complete_creds_passes(self):
        from maddening.cloud.providers import AWSProvider
        AWSProvider().validate_creds_dict({
            "aws_access_key_id": "k", "aws_secret_access_key": "s",
        })

    def test_skypilot_cloud_name(self):
        from maddening.cloud.providers import AWSProvider
        assert AWSProvider().skypilot_cloud_name() == "aws"


class TestAWSProviderDelete:
    def test_delete_removes_profile_only(self, tmp_path):
        prov_a = _TmpAWSProvider(tmp_path, profile="prod")
        prov_a.write_credentials({
            "aws_access_key_id": "p", "aws_secret_access_key": "s",
        })
        prov_b = _TmpAWSProvider(tmp_path, profile="dev")
        prov_b.write_credentials({
            "aws_access_key_id": "d", "aws_secret_access_key": "s",
        })
        prov_b.delete_credentials()
        content = prov_b.credential_file_path().read_text()
        assert "[prod]" in content
        assert "[dev]" not in content

    def test_delete_last_profile_removes_file(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.write_credentials({
            "aws_access_key_id": "k", "aws_secret_access_key": "s",
        })
        prov.delete_credentials()
        assert not prov.credential_file_path().exists()

    def test_delete_no_file_is_noop(self, tmp_path):
        prov = _TmpAWSProvider(tmp_path)
        prov.delete_credentials()


class TestAWSProviderEnvVars:
    def test_set_env_vars_sets_profile(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        prov = _TmpAWSProvider(tmp_path)
        saved = prov.set_env_vars({
            "aws_access_key_id": "k", "aws_secret_access_key": "s",
        })
        assert os.environ["AWS_PROFILE"] == "default"
        assert saved["AWS_PROFILE"] is None
        prov.restore_env_vars(saved)
        assert "AWS_PROFILE" not in os.environ

    def test_set_env_vars_with_region(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        prov = _TmpAWSProvider(tmp_path)
        saved = prov.set_env_vars({
            "aws_access_key_id": "k",
            "aws_secret_access_key": "s",
            "region": "us-east-1",
        })
        assert os.environ["AWS_DEFAULT_REGION"] == "us-east-1"
        prov.restore_env_vars(saved)
        assert "AWS_DEFAULT_REGION" not in os.environ


# ---------------------------------------------------------------------------
# GCPProvider (v0.2 #7)
# ---------------------------------------------------------------------------


class _TmpGCPProvider:
    def __new__(cls, base):
        from maddening.cloud.providers import GCPProvider
        inst = GCPProvider.__new__(GCPProvider)
        inst.credential_file_path = lambda: (
            base / ".config" / "gcloud"
            / "application_default_credentials.json"
        )
        return inst


SERVICE_ACCOUNT_EXAMPLE = {
    "type": "service_account",
    "project_id": "demo-proj",
    "private_key_id": "abc",
    "private_key": "-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----\n",
    "client_email": "test@demo-proj.iam.gserviceaccount.com",
    "client_id": "1234567890",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
}


class TestGCPProvider:
    def test_writes_service_account_json(self, tmp_path):
        import json
        prov = _TmpGCPProvider(tmp_path)
        prov.write_credentials({"service_account_json": SERVICE_ACCOUNT_EXAMPLE})
        path = prov.credential_file_path()
        loaded = json.loads(path.read_text())
        assert loaded["client_email"] == SERVICE_ACCOUNT_EXAMPLE["client_email"]

    def test_writes_adc_dict(self, tmp_path):
        import json
        prov = _TmpGCPProvider(tmp_path)
        adc = {"client_id": "abc", "client_secret": "shh", "refresh_token": "r"}
        prov.write_credentials({"application_default_credentials": adc})
        assert json.loads(prov.credential_file_path().read_text()) == adc

    def test_validate_missing_creds_raises(self):
        from maddening.cloud.providers import GCPProvider
        with pytest.raises(ValueError, match="service_account_json"):
            GCPProvider().validate_creds_dict({})

    def test_credential_file_chmod_600(self, tmp_path):
        prov = _TmpGCPProvider(tmp_path)
        prov.write_credentials({"service_account_json": SERVICE_ACCOUNT_EXAMPLE})
        mode = prov.credential_file_path().stat().st_mode & 0o777
        assert mode == 0o600

    def test_delete_credentials(self, tmp_path):
        prov = _TmpGCPProvider(tmp_path)
        prov.write_credentials({"service_account_json": SERVICE_ACCOUNT_EXAMPLE})
        prov.delete_credentials()
        assert not prov.credential_file_path().exists()

    def test_delete_no_file_is_noop(self, tmp_path):
        prov = _TmpGCPProvider(tmp_path)
        prov.delete_credentials()

    def test_set_env_vars_with_project_id(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        prov = _TmpGCPProvider(tmp_path)
        saved = prov.set_env_vars({
            "service_account_json": SERVICE_ACCOUNT_EXAMPLE,
            "project_id": "my-proj-123",
        })
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"].endswith(
            "application_default_credentials.json"
        )
        assert os.environ["GOOGLE_CLOUD_PROJECT"] == "my-proj-123"
        prov.restore_env_vars(saved)
        assert "GOOGLE_CLOUD_PROJECT" not in os.environ

    def test_set_env_vars_without_project_id(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        prov = _TmpGCPProvider(tmp_path)
        saved = prov.set_env_vars({"service_account_json": SERVICE_ACCOUNT_EXAMPLE})
        assert "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        prov.restore_env_vars(saved)

    def test_skypilot_cloud_name(self):
        from maddening.cloud.providers import GCPProvider
        assert GCPProvider().skypilot_cloud_name() == "gcp"


# ---------------------------------------------------------------------------
# Lazy imports via maddening.cloud package
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_aws_provider_lazy_importable(self):
        from maddening.cloud import AWSProvider as A
        from maddening.cloud.providers import AWSProvider as B
        assert A is B

    def test_gcp_provider_lazy_importable(self):
        from maddening.cloud import GCPProvider as A
        from maddening.cloud.providers import GCPProvider as B
        assert A is B
