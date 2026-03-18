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

    def test_exactly_two_providers(self):
        assert set(PROVIDERS.keys()) == {"runpod", "lambda_labs"}
