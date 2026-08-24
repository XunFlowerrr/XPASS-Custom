import os
import subprocess
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.checkpoint_callback import (
    safe_save_checkpoint,
    upload_to_gdrive,
    on_checkpoint_saved,
    FAILED_LOG_PATH,
)


def test_safe_save_checkpoint(tmp_path):
    target_dir = tmp_path / "models" / "art"
    target_file = str(target_dir / "model.pth")

    dummy_state = {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5])}

    # Verify atomic saving
    success = safe_save_checkpoint(dummy_state, target_file)
    assert success is True
    assert os.path.exists(target_file)
    assert not os.path.exists(f"{target_file}.tmp")

    loaded_state = torch.load(target_file, map_location="cpu")
    assert torch.equal(loaded_state["weight"], dummy_state["weight"])
    assert torch.equal(loaded_state["bias"], dummy_state["bias"])


def test_upload_to_gdrive_success(tmp_path):
    dummy_file = tmp_path / "test_model.pth"
    dummy_file.write_bytes(b"dummy model weights data")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        success = upload_to_gdrive(
            local_path=str(dummy_file),
            relative_remote_path="v4_fold1/art/test_model.pth",
            remote_name="Google Drive",
            folder_id="dummy_id",
            max_retries=3,
            backoff_delays=[0.01, 0.01],
        )

        assert success is True
        assert mock_run.call_count == 1
        args, _ = mock_run.call_args
        assert "rclone" in args[0]
        assert str(dummy_file) in args[0]
        assert "Google Drive:v4_fold1/art/test_model.pth" in args[0]


def test_upload_to_gdrive_retry_then_success(tmp_path):
    dummy_file = tmp_path / "test_model.pth"
    dummy_file.write_bytes(b"dummy model weights data")

    with patch("subprocess.run") as mock_run:
        # First 2 attempts fail with network error, 3rd attempt succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="connection timed out"),
            MagicMock(returncode=1, stderr="rate limit exceeded"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        success = upload_to_gdrive(
            local_path=str(dummy_file),
            relative_remote_path="v4_fold1/art/test_model.pth",
            max_retries=5,
            backoff_delays=[0.01, 0.01, 0.01, 0.01, 0.01],
        )

        assert success is True
        assert mock_run.call_count == 3


def test_upload_to_gdrive_all_retries_fail_logs_error(tmp_path):
    dummy_file = tmp_path / "test_model.pth"
    dummy_file.write_bytes(b"dummy model weights data")

    if os.path.exists(FAILED_LOG_PATH):
        os.remove(FAILED_LOG_PATH)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="Fatal network failure")

        success = upload_to_gdrive(
            local_path=str(dummy_file),
            relative_remote_path="v4_fold1/art/test_model.pth",
            max_retries=3,
            backoff_delays=[0.01, 0.01],
        )

        assert success is False
        assert mock_run.call_count == 3
        assert os.path.exists(FAILED_LOG_PATH)

        with open(FAILED_LOG_PATH, "r") as f:
            content = f.read()
        assert str(dummy_file) in content
        assert "v4_fold1/art/test_model.pth" in content

    # Cleanup log file
    if os.path.exists(FAILED_LOG_PATH):
        os.remove(FAILED_LOG_PATH)


def test_on_checkpoint_saved_lifecycle_retention(tmp_path):
    class DummyArgs:
        no_gdrive_upload = False
        delete_local_on_upload = False
        rclone_remote = "Google Drive"
        gdrive_folder_id = "test_folder_id"

    dummy_file = tmp_path / "nima_model.pth"
    dummy_file.write_bytes(b"weights")

    with patch("src.checkpoint_callback.upload_to_gdrive") as mock_upload:
        # 1. Primary model (not temporary, delete_local_on_upload=False)
        mock_upload.return_value = True
        success = on_checkpoint_saved(
            local_path=str(dummy_file),
            dataset_ver="v4_fold1",
            genre="art",
            args=DummyArgs(),
            is_temporary=False,
        )
        assert success is True
        # Local file must be preserved
        assert os.path.exists(dummy_file)

        # 2. Temporary model (is_temporary=True) -> Deleted on upload success
        success_temp = on_checkpoint_saved(
            local_path=str(dummy_file),
            dataset_ver="v4_fold1",
            genre="art",
            args=DummyArgs(),
            is_temporary=True,
        )
        assert success_temp is True
        assert not os.path.exists(dummy_file)


def test_on_checkpoint_saved_failsafe_preservation_on_upload_failure(tmp_path):
    class DummyArgs:
        no_gdrive_upload = False
        delete_local_on_upload = True  # Even if requested to delete

    dummy_file = tmp_path / "critical_model.pth"
    dummy_file.write_bytes(b"weights")

    with patch("src.checkpoint_callback.upload_to_gdrive") as mock_upload:
        # Upload fails
        mock_upload.return_value = False

        success = on_checkpoint_saved(
            local_path=str(dummy_file),
            dataset_ver="v4_fold1",
            genre="art",
            args=DummyArgs(),
            is_temporary=True,  # Even if temporary
        )
        assert success is False
        # FAIL-SAFE: Local file MUST NEVER BE DELETED on upload failure
        assert os.path.exists(dummy_file)
