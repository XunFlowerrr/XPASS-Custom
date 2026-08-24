import os
import time
import subprocess
from datetime import datetime
from typing import Optional, List, Any
import torch

DEFAULT_REMOTE = "Google Drive"
DEFAULT_FOLDER_ID = "1WfoO2zszob9rAe7ya8CkmBt6Ci7p070L"
FAILED_LOG_PATH = "failed_uploads.log"


def safe_save_checkpoint(state_dict: Any, file_path: str) -> bool:
    """Safely and atomically save a PyTorch model state dict to disk.

    Writes to a temporary file (.tmp) first, validates integrity, and performs
    an atomic rename to prevent file corruption in case of interruptions.
    """
    if not file_path:
        return False

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = f"{file_path}.tmp"

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        torch.save(state_dict, tmp_path)

        # Integrity verification
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise IOError(f"Saved temporary file {tmp_path} is missing or empty.")

        os.replace(tmp_path, file_path)
        return True
    except Exception as e:
        print(f"❌ [Save Error] Failed to safely save checkpoint to {file_path}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


def upload_to_gdrive(
    local_path: str,
    relative_remote_path: str,
    remote_name: str = DEFAULT_REMOTE,
    folder_id: str = DEFAULT_FOLDER_ID,
    max_retries: int = 5,
    backoff_delays: Optional[List[float]] = None,
) -> bool:
    """Upload a file to Google Drive using rclone with aggressive retry and exponential backoff.

    Args:
        local_path: Absolute or relative local path to the file.
        relative_remote_path: Destination path relative to the Google Drive root folder.
        remote_name: rclone remote name.
        folder_id: Target Google Drive folder ID.
        max_retries: Maximum number of retry attempts (default: 5).
        backoff_delays: List of sleep durations between attempts.

    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    if not os.path.exists(local_path):
        print(f"❌ [GDrive Upload Error] Local file does not exist: {local_path}")
        return False

    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    filename = os.path.basename(local_path)

    if backoff_delays is None:
        backoff_delays = [2.0, 4.0, 8.0, 16.0, 30.0]

    cmd = [
        "rclone", "copyto",
        local_path,
        f"{remote_name}:{relative_remote_path}",
        "--drive-root-folder-id", folder_id,
        "--retries", "3",
        "--low-level-retries", "10",
        "-q"
    ]

    print(f"☁️ [GDrive] Uploading {filename} ({file_size_mb:.1f} MB) -> {relative_remote_path}...")

    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ [GDrive] Successfully uploaded {filename} to Google Drive ({relative_remote_path})")
                return True
            else:
                err_msg = result.stderr.strip() or f"exit code {result.returncode}"
                print(f"⚠️ [GDrive] Attempt {attempt}/{max_retries} failed for {filename}: {err_msg}")
        except subprocess.TimeoutExpired:
            print(f"⚠️ [GDrive] Attempt {attempt}/{max_retries} timed out (300s) for {filename}")
        except Exception as e:
            print(f"⚠️ [GDrive] Attempt {attempt}/{max_retries} error for {filename}: {e}")

        if attempt < max_retries:
            delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
            print(f"   ⏳ Retrying upload in {delay:.0f}s...")
            time.sleep(delay)

    # All retries exhausted: record to failed log
    log_line = f"[{datetime.now().isoformat()}] FAILED: {local_path} -> {relative_remote_path}\n"
    try:
        with open(FAILED_LOG_PATH, "a") as f:
            f.write(log_line)
    except Exception as log_err:
        print(f"⚠️ Could not write to {FAILED_LOG_PATH}: {log_err}")

    print(f"❌ [GDrive] All {max_retries} upload attempts failed for {filename}. Logged to {FAILED_LOG_PATH}.")
    return False


def on_checkpoint_saved(
    local_path: str,
    dataset_ver: str,
    genre: str,
    args: Any = None,
    is_temporary: bool = False,
    delete_local_after_upload: bool = False,
) -> bool:
    """Callback triggered whenever a model checkpoint is saved.

    Handles upload to Google Drive and executes fail-safe local retention/cleanup.

    Args:
        local_path: Path to the local checkpoint .pth file.
        dataset_ver: Dataset version or fold identifier (e.g. 'v4_fold1').
        genre: Domain/Genre name (e.g. 'art').
        args: Command-line arguments namespace.
        is_temporary: Whether this checkpoint is temporary (e.g., intermediate user finetune model).
        delete_local_after_upload: Force deletion of local file if upload succeeds.

    Returns:
        bool: True if process succeeded.
    """
    if not os.path.exists(local_path):
        return False

    # Check if upload is disabled
    if args is not None and getattr(args, "no_gdrive_upload", False):
        return True

    remote_name = getattr(args, "rclone_remote", DEFAULT_REMOTE) if args else DEFAULT_REMOTE
    folder_id = getattr(args, "gdrive_folder_id", DEFAULT_FOLDER_ID) if args else DEFAULT_FOLDER_ID
    delete_all = getattr(args, "delete_local_on_upload", False) if args else False

    filename = os.path.basename(local_path)
    relative_remote_path = f"{dataset_ver}/{genre}/{filename}"

    upload_ok = upload_to_gdrive(
        local_path=local_path,
        relative_remote_path=relative_remote_path,
        remote_name=remote_name,
        folder_id=folder_id,
        max_retries=5,
    )

    if upload_ok:
        # Determine whether to delete local copy
        if is_temporary or delete_local_after_upload or delete_all:
            try:
                os.remove(local_path)
                print(f"🗑️ [Cleanup] Deleted local file after upload: {filename}")
            except OSError as e:
                print(f"⚠️ [Cleanup Warning] Could not delete local file {local_path}: {e}")
        return True
    else:
        # Fail-safe: Always keep local copy if upload fails
        print(f"🛡️ [Fail-Safe] Retaining local file on disk: {local_path}")
        return False
