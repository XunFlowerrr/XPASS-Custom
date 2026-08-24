import os
import time
import subprocess
import atexit
import threading
import concurrent.futures
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


class UploadManager:
    """Thread-safe background upload manager to upload checkpoints asynchronously without blocking training."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_workers: int = 2):
        if getattr(self, "_initialized", False):
            return
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GDriveUploader")
        self._futures: List[concurrent.futures.Future] = []
        self._futures_lock = threading.Lock()
        self._completed_count = 0
        self._failed_count = 0
        self._initialized = True
        atexit.register(self.shutdown)

    def submit_upload_task(
        self,
        local_path: str,
        relative_remote_path: str,
        remote_name: str = DEFAULT_REMOTE,
        folder_id: str = DEFAULT_FOLDER_ID,
        should_delete_local: bool = False,
    ) -> concurrent.futures.Future:
        """Submit an upload task to the background thread pool."""
        filename = os.path.basename(local_path)

        def _worker() -> bool:
            success = upload_to_gdrive(
                local_path=local_path,
                relative_remote_path=relative_remote_path,
                remote_name=remote_name,
                folder_id=folder_id,
            )
            if success:
                with self._futures_lock:
                    self._completed_count += 1
                if should_delete_local:
                    try:
                        if os.path.exists(local_path):
                            os.remove(local_path)
                            print(f"🗑️ [Cleanup] Deleted local file after background upload: {filename}")
                    except OSError as e:
                        print(f"⚠️ [Cleanup Warning] Could not delete local file {local_path}: {e}")
            else:
                with self._futures_lock:
                    self._failed_count += 1
                print(f"🛡️ [Fail-Safe] Retaining local file on disk: {local_path}")
            return success

        with self._futures_lock:
            self._futures = [f for f in self._futures if not f.done()]
            future = self.executor.submit(_worker)
            self._futures.append(future)
            pending = len(self._futures)
            print(f"🚀 [Async GDrive] Dispatched {filename} to background upload queue (Pending: {pending})")
            return future

    def wait_for_all_uploads(self, timeout: Optional[float] = None) -> None:
        """Wait for all currently submitted background upload tasks to complete."""
        with self._futures_lock:
            pending_futures = list(self._futures)

        if not pending_futures:
            return

        print(f"\n⏳ [UploadQueue] Waiting for {len(pending_futures)} pending background upload(s) to finish...")
        done, not_done = concurrent.futures.wait(pending_futures, timeout=timeout)
        if not_done:
            print(f"⚠️ [UploadQueue] {len(not_done)} upload(s) did not complete within timeout.")
        else:
            print(f"✅ [UploadQueue] All background uploads finished (Completed: {self._completed_count}, Failed: {self._failed_count})")

    def shutdown(self) -> None:
        """Gracefully wait for pending uploads and shut down executor on exit."""
        self.wait_for_all_uploads(timeout=300)
        self.executor.shutdown(wait=False)


_upload_manager: Optional[UploadManager] = None


def get_upload_manager(max_workers: int = 2) -> UploadManager:
    """Retrieve or initialize the singleton UploadManager instance."""
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = UploadManager(max_workers=max_workers)
    return _upload_manager


def wait_for_all_uploads(timeout: Optional[float] = None) -> None:
    """Global helper to wait for all background upload tasks to finish."""
    if _upload_manager is not None:
        _upload_manager.wait_for_all_uploads(timeout=timeout)


def on_checkpoint_saved(
    local_path: str,
    dataset_ver: str,
    genre: str,
    args: Any = None,
    is_temporary: bool = False,
    delete_local_after_upload: bool = False,
    async_upload: bool = True,
) -> bool:
    """Callback triggered whenever a model checkpoint is saved.

    Handles asynchronous or synchronous upload to Google Drive and executes fail-safe local retention/cleanup.

    Args:
        local_path: Path to the local checkpoint .pth file.
        dataset_ver: Dataset version or fold identifier (e.g. 'v4_fold1').
        genre: Domain/Genre name (e.g. 'art').
        args: Command-line arguments namespace.
        is_temporary: Whether this checkpoint is temporary (e.g., intermediate user finetune model).
        delete_local_after_upload: Force deletion of local file if upload succeeds.
        async_upload: If True and not --sync_upload, runs upload asynchronously in background.

    Returns:
        bool: True if process or dispatch succeeded.
    """
    if not os.path.exists(local_path):
        return False

    # Check if upload is disabled
    if args is not None and getattr(args, "no_gdrive_upload", False):
        return True

    remote_name = getattr(args, "rclone_remote", DEFAULT_REMOTE) if args else DEFAULT_REMOTE
    folder_id = getattr(args, "gdrive_folder_id", DEFAULT_FOLDER_ID) if args else DEFAULT_FOLDER_ID
    delete_all = getattr(args, "delete_local_on_upload", False) if args else False
    should_delete = is_temporary or delete_local_after_upload or delete_all
    is_sync = getattr(args, "sync_upload", False) or not async_upload

    filename = os.path.basename(local_path)
    relative_remote_path = f"{dataset_ver}/{genre}/{filename}"

    if is_sync:
        upload_ok = upload_to_gdrive(
            local_path=local_path,
            relative_remote_path=relative_remote_path,
            remote_name=remote_name,
            folder_id=folder_id,
            max_retries=5,
        )

        if upload_ok:
            if should_delete:
                try:
                    os.remove(local_path)
                    print(f"🗑️ [Cleanup] Deleted local file after upload: {filename}")
                except OSError as e:
                    print(f"⚠️ [Cleanup Warning] Could not delete local file {local_path}: {e}")
            return True
        else:
            print(f"🛡️ [Fail-Safe] Retaining local file on disk: {local_path}")
            return False
    else:
        # Asynchronous background upload
        num_workers = getattr(args, "upload_workers", 2) if args else 2
        manager = get_upload_manager(max_workers=num_workers)
        manager.submit_upload_task(
            local_path=local_path,
            relative_remote_path=relative_remote_path,
            remote_name=remote_name,
            folder_id=folder_id,
            should_delete_local=should_delete,
        )
        return True
