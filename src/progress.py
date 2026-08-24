import time
from typing import Optional, List


def format_time(seconds: Optional[float]) -> str:
    """Format duration in seconds to standard HH:MM:SS or Xd HH:MM:SS format."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    total_seconds = int(round(seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressTracker:
    """Multi-level progress tracker and real-time ETA estimator across genres, folds, users, and epochs."""

    def __init__(
        self,
        total_genres: int = 1,
        total_folds: int = 1,
        total_models: int = 1,
        max_epochs: int = 200,
        mode_name: str = "Train",
    ):
        self.total_genres = max(1, total_genres)
        self.total_folds = max(1, total_folds)
        self.total_models = max(1, total_models)
        self.max_epochs = max(1, max_epochs)
        self.mode_name = mode_name

        self.total_max_epochs = self.total_models * self.max_epochs

        # Hierarchy state
        self.current_genre_idx = 1
        self.current_genre_name = ""
        self.current_fold_idx = 1
        self.current_fold_name = ""
        self.current_model_idx = 1
        self.current_model_name: Optional[str] = None
        self.current_epoch = 0

        # Timing state
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.epoch_durations: List[float] = []
        self.completed_models = 0
        self.completed_epochs_current_model = 0
        self.total_executed_epochs = 0

    def print_initial_summary(self):
        """Print overall task summary before starting execution."""
        print("\n" + "=" * 70)
        print(f"🚀 [ProgressTracker] Initialized {self.mode_name}")
        print(f"   • Total Genres:        {self.total_genres}")
        print(f"   • Total Folds:         {self.total_folds}")
        print(f"   • Total Model Runs:    {self.total_models}")
        print(f"   • Max Epochs per Run:  {self.max_epochs}")
        print(f"   • Total Max Epochs:    {self.total_max_epochs:,}")
        print("=" * 70 + "\n", flush=True)

    def set_context(
        self,
        genre_idx: Optional[int] = None,
        genre_name: Optional[str] = None,
        fold_idx: Optional[int] = None,
        fold_name: Optional[str] = None,
        model_idx: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        """Update current active hierarchy context."""
        if genre_idx is not None:
            self.current_genre_idx = genre_idx
        if genre_name is not None:
            self.current_genre_name = str(genre_name)
        if fold_idx is not None:
            self.current_fold_idx = fold_idx
        if fold_name is not None:
            self.current_fold_name = str(fold_name)
        if model_idx is not None:
            self.current_model_idx = model_idx
        if model_name is not None:
            self.current_model_name = str(model_name)

    def start_epoch(self, epoch: int):
        """Record the start timestamp of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self, epoch: int):
        """Record the end of an epoch and update duration statistics."""
        self.current_epoch = epoch
        self.completed_epochs_current_model = epoch + 1
        duration = max(0.001, time.time() - self.epoch_start_time)
        self.epoch_durations.append(duration)
        # Keep recent 20 epoch durations for responsive EMA/speed tracking
        if len(self.epoch_durations) > 20:
            self.epoch_durations.pop(0)
        self.total_executed_epochs += 1

    def finish_model(self, early_stopped: bool = False):
        """Record the completion of a model training run."""
        self.completed_models += 1
        self.completed_epochs_current_model = 0

    def get_speed(self) -> float:
        """Calculate average seconds per epoch (recent window average)."""
        if not self.epoch_durations:
            elapsed = time.time() - self.start_time
            return elapsed / max(1, self.total_executed_epochs) if self.total_executed_epochs > 0 else 0.0
        return sum(self.epoch_durations) / len(self.epoch_durations)

    def get_elapsed_seconds(self) -> float:
        """Total elapsed seconds since tracker initialization."""
        return time.time() - self.start_time

    def get_remaining_pessimistic_epochs(self) -> int:
        """Calculate remaining epochs assuming full max_epochs for remaining models."""
        if self.completed_models >= self.total_models:
            return 0
        remaining_models = max(0, self.total_models - self.completed_models - 1)
        remaining_current_model_epochs = max(0, self.max_epochs - self.completed_epochs_current_model)
        return (remaining_models * self.max_epochs) + remaining_current_model_epochs

    def get_eta_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds based on pessimistic epoch count and speed."""
        speed = self.get_speed()
        if speed <= 0 or self.total_executed_epochs == 0:
            return None
        remaining_epochs = self.get_remaining_pessimistic_epochs()
        return remaining_epochs * speed

    def get_progress_prefix(self, phase: str = "Train") -> str:
        """Build progress hierarchy tag string."""
        parts = []
        if self.total_genres > 1 or self.current_genre_name:
            g_tag = f"{self.current_genre_name} {self.current_genre_idx}/{self.total_genres}" if self.current_genre_name else f"Genre {self.current_genre_idx}/{self.total_genres}"
            parts.append(g_tag)

        if self.total_folds > 1 or self.current_fold_name:
            f_tag = f"{self.current_fold_name} {self.current_fold_idx}/{self.total_folds}" if self.current_fold_name else f"Fold {self.current_fold_idx}/{self.total_folds}"
            parts.append(f_tag)

        if self.current_model_name is not None and self.total_models > (self.total_genres * self.total_folds):
            # Per-user finetune mode
            u_tag = f"user {self.current_model_name} {self.current_model_idx}/{self.total_models}"
            parts.append(u_tag)

        epoch_tag = f"Ep {self.current_epoch + 1}/{self.max_epochs}"
        if phase:
            epoch_tag += f" [{phase}]"
        parts.append(epoch_tag)

        return "[" + " | ".join(parts) + "]"

    def get_timing_info(self) -> str:
        """Build timing and speed tag string."""
        elapsed_str = format_time(self.get_elapsed_seconds())
        eta_seconds = self.get_eta_seconds()
        eta_str = format_time(eta_seconds)
        speed = self.get_speed()
        speed_str = f"{speed:.1f}s/ep" if speed > 0 else "--s/ep"

        return f"[Elapsed: {elapsed_str} | ETA: {eta_str} | {speed_str}]"

    def get_full_header(self, phase: str = "Train") -> str:
        """Combine hierarchy and timing into a single informative header."""
        return f"{self.get_progress_prefix(phase)} {self.get_timing_info()}"
