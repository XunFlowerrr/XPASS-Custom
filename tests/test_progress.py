import time
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.progress import ProgressTracker, format_time
from src.methods import source_only


def test_format_time():
    assert format_time(0) == "00:00:00"
    assert format_time(59) == "00:00:59"
    assert format_time(61) == "00:01:01"
    assert format_time(3665) == "01:01:05"
    assert format_time(90061) == "1d 01:01:01"
    assert format_time(None) == "--:--:--"
    assert format_time(-5) == "--:--:--"


def test_progress_tracker_determination():
    # 3 genres, 5 folds, 30 users each -> 450 total models
    total_genres = 3
    total_folds = 5
    total_models = 3 * 5 * 30
    max_epochs = 200

    tracker = ProgressTracker(
        total_genres=total_genres,
        total_folds=total_folds,
        total_models=total_models,
        max_epochs=max_epochs,
        mode_name="PIAA Finetune"
    )

    assert tracker.total_genres == 3
    assert tracker.total_folds == 5
    assert tracker.total_models == 450
    assert tracker.total_max_epochs == 450 * 200
    assert tracker.get_remaining_pessimistic_epochs() == 450 * 200


def test_progress_tracker_timing_and_eta():
    tracker = ProgressTracker(
        total_genres=2,
        total_folds=1,
        total_models=2,
        max_epochs=10,
        mode_name="Test"
    )
    tracker.set_context(genre_idx=1, genre_name="art", fold_idx=1, fold_name="v4_fold1", model_idx=1, model_name="u1")

    # Simulate epoch 0 taking 2.0s
    tracker.start_epoch(0)
    time.sleep(0.05)
    tracker.end_epoch(0)
    # Manually inject duration for testing calculations
    tracker.epoch_durations[-1] = 2.0

    speed = tracker.get_speed()
    assert speed == 2.0

    # Model 1 has completed epoch 0 out of 10. Remaining for current model: 9 epochs.
    # Model 2 has 10 epochs remaining. Total remaining: 19 epochs.
    assert tracker.get_remaining_pessimistic_epochs() == 19
    assert tracker.get_eta_seconds() == 38.0  # 19 epochs * 2.0s = 38.0s

    prefix = tracker.get_progress_prefix(phase="Train")
    assert "art 1/2" in prefix
    assert "Ep 1/10" in prefix

    timing = tracker.get_timing_info()
    assert "ETA: 00:00:38" in timing
    assert "2.0s/ep" in timing

    # Finish model 1 early at epoch 2
    tracker.finish_model(early_stopped=True)
    assert tracker.completed_models == 1

    # Start model 2
    tracker.set_context(genre_idx=2, genre_name="fashion", fold_idx=1, fold_name="v4_fold1", model_idx=2, model_name="u2")
    tracker.start_epoch(0)
    # Before epoch 0 completes: 10 epochs remaining for model 2
    assert tracker.get_remaining_pessimistic_epochs() == 10
    assert tracker.get_eta_seconds() == 20.0

    tracker.end_epoch(0)
    tracker.epoch_durations[-1] = 2.0
    # After epoch 0 completes: 9 epochs remaining for model 2
    assert tracker.get_remaining_pessimistic_epochs() == 9
    assert tracker.get_eta_seconds() == 18.0


def test_mock_training_with_progress_tracker(tmp_path):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 7)

        def forward(self, x):
            return self.fc(x)

    class DummyArgs:
        num_epochs = 3
        genre = "art"
        dataset_ver = "v4_fold1"
        lr_decay_factor = 0.5
        lr_patience = 2
        max_patience_epochs = 5
        no_gdrive_upload = True

    # Mock dataset with 8 samples
    x = torch.randn(8, 4)
    y = torch.softmax(torch.randn(8, 7), dim=1)
    ds = TensorDataset(x, y)

    def collate(batch):
        return {
            'image': torch.stack([item[0] for item in batch]),
            'Aesthetic': torch.stack([item[1] for item in batch]),
        }

    loader = DataLoader(ds, batch_size=4, collate_fn=collate)
    src_dataloaders = (loader, loader, loader)

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    best_model_path = str(tmp_path / "best_model.pth")
    components = {}

    tracker = ProgressTracker(
        total_genres=1,
        total_folds=1,
        total_models=1,
        max_epochs=3,
        mode_name="Mock Train"
    )
    tracker.set_context(genre_idx=1, genre_name="art", fold_idx=1, fold_name="fold1")

    # Run trainer with tracker attached
    source_only.trainer(
        src_dataloaders,
        model,
        optimizer,
        DummyArgs(),
        torch.device('cpu'),
        best_model_path,
        components,
        tracker=tracker
    )

    assert tracker.completed_models == 1
    assert tracker.total_executed_epochs == 3
