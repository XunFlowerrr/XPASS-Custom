<p align="center">
  <img src="logo.png" alt="proj-xpass-logo" width="750">
</p>

## Overview

**XPASS-SIMPLE** is a base codebase for Personalized Image Aesthetic Assessment (PIAA) research. Using the XPASS dataset (three domains: artworks, fashion images, and scenery), it supports training and inference of General Image Aesthetic Assessment (GIAA) models and Personalized Aesthetic Assessment (PIAA) models within each domain.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Required Data](#required-data)
3. [Training (GIAA)](#training-giaa)
4. [Training (PIAA)](#training-piaa)
5. [Result Aggregation](#result-aggregation)
6. [Feature Dimensions](#feature-dimensions)
7. [Commit Message Convention](#commit-message-convention)
8. [Data Statistics](#data-statistics)

---

## Environment Setup

### Using `uv` (Recommended)

```bash
# Install dependencies and sync virtual environment
uv sync

# Run scripts directly with uv
uv run python -m src.train_GIAA --genre art
```

### Using standard `venv` / `pip`

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install .
```
---

## Required Data

### Metadata (`data/maked/`)

| File | Description |
|---|---|
| `users.csv` | User profiles (demographics, personality, interests keyed by `user_id`) |
| `ratings.csv` | Per-sample ratings (`user_id`, `sample_file`, `genre`, `Aesthetic`, ...) |
| `QIP_{genre}.csv` | Image perceptual quality features per genre (art / fashion). Scenery uses `QIP_scenery_image.csv` |

### Images (`data/samples/`)

- art / fashion: `data/samples/{genre}/{filename}`
- scenery: `data/samples/scenery_image/{filename}` (samples with `.mp4` names reference `.jpg` files)

### Split Files (`data/split/{dataset_ver}/{genre}/`)

Referenced by `--dataset_ver` (e.g. `v3_fold1`) and `--genre`.

- `train_images_GIAA.txt` / `val_images_GIAA.txt` — image lists for GIAA training/validation
- `train_users_GIAA.txt` / `val_users_GIAA.txt` — user lists for GIAA pretraining
- `train_PIAA.txt` / `val_PIAA.txt` / `test_PIAA.txt` — for PIAA (format: `user_id\tfilename`)

> **Note:** The GIAA pool and PIAA pool are kept separate. `train/val_giaa_dataset` automatically excludes user IDs present in `train/val/test_PIAA.txt` to prevent leakage of PIAA users.

---

## Training (GIAA)

Trains NIMA (backbone + aesthetic score head) with EMD loss. Training, validation, and testing are all performed within the single specified genre (`--genre`).

#### Key Arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--genre` | str | (required) | Training genre (`art` / `fashion` / `scenery` / `all`) |
| `--dataset_ver` | str | `v3_all` | Data split version (`_all` suffix runs all folds sequentially) |
| `--backbone` | str | `clip_vit_b16` | Backbone (`clip_vit_b16` only) |
| `--root_dir` | str | `{repo}/data` | Root directory for image data |
| `--num_epochs` | int | `200` | Maximum number of epochs |
| `--batch_size` | int | `32` | Batch size |
| `--lr` | float | `1e-5` | Learning rate |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateau decay factor |
| `--lr_patience` | int | `5` | ReduceLROnPlateau patience |
| `--max_patience_epochs` | int | `10` | Early stopping patience (epochs) |
| `--dropout` | float | `0.1` | Dropout rate |
| `--num_workers` | int | `4` | DataLoader worker count |
| `--device` | str | `auto` | Target device (`auto`: cuda/mps/cpu auto-detection, `cuda`, `mps`, `cpu`) |

#### Example Commands

```bash
# art
python -m src.train_GIAA --genre art

# Train all genres sequentially
python -m src.train_GIAA --genre all
```

Trained models are saved to `models_pth/{dataset_ver}/{genre}/` and test result JSONs to `reports/exp/{dataset_ver}/{genre}/`.

---

## Training (PIAA)

Initializes from a GIAA-trained NIMA and trains a personalized aesthetic assessment model (`ICI` or `MIR`) in two stages: `PIAA_pretrain` (shared model pretraining across all users) → `PIAA_finetune` (per-user fine-tuning). Loss function is fixed to MSE.

#### Key Arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--genre` | str | (required) | Training genre (`art` / `fashion` / `scenery` / `all`) |
| `--dataset_ver` | str | `v3_all` | Data split version |
| `--piaa_mode` | str | `PIAA_pretrain` | PIAA mode (`PIAA_pretrain` / `PIAA_finetune`) |
| `--model_type` | str | `ICI` | PIAA model (`ICI`: interaction-based / `MIR`: MLP Interaction Regression) |
| `--backbone` | str | `clip_vit_b16` | Backbone (`clip_vit_b16` only) |
| `--root_dir` | str | `{repo}/data` | Root directory for image data |
| `--num_epochs` | int | `200` | Maximum number of epochs |
| `--batch_size` | int | `32` (pretrain) / `16` (finetune) | Batch size (auto-set by mode if not specified) |
| `--lr` | float | `5e-6` (pretrain) / `1e-5` (finetune) | Learning rate (auto-set by mode if not specified) |
| `--lr_decay_factor` | float | `0.5` | ReduceLROnPlateau decay factor |
| `--lr_patience` | int | `5` | ReduceLROnPlateau patience |
| `--max_patience_epochs` | int | `10` | Early stopping patience (epochs) |
| `--dropout` | float | `0.1` | Dropout rate |
| `--num_workers` | int | `4` | DataLoader worker count |
| `--device` | str | `auto` | Target device (`auto`: cuda/mps/cpu auto-detection, `cuda`, `mps`, `cpu`) |
| `--start_fold` | int | `1` | Fold number to resume from (used when `--dataset_ver` ends with `_all`) |
| `--no_save_model` | flag | `False` | Keep best model in memory without saving to disk |
| `--keep_finetune_pth` | flag | `False` | Retain `*_finetune.pth` files instead of deleting them after finetuning |

#### Example Commands

```bash
# Pretrain
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --piaa_mode PIAA_pretrain --batch_size 128

# Finetune
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --piaa_mode PIAA_finetune --batch_size 16

# MIR: Pretrain / Finetune
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --model_type MIR --piaa_mode PIAA_pretrain --batch_size 128
python -m src.train_PIAA --genre art --dataset_ver v3_all \
  --model_type MIR --piaa_mode PIAA_finetune --batch_size 16
```

> **Prerequisite:** `PIAA_pretrain` requires a GIAA-trained NIMA (`*NIMA*.pth`) in `models_pth/{dataset_ver}/{genre}/`. `PIAA_finetune` loads `*_pretrain.pth` from the same directory.

Result JSONs are saved to `reports/exp/{dataset_ver}/{genre}/`.

### Loading the New Finetune Checkpoints for Inference

New `*_finetune.pth` files contain only the per-user trainable delta (about
7 MB), rather than a full copy of the frozen NIMA/CLIP tower. Inference must
therefore load two matching files:

1. the shared `*_pretrain.pth`; and
2. the user's `*_finetune.pth` delta.

They must have the same dataset fold, genre, and model type (ICI or MIR).
`src.train_PIAA --piaa_mode PIAA_finetune` performs this reconstruction and
test inference automatically after training. For inference without training,
load the shared state first and the user delta second:

```python
base_state = torch.load(pretrain_path, map_location=device)
user_state = torch.load(user_delta_path, map_location=device)

if is_trainable_only_state(user_state):
    model.load_state_dict(base_state, strict=False)
    model.load_state_dict(user_state, strict=False)
else:
    # Old full finetune checkpoints remain supported.
    model.load_state_dict(user_state)

model.eval()
with torch.no_grad():
    prediction = model(image, traits.float(), qip.float(), genre)
```

See [`PROJECT_UPDATE.md`](PROJECT_UPDATE.md#14-using-the-new-finetune-inference-path)
for a complete model-construction example, input shapes, and operational notes.

---

## Result Aggregation

Aggregates inference result JSONs from each fold and outputs per-fold and per-user averages and standard deviations for PIAA metrics (SROCC / NDCG@10 / MAE / CCC). Used to evaluate cross-validation results.

```bash
# Aggregate ICI finetune results from all v3 folds
python -m src.analysis aggregate --version v3 --genre art --pattern finetune --method ICI
```

#### Key Arguments

| Argument | Type | Default | Description |
|------|------|------|------|
| `--version` | str | (required) | Data split version (e.g. `v3`) |
| `--genre` | str | (required) | Genre (`art` / `fashion` / `scenery`) |
| `--pattern` | str | `""` | Glob pattern for JSON filenames (e.g. `finetune`, `pretrain`) |
| `--method` | str | `None` | Filter by model (`ICI` / `MIR`) |
| `--folds` | int+ | `None` | Fold numbers to aggregate (e.g. `--folds 1 3 5`). Omit for all folds |
| `--ids` / `--min-id` / `--max-id` | int | `None` | Filter by run ID |
| `--reports_dir` | str | `{repo}/reports/exp` | Root directory for result JSONs |

---

## Feature Dimensions

### Personal Traits Vector (116 dimensions)

The `traits` vector represents user-specific characteristics and preferences, consisting of 116 dimensions split into two categories.

#### 1. Score Vector (70 dimensions)

Survey responses on personality and interests. Each question is rated on a 7-point scale (0–6) and encoded as a one-hot vector of 7 dimensions per question.

- **Q1–Q10** (70 dimensions): 10 personality trait questions based on the Big Five personality model
- Interest fields (`art_interest`, `fashion_interest`, `photoVideo_interest`) are also included as 7-dimensional one-hot vectors, for a combined total of 70 dimensions.

#### 2. Attribute Vector (46 dimensions)

| Attribute | Dimensions | Description |
|------|--------|------|
| age_onehot | 5 | Age group (5 bins) |
| gender_onehot | 3 | Gender (3 categories) |
| edu_onehot | 7 | Education level (7 categories) |
| nationality_onehot | 4 | Nationality (4 categories) |
| art_learn_onehot | 2 | Art learning experience (yes/no) |
| fashion_learn_onehot | 2 | Fashion learning experience (yes/no) |
| photoVideo_learn_onehot | 2 | Photo/video learning experience (yes/no) |

**Total: 70 + 46 = 116 dimensions**

### Image Perceptual Quality Vector - QIP (45 dimensions)

The `QIP` vector contains objective visual features extracted from each image.

| Category | Dimensions | Content |
|---|---|---|
| Basic image properties | 6 | Image size, aspect ratio, RMS contrast, luminance entropy, complexity, edge density |
| Color properties | 20 | Color entropy, mean/std of RGB / Lab / HSV |
| Composition and balance | 6 | Mirror symmetry, DCM distance, DCM position (x, y), balance |
| Symmetry features | 3 | CNN symmetry (left-right / top-bottom / composite) |
| Texture and frequency | 8 | Fourier gradient/sigma, 2D/3D fractal dimension, self-similarity (PHOG/CNN), anisotropy, homogeneity |
| Visual complexity | 3 | 1st/2nd-order EOE, sparsity, variability |

**Total: 45 dimensions** (excluding the `img_file` column)

---

## Commit Message Convention

| Prefix | Purpose |
|----------------|------|
| `feat:` | Add new feature or module |
| `fix:` | Fix a bug or defect |
| `refactor:` | Internal restructuring or code reorganization (no behavior change) |
| `exp:` | Add or update experiment-related files |
| `data:` | Add or update data files |
| `docs:` | Update documentation |
| `conf:` | Change configuration files |
| `chore:` | Miscellaneous tasks (dependency updates, `.gitignore` fixes, etc.) |

---
