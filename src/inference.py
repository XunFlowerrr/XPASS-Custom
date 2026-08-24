import os
import numpy as np
import copy
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

import torch

from .data import make_loader
from .train_common import num_bins, is_trainable_only_state, FROZEN_STATE_PREFIX
from .argflags import REPORTS_ROOT


def _build_eval_model(num_bins_, num_attr, num_pt, genres, backbone_dict, args, device):
    """Build a PIAA model for evaluation/inference."""
    from .train_common import build_piaa_model
    return build_piaa_model(num_bins_, num_attr, num_pt, genres, backbone_dict, args).to(device)


def inference(train_dataset, val_dataset, test_dataset, args, device, model, eval_split, experiment_name='', model_path=None):
    """Per-user inference: load each user's best model and evaluate on a chosen split (val or test).

    eval_split: 'Test' or 'Val'
    model_path: Path to the loaded model (used for filename generation)
    Returns mean_user_srocc, mean_user_mse
    """
    from .evaluate import evaluate

    batch_size = args.batch_size
    user_sroccs = []
    user_mses = []
    user_ndcgs = []
    user_maes = []
    user_cccs = []
    per_user_results = {}

    # derive unique user ids from the train dataset (same approach as PIAA)
    try:
        unique_user_ids = np.unique(train_dataset.data['user_id'].values)
    except Exception:
        unique_user_ids = []

    # choose which dataset to evaluate: validation or test
    if eval_split == 'Val':
        source_dataset = val_dataset
    else:
        source_dataset = test_dataset

    for uid in unique_user_ids:
        if pd.isna(uid):
            continue

        # prepare per-user test dataset (filter selected split)
        user_test_ds = copy.deepcopy(source_dataset)
        try:
            user_test_ds.data = user_test_ds.data[user_test_ds.data['user_id'] == uid]
        except Exception:
            # if dataset structure differs, skip
            print(f"Skipping user {uid}: could not filter test dataset for user_id")
            continue

        if len(user_test_ds) == 0:
            print(f"No test samples for user {uid}, skipping.")
            continue

        user_test_loader = make_loader(user_test_ds, batch_size, args.num_workers)
        _, _, final_srocc, final_mse, final_ndcg, final_mae, final_ccc = evaluate(model, user_test_loader, device, PIAA=True)

        user_sroccs.append(final_srocc if final_srocc is not None else np.nan)
        user_mses.append(final_mse if final_mse is not None else np.nan)
        user_ndcgs.append(final_ndcg if final_ndcg is not None else np.nan)
        user_maes.append(final_mae if final_mae is not None else np.nan)
        user_cccs.append(final_ccc if final_ccc is not None else np.nan)
        per_user_results[str(uid)] = {
            'srocc': float(final_srocc) if final_srocc is not None else None,
            'ndcg@10': float(final_ndcg) if final_ndcg is not None else None,
            'mae': float(final_mae) if final_mae is not None else None,
            'ccc': float(final_ccc) if final_ccc is not None else None,
        }

    # user-average
    mean_user_srocc = np.mean(user_sroccs) if len(user_sroccs) > 0 else np.nan
    mean_user_mse = np.mean(user_mses) if len(user_mses) > 0 else np.nan
    mean_user_ndcg = np.mean(user_ndcgs) if len(user_ndcgs) > 0 else np.nan
    mean_user_mae = np.mean(user_maes) if len(user_maes) > 0 else np.nan
    mean_user_ccc = np.mean(user_cccs) if len(user_cccs) > 0 else np.nan

    if eval_split == 'Test':
        print(f"[{args.genre} {eval_split}] Avg SROCC={mean_user_srocc:.4f}  "
              f"MSE={mean_user_mse:.4f}  NDCG@10={mean_user_ndcg:.4f}")

    # Save results to JSON file (only for Test split)
    if eval_split == 'Test':
        save_dir = os.path.join(REPORTS_ROOT, args.dataset_ver,
                                args.genre)
        os.makedirs(save_dir, exist_ok=True)

        # Use model_path basename for both experiment_name field and json filename
        if model_path:
            model_basename = os.path.splitext(os.path.basename(model_path))[0]
            json_filename = f"{model_basename}.json"
            display_name = model_basename
        else:
            json_filename = f"{experiment_name}.json"
            display_name = experiment_name

        # Prepare per_user_metrics in the same format as other models: {uid: {genre: {srocc, ndcg@10}}}
        per_user_metrics_formatted = {}
        for uid_str, metrics_val in per_user_results.items():
            per_user_metrics_formatted[uid_str] = {
                args.genre: metrics_val
            }

        result_data = {
            'experiment_name': display_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'GIAA',
            'genres': [args.genre],
            'average_metrics': {
                args.genre: {
                    'srocc': float(mean_user_srocc) if not np.isnan(mean_user_srocc) else None,
                    'ndcg@10': float(mean_user_ndcg) if not np.isnan(mean_user_ndcg) else None,
                    'mae': float(mean_user_mae) if not np.isnan(mean_user_mae) else None,
                    'ccc': float(mean_user_ccc) if not np.isnan(mean_user_ccc) else None,
                }
            },
            'per_user_metrics': per_user_metrics_formatted
        }

        json_path = os.path.join(save_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"Test results saved to {json_path}")

    return mean_user_srocc, mean_user_mse


def inference_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict,
                       pretrained_model_dict=None):
    """
    Inference for all users after finetuning.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}}
        args: arguments
        device: device
        dirname: directory name for saving models
        experiment_name: experiment name
        backbone_dict: dict of {genre: backbone_type}
        pretrained_model_dict: {genre: path} to the PIAA pretrain checkpoint. Finetune
            checkpoints only store the trainable half, so the frozen NIMA weights are
            restored from here. Discovered automatically when omitted.
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre_str = '-'.join(genres)

    if num_attr is None or num_pt is None:
        _sample = datasets_dict[genres[0]]['test'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt
    all_user_ids = set()
    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_plcc_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    for genre in genres:
        all_user_ids.update(datasets_dict[genre]['test'].data['user_id'].values)

    model_name_base = experiment_name

    results = {}

    # One model and one copy of the frozen NIMA weights for every user: the
    # backbone is identical across users, so rebuilding CLIP per user only costs
    # time. Each user's checkpoint then supplies the trainable half on top.
    model_user = _build_eval_model(num_bins, num_attr, num_pt, genres, backbone_dict, args, device)

    if pretrained_model_dict is None:
        pretrained_model_dict = _tp.discover_pretrained_models(
            args.dataset_ver, genres[0], 'PIAA_finetune', getattr(args, 'model_type', None))
    base_state = torch.load(pretrained_model_dict[genres[0]], map_location=device)

    # Sharing one model across users relies on base_state covering every frozen
    # tensor; anything it misses would otherwise keep the previous user's values.
    missing_frozen = [k for k in model_user.state_dict()
                      if k.startswith(FROZEN_STATE_PREFIX) and k not in base_state]
    if missing_frozen:
        print(f"[Reuse] Pretrain checkpoint is missing {len(missing_frozen)} frozen key(s) "
              f"(e.g. {missing_frozen[:3]}); rebuilding the model per user.")

    for uid in sorted(list(all_user_ids)):
        print(f"Running inference for user {uid} using saved best model...")
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{model_name_base}_finetune.pth')
        if missing_frozen:
            model_user = _build_eval_model(num_bins, num_attr, num_pt, genres, backbone_dict, args, device)
        try:
            user_state = torch.load(best_model_path, map_location=device)
            if is_trainable_only_state(user_state):
                # Frozen half from the pretrain checkpoint, trainable half from the user.
                model_user.load_state_dict(base_state, strict=False)
                model_user.load_state_dict(user_state, strict=False)
            else:
                # Full checkpoint written by an older run.
                model_user.load_state_dict(user_state)
        except Exception as e:
            print(f"Warning: best model not found for user {uid} at {best_model_path}, skipping. Error: {e}")
            results[uid] = (np.nan, np.nan)
            continue

        # In-domain evaluation
        test_loaders_dict = {}
        total_test_samples = 0
        for genre in genres:
            user_test_ds = copy.copy(datasets_dict[genre]['test'])
            user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
            if len(user_test_ds) > 0:
                test_loaders_dict[genre] = make_loader(user_test_ds, batch_size, args.num_workers)
                total_test_samples += len(user_test_ds)
        if total_test_samples == 0:
            print(f"No test samples for user {uid}, skipping.")
            results[uid] = ({}, np.nan)
        else:
            genre_metrics, total_mae = evaluate(model_user, test_loaders_dict, device)
            for genre, metrics in genre_metrics.items():
                genre_srocc_list[genre].append(metrics['srocc'])
                genre_mae_list[genre].append(metrics['mae'])
                genre_ndcg_list[genre].append(metrics['ndcg@10'])
                genre_ccc_list[genre].append(metrics['ccc'])
                genre_plcc_list[genre].append(metrics['plcc'])
            results[uid] = (genre_metrics, total_mae)

    # Calculate genre-specific averages
    genre_avg_metrics = {}
    for genre in genres:
        if genre in genre_srocc_list and len(genre_srocc_list[genre]) > 0:
            genre_avg_metrics[genre] = {
                'srocc': np.mean(genre_srocc_list[genre]),
                'mae': np.mean(genre_mae_list[genre]),
                'ndcg@10': np.mean(genre_ndcg_list[genre]),
                'ccc': np.mean(genre_ccc_list[genre]),
                'plcc': np.mean(genre_plcc_list[genre]),
            }

    for genre, metrics in genre_avg_metrics.items():
        print(f"[{genre} Test] Avg SROCC={metrics['srocc']:.4f} PLCC={metrics['plcc']:.4f} "
              f"MAE={metrics['mae']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  CCC={metrics['ccc']:.4f}")

    # Save test performance to JSON
    _prefix = genre_str
    save_dir = os.path.join(REPORTS_ROOT, args.dataset_ver, genre_str)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare per-user results
    per_user_results = {}
    for uid, (genre_metrics_user, total_mae) in results.items():
        if isinstance(genre_metrics_user, dict):
            per_user_results[str(uid)] = {
                genre: {'srocc': float(metrics['srocc']), 'mae': float(metrics['mae']), 'ndcg@10': float(metrics['ndcg@10']), 'ccc': float(metrics['ccc']), 'plcc': float(metrics['plcc'])}
                for genre, metrics in genre_metrics_user.items()
            }

    result_data = {
        'experiment_name': model_name_base,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_finetune',
        'genres': genres,
        'average_metrics': {
            genre: {'srocc': float(metrics['srocc']), 'plcc': float(metrics['plcc']), 'mae': float(metrics['mae']), 'ndcg@10': float(metrics['ndcg@10']), 'ccc': float(metrics['ccc'])}
            for genre, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }

    # Remove trailing mode suffix to avoid duplication (e.g., "name_finetune_finetune.json")
    base_name = model_name_base.removesuffix('_finetune')
    json_filename = f"{_prefix}_{args.model_type}_{base_name}_finetune.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return results


def evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=None):
    """
    Pretrain後のベストモデルでval_piaa_datasetを使い、
    ユーザーごとにSROCC/NDCGを算出する。
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict_user.keys())
    genre = genres[0]

    if num_attr is None or num_pt is None:
        _sample = datasets_dict_user[genre]['val'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt

    model = _build_eval_model(num_bins, num_attr, num_pt, genres, backbone_dict, args, device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    all_user_ids = set(datasets_dict_user[genre]['val'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    user_metrics = {}
    for uid in unique_user_ids:
        user_val_ds = copy.copy(datasets_dict_user[genre]['val'])
        user_val_ds.data = datasets_dict_user[genre]['val'].data[datasets_dict_user[genre]['val'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_val_ds) > 0:
            val_loaders_dict = {genre: make_loader(user_val_ds, batch_size, args.num_workers)}
            genre_metrics, _ = evaluate(model, val_loaders_dict, device)
            user_metrics[uid] = genre_metrics

    print(f"Pretrain val PIAA evaluation done: {len(user_metrics)} users evaluated")


def inference_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, model_state_dict=None):
    """
    Per-user evaluation after pretraining, separated from training.
    Args:
        datasets_dict: dict of {genre: {'train': ds, 'val': ds, 'test': ds}} (PIAA data)
        model_state_dict: if provided, load from this state dict instead of best_model_path
    """
    from . import train_PIAA as _tp
    from .evaluate import evaluate_piaa as evaluate
    num_attr = _tp.num_attr
    num_pt = _tp.num_pt

    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    if num_attr is None or num_pt is None:
        _sample = datasets_dict[genre]['train'][0]
        num_attr = len(_sample['QIP'])
        num_pt = len(_sample['traits'])
        _tp.num_attr = num_attr
        _tp.num_pt = num_pt

    model = _build_eval_model(num_bins, num_attr, num_pt, genres, backbone_dict, args, device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    all_user_ids = set(datasets_dict[genre]['test'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))

    genre_srocc_list = defaultdict(list)
    genre_mae_list = defaultdict(list)
    genre_ndcg_list = defaultdict(list)
    genre_ccc_list = defaultdict(list)
    per_user_results = {}

    for uid in unique_user_ids:
        user_test_ds = copy.copy(datasets_dict[genre]['test'])
        user_test_ds.data = datasets_dict[genre]['test'].data[datasets_dict[genre]['test'].data['user_id'] == uid].reset_index(drop=True)
        if len(user_test_ds) > 0:
            test_loaders_dict = {genre: make_loader(user_test_ds, batch_size, args.num_workers)}
            genre_metrics, total_mae = evaluate(model, test_loaders_dict, device)
            for g, metrics in genre_metrics.items():
                genre_srocc_list[g].append(metrics['srocc'])
                genre_mae_list[g].append(metrics['mae'])
                genre_ndcg_list[g].append(metrics['ndcg@10'])
                genre_ccc_list[g].append(metrics['ccc'])
            per_user_results[str(uid)] = {
                g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
                for g, metrics in genre_metrics.items()
            }

    genre_avg_metrics = {}
    for g in genres:
        if g in genre_srocc_list and len(genre_srocc_list[g]) > 0:
            genre_avg_metrics[g] = {
                'srocc': np.mean(genre_srocc_list[g]),
                'mae': np.mean(genre_mae_list[g]),
                'ndcg@10': np.mean(genre_ndcg_list[g]),
                'ccc': np.mean(genre_ccc_list[g]),
            }

    for g, metrics in genre_avg_metrics.items():
        print(f"[{g} Test] Avg SROCC={metrics['srocc']:.4f}  MAE={metrics['mae']:.4f}  "
              f"NDCG@10={metrics['ndcg@10']:.4f}  CCC={metrics['ccc']:.4f}")

    # Save test performance to JSON
    save_dir = os.path.join(REPORTS_ROOT, args.dataset_ver, genre_str)
    os.makedirs(save_dir, exist_ok=True)

    model_basename = os.path.splitext(os.path.basename(best_model_path))[0]
    result_data = {
        'experiment_name': model_basename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'PIAA_pretrain',
        'genres': genres,
        'average_metrics': {
            g: {'srocc': float(metrics['srocc']), 'ndcg@10': float(metrics['ndcg@10']), 'mae': float(metrics['mae']), 'ccc': float(metrics['ccc'])}
            for g, metrics in genre_avg_metrics.items()
        },
        'per_user_metrics': per_user_results
    }

    base_name = model_basename.removesuffix('_pretrain')
    json_filename = f"{base_name}_pretrain.json"
    json_path = os.path.join(save_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"Test results saved to {json_path}")

    return None
