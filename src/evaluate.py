import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import contextlib
from collections import defaultdict
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score

from .train_common import earth_mover_distance, num_bins

_criterion_mse = nn.MSELoss()


def _get_autocast(device):
    """Return an autocast context manager compatible with device (cuda, mps, cpu)."""
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type == 'cuda':
        return autocast(device_type='cuda')
    elif dev_type == 'mps':
        try:
            return autocast(device_type='mps')
        except (TypeError, ValueError, RuntimeError):
            return contextlib.nullcontext()
    return contextlib.nullcontext()


def evaluate(model, dataloader, device, PIAA=False, epoch: int = None, phase_name: str = "Val"):
    model.eval()
    running_emd_loss = 0.0
    running_mse_loss = 0.0
    running_mae_loss = 0.0
    scale = torch.arange(0, num_bins).to(device)
    desc = f"Epoch {epoch} [{phase_name}]" if epoch is not None else phase_name
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=1, ncols=120, colour="#fffb00", ascii="-=")
    mean_pred = []
    mean_target = []
    user_id_list = []
    autocast_ctx = _get_autocast(device)
    with torch.no_grad():
        for sample in progress_bar:
            images = sample['image'].to(device)
            aesthetic_score_histogram = sample['Aesthetic'].to(device)
            if PIAA:
                user_id_list.extend(sample['user_id'])
            with autocast_ctx:
                aesthetic_logits = model(images)
                prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
                loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

            outputs_mean = torch.sum(prob_aesthetic * scale, dim=1, keepdim=True)
            if aesthetic_score_histogram.shape[-1] == 1:
                # PIAA: scalar score in [0, 1], scale to [0, num_bins-1]
                target_mean = aesthetic_score_histogram * (num_bins - 1)
            else:
                # GIAA: histogram, compute expected value
                target_mean = torch.sum(aesthetic_score_histogram * scale, dim=1, keepdim=True)
            mean_pred.append(outputs_mean.view(-1).cpu().numpy())
            mean_target.append(target_mean.view(-1).cpu().numpy())
            # Normalize to [0, 1] for scale-consistent comparison with PIAA
            outputs_mean_norm = outputs_mean / (num_bins - 1)
            target_mean_norm = target_mean / (num_bins - 1)
            running_mse_loss += _criterion_mse(outputs_mean_norm, target_mean_norm).item()
            running_mae_loss += F.l1_loss(outputs_mean_norm, target_mean_norm).item()
            running_emd_loss += loss.item()
            progress_bar.set_postfix({'EMD': loss.item()})

    predicted_scores = np.concatenate(mean_pred, axis=0)
    true_scores = np.concatenate(mean_target, axis=0)
    srocc_GIAA, _ = spearmanr(predicted_scores, true_scores)

    mu_p, mu_t = predicted_scores.mean(), true_scores.mean()
    var_p, var_t = predicted_scores.var(), true_scores.var()
    cov = ((predicted_scores - mu_p) * (true_scores - mu_t)).mean()
    ccc = float(2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8))

    srocc_PIAA = 0
    ndcg_PIAA = 0
    if PIAA:
        unique_user_ids = np.unique(user_id_list)
        sroccs = []
        ndcgs = []
        for uid in unique_user_ids:
            uid_mask = (user_id_list == uid)
            if np.sum(uid_mask) > 1:
                uid_srocc, _ = spearmanr(predicted_scores[uid_mask], true_scores[uid_mask])
                sroccs.append(uid_srocc)
                uid_ndcg = ndcg_score([true_scores[uid_mask]], [predicted_scores[uid_mask]], k=10)
                ndcgs.append(uid_ndcg)
        srocc_PIAA = np.mean(sroccs)
        ndcg_PIAA = np.mean(ndcgs) if len(ndcgs) > 0 else 0

    emd_loss = running_emd_loss / len(dataloader)
    mse_loss = running_mse_loss / len(dataloader)
    mae_loss = running_mae_loss / len(dataloader)
    return emd_loss, srocc_GIAA, srocc_PIAA, mse_loss, ndcg_PIAA, mae_loss, ccc


# ─── PIAA Evaluation ──────────────────────────────────────────────────────────

def _collect_user_ids(user_ids) -> list:
    """Convert various user_id formats from collate_fn to a flat list of ints."""
    if isinstance(user_ids, torch.Tensor):
        return user_ids.view(-1).cpu().numpy().tolist()
    result = []
    for u in user_ids:
        result.append(int(u.item()) if isinstance(u, torch.Tensor) else int(u))
    return result


def evaluate_piaa(model, dataloaders_dict, device, epoch: int = None, phase_name: str = "Val"):
    """
    Evaluate PIAA model across all genres.
    Returns:
        genre_metrics: dict of {genre: {'srocc': float, 'mae': float, 'ndcg@10': float, 'ccc': float}}
        total_mae_loss: average MAE loss across all genres
    """
    model.eval()

    desc = f"Epoch {epoch} [{phase_name}]" if epoch is not None else phase_name
    total_expected_batches = sum(len(loader) for loader in dataloaders_dict.values())
    progress_bar = tqdm(total=total_expected_batches, leave=True, desc=desc, position=1, ncols=120, colour="#fffb00", ascii="-=")

    genre_predictions = defaultdict(list)
    genre_targets = defaultdict(list)
    genre_user_ids = defaultdict(list)
    genre_mae_losses = {}
    genre_batch_counts = {}

    component_interaction = defaultdict(float)
    component_direct = defaultdict(float)
    component_batch_counts = defaultdict(int)
    autocast_ctx = _get_autocast(device)

    with torch.no_grad():
        for genre, dataloader in dataloaders_dict.items():
            running_mae = 0.0
            batch_count = 0

            for sample in dataloader:
                images = sample['image'].to(device)
                target = sample['Aesthetic'].to(device).view(-1, 1)
                sample_pt = sample['traits'].float().to(device)
                sample_attr = sample['QIP'].float().to(device)

                if 'user_id' in sample:
                    genre_user_ids[genre].extend(_collect_user_ids(sample['user_id']))

                with autocast_ctx:
                    outputs = model(images, sample_pt, sample_attr, genre)
                outputs = outputs.view(-1, 1)
                genre_predictions[genre].append(outputs.view(-1).cpu().numpy())
                genre_targets[genre].append(target.view(-1).cpu().numpy())
                mae = F.l1_loss(outputs, target)
                running_mae += mae.item()
                component_interaction[genre] += getattr(model, '_last_interaction_mean', 0.0)
                component_direct[genre] += getattr(model, '_last_direct_mean', 0.0)
                component_batch_counts[genre] += 1
                batch_count += 1
                progress_bar.update(1)
                progress_bar.set_postfix({'MAE': mae.item(), 'genre': genre})

            genre_mae_losses[genre] = running_mae
            genre_batch_counts[genre] = batch_count

    model._eval_component_stats = {}
    for genre in dataloaders_dict.keys():
        n = max(component_batch_counts[genre], 1)
        i_mean = component_interaction[genre] / n
        d_mean = component_direct[genre] / n
        model._eval_component_stats[genre] = {
            'interaction_mean': i_mean,
            'direct_mean': d_mean,
            'ratio': i_mean / (i_mean + d_mean) if (i_mean + d_mean) > 0 else 0.0,
        }

    progress_bar.close()

    genre_metrics = {}
    total_mae_loss = 0.0

    for genre in dataloaders_dict.keys():
        if len(genre_predictions[genre]) == 0:
            continue

        predicted_scores = np.concatenate(genre_predictions[genre], axis=0)
        true_scores = np.concatenate(genre_targets[genre], axis=0)
        user_ids = genre_user_ids[genre]
        if len(user_ids) == 0:
            raise ValueError(
                f"No user_id found for genre '{genre}'. "
                f"The dataset must contain 'user_id' field for per-user SROCC computation."
            )

        unique_user_ids = np.unique(user_ids)
        sroccs = []
        plccs = []
        ndcgs = []
        cccs = []
        for uid in unique_user_ids:
            uid_mask = (np.array(user_ids) == uid)
            n_samples = np.sum(uid_mask)
            if n_samples <= 1:
                raise ValueError(
                    f"User {uid} in genre '{genre}' has only {n_samples} sample(s). "
                    f"At least 2 samples are required for SROCC computation."
                )
            uid_pred = predicted_scores[uid_mask]
            uid_true = true_scores[uid_mask]
            uid_srocc, _ = spearmanr(uid_pred, uid_true)
            sroccs.append(uid_srocc)
            uid_plcc, _ = pearsonr(uid_pred, uid_true)
            plccs.append(uid_plcc)
            uid_ndcg = ndcg_score([uid_true], [uid_pred], k=10)
            ndcgs.append(uid_ndcg)
            mu_p, mu_t = uid_pred.mean(), uid_true.mean()
            var_p, var_t = uid_pred.var(), uid_true.var()
            cov = ((uid_pred - mu_p) * (uid_true - mu_t)).mean()
            uid_ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t) ** 2 + 1e-8)
            cccs.append(float(uid_ccc))
        genre_srocc = np.mean(sroccs) if len(sroccs) > 0 else 0.0
        genre_plcc = np.mean(plccs) if len(plccs) > 0 else 0.0
        genre_ndcg = np.mean(ndcgs) if len(ndcgs) > 0 else 0.0
        genre_ccc = np.mean(cccs) if len(cccs) > 0 else 0.0

        genre_mae = genre_mae_losses[genre] / genre_batch_counts[genre] if genre_batch_counts[genre] > 0 else 0.0

        genre_metrics[genre] = {
            'srocc': genre_srocc,
            'plcc': genre_plcc,
            'mae': genre_mae,
            'ndcg@10': genre_ndcg,
            'ccc': genre_ccc,
        }
        total_mae_loss += genre_mae_losses[genre]

    total_batch_count = sum(genre_batch_counts.values())
    total_mae_loss = total_mae_loss / total_batch_count if total_batch_count > 0 else 0.0

    return genre_metrics, total_mae_loss
