import os
import contextlib

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

import copy

from ..train_common import earth_mover_distance, build_piaa_model, num_bins, trainable_state
from ..data import make_loader
from ..evaluate import evaluate, evaluate_piaa
from ..checkpoint_callback import safe_save_checkpoint, on_checkpoint_saved


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


def _get_grad_scaler(device):
    """Return a GradScaler compatible with device (cuda, mps, cpu)."""
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    if dev_type == 'cuda':
        return GradScaler('cuda')
    elif dev_type == 'mps':
        try:
            return GradScaler('mps')
        except (TypeError, ValueError, RuntimeError):
            try:
                return GradScaler('mps', enabled=False)
            except (TypeError, ValueError, RuntimeError):
                return GradScaler('cuda', enabled=False)
    try:
        return GradScaler('cpu', enabled=False)
    except (TypeError, ValueError, RuntimeError):
        return GradScaler('cuda', enabled=False)


def setup(model, args, device):
    """No extra components for source-only training."""
    return {}


def _train_one_epoch(model, dataloader, optimizer, scaler, device, args, epoch: int = None, tracker=None):
    model.train()
    running_emd_loss = 0.0
    if tracker is not None:
        desc = tracker.get_full_header(phase="Train")
    else:
        desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(dataloader, leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")
    autocast_ctx = _get_autocast(device)
    for sample in progress_bar:
        images = sample['image'].to(device)
        aesthetic_score_histogram = sample['Aesthetic'].to(device)

        optimizer.zero_grad()
        with autocast_ctx:
            aesthetic_logits = model(images)
            prob_aesthetic = F.softmax(aesthetic_logits, dim=1)
            loss = earth_mover_distance(prob_aesthetic, aesthetic_score_histogram).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_val = loss.item()
        running_emd_loss += loss_val
        progress_bar.set_postfix({'Train EMD': loss_val})

    return running_emd_loss / len(dataloader)


def trainer(src_dataloaders, model, optimizer, args, device, best_modelname, components, tracker=None):
    train_dataloader, val_dataloader, _ = src_dataloaders

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay_factor, patience=args.lr_patience)
    scaler = _get_grad_scaler(device)

    best_val_emd = float('inf')
    patience = 0
    for epoch in range(args.num_epochs):
        if tracker is not None:
            tracker.start_epoch(epoch)

        train_emd = _train_one_epoch(model, train_dataloader, optimizer, scaler, device, args, epoch=epoch, tracker=tracker)

        val_emd, val_srocc, _, val_mse, _, _, val_ccc = evaluate(
            model, val_dataloader, device, epoch=epoch, phase_name="Val")

        if tracker is not None:
            tracker.end_epoch(epoch)
            timing_str = f"  {tracker.get_timing_info()}"
        else:
            timing_str = ""

        tqdm.write(f"[{args.genre}] epoch {epoch}  Train EMD {train_emd:.4f}  "
                   f"Val EMD {val_emd:.4f}  SROCC {val_srocc:.4f}  CCC {val_ccc:.4f}{timing_str}")

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_emd)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        if val_emd < best_val_emd:
            best_val_emd = val_emd
            patience = 0
            safe_save_checkpoint(model.state_dict(), best_modelname)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"Validation loss has not decreased for {args.max_patience_epochs} epochs. Stopping training.")
                break

    if tracker is not None:
        tracker.finish_model(early_stopped=(patience >= args.max_patience_epochs))

    model.load_state_dict(torch.load(best_modelname, map_location=device))
    dataset_ver = getattr(args, 'dataset_ver', 'default')
    genre = getattr(args, 'genre', 'default')
    on_checkpoint_saved(best_modelname, dataset_ver, genre, args=args, is_temporary=False)


# ─── PIAA ─────────────────────────────────────────────────────────────────────

def _train_one_epoch_piaa(model, dataloader, optimizer, scaler, device, args, genre, epoch=None, tracker=None):
    model.train()
    running_loss = 0.0
    total_batches = 0

    if tracker is not None:
        desc = tracker.get_full_header(phase="Train")
    else:
        desc = f"Epoch {epoch} [Train]" if epoch is not None else "Train"
    progress_bar = tqdm(total=len(dataloader), leave=True, desc=desc, position=0, ncols=120, colour="#00ff00", ascii="-=")
    autocast_ctx = _get_autocast(device)

    for sample in dataloader:
        optimizer.zero_grad()
        images = sample['image'].to(device)
        sample_pt = sample['traits'].float().to(device)
        sample_attr = sample['QIP'].float().to(device)
        aesthetic_scores = sample['Aesthetic'].to(device).view(-1, 1)

        with autocast_ctx:
            score_pred = model(images, sample_pt, sample_attr, genre)
            loss = F.mse_loss(score_pred, aesthetic_scores)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        running_loss += loss_val
        total_batches += 1
        progress_bar.update(1)
        progress_bar.set_postfix({'loss': loss_val})

    progress_bar.close()
    n = total_batches if total_batches > 0 else 1
    return running_loss / n


def trainer_pretrain(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                     num_attr, num_pt, tracker=None):
    """
    Pretrain trainer for a single genre.
    Trains on GIAA data with NIMA initialization, uses val GIAA data for early stopping.
    Returns:
        best_model_path: path to saved best model
        best_state_dict: state dict if args.no_save_model, else None
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    _cache_bytes = int(getattr(args, 'pixel_cache_gb', 0.0) * 1024 ** 3)
    if _cache_bytes:
        _mb = (datasets_dict[genre]['train'].warm_pixel_cache(_cache_bytes)
               + datasets_dict[genre]['val'].warm_pixel_cache(_cache_bytes)) / 1e6
        print(f"[PixelCache] Decoded {_mb:.0f} MB of images once; epochs reuse them.")
    train_loader = make_loader(datasets_dict[genre]['train'], batch_size, args.num_workers,
                               shuffle=True, drop_last=True, reused=True)
    val_loaders_dict = {genre: make_loader(datasets_dict[genre]['val'], batch_size,
                                           args.num_workers, reused=True)}

    model = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)

    pretrained_path = pretrained_model_dict[genre]
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Error: Pretrained NIMA model file not found: {pretrained_path}")
    try:
        state = torch.load(pretrained_path, map_location=device)
        model.nima_dict[genre].load_state_dict(state)
        print(f"Loaded NIMA weights for {genre} from {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load NIMA weights for {genre} from {pretrained_path}: {e}")

    model.freeze_backbone()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

    best_val_ccc = -float('inf')
    patience = 0
    best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_{experiment_name}_pretrain.pth')
    best_state_dict = None

    scaler = _get_grad_scaler(device)
    for epoch in range(args.num_epochs):
        if tracker is not None:
            tracker.start_epoch(epoch)

        train_loss = _train_one_epoch_piaa(model, train_loader, optimizer, scaler, device, args, genre, epoch=epoch, tracker=tracker)
        genre_metrics, val_mae = evaluate_piaa(model, val_loaders_dict, device, epoch=epoch, phase_name="Val")

        if tracker is not None:
            tracker.end_epoch(epoch)
            timing_str = f"  {tracker.get_timing_info()}"
        else:
            timing_str = ""

        val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

        if genre in genre_metrics:
            m = genre_metrics[genre]
            tqdm.write(f"[{genre}] epoch {epoch}  Train Loss {train_loss:.4f}  "
                       f"Val MAE {m['mae']:.4f}  SROCC {m['srocc']:.4f}  "
                       f"NDCG@10 {m['ndcg@10']:.4f}  CCC {m['ccc']:.4f}{timing_str}")

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_ccc)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (epoch {epoch}) <<<")

        if val_ccc > best_val_ccc:
            best_val_ccc = val_ccc
            patience = 0
            if args.no_save_model:
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                safe_save_checkpoint(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= args.max_patience_epochs:
                print(f"Pretrain: early stopping at epoch {epoch}")
                break

    if tracker is not None:
        tracker.finish_model(early_stopped=(patience >= args.max_patience_epochs))

    if not args.no_save_model:
        dataset_ver = getattr(args, 'dataset_ver', 'default')
        on_checkpoint_saved(best_model_path, dataset_ver, genre, args=args, is_temporary=False)

    return best_model_path, best_state_dict


def trainer_finetune(datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
                     num_attr, num_pt, tracker=None):
    """
    Finetune trainer for a single genre, one model per user.
    """
    batch_size = args.batch_size
    genres = list(datasets_dict.keys())
    genre = genres[0]
    genre_str = genre

    all_user_ids = set(datasets_dict[genre]['train'].data['user_id'].values)
    unique_user_ids = sorted(list(all_user_ids))
    _cache_bytes = int(getattr(args, 'pixel_cache_gb', 0.0) * 1024 ** 3)

    pretrained_path = pretrained_model_dict[genre]
    if pretrained_path is None or not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Error: Pretrained model file not found: {pretrained_path}")

    # Build the model and read the pretrain checkpoint once instead of once per
    # user: every user starts from the same weights, and `create_model_and_transforms`
    # rebuilds an 86M-parameter CLIP tower each time it is called.
    model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
    try:
        base_state = torch.load(pretrained_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load model weights from {pretrained_path}: {e}")

    incompatible = model_user.load_state_dict(base_state, strict=False)
    if incompatible.unexpected_keys:
        print(f"[load_state_dict] Ignored unexpected keys: {incompatible.unexpected_keys}")
    print(f"Loaded PIAA weights from {pretrained_path}")

    # Reusing one model object is only equivalent to rebuilding it per user when
    # the checkpoint covers every key -- otherwise the keys it misses would carry
    # the previous user's trained values instead of a fresh initialization.
    reuse_model = not incompatible.missing_keys
    if not reuse_model:
        print(f"[Reuse] Checkpoint is missing {len(incompatible.missing_keys)} key(s) "
              f"(e.g. {incompatible.missing_keys[:3]}); rebuilding the model per user "
              f"to keep initialization semantics unchanged.")

    _full_state = model_user.state_dict()
    _kept = len(trainable_state(_full_state))
    print(f"[Checkpoint] Saving {_kept} trainable tensors per user; the remaining "
          f"{len(_full_state) - _kept} frozen tensors are restored from the pretrain checkpoint at load time.")

    for uid_idx, uid in enumerate(unique_user_ids):
        print(f"Training for user {uid} ({uid_idx + 1}/{len(unique_user_ids)})...")
        if tracker is not None:
            tracker.set_context(model_idx=uid_idx + 1, model_name=str(uid))

        user_train_ds = copy.copy(datasets_dict[genre]['train'])
        user_train_ds.data = datasets_dict[genre]['train'].data[datasets_dict[genre]['train'].data['user_id'] == uid].reset_index(drop=True)

        user_val_ds = copy.copy(datasets_dict[genre]['val'])
        user_val_ds.data = datasets_dict[genre]['val'].data[datasets_dict[genre]['val'].data['user_id'] == uid].reset_index(drop=True)

        total_train_samples = len(user_train_ds)
        total_val_samples = len(user_val_ds)
        print(f"User {uid}: train {total_train_samples} val {total_val_samples} samples")
        if total_train_samples == 0 or total_val_samples == 0:
            print(f"Skipping user {uid}: insufficient data")
            if tracker is not None:
                tracker.finish_model(early_stopped=True)
            continue

        # Decode this user's images once here, in the parent, so the forked workers
        # share them; every epoch would otherwise re-read the same few hundred JPEGs.
        _cached = (user_train_ds.warm_pixel_cache(_cache_bytes)
                   + user_val_ds.warm_pixel_cache(_cache_bytes))
        if uid == unique_user_ids[0] and _cached:
            print(f"[PixelCache] {_cached / 1e6:.0f} MB per user held in RAM across epochs.")
        train_loader = make_loader(user_train_ds, batch_size, args.num_workers,
                                   shuffle=True, drop_last=True, reused=True)
        val_loaders_dict = {genre: make_loader(user_val_ds, batch_size, args.num_workers,
                                               reused=True)}

        if reuse_model:
            # Every parameter and buffer is covered by the checkpoint, so reloading
            # it returns the model to exactly the state a fresh build would produce.
            model_user.load_state_dict(base_state, strict=False)
        else:
            model_user = build_piaa_model(num_bins, num_attr, num_pt, genres, backbone_dict, args).to(device)
            model_user.load_state_dict(base_state, strict=False)

        model_user.freeze_backbone()
        if uid == unique_user_ids[0]:
            total_params = sum(p.numel() for p in model_user.parameters())
            trainable_params = sum(p.numel() for p in model_user.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            print(f"[Freeze] Backbone frozen: {frozen_params:,} frozen / {trainable_params:,} trainable / {total_params:,} total")

        optimizer_user = optim.AdamW(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr)
        scheduler_user = optim.lr_scheduler.ReduceLROnPlateau(optimizer_user, mode='max', factor=args.lr_decay_factor, patience=args.lr_patience)

        best_val_ccc = -float('inf')
        patience = 0
        best_model_path = os.path.join(dirname, f'{genre_str}_{args.model_type}_user_{uid}_{experiment_name}_finetune.pth')

        scaler = _get_grad_scaler(device)

        # epoch 0 前に pretrain 重みを保存しておく:
        # val_ccc が一度も改善しない (NaN 等) ユーザーでも .pth が必ず存在し、
        # inference 時の "best model not found" によるユーザー欠損を防ぐ。
        safe_save_checkpoint(trainable_state(model_user.state_dict()), best_model_path)

        for epoch in range(args.num_epochs):
            if tracker is not None:
                tracker.start_epoch(epoch)

            train_loss = _train_one_epoch_piaa(model_user, train_loader, optimizer_user, scaler, device, args, genre, epoch=epoch, tracker=tracker)
            genre_metrics, val_mae = evaluate_piaa(model_user, val_loaders_dict, device, epoch=epoch, phase_name="Val")

            if tracker is not None:
                tracker.end_epoch(epoch)
                timing_str = f"  {tracker.get_timing_info()}"
            else:
                timing_str = ""

            val_ccc = genre_metrics[genre]['ccc'] if genre in genre_metrics else -float('inf')

            prev_lr = optimizer_user.param_groups[0]['lr']
            scheduler_user.step(val_ccc)
            cur_lr = optimizer_user.param_groups[0]['lr']
            if cur_lr < prev_lr:
                tqdm.write(f">>> LR reduced: {prev_lr:.2e} -> {cur_lr:.2e}  (user {uid}, epoch {epoch}) <<<")

            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                patience = 0
                safe_save_checkpoint(trainable_state(model_user.state_dict()), best_model_path)
            else:
                patience += 1
                if patience >= args.max_patience_epochs:
                    print(f"User {uid}: early stopping at epoch {epoch}")
                    break

        if tracker is not None:
            tracker.finish_model(early_stopped=(patience >= args.max_patience_epochs))

        dataset_ver = getattr(args, 'dataset_ver', 'default')
        on_checkpoint_saved(best_model_path, dataset_ver, genre, args=args, is_temporary=False)
