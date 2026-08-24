import os
import copy
from datetime import datetime
import torch

from .argflags import parse_arguments, model_dir, MODELS_ROOT, get_device
from .data import load_data, build_global_encoders
from .train_common import discover_folds
from .methods import source_only
from .inference import inference_finetune, evaluate_pretrain_on_val_piaa, inference_pretrain

num_attr = None  # Determined dynamically from dataset
num_pt = None    # Determined dynamically from dataset


def discover_pretrained_models(dataset_ver, genre, piaa_mode='PIAA_finetune', model_type=None):
    """Auto-discover pretrained model files under models_pth/{dataset_ver}/{genre}/.

    For PIAA_pretrain: finds the NIMA .pth.
    For PIAA_finetune: finds *_pretrain.pth, filtered by model_type (ICI or MIR) if specified.
    """
    genre_dir = os.path.join(MODELS_ROOT, dataset_ver, genre)
    if not os.path.isdir(genre_dir):
        raise FileNotFoundError(f"Directory not found: {genre_dir}")

    if piaa_mode == 'PIAA_pretrain':
        nima_files = [f for f in os.listdir(genre_dir) if 'NIMA' in f and f.endswith('.pth')]
        if len(nima_files) == 1:
            return {genre: os.path.join(genre_dir, nima_files[0])}
        elif len(nima_files) > 1:
            raise ValueError(f"Multiple NIMA pth files found in {genre_dir}: {nima_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No NIMA pth file found in {genre_dir}")
    else:
        all_pretrain_files = [f for f in os.listdir(genre_dir) if f.endswith('_pretrain.pth')]
        if model_type is not None:
            pretrain_files = [f for f in all_pretrain_files if f'_{model_type}_' in f]
        else:
            pretrain_files = all_pretrain_files
        if len(pretrain_files) == 1:
            return {genre: os.path.join(genre_dir, pretrain_files[0])}
        elif len(pretrain_files) > 1:
            raise ValueError(f"Multiple pretrain pth files found in {genre_dir}: {pretrain_files}. Please specify --pretrained_model explicitly.")
        else:
            raise FileNotFoundError(f"No pretrain pth file found in {genre_dir}")


def run_main(args):
    global num_pt, num_attr

    genre = args.genre.strip()
    if ',' in genre:
        raise ValueError(
            f"Multi-domain training has been removed. "
            f"Specify a single genre (e.g., --genre art), got: '{genre}'"
        )
    genres = [genre]
    print(f"Training with genre: {genre}")

    backbone_dict = {genre: args.backbone}
    print(f"Backbone: {backbone_dict[genre]}")

    pretrained_model_dict = discover_pretrained_models(args.dataset_ver, genre, args.piaa_mode, getattr(args, 'model_type', None))
    print(f"Auto-discovered pretrained models: {pretrained_model_dict}")

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"Only_{run_name}"

    print(args)

    global_trait_encoders, global_age_bins = build_global_encoders(args.root_dir)

    _, train_piaa_dataset, train_giaa_dataset, _, val_piaa_dataset, val_giaa_dataset, test_piaa_dataset = load_data(
        args, global_trait_encoders=global_trait_encoders, global_age_bins=global_age_bins)

    datasets_dict = {genre: {'train': train_giaa_dataset, 'val': val_giaa_dataset, 'test': test_piaa_dataset}}
    datasets_dict_user = {genre: {'train': train_piaa_dataset, 'val': val_piaa_dataset, 'test': test_piaa_dataset}}

    _sample = train_giaa_dataset[0]
    num_pt = len(_sample['traits'])
    num_attr = len(_sample['QIP'])
    print(f"Detected num_pt={num_pt}, num_attr={num_attr} from dataset")

    device = get_device(getattr(args, 'device', 'auto'))
    print(f"Using device: {device}")
    dirname = os.path.join(model_dir(args), genre)
    os.makedirs(dirname, exist_ok=True)

    if args.piaa_mode == 'PIAA_pretrain':
        best_model_path, best_state_dict = source_only.trainer_pretrain(
            datasets_dict, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
            num_attr, num_pt)
        evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=best_state_dict)
        inference_pretrain(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, model_state_dict=best_state_dict)
    elif args.piaa_mode == 'PIAA_finetune':
        source_only.trainer_finetune(
            datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
            num_attr, num_pt)
        inference_finetune(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict)
        if not args.keep_finetune_pth:
            for pth_file in [f for f in os.listdir(dirname) if f.endswith('_finetune.pth')]:
                os.remove(os.path.join(dirname, pth_file))
            print(f"Deleted temporary finetune model files from {dirname}")
        else:
            print(f"Kept finetune model files in {dirname} (--keep_finetune_pth)")
    else:
        raise ValueError(f"Error: --piaa_mode must be 'PIAA_pretrain' or 'PIAA_finetune', got: {args.piaa_mode}")

if __name__ == '__main__':
    parser = parse_arguments(parse=False)
    parser.add_argument('--model_type', type=str, default='ICI', choices=['ICI', 'MIR'],
                        help='PIAA model architecture: ICI (Interaction-based) or MIR (MLP Interaction Regression)')
    parser.add_argument('--keep_finetune_pth', action='store_true', default=False,
                        help='Keep *_finetune.pth files after inference (default: delete them)')
    import sys
    parser.set_defaults(lr=5e-6, batch_size=32)
    args = parser.parse_args()
    if args.piaa_mode == 'PIAA_finetune':
        if not any(a.startswith('--lr') for a in sys.argv[1:]):
            args.lr = 1e-5
        if not any(a.startswith('--batch_size') for a in sys.argv[1:]):
            args.batch_size = 16

    ALL_GENRES = ['art', 'fashion', 'scenery']
    if args.genre == 'all':
        source_genres = list(ALL_GENRES)
        print(f"--genre all: running genres sequentially: {source_genres}")
    else:
        source_genres = [args.genre]

    for source in source_genres:
        args_outer = copy.deepcopy(args)
        args_outer.genre = source
        if len(source_genres) > 1:
            print(f"\n{'@'*60}\n  Genre: {source}\n{'@'*60}\n")

        if args_outer.dataset_ver.endswith('_all'):
            version_prefix = args_outer.dataset_ver[:-4]
            folds = discover_folds(args_outer.root_dir, version_prefix)
            if not folds:
                raise ValueError(f"No fold directories found for version '{version_prefix}' in {os.path.join(args_outer.root_dir, 'split')}")
            print(f"Running all {len(folds)} folds sequentially: {folds}")
            for i, fold in enumerate(folds):
                if i + 1 < args_outer.start_fold:
                    print(f"Skipping fold {i+1}/{len(folds)}: {fold} (start_fold={args_outer.start_fold})")
                    continue
                print(f"\n{'='*60}")
                print(f"  Fold {i+1}/{len(folds)}: {fold}")
                print(f"{'='*60}\n")
                args_fold = copy.deepcopy(args_outer)
                args_fold.dataset_ver = fold
                run_main(args_fold)
        else:
            models_base = MODELS_ROOT
            fold_dirs = sorted([
                d for d in os.listdir(models_base)
                if d.startswith(f'{args_outer.dataset_ver}_fold') and os.path.isdir(os.path.join(models_base, d))
            ]) if os.path.exists(models_base) else []

            if fold_dirs:
                print(f"{MODELS_ROOT}/{args_outer.dataset_ver}/ not found. Running fold structure: {fold_dirs}")
                for i, fold in enumerate(fold_dirs):
                    if i + 1 < args_outer.start_fold:
                        print(f"Skipping fold {i+1}/{len(fold_dirs)}: {fold} (start_fold={args_outer.start_fold})")
                        continue
                    print(f"\n{'='*60}")
                    print(f"  Fold {i+1}/{len(fold_dirs)}: {fold}")
                    print(f"{'='*60}\n")
                    args_fold = copy.deepcopy(args_outer)
                    args_fold.dataset_ver = fold
                    run_main(args_fold)
            else:
                run_main(args_outer)
