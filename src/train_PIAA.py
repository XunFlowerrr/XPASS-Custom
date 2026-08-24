import os
import copy
from .argflags import parse_arguments, model_dir, MODELS_ROOT, get_device
from .data import load_data, build_global_encoders
from .train_common import discover_folds
from .methods import source_only
from .inference import inference_finetune, evaluate_pretrain_on_val_piaa, inference_pretrain
from .progress import ProgressTracker

num_attr = None  # Determined dynamically from dataset
num_pt = None    # Determined dynamically from dataset


def count_users_for_run(root_dir, fold, genre):
    """Count unique user IDs in train_PIAA.txt for a fold/genre if available."""
    split_file = os.path.join(root_dir, 'split', fold, genre, 'train_PIAA.txt')
    if os.path.exists(split_file):
        try:
            with open(split_file, 'r') as f:
                lines = [line.strip().split()[0] for line in f if line.strip() and len(line.strip().split()) >= 2]
            return len(set(lines))
        except Exception:
            return 0
    return 0


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


def run_main(args, tracker=None):
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
            num_attr, num_pt, tracker=tracker)
        evaluate_pretrain_on_val_piaa(datasets_dict_user, args, device, backbone_dict, best_model_path, model_state_dict=best_state_dict)
        inference_pretrain(datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict, best_model_path, model_state_dict=best_state_dict)
    elif args.piaa_mode == 'PIAA_finetune':
        source_only.trainer_finetune(
            datasets_dict_user, args, device, dirname, experiment_name, backbone_dict, pretrained_model_dict,
            num_attr, num_pt, tracker=tracker)
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

    # Pre-calculate scheduled runs
    scheduled_runs = []
    for source in source_genres:
        if args.dataset_ver.endswith('_all'):
            version_prefix = args.dataset_ver[:-4]
            folds = discover_folds(args.root_dir, version_prefix)
            active_folds = [f for i, f in enumerate(folds) if (i + 1) >= args.start_fold]
            for f in active_folds:
                scheduled_runs.append((source, f))
        else:
            scheduled_runs.append((source, args.dataset_ver))

    # Pre-calculate total models / users
    if args.piaa_mode == 'PIAA_finetune':
        total_models = sum(count_users_for_run(args.root_dir, f, g) for g, f in scheduled_runs)
        if total_models == 0:
            total_models = len(scheduled_runs)
    else:
        total_models = len(scheduled_runs)

    tracker = ProgressTracker(
        total_genres=len(source_genres),
        total_folds=max(1, len(scheduled_runs) // len(source_genres)),
        total_models=total_models,
        max_epochs=args.num_epochs,
        mode_name=f"PIAA {args.piaa_mode} ({args.model_type})"
    )
    tracker.print_initial_summary()

    for g_idx, source in enumerate(source_genres):
        args_outer = copy.deepcopy(args)
        args_outer.genre = source
        if len(source_genres) > 1:
            print(f"\n{'@'*60}\n  Genre: {source} ({g_idx + 1}/{len(source_genres)})\n{'@'*60}\n")

        if args_outer.dataset_ver.endswith('_all'):
            version_prefix = args_outer.dataset_ver[:-4]
            folds = discover_folds(args_outer.root_dir, version_prefix)
            if not folds:
                raise ValueError(f"No fold directories found for version '{version_prefix}' in {os.path.join(args_outer.root_dir, 'split')}")
            active_folds = [f for i, f in enumerate(folds) if (i + 1) >= args_outer.start_fold]
            print(f"Running all {len(active_folds)} folds sequentially: {active_folds}")
            for f_idx, fold in enumerate(active_folds):
                print(f"\n{'='*60}")
                print(f"  Fold {f_idx + 1}/{len(active_folds)}: {fold}")
                print(f"{'='*60}\n")
                args_fold = copy.deepcopy(args_outer)
                args_fold.dataset_ver = fold
                tracker.set_context(genre_idx=g_idx + 1, genre_name=source, fold_idx=f_idx + 1, fold_name=fold)
                run_main(args_fold, tracker=tracker)
        else:
            models_base = MODELS_ROOT
            fold_dirs = sorted([
                d for d in os.listdir(models_base)
                if d.startswith(f'{args_outer.dataset_ver}_fold') and os.path.isdir(os.path.join(models_base, d))
            ]) if os.path.exists(models_base) else []

            if fold_dirs:
                print(f"{MODELS_ROOT}/{args_outer.dataset_ver}/ not found. Running fold structure: {fold_dirs}")
                active_folds = [f for i, f in enumerate(fold_dirs) if (i + 1) >= args_outer.start_fold]
                for f_idx, fold in enumerate(active_folds):
                    print(f"\n{'='*60}")
                    print(f"  Fold {f_idx + 1}/{len(active_folds)}: {fold}")
                    print(f"{'='*60}\n")
                    args_fold = copy.deepcopy(args_outer)
                    args_fold.dataset_ver = fold
                    tracker.set_context(genre_idx=g_idx + 1, genre_name=source, fold_idx=f_idx + 1, fold_name=fold)
                    run_main(args_fold, tracker=tracker)
            else:
                tracker.set_context(genre_idx=g_idx + 1, genre_name=source, fold_idx=1, fold_name=args_outer.dataset_ver)
                run_main(args_outer, tracker=tracker)
