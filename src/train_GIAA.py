import os
import copy
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .argflags import parse_arguments, model_dir, get_device
from .data import load_data, collate_fn
from .train_common import NIMA, num_bins
from .methods import source_only
from .inference import inference


def run_main(args):
    batch_size = args.batch_size
    print(args, flush=True)

    _backbone_abbr = {'clip_vit_b16': 'CLViT'}
    backbone_suffix = f'_{_backbone_abbr[args.backbone]}' if args.backbone in _backbone_abbr else ''
    method_tag = 'Only' + backbone_suffix

    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.genre}_{method_tag}_PAA({run_name})"
    model_basename = f'{args.genre}_{method_tag}_NIMA_{run_name}.pth'

    device = get_device(getattr(args, 'device', 'auto'))
    print(f"Using device: {device}")
    dirname = os.path.join(model_dir(args), args.genre)
    best_modelname = os.path.join(dirname, model_basename)

    model = NIMA(num_bins, backbone=args.backbone, dropout=args.dropout).to(device)
    model.freeze_backbone()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    components = source_only.setup(model, args, device)

    (train_giaa_dataset, train_piaa_dataset, _,
     val_giaa_dataset, val_piaa_dataset, _,
     test_piaa_dataset) = load_data(args)

    train_giaa_loader = DataLoader(train_giaa_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=args.num_workers, timeout=(300 if args.num_workers > 0 else 0), collate_fn=collate_fn)
    val_giaa_loader = DataLoader(val_giaa_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=args.num_workers, timeout=(300 if args.num_workers > 0 else 0), collate_fn=collate_fn)
    test_piaa_loader = DataLoader(test_piaa_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=args.num_workers, timeout=(300 if args.num_workers > 0 else 0), collate_fn=collate_fn)
    src_dataloaders = (train_giaa_loader, val_giaa_loader, test_piaa_loader)

    source_only.trainer(src_dataloaders, model, optimizer, args, device, best_modelname, components)

    inference(train_piaa_dataset, val_piaa_dataset, test_piaa_dataset,
              args, device, model, eval_split="Val",
              experiment_name=experiment_name, model_path=best_modelname)
    inference(train_piaa_dataset, val_piaa_dataset, test_piaa_dataset,
              args, device, model, eval_split="Test",
              experiment_name=experiment_name, model_path=best_modelname)


# ──────────────────────────────────────────────────────────────────────────────

from .train_common import discover_folds  # noqa: E402

if __name__ == '__main__':
    args = parse_arguments()

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
                raise ValueError(
                    f"No fold directories found for version '{version_prefix}' in "
                    f"{os.path.join(args_outer.root_dir, 'split')}")
            print(f"Running all {len(folds)} folds sequentially: {folds}")
            for i, fold in enumerate(folds):
                if i + 1 < args_outer.start_fold:
                    print(f"Skipping fold {i+1}/{len(folds)}: {fold} (start_fold={args_outer.start_fold})")
                    continue
                print(f"\n{'='*60}\n  Fold {i+1}/{len(folds)}: {fold}\n{'='*60}\n")
                args_fold = copy.deepcopy(args_outer)
                args_fold.dataset_ver = fold
                run_main(args_fold)
        else:
            run_main(args_outer)