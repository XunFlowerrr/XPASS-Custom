import argparse
import os

# Project root (parent of the `src` directory). All data/model/report paths are
# resolved relative to this so the project is self-contained and CWD-independent.
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJ_ROOT, 'data')
MODELS_ROOT = os.path.join(PROJ_ROOT, 'models_pth')
REPORTS_ROOT = os.path.join(PROJ_ROOT, 'reports', 'exp')


def parse_arguments(parse=True):
    parser = argparse.ArgumentParser(description='Training and Testing the Combined Model for data splitting')

    parser.add_argument('--num_workers', type=int, default=4)
    # Dataset version identifier (string)
    parser.add_argument('--dataset_ver', type=str, default='v3_all', help='Dataset version (e.g., v3) used to locate split files and tag outputs')
    parser.add_argument('--start_fold', type=int, default=1, help='Fold number to start from (1-indexed). Use to resume from a specific fold when dataset_ver ends with _all.')
    parser.add_argument('--trait', type=str, default=None)
    parser.add_argument('--value', type=str, default=None)
    parser.add_argument('--genre', type=str, required=True, help='Dataset genre (e.g., art, fashion, scenery)')

    parser.add_argument('--backbone', type=str, default='clip_vit_b16',
                        choices=['clip_vit_b16'],
                        help='Backbone architecture for feature extraction')
    parser.add_argument('--root_dir', type=str, default=DATA_ROOT)
    parser.add_argument('--samples_root', type=str, default=None,
                        help='Override path to samples directory (default: <root_dir>/samples)')
    parser.add_argument('--piaa_mode', type=str, default='PIAA_pretrain')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_patience_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--no_save_model', action='store_true', default=False,
                        help='If set, keep best model in memory instead of saving to disk')

    if parse:
        return parser.parse_args()
    else:
        return parser

def model_dir(args):
    # Model directory scoped by dataset version instead of fold
    return os.path.join(MODELS_ROOT, f'{args.dataset_ver}')
