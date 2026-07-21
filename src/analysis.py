"""Result aggregation utilities for XPASS-SIMPLE (in-domain evaluation).

`aggregate` collects per-fold PIAA inference JSONs and reports fold/user-averaged
metrics (SROCC / NDCG@10 / MAE / CCC).
"""
import math
import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "exp"


def _spearman(x, y):
    """Spearman rank correlation (handles ties via average rank)."""
    n = len(x)

    def _rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and a[order[j]] == a[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    )
    return num / (den + 1e-10)


def _ndcg_at_k(true_scores, pred_scores, k=10):
    """NDCG@k with exponential gain (2^rel - 1), matching sklearn default."""
    n = len(true_scores)
    k = min(k, n)
    order = sorted(range(n), key=lambda i: pred_scores[i], reverse=True)
    dcg = sum(
        (2.0 ** true_scores[order[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    ideal = sorted(range(n), key=lambda i: true_scores[i], reverse=True)
    idcg = sum(
        (2.0 ** true_scores[ideal[i]] - 1.0) / math.log2(i + 2)
        for i in range(k)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


def _stats(vals):
    avg = sum(vals) / len(vals)
    std = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
    return avg, std


def aggregate(args):
    """指定された version と genre の各 fold から JSON を集約し，平均指標を出力する。"""
    version = args.version
    genre = args.genre
    pattern = args.pattern

    method = args.method  # e.g., "ICI" (optional)
    min_id = args.min_id
    max_id = args.max_id
    ids = set(args.ids) if args.ids is not None else None
    reports_dir = Path(args.reports_dir)

    fold_dirs = sorted(reports_dir.glob(f"{version}_fold*"))
    if not fold_dirs:
        print(
            f"Error: No fold directories found for version '{version}' in {reports_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.folds is not None:
        fold_set = set(args.folds)
        fold_dirs = [
            d for d in fold_dirs
            if int(d.name.split("fold")[-1]) in fold_set
        ]
        if not fold_dirs:
            print(
                f"Error: No matching fold directories for folds {args.folds}",
                file=sys.stderr,
            )
            sys.exit(1)

    all_user_mae   = {}
    all_user_ndcg  = {}
    all_user_srocc = {}
    all_user_ccc   = {}

    for fold_dir in fold_dirs:
        genre_dir = fold_dir / genre
        if not genre_dir.is_dir():
            print(f"Error: Genre directory not found: {genre_dir}", file=sys.stderr)
            sys.exit(1)

        # pattern に一致する JSON を検索（各 fold/genre に対して1つだけ存在する想定）
        if method and pattern:
            glob_pattern = f"*{method}*{pattern}*.json"
        elif method:
            glob_pattern = f"*{method}*.json"
        elif pattern:
            glob_pattern = f"*{pattern}*.json"
        else:
            glob_pattern = "*.json"
        matched_jsons = list(genre_dir.glob(glob_pattern))
        if min_id is not None or max_id is not None or ids is not None:
            def _extract_id(p):
                m = re.search(r'-(\d+)[_.]', p.name)
                return int(m.group(1)) if m else -1
            matched_jsons = [
                p for p in matched_jsons
                if (min_id is None or _extract_id(p) >= min_id)
                and (max_id is None or _extract_id(p) <= max_id)
                and (ids is None or _extract_id(p) in ids)
            ]
        if len(matched_jsons) == 0:
            print(f"Error: No JSON matching '{glob_pattern}' found in {genre_dir}", file=sys.stderr)
            sys.exit(1)
        if len(matched_jsons) > 1:
            print(
                f"Error: Multiple JSONs matching '{glob_pattern}' found in {genre_dir}: {[f.name for f in matched_jsons]}",
                file=sys.stderr,
            )
            sys.exit(1)

        json_path = matched_jsons[0]
        with open(json_path) as f:
            data = json.load(f)

        per_user = data.get("per_user_metrics", {})
        for user_id, metrics in per_user.items():
            genre_metrics = metrics.get(genre, {})
            mae  = genre_metrics.get("mae")
            ndcg = genre_metrics.get("ndcg@10")
            srocc = genre_metrics.get("srocc")
            ccc = genre_metrics.get("ccc")
            if mae is not None:
                all_user_mae.setdefault(user_id, []).append(mae)
            if ndcg is not None:
                all_user_ndcg.setdefault(user_id, []).append(ndcg)
            if srocc is not None:
                all_user_srocc.setdefault(user_id, []).append(srocc)
            if ccc is not None:
                all_user_ccc.setdefault(user_id, []).append(ccc)

        print(f"  Loaded: {json_path.relative_to(reports_dir)} ({len(per_user)} users)")

    if not all_user_mae:
        print("Error: No user metrics found.", file=sys.stderr)
        sys.exit(1)

    user_avg_mae   = [sum(v) / len(v) for v in all_user_mae.values()]
    user_avg_ndcg  = [sum(v) / len(v) for v in all_user_ndcg.values()]
    user_avg_srocc = [sum(v) / len(v) for v in all_user_srocc.values()]
    user_avg_ccc   = [sum(v) / len(v) for v in all_user_ccc.values()]

    avg_mae,   std_mae   = _stats(user_avg_mae)
    avg_ndcg,  std_ndcg  = _stats(user_avg_ndcg)
    avg_srocc, std_srocc = _stats(user_avg_srocc)
    avg_ccc,   std_ccc   = _stats(user_avg_ccc) if user_avg_ccc else (None, None)

    print(f"\n=== Aggregated Results ({version}, {genre}, pattern='{pattern}') ===")
    print(f"  Folds:           {len(fold_dirs)}")
    print(f"  Total users:     {len(all_user_mae)}")
    print(f"  Average MAE:     {avg_mae:.6f} (std: {std_mae:.6f})")
    print(f"  Average NDCG@10: {avg_ndcg:.6f} (std: {std_ndcg:.6f})")
    print(f"  Average SROCC:   {avg_srocc:.6f} (std: {std_srocc:.6f})")
    if avg_ccc is not None:
        print(f"  Average CCC:     {avg_ccc:.6f} (std: {std_ccc:.6f})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Result aggregation for XPASS-SIMPLE',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    agg_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate results across folds",
    )
    agg_parser.add_argument(
        "--version", type=str, required=True, help="Dataset version (e.g., v3)"
    )
    agg_parser.add_argument(
        "--genre", type=str, required=True, help="Genre (art / fashion / scenery)"
    )
    agg_parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Glob pattern to match JSON files (e.g., pretrain, finetune).",
    )
    agg_parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name to filter JSON files (e.g., ICI / MIR).",
    )
    agg_parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold indices to aggregate (e.g., --folds 1 3 5). If omitted, all folds are used.",
    )
    agg_parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to include (e.g., --ids 61 65 70).",
    )
    agg_parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        dest="min_id",
        help="Minimum run ID to include.",
    )
    agg_parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        dest="max_id",
        help="Maximum run ID to include.",
    )
    agg_parser.add_argument(
        "--reports_dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Path to reports/exp directory",
    )

    args = parser.parse_args()
    if args.command == 'aggregate':
        aggregate(args)
    else:
        parser.print_help()
