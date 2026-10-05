import argparse
import pandas as pd
from pathlib import Path


def parse_trec_eval_output(file_path):
    metrics = {}
    with open(file_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "all":
                if parts[0] == "recip_rank":
                    metrics["MRR"] = float(parts[2])
                elif parts[0] == "recall_10":
                    metrics["Recall@10"] = float(parts[2])
                elif parts[0] == "ndcg_cut_10":
                    metrics["NDCG@10"] = float(parts[2])
    return metrics


def compare_datasets(datasets, pool_sizes, output_file):
    data = []
    for pool_size in pool_sizes:
        for results_dir, name in datasets:
            result_file = Path(results_dir) / f"results_pool_{pool_size}.txt"
            if not result_file.exists():
                continue
            metrics = parse_trec_eval_output(result_file)
            if metrics:
                data.append({
                    "pool_size": pool_size,
                    "dataset": name,
                    "MRR": metrics.get("MRR", 0.0),
                    "Recall@10": metrics.get("Recall@10", 0.0),
                    "NDCG@10": metrics.get("NDCG@10", 0.0),
                })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"\n{'Dataset':<25} {'Pool Size':<12} {'MRR':<10} {'Recall@10':<12} {'NDCG@10':<10}")
    print("=" * 75)
    for _, row in df.iterrows():
        print(f"{row['dataset']:<25} {row['pool_size']:<12} {row['MRR']:<10.4f} {row['Recall@10']:<12.4f} {row['NDCG@10']:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=[], help="results_dir:name")
    parser.add_argument("--base_dir", default=None, help="Auto-discover results subdirs here")
    parser.add_argument("--pool_sizes", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.dataset:
        datasets = [item.split(":", 1) for item in args.dataset]
    elif args.base_dir:
        base = Path(args.base_dir)
        datasets = []
        gt = base / "ground_truth" / "results"
        if gt.exists():
            datasets.append((str(gt), "Ground Truth"))
        for d in sorted(base.iterdir()):
            if d.name == "ground_truth" or not d.is_dir():
                continue
            results = d / "results"
            if results.exists():
                datasets.append((str(results), d.name))
    else:
        parser.error("provide --dataset or --base_dir")

    compare_datasets(datasets, args.pool_sizes, args.output)
