import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import load_all_datasets, pretty
from classify import load_classifier, classify
from metrics import EMOTIONS, distribution, kl_divergence, jsd_similarity
from plot import plot_radar


def evaluate(dataset_test: str, simulations_dir: str, output_dir: str, show_neutral: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_all_datasets(dataset_test, simulations_dir)

    labels_path = output_dir / "emotion_labels.csv"
    cached = {}
    if labels_path.exists():
        df_cached = pd.read_csv(labels_path)
        for key, grp in df_cached.groupby("dataset"):
            cached[key] = grp.to_dict("records")

    classifier = None
    distributions = {}
    all_rows = []
    for key, records in datasets.items():
        if key in cached:
            print(f"Skipping {key} (cached)...")
            rows = cached[key]
            labels = [r["emotion"] for r in rows]
        else:
            if classifier is None:
                classifier = load_classifier()
            print(f"Classifying {key} ({len(records)} records)...")
            texts = [r["text"] for r in records]
            labels = classify(texts, classifier)
            rows = [{"dataset": key, "id": record["id"],
                     "username": record["username"], "emotion": label}
                    for record, label in zip(records, labels)]
        distributions[key] = distribution(labels)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(labels_path, index=False)

    gt_dist = distributions["ground_truth"]
    kl_scores = {k: kl_divergence(gt_dist, v) for k, v in distributions.items() if k != "ground_truth"}

    summary_rows = [
        {"dataset": pretty(k), **{e: distributions[k].get(e, 0) for e in EMOTIONS},
         "kl_from_gt": kl_scores.get(k, 0.0)}
        for k in distributions
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "emotion_summary.csv", index=False)

    emotions = EMOTIONS if show_neutral else [e for e in EMOTIONS if e != "neutral"]
    sim_rows = []
    for k, dist in distributions.items():
        if k == "ground_truth":
            sim_rows.append({"dataset": k, "similarity": 1.0, "offset_pct": 0.0})
        else:
            s = jsd_similarity(gt_dist, dist, emotions)
            sim_rows.append({"dataset": k, "similarity": s, "offset_pct": -(abs(s - 1.0)) * 100})
    pd.DataFrame(sim_rows).to_csv(output_dir / "similarity.csv", index=False)

    plot_radar(distributions, output_dir / "emotion_radar.pdf", show_neutral=show_neutral)
    print(f"Emotion analysis written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_test",    required=True)
    parser.add_argument("--simulations_dir", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--neutral", action="store_true", default=False)
    args = parser.parse_args()
    evaluate(args.dataset_test, args.simulations_dir, args.output_dir, args.neutral)
