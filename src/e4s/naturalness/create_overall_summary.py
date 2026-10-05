import json
import sys
from pathlib import Path

def create_overall_summary(results_dir):
    results_dir = Path(results_dir)

    datasets = {}

    for summary_file in results_dir.glob("*_summary.json"):
        dataset_name = summary_file.stem.replace("_summary", "")

        with open(summary_file) as f:
            data = json.load(f)

        if "total" not in data or "naturalness_score" not in data:
            print(f"Skipping {dataset_name} - incompatible format")
            continue

        datasets[dataset_name] = {
            "total_conversations": data["total"],
            "naturalness_score": data["naturalness_score"]["mean"],
            "coherence_score": data["coherence_score"]["mean"],
            "persona_contradiction_rate": data["persona_contradiction_rate"]["mean"],
            "self_contradiction_rate": data["self_contradiction_rate"]["mean"],
            "entailment_rate": data["entailment_rate"]["mean"],
            "neutral_rate": data["neutral_rate"]["mean"],
            "contradiction_rate": data["contradiction_rate"]["mean"]
        }

    sorted_datasets = sorted(datasets.items(), key=lambda x: x[1]["naturalness_score"], reverse=True)

    overall = {
        "metric_definitions": {
            "coherence_score": "Turn-to-turn coherence: 1.0 for entailment, 0.5 for neutral, 0.0 for contradiction",
            "persona_contradiction_rate": "Proportion of turns contradicting speaker persona (threshold: 0.7 confidence)",
            "self_contradiction_rate": "Proportion of turns contradicting speaker's previous turns (window: 5 turns, threshold: 0.7)",
            "entailment_rate": "Proportion of turn-to-turn transitions with entailment",
            "neutral_rate": "Proportion of turn-to-turn transitions with neutral",
            "contradiction_rate": "Proportion of turn-to-turn transitions with contradiction",
            "naturalness_score": "0.6*coherence + 0.2*(1-PCR) + 0.2*(1-SCR)"
        },
        "datasets": dict(sorted_datasets),
        "ranking_by_naturalness": [name for name, _ in sorted_datasets]
    }

    output_file = results_dir / "overall_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print(f"\nOverall Summary:")
    print("=" * 100)
    print(f"{'Dataset':<20} {'Naturalness':<12} {'Coherence':<12} {'PCR':<8} {'SCR':<8} {'ER':<8} {'NR':<8} {'CR':<8}")
    print("-" * 100)
    for name, metrics in sorted_datasets:
        print(f"{name:<20} {metrics['naturalness_score']:<12.3f} {metrics['coherence_score']:<12.3f} "
              f"{metrics['persona_contradiction_rate']:<8.3f} {metrics['self_contradiction_rate']:<8.3f} "
              f"{metrics['entailment_rate']:<8.3f} {metrics['neutral_rate']:<8.3f} {metrics['contradiction_rate']:<8.3f}")
    print("=" * 100)
    print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    create_overall_summary(results_dir)
