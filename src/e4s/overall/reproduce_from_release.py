import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

E4S = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(E4S / "emotionality"), str(E4S / "consistency"), str(E4S / "adherence")]
from metrics import EMOTIONS, distribution, jsd_similarity
from evaluator import evaluate_all
from similarity import weighted_mrr_similarity

REF = (None, "reference")


def read(data, name):
    return [json.loads(l) for l in open(data / f"{name}.jsonl") if l.strip()]


def key(row):
    return row["model"], row["config"]


def emotionality_naturalness(data):
    emo, nat = defaultdict(list), defaultdict(list)
    for r in read(data, "instance_scores"):
        emo[key(r)].append(r["emotion"])
        nat[key(r)].append(r["naturalness"]["score"])
    ref_dist, ref_nat = distribution(emo[REF]), np.mean(nat[REF])
    emotionality = {k: jsd_similarity(ref_dist, distribution(v), EMOTIONS) for k, v in emo.items()}
    naturalness = {k: 1 - abs(np.mean(v) - ref_nat) / ref_nat for k, v in nat.items()}
    return emotionality, naturalness


def adherence(data):
    rr = defaultdict(lambda: defaultdict(list))
    for r in read(data, "adherence_ranks"):
        rr[key(r)][r["pool"]].append(r["rr"])
    pools = sorted(rr[REF])
    spans = [pools[1] - pools[0]] + [(pools[i + 1] - pools[i - 1]) / 2 for i in range(1, len(pools) - 1)] + [pools[-1] - pools[-2]]
    weights = [s * 0.9 ** i for i, s in enumerate(spans)]
    mrr = {k: [round(float(np.mean(v[p])), 4) for p in pools] for k, v in rr.items()}
    return {k: weighted_mrr_similarity(mrr[REF], m, weights) for k, m in mrr.items()}


def consistency(data):
    pairs = defaultdict(list)
    for r in read(data, "consistency_pairs"):
        pairs[key(r)].append((float(r["same"]), r["score"]))
    overall = {k: evaluate_all(np.array([a for a, _ in v]), np.array([b for _, b in v]))["overall"] for k, v in pairs.items()}
    return {k: 1 - abs(v - overall[REF]) / overall[REF] for k, v in overall.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="local copy of irlab-udc/erisk-depression-personas")
    data = Path(parser.parse_args().data)

    emo, nat = emotionality_naturalness(data)
    adh, con = adherence(data), consistency(data)
    rows = sorted(
        ((f"{m} {c}", adh[m, c], con[m, c], nat[m, c], emo[m, c]) for m, c in emo if (m, c) != REF),
        key=lambda r: -sum(r[1:]),
    )
    print(f"{'Configuration':16} {'Adh.':>6} {'Cons.':>6} {'Nat.':>6} {'Emo.':>6} {'Overall':>8}")
    for name, *dims in rows:
        print(f"{name:16} " + " ".join(f"{d:6.3f}" for d in dims) + f" {sum(dims) / 4:8.3f}")


if __name__ == "__main__":
    main()
