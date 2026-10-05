import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytrec_eval

E4S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E4S / "consistency"))
sys.path.insert(0, str(E4S / "emotionality"))
sys.path.insert(0, str(E4S / "adherence"))
from evaluator import evaluate_all
from prepare_datasets import SEED, load_replies_by_user, load_sim_by_user, make_test_pairs, write_pairs
from metrics import EMOTIONS, distribution, jsd_similarity
from similarity import compute_similarity

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}
POOL_SIZES = [1, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
TEST = Path("data/dataset_test.jsonl")
TRAIN = Path("data/dataset_train.jsonl")
SIMULATIONS = Path("data/simulations")
RESULTS = Path("data/results")
VERIFIER = Path("data/consistency/model")
VERIFIER_ITERATIONS = 100

_spec = importlib.util.spec_from_file_location("cngdist", E4S / "consistency" / "pan23-verif-baseline-cngdist.py")
cngdist = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cngdist)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def user_reply(record):
    return next((t["text"] for t in reversed(record["conversation"]) if t["is_user"]), None)


def naturalness(test_ids, out):
    datasets = {}
    for f in sorted((RESULTS / "naturalness").glob("*.json")):
        if f.name.endswith("_summary.json") or f.name == "overall_summary.json":
            continue
        rows = [r for r in json.load(open(f)) if r["id"] in test_ids]
        keys = [k for k in rows[0] if k != "id"]
        datasets[f.stem] = {"total_conversations": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in keys}}
    datasets = dict(sorted(datasets.items(), key=lambda x: x[1]["naturalness_score"], reverse=True))
    (out / "naturalness").mkdir(parents=True, exist_ok=True)
    json.dump({"datasets": datasets}, open(out / "naturalness" / "overall_summary.json", "w"), indent=2)


def emotionality(test_ids, out):
    labels = defaultdict(list)
    reader = csv.DictReader(open(RESULTS / "emotionality" / "emotion_labels.csv"))
    kept = [r for r in reader if r["id"] in test_ids]
    for r in kept:
        labels[r["dataset"]].append(r["emotion"])
    gt = distribution(labels["ground_truth"])
    (out / "emotionality").mkdir(parents=True, exist_ok=True)
    with open(out / "emotionality" / "emotion_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(kept)
    with open(out / "emotionality" / "similarity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "similarity", "offset_pct"])
        for key, lab in labels.items():
            s = 1.0 if key == "ground_truth" else jsd_similarity(gt, distribution(lab), EMOTIONS)
            w.writerow([key, s, -(abs(s - 1.0)) * 100])


def consistency(users, out):
    sources = {"ground_truth": load_replies_by_user(str(TEST))}
    for f in sorted(SIMULATIONS.glob("*.jsonl")):
        sources[f.stem] = load_sim_by_user(str(f))
    train = load_replies_by_user(str(TRAIN))

    counts = {}
    for name, text2 in sources.items():
        res = out / "consistency" / name
        res.mkdir(parents=True, exist_ok=True)
        if users is None:
            pairs_dir, answers_dir = Path("data/consistency") / name / "test", RESULTS / "consistency" / name
        else:
            keep = lambda d: {u: v for u, v in d.items() if u in users}
            pairs_dir, answers_dir = res / "test", res
            write_pairs(make_test_pairs(keep(train), keep(text2), random.Random(SEED)), pairs_dir)
            cngdist.test(str(pairs_dir / "pairs.jsonl"), res, VERIFIER, VERIFIER_ITERATIONS)
        truth = {t["id"]: int(t["same"]) for t in read_jsonl(pairs_dir / "truth.jsonl")}
        answers = {a["id"]: a["value"] for a in read_jsonl(answers_dir / "answers.jsonl")}
        ids = sorted(truth)
        gt = np.array([truth[i] for i in ids], dtype=np.float64)
        pred = np.array([answers.get(i, 0.5) for i in ids], dtype=np.float64)
        json.dump(evaluate_all(gt, pred), open(res / "out.json", "w"), indent=4, sort_keys=True)
        counts[name] = (int(gt.sum()), int(len(gt) - gt.sum()))
    return counts


def adherence(users, test, out):
    test_users = {r["username"] for r in test}
    sources = {"ground_truth": ("Ground Truth", [(r["username"], user_reply(r)) for r in test if user_reply(r) is not None])}
    for f in sorted(SIMULATIONS.glob("*.jsonl")):
        recs = [(r["username"], r.get("simulation")) for r in read_jsonl(f) if r.get("type") in TARGET_TYPES and r["username"] in test_users]
        sources[f.stem.replace(".", "_")] = (f.stem.replace(".", "_"), [(u, t) for u, t in recs if t])

    rows = []
    for dirname, (label, recs) in sources.items():
        qid_user = {str(i): u for i, u in enumerate(sorted({u for u, _ in recs}))}
        doc_user = {str(i): u for i, (u, _) in enumerate(recs)}
        keep = lambda u: users is None or u in users
        qrels = defaultdict(dict)
        for line in open(Path("data/adherence") / dirname / "qrels.txt"):
            q, _, d, rel = line.split()
            if keep(qid_user[q]):
                qrels[q][d] = int(rel)
        evaluator = pytrec_eval.RelevanceEvaluator(dict(qrels), {"recip_rank", "recall_10", "ndcg_cut_10"})
        for pool in POOL_SIZES:
            run = defaultdict(dict)
            for line in open(Path("data/adherence") / dirname / "runs" / f"run_pool_{pool}.txt"):
                q, _, d, _, score, _ = line.split()
                if keep(qid_user[q]) and keep(doc_user[d]):
                    run[q][d] = float(score)
            res = evaluator.evaluate(dict(run))
            agg = {m: float(f"{sum(r.get(m, 0.0) for r in res.values()) / len(res):.4f}") for m in ["recip_rank", "recall_10", "ndcg_cut_10"]}
            rows.append({"pool_size": pool, "dataset": label, "MRR": agg["recip_rank"], "Recall@10": agg["recall_10"], "NDCG@10": agg["ndcg_cut_10"]})

    (out / "adherence").mkdir(parents=True, exist_ok=True)
    with open(out / "adherence" / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    compute_similarity(out / "adherence" / "comparison.csv", "Ground Truth", out / "adherence" / "similarity.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", default=None, help="users.jsonl whose ids define the users to keep; all users if omitted")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    test = [r for r in read_jsonl(TEST) if r["type"] in TARGET_TYPES]
    users = None
    if args.users:
        ids = {r["id"] for r in read_jsonl(args.users)}
        users = {r["username"] for r in read_jsonl("data/personas_p.jsonl") if r["id"] in ids}
    test_ids = {r["id"] for r in test if users is None or r["username"] in users}
    out = Path(args.output)

    naturalness(test_ids, out)
    emotionality(test_ids, out)
    counts = consistency(users, out)
    adherence(users, test, out)
    print(f"{len(test_ids)} test replies; consistency pairs (same, diff): {sorted(set(counts.values()))}")


if __name__ == "__main__":
    main()
