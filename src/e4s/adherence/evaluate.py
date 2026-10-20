import argparse
import pytrec_eval


def evaluate(qrels_file, run_file, output_file):
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split()
            qrels.setdefault(qid, {})[doc_id] = int(rel)

    run = {}
    with open(run_file) as f:
        for line in f:
            qid, _, doc_id, _, score, _ = line.strip().split()
            run.setdefault(qid, {})[doc_id] = float(score)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'recall_10', 'ndcg_cut_10'})
    results = evaluator.evaluate(run)

    agg = {'recip_rank': 0.0, 'recall_10': 0.0, 'ndcg_cut_10': 0.0}
    for metrics in results.values():
        for metric in agg:
            agg[metric] += metrics.get(metric, 0.0)
    n = len(results)

    with open(output_file, 'w') as f:
        for metric, v in agg.items():
            f.write(f"{metric}\tall\t{v / n:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    evaluate(args.qrels, args.run, args.output)
