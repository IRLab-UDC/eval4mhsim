import argparse
import random
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig

random.seed(42)


def load_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            qid, _, doc_id, _ = parts
            if int(qid) not in qrels:
                qrels[int(qid)] = []
            qrels[int(qid)].append(int(doc_id))
    return qrels


def save_trec_run(results_dict, output_file, run_name="colbert"):
    with open(output_file, "w") as f:
        for qid in sorted(results_dict.keys()):
            pids, ranks, scores = results_dict[qid]
            for pid, rank, score in zip(pids, ranks, scores):
                f.write(f"{qid} Q0 {pid} {rank} {score} {run_name}\n")


def search(index_name, queries_file, qrels_file, output_file, experiment, collection_file=None, pool_size=None, k=1000):
    queries = Queries(queries_file)
    qrels = load_qrels(qrels_file) if pool_size is not None else {}

    with Run().context(RunConfig(experiment=experiment)):
        searcher = Searcher(index=index_name, config=ColBERTConfig(query_maxlen=512))

    if pool_size is not None:
        with open(collection_file) as f:
            num_docs = sum(1 for _ in f)
        all_doc_ids = list(range(num_docs))
    else:
        all_doc_ids = None

    all_results = {}
    for qid in sorted(queries.data.keys()):
        if pool_size is not None and qrels:
            correct_ids = set(qrels[qid])
            distractors_pool = [d for d in all_doc_ids if d not in correct_ids]
            num_distractors = min(max(pool_size - len(correct_ids), 0), len(distractors_pool))
            selected = list(correct_ids) + random.sample(distractors_pool, num_distractors)
            results = searcher.search(queries.data[qid], k=k, pids=selected)
        else:
            results = searcher.search(queries.data[qid], k=k)

        pids, ranks, scores = results
        all_results[qid] = (list(pids), list(ranks), list(scores))

    save_trec_run(all_results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pool_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=1000)
    args = parser.parse_args()

    search(
        index_name=args.index_name,
        queries_file=args.queries,
        qrels_file=args.qrels,
        output_file=args.output,
        experiment=args.experiment,
        collection_file=args.collection,
        pool_size=args.pool_size,
        k=args.k,
    )
