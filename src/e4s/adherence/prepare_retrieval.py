import argparse
import json
from pathlib import Path

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") in TARGET_TYPES:
                records.append(r)
    return records


def prepare_ground_truth(input_path, output_dir):
    records = load_records(input_path)

    users = {}
    for r in records:
        if r["username"] not in users:
            users[r["username"]] = r["system_prompt"]
    user_to_qid = {u: i for i, u in enumerate(sorted(users))}

    documents, qrels = [], []
    doc_id = 0
    for r in records:
        reply = next((t["text"] for t in reversed(r["conversation"]) if t["is_user"]), None)
        if reply is None:
            continue
        documents.append((doc_id, reply))
        qrels.append((user_to_qid[r["username"]], doc_id))
        doc_id += 1

    _write_outputs(output_dir, user_to_qid, users, documents, qrels)


def prepare_simulation_with_personas(sim_path, personas_path, output_dir, field="simulation"):
    sim_records = load_records(sim_path)

    personas = {}
    with open(personas_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") in TARGET_TYPES and r["username"] not in personas:
                personas[r["username"]] = r["system_prompt"]

    sim_records = [r for r in sim_records if r["username"] in personas]
    users = {r["username"] for r in sim_records}
    user_to_qid = {u: i for i, u in enumerate(sorted(users))}

    documents, qrels = [], []
    doc_id = 0
    for r in sim_records:
        text = r.get(field)
        if not text:
            continue
        documents.append((doc_id, text))
        qrels.append((user_to_qid[r["username"]], doc_id))
        doc_id += 1

    _write_outputs(output_dir, user_to_qid, personas, documents, qrels)


def _write_outputs(output_dir, user_to_qid, personas, documents, qrels):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "documents.tsv", "w") as f:
        for doc_id, text in documents:
            clean = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{doc_id}\t{clean}\n")

    with open(output_dir / "queries.tsv", "w") as f:
        for username, qid in sorted(user_to_qid.items(), key=lambda x: x[1]):
            clean = personas[username].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{qid}\t{clean}\n")

    with open(output_dir / "qrels.txt", "w") as f:
        for qid, doc_id in qrels:
            f.write(f"{qid} 0 {doc_id} 1\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--personas", default=None, help="dataset_test.jsonl path (required for simulation files)")
    parser.add_argument("--field", default="simulation", help="reply field name in simulation files")
    args = parser.parse_args()

    if args.personas:
        prepare_simulation_with_personas(args.input, args.personas, args.output_dir, args.field)
    else:
        prepare_ground_truth(args.input, args.output_dir)
