import re
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DNLIEvaluator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("zayn1111/deberta-v3-dnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("zayn1111/deberta-v3-dnli")
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, pairs: List[Tuple[str, str]], batch_size: int = 512) -> List[Tuple[str, float]]:
        if not pairs:
            return []

        labels = ["entailment", "neutral", "contradiction"]
        results = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises, hypotheses = zip(*batch)

            inputs = self.tokenizer(
                list(premises), list(hypotheses),
                return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                probs = torch.softmax(self.model(**inputs).logits, dim=-1)

            for p in probs:
                idx = torch.argmax(p).item()
                results.append((labels[idx], p[idx].item()))

        return results

    def evaluate_conversation(self, turns: List[Tuple[str, str]], personas: Dict[str, List[str]]) -> Dict:
        coherence_pairs = []
        persona_pairs = []
        self_pairs = []
        self_pair_turn_map = []

        speaker_history = {}

        for i, (speaker, text) in enumerate(turns):
            if i > 0:
                coherence_pairs.append((turns[i-1][1], text))

            if speaker in personas:
                for attr in personas[speaker]:
                    persona_pairs.append((attr, text))

            if speaker in speaker_history:
                turn_start = len(self_pairs)
                for prev_text in speaker_history[speaker][-5:]:
                    self_pairs.append((prev_text, text))
                self_pair_turn_map.append((turn_start, len(self_pairs)))
            else:
                self_pair_turn_map.append(None)

            speaker_history.setdefault(speaker, []).append(text)

        coherence_results = self.predict_batch(coherence_pairs)
        persona_results = self.predict_batch(persona_pairs)
        self_results = self.predict_batch(self_pairs)

        label_counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
        coherence_scores = []
        for label, _ in coherence_results:
            label_counts[label] += 1
            coherence_scores.append(1.0 if label == "entailment" else (0.5 if label == "neutral" else 0.0))

        persona_contradictions = sum(1 for label, conf in persona_results if label == "contradiction" and conf > 0.7)
        total_persona_checks = len(persona_pairs)

        self_contradictions = 0
        for mapping in self_pair_turn_map:
            if mapping:
                start, end = mapping
                if any(label == "contradiction" and conf > 0.7 for label, conf in self_results[start:end]):
                    self_contradictions += 1

        cs = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
        pcr = persona_contradictions / total_persona_checks if total_persona_checks > 0 else 0.0
        scr = self_contradictions / len(turns) if turns else 0.0
        total_labels = sum(label_counts.values())
        er = label_counts["entailment"] / total_labels if total_labels > 0 else 0.0
        nr = label_counts["neutral"] / total_labels if total_labels > 0 else 0.0
        cr = label_counts["contradiction"] / total_labels if total_labels > 0 else 0.0

        naturalness = 0.6 * cs + 0.2 * (1 - pcr) + 0.2 * (1 - scr)

        return {
            "naturalness_score": naturalness,
            "coherence_score": cs,
            "persona_contradiction_rate": pcr,
            "self_contradiction_rate": scr,
            "entailment_rate": er,
            "neutral_rate": nr,
            "contradiction_rate": cr
        }


def split_persona(system_prompt: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', system_prompt.strip())
    return [s.strip() for s in sentences if s.strip()]


def load_ground_truth(file_path: str) -> List[Dict]:
    target_types = {"user_reply_to_op", "user_reply_to_op_with_response"}
    conversations = []
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") not in target_types:
                continue
            op_text = r["conversation"][0]["text"]
            reply = r["conversation"][1]["text"]
            conversations.append({
                "id": r["id"],
                "turns": [("OP", op_text), ("USER", reply)],
                "personas": {"USER": split_persona(r["system_prompt"])},
            })
    return conversations


def load_simulations(file_path: str, personas_file: str = "data/personas_p.jsonl") -> List[Dict]:
    persona_by_username = {}
    with open(personas_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            persona_by_username[r["username"]] = r.get("system_prompt", "")

    conversations = []
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            persona = split_persona(persona_by_username.get(r["username"], ""))
            conversations.append({
                "id": r["id"],
                "turns": [("OP", r["op_text"]), ("USER", r["simulation"])],
                "personas": {"USER": persona},
            })
    return conversations


def load_file(file_path: str, personas_file: str = "data/personas_p.jsonl") -> List[Dict]:
    first = json.loads(open(file_path).readline())
    if "simulation" in first:
        return load_simulations(file_path, personas_file)
    return load_ground_truth(file_path)


def evaluate_dataset(input_file: str, output_file: str, limit: int = None, personas_file: str = "data/personas_p.jsonl"):
    evaluator = DNLIEvaluator()
    conversations = load_file(input_file, personas_file)

    if limit:
        conversations = conversations[:limit]

    print(f"Evaluating {len(conversations)} conversations...")

    results = []
    for i, conv in enumerate(conversations, 1):
        metrics = evaluator.evaluate_conversation(conv["turns"], conv["personas"])
        results.append({"id": conv["id"], **metrics})

        if i % 100 == 0:
            avg = sum(r["naturalness_score"] for r in results) / len(results)
            print(f"{i}/{len(conversations)} | avg naturalness: {avg:.3f}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metrics_keys = ["naturalness_score", "coherence_score", "persona_contradiction_rate",
                    "self_contradiction_rate", "entailment_rate", "neutral_rate", "contradiction_rate"]

    summary = {
        "total": len(results),
        "metric_definitions": {
            "coherence_score": "Turn-to-turn coherence: 1.0 for entailment, 0.5 for neutral, 0.0 for contradiction",
            "persona_contradiction_rate": "Proportion of persona-attribute/reply pairs with contradiction (conf > 0.7)",
            "self_contradiction_rate": "Proportion of turns contradicting speaker's previous turns (window: 5, conf > 0.7)",
            "entailment_rate": "Proportion of turn-to-turn transitions with entailment",
            "neutral_rate": "Proportion of turn-to-turn transitions with neutral",
            "contradiction_rate": "Proportion of turn-to-turn transitions with contradiction",
            "naturalness_score": "0.6*coherence + 0.2*(1-PCR) + 0.2*(1-SCR)"
        },
    }
    for key in metrics_keys:
        vals = [r[key] for r in results]
        summary[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}

    summary_path = Path(output_file.replace(".json", "_summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults → {output_path}")
    print(f"Summary → {summary_path}")
    print(f"naturalness={summary['naturalness_score']['mean']:.3f}  coherence={summary['coherence_score']['mean']:.3f}  PCR={summary['persona_contradiction_rate']['mean']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?")
    parser.add_argument("-o", "--output")
    parser.add_argument("-l", "--limit", type=int, default=None)
    parser.add_argument("--personas", default="data/personas_p.jsonl")
    parser.add_argument("--batch", action="store_true", help="Evaluate all simulations + ground truth, output to data/results/naturalness/")
    args = parser.parse_args()

    if args.batch:
        import glob
        results_dir = Path("data/results/naturalness")
        results_dir.mkdir(parents=True, exist_ok=True)

        _PERSONA_FILE_MAP = [
            ("persona_power",     "data/personas_power.jsonl"),
            ("persona_powe",      "data/personas_powe.jsonl"),
            ("persona_pow",       "data/personas_pow.jsonl"),
            ("persona_optimized", "data/personas_po.jsonl"),
            ("persona_default",   "data/personas_p.jsonl"),
        ]

        def infer_personas(sim_path: str) -> str:
            for tag, personas_file in _PERSONA_FILE_MAP:
                if tag in sim_path:
                    return personas_file
            return args.personas

        files = [("ground_truth", "data/dataset_test.jsonl")] + [
            (re.sub(r"_t[\d.]+_maxtok\d+$", "", Path(p).stem), p)
            for p in sorted(glob.glob("data/simulations/*.jsonl"))
        ]
        for name, path in files:
            out = results_dir / f"{name}.json"
            if out.exists():
                print(f"\n=== {name} — skipping (already exists) ===")
                continue
            print(f"\n=== {name} ===")
            personas = infer_personas(path)
            evaluate_dataset(path, str(out), args.limit, personas)
    else:
        if not args.input_file or not args.output:
            parser.error("input_file and -o are required without --batch")
        evaluate_dataset(args.input_file, args.output, args.limit, args.personas)
