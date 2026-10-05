import json
import re
import glob
from pathlib import Path

_SUFFIX_RE = re.compile(r'conversations_temp[\d.]+_top_p[\d.]+_')

_MODEL_NAMES = {
    "google_gemma-3-4b-it":          "Gemma 3 4B",
    "google_gemma-3-12b-it":         "Gemma 3 12B",
    "google_gemma-3-27b-it":         "Gemma 3 27B",
    "Qwen_Qwen3-4B-Instruct-2507":   "Qwen3 4B",
    "Qwen_Qwen3-14B":                "Qwen3 14B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "Qwen3 30B",
}


def normalize_key(stem: str) -> str:
    return _SUFFIX_RE.sub('', stem)


def pretty(key: str) -> str:
    if key == "ground_truth":
        return "PersonaChat"
    for slug, name in _MODEL_NAMES.items():
        if key == slug:
            return name
    return key


def _user2_turns(conversation: str) -> list[str]:
    turns = []
    for line in conversation.split("\n"):
        if line.startswith("User 2:"):
            text = line[len("User 2:"):].strip()
            if text:
                turns.append(text)
    return turns


def load_ground_truth(gt_path: str) -> list[dict]:
    records = []
    with open(gt_path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            turns = _user2_turns(r["conversation"])
            if turns:
                records.append({"id": i, "text": " ".join(turns)})
    return records


def load_simulation(sim_path: str) -> list[dict]:
    records = []
    with open(sim_path) as f:
        for line in f:
            r = json.loads(line)
            turns = _user2_turns(r["conversation"])
            if turns:
                records.append({"id": r["id"], "text": " ".join(turns)})
    return records


def load_all_datasets(gt_path: str, simulations_dir: str) -> dict[str, list[dict]]:
    datasets = {"ground_truth": load_ground_truth(gt_path)}
    for sim_file in sorted(glob.glob(f"{simulations_dir}/*.jsonl")):
        stem = Path(sim_file).stem
        key = normalize_key(stem)
        datasets[key] = load_simulation(sim_file)
    return datasets
