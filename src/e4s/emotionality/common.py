import json
import re
import glob
from pathlib import Path

TARGET_TYPES = {"user_reply_to_op", "user_reply_to_op_with_response"}
_SUFFIX_RE = re.compile(r'_t[\d._]+_maxtok\d+')

_MODEL_NAMES = {
    "google_gemma-3-1b-it":  "Gemma 3 1B",
    "google_gemma-3-4b-it":  "Gemma 3 4B",
    "google_gemma-3-12b-it": "Gemma 3 12B",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "qwen_qwen3-1.7b":       "Qwen3 1.7B",
    "qwen_qwen3-4b":         "Qwen3 4B",
    "qwen_qwen3-14b":        "Qwen3 14B",
    "qwen_qwen3-30b":        "Qwen3 30B",
}

_SCENARIO_NAMES = {
    "persona_none_fs0":               "ZS",
    "persona_none_fs10same_user":     "ICL",
    "persona_none_fs10random_user":   "ICLR",
    "persona_default_fs0":            "ZS-P",
    "persona_default_fs10same_user":  "ICL-P",
    "persona_default_fs10random_user": "ICLR-P",
    "persona_optimized_fs0":              "ZS-PO",
    "persona_optimized_fs10same_user":    "ICL-PO",
    "persona_optimized_fs10random_user":  "ICLR-PO",
    "persona_pow_fs0":                    "ZS-POW",
    "persona_pow_fs10same_user":          "ICL-POW",
    "persona_pow_fs10random_user":        "ICLR-POW",
    "persona_powe_fs0":                   "ZS-POWE",
    "persona_powe_fs10same_user":         "ICL-POWE",
    "persona_powe_fs10random_user":       "ICLR-POWE",
    "persona_power_fs0":                  "ZS-POWER",
    "persona_power_fs10same_user":        "ICL-POWER",
    "persona_power_fs10random_user":      "ICLR-POWER",
}


def normalize_key(key: str) -> str:
    return _SUFFIX_RE.sub('', key)


def pretty(key: str) -> str:
    if key == "ground_truth":
        return "Reddit"
    key = normalize_key(key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            scenario_label = _SCENARIO_NAMES.get(scenario_tag, scenario_tag)
            return f"\\textsc{{{model_name} {scenario_label}}}"
    return key


def load_ground_truth(dataset_test_path: str) -> list[dict]:
    """Returns list of {username, text, id} for all TARGET_TYPES records."""
    records = []
    with open(dataset_test_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") not in TARGET_TYPES:
                continue
            reply = next((t["text"] for t in reversed(r["conversation"]) if t["is_user"]), None)
            if reply:
                records.append({"username": r["username"], "text": reply, "id": r["id"]})
    return records


def load_simulation(sim_path: str) -> list[dict]:
    """Returns list of {username, text, id} for simulation field."""
    records = []
    with open(sim_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") not in TARGET_TYPES:
                continue
            text = r.get("simulation")
            if text:
                records.append({"username": r["username"], "text": text, "id": r["id"]})
    return records


def load_all_datasets(dataset_test_path: str, simulations_dir: str) -> dict[str, list[dict]]:
    """Returns {dataset_key: [records]} for GT and all sim files."""
    datasets = {"ground_truth": load_ground_truth(dataset_test_path)}
    for sim_file in sorted(glob.glob(f"{simulations_dir}/*.jsonl")):
        stem = Path(sim_file).stem
        key = normalize_key(stem)
        datasets[key] = load_simulation(sim_file)
    return datasets
