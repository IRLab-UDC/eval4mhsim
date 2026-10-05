import json
import re
import argparse
from pathlib import Path

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


_PARAMS_RE = re.compile(r'_t[\d._]+_maxtok\d+')


def stem_to_key(stem: str) -> str:
    return _PARAMS_RE.sub('', stem)


def pretty_name(key: str) -> str:
    if key == "ground_truth":
        return "Reddit"
    key = _PARAMS_RE.sub('', key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            return f"\\textsc{{{model_name} {_SCENARIO_NAMES.get(scenario_tag, scenario_tag)}}}"
    return key


def format_row(name: str, m: dict, sim: float, offset: float, italic: bool) -> str:
    def f(v):
        s = f"{v:.3f}"
        return f"\\textit{{{s}}}" if italic else s

    offset_str = f"{offset:+.2f}\\%"
    if italic:
        offset_str = f"\\textit{{{offset_str}}}"

    name_cell = f"\\textit{{{name}}}" if italic else name
    return (
        f"    {name_cell:<30} & "
        f"{f(m['coherence_score'])} & "
        f"{f(m['persona_contradiction_rate'])} & "
        f"{f(m['entailment_rate'])} & "
        f"{f(m['neutral_rate'])} & "
        f"{f(m['contradiction_rate'])} & "
        f"{f(m['naturalness_score'])} & "
        f"{f(sim)} & "
        f"{offset_str} \\\\"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/naturalness/overall_summary.json")
    parser.add_argument("--output", default="data/results/naturalness/table_naturalness.tex")
    parser.add_argument("--configs", nargs="*", default=None, help="scenario tags to include (substring match)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    datasets = data["datasets"]

    gt_key = next((k for k in datasets if stem_to_key(k) == "ground_truth"), None)
    if gt_key is None:
        raise ValueError("ground_truth entry not found in overall_summary.json")

    gt_naturalness = datasets[gt_key]["naturalness_score"]

    sim_rows = {k: v for k, v in datasets.items() if k != gt_key}
    if args.configs:
        sim_rows = {k: v for k, v in sim_rows.items() if any(c in k for c in args.configs)}
    sorted_rows = sorted(sim_rows.items(), key=lambda x: x[1]["naturalness_score"])

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Naturalness evaluation using Dialogue NLI on 919 test records. Metrics: Coherence Score (CS), Persona Contradiction Rate (PCR), Entailment Rate (ER), Neutral Rate (NR), Contradiction Rate (CR). Self-Contradiction Rate is omitted as it is always 0 in single-turn settings. Naturalness $= 0.6 \\times \\text{CS} + 0.2 \\times (1 - \\text{PCR}) + 0.2 \\times (1 - \\text{SCR})$. Sim.\\ $= 1 - |\\text{nat} - \\text{nat}_{gt}| / \\text{nat}_{gt}$, where 1.0 is perfect alignment with ground truth. Offset is the signed relative difference from ground truth.}",
        "  \\label{tab:naturalness_results}",
        "  \\begin{tabular}{lcccccccr}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{CS} & \\textbf{PCR} & \\textbf{ER} & \\textbf{NR} & \\textbf{CR} & \\textbf{Naturalness} & \\textbf{Sim.} & \\textbf{Offset} \\\\",
        "    \\midrule",
    ]

    gt_name = pretty_name(stem_to_key(gt_key))
    lines.append(format_row(gt_name, datasets[gt_key], 1.0, 0.0, italic=True))
    lines.append("    \\midrule")

    for key, metrics in sorted_rows:
        name = pretty_name(stem_to_key(key))
        sim = 1 - abs(metrics["naturalness_score"] - gt_naturalness) / gt_naturalness
        offset = -(abs(metrics["naturalness_score"] - gt_naturalness) / gt_naturalness) * 100
        lines.append(format_row(name, metrics, sim, offset, italic=False))

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
    ]

    output = "\n".join(lines) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(output)
    print(f"Table written to {args.output}")


if __name__ == "__main__":
    main()
