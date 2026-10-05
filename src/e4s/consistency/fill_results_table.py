import re
import json
import argparse
from pathlib import Path

_PARAMS_RE = re.compile(r'_t[\d._]+_maxtok\d+')

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


def pretty_name(key: str) -> str:
    if key == "ground_truth":
        return "Reddit"
    key = _PARAMS_RE.sub('', key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            return f"\\textsc{{{model_name} {_SCENARIO_NAMES.get(scenario_tag, scenario_tag)}}}"
    return key

METRICS = ["F1", "auc", "brier", "c@1", "f_05_u", "overall"]


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
        f"{f(m['F1'])} & "
        f"{f(m['auc'])} & "
        f"{f(m['brier'])} & "
        f"{f(m['c@1'])} & "
        f"{f(m['f_05_u'])} & "
        f"{f(m['overall'])} & "
        f"{f(sim)} & "
        f"{offset_str} \\\\"
    )


def load_results(results_dir: Path) -> dict:
    datasets = {}
    for out_file in results_dir.glob("*/out.json"):
        name = out_file.parent.name
        with open(out_file) as f:
            datasets[name] = json.load(f)
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/results/consistency")
    parser.add_argument("--output", default="data/results/consistency/table_consistency.tex")
    parser.add_argument("--configs", nargs="*", default=None, help="scenario tags to include (substring match)")
    args = parser.parse_args()

    results_dir = Path(args.input)
    datasets = load_results(results_dir)

    if "ground_truth" not in datasets:
        raise ValueError("ground_truth entry not found in results")

    gt = datasets["ground_truth"]
    gt_overall = gt["overall"]

    sim_rows = {k: v for k, v in datasets.items() if k != "ground_truth"}
    if args.configs:
        sim_rows = {k: v for k, v in sim_rows.items() if any(c in k for c in args.configs)}
    sorted_rows = sorted(sim_rows.items(), key=lambda x: abs(x[1]["overall"] - gt_overall))

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Consistency evaluation using authorship verification. Metrics: F1, AUC, Brier score, c@1, and $F_{0.5}^u$. The overall score is their mean. Sim.\\ $= 1 - |\\text{overall} - \\text{overall}_{gt}| / \\text{overall}_{gt}$, where 1.0 is perfect alignment with ground truth. Offset is the signed relative difference from ground truth.}",
        "  \\label{tab:consistency_results}",
        "  \\begin{tabular}{lccccccccr}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{F1} & \\textbf{AUC} & \\textbf{Brier} & \\textbf{c@1} & $\\mathbf{F_{0.5}^u}$ & \\textbf{\\emph{Consistency}} & \\textbf{Sim.} & \\textbf{Offset} \\\\",
        "    \\midrule",
    ]

    gt_name = pretty_name("ground_truth")
    lines.append(format_row(gt_name, gt, 1.0, 0.0, italic=True))
    lines.append("    \\midrule")

    for key, metrics in sorted_rows:
        name = pretty_name(key)
        sim = 1 - abs(metrics["overall"] - gt_overall) / gt_overall
        offset = -(abs(metrics["overall"] - gt_overall) / gt_overall) * 100
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
