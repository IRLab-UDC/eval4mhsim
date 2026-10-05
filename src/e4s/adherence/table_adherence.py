import re
import argparse
import pandas as pd
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


def pretty_name(key: str) -> str:
    if key in ("Ground Truth", "ground_truth"):
        return "Reddit"
    key = _PARAMS_RE.sub('', key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            return f"\\textsc{{{model_name} {_SCENARIO_NAMES.get(scenario_tag, scenario_tag)}}}"
    return key


def build_table(similarity_csv, reference, output_file, configs=None):
    df = pd.read_csv(similarity_csv)

    ref_row = df[df['dataset'] == reference]
    if ref_row.empty:
        raise ValueError(f"Reference '{reference}' not found in {similarity_csv}")

    sim_rows = df[df['dataset'] != reference]
    if configs:
        sim_rows = sim_rows[sim_rows['dataset'].apply(lambda k: any(c in k for c in configs))]
    sim_rows = sim_rows.sort_values('similarity', ascending=False)

    def fmt(v, italic=False):
        s = f"{v:.3f}"
        return f"\\textit{{{s}}}" if italic else s

    def fmt_offset(v, italic=False):
        s = f"{v:+.2f}\\%"
        return f"\\textit{{{s}}}" if italic else s

    def name_cell(key, italic=False):
        n = pretty_name(key)
        return f"\\textit{{{n}}}" if italic else n

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Weighted similarity between ground truth and simulated datasets based on MRR degradation curve alignment. Perfect similarity equals 1.0.}",
        "  \\label{tab:adherence_similarity}",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Sim.} & \\textbf{Offset} \\\\",
        "    \\midrule",
    ]

    row = ref_row.iloc[0]
    lines.append(
        f"    {name_cell(row['dataset'], italic=True):<35} & "
        f"{fmt(row['similarity'], italic=True)} & "
        f"{fmt_offset(row['offset_pct'], italic=True)} \\\\"
    )
    lines.append("    \\midrule")

    for _, row in sim_rows.iterrows():
        lines.append(
            f"    {name_cell(row['dataset']):<35} & "
            f"{fmt(row['similarity'])} & "
            f"{fmt_offset(-abs(row['offset_pct']))} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    output = "\n".join(lines) + "\n"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(output)
    print(f"Table written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", required=True, help="Path to similarity.csv")
    parser.add_argument("--reference", required=True, help="Reference dataset name (e.g. 'Ground Truth')")
    parser.add_argument("--output", required=True, help="Output .tex file path")
    parser.add_argument("--configs", nargs="*", default=None, help="scenario tags to include (substring match)")
    args = parser.parse_args()

    build_table(args.similarity, args.reference, args.output, args.configs)
