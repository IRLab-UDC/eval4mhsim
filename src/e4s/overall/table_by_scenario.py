import re
import json
import argparse
import pandas as pd
from pathlib import Path

_PARAMS_RE = re.compile(r'_t[\d._]+_maxtok\d+')

_MODEL_SLUGS = [
    "google_gemma-3-4b-it",
    "google_gemma-3-12b-it",
    "google_gemma-3-27b-it",
]

_MODEL_NAMES = {
    "google_gemma-3-4b-it":  "4B",
    "google_gemma-3-12b-it": "12B",
    "google_gemma-3-27b-it": "27B",
}

_SCENARIO_ORDER = [
    "persona_none_fs0",
    "persona_none_fs10same_user",
    "persona_none_fs10random_user",
    "persona_default_fs0",
    "persona_default_fs10same_user",
    "persona_default_fs10random_user",
    "persona_optimized_fs0",
    "persona_optimized_fs10same_user",
    "persona_optimized_fs10random_user",
    "persona_pow_fs0",
    "persona_pow_fs10same_user",
    "persona_pow_fs10random_user",
    "persona_powe_fs0",
    "persona_powe_fs10same_user",
    "persona_powe_fs10random_user",
    "persona_power_fs0",
    "persona_power_fs10same_user",
    "persona_power_fs10random_user",
]

_SCENARIO_NAMES = {
    "persona_none_fs0":                "ZS",
    "persona_none_fs10same_user":      "ICL",
    "persona_none_fs10random_user":    "ICLR",
    "persona_default_fs0":             "ZS-P",
    "persona_default_fs10same_user":   "ICL-P",
    "persona_default_fs10random_user": "ICLR-P",
    "persona_optimized_fs0":           "ZS-PO",
    "persona_optimized_fs10same_user": "ICL-PO",
    "persona_optimized_fs10random_user": "ICLR-PO",
    "persona_pow_fs0":                 "ZS-POW",
    "persona_pow_fs10same_user":       "ICL-POW",
    "persona_pow_fs10random_user":     "ICLR-POW",
    "persona_powe_fs0":                "ZS-POWE",
    "persona_powe_fs10same_user":      "ICL-POWE",
    "persona_powe_fs10random_user":    "ICLR-POWE",
    "persona_power_fs0":               "ZS-POWER",
    "persona_power_fs10same_user":     "ICL-POWER",
    "persona_power_fs10random_user":   "ICLR-POWER",
}


def normalize(key: str) -> str:
    key = _PARAMS_RE.sub('', key)
    if key == 'Ground Truth':
        key = 'ground_truth'
    return key


def _load_similarity_csv(path: str) -> dict:
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        key = normalize(row['dataset'])
        result[key] = {'sim': row['similarity'], 'offset': -abs(row['offset_pct'])}
    return result


def load_adherence(similarity_csv: str) -> dict:
    return _load_similarity_csv(similarity_csv)


def load_emotionality(similarity_csv: str) -> dict:
    return _load_similarity_csv(similarity_csv)


def load_consistency(consistency_dir: str) -> dict:
    base = Path(consistency_dir)
    datasets = {}
    for out_file in base.glob("*/out.json"):
        key = normalize(out_file.parent.name)
        with open(out_file) as f:
            datasets[key] = json.load(f)

    gt_overall = datasets['ground_truth']['overall']
    result = {}
    for key, m in datasets.items():
        if key == 'ground_truth':
            result[key] = {'sim': 1.0, 'offset': 0.0}
        else:
            sim = 1 - abs(m['overall'] - gt_overall) / gt_overall
            result[key] = {'sim': sim, 'offset': -(abs(m['overall'] - gt_overall) / gt_overall) * 100}
    return result


def load_naturalness(overall_summary: str) -> dict:
    with open(overall_summary) as f:
        data = json.load(f)

    datasets = {normalize(k): v for k, v in data['datasets'].items()}
    gt_nat = datasets['ground_truth']['naturalness_score']

    result = {}
    for key, m in datasets.items():
        if key == 'ground_truth':
            result[key] = {'sim': 1.0, 'offset': 0.0}
        else:
            sim = 1 - abs(m['naturalness_score'] - gt_nat) / gt_nat
            result[key] = {'sim': sim, 'offset': -(abs(m['naturalness_score'] - gt_nat) / gt_nat) * 100}
    return result


def compute_e4s(adherence_csv, consistency_dir, naturalness_json, emotionality_csv):
    adh = load_adherence(adherence_csv)
    con = load_consistency(consistency_dir)
    nat = load_naturalness(naturalness_json)
    emo = load_emotionality(emotionality_csv)

    all_keys = (set(adh) | set(con) | set(nat) | set(emo)) - {'ground_truth'}
    e4s = {}
    for key in all_keys:
        a = adh.get(key)
        c = con.get(key)
        n = nat.get(key)
        e = emo.get(key)
        if not (a and c and n and e):
            continue
        e4s[key] = (a['sim'] + c['sim'] + n['sim'] + e['sim']) / 4
    return e4s


def build_model_by_scenario(e4s, output_file, configs=None):
    scenario_order = [s for s in _SCENARIO_ORDER if not configs or any(c in s for c in configs)]
    n_scenarios = len(scenario_order)
    n_models = len(_MODEL_SLUGS)

    ranked_per_model = {}
    for slug in _MODEL_SLUGS:
        scores = {s: e4s.get(f"{slug}_{s}") for s in scenario_order}
        ranked_per_model[slug] = sorted([(s, v) for s, v in scores.items() if v is not None], key=lambda x: x[1])

    # col spec: 3 subcolumns per model
    col_spec = "@{}" + " ".join("l r r" for _ in _MODEL_SLUGS) + "@{}"

    # top header: model names spanning 3 cols each
    top_header_parts = []
    cmidrules = []
    col = 1
    for slug in _MODEL_SLUGS:
        top_header_parts.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{_MODEL_NAMES[slug]}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    top_header = " & ".join(top_header_parts)

    # sub header
    sub_header = " & ".join(["\\textbf{Strategy} & \\textbf{Rel.} & \\textbf{Cum.}"] * n_models)

    lines = [
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Strategy quality ladder per model size, ranked from best (top) to worst (bottom) by \\textsc{e4s} score. Rel.\\ is the step-wise gain over the previous rank; Cum.\\ is the cumulative gain over the worst-performing strategy. The worst row shows the base \\textsc{e4s} score.}",
        "  \\label{tab:model_by_scenario}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {top_header} \\\\",
        f"    {'  '.join(cmidrules)}",
        f"    {sub_header} \\\\",
        "    \\midrule",
    ]

    for i in range(n_scenarios - 1, -1, -1):
        cells = []
        for slug in _MODEL_SLUGS:
            ranked = ranked_per_model[slug]
            if i >= len(ranked):
                cells += ["---", "---", "---"]
                continue
            s, v = ranked[i]
            name = _SCENARIO_NAMES.get(s, s)
            base_v = ranked[0][1]
            if i == 0:
                name_cell = f"\\textit{{{name}}}"
                rel_cell = f"\\textit{{{base_v:.3f}}}"
                cum_cell = "---"
            else:
                prev_v = ranked[i - 1][1]
                rel_gain = (v - prev_v) / prev_v * 100
                cum_gain = (v - base_v) / base_v * 100
                name_cell = f"\\textbf{{{name}}}" if i == n_scenarios - 1 else name
                rel_cell = f"+{rel_gain:.1f}\\%"
                cum_cell = f"+{cum_gain:.1f}\\%"
            cells += [name_cell, rel_cell, cum_cell]

        lines.append("    " + " & ".join(cells) + " \\\\")

    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}"]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text("\n".join(lines) + "\n")
    print(f"Table written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adherence",    required=True, help="data/adherence/similarity.csv")
    parser.add_argument("--consistency",  required=True, help="data/results/consistency/")
    parser.add_argument("--naturalness",  required=True, help="data/results/naturalness/overall_summary.json")
    parser.add_argument("--emotionality", required=True, help="data/results/emotionality/similarity.csv")
    parser.add_argument("--output",       required=True)
    parser.add_argument("--configs", nargs="*", default=None, help="scenario tags to include (substring match)")
    args = parser.parse_args()

    e4s = compute_e4s(args.adherence, args.consistency, args.naturalness, args.emotionality)
    build_model_by_scenario(e4s, args.output, args.configs)
