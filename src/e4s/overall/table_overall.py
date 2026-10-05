import re
import json
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


def normalize(key: str) -> str:
    key = _PARAMS_RE.sub('', key)
    if key == 'Ground Truth':
        key = 'ground_truth'
    return key


def pretty_name(key: str) -> str:
    if key == "ground_truth":
        return "Reddit"
    key = _PARAMS_RE.sub('', key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            return f"\\textsc{{{model_name} {_SCENARIO_NAMES.get(scenario_tag, scenario_tag)}}}"
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


def rank_positions(scores: dict, ref_key: str) -> dict:
    non_ref = {k: v['sim'] for k, v in scores.items() if k != ref_key}
    sorted_keys = sorted(non_ref, key=lambda k: non_ref[k], reverse=True)
    return {k: i + 1 for i, k in enumerate(sorted_keys)}


def rank_fmt(s: str, pos: int, ranks: dict) -> str:
    top = sorted(set(ranks.values()))[:3]
    if top and pos == top[0]:
        return f"\\textbf{{{s}}}"
    if len(top) > 1 and pos == top[1]:
        return f"\\underline{{{s}}}"
    if len(top) > 2 and pos == top[2]:
        return f"\\textit{{{s}}}"
    return s


def fmt_score(v: float, pos: int, ranks: dict) -> str:
    return rank_fmt(f"{v:.3f}", pos, ranks)


def fmt_offset(v: float) -> str:
    return f"{v:+.2f}\\%"


def fmt_pos(pos: int, ranks: dict) -> str:
    return rank_fmt(str(pos), pos, ranks)


def build_table(adherence_csv, consistency_dir, naturalness_json, emotionality_csv, output_file, configs=None):
    ref_key = 'ground_truth'

    adh = load_adherence(adherence_csv)
    con = load_consistency(consistency_dir)
    nat = load_naturalness(naturalness_json)
    emo = load_emotionality(emotionality_csv) if emotionality_csv else {}

    sim_keys = (set(adh) | set(con) | set(nat)) - {ref_key}
    if configs:
        sim_keys = {k for k in sim_keys if any(c in k for c in configs)}

    def filter_scores(d):
        return {k: v for k, v in d.items() if k == ref_key or k in sim_keys}

    adh_ranks = rank_positions(filter_scores(adh), ref_key)
    con_ranks = rank_positions(filter_scores(con), ref_key)
    nat_ranks = rank_positions(filter_scores(nat), ref_key)
    emo_ranks = rank_positions(filter_scores(emo), ref_key) if emo else {}
    rows = []
    for key in sim_keys:
        a = adh.get(key)
        c = con.get(key)
        n = nat.get(key)
        e = emo.get(key) if emo else None
        if not (a and c and n):
            continue
        if emo and not e:
            continue
        dims = [a['sim'], c['sim'], n['sim']] + ([e['sim']] if e else [])
        e4s_sim = sum(dims) / len(dims)
        e4s_off = -(abs(e4s_sim - 1.0)) * 100
        rows.append((key, a, c, n, e, e4s_sim, e4s_off))

    e4s_ranks = {key: i + 1 for i, (key, *_) in enumerate(
        sorted(rows, key=lambda r: r[5], reverse=True)
    )}
    rows.sort(key=lambda r: r[5], reverse=True)

    n_dims = 4 if emo else 3
    dim_caption = ("four" if emo else "three")
    emo_dims = (" Emotionality," if emo else "")
    col_spec = "l" + "rrc" * n_dims + "rrc"
    emo_header = (" & \\multicolumn{3}{c}{\\textbf{Emotionality}}" if emo else "")
    emo_cmidrule = (" \\cmidrule(lr){11-13}" if emo else "")
    emo_subheader = (" & Score & Off. & Pos." if emo else "")
    e4s_cols = (14 if emo else 11)

    lines = [
        "\\begin{table*}[ht]",
        "  \\centering",
        f"  \\caption{{Overall (\\textsc{{e4s}}) evaluation across {dim_caption} dimensions using Sim.\\ scores."
        f" \\textsc{{e4s}} score is the unweighted mean of Adherence, Consistency, Naturalness,{emo_dims}"
        " and Emotionality ($1-\\text{{JSD}}$) Sim.\\ scores. Rows sorted by \\textsc{{e4s}} score."
        " Top performers highlighted: \\textbf{bold} for first, \\underline{underline} for second,"
        " \\textit{italic} for third in each column.}",
        "  \\label{tab:overall_results}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    & \\multicolumn{3}{c}{\\textbf{Adherence}}"
        " & \\multicolumn{3}{c}{\\textbf{Consistency}}"
        " & \\multicolumn{3}{c}{\\textbf{Naturalness}}"
        f"{emo_header}"
        " & \\multicolumn{3}{c}{\\textbf{\\textsc{e4s}}} \\\\",
        f"    \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}"
        f"{emo_cmidrule} \\cmidrule(lr){{{e4s_cols}-{e4s_cols+2}}}",
        "    \\textbf{Dataset}"
        " & Score & Off. & Pos."
        " & Score & Off. & Pos."
        " & Score & Off. & Pos."
        f"{emo_subheader}"
        " & Score & Off. & Pos. \\\\",
        "    \\midrule",
    ]

    ref_name = pretty_name(ref_key)
    ref_emo_cols = " & \\textit{1.000} & \\textit{---} & \\textit{---}" if emo else ""
    lines.append(
        f"    \\textit{{{ref_name}}}"
        " & \\textit{1.000} & \\textit{---} & \\textit{---}"
        " & \\textit{1.000} & \\textit{---} & \\textit{---}"
        " & \\textit{1.000} & \\textit{---} & \\textit{---}"
        f"{ref_emo_cols}"
        " & \\textit{1.000} & \\textit{---} & \\textit{---} \\\\"
    )
    lines.append("    \\midrule")

    for key, a, c, n, e, e4s_sim, e4s_off in rows:
        name = pretty_name(key)
        ap  = adh_ranks.get(key, 0)
        cp  = con_ranks.get(key, 0)
        np_ = nat_ranks.get(key, 0)
        xp  = e4s_ranks.get(key, 0)
        emo_cols = ""
        if e:
            ep = emo_ranks.get(key, 0)
            emo_cols = (
                f" & {fmt_score(e['sim'], ep, emo_ranks)}"
                f" & {fmt_offset(e['offset'])}"
                f" & {fmt_pos(ep, emo_ranks)}"
            )
        lines.append(
            f"    {name:<20}"
            f" & {fmt_score(a['sim'], ap, adh_ranks)}"
            f" & {fmt_offset(a['offset'])}"
            f" & {fmt_pos(ap, adh_ranks)}"
            f" & {fmt_score(c['sim'], cp, con_ranks)}"
            f" & {fmt_offset(c['offset'])}"
            f" & {fmt_pos(cp, con_ranks)}"
            f" & {fmt_score(n['sim'], np_, nat_ranks)}"
            f" & {fmt_offset(n['offset'])}"
            f" & {fmt_pos(np_, nat_ranks)}"
            f"{emo_cols}"
            f" & {fmt_score(e4s_sim, xp, e4s_ranks)}"
            f" & {fmt_offset(e4s_off)}"
            f" & {fmt_pos(xp, e4s_ranks)} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table*}",
    ]

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

    build_table(args.adherence, args.consistency, args.naturalness, args.emotionality, args.output, args.configs)
