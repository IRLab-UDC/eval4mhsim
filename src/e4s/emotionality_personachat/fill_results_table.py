import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import pretty, normalize_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/original_e4s/emotionality/results/similarity.csv")
    parser.add_argument("--output", default="data/original_e4s/emotionality/results/table_emotionality.tex")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["dataset"] = df["dataset"].apply(normalize_key)

    gt_row = df[df["dataset"] == "ground_truth"].iloc[0]
    sim_rows = df[df["dataset"] != "ground_truth"].sort_values("similarity", ascending=False)

    def fmt(v, italic=False):
        s = f"{v:.3f}"
        return f"\\textit{{{s}}}" if italic else s

    def fmt_offset(v, italic=False):
        s = f"{v:+.2f}\\%"
        return f"\\textit{{{s}}}" if italic else s

    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Emotionality evaluation on \\textsc{Eval4Sim} simulations. Similarity is $1 - \\text{JSD}(P_{\\text{gt}} \\| P_{\\text{sim}}) / \\ln 2$,"
        " where JSD is the Jensen--Shannon divergence between the \\textsc{PersonaChat} and simulation emotion distributions"
        " over Ekman emotions, normalized to $[0, 1]$. A score of 1.0 indicates identical distributions."
        " Offset is the signed relative difference from perfect alignment.}",
        "  \\label{tab:emotionality_e4s}",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    \\textbf{Model} & \\textbf{Sim.} & \\textbf{Offset} \\\\",
        "    \\midrule",
    ]

    lines.append(
        f"    \\textit{{{pretty('ground_truth')}}}"
        f" & {fmt(gt_row['similarity'], italic=True)}"
        f" & {fmt_offset(gt_row['offset_pct'], italic=True)} \\\\"
    )
    lines.append("    \\midrule")

    for _, row in sim_rows.iterrows():
        name = pretty(row["dataset"])
        lines.append(
            f"    {name}"
            f" & {fmt(row['similarity'])}"
            f" & {fmt_offset(-abs(row['offset_pct']))} \\\\"
        )

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Table written to {args.output}")


if __name__ == "__main__":
    main()
