import argparse
import re
from pathlib import Path


def clean(cell):
    cell = re.sub(r"F_\{([0-9.]+)\}\^u", r"F<sub>\1</sub><sup>u</sup>", cell)
    while re.search(r"\\[a-zA-Z]+\{([^{}]*)\}", cell):
        cell = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace("\\%", "%").replace("~", " ").replace("$", "")
    cell = cell.replace("\\", "").replace("{", "").replace("}", "").strip()
    return "+0.00%" if cell == "-0.00%" else cell


def parse(tex):
    body = tex[tex.index("\\begin{tabular"):tex.index("\\end{tabular}")]
    rows = [r.split("\\\\")[0] for r in body.split("\n") if "&" in r]
    return [[clean(c) for c in r.split("&")] for r in rows]


def to_markdown(rows, sort_by=None):
    header, data = rows[0], rows[1:]
    if sort_by:
        col = header.index(sort_by)
        reference, rest = data[0], data[1:]
        data = [reference] + sorted(rest, key=lambda r: float(r[col]), reverse=True)
    data[0] = [f"*{c}*" for c in data[0]]
    header[0] = "Configuration"
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in data]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="LaTeX table written by a fill_results_table.py script")
    parser.add_argument("--sort-by", default=None, help="column to sort by, descending; the first row (reference) stays first")
    args = parser.parse_args()
    print(to_markdown(parse(Path(args.input).read_text()), args.sort_by))
