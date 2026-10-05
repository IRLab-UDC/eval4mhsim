import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import pretty
from metrics import EMOTIONS, distribution

_REPORTED_SCENARIOS = {
    "ZS", "ZS-P", "ZS-POWER",
    "ICL", "ICL-P", "ICL-POWER",
    "ICLR", "ICLR-P", "ICLR-POWER",
}

LINE_WIDTH = 2
MARKER_SIZE = 12
FILL_ALPHA = 0.25
GRID_ALPHA = 0.6
LABEL_SIZE = 26
LEGEND_SIZE = 30
TITLE_SIZE = 24
GT_COLOR = "#ff4500"

SIM_COLORS     = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]
SIM_MARKERS    = ["o", "s", "^", "D", "v", "P"]
SIM_LINESTYLES = ["--", "-.", ":", "--", "-.", ":"]


def _get_emotions(show_neutral: bool) -> list[str]:
    return EMOTIONS if show_neutral else [e for e in EMOTIONS if e != "neutral"]


def _get_angles(emotions: list[str]) -> list[float]:
    return np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist() + [0]


def _polygon_area(values: list[float], angles: list[float]) -> float:
    vals = np.array(values + [values[0]])
    angs = np.array(angles)
    return 0.5 * abs(np.sum(vals[:-1] * vals[1:] * np.sin(np.diff(angs))))


def _iou(a: list[float], b: list[float], angles: list[float]) -> float:
    intersection = _polygon_area([min(x, y) for x, y in zip(a, b)], angles)
    area_a = _polygon_area(a, angles)
    area_b = _polygon_area(b, angles)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 1.0


def _radar(ax, values, angles, color, marker, label, linestyle="-"):
    vals = values + [values[0]]
    ax.plot(angles, vals, color=color, linewidth=LINE_WIDTH, linestyle=linestyle,
            marker=marker, markersize=MARKER_SIZE, label=label)
    ax.fill(angles, vals, color=color, alpha=FILL_ALPHA)


def _clean(name: str) -> str:
    name = name.replace("\\textsc{", "").replace("}", "")
    name = name.replace("Gemma 3 ", "").replace("gemma 3 ", "")
    return name


def _setup_ax(ax, emotions, angles, vmax):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([e.capitalize() for e in emotions], size=LABEL_SIZE)
    ax.set_ylim(0, vmax)
    ax.yaxis.set_visible(False)
    ax.grid(False)
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, vmax], color="grey", linewidth=0.5, alpha=GRID_ALPHA)
    ax.plot(angles, [vmax] * len(angles), color="grey", linewidth=1.0, alpha=GRID_ALPHA)


def plot_radar(distributions: dict, output_path: Path, k: int = 3, show_neutral: bool = True, scores: dict = None):
    emotions = _get_emotions(show_neutral)
    angles = _get_angles(emotions)

    gt_dist = distributions["ground_truth"]
    gt_vals = [gt_dist.get(e, 0) for e in emotions]

    keys = [k_ for k_ in distributions if k_ != "ground_truth"
            and pretty(k_).rstrip("}").rsplit(" ", 1)[-1] in _REPORTED_SCENARIOS]
    if scores is None:
        scores = {k_: _iou(gt_vals, [distributions[k_].get(e, 0) for e in emotions], angles) for k_ in keys}
    ranked = sorted(keys, key=lambda k_: scores[k_])

    best  = ranked[-k:][::-1]
    worst = ranked[k-1::-1]

    all_keys = best + worst
    color_map     = {key: SIM_COLORS[i]     for i, key in enumerate(all_keys)}
    marker_map    = {key: SIM_MARKERS[i]    for i, key in enumerate(all_keys)}
    linestyle_map = {key: SIM_LINESTYLES[i] for i, key in enumerate(all_keys)}

    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(16, 8),
                                             subplot_kw=dict(projection="polar"))
    fig.subplots_adjust(bottom=0.18, wspace=0.4)

    for ax, group, title in [(ax_best, best, "Best"), (ax_worst, worst, "Worst")]:
        all_vals = [gt_vals] + [[distributions[k_].get(e, 0) for e in emotions] for k_ in group]
        vmax = max(max(v) for v in all_vals) * 1.1
        _setup_ax(ax, emotions, angles, vmax)
        for key in group:
            sim_vals = [distributions[key].get(e, 0) for e in emotions]
            _radar(ax, sim_vals, angles, color_map[key], marker_map[key],
                   _clean(pretty(key)), linestyle_map[key])
        _radar(ax, gt_vals, angles, GT_COLOR, "o", "Reddit")
        ax.set_title(title, fontsize=TITLE_SIZE + 2, fontweight="bold", pad=20)

    # shared legend below
    legend_handles = [Line2D([0], [0], color=GT_COLOR, marker="o", linewidth=LINE_WIDTH,
                             markersize=MARKER_SIZE, label="Reddit")]
    for key in all_keys:
        legend_handles.append(
            Line2D([0], [0], color=color_map[key], marker=marker_map[key],
                   linestyle=linestyle_map[key], linewidth=LINE_WIDTH,
                   markersize=MARKER_SIZE, label=_clean(pretty(key)))
        )

    gt_handle   = legend_handles[0]
    best_handles  = [Line2D([0], [0], color=color_map[key], marker=marker_map[key],
                            linestyle=linestyle_map[key], linewidth=LINE_WIDTH,
                            markersize=MARKER_SIZE, label=_clean(pretty(key))) for key in best]
    worst_handles = [Line2D([0], [0], color=color_map[key], marker=marker_map[key],
                            linestyle=linestyle_map[key], linewidth=LINE_WIDTH,
                            markersize=MARKER_SIZE, label=_clean(pretty(key))) for key in worst]

    fig.legend(handles=[gt_handle], loc="lower center", ncol=1,
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.legend(handles=best_handles, loc="lower center", ncol=len(best_handles),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.legend(handles=worst_handles, loc="lower center", ncol=len(worst_handles),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.16))

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="emotion_labels.csv")
    parser.add_argument("--output",  required=True, help="output PDF path")
    parser.add_argument("--k",       type=int, default=3)
    parser.add_argument("--neutral", action="store_true", default=False,
                        help="include neutral emotion axis")
    parser.add_argument("--similarity", default=None,
                        help="similarity.csv to rank configurations by; polygon IoU if omitted")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    distributions = {
        key: distribution(group["emotion"].tolist())
        for key, group in df.groupby("dataset")
    }
    scores = None
    if args.similarity:
        scores = pd.read_csv(args.similarity).set_index("dataset")["similarity"].to_dict()
    plot_radar(distributions, Path(args.output), args.k, args.neutral, scores)
