import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

STYLES = {
    "Ground Truth": {'marker': 'o', 'color': '#1f77b4'},
}

_FALLBACK_MARKERS = ['x', '+', '*', 'H', 'D', '^', 'd', 's', 'v', '<', '>', 'p', 'h']
_FALLBACK_COLORS  = ['#aec7e8', '#d62728', '#ff7f0e', '#9467bd', '#8c564b',
                     '#e377c2', '#2ca02c', '#17becf', '#7f7f7f', '#bcbd22',
                     '#ffbb78', '#c5b0d5', '#f7b6d2']

_PARAMS_RE = re.compile(r'_t[\d._]+_maxtok\d+')


def pretty_name(key: str) -> str:
    if key in ("Ground Truth", "ground_truth"):
        return "Reddit"
    key = _PARAMS_RE.sub('', key)
    for model_slug, model_name in _MODEL_NAMES.items():
        if key.startswith(model_slug):
            scenario_tag = key[len(model_slug) + 1:]
            return f"{model_name} {_SCENARIO_NAMES.get(scenario_tag, scenario_tag)}"
    return key


def plot_comparison(csv_file, output_file):
    df = pd.read_csv(csv_file)
    datasets = df['dataset'].unique().tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    fallback_idx = 0
    for dataset in datasets:
        sub = df[df['dataset'] == dataset].sort_values('pool_size')
        label = pretty_name(dataset)
        if dataset in STYLES:
            style = STYLES[dataset]
        else:
            style = {
                'marker': _FALLBACK_MARKERS[fallback_idx % len(_FALLBACK_MARKERS)],
                'color':  _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)],
            }
            fallback_idx += 1

        ax.plot(sub['pool_size'].values, sub['MRR'].values,
                marker=style['marker'], color=style['color'],
                linestyle='-', linewidth=3, markersize=12, label=label)

    ax.set_xlabel('Distractors', fontsize=20)
    ax.set_ylabel('MRR', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.grid(True, alpha=0.3, linestyle=':', color='gray')

    fig.tight_layout(pad=0.2)
    fig.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    plot_comparison(args.csv, args.output)
