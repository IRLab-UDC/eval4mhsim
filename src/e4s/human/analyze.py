import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.multitest import multipletests

TARGET = "POWER-ICL"
HYPOTHESES = {("POWER-ICL vs P-ICL", "identity"): "H1", ("POWER-ICL vs POWER-ZS", "identity"): "H2",
              ("POWER-ICL vs POWER-ZS", "tone"): "H3"}


def agreement(g):
    pairs = [a == b for _, x in g.groupby("instance") for a, b in itertools.combinations(x.choice, 2)]
    po = np.mean(pairs)
    pa = (g.choice == TARGET).mean()
    pe = pa ** 2 + (1 - pa) ** 2
    pe_ac1 = 2 * pa * (1 - pa)
    counts = g.groupby("instance").choice.value_counts().unstack(fill_value=0)
    n = counts.values.sum()
    observed = sum((r.sum() ** 2 - (r ** 2).sum()) / (r.sum() - 1) for r in counts.values)
    expected = (n ** 2 - (counts.sum(0).values ** 2).sum()) / (n - 1)
    return {"pairwise": po, "gwet_ac1": (po - pe_ac1) / (1 - pe_ac1), "krippendorff_alpha": 1 - observed / expected,
            "unanimous": g.groupby("instance").choice.nunique().eq(1).mean()}


def mixed_p(g):
    d = g.assign(win=(g.choice == TARGET).astype(int))
    fit = BinomialBayesMixedGLM.from_formula("win ~ 1", {"item": "0 + C(instance)", "annotator": "0 + C(annotator)"}, d).fit_vb()
    return 2 * norm.sf(abs(fit.fe_mean[0] / fit.fe_sd[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="human_assessment")
    a = p.parse_args()
    data = Path(a.data)
    raw = pd.read_csv(data / "judgments.csv")
    j = raw[raw.valid]
    items = pd.read_csv(data / "items.csv").set_index(["comparison", "instance"])
    rows, matches = [], []
    for (comp, q), g in j.groupby(["comparison", "question"], sort=False):
        win = g.groupby("instance").choice.apply(lambda c: (c == TARGET).mean())
        per_annotator = g.groupby("annotator").choice.apply(lambda c: (c == TARGET).mean())
        row = {"comparison": comp, "question": q, "hypothesis": HYPOTHESES.get((comp, q), "-"),
               "annotators": g.annotator.nunique(), "judgments": len(g), "win_rate": (g.choice == TARGET).mean(),
               "majority_win": (win > .5).sum() / (win != .5).sum(), "items_won": int((win > .5).sum()),
               "items_lost": int((win < .5).sum()), "unanimous_won": int((win == 1).sum()),
               "unanimous_lost": int((win == 0).sum()), "annotators_above_half": int((per_annotator > .5).sum()),
               "annotators_half": int((per_annotator == .5).sum()), "annotators_below_half": int((per_annotator < .5).sum()),
               **agreement(g), "p": mixed_p(g)}
        if q == "tone":
            closer = items.loc[comp].classifier_closer.reindex(win.index)
            decided = win != .5
            match = (win[decided] > .5) == (closer[decided] == TARGET)
            row["classifier_match"] = match.mean()
            matches.append(match)
        rows.append(row)
    res = pd.DataFrame(rows)
    tested = res.hypothesis != "-"
    res.loc[tested, "p_holm"] = multipletests(res.loc[tested, "p"], method="holm")[1]
    summary = pd.DataFrame([{"participants": raw.annotator.nunique(), "annotators": j.annotator.nunique(),
                             "judgments": len(j), "reply_a": (j.choice == j.reply_a).mean(),
                             "classifier_match": pd.concat(matches).mean()}])
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print()
    print(summary.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
