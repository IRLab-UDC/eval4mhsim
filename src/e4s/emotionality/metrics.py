from collections import Counter
import numpy as np

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]


def distribution(labels: list[str]) -> dict:
    counts = Counter(labels)
    total = len(labels)
    return {e: counts.get(e, 0) / total for e in EMOTIONS}


def kl_divergence(p: dict, q: dict, eps: float = 1e-8) -> float:
    total = 0.0
    for e in EMOTIONS:
        pi = p.get(e, 0.0)
        qi = q.get(e, 0.0)
        if pi > 0.0:
            total += pi * np.log(pi / max(qi, eps))
    return total


def jsd_similarity(gt: dict, sim: dict, emotions: list[str] = EMOTIONS) -> float:
    """1 - JSD(gt || sim) / ln(2), normalized to [0, 1]; 1.0 means identical distributions."""
    p = np.array([gt.get(e, 0.0)  for e in emotions], dtype=float)
    q = np.array([sim.get(e, 0.0) for e in emotions], dtype=float)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return 1.0 - jsd / np.log(2)
