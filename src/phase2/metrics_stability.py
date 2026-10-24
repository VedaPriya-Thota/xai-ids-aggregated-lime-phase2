"""Stability metrics and cluster-bootstrap inference."""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
from scipy.stats import rankdata


def top_k(coef: np.ndarray, k: int, mask: np.ndarray | None = None) -> set:
    """Indices of the k largest |coef| (stable tie-break by index); optionally restricted to a boolean mask."""
    a = np.abs(np.asarray(coef, dtype=float)).copy()
    if mask is not None:
        a[~np.asarray(mask, dtype=bool)] = -1.0
    order = np.argsort(-a, kind="stable")[:k]
    return {int(i) for i in order if a[i] >= 0}


def jaccard(a: set, b: set) -> float:
    u = a | b
    return float("nan") if not u else len(a & b) / len(u)


def jaccard_topk(u: np.ndarray, v: np.ndarray, k: int = 5, mask: np.ndarray | None = None) -> float:
    return jaccard(top_k(u, k, mask), top_k(v, k, mask))


def spearman_abs(u: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Spearman correlation of the importance rankings (|coef|), over all 150 features or a masked subset.

    Returns NaN when undefined (fewer than 2 elements or a constant ranking)."""
    a, b = np.abs(np.asarray(u, float)), np.abs(np.asarray(v, float))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sign_consistency_topk(u: np.ndarray, v: np.ndarray, k: int = 5) -> float:
    """Among features in BOTH top-k sets, the fraction whose coefficient signs agree (NaN if no overlap)."""
    common = sorted(top_k(u, k) & top_k(v, k))
    if not common:
        return float("nan")
    return float(np.mean([np.sign(u[i]) == np.sign(v[i]) for i in common]))


def sign_agreement_with_aggregate(runs: np.ndarray, agg: np.ndarray, k: int = 5) -> float:
    """For the aggregate's top-k features, the mean fraction of individual runs sharing the aggregate's sign."""
    feats = sorted(top_k(agg, k))
    return float(np.mean([(np.sign(runs[:, i]) == np.sign(agg[i])).mean() for i in feats]))


def pairwise_single_run_stability(runs: np.ndarray, k: int = 5, present: np.ndarray | None = None) -> dict:
    """Mean over all run pairs of top-k Jaccard, full-ranking Spearman and top-k sign consistency."""
    pairs = list(itertools.combinations(range(len(runs)), 2))
    j = [jaccard_topk(runs[a], runs[b], k, present) for a, b in pairs]
    s = [spearman_abs(runs[a], runs[b], present) for a, b in pairs]
    g = [sign_consistency_topk(runs[a], runs[b], k) for a, b in pairs]
    return {"jaccard": float(np.nanmean(j)), "spearman": float(np.nanmean(s)) if not np.all(np.isnan(s)) else float("nan"),
            "sign_consistency": float(np.nanmean(g)) if not np.all(np.isnan(g)) else float("nan"), "n_pairs": len(pairs)}


def expected_random_jaccard(n: int, k: int) -> float:
    """Exact expected Jaccard of two independent uniformly random k-subsets of n items (chance baseline)."""
    if k > n or k < 1:
        raise ValueError("require 1 <= k <= n")
    return float(sum(comb(k, i) * comb(n - k, k - i) / comb(n, k) * i / (2 * k - i) for i in range(0, k + 1)))


def cluster_bootstrap(values: np.ndarray, clusters, stat=np.mean, n_boot: int = 2000, seed: int = 0, groups=None) -> dict:
    """Percentile 95% CI by resampling whole clusters (identical-content groups) with replacement.

    `values` are per-window statistics (already paired where applicable). If `groups` is given, the statistic is
    the unweighted mean over groups of the per-group weighted mean (macro-average); any resample in which a group
    has no windows is discarded and counted in `n_empty_group`.
    """
    v = np.asarray(values, dtype=float)
    cl = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, cl = v[ok], cl[ok]
    gr = None if groups is None else np.asarray(groups)[ok]
    uniq, inv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(seed)
    point = _stat(v, np.ones_like(v), gr)
    boots, empty = [], 0
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        w = counts[inv].astype(float)
        s = _stat(v, w, gr)
        if np.isnan(s):
            empty += 1
            continue
        boots.append(s)
    b = np.asarray(boots)
    lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) else (float("nan"), float("nan")))
    p = float("nan") if not len(b) else min(1.0, 2 * min((np.sum(b <= 0) + 1) / (len(b) + 1), (np.sum(b >= 0) + 1) / (len(b) + 1)))
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "p_boot": float(p),
            "n_windows": int(len(v)), "n_clusters": int(len(uniq)), "n_boot": int(len(b)), "n_empty_group": int(empty)}


def _stat(v, w, gr):
    if gr is None:
        return float("nan") if w.sum() == 0 else float((v * w).sum() / w.sum())
    means = []
    for g in np.unique(gr):
        m = gr == g
        if w[m].sum() > 0:
            means.append((v[m] * w[m]).sum() / w[m].sum())
        else:
            return float("nan")
    return float(np.mean(means))


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
