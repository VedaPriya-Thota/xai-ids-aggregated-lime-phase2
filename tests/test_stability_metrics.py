"""Jaccard, Spearman, sign consistency, chance baseline, cluster bootstrap, Holm."""
from __future__ import annotations

import numpy as np
import pytest

from src.phase2 import metrics_stability as ms


def v(**kw):
    x = np.zeros(150)
    for k, val in kw.items():
        x[int(k[1:])] = val
    return x


def test_jaccard_cases_and_masks():
    a, b = v(f1=5, f2=4, f3=3, f4=2, f5=1), v(f1=5, f2=4, f3=3, f4=2, f5=1)
    assert ms.jaccard_topk(a, b, 5) == 1.0
    assert ms.jaccard_topk(v(f1=1), v(f2=1), 1) == 0.0
    assert ms.jaccard_topk(v(f1=3, f2=2), v(f1=3, f3=2), 2) == pytest.approx(1 / 3)
    m = np.zeros(150, bool)
    assert np.isnan(ms.jaccard_topk(a, b, 5, mask=m))      # nothing eligible -> undefined, not 0 or 1


def test_topk_tiebreak_stable_by_index():
    assert ms.top_k(np.ones(150), 3) == {0, 1, 2}


def test_spearman_edges():
    r = np.arange(150, dtype=float)
    assert ms.spearman_abs(r, r) == pytest.approx(1.0)
    assert ms.spearman_abs(r, r[::-1]) == pytest.approx(-1.0)
    assert np.isnan(ms.spearman_abs(np.ones(150), r))
    assert np.isnan(ms.spearman_abs(r, r, mask=np.eye(150, dtype=bool)[0]))


def test_sign_consistency():
    assert ms.sign_consistency_topk(v(f1=2, f2=-1), v(f1=3, f2=1), 2) == 0.5
    assert np.isnan(ms.sign_consistency_topk(v(f1=2), v(f2=2), 1))
    runs = np.stack([v(f1=1), v(f1=2), v(f1=-1)])
    assert ms.sign_agreement_with_aggregate(runs, runs.mean(0), 1) == pytest.approx(2 / 3)


def test_pairwise_single_run_stability():
    runs = np.stack([v(f1=3, f2=2, f3=1)] * 3)
    r = ms.pairwise_single_run_stability(runs, 3)
    assert r["jaccard"] == 1.0 and r["n_pairs"] == 3 and r["sign_consistency"] == 1.0


def test_expected_random_jaccard_exact_and_monte_carlo():
    assert ms.expected_random_jaccard(150, 5) == pytest.approx(0.019, abs=0.001)
    assert ms.expected_random_jaccard(10, 10) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(20000):
        a, b = set(rng.choice(20, 5, replace=False)), set(rng.choice(20, 5, replace=False))
        sims.append(len(a & b) / len(a | b))
    assert ms.expected_random_jaccard(20, 5) == pytest.approx(np.mean(sims), abs=0.01)
    with pytest.raises(ValueError):
        ms.expected_random_jaccard(3, 5)


def test_cluster_bootstrap_resamples_clusters_not_windows():
    vals = np.array([1.0] * 10 + [0.0] * 10)
    cl = ["a"] * 10 + ["b"] * 10                              # two clusters only
    r = ms.cluster_bootstrap(vals, cl, n_boot=2000, seed=1)
    assert r["n_clusters"] == 2 and r["n_boot"] == 2000 and r["estimate"] == 0.5
    assert r["ci_low"] == 0.0 and r["ci_high"] == 1.0         # window-level bootstrap would give a far tighter CI
    r2 = ms.cluster_bootstrap(vals, cl, n_boot=2000, seed=1)
    assert r == r2                                            # deterministic


def test_cluster_bootstrap_macro_and_empty_group_handling_and_nan_drop():
    vals = np.array([1.0, 1.0, np.nan, 3.0, 3.0, 3.0])
    cl = ["a", "b", "c", "d", "e", "f"]
    g = ["x", "x", "x", "y", "y", "y"]
    r = ms.cluster_bootstrap(vals, cl, groups=g, n_boot=1000, seed=3)
    assert r["estimate"] == pytest.approx(2.0) and r["n_windows"] == 5
    assert ms.cluster_bootstrap(vals, cl, n_boot=1000)["n_boot"] >= 1000 - 0


def test_holm():
    adj = ms.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
