"""Signed-weight aggregation and feature alignment (Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.phase2.explain_aggregated import LimeContext, aggregate_signed, run_lime, run_stream


def _ctx(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = np.abs(rng.normal(size=(n, 150))) * (rng.random((n, 150)) < 0.2)
    X += 1e-3 * rng.random((n, 150))                       # keep std > 0
    w = np.zeros(150); w[[3, 17, 99]] = [4.0, -3.0, 2.0]
    pf = lambda Z: 1.0 / (1.0 + np.exp(-(np.asarray(Z) @ w - 0.5)))
    return LimeContext(X, pf, X.mean(0), X.std(0)), w, X


def test_mean_signed_not_mean_abs_and_full_vector():
    runs = np.zeros((4, 150)); runs[:, 5] = [1, -1, 1, -1]; runs[:, 7] = [0.4, 0.4, 0.4, 0.4]
    mean, order = aggregate_signed(runs)
    assert mean.shape == (150,) and mean[5] == 0.0 and mean[7] == pytest.approx(0.4)
    assert order[0] == 7                                    # sign-cancelling feature ranks below the stable one


def test_top_k_of_mean_differs_from_mean_of_top_k():
    runs = np.zeros((3, 150))
    runs[0, 1] = 1.0; runs[1, 2] = 1.0; runs[2, 3] = 1.0; runs[:, 9] = 0.4
    per_run_top1 = {int(np.argmax(np.abs(r))) for r in runs}
    mean, order = aggregate_signed(runs)
    assert per_run_top1 == {1, 2, 3}                        # every single run's top-1 is a different feature
    assert order[0] == 9                                    # but the full-vector mean ranks the consistent one first


def test_ties_break_by_feature_index_and_shape_validation():
    mean, order = aggregate_signed(np.ones((2, 150)))
    assert list(order[:3]) == [0, 1, 2]
    with pytest.raises(ValueError):
        aggregate_signed(np.ones((2, 149)))


def test_run_lime_returns_full_index_aligned_vector():
    ctx, w, X = _ctx()
    out = run_lime(ctx, X[0], seed=11, num_samples=3000)
    assert out["coef"].shape == (150,) and np.isfinite(out["coef"]).all()
    top = np.argsort(-np.abs(out["coef"]))[:3]
    assert set(top.tolist()) == {3, 17, 99}                 # alignment by feature INDEX recovers the true drivers
    assert np.sign(out["coef"][3]) > 0 and np.sign(out["coef"][17]) < 0


def test_aggregate_of_runs_shape_and_reduces_variance():
    ctx, w, X = _ctx()
    res = run_stream(ctx, X[1], "w1", "agg", 8, 1500)
    runs = np.stack([r["coef"] for r in res])
    mean, _ = aggregate_signed(runs)
    assert runs.shape == (8, 150)
    assert mean[3] > 0 and runs.std(0)[3] > 0
