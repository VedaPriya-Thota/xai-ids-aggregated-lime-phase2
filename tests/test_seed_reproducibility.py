"""Seeding, determinism, ESS re-derivation and LIME provenance."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.phase2.explain_aggregated import LimeContext, lime_ess, run_lime, seed_for
from src.phase2.frozen import LIME_SHA256, assert_lime_provenance
from tests.test_aggregation import _ctx


def test_seed_for_is_deterministic_and_separates_streams_windows_runs():
    assert seed_for("agg", "w", 0) == seed_for("agg", "w", 0)
    s = {seed_for(a, b, c) for a in ("agg", "single", "repA", "repB", "cm") for b in ("w1", "w2") for c in range(5)}
    assert len(s) == 50
    assert 0 <= seed_for("agg", "w", 0) < 2 ** 32


def test_fixed_seed_reproduces_bit_identically_and_is_order_independent():
    ctx, _, X = _ctx()
    a = run_lime(ctx, X[0], 5, 2000)["coef"]
    run_lime(ctx, X[1], 6, 2000)                         # unrelated call in between must not matter (fresh explainer)
    b = run_lime(ctx, X[0], 5, 2000)["coef"]
    c = run_lime(ctx, X[0], 6, 2000)["coef"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_rederived_weights_reproduce_standin_coefficients_and_ess():
    ctx, _, X = _ctx()
    row, seed, n = X[2], 77, 2500
    out = run_lime(ctx, row, seed, n)
    z = ctx.mean + np.random.RandomState(seed).normal(size=(n, 150)) * ctx.std
    z[0] = row
    d = np.linalg.norm((z - row) / ctx.std, axis=1)
    w = np.sqrt(np.exp(-d ** 2 / (np.sqrt(150) * 0.75) ** 2))
    reg = Ridge(alpha=1.0).fit(z, ctx.predict_attack(z), sample_weight=w)
    assert np.allclose(reg.coef_, out["coef"], atol=1e-12)
    assert lime_ess(ctx, row, seed, n) == (w.sum() ** 2 / (w ** 2).sum())


def test_vendored_lime_provenance_guard():
    info = assert_lime_provenance()
    assert info["version"] == "vendored-compat-0.2.0.1" and info["sha256"] == LIME_SHA256
