"""Raw-syscall mapping/edit invariants, R2/R1 pools, matched random draws, sign convention.

Uses the REAL frozen vectorizer/selector/model (read-only) but no raw dataset.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.phase2 import evaluate_perturb as ev
from src.phase2.frozen import attack_logit
from src.phase2.perturb_raw_sequence import actionable_mask, apply_edit, occurrences
from src.phase3c import choose_neutral, tokens

BASE = Path(__file__).resolve().parents[1] / "results/experiments/phase3a_strict_reconstructed_mlp"
FREQ = {3: 1000, 5: 500, 6: 100, 10: 50}


@pytest.fixture(scope="module")
def frozen():
    vec = joblib.load(BASE / "preprocessing/tfidf_vectorizer.joblib")
    sel = joblib.load(BASE / "preprocessing/chi2_selector.joblib")
    model = joblib.load(BASE / "mlp_model.joblib")
    names = pd.read_csv(BASE / "preprocessing/selected_feature_names.csv").sort_values("model_column_index").feature_name.tolist()
    fz = SimpleNamespace(vec=vec, sel=sel, model=model, names=names, freq=dict(FREQ),
                         predict_attack=lambda X: model.predict_proba(np.asarray(X, float))[:, 1])
    return fz


def _X(fz, seq):
    Z = fz.sel.transform(fz.vec.transform([tokens(np.asarray(seq))]))
    return np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z)[0]


def _windows(fz, n=60, seed=0):
    rng = np.random.default_rng(seed)
    ids = sorted({int(t) for nm in fz.names for t in nm.split("_")})
    out = []
    for _ in range(n):
        w = rng.choice(ids, 30)
        for _ in range(rng.integers(1, 6)):                       # plant selected bigrams at random positions
            a, b = map(int, fz.names[rng.integers(0, 150)].split("_"))
            i = rng.integers(0, 29)
            w[i], w[i + 1] = a, b
        out.append(w.astype(np.int64))
    return out


def test_attack_logit_matches_frozen_probability(frozen):
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(50, 150))) * (rng.random((50, 150)) < 0.1)
    p = frozen.model.predict_proba(X)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-attack_logit(frozen.model, X))), p, atol=1e-12)


def test_selected_tfidf_feature_present_iff_bigram_occurs_in_raw_window(frozen):
    for w in _windows(frozen):
        present = _X(frozen, w) > 0
        occurs = np.array([len(occurrences(w, nm)) > 0 for nm in frozen.names])
        assert (present == occurs).all()


def test_edit_invariants_length_positions_and_policy(frozen):
    for w in _windows(frozen, 80, seed=3):
        pres = np.flatnonzero(_X(frozen, w) > 0)
        if len(pres) < 3:
            continue
        r = apply_edit(w, pres[:3], frozen.names, frozen.freq)
        assert len(r.sequence) == 30 and r.length_preserved
        assert 0 not in r.positions_edited and len(set(r.positions_edited)) == len(r.positions_edited)
        assert r.changed_calls == len(r.positions_edited)
        allowed = {i + 1 for j in pres[:3] for i in occurrences(w, frozen.names[j])}
        assert set(r.positions_edited) <= allowed
        for pos in r.positions_edited:
            assert int(r.sequence[pos]) in FREQ and r.sequence[pos] != w[pos]
        assert (np.array(w) == r.sequence)[[i for i in range(30) if i not in r.positions_edited]].all()


def test_consumed_occurrence_is_logged_not_silently_skipped():
    names = ["1_2", "2_3"]
    w = np.array([1, 2, 3] + [9] * 27, dtype=np.int64)
    r = apply_edit(w, [0, 1], names, {3: 10, 4: 5})
    assert r.sequence[1] == 4 or r.sequence[1] == 3                # second call of "1_2" replaced
    assert r.n_features_edited == 1 and not r.success_all
    assert r.failures == [{"feature": "2_3", "reason": "occurrence_consumed_by_prior_edit"}]


def test_overlapping_bigrams_each_position_edited_once():
    w = np.array([7] * 30, dtype=np.int64)
    r = apply_edit(w, [0], ["7_7"], {3: 10, 5: 1})
    assert len(r.sequence) == 30 and len(set(r.positions_edited)) == len(r.positions_edited)
    assert (r.sequence[r.positions_edited] == 3).all() and 0 not in r.positions_edited


def test_absent_feature_is_reported_absent():
    r = apply_edit(np.array([1, 2] * 15, dtype=np.int64), [0], ["8_9"], {3: 1})
    assert r.n_features_edited == 0 and r.failures[0]["reason"] == "absent_in_original_window"


def test_replacement_policy_is_phase1_choose_neutral():
    assert choose_neutral({3: 9, 5: 4}, original=3, preceding=1, target_pair=(1, 3)) == 5
    assert choose_neutral({3: 9, 5: 4}, original=7, preceding=1, target_pair=(1, 3)) == 5


def test_guided_and_random_use_identical_edit_semantics(frozen):
    w = _windows(frozen, 30, seed=5)[0]
    pres = np.flatnonzero(_X(frozen, w) > 0)[:3]
    p0 = float(frozen.predict_attack(_X(frozen, w)[None])[0]); l0 = float(attack_logit(frozen.model, _X(frozen, w)[None])[0])
    row = ev.evaluate_arm(frozen, w, p0, l0, [pres, pres], frozen.names)
    direct = apply_edit(w, pres, frozen.names, frozen.freq)
    p1 = float(frozen.predict_attack(_X(frozen, direct.sequence)[None])[0])
    assert row[0]["prob_after"] == pytest.approx(p1) and row[0] == row[1]


def test_pools_follow_predicted_class_sign_presence_and_actionability():
    present = np.zeros(150, bool); present[[1, 2, 3, 4, 5]] = True
    act = np.ones(150, bool); act[5] = False
    coef = np.zeros(150); coef[[1, 2, 3, 4, 5, 6]] = [3, -2, 1, 0.5, 9, 8]
    a = ev.pools(coef, present, act, pred_attack=1)
    n = ev.pools(coef, present, act, pred_attack=0)
    assert a["r2"].tolist() == [1, 3, 4] and n["r2"].tolist() == [2]
    assert a["r1"].tolist() == [1, 2, 3, 4]                         # R1 ignores sign; 5 non-actionable, 6 absent


def test_select_guided_top_abs_and_never_padded():
    coef = np.zeros(150); coef[[4, 9, 2, 7]] = [-5, 5, 3, 1]
    assert ev.select_guided(coef, np.array([2, 4, 7, 9])).tolist() == [4, 9, 2]    # tie |5|: lower index first
    assert ev.select_guided(coef, np.array([2, 4])).size == 0


def test_matched_draws_pool_only_reproducible_and_shared_across_methods():
    perms = ev.priority_permutations("wX", 50)
    assert np.array_equal(perms, ev.priority_permutations("wX", 50))
    assert not np.array_equal(perms, ev.priority_permutations("wY", 50))
    pool = np.array([3, 8, 12, 40, 77])
    d1 = [ev.draw_from_pool(p, pool) for p in perms]
    d2 = [ev.draw_from_pool(p, pool.copy()) for p in perms]
    assert all(set(d.tolist()) <= set(pool.tolist()) and len(d) == 3 for d in d1) and all((a == b).all() for a, b in zip(d1, d2))
    sub = np.array([3, 8, 12, 40])                                   # a smaller pool keeps the SAME priority order
    for p, d in zip(perms[:5], [ev.draw_from_pool(p, sub) for p in perms[:5]]):
        assert d.tolist() == [f for f in p if f in set(sub.tolist())][:3]
    assert len({tuple(sorted(d.tolist())) for d in d1}) > 1          # draws actually vary


def test_outcome_sign_convention():
    o = ev.outcomes(0.9, 2.0, np.array([0.3]), np.array([-0.8]))
    assert o["attack_logit_drop"][0] == pytest.approx(2.8) and o["direction_aware_drop"][0] == pytest.approx(2.8)
    assert o["flip_attack_to_normal"][0] == 1.0 and o["flip_any"][0] == 1.0
    o = ev.outcomes(0.2, -1.4, np.array([0.7]), np.array([0.8]))
    assert o["attack_logit_drop"][0] == pytest.approx(-2.2) and o["direction_aware_drop"][0] == pytest.approx(2.2)
    assert o["flip_attack_to_normal"][0] == 0.0 and o["flip_any"][0] == 1.0


def test_evaluate_method_statuses_budget_three_and_no_padding(frozen):
    w = None
    for cand in _windows(frozen, 200, seed=9):                      # find a window with >=5 present, actionable features
        x = _X(frozen, cand)
        if (x > 0).sum() >= 5:
            w, row = cand, x
            break
    assert w is not None
    p0 = float(frozen.predict_attack(row[None])[0]); l0 = float(attack_logit(frozen.model, row[None])[0])
    act = actionable_mask(w, row > 0, frozen.names, frozen.freq)
    perms = ev.priority_permutations("wT", 7)
    s = 1.0 if p0 >= 0.5 else -1.0
    pres = np.flatnonzero((row > 0) & act)

    def coef_with(k):
        c = np.zeros(150); c[pres[:k]] = s * np.arange(k, 0, -1); return c
    r4 = ev.evaluate_method(frozen, "wT", w, row, p0, l0, coef_with(4), act, perms, frozen.names)
    assert r4["status"] == "ok" and len(r4["r2_draws"]) == 7 and r4["guided"]["n_features_requested"] == 3
    r3 = ev.evaluate_method(frozen, "wT", w, row, p0, l0, coef_with(3), act, perms, frozen.names)
    assert r3["status"] == "R2_uninformative" and "guided" in r3 and "r2_draws" not in r3
    r2 = ev.evaluate_method(frozen, "wT", w, row, p0, l0, coef_with(2), act, perms, frozen.names)
    assert r2["status"] == "guided_infeasible" and "guided" not in r2        # never padded with absent/unsupported features
