"""Guided vs matched-random raw-syscall perturbation evaluation.

Sign convention (explicit):
  attack_logit_drop        = logit_before - logit_after          (positive => attack logit decreased)
  direction_aware_drop     = s * attack_logit_drop, s=+1 if the frozen model predicted attack, -1 if normal
                             (positive => the edit moved the model AWAY from its original decision)
  prob_change              = p_attack_after - p_attack_before
Primary pooled outcome is `direction_aware_drop`; `attack_logit_drop` is reported as specified. Neither is
interpreted on its own: effectiveness is judged against the matched random control (guided minus R2).
"""
from __future__ import annotations

import numpy as np

from src.phase2.explain_aggregated import seed_for
from src.phase2.frozen import attack_logit
from src.phase2.perturb_raw_sequence import apply_edit
from src.phase3c import tokens

B_PERTURB = 3          # perturbation budget (never 5)
MIN_POOL_GUIDED = 3
MIN_POOL_R2 = 4


def predicted_class_sign(prob: float) -> int:
    return 1 if prob >= 0.5 else -1


def pools(coef: np.ndarray, present: np.ndarray, actionable: np.ndarray, pred_attack: int) -> dict:
    """R2 pool = present & actionable & sign-supporting for the predicted class; R1 pool = present & actionable."""
    s = 1.0 if pred_attack else -1.0
    base = present & actionable
    return {"r1": np.flatnonzero(base), "r2": np.flatnonzero(base & (s * coef > 0))}


def select_guided(coef: np.ndarray, pool: np.ndarray, b: int = B_PERTURB) -> np.ndarray:
    """Top-b pool features by |coefficient| (stable tie-break by feature index). Never padded."""
    if len(pool) < b:
        return np.array([], dtype=int)
    order = sorted(pool.tolist(), key=lambda i: (-abs(coef[i]), i))
    return np.array(order[:b], dtype=int)


def priority_permutations(window_id: str, n_draws: int, n_features: int = 150) -> np.ndarray:
    """Seeded priority orders over all features, shared by every method so draws are matched across methods."""
    return np.stack([np.random.default_rng(seed_for("rand", window_id, d)).permutation(n_features) for d in range(n_draws)])


def draw_from_pool(perm: np.ndarray, pool: np.ndarray, b: int = B_PERTURB) -> np.ndarray:
    inpool = set(pool.tolist())
    return np.array([f for f in perm if int(f) in inpool][:b], dtype=int)


def score_sequences(fz, seqs) -> tuple[np.ndarray, np.ndarray]:
    X = fz.sel.transform(fz.vec.transform([tokens(s) for s in seqs]))
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
    return fz.predict_attack(X), attack_logit(fz.model, X)


def outcomes(p0: float, l0: float, p1: np.ndarray, l1: np.ndarray) -> dict:
    s = predicted_class_sign(p0)
    drop = l0 - l1
    return {"attack_logit_drop": drop, "direction_aware_drop": s * drop, "prob_change": p1 - p0,
            "flip_any": ((p1 >= 0.5) != (p0 >= 0.5)).astype(float),
            "flip_attack_to_normal": (((p1 < 0.5) & (p0 >= 0.5))).astype(float)}


def evaluate_arm(fz, window: np.ndarray, p0: float, l0: float, feature_sets, names) -> list[dict]:
    """Apply raw-sequence edits for each feature set; score once in a batch."""
    edits = [apply_edit(window, fs, names, fz.freq) for fs in feature_sets]
    for e in edits:
        if not e.length_preserved or len(e.sequence) != 30:
            raise RuntimeError("Edited window length != 30")
    p1, l1 = score_sequences(fz, [e.sequence for e in edits])
    out = outcomes(p0, l0, p1, l1)
    rows = []
    for i, e in enumerate(edits):
        rows.append({**{k: float(v[i]) for k, v in out.items()}, "prob_after": float(p1[i]), "logit_after": float(l1[i]),
                     "edit_success": bool(e.success_all), "n_features_requested": e.n_requested,
                     "n_features_edited": e.n_features_edited, "n_positions_edited": len(e.positions_edited),
                     "failure_reasons": ";".join(f"{f['feature']}:{f['reason']}" for f in e.failures)})
    return rows


def evaluate_method(fz, window_id: str, window: np.ndarray, row: np.ndarray, p0: float, l0: float, coef: np.ndarray,
                    actionable: np.ndarray, perms: np.ndarray, names, b: int = B_PERTURB) -> dict:
    """Guided + R2 (primary) + R1 (secondary) outcomes for one explanation vector of one window."""
    present = row > 0
    pred_attack = int(p0 >= 0.5)
    pl = pools(coef, present, actionable, pred_attack)
    res = {"pool_r2": len(pl["r2"]), "pool_r1": len(pl["r1"]), "status": "ok"}
    guided_feats = select_guided(coef, pl["r2"], b)
    res["guided_features"] = ";".join(names[i] for i in guided_feats)
    if len(guided_feats) == 0:
        res["status"] = "guided_infeasible"      # fewer than 3 eligible features: never padded
        return res
    res["guided"] = evaluate_arm(fz, window, p0, l0, [guided_feats], names)[0]
    if len(pl["r2"]) < MIN_POOL_R2:
        res["status"] = "R2_uninformative"
        res["r2_draw_features"] = []
    else:
        sets = [draw_from_pool(perm, pl["r2"], b) for perm in perms]
        res["r2_draws"] = evaluate_arm(fz, window, p0, l0, sets, names)
        res["r2_draw_features"] = [";".join(names[i] for i in s) for s in sets]
    if len(pl["r1"]) >= b:
        sets = [draw_from_pool(perm, pl["r1"], b) for perm in perms]
        res["r1_draws"] = evaluate_arm(fz, window, p0, l0, sets, names)
    return res
