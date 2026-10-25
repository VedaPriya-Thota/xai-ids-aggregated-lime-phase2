"""Supplementary post-run analyses computed ONLY from saved outputs (no LIME, model or raw-data access).

1. Conditional attack-to-normal flip rate over windows the model originally predicted as attack (TP and FP).
2. Signed-vector Spearman correlation from the cached LIME coefficient arrays (supplementary to the predeclared |coef| ranking Spearman).
Run from the repository root:  python scripts/phase2_supplementary_analysis.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase2 import metrics_stability as ms          # noqa: E402
from src.phase2.explain_aggregated import aggregate_signed   # noqa: E402
from src.phase2.frozen import file_key                  # noqa: E402

R = ROOT / "results/phase2"
N_BOOT, SEED = 2000, 20261023 + 5000


def conditional_flips() -> pd.DataFrame:
    P = pd.read_csv(R / "perturbation_results.csv")
    A = P[P.pred_attack == 1]
    rows, seed = [], SEED
    for m in ["single", "agg", "cm"]:
        g = A[(A.method == m) & (A.arm == "guided")].set_index("window_id")
        for ctrl in ["r2", "r1"]:
            c = A[(A.method == m) & (A.arm == ctrl)].set_index("window_id")
            d = g[["group", "cluster", "flip_attack_to_normal"]].join(c[["flip_attack_to_normal"]], rsuffix="_ctrl", how="inner")
            d["diff"] = d.flip_attack_to_normal - d.flip_attack_to_normal_ctrl
            for name, col in (("guided flip rate", "flip_attack_to_normal"), (f"{ctrl.upper()} matched-random flip rate", "flip_attack_to_normal_ctrl"),
                              (f"guided minus {ctrl.upper()}", "diff")):
                if ctrl == "r1" and name == "guided flip rate":
                    continue
                for scope, sub, groups in (("TP", d[d.group == "TP"], None), ("FP", d[d.group == "FP"], None),
                                           ("TP+FP pooled by window", d, None), ("TP+FP equal-weight macro", d, d.group.to_numpy())):
                    if len(sub):
                        r = ms.cluster_bootstrap(sub[col].to_numpy(), sub.cluster.to_numpy(), n_boot=N_BOOT, seed=seed,
                                                 groups=None if groups is None else sub.group.to_numpy()); seed += 1
                        rows.append({"method": m, "control": ctrl.upper(), "quantity": name, "scope": scope,
                                     "denominator": "windows originally predicted attack (TP, FP) that have both arms",
                                     "guided_flip_windows": int((sub.flip_attack_to_normal >= 0.5).sum()) if col == "flip_attack_to_normal" else np.nan, **r})
    # guided single vs guided aggregate, same windows
    gs = A[(A.method == "single") & (A.arm == "guided")].set_index("window_id")
    ga = A[(A.method == "agg") & (A.arm == "guided")].set_index("window_id")
    d = ga[["group", "cluster", "flip_attack_to_normal"]].join(gs[["flip_attack_to_normal"]], rsuffix="_s", how="inner")
    d["diff"] = d.flip_attack_to_normal - d.flip_attack_to_normal_s
    for scope, sub in (("TP", d[d.group == "TP"]), ("FP", d[d.group == "FP"]), ("TP+FP pooled by window", d)):
        r = ms.cluster_bootstrap(sub["diff"].to_numpy(), sub.cluster.to_numpy(), n_boot=N_BOOT, seed=seed); seed += 1
        rows.append({"method": "agg vs single", "control": "-", "quantity": "aggregate guided minus single-run guided flip rate", "scope": scope,
                     "denominator": "windows originally predicted attack with both guided arms", "guided_flip_windows": np.nan, **r})
    return pd.DataFrame(rows)


def signed_spearman() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    S = pd.read_csv(R / "stability_results.csv")
    out = []
    for r in S.itertuples():
        k = file_key(r.window_id)
        load = lambda s: np.load(R / "lime_runs" / f"{k}_{s}.npz")["coef"]
        agg, ra, rb, cm = load("agg"), load("repA"), load("repB"), load("cm")
        pairs = list(itertools.combinations(range(len(agg)), 2))
        sig = lambda u, v: float(spearmanr(u, v)[0])
        absr = lambda u, v: ms.spearman_abs(u, v)
        out.append({"window_id": r.window_id, "group": r.group, "cluster": r.cluster,
                    "single_signed": float(np.mean([sig(agg[a], agg[b]) for a, b in pairs])),
                    "rep_signed": sig(aggregate_signed(ra)[0], aggregate_signed(rb)[0]), "cm_signed": sig(cm[0], cm[1]),
                    "single_abs_recomputed": float(np.mean([absr(agg[a], agg[b]) for a, b in pairs])),
                    "rep_abs_recomputed": absr(aggregate_signed(ra)[0], aggregate_signed(rb)[0]), "cm_abs_recomputed": absr(cm[0], cm[1]),
                    "single_abs_saved": r.single_spearman, "rep_abs_saved": r.rep_spearman, "cm_abs_saved": r.cm_spearman})
    W = pd.DataFrame(out)
    recon = {c: float(np.abs(W[c + "_recomputed"] - W[c + "_saved"]).max()) for c in ("single_abs", "rep_abs", "cm_abs")}
    W["d_rep_minus_single_signed"] = W.rep_signed - W.single_signed
    W["d_rep_minus_cm_signed"] = W.rep_signed - W.cm_signed
    W["d_rep_minus_cm_abs"] = W.rep_abs_saved - W.cm_abs_saved
    rows, seed = [], SEED + 500
    for col, name in (("single_signed", "single run: pairwise signed-vector Spearman"), ("rep_signed", "20-run aggregate replicates: signed-vector Spearman"),
                      ("cm_signed", "compute-matched 1x100k: signed-vector Spearman"), ("d_rep_minus_single_signed", "paired: aggregate minus single (signed)"),
                      ("d_rep_minus_cm_signed", "paired: aggregate minus compute-matched (signed)"),
                      ("d_rep_minus_cm_abs", "paired: aggregate minus compute-matched (|w| ranking, predeclared metric)")):
        rows.append({"quantity": name, "group": "ALL (macro, equal-weight groups)",
                     **ms.cluster_bootstrap(W[col].to_numpy(), W.cluster.to_numpy(), n_boot=N_BOOT, seed=seed, groups=W.group.to_numpy())}); seed += 1
    return W, pd.DataFrame(rows), recon


if __name__ == "__main__":
    cf = conditional_flips(); cf.to_csv(R / "supplementary_conditional_flip.csv", index=False)
    W, summ, recon = signed_spearman()
    W.to_csv(R / "supplementary_signed_spearman_per_window.csv", index=False); summ.to_csv(R / "supplementary_signed_spearman_summary.csv", index=False)
    print("cache reconciliation (max |recomputed abs-Spearman - saved|):", recon)
    pd.set_option("display.width", 220)
    print(cf[cf.control == "R2"][["method", "quantity", "scope", "guided_flip_windows", "n_windows", "n_clusters", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print(cf[cf.method == "agg vs single"][["quantity", "scope", "n_windows", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print(summ[["quantity", "estimate", "ci_low", "ci_high"]].round(3).to_string(index=False))
