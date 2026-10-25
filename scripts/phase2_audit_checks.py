"""Audit: verify every headline number in docs/phase2_results.md and docs/phase2_research_gap.md against the saved CSVs.
Uses only saved outputs (no LIME, model, or experiment re-run). Run from the repository root:
    python scripts/phase2_audit_checks.py [--write]     # --write saves results/phase2/audit_checks.csv
Exit status is non-zero if any check FAILS. Checks needing the raw dataset are reported as SKIP when it is absent.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase2 import metrics_stability as ms  # noqa: E402

R = ROOT / "results/phase2"
rows: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    rows.append((name, "SKIP" if ok is None else ("PASS" if ok else "FAIL"), detail))


def close(a, b, tol) -> bool:
    return abs(float(a) - float(b)) <= tol


sel = pd.read_csv(R / "selected_windows.csv")
S = pd.read_csv(R / "stability_results.csv")
P = pd.read_csv(R / "perturbation_results.csv")
st = pd.read_csv(R / "stability_summary.csv")
pc = pd.read_csv(R / "paired_comparisons.csv")
ex = pd.read_csv(R / "exclusions.csv")
rt = pd.read_csv(R / "runtime_summary.csv")
ed = pd.read_csv(R / "edit_diagnostics.csv")
el = pd.read_csv(R / "eligibility_report.csv")
audit = pd.read_csv(R / "eligibility_audit_all_windows.csv.gz")
fm = json.loads((R / "freeze_manifest.json").read_text())
proto_txt = (ROOT / "docs/phase2_protocol.yaml").read_text()

# ---- 1. sample
check("160 selected windows, 40 per group", len(sel) == 160 and (sel.group.value_counts() == 40).all() and set(sel.group) == {"TP", "FN", "TN", "FP"})
check("no duplicate windows", not sel.window_id.duplicated().any())
check("152 unique identical-content clusters", sel.cluster.nunique() == 152, f"{sel.cluster.nunique()}")
check("global cap: max 2 windows per cluster (all groups combined)", sel.groupby("cluster").size().max() <= 2, f"max={sel.groupby('cluster').size().max()}")
check("per-group clusters = 38/40/40/38 (TP/FN/TN/FP)", sel.groupby("group").cluster.nunique().to_dict() == {"FN": 40, "FP": 38, "TN": 40, "TP": 38}, str(sel.groupby("group").cluster.nunique().to_dict()))
a = audit.set_index("window_id").loc[sel.window_id]
check("every selected window: >=3 present, ESS>=100, aggregate R2 pool>=4", bool(a.nnz_ge3.all() and (a.ess >= 100).all() and (a.pool_r2_agg >= 4).all()))
check("all selected windows are from the test split", set(sel.split) == {"test"})
check("selected-window manifest hash equals freeze manifest", hashlib.sha256((R / "selected_windows.csv").read_bytes()).hexdigest() == fm["selected_windows_sha256"])
check("protocol hash equals freeze manifest", hashlib.sha256((ROOT / "docs/phase2_protocol.yaml").read_bytes()).hexdigest() == fm["protocol_sha256"])
check("eligibility_report final_selected = 40 each", (el.final_selected == 40).all())
check("protocol: b=3, k=5, bootstrap>=1000, cap=2", "budget_b: 3" in proto_txt and "k: 5" in proto_txt and "bootstrap_replicates: 2000" in proto_txt and "global_cluster_cap: 2" in proto_txt)
check("all stability windows = selected windows", set(S.window_id) == set(sel.window_id) and len(S) == 160)

# ---- 2. macro-averages are equal-weight over groups
def macro(df, col):
    return df.groupby("group")[col].mean().mean()


for col, qprefix in [("single_jaccard", "single-run: pairwise top-5 Jaccard"), ("rep_jaccard", "20-run aggregate: Jaccard between"),
                     ("cm_jaccard", "compute-matched single run (100k): Jaccard"), ("single_spearman", "single-run: pairwise full-ranking Spearman"),
                     ("rep_spearman", "20-run aggregate: Spearman"), ("cm_spearman", "compute-matched single run: Spearman")]:
    r = st[st.quantity.str.startswith(qprefix) & st.group.str.startswith("ALL")].iloc[0]
    check(f"macro (equal-weight over groups) reproduces summary: {col}", close(macro(S, col), r.estimate, 1e-9), f"{macro(S, col):.4f}")
# ---- 3. headline stability values vs docs
def srow(prefix):
    return st[st.quantity.str.startswith(prefix) & st.group.str.startswith("ALL")].iloc[0]


claims = [("single-run: pairwise top-5 Jaccard", 0.775, 0.755, 0.795), ("20-run aggregate: Jaccard between", 0.923, 0.902, 0.944),
          ("compute-matched single run (100k): Jaccard", 0.994, 0.986, 1.000), ("single-run: pairwise full-ranking Spearman", 0.806, 0.801, 0.811),
          ("20-run aggregate: Spearman", 0.977, 0.976, 0.978), ("compute-matched single run: Spearman", 0.961, 0.959, 0.963),
          ("PAIRED difference: aggregate-replicate Jaccard", 0.148, 0.125, 0.172), ("PAIRED difference: aggregate-replicate Spearman", 0.171, 0.166, 0.176),
          ("PAIRED difference: aggregate Jaccard minus compute-matched", -0.071, -0.093, -0.048)]
for pref, e, lo, hi in claims:
    r = srow(pref)
    check(f"stability doc claim: {pref[:60]}", close(r.estimate, e, 6e-4) and close(r.ci_low, lo, 6e-4) and close(r.ci_high, hi, 6e-4),
          f"csv {r.estimate:.4f} [{r.ci_low:.4f},{r.ci_high:.4f}] vs doc {e} [{lo},{hi}]")
check("chance baselines 0.019 / 0.494", close(srow("chance baseline: random top-5 of 150").estimate, 0.019, 6e-4) and close(srow("chance baseline: random top-5 among").estimate, 0.494, 6e-4))
check("unrestricted aggregate top-5 present share 13.6%", close(srow("share of unrestricted").estimate, 0.136, 6e-4))
check("surrogate R2 0.385 / 0.442", close(srow("surrogate weighted R2, single").estimate, 0.385, 6e-4) and close(srow("surrogate weighted R2, compute").estimate, 0.442, 6e-4))
check("ESS 4,153 (5k) / 92,560 (100k)", close(srow("effective sample size, 5k").estimate, 4153, 1) and close(srow("effective sample size, 100k").estimate, 92560, 1))
check("present-only Jaccard 0.823 vs 0.948", close(srow("single-run Jaccard, present").estimate, 0.823, 6e-4) and close(srow("aggregate replicate Jaccard, present").estimate, 0.948, 6e-4))
g = S.groupby("group")[["single_jaccard", "rep_jaccard"]].mean().round(2)
check("by-group Jaccard single/aggregate: FN .72/.90 FP .81/.91 TN .71/.90 TP .86/.98",
      g.loc["FN"].tolist() == [0.72, 0.90] and g.loc["FP"].tolist() == [0.81, 0.91] and g.loc["TN"].tolist() == [0.71, 0.90] and g.loc["TP"].tolist() == [0.86, 0.98], str(g.to_dict("index")))
for c in ("single_sign_consistency", "rep_sign_consistency", "cm_sign_consistency"):
    check(f"top-5 sign consistency == 1.0 for every window ({c}); saturated, no NaN", bool((S[c] == 1).all()))

# ---- 4. perturbation comparisons
def prow(outcome, comp, group="ALL"):
    d = pc[(pc.outcome == outcome) & (pc.comparison == comp) & pc.group.str.startswith(group)]
    return d.iloc[0]


pcl = [("single guided - single R2", 0.821, 0.602, 1.037, 0.003), ("agg guided - agg R2", 0.805, 0.609, 1.007, 0.003), ("agg guided - single guided", -0.051, -0.150, 0.054, 0.353)]
for comp, e, lo, hi, ph in pcl:
    r = prow("direction_aware_drop", comp)
    ok = close(r.estimate, e, 6e-4) and close(r.ci_low, lo, 6e-4) and close(r.ci_high, hi, 6e-4) and close(r.p_holm_primary_family, ph, 6e-4)
    check(f"perturbation doc claim: {comp}", ok, f"csv {r.estimate:.4f} [{r.ci_low:.4f},{r.ci_high:.4f}] holm {r.p_holm_primary_family:.4f}")
r = prow("direction_aware_drop", "(agg guided - agg R2) - (single guided - single R2)  [difference in effect over control]")
check("difference in effect over control -0.021 [-0.148,+0.095]", close(r.estimate, -0.021, 6e-4) and close(r.ci_low, -0.148, 6e-4) and close(r.ci_high, 0.095, 6e-4))
r = prow("direction_aware_drop", "cm guided - agg guided [secondary]"); check("cm guided - agg guided -0.101 [-0.267,+0.065]", close(r.estimate, -0.101, 6e-4) and close(r.ci_low, -0.267, 6e-4) and close(r.ci_high, 0.065, 6e-4))
r = prow("direction_aware_drop", "cm guided - cm R2 [secondary]"); check("cm guided - cm R2 +0.520 [+0.293,+0.728]", close(r.estimate, 0.520, 6e-4) and close(r.ci_low, 0.293, 6e-4) and close(r.ci_high, 0.728, 6e-4))
r = prow("flip_attack_to_normal", "single guided - single R2"); r2 = prow("flip_attack_to_normal", "agg guided - agg R2"); r3 = prow("flip_attack_to_normal", "agg guided - single guided")
check("macro flip (zeros for predicted-normal groups): +0.045 / +0.047 / 0.000", close(r.estimate, 0.045, 6e-4) and close(r2.estimate, 0.047, 6e-4) and close(r3.estimate, 0.0, 1e-12))
by = {g_: prow("direction_aware_drop", "agg guided - agg R2", g_).estimate for g_ in ["TP", "TN", "FP", "FN"]}
check("agg guided-R2 by group TP +1.26 TN +1.14 FP +0.74 FN +0.09", close(by["TP"], 1.26, 6e-3) and close(by["TN"], 1.14, 6e-3) and close(by["FP"], 0.74, 6e-3) and close(by["FN"], 0.09, 6e-3), str({k: round(v, 3) for k, v in by.items()}))
r = [prow("direction_aware_drop", c).estimate for c in ("agg guided - agg R1 [secondary R1]", "single guided - single R1 [secondary R1]")]
check("guided beats R1 by about 1.6 logit units (agg/single)", all(1.5 <= x <= 1.7 for x in r), str([round(x, 3) for x in r]))
# Holm recomputation
fam = {c: prow("direction_aware_drop", c).p_boot for c, *_ in pcl}
hol = ms.holm(fam)
check("Holm-adjusted p recomputed from bootstrap p-values", all(close(hol[c], prow("direction_aware_drop", c).p_holm_primary_family, 1e-9) for c in fam), str({k: round(v, 4) for k, v in hol.items()}))

# ---- 5. independent recomputation of paired macro estimates from the per-window CSV
def wide(col):
    gd = P[P.arm == "guided"].pivot(index="window_id", columns="method", values=col)
    r2 = P[P.arm == "r2"].pivot(index="window_id", columns="method", values=col)
    meta = P.drop_duplicates("window_id").set_index("window_id")[["group", "cluster"]]
    return meta.join(gd.add_prefix("g_")).join(r2.add_prefix("r_"))


def boot(d, col, B=2000, seed=777):
    rng = np.random.default_rng(seed)
    cl = d.cluster.to_numpy(); u, inv = np.unique(cl, return_inverse=True); v = d[col].to_numpy(); gr = d.group.to_numpy(); out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv]
        ms_ = [(v[gr == k] * w[gr == k]).sum() / w[gr == k].sum() for k in np.unique(gr) if w[gr == k].sum() > 0]
        if len(ms_) == len(np.unique(gr)): out.append(np.mean(ms_))
    return np.percentile(out, [2.5, 97.5])


W = wide("direction_aware_drop")
for name, f, comp in [("single guided-R2", lambda d: d.g_single - d.r_single, "single guided - single R2"), ("agg guided-R2", lambda d: d.g_agg - d.r_agg, "agg guided - agg R2"),
                      ("agg-single guided", lambda d: d.g_agg - d.g_single, "agg guided - single guided")]:
    d = W.copy(); d["v"] = f(d); d = d.dropna(subset=["v"])
    est = d.groupby("group").v.mean().mean(); lo, hi = boot(d, "v"); r = prow("direction_aware_drop", comp)
    check(f"independent recompute: {name}", close(est, r.estimate, 1e-9) and close(lo, r.ci_low, 0.03) and close(hi, r.ci_high, 0.03), f"est {est:.4f} CI [{lo:.3f},{hi:.3f}] vs saved [{r.ci_low:.3f},{r.ci_high:.3f}] (different bootstrap seed)")
# ---- 6. exclusions, edits, runtimes
e_all = ex[ex.group == "ALL"].set_index("method")
check("exclusions: single 1 infeasible+15 uninformative=16 (10.0%); cm 2+18=20 (12.5%); agg 0",
      (e_all.loc["single", "guided_infeasible_pool_lt3"], e_all.loc["single", "R2_uninformative_pool_eq3"], e_all.loc["cm", "guided_infeasible_pool_lt3"], e_all.loc["cm", "R2_uninformative_pool_eq3"]) == (1, 15, 2, 18)
      and close(e_all.loc["single", "pct_excluded_from_primary_R2_comparison"], 10.0, 0.05) and close(e_all.loc["cm", "pct_excluded_from_primary_R2_comparison"], 12.5, 0.05) and e_all.loc["agg", "pct_excluded_from_primary_R2_comparison"] == 0)
check("edit success ~79-80% (guided)", ed.edit_success_rate.between(0.78, 0.81).all(), str(ed.edit_success_rate.round(3).tolist()))
check("all edit failures are 'occurrence_consumed_by_prior_edit'", set(re.findall(r":(\w+)", " ".join(P.failure_reasons.dropna().astype(str)))) <= {"occurrence_consumed_by_prior_edit", "absent_in_original_window"} and "absent_in_original_window" not in " ".join(P.failure_reasons.dropna().astype(str)))
rtv = rt.set_index("condition")
vals = rtv.per_window_s.round(3).tolist()
check("runtime per window: 0.118 / 2.245 / 1.169 / 4.658 / 1.181; total 9.37", vals[:5] == [0.118, 2.245, 1.169, 4.658, 1.181] and close(rt.per_window_s.iloc[-1], 9.37, 6e-3), str(vals))
check("runtime ratios 19.1x aggregate, 9.9x compute-matched; 505,000 model calls", close(rt.ratio_vs_single_run.iloc[1], 19.1, 0.06) and close(rt.ratio_vs_single_run.iloc[2], 9.9, 0.06) and int(rt.model_calls_per_window.iloc[-1]) == 505000)
# ---- 7. docs agree with each other
res = (ROOT / "docs/phase2_results.md").read_text(); gap = (ROOT / "docs/phase2_research_gap.md").read_text()
for tok in ["0.775", "0.923", "+0.148", "+0.125 to +0.172", "0.806", "0.977", "+0.821", "+0.805", "-0.051", "-0.150 to +0.054", "0.994"]:
    check(f"research-gap value '{tok}' also in results doc", tok in res and tok in gap.replace("0.8 logit", "") or (tok in gap and tok in res), "")
check("research-gap states 38 FP clusters and 19x cost", "38" in gap and "19x" in gap)
# ---- 9. supplementary / audit-reanalysis claims in the results document
cf = pd.read_csv(R / "supplementary_conditional_flip.csv")
def cfr(method, quantity, scope):
    return cf[(cf.method == method) & (cf.quantity == quantity) & (cf.scope == scope)].iloc[0]
for m, flips, n, rate, ctrl, lo, hi, tp, fp, pooled, plo, phi in [("single", 72, 77, 0.935, 0.844, 0.049, 0.133, 0.116, 0.065, 0.091, 0.049, 0.133),
                                                              ("agg", 75, 80, 0.938, 0.843, 0.054, 0.134, 0.126, 0.062, 0.094, 0.054, 0.134),
                                                              ("cm", 65, 71, 0.915, 0.854, 0.019, 0.105, 0.074, 0.047, 0.061, 0.019, 0.105)]:
    g_ = cfr(m, "guided flip rate", "TP+FP pooled by window"); c_ = cfr(m, "R2 matched-random flip rate", "TP+FP pooled by window")
    d_ = cfr(m, "guided minus R2", "TP+FP pooled by window")
    ok = (int(g_.guided_flip_windows) == flips and int(g_.n_windows) == n and close(g_.estimate, rate, 6e-4) and close(c_.estimate, ctrl, 6e-4)
          and close(d_.estimate, pooled, 6e-4) and close(d_.ci_low, plo, 6e-4) and close(d_.ci_high, phi, 6e-4)
          and close(cfr(m, "guided minus R2", "TP").estimate, tp, 6e-4) and close(cfr(m, "guided minus R2", "FP").estimate, fp, 6e-4))
    check(f"conditional flip table ({m}): {flips}/{n} guided flips, rate, R2 rate, guided-minus-R2 TP/FP/pooled + CI", ok, f"pooled diff {d_.estimate:.4f} [{d_.ci_low:.4f},{d_.ci_high:.4f}]")
z = cf[(cf.method == "agg vs single") & (cf.scope == "TP+FP pooled by window")].iloc[0]
check("conditional flip, aggregate minus single guided = 0.000 [0,0], 80 windows", z.estimate == 0 and z.ci_low == 0 and z.ci_high == 0 and int(z.n_windows) == 80)
# denominators: conditional table must only contain originally-predicted-attack windows
pa = set(P[P.pred_attack == 1].window_id)
check("conditional flip denominators contain only TP/FP (predicted-attack) windows", set(P[P.pred_attack == 1].group) == {"TP", "FP"} and len(pa) == 80)
check("predicted-normal windows (TN, FN) have flip_attack_to_normal == 0 in every arm", bool((P[P.pred_attack == 0].flip_attack_to_normal == 0).all()))
ss = pd.read_csv(R / "supplementary_signed_spearman_summary.csv").set_index("quantity")
exp = {"single run: pairwise signed-vector Spearman": (0.873, 0.872, 0.875), "20-run aggregate replicates: signed-vector Spearman": (0.989, 0.989, 0.989),
       "compute-matched 1x100k: signed-vector Spearman": (0.984, 0.983, 0.984), "paired: aggregate minus single (signed)": (0.116, 0.114, 0.117),
       "paired: aggregate minus compute-matched (signed)": (0.005, 0.004, 0.006),
       "paired: aggregate minus compute-matched (|w| ranking, predeclared metric)": (0.016, 0.014, 0.018)}
for q, (e, lo, hi) in exp.items():
    r = ss.loc[q]; check(f"supplementary Spearman: {q}", close(r.estimate, e, 6e-4) and close(r.ci_low, lo, 1.1e-3) and close(r.ci_high, hi, 1.1e-3), f"{r.estimate:.4f} [{r.ci_low:.4f},{r.ci_high:.4f}]")
W_ = pd.read_csv(R / "supplementary_signed_spearman_per_window.csv")
check("cached coefficients reproduce saved |w| Spearman (max abs diff <= 1e-12)", max(np.abs(W_[c + "_recomputed"] - W_[c + "_saved"]).max() for c in ("single_abs", "rep_abs", "cm_abs")) <= 1e-12)
gd = P[P.arm == "guided"].set_index("window_id"); ga, gs = gd[gd.method == "agg"], gd[gd.method == "single"]
jj = ga[["guided_features", "direction_aware_drop"]].join(gs[["guided_features", "direction_aware_drop"]], rsuffix="_s", how="inner")
same_set = np.array([set(x.split(";")) == set(y.split(";")) for x, y in zip(jj.guided_features, jj.guided_features_s)]); same_ord = (jj.guided_features == jj.guided_features_s).to_numpy()
dd = (jj.direction_aware_drop - jj.direction_aware_drop_s).to_numpy()
check("guided top-3: identical set in 132/159 (83.0%), identical ordered list 89, same-set-different-order 43 (16 with different outcome), different set 27",
      (len(jj), same_set.sum(), same_ord.sum(), (same_set & ~same_ord).sum(), int((np.abs(dd[same_set & ~same_ord]) > 1e-9).sum()), (~same_set).sum()) == (159, 132, 89, 43, 16, 27), "")
check("identical ordered guided lists give identical outcomes (max |diff| = 0)", float(np.abs(dd[same_ord]).max()) == 0.0)
check("different-set windows: mean aggregate-minus-single = -0.197", close(dd[~same_set].mean(), -0.197, 6e-4), f"{dd[~same_set].mean():.4f}")

# ---- 8. optional: cluster IDs vs raw data (needs the external dataset; SKIP otherwise)
raw = ROOT / "data/raw/ADFA-LD"
if raw.exists():
    from pathlib import Path as _P
    from src import phase3a
    from src.config import load_config
    cfg = load_config("configs/baseline.yaml")
    manifest, _, _ = phase3a.load_frozen_manifest(_P(ROOT / "data/processed/trace_manifest.csv"), cfg)
    calls, _root = phase3a._verify_manifest_against_raw(manifest, cfg)       # Phase 1 verifier: IDs, labels, lengths
    cid = {t: "C" + hashlib.sha256(np.asarray(c, dtype=np.int64).tobytes()).hexdigest()[:12] for t, c in calls.items()}
    check("manifest verified against raw dataset (Phase 1 verifier)", True)
    check("selected cluster IDs equal sha256 of raw syscall sequences", all(cid[t] == c for t, c in zip(sel.trace_id, sel.cluster)))
    # Independent check that a cluster is a CONTENT equivalence class: equal clusters <=> equal call sequences
    bytes_by_cluster = {}
    for t, c in zip(sel.trace_id, sel.cluster):
        bytes_by_cluster.setdefault(c, set()).add(np.asarray(calls[t], dtype=np.int64).tobytes())
    check("every selected cluster maps to exactly one distinct syscall sequence", all(len(v) == 1 for v in bytes_by_cluster.values()))
else:
    check("cluster IDs recomputed from raw traces", None, "raw ADFA-LD absent in this checkout (SKIP, not PASS)")

df = pd.DataFrame(rows, columns=["check", "result", "detail"])
if "--write" in sys.argv:                      # opt-in, so routine runs/tests never dirty the repository
    df.to_csv(R / "audit_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 200)
print(df.to_string(index=False)); print("\nSUMMARY:", df.result.value_counts().to_dict())
sys.exit(1 if (df.result == "FAIL").any() else 0)
