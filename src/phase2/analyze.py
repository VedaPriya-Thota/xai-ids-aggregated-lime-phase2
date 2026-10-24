"""Tables and figures from the saved per-window results (no recomputation of any experiment)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.phase2 import metrics_stability as ms

GROUPS = ["TP", "FN", "TN", "FP"]
BOOT_SEED = 20261023


def _summ(df: pd.DataFrame, col: str, name: str, n_boot: int, seed: int) -> list[dict]:
    rows = []
    d = df.dropna(subset=[col])
    if len(d):
        r = ms.cluster_bootstrap(d[col].to_numpy(), d.cluster.to_numpy(), n_boot=n_boot, seed=seed, groups=d.group.to_numpy())
        rows.append({"quantity": name, "group": "ALL (macro, equal-weight groups)", **r})
    for g in GROUPS:
        s = d[d.group == g]
        if len(s):
            rows.append({"quantity": name, "group": g, **ms.cluster_bootstrap(s[col].to_numpy(), s.cluster.to_numpy(), n_boot=n_boot, seed=seed)})
    return rows


def stability_tables(S: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows, seed = [], BOOT_SEED
    S = S.copy()
    S["d_jaccard_agg_minus_single"] = S.rep_jaccard - S.single_jaccard
    S["d_spearman_agg_minus_single"] = S.rep_spearman - S.single_spearman
    S["d_jaccard_cm_minus_single"] = S.cm_jaccard - S.single_jaccard
    S["d_jaccard_agg_minus_cm"] = S.rep_jaccard - S.cm_jaccard
    for col, name in [("single_jaccard", "single-run: pairwise top-5 Jaccard (20 repeated runs)"),
                      ("single_spearman", "single-run: pairwise full-ranking Spearman"),
                      ("single_sign_consistency", "single-run: top-5 sign consistency"),
                      ("rep_jaccard", "20-run aggregate: Jaccard between two independent aggregates"),
                      ("rep_spearman", "20-run aggregate: Spearman between two independent aggregates"),
                      ("rep_sign_consistency", "20-run aggregate: top-5 sign consistency between replicates"),
                      ("cm_jaccard", "compute-matched single run (100k): Jaccard between two runs"),
                      ("cm_spearman", "compute-matched single run: Spearman between two runs"),
                      ("cm_sign_consistency", "compute-matched single run: top-5 sign consistency"),
                      ("single_jaccard_present", "single-run Jaccard, present features only (secondary)"),
                      ("rep_jaccard_present", "aggregate replicate Jaccard, present features only (secondary)"),
                      ("agg_sign_agreement_with_runs", "aggregate top-5: share of the 20 runs agreeing in sign"),
                      ("d_jaccard_agg_minus_single", "PAIRED difference: aggregate-replicate Jaccard minus single-run Jaccard"),
                      ("d_spearman_agg_minus_single", "PAIRED difference: aggregate-replicate Spearman minus single-run Spearman"),
                      ("d_jaccard_cm_minus_single", "PAIRED difference: compute-matched Jaccard minus single-run Jaccard (secondary)"),
                      ("d_jaccard_agg_minus_cm", "PAIRED difference: aggregate Jaccard minus compute-matched Jaccard (secondary)"),
                      ("chance_jaccard_150", "chance baseline: random top-5 of 150 features"),
                      ("chance_jaccard_present", "chance baseline: random top-5 among the window's present features"),
                      ("top5_present_share_agg", "share of unrestricted aggregate top-5 that are present features"),
                      ("r2_single_runs_mean", "surrogate weighted R2, single runs (mean of 20)"), ("r2_cm", "surrogate weighted R2, compute-matched"),
                      ("ess_agg_run0", "effective sample size, 5k run"), ("ess_cm", "effective sample size, 100k run")]:
        if col in S:
            rows += _summ(S, col, name, n_boot, seed)
            seed += 1
    return pd.DataFrame(rows)


def runtime_table(S: pd.DataFrame) -> pd.DataFrame:
    c = lambda x: float(S[x].sum())
    parts = {"single run (1 x 5k) [primary baseline]": c("runtime_single_s"),
             "20-run aggregate (20 x 5k) [improvement]": c("runtime_agg_s"),
             "compute-matched single run (1 x 100k)": c("runtime_cm_primary_s"),
             "extra: 2 independent aggregate replicates (40 x 5k)": c("runtime_rep_a_s") + c("runtime_rep_b_s"),
             "extra: second compute-matched run for its stability (1 x 100k)": c("runtime_cm_stability_extra_s")}
    n = len(S)
    rows = [{"condition": k, "total_s": v, "per_window_s": v / n, "model_calls_per_window": m}
            for (k, v), m in zip(parts.items(), [S.model_calls_single.iloc[0], S.model_calls_agg.iloc[0], S.model_calls_cm_primary.iloc[0],
                                                S.model_calls_replicates_extra.iloc[0], S.model_calls_cm_primary.iloc[0]])]
    base = parts["single run (1 x 5k) [primary baseline]"]
    for r in rows:
        r["ratio_vs_single_run"] = r["total_s"] / base if base else np.nan
    rows.append({"condition": "all LIME computation for the full design", "total_s": sum(parts.values()), "per_window_s": sum(parts.values()) / n,
                 "model_calls_per_window": int(sum(r["model_calls_per_window"] for r in rows)), "ratio_vs_single_run": sum(parts.values()) / base if base else np.nan})
    return pd.DataFrame(rows)


def _wide(P: pd.DataFrame, col: str) -> pd.DataFrame:
    g = P[P.arm == "guided"].pivot(index="window_id", columns="method", values=col).add_prefix("guided_")
    r2 = P[P.arm == "r2"].pivot(index="window_id", columns="method", values=col).add_prefix("r2_")
    r1 = P[P.arm == "r1"].pivot(index="window_id", columns="method", values=col).add_prefix("r1_")
    meta = P.drop_duplicates("window_id").set_index("window_id")[["group", "cluster"]]
    return meta.join(g).join(r2).join(r1)


def perturbation_tables(P: pd.DataFrame, n_boot: int, sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summ_rows, cmp_rows, excl = [], [], []
    seed = BOOT_SEED + 1000
    for col in ["direction_aware_drop", "attack_logit_drop", "prob_change", "flip_attack_to_normal", "flip_any"]:
        W = _wide(P, col).reset_index()
        for m in ["single", "agg", "cm"]:
            ok2 = W[[f"guided_{m}", f"r2_{m}"]].notna().all(axis=1)
            for arm_name, c in (("guided", f"guided_{m}"), ("R2 matched random (mean of draws)", f"r2_{m}")):
                d = W[ok2].assign(v=W[c])
                for r in _summ(d, "v", f"{col} | {m} | {arm_name}", n_boot, seed):
                    summ_rows.append({"outcome": col, "method": m, "arm": arm_name, **r}); seed += 1
        comps = [("single guided - single R2", lambda W: W.guided_single - W.r2_single, ["guided_single", "r2_single"], True),
                 ("agg guided - agg R2", lambda W: W.guided_agg - W.r2_agg, ["guided_agg", "r2_agg"], True),
                 ("agg guided - single guided", lambda W: W.guided_agg - W.guided_single, ["guided_agg", "guided_single"], True),
                 ("(agg guided - agg R2) - (single guided - single R2)  [difference in effect over control]",
                  lambda W: (W.guided_agg - W.r2_agg) - (W.guided_single - W.r2_single), ["guided_agg", "r2_agg", "guided_single", "r2_single"], True),
                 ("cm guided - agg guided [secondary]", lambda W: W.guided_cm - W.guided_agg, ["guided_cm", "guided_agg"], False),
                 ("cm guided - cm R2 [secondary]", lambda W: W.guided_cm - W.r2_cm, ["guided_cm", "r2_cm"], False),
                 ("single guided - single R1 [secondary R1]", lambda W: W.guided_single - W.r1_single, ["guided_single", "r1_single"], False),
                 ("agg guided - agg R1 [secondary R1]", lambda W: W.guided_agg - W.r1_agg, ["guided_agg", "r1_agg"], False)]
        for name, fn, need, primary in comps:
            if not all(c in W for c in need):
                continue
            d = W.dropna(subset=need).copy()
            d["v"] = fn(d)
            for r in _summ(d, "v", name, n_boot, seed):
                sd = d.v.std(ddof=1) if r["group"].startswith("ALL") and len(d) > 1 else np.nan
                cmp_rows.append({"outcome": col, "comparison": name, "primary_family": primary and col == "direction_aware_drop", **r,
                                 "cohens_dz_window_level": float(d.v.mean() / sd) if sd and sd > 0 else np.nan}); seed += 1
    for m in ["single", "agg", "cm"]:
        for g in GROUPS + ["ALL"]:
            tot = sel if g == "ALL" else sel[sel.group == g]
            s = P[(P.method == m) & P.window_id.isin(tot.window_id)].drop_duplicates("window_id")
            n = len(tot)
            if n:
                # a window whose pool has < 3 eligible features produces no perturbation rows at all (never padded)
                infeasible = n - len(s)
                uninf = int((s.status == "R2_uninformative").sum())
                excl.append({"method": m, "group": g, "windows_in_frozen_sample": n, "guided_infeasible_pool_lt3": int(infeasible),
                             "R2_uninformative_pool_eq3": uninf, "informative_R2_pool_ge4": int((s.status == "ok").sum()),
                             "pct_guided_infeasible": 100 * infeasible / n, "pct_R2_uninformative": 100 * uninf / n,
                             "pct_excluded_from_primary_R2_comparison": 100 * (infeasible + uninf) / n})
    return pd.DataFrame(summ_rows), pd.DataFrame(cmp_rows), pd.DataFrame(excl)


def edit_diagnostics(P: pd.DataFrame) -> pd.DataFrame:
    g = P[P.arm == "guided"]
    return g.groupby("method").agg(windows=("window_id", "count"), edit_success_rate=("edit_success", "mean"),
                                   mean_features_edited=("n_features_edited", "mean"), mean_positions_edited=("n_positions_edited", "mean"),
                                   windows_with_failures=("failure_reasons", lambda s: int((s.fillna("") != "").sum()))).reset_index()


def figures(root: Path, S: pd.DataFrame, P: pd.DataFrame, rt: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = root / "figures"; fd.mkdir(exist_ok=True)
    out = []
    # 1. Jaccard distributions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos, data, labels = 0, [], []
    for g in GROUPS:
        d = S[S.group == g]
        for col, lab in (("single_jaccard", "single"), ("rep_jaccard", "20-run agg."), ("cm_jaccard", "compute-matched")):
            data.append(d[col].dropna().to_numpy()); labels.append(f"{g}\n{lab}")
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    cols = ["#9ecae1", "#fdae6b", "#a1d99b"] * 4
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("top-5 Jaccard (per window)"); ax.set_title("Explanation stability: single-run vs 20-run aggregate vs compute-matched")
    ax.axhline(S.chance_jaccard_150.mean(), ls=":", c="grey"); ax.text(0.6, S.chance_jaccard_150.mean() + 0.01, "chance (150 features)", fontsize=7, color="grey")
    fig.tight_layout(); fig.savefig(fd / "fig1_top5_jaccard.png", dpi=160); plt.close(fig); out.append("fig1_top5_jaccard.png")
    # 2/3. guided vs matched random
    for fig_name, col, ylab, ttl in (("fig2_guided_vs_random_logit_drop.png", "direction_aware_drop", "direction-aware attack-logit drop\n(positive = moved away from original decision)", "Guided vs matched-random (R2) raw-syscall edits"),
                                     ("fig3_guided_vs_random_flip_rate.png", "flip_attack_to_normal", "attack-to-normal flip rate\n(over predicted-attack windows)", "Attack-to-normal flips: guided vs matched random (R2)")):
        W = _wide(P, col).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, m in zip(axes, ["single", "agg"]):
            xs, w = np.arange(len(GROUPS)), 0.38
            gv, rv = [], []
            for g in GROUPS:
                d = W[(W.group == g)].dropna(subset=[f"guided_{m}", f"r2_{m}"])
                if col == "flip_attack_to_normal":
                    d = d[d.group.isin(["TP", "FP"])]
                gv.append(d[f"guided_{m}"].mean() if len(d) else np.nan); rv.append(d[f"r2_{m}"].mean() if len(d) else np.nan)
            ax.bar(xs - w / 2, gv, w, label="guided top-3", color="#fd8d3c"); ax.bar(xs + w / 2, rv, w, label="R2 random (mean of draws)", color="#9e9ac8")
            ax.set_xticks(xs); ax.set_xticklabels(GROUPS); ax.set_title({"single": "single-run LIME", "agg": "20-run aggregated LIME"}[m]); ax.axhline(0, c="k", lw=0.5)
        axes[0].set_ylabel(ylab, fontsize=8); axes[0].legend(fontsize=7); fig.suptitle(ttl, fontsize=10)
        fig.tight_layout(); fig.savefig(fd / fig_name, dpi=160); plt.close(fig); out.append(fig_name)
    # 4. runtime
    fig, ax = plt.subplots(figsize=(8, 3.8))
    r = rt.iloc[:-1]
    ax.barh(range(len(r)), r.per_window_s, color="#6baed6"); ax.set_yticks(range(len(r))); ax.set_yticklabels([c[:55] for c in r.condition], fontsize=7)
    ax.set_xlabel("measured seconds per window (single core)"); ax.set_title("LIME computation cost per window"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(fd / "fig4_runtime.png", dpi=160); plt.close(fig); out.append("fig4_runtime.png")
    # 5. replicate stability vs single-run stability
    fig, ax = plt.subplots(figsize=(5, 5))
    for g, c in zip(GROUPS, ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]):
        d = S[S.group == g]; ax.scatter(d.single_jaccard, d.rep_jaccard, s=14, label=g, c=c, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("single-run top-5 Jaccard (pairwise)"); ax.set_ylabel("20-run aggregate replicate Jaccard"); ax.legend(fontsize=7)
    ax.set_title("Per-window stability: aggregate vs single run", fontsize=9)
    fig.tight_layout(); fig.savefig(fd / "fig5_replicate_stability.png", dpi=160); plt.close(fig); out.append("fig5_replicate_stability.png")
    return out


def run(root="results/phase2") -> None:
    root = Path(root) if Path(root).is_absolute() else PROJECT_ROOT / root
    n_boot = yaml.safe_load((PROJECT_ROOT / "docs/phase2_protocol.yaml").read_text())["statistics"]["bootstrap_replicates"]
    assert n_boot >= 1000
    S, P = pd.read_csv(root / "stability_results.csv"), pd.read_csv(root / "perturbation_results.csv")
    st = stability_tables(S, n_boot); st.to_csv(root / "stability_summary.csv", index=False)
    rt = runtime_table(S); rt.to_csv(root / "runtime_summary.csv", index=False)
    sel = pd.read_csv(root / "selected_windows.csv")
    ps, pc, ex = perturbation_tables(P, n_boot, sel)
    ps.to_csv(root / "perturbation_summary.csv", index=False)
    primary = pc[(pc.primary_family) & pc.group.str.startswith("ALL") & (pc.comparison != pc.comparison.iloc[0] + "x")]
    fam = primary[primary.comparison.isin(["single guided - single R2", "agg guided - agg R2", "agg guided - single guided"])]
    adj = ms.holm({r.comparison: r.p_boot for r in fam.itertuples()}) if len(fam) else {}
    pc["p_holm_primary_family"] = [adj.get(c) if (pf and g.startswith("ALL")) else np.nan for c, pf, g in zip(pc.comparison, pc.primary_family, pc.group)]
    pc.to_csv(root / "paired_comparisons.csv", index=False)
    ex.to_csv(root / "exclusions.csv", index=False)
    ed = edit_diagnostics(P); ed.to_csv(root / "edit_diagnostics.csv", index=False)
    figs = figures(root, S, P, rt)
    print(f"[analyze] wrote summaries and {len(figs)} figures to {root}")
