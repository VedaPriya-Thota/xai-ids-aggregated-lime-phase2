"""Prints the headline before/after numbers of Phase 2 from the saved result files (read-only, runs in about 1 second)."""
import pandas as pd

R = "results/phase2/"
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)


def line(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


s = pd.read_csv(R + "stability_summary.csv")
s = s[s.group.str.startswith("ALL")].set_index("quantity")
line("1. STABILITY across repeated LIME runs (higher = more stable)")
for label, q in [
    ("BEFORE  single run (5,000 samples)   top-5 Jaccard", "single-run: pairwise top-5 Jaccard (20 repeated runs)"),
    ("AFTER   20-run aggregate             top-5 Jaccard", "20-run aggregate: Jaccard between two independent aggregates"),
    ("         paired difference (after - before)        ", "PAIRED difference: aggregate-replicate Jaccard minus single-run Jaccard"),
    ("BEFORE  single run                   Spearman     ", "single-run: pairwise full-ranking Spearman"),
    ("AFTER   20-run aggregate             Spearman     ", "20-run aggregate: Spearman between two independent aggregates"),
    ("reference: one 100,000-sample run    top-5 Jaccard", "compute-matched single run (100k): Jaccard between two runs"),
]:
    r = s.loc[q]
    print(f"{label}: {r.estimate:+.3f}  95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]")

p = pd.read_csv(R + "paired_comparisons.csv")
p = p[(p.outcome == "direction_aware_drop") & p.group.astype(str).str.startswith("ALL")].set_index("comparison")
line("2. PERTURBATION: logit drop (positive = the edit moved the model away from its original decision)")
for label, q in [
    ("single-run guided  minus matched random", "single guided - single R2"),
    ("aggregate guided   minus matched random", "agg guided - agg R2"),
    ("aggregate guided   minus single guided (THE IMPROVEMENT)", "agg guided - single guided"),
]:
    r = p.loc[q]
    print(f"{label:58s}: {r.estimate:+.3f}  95% CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  n={int(r.n_windows)}  Holm p={r.p_holm_primary_family:.3f}")

t = pd.read_csv(R + "runtime_summary.csv").iloc[:3]
line("3. COST per window")
for _, r in t.iterrows():
    print(f"{r.condition:48s} {r.per_window_s:6.3f} s   {int(r.model_calls_per_window):>7,} model calls   {r.ratio_vs_single_run:5.1f}x")
print()
