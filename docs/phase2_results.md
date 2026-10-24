# Phase 2 results: 20-run aggregated LIME vs single-run LIME (held-out test split, run once)

All numbers are copied from `results/phase2/*.csv` produced by `python -m src.phase2.run_phase2 final` followed by `analyze`.
Protocol and the selected-window manifest were frozen before any perturbation outcome was computed (`results/phase2/freeze_manifest.json`;
protocol SHA-256 `d99a3f57…7017`, manifest SHA-256 `44388056…9778`). Confidence intervals are 95% percentile intervals from a
2,000-replicate bootstrap over identical-content clusters; the headline estimate is the equal-weight macro-average of the four outcome groups.
Interpretation is separated from measurement below. Nothing here shows that explanations are *faithful* to the model.

## 1. Sample (measured)
| Group | Test windows | Selected | Traces | Independent content clusters |
|---|---:|---:|---:|---:|
| TP | 1,195 | 40 | 38 | 38 |
| FN | 428 | 40 | 40 | 40 |
| TN | 11,681 | 40 | 40 | 40 |
| FP | 432 | 40 | 38 | **38** |
All 160 windows have ESS >= 100, >= 4 present, actionable, sign-supporting features under the 20-run aggregate, and obey the global two-per-cluster cap
(verified programmatically). The full staged audit is `results/phase2/eligibility_report.csv`; FP/FN were scanned exhaustively, TP/TN lazily
(their R2-eligible counts cover only the windows examined). 152 distinct clusters cover the 160 windows.

## 2. Stability (measured; k = 5, 150-feature ranking)
| Quantity (macro over groups) | Estimate | 95% CI |
|---|---:|---|
| Single run: pairwise top-5 Jaccard (20 runs) | 0.775 | 0.755 to 0.795 |
| 20-run aggregate: Jaccard between two independent aggregates | 0.923 | 0.902 to 0.944 |
| Compute-matched single run (100k): Jaccard between two runs | 0.994 | 0.986 to 1.000 |
| Single run: full-ranking Spearman | 0.806 | 0.801 to 0.811 |
| 20-run aggregate replicates: Spearman | 0.977 | 0.976 to 0.978 |
| Compute-matched: Spearman | 0.961 | 0.959 to 0.963 |
| Paired difference, aggregate minus single (Jaccard) | +0.148 | +0.125 to +0.172 |
| Paired difference, aggregate minus single (Spearman) | +0.171 | +0.166 to +0.176 |
| Paired difference, aggregate minus compute-matched (Jaccard) | -0.071 | -0.093 to -0.048 |
Chance baselines: top-5 of 150 features 0.019; top-5 among a window's present features 0.494. Top-5 sign consistency is 1.000 in every condition
(saturated, so uninformative). Only 13.6% of the unrestricted aggregate top-5 are present features. Surrogate weighted R^2: 0.385 (single runs), 0.442 (compute-matched);
mean ESS 4,153 (5k runs) and 92,560 (100k runs). Present-features-only Jaccard (secondary): 0.823 single vs 0.948 aggregate.
By group, aggregate vs single Jaccard: FN 0.90 vs 0.72, FP 0.91 vs 0.81, TN 0.90 vs 0.71, TP 0.98 vs 0.86.

## 3. Raw-syscall perturbation (measured; b = 3; primary outcome = direction-aware logit drop)
`direction_aware_drop = s * (logit_before - logit_after)`, s = +1 if the model predicted attack, -1 if normal (positive = moved away from its own decision).
Edits follow the Phase 1 raw-sequence rules; guided and random arms use the same function. About 79-80% of guided edits applied all three features;
all failures are the logged "occurrence consumed by a prior edit" cascade.
| Paired comparison (macro) | Estimate | 95% CI | Holm-adjusted p (primary family) |
|---|---:|---|---:|
| Single-run guided minus its R2 control | +0.821 | +0.602 to +1.037 | 0.003 |
| Aggregate guided minus its R2 control | +0.805 | +0.609 to +1.007 | 0.003 |
| Aggregate guided minus single-run guided | -0.051 | -0.150 to +0.054 | 0.353 |
| Difference in effect over control (aggregate minus single) | -0.021 | -0.148 to +0.095 | secondary |
| Compute-matched guided minus aggregate guided (secondary) | -0.101 | -0.267 to +0.065 | secondary |
| Compute-matched guided minus its R2 control (secondary) | +0.520 | +0.293 to +0.728 | secondary |
Attack-to-normal flip rate (macro-average over the four groups; windows the model predicted as normal cannot flip attack-to-normal and contribute 0): guided minus R2 is +0.045 (single) and +0.047 (aggregate); aggregate minus single guided is exactly 0.000.
By group (aggregate guided minus R2, direction-aware drop): TP +1.26, TN +1.14, FP +0.74, FN +0.09 (FN effect is small).
R1 (secondary, any present feature) is beaten by guided edits by about +1.6 logit units.
Windows excluded from the primary R2 comparison: single-run 16 of 160 (10.0%: 1 infeasible, 15 uninformative); compute-matched 20 of 160 (12.5%); aggregate 0.
Raw `attack_logit_drop` (not direction-aware) and probability-change comparisons are in `results/phase2/paired_comparisons.csv`; they are mostly not distinguishable from zero.

## 4. Cost (measured, single core)
Per window: single run 0.118 s; 20-run aggregate 2.245 s (19.1x); compute-matched 1x100k run 1.169 s (9.9x);
two independent aggregate replicates 4.658 s; second compute-matched run 1.181 s. Entire design: 9.37 s per window, 505,000 model calls.

## 5. Interpretation (supported by the measurements above)
* Aggregation clearly improved explanation **stability** over single-run LIME (Jaccard +0.15, Spearman +0.17), on a 150-feature ranking whose top-5 are mostly absent features.
* Ranking by LIME weight within the sign-supporting present pool **did add effect beyond sign and presence** (guided beat R2 by about 0.8 logit units, both single-run and aggregate).
* Aggregation did **not** measurably improve perturbation effectiveness over single-run LIME: the paired difference is near zero with a CI spanning zero, flip rates are identical.
* A single 100k-sample run, at about half the cost of the aggregate, was **more stable** than the 20-run aggregate. Aggregation therefore showed no advantage over simply spending the same budget on one larger run.
## 6. Not shown / unverified
* That stability implies faithfulness; that gains would hold with official LIME, other models, or real attacker edits; why FN effects are small (not investigated).
* The primary stability contrast compares different statistics (pairwise agreement of single runs vs agreement of two aggregates); it is the natural comparison but not a like-for-like resampling of the same estimator.
* Sample sizes are 40 windows per group from 38-40 clusters; TP/TN were sampled from lazily screened candidates.
