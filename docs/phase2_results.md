# Phase 2 results: 20-run aggregated LIME vs single-run LIME (held-out test split, run once)

All numbers are copied from `results/phase2/*.csv` produced by `python -m src.phase2.run_phase2 final` followed by `analyze`, and are re-verified by
`python scripts/phase2_audit_checks.py` (85 checks pass with the raw dataset present; without it the three raw-data checks collapse to one SKIP, reported as SKIP, not PASS).
Protocol and selected-window manifest were frozen before any perturbation outcome was computed (`results/phase2/freeze_manifest.json`;
protocol SHA-256 `d99a3f57…7017`, manifest SHA-256 `44388056…9778`). Confidence intervals are 95% percentile intervals from a
2,000-replicate bootstrap over identical-content clusters; the headline estimate is the equal-weight macro-average of the four outcome groups
(mean of the four per-group means). Paired comparisons use the same window in both arms. Measurement and interpretation are kept separate below.
Nothing here shows that explanations are *faithful* to the model.

## 1. Sample (measured)
| Group | Test windows | Selected | Traces | Independent content clusters |
|---|---:|---:|---:|---:|
| TP | 1,195 | 40 | 38 | 38 |
| FN | 428 | 40 | 40 | 40 |
| TN | 11,681 | 40 | 40 | 40 |
| FP | 432 | 40 | 38 | **38** |

All 160 windows have ESS >= 100, >= 4 present, actionable, sign-supporting features under the 20-run aggregate, and obey the global two-per-cluster cap
(verified programmatically; 152 distinct clusters cover the 160 windows). The staged audit is `results/phase2/eligibility_report.csv`; FP/FN were scanned
exhaustively, TP/TN lazily (their R2-eligible counts cover only the windows examined).

## 2. Stability (measured; k = 5; 150 features)
**Metric definitions (exact).**
* *Top-5 Jaccard:* overlap of the sets of the five features with the largest |coefficient| (ties broken by feature index), over all 150 features.
* *Full-ranking Spearman:* Spearman rank correlation of the **absolute** coefficients |w| over the 150 features (an importance-ranking agreement; it is **not** a correlation of signed coefficient vectors). A signed-vector version is given as a supplementary analysis below.
* *Top-5 sign consistency:* for two explanations, the fraction of features **in both top-5 sets** whose coefficient signs agree. It says nothing about features that appear in only one set.
* *Estimands differ.* Single run = mean over all 190 pairs of the 20 individual 5k runs (agreement between two individual explanations). 20-run aggregate = agreement between two independent 20-run aggregates (one pair per window). Compute-matched = agreement between two independent 100k runs (one pair per window). The comparison is therefore between different estimands and procedures, made per window and reported cautiously.

| Quantity (macro over groups) | Estimate | 95% CI |
|---|---:|---|
| Single run: pairwise top-5 Jaccard (190 pairs of 20 runs) | 0.775 | 0.755 to 0.795 |
| 20-run aggregate: Jaccard between two independent aggregates | 0.923 | 0.902 to 0.944 |
| Compute-matched single run (100k): Jaccard between two runs | 0.994 | 0.986 to 1.000 |
| Single run: |w|-ranking Spearman | 0.806 | 0.801 to 0.811 |
| 20-run aggregate replicates: |w|-ranking Spearman | 0.977 | 0.976 to 0.978 |
| Compute-matched: |w|-ranking Spearman | 0.961 | 0.959 to 0.963 |
| Paired difference, aggregate minus single (Jaccard) | +0.148 | +0.125 to +0.172 |
| Paired difference, aggregate minus single (|w| Spearman) | +0.171 | +0.166 to +0.176 |
| Paired difference, aggregate minus compute-matched (Jaccard) | -0.071 | -0.093 to -0.048 |
| Paired difference, aggregate minus compute-matched (|w| Spearman; audit reanalysis) | +0.016 | +0.014 to +0.018 |

**Supplementary (post-run, from the saved coefficient caches; `results/phase2/supplementary_signed_spearman_summary.csv`):** signed-vector Spearman is 0.873 (single run),
0.989 (aggregate replicates) and 0.984 (compute-matched); paired aggregate minus single +0.116 (0.114 to 0.117), aggregate minus compute-matched +0.005 (0.004 to 0.006).
The cached coefficients reproduce the saved |w| Spearman values to <= 1.1e-16. The predeclared metric remains the |w| ranking Spearman; the signed version is not a replacement.

Other measured values: chance baselines for top-5 Jaccard are 0.019 (random 5 of 150) and 0.494 (random 5 among a window's present features). Top-5 sign consistency is exactly 1.000
in every window of every condition (no NaN), so it is **saturated and uninformative**. Only 13.6% of the unrestricted aggregate top-5 are present features. Surrogate weighted R^2: 0.385 (single runs), 0.442 (compute-matched);
mean ESS 4,153 (5k runs) and 92,560 (100k runs). Present-features-only Jaccard (secondary): 0.823 single vs 0.948 aggregate.
By group, aggregate vs single Jaccard: FN 0.90 vs 0.72, FP 0.91 vs 0.81, TN 0.90 vs 0.71, TP 0.98 vs 0.86.

## 3. Raw-syscall perturbation (measured; b = 3)
`direction_aware_drop = s * (logit_before - logit_after)`, s = +1 if the model predicted attack, -1 if normal: positive means the edit moved the model **away from its own original decision**. It is distinct from the literal
`attack_logit_drop = logit_before - logit_after` (positive = attack logit decreased), which for predicted-normal windows has the opposite meaning; both are in `paired_comparisons.csv`, and the raw drop is mostly not distinguishable from zero when pooled.
Edits follow the Phase 1 raw-sequence rules; guided and random arms call the same function. About 79-80% of guided edits applied all three features; all failures are the logged "occurrence consumed by a prior edit" cascade.

| Paired comparison (direction-aware drop, macro) | Estimate | 95% CI | Holm-adjusted p (primary family) |
|---|---:|---|---:|
| Single-run guided minus its R2 control | +0.821 | +0.602 to +1.037 | 0.003 |
| Aggregate guided minus its R2 control | +0.805 | +0.609 to +1.007 | 0.003 |
| Aggregate guided minus single-run guided | -0.051 | -0.150 to +0.054 | 0.353 |
| Difference in effect over control (aggregate minus single) | -0.021 | -0.148 to +0.095 | secondary |
| Compute-matched guided minus aggregate guided (secondary) | -0.101 | -0.267 to +0.065 | secondary |
| Compute-matched guided minus its R2 control (secondary) | +0.520 | +0.293 to +0.728 | secondary |

By group (aggregate guided minus R2, direction-aware drop): TP +1.26, TN +1.14, FP +0.74, FN +0.09 (the FN effect is small). Guided edits beat the R1 control (any present feature, secondary) by about +1.6 logit units.
Windows excluded from the primary R2 comparison: single run 16 of 160 (10.0%: 1 infeasible, 15 uninformative); compute-matched 20 of 160 (12.5%: 2 + 18); aggregate 0.

**Why aggregate and single-run perturbations barely differ (measured).** The guided top-3 *set* is identical under both explanations in 132 of 159 windows (83.0%), and the identical *ordered* list in 89 (56%); where the ordered list is identical the outcomes are identical (max difference 0.0).
In the 43 windows with the same set but a different rank order, 16 outcomes differ (edits are applied in rank order and can cascade). In the 27 windows with a different set, the mean aggregate-minus-single difference is -0.197 (descriptive, no CI).

**Flip rates: two different quantities, both reported.**
* *Macro flip rate (four groups equally weighted; windows the model predicted as normal cannot flip attack-to-normal and contribute 0):* guided minus R2 is +0.045 (single) and +0.047 (aggregate); aggregate minus single guided is 0.000. Because two of the four groups contribute structural zeros, this understates the effect among attack-predicted windows.
* *Conditional flip rate (denominator = windows originally predicted attack, i.e. TP and FP that have both arms; supplementary, from saved outputs; `results/phase2/supplementary_conditional_flip.csv`):*

| Explanation | Guided flips (windows) | Guided rate | R2 rate | Guided minus R2 (TP / FP / pooled over windows) |
|---|---:|---:|---:|---|
| Single run | 72 of 77 (73 clusters) | 0.935 (0.861-0.987) | 0.844 (0.776-0.902) | +0.116 / +0.065 / **+0.091** (0.049 to 0.133) |
| 20-run aggregate | 75 of 80 (76 clusters) | 0.938 (0.866-0.988) | 0.843 (0.775-0.898) | +0.126 / +0.062 / **+0.094** (0.054 to 0.134) |
| Compute-matched | 65 of 71 (67 clusters) | 0.915 (0.836-0.985) | 0.854 (0.781-0.915) | +0.074 / +0.047 / **+0.061** (0.019 to 0.105) |

Aggregate minus single-run guided conditional flip rate: 0.000 (0.000 to 0.000; 80 windows). Random edits alone flip about 84% of attack-predicted windows, so the edit rule itself is very disruptive; a flip is weak evidence on its own and is interpreted only relative to the matched random control.
Sample sizes differ across arms (77 / 80 / 71) because windows with too few eligible features are excluded from each arm's paired comparison.

## 4. Cost (measured, single core)
Per window: single run 0.118 s; 20-run aggregate 2.245 s (19.1x); compute-matched 1x100k run 1.169 s (9.9x); two independent aggregate replicates 4.658 s; second compute-matched run 1.181 s.
Entire design: 9.37 s per window, 505,000 model calls. (Aggregate runs used for eligibility screening were cached and reused; their measured runtimes are those of the original computation.)

## 5. Interpretation (supported by the measurements above)
* Aggregation clearly improved explanation **stability** over single-run LIME on every reported metric (Jaccard +0.15, |w| Spearman +0.17, signed Spearman +0.12), on a 150-feature ranking whose top-5 are mostly absent features. The comparison is between different estimands (see section 2).
* Ranking by LIME weight within the sign-supporting present pool **did add effect beyond sign and presence** (guided beat R2 by about 0.8 logit units, for both single-run and aggregated explanations; conditional flip rate about +0.09).
* Aggregation did **not** measurably improve perturbation effectiveness over single-run LIME: the paired difference is near zero with a CI spanning zero and flip rates are identical, consistent with the guided top-3 sets being the same in 83% of windows.
* **Compute-matched comparison is mixed, not one-sided.** A single 100k-sample run had *higher top-5 Jaccard* than the 20-run aggregate (0.994 vs 0.923) at about half the runtime (1.17 s vs 2.25 s per window), while the aggregate had slightly *higher* |w| and signed Spearman (+0.016 and +0.005). Aggregation therefore showed no advantage over spending the same budget on one larger run in top-5 agreement, and a very small one in full-ranking agreement.

## 6. Not shown / unverified
* That stability implies faithfulness; that results would hold with official LIME, other models, or real attacker edits; why FN effects are small (not investigated).
* Sign consistency (saturated at 1.0) gives no information about sign stability beyond shared top-5 features.
* Sample sizes are 40 windows per group from 38-40 clusters; TP/TN were sampled from lazily screened candidates.
* The raw dataset is not part of the package; the cluster-ID recomputation from raw traces passes only where the dataset is present (it did in the working checkout; it is SKIP in a clean extraction).

## 7. Post-run audit
`docs/phase2_postfreeze_amendments.md` lists every code change after the freeze, with hashes and evidence. `python scripts/phase2_audit_checks.py` re-derives every number in this document from the saved CSVs;
`python scripts/phase2_supplementary_analysis.py` regenerates the supplementary tables from saved outputs only.
