# Phase 2 research gap and improvement (one page)

**Original paper and pipeline.** Gaspar, Silva & Silva (2024, *IEEE Access* 12:30164-30175, doi:10.1109/ACCESS.2024.3368377) apply LIME (Ribeiro et al., KDD 2016) and SHAP (Lundberg & Lee, NIPS 2017)
to an MLP intrusion detector on ADFA-LD system-call windows and evaluate explanations with perturbation and a user survey. Phase 1 independently reconstructs the pipeline
(30-call windows, TF-IDF 2-grams, chi-squared selection to 150 features, frozen MLP, vendored LIME stand-in). The authors' 52,656-instance data and scripts were not recoverable, so this is a reconstruction, not an exact replication.

**Gap.** The paper evaluates LIME and SHAP explanations but does not quantify the run-to-run stability of the stochastic LIME explanations, nor compare explanation-guided perturbations against matched random controls.
Related work motivates both: LIME stability indices were proposed because repeated LIME runs can disagree (Visani et al., *J. Oper. Res. Soc.* 73(1):91-101, 2022); feature-removal evaluations are interpreted against a random baseline (Hooker et al., NeurIPS 2019);
and explanation methods in security are assessed on criteria including stability and efficiency (Warnecke et al., IEEE EuroS&P 2020).

**Single improvement.** Replace single-run LIME by the mean signed coefficient over 20 independently seeded runs (full 150-feature vectors averaged, ranked by absolute mean), on the unchanged frozen baseline.

**Design.** 160 held-out test windows (up to 40 each of TP/FN/TN/FP; at most two per identical-content cluster across all groups; ESS >= 100; >= 4 present, actionable, sign-supporting features).
Baseline: a fresh seeded single run (5,000 samples). Controls: independent aggregate replicates, a compute-matched single run (100,000 samples), and matched random draws (R2: 100 per window; R1 secondary).
Perturbations edit the raw 30-call window with the Phase 1 rules (budget b = 3), then re-run the frozen vectorizer, selector and MLP. Metrics: top-5 Jaccard, full-ranking Spearman, sign consistency; direction-aware attack-logit drop,
probability change and flip rate versus the matched random control. Inference: paired differences, 2,000-replicate cluster bootstrap, Holm correction on three pre-declared comparisons.

**Measured findings** (details and CIs in `phase2_results.md`). Aggregation raised top-5 Jaccard from 0.775 to 0.923 (paired +0.148, CI +0.125 to +0.172) and the Spearman correlation of absolute-coefficient rankings from 0.806 to 0.977 (these compare pairwise agreement among single runs with agreement between two independent 20-run aggregates, which are different estimands).
Guided edits beat matched random edits by about 0.8 logit units for both single-run (+0.821) and aggregated (+0.805) explanations, but aggregated and single-run guided edits did not differ (-0.051, CI -0.150 to +0.054; identical flip rates; the guided top-3 set is the same in 83% of windows).
Among windows predicted as attack, guided edits flipped the decision to normal in about 94% versus about 84% for matched random edits (+0.09).
A single 100k-sample run had higher top-5 Jaccard than the aggregate (0.994) at about half the cost (1.17 s vs 2.25 s per window; the aggregate cost 19x a single run), although the aggregate had slightly higher ranking-correlation. Whole design: 9.4 s per window.

**Interpretation supported by the data.** Aggregation buys stability, not better perturbation effects, and gave no top-5 advantage over spending the same budget on one larger run (a very small ranking-correlation advantage remained). **Unverified:** that stability implies faithfulness; behaviour with official LIME, other models or data.

**Limitations.** (1) Independent reconstruction, not exact replication of the unavailable dataset. (2) Duplicate traces exist; the global cluster cap mitigates but does not eliminate dependence or train/test content overlap.
(3) FP has 38 independent clusters (40 windows). (4) The LIME used is a vendored stand-in. (5) Aggregation costs about 20x the explanation computation of a single run (measured above, replicates and controls extra).
(6) Raw-syscall edits are model-input perturbations, not proof of attacker behaviour. (7) Results apply to the frozen reconstructed baseline only.

**References.** Gaspar et al. (2024) above; Ribeiro, Singh & Guestrin, "Why Should I Trust You?", KDD 2016 (arXiv:1602.04938); Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NIPS 30, 4765-4774, 2017;
Visani, Bagli, Chesani, Poluzzi & Capuzzo, "Statistical stability indices for LIME", J. Oper. Res. Soc. 73(1):91-101, 2022; Hooker, Erhan, Kindermans & Kim, "A Benchmark for Interpretability Methods in Deep Neural Networks", NeurIPS 2019;
Warnecke, Arp, Wressnegger & Rieck, "Evaluating Explanation Methods for Deep Learning in Security", IEEE EuroS&P 2020 (arXiv:1906.02108).
