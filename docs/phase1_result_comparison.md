# Phase 1 result comparison: paper vs. our independent reconstruction

## Scope and labels

This document compares Gaspar, Silva, and Silva (2024) with the frozen reconstruction in this repository. Values
carry one of four labels:

- **A — paper-reported** (or derived from paper-reported counts, marked *derived*);
- **B — our Phase 3A** MLP results;
- **C — our Phase 3B** LIME/SHAP explanation results;
- **D — our Phase 3C** extension results.

> **Replication status:** independent reconstruction on public raw ADFA-LD. The authors' 52,656-instance dataset and
> their full preprocessing/split procedure were not recovered, so this is **not an exact numerical replication**, and
> the comparison below is not a controlled head-to-head performance comparison. Differences are not evidence of an
> implementation error, and neither side is called "better".

## A. Dataset and protocol comparison

| Item | Paper (A) | Ours | Relationship |
|---|---|---|---|
| Dataset / source | "Generated version of ADFA-LD" for an IoT IDS (Sec. IV.A, V.A); not publicly recovered | Public raw ADFA-LD Kaggle mirror, windowed locally | Same dataset family, **different data artifact**; reconstruction |
| Number of instances | 52,656 | 88,828 non-overlapping windows (train 61,553 / validation 13,539 / test 13,736) | Differs |
| Class support | Attack 26,870, normal 25,786 (*derived* from confusion matrix; ≈51.0% attack) | Normal 78,599, attack 10,229 windows (11.5% attack); test: 12,113 normal / 1,623 attack | Very different class balance |
| Syscall vocabulary | 149 distinct calls | 175 observed IDs, range 1–340 | Differs |
| Instance representation | 30 system calls | 30-call windows | Aligned |
| Window stride / short traces | Not reported | Stride 30 (non-overlapping); incomplete tails dropped | Our assumption (A2, A3) |
| Preprocessing | 2-gram TF-IDF → chi-square (Sec. V.A) | Adjacent-call tokens `a_b` → TF-IDF fitted on train only → chi-square fitted on train only | Aligned at method level; settings ours |
| Initial n-grams | 2,805 | 3,379 (train-fitted TF-IDF vocabulary) | Differs |
| Selected features | 150 | 150 | Count matches |
| Split / evaluation protocol | Confusion matrix covers all 52,656 instances; no train/test split described in the paper | Trace-disjoint stratified split, 70/15/15 of traces, seed 42; metrics on the 13,736-window held-out test set | Differs |
| MLP framework | TensorFlow/Keras | scikit-learn `MLPClassifier` | Differs |
| MLP architecture | Input + 3 dense layers: ReLU, ReLU, sigmoid. **Hidden widths not reported** | **150 → 64 → 32 → 1** (ReLU, ReLU, logistic output); verified from the saved model's weight shapes | Family aligned; widths are our choice. **No exact architectural reproduction is claimed** |
| MLP training | Not reported | Adam, lr 0.001, batch 256, max 100 iterations, internal early stopping (10% of train, patience 10), no class weights/resampling; stopped at iteration 21 | Our choices (A11–A14) |
| LIME implementation | Official `LimeTabularExplainer`; top-10 features per instance | Local compatible stand-in (`lime/`), **not official LIME**; top-10; 5,000 samples | Method intent aligned; not the same implementation |
| SHAP implementation | `KernelExplainer`; top-10 | `shap.KernelExplainer` (0.50.0); 100-row train-only background (50 normal + 50 attack); `nsamples=512`; explains attack probability | Same method; settings ours |
| Explained instances | Every dataset instance | 20 deterministic test windows: 5 TP, 5 FN, 5 TN, 5 FP | Differs (sample) |
| Perturbation methodology | Remove / replace the 10 most important features ("system calls") using class-exclusive or random class-exclusive calls, over all instances | Phase 3C: raw-sequence replacement at top-10 2-gram occurrences, matched random control, 20 instances | **Extension, not reproduction** (section D) |
| Exact reproduction? | — | No for every row above that says "differs"; overall an independent reconstruction | Not an exact replication |

**Feature terminology.** Features are adjacent syscall transitions (2-grams), such as `221_197` (call 221 followed
by 197), not isolated syscall IDs. The paper's text calls them "system calls" while its Fig. 8 shows 2-gram features.

## B. MLP results

### B.1 Paper-reported (A)

Table 1 and the confusion-matrix discussion (Sec. V.C): accuracy 0.9371, precision 0.9806, sensitivity 0.8946,
specificity 0.9816, F1 0.9356; TP 24,038, FN 2,832, TN 25,311, FP 475; total 52,656. Support (26,870 attack / 25,786
normal) is *derived* from these counts. Computing accuracy from the counts gives 0.93720; Table 1 prints 0.9371, a
rounding-level difference in the paper; we quote the printed value.

### B.2 Our Phase 3A (B) — test set, 13,736 windows

Accuracy 0.937391 · balanced accuracy 0.850313 · normal precision/recall/F1 0.964654 / 0.964336 / 0.964495 ·
attack precision/recall/F1 0.734481 / 0.736291 / 0.735385 · macro F1 0.849940 · ROC-AUC 0.959989 · PR-AUC 0.739275.

| Actual \ Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 11,681 | 432 |
| Attack | 428 | 1,195 |

### B.3 Closest correspondence (not equivalence)

| Paper metric (positive = anomaly) | Paper (A) | Ours (B) |
|---|---:|---:|
| Accuracy | 0.9371 | 0.9374 |
| Precision | 0.9806 | 0.7345 (attack precision) |
| Sensitivity | 0.8946 | 0.7363 (attack recall) |
| Specificity | 0.9816 | 0.9643 (normal recall) |
| F1 | 0.9356 | 0.7354 (attack F1) |
| Balanced accuracy, macro F1, ROC-AUC, PR-AUC | not reported | 0.8503, 0.8499, 0.9600, 0.7393 |

Our reconstructed pipeline achieved 93.74% accuracy, numerically close to the paper's reported 93.71%, but the
datasets, evaluation setup, and implementation are not identical; therefore this is not an exact numerical
replication.

Our attack F1, ROC-AUC and PR-AUC are not directly equivalent to the paper's reported metrics when the underlying
evaluation setup differs, and here it does:

- **Class balance.** The paper's data are ≈51% attack, so an always-"attack" classifier scores ≈51% accuracy. Our test
  set is ≈12% attack, so an always-"normal" classifier already scores 88.2%. Similar accuracy therefore does not mean
  similar difficulty or similar quality. Precision, F1 and PR-AUC depend directly on class prevalence.
- **Evaluation basis.** The paper's counts cover its full dataset; ours cover a held-out test split about four times
  smaller, so raw counts (TP 24,038 vs 1,195) are not comparable, only rates are, and even those describe different
  populations.
- **No counterpart.** The paper reports no ROC-AUC or PR-AUC, so ours have nothing to be compared with.

### B.4 Confusion counts

| Source | TP | FN | TN | FP | Total |
|---|---:|---:|---:|---:|---:|
| Paper (A) | 24,038 | 2,832 | 25,311 | 475 | 52,656 |
| Ours (B), held-out test | 1,195 | 428 | 11,681 | 432 | 13,736 |

## C. Explanations (Phase 3B)

Detail: [`phase3b_xai_explanations.md`](phase3b_xai_explanations.md). Both methods explain the same 20 test windows
(5 TP, 5 FN, 5 TN, 5 FP, seed 42) with top-10 **2-gram** features. Explanations are local to each window.

| Statistic (C) | Value |
|---|---:|
| Mean top-10 Jaccard overlap, LIME stand-in vs Kernel SHAP | 0.08264 |
| Median top-10 Jaccard overlap | 0.05263 |
| Mean sign agreement on overlapping features (14 of 20 windows have overlap) | 0.40476 |

The paper reports no comparable overlap statistics. The low overlap describes these 20 windows only and does not show
that either method is better. LIME here is the vendored stand-in, not the official package (the official
`lime==0.2.0.1` could not be installed in the offline run environment); it must not be described as official-LIME
output.

## D. Perturbation (Phase 3C) — extension, not reproduction

Detail and the "Paper vs. Phase 3C intervention" subsection:
[`phase3c_perturbation_extension.md`](phase3c_perturbation_extension.md).

The paper perturbs the 10 most important features ("system calls") of each instance by removing them or replacing
them with class-exclusive or random class-exclusive calls, and reports the share of changed predictions; it does not
say whether the edit acts on the raw sequence or the feature vector. Phase 3C instead takes the top-10 LIME/SHAP
2-gram features, finds their occurrences in the original 30-call sequence, replaces the second call with a
training-derived replacement value (syscall ID 3, the most frequent call in training-normal data; ID 5 only where the
original call was already 3), re-scores with the frozen pipeline, and compares with matched random edits of the same
count. We do not claim ID 3 is causally neutral.

| Explainer (D) | Windows edited | Guided mean abs Δ attack prob. | Random mean abs Δ | Guided / random flips | Mean edits per window |
|---|---:|---:|---:|---:|---:|
| LIME stand-in | 15/20 | 0.234926 | 0.156690 | 9/20 vs 6/20 | 4.60 |
| Kernel SHAP | 20/20 | 0.352012 | 0.300972 | 9/20 vs 8/20 | 11.25 |

Direction consistency is 13/20 (65%) for LIME and 19/20 (95%) for SHAP; for LIME, 5 of the 13 are windows with no
edit and no change that the code counts as consistent (8/15 among edited windows). These are descriptive results on
20 windows with one random draw per window; no significance test was run, and an edit also changes neighbouring
2-grams, so the results are not causal evidence and not evidence that one explainer is better.

## E. Which paper results are reconstructed

1. **Classification result** (paper Table 1 / Sec. V.C): reconstructed as the Phase 3A MLP on the same task (binary
   normal/attack), the same instance length (30 calls) and the same documented feature pipeline (2-gram TF-IDF,
   chi-square to 150 features). Numbers are not comparable one-to-one (section B.3).
2. **LIME and Kernel SHAP top-10 explanations** (Sec. IV.A): reconstructed on a 20-window sample, with real
   `KernelExplainer` and a LIME stand-in.

Phase 3C is an **additional extension analysis**, not a reconstruction of the paper's perturbation experiments
(Figs. 5–6, Sec. V.D–F).

## F. Claims we do not make

This project does **not** claim:

1. that the authors' exact 52,656-instance dataset or their processed data were recovered or reproduced;
2. that the paper's exact preprocessing or split procedure was recovered;
3. exact numerical reproduction of any paper result, including accuracy;
4. the paper's hidden-layer sizes (they are not reported; ours are 64/32);
5. that the official `lime` package was run (the stand-in was);
6. that Phase 3C reproduces the paper's perturbation protocol;
7. that perturbation results establish causality, or that syscall ID 3 is causally neutral;
8. that the 20-window explanation and perturbation results generalise to the full test set;
9. that LIME is better than SHAP or the reverse;
10. statistical significance — none was tested.

## Reproducibility anchors

Seed 42 · window 30 / stride 30 · trace-disjoint stratified split, validation 15%, test 15% · trace-manifest SHA-256
`765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1` · split-assignment SHA-256
`5c2b0745cccde2c060ce139fe92e383510ab4237eb0d53b64b9c5f29024e04ba`. Provenance:
[`paper_dataset_forensics.md`](paper_dataset_forensics.md), `reports/phase2_5_dataset_audit.md`.
