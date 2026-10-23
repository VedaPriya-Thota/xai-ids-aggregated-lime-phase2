# Phase 1 Replication Summary (one page)

## 1. Objective
We reconstruct the main workflow of Gaspar, Silva & Silva (2024), *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron* (IEEE Access 12), and compare our behaviour with the paper's reported results. The goal is an honest, reproducible reconstruction, not forced agreement. **This is not an exact replication.**

## 2. Dataset challenge
The authors' 52,656-instance generated dataset (149 syscalls) was not recovered. We rebuilt from public raw ADFA-LD: 5,951 traces (5,205 normal, 746 attack) → 88,828 non-overlapping 30-call windows (11.5% attack), 175 observed syscall IDs. Dataset, split and evaluation basis therefore differ from the paper's.

## 3. Our pipeline
30-call windows → adjacent 2-gram tokens → train-only TF-IDF (3,379 initial tokens) → train-only chi-square to 150 features → scikit-learn MLP **150 → 64 → 32 → 1** (ReLU, ReLU, logistic) → LIME stand-in and Kernel SHAP. Trace-disjoint stratified split, seed 42, 15% validation / 15% test, no class weighting or resampling. The paper reports two ReLU hidden layers but not their widths or training settings, so 64/32 and the training settings are our choices; we claim no exact architectural reproduction.

## 4. Phase 3A — MLP (our result, B)
On 13,736 held-out test windows: accuracy 0.9374, attack precision 0.7345, attack recall 0.7363, attack F1 0.7354, ROC-AUC 0.9600, PR-AUC 0.7393 (TN 11,681, FP 432, FN 428, TP 1,195). The paper (A) reports accuracy 0.9371, precision 0.9806, sensitivity 0.8946, specificity 0.9816, F1 0.9356 over its full dataset (TP 24,038, FN 2,832, TN 25,311, FP 475). Our accuracy is numerically close to the paper's, but datasets, evaluation setup and implementation differ, so this is not an exact numerical replication. Our attack F1, ROC-AUC and PR-AUC are not directly equivalent to the paper's metrics: the paper's data are ≈51% attack, our test set ≈12% (an always-"normal" classifier scores 88.2% on ours), and the paper reports no AUCs.

## 5. Phase 3B — explanations (our result, C)
20 deterministic test windows (5 TP, 5 FN, 5 TN, 5 FP), top-10 **2-gram** features each; explanations are local to each window. LIME is a local stand-in, **not the official package** (it could not be installed offline); SHAP is the real `KernelExplainer` (100-row train-only background, `nsamples=512`). Mean LIME–SHAP top-10 Jaccard overlap 0.083, mean sign agreement 0.405 (14 windows with overlap). Descriptive only; neither method is claimed better.

## 6. Phase 3C — our extension (D), not the paper's experiment
The paper perturbs the 10 most important features/"system calls" by removal or class-exclusive/random replacement. Phase 3C does not reproduce that. We map top-10 2-grams to the raw 30-call sequence, replace the second call with syscall ID 3 (most frequent in training-normal data; not claimed causally neutral), re-score with the frozen pipeline, and compare with matched random edits. Mean absolute attack-probability change, guided vs random: LIME stand-in 0.235 vs 0.157; Kernel SHAP 0.352 vs 0.301; label flips 9 vs 6 and 9 vs 8 of 20. Descriptive, 20 windows, no significance test, not causal evidence.

## 7. Limitations
Reconstructed data and protocol; unreported paper hyperparameters; LIME stand-in not validated against official LIME; 20-window sample; single seed; survey not replicated; frozen artifacts were produced under package versions that differ from `requirements.txt` (see `docs/frozen_artifact_notes.md`).

## 8. Conclusion
The repository provides a frozen, reproducible, independent reconstruction of the paper's pipeline plus one clearly labelled extension. Full comparison: `docs/phase1_result_comparison.md`; claims we do not make: `README.md` §10.
