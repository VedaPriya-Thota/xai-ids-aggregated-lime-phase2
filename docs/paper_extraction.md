# Paper extraction

**Paper:** Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems:
LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164-30175.
DOI: 10.1109/ACCESS.2024.3368377

**Reading coverage:** extracted originally from journal pp. 30168-30171 (Sections IV-V.D, Figs 2-9, Table 1). During the
final documentation pass the full 12-page paper PDF was re-checked by OCR text search for every item marked "Not found"
and for the perturbation description (Sec. IV.B-C, V.D-F). No hidden-layer widths, optimizer/training settings, train/test
split, or LIME/SHAP hyperparameters are reported. OCR is imperfect: re-read the cited pages before quoting a "not
reported" claim.

Match status values: **Confirmed** (in paper) / **Derived** (computed from paper numbers) /
**Not found** (assumption used) / **Unverified** (only from a secondary summary) / **Deviation** (we differ on purpose).

## A. Confirmed from the paper

| Paper item | Extracted detail | Evidence/source | Replication implementation | Match status |
|---|---|---|---|---|
| Task | Binary IDS: 0 = normal, 1 = anomaly | Sec. V.C, p. 30171 | `label_map` normal 0 / attack 1 | Confirmed |
| Dataset | Generated version of ADFA-LD, IoT IDS context | Sec. IV.A p. 30168; Sec. V.A p. 30170 | Rebuild from raw ADFA-LD | Deviation (A1) |
| Instance | Sequence of 30 system calls | Sec. V.A, p. 30170 | `window_size: 30` | Confirmed |
| Vocabulary | 149 distinct system calls | Sec. V.A, p. 30170 | Report our count in the audit | Confirmed (ours may differ) |
| Dataset size | 52,656 instances | Sec. V.C text, p. 30171 | Report our window count | Confirmed (ours will differ) |
| Attack types | Password brute force, DoS, network scan | Sec. IV.A, V.A | Attack_Data_Master, mapping unverified | Confirmed, mapping unverified (A5) |
| Vectorization | TF-IDF with n=2 (bigrams) | Sec. V.A, p. 30170 | `a_b` 2-gram tokens built from adjacent calls, then `TfidfVectorizer` over them (A9) | Confirmed (paper); implementation ours |
| Initial features | 2,805 n-grams | Fig. 8, p. 30170 | Report our initial count | Confirmed (ours may differ) |
| Feature reduction | Chi-squared test to 150 features | Sec. V.A, Fig. 8 | `SelectKBest(chi2, k=150)`, train only | Confirmed |
| Feature identity | Features are bigrams (e.g. '91 195') although text says "system calls" | Fig. 8 vs. Sec. V.D | Features are bigram columns | Confirmed; terminology mismatch noted |
| Model | TensorFlow/Keras MLPClassifier, input + 3 dense layers | Sec. V.C, p. 30170-30171 | scikit-learn `MLPClassifier`, 150 -> 64 -> 32 -> 1 (A11, A25) | Confirmed (paper); Deviation (framework, widths) |
| Activations | Dense 1 and 2 ReLU, dense 3 sigmoid | Sec. V.C | ReLU, ReLU, logistic (sigmoid) output | Confirmed (paper); aligned |
| Input dimension | 150 | Sec. V.C | 150 selected features | Confirmed |
| Confusion matrix (text) | TP 24,038; FN 2,832; TN 25,311; FP 475 (sum 52,656) | Sec. V.C, p. 30171 | Reference values | Confirmed |
| Figure 9 | Cell labels appear inconsistent with the text (25,311 sits in a Positive/Positive cell) | Fig. 9, p. 30171 | Use text values as reference | Confirmed inconsistency |
| Table 1 | Accuracy 0.9371, Precision 0.9806, Sensitivity 0.8946, Specificity 0.9816, F1 0.9356 | Table 1, p. 30171 | `docs/phase1_result_comparison.md` | Confirmed |
| Evaluation basis | Counts cover all 52,656 instances; no split described in pages read | Fig. 9, Sec. V.C | Held-out test set | Deviation (A8) |
| LIME | `LimeTabularExplainer` given data, feature names, class names; top-10 features per instance | Sec. IV.A, p. 30168 | Vendored LIME-compatible stand-in (not official LIME), top-10 (A15) | Confirmed (paper); Deviation (implementation) |
| SHAP | `KernelExplainer`, top-10 features per instance | Sec. IV.A, p. 30168 | `shap.KernelExplainer`, top-10 (A16) | Confirmed (paper); same method, settings ours |
| Explained set | Explanations for all instances of the dataset | Sec. IV.A-C | 20 seeded-random test windows, 5 per outcome group (A17) | Deviation (A17) |
| Perturbation level | Edits are described on "system calls" in "input instances"; the paper does not say whether they act on the raw sequence or the processed feature vector | Sec. IV.B-C, V.D-F | Phase 3C edits the raw sequence (A19) | Not stated in the paper |
| Perturbation batch 1 | Remove the 10 most important features/system calls from every instance, re-run model | Fig. 5, p. 30169 | Not reproduced; Phase 3C is an extension (A19-A21) | Confirmed (paper); not reproduced |
| Perturbation batch 2 | Iteratively replace class-exclusive features (X from TP-only, Y from FN-only sets) in FN/TP instances; repeat for TN/FP | Fig. 6, p. 30169 | Not reproduced (A21) | Confirmed (paper); not reproduced |
| Perturbation batch 3 | Same as batch 2 but X and Y chosen randomly | Fig. 6, p. 30169 | Not reproduced (A21) | Confirmed (paper); not reproduced |
| Example result | `setsockopt` replaced by `accept`: 90% of affected FN became TP | Sec. V.D, p. 30171 | Optional comparison target | Confirmed |
| Class-exclusive features | Some features appear only in FN (or TP) explanations | Sec. V.D | Optional analysis | Confirmed |

## B. Derived or unverified

| Paper item | Detail | Source | Status |
|---|---|---|---|
| Class counts | Attack 26,870 (TP+FN); normal 25,786 (TN+FP) | Derived from the text counts | Derived (not stated directly) |
| Accuracy check | (24,038 + 25,311) / 52,656 = 0.93720 | Derived | Table 1 prints 0.9371 (rounding-level difference) |
| Survey | 24 participants comparing LIME vs SHAP | Secondary summary only | Unverified; survey not replicated (A22) |
| Other perturbation results | e.g. `nfsservctl` replacement moving ~100% of some FP to TN | Secondary summary only | Unverified until pp. 9-12 read |

## C. Not found in pages read (assumptions used)

| Paper item | Our replication value | Assumption ID |
|---|---|---|
| Hidden-layer widths | 64, 32 | A11 |
| Optimizer, learning rate, batch size, epochs | Adam, 1e-3, 256, max 100 | A12 |
| Early stopping | scikit-learn internal (10% of train, patience 10) | A13 |
| Train/validation/test split, stratification | 70/15/15 trace-disjoint stratified | A7, A8 |
| Random seed | 42 | A23 |
| Window generation, stride, short-trace handling | stride 30, drop short traces | A2, A3 |
| Class balancing | none | A6 |
| LIME num_samples, kernel width, discretization | 5000, default, false | A15 |
| SHAP background size and sampling | 100 train-only rows (50 normal + 50 attack), nsamples=512 | A16 |
| How a system call is "removed"/"replaced" | Replace the second call of a top-10 2-gram occurrence in the raw sequence with syscall ID 3 (Phase 3C extension) | A19 |
| Package versions, hardware | See docs/frozen_artifact_notes.md | A24 |
