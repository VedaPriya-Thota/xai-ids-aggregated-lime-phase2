# XAI for IDS: LIME and SHAP on an MLP — independent reconstruction

Course project (DA351 Explainable AI). We rebuild the main pipeline of one published paper on public ADFA-LD data,
explain the resulting model with LIME and SHAP, and add one perturbation analysis of our own.

> **Not an exact replication.** We did not recover the authors' dataset or their full preprocessing/split protocol,
> so every number here comes from an *independent reconstruction*. Where our numbers sit near the paper's, that is not
> evidence of exact reproduction. Phase 3C is our **extension**, not a reproduction of the paper's perturbation
> experiment.

**Status:** course Phase 1 (replication) complete. Course Phase 2 (research gap + extension) is not part of this snapshot.

## 1. Project overview and terminology

| Label | Meaning |
|---|---|
| Course Phase 1 | Replication of the research paper — what this repository delivers |
| Course Phase 2 | Research gap + extension — not in this snapshot |
| Repository Phase 2.5 / 2.6 | Dataset reconciliation/audit and baseline freeze |
| Repository Phase 3A | MLP training and evaluation |
| Repository Phase 3B | LIME + SHAP explanations |
| Repository Phase 3C | Perturbation analysis — **our extension** |

## 2. Original paper

Gaspar, D., Silva, P., & Silva, C. (2024). *Explainable AI for Intrusion Detection Systems: LIME and SHAP
Applicability on Multi-Layer Perceptron.* IEEE Access, 12, 30164–30175.
DOI: [10.1109/ACCESS.2024.3368377](https://doi.org/10.1109/ACCESS.2024.3368377)

The paper classifies 30-call syscall sequences as normal/attack with a Keras MLP (2-gram TF-IDF, chi-square to 150
features), explains predictions with LIME and SHAP (`KernelExplainer`), validates the explanations by perturbing the
top-10 features, and adds a user survey (not replicated). Extraction notes: [`docs/paper_extraction.md`](docs/paper_extraction.md).

## 3. What we reproduced (at the level of method, not numbers)

- The pipeline: 30-call instances → 2-gram TF-IDF → chi-square to **150** features → MLP (two ReLU hidden layers, sigmoid output) → LIME and SHAP top-10 explanations.
- A trace-disjoint, seed-42, leakage-checked reconstruction on public raw ADFA-LD, with frozen artifacts, hashes and 67 tests.
- A paper-vs-ours comparison (section 9) with every difference stated.

## 4. What we could not reproduce exactly

The authors' generated 52,656-instance dataset (ours: 88,828 windows); their 149-call vocabulary (ours: 175); their
split/evaluation protocol; the MLP hidden-layer widths and training settings (not reported in the paper); the official
LIME run (we use a local stand-in); explanations for every instance (we explain 20); the paper's perturbation protocol;
the user survey. Details: [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md),
[`docs/paper_dataset_forensics.md`](docs/paper_dataset_forensics.md).

## 5. Our pipeline

```
public raw ADFA-LD traces → label (Training/Validation = normal, Attack = attack)
  → 30-call non-overlapping windows (stride 30, incomplete tails dropped)
  → trace-disjoint stratified split, seed 42 (70/15/15 of traces)
  → adjacent 2-gram tokens "a_b" → TF-IDF (train-only) → chi-square, k=150 (train-only)
  → MLP                                                  [Phase 3A]
  → LIME stand-in + Kernel SHAP, top-10, 20 windows      [Phase 3B]
  → raw-sequence perturbation vs matched random          [Phase 3C, our extension]
```

## 6. Phase 3A — MLP

scikit-learn `MLPClassifier`, architecture **150 → 64 → 32 → 1** (ReLU, ReLU, logistic output), Adam, lr 0.001,
batch 256, max 100 iterations, internal early stopping, no class weights or resampling. The paper reports two ReLU
hidden layers and a sigmoid output but **no hidden-layer widths**, so 64/32 is our choice and we make no claim of
exact architectural reproduction. Headline: 13,736 held-out test windows, accuracy 0.9374, attack F1 0.7354,
ROC-AUC 0.9600. Outputs: `results/experiments/phase3a_strict_reconstructed_mlp/`.

## 7. Phase 3B — LIME + SHAP

20 deterministic test windows (5 TP, 5 FN, 5 TN, 5 FP); top-10 features per window are **2-gram syscall
transitions** (e.g. `221_197`). LIME is a **local stand-in, not the official `lime` package**; SHAP is the real
`KernelExplainer` (100-row train-only background, `nsamples=512`). All explanations are **local to the selected
window**, not global feature importance. LIME–SHAP mean top-10 Jaccard overlap is 0.083. Full detail:
[`docs/phase3b_xai_explanations.md`](docs/phase3b_xai_explanations.md).

## 8. Phase 3C — our extension

> **Not the paper's perturbation experiment.** We map each top-10 2-gram back to the original 30-call sequence,
> replace the *second* syscall of each occurrence with syscall ID 3 (the most frequent call in training-normal data;
> ID 5 only where the original call was already 3), and re-score with the frozen TF-IDF → chi-square → MLP. We compare
> with matched random edits of the same count. Syscall ID 3 is a controlled replacement value, **not claimed causally
> neutral**. Results are descriptive on 20 windows with no significance test. Detail:
> [`docs/phase3c_perturbation_extension.md`](docs/phase3c_perturbation_extension.md).

| Explainer | Windows edited | Guided mean abs Δp | Random mean abs Δp | Guided / random label flips |
|---|---:|---:|---:|---:|
| LIME stand-in | 15/20 | 0.2349 | 0.1567 | 9/20 vs 6/20 |
| Kernel SHAP | 20/20 | 0.3520 | 0.3010 | 9/20 vs 8/20 |

## 9. Paper vs. our results

**Setup** (full table with sources: [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md)):

| Item | Paper | Ours | Exact reproduction? |
|---|---|---|---|
| Dataset / source | "Generated version of ADFA-LD" (IoT IDS); not recovered | Public raw ADFA-LD Kaggle mirror, windowed by us | No — reconstruction |
| Instances | 52,656 | 88,828 windows (13,736 test) | No |
| Syscall vocabulary | 149 | 175 observed (IDs 1–340) | No |
| Preprocessing | 30-call instances; 2-gram TF-IDF (2,805 initial bigrams) → chi-square | 30-call windows, stride 30; `a_b` tokens; train-only TF-IDF (3,379 initial) → train-only chi-square | Method-level match; details ours |
| Selected features | 150 | 150 | Count matches |
| MLP framework | TensorFlow/Keras | scikit-learn `MLPClassifier` | No |
| MLP architecture | Input + 3 dense layers (ReLU, ReLU, sigmoid); widths not reported | 150 → 64 → 32 → 1 | Family only; widths are our choice |
| Test / evaluation protocol | Counts over the full 52,656 instances; no split described in the paper | Trace-disjoint stratified split; metrics on the 13,736-window held-out test set | No |
| LIME implementation | Official `LimeTabularExplainer`, top-10 | Local stand-in, top-10, 5,000 samples | No |
| SHAP implementation | `KernelExplainer`, top-10 | `shap.KernelExplainer` 0.50.0, 100-row background, `nsamples=512` | Same method; settings ours |
| Explained instances | All dataset instances | 20 (5 TP / 5 FN / 5 TN / 5 FP) | No |
| Perturbation methodology | Remove / replace the 10 most important features ("system calls") with class-exclusive or random class-exclusive calls | Raw-sequence replacement at top-10 2-gram occurrences + matched random edits (Phase 3C) | **No — extension** |
| Overall status | — | Independent reconstruction; Phase 3C is an extension | Not an exact replication |

**Classification results.** Paper metrics treat "anomaly" as positive; the closest counterparts are our attack-class values.

| Metric | Paper (A) | Ours, Phase 3A (B) |
|---|---:|---:|
| Accuracy | 0.9371 | 0.9374 |
| Precision | 0.9806 | 0.7345 (attack) |
| Sensitivity / recall | 0.8946 | 0.7363 (attack) |
| Specificity | 0.9816 | 0.9643 (normal recall) |
| F1 | 0.9356 | 0.7354 (attack) |
| ROC-AUC, PR-AUC, balanced acc., macro F1 | not reported | 0.9600, 0.7393, 0.8503, 0.8499 |

Our reconstructed pipeline achieved 93.74% accuracy, numerically close to the paper's reported 93.71%, but the
datasets, evaluation setup, and implementation are not identical; therefore this is not an exact numerical
replication. Our attack F1, ROC-AUC and PR-AUC are not directly equivalent to the paper's reported metrics because the
underlying evaluation setup differs: the paper's data are ~51% attack (derived from its confusion matrix) and our test set is ~12% attack (an
always-"normal" classifier would already score 88.2% accuracy on ours), and the paper reports no AUC values.

Result categories used throughout the docs: **A** paper-reported · **B** our Phase 3A · **C** our Phase 3B
explanations · **D** our Phase 3C extension.

## 10. Limitations and claims we do not make

**Claims we do not make:**

- We do not claim to have recovered the authors' exact processed dataset.
- We do not claim exact numerical reproduction of any paper result.
- We do not claim the paper's hidden-layer sizes (the paper does not report them).
- We do not claim Phase 3C reproduces the paper's perturbation protocol.
- We do not claim perturbation results establish causality, or that syscall ID 3 is causally neutral.
- We do not claim LIME is better than SHAP, or the reverse.
- We do not claim statistical significance; no significance test was run.
- We do not claim the LIME results come from the official LIME package.

**Limitations:** reconstructed dataset and protocol (sections 4, 9); 20-window explanation sample; moderate local
fidelity of the LIME stand-in (surrogate R² ≈ 0.40); small SHAP sampling budget; single seed (42); the survey was not
replicated. Reviewing the frozen files? Some contain legacy labels or stale metadata:
[`docs/frozen_artifact_notes.md`](docs/frozen_artifact_notes.md).

## 11. Reproducibility

Seed 42 everywhere. Raw data are not committed: obtain the public mirror
(<https://www.kaggle.com/datasets/yianqaq/adfa-ld>; checksum in [`data/README.md`](data/README.md)) and extract it to
`data/raw/ADFA-LD/` (or pass `--raw-data-dir` / set `ADFA_LD_RAW_DIR`). Frozen trace-manifest SHA-256:
`765289887ac57f0a7f08b0d2761c9507e65f8ff98805ab69dc08a6cc312bf4f1`.

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q          # 67 tests
```

**Environment caveat:** the frozen artifacts record Python 3.13.5 / scikit-learn 1.8.0 / NumPy 2.3.5 / SHAP 0.50.0,
which differ from the `requirements.txt` pins; the tests pass under scikit-learn 1.8.0 (not tested under the pins).
See [`docs/frozen_artifact_notes.md`](docs/frozen_artifact_notes.md). The vendored `lime/` directory shadows any
installed official `lime` package when the repo root is on `sys.path`.

Regenerating stages is optional (the frozen results are committed). Reruns write to **new** directories and refuse to
overwrite existing ones; without the `--*-dir` flags, 3B/3C read the frozen directories:

```bash
python scripts/run_reconciliation.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD --no-fit   # Phase 2.5 audit
PYTHONPATH=. python scripts/run_phase3a.py --config configs/baseline.yaml --raw-data-dir /path/to/ADFA-LD \
    --experiment-dir results/experiments/phase3a_rerun
PYTHONPATH=. python scripts/run_phase3b.py --raw-data-dir /path/to/ADFA-LD --nsamples 512 --lime-num-samples 5000 \
    --phase3a-dir results/experiments/phase3a_rerun --experiment-dir results/experiments/phase3b_rerun
PYTHONPATH=. python scripts/run_phase3c.py --raw-data-dir /path/to/ADFA-LD \
    --phase3a-dir results/experiments/phase3a_rerun --phase3b-dir results/experiments/phase3b_rerun \
    --experiment-dir results/experiments/phase3c_rerun
```

## 12. Repository structure

| Path | Contents |
|---|---|
| `src/`, `scripts/` | Pipeline code and command-line entry points (`run_reconciliation.py`, `run_phase3a.py`, `run_phase3b.py`, `run_phase3c.py`) |
| `lime/` | Vendored LIME-compatible stand-in (**not** official LIME) |
| `tests/` | 67 tests |
| `configs/` | `baseline.yaml` (frozen; contains legacy scaffold sections, see frozen-artifact notes), `reconciliation.yaml` |
| `data/` | Provenance notes and the frozen trace manifest; raw data never committed |
| `results/experiments/` | Frozen Phase 3A, 3B, 3C outputs (`phase3a_strict_reconstructed_mlp/`, `phase3b_xai_explanations/`, `phase3c_perturbation_evaluation/`) |
| `results/{tables,figures,logs}/` | Phase 2.5 reconciliation outputs (`models/`, `explanations/` are empty placeholders) |
| `reports/` | One-page summary, Phase 2.5 audit, Phase 2.5/2.6 status |
| `docs/` | See below |

| Document | Purpose |
|---|---|
| [`docs/phase1_result_comparison.md`](docs/phase1_result_comparison.md) | Detailed paper-vs-ours comparison |
| [`docs/phase3b_xai_explanations.md`](docs/phase3b_xai_explanations.md) | Phase 3B detail |
| [`docs/phase3c_perturbation_extension.md`](docs/phase3c_perturbation_extension.md) | Phase 3C detail, incl. "Paper vs. Phase 3C intervention" |
| [`docs/replication_protocol.md`](docs/replication_protocol.md), [`ASSUMPTIONS.md`](ASSUMPTIONS.md) | Fixed decisions and every assumption (A1–A25) |
| [`docs/paper_extraction.md`](docs/paper_extraction.md) | What the paper states |
| [`docs/paper_dataset_forensics.md`](docs/paper_dataset_forensics.md) | Search for the authors' dataset |
| [`docs/lime_official_package_verification.md`](docs/lime_official_package_verification.md) | Plan to check the stand-in against official LIME |
| [`docs/frozen_artifact_notes.md`](docs/frozen_artifact_notes.md) | Legacy labels, stale metadata, environment versions |
| [`reports/phase1_replication_summary.md`](reports/phase1_replication_summary.md) | One-page replication summary |

## 13. References and AI usage

- Gaspar, D., Silva, P., & Silva, C. (2024), IEEE Access, 12, 30164–30175, DOI 10.1109/ACCESS.2024.3368377.
- ADFA-LD, public raw mirror: <https://www.kaggle.com/datasets/yianqaq/adfa-ld> (see `data/README.md`).
- AI tools were used for drafting and code support; tools, prompts and how outputs were checked are in [`AI_USAGE.md`](AI_USAGE.md).
