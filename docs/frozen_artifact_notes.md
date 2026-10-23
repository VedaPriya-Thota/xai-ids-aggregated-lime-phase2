# Frozen artifact notes — legacy labels and known inconsistencies

The files below were produced by the Phase 2.5–3C runs and are kept byte-for-byte as evidence. Several are also
recorded as protected inputs (SHA-256) by Phase 3B/3C, so editing them would break the recorded provenance. A few
contain stale or historical wording. This page lists each one and gives the authoritative value, so nobody has to
guess. Where this page and a frozen file disagree, this page and the saved model/metrics are right.

| # | Where | What it says | Authoritative value / how to read it |
|---|---|---|---|
| 1 | `src/phase3a.py`, `model_cfg` literal (the dict written to metadata on a fresh run) | `"architecture": "150 input -> 128 ReLU -> 64 ReLU -> 1 logistic output"`, `"batch_size": 128` | The `MLPClassifier(...)` call a few lines below builds hidden layers **(64, 32)** with **batch_size=256**. The saved model confirms it: weight shapes (150,64), (64,32), (32,1); `model.batch_size == 256`. The frozen `config_snapshot.json` / `reproducibility_metadata.json` also record 64→32 and 256. The stale literal is metadata only and does not affect training; it was left unedited because the implementation is frozen. A fresh Phase 3A rerun from this source would write the stale text into its metadata. The frozen metadata also carries `n_iter_actual` and `final_training_loss`, which this source does not write, so the committed source is not a byte-identical record of the code that produced the frozen metadata |
| 2 | `configs/baseline.yaml`, sections `model`, `training`, `lime`, `shap`, `explain_instances`, `perturbation` | e.g. `hidden_units: [128, 64]`, `batch_size: 128`, Keras-style early stopping on `val_loss`, SHAP background 50, `nsamples: auto`, first-in-test-order selection, "set bigram column to 0" removal | Early scaffold values. Phase 3A/3B/3C read only `seed`, `paths`, `data`, `features`, `split` from this file (one unit test reads `model`/`training` only to assert that no class weighting or resampling is configured). The real settings are in the code and in the saved metadata (see Phase 3A/3B/3C docs). The file's SHA-256 (`07e7d3a6…`) is recorded as protected provenance in the Phase 3C outputs, so it is not edited |
| 3 | `reports/phase2_5_dataset_audit.md`, reconciliation table row "Syscall IDs … 171" | 171 | **175** (same file's Vocabulary section, `phase2_5_dataset_audit.json`, 175 rows in `reconciliation_syscall_frequency.csv`). File is hash-protected, not edited |
| 4 | `reports/PHASE2_5_EXECUTION_STATUS.md`, `docs/paper_dataset_forensics.md` | "46 passed" / "45 tests"; "do not start MLP/LIME/SHAP yet" | Historical snapshots from before Phase 3. Current suite: 67 tests |
| 5 | Phase 3B/3C generated `README.md`, `RESULTS.md`; `neutral_replacement_metadata.json`; code names `choose_neutral`, `neutral_default` | "neutral policy", "neutral replacement" | Historical label for the *controlled replacement value* (syscall ID 3, with ID 5 as fallback; see [`phase3c_perturbation_extension.md`](phase3c_perturbation_extension.md)). No causal neutrality is claimed |
| 6 | Phase 3B `reproducibility_metadata.json`, field `lime.method` = `lime.lime_tabular.LimeTabularExplainer` | Looks like the official class | It names the vendored class. `software.lime = vendored-compat-0.2.0.1` is the accurate field. It is not official LIME |
| 7 | Phase 3B `*_global_summary.csv`, `*_global_top_features.png` | "global" | Aggregates of 20 local explanations, not global feature importance ([`phase3b_xai_explanations.md`](phase3b_xai_explanations.md), §7) |
| 8 | Phase 3B/3C `README.md` text "LimeTabularExplainer-compatible implementation" | — | Accurate; it is the vendored stand-in |

## Software environment: what was recorded vs. what `requirements.txt` pins

| Package | Recorded in the frozen Phase 3A/3B/3C metadata | `requirements.txt` pin |
|---|---|---|
| Python | 3.13.5 | (notes mention 3.12) |
| NumPy | 2.3.5 | 1.26.4 |
| pandas | 2.2.3 | 2.2.2 |
| scikit-learn | 1.8.0 | 1.4.2 |
| joblib | 1.5.3 | 1.4.2 |
| matplotlib | 3.10.8 | 3.8.4 |
| SHAP | 0.50.0 | 0.45.1 |
| LIME | vendored stand-in | `lime==0.2.0.1` (official; not what runs, see below) |
| TensorFlow / Keras | not used | 2.16.2 / 3.3.3 (listed but unused) |

So the frozen artifacts were **not** produced under the versions `requirements.txt` pins. `requirements.txt` was left
unchanged in this documentation pass. To load the frozen `.joblib` files exactly as produced, use the recorded
versions above. During this cleanup the test suite (67 tests) was run under Python 3.12.3, scikit-learn 1.8.0, NumPy
2.4.4, pandas 3.0.2, SHAP 0.52.0, pytest 9.1.1 and passed; it was not run under the `requirements.txt` pins.

**`lime/` shadows the official package.** The vendored package is named `lime` and sits at the repository root. In the
documented commands (`PYTHONPATH=.`, `python -m pytest`) the root precedes site-packages on `sys.path`, so
`import lime` resolves to the vendored code even if `lime==0.2.0.1` is installed. The official package is therefore
not what runs; see [`lime_official_package_verification.md`](lime_official_package_verification.md).
