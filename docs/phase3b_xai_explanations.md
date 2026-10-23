# Phase 3B — LIME and SHAP explanations

Repository Phase 3B explains the frozen Phase 3A MLP with LIME and SHAP. Frozen outputs:
`results/experiments/phase3b_xai_explanations/`. Code: `src/phase3b.py`, `lime/lime_tabular.py`,
`scripts/run_phase3b.py`. Nothing is refitted: the model, TF-IDF vectorizer and chi-square selector are loaded
from Phase 3A.

## 1. Explanations are local

Every LIME and SHAP result in this repository explains **one prediction of one test window**: which of the 150
features pushed *that* window's attack probability up or down. They are local explanations. They are not global
feature importances and must not be read as such.

## 2. The explained instances

- **20 deterministic test windows: 5 TP, 5 FN, 5 TN, 5 FP**, outcome defined by the frozen 0.5-threshold prediction.
- Within each outcome group, 5 windows are drawn by seeded random sampling (`random_state=42`) and sorted by window
  index; they are not simply the first of each type. List: `selected_test_instances.csv`.
- The paper explains every instance of its dataset; this is a 20-window sample, so all Phase 3B statistics are
  descriptive for these 20 windows only.

## 3. What a "feature" is

The 150 model inputs are chi-square-selected TF-IDF weights of syscall **2-gram transitions**, written `a_b` (call
*a* immediately followed by call *b*, e.g. `221_197`). Top-10 features per instance are ranked by absolute
attribution; the sign is reported separately (`signed_attribution`). A feature is a transition, not an isolated
syscall.

## 4. LIME — local stand-in, not the official package

| Setting | Value |
|---|---|
| Implementation | Project-vendored `LimeTabularExplainer`-compatible class (`lime/lime_tabular.py`, 70 lines), version tag `vendored-compat-0.2.0.1`. **Not the official `lime` package.** |
| Method | Gaussian perturbations around the training mean/std, exponential kernel weighting (width 0.75·√150 ≈ 9.19), weighted Ridge(α=1) surrogate on all 150 features, explained quantity = attack-class probability; top-10 by absolute coefficient |
| Reference data | Full transformed TRAIN matrix (61,553 × 150) |
| Other settings | `num_samples=5000`, `num_features=10`, `discretize_continuous=False`, `random_state=42` |
| Surrogate fit | Weighted R² of the local surrogates: 0.21–0.44, mean 0.40 (`lime_surrogate_metadata.csv`), i.e. moderate local fidelity |

The official package could not be installed in the offline run environment. The stand-in implements a subset of the
`LimeTabularExplainer` API and has **not** been compared against the official package, so its rankings should not be
presented as official-LIME output. Verification plan: [`lime_official_package_verification.md`](lime_official_package_verification.md).

## 5. SHAP — `KernelExplainer`

| Setting | Value |
|---|---|
| Implementation | Real `shap.KernelExplainer` (SHAP 0.50.0 per frozen metadata) |
| Explained output | Attack-class probability (`predict_proba[:, 1]`) |
| Background | 100 TRAIN windows, 50 normal + 50 attack (seeded selection; `shap_background_instances.csv`). The 50/50 mix is a design choice; the training set is ~11% attack. SHAP's baseline value (0.362) is relative to this background |
| Sampling | `nsamples=512`, `l1_reg="num_features(150)"`, seed 42 |

`nsamples=512` is a small coalition budget for 150 features; approximation error was not measured.

## 6. Results (C — our Phase 3B explanation results)

| Statistic (20 instances) | Value |
|---|---:|
| Mean top-10 Jaccard overlap, LIME vs SHAP | 0.08264 |
| Median top-10 Jaccard overlap | 0.05263 |
| Mean sign agreement on overlapping features | 0.40476 |

Six of the 20 instances share no top-10 feature between the two methods. The sign-agreement mean covers only the 14
instances with at least one shared feature (28 shared features in total). The low overlap says the two methods'
top-10 lists differ on these windows; it does not say which method is better, and no such claim is made.

## 7. Files named "global"

`lime_global_summary.csv`, `shap_global_summary.csv` and `figures/*_global_top_features.png` are descriptive
aggregates of the 20 local explanations: for each feature, the mean absolute attribution over those instances where
it appeared in the top-10 (instances where it did not appear are not counted as zero). They are **not** global
feature importance. The file names are historical and unchanged because they are frozen artifacts.

## 8. Limitations

20-window sample; LIME stand-in not validated against official LIME; modest SHAP sampling budget; single seed; no
statistical test; explanations describe the model's behaviour, not ground-truth attack causes.
