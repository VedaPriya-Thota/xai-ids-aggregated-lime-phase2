# Phase 3C — Explanation-guided perturbation analysis (our extension)

> **Status: extension / reconstruction analysis — NOT a reproduction of the paper's perturbation experiment.**
> Phase 3C is inspired by the paper's idea of validating LIME/SHAP explanations by perturbing the inputs they
> point to. It uses a different intervention, a different sample size and different outcome measures.
> Do not compare its numbers with the paper's perturbation tables/figures.

Repository phase labels: Phase 3A = MLP, Phase 3B = LIME + SHAP, **Phase 3C = this analysis**. All three belong to
course Phase 1 (replication) work; Phase 3C is the part that goes beyond the paper's published protocol.
Frozen outputs: `results/experiments/phase3c_perturbation_evaluation/`. Code: `src/phase3c.py`, `scripts/run_phase3c.py`.

## 1. Paper vs. Phase 3C intervention

**What the paper does** (Sec. IV.B–C, V.D–F; paper-reported, checked against the paper PDF). For every instance it
takes the 10 most important features ("system calls", in the paper's wording) from the LIME or SHAP explanation and
perturbs them in three batches: (1) the 10 features are *removed* from each instance; (2) they are *replaced* by a
system call that appears only in the explanations of the opposite outcome class (e.g. FN→TP), iterating over the
candidate calls; (3) like (2) but with the replacement call chosen randomly (repeated 5 times). The effect is
reported as the percentage of instances whose prediction changes. The paper uses "features" and "system calls"
interchangeably (its features are 2-grams, Fig. 8) and does **not** state whether the edit is applied to the raw
call sequence or to the processed feature vector; no code was recovered.

**What Phase 3C does.** The original paper validates LIME/SHAP explanations by perturbing the most important
features of each explained instance, including feature removal and class-exclusive/random replacement
experiments. Our Phase 3C does not reproduce that intervention protocol exactly.

Instead, we take the top-10 LIME/SHAP 2-gram explanations, locate actionable occurrences of those transitions in
the original 30-call syscall sequence, replace the second syscall with our training-derived replacement value, and
rerun the frozen TF-IDF → chi-square → MLP pipeline. We then compare these explanation-guided edits with matched
random edits using the same number of raw-sequence modifications.

Therefore, Phase 3C should be interpreted as an extension of the paper's perturbation-validation idea rather than a
numerical reproduction of the paper's perturbation experiment.

In this run, syscall ID 3 was used as the controlled replacement value because it was the most frequent syscall in
training-normal data. **We do not claim that syscall ID 3 is causally neutral.** Replacing a call with it is an
intervention like any other and can change the model output.

| Aspect | Paper | Phase 3C (ours) |
|---|---|---|
| Items perturbed | 10 most important features/"system calls" per instance | Actionable occurrences of the top-10 **2-gram** features, in the raw 30-call sequence |
| Operation | Remove; replace with class-exclusive calls; replace with random class-exclusive calls | Replace the **second** call of an occurrence with one training-derived replacement value (no removal, no class-exclusive sets) |
| Representation edited | Not stated in the paper | Raw sequence, then re-tokenised → frozen TF-IDF → frozen chi² → frozen MLP |
| Control | Random choice of the replacement call | Matched random edits: same number of raw-position edits at random positions, same replacement policy |
| Instances | All instances of the dataset | 20 selected test windows (5 TP, 5 FN, 5 TN, 5 FP) |
| Outcome | % of instances whose prediction changes (per class) | Attack-probability change, label flips, guided-vs-random comparison (descriptive) |
| Explainers | Official LIME, KernelExplainer | Local LIME stand-in (not official LIME), KernelExplainer |

## 2. Procedure as implemented

For each explainer (`lime_compatible`, `shap_kernel`) and each of the 20 Phase 3B instances:

1. Take the instance's top-10 explanation features, in rank order. Each is a 2-gram `a_b` (call *a* immediately
   followed by call *b*).
2. In the instance's original 30-call window, find every adjacent occurrence of `a, b`. A feature with no occurrence
   is **not actionable**; it is logged, not skipped silently.
3. Replace the *second* call (*b*) of each occurrence with the replacement value. Each raw position is edited at most
   once.
4. Re-tokenise the edited 30-call sequence into 2-grams and score it with the frozen pipeline (no refitting,
   retraining or rebalancing). Record the change in attack probability and whether the 0.5-threshold label flips.
5. **Matched random control:** apply the same *number* of edits to a fresh copy of the original sequence, at positions
   chosen by a seeded shuffle over all 30 positions (seed derived from SHA-256 of `42|method|selection_id`), with the
   same replacement policy. One random draw per instance and method; random positions may coincide with guided ones.

**Replacement value.** The default is syscall ID **3**, the most frequent call in training-normal windows
(229,341 of 1,640,130 calls). If the call being replaced is already 3 the next most frequent ID is used instead. In
this run 295 of the 317 guided edits used ID 3; the other 22 edited a call that was already 3 and used ID 5. (The
random control used 3 for 260 edits and 5 for 57.) The code and one artifact name this value "neutral"
(`choose_neutral`, `neutral_replacement_metadata.json`); that is a historical label, see
[`frozen_artifact_notes.md`](frozen_artifact_notes.md).

**Side effect of the design.** Editing one position changes up to two adjacent 2-grams, so an edit is not isolated to
the target feature. Results measure how the model output responds to this edit policy.

## 3. Results (D — our Phase 3C results)

Frozen values, 20 instances per explainer (`aggregate_method_summary.csv`):

| Explainer | Instances with ≥1 edit | Mean abs. Δ attack prob., guided | Mean abs. Δ, random | Label flips, guided | Label flips, random |
|---|---:|---:|---:|---:|---:|
| LIME stand-in | 15/20 | 0.234926 | 0.156690 | 9/20 | 6/20 |
| Kernel SHAP | 20/20 | 0.352012 | 0.300972 | 9/20 | 8/20 |

Reading these numbers:

- Means are over **all 20 instances**. The 5 LIME instances with no actionable feature have zero change for both
  arms, which lowers the LIME means.
- Edit volume differs: mean 4.60 edits per instance for LIME vs 11.25 for SHAP, so the two explainers' rows are not
  like-for-like.
- Guided > random on both explainers, but the SHAP margin is small (0.352 vs 0.301) and the random arm is a single
  draw per instance. **No significance test was run; no statistical significance is claimed.**
- **Direction consistency** (`direction_consistent_with_explanation`: the probability moved opposite to the summed
  signed attribution of the edited features) is 13/20 = 65% for LIME and 19/20 = 95% for SHAP. For LIME, 5 of the 13
  are non-actionable instances with no edit and no change, which the code counts as consistent. Among the 15 LIME
  instances that were actually edited it is 8/15 ≈ 53%; for SHAP all 20 were edited.
- Per-outcome-group numbers are in `aggregate_group_summary.csv` and `RESULTS.md`; groups contain 5 instances each.

## 4. What this analysis does and does not show

It shows that, for these 20 windows, edits placed at explanation-selected transitions moved the frozen model's output
more than the same number of randomly placed edits did, under this specific edit policy.

It does **not** show that the explanations are correct or faithful, that the perturbation results establish
causality, that the result generalises beyond 20 instances, that syscall ID 3 is causally neutral, that either
explainer is better than the other, or that the paper's perturbation results were reproduced.
