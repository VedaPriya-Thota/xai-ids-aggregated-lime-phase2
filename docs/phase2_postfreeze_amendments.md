# Phase 2 post-freeze amendments (audit record)

`results/phase2/freeze_manifest.json` (identical to `FINAL_RUN_LOCK.json`, written 2026-10-24T05:52:37Z, **before any perturbation outcome was computed**) records SHA-256 hashes of the Phase 2 code,
the protocol and the selected-window manifest. It is kept unedited as a historical record. After the freeze, exactly two recorded source files were changed. This note states what, why, and what the evidence does and does not establish.

## What did not change
* `docs/phase2_protocol.yaml` (`d99a3f57…7017`) and `results/phase2/selected_windows.csv` (`44388056…9778`) still match the freeze manifest (checked by `scripts/phase2_audit_checks.py`).
* Seven of the nine recorded code files (`__init__`, `evaluate_perturb`, `explain_aggregated`, `metrics_stability`, `perturb_raw_sequence`, `run_phase2`, `sampling`) are byte-identical to the frozen hashes. **All nine frozen hashes equal the files at pre-freeze commit `b4de7c9`.**
* The per-window results (`lime_runs/`, `window_results/`, `perturbation_results.csv`, `stability_results.csv`, `selected_windows.csv`, `runtime_results.csv`) are byte-identical between the commit that first recorded them (`72b3936`) and the delivered commit (`ae8461c`) and the audit commit (`git diff --stat` empty).

## What changed
| File | Frozen hash (= `b4de7c9`) | After exclusion fix (`72b3936`) | Delivered (`ae8461c`) | After this audit |
|---|---|---|---|---|
| `src/phase2/analyze.py` | `f466257e…ba144` | `4d80026a…ece9` | `dc3a4f8d…14c9` | `c73bd10c…7cd9` |
| `src/phase2/frozen.py` | `8c2200c0…54ce7` | `67cf0aff…77af9` | `67cf0aff…77af9` | `67cf0aff…77af9` |

1. **`frozen.py`, `verify_protected()` (commit `72b3936`).** Added support for an `append_only_authorized` entry in `results/phase2/protected_hashes_baseline.json`, so the one authorised edit (append-only disclosure in `AI_USAGE.md`) passes while any change to its original bytes still fails.
   Purpose: verification logic only. It is a guard that runs before and after the experiment; it computes no explanation, perturbation or statistic. The 167-file baseline hashes themselves are unchanged. `AI_USAGE.md` original size 4,514 bytes.
2. **`analyze.py`, exclusion bookkeeping (`72b3936`).** `perturbation_tables()` gained a `sel` argument and now counts exclusions against the frozen 160-window sample. The original code counted only windows that had perturbation rows, so a window whose pool had fewer than three eligible features (guided infeasible, no rows by design) was invisible.
   Original output (regenerated in this audit with the pre-amendment code, since it was never committed): single-run 159 windows, 0 infeasible, 15 uninformative; compute-matched 158 windows, 0 infeasible, 18 uninformative. Corrected: single-run 1 infeasible + 15 = 16 of 160 (10.0%); compute-matched 2 + 18 = 20 of 160 (12.5%); aggregate 0. The table's columns were renamed accordingly.
3. **`analyze.py`, figure presentation (`ae8461c` and this audit).** Figure 1: shorter axis labels and a legend. Figures 2-3: legend placement; figure 3 additionally marks FN/TN as "n/a (predicted normal)" and states in its title that the denominator is windows predicted attack. No plotted value changed (figure 3 already used TP/FP only).

## Did the changes affect experimental outputs?
* **Could they have?** The amended code paths read already-saved CSVs (`analyze.py`) or check hashes (`frozen.py`). Neither is called while LIME runs, edits are applied, or outcomes are scored. This is established by reading the diffs (`git diff b4de7c9 HEAD -- src/phase2/analyze.py src/phase2/frozen.py`) and by the fact that no other source file changed since `b4de7c9`.
* **Empirical check (this audit).** The *pre-amendment* `analyze.py` from `b4de7c9` was run on the saved outputs. Five of six tables (`stability_summary`, `runtime_summary`, `perturbation_summary`, `paired_comparisons`, `edit_diagnostics`) were numerically identical to the current ones; only `exclusions.csv` differs, as described in item 2.
  Re-running the delivered `analyze` on a clean extraction reproduced every committed table and figure byte-for-byte.
* **What Git cannot establish.** The test-run outputs were first committed in `72b3936`, after the amendments were made in the working tree, so Git history alone does not date their generation. The supporting evidence is: the freeze manifest's file timestamp (05:52:37Z) precedes the result CSV timestamps (06:15:40Z) in the delivered package; the recorded code hashes equal `b4de7c9`; and the second launch of `final` (the first died before writing the lock or any window result) refuses to proceed unless the protocol and manifest hashes match.
  The recorded command line in the manifest (`.../run_phase2.py final`) is `sys.argv[0]` of `python -u -m src.phase2.run_phase2 final`.

## Other post-freeze additions (not hashed in the manifest)
`docs/phase2_results.md`, `docs/phase2_research_gap.md`, `docs/phase2_README.md`, `requirements-phase2.txt`, `scripts/phase2_audit_checks.py`, `scripts/phase2_supplementary_analysis.py`, `tests/test_phase1_protection.py` (one added test), `tests/test_phase2_audit.py`, an append-only paragraph in `AI_USAGE.md`, and the `append_only_authorized` key in the baseline JSON.
Four result files (`random_draw_log.csv.gz` and `eligibility_audit_all_windows.csv.gz`, for the pilot and the final run) were present in the delivered ZIP but missing from Git history and from the patch, because Phase 1's `.gitignore` ignores `*.gz`; they are now force-added (the protected `.gitignore` was not edited).


## Documentation-only amendment for submission (README)
* **What.** The Phase 1 `README.md` (one of the 167 hash-protected files) was moved **byte-for-byte** to `docs/PHASE1_README.md` and a new top-level `README.md` describing the whole project (Phase 1 and Phase 2) replaced it. In `results/phase2/protected_hashes_baseline.json` the key `README.md` was renamed to `docs/PHASE1_README.md`; its SHA-256 value is unchanged, so `verify` still checks the original Phase 1 README bytes (167 files, none changed).
* **Why.** The Phase 1 README states that Phase 2 is not part of the snapshot and gives Phase 1 test counts and pins; submission guidelines require a README that explains execution, dependencies and reproduction for the delivered code.
* **Effect on experiments.** None: documentation only. No code, protocol, selected-window manifest, model or result file was touched.
* **Other additions in the same change.** `requirements-phase2-py311.txt` (identical pins except `shap`), `scripts/show_headline_results.py` (read-only summary of saved CSVs), `.gitattributes` (`* -text`, so Git never rewrites line endings and protected-file hashes survive a Windows clone), and one sentence in `docs/phase2_README.md` naming this change.
