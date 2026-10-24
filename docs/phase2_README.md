# Phase 2: aggregated LIME vs single-run LIME (run instructions)

Everything is additive: Phase 1 code, artifacts and results are never modified (`python -m src.phase2.run_phase2 verify`
re-hashes all protected files against `results/phase2/protected_hashes_baseline.json`).

## Prerequisites
* Python 3.12, `pip install -r requirements-phase2.txt` (versions actually used; the vendored `lime/` stand-in is used, do **not** install official `lime`).
* Raw ADFA-LD at `data/raw/ADFA-LD/{Training_Data_Master,Validation_Data_Master,Attack_Data_Master}` (Kaggle mirror, archive SHA-256
  `686825e8…a6454`; see `data/README.md`). The loader verifies the manifest, the raw-tree hash and reproduces every saved prediction (<= 1e-10).

## Commands (run from the repository root)
```bash
python -m pytest -q                              # Phase 1 + Phase 2 tests (108 in the reference run)
python -m src.phase2.run_phase2 verify           # provenance + protected-file check
python -m src.phase2.run_phase2 pilot            # VALIDATION split, 10 windows/group, writes results/phase2/pilot/
python -m src.phase2.run_phase2 final            # held-out TEST split, run once; writes results/phase2/ and FINAL_RUN_LOCK.json
python -m src.phase2.run_phase2 analyze --root results/phase2        # tables + figures (no recomputation of experiments)
```
`final` refuses to start a second time (`FINAL_RUN_LOCK.json`) and, if resumed, refuses unless the protocol and selected-window
manifest hashes equal the frozen ones in `results/phase2/freeze_manifest.json`. All per-window work is cached, so an interrupted run resumes
without changing any value. Cost (single core): about 9 s of LIME per window plus the one-off eligibility scan.

## Where things are
| Item | Path |
|---|---|
| Locked protocol / seeds | `docs/phase2_protocol.yaml`, `docs/seeds.txt` |
| Code | `src/phase2/` (`frozen`, `explain_aggregated`, `metrics_stability`, `perturb_raw_sequence`, `evaluate_perturb`, `sampling`, `run_phase2`, `analyze`) |
| Tests | `tests/test_aggregation.py`, `test_stability_metrics.py`, `test_perturbation_controls.py`, `test_seed_reproducibility.py`, `test_sampling_cap.py`, `test_phase1_protection.py` |
| Outputs | `results/phase2/{eligibility_report,selected_windows,stability_results,perturbation_results,runtime_results}.csv`, `*_summary.csv`, `paired_comparisons.csv`, `figures/` |
| Individual LIME runs (audit) | `results/phase2/lime_runs/*.npz` (full signed 150-vectors, seeds, runtimes, R^2) |
| Random-control draw log | `results/phase2/random_draw_log.csv.gz` |
