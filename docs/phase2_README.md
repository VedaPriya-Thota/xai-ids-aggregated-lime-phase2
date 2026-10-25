# Phase 2: aggregated LIME vs single-run LIME (run instructions)

Everything is additive: Phase 1 code, artifacts and results are never modified. The single authorised edit to a Phase 1 file is an append-only disclosure in `AI_USAGE.md`.
`python -m src.phase2.run_phase2 verify` re-hashes all 167 protected files against `results/phase2/protected_hashes_baseline.json` (for `AI_USAGE.md` the original bytes must still hash to the baseline value).

## A. What works WITHOUT the raw dataset (cached artifacts only)
```bash
pip install -r requirements-phase2.txt                 # versions actually used; do NOT install official `lime` (the vendored stand-in in lime/ is hash-guarded)
python -m pytest -q                                    # 109 passed, 1 skipped (the skipped test needs the raw dataset)
python -m src.phase2.run_phase2 verify                 # LIME provenance + protected-file integrity
python -m src.phase2.run_phase2 analyze --root results/phase2   # regenerates all summary tables and the 5 figures from saved per-window results (deterministic)
python scripts/phase2_audit_checks.py                  # re-derives every number in docs/phase2_results.md from the saved CSVs (the raw-data cluster check reports SKIP)
python scripts/phase2_supplementary_analysis.py        # conditional flip rates + signed-vector Spearman, from saved outputs only
```
These checks do **not** prove that the experiment can be re-run end to end: that needs the raw data (section B).

## B. Raw-data prerequisite for a full rerun (`pilot`, `final`, full provenance verification)
* **Source:** public Kaggle mirror of ADFA-LD, <https://www.kaggle.com/datasets/yianqaq/adfa-ld> (`archive.zip`); see `data/README.md`. It is not the paper authors' own generated dataset.
* **Archive checksum:** `sha256sum archive.zip` must print `686825e85e2cd513e5b6462adcc55c3c6d81e8bf2bac24dd912a91fa014a6454`.
* **Location:** the archive also contains unrelated files (`TRAINING/`, `VALIDATE/`, `run.py`, a PDF); only the `ADFA-LD/` folder is used. Place it so that these exist:
  `data/raw/ADFA-LD/Training_Data_Master/`, `data/raw/ADFA-LD/Validation_Data_Master/`, `data/raw/ADFA-LD/Attack_Data_Master/` (5,951 `.txt` traces).
* **Raw-tree checksum (not performed by the loader; run it yourself):**
  ```bash
  python -c "from pathlib import Path; from src.audit_artifacts import _tree_sha256; print(_tree_sha256(Path('data/raw/ADFA-LD').resolve()))"
  # expected 38454c054b5d50a082d52dca2e3b0c1bdafad8dafaf43085b42b966967383a13  (extra or hidden files such as .DS_Store change it)
  ```
* **What the Phase 2 loader (`load_frozen`) does verify:** Phase 1 artifact hashes (manifest, split assignment, confusion matrix), manifest trace IDs / labels / lengths against the raw files, that every raw window tokenises to exactly the text the frozen vectorizer saw, and that every saved prediction of the requested split is reproduced within 1e-10. It does **not** hash the raw tree.
* With the dataset present, `python -m pytest -q` runs the additional frozen-load test (110 total) and `scripts/phase2_audit_checks.py` also recomputes the selected cluster IDs from the raw traces.

```bash
python -m src.phase2.run_phase2 pilot            # VALIDATION split, 10 windows/group, writes results/phase2/pilot/   (only with raw data)
python -m src.phase2.run_phase2 final            # held-out TEST split, run once; writes results/phase2/ and FINAL_RUN_LOCK.json (only with raw data)
```
`final` refuses to start a second time (`FINAL_RUN_LOCK.json`) and, if resumed, refuses unless the protocol and selected-window manifest hashes equal the frozen ones in `results/phase2/freeze_manifest.json`.
All per-window work is cached, so an interrupted run resumes without changing any value. Cost (single core): about 9 s of LIME per window plus a one-off eligibility scan of tens of minutes.
**Do not rerun `final` to "refresh" results: it is the single held-out evaluation.**

## Where things are
| Item | Path |
|---|---|
| Locked protocol / seeds | `docs/phase2_protocol.yaml`, `docs/seeds.txt` |
| Code | `src/phase2/` (`frozen`, `explain_aggregated`, `metrics_stability`, `perturb_raw_sequence`, `evaluate_perturb`, `sampling`, `run_phase2`, `analyze`); audit scripts in `scripts/phase2_*.py` |
| Tests | `tests/test_aggregation.py`, `test_stability_metrics.py`, `test_perturbation_controls.py`, `test_seed_reproducibility.py`, `test_sampling_cap.py`, `test_phase1_protection.py`, `test_phase2_audit.py` |
| Results and report | `results/phase2/*.csv`, `results/phase2/figures/` (5 figures), `docs/phase2_results.md`, `docs/phase2_research_gap.md` |
| Post-freeze changes | `docs/phase2_postfreeze_amendments.md` (the historical `freeze_manifest.json` is deliberately left unedited) |
| Individual LIME runs (audit) | `results/phase2/lime_runs/*.npz` (full signed 150-vectors, seeds, runtimes, R^2) |
| Random-control draw log | `results/phase2/random_draw_log.csv.gz` (note: Phase 1's `.gitignore` ignores `*.gz`, so these files are force-added to Git) |
