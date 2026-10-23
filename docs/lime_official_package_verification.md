# Official LIME Package Verification Plan

## Current status

The official `lime==0.2.0.1` package could not be installed for the completed Phase 3B/3C runs because the execution environment had no usable internet/DNS access. The completed explanations therefore use the repository's **vendored LIME-compatible local-surrogate reconstruction**, not the official external LIME package.

The local implementation follows the intended workflow used for this project: seeded Gaussian perturbation around the training mean, exponential locality weighting, a weighted Ridge local surrogate, and extraction of the top 10 feature weights (details: `phase3b_xai_explanations.md`). It implements a subset of the `LimeTabularExplainer` API, has not been compared line-by-line or numerically with the official package, and must not be represented as the official `lime` package.

## Future verification procedure

The official-package check should be performed separately in an internet-enabled environment and must not overwrite the frozen Phase 3B artifacts.

1. Clone or check out the frozen Phase 1 repository state/tag.
2. Place the same raw ADFA-LD archive outside Git, preserving the recorded archive SHA-256 and extracted layout.
3. Verify the dataset/manifest and frozen Phase 3A artifact hashes before running anything.
4. Install the repository's exact requirements and then install the official package explicitly:

   `pip install lime==0.2.0.1`

   **Shadowing warning:** the repository contains a vendored package named `lime/` at its root. Whenever the repo root precedes site-packages on `sys.path` (e.g. `PYTHONPATH=.` or `python -m pytest`), `import lime` resolves to the vendored code even if the official package is installed. The verification script must therefore run from outside the repo root (or the vendored directory must be renamed in the verification checkout), and should print `lime.__file__` and `lime.__version__` to prove which implementation ran.

5. Run a **separate official-LIME verification script** using:
   - the same 20 selected test IDs;
   - seed **42**;
   - the same LIME settings, including **5,000 samples**;
   - the same frozen model, TF-IDF vectorizer, and chi-square selector.
6. Write all official-LIME outputs to a new verification directory. Do not overwrite `results/experiments/phase3b_xai_explanations/` or the Phase 3C outputs.
7. Compare the vendored and official implementations using at least:
   - top-10 feature overlap (Jaccard);
   - signed-weight agreement on overlapping features;
   - downstream perturbation output sensitivity under the same intervention policy.
8. Record package versions, Python version, platform, hashes, and the verification command in a new reproducibility record.

## CLI note

The current repository does **not** provide a verified `--require-official-lime` option. A future verification script or flag must be implemented and tested before such a command is documented as an available CLI.

## Interpretation rule

If the official and vendored implementations differ, the comparison should be reported as an implementation/environment verification result. It should not be used to retroactively change the frozen Phase 3B/3C results or to claim that the original paper has been exactly reproduced.
