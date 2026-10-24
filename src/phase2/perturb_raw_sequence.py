"""Raw-syscall perturbation using the ESTABLISHED Phase 1 edit rules (src/phase3c.guided_edits).

Edit semantics (inherited unchanged from Phase 1, documented in docs/phase3c_perturbation_extension.md):
  * A selected TF-IDF feature is a bigram "a_b". It maps to EVERY adjacent occurrence (a, b) in the 30-call window.
  * The SECOND call (b) of each occurrence is replaced by the most frequent syscall in train-normal windows
    (the next most frequent if that equals the original or would recreate the same bigram). Position 0 is never edited.
  * Each raw position is edited at most once; later features cannot re-edit an already edited position.
    Because editing a position also changes its neighbouring bigrams, edits can cascade; a later feature whose
    occurrence was consumed by an earlier edit is logged as not actionable (reason recorded), never skipped silently.
  * The window length (30) is preserved; no padding, insertion or deletion is performed.
  * The edited window is re-tokenised and passed through the frozen vectorizer, selector and MLP.
Guided and random-control edits call the SAME function, so edit semantics are identical across arms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.phase3c import guided_edits  # Phase 1 implementation, reused (not re-implemented)


@dataclass
class EditResult:
    sequence: np.ndarray
    n_requested: int
    n_features_edited: int
    positions_edited: list = field(default_factory=list)
    failures: list = field(default_factory=list)       # [{feature, reason}]
    length_preserved: bool = True
    success_all: bool = False
    changed_calls: int = 0


def occurrences(window: np.ndarray, feature_name: str) -> list:
    a, b = map(int, feature_name.split("_", 1))
    return [i for i in range(len(window) - 1) if int(window[i]) == a and int(window[i + 1]) == b]


def apply_edit(window: np.ndarray, feature_idx, names: list, freq: dict) -> EditResult:
    """Apply Phase 1 raw-sequence edits for the features `feature_idx` (processed in the given priority order)."""
    window = np.asarray(window, dtype=np.int64)
    feats = pd.DataFrame({"rank": range(1, len(feature_idx) + 1), "feature_name": [names[int(i)] for i in feature_idx]})
    edited, actionable, log, edited_feature_names = guided_edits(window, feats, freq)
    if len(edited) != len(window):
        raise RuntimeError("Edit changed the window length")
    failures = []
    for row in log:
        if row["status"] != "edited":
            orig = occurrences(window, row["feature_name"])
            failures.append({"feature": row["feature_name"],
                             "reason": "absent_in_original_window" if not orig else "occurrence_consumed_by_prior_edit"})
    positions = sorted({int(a["position"]) for a in actionable})
    return EditResult(sequence=edited, n_requested=len(feature_idx), n_features_edited=len(edited_feature_names),
                      positions_edited=positions, failures=failures, length_preserved=len(edited) == len(window),
                      success_all=(len(failures) == 0 and len(edited_feature_names) == len(feature_idx)),
                      changed_calls=int((edited != window).sum()))


def actionable_mask(window: np.ndarray, present: np.ndarray, names: list, freq: dict) -> np.ndarray:
    """Present features whose single-feature Phase 1 edit actually changes the raw window."""
    mask = np.zeros(len(names), dtype=bool)
    for j in np.flatnonzero(present):
        r = apply_edit(window, [j], names, freq)
        mask[j] = r.n_features_edited == 1 and r.changed_calls > 0
    return mask
