"""Phase 1 artifacts are unchanged and the frozen pipeline is used read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.phase2 import frozen as F

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ADFA-LD"


def test_protected_phase1_files_unchanged():
    out = F.verify_protected()
    assert out["n_files_verified"] >= 100 and not out["changed"]


def test_tampering_is_detected(tmp_path):
    rec = json.loads(F.BASELINE_HASHES.read_text())
    k = next(iter(rec["sha256"]))
    rec["sha256"][k] = "0" * 64
    p = tmp_path / "b.json"; p.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="modified"):
        F.verify_protected(p)


def test_frozen_hashes_in_protocol_match_phase1_constants():
    import yaml
    from src import phase3c
    P = yaml.safe_load((ROOT / "docs/phase2_protocol.yaml").read_text())
    assert P["provenance"]["manifest_sha256"] == phase3c.EXPECTED_MANIFEST_SHA
    assert P["provenance"]["assignment_sha256"] == phase3c.EXPECTED_ASSIGNMENT_SHA
    assert P["perturbation"]["budget_b"] == 3 and P["stability"]["k"] == 5 and P["statistics"]["bootstrap_replicates"] >= 1000
    assert P["conditions"]["compute_matched"]["num_samples"] == 20 * P["lime"]["primary_num_samples"]


@pytest.mark.skipif(not RAW.exists(), reason="raw ADFA-LD not present")
def test_frozen_load_reproduces_phase1_validation_predictions_and_is_unmodified():
    fz = F.load_frozen(splits=("validation",))
    assert fz.X["validation"].shape == (13539, 150) and fz.X_train.shape == (61553, 150)
    assert fz.provenance["validation_max_abs_prediction_diff"] <= 1e-10
    assert len(fz.names) == 150 and np.isfinite(fz.train_std).all() and (fz.train_std > 0).all()
    F.verify_protected()                                              # loading must not have touched any protected file


def test_append_only_exception_allows_append_but_not_edits(tmp_path):
    rec = json.loads(F.BASELINE_HASHES.read_text())
    assert "AI_USAGE.md" in rec["append_only_authorized"]
    out = F.verify_protected()
    assert out["append_only_verified"] in ([], ["AI_USAGE.md"])
    # simulate an edit of the ORIGINAL bytes: the baseline hash must then fail
    rec2 = json.loads(json.dumps(rec)); rec2["sha256"]["AI_USAGE.md"] = "0" * 64
    p = tmp_path / "b.json"; p.write_text(json.dumps(rec2))
    with pytest.raises(RuntimeError, match="modified"):
        F.verify_protected(p)
