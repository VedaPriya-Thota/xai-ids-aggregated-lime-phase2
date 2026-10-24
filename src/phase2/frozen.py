"""Read-only access to the frozen Phase 1 baseline plus provenance and protection guards."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import phase3a, phase3c
from src.config import PROJECT_ROOT, load_config

SEED = 42
N_FEATURES = 150
WINDOW = 30
P3A_REL = Path("results/experiments/phase3a_strict_reconstructed_mlp")
BASELINE_HASHES = PROJECT_ROOT / "results/phase2/protected_hashes_baseline.json"
LIME_VERSION = "vendored-compat-0.2.0.1"
# Full SHA-256 of the vendored stand-in (verified against the actual files; the repo documents none).
LIME_SHA256 = {
    "lime/lime_tabular.py": "f18d90e8a7a2d3eaa8d8df2aece4eb7691f0c4249ca03bfb44ebd8f5218cf228",
    "lime/__init__.py": "c8b270015bc00b3c8b2e6bfde98bda3641310859ac894befc12d45b005305098",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def assert_lime_provenance() -> dict[str, Any]:
    """Fail fast unless `lime` is the vendored stand-in from this repository (never official LIME)."""
    import lime  # noqa: WPS433 (import inside function: resolve at call time, after sys.path is final)

    lime_file = Path(lime.__file__).resolve()
    expected_dir = (PROJECT_ROOT / "lime").resolve()
    if expected_dir not in lime_file.parents:
        raise RuntimeError(f"`lime` resolves to {lime_file}, not the vendored stand-in under {expected_dir}")
    if getattr(lime, "__version__", None) != LIME_VERSION:
        raise RuntimeError(f"lime.__version__={getattr(lime, '__version__', None)!r}, expected {LIME_VERSION!r}")
    actual = {rel: sha256_bytes((PROJECT_ROOT / rel).read_bytes()) for rel in LIME_SHA256}
    if actual != LIME_SHA256:
        raise RuntimeError(f"Vendored LIME hash mismatch: {actual}")
    return {"path": str(lime_file), "version": LIME_VERSION, "sha256": actual}


def verify_protected(baseline: Path = BASELINE_HASHES) -> dict[str, Any]:
    """Every file recorded at baseline must still exist with an identical SHA-256."""
    rec = json.loads(Path(baseline).read_text())["sha256"]
    changed, missing = [], []
    for rel, h in rec.items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif sha256_bytes(p.read_bytes()) != h:
            changed.append(rel)
    if changed or missing:
        raise RuntimeError(f"Protected Phase 1 files modified={changed} missing={missing}")
    return {"n_files_verified": len(rec), "changed": [], "missing": []}


_ACT = {
    "relu": lambda a: np.maximum(a, 0.0),
    "identity": lambda a: a,
    "tanh": np.tanh,
    "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)),
}


def attack_logit(model, X: np.ndarray) -> np.ndarray:
    """Exact pre-sigmoid output of the frozen binary MLP (no probability saturation)."""
    if getattr(model, "out_activation_", None) != "logistic" or len(model.coefs_[-1][0]) != 1:
        raise RuntimeError("Expected a binary MLPClassifier with a single logistic output unit")
    a = np.asarray(X, dtype=float)
    for w, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        a = _ACT[model.activation](a @ w + b)
    return (a @ model.coefs_[-1] + model.intercepts_[-1]).ravel()


def window_id(trace_id: str, start: int) -> str:
    return f"{trace_id}#{int(start)}"


def file_key(wid: str) -> str:
    return hashlib.sha1(wid.encode()).hexdigest()[:16]


@dataclass
class Frozen:
    cfg: dict
    manifest: pd.DataFrame
    vec: Any
    sel: Any
    model: Any
    names: list
    windows: pd.DataFrame            # all splits: window_id, trace_id, start, label, split, split_index, cluster
    raw: dict                        # (trace_id, start) -> int64[30]
    X: dict = field(default_factory=dict)       # split -> float[n,150]
    prob: dict = field(default_factory=dict)    # split -> attack probability (recomputed, verified)
    freq: dict = field(default_factory=dict)    # syscall frequency in train-normal windows (Phase 1 policy)
    train_mean: np.ndarray | None = None
    train_std: np.ndarray | None = None
    X_train: np.ndarray | None = None
    hashes: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def predict_attack(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(X, float))[:, 1]


def load_frozen(splits=("validation",), raw_data_dir: str = "data/raw/ADFA-LD", tol: float = 1e-10) -> Frozen:
    """Verify and load the frozen baseline. Nothing is fitted; anything inconsistent raises."""
    provenance = {"lime": assert_lime_provenance(), "protected": verify_protected()}
    verified, hashes = phase3c.verify_inputs()
    cfg = load_config("configs/baseline.yaml")
    manifest = verified["manifest"]
    calls, _root = phase3a._verify_manifest_against_raw(manifest, cfg)
    texts, labels, windows = phase3a._window_texts(calls, manifest)
    windows = windows.reset_index(drop=True).copy()
    windows["split_index"] = windows.groupby("split").cumcount()
    windows["window_id"] = [window_id(t, s) for t, s in zip(windows.trace_id, windows.start)]
    clusters = {t: "C" + sha256_bytes(np.asarray(c, dtype=np.int64).tobytes())[:12] for t, c in calls.items()}
    windows["cluster"] = windows.trace_id.map(clusters)

    raw = {}
    for t, seq in calls.items():
        seq = np.asarray(seq, dtype=np.int64)
        for start in range(0, len(seq) - 29, 30):
            raw[(t, int(start))] = seq[start:start + 30].copy()
    # Gate: raw windows must tokenise to exactly the strings the frozen vectorizer saw in Phase 1.
    for split in ("train", "validation", "test"):
        w = windows[windows.split == split]
        for i, (t, s) in enumerate(zip(w.trace_id, w.start)):
            if phase3c.tokens(raw[(t, int(s))]) != texts[split][i]:
                raise RuntimeError(f"Raw window / frozen text mismatch at {split}[{i}] {t}#{s}")

    vec, sel, model = verified["vectorizer"], verified["selector"], verified["model"]
    names_df = verified["feature_names"].sort_values("model_column_index")
    names = names_df["feature_name"].tolist()
    if len(names) != N_FEATURES:
        raise RuntimeError("Unexpected feature dimension")
    fz = Frozen(cfg=cfg, manifest=manifest, vec=vec, sel=sel, model=model, names=names, windows=windows, raw=raw,
                hashes=hashes, provenance=provenance)

    def feats(split):
        Z = sel.transform(vec.transform(texts[split]))
        return np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, dtype=float)

    fz.X_train = feats("train")
    if fz.X_train.shape[1] != N_FEATURES:
        raise RuntimeError("Unexpected train feature dimension")
    fz.train_mean, fz.train_std = fz.X_train.mean(0), fz.X_train.std(0)
    if (fz.train_std == 0).any():
        raise RuntimeError("Zero-variance training feature: ESS re-derivation would be invalid")
    base = PROJECT_ROOT / P3A_REL
    for split in splits:
        X = feats(split)
        p = fz.predict_attack(X)
        saved = pd.read_csv(base / f"{split}_predictions.csv").sort_values("window_index")
        ref = saved.predicted_attack_probability.to_numpy()
        if len(ref) != len(p) or np.abs(p - ref).max() > tol:
            raise RuntimeError(f"{split}: frozen predictions not reproduced within {tol}")
        if not ((p >= 0.5).astype(int) == saved.predicted_label.to_numpy()).all():
            raise RuntimeError(f"{split}: predicted labels differ from frozen predictions")
        fz.X[split], fz.prob[split] = X, p
        provenance[f"{split}_max_abs_prediction_diff"] = float(np.abs(p - ref).max())
    # Replacement policy frequencies: Phase 1 definition (train-normal windows), via Phase 1 code.
    fz.freq = phase3c.normal_training_frequency(manifest, raw)
    return fz


def outcome_group(label: int, pred: int) -> str:
    return {(1, 1): "TP", (1, 0): "FN", (0, 0): "TN", (0, 1): "FP"}[(int(label), int(pred))]
