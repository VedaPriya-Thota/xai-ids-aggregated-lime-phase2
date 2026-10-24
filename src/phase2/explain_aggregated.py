"""Seeded single-run and multi-run aggregated LIME on the frozen baseline (vendored stand-in only)."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

N_FEATURES = 150
BASE_SEED = 42


def seed_for(stream: str, window_id: str, run: int, base: int = BASE_SEED) -> int:
    """Deterministic 32-bit seed for (stream, window, run); independent across all three."""
    h = hashlib.sha256(f"{base}|{stream}|{window_id}|{int(run)}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class LimeContext:
    X_train: np.ndarray                      # frozen transformed TRAIN matrix (explainer background)
    predict_attack: Callable[[np.ndarray], np.ndarray]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_frozen(cls, fz) -> "LimeContext":
        return cls(fz.X_train, fz.predict_attack, fz.train_mean, fz.train_std)


def run_lime(ctx: LimeContext, row: np.ndarray, seed: int, num_samples: int) -> dict:
    """One fresh vendored-LIME explanation. Returns the full signed 150-vector indexed by feature."""
    from lime.lime_tabular import LimeTabularExplainer      # resolved after assert_lime_provenance()

    row = np.asarray(row, dtype=float).reshape(-1)
    if row.shape[0] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {row.shape[0]}")
    t0 = time.perf_counter()
    ex = LimeTabularExplainer(ctx.X_train, mode="classification", discretize_continuous=False, random_state=int(seed))
    e = ex.explain_instance(row, ctx.predict_attack, labels=(1,), num_features=N_FEATURES, num_samples=int(num_samples))
    runtime = time.perf_counter() - t0
    pairs = e.local_exp[1]
    idx = np.array([i for i, _ in pairs])
    if len(pairs) != N_FEATURES or len(set(idx.tolist())) != N_FEATURES:
        raise RuntimeError("LIME did not return one coefficient per feature")
    coef = np.zeros(N_FEATURES)
    coef[idx] = [w for _, w in pairs]                      # align by feature index, never by display string
    return {"coef": coef, "intercept": e.intercept[1], "r2": e.score[1], "local_pred": e.local_pred[1],
            "runtime_s": runtime, "num_samples": int(num_samples), "seed": int(seed),
            "kernel_width": float(ex.kernel_width)}


def lime_ess(ctx: LimeContext, row: np.ndarray, seed: int, num_samples: int, kernel_width: float | None = None) -> float:
    """Effective sample size (sum w)^2 / sum w^2 of the kernel weights for the exact samples a run draws.

    The stand-in does not expose its weights, so they are re-derived from the same RNG stream and the
    stand-in's published formula. tests/test_seed_reproducibility.py verifies that these re-derived weights
    reproduce the stand-in's coefficients exactly.
    """
    row = np.asarray(row, dtype=float).reshape(-1)
    kw = kernel_width if kernel_width is not None else np.sqrt(ctx.X_train.shape[1]) * 0.75
    z = np.random.RandomState(int(seed)).normal(size=(int(num_samples), row.shape[0]))
    z = ctx.mean + z * ctx.std
    z[0] = row
    dist = np.linalg.norm((z - row) / ctx.std, axis=1)
    w = np.sqrt(np.exp(-(dist ** 2) / (kw ** 2)))
    return float(w.sum() ** 2 / (w ** 2).sum())


def run_stream(ctx: LimeContext, row: np.ndarray, window_id: str, stream: str, n_runs: int, num_samples: int) -> list[dict]:
    return [run_lime(ctx, row, seed_for(stream, window_id, r), num_samples) for r in range(n_runs)]


def aggregate_signed(coefs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean SIGNED coefficient per feature over all runs, and the ranking by |mean| (stable tie-break by index).

    The complete 150-vector of every run is averaged; top-k lists are never averaged.
    """
    c = np.asarray(coefs, dtype=float)
    if c.ndim != 2 or c.shape[1] != N_FEATURES:
        raise ValueError(f"coefs must have shape (runs, {N_FEATURES}); got {c.shape}")
    mean = c.mean(axis=0)
    order = np.argsort(-np.abs(mean), kind="stable")
    return mean, order
