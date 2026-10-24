"""Phase 2 orchestration.  Usage (from the repository root):

    python -m src.phase2.run_phase2 verify
    python -m src.phase2.run_phase2 pilot            # validation split, 10 windows/group
    python -m src.phase2.run_phase2 final            # held-out test split; refuses to run twice
    python -m src.phase2.run_phase2 analyze --root results/phase2            # tables + figures

Per-window results are cached, so interrupted runs resume without changing any value.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PROJECT_ROOT
from src.phase2 import evaluate_perturb as ev
from src.phase2 import metrics_stability as ms
from src.phase2.explain_aggregated import LimeContext, aggregate_signed, lime_ess, run_lime, seed_for
from src.phase2.frozen import file_key, load_frozen, outcome_group, verify_protected
from src.phase2.perturb_raw_sequence import actionable_mask
from src.phase2.sampling import GROUP_ORDER, eligibility_table, select_windows, verify_cluster_cap

PROTOCOL_PATH = PROJECT_ROOT / "docs/phase2_protocol.yaml"
K_STAB = 5


def protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ cached LIME streams
def get_runs(ctx: LimeContext, row: np.ndarray, wid: str, stream: str, n_runs: int, n_samples: int, root: Path) -> dict:
    p = root / "lime_runs" / f"{file_key(wid)}_{stream}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    res = [run_lime(ctx, row, seed_for(stream, wid, r), n_samples) for r in range(n_runs)]
    out = {"coef": np.stack([r["coef"] for r in res]), "r2": np.array([r["r2"] for r in res]),
           "runtime_s": np.array([r["runtime_s"] for r in res]), "seeds": np.array([r["seed"] for r in res]),
           "local_pred": np.array([r["local_pred"] for r in res]), "intercept": np.array([r["intercept"] for r in res]),
           "num_samples": np.array([n_samples])}
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    return out


class Experiment:
    def __init__(self, split: str, root: Path, targets: dict, scan_all=("FP", "FN"), full_ess_audit: bool = True):
        self.P = protocol()
        self.split, self.root, self.targets = split, Path(root), targets
        self.scan_all, self.full_ess_audit = tuple(scan_all), full_ess_audit
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_primary = self.P["lime"]["primary_num_samples"]
        self.n_runs = self.P["conditions"]["aggregated"]["runs"]
        self.n_cm = self.P["conditions"]["compute_matched"]["num_samples"]
        assert self.n_cm == self.n_runs * self.n_primary, "compute-matched budget must equal 20 ordinary runs"
        self.n_draws = self.P["controls"]["R2_primary"]["draws"]
        self.fz = load_frozen(splits=(split,))
        self.ctx = LimeContext.from_frozen(self.fz)
        w = self.fz.windows[self.fz.windows.split == split].reset_index(drop=True)
        self.X, self.p = self.fz.X[split], self.fz.prob[split]
        w["pred"] = (self.p >= 0.5).astype(int)
        w["group"] = [outcome_group(l, q) for l, q in zip(w.label, w.pred)]
        w["nnz"] = (self.X > 0).sum(1)
        self.cands = w
        self._act: dict = {}
        self._ess = json.loads((self.root / "ess_cache.json").read_text()) if (self.root / "ess_cache.json").exists() else {}

    # --- helpers
    def row_of(self, r):
        return self.X[int(r["split_index"])]

    def window_of(self, r):
        return self.fz.raw[(r["trace_id"], int(r["start"]))]

    def actionable(self, r):
        wid = r["window_id"]
        if wid not in self._act:
            self._act[wid] = actionable_mask(self.window_of(r), self.row_of(r) > 0, self.fz.names, self.fz.freq)
        return self._act[wid]

    def ess_fn(self, r) -> float:
        wid = r["window_id"]
        if wid not in self._ess:
            self._ess[wid] = lime_ess(self.ctx, self.row_of(r), seed_for("agg", wid, 0), self.n_primary)
        return self._ess[wid]

    def pool_fn(self, r) -> int:
        runs = get_runs(self.ctx, self.row_of(r), r["window_id"], "agg", self.n_runs, self.n_primary, self.root)
        agg, _ = aggregate_signed(runs["coef"])
        pl = ev.pools(agg, self.row_of(r) > 0, self.actionable(r), int(r["pred"]))
        return len(pl["r2"])

    # --- stage 1: sampling
    def build_sample(self):
        t0 = time.time()
        sel, audit = select_windows(self.cands, self.ess_fn, self.pool_fn, self.targets,
                                    cap=self.P["sampling"]["global_cluster_cap"], seed=self.P["sampling"]["order_seed"],
                                    scan_all=self.scan_all, full_ess_audit=self.full_ess_audit)
        (self.root / "ess_cache.json").write_text(json.dumps(self._ess))
        elig = eligibility_table(audit, sel, self.targets, self.scan_all)
        elig.to_csv(self.root / "eligibility_report.csv", index=False)
        audit.to_csv(self.root / "eligibility_audit_all_windows.csv.gz", index=False)
        sel = sel.merge(self.cands[["window_id", "start", "label", "pred", "split_index"]], on="window_id")
        sel["split"] = self.split
        sel["predicted_attack_probability"] = self.p[sel.split_index.to_numpy()]
        sel["order_seed"] = self.P["sampling"]["order_seed"]
        sel.to_csv(self.root / "selected_windows.csv", index=False)
        verify_cluster_cap(sel, self.P["sampling"]["global_cluster_cap"])
        print(f"[sampling] {len(sel)} windows selected in {time.time()-t0:.0f}s\n{elig.to_string(index=False)}", flush=True)
        return sel

    # --- stage 2: per window
    def process_window(self, r) -> dict:
        wid, row, window = r["window_id"], self.row_of(r), self.window_of(r)
        N, R = self.n_primary, self.n_runs
        agg_runs = get_runs(self.ctx, row, wid, "agg", R, N, self.root)
        single = get_runs(self.ctx, row, wid, "single", 1, N, self.root)
        rep_a = get_runs(self.ctx, row, wid, "repA", R, N, self.root)
        rep_b = get_runs(self.ctx, row, wid, "repB", R, N, self.root)
        cm = get_runs(self.ctx, row, wid, "cm", 2, self.n_cm, self.root)
        present = row > 0
        agg, _ = aggregate_signed(agg_runs["coef"])
        a_mean, _ = aggregate_signed(rep_a["coef"])
        b_mean, _ = aggregate_signed(rep_b["coef"])
        n_present = int(present.sum())
        st = {"window_id": wid, "group": r["group"], "cluster": r["cluster"], "trace_id": r["trace_id"], "n_present": n_present}
        pw = ms.pairwise_single_run_stability(agg_runs["coef"], K_STAB)
        pwp = ms.pairwise_single_run_stability(agg_runs["coef"], K_STAB, present)
        st.update({"single_jaccard": pw["jaccard"], "single_spearman": pw["spearman"], "single_sign_consistency": pw["sign_consistency"],
                   "single_jaccard_present": pwp["jaccard"], "single_spearman_present": pwp["spearman"]})
        st.update({"rep_jaccard": ms.jaccard_topk(a_mean, b_mean, K_STAB), "rep_spearman": ms.spearman_abs(a_mean, b_mean),
                   "rep_sign_consistency": ms.sign_consistency_topk(a_mean, b_mean, K_STAB),
                   "rep_jaccard_present": ms.jaccard_topk(a_mean, b_mean, K_STAB, present),
                   "rep_spearman_present": ms.spearman_abs(a_mean, b_mean, present),
                   "cm_jaccard": ms.jaccard_topk(cm["coef"][0], cm["coef"][1], K_STAB),
                   "cm_spearman": ms.spearman_abs(cm["coef"][0], cm["coef"][1]),
                   "cm_sign_consistency": ms.sign_consistency_topk(cm["coef"][0], cm["coef"][1], K_STAB),
                   "agg_sign_agreement_with_runs": ms.sign_agreement_with_aggregate(agg_runs["coef"], agg, K_STAB),
                   "chance_jaccard_150": ms.expected_random_jaccard(150, K_STAB),
                   "chance_jaccard_present": ms.expected_random_jaccard(n_present, min(K_STAB, n_present)) if n_present >= 1 else np.nan,
                   "top5_present_share_agg": len(ms.top_k(agg, K_STAB) & set(np.flatnonzero(present).tolist())) / K_STAB,
                   "r2_single_runs_mean": float(agg_runs["r2"].mean()), "r2_single": float(single["r2"][0]), "r2_cm": float(cm["r2"][0]),
                   "ess_agg_run0": self.ess_fn(r), "ess_single": lime_ess(self.ctx, row, int(single["seeds"][0]), N),
                   "ess_cm": lime_ess(self.ctx, row, int(cm["seeds"][0]), self.n_cm),
                   "runtime_single_s": float(single["runtime_s"].sum()), "runtime_agg_s": float(agg_runs["runtime_s"].sum()),
                   "runtime_rep_a_s": float(rep_a["runtime_s"].sum()), "runtime_rep_b_s": float(rep_b["runtime_s"].sum()),
                   "runtime_cm_primary_s": float(cm["runtime_s"][0]), "runtime_cm_stability_extra_s": float(cm["runtime_s"][1]),
                   "samples_single": N, "samples_agg_total": N * R, "samples_cm": self.n_cm,
                   "model_calls_single": N, "model_calls_agg": N * R, "model_calls_cm_primary": self.n_cm,
                   "model_calls_replicates_extra": 2 * N * R})
        # perturbation
        p0 = float(self.p[int(r["split_index"])])
        l0 = float(ev.attack_logit(self.fz.model, row[None, :])[0])
        perms = ev.priority_permutations(wid, self.n_draws)
        act = self.actionable(r)
        methods = {"single": single["coef"][0], "agg": agg, "cm": cm["coef"][0]}
        rows, draws = [], []
        for m, coef in methods.items():
            t0 = time.perf_counter()
            res = ev.evaluate_method(self.fz, wid, window, row, p0, l0, coef, act, perms, self.fz.names)
            base = {"window_id": wid, "group": r["group"], "cluster": r["cluster"], "trace_id": r["trace_id"], "method": m,
                    "pred_attack": int(p0 >= 0.5), "prob_before": p0, "logit_before": l0, "status": res["status"],
                    "pool_r2": res["pool_r2"], "pool_r1": res["pool_r1"], "guided_features": res.get("guided_features", "")}
            if "guided" in res:
                rows.append({**base, "arm": "guided", "n_draws": 1, **_flat(res["guided"])})
            for arm, key, feats in (("r2", "r2_draws", res.get("r2_draw_features")), ("r1", "r1_draws", None)):
                if key in res:
                    d = res[key]
                    rows.append({**base, "arm": arm, "n_draws": len(d), **_mean_rows(d)})
                    for i, x in enumerate(d):
                        draws.append({"window_id": wid, "method": m, "arm": arm, "draw": i, "seed": seed_for("rand", wid, i),
                                      "features": feats[i] if feats else "", **{k: x[k] for k in ("direction_aware_drop", "attack_logit_drop", "prob_change", "flip_any", "edit_success")}})
            st[f"perturb_runtime_{m}_s"] = time.perf_counter() - t0
        return {"stability": st, "perturbation": rows, "draws": draws}

    def run(self, sel: pd.DataFrame):
        wdir = self.root / "window_results"
        wdir.mkdir(exist_ok=True)
        t0 = time.time()
        for i, r in enumerate(sel.to_dict("records")):
            p = wdir / f"{file_key(r['window_id'])}.pkl"
            if p.exists():
                continue
            full = self.cands[self.cands.window_id == r["window_id"]].iloc[0].to_dict()
            pickle.dump(self.process_window(full), open(p, "wb"))
            if i % 5 == 0:
                print(f"[run] {i+1}/{len(sel)} windows, {time.time()-t0:.0f}s", flush=True)
        self.assemble(sel)

    def assemble(self, sel: pd.DataFrame):
        assemble_root(self.root, sel)


def assemble_root(root: Path, sel: pd.DataFrame) -> None:
    """Rebuild the CSV outputs from the cached per-window results (no experiment is recomputed)."""
    root = Path(root)
    st, pr, dr = [], [], []
    for r in sel.to_dict("records"):
        o = pickle.load(open(root / "window_results" / f"{file_key(r['window_id'])}.pkl", "rb"))
        st.append(o["stability"]); pr += o["perturbation"]; dr += o["draws"]
    S = pd.DataFrame(st)
    S.to_csv(root / "stability_results.csv", index=False)
    P = pd.DataFrame(pr)
    P["edit_success"] = P["edit_success"].astype(float)
    P.to_csv(root / "perturbation_results.csv", index=False)
    D = pd.DataFrame(dr)
    D["edit_success"] = D["edit_success"].astype(float)
    with gzip.open(root / "random_draw_log.csv.gz", "wt") as f:
        D.to_csv(f, index=False)
    rt = S[["window_id", "group"] + [c for c in S.columns if c.startswith("runtime_") or c.startswith("model_calls_") or c.startswith("samples_")]].copy()
    rt.to_csv(root / "runtime_results.csv", index=False)
    print(f"[assemble] wrote {len(S)} windows, {len(pr)} perturbation rows, {len(dr)} draw rows", flush=True)


def _flat(x: dict) -> dict:
    keys = ["attack_logit_drop", "direction_aware_drop", "prob_change", "prob_after", "logit_after", "flip_any", "flip_attack_to_normal",
            "edit_success", "n_features_requested", "n_features_edited", "n_positions_edited", "failure_reasons"]
    out = {k: x[k] for k in keys}
    out["edit_success"] = float(out["edit_success"])        # guided: 0/1; random arms: fraction of draws fully applied
    return out


def _mean_rows(d: list) -> dict:
    num = ["attack_logit_drop", "direction_aware_drop", "prob_change", "prob_after", "logit_after", "flip_any", "flip_attack_to_normal",
           "n_features_edited", "n_positions_edited"]
    out = {k: float(np.mean([x[k] for x in d])) for k in num}
    out["edit_success"] = float(np.mean([x["edit_success"] for x in d]))     # fraction of draws fully applied
    out["n_features_requested"] = d[0]["n_features_requested"]
    out["failure_reasons"] = f"{sum(1 for x in d if x['failure_reasons'])}/{len(d)} draws with failures"
    return out


# ------------------------------------------------------------------ commands
def cmd_verify(_):
    from src.phase2.frozen import assert_lime_provenance
    print(json.dumps({"lime": assert_lime_provenance(), "protected": verify_protected()}, indent=1))


def cmd_pilot(a):
    P = protocol()
    root = PROJECT_ROOT / P["outputs"]["pilot_root"]
    t0 = time.time()
    exp = Experiment("validation", root, {g: P["splits"]["pilot"]["windows_per_group"] for g in GROUP_ORDER},
                     scan_all=(), full_ess_audit=False)        # pilot: lazy eligibility evaluation (audit is partial by design)
    sel = exp.build_sample()
    t1 = time.time()
    exp.run(sel)
    n = len(sel)
    # reproducibility: recompute two windows' single-run explanation from scratch and compare bit-for-bit
    rep = []
    for r in sel.head(2).to_dict("records"):
        full = exp.cands[exp.cands.window_id == r["window_id"]].iloc[0]
        again = run_lime(exp.ctx, exp.row_of(full), seed_for("single", r["window_id"], 0), exp.n_primary)["coef"]
        rep.append(bool(np.array_equal(again, get_runs(exp.ctx, exp.row_of(full), r["window_id"], "single", 1, exp.n_primary, root)["coef"][0])))
    S = pd.read_csv(root / "stability_results.csv")
    per_window_s = float(S[["runtime_single_s", "runtime_agg_s", "runtime_rep_a_s", "runtime_rep_b_s", "runtime_cm_primary_s", "runtime_cm_stability_extra_s"]].sum(axis=1).mean())
    summary = {"windows": n, "per_group": sel.group.value_counts().to_dict(), "bit_identical_rerun": rep,
               "mean_lime_runtime_per_window_s": per_window_s, "sampling_seconds": t1 - t0, "run_seconds": time.time() - t1,
               "projected_final_lime_seconds_160_windows": per_window_s * 160,
               "protected_files": verify_protected(), "python": platform.python_version()}
    (root / "pilot_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


def cmd_final(a):
    P = protocol()
    root = PROJECT_ROOT / P["outputs"]["final_root"]
    lock = root / "FINAL_RUN_LOCK.json"
    if lock.exists() and not a.force_rerun:
        sys.exit("Final held-out run already started (FINAL_RUN_LOCK.json). Refusing a second run. Use --force-rerun only to RESUME the same frozen run.")
    pilot = PROJECT_ROOT / P["outputs"]["pilot_root"] / "pilot_summary.json"
    if not pilot.exists():
        sys.exit("Pilot has not been run (results/phase2/pilot/pilot_summary.json missing).")
    exp = Experiment("test", root, {g: P["sampling"]["targets"][g] for g in GROUP_ORDER})
    sel = exp.build_sample()
    freeze = {"protocol_sha256": sha256_file(PROTOCOL_PATH), "selected_windows_sha256": sha256_file(root / "selected_windows.csv"),
              "code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted((PROJECT_ROOT / "src/phase2").glob("*.py"))},
              "command": " ".join(sys.argv), "n_selected": len(sel), "protected_files": verify_protected(),
              "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not lock.exists():
        lock.write_text(json.dumps(freeze, indent=1))
        (root / "freeze_manifest.json").write_text(json.dumps(freeze, indent=1))
    else:                                                       # resuming: the sample must be identical to the frozen one
        old = json.loads((root / "freeze_manifest.json").read_text())
        if old["selected_windows_sha256"] != freeze["selected_windows_sha256"] or old["protocol_sha256"] != freeze["protocol_sha256"]:
            sys.exit("Resume refused: protocol or selected-window manifest differs from the frozen versions.")
    exp.run(sel)
    verify_protected()
    print("[final] complete; protected Phase 1 files unchanged")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("pilot").set_defaults(fn=cmd_pilot)
    f = sub.add_parser("final"); f.add_argument("--force-rerun", action="store_true"); f.set_defaults(fn=cmd_final)
    asm = sub.add_parser("assemble"); asm.add_argument("--root", required=True)
    asm.set_defaults(fn=lambda a: assemble_root(PROJECT_ROOT / a.root, pd.read_csv(PROJECT_ROOT / a.root / "selected_windows.csv")))
    an = sub.add_parser("analyze"); an.add_argument("--root", default="results/phase2"); an.set_defaults(fn=lambda a: __import__("src.phase2.analyze", fromlist=["run"]).run(a.root))
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
