"""Eligibility screening and deterministic sampling with a GLOBAL identical-content cluster cap.

Selection is a seeded two-pass greedy per outcome group (groups processed scarcest-first):
pass 1 takes at most one window per cluster, pass 2 allows a second. A window is selected only if it
satisfies ALL eligibility conditions *and* its cluster has not reached the cap across ALL groups combined.
Ineligible windows are never selected and later relabelled; every candidate keeps its stage flags in the audit.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

GROUP_ORDER = ["FP", "FN", "TP", "TN"]
ESS_MIN = 100.0
MIN_PRESENT = 3
MIN_POOL_GUIDED = 3
MIN_POOL_R2 = 4


def order_key(seed: int, window_id: str) -> str:
    return hashlib.sha256(f"{seed}|order|{window_id}".encode()).hexdigest()


def select_windows(cands: pd.DataFrame, ess_fn: Callable[[pd.Series], float], pool_fn: Callable[[pd.Series], int],
                   targets: dict, cap: int = 2, seed: int = 42, scan_all=("FP", "FN"), full_ess_audit: bool = True,
                   ess_min: float = ESS_MIN, min_present: int = MIN_PRESENT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cands columns: window_id, trace_id, group, cluster, nnz. Returns (selected, audit).

    `scan_all` groups get an exhaustive LIME eligibility scan (exact stage counts); `full_ess_audit` computes the
    cheap ESS stage for every window. Neither changes WHICH windows are selected, only how complete the audit is
    (selection evaluates eligibility lazily in the same deterministic order)."""
    c = cands.copy()
    c["order"] = [order_key(seed, w) for w in c.window_id]
    audit = c.set_index("window_id")[["trace_id", "group", "cluster", "nnz"]].copy()
    for col, default in [("nnz_ge3", False), ("ess", np.nan), ("ess_pass", False), ("lime_screened", False),
                         ("pool_r2_agg", -1), ("feature_eligible", False), ("r2_eligible", False),
                         ("blocked_by_cap", False), ("selected", False), ("selection_pass", 0), ("exclusion_reason", "")]:
        audit[col] = default
    audit["nnz_ge3"] = audit.nnz >= min_present
    rows = {r.window_id: r for r in c.itertuples(index=False)}

    def ensure_ess(wid: str) -> None:
        if audit.at[wid, "nnz_ge3"] and np.isnan(audit.at[wid, "ess"]):
            e = ess_fn(pd.Series(rows[wid]._asdict()))
            audit.at[wid, "ess"] = e
            audit.at[wid, "ess_pass"] = bool(e >= ess_min)

    # Cheap stage (no LIME): effective sample size of the kernel weights for windows with >=3 present features.
    if full_ess_audit:
        for wid in audit.index[audit.nnz_ge3]:
            ensure_ess(wid)

    def screen(wid: str) -> None:
        if audit.at[wid, "lime_screened"]:
            return
        n = int(pool_fn(pd.Series(rows[wid]._asdict())))
        audit.at[wid, "lime_screened"] = True
        audit.at[wid, "pool_r2_agg"] = n
        audit.at[wid, "feature_eligible"] = n >= MIN_POOL_GUIDED
        audit.at[wid, "r2_eligible"] = n >= MIN_POOL_R2

    for g in scan_all:                                       # exact stage counts for the scarce groups
        for wid in audit.index[(audit.group == g) & audit.nnz_ge3]:
            ensure_ess(wid)
            if audit.at[wid, "ess_pass"]:
                screen(wid)

    count: dict = defaultdict(int)
    selected: list = []
    for g in GROUP_ORDER:
        target = int(targets.get(g, 0))
        order = c[c.group == g].sort_values("order").window_id.tolist()
        chosen: list = []
        for pass_no, limit in ((1, 1), (2, cap)):
            for wid in order:
                if len(chosen) >= target:
                    break
                if wid in chosen or not audit.at[wid, "nnz_ge3"]:
                    continue
                ensure_ess(wid)
                if not audit.at[wid, "ess_pass"]:
                    continue
                cl = audit.at[wid, "cluster"]
                if count[cl] >= limit:
                    if pass_no == 2 and audit.at[wid, "r2_eligible"]:
                        audit.at[wid, "blocked_by_cap"] = True      # eligible, but its cluster is full
                    continue
                screen(wid)
                if not audit.at[wid, "r2_eligible"]:
                    continue
                chosen.append(wid)
                count[cl] += 1
                audit.at[wid, "selected"] = True
                audit.at[wid, "selection_pass"] = pass_no
        selected += chosen
    reasons = np.select(
        [~audit.nnz_ge3, ~audit.ess_pass, audit.lime_screened & ~audit.feature_eligible,
         audit.lime_screened & audit.feature_eligible & ~audit.r2_eligible, audit.blocked_by_cap & ~audit.selected,
         audit.r2_eligible & ~audit.selected, ~audit.lime_screened & ~audit.selected],
        ["fewer_than_3_present", "ess_below_threshold", "pool_below_3", "pool_below_4_R2_uninformative",
         "global_cluster_cap", "target_reached", "not_screened_target_reached"], default="")
    audit["exclusion_reason"] = reasons
    sel = audit.loc[selected].reset_index()
    sel["rank_in_group"] = sel.groupby("group").cumcount() + 1
    verify_cluster_cap(sel, cap)
    return sel, audit.reset_index()


def verify_cluster_cap(selected: pd.DataFrame, cap: int = 2) -> None:
    worst = selected.groupby("cluster").size().max() if len(selected) else 0
    if worst > cap:
        bad = selected.groupby("cluster").size()
        raise RuntimeError(f"Global cluster cap violated: {bad[bad > cap].to_dict()}")
    if selected.window_id.duplicated().any():
        raise RuntimeError("Duplicate window selected")


def eligibility_table(audit: pd.DataFrame, selected: pd.DataFrame, targets: dict, scan_all=("FP", "FN")) -> pd.DataFrame:
    rows = []
    for g in GROUP_ORDER:
        a = audit[audit.group == g]
        s = selected[selected.group == g]
        full = g in scan_all
        rows.append({
            "group": g, "total_windows": len(a), "present_ge3": int(a.nnz_ge3.sum()), "ess_pass": int(a.ess_pass.sum()),
            "lime_screened": int(a.lime_screened.sum()), "complete_lime_scan": full,
            "feature_eligible_pool_ge3": int(a.feature_eligible.sum()), "r2_eligible_pool_ge4": int(a.r2_eligible.sum()),
            "removed_by_global_cap": int(a.blocked_by_cap.sum()) if full else np.nan,
            "target": targets.get(g), "final_selected": len(s), "selected_traces": s.trace_id.nunique(),
            "selected_independent_clusters": s.cluster.nunique(),
            "r2_eligible_traces": a[a.r2_eligible].trace_id.nunique(), "r2_eligible_clusters": a[a.r2_eligible].cluster.nunique()})
    return pd.DataFrame(rows)
