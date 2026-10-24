"""Global identical-content cluster cap, eligibility enforcement and deterministic selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.phase2.sampling import GROUP_ORDER, select_windows, verify_cluster_cap


def make(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g, n, ncl in (("FP", 60, 12), ("FN", 90, 30), ("TP", 200, 40), ("TN", 400, 60)):
        for i in range(n):
            cl = f"C{g}{i % ncl}" if g in ("FP", "TN") else f"CA{i % ncl}"      # TP and FN share attack clusters
            rows.append({"window_id": f"{g}-{i}", "trace_id": f"T{g}{i % ncl}", "group": g, "cluster": cl, "nnz": int(rng.integers(0, 12))})
    # make FP and TN share some clusters so the cap must hold ACROSS groups
    df = pd.DataFrame(rows)
    df.loc[df.group == "TN", "cluster"] = df.loc[df.group == "TN", "cluster"].str.replace("CTN", "CFP", regex=False)
    return df


ESS = lambda r: 50.0 if int(r["window_id"].split("-")[1]) % 7 == 0 else 500.0
POOL = lambda r: int(r["nnz"]) - 2 if r["nnz"] >= 3 else 0


def test_global_cap_eligibility_and_targets():
    df = make()
    sel, audit = select_windows(df, ESS, POOL, {g: 40 for g in GROUP_ORDER}, cap=2, seed=42)
    verify_cluster_cap(sel, 2)
    assert sel.groupby("cluster").size().max() <= 2
    a = audit.set_index("window_id").loc[sel.window_id]
    assert (a.ess >= 100).all() and (a.pool_r2_agg >= 4).all() and a.nnz_ge3.all()   # only eligible windows are ever selected
    assert sel.groupby("group").size().max() <= 40
    fp_eligible_clusters = audit[(audit.group == "FP") & audit.r2_eligible].cluster.nunique()
    assert (sel.group == "FP").sum() <= 2 * fp_eligible_clusters


def test_undersized_group_uses_all_available_not_relaxed():
    df = make()
    sel, audit = select_windows(df, ESS, POOL, {g: 40 for g in GROUP_ORDER}, cap=2, seed=42)
    fp = sel[sel.group == "FP"]
    cap_limited = audit[(audit.group == "FP") & audit.r2_eligible]
    # whatever FP could be selected under the cap was selected: no eligible FP window is left unselected unless its cluster is full
    left = cap_limited[~cap_limited.selected]
    full = sel.groupby("cluster").size()
    assert all(full.get(c, 0) >= 2 or len(fp) >= 40 for c in left.cluster)


def test_deterministic_and_input_order_independent():
    df = make()
    a, _ = select_windows(df, ESS, POOL, {g: 40 for g in GROUP_ORDER}, seed=42)
    b, _ = select_windows(df.sample(frac=1, random_state=3).reset_index(drop=True), ESS, POOL, {g: 40 for g in GROUP_ORDER}, seed=42)
    c, _ = select_windows(df, ESS, POOL, {g: 40 for g in GROUP_ORDER}, seed=7)
    assert sorted(a.window_id) == sorted(b.window_id) and sorted(a.window_id) != sorted(c.window_id)


def test_pass_one_prefers_distinct_clusters():
    df = make()
    sel, _ = select_windows(df, ESS, POOL, {"TP": 10, "TN": 0, "FP": 0, "FN": 0}, seed=1)
    assert sel.cluster.nunique() == len(sel) == 10                    # 40 clusters available -> all distinct


def test_verify_cluster_cap_raises_on_violation_and_duplicates():
    bad = pd.DataFrame({"window_id": ["a", "b", "c"], "cluster": ["x", "x", "x"]})
    with pytest.raises(RuntimeError):
        verify_cluster_cap(bad, 2)
    with pytest.raises(RuntimeError):
        verify_cluster_cap(pd.DataFrame({"window_id": ["a", "a"], "cluster": ["x", "y"]}), 2)


def test_ineligible_ess_never_selected_even_if_scarce():
    df = make()
    sel, audit = select_windows(df, lambda r: 10.0, POOL, {g: 40 for g in GROUP_ORDER})
    assert len(sel) == 0 and (audit.exclusion_reason.isin(["fewer_than_3_present", "ess_below_threshold"])).all()
