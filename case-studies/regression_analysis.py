"""Regression analysis: what predicts pass/fail?

Questions an analysis engineer at Cursor would ask:
  1. Which procedural features correlate with passing? (logistic regression)
  2. Does trajectory length predict pass rate? (calibration)
  3. What's the edit:test ratio in passing vs failing runs?
  4. Can we predict outcome from the first N atoms alone?
  5. Does OOD score predict failure? (anomaly as failure signal)
  6. What's the cost efficiency of passing runs vs failing runs?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
RES = HERE / "results"

# ── Load data ────────────────────────────────────────────────────────────────


def load_labeled():
    """Load SWE-agent-LM-32B trajectories with resolved labels."""
    fp_rows = [
        json.loads(l) for l in (RES / "fingerprints_child_n500.jsonl").read_text().splitlines()
    ]
    rich_rows = {}
    rf = RES / "rich_features_20250511_sweagent_lm_32b.jsonl"
    if rf.exists():
        for l in rf.read_text().splitlines():
            d = json.loads(l)
            rich_rows[d["instance_id"]] = d
    ood_rows = {}
    ood_f = RES / "ood_analysis_v1/ood_scores_canonical.jsonl"
    if ood_f.exists():
        for l in ood_f.read_text().splitlines():
            d = json.loads(l)
            if d["agent"] == "SWE-agent-LM-32B":
                ood_rows[d["instance_id"]] = d["score"]

    records = []
    for r in fp_rows:
        if r.get("resolved") is None:
            continue
        iid = r["instance_id"]
        atoms = [a for a in r.get("atoms_canonical", []) if a != "think"]
        n = max(1, len(atoms))
        cnt = Counter(atoms)
        rich = rich_rows.get(iid, {})
        records.append(
            {
                "instance_id": iid,
                "resolved": int(r["resolved"]),
                # Procedural composition
                "edit_frac": cnt.get("edit", 0) / n,
                "read_frac": cnt.get("read_file", 0) / n,
                "test_frac": cnt.get("run_test", 0) / n,
                "search_frac": cnt.get("search_repo", 0) / n,
                "create_frac": cnt.get("create_file", 0) / n,
                "delete_frac": cnt.get("delete_file", 0) / n,
                "other_frac": cnt.get("other", 0) / n,
                "error_frac": cnt.get("error", 0) / n,
                # Ratio features
                "edit_to_test": cnt.get("edit", 0) / max(1, cnt.get("run_test", 1)),
                "test_after_edit": cnt.get("run_test", 0) / max(1, cnt.get("edit", 1)),
                # Trajectory stats
                "n_actions": n,
                "n_steps": r.get("n_steps", n),
                # Rich features
                "tokens_per_action": rich.get("tokens_per_action", 0),
                "n_files_total": rich.get("n_files_total", 0),
                "n_files_read": rich.get("n_files_read", 0),
                "n_files_edited": rich.get("n_files_edited", 0),
                "instance_cost": rich.get("instance_cost", 0),
                # OOD score
                "ood_score": ood_rows.get(iid, np.nan),
            }
        )
    return pd.DataFrame(records)


def early_atoms(r, k=5):
    """First k action atoms (excluding think)."""
    atoms = [a for a in r.get("atoms_canonical", []) if a != "think"]
    cnt = Counter(atoms[:k])
    n = max(1, k)
    return {
        f"early_{a}": cnt.get(a, 0) / n
        for a in ["edit", "read_file", "run_test", "search_repo", "create_file", "error"]
    }


# ── Analysis 1: Feature correlations with resolved ──────────────────────────


def analysis_correlations(df):
    print("=" * 72)
    print("1. POINT-BISERIAL CORRELATIONS WITH PASS/FAIL")
    print("   (positive = more of this feature → more likely to pass)")
    print("=" * 72)
    features = [
        "edit_frac",
        "read_frac",
        "test_frac",
        "search_frac",
        "create_frac",
        "delete_frac",
        "error_frac",
        "other_frac",
        "edit_to_test",
        "test_after_edit",
        "n_actions",
        "tokens_per_action",
        "n_files_total",
        "n_files_read",
        "n_files_edited",
        "instance_cost",
        "ood_score",
    ]
    df_clean = df.dropna(subset=["resolved"])
    corrs = []
    for f in features:
        col = df_clean[f].fillna(df_clean[f].median())
        r = col.corr(df_clean["resolved"])
        corrs.append((r, f))
    corrs.sort(reverse=True)
    for r, f in corrs:
        bar = "▓" * int(abs(r) * 20)
        sign = "+" if r > 0 else "-"
        print(f"  {sign}{bar:<20s}  {r:+.3f}  {f}")
    print()


# ── Analysis 2: Logistic regression ─────────────────────────────────────────


def analysis_logistic(df):
    print("=" * 72)
    print("2. LOGISTIC REGRESSION: which features predict pass?")
    print("   (5-fold CV ROC-AUC)")
    print("=" * 72)
    features = [
        "edit_frac",
        "read_frac",
        "test_frac",
        "search_frac",
        "create_frac",
        "error_frac",
        "edit_to_test",
        "test_after_edit",
        "n_actions",
        "tokens_per_action",
        "n_files_total",
        "instance_cost",
        "ood_score",
    ]
    df_c = df.dropna(subset=["resolved"]).copy()
    for f in features:
        df_c[f] = df_c[f].fillna(df_c[f].median())
    X = df_c[features].values
    y = df_c["resolved"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    coefs = np.zeros(len(features))
    for tr, te in skf.split(X_s, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_s[tr], y[tr])
        prob = clf.predict_proba(X_s[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        coefs += clf.coef_[0]
    coefs /= 5

    print(f"  5-fold CV ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print("  random baseline:   0.500")
    print()
    print("  Feature coefficients (larger positive → more predictive of PASS):")
    ranked = sorted(zip(coefs, features, strict=False), reverse=True)
    for c, f in ranked:
        bar = "▓" * int(abs(c) * 12)
        sign = "+" if c > 0 else "-"
        print(f"  {sign}{bar:<14s}  {c:+.3f}  {f}")
    print()


# ── Analysis 3: Pass rate by trajectory length ───────────────────────────────


def analysis_length_vs_pass(df):
    print("=" * 72)
    print("3. PASS RATE BY TRAJECTORY LENGTH (deciles)")
    print("=" * 72)
    df_c = df.dropna(subset=["resolved"]).copy()
    df_c["length_decile"] = pd.qcut(df_c["n_actions"], q=10, labels=False, duplicates="drop")
    stats = (
        df_c.groupby("length_decile")
        .agg(
            n=("resolved", "count"),
            pass_rate=("resolved", "mean"),
            median_len=("n_actions", "median"),
        )
        .reset_index()
    )
    print(f"  {'Decile':>8s}  {'Median len':>11s}  {'n':>4s}  {'Pass rate':>10s}")
    for _, row in stats.iterrows():
        bar = "█" * int(row.pass_rate * 20)
        print(
            f"  {int(row.length_decile)+1:>8d}  {row.median_len:>11.0f}  {row.n:>4.0f}  {row.pass_rate:>8.1%}  {bar}"
        )
    print()


# ── Analysis 4: Edit:test ratio vs pass rate ──────────────────────────────────


def analysis_edit_test_ratio(df):
    print("=" * 72)
    print("4. EDIT:TEST RATIO VS PASS RATE")
    print("   (how many edits per test run)")
    print("=" * 72)
    df_c = df.dropna(subset=["resolved"]).copy()
    # Bin edit_to_test into buckets
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf")]
    labels = [
        "<0.5 (tests >> edits)",
        "0.5–1.0",
        "1.0–1.5",
        "1.5–2.0",
        "2.0–3.0",
        ">3.0 (edits >> tests)",
    ]
    df_c["et_bucket"] = pd.cut(df_c["edit_to_test"], bins=bins, labels=labels)
    stats = (
        df_c.groupby("et_bucket", observed=True)
        .agg(n=("resolved", "count"), pass_rate=("resolved", "mean"))
        .reset_index()
    )
    print(f"  {'Edit:Test ratio':>25s}  {'n':>4s}  {'Pass rate':>10s}")
    for _, row in stats.iterrows():
        bar = "█" * int(row.pass_rate * 20)
        print(f"  {row.et_bucket!s:>25s}  {row.n:>4.0f}  {row.pass_rate:>8.1%}  {bar}")
    print()


# ── Analysis 5: Early prediction (first 5 atoms) ─────────────────────────────


def analysis_early_prediction():
    print("=" * 72)
    print("5. EARLY PREDICTION: pass/fail from first 5 actions alone")
    print("=" * 72)
    fp_rows = [
        json.loads(l) for l in (RES / "fingerprints_child_n500.jsonl").read_text().splitlines()
    ]
    records = []
    for r in fp_rows:
        if r.get("resolved") is None:
            continue
        ea = early_atoms(r, k=5)
        ea["resolved"] = int(r["resolved"])
        ea["n_actions"] = len([a for a in r.get("atoms_canonical", []) if a != "think"])
        records.append(ea)
    df_e = pd.DataFrame(records)
    features = [c for c in df_e.columns if c.startswith("early_")]
    X = df_e[features].values
    y = df_e["resolved"].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X_s, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[tr], y[tr])
        prob = clf.predict_proba(X_s[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    print(f"  ROC-AUC from first 5 actions: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(
        f"  Interpretation: {np.mean(aucs):.0%} chance of ranking a passing trajectory above a failing one"
    )
    print("  (random = 50%; full-trajectory model = ~60%)")
    print()


# ── Analysis 6: Cost efficiency ───────────────────────────────────────────────


def analysis_cost_efficiency(df):
    print("=" * 72)
    print("6. COST EFFICIENCY: $ per resolution")
    print("=" * 72)
    df_c = df.dropna(subset=["resolved", "instance_cost"]).copy()
    df_c = df_c[df_c["instance_cost"] > 0]
    pass_cost = df_c[df_c["resolved"] == 1]["instance_cost"].mean()
    fail_cost = df_c[df_c["resolved"] == 0]["instance_cost"].mean()
    pass_rate = df_c["resolved"].mean()
    expected_cost_per_resolution = df_c["instance_cost"].mean() / max(0.001, pass_rate)
    print(f"  avg cost, passing runs:   ${pass_cost:.3f}")
    print(f"  avg cost, failing runs:   ${fail_cost:.3f}")
    print(f"  overall pass rate:        {pass_rate:.1%}")
    print(
        f"  expected cost/resolution: ${expected_cost_per_resolution:.2f}  (avg_cost / pass_rate)"
    )
    print()
    # Top 20% of trajectories by cost — do they pass more?
    threshold = df_c["instance_cost"].quantile(0.8)
    expensive = df_c[df_c["instance_cost"] >= threshold]["resolved"].mean()
    cheap = df_c[df_c["instance_cost"] < threshold]["resolved"].mean()
    print(f"  Pass rate, top 20% most expensive: {expensive:.1%}")
    print(f"  Pass rate, bottom 80%:             {cheap:.1%}")
    print()


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_labeled()
    print(
        f"Loaded {len(df)} labeled trajectories  "
        f"({df['resolved'].sum():.0f} pass / {(1-df['resolved']).sum():.0f} fail)\n"
    )

    analysis_correlations(df)
    analysis_logistic(df)
    analysis_length_vs_pass(df)
    analysis_edit_test_ratio(df)
    analysis_early_prediction()
    analysis_cost_efficiency(df)
