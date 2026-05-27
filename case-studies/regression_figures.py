"""Figures from the regression analysis."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from scripts.theme import GRAY, NEAR_BLACK, register

register()

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = RES / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    rows = []
    for r in [
        json.loads(l) for l in (RES / "fingerprints_child_n500.jsonl").read_text().splitlines()
    ]:
        if r.get("resolved") is None:
            continue
        atoms = [a for a in r.get("atoms_canonical", []) if a != "think"]
        n = max(1, len(atoms))
        cnt = Counter(atoms)
        rich = {}
        # Try to join rich features
        rows.append(
            {
                "resolved": int(r["resolved"]),
                "edit_frac": cnt.get("edit", 0) / n,
                "read_frac": cnt.get("read_file", 0) / n,
                "test_frac": cnt.get("run_test", 0) / n,
                "search_frac": cnt.get("search_repo", 0) / n,
                "create_frac": cnt.get("create_file", 0) / n,
                "error_frac": cnt.get("error", 0) / n,
                "edit_to_test": cnt.get("edit", 0) / max(1, cnt.get("run_test", 1)),
                "n_actions": n,
            }
        )
    df = pd.DataFrame(rows)
    # Join rich features
    rf = RES / "rich_features_20250511_sweagent_lm_32b.jsonl"
    if rf.exists():
        rd = {json.loads(l)["instance_id"]: json.loads(l) for l in rf.read_text().splitlines()}
        # re-load with instance_id
    fp_rows = [
        json.loads(l) for l in (RES / "fingerprints_child_n500.jsonl").read_text().splitlines()
    ]
    rich_df_rows = []
    for r in fp_rows:
        if r.get("resolved") is None:
            continue
        atoms = [a for a in r.get("atoms_canonical", []) if a != "think"]
        n = max(1, len(atoms))
        cnt = Counter(atoms)
        rich = rd.get(r["instance_id"], {})
        rich_df_rows.append(
            {
                "resolved": int(r["resolved"]),
                "edit_frac": cnt.get("edit", 0) / n,
                "read_frac": cnt.get("read_file", 0) / n,
                "test_frac": cnt.get("run_test", 0) / n,
                "search_frac": cnt.get("search_repo", 0) / n,
                "create_frac": cnt.get("create_file", 0) / n,
                "error_frac": cnt.get("error", 0) / n,
                "edit_to_test": cnt.get("edit", 0) / max(1, cnt.get("run_test", 1)),
                "n_actions": n,
                "instance_cost": rich.get("instance_cost", 0),
                "n_files_total": rich.get("n_files_total", 0),
                "tokens_per_action": rich.get("tokens_per_action", 0),
            }
        )
    return pd.DataFrame(rich_df_rows)


def save(chart, name):
    path = OUT / f"{name}.png"
    chart.save(str(path), scale_factor=2)
    print(f"  saved {name}.png")
    return path


def fig_feature_importance():
    df = load()
    feats = [
        "create_frac",
        "edit_frac",
        "test_frac",
        "search_frac",
        "n_actions",
        "error_frac",
        "edit_to_test",
        "read_frac",
        "n_files_total",
        "instance_cost",
    ]
    corrs = [(df[f].fillna(0).corr(df["resolved"]), f) for f in feats]
    corrs.sort(reverse=True)

    LABELS = {
        "create_frac": "create-file fraction",
        "edit_frac": "edit fraction",
        "test_frac": "run-test fraction",
        "search_frac": "search fraction",
        "n_actions": "trajectory length",
        "error_frac": "error fraction",
        "edit_to_test": "edit-to-test ratio",
        "read_frac": "read-file fraction",
        "n_files_total": "files touched",
        "instance_cost": "API cost",
    }
    data = [
        {"feature": LABELS.get(f, f), "r": r, "direction": "positive" if r >= 0 else "negative"}
        for r, f in corrs
    ]
    dfp = pd.DataFrame(data)

    chart = (
        alt.Chart(dfp)
        .mark_bar()
        .encode(
            x=alt.X(
                "r:Q",
                title="Correlation with pass (point-biserial r)",
                scale=alt.Scale(domain=[-0.6, 0.5]),
            ),
            y=alt.Y("feature:N", sort=list(dfp["feature"]), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["positive", "negative"], range=["#1A7A4A", "#C0392B"]),
                legend=None,
            ),
            tooltip=["feature", alt.Tooltip("r:Q", format=".3f")],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=NEAR_BLACK, strokeWidth=1)
        .encode(x="x:Q")
    )
    return save(
        (chart + zero).properties(
            title="Feature correlations with task outcome", width=420, height=300
        ),
        "fig_regression_correlations",
    )


def fig_length_vs_pass():
    df = load()
    bins = [0, 10, 20, 30, 40, 50, 60, float("inf")]
    labels = ["1–10", "11–20", "21–30", "31–40", "41–50", "51–60", ">60"]
    df["bin"] = pd.cut(df["n_actions"], bins=bins, labels=labels, right=True)
    stats = (
        df.groupby("bin", observed=True)
        .agg(n=("resolved", "count"), pass_rate=("resolved", "mean"), mid=("n_actions", "median"))
        .reset_index()
    )
    stats = stats[stats["n"] >= 5]

    chart = (
        alt.Chart(stats)
        .mark_line(point=True, color="#2471A3")
        .encode(
            x=alt.X(
                "bin:N",
                sort=labels,
                title="Tool calls per task (bash commands + file operations + tests)",
            ),
            y=alt.Y(
                "pass_rate:Q",
                title="Pass rate",
                scale=alt.Scale(domain=[0, 1.0]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=["bin", alt.Tooltip("pass_rate:Q", format=".0%"), alt.Tooltip("n:Q")],
        )
    )
    # annotate n on each point
    txt = (
        alt.Chart(stats)
        .mark_text(dy=-12, fontSize=8, color=NEAR_BLACK)
        .encode(x=alt.X("bin:N", sort=labels), y=alt.Y("pass_rate:Q"), text=alt.Text("n:Q"))
    )
    return save(
        (chart + txt).properties(title="Pass rate by trajectory length", width=400, height=260),
        "fig_regression_length",
    )


def fig_edit_test_ratio():
    """KDE density of edit-to-test ratio, pass vs fail overlaid.

    Avoids arbitrary bin edges; shows where the two outcome distributions
    diverge (passing runs cluster near ratio=1, failing runs have a long
    right tail of edit-heavy trajectories with little testing).
    """
    from scipy.stats import gaussian_kde

    df = load()
    # Cap at 6 to keep x-axis readable (99th pct ≈ 5.5)
    cap = 6.0
    pass_vals = df[df["resolved"] == 1]["edit_to_test"].clip(0, cap).dropna().values
    fail_vals = df[df["resolved"] == 0]["edit_to_test"].clip(0, cap).dropna().values

    xs = np.linspace(0, cap, 300)
    pass_kde = gaussian_kde(pass_vals, bw_method=0.3)(xs)
    fail_kde = gaussian_kde(fail_vals, bw_method=0.3)(xs)

    kde_df = pd.DataFrame(
        {
            "ratio": np.concatenate([xs, xs]),
            "density": np.concatenate([pass_kde, fail_kde]),
            "outcome": ["pass"] * len(xs) + ["fail"] * len(xs),
        }
    )

    chart = (
        alt.Chart(kde_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "ratio:Q",
                title="Edit-to-test ratio (edits per test run, capped at 6)",
                axis=alt.Axis(tickCount=7),
            ),
            y=alt.Y("density:Q", title="Density", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["pass", "fail"], range=["#1A7A4A", "#C0392B"]),
                legend=alt.Legend(title=None, orient="top-right"),
            ),
        )
    )
    # vertical reference at ratio=1 (one edit per test run)
    ref = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4], strokeWidth=1)
        .encode(x="x:Q")
    )

    return save(
        (chart + ref).properties(
            title="Edit-to-test ratio distribution by outcome", width=400, height=240
        ),
        "fig_regression_edit_test",
    )


def fig_roc_curve():
    df = load()
    feats = [
        "create_frac",
        "edit_frac",
        "test_frac",
        "n_actions",
        "error_frac",
        "edit_to_test",
        "read_frac",
        "n_files_total",
    ]
    df_c = df.dropna(subset=["resolved"]).copy()
    for f in feats:
        df_c[f] = df_c[f].fillna(0)
    X = StandardScaler().fit_transform(df_c[feats].values)
    y = df_c["resolved"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs, mean_fpr = [], np.linspace(0, 1, 100)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        fpr, tpr, _ = roc_curve(y[te], clf.predict_proba(X[te])[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    roc_auc = auc(mean_fpr, mean_tpr)

    roc_df = pd.DataFrame({"fpr": mean_fpr, "tpr": mean_tpr})
    diag_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    curve = (
        alt.Chart(roc_df)
        .mark_line(color="#2471A3", strokeWidth=2)
        .encode(
            x=alt.X("fpr:Q", title="False positive rate"),
            y=alt.Y("tpr:Q", title="True positive rate"),
        )
    )
    diag = alt.Chart(diag_df).mark_line(color=GRAY, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    label = (
        alt.Chart(pd.DataFrame({"x": [0.6], "y": [0.2], "t": [f"AUC = {roc_auc:.2f}"]}))
        .mark_text(fontSize=11, color="#2471A3")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )

    return save(
        (diag + curve + label).properties(
            title="ROC curve, pass prediction from procedure features", width=300, height=300
        ),
        "fig_regression_roc",
    )


if __name__ == "__main__":
    print("Generating regression figures...")
    fig_feature_importance()
    fig_length_vs_pass()
    fig_edit_test_ratio()
    fig_roc_curve()
    print("Done.")
