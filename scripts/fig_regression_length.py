"""Logistic regression: resolution probability vs trajectory length.

Data: bidirect-align-dev-traces/output/paper2_pilot/bpe_sequences_extended.jsonl
      bidirect-align-dev-traces/output/paper2_pilot/extended_pass_fail.json

Shows the fitted P(resolve) curve with a 95% bootstrap band as a function of
canonical trajectory length on a log10 x-axis.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/scripts")
from figtheme import init, style_axes, BLUE, RULE, INK

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")

init()

rows = [json.loads(l) for l in open(ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl")]
pf = json.load(open(ROOT / "output/paper2_pilot/extended_pass_fail.json"))
res = {k: set(v.get("resolved", [])) for k, v in pf.items()}

L, y = [], []
for r in rows:
    s = res.get(r["submission"])
    if s is None:
        continue
    n = r.get("canonical_length", len(r["canonical"]))
    if n >= 1:
        L.append(n)
        y.append(int(r["instance_id"] in s))
L = np.array(L, float)
y = np.array(y)
X = np.log10(L).reshape(-1, 1)

clf = LogisticRegression().fit(X, y)
slope = clf.coef_[0, 0]

grid = np.linspace(np.log10(L.min()), np.log10(L.max()), 200)
p = clf.predict_proba(grid.reshape(-1, 1))[:, 1]

rng = np.random.default_rng(42)
boot = []
for _ in range(300):
    idx = rng.integers(0, len(L), len(L))
    b = LogisticRegression().fit(X[idx], y[idx])
    boot.append(b.predict_proba(grid.reshape(-1, 1))[:, 1])
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

x_vals = 10 ** grid

fig, ax = plt.subplots(figsize=(6.2, 4.4))

ax.fill_between(x_vals, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
ax.plot(x_vals, p, color=BLUE, linewidth=2.0)

ax.set_xscale("log")
ax.set_xlim(x_vals[0], x_vals[-1])
ax.set_ylim(0, 0.65)
ax.set_xlabel("trajectory length (canonical actions)")
ax.set_ylabel("P(resolve)")
ax.set_title("Resolution probability vs trajectory length")

ax.yaxis.grid(False)
ax.xaxis.grid(False)
style_axes(ax)

fig.savefig(
    "/Users/hamidaho/learning-from-dev/procgrep/docs/figures/fig_regression_length.png",
    dpi=200,
    facecolor="white",
    bbox_inches="tight",
)
plt.close(fig)

p10 = clf.predict_proba([[np.log10(10)]])[0, 1]
p60 = clf.predict_proba([[np.log10(60)]])[0, 1]
print(f"n={len(L)}, slope={slope:.3f} log-odds/decade")
print(f"P(resolve): length 10 -> {p10:.3f} | length 60 -> {p60:.3f}")
print("wrote fig_regression_length.png")
