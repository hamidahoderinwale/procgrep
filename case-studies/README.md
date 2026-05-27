# procgrep — empirical case studies

Analysis scripts for the paper *Procedural Grep: Structural Variation for Agent Rollouts*.
All scripts run against fingerprint JSONL files produced by `pull_and_fingerprint.py`
from the public SWE-bench S3 archive (`s3://swe-bench-submissions/`).

## Setup

```bash
pip install procgrep  # or: pip install -e ../
pip install altair vl-convert-python pandas numpy scikit-learn scipy
```

## Data pipeline

| Script | What it does |
|---|---|
| `pull_and_fingerprint.py` | Pull `.traj` files from S3 and extract canonical + native atom sequences |
| `pull_from_cache.py` | Fingerprint agents from the local trajectory cache (DARS, Claude-3.7) |
| `pull_openhands.py` | OpenHands tool-call format adapter |
| `pull_tools_claude37.py` | Claude native XML `<function_calls>` format adapter |
| `extract_patches.py` | Extract patch metadata for reward-hacking analysis |
| `extract_rich_features.py` | Token counts, file counts, cost per trajectory |

## Analysis

| Script | What it produces |
|---|---|
| `multi_agent_analysis.py` | JSD matrices, identification probe |
| `discriminative_procedures_analysis.py` | BPE-learned habits exclusive to each agent |
| `discriminative_bigrams.py` | Transition-level fingerprints, positional divergence curves |
| `positional_divergence.py` | Which step diverges most across agents |
| `metric_comparison.py` | JSD vs KL / Hellinger / TV / Cosine robustness check |
| `behavioral_features.py` | Exploration vs exploitation zeitgeist features |
| `tier1b_matched_pairs.py` | Same-outcome-different-procedure (SODP) matched pair analysis |
| `regression_analysis.py` | Outcome prediction from procedural features (AUC = 0.81) |
| `identification_probe_stratified.py` | Stratified k-fold agent identification probe |
| `ood_analysis.py` | Per-trajectory OOD score distribution |
| `patch_type_figures.py` | Fix-type breakdown (source-only vs source+tests) by agent |

## Figures

```bash
python make_figures.py         # all 14 paper figures → results/paper_figures/
python regression_figures.py   # regression figures
python patch_type_figures.py   # fix-type dot plot
```

## CLI

```bash
procgrep compare agent_a.jsonl agent_b.jsonl --name-a "Agent A" --name-b "Agent B"
```
