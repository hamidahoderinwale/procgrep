# procgrep: empirical case studies

Reproduces the empirical results in *Procedural Grep: Structural Variation for Agent Rollouts*. Each script reads fingerprint JSONLs (one row per trajectory: `instance_id`, `atoms_canonical`, `atoms_native`, `resolved`) and writes a JSON or PNG.

## Setup

```bash
pip install procgrep altair vl-convert-python pandas numpy scikit-learn scipy
```

## Data pipeline

Adapter scripts convert raw trajectory formats into a uniform fingerprint JSONL.

| Script | Input format |
|---|---|
| `pull_and_fingerprint.py` | SWE-agent `.traj` JSON from S3 |
| `pull_tools_claude37.py` | Claude native `<function_calls>` text |
| `pull_openhands.py` | OpenHands tool-call traces |
| `pull_from_cache.py` | Local SWE-agent / DARS trajectory cache |
| `extract_patches.py` | S3 `patch.diff` blobs (for fix-type analysis) |

## Analysis

**Per-agent fingerprints**
- `multi_agent_analysis.py`: pairwise JSD matrix, identification probe
- `behavioral_features.py`: search-first %, edit streaks, recovery, interleave
- `discriminative_procedures_analysis.py`: BPE habits exclusive to each agent
- `discriminative_bigrams.py`: transition-level fingerprints

**Cross-agent comparison**
- `positional_divergence.py`: per-step JSD between agent pairs
- `metric_comparison.py`: JSD vs KL / Hellinger / TV / Cosine robustness
- `identification_probe_stratified.py`: stratified k-fold agent identification
- `tier1b_matched_pairs.py`: same-outcome-different-procedure (SODP) pairs

**Outcome-aware**
- `regression_analysis.py`: outcome prediction from procedural features (AUC = 0.81)
- `ood_analysis.py`: per-trajectory OOD score distribution
- `patch_type_figures.py`: source-only vs source+tests breakdown

## Figures

```bash
python make_figures.py          # all paper figures → results/paper_figures/
```

## CLI

```bash
procgrep compare agent_a.jsonl agent_b.jsonl --name-a "Agent A" --name-b "Agent B"
```
