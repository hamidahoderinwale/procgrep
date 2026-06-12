# procgrep

`procgrep` reads agent trace logs and tells you how agents differ in *how they work*, not just whether they passed.

[**Live explorer**](https://midah-procgrep-explorer.hf.space) · [**Interactive essay**](https://hamidah.me/procgrep)

![Replaying one agent trajectory step by step; a structural query fires the instant it matches](docs/figures/replay.gif)

It does not run agents. It does not call any model. It reads files and produces numbers and comparisons.

---

## What it does

When a coding agent attempts a task, it leaves behind a sequence of actions — search the repo, read a file, edit a file, run tests, submit. `procgrep` converts those action sequences into comparable representations and answers questions like:

- **Are these two agents doing the same thing?** Feed both sets of traces and get a divergence score.
- **Which agent is more consistent?** Measure how much an agent's procedure varies across tasks.
- **What makes one agent fail where another succeeds?** Find the action-sequence patterns exclusive to failures.
- **Did fine-tuning change how the model works, not just whether it passes?** Compare parent and child trajectories across four structural axes.
- **Is this trajectory heading toward failure?** Match against known failure patterns early enough to act on it.

---

## Quickstart

```bash
pip install procgrep
```

```python
from procgrep import fit_bpe, encode, jsd_matrix
from procgrep.io import read_jsonl
from procgrep import canonicalize

# Load traces and convert to a shared action alphabet
traces = canonicalize(list(read_jsonl("traces/raw.jsonl")), adapter="swe-agent")

# Learn what recurring action sequences look like across this corpus
vocab = fit_bpe([t.atoms for t in traces], vocab_size=64, seed=0)

# Encode each trajectory as a distribution over those sequences
fingerprints = encode(traces, vocab=vocab)

# How different are two groups of agents?
matrix = jsd_matrix(fingerprints, group_by="agent")
for row in matrix.to_records():
    print(row)
```

---

## Core concepts

**Atoms.** Every agent action maps to one of nine canonical types: `search_repo`, `read_file`, `edit`, `run_test`, `create_file`, `delete_file`, `think`, `submit`, `other`. This shared alphabet makes agents running on different scaffolds comparable.

**Trajectory.** One agent attempting one task, represented as an ordered list of atoms. This is what `procgrep` ingests.

**Procedures.** Recurring multi-step patterns learned from the corpus via BPE (the same algorithm used to tokenize text for language model training). A procedure might be `search_repo → read_file → think` — a pattern that appears frequently enough to be worth naming.

**Fingerprint.** A trajectory encoded as a distribution over procedures — how often each procedure appeared. Two trajectories with similar fingerprints approached the problem similarly.

**JSD.** Jensen-Shannon divergence: how different two fingerprints are. 0 = identical, 1 = completely non-overlapping. Used to compare agents, groups, or training conditions.

---

## Main uses

### Compare two agents

```python
from procgrep import fit_bpe, encode, jsd

traces_a = ...  # agent A traces
traces_b = ...  # agent B traces
all_traces = traces_a + traces_b

vocab = fit_bpe([t.atoms for t in all_traces], vocab_size=64, seed=0)
fps_a = encode(traces_a, vocab=vocab)
fps_b = encode(traces_b, vocab=vocab)

# Mean fingerprint for each group
import numpy as np
mean_a = np.mean([fp.distribution() for fp in fps_a], axis=0)
mean_b = np.mean([fp.distribution() for fp in fps_b], axis=0)
print("JSD:", jsd(mean_a, mean_b))
```

### Find what makes an agent fail

```python
from procgrep import discriminative_procedures

# Which procedures separate passing from failing trajectories?
top = discriminative_procedures(
    fingerprints,
    vocab,
    group_a="pass",
    group_b="fail",
    k=10,
    ranking="log_odds",
)
for m in top:
    print(m.procedure, "  pass rate:", m.p_a, "  fail rate:", m.p_b)
```

### Check if a trajectory is using a known failure pattern

```python
from procgrep import load_patterns, match_patterns

patterns = load_patterns("examples/rules/known_failure_patterns.yaml")
report = match_patterns(traces, patterns)

for trace_id, violations in report.violations.items():
    print(trace_id, violations)
```

The `examples/rules/known_failure_patterns.yaml` file includes patterns validated on SWE-bench: edit streaks without tests, stuck reading loops, no exploration before editing.

### Score a trajectory against a procedural spec

A reward spec defines what a good trajectory looks like — which phases it should go through, which patterns are failures, which are bonuses. Returns a 0–1 score per trajectory.

```python
from procgrep.reward import load_spec, score

spec = load_spec("examples/rules/reward_spec_swe_agent.yaml")
result = score(trajectory.atoms, spec)

print(result.proc_score)           # e.g. 0.75
print(result.satisfied_phases)     # ["exploration", "implementation", "test_verification"]
print(result.triggered_penalties)  # ["stuck_reading"]
```

The `stuck_reading` penalty (`read_file → think → read_file → think`) fires in the first 12 steps on 80% of trajectories that eventually fail, making it usable as a real-time circuit breaker before the context budget is exhausted.

### Compare parent and child after fine-tuning

```python
from procgrep import lineage_diff

diff = lineage_diff(
    parent_trajectories,
    child_trajectories,
    along=["vocabulary", "entropy", "outcome_quadrant", "conditional"],
)
print(diff.summary())
```

This tells you: did the child preserve the parent's action repertoire? Did the distribution concentrate (mode collapse)? Do failing child trajectories look less like the parent than passing ones? Where in the sequence does the child diverge?

---

## CLI

```bash
# Convert raw traces to canonical atom sequences
procgrep canonicalize --input traces/raw.jsonl --adapter swe-agent --output traces/canonical.jsonl

# Learn a procedure vocabulary
procgrep fit-bpe --input traces/canonical.jsonl --vocab-size 64 --output vocab.json

# Encode and compare
procgrep encode --input traces/canonical.jsonl --vocab vocab.json --output fingerprints.jsonl
procgrep jsd --input fingerprints.jsonl --group-by agent --output jsd_matrix.json
```

---

## Installation

```bash
pip install procgrep           # core
pip install procgrep[reward]   # includes pyyaml for reward spec scoring
pip install procgrep[dev]      # development dependencies
```

---

## Notes

- Python 3.10+. No LLM SDK required.
- Built-in adapters for SWE-agent, mini-swe-agent, OpenHands, Agentless, DARS, Moatless, SWE-smith, GumTree, and ReAct-text trajectories.
- All random operations take a `seed` argument; default is `0`.
- `ruff` and `mypy --strict` clean.

See [METRICS.md](METRICS.md) for the full list of measurements, [STUDIES.md](STUDIES.md) for worked case studies, and [FAQ.md](FAQ.md) for common questions. Runnable demos are in [`examples/`](examples/); the live-explorer backend is in [`space/`](space/); the essay, figures, and reference pages are in [`docs/`](docs/).
