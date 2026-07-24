# Metrics

What `procgrep` can measure about a group of agent trajectories, and what each number tells you.

Each metric is defined per-group (e.g. per-agent, per-training-condition). Several are correlated, so you usually don't need all of them. Start with JSD and entropy; add others if they answer a specific question.

---

## From the procgrep API

### Trajectory length

**What it tells you.** How many steps an agent takes on average.

Long trajectories aren't better. They often indicate the agent is stuck. Claude-4 averages 65 steps with a 59% pass rate; Claude-3 Opus averages 17 steps with an 11.7% pass rate. Length alone doesn't predict outcome, but length combined with action type does.

```python
mean_length = sum(len(t.atoms) for t in group) / len(group)
```

---

### Entropy

**What it tells you.** How spread out an agent's actions are.

High entropy = the agent uses many different action types across its trajectories. Low entropy = it relies on a few. A distilled child model typically has lower entropy than its parent: it over-learned the focused patterns from successful demonstrations.

```python
from procgrep import entropies_per_group
stats = entropies_per_group(fingerprints, group_by="agent")
# Returns median, IQR, range per group
```

---

### Jensen-Shannon divergence (JSD)

**What it tells you.** How different two agents' procedure distributions are.

0 = identical. 1 = completely non-overlapping. The most useful single number for comparing agents. Pair with within-group JSD as a baseline: if two groups differ by 0.30 but each group varies internally by 0.25, the difference is marginal.

From 2,639 SWE-bench trajectories: same-era agents (Claude-3, GPT-4) differ by ~0.07. Different scaffolds (SWE-agent vs tools-format) on the same model differ by ~0.49. The child model (SWE-agent-LM-32B) differs from its parent (Claude-3.7) by only 0.10, the closest pair in the corpus.

```python
from procgrep import jsd_matrix
matrix = jsd_matrix(fingerprints, group_by="agent")
```

---

### Effective vocabulary size

**What it tells you.** How many distinct action patterns an agent actually uses, accounting for how often it uses each.

Two agents might both have 64 procedures in their vocabulary, but one might use 3 of them 90% of the time (low effective size) while the other spreads usage more evenly (high effective size). SWE-agent-LM-32B has the lowest effective vocabulary: it concentrates heavily on read-file loops.

```python
from procgrep import effective_vocab_size_per_group
evs = effective_vocab_size_per_group(fingerprints, group_by="agent")
```

---

### Procedure concentration

**What it tells you.** Whether an agent has one dominant approach or many.

High concentration = specialist (this agent mostly does one thing, e.g. edit-heavy). Low concentration = generalist. Use this when you want to understand whether an agent's procedure is narrow or varied, without computing the full vocabulary.

```python
# 3 lines: group mean fingerprint, then sum of squares
import numpy as np
mean_dist = np.mean([fp.distribution() for fp in group_fps], axis=0)
hhi = float(np.sum(mean_dist ** 2))
```

---

### Atom-frequency Gini

**What it tells you.** Whether an agent relies heavily on a few action types (high Gini) or uses the full alphabet evenly (low Gini).

Unlike entropy (which operates on learned procedures), Gini operates directly on raw atom counts, no vocabulary needed. Useful for a quick first look without fitting BPE.

GPT-4o has the highest Gini in our corpus: ~53% of its actions are edits.

```python
from collections import Counter
import numpy as np

def gini(atoms):
    counts = sorted(Counter(atoms).values())
    n = len(counts)
    return sum((2*i - n - 1) * c for i, c in enumerate(counts, 1)) / (n * sum(counts))
```

---

### Procedural reward score

**What it tells you.** How well a single trajectory followed a user-defined procedure: which phases it completed, which failure patterns it triggered, which best practices it earned.

Unlike the metrics above (which describe a group), this scores one trajectory at a time. Useful for RL training signal and for real-time monitoring.

From 498 child model trajectories: passing trajectories score 0.902, failing score 0.723. On 23 matched parent-pass/child-fail pairs, the score drops by 0.34 on average, and the `stuck_reading` penalty fires in the first 12 steps on 80% of those failures.

```python
from procgrep.reward import load_spec, score

spec = load_spec("examples/rules/reward_spec_swe_agent.yaml")
result = score(trajectory.atoms, spec)
print(result.proc_score, result.triggered_penalties)
```

---

## Requiring additional data

These two metrics need patch diffs or task labels beyond what `procgrep` reads from trace logs.

### Edit-certificate Jaccard

How structurally similar are two agents' patches on the same task? (0 = nothing in common, 1 = identical edit shape.) Operates on the *output*, not the procedure, complementary to JSD.

Computed in the `bidirect-align-dev-traces` companion repository; not yet in the `procgrep` API.

---

### Composition-failure rate

What fraction of failures happened on tasks where the agent had previously solved all the sub-tasks individually? Separates "agent can't compose" from "agent never learned this skill."

Computed in `bidirect-align-dev-traces/scripts/compositional_generalization.py`.

---

## Which metrics to use

Most analyses need only JSD and entropy. Add others when:

| Question | Metric to add |
|---|---|
| Is the agent consistent within itself? | Within-group pairwise JSD |
| Does it rely on one approach or many? | Effective vocabulary size or Gini |
| Is this trajectory good or bad? | Procedural reward score |
| Did fine-tuning change the procedure? | All of the above via `lineage_diff` |
| Are failures structurally different from patches? | Edit-certificate Jaccard |
