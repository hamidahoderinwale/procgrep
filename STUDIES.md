# Studies

How to use `procgrep` to answer real questions about agent behavior, with worked examples following the scientific method.

---

## How an experiment works

You have two versions of an agent, maybe different temperatures, different scaffolds, different training. You want to know: do they actually behave differently, and if so, how?

**The idea.** Run both versions on the same set of tasks. Convert their action logs to procedure distributions. Compare those distributions. The comparison tells you whether there's a real structural difference, and the discriminative procedure analysis tells you what that difference looks like concretely.

**Arms.** Each version of the agent you're testing is called an arm. If you're testing temperature=0.2 vs temperature=0.8, those are two arms. Everything else (the model, the scaffold, the tasks) stays the same. That's how you know the temperature is what caused any difference you find.

**The noise floor.** Before comparing two arms to each other, check how consistent each arm is with itself. If arm A varies a lot internally (its trajectories look different from each other), a small difference between arm A and arm B might just be noise. The within-arm divergence score is your baseline: across-arm differences only mean something if they're larger than this.

---

## Worked example: does a change actually help?

**Scenario.** You tuned a prompt to make your agent search the codebase before editing. Your eval shows a modest pass-rate improvement (+3pp). You want to know if the procedure actually changed, or if you just got lucky on a few tasks.

### Step 1: State your hypothesis

> "The new prompt causes the agent to search before editing more often. This procedural change is what drives the pass-rate improvement."

This is falsifiable: if the procedure didn't change, the pass-rate improvement was from something else.

### Step 2: Collect traces

Run both versions on the same 50 tasks. Save the trace logs, one JSONL
per arm, one trajectory per line.

### Step 3: Convert to a shared representation

```python
from procgrep import canonicalize, fit_bpe, encode
from procgrep.io import read_jsonl

traces_before = canonicalize(list(read_jsonl("traces_before.jsonl")), adapter="swe-agent")
traces_after  = canonicalize(list(read_jsonl("traces_after.jsonl")),  adapter="swe-agent")
all_traces = traces_before + traces_after

# Learn what action sequences look like across both versions together
vocab = fit_bpe([t.atoms for t in all_traces], vocab_size=64, seed=0)

fps_before = encode(traces_before, vocab=vocab)
fps_after  = encode(traces_after,  vocab=vocab)
```

### Step 4: Check your noise floor

```python
from procgrep import jsd
import numpy as np

# How consistent is each arm with itself?
def within_jsd(fps):
    dists = [fp.distribution() for fp in fps]
    pairs = [(dists[i], dists[j]) for i in range(len(dists)) for j in range(i+1, len(dists))]
    return np.mean([jsd(a, b) for a, b in pairs[:200]])  # sample for speed

noise_before = within_jsd(fps_before)
noise_after  = within_jsd(fps_after)
print(f"Within-arm JSD: before={noise_before:.3f}  after={noise_after:.3f}")
```

If both are around 0.20, you need an across-arm difference well above 0.20 to conclude something real changed.

### Step 5: Measure the across-arm difference

```python
mean_before = np.mean([fp.distribution() for fp in fps_before], axis=0)
mean_after  = np.mean([fp.distribution() for fp in fps_after],  axis=0)
across = jsd(mean_before, mean_after)
print(f"Across-arm JSD: {across:.3f}")

if across > 2 * max(noise_before, noise_after):
    print("Real procedural difference, worth investigating further")
else:
    print("Difference is within noise: the procedure didn't change much")
```

### Step 6: Find what changed

```python
from procgrep import discriminative_procedures

# set agent="before" / "after" on the Trace objects in step 3
top = discriminative_procedures(
    fps_before + fps_after,
    vocab,
    group_a="before",
    group_b="after",
    k=10,
    ranking="log_odds",
)

print("Procedures that increased after the change:")
for m in top:
    if m.p_b > m.p_a:
        print(f"  {m.procedure}  before={m.p_a:.3f}  after={m.p_b:.3f}")
```

If your hypothesis was right, you'd see `search_repo → read_file → think` appear in the "increased" list. If instead you see `edit → edit → edit` increased, the prompt had an unintended effect.

### Step 7: Check if the changed procedures predict outcomes

```python
# Split each arm's trajectories by outcome (pass/fail)
# Then check whether the discriminative procedures correlate with pass

pass_fps  = [fp for fp, t in zip(fps_after, traces_after) if t.metadata.get("resolved")]
fail_fps  = [fp for fp, t in zip(fps_after, traces_after) if not t.metadata.get("resolved")]

outcome_top = discriminative_procedures(
    pass_fps + fail_fps, vocab,
    group_a="pass", group_b="fail",
    k=5, ranking="log_odds",
)

print("Procedures that predict passing:")
for m in outcome_top:
    if m.p_a > m.p_b:
        print(f"  {m.procedure}  pass={m.p_a:.3f}  fail={m.p_b:.3f}")
```

If the procedures that increased after your change are the same ones that predict passing, your hypothesis holds: the prompt change caused a procedural change, and that procedural change drives outcomes.

If they don't overlap, the pass-rate improvement is likely from task-selection luck or some other mechanism, not the procedure you intended to change.

---

## Shipped studies

Each has a runnable example; the docstring in the example is the full spec.

| Study | Question | Example |
|---|---|---|
| Agent attribution baseline | Do BPE procedures identify an agent better than raw atom frequencies? | `examples/python/10_agent_attribution.py` |
| Within-trajectory drift | Are fingerprints stationary along a trajectory? | `examples/python/11_within_trajectory_drift.py` |
| Cross-language attribution | Does a fingerprint transfer across held-out languages (gumtree atoms)? | `examples/python/13_cross_language_attribution.py` |

## Study ideas

Same shape as the worked example: hypothesis, data, measurement, falsifier.

| Idea | One-line hypothesis |
|---|---|
| Temperature sweep | Across-arm JSD has a knee at some critical temperature, not a linear rise. |
| Outcome from prefixes | A length-K procedural prefix predicts pass/fail above base rate; the knee in accuracy-vs-K marks the earliest reliable signal. |
| Prompt-optimizer audit | DSPy compilation shifts the procedural fingerprint, not just accuracy. |
| Reasoning-budget sweep | Fingerprints shift non-monotonically with thinking budget; high budgets show stuck-think loops. |
| Pattern-matcher cross-check | The Level 1 matcher reproduces the BPE-derived claims on the same corpus. |
| Seed sensitivity | Establishes the JSD noise floor at fixed temperature. |
| Cross-scaffold replication | The three-regime claim holds on a held-out benchmark. |
| Difficulty strata | Procedures differ by task difficulty (gold-patch size). |
| New scaffolds | Aider / OpenHands / Cline / Roo fit the paper's cells, or define new ones. |
| Cross-benchmark portability | Fingerprints transfer across SWE-bench variants. |
| Tool-restriction ablations | Disabling run_test or capping edits changes procedure predictably. |
| Cost regression | Fingerprints predict cost per task. |
| Version drift | Fingerprints track what changed between model releases. |

## Notes on running a study

- Fit BPE **once across all arms**, then encode each arm under that
  shared vocabulary. Refitting per arm makes the vocabulary itself
  arm-specific and breaks the comparison.
- Stratify by cell or sub-population when interpreting accuracy
  metrics. Pooled accuracy across cells with different base rates
  is easy to misread.
- Report within-arm JSD alongside across-arm JSD. The former is the
  noise floor; only across-arm JSD significantly above it constitutes
  signal.
- Save the learned vocabulary as JSON and check it into version
  control. The vocabulary is the bridge between raw traces and every
  downstream artifact; making it reproducible is the cheapest way to
  make the whole pipeline reproducible.
