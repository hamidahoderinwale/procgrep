# procgrep

`procgrep` reads agent trace logs and tells you how agents differ in *how they work*, not just whether they passed.

[**Live explorer**](https://midah-procgrep-explorer.hf.space) · [**Interactive essay**](https://hamidah.me/procgrep) · [**Paper (arXiv)**](https://arxiv.org/abs/2606.16988)

## What it does

When a coding agent attempts a task, it leaves a sequence of actions: search the repo, read a file, edit a file, run tests, submit. `procgrep` converts those sequences into comparable representations and answers questions like:

- **Are these two agents doing the same thing?** Feed both sets of traces and get a divergence score.
- **Which agent is more consistent?** Measure how much an agent's procedure varies across tasks.
- **What makes one agent fail where another succeeds?** Find the action-sequence patterns exclusive to failures.
- **Did fine-tuning change how the model works, not just whether it passes?** Compare parent and child trajectories across four structural axes.
- **Is this trajectory heading toward failure?** Match against known failure patterns early enough to act on it.
- **Did my intervention actually change anything?** Identically configured agents already differ run to run; procgrep measures that noise floor so a claimed effect is read against it, not against zero.

Beyond reading traces, procgrep lets you *program* a target procedure: specify it, compile it to a reward, decode mask, guard, or scaffold config, and verify the result against the noise floor of same-condition runs.

## Quickstart

```bash
pip install procgrep
```

Three ways in: the `procgrep` CLI (`procgrep report <traces.jsonl | HF dataset id>` prints a one-shot corpus overview), the Python API below, or the hosted [live explorer](https://midah-procgrep-explorer.hf.space) with nothing installed.

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

## Core concepts

**Atoms.** Each agent action is normalized to a canonical type: `localize`, `search_repo`, `read_file`, `edit`, `run_test`, `create_file`, `delete_file`, `submit`, `think`, with `error` and `other` as catch-alls. This shared alphabet makes agents on different scaffolds comparable. Adapters can emit finer-grained atoms where they help (node-typed AST edits for GumTree traces); the procedures layered on top are learned per corpus, not fixed.

**Trajectory.** One agent attempting one task, represented as an ordered list of atoms. This is what `procgrep` ingests.

**Procedures.** Recurring multi-step patterns learned from the corpus via BPE (the same algorithm used to tokenize text for language model training). A procedure might be `search_repo → read_file → think`, a pattern frequent enough to be worth naming.

**Fingerprint.** A trajectory encoded as a distribution over procedures: how often each appeared. Two trajectories with similar fingerprints approached the problem similarly. It is a plain count vector over the procedure vocabulary, not a hash or an embedding: unlike a hash, two trajectories can legitimately share one; unlike an embedding, you can read exactly which procedures make two fingerprints differ.

**JSD.** Jensen-Shannon divergence: how different two fingerprints are. 0 = identical, 1 = completely non-overlapping. Used to compare agents, groups, or training conditions.

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

The `examples/rules/known_failure_patterns.yaml` file includes patterns validated on SWE-bench: edit streaks without tests, stuck reading loops, no exploration before editing. The matcher expresses contiguous runs, prefix requirements, and absence over the atom sequence, not temporal windows, variable binding, or probabilistic thresholds.

### Score a trajectory against a procedural spec

A reward spec defines what a good trajectory looks like: which phases it should go through, which patterns are failures, which are bonuses, and returns a 0–1 score per trajectory.

```python
from procgrep.reward import load_spec, score

spec = load_spec("examples/rules/reward_spec_swe_agent.yaml")
result = score(trajectory.atoms, spec)

print(result.score)                # e.g. 0.75
print(result.satisfied_phases)     # ["exploration", "implementation", "test_verification"]
print(result.triggered_penalties)  # ["stuck_reading"]
```

The `stuck_reading` penalty (`read_file → think → read_file → think`) fires in the first 12 steps on 80% of trajectories that eventually fail, usable as a real-time circuit breaker before the context budget is exhausted.

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

## Programming procedures

procgrep turns a target procedure into something you specify, hand to a scaffold, and verify, with no model in the loop.

A `ProcedureSpec` is the unit: a declarative, validated description of the procedure you want, learnable from the trajectories that passed.

```python
from procgrep import ProcedureSpec, enforce, verify, optimize, fit_bpe

# Derive a spec from the winning trajectories (e.g. test-after-edit, no long edit streaks)
vocab = fit_bpe([t.atoms for t in traces], vocab_size=64, seed=0)
spec = ProcedureSpec.from_winners(traces, vocab)

# Render it for a scaffold to apply. procgrep emits the artifact; it never runs the agent.
swe_cfg = enforce(spec, mode="prompt", scaffold="swe-agent")   # a SWE-agent config fragment
skill   = enforce(spec, mode="prompt", scaffold="openhands")   # an OpenHands SKILL.md
guard   = enforce(spec, mode="guard")                          # patterns + a streaming check
reward  = enforce(spec, mode="reward")                         # a dense per-step process reward for RL
decode  = enforce(spec, mode="decode")                         # an allowed(prefix) mask for constrained decoding

# Score any trajectory against the spec
result = spec.score(trajectory.atoms)   # result.score, result.satisfied_phases, ...

# Tune the spec's caps and phase set to better separate winners from losers (offline, model-free)
best_spec, opt_report = optimize(spec, traces)

# After running an agent under the spec, check whether behavior (and outcome) moved
report = verify(before_traces, after_traces, spec, vocab)
print(report.behavior_moved, report.outcome_delta, report.verdict)
```

`verify` answers two questions an outcome metric cannot: did behavior change, and did that change move the result. Both are read against the same-condition floor: identical configurations already differ across runs (in ours, by more than most published intervention effects), so the reference point is never zero.

`enforce` emits four artifact modes: `prompt`, `guard`, `reward` (a dense per-step process reward), and `decode` (an `allowed(prefix)` mask for constrained decoding). `optimize` tunes a spec's caps and phases offline. All are model-free: they emit artifacts or score traces; none runs an agent.

[`runner/`](runner/) (`procgrep-runner`, a separate package) runs paired baseline/enforced arms via mini-swe-agent, seals a run manifest, and feeds the traces back to `verify` with paired-bootstrap confidence intervals. It lives outside the core so procgrep only emits and measures, which is what keeps the measurements reproducible.

## CLI

```bash
# One-shot overview of any corpus: what agents, lengths, action mix, top procedures
procgrep report traces/canonical.jsonl
procgrep report nebius/SWE-rebench-openhands-trajectories --limit 300
```

which streams the dataset, auto-picks an adapter, and prints:

```
adapter    openhands  (confidence 0.95)

corpus            300 traces from nebius/SWE-rebench-openhands-trajectories
length            median 105 atoms, mean 111
exact duplicates    0.0%

action mix        think 44%  read_file 15%  run_code 11%  search_repo 9%  run_test 6%  edit 5%

procedures        64 learned; top by share:
   4.9%  read_file▁think
   4.5%  read_file▁think▁read_file▁think
   4.2%  run_test▁think
```

Traces that fail to parse are counted, not averaged in: an unmatched corpus
reports `parse yield 0/60 non-empty (adapter mismatch? try --dry-run)`.

```bash
# Ask in English; the model compiles the question to a regex ONCE, matching
# stays deterministic grep. Prints the regex + what it literally matches, and
# refuses questions the regex layer can't express. Needs ANTHROPIC_API_KEY.
procgrep ask "did it submit without ever running a test?" nebius/SWE-rebench-openhands-trajectories

# Watch a rollout live: tail a file of atoms (or --demo) in a local web view
procgrep watch --demo

# Convert raw traces to canonical atom sequences
procgrep canonicalize --input traces/raw.jsonl --adapter swe-agent --output traces/canonical.jsonl

# Learn a procedure vocabulary
procgrep fit-bpe --input traces/canonical.jsonl --vocab-size 64 --output vocab.json

# Encode and compare
procgrep encode --input traces/canonical.jsonl --vocab vocab.json --output fingerprints.jsonl
procgrep jsd --input fingerprints.jsonl --group-by agent --output jsd_matrix.json

# Inspect the procedure hierarchy: see which sub-procedures recur and compose
procgrep vocab-tree --vocab vocab.json          # or --input traces/canonical.jsonl
```

### The procedure hierarchy

BPE builds procedures bottom-up, so the vocabulary is a hierarchy: every merged procedure decomposes into the two tokens it was glued from, down to atoms. `vocab-tree` (and `procgrep.render_vocab_tree`) renders it. On the bundled sample traces:

```bash
procgrep canonicalize --input examples/data/synthetic_traces.jsonl --adapter swe-agent --output canonical.jsonl
procgrep vocab-tree --input canonical.jsonl
```

```text
5 atoms: edit, read_file, run_test, search_repo, submit
6 merges, 3 maximal procedures:

edit -> edit -> edit -> edit
  edit -> edit
    edit
    edit
  edit -> edit
    edit
    edit

edit -> run_test -> edit -> run_test -> submit
  edit -> run_test
    edit
    run_test
  edit -> run_test -> submit
    edit -> run_test
      edit
      run_test
    submit

search_repo -> read_file
  search_repo
  read_file
```

Read the trees as the corpus's habits: an edit-streak block, an edit-test loop that ends in submit, and the search-then-read pair.

## Installation

```bash
pip install procgrep           # core
pip install procgrep[reward]   # includes pyyaml for reward spec scoring
pip install procgrep[dev]      # development dependencies
```

## Notes

- Python 3.10+. No LLM SDK required; procgrep never calls a model.
- Adapters cover SWE-agent, OpenHands, Agentless, and more, plus interactive sessions (`claude-code`, Cursor). To add a scaffold, register a `TraceAdapter`; see `examples/python/05_custom_adapter.py`.
- **Try it on your own sessions.** `python examples/procgrep_view.py` opens the panel on your local Claude Code and Cursor sessions.
- **Privacy.** Ingest keeps only atoms and hashed identifiers; `to_shareable()` never exports prompts, code, or paths.
- **procgrep is accurate about the log, not the behavior.** Steps a scaffold never surfaces as tool calls are invisible (one Moatless trace hid 71 internal test runs); audit each new adapter once.

## Where things live

- [`examples/`](examples/) — runnable demos
- [`docs/`](docs/) — the essay and figures, plus [METRICS.md](docs/METRICS.md) (measurement reference) and [STUDIES.md](docs/STUDIES.md) (worked case studies)
- [`hf-space/`](hf-space/) — the live-explorer backend
- [`midah/procgrep-spines`](https://huggingface.co/datasets/midah/procgrep-spines) — the precomputed spine store; regenerate with `analysis/build_spines.py`
- Cite via `CITATION.cff` (GitHub's *Cite this repository*), with the paper as the primary citation
