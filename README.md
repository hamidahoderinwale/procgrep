# procgrep

Procedural fingerprinting of LLM coding-agent rollouts.

### Intent

`procgrep` answers one specific kind of question: *do two agents, holding
some factors fixed and varying others, produce structurally distinct
procedures?* Given trace logs from agents attempting coding tasks, it
canonicalizes those traces into a shared atom alphabet, learns a BPE
motif vocabulary over the canonical sequences, encodes each trajectory
as a motif-frequency distribution, and supports cross-group comparison
via Jensen-Shannon divergence, leave-one-group-out predictive probes,
UMAP projection, and pattern matching.

It is post-hoc analysis. `procgrep` does not run agents, call models,
or require an LLM SDK. It reads trace files and emits structural
artifacts.

The library was extracted from the analysis pipeline used in *Procedural
Grep: Structural Variation for Agent Rollouts*, where it characterized
the procedural fingerprints of nine LLM coding agents across five
paradigm-by-scaffold configurations.

### Modes of use

Three ways `procgrep` is typically applied, ordered by how mature each
is in the current MVP.

1. **Research instrument.** Build a JSD matrix and leave-one-group-out
   probe over a corpus of agents to characterize where procedural
   structure lives (scaffold vs paradigm vs base model). This is the
   primary supported mode and the one the paper exercises.
2. **Controlled-eval harness.** Hold base model, scaffold, and
   benchmark fixed; vary one factor (temperature, seed, RLHF variant,
   scaffold flag). `procgrep` measures within-arm procedural
   consistency and across-arm structural sensitivity. See the recipe
   under "Controlled evaluations" below.
3. **Deployment signal.** Run the Level 1 pattern matcher against
   procedural prefixes (first K atoms) of live trajectories to flag
   runs heading toward known failure shapes (stuck edit loops,
   missing localize-before-edit, missing tests-before-submit).

### Capabilities

1. **Canonicalize heterogeneous trace formats** into a shared atom
   alphabet. Built-in adapters for SWE-agent, Agentless, DARS, and
   Moatless; custom adapters plug in through a `TraceAdapter` protocol.
2. **Learn a BPE motif vocabulary** from a corpus of canonical-atom
   sequences. Vocabulary size `V` is configurable; the learned
   vocabulary is a persisted artifact that downstream commands consume
   by file path.
3. **Encode trajectories as motif-frequency distributions** under a
   fixed vocabulary, with L1 normalization.
4. **Compute Jensen-Shannon divergence** between fingerprints at any
   group granularity: per-agent, per-group, per-controlled-eval-arm.
5. **Project the fingerprint space** with UMAP for visual inspection.
6. **Run a leave-one-group-out predictive probe** to test whether
   procedural structure transfers across groups, or whether each
   group is structurally distinct.
7. **Match procedural patterns** against trace sequences via a YAML
   rule format. Each rule is a regex over the atom sequence with a
   `must_hold` flag. The compositional invariant DSL (procedural-DSPy)
   is future work; this library ships the pattern matcher only.
8. **Group-level descriptive and discriminative statistics**:
   top-K atom frequencies per group, effective vocabulary size
   (perplexity) per group, per-trajectory entropy summarized by
   group, and top-K motifs ranked by log-odds or by JSD
   contribution between any two groups. Suitable for controlled-eval
   summaries and per-group descriptive tables.

### Use-cases

#### 1. Cross-scaffold structural audit

You have N agents across M scaffolds attempting the same benchmark.
You want to know whether the scaffold dominates structure, or the
training paradigm does, or the base model does. Fingerprint each
agent, build the JSD matrix, and inspect the resulting block
structure. The leave-one-scaffold-out probe quantifies predictive
transfer.

#### 2. Procedural diff for two agents

Two agents produce comparable success rates on the same benchmark.
Are they doing the same thing? Fingerprint both, compute the JSD,
and rank motifs by their contribution to the divergence.

#### 3. Within-scaffold paradigm signature

Within one scaffold, you have agents trained under different
paradigms (for example, dense-RLHF vs extended-thinking). Fingerprint
each, compare the motif distributions, and surface paradigm-specific
signatures like stuck edit loops.

#### 4. Saturation and coverage probing

Sweep the BPE vocabulary size `V` and measure when the JSD matrix
stabilizes. Identifies the procedural resolution at which group
distinctions emerge.

#### 5. Controlled evaluations

Hold base model, scaffold, and benchmark fixed; vary one factor
(temperature, seed, RLHF variant, scaffold flag). Capture N traces
per arm. `procgrep` measures within-arm procedural consistency and
across-arm structural sensitivity to the factor, including a
leave-one-arm-out predictive probe.

**Recipe.** Three steps:

1. **Hold fixed.** Base model, scaffold, benchmark task set. Anything
   not under study stays constant across arms.
2. **Vary one factor.** Pick 3 to 8 levels (for example, T in
   {0, 0.2, 0.4, 0.6, 0.8, 1.0}). Capture N traces per arm. N >= 30
   is a safe default; the seed-sensitivity study in
   [STUDIES.md](STUDIES.md) establishes a principled floor.
3. **Encode and compare.** Assign each trace a `group` label equal to
   its arm. Fit one shared BPE vocabulary across all arms (so the
   vocabulary itself is not an arm-specific confound). Compute the
   JSD matrix over arm-mean fingerprints, then run
   `leave_one_group_out` with `label_field="group"` to test whether
   each arm is structurally novel.

### Non-goals (MVP)

- No compositional invariant DSL with temporal operators. The pattern
  matcher in `procgrep.patterns` ships regex over atom sequences only.
  The full DSL (procedural-DSPy) is future work.
- No probabilistic invariants over the procedural distribution.
- No live agent execution or model calls. `procgrep` is a post-hoc
  analysis library.
- No web dashboard. Figures emit as static PNG or SVG.

### How does this relate to DSPy?

`procgrep` and DSPy operate on different layers and are complementary,
not competing.

- **DSPy** is a natural-language layer instrument: it compiles and
  optimizes prompts and demonstrations that an agent receives.
- **`procgrep`** is a procedural-layer instrument: it characterizes
  the tool-call trajectory the agent produces *after* it has read its
  prompt.

`procgrep` does not depend on DSPy and does not call any LLM. The two
are useful together in the controlled-eval setting: a researcher can
optimize an agent with DSPy and use `procgrep` to measure whether the
NL-layer optimization produced a procedural-layer shift, or whether
procedure is mostly determined by scaffold and model rather than
prompt. The case study sketched as "DSPy compile-time procedural
audit" in [STUDIES.md](STUDIES.md) walks through this.

The name "procedural-DSPy" appears in the paper's future-work section
as a label for a compositional invariant DSL with temporal operators
over the procedural layer. That DSL is not part of the MVP.

### Installation

```bash
pip install procgrep
```

Or from source:

```bash
git clone https://github.com/hamidahoderinwale/procgrep.git
cd procgrep
pip install -e ".[dev]"
pre-commit install
```

### Examples

#### CLI: build a fingerprint and a JSD matrix

```bash
# Canonicalize raw traces into atom sequences.
procgrep canonicalize \
    --input traces/raw.jsonl \
    --adapter swe-agent \
    --output traces/canonical.jsonl

# Learn a BPE motif vocabulary at V=200.
procgrep fit-bpe \
    --input traces/canonical.jsonl \
    --vocab-size 200 \
    --output vocab.json

# Encode each trajectory as a motif-frequency distribution.
procgrep encode \
    --input traces/canonical.jsonl \
    --vocab vocab.json \
    --output traces/encoded.jsonl

# Compute the pairwise JSD matrix grouped by agent.
procgrep jsd \
    --input traces/encoded.jsonl \
    --group-by agent \
    --output jsd_matrix.json
```

The library ships a tiny synthetic corpus and rule file under
[`examples/`](examples/) so the pipeline above can be run end-to-end
without supplying your own traces.

#### Python: fingerprint and compare two agents

There are two equivalent entry points. The public API takes already-
adapted `Trace` objects; the `procgrep.io` helpers handle JSONL
serialization for callers that prefer to work with file paths.

```python
from procgrep import canonicalize, encode, fit_bpe, jsd_matrix
from procgrep.io import read_jsonl

# Read raw scaffold-specific JSONL and canonicalize via a built-in adapter.
raw_records = list(read_jsonl("traces/raw.jsonl"))
traces = canonicalize(raw_records, adapter="swe-agent")

# Learn a vocabulary over the canonical atom sequences.
vocab = fit_bpe((t.atoms for t in traces), vocab_size=200, seed=0)

# Encode and compute the JSD matrix.
fingerprints = encode(traces, vocab=vocab)
matrix = jsd_matrix(fingerprints, group_by="agent")
for record in matrix.to_records():
    print(record)
```

#### Python: group-level descriptive stats

```python
from procgrep import (
    atom_frequencies_per_group,
    effective_vocab_size_per_group,
    entropies_per_group,
    discriminative_motifs,
    canonicalize, fit_bpe, encode,
)
from procgrep.io import read_jsonl

traces = canonicalize(list(read_jsonl("traces/raw.jsonl")), adapter="swe-agent")
vocab  = fit_bpe((t.atoms for t in traces), vocab_size=200, seed=0)
fps    = encode(traces, vocab=vocab)

# Which raw atoms dominate each group?
print(atom_frequencies_per_group(traces, k=10, group_by="agent"))

# How diverse is each group's procedural vocabulary?
print(effective_vocab_size_per_group(fps, group_by="agent"))

# Per-trajectory entropy summary (median, IQR, range) per group.
print(entropies_per_group(fps, group_by="agent"))

# Top motifs separating two arms.
top = discriminative_motifs(
    fps, vocab,
    group_a="arm_temp_0_2",
    group_b="arm_temp_0_8",
    k=10,
    ranking="log_odds",
)
for m in top:
    print(m.motif, m.log_odds, m.p_a, m.p_b)
```

#### Python: match a procedural pattern

```python
from procgrep import canonicalize, load_patterns, match_patterns
from procgrep.io import read_jsonl

raw_records = list(read_jsonl("traces/raw.jsonl"))
traces = canonicalize(raw_records, adapter="swe-agent")

patterns = load_patterns("examples/rules/stuck_edit_loop.yaml")
report = match_patterns(traces, patterns)

print("pass rate per rule:", report.pass_rate_per_rule)
for trace_id, failing_rules in report.violations.items():
    print(trace_id, failing_rules)
```

Example rule file:

```yaml
rules:
  - name: no_long_edit_loops
    description: No run of 5 or more consecutive edits without an intervening test.
    pattern: "(edit ){5,}"
    must_hold: false

  - name: localize_before_edit
    description: A localize atom must precede the first edit atom.
    pattern: "localize .* edit"
    must_hold: true
```

### Suggested studies

See [STUDIES.md](STUDIES.md) for a ranked list of case studies that
`procgrep` is well-suited to, including the temperature-sweep phase
transition, success-vs-failure procedural prefix signature, and the
DSPy compile-time procedural audit.

### Notes

- Python 3.10+, `from __future__ import annotations` at the top of every module.
- Type hints on every public signature; `mypy --strict` clean.
- `ruff check` + `ruff format` clean; `pre-commit` hooks enforce this on commit.
- Single-purpose functions, top-down readable modules, docstrings on the public API.
- Deterministic seeds for any randomness; default seed is `0`.

### Citation

If you use `procgrep` in research, please cite the paper that this
library was extracted from. A BibTeX entry will be added on first
release.

### License

MIT. See [LICENSE](LICENSE).
