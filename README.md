# procgrep

Procedural fingerprinting of LLM coding-agent rollouts.

## Intent

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
paradigm-by-scaffold cells.

## Capabilities

1. **Canonicalize heterogeneous trace formats** into a shared atom
   alphabet. Built-in adapters for SWE-agent, Agentless, DARS, and
   Moatless; custom adapters plug in through a `TraceAdapter` protocol.
2. **Learn a BPE motif vocabulary** from a corpus of canonical-atom
   sequences. Vocabulary size `V` is configurable; the learned
   vocabulary is a persisted, hashable artifact that downstream
   commands consume by file path.
3. **Encode trajectories as motif-frequency distributions** under a
   fixed vocabulary, with L1 normalization.
4. **Compute Jensen-Shannon divergence** between fingerprints at any
   group granularity: per-agent, per-cell, per-controlled-eval-arm.
5. **Project the fingerprint space** with UMAP for visual inspection.
6. **Run a leave-one-group-out predictive probe** to test whether
   procedural structure transfers across groups, or whether each
   group is structurally distinct.
7. **Match procedural patterns** against trace sequences via a YAML
   rule format. Each rule is a regex over the atom sequence with a
   `must_hold` flag. The compositional invariant DSL (procedural-DSPy)
   is future work; this library ships the pattern-matcher only.

## Use-cases

### 1. Cross-scaffold structural audit

You have N agents across M scaffolds attempting the same benchmark.
You want to know whether the scaffold dominates structure, or the
training paradigm does, or the base model does. Fingerprint each
agent, build the JSD matrix, and inspect the resulting block
structure. The leave-one-scaffold-out probe quantifies predictive
transfer.

### 2. Procedural diff for two agents

Two agents produce comparable success rates on the same benchmark.
Are they doing the same thing? Fingerprint both, compute the JSD,
and rank motifs by their contribution to the divergence.

### 3. Within-scaffold paradigm signature

Within one scaffold cell, you have agents trained under different
paradigms (for example, dense-RLHF vs extended-thinking). Fingerprint
each, compare the motif distributions, and surface paradigm-specific
signatures like stuck-edit-loops.

### 4. Saturation and coverage probing

Sweep the BPE vocabulary size `V` and measure when the JSD matrix
stabilizes. Identifies the procedural resolution at which group
distinctions emerge.

### 5. Controlled evaluations

Hold base model, scaffold, and benchmark fixed; vary one factor
(temperature, seed, RLHF variant, scaffold flag). Capture N traces
per arm. `procgrep` measures within-arm procedural consistency and
across-arm structural sensitivity to the factor, including a
leave-one-arm-out predictive probe.

## Non-goals (MVP)

- No compositional invariant DSL with temporal operators. The pattern
  matcher in `procgrep.patterns` ships regex over atom sequences only.
  The full DSL (procedural-DSPy) is future work.
- No probabilistic invariants over the procedural distribution.
- No live agent execution or model calls. `procgrep` is a post-hoc
  analysis library.
- No web dashboard. Figures emit as static PNG or SVG.

## Installation

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

## Examples

### CLI: build a fingerprint and a JSD matrix

```bash
# Canonicalize raw traces into atom sequences.
procgrep canonicalize \
    --input traces/raw/*.jsonl \
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

### Python: fingerprint + compare two agents

```python
from pathlib import Path

from procgrep import canonicalize, encode, fit_bpe, jsd_matrix, load_traces

traces = load_traces(Path("traces/raw"))
atoms = canonicalize(traces, adapter="swe-agent")

vocab = fit_bpe(atoms, vocab_size=200, seed=0)
fingerprints = encode(atoms, vocab=vocab)

matrix = jsd_matrix(fingerprints, group_by="agent")
print(matrix.to_records())
```

### Python: match a procedural pattern

```python
from procgrep import load_patterns, match_patterns

patterns = load_patterns("rules/stuck_edit_loop.yaml")
report = match_patterns(atoms, patterns)
for trace_id, violations in report.violations.items():
    print(trace_id, violations)
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

## Coding-style anchors

- Python 3.10+, `from __future__ import annotations` at the top of every module.
- Type hints on every public signature; `mypy --strict` clean.
- `ruff check` + `ruff format` clean; `pre-commit` hooks enforce this on commit.
- Single-purpose functions, top-down readable modules, docstrings on the public API.
- Deterministic seeds for any randomness; default seed is `0`.

## Citation

If you use `procgrep` in research, please cite the paper that this
library was extracted from. A BibTeX entry will be added on first
release.

## License

MIT. See [LICENSE](LICENSE).
