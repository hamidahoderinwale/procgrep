# procgrep

Procedural fingerprinting of LLM coding-agent trajectories.

### Intent

`procgrep` answers one question: *do two agents, holding some factors
fixed and varying others, produce structurally distinct procedures?*
It canonicalizes trace logs into a shared atom alphabet, learns a BPE
vocabulary, and encodes each trajectory as a procedure-frequency
distribution for cross-group comparison.

Post-hoc analysis only. `procgrep` does not run agents or call models.
It reads trace files and emits structural artifacts.

### Nomenclature

- **Atom.** The canonical, scaffold-agnostic unit of agent action. One
  assistant turn maps to one atom, plus an optional preceding
  `ATOM_THINK` when the turn carried a non-empty rationale. The built-in
  alphabet is small and finite (`ATOM_EDIT`, `ATOM_READ_FILE`,
  `ATOM_RUN_TEST`, `ATOM_SEARCH_REPO`, `ATOM_CREATE_FILE`,
  `ATOM_DELETE_FILE`, `ATOM_SUBMIT`, `ATOM_THINK`, `ATOM_OTHER`); custom
  alphabets are atoms too — anything `procgrep` treats as the
  indivisible procedural symbol.
- **Trajectory.** The ordered atom sequence for one rollout (one agent
  attempting one task). The unit `procgrep` ingests and compares.
- **BPE vocabulary (also called "procedures").** Learned via byte-pair
  encoding over a corpus of trajectories. Each procedure is a recurring
  sub-sequence of atoms (length ≥ 1) that the BPE merge process found
  frequent enough to be worth its own symbol. Vocabulary size `V` is
  configurable.
- **Procedural space.** The simplex of procedure-frequency distributions
  under a fixed BPE vocabulary. Each trajectory encodes to one point;
  group-level comparison (JSD, leave-one-group-out probe, discriminative
  procedures) operates over it.

### Modes of use

1. **Research instrument.** Build a JSD matrix and leave-one-group-out
   probe over a corpus of agents to characterize where procedural
   structure lives (scaffold vs paradigm vs base model). The mode the
   paper exercises.
2. **Controlled-eval harness.** Hold base model, scaffold, and
   benchmark fixed; vary one factor (temperature, seed, RLHF variant,
   scaffold flag). Measures within-arm consistency and across-arm
   sensitivity. See the "Controlled evaluations" recipe below.
3. **Deployment signal.** Run the Level 1 pattern matcher against
   prefixes (first K atoms) of live trajectories to flag known failure
   shapes (stuck edit loops, missing localize-before-edit, missing
   tests-before-submit).

### Capabilities

1. **Canonicalize heterogeneous trace formats** into a shared atom
   alphabet. Built-in adapters for SWE-agent, Agentless, DARS, and
   Moatless at the agent-action layer; a gumtree adapter for AST-edit-
   script traces (language-neutral across Python, JS, Java via the
   GumTreeDiff CLI). Custom adapters plug in through a `TraceAdapter`
   protocol.
2. **Learn a BPE vocabulary** from canonical-atom sequences. Vocabulary
   size `V` is configurable; the learned vocabulary is a persisted
   artifact downstream commands consume by file path.
3. **Encode trajectories as procedure-frequency distributions** under a
   fixed vocabulary, with L1 normalization.
4. **Compute Jensen-Shannon divergence** between fingerprints at any
   group granularity: per-agent, per-group, per-controlled-eval-arm.
5. **Project the fingerprint space** with UMAP for visual inspection.
6. **Run a leave-one-group-out predictive probe** to test whether
   procedural structure transfers across groups.
7. **Match procedural patterns** against trace sequences via a YAML
   rule format. Each rule is a regex over the atom sequence with a
   `must_hold` flag.
8. **Group-level descriptive and discriminative statistics**:
   - top-K atom frequencies per group
   - effective vocabulary size (perplexity) per group
   - per-trajectory entropy summarized by group
   - top-K procedures ranked by log-odds or by JSD contribution between any two groups

### Use-cases

#### 1. Cross-scaffold structural audit

N agents across M scaffolds on the same benchmark: does the scaffold
dominate structure, the training paradigm, or the base model?
Fingerprint each agent, build the JSD matrix, and inspect block
structure; the leave-one-scaffold-out probe quantifies predictive
transfer.

#### 2. Procedural diff for two agents

Two agents with comparable success rates on the same benchmark — are
they doing the same thing? Fingerprint both, compute the JSD, and rank
procedures by divergence contribution.

#### 3. Within-scaffold paradigm signature

Within one scaffold, compare agents trained under different paradigms
(e.g. dense-RLHF vs extended-thinking). Surface paradigm-specific
signatures like stuck edit loops.

#### 4. Saturation and coverage probing

Sweep BPE vocabulary size `V` and measure when the JSD matrix
stabilizes — identifies the procedural resolution at which group
distinctions emerge.

#### 5. Controlled evaluations

Hold base model, scaffold, and benchmark fixed; vary one factor
(temperature, seed, RLHF variant, scaffold flag). Measures within-arm
consistency and across-arm sensitivity, including a leave-one-arm-out
predictive probe.

**Recipe.** Three steps:

1. **Hold fixed.** Base model, scaffold, benchmark task set. Anything
   not under study stays constant across arms.
2. **Vary one factor.** Pick 3 to 8 levels (e.g. T in
   {0, 0.2, 0.4, 0.6, 0.8, 1.0}). Capture N trajectories per arm;
   N >= 30 is a safe default. The seed-sensitivity study in
   [STUDIES.md](STUDIES.md) establishes a principled floor.
3. **Encode and compare.**
   - Assign each trajectory a `group` label equal to its arm.
   - Fit one shared BPE vocabulary across all arms (so the vocabulary is not an arm-specific confound).
   - Compute the JSD matrix over arm-mean fingerprints.
   - Run `leave_one_group_out` with `label_field="group"` to test whether each arm is structurally novel.


### How does this relate to DSPy?

`procgrep` and DSPy operate on different layers and are complementary.

- **DSPy** is a natural-language-layer instrument: it compiles and
  optimizes the prompts and demonstrations an agent receives.
- **`procgrep`** is a procedural-layer instrument: it characterizes
  the tool-call trajectory an agent produces *after* reading its prompt.

`procgrep` does not depend on DSPy or call any LLM. The two compose in
the controlled-eval setting: optimize an agent with DSPy, then use
`procgrep` to measure whether the NL-layer optimization produced a
procedural-layer shift, or whether procedure is determined by scaffold
and model rather than prompt. See "DSPy compile-time procedural audit"
in [STUDIES.md](STUDIES.md).

The name "procedural-DSPy" appears in the paper's future-work section
as a label for a compositional invariant DSL with temporal operators
over the procedural layer. Not part of the MVP.

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

# Learn a BPE procedure vocabulary at V=200.
procgrep fit-bpe \
    --input traces/canonical.jsonl \
    --vocab-size 200 \
    --output vocab.json

# Encode each trajectory as a procedure-frequency distribution.
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
[`examples/`](examples/) so the pipeline above runs end-to-end without
supplying your own traces.

#### Fingerprint and compare two agents

Two equivalent entry points. The public API takes already-adapted
`Trace` objects; the `procgrep.io` helpers handle JSONL serialization
for callers working with file paths.

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

#### Group-level descriptive stats

```python
from procgrep import (
    atom_frequencies_per_group,
    effective_vocab_size_per_group,
    entropies_per_group,
    discriminative_procedures,
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

# Top procedures separating two arms.
top = discriminative_procedures(
    fps, vocab,
    group_a="arm_temp_0_2",
    group_b="arm_temp_0_8",
    k=10,
    ranking="log_odds",
)
for m in top:
    print(m.procedure, m.log_odds, m.p_a, m.p_b)
```

#### Match a procedural pattern

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

### Lineage diff (preview)

Given a parent model and a child produced by a documented training
procedure (distillation, SFT, RLHF, instruction tuning, version step),
characterize what the procedure did at the procedural level. Useful
for verifying whether a method paper's preservation claims hold up in
trajectories.

```python
from procgrep import lineage_diff

diff = lineage_diff(
    parent=parent_trajectories,    # canonical traces from base model
    child=child_trajectories,      # canonical traces from post-trained model
    on_tasks=swe_bench_verified,   # shared task suite
    along=[
        "vocabulary",     # procedures preserved / lost
        "entropy",        # mode-concentration shift
        "conditional",    # P(next-procedure | prefix) divergence
        "recovery",       # post-error procedural patterns
        "failures",       # which tasks fail with which signatures
        "ood",            # ID-vs-OOD decomposition (atom-cossim + text-embed)
    ],
)

print(diff.summary())          # human-readable findings table
diff.report.to_markdown()      # claim-by-claim audit output
```

The diff object composes the existing primitives (JSD matrix,
leave-one-group-out probe, discriminative procedures, BPE vocabulary)
into a structured characterization of the training-procedure delta.
Atom- and procedure-level primitives remain available for finer-grained
probing (see `atom_frequencies_per_group`, `discriminative_procedures`,
`entropies_per_group`). Pass `change_point_phases=True` for data-driven
phase detection — phase labels emerge from the trajectory distribution
rather than being imposed.

**What you can study with `lineage_diff`.** Given parent/child trajectory sets,
the following analyses fall out of the built-in axes and composable primitives:

*Direct from the four axes:*

- **Vocabulary preservation** (`vocabulary` axis). Does the child keep the
  parent's atom repertoire? Jaccard between atom sets.
- **Mode-collapse signal** (`entropy` axis). Shift in per-trajectory entropy
  bits. Output-only supervision (SFT, distillation) is expected to narrow
  the policy.
- **Asymmetric transmission** (`outcome_quadrant` axis). Compare `pass_jaccard`
  to `fail_jaccard`. Tests whether passing procedures inherit while failing
  ones diverge, or vice versa.
- **Where divergence lives** (`conditional` axis). Markov-conditional JSD
  over k-gram transitions surfaces *where in the sequence* the procedures
  part ways: early scoping, mid-edit, late testing.

*Composed from existing primitives:*

- **Procedural-transfer score.** Train a next-action predictor on parent
  rollouts; evaluate on child rollouts. Δ from the parent-on-parent baseline
  measures how much sequence structure transfers. Sharper than vocabulary
  Jaccard because it tests order, not just symbol-set overlap.
- **Failure-signature inheritance.** Check whether the parent's failure-
  correlated procedures (edit-streak runs, missing-localize-before-edit,
  missing-test-before-submit) reappear in the child at the same rates.
  Tests whether the post-training method transmits failure modes, not just
  success patterns.
- **Compositional-task transfer.** Same compositional task subset, parent
  vs child failure rate. Whether the method shifted compositional
  generalization, and in which direction.
- **Discriminative-procedure transfer.** Mine procedures over-represented
  in parent-failing rollouts via `discriminative_procedures`; check whether
  they reappear in child-failing rollouts.

*Requires logprobs / tokens (when you control inference):*

- **Branch entropy at decision tokens.** Per-token top-K logprobs let you
  compute Shannon entropy at each output token. Sampled at tool-call
  openings, this distinguishes a confidently-wrong child from a hesitantly-
  wrong one.
- **Structural vs natural-language divergence cross-tab.** Two distances
  per pair: structural (atom sequences) and NL (CoT / rationale embedding).
  The 2×2 separates faithful distillation, style-only changes, behavior-
  only changes, and full model swaps.

**Hierarchical multi-resolution (optional).** Pass `alphabet=["canonical",
"native"]` to compute each axis once per alphabet in one call. The
canonical layer is cross-comparable across scaffolds (catalog
aggregation); the native layer preserves scaffold-specific richness
(within-scaffold depth). Each `AxisResult.alphabet` records its view,
so a single `LineageDiff` carries both layers unambiguously. Pair with
`canonical_projection: Callable[[Atom], Atom]` to translate native
atoms to canonical at diff time. See
[`examples/python/15_multi_resolution_diff.py`](examples/python/15_multi_resolution_diff.py)
for a runnable demo on synthetic data.

### Suggested studies

See [STUDIES.md](STUDIES.md) for a ranked list of case studies
`procgrep` is well-suited to: the temperature-sweep phase transition,
success-vs-failure procedural prefix signature, and the DSPy
compile-time procedural audit.

### Notes

- Python 3.10+, `from __future__ import annotations` at the top of every module.
- Type hints on every public signature; `mypy --strict` clean.
- `ruff check` + `ruff format` clean; `pre-commit` hooks enforce this on commit.
- Single-purpose functions, top-down readable modules, docstrings on the public API.
- Deterministic seeds for any randomness; default seed is `0`.

### Citation

If you use `procgrep` in research, please cite the paper this library
was extracted from. A BibTeX entry will be added on first release.

### License

MIT. See [LICENSE](LICENSE).
