# Suggested studies

A ranked list of case studies that `procgrep` is well-suited to. Each
entry names a hypothesis, the data shape required, the expected
finding (and the result that would falsify it), and the `procgrep`
capabilities the study exercises. The studies use the existing
public API; no new code is required.

Ranking weights interestingness, practical leverage, methodological
contribution, and how directly the study cites back to the paper.

## Glossary

- **Controlled eval**: hold all factors fixed, vary one, capture N
  traces per arm.
- **Within-arm JSD**: procedural consistency of an arm. Treated as
  a noise floor when interpreting larger JSD values.
- **Across-arm JSD**: procedural sensitivity to the varied factor.
  The signal of interest.
- **Procedural prefix**: the first K atoms of a trajectory, used as
  a predictor before completion.
- **NL layer / procedural layer**: prompt-shape vs trajectory-shape.
  DSPy lives in the first, `procgrep` in the second.

## Top five

### 1. Temperature-sweep procedural phase transition

- **Hypothesis.** Below some critical temperature `T*`, procedural
  fingerprints stay tight (across-arm JSD low); above `T*`, they
  fan out non-linearly. The knee is detectable as a discontinuity in
  the JSD-vs-T curve.
- **Data.** One agent, T in {0, 0.2, 0.4, 0.6, 0.8, 1.0}, N=30 per
  arm. Captured under matched seeds for everything but T.
- **Falsifier.** If across-arm JSD grows linearly with T (no knee),
  or stays at noise-floor for all T, there is no phase transition to
  characterize.
- **Capabilities exercised.** canonicalize, fit_bpe (shared vocab
  across arms), encode, jsd_matrix, umap_project, leave_one_group_out.
- **Why it ranks #1.** Practical leverage (T-setting guidance for
  production), publishable framing (procedural phase transition),
  uses every primitive.

### 2. Success-vs-failure procedural prefix signature

- **Hypothesis.** A classifier trained on procedural prefixes of
  length K can predict trajectory outcome (resolved vs partial vs
  failed) with accuracy above the per-cell base rate. The accuracy-
  vs-K curve has a knee that identifies the earliest reliable
  signal.
- **Data.** Existing corpus (no new collection required); join
  outcome labels to traces and slice each sequence to K atoms for
  K in {3, 5, 10, 20, full}. Stratify per cell to prevent the
  classifier from cheating on cell-level base rates.
- **Falsifier.** Accuracy at chance for every K, in every cell.
- **Capabilities exercised.** canonicalize, fit_bpe (refit per K
  so the vocabulary fits short prefixes), encode, leave_one_group_out
  with cells as groups.
- **Why it ranks #2.** Deployment-grade signal; a practitioner can
  early-stop bad trajectories. Reuses existing corpus.

### 3. Task-controlled procedural comparison

- **Hypothesis.** Two findings on the same setup, where `group` is
  composed from `instance_id` together with `agent` or `outcome`
  rather than from cell labels alone.

  (a) **Cross-agent at fixed task.** Holding `instance_id` fixed,
  the cross-agent JSD averaged across instances remains in the high
  range the paper reports across cells. Removes task heterogeneity
  as a possible confound of the headline cross-scaffold claim.

  (b) **Success-vs-failure at fixed task.** On instances where two
  trajectories share `instance_id` but differ in `outcome` (one
  `resolved`, one `unresolved`), the JSD between the paired
  trajectories is well above the within-arm noise floor, and
  `discriminative_motifs` names which motifs carry the gap.

- **Data.** Existing corpus. One load-time preprocessor pulls
  `instance_id` and `outcome` from the eval JSON into
  `Trace.metadata`, then composes the `group` label per analysis.
  No new collection.
- **Falsifier.** (a) Same-task cross-agent JSD drops to within-cell
  levels — would mean task identity was carrying the paper's
  cross-scaffold gap. (b) Same-task success-vs-failure JSD is
  indistinguishable from within-arm noise — would mean outcome
  lacks a procedural signature once task is held fixed.
- **Capabilities exercised.** canonicalize, fit_bpe (one shared
  vocabulary across the whole corpus, not per partition), encode,
  jsd_matrix (varying `group_by`), discriminative_motifs.
- **Why it ranks here.** (a) is the methodologically strongest
  version of the paper's cross-scaffold claim — same data, stronger
  control. (b) complements #2 by replacing the prefix-classifier
  view with a same-instance JSD pair, producing motif-level findings
  that map directly to Level 1 pattern rules. Both analyses share
  one data-prep step; the marginal cost of running both once the
  preprocessor exists is one `group_by` argument change. See
  `examples/python/06_task_controlled.py` for a worked example on a
  synthetic SWE-bench-shaped fixture.

### 4. DSPy compile-time procedural audit (the two-layers demo)

- **Hypothesis.** DSPy compilation (e.g., MIPRO over a SWE-bench-
  Lite split) shifts the downstream procedural fingerprint in a
  structured way, not only the task accuracy.
- **Data.** Same agent before and after DSPy compile, N=50 traces
  each, matched task split.
- **Falsifier.** Either no detectable procedural shift (suggests
  procedure is driven by scaffold not prompt; itself an interesting
  null), or shift orthogonal to outcome-relevant motifs.
- **Capabilities exercised.** canonicalize, fit_bpe (shared vocab),
  encode, jsd, optional patterns to spot specific motif changes.
- **Why it ranks #3.** Cleanest empirical demonstration of the
  two-layers framing the paper uses. Pairs `procgrep` with DSPy
  rather than competing with it.

### 5. Reasoning-budget sweep

- **Hypothesis.** Procedural fingerprint shifts non-monotonically
  with reasoning budget. High budgets produce stuck-think-loops
  with a distinct motif signature, analogous to the stuck-edit-loop
  identified at the action layer.
- **Data.** One extended-thinking model, max-thinking-tokens in
  {0, 2k, 8k, 32k}, N=30 per arm.
- **Falsifier.** Monotone or flat JSD-vs-budget curve.
- **Capabilities exercised.** Same harness as #1.
- **Why it ranks #5.** Complementary to study #1; uses the same
  controlled-eval recipe; relevant to current reasoning-model
  discussion.

### 6. Pattern-matcher cross-validation

- **Hypothesis.** The Level 1 pattern matcher in
  `procgrep.patterns` reproduces the paper's BPE-derived qualitative
  claims when run on the same corpus. Specifically: stuck-edit-loop
  violations concentrate in SWE-agent extended-thinking; localize-
  before-edit holds strictly in Agentless and Moatless but loosely
  in dense-RLHF SWE-agent.
- **Data.** Existing corpus + `examples/rules/stuck_edit_loop.yaml`.
- **Falsifier.** Per-cell violation rates do not differ by more
  than the within-cell trace-to-trace variation.
- **Capabilities exercised.** canonicalize, patterns.
- **Why it ranks #6.** Cheap, second-method confirmation of the
  paper's BPE story, demonstrates the Level 1 matcher's usefulness.
  The expected output (per-cell violation rates) is exactly the
  demo paragraph the paper's §Implications can host.

## Lower tier

| # | Study | Best feature |
|---|---|---|
| 7 | Seed sensitivity at fixed T | Establishes the JSD noise floor; foundational. |
| 8 | Cross-scaffold replication on a held-out benchmark | Validates the paper's three-regime claim on new data. |
| 9 | Difficulty-stratified procedures (easy/medium/hard by gold-patch size) | Cheap; pairs with the prefix-signature study. |
| 10 | Newer scaffolds (Aider, OpenHands, Cline, Roo) plotted against the paper's cells | Catalogs whether new scaffolds fit existing cells or define new ones. |
| 11 | Cross-benchmark generalization (SWE-bench-Lite vs Verified vs HumanEval) | Tests fingerprint portability across task sets. |
| 12 | Tool-restriction ablations (disable run_test, force one edit per turn) | Mechanistic; clean controlled-eval shape. |
| 13 | Cost-vs-procedure regression (predict $/task from fingerprint) | Practitioner-facing; ties procedure to budget. |
| 14 | Model-version longitudinal drift (GPT-4 to GPT-4o to GPT-4-Turbo) | Tracks "what changed" in releases. |

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
