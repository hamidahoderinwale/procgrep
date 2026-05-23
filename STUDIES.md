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

### 3. DSPy compile-time procedural audit (the two-layers demo)

- **Hypothesis.** DSPy compilation (e.g., MIPRO over a SWE-bench-
  Lite split) shifts the downstream procedural fingerprint in a
  structured way, not only the task accuracy.
- **Data.** Same agent before and after DSPy compile, N=50 traces
  each, matched task split.
- **Falsifier.** Either no detectable procedural shift (suggests
  procedure is driven by scaffold not prompt; itself an interesting
  null), or shift orthogonal to outcome-relevant procedures.
- **Capabilities exercised.** canonicalize, fit_bpe (shared vocab),
  encode, jsd, optional patterns to spot specific procedure changes.
- **Why it ranks #3.** Cleanest empirical demonstration of the
  two-layers framing the paper uses. Pairs `procgrep` with DSPy
  rather than competing with it.

### 4. Reasoning-budget sweep

- **Hypothesis.** Procedural fingerprint shifts non-monotonically
  with reasoning budget. High budgets produce stuck-think-loops
  with a distinct procedure signature, analogous to the stuck-edit-loop
  identified at the action layer.
- **Data.** One extended-thinking model, max-thinking-tokens in
  {0, 2k, 8k, 32k}, N=30 per arm.
- **Falsifier.** Monotone or flat JSD-vs-budget curve.
- **Capabilities exercised.** Same harness as #1.
- **Why it ranks #4.** Complementary to study #1; uses the same
  controlled-eval recipe; relevant to current reasoning-model
  discussion.

### 5. Pattern-matcher cross-validation

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
- **Why it ranks #5.** Cheap, second-method confirmation of the
  paper's BPE story, demonstrates the Level 1 matcher's usefulness.
  The expected output (per-cell violation rates) is exactly the
  demo paragraph the paper's §Implications can host.

## Lower tier

| # | Study | Best feature |
|---|---|---|
| 6 | Seed sensitivity at fixed T | Establishes the JSD noise floor; foundational. |
| 7 | Cross-scaffold replication on a held-out benchmark | Validates the paper's three-regime claim on new data. |
| 8 | Difficulty-stratified procedures (easy/medium/hard by gold-patch size) | Cheap; pairs with the prefix-signature study. |
| 9 | Newer scaffolds (Aider, OpenHands, Cline, Roo) plotted against the paper's cells | Catalogs whether new scaffolds fit existing cells or define new ones. |
| 10 | Cross-benchmark generalization (SWE-bench-Lite vs Verified vs HumanEval) | Tests fingerprint portability across task sets. |
| 11 | Tool-restriction ablations (disable run_test, force one edit per turn) | Mechanistic; clean controlled-eval shape. |
| 12 | Cost-vs-procedure regression (predict $/task from fingerprint) | Practitioner-facing; ties procedure to budget. |
| 13 | Model-version longitudinal drift (GPT-4 to GPT-4o to GPT-4-Turbo) | Tracks "what changed" in releases. |
| 14 | Agent attribution stylometry baseline (LOGO probe with `label_field="agent"` + naive baselines) | Methodology pre-requisite for every other study; tells you whether BPE procedures carry information beyond raw atom frequencies. |
| 15 | Within-trajectory drift (prefix/middle/suffix slice fingerprints) | Tests the stationarity assumption every other study makes implicitly. Free; reuses existing corpora. |
| 16 | Cross-language attribution via gumtree AST atoms | Strongest form of the attribution claim: does an agent's procedural fingerprint transfer across held-out languages? Only possible once gumtree atoms replace the language-specific tool-surface atoms. |

## Studies 14-16: details

### 14. Agent attribution stylometry baseline

- **Hypothesis.** A LOGO probe trained on procedural fingerprints
  predicts the agent label on a held-out group meaningfully above
  three naive baselines: (a) raw atom-frequency distribution, (b)
  trajectory-length only, (c) majority-class.
- **Data.** Any existing multi-agent corpus with a sensible
  grouping field (task family, instance id, language). The
  bundled `examples/synthetic_gumtree_traces.jsonl` runs as a
  smoke test; the meaningful version uses a real same-task-
  multi-model corpus (held-out task family as the LOGO group).
- **Falsifier.** BPE-procedure accuracy at or below raw-atom accuracy
  in every fold. The procedure vocabulary then isn't earning its keep
  on this corpus.
- **Capabilities exercised.** `canonicalize`, `fit_bpe`, `encode`,
  `leave_one_group_out`, plus a small inline scikit-learn LOGO
  used for the atom-frequency and length-only baselines.
- **Why it ranks here.** It is the methodology pre-requisite for
  every other study: without a stylometry baseline, every "BPE
  finds X" claim is suspect because the simpler representation
  may already find X. Cheap; reuses any existing corpus.
- **Shipped example.** `examples/python/10_agent_attribution.py`.

### 15. Within-trajectory drift

- **Hypothesis.** Procedural fingerprints are *not* stationary
  along a trajectory. The mean fingerprint of the first third
  differs from the mean fingerprint of the last third by JSD
  meaningfully above the JSD between two random splits of the
  same third. A classifier predicts slice position from
  fingerprint above chance (1/3).
- **Data.** Any existing corpus; trajectories with fewer than 3
  atoms are dropped. The bundled gumtree fixture serves as the
  smoke test.
- **Falsifier.** Slice-to-slice JSD at the noise floor (the
  JSD between random splits of the same slice); LOGO probe
  accuracy near 1/3.
- **Capabilities exercised.** `canonicalize`, `fit_bpe`,
  `encode`, `jsd`, `leave_one_group_out`.
- **Why it ranks here.** Free, methodological. Negative result
  (no drift) validates the stationarity assumption every other
  study makes; positive result (drift) opens a "what changes
  along the trajectory" sub-study and motivates per-slice
  fingerprints throughout.
- **Shipped example.** `examples/python/11_within_trajectory_drift.py`.

### 16. Cross-language attribution via gumtree atoms

- **Hypothesis.** Using gumtree's AST-edit-script atoms (which
  are language-neutral at the operation layer), an agent's
  procedural fingerprint transfers across held-out languages.
  LOGO probe with language as the group and agent as the
  prediction target scores above chance.
- **Data.** A same-task-multi-language corpus where each (agent,
  language) pair has N ≥ 5 traces. Each trace is a gumtree edit
  script, generated by running `gumtree jsondiff <before>
  <after>` on the agent's start-of-task and end-of-task source
  files. The bundled `synthetic_gumtree_traces.jsonl` is a
  21-trace smoke fixture; the meaningful version uses real
  agent outputs.
- **Falsifier.** Held-out language accuracy at chance. Either
  the agents are too similar at the AST-operation layer, or the
  language-specific node-type vocabulary swamps the agent
  signal (e.g. Python's `Name` vs JS's `Identifier` vs Java's
  `SimpleName` are the same concept but distinct atoms).
- **Capabilities exercised.** `canonicalize` with the `gumtree`
  adapter, `fit_bpe`, `encode`, `leave_one_group_out`,
  `discriminative_procedures`.
- **Why it ranks here.** Strongest form of the attribution
  claim; no other existing code-stylometry framework can ask
  this question because tokens-of-source-text approaches are
  intrinsically language-specific. Pairs with study #14: a
  follow-on natural step once the within-language attribution
  baseline is established.
- **Shipped example.** `examples/python/13_cross_language_attribution.py`.
- **Follow-on.** A "coarsen-by-concept" mapping (e.g.
  `{Name, Identifier, SimpleName} -> Identifier`) would let
  the same atom space straddle languages without requiring the
  classifier to learn the language-specific synonym table from
  data. Treat this as a separate study once the smoke version
  is calibrated.

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
