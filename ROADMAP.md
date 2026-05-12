# Roadmap

procgrep follows a cite-anchored MVP discipline: every shipped
capability is justified by a paper claim it enables or a use case
it serves. This roadmap names the next milestones in that frame.

## Current state

**v0.1.3** ships the MVP plus convenience helpers and project
boilerplate:

- Canonicalize, fit BPE, encode, JSD, UMAP, leave-one-group-out probe.
- Level 1 pattern matcher (regex over canonical atoms).
- Group-level descriptive and discriminative stats (atom
  frequencies, effective vocabulary size, per-trajectory entropy,
  top discriminative motifs).
- Five Python example scripts under `examples/python/` and a
  bundled synthetic corpus.
- CI on Python 3.10, 3.11, 3.12, 3.13.
- Built-in adapters: SWE-agent, Agentless, DARS, Moatless.

## Planned

### v0.1.4 - v0.1.x: Level 1.2 helpers and quality of life

- `top_discriminative_motifs_with_ci(...)`: bootstrap confidence
  intervals on JSD and discriminative-motif rankings.
- `effect_sizes(...)`: Cohen's h and log-odds-ratio helpers on
  motif frequencies.
- `probe_feature_importances(...)`: classifier feature importances
  from the leave-one-group-out probe.
- Migrate dependency management to uv where appropriate.

These additions are strictly additive and unlock paper-quality
reporting (confidence intervals, effect sizes) for controlled-eval
studies.

### v0.2: Adapter modularization

When the number of built-in adapters crosses about five, refactor
following the DSPy adapters pattern:

```
src/procgrep/adapters/
├── __init__.py     (registry init + re-exports)
├── base.py         (TraceAdapter protocol + make_action_adapter)
├── swe_agent.py
├── agentless.py
├── dars.py
├── moatless.py
└── <new>.py
```

Backwards-compatible because `procgrep.canonicalize` continues to
re-export the same callables. Triggered by a real need (a new
adapter that benefits from per-file isolation), not on a calendar.

### v0.3: Two-level examples hierarchy

When example-script count crosses about ten, restructure following
the sentence-transformers pattern:

```
examples/python/
├── quickstart/
├── controlled_eval/
├── deployment_signal/
└── adapters/
```

Triggered by example count, not calendar.

### v1.0: Documentation site and Level 2 invariants

Two milestones together because each is large.

- **Documentation site**: mkdocs-material at `docs/`, with API
  reference, tutorials, deep-dive on the BPE choice, and a
  cookbook of recipes. Mirrors the DSPy and SWE-agent docs layout.
- **Procedural-DSPy (Level 2 invariant DSL)**: compositional
  invariants with temporal operators and probabilistic predicates.
  This is the named future direction in the paper's
  `Implications` section. Will land first as
  `procgrep.experimental.dsl` before promotion.

## Out of scope (will not ship)

- Live agent execution. procgrep is post-hoc analysis only.
- Direct LLM SDK integration. The library reads trace files; it
  does not call models.
- Embedding-based fingerprints. The paper commits to count-based
  representations for interpretability. Embedding-based comparison
  is a different research direction.
- Web dashboard. Figures emit as PNG/SVG; live dashboards belong
  in downstream tooling, not the library.

## How to influence the roadmap

Open an issue describing the use case, the data shape, and the
question procgrep would help answer. If the answer requires a
capability not on the roadmap, that is the strongest argument for
adding it.
