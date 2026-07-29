# procgrep examples

A tiny synthetic corpus, a rule file, and runnable examples for the whole
pipeline. Everything runs from the repository root after
`pip install -e ".[dev]"`, e.g. `python examples/python/01_quickstart.py`.

## Layout

```
examples/
├── data/
│   ├── synthetic_traces.jsonl         (6 trajectories, 2 agents, 2 groups)
│   ├── synthetic_task_traces.jsonl    (12 trajectories with instance_id + outcome)
│   └── synthetic_gumtree_traces.jsonl (21 AST edit-script traces, 2 agents x 3 languages)
├── rules/
│   └── stuck_edit_loop.yaml           (4 pattern-matcher rules)
├── python/                            (numbered walkthroughs, table below)
├── paper/                             (reproduces the paper's empirical results)
├── procgrep_view.py                   (panel over your local Claude Code + Cursor sessions)
├── procgrep_export.py                 (privacy-preserving shareable export of those sessions)
└── claude_code_fingerprint.py         (fingerprint Claude Code transcripts, contrast styles)
```

The synthetic corpus has two agents (`editor`, `searcher`) across two groups
(`control`, `treatment`); trajectory `syn-004` deliberately violates
`no_long_edit_loops` so the pattern matcher has something to report.

Everything in `paper/` corresponds to the paper (*Agent trajectories as
programs: fingerprinting and programming coding-agent behavior*): the `pull_*`
and `extract_*` scripts build the fingerprint JSONLs, and the analysis scripts
reproduce the paper's empirical results from them. Run those from inside
`examples/paper/`.

## What each walkthrough demonstrates

| Script | Capability |
|---|---|
| `01_quickstart.py` | Full pipeline (`canonicalize` → `fit_bpe` → `encode` → `jsd_matrix`) in Python. |
| `02_controlled_eval.py` | Controlled eval: within-arm JSD as the noise floor, across-arm JSD as signal, leave-one-arm-out probe. |
| `03_discriminative_procedures.py` | Stats helpers: per-group atom frequencies, effective vocab size, entropy summary, discriminative procedures. |
| `04_deployment_signal.py` | Prefix-by-prefix pattern matching to flag one trajectory mid-stream. |
| `05_custom_adapter.py` | Registering a `TraceAdapter` for a new scaffold, then running the pipeline on it. |
| `06_task_controlled.py` | Task-controlled comparison: hold `instance_id` fixed while comparing agents. |
| `07_live_monitor.py` | The production-loop version of 04: monitor many running trajectories at once. |
| `08_metric_orthogonality.py` | Pairwise correlation of the candidate metrics; suggests an independent subset. Needs ~10+ groups. |
| `09_match_agent_to_task.py` | Pick the agent per task by procedural fit (smallest JSD to what works), vs best-overall and random. Needs `instance_id` + `outcome`. |
| `10_agent_attribution.py` | Attribution stylometry: LOGO probe vs three naive baselines. Does BPE beat raw atom frequencies? |
| `11_within_trajectory_drift.py` | Prefix/middle/suffix slice fingerprints: is the procedure stationary along a trajectory? |
| `12_gumtree_adapter.py` | The gumtree AST-edit-script adapter across Python, JavaScript, and Java. |
| `13_cross_language_attribution.py` | LOGO probe with language held out: is the fingerprint language-portable on AST atoms? |
| `14_lineage_diff_case_study_1.py` | `lineage_diff` wired into a real parent→child audit (orchestration skeleton; you supply the traces). |
| `15_multi_resolution_diff.py` | One `lineage_diff` call under multiple atom alphabets: coarse and fine answers together. |
| `16_live_fingerprint.py` | The rolling procedure mix as a session unfolds; streams a trace into the live page. |

## CLI quick run

Each subcommand reads and writes file paths, so commands compose:

```bash
procgrep canonicalize \
    --input examples/data/synthetic_traces.jsonl \
    --adapter swe-agent \
    --output /tmp/canonical.jsonl

procgrep report /tmp/canonical.jsonl

procgrep fit-bpe --input /tmp/canonical.jsonl --vocab-size 20 --output /tmp/vocab.json
procgrep encode --input /tmp/canonical.jsonl --vocab /tmp/vocab.json --output /tmp/fingerprints.jsonl
procgrep jsd --input /tmp/fingerprints.jsonl --group-by agent --output /tmp/jsd.json

procgrep match-patterns \
    --input /tmp/canonical.jsonl \
    --rules examples/rules/stuck_edit_loop.yaml \
    --output /tmp/violations.json
```
