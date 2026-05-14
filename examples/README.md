# procgrep examples

Tiny synthetic corpus, a rule file, and runnable Python and CLI
examples for the pipeline. All examples are runnable from the
repository root after `pip install -e ".[dev]"`.

## Layout

```
examples/
├── README.md                       (this file)
├── synthetic_traces.jsonl          (6 trajectories, 2 agents, 2 groups)
├── synthetic_task_traces.jsonl     (12 trajectories, 3 instances x 2 agents x 2 outcomes)
├── rules/
│   └── stuck_edit_loop.yaml        (4 pattern-matcher rules)
└── python/
    ├── 01_quickstart.py            (end-to-end Python API)
    ├── 02_controlled_eval.py       (within/across-arm JSD + probe)
    ├── 03_discriminative_motifs.py (v0.1.1 stats helpers)
    ├── 04_deployment_signal.py     (prefix-by-prefix pattern matching)
    ├── 05_custom_adapter.py        (register a TraceAdapter for a new scaffold)
    └── 06_task_controlled.py       (same-task cross-agent and success-vs-failure JSD)
```

The synthetic corpus has two agents (`editor`, `searcher`) across two
groups (`control`, `treatment`), with one trajectory (`syn-004`)
deliberately violating `no_long_edit_loops` so the pattern matcher
has something to report.

## Python examples

Run any script directly:

```bash
python examples/python/01_quickstart.py
python examples/python/02_controlled_eval.py
python examples/python/03_discriminative_motifs.py
python examples/python/04_deployment_signal.py
python examples/python/05_custom_adapter.py
python examples/python/06_task_controlled.py
```

What each script demonstrates:

| Script | Capability |
|---|---|
| `01_quickstart.py` | Full pipeline (`canonicalize` -> `fit_bpe` -> `encode` -> `jsd_matrix`) end to end in Python. |
| `02_controlled_eval.py` | The controlled-eval workflow: within-arm JSD as the noise floor, across-arm JSD as the signal, leave-one-arm-out probe. |
| `03_discriminative_motifs.py` | The v0.1.1 stats helpers: per-group atom frequencies, effective vocabulary size, per-trajectory entropy summary, top discriminative motifs between two groups. |
| `04_deployment_signal.py` | Prefix-by-prefix pattern matching to flag a trajectory mid-stream. Simulates the runtime use of the Level 1 matcher. |
| `05_custom_adapter.py` | Registering a `TraceAdapter` for a non-built-in scaffold and running the rest of the pipeline against its output. |
| `06_task_controlled.py` | Task-controlled comparison (STUDIES.md study #3): same-task cross-agent JSD and same-task success-vs-failure JSD, both via a load-time preprocessor that composes `group` from `instance_id` and `outcome`. |

## CLI quick run

The Python API has a CLI mirror. Each subcommand reads inputs from
file paths and writes outputs to file paths so commands compose.

```bash
procgrep canonicalize \
    --input examples/synthetic_traces.jsonl \
    --adapter swe-agent \
    --output /tmp/canonical.jsonl

procgrep fit-bpe \
    --input /tmp/canonical.jsonl \
    --vocab-size 20 \
    --output /tmp/vocab.json

procgrep encode \
    --input /tmp/canonical.jsonl \
    --vocab /tmp/vocab.json \
    --output /tmp/fingerprints.jsonl

procgrep jsd \
    --input /tmp/fingerprints.jsonl \
    --group-by agent \
    --output /tmp/jsd.json

procgrep match-patterns \
    --input /tmp/canonical.jsonl \
    --rules examples/rules/stuck_edit_loop.yaml \
    --output /tmp/violations.json
```
