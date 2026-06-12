# procgrep examples

Tiny synthetic corpus, a rule file, and runnable Python and CLI
examples for the pipeline. All examples are runnable from the
repository root after `pip install -e ".[dev]"`.

## Layout

```
examples/
├── README.md                          (this file)
├── synthetic_traces.jsonl             (6 trajectories, 2 agents, 2 groups)
├── synthetic_task_traces.jsonl        (12 trajectories with instance_id + outcome metadata)
├── synthetic_gumtree_traces.jsonl     (21 gumtree-flavored AST edit-script traces, 2 agents x 3 languages)
├── rules/
│   └── stuck_edit_loop.yaml           (4 pattern-matcher rules)
└── python/
    ├── 01_quickstart.py               (end-to-end Python API)
    ├── 02_controlled_eval.py          (within/across-arm JSD + probe)
    ├── 03_discriminative_procedures.py    (v0.1.1 stats helpers)
    ├── 04_deployment_signal.py        (prefix-by-prefix pattern matching)
    ├── 05_custom_adapter.py           (register a TraceAdapter for a new scaffold)
    ├── 08_metric_orthogonality.py     (candidate dimensions + pairwise correlation)
    ├── 09_match_agent_to_task.py      (pick agent per task by procedural fit vs baselines)
    ├── 10_agent_attribution.py        (LOGO attribution + 3 naive baselines)
    ├── 11_within_trajectory_drift.py  (prefix/middle/suffix slice analysis)
    ├── 12_gumtree_adapter.py          (gumtree AST edit-script adapter demo)
    └── 13_cross_language_attribution.py (LOGO probe across held-out languages)
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
python examples/python/03_discriminative_procedures.py
python examples/python/04_deployment_signal.py
python examples/python/05_custom_adapter.py
python examples/python/08_metric_orthogonality.py
python examples/python/09_match_agent_to_task.py
python examples/python/10_agent_attribution.py
python examples/python/11_within_trajectory_drift.py
python examples/python/12_gumtree_adapter.py
python examples/python/13_cross_language_attribution.py
```

What each script demonstrates:

| Script | Capability |
|---|---|
| `01_quickstart.py` | Full pipeline (`canonicalize` -> `fit_bpe` -> `encode` -> `jsd_matrix`) end to end in Python. |
| `02_controlled_eval.py` | The controlled-eval workflow: within-arm JSD as the noise floor, across-arm JSD as the signal, leave-one-arm-out probe. |
| `03_discriminative_procedures.py` | The v0.1.1 stats helpers: per-group atom frequencies, effective vocabulary size, per-trajectory entropy summary, top discriminative procedures between two groups. |
| `04_deployment_signal.py` | Prefix-by-prefix pattern matching to flag a trajectory mid-stream. Simulates the runtime use of the Level 1 matcher. |
| `05_custom_adapter.py` | Registering a `TraceAdapter` for a non-built-in scaffold and running the rest of the pipeline against its output. |
| `08_metric_orthogonality.py` | Compute the six procgrep-side candidate dimensions per group and check their pairwise correlation. Suggests an independent subset above a configurable threshold. Needs ~10+ groups to be meaningful; small fixtures yield degenerate output by design. See [METRICS.md](../METRICS.md). |
| `09_match_agent_to_task.py` | Match each task to the agent whose past procedure most resembles "what works on this kind of task" (smallest JSD between agent signature and task reference). Compares against the best-overall agent and a random baseline. Needs traces with `instance_id` + `outcome` metadata. |
| `10_agent_attribution.py` | Author-attribution stylometry baseline. Leave-one-group-out probe with agent as the prediction target, compared against three naive baselines (raw atom-frequency, length-only, majority-class). Tells you whether BPE procedures carry information beyond the marginal atom distribution. |
| `11_within_trajectory_drift.py` | Slice each trace into prefix/middle/suffix, encode each slice as its own fingerprint, then ask whether the slice position is learnable and whether attribution transfers across slice positions. Kills the stationarity assumption other analyses make implicitly. |
| `12_gumtree_adapter.py` | End-to-end demo of the gumtree adapter on AST-edit-script traces across Python, JavaScript, and Java. Shows the fine-grained node-typed atom vocabulary (`ast_insert:MethodInvocation`, ...) and the `parse_gumtree_jsondiff` helper that converts raw `gumtree jsondiff` output into adapter input. |
| `13_cross_language_attribution.py` | Strongest form of the attribution baseline: LOGO probe with language as the held-out group and agent as the prediction target. Tests whether the agent's procedural fingerprint is language-portable when built on gumtree AST atoms. |

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
