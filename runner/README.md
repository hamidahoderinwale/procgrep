# procgrep-runner: paired-arm agent execution for procgrep

procgrep-runner executes agents under a `ProcedureSpec` in sandboxes and feeds
the resulting traces back to `procgrep.verify`. It is the runner the procgrep
roadmap promised, kept outside the core on purpose: procgrep emits enforcement
artifacts and measures traces; this package hosts the scaffold
([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)) that runs the
model.

## Installation

```bash
cd runner && uv sync          # dev install; procgrep resolved from the parent repo
uv sync --extra swebench      # + HF datasets, for loading SWE-bench task lists
```

## Quickstart

```bash
# Paired run: a baseline arm and an enforced arm over the same tasks.
procgrep-runner run --spec spec.yaml -o runs/pilot_v1 \
  --mode guard --on-violation block \
  --subset verified --limit 5 -m claude-sonnet-5 --environment-class docker

# Close the loop: traces -> atoms -> verify, with paired-bootstrap CIs.
procgrep-runner measure --run-dir runs/pilot_v1 --winners winners.jsonl
```

Or from Python:

```python
from procgrep.reward import ProcedureSpec
from procgrep_runner import build_config, measure_run, run_paired

spec = ProcedureSpec.from_yaml("spec.yaml")
config = build_config(model_name="claude-sonnet-5", environment_class="docker")
run_paired(spec=spec, instances=instances, config=config, run_dir="runs/pilot_v1", mode="guard")
summary = measure_run("runs/pilot_v1", winners=winner_traces)
print(summary["verify"]["verdict"], summary["bootstrap"])
```

## What a run produces

```
runs/pilot_v1/
  manifest.json          # sealed identity: spec sha256, model, scaffold version, tasks, seeds
  spec.yaml              # the enforced spec, content-addressed by the manifest
  instances.jsonl        # frozen task list; later steps never touch the network
  config.json            # the merged scaffold config
  arms/{baseline,enforced}/*.traj.json
  rows.jsonl             # one row per (instance, arm, replicate)
  measure.summary.json   # verify verdict + 95% bootstrap CIs, after `measure`
```

**Reproducibility contract.** Model sampling is not reproducible, even at
temperature 0. What is bit-reproducible is the measurement path: given the
trace files, ingest, the seeded vocabulary, `verify`, and the seeded bootstrap
always produce the same result, and the manifest pins everything that path
needs. Each instance runs in its own subprocess, so one stalled sandbox never
takes down a batch.

**Enforcement modes.** `guard` checks every action against the spec at
execute time (block substitutes a notice command, steer injects the rule text
after the violating action); `prompt` appends the spec's rendered ruleset to
the system prompt. `reward` and `decode` artifacts are training-loop and
local-inference concerns; this runner does not consume them.

**Note.** Trace atoms record the model's *attempted* actions; what a guard
blocked is in `guard_events` (trajectory) and the `guard_blocked` counts
(rows). Read the behavior axis with that in mind for block mode.

## Status

Offline-tested (scripted model, no API calls or docker needed). The
direction-control runs (positive/negative/placebo), the eps calibration, and
the SWE-bench pilot are designed but not yet run; see
`plateau/RUNNER_DESIGN.md` (local) for the sequencing and sample plan.
