"""procgrep-runner CLI: run paired arms, run one instance, measure a run.

Intent: the three entry points of the loop. ``run`` prepares and executes a
paired run; ``instance`` is the per-task child the batch shells out to for
crash isolation (its last stdout line is the row as JSON); ``measure`` closes
the loop back to `procgrep.verify`. Read this when scripting a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from procgrep.io import read_jsonl, records_to_traces
from procgrep.reward import ProcedureSpec
from procgrep_runner import measure as measure_mod
from procgrep_runner import run as run_mod
from procgrep_runner.manifest import read_manifest
from procgrep_runner.tasks import (
    INSTANCES_NAME,
    load_swebench_instances,
    read_instances_jsonl,
)

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


@app.command()
def run(
    spec: Annotated[Path, typer.Option("--spec", help="ProcedureSpec YAML.")],
    run_dir: Annotated[
        Path, typer.Option("-o", "--run-dir", help="Fresh run directory (never overwritten).")
    ],
    mode: Annotated[str, typer.Option(help="Enforcement mode: prompt or guard.")] = "guard",
    on_violation: Annotated[str, typer.Option(help="guard mode: block, steer, or warn.")] = "block",
    subset: Annotated[str, typer.Option(help="SWE-bench subset.")] = "verified",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "test",
    instance_id: Annotated[
        list[str] | None, typer.Option("-i", "--instance", help="Instance id (repeatable).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="First N instances when no ids given.")] = None,
    instances_file: Annotated[
        Path | None,
        typer.Option(help="Frozen instances.jsonl (skips the HF datasets load)."),
    ] = None,
    model: Annotated[str | None, typer.Option("-m", "--model", help="Model name override.")] = None,
    model_class: Annotated[str | None, typer.Option(help="Model class override.")] = None,
    environment_class: Annotated[str | None, typer.Option(help="docker, swerex_modal, ...")] = None,
    config_spec: Annotated[
        list[str] | None,
        typer.Option("-c", "--config", help="Extra config specs, merged in order."),
    ] = None,
    cost_limit: Annotated[float | None, typer.Option(help="Per-run model cost cap (USD).")] = None,
    replicates: Annotated[int, typer.Option(help="Replicates per (instance, arm).")] = 1,
    seed: Annotated[int, typer.Option(help="Recorded run seed.")] = 0,
    instance_timeout: Annotated[
        float, typer.Option(help="Seconds per instance subprocess.")
    ] = 7200,
    isolate: Annotated[
        bool, typer.Option("--isolate/--no-isolate", help="Subprocess per instance.")
    ] = True,
) -> None:
    """Execute baseline and enforced arms over the same tasks."""
    if instances_file is not None:
        instances = read_instances_jsonl(instances_file)
    else:
        instances = load_swebench_instances(
            subset, split, instance_ids=instance_id or None, limit=limit
        )
    config = run_mod.build_config(
        config_specs=list(config_spec or []) or None,
        model_name=model,
        model_class=model_class,
        environment_class=environment_class,
        cost_limit=cost_limit,
    )
    rows = run_mod.run_paired(
        spec=ProcedureSpec.from_yaml(spec),
        instances=instances,
        config=config,
        run_dir=run_dir,
        mode=mode,
        on_violation=on_violation,  # type: ignore[arg-type]
        replicates=replicates,
        seed=seed,
        subset=subset,
        split=split,
        isolate=isolate,
        instance_timeout=instance_timeout,
    )
    done = sum(1 for r in rows if r.get("exit_status") == "Submitted")
    typer.echo(f"{len(rows)} runs, {done} submitted -> {run_dir}")


@app.command()
def instance(
    run_dir: Annotated[Path, typer.Option(help="Prepared run directory.")],
    instance_id: Annotated[str, typer.Option()],
    arm: Annotated[str, typer.Option(help="baseline or enforced.")],
    replicate: Annotated[int, typer.Option()] = 0,
) -> None:
    """Run one (instance, arm, replicate) from a prepared run directory.

    Child entry point for subprocess isolation; prints the row as the last
    stdout line.
    """
    manifest = read_manifest(run_dir)
    instances = {i["instance_id"]: i for i in read_instances_jsonl(run_dir / INSTANCES_NAME)}
    config = json.loads((run_dir / run_mod.CONFIG_NAME).read_text())
    row = run_mod.run_instance(
        instances[instance_id],
        spec=ProcedureSpec.from_yaml(run_dir / "spec.yaml"),
        arm=arm,
        mode=manifest.mode,
        config=config,
        run_dir=run_dir,
        on_violation=manifest.on_violation or "block",  # type: ignore[arg-type]
        replicate=replicate,
    )
    typer.echo(json.dumps(row))


@app.command()
def measure(
    run_dir: Annotated[Path, typer.Option(help="Completed run directory.")],
    winners: Annotated[
        Path | None,
        typer.Option(help="Canonical traces JSONL with the outcome field, to derive the target."),
    ] = None,
    grades: Annotated[
        Path | None,
        typer.Option(help='JSON {"baseline": {id: bool}, "enforced": {id: bool}}.'),
    ] = None,
    vocab_size: Annotated[int, typer.Option()] = 64,
    n_boot: Annotated[int, typer.Option()] = 1000,
    jsd_eps: Annotated[float, typer.Option(help="verify's behavior-moved threshold.")] = 1e-3,
) -> None:
    """Feed a run's traces back to `procgrep.verify`; write measure.summary.json."""
    summary = measure_mod.measure_run(
        run_dir,
        winners=list(records_to_traces(read_jsonl(winners))) if winners else None,
        grades=json.loads(grades.read_text()) if grades else None,
        vocab_size=vocab_size,
        n_boot=n_boot,
        jsd_improvement_eps=jsd_eps,
    )
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
