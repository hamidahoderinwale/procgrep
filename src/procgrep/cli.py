"""Command-line interface for `procgrep`.

Subcommands chain file paths: each one reads JSONL/JSON inputs, calls
a library function, and writes JSONL/JSON outputs. Nothing flows
between commands except on disk.

Subcommands: `canonicalize`, `fit-bpe`, `encode`, `jsd`, `umap`,
`probe`, `match-patterns`, `list-adapters`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from procgrep import canonicalize as canonicalize_fn
from procgrep import fit_bpe, jsd_matrix, leave_one_group_out, load_vocab, umap_project
from procgrep import save_vocab as save_vocab_fn
from procgrep.canonicalize import list_adapters
from procgrep.encode import encode as encode_fn
from procgrep.io import (
    fingerprints_to_records,
    read_jsonl,
    records_to_fingerprints,
    records_to_traces,
    traces_to_records,
    write_json,
    write_jsonl,
)
from procgrep.patterns import load_patterns as load_patterns_fn
from procgrep.patterns import match_patterns as match_patterns_fn

app = typer.Typer(
    name="procgrep",
    help="Procedural fingerprinting of LLM coding-agent trajectories.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def canonicalize(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Raw trace JSONL.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Canonical trace JSONL.")],
    adapter: Annotated[
        str,
        typer.Option("--adapter", "-a", help="Adapter name; see `procgrep list-adapters`."),
    ],
    trace_id_field: Annotated[
        str, typer.Option(help="Field in each raw record holding the trace id.")
    ] = "trace_id",
    agent_field: Annotated[str, typer.Option(help="Field holding the agent name.")] = "agent",
    group_field: Annotated[
        str,
        typer.Option(help="Field holding the group label; empty string disables."),
    ] = "group",
) -> None:
    """Canonicalize raw scaffold traces into atom sequences."""
    traces = canonicalize_fn(
        read_jsonl(input_path),
        adapter=adapter,
        trace_id_field=trace_id_field,
        agent_field=agent_field,
        group_field=group_field if group_field else None,
    )
    n = write_jsonl(output_path, traces_to_records(traces))
    typer.echo(f"wrote {n} canonical traces to {output_path}")


@app.command(name="fit-bpe")
def fit_bpe_cmd(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Canonical trace JSONL.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Vocabulary JSON.")],
    vocab_size: Annotated[
        int, typer.Option("--vocab-size", "-V", help="Target vocabulary size.")
    ] = 200,
    seed: Annotated[int, typer.Option(help="Provenance seed.")] = 0,
    min_pair_frequency: Annotated[int, typer.Option(help="Minimum pair count to merge.")] = 2,
) -> None:
    """Learn a BPE procedure vocabulary from canonical traces."""
    traces = list(records_to_traces(read_jsonl(input_path)))
    vocab = fit_bpe(
        (t.atoms for t in traces),
        vocab_size=vocab_size,
        seed=seed,
        min_pair_frequency=min_pair_frequency,
    )
    save_vocab_fn(vocab, output_path)
    typer.echo(
        f"learned vocabulary: {len(vocab.atoms)} atoms + {len(vocab.merges)} merges "
        f"= {vocab.size} tokens; wrote {output_path}"
    )


@app.command()
def encode(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Canonical trace JSONL.")],
    vocab_path: Annotated[Path, typer.Option("--vocab", "-v", help="Vocabulary JSON.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Fingerprint JSONL.")],
) -> None:
    """Encode canonical traces as procedure-frequency fingerprints."""
    vocab = load_vocab(vocab_path)
    traces = records_to_traces(read_jsonl(input_path))
    fingerprints = encode_fn(traces, vocab=vocab)
    n = write_jsonl(output_path, fingerprints_to_records(fingerprints))
    typer.echo(f"wrote {n} fingerprints to {output_path}")


@app.command()
def jsd(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Fingerprint JSONL.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="JSD matrix JSON.")],
    group_by: Annotated[str, typer.Option(help="'group' or 'agent'.")] = "group",
    base: Annotated[float, typer.Option(help="Logarithm base (2 keeps JSD in [0, 1]).")] = 2.0,
) -> None:
    """Compute pairwise JSD between group-mean fingerprints."""
    fingerprints = list(records_to_fingerprints(read_jsonl(input_path)))
    matrix = jsd_matrix(fingerprints, group_by=group_by, base=base)  # type: ignore[arg-type]
    payload = {
        "groups": list(matrix.groups),
        "records": matrix.to_records(),
        "base": matrix.base,
    }
    write_json(output_path, payload)
    typer.echo(f"wrote JSD matrix ({len(matrix.groups)} groups) to {output_path}")


@app.command()
def umap(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Fingerprint JSONL.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="UMAP coords JSON.")],
    granularity: Annotated[str, typer.Option(help="'trace' or 'group'.")] = "trace",
    n_neighbors: Annotated[int, typer.Option(help="UMAP n_neighbors.")] = 15,
    min_dist: Annotated[float, typer.Option(help="UMAP min_dist.")] = 0.25,
    metric: Annotated[str, typer.Option(help="Distance metric.")] = "cosine",
    seed: Annotated[int, typer.Option(help="UMAP seed.")] = 0,
) -> None:
    """Project fingerprints to 2D with UMAP."""
    fingerprints = list(records_to_fingerprints(read_jsonl(input_path)))
    result = umap_project(
        fingerprints,
        granularity=granularity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        seed=seed,
    )
    payload = {
        "labels": list(result.labels),
        "coords": result.coords.tolist(),
        "n_neighbors": result.n_neighbors,
        "min_dist": result.min_dist,
        "metric": result.metric,
        "seed": result.seed,
    }
    write_json(output_path, payload)
    typer.echo(f"wrote UMAP coords ({len(result.labels)} points) to {output_path}")


@app.command()
def probe(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Fingerprint JSONL.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Probe result JSON.")],
    label_field: Annotated[str, typer.Option(help="'group' or 'agent'.")] = "group",
    seed: Annotated[int, typer.Option(help="Classifier seed.")] = 0,
) -> None:
    """Run a leave-one-group-out predictive-transfer probe."""
    fingerprints = list(records_to_fingerprints(read_jsonl(input_path)))
    result = leave_one_group_out(fingerprints, label_field=label_field, seed=seed)
    write_json(output_path, asdict(result))
    typer.echo(
        f"probe: overall accuracy {result.overall_accuracy:.3f} "
        f"across {len(result.groups)} groups; wrote {output_path}"
    )


@app.command(name="match-patterns")
def match_patterns_cmd(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Canonical trace JSONL.")],
    rules_path: Annotated[Path, typer.Option("--rules", "-r", help="YAML rules file.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Report JSON.")],
) -> None:
    """Match procedural patterns against canonical atom sequences."""
    traces = list(records_to_traces(read_jsonl(input_path)))
    patterns = load_patterns_fn(rules_path)
    report = match_patterns_fn(traces, patterns)
    payload = {
        "patterns": [
            {"name": p.name, "description": p.description, "must_hold": p.must_hold}
            for p in report.patterns
        ],
        "violations": report.violations,
        "pass_rate_per_rule": report.pass_rate_per_rule,
    }
    write_json(output_path, payload)
    n_violators = len(report.violations)
    typer.echo(
        f"evaluated {len(report.patterns)} rules on {len(traces)} traces; "
        f"{n_violators} traces violated at least one rule; wrote {output_path}"
    )


@app.command(name="list-adapters")
def list_adapters_cmd() -> None:
    """List registered trace adapters."""
    names = list_adapters()
    if not names:
        typer.echo("no adapters registered")
        return
    for name in names:
        typer.echo(name)


if __name__ == "__main__":
    app()
