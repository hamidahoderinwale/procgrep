"""Command-line interface for `procgrep`.

Subcommands chain file paths: each one reads JSONL/JSON inputs, calls
a library function, and writes JSONL/JSON outputs. Nothing flows
between commands except on disk.

Subcommands: `canonicalize`, `fit-bpe`, `encode`, `jsd`, `umap`,
`probe`, `match-patterns`, `list-adapters`.

`clients` and `cursor` are the exception to the chaining rule: the sessions they
read are already on disk in a client-specific place, so they locate that place
themselves and produce the first JSONL the other subcommands consume.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from procgrep import canonicalize as canonicalize_fn
from procgrep import (
    fit_bpe,
    jsd_matrix,
    leave_one_group_out,
    load_vocab,
    render_vocab_tree,
    umap_project,
)
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
        fit_corpus=input_path.name,
    )
    save_vocab_fn(vocab, output_path)
    typer.echo(
        f"learned vocabulary: {len(vocab.atoms)} atoms + {len(vocab.merges)} merges "
        f"= {vocab.size} tokens (vocab {vocab.spec.compact()}); wrote {output_path}"
    )


@app.command(name="vocab-tree")
def vocab_tree_cmd(
    vocab_path: Annotated[
        Path | None, typer.Option("--vocab", "-v", help="Existing vocabulary JSON.")
    ] = None,
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Canonical trace JSONL to fit a vocab from instead."),
    ] = None,
    vocab_size: Annotated[
        int, typer.Option("--vocab-size", "-V", help="Target size when fitting from --input.")
    ] = 64,
    seed: Annotated[int, typer.Option(help="Provenance seed when fitting.")] = 0,
) -> None:
    """Print the BPE procedure vocabulary as merge trees (the procedure hierarchy)."""
    if vocab_path is not None:
        vocab = load_vocab(vocab_path)
    elif input_path is not None:
        traces = list(records_to_traces(read_jsonl(input_path)))
        vocab = fit_bpe((t.atoms for t in traces), vocab_size=vocab_size, seed=seed)
    else:
        typer.echo("provide --vocab or --input", err=True)
        raise typer.Exit(1)
    typer.echo(render_vocab_tree(vocab))


@app.command()
def encode(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Canonical trace JSONL.")],
    vocab_path: Annotated[Path, typer.Option("--vocab", "-v", help="Vocabulary JSON.")],
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Fingerprint JSONL.")],
    unknown: Annotated[
        str, typer.Option(help="Actions absent from the vocabulary: error or ignore.")
    ] = "error",
) -> None:
    """Encode canonical traces as procedure-frequency fingerprints."""
    if unknown not in {"error", "ignore"}:
        raise typer.BadParameter("must be 'error' or 'ignore'", param_hint="--unknown")
    vocab = load_vocab(vocab_path)
    traces = records_to_traces(read_jsonl(input_path))
    try:
        fingerprints = encode_fn(
            traces, vocab=vocab, unknown="ignore" if unknown == "ignore" else "error"
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
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
        "vocab_spec": matrix.vocab_spec,
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
    try:
        result = leave_one_group_out(fingerprints, label_field=label_field, seed=seed)
    except ValueError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from exc
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


@app.command()
def compare(
    agent_a: Annotated[Path, typer.Argument(help="Comparison JSONL for agent A.")],
    agent_b: Annotated[Path, typer.Argument(help="Comparison JSONL for agent B.")],
    name_a: Annotated[str, typer.Option("--name-a", help="Display name for agent A.")] = "",
    name_b: Annotated[str, typer.Option("--name-b", help="Display name for agent B.")] = "",
    top_n: Annotated[int, typer.Option(help="Discriminative bigrams to show per side.")] = 5,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write JSON report here (optional).")
    ] = None,
    base: Annotated[float, typer.Option(help="JSD logarithm base; must be greater than 1.")] = 2.0,
) -> None:
    """Compare pooled actions, transitions, positions, and pass rates for two agents."""
    from procgrep.comparison import build_comparison

    try:
        report = build_comparison(
            list(read_jsonl(agent_a)),
            list(read_jsonl(agent_b)),
            name_a=name_a or agent_a.stem,
            name_b=name_b or agent_b.stem,
            top_n=top_n,
            base=base,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(report.summary())
    if output:
        write_json(output, report.to_dict())
        typer.echo(f"report written to {output}")


@app.command()
def curate(
    dataset: Annotated[str, typer.Argument(help="HF dataset id, or a local canonical JSONL path.")],
    limit: Annotated[int, typer.Option(help="Max rows to stream from the Hub.")] = 3000,
    target: Annotated[
        int | None, typer.Option(help="Diverse-subset size (default: dedup cluster count).")
    ] = None,
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-V")] = 128,
    near_dup_jsd: Annotated[float, typer.Option(help="JSD threshold for near-duplicates.")] = 0.05,
    export: Annotated[
        Path | None, typer.Option("--export", help="Write the diverse subset (canonical JSONL).")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the inferred ingestion plan and exit.")
    ] = False,
) -> None:
    """Measure structural redundancy and select a procedurally-diverse subset.

    DATASET may be a Hub id (dynamically sniffed + streamed) or a local
    canonical-trace JSONL produced by `procgrep canonicalize`.
    """
    from procgrep.curate import curate as curate_fn
    from procgrep.ingest import ingest as ingest_fn
    from procgrep.ingest import plan as plan_fn

    if Path(dataset).exists():
        traces = list(records_to_traces(read_jsonl(Path(dataset))))
    else:
        if dry_run:
            typer.echo(plan_fn(dataset).summary())
            return
        traces, plan = ingest_fn(dataset, limit=limit)
        typer.echo(plan.summary())
        typer.echo("")

    report = curate_fn(traces, vocab_size=vocab_size, near_dup_jsd=near_dup_jsd, target_size=target)
    typer.echo(report.summary())
    if export is not None:
        subset = [traces[i] for i in report.subset_indices]
        n = write_jsonl(export, traces_to_records(subset))
        typer.echo(f"\nwrote {n} diverse traces to {export}")


@app.command()
def grep(
    pattern: Annotated[str, typer.Argument(help="Regex over the space-joined atom sequence.")],
    dataset: Annotated[str, typer.Argument(help="HF dataset id, or a local canonical JSONL path.")],
    limit: Annotated[int, typer.Option(help="Max rows to stream from the Hub.")] = 3000,
    show: Annotated[int, typer.Option(help="Max matching traces to print.")] = 20,
) -> None:
    """Find trajectories whose action sequence matches a structural pattern.

    Examples (the pattern is a regex over `" ".join(atoms)`):
      "(edit ){5,}"                 edit streak of 5+
      "run_test( \\w+)* submit"      ran a test before submitting
      "^(?:(?!run_test).)*submit "   submitted, never ran a test
    """
    import re

    from procgrep.ingest import ingest as ingest_fn

    if Path(dataset).exists():
        traces = list(records_to_traces(read_jsonl(Path(dataset))))
    else:
        traces, _ = ingest_fn(dataset, limit=limit)
    rx = re.compile(pattern)
    hits = [t for t in traces if rx.search(" ".join(t.atoms) + " ")]
    for t in hits[:show]:
        preview = " ".join(t.atoms[:28]) + (" …" if len(t.atoms) > 28 else "")
        typer.echo(f"{t.trace_id}\t{t.agent}\t{preview}")
    rate = len(hits) / max(len(traces), 1)
    typer.echo(f"\n{len(hits)}/{len(traces)} traces matched ({rate:.1%})")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="An English question about agent behavior.")],
    dataset: Annotated[
        str | None,
        typer.Argument(help="Optional HF dataset id or canonical JSONL path to run the query on."),
    ] = None,
    model: Annotated[str, typer.Option(help="Model that compiles the question.")] = "claude-opus-5",
    limit: Annotated[int, typer.Option(help="Max rows to stream from the Hub.")] = 3000,
    show: Annotated[int, typer.Option(help="Max matching traces to print.")] = 20,
) -> None:
    """Compile an English question to a spine regex, then optionally run it.

    The model writes the query; matching stays deterministic `grep`. The
    compiled regex and its literal paraphrase are always printed so the
    translation can be checked before trusting counts. Needs
    ANTHROPIC_API_KEY. Questions the regex layer cannot express (temporal
    windows, variable binding, thresholds) are refused with the nearest
    expressible alternative.
    """
    import re

    from procgrep.ask import AskError, compile_query

    try:
        compiled = compile_query(question, model=model)
    except AskError as exc:
        typer.echo(f"ask failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not compiled.expressible:
        typer.echo("not expressible as a spine regex.")
        typer.echo(f"why: {compiled.reason}")
        raise typer.Exit(2)

    typer.echo(f"regex      {compiled.regex}")
    typer.echo(f"matches    {compiled.paraphrase}")
    if dataset is None:
        return

    from procgrep.ingest import ingest as ingest_fn

    if Path(dataset).exists():
        traces = list(records_to_traces(read_jsonl(Path(dataset))))
    else:
        traces, _ = ingest_fn(dataset, limit=limit)
    rx = re.compile(compiled.regex or "")
    hits = [t for t in traces if rx.search(" ".join(t.atoms) + " ")]
    typer.echo("")
    for t in hits[:show]:
        preview = " ".join(t.atoms[:28]) + (" …" if len(t.atoms) > 28 else "")
        typer.echo(f"{t.trace_id}\t{t.agent}\t{preview}")
    rate = len(hits) / max(len(traces), 1)
    typer.echo(f"\n{len(hits)}/{len(traces)} traces matched ({rate:.1%})")


@app.command()
def report(
    dataset: Annotated[str, typer.Argument(help="HF dataset id, or a local canonical JSONL path.")],
    limit: Annotated[int, typer.Option(help="Max rows to stream from the Hub.")] = 3000,
    config: Annotated[
        str | None, typer.Option(help="HF config name for multi-config datasets.")
    ] = None,
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-V")] = 64,
    top: Annotated[int, typer.Option(help="Top procedures to list.")] = 8,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the full report as JSON.")
    ] = None,
) -> None:
    """One-shot corpus overview: ingest, learn procedures, and print what is observed.

    DATASET may be a Hub id (dynamically sniffed + streamed) or a local
    canonical-trace JSONL produced by `procgrep canonicalize`.
    """
    import json as json_lib

    from procgrep.ingest import ingest as ingest_fn
    from procgrep.report import build_report

    if Path(dataset).exists():
        traces = list(records_to_traces(read_jsonl(Path(dataset))))
    else:
        traces, plan = ingest_fn(dataset, limit=limit, config=config)
        typer.echo(plan.summary())
        typer.echo("")

    rep = build_report(traces, source=dataset, vocab_size=vocab_size, top=top)
    typer.echo(rep.summary())
    if json_out is not None:
        json_out.write_text(json_lib.dumps(rep.to_dict(), indent=2) + "\n")
        typer.echo(f"\nwrote {json_out}")


@app.command()
def floor(
    dataset: Annotated[str, typer.Argument(help="HF dataset id, or a local canonical JSONL path.")],
    condition: Annotated[
        str,
        typer.Option(
            "--condition",
            "-c",
            help="Field naming identically-configured runs: 'agent', 'group', or any "
            "metadata key / dataset column.",
        ),
    ],
    adapter: Annotated[
        str | None,
        typer.Option("--adapter", "-a", help="Adapter override for Hub ingestion."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Max rows to stream from the Hub.")] = 3000,
    config: Annotated[
        str | None, typer.Option(help="HF config name for multi-config datasets.")
    ] = None,
    vocab_size: Annotated[int, typer.Option("--vocab-size", "-V")] = 64,
    seed: Annotated[
        int, typer.Option(help="Seeds the vocabulary fit and the Monte Carlo draws.")
    ] = 0,
    n_draws: Annotated[int, typer.Option(help="Monte Carlo draws per group size.")] = 200,
    delta: Annotated[
        list[float] | None,
        typer.Option(
            "--delta",
            help="Target detectable difference; repeat the flag for a grid "
            "(default: 0.05 0.1 0.2).",
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write the JSON report here, not stdout.")
    ] = None,
) -> None:
    """Measure your eval harness's noise floor and the runs a target effect needs.

    Groups trajectories by --condition (runs sharing a value count as
    identically configured) and reports, per condition cell, how far apart
    two disjoint groups of n same-condition runs land anyway -- across a
    grid of n -- plus seeds_needed: the smallest n whose floor sits below
    each target difference. Read any claimed effect against this floor,
    not against zero. JSON only; ingestion notes go to stderr so stdout
    stays parseable.
    """
    import json as json_lib

    from procgrep.floor import DEFAULT_DELTAS, measure_floor
    from procgrep.ingest import ingest as ingest_fn

    if Path(dataset).exists():
        traces = list(records_to_traces(read_jsonl(Path(dataset))))
    else:
        traces, plan = ingest_fn(dataset, limit=limit, config=config, adapter=adapter)
        typer.echo(plan.summary(), err=True)

    try:
        rep = measure_floor(
            traces,
            condition=condition,
            source=dataset,
            vocab_size=vocab_size,
            seed=seed,
            n_draws=n_draws,
            deltas=tuple(delta) if delta else DEFAULT_DELTAS,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = json_lib.dumps(rep.to_dict(), indent=2)
    if out is not None:
        out.write_text(payload + "\n")
        typer.echo(f"wrote {out}", err=True)
    else:
        typer.echo(payload)


@app.command()
def watch(
    tail: Annotated[
        str | None, typer.Option(help="Follow a file; each appended line is one action atom.")
    ] = None,
    demo: Annotated[
        bool, typer.Option("--demo", help="Replay a built-in rollout (no agent needed).")
    ] = False,
    port: Annotated[int, typer.Option(help="Local port for the live page.")] = 7870,
    interval: Annotated[float, typer.Option(help="Seconds between --demo steps.")] = 0.6,
) -> None:
    """Stream a rollout into a live local web view as it happens."""
    from procgrep.watch import serve

    raise SystemExit(serve(tail=tail, demo=demo, port=port, interval=interval))


@app.command()
def clients() -> None:
    """Show the coding clients on this machine whose sessions procgrep can read."""
    from procgrep.ingest.clients import discover_clients

    found = discover_clients()
    if not found:
        typer.echo("no local sessions found (looked for Cursor and Claude Code)")
        raise SystemExit(1)
    for client in found:
        typer.echo(f"{client.name}\t{client.adapter}\t{client.size_label}\t{client.path}")


@app.command()
def cursor(
    output_path: Annotated[
        Path, typer.Option("--output", "-o", help="Canonical trace JSONL to write.")
    ] = Path("cursor-traces.jsonl"),
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Cursor state.vscdb; found automatically when omitted."),
    ] = None,
) -> None:
    """Canonicalize your own Cursor sessions, locating Cursor's store for you.

    Reads the SQLite store Cursor already keeps, so nothing needs to be running
    and no exporter is involved. Only structure crosses the boundary: turn type
    and tool name, never prompt text, tool arguments, or code.
    """
    from procgrep.ingest.adapters.cursor_vscdb import read_state_vscdb, session_rework
    from procgrep.ingest.clients import find_client

    client = find_client("cursor", path=db)
    if client is None:
        where = f" at {db}" if db else ""
        typer.echo(f"no Cursor session store found{where}; pass --db to point at one")
        raise SystemExit(1)
    typer.echo(f"reading {client.name} ({client.size_label}) from {client.path}")

    records = list(read_state_vscdb(client.path))
    traces = canonicalize_fn(records, adapter="cursor-vscdb")
    n = write_jsonl(output_path, traces_to_records(traces))
    atoms = sum(len(t.atoms) for t in traces)
    typer.echo(f"wrote {n} traces ({atoms} atoms) to {output_path}")

    # Rework is the session-level signal this store makes available and a
    # dataset of autonomous rollouts cannot: re-editing a file the same session
    # already edited, i.e. correction rather than forward progress.
    stats = [session_rework(r["events"]) for r in records]
    prompts = sum(int(s["prompts"]) for s in stats)
    rework = sum(int(s["rework_prompts"]) for s in stats)
    if prompts:
        typer.echo(f"rework: {rework}/{prompts} prompts re-edited a file ({rework / prompts:.1%})")


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
