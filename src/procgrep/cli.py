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
from typing import Annotated, Any

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
    )
    save_vocab_fn(vocab, output_path)
    typer.echo(
        f"learned vocabulary: {len(vocab.atoms)} atoms + {len(vocab.merges)} merges "
        f"= {vocab.size} tokens; wrote {output_path}"
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
    agent_a: Annotated[Path, typer.Argument(help="Fingerprint JSONL for agent A.")],
    agent_b: Annotated[Path, typer.Argument(help="Fingerprint JSONL for agent B.")],
    name_a: Annotated[str, typer.Option("--name-a", help="Display name for agent A.")] = "",
    name_b: Annotated[str, typer.Option("--name-b", help="Display name for agent B.")] = "",
    top_n: Annotated[int, typer.Option(help="Discriminative bigrams to show per side.")] = 5,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write JSON report here (optional).")
    ] = None,
) -> None:
    """Compare two agents: JSD, discriminative bigrams, positional divergence, pass rates."""
    import json as _json
    from collections import Counter as _Counter

    import numpy as _np

    label_a = name_a or agent_a.stem
    label_b = name_b or agent_b.stem

    def _load(path: Path) -> list[dict[str, Any]]:
        return [_json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    rows_a = _load(agent_a)
    rows_b = _load(agent_b)

    # Guard empty input: the positional-divergence section indexes
    # ``rows[0]`` and the distribution helpers assume at least one row,
    # so an empty JSONL would crash with an opaque IndexError.
    if not rows_a or not rows_b:
        empty = label_a if not rows_a else label_b
        typer.echo(f"error: no trajectories in {empty}; both inputs must be non-empty")
        raise typer.Exit(1)

    typer.echo(f"\n{'=' * 64}")
    typer.echo(f"  {label_a}  ({len(rows_a)} trajectories)")
    typer.echo(f"  {label_b}  ({len(rows_b)} trajectories)")
    typer.echo(f"{'=' * 64}")

    canon_atoms = [
        "edit",
        "read_file",
        "run_test",
        "search_repo",
        "create_file",
        "delete_file",
        "think",
        "error",
        "other",
    ]
    eps = 1e-9

    def _dist(
        rows: list[dict[str, Any]], key: str = "atoms_canonical"
    ) -> tuple[_np.ndarray[Any, _np.dtype[Any]], list[str]]:
        cnt: _Counter[str] = _Counter()
        for r in rows:
            for a in r.get(key, []):
                cnt[str(a)] += 1
        vocab = canon_atoms if key == "atoms_canonical" else sorted(cnt)
        v = _np.array([cnt.get(a, 0) + eps for a in vocab], dtype=float)
        return v / v.sum(), vocab

    def _jsd_val(p: _np.ndarray[Any, _np.dtype[Any]], q: _np.ndarray[Any, _np.dtype[Any]]) -> float:
        m = (p + q) / 2

        def kl(a: _np.ndarray[Any, _np.dtype[Any]], b: _np.ndarray[Any, _np.dtype[Any]]) -> float:
            return float(_np.sum(a * _np.log(a / b)))

        return (kl(p, m) + kl(q, m)) / 2

    # 1. Unigram JSD.
    pa, _canon_v = _dist(rows_a, "atoms_canonical")
    pb, _ = _dist(rows_b, "atoms_canonical")
    canon_jsd = _jsd_val(pa, pb)

    # Native vocab must be shared (each agent may use different tool names)
    nat_cnt_a: _Counter[str] = _Counter()
    nat_cnt_b: _Counter[str] = _Counter()
    for r in rows_a:
        for a in r.get("atoms_native", []):
            nat_cnt_a[a] += 1
    for r in rows_b:
        for a in r.get("atoms_native", []):
            nat_cnt_b[a] += 1
    nat_vocab = sorted(set(nat_cnt_a) | set(nat_cnt_b))
    nat_a = _np.array([nat_cnt_a.get(a, 0) + eps for a in nat_vocab], dtype=float)
    nat_a /= nat_a.sum()
    nat_b = _np.array([nat_cnt_b.get(a, 0) + eps for a in nat_vocab], dtype=float)
    nat_b /= nat_b.sum()
    native_jsd = _jsd_val(nat_a, nat_b)

    typer.echo(f"\n  Canonical JSD : {canon_jsd:.4f}  (atom-level composition)")
    typer.echo(f"  Native JSD    : {native_jsd:.4f}  (scaffold-specific tool usage)")

    # 2. Pass rates.
    def _pass_rate(rows: list[dict[str, Any]]) -> str:
        labeled = [r for r in rows if r.get("resolved") is not None]
        if not labeled:
            return "n/a"
        return f"{sum(bool(r['resolved']) for r in labeled) / len(labeled):.1%} ({len(labeled)} labeled)"

    typer.echo(f"\n  Pass rate  {label_a}: {_pass_rate(rows_a)}")
    typer.echo(f"  Pass rate  {label_b}: {_pass_rate(rows_b)}")

    # 3. Trajectory length.
    def _median_len(rows: list[dict[str, Any]]) -> float:
        lens = [len(r.get("atoms_canonical", [])) for r in rows]
        return float(_np.median(lens)) if lens else 0

    typer.echo(f"\n  Median steps  {label_a}: {_median_len(rows_a):.0f}")
    typer.echo(f"  Median steps  {label_b}: {_median_len(rows_b):.0f}")

    # 4. Discriminative bigrams.
    def _bigram_dist(
        rows: list[dict[str, Any]],
    ) -> tuple[_np.ndarray[Any, _np.dtype[Any]], list[str]]:
        cnt: _Counter[str] = _Counter()
        for r in rows:
            seq = r.get("atoms_canonical", [])
            for i in range(len(seq) - 1):
                cnt[f"{seq[i]}|{seq[i + 1]}"] += 1
        vocab = sorted(cnt)
        v = _np.array([cnt.get(b, 0) + eps for b in vocab], dtype=float)
        return v / v.sum(), vocab

    # Build shared bigram vocab
    bg_cnt_a: _Counter[str] = _Counter()
    bg_cnt_b: _Counter[str] = _Counter()
    for r in rows_a:
        seq = r.get("atoms_canonical", [])
        for i in range(len(seq) - 1):
            bg_cnt_a[f"{seq[i]}|{seq[i + 1]}"] += 1
    for r in rows_b:
        seq = r.get("atoms_canonical", [])
        for i in range(len(seq) - 1):
            bg_cnt_b[f"{seq[i]}|{seq[i + 1]}"] += 1
    bg_vocab = sorted(set(bg_cnt_a) | set(bg_cnt_b))
    bpa = _np.array([bg_cnt_a.get(b, 0) + eps for b in bg_vocab], dtype=float)
    bpa /= bpa.sum()
    bpb = _np.array([bg_cnt_b.get(b, 0) + eps for b in bg_vocab], dtype=float)
    bpb /= bpb.sum()
    bigram_jsd = _jsd_val(bpa, bpb)

    diff = bpa - bpb
    ranked_desc = sorted(zip(diff, bg_vocab, strict=False), reverse=True)
    ranked_asc = sorted(zip(diff, bg_vocab, strict=False))
    # Top A = highest positive delta (most over-represented in A)
    top_a = [(d, bg) for d, bg in ranked_desc if d > 0][:top_n]
    # Top B = most negative delta (most over-represented in B)
    top_b = [(abs(d), bg) for d, bg in ranked_asc if d < 0][:top_n]

    typer.echo(f"\n  Bigram JSD    : {bigram_jsd:.4f}  (transition structure)")
    typer.echo(f"\n  {label_a} signature (over-represented transitions):")
    for d, bg in top_a:
        src, tgt = bg.split("|")
        typer.echo(f"    {src:14s}→ {tgt:14s}  Δp={d:+.4f}")
    typer.echo(f"\n  {label_b} signature (over-represented transitions):")
    for d, bg in top_b:
        src, tgt = bg.split("|")
        typer.echo(f"    {src:14s}→ {tgt:14s}  Δp={-d:+.4f}")

    # 5. Positional divergence.
    seqs_a: list[list[str]] = [list(r.get("atoms_canonical", [])) for r in rows_a]
    seqs_b: list[list[str]] = [list(r.get("atoms_canonical", [])) for r in rows_b]
    peak_k, peak_jsd = 0, 0.0
    pos_jsds = []
    for k in range(40):

        def _pos_dist(seqs: list[list[str]], k: int) -> _np.ndarray[Any, _np.dtype[Any]] | None:
            cnt: _Counter[str] = _Counter()
            n = 0
            for seq in seqs:
                if k < len(seq):
                    cnt[seq[k]] += 1
                    n += 1
            if n < 10:
                return None
            v = _np.array([cnt.get(a, 0) + eps for a in canon_atoms], dtype=float)
            result: _np.ndarray[Any, _np.dtype[Any]] = v / v.sum()
            return result

        pa_k = _pos_dist(seqs_a, k)
        pb_k = _pos_dist(seqs_b, k)
        if pa_k is None or pb_k is None:
            break
        d = _jsd_val(pa_k, pb_k)
        pos_jsds.append(d)
        if d > peak_jsd:
            peak_jsd = d
            peak_k = k

    typer.echo(f"\n  Peak positional divergence: step {peak_k}  (JSD={peak_jsd:.3f})")
    typer.echo(
        f"  {'Step':>6s}  "
        + "  ".join(f"{canon_atoms[i][:6]:>6s}" for i in range(len(canon_atoms)))
    )
    for label, rows in [(label_a, rows_a), (label_b, rows_b)]:
        if rows and peak_k < len(rows[0].get("atoms_canonical", [])):
            cnt: _Counter[str] = _Counter()
            n = 0
            for r in rows:
                seq = r.get("atoms_canonical", [])
                if peak_k < len(seq):
                    cnt[seq[peak_k]] += 1
                    n += 1
            dist_str = "  ".join(f"{cnt.get(a, 0) / max(1, n):>6.2f}" for a in canon_atoms)
            typer.echo(f"  {label[:18]:>18s}  {dist_str}")

    typer.echo(f"\n{'=' * 64}\n")

    # 6. Optional JSON output.
    if output:
        report = {
            "agent_a": label_a,
            "agent_b": label_b,
            "n_a": len(rows_a),
            "n_b": len(rows_b),
            "canonical_jsd": round(canon_jsd, 5),
            "native_jsd": round(native_jsd, 5),
            "bigram_jsd": round(bigram_jsd, 5),
            "peak_step": peak_k,
            "peak_step_jsd": round(peak_jsd, 5),
            "positional_curve": [round(v, 5) for v in pos_jsds],
            "discriminative_bigrams": {
                label_a: [{"bigram": bg, "delta_p": round(d, 5)} for d, bg in top_a],
                label_b: [{"bigram": bg, "delta_p": round(d, 5)} for d, bg in top_b],
            },
        }
        write_json(output, report)
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
