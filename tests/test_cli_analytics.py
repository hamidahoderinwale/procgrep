"""Happy-path tests for the analytic CLI commands.

Covers the under-tested `compare`, `curate`, and `grep` subcommands on tiny
synthetic JSONL fixtures built in a temp dir. The empty-input guard for
`compare` lives in `tests/test_cli.py`; here we assert exit 0 and the
expected output keys / report fields on non-empty inputs. No network: the
local-path branch of `curate` and `grep` is used.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from procgrep.cli import app

runner = CliRunner()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# --- compare ----------------------------------------------------------------


def _compare_rows(edit_heavy: bool, resolved: bool) -> list[dict[str, object]]:
    """A handful of comparison rows for one agent.

    `compare` reads `atoms_canonical`, `atoms_native`, and `resolved`.
    """
    if edit_heavy:
        canonical = ["edit", "edit", "edit", "run_test", "submit"]
        native = ["str_replace", "str_replace", "str_replace", "pytest", "submit"]
    else:
        canonical = ["search_repo", "read_file", "edit", "run_test", "submit"]
        native = ["grep", "open", "str_replace", "pytest", "submit"]
    return [
        {
            "trace_id": f"t{i}",
            "atoms_canonical": canonical,
            "atoms_native": native,
            "resolved": resolved,
        }
        for i in range(12)
    ]


def test_compare_happy_path_writes_report(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(a, _compare_rows(edit_heavy=True, resolved=False))
    _write_jsonl(b, _compare_rows(edit_heavy=False, resolved=True))

    result = runner.invoke(
        app, ["compare", str(a), str(b), "--name-a", "A", "--name-b", "B", "-o", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    assert "Canonical JSD" in result.stdout
    assert "Bigram JSD" in result.stdout

    payload = json.loads(out.read_text())
    for key in (
        "agent_a",
        "agent_b",
        "n_a",
        "n_b",
        "canonical_jsd",
        "native_jsd",
        "bigram_jsd",
        "peak_step",
        "positional_curve",
        "discriminative_bigrams",
    ):
        assert key in payload, key
    assert payload["agent_a"] == "A"
    assert payload["n_a"] == 12
    assert payload["canonical_jsd"] >= 0.0


def test_compare_prints_without_output_file(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, _compare_rows(edit_heavy=True, resolved=False))
    _write_jsonl(b, _compare_rows(edit_heavy=False, resolved=True))

    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 0, result.stdout
    # Falls back to the file stem as the display name.
    assert "a" in result.stdout
    assert "Pass rate" in result.stdout


# --- curate -----------------------------------------------------------------


def _canonical_traces() -> list[dict[str, object]]:
    """Canonical-trace records (the `atoms` schema `records_to_traces` reads)."""
    shapes = [
        ["search_repo", "read_file", "edit", "run_test", "submit"],
        ["search_repo", "read_file", "edit", "run_test", "submit"],  # exact dup
        ["edit", "edit", "edit", "edit", "submit"],
        ["read_file", "edit", "run_test", "edit", "run_test", "submit"],
        ["search_repo", "search_repo", "read_file", "edit", "submit"],
        ["edit", "run_test", "submit"],
    ]
    return [
        {"trace_id": f"c{i}", "agent": "ed", "group": "g", "atoms": shape}
        for i, shape in enumerate(shapes)
    ]


def test_curate_local_jsonl_reports_redundancy(tmp_path: Path) -> None:
    corpus = tmp_path / "canon.jsonl"
    _write_jsonl(corpus, _canonical_traces())

    result = runner.invoke(app, ["curate", str(corpus), "--vocab-size", "32"])
    assert result.exit_code == 0, result.stdout
    assert "corpus" in result.stdout
    assert "exact duplicates" in result.stdout
    assert "diverse subset" in result.stdout


def test_curate_exports_diverse_subset(tmp_path: Path) -> None:
    corpus = tmp_path / "canon.jsonl"
    export = tmp_path / "subset.jsonl"
    _write_jsonl(corpus, _canonical_traces())

    result = runner.invoke(
        app, ["curate", str(corpus), "--vocab-size", "32", "--target", "3", "--export", str(export)]
    )
    assert result.exit_code == 0, result.stdout
    assert export.exists()
    lines = [ln for ln in export.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    # Each exported record is a valid canonical trace.
    rec = json.loads(lines[0])
    assert "trace_id" in rec
    assert "atoms" in rec


# --- grep -------------------------------------------------------------------


def test_grep_local_jsonl_matches_pattern(tmp_path: Path) -> None:
    corpus = tmp_path / "canon.jsonl"
    _write_jsonl(corpus, _canonical_traces())

    # An edit streak of 4+ matches only the all-edit trace.
    result = runner.invoke(app, ["grep", "(edit ){4,}", str(corpus)])
    assert result.exit_code == 0, result.stdout
    assert "matched" in result.stdout
    assert "1/6" in result.stdout


def test_grep_reports_zero_matches(tmp_path: Path) -> None:
    corpus = tmp_path / "canon.jsonl"
    _write_jsonl(corpus, _canonical_traces())

    result = runner.invoke(app, ["grep", "delete_file", str(corpus)])
    assert result.exit_code == 0, result.stdout
    assert "0/6" in result.stdout
