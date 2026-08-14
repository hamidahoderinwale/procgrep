"""Tests for the `procgrep floor` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from procgrep.cli import app


def _write_corpus(path: Path, n_per_condition: int = 16) -> None:
    """Two-condition canonical corpus with mild within-condition variation."""
    records = []
    for cond, block in (("a", ["edit", "run_test"]), ("b", ["search_repo", "read_file"])):
        for i in range(n_per_condition):
            records.append(
                {
                    "trace_id": f"{cond}{i}",
                    "agent": "model-x",
                    "atoms": block * (3 + i % 4) + ["submit"],
                    "metadata": {"condition": cond},
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_floor_prints_json_with_curves_and_spec(tmp_path: Path) -> None:
    corpus = tmp_path / "canonical.jsonl"
    _write_corpus(corpus)
    result = CliRunner().invoke(
        app,
        ["floor", str(corpus), "--condition", "condition", "--n-draws", "25", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["condition"] == "condition"
    assert payload["vocab_spec"]["content_hash"]
    assert [c["condition"] for c in payload["cells"]] == ["a", "b"]
    for cell in payload["cells"]:
        assert cell["floor"][0]["n"] == 5
        assert cell["floor"][-1]["n"] == cell["full_n"]
        assert [row["delta"] for row in cell["seeds_needed"]] == [0.05, 0.1, 0.2]
        assert cell["vocab_spec"] == (
            f"{payload['vocab_spec']['content_hash']}:{payload['vocab_spec']['vocab_size']}"
        )


def test_floor_writes_out_file_and_is_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "canonical.jsonl"
    _write_corpus(corpus)
    outputs = []
    for name in ("one.json", "two.json"):
        out = tmp_path / name
        result = CliRunner().invoke(
            app,
            [
                "floor",
                str(corpus),
                "--condition",
                "condition",
                "--n-draws",
                "25",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        outputs.append(out.read_text())
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["cells"]


def test_floor_refuses_when_every_cell_is_too_small(tmp_path: Path) -> None:
    corpus = tmp_path / "canonical.jsonl"
    _write_corpus(corpus, n_per_condition=4)
    result = CliRunner().invoke(app, ["floor", str(corpus), "--condition", "condition"])
    assert result.exit_code == 1
    assert "no condition cell" in result.output


def test_floor_refuses_a_missing_condition_field(tmp_path: Path) -> None:
    corpus = tmp_path / "canonical.jsonl"
    _write_corpus(corpus)
    result = CliRunner().invoke(app, ["floor", str(corpus), "--condition", "nope"])
    assert result.exit_code == 1
    assert "missing on every trace" in result.output
