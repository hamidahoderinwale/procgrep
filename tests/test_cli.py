"""Integration tests for the `procgrep` CLI (typer app)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from procgrep.cli import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "examples" / "data" / "synthetic_traces.jsonl"


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("canonicalize", "fit-bpe", "encode", "jsd", "grep", "list-adapters"):
        assert cmd in result.stdout


def test_list_adapters_includes_known_scaffolds() -> None:
    result = runner.invoke(app, ["list-adapters"])
    assert result.exit_code == 0
    for name in ("swe-agent", "openhands", "mini-swe-agent"):
        assert name in result.stdout


def test_canonicalize_fit_encode_jsd_pipeline(tmp_path: Path) -> None:
    canon = tmp_path / "canon.jsonl"
    vocab = tmp_path / "vocab.json"
    fps = tmp_path / "fps.jsonl"
    jsd_out = tmp_path / "jsd.json"

    r1 = runner.invoke(app, ["canonicalize", "-i", str(SYNTH), "-o", str(canon), "-a", "swe-agent"])
    assert r1.exit_code == 0, r1.stdout
    assert canon.exists()
    assert canon.read_text().strip()

    r2 = runner.invoke(app, ["fit-bpe", "-i", str(canon), "-o", str(vocab), "-V", "50"])
    assert r2.exit_code == 0, r2.stdout
    assert json.loads(vocab.read_text())  # valid vocab JSON

    r3 = runner.invoke(app, ["encode", "-i", str(canon), "-v", str(vocab), "-o", str(fps)])
    assert r3.exit_code == 0, r3.stdout
    assert fps.exists()
    assert fps.read_text().strip()

    r4 = runner.invoke(app, ["jsd", "-i", str(fps), "-o", str(jsd_out), "--group-by", "group"])
    assert r4.exit_code == 0, r4.stdout
    payload = json.loads(jsd_out.read_text())
    assert "groups" in payload
    assert "records" in payload


def test_compare_errors_on_empty_input(tmp_path: Path) -> None:
    # An empty JSONL previously crashed with an IndexError on rows[0];
    # it should now exit cleanly with a non-zero code and a message.
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    nonempty = tmp_path / "rows.jsonl"
    nonempty.write_text(json.dumps({"atoms_canonical": ["edit", "run_test"]}) + "\n")

    result = runner.invoke(app, ["compare", str(empty), str(nonempty)])
    assert result.exit_code == 1
    assert "no trajectories" in result.stdout


def test_grep_over_local_canonical_jsonl(tmp_path: Path) -> None:
    canon = tmp_path / "canon.jsonl"
    runner.invoke(app, ["canonicalize", "-i", str(SYNTH), "-o", str(canon), "-a", "swe-agent"])
    result = runner.invoke(app, ["grep", "edit", str(canon)])
    assert result.exit_code == 0, result.stdout
    assert "matched" in result.stdout


def test_canonicalize_unknown_adapter_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "canonicalize",
            "-i",
            str(SYNTH),
            "-o",
            str(tmp_path / "x.jsonl"),
            "-a",
            "no-such-adapter",
        ],
    )
    assert result.exit_code != 0


def test_vocab_tree_from_existing_vocab(tmp_path: Path) -> None:
    canon = tmp_path / "canon.jsonl"
    vocab = tmp_path / "vocab.json"
    runner.invoke(app, ["canonicalize", "-i", str(SYNTH), "-o", str(canon), "-a", "swe-agent"])
    runner.invoke(app, ["fit-bpe", "-i", str(canon), "-o", str(vocab), "-V", "30"])
    result = runner.invoke(app, ["vocab-tree", "-v", str(vocab)])
    assert result.exit_code == 0, result.stdout
    assert "atoms:" in result.stdout
    assert "maximal procedures" in result.stdout


def test_vocab_tree_requires_input_or_vocab() -> None:
    result = runner.invoke(app, ["vocab-tree"])
    assert result.exit_code != 0
