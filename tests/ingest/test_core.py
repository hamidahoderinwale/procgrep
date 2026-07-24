"""Tests for `procgrep.ingest.core` format detection (sniffers).

These exercise the download-free sniffing path; the datasets-server-backed
introspect/plan/ingest paths need network and are covered by integration use.
"""

from __future__ import annotations

from procgrep.ingest.core import DatasetSchema, _first_present, sniff


def _schema(columns: tuple[str, ...], rows: tuple[dict, ...]) -> DatasetSchema:
    return DatasetSchema(
        dataset="d", config="default", split="test", columns=columns, sample_rows=rows
    )


def _top(schema: DatasetSchema) -> tuple[str, float]:
    ranked = sniff(schema)
    return ranked[0].adapter, ranked[0].confidence


# --- per-format detection ---------------------------------------------------


def test_sniff_openhands_from_tool_calls() -> None:
    schema = _schema(
        ("messages",),
        (
            {
                "messages": [
                    {"role": "assistant", "tool_calls": [{"function": {"name": "execute_bash"}}]}
                ]
            },
        ),
    )
    adapter, conf = _top(schema)
    assert adapter == "openhands"
    assert conf > 0.5


def test_sniff_mini_swe_from_extra_actions() -> None:
    schema = _schema(
        ("messages",),
        (
            {
                "messages": [
                    {"role": "assistant", "content": "", "extra": {"actions": [{"command": "ls"}]}}
                ]
            },
        ),
    )
    adapter, conf = _top(schema)
    assert adapter == "mini-swe-agent"
    assert conf > 0.5


def test_sniff_swe_agent_from_action_observation_turns() -> None:
    schema = _schema(
        ("trajectory",),
        ({"trajectory": [{"action": "ls", "observation": "files"}]},),
    )
    adapter, conf = _top(schema)
    assert adapter == "swe-agent"
    assert conf > 0.5


def test_sniff_react_text_from_fenced_content() -> None:
    schema = _schema(
        ("messages",),
        ({"messages": [{"role": "assistant", "content": "do it\n```bash\nls\n```"}]},),
    )
    adapter, conf = _top(schema)
    assert adapter == "react-text"
    assert conf > 0.5


def test_sniff_swe_smith_from_ids_and_plain_messages() -> None:
    schema = _schema(
        ("messages", "instance_id", "model"),
        (
            {
                "messages": [{"role": "assistant", "content": "plain reasoning, no fence"}],
                "instance_id": "x",
                "model": "m",
            },
        ),
    )
    adapter, conf = _top(schema)
    assert adapter == "swe-smith"
    assert conf > 0.5


# --- ranking invariants -----------------------------------------------------


def test_sniff_returns_all_sniffers_ranked_in_unit_range() -> None:
    schema = _schema(
        ("messages",),
        ({"messages": [{"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]}]},),
    )
    ranked = sniff(schema)
    assert len(ranked) == 5
    confs = [r.confidence for r in ranked]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_sniff_unknown_schema_scores_zero() -> None:
    schema = _schema(("text", "label"), ({"text": "hello", "label": 1},))
    ranked = sniff(schema)
    assert all(r.confidence == 0.0 for r in ranked)


# --- helper -----------------------------------------------------------------


def test_first_present_picks_first_match() -> None:
    assert _first_present(("a", "trace_id", "b"), ("id", "trace_id")) == "trace_id"
    assert _first_present(("a", "b"), ("id", "uid")) is None
