"""Tests for the dynamic ingestion core and the Hub discovery helper.

`procgrep.ingest.core` (introspect / sniff / plan / ingest) and
`procgrep.ingest.discover` (ranking + dedup) are exercised here with the
network mocked: introspection's HTTP call (`_get_json`) and the streaming
`load_dataset` are patched, and discovery's `huggingface_hub.list_datasets`
is mocked, following the pattern in `tests/test_hf.py`. No network.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from procgrep.ingest import core
from procgrep.ingest import discover as discover_mod
from procgrep.ingest.core import (
    DatasetSchema,
    IngestionPlan,
    ingest,
    introspect,
    plan,
    sniff,
)
from procgrep.types import ATOM_EDIT, ATOM_RUN_TEST, ATOM_THINK

# --- schema builders --------------------------------------------------------


def _swe_agent_row() -> dict[str, Any]:
    # SWE-agent .traj turns: {action, observation, ...} with no `role`.
    return {
        "instance_id": "django__django-1",
        "model": "gpt-4",
        "repo": "django/django",
        "trajectory": [
            {"action": "search_dir foo", "observation": "..."},
            {"action": "edit 1:2", "observation": "..."},
        ],
    }


def _openhands_row() -> dict[str, Any]:
    # Assistant turn carrying structured tool_calls -> openhands.
    return {
        "instance_id": "x-1",
        "model": "claude",
        "messages": [
            {"role": "system", "content": "..."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "str_replace_editor"}}],
            },
        ],
    }


def _schema(rows: list[dict[str, Any]]) -> DatasetSchema:
    columns = tuple(rows[0].keys()) if rows else ()
    return DatasetSchema("acme/traces", "default", "train", columns, tuple(rows))


# --- introspect (server path mocked) ----------------------------------------


def test_introspect_via_server_parses_splits_and_rows() -> None:
    splits_payload = {"splits": [{"config": "default", "split": "train"}]}
    rows_payload = {
        "features": [{"name": "trajectory"}, {"name": "instance_id"}, {"name": "model"}],
        "rows": [{"row": _swe_agent_row()}],
    }

    def fake_get_json(url: str, *, timeout: float) -> dict[str, Any]:
        return splits_payload if "/splits" in url else rows_payload

    with patch.object(core, "_get_json", side_effect=fake_get_json):
        schema = introspect("acme/traces")

    assert schema.config == "default"
    assert schema.split == "train"
    assert schema.columns == ("trajectory", "instance_id", "model")
    assert len(schema.sample_rows) == 1


def test_introspect_falls_back_to_datasets_on_server_failure() -> None:
    # Server path raises (no splits); the datasets streaming fallback runs.
    def boom(url: str, *, timeout: float) -> dict[str, Any]:
        raise RuntimeError("datasets-server 500")

    fake_rows = [_swe_agent_row()]

    class _FakeStream:
        def take(self, n: int) -> list[dict[str, Any]]:
            return fake_rows[:n]

    def fake_load_dataset(*args: Any, **kwargs: Any) -> _FakeStream:
        return _FakeStream()

    with (
        patch.object(core, "_get_json", side_effect=boom),
        patch("procgrep.ingest.hf._import_load_dataset", return_value=fake_load_dataset),
    ):
        schema = introspect("acme/traces")

    assert schema.dataset == "acme/traces"
    assert "trajectory" in schema.columns


# --- sniff ------------------------------------------------------------------


def test_sniff_ranks_openhands_top_on_tool_calls() -> None:
    ranked = sniff(_schema([_openhands_row()]))
    assert ranked[0].adapter == "openhands"
    assert ranked[0].confidence >= 0.9
    # Confidences are clamped into [0, 1] and sorted descending.
    confs = [r.confidence for r in ranked]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_sniff_detects_swe_agent_shape() -> None:
    ranked = sniff(_schema([_swe_agent_row()]))
    top = ranked[0]
    assert top.adapter == "swe-agent"
    assert top.confidence >= 0.9


def test_sniff_unknown_schema_is_all_zero() -> None:
    ranked = sniff(_schema([{"text": "hello", "label": 1}]))
    assert all(r.confidence == 0.0 for r in ranked)


# --- plan -------------------------------------------------------------------


def test_plan_infers_adapter_and_fields() -> None:
    schema = _schema([_swe_agent_row()])
    with patch.object(core, "introspect", return_value=schema):
        p = plan("acme/traces", probe=False)
    assert isinstance(p, IngestionPlan)
    assert p.adapter == "swe-agent"
    assert p.trace_id_field == "instance_id"
    assert p.agent_field == "model"
    assert p.group_field == "repo"
    assert p.candidate is True  # has a conversation/turn column


def test_plan_explicit_adapter_overrides_inference() -> None:
    schema = _schema([_openhands_row()])
    with patch.object(core, "introspect", return_value=schema):
        p = plan("acme/traces", adapter="swe-agent", probe=False)
    assert p.adapter == "swe-agent"


def test_plan_raises_when_no_adapter_inferable() -> None:
    schema = _schema([{"text": "hi", "label": 0}])
    with (
        patch.object(core, "introspect", return_value=schema),
        pytest.raises(ValueError, match="could not infer an adapter"),
    ):
        plan("acme/traces", probe=False)


def test_plan_summary_renders_fields_and_candidates() -> None:
    schema = _schema([_swe_agent_row()])
    with patch.object(core, "introspect", return_value=schema):
        p = plan("acme/traces", probe=False)
    text = p.summary()
    assert "acme/traces" in text
    assert "swe-agent" in text
    assert "considered" in text


# --- ingest -----------------------------------------------------------------


def _swe_smith_row(traj_id: str, action: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "..."},
        {"role": "assistant", "thought": "Thinking", "action": action},
    ]
    return {
        "messages": json.dumps(messages),
        "instance_id": traj_id,
        "model": "claude-3-7-sonnet",
        "resolved": True,
    }


def test_ingest_streams_and_canonicalizes() -> None:
    rows = [_swe_smith_row("t1", "edit"), _swe_smith_row("t2", "pytest")]
    schema = DatasetSchema("acme/smith", "default", "train", tuple(rows[0].keys()), tuple(rows))

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return rows

    with (
        patch.object(core, "introspect", return_value=schema),
        patch("procgrep.ingest.hf._import_load_dataset", return_value=fake_load_dataset),
    ):
        traces, p = ingest("acme/smith", limit=10)

    assert p.adapter == "swe-smith"
    assert len(traces) == 2
    assert traces[0].atoms == [ATOM_THINK, ATOM_EDIT]
    assert traces[1].atoms == [ATOM_THINK, ATOM_RUN_TEST]


def test_ingest_honors_limit() -> None:
    rows = [_swe_smith_row(f"t{i}", "edit") for i in range(5)]
    schema = DatasetSchema("acme/smith", "default", "train", tuple(rows[0].keys()), tuple(rows))

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return rows

    with (
        patch.object(core, "introspect", return_value=schema),
        patch("procgrep.ingest.hf._import_load_dataset", return_value=fake_load_dataset),
    ):
        traces, _ = ingest("acme/smith", limit=2)

    assert len(traces) == 2


# --- discover (ranking + dedup) ---------------------------------------------


class _FakeDatasetInfo:
    def __init__(self, ident: str, downloads: int, likes: int = 0) -> None:
        self.id = ident
        self.downloads = downloads
        self.likes = likes
        self.last_modified = "2025-01-01"
        self.tags = ("agent", "trajectories")


def test_discover_dedups_and_ranks_by_downloads() -> None:
    # `low` appears in two query results; it must dedup to one entry. The final
    # list is ranked by downloads descending.
    by_query = {
        "trajectories": [_FakeDatasetInfo("org/high", 1000), _FakeDatasetInfo("org/low", 10)],
        "traces": [_FakeDatasetInfo("org/low", 10), _FakeDatasetInfo("org/mid", 200)],
    }

    def fake_list_datasets(
        *, search: str | None = None, author: str | None = None, limit: int = 50
    ) -> list[_FakeDatasetInfo]:
        if author is not None:
            return []
        return by_query.get(search or "", [])

    with patch("huggingface_hub.list_datasets", fake_list_datasets):
        metas = discover_mod.discover(queries=("trajectories", "traces"), authors=())

    ids = [m.id for m in metas]
    assert ids == ["org/high", "org/mid", "org/low"]  # ranked, deduped
    assert len(ids) == len(set(ids))


def test_discover_filters_below_min_downloads() -> None:
    def fake_list_datasets(
        *, search: str | None = None, author: str | None = None, limit: int = 50
    ) -> list[_FakeDatasetInfo]:
        if search == "trajectories":
            return [_FakeDatasetInfo("org/big", 500), _FakeDatasetInfo("org/tiny", 1)]
        return []

    with patch("huggingface_hub.list_datasets", fake_list_datasets):
        metas = discover_mod.discover(queries=("trajectories",), authors=(), min_downloads=100)

    assert [m.id for m in metas] == ["org/big"]


def test_discover_crawls_authors() -> None:
    def fake_list_datasets(
        *, search: str | None = None, author: str | None = None, limit: int = 50
    ) -> list[_FakeDatasetInfo]:
        if author == "nebius":
            return [_FakeDatasetInfo("nebius/swe-traces", 42)]
        return []

    with patch("huggingface_hub.list_datasets", fake_list_datasets):
        metas = discover_mod.discover(queries=(), authors=("nebius",))

    assert [m.id for m in metas] == ["nebius/swe-traces"]
    assert metas[0].tags == ("agent", "trajectories")
