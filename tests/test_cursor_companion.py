"""Tests for `procgrep.ingest.adapters.cursor_companion`.

Treats cursor-companion as a standard ingest adapter: a separate project's
trace source woven in only as an adapter. Covers the event-type to atom
mapping, the unknown-type fallback, empty/malformed records, and the
`_parse_details` helper on dict, JSON-string, and garbage input.
"""

from __future__ import annotations

import json
from typing import Any

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.cursor_companion import (
    ATOM_PROMPT_AI,
    _parse_details,
    cursor_companion_adapter,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
)


def _record(*types: str) -> dict[str, Any]:
    """A companion session record with one event per given type."""
    return {
        "trace_id": "session-1",
        "agent": "dev-1",
        "events": [{"type": t, "timestamp": i} for i, t in enumerate(types)],
    }


# --- event to atom mapping --------------------------------------------------


def test_edit_event_types_map_to_edit() -> None:
    record = _record("code_change", "file_change", "entry_created", "edit", "file_save")
    assert cursor_companion_adapter(record) == [ATOM_EDIT] * 5


def test_prompt_event_types_map_to_prompt_ai() -> None:
    record = _record("prompt", "ai_prompt", "llm_prompt")
    assert cursor_companion_adapter(record) == [ATOM_PROMPT_AI] * 3


def test_terminal_event_types_map_to_run_test() -> None:
    record = _record("terminal", "terminal_command", "command_run")
    assert cursor_companion_adapter(record) == [ATOM_RUN_TEST] * 3


def test_file_read_event_types_map_to_read_file() -> None:
    record = _record("file_open", "file_read")
    assert cursor_companion_adapter(record) == [ATOM_READ_FILE] * 2


def test_search_event_types_map_to_search_repo() -> None:
    record = _record("file_search", "search", "grep")
    assert cursor_companion_adapter(record) == [ATOM_SEARCH_REPO] * 3


def test_event_type_is_case_insensitive() -> None:
    record = _record("Code_Change", "PROMPT", "Terminal")
    assert cursor_companion_adapter(record) == [ATOM_EDIT, ATOM_PROMPT_AI, ATOM_RUN_TEST]


def test_order_is_preserved() -> None:
    record = _record("prompt", "code_change", "terminal", "file_open")
    assert cursor_companion_adapter(record) == [
        ATOM_PROMPT_AI,
        ATOM_EDIT,
        ATOM_RUN_TEST,
        ATOM_READ_FILE,
    ]


# --- unknown / fallback -----------------------------------------------------


def test_unknown_event_type_maps_to_other() -> None:
    record = _record("warp_drive", "code_change")
    assert cursor_companion_adapter(record) == [ATOM_OTHER, ATOM_EDIT]


def test_missing_type_falls_back_to_details_type() -> None:
    record = {
        "events": [
            {"timestamp": 0, "details": {"type": "code_change"}},
            {"timestamp": 1, "details": json.dumps({"type": "prompt"})},
        ]
    }
    assert cursor_companion_adapter(record) == [ATOM_EDIT, ATOM_PROMPT_AI]


def test_blank_type_with_no_details_maps_to_other() -> None:
    record = {"events": [{"type": "", "timestamp": 0}]}
    assert cursor_companion_adapter(record) == [ATOM_OTHER]


# --- empty / malformed records ----------------------------------------------


def test_empty_record_returns_empty() -> None:
    assert cursor_companion_adapter({}) == []


def test_missing_events_returns_empty() -> None:
    assert cursor_companion_adapter({"trace_id": "x", "agent": "y"}) == []


def test_null_events_returns_empty() -> None:
    assert cursor_companion_adapter({"events": None}) == []


def test_non_list_events_returns_empty() -> None:
    assert cursor_companion_adapter({"events": "not a list"}) == []


def test_non_mapping_events_are_skipped() -> None:
    record = {"events": ["bogus", 42, None, {"type": "code_change"}]}
    assert cursor_companion_adapter(record) == [ATOM_EDIT]


# --- _parse_details ---------------------------------------------------------


def test_parse_details_passes_through_dict() -> None:
    assert _parse_details({"type": "edit", "n": 1}) == {"type": "edit", "n": 1}


def test_parse_details_decodes_json_string() -> None:
    assert _parse_details(json.dumps({"type": "prompt"})) == {"type": "prompt"}


def test_parse_details_json_string_non_dict_yields_empty() -> None:
    # Valid JSON that is not an object decodes to {} rather than leaking a list.
    assert _parse_details("[1, 2, 3]") == {}


def test_parse_details_garbage_string_yields_empty() -> None:
    assert _parse_details("{not valid json") == {}


def test_parse_details_non_string_non_dict_yields_empty() -> None:
    assert _parse_details(42) == {}
    assert _parse_details(None) == {}


# --- registration -----------------------------------------------------------


def test_adapter_is_registered_under_cursor_companion() -> None:
    # Importing the module self-registers the adapter; the registry returns it.
    assert get_adapter("cursor-companion") is cursor_companion_adapter


# --- feature-based decomposition (composite events) -------------------------


def test_prompt_with_edit_decomposes_into_prompt_then_edit() -> None:
    record = {"events": [{"type": "prompt_with_edit", "lines_added": 12}]}
    assert cursor_companion_adapter(record) == [ATOM_PROMPT_AI, ATOM_EDIT]


def test_prompt_with_edit_decomposes_without_line_counts() -> None:
    # the type alone implies an edit, even when the exporter omits line counts
    record = {"events": [{"type": "prompt_with_edit"}]}
    assert cursor_companion_adapter(record) == [ATOM_PROMPT_AI, ATOM_EDIT]


def test_context_files_imply_read_before_prompt() -> None:
    record = {"events": [{"type": "prompt", "context_files": ["a.py", "b.py"]}]}
    assert cursor_companion_adapter(record) == [ATOM_READ_FILE, ATOM_PROMPT_AI]


def test_line_counts_imply_edit_on_any_type() -> None:
    record = {"events": [{"type": "prompt", "lines_removed": 3}]}
    assert cursor_companion_adapter(record) == [ATOM_PROMPT_AI, ATOM_EDIT]


def test_edit_type_with_line_counts_yields_single_edit() -> None:
    record = {"events": [{"type": "file_save", "lines_added": 4}]}
    assert cursor_companion_adapter(record) == [ATOM_EDIT]
