"""Tests for `procgrep.ingest.adapters.claude_code`.

Treats Claude Code transcripts as a standard ingest adapter: a separate tool's
trace source woven in via the shared feature-based mapper. Covers the
line-to-action flatten (including multi-tool assistant lines), the human-prompt
vs tool-result distinction, Bash command classification, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.claude_code import (
    ATOM_PROMPT_AI,
    claude_code_adapter,
    summarize_transcript,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
)


def _assistant(*tools: tuple[str, str]) -> dict:  # type: ignore[type-arg]
    """An assistant line carrying tool_use blocks: (name, command)."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": name, "input": {"command": cmd}}
                for name, cmd in tools
            ]
        },
    }


def _user_prompt(text: str = "do the thing") -> dict:  # type: ignore[type-arg]
    return {"type": "user", "message": {"content": text}}


def _tool_result() -> dict:  # type: ignore[type-arg]
    return {"type": "user", "message": {"content": [{"type": "tool_result", "content": "ok"}]}}


def test_human_prompt_maps_to_prompt_ai() -> None:
    record = {"events": [_user_prompt()]}
    assert claude_code_adapter(record) == [ATOM_PROMPT_AI]


def test_tool_result_user_event_is_not_a_prompt() -> None:
    # A tool result echoed back as a user event is the agent's output, not a turn.
    record = {"events": [_tool_result()]}
    assert claude_code_adapter(record) == []


def test_assistant_tools_map_to_atoms() -> None:
    record = {"events": [_assistant(("Read", ""), ("Edit", ""), ("Grep", ""))]}
    assert claude_code_adapter(record) == [ATOM_READ_FILE, ATOM_EDIT, ATOM_SEARCH_REPO]


def test_multi_tool_line_explodes_into_several_atoms() -> None:
    record = {"events": [_assistant(("Write", ""), ("Write", ""))]}
    assert claude_code_adapter(record) == [ATOM_EDIT, ATOM_EDIT]


def test_bash_test_command_is_run_test_other_bash_is_other() -> None:
    record = {"events": [_assistant(("Bash", "pytest -q"), ("Bash", "git status"))]}
    assert claude_code_adapter(record) == [ATOM_RUN_TEST, "other"]


def test_file_history_snapshot_is_an_edit() -> None:
    record = {"events": [{"type": "file-history-snapshot"}]}
    assert claude_code_adapter(record) == [ATOM_EDIT]


def test_prompt_anchored_session_orders_correctly() -> None:
    record = {
        "events": [
            _user_prompt("fix the bug"),
            _assistant(("Grep", ""), ("Read", "")),
            _assistant(("Edit", ""), ("Bash", "pytest")),
            _tool_result(),
        ]
    }
    assert claude_code_adapter(record) == [
        ATOM_PROMPT_AI,
        ATOM_SEARCH_REPO,
        ATOM_READ_FILE,
        ATOM_EDIT,
        ATOM_RUN_TEST,
    ]


def test_empty_and_malformed_records_are_lenient() -> None:
    assert claude_code_adapter({}) == []
    assert claude_code_adapter({"events": "not a list"}) == []
    assert claude_code_adapter({"events": [None, 42, "x"]}) == []


def test_adapter_is_registered_under_claude_code() -> None:
    assert get_adapter("claude-code") is claude_code_adapter


def test_summarize_counts_turns_words_tools_without_storing_text() -> None:
    record = {
        "events": [
            _user_prompt("fix the bug now"),  # 4 words
            _assistant(("Read", ""), ("Edit", "")),
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "I will look"}]}},
            {"type": "file-history-snapshot"},
            _tool_result(),  # not a human turn
        ]
    }
    s = summarize_transcript(record)
    assert s["human_turns"] == 1
    assert s["prompt_words"] == 4
    assert s["tool_calls"] == 2
    assert s["tools"] == {"Read": 1, "Edit": 1}
    assert s["reasoning_words"] == 3
    assert s["file_snapshots"] == 1
    assert s["prompt_words_per_turn"] == 4.0
    # only counts are retained -- no message text leaks into the summary
    assert "fix the bug now" not in str(s)
