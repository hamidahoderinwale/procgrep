"""Tests for `procgrep.ingest.adapters.gemini_cli`.

Gemini CLI sessions come in two shapes: a native single JSON object (``user`` /
``gemini`` messages with a ``toolCalls`` list) and a Claude-Code-schema JSONL
that routes to the claude-code adapter. Covers the native flatten, the
tool-to-atom mapping (including run_shell_command sub-classing and shell-read
recovery), the injected-onboarding-prompt guard, CC-schema routing, and
registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.gemini_cli import (
    gemini_cli_adapter,
    load_gemini_session,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_VERSION_CONTROL,
)


def _user(text: str) -> dict:  # type: ignore[type-arg]
    return {"type": "user", "content": [{"text": text}]}


def _gemini(*calls: dict) -> dict:  # type: ignore[type-arg]
    return {"type": "gemini", "content": "...", "toolCalls": list(calls)}


def _call(name: str, **args: object) -> dict:  # type: ignore[type-arg]
    return {"name": name, "args": args}


def test_user_message_maps_to_prompt_ai() -> None:
    assert gemini_cli_adapter({"events": [_user("review this PR")]}) == [ATOM_PROMPT_AI]


def test_injected_onboarding_prompt_is_not_a_human_prompt() -> None:
    # The Gemini CLI auto-init preamble arrives as a user message but is harness
    # injected, not the human typing -- it must not become a prompt boundary.
    preamble = "You are an AI agent that brings the power of Gemini directly into the terminal."
    assert gemini_cli_adapter({"events": [_user(preamble)]}) == []


def test_tool_calls_map_to_action_atoms() -> None:
    record = {"events": [_gemini(_call("read_file"), _call("grep_search"), _call("replace"))]}
    assert gemini_cli_adapter(record) == [ATOM_READ_FILE, ATOM_SEARCH_REPO, ATOM_EDIT]


def test_run_shell_command_is_classified_with_read_recovery() -> None:
    record = {
        "events": [
            _gemini(
                _call("run_shell_command", command="pytest -q"),
                _call("run_shell_command", command="git push"),
                _call("run_shell_command", command="cat go.mod"),  # shell read -> read_file
            )
        ]
    }
    assert gemini_cli_adapter(record) == [ATOM_RUN_TEST, ATOM_VERSION_CONTROL, ATOM_READ_FILE]


def test_write_file_and_symbol_edits_are_edits_unknown_is_other() -> None:
    record = {
        "events": [_gemini(_call("write_file"), _call("insert_after_symbol"), _call("ask_user"))]
    }
    assert gemini_cli_adapter(record) == [ATOM_EDIT, ATOM_EDIT, "other"]


def test_prompt_anchored_order_preserved() -> None:
    record = {
        "events": [
            _user("fix it"),
            _gemini(_call("read_file"), _call("replace")),
            _user("now test"),
            _gemini(_call("run_shell_command", command="pytest")),
        ]
    }
    assert gemini_cli_adapter(record) == [
        ATOM_PROMPT_AI,
        ATOM_READ_FILE,
        ATOM_EDIT,
        ATOM_PROMPT_AI,
        ATOM_RUN_TEST,
    ]


def test_claude_code_schema_session_is_routed() -> None:
    # A few Gemini CLI sessions are byte-identical to a Claude Code transcript;
    # they route to the claude-code adapter rather than the native mapper.
    record = {
        "events": [
            {"type": "user", "message": {"content": "do it"}},
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]},
            },
        ]
    }
    assert gemini_cli_adapter(record) == [ATOM_PROMPT_AI, ATOM_READ_FILE]


def test_empty_and_malformed_records_are_lenient() -> None:
    assert gemini_cli_adapter({}) == []
    assert gemini_cli_adapter({"events": "not a list"}) == []


def test_load_session_native_and_jsonl() -> None:
    native = {"sessionId": "abc", "messages": [_user("hi")]}
    rec = load_gemini_session(native)
    assert rec["trace_id"] == "abc"
    assert rec["agent"] == "gemini-cli"
    assert gemini_cli_adapter(rec) == [ATOM_PROMPT_AI]


def test_adapter_is_registered() -> None:
    assert get_adapter("gemini-cli") is gemini_cli_adapter
