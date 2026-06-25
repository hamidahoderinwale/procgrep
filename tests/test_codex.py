"""Tests for `procgrep.ingest.adapters.codex`.

Codex sessions are JSONL of tagged records; tool calls are ``response_item``
``function_call`` / ``custom_tool_call`` payloads (not claude-code tool_use
blocks), and shell commands hide inside JSON-encoded ``arguments`` -- the schema
gap that made Codex "0 computable" before. Covers the user_message prompt source
(vs injected response_item turns), shell classification including shell-read
recovery, MCP-name mapping, apply_patch edits, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.codex import (
    codex_adapter,
    load_codex_session,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
    ATOM_VERSION_CONTROL,
)


def _user_message(text: str) -> dict:  # type: ignore[type-arg]
    return {"type": "event_msg", "payload": {"type": "user_message", "message": text}}


def _shell(command: str) -> dict:  # type: ignore[type-arg]
    import json

    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "shell_command",
            "arguments": json.dumps({"command": command}),
        },
    }


def _function_call(name: str, **args: object) -> dict:  # type: ignore[type-arg]
    import json

    return {
        "type": "response_item",
        "payload": {"type": "function_call", "name": name, "arguments": json.dumps(args)},
    }


def _apply_patch() -> dict:  # type: ignore[type-arg]
    return {
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch"},
    }


def _injected_user() -> dict:  # type: ignore[type-arg]
    # A response_item role=user message carries the AGENTS.md / skill preamble;
    # it must NOT be counted as a human prompt -- only event_msg user_message is.
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "# AGENTS.md instructions"}],
        },
    }


def test_user_message_maps_to_prompt_ai() -> None:
    assert codex_adapter({"events": [_user_message("commit this")]}) == [ATOM_PROMPT_AI]


def test_injected_response_item_user_is_not_a_prompt() -> None:
    # The injected AGENTS.md/developer turn is not a human prompt and yields nothing.
    assert codex_adapter({"events": [_injected_user()]}) == []


def test_shell_commands_are_classified_with_read_recovery() -> None:
    record = {
        "events": [
            _shell("pytest -q"),
            _shell("git commit -m x"),
            _shell("sed -n '1,40p' main.go"),  # codex reads files via sed -> read_file
        ]
    }
    assert codex_adapter(record) == [ATOM_RUN_TEST, ATOM_VERSION_CONTROL, ATOM_READ_FILE]


def test_apply_patch_is_an_edit() -> None:
    assert codex_adapter({"events": [_apply_patch()]}) == [ATOM_EDIT]


def test_update_plan_is_think_and_unknown_call_is_other() -> None:
    record = {"events": [_function_call("update_plan"), _function_call("view_image")]}
    assert codex_adapter(record) == [ATOM_THINK, "other"]


def test_mcp_tools_map_by_action() -> None:
    record = {
        "events": [
            _function_call("mcp__filesystem__read_text_file", path="a"),
            _function_call("mcp__filesystem__search_files", pattern="x"),
            _function_call("mcp__filesystem__write_file", path="a"),
            _function_call("mcp__codex_apps__github_create_pull_request"),  # unmapped -> other
        ]
    }
    assert codex_adapter(record) == [ATOM_READ_FILE, ATOM_SEARCH_REPO, ATOM_EDIT, "other"]


def test_empty_and_malformed_records_are_lenient() -> None:
    assert codex_adapter({}) == []
    assert codex_adapter({"events": "not a list"}) == []
    assert codex_adapter({"events": [None, 42, "x"]}) == []


def test_load_session_reads_session_meta_id() -> None:
    lines = [
        {"type": "session_meta", "payload": {"id": "019d3d40"}},
        _user_message("hi"),
    ]
    rec = load_codex_session(lines)
    assert rec["trace_id"] == "019d3d40"
    assert rec["agent"] == "codex"
    assert codex_adapter(rec) == [ATOM_PROMPT_AI]


def test_adapter_is_registered() -> None:
    assert get_adapter("codex") is codex_adapter
