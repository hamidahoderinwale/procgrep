"""Tests for `procgrep.ingest.adapters.opencode`.

OpenCode sessions are a single JSON object (``info`` + ``messages``); each
message's ``parts`` carry a user text prompt or assistant tool calls. Covers the
message-to-action flatten, the tool-to-atom mapping (including bash sub-classing
and shell-read recovery), the empty-prompt guard, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.opencode import (
    load_opencode_session,
    opencode_adapter,
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
    return {"info": {"role": "user"}, "parts": [{"type": "text", "text": text}]}


def _assistant(*parts: dict) -> dict:  # type: ignore[type-arg]
    return {"info": {"role": "assistant"}, "parts": list(parts)}


def _tool(name: str, command: str = "") -> dict:  # type: ignore[type-arg]
    return {"type": "tool", "tool": name, "state": {"input": {"command": command}}}


def test_human_text_part_maps_to_prompt_ai() -> None:
    record = {"events": [_user("do the thing")]}
    assert opencode_adapter(record) == [ATOM_PROMPT_AI]


def test_empty_user_text_is_not_a_prompt() -> None:
    record = {"events": [_user("   ")]}
    assert opencode_adapter(record) == []


def test_tool_parts_map_to_action_atoms() -> None:
    record = {"events": [_assistant(_tool("read"), _tool("grep"), _tool("edit"))]}
    assert opencode_adapter(record) == [ATOM_READ_FILE, ATOM_SEARCH_REPO, ATOM_EDIT]


def test_bash_part_is_subclassified_and_shell_read_recovered() -> None:
    record = {
        "events": [
            _assistant(
                _tool("bash", "pytest -q"),
                _tool("bash", "git status"),
                _tool("bash", "cat README.md"),  # shell read -> read_file
                _tool("bash", "ls -la"),  # directory listing -> search_repo
            )
        ]
    }
    assert opencode_adapter(record) == [
        ATOM_RUN_TEST,
        ATOM_VERSION_CONTROL,
        ATOM_READ_FILE,
        ATOM_SEARCH_REPO,
    ]


def test_prompt_anchored_order_is_preserved() -> None:
    record = {
        "events": [
            _user("fix the bug"),
            _assistant(_tool("grep"), _tool("read")),
            _assistant(_tool("edit"), _tool("bash", "pytest")),
        ]
    }
    assert opencode_adapter(record) == [
        ATOM_PROMPT_AI,
        ATOM_SEARCH_REPO,
        ATOM_READ_FILE,
        ATOM_EDIT,
        ATOM_RUN_TEST,
    ]


def test_unknown_tool_is_other() -> None:
    record = {"events": [_assistant(_tool("question"), _tool("task"))]}
    assert opencode_adapter(record) == ["other", "other"]


def test_empty_and_malformed_records_are_lenient() -> None:
    assert opencode_adapter({}) == []
    assert opencode_adapter({"events": "not a list"}) == []
    assert opencode_adapter({"events": [None, 42, "x"]}) == []


def test_load_session_wraps_messages_and_id() -> None:
    obj = {"info": {"id": "ses_abc"}, "messages": [_user("hi")]}
    rec = load_opencode_session(obj)
    assert rec["trace_id"] == "ses_abc"
    assert rec["agent"] == "opencode"
    assert opencode_adapter(rec) == [ATOM_PROMPT_AI]


def test_adapter_is_registered() -> None:
    assert get_adapter("opencode") is opencode_adapter
