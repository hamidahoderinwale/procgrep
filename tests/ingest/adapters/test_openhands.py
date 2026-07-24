"""Tests for `procgrep.ingest.adapters.openhands`."""

from __future__ import annotations

import json

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.openhands import (
    _arguments,
    _classify_tool_call,
    _tool_name,
    openhands_adapter,
)
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
)


def _call(name: str, **args: object) -> dict:
    return {"function": {"name": name, "arguments": json.dumps(args)}}


def _assistant(content: str, *calls: dict) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


# --- argument + name extraction ---------------------------------------------


def test_arguments_parses_json_string() -> None:
    assert _arguments(_call("execute_bash", command="pytest")) == {"command": "pytest"}


def test_arguments_accepts_dict_arguments() -> None:
    call = {"function": {"name": "execute_bash", "arguments": {"command": "ls"}}}
    assert _arguments(call) == {"command": "ls"}


def test_arguments_degrades_on_malformed_json() -> None:
    call = {"function": {"name": "execute_bash", "arguments": "{not valid json"}}
    assert _arguments(call) == {}


def test_tool_name_extracted_and_missing_safe() -> None:
    assert _tool_name(_call("Execute_Bash")) == "Execute_Bash"  # lowercasing happens in _classify
    assert _tool_name({}) == ""


# --- tool-call classification -----------------------------------------------


def test_classify_bash_delegates_to_command_classifier() -> None:
    assert _classify_tool_call(_call("execute_bash", command="pytest tests/")) == ATOM_RUN_TEST
    assert (
        _classify_tool_call(_call("execute_bash", command="grep -r foo src/")) == ATOM_SEARCH_REPO
    )
    assert _classify_tool_call(_call("execute_bash", command="cat a.py")) == ATOM_READ_FILE


def test_classify_editor_commands() -> None:
    assert (
        _classify_tool_call(_call("str_replace_editor", command="view", path="a.py"))
        == ATOM_READ_FILE
    )
    assert (
        _classify_tool_call(_call("str_replace_editor", command="create", path="a.py"))
        == ATOM_CREATE_FILE
    )
    assert _classify_tool_call(_call("str_replace_editor", command="str_replace")) == ATOM_EDIT
    assert _classify_tool_call(_call("str_replace_editor", command="insert")) == ATOM_EDIT


def test_classify_finish_and_unknown() -> None:
    assert _classify_tool_call(_call("finish")) == ATOM_SUBMIT
    assert _classify_tool_call(_call("some_unknown_tool")) == ATOM_OTHER


# --- full adapter -----------------------------------------------------------


def test_adapter_emits_think_before_tool_calls() -> None:
    record = {
        "messages": [_assistant("Let me look around.", _call("execute_bash", command="cat x.py"))]
    }
    assert openhands_adapter(record) == [ATOM_THINK, ATOM_READ_FILE]


def test_adapter_no_think_when_content_empty() -> None:
    record = {"messages": [_assistant("", _call("execute_bash", command="cat x.py"))]}
    assert openhands_adapter(record) == [ATOM_READ_FILE]


def test_adapter_full_trajectory() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "fix the bug"},
            _assistant("explore", _call("execute_bash", command="grep -r bug src/")),
            _assistant("", _call("str_replace_editor", command="str_replace", path="a.py")),
            _assistant("verify", _call("execute_bash", command="pytest")),
            _assistant("done", _call("finish")),
        ]
    }
    assert openhands_adapter(record) == [
        ATOM_THINK,
        ATOM_SEARCH_REPO,
        ATOM_EDIT,
        ATOM_THINK,
        ATOM_RUN_TEST,
        ATOM_THINK,
        ATOM_SUBMIT,
    ]


def test_adapter_degrades_on_missing_or_bad_fields() -> None:
    assert openhands_adapter({}) == []
    assert openhands_adapter({"messages": None}) == []
    assert openhands_adapter({"messages": [{"role": "assistant"}]}) == []


def test_registered_under_openhands() -> None:
    record = {"messages": [_assistant("", _call("finish"))]}
    assert get_adapter("openhands")(record) == [ATOM_SUBMIT]
